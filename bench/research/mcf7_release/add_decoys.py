"""Append the planted decoys to a copy of a protein list, in that list's own columns and delimiter.

    python bench/research/mcf7_release/add_decoys.py LIST OUT

The lab list's headers are not known in advance, so the accession, gene and fold-change columns are found by name
(case-insensitive). When no header looks like a fold change, the one remaining column whose values are nearly all
numbers is used; --accession-col, --gene-col and --fold-col name a column outright. A decoy's fold change is written
on the list's own scale: log2 when the list has values at or below zero or under 1, a plain ratio (2 ** log2fc)
when every value is 1 or more. Other columns stay empty. It prints the header names it did and did not use and
how many rows it wrote, never the list's contents.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATTERNS = {
    "accession": re.compile(r"accession|uniprot|protein[ _]?id|^entry$|^id$", re.I),
    "gene": re.compile(r"gene|symbol", re.I),
    "fold": re.compile(r"fold|log2|lfc|fc$|ratio", re.I),
}


def find_columns(header: list[str]) -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for key, pat in PATTERNS.items():
        found[key] = next((h for h in header if pat.search(h.strip()) and h not in found.values()), None)
    return found


def numeric_fraction(rows: list[dict], col: str) -> float:
    return len(numbers(rows, col)) / max(1, sum(1 for r in rows if (r.get(col) or "").strip()))


def numbers(rows: list[dict], col: str | None) -> list[float]:
    out = []
    for r in rows:
        try:
            out.append(float((r.get(col) or "").strip()))
        except ValueError:
            pass
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("list")
    ap.add_argument("out")
    ap.add_argument("--decoys", default=str(HERE / "decoys.csv"))
    ap.add_argument("--accession-col", help="header of the accession column, if the name is not recognised")
    ap.add_argument("--gene-col", help="header of the gene column, if the name is not recognised")
    ap.add_argument("--fold-col", help="header of the fold-change column, if the name is not recognised")
    args = ap.parse_args(argv)
    text = Path(args.list).read_text(encoding="utf-8-sig", errors="replace")
    first = text.splitlines()[0] if text else ""
    delim = "\t" if "\t" in first else ";" if first.count(";") > first.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    header, rows = list(reader.fieldnames or []), list(reader)
    cols = find_columns(header)
    for key, given in (("accession", args.accession_col), ("gene", args.gene_col), ("fold", args.fold_col)):
        if given:
            if given not in header:
                print(f"--{key}-col {given!r} is not a header of the list ({len(header)} columns); nothing written")
                return 1
            cols[key] = given
    if cols["fold"] is None:  # no header looks like a fold change: take the one other column that is all numbers
        rest = [h for h in header if h not in cols.values() and rows and numeric_fraction(rows, h) >= 0.9]
        if len(rest) == 1:
            cols["fold"] = rest[0]
    if not (cols["accession"] or cols["gene"]):
        print(f"no accession or gene column found in the header ({len(header)} columns); nothing written")
        return 1
    values = numbers(rows, cols["fold"])
    ratio = bool(values) and min(values) >= 1
    with open(args.decoys, newline="", encoding="utf-8") as f:
        decoys = list(csv.DictReader(f))
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=header, delimiter=delim, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    for d in decoys:
        row = dict.fromkeys(header, "")
        if cols["accession"]:
            row[cols["accession"]] = d["accession"]
        if cols["gene"]:
            row[cols["gene"]] = d["gene"]
        if cols["fold"]:
            fc = float(d["log2fc"])
            row[cols["fold"]] = f"{2 ** fc:.2f}" if ratio else f"{fc:.2f}"
        w.writerow(row)
    Path(args.out).write_text(out.getvalue(), encoding="utf-8")
    unused = [h for h in header if h not in cols.values()]
    print(f"columns used: accession={cols['accession']!r} gene={cols['gene']!r} fold={cols['fold']!r} "
          f"({'ratio' if ratio else 'log2'} scale); not used: {unused}; "
          f"{len(rows)} list rows + {len(decoys)} decoys -> {args.out}")
    if cols["fold"] is None:
        print("WARNING: no fold-change column found; decoy rows have an empty fold change and are easier to spot. "
              "Rerun with --fold-col <header>.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
