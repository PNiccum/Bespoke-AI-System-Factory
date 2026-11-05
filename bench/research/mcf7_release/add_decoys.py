"""Append the planted decoys to a copy of a protein list, in that list's own columns and delimiter.

    python bench/research/mcf7_release/add_decoys.py LIST OUT

The lab list's headers are not known in advance, so the accession, gene and fold-change columns are found by name
(case-insensitive). A decoy's fold change is written on the list's own scale: log2 when the list has values at or
below zero or under 1, a plain ratio (2 ** log2fc) when every value is 1 or more. Other columns stay empty. It
prints only which columns it used and how many rows it wrote, never the list's contents.
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
    args = ap.parse_args(argv)
    text = Path(args.list).read_text(encoding="utf-8-sig", errors="replace")
    first = text.splitlines()[0] if text else ""
    delim = "\t" if "\t" in first else ";" if first.count(";") > first.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delim)
    header, rows = list(reader.fieldnames or []), list(reader)
    cols = find_columns(header)
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
    print(f"columns used: accession={cols['accession']!r} gene={cols['gene']!r} fold={cols['fold']!r} "
          f"({'ratio' if ratio else 'log2'} scale); {len(rows)} list rows + {len(decoys)} decoys -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
