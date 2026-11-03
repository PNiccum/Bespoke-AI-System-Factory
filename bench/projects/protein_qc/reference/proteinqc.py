"""Plausibility checks for a protein list before any interpretation (reference solution for the protein_qc project)."""
import argparse
import csv
import json
import re
import sys

ACCESSION = re.compile(r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?$")
ORDER = ["bad_accession", "duplicate", "not_in_reference", "wrong_organism", "gene_mismatch", "not_expressed",
         "missing_value"]


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        text = f.read()
    first = text.splitlines()[0] if text else ""
    reader = csv.DictReader(text.splitlines(), delimiter="\t" if "\t" in first else ",")
    return [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()} for r in reader]


def load_list(path):
    out = []
    for r in _rows(path):
        fc = r.get("log2fc", "")
        out.append({"accession": r.get("accession", ""), "gene": r.get("gene", ""),
                    "log2fc": float(fc) if fc else None})
    return out


def load_reference(path):
    ref = {}
    for r in _rows(path):
        ref[r["entry"]] = {"genes": r.get("gene names", "").split(), "organism": r.get("organism", "")}
    return ref


def _cell(name):
    return re.sub(r"[\s-]", "", name).upper()


def load_expression(path):
    return {(r["gene"].upper(), _cell(r["cell line"])): float(r["ntpm"]) for r in _rows(path)}


def check(rows, reference, expression=None, cell_line=None, organism="Homo sapiens", min_ntpm=1.0):
    issues, seen = [], {}

    def add(i, row, code, detail):
        issues.append({"row": i, "accession": row["accession"], "gene": row["gene"], "issue": code, "detail": detail})

    for i, row in enumerate(rows, 1):
        acc = row["accession"]
        if row.get("log2fc") is None:
            add(i, row, "missing_value", "no log2fc")
        if not ACCESSION.match(acc):
            add(i, row, "bad_accession", f"{acc!r} is not a UniProt accession")
            continue
        canon = acc.split("-")[0]
        if canon in seen:
            add(i, row, "duplicate", f"same protein as row {seen[canon]}")
        else:
            seen[canon] = i
        entry = reference.get(canon)
        if entry is None:
            add(i, row, "not_in_reference", f"{canon} not found")
            continue
        if not entry["organism"].startswith(organism):
            add(i, row, "wrong_organism", entry["organism"])
        names = [g.upper() for g in entry["genes"]]
        if row["gene"] and row["gene"].upper() not in names:
            add(i, row, "gene_mismatch", f"{canon} is {entry['genes'][0] if entry['genes'] else '?'}")
        if expression is not None and cell_line and entry["genes"]:
            level = expression.get((entry["genes"][0].upper(), _cell(cell_line)))
            if level is not None and level < min_ntpm:
                add(i, row, "not_expressed", f"{level} nTPM in {cell_line}")
    issues.sort(key=lambda x: (x["row"], ORDER.index(x["issue"])))
    return issues


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("list")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--expression")
    ap.add_argument("--cell-line")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        rows, ref = load_list(a.list), load_reference(a.reference)
        expr = load_expression(a.expression) if a.expression else None
    except OSError as e:
        print(f"cannot read: {e}", file=sys.stderr)
        return 2
    issues = check(rows, ref, expr, a.cell_line)
    if a.json:
        print(json.dumps(issues, indent=1))
    else:
        for x in issues:
            print("\t".join(str(x[k]) for k in ("row", "accession", "gene", "issue", "detail")))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
