"""Drug-target status of every protein in a list (reference solution for the target_flags project)."""
import argparse
import csv
import json
import sys

SECTIONS = [("approved", "Approved drug targets"), ("clinical", "In clinical trials"), ("preclinical", "Preclinical"),
            ("literature", "Suggested in the literature"), ("none", "No known targeting")]


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        text = f.read()
    first = text.splitlines()[0] if text else ""
    return [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
            for r in csv.DictReader(text.splitlines(), delimiter="\t" if "\t" in first else ",")]


def load_list(path):
    return [{"accession": r.get("accession", ""), "gene": r.get("gene", ""),
             "log2fc": float(r["log2fc"]) if r.get("log2fc") else None} for r in _read(path)]


def load_targets(path):
    return [{"gene": r["gene"], "drug": r["drug"], "max_phase": int(float(r["max_phase"])),
             "mechanism": r.get("mechanism", ""), "source": r.get("source", "")} for r in _read(path)]


def load_literature(path):
    return [{"gene": r["gene"], "pmid": r["pmid"], "note": r.get("note", "")} for r in _read(path)]


def classify(rows, targets, literature=()):
    out, seen = [], set()
    for row in rows:
        gene = row["gene"].upper()
        if gene in seen:
            continue
        seen.add(gene)
        hits = [t for t in targets if t["gene"].upper() == gene]
        pmids = sorted({str(x["pmid"]) for x in literature if x["gene"].upper() == gene})
        phase = max((t["max_phase"] for t in hits), default=None)
        if phase is None:
            tier = "literature" if pmids else "none"
        else:
            tier = "approved" if phase >= 4 else "clinical" if phase >= 1 else "preclinical"
        out.append({"gene": gene, "log2fc": row.get("log2fc"), "tier": tier, "max_phase": phase,
                    "drugs": sorted({t["drug"] for t in hits}), "pmids": pmids})
    return out


def candidates(classified, min_abs_log2fc=1.0):
    pick = [c for c in classified if c["tier"] in ("literature", "none") and c["log2fc"] is not None
            and abs(c["log2fc"]) >= min_abs_log2fc]
    return sorted(pick, key=lambda c: (-abs(c["log2fc"]), c["gene"]))


def report(classified, title="Drug-target status"):
    lines = [f"# {title}", ""]
    for tier, heading in SECTIONS:
        lines.append(f"## {heading}")
        for c in (c for c in classified if c["tier"] == tier):
            fc = "n/a" if c["log2fc"] is None else f"{c['log2fc']:.2f}"
            extra = "; ".join(x for x in (", ".join(c["drugs"]), ", ".join(f"PMID:{p}" for p in c["pmids"])) if x)
            lines.append(f"- {c['gene']} (log2FC {fc})" + (f": {extra}" if extra else ""))
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("list")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--literature")
    ap.add_argument("--min-fc", type=float, default=1.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        rows, targets = load_list(a.list), load_targets(a.targets)
        lit = load_literature(a.literature) if a.literature else []
    except OSError as e:
        print(f"cannot read: {e}", file=sys.stderr)
        return 2
    c = classify(rows, targets, lit)
    if a.json:
        print(json.dumps({"proteins": c, "candidates": candidates(c, a.min_fc)}, indent=1))
    else:
        print(report(c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
