"""Check an AI-written report's claims against the data and the cited sources (reference solution for claim_audit)."""
import argparse
import csv
import json
import re
import sys

INFO = {"resolved_alias"}
ACC = r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
CITE = re.compile(rf"^(?:(PMID):(\d+)|(DOI):(10\.\d{{4,9}}/\S+)|(UNIPROT):({ACC}))$", re.I)


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        text = f.read()
    first = text.splitlines()[0] if text else ""
    return [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()}
            for r in csv.DictReader(text.splitlines(), delimiter="\t" if "\t" in first else ",")]


def load_dataset(path):
    return {r["gene"].upper(): float(r["log2fc"]) if r.get("log2fc") else None for r in _read(path)}


def load_aliases(path):
    out = {}
    for r in _read(path):
        out.setdefault(r["symbol"].upper(), []).append(r["alias"])
    return out


def load_pathways(path):
    out = {}
    for r in _read(path):
        p = out.setdefault(r["pathway_id"].lower(), {"name": r["pathway_name"], "genes": set()})
        p["genes"].add(r["gene"].upper())
    return out


def norm_citation(c):
    m = CITE.match(c.strip())
    if not m:
        return None
    if m.group(1):
        return f"PMID:{m.group(2)}"
    if m.group(3):
        return f"DOI:{m.group(4).lower()}"
    return f"UniProt:{m.group(6).upper()}"


def mentions(text, names):
    return any(re.search(rf"(?<![A-Za-z0-9]){re.escape(n)}(?![A-Za-z0-9])", text, re.I) for n in names)


def audit(claims, dataset, sources, aliases=None, pathways=None):
    aliases, pathways = aliases or {}, pathways or {}
    reverse = {a.upper(): sym for sym, names in aliases.items() for a in names}
    srcs = {norm_citation(k) or k: v for k, v in sources.items()}
    findings, rows = [], []
    for c in claims:
        cid, raw = c["id"], c["protein"].strip()
        mine = []

        def add(code, detail="", mine=mine, cid=cid):
            mine.append({"claim": cid, "code": code, "detail": detail})

        gene = raw.upper()
        if gene not in dataset and gene in reverse:
            gene = reverse[gene]
            add("resolved_alias", f"{raw} is {gene}")
        names = [gene] + aliases.get(gene, [])
        if gene not in dataset:
            add("not_in_dataset", gene)
        else:
            fc, d = dataset[gene], c.get("direction")
            if fc is not None and ((d == "up" and fc <= 0) or (d == "down" and fc >= 0)):
                add("direction_mismatch", f"claims {d}, log2FC {fc}")
        cites = c.get("citations") or []
        if not cites:
            add("no_citation")
        for cite in cites:
            key = norm_citation(cite)
            if key is None:
                add("bad_citation", cite)
            elif key not in srcs:
                add("citation_not_found", key)
            else:
                s = srcs[key]
                if not mentions(f"{s.get('title', '')} {s.get('abstract', '')}", names):
                    add("citation_off_topic", key)
        if c.get("pathway"):
            want = c["pathway"].strip().lower()
            hit = pathways.get(want) or next((p for p in pathways.values() if p["name"].lower() == want), None)
            if hit is None or gene not in hit["genes"]:
                add("pathway_not_supported", c["pathway"])
        findings += mine
        rows.append({"id": cid, "protein": gene, "passed": not [f for f in mine if f["code"] not in INFO]})
    grounded = sum(r["passed"] for r in rows) / len(rows) if rows else 0.0
    return {"findings": findings, "claims": rows, "grounded": grounded}


def to_markdown(result):
    lines = ["# Claim audit", "", f"Grounded: {sum(r['passed'] for r in result['claims'])} of {len(result['claims'])} "
             f"claims ({result['grounded']:.0%})", "", "| Claim | Protein | Result | Findings |", "|---|---|---|---|"]
    for r in result["claims"]:
        codes = ", ".join(f["code"] for f in result["findings"] if f["claim"] == r["id"])
        lines.append(f"| {r['id']} | {r['protein']} | {'pass' if r['passed'] else 'FAIL'} | {codes} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("claims")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sources", required=True)
    ap.add_argument("--aliases")
    ap.add_argument("--pathways")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        with open(a.claims, encoding="utf-8") as f:
            claims = json.load(f)
        with open(a.sources, encoding="utf-8") as f:
            sources = json.load(f)
        result = audit(claims, load_dataset(a.dataset), sources,
                       load_aliases(a.aliases) if a.aliases else None, load_pathways(a.pathways) if a.pathways else None)
    except (OSError, ValueError) as e:
        print(f"cannot read: {e}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1) if a.json else to_markdown(result))
    return 0 if all(r["passed"] for r in result["claims"]) else 1


if __name__ == "__main__":
    sys.exit(main())
