"""Gather per-protein evidence from UniProt and Europe PMC, with sources (reference solution for evidence_table)."""
import argparse
import csv
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

UNIPROT = "https://rest.uniprot.org/uniprotkb/{}.json"
EUROPEPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def uniprot_url(accession):
    return UNIPROT.format(accession)


def europepmc_url(gene, context="MCF7", page_size=25):
    query = f'"{gene}" AND "{context}"' if context else f'"{gene}"'
    return f"{EUROPEPMC}?{urllib.parse.urlencode({'query': query, 'format': 'json', 'pageSize': page_size})}"


def parse_uniprot(record):
    genes = record.get("genes") or []
    gene = ((genes[0].get("geneName") or {}).get("value", "")) if genes else ""
    function = next((t["value"] for c in record.get("comments", []) if c.get("commentType") == "FUNCTION"
                     for t in c.get("texts", [])[:1]), "")
    go_bp, reactome = set(), set()
    for x in record.get("uniProtKBCrossReferences", []):
        props = {p["key"]: p["value"] for p in x.get("properties", [])}
        if x.get("database") == "GO" and props.get("GoTerm", "").startswith("P:"):
            go_bp.add(props["GoTerm"][2:])
        elif x.get("database") == "Reactome":
            reactome.add((x["id"], props.get("PathwayName", "")))
    return {"accession": record.get("primaryAccession", ""), "gene": gene, "function": function,
            "go_bp": sorted(go_bp), "reactome": [list(r) for r in sorted(reactome)]}


def parse_europepmc(resp):
    papers = [{"pmid": r["pmid"], "doi": r.get("doi"), "title": r.get("title", ""),
               "year": int(r["pubYear"]) if r.get("pubYear") else None}
              for r in (resp.get("resultList") or {}).get("result", []) if r.get("pmid")]
    return {"hits": int(resp.get("hitCount", 0)), "papers": papers}


def cache_name(url):
    return hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest() + ".json"  # a file name, not security


class Fetcher:
    """GET JSON. With from_dir, read saved responses only (named by cache_name). With cache_dir, save and reuse."""

    def __init__(self, from_dir=None, cache_dir=None):
        self.from_dir, self.cache_dir = from_dir and Path(from_dir), cache_dir and Path(cache_dir)

    def __call__(self, url):
        if self.from_dir:
            return json.loads((self.from_dir / cache_name(url)).read_text(encoding="utf-8"))
        if self.cache_dir and (self.cache_dir / cache_name(url)).exists():
            return json.loads((self.cache_dir / cache_name(url)).read_text(encoding="utf-8"))
        if not url.startswith("https://"):
            raise ValueError(f"only https URLs are fetched: {url}")
        request = urllib.request.Request(url, headers={"Accept": "application/json"})  # noqa: S310 (https checked)
        with urllib.request.urlopen(request, timeout=30) as r:  # noqa: S310
            data = json.loads(r.read().decode("utf-8"))
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / cache_name(url)).write_text(json.dumps(data), encoding="utf-8")
        return data


def load_list(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    return [{"accession": r["accession"], "gene": r.get("gene", ""),
             "log2fc": float(r["log2fc"]) if r.get("log2fc") else None} for r in rows]


def build_table(rows, fetch, context="MCF7"):
    table = []
    for row in rows:
        entry = {"accession": row["accession"], "gene": row["gene"], "log2fc": row.get("log2fc"), "function": "",
                 "go_bp": [], "reactome": [], "papers_hits": 0, "top_pmids": [], "sources": [], "error": None}
        try:
            u = uniprot_url(row["accession"])
            up = parse_uniprot(fetch(u))
            gene = up["gene"] or row["gene"]
            e = europepmc_url(gene, context)
            pm = parse_europepmc(fetch(e))
        except Exception as exc:  # one failed protein must not stop the table
            entry["error"] = f"{type(exc).__name__}: {exc}"
            table.append(entry)
            continue
        entry.update(gene=gene, function=up["function"], go_bp=up["go_bp"], reactome=up["reactome"],
                     papers_hits=pm["hits"], top_pmids=[p["pmid"] for p in pm["papers"][:3]], sources=[u, e])
        table.append(entry)
    return table


def pathway_counts(table, min_proteins=2):
    groups = {}
    for t in table:
        for _id, name in t["reactome"]:
            groups.setdefault(name, set()).add(t["gene"])
    out = [(name, len(g), sorted(g)) for name, g in groups.items() if len(g) >= min_proteins]
    return sorted(out, key=lambda x: (-x[1], x[0]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("list")
    ap.add_argument("--context", default="MCF7")
    ap.add_argument("--from-dir")
    ap.add_argument("--cache")
    ap.add_argument("-o", "--out")
    a = ap.parse_args(argv)
    try:
        rows = load_list(a.list)
    except OSError as e:
        print(f"cannot read: {e}", file=sys.stderr)
        return 2
    table = build_table(rows, Fetcher(a.from_dir, a.cache), a.context)
    text = json.dumps(table, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 1 if any(t["error"] for t in table) else 0


if __name__ == "__main__":
    sys.exit(main())
