"""Filter and re-check a DAVID Functional Annotation Chart (reference solution for the david_filter project)."""
import argparse
import csv
import json
import math
import sys
from collections import Counter

INTS = {"Count": "count", "List Total": "list_total", "Pop Hits": "pop_hits", "Pop Total": "pop_total"}
FLOATS = {"%": "percent", "PValue": "pvalue", "Fold Enrichment": "fold", "Bonferroni": "bonferroni",
          "Benjamini": "benjamini", "FDR": "fdr"}


def read_chart(path):
    terms = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if not (r.get("Term") or "").strip():
                continue
            t = {"category": r["Category"].strip(), "term": r["Term"].strip(),
                 "genes": [g.strip().upper() for g in (r.get("Genes") or "").split(",") if g.strip()]}
            t.update({k: int(float(r[c])) for c, k in INTS.items()})
            t.update({k: float(r[c]) for c, k in FLOATS.items()})
            terms.append(t)
    return terms


def filter_terms(terms, min_count=5, min_fold=1.5, max_benjamini=0.05, categories=None):
    cats = {c.upper() for c in categories} if categories else None
    kept = [t for t in terms if t["count"] >= min_count and t["fold"] >= min_fold and t["benjamini"] <= max_benjamini
            and (cats is None or t["category"].upper() in cats)]
    return sorted(kept, key=lambda t: (t["benjamini"], -t["fold"], t["term"]))


def upper_tail(N, K, n, k):
    k = max(k, 0)
    return sum(math.comb(K, i) * math.comb(N - K, n - i) for i in range(k, min(K, n) + 1)) / math.comb(N, n)


def recheck(term, rel_tol=0.05):
    c, n, K, N = term["count"], term["list_total"], term["pop_hits"], term["pop_total"]
    fold = (c / n) / (K / N)
    p_hyper = upper_tail(N, K, n, c)
    p_ease = upper_tail(N - 1, K - 1, n - 1, c - 1)
    return {"fold": fold, "p_hyper": p_hyper, "p_ease": p_ease,
            "fold_ok": math.isclose(fold, term["fold"], rel_tol=rel_tol),
            "p_ok": math.isclose(p_ease, term["pvalue"], rel_tol=rel_tol)}


def gene_counts(terms):
    counts = Counter(g for t in terms for g in set(t["genes"]))
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))


def to_markdown(terms):
    lines = ["| Category | Term | Count | Fold | Benjamini |", "|---|---|---|---|---|"]
    lines += [f"| {t['category']} | {t['term']} | {t['count']} | {t['fold']:.2f} | {t['benjamini']:.2g} |" for t in terms]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("chart")
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--min-fold", type=float, default=1.5)
    ap.add_argument("--max-benjamini", type=float, default=0.05)
    ap.add_argument("--category", action="append")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--recheck", action="store_true")
    a = ap.parse_args(argv)
    try:
        terms = read_chart(a.chart)
    except OSError as e:
        print(f"cannot read: {e}", file=sys.stderr)
        return 2
    kept = filter_terms(terms, a.min_count, a.min_fold, a.max_benjamini, a.category)
    print(json.dumps(kept, indent=1) if a.json else to_markdown(kept), end="" if not a.json else "\n")
    if a.recheck:
        checks = [(t["term"], recheck(t)) for t in kept]
        bad = [term for term, r in checks if not (r["fold_ok"] and r["p_ok"])]
        for term in bad:
            print(f"recheck failed: {term}", file=sys.stderr)
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
