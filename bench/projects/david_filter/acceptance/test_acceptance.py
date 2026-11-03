import json
import math
import subprocess
import sys

import pytest
from conftest import ROOT
from davidfilter import filter_terms, gene_counts, read_chart, recheck

CHART = ROOT / "acceptance" / "chart.tsv"


def names(terms):
    return [t["term"].split("~")[-1].split(":")[-1] for t in terms]


def test_read_chart_types_and_genes():
    terms = read_chart(str(CHART))
    assert len(terms) == 11  # the trailing blank line is not a term
    t = terms[0]
    assert t["category"] == "GOTERM_BP_DIRECT" and t["term"] == "GO:0051301~cell division"
    assert (t["count"], t["list_total"], t["pop_hits"], t["pop_total"]) == (18, 90, 380, 20581)
    assert t["pvalue"] == pytest.approx(5.111e-13) and t["benjamini"] == pytest.approx(2.1e-06)
    assert t["genes"][:3] == ["CDK1", "CCNB1", "CCNA2"] and len(t["genes"]) == 18


def test_default_filter_thresholds_and_order():
    kept = filter_terms(read_chart(str(CHART)))
    assert names(kept) == ["Cell Cycle, Mitotic", "Cell cycle", "cell division", "DNA replication",
                           "DNA replication", "nucleoplasm", "G1/S transition of mitotic cell cycle",
                           "chromosome segregation"]


def test_thresholds_are_inclusive_and_categories_filter():
    terms = read_chart(str(CHART))
    assert "chromosome segregation" in names(filter_terms(terms, min_count=6, max_benjamini=0.019))
    assert "chromosome segregation" not in names(filter_terms(terms, min_count=7))
    kegg = filter_terms(terms, min_fold=1.0, max_benjamini=1.0, categories=["kegg_pathway"])
    assert names(kegg) == ["Cell cycle", "DNA replication", "Pathways in cancer"]


def test_recheck_recomputes_fold_and_pvalues():
    terms = {t["term"]: t for t in read_chart(str(CHART))}
    r = recheck(terms["hsa04110:Cell cycle"])
    assert r["fold"] == pytest.approx((14 / 90) / (157 / 20581))
    N, K, n, k = 20581, 157, 90, 14
    p = sum(math.comb(K, i) * math.comb(N - K, n - i) for i in range(k, n + 1)) / math.comb(N, n)
    assert r["p_hyper"] == pytest.approx(p, rel=1e-6)
    assert r["p_ease"] > r["p_hyper"] and r["fold_ok"] and r["p_ok"]


def test_recheck_flags_a_misreported_fold():
    terms = {t["term"]: t for t in read_chart(str(CHART))}
    r = recheck(terms["hsa03030:DNA replication"])
    assert not r["fold_ok"] and r["p_ok"] and r["fold"] == pytest.approx(38.11, rel=1e-3)


def test_gene_counts_rank_genes_by_how_many_kept_terms_hold_them():
    kept = filter_terms(read_chart(str(CHART)))
    counts = gene_counts(kept)
    assert counts[:7] == [("AURKA", 8), ("CCNA2", 8), ("CCNB1", 8), ("CDC20", 8), ("CDK1", 8), ("PLK1", 8),
                          ("AURKB", 6)]
    assert counts == sorted(counts, key=lambda x: (-x[1], x[0]))
    assert "CASP3" not in dict(counts)  # only in the apoptosis term, which the filter drops


def test_cli_json_and_recheck_exit_code():
    base = [sys.executable, str(ROOT / "davidfilter.py"), str(CHART)]
    r = subprocess.run(base + ["--json", "--category", "KEGG_PATHWAY"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and [t["term"] for t in json.loads(r.stdout)] == ["hsa04110:Cell cycle",
                                                                             "hsa03030:DNA replication"]
    r = subprocess.run(base + ["--recheck"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "hsa03030" in r.stderr and "|" in r.stdout
    r = subprocess.run(base + ["--recheck", "--category", "REACTOME_PATHWAY"], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0
