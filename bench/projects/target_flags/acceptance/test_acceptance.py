import json
import re
import subprocess
import sys

from conftest import ROOT
from targetflags import candidates, classify, load_list, load_literature, load_targets, report

A = ROOT / "acceptance"


def classified():
    return classify(load_list(str(A / "list.csv")), load_targets(str(A / "targets.tsv")),
                    load_literature(str(A / "literature.tsv")))


def by_gene():
    return {c["gene"]: c for c in classified()}


def test_one_entry_per_gene_in_list_order():
    genes = [c["gene"] for c in classified()]
    assert genes == ["CDK4", "AURKA", "PLK1", "TOP2A", "ESR1", "TYMS", "RRM2", "KIF11", "MKI67", "PCNA", "NUSAP1",
                     "CENPF", "HMGB2", "TPX2"]  # the lower-case hmgb2 repeat is dropped


def test_tiers_follow_the_highest_phase():
    g = by_gene()
    tiers = {k: v["tier"] for k, v in g.items()}
    assert [k for k, t in tiers.items() if t == "approved"] == ["CDK4", "TOP2A", "ESR1", "TYMS", "RRM2"]
    assert [k for k, t in tiers.items() if t == "clinical"] == ["AURKA", "PLK1", "KIF11", "PCNA"]
    assert tiers["HMGB2"] == "preclinical" and g["HMGB2"]["max_phase"] == 0
    assert tiers["NUSAP1"] == "literature" and tiers["CENPF"] == "literature"
    assert tiers["MKI67"] == "none" and g["MKI67"]["max_phase"] is None and tiers["TPX2"] == "none"


def test_drugs_and_pmids_are_sorted_unique_and_matched_ignoring_case():
    g = by_gene()
    assert g["ESR1"]["drugs"] == ["FULVESTRANT", "TAMOXIFEN"]  # one targets row spells the gene esr1
    assert g["PLK1"]["drugs"] == ["ONVANSERTIB", "VOLASERTIB"] and g["PLK1"]["max_phase"] == 3
    assert g["PLK1"]["pmids"] == ["90000004"] and g["PLK1"]["tier"] == "clinical"
    assert g["PCNA"]["drugs"] == ["AOH1996"]
    assert g["NUSAP1"]["pmids"] == ["90000001", "90000003"]


def test_candidates_are_untargeted_changed_proteins_by_size_of_change():
    assert [c["gene"] for c in candidates(classified())] == ["MKI67", "NUSAP1", "TPX2", "CENPF"]
    assert [c["gene"] for c in candidates(classified(), min_abs_log2fc=2.0)] == ["MKI67", "NUSAP1"]


def test_report_sections_in_order_with_each_gene_once():
    md = report(classified())
    assert md.splitlines()[0].startswith("# ")
    heads = [line for line in md.splitlines() if line.startswith("## ")]
    assert heads == ["## Approved drug targets", "## In clinical trials", "## Preclinical",
                     "## Suggested in the literature", "## No known targeting"]
    items = [line for line in md.splitlines() if line.startswith("- ")]
    assert len(items) == 14
    assert re.search(r"^- NUSAP1 .*PMID:90000001.*PMID:90000003", md, re.M)
    assert re.search(r"^- CDK4 .*PALBOCICLIB", md, re.M)
    assert md.index("- MKI67") > md.index("## No known targeting")


def test_cli_json():
    r = subprocess.run([sys.executable, str(ROOT / "targetflags.py"), str(A / "list.csv"), "--targets",
                        str(A / "targets.tsv"), "--literature", str(A / "literature.tsv"), "--json", "--min-fc", "2"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert len(data["proteins"]) == 14 and [c["gene"] for c in data["candidates"]] == ["MKI67", "NUSAP1"]
    r = subprocess.run([sys.executable, str(ROOT / "targetflags.py"), str(A / "missing.csv"), "--targets",
                        str(A / "targets.tsv")], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and r.stderr
