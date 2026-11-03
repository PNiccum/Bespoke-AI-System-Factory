import json
import subprocess
import sys

from conftest import ROOT
from proteinqc import check, load_expression, load_list, load_reference

A = ROOT / "acceptance"


def issues(**kw):
    rows = load_list(str(A / "list.csv"))
    return check(rows, load_reference(str(A / "reference.tsv")), **kw)


def codes(found):
    return [(x["row"], x["issue"]) for x in found]


def test_load_list_reads_headers_case_insensitively_and_empty_values_as_none():
    rows = load_list(str(A / "list.csv"))
    assert len(rows) == 12
    assert rows[0] == {"accession": "P06493", "gene": "CDK1", "log2fc": 2.1}
    assert rows[10]["log2fc"] is None and rows[11]["log2fc"] == -0.4


def test_list_level_problems_in_row_order():
    assert codes(issues()) == [
        (5, "gene_mismatch"),     # P04626 is ERBB2 (HER2), not ERBB3 (HER3)
        (7, "duplicate"),         # an isoform of row 1
        (8, "bad_accession"),
        (9, "wrong_organism"),    # mouse Cdk1
        (10, "not_in_reference"),
        (11, "missing_value"),
    ]


def test_details_point_at_the_cause():
    found = {x["row"]: x for x in issues()}
    assert "row 1" in found[7]["detail"]
    assert "ERBB2" in found[5]["detail"]
    assert "Mus musculus" in found[9]["detail"]


def test_gene_match_uses_synonyms_and_ignores_case():
    rows = [{"accession": "P04626", "gene": "her2", "log2fc": 1.0}, {"accession": "P03372", "gene": "esr1", "log2fc": 1.0}]
    assert check(rows, load_reference(str(A / "reference.tsv"))) == []


def test_not_expressed_in_the_cell_line_ignores_case_and_hyphens():
    expr = load_expression(str(A / "expression.tsv"))
    found = issues(expression=expr, cell_line="mcf-7")
    assert (6, "not_expressed") in codes(found)
    assert [c for c in codes(found) if c[1] == "not_expressed"] == [(6, "not_expressed")]
    assert all(c[1] != "not_expressed" for c in codes(issues(expression=expr, cell_line="MDA-MB-231")))


def test_cli_json_and_exit_codes(tmp_path):
    cmd = [sys.executable, str(ROOT / "proteinqc.py"), str(A / "list.csv"), "--reference", str(A / "reference.tsv"),
           "--expression", str(A / "expression.tsv"), "--cell-line", "MCF7", "--json"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert r.returncode == 1
    assert len(json.loads(r.stdout)) == 7
    clean = tmp_path / "ok.csv"
    clean.write_text("accession,gene,log2fc\nP12004,PCNA,1.7\n", encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "proteinqc.py"), str(clean), "--reference", str(A / "reference.tsv")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() == ""
    r = subprocess.run([sys.executable, str(ROOT / "proteinqc.py"), str(tmp_path / "nope.csv"), "--reference",
                        str(A / "reference.tsv")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and r.stderr
