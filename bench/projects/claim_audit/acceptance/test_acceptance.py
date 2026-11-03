import json
import subprocess
import sys

import pytest
from claimaudit import audit, load_aliases, load_dataset, load_pathways
from conftest import ROOT

A = ROOT / "acceptance"


def run(claims=None):
    claims = claims if claims is not None else json.loads((A / "claims.json").read_text(encoding="utf-8"))
    return audit(claims, load_dataset(str(A / "dataset.csv")), json.loads((A / "sources.json").read_text("utf-8")),
                 load_aliases(str(A / "aliases.tsv")), load_pathways(str(A / "pathways.tsv")))


def codes(result, cid):
    return [f["code"] for f in result["findings"] if f["claim"] == cid]


def test_grounded_claims_pass_and_aliases_resolve():
    r = run()
    assert [c["id"] for c in r["claims"] if c["passed"]] == ["c1", "c2", "c8", "c10"]
    assert codes(r, "c1") == [] and codes(r, "c2") == ["resolved_alias"]
    assert next(c for c in r["claims"] if c["id"] == "c2")["protein"] == "CDK1"
    assert r["grounded"] == pytest.approx(0.4)


def test_data_checks_direction_and_membership():
    r = run()
    assert codes(r, "c3") == ["direction_mismatch"]          # ESR1 is down in the data
    # HER3 is ERBB3, which is not in the list, and the cited paper is about CDK1
    assert codes(r, "c4") == ["resolved_alias", "not_in_dataset", "citation_off_topic"]


def test_citation_checks():
    r = run()
    assert codes(r, "c5") == ["no_citation"]
    assert codes(r, "c6") == ["bad_citation", "citation_not_found"]
    assert codes(r, "c7") == ["citation_off_topic"]          # the source names CDK10, not CDK1


def test_citation_ids_are_normalized_and_alias_mentions_count():
    r = run()
    assert codes(r, "c8") == []   # doi:10.1000/XYZ1 is the DOI:10.1000/xyz1 source; pathway named, any case
    assert codes(r, "c10") == []  # the source says "Vimentin", an alias of VIM


def test_pathway_claims_need_the_protein_in_that_pathway():
    r = run()
    assert codes(r, "c9") == ["pathway_not_supported"]
    ok = run([{"id": "p", "protein": "PCNA", "direction": "up", "pathway": "HSA03030", "citations": ["PMID:90000103"]}])
    assert ok["claims"][0]["passed"]
    unknown = run([{"id": "q", "protein": "PCNA", "pathway": "Hippo signaling", "citations": ["PMID:90000103"]}])
    assert codes(unknown, "q") == ["pathway_not_supported"]


def test_empty_claims_and_down_direction():
    assert run([])["grounded"] == 0.0 and run([])["claims"] == []
    r = run([{"id": "d", "protein": "esr1", "direction": "down", "citations": ["PMID:90000103"]},
             {"id": "e", "protein": "PCNA", "direction": "down", "citations": ["PMID:90000103"]}])
    assert codes(r, "d") == [] and codes(r, "e") == ["direction_mismatch"]


def test_cli_exit_code_is_a_gate(tmp_path):
    base = [sys.executable, str(ROOT / "claimaudit.py")]
    opts = ["--dataset", str(A / "dataset.csv"), "--sources", str(A / "sources.json"), "--aliases",
            str(A / "aliases.tsv"), "--pathways", str(A / "pathways.tsv")]
    r = subprocess.run(base + [str(A / "claims.json")] + opts + ["--json"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and json.loads(r.stdout)["grounded"] == pytest.approx(0.4)
    good = tmp_path / "good.json"
    good.write_text(json.dumps([{"id": "c1", "protein": "CDK1", "direction": "up", "citations": ["PMID:90000101"]}]))
    r = subprocess.run(base + [str(good)] + opts, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "c1" in r.stdout
