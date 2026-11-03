import json
import subprocess
import sys
import urllib.parse

from conftest import ROOT
from evidence import (
    Fetcher,
    build_table,
    europepmc_url,
    load_list,
    parse_europepmc,
    parse_uniprot,
    pathway_counts,
    uniprot_url,
)

A = ROOT / "acceptance"
SAVED = Fetcher(from_dir=str(A / "responses"))


def table():
    return build_table(load_list(str(A / "list.csv")), SAVED)


def test_urls():
    assert uniprot_url("P06493") == "https://rest.uniprot.org/uniprotkb/P06493.json"
    u = urllib.parse.urlparse(europepmc_url("CDK1", "MCF7", 25))
    assert u.netloc == "www.ebi.ac.uk" and u.path == "/europepmc/webservices/rest/search"
    q = urllib.parse.parse_qs(u.query)
    assert q["query"] == ['"CDK1" AND "MCF7"'] and q["format"] == ["json"] and q["pageSize"] == ["25"]


def test_parse_uniprot_keeps_biological_process_terms_and_unique_reactome_pathways():
    rec = SAVED(uniprot_url("P06493"))
    p = parse_uniprot(rec)
    assert p["accession"] == "P06493" and p["gene"] == "CDK1"
    assert p["function"].startswith("Plays a key role")
    assert p["go_bp"] == ["G2/M transition of mitotic cell cycle", "cell division"]
    assert [list(x) for x in p["reactome"]] == [["R-HSA-69275", "G2/M Transition"], ["R-HSA-69278", "Cell Cycle, Mitotic"]]


def test_parse_europepmc_skips_records_without_a_pmid():
    p = parse_europepmc(SAVED(europepmc_url("CDK1")))
    assert p["hits"] == 412
    assert [x["pmid"] for x in p["papers"]] == ["90000201", "90000202", "90000203", "90000204"]
    assert p["papers"][0]["doi"] == "10.1000/f201" and p["papers"][0]["year"] == 2021 and p["papers"][1]["doi"] is None


def test_table_rows_carry_their_sources_and_a_failure_does_not_stop_the_rest():
    t = table()
    assert [r["gene"] for r in t] == ["CDK1", "PCNA", "CCNB1", "FAKE1"]
    cdk1 = t[0]
    assert cdk1["log2fc"] == 2.1 and cdk1["papers_hits"] == 412 and cdk1["top_pmids"] == ["90000201", "90000202", "90000203"]
    assert cdk1["sources"] == [uniprot_url("P06493"), europepmc_url("CDK1")] and cdk1["error"] is None
    assert t[2]["papers_hits"] == 0 and t[2]["top_pmids"] == []
    assert t[3]["error"] and t[3]["go_bp"] == [] and t[3]["sources"] == []


def test_pathway_counts_keep_pathways_with_enough_proteins():
    assert pathway_counts(table()) == [("Cell Cycle, Mitotic", 3, ["CCNB1", "CDK1", "PCNA"]),
                                       ("G2/M Transition", 2, ["CCNB1", "CDK1"])]
    assert pathway_counts(table(), min_proteins=3) == [("Cell Cycle, Mitotic", 3, ["CCNB1", "CDK1", "PCNA"])]


def test_cli_from_saved_responses(tmp_path):
    out = tmp_path / "table.json"
    r = subprocess.run([sys.executable, str(ROOT / "evidence.py"), str(A / "list.csv"), "--from-dir",
                        str(A / "responses"), "-o", str(out)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1  # FAKE1 has no saved record
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 4 and data[1]["go_bp"] == ["DNA replication"]
    ok = tmp_path / "ok.csv"
    ok.write_text("accession,gene,log2fc\nP12004,PCNA,1.7\n", encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "evidence.py"), str(ok), "--from-dir", str(A / "responses")],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and json.loads(r.stdout)[0]["top_pmids"] == ["90000211"]
