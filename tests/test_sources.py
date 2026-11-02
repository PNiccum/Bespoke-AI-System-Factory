"""Research connectors: each public API's reply becomes Doc records a model may read."""

import httpx

from factory.net.http import CallBudget
from factory.pipeline.sources import EuropePMC, NCBIGene, OpenAlex, ReferenceDocs, best_window, page_text
from tests.test_http import mock

OPENALEX = {"results": [{"id": "https://openalex.org/W1", "display_name": "Primer3 revisited", "doi": "https://doi.org/10.1/x",
                         "publication_year": 2012, "authorships": [{"author": {"display_name": "A. Untergasser"}}],
                         "abstract_inverted_index": {"Primer3": [0], "designs": [1], "primers": [2]}}]}
EUROPEPMC = {"resultList": {"result": [{"id": "123", "pmcid": "PMC9", "source": "MED", "title": "Primer Tm",
                                        "pubYear": "2020", "authorString": "B. Smith", "journalTitle": "NAR",
                                        "abstractText": "<h4>Aim</h4>Tm should be <i>55-65</i> C."}]}}
ESEARCH = {"esearchresult": {"idlist": ["672"]}}
ESUMMARY = {"result": {"672": {"name": "BRCA1", "description": "BRCA1 DNA repair associated",
                               "organism": {"scientificname": "Homo sapiens"}, "summary": "A tumor suppressor.",
                               "chromosome": "17", "maplocation": "17q21.31", "otheraliases": "RNF53"}}}


def test_openalex_rebuilds_the_abstract(tmp_path):
    http, calls = mock(lambda r: httpx.Response(200, json=OPENALEX))
    (d,) = OpenAlex(tmp_path, CallBudget("t", 5), http=http).search("primer design", 5)
    assert (d.id, d.text, d.year, d.license) == ("openalex:W1", "Primer3 designs primers", 2012, "public")
    assert "Untergasser" in d.citation and "has_abstract" in calls[0]


def test_europepmc_strips_markup(tmp_path):
    http, _ = mock(lambda r: httpx.Response(200, json=EUROPEPMC))
    (d,) = EuropePMC(tmp_path, CallBudget("t", 5), http=http).search("primer tm", 5)
    assert d.id == "europepmc:PMC9" and d.text == "AimTm should be 55-65 C."


def test_ncbi_gene_searches_then_summarizes(tmp_path):
    http, calls = mock(lambda r: httpx.Response(200, json=ESEARCH if "esearch" in str(r.url) else ESUMMARY))
    (d,) = NCBIGene(tmp_path, CallBudget("t", 5), http=http).search("BRCA1[sym] AND human[orgn]", 3)
    assert d.id == "ncbi_gene:672" and "17q21.31" in d.text and "Homo sapiens" in d.title
    assert len(calls) == 2


def test_open_access_papers_add_the_full_text_passage_that_matches_the_query(tmp_path):
    oa = {"resultList": {"result": [{**EUROPEPMC["resultList"]["result"][0], "isOpenAccess": "Y"}]}}
    body = "<article><sec><p>" + "Background text. " * 400 + "</p><p>We used SantaLucia nearest-neighbor parameters " \
           "with 50 mM Na+ to compute Tm.</p><p>" + "Discussion text. " * 400 + "</p></sec></article>"

    def reply(r):
        return httpx.Response(200, text=body) if "fullTextXML" in str(r.url) else httpx.Response(200, json=oa)
    http, calls = mock(reply)
    (d,) = EuropePMC(tmp_path, CallBudget("t", 5), http=http).search("SantaLucia nearest-neighbor Tm", 5)
    assert d.text.startswith("AimTm should be 55-65 C.") and "From the full text:" in d.text
    assert "SantaLucia nearest-neighbor parameters with 50 mM Na+" in d.text and len(d.text) <= 4000
    assert any("PMC9/fullTextXML" in c for c in calls)


def test_reference_docs_read_the_matching_part_of_the_matching_manual(tmp_path):
    page = "<html><script>var x;</script><h1>E-utilities</h1><p>" + "Intro. " * 800 + \
           "</p><p>EFetch returns records; use rettype=fasta and retmode=text for nucleotide sequences.</p></html>"
    http, calls = mock(lambda r: httpx.Response(200, text=page))
    docs = ReferenceDocs(tmp_path, CallBudget("t", 5), http=http).search("efetch rettype retmode fasta", 1)
    assert [d.id for d in docs] == ["docs:ncbi_eutils_indepth"] and docs[0].connector == "docs"
    assert "rettype=fasta and retmode=text" in docs[0].text and "var x" not in docs[0].text
    assert calls == ["https://www.ncbi.nlm.nih.gov/books/NBK25499/"]


def test_a_query_that_matches_no_manual_reads_nothing(tmp_path):
    http, calls = mock(lambda r: httpx.Response(200, text="x"))
    assert ReferenceDocs(tmp_path, CallBudget("t", 5), http=http).search("quantum chromodynamics", 3) == []
    assert calls == []


def test_page_text_and_best_window():
    assert page_text("<p>A &amp; B</p><div>C</div>") == "A & B\nC"
    text = "Filler sentence here. " * 100 + "The esearch utility returns UIDs. " + "More filler text. " * 100
    assert "esearch utility returns UIDs" in best_window(text, "esearch UIDs", 300)
