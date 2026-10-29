"""Research connectors: each public API's reply becomes Doc records a model may read."""

import httpx

from factory.net.http import CallBudget
from factory.pipeline.sources import EuropePMC, NCBIGene, OpenAlex
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
