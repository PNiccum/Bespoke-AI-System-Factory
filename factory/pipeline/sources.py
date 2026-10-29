"""Research sources: plain-code connectors to public APIs. Searching needs no model, so researchers are code.

Each connector turns a query into ``Doc`` records: a title, a link, the text a model may read (an abstract or a gene
summary), a citation and a license. All three free sources here are open to any model:

- OpenAlex (scholarly works across fields; abstracts arrive as an inverted index and are rebuilt here)
- Europe PMC (PubMed and PubMed Central, with abstracts and open-access flags)
- NCBI Gene (gene summaries, through the E-utilities)

Elsevier full text is not here: it may only go to ARC models, so its connector will tag every document ``arc_only``
when it is added after the freeze (see blackboard.py). Keys, where an API takes one, come from the environment and
are sent with the request only, never cached or logged.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from factory.net.http import CachedAPI, CallBudget
from factory.pipeline.blackboard import License

MAX_TEXT = 4000  # characters of each document a model reads


class Doc(BaseModel):
    id: str  # connector-prefixed, e.g. "openalex:W123", "europepmc:PMC456", "ncbi_gene:672"
    connector: str
    title: str
    url: str
    year: int | None = None
    text: str  # what a model may read: abstract or summary, trimmed to MAX_TEXT
    citation: str
    license: License = "public"


def rebuild_abstract(inverted: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; put the words back in order."""
    if not inverted:
        return ""
    words = sorted((pos, word) for word, positions in inverted.items() for pos in positions)
    return " ".join(w for _, w in words)


class OpenAlex(CachedAPI):
    base = "https://api.openalex.org"

    def search(self, query: str, n: int = 5) -> list[Doc]:
        data = self.get("works", {"search": query, "per-page": n, "filter": "has_abstract:true"})
        docs = []
        for w in data.get("results", [])[:n]:
            wid = w["id"].rsplit("/", 1)[-1]
            authors = ", ".join(a["author"]["display_name"] for a in w.get("authorships", [])[:3])
            docs.append(Doc(id=f"openalex:{wid}", connector="openalex", title=w.get("display_name") or "",
                            url=w.get("doi") or w["id"], year=w.get("publication_year"),
                            text=rebuild_abstract(w.get("abstract_inverted_index"))[:MAX_TEXT],
                            citation=f"{authors} ({w.get('publication_year')}). {w.get('display_name')}."))
        return docs


class EuropePMC(CachedAPI):
    base = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    def search(self, query: str, n: int = 5) -> list[Doc]:
        data = self.get("search", {"query": query, "format": "json", "resultType": "core", "pageSize": n})
        docs = []
        for r in data.get("resultList", {}).get("result", [])[:n]:
            rid = r.get("pmcid") or r.get("pmid") or r.get("id")
            text = re.sub(r"<[^>]+>", "", r.get("abstractText") or "")
            docs.append(Doc(id=f"europepmc:{rid}", connector="europepmc", title=r.get("title") or "",
                            url=f"https://europepmc.org/article/{r.get('source', 'MED')}/{r.get('id')}",
                            year=int(r["pubYear"]) if str(r.get("pubYear", "")).isdigit() else None,
                            text=text[:MAX_TEXT],
                            citation=f"{r.get('authorString', '')} ({r.get('pubYear')}). {r.get('title')} "
                                     f"{r.get('journalTitle', '')}."))
        return docs


class NCBIGene(CachedAPI):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    min_interval = 0.34  # 3 requests a second without a key

    def search(self, query: str, n: int = 3) -> list[Doc]:
        ids = self.get("esearch.fcgi", {"db": "gene", "term": query, "retmode": "json", "retmax": n})
        idlist = ids.get("esearchresult", {}).get("idlist", [])[:n]
        if not idlist:
            return []
        summary = self.get("esummary.fcgi", {"db": "gene", "id": ",".join(idlist), "retmode": "json"})
        docs = []
        for gid in idlist:
            g = summary.get("result", {}).get(gid, {})
            organism = g.get("organism", {}).get("scientificname", "")
            docs.append(Doc(id=f"ncbi_gene:{gid}", connector="ncbi_gene",
                            title=f"{g.get('name', gid)}: {g.get('description', '')} ({organism})",
                            url=f"https://www.ncbi.nlm.nih.gov/gene/{gid}",
                            text=(f"{g.get('summary', '')} Chromosome {g.get('chromosome', '?')}, "
                                  f"location {g.get('maplocation', '?')}. Aliases: {g.get('otheraliases', '')}.")
                            [:MAX_TEXT],
                            citation=f"NCBI Gene {gid}: {g.get('name', '')}, {organism}."))
        return docs


Searcher = Callable[[str, int], list[Doc]]
CONNECTORS = ("openalex", "europepmc", "ncbi_gene")


def connectors(cache_dir: Path, calls_per_round: int = 60) -> dict[str, Searcher]:
    """The live connectors, sharing one call budget per research round."""
    budget = CallBudget("research", calls_per_round)
    contact = os.environ.get("SIMPLE_CONTACT_EMAIL", "")
    openalex_auth = {"api_key": os.environ["OPENALEX_API_KEY"]} if os.environ.get("OPENALEX_API_KEY") else {}
    ncbi_auth = {"api_key": os.environ["NCBI_API_KEY"]} if os.environ.get("NCBI_API_KEY") else {}
    return {
        "openalex": OpenAlex(cache_dir / "openalex", budget, auth_params=openalex_auth | (
            {"mailto": contact} if contact else {})).search,
        "europepmc": EuropePMC(cache_dir / "europepmc", budget).search,
        "ncbi_gene": NCBIGene(cache_dir / "ncbi", budget, auth_params=ncbi_auth).search,
    }
