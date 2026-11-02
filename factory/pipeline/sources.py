"""Research sources: plain-code connectors to public APIs. Searching needs no model, so researchers are code.

Each connector turns a query into ``Doc`` records: a title, a link, the text a model may read (an abstract, the part
of a full text or a manual that matches the query, or a gene summary), a citation and a license. All four free
sources here are open to any model:

- OpenAlex (scholarly works across fields; abstracts arrive as an inverted index and are rebuilt here)
- Europe PMC (PubMed and PubMed Central; for open-access papers, the passage of the full text that best matches the
  query follows the abstract, since methods and parameters are rarely in an abstract)
- NCBI Gene (gene summaries, through the E-utilities)
- Reference documentation (``docs``): a fixed catalog of manuals and API references (NCBI E-utilities, RefSeq,
  Primer3, Biopython). The first live run (2026-10-30) showed why: how-to topics such as calling the E-utilities
  cannot be supported by paper abstracts, so the wiki verifier rejected every page written from them.

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
FULL_TEXTS = 2  # open-access full texts fetched per Europe PMC search


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 2}


def best_window(text: str, query: str, size: int) -> str:
    """The stretch of ``text`` (about ``size`` characters, cut at paragraph or sentence ends where it can) that holds
    the most query words."""
    if len(text) <= size:
        return text
    want = words(query)
    step = max(size // 2, 1)
    starts = range(0, max(len(text) - size, 0) + step, step)
    best = max(starts, key=lambda i: sum(w in want for w in re.findall(r"[a-z0-9_]+", text[i:i + size].lower())))
    window = text[best:best + size]
    if best > 0 and (cut := window.find(". ")) != -1 and cut < size // 4:
        window = window[cut + 2:]
    return window.strip()


def page_text(html: str) -> str:
    """Readable text from an HTML or XML page: no scripts, styles or tags, paragraphs kept apart."""
    html = re.sub(r"(?is)<(script|style|nav|header|footer)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?i)</?(p|div|h[1-6]|li|tr|br|sec|title|table)[^>]*>", "\n", html)
    text = re.sub(r"<[^>]+>", " ", html)
    for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
                         ("&#39;", "'")):
        text = text.replace(entity, char)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\s*\n\s*", "\n", text).strip()


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
        docs, full_texts = [], 0
        for r in data.get("resultList", {}).get("result", [])[:n]:
            rid = r.get("pmcid") or r.get("pmid") or r.get("id")
            text = re.sub(r"<[^>]+>", "", r.get("abstractText") or "")
            if r.get("isOpenAccess") == "Y" and r.get("pmcid") and full_texts < FULL_TEXTS:
                full_texts += 1
                body = page_text(self.get_text(f"{self.base}/{r['pmcid']}/fullTextXML", accept="application/xml"))
                if body:
                    text = (text[:1500] + "\n\nFrom the full text:\n"
                            + best_window(body, query, MAX_TEXT - min(len(text), 1500) - 30))
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


# The reference documentation the ``docs`` connector can read: key, title, address, words it is about. A search
# picks the entries whose title and words share the most with the query, then reads the matching part of each page.
DOCS_CATALOG = (
    ("ncbi_eutils_intro", "A General Introduction to the E-utilities (NCBI Entrez Programming Utilities Help)",
     "https://www.ncbi.nlm.nih.gov/books/NBK25497/",
     "entrez eutils esearch efetch esummary elink epost api ncbi gene nucleotide database rate limit key"),
    ("ncbi_eutils_samples", "Sample Applications of the E-utilities (NCBI Entrez Programming Utilities Help)",
     "https://www.ncbi.nlm.nih.gov/books/NBK25498/",
     "entrez eutils esearch efetch elink pipeline gene mrna sequence download fasta example"),
    ("ncbi_eutils_indepth", "The E-utilities In-Depth: Parameters, Syntax and More (NCBI)",
     "https://www.ncbi.nlm.nih.gov/books/NBK25499/",
     "entrez eutils efetch esearch parameters rettype retmode fasta genbank db term retmax api key email tool "
     "rate limit"),
    ("ncbi_refseq_faq", "RefSeq Frequently Asked Questions (NCBI)", "https://www.ncbi.nlm.nih.gov/books/NBK50679/",
     "refseq accession prefix nm_ xm_ nr_ mrna transcript curated predicted reviewed validated select"),
    ("primer3_manual", "Primer3 manual", "https://primer3.org/manual.html",
     "primer design pcr tm melting gc clamp product size self complementarity hairpin penalty settings"),
    ("biopython_meltingtemp", "Bio.SeqUtils.MeltingTemp (Biopython API)",
     "https://biopython.org/docs/latest/api/Bio.SeqUtils.MeltingTemp.html",
     "melting temperature tm nearest neighbor santalucia wallace gc salt correction primer python"),
    ("biopython_entrez", "Bio.Entrez (Biopython API)", "https://biopython.org/docs/latest/api/Bio.Entrez.html",
     "entrez efetch esearch email api key python ncbi"),
)


class ReferenceDocs(CachedAPI):
    """Manuals and API references from DOCS_CATALOG, read from their public pages."""

    def search(self, query: str, n: int = 3) -> list[Doc]:
        want = words(query)
        scored = sorted(((len(want & words(f"{title} {about}")), key, title, url)
                         for key, title, url, about in DOCS_CATALOG), reverse=True)
        docs = []
        for score, key, title, url in scored[:n]:
            if score == 0:
                break
            text = page_text(self.get_text(url))
            if text:
                docs.append(Doc(id=f"docs:{key}", connector="docs", title=title, url=url,
                                text=best_window(text, query, MAX_TEXT), citation=f"{title}. {url}"))
        return docs


Searcher = Callable[[str, int], list[Doc]]
CONNECTORS = ("openalex", "europepmc", "ncbi_gene", "docs")


def connectors(cache_dir: Path, calls_per_round: int = 60) -> dict[str, Searcher]:
    """The live connectors, sharing one call budget per research round."""
    budget = CallBudget("research", calls_per_round)
    contact = os.environ.get("ASPIRE_CONTACT_EMAIL", "")
    openalex_auth = {"api_key": os.environ["OPENALEX_API_KEY"]} if os.environ.get("OPENALEX_API_KEY") else {}
    ncbi_auth = {"api_key": os.environ["NCBI_API_KEY"]} if os.environ.get("NCBI_API_KEY") else {}
    return {
        "openalex": OpenAlex(cache_dir / "openalex", budget, auth_params=openalex_auth | (
            {"mailto": contact} if contact else {})).search,
        "europepmc": EuropePMC(cache_dir / "europepmc", budget).search,
        "ncbi_gene": NCBIGene(cache_dir / "ncbi", budget, auth_params=ncbi_auth).search,
        "docs": ReferenceDocs(cache_dir / "docs", budget).search,
    }
