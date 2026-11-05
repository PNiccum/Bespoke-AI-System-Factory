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

import json
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path

import httpx
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


BLOCK_PAGE = re.compile(r"(?i)captcha|enable javascript|javascript is (disabled|required)|just a moment|"
                        r"access denied|attention required|checking your browser|unusual traffic")
MIN_DOC_TEXT = 400  # a manual page shorter than this is a stub, a redirect or a block page


def readable(raw: str) -> str:
    """Text from a fetched manual: a JSON help record (UniProt's REST help returns {"title", "content"}) gives its
    title and content; anything else is read as HTML."""
    if raw.lstrip().startswith("{"):
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            return f"{data.get('title', '')}\n{data['content']}".strip()
    return page_text(raw)


def looks_blocked(text: str) -> bool:
    """True for what a site serves a script instead of the page: a captcha, a "turn on JavaScript" stub, a
    bot-check, or too little text to be a manual."""
    return len(text) < MIN_DOC_TEXT or (len(text) < 3000 and bool(BLOCK_PAGE.search(text)))


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
    # NCBI Bookshelf pages (the E-utilities manual, the RefSeq FAQ) answer scripts with a reCAPTCHA page, so the
    # E-utilities come from Biopython's Entrez chapter and the RefSeq prefixes from the release notes on NCBI's FTP.
    ("eutils_biopython_tutorial", "Accessing NCBI's Entrez databases (Biopython Tutorial, E-utilities)",
     "https://biopython.org/docs/latest/Tutorial/chapter_entrez.html",
     "entrez eutils esearch efetch esummary elink epost einfo api ncbi gene nucleotide database rettype retmode "
     "fasta genbank retmax rate limit key email tool parameters pipeline mrna sequence download example"),
    ("refseq_release_notes", "RefSeq release notes: accession prefixes and molecule types (NCBI FTP)",
     "https://ftp.ncbi.nlm.nih.gov/refseq/release/release-notes/RefSeq-release237.txt",
     "refseq accession prefix nm_ xm_ nr_ np_ mrna transcript curated predicted reviewed validated molecule type"),
    ("primer3_manual", "Primer3 manual", "https://primer3.org/manual.html",
     "primer design pcr tm melting gc clamp product size self complementarity hairpin penalty settings"),
    ("biopython_meltingtemp", "Bio.SeqUtils.MeltingTemp (Biopython API)",
     "https://biopython.org/docs/latest/api/Bio.SeqUtils.MeltingTemp.html",
     "melting temperature tm nearest neighbor santalucia wallace gc salt correction primer python"),
    ("biopython_entrez", "Bio.Entrez (Biopython API)", "https://biopython.org/docs/latest/api/Bio.Entrez.html",
     "entrez efetch esearch email api key python ncbi"),
    # Added 2026-11-04: the project tiers stopped at research on file formats and APIs with no manual here
    ("uniprot_accessions", "UniProt help: accession numbers", "https://rest.uniprot.org/help/accession_numbers",
     "uniprot uniprotkb accession number format regex primary secondary entry name identifier"),
    ("uniprot_isoforms", "UniProt help: canonical and isoform sequences",
     "https://rest.uniprot.org/help/canonical_and_isoforms",
     "uniprot isoform suffix dash canonical sequence variant accession protein"),
    ("uniprot_api_queries", "UniProt help: programmatic access, query syntax",
     "https://rest.uniprot.org/help/api_queries",
     "uniprot rest api query search stream format json tsv fasta fields organism reviewed download"),
    ("uniprot_return_fields", "UniProt help: return fields", "https://rest.uniprot.org/help/return_fields",
     "uniprot return fields columns json schema cross references go reactome xref gene names organism"),
    ("europepmc_fields", "Europe PMC REST search fields",
     "https://www.ebi.ac.uk/europepmc/webservices/rest/fields?format=json",
     "europe pmc rest api search query fields syntax json result pmid pmcid abstract title author"),
    # Added 2026-11-05: four runs stopped on one API schema or export format each. These addresses have not been
    # fetched from the cloud container (its network allows no host here); a page that errors or is blocked skips itself.
    ("europepmc_rest", "Europe PMC RESTful web service (search syntax, quoting, paging)",
     "https://europepmc.org/RestfulWebService",
     "europe pmc rest web service search query syntax quoting phrase boolean field and or not pageSize cursorMark "
     "resultType format json parameters"),
    ("david_functional_annotation", "DAVID functional annotation tool: help and chart output",
     "https://davidbioinformatics.nih.gov/content.jsp?file=functional_annotation.html",
     "david functional annotation chart report export download table columns term count percent pvalue fold "
     "enrichment benjamini bonferroni fdr ease score gene list tsv"),
    ("opentargets_graphql", "Open Targets Platform: GraphQL API", "https://platform-docs.opentargets.org/data-access/graphql-api",
     "open targets platform graphql api query target disease drug associated evidence clinical phase known drugs"),
    ("opentargets_schema", "Open Targets Platform GraphQL schema (SDL)",
     "https://api.platform.opentargets.org/api/v4/graphql/schema",
     "open targets graphql schema type target drug knowndrugs clinical phase mechanism of action disease evidence "
     "association score fields"),
    ("cellosaurus_description", "Cellosaurus: description of the cell line knowledge resource",
     "https://www.cellosaurus.org/description.html",
     "cellosaurus cell line name nomenclature identifier cvcl accession synonyms mcf-7 mcf7 normalization "
     "misidentified naming conventions"),
    ("go_ontology_docs", "Gene Ontology: ontology documentation (term IDs, relations, cell cycle terms)",
     "https://geneontology.org/docs/ontology-documentation/",
     "gene ontology go term id GO: biological process cell cycle phase g1 s g2 m mitosis regulation of cell cycle "
     "reactome pathway mapping identifiers is_a part_of"),
    ("hpa_download", "Human Protein Atlas: downloadable data", "https://www.proteinatlas.org/about/download",
     "human protein atlas hpa download tsv rna expression cell line tissue file format columns"),
    ("go_annotations", "Gene Ontology: GO annotations", "https://geneontology.org/docs/go-annotations/",
     "gene ontology go term annotation evidence code aspect biological process molecular function"),
    ("chembl_api", "ChEMBL web services API documentation",
     "https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services",
     "chembl api target molecule activity mechanism drug json rest"),
    ("openalex_works", "OpenAlex API: works", "https://docs.openalex.org/api-entities/works",
     "openalex api works filter search json fields cited by abstract inverted index authorships"),
    ("openalex_work_object", "OpenAlex API: the Work object (record fields)",
     "https://help.openalex.org/data/works/attributes/",
     "openalex work object record json schema fields doi title publication year authorships cited by count "
     "abstract inverted index primary location open access"),
    ("openalex_lists", "OpenAlex API: get lists of entities (endpoint, paging, per-page)",
     "https://help.openalex.org/api/paging/",
     "openalex api endpoint query parameters list works page per page cursor paging sort select sample"),
    ("openalex_filters", "OpenAlex API: filter entity lists",
     "https://help.openalex.org/api/filtering/",
     "openalex api filter query parameters works publication year from to date type is oa search"),
    ("apache_logs", "Apache HTTP Server: log files", "https://httpd.apache.org/docs/2.4/logs.html",
     "apache access log common combined format timestamp timezone offset quoted request status bytes"),
    ("python_csv", "Python csv module", "https://docs.python.org/3/library/csv.html",
     "python csv tsv reader writer dialect sniffer delimiter detection quoting escape header"),
    ("python_codecs", "Python codecs: standard encodings", "https://docs.python.org/3/library/codecs.html",
     "python utf-8 utf-8-sig bom byte order mark decode error handling encoding"),
    ("python_datetime", "Python datetime module", "https://docs.python.org/3/library/datetime.html",
     "python datetime strptime timezone utc offset parse format timestamp"),
    ("python_json", "Python json module", "https://docs.python.org/3/library/json.html",
     "python json dump load output format ensure ascii indent"),
    ("gfm_spec", "GitHub Flavored Markdown spec (tables)", "https://github.github.com/gfm/",
     "markdown table pipe escaping gfm commonmark syntax"),
    ("pytest_layout", "pytest: good integration practices (src layout, imports)",
     "https://docs.pytest.org/en/stable/explanation/goodpractices.html",
     "pytest src layout test import path conftest rootdir pythonpath package"),
    ("pyproject_guide", "Python Packaging: writing pyproject.toml",
     "https://packaging.python.org/en/latest/guides/writing-pyproject-toml/",
     "pyproject toml packaging entry point console scripts build backend dependencies src layout pip install"),
)


# served as JSON, and refused (406) when asked for text/html: UniProt help {"title", "content"}, Europe PMC's field list
JSON_HELP = ("https://rest.uniprot.org/help/", "https://www.ebi.ac.uk/europepmc/webservices/rest/")


class ReferenceDocs(CachedAPI):
    """Manuals and API references from DOCS_CATALOG, read from their public pages."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._skips = threading.local()  # topics are researched in parallel threads, each calling search()

    @property
    def skipped(self) -> list[str]:
        """The pages the last ``search`` on this thread passed over, and why (a 406, a captcha, a stub). Per thread,
        so concurrent searches for different topics never see each other's skips. See ``skipped_pages``."""
        return getattr(self._skips, "items", [])

    def search(self, query: str, n: int = 3) -> list[Doc]:
        want = words(query)
        scored = sorted(((len(want & words(f"{title} {about}")), key, title, url)
                         for key, title, url, about in DOCS_CATALOG), reverse=True)
        docs = []
        self._skips.items = []
        for score, key, title, url in scored[:n]:
            if score == 0:
                break
            accept = "application/json" if url.startswith(JSON_HELP) else "text/html"
            try:  # one page that errors skips itself: on 2026-11-04 a 406 from one entry emptied 25 of 32 searches
                text = readable(self.get_text(url, accept=accept))
            except httpx.HTTPError as e:
                self._skips.items.append(f"{key}: {type(e).__name__}: {str(e)[:120]}")
                continue
            if looks_blocked(text):
                self._skips.items.append(f"{key}: blocked or stub page ({len(text)} chars)")
            else:
                docs.append(Doc(id=f"docs:{key}", connector="docs", title=title, url=url,
                                text=best_window(text, query, MAX_TEXT), citation=f"{title}. {url}"))
        return docs


class PyPI(CachedAPI):
    """An exact-name lookup of a Python package: its summary, newest release, licence and README. PyPI has no search
    API, so the name comes from a model or a search of GitHub, and this reply is what proves the package exists (a
    model's invented package name returns nothing and is dropped)."""
    base = "https://pypi.org/pypi"

    def lookup(self, name: str) -> dict:
        name = re.sub(r"[^A-Za-z0-9._-]", "", name.strip())
        return self.get(f"{name}/json") if name else {}

    def search(self, query: str, n: int = 3) -> list[Doc]:
        docs = []
        for name in re.split(r"[\s,;]+", query.strip())[:n]:
            info = (self.lookup(name) or {}).get("info")
            if not info:
                continue
            text = f"{info.get('summary') or ''}\n\n{info.get('description') or ''}".strip()
            docs.append(Doc(id=f"pypi:{info['name'].lower()}", connector="pypi", title=f"{info['name']} {info['version']}",
                            url=f"https://pypi.org/project/{info['name']}/", text=text[:MAX_TEXT],
                            citation=f"{info['name']} {info['version']} on PyPI. https://pypi.org/project/{info['name']}/"))
        return docs


class GitHubRepos(CachedAPI):
    """Repository search on GitHub (public repositories only; ``GITHUB_TOKEN`` raises the rate limit if set, and is
    sent with the request only). Used to find existing tools and reference implementations for a task."""
    base = "https://api.github.com"
    min_interval = 2.0  # 30 searches a minute with a token, 10 without

    def repos(self, query: str, n: int = 5) -> list[dict]:
        data = self.get("search/repositories", {"q": f"{query} language:python archived:false fork:false",
                                                "sort": "stars", "order": "desc", "per_page": n})
        return data.get("items", [])[:n]

    def search(self, query: str, n: int = 5) -> list[Doc]:
        docs = []
        for r in self.repos(query, n):
            lic = (r.get("license") or {}).get("spdx_id") or "no licence"
            docs.append(Doc(id=f"github:{r['full_name'].lower()}", connector="github", title=r["full_name"],
                            url=r["html_url"], year=int((r.get("pushed_at") or "0")[:4]) or None,
                            text=(f"{r.get('description') or ''} Stars {r.get('stargazers_count')}, licence {lic}, "
                                  f"last push {(r.get('pushed_at') or '')[:10]}, topics: "
                                  f"{', '.join(r.get('topics') or [])}.")[:MAX_TEXT],
                            citation=f"{r['full_name']} on GitHub. {r['html_url']}"))
        return docs


Searcher = Callable[[str, int], list[Doc]]


def skipped_pages(searcher: Searcher) -> list[str]:
    """What a searcher passed over in its last call on this thread: a connector's bound ``search`` method exposes
    ``skipped`` on its instance; a plain function (a test engine) has none. The Searcher signature stays a callable
    returning documents, so every engine keeps working unchanged."""
    return list(getattr(getattr(searcher, "__self__", None), "skipped", []))
CONNECTORS = ("openalex", "europepmc", "ncbi_gene", "docs", "pypi", "github")


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
        "pypi": PyPI(cache_dir / "pypi", budget).search,
        "github": GitHubRepos(cache_dir / "github", budget, headers=(
            {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"} if os.environ.get("GITHUB_TOKEN") else {})
        ).search,
    }
