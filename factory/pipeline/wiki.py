"""The wiki: what ASPIRE has learned, shared by every project, one page per topic.

A page is written by the librarian from approved sources and checked by a verifier of a different model family. Only
a verified page with enough sources counts as covering its topic, so a later project on the same subject skips the
research. Each page is two files: ``<slug>.json`` (the record) and ``<slug>.md`` (the same page for people to read).
A page built from any ARC-only source is ARC-only itself.

From the public "LLM wiki" pattern (Karpathy's gist and its implementations): ``index.md`` lists every page with its
one-line summary and status and is rewritten on every save, ``log.md`` records each save in order, and ``lint()``
reports broken links, orphan pages, pages that did not pass and pages with too few sources, in plain code.
"""

from __future__ import annotations

import re
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from factory.pipeline.blackboard import License

MIN_SOURCES = 2


def enough_sources(ids) -> bool:
    """MIN_SOURCES documents, or one reference manual (a ``docs:`` id): for how a tool, API or format works, the
    manual is the authority, and a second source rarely exists (the 2026-11-02 primer run lost the NCBI query-syntax
    topic to this rule with the E-utilities manual approved)."""
    ids = list(ids)
    return len(ids) >= MIN_SOURCES or any(str(i).startswith("docs:") for i in ids)
STOP = frozenset("a an and for in of on the to with".split())


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "topic"


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP}


class WikiPage(BaseModel):
    topic: str
    summary: str  # one paragraph
    body: str  # markdown; claims cite sources as [source id]
    sources: list[str]  # doc ids the body cites
    citations: dict[str, str] = {}  # doc id -> citation
    license: License = "public"
    verified: bool = False
    issues: list[str] = []  # what the verifier found, when it did not pass the page
    links: list[str] = []  # slugs of other wiki pages the body links to
    projects: list[str] = []  # projects that needed it
    updated: str = ""

    @property
    def slug(self) -> str:
        return slugify(self.topic)

    def markdown(self) -> str:
        refs = "\n".join(f"- [{i}] {self.citations.get(i, i)}" for i in self.sources)
        status = "verified" if self.verified else "not verified: " + "; ".join(self.issues)
        return f"# {self.topic}\n\n_{status}; license {self.license}_\n\n{self.summary}\n\n{self.body}\n\n" \
               f"## Sources\n{refs}\n"


# Research runs several topics at once, and bench projects run side by side on one wiki: a page is never read while
# another thread is half way through writing it. One lock for the process, since two Wiki objects may share a folder.
_LOCK = threading.RLock()


class Wiki:
    def __init__(self, root: Path):
        self.root = root

    def pages(self) -> list[WikiPage]:
        if not self.root.exists():
            return []
        with _LOCK:
            return [WikiPage.model_validate_json(p.read_text(encoding="utf-8"))
                    for p in sorted(self.root.glob("*.json"))]

    def save(self, page: WikiPage) -> Path:
        with _LOCK:
            return self._save(page)

    def _save(self, page: WikiPage) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        old = self.get(page.topic)
        if old:
            page.projects = sorted(set(old.projects) | set(page.projects))
        page.updated = datetime.now(UTC).isoformat()
        (self.root / f"{page.slug}.md").write_text(page.markdown(), encoding="utf-8")
        path = self.root / f"{page.slug}.json"
        path.write_text(page.model_dump_json(indent=1), encoding="utf-8")
        self.write_index()
        status = "verified" if page.verified else "not verified"
        with (self.root / "log.md").open("a", encoding="utf-8") as log:
            log.write(f"- {page.updated} saved [{page.topic}]({page.slug}.md), {status}, "
                      f"{len(page.sources)} sources, projects {', '.join(page.projects) or '-'}\n")
        return path

    def write_index(self) -> Path:
        lines = ["# Wiki index", "", "| Page | Status | Sources | License | Summary |", "|---|---|---|---|---|"]
        for p in self.pages():
            summary = " ".join(p.summary.split())[:160].replace("|", "/")
            lines.append(f"| [{p.topic}]({p.slug}.md) | {'verified' if p.verified else 'not verified'} | "
                         f"{len(p.sources)} | {p.license} | {summary} |")
        path = self.root / "index.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def lint(self) -> list[str]:
        """Problems a person should look at: no model is called."""
        pages = self.pages()
        slugs = {p.slug for p in pages}
        linked = {s for p in pages for s in p.links}
        out = []
        for p in pages:
            out += [f"{p.slug}: links to missing page {s}" for s in p.links if s not in slugs]
            if not p.verified:
                out.append(f"{p.slug}: not verified ({'; '.join(p.issues)[:120] or 'no reason recorded'})")
            elif not enough_sources(p.sources):
                out.append(f"{p.slug}: verified but cites {len(p.sources)} source(s)")
            if len(pages) > 1 and p.slug not in linked and not p.links:
                out.append(f"{p.slug}: orphan (no links in or out)")
        return out

    def relevant(self, text: str, limit: int = 40) -> list[WikiPage]:
        """The verified pages most relevant to a text (a task and its plan), best first, at most ``limit``. A growing
        wiki can hold more pages than fit in a prompt; listing the first 40 alphabetically hid the rest from the
        knowledge evaluator. Ranked by the share of a page's topic and summary words that the text contains; ties
        keep file order."""
        want = words(text)
        scored = []
        for i, page in enumerate(p for p in self.pages() if p.verified):
            have = words(f"{page.topic} {page.summary}")
            scored.append((-len(want & have) / max(1, len(have)), i, page))
        return [page for _, _, page in sorted(scored, key=lambda t: t[:2])[:limit]]

    def get(self, topic: str) -> WikiPage | None:
        path = self.root / f"{slugify(topic)}.json"
        with _LOCK:
            return WikiPage.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None

    def find(self, topic: str, threshold: float = 0.6) -> WikiPage | None:
        """The page for a topic: the same slug, or else the page whose topic shares most words with it."""
        if page := self.get(topic):
            return page
        want, best, score = words(topic), None, 0.0
        for page in self.pages():
            have = words(page.topic)
            overlap = len(want & have) / max(1, len(want | have))
            if overlap > score:
                best, score = page, overlap
        return best if score >= threshold else None

    def covers(self, topic: str) -> WikiPage | None:
        """A verified page with enough sources for this topic, or None."""
        page = self.find(topic)
        return page if page and page.verified and enough_sources(page.sources) else None
