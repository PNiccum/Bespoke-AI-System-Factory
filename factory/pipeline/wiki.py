"""The wiki: what SIMPLE has learned, shared by every project, one page per topic.

A page is written by the librarian from approved sources and checked by a verifier of a different model family. Only
a verified page with enough sources counts as covering its topic, so a later project on the same subject skips the
research. Each page is two files: ``<slug>.json`` (the record) and ``<slug>.md`` (the same page for people to read).
A page built from any ARC-only source is ARC-only itself.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from factory.pipeline.blackboard import License

MIN_SOURCES = 2
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


class Wiki:
    def __init__(self, root: Path):
        self.root = root

    def pages(self) -> list[WikiPage]:
        if not self.root.exists():
            return []
        return [WikiPage.model_validate_json(p.read_text()) for p in sorted(self.root.glob("*.json"))]

    def save(self, page: WikiPage) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        old = self.get(page.topic)
        if old:
            page.projects = sorted(set(old.projects) | set(page.projects))
        page.updated = datetime.now(UTC).isoformat()
        (self.root / f"{page.slug}.md").write_text(page.markdown())
        path = self.root / f"{page.slug}.json"
        path.write_text(page.model_dump_json(indent=1))
        return path

    def get(self, topic: str) -> WikiPage | None:
        path = self.root / f"{slugify(topic)}.json"
        return WikiPage.model_validate_json(path.read_text()) if path.exists() else None

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
        return page if page and page.verified and len(page.sources) >= MIN_SOURCES else None
