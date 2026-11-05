"""Remembered reuse candidates: what the prior-art scout found, kept as wiki pages keyed by task kind.

A page such as "apache log parsing" lists the tools the scout judged usable (``import`` or ``reference``) with the
facts that decided each one: URL, licence, date of the latest release or push, the rule's reason, the date fetched
and the projects that found it. A later run of the same kind of task reads the page instead of searching GitHub and
PyPI again. Pages live in ``<wiki>/reuse/`` (a ``.json`` record and a readable ``.md`` page each, plus an
``index.md``), apart from the research pages, which are written from approved sources and verified by a second model.
Nothing here is written by a model: the task kind is a few words from the scout's one ARC proposal and every other
field comes from code, so a page is as trustworthy as the lookups behind it. A page older than ``MAX_AGE_DAYS`` is
not used (the facts go stale), and the next search refreshes it.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

from factory.pipeline.scout import Candidate
from factory.pipeline.wiki import slugify, words

MAX_AGE_DAYS = 180
MATCH = 0.67  # share of words two task kinds must have in common for a lookup to count them as the same kind
_LOCK = threading.RLock()


class ReusePage(BaseModel):
    kind: str  # the task kind, e.g. "apache log parsing"
    candidates: list[Candidate] = []
    projects: list[str] = []  # projects whose search found or reused it
    updated: str = ""  # when a search last refreshed the page (ISO)

    @property
    def slug(self) -> str:
        return slugify(self.kind)

    def markdown(self) -> str:
        lines = [f"# Reuse: {self.kind}", "", f"_Updated {self.updated[:10]}; projects "
                 f"{', '.join(self.projects) or '-'}. Facts fetched by code; nothing here is model-written._", ""]
        for c in self.candidates:
            lines.append(f"- **[{c.use}] {c.name}** ({c.kind}): {c.summary[:200]} Licence {c.license or 'unknown'}, "
                         f"updated {c.updated or '?'}, fetched {c.fetched}. {c.why}. <{c.url}>")
        return "\n".join(lines) + "\n"


class ReuseMemory:
    def __init__(self, wiki_root: Path, now=lambda: datetime.now(UTC)):
        self.root, self.now = Path(wiki_root) / "reuse", now

    def pages(self) -> list[ReusePage]:
        with _LOCK:
            if not self.root.exists():
                return []
            out = []
            for p in sorted(self.root.glob("*.json")):
                try:
                    out.append(ReusePage.model_validate_json(p.read_text(encoding="utf-8")))
                except (ValueError, OSError):
                    continue  # a corrupt or old-schema page is ignored; the next search rewrites it
            return out

    def find(self, kind: str) -> ReusePage | None:
        """The page for this task kind (same slug, or most shared words, at least ``MATCH``), fresh or not."""
        want, best, score = words(kind), None, 0.0
        for page in self.pages():
            have = words(page.kind)
            overlap = len(want & have) / max(1, len(want | have))
            if page.slug == slugify(kind):
                return page
            if overlap > score:
                best, score = page, overlap
        return best if score >= MATCH else None

    def recall(self, kind: str) -> ReusePage | None:
        """A page with candidates that is younger than ``MAX_AGE_DAYS``, or None (search again)."""
        page = self.find(kind) if kind.strip() else None
        if not page or not page.candidates or not page.updated:
            return None
        try:
            fresh = datetime.fromisoformat(page.updated) > self.now() - timedelta(days=MAX_AGE_DAYS)
        except ValueError:
            return None
        return page if fresh else None

    def remember(self, kind: str, candidates: list[Candidate], project: str = "") -> ReusePage | None:
        """Save the usable candidates (not ``skip``) under the task kind. A fresh search replaces the page's old
        candidates, so a tool that went stale or changed licence drops out; only the projects list carries over. A
        page is only ever replaced under its own kind (same slug), never one a loose word match found. Nothing
        usable, nothing saved."""
        good = [c for c in candidates if c.use != "skip"]
        if not good or not kind.strip():
            return None
        with _LOCK:
            page = self._load(slugify(kind)) or ReusePage(kind=kind.strip())
            page.candidates = good
            page.projects = sorted({*page.projects, *([project] if project else [])})
            page.updated = self.now().isoformat()
            self._write(page)
            self._index()
            return page

    def _load(self, slug: str) -> ReusePage | None:
        try:
            return ReusePage.model_validate_json((self.root / f"{slug}.json").read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def _write(self, page: ReusePage) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name, text in ((f"{page.slug}.md", page.markdown()), (f"{page.slug}.json", page.model_dump_json(indent=1))):
            tmp = self.root / (name + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(self.root / name)  # a reader never sees a half-written file

    def touch(self, page: ReusePage, project: str) -> None:
        """A project reused this page: record it without refreshing the facts' age."""
        if project and project not in page.projects:
            with _LOCK:
                current = self._load(page.slug) or page  # not the snapshot recall read: a refresh may have come since
                current.projects = sorted({*current.projects, project})
                self._write(current)

    def _index(self) -> None:
        lines = ["# Reuse index", "", "| Task kind | Candidates | Updated |", "|---|---|---|"]
        lines += [f"| [{p.kind}]({p.slug}.md) | {', '.join(c.name for c in p.candidates)[:120]} | {p.updated[:10]} |"
                  for p in self.pages()]
        (self.root / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
