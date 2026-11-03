"""The shared blackboard: one append-only log per project that every agent reads and writes.

A project is a folder, ``<workspace>/projects/<id>/``, holding ``task.txt`` (what was asked, written once the person
approves the plan), ``blackboard.jsonl``, and the files agents produce. Each entry is typed (a question, a plan, a
vote, a source, a wiki page, a critique), names who wrote it and which entries it builds on. So the log is also the
project's provenance, the recording a demo replays, and what the auditors read when a project goes wrong.

Every entry carries a license. Text that may not leave ARC (Elsevier full text, lab data) is ``arc_only``, and
``for_paid`` leaves it out of anything sent to Claude or OpenAI.
"""

from __future__ import annotations

import hashlib
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

License = Literal["public", "arc_only"]

# What an entry can be. A new stage adds its kinds here, so a typo in an agent's code fails at once instead of
# writing entries nobody reads.
KINDS = frozenset({
    "request",  # the person's first words
    "question",  # the Interviewer's questions
    "answer",  # the person's answers, or their changes to a proposed plan
    "plan",  # a proposed plan, with the problems plain code found in it
    "approval",  # the person approved a plan
    "critique",  # the plan critic's weaknesses of Plan A, and its Plan B
    "analysis",  # pros and cons of each plan
    "edit",  # the editor's versions of both plans
    "recheck",  # the plan critic's check of which of its weaknesses the edited Plan A still has
    "vote",  # one agent's vote between the plans
    "decision",  # the winning plan, counted in code
    "topics",  # what the knowledge evaluator says the work needs
    "coverage",  # which of those topics the wiki already covers
    "query",  # one search the research orchestrator ran, and what it found
    "source",  # a document the research orchestrator approved
    "wiki_check",  # the wiki verifier's check of a page
    "wiki_page",  # a page the librarian wrote or updated
    "knowledge",  # the knowledge loop's result
    "design",  # the head orchestrator's design of the solution
    "orders",  # a sub-orchestrator's work orders for one task
    "build",  # what a builder wrote for one work order
    "checks",  # plain-code checks of the solution
    "critic_review",  # the project critic's issues
    "fixes",  # the problem solver's fixes and the critic's verdicts
    "final",  # the final evaluator's QA and the deliverable
    "audit",  # an auditor's findings in the re-design loop
    "debate",  # one turn of the paid expert debate
    "redesign",  # the re-design report
    "file",  # a file an agent wrote into the project folder
    "note",  # anything else worth keeping
})


class Entry(BaseModel):
    id: str
    at: str
    kind: str
    author: str  # an agent name, "person", or "code"
    data: dict
    refs: list[str] = []  # entries this one builds on
    license: License = "public"


class Blackboard:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()  # researchers will post from parallel threads

    def post(self, kind: str, author: str, data: dict, refs: list[str] | tuple[str, ...] = (),
             license: License = "public") -> Entry:
        if kind not in KINDS:
            raise ValueError(f"unknown blackboard entry kind {kind!r}")
        with self._lock:
            n = len(self.entries()) + 1
            entry = Entry(id=f"{n:04d}-{kind}", at=datetime.now(UTC).isoformat(), kind=kind, author=author, data=data,
                          refs=list(refs), license=license)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(entry.model_dump_json() + "\n")
        return entry

    def entries(self, kind: str | None = None) -> list[Entry]:
        if not self.path.exists():
            return []
        rows = [Entry.model_validate_json(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [e for e in rows if kind is None or e.kind == kind]

    def latest(self, kind: str) -> Entry | None:
        found = self.entries(kind)
        return found[-1] if found else None

    def get(self, entry_id: str) -> Entry:
        for e in self.entries():
            if e.id == entry_id:
                return e
        raise KeyError(entry_id)

    @staticmethod
    def for_paid(entries: list[Entry]) -> list[Entry]:
        """The entries that may be shown to a paid provider: everything not tagged ARC-only."""
        return [e for e in entries if e.license == "public"]


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.id = root.name
        self.board = Blackboard(root / "blackboard.jsonl")

    @classmethod
    def create(cls, base: Path, title: str) -> Project:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40].strip("-") or "project"
        stem = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{slug}"
        for n in range(1, 100):  # two projects with the same title in the same second get -2, -3, ...
            root = base / (stem if n == 1 else f"{stem}-{n}")
            try:
                root.mkdir(parents=True, exist_ok=False)
                return cls(root)
            except FileExistsError:
                continue
        raise FileExistsError(f"too many projects named {stem}")

    @classmethod
    def open(cls, root: Path) -> Project:
        if not (root / "blackboard.jsonl").exists():
            raise FileNotFoundError(f"{root} is not a project (no blackboard.jsonl)")
        return cls(root)

    def write_file(self, rel: str, text: str, author: str, refs: list[str] | tuple[str, ...] = (),
                   license: License = "public") -> Path:
        """Write a file inside the project folder and record it on the blackboard."""
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"{rel!r} is outside the project folder")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.board.post("file", author, {"path": rel, "sha256": hashlib.sha256(text.encode()).hexdigest()}, refs,
                        license)
        return path
