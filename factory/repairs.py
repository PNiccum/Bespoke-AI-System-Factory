"""The repair library: QA repairs that stuck, kept so the solver can see the closest ones next time.

When final QA rejects a build, ``CriticLoop.repair`` has the solver fix QA's findings. Each applied fix is saved here as
an issue plus a diff, *pending*. Only when the build is then approved by final QA does code mark it *stuck*; a repair of
a build that was rejected, rebuilt or re-designed is deleted, so the library holds only repairs with a good outcome.
Before the next repair the solver is shown the closest stuck repairs (word overlap on the issue text, no model).

``fingerprint`` is the event log's fingerprint of the issue, so code that recognises a known failure
(``factory/events.py``, the deterministic-fix work) can look entries up by the same key instead of by similarity.
"""

from __future__ import annotations

import difflib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from factory.events import fingerprint
from factory.pipeline.wiki import words

MAX_DIFF = 1200  # characters of diff kept per repair
MAX_ISSUE = 600
MIN_SIMILARITY = 0.15  # below this an entry is not "close" to the issue and is not shown


class Repair(BaseModel):
    id: str
    at: str
    run: str = ""
    project: str = ""
    issue: str  # what QA found (or the failing check), as the solver saw it
    fingerprint: str
    explanation: str = ""  # the solver's one-line account of the change
    files: list[str] = []
    diff: str  # unified diff of the files the repair changed, cut at MAX_DIFF
    status: Literal["pending", "stuck"] = "pending"
    evidence: str = ""  # what made it stuck, e.g. "delivered in round 1 after 1 QA repair(s)"
    stuck_at: str = ""


def diff_of(before: dict[str, str], after: dict[str, str], paths: list[str]) -> str:
    out = []
    for rel in paths:
        a, b = before.get(rel, ""), after.get(rel, "")
        if a != b:
            out += difflib.unified_diff(a.splitlines(), b.splitlines(), f"a/{rel}", f"b/{rel}", n=2, lineterm="")
    text = "\n".join(out)
    return text if len(text) <= MAX_DIFF else text[:MAX_DIFF] + "\n... (diff cut)"


def similarity(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


class RepairLibrary:
    def __init__(self, root: Path):
        self.root = root

    @classmethod
    def of(cls, cfg) -> RepairLibrary:
        return cls(cfg.work_dir / "repairs")

    def _path(self, rid: str) -> Path:
        return self.root / f"{rid}.json"

    def _save(self, r: Repair) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(r.id).write_text(r.model_dump_json(indent=1), encoding="utf-8")

    def all(self) -> list[Repair]:
        if not self.root.exists():
            return []
        out = []
        for p in self.root.glob("*.json"):
            try:
                out.append(Repair.model_validate_json(p.read_text(encoding="utf-8")))
            except ValueError:
                continue  # a file cut short by a stopped run
        return sorted(out, key=lambda r: (r.at, r.id))

    def stuck(self) -> list[Repair]:
        return [r for r in self.all() if r.status == "stuck"]

    def add(self, issue: str, diff: str, *, explanation: str = "", files: list[str] = (), run: str = "",
            project: str = "") -> str:
        """Save an applied repair as pending. Returns its id ('' for an empty diff, which repairs nothing)."""
        if not diff.strip():
            return ""
        issue = " ".join(issue.split())[:MAX_ISSUE]
        r = Repair(id=uuid.uuid4().hex[:10], at=datetime.now(UTC).isoformat(), run=run, project=project, issue=issue,
                   fingerprint=fingerprint("qa", "final QA repair", issue), explanation=explanation[:300],
                   files=list(files), diff=diff)
        self._save(r)
        return r.id

    def confirm(self, ids: list[str], evidence: str) -> int:
        """The build these repairs belong to was approved: they stuck. A stuck entry for the same fingerprint is
        replaced, so the library keeps the newest repair of each problem, not a pile of repeats."""
        n = 0
        for rid in ids:
            p = self._path(rid)
            if not p.exists():
                continue
            r = Repair.model_validate_json(p.read_text(encoding="utf-8"))
            for old in self.stuck():
                if old.fingerprint == r.fingerprint and old.id != r.id:
                    self._path(old.id).unlink(missing_ok=True)
            r.status, r.evidence, r.stuck_at = "stuck", evidence[:200], datetime.now(UTC).isoformat()
            self._save(r)
            n += 1
        return n

    def discard(self, ids: list[str]) -> None:
        for rid in ids:
            self._path(rid).unlink(missing_ok=True)

    def closest(self, texts: list[str], n: int = 3) -> list[Repair]:
        """The stuck repairs whose issue is nearest any of these texts, nearest first."""
        scored = [(max((similarity(t, r.issue) for t in texts), default=0.0), r) for r in self.stuck()]
        return [r for s, r in sorted(scored, key=lambda x: (x[0], x[1].at), reverse=True) if s >= MIN_SIMILARITY][:n]

    def section(self, texts: list[str], n: int = 3) -> str:
        """Text for the solver's prompt: the closest stuck repairs, or '' when there are none."""
        found = self.closest(texts, n)
        if not found:
            return ""
        out = ["\n## Past repairs of similar findings",
               "Each was applied and final QA then approved the result. Follow the pattern, not the letter: the files "
               "here may differ from the ones in these diffs."]
        for r in found:
            out.append(f"\n### Finding: {r.issue}\nFix: {r.explanation or 'see diff'}\n```diff\n{r.diff}\n```")
        return "\n".join(out) + "\n"


def library_of(llm) -> RepairLibrary | None:
    """The library for a run's workspace, or None for an LLM stand-in that has no config."""
    cfg = getattr(llm, "cfg", None)
    return RepairLibrary.of(cfg) if cfg is not None and hasattr(cfg, "work_dir") else None
