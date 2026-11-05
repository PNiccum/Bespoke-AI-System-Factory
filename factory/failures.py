"""The failure archive: every project the re-design loop audits becomes a case ASPIRE keeps.

A case is a folder, ``<workspace>/failures/<id>/``, holding ``case.json`` (what was asked, what went wrong, the
causes the auditors traced, and every fix tried) and ``project/``, a snapshot of the project as it stood when the
knowledge loop finished: the task, the plan, the knowledge, and the blackboard up to that point, without anything
production wrote. Replaying a case runs production, the critic loop and final QA again from that snapshot.

The archive does three jobs:

1. **Held-out tests.** A case is real work ASPIRE failed. A proposed fix to an agent the case blames is replayed on
   it (``replay``); the proposer only sees the auditors' guidance, never the case itself.
2. **Regression tests.** Once a fix is adopted and its case delivers, the case is ``fixed``. Later proposals for an
   agent must still deliver the fixed cases that blamed it, so an old failure cannot quietly come back.
3. **Reuse.** The auditors of a new failure see the fixes that worked before (``past_fixes``), and so does the
   proposer of a new prompt.

Replays use free models only, but each one is a whole build, so ``MAX_REPLAYS`` caps them per proposal.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from factory.config import FactoryConfig

# Why a project failed, from a fixed list, so the cause decides what happens next instead of an auditor's prose.
CauseKind = Literal["prompt", "missing_knowledge", "loop_rule", "model_limit", "tool_or_code"]
CAUSE_KINDS: dict[str, str] = {
    "prompt": "a model agent was instructed badly or too vaguely (fixed by changing that agent's prompt)",
    "missing_knowledge": "the wiki lacked a fact or method the work needed (fixed by research)",
    "loop_rule": "a stop rule, round limit, threshold or hand-off between stages was wrong",
    "model_limit": "the model could not do it however it was asked (fixed by a stronger model)",
    "tool_or_code": "a tool, data source or piece of ASPIRE's code was missing or wrong",
}
MAX_REPLAYS = 3  # the target case plus up to two regression cases, per proposal
SNAPSHOT_SKIP = {"output", "redesign"}  # what production and re-design wrote


class Cause(BaseModel):
    kind: CauseKind
    agent: str = ""  # the agent it concerns, when there is one
    evidence: list[str] = []  # blackboard entry ids that show it
    fix: str  # what should change
    topic: str = ""  # for missing_knowledge: what to research


class Tried(BaseModel):
    at: str
    agent: str
    change: str
    status: str  # adopted | rejected at <gate> | awaiting a person | researched | queued for research | ...
    detail: str = ""  # the gates and proposal folder, or the wiki page


class Fix(BaseModel):
    at: str
    agent: str
    proposal: str
    change: str
    reverted: bool = False  # rolled back since


class Case(BaseModel):
    id: str
    at: str
    project: str  # the original project folder
    task: str
    problems: list[str] = []
    causes: list[Cause] = []
    tried: list[Tried] = []
    fixes: list[Fix] = []
    fingerprint: str = ""  # set when the case came from an error recurring across runs (factory/events.py)

    @property
    def status(self) -> Literal["open", "fixed"]:
        return "fixed" if any(not f.reverted for f in self.fixes) else "open"

    def blames(self, agent: str) -> bool:
        return any(c.agent == agent for c in self.causes)


def now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


class FailureArchive:
    def __init__(self, root: Path):
        self.root = root

    @classmethod
    def of(cls, cfg: FactoryConfig) -> FailureArchive:
        return cls(cfg.work_dir / "failures")

    def path(self, case_id: str) -> Path:
        return self.root / case_id / "case.json"

    def open_case(self, project_root: Path) -> Case:
        """Snapshot a failed project as it stood before production, and start its case."""
        base = project_root.name
        case_id = base
        for n in range(2, 100):  # a project re-designed twice gets -2, -3, ...
            if not (self.root / case_id).exists():
                break
            case_id = f"{base}-{n}"
        snap = self.root / case_id / "project"
        shutil.copytree(project_root, snap, ignore=lambda d, names: [n for n in names if Path(d) == project_root
                                                                      and n in SNAPSHOT_SKIP])
        board = snap / "blackboard.jsonl"
        lines = board.read_text(encoding="utf-8").splitlines()
        kinds = [json.loads(line)["kind"] for line in lines]
        cut = max((i for i, k in enumerate(kinds) if k == "knowledge"), default=len(lines) - 1) + 1
        board.write_text("".join(f"{line}\n" for line in lines[:cut]), encoding="utf-8")
        task_file = project_root / "task.txt"
        task = task_file.read_text(encoding="utf-8")[:2000] if task_file.exists() else ""
        case = Case(id=case_id, at=now(), project=str(project_root), task=task)
        self.save(case)
        return case

    def open_recurrence_case(self, fingerprint: str, message: str, stage: str, runs: list[str], agent: str) -> Case:
        """Start a case for an error that kept coming back. It has no project snapshot, because it is not one
        project's failure, so nothing is replayed for it: the gates and the next ``recurring`` check judge the change."""
        case = Case(id=f"events-{fingerprint}-{now()}", at=now(), project="", fingerprint=fingerprint,
                    task=f"An error recurred in {len(runs)} runs ({', '.join(runs)}) at stage '{stage}': {message}",
                    problems=[message], causes=[Cause(kind="prompt", agent=agent, fix=f"stop this recurring error: {message}")])
        self.save(case)
        return case

    def replayable(self, case_id: str) -> bool:
        return (self.root / case_id / "project").is_dir()

    def last_case_stamps(self) -> dict[str, str]:
        """For each fingerprint a recurrence case was opened for: when the newest one was opened."""
        return {c.fingerprint: c.at for c in self.cases() if c.fingerprint}

    def save(self, case: Case) -> None:
        self.path(case.id).parent.mkdir(parents=True, exist_ok=True)
        self.path(case.id).write_text(case.model_dump_json(indent=1), encoding="utf-8")

    def get(self, case_id: str) -> Case:
        return Case.model_validate_json(self.path(case_id).read_text(encoding="utf-8"))

    def cases(self) -> list[Case]:
        if not self.root.exists():
            return []
        return sorted((Case.model_validate_json(p.read_text(encoding="utf-8")) for p in self.root.glob("*/case.json")),
                      key=lambda c: c.at)

    def record_fix(self, case_id: str, agent: str, proposal: str, change: str) -> None:
        case = self.get(case_id)
        case.fixes.append(Fix(at=now(), agent=agent, proposal=proposal, change=change))
        self.save(case)

    def revert_fixes(self, agent: str) -> list[str]:
        """Mark the latest fix to ``agent`` in every case as rolled back; those cases are open again."""
        reopened = []
        for case in self.cases():
            live = [f for f in case.fixes if f.agent == agent and not f.reverted]
            if live:
                live[-1].reverted = True
                self.save(case)
                reopened.append(case.id)
        return reopened

    def regressions(self, agent: str, exclude: str = "", limit: int = MAX_REPLAYS - 1) -> list[Case]:
        """Fixed cases that blamed ``agent``, newest first: a new prompt for it must still deliver them."""
        fixed = [c for c in self.cases() if c.status == "fixed" and c.blames(agent) and c.id != exclude
                 and self.replayable(c.id)]
        return fixed[::-1][:limit]

    def past_fixes(self, agent: str | None = None, limit: int = 5) -> str:
        """The fixes that worked before (adopted and not rolled back), as text for an auditor or proposer."""
        rows = [(c, f) for c in self.cases() for f in c.fixes if not f.reverted and (agent is None or f.agent == agent)]
        rows = rows[-limit:]
        if not rows:
            return "- none yet"
        out = []
        for c, f in rows:
            kinds = sorted({k.kind for k in c.causes if k.agent == f.agent}) or ["prompt"]
            out.append(f"- `{f.agent}` ({', '.join(kinds)}) in case {c.id}: {f.change} (proposal {f.proposal})")
        return "\n".join(out)


def replay(cfg: FactoryConfig, archive: FailureArchive, case: Case, agents: list, *, providers: dict | None = None):
    """Build a case again from its snapshot with these agents, on free models. Solved means delivered. A replay never
    starts a re-design of its own."""
    from factory.bench import Outcome
    from factory.llm import LLM, new_run_dir
    from factory.pipeline.blackboard import Project
    from factory.pipeline.knowledge import default_searchers
    from factory.pipeline.run import build
    from factory.pipeline.wiki import Wiki
    from factory.registry import Registry, ScoreBook

    run_dir = new_run_dir(cfg, f"replay-{case.id}")
    root = cfg.work_dir / "replays" / f"{case.id}-{now()}"
    shutil.copytree(archive.root / case.id / "project", root)
    llm = LLM(cfg, run_dir, use_cache=False, **({"providers": providers} if providers else {}))
    registry = Registry(agents, cfg, ScoreBook(run_dir / "scores.json"))
    try:
        result = build(llm, registry, Project.open(root), Wiki(cfg.work_dir / "wiki"),
                       default_searchers(cfg.work_dir / "research-cache"), redesign=False, learn=False)
        solved, why = result.status == "delivered", result.status
    except Exception as e:  # noqa: BLE001 (a crash is a failed replay, not a failed proposal run)
        solved, why = False, f"{type(e).__name__}: {e}"
    spend = llm.ledger.summary()
    return Outcome(task=case.id, mode="aspire", solved=solved, why=why, model_calls=spend["calls"],
                   tokens=spend["tokens"], real_usd=spend["real_usd"], run_dir=str(run_dir))
