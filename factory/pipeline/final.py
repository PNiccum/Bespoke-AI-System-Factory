"""The final evaluator: QA on the finished solution, then delivery as one folder (a zip) the person can rerun.

Code runs the checks first (every Python file compiles, the tests pass in the sandbox). The final evaluator (DeepSeek,
a family that wrote none of the files) then checks each task's acceptance criterion against the files and the test
results. Code approves only when the checks pass and every task is met; a model saying "approve" is not enough.

On approval the deliverable is ``deliverable/<project id>.zip``: the solution, ``task.txt``, ``plan.json``,
``knowledge.json``, the wiki pages it used, and the public part of the blackboard as provenance. Email delivery is
left out on purpose: it sends data outside the system, so it waits for the person's own decision.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from pydantic import BaseModel

from factory.llm import LLM, complete_validated
from factory.pipeline.blackboard import Project
from factory.pipeline.production import OUTPUT, Checks, Production
from factory.registry import Registry

EVALUATOR = "final_evaluator"


class TaskResult(BaseModel):
    id: str
    met: bool
    evidence: str


class QA(BaseModel):
    tasks: list[TaskResult]
    issues: list[str] = []


class FinalResult(BaseModel):
    approved: bool
    tasks: list[TaskResult]
    issues: list[str]
    deliverable: str = ""


QA_PROMPT = """Do final QA on this solution before it goes to the person. For each task, say whether its acceptance \
criterion is met, with the evidence (a file and what it does, or a test that passed). Be strict: a criterion that \
nothing in the files or tests shows is not met. Judge the delivered result, not how it was made: a criterion about \
the order of work (tests written first, or failing in a certain way before the code existed) is met when those tests \
exist and pass now. The plain-code checks are the record of the test run; a test count or run log written in a \
file that disagrees with them is an issue to list, not an unmet task.

## Tasks and acceptance criteria
{tasks}

## Plain-code checks
{checks}

## Files
{files}

Reply with JSON only: {{"tasks": [{{"id": "T1", "met": true or false, "evidence": "<one sentence>"}}], \
"issues": ["<anything the person would trip over>"]}}"""


class FinalEvaluator:
    def __init__(self, llm: LLM, registry: Registry, project: Project, production: Production):
        self.llm, self.project, self.board, self.production = llm, project, project.board, production
        self.agent = registry.agents[EVALUATOR]

    def run(self, shown_files: str) -> FinalResult:
        plan = json.loads((self.project.root / "plan.json").read_text(encoding="utf-8"))
        checks: Checks = self.production.check()
        tasks = "\n".join(f"- {t['id']} {t['title']}: done when {t['acceptance']}" for t in plan["tasks"])
        data = "public" if self.llm.cfg.data_public else "private"
        qa, _, _ = complete_validated(self.llm, self.agent.role, QA_PROMPT.format(
            tasks=tasks, checks=checks.model_dump_json(indent=1), files=shown_files), self.agent.system, QA,
            tag="final-qa", data=data)
        wanted = {t["id"] for t in plan["tasks"]}
        met = {t.id for t in qa.tasks if t.met}
        issues = list(qa.issues)
        if not checks.ok:
            issues.insert(0, "the plain-code checks fail: " + (", ".join(checks.syntax_errors) or
                                                               ", ".join(checks.failed) or "tests fail"))
        if missing := sorted(wanted - met):
            issues.insert(0, f"acceptance not met: {', '.join(missing)}")
        approved = checks.ok and not (wanted - met)
        result = FinalResult(approved=approved, tasks=qa.tasks, issues=issues)
        if approved:
            result.deliverable = str(self.deliver())
        self.board.post("final", EVALUATOR if approved else "code", result.model_dump())
        return result

    def deliver(self) -> Path:
        root = self.project.root
        out = root / "deliverable" / f"{self.project.id}.zip"
        out.parent.mkdir(parents=True, exist_ok=True)
        public = self.board.for_paid(self.board.entries())
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted((root / OUTPUT).rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(root / OUTPUT).as_posix())
            for name in ("task.txt", "plan.json", "knowledge.json"):
                if (root / name).exists():
                    z.write(root / name, f"aspire/{name}")
            for page in self.production.pages():
                if page.license == "public":  # ARC-only pages stay with the project, never in a deliverable
                    z.writestr(f"aspire/wiki/{page.slug}.md", page.markdown())
            z.writestr("aspire/provenance.jsonl", "".join(e.model_dump_json() + "\n" for e in public))
        return out
