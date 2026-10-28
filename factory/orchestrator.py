"""The orchestrator: plan a task, award each step by contract net, run it, check it, repair it.

A plan is a short graph of steps, each naming the capability it needs and whether it needs a model. For task types
the factory knows (phase 1: ``bug_fix``), the plan comes from a template, so planning costs nothing; a model-written
plan is for task types no template covers (phase 2).

For a bug fix the run is:

1. ``locate_code``: the code_navigator ranks the files the issue is about (plain code).
2. ``run_tests`` (baseline): find which tests fail before any change. Unless the task names them, these are the
   targets the fix must turn green.
3. ``write_patch``: the coder writes a unified diff (a model, ARC first).
4. ``run_tests`` (check): the patch must apply, every target must pass, and no test that passed before may fail.
5. ``review_patch``: a reviewer from a different model family reads the patch and the test report.

A failed check or a rejected review goes back to step 3 with the reason, up to ``max_attempts``. The last attempt
escalates the coder to its paid model, which waits on the spend gate. Every step is announced to ``on_event`` so the
Streamlit app can show what each agent is doing, and recorded in the run's provenance log.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from factory.llm import LLM, BudgetExceeded, TruncatedError, complete_validated
from factory.llm.gate import SpendDenied
from factory.records import ProvenanceEvent, ProvenanceLog
from factory.registry import Announcement, Bid, Registry
from factory.tools.repo import LocateIn, Locations
from factory.tools.sandbox import RunTestsIn, TestReport


class Task(BaseModel):
    description: str  # the issue or request, as written
    repo_path: str
    task_type: str = "bug_fix"
    domain: str = "software"
    failing_tests: list[str] = []  # tests the fix must turn green; the baseline's failures when empty


class Step(BaseModel):
    id: str
    capability: str
    needs_llm: bool
    reason: str  # why this step does or does not need a model


class Plan(BaseModel):
    task_type: str
    steps: list[Step]


TEMPLATES: dict[str, list[Step]] = {
    "bug_fix": [
        Step(id="locate", capability="locate_code", needs_llm=False,
             reason="names, paths and tracebacks in the issue are found by search"),
        Step(id="baseline", capability="run_tests", needs_llm=False,
             reason="which tests fail is a fact pytest reports"),
        Step(id="patch", capability="write_patch", needs_llm=True,
             reason="turning a bug report into a code change needs language understanding"),
        Step(id="check", capability="run_tests", needs_llm=False,
             reason="whether the patch works is a fact pytest reports"),
        Step(id="review", capability="review_patch", needs_llm=True,
             reason="judging scope and hidden defects needs a second reader"),
    ],
}


def plan_for(task: Task) -> Plan:
    if task.task_type not in TEMPLATES:
        raise NotImplementedError(f"no template for task type {task.task_type!r} yet (model-written plans: phase 2)")
    return Plan(task_type=task.task_type, steps=TEMPLATES[task.task_type])


class Defect(BaseModel):
    line: str = ""  # the diff line it is on, quoted
    problem: str


class Review(BaseModel):
    verdict: Literal["accept", "reject"]
    defects: list[Defect] = []


@dataclass
class Event:
    step: str
    agent: str
    status: Literal["started", "done", "failed", "waiting"]
    detail: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class RunResult:
    status: Literal["fixed", "unproven", "failed", "stopped"]
    plan: Plan
    patch: str | None = None
    check: TestReport | None = None
    review: Review | None = None
    attempts: int = 0
    stop_reason: str = ""
    events: list[Event] = field(default_factory=list)


CODER_PROMPT = """Fix this issue.

## Issue
{issue}

## Tests failing now
{failing}

## Where the issue points (most likely first)
{locations}

## Files
{files}
{feedback}
Reply with one unified diff (--- a/path, +++ b/path, @@ hunks) inside a ```diff block and nothing else."""

REVIEW_PROMPT = """Review this patch for the issue below. You did not write it.

## Issue
{issue}

## Patch
```diff
{patch}
```

## Test results with the patch
passed: {passed}
failed: {failed}

Reply with JSON only: {{"verdict": "accept" or "reject", "defects": [{{"line": "<diff line>", "problem": "<what is wrong>"}}]}}.
Reject only for a concrete defect: a wrong result, a missed case the issue names, or a change outside the issue's scope."""


def extract_diff(text: str) -> str:
    """The unified diff in a model's reply: a ```diff block if there is one, else everything from the first ---."""
    fenced = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", text, re.DOTALL)
    body = fenced.group(1) if fenced else text[text.find("---"):] if "---" in text else ""
    return body.strip("\n") + "\n" if body.strip() else ""


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class Orchestrator:
    def __init__(self, llm: LLM, registry: Registry, *, max_attempts: int = 3,
                 on_event: Callable[[Event], None] | None = None, max_file_lines: int = 400):
        self.llm = llm
        self.registry = registry
        self.max_attempts = max_attempts
        self.on_event = on_event or (lambda e: None)
        self.max_file_lines = max_file_lines
        self.provenance = ProvenanceLog(llm.run_dir)
        self.data = "public" if llm.cfg.data_public else "private"

    # -- plumbing ------------------------------------------------------------------------------------------------

    def emit(self, result: RunResult, step: str, agent: str, status, detail: str = "", **data) -> None:
        event = Event(step, agent, status, detail, data)
        result.events.append(event)
        self.on_event(event)

    def award(self, task: Task, step: Step) -> Bid:
        bid = self.registry.award(Announcement(capability=step.capability, domain=task.domain,
                                               task_type=task.task_type))
        if bid is None:
            raise LookupError(f"no registered agent can {step.capability!r}")
        if bid.agent.kind == "llm" and not step.needs_llm:
            raise ValueError(f"step {step.id!r} needs no model, but only {bid.agent.name!r} (a model) bid for it")
        return bid

    def run_tests(self, bid: Bid, task: Task, patch: str | None, tests: list[str]) -> TestReport:
        report = bid.agent.function()(RunTestsIn(repo_path=task.repo_path, patch=patch, tests=tests))
        self.provenance.record(ProvenanceEvent(kind="tool_call", actor=bid.agent.name,
                                               inputs=[sha12(patch)] if patch else [],
                                               detail={"tests": tests, "passed": len(report.passed),
                                                       "failed": report.failed, "errors": report.errors}))
        return report

    # -- the run -------------------------------------------------------------------------------------------------

    def run(self, task: Task) -> RunResult:
        plan = plan_for(task)
        result = RunResult(status="failed", plan=plan)
        steps = {s.id: s for s in plan.steps}
        bids = {s.id: self.award(task, s) for s in plan.steps}
        self.provenance.record(ProvenanceEvent(kind="source", actor="intake", outputs=[sha12(task.description)],
                                               detail={"task": task.model_dump(), "plan": plan.model_dump(),
                                                       "awards": {k: b.agent.name for k, b in bids.items()}}))
        for sid, bid in bids.items():
            self.emit(result, sid, bid.agent.name, "started" if sid == "locate" else "waiting",
                      f"awarded with bid {bid.score}", parts=bid.parts)

        loc_agent = bids["locate"].agent
        locations: Locations = loc_agent.function()(LocateIn(repo_path=task.repo_path, issue_text=task.description))
        self.emit(result, "locate", loc_agent.name, "done",
                  f"{len(locations.hits)} candidate files", hits=[h.model_dump() for h in locations.hits[:5]])

        self.emit(result, "baseline", bids["baseline"].agent.name, "started", "running the tests before any change")
        base = self.run_tests(bids["baseline"], task, None, [])
        broken_before = set(base.failed + base.errors)
        failing = task.failing_tests or sorted(broken_before)
        self.emit(result, "baseline", bids["baseline"].agent.name, "done",
                  f"{len(broken_before)} failing, {len(base.passed)} passing", failing=failing)

        feedback = ""
        for attempt in range(1, self.max_attempts + 1):
            result.attempts = attempt
            escalate = attempt == self.max_attempts and attempt > 1
            coder = bids["patch"].agent
            self.emit(result, "patch", coder.name, "started",
                      f"attempt {attempt}" + (" on the paid model" if escalate else ""))
            try:
                reply = self.llm.complete(coder.role, self.coder_prompt(task, locations, failing, feedback),
                                          system=coder.prompt, tag=f"patch-{attempt}", data=self.data,
                                          escalate=escalate)
            except (SpendDenied, BudgetExceeded) as e:
                return self.stop(result, "patch", coder.name, str(e))
            except TruncatedError as e:
                feedback = f"\n## Your previous reply was cut off\n{e}. Keep the diff short.\n"
                self.emit(result, "patch", coder.name, "failed", str(e))
                continue
            patch = extract_diff(reply.text)
            self.provenance.record(ProvenanceEvent(kind="model_call", actor=coder.name, outputs=[sha12(patch)],
                                                   detail={"model": reply.model, "tag": f"patch-{attempt}",
                                                           "real_usd": reply.real_usd}))
            if not patch:
                feedback = "\n## Your previous reply had no diff\nReply with a ```diff block.\n"
                self.emit(result, "patch", coder.name, "failed", "no diff in the reply")
                continue
            self.emit(result, "patch", coder.name, "done", f"{patch.count(chr(10))} diff lines", model=reply.model)

            checker = bids["check"]
            self.emit(result, "check", checker.agent.name, "started", "running the tests with the patch")
            check = self.run_tests(checker, task, patch, [])
            result.patch, result.check = patch, check
            broken_after = set(check.failed + check.errors)
            still = sorted(set(failing) & broken_after)
            new = sorted(broken_after - broken_before)
            if check.patch_applied is False or check.timed_out or still or new:
                why = ("the patch did not apply:\n" + check.output_tail if check.patch_applied is False
                       else "the tests timed out" if check.timed_out
                       else f"still failing: {still}; newly failing: {new}\n{check.output_tail}")
                self.emit(result, "check", checker.agent.name, "failed", why[:300])
                feedback = f"\n## Your previous patch failed\n```diff\n{patch}```\n{why[-3000:]}\n"
                continue
            self.emit(result, "check", checker.agent.name, "done", f"{len(check.passed)} tests pass")

            reviewer = bids["review"].agent
            self.emit(result, "review", reviewer.name, "started", "reading the patch")
            try:
                review, _, _ = complete_validated(
                    self.llm, reviewer.role, REVIEW_PROMPT.format(issue=task.description, patch=patch,
                                                                  passed=len(check.passed), failed=check.failed),
                    reviewer.prompt, Review, tag=f"review-{attempt}", data=self.data)
            except (SpendDenied, BudgetExceeded) as e:
                return self.stop(result, "review", reviewer.name, str(e))
            except RuntimeError as e:  # no parseable verdict after retries: a person has to look
                return self.stop(result, "review", reviewer.name, str(e)[:500])
            result.review = review
            self.provenance.record(ProvenanceEvent(kind="model_call", actor=reviewer.name, inputs=[sha12(patch)],
                                                   detail={"verdict": review.verdict,
                                                           "defects": [d.model_dump() for d in review.defects]}))
            if review.verdict == "reject":
                problems = "\n".join(f"- {d.line}: {d.problem}" for d in review.defects) or "- (no detail given)"
                self.emit(result, "review", reviewer.name, "failed", problems[:300])
                feedback = f"\n## Your previous patch was rejected in review\n```diff\n{patch}```\n{problems}\n"
                continue
            self.emit(result, "review", reviewer.name, "done", "accepted")
            # Proven only when there were failing tests to turn green; otherwise accepted but unproven.
            result.status = "fixed" if failing else "unproven"
            break
        else:
            result.stop_reason = f"no accepted patch after {self.max_attempts} attempts"

        for sid, bid in bids.items():
            self.registry.scores.record(bid.agent.name, task.task_type, passed=result.status == "fixed")
        self.llm.write_summary()
        return result

    def stop(self, result: RunResult, step: str, agent: str, reason: str) -> RunResult:
        result.status, result.stop_reason = "stopped", reason
        self.emit(result, step, agent, "failed", reason)
        self.provenance.record(ProvenanceEvent(kind="approval", actor="spend_gate", detail={"stopped": reason}))
        self.llm.write_summary()
        return result

    def coder_prompt(self, task: Task, locations: Locations, failing: list[str], feedback: str) -> str:
        root = Path(task.repo_path)
        shown = [h for h in locations.hits if not h.is_test][:3] + [h for h in locations.hits if h.is_test][:1]
        files = []
        for h in shown:
            lines = (root / h.path).read_text(errors="replace").splitlines()[:self.max_file_lines]
            files.append(f"### {h.path}\n```python\n" + "\n".join(lines) + "\n```")
        loc = "\n".join(f"- {h.path}:{h.line} ({'; '.join(h.why)})" for h in locations.hits[:8]) or "- none found"
        return CODER_PROMPT.format(issue=task.description, failing="\n".join(failing) or "none known",
                                   locations=loc, files="\n\n".join(files) or "(none)", feedback=feedback)
