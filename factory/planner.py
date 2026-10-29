"""Intake: turn a request written in plain language into a task SIMPLE can run, using a model only when rules can't.

Plain rules route the clear cases (a diff to review, "write tests for ..."). Only an ambiguous request goes to the
planner model, which picks a task type and lays out the steps as a sequence of building blocks from a fixed catalog.
Code then checks the plan: every block must exist, and every block's inputs must come from an earlier block. The model
never decides whether a step needs a model: that is a property of the block, fixed here.

A request no workflow covers is not run. Its checked plan is recorded as a gap (``gaps.jsonl``): the list of things
users asked for that SIMPLE could not yet do, and the starting point for proposing a new agent or workflow.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from factory.llm import LLM, complete_validated
from factory.orchestrator import Plan, Step, Task, plan_for
from factory.registry import Registry
from factory.workflows import TASK_TYPES


@dataclass(frozen=True)
class Block:
    needs_llm: bool
    reason: str
    after: tuple[str, ...] = ()  # blocks whose output this one needs; any one of them is enough


# The building blocks every workflow is made of. A new block means new plain code or a new agent, so the planner can
# name only these; anything else it asks for is a gap.
BLOCKS: dict[str, Block] = {
    "locate_code": Block(False, "names, paths and tracebacks are found by search"),
    "inspect_diff": Block(False, "what a diff touches, and rule findings, are read by plain code"),
    "run_tests": Block(False, "which tests pass is a fact pytest reports"),
    "write_patch": Block(True, "turning a request into a code change needs language understanding",
                         ("locate_code",)),
    "write_tests": Block(True, "writing tests from a description needs language understanding", ("locate_code",)),
    "score_tests": Block(False, "planted bugs and pytest measure how much the tests catch", ("write_tests",)),
    "review_patch": Block(True, "judging scope and hidden defects needs a second reader", ("write_patch",)),
    "review_tests": Block(True, "judging whether tests check the right behaviour needs a second reader",
                          ("write_tests",)),
    "review_pr": Block(True, "commenting on a change needs a second reader", ("inspect_diff",)),
}

TaskType = Literal["bug_fix", "test_gen", "pr_review", "unsupported"]

PLANNER_PROMPT = """Route this request to a workflow and lay out its steps.

## Request
{request}

## Workflows
{workflows}
- unsupported: none of the above fits

## Building blocks (use only these names, in the order they run)
{blocks}

Reply with JSON only:
{{"task_type": one of {types}, "steps": ["<block>", ...], "missing": ["<a block the request needs that is not \
listed>", ...], "failing_tests": ["<pytest node id the request names>", ...], "why": "<one sentence>"}}"""


class Intake(BaseModel):
    task_type: TaskType
    steps: list[str] = []
    missing: list[str] = []  # blocks the model says the request needs and the catalog lacks
    failing_tests: list[str] = []
    why: str = ""


@dataclass
class Routed:
    task: Task | None  # None when no workflow covers the request
    plan: Plan
    routed_by: Literal["rules", "planner"]
    intake: Intake
    problems: list[str] = field(default_factory=list)  # where the model's plan differed from what code accepts


TESTS_RE = re.compile(r"\b(write|add|generate|create)\b.{0,40}\btests?\b", re.I)
NODE_RE = re.compile(r"\b[\w/.-]+\.py::[\w\[\]-]+")


def route_by_rules(request: str, patch: str = "") -> TaskType | None:
    """The clear cases, decided without a model: a diff is a review; asking for tests is test writing."""
    if patch.strip():
        return "pr_review"
    if TESTS_RE.search(request) and not re.search(r"\b(fix|bug|broken|crash|error|wrong)\b", request, re.I):
        return "test_gen"
    return None


def check_steps(steps: list[str]) -> list[str]:
    """Problems with a sequence of blocks: unknown names, and blocks that run before what they need."""
    problems, seen = [], set()
    for name in steps:
        block = BLOCKS.get(name)
        if block is None:
            problems.append(f"unknown block {name!r}")
            continue
        if block.after and not seen & set(block.after):
            problems.append(f"{name!r} needs {' or '.join(block.after)} before it")
        seen.add(name)
    return problems


def as_plan(task_type: str, steps: list[str]) -> Plan:
    """A plan whose model/no-model marks come from the block catalog, never from the model."""
    return Plan(task_type=task_type, steps=[Step(id=f"{i + 1}-{s}", capability=s, needs_llm=BLOCKS[s].needs_llm,
                                                 reason=BLOCKS[s].reason) for i, s in enumerate(steps) if s in BLOCKS])


def planner_prompt(request: str) -> str:
    workflows = "\n".join(f"- {t}: {label} ({', '.join(s.capability for s in plan_for(Task(description='', repo_path='.', task_type=t)).steps)})"
                          for t, label in TASK_TYPES.items())
    blocks = "\n".join(f"- {name}: {'model' if b.needs_llm else 'plain code'}; {b.reason}" for name, b in BLOCKS.items())
    types = json.dumps([*TASK_TYPES, "unsupported"])
    return PLANNER_PROMPT.format(request=request, workflows=workflows, blocks=blocks, types=types)


def route(request: str, repo_path: str, llm: LLM, registry: Registry, *, patch: str = "",
          gaps: Path | None = None) -> Routed:
    """Route a plain-language request: rules first, the planner model only when the rules can't tell."""
    if task_type := route_by_rules(request, patch):
        task = Task(description=request, repo_path=repo_path, task_type=task_type, patch=patch)
        steps = [s.capability for s in plan_for(task).steps]
        return Routed(task, plan_for(task), "rules", Intake(task_type=task_type, steps=steps, why="matched a rule"))

    planner = registry.agents["planner"]
    data = "public" if llm.cfg.data_public else "private"
    intake, _, _ = complete_validated(llm, planner.role, planner_prompt(request), planner.system, Intake,
                                      tag="plan", data=data)
    problems = check_steps(intake.steps) + [f"missing block {m!r}" for m in intake.missing]
    if intake.task_type == "unsupported" or (intake.task_type == "pr_review" and not patch.strip()):
        if intake.task_type == "pr_review":
            problems.append("a review needs the diff; give it with --patch-file")
        plan = as_plan("unsupported", intake.steps)
        if gaps:
            gaps.parent.mkdir(parents=True, exist_ok=True)
            with gaps.open("a") as f:
                f.write(json.dumps({"at": datetime.now(UTC).isoformat(), "request": request, "steps": intake.steps,
                                    "missing": intake.missing, "problems": problems, "why": intake.why}) + "\n")
        return Routed(None, plan, "planner", intake, problems)

    task = Task(description=request, repo_path=repo_path, task_type=intake.task_type, patch=patch,
                failing_tests=[t for t in intake.failing_tests if NODE_RE.fullmatch(t)])
    plan = plan_for(task)  # a known workflow runs its own checked template; the model's steps are only compared
    template = [s.capability for s in plan.steps]
    if intake.steps != template:
        problems.append(f"planner proposed {intake.steps}; the {intake.task_type} workflow runs {template}")
    return Routed(task, plan, "planner", intake, problems)
