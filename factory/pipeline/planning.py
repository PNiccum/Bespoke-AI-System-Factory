"""The planning loop: three model families improve the approved Plan A, offer a Plan B, and vote between them.

1. The plan critic (DeepSeek) critiques Plan A and writes an alternative, Plan B.
2. The plan analyst (gpt-oss) lists pros and cons for each plan.
3. The editor (GLM, the planner role) edits both plans using those pros and cons.
4. The critic and the analyst vote; each vote names a plan and says why. The editor does not vote: it has just
   edited both plans, so it would be judging its own work.

Code does the parts that need no model. Every plan is checked (acceptance checks, unique ids, no task depending on a
later one); a plan that fails gets one chance to be fixed, and an edit that breaks a plan is thrown away, keeping the
plan as it was before the edit. A plan with problems cannot win. Votes are counted in code; a split vote goes to the
plan with fewer problems, then to Plan A, which the person approved. The loop runs once,
plus one repair round when the winner fails its checks.

The critic's weaknesses are blockers on Plan A (approved 2026-11-03 after the primer build, whose plan kept a
weakness the critic had named: gene ESummary gives no RefSeq transcripts). The editor must say, for each one, what it
changed; the critic then re-checks the edited Plan A. An open weakness that would make the delivered result wrong,
incomplete or not runnable counts as a problem, so Plan A cannot win with it; one that would not (wasted work, order,
an unnecessary dependency) is recorded on the blackboard as noted and lets the build go ahead (2026-11-05: csv_profile
stopped at planning over "T2 depends on T1 unnecessarily", a plan that had reached final QA 5/5 the day before).
The repair round gets the blocking weaknesses too, and the critic checks the repair once more.

Every step is a blackboard entry, and the result is ``plan.json``: the refined plan with tasks and subtasks that the
knowledge loop starts from.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel

from factory.events import log_of
from factory.llm import LLM, complete_validated
from factory.pipeline.blackboard import Project
from factory.pipeline.interview import ProjectPlan, check_plan
from factory.pipeline.precedent import offer
from factory.registry import Registry

CRITIC, ANALYST, EDITOR = "plan_critic", "plan_analyst", "plan_editor"
VOTERS = (CRITIC, ANALYST)  # the editor edits and does not vote


class Critique(BaseModel):
    weaknesses: list[str]
    plan_b: ProjectPlan


class Side(BaseModel):
    pros: list[str] = []
    cons: list[str] = []


class Analysis(BaseModel):
    a: Side
    b: Side


class Edited(BaseModel):
    plan_a: ProjectPlan
    plan_b: ProjectPlan
    changes: list[str] = []
    resolved: dict[str, str] = {}  # weakness id -> what changed in Plan A for it


class Open(BaseModel):
    id: str
    why: str
    blocks: bool = True  # left as is, would the delivered result be wrong, incomplete or not runnable?


class Recheck(BaseModel):
    open: list[Open] = []  # weaknesses the edited Plan A still has


class Vote(BaseModel):
    choice: Literal["A", "B"]
    why: str


class Outcome(BaseModel):
    winner: Literal["A", "B"]
    plan: ProjectPlan
    votes: dict[str, str]  # agent -> "A", "B" or "spoiled"
    problems: list[str] = []  # left in the winning plan; empty unless the repair round also failed


SHAPE = ('{"title", "goal", "deliverable", "tasks": [{"id": "T1", "title", "acceptance", "depends_on": [], '
         '"subtasks": [{"id": "T1.1", "title", "acceptance", "depends_on": []}]}], "assumptions": [], '
         '"out_of_scope": []}')

CRITIC_PROMPT = """A person approved this project plan (Plan A). Find its weaknesses: missing steps, tasks that cannot \
be checked, wrong order, wasted work, risks. Then write a better alternative, Plan B, that solves the same request in \
a different way where that helps. Break each task into subtasks a single agent can finish.

## The person's request, as clarified
{task}

## Plan A
{plan_a}

Reply with JSON only: {{"weaknesses": ["..."], "plan_b": {shape}}}"""

ANALYST_PROMPT = """Two plans solve the same request. List the pros and cons of each: how likely it is to work, how \
much it costs to run, how easy the result is to use and to check, and its risks.

## The request
{task}

## Plan A
{plan_a}

## Plan B
{plan_b}

Reply with JSON only: {{"a": {{"pros": [...], "cons": [...]}}, "b": {{"pros": [...], "cons": [...]}}}}"""

EDITOR_PROMPT = """Edit both plans to fix the cons listed for each while keeping their pros. Keep each plan's own \
approach: do not merge them. Every task and subtask needs an acceptance check anyone can run.

## The request
{task}

## Plan A
{plan_a}

## Plan B
{plan_b}

## Pros and cons
{analysis}

## Weaknesses the critic found in Plan A (each must be fixed in Plan A)
{weaknesses}

Reply with JSON only: {{"plan_a": {shape}, "plan_b": {shape}, "changes": ["<one line per change>"], "resolved": \
{{"W1": "<what you changed in Plan A for it>"}}}}"""

RECHECK_PROMPT = """You found these weaknesses in Plan A. For each one, decide whether the plan below \
still has it. {which} A claimed change that the plan text does not show does not count.

## The request
{task}

## Weaknesses
{weaknesses}

## What the editor says it changed
{resolved}

## {title}
{plan_a}

For each open weakness, "blocks" is true when, left as is, it would make the delivered result wrong, incomplete or \
not runnable, and false when it only costs effort, order or tidiness (wasted work, an unnecessary dependency, style).

Reply with JSON only: {{"open": [{{"id": "W1", "why": "<what is still wrong>", "blocks": true}}]}} (empty when all \
are fixed)"""

# Plan B was never put through the critic's review, so without a recheck of its own it could win only because Plan
# A's open weaknesses spoiled every vote for A (the 2026-11-04 primer run), even when B has the same weaknesses.
WHICH = {"A": ("It is Plan A after editing.", "Edited Plan A"),
         "B": ("It is Plan B, the alternative: it has a weakness unless its own text deals with it.", "Plan B")}

VOTE_PROMPT = """Choose the plan the team should carry out for this request: the one most likely to give the person \
a working result they can run again, at the lowest cost.

## The request
{task}

## Plan A
{plan_a}

## Plan B
{plan_b}

## Pros and cons
{analysis}

Reply with JSON only: {{"choice": "A" or "B", "why": "<one sentence>"}}"""

FIX_PROMPT = """{prompt}

Your plan has these problems: {problems}. Reply again with them fixed, in the same JSON shape."""


def show(plan: ProjectPlan) -> str:
    return json.dumps(plan.model_dump(), indent=1)


class PlanningLoop:
    def __init__(self, llm: LLM, registry: Registry, project: Project):
        self.llm, self.project, self.board = llm, project, project.board
        self.agents = {name: registry.agents[name] for name in (CRITIC, ANALYST, EDITOR)}
        self.data = "public" if llm.cfg.data_public else "private"
        log_of(llm, project)

    def _call(self, agent: str, prompt: str, model_cls, tag: str):
        spec = self.agents[agent]
        out, _, _ = complete_validated(self.llm, spec.role, prompt, spec.system, model_cls, tag=tag, data=self.data)
        return out

    def recheck(self, task: str, weaknesses: dict[str, str], resolved: dict[str, str], plan: ProjectPlan,
                ref: str, label: str = "A", tag: str = "plan-recheck") -> list[str]:
        """The weaknesses the critic still finds in ``plan``, as problems ("W2 still open: ..."). No weaknesses, no
        call. A recheck with no valid reply counts every weakness as open: an unverified fix is not a fix."""
        if not weaknesses:
            return []
        try:
            found = self._call(CRITIC, RECHECK_PROMPT.format(
                task=task, weaknesses="\n".join(f"- {k}: {w}" for k, w in weaknesses.items()),
                resolved="\n".join(f"- {k}: {v}" for k, v in resolved.items()) or "(nothing said)",
                which=WHICH[label][0], title=WHICH[label][1], plan_a=show(plan)), Recheck, tag)
        except RuntimeError:
            found = Recheck(open=[Open(id=k, why="the re-check gave no valid reply") for k in weaknesses])
        left = _open(found, weaknesses)
        self.board.post("recheck", CRITIC, {"plan": label, "open": [o.model_dump() for o in found.open],
                                            "problems": left, "noted": _noted(found, weaknesses)}, refs=[ref])
        return left

    def run(self) -> Outcome:
        with self.llm.stage("planning"):  # token ceiling: budget.stage_max_tokens
            return self._run()

    def _run(self) -> Outcome:
        approval = self.board.latest("approval")
        if approval is None:
            raise RuntimeError("the planning loop starts from an approved Plan A; approve one first")
        if self.board.latest("decision"):
            raise RuntimeError("this project's planning loop already ran")
        plan_a = ProjectPlan.model_validate(self.board.get(approval.data["plan"]).data["plan"])
        task = (self.project.root / "task.txt").read_text(encoding="utf-8")

        # 1. Critique and Plan B, with one fix when Plan B fails its checks
        earlier = offer(self.project, task)  # the Interviewer's lookup if it ran, else made now (same blackboard entry)
        prompt = CRITIC_PROMPT.format(task=task + (f"\n\n{earlier}" if earlier else ""), plan_a=show(plan_a),
                                      shape=SHAPE)
        critique = self._call(CRITIC, prompt, Critique, "plan-critique")
        if problems := check_plan(critique.plan_b):
            critique = self._call(CRITIC, FIX_PROMPT.format(prompt=prompt, problems="; ".join(problems)), Critique,
                                  "plan-critique-fix")
        plan_b = critique.plan_b
        c = self.board.post("critique", CRITIC, {"weaknesses": critique.weaknesses, "plan_b": plan_b.model_dump(),
                                                 "problems": check_plan(plan_b)}, refs=[approval.id])

        # 2. Pros and cons
        analysis = self._call(ANALYST, ANALYST_PROMPT.format(task=task, plan_a=show(plan_a), plan_b=show(plan_b)),
                              Analysis, "plan-analysis")
        an = self.board.post("analysis", ANALYST, analysis.model_dump(), refs=[c.id])

        # 3. Edits; an edit that breaks a plan's checks is dropped
        weaknesses = {f"W{i}": w for i, w in enumerate(critique.weaknesses, 1)}
        listed = "\n".join(f"- {k}: {w}" for k, w in weaknesses.items()) or "(none)"
        edited = self._call(EDITOR, EDITOR_PROMPT.format(task=task, plan_a=show(plan_a), plan_b=show(plan_b),
                                                         analysis=json.dumps(analysis.model_dump()), shape=SHAPE,
                                                         weaknesses=listed),
                            Edited, "plan-edit")
        kept = {}
        for label, before, after in (("A", plan_a, edited.plan_a), ("B", plan_b, edited.plan_b)):
            kept[label] = after if not check_plan(after) or check_plan(before) else before
        plan_a, plan_b = kept["A"], kept["B"]
        ed = self.board.post("edit", EDITOR, {"plan_a": plan_a.model_dump(), "plan_b": plan_b.model_dump(),
                                              "changes": edited.changes, "resolved": edited.resolved,
                                              "dropped": [k for k, p in (("A", edited.plan_a), ("B", edited.plan_b))
                                                          if kept[k] is not p]}, refs=[an.id])

        # 4. The critic re-checks its weaknesses on both plans; open ones are problems
        # (the two re-checks are independent, so they run side by side)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="recheck") as pool:
            fa = pool.submit(self.recheck, task, weaknesses, edited.resolved, plan_a, ed.id)
            fb = pool.submit(self.recheck, task, weaknesses, {}, plan_b, ed.id, label="B", tag="plan-recheck-b")
            open_a, open_b = fa.result(), fb.result()

        # 5. Votes, counted in code
        plans = {"A": plan_a, "B": plan_b}
        problems = {k: check_plan(p) for k, p in plans.items()}
        problems["A"] += open_a
        problems["B"] += open_b
        vote_prompt = VOTE_PROMPT.format(task=task, plan_a=show(plan_a), plan_b=show(plan_b),
                                         analysis=json.dumps(analysis.model_dump()))
        votes = {}

        def cast(agent: str):
            try:
                return self._call(agent, vote_prompt, Vote, f"plan-vote-{agent}")
            except RuntimeError:  # no valid vote after the retries
                return None
        with ThreadPoolExecutor(max_workers=len(VOTERS), thread_name_prefix="vote") as pool:
            cast_votes = list(pool.map(cast, VOTERS))  # the voters do not see each other, so they run side by side
        for agent, v in zip(VOTERS, cast_votes, strict=True):  # posted in a fixed order
            if v is None:
                votes[agent] = "spoiled"
                self.board.post("vote", agent, {"choice": None, "why": "no valid vote", "counted": False},
                                refs=[ed.id])
                continue
            votes[agent] = "spoiled" if problems[v.choice] else v.choice  # a plan with problems cannot win
            self.board.post("vote", agent, {"choice": v.choice, "why": v.why, "counted": votes[agent] != "spoiled"},
                            refs=[ed.id])
        winner = decide(votes, problems)

        # 6. One repair round when the winner still fails its checks or keeps an open weakness
        plan = plans[winner]
        if left := problems[winner]:
            fixed = self._call(EDITOR, FIX_PROMPT.format(prompt=f"Fix this plan.\n\n{show(plan)}\n\nReply with JSON "
                                                         f"only: {SHAPE}", problems="; ".join(left)), ProjectPlan,
                               "plan-repair")
            if not check_plan(fixed):
                still = self.recheck(task, {k: w for k, w in weaknesses.items() if any(p.startswith(f"{k} ")
                                                                                         for p in left)},
                                     {}, fixed, ed.id, label=winner, tag="plan-recheck-repair")
                plan, left = fixed, still
        outcome = Outcome(winner=winner, plan=plan, votes=votes, problems=left)
        d = self.board.post("decision", "code", {"winner": winner, "votes": votes, "problems": left},
                            refs=[e.id for e in self.board.entries("vote")] or [ed.id])
        self.project.write_file("plan.json", json.dumps(plan.model_dump(), indent=1), "code", refs=[d.id])
        return outcome


def _open(found: Recheck, weaknesses: dict[str, str]) -> list[str]:
    """The open weaknesses that block the plan, as problems."""
    return [f"{o.id} still open: {weaknesses[o.id]} ({o.why})" for o in found.open if o.id in weaknesses and o.blocks]


def _noted(found: Recheck, weaknesses: dict[str, str]) -> list[str]:
    """The open weaknesses that do not change the result: recorded, not held against the plan."""
    return [f"{o.id} noted: {weaknesses[o.id]} ({o.why})" for o in found.open if o.id in weaknesses and not o.blocks]


def decide(votes: dict[str, str], problems: dict[str, list[str]]) -> Literal["A", "B"]:
    """Majority of counted votes; a tie goes to the plan with fewer problems, then to Plan A."""
    a, b = (sum(v == k for v in votes.values()) for k in ("A", "B"))
    if a != b:
        return "A" if a > b else "B"
    return "B" if len(problems["B"]) < len(problems["A"]) else "A"
