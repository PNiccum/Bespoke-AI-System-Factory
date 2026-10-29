"""The re-design loop: when a project fails even after research, find out why and propose a fix to ASPIRE itself.

1. Auditor 1 (DeepSeek) reads the project's record and names its successes, failures and shortcomings, what caused
   them, and how the system could be changed.
2. Auditor 2 (gpt-oss) evaluates those proposals and returns suggestions; auditor 1 answers them, and the two settle
   on the problems and the fixes worth making.
3. A summarizer (GLM) writes up the project, its transcript and the auditors' conclusions. Only public blackboard
   entries go in: ARC-only text never reaches a paid model.
4. Claude and ChatGPT (the paid escalation models already in the config) each propose an improvement to the system
   and then answer each other, until both agree or the debate's dollar cap or round limit is reached. Every paid call
   goes through the spend gate, so a person approves it first.
5. Fixes are sorted in plain code. A prompt change to a model agent that is not a judge goes through
   ``factory.extend`` (scope, benchmark, other-family panel) and is applied automatically if it passes every gate:
   Parker decided (2026-10-29) that prompt-only changes need no approval. Anything structural (code, tools, loops,
   roles, a judging agent's prompt) waits for the person, as does a prompt change that fails a gate.
6. The report (``redesign/report.md``) says what was applied, how to undo it, and what waits for approval.

Stop rules: one auditor exchange, DEBATE_ROUNDS rounds, DEBATE_CAP_USD real dollars, MAX_PROMPT_CHANGES prompt
changes per re-design.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from factory.llm import LLM, complete_validated
from factory.llm.cost import BudgetExceeded
from factory.llm.gate import SpendDenied
from factory.pipeline.blackboard import Project
from factory.registry import Registry

AUDITOR_1, AUDITOR_2, SUMMARIZER = "auditor_1", "auditor_2", "summarizer"
DEBATE_ROUNDS, DEBATE_CAP_USD = 2, 0.50
DEBATERS = {"claude": "reviewer", "chatgpt": "coder"}  # roles whose paid escalation step is Claude and ChatGPT
MAX_PROMPT_CHANGES = 2  # each runs the extend gates, which cost free model runs


class Audit(BaseModel):
    successes: list[str] = []
    failures: list[str] = []
    causes: list[str] = []
    proposals: list[str] = []


class Suggestions(BaseModel):
    suggestions: list[str]


class PromptChange(BaseModel):
    agent: str  # a model agent in the registry
    change: str  # what its instructions should do differently


class Settled(BaseModel):
    problems: list[str]
    fixes: list[str]  # structural: code, tools, loops, roles
    prompt_changes: list[PromptChange] = []


class Position(BaseModel):
    proposal: str
    agrees_with_other: bool = False
    kind: Literal["prompt", "structural"] = "structural"
    agent: str = ""  # for a prompt change, the agent whose prompt changes


class Applied(BaseModel):
    agent: str
    change: str
    status: str  # adopted | rejected at <gate> | awaiting a person | not tried: <why>
    detail: str = ""  # gates, or the proposal folder


class RedesignResult(BaseModel):
    problems: list[str]
    fixes: list[str]
    debate: list[dict]
    agreed: bool
    stopped: str  # why the debate ended
    report: str  # path
    prompt_changes: list[Applied] = []


AUDIT_PROMPT = """ASPIRE, a system that builds solutions with teams of agents, failed this project even after \
researching what it lacked. Read its record and say what went well, what failed, what caused each failure (a prompt, \
a missing tool or knowledge source, a loop rule, a model's limits), and how ASPIRE itself could be changed so the next \
project like this succeeds.

## Project record
{record}

Reply with JSON only: {{"successes": [...], "failures": [...], "causes": [...], "proposals": [...]}}"""

SUGGEST_PROMPT = """Another auditor reviewed a failed project. Evaluate its proposals for changing the system: which \
would work, which would not and why, and what it missed.

## Its audit
{audit}

Reply with JSON only: {{"suggestions": ["..."]}}"""

SETTLE_PROMPT = """You audited a failed project; a second auditor answered your proposals. Weigh their suggestions \
and settle the list of problems and the fixes worth making to the system. Sort the fixes: a change to how one of the \
model agents below is instructed is a prompt change; anything else (code, tools, data sources, loop rules, roles) is \
structural.

## Your audit
{audit}

## Their suggestions
{suggestions}

## Model agents
{agents}

Reply with JSON only: {{"problems": ["..."], "fixes": ["<structural fix>"], "prompt_changes": [{{"agent": "<name \
above>", "change": "<what its instructions should do differently>"}}]}}"""

SUMMARY_PROMPT = """Summarize this failed project for two outside experts: what was asked, what the agents did at each \
stage, where it went wrong, and what the auditors concluded. Plain prose, under 600 words.

## Project record
{record}

## Auditors' conclusions
{settled}"""

DEBATE_PROMPT = """You are one of two outside experts advising on ASPIRE, a system that builds solutions with teams of \
agents on free models and calls paid models only where needed. A project failed; read the summary and propose the \
single most valuable change to the system.
{other}
## Summary
{summary}

Reply with JSON only: {{"proposal": "<the change, concretely: which component, what it does differently, how to test \
it>", "agrees_with_other": true or false, "kind": "prompt" if it only changes one agent's instructions else \
"structural", "agent": "<that agent's name, for a prompt change>"}}"""


def record(project: Project, limit: int = 30_000) -> str:
    """The public blackboard as text, newest last, trimmed from the front to the limit."""
    lines = [f"{e.id} {e.author}: {json.dumps(e.data)[:1500]}" for e in project.board.for_paid(project.board.entries())]
    text = "\n".join(lines)
    return text[-limit:]


Improver = Callable[[str, str], tuple[str, str]]  # (agent, change) -> (status, detail)


class Redesign:
    def __init__(self, llm: LLM, registry: Registry, project: Project, improver: Improver | None = None):
        self.llm, self.project, self.board, self.registry = llm, project, project.board, registry
        self.agents = {n: registry.agents[n] for n in (AUDITOR_1, AUDITOR_2, SUMMARIZER)}
        self.data = "public" if llm.cfg.data_public else "private"
        self.improver = improver or self.extend

    def extend(self, agent: str, change: str) -> tuple[str, str]:
        """The real improver: one proposal through factory.extend's gates, adopted if it passes them all."""
        from factory import extend
        result = extend.propose(self.llm.cfg, agent, approver=self.llm.gate.approver, providers=self.llm.providers,
                                guidance=change,
                                adopt_if_passed=True)
        return result.status, "; ".join(f"{g.name}: {g.detail}" for g in result.gates) + f" ({result.folder})"

    def _call(self, agent: str, prompt: str, model_cls, tag: str):
        spec = self.agents[agent]
        out, _, _ = complete_validated(self.llm, spec.role, prompt, spec.system, model_cls, tag=tag, data=self.data)
        return out

    def run(self) -> RedesignResult:
        rec = record(self.project)
        audit = self._call(AUDITOR_1, AUDIT_PROMPT.format(record=rec), Audit, "audit")
        a = self.board.post("audit", AUDITOR_1, audit.model_dump())
        sugg = self._call(AUDITOR_2, SUGGEST_PROMPT.format(audit=audit.model_dump_json(indent=1)), Suggestions,
                          "audit-suggest")
        s = self.board.post("audit", AUDITOR_2, sugg.model_dump(), refs=[a.id])
        model_agents = "\n".join(f"- {a.name}: {a.description}" for a in self.registry.agents.values()
                                 if a.kind == "llm")
        settled = self._call(AUDITOR_1, SETTLE_PROMPT.format(audit=audit.model_dump_json(indent=1),
                                                             suggestions=json.dumps(sugg.suggestions),
                                                             agents=model_agents), Settled, "audit-settle")
        st = self.board.post("audit", AUDITOR_1, settled.model_dump(), refs=[s.id])
        spec = self.agents[SUMMARIZER]
        summary = self.llm.complete(spec.role, SUMMARY_PROMPT.format(record=rec, settled=settled.model_dump_json()),
                                    system=spec.system, tag="redesign-summary", data=self.data).text
        self.board.post("note", SUMMARIZER, {"summary": summary}, refs=[st.id])
        debate, agreed, stopped = self.debate(summary)
        changes = list(settled.prompt_changes)
        if agreed and debate[-1]["kind"] == "prompt":  # the experts' agreed change, when it is a prompt change
            changes.append(PromptChange(agent=debate[-1]["agent"], change=debate[-1]["proposal"]))
        elif agreed:
            settled.fixes.append(debate[-1]["proposal"])
        applied = self.apply(changes, settled)
        report = self.write_report(settled, summary, debate, agreed, stopped, applied)
        self.board.post("redesign", "code", {"problems": settled.problems, "fixes": settled.fixes, "agreed": agreed,
                                             "stopped": stopped, "report": str(report),
                                             "prompt_changes": [a.model_dump() for a in applied]}, refs=[st.id])
        return RedesignResult(problems=settled.problems, fixes=settled.fixes, debate=debate, agreed=agreed,
                              stopped=stopped, report=str(report), prompt_changes=applied)

    def apply(self, changes: list[PromptChange], settled: Settled) -> list[Applied]:
        """Plain code decides which prompt changes may be tried; the extend gates decide which are adopted."""
        from factory.extend import JUDGES
        out, seen = [], set()
        for c in changes:
            spec = self.registry.agents.get(c.agent)
            if spec is None or spec.kind != "llm":
                settled.fixes.append(f"{c.agent}: {c.change}")  # not a model agent: treat as structural
                continue
            if c.agent in seen:
                continue
            seen.add(c.agent)
            if c.agent in JUDGES:
                out.append(Applied(agent=c.agent, change=c.change, status="awaiting a person",
                                   detail="it judges other agents' work, so changing it is structural"))
            elif len([a for a in out if not a.status.startswith("not tried")]) >= MAX_PROMPT_CHANGES:
                out.append(Applied(agent=c.agent, change=c.change,
                                   status=f"not tried: limit of {MAX_PROMPT_CHANGES} per re-design"))
            else:
                status, detail = self.improver(c.agent, c.change)
                out.append(Applied(agent=c.agent, change=c.change, status=status, detail=detail))
            self.board.post("redesign", "code", {"prompt_change": out[-1].model_dump()})
        return out

    def debate(self, summary: str) -> tuple[list[dict], bool, str]:
        """Claude and ChatGPT answer each other until both agree, the cap is reached, or a person declines."""
        turns: list[dict] = []
        last: dict[str, str] = {}
        start = self.llm.ledger.summary()["real_usd"]
        for rnd in range(1, DEBATE_ROUNDS + 1):
            agrees = {}
            for name, role in DEBATERS.items():
                if self.llm.ledger.summary()["real_usd"] - start >= DEBATE_CAP_USD:
                    return turns, False, f"the debate's ${DEBATE_CAP_USD:.2f} cap was reached"
                other_name = next(n for n in DEBATERS if n != name)
                other = (f"\nThe other expert proposed:\n{last[other_name]}\nSay whether you agree, and refine or "
                         "replace it.\n") if other_name in last else ""
                try:
                    c = self.llm.complete(role, DEBATE_PROMPT.format(other=other, summary=summary),
                                          tag=f"debate-{name}-{rnd}", data="public", escalate=True)
                    pos = Position.model_validate(c.json())
                except (SpendDenied, BudgetExceeded) as e:
                    return turns, False, f"paid call not made: {e}"
                except ValueError as e:
                    return turns, False, f"{name} gave no usable answer: {e}"
                last[name], agrees[name] = pos.proposal, pos.agrees_with_other
                turns.append({"round": rnd, "expert": name, "model": c.model, **pos.model_dump()})
                self.board.post("debate", name, turns[-1])
            if rnd > 1 and all(agrees.values()):
                return turns, True, "both experts agree"
        return turns, False, f"{DEBATE_ROUNDS} rounds without agreement"

    def write_report(self, settled: Settled, summary: str, debate: list[dict], agreed: bool, stopped: str,
                     applied: list[Applied] = ()):
        adopted = [a for a in applied if a.status == "adopted"]
        lines = [f"# Re-design report: {self.project.id}", "",
                 (f"{len(adopted)} prompt change(s) passed every gate and were applied. " if adopted else
                  "No change was applied. ") + "Structural fixes wait for your approval.", "",
                 "## What went wrong", *[f"- {p}" for p in settled.problems], "",
                 "## Prompt changes", *([f"- `{a.agent}`: {a.change} **{a.status}**" + (f" ({a.detail})" if a.detail
                                         else "") for a in applied] or ["- None proposed."]), "",
                 "## Structural fixes waiting for your approval", *([f"- {f}" for f in settled.fixes] or ["- None."]),
                 "", "## Summary", summary, "",
                 f"## Expert debate ({'agreed' if agreed else 'no agreement'}: {stopped})"]
        lines += [f"- Round {t['round']}, {t['expert']} ({t['model']}): {t['proposal']}" for t in debate] or \
                 ["- No paid calls were made."]
        lines += ["", "## How to undo or apply", "- Undo an applied prompt change: `python -m factory.extend "
                  "--rollback <agent>` (every change is listed in `registry/history/changelog.md`).",
                  "- Apply a prompt change that waits for you: `python -m factory.extend --agent <name> --guidance "
                  "\"<change>\" --adopt`.", "- A structural fix: a pull request you review."]
        path = self.project.write_file("redesign/report.md", "\n".join(lines) + "\n", "code")
        return path
