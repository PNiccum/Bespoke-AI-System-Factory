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
5. The report (``redesign/report.md``) goes to the person. Nothing changes until they approve: a change to a model
   agent's prompt goes through ``python -m factory.extend``; anything bigger is a pull request.

Stop rules: one auditor exchange, DEBATE_ROUNDS rounds, DEBATE_CAP_USD real dollars.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from factory.llm import LLM, complete_validated
from factory.llm.cost import BudgetExceeded
from factory.llm.gate import SpendDenied
from factory.pipeline.blackboard import Project
from factory.registry import Registry

AUDITOR_1, AUDITOR_2, SUMMARIZER = "auditor_1", "auditor_2", "summarizer"
DEBATE_ROUNDS, DEBATE_CAP_USD = 2, 0.50
DEBATERS = {"claude": "reviewer", "chatgpt": "coder"}  # roles whose paid escalation step is Claude and ChatGPT


class Audit(BaseModel):
    successes: list[str] = []
    failures: list[str] = []
    causes: list[str] = []
    proposals: list[str] = []


class Suggestions(BaseModel):
    suggestions: list[str]


class Settled(BaseModel):
    problems: list[str]
    fixes: list[str]


class Position(BaseModel):
    proposal: str
    agrees_with_other: bool = False


class RedesignResult(BaseModel):
    problems: list[str]
    fixes: list[str]
    debate: list[dict]
    agreed: bool
    stopped: str  # why the debate ended
    report: str  # path


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
and settle the list of problems and the fixes worth making to the system.

## Your audit
{audit}

## Their suggestions
{suggestions}

Reply with JSON only: {{"problems": ["..."], "fixes": ["..."]}}"""

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
it>", "agrees_with_other": true or false}}"""


def record(project: Project, limit: int = 30_000) -> str:
    """The public blackboard as text, newest last, trimmed from the front to the limit."""
    lines = [f"{e.id} {e.author}: {json.dumps(e.data)[:1500]}" for e in project.board.for_paid(project.board.entries())]
    text = "\n".join(lines)
    return text[-limit:]


class Redesign:
    def __init__(self, llm: LLM, registry: Registry, project: Project):
        self.llm, self.project, self.board = llm, project, project.board
        self.agents = {n: registry.agents[n] for n in (AUDITOR_1, AUDITOR_2, SUMMARIZER)}
        self.data = "public" if llm.cfg.data_public else "private"

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
        settled = self._call(AUDITOR_1, SETTLE_PROMPT.format(audit=audit.model_dump_json(indent=1),
                                                             suggestions=json.dumps(sugg.suggestions)), Settled,
                             "audit-settle")
        st = self.board.post("audit", AUDITOR_1, settled.model_dump(), refs=[s.id])
        spec = self.agents[SUMMARIZER]
        summary = self.llm.complete(spec.role, SUMMARY_PROMPT.format(record=rec, settled=settled.model_dump_json()),
                                    system=spec.system, tag="redesign-summary", data=self.data).text
        self.board.post("note", SUMMARIZER, {"summary": summary}, refs=[st.id])
        debate, agreed, stopped = self.debate(summary)
        report = self.write_report(settled, summary, debate, agreed, stopped)
        self.board.post("redesign", "code", {"problems": settled.problems, "fixes": settled.fixes, "agreed": agreed,
                                             "stopped": stopped, "report": str(report)}, refs=[st.id])
        return RedesignResult(problems=settled.problems, fixes=settled.fixes, debate=debate, agreed=agreed,
                              stopped=stopped, report=str(report))

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

    def write_report(self, settled: Settled, summary: str, debate: list[dict], agreed: bool, stopped: str):
        lines = [f"# Re-design report: {self.project.id}", "", "Nothing has changed. These are proposals for you to "
                 "approve or decline.", "", "## What went wrong", *[f"- {p}" for p in settled.problems], "",
                 "## Fixes the auditors agreed on", *[f"- {f}" for f in settled.fixes], "", "## Summary", summary, "",
                 f"## Expert debate ({'agreed' if agreed else 'no agreement'}: {stopped})"]
        lines += [f"- Round {t['round']}, {t['expert']} ({t['model']}): {t['proposal']}" for t in debate] or \
                 ["- No paid calls were made."]
        lines += ["", "## How to apply", "- A change to a model agent's prompt: `python -m factory.extend --agent "
                  "<name>` (gated, benchmarked, then waits for you).", "- Anything bigger: a pull request you review."]
        path = self.project.write_file("redesign/report.md", "\n".join(lines) + "\n", "code")
        return path
