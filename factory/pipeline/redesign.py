"""The re-design loop: when a project fails even after research, find out why and propose a fix to ASPIRE itself.

0. The project is archived as a failure case (factory/failures.py): a snapshot from before production that later
   fixes are replayed on, and that becomes a regression test once a fix works.
1. **Paid plan.** A re-design asks once, up front, for its whole paid plan: a stronger model for the audit (Claude
   Opus for auditor 1, GPT for auditor 2) and for the expert debate, with an upper-bound estimate and a hard cap of
   REDESIGN_CAP_USD. Approved, the calls inside the plan go ahead without asking again; declined, the auditors run on
   their free ARC models and the debate is skipped. Paid models only ever see public blackboard entries.
2. Auditor 1 reads the project's full trace (every blackboard entry, with its id) and the fixes that worked on
   earlier failures, and traces each failure to a cause from a fixed list (``CAUSE_KINDS``): a prompt, missing
   knowledge, a loop rule, a model's limits, or a tool or ASPIRE's code. Each cause names the entries that show it.
3. Auditor 2 evaluates those causes and fixes; auditor 1 answers and the two settle the final list. Plain code drops
   evidence ids that are not on the blackboard; a cause left with no evidence is never acted on automatically.
4. A summarizer (GLM) writes up the project for two outside experts, who propose a change and answer each other until
   both agree, the debate's cap is reached, or DEBATE_ROUNDS pass.
5. Causes are routed in plain code:
   - ``prompt`` for a model agent that is not a judge: ``factory.extend`` writes several candidate prompts, keeps the
     best on the held-out benchmark (or the other family's pick), runs the panel, then replays this failure and the
     agent's fixed failures. A candidate that passes every gate is applied automatically (Parker decided on
     2026-10-29 that prompt-only changes need no approval) and the case is marked fixed.
   - ``missing_knowledge``: the knowledge loop researches the topic now (``research``) or it is queued in
     ``<workspace>/research_queue.jsonl``.
   - Everything else, and any judge's prompt, waits for the person.
6. The report (``redesign/report.md``) says what was applied, how to undo it, what was researched, and what waits.

Stop rules: one auditor exchange, DEBATE_ROUNDS rounds, DEBATE_CAP_USD for the debate and REDESIGN_CAP_USD for the
whole re-design in real dollars, MAX_PROMPT_CHANGES prompt changes and MAX_RESEARCH topics per re-design.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from factory.failures import CAUSE_KINDS, Case, Cause, FailureArchive, Tried
from factory.llm import LLM, DataScopeError, complete_validated
from factory.llm.cost import BudgetExceeded
from factory.llm.gate import SpendDenied, SpendRequest, estimate_usd
from factory.pipeline.blackboard import Project
from factory.registry import Registry

AUDITOR_1, AUDITOR_2, SUMMARIZER = "auditor_1", "auditor_2", "summarizer"
DEBATE_ROUNDS, DEBATE_CAP_USD = 2, 0.50
REDESIGN_CAP_USD = 1.00  # every paid call of one re-design: audit, review, settle and debate
DEBATERS = {"claude": "auditor", "chatgpt": "audit_reviewer"}  # roles whose paid step is Claude Opus and GPT
MAX_PROMPT_CHANGES = 2  # each runs the extend gates, which cost free model runs
MAX_RESEARCH = 2
ENTRY_LIMIT = 4000  # characters of one blackboard entry in the trace
RECORD_LIMIT = {False: 120_000, True: 60_000}  # characters of trace for a free / paid auditor


class Audit(BaseModel):
    successes: list[str] = []
    failures: list[str] = []
    causes: list[Cause] = []


class Suggestions(BaseModel):
    suggestions: list[str]


class Settled(BaseModel):
    problems: list[str]
    causes: list[Cause] = []


class Position(BaseModel):
    proposal: str
    agrees_with_other: bool = False
    kind: Literal["prompt", "structural"] = "structural"
    agent: str = ""  # for a prompt change, the agent whose prompt changes


class Applied(BaseModel):
    agent: str
    change: str
    status: str  # adopted | rejected at <gate> | awaiting a person | not tried: <why> | researched | queued ...
    detail: str = ""  # gates, the proposal folder, or the wiki page


class RedesignResult(BaseModel):
    problems: list[str]
    fixes: list[str]  # structural: wait for the person
    debate: list[dict]
    agreed: bool
    stopped: str  # why the debate ended
    report: str  # path
    prompt_changes: list[Applied] = []
    research: list[Applied] = []
    causes: list[Cause] = []
    case: str = ""  # the failure case id
    paid_plan: str = ""


CAUSE_LIST = "\n".join(f"- {k}: {v}" for k, v in CAUSE_KINDS.items())
CAUSE_JSON = ('{{"kind": "<one of the causes above>", "agent": "<agent name, if one>", "evidence": ["<entry id>", '
              '...], "fix": "<what should change>", "topic": "<for missing_knowledge: what to research>"}}')

AUDIT_PROMPT = """ASPIRE, a system that builds solutions with teams of agents, failed this project even after \
researching what it lacked. Its full trace is below: every blackboard entry, as [id] kind by author. Say what went \
well, what failed, and trace each failure to its cause in ASPIRE (not just in the project), choosing from:
{causes}

Every cause must cite the ids of the entries that show it, and name the agent when it concerns one.

## Fixes that worked on earlier failures
{past}

## Project trace
{record}

Reply with JSON only: {{"successes": [...], "failures": [...], "causes": [""" + CAUSE_JSON + """, ...]}}"""

SUGGEST_PROMPT = """Another auditor traced a failed project to causes in the system. Evaluate each: is the cause \
right, does the evidence show it, would the fix work, and what did it miss?

## Its audit
{audit}

Reply with JSON only: {{"suggestions": ["..."]}}"""

SETTLE_PROMPT = """You audited a failed project; a second auditor answered your causes. Weigh their suggestions and \
settle the problems and their causes. Keep each cause's kind from this list:
{causes}

A prompt cause names one of the model agents below. Keep the evidence ids from your audit.

## Your audit
{audit}

## Their suggestions
{suggestions}

## Model agents
{agents}

Reply with JSON only: {{"problems": ["..."], "causes": [""" + CAUSE_JSON + """, ...]}}"""

SUMMARY_PROMPT = """Summarize this failed project for two outside experts: what was asked, what the agents did at each \
stage, where it went wrong, and what the auditors concluded. Plain prose, under 600 words.

## Project trace
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


def record(project: Project, paid: bool = False, limit: int | None = None) -> str:
    """The project's trace, one line per blackboard entry with its id, kind, author and refs. A paid auditor sees
    public entries only. Past the limit, the middle is cut: the request and plan at the top and the failure at the
    bottom are what an auditor needs most."""
    entries = project.board.entries()
    if paid:
        entries = project.board.for_paid(entries)
    lines = [f"[{e.id}] {e.kind} by {e.author}" + (f" (refs {', '.join(e.refs)})" if e.refs else "") + ": " +
             json.dumps(e.data)[:ENTRY_LIMIT] for e in entries]
    limit = limit or RECORD_LIMIT[paid]
    if sum(len(x) + 1 for x in lines) <= limit:
        return "\n".join(lines)
    head, tail, used = [], [], 0
    for line in lines:  # the first quarter of the budget from the top
        if used + len(line) > limit // 4:
            break
        head.append(line)
        used += len(line) + 1
    for line in reversed(lines[len(head):]):
        if used + len(line) > limit:
            break
        tail.insert(0, line)
        used += len(line) + 1
    return "\n".join([*head, f"... {len(lines) - len(head) - len(tail)} entries omitted ...", *tail])


Improver = Callable[[str, str, str], tuple[str, str]]  # (agent, change, case id) -> (status, detail)
Researcher = Callable[..., object]  # KnowledgeLoop.research: Topic -> WikiPage | None


class Redesign:
    def __init__(self, llm: LLM, registry: Registry, project: Project, improver: Improver | None = None,
                 research: Researcher | None = None, archive: FailureArchive | None = None, strong: bool = True):
        self.llm, self.project, self.board, self.registry = llm, project, project.board, registry
        self.agents = {n: registry.agents[n] for n in (AUDITOR_1, AUDITOR_2, SUMMARIZER)}
        self.data = "public" if llm.cfg.data_public else "private"
        self.improver = improver or self.extend
        self.researcher = research
        self.archive = archive or FailureArchive.of(llm.cfg)
        self.strong = strong  # use the paid audit and debate, if the person approves the plan
        self.plan_declined = ""  # why the paid plan is off, once run() has asked
        self.start_usd = llm.ledger.summary()["real_usd"]
        self.notes: list[str] = []  # paid calls that fell back to free models

    def extend(self, agent: str, change: str, case: str = "") -> tuple[str, str]:
        """The real improver: candidate prompts through factory.extend's gates, the best adopted if it passes them
        all, then replayed on this failure."""
        from factory import extend
        result = extend.propose(self.llm.cfg, agent, approver=self.llm.gate.approver, providers=self.llm.providers,
                                guidance=change, case=case, adopt_if_passed=True)
        return result.status, "; ".join(f"{g.name}: {g.detail}" for g in result.gates) + f" ({result.folder})"

    def spent(self) -> float:
        return self.llm.ledger.summary()["real_usd"] - self.start_usd

    def paid_plan(self, trace_chars: int) -> str:
        """Ask once for every paid call this re-design may make, with an upper-bound estimate. Returns a line for the
        report; on approval, calls inside the plan and under the cap are approved without asking again."""
        if not self.strong:
            self.plan_declined = "the paid plan is off for this re-design"
            return "Free models only (the paid plan was off)."
        calls = [("auditor", trace_chars + 4000), ("audit_reviewer", 12_000), ("auditor", 16_000)]
        calls += [(role, 6000) for _ in range(DEBATE_ROUNDS) for role in DEBATERS.values()]
        steps = {r: self.llm.cfg.escalations[r] for r in DEBATERS.values()}
        estimate = sum(estimate_usd(self.llm.cfg.model_info(steps[r].model), chars, steps[r].max_tokens)
                       for r, chars in calls)
        models = " + ".join(dict.fromkeys(s.model for s in steps.values()))
        what = (f"re-design plan: {len(calls)} paid calls (audit, review, settle, {2 * DEBATE_ROUNDS} debate turns), "
                f"capped at ${REDESIGN_CAP_USD:.2f}")
        req = SpendRequest(role="re-design", provider="+".join(dict.fromkeys(s.provider for s in steps.values())),
                           model=models, estimate_usd=round(min(estimate, REDESIGN_CAP_USD), 2),
                           run_spent_usd=round(self.llm.ledger.real_usd, 4), reason=what)
        original = self.llm.gate.approver
        try:
            for s in steps.values():
                self.llm.spend.check(s.provider)  # no point asking for credit that is not there
            approved = original(req)
        except BudgetExceeded as e:
            approved, what = False, f"{what}; {e}"
        self.llm._log(event="spend_gate", role="re-design", model=models, reason=what, estimate_usd=req.estimate_usd,
                      approved=approved)
        if not approved:
            self.plan_declined = f"the re-design's paid plan (up to ${req.estimate_usd:.2f}) was declined"
            return f"Declined: {what}, up to ${req.estimate_usd:.2f}. The auditors used free models."

        def within_plan(r: SpendRequest) -> bool:
            if r.role in DEBATERS.values() and self.spent() + r.estimate_usd <= REDESIGN_CAP_USD:
                return True
            return original(r)  # anything outside the plan asks as usual
        self.llm.gate.approver = within_plan
        return f"Approved: {what}, estimated at up to ${req.estimate_usd:.2f}."

    def _call(self, agent: str, prompt: str, model_cls, tag: str, paid: bool = False):
        spec = self.agents[agent]
        if paid and self.strong and not self.plan_declined and self.spent() < REDESIGN_CAP_USD:
            try:
                out, _, _ = complete_validated(self.llm, spec.role, prompt, spec.system, model_cls, tag=f"{tag}-paid",
                                               data="public", escalate=True)
                return out
            except (SpendDenied, BudgetExceeded, DataScopeError, RuntimeError) as e:
                self.notes.append(f"{tag}: paid model not used ({str(e)[:160]}); a free model answered")
        out, _, _ = complete_validated(self.llm, spec.role, prompt, spec.system, model_cls, tag=tag, data=self.data)
        return out

    def checked(self, causes: list[Cause]) -> list[Cause]:
        """Keep only evidence ids that are on the blackboard."""
        ids = {e.id for e in self.board.entries()}
        return [c.model_copy(update={"evidence": [i for i in c.evidence if i in ids]}) for c in causes]

    def run(self) -> RedesignResult:
        case = self.archive.open_case(self.project.root)
        paid = self.strong
        rec = record(self.project, paid=paid)
        original = self.llm.gate.approver
        try:
            plan = self.paid_plan(len(rec))
            if self.plan_declined and paid:  # free auditors may read the whole trace, ARC-only entries included
                paid, rec = False, record(self.project)
            return self._run(case, rec, plan, paid)
        finally:
            self.llm.gate.approver = original

    def _run(self, case: Case, rec: str, plan: str, paid: bool) -> RedesignResult:
        audit = self._call(AUDITOR_1, AUDIT_PROMPT.format(causes=CAUSE_LIST, past=self.archive.past_fixes(),
                                                          record=rec), Audit, "audit", paid)
        audit.causes = self.checked(audit.causes)
        a = self.board.post("audit", AUDITOR_1, audit.model_dump())
        sugg = self._call(AUDITOR_2, SUGGEST_PROMPT.format(audit=audit.model_dump_json(indent=1)), Suggestions,
                          "audit-suggest", paid)
        s = self.board.post("audit", AUDITOR_2, sugg.model_dump(), refs=[a.id])
        model_agents = "\n".join(f"- {a.name}: {a.description}" for a in self.registry.agents.values()
                                 if a.kind == "llm")
        settled = self._call(AUDITOR_1, SETTLE_PROMPT.format(causes=CAUSE_LIST, audit=audit.model_dump_json(indent=1),
                                                             suggestions=json.dumps(sugg.suggestions),
                                                             agents=model_agents), Settled, "audit-settle", paid)
        settled.causes = self.checked(settled.causes)
        st = self.board.post("audit", AUDITOR_1, settled.model_dump(), refs=[s.id])
        spec = self.agents[SUMMARIZER]
        summary = self.llm.complete(spec.role, SUMMARY_PROMPT.format(record=record(self.project, paid=True),
                                                                     settled=settled.model_dump_json()),
                                    system=spec.system, tag="redesign-summary", data=self.data).text
        self.board.post("note", SUMMARIZER, {"summary": summary}, refs=[st.id])
        debate, agreed, stopped = self.debate(summary)
        causes = list(settled.causes)
        if agreed:  # the experts' agreed change; a prompt change borrows the auditors' evidence for that agent
            last = debate[-1]
            evidence = sorted({i for c in causes if c.agent == last["agent"] for i in c.evidence})
            causes.append(Cause(kind="prompt" if last["kind"] == "prompt" else "tool_or_code", agent=last["agent"],
                                evidence=evidence, fix=last["proposal"]))
        fixes: list[str] = []
        applied = self.apply(causes, fixes, case.id)
        researched = self.research_gaps(causes)
        fixes += [f"[{c.kind}] " + (f"{c.agent}: " if c.agent else "") + c.fix for c in causes
                  if c.kind not in ("prompt", "missing_knowledge")]
        case = self.archive.get(case.id)  # extend may have recorded a fix meanwhile
        case.problems, case.causes = settled.problems, causes
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        case.tried += [Tried(at=stamp, agent=x.agent, change=x.change, status=x.status, detail=x.detail)
                       for x in [*applied, *researched]]
        self.archive.save(case)
        report = self.write_report(settled, causes, summary, debate, agreed, stopped, applied, researched, fixes,
                                   plan, case.id)
        self.board.post("redesign", "code", {"problems": settled.problems, "fixes": fixes, "agreed": agreed,
                                             "stopped": stopped, "report": str(report), "case": case.id,
                                             "prompt_changes": [x.model_dump() for x in applied],
                                             "research": [x.model_dump() for x in researched]}, refs=[st.id])
        return RedesignResult(problems=settled.problems, fixes=fixes, debate=debate, agreed=agreed, stopped=stopped,
                              report=str(report), prompt_changes=applied, research=researched, causes=causes,
                              case=case.id, paid_plan=plan)

    def apply(self, causes: list[Cause], fixes: list[str], case_id: str = "") -> list[Applied]:
        """Plain code decides which prompt changes may be tried; the extend gates decide which are adopted."""
        from factory.extend import JUDGES
        out, seen, tried = [], set(), 0
        for c in (c for c in causes if c.kind == "prompt"):
            spec = self.registry.agents.get(c.agent)
            if spec is None or spec.kind != "llm":
                fixes.append(f"{c.agent}: {c.fix}")  # not a model agent: treat as structural
                continue
            if c.agent in seen:
                continue
            seen.add(c.agent)
            if not c.evidence:
                out.append(Applied(agent=c.agent, change=c.fix, status="awaiting a person",
                                   detail="no blackboard entry shows this cause, so it is not applied on its own"))
            elif c.agent in JUDGES:
                out.append(Applied(agent=c.agent, change=c.fix, status="awaiting a person",
                                   detail="it judges other agents' work, so changing it is structural"))
            elif tried >= MAX_PROMPT_CHANGES:
                out.append(Applied(agent=c.agent, change=c.fix,
                                   status=f"not tried: limit of {MAX_PROMPT_CHANGES} per re-design"))
            else:
                tried += 1
                status, detail = self.improver(c.agent, c.fix, case_id)
                out.append(Applied(agent=c.agent, change=c.fix, status=status, detail=detail))
            self.board.post("redesign", "code", {"prompt_change": out[-1].model_dump()})
        return out

    def research_gaps(self, causes: list[Cause]) -> list[Applied]:
        """Knowledge gaps go to the knowledge loop now, or to the research queue when none was given."""
        from factory.pipeline.knowledge import Topic
        out = []
        for c in (c for c in causes if c.kind == "missing_knowledge"):
            topic = Topic(name=(c.topic or c.fix)[:80], why=c.fix)
            if not c.evidence:
                out.append(Applied(agent="knowledge", change=topic.name, status="not tried: no evidence on the record"))
            elif len([a for a in out if not a.status.startswith("not tried")]) >= MAX_RESEARCH:
                out.append(Applied(agent="knowledge", change=topic.name,
                                   status=f"not tried: limit of {MAX_RESEARCH} per re-design"))
            elif self.researcher is None:
                queue = self.llm.cfg.work_dir / "research_queue.jsonl"
                queue.parent.mkdir(parents=True, exist_ok=True)
                with queue.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"project": self.project.id, **topic.model_dump()}) + "\n")
                out.append(Applied(agent="knowledge", change=topic.name, status="queued for research",
                                   detail=str(queue)))
            else:
                try:
                    page = self.researcher(topic)
                except Exception as e:  # noqa: BLE001 (a failed search is reported, not fatal to the report)
                    out.append(Applied(agent="knowledge", change=topic.name, status="research failed",
                                       detail=f"{type(e).__name__}: {e}"[:200]))
                else:
                    out.append(Applied(agent="knowledge", change=topic.name,
                                       status="researched" if page else "research found nothing",
                                       detail=f"wiki page: {page.topic}" if page else ""))
            self.board.post("redesign", "code", {"research": out[-1].model_dump()})
        return out

    def debate(self, summary: str) -> tuple[list[dict], bool, str]:
        """The two experts answer each other until both agree, a cap is reached, or a person declines."""
        if self.plan_declined:
            return [], False, f"paid call not made: {self.plan_declined}"
        turns: list[dict] = []
        last: dict[str, str] = {}
        start = self.llm.ledger.summary()["real_usd"]
        for rnd in range(1, DEBATE_ROUNDS + 1):
            agrees = {}
            for name, role in DEBATERS.items():
                if self.llm.ledger.summary()["real_usd"] - start >= DEBATE_CAP_USD:
                    return turns, False, f"the debate's ${DEBATE_CAP_USD:.2f} cap was reached"
                if self.spent() >= REDESIGN_CAP_USD:
                    return turns, False, f"the re-design's ${REDESIGN_CAP_USD:.2f} cap was reached"
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

    def write_report(self, settled: Settled, causes: list[Cause], summary: str, debate: list[dict], agreed: bool,
                     stopped: str, applied: list[Applied], researched: list[Applied], fixes: list[str], plan: str,
                     case_id: str):
        adopted = [a for a in applied if a.status == "adopted"]
        lines = [f"# Re-design report: {self.project.id}", "",
                 (f"{len(adopted)} prompt change(s) passed every gate and were applied. " if adopted else
                  "No change was applied. ") + "Structural fixes wait for your approval.", "",
                 "## What went wrong", *[f"- {p}" for p in settled.problems], "",
                 "## Causes", *([f"- **{c.kind}**" + (f" (`{c.agent}`)" if c.agent else "") + f": {c.fix}" +
                                 (f" [evidence: {', '.join(c.evidence)}]" if c.evidence else " [no evidence]")
                                 for c in causes] or ["- None found."]), "",
                 "## Prompt changes", *([f"- `{a.agent}`: {a.change} **{a.status}**" + (f" ({a.detail})" if a.detail
                                         else "") for a in applied] or ["- None proposed."]), "",
                 "## Knowledge gaps", *([f"- {a.change}: **{a.status}**" + (f" ({a.detail})" if a.detail else "")
                                         for a in researched] or ["- None found."]), "",
                 "## Structural fixes waiting for your approval", *([f"- {f}" for f in fixes] or ["- None."]), "",
                 "## Paid plan", plan, *[f"- {n}" for n in self.notes], f"- Real spend in this re-design: "
                 f"${self.spent():.4f}", "", "## Summary", summary, "",
                 f"## Expert debate ({'agreed' if agreed else 'no agreement'}: {stopped})"]
        lines += [f"- Round {t['round']}, {t['expert']} ({t['model']}): {t['proposal']}" for t in debate] or \
                 ["- No paid calls were made."]
        lines += ["", "## Failure archive", f"This project is case `{case_id}` in `{self.archive.root}`. Prompt fixes "
                  "are replayed on it; once one delivers, it becomes a regression test for that agent.",
                  "", "## How to undo or apply", "- Undo an applied prompt change: `python -m factory.extend "
                  "--rollback <agent>` (every change is listed in `registry/history/changelog.md`).",
                  f"- Apply a prompt change that waits for you: `python -m factory.extend --agent <name> --guidance "
                  f"\"<change>\" --case {case_id} --adopt`.", "- A structural fix: a pull request you review."]
        path = self.project.write_file("redesign/report.md", "\n".join(lines) + "\n", "code")
        return path
