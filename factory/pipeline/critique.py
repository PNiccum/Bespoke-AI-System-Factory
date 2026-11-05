"""The critic loop: a project critic finds what is wrong with the built solution, a problem solver of another model
family proposes fixes, and the critic accepts or declines each one.

The critic (DeepSeek) looks for holes, failures and inefficiencies, and judges ease of use and effectiveness, with the
plain-code checks in front of it (compile errors, failing tests), so its opinion is never the only evidence. Problem
solver 1 (gpt-oss) answers each issue with whole replacement files. Code applies only the fixes the critic accepts,
then runs the checks again.

When the critic declines a fix, help escalates one step at a time, as Parker's design lays out:

1. ``wiki``: the solver tries again with the wiki pages closest to the declined issues.
2. ``research``: the knowledge loop researches what the critic says is missing, then the solver tries again.
3. Declined again after research: the loop stops and asks for the re-design loop (redesign.py).

Stop rules: MAX_ROUNDS critic reviews, MAX_ISSUES issues a round, the three escalation steps above.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, model_validator

from factory.events import log_of
from factory.llm import LLM, complete_validated
from factory.pipeline.blackboard import Project
from factory.pipeline.knowledge import KnowledgeLoop, Topic
from factory.pipeline.production import OUTPUT, Checks, NewFile, Production, parse_files
from factory.pipeline.wiki import Wiki, WikiPage, words
from factory.registry import Registry
from factory.repairs import diff_of, library_of

CRITIC, SOLVER = "project_critic", "problem_solver"
MAX_ROUNDS, MAX_ISSUES, MAX_SHOWN_CHARS = 4, 6, 40_000
STEPS = ("plain", "wiki", "research")


class Issue(BaseModel):
    id: str
    severity: Literal["blocker", "major", "minor"]
    where: str
    problem: str


class Critique(BaseModel):
    # left out by a model now and then (the 2026-11-04 primer critic, three times): then it follows from the issues
    acceptable: bool | None = None
    issues: list[Issue] = []

    @model_validator(mode="after")
    def _infer(self):
        if self.acceptable is None:
            self.acceptable = not any(i.severity in ("blocker", "major") for i in self.issues)
        return self


class Fix(BaseModel):
    issue: str  # issue id
    explanation: str
    files: list[NewFile]


class Fixes(BaseModel):
    fixes: list[Fix]


class Verdict(BaseModel):
    issue: str
    accept: bool
    why: str
    missing_knowledge: str = ""  # what the solver would need to know, when the fix fails for lack of it


class Verdicts(BaseModel):
    verdicts: list[Verdict]


class CriticResult(BaseModel):
    accepted: bool  # the critic found nothing blocking, and the checks pass
    redesign: bool  # declined again after research: the re-design loop should run
    rounds: int
    open_issues: list[Issue] = []
    step: str = "plain"


CRITIC_PROMPT = """Review this solution as a demanding user and engineer. Look for holes, failures, inefficiencies, \
and anything that makes it hard to use or less effective than it should be. Judge it against the task.

## Task
{task}

## Plain-code checks
{checks}

## Files
{files}

Reply with JSON only: {{"acceptable": true or false, "issues": [{{"id": "I1", "severity": "blocker" | "major" | \
"minor", "where": "<file or behaviour>", "problem": "<what is wrong and why it matters>"}}]}}
Set "acceptable" to true only when nothing blocker or major is left. List at most {max_issues} issues."""

SOLVE_PROMPT = """Fix these issues in the solution. For each issue, give whole replacement files.

## Task
{task}

## Issues
{issues}

## Files
{files}
{knowledge}
Reply in plain text blocks, not JSON, so code needs no escaping. For each issue:

=== FIX: <issue id, e.g. I1> ===
EXPLANATION: <what changes and why, one line>
=== FILE: <path under output/> ===
<the whole replacement file>
=== END FILE ===
=== END FIX ===

A fix may hold several FILE blocks. Give only files the fix changes, and each file once: when two issues change the \
same file, put the file in the first of those fixes and say so in the other. Fix a setup failure (tests that cannot \
be imported) first."""

FIX_BLOCK = re.compile(r"^=== FIX: *(.+?) *===[ \t]*\n(.*?)^=== END FIX ===[ \t]*$", re.M | re.S)


def parse_fixes(completion) -> dict:
    """The solver's reply as Fixes data: ``=== FIX: id ===`` blocks holding FILE blocks (see production.parse_files;
    the 2026-11-03 primer build's solver lost fixes to JSON escaping too). A JSON reply in the old shape is accepted."""
    text = completion.text
    blocks = FIX_BLOCK.findall(text)
    if not blocks:
        if "=== FIX:" in text:
            raise ValueError("a FIX block has no closing '=== END FIX ===' line")
        return completion.json()
    fixes = []
    for issue, body in blocks:
        why = re.search(r"^EXPLANATION: *(.*)$", body, re.M)
        fixes.append({"issue": issue.strip(), "explanation": why.group(1).strip() if why else "",
                      "files": parse_files(SimpleNamespace(text=body, json=lambda: {"files": []}))["files"]})
    return {"fixes": fixes}

JUDGE_PROMPT = """You raised these issues; another engineer proposes the fixes below. Accept a fix only if it solves \
its issue without breaking anything else. When a fix fails because its author lacks some knowledge (a method, a \
format, a data source), name that knowledge.

## Issues
{issues}

## Proposed fixes
{fixes}

Reply with JSON only: {{"verdicts": [{{"issue": "I1", "accept": true or false, "why": "<one sentence>", \
"missing_knowledge": "<topic, or empty>"}}]}}"""


class CriticLoop:
    def __init__(self, llm: LLM, registry: Registry, project: Project, wiki: Wiki, production: Production,
                 knowledge: KnowledgeLoop):
        self.llm, self.project, self.board, self.wiki = llm, project, project.board, wiki
        self.production, self.knowledge = production, knowledge
        self.agents = {n: registry.agents[n] for n in (CRITIC, SOLVER)}
        self.library = library_of(llm)  # QA repairs that stuck (factory/repairs.py)
        self.pending_repairs: list[str] = []  # this build's applied repairs, confirmed or dropped by run.build

    def _call(self, agent: str, prompt: str, model_cls, tag: str, pages: list[WikiPage] = (), **parse):
        public = self.llm.cfg.data_public and all(p.license == "public" for p in pages)
        spec = self.agents[agent]
        out, _, _ = complete_validated(self.llm, spec.role, prompt, spec.system, model_cls, tag=tag,
                                       data="public" if public else "private", **parse)
        return out

    def shown_files(self) -> str:
        out, used = [], 0
        for rel in self.production.files():
            text = (self.production.out / rel).read_text(errors="replace", encoding="utf-8")
            if used + len(text) > MAX_SHOWN_CHARS:
                out.append(f"### {rel}\n(not shown: over the size limit)")
                continue
            used += len(text)
            out.append(f"### {rel}\n```\n{text}\n```")
        return "\n\n".join(out) or "(no files)"

    def run(self, checks: Checks) -> CriticResult:
        task = (self.project.root / "task.txt").read_text(encoding="utf-8")
        step, pages, issues = 0, [], []
        for rnd in range(1, MAX_ROUNDS + 1):
            try:
                critique = self._call(CRITIC, CRITIC_PROMPT.format(task=task, checks=checks.model_dump_json(indent=1),
                                                                   files=self.shown_files(), max_issues=MAX_ISSUES),
                                      Critique, f"critique-{rnd}", pages)
            except RuntimeError as e:  # no usable review: the checks speak for it (below), and final QA still runs
                log_of(self.llm).record("pipeline", f"critic gave no usable review: {str(e)[:300]}",
                                        stage="critic loop", severity="warning", refs=[f"round {rnd}"])
                self.board.post("note", "code", {"round": rnd, "text": f"critic gave no usable review: {str(e)[:300]}"})
                critique = Critique(acceptable=False, issues=[])
            issues = [i for i in critique.issues if i.severity != "minor"][:MAX_ISSUES]
            c = self.board.post("critic_review", CRITIC, {"round": rnd, "acceptable": critique.acceptable,
                                                          "issues": [i.model_dump() for i in critique.issues]})
            if (critique.acceptable or not issues) and checks.ok:
                return CriticResult(accepted=True, redesign=False, rounds=rnd, step=STEPS[step])
            if not issues:  # the critic is content but the checks fail: the failures are the issues
                issues = [Issue(id="C1", severity="blocker", where="checks", problem=json_checks(checks))]
            knowledge = ""
            if pages:
                knowledge = "\n## Knowledge (wiki)\n" + "\n\n".join(f"### {p.topic}\n{p.body[:3000]}" for p in pages)
            fixes = self._solve(task, issues, knowledge, rnd, pages, c.id)
            verdicts = self._call(CRITIC, JUDGE_PROMPT.format(issues=_show(issues), fixes=fixes.model_dump_json(
                indent=1)[:MAX_SHOWN_CHARS]), Verdicts, f"judge-{rnd}", pages) if fixes.fixes else Verdicts(verdicts=[])
            accepted = {v.issue for v in verdicts.verdicts if v.accept}
            for fix in fixes.fixes:
                if fix.issue in accepted:
                    for f in fix.files:
                        rel = f.path.removeprefix(f"{OUTPUT}/").lstrip("/")
                        if self.production.inside(rel):
                            self.project.write_file(f"{OUTPUT}/{rel}", f.content, SOLVER, refs=[c.id])
            declined = [v for v in verdicts.verdicts if not v.accept]
            self.board.post("fixes", SOLVER, {"round": rnd, "accepted": sorted(accepted),
                                              "declined": [v.model_dump() for v in declined], "step": STEPS[step]},
                            refs=[c.id])
            before = checks
            checks = self.production.check()
            self.board.post("checks", "code", checks.model_dump(), refs=[c.id])
            # a round that leaves failing checks exactly as they were made no progress, whatever was accepted: it
            # escalates like a declined fix (the 2026-11-03 primer build reran identical checks for three rounds)
            stalled = not checks.ok and checks.model_dump() == before.model_dump()
            if stalled and not declined:
                declined = [Verdict(issue=i.id, accept=False, why="the checks did not change",
                                    missing_knowledge=i.problem[:120]) for i in issues]
                self.board.post("note", "code", {"round": rnd, "text": "fixes left the failing checks unchanged: "
                                                 f"escalating from {STEPS[step]}"}, refs=[c.id])
            if not declined:
                continue  # the critic looks again at the fixed solution
            if step == len(STEPS) - 1:  # declined after research: re-design
                return CriticResult(accepted=False, redesign=True, rounds=rnd, open_issues=issues, step=STEPS[step])
            step += 1
            if STEPS[step] == "wiki":
                pages = self.closest_pages([i.problem for i in issues])
                if not pages:  # nothing relevant in the wiki: go straight to research
                    step += 1
            if STEPS[step] == "research":
                wanted = [v.missing_knowledge or v.why for v in declined][:3]
                found = self.knowledge.research_many([Topic(name=w[:80], why="the critic declined a fix for lack "
                                                                  "of it") for w in wanted])
                pages = [p for p in found if p and p.verified] or self.closest_pages(wanted)
        return CriticResult(accepted=False, redesign=False, rounds=MAX_ROUNDS, open_issues=issues, step=STEPS[step])

    def _solve(self, task: str, issues: list[Issue], knowledge: str, rnd: int, pages, ref: str) -> Fixes:
        """The solver's fixes. A reply that will not parse, or overruns the output limit, is tried once more for the
        first issue alone; if that fails too the round has no fixes, so the unchanged checks escalate it. (The
        2026-11-04 log_spikes build ended with no final QA when one solver reply hit max_tokens three times.)"""
        for tag, these in ((f"solve-{rnd}", issues), (f"solve-{rnd}-first", issues[:1])):
            try:
                return self._call(SOLVER, SOLVE_PROMPT.format(task=task, issues=_show(these), files=self.shown_files(),
                                                              knowledge=knowledge), Fixes, tag, pages,
                                  parse=parse_fixes, fmt="FIX blocks")
            except RuntimeError as e:
                log_of(self.llm).record("pipeline", f"solver gave no usable fixes: {str(e)[:300]}",
                                        stage="critic loop", severity="warning", refs=[f"round {rnd}"])
                self.board.post("note", "code", {"round": rnd, "text": f"solver gave no usable fixes: {str(e)[:300]}"},
                                refs=[ref])
                if len(issues) == 1:
                    break
        return Fixes(fixes=[])

    def repair(self, result, rnd: int) -> bool:
        """Repair what final QA found instead of rejecting on it: the solver gets QA's unmet tasks and issues as
        issues, and its files are applied only if the plain-code checks do not get worse (otherwise they are put back).
        Returns whether any file changed. (2026-11-05: csv_profile and log_spikes passed 5/5 acceptance tests and were
        still stopped by QA findings that one small edit would have cleared.)"""
        issues = qa_issues(result, self.production.check())
        if not issues:
            return False
        task = (self.project.root / "task.txt").read_text(encoding="utf-8")
        note = self.board.post("note", "code", {"round": rnd, "qa_repair": True,  # the tallies count this field
                                                "text": f"final QA found problems: repairing {', '.join(i.id for i in issues)}"})
        before_checks = self.production.check()
        snapshot = {rel: (self.production.out / rel).read_text(errors="replace", encoding="utf-8")
                    for rel in self.production.files()}
        past = self.library.section([i.problem for i in issues]) if self.library else ""
        if past:
            self.board.post("note", "code", {"round": rnd, "text": "showing the solver past repairs that stuck"},
                            refs=[note.id])
        fixes = self._solve(task, issues, past, rnd, [], note.id)
        for fix in fixes.fixes:
            for f in fix.files:
                rel = f.path.removeprefix(f"{OUTPUT}/").lstrip("/")
                if self.production.inside(rel):
                    self.project.write_file(f"{OUTPUT}/{rel}", f.content, SOLVER, refs=[note.id])
        after = {rel: (self.production.out / rel).read_text(errors="replace", encoding="utf-8")
                 for rel in self.production.files()}
        if after == snapshot:
            return False
        checks = self.production.check()
        if before_checks.ok and not checks.ok:  # the repair broke a working solution: put it back
            for rel in set(after) - set(snapshot):
                (self.production.out / rel).unlink()
            for rel, text in snapshot.items():
                self.project.write_file(f"{OUTPUT}/{rel}", text, "code", refs=[note.id])
            log_of(self.llm).record("qa", "a QA repair broke the plain-code checks and was put back",
                                    stage="final QA repair", severity="warning", refs=[note.id])
            self.board.post("note", "code", {"round": rnd, "text": "the QA repair broke the checks: put back"},
                            refs=[note.id])
            return False
        self.board.post("checks", "code", checks.model_dump(), refs=[note.id])
        self._remember(fixes, issues, snapshot, after)
        return True

    def _remember(self, fixes: Fixes, issues: list[Issue], before: dict[str, str], after: dict[str, str]) -> None:
        """Save each applied fix as a pending library entry; ``run.build`` confirms them if QA then approves."""
        if not self.library:
            return
        by_id = {i.id: i for i in issues}
        for fix in fixes.fixes:
            paths = sorted({f.path.removeprefix(f"{OUTPUT}/").lstrip("/") for f in fix.files})
            issue = by_id.get(fix.issue)
            rid = self.library.add(issue.problem if issue else fix.issue, diff_of(before, after, paths),
                                   explanation=fix.explanation, files=paths, run=self.llm.run_dir.name
                                   if hasattr(self.llm, "run_dir") else "", project=self.project.id)
            if rid:
                self.pending_repairs.append(rid)

    def closest_pages(self, texts: list[str], n: int = 3) -> list[WikiPage]:
        want = set().union(*(words(t) for t in texts)) if texts else set()
        scored = [(len(want & words(p.topic + " " + p.summary)), p) for p in self.wiki.pages() if p.verified]
        return [p for s, p in sorted(scored, key=lambda x: -x[0]) if s > 0][:n]


def qa_issues(result, checks: Checks) -> list[Issue]:
    """Final QA's findings as issues for the solver: each unmet task with QA's evidence, the failing checks, and the
    other issues QA listed (the two summary lines code adds are left out)."""
    out = [Issue(id=f"Q{t.id}", severity="blocker", where=t.id,
                 problem=f"final QA says task {t.id} is not met: {t.evidence}") for t in result.tasks if not t.met]
    if not checks.ok:
        out.append(Issue(id="QC", severity="blocker", where="checks", problem=json_checks(checks)))
    skip = ("acceptance not met", "the plain-code checks fail")
    out += [Issue(id=f"QI{n}", severity="major", where="final QA", problem=i)
            for n, i in enumerate(result.issues, 1) if not i.startswith(skip)]
    return out[:MAX_ISSUES]


def _show(issues: list[Issue]) -> str:
    return "\n".join(f"- {i.id} [{i.severity}] {i.where}: {i.problem}" for i in issues)


def json_checks(checks: Checks) -> str:
    parts = [f"{f}: {e}" for f, e in checks.syntax_errors.items()]
    if checks.setup_error:
        parts.append(f"setup failure, fix this first: {checks.setup_error}")
    elif checks.tests_ok is False:
        why = "\n".join(f"  {t}: {r}" for t, r in checks.reasons.items())
        parts.append(f"failing tests: {', '.join(checks.failed) or 'see output'}; {checks.output_tail[-500:]}"
                     + (f"\nwhy each fails:\n{why}" if why else "")
                     + "\nFor each failing test decide whether the test or the code is wrong: fix a test that "
                       "contradicts the task or has a bug (a missing fixture argument, a wrong expected value); fix "
                       "the code when the test is right. Never weaken a check that states a task's acceptance criterion.")
    return "; ".join(parts) or "checks failed"
