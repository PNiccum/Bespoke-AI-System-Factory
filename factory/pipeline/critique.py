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

from typing import Literal

from pydantic import BaseModel

from factory.llm import LLM, complete_validated
from factory.pipeline.blackboard import Project
from factory.pipeline.knowledge import KnowledgeLoop, Topic
from factory.pipeline.production import OUTPUT, Checks, NewFile, Production
from factory.pipeline.wiki import Wiki, WikiPage, words
from factory.registry import Registry

CRITIC, SOLVER = "project_critic", "problem_solver"
MAX_ROUNDS, MAX_ISSUES, MAX_SHOWN_CHARS = 4, 6, 40_000
STEPS = ("plain", "wiki", "research")


class Issue(BaseModel):
    id: str
    severity: Literal["blocker", "major", "minor"]
    where: str
    problem: str


class Critique(BaseModel):
    acceptable: bool
    issues: list[Issue] = []


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
Reply with JSON only: {{"fixes": [{{"issue": "I1", "explanation": "<what changes and why>", "files": [{{"path": \
"<path under output/>", "content": "<whole file>"}}]}}]}}"""

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

    def _call(self, agent: str, prompt: str, model_cls, tag: str, pages: list[WikiPage] = ()):
        public = self.llm.cfg.data_public and all(p.license == "public" for p in pages)
        spec = self.agents[agent]
        out, _, _ = complete_validated(self.llm, spec.role, prompt, spec.system, model_cls, tag=tag,
                                       data="public" if public else "private")
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
            critique = self._call(CRITIC, CRITIC_PROMPT.format(task=task, checks=checks.model_dump_json(indent=1),
                                                               files=self.shown_files(), max_issues=MAX_ISSUES),
                                  Critique, f"critique-{rnd}", pages)
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
            fixes = self._call(SOLVER, SOLVE_PROMPT.format(task=task, issues=_show(issues), files=self.shown_files(),
                                                           knowledge=knowledge), Fixes, f"solve-{rnd}", pages)
            verdicts = self._call(CRITIC, JUDGE_PROMPT.format(issues=_show(issues), fixes=fixes.model_dump_json(
                indent=1)[:MAX_SHOWN_CHARS]), Verdicts, f"judge-{rnd}", pages)
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
            checks = self.production.check()
            self.board.post("checks", "code", checks.model_dump(), refs=[c.id])
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
                found = [self.knowledge.research(Topic(name=w[:80], why="the critic declined a fix for lack of it"))
                         for w in wanted]
                pages = [p for p in found if p and p.verified] or self.closest_pages(wanted)
        return CriticResult(accepted=False, redesign=False, rounds=MAX_ROUNDS, open_issues=issues, step=STEPS[step])

    def closest_pages(self, texts: list[str], n: int = 3) -> list[WikiPage]:
        want = set().union(*(words(t) for t in texts)) if texts else set()
        scored = [(len(want & words(p.topic + " " + p.summary)), p) for p in self.wiki.pages() if p.verified]
        return [p for s, p in sorted(scored, key=lambda x: -x[0]) if s > 0][:n]


def _show(issues: list[Issue]) -> str:
    return "\n".join(f"- {i.id} [{i.severity}] {i.where}: {i.problem}" for i in issues)


def json_checks(checks: Checks) -> str:
    parts = [f"{f}: {e}" for f, e in checks.syntax_errors.items()]
    if checks.setup_error:
        parts.append(f"setup failure, fix this first: {checks.setup_error}")
    elif checks.tests_ok is False:
        parts.append(f"failing tests: {', '.join(checks.failed) or 'see output'}; {checks.output_tail[-500:]}")
    return "; ".join(parts) or "checks failed"
