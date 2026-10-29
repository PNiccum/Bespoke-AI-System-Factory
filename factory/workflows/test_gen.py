"""Write tests for existing code, and prove they are worth keeping without asking a model.

1. ``locate``: the code_navigator finds the code the request names (plain code).
2. ``baseline``: run the suite before any change, so pre-existing failures are not blamed on the new tests.
3. ``write``: the test_writer writes a diff that adds or changes test files only (a model, ARC first).
4. ``check``: the diff must apply and touch only tests, at least one new test must pass, and nothing that passed
   before may fail.
5. ``score``: the mutation_tester changes the target code one small bug at a time; the new tests must catch at
   least ``MIN_SCORE`` of those mutants (plain code).
6. ``review``: a reviewer from another model family checks the tests assert intended behaviour, not just run.

A failure at 3 to 6 goes back to the writer with the reason (surviving mutants are listed), up to ``max_attempts``;
the last attempt escalates to the paid model behind the spend gate. Tests that pass but never reach the mutation
bar are kept as ``unproven``.
"""

from __future__ import annotations

from pathlib import Path

from factory.llm import BudgetExceeded, TruncatedError, complete_validated
from factory.llm.gate import SpendDenied
from factory.orchestrator import Orchestrator, Review, RunResult, Step, Task, extract_diff, sha12
from factory.records import ProvenanceEvent
from factory.tools.diff_inspect import is_test_file, parse_diff
from factory.tools.quality import MutationReport, ScoreTestsIn
from factory.tools.repo import LocateIn, Locations, python_files
from factory.workflows.common import begin, finish

MIN_SCORE = 0.6

TEMPLATE = [
    Step(id="locate", capability="locate_code", needs_llm=False,
         reason="the functions and files the request names are found by search"),
    Step(id="baseline", capability="run_tests", needs_llm=False,
         reason="which tests already fail is a fact pytest reports"),
    Step(id="write", capability="write_tests", needs_llm=True,
         reason="choosing cases and expected values from code needs language understanding"),
    Step(id="check", capability="run_tests", needs_llm=False,
         reason="whether the new tests pass and break nothing is a fact pytest reports"),
    Step(id="score", capability="score_tests", needs_llm=False,
         reason="whether the tests catch bugs is measured with mutants, not judged"),
    Step(id="review", capability="review_tests", needs_llm=True,
         reason="judging whether tests check the intended behaviour needs a second reader"),
]

WRITER_PROMPT = """Write pytest tests for this request.

## Request
{request}

## Code to test
{targets}

## Existing tests nearby
{existing}
{feedback}
Reply with one unified diff (--- a/path, +++ b/path, @@ hunks; --- /dev/null for a new file) inside a ```diff block
and nothing else. Change test files only. Each test must assert an exact result of the code as it is now."""

REVIEW_PROMPT = """Review these new tests. You did not write them.

## Request
{request}

## Tests
```diff
{patch}
```

## Facts
The new tests pass on the current code: {passing}
They caught {killed} of {total} small bugs planted in the code; missed: {survivors}

Reply with JSON only: {{"verdict": "accept" or "reject", "defects": [{{"line": "<diff line>", "problem": "<what is wrong>"}}]}}.
Reject only for a concrete defect: a test that asserts nothing useful, an expected value that contradicts the
request, or a test of something the request did not ask about."""


def targets_of(locations: Locations) -> tuple[list[str], list[str]]:
    """(files to test, functions to test): the strongest source hits, and the names defined in them."""
    source = [h for h in locations.hits if not h.is_test]
    if not source:
        return [], []
    picked = [h for h in source if h.score >= source[0].score / 2][:2]
    functions = sorted({w.removeprefix("definition of ") for h in picked for w in h.why
                        if w.startswith("definition of ")})
    return [h.path for h in picked], functions


def nearby_tests(root: Path, files: list[str]) -> list[str]:
    """A test file that imports one of the target modules, to show the writer the suite's style and imports."""
    modules = [f[:-3].replace("/", ".") for f in files]
    for p in python_files(root):
        rel = p.relative_to(root).as_posix()
        if is_test_file(rel) and any(m in p.read_text(errors="replace") for m in modules):
            return [rel]
    return []


def run(orch: Orchestrator, task: Task) -> RunResult:
    result, bids = begin(orch, task, TEMPLATE)
    root = Path(task.repo_path)

    nav = bids["locate"].agent
    locations: Locations = orch.tool(nav)(LocateIn(repo_path=task.repo_path, issue_text=task.description))
    files, functions = targets_of(locations)
    if not files:
        result.stop_reason = "found no source code the request points at; name the file or function"
        orch.emit(result, "locate", nav.name, "failed", result.stop_reason)
        return finish(orch, task, result, bids, passed=False)
    orch.emit(result, "locate", nav.name, "done", f"testing {', '.join(functions) or 'all of'} in {', '.join(files)}",
              files=files, functions=functions)

    runner = bids["baseline"]
    orch.emit(result, "baseline", runner.agent.name, "started", "running the tests before any change")
    base = orch.run_tests(runner, task, None, [])
    broken_before = set(base.failed + base.errors)
    orch.emit(result, "baseline", runner.agent.name, "done", f"{len(broken_before)} failing, {len(base.passed)} passing")

    shown_tests = [h.path for h in locations.hits if h.is_test][:1] or nearby_tests(root, files)
    target_text = "\n\n".join(f"### {p}\n```python\n{(root / p).read_text(errors='replace')}\n```" for p in files)
    existing = "\n\n".join(f"### {p}\n```python\n{(root / p).read_text(errors='replace')}\n```"
                           for p in shown_tests) or "(none)"

    feedback, best = "", None  # best: (patch, check, score, new tests) that passed but fell short of MIN_SCORE
    writer, scorer, reviewer = bids["write"].agent, bids["score"].agent, bids["review"].agent
    for attempt in range(1, orch.max_attempts + 1):
        result.attempts = attempt
        escalate = attempt == orch.max_attempts and attempt > 1
        orch.emit(result, "write", writer.name, "started",
                  f"attempt {attempt}" + (" on the paid model" if escalate else ""))
        try:
            reply = orch.llm.complete(writer.role, WRITER_PROMPT.format(
                request=task.description, targets=target_text, existing=existing, feedback=feedback),
                system=writer.system, tag=f"tests-{attempt}", data=orch.data, escalate=escalate)
        except (SpendDenied, BudgetExceeded) as e:
            return keep_best(orch, task, result, bids, best, orch.stop(result, "write", writer.name, str(e)))
        except TruncatedError as e:
            feedback = f"\n## Your previous reply was cut off\n{e}. Write fewer, shorter tests.\n"
            orch.emit(result, "write", writer.name, "failed", str(e))
            continue
        patch = extract_diff(reply.text)
        orch.provenance.record(ProvenanceEvent(kind="model_call", actor=writer.name, outputs=[sha12(patch)],
                                               detail={"model": reply.model, "tag": f"tests-{attempt}",
                                                       "real_usd": reply.real_usd}))
        changed = [f.path for f in parse_diff(patch)] if patch else []
        if not changed:
            feedback = "\n## Your previous reply had no diff\nReply with a ```diff block.\n"
            orch.emit(result, "write", writer.name, "failed", "no diff in the reply")
            continue
        if outside := [p for p in changed if not is_test_file(p)]:
            feedback = f"\n## Your previous diff changed non-test files (not applied)\n{outside}. Change tests only.\n"
            orch.emit(result, "write", writer.name, "failed", f"changes non-test files: {outside}")
            continue
        orch.emit(result, "write", writer.name, "done", f"{patch.count(chr(10))} diff lines", model=reply.model)

        checker = bids["check"]
        orch.emit(result, "check", checker.agent.name, "started", "running the suite with the new tests")
        check = orch.run_tests(checker, task, patch, [])
        result.patch, result.check = patch, check
        new = [n for n in check.passed if n.split("::")[0] in changed and n not in base.passed]
        broke = sorted(set(check.failed + check.errors) - broken_before)
        if check.patch_applied is False or check.timed_out or broke or not new:
            why = ("the diff did not apply:\n" + check.output_tail if check.patch_applied is False
                   else "the tests timed out" if check.timed_out
                   else f"failing: {broke}\n{check.output_tail}" if broke
                   else "no new test was collected and passed")
            orch.emit(result, "check", checker.agent.name, "failed", why[:300])
            feedback = f"\n## Your previous tests failed (not applied)\n```diff\n{patch}```\n{why[-3000:]}\n"
            continue
        orch.emit(result, "check", checker.agent.name, "done", f"{len(new)} new tests pass", new_tests=new)

        orch.emit(result, "score", scorer.name, "started", "planting small bugs to see if the tests catch them")
        score: MutationReport = orch.tool(scorer)(ScoreTestsIn(repo_path=task.repo_path, files=files, tests=new,
                                                               functions=functions, patch=patch))
        orch.provenance.record(ProvenanceEvent(kind="tool_call", actor=scorer.name, inputs=[sha12(patch)],
                                               detail=score.model_dump()))
        result.report = {"targets": files, "functions": functions, "new_tests": new, "mutation": score.model_dump(),
                         "min_score": MIN_SCORE}
        if score.error or (score.total and score.score < MIN_SCORE):
            missed = "\n".join(f"- {s['path']}:{s['line']}: {s['change']}" for s in score.survivors)
            why = score.error or f"caught {score.killed} of {score.total} planted bugs; missed:\n{missed}"
            orch.emit(result, "score", scorer.name, "failed", why[:300])
            if not score.error and (best is None or score.score > best[2].score):
                best = (patch, check, score, new)
            feedback = (f"\n## Your previous tests were too weak (not applied)\n```diff\n{patch}```\n{why}\n"
                        "Add assertions that would fail if those changes were made.\n")
            continue
        orch.emit(result, "score", scorer.name, "done", f"caught {score.killed} of {score.total} planted bugs",
                  mutation=score.model_dump())

        orch.emit(result, "review", reviewer.name, "started", "reading the tests")
        survivors = ", ".join(f"{s['path']}:{s['line']} {s['change']}" for s in score.survivors) or "none"
        try:
            review, _, _ = complete_validated(
                orch.llm, reviewer.role, REVIEW_PROMPT.format(request=task.description, patch=patch,
                                                              passing=", ".join(new), killed=score.killed,
                                                              total=score.total, survivors=survivors),
                reviewer.system, Review, tag=f"test-review-{attempt}", data=orch.data)
        except (SpendDenied, BudgetExceeded) as e:
            return orch.stop(result, "review", reviewer.name, str(e))
        except RuntimeError as e:
            return orch.stop(result, "review", reviewer.name, str(e)[:500])
        result.review = review
        orch.provenance.record(ProvenanceEvent(kind="model_call", actor=reviewer.name, inputs=[sha12(patch)],
                                               detail={"verdict": review.verdict,
                                                       "defects": [d.model_dump() for d in review.defects]}))
        if review.verdict == "reject":
            problems = "\n".join(f"- {d.line}: {d.problem}" for d in review.defects) or "- (no detail given)"
            orch.emit(result, "review", reviewer.name, "failed", problems[:300])
            feedback = f"\n## Your previous tests were rejected in review (not applied)\n```diff\n{patch}```\n{problems}\n"
            continue
        orch.emit(result, "review", reviewer.name, "done", "accepted")
        result.status = "tested" if score.total else "unproven"  # no mutation sites: nothing to measure against
        return finish(orch, task, result, bids, passed=result.status == "tested")

    result.stop_reason = f"no accepted tests after {orch.max_attempts} attempts"
    return keep_best(orch, task, result, bids, best, result)


def keep_best(orch, task, result, bids, best, ended: RunResult) -> RunResult:
    """When no attempt met the bar, hand back the strongest passing tests as unproven rather than nothing."""
    if best is not None:
        patch, check, score, new = best
        result.patch, result.check, result.review = patch, check, None
        result.report = {**result.report, "new_tests": new, "mutation": score.model_dump()}
        result.status = "unproven"
        result.stop_reason = (ended.stop_reason + "; " if ended.stop_reason else "") + \
            f"kept the strongest passing tests (caught {score.killed} of {score.total} planted bugs)"
    return finish(orch, task, result, bids, passed=False)
