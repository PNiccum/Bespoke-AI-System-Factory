"""Review a pull request: plain code establishes the facts, a model adds judgement, plain code checks the model.

1. ``inspect``: the diff_inspector parses the diff and runs fixed rules (debug calls, credentials, bare except, new
   TODOs, no tests touched, files that no longer parse).
2. ``baseline`` and ``check``: the suite runs without and with the change, giving the test delta: does the patch
   apply, which tests it fixes, which it breaks.
3. ``review``: the pr_reviewer reads the description, the diff and those facts, and writes a verdict and comments.

Every model comment must name a file and line the diff changes; comments that do not are dropped and the reviewer is
asked again with the reason, up to ``max_attempts`` (the last attempt on the paid model, behind the spend gate). The
facts outrank the model: a patch that does not apply, breaks a test, or carries a blocker finding is never approved.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ValidationError

from factory.llm import BudgetExceeded, TruncatedError, complete_validated
from factory.llm.gate import SpendDenied
from factory.orchestrator import Orchestrator, RunResult, Step, Task, sha12
from factory.records import ProvenanceEvent
from factory.tools.diff_inspect import parse_diff
from factory.tools.quality import InspectDiffIn
from factory.workflows.common import begin, finish

TEMPLATE = [
    Step(id="inspect", capability="inspect_diff", needs_llm=False,
         reason="what a diff changes, and rule violations in it, are found by parsing"),
    Step(id="baseline", capability="run_tests", needs_llm=False,
         reason="the suite's state before the change is a fact pytest reports"),
    Step(id="check", capability="run_tests", needs_llm=False,
         reason="which tests the change fixes or breaks is a fact pytest reports"),
    Step(id="review", capability="review_pr", needs_llm=True,
         reason="judging correctness and scope against the description needs a reader; its comments are then "
                "checked against the diff in plain code"),
]


class Comment(BaseModel):
    file: str
    line: int  # a line number in the new version of the file
    severity: Literal["blocker", "major", "minor", "nit"]
    problem: str
    suggestion: str = ""


class PRReview(BaseModel):
    verdict: Literal["approve", "request_changes", "comment"]
    summary: str
    comments: list[Comment] = []


REVIEW_PROMPT = """Review this pull request.

## Description
{description}

## Diff
```diff
{patch}
```

## Facts found by tools (already reported; do not repeat them)
{findings}

## Tests
{tests}
{feedback}
Reply with JSON only: {{"verdict": "approve" | "request_changes" | "comment", "summary": "<two sentences>",
"comments": [{{"file": "<path>", "line": <line in the new file>, "severity": "blocker" | "major" | "minor" | "nit",
"problem": "<what is wrong>", "suggestion": "<what to do instead>"}}]}}.
Comment only on lines this diff adds or changes. An empty comments list is fine when nothing is wrong."""


def run(orch: Orchestrator, task: Task) -> RunResult:
    result, bids = begin(orch, task, TEMPLATE)
    if not task.patch.strip():
        result.stop_reason = "no diff to review"
        orch.emit(result, "inspect", bids["inspect"].agent.name, "failed", result.stop_reason)
        return finish(orch, task, result, bids, passed=False)
    result.patch = task.patch

    inspector = bids["inspect"].agent
    facts = orch.tool(inspector)(InspectDiffIn(patch=task.patch, repo_path=task.repo_path)).model_dump()
    orch.provenance.record(ProvenanceEvent(kind="tool_call", actor=inspector.name, inputs=[sha12(task.patch)],
                                           detail=facts))
    if not facts["files"]:
        result.stop_reason = "the diff changes no files"
        orch.emit(result, "inspect", inspector.name, "failed", result.stop_reason)
        return finish(orch, task, result, bids, passed=False)
    orch.emit(result, "inspect", inspector.name, "done",
              f"{len(facts['files'])} files, {len(facts['findings'])} rule findings", findings=facts["findings"])

    orch.emit(result, "baseline", bids["baseline"].agent.name, "started", "running the tests before the change")
    base = orch.run_tests(bids["baseline"], task, None, [])
    orch.emit(result, "baseline", bids["baseline"].agent.name, "done",
              f"{len(base.failed + base.errors)} failing, {len(base.passed)} passing")
    orch.emit(result, "check", bids["check"].agent.name, "started", "running the tests with the change")
    after = orch.run_tests(bids["check"], task, task.patch, [])
    result.check = after
    before_broken, after_broken = set(base.failed + base.errors), set(after.failed + after.errors)
    tests = {"applied": after.patch_applied is not False, "timed_out": after.timed_out,
             "newly_failing": sorted(after_broken - before_broken),
             "newly_passing": sorted(before_broken & set(after.passed)),
             "passed": len(after.passed), "failed": len(after_broken)}
    orch.emit(result, "check", bids["check"].agent.name, "done" if tests["applied"] else "failed",
              "the patch does not apply" if not tests["applied"] else
              f"fixes {len(tests['newly_passing'])}, breaks {len(tests['newly_failing'])}", tests=tests)

    blocking = ([] if tests["applied"] else ["the patch does not apply"]) + \
        ([f"it breaks {', '.join(tests['newly_failing'])}"] if tests["newly_failing"] else []) + \
        (["the tests time out"] if tests["timed_out"] else []) + \
        [f"{f['path']}:{f['line']}: {f['message']}" for f in facts["findings"] if f["severity"] == "blocker"]
    changes = {f.path: f for f in parse_diff(task.patch)}
    findings = "\n".join(f"- [{f['severity']}] {f['path']}:{f['line']}: {f['message']}" for f in facts["findings"]) \
        or "- none"
    test_text = ("the patch does not apply" if not tests["applied"] else
                 f"passing {tests['passed']}, failing {tests['failed']}; newly passing: "
                 f"{tests['newly_passing'] or 'none'}; newly failing: {tests['newly_failing'] or 'none'}")

    reviewer = bids["review"].agent
    feedback, review, dropped = "", None, []
    for attempt in range(1, orch.max_attempts + 1):
        result.attempts = attempt
        escalate = attempt == orch.max_attempts and attempt > 1
        orch.emit(result, "review", reviewer.name, "started",
                  f"attempt {attempt}" + (" on the paid model" if escalate else ""))
        prompt = REVIEW_PROMPT.format(description=task.description, patch=task.patch, findings=findings,
                                      tests=test_text, feedback=feedback)
        try:
            if escalate:  # one paid try, no parse retries: each would be another paid call
                reply = orch.llm.complete(reviewer.role, prompt, system=reviewer.prompt, tag=f"pr-review-{attempt}",
                                          data=orch.data, escalate=True)
                candidate = PRReview.model_validate(reply.json())
            else:
                candidate, reply, _ = complete_validated(orch.llm, reviewer.role, prompt, reviewer.prompt, PRReview,
                                                         tag=f"pr-review-{attempt}", data=orch.data)
        except (SpendDenied, BudgetExceeded) as e:
            if review is not None:  # keep the grounded part of the last free review
                break
            return orch.stop(result, "review", reviewer.name, str(e))
        except (RuntimeError, ValueError, ValidationError, TruncatedError) as e:
            orch.emit(result, "review", reviewer.name, "failed", str(e)[:300])
            feedback = "\n## Your previous reply could not be used\nReply with the JSON object only.\n"
            continue
        kept = [c for c in candidate.comments if c.file in changes and changes[c.file].touches(c.line)]
        dropped = [c for c in candidate.comments if c not in kept]
        orch.provenance.record(ProvenanceEvent(kind="model_call", actor=reviewer.name, inputs=[sha12(task.patch)],
                                               detail={"model": reply.model, "verdict": candidate.verdict,
                                                       "kept": len(kept), "dropped": len(dropped)}))
        review = candidate.model_copy(update={"comments": kept})
        if not dropped:
            break
        where = "\n".join(f"- {c.file}:{c.line}: {c.problem}" for c in dropped)
        orch.emit(result, "review", reviewer.name, "failed", f"{len(dropped)} comments not on changed lines")
        feedback = (f"\n## Your previous review commented on lines this diff does not change\n{where}\n"
                    f"Changed files: {', '.join(changes)}. Use line numbers from the + side of the hunks.\n")

    if review is None:
        result.stop_reason = f"no usable review after {orch.max_attempts} attempts"
        return finish(orch, task, result, bids, passed=False)
    overridden = ""
    if blocking and review.verdict == "approve":
        overridden = "approve changed to request_changes: " + "; ".join(blocking)
        review = review.model_copy(update={"verdict": "request_changes"})
    orch.emit(result, "review", reviewer.name, "done",
              f"{review.verdict}, {len(review.comments)} comments" + (" (verdict overridden)" if overridden else ""))
    result.status = "reviewed"
    result.report = {"facts": facts, "tests": tests, "blocking": blocking, "review": review.model_dump(),
                     "dropped_comments": [c.model_dump() for c in dropped], "overridden": overridden}
    return finish(orch, task, result, bids, passed=not dropped)
