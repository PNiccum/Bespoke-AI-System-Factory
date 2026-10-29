"""Command line, one subcommand per task type:

    python -m factory fix --repo PATH --issue "text" [--issue-file FILE] [--test NODE ...]
    python -m factory tests --repo PATH --request "Write tests for parse_date in calc/dates.py"
    python -m factory review --repo PATH --patch-file change.diff [--description "text"]
    python -m factory ask --repo PATH --request "any request in plain words" [--patch-file change.diff]
    python -m factory interview --request "what you want built"

Runs the task end to end, asks on the terminal before any paid call, and writes the bundle to ./exports/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from factory.config import load_config
from factory.export import EXPORTABLE, export_bundle
from factory.llm import LLM, new_run_dir
from factory.llm.gate import SpendRequest
from factory.orchestrator import Event, Orchestrator, Task
from factory.planner import route
from factory.registry import Registry


def ask_on_terminal(req: SpendRequest) -> bool:
    print(f"\n  Paid call needs approval ({req.reason}): {req.role} on {req.model} via {req.provider}, "
          f"up to ${req.estimate_usd:.2f}; this run has spent ${req.run_spent_usd:.2f}.")
    return input("  Approve? [y/N] ").strip().lower() in ("y", "yes")


def show(e: Event) -> None:
    print(f"[{e.step:<8}] {e.agent:<15} {e.status:<8} {e.detail}")


def run_interview(args) -> int:
    """The Interviewer on the terminal: answer its questions, then approve its plan or say what to change."""
    from factory.pipeline.blackboard import Project
    from factory.pipeline.interview import Interview

    cfg = load_config(args.config)
    project = Project.create(cfg.work_dir / "projects", args.request)
    interview = Interview(LLM(cfg, new_run_dir(cfg, "interview"), approver=ask_on_terminal), Registry.load(cfg),
                          project)
    turn = interview.start(args.request)
    while True:
        if turn.data.get("note"):
            print(f"\n{turn.data['note']}")
        if turn.kind == "question":
            for q in turn.data["questions"]:
                print(f"  - {q}")
            turn = interview.reply(input("\nYour answers: "))
            continue
        print(json.dumps(turn.data["plan"], indent=1))
        for problem in turn.data["problems"]:
            print(f"  problem: {problem}")
        answer = input("\nType 'approve', or say what to change: ").strip()
        if answer.lower() == "approve" and not turn.data["problems"]:
            print(f"Approved. Task written to {interview.approve()}")
            return 0
        turn = interview.reply(answer)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m factory")
    sub = ap.add_subparsers(dest="command", required=True)
    fix = sub.add_parser("fix", help="fix a bug in a Python repository")
    fix.add_argument("--repo", required=True)
    fix.add_argument("--issue", help="the bug report")
    fix.add_argument("--issue-file", help="read the bug report from a file")
    fix.add_argument("--test", action="append", default=[], help="a test the fix must turn green (repeatable)")
    fix.add_argument("--allow-test-edits", action="store_true", help="let the fix change test files (off by default)")
    tests = sub.add_parser("tests", help="write tests for existing code, scored by the planted bugs they catch")
    tests.add_argument("--repo", required=True)
    tests.add_argument("--request", required=True, help="what to test, naming the file or function")
    review = sub.add_parser("review", help="review a pull request's diff")
    review.add_argument("--repo", required=True, help="the repository the diff applies to")
    review.add_argument("--patch-file", required=True, help="the diff, as git diff prints it")
    review.add_argument("--description", default="", help="the pull request's description")
    ask = sub.add_parser("ask", help="say what you want in plain words; SIMPLE picks the workflow")
    ask.add_argument("--repo", required=True)
    ask.add_argument("--request", required=True)
    ask.add_argument("--patch-file", help="a diff, when the request is about one")
    interview = sub.add_parser("interview", help="talk through a request until there is a plan you approve")
    interview.add_argument("--request", required=True)
    interview.add_argument("--config", default="configs/factory.yaml")
    for p in (fix, tests, review, ask):
        p.add_argument("--config", default="configs/factory.yaml")
        p.add_argument("--out", default="exports")
    args = ap.parse_args(argv)
    if args.command == "interview":
        return run_interview(args)

    if args.command == "fix":
        issue = Path(args.issue_file).read_text() if args.issue_file else args.issue
        if not issue:
            ap.error("give --issue or --issue-file")
        task = Task(description=issue, repo_path=args.repo, failing_tests=args.test,
                    allow_test_edits=args.allow_test_edits)
    elif args.command == "tests":
        task = Task(description=args.request, repo_path=args.repo, task_type="test_gen")
    elif args.command == "review":
        task = Task(description=args.description or "(no description)", repo_path=args.repo, task_type="pr_review",
                    patch=Path(args.patch_file).read_text())
    cfg = load_config(args.config)
    run_dir = new_run_dir(cfg, args.command)
    llm = LLM(cfg, run_dir, approver=ask_on_terminal)
    registry = Registry.load(cfg)
    if args.command == "ask":
        routed = route(args.request, args.repo, llm, registry,
                       patch=Path(args.patch_file).read_text() if args.patch_file else "",
                       gaps=cfg.work_dir / "gaps.jsonl")
        print(f"Routed by {routed.routed_by}: {routed.intake.task_type} ({routed.intake.why})")
        for p in routed.problems:
            print(f"  note: {p}")
        if routed.task is None:
            print(f"SIMPLE has no workflow for this yet; recorded in {cfg.work_dir / 'gaps.jsonl'}.")
            return 2
        task = routed.task
    result = Orchestrator(llm, registry, on_event=show).run(task)
    print(f"\nResult: {result.status}" + (f" ({result.stop_reason})" if result.stop_reason else ""))
    print(f"Cost: {llm.ledger.summary()}")
    if result.status in EXPORTABLE:
        print(f"Bundle: {export_bundle(result, task, run_dir, Path(args.out))}")
    return 0 if result.status in ("fixed", "tested", "reviewed") else 1


if __name__ == "__main__":
    sys.exit(main())
