"""Command line, one subcommand per task type:

    python -m factory fix --repo PATH --issue "text" [--issue-file FILE] [--test NODE ...]
    python -m factory tests --repo PATH --request "Write tests for parse_date in calc/dates.py"
    python -m factory review --repo PATH --patch-file change.diff [--description "text"]

Runs the task end to end, asks on the terminal before any paid call, and writes the bundle to ./exports/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from factory.config import load_config
from factory.export import EXPORTABLE, export_bundle
from factory.llm import LLM, new_run_dir
from factory.llm.gate import SpendRequest
from factory.orchestrator import Event, Orchestrator, Task
from factory.registry import Registry


def ask_on_terminal(req: SpendRequest) -> bool:
    print(f"\n  Paid call needs approval ({req.reason}): {req.role} on {req.model} via {req.provider}, "
          f"up to ${req.estimate_usd:.2f}; this run has spent ${req.run_spent_usd:.2f}.")
    return input("  Approve? [y/N] ").strip().lower() in ("y", "yes")


def show(e: Event) -> None:
    print(f"[{e.step:<8}] {e.agent:<15} {e.status:<8} {e.detail}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m factory")
    sub = ap.add_subparsers(dest="command", required=True)
    fix = sub.add_parser("fix", help="fix a bug in a Python repository")
    fix.add_argument("--repo", required=True)
    fix.add_argument("--issue", help="the bug report")
    fix.add_argument("--issue-file", help="read the bug report from a file")
    fix.add_argument("--test", action="append", default=[], help="a test the fix must turn green (repeatable)")
    tests = sub.add_parser("tests", help="write tests for existing code, scored by the planted bugs they catch")
    tests.add_argument("--repo", required=True)
    tests.add_argument("--request", required=True, help="what to test, naming the file or function")
    review = sub.add_parser("review", help="review a pull request's diff")
    review.add_argument("--repo", required=True, help="the repository the diff applies to")
    review.add_argument("--patch-file", required=True, help="the diff, as git diff prints it")
    review.add_argument("--description", default="", help="the pull request's description")
    for p in (fix, tests, review):
        p.add_argument("--config", default="configs/factory.yaml")
        p.add_argument("--out", default="exports")
    args = ap.parse_args(argv)

    if args.command == "fix":
        issue = Path(args.issue_file).read_text() if args.issue_file else args.issue
        if not issue:
            ap.error("give --issue or --issue-file")
        task = Task(description=issue, repo_path=args.repo, failing_tests=args.test)
    elif args.command == "tests":
        task = Task(description=args.request, repo_path=args.repo, task_type="test_gen")
    else:
        task = Task(description=args.description or "(no description)", repo_path=args.repo, task_type="pr_review",
                    patch=Path(args.patch_file).read_text())
    cfg = load_config(args.config)
    run_dir = new_run_dir(cfg, args.command)
    llm = LLM(cfg, run_dir, approver=ask_on_terminal)
    result = Orchestrator(llm, Registry.load(cfg), on_event=show).run(task)
    print(f"\nResult: {result.status}" + (f" ({result.stop_reason})" if result.stop_reason else ""))
    print(f"Cost: {llm.ledger.summary()}")
    if result.status in EXPORTABLE:
        print(f"Bundle: {export_bundle(result, task, run_dir, Path(args.out))}")
    return 0 if result.status in ("fixed", "tested", "reviewed") else 1


if __name__ == "__main__":
    sys.exit(main())
