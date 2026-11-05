"""Command line, one subcommand per task type:

    python -m factory fix --repo PATH --issue "text" [--issue-file FILE] [--test NODE ...]
    python -m factory tests --repo PATH --request "Write tests for parse_date in calc/dates.py"
    python -m factory review --repo PATH --patch-file change.diff [--description "text"]
    python -m factory ask --repo PATH --request "any request in plain words" [--patch-file change.diff]
    python -m factory interview --request "what you want built"
    python -m factory plan --project workspace/factory/projects/<id>
    python -m factory research --project workspace/factory/projects/<id>
    python -m factory build --project workspace/factory/projects/<id>

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

    request = Path(args.request_file).read_text(encoding="utf-8") if args.request_file else args.request
    if not request:
        raise SystemExit("give --request or --request-file")
    cfg = load_config(args.config)
    project = Project.create(cfg.work_dir / "projects", request)
    interview = Interview(LLM(cfg, new_run_dir(cfg, "interview"), approver=ask_on_terminal), Registry.load(cfg),
                          project)
    answers = Path(args.answers_file).read_text(encoding="utf-8") if args.answers_file else None
    turn = interview.start(request)
    while answers is not None:  # unattended: every round of questions gets the same written answers
        if turn.kind == "question":
            print("Questions: " + " | ".join(turn.data["questions"]))
            turn = interview.reply(answers)
            continue
        print(json.dumps(turn.data["plan"], indent=1))
        if turn.data["problems"] or not args.approve:
            for problem in turn.data["problems"]:
                print(f"  problem: {problem}")
            print(f"Plan A not approved; project left at {project.root}")
            return 1
        print(f"Approved (--approve). Task written to {interview.approve()}")
        print(f"Next: python -m factory plan --project {project.root}")
        return 0
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
            print(f"Next: python -m factory plan --project {project.root}")
            return 0
        turn = interview.reply(answer)


def run_planning(args) -> int:
    """The planning loop on an approved project: critique, Plan B, pros and cons, edits, vote."""
    from factory.pipeline.blackboard import Project
    from factory.pipeline.planning import PlanningLoop

    cfg = load_config(args.config)
    project = Project.open(Path(args.project))
    llm = LLM(cfg, new_run_dir(cfg, "planning"), approver=ask_on_terminal)
    outcome = PlanningLoop(llm, Registry.load(cfg), project).run()
    print(f"Plan {outcome.winner} won: {outcome.votes}")
    for problem in outcome.problems:
        print(f"  problem left: {problem}")
    print(f"Refined plan: {project.root / 'plan.json'}  Cost: {llm.ledger.summary()}")
    print(f"Next: python -m factory research --project {project.root}")
    return 0 if not outcome.problems else 1


def run_research(args) -> int:
    """The knowledge loop: topics the plan needs, wiki coverage, research, cited and verified wiki pages."""
    from factory.pipeline.blackboard import Project
    from factory.pipeline.knowledge import KnowledgeLoop, default_scout, default_searchers
    from factory.pipeline.wiki import Wiki

    cfg = load_config(args.config)
    project = Project.open(Path(args.project))
    llm = LLM(cfg, new_run_dir(cfg, "research"), approver=ask_on_terminal)
    registry = Registry.load(cfg)
    loop = KnowledgeLoop(llm, registry, project, Wiki(cfg.work_dir / "wiki"),
                         default_searchers(cfg.work_dir / "research-cache"),
                         scout=default_scout(llm, registry, cfg.work_dir / "research-cache",
                                             Wiki(cfg.work_dir / "wiki"), project.root.name))
    result = loop.run(again=args.again)
    for topic, slug in result.covered.items():
        print(f"  covered: {topic} ({cfg.work_dir / 'wiki' / slug}.md)")
    for topic in result.missing:
        print(f"  missing: {topic}")
    for topic, why in result.gaps.items():
        print(f"  gap (the build goes ahead without it): {topic} ({why})")
    print(f"{'Knowledge approved' if result.ok else 'Knowledge incomplete'} after {result.rounds} round(s). "
          f"Cost: {llm.ledger.summary()}")
    if result.ok:
        print(f"Next: python -m factory build --project {project.root}")
    return 0 if result.ok else 1


def run_build(args) -> int:
    """Production, the critic loop, final QA; re-design when the project still fails. Paid calls ask first."""
    from factory.pipeline.blackboard import Project
    from factory.pipeline.knowledge import default_searchers
    from factory.pipeline.run import build
    from factory.pipeline.wiki import Wiki

    cfg = load_config(args.config)
    project = Project.open(Path(args.project))
    llm = LLM(cfg, new_run_dir(cfg, "build"), approver=ask_on_terminal)
    result = build(llm, Registry.load(cfg), project, Wiki(cfg.work_dir / "wiki"),
                   default_searchers(cfg.work_dir / "research-cache"), on_stage=lambda s: print(f"[{s}]"),
                   allow_unapproved=args.allow_unapproved)
    if result.status == "delivered":
        print(f"Delivered: {result.final.deliverable}")
    else:
        print(f"Not delivered after {result.rounds} round(s). Re-design report: {result.redesign.report}")
    print(f"Cost: {llm.ledger.summary()}")
    return 0 if result.status == "delivered" else 1


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
    ask = sub.add_parser("ask", help="say what you want in plain words; ASPIRE picks the workflow")
    ask.add_argument("--repo", required=True)
    ask.add_argument("--request", required=True)
    ask.add_argument("--patch-file", help="a diff, when the request is about one")
    interview = sub.add_parser("interview", help="talk through a request until there is a plan you approve")
    interview.add_argument("--request")
    interview.add_argument("--request-file", help="read the request from a file")
    interview.add_argument("--answers-file", help="answer every question from this file instead of the terminal")
    interview.add_argument("--approve", action="store_true",
                           help="with --answers-file: approve Plan A if code finds no problems with it")
    interview.add_argument("--config", default="configs/factory.yaml")
    plan = sub.add_parser("plan", help="run the planning loop on a project whose Plan A is approved")
    plan.add_argument("--project", required=True, help="the project folder the Interviewer created")
    plan.add_argument("--config", default="configs/factory.yaml")
    research = sub.add_parser("research", help="run the knowledge loop on a project whose plan is decided")
    research.add_argument("--project", required=True)
    research.add_argument("--again", action="store_true",
                          help="run again on a project whose knowledge is already approved (its new result replaces it)")
    research.add_argument("--config", default="configs/factory.yaml")
    build = sub.add_parser("build", help="build, critique, check and deliver a project whose knowledge is approved")
    build.add_argument("--project", required=True)
    build.add_argument("--config", default="configs/factory.yaml")
    build.add_argument("--allow-unapproved", action="store_true",
                       help="build even though the knowledge loop did not approve the knowledge (not a fair run)")
    restart = sub.add_parser("restart", help="start a project again from its approved Plan A, in a new project "
                             "folder (planning, research and build then run fresh)")
    restart.add_argument("--project", required=True)
    restart.add_argument("--config", default="configs/factory.yaml")
    wiki = sub.add_parser("wiki", help="rewrite the wiki's index.md and list problems (broken links, orphans, "
                          "unverified pages); no model calls")
    wiki.add_argument("--config", default="configs/factory.yaml")
    for p in (fix, tests, review, ask):
        p.add_argument("--config", default="configs/factory.yaml")
        p.add_argument("--out", default="exports")
    args = ap.parse_args(argv)
    if args.command == "interview":
        return run_interview(args)
    if args.command == "plan":
        return run_planning(args)
    if args.command == "research":
        return run_research(args)
    if args.command == "build":
        return run_build(args)
    if args.command == "restart":
        from factory.pipeline.blackboard import Project
        src = Project.open(Path(args.project))
        new = Project.restart(src, src.root.parent)
        print(f"new project: {new.root}\nnext: python -m factory plan --project {new.root}")
        return 0
    if args.command == "wiki":
        from factory.pipeline.wiki import Wiki
        w = Wiki(load_config(args.config).work_dir / "wiki")
        print(f"index: {w.write_index()}")
        problems = w.lint()
        print("\n".join(problems) or "no problems")
        return 0

    if args.command == "fix":
        issue = Path(args.issue_file).read_text(encoding="utf-8") if args.issue_file else args.issue
        if not issue:
            ap.error("give --issue or --issue-file")
        task = Task(description=issue, repo_path=args.repo, failing_tests=args.test,
                    allow_test_edits=args.allow_test_edits)
    elif args.command == "tests":
        task = Task(description=args.request, repo_path=args.repo, task_type="test_gen")
    elif args.command == "review":
        task = Task(description=args.description or "(no description)", repo_path=args.repo, task_type="pr_review",
                    patch=Path(args.patch_file).read_text(encoding="utf-8"))
    cfg = load_config(args.config)
    run_dir = new_run_dir(cfg, args.command)
    llm = LLM(cfg, run_dir, approver=ask_on_terminal)
    registry = Registry.load(cfg)
    if args.command == "ask":
        routed = route(args.request, args.repo, llm, registry,
                       patch=Path(args.patch_file).read_text(encoding="utf-8") if args.patch_file else "",
                       gaps=cfg.work_dir / "gaps.jsonl")
        print(f"Routed by {routed.routed_by}: {routed.intake.task_type} ({routed.intake.why})")
        for p in routed.problems:
            print(f"  note: {p}")
        if routed.task is None:
            print(f"ASPIRE has no workflow for this yet; recorded in {cfg.work_dir / 'gaps.jsonl'}.")
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
