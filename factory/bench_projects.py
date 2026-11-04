"""Project benchmark: small tools built from nothing by the whole pipeline, judged by tests no model sees.

The bug-fix benchmark (factory/bench.py) asks for a patch to an existing repository. This one asks for a whole
tool. Each project in bench/projects.yaml carries a request that fixes the tool's interface and a written Plan A,
which stands in for the Interviewer's conversation so every run starts from the same plan. Then the pipeline runs
as it does for a person:

    planning loop -> knowledge loop -> production -> critic loop -> final QA -> deliverable

and plain code judges the result: the acceptance tests in bench/projects/<name>/acceptance/ are copied into a
throwaway copy of the delivered ``output/`` folder and run in the sandbox. A project counts as solved only when the
pipeline delivered it (QA approved) and every acceptance test passes; the outcome also records how far the run got,
so a project that stops in research or fails QA still says where.

    python -m factory.bench_projects                      # every project, free models only
    python -m factory.bench_projects --only wordstats --repeat 3

Paid calls are refused unless --ask is given. Re-design is off unless --redesign is given: it can adopt prompt
changes into ASPIRE itself, which a benchmark run should not do as a side effect. Results land in
bench/results/<time>-projects/ as results.json and summary.md.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from factory.bench import safe_stdout
from factory.config import FactoryConfig, load_config
from factory.llm import LLM, new_run_dir
from factory.llm.gate import SpendRequest, deny_all
from factory.pipeline.blackboard import Project
from factory.pipeline.interview import ProjectPlan, check_plan, render_task
from factory.pipeline.production import OUTPUT
from factory.registry import Registry
from factory.tools.sandbox import RunTestsIn, run_tests

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "bench" / "projects"
AUTHOR = "benchmark"


class ProjectTask(BaseModel):
    name: str
    request: str
    plan: ProjectPlan


class Acceptance(BaseModel):
    accepted: bool
    passed: int
    expected: int  # tests in the acceptance files
    failing: list[str] = []
    why: str = ""


class ProjectOutcome(BaseModel):
    project: str
    trial: int = 1
    reached: str  # the last stage that ran: planning, research, build, qa
    delivered: bool = False
    solved: bool = False  # delivered and every acceptance test passes
    acceptance: Acceptance | None = None
    why: str = ""
    model_calls: int = 0
    tokens: int = 0
    real_usd: float = 0.0
    priced_usd: float = 0.0
    seconds: float = 0.0
    project_dir: str = ""


def load_projects(path: Path) -> list[ProjectTask]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    common = data.get("common", "").strip()
    return [ProjectTask(name=p["name"], request=f"{p['request'].strip()} {common}".strip(),
                        plan=ProjectPlan.model_validate(p["plan"])) for p in data["projects"]]


def seed(project: Project, task: ProjectTask) -> None:
    """The blackboard an approved interview leaves behind, from the written request and Plan A: the request, the
    plan, plan_a.json, task.txt and the approval. The person approved these plans when they approved the task list."""
    if problems := check_plan(task.plan):
        raise ValueError(f"{task.name}: Plan A has problems: {problems}")
    project.board.post("request", "person", {"text": task.request})
    plan = project.board.post("plan", AUTHOR, {"plan": task.plan.model_dump(), "problems": []})
    project.write_file("plan_a.json", json.dumps(task.plan.model_dump(), indent=1), AUTHOR, refs=[plan.id])
    project.write_file("task.txt", render_task(task.request, [], task.plan), AUTHOR, refs=[plan.id])
    project.board.post("approval", "person", {"plan": plan.id, "source": "bench/projects.yaml"}, refs=[plan.id])


def expected_tests(acceptance: Path) -> int:
    return sum(len(re.findall(r"^def test_", p.read_text(encoding="utf-8"), re.M))
               for p in acceptance.glob("test_*.py"))


def accept(output: Path, name: str, projects_dir: Path = PROJECTS) -> Acceptance:
    """Run the project's acceptance tests against a copy of ``output`` (the delivered folder)."""
    source = projects_dir / name / "acceptance"
    expected = expected_tests(source)
    if not output.is_dir() or not any(output.iterdir()):
        return Acceptance(accepted=False, passed=0, expected=expected, why="no output")
    with tempfile.TemporaryDirectory(prefix="aspire-acceptance-") as tmp:
        overlay = Path(tmp) / "overlay"
        shutil.copytree(source, overlay / "acceptance")
        report = run_tests(RunTestsIn(repo_path=str(output), tests=["acceptance"], timeout_s=300), overlay=overlay)
    bad = report.failed + report.errors  # only the acceptance tests ran
    ok = not bad and not report.timed_out and len(report.passed) >= expected > 0
    why = "timed out" if report.timed_out else "" if ok else (report.output_tail[-600:] or f"exit {report.exit_code}")
    return Acceptance(accepted=ok, passed=len(report.passed), expected=expected, failing=bad, why=why)


def run_project(cfg: FactoryConfig, task: ProjectTask, *, approver: Callable[[SpendRequest], bool] = deny_all,
                providers: dict | None = None, redesign: bool = False, trial: int = 1,
                projects_dir: Path = PROJECTS, registry_dir: Path | None = None) -> ProjectOutcome:
    from factory.pipeline.knowledge import KnowledgeLoop, default_searchers
    from factory.pipeline.planning import PlanningLoop
    from factory.pipeline.run import build
    from factory.pipeline.wiki import Wiki

    run_dir = new_run_dir(cfg, f"projects-{task.name}")
    llm = LLM(cfg, run_dir, approver=approver, **({"providers": providers} if providers else {}))
    registry = Registry.load(cfg, registry_dir)
    project = Project.create(cfg.work_dir / "projects", f"bench {task.name}")
    seed(project, task)
    wiki = Wiki(cfg.work_dir / "wiki")
    searchers = default_searchers(cfg.work_dir / "research-cache")
    out = ProjectOutcome(project=task.name, trial=trial, reached="planning", project_dir=str(project.root))
    t0 = time.monotonic()
    try:
        planned = PlanningLoop(llm, registry, project).run()
        if planned.problems:
            out.why = f"plan problems: {'; '.join(planned.problems)}"
        else:
            out.reached = "research"
            knowledge = KnowledgeLoop(llm, registry, project, wiki, searchers).run()
            if not knowledge.ok:
                out.why = f"knowledge not approved; missing: {', '.join(knowledge.missing) or 'unknown'}"
            else:
                out.reached = "build"
                result = build(llm, registry, project, wiki, searchers, redesign=redesign,
                               on_stage=lambda s: setattr(out, "reached", "qa" if s == "final QA" else out.reached))
                out.delivered = result.status == "delivered"
                out.why = "delivered" if out.delivered else \
                    f"{result.status}: {'; '.join(result.final.issues) if result.final else 'no QA result'}"[:1000]
    except Exception as e:  # noqa: BLE001 (a crash, or a refused paid call, is a failed project, not a failed run)
        out.why = f"{type(e).__name__}: {e}"[:1000]
    out.seconds = round(time.monotonic() - t0, 1)
    if out.reached in ("build", "qa"):  # judge whatever was built, delivered or not, to see how close it came
        out.acceptance = accept(project.root / OUTPUT, task.name, projects_dir)
        out.solved = out.delivered and out.acceptance.accepted
    spend = llm.ledger.summary()
    out.model_calls, out.tokens = spend["calls"], spend["tokens"]
    out.real_usd, out.priced_usd = spend["real_usd"], spend["priced_usd"]
    return out


def summarize(outcomes: list[ProjectOutcome]) -> str:
    lines = ["| Project | Trial | Reached | Delivered | Acceptance | Solved | Why | Model calls | Tokens | Real $ | "
             "Minutes |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in outcomes:
        acc = f"{o.acceptance.passed}/{o.acceptance.expected}" if o.acceptance else "-"
        why = o.why.replace("|", "/").replace("\n", " ")[:160]
        lines.append(f"| {o.project} | {o.trial} | {o.reached} | {'yes' if o.delivered else 'no'} | {acc} | "
                     f"{'yes' if o.solved else 'no'} | {why} | {o.model_calls} | {o.tokens:,} | {o.real_usd:.4f} | "
                     f"{o.seconds / 60:.1f} |")
    if outcomes:
        n = len(outcomes)
        lines += ["", f"- **solved**: {sum(o.solved for o in outcomes)}/{n} (delivered "
                      f"{sum(o.delivered for o in outcomes)}/{n}, every acceptance test passing "
                      f"{sum(bool(o.acceptance and o.acceptance.accepted) for o in outcomes)}/{n}), "
                      f"{sum(o.tokens for o in outcomes):,} tokens, ${sum(o.real_usd for o in outcomes):.4f} real "
                      f"spend, ${sum(o.priced_usd for o in outcomes):.4f} if every model were paid"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m factory.bench_projects")
    ap.add_argument("--projects", default="bench/projects.yaml")
    ap.add_argument("--only", action="append", default=[], help="run only this project (repeatable)")
    ap.add_argument("--config", default="configs/factory.yaml")
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--ask", action="store_true", help="ask on the terminal before paid calls (default: refuse)")
    ap.add_argument("--redesign", action="store_true", help="run re-design on a failed project (off by default)")
    ap.add_argument("--repeat", type=int, default=1, help="trials per project")
    ap.add_argument("--parallel", type=int, default=1,
                    help="projects run at once (default 1); ARC calls stay under the provider's max_concurrency")
    args = ap.parse_args(argv)
    safe_stdout()

    cfg = load_config(args.config)
    tasks = [t for t in load_projects(Path(args.projects)) if not args.only or t.name in args.only]
    approver = deny_all
    if args.ask:
        from factory.__main__ import ask_on_terminal
        approver = ask_on_terminal
    out = Path(args.out) / f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-projects"
    out.mkdir(parents=True)
    outcomes: list[ProjectOutcome] = []
    jobs = [(trial, task) for trial in range(1, args.repeat + 1) for task in tasks]

    def record(o: ProjectOutcome) -> None:
        outcomes.append(o)  # saved before anything is printed, so a console error cannot lose a result
        outcomes.sort(key=lambda x: (x.trial, [t.name for t in tasks].index(x.project)))
        (out / "results.json").write_text(json.dumps([x.model_dump() for x in outcomes], indent=1), encoding="utf-8")
        (out / "summary.md").write_text(summarize(outcomes), encoding="utf-8")
        acc = f"{o.acceptance.passed}/{o.acceptance.expected} acceptance" if o.acceptance else "no acceptance run"
        print(f"{o.project:<16} {'solved' if o.solved else 'failed':<7} reached {o.reached}, {acc}: {o.why[:200]}"
              f"  ({o.model_calls} calls, {o.tokens:,} tokens, ${o.real_usd:.4f}, {o.seconds / 60:.1f} min)",
              flush=True)

    if args.parallel <= 1:
        for trial, task in jobs:
            record(run_project(cfg, task, approver=approver, redesign=args.redesign, trial=trial))
    else:  # projects are independent; they share the wiki (locked) and the provider's concurrency gate
        with ThreadPoolExecutor(max_workers=args.parallel, thread_name_prefix="project") as pool:
            futures = [pool.submit(run_project, cfg, task, approver=approver, redesign=args.redesign, trial=trial)
                       for trial, task in jobs]
            for f in as_completed(futures):
                record(f.result())
    print(f"\n{summarize(outcomes)}\nResults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
