"""Benchmark: SIMPLE against a single model call, on the same bug-fix tasks and the same coder model.

The baseline is what someone would do without SIMPLE: paste the issue and every source file into one prompt and
take the diff that comes back. SIMPLE runs its full workflow (locate, baseline tests, patch, check, cross-family
review, repair). Both answers are judged the same way, by plain code: the patch must apply, leave the tests alone,
and make the whole suite pass on a fresh copy.

    python -m factory.bench                      # every task in bench/tasks.yaml, both modes, free models only
    python -m factory.bench --only leap_year --mode simple

Paid calls are refused unless --ask is given, so a benchmark run costs nothing by default. Results land in
bench/results/<time>/ as results.json and summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from factory.config import FactoryConfig, load_config
from factory.llm import LLM, new_run_dir
from factory.llm.gate import SpendRequest, deny_all
from factory.orchestrator import Orchestrator, Task, extract_diff, protected_paths, touched_tests
from factory.registry import Registry, ScoreBook
from factory.registry.spec import AgentSpec
from factory.tools.sandbox import RunTestsIn, run_tests

Mode = Literal["simple", "baseline"]
MODES: tuple[Mode, ...] = ("simple", "baseline")

BASELINE_PROMPT = """Fix this issue.

## Issue
{issue}

## Repository
{files}

Reply with one unified diff (--- a/path, +++ b/path, @@ hunks) inside a ```diff block and nothing else."""


class BenchTask(BaseModel):
    name: str
    repo: str
    issue: str
    hostile: bool = False  # the repository carries instructions aimed at the model
    heldout: bool = False  # never shown to a model proposing a change to SIMPLE; used only to judge the change


class Outcome(BaseModel):
    task: str
    mode: Mode
    solved: bool
    why: str  # what the judge found
    model_calls: int = 0
    tokens: int = 0
    real_usd: float = 0.0
    seconds: float = 0.0
    attempts: int = 0
    run_dir: str = ""


def load_tasks(path: Path, root: Path) -> list[BenchTask]:
    tasks = [BenchTask(**t) for t in yaml.safe_load(path.read_text())["tasks"]]
    for t in tasks:
        t.repo = str((root / t.repo).resolve())
    return tasks


def judge(repo: str, patch: str | None) -> tuple[bool, str]:
    """Solved means: the patch applies, changes no test or .env file, and the whole suite passes."""
    if not patch:
        return False, "no patch"
    if cheats := touched_tests(patch) + protected_paths(patch, repo):
        return False, f"changes {', '.join(cheats)}"
    report = run_tests(RunTestsIn(repo_path=repo, patch=patch))
    if not report.patch_applied:
        return False, "patch does not apply"
    if not report.ok:
        return False, f"failing: {report.failed + report.errors}" if report.failed or report.errors else "suite failed"
    return True, f"{len(report.passed)} tests pass"


SKIP_PARTS = {".git", "__pycache__", ".venv", ".pytest_cache"}


def repo_files(repo: str, limit: int = 40) -> str:
    """Every Python file in the repository, tests included, as the baseline's one prompt shows them."""
    root = Path(repo)
    paths = sorted(p for p in root.rglob("*.py") if not any(part in SKIP_PARTS for part in p.parts))[:limit]
    return "\n\n".join(f"### {p.relative_to(root).as_posix()}\n```python\n{p.read_text(errors='replace')}```"
                       for p in paths)


def run_baseline(llm: LLM, registry: Registry, task: BenchTask) -> tuple[str | None, int]:
    coder = registry.agents["coder"]
    data = "public" if llm.cfg.data_public else "private"
    reply = llm.complete(coder.role, BASELINE_PROMPT.format(issue=task.issue, files=repo_files(task.repo)),
                         system=coder.system, tag=f"baseline-{task.name}", data=data)
    return extract_diff(reply.text) or None, 1


def run_simple(llm: LLM, registry: Registry, task: BenchTask) -> tuple[str | None, int]:
    result = Orchestrator(llm, registry).run(Task(description=task.issue, repo_path=task.repo))
    return result.patch if result.status == "fixed" else None, result.attempts


RUNNERS: dict[Mode, Callable[[LLM, Registry, BenchTask], tuple[str | None, int]]] = {
    "simple": run_simple, "baseline": run_baseline}


def run_one(cfg: FactoryConfig, task: BenchTask, mode: Mode, *, approver: Callable[[SpendRequest], bool] = deny_all,
            providers: dict | None = None, use_cache: bool = False, registry_dir: Path | None = None,
            scores: Path | None = None, agents: list[AgentSpec] | None = None) -> Outcome:
    run_dir = new_run_dir(cfg, f"bench-{mode}-{task.name}")
    llm = LLM(cfg, run_dir, use_cache=use_cache, approver=approver, **({"providers": providers} if providers else {}))
    registry = Registry.load(cfg, registry_dir)
    if agents is not None:  # a candidate registry under evaluation (factory/extend.py)
        registry = Registry(agents, cfg, registry.scores)
    registry.scores = ScoreBook(scores or run_dir / "scores.json")  # every task starts from the same awards
    t0 = time.monotonic()
    try:
        patch, attempts = RUNNERS[mode](llm, registry, task)
    except Exception as e:  # noqa: BLE001 (a crash is a failed task, not a failed benchmark)
        patch, attempts = None, 0
        crash = f"{type(e).__name__}: {e}"
    else:
        crash = ""
    seconds = time.monotonic() - t0
    solved, why = judge(task.repo, patch)
    spend = llm.ledger.summary()
    return Outcome(task=task.name, mode=mode, solved=solved, why=crash or why, model_calls=spend["calls"],
                   tokens=spend["tokens"], real_usd=spend["real_usd"], seconds=round(seconds, 1), attempts=attempts,
                   run_dir=str(run_dir))


def summarize(outcomes: list[Outcome]) -> str:
    lines = ["| Task | Mode | Solved | Why | Model calls | Tokens | Real $ | Seconds |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {o.task} | {o.mode} | {'yes' if o.solved else 'no'} | {o.why} | {o.model_calls} | {o.tokens:,} | "
              f"{o.real_usd:.4f} | {o.seconds} |" for o in outcomes]
    lines.append("")
    for mode in MODES:
        mine = [o for o in outcomes if o.mode == mode]
        if mine:
            solved = sum(o.solved for o in mine)
            lines.append(f"- **{mode}**: {solved}/{len(mine)} solved, {sum(o.tokens for o in mine):,} tokens, "
                         f"${sum(o.real_usd for o in mine):.4f} real spend")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m factory.bench")
    ap.add_argument("--tasks", default="bench/tasks.yaml")
    ap.add_argument("--mode", choices=MODES, action="append", help="simple, baseline, or both (default)")
    ap.add_argument("--only", action="append", default=[], help="run only this task (repeatable)")
    ap.add_argument("--config", default="configs/factory.yaml")
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--cache", action="store_true", help="reuse cached model replies (off: a benchmark measures)")
    ap.add_argument("--ask", action="store_true", help="ask on the terminal before paid calls (default: refuse)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tasks = [t for t in load_tasks(Path(args.tasks), Path.cwd()) if not args.only or t.name in args.only]
    approver = deny_all
    if args.ask:
        from factory.__main__ import ask_on_terminal
        approver = ask_on_terminal
    out = Path(args.out) / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    outcomes = []
    for task in tasks:
        for mode in args.mode or MODES:
            o = run_one(cfg, task, mode, approver=approver, use_cache=args.cache)
            print(f"{task.name:<16} {mode:<9} {'solved' if o.solved else 'failed':<7} {o.why}  "
                  f"({o.model_calls} calls, {o.tokens:,} tokens, ${o.real_usd:.4f}, {o.seconds}s)")
            outcomes.append(o)
            (out / "results.json").write_text(json.dumps([x.model_dump() for x in outcomes], indent=1))
    (out / "summary.md").write_text(summarize(outcomes))
    print(f"\n{summarize(outcomes)}\nResults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
