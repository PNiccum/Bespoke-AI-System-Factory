"""Benchmark: ASPIRE against a single model call, on the same bug-fix tasks and the same coder model.

The baseline is what someone would do without ASPIRE: paste the issue and every source file into one prompt and
take the diff that comes back. ASPIRE runs its full workflow (locate, baseline tests, patch, check, cross-family
review, repair). Both answers are judged the same way, by plain code: the patch must apply, leave the tests alone,
and make the whole suite pass on a fresh copy, with every test that ran before still running.

A solved patch is then judged again with hidden tests (bench/hidden/<task>/), copied in after the patch and never
shown to any model. They catch a patch that special-cases the inputs of the visible tests. ``solved`` is the visible
verdict, the one self-extension's gate uses; ``hidden_solved`` is the strict one.

    python -m factory.bench                      # every task in bench/tasks.yaml, both modes, free models only
    python -m factory.bench --only leap_year --mode aspire
    python -m factory.bench --repeat 3              # three trials of each task, to see how much the models vary

Paid calls are refused unless --ask is given, so a benchmark run costs nothing by default. Results land in
bench/results/<time>/ as results.json and summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from functools import cache
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

Mode = Literal["aspire", "baseline"]
MODES: tuple[Mode, ...] = ("aspire", "baseline")

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
    heldout: bool = False  # never shown to a model proposing a change to ASPIRE; used only to judge the change


class Outcome(BaseModel):
    task: str
    mode: Mode
    solved: bool
    why: str  # what the judge found
    hidden_solved: bool | None = None  # also passes the hidden tests; None when the task has none or it was not solved
    hidden_why: str = ""
    trial: int = 1
    heldout: bool = False
    model_calls: int = 0
    tokens: int = 0
    real_usd: float = 0.0
    priced_usd: float = 0.0  # what the run would have cost if every model were paid (shadow prices for ARC)
    seconds: float = 0.0
    attempts: int = 0
    run_dir: str = ""


def load_tasks(path: Path, root: Path) -> list[BenchTask]:
    tasks = [BenchTask(**t) for t in yaml.safe_load(path.read_text(encoding="utf-8"))["tasks"]]
    for t in tasks:
        t.repo = str((root / t.repo).resolve())
    return tasks


HIDDEN = Path(__file__).resolve().parents[1] / "bench" / "hidden"


@cache
def tests_before(repo: str) -> frozenset[str]:
    """Every test the unpatched repository runs, passing or not."""
    report = run_tests(RunTestsIn(repo_path=repo))
    return frozenset(report.passed + report.failed + report.errors)


def judge(repo: str, patch: str | None) -> tuple[bool, str]:
    """Solved means: the patch applies, changes no test or .env file, the whole suite passes, and every test that ran
    before the patch still runs and passes. The last rule stops a patch that ends pytest early (``os._exit(0)`` at
    import) or switches collection off, which would otherwise pass with nothing run."""
    if not patch:
        return False, "no patch"
    if cheats := touched_tests(patch) + protected_paths(patch, repo):
        return False, f"changes {', '.join(cheats)}"
    report = run_tests(RunTestsIn(repo_path=repo, patch=patch))
    if not report.patch_applied:
        return False, "patch does not apply"
    if not report.ok:
        return False, f"failing: {report.failed + report.errors}" if report.failed or report.errors else "suite failed"
    if missing := sorted(tests_before(repo) - set(report.passed)):
        return False, f"tests did not run: {missing}"
    return True, f"{len(report.passed)} tests pass"


def judge_hidden(task: str, repo: str, patch: str, hidden_root: Path = HIDDEN) -> tuple[bool | None, str]:
    """The patch again, with the task's hidden tests copied in after it. None when the task has no hidden tests."""
    overlay = hidden_root / task
    if not overlay.is_dir():
        return None, "no hidden tests"
    report = run_tests(RunTestsIn(repo_path=repo, patch=patch), overlay=overlay)
    if not report.ok:
        return False, f"hidden failing: {report.failed + report.errors}" if report.failed or report.errors \
            else "hidden suite failed"
    return True, f"{len(report.passed)} tests pass with the hidden ones"


SKIP_PARTS = {".git", "__pycache__", ".venv", ".pytest_cache"}


def repo_files(repo: str, limit: int = 40) -> str:
    """Every Python file in the repository, tests included, as the baseline's one prompt shows them."""
    root = Path(repo)
    paths = sorted(p for p in root.rglob("*.py") if not any(part in SKIP_PARTS for part in p.parts))[:limit]
    return "\n\n".join(f"### {p.relative_to(root).as_posix()}\n```python\n{p.read_text(errors='replace', encoding='utf-8')}```"
                       for p in paths)


def run_baseline(llm: LLM, registry: Registry, task: BenchTask) -> tuple[str | None, int]:
    coder = registry.agents["coder"]
    data = "public" if llm.cfg.data_public else "private"
    reply = llm.complete(coder.role, BASELINE_PROMPT.format(issue=task.issue, files=repo_files(task.repo)),
                         system=coder.system, tag=f"baseline-{task.name}", data=data)
    return extract_diff(reply.text) or None, 1


def run_aspire(llm: LLM, registry: Registry, task: BenchTask) -> tuple[str | None, int]:
    result = Orchestrator(llm, registry).run(Task(description=task.issue, repo_path=task.repo))
    return result.patch if result.status == "fixed" else None, result.attempts


RUNNERS: dict[Mode, Callable[[LLM, Registry, BenchTask], tuple[str | None, int]]] = {
    "aspire": run_aspire, "baseline": run_baseline}


def run_one(cfg: FactoryConfig, task: BenchTask, mode: Mode, *, approver: Callable[[SpendRequest], bool] = deny_all,
            providers: dict | None = None, use_cache: bool = False, registry_dir: Path | None = None,
            scores: Path | None = None, agents: list[AgentSpec] | None = None, trial: int = 1) -> Outcome:
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
    hidden, hidden_why = judge_hidden(task.name, task.repo, patch) if solved and patch else (None, "")
    spend = llm.ledger.summary()
    return Outcome(task=task.name, mode=mode, solved=solved, why=crash or why, hidden_solved=hidden,
                   hidden_why=hidden_why, trial=trial, heldout=task.heldout, model_calls=spend["calls"],
                   tokens=spend["tokens"], real_usd=spend["real_usd"], priced_usd=spend["priced_usd"],
                   seconds=round(seconds, 1), attempts=attempts, run_dir=str(run_dir))


def summarize(outcomes: list[Outcome]) -> str:
    hidden = {True: "yes", False: "no", None: "-"}
    lines = ["| Task | Mode | Trial | Solved | Hidden tests | Why | Model calls | Tokens | Real $ | Seconds |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {o.task} | {o.mode} | {o.trial} | {'yes' if o.solved else 'no'} | {hidden[o.hidden_solved]} | "
              f"{o.hidden_why if o.hidden_solved is False else o.why} | {o.model_calls} | {o.tokens:,} | "
              f"{o.real_usd:.4f} | {o.seconds} |" for o in outcomes]
    lines.append("")
    for mode in MODES:
        mine = [o for o in outcomes if o.mode == mode]
        if mine:
            solved = sum(o.solved for o in mine)
            strict = sum(o.solved and o.hidden_solved is not False for o in mine)
            lines.append(f"- **{mode}**: {solved}/{len(mine)} solved ({strict}/{len(mine)} with the hidden tests), "
                         f"{sum(o.tokens for o in mine):,} tokens, ${sum(o.real_usd for o in mine):.4f} real spend, "
                         f"${sum(o.priced_usd for o in mine):.4f} if every model were paid")
    return "\n".join(lines) + "\n"


def safe_stdout() -> None:
    """Never let a character the console cannot show (a Windows cp1252 console or redirected file, and a model's
    U+2011 in a failure reason) crash a run: print it as '?' instead."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m factory.bench")
    ap.add_argument("--tasks", default="bench/tasks.yaml")
    ap.add_argument("--mode", choices=MODES, action="append", help="aspire, baseline, or both (default)")
    ap.add_argument("--only", action="append", default=[], help="run only this task (repeatable)")
    ap.add_argument("--config", default="configs/factory.yaml")
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--cache", action="store_true", help="reuse cached model replies (off: a benchmark measures)")
    ap.add_argument("--ask", action="store_true", help="ask on the terminal before paid calls (default: refuse)")
    ap.add_argument("--repeat", type=int, default=1, help="trials per task and mode (models vary from run to run)")
    args = ap.parse_args(argv)
    safe_stdout()

    cfg = load_config(args.config)
    tasks = [t for t in load_tasks(Path(args.tasks), Path.cwd()) if not args.only or t.name in args.only]
    approver = deny_all
    if args.ask:
        from factory.__main__ import ask_on_terminal
        approver = ask_on_terminal
    from factory.bench_projects import make_run_dir
    out = make_run_dir(Path(args.out))
    outcomes = []
    for trial in range(1, args.repeat + 1):
        for task in tasks:
            for mode in args.mode or MODES:
                o = run_one(cfg, task, mode, approver=approver, use_cache=args.cache, trial=trial)
                outcomes.append(o)  # saved before anything is printed, so a console error cannot lose a result
                (out / "results.json").write_text(json.dumps([x.model_dump() for x in outcomes], indent=1), encoding="utf-8")
                (out / "summary.md").write_text(summarize(outcomes), encoding="utf-8")
                strict = "" if o.hidden_solved is not False else f" but {o.hidden_why}"
                print(f"{task.name:<16} {mode:<9} {'solved' if o.solved else 'failed':<7} {o.why}{strict}  "
                      f"({o.model_calls} calls, {o.tokens:,} tokens, ${o.real_usd:.4f}, {o.seconds}s)", flush=True)
    print(f"\n{summarize(outcomes)}\nResults: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
