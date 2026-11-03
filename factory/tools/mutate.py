"""Score a set of tests by the bugs they catch: mutation testing in plain code, no model.

A mutant is the target code with one small change a real bug might make: ``+`` for ``-``, ``<`` for ``>=``,
``and`` for ``or``, a constant off by one, a ``return`` of ``None``. A test suite that still passes with a mutant in
place did not notice the change, so the share of mutants it "kills" measures whether the tests check behaviour or
only run the code. This is how the factory judges generated tests without asking a model whether they are good.

This module uses only the standard library, so an exported bundle carries a copy and recomputes the score itself.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:  # present inside ASPIRE; absent in an exported bundle, which carries this file alone
    from factory.tools.sandbox import run_pytest
except ImportError:
    run_pytest = None

COPY_IGNORE = shutil.ignore_patterns(".git", ".venv*", "venv", "__pycache__", "*.pyc", ".pytest_cache", ".env",
                                     ".env.*")
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
SWAPS = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.FloorDiv: ast.Mult,
         ast.Mod: ast.FloorDiv, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE, ast.GtE: ast.Lt,
         ast.Gt: ast.LtE, ast.LtE: ast.Gt, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot,
         ast.IsNot: ast.Is, ast.And: ast.Or, ast.Or: ast.And}


@dataclass
class Mutant:
    path: str  # relative to the repository root
    line: int
    change: str  # e.g. "+ -> -", "31 -> 32", "return -> return None"
    source: str = field(repr=False)  # the whole mutated file


@dataclass
class ScoreIn:
    repo_path: str
    files: list[str]  # files to mutate, relative to the repository root
    tests: list[str]  # pytest paths or node ids to run against each mutant
    functions: list[str] = field(default_factory=list)  # mutate only inside these; empty means the whole file
    patch: str | None = None  # applied first, e.g. the generated tests
    limit: int = 12
    timeout_s: float = 60


@dataclass
class MutationScore:
    total: int
    killed: int
    survivors: list[dict]  # {"path", "line", "change"} for each mutant the tests missed
    killed_list: list[dict]
    error: str = ""  # set when the tests could not be run at all

    @property
    def score(self) -> float:
        return self.killed / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "score": round(self.score, 3)}


def _sites(tree: ast.AST, functions: list[str]):
    """(node, kind) for every place a mutation can go, in source order, restricted to the named functions."""
    scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
              and n.name in functions] if functions else [tree]
    seen = set()
    for scope in scopes:
        for node in ast.walk(scope):
            if id(node) in seen:
                continue
            seen.add(id(node))
            if isinstance(node, (ast.BinOp, ast.BoolOp)) and type(node.op) in SWAPS:
                yield node, "op"
            elif isinstance(node, ast.Compare):
                for i, op in enumerate(node.ops):
                    if type(op) in SWAPS:
                        yield (node, i), "cmp"
            elif isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
                yield node, "const"
            elif isinstance(node, ast.Return) and node.value is not None and not (
                    isinstance(node.value, ast.Constant) and node.value.value is None):
                yield node, "return"


def _apply(site, kind) -> tuple[str, object]:
    """Mutate one site in place; returns (description, undo)."""
    if kind == "op":
        old = site.op
        site.op = SWAPS[type(old)]()
        return f"{_sym(old)} -> {_sym(site.op)}", lambda: setattr(site, "op", old)
    if kind == "cmp":
        node, i = site
        old = node.ops[i]
        node.ops[i] = SWAPS[type(old)]()
        return f"{_sym(old)} -> {_sym(node.ops[i])}", lambda: node.ops.__setitem__(i, old)
    if kind == "const":
        old = site.value
        site.value = (not old) if isinstance(old, bool) else old + 1
        return f"{old!r} -> {site.value!r}", lambda: setattr(site, "value", old)
    old = site.value
    site.value = ast.Constant(None)
    return "return -> return None", lambda: setattr(site, "value", old)


def _sym(op) -> str:
    names = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%",
             ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.GtE: ">=", ast.Gt: ">", ast.LtE: "<=",
             ast.In: "in", ast.NotIn: "not in", ast.Is: "is", ast.IsNot: "is not", ast.And: "and", ast.Or: "or"}
    return names.get(type(op), type(op).__name__)


def mutants(path: str, source: str, functions: list[str] = (), limit: int = 12) -> list[Mutant]:
    """Up to ``limit`` single-change mutants of one file, spread evenly over its mutation sites."""
    tree = ast.parse(source)
    sites = list(_sites(tree, list(functions)))
    if len(sites) > limit:  # an even spread, so the same file always gives the same mutants
        step = len(sites) / limit
        sites = [sites[int(i * step)] for i in range(limit)]
    out = []
    for site, kind in sites:
        change, undo = _apply(site, kind)
        line = (site[0] if kind == "cmp" else site).lineno
        out.append(Mutant(path, line, change, ast.unparse(tree)))
        undo()
    return out


def scrubbed_env() -> dict[str, str]:
    """No credentials, and no bytecode cache: a mutant the same size as the last, written within the same second,
    would otherwise run the stale cached module."""
    env = {k: v for k, v in os.environ.items() if not any(m in k.upper() for m in SECRET_MARKERS)}
    return env | {"PYTHONDONTWRITEBYTECODE": "1"}


def _git_apply(work: Path, patch: str) -> bool:
    for extra in ([], ["--recount"]):
        if subprocess.run(["git", "apply", *extra, "--whitespace=nowarn", "-"], cwd=work, input=patch, text=True, encoding="utf-8", errors="replace",
                          capture_output=True).returncode == 0:
            return True
    return False


def _passes(work: Path, tests: list[str], timeout_s: float) -> bool:
    """Whether the tests pass on ``work`` (a ``repo`` folder). Inside ASPIRE they run in the configured sandbox
    (factory/tools/sandbox.py), like every other test run; a bundle's standalone copy runs them as a child process.
    A timeout (an infinite loop) counts as a caught mutant."""
    args = ["-q", "-x", "-p", "no:cacheprovider", *tests]
    if run_pytest is not None:
        code, _, timed_out = run_pytest(work.parent, args, timeout_s)
        return code == 0 and not timed_out
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", *args], cwd=work, env=scrubbed_env(),
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def score_tests(inp: ScoreIn) -> MutationScore:
    """Run ``inp.tests`` against each mutant of ``inp.files`` on a throwaway copy of the repository."""
    root = Path(inp.repo_path).resolve()
    with tempfile.TemporaryDirectory(prefix="factory-mutants-") as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(root, work, ignore=COPY_IGNORE)
        if inp.patch and not _git_apply(work, inp.patch):
            return MutationScore(0, 0, [], [], error="the test patch did not apply")
        if not _passes(work, inp.tests, inp.timeout_s):
            return MutationScore(0, 0, [], [], error="the tests fail on the unmutated code")
        per_file = max(1, inp.limit // max(1, len(inp.files)))
        killed, survived = [], []
        for rel in inp.files:
            original = (work / rel).read_text(encoding="utf-8")
            for m in mutants(rel, original, inp.functions, per_file):
                (work / rel).write_text(m.source, encoding="utf-8")
                (survived if _passes(work, inp.tests, inp.timeout_s) else killed).append(
                    {"path": m.path, "line": m.line, "change": m.change})
            (work / rel).write_text(original, encoding="utf-8")
        return MutationScore(len(killed) + len(survived), len(killed), survived, killed)


if __name__ == "__main__":  # python mutate.py REPO FILE[,FILE] TEST[,TEST] [FUNCTION,...]
    args = sys.argv[1:]
    result = score_tests(ScoreIn(repo_path=args[0], files=args[1].split(","), tests=args[2].split(","),
                                 functions=args[3].split(",") if len(args) > 3 else []))
    print(result.to_dict())
