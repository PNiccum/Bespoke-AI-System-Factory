"""Run a repository's tests on a throwaway copy, optionally with a patch applied. No model.

Isolation here is a fresh copy of the repository in a temporary directory, an environment with every key and token
removed, and a time limit; the original checkout is never written to. It is not a container: generated code still
runs as the current user, which is why the factory only runs code from tasks a person submitted.

``prove_fix`` is the bug-fix check: the named tests must fail without the patch and pass with it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from pydantic import BaseModel, Field

COPY_IGNORE = shutil.ignore_patterns(".git", ".venv*", "venv", "__pycache__", "*.pyc", ".pytest_cache", "node_modules",
                                     ".env", ".env.*")
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


class RunTestsIn(BaseModel):
    repo_path: str = Field(description="Root of the repository")
    patch: str | None = Field(None, description="Unified diff to apply before running, or none")
    tests: list[str] = Field([], description="pytest node ids or paths; empty runs the whole suite")
    timeout_s: float = Field(300, gt=0, le=3600)


class TestReport(BaseModel):
    patch_applied: bool | None  # None when no patch was given
    exit_code: int
    passed: list[str] = []
    failed: list[str] = []  # assertion failures
    errors: list[str] = []  # errors while collecting or setting up
    skipped: int = 0
    seconds: float
    timed_out: bool = False
    output_tail: str = ""  # last lines of pytest's output, for the coder and the person reading the run

    @property
    def ok(self) -> bool:
        return self.patch_applied is not False and not self.timed_out and self.exit_code == 0


class FixProof(BaseModel):
    before: TestReport
    after: TestReport

    @property
    def proven(self) -> bool:
        """The tests failed without the patch and pass with it."""
        return bool(self.before.failed or self.before.errors) and self.after.ok


def scrubbed_env() -> dict[str, str]:
    """The current environment without anything that looks like a credential."""
    return {k: v for k, v in os.environ.items() if not any(m in k.upper() for m in SECRET_MARKERS)}


def apply_patch(work: Path, patch: str) -> tuple[bool, str]:
    """Apply a unified diff. Models often get hunk line counts wrong, so a diff that fails as written is tried once
    more with the counts recomputed (``--recount``, which only suits single-file diffs: it would read the next
    file's ``---`` header as a removed line)."""
    errors = ""
    for extra in ([], ["--recount"]):
        r = subprocess.run(["git", "apply", *extra, "--whitespace=nowarn", "-"], cwd=work, input=patch, text=True,
                           capture_output=True)
        if r.returncode == 0:
            return True, ""
        errors = r.stderr
    return False, errors


def node_id(case: ET.Element) -> str:
    """pytest's own id for a test (``tests/test_x.py::TestC::test_y``), rebuilt from a JUnit testcase."""
    file, classname, name = case.get("file"), case.get("classname", ""), case.get("name", "")
    if not file:
        return f"{classname}::{name}"
    module = file[:-3].replace("/", ".").replace("\\", ".")
    rest = classname[len(module) + 1:] if classname.startswith(module) else ""
    return "::".join(p for p in (file.replace("\\", "/"), rest, name) if p)


def read_junit(path: Path) -> tuple[list[str], list[str], list[str], int]:
    passed, failed, errors, skipped = [], [], [], 0
    if not path.exists():
        return passed, failed, errors, skipped
    for case in ET.parse(path).getroot().iter("testcase"):
        node = node_id(case)
        if case.find("failure") is not None:
            failed.append(node)
        elif case.find("error") is not None:
            errors.append(node)
        elif case.find("skipped") is not None:
            skipped += 1
        else:
            passed.append(node)
    return passed, failed, errors, skipped


def run_tests(inp: RunTestsIn) -> TestReport:
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="factory-tests-") as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(Path(inp.repo_path).resolve(), work, ignore=COPY_IGNORE)
        applied = None
        if inp.patch:
            applied, why = apply_patch(work, inp.patch)
            if not applied:
                return TestReport(patch_applied=False, exit_code=-1, seconds=round(time.monotonic() - t0, 2),
                                  output_tail=why[-2000:])
        junit = Path(tmp) / "junit.xml"
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}",
               "-o", "junit_family=xunit1", *inp.tests]  # xunit1 records each test's file
        try:
            r = subprocess.run(cmd, cwd=work, env=scrubbed_env(), capture_output=True, text=True,
                               timeout=inp.timeout_s)
            code, out, timed_out = r.returncode, r.stdout + r.stderr, False
        except subprocess.TimeoutExpired as e:
            code, out, timed_out = -1, (e.stdout or "") if isinstance(e.stdout, str) else "", True
        passed, failed, errors, skipped = read_junit(junit)
        return TestReport(patch_applied=applied, exit_code=code, passed=passed, failed=failed, errors=errors,
                          skipped=skipped, seconds=round(time.monotonic() - t0, 2), timed_out=timed_out,
                          output_tail="\n".join(out.splitlines()[-40:]))


def prove_fix(repo_path: str, patch: str, tests: list[str], timeout_s: float = 300) -> FixProof:
    before = run_tests(RunTestsIn(repo_path=repo_path, tests=tests, timeout_s=timeout_s))
    after = run_tests(RunTestsIn(repo_path=repo_path, patch=patch, tests=tests, timeout_s=timeout_s))
    return FixProof(before=before, after=after)
