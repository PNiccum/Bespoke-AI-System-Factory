"""Run a repository's tests on a throwaway copy, optionally with a patch applied. No model.

Every run uses a fresh copy of the repository in a temporary directory and a time limit; the original checkout is
never written to. Where the tests run is set by ``ASPIRE_SANDBOX`` (from ``sandbox:`` in the config):

- ``process`` (default): a child process with every key and token removed from its environment. Generated code still
  runs as the current user, with the network.
- ``docker`` or ``podman``: a container from ``ASPIRE_SANDBOX_IMAGE`` (``sandbox/Dockerfile``) with no network, an
  empty environment, a read-only root, no capabilities, and caps on memory, CPU and processes.
- ``apptainer``: the same image as a ``.sif`` file, for clusters such as ARC, with ``--containall --cleanenv``. Its
  network is isolated only where the site lets users run ``--net --network none``; set
  ``ASPIRE_SANDBOX_APPTAINER_NET=1`` there.

``prove_fix`` is the bug-fix check: the named tests must fail without the patch and pass with it.
"""

from __future__ import annotations

import os
import re
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
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


RUNNERS = ("process", "docker", "podman", "apptainer")


class SandboxError(RuntimeError):
    """The configured sandbox cannot run: an unknown runner, or its program is not installed."""


def sandbox_runner() -> str:
    runner = os.environ.get("ASPIRE_SANDBOX", "process")
    if runner not in RUNNERS:
        raise SandboxError(f"ASPIRE_SANDBOX={runner!r}; use one of {', '.join(RUNNERS)}")
    if runner != "process" and not shutil.which(runner):
        raise SandboxError(f"ASPIRE_SANDBOX={runner} but {runner} is not installed; install it or use process")
    return runner


def pytest_command(tmp: Path, pytest_args: list[str]) -> tuple[list[str], dict | None, str | None]:
    """The command that runs pytest in ``work``, its environment, and the container's name (to stop it on a
    timeout). ``tmp`` holds the repository copy and receives the junit report."""
    runner = sandbox_runner()
    if runner == "process":
        return [sys.executable, "-m", "pytest", f"--junitxml={tmp / 'junit.xml'}", *pytest_args], scrubbed_env(), None
    image = os.environ.get("ASPIRE_SANDBOX_IMAGE", "aspire-sandbox")
    # env -i: start from an empty environment; base images set variables of their own (python's sets GPG_KEY).
    inner = ["env", "-i", "HOME=/tmp", "PATH=/usr/local/bin:/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE=1",
             "python", "-m", "pytest", "--junitxml=/work/junit.xml", *pytest_args]
    if runner == "apptainer":
        net = ["--net", "--network", "none"] if os.environ.get("ASPIRE_SANDBOX_APPTAINER_NET") == "1" else []
        return ["apptainer", "exec", "--containall", "--cleanenv", "--no-home", *net, "--bind", f"{tmp}:/work",
                "--pwd", "/work/repo", image, *inner], None, None
    name = f"aspire-sandbox-{tmp.name}"
    user = ["--user", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
    limits = os.environ.get("ASPIRE_SANDBOX_LIMITS", "2g,2,256").split(",")  # memory, CPUs, processes
    return [runner, "run", "--rm", "--name", name, "--network", "none", "--read-only", "--tmpfs", "/tmp",  # noqa: S108  # nosec B108 (inside the container)
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--memory", limits[0], "--cpus", limits[1],
            "--pids-limit", limits[2], *user, "-v", f"{tmp}:/work", "-w", "/work/repo", image, *inner], None, name


def run_pytest(tmp: Path, pytest_args: list[str], timeout_s: float) -> tuple[int, str, bool]:
    """Run pytest on ``tmp/repo`` in the configured sandbox: (exit code, output, timed out). The junit report lands
    at ``tmp/junit.xml``."""
    cmd, env, container = pytest_command(tmp, pytest_args)
    if env is not None:
        env = env | {"PYTHONDONTWRITEBYTECODE": "1"}  # a same-size rewrite in the same second would hit a stale .pyc
    try:
        r = subprocess.run(cmd, cwd=tmp / "repo", env=env, capture_output=True, text=True, timeout=timeout_s)
        return r.returncode, r.stdout + r.stderr, False
    except subprocess.TimeoutExpired as e:
        if container:  # stopping the client leaves the container running
            subprocess.run([cmd[0], "rm", "-f", container], capture_output=True)
        return -1, (e.stdout or "") if isinstance(e.stdout, str) else "", True


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
    """The current environment without anything that looks like a credential. Git's environment config
    (GIT_CONFIG_COUNT, GIT_CONFIG_KEY_n, GIT_CONFIG_VALUE_n) goes as a whole: its values can hold auth headers, and
    dropping only the KEY_n entries leaves a count git refuses to run with."""
    return {k: v for k, v in os.environ.items()
            if not any(m in k.upper() for m in SECRET_MARKERS) and not k.upper().startswith("GIT_CONFIG_")}


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
    # Code under test could rewrite this report, but expat 2.4+ (bundled with Python 3.11+) caps entity
    # expansion and never fetches external entities, so the known XML attacks do not apply.
    for case in ET.parse(path).getroot().iter("testcase"):  # noqa: S314  # nosec B314
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


def steady_output(out: str, tmp: Path) -> str:
    """pytest's output without what changes from run to run (the temporary folder, timings), so the same failure
    gives the same text: a model prompt built from it is then the same request, which the response cache and a
    recorded replay depend on."""
    out = out.replace(str(tmp), "<sandbox>").replace("/work/", "<sandbox>/")
    return re.sub(r" in \d+(\.\d+)?s( \(\d+:\d\d:\d\d\))?", "", out)


def run_tests(inp: RunTestsIn, overlay: Path | None = None) -> TestReport:
    """``overlay`` is a folder copied over the repository after the patch: the benchmark's hidden tests, which the
    patch's author never saw. It is not part of RunTestsIn, so no MCP client or model can set it."""
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
        if overlay is not None:
            shutil.copytree(overlay, work, dirs_exist_ok=True, ignore=COPY_IGNORE)
        junit = Path(tmp) / "junit.xml"
        # xunit1 records each test's file
        code, out, timed_out = run_pytest(Path(tmp), ["-q", "-p", "no:cacheprovider", "-o", "junit_family=xunit1",
                                                      *inp.tests], inp.timeout_s)
        passed, failed, errors, skipped = read_junit(junit)
        return TestReport(patch_applied=applied, exit_code=code, passed=passed, failed=failed, errors=errors,
                          skipped=skipped, seconds=round(time.monotonic() - t0, 2), timed_out=timed_out,
                          output_tail="\n".join(steady_output(out, Path(tmp)).splitlines()[-40:]))


def prove_fix(repo_path: str, patch: str, tests: list[str], timeout_s: float = 300) -> FixProof:
    before = run_tests(RunTestsIn(repo_path=repo_path, tests=tests, timeout_s=timeout_s))
    after = run_tests(RunTestsIn(repo_path=repo_path, patch=patch, tests=tests, timeout_s=timeout_s))
    return FixProof(before=before, after=after)
