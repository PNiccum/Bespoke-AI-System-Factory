"""The container sandbox: no network, no keys, a read-only root, and a time limit that stops the container too.
The command-building tests run anywhere; the live ones need docker and the simple-sandbox image."""

import shutil
import subprocess
from pathlib import Path

import pytest

import factory.tools.sandbox as sandbox
from factory.tools.sandbox import RunTestsIn, SandboxError, pytest_command, run_tests

FIXTURES = Path(__file__).parent / "fixtures"


def has_image():
    if not shutil.which("docker"):
        return False
    r = subprocess.run(["docker", "image", "inspect", "simple-sandbox"], capture_output=True)
    return r.returncode == 0


live = pytest.mark.skipif(not has_image(), reason="needs docker and the simple-sandbox image (sandbox/Dockerfile)")


@pytest.fixture
def runner(monkeypatch):
    def use(name):
        monkeypatch.setenv("SIMPLE_SANDBOX", name)
        monkeypatch.setattr(sandbox.shutil, "which", lambda prog: f"/usr/bin/{prog}")
    return use


def test_docker_runs_with_no_network_no_env_and_a_read_only_root(runner, tmp_path):
    runner("docker")
    cmd, env, name = pytest_command(tmp_path, ["-q"])
    joined = " ".join(cmd)
    for flag in ("--network none", "--read-only", "--cap-drop ALL", "no-new-privileges", "--memory 2g",
                 "--pids-limit 256", f"{tmp_path}:/work"):
        assert flag in joined
    assert env is None and name and cmd[:2] == ["docker", "run"]  # docker passes no host variables by default


def test_apptainer_cleans_the_environment_and_isolates_the_network_only_when_allowed(runner, tmp_path, monkeypatch):
    runner("apptainer")
    cmd, _, _ = pytest_command(tmp_path, [])
    assert "--containall" in cmd and "--cleanenv" in cmd and "--network" not in cmd
    monkeypatch.setenv("SIMPLE_SANDBOX_APPTAINER_NET", "1")
    assert "--network" in pytest_command(tmp_path, [])[0]


def test_an_unknown_or_missing_runner_says_what_to_do(monkeypatch, tmp_path):
    monkeypatch.setenv("SIMPLE_SANDBOX", "vm")
    with pytest.raises(SandboxError, match="use one of"):
        pytest_command(tmp_path, [])
    monkeypatch.setenv("SIMPLE_SANDBOX", "podman")
    monkeypatch.setattr(sandbox.shutil, "which", lambda prog: None)
    with pytest.raises(SandboxError, match="not installed"):
        pytest_command(tmp_path, [])


def hostile_repo(tmp_path):
    repo = tmp_path / "hostile"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_escape.py").write_text('''import os, socket

def test_no_network():
    s = socket.socket()
    s.settimeout(3)
    try:
        s.connect(("1.1.1.1", 80))
        reached = True
    except OSError:
        reached = False
    assert not reached

def test_no_keys():
    assert not [k for k in os.environ if "KEY" in k or "TOKEN" in k]

def test_root_is_read_only():
    try:
        open("/usr/escaped.txt", "w").write("x")
        wrote = True
    except OSError:
        wrote = False
    assert not wrote
''')
    return repo


@live
def test_hostile_tests_cannot_reach_the_network_keys_or_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLE_SANDBOX", "docker")
    monkeypatch.setenv("ARC_API_KEY", "canary-7f3a9c2e4b1d8065")
    report = run_tests(RunTestsIn(repo_path=str(hostile_repo(tmp_path))))
    assert report.ok, report.output_tail
    assert len(report.passed) == 3


@live
def test_a_timeout_stops_the_container_too(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMPLE_SANDBOX", "docker")
    (tmp_path / "slow" / "tests").mkdir(parents=True)
    (tmp_path / "slow" / "tests" / "test_slow.py").write_text("import time\n\ndef test_slow():\n    time.sleep(600)\n")
    report = run_tests(RunTestsIn(repo_path=str(tmp_path / "slow"), timeout_s=3))
    assert report.timed_out
    running = subprocess.run(["docker", "ps", "-q", "--filter", "name=simple-sandbox-"], capture_output=True, text=True)
    assert running.stdout.strip() == ""


@live
def test_the_buggy_fixture_gives_the_same_report_in_a_container(monkeypatch):
    patch = (FIXTURES / "buggy_repo.patch").read_text()
    reports = {}
    for name in ("process", "docker"):
        monkeypatch.setenv("SIMPLE_SANDBOX", name)
        before = run_tests(RunTestsIn(repo_path=str(FIXTURES / "buggy_repo")))
        after = run_tests(RunTestsIn(repo_path=str(FIXTURES / "buggy_repo"), patch=patch))
        reports[name] = (before.passed, before.failed, after.passed, after.failed)
    assert reports["process"] == reports["docker"]
