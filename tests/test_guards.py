"""Guards from the risk register: secrets kept out of bundles, the registry version in every bundle, ASPIRE's own
safety code protected from patches, and tests that may not quietly disappear."""

import json
import zipfile

import pytest

from factory.export import export_bundle
from factory.llm import LLM
from factory.llm.gate import approve_all
from factory.orchestrator import ASPIRE_ROOT, Orchestrator, Task, protected_paths
from factory.registry import Registry, ScoreBook
from factory.secrets import SecretFound, guard_bundle, scan_text
from tests.test_orchestrator import ACCEPT, GOOD, ISSUE, REPO, ROOT, Scripted, cfg, diff, run  # noqa: F401

FAKE_KEY = "sk-ant-api03-" + "x" * 40


def test_scan_finds_key_shapes_and_env_values(monkeypatch):
    assert "anthropic key" in scan_text(f"token: {FAKE_KEY}")
    assert "private key" in scan_text("-----BEGIN RSA PRIVATE KEY-----")
    monkeypatch.setenv("ARC_API_KEY", "plainlookingvalue123")
    assert "value of $ARC_API_KEY" in scan_text("error: auth failed for plainlookingvalue123")
    assert scan_text("parse_date swaps month and day") == []


def test_a_bundle_holding_a_secret_is_deleted_not_zipped(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "evidence.json").write_text(json.dumps({"output_tail": f"KEY={FAKE_KEY}"}))
    with pytest.raises(SecretFound, match="evidence.json: anthropic key"):
        guard_bundle(tmp_path / "b")
    assert not (tmp_path / "b").exists()


def test_export_refuses_a_run_whose_records_hold_a_key(cfg, tmp_path, monkeypatch):
    result, _, _, _ = run(cfg, tmp_path, arc=[diff(GOOD), ACCEPT])
    monkeypatch.setenv("OPENAI_API_KEY", "leakedvalue-0123456789")
    (tmp_path / "provenance.jsonl").write_text('{"detail": "leakedvalue-0123456789"}\n')
    with pytest.raises(SecretFound, match="provenance.jsonl"):
        export_bundle(result, Task(description=ISSUE, repo_path=str(REPO)), tmp_path, tmp_path / "out")
    assert not list((tmp_path / "out").glob("*.zip"))


def test_every_bundle_names_the_registry_that_awarded_its_steps(cfg, tmp_path):
    result, _, _, registry = run(cfg, tmp_path, arc=[diff(GOOD), ACCEPT])
    zip_path = export_bundle(result, Task(description=ISSUE, repo_path=str(REPO)), tmp_path, tmp_path / "out")
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith("registry.json"))
        stamp = json.loads(z.read(name))
    assert "coder" in stamp["agents"] and len(stamp["specs_sha256"]) == 64
    assert stamp["scores"] == {}  # the track record as it stood when the steps were awarded, before this run


def test_the_registry_hash_changes_with_any_spec(cfg, tmp_path):
    reg = Registry.load(cfg, ROOT / "registry")
    reg.scores = ScoreBook(tmp_path / "scores.json")
    before = reg.version()["specs_sha256"]
    reg.agents["coder"] = reg.agents["coder"].model_copy(update={"prompt": "something else"})
    assert reg.version()["specs_sha256"] != before


GATE_PATCH = """--- a/factory/llm/gate.py
+++ b/factory/llm/gate.py
@@ -1 +1 @@
-x
+y
"""


def test_simple_may_not_patch_its_own_safety_code():
    assert protected_paths(GATE_PATCH, str(ASPIRE_ROOT)) == ["factory/llm/gate.py"]
    assert protected_paths(GATE_PATCH, str(REPO)) == []  # another repo's factory/ folder is ordinary code
    env_patch = "--- /dev/null\n+++ b/.env\n@@ -0,0 +1 @@\n+K=v\n"
    assert protected_paths(env_patch, str(REPO)) == [".env"]


DROPS_MONEY_TEST = GOOD + """--- a/tests/test_money.py
+++ /dev/null
@@ -1,5 +0,0 @@
-from calc.money import add_tax
-
-
-def test_add_tax():
-    assert add_tax(100) == 105.0
"""


def test_a_fix_that_removes_a_passing_test_fails_the_check(cfg, tmp_path):
    providers = {"arc": Scripted(diff(DROPS_MONEY_TEST), diff(GOOD), ACCEPT), "openai": Scripted(),
                 "claude": Scripted()}
    llm = LLM(cfg, tmp_path, use_cache=False, providers=providers, approver=approve_all)
    registry = Registry.load(cfg, ROOT / "registry")
    registry.scores = ScoreBook(tmp_path / "scores.json")
    events = []
    task = Task(description=ISSUE, repo_path=str(REPO), allow_test_edits=True)
    result = Orchestrator(llm, registry, on_event=events.append).run(task)
    assert result.status == "fixed" and result.attempts == 2
    assert any(e.step == "check" and "fewer tests ran (1 vs 2)" in e.detail for e in events)


def test_bug_fix_rerun_gives_the_tests_no_keys(monkeypatch, tmp_path):
    """Runs the real rerun.py: a generated test that looks for the shell's keys must not find them."""
    import subprocess
    import sys

    from factory.export import RERUN

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "test_env.py").write_text(
        "import os\n\ndef test_no_keys():\n    assert not {'ANTHROPIC_API_KEY', 'HTTPS_PROXY', 'https_proxy', 'GIT_CONFIG_COUNT'} & set(os.environ)\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "rerun.py").write_text(RERUN)
    (bundle / "workflow.json").write_text(json.dumps({"targets": []}))
    (bundle / "fix.patch").write_text("diff --git a/x.txt b/x.txt\nnew file mode 100644\n--- /dev/null\n+++ b/x.txt\n@@ -0,0 +1 @@\n+x\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak-123456")
    monkeypatch.setenv("HTTPS_PROXY", "http://user:pw@proxy.example:8080")
    monkeypatch.setenv("https_proxy", "http://user:pw@proxy.example:8080")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")  # a complete, valid git config so the script's own git calls work
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "user.name")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "x")
    r = subprocess.run([sys.executable, str(bundle / "rerun.py"), str(repo)], capture_output=True, text=True)
    assert r.returncode == 0 and "whole suite: PASS" in r.stdout, r.stdout + r.stderr


def test_review_rerun_gives_its_pytest_no_keys(monkeypatch, tmp_path):
    """Runs the review template's own broken(): a test that looks for the shell's key must not find it."""
    import sys

    from factory.workflows.export import RERUN_REVIEW

    (tmp_path / "diff_inspect.py").write_text("def inspect(*a):\n    return {}\n")  # the bundle ships its own
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "diff_inspect", raising=False)  # undone after the test
    ns: dict = {"__name__": "rerun_under_test", "__file__": str(tmp_path / "rerun.py")}
    exec(compile(RERUN_REVIEW, "rerun.py", "exec"), ns)  # noqa: S102
    repo = tmp_path / "repo"
    repo.mkdir()
    marker = tmp_path / "ran"
    (repo / "test_env.py").write_text(
        f"import os, pathlib\n\ndef test_no_key():\n    pathlib.Path({str(marker)!r}).write_text('x')\n"
        "    assert 'ANTHROPIC_API_KEY' not in os.environ\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak-123456")
    assert ns["broken"](repo, None) == set() and marker.exists()  # the test ran, and did not see the key
    marker.unlink()
    ns["clean_env"] = lambda: dict(ns["os"].environ)  # a regression: no scrubbing
    assert ns["broken"](repo, None) and marker.exists()  # the same test now fails, so this one would notice


def test_mutation_runs_get_the_same_scrubbed_environment(monkeypatch):
    from factory.tools.mutate import scrubbed_env

    monkeypatch.setenv("MY_CREDENTIAL", "abc")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("HTTPS_PROXY", "http://user:pw@proxy.example:8080")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    env = scrubbed_env()
    assert not {"MY_CREDENTIAL", "GIT_CONFIG_COUNT", "HTTPS_PROXY", "ANTHROPIC_API_KEY"} & set(env)
    assert "PATH" in env and env["PYTHONDONTWRITEBYTECODE"] == "1"
