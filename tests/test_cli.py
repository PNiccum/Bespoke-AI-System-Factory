"""The terminal command, with scripted models: each subcommand's happy path, its exit code, and its errors."""

import json
import shutil

import pytest

import factory.__main__ as cli
from factory.config import load_config
from factory.llm import LLM
from tests.test_orchestrator import ACCEPT, GOOD, ISSUE, REPO, ROOT, Scripted, diff


@pytest.fixture
def scripted(monkeypatch, tmp_path):
    """Run the CLI with its workspace in tmp_path and the given ARC replies; returns the ARC provider."""
    arc = Scripted()
    shutil.copytree(ROOT / "registry", tmp_path / "registry")  # the agent specs; the track record starts empty

    def config(path, *a, **kw):
        cfg = load_config(path, *a, **kw)
        cfg.root, cfg.tool_transport = tmp_path, "direct"
        return cfg

    monkeypatch.setattr(cli, "load_config", config)
    monkeypatch.setattr(cli, "LLM", lambda cfg, run_dir, **kw: LLM(
        cfg, run_dir, use_cache=False, providers={"arc": arc, "openai": Scripted(), "claude": Scripted()}, **kw))
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")  # any paid-call question is declined
    return arc


def run(*argv, config=str(ROOT / "configs" / "factory.yaml")):
    return cli.main([*argv, "--config", config])


def test_fix_proves_the_fix_and_writes_a_bundle(scripted, tmp_path, capsys):
    scripted.texts += [diff(GOOD), ACCEPT]
    assert run("fix", "--repo", str(REPO), "--issue", ISSUE, "--out", str(tmp_path / "exports")) == 0
    out = capsys.readouterr().out
    assert "Result: fixed" in out and "Bundle:" in out and list((tmp_path / "exports").glob("bundle-*.zip"))
    assert "[review  ] reviewer" in out  # agent activity is shown step by step


def test_fix_reads_the_issue_from_a_file(scripted, tmp_path, capsys):
    issue = tmp_path / "issue.txt"
    issue.write_text(ISSUE, encoding="utf-8")
    scripted.texts += [diff(GOOD), ACCEPT]
    assert run("fix", "--repo", str(REPO), "--issue-file", str(issue), "--out", str(tmp_path / "x")) == 0
    assert "parse_date" in scripted.requests[0].messages[0]["content"]


def test_fix_without_an_issue_is_a_usage_error(scripted):
    with pytest.raises(SystemExit) as e:
        run("fix", "--repo", str(REPO))
    assert e.value.code == 2


def test_a_declined_paid_step_falls_back_to_free_and_spends_nothing(scripted, tmp_path, capsys):
    scripted.texts += ["no diff here", "still none", "none again"]  # the third would be paid; declined, it runs free
    assert run("fix", "--repo", str(REPO), "--issue", ISSUE, "--out", str(tmp_path / "x")) == 1
    out = capsys.readouterr().out
    assert "Result: failed" in out and "trying once more on the free model" in out and "'real_usd': 0.0" in out and "Bundle:" not in out


def test_ask_records_a_request_no_workflow_covers(scripted, tmp_path, capsys):
    scripted.texts.append(json.dumps({"task_type": "unsupported", "steps": [], "missing": ["deploy"],
                                      "why": "nothing deploys"}))
    assert run("ask", "--repo", str(REPO), "--request", "deploy this to production") == 2
    assert "no workflow for this yet" in capsys.readouterr().out
    gaps = (tmp_path / "workspace" / "factory" / "gaps.jsonl").read_text(encoding="utf-8")
    assert json.loads(gaps)["missing"] == ["deploy"]


def test_ask_routes_a_clear_request_by_rules_with_no_model_call(scripted, capsys, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "Orchestrator", lambda llm, registry, on_event: type(
        "Stub", (), {"run": lambda self, task: seen.append(task) or type(
            "R", (), {"status": "failed", "stop_reason": "stub"})()})())
    assert run("ask", "--repo", str(REPO), "--request", "Write tests for days_in_month in calc/dates.py") == 1
    assert "Routed by rules: test_gen" in capsys.readouterr().out
    assert seen[0].task_type == "test_gen" and scripted.requests == []


def test_interview_runs_unattended_from_written_answers_and_approves(scripted, tmp_path, capsys):
    from tests.test_interview import PLAN, ask, propose
    request, answers = tmp_path / "request.md", tmp_path / "answers.md"
    request.write_text("Design PCR primers for a human gene", encoding="utf-8")
    answers.write_text("Human genes only; a Python script.", encoding="utf-8")
    scripted.texts += [ask("Which species?"), propose(PLAN)]
    assert run("interview", "--request-file", str(request), "--answers-file", str(answers), "--approve") == 0
    assert "Human genes only" in scripted.requests[1].messages[0]["content"]
    assert list((tmp_path).rglob("plan_a.json")) and "Approved (--approve)" in capsys.readouterr().out


def test_unattended_interview_without_approve_leaves_plan_a_unapproved(scripted, tmp_path, capsys):
    from tests.test_interview import PLAN, propose
    answers = tmp_path / "answers.md"
    answers.write_text("x", encoding="utf-8")
    scripted.texts += [propose(PLAN)]
    assert run("interview", "--request", "Design PCR primers", "--answers-file", str(answers)) == 1
    assert not list(tmp_path.rglob("plan_a.json")) and "not approved" in capsys.readouterr().out
