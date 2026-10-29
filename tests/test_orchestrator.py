"""The bug-fix run end to end, with scripted models: plan, award, patch, check, review, repair, escalate."""

import json
from pathlib import Path

import pytest

from factory.config import load_config
from factory.llm import LLM, Usage
from factory.llm.gate import approve_all, deny_all
from factory.llm.providers import RawResult
from factory.orchestrator import Orchestrator, Task, extract_diff, plan_for
from factory.records import ProvenanceLog
from factory.registry import Registry, ScoreBook

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
REPO = FIXTURES / "buggy_repo"
GOOD = (FIXTURES / "buggy_repo.patch").read_text()
WRONG = GOOD.replace("int(year), int(month), int(day)", "int(year), int(day), int(day)")  # applies, still fails
BREAKS_MONEY = GOOD + """--- a/calc/money.py
+++ b/calc/money.py
@@ -1,2 +1,2 @@
 def add_tax(amount, rate=0.05):
-    return round(amount * (1 + rate), 2)
+    return amount
"""
ISSUE = 'parse_date("2026-10-28") returns (2026, 28, 10): month and day are swapped.'
ACCEPT = json.dumps({"verdict": "accept", "defects": []})
REJECT = json.dumps({"verdict": "reject", "defects": [{"line": "+    return", "problem": "drops the day check"}]})


class Scripted:
    def __init__(self, *texts):
        self.texts = list(texts)
        self.requests = []

    def complete(self, req):
        self.requests.append(req)
        return RawResult(self.texts.pop(0), "", "end_turn", Usage(500, 200))


def diff(patch):
    return f"Here is the fix.\n```diff\n{patch}```"


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(ROOT / "configs" / "factory.yaml")
    cfg.root = tmp_path
    cfg.tool_transport = "direct"  # tests/test_mcp.py covers the MCP route
    return cfg


def run(cfg, tmp_path, arc=(), openai=(), approver=approve_all, **kw):
    providers = {"arc": Scripted(*arc), "openai": Scripted(*openai), "claude": Scripted()}
    llm = LLM(cfg, tmp_path, use_cache=False, providers=providers, approver=approver)
    registry = Registry.load(cfg, ROOT / "registry")
    registry.scores = ScoreBook(tmp_path / "scores.json")
    events = []
    result = Orchestrator(llm, registry, on_event=events.append, **kw).run(Task(description=ISSUE,
                                                                                 repo_path=str(REPO)))
    return result, providers, events, registry


def test_a_clean_fix_is_located_patched_proven_and_reviewed(cfg, tmp_path):
    result, p, events, registry = run(cfg, tmp_path, arc=[diff(GOOD), ACCEPT])
    assert result.status == "fixed" and result.attempts == 1 and result.patch == GOOD
    assert result.review.verdict == "accept" and len(result.check.passed) == 2
    done = [(e.step, e.agent) for e in events if e.status == "done"]
    assert done == [("locate", "code_navigator"), ("baseline", "test_runner"), ("patch", "coder"),
                    ("check", "test_runner"), ("review", "reviewer")]
    coder_prompt = p["arc"].requests[0].messages[0]["content"]
    assert "calc/dates.py" in coder_prompt and "tests/test_dates.py::test_parse_date_order" in coder_prompt
    assert registry.scores.pass_rate("coder", "bug_fix") == 1.0
    kinds = [e.kind for e in ProvenanceLog(tmp_path).events()]
    assert kinds[0] == "source" and kinds.count("model_call") == 2 and kinds.count("tool_call") == 2


def test_the_plan_marks_only_patch_and_review_as_needing_a_model():
    plan = plan_for(Task(description="x", repo_path="."))
    assert [s.id for s in plan.steps if s.needs_llm] == ["patch", "review"]
    assert all(s.reason for s in plan.steps)


def test_a_patch_that_still_fails_goes_back_with_the_test_output(cfg, tmp_path):
    result, p, events, _ = run(cfg, tmp_path, arc=[diff(WRONG), diff(GOOD), ACCEPT])
    assert result.status == "fixed" and result.attempts == 2
    retry = p["arc"].requests[1].messages[0]["content"]
    assert "Your previous patch failed" in retry and "still failing" in retry


def test_a_patch_that_breaks_a_passing_test_is_not_accepted(cfg, tmp_path):
    result, p, _, _ = run(cfg, tmp_path, arc=[diff(BREAKS_MONEY), diff(GOOD), ACCEPT])
    assert result.status == "fixed" and result.attempts == 2
    assert "newly failing: ['tests/test_money.py::test_add_tax']" in p["arc"].requests[1].messages[0]["content"]


def test_a_rejected_review_sends_the_defects_back_to_the_coder(cfg, tmp_path):
    result, p, _, _ = run(cfg, tmp_path, arc=[diff(GOOD), REJECT, diff(GOOD), ACCEPT])
    assert result.status == "fixed" and result.attempts == 2
    retry = p["arc"].requests[2].messages[0]["content"]
    assert "drops the day check" in retry and "(not applied)" in retry and "never diff on top of one" in retry


def test_the_last_attempt_escalates_and_waits_on_the_spend_gate(cfg, tmp_path):
    result, p, events, _ = run(cfg, tmp_path, arc=[diff(WRONG), diff(WRONG)], approver=deny_all)
    assert result.status == "stopped" and "declined" in result.stop_reason
    assert p["openai"].requests == []  # nothing paid was sent
    assert events[-1].status == "failed" and events[-1].step == "patch"


def test_an_approved_escalation_uses_the_paid_coder_and_a_free_reviewer(cfg, tmp_path):
    result, p, _, _ = run(cfg, tmp_path, arc=[diff(WRONG), diff(WRONG), ACCEPT], openai=[diff(GOOD)])
    assert result.status == "fixed" and result.attempts == 3
    assert p["openai"].requests[0].model == "gpt-6.1-sol"
    assert p["arc"].requests[-1].model == "DeepSeek-V4.1-Flash-thinking-max"  # review stays free and cross-family


def test_unknown_task_types_are_refused_until_the_model_planner_exists():
    with pytest.raises(NotImplementedError, match="phase 2"):
        plan_for(Task(description="x", repo_path=".", task_type="write_novel"))


def test_extract_diff_takes_a_fenced_block_or_a_bare_diff():
    assert extract_diff(diff(GOOD)) == GOOD
    assert extract_diff("Sure:\n" + GOOD) == GOOD
    assert extract_diff("I could not find the bug.") == ""


def test_the_exported_bundle_reruns_the_fix_without_the_factory(cfg, tmp_path):
    import subprocess
    import sys
    import zipfile

    from factory.export import export_bundle

    result, _, _, _ = run(cfg, tmp_path, arc=[diff(GOOD), ACCEPT])
    zip_path = export_bundle(result, Task(description=ISSUE, repo_path=str(REPO)), tmp_path, tmp_path / "out")
    with zipfile.ZipFile(zip_path) as z:
        names = {Path(n).name for n in z.namelist()}
        z.extractall(tmp_path / "unzipped")
    assert {"fix.patch", "rerun.py", "workflow.json", "evidence.json", "provenance.jsonl", "llm_calls.jsonl",
            "cost_summary.json", "README.md"} <= names
    bundle = next((tmp_path / "unzipped").iterdir())
    workflow = json.loads((bundle / "workflow.json").read_text())
    assert workflow["targets"] == ["tests/test_dates.py::test_parse_date_order"]
    assert [s["agent"] for s in workflow["steps"]] == ["code_navigator", "test_runner", "coder", "test_runner",
                                                       "reviewer"]
    rerun = subprocess.run([sys.executable, str(bundle / "rerun.py"), str(REPO)], capture_output=True, text=True)
    assert rerun.returncode == 0 and "target tests: PASS" in rerun.stdout and "whole suite: PASS" in rerun.stdout


def test_a_run_without_a_patch_has_nothing_to_export(cfg, tmp_path):
    from factory.export import export_bundle

    result, _, _, _ = run(cfg, tmp_path, arc=["I can't.", "Still can't."], approver=deny_all)
    with pytest.raises(ValueError, match="nothing to export"):
        export_bundle(result, Task(description=ISSUE, repo_path=str(REPO)), tmp_path, tmp_path / "out")
