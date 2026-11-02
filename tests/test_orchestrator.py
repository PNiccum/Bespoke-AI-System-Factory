"""The bug-fix run end to end, with scripted models: plan, award, patch, check, review, repair, escalate."""

import json
from pathlib import Path

import pytest

from factory.config import load_config
from factory.llm import LLM, Usage
from factory.llm.gate import approve_all, deny_all
from factory.llm.providers import RawResult
from factory.orchestrator import Orchestrator, Task, extract_diff, plan_for, touched_tests
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


SKIPS_TEST = """--- a/tests/test_dates.py
+++ b/tests/test_dates.py
@@ -1,5 +1,7 @@
+import pytest
 from calc.dates import parse_date
 
 
+@pytest.mark.skip
 def test_parse_date_order():
     assert parse_date("2026-10-28") == (2026, 10, 28)
"""
IGNORES_TEST = """--- /dev/null
+++ b/conftest.py
@@ -0,0 +1 @@
+collect_ignore = ["tests/test_dates.py"]
"""


@pytest.mark.parametrize("cheat", [SKIPS_TEST, IGNORES_TEST], ids=["skip-marker", "conftest-ignore"])
def test_a_patch_that_touches_tests_is_refused_before_it_runs(cfg, tmp_path, cheat):
    result, p, events, _ = run(cfg, tmp_path, arc=[diff(cheat), diff(GOOD), ACCEPT])
    assert result.status == "fixed" and result.attempts == 2
    assert any(e.step == "patch" and e.status == "failed" and "changes tests" in e.detail for e in events)
    assert "Fix the code, not the tests" in p["arc"].requests[1].messages[0]["content"]


def test_with_test_edits_allowed_a_skipped_target_still_fails_the_check(cfg, tmp_path):
    providers = {"arc": Scripted(diff(SKIPS_TEST), diff(GOOD), ACCEPT), "openai": Scripted(), "claude": Scripted()}
    llm = LLM(cfg, tmp_path, use_cache=False, providers=providers, approver=approve_all)
    registry = Registry.load(cfg, ROOT / "registry")
    registry.scores = ScoreBook(tmp_path / "scores.json")
    events = []
    task = Task(description=ISSUE, repo_path=str(REPO), allow_test_edits=True)
    result = Orchestrator(llm, registry, on_event=events.append).run(task)
    assert result.status == "fixed" and result.attempts == 2
    assert any(e.step == "check" and "did not run" in e.detail for e in events)


def test_touched_tests_names_test_files_and_pytest_config():
    assert touched_tests(SKIPS_TEST) == ["tests/test_dates.py"]
    assert touched_tests(IGNORES_TEST) == ["conftest.py"]
    assert touched_tests(GOOD) == []


def test_a_rejected_review_sends_the_defects_back_to_the_coder(cfg, tmp_path):
    result, p, _, _ = run(cfg, tmp_path, arc=[diff(GOOD), REJECT, diff(GOOD), ACCEPT])
    assert result.status == "fixed" and result.attempts == 2
    retry = p["arc"].requests[2].messages[0]["content"]
    assert "drops the day check" in retry and "(not applied)" in retry and "never diff on top of one" in retry


def test_the_last_attempt_escalates_and_waits_on_the_spend_gate(cfg, tmp_path):
    result, p, events, _ = run(cfg, tmp_path, arc=[diff(WRONG), diff(WRONG), diff(WRONG)], approver=deny_all)
    assert result.status == "failed" and result.attempts == 3 and "after 3 attempts" in result.stop_reason
    assert p["openai"].requests == []  # nothing paid was sent
    assert any(e.status == "waiting" and "declined" in e.detail and "free model" in e.detail for e in events)


def test_a_declined_escalation_still_gets_its_free_attempt(cfg, tmp_path):
    result, p, _, _ = run(cfg, tmp_path, arc=[diff(WRONG), diff(WRONG), diff(GOOD), ACCEPT], approver=deny_all)
    assert result.status == "fixed" and result.attempts == 3
    assert p["openai"].requests == []


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

    result, _, _, _ = run(cfg, tmp_path, arc=["I can't.", "Still can't.", "No."], approver=deny_all)
    with pytest.raises(ValueError, match="nothing to export"):
        export_bundle(result, Task(description=ISSUE, repo_path=str(REPO)), tmp_path, tmp_path / "out")


def test_the_coder_sees_the_code_the_failing_tests_import_even_when_locate_finds_nothing(cfg, tmp_path):
    from factory.tools.repo import Locations

    bench = ROOT / "bench" / "repos"
    o = Orchestrator.__new__(Orchestrator)
    o.max_file_lines = 400
    issue = "The visited-pages summary lists pages alphabetically; it should list them in the order first visited."
    shown = o.shown_files(Task(description=issue, repo_path=str(bench / "dedupe_order")), Locations(names=[], hits=[]),
                          ["tests/test_summary.py::test_visited"])
    assert [rel for rel, _ in shown] == ["report/clean.py", "report/summary.py", "tests/test_summary.py"]
    assert "sorted(set(items))" in shown[0][1]


def test_in_a_large_repository_the_failing_tests_imports_still_reach_the_coder(tmp_path):
    from factory.orchestrator import SMALL_REPO_FILES
    from factory.tools.repo import Locations

    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    for i in range(SMALL_REPO_FILES + 2):
        (tmp_path / "pkg" / f"m{i}.py").write_text(f"X = {i}\n")
    (tmp_path / "tests" / "test_m.py").write_text("from pkg.m7 import X\n\n\ndef test_x():\n    assert X == 0\n")
    o = Orchestrator.__new__(Orchestrator)
    o.max_file_lines = 400
    shown = o.shown_files(Task(description="X is wrong", repo_path=str(tmp_path)), Locations(names=[], hits=[]),
                          ["tests/test_m.py::test_x"])
    assert [rel for rel, _ in shown] == ["pkg/m7.py", "tests/test_m.py"]
