"""The test_gen and pr_review task types end to end, with scripted models, plus their plain-code agents."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from factory.export import export_bundle
from factory.llm import LLM
from factory.llm.gate import approve_all, deny_all
from factory.orchestrator import Orchestrator, Task, plan_for
from factory.registry import Registry, ScoreBook
from factory.tools.diff_inspect import inspect, parse_diff
from factory.tools.mutate import ScoreIn, mutants, score_tests
from tests.test_orchestrator import ACCEPT, BREAKS_MONEY, GOOD, REJECT, REPO, Scripted, cfg, diff  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
REQUEST = "Write tests for days_in_month in calc/dates.py"
STRONG = """--- /dev/null
+++ b/tests/test_days.py
@@ -0,0 +1,11 @@
+from calc.dates import days_in_month
+
+
+def test_long_months():
+    for m in (1, 3, 5, 7, 8, 10, 12):
+        assert days_in_month(m) == 31
+
+
+def test_short_months():
+    for m in (4, 6, 9, 11):
+        assert days_in_month(m) == 30
"""
WEAK = """--- /dev/null
+++ b/tests/test_days.py
@@ -0,0 +1,5 @@
+from calc.dates import days_in_month
+
+
+def test_january():
+    assert days_in_month(1) == 31
"""
FAILING = """--- /dev/null
+++ b/tests/test_days.py
@@ -0,0 +1,5 @@
+from calc.dates import days_in_month
+
+
+def test_february():
+    assert days_in_month(2) == 28
"""
LEAKY = """--- a/calc/money.py
+++ b/calc/money.py
@@ -1,2 +1,4 @@
+API_KEY = "abcd1234efgh5678"
 def add_tax(amount, rate=0.05):
+    print("adding tax")
     return round(amount * (1 + rate), 2)
"""


def run(cfg, tmp_path, task, arc=(), openai=(), claude=(), approver=approve_all):
    providers = {"arc": Scripted(*arc), "openai": Scripted(*openai), "claude": Scripted(*claude)}
    llm = LLM(cfg, tmp_path, use_cache=False, providers=providers, approver=approver)
    registry = Registry.load(cfg, ROOT / "registry")
    registry.scores = ScoreBook(tmp_path / "scores.json")
    events = []
    result = Orchestrator(llm, registry, on_event=events.append).run(task)
    return result, providers, events


def gen(cfg, tmp_path, *arc, **kw):
    return run(cfg, tmp_path, Task(description=REQUEST, repo_path=str(REPO), task_type="test_gen"), arc, **kw)


def review_json(verdict="approve", comments=()):
    return json.dumps({"verdict": verdict, "summary": "Swaps month and day back.", "comments": list(comments)})


def pr(cfg, tmp_path, patch, *arc, **kw):
    return run(cfg, tmp_path, Task(description="Fix parse_date swapping month and day", repo_path=str(REPO),
                                   task_type="pr_review", patch=patch), arc, **kw)


def unzip(zip_path, into):
    with zipfile.ZipFile(zip_path) as z:
        names = {Path(n).name for n in z.namelist()}
        z.extractall(into)
    return names, next(Path(into).iterdir())


# -- plans ---------------------------------------------------------------------------------------------------------

def test_the_new_plans_use_a_model_only_to_write_and_to_judge():
    tg = plan_for(Task(description="x", repo_path=".", task_type="test_gen"))
    assert [s.id for s in tg.steps if s.needs_llm] == ["write", "review"]
    pr_plan = plan_for(Task(description="x", repo_path=".", task_type="pr_review"))
    assert [s.id for s in pr_plan.steps if s.needs_llm] == ["review"]
    assert all(s.reason for s in tg.steps + pr_plan.steps)


def test_each_new_step_is_awarded_to_the_right_agent(cfg, tmp_path):
    _, _, events = gen(cfg, tmp_path, diff(STRONG), ACCEPT)
    awards = {e.step: e.agent for e in events if "parts" in e.data}
    assert awards == {"locate": "code_navigator", "baseline": "test_runner", "write": "test_writer",
                      "check": "test_runner", "score": "mutation_tester", "review": "reviewer"}


# -- test generation -----------------------------------------------------------------------------------------------

def test_strong_tests_pass_catch_every_planted_bug_and_are_reviewed(cfg, tmp_path):
    result, p, events = gen(cfg, tmp_path, diff(STRONG), ACCEPT)
    assert result.status == "tested" and result.attempts == 1 and result.patch == STRONG
    assert result.report["targets"] == ["calc/dates.py"] and result.report["functions"] == ["days_in_month"]
    assert result.report["new_tests"] == ["tests/test_days.py::test_long_months", "tests/test_days.py::test_short_months"]
    m = result.report["mutation"]
    assert m["total"] == m["killed"] == 11
    prompt = p["arc"].requests[0].messages[0]["content"]
    assert "def days_in_month" in prompt and "tests/test_dates.py" in prompt
    assert "caught 11 of 11" in p["arc"].requests[1].messages[0]["content"]  # the reviewer is told the score


def test_weak_tests_go_back_with_the_bugs_they_missed(cfg, tmp_path):
    result, p, events = gen(cfg, tmp_path, diff(WEAK), diff(STRONG), ACCEPT)
    assert result.status == "tested" and result.attempts == 2
    retry = p["arc"].requests[1].messages[0]["content"]
    assert "too weak" in retry and "calc/dates.py:8" in retry and "3 -> 4" in retry
    assert [e.step for e in events if e.status == "failed"] == ["score"]


def test_a_failing_new_test_is_sent_back(cfg, tmp_path):
    result, p, _ = gen(cfg, tmp_path, diff(FAILING), diff(STRONG), ACCEPT)
    assert result.status == "tested" and result.attempts == 2
    assert "test_february" in p["arc"].requests[1].messages[0]["content"]


def test_a_diff_that_changes_source_code_is_refused_unrun(cfg, tmp_path):
    result, p, events = gen(cfg, tmp_path, diff(GOOD), diff(STRONG), ACCEPT)
    assert result.status == "tested" and result.attempts == 2
    assert "changed non-test files" in p["arc"].requests[1].messages[0]["content"]
    assert not any(e.step == "check" and e.status == "failed" for e in events)


def test_a_rejected_review_sends_the_tests_back(cfg, tmp_path):
    result, p, _ = gen(cfg, tmp_path, diff(STRONG), REJECT, diff(STRONG), ACCEPT)
    assert result.status == "tested" and result.attempts == 2
    assert "drops the day check" in p["arc"].requests[2].messages[0]["content"]


def test_tests_that_never_reach_the_bar_are_kept_as_unproven_without_paying(cfg, tmp_path):
    result, p, _ = gen(cfg, tmp_path, diff(WEAK), diff(WEAK), approver=deny_all)
    assert result.status == "unproven" and result.patch == WEAK
    assert result.report["mutation"]["killed"] < result.report["mutation"]["total"]
    assert "declined" in result.stop_reason and "kept the strongest" in result.stop_reason
    assert p["openai"].requests == []


def test_the_last_attempt_escalates_the_test_writer_when_approved(cfg, tmp_path):
    result, p, _ = gen(cfg, tmp_path, diff(WEAK), diff(WEAK), ACCEPT, openai=[diff(STRONG)])
    assert result.status == "tested" and result.attempts == 3
    assert p["openai"].requests[0].model == "gpt-6.1-sol"


def test_the_tests_bundle_rechecks_itself_without_the_factory(cfg, tmp_path):
    task = Task(description=REQUEST, repo_path=str(REPO), task_type="test_gen")
    result, _, _ = gen(cfg, tmp_path, diff(STRONG), ACCEPT)
    names, bundle = unzip(export_bundle(result, task, tmp_path, tmp_path / "out"), tmp_path / "unzipped")
    assert {"tests.patch", "mutate.py", "rerun.py", "workflow.json", "evidence.json", "README.md"} <= names
    rerun = subprocess.run([sys.executable, str(bundle / "rerun.py"), str(REPO)], capture_output=True, text=True)
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert "new tests: PASS" in rerun.stdout and "caught: 11 of 11" in rerun.stdout


# -- pull request review -------------------------------------------------------------------------------------------

def test_a_good_pr_is_reviewed_with_its_test_delta(cfg, tmp_path):
    comment = {"file": "calc/dates.py", "line": 4, "severity": "nit", "problem": "no comment on the order"}
    result, p, events = pr(cfg, tmp_path, GOOD, review_json(comments=[comment]))
    assert result.status == "reviewed" and result.attempts == 1
    rep = result.report
    assert rep["review"]["verdict"] == "approve" and rep["review"]["comments"][0]["line"] == 4
    assert rep["tests"]["newly_passing"] == ["tests/test_dates.py::test_parse_date_order"]
    assert rep["tests"]["newly_failing"] == [] and not rep["overridden"]
    assert [f["rule"] for f in rep["facts"]["findings"]] == ["no_tests"]
    prompt = p["arc"].requests[0].messages[0]["content"]
    assert "newly passing: ['tests/test_dates.py::test_parse_date_order']" in prompt
    assert p["arc"].requests[0].model == "DeepSeek-V4.1-Flash-thinking-max"


def test_comments_off_the_diff_are_dropped_and_the_reviewer_asked_again(cfg, tmp_path):
    off = [{"file": "calc/money.py", "line": 1, "severity": "major", "problem": "unrelated"},
           {"file": "calc/dates.py", "line": 40, "severity": "minor", "problem": "past the hunk"}]
    on = [{"file": "calc/dates.py", "line": 4, "severity": "minor", "problem": "fine but untested"}]
    result, p, _ = pr(cfg, tmp_path, GOOD, review_json(comments=off), review_json(comments=on))
    assert result.status == "reviewed" and result.attempts == 2
    assert [c["line"] for c in result.report["review"]["comments"]] == [4]
    retry = p["arc"].requests[1].messages[0]["content"]
    assert "calc/money.py:1" in retry and "does not change" in retry


def test_facts_outrank_the_model_when_a_pr_breaks_a_test(cfg, tmp_path):
    result, _, _ = pr(cfg, tmp_path, BREAKS_MONEY, review_json())
    rep = result.report
    assert rep["review"]["verdict"] == "request_changes"
    assert "tests/test_money.py::test_add_tax" in rep["overridden"]


def test_rule_findings_catch_secrets_and_debug_prints(cfg, tmp_path):
    result, p, _ = pr(cfg, tmp_path, LEAKY, review_json())
    rules = {f["rule"]: f for f in result.report["facts"]["findings"]}
    assert rules["secret"]["severity"] == "blocker" and rules["secret"]["line"] == 1
    assert rules["debug_print"]["line"] == 3 and "no_tests" in rules
    assert result.report["review"]["verdict"] == "request_changes"  # a blocker finding forbids approval
    assert "looks like a hard-coded credential" in p["arc"].requests[0].messages[0]["content"]


def test_an_unparseable_review_escalates_only_with_approval(cfg, tmp_path):
    bad = ["not json"] * 6  # two free attempts, three parse tries each
    result, p, _ = pr(cfg, tmp_path, GOOD, *bad, approver=deny_all)
    assert result.status == "stopped" and p["claude"].requests == []
    result, p, _ = pr(cfg, tmp_path, GOOD, *bad, claude=[review_json()])
    assert result.status == "reviewed" and p["claude"].requests[0].model == "claude-sonnet-5-5"


def test_an_empty_diff_is_not_reviewed(cfg, tmp_path):
    result, p, _ = pr(cfg, tmp_path, "")
    assert result.status == "failed" and "no diff" in result.stop_reason and p["arc"].requests == []


def test_the_review_bundle_rechecks_its_facts_without_the_factory(cfg, tmp_path):
    task = Task(description="Fix parse_date", repo_path=str(REPO), task_type="pr_review", patch=GOOD)
    result, _, _ = pr(cfg, tmp_path, GOOD, review_json())
    names, bundle = unzip(export_bundle(result, task, tmp_path, tmp_path / "out"), tmp_path / "unzipped")
    assert {"change.patch", "review.md", "review.json", "diff_inspect.py", "rerun.py"} <= names
    assert "**Verdict: approve**" in (bundle / "review.md").read_text()
    rerun = subprocess.run([sys.executable, str(bundle / "rerun.py"), str(REPO)], capture_output=True, text=True)
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    assert "rule findings: 1 (recorded 1): PASS" in rerun.stdout and "fixes 1, breaks 0: PASS" in rerun.stdout


# -- the plain-code agents on their own ----------------------------------------------------------------------------

def test_mutants_are_single_changes_spread_evenly_and_repeatable():
    src = (REPO / "calc" / "dates.py").read_text()
    a = mutants("calc/dates.py", src, ["days_in_month"], limit=8)
    assert len(a) == 8 and {m.change for m in a} >= {"return -> return None", "in -> not in", "31 -> 32"}
    assert [m.change for m in a] == [m.change for m in mutants("calc/dates.py", src, ["days_in_month"], limit=8)]
    assert all("parse_date" in m.source for m in a)  # the rest of the file is untouched
    whole = mutants("calc/dates.py", src, [], limit=50)
    assert any(m.line == 4 for m in whole)  # no function named: parse_date is mutated too


def test_score_tests_reports_an_error_when_the_tests_fail_unmutated():
    score = score_tests(ScoreIn(repo_path=str(REPO), files=["calc/dates.py"],
                                tests=["tests/test_dates.py::test_parse_date_order"]))
    assert score.error and score.total == 0


def test_parse_diff_numbers_added_lines_on_the_new_side():
    f = parse_diff(LEAKY)[0]
    assert f.path == "calc/money.py" and sorted(f.added) == [1, 3] and f.removed == 0
    assert f.touches(2) and not f.touches(9)
    new = parse_diff(STRONG)[0]
    assert new.old_path == "" and len(new.added) == 11


def test_inspect_flags_a_patch_that_breaks_syntax():
    broken = GOOD.replace("+    return int(year), int(month), int(day)", "+    return int(year), int(month), int(day")
    facts = inspect(broken, str(REPO))
    assert facts["applied"] and any(f["rule"] == "syntax" for f in facts["findings"])
    assert inspect("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n", str(REPO))["applied"] is False


@pytest.mark.parametrize("task_type", ["test_gen", "pr_review"])
def test_the_new_task_types_can_route_their_tool_agents_through_mcp(cfg, tmp_path, task_type):
    cfg.tool_transport = "mcp"
    if task_type == "test_gen":
        result, _, _ = gen(cfg, tmp_path, diff(STRONG), ACCEPT)
        assert result.status == "tested" and result.report["mutation"]["killed"] == 11
    else:
        result, _, _ = pr(cfg, tmp_path, LEAKY, review_json())
        assert result.status == "reviewed"
        assert {f["rule"] for f in result.report["facts"]["findings"]} >= {"secret", "debug_print"}
