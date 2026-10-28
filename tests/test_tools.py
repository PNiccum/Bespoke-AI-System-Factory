"""Plain-code agents: locating code from an issue, and running tests on a throwaway copy."""

import os
from pathlib import Path

import pytest

from factory.tools.repo import LocateIn, issue_names, locate
from factory.tools.sandbox import RunTestsIn, prove_fix, run_tests, scrubbed_env

FIXTURES = Path(__file__).parent / "fixtures"
REPO = FIXTURES / "buggy_repo"
PATCH = (FIXTURES / "buggy_repo.patch").read_text()
ISSUE = """parse_date returns the month and day in the wrong order.

Traceback (most recent call last):
  File "tests/test_dates.py", line 5, in test_parse_date_order
  File "calc/dates.py", line 4, in parse_date
AssertionError
"""


def test_issue_names_finds_calls_snake_case_and_frames():
    names, frames, paths = issue_names("Calling `DateParser` or parse_date() breaks; see calc/money.py\n"
                                       '  File "calc/dates.py", line 4')
    assert {"DateParser", "parse_date"} <= set(names)
    assert frames == [("calc/dates.py", 4)] and paths == ["calc/money.py"]


def test_locate_ranks_the_traceback_frame_and_definition_first():
    out = locate(LocateIn(repo_path=str(REPO), issue_text=ISSUE))
    top = out.hits[0]
    assert top.path == "calc/dates.py" and top.line == 4
    assert "definition of parse_date" in top.why and "error raised at line 4" in top.why and not top.is_test
    assert next(h for h in out.hits if h.path == "tests/test_dates.py").is_test
    assert "calc/money.py" not in [h.path for h in out.hits]  # never named, never mentioned


def test_locate_without_names_returns_nothing_rather_than_guessing(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    assert locate(LocateIn(repo_path=str(tmp_path), issue_text="it is broken")).hits == []


def test_run_tests_reports_each_test_and_leaves_the_original_untouched():
    before = (REPO / "calc" / "dates.py").read_text()
    report = run_tests(RunTestsIn(repo_path=str(REPO), patch=PATCH))
    assert report.patch_applied and report.ok
    assert len(report.passed) == 2 and report.failed == []
    assert (REPO / "calc" / "dates.py").read_text() == before


def test_prove_fix_needs_a_failure_before_and_a_pass_after():
    proof = prove_fix(str(REPO), PATCH, ["tests/test_dates.py"])
    assert proof.before.failed and proof.after.ok and proof.proven
    unrelated = prove_fix(str(REPO), PATCH, ["tests/test_money.py"])  # passed before too: proves nothing
    assert not unrelated.proven


def test_a_patch_that_does_not_apply_is_reported_not_run():
    report = run_tests(RunTestsIn(repo_path=str(REPO), patch="--- a/nope.py\n+++ b/nope.py\n@@ -1 +1 @@\n-x\n+y\n"))
    assert report.patch_applied is False and not report.ok and report.passed == []


def test_timeouts_stop_the_run(tmp_path):
    (tmp_path / "test_slow.py").write_text("import time\n\ndef test_slow():\n    time.sleep(30)\n")
    report = run_tests(RunTestsIn(repo_path=str(tmp_path), timeout_s=2))
    assert report.timed_out and not report.ok


def test_keys_never_reach_the_code_under_test(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    env = scrubbed_env()
    assert "ANTHROPIC_API_KEY" not in env and "GITHUB_TOKEN" not in env and "PATH" in env
