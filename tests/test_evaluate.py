"""The evaluation experiments: hostile patches, zero-model reruns, the labeled review set, reviewers, cost report."""

import json
import subprocess
import sys

import pytest

from factory.bench import Outcome, judge, judge_hidden, load_tasks
from factory.evaluate import (
    ATTACKS,
    NO_MODEL,
    Scripted,
    Verdict,
    cost_report,
    export_for,
    load_review_set,
    rerun_bundle,
    reviewer_cfg,
    reviewers_summary,
    run_guards,
    run_reviewers,
    wilson,
)
from tests.test_orchestrator import ROOT, cfg  # noqa: F401

TASKS = {t.name: t for t in load_tasks(ROOT / "bench" / "tasks.yaml", ROOT)}
REVIEW_SET = load_review_set()


def test_every_plain_code_attack_is_blocked_by_the_judge_and_the_fix_check(cfg):
    rows = {r.attack: r for r in run_guards(cfg, TASKS)}
    assert rows.keys() == {a.name for a in ATTACKS}
    for a in ATTACKS:
        r = rows[a.name]
        if a.plain_code_can_stop:
            assert r.judge_blocked and r.check_blocked, r
        else:  # passes every visible test: the hidden tests are what catch it
            assert not r.judge_blocked and r.hidden_blocked, r


def test_a_patch_that_ends_pytest_early_is_not_solved():
    task = TASKS["leap_year"]
    exit0 = next(a.patch for a in ATTACKS if a.name == "exit-at-import")
    solved, why = judge(task.repo, exit0)
    assert not solved and why.startswith("tests did not run")


def test_the_review_set_is_labeled_and_covers_every_task():
    assert len(REVIEW_SET) == len({p.id for p in REVIEW_SET}) >= 30
    assert {p.task for p in REVIEW_SET} == set(TASKS) - {"injection"}
    assert sum(p.label == "bad" for p in REVIEW_SET) >= 20 and {p.label for p in REVIEW_SET} == {"good", "bad"}


@pytest.mark.parametrize("p", REVIEW_SET, ids=[p.id for p in REVIEW_SET])
def test_every_review_set_patch_passes_the_visible_tests_and_the_hidden_tests_tell_good_from_bad(p):
    task = TASKS[p.task]
    assert judge(task.repo, p.patch)[0], "a reviewer only ever sees patches that pass the visible tests"
    hidden, why = judge_hidden(p.task, task.repo, p.patch)
    assert hidden is (p.label == "good"), why


@pytest.mark.parametrize("name", sorted(TASKS))
def test_the_hidden_tests_fail_on_the_unpatched_repository(name):
    assert judge_hidden(name, TASKS[name].repo, "")[0] is False


def test_a_bundle_reruns_with_no_model_and_the_hidden_tests_catch_a_wrong_fix(tmp_path):
    task = TASKS["leap_year"]
    golden = (ROOT / "bench" / "golden" / "leap_year.patch").read_text()
    wrong = next(p.patch for p in REVIEW_SET if p.id == "leap_year-hardcode")
    assert rerun_bundle(export_for(task, golden, tmp_path / "g"), ROOT / task.repo)[:2] == (True, "")
    passed, blocked, _ = rerun_bundle(export_for(task, wrong, tmp_path / "w"), ROOT / task.repo)
    assert passed and not blocked  # the visible tests cannot tell
    hidden_repo = tmp_path / "repo"
    subprocess.run(["cp", "-r", task.repo, str(hidden_repo)], check=True)
    subprocess.run(["cp", "-r", f"{ROOT}/bench/hidden/leap_year/.", str(hidden_repo)], check=True)
    assert not rerun_bundle(export_for(task, wrong, tmp_path / "w2"), hidden_repo)[0]


@pytest.mark.parametrize("code,stopped", [("import factory", "import 'factory'"),
                                          ("import socket; socket.create_connection(('127.0.0.1', 9))",
                                           "socket.connect")])
def test_the_rerun_guard_stops_an_import_of_aspire_or_any_socket(tmp_path, code, stopped):
    script = tmp_path / "rerun.py"
    script.write_text(code + "\n")
    r = subprocess.run([sys.executable, "-I", "-c", NO_MODEL, str(script)], capture_output=True, text=True)
    assert r.returncode != 0 and stopped in r.stderr


def test_a_same_family_reviewer_is_allowed_only_in_the_experiment(cfg):
    glm = reviewer_cfg(cfg, "arc:GLM-5.3-thinking-high")
    assert glm.roles["reviewer"].model == "GLM-5.3-thinking-high" and glm.roles["reviewer"].fallbacks == []
    assert cfg.roles["reviewer"].model != "GLM-5.3-thinking-high"  # the real config is untouched
    with pytest.raises(ValueError, match="not in the model catalog"):
        reviewer_cfg(cfg, "arc:no-such-model")


def test_reviewers_are_scored_against_the_labels(cfg):
    patches = [p for p in REVIEW_SET if p.task == "leap_year"]  # hardcode, partial (bad); alternative, golden (good)
    accept = json.dumps({"verdict": "accept", "defects": []})
    reject = json.dumps({"verdict": "reject", "defects": [{"line": "+", "problem": "special-cases 1900"}]})
    replies = [accept if p.label == "good" or p.kind == "partial" else reject for p in patches]
    arc = Scripted(*replies)
    rows = run_reviewers(cfg, TASKS, ["arc:DeepSeek-V4.1-Flash-thinking-max"], patches, workers=1,
                         providers={"arc": arc, "openai": Scripted(), "claude": Scripted()})
    assert [r.verdict for r in rows] == ["reject" if r == reject else "accept" for r in replies]
    assert all(not r.same_family_as_coder and r.family == "deepseek" for r in rows)
    text = reviewers_summary(rows)
    assert "| arc:DeepSeek-V4.1-Flash-thinking-max | deepseek | no | 1/2" in text  # one wrong patch got through
    assert "good patches rejected 0/2" in text


def test_the_cost_report_gives_intervals_and_the_break_even():
    def o(task, mode, solved, tokens, heldout=False):
        return Outcome(task=task, mode=mode, solved=solved, why="", tokens=tokens, priced_usd=tokens / 1e5,
                       hidden_solved=solved or None, heldout=heldout)
    rows = [o("a", "aspire", True, 3000), o("b", "aspire", True, 3000, True),
            o("a", "baseline", True, 1000), o("b", "baseline", False, 1000, True)]
    text = cost_report(rows)
    assert "| aspire (all) | 2 | 2/2 (34%-100%) |" in text and "| baseline (held out) | 1 | 0/1" in text
    assert "ASPIRE alone solved 1, the single call alone solved 0" in text
    assert "3.0x one single call" in text and "breaks even after 3 uses" in text


def test_wilson_is_a_sane_interval():
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(8, 8)
    assert 0.6 < lo < 0.7 and hi == 1.0
    assert wilson(5, 10)[0] < 0.5 < wilson(5, 10)[1]


def test_verdicts_round_trip_through_json():
    v = Verdict(patch="x", task="t", label="bad", kind="hardcode", author="planted", reviewer="arc:m",
                family="glm", same_family_as_coder=True, verdict="accept")
    assert Verdict(**json.loads(v.model_dump_json())) == v


def test_the_bench_command_repeats_trials_and_reports_hidden_and_shadow_cost(tmp_path, monkeypatch, capsys):
    import factory.bench as bench
    from factory.llm import LLM

    golden = (ROOT / "bench" / "golden" / "leap_year.patch").read_text()
    wrong = next(p.patch for p in REVIEW_SET if p.id == "leap_year-hardcode")
    arc = Scripted(f"```diff\n{golden}```", f"```diff\n{wrong}```")
    monkeypatch.setattr(bench, "LLM", lambda cfg, run_dir, **kw: LLM(
        cfg, run_dir, use_cache=False, approver=kw["approver"],
        providers={"arc": arc, "openai": Scripted(), "claude": Scripted()}))
    monkeypatch.chdir(ROOT)
    assert bench.main(["--only", "leap_year", "--mode", "baseline", "--repeat", "2", "--out", str(tmp_path)]) == 0
    rows = json.loads(next(tmp_path.glob("*/results.json")).read_text())
    assert [(r["trial"], r["solved"], r["hidden_solved"]) for r in rows] == [(1, True, True), (2, True, False)]
    assert all(r["heldout"] for r in rows)
    summary = next(tmp_path.glob("*/summary.md")).read_text()
    assert "2/2 solved (1/2 with the hidden tests)" in summary and "if every model were paid" in summary

    from factory.evaluate import main
    assert main(["report", str(next(tmp_path.glob("*")))]) == 0
    assert "| baseline (held out) | 2 | 2/2" in capsys.readouterr().out
