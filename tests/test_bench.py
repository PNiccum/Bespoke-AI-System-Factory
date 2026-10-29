"""The benchmark: every task is a real failing bug with a known fix, and both modes are judged by the same code."""

import pytest

from factory.bench import judge, load_tasks, run_one, summarize
from factory.tools.sandbox import RunTestsIn, run_tests
from tests.test_orchestrator import ACCEPT, ROOT, Scripted, cfg, diff  # noqa: F401

TASKS = load_tasks(ROOT / "bench" / "tasks.yaml", ROOT)


def golden(name):
    return (ROOT / "bench" / "golden" / f"{name}.patch").read_text()


def test_there_are_six_to_eight_tasks_one_of_them_hostile():
    assert 6 <= len(TASKS) <= 8 and len({t.name for t in TASKS}) == len(TASKS)
    assert [t.name for t in TASKS if t.hostile] == ["injection"]


@pytest.mark.parametrize("task", TASKS, ids=[t.name for t in TASKS])
def test_each_task_fails_as_given_and_its_known_fix_solves_it(task):
    assert not run_tests(RunTestsIn(repo_path=task.repo)).ok
    assert judge(task.repo, None) == (False, "no patch")
    solved, why = judge(task.repo, golden(task.name))
    assert solved, why


def test_a_patch_that_deletes_the_failing_test_does_not_count():
    task = next(t for t in TASKS if t.name == "leap_year")
    cheat = "--- a/tests/test_years.py\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-from cal.years import days_in_year, is_leap\n"
    assert judge(task.repo, cheat) == (False, "changes tests/test_years.py")


@pytest.mark.parametrize("mode,replies,calls", [("baseline", 1, 1), ("aspire", 2, 2)])
def test_both_modes_run_on_the_same_coder_and_are_judged_alike(cfg, mode, replies, calls):
    task = next(t for t in TASKS if t.name == "leap_year")
    arc = Scripted(*[diff(golden("leap_year")), ACCEPT][:replies])
    o = run_one(cfg, task, mode, providers={"arc": arc, "openai": Scripted(), "claude": Scripted()},
                registry_dir=ROOT / "registry")
    assert o.solved and o.model_calls == calls and o.real_usd == 0
    assert arc.requests[0].model == cfg.roles["coder"].model
    if mode == "baseline":  # one prompt holding every file, tests included
        prompt = arc.requests[0].messages[0]["content"]
        assert "cal/years.py" in prompt and "tests/test_years.py" in prompt
    assert "1/1 solved" in summarize([o])


def test_a_crash_is_a_failed_task_not_a_failed_benchmark(cfg):
    task = next(t for t in TASKS if t.name == "slug_case")
    o = run_one(cfg, task, "baseline", providers={"arc": Scripted(), "openai": Scripted(), "claude": Scripted()},
                registry_dir=ROOT / "registry")
    assert not o.solved and "IndexError" in o.why
