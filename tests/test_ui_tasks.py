"""The web page and run session with the test_gen and pr_review task types."""

from pathlib import Path

import pytest

from factory.orchestrator import Task
from tests.test_orchestrator import ACCEPT, GOOD, REPO, diff
from tests.test_ui import cfg, session  # noqa: F401  (fixture)
from tests.test_workflows import REQUEST, STRONG, review_json

ROOT = Path(__file__).resolve().parents[1]


def test_a_test_gen_run_from_the_page_leaves_a_bundle(cfg, tmp_path):
    s, _ = session(cfg, tmp_path, arc=[diff(STRONG), ACCEPT])
    s.start(Task(description=REQUEST, repo_path=str(REPO), task_type="test_gen"))
    s.wait(120)
    assert s.result.status == "tested" and s.bundle.exists()


def test_a_pr_review_run_from_the_page_leaves_a_bundle(cfg, tmp_path):
    s, _ = session(cfg, tmp_path, arc=[review_json()])
    s.start(Task(description="Fix parse_date", repo_path=str(REPO), task_type="pr_review", patch=GOOD))
    s.wait(60)
    assert s.result.status == "reviewed" and s.bundle.exists()


@pytest.mark.parametrize("task_type, first_step", [("test_gen", "locate"), ("pr_review", "inspect")])
def test_choosing_a_task_type_shows_its_plan_and_inputs(task_type, first_step):
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    at.sidebar.selectbox[0].select(task_type).run()
    assert not at.exception
    steps = at.table[0].value["step"].tolist()
    assert steps[0] == first_step and "review" in steps
    areas = [a.label for a in at.sidebar.text_area]
    assert areas == (["What to test"] if task_type == "test_gen"
                     else ["Pull request description", "Diff (unified, as git diff prints it)"])


@pytest.mark.parametrize("task_type", ["test_gen", "pr_review"])
def test_the_page_shows_a_finished_run_of_each_new_type(cfg, tmp_path, task_type):
    testing = pytest.importorskip("streamlit.testing.v1")
    if task_type == "test_gen":
        s, _ = session(cfg, tmp_path, arc=[diff(STRONG), ACCEPT])
        s.start(Task(description=REQUEST, repo_path=str(REPO), task_type="test_gen"))
    else:
        comment = {"file": "calc/dates.py", "line": 4, "severity": "nit", "problem": "say why"}
        s, _ = session(cfg, tmp_path, arc=[review_json(comments=[comment])])
        s.start(Task(description="Fix parse_date", repo_path=str(REPO), task_type="pr_review", patch=GOOD))
    s.wait(120)
    at = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.session_state["session"] = s
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert ("caught **11 of 11**" in text) if task_type == "test_gen" else ("Verdict: approve" in text)
    assert at.success
