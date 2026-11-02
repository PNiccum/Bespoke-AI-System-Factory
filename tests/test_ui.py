"""The web run session: a background run whose paid calls wait for a click, and a page that renders."""

import time
from pathlib import Path

import pytest

from factory.config import load_config
from factory.orchestrator import Task
from factory.ui.session import RunSession
from tests.test_orchestrator import ACCEPT, GOOD, ISSUE, REPO, WRONG, Scripted, diff

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(ROOT / "configs" / "factory.yaml")
    cfg.root = tmp_path
    cfg.registry_dir = ROOT / "registry"
    return cfg


def session(cfg, tmp_path, arc=(), openai=(), **kw):
    providers = {"arc": Scripted(*arc), "openai": Scripted(*openai), "claude": Scripted()}
    return RunSession(cfg, providers=providers, exports_dir=tmp_path / "exports", **kw), providers


def wait_for(cond, timeout=30):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise TimeoutError
        time.sleep(0.05)


def test_a_free_run_finishes_and_leaves_a_bundle(cfg, tmp_path):
    s, _ = session(cfg, tmp_path, arc=[diff(GOOD), ACCEPT])
    s.start(Task(description=ISSUE, repo_path=str(REPO)))
    s.wait(60)
    assert s.result.status == "fixed" and s.bundle.exists() and s.pending is None
    assert [e.step for e in s.events if e.status == "done"][-1] == "review"


def test_a_paid_call_waits_for_a_click_and_proceeds_when_approved(cfg, tmp_path):
    s, p = session(cfg, tmp_path, arc=[diff(WRONG), diff(WRONG), ACCEPT], openai=[diff(GOOD)])
    s.start(Task(description=ISSUE, repo_path=str(REPO)))
    wait_for(lambda: s.pending is not None)
    assert s.running and s.pending.request.model == "gpt-6.1-sol" and p["openai"].requests == []
    s.answer(True)
    s.wait(60)
    assert s.result.status == "fixed" and len(p["openai"].requests) == 1


def test_no_answer_in_time_counts_as_a_refusal(cfg, tmp_path):
    s, p = session(cfg, tmp_path, arc=[diff(WRONG), diff(WRONG), diff(WRONG)], answer_timeout_s=0.2)
    s.start(Task(description=ISSUE, repo_path=str(REPO)))
    s.wait(60)
    assert s.result.status == "failed" and p["openai"].requests == [] and s.bundle is None
    assert any("declined" in e.detail and "free model" in e.detail for e in s.events)


def test_only_one_run_at_a_time(cfg, tmp_path):
    s, _ = session(cfg, tmp_path, arc=[diff(WRONG), diff(WRONG)])
    s.start(Task(description=ISSUE, repo_path=str(REPO)))
    wait_for(lambda: s.pending is not None)
    with pytest.raises(RuntimeError, match="already"):
        s.start(Task(description=ISSUE, repo_path=str(REPO)))
    s.answer(False)
    s.wait(60)


def test_the_page_renders_before_any_run():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "ASPIRE"
    assert any("Submit a task" in i.value for i in at.info)
