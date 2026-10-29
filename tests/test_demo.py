"""Recorded runs: a real run's model replies, replayed with no network and no model; everything else runs again."""

import json

import pytest

from factory.demo import NotRecorded, Offline, record, replay
from factory.export import export_bundle
from factory.llm import LLM
from factory.llm.gate import approve_all
from factory.orchestrator import Orchestrator, Task
from factory.registry import Registry, ScoreBook
from tests.test_orchestrator import ACCEPT, GOOD, ISSUE, REPO, ROOT, WRONG, Scripted, cfg, diff  # noqa: F401


def recorded_run(cfg, tmp_path, *replies):
    cfg.registry_dir = ROOT / "registry"
    run_dir = tmp_path / "original"
    run_dir.mkdir()
    llm = LLM(cfg, run_dir, use_cache=True, approver=approve_all,
              providers={"arc": Scripted(*replies), "openai": Scripted(), "claude": Scripted()})
    registry = Registry.load(cfg, ROOT / "registry")
    registry.scores = ScoreBook(tmp_path / "scores.json")
    result = Orchestrator(llm, registry).run(Task(description=ISSUE, repo_path=str(REPO)))
    return result, run_dir


def test_a_replay_reruns_everything_but_the_models_and_exports_a_real_bundle(cfg, tmp_path, monkeypatch):
    original, run_dir = recorded_run(cfg, tmp_path, diff(WRONG), diff(GOOD), ACCEPT)
    assert original.status == "fixed" and original.attempts == 2
    recording = record(run_dir, tmp_path / "demo.json", cfg)
    assert len(recording["replies"]) == 3 and recording["task"]["repo_path"] == "tests/fixtures/buggy_repo"

    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, cfg, d=None: original_registry(cfg, tmp_path)))
    events = []
    result, replay_dir = replay(cfg, json.loads((tmp_path / "demo.json").read_text()), on_event=events.append)
    assert result.status == "fixed" and result.attempts == 2 and result.patch == GOOD
    assert any(e.step == "check" and e.status == "done" for e in events)  # the tests really ran again
    task = Task(description=ISSUE, repo_path=str(REPO))
    assert export_bundle(result, task, replay_dir, tmp_path / "out").exists()


def original_registry(cfg, tmp_path):
    from factory.registry.spec import AgentSpec
    specs = [AgentSpec.load(p) for p in sorted((ROOT / "registry" / "agents").glob("*.yaml"))]
    return Registry(specs, cfg, ScoreBook(tmp_path / "replay-scores.json"))


def test_a_reply_missing_from_the_recording_is_never_fetched(cfg, tmp_path, monkeypatch):
    _, run_dir = recorded_run(cfg, tmp_path, diff(GOOD), ACCEPT)
    recording = record(run_dir, tmp_path / "demo.json", cfg)
    recording["replies"].pop(next(iter(recording["replies"])))  # drop the coder's reply
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, cfg, d=None: original_registry(cfg, tmp_path)))
    with pytest.raises(NotRecorded, match="offline replay"):  # tried the fallbacks, sent nothing anywhere
        replay(cfg, recording)


def test_the_offline_provider_answers_nothing():
    from factory.llm.providers import Request
    with pytest.raises(NotRecorded):
        Offline().complete(Request(model="m", messages=[], max_tokens=1))


def test_the_same_failure_gives_the_same_test_output_so_prompts_repeat():
    from factory.tools.sandbox import RunTestsIn, run_tests
    a, b = (run_tests(RunTestsIn(repo_path=str(REPO))) for _ in range(2))
    assert a.output_tail == b.output_tail and "factory-tests-" not in a.output_tail


def test_the_web_session_replays_a_recording_and_offers_the_bundle(cfg, tmp_path, monkeypatch):
    from factory.demo import recorded_task
    from factory.ui.session import RunSession
    _, run_dir = recorded_run(cfg, tmp_path, diff(GOOD), ACCEPT)
    recording = record(run_dir, tmp_path / "demo.json", cfg)
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, cfg, d=None: original_registry(cfg, tmp_path)))
    session = RunSession(cfg, exports_dir=tmp_path / "exports")
    session.start(recorded_task(recording), recording=recording)
    session.wait(60)
    assert session.result.status == "fixed" and session.bundle.exists() and session.error is None
