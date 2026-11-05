"""Stage-tagged replay: a recorded run's replies drive the real pipeline code offline, in seconds."""

from pathlib import Path

import pytest

from factory.demo import NotRecorded
from factory.llm import LLM
from factory.llm.gate import approve_all
from factory.pipeline.blackboard import Project
from factory.pipeline.interview import Interview
from factory.pipeline.planning import PlanningLoop
from factory.registry import Registry
from factory.replay import RefusedRecording, load, record_run, replay_providers
from factory.secrets import SecretFound
from tests.test_interview import propose
from tests.test_orchestrator import ROOT, Scripted, cfg  # noqa: F401
from tests.test_planning import ANALYSIS, critique, edit, recheck, vote

FIXTURES = Path(__file__).parent / "fixtures" / "replays"


def planning(cfg, run_dir, providers, tmp_path):
    """The approved interview a planning run starts from, then the planning loop, on the given providers."""
    seed = LLM(cfg, run_dir / "seed", use_cache=False, approver=approve_all,
               providers={"arc": Scripted(propose()), "openai": Scripted(), "claude": Scripted()})
    project = Project.create(tmp_path / "projects", "primers")
    iv = Interview(seed, Registry.load(cfg, ROOT / "registry"), project)
    iv.start("Make me a primer design tool")
    iv.approve()
    llm = LLM(cfg, run_dir, approver=approve_all, providers=providers)
    return PlanningLoop(llm, Registry.load(cfg, ROOT / "registry"), project), llm


def recorded_planning(cfg, tmp_path):
    """The 2026-11-05 csv_profile case: one weakness stays open but does not block the plan."""
    run_dir = tmp_path / "original"
    (run_dir / "seed").mkdir(parents=True)
    replies = [critique(), ANALYSIS, edit(), recheck("W1", blocks=False), recheck(), vote("A"), vote("A")]
    pl, llm = planning(cfg, run_dir, {"arc": Scripted(*replies), "openai": Scripted(), "claude": Scripted()}, tmp_path)
    out = pl.run()
    assert out.winner == "A" and out.problems == []
    return run_dir, llm, out


def test_a_recorded_planning_run_replays_offline_to_the_same_result(cfg, tmp_path):
    run_dir, llm, original = recorded_planning(cfg, tmp_path)
    rec = record_run(run_dir, tmp_path / "planning.json", llm.cache, public_data=True)
    assert sum(len(v) for v in rec["replies"].values()) == 7 and "plan-recheck-b" in rec["replies"]

    cfg2 = cfg.model_copy(deep=True)
    cfg2.root = tmp_path / "replay"
    providers, provider = replay_providers(cfg2, load(tmp_path / "planning.json"))
    replay_dir = tmp_path / "replay-run"
    (replay_dir / "seed").mkdir(parents=True)
    pl, _ = planning(cfg2, replay_dir, providers, tmp_path / "replay-project")
    out = pl.run()
    assert out.winner == original.winner and out.votes == original.votes and out.problems == original.problems
    assert provider.unused() == {}  # the code asked for every recorded reply, and none more


def test_replay_matches_by_stage_so_a_changed_prompt_still_gets_its_reply(cfg, tmp_path):
    run_dir, llm, _ = recorded_planning(cfg, tmp_path)
    rec = record_run(run_dir, tmp_path / "p.json", llm.cache, public_data=True)
    providers, provider = replay_providers(cfg, rec)
    from factory.llm.providers import Request
    tag = "plan-recheck-b"
    got = provider.complete(Request(model="m", messages=[{"role": "user", "content": "a prompt never seen"}],
                                    max_tokens=10, tag=tag))
    assert got.text == rec["replies"][tag][0]["text"]


def test_a_stage_that_asks_for_more_than_was_recorded_fails_and_sends_nothing(cfg, tmp_path):
    from factory.llm.providers import Request
    run_dir, llm, _ = recorded_planning(cfg, tmp_path)
    rec = record_run(run_dir, tmp_path / "p.json", llm.cache, public_data=True)
    _, provider = replay_providers(cfg, rec)
    ask = Request(model="m", messages=[], max_tokens=1, tag="plan-critique")
    provider.complete(ask)
    with pytest.raises(NotRecorded, match="plan-critique"):  # recorded once, asked twice
        provider.complete(ask)
    with pytest.raises(NotRecorded):
        provider.complete(Request(model="m", messages=[], max_tokens=1, tag="never-recorded"))
    assert "plan-critique" not in provider.unused()


def test_cache_hits_are_recorded_too(cfg, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg.credits = {}
    llm = LLM(cfg, run_dir, approver=approve_all, providers={"arc": Scripted("one"), "openai": Scripted(),
                                                             "claude": Scripted()})
    llm.complete("historian", "same prompt", tag="s")
    llm.complete("historian", "same prompt", tag="s")  # answered from the cache
    rec = record_run(run_dir, tmp_path / "r.json", llm.cache, public_data=True)
    assert [r["text"] for r in rec["replies"]["s"]] == ["one", "one"]


def test_recording_needs_the_public_data_statement(cfg, tmp_path):
    run_dir, llm, _ = recorded_planning(cfg, tmp_path)
    with pytest.raises(RefusedRecording, match="public"):
        record_run(run_dir, tmp_path / "p.json", llm.cache)
    assert not (tmp_path / "p.json").exists()


def test_a_reply_with_a_key_or_a_deny_listed_line_is_not_written(cfg, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    llm = LLM(cfg, run_dir, approver=approve_all, providers={
        "arc": Scripted("token sk-ant-" + "a" * 30, "the lab protein list: Q9UBN7 and Q9Y6K9"), "openai": Scripted(),
        "claude": Scripted()})
    llm.complete("historian", "p1", tag="a")
    with pytest.raises(SecretFound):
        record_run(run_dir, tmp_path / "k.json", llm.cache, public_data=True)
    run2 = tmp_path / "run2"
    run2.mkdir()
    llm2 = LLM(cfg, run2, use_cache=True, approver=approve_all, providers={
        "arc": Scripted("the lab protein list: Q9UBN7 and Q9Y6K9"), "openai": Scripted(), "claude": Scripted()})
    llm2.complete("historian", "p2", tag="a")
    deny = tmp_path / "lab.txt"
    deny.write_text("Q9UBN7\nTP53\n")
    with pytest.raises(RefusedRecording, match="deny list") as e:
        record_run(run2, tmp_path / "d.json", llm2.cache, public_data=True, deny_list=deny)
    assert "Q9UBN7" not in str(e.value) and not (tmp_path / "d.json").exists()


def test_replay_does_not_change_cache_keys():
    from factory.llm.providers import Request
    a = Request(model="m", messages=[], max_tokens=1)
    b = Request(model="m", messages=[], max_tokens=1, tag="stage")
    assert a.as_dict() == b.as_dict()


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")) or [pytest.param(None, marks=pytest.mark.skip(
    reason="no recorded runs committed yet (python -m factory.replay record ...)"))], ids=lambda p: getattr(p, "name", ""))
def test_committed_recordings_hold_no_secrets(path):
    from factory.secrets import scan_text
    assert scan_text(path.read_text(encoding="utf-8")) == []

