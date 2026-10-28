"""The registry loads the shipped agent specs and awards work by deterministic contract-net bids."""

from pathlib import Path

import pytest

from factory.config import load_config
from factory.registry import AgentSpec, Announcement, Registry, ScoreBook

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs" / "factory.yaml")


@pytest.fixture
def registry(cfg, tmp_path):
    reg = Registry.load(cfg)
    reg.scores = ScoreBook(tmp_path / "scores.json")  # never touch the repo's track record in tests
    return reg


def spec(name, **kw):
    base = {"name": name, "description": "d", "kind": "tool", "entrypoint": "m:f", "capabilities": ["write_patch"]}
    return AgentSpec(**{**base, **kw})


def test_shipped_specs_load_and_use_configured_roles(registry):
    assert {"planner", "code_navigator", "coder", "test_runner", "reviewer"} <= set(registry.agents)
    assert registry.agents["test_runner"].kind == "tool"


def test_only_agents_with_the_capability_bid(registry):
    bids = registry.bids(Announcement(capability="run_tests", domain="software"))
    assert [b.agent.name for b in bids] == ["test_runner"]
    assert registry.award(Announcement(capability="compose_music")) is None


def test_plain_code_outbids_a_model_for_the_same_job(cfg, tmp_path):
    reg = Registry([spec("script"), spec("llm_writer", kind="llm", role="coder", entrypoint=None)], cfg,
                   ScoreBook(tmp_path / "s.json"))
    award = reg.award(Announcement(capability="write_patch"))
    assert award.agent.name == "script" and award.parts["tool"] > 0


def test_track_record_moves_the_award(cfg, tmp_path):
    reg = Registry([spec("a"), spec("b")], cfg, ScoreBook(tmp_path / "s.json"))
    assert reg.award(Announcement(capability="write_patch", task_type="bug_fix")).agent.name == "a"  # tie: by name
    for _ in range(3):
        reg.scores.record("a", "bug_fix", passed=False)
        reg.scores.record("b", "bug_fix", passed=True)
    assert reg.award(Announcement(capability="write_patch", task_type="bug_fix")).agent.name == "b"
    assert ScoreBook(tmp_path / "s.json").pass_rate("b", "bug_fix") == 1.0  # persisted


def test_domain_fit_counts(cfg, tmp_path):
    reg = Registry([spec("general"), spec("finance_only", domains=["finance"])], cfg, ScoreBook(tmp_path / "s.json"))
    assert reg.award(Announcement(capability="write_patch", domain="finance")).score == \
        reg.award(Announcement(capability="write_patch", domain="any")).score
    bids = {b.agent.name: b for b in reg.bids(Announcement(capability="write_patch", domain="software"))}
    assert bids["general"].parts["domain"] > bids["finance_only"].parts["domain"]


def test_specs_are_validated(cfg, tmp_path):
    with pytest.raises(ValueError, match="needs a role"):
        AgentSpec(name="x", description="d", kind="llm", capabilities=["c"])
    with pytest.raises(ValueError, match="entrypoint"):
        AgentSpec(name="x", description="d", kind="tool", capabilities=["c"])
    with pytest.raises(ValueError, match="does not define"):
        Registry([spec("x", kind="llm", role="nobody", entrypoint=None)], cfg, ScoreBook(tmp_path / "s.json"))
    with pytest.raises(ValueError, match="duplicate"):
        Registry([spec("x"), spec("x")], cfg, ScoreBook(tmp_path / "s.json"))


def test_every_shipped_tool_agent_points_at_a_real_function(registry):
    for agent in registry.agents.values():
        if agent.kind == "tool":
            assert callable(agent.function()), agent.name
