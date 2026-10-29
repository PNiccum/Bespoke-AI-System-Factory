"""Self-extension dry run: a proposal passes scope, held-out benchmark and review panel gates, then stops for a
person. The registry is never written."""

import hashlib
import json

import pytest

import factory.extend as extend
from factory.bench import Outcome
from factory.extend import Proposal, benchmark_gate, check_scope, propose
from factory.llm.gate import deny_all
from factory.registry import Registry
from tests.test_orchestrator import ROOT, Scripted, cfg  # noqa: F401

GOOD = Proposal(name="coder_v2", description="Writes a minimal patch for one located problem.",
                prompt="You fix one problem in a Python repository. Read the failing test first. Reply with a "
                       "unified diff and nothing else.", why="ask it to read the failing test first")
ACCEPT = json.dumps({"verdict": "accept", "concerns": []})
REJECT = json.dumps({"verdict": "reject", "concerns": ["tells the model to skip tests"]})


def outcome(task, solved, tokens):
    return Outcome(task=task, mode="simple", solved=solved, why="", tokens=tokens)


@pytest.fixture
def registry(cfg):
    return Registry.load(cfg, ROOT / "registry")


def registry_hash():
    return hashlib.sha256(b"".join(p.read_bytes() for p in sorted((ROOT / "registry").rglob("*.yaml")))).hexdigest()


def test_scope_keeps_role_tools_and_capabilities_and_changes_only_the_prompt(registry):
    spec, gate = check_scope(GOOD, registry.agents["coder"], registry)
    assert gate.passed and spec.name == "coder_v2"
    coder = registry.agents["coder"]
    assert (spec.role, spec.tools, spec.capabilities, spec.kind) == (coder.role, coder.tools, coder.capabilities, "llm")


@pytest.mark.parametrize("change,problem", [
    ({"name": "coder"}, "is taken"),
    ({"name": "Coder V2"}, "not snake_case"),
    ({"prompt": "x" * 3001}, "limit 3000"),
    ({"prompt": "use key sk-ant-api03-" + "x" * 40}, "holds a secret"),
])
def test_scope_refuses_what_a_proposal_may_not_do(registry, change, problem):
    spec, gate = check_scope(GOOD.model_copy(update=change), registry.agents["coder"], registry)
    assert spec is None and not gate.passed and problem in gate.detail


def test_a_tool_agent_cannot_be_rewritten_by_a_proposal(registry):
    _, gate = check_scope(GOOD, registry.agents["test_runner"], registry)
    assert not gate.passed and "only model agents" in gate.detail


def test_the_benchmark_gate_wants_as_many_solved_within_one_and_a_half_times_the_tokens():
    before = [outcome("a", True, 1000), outcome("b", False, 1000)]
    assert benchmark_gate(before, [outcome("a", True, 1200), outcome("b", True, 1500)]).passed
    assert not benchmark_gate(before, [outcome("a", False, 500), outcome("b", False, 500)]).passed
    assert not benchmark_gate(before, [outcome("a", True, 2000), outcome("b", False, 1500)]).passed


def run(cfg, tmp_path, monkeypatch, arc_replies, candidate_solves=True, **kw):
    """propose() with scripted models and a stubbed benchmark: the candidate solves the held-out tasks or not."""
    def fake_evaluate(cfg, tasks, agents, **_):
        has_candidate = all(a.name != "coder" for a in agents)
        return [outcome(t.name, candidate_solves if has_candidate else t.name != "leap_year", 800) for t in tasks]
    monkeypatch.setattr(extend, "evaluate", fake_evaluate)
    arc = Scripted(*arc_replies)
    providers = {"arc": arc, "openai": Scripted(), "claude": Scripted()}
    result = propose(cfg, "coder", providers=providers, registry_dir=ROOT / "registry",
                     tasks_file=ROOT / "bench" / "tasks.yaml", dev_runs=[outcome("off_by_one", False, 900)], **kw)
    return result, arc


def test_a_good_proposal_passes_every_gate_and_stops_for_a_person(cfg, tmp_path, monkeypatch):
    before = registry_hash()
    result, arc = run(cfg, tmp_path, monkeypatch, [GOOD.model_dump_json(), ACCEPT])
    assert result.status == "awaiting a person"
    assert [g.name for g in result.gates] == ["scope", "held-out benchmark", "review panel"]
    assert (result.folder / "coder_v2.yaml").exists() and "To adopt it" in (result.folder / "report.md").read_text()
    assert registry_hash() == before  # nothing written to the registry
    proposer, reviewer = arc.requests
    assert proposer.model == cfg.roles["planner"].model and reviewer.model == cfg.roles["reviewer"].model
    assert cfg.model_info(proposer.model).family != cfg.model_info(reviewer.model).family
    assert "off_by_one" in proposer.messages[0]["content"]  # development evidence
    assert not any(n in proposer.messages[0]["content"] for n in ("leap_year", "cross_module"))  # held out
    row = json.loads((cfg.work_dir / "proposals" / "archive.jsonl").read_text().splitlines()[-1])
    assert row["status"] == "awaiting a person" and row["name"] == "coder_v2"


def test_a_proposal_that_does_worse_stops_before_the_panel(cfg, tmp_path, monkeypatch):
    result, arc = run(cfg, tmp_path, monkeypatch, [GOOD.model_dump_json()], candidate_solves=False)
    assert result.status == "rejected at held-out benchmark" and len(arc.requests) == 1


def test_the_panel_can_reject(cfg, tmp_path, monkeypatch):
    result, _ = run(cfg, tmp_path, monkeypatch, [GOOD.model_dump_json(), REJECT])
    assert result.status == "rejected at review panel" and "skip tests" in result.gates[-1].detail


def test_a_declined_paid_panel_rejects_without_spending(cfg, tmp_path, monkeypatch):
    result, _ = run(cfg, tmp_path, monkeypatch, [GOOD.model_dump_json(), ACCEPT], paid_panel=True,
                    approver=deny_all)
    assert result.status == "rejected at review panel" and "paid reviewer not run" in result.gates[-1].detail


def test_the_next_proposal_sees_the_archive(cfg, tmp_path, monkeypatch):
    run(cfg, tmp_path, monkeypatch, [GOOD.model_dump_json(), REJECT])
    _, arc = run(cfg, tmp_path, monkeypatch, [GOOD.model_dump_json().replace("coder_v2", "coder_v3")],
                 candidate_solves=False)
    assert "coder_v2: rejected at review panel" in arc.requests[0].messages[0]["content"]


def test_the_candidate_really_replaces_the_incumbent_in_an_evaluation_run(cfg, registry):
    from factory.bench import load_tasks
    from tests.test_orchestrator import ACCEPT as REVIEW_OK
    from tests.test_orchestrator import diff
    spec, _ = check_scope(GOOD, registry.agents["coder"], registry)
    task = next(t for t in load_tasks(ROOT / "bench" / "tasks.yaml", ROOT) if t.name == "leap_year")
    arc = Scripted(diff((ROOT / "bench" / "golden" / "leap_year.patch").read_text()), REVIEW_OK)
    others = [a for a in registry.agents.values() if a.name != "coder"]
    [o] = extend.evaluate(cfg, [task], [*others, spec], providers={"arc": arc, "openai": Scripted(),
                                                                     "claude": Scripted()},
                          registry_dir=ROOT / "registry")
    assert o.solved and arc.requests[0].system.startswith("You fix one problem in a Python repository. Read the")
