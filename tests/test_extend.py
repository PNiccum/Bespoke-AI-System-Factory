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
    return Outcome(task=task, mode="aspire", solved=solved, why="", tokens=tokens)


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


@pytest.fixture
def own_registry(tmp_path):
    """A copy of the registry that adoption may write to."""
    import shutil
    shutil.copytree(ROOT / "registry", tmp_path / "registry")
    return tmp_path / "registry"


def adopt_run(cfg, monkeypatch, own, agent, replies, **kw):
    """propose() for ``agent`` with adoption on, the benchmark stubbed to pass, and the scripted panel replies."""
    monkeypatch.setattr(extend, "evaluate", lambda cfg, tasks, agents, **_: [outcome(t.name, True, 800) for t in tasks])
    arc = Scripted(GOOD.model_copy(update={"name": f"{agent}_v2"}).model_dump_json(), *replies)
    return propose(cfg, agent, providers={"arc": arc, "openai": Scripted(), "claude": Scripted()}, registry_dir=own,
                   tasks_file=ROOT / "bench" / "tasks.yaml", dev_runs=[], adopt_if_passed=True, **kw)


def test_a_passed_prompt_change_is_adopted_logged_and_can_be_rolled_back(cfg, monkeypatch, own_registry):
    before = (own_registry / "agents" / "coder.yaml").read_text()
    result = adopt_run(cfg, monkeypatch, own_registry, "coder", [ACCEPT])
    assert result.status == "adopted" and "--rollback coder" in (result.folder / "report.md").read_text()
    spec = Registry.load(cfg, own_registry).agents["coder"]
    assert spec.name == "coder" and spec.prompt == GOOD.prompt and spec.role == "coder"  # same name: the pipeline uses it
    assert "adopted a new prompt for `coder`" in (own_registry / "history" / "changelog.md").read_text()
    extend.rollback(cfg, "coder", own_registry)
    assert (own_registry / "agents" / "coder.yaml").read_text() == before
    assert "rolled back `coder`" in (own_registry / "history" / "changelog.md").read_text()


def test_the_guidance_reaches_the_proposer(cfg, monkeypatch, own_registry):
    monkeypatch.setattr(extend, "evaluate", lambda cfg, tasks, agents, **_: [outcome(t.name, True, 800) for t in tasks])
    arc = Scripted(GOOD.model_dump_json(), ACCEPT)
    propose(cfg, "coder", providers={"arc": arc, "openai": Scripted(), "claude": Scripted()}, registry_dir=own_registry,
            tasks_file=ROOT / "bench" / "tasks.yaml", dev_runs=[], guidance="cite the wiki page")
    assert "cite the wiki page" in arc.requests[0].messages[0]["content"]


def test_an_agent_the_benchmark_cannot_judge_gets_two_panel_families(cfg, monkeypatch, own_registry):
    monkeypatch.setattr(extend, "evaluate", lambda *a, **k: pytest.fail("the benchmark does not exercise the builder"))
    arc = Scripted(GOOD.model_copy(update={"name": "builder_v2"}).model_dump_json(), ACCEPT, ACCEPT)
    result = propose(cfg, "builder", providers={"arc": arc, "openai": Scripted(), "claude": Scripted()},
                     registry_dir=own_registry, tasks_file=ROOT / "bench" / "tasks.yaml", adopt_if_passed=True)
    assert result.status == "adopted" and "not exercised" in result.gates[1].detail
    families = [cfg.model_info(r.model).family for r in arc.requests[1:]]
    assert families == ["deepseek", "openai"]  # DeepSeek, then gpt-oss


def test_a_judges_prompt_is_never_adopted_without_a_person(cfg, monkeypatch, own_registry):
    before = (own_registry / "agents" / "reviewer.yaml").read_text()
    result = adopt_run(cfg, monkeypatch, own_registry, "reviewer", [ACCEPT])
    assert result.status == "awaiting a person" and "structural" in result.gates[-1].detail
    assert (own_registry / "agents" / "reviewer.yaml").read_text() == before


def test_adoptions_stop_at_the_weekly_limit(cfg, monkeypatch, own_registry):
    for n in range(extend.MAX_ADOPTIONS_PER_WEEK):
        (own_registry / "history" / f"a{n}").mkdir(parents=True)
        (own_registry / "history" / f"a{n}" / "x.yaml").write_text("x")
    result = adopt_run(cfg, monkeypatch, own_registry, "coder", [ACCEPT])
    assert result.status == "awaiting a person" and "this week" in result.gates[-1].detail


def test_a_person_can_adopt_past_the_limits(cfg, monkeypatch, own_registry):
    result = adopt_run(cfg, monkeypatch, own_registry, "reviewer", [ACCEPT], by_person=True)
    assert result.status == "adopted"
