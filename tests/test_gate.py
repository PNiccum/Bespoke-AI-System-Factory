"""The spend gate: paid calls need approval and remaining credit; free calls never wait."""

import json

import pytest

from factory.config import Budget, Credits
from factory.llm import LLM, BudgetExceeded, Usage
from factory.llm.gate import SpendDenied, SpendLedger, SpendRequest, deny_all, estimate_usd
from factory.llm.providers import RawResult
from tests.helpers import paid_config


class FakeProvider:
    def __init__(self, n=10, inp=1000, out=500):
        self.calls = 0
        self.inp, self.out = inp, out

    def complete(self, req):
        self.calls += 1
        return RawResult(f"reply {self.calls}", "", "end_turn", Usage(self.inp, self.out))


class Recorder:
    """An approver that answers from a script and remembers what it was shown."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.seen: list[SpendRequest] = []

    def __call__(self, req):
        self.seen.append(req)
        return self.answers.pop(0)


@pytest.fixture
def cfg(tmp_path):
    cfg = paid_config(tmp_path)
    cfg.root = tmp_path
    return cfg


def make(cfg, tmp_path, approver=None, **fake):
    p = {"arc": FakeProvider(**fake), "claude": FakeProvider(**fake), "openai": FakeProvider(**fake)}
    return LLM(cfg, tmp_path, use_cache=False, providers=p, approver=approver), p


def test_free_calls_never_ask(cfg, tmp_path):
    llm, p = make(cfg, tmp_path, approver=deny_all)
    llm.complete("writer", "x")
    llm.complete("coder", "x")
    assert p["arc"].calls == 2


def test_without_an_approver_every_paid_call_is_refused(cfg, tmp_path):
    llm, p = make(cfg, tmp_path)
    with pytest.raises(SpendDenied, match="first paid call"):
        llm.complete("agent", "x", data="public")
    assert p["claude"].calls == 0


def test_first_paid_call_per_role_asks_once_then_runs_freely_under_the_soft_limit(cfg, tmp_path):
    cfg.budget = Budget(approve_above_usd=5.0)
    ask = Recorder(True)
    llm, p = make(cfg, tmp_path, approver=ask)
    llm.complete("agent", "one", data="public")
    llm.complete("agent", "two", data="public")
    assert p["claude"].calls == 2 and len(ask.seen) == 1
    assert ask.seen[0].reason == "first paid call for this role" and ask.seen[0].model == "claude-sonnet-5-5"


def test_escalation_goes_through_the_gate_to_the_paid_step(cfg, tmp_path):
    ask = Recorder(False, True)
    llm, p = make(cfg, tmp_path, approver=ask)
    with pytest.raises(SpendDenied):
        llm.complete("coder", "fix it", data="public", escalate=True)
    assert p["openai"].calls == 0
    c = llm.complete("coder", "fix it", data="public", escalate=True)
    assert c.model == "gpt-6.1-sol" and p["openai"].calls == 1
    assert [r.provider for r in ask.seen] == ["openai", "openai"]


def test_passing_the_soft_limit_asks_again(cfg, tmp_path):
    cfg.budget = Budget(approve_above_usd=0.05)
    ask = Recorder(True, False)
    llm, p = make(cfg, tmp_path, approver=ask, inp=10_000, out=4_000)  # about $0.06 a call at Sonnet prices
    llm.complete("agent", "one", data="public", max_tokens=100)
    with pytest.raises(SpendDenied, match="soft limit"):
        llm.complete("agent", "two", data="public", max_tokens=100)
    assert p["claude"].calls == 1 and ask.seen[1].run_spent_usd > 0.05


def test_gate_decisions_are_logged_without_prompt_text(cfg, tmp_path):
    llm, _ = make(cfg, tmp_path, approver=Recorder(True))
    llm.complete("agent", "SECRET", data="public")
    lines = [json.loads(l) for l in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    gate = [l for l in lines if l.get("event") == "spend_gate"]
    assert gate and gate[0]["reason"] == "first paid call for this role"
    assert "prompt_sha256" in lines[-1] and "SECRET" not in (tmp_path / "llm_calls.jsonl").read_text()


def test_spend_ledger_persists_across_runs_and_stops_at_a_providers_credit(tmp_path):
    credits = Credits(by_provider={"claude": 1.0, "openai": 1.0}, reserve_usd=0.0)
    path = tmp_path / "spend.json"
    SpendLedger(path, credits).add("claude", 0.6)
    later = SpendLedger(path, credits)
    later.check("claude")
    later.add("claude", 0.5)
    with pytest.raises(BudgetExceeded, match="claude credit used up"):
        later.check("claude")
    later.check("openai")


def test_the_demo_reserve_is_never_spent(tmp_path):
    credits = Credits(by_provider={"claude": 20.0, "openai": 15.0}, reserve_usd=10.0)
    ledger = SpendLedger(tmp_path / "spend.json", credits)
    ledger.add("claude", 15.0)
    ledger.add("openai", 10.0)  # $25 spent: what is left is the reserve
    with pytest.raises(BudgetExceeded, match="reserve"):
        ledger.check("openai")


def test_credit_is_checked_before_a_person_is_asked(cfg, tmp_path):
    cfg.credits = Credits(by_provider={"claude": 0.01}, reserve_usd=0.0)
    (cfg.work_dir).mkdir(parents=True)
    (cfg.work_dir / "spend.json").write_text(json.dumps({"claude": 0.02}))
    ask = Recorder(True)
    llm, p = make(cfg, tmp_path, approver=ask)
    with pytest.raises(BudgetExceeded):
        llm.complete("agent", "x", data="public")
    assert ask.seen == [] and p["claude"].calls == 0


def test_estimate_is_an_upper_bound_from_prompt_length_and_max_tokens(cfg):
    sonnet = cfg.model_info("claude-sonnet-5-5")  # $2 in, $10 out per million
    assert estimate_usd(sonnet, 4_000_000, 0) == pytest.approx(2.0)
    assert estimate_usd(sonnet, 0, 100_000) == pytest.approx(1.0)
