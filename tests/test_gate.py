"""The spend gate: paid calls need approval and remaining credit; free calls never wait."""

import json
from pathlib import Path

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


def _add_in_process(args):
    path, n = args
    ledger = SpendLedger(Path(path), Credits(by_provider={"claude": 1000.0}, reserve_usd=0.0))
    for _ in range(n):
        ledger.add("claude", 0.01)


def test_parallel_processes_do_not_overwrite_each_others_totals(tmp_path):
    from multiprocessing import Pool
    path = tmp_path / "spend.json"
    with Pool(6) as pool:
        pool.map(_add_in_process, [(str(path), 25)] * 6)
    assert SpendLedger(path, Credits(by_provider={"claude": 1000.0}, reserve_usd=0.0)).spent["claude"] == pytest.approx(1.5)


def test_concurrent_reservations_cannot_all_pass_before_any_is_charged(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    credits = Credits(by_provider={"claude": 1.0}, reserve_usd=0.0)
    ledgers = [SpendLedger(tmp_path / "spend.json", credits) for _ in range(8)]  # separate runs, one workspace

    def go(l):
        try:
            return l.reserve("claude", 0.3)
        except BudgetExceeded:
            return None
    with ThreadPoolExecutor(8) as pool:
        got = [r for r in pool.map(go, ledgers) if r]
    assert len(got) == 3  # 3 x $0.30 fit in $1.00, the rest are refused
    ledgers[0].release(got[0])
    assert ledgers[1].reserve("claude", 0.3)  # a released hold frees its credit


def test_settle_replaces_the_hold_with_the_real_cost_and_the_reserve_still_binds(tmp_path):
    credits = Credits(by_provider={"claude": 20.0, "openai": 15.0}, reserve_usd=10.0)
    ledger = SpendLedger(tmp_path / "spend.json", credits)
    rid = ledger.reserve("claude", 1.0)
    ledger.settle(rid, "claude", 0.4)
    assert ledger.spent == {"claude": 0.4}
    assert not ledger.reserved_path.read_text().strip("{} \n")
    ledger.add("claude", 19.0)  # $19.40 spent; $25 is the most that may be spent
    with pytest.raises(BudgetExceeded, match="reserve"):
        ledger.reserve("openai", 6.0)


def test_a_failed_paid_call_releases_its_hold(cfg, tmp_path):
    llm, p = make(cfg, tmp_path, approver=Recorder(True))
    def boom(req):
        raise ValueError("boom")
    p["claude"].complete = boom
    with pytest.raises(ValueError):
        llm.complete("agent", "x", data="public")
    assert llm.ledger.held_usd == 0.0
    assert not llm.spend._read_reserved()


def test_run_cap_counts_calls_in_flight(cfg, tmp_path):
    from factory.llm.cost import Ledger
    ledger = Ledger(Budget(max_real_usd=1.0))
    ledger.hold(0.6)
    ledger.hold(0.6)  # the first hold alone is under the cap, so this passes; the cap is now spoken for
    with pytest.raises(BudgetExceeded, match="in flight"):
        ledger.hold(0.1)
    ledger.unhold(0.6)
    ledger.unhold(0.6)
    assert ledger.held_usd == 0.0
    free = Ledger(Budget(max_real_usd=0.0))
    free.check(paid=False)  # free calls are never blocked by the dollar cap


def test_a_corrupt_ledger_file_fails_closed(tmp_path):
    path = tmp_path / "spend.json"
    path.write_text("")
    with pytest.raises(BudgetExceeded, match="unreadable"):
        SpendLedger(path, Credits(by_provider={"claude": 5.0}, reserve_usd=0.0)).check("claude")


def test_a_stale_reservation_stops_counting_and_a_corrupt_reservation_file_fails_closed(tmp_path):
    credits = Credits(by_provider={"claude": 1.0}, reserve_usd=0.0)
    ledger = SpendLedger(tmp_path / "spend.json", credits)
    ledger.reserve("claude", 0.9, ttl_s=0.0)  # lapses at once
    ledger.reserve("claude", 0.9)  # would be refused if the stale hold still counted
    ledger.reserved_path.write_text("{not json")
    with pytest.raises(BudgetExceeded, match="unreadable"):
        ledger.reserve("claude", 0.1)


def test_partial_settle_keeps_the_rest_of_the_hold(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", Credits(by_provider={"claude": 5.0}, reserve_usd=0.0))
    rid = ledger.reserve("claude", 1.0)
    ledger.settle(rid, "claude", 0.1, part=0.4)
    assert ledger._read_reserved()[rid]["usd"] == pytest.approx(0.6) and ledger.spent["claude"] == pytest.approx(0.1)
    ledger.settle(rid, "claude", 0.0, part=0.6)
    assert ledger._read_reserved() == {}


def test_a_settle_failure_after_a_paid_call_is_logged_with_its_cost(cfg, tmp_path):
    llm, _ = make(cfg, tmp_path, approver=Recorder(True))

    def broken(*a, **k):
        raise OSError("disk full")
    llm.spend.settle = broken
    with pytest.raises(OSError):
        llm.complete("agent", "x", data="public")
    lines = [json.loads(l) for l in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    lost = [l for l in lines if l.get("event") == "spend_unrecorded"]
    assert lost and lost[0]["real_usd"] > 0
    assert llm.ledger.held_usd == 0.0


def test_settle_records_spend_before_dropping_the_hold(tmp_path, monkeypatch):
    ledger = SpendLedger(tmp_path / "spend.json", Credits(by_provider={"claude": 5.0}, reserve_usd=0.0))
    rid = ledger.reserve("claude", 1.0)
    real_write = SpendLedger._write

    def fail_on_reservations(path, data):
        if path.name.endswith(".reserved.json"):
            raise OSError("disk full")
        real_write(path, data)
    monkeypatch.setattr(SpendLedger, "_write", staticmethod(fail_on_reservations))
    with pytest.raises(OSError):
        ledger.settle(rid, "claude", 0.4)
    assert ledger.spent["claude"] == pytest.approx(0.4)  # the cost is on the books
    assert rid in ledger._read_reserved()  # and the hold still covers it (conservative, never under-counted)


def test_a_failed_release_still_drops_the_runs_in_memory_hold(cfg, tmp_path):
    llm, _ = make(cfg, tmp_path, approver=Recorder(True))

    def broken(rid):
        raise OSError("disk full")
    llm.spend.release = broken
    llm.ledger.hold(0.5)
    from factory.llm.client import _Hold
    llm._release(_Hold("rid", 0.5))  # logged, not raised: it must not mask the call's own result
    assert llm.ledger.held_usd == 0.0


def test_a_corrupt_reservations_file_fails_settle_before_any_cost_is_written(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", Credits(by_provider={"claude": 5.0}, reserve_usd=0.0))
    rid = ledger.reserve("claude", 1.0)
    ledger.reserved_path.write_text("[]")
    with pytest.raises(BudgetExceeded):
        ledger.settle(rid, "claude", 0.4)
    assert not ledger.path.exists()
    (tmp_path / "spend.json").write_text("[]")
    with pytest.raises(BudgetExceeded, match="unreadable"):
        ledger.check("claude")


def test_reserve_replaces_an_earlier_hold_atomically(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", Credits(by_provider={"claude": 1.0}, reserve_usd=0.0))
    old = ledger.reserve("claude", 0.8)
    new = ledger.reserve("claude", 0.9, replaces=old)  # fits only because the old hold is swapped out
    assert set(ledger._read_reserved()) == {new}
    with pytest.raises(BudgetExceeded):
        ledger.reserve("claude", 1.5, replaces=new)  # refused: the hold it would replace stays
    assert set(ledger._read_reserved()) == {new}
