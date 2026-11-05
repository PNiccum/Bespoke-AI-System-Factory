"""The planner's risk list: picked from past events by code, synthetic data."""

from factory.events import Event, EventLog, run_events
from factory.llm import LLM
from factory.llm.gate import approve_all
from factory.pipeline.blackboard import Project
from factory.pipeline.interview import Interview
from factory.pipeline.planning import PlanningLoop
from factory.registry import Registry
from factory.risks import render, risk_list, risks_for
from tests.test_interview import propose
from tests.test_orchestrator import ROOT, Scripted, cfg  # noqa: F401
from tests.test_planning import ANALYSIS, critique, edit, recheck, vote


def history(base):
    a, b = EventLog(base / "r1" / "events.jsonl", "r1"), EventLog(base / "r2" / "events.jsonl", "r2")
    a.record("qa", "final QA did not approve: no test covers the CSV header row", stage="final QA")
    b.record("qa", "final QA did not approve: no test covers the CSV header row", stage="final QA")
    a.record("pipeline", "knowledge loop failed: primer3 not installed", stage="knowledge")
    a.resolve("E0002", "installed in round 2")
    a.record("llm", "timeout after 30 s", stage="x")  # transport noise: never listed
    a.record("pipeline", "gene ESummary gives no RefSeq transcripts", stage="critic loop")
    b.record("pipeline", "gene ESummary gives no RefSeq transcripts", stage="critic loop")
    b.resolve("E0002", "fixed")
    return a, b


def test_open_repeated_problems_rank_first_and_noise_and_fixed_are_left_out(tmp_path):
    history(tmp_path)
    risks = risk_list(run_events(tmp_path), "profile a CSV file and report its header row")
    assert [r.stage for r in risks] == ["final QA", "critic loop"]  # overlap with the task ranks QA first
    assert risks[0].open == 2 and risks[0].runs == 2
    assert all("timeout" not in r.message and "primer3" not in r.message for r in risks)


def test_a_fix_that_did_not_hold_ranks_higher(tmp_path):
    a, b = EventLog(tmp_path / "r1" / "events.jsonl", "r1"), EventLog(tmp_path / "r2" / "events.jsonl", "r2")
    a.record("pipeline", "stage alpha failed", stage="s")
    a.resolve("E0001", "fixed")
    b.record("pipeline", "stage alpha failed", stage="s")
    (r,) = risk_list(run_events(tmp_path), "anything")
    assert r.returned_after_fix and r.score >= 3


def test_domain_tag_filters_and_boosts():
    class Tagged(Event):  # PR A's domain tag, stood in for here
        domain: str = ""

    def mk(i, d):
        return Tagged(id=i, run="r", source="qa", message=f"problem {i}", fingerprint=i, domain=d)

    es = [mk("aaa", "bio"), mk("bbb", "web"), mk("ccc", "")]
    got = [r.fingerprint for r in risk_list(es, "x", domain="bio")]
    assert got == ["aaa", "ccc"]  # same domain first, untagged kept, other domain dropped
    assert len(risk_list(es, "x")) == 3


def test_empty_history_changes_nothing(tmp_path):
    assert render(risk_list([], "x")) == ""
    assert risks_for(object(), "x") == []


def test_limit_and_render(tmp_path):
    log = EventLog(tmp_path / "r1" / "events.jsonl", "r1")
    for i in range(8):
        log.record("pipeline", f"distinct failure {chr(97 + i) * 4}", stage="s")
    risks = risk_list(run_events(tmp_path), "task", limit=3)
    text = render(risks)
    assert len(risks) == 3 and text.count("\n- ") == 3 and "not instructions" in text


def test_planners_see_the_risks_and_the_pick_is_recorded(cfg, tmp_path):
    history(tmp_path / "runs")
    llm = LLM(
        cfg,
        tmp_path,
        use_cache=False,
        providers={"arc": Scripted(propose()), "openai": Scripted(), "claude": Scripted()},
        approver=approve_all,
    )
    cfg.root = tmp_path
    project = Project.create(tmp_path / "projects", "csv")
    iv = Interview(llm, Registry.load(cfg, ROOT / "registry"), project)
    iv.start("Profile a CSV file")
    iv.approve()
    arc = Scripted(critique(), ANALYSIS, edit(), recheck(), recheck(), vote("B"), vote("B"))
    llm.providers["arc"] = arc
    PlanningLoop(llm, Registry.load(cfg, ROOT / "registry"), project).run()
    assert (
        "Known risks from earlier runs" in arc.requests[0].messages[-1]["content"]
        and "CSV header row" in arc.requests[0].messages[-1]["content"]
    )
    note = next(e for e in project.board.entries() if e.kind == "note")
    assert note.data["risks"][0]["stage"] == "final QA"
