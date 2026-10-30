"""The re-design loop's audits, paid plan, knowledge gaps and failure archive. Models are scripted."""

import json

from factory.failures import Case, Cause, FailureArchive, Fix
from factory.pipeline.redesign import REDESIGN_CAP_USD, record
from tests.test_build import prompt, redesign_with
from tests.test_orchestrator import cfg  # noqa: F401


def test_a_cause_without_evidence_on_the_board_is_not_applied(cfg, tmp_path):
    result, tried, report, _, _ = redesign_with(cfg, tmp_path, [prompt("builder", "x", evidence=["9999-made-up"])])
    assert tried == [] and result.prompt_changes[0].status == "awaiting a person"
    assert result.causes[1].evidence == []  # the made-up id was dropped
    assert "[no evidence]" in report


def test_a_knowledge_gap_is_researched(cfg, tmp_path):
    asked = []

    def research(topic):
        asked.append(topic.name)
        return None
    gap = {"kind": "missing_knowledge", "evidence": ["0001-request"], "fix": "primer Tm salt correction",
           "topic": "salt-adjusted Tm"}
    result, _, report, _, _ = redesign_with(cfg, tmp_path, [gap], research=research)
    assert asked == ["salt-adjusted Tm"] and result.research[0].status == "research found nothing"
    assert "## Knowledge gaps" in report and "salt-adjusted Tm" in report
    assert not any("salt" in f for f in result.fixes)  # research, not a structural fix


def test_without_a_knowledge_loop_a_gap_is_queued(cfg, tmp_path):
    gap = {"kind": "missing_knowledge", "evidence": ["0001-request"], "fix": "x", "topic": "Primer3 settings"}
    result, _, _, _, _ = redesign_with(cfg, tmp_path, [gap])
    assert result.research[0].status == "queued for research"
    row = json.loads((cfg.work_dir / "research_queue.jsonl").read_text().splitlines()[0])
    assert row["name"] == "Primer3 settings"


def test_the_failure_is_archived_from_before_production(cfg, tmp_path):
    result, _, _, _, project = redesign_with(cfg, tmp_path, [prompt("builder", "check GC content")])
    archive = FailureArchive.of(cfg)
    case = archive.get(result.case)
    snap = archive.root / case.id / "project"
    kinds = [json.loads(line)["kind"] for line in (snap / "blackboard.jsonl").read_text().splitlines()]
    assert kinds[-1] == "knowledge" and "audit" not in kinds
    assert not (snap / "redesign").exists()
    assert [c.kind for c in case.causes] == ["tool_or_code", "prompt"]
    assert case.tried[0].agent == "builder" and case.tried[0].status == "adopted"


def test_the_improver_gets_the_case_and_the_auditors_see_past_fixes(cfg, tmp_path):
    archive = FailureArchive.of(cfg)
    archive.save(Case(id="old", at="20261001T000000Z", project="p", task="t",
                      causes=[Cause(kind="prompt", agent="builder", fix="x")],
                      fixes=[Fix(at="20261002T000000Z", agent="builder", proposal="p1", change="list every file")]))
    _, _, _, llm, _ = redesign_with(cfg, tmp_path, [])
    audit_prompt = llm.providers["arc"].requests[0].messages[0]["content"]
    assert "list every file" in audit_prompt and "missing_knowledge" in audit_prompt


def test_an_approved_paid_plan_asks_once_and_uses_the_stronger_models(cfg, tmp_path):
    asked = []

    def approver(req):
        asked.append(req)
        return True
    audit = json.dumps({"failures": ["T2 unmet"], "causes": []})
    settle = json.dumps({"problems": ["primers unchecked"], "causes": []})
    pos = json.dumps({"proposal": "add a Primer3 tool", "agrees_with_other": True})
    result, _, report, llm, _ = redesign_with(cfg, tmp_path, [], approver=approver,
                                              claude=(audit, settle, pos, pos),
                                              openai=(json.dumps({"suggestions": ["ok"]}), pos, pos))
    assert [r.role for r in asked] == ["re-design"]  # one question for the whole plan
    assert 0 < asked[0].estimate_usd <= REDESIGN_CAP_USD and "capped at $1.00" in asked[0].reason
    assert {r.model for r in llm.providers["claude"].requests} == {"claude-opus-5-5"}
    assert {r.model for r in llm.providers["openai"].requests} == {"gpt-6.1-sol"}
    assert result.agreed and "Approved: re-design plan" in report
    assert llm.ledger.summary()["real_usd"] < REDESIGN_CAP_USD


def test_the_trace_keeps_the_start_and_the_end():
    class Board:
        def entries(self):
            from factory.pipeline.blackboard import Entry
            return [Entry(id=f"{i:04d}-note", at="", kind="note", author="a", data={"x": "y" * 50}) for i in range(500)]

        for_paid = staticmethod(lambda e: e)

    class P:
        board = Board()
    text = record(P(), limit=5000)
    assert "[0000-note]" in text and "[0499-note]" in text and "entries omitted" in text and len(text) < 5200
