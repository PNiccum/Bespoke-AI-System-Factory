"""Production, the critic loop, final QA and re-design: models write and judge, code writes files and decides."""

import json
import zipfile

import pytest

from factory.llm import Usage
from factory.llm.gate import approve_all, deny_all
from factory.llm.providers import RawResult
from factory.pipeline import critique as critique_module
from factory.pipeline.critique import CriticLoop
from factory.pipeline.knowledge import KnowledgeLoop
from factory.pipeline.production import Production
from factory.pipeline.redesign import Redesign
from factory.pipeline.run import build
from factory.pipeline.wiki import Wiki, WikiPage
from factory.registry import Registry
from tests.test_knowledge import planned, topics
from tests.test_orchestrator import ROOT, Scripted, cfg  # noqa: F401

CODE = "def tm(seq):\n    return 2 * (seq.count('A') + seq.count('T')) + 4 * (seq.count('G') + seq.count('C'))\n"
TEST = "from primer import tm\n\n\ndef test_tm():\n    assert tm('ATGC') == 12\n"
WRONG = "from primer import tm\n\n\ndef test_tm():\n    assert tm('ATGC') == 99\n"


def design():
    return json.dumps({"language": "python", "entry_point": "primer.py", "run_command": "python primer.py ACGT",
                       "layout": ["primer.py: Tm by the Wallace rule", "tests/test_primer.py: checks"],
                       "task_notes": {"T1": "fetch", "T2": "design"}, "template": "PCR primer design"})


def orders(task, *writes):
    return json.dumps({"orders": [{"id": f"{task}.1", "write": list(writes), "read": [],
                                   "instructions": "write it"}]})


def built(*files):
    return json.dumps({"files": [{"path": p, "content": c} for p, c in files], "notes": "done"})


def review(ok=True, *issues):
    return json.dumps({"acceptable": ok, "issues": [{"id": i, "severity": "major", "where": "primer.py",
                                                     "problem": "Tm ignores salt"} for i in issues]})


def fixes(*issue_ids, content=CODE):
    return json.dumps({"fixes": [{"issue": i, "explanation": "add salt", "files": [{"path": "primer.py",
                                                                                    "content": content}]}
                                 for i in issue_ids]})


def verdicts(**accept):
    return json.dumps({"verdicts": [{"issue": i, "accept": a, "why": "ok" if a else "still wrong",
                                     "missing_knowledge": "" if a else "salt-corrected Tm"} for i, a in accept.items()]})


def qa(*met, check="python primer.py ACGT prints nothing; expected a melting temperature"):
    return json.dumps({"tasks": [{"id": t, "met": m, "evidence": "test_tm", **({} if m else {"failing_check": check})}
                                 for t, m in met], "issues": []})


PRODUCE = [design(), orders("T1", "primer.py"), built(("primer.py", CODE)),
           orders("T2", "tests/test_primer.py"), built(("tests/test_primer.py", TEST))]


def project_after_knowledge(cfg, tmp_path, approver=approve_all):
    llm, project = planned(cfg, tmp_path)
    wiki = Wiki(tmp_path / "wiki")
    wiki.save(WikiPage(topic="PCR primer design", summary="Tm rules", body="Wallace rule [a:1] [b:2]",
                       sources=["a:1", "b:2"], verified=True))
    llm.providers["arc"] = Scripted(topics("PCR primer design"))
    KnowledgeLoop(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {}).run()
    llm.gate.approver = approver
    return llm, project, wiki


def parts(cfg, llm, project, wiki):
    registry = Registry.load(cfg, ROOT / "registry")
    production = Production(llm, registry, project, wiki)
    knowledge = KnowledgeLoop(llm, registry, project, wiki, {})
    return registry, production, CriticLoop(llm, registry, project, wiki, production, knowledge)


def test_built_solution_is_reviewed_checked_and_delivered(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    arc = Scripted(*PRODUCE, review(True), qa(("T1", True), ("T2", True)))
    llm.providers["arc"] = arc
    result = build(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {})
    assert result.status == "delivered" and result.rounds == 1
    out = project.root / "output"
    assert (out / "primer.py").read_text() == CODE and json.loads((out / "run.json").read_text())["command"]
    assert "Tm by the Wallace rule" in (out / "README.md").read_text()
    names = zipfile.ZipFile(result.final.deliverable).namelist()
    assert {"primer.py", "tests/test_primer.py", "aspire/task.txt", "aspire/provenance.jsonl",
            "aspire/wiki/pcr-primer-design.md"} <= set(names)
    checks = project.board.latest("checks").data
    assert checks["tests_ok"] is True and checks["passed"]
    families = {cfg.model_info(r.model).family for r in arc.requests}
    assert families == {"glm", "deepseek"}  # builders GLM; critic and QA DeepSeek


def test_files_outside_output_and_syntax_errors_are_caught(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    llm.providers["arc"] = Scripted(design(), orders("T1", "x"), built(("../escape.py", "x"), ("bad.py", "def (:")),
                                    orders("T2", "y"), built())
    checks = Production(llm, Registry.load(cfg, ROOT / "registry"), project, wiki).run()
    assert not (project.root / "escape.py").exists()
    assert "outside" in project.board.entries("build")[0].data["refused"][0]
    assert "bad.py" in checks.syntax_errors and not checks.ok


def test_an_accepted_fix_is_applied_and_the_critic_looks_again(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    llm.providers["arc"] = Scripted(*PRODUCE[:3], orders("T2", "tests/test_primer.py"),
                                    built(("tests/test_primer.py", WRONG)))
    _, production, critic = parts(cfg, llm, project, wiki)
    checks = production.run()
    assert checks.tests_ok is False
    fixed_test = TEST
    llm.providers["arc"] = Scripted(review(False, "I1"), json.dumps({"fixes": [{"issue": "I1", "explanation": "fix",
                                    "files": [{"path": "tests/test_primer.py", "content": fixed_test}]}]}),
                                    verdicts(I1=True), review(True))
    result = critic.run(checks)
    assert result.accepted and result.rounds == 2
    assert (project.root / "output" / "tests" / "test_primer.py").read_text() == TEST


def test_a_solver_reply_that_will_not_parse_is_retried_for_the_first_issue_alone(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    llm.providers["arc"] = Scripted(*PRODUCE[:3], orders("T2", "tests/test_primer.py"),
                                    built(("tests/test_primer.py", WRONG)))
    _, production, critic = parts(cfg, llm, project, wiki)
    checks = production.run()
    cut = "=== FIX: I1 ===\nEXPLANATION: fix\n=== FILE: tests/test_primer.py ===\n" + "x = 1\n" * 50
    one = json.dumps({"fixes": [{"issue": "I1", "explanation": "fix",
                                 "files": [{"path": "tests/test_primer.py", "content": TEST}]}]})
    arc = Scripted(review(False, "I1", "I2"), cut, cut, cut, one, verdicts(I1=True), review(True))
    llm.providers["arc"] = arc
    result = critic.run(checks)
    assert result.accepted and arc.texts == []
    assert "I2" not in arc.requests[4].messages[0]["content"].split("## Issues")[1].split("## Files")[0]
    assert any("solver gave no usable fixes" in e.data.get("text", "") for e in project.board.entries("note"))


def test_declined_fixes_escalate_to_research_then_redesign(cfg, tmp_path, monkeypatch):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    llm.providers["arc"] = Scripted(*PRODUCE)
    _, production, critic = parts(cfg, llm, project, wiki)
    checks = production.run()
    researched = []
    monkeypatch.setattr(critic.knowledge, "research", lambda topic: researched.append(topic.name))
    monkeypatch.setattr(critic, "closest_pages", lambda texts, n=3: [])  # nothing in the wiki: straight to research
    llm.providers["arc"] = Scripted(review(False, "I1"), fixes("I1"), verdicts(I1=False),
                                    review(False, "I1"), fixes("I1"), verdicts(I1=False))
    result = critic.run(checks)
    assert researched == ["salt-corrected Tm"]
    assert result.redesign and result.step == "research" and result.rounds == 2


def test_wiki_pages_go_to_the_solver_after_a_first_decline(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(critique_module, "MAX_ROUNDS", 2)
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    llm.providers["arc"] = Scripted(*PRODUCE)
    _, production, critic = parts(cfg, llm, project, wiki)
    checks = production.run()
    arc = Scripted(review(False, "I1"), fixes("I1"), verdicts(I1=False), review(False, "I1"), fixes("I1"),
                   verdicts(I1=True))
    llm.providers["arc"] = arc
    result = critic.run(checks)
    assert "Wallace rule" in arc.requests[4].messages[-1]["content"]  # the second solve sees the wiki page
    assert not result.redesign


def test_failed_qa_twice_leads_to_redesign_and_the_debate_waits_for_a_person(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path, approver=deny_all)
    audit = json.dumps({"successes": ["tests"], "failures": ["T2 unmet"],
                        "causes": [{"kind": "tool_or_code", "evidence": ["0001-request"], "fix": "no Primer3"}]})
    llm.providers["arc"] = Scripted(*PRODUCE, review(True), qa(("T1", True), ("T2", False)),
                                    *PRODUCE, review(True), qa(("T1", True), ("T2", False)),
                                    audit, json.dumps({"suggestions": ["wrap primer3-py"]}),
                                    json.dumps({"problems": ["no design tool"], "causes": [
                                        {"kind": "tool_or_code", "evidence": ["0001-request"],
                                         "fix": "add Primer3 tool"}]}),
                                    "The project asked for primers; T2 was never met.")
    result = build(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {})
    assert result.status == "redesign" and result.rounds == 2
    assert result.redesign.stopped.startswith("paid call not made")
    report = (project.root / "redesign" / "report.md").read_text()
    assert "No change was applied" in report and "[tool_or_code] add Primer3 tool" in report
    assert "Declined: re-design plan" in report  # asked once for the whole paid plan, then used free auditors
    assert "acceptance not met: T2" in project.board.entries("final")[0].data["issues"][0]


def test_a_stalled_critic_still_gets_final_qa_and_a_working_result_is_delivered(cfg, tmp_path, monkeypatch):
    from factory.pipeline.critique import CriticLoop, CriticResult
    llm, project, wiki = project_after_knowledge(cfg, tmp_path, approver=deny_all)
    monkeypatch.setattr(CriticLoop, "run", lambda self, checks: CriticResult(accepted=False, redesign=True,
                                                                               rounds=3, step="research"))
    llm.providers["arc"] = Scripted(*PRODUCE, qa(("T1", True), ("T2", True)))
    result = build(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {}, redesign=False)
    assert result.status == "delivered" and result.final.approved


def test_the_paid_debate_runs_to_agreement_under_its_cap(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    llm.providers["claude"] = Scripted(json.dumps({"proposal": "add a Primer3 tool"}),
                                       json.dumps({"proposal": "add a Primer3 tool", "agrees_with_other": True}))
    llm.providers["openai"] = Scripted(json.dumps({"proposal": "wrap primer3-py"}),
                                       json.dumps({"proposal": "wrap primer3-py", "agrees_with_other": True}))
    turns, agreed, stopped = Redesign(llm, Registry.load(cfg, ROOT / "registry"), project).debate("summary")
    assert agreed and stopped == "both experts agree" and len(turns) == 4
    assert "add a Primer3 tool" in llm.providers["openai"].requests[1].messages[-1]["content"]
    assert llm.ledger.summary()["real_usd"] < 0.5


def test_production_needs_the_knowledge_loop(cfg, tmp_path):
    llm, project = planned(cfg, tmp_path)
    with pytest.raises(RuntimeError, match="knowledge loop"):
        Production(llm, Registry.load(cfg, ROOT / "registry"), project, Wiki(tmp_path / "w")).run()


def redesign_with(cfg, tmp_path, causes, research=None, approver=deny_all, claude=(), openai=()):
    """Re-design with scripted auditors and a recording improver in place of factory.extend. With the default
    approver the paid plan is declined, so every audit call goes to ARC."""
    llm, project, wiki = project_after_knowledge(cfg, tmp_path, approver=approver)
    settle = json.dumps({"problems": ["primers unchecked"], "causes": [
        {"kind": "tool_or_code", "evidence": ["0001-request"], "fix": "add Primer3 tool"}, *causes]})
    arc = [json.dumps({"failures": ["T2 unmet"]}), json.dumps({"suggestions": []}), settle, "summary"]
    if approver is not deny_all:
        arc = arc[3:]  # audit, review and settle went to the paid models
    llm.providers["arc"] = Scripted(*arc)
    llm.providers["claude"], llm.providers["openai"] = Scripted(*claude), Scripted(*openai)
    tried = []

    def improver(agent, change, case):
        tried.append((agent, case))
        return "adopted", "scope: ok"
    result = Redesign(llm, Registry.load(cfg, ROOT / "registry"), project, improver=improver,
                      research=research).run()
    return result, [a for a, _ in tried], (project.root / "redesign" / "report.md").read_text(), llm, project


def prompt(agent, change, evidence=("0001-request",)):
    return {"kind": "prompt", "agent": agent, "evidence": list(evidence), "fix": change}


def test_prompt_changes_are_applied_and_structural_fixes_wait(cfg, tmp_path):
    result, tried, report, _, _ = redesign_with(cfg, tmp_path, [
        prompt("builder", "check primer GC content"),
        prompt("final_evaluator", "be less strict"),  # a judge: structural
        prompt("test_runner", "run faster"),  # a tool agent: structural
        prompt("nobody", "x")])
    assert tried == ["builder"]
    status = {a.agent: a.status for a in result.prompt_changes}
    assert status == {"builder": "adopted", "final_evaluator": "awaiting a person"}
    assert "test_runner: run faster" in result.fixes and "[tool_or_code] add Primer3 tool" in result.fixes
    assert "1 prompt change(s) passed every gate and were applied" in report and "--rollback" in report


def test_prompt_changes_stop_at_the_limit(cfg, tmp_path):
    result, tried, _, _, _ = redesign_with(cfg, tmp_path, [prompt(a, "x") for a in
                                                            ("builder", "librarian", "summarizer")])
    assert tried == ["builder", "librarian"]
    assert result.prompt_changes[-1].status.startswith("not tried")


def test_a_work_order_with_no_usable_reply_is_recorded_and_the_build_goes_on(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    llm.providers["arc"] = Scripted(design(), orders("T1", "primer.py"), "not json", "still not", "nope",
                                    orders("T2", "tests/test_primer.py"), built(("tests/test_primer.py", TEST)))
    llm.providers["openai"] = Scripted("not json either")  # the one paid rescue attempt, approved, also fails
    checks = Production(llm, Registry.load(cfg, ROOT / "registry"), project, wiki).run()
    first, second = project.board.entries("build")
    assert first.data["written"] == [] and "no valid output after 3 attempts" in first.data["failed"]
    assert "paid rescue failed" in first.data["failed"] and len(llm.providers["openai"].requests) == 1
    assert second.data["written"] == ["tests/test_primer.py"] and checks.tests_ran


def test_a_stage_that_crashes_goes_to_redesign_not_out_of_build(cfg, tmp_path, monkeypatch):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)

    def crash(self, feedback=""):
        raise RuntimeError("design: no valid output after 3 attempts")
    monkeypatch.setattr(Production, "run", crash)
    result = build(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {}, redesign=False)
    assert result.status == "stopped" and result.rounds == 1
    note = project.board.latest("note").data
    assert note == {"stage": "production, round 1", "error": "design: no valid output after 3 attempts"}
    (event,) = [e for e in llm.events.events() if e.source == "pipeline"]
    assert event.stage == "production, round 1" and event.status == "open"  # nothing corrected it


def test_a_new_build_moves_stale_output_aside(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    project.write_file("output/old_pkg/__init__.py", "x = 1\n", "code")
    llm.providers["arc"] = Scripted(*PRODUCE, review(True), qa(("T1", True), ("T2", True)))
    assert build(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {}).status == "delivered"
    assert not (project.root / "output" / "old_pkg").exists()
    moved = list((project.root / "superseded").glob("*/old_pkg/__init__.py"))
    assert len(moved) == 1
    assert any("superseded/" in e.data.get("text", "") for e in project.board.entries("note"))


def test_tests_that_cannot_import_are_a_setup_failure(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    broken = "from missing_pkg import tm\n\n\ndef test_tm():\n    assert tm('A')\n"
    llm.providers["arc"] = Scripted(design(), orders("T1", "primer.py"), built(("primer.py", CODE)),
                                    orders("T2", "tests/test_primer.py"), built(("tests/test_primer.py", broken)))
    checks = Production(llm, Registry.load(cfg, ROOT / "registry"), project, wiki).run()
    assert "could not be imported" in checks.setup_error and "missing_pkg" in checks.setup_error
    assert critique_module.json_checks(checks).startswith("setup failure, fix this first")


def test_builder_files_come_as_raw_blocks():
    from factory.pipeline.production import parse_files

    class C:
        def __init__(self, text):
            self.text = text

        def json(self):
            return json.loads(self.text)

    body = 'print("a \\\\ b")\nx = {"k": 1}\n'
    reply = f"Here you go.\n=== FILE: output/tool.py ===\n{body}=== END FILE ===\n" \
            "=== FILE: `tests/test_tool.py` ===\ndef test_x():\n    assert True\n=== END FILE ===\nNOTES: two files"
    data = parse_files(C(reply))
    assert data["files"][0] == {"path": "output/tool.py", "content": body}
    assert data["files"][1]["path"] == "tests/test_tool.py" and data["notes"] == "two files"
    assert parse_files(C(built(("a.py", "x = 1\n"))))["files"][0]["path"] == "a.py"  # the old JSON shape
    with pytest.raises(ValueError, match="END FILE"):
        parse_files(C("=== FILE: a.py ===\nx = 1\n"))


def test_a_code_fence_around_a_file_body_is_removed_except_in_markdown():
    from factory.pipeline.production import NewFile, unfence
    assert NewFile(path="pkg/design.py", content="```python\nx = 1\n```\n").content == "x = 1\n"
    assert unfence("a.py", "  ```\nx = 1\n```") == "x = 1\n"
    assert unfence("a.py", "x = '```'\n") == "x = '```'\n"
    assert unfence("README.md", "```bash\nrun\n```\n") == "```bash\nrun\n```\n"


def test_a_review_without_acceptable_follows_from_its_issues():
    from factory.pipeline.critique import Critique
    assert Critique.model_validate({"issues": []}).acceptable is True
    assert Critique.model_validate({"issues": [{"id": "I1", "severity": "major", "where": "x", "problem": "y"}]}
                                   ).acceptable is False


def test_a_critic_with_no_usable_review_lets_the_checks_speak(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    llm.providers["arc"] = Scripted(*PRODUCE)
    _, production, critic = parts(cfg, llm, project, wiki)
    checks = production.run()
    assert checks.ok
    llm.providers["arc"] = Scripted("no json here", "still none", "nothing")
    result = critic.run(checks)
    assert result.accepted and result.rounds == 1
    assert any("critic gave no usable review" in e.data.get("text", "") for e in project.board.entries("note"))


def test_solver_fixes_come_as_raw_blocks():
    from types import SimpleNamespace

    from factory.pipeline.critique import parse_fixes
    reply = ("=== FIX: I1 ===\nEXPLANATION: add salt\n=== FILE: primer.py ===\n" + CODE + "=== END FILE ===\n"
             "=== END FIX ===\n=== FIX: I2 ===\nEXPLANATION: none needed\n=== END FIX ===")
    data = parse_fixes(SimpleNamespace(text=reply, json=lambda: {}))
    assert data["fixes"][0] == {"issue": "I1", "explanation": "add salt",
                                "files": [{"path": "primer.py", "content": CODE}]}
    assert data["fixes"][1]["files"] == []
    assert parse_fixes(SimpleNamespace(text=fixes("I1"), json=lambda: json.loads(fixes("I1"))))["fixes"][0]["issue"] \
        == "I1"


def test_accepted_fixes_that_change_nothing_escalate(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(critique_module, "MAX_ROUNDS", 1)
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    llm.providers["arc"] = Scripted(*PRODUCE[:3], orders("T2", "tests/test_primer.py"),
                                    built(("tests/test_primer.py", WRONG)))
    _, production, critic = parts(cfg, llm, project, wiki)
    checks = production.run()
    llm.providers["arc"] = Scripted(review(False, "I1"), fixes("I1"), verdicts(I1=True))  # rewrites primer.py as is
    result = critic.run(checks)
    assert not result.accepted and result.step == "wiki"
    assert any("unchanged" in e.data.get("text", "") for e in project.board.entries("note"))


def test_final_qa_judges_the_delivered_result_not_the_process():
    from factory.pipeline.final import QA_PROMPT
    assert "Judge the delivered result, not how it was made" in QA_PROMPT
    assert "an issue to list, not an unmet task" in QA_PROMPT


def test_final_qa_findings_are_repaired_before_anything_is_rejected(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path, approver=deny_all)
    llm.providers["arc"] = Scripted(*PRODUCE, review(True), qa(("T1", True), ("T2", False)),
                                    fixes("QT2", content=CODE + "# salt-corrected\n"), qa(("T1", True), ("T2", True)))
    result = build(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {}, redesign=False)
    assert result.status == "delivered" and result.rounds == 1 and result.repairs == 1
    qa_events = [e for e in llm.events.events() if e.source == "qa"]  # QA said no, then a repair got it approved
    assert qa_events and all(e.status == "corrected" and "1 QA repair" in e.how for e in qa_events)


def test_a_qa_repair_that_breaks_the_checks_is_put_back(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path, approver=deny_all)
    bad = json.dumps({"fixes": [{"issue": "QT2", "explanation": "x", "files": [
        {"path": "tests/test_primer.py", "content": WRONG}]}]})
    llm.providers["arc"] = Scripted(*PRODUCE, review(True), qa(("T1", True), ("T2", False)), bad,
                                    *PRODUCE, review(True), qa(("T1", True), ("T2", True)))
    result = build(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {}, redesign=False)
    assert result.repairs == 0 and result.rounds == 2  # the broken repair was put back; a full rebuild followed
    assert (project.root / "output" / "tests" / "test_primer.py").read_text() == TEST


def test_failing_checks_tell_the_solver_why_each_test_fails(cfg, tmp_path):
    from factory.pipeline.critique import json_checks
    llm, project, wiki = project_after_knowledge(cfg, tmp_path, approver=deny_all)
    llm.providers["arc"] = Scripted(design(), orders("T1", "primer.py"), built(("primer.py", CODE)),
                                    orders("T2", "tests/test_primer.py"), built(("tests/test_primer.py", WRONG)))
    registry = Registry.load(cfg, ROOT / "registry")
    checks = Production(llm, registry, project, wiki).run()
    assert not checks.ok and "assert 12 == 99" in " ".join(checks.reasons.values())
    text = json_checks(checks)
    assert "why each fails" in text and "the test or the code is wrong" in text


def test_qa_cannot_reject_green_tests_without_naming_a_failing_check(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path, approver=deny_all)
    llm.providers["arc"] = Scripted(*PRODUCE, review(True), qa(("T1", True), ("T2", False), check=""))
    result = build(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {}, redesign=False)
    assert result.status == "delivered" and result.repairs == 0 and result.final.approved
    t2 = next(t for t in result.final.tasks if t.id == "T2")
    assert t2.met and t2.overridden and "named no failing check" in t2.evidence


def test_qa_naming_a_test_that_passed_is_overridden_but_a_concrete_failure_stands():
    from factory.pipeline.final import QA, TaskResult, defer_to_code
    from factory.pipeline.production import Checks
    green = Checks(compiled=["a.py"], syntax_errors={}, tests_ran=True, tests_ok=True, passed=["tests/t.py::test_a"])
    qa_ = QA(tasks=[TaskResult(id="T1", met=False, evidence="x", failing_check="tests/t.py::test_a fails"),
                    TaskResult(id="T2", met=False, evidence="x", failing_check="run `a.py 3`: expected 9, got 6")])
    defer_to_code(qa_, green)
    assert qa_.tasks[0].met and not qa_.tasks[1].met
    prefix = green.model_copy(update={"passed": ["test_a"]})  # a passed id that is a prefix of a failing one
    qp = QA(tasks=[TaskResult(id="T1", met=False, evidence="x", failing_check="test_ab fails: expected 1, got 2")])
    defer_to_code(qp, prefix)
    assert not qp.tasks[0].met
    ctx = QA(tasks=[TaskResult(id="T1", met=False, evidence="x",
                               failing_check="tests/t.py::test_a passes but `a.py 3` prints 6, expected 9")])
    defer_to_code(ctx, green)  # a passing test named as context does not hide a concrete failure
    assert not ctx.tasks[0].met
    wrong = QA(tasks=[TaskResult(id="T1", met=False, evidence="x", failing_check="tests/t.py::test_a fails: got 6")])
    defer_to_code(wrong, green)  # an observed result, however short, stands
    assert not wrong.tasks[0].met
    bare = QA(tasks=[TaskResult(id="T1", met=False, evidence="x", failing_check="tests/t.py::test_a is wrong")])
    defer_to_code(bare, green)
    assert bare.tasks[0].met and "passes in the sandbox" in bare.tasks[0].override_reason
    for ran, ok in ((False, None), (True, None)):  # no tests ran: nothing is green, QA stands
        qn = QA(tasks=[TaskResult(id="T1", met=False, evidence="x")])
        defer_to_code(qn, green.model_copy(update={"tests_ran": ran, "tests_ok": ok}))
        assert not qn.tasks[0].met
    red = green.model_copy(update={"tests_ok": False})
    qb = QA(tasks=[TaskResult(id="T1", met=False, evidence="x")])
    defer_to_code(qb, red)
    assert not qb.tasks[0].met  # failing checks: QA's word is not second-guessed


def test_a_resumed_build_starts_at_the_critic_on_the_earlier_builds_files(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    llm.providers["arc"] = Scripted(*PRODUCE, review(True), qa(("T1", True), ("T2", True)))
    assert build(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {}, redesign=False).status == "delivered"
    files = sorted(p.name for p in (project.root / "output").rglob("*") if p.is_file())
    llm.providers["arc"] = Scripted(review(True), qa(("T1", True), ("T2", True)))  # no design, orders or builds
    again = build(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {}, redesign=False, resume=True)
    assert again.status == "delivered" and again.rounds == 1
    assert sorted(p.name for p in (project.root / "output").rglob("*") if p.is_file()) == files
    assert not (project.root / "superseded").exists()  # the files were kept, not moved aside
    assert project.board.latest("note").data["resumed"] == len(files) or any(
        e.data.get("resumed") for e in project.board.entries("note"))
    assert not llm.providers["arc"].texts  # every scripted reply was used: the critic and QA ran, nothing else


class CutFirstBuild(Scripted):
    """Cuts off the first builder reply (picked by its prompt, not its position) after one finished file."""

    def complete(self, req):
        first_build = "=== FILE:" in req.messages[-1]["content"] and "cut off" not in req.messages[-1]["content"]
        if first_build and not getattr(self, "cut", False):
            self.cut = True
            self.requests.append(req)
            return RawResult(f"=== FILE: output/primer.py ===\n{CODE}=== END FILE ===\n=== FILE: util.py ===\nx = (",
                             "", "max_tokens", Usage(500, 200))
        return super().complete(req)


def salvage_notes(project):
    return [e.data for e in project.board.entries("note") if "salvaged" in e.data]


def test_a_cut_off_build_keeps_its_finished_files_and_says_so_on_the_board(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    llm.providers["arc"] = CutFirstBuild(design(), orders("T1", "primer.py", "util.py"),
                                         "=== FILE: util.py ===\nX = 1\n=== END FILE ===\nNOTES: rest",
                                         orders("T2", "tests/test_primer.py"), built(("tests/test_primer.py", TEST)))
    Production(llm, Registry.load(cfg, ROOT / "registry"), project, wiki).run()
    out = project.root / "output"
    assert llm.providers["arc"].cut  # the double really cut a builder reply
    assert (out / "primer.py").read_text() == CODE and (out / "util.py").read_text() == "X = 1\n"
    assert salvage_notes(project) == [{"order": "T1.1", "salvaged": ["primer.py"],
                                       "text": "cut-off builder reply: finished files kept and written"}]


def test_kept_files_are_not_reported_as_written_when_the_retry_fails(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path, approver=deny_all)  # no paid rescue
    llm.providers["arc"] = CutFirstBuild(design(), orders("T1", "primer.py", "util.py"), "no blocks", "still none",
                                         orders("T2", "tests/test_primer.py"), built(("tests/test_primer.py", TEST)))
    Production(llm, Registry.load(cfg, ROOT / "registry"), project, wiki).run()
    assert llm.providers["arc"].cut
    assert not (project.root / "output" / "primer.py").exists() and salvage_notes(project) == []
    failed = project.board.entries("build")[0].data
    assert failed["written"] == [] and failed["failed"] and failed["salvage_lost"] == ["primer.py"]


def test_a_kept_file_the_retry_writes_again_is_not_counted_as_saved(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    again = "=== FILE: primer.py ===\nX = 2\n=== END FILE ===\n=== FILE: util.py ===\nX = 1\n=== END FILE ==="
    llm.providers["arc"] = CutFirstBuild(design(), orders("T1", "primer.py", "util.py"), again,
                                         orders("T2", "tests/test_primer.py"), built(("tests/test_primer.py", TEST)))
    Production(llm, Registry.load(cfg, ROOT / "registry"), project, wiki).run()
    assert llm.providers["arc"].cut and salvage_notes(project) == []
    assert (project.root / "output" / "primer.py").read_text() == "X = 2\n"


def test_a_paid_rescue_that_rewrites_kept_files_is_not_counted_as_salvage(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)  # approve_all: the rescue is approved
    again = "=== FILE: primer.py ===\nX = 2\n=== END FILE ===\n=== FILE: util.py ===\nX = 1\n=== END FILE ==="
    llm.providers["arc"] = CutFirstBuild(design(), orders("T1", "primer.py", "util.py"), "no blocks", "still none",
                                         orders("T2", "tests/test_primer.py"), built(("tests/test_primer.py", TEST)))
    llm.providers["openai"] = rescue = Scripted(again)
    Production(llm, Registry.load(cfg, ROOT / "registry"), project, wiki).run()
    assert llm.providers["arc"].cut and len(rescue.requests) == 1  # the paid step ran, through the gate
    assert salvage_notes(project) == [] and (project.root / "output" / "primer.py").read_text() == "X = 2\n"


def test_a_paid_rescue_replaces_the_kept_files_instead_of_merging_them(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)  # approve_all on a scripted provider: no real spend
    llm.providers["arc"] = CutFirstBuild(design(), orders("T1", "primer.py", "util.py"), "no blocks", "still none",
                                         orders("T2", "tests/test_primer.py"), built(("tests/test_primer.py", TEST)))
    llm.providers["openai"] = Scripted("=== FILE: util.py ===\nX = 1\n=== END FILE ===")
    Production(llm, Registry.load(cfg, ROOT / "registry"), project, wiki).run()
    # the rescue never saw the kept primer.py, so its reply stands alone: nothing kept is written or counted
    assert llm.providers["arc"].cut and not (project.root / "output" / "primer.py").exists()
    assert salvage_notes(project) == []


def test_salvage_merge_leaves_a_malformed_reply_for_validation_to_reject():
    from factory.pipeline.production import FileSalvage
    sv = FileSalvage()
    sv.keep("=== FILE: output/a.py ===\nA = 1\n=== END FILE ===\n")
    assert list(sv.files) == ["a.py"]
    for bad in ([1, 2], {"files": "x"}, {"other": 1}):
        assert sv.merge(bad) == bad  # no exception: validation fails and the retry loop carries on
    assert sv.merge({"files": [{"path": "b.py"}, {"path": "a.py", "content": "A = 2\n"}]})["files"] == [
        {"path": "a.py", "content": "A = 2\n"}] and sv.kept() == []


def test_a_build_that_is_not_cut_off_leaves_no_salvage_note(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)
    llm.providers["arc"] = Scripted(*PRODUCE)
    Production(llm, Registry.load(cfg, ROOT / "registry"), project, wiki).run()
    assert salvage_notes(project) == [] and all(not e.data.get("salvage_lost") for e in project.board.entries("build"))


def test_final_qa_prompt_tells_qa_to_name_a_failing_check():
    from factory.pipeline.final import QA_PROMPT
    assert "ONLY by naming a concrete failing check" in QA_PROMPT and '"failing_check"' in QA_PROMPT


def test_qa_is_not_overridden_when_no_test_passed():
    from factory.pipeline.final import QA, TaskResult, defer_to_code
    from factory.pipeline.production import Checks
    empty = Checks(compiled=["a.py"], syntax_errors={}, tests_ran=True, tests_ok=True, passed=[])  # zero collected
    q = QA(tasks=[TaskResult(id="T1", met=False, evidence="the feature is absent")])
    defer_to_code(q, empty)
    assert not q.tasks[0].met
