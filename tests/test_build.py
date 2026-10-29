"""Production, the critic loop, final QA and re-design: models write and judge, code writes files and decides."""

import json
import zipfile

import pytest

from factory.llm.gate import approve_all, deny_all
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


def qa(*met):
    return json.dumps({"tasks": [{"id": t, "met": m, "evidence": "test_tm"} for t, m in met], "issues": []})


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
    audit = json.dumps({"successes": ["tests"], "failures": ["T2 unmet"], "causes": ["no Primer3"],
                        "proposals": ["add a Primer3 tool"]})
    llm.providers["arc"] = Scripted(*PRODUCE, review(True), qa(("T1", True), ("T2", False)),
                                    *PRODUCE, review(True), qa(("T1", True), ("T2", False)),
                                    audit, json.dumps({"suggestions": ["wrap primer3-py"]}),
                                    json.dumps({"problems": ["no design tool"], "fixes": ["add Primer3 tool"]}),
                                    "The project asked for primers; T2 was never met.")
    result = build(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {})
    assert result.status == "redesign" and result.rounds == 2
    assert result.redesign.stopped.startswith("paid call not made")
    report = (project.root / "redesign" / "report.md").read_text()
    assert "No change was applied" in report and "add Primer3 tool" in report
    assert "acceptance not met: T2" in project.board.entries("final")[0].data["issues"][0]


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


def redesign_with(cfg, tmp_path, prompt_changes, debate_last=None):
    """Re-design with scripted auditors and a recording improver in place of factory.extend."""
    llm, project, wiki = project_after_knowledge(cfg, tmp_path, approver=deny_all)
    llm.providers["arc"] = Scripted(json.dumps({"failures": ["T2 unmet"]}), json.dumps({"suggestions": []}),
                                    json.dumps({"problems": ["primers unchecked"], "fixes": ["add Primer3 tool"],
                                                "prompt_changes": prompt_changes}), "summary")
    tried = []

    def improver(agent, change):
        tried.append(agent)
        return "adopted", "scope: ok"
    result = Redesign(llm, Registry.load(cfg, ROOT / "registry"), project, improver=improver).run()
    return result, tried, (project.root / "redesign" / "report.md").read_text()


def test_prompt_changes_are_applied_and_structural_fixes_wait(cfg, tmp_path):
    result, tried, report = redesign_with(cfg, tmp_path, [
        {"agent": "builder", "change": "check primer GC content"},
        {"agent": "final_evaluator", "change": "be less strict"},  # a judge: structural
        {"agent": "test_runner", "change": "run faster"},  # a tool agent: structural
        {"agent": "nobody", "change": "x"}])
    assert tried == ["builder"]
    status = {a.agent: a.status for a in result.prompt_changes}
    assert status == {"builder": "adopted", "final_evaluator": "awaiting a person"}
    assert "test_runner: run faster" in result.fixes and "add Primer3 tool" in result.fixes
    assert "1 prompt change(s) passed every gate and were applied" in report and "--rollback" in report


def test_prompt_changes_stop_at_the_limit(cfg, tmp_path):
    result, tried, _ = redesign_with(cfg, tmp_path, [{"agent": a, "change": "x"} for a in
                                                      ("builder", "librarian", "summarizer")])
    assert tried == ["builder", "librarian"]
    assert result.prompt_changes[-1].status.startswith("not tried")
