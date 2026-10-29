"""The blackboard and the Interviewer: questions are capped, plans are checked by code, only the person approves."""

import json

import pytest

from factory.llm import LLM
from factory.llm.gate import approve_all
from factory.pipeline.blackboard import Blackboard, Project
from factory.pipeline.interview import MAX_QUESTIONS, Interview, ProjectPlan, check_plan, render_task
from factory.registry import Registry
from tests.test_orchestrator import ROOT, Scripted, cfg  # noqa: F401

PLAN = {"title": "Primer tool", "goal": "Design PCR primers for a gene", "deliverable": "a Python script",
        "tasks": [{"id": "T1", "title": "Fetch the gene from NCBI", "acceptance": "the FASTA file is saved"},
                  {"id": "T2", "title": "Design primers", "acceptance": "primers meet the Tm range",
                   "depends_on": ["T1"]}],
        "assumptions": ["human genes only"]}


def ask(*questions):
    return json.dumps({"action": "ask", "questions": list(questions), "note": "A few questions first."})


def propose(plan=PLAN):
    return json.dumps({"action": "propose", "plan": plan, "note": "Here is Plan A."})


def interview(cfg, tmp_path, *replies):
    arc = Scripted(*replies)
    llm = LLM(cfg, tmp_path, use_cache=False, providers={"arc": arc, "openai": Scripted(),
                                                                  "claude": Scripted()}, approver=approve_all)
    project = Project.create(tmp_path / "projects", "Design PCR primers")
    return Interview(llm, Registry.load(cfg, ROOT / "registry"), project), arc, project


def test_blackboard_entries_are_typed_numbered_and_licensed(tmp_path):
    board = Blackboard(tmp_path / "b.jsonl")
    a = board.post("request", "person", {"text": "x"})
    b = board.post("note", "librarian", {"text": "from a paper"}, refs=[a.id], license="arc_only")
    assert (a.id, b.id) == ("0001-request", "0002-note")
    assert board.get(b.id).refs == [a.id]
    assert board.latest("note") == b
    assert Blackboard.for_paid(board.entries()) == [a]
    with pytest.raises(ValueError, match="unknown"):
        board.post("vote", "x", {})


def test_project_files_stay_inside_the_folder_and_are_recorded(tmp_path):
    project = Project.create(tmp_path, "My project!")
    assert project.id.endswith("-my-project")
    project.write_file("out/a.txt", "hi", "coder")
    assert project.board.latest("file").data["path"] == "out/a.txt"
    with pytest.raises(ValueError, match="outside"):
        project.write_file("../escape.txt", "no", "coder")
    assert Project.create(tmp_path, "My project!").id != project.id  # same second, same title


def test_plan_checks_find_missing_acceptance_duplicates_and_forward_dependencies():
    plan = ProjectPlan.model_validate({**PLAN, "tasks": [
        {"id": "T1", "title": "a", "acceptance": "", "depends_on": ["T2"]},
        {"id": "T2", "title": "b", "acceptance": "ok"}, {"id": "T2", "title": "c", "acceptance": "ok"}]})
    problems = check_plan(plan)
    assert "T1 has no acceptance check" in problems
    assert any("T1 depends on T2" in p for p in problems)
    assert "task id T2 is used twice" in problems
    assert check_plan(ProjectPlan.model_validate(PLAN)) == []


def test_questions_then_plan_then_approval_writes_task_txt(cfg, tmp_path):
    iv, arc, project = interview(cfg, tmp_path, ask("Which organism?", "Which gene?"), propose())
    turn = iv.start("Make me a tool that designs PCR primers")
    assert turn.kind == "question" and len(turn.data["questions"]) == 2
    turn = iv.reply("Human, BRCA1")
    assert turn.kind == "plan" and turn.data["problems"] == []
    assert "Human, BRCA1" in arc.requests[1].messages[-1]["content"]  # the model sees the answers
    path = iv.approve()
    text = path.read_text()
    assert "Design primers (after T1)" in text and "Human, BRCA1" in text
    assert json.loads((project.root / "plan_a.json").read_text())["title"] == "Primer tool"
    kinds = [e.kind for e in project.board.entries()]
    assert kinds == ["request", "question", "answer", "plan", "file", "file", "approval"]
    with pytest.raises(RuntimeError, match="already approved"):
        iv.reply("one more thing")


def test_question_budget_is_enforced_in_code(cfg, tmp_path):
    many = ask("q1", "q2", "q3", "q4")
    iv, arc, _ = interview(cfg, tmp_path, many, many, ask("q7"), propose())
    assert len(iv.start("something").data["questions"]) == 3  # three per turn at most
    assert len(iv.reply("a").data["questions"]) == 3
    assert iv.questions_asked == MAX_QUESTIONS
    turn = iv.reply("b")  # out of questions: the model asks anyway, is told to propose, and does
    assert turn.kind == "plan"
    assert "must propose" in arc.requests[-1].messages[-1]["content"]


def test_a_plan_with_problems_goes_back_once_and_cannot_be_approved(cfg, tmp_path):
    bad = {**PLAN, "tasks": [{"id": "T1", "title": "a", "acceptance": ""}]}
    iv, arc, _ = interview(cfg, tmp_path, propose(bad), propose(bad))
    turn = iv.start("something")
    assert "has these problems" in arc.requests[1].messages[-1]["content"]
    assert turn.data["problems"] == ["T1 has no acceptance check"]
    with pytest.raises(RuntimeError, match="still has problems"):
        iv.approve()


def test_changes_to_a_plan_get_a_new_plan_not_more_questions(cfg, tmp_path):
    iv, arc, _ = interview(cfg, tmp_path, propose(), ask("why?"), propose({**PLAN, "title": "Primer tool v2"}))
    iv.start("something")
    turn = iv.reply("Use mouse genes instead")
    assert turn.kind == "plan" and turn.data["plan"]["title"] == "Primer tool v2"


def test_render_task_lists_assumptions():
    text = render_task("req", [], ProjectPlan.model_validate(PLAN))
    assert "## Assumptions\n- human genes only" in text
