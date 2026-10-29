"""The planning loop: three families critique, weigh, edit and vote; code checks plans and counts the votes."""

import json

import pytest

from factory.llm import LLM
from factory.llm.gate import approve_all
from factory.pipeline.blackboard import Project
from factory.pipeline.interview import Interview
from factory.pipeline.planning import PlanningLoop, decide
from factory.registry import Registry
from tests.test_interview import PLAN, propose
from tests.test_orchestrator import ROOT, Scripted, cfg  # noqa: F401

PLAN_B = {**PLAN, "title": "Primer tool, Primer3 based",
          "tasks": [{"id": "T1", "title": "Fetch the gene", "acceptance": "FASTA saved",
                     "subtasks": [{"id": "T1.1", "title": "Query NCBI E-utilities", "acceptance": "HTTP 200"}]},
                    {"id": "T2", "title": "Run Primer3", "acceptance": "3 primer pairs", "depends_on": ["T1"]}]}
BROKEN = {**PLAN, "tasks": [{"id": "T1", "title": "x", "acceptance": ""}]}
ANALYSIS = json.dumps({"a": {"pros": ["simple"], "cons": ["no subtasks"]}, "b": {"pros": ["uses Primer3"],
                                                                             "cons": []}})


def critique(plan_b=PLAN_B):
    return json.dumps({"weaknesses": ["no subtasks"], "plan_b": plan_b})


def edit(a=PLAN, b=PLAN_B):
    return json.dumps({"plan_a": a, "plan_b": b, "changes": ["added subtasks"]})


def vote(choice):
    return json.dumps({"choice": choice, "why": "more checkable"})


def approved_project(cfg, tmp_path):
    arc = Scripted(propose())
    llm = LLM(cfg, tmp_path, use_cache=False, providers={"arc": arc, "openai": Scripted(), "claude": Scripted()},
              approver=approve_all)
    project = Project.create(tmp_path / "projects", "primers")
    iv = Interview(llm, Registry.load(cfg, ROOT / "registry"), project)
    iv.start("Make me a primer design tool")
    iv.approve()
    return llm, project


def loop(cfg, tmp_path, *replies):
    llm, project = approved_project(cfg, tmp_path)
    arc = Scripted(*replies)
    llm.providers["arc"] = arc
    return PlanningLoop(llm, Registry.load(cfg, ROOT / "registry"), project), arc, project


def test_three_families_critique_weigh_edit_and_vote(cfg, tmp_path):
    pl, arc, project = loop(cfg, tmp_path, critique(), ANALYSIS, edit(), vote("B"), vote("B"), vote("A"))
    out = pl.run()
    assert out.winner == "B" and out.votes == {"plan_critic": "B", "plan_analyst": "B", "plan_editor": "A"}
    assert json.loads((project.root / "plan.json").read_text())["tasks"][0]["subtasks"][0]["id"] == "T1.1"
    families = {cfg.model_info(r.model).family for r in arc.requests[:3]}
    assert families == {"deepseek", "openai", "glm"}  # critic, analyst, editor
    kinds = [e.kind for e in project.board.entries()][-8:]
    assert kinds == ["critique", "analysis", "edit", "vote", "vote", "vote", "decision", "file"]
    with pytest.raises(RuntimeError, match="already ran"):
        pl.run()


def test_an_edit_that_breaks_a_plan_is_dropped(cfg, tmp_path):
    pl, _, project = loop(cfg, tmp_path, critique(), ANALYSIS, edit(b=BROKEN), vote("B"), vote("B"), vote("B"))
    out = pl.run()
    assert out.winner == "B" and out.plan.title == PLAN_B["title"]
    assert project.board.latest("edit").data["dropped"] == ["B"]


def test_a_plan_with_problems_cannot_win(cfg, tmp_path):
    # Plan B stays broken after its one fix, so votes for it are spoiled and A wins.
    pl, _, _ = loop(cfg, tmp_path, critique(BROKEN), critique(BROKEN), ANALYSIS, edit(b=BROKEN),
                    vote("B"), vote("B"), vote("A"))
    out = pl.run()
    assert out.winner == "A" and out.votes["plan_critic"] == "spoiled" and out.problems == []


def test_the_loop_needs_an_approved_plan(cfg, tmp_path):
    project = Project.create(tmp_path / "projects", "x")
    llm = LLM(cfg, tmp_path, use_cache=False, providers={"arc": Scripted(), "openai": Scripted(),
                                                          "claude": Scripted()}, approver=approve_all)
    with pytest.raises(RuntimeError, match="approved Plan A"):
        PlanningLoop(llm, Registry.load(cfg, ROOT / "registry"), project).run()


@pytest.mark.parametrize("votes,problems,expected", [
    ({"x": "A", "y": "B", "z": "B"}, {"A": [], "B": []}, "B"),
    ({"x": "A", "y": "B", "z": "spoiled"}, {"A": [], "B": []}, "A"),  # tie: the person's plan
    ({"x": "A", "y": "B", "z": "spoiled"}, {"A": ["p"], "B": []}, "B"),  # tie: fewer problems
])
def test_votes_are_counted_in_code(votes, problems, expected):
    assert decide(votes, problems) == expected
