"""The planning loop: three families critique, weigh, edit and vote; code checks plans and counts the votes."""

import json
import threading

import pytest

from factory.llm import LLM, Usage
from factory.llm.gate import approve_all
from factory.llm.providers import RawResult
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


def recheck(*open_ids, blocks=True):
    return json.dumps({"open": [{"id": i, "why": "still no subtasks", "blocks": blocks} for i in open_ids]})


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


class ByPlan(Scripted):
    """Scripted, except that a group of replies for calls that run side by side on two threads is answered by who is
    asking instead of by arrival order: ``(reply_for_a, reply_for_b)`` for the two re-checks (by the plan the prompt
    asks about) and ``{"plan_critic": reply, "plan_analyst": reply}`` for the two votes (by the request's tag)."""

    lock = threading.Lock()

    def complete(self, req):
        with self.lock:
            group = self.texts[0]
            if not isinstance(group, (tuple, dict)):
                return super().complete(req)
            if isinstance(group, tuple):
                key = 1 if "It is Plan B" in req.messages[-1]["content"] else 0
            else:
                key = req.tag.removeprefix("plan-vote-")
            reply = group[key]
            rest = {k: v for k, v in (group.items() if isinstance(group, dict) else enumerate(group)) if k != key}
            if rest:
                self.texts[0] = rest if isinstance(group, dict) else tuple(rest.get(i) for i in range(2))
                if isinstance(group, tuple) and all(x is None for x in self.texts[0]):
                    self.texts.pop(0)
            else:
                self.texts.pop(0)
            self.requests.append(req)
            return RawResult(reply, "", "end_turn", Usage(500, 200))


def loop(cfg, tmp_path, *replies):
    llm, project = approved_project(cfg, tmp_path)
    arc = ByPlan(*replies)
    llm.providers["arc"] = arc
    return PlanningLoop(llm, Registry.load(cfg, ROOT / "registry"), project), arc, project


def test_three_families_critique_weigh_edit_and_vote(cfg, tmp_path):
    pl, arc, project = loop(cfg, tmp_path, critique(), ANALYSIS, edit(), recheck(), recheck(), vote("B"), vote("B"))
    out = pl.run()
    assert out.winner == "B" and out.votes == {"plan_critic": "B", "plan_analyst": "B"}  # the editor never votes
    assert json.loads((project.root / "plan.json").read_text())["tasks"][0]["subtasks"][0]["id"] == "T1.1"
    families = {cfg.model_info(r.model).family for r in arc.requests[:3]}
    assert families == {"deepseek", "openai", "glm"}  # critic, analyst, editor
    kinds = [e.kind for e in project.board.entries()][-9:]
    assert kinds == ["critique", "analysis", "edit", "recheck", "recheck", "vote", "vote", "decision", "file"]
    with pytest.raises(RuntimeError, match="already ran"):
        pl.run()


def test_an_edit_that_breaks_a_plan_is_dropped(cfg, tmp_path):
    pl, _, project = loop(cfg, tmp_path, critique(), ANALYSIS, edit(b=BROKEN), recheck(), recheck(), vote("B"), vote("B"))
    out = pl.run()
    assert out.winner == "B" and out.plan.title == PLAN_B["title"]
    assert project.board.latest("edit").data["dropped"] == ["B"]


def test_a_plan_with_problems_cannot_win(cfg, tmp_path):
    # Plan B stays broken after its one fix, so votes for it are spoiled and A wins.
    pl, _, _ = loop(cfg, tmp_path, critique(BROKEN), critique(BROKEN), ANALYSIS, edit(b=BROKEN), recheck(), recheck(),
                    {"plan_critic": vote("B"), "plan_analyst": vote("A")})
    out = pl.run()
    assert out.winner == "A" and out.votes["plan_critic"] == "spoiled" and out.problems == []


def test_a_split_vote_goes_to_plan_a(cfg, tmp_path):
    pl, _, _ = loop(cfg, tmp_path, critique(), ANALYSIS, edit(), recheck(), recheck(),
                    {"plan_critic": vote("B"), "plan_analyst": vote("A")})
    assert pl.run().winner == "A"


def test_the_loop_needs_an_approved_plan(cfg, tmp_path):
    project = Project.create(tmp_path / "projects", "x")
    llm = LLM(cfg, tmp_path, use_cache=False, providers={"arc": Scripted(), "openai": Scripted(),
                                                          "claude": Scripted()}, approver=approve_all)
    with pytest.raises(RuntimeError, match="approved Plan A"):
        PlanningLoop(llm, Registry.load(cfg, ROOT / "registry"), project).run()


@pytest.mark.parametrize("votes,problems,expected", [
    ({"x": "B", "y": "B"}, {"A": [], "B": []}, "B"),
    ({"x": "B", "y": "spoiled"}, {"A": [], "B": []}, "B"),
    ({"x": "A", "y": "B"}, {"A": [], "B": []}, "A"),  # split: the person's plan
    ({"x": "A", "y": "B"}, {"A": ["p"], "B": []}, "B"),  # split: fewer problems
])
def test_votes_are_counted_in_code(votes, problems, expected):
    assert decide(votes, problems) == expected


def test_an_open_weakness_keeps_plan_a_from_winning_until_repaired(cfg, tmp_path):
    fixed = {**PLAN, "title": "Primer tool with subtasks"}
    pl, arc, project = loop(cfg, tmp_path, critique(), ANALYSIS, edit(), (recheck("W1"), recheck()), vote("A"),
                            vote("A"), json.dumps(fixed), recheck())
    out = pl.run()
    assert out.winner == "B"  # both votes for A were spoiled by its open weakness
    assert "W1: no subtasks" in arc.requests[2].messages[-1]["content"]  # the editor saw the blocker


def test_plan_a_with_its_blocker_fixed_can_win(cfg, tmp_path):
    pl, _, project = loop(cfg, tmp_path, critique(), ANALYSIS, edit(), recheck(), recheck(), vote("A"), vote("A"))
    assert pl.run().winner == "A" and project.board.latest("recheck").data["problems"] == []


def test_plan_b_with_the_same_open_weakness_does_not_win_by_default(cfg, tmp_path):
    pl, arc, project = loop(cfg, tmp_path, critique(), ANALYSIS, edit(), recheck("W1"), recheck("W1"), vote("A"),
                            vote("A"), json.dumps(PLAN), recheck())
    out = pl.run()
    assert out.votes == {"plan_critic": "spoiled", "plan_analyst": "spoiled"}
    assert out.winner == "A" and out.problems == []  # a tie on problems goes to A, whose repair then closes W1
    b = [e.data for e in project.board.entries("recheck") if e.data["plan"] == "B"]
    assert b and b[0]["problems"][0].startswith("W1 still open")
    assert "It is Plan B, the alternative" in arc.requests[4].messages[0]["content"]


def test_a_repair_that_leaves_the_blocker_open_is_reported(cfg, tmp_path):
    pl, _, _ = loop(cfg, tmp_path, critique(BROKEN), critique(BROKEN), ANALYSIS, edit(b=BROKEN),
                    (recheck("W1"), recheck()), vote("A"), vote("A"), json.dumps(PLAN), recheck("W1"))
    out = pl.run()
    assert out.winner == "A" and out.problems and out.problems[0].startswith("W1 still open")


def test_restart_copies_the_project_up_to_its_approval(cfg, tmp_path):
    pl, _, project = loop(cfg, tmp_path, critique(), ANALYSIS, edit(), recheck(), recheck(), vote("B"), vote("B"))
    pl.run()
    new = Project.restart(project, tmp_path / "projects")
    assert new.root != project.root and new.board.latest("approval") is not None
    assert new.board.latest("decision") is None and not (new.root / "plan.json").exists()
    assert (new.root / "task.txt").read_text() == (project.root / "task.txt").read_text()
    llm = pl.llm
    llm.providers["arc"] = Scripted(critique(), ANALYSIS, edit(), recheck(), recheck(), vote("A"), vote("A"))
    assert PlanningLoop(llm, Registry.load(cfg, ROOT / "registry"), new).run().winner == "A"


def test_an_open_weakness_that_does_not_change_the_result_is_noted_and_the_plan_goes_ahead(cfg, tmp_path):
    # csv_profile, 2026-11-05: "T2 depends on T1 unnecessarily" kept a plan from a build that had reached QA 5/5
    pl, _, project = loop(cfg, tmp_path, critique(), ANALYSIS, edit(), (recheck("W1", blocks=False), recheck()),
                          vote("A"), vote("A"))
    out = pl.run()
    assert out.winner == "A" and out.problems == [] and out.votes == {"plan_critic": "A", "plan_analyst": "A"}
    a = [e.data for e in project.board.entries("recheck") if e.data["plan"] == "A"][0]
    assert a["problems"] == [] and a["noted"] == ["W1 noted: no subtasks (still no subtasks)"]
    assert a["open"][0]["blocks"] is False


def test_a_recheck_reply_without_the_blocks_field_still_blocks(cfg, tmp_path):
    old = json.dumps({"open": [{"id": "W1", "why": "still no subtasks"}]})  # a model that ignored the new field
    pl, _, _ = loop(cfg, tmp_path, critique(), ANALYSIS, edit(), old, old, vote("A"), vote("A"),
                    json.dumps(PLAN), old)
    out = pl.run()
    assert out.problems and out.problems[0].startswith("W1 still open")


def test_the_voters_are_told_each_plans_blocking_problems_and_noted_count(cfg, tmp_path):
    # MCF7 run, 2026-11-05: both votes went to A without knowing A had blocking problems, so both were spoiled
    pl, arc, _ = loop(cfg, tmp_path, critique(), ANALYSIS, edit(), (recheck("W1"), recheck()), vote("B"), vote("B"))
    pl.run()
    prompts = [r.messages[-1]["content"] for r in arc.requests if "Problems the code found" in r.messages[-1]["content"]]
    assert len(prompts) == 2 and prompts[0] == prompts[1]  # both voters get the same text
    prompt = prompts[0]
    assert "Plan A: blocking problems: W1 still open" in prompt
    assert "Plan B: blocking problems: none. Noted weaknesses (not blocking): 0" in prompt


def test_noted_weaknesses_carried_into_the_build_are_summarised_on_the_board(cfg, tmp_path):
    pl, _, project = loop(cfg, tmp_path, critique(), ANALYSIS, edit(), (recheck("W1", blocks=False), recheck()),
                          vote("A"), vote("A"))
    out = pl.run()
    assert out.noted == ["W1 noted: no subtasks (still no subtasks)"]
    note = project.board.latest("note")
    assert note.data["text"].startswith("Plan A goes ahead with 1 open noted weakness")
    assert project.board.latest("decision").data["noted_count"] == 1


def test_a_long_noted_list_is_cut_in_the_summary_but_kept_whole(cfg, tmp_path):
    ws = [f"weak{i}" for i in range(1, 9)]
    many = json.dumps({"weaknesses": ws, "plan_b": PLAN_B})
    pl, _, project = loop(cfg, tmp_path, many, ANALYSIS, edit(),
                          (recheck(*(f"W{i}" for i in range(1, 9)), blocks=False), recheck()), vote("A"), vote("A"))
    out = pl.run()
    note = project.board.latest("note")
    assert len(out.noted) == 8 and note.data["noted"] == out.noted
    assert "and 3 more" in note.data["text"] and "W6 noted" not in note.data["text"]


def test_noted_weaknesses_survive_a_repair_round_and_add_what_the_repair_left_noted(cfg, tmp_path):
    two = json.dumps({"weaknesses": ["no subtasks", "extra step"], "plan_b": PLAN_B})
    a = json.dumps({"open": [{"id": "W1", "why": "x", "blocks": True}, {"id": "W2", "why": "y", "blocks": False}]})
    repaired = json.dumps({"open": [{"id": "W1", "why": "z", "blocks": False}]})
    pl, _, _ = loop(cfg, tmp_path, two, ANALYSIS, edit(), (a, a), vote("A"), vote("A"),
                    json.dumps({**PLAN, "title": "repaired"}), repaired)
    out = pl.run()
    assert out.winner == "A" and out.problems == []
    assert sorted(n.split(" noted")[0] for n in out.noted) == ["W1", "W2"]
