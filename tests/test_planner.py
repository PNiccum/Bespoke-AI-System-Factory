"""Intake: rules route the clear requests, the planner model only the ambiguous ones, and code checks its plan."""

import json

import pytest

from factory.llm import LLM
from factory.llm.gate import approve_all
from factory.orchestrator import Task, plan_for
from factory.planner import BLOCKS, check_steps, route, route_by_rules
from factory.registry import Registry
from factory.registry.spec import DATA_RULE
from factory.workflows import TASK_TYPES
from tests.test_orchestrator import REPO, ROOT, Scripted, cfg  # noqa: F401

BUG_FIX_STEPS = ["locate_code", "run_tests", "write_patch", "run_tests", "review_patch"]


def routed(cfg, tmp_path, request, *replies, patch=""):
    arc = Scripted(*replies)
    llm = LLM(cfg, tmp_path, use_cache=False, providers={"arc": arc, "openai": Scripted(), "claude": Scripted()},
              approver=approve_all)
    r = route(request, str(REPO), llm, Registry.load(cfg, ROOT / "registry"), patch=patch,
              gaps=tmp_path / "gaps.jsonl")
    return r, arc, llm


@pytest.mark.parametrize("request_,patch,expected", [
    ("Write tests for days_in_month in calc/dates.py", "", "test_gen"),
    ("please add unit tests covering parse_date", "", "test_gen"),
    ("Look this over", "--- a/x.py\n+++ b/x.py\n", "pr_review"),
    ("The tests fail because parse_date has a bug", "", None),
    ("parse_date gets month and day backwards", "", None),
])
def test_clear_requests_are_routed_by_rules_without_a_model(request_, patch, expected):
    assert route_by_rules(request_, patch) == expected


def test_every_workflow_is_a_valid_sequence_of_blocks():
    for task_type in TASK_TYPES:
        steps = [s.capability for s in plan_for(Task(description="", repo_path=".", task_type=task_type)).steps]
        assert check_steps(steps) == [], task_type
        assert all(BLOCKS[s].needs_llm == step.needs_llm for s, step in
                   zip(steps, plan_for(Task(description="", repo_path=".", task_type=task_type)).steps, strict=True))


def test_blocks_out_of_order_or_unknown_are_named():
    assert check_steps(["write_patch", "locate_code"]) == ["'write_patch' needs locate_code before it"]
    assert check_steps(["locate_code", "translate_text"]) == ["unknown block 'translate_text'"]


def test_a_rule_routed_request_makes_no_model_call(cfg, tmp_path):
    r, arc, llm = routed(cfg, tmp_path, "Write tests for days_in_month in calc/dates.py")
    assert r.routed_by == "rules" and r.task.task_type == "test_gen" and arc.requests == []
    assert llm.ledger.summary()["calls"] == 0


def test_an_ambiguous_request_goes_to_the_free_planner(cfg, tmp_path):
    reply = json.dumps({"task_type": "bug_fix", "steps": BUG_FIX_STEPS, "missing": [],
                        "failing_tests": ["tests/test_dates.py::test_parse_date_order", "not a node id"],
                        "why": "describes wrong behaviour"})
    r, arc, llm = routed(cfg, tmp_path, "parse_date gets month and day backwards", reply)
    assert r.routed_by == "planner" and r.task.task_type == "bug_fix" and r.problems == []
    assert r.task.failing_tests == ["tests/test_dates.py::test_parse_date_order"]
    assert arc.requests[0].model == cfg.roles["planner"].model and llm.ledger.summary()["real_usd"] == 0
    assert arc.requests[0].system.endswith(DATA_RULE)


def test_a_known_workflow_runs_its_template_whatever_the_planner_proposed(cfg, tmp_path):
    reply = json.dumps({"task_type": "bug_fix", "steps": ["write_patch"], "why": "just patch it"})
    r, _, _ = routed(cfg, tmp_path, "parse_date gets month and day backwards", reply)
    assert [s.capability for s in r.plan.steps] == BUG_FIX_STEPS
    assert "'write_patch' needs locate_code before it" in r.problems


def test_an_unsupported_request_is_not_run_and_is_recorded_as_a_gap(cfg, tmp_path):
    reply = json.dumps({"task_type": "unsupported", "steps": ["locate_code", "translate_text"],
                        "missing": ["translate_text"], "why": "no workflow translates documents"})
    r, _, _ = routed(cfg, tmp_path, "Translate the README into French", reply)
    assert r.task is None and [s.capability for s in r.plan.steps] == ["locate_code"]
    gap = json.loads((tmp_path / "gaps.jsonl").read_text())
    assert gap["request"] == "Translate the README into French" and gap["missing"] == ["translate_text"]
    assert "unknown block 'translate_text'" in gap["problems"]


def test_a_review_without_a_diff_is_not_run(cfg, tmp_path):
    reply = json.dumps({"task_type": "pr_review", "steps": ["inspect_diff", "run_tests", "review_pr"], "why": "x"})
    r, _, _ = routed(cfg, tmp_path, "Can you look over my latest change?", reply)
    assert r.task is None and any("--patch-file" in p for p in r.problems)
