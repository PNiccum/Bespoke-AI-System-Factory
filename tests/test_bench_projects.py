"""The project benchmark: every project's plan is sound, its acceptance tests pass on a known-good solution and fail
without one, and the harness records how far a run got."""

import json
import shutil
from types import SimpleNamespace

import pytest

import factory.pipeline.knowledge as knowledge_mod
import factory.pipeline.planning as planning_mod
import factory.pipeline.run as run_mod
from factory.bench_projects import PROJECTS, accept, load_projects, run_project, seed, summarize
from factory.pipeline.blackboard import Project
from factory.pipeline.interview import ProjectPlan, check_plan
from tests.test_orchestrator import ROOT, Scripted, cfg  # noqa: F401

TASKS = load_projects(ROOT / "bench" / "projects.yaml")


def test_four_projects_with_sound_plans_and_a_fixed_interface():
    assert [t.name for t in TASKS] == ["wordstats", "csv_profile", "log_spikes", "openalex_digest"]
    for t in TASKS:
        assert check_plan(t.plan) == []
        assert "standard library only" in t.request  # the common part is appended
        assert (PROJECTS / t.name / "acceptance" / "conftest.py").exists()


@pytest.mark.parametrize("name", [t.name for t in TASKS])
def test_acceptance_passes_on_the_reference_and_fails_without_it(name, tmp_path):
    a = accept(PROJECTS / name / "reference", name)
    assert a.accepted and a.passed == a.expected >= 5, a
    assert accept(tmp_path / "missing", name).why == "no output"
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "README.md").write_text("nothing here\n")
    wrong = accept(tmp_path / "out", name)
    assert not wrong.accepted and wrong.passed == 0 and wrong.failing


PROTEOMICS = load_projects(ROOT / "bench" / "projects_proteomics.yaml")


def test_the_proteomics_tier_is_separate_with_sound_plans():
    assert [t.name for t in PROTEOMICS] == ["protein_qc", "evidence_table", "david_filter", "target_flags",
                                            "claim_audit"]
    assert not {t.name for t in PROTEOMICS} & {t.name for t in TASKS}
    for t in PROTEOMICS:
        assert check_plan(t.plan) == [] and "standard library only" in t.request
        assert (PROJECTS / t.name / "acceptance" / "conftest.py").exists()


@pytest.mark.parametrize("name", [t.name for t in PROTEOMICS])
def test_proteomics_acceptance_passes_on_the_reference_and_fails_without_it(name, tmp_path):
    a = accept(PROJECTS / name / "reference", name)
    assert a.accepted and a.passed == a.expected >= 5, a
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "README.md").write_text("nothing here\n")
    wrong = accept(tmp_path / "out", name)
    assert not wrong.accepted and wrong.passed == 0 and wrong.failing


def test_seed_leaves_what_an_approved_interview_leaves(tmp_path):
    project = Project.create(tmp_path, "bench wordstats")
    seed(project, TASKS[0])
    approval = project.board.latest("approval")
    plan = ProjectPlan.model_validate(project.board.get(approval.data["plan"]).data["plan"])
    assert plan == TASKS[0].plan
    assert json.loads((project.root / "plan_a.json").read_text()) == plan.model_dump()
    assert "top_words(text, n=10, stopwords=None)" in (project.root / "task.txt").read_text()


def fake_pipeline(monkeypatch, *, problems=(), knowledge_ok=True, status="delivered", files_from=None):
    class Planning:
        def __init__(self, *a):
            pass

        def run(self):
            return SimpleNamespace(problems=list(problems))

    class Knowledge:
        def __init__(self, *a, **kw):
            pass

        def run(self):
            return SimpleNamespace(ok=knowledge_ok, missing=[] if knowledge_ok else ["primer3 settings"])

    def build(llm, registry, project, wiki, searchers, on_stage, redesign, resume=False):
        assert redesign is False  # a benchmark run never changes ASPIRE
        on_stage("production, round 1")
        if files_from:
            shutil.copytree(files_from, project.root / "output")
        on_stage("final QA")
        final = SimpleNamespace(issues=[] if status == "delivered" else ["T2 not met"])
        return SimpleNamespace(status=status, final=final)

    monkeypatch.setattr(planning_mod, "PlanningLoop", Planning)
    monkeypatch.setattr(knowledge_mod, "KnowledgeLoop", Knowledge)
    monkeypatch.setattr(knowledge_mod, "default_searchers", lambda cache: {})
    monkeypatch.setattr(knowledge_mod, "default_scout", lambda llm, registry, cache: None)
    monkeypatch.setattr(run_mod, "build", build)


def go(cfg, task):
    providers = {"arc": Scripted(), "openai": Scripted(), "claude": Scripted()}
    return run_project(cfg, task, providers=providers, registry_dir=ROOT / "registry")


def test_delivered_and_accepted_is_solved(cfg, monkeypatch):
    fake_pipeline(monkeypatch, files_from=PROJECTS / "wordstats" / "reference")
    o = go(cfg, TASKS[0])
    assert o.solved and o.delivered and o.reached == "qa" and o.acceptance.accepted
    assert "1/1" in summarize([o])


def test_qa_rejected_is_judged_but_not_solved(cfg, monkeypatch):
    fake_pipeline(monkeypatch, status="stopped", files_from=PROJECTS / "wordstats" / "reference")
    o = go(cfg, TASKS[0])
    assert not o.solved and o.acceptance.accepted and "T2 not met" in o.why


def test_stops_in_research_without_judging(cfg, monkeypatch):
    fake_pipeline(monkeypatch, knowledge_ok=False)
    o = go(cfg, TASKS[0])
    assert o.reached == "research" and o.acceptance is None and "primer3 settings" in o.why


def test_plan_problems_stop_the_run(cfg, monkeypatch):
    fake_pipeline(monkeypatch, problems=["T3 depends on T9"])
    o = go(cfg, TASKS[0])
    assert o.reached == "planning" and not o.solved and "T9" in o.why


def test_a_crash_is_a_failed_project(cfg, monkeypatch):
    fake_pipeline(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("builder reply cut off")

    monkeypatch.setattr(run_mod, "build", boom)
    o = go(cfg, TASKS[0])
    assert not o.solved and o.reached == "build" and "cut off" in o.why and o.acceptance.why == "no output"


def test_results_are_saved_before_a_console_that_cannot_print_them(cfg, monkeypatch, tmp_path):
    """A Windows cp1252 console (or a redirected file) and a model's U+2011 in the reason crashed the first live run
    after the project finished, losing its result. Now the result is on disk first and the print replaces it."""
    import io
    import sys

    import factory.bench_projects as bp

    o = bp.ProjectOutcome(project="wordstats", reached="research", why="non‑breaking hyphen")
    monkeypatch.setattr(bp, "run_project", lambda *a, **k: o)
    monkeypatch.setattr(bp, "load_config", lambda path: cfg)
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", console)
    assert bp.main(["--only", "wordstats", "--out", str(tmp_path)]) == 0
    run = next(tmp_path.iterdir())
    assert json.loads((run / "results.json").read_text(encoding="utf-8"))[0]["why"] == "non‑breaking hyphen"
    assert "wordstats" in (run / "summary.md").read_text(encoding="utf-8")
    console.flush()
    assert b"non?breaking" in console.buffer.getvalue()


def test_resume_keeps_the_finished_stages_and_goes_on_from_the_first_unfinished_one(cfg, monkeypatch):
    from factory.bench_projects import latest_project
    fake_pipeline(monkeypatch, files_from=PROJECTS / "wordstats" / "reference")
    monkeypatch.setattr(planning_mod.PlanningLoop, "run", lambda self: pytest.fail("planning ran again"))
    monkeypatch.setattr(knowledge_mod.KnowledgeLoop, "run", lambda self: pytest.fail("research ran again"))
    earlier = Project.create(cfg.work_dir / "projects", "bench wordstats")  # what a run that died at build left
    seed(earlier, TASKS[0])
    earlier.board.post("decision", "code", {"winner": "A", "votes": {}, "problems": []})
    earlier.board.post("knowledge", "code", {"ok": True, "missing": []})
    assert latest_project(cfg, TASKS[0]) == earlier.root
    providers = {"arc": Scripted(), "openai": Scripted(), "claude": Scripted()}
    o = run_project(cfg, TASKS[0], providers=providers, registry_dir=ROOT / "registry", resume=earlier.root)
    assert o.resumed == ["planning", "research"] and o.project_dir == str(earlier.root)
    assert o.solved and o.reached == "qa"
    assert "(resumed after planning, research)" in summarize([o])
    with pytest.raises(ValueError, match="no approved plan"):
        run_project(cfg, TASKS[0], providers=providers, registry_dir=ROOT / "registry",
                    resume=Project.create(cfg.work_dir / "projects", "bench wordstats").root)
