"""Precedent: delivered projects with a similar request are found by code and offered as a template, with no model."""

import zipfile

from factory.pipeline.blackboard import Project
from factory.pipeline.precedent import find, fingerprint, offer, render, similarity, stem
from tests.test_interview import PLAN, interview, propose
from tests.test_orchestrator import ROOT, Scripted, cfg  # noqa: F401

SPIKES = "Write a script that finds traffic spikes in Apache access logs, per minute."


def delivered(base, title, request, *, approved=True, arc_only=False, zip_exists=True):
    p = Project.create(base, title)
    p.board.post("request", "person", {"text": request})
    p.board.post("plan", "interviewer", {"plan": {**PLAN, "title": title}, "problems": []})
    p.board.post("design", "head", {"language": "python", "entry_point": "main.py", "run_command": "python main.py",
                                    "layout": ["main.py: reads the log", "tests/test_main.py: checks"]})
    if arc_only:
        p.board.post("note", "librarian", {"text": "lab data"}, license="arc_only")
    out = p.root / "deliverable" / f"{p.id}.zip"
    if zip_exists:
        out.parent.mkdir(parents=True)
        with zipfile.ZipFile(out, "w") as z:
            z.writestr("main.py", "print(1)")
            z.writestr("aspire/task.txt", "x")
    p.board.post("final", "code", {"approved": approved, "issues": [], "tasks": [],
                                   "deliverable": str(out) if approved else None})
    return p


def test_the_fingerprint_ignores_case_order_plurals_and_filler_words():
    a = fingerprint("Find spikes in the Apache logs")
    b = fingerprint("apache LOG spike: find them, please")
    assert a.hash == b.hash and a.keywords == ["apache", "find", "log", "spike"]
    assert stem("logs") == "log" and stem("class") == "class" and stem("queries") == "query"
    assert stem("databases") == "database" and stem("spiking") == "spik"


def test_similarity_scores_shared_content_and_needs_two_shared_words():
    mine = fingerprint("Detect spikes in Apache access logs per minute")
    close, shared = similarity(mine, fingerprint(SPIKES))
    assert close >= 0.4 and {"apache", "log", "spike"} <= set(shared)
    assert similarity(mine, fingerprint("Design PCR primers for a gene"))[0] == 0.0
    assert similarity(mine, fingerprint("Apache web server tuning guide"))[0] == 0.0  # one shared word is not a match
    assert similarity(mine, mine)[0] == 1.0


def test_find_returns_only_delivered_projects_best_first(tmp_path):
    base = tmp_path / "projects"
    near = delivered(base, "spikes", SPIKES)
    delivered(base, "primers", "Design PCR primers for a human gene")
    delivered(base, "failed", SPIKES, approved=False)
    delivered(base, "gone", SPIKES, zip_exists=False)
    exact = delivered(base, "same", "apache access log spike detection per minute, traffic")
    found = find(base, "Detect spikes in Apache access logs per minute")
    assert {p.project for p in found} == {near.id, exact.id}
    assert found[0].score >= found[1].score
    top = found[0]
    assert top.language == "python" and top.run_command == "python main.py" and top.files == ["main.py"]
    assert top.layout and top.deliverable.endswith(".zip")
    assert find(base, "Detect spikes in Apache access logs per minute", exclude=near.id) != found
    assert find(base, "") == [] and find(tmp_path / "nowhere", SPIKES) == []


def test_render_names_the_template_and_is_empty_without_matches(tmp_path):
    base = tmp_path / "projects"
    delivered(base, "spikes", SPIKES)
    text = render(find(base, "Apache log spikes per minute"))
    assert "Earlier delivered projects" in text and "main.py: reads the log" in text and "python main.py" in text
    assert render([]) == ""


def test_offer_is_recorded_once_and_arc_only_matches_stay_off_paid_prompts(tmp_path):
    base = tmp_path / "projects"
    delivered(base, "lab", SPIKES, arc_only=True)
    new = Project.create(base, "new")
    new.board.post("request", "person", {"text": "Apache log spikes per minute"})
    text = offer(new, "ignored: the board's request is used")
    entry = new.board.latest("precedent")
    assert "Earlier delivered projects" in text and entry.license == "arc_only" and entry.author == "code"
    assert entry not in new.board.for_paid(new.board.entries())
    delivered(base, "later", SPIKES)  # a later delivery does not change a recorded offer
    assert offer(new, "") == text and len(new.board.entries("precedent")) == 1


def test_no_match_is_recorded_as_an_empty_offer(tmp_path):
    new = Project.create(tmp_path / "projects", "new")
    new.board.post("request", "person", {"text": "Design PCR primers"})
    assert offer(new, "") == ""
    assert new.board.latest("precedent").data["matches"] == []


def test_the_interviewer_sees_an_earlier_delivery(cfg, tmp_path):
    base = tmp_path / "projects"
    delivered(base, "spikes", SPIKES)
    iv, arc, project = interview(cfg, tmp_path, propose())
    iv.start("Apache log spikes per minute")
    sent = arc.requests[0].messages[-1]["content"]
    assert "Earlier delivered projects" in sent and "main.py: reads the log" in sent
    assert project.board.latest("precedent").data["matches"][0]["language"] == "python"


def test_the_planning_critic_sees_an_earlier_delivery(cfg, tmp_path):
    from tests.test_planning import PLAN_B, critique, loop

    delivered(tmp_path / "projects", "primers", "Make me a primer design tool, PCR primers")
    pl, arc, project = loop(cfg, tmp_path, critique(), "not json", "not json", "not json")
    try:
        pl.run()
    except RuntimeError:
        pass  # only the critic's prompt matters here
    assert "Earlier delivered projects" in arc.requests[0].messages[-1]["content"]
    assert project.board.latest("precedent") is not None and PLAN_B
