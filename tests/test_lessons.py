"""Lessons page and known-bad sources, on synthetic data."""

import json
from datetime import UTC, datetime, timedelta

from factory.events import EventLog
from factory.llm import LLM
from factory.pipeline.blackboard import Project
from factory.pipeline.knowledge import Topic
from factory.pipeline.lessons import BadSources, domain_of, lessons_dir, read_lessons, write_lessons
from tests.test_knowledge import Engine, check, doc, draft, knowledge, queries, review, topics  # noqa: F401
from tests.test_orchestrator import ROOT, Scripted, cfg  # noqa: F401


def test_domain_comes_from_the_request_tag_or_its_words(tmp_path):
    p = Project.create(tmp_path, "t")
    p.board.post("request", "person", {"text": "Filter the protein list from a mass spec run"})
    assert domain_of(p) == "proteomics"
    q = Project.create(tmp_path, "u")
    q.board.post("request", "person", {"text": "anything", "domain": "Lab Automation"})
    assert domain_of(q) == "lab-automation"
    assert domain_of(Project.create(tmp_path, "v")) == "general"


def test_bad_sources_are_per_topic_expire_and_clear(tmp_path):
    bad = BadSources(tmp_path / "lessons" / "bad_sources.json")
    bad.record("proteomics", "UniProt API", "web:1", "paywalled", title="T", project="p1")
    bad.record("proteomics", "uniprot  api", "web:1", "paywalled again", project="p2")
    k = bad.known("proteomics", "UniProt API")
    assert k["web:1"]["count"] == 2 and k["web:1"]["projects"] == ["p1", "p2"]
    assert bad.known("proteomics", "other topic") == {} and bad.known("software", "UniProt API") == {}
    later = datetime.now(UTC) + timedelta(days=31)
    assert bad.known("proteomics", "UniProt API", now=later) == {}  # a source gets another chance
    assert bad.clear("proteomics", "UniProt API", "web:1") and not bad.clear("proteomics", "UniProt API", "web:1")
    assert bad.known("proteomics", "UniProt API") == {}


def test_a_rejected_document_is_remembered_and_skipped_in_the_next_project(cfg, tmp_path):
    kl, arc, project, wiki = knowledge(cfg, tmp_path, topics("PCR primer design"), queries(), review(1, 2, reject=[3]),
                                       draft(1, 2), check(True), topics(enough=True))
    assert kl.run().ok
    (r,) = project.board.entries("rejected")
    assert r.data["id"] == "europepmc:PMC3" and r.data["why"] == "off topic"
    saved = BadSources(tmp_path / "lessons" / "bad_sources.json").known(kl.domain, "PCR primer design")
    assert list(saved) == ["europepmc:PMC3"]

    # a second project finds PMC3 again for the same topic (in its own workspace, sharing the list): no model sees it
    (tmp_path / "second").mkdir()
    kl2, arc2, project2, _ = knowledge(cfg, tmp_path / "second", queries(), review(4, 5), draft(4, 5), check(True),
                                       docs=(3, 4, 5))
    kl2.bad = kl.bad
    kl2.research(Topic(name="PCR primer design", why="needed"))
    assert "PMC3" not in arc2.requests[1].messages[-1]["content"] and "PMC4" in arc2.requests[1].messages[-1]["content"]
    assert "Sources that failed for this topic" in arc2.requests[0].messages[-1]["content"]
    assert [e.data["skipped_known_bad_sources"] for e in project2.board.entries("note")
            if "skipped_known_bad_sources" in e.data] == [["europepmc:PMC3"]]


def lessons_reply(*items):
    return json.dumps({"lessons": [{"lesson": t, "kind": k, "evidence": ev} for t, k, ev in items]})


def test_a_lesson_must_cite_entries_that_exist(cfg, tmp_path):
    p = Project.create(tmp_path / "projects", "primers")
    p.board.post("request", "person", {"text": "Design PCR primers for this gene"})
    s = p.board.post("source", "research_orchestrator", {"id": "europepmc:PMC1", "title": "Primer study"})
    n = p.board.post("note", "code", {"text": "only 1 approved source after 3 rounds"})
    reply = lessons_reply(
        ("Primer papers from Europe PMC pass review; start there.", "worked", [s.id, "9999-source"]),
        ("Invented lesson with no real evidence.", "failed", ["9999-note"]),
        ("A lesson with no evidence at all.", "watch", []),
        ("Check source counts early; one round left the topic short.", "watch", [n.id]),
        ("Check source counts early; one round left the topic short.", "watch", [n.id]))
    llm = LLM(cfg, tmp_path, use_cache=False, providers={"arc": Scripted(reply)})
    directory = lessons_dir(tmp_path)
    page = write_lessons(llm, p, "delivered", directory)
    rows = read_lessons(directory)
    assert [r["lesson"] for r in rows] == ["Primer papers from Europe PMC pass review; start there.",
                                           "Check source counts early; one round left the topic short."]
    assert rows[0]["evidence"] == [s.id]  # the unknown id is removed
    assert rows[0]["domain"] == "bioinformatics" and page.name == "bioinformatics.md"
    text = page.read_text()
    assert s.id in text and "Invented" not in text and "no evidence at all" not in text


def test_the_same_lesson_is_not_added_twice_and_a_failed_call_adds_nothing(cfg, tmp_path):
    p = Project.create(tmp_path / "projects", "t")
    p.board.post("request", "person", {"text": "x"})
    n = p.board.post("note", "code", {"text": "something"})
    reply = lessons_reply(("Keep it small.", "watch", [n.id]))
    directory = lessons_dir(tmp_path)
    llm = LLM(cfg, tmp_path, use_cache=False, providers={"arc": Scripted(reply, reply)})
    assert write_lessons(llm, p, "delivered", directory)
    assert write_lessons(llm, p, "delivered", directory) is None
    assert len(read_lessons(directory)) == 1
    (tmp_path / "b").mkdir()
    bad = LLM(cfg, tmp_path / "b", use_cache=False, providers={"arc": Scripted("no json", "no json", "no json")})
    assert write_lessons(bad, p, "delivered", directory) is None


def test_the_run_events_are_shown_to_the_lessons_call(cfg, tmp_path):
    p = Project.create(tmp_path / "projects", "t")
    p.board.post("request", "person", {"text": "x"})
    (tmp_path / "run").mkdir()
    arc = Scripted(lessons_reply())
    llm = LLM(cfg, tmp_path / "run", use_cache=False, providers={"arc": arc})
    EventLog(llm.run_dir / "events.jsonl", "run").record("qa", "final QA failed: 3 tests", stage="final QA")
    write_lessons(llm, p, "re-designed (not delivered)", lessons_dir(tmp_path))
    prompt = arc.requests[0].messages[-1]["content"]
    assert "final QA failed" in prompt and "re-designed" in prompt


def test_arc_only_entries_are_not_copied_into_the_prompt(cfg, tmp_path):
    p = Project.create(tmp_path / "projects", "t")
    p.board.post("request", "person", {"text": "x"})
    p.board.post("source", "research_orchestrator", {"id": "lab:1", "title": "Secret lab list"}, license="arc_only")
    arc = Scripted(lessons_reply())
    write_lessons(LLM(cfg, tmp_path, use_cache=False, providers={"arc": arc}), p, "delivered", lessons_dir(tmp_path))
    assert "Secret lab list" not in arc.requests[0].messages[-1]["content"]
