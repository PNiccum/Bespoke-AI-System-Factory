"""The read-only inspection pages: the wiki viewer, the project browser, and runs and spend."""

import json
from pathlib import Path

import pytest

import factory.config
from factory.config import load_config
from factory.pipeline import cite
from factory.pipeline.blackboard import Project
from factory.pipeline.wiki import Wiki, WikiPage
from factory.ui import inspect

ROOT = Path(__file__).resolve().parents[1]
TEXT = "Primers with a Tm of 55 to 65 C work best. Both primers should be designed together."


def body(pages):
    blocks = cite.parse("## Melting\nPrimers should melt at 55-65 C. {{q:pmc:1|primers with a Tm of 55 to 65 C}}\n"
                        "See [[gc-content]] and [[nowhere]]. {{limit}}")
    cite.check(blocks, {"pmc:1": TEXT})
    return cite.render(blocks, pages)


@pytest.fixture
def wiki(tmp_path):
    w = Wiki(tmp_path / "wiki")
    w.save(WikiPage(topic="GC content", summary="GC share.", body="GC is $2 cheap [[primer-melting]].",
                    sources=["docs:gc"], verified=True, links=["primer-melting"]))
    w.save(WikiPage(topic="Primer melting", summary="Tm range.", body=body({"gc-content": "GC content"}),
                    sources=["pmc:1"], citations={"pmc:1": "Smith 2020"}, verified=True,
                    links=["gc-content", "nowhere"]))
    w.save(WikiPage(topic="Old topic", summary="x", body="y", sources=[], issues=["quote not found"]))
    return w


def test_a_page_shows_its_quotes_and_links_to_other_pages_in_the_viewer(wiki):
    pages = wiki.pages()
    v = inspect.view(wiki.get("Primer melting"), pages)
    assert v.quotes and v.quotes[0][0] == 1 and "primers with a Tm of 55 to 65 C" in v.quotes[0][1]
    assert "[^1]" not in v.body and "\\[pmc:1\\]¹" in v.body
    assert "[GC content](?page=gc-content)" in v.body and ".md)" not in v.body
    assert v.links == ["gc-content"] and v.missing == ["nowhere"] and v.backlinks == ["gc-content"]
    assert "\\$2" in inspect.view(wiki.get("GC content"), pages).body


def test_status_search_and_log(wiki):
    status = {r["topic"]: r["status"] for r in inspect.wiki_rows(wiki.pages())}
    # one reference manual is enough; one paper is not
    assert status == {"GC content": "verified", "Primer melting": "verified, too few sources",
                      "Old topic": "not verified"}
    assert [p.topic for p in wiki.pages() if inspect.matches(p, "TM range")] == ["Primer melting"]
    log = inspect.wiki_log(wiki)
    assert len(log) == 3 and "Old topic" in log[0] and "(?page=old-topic)" in log[0]


def test_project_stage_and_summary(tmp_path):
    p = Project.create(tmp_path / "projects", "primer designer")
    p.board.post("request", "person", {"text": "Design primers for BRCA1"})
    p.board.post("approval", "person", {"plan": "0001"})
    p.board.post("decision", "code", {"winner": "A"}, license="arc_only")
    rows = inspect.project_rows(inspect.projects(tmp_path / "projects"))
    assert rows == [{"project": p.id, "stage": "plan decided", "entries": 3, "arc_only": 1,
                     "request": "Design primers for BRCA1"}]
    assert inspect.entry_summary({"topic": "  GC\ncontent "}) == "GC content"
    assert inspect.entry_summary({"winner": "A"}) == '{"winner": "A"}'


CALLS = [
    {"tag": "a", "role": "builder", "model": "qwen", "provider": "arc", "input_tokens": 10, "output_tokens": 5,
     "real_usd": 0.0, "priced_usd": 0.01},
    {"tag": "a", "role": "builder", "model": "qwen", "provider": "arc", "cached": True},
    {"tag": "b", "role": "builder", "model": "qwen", "provider": "arc", "error": "ReadTimeout", "attempt": 0},
    {"event": "spend_gate", "role": "critic", "model": "gpt", "provider": "openai", "reason": "stuck"},
    {"tag": "c", "role": "critic", "model": "gpt", "provider": "openai", "input_tokens": 100, "output_tokens": 50,
     "real_usd": 0.02, "priced_usd": 0.02},
]


def test_call_totals_count_cache_errors_and_gates_apart():
    assert inspect.call_totals(CALLS) == {"calls": 2, "cached": 1, "errors": 1, "gates": 1, "real_usd": 0.02,
                                          "priced_usd": 0.03, "tokens": 165}
    assert [(r["model"], r["calls"], r["cached"], r["errors"]) for r in inspect.by_model(CALLS)] == [
        ("gpt", 1, 0, 0), ("qwen", 1, 1, 1)]


def test_a_cut_short_log_line_is_skipped(tmp_path):
    path = tmp_path / "llm_calls.jsonl"
    path.write_text(json.dumps(CALLS[0]) + '\n{"tag": "x", "ro', encoding="utf-8")
    assert len(inspect.read_jsonl(path)) == 1


# --------------------------------------------------------------------------
# The pages themselves
# --------------------------------------------------------------------------

@pytest.fixture
def app(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    import streamlit as st
    cfg = load_config(ROOT / "configs" / "factory.yaml")
    cfg.root = tmp_path
    monkeypatch.setattr(factory.config, "load_config", lambda *_: cfg)
    st.cache_resource.clear()
    yield lambda page: testing.AppTest.from_file(str(ROOT / "pages" / page), default_timeout=30)
    st.cache_resource.clear()


def test_every_page_renders_with_an_empty_workspace(app):
    for page in ("2_Wiki.py", "3_Projects.py", "4_Runs_and_spend.py"):
        at = app(page).run()
        assert not at.exception and at.info, page


def test_the_wiki_page_opens_the_page_the_link_names(app, tmp_path):
    cfg = load_config(ROOT / "configs" / "factory.yaml")
    cfg.root = tmp_path
    w = Wiki(cfg.work_dir / "wiki")
    w.save(WikiPage(topic="GC content", summary="GC share.", body="See [Primer melting](primer-melting.md).",
                    sources=["docs:gc"], verified=True, links=["primer-melting"]))
    w.save(WikiPage(topic="Primer melting", summary="Tm range.", body="Melts at 60 C.", sources=["a", "b"],
                    verified=False, issues=["quote not found"]))
    at = app("2_Wiki.py")
    at.query_params["page"] = "primer-melting"
    at.run()
    assert not at.exception
    assert at.header[0].value == "Primer melting" and "quote not found" in at.error[0].value
    assert [m.value for m in at.metric] == ["2", "1", "1", "0"]
    at.radio[0].set_value("gc-content").run()
    assert at.header[0].value == "GC content" and at.query_params["page"] == "gc-content"
    assert any("(?page=primer-melting)" in m.value for m in at.markdown)


def test_the_project_and_runs_pages_show_what_is_on_disk(app, tmp_path):
    cfg = load_config(ROOT / "configs" / "factory.yaml")
    cfg.root = tmp_path
    p = Project.create(cfg.work_dir / "projects", "primer designer")
    p.board.post("request", "person", {"text": "Design primers"})
    p.write_file("out/primers.py", "print(1)\n", "builder")
    at = app("3_Projects.py").run()
    assert not at.exception and "interviewing" in at.subheader[0].value
    run = tmp_path / cfg.runs_dir / "20261103T000000Z-ui"
    run.mkdir(parents=True)
    (run / "llm_calls.jsonl").write_text("\n".join(json.dumps(c) for c in CALLS), encoding="utf-8")
    at = app("4_Runs_and_spend.py").run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Model calls"] == "2"
