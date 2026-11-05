"""The prior-art scout: model proposals, checked by code against PyPI, GitHub and the standard library."""

import json
from datetime import UTC, datetime

import httpx

from factory.net.http import CallBudget
from factory.pipeline.knowledge import KnowledgeLoop
from factory.pipeline.production import DESIGN_PROMPT
from factory.pipeline.scout import Scout, judge_license, keywords
from factory.pipeline.sources import GitHubRepos, PyPI
from factory.registry import Registry
from tests.test_http import mock
from tests.test_knowledge import Engine, doc, planned, topics
from tests.test_orchestrator import ROOT, Scripted, cfg  # noqa: F401

NOW = datetime(2026, 11, 5, tzinfo=UTC)
PYPI = {
    "apache-log-parser": {"info": {"name": "apache-log-parser", "version": "1.7.0", "summary": "Parse Apache logs",
                                   "description": "Use make_parser(fmt).", "license": "",
                                   "classifiers": ["License :: OSI Approved :: Apache Software License"]},
                          "urls": [{"upload_time_iso_8601": "2025-02-01T00:00:00Z"}]},
    "oldthing": {"info": {"name": "oldthing", "version": "0.1", "summary": "Old", "license": "MIT",
                          "classifiers": []}, "urls": [{"upload_time_iso_8601": "2015-01-01T00:00:00Z"}]},
    "gplthing": {"info": {"name": "gplthing", "version": "2.0", "summary": "GPL",
                          "license_expression": "GPL-3.0-only", "classifiers": []},
                 "urls": [{"upload_time_iso_8601": "2026-01-01T00:00:00Z"}]},
}
REPOS = {"items": [
    {"full_name": "a/logspike", "html_url": "https://github.com/a/logspike", "description": "Spike finder",
     "stargazers_count": 50, "license": {"spdx_id": "MIT"}, "pushed_at": "2026-06-01T00:00:00Z", "topics": ["logs"]},
    {"full_name": "b/nolicence", "html_url": "https://github.com/b/nolicence", "description": "No licence",
     "stargazers_count": 900, "license": None, "pushed_at": "2026-06-01T00:00:00Z"},
]}


def handler(request):
    url = str(request.url)
    if "api.github.com" in url:
        return httpx.Response(200, json=REPOS)
    name = url.split("/pypi/")[1].split("/")[0]
    return httpx.Response(200, json=PYPI[name]) if name in PYPI else httpx.Response(404, json={})


def scout(tmp_path, llm, agent):
    http, calls = mock(handler)
    budget = CallBudget("t", 50)
    return Scout(llm, agent, PyPI(tmp_path / "pypi", budget, http=http),
                 GitHubRepos(tmp_path / "gh", budget, http=http), now=lambda: NOW), calls


PROPOSAL = json.dumps({"queries": ["apache log spikes"], "packages": ["apache-log-parser", "made-up-pkg", "oldthing",
                                                                       "gplthing"],
                       "stdlib": ["statistics", "re", "notamodule", "os.path"]})


def test_proposals_are_checked_by_code_and_rules_decide_their_use(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("ASPIRE_SANDBOX_PACKAGES", raising=False)
    llm, project = planned(cfg, tmp_path)
    llm.providers["arc"] = Scripted(PROPOSAL)
    agent = Registry.load(cfg, ROOT / "registry").agents["research_orchestrator"]
    s, calls = scout(tmp_path, llm, agent)
    found = s.run("Find spikes in Apache logs", "{}")
    by = {c.name: c for c in found.candidates}
    assert "made-up-pkg" not in by and "notamodule" not in by  # invented names are dropped
    assert {by[m].use for m in ("statistics", "re", "os")} == {"import"}
    assert by["apache-log-parser"].use == "reference" and "Apache" in by["apache-log-parser"].license
    assert by["oldthing"].use == "skip" and "stale" in by["oldthing"].why
    assert by["gplthing"].use == "skip" and "copyleft" in by["gplthing"].why
    assert by["a/logspike"].use == "reference" and by["b/nolicence"].use == "skip"
    assert all(c.fetched == "2026-11-05" and c.url.startswith("https://") for c in found.candidates)
    assert "b/nolicence" not in found.render() and "[import] stdlib statistics" in found.render()
    assert any("language%3Apython" in c for c in calls)


def test_a_package_in_the_sandbox_is_importable(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("ASPIRE_SANDBOX_PACKAGES", "apache_log_parser")
    llm, _ = planned(cfg, tmp_path)
    llm.providers["arc"] = Scripted(PROPOSAL)
    s, _ = scout(tmp_path, llm, Registry.load(cfg, ROOT / "registry").agents["research_orchestrator"])
    assert {c.name: c.use for c in s.run("t", "{}").candidates}["apache-log-parser"] == "import"


def test_a_model_failure_falls_back_to_the_tasks_own_words(cfg, tmp_path):
    llm, _ = planned(cfg, tmp_path)
    llm.providers["arc"] = Scripted("nope", "nope", "nope", "nope")
    s, calls = scout(tmp_path, llm, Registry.load(cfg, ROOT / "registry").agents["research_orchestrator"])
    found = s.run("Detect traffic spikes in apache access logs; spikes per minute.", "{}")
    assert found.queries and "spikes" in found.queries[0]
    assert {c.name for c in found.candidates} == {"a/logspike", "b/nolicence"}


def test_the_knowledge_loop_records_prior_art_and_shows_it_to_evaluator_and_designer(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("ASPIRE_SANDBOX_PACKAGES", raising=False)
    llm, project = planned(cfg, tmp_path)
    arc = Scripted(PROPOSAL, topics(enough=True))
    llm.providers["arc"] = arc
    registry = Registry.load(cfg, ROOT / "registry")
    s, _ = scout(tmp_path, llm, registry.agents["research_orchestrator"])
    from factory.pipeline.wiki import Wiki
    result = KnowledgeLoop(llm, registry, project, Wiki(tmp_path / "wiki"), {"europepmc": Engine([doc(1)])},
                           scout=s).run()
    assert result.ok
    assert "apache-log-parser" in arc.requests[1].messages[-1]["content"]  # the evaluator sees the candidates
    entry = project.board.latest("prior_art")
    assert entry and any(c["name"] == "a/logspike" for c in entry.data["candidates"])
    assert (project.root / "reuse.json").exists()
    assert "{reuse}" in DESIGN_PROMPT


def test_connectors_and_helpers(tmp_path):
    http, _ = mock(handler)
    (d,) = PyPI(tmp_path, CallBudget("t", 5), http=http).search("apache-log-parser made-up-pkg", 3)
    assert d.id == "pypi:apache-log-parser" and "make_parser" in d.text
    docs = GitHubRepos(tmp_path / "g", CallBudget("t", 5), http=http).search("logs", 5)
    assert [x.id for x in docs] == ["github:a/logspike", "github:b/nolicence"] and "MIT" in docs[0].text
    assert judge_license("MIT") == (True, "MIT") and not judge_license("")[0] and not judge_license("GPL-3.0")[0]
    assert judge_license("LGPL-2.1")[0]
    assert keywords("parse parse logs logs logs with the data") == "logs parse"


def test_good_candidates_are_remembered_by_task_kind_and_reused_without_searching(cfg, tmp_path, monkeypatch):
    from datetime import timedelta

    from factory.pipeline.reuse_memory import ReuseMemory
    monkeypatch.delenv("ASPIRE_SANDBOX_PACKAGES", raising=False)
    llm, _ = planned(cfg, tmp_path)
    agent = Registry.load(cfg, ROOT / "registry").agents["research_orchestrator"]
    proposal = json.loads(PROPOSAL) | {"kind": "Apache log parsing"}
    llm.providers["arc"] = Scripted(json.dumps(proposal), json.dumps(proposal | {"kind": "apache log parsing"}))
    mem = ReuseMemory(tmp_path / "wiki", now=lambda: NOW)
    s, calls = scout(tmp_path, llm, agent)
    s.memory, s.project = mem, "p1"
    first = s.run("Find spikes in Apache logs", "{}")
    assert not first.recalled and calls
    page = mem.recall("apache log parsing")
    assert {c.name for c in page.candidates} >= {"apache-log-parser", "a/logspike", "statistics"}
    assert not {c.name for c in page.candidates} & {"oldthing", "gplthing", "b/nolicence"}  # skipped ones are not kept
    assert "https://github.com/a/logspike" in (tmp_path / "wiki/reuse/apache-log-parsing.md").read_text()
    assert page.projects == ["p1"] and page.candidates[0].fetched == "2026-11-05"
    calls.clear()
    s.project = "p2"
    again = s.run("Another Apache log task", "{}")
    assert again.recalled and not calls  # no PyPI or GitHub request
    assert {c.name for c in again.usable()} == {c.name for c in first.usable()}
    assert any("remembered for" in c.why for c in again.candidates)
    assert mem.find("apache log parsing").projects == ["p1", "p2"]
    mem.now = lambda: NOW + timedelta(days=200)
    assert mem.recall("apache log parsing") is None  # stale: search again
    assert not mem.remember("empty kind", [c for c in first.candidates if c.use == "skip"])
