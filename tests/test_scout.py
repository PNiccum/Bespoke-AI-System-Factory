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


def test_sandbox_packages_come_from_the_image_list(tmp_path, monkeypatch):
    from factory.pipeline import scout as scout_mod
    req = tmp_path / "requirements.txt"
    req.write_text("# pinned\npytest==8.3.5\nPyYAML==6.0.2\nnot_installed_pkg_zz==1.0\n")
    monkeypatch.setattr(scout_mod, "SANDBOX_REQUIREMENTS", req)
    monkeypatch.delenv("ASPIRE_SANDBOX_PACKAGES", raising=False)
    assert scout_mod.listed_packages(req) == {"pytest", "pyyaml", "not-installed-pkg-zz"}
    monkeypatch.setenv("ASPIRE_SANDBOX", "docker")
    assert "not-installed-pkg-zz" in scout_mod.sandbox_packages()  # the image installs it
    monkeypatch.setenv("ASPIRE_SANDBOX", "process")
    have = scout_mod.sandbox_packages()  # the process sandbox is this interpreter: only what is installed here
    assert "pyyaml" in have and "not-installed-pkg-zz" not in have
    assert "yaml" in scout_mod.sandbox_imports() and "json" in scout_mod.sandbox_imports()


def test_the_shipped_image_list_is_pinned():
    from factory.pipeline.scout import SANDBOX_REQUIREMENTS
    lines = [x.split("#")[0].strip() for x in SANDBOX_REQUIREMENTS.read_text().splitlines()]
    assert all("==" in x for x in lines if x) and any(x.startswith("pytest==") for x in lines)


def test_reference_repos_get_readme_and_files(cfg, tmp_path, monkeypatch):
    monkeypatch.delenv("ASPIRE_SANDBOX_PACKAGES", raising=False)

    def details(request):
        url = str(request.url)
        if url.endswith("/readme"):
            return httpx.Response(200, text="# logspike\nRolling median of counts per minute.")
        if url.endswith("/contents"):
            return httpx.Response(200, json=[{"name": "logspike", "type": "dir"}, {"name": "README.md", "type": "file"}])
        return handler(request)

    http, _ = mock(details)
    llm, _ = planned(cfg, tmp_path)
    llm.providers["arc"] = Scripted(PROPOSAL)
    budget = CallBudget("t", 50)
    s = Scout(llm, Registry.load(cfg, ROOT / "registry").agents["research_orchestrator"],
              PyPI(tmp_path / "p", budget, http=http), GitHubRepos(tmp_path / "g", budget, http=http), now=lambda: NOW)
    found = s.run("t", "{}")
    c = next(c for c in found.candidates if c.name == "a/logspike")
    assert "Rolling median" in c.readme and c.files == ["logspike/", "README.md"]
    assert "Rolling median" in found.render(detail=True) and "Rolling median" not in found.render()
    assert not next(c for c in found.candidates if c.name == "b/nolicence").readme  # skipped: not read


def test_imports_the_sandbox_lacks_block_and_unused_reuse_is_recorded(tmp_path, monkeypatch):
    from factory.pipeline.scout import Candidate, ReuseResult, check_imports, unused_reuse
    monkeypatch.setenv("ASPIRE_SANDBOX", "process")
    out = tmp_path / "output"
    (out / "src" / "spikes").mkdir(parents=True)
    (out / "tests").mkdir()
    (out / "src" / "spikes" / "__init__.py").write_text("import json\nimport matplotlib.pyplot as plt\nfrom . import x\n")
    (out / "tests" / "test_a.py").write_text("import pytest\nfrom spikes import run\nimport helper\n")
    (out / "helper.py").write_text("import os\n")
    files = ["src/spikes/__init__.py", "tests/test_a.py", "helper.py"]
    assert check_imports(out, files) == {"matplotlib": ["src/spikes/__init__.py"]}
    reuse = ReuseResult(candidates=[
        Candidate(kind="stdlib", name="statistics", url="u", summary="s", use="import", why="w", fetched="d"),
        Candidate(kind="stdlib", name="json", url="u", summary="s", use="import", why="w", fetched="d"),
        Candidate(kind="github", name="a/logspike", url="u", summary="s", use="reference", why="w", fetched="d")])
    claimed = ["statistics: medians", "json: output", "a/logspike: rolling window idea"]
    assert unused_reuse(claimed, reuse, out, files) == ["statistics"]


def test_checks_with_an_unavailable_import_are_not_ok():
    from factory.pipeline.critique import json_checks
    from factory.pipeline.production import Checks
    c = Checks(compiled=["a.py"], syntax_errors={}, tests_ran=True, tests_ok=True,
               unavailable_imports={"matplotlib": ["a.py"]})
    assert not c.ok and "matplotlib in a.py" in json_checks(c)
