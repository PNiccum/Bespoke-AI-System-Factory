"""The knowledge loop: topics, wiki coverage, plain-code research, approval, cited pages, verification, re-check."""

import json

import pytest

import factory.pipeline.knowledge as knowledge_module
from factory.pipeline.knowledge import KnowledgeLoop, cited
from factory.pipeline.planning import PlanningLoop
from factory.pipeline.sources import Doc, rebuild_abstract
from factory.pipeline.wiki import Wiki, WikiPage
from factory.registry import Registry
from tests.test_orchestrator import ROOT, Scripted, cfg  # noqa: F401
from tests.test_planning import ANALYSIS, approved_project, critique, edit, recheck, vote


def doc(i, license="public"):
    return Doc(id=f"europepmc:PMC{i}", connector="europepmc", title=f"Primer design study {i}",
               url=f"https://europepmc.org/{i}", year=2020, text=f"Primers should have a Tm of 55-65 C ({i}).",
               citation=f"Author {i} (2020). Primer design study {i}.", license=license)


class Engine:
    def __init__(self, docs):
        self.docs, self.queries = docs, []

    def __call__(self, query, n):
        self.queries.append(query)
        return self.docs[:n]


def topics(*names, enough=False, dropped=()):
    return json.dumps({"enough": enough, "topics": [{"name": n, "why": "needed", "needed_for": ["T2"],
                                                     "hints": ["primer"]} for n in names],
                       "dropped": [{"name": d, "why": "the plan states it"} for d in dropped]})


def queries(connector="europepmc"):
    return json.dumps({"queries": [{"connector": connector, "query": "PCR primer design melting temperature"}]})


def review(*approved, reject=()):
    return json.dumps({"verdicts": [{"id": f"europepmc:PMC{i}", "approve": True, "why": "on topic"} for i in approved]
                      + [{"id": f"europepmc:PMC{i}", "approve": False, "why": "off topic"} for i in reject]})


def draft(*cites, bad=()):
    """A tagged draft quoting each cited document; documents in ``bad`` get a quote they do not contain."""
    body = " ".join(f"Tm should be 55-65 C. {{{{q:europepmc:PMC{i}|" +
                    (f"Primers should have a Tm of 55-65 C ({i})" if i not in bad else "Tm must be 70 C") + "}}"
                    for i in cites)
    return json.dumps({"summary": "How to pick PCR primers.",
                       "body": body + " So both primers should match. {{inference}} Costs are not reported. {{limit}}"})


def check(ok=True):
    return json.dumps({"accurate": ok, "issues": [] if ok else ["Tm range: not in the source"]})


def planned(cfg, tmp_path):
    llm, project = approved_project(cfg, tmp_path)
    llm.providers["arc"] = Scripted(critique(), ANALYSIS, edit(), recheck(), vote("A"), vote("A"))
    PlanningLoop(llm, Registry.load(cfg, ROOT / "registry"), project).run()
    return llm, project


def knowledge(cfg, tmp_path, *replies, docs=(1, 2, 3), wiki=None, engines=None):
    llm, project = planned(cfg, tmp_path)
    arc = Scripted(*replies)
    llm.providers["arc"] = arc
    engines = engines or {"europepmc": Engine([doc(i) for i in docs])}
    wiki = wiki or Wiki(tmp_path / "wiki")
    return KnowledgeLoop(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, engines), arc, project, wiki


def test_missing_topic_is_researched_written_verified_then_rechecked(cfg, tmp_path):
    kl, arc, project, wiki = knowledge(cfg, tmp_path, topics("PCR primer design"), queries(), review(1, 2, reject=[3]),
                                       draft(1, 2), check(True), topics(enough=True))
    result = kl.run()
    assert result.ok and result.covered == {"PCR primer design": "pcr-primer-design"} and result.rounds == 2
    page = wiki.covers("PCR primer design")
    assert page and page.verified and page.sources == ["europepmc:PMC1", "europepmc:PMC2"]
    assert (wiki.root / "pcr-primer-design.md").read_text().startswith("# PCR primer design")
    assert (project.root / "sources" / "europepmc_PMC1.json").exists()
    assert not (project.root / "sources" / "europepmc_PMC3.json").exists()  # rejected
    families = [cfg.model_info(r.model).family for r in arc.requests]
    assert families[4] != families[3]  # the verifier is not the librarian's family
    assert "How to pick PCR primers" in arc.requests[5].messages[-1]["content"]  # the re-check sees the new page
    kinds = [e.kind for e in project.board.entries()]
    assert kinds[-1] == "file" and kinds[-2] == "knowledge" and "wiki_check" in kinds
    with pytest.raises(RuntimeError, match="already approved"):
        kl.run()
    arc.texts.append(topics(enough=True))
    assert kl.run(again=True).ok and project.board.latest("knowledge").data["rounds"] == 1


def test_a_topic_the_wiki_covers_needs_no_research(cfg, tmp_path):
    wiki = Wiki(tmp_path / "wiki")
    wiki.save(WikiPage(topic="PCR primer design", summary="s", body="b", sources=["a", "b"], verified=True))
    kl, arc, _, _ = knowledge(cfg, tmp_path, topics("pcr primer design"), wiki=wiki)
    result = kl.run()
    assert result.ok and len(arc.requests) == 1


def test_an_inaccurate_page_gets_one_rewrite_then_stays_unverified(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge_module, "MAX_ROUNDS", 2)  # one research pass, then the check
    kl, arc, project, wiki = knowledge(cfg, tmp_path, topics("Primer3 settings"), queries(), review(1, 2),
                                       draft(1, 2), check(False), draft(1, 2), check(False),
                                       topics("Primer3 settings"))
    result = kl.run()
    assert not result.ok and result.missing == ["Primer3 settings"]
    page = wiki.get("Primer3 settings")
    assert not page.verified and page.issues == ["Tm range: not in the source"]
    assert "Fix these problems" in arc.requests[5].messages[-1]["content"]
    assert wiki.covers("Primer3 settings") is None


def test_uncited_drafts_never_reach_the_verifier(cfg, tmp_path):
    kl, arc, _, wiki = knowledge(cfg, tmp_path, topics("Primer3 settings"), queries(), review(1, 2), draft(1),
                                 draft(1, 2), check(True), topics(enough=True))
    assert kl.run().ok
    assert "cites 1 documents; it needs 2" in arc.requests[4].messages[-1]["content"]


def test_too_few_approved_sources_searches_again_then_gives_up(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge_module, "MAX_ROUNDS", 2)
    kl, arc, project, _ = knowledge(cfg, tmp_path, topics("Rare topic"), queries(), review(1, reject=[2, 3]),
                                    queries("openalex"), topics("Rare topic"), docs=(1, 2, 3))
    result = kl.run()
    assert not result.ok
    q = project.board.entries("query")
    assert q[-1].data["error"] == "unknown engine"  # openalex is not wired in this test
    assert "rejected" in arc.requests[3].messages[-1]["content"]


def test_a_misquoted_document_is_dropped_before_the_verifier(cfg, tmp_path):
    kl, arc, project, wiki = knowledge(cfg, tmp_path, topics("Primer3 settings"), queries(), review(1, 2, 3),
                                       draft(1, 2, 3, bad=[3]), check(True), topics(enough=True))
    assert kl.run().ok
    page = wiki.get("Primer3 settings")
    assert page.sources == ["europepmc:PMC1", "europepmc:PMC2"] and "_(inference)_" in page.body
    assert "quote not found in europepmc:PMC3" in str(project.board.entries("note")[-1].data["quote_check"])
    assert '"Primers should have a Tm of 55-65 C (1)"' in arc.requests[4].messages[-1]["content"]  # quotes to verifier


def test_a_failing_engine_is_recorded_not_fatal(cfg, tmp_path):
    def broken(query, n):
        raise ConnectionError("no route")
    kl, _, project, _ = knowledge(cfg, tmp_path, topics("Rare topic"), queries(), queries(), topics(enough=True),
                                  engines={"europepmc": broken})
    result = kl.run()
    assert not result.ok and result.missing == ["Rare topic"]  # not named again is not the same as not needed
    assert "ConnectionError" in project.board.entries("query")[0].data["error"]


def test_an_earlier_topic_goes_only_when_dropped_with_a_reason(cfg, tmp_path):
    kl, _, project, _ = knowledge(cfg, tmp_path, topics("Rare topic"), queries(), queries(),
                                  topics(enough=True, dropped=["rare topic"]), docs=())
    result = kl.run()
    assert result.ok and result.dropped == {"Rare topic": "the plan states it"}
    assert "Still without a verified page: Rare topic" in kl.llm.providers["arc"].requests[-1].messages[-1]["content"]


def test_arc_only_sources_make_an_arc_only_page(cfg, tmp_path):
    engines = {"europepmc": Engine([doc(1, "arc_only"), doc(2)])}
    kl, _, project, wiki = knowledge(cfg, tmp_path, topics("Licensed topic"), queries(), review(1, 2), draft(1, 2),
                                     check(True), topics(enough=True), engines=engines)
    kl.run()
    assert wiki.get("Licensed topic").license == "arc_only"
    assert all(e.data.get("topic") != "Licensed topic" or e.license == "arc_only"
               for e in project.board.entries("wiki_page"))
    assert project.board.for_paid(project.board.entries("wiki_page")) == []


def test_the_loop_needs_the_planning_loop_first(cfg, tmp_path):
    llm, project = approved_project(cfg, tmp_path)
    with pytest.raises(RuntimeError, match="planning loop"):
        KnowledgeLoop(llm, Registry.load(cfg, ROOT / "registry"), project, Wiki(tmp_path / "w"), {}).run()


def test_helpers():
    assert rebuild_abstract({"design": [1], "Primer": [0], "matters": [2]}) == "Primer design matters"
    assert cited("x [a:1] y [b:2]", ["a:1", "b:2", "c:3"]) == ["a:1", "b:2"]
    wiki_page = WikiPage(topic="PCR primer design", summary="", body="", sources=[])
    assert wiki_page.slug == "pcr-primer-design"


def test_a_reworded_topic_keeps_its_first_name():
    from factory.pipeline.knowledge import same_topic
    earlier = ["SantaLucia nearest-neighbor Tm calculation", "NCBI E-utilities gene to mRNA workflow"]
    assert same_topic("Nearest-neighbor Tm calculation (SantaLucia)", earlier) == earlier[0]
    assert same_topic("Primer scoring heuristics", earlier) == "Primer scoring heuristics"


def test_a_page_written_under_the_first_name_counts_when_the_topic_is_reworded(cfg, tmp_path):
    kl, arc, _, _ = knowledge(cfg, tmp_path, topics("PCR primer design rules"), queries(), review(1, 2),
                              draft(1, 2), check(True), topics("Rules for PCR primer design"))
    result = kl.run()
    assert result.ok and result.covered == {"PCR primer design rules": "pcr-primer-design-rules"}
    assert "Topics asked for earlier: PCR primer design rules" in arc.requests[-1].messages[-1]["content"]


def test_a_failed_knowledge_loop_can_run_again_and_build_refuses_until_it_passes(cfg, tmp_path, monkeypatch):
    from factory.pipeline.run import build
    monkeypatch.setattr(knowledge_module, "MAX_ROUNDS", 1)
    kl, arc, project, wiki = knowledge(cfg, tmp_path, topics("PCR primer design"))
    assert not kl.run().ok
    with pytest.raises(RuntimeError, match="did not approve"):
        build(kl.llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {})
    arc.texts.append(topics(enough=True))
    assert kl.run().ok  # a failed loop may run again


def test_the_evaluator_is_told_to_leave_out_general_software_topics(cfg, tmp_path):
    kl, arc, _, _ = knowledge(cfg, tmp_path, topics(enough=True))
    kl.run()
    assert "Leave out general software engineering" in arc.requests[0].messages[-1]["content"]


def test_only_verified_pages_count_toward_enough_and_an_approval_names_them(cfg, tmp_path):
    wiki = Wiki(tmp_path / "wiki")
    wiki.save(WikiPage(topic="NCBI E-utilities usage", summary="efetch", body="b", sources=["a", "b"], verified=True))
    wiki.save(WikiPage(topic="Primer Tm calculation", summary="rejected", body="b", sources=["a"], verified=False))
    kl, arc, _, _ = knowledge(cfg, tmp_path, topics(enough=True), wiki=wiki)
    result = kl.run()
    shown = arc.requests[0].messages[-1]["content"]
    assert "NCBI E-utilities usage" in shown and "Primer Tm calculation" not in shown
    assert result.ok and result.covered == {"NCBI E-utilities usage": "ncbi-e-utilities-usage"}


def test_one_reference_manual_is_enough_sources_for_a_page():
    from factory.pipeline.wiki import enough_sources

    assert enough_sources(["docs:ncbi_eutils_indepth"]) and enough_sources(["europepmc:1", "openalex:2"])
    assert not enough_sources(["europepmc:1"]) and not enough_sources([])


def test_a_topic_first_asked_for_in_the_last_round_is_deferred_not_missing(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge_module, "MAX_ROUNDS", 2)
    kl, _, project, _ = knowledge(cfg, tmp_path, topics("Primer rules"), queries(), queries(),
                                  topics("Retry strategy", dropped=["Primer rules"]), docs=())
    result = kl.run()
    assert result.ok and result.missing == [] and result.deferred == ["Retry strategy"]
    assert project.board.latest("topics").data["deferred"] == ["Retry strategy"]
