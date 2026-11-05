"""PR A: the knowledge loop, the plain-code checks and the sources report to the event log; the domain tag; the
cross-run tallies. Models are scripted."""

import httpx

import factory.pipeline.knowledge as knowledge_module
from factory.events import EventLog, by_domain
from factory.pipeline.blackboard import Project
from factory.pipeline.domain import classify, project_domain
from factory.pipeline.sources import ReferenceDocs
from factory.ui import inspect
from tests.test_knowledge import check, draft, knowledge, queries, review, topics
from tests.test_orchestrator import cfg  # noqa: F401


def events_of(kl):
    return kl.events.events()


def test_a_topic_that_stalls_is_an_open_event_and_a_verified_page_corrects_it(cfg, tmp_path, monkeypatch):
    # first page rejected, then the gap search finds nothing new: the topic stays without a page -> open events
    monkeypatch.setattr(knowledge_module, "MAX_OPEN_GAPS", 0)
    monkeypatch.setattr(knowledge_module, "MAX_ROUNDS", 2)
    kl, _, _, _ = knowledge(cfg, tmp_path, topics("Primer3 settings"), queries(), review(1, 2), draft(1, 2),
                            check(False), draft(1, 2), check(False), queries(), topics("Primer3 settings"),
                            queries(), queries(), queries())
    assert not kl.run().ok
    ev = events_of(kl)
    kinds = [e.message.split(":")[0] for e in ev]
    assert "page rejected (draft 1)" in kinds and "knowledge loop stopped" in kinds
    assert all(e.status == "open" and "topic:Primer3 settings" in e.refs for e in ev if e.source == "knowledge")
    assert all(e.domain == "bio" for e in ev)  # the project ("primer design tool") was tagged at the interview


def test_a_rewrite_that_verifies_corrects_the_rejection(cfg, tmp_path):
    kl, _, _, wiki = knowledge(cfg, tmp_path, topics("Primer3 settings"), queries(), review(1, 2),
                               draft(1, 2), check(False), draft(1, 2), check(True), topics(enough=True))
    assert kl.run().ok and wiki.get("Primer3 settings").verified
    (rej,) = [e for e in events_of(kl) if e.message.startswith("page rejected")]
    assert rej.status == "corrected" and "verified page now covers" in rej.how


def test_a_failing_engine_and_a_deferred_topic_are_events(cfg, tmp_path, monkeypatch):
    def broken(query, n):
        raise ConnectionError("no route")
    kl, _, _, _ = knowledge(cfg, tmp_path, topics("Rare topic"), queries(), queries(), queries(),
                            topics(enough=True), engines={"europepmc": broken})
    kl.run()
    ev = events_of(kl)
    assert any(e.source == "sources" and "ConnectionError" in e.message for e in ev)
    assert any("knowledge loop stopped" in e.message for e in ev)


def test_a_docs_connector_reports_the_pages_it_skipped(tmp_path):
    def handler(request):
        if "uniprot" in str(request.url):
            return httpx.Response(406)
        return httpx.Response(200, text="<html><body>Please enable JavaScript to continue</body></html>")
    from factory.net.http import CallBudget
    docs = ReferenceDocs(tmp_path, CallBudget("docs", 100), http=httpx.Client(transport=httpx.MockTransport(handler)))
    found = docs.search("uniprot rest api json fields", 3)
    assert found == [] and docs.skipped and any("HTTPStatusError" in s or "blocked" in s for s in docs.skipped)


def test_a_skipped_catalog_page_is_an_event_with_the_topic_ref(cfg, tmp_path):
    def handler(request):
        return httpx.Response(406)
    from factory.net.http import CallBudget
    docs = ReferenceDocs(tmp_path / "docs", CallBudget("docs", 100),
                         http=httpx.Client(transport=httpx.MockTransport(handler)))
    kl, _, _, _ = knowledge(cfg, tmp_path, topics("UniProt REST API"), queries("docs"), queries("docs"),
                            queries("docs"), topics(enough=True), engines={"docs": docs.search})
    kl.run()
    skips = [e for e in events_of(kl) if "page skipped" in e.message]
    assert skips and all("topic:UniProt REST API" in e.refs and "HTTPStatusError" in e.message for e in skips)


def test_skipped_pages_are_per_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from factory.net.http import CallBudget
    from factory.pipeline.sources import skipped_pages

    def handler(request):
        return httpx.Response(406)
    docs = ReferenceDocs(tmp_path, CallBudget("docs", 100), http=httpx.Client(transport=httpx.MockTransport(handler)))

    def one(q):
        docs.search(q, 2)
        return skipped_pages(docs.search)
    with ThreadPoolExecutor(2) as pool:
        a, b = pool.map(one, ["uniprot rest api json", "europe pmc rest search"])
    assert a and b and a != b  # each thread kept its own list
    assert any("uniprot" in x for x in a) and not any("uniprot" in x for x in b)
    assert skipped_pages(lambda q, n: []) == []  # a plain function reports nothing


def test_classify_is_deterministic_and_bio_wins_a_tie():
    assert classify("Make me a primer design tool for NCBI genes") == "bio"
    assert classify("Profile a CSV: columns, nulls, summary statistics") == "data"
    assert classify("Fix the failing test in this repository") == "code"
    assert classify("Hello") == "other"
    assert classify("protein csv") == "bio"  # one hint each: bio wins the tie
    assert classify("a general internal client with a different summary") == "text"  # gene/rna/cli/diff: no match
    assert classify("a general internal client, suitable and different") == "other"
    assert classify("proteomics of the proteome") == "bio"  # stem hints match word starts
    # generic words do not count: a bio request that mentions a report and a document several times stays bio
    mixed = ("Write a report on the protein list: the report should document each gene in a table, and the "
             "document must profile every function of the module")
    assert classify(mixed) == "bio"
    assert classify("write a report that documents the functions in a table") == "other"


def test_a_project_without_a_tag_is_classified_from_its_task_text(tmp_path):
    p = Project.create(tmp_path, "old")
    p.board.post("request", "person", {"text": "Summarise these articles from OpenAlex"})
    assert project_domain(p) == "text"
    p2 = Project.create(tmp_path, "tagged")
    p2.board.post("request", "person", {"text": "whatever", "domain": "code"})
    assert project_domain(p2) == "code"


def test_by_domain_groups_events_and_names_the_untagged(tmp_path):
    rows = []
    for run, domain, msgs in (("r1", "bio", ["timeout", "timeout"]), ("r2", "bio", ["timeout"]), ("r3", "", ["boom"])):
        log = EventLog(tmp_path / run / "events.jsonl", run)
        log.domain = domain
        ids = [log.record("llm", m, stage="s") for m in msgs]
        if run == "r1":
            log.resolve(ids[0], "retry 1 succeeded")
        rows += log.events()
    out = by_domain(rows)
    assert [r["domain"] for r in out] == ["(untagged)", "bio"]
    bio = out[1]
    assert (bio["runs"], bio["events"], bio["corrected"], bio["open"]) == (2, 3, 1, 2)
    assert bio["most frequent"] == "s: timeout (3x)"
    assert out[0]["most frequent"] == "s: boom (1x)"


def test_stage_tallies_count_from_the_blackboards(cfg, tmp_path):
    kl, _, project, _ = knowledge(cfg, tmp_path, topics("Primer3 settings"), queries(), review(1, 2), draft(1, 2),
                                  check(True), topics(enough=True))
    assert kl.run().ok
    tallies = {(r["stage"], r["measure"]): r["value"] for r in inspect.stage_tallies([project])}
    assert tallies[("wiki", "pages verified per check")] == "1/1 (100%)"
    assert tallies[("interview", "plans approved per plan proposed")].startswith("1/")
    assert tallies[("final QA", "approved on first QA")] == "-"


def test_time_lost_counts_cut_offs_and_failures():
    calls = [{"seconds": 10, "stop_reason": "end_turn"}, {"seconds": 30, "stop_reason": "max_tokens"},
             {"error": "boom"}, {"fallback_to": "x", "error": "404"}, {"cached": True, "seconds": 5}]
    t = inspect.time_lost(calls)
    assert t == {"model_seconds": 40.0, "lost_seconds": 30.0, "lost_share": 0.75, "cut_off": 1, "refused": 0,
                 "failed_calls": 2, "fallbacks": 1}


def test_a_budget_deferred_topic_is_recorded_once(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge_module, "MAX_RESEARCHED", 1)
    monkeypatch.setattr(knowledge_module, "MAX_ROUNDS", 1)
    kl, _, _, _ = knowledge(cfg, tmp_path, topics("Primer3 settings", "Salt correction"), queries(), review(1, 2),
                            draft(1, 2), check(True))
    kl.run()
    salt = [e for e in events_of(kl) if "topic:Salt correction" in e.refs]
    assert len(salt) == 1 and salt[0].message.startswith("topic deferred: research budget")
