"""PR A: the knowledge loop, the plain-code checks and the sources report to the event log; the domain tag; the
cross-run tallies. Models are scripted."""

import httpx

import factory.pipeline.knowledge as knowledge_module
from factory.events import EventLog, by_domain, log_of
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


def test_a_wrapped_searcher_reports_no_skips(tmp_path):
    from factory.net.http import CallBudget
    from factory.pipeline.sources import skipped_pages
    docs = ReferenceDocs(tmp_path, CallBudget("docs", 100),
                         http=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(406))))
    docs.search("uniprot rest api json fields", 3)
    from functools import partial
    assert skipped_pages(docs.search) and skipped_pages(partial(docs.search, n=3)) == skipped_pages(docs.search)
    assert skipped_pages(lambda q, n: docs.search(q, n)) == []  # documented: a lambda cannot be looked through


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
    assert classify("a sequence of steps with an abstract base class") == "other"  # not a bio or a text request


def test_a_project_without_a_tag_is_classified_from_its_request_text_only(tmp_path):
    p = Project.create(tmp_path, "old")
    p.board.post("request", "person", {"text": "Summarise these articles from OpenAlex"})
    (p.root / "task.txt").write_text("design PCR primers for a gene", encoding="utf-8")  # not what the request said
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
    # a QA repair is counted from the note's qa_repair field, not from its wording
    project.board.post("final", "qa", {"approved": False})
    project.board.post("note", "code", {"text": "final QA found problems: repairing I1"})
    tallies = {(r["stage"], r["measure"]): r["value"] for r in inspect.stage_tallies([project])}
    assert tallies[("final QA", "approved on first QA")] == "0/1 (0%)"
    assert tallies[("final QA", "projects that needed a QA repair")] == "0/1 (0%)"
    project.board.post("note", "code", {"qa_repair": True, "text": "whatever the wording"})
    tallies = {(r["stage"], r["measure"]): r["value"] for r in inspect.stage_tallies([project])}
    assert tallies[("final QA", "projects that needed a QA repair")] == "1/1 (100%)"


def test_time_lost_counts_cut_offs_and_failures():
    calls = [{"seconds": 10, "stop_reason": "end_turn"}, {"seconds": 30, "stop_reason": "max_tokens"},
             {"error": "boom"}, {"fallback_to": "x", "error": "404"}, {"cached": True, "seconds": 5},
             {"error": "503", "attempt": 0}]
    t = inspect.time_lost(calls)
    assert t == {"model_seconds": 40.0, "lost_seconds": 30.0, "lost_share": 0.75, "cut_off": 1, "refused": 0,
                 "failed_calls": 1, "retries": 1, "fallbacks": 1}


def test_a_budget_deferred_topic_is_recorded_once(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge_module, "MAX_RESEARCHED", 1)
    monkeypatch.setattr(knowledge_module, "MAX_ROUNDS", 1)
    kl, _, _, _ = knowledge(cfg, tmp_path, topics("Primer3 settings", "Salt correction"), queries(), review(1, 2),
                            draft(1, 2), check(True))
    kl.run()
    salt = [e for e in events_of(kl) if "topic:Salt correction" in e.refs]
    assert len(salt) == 1 and salt[0].message.startswith("topic deferred: research budget")


def test_a_topic_deferred_in_round_one_and_named_again_is_not_researched_or_recorded_twice(cfg, tmp_path,
                                                                                            monkeypatch):
    monkeypatch.setattr(knowledge_module, "MAX_RESEARCHED", 1)
    monkeypatch.setattr(knowledge_module, "MAX_ROUNDS", 2)
    kl, _, _, wiki = knowledge(cfg, tmp_path, topics("Primer3 settings", "Salt correction"), queries(), review(1, 2),
                               draft(1, 2), check(True), topics("Salt correction"))
    result = kl.run()
    assert result.deferred == ["Salt correction"] and wiki.get("Salt correction") is None  # budget spent in round one
    salt = [e for e in events_of(kl) if "topic:Salt correction" in e.refs]
    assert len(salt) == 1 and salt[0].message.startswith("topic deferred: research budget")
    cov = kl.board.latest("coverage").data
    assert cov["deferred"] == ["Salt correction"] and cov["missing"] == []  # named once, not once per round


def test_record_once_skips_a_repeat_while_the_first_is_open(tmp_path):
    log = EventLog(tmp_path / "r" / "events.jsonl", "r")
    first = log.record("checks", "2 failing test(s): a, b", stage="checks", once=True)
    assert first == "E0001"
    assert log.record("checks", "3 failing test(s): a, b", stage="checks", once=True) == ""  # same fingerprint
    assert log.record("checks", "2 failing test(s): a, b", stage="checks") == "E0002"  # without once= it is written
    log.resolve_open("a later check passed", source="checks")
    assert log.record("checks", "2 failing test(s): a, b", stage="checks", once=True) == "E0005"  # both corrected: new one
    # the fingerprint blanks paths and digits, so two files (or two tests) with the same message shape are told
    # apart by their refs, not by the message
    assert log.record("checks", "syntax error in src/a.py: bad", stage="checks", once=True, refs=["file:src/a.py"])
    assert log.record("checks", "syntax error in src/b.py: bad", stage="checks", once=True, refs=["file:src/b.py"])
    assert log.record("checks", "syntax error in src/b.py: bad", stage="checks", once=True, refs=["file:src/b.py"]) == ""
    assert log.record("checks", "1 failing test(s): tests/t.py::test_1", stage="checks", once=True, refs=["test:t1"])
    assert log.record("checks", "1 failing test(s): tests/t.py::test_2", stage="checks", once=True, refs=["test:t2"])
    assert len(log.open_events()) == 5


def test_log_of_with_a_project_sets_the_domain(tmp_path):
    class Fake:
        events = EventLog(tmp_path / "r" / "events.jsonl", "r")
    p = Project.create(tmp_path, "p")
    p.board.post("request", "person", {"text": "x", "domain": "code"})
    assert log_of(Fake(), p) is Fake.events and Fake.events.domain == "code"
    other = Project.create(tmp_path, "later")  # no request yet: the tag already set is kept, never blanked
    assert log_of(Fake(), other).domain == "code"
    assert log_of(object(), p).record("x", "y") == ""  # a stand-in without a log: still a NullLog


def test_a_verified_page_leaves_a_connector_fault_open(cfg, tmp_path):
    kl, _, _, _ = knowledge(cfg, tmp_path, topics("Primer3 settings"), queries(), review(1, 2), draft(1, 2),
                            check(True), topics(enough=True))
    kl.events.record("sources", "engine failed: ConnectError", stage="research", refs=["topic:Primer3 settings"])
    kl.events.record("knowledge", "topic still has no verified page", stage="research",
                     refs=["topic:Primer3 settings"])
    kl.run()
    by = {e.message[:14]: e.status for e in kl.events.events()}
    assert by["engine failed:"] == "open" and by["topic still ha"] == "corrected"


def test_a_check_that_ran_no_tests_corrects_only_syntax_errors(tmp_path):
    from types import SimpleNamespace

    from factory.pipeline.production import Checks, Production
    log = EventLog(tmp_path / "r" / "events.jsonl", "r")
    p = Project.create(tmp_path, "p")
    p.board.post("request", "person", {"text": "x", "domain": "code"})
    me = SimpleNamespace(llm=SimpleNamespace(events=log), project=p)
    Production._record_checks(me, Checks(compiled=["a.py"], syntax_errors={"a.py": "bad"}, tests_ran=True, tests_ok=False,
                                         failed=["test_a"]))
    assert [(e.message[:6], e.domain) for e in log.events()] == [("syntax", "code"), ("1 fail", "code")]
    Production._record_checks(me, Checks(compiled=["a.py"], syntax_errors={}, tests_ran=False, tests_ok=None))
    assert {e.message[:6]: e.status for e in log.events()} == {"syntax": "corrected", "1 fail": "open"}
    # syntax fixed while a test still fails: the syntax event is corrected, the test event stays, and is not doubled
    log2 = EventLog(tmp_path / "r2" / "events.jsonl", "r2")
    me2 = SimpleNamespace(llm=SimpleNamespace(events=log2), project=p)
    Production._record_checks(me2, Checks(compiled=["a.py"], syntax_errors={"a.py": "bad"}, tests_ran=True,
                                          tests_ok=False, failed=["test_a"]))
    Production._record_checks(me2, Checks(compiled=["a.py"], syntax_errors={}, tests_ran=True, tests_ok=False,
                                          failed=["test_a"]))
    assert [(e.message[:6], e.status) for e in log2.events()] == [("syntax", "corrected"), ("1 fail", "open")]
    # per file: a.py fixed while b.py still fails corrects a.py's event only
    log3 = EventLog(tmp_path / "r3" / "events.jsonl", "r3")
    me3 = SimpleNamespace(llm=SimpleNamespace(events=log3), project=p)
    Production._record_checks(me3, Checks(compiled=[], syntax_errors={"a.py": "bad", "b.py": "bad"}, tests_ran=False,
                                          tests_ok=None))
    Production._record_checks(me3, Checks(compiled=["a.py"], syntax_errors={"b.py": "bad"}, tests_ran=False,
                                          tests_ok=None))
    assert {e.message[16:20]: e.status for e in log3.events()} == {"a.py": "corrected", "b.py": "open"}
    Production._record_checks(me, Checks(compiled=["a.py"], syntax_errors={}, tests_ran=True, tests_ok=True))
    assert not log.open_events()


def test_parallel_records_get_distinct_ids(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    log = EventLog(tmp_path / "r" / "events.jsonl", "r")
    with ThreadPoolExecutor(8) as pool:
        ids = list(pool.map(lambda i: log.record("x", f"problem {i}"), range(40)))
    assert len(set(ids)) == 40 and len(log.events()) == 40


def test_a_changing_set_of_failing_tests_does_not_stack_events(tmp_path):
    from types import SimpleNamespace

    from factory.pipeline.production import Checks, Production
    log = EventLog(tmp_path / "r" / "events.jsonl", "r")
    p = Project.create(tmp_path, "p")
    me = SimpleNamespace(llm=SimpleNamespace(events=log), project=p)

    def run(*failed):
        Production._record_checks(me, Checks(compiled=["a.py"], syntax_errors={}, tests_ran=True, tests_ok=not failed,
                                             failed=list(failed)))
        return [(e.message[:1], e.status) for e in log.events()]

    assert run("test_a") == [("1", "open")]
    assert run("test_a") == [("1", "open")]  # the same failure again: nothing new
    assert run("test_a", "test_b") == [("1", "open"), ("2", "open")]  # test_b is new
    assert run("test_b") == [("1", "corrected"), ("2", "open")]  # test_a passes now; test_b is already known
    assert run() == [("1", "corrected"), ("2", "corrected")]
    assert [e.how for e in log.events()] == ["those tests pass in a later run", "a later test run passed"]


def test_a_reply_cut_off_at_max_tokens_is_a_warning_corrected_by_a_later_valid_reply(cfg, tmp_path):
    from pydantic import BaseModel

    from factory.llm import LLM
    from factory.llm.client import RawResult, Usage, complete_validated
    from tests.test_fallbacks import approve_all

    class Out(BaseModel):
        a: int

    class P:
        n = 0

        def complete(self, req):
            P.n += 1
            return RawResult('{"a": ', "", "max_tokens", Usage(10, 10)) if P.n == 1 else \
                RawResult('{"a": 1}', "", "end_turn", Usage(10, 10))

    llm = LLM(cfg, tmp_path, use_cache=False, providers={"arc": P()}, approver=approve_all)
    out, _, _ = complete_validated(llm, "coder", "write it", "", Out, tag="build-T1.1")
    (ev,) = llm.events.events()  # one event for the cut-off, from where the model answered
    assert out.a == 1 and ev.severity == "error" and "TruncatedError" in ev.message and "max_tokens" in ev.message
    assert ev.status == "corrected" and ev.how.startswith("a later attempt returned valid output")


def test_a_timeout_and_an_import_failure_are_check_events(tmp_path):
    from types import SimpleNamespace

    from factory.pipeline.production import Checks, Production
    log = EventLog(tmp_path / "r" / "events.jsonl", "r")
    me = SimpleNamespace(llm=SimpleNamespace(events=log), project=Project.create(tmp_path, "p"))
    failing = dict(compiled=["a.py"], syntax_errors={}, tests_ran=True, tests_ok=False)
    Production._record_checks(me, Checks(**failing), timed_out=True)
    Production._record_checks(me, Checks(**failing), timed_out=True)  # once: not stacked
    Production._record_checks(me, Checks(**failing, setup_error="No module named pkg"))
    Production._record_checks(me, Checks(**failing, output_tail="FAILED somewhere"))
    assert [e.message[:22] for e in log.events()] == ["the test run timed out", "tests could not be imp",
                                                      "pytest failed without "]
    Production._record_checks(me, Checks(compiled=["a.py"], syntax_errors={}, tests_ran=True, tests_ok=True))
    assert not log.open_events()


def test_a_gap_the_gate_allows_through_is_an_event(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge_module, "MAX_ROUNDS", 2)
    kl, _, _, _ = knowledge(cfg, tmp_path, topics("Primer3 settings"), queries(), review(1, 2), draft(1, 2),
                            check(False), draft(1, 2), check(False), queries(), topics("Primer3 settings"),
                            queries(), queries(), queries())
    result = kl.run()
    gaps = [e for e in events_of(kl) if e.message.startswith("build goes ahead with a gap")]
    assert result.ok and result.gaps and len(gaps) == 1 and gaps[0].stage == "knowledge gate"
    assert gaps[0].status == "open" and "topic:Primer3 settings" in gaps[0].refs
