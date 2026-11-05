"""The event log and the historian: code decides what happened and what was corrected; the model only explains."""

import json

from factory.events import EventLog, fingerprint, log_of, meta_analysis, run_events, summary
from factory.historian import write_meta_history, write_run_history
from factory.llm import LLM, Usage
from factory.llm.providers import RawResult, TransientHTTPError
from tests.test_fallbacks import ByModel, cfg, llm_with, not_found  # noqa: F401


def test_an_event_stays_open_until_code_corrects_it(tmp_path):
    log = EventLog(tmp_path / "events.jsonl", "r1")
    a = log.record("pipeline", "production failed", stage="production")
    b = log.record("qa", "QA said no", severity="warning")
    assert [e.status for e in log.events()] == ["open", "open"]
    assert log.resolve(a, "rebuilt in round 2", ["p1"]) == 1
    assert log.resolve(a, "again") == 0  # already corrected
    ev = {e.id: e for e in log.events()}
    assert ev[a].status == "corrected" and ev[a].how == "rebuilt in round 2" and ev[b].status == "open"
    assert [e.id for e in log.open_events()] == [b]


def test_capture_records_and_reraises(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    try:
        with log.capture("pipeline", "build"):
            raise KeyError("boom")
    except KeyError:
        pass
    assert "KeyError" in log.events()[0].message and log.events()[0].status == "open"


def test_fingerprint_ignores_numbers_and_paths():
    assert fingerprint("llm", "s", "timeout after 30 s at /tmp/a/b") == fingerprint("llm", "s", "timeout after 90 s at /x/y")
    assert fingerprint("llm", "s", "timeout") != fingerprint("llm", "s", "refused")


def test_a_truncated_log_line_is_skipped(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.record("code", "ok")
    with log.path.open("a") as f:
        f.write('{"type": "event", "id"')
    assert len(log.events()) == 1


def test_a_retry_that_succeeds_corrects_the_transient_error(cfg, tmp_path):
    cfg.providers["arc"].stream_retries = 1
    import factory.llm.client as client
    client.time.sleep = lambda s: None

    class Flaky(ByModel):
        def complete(self, req):
            self.models.append(req.model)
            if len(self.models) == 1:
                raise TransientHTTPError("503: overloaded")
            return RawResult("ok", "", "end_turn", Usage(1, 1))
    llm_with(cfg, tmp_path, Flaky({})).complete("coder", "hi", tag="t")
    (e,) = EventLog(tmp_path / "events.jsonl").events()
    assert e.status == "corrected" and "retry 1 succeeded" in e.how and e.severity == "warning"


def test_a_fallback_model_that_answers_corrects_the_missing_model(cfg, tmp_path):
    arc = ByModel({"GLM-5.3-thinking-high": not_found("GLM-5.3-thinking-high")})
    llm_with(cfg, tmp_path, arc).complete("coder", "fix it")
    (e,) = EventLog(tmp_path / "events.jsonl").events()
    assert e.status == "corrected" and "GLM-5.3 answered" in e.how


def test_a_failure_nothing_recovers_stays_open(cfg, tmp_path):
    arc = ByModel({m: not_found(m) for m in ("GLM-5.3-thinking-high", "GLM-5.3", "gpt-oss-120b-thinking-high")})
    try:
        llm_with(cfg, tmp_path, arc).complete("coder", "fix it")
    except Exception:
        pass
    events = EventLog(tmp_path / "events.jsonl").events()
    assert events and events[-1].status == "open" and events[-1].severity == "error"


def test_log_of_an_llm_stand_in_drops_events():
    class Bare:
        pass
    assert log_of(Bare()).record("x", "y") == ""


def make_runs(base):
    for name, fixed in (("20261101T000000Z-a", True), ("20261102T000000Z-b", False)):
        log = EventLog(base / name / "events.jsonl", name)
        e = log.record("pipeline", "final QA failed: 3 tests", stage="final QA")
        if fixed:
            log.resolve(e, "delivered after 1 QA repair(s)")
        log.record("llm", "timeout", stage="x", severity="warning")


def test_meta_analysis_groups_across_runs_and_flags_a_fix_that_did_not_hold(tmp_path):
    make_runs(tmp_path)
    events = run_events(tmp_path)
    assert summary(events) == {"events": 4, "open": 3, "corrected": 1, "runs": 2}
    g = meta_analysis(events)
    qa = next(x for x in g if x.stage == "final QA")
    assert qa.count == 2 and qa.corrected == 1 and qa.open == 1 and qa.returned_after_fix
    assert not next(x for x in g if x.stage == "x").returned_after_fix


class Scripted:
    def __init__(self, text):
        self.text = text

    def complete(self, req):
        return RawResult(self.text, "", "end_turn", Usage(10, 10))


def test_the_historian_cannot_invent_events_or_mark_things_fixed(cfg, tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    log = EventLog(run / "events.jsonl", "run1")
    a = log.record("pipeline", "final QA failed", stage="final QA")
    b = log.record("pipeline", "solver crashed", stage="critic loop")
    log.resolve(a, "delivered after 1 QA repair(s)")
    reply = json.dumps({"summary": "Two things went wrong.", "entries": [
        {"ids": [a], "title": "QA rejected", "what_happened": "QA found problems.", "how_it_was_fixed": "A repair worked."},
        {"ids": [b, "E9999"], "title": "Crash", "what_happened": "The solver crashed.", "how_it_was_fixed": "Fixed itself."},
        {"ids": ["E9999"], "title": "Invented", "what_happened": "never happened"}]})
    llm = LLM(cfg, tmp_path, use_cache=False, providers={"arc": Scripted(reply)})
    text = write_run_history(llm, run).read_text()
    assert "1 corrected" in text and "1 not corrected" in text
    assert "A repair worked." in text and "Corrected by (recorded by code)" in text
    assert "Fixed itself." not in text  # that event is open in the log, so no fix is told
    assert "Invented" not in text and "E9999" not in text.split("## Every event")[0].split("Crash")[0]
    assert "not corrected; E0002" in text


def test_the_report_is_still_written_when_the_model_gives_nothing_usable(cfg, tmp_path):
    run = tmp_path / "run1"
    EventLog(run / "events.jsonl", "run1").record("code", "something broke")
    llm = LLM(cfg, tmp_path, use_cache=False, providers={"arc": Scripted("no json here")})
    text = write_run_history(llm, run).read_text()
    assert "something broke" in text and "**open**" in text


def test_a_run_without_errors_has_no_history(cfg, tmp_path):
    (tmp_path / "run1").mkdir()
    llm = LLM(cfg, tmp_path, use_cache=False, providers={"arc": Scripted("{}")})
    assert write_run_history(llm, tmp_path / "run1") is None


def test_the_meta_analysis_drops_unknown_fingerprints(cfg, tmp_path):
    runs = tmp_path / "runs"
    make_runs(runs)
    known = meta_analysis(run_events(runs))[0].fingerprint
    reply = json.dumps({"summary": "A pattern.", "patterns": [
        {"fingerprints": [known], "finding": "QA keeps failing", "suggestion": "look at QA"},
        {"fingerprints": ["deadbeef0000"], "finding": "made up"}]})
    (tmp_path / "h").mkdir()
    llm = LLM(cfg, tmp_path / "h", use_cache=False, providers={"arc": Scripted(reply)})
    text = write_meta_history(llm, runs, tmp_path / "out" / "meta.md").read_text()
    assert "QA keeps failing" in text and "made up" not in text and "Corrected once, then came back" in text
