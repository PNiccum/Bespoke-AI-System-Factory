"""The repair library (QA repairs that stuck, shown to the solver) and the per-stage model tally."""

import json

from factory.llm.gate import deny_all
from factory.pipeline.run import build
from factory.registry import Registry
from factory.repairs import RepairLibrary, diff_of
from factory.tally import build_tally, stage_of, write_tally
from tests.test_build import CODE, PRODUCE, WRONG, fixes, project_after_knowledge, qa, review
from tests.test_orchestrator import ROOT, Scripted, cfg  # noqa: F401


def lib(tmp_path):
    return RepairLibrary(tmp_path / "repairs")


def test_a_repair_counts_only_once_it_is_confirmed(tmp_path):
    lb = lib(tmp_path)
    rid = lb.add("final QA says task T2 is not met: salt is ignored", "-a\n+b", explanation="add salt")
    assert lb.closest(["task T2 salt ignored"]) == []  # pending: not shown
    assert lb.confirm([rid], "delivered in round 1") == 1
    (r,) = lb.closest(["task T2 salt ignored"])
    assert r.status == "stuck" and r.evidence == "delivered in round 1" and r.fingerprint


def test_an_empty_diff_is_not_saved_and_discard_removes_a_pending_repair(tmp_path):
    lb = lib(tmp_path)
    assert lb.add("x", "  ") == ""
    rid = lb.add("some finding", "-a\n+b")
    lb.discard([rid])
    assert lb.all() == []


def test_closest_ranks_by_overlap_and_ignores_unrelated_repairs(tmp_path):
    lb = lib(tmp_path)
    ids = [lb.add("csv header row is counted as data", "-a\n+b"), lb.add("tm ignores salt concentration", "-a\n+b")]
    lb.confirm(ids, "ok")
    assert [r.issue for r in lb.closest(["Tm ignores the salt concentration"])] == ["tm ignores salt concentration"]
    assert lb.closest(["completely different matter"]) == []
    assert lb.section(["unrelated words only"]) == ""


def test_a_newer_repair_of_the_same_problem_replaces_the_older_one(tmp_path):
    lb = lib(tmp_path)
    a = lb.add("tm ignores salt 12", "-a\n+b")
    lb.confirm([a], "ok")
    b = lb.add("tm ignores salt 99", "-a\n+c")  # numbers are blanked in the fingerprint: the same problem
    lb.confirm([b], "ok")
    assert [r.id for r in lb.stuck()] == [b]


def test_diff_of_covers_new_and_changed_files_only():
    d = diff_of({"a.py": "x\n", "b.py": "y\n"}, {"a.py": "x\n", "b.py": "z\n", "c.py": "n\n"}, ["a.py", "b.py", "c.py"])
    assert "b/b.py" in d and "b/c.py" in d and "a/a.py" not in d


def test_a_stuck_repair_is_saved_then_shown_to_the_solver_next_time(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path, approver=deny_all)
    llm.providers["arc"] = Scripted(*PRODUCE, review(True), qa(("T1", True), ("T2", False)),
                                    fixes("QT2", content=CODE + "# salt-corrected\n"), qa(("T1", True), ("T2", True)))
    result = build(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {}, redesign=False)
    assert result.status == "delivered" and result.repairs == 1
    (saved,) = RepairLibrary.of(cfg).stuck()
    assert "task T2 is not met" in saved.issue and "+# salt-corrected" in saved.diff and saved.files == ["primer.py"]

    # a second build with a similar QA finding: the solver's prompt carries the saved repair
    (tmp_path / "again").mkdir()
    llm2, project2, wiki2 = project_after_knowledge(cfg, tmp_path / "again", approver=deny_all)
    arc = Scripted(*PRODUCE, review(True), qa(("T1", True), ("T2", False)),
                   fixes("QT2", content=CODE + "# salt-2\n"), qa(("T1", True), ("T2", True)))
    llm2.providers["arc"] = arc
    build(llm2, Registry.load(cfg, ROOT / "registry"), project2, wiki2, {}, redesign=False)
    solve = next(r for r in arc.requests if "Fix these issues" in json.dumps(r.messages))
    assert "Past repairs of similar findings" in json.dumps(solve.messages) and "salt-corrected" in json.dumps(solve.messages)


def test_a_repair_that_does_not_end_in_approval_is_not_kept(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path, approver=deny_all)
    bad = json.dumps({"fixes": [{"issue": "QT2", "explanation": "x", "files": [
        {"path": "tests/test_primer.py", "content": WRONG}]}]})
    llm.providers["arc"] = Scripted(*PRODUCE, review(True), qa(("T1", True), ("T2", False)), bad,
                                    *PRODUCE, review(True), qa(("T1", True), ("T2", True)))
    build(llm, Registry.load(cfg, ROOT / "registry"), project, wiki, {}, redesign=False)
    assert RepairLibrary.of(cfg).all() == []  # the broken repair was put back and never saved


def _calls(run, *rows):
    run.mkdir(parents=True)
    (run / "llm_calls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_stage_names_drop_rounds_and_retry_suffixes():
    assert stage_of("solve-2-first", "coder") == stage_of("solve-3", "coder") == "solve"
    assert stage_of("", "coder") == "coder"


def test_the_tally_counts_each_model_and_each_handoff(tmp_path):
    ok = {"role": "coder", "stop_reason": "end_turn", "seconds": 4, "input_tokens": 10, "output_tokens": 5}
    _calls(tmp_path / "runs" / "r1",
           {**ok, "tag": "solve-1", "model": "A", "fallback_to": "B", "error": "404"},
           {**ok, "tag": "solve-1", "model": "B"},
           {**ok, "tag": "solve-2", "model": "A", "stop_reason": "max_tokens"},
           {**ok, "tag": "solve-2", "model": "B"},
           {**ok, "tag": "solve-3", "model": "A", "cached": True})  # cached: not counted
    tallies, hand = build_tally(tmp_path / "runs")
    (t,) = tallies
    a, b = {m.model: m for m in t.models}["A"], {m.model: m for m in t.models}["B"]
    assert (a.calls, a.failed_over, a.cut_off, a.finished) == (2, 1, 1, 0) and (b.calls, b.finished) == (2, 2)
    assert {(h.reason, h.from_model, h.to_model, h.count, h.to_finished) for h in hand} == {
        ("unavailable", "A", "B", 1, 1), ("cut off", "A", "B", 1, 1)}
    assert "not enough data" in t.verdict


def test_a_model_is_called_ahead_only_with_enough_calls_on_both(tmp_path):
    rows = [{"role": "coder", "tag": f"s-{i}", "model": m, "stop_reason": "end_turn" if ok else "max_tokens",
             "seconds": 1} for m, ok in [("A", True)] * 6 + [("B", True)] * 3 + [("B", False)] * 3 for i in [0]]
    rows = [{**r, "tag": f"s-{n}"} for n, r in enumerate(rows)]
    _calls(tmp_path / "runs" / "r1", *rows)
    (t,) = build_tally(tmp_path / "runs")[0]
    assert "A finishes more often" in t.verdict
    path = write_tally(tmp_path / "runs", tmp_path / "out")
    assert path and "A finishes more often" in path.read_text() and (tmp_path / "out" / "model_tally.json").exists()
    assert write_tally(tmp_path / "none", tmp_path / "out2") is None
