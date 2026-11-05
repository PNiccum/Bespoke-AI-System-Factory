"""Wall-clock speedups: independent research topics and work orders run at once, a cut-off reply is retried on a
lighter model of the same family, projects can run side by side, and the profiler reads the call logs."""

import json
import re
import threading
import time

import pytest

from factory.bench_projects import ProjectOutcome
from factory.config import load_config
from factory.llm import LLM, complete_validated
from factory.llm.client import provider_gate
from factory.llm.cost import Usage
from factory.llm.gate import approve_all
from factory.llm.providers import RawResult
from factory.pipeline.knowledge import KnowledgeLoop, Topic
from factory.pipeline.production import Design, Production, WorkOrder, order_waves
from factory.pipeline.wiki import Wiki
from factory.profile import profile, report
from factory.registry import Registry
from tests.test_build import project_after_knowledge
from tests.test_knowledge import Engine, check, doc, planned, review
from tests.test_orchestrator import ROOT, cfg  # noqa: F401


class Concurrent:
    """A provider that answers by looking at the prompt, takes a moment per call, and records the most calls it
    saw in flight at once."""

    def __init__(self, answer, delay=0.2):
        self.answer, self.delay = answer, delay
        self.requests, self.in_flight, self.most = [], 0, 0
        self._lock = threading.Lock()

    def complete(self, req):
        with self._lock:
            self.requests.append(req)
            self.in_flight += 1
            self.most = max(self.most, self.in_flight)
        try:
            time.sleep(self.delay)
            return RawResult(self.answer(req.messages[-1]["content"]), "", "end_turn", Usage(500, 200))
        finally:
            with self._lock:
                self.in_flight -= 1


def W(id, write, read=()):
    return WorkOrder(id=id, write=list(write), read=list(read), instructions="x")


def ids(waves):
    return [[o.id for o in w] for w in waves]


def test_order_waves_keep_every_dependency_and_run_the_rest_together():
    assert ids(order_waves([W("a", ["x.py"]), W("b", ["y.py"]), W("c", ["t.py"], read=["x.py"])])) == [["a", "b"], ["c"]]
    # rewriting a file an earlier order reads waits for that read; output/ prefixes name the same file
    assert ids(order_waves([W("a", ["t.py"], read=["x.py"]), W("b", ["output/x.py"])])) == [["a"], ["b"]]
    assert ids(order_waves([W("a", ["x.py"]), W("b", ["x.py"])])) == [["a"], ["b"]]
    assert ids(order_waves([])) == []


def test_independent_work_orders_are_built_at_once(cfg, tmp_path):
    llm, project, wiki = project_after_knowledge(cfg, tmp_path)

    def answer(prompt):
        path = re.search(r'"write": \[\s*"([^"]+)"', prompt).group(1)
        return f"=== FILE: {path} ===\nX = 1\n=== END FILE ===\nNOTES: ok"

    arc = Concurrent(answer)
    llm.providers["arc"] = arc
    production = Production(llm, Registry.load(cfg, ROOT / "registry"), project, wiki)
    d = Design(language="python", entry_point="a.py", run_command="python a.py", layout=["a.py"])
    wave = [W("T1.1", ["a.py"]), W("T1.2", ["b.py"]), W("T1.3", ["c.py"])]
    production.build_wave(wave, "task", d, "(none)", [], "ref")
    assert arc.most == 3 and sorted(production.files()) == ["a.py", "b.py", "c.py"]


def test_topics_are_researched_at_once_and_results_keep_their_order(cfg, tmp_path):
    llm, project = planned(cfg, tmp_path)

    def answer(prompt):
        if prompt.startswith("Write searches"):
            return json.dumps({"queries": [{"connector": "europepmc", "query": "primer"}]})
        if prompt.startswith("Approve the documents"):
            return review(1, 2)
        if prompt.startswith("Write a wiki page"):
            q = "Primers should have a Tm of 55-65 C"
            return json.dumps({"summary": "s", "body": f"Tm is 55-65 C. {{{{q:europepmc:PMC1|{q} (1)}}}} "
                                                       f"Again. {{{{q:europepmc:PMC2|{q} (2)}}}}"})
        return check(True)

    arc = Concurrent(answer)
    llm.providers["arc"] = arc
    kl = KnowledgeLoop(llm, Registry.load(cfg, ROOT / "registry"), project, Wiki(tmp_path / "wiki"),
                       {"europepmc": Engine([doc(1), doc(2)])})
    names = ["Primer melting temperature", "RefSeq FASTA headers", "MANE Select transcripts"]
    pages = kl.research_many([Topic(name=n, why="needed") for n in names])
    assert [p.topic for p in pages] == names and all(p.verified for p in pages)
    assert arc.most >= 3


def test_a_cut_off_reply_is_retried_on_the_roles_lighter_model(cfg, tmp_path):
    from pydantic import BaseModel

    class Out(BaseModel):
        a: int

    seen = []

    class P:
        def complete(self, req):
            seen.append(req.model)
            if len(seen) == 1:
                return RawResult('{"a": ', "", "max_tokens", Usage(10, 10))
            return RawResult('{"a": 1}', "", "end_turn", Usage(10, 10))

    llm = LLM(cfg, tmp_path, use_cache=False, providers={"arc": P()}, approver=approve_all)
    out, _, errors = complete_validated(llm, "coder", "write it", "", Out, tag="build-T1.1")
    assert out.a == 1 and len(errors) == 1
    assert seen == ["GLM-5.3-thinking-high", "GLM-5.3"]  # same family, no thinking budget to overrun
    assert cfg.model_info(seen[0]).family == cfg.model_info(seen[1]).family


def test_every_lighter_model_is_a_free_fallback_of_its_role(cfg):
    for name, role in cfg.roles.items():
        if role.on_truncation:
            assert role.on_truncation in role.fallbacks, name
            assert not cfg.model_info(role.on_truncation).real, name
    assert cfg.roles["plan_critic"].on_truncation == "DeepSeek-V4.1-Flash-thinking-low"


def test_a_lighter_model_outside_the_fallbacks_is_refused(tmp_path):
    import yaml
    data = yaml.safe_load((ROOT / "configs" / "factory.yaml").read_text())
    data["roles"]["coder"]["on_truncation"] = "DeepSeek-V4.1-Flash"  # another family: could break separation
    data["models_file"] = str(ROOT / "configs" / "models.yaml")
    path = tmp_path / "configs" / "factory.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="must be one of its fallbacks"):
        load_config(path)


def test_runs_share_one_concurrency_gate_per_provider(cfg, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = LLM(cfg, tmp_path / "a", use_cache=False, providers={"arc": None})
    b = LLM(cfg, tmp_path / "b", use_cache=False, providers={"arc": None})
    assert a._gates["arc"] is b._gates["arc"] is provider_gate("arc", 99)


def test_bench_projects_run_side_by_side_and_report_in_task_order(tmp_path, monkeypatch):
    import factory.bench_projects as bp

    running, most, lock = [0], [0], threading.Lock()

    def fake_run(cfg, task, *, approver, redesign, trial, resume=None):
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        time.sleep(0.2 if task.name == "a" else 0.05)
        with lock:
            running[0] -= 1
        return ProjectOutcome(project=task.name, trial=trial, reached="build", project_dir="x")

    class T:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(bp, "run_project", fake_run)
    monkeypatch.setattr(bp, "load_projects", lambda path: [T("a"), T("b"), T("c")])
    monkeypatch.setattr(bp, "load_config", lambda path: None)
    assert bp.main(["--parallel", "3", "--out", str(tmp_path)]) == 0
    rows = json.loads(next(tmp_path.glob("*-projects/results.json")).read_text())
    assert most[0] == 3 and [r["project"] for r in rows] == ["a", "b", "c"]


def test_profile_finds_serial_calls_and_time_lost_to_cut_off_replies(tmp_path):
    run = tmp_path / "20261104T000000Z-build"
    run.mkdir()
    rows = [
        {"at": "2026-11-04T00:01:40+00:00", "tag": "build-T1.1", "model": "GLM-5.3-thinking-high", "seconds": 100,
         "stop_reason": "max_tokens", "output_tokens": 4000},
        {"at": "2026-11-04T00:02:30+00:00", "tag": "build-T1.1", "model": "GLM-5.3", "seconds": 50,
         "stop_reason": "end_turn", "output_tokens": 1000},
        {"at": "2026-11-04T00:02:31+00:00", "tag": "verify-x-1", "model": "DeepSeek-V4.1-Flash-thinking-max",
         "error": "APIConnectionError: Connection error.", "attempt": 0},
    ]
    (run / "llm_calls.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    p = profile([tmp_path])
    assert p["runs"][0]["in_flight"] == 1.0 and p["runs"][0]["wall_s"] == 150
    assert p["cut_off_s"] == 100 and p["stages"][0]["stage"] == "build"
    assert p["errors"] == {"APIConnectionError": 1}
    assert "67% of model time" in report(p)


def test_the_wiki_verifier_starts_on_the_low_thinking_budget_of_the_critics_family(cfg):
    from pathlib import Path

    from factory.registry import Registry
    role = cfg.roles["wiki_verifier"]
    assert role.model == "DeepSeek-V4.1-Flash-thinking-low"
    assert "DeepSeek-V4.1-Flash-thinking-max" in role.fallbacks  # the slow budget stays as a fallback
    assert cfg.model_info(role.model).family == cfg.model_info(cfg.roles["plan_critic"].model).family
    assert cfg.model_info(role.model).family != cfg.model_info(cfg.roles["planner"].model).family  # not the librarian's
    assert Registry.load(cfg, Path(__file__).resolve().parents[1] / "registry").agents["wiki_verifier"].role == "wiki_verifier"


def test_a_work_order_writing_many_files_is_split_so_one_reply_stays_short():
    from factory.pipeline.production import MAX_ORDER_FILES, split_orders
    big = W("T1.1", [f"f{i}.py" for i in range(7)], read=["base.py"])
    parts = split_orders([W("T1.0", ["one.py"]), big])
    assert [o.id for o in parts] == ["T1.0", "T1.1a", "T1.1b", "T1.1c"]
    assert all(len(o.write) <= MAX_ORDER_FILES for o in parts)
    assert [w for o in parts[1:] for w in o.write] == big.write  # every file, once, in order
    assert parts[3].read == ["base.py", *big.write[:6]]  # a later part reads the earlier parts' files
    assert ids(order_waves(parts)) == [["T1.0", "T1.1a"], ["T1.1b"], ["T1.1c"]]  # so the waves keep them in order


def _rescue_llm(cfg, tmp_path, approver, free_replies=3):
    from tests.test_llm_client import FakeProvider, ok
    bad = ok('{"a": "no"}')
    providers = {"arc": FakeProvider([bad] * free_replies), "openai": FakeProvider([ok('{"a": 7}')])}
    return LLM(cfg, tmp_path, use_cache=False, providers=providers, approver=approver), providers


def test_after_the_free_attempts_fail_one_paid_attempt_goes_through_the_spend_gate(tmp_path):
    from pydantic import BaseModel

    from factory.llm.gate import approve_all
    from tests.helpers import paid_config

    class Out(BaseModel):
        a: int

    cfg = paid_config(tmp_path)
    asked = []
    llm, p = _rescue_llm(cfg, tmp_path, lambda req: asked.append(req) or True)
    out, c, errors = complete_validated(llm, "grader", "grade it", "", Out, tag="eval", data="public")
    assert out.a == 7 and c.model == "gpt-6.1-sol" and len(errors) == 3
    assert p["arc"].calls == 3 and p["openai"].calls == 1
    assert asked and asked[0].role == "grader" and asked[0].model == "gpt-6.1-sol"  # a person saw the estimate

    # declined: no paid call, the usual error, and the reason is recorded
    llm, p = _rescue_llm(cfg, tmp_path, lambda req: False)
    with pytest.raises(RuntimeError, match="paid rescue skipped"):
        complete_validated(llm, "grader", "grade it", "", Out, tag="eval", data="public")
    assert p["openai"].calls == 0

    # private data never goes to a paid provider, whatever the approver says
    llm, p = _rescue_llm(cfg, tmp_path, approve_all)
    with pytest.raises(RuntimeError, match="no valid output"):
        complete_validated(llm, "grader", "grade it", "", Out, tag="eval", data="private")
    assert p["openai"].calls == 0

    # a role with no paid step, or a caller that turned the rescue off, behaves as before
    llm, p = _rescue_llm(cfg, tmp_path, approve_all)
    with pytest.raises(RuntimeError, match="no valid output"):
        complete_validated(llm, "grader", "grade it", "", Out, tag="eval", data="public", rescue=False)
    assert p["openai"].calls == 0
