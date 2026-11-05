"""The agent loop: multi-step tool use, typed inputs, allow-lists, caps, decision records, replay."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from factory.agents.loop import replay, run_agent
from factory.agents.tools import Toolbox, ToolError, tool
from factory.config import AgentCaps
from factory.llm import LLM, Usage
from factory.llm.gate import approve_all
from factory.llm.providers import RawResult, to_openai_messages
from tests.helpers import paid_config

ROOT = Path(__file__).resolve().parents[1]


class Scripted:
    """A fake provider that plays back turns and records the requests it saw."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def complete(self, req):
        self.requests.append(req)
        return self.turns.pop(0)


def turn(text="", calls=(), inp=100, out=50):
    calls = [{"id": f"t{i}", "name": n, "input": a} for i, (n, a) in enumerate(calls)]
    return RawResult(text, "", "tool_use" if calls else "end_turn", Usage(inp, out), calls)


@pytest.fixture
def cfg(tmp_path):
    cfg = paid_config(tmp_path)  # agents are tested on a Claude driver
    cfg.root = tmp_path
    return cfg


def make(cfg, tmp_path, turns):
    p = Scripted(turns)
    return LLM(cfg, tmp_path, use_cache=False, providers={"claude": p, "arc": p}, approver=approve_all), p


NOTES = {"n1": "Retry with backoff cut failed uploads by 40% in the staging service.",
         "n2": "The cache layer serves repeated requests without calling the model."}


class SearchIn(BaseModel):
    query: str
    k: int = 5


class ReadIn(BaseModel):
    ref: str


class CheckIn(BaseModel):
    statement: str
    ref: str
    quote: str


def note_tools(notes: dict[str, str]) -> Toolbox:
    """A small toolbox over a dict of notes: search by shared words, read one, check a quote is verbatim."""

    @tool("search", "Find notes sharing words with the query.", SearchIn)
    def search(inp):
        words = set(inp.query.lower().split())
        hits = sorted(notes, key=lambda r: -len(words & set(notes[r].lower().split())))
        return [{"ref": r, "preview": notes[r][:60]} for r in hits[:inp.k] if words & set(notes[r].lower().split())]

    @tool("read", "Read one note in full.", ReadIn)
    def read(inp):
        if inp.ref not in notes:
            raise ToolError(f"no note {inp.ref!r}")
        return {"ref": inp.ref, "text": notes[inp.ref]}

    @tool("check_statement", "Check that a quote is verbatim in a note.", CheckIn)
    def check_statement(inp):
        return {"pass": inp.quote in notes.get(inp.ref, "")}

    return Toolbox([search, read, check_statement])


def test_multi_step_tool_use_reaches_an_answer_and_logs_each_decision(cfg, tmp_path):
    turns = [turn("Searching first.", [("search", {"query": "failed uploads retry"})]),
             turn("Reading the hit.", [("read", {"ref": "n1"})]),
             turn("Checking my statement.", [("check_statement", {
                 "statement": "Retries cut failed uploads by 40%.",
                 "ref": "n1", "quote": "Retry with backoff cut failed uploads by 40%"})]),
             turn("Done: one eligible statement.")]
    llm, p = make(cfg, tmp_path, turns)
    r = run_agent(llm, "agent", "You are a test agent.", "Find one fact.", note_tools(NOTES), data="public")
    assert r.status == "done" and r.steps == 4 and r.tool_calls == 3
    assert "Done" in r.final_text
    # each turn sends the whole history, with tool results paired to their calls
    last = p.requests[-1].messages
    assert last[-1]["content"][0]["type"] == "tool_result" and last[-1]["content"][0]["tool_use_id"] == "t0"
    assert {t["name"] for t in p.requests[0].tools} == {"search", "read", "check_statement"}
    log = [json.loads(l) for l in r.log_path.read_text().splitlines()]
    tools_used = [(e["tool"], e["ok"], e["reason"]) for e in log if e["event"] == "tool"]
    assert tools_used[0] == ("search", True, "Searching first.")
    assert '"pass": true' in [e for e in log if e["event"] == "tool"][2]["result"]
    assert log[-1]["event"] == "stop" and log[-1]["status"] == "done"


def test_errors_go_back_to_the_agent_not_up_the_stack(cfg, tmp_path):
    turns = [turn("", [("read", {"ref": "nope"}), ("search", {"k": "lots"}), ("delete_tests", {})]),
             turn("Giving up.")]
    llm, p = make(cfg, tmp_path, turns)
    r = run_agent(llm, "agent", "s", "t", note_tools(NOTES), data="public")
    results = p.requests[1].messages[-1]["content"]
    assert all(x.get("is_error") for x in results)
    assert "no note" in results[0]["content"]
    assert "invalid arguments" in results[1]["content"]
    assert "not available to this agent" in results[2]["content"]
    assert r.status == "done"


def test_caps_stop_the_run_with_a_report(cfg, tmp_path):
    loop_forever = [turn("again", [("search", {"query": "x"})]) for _ in range(10)]
    llm, _ = make(cfg, tmp_path, loop_forever)
    r = run_agent(llm, "agent", "s", "t", note_tools(NOTES), AgentCaps(max_steps=3), data="public")
    assert r.status == "cap:steps" and r.steps == 3 and "stopped at a cap (steps)" in r.report()

    llm, _ = make(cfg, tmp_path / "b", loop_forever) if (tmp_path / "b").mkdir() is None else None
    r = run_agent(llm, "agent", "s", "t", note_tools(NOTES), AgentCaps(max_tokens=250), data="public")
    assert r.status == "cap:tokens" and r.steps == 2  # 150 tokens per turn


def test_allow_list_is_the_toolbox():
    box = note_tools(NOTES).subset(["search"])
    assert box.call("read", {"ref": "n1"}).ok is False
    with pytest.raises(KeyError):
        box.subset(["read"])


def test_tool_schema_comes_from_the_input_model():
    class In(BaseModel):
        n: int

    @tool("double", "Double a number.", In)
    def double(inp):
        if inp.n < 0:
            raise ToolError("negative")
        return {"n": inp.n * 2}

    box = Toolbox([double])
    assert box.schemas()[0]["input_schema"]["properties"]["n"]["type"] == "integer"
    assert box.call("double", {"n": 2}).content == '{"n": 4}'
    assert box.call("double", {"n": -1}).content == "negative"


def test_replay_reproduces_tool_steps_without_the_model(cfg, tmp_path):
    turns = [turn("", [("search", {"query": "cache model"}), ("read", {"ref": "n2"})]), turn("ok")]
    llm, _ = make(cfg, tmp_path, turns)
    r = run_agent(llm, "agent", "s", "t", note_tools(NOTES), data="public")
    assert replay(r.log_path, note_tools(NOTES)) == [("search", True, True), ("read", True, True)]
    changed = {**NOTES, "n2": "The cache layer was removed."}
    assert replay(r.log_path, note_tools(changed))[1] == ("read", True, False)  # drift is visible


def test_openai_translation_of_tool_history():
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "text", "text": "hm"},
                                              {"type": "tool_use", "id": "a", "name": "search", "input": {"q": 1}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "[]"}]}]
    out = to_openai_messages(msgs)
    assert out[1]["tool_calls"][0]["function"] == {"name": "search", "arguments": '{"q": 1}'}
    assert out[2] == {"role": "tool", "tool_call_id": "a", "content": "[]"}


def test_compaction_stubs_old_results_and_keeps_references(cfg, tmp_path):
    turns = [turn("", [("search", {"query": "failed uploads retry"})]) for i in range(6)] + [turn("done")]
    llm, p = make(cfg, tmp_path, turns)
    r = run_agent(llm, "agent", "s", "t", note_tools(NOTES), AgentCaps(compact_every=3, keep_recent=2), data="public")
    assert r.status == "done"
    contents = [b["content"] for m in r.messages if m["role"] == "user" and isinstance(m["content"], list)
                for b in m["content"]]
    assert contents[0].startswith("[earlier result of search") and "references: n1" in contents[0]
    assert not contents[-1].startswith("[earlier result")  # recent results stay whole
    log = [json.loads(l) for l in r.log_path.read_text().splitlines()]
    assert [e["step"] for e in log if e["event"] == "compact"] == [3, 6]
    # between compactions the history only grows: each request starts with the previous one (cacheable prefix)
    a, b = p.requests[3].messages, p.requests[4].messages
    assert b[:len(a)] == a


def test_cache_breakpoint_marks_only_the_newest_block():
    from factory.llm.providers import with_cache_breakpoint
    msgs = [{"role": "user", "content": "task"}, {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "r"}]}]
    out = with_cache_breakpoint(msgs)
    assert out[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in msgs[-1]["content"][-1]  # the caller's history is not modified
    assert with_cache_breakpoint([{"role": "user", "content": "hi"}])[0]["content"][0]["text"] == "hi"


def test_run_agent_treats_data_as_private_unless_the_caller_says_public(cfg, tmp_path):
    """Fail closed: a caller that forgets ``data=`` must not be able to send lab data to a public-only provider."""
    from factory.llm.client import DataScopeError

    llm, p = make(cfg, tmp_path, [turn("Done.")])
    with pytest.raises(DataScopeError):
        run_agent(llm, "agent", "s", "t", note_tools(NOTES))
    assert run_agent(llm, "agent", "s", "t", note_tools(NOTES), data="public").status == "done"
