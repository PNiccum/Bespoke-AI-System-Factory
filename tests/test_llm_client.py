"""Client behaviour that every pipeline step relies on, tested with a fake provider."""

import json
from pathlib import Path

import pytest

from factory.config import Budget
from factory.llm import LLM, BudgetExceeded, DataScopeError, RefusedError, TruncatedError, Usage, price_call
from factory.llm.gate import approve_all
from factory.llm.providers import RawResult
from tests.helpers import paid_config

ROOT = Path(__file__).resolve().parents[1]


class FakeProvider:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def complete(self, req):
        self.calls += 1
        return self.results.pop(0)


def ok(text="hello", inp=1000, out=500):
    return RawResult(text, "", "end_turn", Usage(input_tokens=inp, output_tokens=out))


@pytest.fixture
def cfg(tmp_path):
    cfg = paid_config(tmp_path)  # the client's Claude paths need paid roles; the shipped config starts on ARC
    cfg.root = tmp_path  # cache and runs go to the temp dir
    return cfg


def make_llm(cfg, tmp_path, arc=(), claude=()):
    providers = {"arc": FakeProvider(arc), "claude": FakeProvider(claude)}
    return LLM(cfg, tmp_path, providers=providers, approver=approve_all), providers


def test_identical_request_is_served_from_cache(cfg, tmp_path):
    llm, p = make_llm(cfg, tmp_path, arc=[ok("first")])
    a = llm.complete("writer", "same prompt")
    b = llm.complete("writer", "same prompt")
    assert p["arc"].calls == 1
    assert b.cached and b.text == "first" and b.priced_usd == 0
    assert a.priced_usd > 0  # free ARC model, shadow-priced


def test_truncated_response_raises_and_is_not_cached(cfg, tmp_path):
    cut = RawResult("partial", "", "max_tokens", Usage(10, 10))
    llm, p = make_llm(cfg, tmp_path, arc=[cut, ok("full")])
    with pytest.raises(TruncatedError):
        llm.complete("writer", "long prompt")
    assert llm.complete("writer", "long prompt").text == "full"
    assert p["arc"].calls == 2


def test_refusal_raises(cfg, tmp_path):
    llm, _ = make_llm(cfg, tmp_path, claude=[RawResult("", "", "refusal", Usage(10, 0))])
    with pytest.raises(RefusedError):
        llm.complete("agent", "x", data="public")


def test_private_data_never_reaches_public_only_provider(cfg, tmp_path):
    llm, p = make_llm(cfg, tmp_path, claude=[ok()])
    with pytest.raises(DataScopeError):
        llm.complete("agent", "private notes")  # data defaults to private
    assert p["claude"].calls == 0
    assert llm.complete("agent", "public example", data="public").text == "hello"


def test_budget_stops_the_next_call_on_real_spend_only(cfg, tmp_path):
    cfg.budget = Budget(max_real_usd=0.001)
    llm, _ = make_llm(cfg, tmp_path, arc=[ok(inp=10_000, out=10_000), ok()], claude=[ok(inp=10_000, out=10_000), ok()])
    llm.complete("writer", "free ARC call, large shadow cost")
    llm.complete("writer", "second free call is not stopped")  # shadow dollars never stop a run
    llm.complete("agent", "paid call", data="public")
    with pytest.raises(BudgetExceeded, match="real spend"):
        llm.complete("agent", "next paid call", data="public")


def test_shadow_and_real_costs(cfg):
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    # GLM is free and shadow-priced as Claude Sonnet 5.5 ($2 in, $10 out).
    assert price_call(cfg.models["GLM-5.3-thinking-high"], cfg.models, usage) == (12.0, 0.0)
    assert price_call(cfg.models["claude-opus-5-5"], cfg.models, usage) == (24.0, 24.0)
    assert price_call(cfg.models["claude-opus-5-5"], cfg.models, usage, batch=True) == (12.0, 12.0)


def test_log_has_cost_but_no_prompt_text(cfg, tmp_path):
    llm, _ = make_llm(cfg, tmp_path, arc=[ok()])
    llm.complete("writer", "SECRET PROMPT TEXT", tag="t1")
    lines = [json.loads(l) for l in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    assert lines[-1]["tag"] == "t1" and lines[-1]["priced_usd"] > 0
    assert "SECRET" not in (tmp_path / "llm_calls.jsonl").read_text()
    summary = llm.write_summary()
    assert summary["real_usd"] == 0 and summary["shadow_usd"] > 0


def test_public_only_provider_needs_inputs_declared_public(cfg, tmp_path):
    cfg.data_public = False
    llm, p = make_llm(cfg, tmp_path, claude=[ok()])
    with pytest.raises(DataScopeError, match="does not declare its inputs public"):
        llm.complete("agent", "a public example", data="public")
    assert p["claude"].calls == 0
    cfg.data_public = True
    assert llm.complete("agent", "a public example", data="public").text == "hello"


def test_openai_reasoning_models_get_max_completion_tokens():
    from factory.config import ProviderConfig
    from factory.llm.providers import OpenAIProvider, Request

    sent = {}

    class Http:
        def post(self, url, json, headers):
            sent.update(json)

            class R:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
            return R()

    p = OpenAIProvider("openai", ProviderConfig(format="openai", base_url="https://api.openai.com",
                                                 token_param="max_completion_tokens"))
    p._http = Http()
    assert p.complete(Request(model="gpt-6.1-sol", messages=[{"role": "user", "content": "x"}], max_tokens=50)).text == "ok"
    assert sent["max_completion_tokens"] == 50 and "max_tokens" not in sent


class FakeBatchProvider:
    def __init__(self, results):
        self.results, self.submitted, self.polls = results, [], 0

    def submit_batch(self, reqs):
        self.submitted.append(sorted(reqs))
        return "batch-1"

    def batch_done(self, batch_id):
        self.polls += 1
        return self.polls >= 2

    def batch_results(self, batch_id):
        return self.results


def test_batch_uses_cache_prices_at_half_and_returns_errors_in_place(cfg, tmp_path):
    cfg.providers["claude"].batch = True
    fake = FakeBatchProvider({"i1": ok("two", inp=1000, out=500),
                              "i2": RawResult("cut", "", "max_tokens", Usage(10, 10))})
    llm = LLM(cfg, tmp_path, providers={"arc": FakeProvider([]), "claude": fake}, approver=approve_all)
    llm.cache.put(__import__("factory.llm.cache", fromlist=["request_key"]).request_key({"provider": "claude",
        **__import__("factory.llm.providers", fromlist=["Request"]).Request(
            model=cfg.roles["agent"].model, messages=[{"role": "user", "content": "one"}],
            max_tokens=cfg.roles["agent"].max_tokens, effort=cfg.roles["agent"].effort).as_dict()}),
        {"text": "cached one", "thinking": "", "stop_reason": "end_turn", "usage": {}})
    out = llm.complete_batch("agent", [{"user": "one"}, {"user": "two"}, {"user": "three"}],
                             data="public", poll_seconds=0)
    assert out[0].cached and out[0].text == "cached one"
    assert out[1].text == "two" and not out[1].cached
    assert isinstance(out[2], TruncatedError)
    assert fake.submitted == [["i1", "i2"]]  # only the uncached items were sent
    full, _ = price_call(cfg.model_info(cfg.roles["agent"].model), cfg.models, Usage(1000, 500))
    assert abs(out[1].priced_usd - full / 2) < 1e-9  # batch price
    assert not list((cfg.work_dir / "batches").glob("*.json"))  # finished batches are not resumed


def test_batch_resumes_a_submitted_batch_after_an_interruption(cfg, tmp_path):
    cfg.providers["claude"].batch = True

    class Interrupted(FakeBatchProvider):
        def batch_done(self, batch_id):
            raise KeyboardInterrupt  # the process dies while waiting

    first = Interrupted({})
    llm = LLM(cfg, tmp_path, providers={"arc": FakeProvider([]), "claude": first}, approver=approve_all)
    with pytest.raises(KeyboardInterrupt):
        llm.complete_batch("agent", [{"user": "x"}], data="public", poll_seconds=0)
    second = FakeBatchProvider({"i0": ok("x done")})
    llm2 = LLM(cfg, tmp_path, providers={"arc": FakeProvider([]), "claude": second}, approver=approve_all)
    out = llm2.complete_batch("agent", [{"user": "x"}], data="public", poll_seconds=0)
    assert out[0].text == "x done" and second.submitted == []  # waited for the first batch, did not resubmit


def test_providers_without_batch_api_fall_back_to_single_calls(cfg, tmp_path):
    llm, p = make_llm(cfg, tmp_path, arc=[ok("a"), ok("b")])
    out = llm.complete_batch("writer", [{"user": "p1"}, {"user": "p2"}], data="public", workers=1)
    assert [c.text for c in out] == ["a", "b"] and p["arc"].calls == 2


def test_openai_batch_round_trip_with_a_fake_server():
    import json as _json

    from factory.config import ProviderConfig
    from factory.llm.providers import OpenAIProvider, Request

    class Resp:
        def __init__(self, payload=None, text=""):
            self.payload, self.text, self.status_code = payload, text, 200

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class Http:
        def __init__(self):
            self.uploaded = None

        def post(self, url, headers=None, json=None, data=None, files=None):
            if url.endswith("/v1/files"):
                self.uploaded = files["file"][1].decode()
                return Resp({"id": "file-in"})
            return Resp({"id": "batch-9"})

        def get(self, url, headers=None):
            if url.endswith("/v1/batches/batch-9"):
                return Resp({"status": "completed", "output_file_id": "file-out"})
            body = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5}}
            return Resp(text=_json.dumps({"custom_id": "i0", "response": {"status_code": 200, "body": body}}))

    p = OpenAIProvider("openai", ProviderConfig(format="openai", base_url="https://x", token_param="max_completion_tokens"))
    p._http = Http()
    bid = p.submit_batch({"i0": Request(model="gpt-6.1-sol", messages=[{"role": "user", "content": "q"}], max_tokens=9)})
    line = _json.loads(p._http.uploaded)
    assert bid == "batch-9" and line["body"]["max_completion_tokens"] == 9 and line["url"] == "/v1/chat/completions"
    assert p.batch_done(bid) and p.batch_results(bid)["i0"].text == "hi"


def test_anthropic_batch_results_parse_succeeded_and_failed_items():
    from factory.config import ProviderConfig
    from factory.llm.providers import AnthropicProvider

    class Batches:
        def results(self, batch_id):
            msg = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                   "usage": {"input_tokens": 3, "output_tokens": 2}}
            return [{"custom_id": "i0", "result": {"type": "succeeded", "message": msg}},
                    {"custom_id": "i1", "result": {"type": "expired"}}]

    p = AnthropicProvider("claude", ProviderConfig(format="anthropic"))
    p._client = type("C", (), {"messages": type("M", (), {"batches": Batches()})()})()
    r = p.batch_results("b")
    assert r["i0"].text == "ok" and r["i0"].usage.output_tokens == 2 and "expired" in r["i1"]


def test_validated_retry_changes_the_request_so_the_cache_cannot_repeat_a_bad_reply(cfg, tmp_path):
    from pydantic import BaseModel

    from factory.llm import complete_validated

    class Out(BaseModel):
        x: int

    llm, p = make_llm(cfg, tmp_path, arc=[ok('{"x": "not a number"}'), ok('{"x": 3}')])
    out, c, errors = complete_validated(llm, "writer", "give x", "sys", Out, tag="t")
    assert out.x == 3 and len(errors) == 1 and p["arc"].calls == 2  # the retry reached the model


def test_a_zero_dollar_cap_stops_paid_calls_but_never_free_ones(cfg, tmp_path):
    cfg.budget = Budget(max_real_usd=0.0)  # the ARC-only setting
    llm, p = make_llm(cfg, tmp_path, arc=[ok("free")], claude=[ok("paid")])
    assert llm.complete("writer", "an ARC role").text == "free"
    with pytest.raises(Exception, match="cost cap"):
        llm.complete("agent", "a Claude role", data="public")
    assert p["claude"].calls == 0


def test_a_cut_off_reply_is_retried_with_a_request_for_a_shorter_one(cfg, tmp_path):
    from pydantic import BaseModel

    from factory.llm import complete_validated

    class Out(BaseModel):
        a: int

    seen = []
    cut = RawResult('{"a": ', "", "max_tokens", Usage(10, 10))
    llm, p = make_llm(cfg, tmp_path, arc=[cut, ok('{"a": 1}')])
    real = p["arc"].complete
    p["arc"].complete = lambda req: seen.append(req.messages[-1]["content"]) or real(req)
    out, _, errors = complete_validated(llm, "writer", "write it", "", Out, tag="t", data="public")
    assert out.a == 1 and len(errors) == 1
    assert "cut off at the output limit" in seen[1] and "much shorter" in seen[1]


def test_a_cut_off_reply_keeps_its_finished_pieces_and_asks_only_for_the_rest(cfg, tmp_path):
    from factory.llm import complete_validated
    from factory.pipeline.production import Built, FileSalvage, parse_files

    seen = []
    cut = RawResult("=== FILE: a.py ===\nprint(1)\n=== END FILE ===\n=== FILE: b.py ===\nprint(", "",
                    "max_tokens", Usage(10, 10))
    rest = ok("=== FILE: b.py ===\nprint(2)\n=== END FILE ===\nNOTES: done")
    llm, p = make_llm(cfg, tmp_path, arc=[cut, rest])
    real = p["arc"].complete
    p["arc"].complete = lambda req: seen.append(req.messages[-1]["content"]) or real(req)
    out, _, errors = complete_validated(llm, "writer", "write it", "", Built, tag="t", data="public",
                                        parse=parse_files, fmt="FILE blocks", salvage=FileSalvage())
    assert {f.path: f.content for f in out.files} == {"a.py": "print(1)\n", "b.py": "print(2)\n"}
    assert "already received and kept, do not write them again: a.py" in seen[1]
    assert len(errors) == 1
