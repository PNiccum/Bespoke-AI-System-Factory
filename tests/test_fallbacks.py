"""Free fallback models: when an ARC model is down or renamed, the role moves to the next free model it lists."""

from pathlib import Path

import httpx
import pytest

from factory.config import load_config
from factory.llm import LLM, Usage
from factory.llm.client import model_unavailable
from factory.llm.gate import approve_all
from factory.llm.providers import RawResult, TransientHTTPError

ROOT = Path(__file__).resolve().parents[1]


class ByModel:
    """Fails for the models it is told are down, answers for the rest, and records what was asked."""

    def __init__(self, down: dict[str, Exception]):
        self.down = down
        self.models = []

    def complete(self, req):
        self.models.append(req.model)
        if req.model in self.down:
            raise self.down[req.model]
        return RawResult(f"from {req.model}", "", "end_turn", Usage(100, 50))


def not_found(model):
    request = httpx.Request("POST", "https://llm-api.arc.vt.edu/api/v1/messages")
    return httpx.HTTPStatusError(f"404 model {model} not found", request=request,
                                 response=httpx.Response(404, request=request))


@pytest.fixture
def cfg(tmp_path):
    cfg = load_config(ROOT / "configs" / "factory.yaml")
    cfg.root = tmp_path
    cfg.providers["arc"].stream_retries = 0  # no waiting between retries in tests
    return cfg


def llm_with(cfg, tmp_path, arc):
    return LLM(cfg, tmp_path, use_cache=False, providers={"arc": arc, "openai": ByModel({}), "claude": ByModel({})},
               approver=approve_all)


def test_a_missing_model_falls_back_to_the_next_free_one(cfg, tmp_path):
    arc = ByModel({"GLM-5.3-thinking-high": not_found("GLM-5.3-thinking-high")})
    c = llm_with(cfg, tmp_path, arc).complete("coder", "fix it")
    assert c.model == "GLM-5.3" and c.text == "from GLM-5.3" and c.real_usd == 0
    assert arc.models == ["GLM-5.3-thinking-high", "GLM-5.3"]
    assert '"fallback_to": "GLM-5.3"' in (tmp_path / "llm_calls.jsonl").read_text()


def test_a_gateway_that_keeps_failing_moves_down_the_whole_list(cfg, tmp_path):
    busy = TransientHTTPError("503: overloaded")
    arc = ByModel({"DeepSeek-V4.1-Flash-thinking-max": busy, "DeepSeek-V4.1-Flash-thinking-low": busy})
    c = llm_with(cfg, tmp_path, arc).complete("reviewer", "review it")
    assert c.model == "DeepSeek-V4.1-Flash"


def test_when_every_fallback_is_down_the_last_error_is_raised(cfg, tmp_path):
    arc = ByModel({m: not_found(m) for m in ("GLM-5.3-thinking-high", "GLM-5.3", "gpt-oss-120b-thinking-high")})
    with pytest.raises(httpx.HTTPStatusError, match="gpt-oss-120b-thinking-high"):
        llm_with(cfg, tmp_path, arc).complete("coder", "fix it")


def test_an_ordinary_bad_request_does_not_fall_back(cfg, tmp_path):
    request = httpx.Request("POST", "https://x")
    bad = httpx.HTTPStatusError("400 messages: field required", request=request,
                                response=httpx.Response(400, request=request))
    arc = ByModel({"GLM-5.3-thinking-high": bad})
    with pytest.raises(httpx.HTTPStatusError):
        llm_with(cfg, tmp_path, arc).complete("coder", "fix it")
    assert arc.models == ["GLM-5.3-thinking-high"] and not model_unavailable(bad)


def test_fallbacks_must_be_free_and_keep_families_apart(tmp_path):
    text = (ROOT / "configs" / "factory.yaml").read_text()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "models.yaml").write_text((ROOT / "configs" / "models.yaml").read_text())
    paid = text.replace("fallbacks: [DeepSeek-V4.1-Flash-thinking-low, DeepSeek-V4.1-Flash]",
                        "fallbacks: [claude-sonnet-5-5]")
    (tmp_path / "configs" / "factory.yaml").write_text(paid)
    with pytest.raises(ValueError, match="must be free"):
        load_config(tmp_path / "configs" / "factory.yaml")
    same_family = text.replace("fallbacks: [DeepSeek-V4.1-Flash-thinking-low, DeepSeek-V4.1-Flash]",
                               "fallbacks: [GLM-5.3]")
    (tmp_path / "configs" / "factory.yaml").write_text(same_family)
    with pytest.raises(ValueError, match="family separation"):
        load_config(tmp_path / "configs" / "factory.yaml")


def test_the_paid_step_never_falls_back(cfg):
    assert cfg.escalations["coder"].fallbacks == [] and cfg.escalations["reviewer"].fallbacks == []
