"""Provider adapters: one request shape in, one result shape out.

Two wire formats are supported, so a role can move between ARC and the Claude
API with a config change:

- ``anthropic``: the Messages API, used by the Claude API and (per the previous
  build) by the ARC gateway.
- ``openai``: OpenAI-compatible chat completions, for gateways that serve it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import anthropic
import httpx

from factory.config import ProviderConfig
from factory.llm.cost import Usage

try:  # anthropic>=1.0 runs on httpx2; its transport errors are a separate class
    import httpx2
    _HTTPX2_ERRORS: tuple = (httpx2.TransportError,)
except ImportError:  # pragma: no cover
    _HTTPX2_ERRORS = ()

# Retried with backoff by the client. The ARC gateway occasionally drops long
# streams mid-response, which the SDK does not retry once streaming has begun.
TRANSIENT = (
    anthropic.APIConnectionError, anthropic.APITimeoutError, anthropic.InternalServerError,
    anthropic.RateLimitError, httpx.TransportError, *_HTTPX2_ERRORS,
)


class TransientHTTPError(RuntimeError):
    """A 429 or 5xx from an OpenAI-compatible endpoint."""


@dataclass
class Request:
    model: str
    messages: list[dict]
    max_tokens: int
    system: str | None = None
    effort: str | None = None
    params: dict = field(default_factory=dict)
    tools: list[dict] | None = None  # Anthropic tool schemas: {name, description, input_schema}

    def as_dict(self) -> dict:
        d = {"model": self.model, "messages": self.messages, "max_tokens": self.max_tokens,
             "system": self.system, "effort": self.effort, "params": self.params}
        if self.tools is not None:  # only when present, so requests without tools keep their cache keys
            d["tools"] = self.tools
        return d


@dataclass
class RawResult:
    text: str
    thinking: str
    stop_reason: str | None  # normalised: end_turn, max_tokens, refusal, tool_use, or the provider's value
    usage: Usage
    tool_calls: list[dict] = field(default_factory=list)  # [{id, name, input}]


def resolve_key(cfg: ProviderConfig) -> str | None:
    if cfg.api_key_env and os.environ.get(cfg.api_key_env):
        return os.environ[cfg.api_key_env].strip()
    if cfg.api_key_file:
        path = cfg.api_key_file.expanduser()
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return None


class AnthropicProvider:
    def __init__(self, name: str, cfg: ProviderConfig):
        self.name, self.cfg = name, cfg
        self._client = None

    @property
    def client(self) -> anthropic.Anthropic:
        # Created on first use so a config loads without every key present.
        if self._client is None:
            key = resolve_key(self.cfg)
            if self.cfg.base_url and not key:
                raise RuntimeError(f"provider {self.name!r}: no API key "
                                   f"(set {self.cfg.api_key_env} or create {self.cfg.api_key_file})")
            kwargs = {"timeout": self.cfg.timeout_s, "max_retries": 3}
            if self.cfg.base_url:
                kwargs["base_url"] = self.cfg.base_url
            if key:
                kwargs["api_key"] = key
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def complete(self, req: Request) -> RawResult:
        return self._stream(self._kwargs(req))

    def _kwargs(self, req: Request) -> dict:
        cache = self.cfg.prompt_caching
        messages = with_cache_breakpoint(req.messages) if cache else req.messages
        kwargs = dict(model=req.model, max_tokens=req.max_tokens, messages=messages, **req.params)
        if req.system:
            kwargs["system"] = ([{"type": "text", "text": req.system, "cache_control": EPHEMERAL}]
                                if cache else req.system)
        if req.effort:
            kwargs["output_config"] = {"effort": req.effort}
        if req.tools:
            kwargs["tools"] = ([*req.tools[:-1], {**req.tools[-1], "cache_control": EPHEMERAL}]
                               if cache else req.tools)
        return kwargs

    # --- Message Batches: half price, results within 24 hours -------------------------------------------------
    def submit_batch(self, reqs: dict[str, Request]) -> str:
        b = self.client.messages.batches.create(
            requests=[{"custom_id": cid, "params": self._kwargs(r)} for cid, r in reqs.items()])
        return b.id

    def batch_done(self, batch_id: str) -> bool:
        return _field(self.client.messages.batches.retrieve(batch_id), "processing_status") == "ended"

    def batch_results(self, batch_id: str) -> dict[str, RawResult | str]:
        """custom_id -> result, or an error string for items that failed, expired, or were canceled."""
        out: dict[str, RawResult | str] = {}
        for entry in self.client.messages.batches.results(batch_id):
            res = _field(entry, "result")
            kind = _field(res, "type")
            out[_field(entry, "custom_id")] = (_from_message(_field(res, "message")) if kind == "succeeded"
                                               else f"batch item {kind}: {str(_field(res, 'error'))[:200]}")
        return out

    def _stream(self, kwargs: dict) -> RawResult:
        """Accumulate a streamed response from raw events.

        The SDK's own accumulator rejects content-block types it doesn't know,
        which the ARC gateway occasionally sends; only text, thinking, stop
        reason and usage are collected here.
        """
        text, thinking = [], []
        stop_reason, usage = None, Usage()
        tools: dict[int, dict] = {}  # content-block index -> tool call being streamed
        for event in self.client.messages.create(stream=True, **kwargs):
            kind = _field(event, "type")
            if kind == "message_start":
                _read_usage(_field(_field(event, "message"), "usage"), usage)
            elif kind == "content_block_start":
                block = _field(event, "content_block")
                if _field(block, "type") == "tool_use":
                    tools[_field(event, "index")] = {"id": _field(block, "id"), "name": _field(block, "name"),
                                                     "json": []}
            elif kind == "content_block_delta":
                delta = _field(event, "delta")
                dtype = _field(delta, "type")
                if dtype == "text_delta":
                    text.append(_field(delta, "text") or "")
                elif dtype == "thinking_delta":
                    thinking.append(_field(delta, "thinking") or "")
                elif dtype == "input_json_delta" and _field(event, "index") in tools:
                    tools[_field(event, "index")]["json"].append(_field(delta, "partial_json") or "")
            elif kind == "message_delta":
                stop_reason = _field(_field(event, "delta"), "stop_reason") or stop_reason
                _read_usage(_field(event, "usage"), usage)
        calls = [{"id": t["id"], "name": t["name"], "input": _parse_args("".join(t["json"]))}
                 for _, t in sorted(tools.items())]
        return RawResult("".join(text), "".join(thinking), stop_reason, usage, calls)


def _from_message(msg) -> RawResult:
    """A complete (non-streamed) Messages API response as a RawResult."""
    text, thinking, calls = [], [], []
    for b in _field(msg, "content") or []:
        t = _field(b, "type")
        if t == "text":
            text.append(_field(b, "text") or "")
        elif t == "thinking":
            thinking.append(_field(b, "thinking") or "")
        elif t == "tool_use":
            calls.append({"id": _field(b, "id"), "name": _field(b, "name"), "input": _field(b, "input") or {}})
    usage = Usage()
    _read_usage(_field(msg, "usage"), usage)
    return RawResult("".join(text), "".join(thinking), _field(msg, "stop_reason"), usage, calls)


class OpenAIProvider:
    def __init__(self, name: str, cfg: ProviderConfig):
        if not cfg.base_url:
            raise ValueError(f"provider {name!r}: openai format needs a base_url")
        self.name, self.cfg = name, cfg
        self._http = httpx.Client(timeout=cfg.timeout_s)

    def _headers(self) -> dict:
        key = resolve_key(self.cfg)
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _url(self, path: str) -> str:
        return f"{self.cfg.base_url.rstrip('/')}{path}"

    def _body(self, req: Request) -> dict:
        messages = ([{"role": "system", "content": req.system}] if req.system else []) + to_openai_messages(req.messages)
        body = {"model": req.model, "messages": messages, self.cfg.token_param: req.max_tokens, **req.params}
        if req.tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                               "parameters": t["input_schema"]}} for t in req.tools]
        return body

    def complete(self, req: Request) -> RawResult:
        r = self._http.post(self._url("/v1/chat/completions"), json=self._body(req), headers=self._headers())
        if r.status_code == 429 or r.status_code >= 500:
            raise TransientHTTPError(f"{r.status_code}: {r.text[:200]}")
        if r.status_code >= 400:  # say why: the provider's message is the useful part of a client error
            raise httpx.HTTPStatusError(f"{r.status_code} from {self.name}: {r.text[:500]}", request=r.request,
                                        response=r)
        return _from_chat_completion(r.json())

    # --- Batch API: a JSONL file of requests, half price, results within 24 hours ------------------------------
    def submit_batch(self, reqs: dict[str, Request]) -> str:
        lines = "\n".join(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                                      "body": self._body(r)}) for cid, r in reqs.items())
        f = self._http.post(self._url("/v1/files"), headers=self._headers(), data={"purpose": "batch"},
                            files={"file": ("batch.jsonl", lines.encode(), "application/jsonl")})
        f.raise_for_status()
        b = self._http.post(self._url("/v1/batches"), headers=self._headers(), json={
            "input_file_id": f.json()["id"], "endpoint": "/v1/chat/completions", "completion_window": "24h"})
        b.raise_for_status()
        return b.json()["id"]

    def _batch(self, batch_id: str) -> dict:
        r = self._http.get(self._url(f"/v1/batches/{batch_id}"), headers=self._headers())
        r.raise_for_status()
        return r.json()

    def batch_done(self, batch_id: str) -> bool:
        return self._batch(batch_id)["status"] in ("completed", "failed", "expired", "cancelled")

    def batch_results(self, batch_id: str) -> dict[str, RawResult | str]:
        b = self._batch(batch_id)
        out: dict[str, RawResult | str] = {}
        for key in ("output_file_id", "error_file_id"):
            if not b.get(key):
                continue
            r = self._http.get(self._url(f"/v1/files/{b[key]}/content"), headers=self._headers())
            r.raise_for_status()
            for line in r.text.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                resp = rec.get("response") or {}
                if resp.get("status_code") == 200:
                    out[rec["custom_id"]] = _from_chat_completion(resp["body"])
                else:
                    out[rec["custom_id"]] = f"batch item failed: {str(rec.get('error') or resp)[:200]}"
        return out  # items absent here (an expired or failed batch) are reported missing by the caller


def _from_chat_completion(data: dict) -> RawResult:
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    u = data.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    usage = Usage(input_tokens=(u.get("prompt_tokens") or 0) - cached,
                  output_tokens=u.get("completion_tokens") or 0, cache_read_tokens=cached)
    stop = {"length": "max_tokens", "stop": "end_turn", "content_filter": "refusal",
            "tool_calls": "tool_use"}.get(choice.get("finish_reason"), choice.get("finish_reason"))
    calls = [{"id": c.get("id"), "name": (c.get("function") or {}).get("name"),
              "input": _parse_args((c.get("function") or {}).get("arguments") or "")}
             for c in msg.get("tool_calls") or []]
    return RawResult(msg.get("content") or "", msg.get("reasoning_content") or "", stop, usage, calls)


EPHEMERAL = {"type": "ephemeral"}


def with_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """A copy of the history whose last content block is a cache breakpoint.

    Marking the newest message each turn caches the whole conversation so far; the next turn finds the
    previous breakpoint as a cached prefix and pays the cache rate for it. The input list is not changed.
    """
    if not messages:
        return messages
    *head, last = messages
    content = last["content"]
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else [dict(b) for b in content]
    if not blocks:
        return messages
    blocks[-1]["cache_control"] = EPHEMERAL
    return [*head, {**last, "content": blocks}]


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Translate the canonical (Anthropic-format) history: tool_use blocks become tool_calls, tool_result blocks
    become role "tool" messages. Plain string contents pass through unchanged."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        if m["role"] == "assistant":
            calls = [{"id": b["id"], "type": "function",
                      "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))}}
                     for b in content if b.get("type") == "tool_use"]
            out.append({"role": "assistant", "content": text or None, **({"tool_calls": calls} if calls else {})})
        else:
            for b in content:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                "content": c if isinstance(c, str) else json.dumps(c)})
            if text:
                out.append({"role": "user", "content": text})
    return out


def _parse_args(raw: str):
    """Tool arguments as a dict; unparseable arguments are passed on as text so the tool can report the error."""
    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_unparsed": raw}


def make_provider(name: str, cfg: ProviderConfig):
    return AnthropicProvider(name, cfg) if cfg.format == "anthropic" else OpenAIProvider(name, cfg)


def _read_usage(u, usage: Usage) -> None:
    if u is None:
        return
    usage.input_tokens = _field(u, "input_tokens") or usage.input_tokens
    usage.output_tokens = _field(u, "output_tokens") or usage.output_tokens
    usage.cache_read_tokens = _field(u, "cache_read_input_tokens") or usage.cache_read_tokens
    usage.cache_write_tokens = _field(u, "cache_creation_input_tokens") or usage.cache_write_tokens


def _field(obj, name):
    """Attribute or dict key: stream events may arrive as either."""
    if obj is None:
        return None
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
