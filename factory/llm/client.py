"""The single entry point for model calls.

``LLM.complete(role, ...)`` picks the provider and model from the factory config,
so moving a role between ARC and the Claude API is a config change. Every
call is:

- checked against the provider's data scope (data not declared public never
  goes to a ``public_only`` provider such as the Claude API);
- checked against the run budget before it is made, and, when the model is
  paid, put through the spend gate (approval and remaining credit, see gate.py);
- answered from the response cache when the identical request was seen;
- retried on transient errors, and treated as a failure (never parsed) when
  truncated or refused;
- priced (shadow prices for free models) and logged, without prompt text but
  with its hash for provenance, to ``runs/<run-id>/llm_calls.jsonl``.

``escalate=True`` sends the call to the role's paid escalation step instead of
its free default.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import anthropic
import httpx
from pydantic import ValidationError

from factory.config import FactoryConfig, RoleConfig
from factory.events import FILE as EVENTS_FILE
from factory.events import EventLog, log_of
from factory.llm.cache import ResponseCache, request_key
from factory.llm.cost import BudgetExceeded, Ledger, Usage, price_call
from factory.llm.gate import Approver, SpendDenied, SpendGate, SpendLedger, SpendRequest, estimate_usd
from factory.llm.providers import TRANSIENT, RawResult, Request, TransientHTTPError, make_provider

DataClass = Literal["public", "private"]


class DataScopeError(RuntimeError):
    """Private data was about to be sent to a provider that may only see public documents."""


class TruncatedError(RuntimeError):
    """The response hit its token limit; it is logged and never parsed."""


class RefusedError(RuntimeError):
    """The model declined the request."""


@dataclass
class Completion:
    role: str
    model: str
    text: str
    thinking: str
    stop_reason: str | None
    usage: Usage
    priced_usd: float
    real_usd: float
    seconds: float
    cached: bool = False
    tool_calls: list = None  # [{id, name, input}] when the model asked for tools

    def __post_init__(self):
        self.tool_calls = self.tool_calls or []

    def json(self):
        """Parse the first JSON object or array in the response text."""
        return extract_json(self.text)


def extract_json(text: str):
    # Models often wrap JSON in a ```json fence or add a sentence around it.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = min((i for i in (candidate.find("{"), candidate.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError("no JSON found in response")
    obj, _ = json.JSONDecoder().raw_decode(candidate[start:])
    return obj


def model_unavailable(e: Exception) -> bool:
    """Whether a failed call means the model cannot be reached at all, so a fallback model is worth trying: the
    gateway kept failing through every retry, or it does not know the model. Other errors (a bad request, a refusal,
    a spend limit) would fail the same way on any model."""
    if isinstance(e, (*TRANSIENT, TransientHTTPError)):
        return True
    if not isinstance(e, (httpx.HTTPStatusError, anthropic.APIStatusError)):
        return False
    code = getattr(e, "status_code", None) or e.response.status_code
    text = str(e).lower()
    return code == 404 or (code in (400, 422) and "model" in text
                           and any(w in text for w in ("not found", "unknown", "does not exist", "not available")))


def new_run_dir(cfg: FactoryConfig, label: str = "") -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base = cfg.root / cfg.runs_dir / (f"{stamp}-{label}" if label else stamp)
    for n in range(1, 100):  # two runs with the same label in the same second get -2, -3, ...
        path = base if n == 1 else base.with_name(f"{base.name}-{n}")
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"too many runs named {base.name}")


# One concurrency gate per provider for the whole process, not one per LLM: projects run side by side
# (bench_projects --parallel) each have their own LLM, and ARC's limit is per user, not per run.
_GATES: dict[str, threading.BoundedSemaphore] = {}
_GATES_LOCK = threading.Lock()


def provider_gate(name: str, limit: int) -> threading.BoundedSemaphore:
    with _GATES_LOCK:
        if name not in _GATES:
            _GATES[name] = threading.BoundedSemaphore(limit)
        return _GATES[name]


class LLM:
    def __init__(self, cfg: FactoryConfig, run_dir: Path, use_cache: bool = True, providers: dict | None = None,
                 approver: Approver | None = None):
        self.cfg = cfg
        self.run_dir = run_dir
        self.log_path = run_dir / "llm_calls.jsonl"
        self.ledger = Ledger(cfg.budget)
        self.gate = SpendGate(approver, cfg.budget.approve_above_usd)
        self.spend = SpendLedger(cfg.work_dir / "spend.json", cfg.credits)
        self.cache = ResponseCache(cfg.work_dir / "cache.sqlite") if use_cache else None
        # Tests pass fake providers; otherwise one adapter per configured provider.
        self.providers = providers or {name: make_provider(name, p) for name, p in cfg.providers.items()}
        self._gates = {name: provider_gate(name, p.max_concurrency) for name, p in cfg.providers.items()}
        self._log_lock = threading.Lock()
        self.events = EventLog(run_dir / EVENTS_FILE, run_dir.name)  # every error this run meets (factory/events.py)
        # Which tier and which model per role this run used, so a run can be reproduced from its directory.
        info = run_dir / "run_info.json"
        if not info.exists():
            info.write_text(json.dumps({"tier": cfg.tier, "role_tiers": cfg.role_tiers, "roles": {
                r: {"provider": rc.provider, "model": rc.model, "max_tokens": rc.max_tokens, "effort": rc.effort,
                    "on_truncation": rc.on_truncation}
                for r, rc in cfg.roles.items()}}, indent=1), encoding="utf-8")

    def complete(self, role: str, user: str, system: str | None = None, *, tag: str = "",
                 data: DataClass = "private", max_tokens: int | None = None, escalate: bool = False,
                 lighter: bool = False) -> Completion:
        return self.converse(role, [{"role": "user", "content": user}], system, tag=tag, data=data,
                             max_tokens=max_tokens, escalate=escalate, lighter=lighter)

    def role_config(self, role: str, escalate: bool = False) -> RoleConfig:
        """The model a role uses: its configured default, or its paid escalation step."""
        if not escalate:
            return self.cfg.roles[role]
        if role not in self.cfg.escalations:
            raise ValueError(f"role {role!r} has no escalation step (quality: in the config)")
        return self.cfg.escalations[role]

    def converse(self, role: str, messages: list[dict], system: str | None = None, *, tools: list[dict] | None = None,
                 tag: str = "", data: DataClass = "private", max_tokens: int | None = None,
                 escalate: bool = False, lighter: bool = False) -> Completion:
        """One model turn over a full message history (canonical Anthropic format), optionally with tools. When
        the role's model is unavailable, its free fallbacks are tried in order. ``lighter`` (a retry after a reply
        hit max_tokens) starts from the role's ``on_truncation`` model, a faster one of the same family."""
        rc = self.role_config(role, escalate)
        models = [rc.model, *rc.fallbacks]
        if lighter and rc.on_truncation and not escalate:
            models = [rc.on_truncation, *(m for m in models if m != rc.on_truncation)]
        skipped: list[str] = []  # events for models that were unavailable; corrected if a fallback answers
        for i, model in enumerate(models):
            try:
                done = self._converse_one(role, rc.model_copy(update={"model": model}), messages, system, tools,
                                          tag, data, max_tokens)
                if skipped:
                    self.events.resolve(skipped, f"fallback model {model} answered", [tag or role])
                return done
            except Exception as e:
                last = i == len(models) - 1 or not model_unavailable(e)
                skipped.append(self.events.record("llm", f"{role}: {model}: {type(e).__name__}: {e}"[:400],
                                                  stage=tag or role, severity="error" if last else "warning"))
                if last:
                    raise
                self._log(tag=tag, role=role, model=model, provider=rc.provider, fallback_to=models[i + 1],
                          error=f"{type(e).__name__}: {e}"[:300])
        raise AssertionError("unreachable")

    def _converse_one(self, role: str, rc: RoleConfig, messages: list[dict], system: str | None,
                      tools: list[dict] | None, tag: str, data: DataClass, max_tokens: int | None) -> Completion:
        provider_cfg = self.cfg.providers[rc.provider]
        if provider_cfg.data_scope == "public_only" and data != "public":
            raise DataScopeError(f"role {role!r} uses {rc.provider!r}, which may only receive public "
                                 f"data; this call is marked {data!r}")
        if provider_cfg.data_scope == "public_only" and not self.cfg.data_public:
            raise DataScopeError(f"role {role!r} uses {rc.provider!r}, which may only receive public data, "
                                 f"and this config does not declare its inputs public (data_public)")
        req = Request(model=rc.model, messages=messages, max_tokens=max_tokens or rc.max_tokens, system=system,
                      effort=rc.effort, params=rc.params, tools=tools)
        key = request_key({"provider": rc.provider, **req.as_dict()})

        if self.cache and (hit := self.cache.get(key)):
            raw = RawResult(hit["text"], hit["thinking"], hit["stop_reason"], Usage(**hit["usage"]),
                            hit.get("tool_calls", []))
            self.ledger.add(0.0, 0.0, 0, cached=True)
            self._log(tag=tag, role=role, model=rc.model, provider=rc.provider, cached=True)
            return Completion(role, rc.model, raw.text, raw.thinking, raw.stop_reason, raw.usage, 0.0, 0.0, 0.0, True,
                              raw.tool_calls)

        info = self.cfg.model_info(rc.model)
        self.ledger.check(paid=info.real)
        if info.real:
            prompt_chars = len(json.dumps(messages)) + len(system or "")
            self._check_spend(role, rc, estimate_usd(info, prompt_chars, req.max_tokens))
        raw, seconds = self._call_with_retries(rc.provider, req, role, tag)
        priced, real = price_call(info, self.cfg.models, raw.usage)
        self.ledger.add(priced, real, raw.usage.total)
        self.spend.add(rc.provider, real)
        self._log(tag=tag, role=role, model=rc.model, provider=rc.provider, prompt_sha256=key,
                  stop_reason=raw.stop_reason,
                  seconds=round(seconds, 2), priced_usd=round(priced, 6), real_usd=round(real, 6),
                  **asdict(raw.usage), **({"tool_calls": [c["name"] for c in raw.tool_calls]} if raw.tool_calls else {}))

        if raw.stop_reason == "max_tokens":
            raise TruncatedError(f"{tag or role}: response hit max_tokens ({req.max_tokens}); retry with smaller input")
        if raw.stop_reason == "refusal":
            raise RefusedError(f"{tag or role}: model declined the request")
        if self.cache:
            entry = {"text": raw.text, "thinking": raw.thinking, "stop_reason": raw.stop_reason,
                     "usage": asdict(raw.usage)}
            if raw.tool_calls:
                entry["tool_calls"] = raw.tool_calls
            self.cache.put(key, entry)
        return Completion(role, rc.model, raw.text, raw.thinking, raw.stop_reason, raw.usage,
                          priced, real, round(seconds, 2), tool_calls=raw.tool_calls)

    def complete_batch(self, role: str, items: list[dict], *, data: DataClass = "private", poll_seconds: float = 60,
                       max_wait_s: float = 26 * 3600, workers: int = 8) -> list:
        """Many independent calls for one role. Returns, in order, a Completion or the exception for each item.

        Each item is {"user": ..., "system": ..., "tag": ..., "max_tokens": ...}. Cached requests are answered
        from the cache; the rest go to the provider's batch API at the batch price, or, for a provider without
        one (ARC), to concurrent single calls. A submitted batch is recorded under the workspace, so a rerun
        after an interruption waits for the same batch instead of paying for a new one.
        """
        rc = self.cfg.roles[role]
        provider_cfg = self.cfg.providers[rc.provider]
        if not provider_cfg.batch:
            from concurrent.futures import ThreadPoolExecutor

            def one(it):
                try:
                    return self.complete(role, it["user"], it.get("system"), tag=it.get("tag", ""), data=data,
                                         max_tokens=it.get("max_tokens"))
                except Exception as e:  # returned, not raised: one bad item must not lose the others
                    return e
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(one, items))

        if provider_cfg.data_scope == "public_only" and (data != "public" or not self.cfg.data_public):
            raise DataScopeError(f"role {role!r} uses {rc.provider!r}, which may only receive public data")
        reqs, keys, out = {}, [], [None] * len(items)
        for i, it in enumerate(items):
            req = Request(model=rc.model, messages=[{"role": "user", "content": it["user"]}],
                          max_tokens=it.get("max_tokens") or rc.max_tokens, system=it.get("system"),
                          effort=rc.effort, params=rc.params)
            key = request_key({"provider": rc.provider, **req.as_dict()})
            keys.append(key)
            if self.cache and (hit := self.cache.get(key)):
                raw = RawResult(hit["text"], hit["thinking"], hit["stop_reason"], Usage(**hit["usage"]),
                                hit.get("tool_calls", []))
                self.ledger.add(0.0, 0.0, 0, cached=True)
                self._log(tag=it.get("tag", ""), role=role, model=rc.model, provider=rc.provider, cached=True)
                out[i] = Completion(role, rc.model, raw.text, raw.thinking, raw.stop_reason, raw.usage, 0.0, 0.0,
                                    0.0, True, raw.tool_calls)
            else:
                reqs[f"i{i}"] = req
        if not reqs:
            return out

        info = self.cfg.model_info(rc.model)
        self.ledger.check(paid=info.real)
        if info.real:  # one approval covers the whole batch, at the batch price
            est = sum(estimate_usd(info, len(json.dumps(r.messages)) + len(r.system or ""), r.max_tokens)
                      for r in reqs.values()) * info.batch_discount
            self._check_spend(role, rc, est)
        provider = self.providers[rc.provider]
        pending_dir = self.cfg.work_dir / "batches"
        pending_dir.mkdir(parents=True, exist_ok=True)
        batch_key = request_key({"provider": rc.provider, "keys": [keys[int(c[1:])] for c in sorted(reqs)]})
        pending = pending_dir / f"{batch_key[:24]}.json"
        if pending.exists():
            batch_id = json.loads(pending.read_text(encoding="utf-8"))["batch_id"]  # resume the batch an earlier run submitted
        else:
            batch_id = provider.submit_batch(reqs)
            pending.write_text(json.dumps({"batch_id": batch_id, "provider": rc.provider, "role": role,
                                           "n": len(reqs), "submitted": datetime.now(UTC).isoformat()}), encoding="utf-8")
        self._log(event="batch_submitted", role=role, model=rc.model, provider=rc.provider, batch_id=batch_id,
                  n=len(reqs))
        t0 = time.monotonic()
        while not provider.batch_done(batch_id):
            if time.monotonic() - t0 > max_wait_s:
                raise TimeoutError(f"batch {batch_id} not finished after {max_wait_s / 3600:.0f} h; rerun to resume")
            time.sleep(poll_seconds)
        results = provider.batch_results(batch_id)
        for cid, req in reqs.items():
            i = int(cid[1:])
            tag = items[i].get("tag", "")
            raw = results.get(cid, "batch item missing from the results")
            if isinstance(raw, str):
                self._log(tag=tag, role=role, model=rc.model, provider=rc.provider, batch=True, error=raw)
                out[i] = RuntimeError(f"{tag or role}: {raw}")
                continue
            priced, real = price_call(info, self.cfg.models, raw.usage, batch=True)
            self.ledger.add(priced, real, raw.usage.total)
            self.spend.add(rc.provider, real)
            self._log(tag=tag, role=role, model=rc.model, provider=rc.provider, batch=True,
                      stop_reason=raw.stop_reason, priced_usd=round(priced, 6), real_usd=round(real, 6),
                      **asdict(raw.usage))
            if raw.stop_reason == "max_tokens":
                out[i] = TruncatedError(f"{tag or role}: response hit max_tokens ({req.max_tokens})")
            elif raw.stop_reason == "refusal":
                out[i] = RefusedError(f"{tag or role}: model declined the request")
            else:
                if self.cache:
                    self.cache.put(keys[i], {"text": raw.text, "thinking": raw.thinking, "stop_reason": raw.stop_reason,
                                             "usage": asdict(raw.usage)})
                out[i] = Completion(role, rc.model, raw.text, raw.thinking, raw.stop_reason, raw.usage, priced, real,
                                    0.0)
        pending.unlink(missing_ok=True)  # finished: a later identical request set is answered from the cache
        return out

    def _check_spend(self, role: str, rc: RoleConfig, estimate: float) -> None:
        """Remaining credit first (no point asking a person to approve what cannot be paid), then approval."""
        self.spend.check(rc.provider)
        req = SpendRequest(role=role, provider=rc.provider, model=rc.model, estimate_usd=round(estimate, 4),
                           run_spent_usd=round(self.ledger.real_usd, 4), reason="")
        try:
            self.gate.check(req)
        finally:
            if req.reason:  # a person was asked; record what they saw, whatever they answered
                self._log(event="spend_gate", role=role, model=rc.model, provider=rc.provider, reason=req.reason,
                          estimate_usd=req.estimate_usd, run_spent_usd=req.run_spent_usd)

    def _call_with_retries(self, provider: str, req: Request, role: str, tag: str) -> tuple[RawResult, float]:
        retries = self.cfg.providers[provider].stream_retries
        seen: list[str] = []
        for attempt in range(retries + 1):
            try:
                with self._gates[provider]:
                    t0 = time.monotonic()
                    raw = self.providers[provider].complete(req)
                    if seen:
                        self.events.resolve(seen, f"retry {attempt} succeeded", [tag or role])
                    return raw, time.monotonic() - t0
            except (*TRANSIENT, TransientHTTPError) as e:
                seen.append(self.events.record("llm", f"{provider}/{req.model}: {type(e).__name__}: {e}"[:400],
                                               stage=tag or role, severity="warning"))
                self._log(tag=tag, role=role, model=req.model, provider=provider,
                          error=f"{type(e).__name__}: {e}"[:300], attempt=attempt)
                if attempt == retries:
                    raise
                time.sleep(min(30 * 2 ** attempt, 300))
        raise AssertionError("unreachable")

    def _log(self, **record) -> None:
        with self._log_lock, self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now(UTC).isoformat(), **record}) + "\n")

    def write_summary(self) -> dict:
        summary = self.ledger.summary()
        (self.run_dir / "cost_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def complete_validated(llm: LLM, role: str, prompt: str, system: str, model_cls, tag: str,
                       attempts: int = 3, data: DataClass = "private", max_tokens: int | None = None,
                       escalate: bool = False, parse=None, fmt: str = "JSON", rescue: bool = True):
    """Call the model until the output parses and validates, or give up.

    ``rescue``: when every free attempt fails and the role has a paid ``quality`` step, one more attempt goes to that
    step, through the spend gate (so a person, or the standing approval, decides) and only for public data. A
    declined or unaffordable rescue leaves the usual "no valid output" error.

    A reply that does not parse is still cached, so a retry must change the request: it names the problem and
    asks for the corrected JSON. (Repeating the identical request would only return the cached bad reply.)
    ``parse`` turns the completion into the data to validate (default: its first JSON object); ``fmt`` names the
    format in the retry request.
    """
    errors, cut_off = [], False
    events, started = log_of(llm), datetime.now(UTC).isoformat()

    def settled(result):  # the attempts that failed before this one worked are corrected, by this success
        if errors:
            events.resolve_open(f"a later attempt returned valid output ({len(errors)} failed first)",
                                [tag], source="llm", since=started)
        return result
    for i in range(attempts):
        p = prompt if i == 0 else (
            f"{prompt}\n\nYour previous reply was cut off at the output limit. Reply again with only the {fmt}, "
            "much shorter: keep only what is asked for, drop comments, docstrings and repetition, and cover fewer "
            "cases if you must." if cut_off else
            f"{prompt}\n\nYour previous reply could not be used ({errors[-1][:160]}). "
            f"Reply again with only the corrected {fmt}, nothing else.")
        try:
            # after a cut-off, the retry goes to the role's lighter same-family model: in the Nov 2-4 logs 51
            # replies hit max_tokens (7.3 h of model time), and the same thinking model often overran again
            c = llm.complete(role, p, system=system, tag=tag, data=data, max_tokens=max_tokens, escalate=escalate,
                             lighter=cut_off)
            return settled((model_cls.model_validate(parse(c) if parse else c.json()), c, errors))
        except TruncatedError as e:
            errors.append(str(e))
            events.record("llm", f"{tag}: reply cut off at max_tokens", stage=tag, severity="warning")
            cut_off = True
        except (ValueError, ValidationError) as e:
            errors.append(f"{type(e).__name__}: {str(e)[:200]}")
            events.record("llm", f"{tag}: unusable output: {errors[-1]}", stage=tag, severity="warning")
            cut_off = False
    if rescue and not escalate and data == "public" and role in llm.cfg.escalations:
        p = (f"{prompt}\n\nA weaker model could not produce usable output ({errors[-1][:160]}). "
             f"Reply with only the {fmt}, nothing else.")
        try:
            c = llm.complete(role, p, system=system, tag=f"{tag}-paid", data=data, max_tokens=max_tokens,
                             escalate=True)
            return settled((model_cls.model_validate(parse(c) if parse else c.json()), c, errors))
        except (SpendDenied, BudgetExceeded, DataScopeError) as e:  # not approved or not allowed: no spend
            errors.append(f"paid rescue skipped: {str(e)[:160]}")
        except (TruncatedError, ValueError, ValidationError) as e:
            errors.append(f"paid rescue failed: {type(e).__name__}: {str(e)[:160]}")
    events.record("llm", f"{tag}: no valid output after {attempts} attempts: {errors[-1][:200]}", stage=tag)
    raise RuntimeError(f"{tag}: no valid output after {attempts} attempts: {errors}")
