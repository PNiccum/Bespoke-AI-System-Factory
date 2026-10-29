"""Cached, throttled, budgeted HTTP for any JSON or file API (lifted from Bespoke_LLM's literature clients).

Every JSON response is cached on disk keyed by its URL (never the key), so re-running a round costs no calls until
an entry is older than ``max_age_days``. Each client has a call budget per round; reaching it raises
``BudgetExhausted``, which ends the round cleanly at its checkpoint instead of running on.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import httpx


class BudgetExhausted(RuntimeError):
    pass


class CallBudget:
    def __init__(self, name: str, limit: int):
        self.name, self.limit, self.used = name, limit, 0
        self._lock = threading.Lock()

    def spend(self) -> None:
        with self._lock:
            if self.used >= self.limit:
                raise BudgetExhausted(f"{self.name}: {self.limit} calls this round")
            self.used += 1


class CachedAPI:
    base: str = ""
    min_interval: float = 0.2  # <= 5 requests/second across all threads (both APIs throttle bursts)

    def __init__(self, cache_dir: Path, budget: CallBudget, headers: dict | None = None, max_age_days: float = 30,
                 http: httpx.Client | None = None, auth_params: dict | None = None):
        self.cache = cache_dir
        self.budget, self.headers = budget, headers or {}
        self.auth_params = auth_params or {}  # added to the request only, never to the cache key
        self.max_age = timedelta(days=max_age_days)
        self.http = http or httpx.Client(timeout=120, follow_redirects=True)
        self._lock, self._last = threading.Lock(), 0.0

    def _throttle(self):
        with self._lock:
            wait = self._last + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def _request(self, url: str, headers: dict) -> httpx.Response:
        self.budget.spend()
        for attempt in range(8):
            self._throttle()
            try:
                r = self.http.get(url, headers=headers)
            except httpx.TransportError:
                time.sleep(min(30, 2 ** attempt))
                continue
            if r.status_code == 429 and "budget" in r.text.lower():  # a daily allowance is spent: a cap, not a blip
                raise BudgetExhausted(f"{self.base or 'download'}: daily allowance spent ({r.text[:120]})")
            if r.status_code == 429 or r.status_code >= 500:  # throttled or flaky
                time.sleep(min(30, 2 ** attempt))
                continue
            return r
        raise RuntimeError(f"gave up after 8 attempts: {url.split('?')[0]}")

    def get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self.base}/{path}" + (f"?{urlencode(sorted(params.items()))}" if params else "")
        cache_path = self.cache / f"{hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if datetime.fromisoformat(cached["fetched_at"]) > datetime.now(UTC) - self.max_age:
                return cached["data"]
        sent = url + (("&" if "?" in url else "?") + urlencode(self.auth_params) if self.auth_params else "")
        r = self._request(sent, {"Accept": "application/json", **self.headers})
        if r.status_code == 404:  # unknown id: cached as empty so it is not asked again
            data = {}
        else:
            if r.status_code >= 400:
                raise httpx.HTTPStatusError(f"{r.status_code} {r.text[:300]}", request=r.request, response=r)
            data = r.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"url": url, "fetched_at": datetime.now(UTC).isoformat(),
                                          "data": data}), encoding="utf-8")
        return data

    def download(self, url: str, accept: str = "application/pdf", headers: dict | None = None) -> bytes | None:
        """A file, uncached (the caller files it). None unless the reply is a PDF."""
        r = self._request(url, {"Accept": accept, "User-Agent": "Mozilla/5.0 (bespoke-ai-system-factory)",
                                **(headers or {})})
        if r.status_code != 200 or not r.content.startswith(b"%PDF"):
            return None
        return r.content
