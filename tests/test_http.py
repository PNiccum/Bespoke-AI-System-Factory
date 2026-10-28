"""The cached HTTP client: hard call budgets, an on-disk cache, and keys kept out of it."""

import httpx
import pytest

from factory.net.http import BudgetExhausted, CachedAPI, CallBudget


def mock(handler):
    calls = []

    def h(request):
        calls.append(str(request.url))
        return handler(request)
    return httpx.Client(transport=httpx.MockTransport(h)), calls


def test_budget_is_a_hard_stop():
    b = CallBudget("x", 2)
    b.spend(), b.spend()
    with pytest.raises(BudgetExhausted):
        b.spend()


def test_responses_are_cached_and_404s_are_not_asked_again(tmp_path):
    http, calls = mock(lambda r: httpx.Response(404) if "missing" in str(r.url) else httpx.Response(200, json={"a": 1}))
    api = CachedAPI(tmp_path, CallBudget("t", 10), http=http)
    api.base = "https://example.org"
    assert api.get("x", {"q": 1}) == api.get("x", {"q": 1}) == {"a": 1}
    assert api.get("missing") == api.get("missing") == {}
    assert len(calls) == 2


def test_a_spent_daily_allowance_is_a_cap_and_keys_stay_out_of_the_cache(tmp_path):
    http, calls = mock(lambda r: httpx.Response(429, text="daily budget exhausted"))
    api = CachedAPI(tmp_path, CallBudget("t", 10), http=http, auth_params={"api_key": "secret"})
    api.base = "https://example.org"
    with pytest.raises(BudgetExhausted):
        api.get("x")
    assert len(calls) == 1 and "api_key=secret" in calls[0]
    http, _ = mock(lambda r: httpx.Response(200, json={}))
    api.http = http
    api.get("x")
    assert "secret" not in "".join(p.read_text() for p in tmp_path.glob("*.json"))
