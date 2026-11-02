import pytest

from net.retry import retry


def test_other_exceptions_are_not_retried():
    calls = []

    @retry(times=3, on=(TimeoutError,), sleep=lambda s: None)
    def f():
        calls.append(1)
        raise KeyError("x")

    with pytest.raises(KeyError):
        f()
    assert len(calls) == 1


def test_last_exception_is_raised():
    @retry(times=2, on=(TimeoutError,), sleep=lambda s: None)
    def f():
        raise TimeoutError("slow")

    with pytest.raises(TimeoutError):
        f()


def test_success_after_a_retry():
    results = iter([TimeoutError(), "ok"])

    @retry(times=3, on=(TimeoutError,), sleep=lambda s: None)
    def f():
        r = next(results)
        if isinstance(r, Exception):
            raise r
        return r

    assert f() == "ok"
