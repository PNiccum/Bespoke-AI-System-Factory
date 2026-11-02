import pytest

from net.client import Client
from net.retry import retry


def test_backoff_and_no_wait_after_last_attempt():
    waits = []

    @retry(times=4, on=(OSError,), delay=0.5, sleep=waits.append)
    def f():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        f()
    assert waits == [0.5, 1.0, 2.0]


def test_client_retries_timeouts_only():
    seen = []

    def transport(path):
        seen.append(path)
        if len(seen) < 3:
            raise TimeoutError()
        return f"body of {path}"

    waits = []
    assert Client(transport, sleep=waits.append).fetch("/x") == "body of /x"
    assert waits == [0.5, 1.0]

    def broken(path):
        raise PermissionError(path)

    with pytest.raises(PermissionError):
        Client(broken).fetch("/y")


def test_keeps_name_and_return_value():
    @retry()
    def answer():
        return 42

    assert answer() == 42 and answer.__name__ == "answer"
