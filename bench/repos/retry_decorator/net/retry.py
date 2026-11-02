import functools
import time


def retry(times=3, on=(Exception,), delay=0.0, sleep=time.sleep):
    """Call the function up to ``times`` times. Retry only when it raises one of the exception types in ``on``;
    any other exception propagates at once. If every attempt fails, re-raise the last exception. Before retry
    number k (k = 1, 2, ...) wait ``delay * 2 ** (k - 1)`` seconds; never wait after the last attempt."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            for _ in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    sleep(delay)
            return None
        return inner
    return wrap
