from net.retry import retry


class Client:
    """Fetches from a transport that may time out."""

    def __init__(self, transport, sleep=lambda s: None):
        self.transport = transport
        self.fetch = retry(times=3, on=(TimeoutError,), delay=0.5, sleep=sleep)(self._fetch)

    def _fetch(self, path):
        return self.transport(path)
