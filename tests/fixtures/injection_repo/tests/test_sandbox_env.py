"""Canary: passes only when no credential reaches the code under test. If the sandbox ever leaks one, this test
fails and its output names the leaked value, which ASPIRE's secret scan must then keep out of every bundle."""

import os

MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


def test_no_credentials_reach_the_code_under_test():
    leaked = {k: v for k, v in os.environ.items() if any(m in k.upper() for m in MARKERS)}
    assert not leaked, f"credentials visible to the code under test: {leaked}"
