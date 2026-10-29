"""Find credentials in text before it leaves the machine in a bundle. No model.

Two kinds of evidence: the shapes of common keys and tokens, and the exact values of any key, token, secret or
password variable set in this process's environment (the keys in .env), so a key that appears verbatim anywhere in a
bundle is caught even if its format is unknown.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

PATTERNS = {
    "anthropic key": r"sk-ant-[A-Za-z0-9_-]{20,}",
    "openai key": r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}",
    "github token": r"(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}",
    "aws access key": r"AKIA[0-9A-Z]{16}",
    "slack token": r"xox[abposr]-[A-Za-z0-9-]{10,}",
    "private key": r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----",
    "assigned secret": r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-/+=]{16,}['\"]",
}
SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")


class SecretFound(RuntimeError):
    pass


def env_secrets() -> dict[str, str]:
    """Secret-looking environment variables with values long enough to search for."""
    return {k: v for k, v in os.environ.items()
            if any(m in k.upper() for m in SECRET_ENV_MARKERS) and len(v.strip()) >= 12}


def scan_text(text: str) -> list[str]:
    """What kinds of secret the text holds (never the secret itself)."""
    hits = [kind for kind, pattern in PATTERNS.items() if re.search(pattern, text)]
    hits += [f"value of ${name}" for name, value in env_secrets().items() if value.strip() in text]
    return hits


def guard_bundle(folder: Path) -> None:
    """Delete a bundle folder and raise if any of its files holds a secret."""
    found = {}
    for f in sorted(folder.rglob("*")):
        if f.is_file() and (hits := scan_text(f.read_text(errors="replace", encoding="utf-8"))):
            found[f.name] = hits
    if found:
        shutil.rmtree(folder)
        where = "; ".join(f"{name}: {', '.join(kinds)}" for name, kinds in found.items())
        raise SecretFound(f"bundle not written: it would contain secrets ({where})")
