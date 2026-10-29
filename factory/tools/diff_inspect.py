"""Read a pull request's diff in plain code: what it changes, and the problems a rule can find. No model.

``inspect`` parses a unified diff into the files and lines it changes and runs fixed rules over the added lines:
leftover debugging, hard-coded credentials, bare ``except:``, new TODOs, source changed with no test touched, and
Python files that no longer parse once the patch is applied. These are facts, so the model reviewer is told them
rather than asked to find them, and a model comment is kept only if it points at a line the diff changes.

This module uses only the standard library, so an exported bundle carries a copy and re-runs the inspection itself.
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

RULES = [  # (rule, severity, pattern over an added line, message); test files skip the debug rules
    ("debug_print", "minor", re.compile(r"^\s*print\("), "leftover print() call"),
    ("debugger", "major", re.compile(r"\bbreakpoint\(\)|\bpdb\.set_trace\(|^\s*import i?pdb\b"),
     "debugger call left in"),
    ("secret", "blocker",
     re.compile(r"""(?i)\b\w*(?:key|token|secret|password)\w*\s*[:=]\s*["'][^"'\s]{8,}["']|\bsk-[A-Za-z0-9_-]{16,}"""),
     "looks like a hard-coded credential"),
    ("bare_except", "minor", re.compile(r"^\s*except\s*:"), "bare except: hides every error"),
    ("todo", "info", re.compile(r"#\s*(?:TODO|FIXME|XXX)\b"), "new TODO"),
]
DEBUG_RULES = {"debug_print", "debugger"}


@dataclass
class FileChange:
    path: str  # the new path ("" when the file is deleted)
    old_path: str
    added: dict[int, str] = field(default_factory=dict)  # new line number -> text
    removed: int = 0
    hunks: list[tuple[int, int]] = field(default_factory=list)  # (first new line, count)

    def touches(self, line: int) -> bool:
        """The line is added by the diff or sits inside one of its hunks."""
        return line in self.added or any(start <= line < start + max(count, 1) for start, count in self.hunks)


@dataclass
class Finding:
    rule: str
    severity: str  # blocker, major, minor, info
    path: str
    line: int
    message: str
    text: str = ""


def is_test_file(path: str) -> bool:
    parts = path.split("/")
    return parts[-1].startswith("test_") or parts[-1].endswith("_test.py") or "tests" in parts[:-1]


def parse_diff(patch: str) -> list[FileChange]:
    files: list[FileChange] = []
    cur, new_line = None, 0
    old = ""
    for raw in patch.splitlines():
        if raw.startswith("--- "):
            old = raw[4:].split("\t")[0]
            old = "" if old == "/dev/null" else old.removeprefix("a/")
        elif raw.startswith("+++ "):
            new = raw[4:].split("\t")[0]
            cur = FileChange(path="" if new == "/dev/null" else new.removeprefix("b/"), old_path=old)
            files.append(cur)
        elif raw.startswith("@@") and cur is not None:
            m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", raw)
            if m:
                new_line = int(m.group(1))
                cur.hunks.append((new_line, int(m.group(2) or 1)))
        elif cur is not None and cur.hunks:
            if raw.startswith("+"):
                cur.added[new_line] = raw[1:]
                new_line += 1
            elif raw.startswith("-"):
                cur.removed += 1
            elif not raw.startswith("\\"):
                new_line += 1
    return files


def syntax_errors(repo_path: str, patch: str, files: list[FileChange]) -> tuple[bool, list[Finding]]:
    """(patch applied, a finding per changed Python file that no longer parses)."""
    with tempfile.TemporaryDirectory(prefix="factory-inspect-") as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(Path(repo_path).resolve(), work,
                        ignore=shutil.ignore_patterns(".git", ".venv*", "__pycache__", ".env", ".env.*"))
        applied = any(subprocess.run(["git", "apply", *extra, "--whitespace=nowarn", "-"], cwd=work, input=patch,
                                     text=True, capture_output=True).returncode == 0
                      for extra in ([], ["--recount"]))
        if not applied:
            return False, []
        out = []
        for f in files:
            if f.path.endswith(".py") and (work / f.path).exists():
                try:
                    ast.parse((work / f.path).read_text(errors="replace", encoding="utf-8"))
                except SyntaxError as e:
                    out.append(Finding("syntax", "blocker", f.path, e.lineno or 1, f"does not parse: {e.msg}"))
        return True, out


def inspect(patch: str, repo_path: str | None = None) -> dict:
    """Everything plain code can say about a diff, as JSON-ready data."""
    files = parse_diff(patch)
    findings: list[Finding] = []
    for f in files:
        test = is_test_file(f.path)
        for n, text in f.added.items():
            for rule, severity, pattern, message in RULES:
                if test and rule in DEBUG_RULES:
                    continue
                if pattern.search(text):
                    findings.append(Finding(rule, severity, f.path, n, message, text.strip()[:120]))
    source = [f for f in files if f.path.endswith(".py") and not is_test_file(f.path)]
    if source and not any(is_test_file(f.path) for f in files):
        findings.append(Finding("no_tests", "info", source[0].path, next(iter(source[0].added), 1),
                                "source changed, but no test was added or changed"))
    applied = None
    if repo_path:
        applied, errors = syntax_errors(repo_path, patch, files)
        findings += errors
    return {
        "files": [{"path": f.path or f.old_path, "added": len(f.added), "removed": f.removed,
                   "test": is_test_file(f.path or f.old_path)} for f in files],
        "applied": applied,
        "findings": [asdict(x) for x in findings],
    }


if __name__ == "__main__":  # python diff_inspect.py change.patch [REPO]
    import json

    print(json.dumps(inspect(Path(sys.argv[1]).read_text(encoding="utf-8"), sys.argv[2] if len(sys.argv) > 2 else None), indent=2))
