"""Find the code an issue is about, without a model.

The issue text is mined for names a person would write when reporting a bug: file paths, traceback frames,
``function_names()``, ``ClassNames`` and words in backticks. Each Python file in the repository is then scored by
where those names appear: the innermost traceback frame (where the error was raised) scores highest, then the outer
frames and paths named outright, then a definition (``def`` or ``class``), and a plain mention least. The result is a short ranked list the coder reads first.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist", ".tox", ".eggs"}
WEIGHTS = {"raised_at": 10.0, "frame": 6.0, "path": 6.0, "definition": 4.0, "mention": 1.0}
STOPWORDS = {"self", "none", "true", "false", "return", "import", "error", "test", "tests", "python", "file", "line"}


class LocateIn(BaseModel):
    repo_path: str = Field(description="Root of the repository")
    issue_text: str = Field(description="The bug report or task, as written")
    limit: int = Field(10, ge=1, le=50)


class Hit(BaseModel):
    path: str  # relative to the repository root
    line: int  # 1-based; the strongest evidence in the file
    score: float
    why: list[str]  # e.g. ["definition of parse_date", "mentions parse_date"]
    is_test: bool  # a test file: shows how the bug is exercised, rarely where it is fixed


class Locations(BaseModel):
    names: list[str]  # what was looked for, so a person can see why a file was picked
    hits: list[Hit]


def issue_names(text: str) -> tuple[list[str], list[tuple[str, int]], list[str]]:
    """(identifiers, traceback frames as (path, line), file paths) mentioned in an issue."""
    frames = [(p, int(n)) for p, n in re.findall(r'File "([^"]+\.py)", line (\d+)', text)]
    paths = sorted(set(re.findall(r"[\w./-]+\.py\b", text)) - {p for p, _ in frames})
    names = set(re.findall(r"`([A-Za-z_][\w.]*)`", text))
    names |= set(re.findall(r"\b([A-Za-z_]\w*)\(", text))  # called: parse_date(
    names |= {w for w in re.findall(r"\b[A-Za-z_]\w*\b", text)
              if "_" in w.strip("_") or re.fullmatch(r"(?:[A-Z][a-z0-9]+){2,}", w)}  # snake_case, CamelCase
    names = {n.split(".")[-1] for n in names}
    return sorted(n for n in names if len(n) > 2 and n.lower() not in STOPWORDS), frames, paths


def is_test_file(rel: str) -> bool:
    parts = rel.split("/")
    return parts[-1].startswith("test_") or parts[-1].endswith("_test.py") or "tests" in parts[:-1]


def python_files(root: Path):
    for p in sorted(root.rglob("*.py")):
        if not SKIP_DIRS & set(p.relative_to(root).parts[:-1]):
            yield p


def locate(inp: LocateIn) -> Locations:
    root = Path(inp.repo_path).resolve()
    names, frames, paths = issue_names(inp.issue_text)
    # path -> [total score, weight of the strongest evidence, its line, reasons]
    scored: dict[str, list] = {}

    def add(rel: str, weight: float, line: int, why: str):
        entry = scored.setdefault(rel, [0.0, 0.0, line, []])
        if why in entry[3]:  # each reason counts once per file, however often the name recurs
            return
        entry[3].append(why)
        entry[0] += weight
        if weight > entry[1]:
            entry[1], entry[2] = weight, line

    for f in python_files(root):
        rel = f.relative_to(root).as_posix()
        for i, (frame_path, n) in enumerate(frames):
            if frame_path.replace("\\", "/").endswith(rel):
                raised = i == len(frames) - 1
                add(rel, WEIGHTS["raised_at" if raised else "frame"], n,
                    f"error raised at line {n}" if raised else f"traceback frame at line {n}")
        for p in paths:
            if p.replace("\\", "/").endswith(rel):
                add(rel, WEIGHTS["path"], 1, f"named in the issue as {p}")
        if not names:
            continue
        try:
            lines = f.read_text(errors="replace", encoding="utf-8").splitlines()
        except OSError:
            continue
        for i, text in enumerate(lines, 1):
            for name in names:
                if name not in text:
                    continue
                if re.match(rf"\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b", text):
                    add(rel, WEIGHTS["definition"], i, f"definition of {name}")
                elif re.search(rf"\b{re.escape(name)}\b", text):
                    add(rel, WEIGHTS["mention"], i, f"mentions {name}")

    hits = [Hit(path=p, line=line, score=round(total, 2), why=why, is_test=is_test_file(p))
            for p, (total, _, line, why) in scored.items()]
    hits.sort(key=lambda h: (-h.score, h.path))
    return Locations(names=names, hits=hits[:inp.limit])
