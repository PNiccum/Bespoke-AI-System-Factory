"""Provenance records and JSONL helpers.

Every artifact the factory produces must trace back to what made it: the input files (by content hash), the models
and prompts that touched it (by request hash, see ``llm_calls.jsonl``), and the code that wrote it. A run keeps an
append-only ``provenance.jsonl``; an exported bundle carries a copy.

Adapted from Bespoke_LLM, where every record carried the document and chunk it came from.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Source(BaseModel):
    """One input the factory was given or fetched."""

    source_id: str  # first 12 hex chars of the sha256, stable across renames
    sha256: str
    location: str  # path or URL as given
    origin: Literal["task", "fetched", "generated"] = "task"
    licence: str | None = None
    # Whether it may go to a public_only provider (Claude, OpenAI). Unknown counts as not public.
    public: bool = False

    @classmethod
    def from_file(cls, path: Path, **kw) -> Source:
        digest = sha256_file(path)
        return cls(source_id=digest[:12], sha256=digest, location=str(path), **kw)


class ProvenanceEvent(BaseModel):
    """One step that read, wrote, or decided something in a run."""

    at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    kind: Literal["source", "model_call", "tool_call", "file_write", "approval", "note"]
    actor: str  # the agent, tool, or person responsible
    inputs: list[str] = []  # source ids, file hashes, or request hashes this step used
    outputs: list[str] = []  # what it produced, by path or hash
    detail: dict = {}


class ProvenanceLog:
    """Append-only provenance for one run (``<run_dir>/provenance.jsonl``)."""

    def __init__(self, run_dir: Path):
        self.path = run_dir / "provenance.jsonl"
        self._lock = threading.Lock()

    def record(self, event: ProvenanceEvent) -> ProvenanceEvent:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(event.model_dump_json() + "\n")
        return event

    def add_source(self, source: Source, actor: str = "intake") -> Source:
        self.record(ProvenanceEvent(kind="source", actor=actor, outputs=[source.source_id],
                                    detail=source.model_dump()))
        return source

    def events(self) -> list[ProvenanceEvent]:
        return list(read_jsonl(self.path, ProvenanceEvent)) if self.path.exists() else []


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path, model: type[T]) -> Iterator[T]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield model.model_validate_json(line)


def write_manifest(work_dir: Path, stage: str, **info) -> None:
    """Record what produced a stage's outputs so runs are reproducible."""
    info = {"stage": stage, "finished_at": datetime.now(UTC).isoformat(), **info}
    path = work_dir / "manifests" / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, default=str))
