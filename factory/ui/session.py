"""A run in a background thread that a web page can watch and answer.

Streamlit reruns its script on every click, so a run cannot live inside the script. ``RunSession`` holds one run:
it starts the orchestrator on a thread, collects its events, and turns each spend-gate question into a pending
approval that the page shows as a card. The run's thread waits until a person answers, or until ``answer_timeout_s``
passes, which counts as a refusal. Nothing here imports Streamlit, so it is tested directly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from factory.config import FactoryConfig
from factory.export import export_bundle
from factory.llm import LLM, new_run_dir
from factory.llm.gate import SpendRequest
from factory.orchestrator import Event, Orchestrator, RunResult, Task
from factory.registry import Registry


@dataclass
class PendingApproval:
    request: SpendRequest
    answered: threading.Event = field(default_factory=threading.Event)
    approved: bool = False


class RunSession:
    def __init__(self, cfg: FactoryConfig, *, providers: dict | None = None, answer_timeout_s: float = 900,
                 exports_dir: Path | None = None):
        self.cfg = cfg
        self.providers = providers  # tests pass fakes; None means the configured providers
        self.answer_timeout_s = answer_timeout_s
        self.exports_dir = exports_dir or cfg.root / "exports"
        self.events: list[Event] = []
        self.pending: PendingApproval | None = None
        self.result: RunResult | None = None
        self.bundle: Path | None = None
        self.error: str | None = None
        self.llm: LLM | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, task: Task) -> None:
        if self.running:
            raise RuntimeError("a run is already in progress")
        self.events, self.result, self.bundle, self.error, self.pending = [], None, None, None, None
        run_dir = new_run_dir(self.cfg, "ui")
        self.llm = LLM(self.cfg, run_dir, providers=self.providers, approver=self._ask)
        self._thread = threading.Thread(target=self._run, args=(task, run_dir), daemon=True)
        self._thread.start()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def answer(self, approved: bool) -> None:
        """Called by the page when a person clicks Approve or Reject."""
        with self._lock:
            pending = self.pending
        if pending:
            pending.approved = approved
            pending.answered.set()

    def _ask(self, req: SpendRequest) -> bool:
        pending = PendingApproval(req)
        with self._lock:
            self.pending = pending
        answered = pending.answered.wait(self.answer_timeout_s)
        with self._lock:
            self.pending = None
        return answered and pending.approved

    def _run(self, task: Task, run_dir: Path) -> None:
        try:
            registry = Registry.load(self.cfg)
            self.result = Orchestrator(self.llm, registry, on_event=self.events.append).run(task)
            if self.result.status in ("fixed", "unproven"):
                self.bundle = export_bundle(self.result, task, run_dir, self.exports_dir)
        except Exception as e:  # shown on the page; the traceback is in the server log
            self.error = f"{type(e).__name__}: {e}"
            raise
