"""Selecting agents by contract net, with bids computed from specs and history instead of model calls.

The planner announces a subtask (a capability, a domain, a task type). Every registered agent that has the
capability bids; the best bid is awarded the work. A bid is a score built from facts the factory already holds,
so selection costs nothing and gives the same answer every time:

- domain fit: the agent names the task's domain, or works in any domain;
- track record: the share of past runs of this task type the agent passed (unknown counts as even odds);
- cost: plain code beats a free model, and a free model beats a paid one.

Asking each agent to judge its own fit with a model call would spend credit on every announcement and trust a
model's view of itself, which is poorly calibrated.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from factory.config import FactoryConfig
from factory.registry.spec import AgentSpec

WEIGHTS = {"domain": 0.3, "record": 0.5, "tool": 0.3, "free": 0.2}


class Announcement(BaseModel):
    capability: str
    domain: str = "any"
    task_type: str = ""  # e.g. "bug_fix"; keys the track record


@dataclass
class Bid:
    agent: AgentSpec
    score: float
    parts: dict[str, float]  # what the score is made of, for the run log and the Streamlit view


class ScoreBook:
    """Pass and run counts per agent and task type, kept across runs (``<registry>/scores.json``)."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self.counts: dict[str, dict[str, list[int]]] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def record(self, agent: str, task_type: str, passed: bool) -> None:
        with self._lock:
            c = self.counts.setdefault(agent, {}).setdefault(task_type, [0, 0])
            c[0] += passed
            c[1] += 1
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.counts, indent=1), encoding="utf-8")

    def pass_rate(self, agent: str, task_type: str) -> float | None:
        passed, runs = self.counts.get(agent, {}).get(task_type, [0, 0])
        return passed / runs if runs else None


class Registry:
    def __init__(self, agents: list[AgentSpec], cfg: FactoryConfig, scores: ScoreBook):
        names = [a.name for a in agents]
        if dupes := sorted({n for n in names if names.count(n) > 1}):
            raise ValueError(f"duplicate agent names: {dupes}")
        for a in agents:
            if a.role and a.role not in cfg.roles:
                raise ValueError(f"agent {a.name!r} uses role {a.role!r}, which the config does not define")
        self.agents = {a.name: a for a in agents}
        self.cfg = cfg
        self.scores = scores

    @classmethod
    def load(cls, cfg: FactoryConfig, directory: Path | None = None) -> Registry:
        directory = directory or cfg.root / cfg.registry_dir
        specs = [AgentSpec.load(p) for p in sorted((directory / "agents").glob("*.yaml"))]
        return cls(specs, cfg, ScoreBook(directory / "scores.json"))

    def version(self) -> dict:
        """What the awards depend on: a hash of every agent spec and a snapshot of the track record. Written into
        each run and bundle, so a run on another machine can be checked against the registry that made it."""
        specs = json.dumps({n: a.model_dump(mode="json") for n, a in sorted(self.agents.items())}, sort_keys=True)
        scores = json.dumps(self.scores.counts, sort_keys=True)
        return {"specs_sha256": hashlib.sha256(specs.encode()).hexdigest(),
                "scores_sha256": hashlib.sha256(scores.encode()).hexdigest(),
                "agents": sorted(self.agents), "scores": json.loads(scores)}

    def is_free(self, agent: AgentSpec) -> bool:
        return agent.kind == "tool" or not self.cfg.model_info(self.cfg.roles[agent.role].model).real

    def bids(self, ann: Announcement) -> list[Bid]:
        """Every bid for an announcement, best first. Agents without the capability do not bid."""
        out = []
        for a in self.agents.values():
            if ann.capability not in a.capabilities:
                continue
            rate = self.scores.pass_rate(a.name, ann.task_type)
            parts = {
                "domain": WEIGHTS["domain"] if ann.domain in a.domains or "any" in a.domains else 0.0,
                "record": WEIGHTS["record"] * (0.5 if rate is None else rate),
                "tool": WEIGHTS["tool"] if a.kind == "tool" else 0.0,
                "free": WEIGHTS["free"] if self.is_free(a) else 0.0,
            }
            out.append(Bid(a, round(sum(parts.values()), 4), parts))
        return sorted(out, key=lambda b: (-b.score, b.agent.name))  # name breaks ties, so awards are repeatable

    def award(self, ann: Announcement) -> Bid | None:
        bids = self.bids(ann)
        return bids[0] if bids else None
