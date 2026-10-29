"""The spend gate: no paid call is made without a person's say-so.

Two checks run before every call to a paid model (a catalog model with ``real: true``):

- **Approval.** The first paid call a run makes for a role, and any paid call that would take the run past
  ``budget.approve_above_usd``, is put to an approver: a function that sees what is about to be spent and returns
  True or False. The CLI asks on the terminal; the Streamlit app shows an approve or reject card; tests pass a stub.
  With no approver, every such call is refused.
- **Credit.** ``SpendLedger`` keeps real spend per provider across runs in ``<workspace>/spend.json``. A paid call is
  refused when its provider's credit is used up, or when total spend has reached the credit minus the reserve held
  back for the demo.

The estimate shown to the approver is an upper bound: the prompt's length at about four characters a token, plus
the role's full ``max_tokens`` of output.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from factory.config import Credits, ModelInfo
from factory.llm.cost import BudgetExceeded


class SpendDenied(RuntimeError):
    """A person (or the default approver) declined a paid call."""


@dataclass
class SpendRequest:
    role: str
    provider: str
    model: str
    estimate_usd: float  # upper bound for this call
    run_spent_usd: float  # real spend so far in this run
    reason: str  # why approval is being asked: "first paid call for this role" or "over the run's soft limit"


Approver = Callable[[SpendRequest], bool]


def deny_all(req: SpendRequest) -> bool:
    return False


def approve_all(req: SpendRequest) -> bool:
    return True


def estimate_usd(info: ModelInfo, prompt_chars: int, max_tokens: int) -> float:
    """Upper-bound cost of one call to a paid model."""
    return (prompt_chars / 4 * info.input + max_tokens * info.output) / 1e6


class SpendGate:
    """Asks the approver when a run first pays for a role, and again whenever it would pass the soft limit."""

    def __init__(self, approver: Approver | None, approve_above_usd: float | None):
        self.approver = approver or deny_all
        self.approve_above_usd = approve_above_usd
        self._approved_roles: set[tuple[str, str]] = set()
        self._lock = threading.Lock()

    def check(self, req: SpendRequest) -> None:
        with self._lock:
            first = (req.role, req.model) not in self._approved_roles
            over = (self.approve_above_usd is not None
                    and req.run_spent_usd + req.estimate_usd > self.approve_above_usd)
            if not (first or over):
                return
            req.reason = "over the run's soft limit" if over else "first paid call for this role"
            if not self.approver(req):
                raise SpendDenied(f"paid call to {req.model} for {req.role!r} declined ({req.reason}, "
                                  f"up to ${req.estimate_usd:.2f})")
            self._approved_roles.add((req.role, req.model))


class SpendLedger:
    """Real spend per provider, kept across runs so the semester's credit is never overdrawn."""

    def __init__(self, path: Path, credits: Credits):
        self.path = path
        self.credits = credits
        self._lock = threading.Lock()
        self.spent: dict[str, float] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def check(self, provider: str) -> None:
        with self._lock:
            limit = self.credits.by_provider.get(provider)
            if limit is not None and self.spent.get(provider, 0.0) >= limit:
                raise BudgetExceeded(f"{provider} credit used up: ${self.spent[provider]:.2f} of ${limit:.2f}")
            if self.credits.by_provider:
                usable = sum(self.credits.by_provider.values()) - self.credits.reserve_usd
                total = sum(self.spent.values())
                if total >= usable:
                    raise BudgetExceeded(f"total paid spend ${total:.2f} has reached the credit minus the "
                                         f"${self.credits.reserve_usd:.2f} reserve")

    def add(self, provider: str, usd: float) -> None:
        if usd <= 0:
            return
        with self._lock:
            self.spent[provider] = round(self.spent.get(provider, 0.0) + usd, 6)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.spent, indent=1), encoding="utf-8")
