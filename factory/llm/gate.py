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

import contextlib
import json
import os
import threading
import time
import uuid
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


def _retry_on_permission_error(fn, tries: int = 50, wait_s: float = 0.02):
    """Windows refuses to replace or read a file another process has open at that instant; try again briefly."""
    for attempt in range(tries):
        try:
            return fn()
        except PermissionError:
            if attempt == tries - 1:
                raise
            time.sleep(wait_s)


@contextlib.contextmanager
def _file_lock(path: Path):
    """An exclusive lock on ``path`` shared by every process on this machine (fcntl, or msvcrt on Windows)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as f:
        if os.name == "nt":
            import msvcrt
            for attempt in range(30):  # each try blocks ~10 s; a persistent error is raised, not spun on
                try:
                    f.seek(0)  # msvcrt locks from the current position: lock and unlock the same byte
                    msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    if attempt == 29:
                        raise
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


RESERVATION_TTL_S = 2 * 3600  # one call's default; a batch passes its own wait limit. A crashed run's hold expires after this


class SpendLedger:
    """Real spend per provider, kept across runs so the semester's credit is never overdrawn.

    Safe for parallel runs. Every read-modify-write happens under a file lock and rereads ``spend.json`` first, so
    two processes cannot overwrite each other's totals, and the file is replaced atomically. A paid call first
    ``reserve``s its upper-bound estimate (kept in ``spend.reserved.json``); the credit check counts spend plus every
    live reservation, so calls that start together cannot all pass before any is charged. ``settle`` swaps the
    reservation for the real cost; ``release`` drops it when the call fails.
    """

    def __init__(self, path: Path, credits: Credits):
        self.path = path
        self.credits = credits
        self.reserved_path = path.with_name(path.stem + ".reserved.json")
        self._lock = threading.Lock()
        self._flock = path.with_name(path.stem + ".lock")

    @property
    def spent(self) -> dict[str, float]:
        """Current totals, read fresh from disk (other processes may have added to them)."""
        return self._read_spent()  # lock-free: writes are atomic replaces, so a read never sees a partial file

    @contextlib.contextmanager
    def _locked(self):
        with self._lock, _file_lock(self._flock):
            yield

    def _read_spent(self) -> dict[str, float]:
        try:
            spent = json.loads(_retry_on_permission_error(lambda: self.path.read_text(encoding="utf-8")))
            if not isinstance(spent, dict):
                raise ValueError("not a JSON object")
            return spent
        except FileNotFoundError:
            return {}
        except ValueError as e:  # fail closed: an unreadable ledger must not look like zero spend
            raise BudgetExceeded(f"{self.path} is unreadable ({e}); fix or restore it before paid calls") from e

    def _read_reserved(self) -> dict[str, dict]:
        try:
            held = json.loads(self.reserved_path.read_text(encoding="utf-8"))
            now = time.time()
            for v in held.values():
                v["provider"], v["usd"], v["at"]  # a malformed entry raises KeyError here, below, as unreadable
            return {k: v for k, v in held.items() if now - v["at"] < v.get("ttl", RESERVATION_TTL_S)}
        except FileNotFoundError:
            return {}
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise BudgetExceeded(f"{self.reserved_path} is unreadable ({e}); delete it if no run is in flight") from e

    @staticmethod
    def _write(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        _retry_on_permission_error(lambda: os.replace(tmp, path))

    def _refuse_if_over(self, provider: str, spent: dict[str, float], held: dict[str, dict], estimate: float,
                        ignoring: str | None = None) -> None:
        held = {k: v for k, v in held.items() if k != ignoring}
        by = {}
        for h in held.values():
            by[h["provider"]] = by.get(h["provider"], 0.0) + h["usd"]
        used = spent.get(provider, 0.0) + by.get(provider, 0.0)
        limit = self.credits.by_provider.get(provider)
        if limit is not None and (used + estimate > limit + 1e-9 or used >= limit):
            raise BudgetExceeded(f"{provider} credit used up: ${used:.2f} spent or held plus ${estimate:.2f} "
                                 f"for this call would pass ${limit:.2f}")
        if self.credits.by_provider:
            usable = sum(self.credits.by_provider.values()) - self.credits.reserve_usd
            total = sum(spent.values()) + sum(by.values())
            if total + estimate > usable + 1e-9 or total >= usable:
                raise BudgetExceeded(f"total paid spend ${total:.2f} spent or held plus ${estimate:.2f} for this "
                                     f"call would pass the credit minus the ${self.credits.reserve_usd:.2f} reserve")

    def check(self, provider: str, estimate: float = 0.0, ignoring: str | None = None) -> None:
        """Refuse if this call could not be paid for. Does not hold anything; use ``reserve`` before a paid call."""
        with self._locked():
            self._refuse_if_over(provider, self._read_spent(), self._read_reserved(), estimate, ignoring)

    def reserve(self, provider: str, estimate: float, ttl_s: float = RESERVATION_TTL_S,
                replaces: str | None = None) -> str:
        """Check and hold ``estimate`` in one step; returns the reservation id for ``settle`` or ``release``.
        The hold lapses after ``ttl_s`` (so a crashed run cannot block credit for ever); set it past the call's wait.
        ``replaces`` names an earlier hold for the same work: it is ignored by the check and dropped in the same
        locked write, so a refusal leaves it in place."""
        with self._locked():
            held = self._read_reserved()
            self._refuse_if_over(provider, self._read_spent(), held, estimate, ignoring=replaces)
            held.pop(replaces, None)
            rid = uuid.uuid4().hex
            held[rid] = {"provider": provider, "usd": round(estimate, 6), "at": time.time(), "ttl": ttl_s}
            self._write(self.reserved_path, held)
            return rid

    def release(self, rid: str) -> None:
        with self._locked():
            held = self._read_reserved()
            if held.pop(rid, None) is not None:
                self._write(self.reserved_path, held)

    def settle(self, rid: str | None, provider: str, usd: float, part: float | None = None) -> None:
        """Charge the real cost and drop the reservation (all of it, or just ``part`` dollars of it), in one locked
        step. Nothing to do, and no lock taken, for a free call."""
        if (rid is None or part == 0) and usd <= 0:
            return
        with self._locked():
            held = self._read_reserved() if rid is not None else {}  # read first: a corrupt file raises before any write
            if usd > 0:  # spend first: if a write fails, the hold is still there and the credit stays covered
                spent = self._read_spent()
                spent[provider] = round(spent.get(provider, 0.0) + usd, 6)
                self._write(self.path, spent)
            if rid is not None:
                if rid in held:
                    left = None if part is None else round(held[rid]["usd"] - part, 6)
                    if left is None or left <= 0:
                        del held[rid]
                    else:
                        held[rid]["usd"] = left
                    self._write(self.reserved_path, held)

    def add(self, provider: str, usd: float) -> None:
        self.settle(None, provider, usd)
