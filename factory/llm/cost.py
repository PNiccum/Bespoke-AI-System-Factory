"""Pricing and per-run budgets.

Every call is priced, including calls to free ARC models, which borrow the
price of the Claude model named in their ``shadow_of`` entry. The ledger keeps
two totals: ``priced`` (what the run would cost if every model were paid) and
``real`` (what was actually spent). The dollar cap applies to ``real`` only, so a
run on free models is never stopped by a cost that does not exist; ``priced`` and
the shadow cost are reported, never enforced (see ``Budget`` in factory/config.py).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from factory.config import Budget, ModelInfo


class BudgetExceeded(RuntimeError):
    """A run cap was reached; the caller stops and reports instead of continuing."""


class StageCapReached(RuntimeError):
    """One pipeline stage used its token ceiling (``Budget.stage_max_tokens``). Unlike BudgetExceeded this ends only
    the stage: the pipeline moves on (the critic loop to final QA) instead of stopping the run."""

    def __init__(self, stage: str, used: int, limit: int):
        super().__init__(f"stage token cap reached: {stage} used {used:,} of {limit:,}")
        self.stage, self.used, self.limit = stage, used, limit


@dataclass
class Usage:
    input_tokens: int = 0  # uncached input only
    output_tokens: int = 0  # includes thinking
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


def price_call(model: ModelInfo, catalog: dict[str, ModelInfo], usage: Usage,
               batch: bool = False) -> tuple[float, float]:
    """(priced_usd, real_usd) for one call."""
    rates = model
    if not model.real:
        if model.shadow_of is None:
            return 0.0, 0.0
        rates = catalog[model.shadow_of]
    cache_read = rates.cache_read if rates.cache_read is not None else rates.input
    cache_write = rates.cache_write if rates.cache_write is not None else rates.input
    usd = (usage.input_tokens * rates.input + usage.output_tokens * rates.output
           + usage.cache_read_tokens * cache_read + usage.cache_write_tokens * cache_write) / 1e6
    if batch:
        usd *= rates.batch_discount
    return usd, (usd if model.real else 0.0)


@dataclass
class Ledger:
    budget: Budget
    priced_usd: float = 0.0
    real_usd: float = 0.0
    tokens: int = 0
    calls: int = 0
    cached_calls: int = 0
    started: float = field(default_factory=time.monotonic)
    stage: str = ""  # the stage now running (set by LLM.stage), "" outside any
    stage_tokens: int = 0  # tokens used since that stage was entered
    by_stage: dict = field(default_factory=dict)  # stage -> tokens, over every entry
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, priced: float, real: float, tokens: int, cached: bool = False) -> None:
        with self._lock:
            self.priced_usd += priced
            self.real_usd += real
            self.tokens += tokens
            if self.stage:
                self.stage_tokens += tokens
                self.by_stage[self.stage] = self.by_stage.get(self.stage, 0) + tokens
            self.calls += 1
            self.cached_calls += cached

    def check(self, paid: bool = True) -> None:
        """Raise BudgetExceeded if any cap has been reached. Called before every call. The dollar cap counts real
        spend, so it stops paid calls only: a free (ARC) call costs nothing and is never blocked by it, while with a
        cap of $0 every paid call is refused."""
        b = self.budget
        if paid and b.max_real_usd is not None and self.real_usd >= b.max_real_usd:
            raise BudgetExceeded(f"cost cap reached: ${self.real_usd:.2f} of ${b.max_real_usd:.2f} real spend")
        if b.max_tokens is not None and self.tokens >= b.max_tokens:
            raise BudgetExceeded(f"token cap reached: {self.tokens:,} of {b.max_tokens:,}")
        limit = b.stage_max_tokens.get(self.stage) if self.stage else None
        if limit is not None and self.stage_tokens >= limit:
            raise StageCapReached(self.stage, self.stage_tokens, limit)
        if b.max_seconds is not None and time.monotonic() - self.started >= b.max_seconds:
            raise BudgetExceeded(f"time cap reached: {b.max_seconds:.0f} s")

    def summary(self) -> dict:
        return {
            "calls": self.calls, "cached_calls": self.cached_calls, "tokens": self.tokens,
            "priced_usd": round(self.priced_usd, 4), "real_usd": round(self.real_usd, 4),
            "shadow_usd": round(self.priced_usd - self.real_usd, 4),  # estimate if the free models were paid

            "seconds": round(time.monotonic() - self.started, 1),
            "tokens_by_stage": dict(self.by_stage),
        }
