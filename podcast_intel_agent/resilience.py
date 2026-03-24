"""Retries with exponential backoff + jitter, and a sync token bucket for LLM throttling."""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_sync(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> T:
    """Run ``fn`` until it succeeds or ``max_attempts`` is exhausted."""
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts - 1:
                raise
            delay = min(max_delay, base_delay * (2**attempt))
            delay *= 0.5 + random.random()
            time.sleep(delay)
    raise RuntimeError("retry_sync fell through") from last_exc


class TokenBucket:
    """Simple refillable token bucket (thread-safe)."""

    def __init__(self, *, capacity: float, refill_per_second: float) -> None:
        self._capacity = capacity
        self._refill_per_second = refill_per_second
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        if tokens <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._last = now
                if self._refill_per_second > 0:
                    self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_second)
                else:
                    self._tokens = self._capacity
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self._refill_per_second if self._refill_per_second > 0 else 0.25
            time.sleep(min(max(wait, 0.01), 2.0))
