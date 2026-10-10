"""Memory budget admission control (plan §4.5).

- at the soft limit: pause (defer) new inference admission
- at the hard limit: reject new inference requests and mark
  ``memory_pressure``; never terminate a worker or in-flight generation
  from this check alone
- resume only after memory has stayed below the soft limit for the full
  cooldown interval; any excursion back above soft restarts the cooldown

Measurements are supplied by the caller (sampled RSS plus reported
accelerator allocation), so this module is deterministic and testable.
Status carries only low-cardinality state and numeric counters.
"""

from __future__ import annotations

import time

from indexedcode.workers.errors import MemoryAdmissionError  # noqa: F401


class MemoryAdmission:
    """Admission decisions for new inference requests."""

    def __init__(self, soft_mb, hard_mb, cooldown_s, clock=time.monotonic):
        if soft_mb <= 0 or hard_mb <= 0 or cooldown_s <= 0:
            raise MemoryAdmissionError(
                "soft/hard limits and cooldown must be positive"
            )
        if soft_mb >= hard_mb:
            raise MemoryAdmissionError(
                f"soft limit ({soft_mb} MB) must be below the hard limit"
                f" ({hard_mb} MB)"
            )
        self._soft_mb = soft_mb
        self._hard_mb = hard_mb
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._state = "normal"
        # timestamp of the most recent reading at/above the soft limit
        self._last_over_soft_at = None
        self._deferred_count = 0
        self._rejected_count = 0
        self._high_water_mb = 0.0

    @property
    def state(self) -> str:
        return self._state

    @property
    def deferred_count(self) -> int:
        return self._deferred_count

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    def record(self, total_mb: float) -> str:
        """Feed one measurement; return "admit", "defer", or "reject"."""
        now = self._clock()
        if total_mb > self._high_water_mb:
            self._high_water_mb = total_mb

        if total_mb >= self._hard_mb:
            self._last_over_soft_at = now
            self._state = "memory_pressure"
            self._rejected_count += 1
            return "reject"

        if total_mb >= self._soft_mb:
            self._last_over_soft_at = now
            if self._state == "memory_pressure":
                self._rejected_count += 1
                return "reject"
            self._state = "deferred"
            self._deferred_count += 1
            return "defer"

        # below the soft limit
        if self._state == "normal":
            return "admit"
        if (
            self._last_over_soft_at is not None
            and now - self._last_over_soft_at >= self._cooldown_s
        ):
            self._state = "normal"
            self._last_over_soft_at = None
            return "admit"
        self._deferred_count += 1
        return "defer"

    def status(self) -> dict:
        return {
            "state": self._state,
            "deferred_count": self._deferred_count,
            "rejected_count": self._rejected_count,
            "high_water_mb": self._high_water_mb,
            "soft_limit_mb": self._soft_mb,
            "hard_limit_mb": self._hard_mb,
            "cooldown_s": self._cooldown_s,
        }
