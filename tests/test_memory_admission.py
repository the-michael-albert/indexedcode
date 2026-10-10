"""Memory budget admission control (plan §4.5).

- soft limit: pause (defer) new inference admission
- hard limit: reject new inference requests and mark memory_pressure;
  never terminate a worker or in-flight generation from this check alone
- resume only after memory stays below the soft limit for a cooldown
  interval
- report counts and a high-water mark only (no paths/source/environment)

All measurements are injected as numbers, so tests are deterministic and
no real process sampling is required.
"""

import pytest

from indexedcode.workers.memory import MemoryAdmission


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture()
def clock():
    return FakeClock()


@pytest.fixture()
def admission(clock):
    return MemoryAdmission(
        soft_mb=2300, hard_mb=2560, cooldown_s=30, clock=clock.time
    )


def test_admits_within_soft_limit(admission):
    assert admission.record(1000.0) == "admit"
    assert admission.state == "normal"
    assert admission.status()["state"] == "normal"


def test_soft_limit_defers_new_admission(admission):
    assert admission.record(2400.0) == "defer"
    assert admission.state == "deferred"


def test_hard_limit_rejects_and_marks_memory_pressure(admission):
    assert admission.record(2600.0) == "reject"
    assert admission.state == "memory_pressure"


def test_boundary_values_count_as_over_the_limit(admission):
    assert admission.record(2300.0) == "defer"
    assert MemoryAdmission(
        2300, 2560, 30, clock=lambda: 0.0
    ).record(2560.0) == "reject"


def test_pressure_never_kills_or_clears_state_by_itself(admission, clock):
    admission.record(2600.0)
    # staying above soft keeps pressure; no exception, no state loss
    assert admission.record(2500.0) == "reject"
    assert admission.state == "memory_pressure"


def test_resume_requires_below_soft_for_full_cooldown(admission, clock):
    admission.record(2400.0)  # enter deferred
    # drop below soft, but not for the full cooldown
    clock.advance(1)
    assert admission.record(1000.0) == "defer"
    clock.advance(28)  # total 29s < 30s cooldown
    assert admission.record(1000.0) == "defer"
    clock.advance(1)  # 30s reached while staying below soft
    assert admission.record(1000.0) == "admit"
    assert admission.state == "normal"


def test_excursion_above_soft_restarts_cooldown(admission, clock):
    admission.record(1000.0)
    admission.record(2400.0)  # defer
    clock.advance(20)
    admission.record(1000.0)  # below soft, cooldown running
    clock.advance(5)
    admission.record(2400.0)  # excursion restarts cooldown
    clock.advance(25)
    assert admission.record(1000.0) == "defer"
    clock.advance(5)
    assert admission.record(1000.0) == "admit"


def test_counts_and_high_water_track_all_outcomes(admission, clock):
    admission.record(2400.0)      # defer 1
    admission.record(2400.0)      # defer 2
    admission.record(2600.0)      # reject 1
    admission.record(2500.0)      # reject 2 (still >= soft)
    clock.advance(60)
    admission.record(1000.0)      # resume path
    status = admission.status()
    assert status["deferred_count"] == 2
    assert status["rejected_count"] == 2
    assert status["high_water_mb"] == 2600.0
    assert status["soft_limit_mb"] == 2300
    assert status["hard_limit_mb"] == 2560


def test_status_carries_no_environment_or_path_data(admission):
    admission.record(2600.0)
    status = admission.status()
    allowed = {
        "state", "deferred_count", "rejected_count", "high_water_mb",
        "soft_limit_mb", "hard_limit_mb", "cooldown_s",
    }
    assert set(status) <= allowed
    for value in status.values():
        assert not isinstance(value, str) or value in (
            "normal", "deferred", "memory_pressure")


def test_soft_must_be_below_hard():
    from indexedcode.workers.memory import MemoryAdmissionError

    with pytest.raises(MemoryAdmissionError):
        MemoryAdmission(soft_mb=2560, hard_mb=2300, cooldown_s=30)
    with pytest.raises(MemoryAdmissionError):
        MemoryAdmission(soft_mb=2560, hard_mb=2560, cooldown_s=30)
