"""Middle-out context trimming for worker summaries.

The token-level trim itself needs a loaded llama model (GPU-gated), so
these tests cover the budget arithmetic the trim relies on.
"""

from __future__ import annotations

from indexedcode.workers.hardwares import _middle_cut_bounds


def test_splits_budget_in_half_with_marker():
    head, tail = _middle_cut_bounds(1000, 200, marker_tokens=20)
    assert head + tail + 20 == 200
    assert abs(head - tail) <= 1


def test_head_keeps_extra_token_on_odd_budget():
    head, tail = _middle_cut_bounds(1000, 201, marker_tokens=20)
    assert head + tail + 20 == 201
    assert head == tail + 1


def test_marker_larger_than_budget_cuts_everything():
    assert _middle_cut_bounds(1000, 10, marker_tokens=30) == (0, 0)


def test_within_budget_is_untouched_by_caller():
    # sanity: the arithmetic never exceeds the budget it was given
    for budget, marker in ((50, 5), (64, 0), (7, 3), (1, 1)):
        head, tail = _middle_cut_bounds(999, budget, marker)
        assert head + tail + min(marker, budget) <= max(budget, 0)
