"""Not real tests - a checklist of what's worth covering next. Each stub is
skipped; delete the skip and fill in the body when you write it for real."""

import pytest


@pytest.mark.skip(reason="TODO")
def test_advance_never_produces_soc_outside_zero_and_target():
    """Run advance() for a few archetypes over a few days; assert
    0 <= soc <= target_soc for every row. Catches ramp/drop arithmetic bugs."""


@pytest.mark.skip(reason="TODO")
def test_advance_produces_one_row_per_30_minutes_with_no_gaps():
    """The returned DataFrame's index should be a complete 30-min grid from
    start to end_time, no missing or duplicate timestamps."""


@pytest.mark.skip(reason="TODO")
def test_charging_schedule_picks_exactly_the_slots_needed():
    """build_charging_schedule's number of charging slots should match
    ceil(kwh_needed / energy_per_slot), not more, not fewer."""


@pytest.mark.skip(reason="TODO")
def test_charging_schedule_picks_the_cheapest_available_slots():
    """Given a known price Series, assert the chosen slots are exactly the
    N cheapest - not just the first N by time."""


@pytest.mark.skip(reason="TODO")
def test_resuming_a_run_state_continues_from_the_same_point():
    """advance() to time T, then advance() the returned RunState to time T+1
    should give the same result as advancing straight to T+1 in one call
    (mock random() to make both paths deterministic)."""


@pytest.mark.skip(reason="TODO")
def test_get_or_advance_run_does_not_reduce_or_lose_history():
    """Calling get_or_advance_run twice with a later end_dt the second time
    should only add rows, never change or drop earlier ones."""


@pytest.mark.skip(reason="TODO")
def test_weighted_quantiles_matches_plain_quantile_when_weights_are_equal():
    """weighted_quantiles(df, uniform_weights, [0.5]) should equal
    df.median(axis=1) when every column has the same weight."""


@pytest.mark.skip(reason="TODO")
def test_population_weights_sum_to_one():
    """get_population_runs(...)["weights"].sum() should be ~1.0 regardless
    of runs_per_archetype."""


@pytest.mark.skip(reason="TODO")
def test_require_rows_raises_a_clear_error_on_empty_window():
    """dashboard._require_rows should raise ValueError (not let an empty
    DataFrame reach numpy) when the window predates the cache."""


@pytest.mark.skip(reason="TODO")
def test_slice_window_is_start_inclusive_end_exclusive():
    """A row exactly at `start` should be included; a row exactly at `end`
    should not."""
