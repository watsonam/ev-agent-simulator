from datetime import date, datetime, time

import pandas as pd

from archetypes import FlatWindow, State
from simulation import _next_occurrence, get_transition_probability, plugged_in_share


def test_next_occurrence_pushes_to_next_day_when_already_past():
    current = datetime(2026, 7, 13, 19, 30)
    result = _next_occurrence(current, date(2026, 7, 13), time(8, 47))
    assert result == datetime(2026, 7, 14, 8, 47)


def test_next_occurrence_keeps_same_day_when_still_ahead():
    current = datetime(2026, 7, 13, 6, 0)
    result = _next_occurrence(current, date(2026, 7, 13), time(8, 47))
    assert result == datetime(2026, 7, 13, 8, 47)


def test_flat_window_probability_is_zero_outside_the_window():
    window = FlatWindow(probability=0.85, start=time(13, 0), end=time(17, 0))
    assert get_transition_probability(window, time(12, 30)) == 0.0
    assert get_transition_probability(window, time(13, 0)) == 0.85
    assert get_transition_probability(window, time(17, 0)) == 0.0


def test_plugged_in_share_unweighted():
    state_df = pd.DataFrame({
        "run_0": [State.PLUGGED_CHARGING, State.DRIVING],
        "run_1": [State.PLUGGED_IDLE, State.PARKED],
    })
    result = plugged_in_share(state_df)
    assert result.iloc[0] == 1.0  # both runs plugged in
    assert result.iloc[1] == 0.0  # neither run plugged in
