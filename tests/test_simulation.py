import random
from datetime import date, datetime, time, timedelta

import pytest

import archetypes
import simulation


# These are straight off the spreadsheet (Sheet1, rows 2-7).
SPREADSHEET = {
    "average_uk":          dict(pop=40, miles=9435,  battery=60,   eff=3.5, freq=1.0, kw=7.0, plugin=time(18, 0), plugout=time(7, 0),  target=0.8),
    "intelligent_octopus": dict(pop=30, miles=28105, battery=72.5, eff=3.5, freq=1.0, kw=7.0, plugin=time(18, 0), plugout=time(7, 0),  target=0.8),
    "infrequent_charging": dict(pop=10, miles=9435,  battery=60,   eff=3.5, freq=0.2, kw=7.0, plugin=time(18, 0), plugout=time(7, 0),  target=0.8),
    "infrequent_driving":  dict(pop=10, miles=5700,  battery=60,   eff=3.5, freq=1.0, kw=7.0, plugin=time(18, 0), plugout=time(7, 0),  target=0.8),
    "scheduled_charging":  dict(pop=9,  miles=9435,  battery=60,   eff=3.5, freq=1.0, kw=7.0, plugin=time(22, 0), plugout=time(9, 0),  target=0.8),
    "always_plugged_in":   dict(pop=1,  miles=9435,  battery=60,   eff=3.5, freq=1.0, kw=7.0, plugin=time(0, 0),  plugout=time(23, 59), target=0.8),
}


@pytest.mark.parametrize("name, s", SPREADSHEET.items())
def test_config_matches_spreadsheet_inputs(name, s):
    cfg = getattr(archetypes.ArchetypeFactory, name)()
    assert cfg.population_share * 100 == s["pop"]
    assert cfg.miles_per_year == s["miles"]
    assert cfg.battery_kwh == s["battery"]
    assert cfg.efficiency_mi_per_kwh == s["eff"]
    assert cfg.plugin_frequency_per_day == s["freq"]
    assert cfg.charger_kw == s["kw"]
    assert cfg.plugin_time == s["plugin"]
    assert cfg.plugout_time == s["plugout"]
    assert cfg.target_soc == s["target"]


@pytest.mark.parametrize("name", SPREADSHEET)
def test_population_shares_sum_to_one(name):
    total = sum(getattr(archetypes.ArchetypeFactory, n)().population_share for n in SPREADSHEET)
    assert total == pytest.approx(1.0)


def test_calendar_days_sum_to_year():
    assert archetypes.WEEKDAYS_PER_YEAR + archetypes.WEEKEND_DAYS_PER_YEAR == 365


def test_weekday_kwh_is_ratio_times_weekend():
    cfg = archetypes.ArchetypeFactory.average_uk()
    assert cfg.weekday_kwh_per_day == pytest.approx(cfg.weekday_weekend_ratio * cfg.weekend_kwh_per_day)


def test_split_trip_conserves_soc_drop():
    random.seed(0)
    for _ in range(100):
        soc_after, remaining, drop_per_slot, arrival = simulation.split_trip(0.8, 0.2, {"x": 0.85})
        assert arrival == pytest.approx(0.6)
        assert soc_after - remaining * drop_per_slot == pytest.approx(arrival)


def test_charging_schedule_sized_to_deficit():
    cfg = archetypes.ArchetypeFactory.intelligent_octopus()
    latest = simulation.latest_price_date()
    arrival = datetime.combine(latest - timedelta(days=1), time(18, 0))
    deadline = datetime.combine(latest, time(7, 0))
    schedule = simulation.build_charging_schedule(cfg, arrival, 0.45, deadline)
    assert int((schedule > 0).sum()) == 8


def test_past_price_gap_raises():
    with pytest.raises(ValueError):
        simulation._prices_with_dates(date(2020, 1, 1))


def test_sample_run_trajectory_drops_leading_nan():
    # Near the earliest date the view window starts before a run's first slot,
    # so the aligned population has leading NaN rows - these must be dropped,
    # not fed to state.name (which crashed the live app).
    import pandas as pd
    import streamlit_app
    from archetypes import State

    idx = pd.to_datetime(["2026-07-09 17:30", "2026-07-09 18:00", "2026-07-09 18:30"])
    state = pd.DataFrame({"average_uk_0": [float("nan"), State.PLUGGED_IDLE, State.DRIVING]}, index=idx)
    soc = pd.DataFrame({"average_uk_0": [float("nan"), 0.8, 0.7]}, index=idx)
    population = {"soc": soc, "state": state, "cost": soc, "weights": pd.Series({"average_uk_0": 1.0})}

    soc_run, plugged_in, state_names = streamlit_app.sample_run_trajectory(
        population, "average_uk", idx[0].to_pydatetime(), datetime(2026, 7, 9, 19, 0)
    )
    assert list(state_names) == ["PLUGGED_IDLE", "DRIVING"]
    assert len(soc_run) == 2
