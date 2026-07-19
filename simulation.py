import pickle
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from math import ceil
from pathlib import Path
from random import gauss, random
from typing import TypedDict, cast
from uuid import uuid4

import numpy as np
import pandas as pd

from archetypes import (
    ArchetypeConfig,
    ArchetypeFactory,
    ChargingStrategy,
    FlatWindow,
    GaussianDeparture,
    State,
)

MARKET_INDEX_CSV = Path(__file__).parent / "data" / "market_index.csv"
CACHE_DIR = Path(__file__).parent / "data" / "cache"
RUN_CACHE_VERSION = 2
RUN_CACHE_DIR = CACHE_DIR / f"runs_v{RUN_CACHE_VERSION}"
LOOKBACK_DAYS = 7


class PopulationResult(TypedDict):
    soc: pd.DataFrame
    state: pd.DataFrame
    cost: pd.DataFrame
    weights: pd.Series


_ARCHETYPES = {
    "average_uk": ArchetypeFactory.average_uk,
    "intelligent_octopus": ArchetypeFactory.intelligent_octopus,
    "infrequent_charging": ArchetypeFactory.infrequent_charging,
    "infrequent_driving": ArchetypeFactory.infrequent_driving,
    "scheduled_charging": ArchetypeFactory.scheduled_charging,
    "always_plugged_in": ArchetypeFactory.always_plugged_in,
}


def get_day_type(d: date) -> str:
    return "weekend" if d.weekday() >= 5 else "weekday"


def get_archetype(name: str) -> ArchetypeConfig:
    try:
        return _ARCHETYPES[name]()
    except KeyError:
        raise ValueError(f"Unknown archetype '{name}'. Options: {list(_ARCHETYPES)}")


@lru_cache(maxsize=1)
def _load_price_data(_mtime: float) -> pd.DataFrame:
    return pd.read_csv(MARKET_INDEX_CSV, parse_dates=["settlementDate"])


def _price_data() -> pd.DataFrame:
    return _load_price_data(MARKET_INDEX_CSV.stat().st_mtime)


def _slot_start(period: int) -> time:
    minutes = (period - 1) * 30
    return time(minutes // 60, minutes % 60)


def get_prices(d: date) -> pd.Series:
    df = _price_data()
    window = cast(pd.DataFrame, df[df["settlementDate"] <= pd.Timestamp(d)])
    if window.empty:
        raise ValueError(f"No price data on or before {d} in {MARKET_INDEX_CSV}")
    latest = window.sort_values("settlementDate").groupby("settlementPeriod").tail(1).sort_values("settlementPeriod")
    index = [_slot_start(p) for p in latest["settlementPeriod"]]
    return cast(pd.Series, pd.Series(latest["price"].values, index=index, name="price_gbp_per_mwh"))


def latest_price_date() -> date:
    return _price_data()["settlementDate"].max().date()


def _prices_with_dates(d: date) -> pd.Series:
    df = _price_data()
    window = cast(pd.DataFrame, df[df["settlementDate"] <= pd.Timestamp(d)])
    if window.empty:
        raise ValueError(f"No price data on or before {d} in {MARKET_INDEX_CSV}")
    latest = window.sort_values("settlementDate").groupby("settlementPeriod").tail(1)
    index = [datetime.combine(d, _slot_start(p)) for p in latest["settlementPeriod"]]
    return cast(
        pd.Series,
        pd.Series(latest["price"].values, index=index, name="price_gbp_per_mwh").sort_index(),
    )


def prices_in_window(start: datetime, end: datetime) -> pd.Series:
    dates = sorted({start.date(), end.date()})
    all_prices = pd.concat([_prices_with_dates(d) for d in dates])
    return cast(pd.Series, all_prices[(all_prices.index >= start) & (all_prices.index < end)])


def build_charging_schedule(
    archetype: ArchetypeConfig,
    arrival_time: datetime,
    arrival_soc: float,
    deadline: datetime,
) -> pd.Series:
    kwh_needed = max(0.0, (archetype.target_soc - arrival_soc) * archetype.battery_kwh)
    energy_per_slot = archetype.charger_kw * 0.5
    n_slots_needed = ceil(kwh_needed / energy_per_slot)

    window_prices = prices_in_window(arrival_time, deadline)

    cheapest_slots = window_prices.nsmallest(n_slots_needed).index
    schedule = pd.Series(0.0, index=window_prices.index, name="charging_kwh")
    schedule[cheapest_slots] = energy_per_slot
    return schedule


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def get_transition_probability(
    transitions: FlatWindow | dict, lookup_time: time
) -> float:
    if isinstance(transitions, dict):
        return transitions.get(lookup_time, 0.0)
    if isinstance(transitions, FlatWindow):
        return (
            transitions.probability
            if transitions.start <= lookup_time < transitions.end
            else 0.0
        )
    raise TypeError(f"Unknown transition curve type: {type(transitions)}")


def curve_is_enabled(curve: FlatWindow | dict) -> bool:
    if isinstance(curve, dict):
        return bool(curve)
    return curve.probability > 0


def curve_probability(curve: FlatWindow | dict) -> float:
    if isinstance(curve, dict):
        values = set(curve.values())
        return next(iter(values)) if values else 0.0
    return curve.probability


def split_trip(
    soc: float, trip_soc_drop: float, duration_curve: FlatWindow | dict
) -> tuple[float, int, float, float]:
    duration = 1 if random() < curve_probability(duration_curve) else 2
    drop_per_slot = trip_soc_drop / duration
    arrival_soc = max(0.0, soc - trip_soc_drop)
    return max(0.0, soc - drop_per_slot), duration - 1, drop_per_slot, arrival_soc


def charging_destination(archetype: ArchetypeConfig, soc: float) -> State:
    return (
        State.PLUGGED_CHARGING
        if random() < archetype.plugin_frequency_per_day or soc <= archetype.plugin_soc
        else State.PLUGGED_IDLE
    )


def sample_departure_time(transitions: GaussianDeparture) -> time:
    mean_minutes = _minutes(transitions.mean)
    minutes = round(gauss(mean_minutes, transitions.std_minutes))
    minutes = max(0, min(23 * 60 + 59, minutes))
    return time(minutes // 60, minutes % 60)


def next_occurrence(current: datetime, day: date, clock_time: time) -> datetime:
    candidate = datetime.combine(day, clock_time)
    if candidate <= current:
        candidate += timedelta(days=1)
    return candidate


@dataclass
class RunState:
    simulation_time: datetime
    state: State
    soc: float
    idle_departure_time: datetime | None = None
    parked_departure_time: datetime | None = None
    io_schedule: pd.Series | None = None
    drive_day_date: date | None = None
    drive_today: bool = True
    drive_destination: State | None = None
    trip_slots_remaining: int = 0
    trip_drop_per_slot: float = 0.0


def initial_state(archetype: ArchetypeConfig, start_date: date) -> RunState:
    return RunState(
        simulation_time=datetime.combine(start_date, archetype.plugin_time),
        state=State.PLUGGED_CHARGING,
        soc=archetype.plugin_soc,
    )


def advance(
    archetype: ArchetypeConfig, run_state: RunState, end_time: datetime
) -> tuple[pd.DataFrame, RunState]:
    charge_soc_per_slot = archetype.charger_kw * 0.5 / archetype.battery_kwh

    simulation_time = run_state.simulation_time
    state = run_state.state
    soc = run_state.soc
    idle_departure_time = run_state.idle_departure_time
    parked_departure_time = run_state.parked_departure_time
    io_schedule = run_state.io_schedule
    drive_day_date = run_state.drive_day_date
    drive_today = run_state.drive_today
    drive_destination = run_state.drive_destination
    trip_slots_remaining = run_state.trip_slots_remaining
    trip_drop_per_slot = run_state.trip_drop_per_slot

    span_days = (end_time.date() - simulation_time.date()).days
    price_dates = [
        simulation_time.date() + timedelta(days=n) for n in range(span_days + 1)
    ]
    prices = pd.concat([_prices_with_dates(d) for d in price_dates])

    rows = []

    print(f"picking {archetype.name} back up from {simulation_time} at soc {soc:.2f} ({archetype.charging_strategy.name})")

    while simulation_time < end_time:
        current_date = simulation_time.date()
        if drive_day_date != current_date:
            drive_day_date = current_date
            drive_today = random() < archetype.weekday_drive_probability
        day_type = get_day_type(current_date)
        transitions = (
            archetype.weekday_transitions
            if day_type == "weekday"
            else archetype.weekend_transitions
        )
        daily_kwh = (
            archetype.weekday_kwh_per_day
            if day_type == "weekday"
            else archetype.weekend_kwh_per_day
        )
        trip_soc_drop = daily_kwh / archetype.battery_kwh / 2
        lookup_time = time(simulation_time.hour, simulation_time.minute)
        sample = random()
        state_before = state
        soc_before_charging = soc

        if state == State.PLUGGED_CHARGING:
            if archetype.charging_strategy == ChargingStrategy.SCHEDULED_PRICE:
                if io_schedule is None:
                    deadline = next_occurrence(
                        simulation_time, current_date, archetype.plugout_time
                    )
                    io_schedule = build_charging_schedule(
                        archetype, simulation_time, soc, deadline
                    )
                    charging_slots = list(cast(pd.Series, io_schedule[io_schedule > 0]).index)
                    print(f"{simulation_time} octopus grabs its cheapest slots {charging_slots}, wants to be full by {deadline}")
                if cast(float, io_schedule.get(simulation_time, 0.0)) > 0:
                    soc = min(archetype.target_soc, soc + charge_soc_per_slot)
                if (
                    soc >= archetype.target_soc
                    or simulation_time >= cast(datetime, io_schedule.index.max())
                ):
                    state = State.PLUGGED_IDLE
                    io_schedule = None
            elif archetype.charging_strategy == ChargingStrategy.FIXED_TIME:
                if lookup_time >= archetype.plugin_time:
                    soc = min(archetype.target_soc, soc + charge_soc_per_slot)
                    if soc >= archetype.target_soc:
                        state = State.PLUGGED_IDLE
            else:
                soc = min(archetype.target_soc, soc + charge_soc_per_slot)
                if soc >= archetype.target_soc:
                    state = State.PLUGGED_IDLE

        elif state == State.PLUGGED_IDLE:
            curve = transitions.plugged_idle_to_driving
            if isinstance(curve, GaussianDeparture):
                if idle_departure_time is None:
                    idle_departure_time = next_occurrence(
                        simulation_time, current_date, sample_departure_time(curve)
                    )
                    print(f"{simulation_time} they head out around {idle_departure_time}")
                if simulation_time >= idle_departure_time:
                    duration_curve = (
                        transitions.driving_to_parked
                        if curve_is_enabled(transitions.driving_to_parked)
                        else transitions.driving_to_plugged_in
                    )
                    soc, trip_slots_remaining, trip_drop_per_slot, arrival_soc = split_trip(
                        soc, trip_soc_drop, duration_curve
                    )
                    drive_destination = (
                        State.PARKED
                        if curve_is_enabled(transitions.driving_to_parked)
                        else charging_destination(archetype, arrival_soc)
                    )
                    state = State.DRIVING
                    idle_departure_time = None
            else:
                idle_departure_time = None
                if drive_today and sample < get_transition_probability(curve, lookup_time):
                    duration_curve = (
                        transitions.driving_to_parked
                        if curve_is_enabled(transitions.driving_to_parked)
                        else transitions.driving_to_plugged_in
                    )
                    soc, trip_slots_remaining, trip_drop_per_slot, arrival_soc = split_trip(
                        soc, trip_soc_drop, duration_curve
                    )
                    drive_destination = (
                        State.PARKED
                        if curve_is_enabled(transitions.driving_to_parked)
                        else charging_destination(archetype, arrival_soc)
                    )
                    state = State.DRIVING

        elif state == State.PARKED:
            curve = transitions.parked_to_driving
            if isinstance(curve, GaussianDeparture):
                if parked_departure_time is None:
                    parked_departure_time = next_occurrence(
                        simulation_time, current_date, sample_departure_time(curve)
                    )
                    print(f"{simulation_time} they drive home around {parked_departure_time}")
                if simulation_time >= parked_departure_time:
                    soc, trip_slots_remaining, trip_drop_per_slot, arrival_soc = split_trip(
                        soc, trip_soc_drop, transitions.driving_to_plugged_in
                    )
                    drive_destination = charging_destination(archetype, arrival_soc)
                    state = State.DRIVING
                    parked_departure_time = None
            else:
                parked_departure_time = None
                if sample < get_transition_probability(curve, lookup_time):
                    soc, trip_slots_remaining, trip_drop_per_slot, arrival_soc = split_trip(
                        soc, trip_soc_drop, transitions.driving_to_plugged_in
                    )
                    drive_destination = charging_destination(archetype, arrival_soc)
                    state = State.DRIVING

        elif state == State.DRIVING:
            if trip_slots_remaining > 0:
                soc = max(0.0, soc - trip_drop_per_slot)
                trip_slots_remaining -= 1
            else:
                assert drive_destination is not None
                state = drive_destination
                drive_destination = None

        energy_delivered_kwh = (
            max(0.0, (soc - soc_before_charging)) * archetype.battery_kwh
        )
        if energy_delivered_kwh > 0:
            if simulation_time not in prices.index:
                raise ValueError(f"No price for {simulation_time}, but {archetype.name} charged this slot")
            price_gbp_per_mwh = prices[simulation_time]
        else:
            price_gbp_per_mwh = 0.0
        cost = energy_delivered_kwh * price_gbp_per_mwh / 1000

        marker = (
            f"  <-- {state_before.name} -> {state.name}"
            if state != state_before
            else ""
        )
        cost_note = f" spent £{cost:.4f}" if cost > 0 else ""
        print(f"  {simulation_time} {day_type} {state.name} soc={soc:.2f}{cost_note}{marker}")

        if state in (State.PLUGGED_CHARGING, State.PLUGGED_IDLE):
            logged_state = State.PLUGGED_CHARGING if energy_delivered_kwh > 0 else State.PLUGGED_IDLE
        else:
            logged_state = state
        rows.append({"time": simulation_time, "state": logged_state, "soc": soc, "cost": cost})
        simulation_time += timedelta(minutes=30)

    new_run_state = RunState(
        simulation_time,
        state,
        soc,
        idle_departure_time,
        parked_departure_time,
        io_schedule,
        drive_day_date,
        drive_today,
        drive_destination,
        trip_slots_remaining,
        trip_drop_per_slot,
    )
    return pd.DataFrame(rows).set_index("time"), new_run_state


def simulate_week(
    archetype: ArchetypeConfig, start_date: date, days: float = 7
) -> pd.DataFrame:
    run_state = initial_state(archetype, start_date)
    end_time = run_state.simulation_time + timedelta(days=days)
    df, _ = advance(archetype, run_state, end_time)
    return df


def run_monte_carlo(
    archetype: ArchetypeConfig, start_date: date, n_runs: int, days: float = 7
) -> dict[str, pd.DataFrame]:
    soc_runs = {}
    state_runs = {}
    cost_runs = {}

    for run_number in range(n_runs):
        df = simulate_week(archetype, start_date, days=days)
        soc_runs[run_number] = df["soc"]
        state_runs[run_number] = df["state"]
        cost_runs[run_number] = df["cost"]

    return {
        "soc": pd.DataFrame(soc_runs),
        "state": pd.DataFrame(state_runs),
        "cost": pd.DataFrame(cost_runs),
    }


def run_monte_carlo_cached(
    archetype_name: str, start_date: date, n_runs: int, days: float = 7
) -> dict[str, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = (
        CACHE_DIR / f"{archetype_name}_{start_date}_{n_runs}runs_{days}days.pkl"
    )

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    archetype = get_archetype(archetype_name)
    result = run_monte_carlo(archetype, start_date, n_runs, days=days)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def recapitulate_population(
    start_date: date, runs_per_archetype: int = 500, days: float = 7
) -> PopulationResult:
    soc_columns = {}
    state_columns = {}
    cost_columns = {}
    weights = {}

    for name in _ARCHETYPES:
        archetype = get_archetype(name)
        result = run_monte_carlo_cached(name, start_date, runs_per_archetype, days=days)
        weight_per_run = archetype.population_share / runs_per_archetype
        for run_number in result["soc"].columns:
            key = f"{name}_{run_number}"
            soc_columns[key] = result["soc"][run_number]
            state_columns[key] = result["state"][run_number]
            cost_columns[key] = result["cost"][run_number]
            weights[key] = weight_per_run

    return {
        "soc": pd.DataFrame(soc_columns),
        "state": pd.DataFrame(state_columns),
        "cost": pd.DataFrame(cost_columns),
        "weights": pd.Series(weights),
    }


def get_or_advance_run(
    archetype_name: str, run_number: int, end_dt: datetime
) -> pd.DataFrame:
    RUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RUN_CACHE_DIR / f"{archetype_name}_{run_number}.pkl"
    archetype = get_archetype(archetype_name)

    trajectory = pd.DataFrame(columns=["state", "soc", "cost"]).rename_axis("time")
    run_state = initial_state(archetype, end_dt.date() - timedelta(days=LOOKBACK_DAYS))
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                trajectory, run_state = pickle.load(f)
        except (pickle.PickleError, EOFError, OSError):  # this is to prevent cache re run from failing in Streamlit cloud
            pass

    if run_state.simulation_time < end_dt:
        new_rows, run_state = advance(archetype, run_state, end_dt)
        trajectory = pd.concat([trajectory, new_rows])

    try:
        tmp_path = cache_path.with_suffix(f".{uuid4().hex}.pkl.tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump((trajectory, run_state), f)
        tmp_path.replace(cache_path)
    except (pickle.PickleError, OSError): # this is to prevent cache re run from failing in Streamlit cloud
        pass
    return trajectory


def get_population_runs(end_dt: datetime, runs_per_archetype: int) -> PopulationResult:
    soc_columns = {}
    state_columns = {}
    cost_columns = {}
    weights = {}

    for name in _ARCHETYPES:
        archetype = get_archetype(name)
        weight_per_run = archetype.population_share / runs_per_archetype
        for run_number in range(runs_per_archetype):
            trajectory = get_or_advance_run(name, run_number, end_dt)
            key = f"{name}_{run_number}"
            soc_columns[key] = trajectory["soc"]
            state_columns[key] = trajectory["state"]
            cost_columns[key] = trajectory["cost"]
            weights[key] = weight_per_run

    return {
        "soc": pd.DataFrame(soc_columns),
        "state": pd.DataFrame(state_columns),
        "cost": pd.DataFrame(cost_columns),
        "weights": pd.Series(weights),
    }


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    mask = ~np.isnan(values)
    if not mask.any():
        return float("nan")
    values, weights = values[mask], weights[mask]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum_weights = np.cumsum(weights) - 0.5 * weights
    cum_weights /= weights.sum()
    return np.interp(quantile, cum_weights, values)


def weighted_quantiles(
    df: pd.DataFrame, weights: pd.Series, quantiles: list[float]
) -> pd.DataFrame:
    weight_array = weights.reindex(df.columns).to_numpy(dtype=float)
    result = {
        q: df.apply(
            lambda row: _weighted_quantile(row.to_numpy(dtype=float), weight_array, q),
            axis=1,
        )
        for q in quantiles
    }
    return pd.DataFrame(result)


def weighted_mean(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    weight_array = weights.reindex(df.columns).to_numpy(dtype=float)
    present_weight = df.notna().mul(weight_array, axis=1).sum(axis=1)
    return df.mul(weight_array, axis=1).sum(axis=1) / present_weight


def plugged_in_share(
    state_df: pd.DataFrame,
    weights: pd.Series | None = None,
    states: tuple[State, ...] = (State.PLUGGED_CHARGING, State.PLUGGED_IDLE),
) -> pd.Series:
    plugged_in = state_df.isin(list(states)).astype(float)
    if weights is None:
        return plugged_in.mean(axis=1)
    weight_array = weights.reindex(plugged_in.columns)
    return plugged_in.mul(weight_array, axis=1).sum(axis=1) / weight_array.sum()