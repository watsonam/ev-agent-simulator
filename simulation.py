import pickle
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from math import ceil
from pathlib import Path
from random import gauss, random
from typing import TypedDict

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
RUN_CACHE_DIR = CACHE_DIR / "runs"
LOOKBACK_DAYS = 7  # how many days back a new run starts simulating from, before the date you actually asked for


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
def _load_price_data(mtime: float) -> pd.DataFrame:
    """market_index.csv, parsed once per mtime. `mtime` is the cache key, so
    a call with the same mtime returns the cached DataFrame instead of
    re-reading the file; a new mtime (update_day_ahead_prices appended rows)
    evicts it and reads fresh. Without this, get_prices() re-read and
    re-parsed the whole ~9,000-row file on every call - the dominant cost
    when advancing hundreds of Monte Carlo runs."""
    return pd.read_csv(MARKET_INDEX_CSV, parse_dates=["settlementDate"])


def _price_data() -> pd.DataFrame:
    return _load_price_data(MARKET_INDEX_CSV.stat().st_mtime)


def get_prices(d: date) -> pd.Series:
    """Half-hourly EPEX day-ahead price (£/MWh) for `d`, indexed by slot start time."""
    df = _price_data()
    day = df[df["settlementDate"] == pd.Timestamp(d)].sort_values("settlementPeriod")
    if day.empty:
        raise ValueError(f"No price data for {d} in {MARKET_INDEX_CSV}")

    def slot_start(period: int) -> time:
        minutes = (period - 1) * 30
        return time(minutes // 60, minutes % 60)

    index = [slot_start(p) for p in day["settlementPeriod"]]
    return pd.Series(day["price"].values, index=index, name="price_gbp_per_mwh")


def latest_price_date() -> date:
    return _price_data()["settlementDate"].max().date()


def slice_day(df: pd.DataFrame, d: date) -> pd.DataFrame:
    """Rows for calendar day `d` only, from a datetime-indexed DataFrame."""
    start = datetime.combine(d, time.min)
    return df[(df.index >= start) & (df.index < start + timedelta(days=1))]


def _prices_with_dates(d: date) -> pd.Series:
    """get_prices(d) indexed by bare time - re-index by full datetime so two
    days can be concatenated without same-time-of-day collisions."""
    prices = get_prices(d)
    return pd.Series(
        prices.values,
        index=[datetime.combine(d, t) for t in prices.index],
        name=prices.name,
    )


def build_charging_schedule(
    archetype: ArchetypeConfig,
    arrival_time: datetime,
    arrival_soc: float,
    deadline: datetime,
) -> pd.Series:
    """Cheapest-slots schedule for archetypes with price-driven charging
    (Intelligent Octopus). Sized to the ACTUAL SoC deficit at arrival - not
    archetype.kwh_per_plugin, which is only a population average - because
    in the continuous simulation, arrival_soc is an emergent per-run result
    that varies. Picks the N cheapest half-hour slots between arrival_time
    and deadline using real EPEX day-ahead prices. Average UK-style "charge
    immediately" archetypes don't need this - simulate_week's
    PLUGGED_CHARGING ramp already encodes that directly. Returns a Series
    of kWh charged per slot (0 where not charging)."""
    kwh_needed = max(0.0, (archetype.target_soc - arrival_soc) * archetype.battery_kwh)
    energy_per_slot = archetype.charger_kw * 0.5
    n_slots_needed = ceil(kwh_needed / energy_per_slot)

    dates = sorted({arrival_time.date(), deadline.date()})
    all_prices = pd.concat([_prices_with_dates(dd) for dd in dates])
    window_prices = all_prices[
        (all_prices.index >= arrival_time) & (all_prices.index < deadline)
    ]

    cheapest_slots = window_prices.nsmallest(n_slots_needed).index
    schedule = pd.Series(0.0, index=window_prices.index, name="charging_kwh")
    schedule[cheapest_slots] = energy_per_slot
    return schedule


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def get_transition_probability(
    transitions: FlatWindow | dict, lookup_time: time
) -> float:
    """P(transition) at this slot - for the two literal-probability curve
    shapes only. GaussianDeparture doesn't go through here: it's sampled
    once via sample_departure_time(), not looked up per slot."""
    if isinstance(transitions, dict):
        return transitions.get(lookup_time, 0.0)
    if isinstance(transitions, FlatWindow):
        return (
            transitions.probability
            if transitions.start <= lookup_time < transitions.end
            else 0.0
        )
    raise TypeError(f"Unknown transition curve type: {type(transitions)}")


def sample_departure_time(transitions: GaussianDeparture) -> time:
    """Draw one departure time from Normal(mean, std) - decided once, not
    re-checked every slot. Clamped to a valid time-of-day."""
    mean_minutes = _minutes(transitions.mean)
    minutes = round(gauss(mean_minutes, transitions.std_minutes))
    minutes = max(0, min(23 * 60 + 59, minutes))
    return time(minutes // 60, minutes % 60)


def _next_occurrence(current: datetime, day: date, clock_time: time) -> datetime:
    """Attach `clock_time` to `day`; if that's already in the past relative
    to `current`, push it to the next day instead. Needed because a bare
    `time` comparison across a midnight crossing is wrong - it has no
    notion of which calendar day it's on (e.g. 19:30 this evening would
    compare as "later than" 08:47 tomorrow morning, purely by clock value)."""
    candidate = datetime.combine(day, clock_time)
    if candidate <= current:
        candidate += timedelta(days=1)
    return candidate


@dataclass
class RunState:
    """
    Contains parameters needed to continue simulation run from a certain state. 
    We cache previous simulations and then incrementally simulate 30 minute slots on load where missing.
    """

    simulation_time: datetime
    state: State
    soc: float
    idle_departure_time: datetime | None = None
    parked_departure_time: datetime | None = None
    io_schedule: pd.Series | None = None


def initial_state(archetype: ArchetypeConfig, start_date: date) -> RunState:
    """Initialise state at plug in soc given archetype in spreadsheet"""
    return RunState(
        simulation_time=datetime.combine(start_date, archetype.plugin_time),
        state=State.PLUGGED_CHARGING,
        soc=archetype.plugin_soc,
    )


def advance(
    archetype: ArchetypeConfig, run_state: RunState, end_time: datetime
) -> tuple[pd.DataFrame, RunState]:
    """Simulate archetype forward from run_state.simulation_time to end_time.
    Does NOT reset state at midnight, so the overnight plug-in-to-departure
    cycle isn't chopped in half at a day boundary.

    archetype.charging_strategy decides what happens once PLUGGED_CHARGING:
      IMMEDIATE - ramp every slot (deterministic), same as before.
      SCHEDULED_PRICE (Intelligent Octopus) - build a real EPEX cheapest-
        slots schedule against the ACTUAL arrival time/SoC (not the
        archetype average) and only add energy in scheduled slots.
      FIXED_TIME (Scheduled charging) - sit in PLUGGED_CHARGING without
        gaining SoC until archetype.plugin_time, then ramp normally.

    Returns (new rows only, indexed by time with `state`/`soc`/`cost`
    columns) and the RunState at end_time, so a later call can resume
    exactly where this one stopped."""

    charge_soc_per_slot = archetype.charger_kw * 0.5 / archetype.battery_kwh

    simulation_time = run_state.simulation_time
    state = run_state.state
    soc = run_state.soc
    idle_departure_time = run_state.idle_departure_time
    parked_departure_time = run_state.parked_departure_time
    io_schedule = run_state.io_schedule

    # Prefetch prices for the whole window once - calling get_prices() (which
    # re-reads the CSV) every slot would be wasteful.
    span_days = (end_time.date() - simulation_time.date()).days
    price_dates = [
        simulation_time.date() + timedelta(days=n) for n in range(span_days + 1)
    ]
    prices = pd.concat([_prices_with_dates(d) for d in price_dates])

    rows = []

    print(
        f"[{archetype.name}] strategy={archetype.charging_strategy.name} resume={simulation_time} soc={soc:.4f}"
    )

    while simulation_time < end_time:
        current_date = simulation_time.date()
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
        trip_soc_drop = (
            daily_kwh / archetype.battery_kwh / 2
        )  # 2 legs/day, split evenly
        lookup_time = time(simulation_time.hour, simulation_time.minute)
        sample = random()
        state_before = state
        soc_before_charging = soc

        if state == State.PLUGGED_CHARGING:
            if archetype.charging_strategy == ChargingStrategy.SCHEDULED_PRICE:
                if io_schedule is None:
                    deadline = _next_occurrence(
                        simulation_time, current_date, archetype.plugout_time
                    )
                    io_schedule = build_charging_schedule(
                        archetype, simulation_time, soc, deadline
                    )
                    print(
                        f"  [{simulation_time}] IO schedule built: deadline={deadline} "
                        f"slots={list(io_schedule[io_schedule > 0].index)}"
                    )
                if io_schedule.get(simulation_time, 0.0) > 0:
                    soc = min(archetype.target_soc, soc + charge_soc_per_slot)
                if (
                    soc >= archetype.target_soc
                    or simulation_time >= io_schedule.index.max()
                ):
                    state = State.PLUGGED_IDLE
                    io_schedule = None
            elif archetype.charging_strategy == ChargingStrategy.FIXED_TIME:
                if lookup_time >= archetype.plugin_time:
                    soc = min(archetype.target_soc, soc + charge_soc_per_slot)
                    if soc >= archetype.target_soc:
                        state = State.PLUGGED_IDLE
            else:  # IMMEDIATE
                soc = min(archetype.target_soc, soc + charge_soc_per_slot)
                if soc >= archetype.target_soc:
                    state = State.PLUGGED_IDLE

        elif state == State.PLUGGED_IDLE:
            curve = transitions.plugged_idle_to_driving
            if isinstance(curve, GaussianDeparture):
                if idle_departure_time is None:
                    idle_departure_time = _next_occurrence(
                        simulation_time, current_date, sample_departure_time(curve)
                    )
                    print(
                        f"  [{simulation_time}] sampled idle departure time: {idle_departure_time}"
                    )
                if simulation_time >= idle_departure_time:
                    state, soc = State.DRIVING, soc - trip_soc_drop
                    idle_departure_time = None
            else:
                # Not a Gaussian day (e.g. weekday dict after a weekend
                # Gaussian sample) - clear any stale sample so it can't
                # leak into a future weekend and fire instantly there.
                idle_departure_time = None
                if sample < get_transition_probability(curve, lookup_time):
                    state, soc = State.DRIVING, soc - trip_soc_drop

        elif state == State.PARKED:
            curve = transitions.parked_to_driving
            if isinstance(curve, GaussianDeparture):
                if parked_departure_time is None:
                    parked_departure_time = _next_occurrence(
                        simulation_time, current_date, sample_departure_time(curve)
                    )
                    print(
                        f"  [{simulation_time}] sampled parked departure time: {parked_departure_time}"
                    )
                if simulation_time >= parked_departure_time:
                    state, soc = State.DRIVING, soc - trip_soc_drop
                    parked_departure_time = None
            else:
                parked_departure_time = (
                    None  # same leak-prevention as idle_departure_time above
                )
                if sample < get_transition_probability(curve, lookup_time):
                    state, soc = State.DRIVING, soc - trip_soc_drop

        elif state == State.DRIVING:
            if sample < get_transition_probability(
                transitions.driving_to_parked, lookup_time
            ):
                state = State.PARKED
            elif sample < get_transition_probability(
                transitions.driving_to_plugged_in, lookup_time
            ):
                state = State.PLUGGED_CHARGING

        energy_delivered_kwh = (
            max(0.0, (soc - soc_before_charging)) * archetype.battery_kwh
        )
        price_gbp_per_mwh = prices.get(simulation_time, 0.0)
        cost = energy_delivered_kwh * price_gbp_per_mwh / 1000

        marker = (
            f"  <-- {state_before.name} -> {state.name}"
            if state != state_before
            else ""
        )
        cost_note = f" cost=£{cost:.4f}" if cost > 0 else ""
        print(
            f"  [{simulation_time}] {day_type:7s} state={state.name:16s} soc={soc:.4f}{cost_note}{marker}"
        )

        rows.append({"time": simulation_time, "state": state, "soc": soc, "cost": cost})
        simulation_time += timedelta(minutes=30)

    new_run_state = RunState(
        simulation_time,
        state,
        soc,
        idle_departure_time,
        parked_departure_time,
        io_schedule,
    )
    return pd.DataFrame(rows).set_index("time"), new_run_state


def simulate_week(
    archetype: ArchetypeConfig, start_date: date, days: float = 7
) -> pd.DataFrame:
    """One-shot simulation from a fresh start - a thin wrapper around
    initial_state()/advance() for callers that don't need to pause/resume."""
    run_state = initial_state(archetype, start_date)
    end_time = run_state.simulation_time + timedelta(days=days)
    df, _ = advance(archetype, run_state, end_time)
    return df


def run_monte_carlo(
    archetype: ArchetypeConfig, start_date: date, n_runs: int, days: float = 7
) -> dict[str, pd.DataFrame]:
    """Repeat simulate_week n_runs times. Every run shares the same time
    index (simulate_week steps a fixed 30-min grid regardless of state, so
    there's nothing to align), so we just collect each run's soc/state/cost
    column into its own dict and wrap each into a wide DataFrame at the end.

    Returns {"soc": df, "state": df, "cost": df}, each shaped time x run_number.
    """
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
    """Same as run_monte_carlo, but caches to disk - 500 runs x 6 archetypes
    takes minutes, no reason to recompute every time nothing's changed."""
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
    """Combines all 6 archetypes into one population-level sample for the
    "recapitulate population-level observations" chart from the brief.

    Every archetype gets the SAME number of runs (runs_per_archetype), not
    a count proportional to population_share - a rare archetype like Always
    plugged-in (1%) still needs enough of its own runs to characterize its
    own spread; population weighting is applied separately at aggregation
    time (see weighted_quantiles) via the "weights" Series in the return
    value, rather than being baked into how many times each archetype runs.

    Columns keep their real datetime index. Every archetype's plugin_time is
    a multiple of 30 minutes, so all archetypes land on the same half-hourly
    grid despite starting at different clock times (18:00, 22:00, 00:00) -
    pooling on the raw datetime index is safe, and lets callers slice the
    result down to a single calendar day (see slice_day) for a "population
    right now" view rather than only a whole-week one.
    """
    soc_columns = {}
    state_columns = {}
    cost_columns = {}
    weights = {}

    for name in _ARCHETYPES:
        archetype = get_archetype(name)
        result = run_monte_carlo_cached(name, start_date, runs_per_archetype, days=days)
        # Each run's weight is population_share split evenly across this
        # archetype's runs, so summing all of Average UK's run weights
        # gives back exactly 0.40 (its population_share).
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
    """Get one Monte Carlo run's full history up to `end_dt`, extending it if
    the cached copy doesn't reach that far yet. This is where the incremental
    cache lives - the reason changing the date in the dashboard is fast.

    Each (archetype, run_number) pair owns one pickle file holding two things:
      - trajectory: every 30-min row simulated so far (state, soc, cost)
      - run_state:  a snapshot of where the run paused (time, soc, state, and
                    any half-finished Intelligent Octopus charging schedule)

    Steps:
      1. Load that file if it exists; otherwise start a fresh run LOOKBACK_DAYS
         before end_dt
      2. If the run hasn't reached end_dt, advance() it the rest of the way and
         append the new rows. If the cache is already past end_dt, do nothing.
      3. Save the extended (trajectory, run_state) back to the same file.

    Raises ValueError if advance() needs a price we don't have (e.g. tomorrow's
    day-ahead prices haven't been published yet) - that's a real data gap, not
    something to silently paper over, so it's left to fail rather than caught.

    Returns the whole accumulated trajectory
    """
    RUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RUN_CACHE_DIR / f"{archetype_name}_{run_number}.pkl"
    archetype = get_archetype(archetype_name)

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            trajectory, run_state = pickle.load(f)
    else:
        trajectory = pd.DataFrame(columns=["state", "soc", "cost"]).rename_axis("time")
        run_state = initial_state(
            archetype, end_dt.date() - timedelta(days=LOOKBACK_DAYS)
        )

    if run_state.simulation_time < end_dt:
        new_rows, run_state = advance(archetype, run_state, end_dt)
        trajectory = pd.concat([trajectory, new_rows])

    # Write to a temp file then rename into place, so a process kill
    # mid-write can never leave a truncated, permanently-unreadable cache
    # file at cache_path - the rename is atomic, the old file stays intact
    # until the new one is fully written.
    tmp_path = cache_path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump((trajectory, run_state), f)
    tmp_path.replace(cache_path)
    return trajectory


def get_population_runs(end_dt: datetime, runs_per_archetype: int) -> PopulationResult:
    """Same shape as recapitulate_population, but sourced from the
    incremental per-run cache (get_or_advance_run) instead of re-simulating
    a fresh date-bounded window every time. Returns full accumulated
    history per run - callers slice_day() the day they want to look at."""
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
    """One row's weighted quantile. Sorts by value, finds the midpoint of
    each value's cumulative weight interval, then interpolates - the
    standard way to generalize "quantile" when observations aren't equally
    weighted."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum_weights = np.cumsum(weights) - 0.5 * weights
    cum_weights /= weights.sum()
    return np.interp(quantile, cum_weights, values)


def weighted_quantiles(
    df: pd.DataFrame, weights: pd.Series, quantiles: list[float]
) -> pd.DataFrame:
    """Row-wise weighted quantiles across df's columns. `weights` maps
    column name -> weight (from recapitulate_population's "weights").
    Returns a DataFrame indexed like df, one column per requested quantile."""
    weight_array = weights.reindex(df.columns).to_numpy(dtype=float)
    result = {
        q: df.apply(
            lambda row: _weighted_quantile(row.to_numpy(dtype=float), weight_array, q),
            axis=1,
        )
        for q in quantiles
    }
    return pd.DataFrame(result)


def plugged_in_share(
    state_df: pd.DataFrame, weights: pd.Series | None = None
) -> pd.Series:
    """Fraction of runs plugged in (PLUGGED_CHARGING or PLUGGED_IDLE) at each
    time slot, one row per slot. weights=None weights every run equally -
    the same reduction used for a single archetype's runs and, with real
    population weights, for the whole population."""
    plugged_in = state_df.isin([State.PLUGGED_CHARGING, State.PLUGGED_IDLE]).astype(
        float
    )
    if weights is None:
        return plugged_in.mean(axis=1)
    weight_array = weights.reindex(plugged_in.columns)
    return plugged_in.mul(weight_array, axis=1).sum(axis=1) / weight_array.sum()


if __name__ == "__main__":
    ARCHETYPE_NAME = "average_uk"  # change this to test other archetypes
    START_DATE = date(2026, 6, 24)

    archetype = get_archetype(ARCHETYPE_NAME)
    print(
        f"\n{'=' * 80}\n{ARCHETYPE_NAME} ({archetype.charging_strategy.name})\n{'=' * 80}"
    )
    df = simulate_week(archetype, START_DATE)

    state_str = df["state"].astype(str)
    plugin_events = df[
        (state_str == "State.PLUGGED_CHARGING")
        & (state_str.shift(1) != "State.PLUGGED_CHARGING")
    ]
    print(f"\n--- {ARCHETYPE_NAME}: plug-in events (time, soc) ---")
    print(plugin_events[["soc"]].to_string())

    N_RUNS = 20
    print(f"\n{'=' * 80}\nrun_monte_carlo: {ARCHETYPE_NAME}, {N_RUNS} runs\n{'=' * 80}")
    result = run_monte_carlo(archetype, START_DATE, n_runs=N_RUNS)

    soc_df, state_df, cost_df = result["soc"], result["state"], result["cost"]
    print(f"soc shape:   {soc_df.shape}")
    print(f"state shape: {state_df.shape}")
    print(f"cost shape:  {cost_df.shape}")

    # sanity checks - same bounds we checked by hand earlier
    bad_soc = soc_df[(soc_df < -1e-9) | (soc_df > archetype.target_soc + 1e-4)]
    print(f"\nout-of-bounds soc values: {bad_soc.count().sum()}")
    print(f"NaN soc values:           {soc_df.isna().sum().sum()}")
    print(f"NaN cost values:          {cost_df.isna().sum().sum()}")

    print(
        f"\ntotal cost per run: min=£{cost_df.sum().min():.2f}  "
        f"mean=£{cost_df.sum().mean():.2f}  max=£{cost_df.sum().max():.2f}"
    )

    print("\nsoc percentile bands (10/50/90), last 5 slots:")
    print(soc_df.quantile([0.1, 0.5, 0.9], axis=1).T.tail())

    RUNS_PER_ARCHETYPE = 500
    print(
        f"\n{'=' * 80}\nrecapitulate_population: {RUNS_PER_ARCHETYPE} runs per archetype (x6)\n{'=' * 80}"
    )
    population = recapitulate_population(
        START_DATE, runs_per_archetype=RUNS_PER_ARCHETYPE
    )
    pop_soc, pop_state, pop_cost, weights = (
        population["soc"],
        population["state"],
        population["cost"],
        population["weights"],
    )

    print(
        f"soc shape:   {pop_soc.shape}   (expect (days*48) x {RUNS_PER_ARCHETYPE * 6})"
    )
    print(f"weights sum: {weights.sum():.4f}  (expect 1.0)")
    bad_pop_soc = pop_soc[(pop_soc < -1e-9) | (pop_soc > 0.8001)]
    print(f"out-of-bounds soc values: {bad_pop_soc.count().sum()}")
    print(f"NaN soc values:           {pop_soc.isna().sum().sum()}")

    print(
        f"\npopulation total cost: min=£{pop_cost.sum().min():.2f}  "
        f"mean=£{pop_cost.sum().mean():.2f}  max=£{pop_cost.sum().max():.2f}"
    )

    print("\npopulation soc WEIGHTED percentile bands (5/50/95), last 5 slots:")
    bands = weighted_quantiles(pop_soc, weights, [0.05, 0.5, 0.95])
    print(bands.tail())

    print("\n% plugged in (PLUGGED_CHARGING or PLUGGED_IDLE), weighted, last 5 slots:")
    plugged_in = pop_state.isin([State.PLUGGED_CHARGING, State.PLUGGED_IDLE]).astype(
        float
    )
    weighted_share = plugged_in.mul(weights, axis=1).sum(axis=1) / weights.sum()
    print(weighted_share.tail())
