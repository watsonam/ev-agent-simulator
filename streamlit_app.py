from datetime import date, datetime, time, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from archetypes import ArchetypeConfig, State
from elexon_client import update_day_ahead_prices
from simulation import (
    LOOKBACK_DAYS,
    MARKET_INDEX_CSV,
    RUN_CACHE_DIR,
    PopulationResult,
    build_charging_schedule,
    get_archetype,
    get_population_runs,
    get_prices,
    latest_price_date,
    next_occurrence,
    plugged_in_share,
    prices_in_window,
    slice_day,
    weighted_mean,
    weighted_quantiles,
)

ARCHETYPE_NAMES = [
    "average_uk",
    "intelligent_octopus",
    "infrequent_charging",
    "infrequent_driving",
    "scheduled_charging",
    "always_plugged_in",
]
RUNS_PER_ARCHETYPE = 200
EVENING_START = time(18, 0)

COLOR_SOC = "#1864ab"
COLOR_BAND_EDGE = "#74c0fc"
COLOR_BAND_FILL = "rgba(24, 100, 171, 0.12)"
COLOR_OCCUPANCY = "rgba(134, 142, 150, 0.35)"
COLOR_OCCUPANCY_FILL = "rgba(134, 142, 150, 0.25)"
COLOR_PRICE = "#495057"
COLOR_TODAY_MARKER = "#868e96"
COLOR_WEEKEND = "#845ef7"


def auto_range(values, pad_frac: float = 0.1) -> list[float]:
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * pad_frac or 0.05
    return [lo - pad, hi + pad]


def archetype_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[[c for c in df.columns if c.startswith(f"{name}_")]]


def slice_window(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index < end)]


def sim_window(sim_date: date, now: datetime) -> tuple[datetime, datetime, datetime]:
    start = datetime.combine(sim_date - timedelta(days=1), EVENING_START)
    midnight = datetime.combine(sim_date, time.min)
    if sim_date == now.date():
        end = now
    else:
        end = datetime.combine(sim_date + timedelta(days=1), time.min)
    return start, midnight, end


def time_note(sim_date: date, now: datetime) -> str:
    return f"up to {now.strftime('%H:%M')} today" if sim_date == now.date() else "the full day"


def is_cache_cold(runs_per_archetype: int) -> bool:
    if not RUN_CACHE_DIR.exists():
        return True
    cached = len(list(RUN_CACHE_DIR.glob("*.pkl")))
    return cached < len(ARCHETYPE_NAMES) * runs_per_archetype


def _require_rows(df: pd.DataFrame, population: PopulationResult, window_start: datetime, end_dt: datetime) -> None:
    if df.empty:
        cached_from = population["soc"].index.min()
        raise ValueError(
            f"No simulated data between {window_start} and {end_dt}. The cached "
            f"runs start at {cached_from} - pick a more recent date."
        )


def sample_run_trajectory(population: PopulationResult, name: str, window_start: datetime, end_dt: datetime) -> tuple[pd.Series, pd.Series, pd.Series]:
    run_col = f"{name}_0"
    # A run starts at its own plugin_time, so near the earliest selectable date
    # the window can reach back before this run has any slots - those rows come
    # through as NaN once the archetypes are aligned. Drop them so we only plot
    # real simulated slots.
    state_run = slice_window(population["state"][[run_col]], window_start, end_dt)[run_col].dropna()
    soc_run = slice_window(population["soc"][[run_col]], window_start, end_dt)[run_col].reindex(state_run.index)
    _require_rows(state_run.to_frame(), population, window_start, end_dt)
    plugged_in = state_run.isin([State.PLUGGED_CHARGING, State.PLUGGED_IDLE]).astype(float)
    return soc_run, plugged_in, state_run.map(lambda s: s.name)


def charging_session_window(state: pd.Series, archetype: ArchetypeConfig) -> tuple[datetime, datetime] | None:
    charging = state == "PLUGGED_CHARGING"
    arrivals = charging & ~charging.shift(1, fill_value=False)
    if not arrivals.any():
        return None
    arrival_time = arrivals[arrivals].index[-1]
    deadline = next_occurrence(arrival_time, arrival_time.date(), archetype.plugout_time)
    return arrival_time, deadline


def scheduled_slots(archetype: ArchetypeConfig, arrival: datetime, arrival_soc: float, deadline: datetime) -> pd.DataFrame:
    schedule = build_charging_schedule(archetype, arrival, arrival_soc, deadline)
    charged = schedule[schedule > 0].index
    prices = prices_in_window(arrival, deadline).reindex(charged)
    return pd.DataFrame({"Time": prices.index.strftime("%a %H:%M"), "£/MWh": prices.values})


def population_summary(population: PopulationResult, window_start: datetime, end_dt: datetime) -> tuple[pd.DataFrame, pd.Series]:
    weights = population["weights"]
    pop_soc = slice_window(population["soc"], window_start, end_dt)
    pop_state = slice_window(population["state"], window_start, end_dt)
    _require_rows(pop_soc, population, window_start, end_dt)

    pct_plugged_in = plugged_in_share(pop_state, weights) * 100
    bands = weighted_quantiles(pop_soc, weights, [0.05, 0.95])
    bands.columns = ["p05", "p95"]
    bands["mean"] = weighted_mean(pop_soc, weights)
    return bands, pct_plugged_in


def cost_totals(population: PopulationResult, name: str, archetype: ArchetypeConfig, d: date) -> tuple[pd.Series, pd.Series]:
    day_soc = slice_day(archetype_columns(population["soc"], name), d)
    day_cost = slice_day(archetype_columns(population["cost"], name), d)
    total_kwh = day_soc.diff().clip(lower=0).sum() * archetype.battery_kwh
    total_cost = day_cost.sum()
    return total_cost, total_kwh


def savings_row(population: PopulationResult, name: str, archetype: ArchetypeConfig, sim_date: date, earliest: date) -> dict:
    total_cost, total_kwh = cost_totals(population, name, archetype, sim_date)
    day_shown = sim_date
    fallback_date = sim_date - timedelta(days=1)
    if total_kwh.sum() <= 0 and fallback_date >= earliest:
        total_cost, total_kwh = cost_totals(population, name, archetype, fallback_date)
        day_shown = fallback_date

    charged = total_kwh > 0
    per_kwh = (total_cost[charged] / total_kwh[charged]).mean() if charged.any() else None
    return {
        "Archetype": archetype.name,
        "Day shown": day_shown,
        "£/kWh": per_kwh,
        "Total cost (£)": total_cost[charged].mean() if charged.any() else None,
        "Energy (kWh)": total_kwh[charged].mean() if charged.any() else None,
    }


def savings_table(population: PopulationResult, archetypes: dict[str, ArchetypeConfig], sim_date: date, earliest: date) -> pd.DataFrame:
    df = pd.DataFrame([savings_row(population, name, a, sim_date, earliest) for name, a in archetypes.items()])
    baseline = df.loc[df["Archetype"] == archetypes["average_uk"].name, "£/kWh"].iloc[0]
    df["Savings vs Average (UK) (£)"] = (baseline - df["£/kWh"]) * df["Energy (kWh)"]
    return df


def mark_today(fig: go.Figure, midnight: datetime) -> None:
    fig.add_vline(x=midnight, line_dash="dash", line_color=COLOR_TODAY_MARKER)
    fig.add_annotation(x=midnight, y=1.03, yref="paper", showarrow=False, text="T-1 | Today", font=dict(size=11, color=COLOR_TODAY_MARKER))


def mark_weekends(fig: go.Figure, index: pd.DatetimeIndex) -> None:
    start, end = index.min(), index.max()
    day = start.date()
    while day <= end.date():
        if day.weekday() == 5:
            span_start = max(start, datetime.combine(day, time.min))
            span_end = min(end, datetime.combine(day + timedelta(days=2), time.min))
            if span_start < span_end:
                fig.add_vrect(x0=span_start, x1=span_end, fillcolor=COLOR_WEEKEND, opacity=0.08, layer="below", line_width=0)
        day += timedelta(days=1)


def build_soc_chart(soc: pd.Series, plugged_in: pd.Series, state: pd.Series, midnight: datetime) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    plugged_in_label = plugged_in.map({1.0: "Yes", 0.0: "No"})
    fig.add_trace(go.Scatter(
        x=soc.index, y=plugged_in, mode="lines", line_shape="vh", fill="tozeroy",
        fillcolor=COLOR_OCCUPANCY_FILL, line=dict(width=0), name="Plugged in",
        customdata=plugged_in_label, hovertemplate="Plugged in: %{customdata}<extra></extra>",
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=soc.index, y=soc, mode="lines",
        line=dict(color=COLOR_SOC, width=2.5), name="SoC", customdata=state,
        hovertemplate="SoC: %{y:.3f}<br>State: %{customdata}<extra></extra>",
    ), secondary_y=False)
    fig.update_yaxes(title_text="SoC", range=auto_range(soc), gridcolor="rgba(0,0,0,0.06)", secondary_y=False)
    fig.update_yaxes(visible=False, range=[0, 1], secondary_y=True)
    fig.update_layout(xaxis_title="Time", xaxis=dict(tickformat="%a %H:%M"), hovermode="x unified")
    mark_today(fig, midnight)
    mark_weekends(fig, soc.index)
    return fig


def build_population_chart(bands: pd.DataFrame, pct_plugged_in: pd.Series, midnight: datetime) -> go.Figure:
    edge = dict(color=COLOR_BAND_EDGE, width=1, dash="dot")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=bands.index, y=bands["p05"], mode="lines", line=edge,
        showlegend=False, hovertemplate="p05: %{y:.3f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=bands.index, y=bands["p95"], mode="lines", line=edge,
        fill="tonexty", fillcolor=COLOR_BAND_FILL,
        name="p05-p95 range", hovertemplate="p95: %{y:.3f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=bands.index, y=bands["mean"], mode="lines", line=dict(color=COLOR_SOC, width=2.5),
        name="Mean SoC", hovertemplate="mean: %{y:.3f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=pct_plugged_in.index, y=pct_plugged_in.values, name="% plugged in",
        marker=dict(color=COLOR_OCCUPANCY, line_width=0),
    ), secondary_y=True)

    soc_lo = min(float(bands.to_numpy().min()), 1.0)
    soc_pad = max((1.0 - soc_lo) * 0.08, 0.01)
    fig.update_yaxes(title_text="SoC (weighted mean, p05-p95 range)", range=[soc_lo - soc_pad, 1.0], gridcolor="rgba(0,0,0,0.06)", secondary_y=False)
    fig.update_yaxes(title_text="% plugged in", range=auto_range(pct_plugged_in), showgrid=False, secondary_y=True)
    fig.update_layout(
        xaxis_title="Time", xaxis=dict(tickformat="%a %H:%M"), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    mark_today(fig, midnight)
    mark_weekends(fig, bands.index)
    return fig


def build_price_chart(prices: pd.Series) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[t.strftime("%H:%M") for t in prices.index], y=prices.values, mode="lines", line=dict(color=COLOR_PRICE, width=2)))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
    fig.update_layout(yaxis_title="£/MWh", xaxis_title="Time")
    return fig


def savings_column_config() -> dict:
    return {
        "£/kWh": st.column_config.NumberColumn(format="£%.4f"),
        "Total cost (£)": st.column_config.NumberColumn(format="£%.2f"),
        "Energy (kWh)": st.column_config.NumberColumn(format="%.1f"),
        "Savings vs Average (UK) (£)": st.column_config.NumberColumn(format="£%.2f"),
    }


def render_controls(now: datetime, latest: date, earliest: date) -> tuple[date, str]:
    archetypes = {name: get_archetype(name) for name in ARCHETYPE_NAMES}
    cols = st.columns(2)
    with cols[0]:
        sim_date = st.date_input(
            "Day to analyse",
            value=now.date(),
            min_value=earliest,
            max_value=now.date(),
        )
    with cols[1]:
        selected_name = st.selectbox("Archetype", ARCHETYPE_NAMES)
    return sim_date, selected_name


def render_plugin_behaviour(population: PopulationResult, name: str, window_start: datetime, end_dt: datetime, midnight: datetime) -> None:
    soc, plugged_in, state = sample_run_trajectory(population, name, window_start, end_dt)

    st.header("Plug-in behaviour")
    if name == "intelligent_octopus":
        archetype = get_archetype(name)
        session = charging_session_window(state, archetype)
        if session is not None:
            arrival, deadline = session
            pos = soc.index.get_loc(arrival)
            arrival_soc = soc.iloc[pos - 1] if pos > 0 else soc.iloc[pos]
            slots = scheduled_slots(archetype, arrival, arrival_soc, deadline)
            slots_text = ", ".join(f"{row.Time} ({row._2:.2f})" for row in slots.itertuples())
            st.caption(
                f"Charging slots between {arrival.strftime('%a %H:%M')} and "
                f"{deadline.strftime('%a %H:%M')} (£/MWh): {slots_text}"
            )
        else:
            st.caption("Didn't plug in and charge within this window.")
    st.plotly_chart(build_soc_chart(soc, plugged_in, state, midnight), width="stretch")


def render_population(population: PopulationResult, window_start: datetime, end_dt: datetime, midnight: datetime, note: str) -> None:
    st.header("Population on this day")
    st.caption(
        f"{RUNS_PER_ARCHETYPE} runs per archetype, weighted by population share - all 6 archetypes, "
        f"from {EVENING_START.strftime('%-I%p').lower()} the day before (T-1) through {note}"
    )
    bands, pct_plugged_in = population_summary(population, window_start, end_dt)
    st.plotly_chart(build_population_chart(bands, pct_plugged_in, midnight), width="stretch")


def render_savings(population: PopulationResult, archetypes: dict[str, ArchetypeConfig], sim_date: date, earliest: date) -> None:
    st.header("Savings")
    st.caption(
        "£/kWh averaged across the runs that charged that day - the fair comparison across archetypes "
        "since they use very different amounts of energy (Intelligent Octopus has 3x Average UK's annual "
        "mileage). Falls back to the previous day where an archetype hasn't charged yet today."
    )
    st.dataframe(
        savings_table(population, archetypes, sim_date, earliest),
        hide_index=True,
        width="stretch",
        column_config=savings_column_config(),
    )


def render_price_curve(sim_date: date) -> None:
    st.header("Price curve")
    st.plotly_chart(build_price_chart(get_prices(sim_date)), width="stretch")


def main() -> None:
    st.set_page_config(page_title="EV Charging Behaviour Simulator", layout="wide")
    st.title("EV Charging Behaviour Simulator")

    try:
        update_day_ahead_prices(MARKET_INDEX_CSV)
    except Exception:
        pass
    now = datetime.now()
    latest = latest_price_date()
    earliest = now.date() - timedelta(days=LOOKBACK_DAYS)

    sim_date, selected_name = render_controls(now, latest, earliest)
    archetypes = {name: get_archetype(name) for name in ARCHETYPE_NAMES}

    window_start, midnight, end_dt = sim_window(sim_date, now)
    note = time_note(sim_date, now)
    spinner_text = (
        f"First-time setup: simulating {RUNS_PER_ARCHETYPE * len(ARCHETYPE_NAMES)} runs "
        "across 6 archetypes. Takes a few minutes, only happens once - every later "
        "visit reads from cache and is instant."
        if is_cache_cold(RUNS_PER_ARCHETYPE) else "Loading..."
    )
    with st.spinner(spinner_text):
        population = get_population_runs(end_dt, RUNS_PER_ARCHETYPE)

    render_plugin_behaviour(population, selected_name, window_start, end_dt, midnight)
    render_population(population, window_start, end_dt, midnight, note)
    render_savings(population, archetypes, sim_date, earliest)
    render_price_curve(sim_date)


if __name__ == "__main__":
    main()
