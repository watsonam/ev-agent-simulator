from datetime import date, datetime, time, timedelta
from typing import cast
from zoneinfo import ZoneInfo

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
    get_day_type,
    get_population_runs,
    get_prices,
    latest_price_date,
    next_occurrence,
    plugged_in_share,
    prices_in_window,
    weighted_mean,
    weighted_quantiles,
)

UK_TZ = ZoneInfo("Europe/London")


def uk_now() -> datetime:
    return datetime.now(UK_TZ).replace(tzinfo=None)

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
COLOR_CHARGING_FILL = "rgba(24, 100, 171, 0.22)"
COLOR_CHARGING_BAR = "rgba(24, 100, 171, 0.45)"
COLOR_PRICE = "#495057"
COLOR_TODAY_MARKER = "#868e96"
COLOR_WEEKEND = "#845ef7"


def auto_range(values, pad_frac: float = 0.1) -> list[float]:
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * pad_frac or 0.05
    return [lo - pad, hi + pad]


def archetype_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return cast(pd.DataFrame, df[[c for c in df.columns if c.startswith(f"{name}_")]])


def slice_window(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return cast(pd.DataFrame, df[(df.index >= start) & (df.index < end)])


def sim_window(sim_date: date, now: datetime) -> tuple[datetime, datetime, datetime]:
    start = datetime.combine(sim_date - timedelta(days=1), EVENING_START)
    midnight = datetime.combine(sim_date, time.min)
    if sim_date == now.date():
        end = now
    else:
        end = datetime.combine(sim_date + timedelta(days=1), time.min)
    return start, midnight, end


def time_note(sim_date: date, now: datetime) -> str:
    return f"to {now.strftime('%H:%M')} today" if sim_date == now.date() else "to end of day"


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
    state_df = cast(pd.DataFrame, population["state"][[run_col]])
    soc_df = cast(pd.DataFrame, population["soc"][[run_col]])
    state_run = cast(pd.Series, slice_window(state_df, window_start, end_dt)[run_col]).dropna()
    soc_run = cast(pd.Series, slice_window(soc_df, window_start, end_dt)[run_col]).reindex(state_run.index)
    _require_rows(state_run.to_frame(), population, window_start, end_dt)
    plugged_in = state_run.isin([State.PLUGGED_CHARGING, State.PLUGGED_IDLE]).astype(float)
    state_names = state_run.map(lambda s: s.name)
    return soc_run, plugged_in, state_names


def charging_session_window(state: pd.Series, archetype: ArchetypeConfig) -> tuple[datetime, datetime] | None:
    plugged = state.isin(["PLUGGED_CHARGING", "PLUGGED_IDLE"])
    arrivals = plugged & ~plugged.shift(1, fill_value=False)
    if not arrivals.any():
        return None
    arrival_time = arrivals[arrivals].index[-1]
    deadline = next_occurrence(arrival_time, arrival_time.date(), archetype.plugout_time)
    return arrival_time, deadline


def scheduled_slots(archetype: ArchetypeConfig, arrival: datetime, arrival_soc: float, deadline: datetime) -> pd.DataFrame:
    schedule = build_charging_schedule(archetype, arrival, arrival_soc, deadline)
    charged = cast(pd.Series, schedule[schedule > 0]).index
    prices = prices_in_window(arrival, deadline).reindex(charged)
    return pd.DataFrame({"Time": pd.DatetimeIndex(prices.index).strftime("%a %H:%M"), "£/MWh": prices.values})


def population_summary(population: PopulationResult, window_start: datetime, end_dt: datetime) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    weights = population["weights"]
    pop_soc = slice_window(population["soc"], window_start, end_dt)
    pop_state = slice_window(population["state"], window_start, end_dt)
    _require_rows(pop_soc, population, window_start, end_dt)

    pct_plugged_in = plugged_in_share(pop_state, weights) * 100
    pct_charging = plugged_in_share(pop_state, weights, states=(State.PLUGGED_CHARGING,)) * 100
    bands = weighted_quantiles(pop_soc, weights, [0.05, 0.95])
    bands.columns = ["p05", "p95"]
    bands["mean"] = weighted_mean(pop_soc, weights)
    return bands, pct_plugged_in, pct_charging


def cost_totals(population: PopulationResult, name: str, archetype: ArchetypeConfig) -> tuple[pd.Series, pd.Series]:
    energy = archetype_columns(population["soc"], name).diff().clip(lower=0) * archetype.battery_kwh
    return archetype_columns(population["cost"], name).sum(), energy.sum()


def savings_row(population: PopulationResult, name: str, archetype: ArchetypeConfig) -> dict:
    total_cost, total_kwh = cost_totals(population, name, archetype)
    charged = total_kwh > 0
    per_kwh = (total_cost[charged] / total_kwh[charged]).mean() if charged.any() else None
    return {
        "Archetype": archetype.name,
        "£/kWh": per_kwh,
        "Total cost (£)": total_cost[charged].mean() if charged.any() else None,
        "Energy (kWh)": total_kwh[charged].mean() if charged.any() else None,
    }


def savings_table(population: PopulationResult, archetypes: dict[str, ArchetypeConfig]) -> pd.DataFrame:
    df = pd.DataFrame([savings_row(population, name, a) for name, a in archetypes.items()])
    baseline = df.loc[df["Archetype"] == archetypes["average_uk"].name, "£/kWh"].iloc[0]
    df["Savings vs Average (UK) (£)"] = (baseline - df["£/kWh"]) * df["Energy (kWh)"]
    return df


def mark_today(fig: go.Figure, midnight: datetime) -> None:
    fig.add_vline(x=midnight, line_dash="dash", line_color=COLOR_TODAY_MARKER)
    fig.add_annotation(x=midnight, y=1.03, yref="paper", showarrow=False, text="T-1 | Today", font=dict(size=11, color=COLOR_TODAY_MARKER))


def mark_weekends(fig: go.Figure, index: pd.DatetimeIndex) -> None:
    start = cast(pd.Timestamp, index.min())
    end = cast(pd.Timestamp, index.max())
    day = start.date()
    while day <= end.date():
        if day.weekday() == 5:
            span_start = max(start, pd.Timestamp(datetime.combine(day, time.min)))
            span_end = min(end, pd.Timestamp(datetime.combine(day + timedelta(days=2), time.min)))
            if span_start < span_end:
                fig.add_vrect(x0=span_start, x1=span_end, fillcolor=COLOR_WEEKEND, opacity=0.08, layer="below", line_width=0)
        day += timedelta(days=1)


def build_soc_chart(soc: pd.Series, plugged_in: pd.Series, state: pd.Series, midnight: datetime) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    plugged_in_label = plugged_in.map({1.0: "Yes", 0.0: "No"})
    charging = (state == "PLUGGED_CHARGING").astype(float)
    fig.add_trace(go.Scatter(
        x=soc.index, y=plugged_in, mode="lines", line_shape="vh", fill="tozeroy",
        fillcolor=COLOR_OCCUPANCY_FILL, line=dict(width=0), name="Plugged in",
        customdata=plugged_in_label, hovertemplate="Plugged in: %{customdata}<extra></extra>",
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=soc.index, y=charging, mode="lines", line_shape="vh", fill="tozeroy",
        fillcolor=COLOR_CHARGING_FILL, line=dict(width=0), name="Charging", hoverinfo="skip",
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
    mark_weekends(fig, pd.DatetimeIndex(soc.index))
    return fig


def build_population_chart(bands: pd.DataFrame, pct_plugged_in: pd.Series, pct_charging: pd.Series, midnight: datetime) -> go.Figure:
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
    fig.add_trace(go.Bar(
        x=pct_charging.index, y=pct_charging.values, name="% charging",
        marker=dict(color=COLOR_CHARGING_BAR, line_width=0),
    ), secondary_y=True)

    soc_lo = min(float(bands.to_numpy().min()), 1.0)
    soc_pad = max((1.0 - soc_lo) * 0.08, 0.01)
    fig.update_yaxes(title_text="SoC (weighted mean, p05-p95 range)", range=[soc_lo - soc_pad, 1.0], gridcolor="rgba(0,0,0,0.06)", secondary_y=False)
    fig.update_yaxes(title_text="% plugged in", range=auto_range(pct_plugged_in), showgrid=False, secondary_y=True)
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Time", xaxis=dict(tickformat="%a %H:%M"), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    mark_today(fig, midnight)
    mark_weekends(fig, pd.DatetimeIndex(bands.index))
    return fig


def build_price_chart(prices: pd.Series) -> go.Figure:
    fig = go.Figure(go.Scatter(x=[cast(time, t).strftime("%H:%M") for t in prices.index], y=prices.values, mode="lines", line=dict(color=COLOR_PRICE, width=2)))
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


def render_plugin_behaviour(population: PopulationResult, name: str, window_start: datetime, end_dt: datetime, midnight: datetime, day_type: str) -> None:
    soc, plugged_in, state = sample_run_trajectory(population, name, window_start, end_dt)

    st.header(f"Plug-in behaviour ({day_type})")
    if name == "intelligent_octopus":
        archetype = get_archetype(name)
        session = charging_session_window(state, archetype)
        if session is not None:
            arrival, deadline = session
            pos = cast(int, soc.index.get_loc(arrival))
            arrival_soc = soc.iloc[pos - 1] if pos > 0 else soc.iloc[pos]
            slots = scheduled_slots(archetype, arrival, arrival_soc, deadline)
            slots_text = ", ".join(f"{t} ({p:.2f})" for t, p in zip(slots["Time"], slots["£/MWh"]))
            st.caption(
                f"Charging slots between {arrival.strftime('%a %H:%M')} and "
                f"{deadline.strftime('%a %H:%M')} (£/MWh): {slots_text}"
            )
        else:
            st.caption("Didn't plug in and charge within this window.")
    st.plotly_chart(build_soc_chart(soc, plugged_in, state, midnight), width="stretch")


def render_population(population: PopulationResult, window_start: datetime, end_dt: datetime, midnight: datetime, note: str, day_type: str) -> None:
    st.header(f"Population on this day ({day_type})")
    st.caption(
        f"Mean state of charge (5th-95th percentile band) and the share of cars plugged in, "
        f"across all 6 archetypes weighted by population share ({RUNS_PER_ARCHETYPE} runs each). "
        f"From {EVENING_START.strftime('%-I%p').lower()} the day before (T-1) {note}."
    )
    bands, pct_plugged_in, pct_charging = population_summary(population, window_start, end_dt)
    st.plotly_chart(build_population_chart(bands, pct_plugged_in, pct_charging, midnight), width="stretch")


def render_savings(population: PopulationResult, archetypes: dict[str, ArchetypeConfig]) -> None:
    idx = pd.DatetimeIndex(population["soc"].index)
    days = (cast(pd.Timestamp, idx.max()).date() - cast(pd.Timestamp, idx.min()).date()).days + 1
    st.header("Savings")
    st.caption(
        f"Over the last {days} days. £/kWh is each run's total cost over its total energy, averaged across runs. "
        "Total cost and energy are per-run means. Savings is Average (UK)'s rate applied to the same energy."
    )
    st.dataframe(
        savings_table(population, archetypes),
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
    now = uk_now()
    latest = latest_price_date()
    earliest = now.date() - timedelta(days=LOOKBACK_DAYS)

    sim_date, selected_name = render_controls(now, latest, earliest)
    archetypes = {name: get_archetype(name) for name in ARCHETYPE_NAMES}

    window_start, midnight, end_dt = sim_window(sim_date, now)
    note = time_note(sim_date, now)
    spinner_text = (
        f"First-time setup: simulating {RUNS_PER_ARCHETYPE * len(ARCHETYPE_NAMES)} runs "
        "across 6 archetypes. Reads from cache from then on."
        if is_cache_cold(RUNS_PER_ARCHETYPE) else "Loading..."
    )
    with st.spinner(spinner_text):
        population = get_population_runs(end_dt, RUNS_PER_ARCHETYPE)

    day_type = get_day_type(sim_date)
    render_plugin_behaviour(population, selected_name, window_start, end_dt, midnight, day_type)
    render_population(population, window_start, end_dt, midnight, note, day_type)
    render_savings(population, archetypes)
    render_price_curve(sim_date)


if __name__ == "__main__":
    main()
