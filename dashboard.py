from datetime import date, datetime, time, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from archetypes import ArchetypeConfig
from elexon_client import update_day_ahead_prices
from simulation import (
    LOOKBACK_DAYS,
    MARKET_INDEX_CSV,
    PopulationResult,
    get_archetype,
    get_population_runs,
    get_prices,
    latest_price_date,
    plugged_in_share,
    slice_day,
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
RUNS_PER_ARCHETYPE = 200  # cheap: runs are cached and extended, not recomputed each view
EVENING_START = time(18, 0)  # both timeline charts start here the day before, for overnight context

# Shared palette - one look across all charts, not styled per-chart.
COLOR_SOC = "#1864ab"
COLOR_BAND_EDGE = "#74c0fc"
COLOR_BAND_FILL = "rgba(24, 100, 171, 0.12)"
COLOR_OCCUPANCY = "rgba(134, 142, 150, 0.35)"
COLOR_OCCUPANCY_FILL = "rgba(134, 142, 150, 0.25)"
COLOR_PRICE = "#495057"
COLOR_TODAY_MARKER = "#868e96"
COLOR_WEEKEND = "#845ef7"


# --- Pure data helpers -------------------------------------------------

def auto_range(values, pad_frac: float = 0.1) -> list[float]:
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * pad_frac or 0.05
    return [lo - pad, hi + pad]


def archetype_columns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[[c for c in df.columns if c.startswith(f"{name}_")]]


def slice_window(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return df[(df.index >= start) & (df.index < end)]


def sim_window(sim_date: date, now: datetime) -> tuple[datetime, datetime, datetime]:
    """(start, midnight, end) of the chart window: 6pm the day before, the
    sim_date/T-1 boundary, and now (or end of sim_date if it's a past date)."""
    start = datetime.combine(sim_date - timedelta(days=1), EVENING_START)
    midnight = datetime.combine(sim_date, time.min)
    if sim_date == now.date():
        end = now
    else:
        end = datetime.combine(sim_date + timedelta(days=1), time.min)
    return start, midnight, end


def time_note(sim_date: date, now: datetime) -> str:
    return f"up to {now.strftime('%H:%M')} today" if sim_date == now.date() else "the full day"


def _require_rows(df: pd.DataFrame, population: PopulationResult, window_start: datetime, end_dt: datetime) -> None:
    """Fail clearly if the window is before the cache's earliest data,
    instead of an empty DataFrame crashing later with no useful message."""
    if df.empty:
        cached_from = population["soc"].index.min()
        raise ValueError(
            f"No simulated data between {window_start} and {end_dt}. The cached "
            f"runs start at {cached_from} - pick a more recent date."
        )


def median_trajectory(population: PopulationResult, name: str, window_start: datetime, end_dt: datetime) -> tuple[pd.Series, pd.Series]:
    """Median SoC and majority-vote plugged-in status across one archetype's
    runs - weighted_quantiles/plugged_in_share with every run weighted
    equally, the same reductions the population view uses with real weights."""
    soc_runs = slice_window(archetype_columns(population["soc"], name), window_start, end_dt)
    state_runs = slice_window(archetype_columns(population["state"], name), window_start, end_dt)
    _require_rows(soc_runs, population, window_start, end_dt)
    uniform_weights = pd.Series(1.0, index=soc_runs.columns)
    median_soc = weighted_quantiles(soc_runs, uniform_weights, [0.5])[0.5]
    median_plugged_in = (plugged_in_share(state_runs) >= 0.5).astype(float)
    return median_soc, median_plugged_in


def population_summary(population: PopulationResult, window_start: datetime, end_dt: datetime) -> tuple[pd.DataFrame, pd.Series]:
    """Weighted SoC percentile bands and weighted %-plugged-in, across all archetypes."""
    weights = population["weights"]
    pop_soc = slice_window(population["soc"], window_start, end_dt)
    pop_state = slice_window(population["state"], window_start, end_dt)
    _require_rows(pop_soc, population, window_start, end_dt)

    pct_plugged_in = plugged_in_share(pop_state, weights) * 100
    bands = weighted_quantiles(pop_soc, weights, [0.05, 0.5, 0.95])
    bands.columns = ["p05", "p50", "p95"]
    return bands, pct_plugged_in


def cost_totals(population: PopulationResult, name: str, archetype: ArchetypeConfig, d: date) -> tuple[pd.Series, pd.Series]:
    """Per-run total cost and kWh delivered on day `d`, one value per run."""
    day_soc = slice_day(archetype_columns(population["soc"], name), d)
    day_cost = slice_day(archetype_columns(population["cost"], name), d)
    total_kwh = day_soc.diff().clip(lower=0).sum() * archetype.battery_kwh
    total_cost = day_cost.sum()
    return total_cost, total_kwh


def savings_row(population: PopulationResult, name: str, archetype: ArchetypeConfig, sim_date: date, earliest: date) -> dict:
    """£/kWh, averaged across whichever runs actually charged that day. Falls
    back to the previous day if none did yet (e.g. a partial "today")."""
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


# --- Chart builders (pure: data in, go.Figure out) ----------------------

def mark_today(fig: go.Figure, midnight: datetime) -> None:
    """Vertical line + label at midnight, separating T-1 from sim_date."""
    fig.add_vline(x=midnight, line_dash="dash", line_color=COLOR_TODAY_MARKER)
    fig.add_annotation(x=midnight, y=1.03, yref="paper", showarrow=False, text="T-1 | Today", font=dict(size=11, color=COLOR_TODAY_MARKER))


def mark_weekends(fig: go.Figure, index: pd.DatetimeIndex) -> None:
    """Light shading behind Saturday/Sunday - weekend transition tables
    differ (errands, not commutes), so this should be visible at a glance."""
    start, end = index.min(), index.max()
    day = start.date()
    while day <= end.date():
        if day.weekday() == 5:  # Saturday
            span_start = max(start, datetime.combine(day, time.min))
            span_end = min(end, datetime.combine(day + timedelta(days=2), time.min))
            if span_start < span_end:
                fig.add_vrect(x0=span_start, x1=span_end, fillcolor=COLOR_WEEKEND, opacity=0.08, layer="below", line_width=0)
        day += timedelta(days=1)


def build_soc_chart(median_soc: pd.Series, median_plugged_in: pd.Series, midnight: datetime) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=median_soc.index, y=median_plugged_in, mode="lines", line_shape="hv", fill="tozeroy",
        fillcolor=COLOR_OCCUPANCY_FILL, line=dict(width=0), name="Plugged in",
    ), secondary_y=True)
    fig.add_trace(go.Scatter(x=median_soc.index, y=median_soc, mode="lines", line=dict(color=COLOR_SOC, width=2.5), name="SoC"), secondary_y=False)
    fig.update_yaxes(title_text="SoC", range=auto_range(median_soc), gridcolor="rgba(0,0,0,0.06)", secondary_y=False)
    fig.update_yaxes(visible=False, range=[0, 1], secondary_y=True)
    fig.update_layout(xaxis_title="Time", xaxis=dict(tickformat="%a %H:%M"))
    mark_today(fig, midnight)
    mark_weekends(fig, median_soc.index)
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
        x=bands.index, y=bands["p50"], mode="lines", line=dict(color=COLOR_SOC, width=2.5),
        name="Median SoC", hovertemplate="median: %{y:.3f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Bar(
        x=pct_plugged_in.index, y=pct_plugged_in.values, name="% plugged in",
        marker=dict(color=COLOR_OCCUPANCY, line_width=0),
    ), secondary_y=True)

    soc_lo = min(float(bands.to_numpy().min()), 1.0)
    soc_pad = max((1.0 - soc_lo) * 0.08, 0.01)
    fig.update_yaxes(title_text="SoC (weighted percentile)", range=[soc_lo - soc_pad, 1.0], gridcolor="rgba(0,0,0,0.06)", secondary_y=False)
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


# --- Streamlit rendering (imperative shell: one function per section) --

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
    st.header("Plug-in behaviour")
    median_soc, median_plugged_in = median_trajectory(population, name, window_start, end_dt)
    st.plotly_chart(build_soc_chart(median_soc, median_plugged_in, midnight), use_container_width=True)


def render_population(population: PopulationResult, window_start: datetime, end_dt: datetime, midnight: datetime, note: str) -> None:
    st.header("Population on this day")
    st.caption(
        f"{RUNS_PER_ARCHETYPE} runs per archetype, weighted by population share - all 6 archetypes, "
        f"from {EVENING_START.strftime('%-I%p').lower()} the day before (T-1) through {note}"
    )
    bands, pct_plugged_in = population_summary(population, window_start, end_dt)
    st.plotly_chart(build_population_chart(bands, pct_plugged_in, midnight), use_container_width=True)


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
        use_container_width=True,
        column_config=savings_column_config(),
    )


def render_price_curve(sim_date: date) -> None:
    st.header("Price curve")
    st.plotly_chart(build_price_chart(get_prices(sim_date)), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="EV Charging Behaviour Simulator", layout="wide")
    st.title("EV Charging Behaviour Simulator")

    update_day_ahead_prices(MARKET_INDEX_CSV)
    now = datetime.now()
    latest = latest_price_date()
    earliest = now.date() - timedelta(days=LOOKBACK_DAYS)  # matches what the cache guarantees

    sim_date, selected_name = render_controls(now, latest, earliest)
    archetypes = {name: get_archetype(name) for name in ARCHETYPE_NAMES}

    window_start, midnight, end_dt = sim_window(sim_date, now)
    note = time_note(sim_date, now)
    with st.spinner("Fetching simulated runs..."):
        population = get_population_runs(end_dt, RUNS_PER_ARCHETYPE)

    render_plugin_behaviour(population, selected_name, window_start, end_dt, midnight)
    render_population(population, window_start, end_dt, midnight, note)
    render_savings(population, archetypes, sim_date, earliest)
    render_price_curve(sim_date)


if __name__ == "__main__":
    main()
