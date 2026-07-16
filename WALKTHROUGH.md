# Code walkthrough

A map of how the simulator works, so you can read it top-down and extend it.

## The one-sentence version

Each simulated driver is a **state machine** stepped in 30-minute slots; we run
many of them (a **Monte Carlo**), cache each run to disk so it can be **resumed**
rather than recomputed, then the dashboard reads those runs back and draws them.

## The four files

| File | Role |
|---|---|
| `archetypes.py` | The **inputs**. Six driver types and their numbers (mileage, battery, plug-in times, transition probabilities). Pure config, no simulation logic. |
| `simulation.py` | The **engine**. The state machine, the Monte Carlo loop, the disk cache, and the weighted-percentile maths. |
| `elexon_client.py` | The **prices**. Fetches EPEX day-ahead electricity prices and keeps `data/market_index.csv` up to date. |
| `streamlit_app.py` | The **view**. Streamlit UI: reads the runs, slices the window it needs, draws the charts. No simulation logic, and no statistics beyond what `simulation.py` already provides. |

Data flows one way: `archetypes → simulation → dashboard`. Prices feed in from
`elexon_client`. The dashboard never simulates; it only reads and plots.

---

## 1. The state machine (`simulation.py`)

A driver is always in exactly one of four **states** (`State` enum):

- `DRIVING` – on the road, battery draining
- `PARKED` – stopped somewhere with no charger
- `PLUGGED_CHARGING` – plugged in and adding charge
- `PLUGGED_IDLE` – plugged in but full (target reached), sitting there

Every 30 minutes we look at the current state and roll to decide the next one.
The probabilities come from the archetype's **transition tables** (e.g.
"between 7:00 and 8:30 there's a rising chance of leaving for work"). That is
the whole model: **`advance()`** is a `while` loop that steps time forward one
slot at a time, updating `state`, `soc` (state of charge, 0–1), and `cost`.

Three **charging strategies** decide what `PLUGGED_CHARGING` actually does:

- `IMMEDIATE` – charge every slot until full (most archetypes)
- `SCHEDULED_PRICE` – Intelligent Octopus: pick the *cheapest* slots overnight
- `FIXED_TIME` – wait until a set clock time, then charge

### The functions that make it up

- `initial_state(archetype, start_date)` → a fresh `RunState`: plugged in at the
  archetype's usual plug-in time, at its usual starting charge.
- `advance(archetype, run_state, end_time)` → runs the loop from `run_state`'s
  current time up to `end_time`. Returns **(new rows, new RunState)**.
- `simulate_week(...)` → convenience wrapper: `initial_state` then `advance` for
  N days. Use this when you just want one run and don't care about resuming.

**`RunState`** is the key idea: it's a snapshot of *everything needed to pause
and resume* a run — current time, current charge, current state, and any
half-finished charging plan. Because it's just a few plain values, we can save
it to disk and pick the run back up later exactly where it stopped.

---

## 2. Monte Carlo + the incremental cache

We don't simulate one driver, we simulate **200 per archetype** and look at the
spread. That's the Monte Carlo. Naively you'd re-run all of them every time the
user changes something — slow. Instead each run is **cached and resumed**.

### `get_or_advance_run(archetype, run_number, end_dt)` — the heart of it

This is the function you flagged. Read it as three steps:

1. **Load or start.** Each `(archetype, run_number)` pair has one pickle file:
   `data/cache/runs/average_uk_0.pkl`, `average_uk_1.pkl`, … Inside is a tuple
   `(trajectory, run_state)`:
   - `trajectory` = every slot simulated so far (a DataFrame of state/soc/cost)
   - `run_state` = the `RunState` snapshot of where it paused
   If the file doesn't exist yet, we start fresh `LOOKBACK_DAYS` (7) before
   `end_dt`, so the run has a week of warm-up and isn't a cold reset on the day
   you're viewing.

2. **Advance only if needed.** If the cached run already reaches `end_dt`, do
   nothing. Otherwise `advance()` it forward to `end_dt` and append the new rows.
   *This is the payoff:* history already simulated is never redone.

3. **Save and return.** Write the extended `(trajectory, run_state)` back to the
   same file, and return the full trajectory.

If `advance()` needs a price that doesn't exist yet (asking past the last day in
`market_index.csv`), it raises `ValueError` and `get_or_advance_run` lets it
propagate - no retry, no silent truncation. That mostly matters for Intelligent
Octopus, whose overnight schedule looks up to a day past "now"; `elexon_client`
fetches through tomorrow (not just today) specifically so that lookup normally
has data. If it's still missing - tomorrow's day-ahead auction hasn't published
yet - the function fails loudly rather than quietly showing a shorter window.

The same principle applies one level down, inside the slot loop itself: a date
can be *present* in `market_index.csv` but missing specific half-hour rows (a
partial fetch, which happened for real - see "Known simplifications" below).
`get_prices()`'s empty-day check doesn't catch that. So the cost calculation
only looks the price up when energy was actually delivered that slot, and
raises if it's missing, rather than defaulting to £0 - a slot that's never
charged doesn't care what the price was, but one that is must have a real
number, or the whole day's cost is silently wrong.

Cache writes are atomic (write to a `.<uuid>.pkl.tmp` file, then rename over
the real one), so a killed process can never leave a truncated file. The temp
name includes a UUID, not just the PID - Streamlit runs each user session as
a thread inside one shared process, so concurrent sessions have the same PID
and would otherwise race on the same temp file.

### A concrete timeline

- **Monday, first ever load.** No cache files. All 1,200 runs (200 × 6) simulate
  a warm-up week up to Monday. Slow (~5 min), one time only. Files written.
- **Monday, change the archetype dropdown.** Every run's cache already reaches
  Monday → step 2 does nothing → instant.
- **Tuesday.** Each run is 1 day behind → `advance()` adds one day → fast.

So the cost is paid once, then each later view is cheap. That's the entire
reason the code is shaped this way.

### `get_population_runs(end_dt, runs_per_archetype)`

Loops all 6 archetypes × 200 runs, calling `get_or_advance_run` for each, and
packs them into a **`PopulationResult`** — four wide tables keyed by run:

```
{ "soc":   DataFrame (time × 1200 runs),
  "state": DataFrame (time × 1200 runs),
  "cost":  DataFrame (time × 1200 runs),
  "weights": Series  (one weight per run) }
```

Columns are named `average_uk_0`, `average_uk_1`, …, `intelligent_octopus_0`, …

### The weights (population share)

The six archetypes aren't equally common. Each run's **weight** is
`population_share / runs_per_archetype`, so all of Average UK's 200 runs together
sum to 0.40 (its 40% share), Intelligent Octopus's to 0.30, etc. We run the same
*count* of each (so rare types are still well-sampled) but down-weight them at
the end.

### The two reductions everything else is built from

Two functions turn a wide DataFrame of runs into one line/band per time slot.
Both live in `simulation.py`, next to each other, and both take an optional
`weights` argument (`None` = every run counts equally):

- **`weighted_quantiles(df, weights, quantiles)`** — row-wise weighted quantile.
  With `quantiles=[0.5]` and uniform weights it's a plain median; with
  `[0.05, 0.5, 0.95]` and real population weights it's the population bands.
- **`plugged_in_share(state_df, weights=None)`** — weighted fraction of runs
  plugged in per slot. Uniform weights → one archetype's own share; population
  weights → the whole population's share.

`streamlit_app.py` never recomputes these - it only decides *which weights to pass*:
uniform for a single archetype's "typical day" view, `population["weights"]`
for the population view. That's the only difference between the two charts'
underlying maths.

---

## 3. The dashboard (`streamlit_app.py`)

Structured in three layers, top to bottom in the file:

1. **Pure data helpers** — take the `PopulationResult`, return numbers. No
   Streamlit, no plotting, and no statistics of their own: `median_trajectory`
   and `population_summary` both just slice the runs they need and hand them to
   `weighted_quantiles`/`plugged_in_share` in `simulation.py` (see above) with
   different weights. `savings_table` is the one helper with its own maths
   (per-run £/kWh), since nothing in `simulation.py` already does that.
2. **Chart builders** — take numbers, return a Plotly `Figure`. Pure.
3. **`render_*` functions** — the only place that touches `st.*`. Each owns one
   page section (header + caption + chart).

`main()` ties it together and reads as a flat list of steps: get inputs → work
out the time window → fetch the population once → render each section.
`is_cache_cold()` decides the spinner text shown during that fetch - a rough
"do we have anywhere near enough cached runs yet" check, purely cosmetic
(tells first-time visitors to expect a few minutes; doesn't change behaviour).

### Key transforms

- **Individual chart** = the *median* across one archetype's 200 runs. Median
  SoC is a clean line; "plugged in" is shaded where *more than half* the runs
  are plugged in (so it stays a crisp on/off block, not a blur).
- **Population chart** = *weighted* percentile bands (p05/p50/p95) across all
  1,200 runs, plus weighted % plugged-in as bars.
- **T-1 window**: both charts start at 6pm the evening *before* the selected day,
  so you see the overnight charge in context. `midnight` draws the divider line.

---

## How to extend it

**Add a new archetype.** In `archetypes.py`, add a factory method returning an
`ArchetypeConfig` (copy an existing one, change the numbers/transition tables),
then add it to `_ARCHETYPES` in `simulation.py` and `ARCHETYPE_NAMES` in
`streamlit_app.py`. Delete `data/cache/runs/` so it re-simulates with the new type.

**Change the number of runs.** `RUNS_PER_ARCHETYPE` in `streamlit_app.py`. More runs
= smoother bands, slower first load. Cache is per-run so raising it only
simulates the *new* runs.

**Add a chart.** Write a pure `build_*` function (numbers → Figure) and a
`render_*` function (calls it, adds header/caption), then call the `render_*`
from `main()`. Keep the maths in a data helper, not in the chart builder.

**Change driver behaviour.** That's `advance()` in `simulation.py` — the state
machine. After changing it, delete `data/cache/runs/` so runs re-simulate.

> **The cache is keyed only by archetype + run number, not by the logic.** If
> you change `advance()`, an archetype's numbers, or `LOOKBACK_DAYS`, the old
> pickles are stale. Delete `data/cache/runs/` to force a clean re-simulation.

**Errors aren't caught and worked around.** If `advance()` can't get a price it
needs, it raises and that propagates all the way up - no retry, no fallback
window. Keep new code in this style: if something's missing, fail with a clear
message rather than silently doing less than what was asked.

## Known simplifications

- **Long trips** (e.g. 150-mile weekend drives) are folded into the average
  weekend daily energy, not simulated as occasional large events. Totals are
  right; the day-to-day *shape* is smoother than reality. See
  `long_trip_kwh_year` in `archetypes.py`.
- **Scheduled charging**'s commute timing (`parked_to_driving`,
  `driving_to_parked`, `driving_to_plugged_in`) is reused from Average UK -
  only its morning-departure curve (`plugged_idle_to_driving`) has its own
  table, shifted +2h so it centers on this archetype's own `plugout_time`
  (09:00) instead of Average UK's (07:00). See `SCHEDULED_CHARGING_WEEKDAY_TRANSITIONS`
  in `archetypes.py`.
- **One trip pattern per day**, split into two equal legs (out and back).
- **`infrequent_driving`'s day count is approximate.** `weekday_drive_probability=0.6`
  is a round choice ("~3 days/week"), not solved to make its trip size exactly
  match Average UK's (it lands at 86% of Average UK's per-trip kWh, close but
  not identical - see below).

## Incidents worth knowing about

**`market_index.csv` can have partial-day gaps.** `elexon_client.update_day_ahead_prices`
fetches whole calendar dates; if it happens to run mid-day (e.g. genuinely
"today" is that date), Elexon may only have published part of that day's
settlement periods so far, and the gap never gets backfilled since later
calls only look forward from the last date, not back to patch holes. This
actually happened for 14 July 2026 (only the first 29 of 48 periods were
ever fetched) and, before the fix above, silently priced that evening's
charging at £0 - which then wrecked the Savings table's baseline comparison
for every other archetype that day. If a Savings number looks implausible,
check whether the date in question has all 48 rows in `market_index.csv`
before assuming it's a simulation bug.

**`infrequent_charging` originally modeled "doesn't drive to work" instead
of "doesn't charge often."** The spreadsheet's own descriptive note said
"someone who doesn't drive to work" - but its numbers (`miles_per_year`
identical to Average UK, `kwh_per_plugin=37` vs Average UK's 7,
`plugin_soc=0.18` vs Average UK's 0.68) only make sense for someone who
drives *as much* as Average UK but charges far less often, letting soc
drift down over several days before a big top-up. The sheet's own notes
were an early draft and the numbers are the better signal here. Fixed by
switching `infrequent_charging` back to Average UK's transition tables and
`weekday_weekend_ratio`, and making `plugin_frequency_per_day` (already an
existing field, previously only used for `initial_state`'s starting soc)
double as the per-plug-in probability of actually starting a charge, in
`advance()`'s `DRIVING -> PLUGGED_CHARGING` branch. First pass used a pure
Bernoulli gate and it went badly: 12.6% of all slots landed at soc==0.0
(some charge-free streaks ran 15 days, well past what a 60kWh battery can
absorb at ~8kWh/day). Fixed by adding `or soc <= archetype.plugin_soc` to
the gate, forcing a charge once they'd hit the deficit level the sheet's
own numbers imply, capping the worst-case gap at ~7 days instead of an
unbounded geometric tail - soc never truncates to 0.0 anymore, and the
annualized mileage lands at 89%, matching every other archetype's known
long-trip gap. Same change also fixed a latent bug affecting every
archetype's driving-soc-drop: `soc - trip_soc_drop` wasn't clamped at 0,
just never got exercised before because no archetype went that low.

**`infrequent_driving` drove every weekday, just with smaller trips.** It
reused Average UK's transition tables and formula unchanged - only
`miles_per_year` was lower - so the *frequency* of driving matched Average UK
(363 days/year) and the annual mileage difference was absorbed entirely into
smaller trips (4.6 kWh/day vs Average UK's 8.0 kWh/day). That's a reasonable
model but not what "infrequent" implies. Fixed by adding
`weekday_drive_probability` to `ArchetypeConfig` (default 1.0, so every other
archetype is unaffected): each weekday, `advance()` now samples once whether
that day includes a commute at all (`RunState.drive_today`/`drive_day_date`),
and `weekend_kwh_per_day`'s formula concentrates the annual weekday budget
into only the days actually driven, so each trip that does happen is close to
full-size (6.9 kWh vs Average UK's 8.0) rather than diluted across every day.
`infrequent_driving` now drives ~156 weekdays/year instead of 260.

**`scheduled_charging` was dropping SoC while still shown plugged in.**
It reused Average UK's `plugged_idle_to_driving` curve unchanged, which
fires the morning departure at 06:00-08:30 - but `scheduled_charging`'s own
`plugout_time` is 09:00. The car was leaving (and SoC dropping) up to 2.5
hours before its own configured plug-out time. Fixed by giving it its own
`SCHEDULED_CHARGING_WEEKDAY_TRANSITIONS`, the same curve shifted +2h so the
departure centers on 09:00. Its other three curves (leaving/arriving work)
are unaffected since they're about the commute, not home charging.

**The "Plug-in behaviour" chart shows one real run, not an aggregate.**
It used to plot `median_soc` (a rank-order statistic across 200 runs)
against a majority-vote "plugged in" flag - two independent reductions
that can genuinely disagree near a 50/50 split (e.g. median SoC already
reflecting the "departed" group while a majority of runs are still idle),
which looked like bugs but wasn't fixable by aggregating differently:
state is categorical, so there's no meaningful "median state" to fall
back on. `sample_run_trajectory` replaced `median_trajectory`: the chart
now plots one real simulated run (`{name}_0`), so SoC, the "Plugged in"
shading, and the hover's state label always agree, by construction. The
"Population on this day" chart still covers the aggregate/percentile view
across all runs.

SoC renders as a plain diagonal line between slots (matches how the
chart looked before this investigation started); the "Plugged in" fill
uses `line_shape="hv"` (a step), which is correct for boolean data.
`hovermode="x"` on this chart makes hover snap to the actual half-hourly
sample instead of interpolating a fake in-between value off the drawn
line's pixel position, and `hoverinfo="skip"` on the "Plugged in" trace
stops it popping its own redundant tooltip. The diagonal is a rendering
choice, not a claim about what happens *between* samples - the
simulation only ever computes SoC at 30-minute slot boundaries; a trip's
whole energy cost is subtracted in the single slot its departure fires
in, not drained gradually across however long the drive takes (see "One
trip pattern per day" above).
