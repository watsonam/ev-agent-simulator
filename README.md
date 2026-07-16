# EV Charging Behaviour Simulator

An agent-based simulator of how different types of EV driver charge over a day, with a Streamlit dashboard on top.

## Run it

Hosted: https://ev-agent-simulator-5m5uzaqkf2h7e6gcukxaup.streamlit.app/

Locally:
```
uv sync
uv run streamlit run streamlit_app.py
```

Four States:
- Driving
- Parked
- Plugged in Charging
- Plugged Idle

Charging Strategy:
- Immediate -> assumed for Average (UK), Infrequent Charging, Infrequent Driving, Always plugged-in
- Scheduled Price -> assumed for Intelligent Octopus, based off the cheapest Day-Ahead slots
- Fixed Time -> assumed for Scheduled charging archetype

Calculation assumptions:
- Assume a number of long trips per year for each archetype
- 260 weekdays per year, 105 weekends
- Calculate a per weekday and weekend kwh use, subtract the long-trip kwh first, then split what's left across the two daily trips (commute out, commute back)
- A trip's length is sampled up front - 85% chance it takes one 30 min slot, 15% chance two - and the kwh is split evenly across whatever got sampled, so SoC moves every slot spent driving, not just the last one

Two ways of encoding a transition. Weekdays use a per-slot probability table which is good for a commute schedule. Weekends use two simpler curve shapes: GaussianDeparture, where a departure time is sampled once from a normal distribution around a mean (e.g. leave around 10am, back around 2pm), and FlatWindow, a constant probability held within a start/end window and zero outside it. Weekends are more spread out and less regimented, so a sampled time and a flat window fit better than a hand-tuned per-slot ramp.

Weekday Transitions:
- Average UK -> typical commuter, leaving the house sometime between 6am and 8:30am using a conditional probability, so if they haven't left by 8am there's a 100% probability they've left by 8:30am. Leaving work sometime between 16:30 and 19:30, returning home to plug in. This is encoded with a Markov chain transition matrix
- Intelligent Octopus follows the same average uk weekday and weekend transitions but what changes is the charging schedule is based off the wholesale day-ahead prices. Based off the Soc requirement at plug-in, this works out to be between 5 and 8 30 minute slots so it chooses the cheapest overnight. Assumption is that wholesale = retail here.
- Infrequent Driving - same transitions as Average, but more long trip days a year and a lower weekday to weekend ratio (i.e. more driving on the weekend). Long trips are subtracted before the per-day kwh average is calculated, so they never show up as extra driving in the dashboard - see limitations
- Scheduled Charging, similar to the Average Uk transitions but centered around 9am to reflect the later plug out time.
- Infrequent Charging - Same transition times as Average UK but I've used plugin_frequency_per_day as a probability of plugging in, so SoC drifts down over several days of driving before a larger top up.
- Always plugged-in: charges wherever it stops, not just at home - so the transition to Parked is disabled and folded into Plugged in instead. Because it never visits Parked, the transition that normally triggers the evening trip home never fires either, so it only completes one trip a day, not two - see limitations (I basically ran out of time on this)

Weekend Transitions:
- Average UK -> a single trip out sometime around 10am (Gaussian, not a fixed window), a few hours out, then back sometime around 2pm. Most other archetypes reuse this
- Always plugged-in has the same one-trip issue on weekends as on weekdays, for the same reason.

The mechanics:
- Start at an initial state, based off the spreadsheet parameters i.e. usual plug in time and usual starting charge
- Run a loop from the start state based off the transition matrices
- For the plug-in behaviour graph I show one real run, the first
- For the recapitulated population, use 200 weighted runs, show the mean soc and percentile bands around it, overlayed with the % of the population plugged in. Weight is tied to each archetype's share of the population
- Cost savings per day are shown for each archetype vs the average (uk) cost, alongside the daily price curve

Limitations:
- Long trips are subtracted from the daily kwh budget before the routine driving average is calculated, so they never actually appear as extra driving in the plug-in behaviour dashboard - this needs a rethink
- Always plugged-in only completes one trip a day (not two), on both weekdays and weekends, because it never visits Parked - the state that normally triggers the second trip. Simulates ~53% of its target annual mileage. Again ran out of time on this
- Every day, weekday and weekend, is modelled as a single round trip out and back (two legs). That's a fair picture of a weekday commute, but weekend behaviour is really much more varied - multiple short errands, a single long day out, or maybe nothing at all. I chose one round trip anyway because the dashboard is about energy drawn and when the car plugs in, not the exact count of trips
- Prices come from Elexon, who publish tomorrow's day-ahead prices only during the afternoon before, and settle today's own prices a little behind real time