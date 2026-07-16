from dataclasses import dataclass
from datetime import time
from enum import Enum, auto

WEEKDAYS_PER_YEAR = 260
WEEKEND_DAYS_PER_YEAR = 100


class State(Enum):
    DRIVING = auto()
    PARKED = auto()
    PLUGGED_CHARGING = auto()
    PLUGGED_IDLE = auto()


class ChargingStrategy(Enum):
    IMMEDIATE = auto()       # ramp every slot from arrival (Average UK-style)
    SCHEDULED_PRICE = auto()  # cheapest EPEX slots against actual arrival time/SoC (Intelligent Octopus)
    FIXED_TIME = auto()      # arrive, then wait until archetype.plugin_time to start ramping (Scheduled charging)


@dataclass(frozen=True)
class WeekdayTransitions:
    """Half-hourly P(from->to) keyed by clock time. Complement (1 - p) is P(stay)."""

    plugged_idle_to_driving: dict[time, float]
    parked_to_driving: dict[time, float]
    driving_to_parked: dict[time, float]
    driving_to_plugged_in: dict[time, float]


@dataclass(frozen=True)
class GaussianDeparture:
    """Departure time modeled as Normal(mean, std) rather than a literal per-slot table."""

    mean: time
    std_minutes: float


@dataclass(frozen=True)
class FlatWindow:
    """Constant probability, but only within [start, end) - 0 outside it."""

    probability: float
    start: time
    end: time


@dataclass(frozen=True)
class WeekendTransitions:
    """
    Same 4 states as weekday, looser timing (one errand trip, wider jitter).
    Considered and rejected: zero-trip lazy days (20% prob), multiple errands,
    semi-Markov dwell times for variable trip length. Flagged as later refinements.
    """

    plugged_idle_to_driving: GaussianDeparture
    driving_to_parked: FlatWindow
    parked_to_driving: GaussianDeparture
    driving_to_plugged_in: FlatWindow


@dataclass(frozen=True)
class ArchetypeConfig:
    name: str
    population_share: float
    miles_per_year: float
    battery_kwh: float
    efficiency_mi_per_kwh: float
    plugin_frequency_per_day: float
    charger_kw: float
    plugin_time: time
    plugout_time: time
    target_soc: float
    long_trip_days_per_year: float
    long_trip_miles: float
    weekday_weekend_ratio: float
    weekday_transitions: WeekdayTransitions
    weekend_transitions: WeekendTransitions
    charging_strategy: ChargingStrategy
    weekday_drive_probability: float = 1.0  # fraction of weekdays actually driven, see infrequent_driving()

    @property
    def kwh_per_year(self) -> float:
        return self.miles_per_year / self.efficiency_mi_per_kwh

    @property
    def plugins_per_year(self) -> float:
        return self.plugin_frequency_per_day * 365

    @property
    def kwh_per_plugin(self) -> float:
        return self.kwh_per_year / self.plugins_per_year

    @property
    def soc_requirement(self) -> float:
        return self.kwh_per_plugin / self.battery_kwh

    @property
    def plugin_soc(self) -> float:
        return self.target_soc - self.soc_requirement

    @property
    def charging_duration_hrs(self) -> float:
        return self.kwh_per_plugin / self.charger_kw

    # --- Weekday/weekend kWh split ---
    # Derived from raw inputs only (long_trip_days_per_year, long_trip_miles,
    # weekday_weekend_ratio), so this generalizes to every archetype for free —
    # no per-archetype override of the formula itself, just its inputs.

    @property
    def long_trip_kwh(self) -> float:
        return self.long_trip_miles / self.efficiency_mi_per_kwh

    @property
    def long_trip_kwh_year(self) -> float:
        # Subtracted from the annual total below, never simulated as a
        # discrete event - long trips are folded into the weekend daily
        # average, not modeled as an occasional large SoC drop.
        return self.long_trip_kwh * self.long_trip_days_per_year

    @property
    def remaining_kwh_year(self) -> float:
        # round() matches the spreadsheet's cached kWh/year cell (2696, not the
        # raw 2695.71...), which is what the long-trip split was built against.
        return round(self.kwh_per_year) - self.long_trip_kwh_year

    @property
    def weekend_kwh_per_day(self) -> float:
        # weekday_drive_probability < 1 concentrates the weekday budget into
        # fewer actually-driven days, so weekday_kwh_per_day (below) comes out
        # as a full-size trip instead of a diluted one - see infrequent_driving().
        weighted_days = (
            WEEKDAYS_PER_YEAR * self.weekday_weekend_ratio * self.weekday_drive_probability
            + WEEKEND_DAYS_PER_YEAR
        )
        return self.remaining_kwh_year / weighted_days

    @property
    def weekday_kwh_per_day(self) -> float:
        return self.weekday_weekend_ratio * self.weekend_kwh_per_day


AVERAGE_UK_WEEKDAY_TRANSITIONS = WeekdayTransitions(
    plugged_idle_to_driving={
        time(6, 0): 0.02,
        time(6, 30): 0.14,
        time(7, 0): 0.4,
        time(7, 30): 0.68,
        time(8, 0): 0.88,
        time(8, 30): 1.0,
    },
    parked_to_driving={
        time(16, 30): 0.07,
        time(17, 0): 0.18,
        time(17, 30): 0.33,
        time(18, 0): 0.5,
        time(18, 30): 0.63,
        time(19, 0): 0.75,
        time(19, 30): 1.0,
    },
    driving_to_parked={
        time(6, 0): 0.85,
        time(6, 30): 0.85,
        time(7, 0): 0.85,
        time(7, 30): 0.85,
        time(8, 0): 0.85,
        time(8, 30): 0.85,
    },
    driving_to_plugged_in={
        time(17, 0): 0.85,
        time(17, 30): 0.85,
        time(18, 0): 0.85,
        time(18, 30): 0.85,
        time(19, 0): 0.85,
        time(19, 30): 0.85,
        time(20, 0): 0.85,
        time(20, 30): 0.85,
        time(21, 0): 0.85,
    },
)


ALWAYS_PLUGGED_IN_WEEKDAY_TRANSITIONS = WeekdayTransitions(
    # Charges wherever they stop, not just at home: driving_to_parked is
    # disabled (empty) and its window folded into driving_to_plugged_in, so
    # every arrival - morning at work, evening at home - starts charging.
    plugged_idle_to_driving=AVERAGE_UK_WEEKDAY_TRANSITIONS.plugged_idle_to_driving,
    parked_to_driving=AVERAGE_UK_WEEKDAY_TRANSITIONS.parked_to_driving,
    driving_to_parked={},
    driving_to_plugged_in={
        **AVERAGE_UK_WEEKDAY_TRANSITIONS.driving_to_parked,
        **AVERAGE_UK_WEEKDAY_TRANSITIONS.driving_to_plugged_in,
    },
)


ALWAYS_PLUGGED_IN_WEEKEND_TRANSITIONS = WeekendTransitions(
    plugged_idle_to_driving=GaussianDeparture(mean=time(10, 0), std_minutes=90),
    driving_to_parked=FlatWindow(probability=0.0, start=time(0, 0), end=time(0, 0)),
    parked_to_driving=GaussianDeparture(mean=time(14, 0), std_minutes=60),
    driving_to_plugged_in=FlatWindow(probability=0.85, start=time(8, 0), end=time(17, 0)),
)


AVERAGE_UK_WEEKEND_TRANSITIONS = WeekendTransitions(
    plugged_idle_to_driving=GaussianDeparture(mean=time(10, 0), std_minutes=90),
    driving_to_parked=FlatWindow(probability=0.85, start=time(8, 0), end=time(13, 0)),
    parked_to_driving=GaussianDeparture(mean=time(14, 0), std_minutes=60),
    driving_to_plugged_in=FlatWindow(probability=0.85, start=time(13, 0), end=time(17, 0)),
)


SCHEDULED_CHARGING_WEEKDAY_TRANSITIONS = WeekdayTransitions(
    # plugged_idle_to_driving shifted +2h from average_uk's (06:00-08:30 ->
    # 08:00-10:30) so the morning departure centers on this archetype's own
    # plugout_time (09:00), not average_uk's (07:00). parked_to_driving/
    # driving_to_parked/driving_to_plugged_in are about the work commute,
    # not home charging, so they're unaffected and reused as-is.
    plugged_idle_to_driving={
        time(8, 0): 0.02,
        time(8, 30): 0.14,
        time(9, 0): 0.4,
        time(9, 30): 0.68,
        time(10, 0): 0.88,
        time(10, 30): 1.0,
    },
    parked_to_driving=AVERAGE_UK_WEEKDAY_TRANSITIONS.parked_to_driving,
    driving_to_parked=AVERAGE_UK_WEEKDAY_TRANSITIONS.driving_to_parked,
    driving_to_plugged_in=AVERAGE_UK_WEEKDAY_TRANSITIONS.driving_to_plugged_in,
)


class ArchetypeFactory:
    @staticmethod
    def average_uk() -> ArchetypeConfig:
        return ArchetypeConfig(
            name="Average (UK)",
            charging_strategy=ChargingStrategy.IMMEDIATE,
            population_share=0.40,
            miles_per_year=9435,
            battery_kwh=60,
            efficiency_mi_per_kwh=3.5,
            plugin_frequency_per_day=1.0,
            charger_kw=7.0,
            plugin_time=time(18, 0),
            plugout_time=time(7, 0),
            target_soc=0.8,
            long_trip_days_per_year=5,
            long_trip_miles=150,
            weekday_weekend_ratio=2.0,
            weekday_transitions=AVERAGE_UK_WEEKDAY_TRANSITIONS,
            weekend_transitions=AVERAGE_UK_WEEKEND_TRANSITIONS,
        )

    @staticmethod
    def infrequent_charging() -> ArchetypeConfig:
        # Drives the same pattern as average_uk (same miles_per_year, same
        # transition tables) - "infrequent" describes charging, not driving:
        # plugin_frequency_per_day=0.2 doubles as the per-plug-in probability
        # of actually starting a charge (see the DRIVING->PLUGGED_CHARGING
        # branch in advance()), so soc drifts down over several days of
        # driving before a big top-up. Matches the sheet's own kwh_per_plugin
        # (37, vs average_uk's 7) and plugin_soc (0.18, a ~0.6 soc deficit)
        # far better than a no-weekday-driving reading did.
        return ArchetypeConfig(
            name="Infrequent charging",
            charging_strategy=ChargingStrategy.IMMEDIATE,
            population_share=0.10,
            miles_per_year=9435,
            battery_kwh=60,
            efficiency_mi_per_kwh=3.5,
            plugin_frequency_per_day=0.2,
            charger_kw=7.0,
            plugin_time=time(18, 0),
            plugout_time=time(7, 0),
            target_soc=0.8,
            long_trip_days_per_year=5,
            long_trip_miles=150,
            weekday_weekend_ratio=2.0,
            weekday_transitions=AVERAGE_UK_WEEKDAY_TRANSITIONS,
            weekend_transitions=AVERAGE_UK_WEEKEND_TRANSITIONS,
        )

    @staticmethod
    def infrequent_driving() -> ArchetypeConfig:
        # Same plug_in/plug_out (18:00/07:00) and transition tables as
        # average_uk - "infrequent" means fewer driving days, not smaller
        # trips, so weekday_drive_probability=0.6 (~3 days/week) skips the
        # commute on the other weekdays entirely, while weekday_kwh_per_day
        # comes out close to a full-size average_uk trip (not a diluted one)
        # because the annual budget is concentrated into fewer driven days.
        return ArchetypeConfig(
            name="Infrequent driving",
            charging_strategy=ChargingStrategy.IMMEDIATE,
            population_share=0.10,
            miles_per_year=5700,
            battery_kwh=60,
            efficiency_mi_per_kwh=3.5,
            plugin_frequency_per_day=1.0,
            charger_kw=7.0,
            plugin_time=time(18, 0),
            plugout_time=time(7, 0),
            target_soc=0.8,
            long_trip_days_per_year=5,
            long_trip_miles=150,
            weekday_weekend_ratio=2.0,
            weekday_drive_probability=0.6,
            weekday_transitions=AVERAGE_UK_WEEKDAY_TRANSITIONS,
            weekend_transitions=AVERAGE_UK_WEEKEND_TRANSITIONS,
        )

    @staticmethod
    def scheduled_charging() -> ArchetypeConfig:
        # Own weekday transitions (SCHEDULED_CHARGING_WEEKDAY_TRANSITIONS) so
        # the morning departure matches this archetype's plugout_time (09:00),
        # not average_uk's (07:00). Charging itself still starts at
        # plugin_time via FIXED_TIME (see ChargingStrategy).
        return ArchetypeConfig(
            name="Scheduled charging",
            charging_strategy=ChargingStrategy.FIXED_TIME,
            population_share=0.09,
            miles_per_year=9435,
            battery_kwh=60,
            efficiency_mi_per_kwh=3.5,
            plugin_frequency_per_day=1.0,
            charger_kw=7.0,
            plugin_time=time(22, 0),
            plugout_time=time(9, 0),
            target_soc=0.8,
            long_trip_days_per_year=5,
            long_trip_miles=150,
            weekday_weekend_ratio=2.0,
            weekday_transitions=SCHEDULED_CHARGING_WEEKDAY_TRANSITIONS,
            weekend_transitions=AVERAGE_UK_WEEKEND_TRANSITIONS,
        )

    @staticmethod
    def always_plugged_in() -> ArchetypeConfig:
        # Sheet's own note: "how can this always be plugged in?" - taking
        # plugin_time=00:00/plugout_time=23:59 literally: charging follows
        # them everywhere, not just at home. driving_to_parked is disabled
        # (see ALWAYS_PLUGGED_IN_*_TRANSITIONS) so every arrival - morning
        # at work, evening at home - starts charging instead of just parking.
        return ArchetypeConfig(
            name="Always plugged-in",
            charging_strategy=ChargingStrategy.IMMEDIATE,
            population_share=0.01,
            miles_per_year=9435,
            battery_kwh=60,
            efficiency_mi_per_kwh=3.5,
            plugin_frequency_per_day=1.0,
            charger_kw=7.0,
            plugin_time=time(0, 0),
            plugout_time=time(23, 59),
            target_soc=0.8,
            long_trip_days_per_year=5,
            long_trip_miles=150,
            weekday_weekend_ratio=2.0,
            weekday_transitions=ALWAYS_PLUGGED_IN_WEEKDAY_TRANSITIONS,
            weekend_transitions=ALWAYS_PLUGGED_IN_WEEKEND_TRANSITIONS,
        )

    @staticmethod
    def intelligent_octopus() -> ArchetypeConfig:
        # long_trip_*, weekday_weekend_ratio and both transition tables are
        # reused from average_uk() - no IO-specific figures exist in the sheet.
        # Sheet's SoC requirement (0.28) / charging duration (2.5hrs) for IO
        # don't match kwh_per_plugin/battery_kwh or kwh_per_plugin/charger_kw
        # (0.30 / 3.14hrs) - they line up with the CNZ report's empirical
        # median charge duration instead. Using the shared formula here.
        return ArchetypeConfig(
            name="Intelligent Octopus average",
            charging_strategy=ChargingStrategy.SCHEDULED_PRICE,
            population_share=0.30,
            miles_per_year=28105,
            battery_kwh=72.5,
            efficiency_mi_per_kwh=3.5,
            plugin_frequency_per_day=1.0,
            charger_kw=7.0,
            plugin_time=time(18, 0),
            plugout_time=time(7, 0),
            target_soc=0.8,
            long_trip_days_per_year=5,
            long_trip_miles=150,
            weekday_weekend_ratio=2.0,
            weekday_transitions=AVERAGE_UK_WEEKDAY_TRANSITIONS,
            weekend_transitions=AVERAGE_UK_WEEKEND_TRANSITIONS,
        )