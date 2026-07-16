from dataclasses import dataclass
from datetime import time
from enum import Enum, auto

WEEKDAYS_PER_YEAR = 260
WEEKEND_DAYS_PER_YEAR = 105


class State(Enum):
    DRIVING = auto()
    PARKED = auto()
    PLUGGED_CHARGING = auto()
    PLUGGED_IDLE = auto()


class ChargingStrategy(Enum):
    IMMEDIATE = auto()
    SCHEDULED_PRICE = auto()
    FIXED_TIME = auto()


@dataclass(frozen=True)
class WeekdayTransitions:
    plugged_idle_to_driving: dict[time, float]
    parked_to_driving: dict[time, float]
    driving_to_parked: dict[time, float]
    driving_to_plugged_in: dict[time, float]


@dataclass(frozen=True)
class GaussianDeparture:
    mean: time
    std_minutes: float


@dataclass(frozen=True)
class FlatWindow:
    probability: float
    start: time
    end: time


@dataclass(frozen=True)
class WeekendTransitions:
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
    weekday_drive_probability: float = 1.0

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

    @property
    def long_trip_kwh(self) -> float:
        return self.long_trip_miles / self.efficiency_mi_per_kwh

    @property
    def long_trip_kwh_year(self) -> float:
        return self.long_trip_kwh * self.long_trip_days_per_year

    @property
    def remaining_kwh_year(self) -> float:
        return round(self.kwh_per_year) - self.long_trip_kwh_year

    @property
    def weekend_kwh_per_day(self) -> float:
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
    plugged_idle_to_driving=AVERAGE_UK_WEEKDAY_TRANSITIONS.plugged_idle_to_driving,
    parked_to_driving={},
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
            long_trip_days_per_year=10,
            long_trip_miles=150,
            weekday_weekend_ratio=1.0,
            weekday_drive_probability=0.5,
            weekday_transitions=AVERAGE_UK_WEEKDAY_TRANSITIONS,
            weekend_transitions=AVERAGE_UK_WEEKEND_TRANSITIONS,
        )

    @staticmethod
    def scheduled_charging() -> ArchetypeConfig:
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