from archetypes import ArchetypeFactory

ALL_ARCHETYPES = [
    ArchetypeFactory.average_uk(),
    ArchetypeFactory.intelligent_octopus(),
    ArchetypeFactory.infrequent_charging(),
    ArchetypeFactory.infrequent_driving(),
    ArchetypeFactory.scheduled_charging(),
    ArchetypeFactory.always_plugged_in(),
]


def test_population_shares_sum_to_one():
    assert sum(a.population_share for a in ALL_ARCHETYPES) == 1.0


def test_plugin_soc_is_below_target_for_every_archetype():
    for archetype in ALL_ARCHETYPES:
        assert 0 < archetype.plugin_soc < archetype.target_soc
