"""Tests for feature encoding constants and observation sizes."""

import numpy as np

from pokerl.features import (
    BATTLE_OBS_SIZE,
    TEAM_PREVIEW_OBS_SIZE,
    NUM_TYPES,
    NUM_STATUSES,
    NUM_WEATHERS,
    NUM_FIELDS,
    NUM_SIDE_CONDITIONS,
)
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition


class TestObservationSizes:
    def test_battle_obs_size_positive(self):
        assert BATTLE_OBS_SIZE > 0

    def test_battle_obs_size_value(self):
        assert BATTLE_OBS_SIZE == 1032

    def test_team_preview_obs_size_positive(self):
        assert TEAM_PREVIEW_OBS_SIZE > 0

    def test_team_preview_obs_size_value(self):
        assert TEAM_PREVIEW_OBS_SIZE == 664


class TestFeatureConstants:
    """Ensure hardcoded constants cover all poke-env enum values."""

    def test_num_types_covers_all(self):
        max_type_val = max(t.value for t in PokemonType if t.value > 0)
        assert NUM_TYPES >= max_type_val, (
            f"NUM_TYPES ({NUM_TYPES}) doesn't cover PokemonType max value ({max_type_val})"
        )

    def test_num_statuses_covers_all(self):
        assert NUM_STATUSES >= len(Status), (
            f"NUM_STATUSES ({NUM_STATUSES}) < len(Status) ({len(Status)})"
        )

    def test_num_weathers_covers_all(self):
        assert NUM_WEATHERS >= len(Weather), (
            f"NUM_WEATHERS ({NUM_WEATHERS}) < len(Weather) ({len(Weather)})"
        )

    def test_num_fields_covers_all(self):
        assert NUM_FIELDS >= len(Field), (
            f"NUM_FIELDS ({NUM_FIELDS}) < len(Field) ({len(Field)})"
        )

    def test_num_side_conditions_covers_all(self):
        assert NUM_SIDE_CONDITIONS >= len(SideCondition), (
            f"NUM_SIDE_CONDITIONS ({NUM_SIDE_CONDITIONS}) < len(SideCondition) ({len(SideCondition)})"
        )
