"""Tests for team sheet parsing and extended observations."""

import numpy as np
import pytest

from pokerl.config import Config
from pokerl.features import (
    BATTLE_OBS_SIZE, BATTLE_OBS_SIZE_EXTENDED,
    TEAM_PREVIEW_OBS_SIZE, TEAM_PREVIEW_OBS_SIZE_EXTENDED,
    TEAM_POKEMON_EXTENDED_SIZE,
)
from pokerl.teamsheet import ParsedPokemon, ParsedTeam, parse_team, _to_move_id


SAMPLE_TEAM = """
Dragonite @ Lum Berry
Ability: Multiscale
EVs: 252 Atk / 4 SpD / 252 Spe
Adamant Nature
- Dragon Dance
- Outrage
- Extreme Speed
- Earthquake

Garchomp @ Rocky Helmet
Ability: Rough Skin
EVs: 252 HP / 4 Atk / 252 Def
Impish Nature
- Stealth Rock
- Earthquake
- Dragon Tail
- Toxic
"""


class TestMoveIdConversion:
    def test_simple_move(self):
        assert _to_move_id("Earthquake") == "earthquake"

    def test_multi_word(self):
        assert _to_move_id("Dragon Dance") == "dragondance"

    def test_hyphenated(self):
        assert _to_move_id("U-turn") == "uturn"

    def test_hyphenated_multi(self):
        assert _to_move_id("Freeze-Dry") == "freezedry"


class TestParseTeam:
    def test_parse_sample_team(self):
        team = parse_team(SAMPLE_TEAM)
        assert len(team.pokemon) == 2
        assert team.pokemon[0].species == "dragonite"
        assert team.pokemon[0].moves == ["dragondance", "outrage", "extremespeed", "earthquake"]
        assert team.pokemon[1].species == "garchomp"
        assert team.pokemon[1].moves == ["stealthrock", "earthquake", "dragontail", "toxic"]

    def test_get_move_objects(self):
        team = parse_team(SAMPLE_TEAM)
        move_objs = team.get_move_objects(gen=9)
        assert len(move_objs) == 2
        assert len(move_objs[0]) == 4
        assert move_objs[0][0].id == "dragondance"
        assert move_objs[1][1].id == "earthquake"

    def test_empty_team(self):
        team = parse_team("")
        assert len(team.pokemon) == 0


class TestExtendedObsSize:
    def test_extended_battle_obs_is_larger(self):
        assert BATTLE_OBS_SIZE_EXTENDED > BATTLE_OBS_SIZE

    def test_extended_team_pokemon_size(self):
        # 55 base + 4 * 37 moves = 203
        assert TEAM_POKEMON_EXTENDED_SIZE == 203

    def test_extended_battle_obs_formula(self):
        # 69 active + 148 active moves + 6*203 our team + 69 opp active +
        # 6*203 opp team + 9 weather + 15 fields + 24+24 side conds + 14 flags
        expected = 69 + 148 + (203 * 6) + 69 + (203 * 6) + 9 + 15 + 24 + 24 + 14
        assert BATTLE_OBS_SIZE_EXTENDED == expected

    def test_extended_preview_obs_is_larger(self):
        assert TEAM_PREVIEW_OBS_SIZE_EXTENDED > TEAM_PREVIEW_OBS_SIZE


class TestConfigTeamSheet:
    def test_team_sheet_obs_default_on(self):
        config = Config()
        assert config.team_sheet_obs

    def test_team_sheet_obs_enabled(self):
        config = Config(team_sheet_obs=True)
        assert config.team_sheet_obs
