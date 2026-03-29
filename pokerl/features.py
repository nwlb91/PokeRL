"""Comprehensive battle state feature extraction for RL agents.

Encodes ALL observable game state into a flat numpy array suitable for
neural network input. Covers:
  - Active Pokemon stats, types, status, boosts, volatile effects
  - Active Pokemon moves (power, type, category, accuracy, PP, effects)
  - Bench Pokemon (HP, status, types, key stats)
  - Opponent active and known bench Pokemon
  - Field state: weather, terrain, side conditions, hazards
  - Battle mechanics: can mega/z/dynamax/tera, force switch, turn count
  - Team preview state
"""

import logging

import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.battle import Battle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.move_category import MoveCategory

NUM_TYPES = 20  # PokemonType enum values 1-20
NUM_STATUSES = 7  # BRN, FNT, FRZ, PAR, PSN, SLP, TOX
NUM_WEATHERS = 9
NUM_FIELDS = 15
NUM_SIDE_CONDITIONS = 24
STAT_NAMES = ["hp", "atk", "def", "spa", "spd", "spe"]
BOOST_STATS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

# Precompute enum-to-index mappings at module load (avoid repeated list scans)
_STATUS_INDEX = {s: i for i, s in enumerate(Status)}
_WEATHER_INDEX = {w: i for i, w in enumerate(Weather)}
_FIELD_INDEX = {f: i for i, f in enumerate(Field)}
_SIDE_COND_INDEX = {sc: i for i, sc in enumerate(SideCondition)}
_CATEGORY_INDEX = {
    MoveCategory.PHYSICAL: 0,
    MoveCategory.SPECIAL: 1,
    MoveCategory.STATUS: 2,
}

def _encode_type_into(buf: np.ndarray, offset: int, ptype: Optional[PokemonType]):
    """Write one-hot PokemonType into buf at offset (20 dims). No allocation."""
    if ptype is not None:
        buf[offset + ptype.value - 1] = 1.0


def _encode_status_into(buf: np.ndarray, offset: int, status: Optional[Status]):
    """Write one-hot Status into buf at offset (7 dims). No allocation."""
    if status is not None:
        buf[offset + _STATUS_INDEX[status]] = 1.0


def _encode_pokemon_base_into(buf: np.ndarray, offset: int, mon: Optional[Pokemon]):
    """Encode core Pokemon features (69 dims) directly into buf at offset."""
    if mon is None:
        return

    o = offset
    buf[o] = mon.current_hp_fraction
    o += 1
    _encode_type_into(buf, o, mon.type_1)
    o += NUM_TYPES
    _encode_type_into(buf, o, mon.type_2)
    o += NUM_TYPES
    _encode_status_into(buf, o, mon.status)
    o += NUM_STATUSES
    boosts = mon.boosts
    for s in BOOST_STATS:
        buf[o] = boosts.get(s, 0) / 6.0
        o += 1
    base = mon.base_stats
    for s in STAT_NAMES:
        buf[o] = base.get(s, 0) / 255.0
        o += 1
    buf[o] = mon.level / 100.0; o += 1
    buf[o] = float(mon.is_dynamaxed); o += 1
    buf[o] = float(mon.is_terastallized); o += 1
    buf[o] = float(mon.fainted); o += 1
    buf[o] = float(mon.active); o += 1
    buf[o] = float(mon.must_recharge); o += 1
    buf[o] = float(mon.preparing is not None); o += 1
    buf[o] = min(mon.protect_counter, 6) / 6.0


def _encode_move_into(buf: np.ndarray, offset: int,
                      move: Optional[Move], pokemon: Optional[Pokemon] = None):
    """Encode a single move (37 dims) directly into buf at offset."""
    if move is None:
        return

    o = offset
    bp = move.base_power if isinstance(move.base_power, (int, float)) else 0
    buf[o] = bp / 250.0; o += 1

    _encode_type_into(buf, o, move.type)
    o += NUM_TYPES

    cat_idx = _CATEGORY_INDEX.get(move.category)
    if cat_idx is not None:
        buf[o + cat_idx] = 1.0
    o += 3

    buf[o] = move.accuracy if move.accuracy else 1.0; o += 1
    buf[o] = move.priority / 7.0; o += 1
    buf[o] = (move.current_pp / move.max_pp) if move.max_pp > 0 else 0.0; o += 1
    buf[o] = float(move.category == MoveCategory.STATUS); o += 1

    has_stab = False
    if pokemon is not None and move.type is not None:
        has_stab = move.type in pokemon.types
    buf[o] = float(has_stab); o += 1

    buf[o] = move.drain; o += 1
    buf[o] = move.recoil; o += 1
    buf[o] = move.heal; o += 1
    buf[o] = move.expected_hits / 5.0; o += 1
    buf[o] = float(move.force_switch); o += 1
    buf[o] = float(move.is_z); o += 1
    buf[o] = move.crit_ratio / 6.0; o += 1

    buf[o] = float(move.self_boost is not None and any(v != 0 for v in (move.self_boost or {}).values()))
    o += 1
    buf[o] = float(move.boosts is not None and any(v != 0 for v in (move.boosts or {}).values()))


def _encode_team_pokemon_into(buf: np.ndarray, offset: int,
                               team: Dict[str, Pokemon], max_size: int = 6):
    """Encode up to max_size Pokemon (55 dims each) into buf at offset."""
    per_mon = 55
    for i, mon in enumerate(team.values()):
        if i >= max_size:
            break
        o = offset + i * per_mon
        buf[o] = mon.current_hp_fraction; o += 1
        _encode_type_into(buf, o, mon.type_1); o += NUM_TYPES
        _encode_type_into(buf, o, mon.type_2); o += NUM_TYPES
        _encode_status_into(buf, o, mon.status); o += NUM_STATUSES
        base = mon.base_stats
        for s in STAT_NAMES:
            buf[o] = base.get(s, 0) / 255.0
            o += 1
        buf[o] = float(mon.fainted); o += 1


TEAM_POKEMON_EXTENDED_SIZE = 203  # 55 base + 4 * 37 moves

def _encode_team_pokemon_extended_into(
    buf: np.ndarray, offset: int,
    team: Dict[str, Pokemon],
    team_moves: Optional[List[List]] = None,
    max_size: int = 6,
):
    """Encode up to max_size Pokemon with full movesets (203 dims each).

    Args:
        buf: Output buffer.
        offset: Write position in buf.
        team: Pokemon dict from battle.team or battle.opponent_team.
        team_moves: Pre-parsed list of Move objects per pokemon (from team sheet).
            Index i corresponds to team slot i. If None, uses pokemon.moves.
        max_size: Max pokemon to encode.
    """
    per_mon = TEAM_POKEMON_EXTENDED_SIZE
    for i, mon in enumerate(team.values()):
        if i >= max_size:
            break
        o = offset + i * per_mon

        # Base encoding (55 dims)
        buf[o] = mon.current_hp_fraction; o += 1
        _encode_type_into(buf, o, mon.type_1); o += NUM_TYPES
        _encode_type_into(buf, o, mon.type_2); o += NUM_TYPES
        _encode_status_into(buf, o, mon.status); o += NUM_STATUSES
        base = mon.base_stats
        for s in STAT_NAMES:
            buf[o] = base.get(s, 0) / 255.0
            o += 1
        buf[o] = float(mon.fainted); o += 1

        # Move encoding (4 * 37 = 148 dims)
        if team_moves is not None and i < len(team_moves):
            moves = team_moves[i]
        else:
            moves = list(mon.moves.values())
        for j in range(4):
            move = moves[j] if j < len(moves) else None
            _encode_move_into(buf, o, move, mon)
            o += 37


def embed_battle(
    battle: Battle,
    our_team_moves: Optional[List[List]] = None,
    opp_team_moves: Optional[List[List]] = None,
) -> np.ndarray:
    """Convert a battle state into a comprehensive feature vector.

    When team_moves are provided (team_sheet_obs mode), bench pokemon are
    encoded with their full movesets (203 dims each) instead of the base
    55 dims, expanding the observation from 1032 to 2808.

    Args:
        battle: The current battle state.
        our_team_moves: Pre-parsed Move objects for our team (from team sheet).
        opp_team_moves: Pre-parsed Move objects for opponent team (from team sheet).

    Returns:
        np.ndarray of shape (BATTLE_OBS_SIZE,) or (BATTLE_OBS_SIZE_EXTENDED,)
    """
    extended = our_team_moves is not None or opp_team_moves is not None
    obs_size = BATTLE_OBS_SIZE_EXTENDED if extended else BATTLE_OBS_SIZE
    team_size = TEAM_POKEMON_EXTENDED_SIZE * 6 if extended else 330

    buf = np.zeros(obs_size, dtype=np.float32)
    o = 0

    # Active Pokemon (69)
    active = battle.active_pokemon
    _encode_pokemon_base_into(buf, o, active)
    o += 69

    # Active Pokemon moves (4 * 37 = 148)
    if active is not None:
        moves = list(active.moves.values())
    else:
        moves = []
    for i in range(4):
        move = moves[i] if i < len(moves) else None
        _encode_move_into(buf, o, move, active)
        o += 37

    # Full team (6 slots) — extended or base encoding
    if extended:
        _encode_team_pokemon_extended_into(buf, o, battle.team, our_team_moves)
    else:
        _encode_team_pokemon_into(buf, o, battle.team)
    o += team_size

    # Opponent active (69)
    _encode_pokemon_base_into(buf, o, battle.opponent_active_pokemon)
    o += 69

    # Opponent team (6 slots) — extended or base encoding
    if extended:
        _encode_team_pokemon_extended_into(buf, o, battle.opponent_team, opp_team_moves)
    else:
        _encode_team_pokemon_into(buf, o, battle.opponent_team)
    o += team_size

    # Weather (9)
    for w in battle.weather:
        buf[o + _WEATHER_INDEX[w]] = 1.0
    o += NUM_WEATHERS

    # Fields (15)
    for f in battle.fields:
        buf[o + _FIELD_INDEX[f]] = 1.0
    o += NUM_FIELDS

    # Side conditions — ours (24)
    for sc, val in battle.side_conditions.items():
        idx = _SIDE_COND_INDEX[sc]
        if sc == SideCondition.SPIKES:
            buf[o + idx] = min(val, 3) / 3.0
        elif sc == SideCondition.TOXIC_SPIKES:
            buf[o + idx] = min(val, 2) / 2.0
        else:
            buf[o + idx] = 1.0
    o += NUM_SIDE_CONDITIONS

    # Side conditions — opponent (24)
    for sc, val in battle.opponent_side_conditions.items():
        idx = _SIDE_COND_INDEX[sc]
        if sc == SideCondition.SPIKES:
            buf[o + idx] = min(val, 3) / 3.0
        elif sc == SideCondition.TOXIC_SPIKES:
            buf[o + idx] = min(val, 2) / 2.0
        else:
            buf[o + idx] = 1.0
    o += NUM_SIDE_CONDITIONS

    # Battle flags (14)
    buf[o] = float(battle.can_mega_evolve)
    buf[o+1] = float(battle.can_z_move)
    buf[o+2] = float(battle.can_dynamax)
    buf[o+3] = float(battle.can_tera)
    buf[o+4] = float(battle.force_switch)
    buf[o+5] = float(battle.trapped)
    buf[o+6] = float(battle.maybe_trapped)
    buf[o+7] = battle.turn / 100.0
    buf[o+8] = float(battle.teampreview)
    buf[o+9] = float(battle.used_dynamax)
    buf[o+10] = float(battle.used_mega_evolve)
    buf[o+11] = float(battle.used_z_move)
    buf[o+12] = float(battle.used_tera)
    if battle.dynamax_turns_left is not None:
        buf[o+13] = battle.dynamax_turns_left / 3.0

    if not np.isfinite(buf).all():
        logger.warning("Non-finite values in battle observation (turn %d), replacing with zeros", battle.turn)
        np.nan_to_num(buf, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

    return buf


def embed_team_preview(
    battle: Battle,
    our_team_moves: Optional[List[List]] = None,
    opp_team_moves: Optional[List[List]] = None,
) -> np.ndarray:
    """Encode battle state during team preview for lead selection.

    Args:
        battle: The current battle state.
        our_team_moves: Pre-parsed Move objects for our team (from team sheet).
        opp_team_moves: Pre-parsed Move objects for opponent team (from team sheet).

    Returns:
        np.ndarray of shape (TEAM_PREVIEW_OBS_SIZE,) or
        (TEAM_PREVIEW_OBS_SIZE_EXTENDED,)
    """
    extended = our_team_moves is not None or opp_team_moves is not None
    obs_size = TEAM_PREVIEW_OBS_SIZE_EXTENDED if extended else TEAM_PREVIEW_OBS_SIZE
    team_block = TEAM_POKEMON_EXTENDED_SIZE * 6 if extended else 330

    buf = np.zeros(obs_size, dtype=np.float32)

    if extended:
        _encode_team_pokemon_extended_into(buf, 0, battle.team, our_team_moves)
        _encode_team_pokemon_extended_into(buf, team_block, battle.opponent_team, opp_team_moves)
    else:
        _encode_team_pokemon_into(buf, 0, battle.team)
        _encode_team_pokemon_into(buf, team_block, battle.opponent_team)

    buf[2 * team_block] = battle.gen / 9.0

    if not np.isfinite(buf).all():
        logger.warning("Non-finite values in team preview observation, replacing with zeros")
        np.nan_to_num(buf, copy=False, nan=0.0, posinf=1.0, neginf=0.0)

    return buf


# Precompute sizes
BATTLE_OBS_SIZE = 1032
BATTLE_OBS_SIZE_EXTENDED = 69 + 148 + (203 * 6) + 69 + (203 * 6) + 9 + 15 + 24 + 24 + 14  # 2808
TEAM_PREVIEW_OBS_SIZE = 664  # 330 our team + 330 opp team + 1 gen + 3 reserved
TEAM_PREVIEW_OBS_SIZE_EXTENDED = (203 * 6) + (203 * 6) + 4  # 2440

# Runtime checks: ensure hardcoded constants cover all enum values.
# If poke-env adds new types/statuses/etc., these will catch the mismatch.
assert NUM_TYPES >= max(t.value for t in PokemonType if t.value > 0), (
    f"NUM_TYPES ({NUM_TYPES}) is smaller than the largest PokemonType value"
)
assert NUM_STATUSES >= len(Status), (
    f"NUM_STATUSES ({NUM_STATUSES}) < len(Status) ({len(Status)})"
)
assert NUM_WEATHERS >= len(Weather), (
    f"NUM_WEATHERS ({NUM_WEATHERS}) < len(Weather) ({len(Weather)})"
)
assert NUM_FIELDS >= len(Field), (
    f"NUM_FIELDS ({NUM_FIELDS}) < len(Field) ({len(Field)})"
)
assert NUM_SIDE_CONDITIONS >= len(SideCondition), (
    f"NUM_SIDE_CONDITIONS ({NUM_SIDE_CONDITIONS}) < len(SideCondition) ({len(SideCondition)})"
)
