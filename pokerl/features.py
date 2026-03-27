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

import numpy as np
from typing import Dict, List, Optional

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


def embed_battle(battle: Battle) -> np.ndarray:
    """Convert a battle state into a comprehensive feature vector.

    Feature breakdown:
      - Active Pokemon: 69
      - Active Pokemon moves (4 moves): 4 * 37 = 148
      - Bench Pokemon (6 slots): 6 * 55 = 330
      - Opponent active Pokemon: 69
      - Opponent known team (6 slots): 6 * 55 = 330
      - Weather: 9
      - Fields: 15
      - Our side conditions: 24
      - Opponent side conditions: 24
      - Battle flags: 14
      Total: 1032

    Returns:
        np.ndarray of shape (1032,)
    """
    buf = np.zeros(BATTLE_OBS_SIZE, dtype=np.float32)
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

    # Full team bench (6 * 55 = 330)
    _encode_team_pokemon_into(buf, o, battle.team)
    o += 330

    # Opponent active (69)
    _encode_pokemon_base_into(buf, o, battle.opponent_active_pokemon)
    o += 69

    # Opponent team (6 * 55 = 330)
    _encode_team_pokemon_into(buf, o, battle.opponent_team)
    o += 330

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

    return buf


def embed_team_preview(battle: Battle) -> np.ndarray:
    """Encode battle state during team preview for lead selection.

    Features:
      - Our team (6 Pokemon): 6 * 55 = 330
      - Opponent team (6 Pokemon): 6 * 55 = 330
      - Format flags: 4 (gen number one-hot isn't needed, just num_gimmicks/4)
      Total: 664

    Returns:
        np.ndarray of shape (664,)
    """
    buf = np.zeros(TEAM_PREVIEW_OBS_SIZE, dtype=np.float32)
    _encode_team_pokemon_into(buf, 0, battle.team)
    _encode_team_pokemon_into(buf, 330, battle.opponent_team)
    buf[660] = battle.gen / 9.0
    return buf


# Precompute sizes
BATTLE_OBS_SIZE = 1032
TEAM_PREVIEW_OBS_SIZE = 664  # 330 our team + 330 opp team + 1 gen + 3 reserved

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
