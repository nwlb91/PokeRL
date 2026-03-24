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


def _encode_type(ptype: Optional[PokemonType]) -> np.ndarray:
    """One-hot encode a PokemonType (20 dims)."""
    vec = np.zeros(NUM_TYPES, dtype=np.float32)
    if ptype is not None:
        vec[ptype.value - 1] = 1.0
    return vec


def _encode_status(status: Optional[Status]) -> np.ndarray:
    """One-hot encode a Status (7 dims)."""
    vec = np.zeros(NUM_STATUSES, dtype=np.float32)
    if status is not None:
        idx = list(Status).index(status)
        vec[idx] = 1.0
    return vec


def _encode_boosts(boosts: Dict[str, int]) -> np.ndarray:
    """Encode stat boosts normalized to [-1, 1] (7 dims)."""
    return np.array(
        [boosts.get(s, 0) / 6.0 for s in BOOST_STATS], dtype=np.float32
    )


def _encode_pokemon_base(mon: Optional[Pokemon], known: bool = True) -> np.ndarray:
    """Encode core Pokemon features (69 dims).

    - HP fraction: 1
    - Types (2 * 20): 40
    - Status: 7
    - Boosts: 7
    - Base stats normalized: 6
    - Level normalized: 1
    - Is dynamaxed: 1
    - Is terastallized: 1
    - Tera type: 20 (if known) -- but we encode it always, zeros if unknown
    - Fainted: 1
    - Active: 1
    - Must recharge: 1
    - Preparing move: 1
    - Protect counter (normalized): 1
    Total: 1 + 40 + 7 + 7 + 6 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 69
    (We skip tera_type encoding here to keep it simpler - 69 dims)
    """
    if mon is None:
        return np.zeros(69, dtype=np.float32)

    features = []

    # HP fraction
    features.append(mon.current_hp_fraction)

    # Types
    features.extend(_encode_type(mon.type_1))
    features.extend(_encode_type(mon.type_2))

    # Status
    features.extend(_encode_status(mon.status))

    # Boosts
    features.extend(_encode_boosts(mon.boosts))

    # Base stats (normalized by 255, max possible base stat)
    for stat in STAT_NAMES:
        features.append(mon.base_stats.get(stat, 0) / 255.0)

    # Level
    features.append(mon.level / 100.0)

    # Flags
    features.append(float(mon.is_dynamaxed))
    features.append(float(mon.is_terastallized))
    features.append(float(mon.fainted))
    features.append(float(mon.active))
    features.append(float(mon.must_recharge))
    features.append(float(mon.preparing))
    features.append(min(mon.protect_counter, 6) / 6.0)

    return np.array(features, dtype=np.float32)


def _encode_move(move: Optional[Move], pokemon: Optional[Pokemon] = None) -> np.ndarray:
    """Encode a single move (37 dims).

    - Base power / 250: 1
    - Type: 20
    - Category (3-dim one-hot): 3
    - Accuracy: 1
    - Priority / 7: 1
    - PP fraction: 1
    - Is status move: 1
    - Has STAB: 1
    - Drain fraction: 1
    - Recoil fraction: 1
    - Heal fraction: 1
    - Expected hits / 5: 1
    - Force switch: 1
    - Is Z move: 1
    - Crit ratio / 6: 1
    - Has self boost: 1
    - Has target boost: 1
    Total: 37
    """
    if move is None:
        return np.zeros(37, dtype=np.float32)

    features = []

    # Base power
    bp = move.base_power if isinstance(move.base_power, (int, float)) else 0
    features.append(bp / 250.0)

    # Type
    features.extend(_encode_type(move.type))

    # Category
    cat = np.zeros(3, dtype=np.float32)
    if move.category == MoveCategory.PHYSICAL:
        cat[0] = 1.0
    elif move.category == MoveCategory.SPECIAL:
        cat[1] = 1.0
    elif move.category == MoveCategory.STATUS:
        cat[2] = 1.0
    features.extend(cat)

    # Accuracy
    features.append(move.accuracy if move.accuracy else 1.0)

    # Priority
    features.append(move.priority / 7.0)

    # PP fraction
    if move.max_pp > 0:
        features.append(move.current_pp / move.max_pp)
    else:
        features.append(0.0)

    # Flags
    features.append(float(move.category == MoveCategory.STATUS))

    # STAB check
    has_stab = False
    if pokemon is not None and move.type is not None:
        has_stab = move.type in pokemon.types
    features.append(float(has_stab))

    # Drain / Recoil / Heal
    features.append(move.drain)
    features.append(move.recoil)
    features.append(move.heal)

    # Expected hits
    features.append(move.expected_hits / 5.0)

    # Force switch
    features.append(float(move.force_switch))

    # Is Z
    features.append(float(move.is_z))

    # Crit ratio
    features.append(move.crit_ratio / 6.0)

    # Boost info
    features.append(float(move.self_boost is not None and any(v != 0 for v in (move.self_boost or {}).values())))
    features.append(float(move.boosts is not None and any(v != 0 for v in (move.boosts or {}).values())))

    return np.array(features, dtype=np.float32)


def _encode_weather(weather: Dict[Weather, int]) -> np.ndarray:
    """One-hot weather encoding (9 dims)."""
    vec = np.zeros(NUM_WEATHERS, dtype=np.float32)
    for w in weather:
        idx = list(Weather).index(w)
        vec[idx] = 1.0
    return vec


def _encode_fields(fields: Dict[Field, int]) -> np.ndarray:
    """One-hot field encoding (15 dims)."""
    vec = np.zeros(NUM_FIELDS, dtype=np.float32)
    for f in fields:
        idx = list(Field).index(f)
        vec[idx] = 1.0
    return vec


def _encode_side_conditions(conditions: Dict[SideCondition, int]) -> np.ndarray:
    """Encode side conditions with layer counts where applicable (24 dims)."""
    vec = np.zeros(NUM_SIDE_CONDITIONS, dtype=np.float32)
    for sc, val in conditions.items():
        idx = list(SideCondition).index(sc)
        # Spikes: up to 3 layers, Toxic Spikes: up to 2 layers
        if sc == SideCondition.SPIKES:
            vec[idx] = min(val, 3) / 3.0
        elif sc == SideCondition.TOXIC_SPIKES:
            vec[idx] = min(val, 2) / 2.0
        else:
            vec[idx] = 1.0
    return vec


def _encode_team_pokemon(team: Dict[str, Pokemon], max_size: int = 6) -> np.ndarray:
    """Encode up to max_size Pokemon from a team (bench info).

    Per Pokemon: HP fraction (1) + types (40) + status (7) + base stats (6) +
                 fainted (1) = 55 dims
    Total: 55 * max_size = 330 dims
    """
    per_mon = 55
    result = np.zeros(per_mon * max_size, dtype=np.float32)

    for i, mon in enumerate(team.values()):
        if i >= max_size:
            break
        offset = i * per_mon
        result[offset] = mon.current_hp_fraction
        result[offset + 1: offset + 21] = _encode_type(mon.type_1)
        result[offset + 21: offset + 41] = _encode_type(mon.type_2)
        result[offset + 41: offset + 48] = _encode_status(mon.status)
        for j, stat in enumerate(STAT_NAMES):
            result[offset + 48 + j] = mon.base_stats.get(stat, 0) / 255.0
        result[offset + 54] = float(mon.fainted)

    return result


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
    features = []

    # --- Active Pokemon ---
    active = battle.active_pokemon
    features.append(_encode_pokemon_base(active))

    # --- Active Pokemon's 4 moves ---
    if active is not None:
        moves = list(active.moves.values())
    else:
        moves = []
    for i in range(4):
        move = moves[i] if i < len(moves) else None
        features.append(_encode_move(move, active))

    # --- Full team (bench info, all 6 slots) ---
    features.append(_encode_team_pokemon(battle.team))

    # --- Opponent active Pokemon ---
    opp_active = battle.opponent_active_pokemon
    features.append(_encode_pokemon_base(opp_active, known=False))

    # --- Opponent team (known info) ---
    features.append(_encode_team_pokemon(battle.opponent_team))

    # --- Weather ---
    features.append(_encode_weather(battle.weather))

    # --- Fields / Terrain ---
    features.append(_encode_fields(battle.fields))

    # --- Side conditions (ours) ---
    features.append(_encode_side_conditions(battle.side_conditions))

    # --- Side conditions (opponent) ---
    features.append(_encode_side_conditions(battle.opponent_side_conditions))

    # --- Battle flags (14 dims) ---
    flags = np.zeros(14, dtype=np.float32)
    flags[0] = float(battle.can_mega_evolve)
    flags[1] = float(battle.can_z_move)
    flags[2] = float(battle.can_dynamax)
    flags[3] = float(battle.can_tera)
    flags[4] = float(battle.force_switch)
    flags[5] = float(battle.trapped)
    flags[6] = float(battle.maybe_trapped)
    flags[7] = battle.turn / 100.0  # normalize turn count
    flags[8] = float(battle.teampreview)
    flags[9] = float(battle.used_dynamax)
    flags[10] = float(battle.used_mega_evolve)
    flags[11] = float(battle.used_z_move)
    flags[12] = float(battle.used_tera)
    if battle.dynamax_turns_left is not None:
        flags[13] = battle.dynamax_turns_left / 3.0
    features.append(flags)

    return np.concatenate(features)


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
    features = []

    # Our team
    features.append(_encode_team_pokemon(battle.team))

    # Opponent team (what we can see in preview)
    features.append(_encode_team_pokemon(battle.opponent_team))

    # Format info
    fmt = np.zeros(4, dtype=np.float32)
    fmt[0] = battle.gen / 9.0
    features.append(fmt)

    return np.concatenate(features)


# Precompute sizes
BATTLE_OBS_SIZE = 1032
TEAM_PREVIEW_OBS_SIZE = 664
