"""Action space management: maps RL actions to poke-env battle orders.

Handles the full action space including:
  - 6 switches (indices 0-5)
  - 4 moves (indices 6-9)
  - 4 moves + mega evolve (indices 10-13) [gen6+]
  - 4 moves + z-move (indices 14-17) [gen7+]
  - 4 moves + dynamax (indices 18-21) [gen8+]
  - 4 moves + terastallize (indices 22-25) [gen9+]

Also handles team preview lead selection (indices 0-5).
"""

import logging

import numpy as np

from poke_env.battle.battle import Battle
from poke_env.player.battle_order import (
    BattleOrder,
    DefaultBattleOrder,
)
from poke_env.player.player import Player

from pokerl.config import Config

logger = logging.getLogger(__name__)


def get_action_mask(battle: Battle, config: Config) -> np.ndarray:
    """Compute a binary mask of legal actions for the current battle state.

    Returns:
        np.ndarray of shape (action_size,) with 1.0 for legal actions, 0.0 otherwise.
    """
    mask = np.zeros(config.action_size, dtype=np.float32)
    team_mons = list(battle.team.values())

    if battle.force_switch:
        # Can only switch; encode available switches
        for mon in battle.available_switches:
            try:
                idx = team_mons.index(mon)
                mask[idx] = 1.0
            except ValueError:
                pass
        # If no switches available (e.g., all fainted), allow default
        if mask.sum() == 0:
            mask[0] = 1.0
        return mask

    # --- Switches ---
    for mon in battle.available_switches:
        try:
            idx = team_mons.index(mon)
            mask[idx] = 1.0
        except ValueError:
            pass

    # --- Moves ---
    if battle.active_pokemon is not None:
        active_moves = list(battle.active_pokemon.moves.values())
        available_move_ids = {m.id for m in battle.available_moves}

        # Handle struggle/recharge
        if (len(battle.available_moves) == 1 and
                battle.available_moves[0].id in ("struggle", "recharge")):
            mask[6] = 1.0
            return mask

        for i, move in enumerate(active_moves):
            if i >= 4:
                break
            if move.id in available_move_ids:
                # Base move
                mask[6 + i] = 1.0

                # Mega evolution
                if config.num_gimmicks >= 1 and battle.can_mega_evolve:
                    mask[10 + i] = 1.0

                # Z-move
                if config.num_gimmicks >= 2 and battle.can_z_move:
                    # Z-moves: only available if the Pokemon has the right Z crystal
                    z_moves = battle.active_pokemon.available_z_moves
                    if z_moves:
                        mask[14 + i] = 1.0

                # Dynamax
                if config.num_gimmicks >= 3 and battle.can_dynamax:
                    mask[18 + i] = 1.0

                # Terastallize
                if config.num_gimmicks >= 4 and battle.can_tera:
                    mask[22 + i] = 1.0

    # If nothing is legal (shouldn't happen), default to first action
    if mask.sum() == 0:
        logger.warning(
            "Action mask is all zeros — no legal actions detected. "
            "Falling back to action 0. This may indicate a bug in "
            "battle state reading (turn %d).",
            battle.turn,
        )
        mask[0] = 1.0

    return mask


def get_team_preview_mask(battle: Battle) -> np.ndarray:
    """Compute a binary mask for team preview lead selection.

    Returns:
        np.ndarray of shape (6,) with 1.0 for each Pokemon available as lead.
    """
    mask = np.zeros(6, dtype=np.float32)
    team_mons = list(battle.team.values())
    for i, mon in enumerate(team_mons):
        if i < 6 and not mon.fainted:
            mask[i] = 1.0
    if mask.sum() == 0:
        mask[0] = 1.0
    return mask


def action_to_order(action: int, battle: Battle, config: Config) -> BattleOrder:
    """Convert an integer action to a poke-env BattleOrder.

    Args:
        action: Integer action index.
        battle: Current battle state.
        config: Training configuration.

    Returns:
        A SingleBattleOrder or DefaultBattleOrder.
    """
    team_mons = list(battle.team.values())

    # Switch actions (0-5)
    if action < 6:
        if action < len(team_mons):
            mon = team_mons[action]
            if mon in battle.available_switches:
                return Player.create_order(mon)
        # Fallback to a valid switch or default
        if battle.available_switches:
            return Player.create_order(battle.available_switches[0])
        return DefaultBattleOrder()

    # Move actions
    if battle.active_pokemon is None:
        return DefaultBattleOrder()

    # Handle struggle/recharge
    if (len(battle.available_moves) == 1 and
            battle.available_moves[0].id in ("struggle", "recharge")):
        return Player.create_order(battle.available_moves[0])

    move_idx = (action - 6) % 4
    gimmick_group = (action - 6) // 4

    active_moves = list(battle.active_pokemon.moves.values())
    if move_idx >= len(active_moves):
        # Fallback: use first available move
        if battle.available_moves:
            return Player.create_order(battle.available_moves[0])
        return DefaultBattleOrder()

    move = active_moves[move_idx]

    mega = (gimmick_group == 1)
    z_move = (gimmick_group == 2)
    dynamax = (gimmick_group == 3)
    terastallize = (gimmick_group == 4)

    return Player.create_order(
        move,
        mega=mega,
        z_move=z_move,
        dynamax=dynamax,
        terastallize=terastallize,
    )


def team_preview_to_order(action: int, battle: Battle) -> str:
    """Convert a team preview action (lead index) to a /team command.

    The chosen Pokemon is placed first; the rest follow in original order.

    Args:
        action: Index of the chosen lead (0-5).
        battle: Current battle state.

    Returns:
        A string like "/team 312456" where the first digit is the lead.
    """
    team_size = len(battle.team)
    lead = action + 1  # 1-indexed

    # Build order: lead first, then rest in original order
    order = [lead]
    for i in range(1, team_size + 1):
        if i != lead:
            order.append(i)

    return "/team " + "".join(str(x) for x in order)
