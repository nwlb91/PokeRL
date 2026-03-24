"""Custom poke-env environment that integrates RL agents with the battle system.

This module provides a PokeRL-specific environment that:
  - Uses our comprehensive feature embedding
  - Handles team preview with learned lead selection
  - Integrates with the PPO agent's action space
  - Provides win probability-shaped rewards
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple, Union, Awaitable

import numpy as np

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.battle import Battle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.player.player import Player
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder

from pokerl.actions import (
    action_to_order,
    get_action_mask,
    get_team_preview_mask,
    team_preview_to_order,
)
from pokerl.agent import PPOAgent, RolloutStep
from pokerl.config import Config
from pokerl.features import embed_battle, embed_team_preview

logger = logging.getLogger(__name__)


class RLPlayer(Player):
    """A poke-env Player controlled by a PPO agent.

    Delegates all decisions to the agent's neural networks:
      - Team preview: uses TeamPreviewNet
      - Battle moves: uses PolicyValueNet with action masking
    """

    def __init__(
        self,
        agent: PPOAgent,
        config: Config,
        collect_data: bool = True,
        deterministic: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agent = agent
        self.config = config
        self.collect_data = collect_data
        self.deterministic = deterministic

        # Per-battle state for rollout collection
        self._battle_observations = []
        self._prev_obs = None
        self._prev_action = None
        self._prev_action_mask = None
        self._prev_log_prob = None
        self._prev_value = None
        self._team_preview_obs = None
        self._team_preview_mask = None
        self._team_preview_action = None
        self._team_preview_log_prob = None
        self._team_preview_value = None

    def teampreview(self, battle: Battle) -> str:
        """Select a lead using the team preview network."""
        obs = embed_team_preview(battle)
        mask = get_team_preview_mask(battle)

        action, log_prob, value = self.agent.select_preview_action(
            obs, mask, deterministic=self.deterministic
        )

        # Store for later reward assignment
        if self.collect_data:
            self._team_preview_obs = obs
            self._team_preview_mask = mask
            self._team_preview_action = action
            self._team_preview_log_prob = log_prob
            self._team_preview_value = value

        return team_preview_to_order(action, battle)

    def choose_move(self, battle: Battle) -> BattleOrder:
        """Select a battle action using the policy network."""
        obs = embed_battle(battle)
        action_mask = get_action_mask(battle, self.config)

        # Store observation for win probability training
        if self.collect_data:
            self._battle_observations.append(obs)

        # If we have a previous step, record its reward (0 for mid-battle)
        if self.collect_data and self._prev_obs is not None:
            step = RolloutStep(
                obs=self._prev_obs,
                action=self._prev_action,
                action_mask=self._prev_action_mask,
                log_prob=self._prev_log_prob,
                value=self._prev_value,
                reward=0.0,  # will be reshaped later
                done=False,
            )
            self.agent.battle_buffer.add(step)

        action, log_prob, value = self.agent.select_battle_action(
            obs, action_mask, deterministic=self.deterministic
        )

        # Store for next step
        if self.collect_data:
            self._prev_obs = obs
            self._prev_action = action
            self._prev_action_mask = action_mask
            self._prev_log_prob = log_prob
            self._prev_value = value

        order = action_to_order(action, battle, self.config)
        return order

    def on_battle_finished(self, won: bool, terminal_reward: float):
        """Called after a battle ends to finalize rollout data.

        Records the final step with the terminal reward.
        """
        if not self.collect_data:
            return

        # Record final battle step
        if self._prev_obs is not None:
            step = RolloutStep(
                obs=self._prev_obs,
                action=self._prev_action,
                action_mask=self._prev_action_mask,
                log_prob=self._prev_log_prob,
                value=self._prev_value,
                reward=terminal_reward,
                done=True,
            )
            self.agent.battle_buffer.add(step)

        # Record team preview step (reward = terminal reward, since lead
        # choice affects the entire game outcome)
        if self._team_preview_obs is not None:
            step = RolloutStep(
                obs=self._team_preview_obs,
                action=self._team_preview_action,
                action_mask=self._team_preview_mask,
                log_prob=self._team_preview_log_prob,
                value=self._team_preview_value,
                reward=terminal_reward,
                done=True,
            )
            self.agent.preview_buffer.add(step)

        # Update agent stats
        self.agent.total_battles += 1
        if won:
            self.agent.wins += 1
        else:
            self.agent.losses += 1

        self._reset_episode_state()

    def get_battle_observations(self):
        """Return collected observations for win probability training."""
        return self._battle_observations

    def _reset_episode_state(self):
        self._battle_observations = []
        self._prev_obs = None
        self._prev_action = None
        self._prev_action_mask = None
        self._prev_log_prob = None
        self._prev_value = None
        self._team_preview_obs = None
        self._team_preview_mask = None
        self._team_preview_action = None
        self._team_preview_log_prob = None
        self._team_preview_value = None


def load_team(path: str) -> str:
    """Load a team from a text file in Showdown format."""
    with open(path, "r") as f:
        return f.read().strip()


def create_player(
    agent: PPOAgent,
    config: Config,
    team_str: str,
    username: str = None,
    collect_data: bool = True,
    deterministic: bool = False,
    server_configuration: ServerConfiguration = None,
) -> RLPlayer:
    """Create an RLPlayer with the given configuration."""
    if server_configuration is None:
        server_configuration = LocalhostServerConfiguration

    account_config = AccountConfiguration(username, None) if username else None

    player = RLPlayer(
        agent=agent,
        config=config,
        collect_data=collect_data,
        deterministic=deterministic,
        account_configuration=account_config,
        battle_format=config.battle_format,
        team=ConstantTeambuilder(team_str),
        server_configuration=server_configuration,
        max_concurrent_battles=1,
    )
    return player
