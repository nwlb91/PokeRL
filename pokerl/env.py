"""Custom poke-env environment that integrates RL agents with the battle system.

This module provides a PokeRL-specific environment that:
  - Uses our comprehensive feature embedding
  - Handles team preview with learned lead selection
  - Integrates with the PPO agent's action space
  - Provides win probability-shaped rewards
  - Supports concurrent battles via per-battle state isolation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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


@dataclass
class _BattleEpisodeState:
    """Per-battle episode state for rollout collection and reward shaping.

    Pending steps are buffered locally (not in the shared RolloutBuffer)
    so that concurrent battles don't interleave steps.  They are flushed
    to the shared buffer in ``finalize_episode()`` after reward shaping.
    """
    battle_observations: list = field(default_factory=list)
    pending_steps: list = field(default_factory=list)  # List[RolloutStep]
    prev_obs: Optional[np.ndarray] = None
    prev_action: Optional[int] = None
    prev_action_mask: Optional[np.ndarray] = None
    prev_log_prob: Optional[float] = None
    prev_value: Optional[float] = None
    team_preview_obs: Optional[np.ndarray] = None
    team_preview_mask: Optional[np.ndarray] = None
    team_preview_action: Optional[int] = None
    team_preview_log_prob: Optional[float] = None
    team_preview_value: Optional[float] = None
    own_fainted: int = 0
    opp_fainted: int = 0
    own_total_hp_lost: float = 0.0
    opp_total_hp_lost: float = 0.0
    prev_own_hp: Optional[float] = None
    prev_opp_hp: Optional[float] = None


@dataclass
class CompletedEpisode:
    """A completed battle episode with all collected data."""
    battle_tag: str
    state: _BattleEpisodeState
    won: bool


class RLPlayer(Player):
    """A poke-env Player controlled by a PPO agent.

    Delegates all decisions to the agent's neural networks:
      - Team preview: uses TeamPreviewNet
      - Battle moves: uses PolicyValueNet with action masking

    Supports concurrent battles by isolating per-battle state
    in a dict keyed by battle.battle_tag.
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

        # Per-battle state keyed by battle.battle_tag
        self._episode_states: Dict[str, _BattleEpisodeState] = {}

        # Completed episodes ready for the trainer to consume
        self._completed_episodes: List[CompletedEpisode] = []

        # Track stale-challenge recovery
        self._challenge_cancel_delay = 0.5  # seconds before retrying

    async def _handle_message(self, message: str):
        """Override to recover from 'already challenging' popups.

        When the Showdown server reports a stale pending challenge, cancel
        it so the next challenge attempt can succeed instead of hanging.
        """
        if "|popup|" in message and "already challenging" in message.lower():
            logger.warning(
                "Stale challenge detected, sending /cancelchallenge"
            )
            await self.ps_client.send_message("/cancelchallenge")
            await asyncio.sleep(self._challenge_cancel_delay)
            return
        await super()._handle_message(message)

    def _get_episode_state(self, battle: Battle) -> _BattleEpisodeState:
        """Get or create the episode state for a specific battle."""
        tag = battle.battle_tag
        if tag not in self._episode_states:
            self._episode_states[tag] = _BattleEpisodeState()
        return self._episode_states[tag]

    def teampreview(self, battle: Battle) -> str:
        """Select a lead using the team preview network."""
        obs = embed_team_preview(battle)
        mask = get_team_preview_mask(battle)

        action, log_prob, value = self.agent.select_preview_action(
            obs, mask, deterministic=self.deterministic
        )

        # Store for later reward assignment
        if self.collect_data:
            state = self._get_episode_state(battle)
            state.team_preview_obs = obs
            state.team_preview_mask = mask
            state.team_preview_action = action
            state.team_preview_log_prob = log_prob
            state.team_preview_value = value

        return team_preview_to_order(action, battle)

    def choose_move(self, battle: Battle) -> BattleOrder:
        """Select a battle action using the policy network."""
        obs = embed_battle(battle)
        action_mask = get_action_mask(battle, self.config)

        # Store observation for win probability training
        if self.collect_data:
            state = self._get_episode_state(battle)
            state.battle_observations.append(obs)

            # Track KO counts for reward shaping
            state.own_fainted = sum(1 for m in battle.team.values() if m.fainted)
            state.opp_fainted = sum(
                1 for m in battle.opponent_team.values() if m.fainted
            )

            # Track damage dealt/received for reward shaping
            own_hp = sum(m.current_hp_fraction for m in battle.team.values())
            opp_hp = sum(m.current_hp_fraction for m in battle.opponent_team.values())
            if state.prev_own_hp is not None:
                own_delta = state.prev_own_hp - own_hp
                opp_delta = state.prev_opp_hp - opp_hp
                state.own_total_hp_lost += max(0.0, own_delta)
                state.opp_total_hp_lost += max(0.0, opp_delta)
            state.prev_own_hp = own_hp
            state.prev_opp_hp = opp_hp

        # If we have a previous step, buffer it locally (not in the shared
        # RolloutBuffer) so concurrent battles don't interleave steps.
        if self.collect_data and (state := self._episode_states.get(battle.battle_tag)) and state.prev_obs is not None:
            step = RolloutStep(
                obs=state.prev_obs,
                action=state.prev_action,
                action_mask=state.prev_action_mask,
                log_prob=state.prev_log_prob,
                value=state.prev_value,
                reward=0.0,  # will be reshaped later
                done=False,
            )
            state.pending_steps.append(step)

        action, log_prob, value = self.agent.select_battle_action(
            obs, action_mask, deterministic=self.deterministic
        )

        # Store for next step
        if self.collect_data:
            state = self._get_episode_state(battle)
            state.prev_obs = obs
            state.prev_action = action
            state.prev_action_mask = action_mask
            state.prev_log_prob = log_prob
            state.prev_value = value

        order = action_to_order(action, battle, self.config)
        return order

    def _battle_finished_callback(self, battle: AbstractBattle):
        """Called by poke-env when an individual battle finishes.

        Moves the episode state from active to completed so the trainer
        can process each battle individually, even during concurrent execution.
        """
        tag = battle.battle_tag
        state = self._episode_states.pop(tag, None)
        if state is None:
            # Non-data-collecting player or unknown battle
            return

        self._completed_episodes.append(
            CompletedEpisode(battle_tag=tag, state=state, won=battle.won)
        )

    def finalize_episode(self, episode: CompletedEpisode, terminal_reward: float):
        """Finalize a completed episode by flushing steps to the shared buffer.

        Called by the trainer after reward shaping has been applied to
        ``episode.state.pending_steps``.  Steps are flushed contiguously
        so that GAE computation sees correct episode boundaries even when
        multiple battles ran concurrently.
        """
        if not self.collect_data:
            return

        state = episode.state

        # Flush all pending mid-battle steps (rewards already shaped)
        for step in state.pending_steps:
            self.agent.battle_buffer.add(step)

        # Record final battle step
        if state.prev_obs is not None:
            step = RolloutStep(
                obs=state.prev_obs,
                action=state.prev_action,
                action_mask=state.prev_action_mask,
                log_prob=state.prev_log_prob,
                value=state.prev_value,
                reward=terminal_reward,
                done=True,
            )
            self.agent.battle_buffer.add(step)

        # Record team preview step (reward = terminal reward, since lead
        # choice affects the entire game outcome)
        if state.team_preview_obs is not None:
            step = RolloutStep(
                obs=state.team_preview_obs,
                action=state.team_preview_action,
                action_mask=state.team_preview_mask,
                log_prob=state.team_preview_log_prob,
                value=state.team_preview_value,
                reward=terminal_reward,
                done=True,
            )
            self.agent.preview_buffer.add(step)

        # Update agent stats
        self.agent.total_battles += 1
        if episode.won:
            self.agent.wins += 1
        else:
            self.agent.losses += 1

    def pop_completed_episodes(self) -> List[CompletedEpisode]:
        """Return and clear all completed episodes.

        The trainer calls this after battle_against() to process
        each completed battle individually.
        """
        episodes = self._completed_episodes
        self._completed_episodes = []
        return episodes

    @staticmethod
    def get_ko_differential(episode: CompletedEpisode) -> float:
        """Return normalised KO differential for a completed episode.

        Positive means we knocked out more of theirs than they knocked out
        of ours.  Used to provide gradient signal even in hopeless matchups.
        """
        s = episode.state
        return (s.opp_fainted - s.own_fainted) / 6.0

    @staticmethod
    def get_damage_differential(episode: CompletedEpisode) -> float:
        """Return normalised damage differential for a completed episode.

        Positive means we dealt more damage than we received.  Each side can
        lose at most 6 HP-fractions (one per Pokemon), so dividing by 6 keeps
        the value in [-1, 1].
        """
        s = episode.state
        return (s.opp_total_hp_lost - s.own_total_hp_lost) / 6.0

    @staticmethod
    def get_battle_observations(episode: CompletedEpisode) -> list:
        """Return collected observations for win probability training."""
        return episode.state.battle_observations


def load_team(path: str) -> str:
    """Load a team from a text file in Showdown format."""
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Team file not found: {path!r}. "
            "Please provide a valid path to a Showdown-format team file."
        )
    if not p.is_file():
        raise ValueError(f"Team path is not a file: {path!r}")

    text = p.read_text().strip()
    if not text:
        raise ValueError(f"Team file is empty: {path!r}")
    return text


def create_player(
    agent: PPOAgent,
    config: Config,
    team_str: str,
    username: str = None,
    collect_data: bool = True,
    deterministic: bool = False,
    max_concurrent: int = 1,
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
        max_concurrent_battles=max_concurrent,
    )
    return player
