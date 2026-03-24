"""Main training loop integrating all components.

Orchestrates:
  - Two main agents (one per team) training via PPO
  - AlphaStar League: periodic snapshots, PFSP opponent selection
  - Win probability estimator training
  - Checkpoint saving/loading for resumable training
  - Reward shaping using win probability deltas
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
)

from pokerl.agent import PPOAgent
from pokerl.checkpoint import CheckpointManager
from pokerl.config import Config
from pokerl.env import RLPlayer, create_player, load_team
from pokerl.league import League
from pokerl.win_probability import WinProbabilityEstimator

logger = logging.getLogger(__name__)


class Trainer:
    """Main training orchestrator."""

    def __init__(self, config: Config, server_configuration: ServerConfiguration = None):
        self.config = config
        self.server_config = server_configuration or LocalhostServerConfiguration

        # Load teams
        self.team1_str = load_team(config.team1_path)
        self.team2_str = load_team(config.team2_path)

        # Create agents
        self.agent1 = PPOAgent(config, agent_id="team1_main")
        self.agent2 = PPOAgent(config, agent_id="team2_main")

        # League
        self.league = League(config)

        # Win probability estimator (shared between both agents)
        self.wp_estimator = WinProbabilityEstimator(config)

        # Checkpoint manager
        self.ckpt_manager = CheckpointManager(config)

        # Battle counter
        self.battle_count = 0

        # Stats tracking
        self.recent_results = []  # list of (team1_won: bool)

    def resume_if_available(self):
        """Resume from latest checkpoint if available."""
        if self.config.resume:
            if self.config.resume_path:
                self.battle_count = self.ckpt_manager.load_from_path(
                    self.config.resume_path,
                    self.agent1, self.agent2,
                    self.league, self.wp_estimator,
                )
            else:
                self.battle_count = self.ckpt_manager.load_latest(
                    self.agent1, self.agent2,
                    self.league, self.wp_estimator,
                )
            logger.info(f"Resumed from battle {self.battle_count}")

    async def train(self):
        """Main training loop."""
        self.resume_if_available()

        logger.info(
            f"Starting training: {self.config.total_battles} battles, "
            f"format={self.config.battle_format}"
        )
        logger.info(f"Action space size: {self.config.action_size}")
        logger.info(f"Battle obs size: 1032, Team preview obs size: 664")

        while self.battle_count < self.config.total_battles:
            await self._run_battle()
            self.battle_count += 1

            # PPO updates when buffer is full enough
            if len(self.agent1.battle_buffer) >= self.config.rollout_steps:
                self._update_agents()

            # Periodic checkpoint + league snapshot
            if self.battle_count % self.config.checkpoint_interval == 0:
                self._checkpoint_and_snapshot()

            # Win probability estimator update
            wp_metrics = self.wp_estimator.maybe_update()
            if wp_metrics:
                logger.info(
                    f"  WP update: loss={wp_metrics.get('wp_loss', 0):.4f}, "
                    f"mean_pred={wp_metrics.get('wp_mean_pred', 0):.3f}"
                )

            # Periodic logging
            if self.battle_count % 10 == 0:
                self._log_stats()

        # Final checkpoint
        self._checkpoint_and_snapshot()
        logger.info("Training complete!")

    async def _run_battle(self):
        """Run a single battle between two players.

        Opponent selection follows the AlphaStar league strategy:
        - Sometimes use the latest main agent
        - Sometimes use a PFSP-selected historical opponent
        """
        # Decide which agents to use
        player1_agent = self.agent1
        player2_agent = self.agent2

        # Check if we should use a league opponent instead
        use_league_opponent = False
        league_opponent = None

        if self.league.agents:
            # Team 1 might play against a league team 2 agent
            league_opponent = self.league.select_opponent(
                self.agent1.agent_id, team_id=0
            )

        # Create players
        player1 = create_player(
            agent=self.agent1,
            config=self.config,
            team_str=self.team1_str,
            collect_data=True,
            server_configuration=self.server_config,
        )

        if league_opponent is not None and np.random.random() < 0.3:
            # Use a frozen league opponent (no data collection)
            frozen_agent = PPOAgent(self.config, agent_id=league_opponent.agent_id)
            frozen_agent.load_weights_only(league_opponent.state_dict)
            frozen_agent.set_eval()

            player2 = create_player(
                agent=frozen_agent,
                config=self.config,
                team_str=self.team2_str,
                collect_data=False,
                deterministic=True,
                server_configuration=self.server_config,
            )
            use_league_opponent = True
        else:
            player2 = create_player(
                agent=self.agent2,
                config=self.config,
                team_str=self.team2_str,
                collect_data=True,
                server_configuration=self.server_config,
            )

        # Run the battle
        await player1.battle_against(player2, n_battles=1)

        # Determine outcome
        p1_won = player1.n_won_battles > 0

        # Compute terminal rewards
        reward_win = 1.0
        reward_loss = -1.0
        p1_reward = reward_win if p1_won else reward_loss
        p2_reward = reward_loss if p1_won else reward_win

        # Apply win probability reward shaping to rollout buffers
        self._apply_reward_shaping(player1, p1_reward, p1_won)
        if not use_league_opponent:
            self._apply_reward_shaping(player2, p2_reward, not p1_won)

        # Finalize episodes
        player1.on_battle_finished(p1_won, p1_reward)
        if not use_league_opponent:
            player2.on_battle_finished(not p1_won, p2_reward)

        # Store trajectories for win probability training
        self.wp_estimator.store_trajectory(
            player1.get_battle_observations(), p1_won
        )
        if not use_league_opponent:
            self.wp_estimator.store_trajectory(
                player2.get_battle_observations(), not p1_won
            )

        # Record in payoff matrix
        self.league.payoff.record_result(
            self.agent1.agent_id,
            self.agent2.agent_id if not use_league_opponent else league_opponent.agent_id,
            p1_won,
        )

        # Track recent results
        self.recent_results.append(p1_won)
        if len(self.recent_results) > 100:
            self.recent_results = self.recent_results[-100:]

    def _apply_reward_shaping(self, player: RLPlayer, terminal_reward: float,
                               won: bool):
        """Apply win probability reward shaping to buffered steps."""
        observations = player.get_battle_observations()
        buffer = player.agent.battle_buffer
        n_obs = len(observations)

        # Shape rewards for steps already in the buffer (from this episode)
        # We walk backwards through the buffer to find steps from this episode
        steps_to_shape = []
        for i in range(len(buffer) - 1, -1, -1):
            step = buffer.steps[i]
            if step.done:
                break  # previous episode
            steps_to_shape.append(i)
        steps_to_shape.reverse()

        for j, buf_idx in enumerate(steps_to_shape):
            step = buffer.steps[buf_idx]
            if j + 1 < len(observations):
                shaped = self.wp_estimator.compute_shaped_reward(
                    observations[j], observations[j + 1],
                    terminal_reward=0.0,
                    done=False,
                )
                step.reward = shaped

    def _update_agents(self):
        """Run PPO updates for both agents."""
        self.agent1.set_train()
        self.agent2.set_train()

        # Battle policy updates
        metrics1 = self.agent1.update_battle_policy()
        metrics2 = self.agent2.update_battle_policy()

        # Team preview policy updates
        preview1 = self.agent1.update_preview_policy()
        preview2 = self.agent2.update_preview_policy()

        if metrics1:
            logger.info(
                f"  Agent1 battle update: "
                f"policy_loss={metrics1.get('policy_loss', 0):.4f}, "
                f"value_loss={metrics1.get('value_loss', 0):.4f}, "
                f"entropy={metrics1.get('entropy', 0):.4f}"
            )
        if metrics2:
            logger.info(
                f"  Agent2 battle update: "
                f"policy_loss={metrics2.get('policy_loss', 0):.4f}, "
                f"value_loss={metrics2.get('value_loss', 0):.4f}, "
                f"entropy={metrics2.get('entropy', 0):.4f}"
            )

    def _checkpoint_and_snapshot(self):
        """Save checkpoint and add agents to league."""
        # Save full checkpoint
        self.ckpt_manager.save(
            self.agent1, self.agent2,
            self.league, self.wp_estimator,
            self.battle_count,
            extra_metadata={
                "recent_win_rate": self._recent_win_rate(),
                "agent1_wins": self.agent1.wins,
                "agent2_wins": self.agent2.wins,
            },
        )

        # Add snapshots to the league
        self.league.add_agent(self.agent1, team_id=0)
        self.league.add_agent(self.agent2, team_id=1)

        logger.info(
            f"Checkpoint at battle {self.battle_count}. "
            f"League size: {len(self.league.agents)}"
        )

    def _log_stats(self):
        """Log training statistics."""
        wr = self._recent_win_rate()
        logger.info(
            f"Battle {self.battle_count}/{self.config.total_battles} | "
            f"Team1 recent WR: {wr:.1%} | "
            f"Agent1: {self.agent1.wins}W/{self.agent1.losses}L | "
            f"Agent2: {self.agent2.wins}W/{self.agent2.losses}L | "
            f"League: {len(self.league.agents)} agents"
        )

    def _recent_win_rate(self) -> float:
        if not self.recent_results:
            return 0.5
        return sum(self.recent_results) / len(self.recent_results)
