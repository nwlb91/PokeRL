"""Main training loop integrating all components.

Orchestrates:
  - Two main agents (one per team) training via PPO
  - AlphaStar League: periodic snapshots, PFSP opponent selection
  - Win probability estimator training
  - Checkpoint saving/loading for resumable training
  - Reward shaping using win probability deltas
  - Concurrent battles via poke-env's max_concurrent_battles
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

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
from pokerl.plateau import PlateauDetector, PlateauInfo
from pokerl.win_probability import WinProbabilityEstimator

logger = logging.getLogger(__name__)


class Trainer:
    """Main training orchestrator with concurrent battle support."""

    def __init__(self, config: Config, server_configuration: ServerConfiguration = None, progress_callback=None):
        self.config = config
        self.server_config = server_configuration or LocalhostServerConfiguration
        self._progress_callback = progress_callback  # callable(battle_count, total_battles) or None

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
        self.greedy_eval_results: List[Tuple[int, float]] = []  # (battle_count, win_rate)
        self.train_wr_history: List[Tuple[int, float]] = []  # (battle_count, win_rate)
        self._latest_explained_variance: float = 0.0
        self._latest_metrics: dict = {}  # most recent PPO metrics, updated each update

        # Training metrics history (sampled every 50 battles alongside win rate)
        self.metrics_history: List[dict] = []  # each entry keyed by metric name

        # Persistent players — reused across battles to avoid reconnections
        self._player1: Optional[RLPlayer] = None
        self._player2: Optional[RLPlayer] = None

        # Eval players — stochastic (sample from policy), no data collection
        self._eval_player1: Optional[RLPlayer] = None
        self._eval_player2: Optional[RLPlayer] = None

        # Plateau detector — window and patience scale with checkpoint interval
        # so that the detector has enough granularity regardless of how often
        # we sample.  Default: 20-observation window, 10-observation patience.
        self.plateau_detector = PlateauDetector(
            window=20,
            patience=10,
            alpha=0.05,
            cv_threshold=0.10,
        )

        # Concurrency
        self._n_concurrent = max(1, config.num_parallel_battles)

    def _get_or_create_player1(self) -> RLPlayer:
        """Reuse persistent player1 or create a new one."""
        if self._player1 is None:
            self._player1 = create_player(
                agent=self.agent1,
                config=self.config,
                team_str=self.team1_str,
                collect_data=True,
                max_concurrent=self._n_concurrent,
                server_configuration=self.server_config,
            )
        return self._player1

    def _get_or_create_player2(self) -> RLPlayer:
        """Reuse persistent player2 or create a new one."""
        if self._player2 is None:
            self._player2 = create_player(
                agent=self.agent2,
                config=self.config,
                team_str=self.team2_str,
                collect_data=True,
                max_concurrent=self._n_concurrent,
                server_configuration=self.server_config,
            )
        return self._player2

    def _get_or_create_eval_player1(self) -> RLPlayer:
        """Reuse persistent eval player1 or create a new one."""
        if self._eval_player1 is None:
            self._eval_player1 = create_player(
                agent=self.agent1,
                config=self.config,
                team_str=self.team1_str,
                collect_data=False,
                deterministic=False,
                max_concurrent=1,
                server_configuration=self.server_config,
            )
        return self._eval_player1

    def _get_or_create_eval_player2(self) -> RLPlayer:
        """Reuse persistent eval player2 or create a new one."""
        if self._eval_player2 is None:
            self._eval_player2 = create_player(
                agent=self.agent2,
                config=self.config,
                team_str=self.team2_str,
                collect_data=False,
                deterministic=False,
                max_concurrent=1,
                server_configuration=self.server_config,
            )
        return self._eval_player2

    async def _run_greedy_eval(self) -> float:
        """Run evaluation battles between the two main agents.

        Agents sample from their policy distributions (stochastic) to
        match real play conditions.  Returns eval win rate for agent1.
        """
        player1 = self._get_or_create_eval_player1()
        player2 = self._get_or_create_eval_player2()

        self.agent1.set_eval()
        self.agent2.set_eval()

        wins_before = player1.n_won_battles
        total_before = player1.n_finished_battles

        n = self.config.greedy_eval_battles
        await player1.battle_against(player2, n_battles=n)

        wins_after = player1.n_won_battles
        total_after = player1.n_finished_battles
        played = total_after - total_before
        wins = wins_after - wins_before

        self.agent1.set_train()
        self.agent2.set_train()

        wr = wins / played if played > 0 else 0.5
        self.greedy_eval_results.append((self.battle_count, wr))
        return wr

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
            f"format={self.config.battle_format}, "
            f"concurrent={self._n_concurrent}"
        )
        logger.info(f"Action space size: {self.config.action_size}")
        logger.info(f"Battle obs size: 1032, Team preview obs size: 664")

        while self.battle_count < self.config.total_battles:
            # Run a batch of concurrent battles
            batch_size = min(
                self._n_concurrent,
                self.config.total_battles - self.battle_count,
            )
            await self._run_battle_batch(batch_size)
            self.battle_count += batch_size

            # PPO updates when buffer is full enough
            if len(self.agent1.battle_buffer) >= self.config.rollout_steps:
                self._update_agents()

            # Periodic checkpoint + league snapshot
            if self.battle_count % self.config.checkpoint_interval == 0:
                self._checkpoint_and_snapshot()

            # Win probability estimator update
            wp_metrics = self.wp_estimator.maybe_update()
            if wp_metrics:
                logger.debug(
                    f"  WP update: loss={wp_metrics.get('wp_loss', 0):.4f}, "
                    f"mean_pred={wp_metrics.get('wp_mean_pred', 0):.3f}"
                )

            # Periodic greedy evaluation
            if (self.config.greedy_eval_interval > 0 and
                    self.battle_count % self.config.greedy_eval_interval == 0):
                greedy_wr = await self._run_greedy_eval()
                logger.info(
                    f"  Greedy eval at battle {self.battle_count}: "
                    f"WR={greedy_wr:.1%} ({self.config.greedy_eval_battles} battles)"
                )

            # Periodic logging + plateau detection
            if self.battle_count % 50 == 0:
                self._log_stats()

                # Feed chosen metric to the plateau detector
                wr = self._recent_win_rate()
                if (self.config.plateau_metric == "greedy_wr"
                        and self.greedy_eval_results):
                    metric_val = self.greedy_eval_results[-1][1]
                elif self.config.plateau_metric == "explained_variance":
                    metric_val = self._latest_explained_variance
                else:
                    metric_val = wr
                plateau_info = self.plateau_detector.update(
                    metric_val, self.battle_count
                )
                if plateau_info.is_plateau and plateau_info.streak == self.plateau_detector.patience:
                    logger.warning(
                        f"Learning plateau detected at battle {self.battle_count} "
                        f"(metric={metric_val:.4f}, slope={plateau_info.slope:.6f}, "
                        f"p={plateau_info.p_value:.3f})"
                    )

            # Progress callback (for GUI progress bar etc.)
            if self._progress_callback:
                self._progress_callback(
                    self.battle_count,
                    self.config.total_battles,
                    self.plateau_detector,
                    self.metrics_history,
                    self.greedy_eval_results,
                    self.train_wr_history,
                )

        # Final checkpoint
        self._checkpoint_and_snapshot()
        logger.info("Training complete!")

    async def _run_battle_batch(self, n_battles: int):
        """Run n_battles concurrently using poke-env's built-in concurrency.

        poke-env multiplexes battles over a single websocket per player.
        With max_concurrent_battles > 1, battle_against(n_battles=N) runs
        up to max_concurrent_battles in parallel.
        """
        player1 = self._get_or_create_player1()
        player2 = self._get_or_create_player2()

        # Snapshot battle counts before this batch
        p1_wins_before = player1.n_won_battles
        p1_total_before = player1.n_finished_battles

        await player1.battle_against(player2, n_battles=n_battles)

        # Compute batch results
        p1_wins_after = player1.n_won_battles
        p1_total_after = player1.n_finished_battles
        battles_played = p1_total_after - p1_total_before
        batch_wins = p1_wins_after - p1_wins_before

        # Process each completed battle
        for i in range(battles_played):
            p1_won = i < batch_wins  # approximate: first batch_wins were wins

            p1_base = 1.0 if p1_won else -1.0
            p2_base = -p1_base

            # Add KO differential bonus so even the losing side has
            # variance in terminal reward (prevents gradient collapse
            # in hopeless matchups).
            ko_w = self.config.ko_reward_weight
            dmg_w = self.config.damage_reward_weight
            p1_reward = (
                p1_base
                + ko_w * player1.get_ko_differential()
                + dmg_w * player1.get_damage_differential()
            )
            p2_reward = (
                p2_base
                + ko_w * player2.get_ko_differential()
                + dmg_w * player2.get_damage_differential()
            )

            # Apply reward shaping
            self._apply_reward_shaping(player1, p1_reward)
            self._apply_reward_shaping(player2, p2_reward)

            # Finalize
            player1.on_battle_finished(p1_won, p1_reward)
            player2.on_battle_finished(not p1_won, p2_reward)

            # Store WP trajectories
            self.wp_estimator.store_trajectory(
                player1.get_battle_observations(), p1_won
            )
            self.wp_estimator.store_trajectory(
                player2.get_battle_observations(), not p1_won
            )

            # Payoff
            self.league.payoff.record_result(
                self.agent1.agent_id, self.agent2.agent_id, p1_won
            )

            self.recent_results.append(p1_won)

        if len(self.recent_results) > 100:
            self.recent_results = self.recent_results[-100:]

    def _apply_reward_shaping(self, player: RLPlayer, terminal_reward: float):
        """Apply vectorized win probability reward shaping."""
        observations = player.get_battle_observations()
        buffer = player.agent.battle_buffer

        if len(observations) < 2:
            return

        # Batch-predict all WP deltas in one forward pass
        shaped_rewards = self.wp_estimator.compute_shaped_rewards_batch(observations)

        # Walk backwards to find un-done steps from current episode
        steps_to_shape = []
        for i in range(len(buffer) - 1, -1, -1):
            if buffer.steps[i].done:
                break
            steps_to_shape.append(i)
        steps_to_shape.reverse()

        # Apply shaped rewards with survival bonus so the losing side
        # receives a small positive per-turn signal even when WP deltas
        # are near zero (prevents gradient starvation in 100-0 matchups).
        surv = self.config.survival_reward_per_turn
        for j, buf_idx in enumerate(steps_to_shape):
            if j < len(shaped_rewards):
                buffer.steps[buf_idx].reward = shaped_rewards[j] + surv
            else:
                buffer.steps[buf_idx].reward = surv

    def _update_agents(self):
        """Run PPO updates for both agents."""
        self.agent1.set_train()
        self.agent2.set_train()

        metrics1 = self.agent1.update_battle_policy()
        metrics2 = self.agent2.update_battle_policy()
        self.agent1.update_preview_policy()
        self.agent2.update_preview_policy()

        if metrics1:
            self._latest_explained_variance = metrics1.get(
                'explained_variance', 0.0
            )
            self._latest_metrics = {
                'policy_loss': metrics1.get('policy_loss', 0.0),
                'value_loss': metrics1.get('value_loss', 0.0),
                'entropy': metrics1.get('entropy', 0.0),
                'explained_variance': metrics1.get('explained_variance', 0.0),
                'mean_episode_return': metrics1.get('mean_episode_return', 0.0),
            }
            msg = (
                f"  Agent1 battle update: "
                f"policy_loss={metrics1.get('policy_loss', 0):.4f}, "
                f"value_loss={metrics1.get('value_loss', 0):.4f}, "
                f"entropy={metrics1.get('entropy', 0):.4f}, "
                f"explained_var={metrics1.get('explained_variance', 0):.4f}, "
                f"mean_ep_return={metrics1.get('mean_episode_return', 0):.4f}"
            )
            if 'mean_uncertainty' in metrics1:
                msg += f", uncertainty={metrics1['mean_uncertainty']:.4f}"
            logger.debug(msg)
        if metrics2:
            msg = (
                f"  Agent2 battle update: "
                f"policy_loss={metrics2.get('policy_loss', 0):.4f}, "
                f"value_loss={metrics2.get('value_loss', 0):.4f}, "
                f"entropy={metrics2.get('entropy', 0):.4f}, "
                f"explained_var={metrics2.get('explained_variance', 0):.4f}, "
                f"mean_ep_return={metrics2.get('mean_episode_return', 0):.4f}"
            )
            if 'mean_uncertainty' in metrics2:
                msg += f", uncertainty={metrics2['mean_uncertainty']:.4f}"
            logger.debug(msg)

    def _checkpoint_and_snapshot(self):
        """Save checkpoint and maybe add agents to league."""
        wr = self._recent_win_rate()

        self.ckpt_manager.save(
            self.agent1, self.agent2,
            self.league, self.wp_estimator,
            self.battle_count,
            extra_metadata={
                "recent_win_rate": wr,
                "greedy_win_rate": (
                    self.greedy_eval_results[-1][1]
                    if self.greedy_eval_results else None
                ),
                "explained_variance": self._latest_explained_variance,
                "agent1_wins": self.agent1.wins,
                "agent2_wins": self.agent2.wins,
            },
        )

        # Admission-gated: only adds if the agent is novel or has improved
        self.league.add_agent(self.agent1, team_id=0, current_win_rate=wr)
        self.league.add_agent(self.agent2, team_id=1, current_win_rate=1 - wr)

        # Periodic pruning of redundant agents
        self.league.maybe_prune(self.battle_count)

        logger.info(
            f"Checkpoint at battle {self.battle_count}. "
            f"League size: {len(self.league.agents)}"
        )

    def _log_stats(self):
        """Log training statistics."""
        wr = self._recent_win_rate()
        greedy_str = ""
        if self.greedy_eval_results:
            _, last_greedy_wr = self.greedy_eval_results[-1]
            greedy_str = f" | Greedy WR: {last_greedy_wr:.1%}"

        # Track training win rate history for GUI charting
        self.train_wr_history.append((self.battle_count, wr))

        # Snapshot latest PPO metrics at the same rate as win rate logging
        if self._latest_metrics:
            self.metrics_history.append({
                'battle_count': self.battle_count,
                **self._latest_metrics,
            })

        logger.info(
            f"Battle {self.battle_count}/{self.config.total_battles} | "
            f"Team1 recent WR: {wr:.1%}{greedy_str} | "
            f"Agent1: {self.agent1.wins}W/{self.agent1.losses}L | "
            f"Agent2: {self.agent2.wins}W/{self.agent2.losses}L | "
            f"League: {len(self.league.agents)} agents"
        )

    def _recent_win_rate(self) -> float:
        if not self.recent_results:
            return 0.5
        return sum(self.recent_results) / len(self.recent_results)
