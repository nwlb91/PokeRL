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
import random
import time
from collections import deque
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
        self.greedy_eval_results: deque = deque(maxlen=5000)  # (battle_count, win_rate)
        self.train_wr_history: deque = deque(maxlen=5000)  # (battle_count, win_rate)
        self._latest_explained_variance: float = 0.0
        self._latest_metrics: dict = {}  # most recent PPO metrics, updated each update

        # Training metrics history (sampled every 50 battles alongside win rate)
        self.metrics_history: deque = deque(maxlen=5000)  # each entry keyed by metric name

        # Persistent players — reused across battles to avoid reconnections
        self._player1: Optional[RLPlayer] = None
        self._player2: Optional[RLPlayer] = None

        # Eval players — stochastic (sample from policy), no data collection
        self._eval_player1: Optional[RLPlayer] = None
        self._eval_player2: Optional[RLPlayer] = None

        # League opponent player (reused, weights swapped as needed)
        self._league_opponent_agent: Optional[PPOAgent] = None
        self._league_opponent_player: Optional[RLPlayer] = None
        self._league_opponent_id: Optional[str] = None

        # Best-model tracking and regression protection
        self.best_eval_win_rate: float = 0.0
        self.regression_counter: int = 0

        # Plateau response: entropy bump
        self._entropy_bump_until: int = 0

        # Plateau detector — window and patience scale with checkpoint interval
        # so that the detector has enough granularity regardless of how often
        # we sample.  Default: 20-observation window, 10-observation patience.
        self.plateau_detector = PlateauDetector(
            window=20,
            patience=10,
            alpha=0.05,
            cv_threshold=0.10,
        )

        # Concurrency — batch post-processing currently assumes one battle at a
        # time (KO/damage tracking, observation lists, and win attribution are
        # per-player singletons that get overwritten across concurrent battles).
        if config.num_parallel_battles > 1:
            logger.warning(
                "num_parallel_battles > 1 is not yet supported correctly "
                "(per-battle reward data is lost). Forcing to 1."
            )
        self._n_concurrent = 1

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

    def _get_league_player(self, league_agent, team_id: int) -> RLPlayer:
        """Get or create a frozen player for a league opponent.

        Reuses the player if possible, only reloading weights when the
        league agent changes.
        """
        from pokerl.league import LeagueAgent

        if (self._league_opponent_player is not None
                and self._league_opponent_id == league_agent.agent_id):
            return self._league_opponent_player

        # Create or reuse the agent shell
        if self._league_opponent_agent is None:
            self._league_opponent_agent = PPOAgent(self.config, agent_id=league_agent.agent_id)

        self._league_opponent_agent.agent_id = league_agent.agent_id
        self._league_opponent_agent.load_weights_only(league_agent.state_dict)

        # Pick the right team string
        team_str = self.team1_str if team_id == 0 else self.team2_str

        # Create player if needed, or reuse existing (weights already swapped)
        if self._league_opponent_player is None:
            self._league_opponent_player = create_player(
                agent=self._league_opponent_agent,
                config=self.config,
                team_str=team_str,
                collect_data=False,
                deterministic=False,
                max_concurrent=self._n_concurrent,
                server_configuration=self.server_config,
            )

        self._league_opponent_id = league_agent.agent_id
        return self._league_opponent_player

    def _get_effective_entropy_coef(self) -> float:
        """Compute the entropy coefficient with annealing and plateau bump."""
        cfg = self.config
        if cfg.entropy_anneal_battles > 0:
            progress = min(1.0, self.battle_count / cfg.entropy_anneal_battles)
            coef = cfg.entropy_coef_start + progress * (cfg.entropy_coef_end - cfg.entropy_coef_start)
        else:
            coef = cfg.entropy_coef
        # Plateau bump
        if self.battle_count < self._entropy_bump_until:
            coef += cfg.plateau_entropy_bump
        return coef

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

        while self.config.infinite_training or self.battle_count < self.config.total_battles:
            # Run a batch of concurrent battles
            if self.config.infinite_training:
                batch_size = self._n_concurrent
            else:
                batch_size = min(
                    self._n_concurrent,
                    self.config.total_battles - self.battle_count,
                )
            await self._run_battle_batch(batch_size)
            self.battle_count += batch_size

            # PPO updates when buffer is full enough
            if len(self.agent1.battle_buffer) >= self.config.rollout_steps:
                self._update_agents()

            # Team preview updates on a separate (lower) threshold so lead
            # selection learns at a comparable rate despite producing only
            # one sample per game.
            if len(self.agent1.preview_buffer) >= self.config.preview_rollout_steps:
                self._update_preview()

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

                # Best-model tracking and regression rollback
                if self.config.best_model_tracking:
                    if greedy_wr > self.best_eval_win_rate:
                        self.best_eval_win_rate = greedy_wr
                        self.regression_counter = 0
                        self.ckpt_manager.save_best(
                            self.agent1, self.agent2,
                            self.league, self.wp_estimator,
                            self.battle_count, greedy_wr,
                        )
                    elif greedy_wr < self.best_eval_win_rate - self.config.regression_threshold:
                        self.regression_counter += 1
                        logger.warning(
                            f"Regression signal {self.regression_counter}/"
                            f"{self.config.regression_eval_window}: "
                            f"current={greedy_wr:.1%}, best={self.best_eval_win_rate:.1%}"
                        )
                        if self.regression_counter >= self.config.regression_eval_window:
                            logger.warning(
                                f"Policy regression confirmed. "
                                f"Rolling back to best model (WR={self.best_eval_win_rate:.1%})."
                            )
                            self.ckpt_manager.load_best(
                                self.agent1, self.agent2,
                                self.league, self.wp_estimator,
                            )
                            self.regression_counter = 0
                    else:
                        self.regression_counter = 0

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
                    # Plateau response
                    if self.config.plateau_action == "entropy_bump":
                        self._entropy_bump_until = (
                            self.battle_count + self.config.plateau_bump_duration
                        )
                        logger.info(
                            f"Plateau response: entropy bump (+{self.config.plateau_entropy_bump}) "
                            f"for {self.config.plateau_bump_duration} battles"
                        )
                    elif self.config.plateau_action == "noise_inject":
                        self._inject_param_noise()
                        logger.info("Plateau response: parameter noise injected")

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

        Selects opponents from the league when available, using the
        configured main/PFSP/self-play distribution.  When a league
        opponent is selected, only the training player collects data.
        """
        # Decide which training player and opponent to use this batch.
        # Alternate which agent gets league exposure each batch.
        if self.battle_count % 2 == 0:
            training_player_fn = self._get_or_create_player1
            live_opponent_fn = self._get_or_create_player2
            training_agent = self.agent1
            live_agent = self.agent2
            training_team_id = 0
        else:
            training_player_fn = self._get_or_create_player2
            live_opponent_fn = self._get_or_create_player1
            training_agent = self.agent2
            live_agent = self.agent1
            training_team_id = 1

        training_player = training_player_fn()
        use_league = False
        opponent_agent_id = live_agent.agent_id

        # Try to select a league opponent
        if self.league.agents:
            result = self.league.select_opponent(
                training_agent.agent_id, training_team_id,
            )
            if result is not None:
                league_agent, kind = result
                if kind in ("pfsp", "self_play"):
                    # Use frozen league opponent
                    opponent_team_id = league_agent.team_id
                    opponent_player = self._get_league_player(
                        league_agent, opponent_team_id,
                    )
                    opponent_agent_id = league_agent.agent_id
                    use_league = True
                    logger.debug(
                        f"League opponent: {league_agent.agent_id} ({kind})"
                    )

        if not use_league:
            opponent_player = live_opponent_fn()
            opponent_agent_id = live_agent.agent_id

        # Snapshot battle counts before this batch
        tp_wins_before = training_player.n_won_battles
        tp_total_before = training_player.n_finished_battles

        await training_player.battle_against(opponent_player, n_battles=n_battles)

        # Compute batch results
        tp_wins_after = training_player.n_won_battles
        tp_total_after = training_player.n_finished_battles
        battles_played = tp_total_after - tp_total_before
        batch_wins = tp_wins_after - tp_wins_before

        # Process each completed battle
        ko_w = self.config.ko_reward_weight
        dmg_w = self.config.damage_reward_weight

        for i in range(battles_played):
            tp_won = i < batch_wins  # approximate: first batch_wins were wins

            tp_base = 1.0 if tp_won else -1.0

            tp_reward = (
                tp_base
                + ko_w * training_player.get_ko_differential()
                + dmg_w * training_player.get_damage_differential()
            )

            # Apply reward shaping and finalize for training player
            self._apply_reward_shaping(training_player, tp_reward)
            training_player.on_battle_finished(tp_won, tp_reward)

            # If fighting the live opponent, also process their data
            if not use_league:
                opp_base = -tp_base
                opp_reward = (
                    opp_base
                    + ko_w * opponent_player.get_ko_differential()
                    + dmg_w * opponent_player.get_damage_differential()
                )
                self._apply_reward_shaping(opponent_player, opp_reward)
                opponent_player.on_battle_finished(not tp_won, opp_reward)

                # Store WP trajectory for live opponent too
                self.wp_estimator.store_trajectory(
                    opponent_player.get_battle_observations(), not tp_won,
                )

            # Store WP trajectory for training player
            self.wp_estimator.store_trajectory(
                training_player.get_battle_observations(), tp_won,
            )

            # Record payoff with correct agent IDs
            self.league.payoff.record_result(
                training_agent.agent_id, opponent_agent_id, tp_won,
            )

            # Track win rate from team1's perspective
            if training_team_id == 0:
                self.recent_results.append(tp_won)
            else:
                self.recent_results.append(not tp_won)

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

    def _inject_param_noise(self):
        """Inject small Gaussian noise into policy parameters to escape plateaus."""
        import torch
        noise_scale = 0.01
        for agent in (self.agent1, self.agent2):
            for param in agent.battle_net.parameters():
                if param.requires_grad:
                    param.data.add_(torch.randn_like(param.data) * noise_scale)

    def _update_agents(self):
        """Run PPO updates for both agents."""
        self.agent1.set_train()
        self.agent2.set_train()

        ent_coef = self._get_effective_entropy_coef()

        metrics1 = self.agent1.update_battle_policy(entropy_coef_override=ent_coef)
        metrics2 = self.agent2.update_battle_policy(entropy_coef_override=ent_coef)

        # Step LR schedulers (agent2 uses inverted metric since wr is from team1's perspective)
        wr = self._recent_win_rate()
        self.agent1.step_lr_scheduler(metric=wr)
        self.agent2.step_lr_scheduler(metric=1.0 - wr)

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

    def _update_preview(self):
        """Run PPO updates for both agents' team preview policies.

        Uses separate rollout threshold and more PPO epochs to compensate
        for the much sparser data (1 step per game vs ~20-40 for battle).
        """
        ent_coef = self._get_effective_entropy_coef()
        self.agent1.update_preview_policy(
            entropy_coef_override=ent_coef,
            ppo_epochs_override=self.config.preview_ppo_epochs,
        )
        self.agent2.update_preview_policy(
            entropy_coef_override=ent_coef,
            ppo_epochs_override=self.config.preview_ppo_epochs,
        )

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
