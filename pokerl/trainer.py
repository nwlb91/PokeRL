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
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch.optim as optim

from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
)

from pokerl.agent import PPOAgent
from pokerl.checkpoint import CheckpointManager
from pokerl.config import Config
from pokerl.env import CompletedEpisode, RLPlayer, create_player, load_team
from pokerl.league import League
from pokerl.plateau import PlateauDetector, PlateauInfo
from pokerl.features import BATTLE_OBS_SIZE, TEAM_PREVIEW_OBS_SIZE
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
        self.league.live_agent_ids = [self.agent1.agent_id, self.agent2.agent_id]

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

        # Per-team baselines for absolute skill measurement
        self._baseline_agent1: Optional[PPOAgent] = None  # best known team1 agent
        self._baseline_player1: Optional[RLPlayer] = None
        self._baseline_agent2: Optional[PPOAgent] = None  # best known team2 agent
        self._baseline_player2: Optional[RLPlayer] = None
        self.baseline_eval_results_team1: deque = deque(maxlen=5000)  # (battle_count, wr)
        self.baseline_eval_results_team2: deque = deque(maxlen=5000)  # (battle_count, wr)
        self._baseline1_promotions: int = 0
        self._baseline2_promotions: int = 0
        self._baseline_ref_wr1: float = 0.5  # baseline1's WR vs baseline2
        self._baseline_ref_wr2: float = 0.5  # baseline2's WR vs baseline1

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

        self._n_concurrent = config.num_parallel_battles

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
                max_concurrent=self._n_concurrent,
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
                max_concurrent=self._n_concurrent,
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

    async def _init_baselines(self):
        """Freeze copies of both agents as baselines for absolute skill measurement.

        Each baseline plays its respective team.  Current agents are evaluated
        against the *opposing* baseline so that promotions reflect the ability
        to beat the strongest known version of the other team.
        """
        self._baseline_agent1 = PPOAgent(self.config, agent_id="baseline_team1")
        self._baseline_agent1.load_weights_only(self.agent1.get_state_dict())
        self._baseline_agent1.set_eval()
        self._baseline_player1 = create_player(
            agent=self._baseline_agent1,
            config=self.config,
            team_str=self.team1_str,
            collect_data=False,
            deterministic=False,
            max_concurrent=1,
            server_configuration=self.server_config,
        )

        self._baseline_agent2 = PPOAgent(self.config, agent_id="baseline_team2")
        self._baseline_agent2.load_weights_only(self.agent2.get_state_dict())
        self._baseline_agent2.set_eval()
        self._baseline_player2 = create_player(
            agent=self._baseline_agent2,
            config=self.config,
            team_str=self.team2_str,
            collect_data=False,
            deterministic=False,
            max_concurrent=1,
            server_configuration=self.server_config,
        )
        logger.info("Per-team baseline agents frozen for absolute skill evaluation")

        # Save initial best snapshots so both best_team*.pt files exist from
        # the start.  Promotions will overwrite them once agents improve.
        t1_path = self.ckpt_manager.checkpoint_dir / "best_team1.pt"
        t2_path = self.ckpt_manager.checkpoint_dir / "best_team2.pt"
        if not t1_path.exists():
            self.ckpt_manager.save_best_team(
                self.agent1, 0, self.battle_count, 0.5,
            )
        if not t2_path.exists():
            self.ckpt_manager.save_best_team(
                self.agent2, 1, self.battle_count, 0.5,
            )

        # Establish reference WRs: how well each baseline performs vs the other
        await self._update_baseline_reference_wrs()

    async def _update_baseline_reference_wrs(self):
        """Run baseline1 vs baseline2 to establish reference win rates.

        These reference WRs are the bar that current agents must beat (plus
        a margin) to earn a promotion.
        """
        n = self.config.baseline_eval_battles

        # baseline1 (team1) vs baseline2 (team2)
        w1_before = self._baseline_player1.n_won_battles
        t1_before = self._baseline_player1.n_finished_battles
        await self._baseline_player1.battle_against(
            self._baseline_player2, n_battles=n,
        )
        played1 = self._baseline_player1.n_finished_battles - t1_before
        wins1 = self._baseline_player1.n_won_battles - w1_before
        self._baseline_ref_wr1 = wins1 / played1 if played1 > 0 else 0.5
        self._baseline_ref_wr2 = 1.0 - self._baseline_ref_wr1

        logger.info(
            f"Baseline reference WRs: team1={self._baseline_ref_wr1:.1%}, "
            f"team2={self._baseline_ref_wr2:.1%}"
        )

    async def _run_baseline_eval(self) -> Tuple[float, float]:
        """Evaluate both agents against the opposing team's baseline.

        Returns (wr1, wr2):
          - wr1: current agent1 (team1) vs baseline2 (team2)
          - wr2: current agent2 (team2) vs baseline1 (team1)
        """
        n = self.config.baseline_eval_battles
        self.agent1.set_eval()
        self.agent2.set_eval()

        # Agent1 (team1) vs baseline2 (team2)
        p1 = self._get_or_create_eval_player1()
        w1_before, t1_before = p1.n_won_battles, p1.n_finished_battles
        await p1.battle_against(self._baseline_player2, n_battles=n)
        played1 = p1.n_finished_battles - t1_before
        wins1 = p1.n_won_battles - w1_before
        wr1 = wins1 / played1 if played1 > 0 else 0.5

        # Agent2 (team2) vs baseline1 (team1)
        p2 = self._get_or_create_eval_player2()
        w2_before, t2_before = p2.n_won_battles, p2.n_finished_battles
        await p2.battle_against(self._baseline_player1, n_battles=n)
        played2 = p2.n_finished_battles - t2_before
        wins2 = p2.n_won_battles - w2_before
        wr2 = wins2 / played2 if played2 > 0 else 0.5

        self.agent1.set_train()
        self.agent2.set_train()

        self.baseline_eval_results_team1.append((self.battle_count, wr1))
        self.baseline_eval_results_team2.append((self.battle_count, wr2))
        return wr1, wr2

    async def _maybe_promote_baselines(self, wr1: float, wr2: float):
        """Promote baselines when current agents outperform the baseline-vs-baseline reference.

        Instead of requiring an absolute WR (e.g. 55%), promotion happens when
        the current agent beats the opposing baseline by more than the baseline
        itself does, plus a configurable margin.
        """
        margin = self.config.baseline_promotion_margin
        promoted = False

        if wr1 > self._baseline_ref_wr1 + margin:
            self._baseline_agent1.load_weights_only(self.agent1.get_state_dict())
            self._baseline1_promotions += 1
            promoted = True
            self.ckpt_manager.save_best_team(
                self.agent1, 0, self.battle_count, wr1,
            )
            logger.info(
                f"Baseline team1 promoted (WR vs BL2={wr1:.1%}, "
                f"ref={self._baseline_ref_wr1:.1%}, margin={margin:.1%}, "
                f"promotion #{self._baseline1_promotions})"
            )

        if wr2 > self._baseline_ref_wr2 + margin:
            self._baseline_agent2.load_weights_only(self.agent2.get_state_dict())
            self._baseline2_promotions += 1
            promoted = True
            self.ckpt_manager.save_best_team(
                self.agent2, 1, self.battle_count, wr2,
            )
            logger.info(
                f"Baseline team2 promoted (WR vs BL1={wr2:.1%}, "
                f"ref={self._baseline_ref_wr2:.1%}, margin={margin:.1%}, "
                f"promotion #{self._baseline2_promotions})"
            )

        # Seed promoted agents into the league as best-tagged opponents
        if wr1 > self._baseline_ref_wr1 + margin:
            self.league.add_best_agent(self.agent1, team_id=0, win_rate=wr1)
        if wr2 > self._baseline_ref_wr2 + margin:
            self.league.add_best_agent(self.agent2, team_id=1, win_rate=wr2)

        if promoted:
            # Re-evaluate baseline-vs-baseline reference since one or both
            # baselines changed.
            await self._update_baseline_reference_wrs()
            self.ckpt_manager.save_combined_best(
                self.agent1, self.agent2,
                self.league, self.wp_estimator,
                self.battle_count,
            )

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

    def _fresh_start_from_checkpoint(self):
        """Load network weights from checkpoint but reset all training state.

        This keeps the agent's learned policy while resetting exploration
        (entropy annealing), learning rate schedules, and optimizer momentum
        so training can resume as if from scratch.
        """
        # First, do a normal resume to load network weights
        self.resume_if_available()

        cfg = self.config

        # Reset battle count so entropy annealing and LR warmup restart
        self.battle_count = 0

        # Reset optimizers and LR schedulers for both agents
        for agent in (self.agent1, self.agent2):
            agent.battle_optimizer = optim.Adam(
                agent.battle_net.parameters(), lr=cfg.lr
            )
            agent.preview_optimizer = optim.Adam(
                agent.preview_net.parameters(), lr=cfg.lr
            )
            agent.battle_lr_scheduler = agent._create_lr_scheduler(
                agent.battle_optimizer
            )
            agent.preview_lr_scheduler = agent._create_lr_scheduler(
                agent.preview_optimizer
            )
            agent.total_battles = 0
            agent.total_updates = 0
            agent.wins = 0
            agent.losses = 0

        # Reset trainer tracking state
        self.recent_results = []
        self.greedy_eval_results = deque(maxlen=5000)
        self.train_wr_history = deque(maxlen=5000)
        self.metrics_history = deque(maxlen=5000)
        self.baseline_eval_results_team1 = deque(maxlen=5000)
        self.baseline_eval_results_team2 = deque(maxlen=5000)
        self._baseline1_promotions = 0
        self._baseline2_promotions = 0
        self._baseline_ref_wr1 = 0.5
        self._baseline_ref_wr2 = 0.5
        self.best_eval_win_rate = 0.0
        self.regression_counter = 0
        self._entropy_bump_until = 0
        self._latest_explained_variance = 0.0
        self._latest_metrics = {}
        self.plateau_detector = PlateauDetector(
            window=20, patience=10, alpha=0.05, cv_threshold=0.10,
        )

        logger.info(
            "Fresh start from checkpoint: network weights loaded, "
            "schedules and exploration reset"
        )

    async def train(self):
        """Main training loop."""
        if self.config.fresh_start and self.config.resume:
            self._fresh_start_from_checkpoint()
        else:
            self.resume_if_available()

        # Freeze per-team baselines for absolute skill evaluation
        if self.config.baseline_eval_enabled:
            await self._init_baselines()

        logger.info(
            f"Starting training: {self.config.total_battles} battles, "
            f"format={self.config.battle_format}, "
            f"concurrent={self._n_concurrent}"
        )
        logger.info(f"Action space size: {self.config.action_size}")
        logger.info(f"Battle obs size: {BATTLE_OBS_SIZE}, Team preview obs size: {TEAM_PREVIEW_OBS_SIZE}")

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

            # Periodic greedy evaluation (agent1 vs agent2)
            if (self.config.greedy_eval_interval > 0 and
                    self.battle_count % self.config.greedy_eval_interval == 0):
                greedy_wr = await self._run_greedy_eval()
                logger.info(
                    f"  Greedy eval at battle {self.battle_count}: "
                    f"WR={greedy_wr:.1%} ({self.config.greedy_eval_battles} battles)"
                )

                # Best-model tracking: save when we hit a new high
                if self.config.best_model_tracking:
                    if greedy_wr > self.best_eval_win_rate:
                        self.best_eval_win_rate = greedy_wr
                        self.regression_counter = 0
                        self.ckpt_manager.save_best(
                            self.agent1, self.agent2,
                            self.league, self.wp_estimator,
                            self.battle_count, greedy_wr,
                        )
                        logger.info(
                            f"  New best model saved (greedy WR={greedy_wr:.1%}) "
                            f"at battle {self.battle_count}"
                        )
                    elif self.config.regression_rollback_enabled:
                        if greedy_wr < self.best_eval_win_rate - self.config.regression_threshold:
                            self.regression_counter += 1
                            logger.warning(
                                f"Regression detected: greedy WR={greedy_wr:.1%} vs "
                                f"best={self.best_eval_win_rate:.1%} "
                                f"(counter={self.regression_counter}/"
                                f"{self.config.regression_eval_window})"
                            )
                            if self.regression_counter >= self.config.regression_eval_window:
                                logger.warning(
                                    "Rolling back to best model after sustained regression!"
                                )
                                self.ckpt_manager.load_best(
                                    self.agent1, self.agent2,
                                    self.league, self.wp_estimator,
                                )
                                self.regression_counter = 0
                        else:
                            self.regression_counter = 0

                # Per-team baseline evaluation (absolute skill measure)
                if (self.config.baseline_eval_enabled
                        and self._baseline_agent1 is not None):
                    bl_interval = (self.config.baseline_eval_interval
                                   or self.config.greedy_eval_interval)
                    if bl_interval > 0 and self.battle_count % bl_interval == 0:
                        wr1, wr2 = await self._run_baseline_eval()
                        margin = self.config.baseline_promotion_margin
                        logger.info(
                            f"  Baseline eval at battle {self.battle_count}: "
                            f"Team1={wr1:.1%} (ref={self._baseline_ref_wr1:.1%}+{margin:.0%}), "
                            f"Team2={wr2:.1%} (ref={self._baseline_ref_wr2:.1%}+{margin:.0%}) "
                            f"(promotions: {self._baseline1_promotions}/{self._baseline2_promotions})"
                        )
                        await self._maybe_promote_baselines(wr1, wr2)

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
                    self.baseline_eval_results_team1,
                    self.baseline_eval_results_team2,
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

        await training_player.battle_against(opponent_player, n_battles=n_battles)

        # Process each completed battle individually via per-battle state
        ko_w = self.config.ko_reward_weight
        dmg_w = self.config.damage_reward_weight

        tp_episodes = training_player.pop_completed_episodes()
        opp_episodes = (
            opponent_player.pop_completed_episodes() if not use_league else []
        )

        # Build a lookup for opponent episodes by battle_tag so we can
        # pair them with the training player's episodes
        opp_by_tag = {ep.battle_tag: ep for ep in opp_episodes}

        for tp_ep in tp_episodes:
            tp_won = tp_ep.won
            tp_base = 1.0 if tp_won else -1.0

            tp_reward = (
                tp_base
                + ko_w * RLPlayer.get_ko_differential(tp_ep)
                + dmg_w * RLPlayer.get_damage_differential(tp_ep)
            )

            # Apply reward shaping and finalize for training player
            self._apply_reward_shaping(training_player, tp_ep)
            training_player.finalize_episode(tp_ep, tp_reward)

            # If fighting the live opponent, also process their data
            opp_ep = opp_by_tag.get(tp_ep.battle_tag)
            if opp_ep is not None:
                opp_base = -tp_base
                opp_reward = (
                    opp_base
                    + ko_w * RLPlayer.get_ko_differential(opp_ep)
                    + dmg_w * RLPlayer.get_damage_differential(opp_ep)
                )
                self._apply_reward_shaping(opponent_player, opp_ep)
                opponent_player.finalize_episode(opp_ep, opp_reward)

                # Store WP trajectory for live opponent too
                self.wp_estimator.store_trajectory(
                    RLPlayer.get_battle_observations(opp_ep), not tp_won,
                )

            # Store WP trajectory for training player
            self.wp_estimator.store_trajectory(
                RLPlayer.get_battle_observations(tp_ep), tp_won,
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

    def _apply_reward_shaping(self, player: RLPlayer, episode: CompletedEpisode):
        """Apply vectorized win probability reward shaping."""
        observations = RLPlayer.get_battle_observations(episode)
        buffer = player.agent.battle_buffer

        if len(observations) < 2:
            return

        # Batch-predict all WP deltas in one forward pass (proper PBRS with gamma)
        shaped_rewards = self.wp_estimator.compute_shaped_rewards_batch(
            observations, gamma=self.config.gamma
        )

        # Walk backwards to find un-done steps from current episode
        steps_to_shape = []
        for i in range(len(buffer) - 1, -1, -1):
            if buffer.steps[i].done:
                break
            steps_to_shape.append(i)
        steps_to_shape.reverse()

        # Validate alignment between observations, shaped rewards, and buffer steps
        if len(shaped_rewards) == 0:
            logger.debug(
                "Reward shaping produced 0 shaped rewards from %d observations; "
                "skipping shaping for this episode.",
                len(observations),
            )

        if len(steps_to_shape) != len(shaped_rewards):
            logger.warning(
                "Reward shaping mismatch: %d buffer steps vs %d shaped rewards "
                "(from %d observations). Applying what we can.",
                len(steps_to_shape),
                len(shaped_rewards),
                len(observations),
            )

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
                "baseline_wr_team1": (
                    self.baseline_eval_results_team1[-1][1]
                    if self.baseline_eval_results_team1 else None
                ),
                "baseline_wr_team2": (
                    self.baseline_eval_results_team2[-1][1]
                    if self.baseline_eval_results_team2 else None
                ),
                "baseline_promotions_team1": self._baseline1_promotions,
                "baseline_promotions_team2": self._baseline2_promotions,
            },
        )

        # Compute exploitation win rates vs opposing team's best league agent
        exploit_wr1 = self._exploit_win_rate(self.agent1.agent_id, opponent_team=1)
        exploit_wr2 = self._exploit_win_rate(self.agent2.agent_id, opponent_team=0)

        # Admission-gated: only adds if the agent is strong, exploits the
        # best, or is novel with a minimum strength floor
        self.league.add_agent(
            self.agent1, team_id=0, current_win_rate=wr,
            exploit_win_rate=exploit_wr1,
        )
        self.league.add_agent(
            self.agent2, team_id=1, current_win_rate=1 - wr,
            exploit_win_rate=exploit_wr2,
        )

        # Periodic pruning of redundant agents
        self.league.maybe_prune(self.battle_count)

        logger.info(
            f"Checkpoint at battle {self.battle_count}. "
            f"League size: {len(self.league.agents)}"
        )

    def _exploit_win_rate(self, agent_id: str,
                          opponent_team: int) -> Optional[float]:
        """Return the payoff EMA win rate of *agent_id* against the opposing
        team's best-tagged league agent, or ``None`` if no best agent exists."""
        best_agents = [
            a for a in self.league.agents
            if a.team_id == opponent_team and a.is_best
        ]
        if not best_agents:
            return None
        # There should be at most one best per team, but take the latest
        best = best_agents[-1]
        wr = self.league.payoff.get_win_rate(agent_id, best.agent_id)
        # Only return a meaningful value if we have actual matchup data
        _, total = self.league.payoff._records.get(
            (agent_id, best.agent_id), (0.5, 0)
        )
        if total == 0:
            return None
        return wr

    def _log_stats(self):
        """Log training statistics."""
        wr = self._recent_win_rate()
        greedy_str = ""
        if self.greedy_eval_results:
            _, last_greedy_wr = self.greedy_eval_results[-1]
            greedy_str = f" | Greedy WR: {last_greedy_wr:.1%}"
        if self.baseline_eval_results_team1:
            _, bl1 = self.baseline_eval_results_team1[-1]
            _, bl2 = self.baseline_eval_results_team2[-1]
            greedy_str += (
                f" | BL: T1={bl1:.1%} T2={bl2:.1%} "
                f"(promo {self._baseline1_promotions}/{self._baseline2_promotions})"
            )

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
