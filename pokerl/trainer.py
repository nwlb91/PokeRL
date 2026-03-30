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
from pokerl.rnd import RNDExploration
from pokerl.scoreboard import Scoreboard
from pokerl.teamsheet import parse_team
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

        # Parse team sheets for extended observations
        self._team1_moves = None
        self._team2_moves = None
        if config.team_sheet_obs:
            parsed1 = parse_team(self.team1_str)
            parsed2 = parse_team(self.team2_str)
            self._team1_moves = parsed1.get_move_objects(config.gen)
            self._team2_moves = parsed2.get_move_objects(config.gen)
            logger.info(
                "Team sheet observation enabled: team1=%d mons, team2=%d mons",
                len(parsed1.pokemon), len(parsed2.pokemon),
            )

        # Create agents
        self.agent1 = PPOAgent(config, agent_id="team1_main")
        self.agent2 = PPOAgent(config, agent_id="team2_main")

        # League
        self.league = League(config)
        self.league.live_agent_ids = [self.agent1.agent_id, self.agent2.agent_id]

        # Win probability estimator (shared between both agents)
        self.wp_estimator = WinProbabilityEstimator(config)

        # RND exploration (shared between both agents)
        self.rnd: Optional[RNDExploration] = None
        if config.rnd_enabled:
            self.rnd = RNDExploration(config)
            logger.info(
                "RND exploration enabled: coef=%.3f, adaptive_temp=%s, "
                "temp_range=[%.2f, %.2f]",
                config.rnd_coef, config.rnd_adaptive_temp,
                config.rnd_temp_min, config.rnd_temp_max,
            )

        # Observations collected for RND predictor training
        self._rnd_observations: list = []

        # Checkpoint manager
        self.ckpt_manager = CheckpointManager(config)

        # Battle counter
        self.battle_count = 0
        self._batch_counter = 0  # alternates which agent trains vs league

        # Last battle count at which each periodic action ran.
        # Using "last fired" instead of "% N == 0" so that large batch
        # sizes (num_parallel_battles > 1) don't cause intervals to be skipped.
        self._last_checkpoint = 0
        self._last_greedy_eval = 0
        self._last_baseline_eval = 0
        self._last_stats_log = 0

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
        self._league_opponent_team_id: Optional[int] = None

        # Elo scoreboards (one per team)
        self.scoreboard_team1 = Scoreboard(
            team_id=0,
            checkpoint_dir=config.checkpoint_dir,
            eval_interval=config.elo_eval_interval,
            eval_games=config.elo_eval_games,
        )
        self.scoreboard_team2 = Scoreboard(
            team_id=1,
            checkpoint_dir=config.checkpoint_dir,
            eval_interval=config.elo_eval_interval,
            eval_games=config.elo_eval_games,
        )
        self._last_elo_eval = 0

        # Dedicated eval players for Elo scoreboard matches
        self._elo_opponent_agent: Optional[PPOAgent] = None
        self._elo_opponent_player: Optional[RLPlayer] = None
        self._elo_opponent_team_id: Optional[int] = None

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

        # Per-team EMA win rates for matchup-aware reward scaling
        self._team1_wr_ema: float = 0.5
        self._team2_wr_ema: float = 0.5

        # Best-model tracking and regression protection
        self.best_eval_win_rate: float = 0.0
        self.regression_counter: int = 0

        # Plateau response: entropy bump (linear decay from start to until)
        self._entropy_bump_start: int = 0
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

    def _setup_player(self, player: RLPlayer, team_id: int):
        """Attach shared resources (RND, team sheets) to a player."""
        if self.rnd is not None:
            player.rnd = self.rnd
        if self._team1_moves is not None:
            if team_id == 0:
                player.our_team_moves = self._team1_moves
                player.opp_team_moves = self._team2_moves
            else:
                player.our_team_moves = self._team2_moves
                player.opp_team_moves = self._team1_moves

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
            self._setup_player(self._player1, team_id=0)
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
            self._setup_player(self._player2, team_id=1)
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
            self._setup_player(self._eval_player1, team_id=0)
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
            self._setup_player(self._eval_player2, team_id=1)
        return self._eval_player2

    async def _async_close_player(self, player: "Optional[RLPlayer]") -> None:
        """Gracefully close a player's WebSocket before dropping it.

        poke-env Player objects hold a PSClient with an open WebSocket.
        Simply dereferencing the player leaks the connection.  This helper
        attempts to close it cleanly so zombie connections don't accumulate
        on the Showdown server.
        """
        if player is None:
            return
        try:
            ps_client = getattr(player, "ps_client", None)
            if ps_client is None:
                return
            # Use the internal async version directly since we're already
            # in the event loop (the public stop_listening() wraps this in
            # handle_threaded_coroutines which can deadlock here).
            stop = getattr(ps_client, "_stop_listening", None)
            if stop is not None:
                await stop()
            else:
                # Fallback: close the websocket directly
                websocket = getattr(ps_client, "websocket", None)
                if websocket is not None:
                    await websocket.close()
        except Exception:
            logger.debug("Failed to close player WebSocket cleanly", exc_info=True)

    async def _cleanup_players(self) -> None:
        """Close all player WebSocket connections on shutdown."""
        for player in [
            self._player1, self._player2,
            self._eval_player1, self._eval_player2,
            self._league_opponent_player,
            self._elo_opponent_player,
            self._baseline_player1, self._baseline_player2,
        ]:
            await self._async_close_player(player)

    async def _get_league_player(self, league_agent, team_id: int) -> RLPlayer:
        """Get or create a frozen player for a league opponent.

        Reuses the player if possible, only reloading weights when the
        league agent changes.
        """
        from pokerl.league import LeagueAgent

        if (self._league_opponent_player is not None
                and self._league_opponent_id == league_agent.agent_id
                and self._league_opponent_team_id == team_id):
            return self._league_opponent_player

        # Create or reuse the agent shell
        if self._league_opponent_agent is None:
            self._league_opponent_agent = PPOAgent(self.config, agent_id=league_agent.agent_id)

        self._league_opponent_agent.agent_id = league_agent.agent_id
        self._league_opponent_agent.load_weights_only(league_agent.state_dict)

        # Pick the right team string
        team_str = self.team1_str if team_id == 0 else self.team2_str

        # Close old player's WebSocket before creating a new one
        await self._async_close_player(self._league_opponent_player)

        # Always recreate the player when agent or team changes to ensure
        # the correct team is used (ConstantTeambuilder is set at creation)
        self._league_opponent_player = create_player(
            agent=self._league_opponent_agent,
            config=self.config,
            team_str=team_str,
            collect_data=False,
            deterministic=False,
            max_concurrent=self._n_concurrent,
            server_configuration=self.server_config,
        )
        self._setup_player(self._league_opponent_player, team_id=team_id)

        self._league_opponent_id = league_agent.agent_id
        self._league_opponent_team_id = team_id
        return self._league_opponent_player

    def _get_effective_rnd_coef(self) -> float:
        """Compute the RND intrinsic reward coefficient with optional annealing."""
        cfg = self.config
        if not cfg.rnd_enabled:
            return 0.0
        if cfg.rnd_anneal_battles > 0:
            progress = min(1.0, self.battle_count / cfg.rnd_anneal_battles)
            return cfg.rnd_coef + progress * (cfg.rnd_coef_end - cfg.rnd_coef)
        return cfg.rnd_coef

    def _get_effective_entropy_coef(self) -> float:
        """Compute the entropy coefficient with annealing and plateau bump."""
        cfg = self.config
        if cfg.entropy_anneal_battles > 0:
            progress = min(1.0, self.battle_count / cfg.entropy_anneal_battles)
            coef = cfg.entropy_coef_start + progress * (cfg.entropy_coef_end - cfg.entropy_coef_start)
        else:
            coef = cfg.entropy_coef
        # Plateau bump — linear decay to avoid discontinuity when bump expires
        if (self._entropy_bump_start < self._entropy_bump_until
                and self.battle_count < self._entropy_bump_until):
            bump_duration = self._entropy_bump_until - self._entropy_bump_start
            bump_remaining = self._entropy_bump_until - self.battle_count
            coef += cfg.plateau_entropy_bump * (bump_remaining / bump_duration)
        return coef

    async def _run_greedy_eval(self) -> Optional[float]:
        """Run evaluation battles between the two main agents.

        Agents sample from their policy distributions (stochastic) to
        match real play conditions.  Returns eval win rate for agent1,
        or None if no battles completed.
        """
        player1 = self._get_or_create_eval_player1()
        player2 = self._get_or_create_eval_player2()

        self.agent1.set_eval()
        self.agent2.set_eval()

        wins_before = player1.n_won_battles
        total_before = player1.n_finished_battles

        n = self.config.greedy_eval_battles
        try:
            await asyncio.wait_for(
                player1.battle_against(player2, n_battles=n),
                timeout=self.config.battle_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Greedy eval timed out after {self.config.battle_timeout}s"
            )

        wins_after = player1.n_won_battles
        total_after = player1.n_finished_battles
        played = total_after - total_before
        wins = wins_after - wins_before

        self.agent1.set_train()
        self.agent2.set_train()

        if played == 0:
            logger.warning("Greedy eval: no battles completed")
            return None
        wr = wins / played
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
        self._setup_player(self._baseline_player1, team_id=0)

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
        self._setup_player(self._baseline_player2, team_id=1)
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
        try:
            await asyncio.wait_for(
                self._baseline_player1.battle_against(
                    self._baseline_player2, n_battles=n,
                ),
                timeout=self.config.battle_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Baseline reference eval timed out after {self.config.battle_timeout}s"
            )
        played1 = self._baseline_player1.n_finished_battles - t1_before
        wins1 = self._baseline_player1.n_won_battles - w1_before
        if played1 > 0:
            self._baseline_ref_wr1 = wins1 / played1
            self._baseline_ref_wr2 = 1.0 - self._baseline_ref_wr1
        else:
            logger.warning("Baseline reference: no battles completed, keeping defaults")

        logger.info(
            f"Baseline reference WRs: team1={self._baseline_ref_wr1:.1%}, "
            f"team2={self._baseline_ref_wr2:.1%}"
        )

    async def _run_baseline_eval(self) -> Tuple[Optional[float], Optional[float]]:
        """Evaluate both agents against the opposing team's baseline.

        Returns (wr1, wr2):
          - wr1: current agent1 (team1) vs baseline2 (team2), or None
          - wr2: current agent2 (team2) vs baseline1 (team1), or None
        """
        n = self.config.baseline_eval_battles
        self.agent1.set_eval()
        self.agent2.set_eval()

        # Agent1 (team1) vs baseline2 (team2)
        p1 = self._get_or_create_eval_player1()
        w1_before, t1_before = p1.n_won_battles, p1.n_finished_battles
        try:
            await asyncio.wait_for(
                p1.battle_against(self._baseline_player2, n_battles=n),
                timeout=self.config.battle_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(f"Baseline eval (team1) timed out after {self.config.battle_timeout}s")
        played1 = p1.n_finished_battles - t1_before
        wins1 = p1.n_won_battles - w1_before
        wr1 = wins1 / played1 if played1 > 0 else None

        # Agent2 (team2) vs baseline1 (team1)
        p2 = self._get_or_create_eval_player2()
        w2_before, t2_before = p2.n_won_battles, p2.n_finished_battles
        try:
            await asyncio.wait_for(
                p2.battle_against(self._baseline_player1, n_battles=n),
                timeout=self.config.battle_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(f"Baseline eval (team2) timed out after {self.config.battle_timeout}s")
        played2 = p2.n_finished_battles - t2_before
        wins2 = p2.n_won_battles - w2_before
        wr2 = wins2 / played2 if played2 > 0 else None

        self.agent1.set_train()
        self.agent2.set_train()

        if wr1 is not None:
            self.baseline_eval_results_team1.append((self.battle_count, wr1))
        if wr2 is not None:
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

    async def _get_elo_opponent_player(self, state_dict: dict,
                                       team_id: int) -> RLPlayer:
        """Get or create an eval player for Elo scoreboard matches.

        Loads *state_dict* weights into a reusable agent/player pair.
        Recreates the player when the team side changes.
        """
        if self._elo_opponent_agent is None:
            self._elo_opponent_agent = PPOAgent(
                self.config, agent_id="elo_eval_opponent"
            )

        self._elo_opponent_agent.load_weights_only(state_dict)
        self._elo_opponent_agent.set_eval()

        if (self._elo_opponent_player is not None
                and self._elo_opponent_team_id == team_id):
            return self._elo_opponent_player

        team_str = self.team1_str if team_id == 0 else self.team2_str
        await self._async_close_player(self._elo_opponent_player)
        self._elo_opponent_player = create_player(
            agent=self._elo_opponent_agent,
            config=self.config,
            team_str=team_str,
            collect_data=False,
            deterministic=False,
            max_concurrent=self._n_concurrent,
            server_configuration=self.server_config,
        )
        self._setup_player(self._elo_opponent_player, team_id=team_id)
        self._elo_opponent_team_id = team_id
        return self._elo_opponent_player

    async def _run_elo_evaluation(self):
        """Cross-evaluate all team1 scoreboard entries vs all team2 entries.

        Each (team1_entry, team2_entry) pair plays n_games.  Results are
        recorded in both scoreboards (inverted for team2).  Only pairs
        that haven't yet accumulated enough games are played, so existing
        match data from prior evaluations is reused.
        """
        import torch as _torch

        n_games = self.config.elo_eval_games
        elo_ckpt_name = f"ckpt_{self.battle_count:06d}"
        elo_ckpt_path = Path(self.config.checkpoint_dir) / f"{elo_ckpt_name}.pt"
        elo_state = {
            "agent1": self.agent1.get_state_dict(),
            "agent2": self.agent2.get_state_dict(),
        }
        _torch.save(elo_state, elo_ckpt_path)

        # Save current weights so we can restore after swapping
        saved_agent1_state = self.agent1.get_state_dict()
        saved_agent2_state = self.agent2.get_state_dict()

        self.agent1.set_eval()
        self.agent2.set_eval()

        # Build candidate lists: existing leaderboard entries + current checkpoint
        t1_candidates = [(e.name, e.checkpoint_path) for e in self.scoreboard_team1.entries]
        t1_candidates.append((elo_ckpt_name, str(elo_ckpt_path)))

        t2_candidates = [(e.name, e.checkpoint_path) for e in self.scoreboard_team2.entries]
        t2_candidates.append((elo_ckpt_name, str(elo_ckpt_path)))

        # Cache loaded checkpoints to avoid redundant disk reads
        ckpt_cache: dict = {}

        def _load_ckpt(path: str) -> dict:
            if path not in ckpt_cache:
                ckpt_cache[path] = _torch.load(
                    path, map_location="cpu", weights_only=False,
                )
            return ckpt_cache[path]

        # Play every (team1, team2) pair, recording in both scoreboards
        for t1_name, t1_path in t1_candidates:
            if not t1_path:
                continue
            for t2_name, t2_path in t2_candidates:
                if not t2_path:
                    continue

                # Prefix opponent names so the BT model distinguishes
                # team1's "initial" (agent1) from team2's "initial" (agent2).
                t2_opp_name = f"opp:{t2_name}"
                t1_opp_name = f"opp:{t1_name}"

                # Skip pairs that already have enough match data
                t1_key = (t1_name, t2_opp_name)
                t1_played = sum(self.scoreboard_team1.matches.get(t1_key, [0, 0]))
                t2_key = (t2_name, t1_opp_name)
                t2_played = sum(self.scoreboard_team2.matches.get(t2_key, [0, 0]))
                already_played = max(t1_played, t2_played)
                if already_played >= n_games:
                    continue

                games_needed = n_games - already_played

                # Load team1 entry's agent1 weights into eval player
                t1_ckpt = _load_ckpt(t1_path)
                agent1_state = t1_ckpt.get("agent1", t1_ckpt.get("agent", {}))
                self.agent1.load_weights_only(agent1_state)
                self.agent1.set_eval()

                # Load team2 entry's agent2 weights into opponent player
                t2_ckpt = _load_ckpt(t2_path)
                agent2_state = t2_ckpt.get("agent2", t2_ckpt.get("agent", {}))
                opp_player = await self._get_elo_opponent_player(
                    agent2_state, team_id=1,
                )

                eval_player1 = self._get_or_create_eval_player1()
                wins_before = eval_player1.n_won_battles
                total_before = eval_player1.n_finished_battles

                try:
                    await asyncio.wait_for(
                        eval_player1.battle_against(
                            opp_player, n_battles=games_needed,
                        ),
                        timeout=self.config.battle_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Elo eval timed out: %s vs %s", t1_name, t2_name,
                    )

                total_played = eval_player1.n_finished_battles - total_before
                wins = eval_player1.n_won_battles - wins_before
                losses = total_played - wins

                # Record in team1 scoreboard (team1 player vs team2 opponent)
                for _ in range(wins):
                    self.scoreboard_team1.record_match(
                        t1_name, t2_opp_name, a_won=True,
                    )
                for _ in range(losses):
                    self.scoreboard_team1.record_match(
                        t1_name, t2_opp_name, a_won=False,
                    )

                # Record in team2 scoreboard (team2 player vs team1 opponent)
                for _ in range(losses):
                    self.scoreboard_team2.record_match(
                        t2_name, t1_opp_name, a_won=True,
                    )
                for _ in range(wins):
                    self.scoreboard_team2.record_match(
                        t2_name, t1_opp_name, a_won=False,
                    )

                logger.info(
                    "Elo eval: '%s' (t1) vs '%s' (t2) = %d/%d t1 wins",
                    t1_name, t2_name, wins, total_played,
                )

        # Restore original weights
        self.agent1.load_weights_only(saved_agent1_state)
        self.agent2.load_weights_only(saved_agent2_state)
        self.agent1.set_train()
        self.agent2.set_train()

        # Compute ratings and decide whether to add to leaderboards
        for scoreboard, team_label in [
            (self.scoreboard_team1, "team1"),
            (self.scoreboard_team2, "team2"),
        ]:
            rating = scoreboard.get_current_rating(elo_ckpt_name)
            logger.info(
                "Elo %s: current rating=%.1f (leaderboard: %s)",
                team_label, rating,
                ", ".join(
                    f"{e.name}={e.rating:.1f}" for e in scoreboard.entries
                ),
            )

            if scoreboard.should_add(rating):
                scoreboard.add_entry(
                    name=elo_ckpt_name,
                    battle_count=self.battle_count,
                    checkpoint_path=str(elo_ckpt_path),
                )
                logger.info(
                    "Elo %s: '%s' added to leaderboard (rating=%.1f)",
                    team_label, elo_ckpt_name, rating,
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
            # Restore per-team EMA win rates from checkpoint metadata
            meta = self.ckpt_manager.get_latest_metadata()
            if meta is not None:
                self._team1_wr_ema = meta.get("team1_wr_ema", 0.5)
                self._team2_wr_ema = meta.get("team2_wr_ema", 0.5)

            # Restore RND state if available
            if self.rnd is not None:
                import torch as _torch
                rnd_path = Path(self.config.checkpoint_dir) / "rnd_state.pt"
                if rnd_path.exists():
                    rnd_state = _torch.load(rnd_path, map_location="cpu", weights_only=False)
                    self.rnd.load_state_dict(rnd_state)
                    logger.info("Restored RND state from checkpoint")

            # Restore Elo eval counter
            if meta is not None:
                self._last_elo_eval = meta.get("last_elo_eval", 0)

            logger.info(f"Resumed from battle {self.battle_count}")

        # Reconstruct scoreboards from JSONL logs (works for both resume and fresh)
        loaded_t1 = self.scoreboard_team1.load_from_log()
        loaded_t2 = self.scoreboard_team2.load_from_log()
        if loaded_t1:
            logger.info("Scoreboard team1 restored from log")
        if loaded_t2:
            logger.info("Scoreboard team2 restored from log")

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
        self._entropy_bump_start = 0
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

        # Initialize Elo scoreboards if no entries loaded from log
        if not self.scoreboard_team1.entries:
            import torch as _torch
            initial_path = Path(self.config.checkpoint_dir) / "initial_elo.pt"
            if not initial_path.exists():
                _torch.save(
                    {"agent1": self.agent1.get_state_dict(),
                     "agent2": self.agent2.get_state_dict()},
                    initial_path,
                )
            self.scoreboard_team1.add_initial(
                "initial", self.battle_count, str(initial_path)
            )
            self.scoreboard_team2.add_initial(
                "initial", self.battle_count, str(initial_path)
            )

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

            # PPO updates when each agent's own buffer is full enough
            # (updating agents independently avoids overfitting on tiny buffers)
            updated_battle = False
            if len(self.agent1.battle_buffer) >= self.config.rollout_steps:
                self._update_agent_battle(self.agent1, "Agent1")
                updated_battle = True
            if len(self.agent2.battle_buffer) >= self.config.rollout_steps:
                self._update_agent_battle(self.agent2, "Agent2")
                updated_battle = True
            if updated_battle:
                self._post_battle_update()

            # Team preview updates on a separate (lower) threshold so lead
            # selection learns at a comparable rate despite producing only
            # one sample per game.
            if len(self.agent1.preview_buffer) >= self.config.preview_rollout_steps:
                self._update_agent_preview(self.agent1)
            if len(self.agent2.preview_buffer) >= self.config.preview_rollout_steps:
                self._update_agent_preview(self.agent2)

            # Periodic checkpoint + league snapshot
            if self.battle_count - self._last_checkpoint >= self.config.checkpoint_interval:
                self._checkpoint_and_snapshot()
                self._last_checkpoint = self.battle_count

            # Win probability estimator update
            wp_metrics = self.wp_estimator.maybe_update()
            if wp_metrics:
                logger.debug(
                    f"  WP update: loss={wp_metrics.get('wp_loss', 0):.4f}, "
                    f"mean_pred={wp_metrics.get('wp_mean_pred', 0):.3f}"
                )

            # Periodic greedy evaluation (agent1 vs agent2)
            if (self.config.greedy_eval_interval > 0 and
                    self.battle_count - self._last_greedy_eval >= self.config.greedy_eval_interval):
                greedy_wr = await self._run_greedy_eval()
                if greedy_wr is None:
                    logger.warning("Skipping eval processing — no battles completed")
                else:
                    logger.info(
                        f"  Greedy eval at battle {self.battle_count}: "
                        f"WR={greedy_wr:.1%} ({self.config.greedy_eval_battles} battles)"
                    )

                # Best-model tracking: save when we hit a new high
                if greedy_wr is not None and self.config.best_model_tracking:
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

                self._last_greedy_eval = self.battle_count

                # Per-team baseline evaluation (absolute skill measure)
                if (self.config.baseline_eval_enabled
                        and self._baseline_agent1 is not None):
                    bl_interval = (self.config.baseline_eval_interval
                                   or self.config.greedy_eval_interval)
                    if (bl_interval > 0 and
                            self.battle_count - self._last_baseline_eval >= bl_interval):
                        wr1, wr2 = await self._run_baseline_eval()
                        self._last_baseline_eval = self.battle_count
                        if wr1 is not None and wr2 is not None:
                            margin = self.config.baseline_promotion_margin
                            logger.info(
                                f"  Baseline eval at battle {self.battle_count}: "
                                f"Team1={wr1:.1%} (ref={self._baseline_ref_wr1:.1%}+{margin:.0%}), "
                                f"Team2={wr2:.1%} (ref={self._baseline_ref_wr2:.1%}+{margin:.0%}) "
                                f"(promotions: {self._baseline1_promotions}/{self._baseline2_promotions})"
                            )
                            await self._maybe_promote_baselines(wr1, wr2)
                        else:
                            logger.warning("Baseline eval: some battles did not complete, skipping")

            # Periodic Elo scoreboard evaluation
            if (self.config.elo_eval_interval > 0 and
                    self.battle_count - self._last_elo_eval >= self.config.elo_eval_interval):
                await self._run_elo_evaluation()
                self._last_elo_eval = self.battle_count

            # Periodic logging + plateau detection
            if self.battle_count - self._last_stats_log >= 50:
                self._last_stats_log = self.battle_count
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
                        self._entropy_bump_start = self.battle_count
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
                    league_size=len(self.league.agents),
                    agent1_wins=self.agent1.wins,
                    agent1_losses=self.agent1.losses,
                    agent2_wins=self.agent2.wins,
                    agent2_losses=self.agent2.losses,
                    elo_leaderboard_team1=self.scoreboard_team1.get_state_for_dashboard(),
                    elo_leaderboard_team2=self.scoreboard_team2.get_state_for_dashboard(),
                )

        # Final checkpoint
        self._checkpoint_and_snapshot()
        await self._cleanup_players()
        logger.info("Training complete!")

    async def _run_battle_batch(self, n_battles: int):
        """Run n_battles concurrently using poke-env's built-in concurrency.

        Selects opponents from the league with probability pfsp_fraction,
        otherwise uses the live opponent.  When a league opponent is
        selected, only the training player collects data.
        """
        # Decide which training player and opponent to use this batch.
        # Alternate which agent gets league exposure each batch.
        # Uses a dedicated counter (not battle_count) because batch_size > 1
        # can make battle_count skip odd values entirely.
        self._batch_counter += 1
        if self._batch_counter % 2 == 0:
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

        # Try to select a frozen league opponent (pfsp_fraction of the time)
        if self.league.agents:
            result = self.league.select_opponent(
                training_agent.agent_id, training_team_id,
            )
            if result is not None:
                league_agent, kind = result
                opponent_team_id = league_agent.team_id
                opponent_player = await self._get_league_player(
                    league_agent, opponent_team_id,
                )
                opponent_agent_id = league_agent.agent_id
                use_league = True
                logger.debug(
                    f"League opponent: {league_agent.agent_id} (pfsp)"
                )

        if not use_league:
            opponent_player = live_opponent_fn()
            opponent_agent_id = live_agent.agent_id

        # Set matchup context (team WR EMA) for conditioned value head
        if self.config.matchup_conditioned_value:
            tp_wr = self._team1_wr_ema if training_team_id == 0 else self._team2_wr_ema
            training_player.matchup_context = np.array([tp_wr], dtype=np.float32)
            if not use_league:
                opp_wr = self._team2_wr_ema if training_team_id == 0 else self._team1_wr_ema
                opponent_player.matchup_context = np.array([opp_wr], dtype=np.float32)

        try:
            await asyncio.wait_for(
                training_player.battle_against(opponent_player, n_battles=n_battles),
                timeout=self.config.battle_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Battle timed out after {self.config.battle_timeout}s "
                f"at battle {self.battle_count}. Processing any completed episodes."
            )
            # Close and reset league player to avoid stale challenge state
            if use_league:
                await self._async_close_player(self._league_opponent_player)
                self._league_opponent_player = None
                self._league_opponent_id = None
                self._league_opponent_team_id = None

        # Process each completed battle individually via per-battle state
        tp_episodes = training_player.pop_completed_episodes()
        opp_episodes = (
            opponent_player.pop_completed_episodes() if not use_league else []
        )

        # Build a lookup for opponent episodes by battle_tag so we can
        # pair them with the training player's episodes
        opp_by_tag = {ep.battle_tag: ep for ep in opp_episodes}

        # Per-team EMA win rates for matchup-aware reward scaling
        tp_wr_ema = self._team1_wr_ema if training_team_id == 0 else self._team2_wr_ema
        opp_wr_ema = self._team2_wr_ema if training_team_id == 0 else self._team1_wr_ema

        for tp_ep in tp_episodes:
            tp_won = tp_ep.won

            tp_reward = self._compute_scaled_reward(
                tp_won,
                RLPlayer.get_ko_differential(tp_ep),
                RLPlayer.get_damage_differential(tp_ep),
                tp_wr_ema,
            )

            # Apply reward shaping and finalize for training player
            self._apply_reward_shaping(training_player, tp_ep)
            training_player.finalize_episode(tp_ep, tp_reward)

            # If fighting the live opponent, also process their data
            opp_ep = opp_by_tag.get(tp_ep.battle_tag)
            if opp_ep is not None:
                opp_reward = self._compute_scaled_reward(
                    not tp_won,
                    RLPlayer.get_ko_differential(opp_ep),
                    RLPlayer.get_damage_differential(opp_ep),
                    opp_wr_ema,
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
            self.league.win_loss.record_result(
                training_agent.agent_id, opponent_agent_id, tp_won,
            )

            # Track win rate from team1's perspective
            if training_team_id == 0:
                self.recent_results.append(tp_won)
            else:
                self.recent_results.append(not tp_won)

            # Update per-team EMA win rates for matchup-aware scaling.
            # Always update both teams (even during league matches where
            # opp_ep is None) so EMAs don't become stale.
            alpha = self.config.reward_wr_ema_alpha
            if training_team_id == 0:
                self._team1_wr_ema += alpha * (float(tp_won) - self._team1_wr_ema)
                self._team2_wr_ema += alpha * (float(not tp_won) - self._team2_wr_ema)
            else:
                self._team2_wr_ema += alpha * (float(tp_won) - self._team2_wr_ema)
                self._team1_wr_ema += alpha * (float(not tp_won) - self._team1_wr_ema)

        if len(self.recent_results) > 100:
            self.recent_results = self.recent_results[-100:]

    def _compute_scaled_reward(
        self, won: bool, ko_diff: float, dmg_diff: float, team_wr_ema: float
    ) -> float:
        """Compute terminal reward with matchup-aware scaling.

        When ``matchup_reward_scaling`` is enabled, the base win/loss reward is
        scaled by the team's running EMA win rate so that expected losses are
        dampened and rare wins amplified.  KO/damage weights are also boosted
        for underdog teams (WR below ``underdog_wr_threshold``).
        """
        if self.config.matchup_reward_scaling:
            wr = max(team_wr_ema, 0.01)  # floor to avoid near-zero issues
            if won:
                base = 1.0 + (1.0 - wr)  # rare wins amplified
            else:
                base = -wr  # expected losses dampened
        else:
            base = 1.0 if won else -1.0

        ko_w = self.config.ko_reward_weight
        dmg_w = self.config.damage_reward_weight

        # Boost KO/damage weights for underdog teams
        if (self.config.matchup_reward_scaling
                and team_wr_ema < self.config.underdog_wr_threshold):
            t = self.config.underdog_wr_threshold
            boost = 1.0 + self.config.underdog_shaping_boost * (t - team_wr_ema) / t
            ko_w *= boost
            dmg_w *= boost

        return base + ko_w * ko_diff + dmg_w * dmg_diff

    def _apply_reward_shaping(self, player: RLPlayer, episode: CompletedEpisode):
        """Apply vectorized win probability reward shaping.

        Operates on the episode's locally-buffered pending_steps (not the
        shared RolloutBuffer) so concurrent battles don't interfere.
        """
        observations = RLPlayer.get_battle_observations(episode)
        pending = episode.state.pending_steps

        if len(observations) < 2:
            return

        # Batch-predict all WP deltas in one forward pass (proper PBRS with gamma)
        shaped_rewards = self.wp_estimator.compute_shaped_rewards_batch(
            observations, gamma=self.config.gamma
        )

        if len(shaped_rewards) == 0:
            logger.debug(
                "Reward shaping produced 0 shaped rewards from %d observations; "
                "skipping shaping for this episode.",
                len(observations),
            )

        if len(pending) != len(shaped_rewards):
            logger.warning(
                "Reward shaping mismatch: %d pending steps vs %d shaped rewards "
                "(from %d observations). Applying what we can.",
                len(pending),
                len(shaped_rewards),
                len(observations),
            )

        # Apply shaped rewards with survival bonus so the losing side
        # receives a small positive per-turn signal even when WP deltas
        # are near zero (prevents gradient starvation in 100-0 matchups).
        surv = self.config.survival_reward_per_turn
        for j, step in enumerate(pending):
            if j < len(shaped_rewards):
                step.reward = shaped_rewards[j] + surv
            else:
                step.reward = surv

        # Compute terminal PBRS correction: -phi(s_{T-1}) for proper telescoping.
        # The terminal step (created in finalize_episode) needs this so the
        # potential-based shaping sums telescope correctly over the episode.
        if len(observations) >= 1:
            last_obs = np.array([observations[-1]], dtype=np.float32)
            last_wp = self.wp_estimator.predict_batch(last_obs)[0]
            episode.state.terminal_pbrs_correction = (
                -self.config.wp_reward_weight * float(last_wp)
            )

        # Add RND intrinsic novelty rewards
        if self.rnd is not None and len(observations) > 0:
            intrinsic = self.rnd.compute_intrinsic_rewards(observations)
            rnd_coef = self._get_effective_rnd_coef()
            # intrinsic has one value per observation; pending_steps[i] corresponds
            # to the transition from obs[i] to obs[i+1]. Assign intrinsic[i] to step[i].
            for j, step in enumerate(pending):
                if j < len(intrinsic):
                    step.reward += rnd_coef * intrinsic[j]

            # Collect observations for RND predictor training
            self._rnd_observations.extend(observations)

    def _inject_param_noise(self):
        """Inject small Gaussian noise into policy parameters to escape plateaus."""
        import torch
        noise_scale = 0.01
        for agent in (self.agent1, self.agent2):
            for param in agent.battle_net.parameters():
                if param.requires_grad:
                    param.data.add_(torch.randn_like(param.data) * noise_scale)

    def _update_agent_battle(self, agent: "PPOAgent", label: str):
        """Run PPO battle update for a single agent."""
        agent.set_train()
        ent_coef = self._get_effective_entropy_coef()
        metrics = agent.update_battle_policy(entropy_coef_override=ent_coef)

        # Step LR scheduler
        wr = self._recent_win_rate()
        if agent is self.agent1:
            agent.step_lr_scheduler(metric=wr)
        else:
            agent.step_lr_scheduler(metric=1.0 - wr)

        if metrics:
            if agent is self.agent1:
                self._latest_explained_variance = metrics.get(
                    'explained_variance', 0.0
                )
                self._latest_metrics = {
                    'policy_loss': metrics.get('policy_loss', 0.0),
                    'value_loss': metrics.get('value_loss', 0.0),
                    'entropy': metrics.get('entropy', 0.0),
                    'explained_variance': metrics.get('explained_variance', 0.0),
                    'mean_episode_return': metrics.get('mean_episode_return', 0.0),
                }
            msg = (
                f"  {label} battle update: "
                f"policy_loss={metrics.get('policy_loss', 0):.4f}, "
                f"value_loss={metrics.get('value_loss', 0):.4f}, "
                f"entropy={metrics.get('entropy', 0):.4f}, "
                f"explained_var={metrics.get('explained_variance', 0):.4f}, "
                f"mean_ep_return={metrics.get('mean_episode_return', 0):.4f}"
            )
            if 'mean_uncertainty' in metrics:
                msg += f", uncertainty={metrics['mean_uncertainty']:.4f}"
            logger.debug(msg)

    def _post_battle_update(self):
        """Run post-battle-update tasks (RND training)."""
        if self.rnd is not None and len(self._rnd_observations) > 0:
            rnd_loss = self.rnd.train_predictor(self._rnd_observations)
            rnd_coef = self._get_effective_rnd_coef()
            self._latest_metrics['rnd_loss'] = rnd_loss
            self._latest_metrics['rnd_coef'] = rnd_coef
            self._latest_metrics['rnd_reward_mean'] = self.rnd.reward_stats.mean
            self._latest_metrics['rnd_novelty_mean'] = self.rnd.novelty_stats.mean
            logger.debug(
                "  RND update: loss=%.4f, coef=%.4f, reward_mean=%.4f, "
                "novelty_mean=%.4f",
                rnd_loss, rnd_coef,
                self.rnd.reward_stats.mean, self.rnd.novelty_stats.mean,
            )
            self._rnd_observations.clear()

    def _update_agent_preview(self, agent: "PPOAgent"):
        """Run PPO preview update for a single agent."""
        ent_coef = self._get_effective_entropy_coef()
        agent.update_preview_policy(
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
                "team1_wr_ema": self._team1_wr_ema,
                "team2_wr_ema": self._team2_wr_ema,
                "last_elo_eval": self._last_elo_eval,
            },
        )

        # Save RND state alongside checkpoint
        if self.rnd is not None:
            import torch as _torch
            rnd_path = Path(self.config.checkpoint_dir) / "rnd_state.pt"
            _torch.save(self.rnd.state_dict(), rnd_path)

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

        # Matchup-aware reward scaling info
        matchup_str = ""
        if self.config.matchup_reward_scaling:
            matchup_str = (
                f" | WR EMA: T1={self._team1_wr_ema:.3f} T2={self._team2_wr_ema:.3f}"
            )

        logger.info(
            f"Battle {self.battle_count}/{self.config.total_battles} | "
            f"Team1 recent WR: {wr:.1%}{greedy_str}{matchup_str} | "
            f"Agent1: {self.agent1.wins}W/{self.agent1.losses}L | "
            f"Agent2: {self.agent2.wins}W/{self.agent2.losses}L | "
            f"League: {len(self.league.agents)} agents"
        )

    def _recent_win_rate(self) -> float:
        if not self.recent_results:
            return 0.5
        return sum(self.recent_results) / len(self.recent_results)
