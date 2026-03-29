"""Checkpoint management for training resumption.

Handles saving and loading of:
  - Agent states (battle + preview networks, optimizers, stats)
  - Win probability estimator state
  - League state (frozen agents, payoff matrix)
  - Training progress metadata
"""

import logging
from pathlib import Path
from typing import Optional

import torch

from pokerl.agent import PPOAgent
from pokerl.config import Config
from pokerl.league import League
from pokerl.win_probability import WinProbabilityEstimator

logger = logging.getLogger(__name__)



class CheckpointManager:
    """Manages periodic saving and loading of training state."""

    def __init__(self, config: Config):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_latest_metadata(self) -> Optional[dict]:
        """Return the metadata dict from the latest checkpoint, or None."""
        latest_path = self.checkpoint_dir / "latest.pt"
        if not latest_path.exists():
            return None
        try:
            state = torch.load(latest_path, map_location="cpu", weights_only=False)
            return state.get("metadata")
        except Exception:
            return None

    def save(
        self,
        agent1: PPOAgent,
        agent2: PPOAgent,
        league: League,
        wp_estimator: WinProbabilityEstimator,
        battle_count: int,
        extra_metadata: Optional[dict] = None,
    ):
        """Save a full training checkpoint.

        Creates:
          - checkpoints/latest.pt (overwritten each time)
          - checkpoints/checkpoint_{battle_count}.pt (periodic)
        """
        state = {
            "agent1": agent1.get_state_dict(),
            "agent2": agent2.get_state_dict(),
            "league": league.get_state_dict(),
            "wp_estimator": wp_estimator.get_state_dict(),
            "battle_count": battle_count,
            "config": {
                "battle_format": self.config.battle_format,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "action_size": self.config.action_size,
            },
        }
        if extra_metadata:
            state["metadata"] = extra_metadata

        # Save latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(state, latest_path)

        # Save numbered checkpoint
        numbered_path = self.checkpoint_dir / f"checkpoint_{battle_count:06d}.pt"
        torch.save(state, numbered_path)

        logger.info(
            f"Saved checkpoint at battle {battle_count} -> {numbered_path}"
        )

    def load_latest(
        self,
        agent1: PPOAgent,
        agent2: PPOAgent,
        league: League,
        wp_estimator: WinProbabilityEstimator,
    ) -> int:
        """Load the latest checkpoint.

        Returns:
            The battle count from the checkpoint, or 0 if none found.
        """
        latest_path = self.checkpoint_dir / "latest.pt"
        if not latest_path.exists():
            logger.info("No checkpoint found, starting fresh.")
            return 0

        return self._load_from_path(
            latest_path, agent1, agent2, league, wp_estimator
        )

    def load_from_path(
        self,
        path: str,
        agent1: PPOAgent,
        agent2: PPOAgent,
        league: League,
        wp_estimator: WinProbabilityEstimator,
    ) -> int:
        """Load a specific checkpoint file."""
        return self._load_from_path(
            Path(path), agent1, agent2, league, wp_estimator
        )

    def _load_from_path(
        self,
        path: Path,
        agent1: PPOAgent,
        agent2: PPOAgent,
        league: League,
        wp_estimator: WinProbabilityEstimator,
    ) -> int:
        logger.info(f"Loading checkpoint from {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)

        agent1.load_state_dict(state["agent1"])
        agent2.load_state_dict(state["agent2"])
        league.load_state_dict(state["league"])
        wp_estimator.load_state_dict(state["wp_estimator"])

        battle_count = state.get("battle_count", 0)
        logger.info(f"Resumed from battle {battle_count}")
        return battle_count

    def save_best(
        self,
        agent1: PPOAgent,
        agent2: PPOAgent,
        league: League,
        wp_estimator: WinProbabilityEstimator,
        battle_count: int,
        win_rate: float,
    ):
        """Save the best-performing model checkpoint (overwrites previous best)."""
        state = {
            "agent1": agent1.get_state_dict(),
            "agent2": agent2.get_state_dict(),
            "league": league.get_state_dict(),
            "wp_estimator": wp_estimator.get_state_dict(),
            "battle_count": battle_count,
            "config": {
                "battle_format": self.config.battle_format,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "action_size": self.config.action_size,
            },
            "metadata": {"best_win_rate": win_rate, "battle_count": battle_count},
        }
        best_path = self.checkpoint_dir / "best.pt"
        torch.save(state, best_path)
        logger.info(f"Saved best model (WR={win_rate:.1%}) at battle {battle_count}")

    def save_best_team(
        self,
        agent: PPOAgent,
        team_id: int,
        battle_count: int,
        win_rate: float,
    ):
        """Save per-team best checkpoint (just this agent's state)."""
        state = {
            "agent": agent.get_state_dict(),
            "battle_count": battle_count,
            "metadata": {"team_id": team_id, "win_rate": win_rate},
        }
        path = self.checkpoint_dir / f"best_team{team_id + 1}.pt"
        torch.save(state, path)
        logger.info(
            f"Saved best team{team_id + 1} agent (WR={win_rate:.1%}) "
            f"at battle {battle_count}"
        )

    def save_combined_best(
        self,
        agent1: PPOAgent,
        agent2: PPOAgent,
        league: League,
        wp_estimator: WinProbabilityEstimator,
        battle_count: int,
    ):
        """Combine per-team bests into best.pt.

        Uses best_team1.pt/best_team2.pt agent weights when available,
        falling back to the current agent if no per-team best exists yet.
        """
        t1_path = self.checkpoint_dir / "best_team1.pt"
        t2_path = self.checkpoint_dir / "best_team2.pt"

        if t1_path.exists():
            a1_state = torch.load(t1_path, map_location="cpu", weights_only=False)["agent"]
        else:
            a1_state = agent1.get_state_dict()

        if t2_path.exists():
            a2_state = torch.load(t2_path, map_location="cpu", weights_only=False)["agent"]
        else:
            a2_state = agent2.get_state_dict()

        state = {
            "agent1": a1_state,
            "agent2": a2_state,
            "league": league.get_state_dict(),
            "wp_estimator": wp_estimator.get_state_dict(),
            "battle_count": battle_count,
            "config": {
                "battle_format": self.config.battle_format,
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "action_size": self.config.action_size,
            },
            "metadata": {"combined_best": True, "battle_count": battle_count},
        }
        best_path = self.checkpoint_dir / "best.pt"
        torch.save(state, best_path)
        logger.info(f"Saved combined best model at battle {battle_count}")

    def load_best(
        self,
        agent1: PPOAgent,
        agent2: PPOAgent,
        league: League,
        wp_estimator: WinProbabilityEstimator,
    ) -> int:
        """Load the best checkpoint. Returns battle_count or 0 if not found."""
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.exists():
            logger.warning("No best checkpoint found, cannot rollback.")
            return 0
        return self._load_from_path(best_path, agent1, agent2, league, wp_estimator)

    def list_checkpoints(self):
        """List all available checkpoint files."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        return [str(cp) for cp in checkpoints]
