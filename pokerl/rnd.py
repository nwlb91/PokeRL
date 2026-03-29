"""Random Network Distillation (RND) for state-space exploration.

Provides two complementary exploration mechanisms:

1. **Intrinsic novelty reward**: Prediction error between a fixed random
   target network and a trained predictor network.  High error = novel state.
   Added to per-step rewards during episode finalization.

2. **Adaptive exploration temperature**: Maps per-state novelty to a
   temperature applied to policy logits during action selection.  Novel
   states get higher temperature (more exploration), familiar states get
   lower temperature (more exploitation).

Reference: Burda et al., "Exploration by Random Network Distillation", 2018.
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pokerl.config import Config
from pokerl.models import RNDPredictorNet, RNDTargetNet

logger = logging.getLogger(__name__)


class RunningMeanStd:
    """Welford's online algorithm for running mean and variance.

    Used to normalize intrinsic rewards to approximately unit variance,
    keeping the RND signal at a stable scale throughout training.
    """

    def __init__(self):
        self.mean: float = 0.0
        self.var: float = 1.0
        self.count: int = 0

    def update(self, values: np.ndarray):
        """Update running statistics with a batch of values."""
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        batch_count = len(values)

        if batch_count == 0:
            return

        total_count = self.count + batch_count

        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total_count

        # Combine variances using parallel algorithm
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count if total_count > 0 else 1.0
        self.count = total_count

    @property
    def std(self) -> float:
        return max(math.sqrt(self.var), 1e-8)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize values to approximately zero mean, unit variance."""
        return (values - self.mean) / self.std

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict):
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


class RNDExploration:
    """Manages RND target/predictor networks and intrinsic reward computation.

    The target network is a fixed random projection (never trained).
    The predictor network is trained to match the target's output for
    observed states.  Prediction error is high for novel states and low
    for familiar ones.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

        self.target = RNDTargetNet(
            hidden_size=config.rnd_hidden_size,
            embedding_dim=config.rnd_embedding_dim,
        ).to(self.device)

        self.predictor = RNDPredictorNet(
            hidden_size=config.rnd_hidden_size,
            embedding_dim=config.rnd_embedding_dim,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.predictor.parameters(), lr=config.rnd_lr
        )

        # Running statistics for reward normalization
        self.reward_stats = RunningMeanStd()

        # Running statistics for novelty temperature (separate from reward
        # stats so temperature calibration is independent)
        self.novelty_stats = RunningMeanStd()

    def compute_intrinsic_rewards(
        self, observations: List[np.ndarray]
    ) -> np.ndarray:
        """Batch-compute normalized intrinsic rewards.

        Args:
            observations: List of observation arrays from an episode.

        Returns:
            Array of per-step intrinsic rewards (normalized to ~unit variance).
        """
        if len(observations) == 0:
            return np.array([], dtype=np.float32)

        with torch.no_grad():
            obs_t = torch.from_numpy(np.stack(observations)).to(self.device)
            target_emb = self.target(obs_t)
            pred_emb = self.predictor(obs_t)
            # Per-observation MSE
            raw_errors = ((target_emb - pred_emb) ** 2).mean(dim=-1).cpu().numpy()

        # Update running stats and normalize
        self.reward_stats.update(raw_errors)
        normalized = self.reward_stats.normalize(raw_errors)

        return normalized.astype(np.float32)

    def compute_novelty_temperature(self, obs: np.ndarray) -> float:
        """Compute exploration temperature for a single observation.

        Maps RND prediction error to a temperature in [temp_min, temp_max]
        via sigmoid of the z-score.  Self-calibrating: "novel" is defined
        relative to the running distribution of errors.

        Args:
            obs: Single observation array.

        Returns:
            Temperature in [temp_min, temp_max].
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            target_emb = self.target(obs_t)
            pred_emb = self.predictor(obs_t)
            raw_error = float(((target_emb - pred_emb) ** 2).mean())

        # Update novelty stats (single value)
        self.novelty_stats.update(np.array([raw_error]))

        # Z-score relative to running distribution
        z = (raw_error - self.novelty_stats.mean) / self.novelty_stats.std

        # Sigmoid mapping to [temp_min, temp_max]
        sigmoid_z = 1.0 / (1.0 + math.exp(-z))
        temp_min = self.config.rnd_temp_min
        temp_max = self.config.rnd_temp_max
        temperature = temp_min + (temp_max - temp_min) * sigmoid_z

        return temperature

    def train_predictor(self, observations: List[np.ndarray]) -> float:
        """Train the predictor network on collected observations.

        Args:
            observations: All observations collected since last training step.

        Returns:
            Mean prediction loss.
        """
        if len(observations) == 0:
            return 0.0

        obs_arr = np.stack(observations)
        obs_t = torch.from_numpy(obs_arr).to(self.device)

        # Train in minibatches for efficiency
        batch_size = min(256, len(observations))
        indices = torch.randperm(len(observations), device=self.device)
        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(observations), batch_size):
            end = min(start + batch_size, len(observations))
            idx = indices[start:end]

            batch_obs = obs_t[idx]
            with torch.no_grad():
                target_emb = self.target(batch_obs)
            pred_emb = self.predictor(batch_obs)

            loss = ((target_emb - pred_emb) ** 2).mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def state_dict(self) -> dict:
        return {
            "target": self.target.state_dict(),
            "predictor": self.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "reward_stats": self.reward_stats.state_dict(),
            "novelty_stats": self.novelty_stats.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.target.load_state_dict(state["target"])
        self.predictor.load_state_dict(state["predictor"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.reward_stats.load_state_dict(state["reward_stats"])
        self.novelty_stats.load_state_dict(state["novelty_stats"])
