"""Win Probability Estimator for reward shaping.

Trains a separate neural network to predict win probability from battle state.
The change in win probability between turns provides dense reward signal,
complementing the sparse win/loss terminal reward.

The estimator is trained on historical battle data:
  - Each (state, outcome) pair is stored in a replay buffer
  - Periodically, the estimator is updated via supervised learning
"""

import random
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pokerl.config import Config
from pokerl.models import WinProbabilityNet
from pokerl.features import BATTLE_OBS_SIZE, BATTLE_OBS_SIZE_EXTENDED


class WinProbabilityEstimator:
    """Trains and serves win probability estimates for reward shaping."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

        obs_size = BATTLE_OBS_SIZE_EXTENDED if config.team_sheet_obs else BATTLE_OBS_SIZE
        self.net = WinProbabilityNet(
            obs_size=obs_size,
            hidden_size=config.wp_hidden_size,
        ).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=config.wp_lr)

        # Replay buffer: stores (obs_array, win_label)
        self.buffer: deque = deque(maxlen=config.wp_buffer_size)

        self.battles_since_update = 0
        self.total_updates = 0

    def predict_batch(self, observations: np.ndarray) -> np.ndarray:
        """Predict win probabilities for a batch of observations.

        Args:
            observations: (N, obs_size) array of battle observations.

        Returns:
            (N,) array of win probabilities in [0, 1].
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(observations).to(self.device)
            return self.net(obs_t).cpu().numpy()

    def predict(self, obs: np.ndarray) -> float:
        """Predict win probability for a single observation."""
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            return self.net(obs_t).item()

    def compute_shaped_rewards_batch(
        self, observations: List[np.ndarray], gamma: float = 0.99
    ) -> np.ndarray:
        """Compute WP-delta shaped rewards for a full trajectory at once.

        Uses proper potential-based reward shaping: gamma * phi(s') - phi(s),
        which preserves the optimal policy under discounting.

        Returns:
            (N-1,) array of shaped rewards for steps 0..N-2.
            The last step gets terminal reward (handled by caller).
        """
        n = len(observations)
        if n < 2:
            return np.zeros(0, dtype=np.float32)

        w = self.config.wp_reward_weight
        obs_stack = np.array(observations, dtype=np.float32)
        wp = self.predict_batch(obs_stack)

        # Proper PBRS: gamma * phi(s') - phi(s)
        wp_delta = gamma * wp[1:] - wp[:-1]
        return (w * wp_delta).astype(np.float32)

    def store_trajectory(self, observations: List[np.ndarray], won: bool):
        """Store a full battle trajectory for training.

        Each observation gets the game outcome as its label.
        We also add interpolated labels based on position in the game.
        """
        n = len(observations)
        if n == 0:
            return

        outcome = 1.0 if won else 0.0

        for i, obs in enumerate(observations):
            progress = (i + 1) / n
            # Exponential ramp: early-game labels stay near 0.5 (uncertain),
            # late-game labels converge quickly to the actual outcome.  This
            # better reflects reality where a single pivotal turn can swing
            # win probability, versus linear interpolation which under-weights
            # late-game shifts.
            confidence = progress * progress  # quadratic ramp
            label = 0.5 + confidence * (outcome - 0.5)
            self.buffer.append((obs, label))

        self.battles_since_update += 1

    def maybe_update(self) -> dict:
        """Update the estimator if enough battles have elapsed."""
        if self.battles_since_update < self.config.wp_train_interval:
            return {}
        if len(self.buffer) < self.config.wp_batch_size:
            return {}

        self.battles_since_update = 0
        return self.train_step()

    def train_step(self) -> dict:
        """Run a training step on the replay buffer."""
        batch_size = min(self.config.wp_batch_size, len(self.buffer))
        # Sample indices to avoid converting entire deque to list
        indices = random.sample(range(len(self.buffer)), batch_size)

        obs_batch = np.array([self.buffer[i][0] for i in indices])
        labels = np.array([self.buffer[i][1] for i in indices], dtype=np.float32)

        obs_t = torch.from_numpy(obs_batch).to(self.device)
        labels_t = torch.from_numpy(labels).to(self.device)

        preds = self.net(obs_t).clamp(1e-7, 1 - 1e-7)
        loss = F.binary_cross_entropy(preds, labels_t)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.total_updates += 1

        return {
            "wp_loss": loss.item(),
            "wp_mean_pred": preds.mean().item(),
            "wp_buffer_size": len(self.buffer),
        }

    def get_state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_updates": self.total_updates,
        }

    def load_state_dict(self, state: dict):
        self.net.load_state_dict(state["net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.total_updates = state.get("total_updates", 0)
