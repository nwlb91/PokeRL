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
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from pokerl.config import Config
from pokerl.models import WinProbabilityNet
from pokerl.features import BATTLE_OBS_SIZE


class WinProbabilityEstimator:
    """Trains and serves win probability estimates for reward shaping."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)

        self.net = WinProbabilityNet(
            obs_size=BATTLE_OBS_SIZE,
            hidden_size=config.wp_hidden_size,
        ).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=config.wp_lr)

        # Replay buffer: stores (obs_array, win_label)
        self.buffer: deque = deque(maxlen=config.wp_buffer_size)

        self.battles_since_update = 0
        self.total_updates = 0

    def predict(self, obs: np.ndarray) -> float:
        """Predict win probability for a single observation.

        Args:
            obs: Battle observation array.

        Returns:
            Win probability in [0, 1].
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            return self.net(obs_t).item()

    def compute_shaped_reward(
        self,
        prev_obs: np.ndarray,
        curr_obs: np.ndarray,
        terminal_reward: float,
        done: bool,
    ) -> float:
        """Compute reward shaped by win probability change.

        The shaped reward is:
            r_shaped = (1 - w) * r_terminal + w * (wp_current - wp_previous)

        where w is the wp_reward_weight. At terminal states, the full
        terminal reward is used.

        Args:
            prev_obs: Previous turn's observation.
            curr_obs: Current turn's observation.
            terminal_reward: Sparse reward (e.g., +1 for win, -1 for loss).
            done: Whether the episode is over.

        Returns:
            Blended reward.
        """
        w = self.config.wp_reward_weight

        if done:
            return terminal_reward

        wp_prev = self.predict(prev_obs)
        wp_curr = self.predict(curr_obs)
        wp_delta = wp_curr - wp_prev

        return (1 - w) * terminal_reward + w * wp_delta

    def store_trajectory(self, observations: List[np.ndarray], won: bool):
        """Store a full battle trajectory for training.

        Each observation gets the game outcome as its label.
        We also add interpolated labels based on position in the game.

        Args:
            observations: List of battle observations from the game.
            won: Whether the agent won.
        """
        n = len(observations)
        if n == 0:
            return

        outcome = 1.0 if won else 0.0

        for i, obs in enumerate(observations):
            # Interpolate: early states get a more uncertain label
            # Late states get labels closer to the outcome
            progress = (i + 1) / n
            # Soft label: blend between 0.5 (uncertain) and outcome
            label = 0.5 + progress * (outcome - 0.5)
            self.buffer.append((obs, label))

        self.battles_since_update += 1

    def maybe_update(self) -> dict:
        """Update the estimator if enough battles have elapsed.

        Returns:
            Training metrics dict, or empty dict if no update.
        """
        if self.battles_since_update < self.config.wp_train_interval:
            return {}
        if len(self.buffer) < self.config.wp_batch_size:
            return {}

        self.battles_since_update = 0
        return self.train_step()

    def train_step(self) -> dict:
        """Run a training step on the replay buffer."""
        batch_size = min(self.config.wp_batch_size, len(self.buffer))
        batch = random.sample(list(self.buffer), batch_size)

        obs_batch = np.array([b[0] for b in batch])
        labels = np.array([b[1] for b in batch])

        obs_t = torch.FloatTensor(obs_batch).to(self.device)
        labels_t = torch.FloatTensor(labels).to(self.device)

        preds = self.net(obs_t)
        loss = F.binary_cross_entropy(preds, labels_t)

        self.optimizer.zero_grad()
        loss.backward()
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
