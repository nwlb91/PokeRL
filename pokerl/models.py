"""Neural network architectures for PokeRL.

Contains:
  - BattleNetwork: shared backbone for battle state processing
  - PolicyValueNet: actor-critic network for battle decisions
  - TeamPreviewNet: separate network for lead selection at team preview
  - WinProbabilityNet: auxiliary model for reward shaping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pokerl.features import BATTLE_OBS_SIZE, TEAM_PREVIEW_OBS_SIZE


class BattleNetwork(nn.Module):
    """Shared feature extractor for battle states."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        layers = []
        prev_size = input_size
        for i in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyValueNet(nn.Module):
    """Actor-critic network for battle decisions.

    Takes battle observation, outputs:
      - action logits (masked by legal action mask)
      - state value estimate
    """

    def __init__(self, obs_size: int, action_size: int,
                 hidden_size: int = 256, num_layers: int = 3):
        super().__init__()
        self.backbone = BattleNetwork(obs_size, hidden_size, num_layers)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor):
        """Forward pass.

        Args:
            obs: (batch, obs_size) battle observation
            action_mask: (batch, action_size) binary mask of legal actions

        Returns:
            logits: (batch, action_size) masked log-probabilities
            value: (batch, 1) state value
        """
        features = self.backbone(obs)
        logits = self.policy_head(features)

        # Mask illegal actions with large negative value
        logits = logits + (action_mask.log().clamp(min=-1e8))

        value = self.value_head(features)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor, action_mask: torch.Tensor,
                              action: torch.Tensor = None):
        """Sample or evaluate an action.

        Args:
            obs: (batch, obs_size)
            action_mask: (batch, action_size)
            action: optional (batch,) action to evaluate

        Returns:
            action, log_prob, entropy, value
        """
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


class TeamPreviewNet(nn.Module):
    """Network for selecting a lead at team preview.

    Takes team preview observation, outputs lead selection logits.
    """

    def __init__(self, obs_size: int = TEAM_PREVIEW_OBS_SIZE,
                 hidden_size: int = 128, num_leads: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, num_leads)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        features = self.net(obs)
        logits = self.policy_head(features)
        logits = logits + (mask.log().clamp(min=-1e8))
        value = self.value_head(features)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor, mask: torch.Tensor,
                              action: torch.Tensor = None):
        logits, value = self.forward(obs, mask)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)


class WinProbabilityNet(nn.Module):
    """Auxiliary network that estimates win probability from battle state.

    Used for reward shaping: the change in win probability between turns
    provides a dense reward signal.
    """

    def __init__(self, obs_size: int = BATTLE_OBS_SIZE,
                 hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns win probability in [0, 1]."""
        return self.net(obs).squeeze(-1)
