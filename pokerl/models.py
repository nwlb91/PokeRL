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

    When uncertainty_heads > 1, maintains an ensemble of policy heads
    for uncertainty-weighted exploration. Actions with higher disagreement
    across heads receive a logit bonus, encouraging exploration of
    uncertain state-action regions.
    """

    def __init__(self, obs_size: int, action_size: int,
                 hidden_size: int = 256, num_layers: int = 3,
                 uncertainty_heads: int = 1, uncertainty_weight: float = 0.5):
        super().__init__()
        self.num_uncertainty_heads = uncertainty_heads
        self.uncertainty_weight = uncertainty_weight

        self.backbone = BattleNetwork(obs_size, hidden_size, num_layers)

        # Primary policy head (always present for backward compatibility)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

        # Additional ensemble heads for uncertainty estimation
        if uncertainty_heads > 1:
            self.ensemble_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, action_size),
                ) for _ in range(uncertainty_heads - 1)
            ])

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor,
                deterministic: bool = False,
                detach_uncertainty: bool = False):
        """Forward pass.

        Args:
            obs: (batch, obs_size) battle observation
            action_mask: (batch, action_size) binary mask of legal actions
            deterministic: if True, use mean logits only (no uncertainty bonus)
            detach_uncertainty: if True, detach std from computation graph so
                gradients only flow through the mean logits.  Used during PPO
                updates to keep importance-sampling ratios consistent while
                preventing the uncertainty bonus from corrupting policy gradients.

        Returns:
            logits: (batch, action_size) masked log-probabilities
            value: (batch, 1) state value
        """
        features = self.backbone(obs)

        if self.num_uncertainty_heads > 1:
            all_logits = torch.stack(
                [self.policy_head(features)]
                + [h(features) for h in self.ensemble_heads],
                dim=0,
            )  # (H, batch, action_size)
            mean_logits = all_logits.mean(dim=0)

            if deterministic:
                logits = mean_logits
            else:
                std_logits = all_logits.std(dim=0)
                if detach_uncertainty:
                    std_logits = std_logits.detach()
                logits = mean_logits + self.uncertainty_weight * std_logits
        else:
            logits = self.policy_head(features)

        # Mask illegal actions with large negative value
        logits = logits + (action_mask.log().clamp(min=-1e8))

        value = self.value_head(features)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor, action_mask: torch.Tensor,
                              action: torch.Tensor = None,
                              deterministic: bool = False,
                              detach_uncertainty: bool = False):
        """Sample or evaluate an action.

        Args:
            obs: (batch, obs_size)
            action_mask: (batch, action_size)
            action: optional (batch,) action to evaluate
            deterministic: if True, no uncertainty bonus in logits
            detach_uncertainty: if True, stop gradients through ensemble std

        Returns:
            action, log_prob, entropy, value
        """
        logits, value = self.forward(
            obs, action_mask,
            deterministic=deterministic,
            detach_uncertainty=detach_uncertainty,
        )
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def mean_ensemble_uncertainty(self, obs: torch.Tensor,
                                   action_mask: torch.Tensor) -> float:
        """Compute mean per-action uncertainty across a batch.

        Returns 0.0 if ensemble is not enabled.
        """
        if self.num_uncertainty_heads <= 1:
            return 0.0
        with torch.no_grad():
            features = self.backbone(obs)
            all_logits = torch.stack(
                [self.policy_head(features)]
                + [h(features) for h in self.ensemble_heads],
                dim=0,
            )
            # Only measure uncertainty over legal actions
            legal = action_mask > 0
            std = all_logits.std(dim=0)
            if legal.any():
                return std[legal].mean().item()
            return std.mean().item()


class TeamPreviewNet(nn.Module):
    """Network for selecting a lead at team preview.

    Takes team preview observation, outputs lead selection logits.
    Supports ensemble policy heads for uncertainty-weighted exploration.
    """

    def __init__(self, obs_size: int = TEAM_PREVIEW_OBS_SIZE,
                 hidden_size: int = 128, num_leads: int = 6,
                 uncertainty_heads: int = 1, uncertainty_weight: float = 0.5):
        super().__init__()
        self.num_uncertainty_heads = uncertainty_heads
        self.uncertainty_weight = uncertainty_weight

        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Primary policy head
        self.policy_head = nn.Linear(hidden_size, num_leads)

        # Additional ensemble heads for uncertainty estimation
        if uncertainty_heads > 1:
            self.ensemble_heads = nn.ModuleList([
                nn.Linear(hidden_size, num_leads)
                for _ in range(uncertainty_heads - 1)
            ])

        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor,
                deterministic: bool = False,
                detach_uncertainty: bool = False):
        features = self.net(obs)

        if self.num_uncertainty_heads > 1:
            all_logits = torch.stack(
                [self.policy_head(features)]
                + [h(features) for h in self.ensemble_heads],
                dim=0,
            )
            mean_logits = all_logits.mean(dim=0)

            if deterministic:
                logits = mean_logits
            else:
                std_logits = all_logits.std(dim=0)
                if detach_uncertainty:
                    std_logits = std_logits.detach()
                logits = mean_logits + self.uncertainty_weight * std_logits
        else:
            logits = self.policy_head(features)

        logits = logits + (mask.log().clamp(min=-1e8))
        value = self.value_head(features)
        return logits, value

    def get_action_and_value(self, obs: torch.Tensor, mask: torch.Tensor,
                              action: torch.Tensor = None,
                              deterministic: bool = False,
                              detach_uncertainty: bool = False):
        logits, value = self.forward(
            obs, mask,
            deterministic=deterministic,
            detach_uncertainty=detach_uncertainty,
        )
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def mean_ensemble_uncertainty(self, obs: torch.Tensor,
                                   mask: torch.Tensor) -> float:
        """Compute mean per-action uncertainty across a batch."""
        if self.num_uncertainty_heads <= 1:
            return 0.0
        with torch.no_grad():
            features = self.net(obs)
            all_logits = torch.stack(
                [self.policy_head(features)]
                + [h(features) for h in self.ensemble_heads],
                dim=0,
            )
            legal = mask > 0
            std = all_logits.std(dim=0)
            if legal.any():
                return std[legal].mean().item()
            return std.mean().item()


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
