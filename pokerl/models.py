"""Neural network architectures for PokeRL.

Contains:
  - BattleNetwork: shared backbone for battle state processing
  - PolicyValueNet: actor-critic network for battle decisions
  - TeamPreviewNet: separate network for lead selection at team preview
  - WinProbabilityNet: auxiliary model for reward shaping
"""

import torch
import torch.nn as nn

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
                 uncertainty_heads: int = 1, uncertainty_weight: float = 0.5,
                 matchup_context_size: int = 0,
                 q_head_enabled: bool = False):
        super().__init__()
        self.num_uncertainty_heads = uncertainty_heads
        self.uncertainty_weight = uncertainty_weight
        self.matchup_context_size = matchup_context_size
        self.q_head_enabled = q_head_enabled
        self.action_size = action_size

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

        # Value head — optionally conditioned on matchup context (e.g. team WR EMA)
        # so the critic can produce different value estimates for different matchup
        # difficulties without affecting the policy head.
        value_input_size = hidden_size + matchup_context_size
        self.value_head = nn.Sequential(
            nn.Linear(value_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Optional Q-value head for action reranking (search)
        self.q_head = None
        if q_head_enabled:
            self.q_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, action_size),
            )

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor,
                deterministic: bool = False,
                detach_uncertainty: bool = False,
                matchup_context: torch.Tensor = None):
        """Forward pass.

        Args:
            obs: (batch, obs_size) battle observation
            action_mask: (batch, action_size) binary mask of legal actions
            deterministic: if True, use mean logits only (no uncertainty bonus)
            detach_uncertainty: if True, detach std from computation graph so
                gradients only flow through the mean logits.  Used during PPO
                updates to keep importance-sampling ratios consistent while
                preventing the uncertainty bonus from corrupting policy gradients.
            matchup_context: optional (batch, matchup_context_size) conditioning
                vector (e.g. team WR EMA) fed only to the value head.

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

        # Value head with optional matchup conditioning
        if self.matchup_context_size > 0:
            if matchup_context is not None:
                value_input = torch.cat([features, matchup_context], dim=-1)
            else:
                # No context provided — pad with zeros so the Linear layer
                # receives the correct input width.
                padding = features.new_zeros(
                    features.shape[0], self.matchup_context_size
                )
                value_input = torch.cat([features, padding], dim=-1)
        else:
            value_input = features
        value = self.value_head(value_input)

        q_values = None
        if self.q_head is not None:
            q_values = self.q_head(features)
            q_values = q_values + (action_mask.log().clamp(min=-1e8))

        return logits, value, q_values

    def get_action_and_value(self, obs: torch.Tensor, action_mask: torch.Tensor,
                              action: torch.Tensor = None,
                              deterministic: bool = False,
                              detach_uncertainty: bool = False,
                              matchup_context: torch.Tensor = None):
        """Sample or evaluate an action.

        Args:
            obs: (batch, obs_size)
            action_mask: (batch, action_size)
            action: optional (batch,) action to evaluate
            deterministic: if True, no uncertainty bonus in logits
            detach_uncertainty: if True, stop gradients through ensemble std
            matchup_context: optional (batch, matchup_context_size) for value head

        Returns:
            action, log_prob, entropy, value[, q_values]
        """
        logits, value, q_values = self.forward(
            obs, action_mask,
            deterministic=deterministic,
            detach_uncertainty=detach_uncertainty,
            matchup_context=matchup_context,
        )
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        if q_values is not None:
            return action, log_prob, entropy, value.squeeze(-1), q_values
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


class RecurrentPolicyValueNet(nn.Module):
    """Actor-critic with LSTM for sequential reasoning across battle turns.

    Architecture: obs → MLP backbone → LSTM → policy/value/Q heads.
    The LSTM hidden state accumulates information across turns within a
    battle, enabling the model to track move usage patterns, infer
    opponent strategies, and remember damage rolls.
    """

    def __init__(self, obs_size: int, action_size: int,
                 hidden_size: int = 256, num_layers: int = 3,
                 lstm_hidden_size: int = 256,
                 uncertainty_heads: int = 1, uncertainty_weight: float = 0.5,
                 matchup_context_size: int = 0,
                 q_head_enabled: bool = False):
        super().__init__()
        self.num_uncertainty_heads = uncertainty_heads
        self.uncertainty_weight = uncertainty_weight
        self.matchup_context_size = matchup_context_size
        self.q_head_enabled = q_head_enabled
        self.action_size = action_size
        self.lstm_hidden_size = lstm_hidden_size

        self.backbone = BattleNetwork(obs_size, hidden_size, num_layers)
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=True)

        # Policy head reads from LSTM output
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size // 2, action_size),
        )

        if uncertainty_heads > 1:
            self.ensemble_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(lstm_hidden_size // 2, action_size),
                ) for _ in range(uncertainty_heads - 1)
            ])

        value_input_size = lstm_hidden_size + matchup_context_size
        self.value_head = nn.Sequential(
            nn.Linear(value_input_size, lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size // 2, 1),
        )

        self.q_head = None
        if q_head_enabled:
            self.q_head = nn.Sequential(
                nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
                nn.ReLU(),
                nn.Linear(lstm_hidden_size // 2, action_size),
            )

    def initial_hidden(self, batch_size: int = 1, device=None):
        """Create zero-initialized LSTM hidden state."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden_size, device=device)
        return (h, c)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor,
                hidden_state=None,
                deterministic: bool = False,
                detach_uncertainty: bool = False,
                matchup_context: torch.Tensor = None):
        """Forward pass for a single timestep.

        Args:
            obs: (batch, obs_size)
            action_mask: (batch, action_size)
            hidden_state: (h, c) tuple for LSTM, or None for zero init
            deterministic, detach_uncertainty, matchup_context: same as PolicyValueNet

        Returns:
            logits, value, q_values, new_hidden_state
        """
        backbone_features = self.backbone(obs)

        # LSTM expects (batch, seq_len, features)
        lstm_input = backbone_features.unsqueeze(1)
        if hidden_state is None:
            hidden_state = self.initial_hidden(obs.shape[0], obs.device)
        lstm_out, new_hidden = self.lstm(lstm_input, hidden_state)
        features = lstm_out.squeeze(1)  # (batch, lstm_hidden_size)

        # Policy logits (same ensemble logic as PolicyValueNet)
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

        logits = logits + (action_mask.log().clamp(min=-1e8))

        # Value head
        if self.matchup_context_size > 0:
            if matchup_context is not None:
                value_input = torch.cat([features, matchup_context], dim=-1)
            else:
                padding = features.new_zeros(features.shape[0], self.matchup_context_size)
                value_input = torch.cat([features, padding], dim=-1)
        else:
            value_input = features
        value = self.value_head(value_input)

        q_values = None
        if self.q_head is not None:
            q_values = self.q_head(features)
            q_values = q_values + (action_mask.log().clamp(min=-1e8))

        return logits, value, q_values, new_hidden

    def forward_sequence(self, obs_seq: torch.Tensor, mask_seq: torch.Tensor,
                         hidden_state=None,
                         detach_uncertainty: bool = False,
                         matchup_context: torch.Tensor = None):
        """Process a full episode sequence for PPO recomputation.

        Args:
            obs_seq: (seq_len, obs_size) single episode observations
            mask_seq: (seq_len, action_size) action masks
            hidden_state: initial hidden state (or None)
            matchup_context: (seq_len, context_size) or None

        Returns:
            logits: (seq_len, action_size)
            values: (seq_len,)
            q_values: (seq_len, action_size) or None
        """
        seq_len = obs_seq.shape[0]
        backbone_features = self.backbone(obs_seq)  # (seq_len, hidden)

        # Process full sequence through LSTM
        lstm_input = backbone_features.unsqueeze(0)  # (1, seq_len, hidden)
        if hidden_state is None:
            hidden_state = self.initial_hidden(1, obs_seq.device)
        lstm_out, _ = self.lstm(lstm_input, hidden_state)
        features = lstm_out.squeeze(0)  # (seq_len, lstm_hidden)

        # Policy logits
        if self.num_uncertainty_heads > 1:
            all_logits = torch.stack(
                [self.policy_head(features)]
                + [h(features) for h in self.ensemble_heads],
                dim=0,
            )
            mean_logits = all_logits.mean(dim=0)
            std_logits = all_logits.std(dim=0)
            if detach_uncertainty:
                std_logits = std_logits.detach()
            logits = mean_logits + self.uncertainty_weight * std_logits
        else:
            logits = self.policy_head(features)

        logits = logits + (mask_seq.log().clamp(min=-1e8))

        # Value
        if self.matchup_context_size > 0:
            if matchup_context is not None:
                value_input = torch.cat([features, matchup_context], dim=-1)
            else:
                padding = features.new_zeros(seq_len, self.matchup_context_size)
                value_input = torch.cat([features, padding], dim=-1)
        else:
            value_input = features
        values = self.value_head(value_input).squeeze(-1)

        q_values = None
        if self.q_head is not None:
            q_values = self.q_head(features)
            q_values = q_values + (mask_seq.log().clamp(min=-1e8))

        return logits, values, q_values

    def get_action_and_value(self, obs, action_mask, action=None,
                              deterministic=False, detach_uncertainty=False,
                              matchup_context=None):
        """Compatibility method for non-recurrent PPO update.

        When called without hidden state (during PPO), treats each
        observation independently (hidden state = zeros).
        For proper recurrent PPO, use forward_sequence instead.
        """
        logits, value, q_values, _ = self.forward(
            obs, action_mask,
            deterministic=deterministic,
            detach_uncertainty=detach_uncertainty,
            matchup_context=matchup_context,
        )
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        if q_values is not None:
            return action, log_prob, entropy, value.squeeze(-1), q_values
        return action, log_prob, entropy, value.squeeze(-1)

    def mean_ensemble_uncertainty(self, obs, action_mask):
        if self.num_uncertainty_heads <= 1:
            return 0.0
        with torch.no_grad():
            backbone_features = self.backbone(obs)
            lstm_input = backbone_features.unsqueeze(1)
            hidden = self.initial_hidden(obs.shape[0], obs.device)
            lstm_out, _ = self.lstm(lstm_input, hidden)
            features = lstm_out.squeeze(1)
            all_logits = torch.stack(
                [self.policy_head(features)]
                + [h(features) for h in self.ensemble_heads],
                dim=0,
            )
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
                              detach_uncertainty: bool = False,
                              matchup_context: torch.Tensor = None):  # noqa: ARG002
        # matchup_context is unused by the preview net but accepted here so
        # PPOAgent._ppo_update() can call battle_net and preview_net with the
        # same interface without branching.
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


class RNDTargetNet(nn.Module):
    """Fixed random network for RND (never trained).

    Maps observations to a low-dimensional embedding.  The random projection
    is frozen at initialization — the prediction error of a learned predictor
    against this target serves as a novelty signal.
    """

    def __init__(self, obs_size: int = BATTLE_OBS_SIZE,
                 hidden_size: int = 256, embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim),
        )
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class RNDPredictorNet(nn.Module):
    """Trainable predictor network for RND.

    Has one extra hidden layer compared to the target so it has enough
    capacity to match the target's output for frequently-seen states while
    still producing high error on novel states.
    """

    def __init__(self, obs_size: int = BATTLE_OBS_SIZE,
                 hidden_size: int = 256, embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
