"""PPO Agent for Pokemon battles.

Manages:
  - Battle policy (PolicyValueNet) for in-battle decisions
  - Team preview policy (TeamPreviewNet) for lead selection
  - Rollout buffer and PPO update logic
  - Win probability reward shaping integration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from pokerl.config import Config
from pokerl.models import PolicyValueNet, TeamPreviewNet
from pokerl.features import BATTLE_OBS_SIZE, TEAM_PREVIEW_OBS_SIZE


@dataclass
class RolloutStep:
    obs: np.ndarray
    action: int
    action_mask: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: bool


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(self):
        self.steps: List[RolloutStep] = []

    def add(self, step: RolloutStep):
        self.steps.append(step)

    def clear(self):
        self.steps = []

    def __len__(self):
        return len(self.steps)

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float, last_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and discounted returns."""
        n = len(self.steps)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            step = self.steps[t]
            if step.done:
                next_value = 0.0
                last_gae = 0.0

            delta = step.reward + gamma * next_value - step.value
            last_gae = delta + gamma * gae_lambda * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + step.value
            next_value = step.value

        return returns, advantages

    def get_batches(self, batch_size: int, returns: np.ndarray,
                    advantages: np.ndarray):
        """Yield mini-batches for PPO update."""
        n = len(self.steps)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            obs = np.array([self.steps[i].obs for i in batch_idx])
            actions = np.array([self.steps[i].action for i in batch_idx])
            masks = np.array([self.steps[i].action_mask for i in batch_idx])
            old_log_probs = np.array([self.steps[i].log_prob for i in batch_idx])
            batch_returns = returns[batch_idx]
            batch_advantages = advantages[batch_idx]

            yield (obs, actions, masks, old_log_probs,
                   batch_returns, batch_advantages)


class PPOAgent:
    """PPO agent with separate battle and team preview networks."""

    def __init__(self, config: Config, agent_id: str = "agent"):
        self.config = config
        self.agent_id = agent_id
        self.device = torch.device(config.device)

        # Battle policy network
        self.battle_net = PolicyValueNet(
            obs_size=BATTLE_OBS_SIZE,
            action_size=config.action_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        ).to(self.device)

        # Team preview network
        self.preview_net = TeamPreviewNet(
            obs_size=TEAM_PREVIEW_OBS_SIZE,
            hidden_size=config.hidden_size // 2,
            num_leads=config.team_preview_action_size,
        ).to(self.device)

        # Optimizers
        self.battle_optimizer = optim.Adam(
            self.battle_net.parameters(), lr=config.lr
        )
        self.preview_optimizer = optim.Adam(
            self.preview_net.parameters(), lr=config.lr
        )

        # Rollout buffers
        self.battle_buffer = RolloutBuffer()
        self.preview_buffer = RolloutBuffer()

        # Training stats
        self.total_battles = 0
        self.total_updates = 0
        self.wins = 0
        self.losses = 0

    def select_battle_action(
        self, obs: np.ndarray, action_mask: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Select a battle action using the policy network.

        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)

            logits, value = self.battle_net(obs_t, mask_t)

            if deterministic:
                action = logits.argmax(dim=-1)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(action)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

        return (action.item(), log_prob.item(), value.squeeze().item())

    def select_preview_action(
        self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Select a lead at team preview.

        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)

            logits, value = self.preview_net(obs_t, mask_t)

            if deterministic:
                action = logits.argmax(dim=-1)
                dist = torch.distributions.Categorical(logits=logits)
                log_prob = dist.log_prob(action)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

        return (action.item(), log_prob.item(), value.squeeze().item())

    def update_battle_policy(self) -> Dict[str, float]:
        """Run PPO update on battle rollout buffer."""
        return self._ppo_update(
            self.battle_buffer, self.battle_net, self.battle_optimizer
        )

    def update_preview_policy(self) -> Dict[str, float]:
        """Run PPO update on team preview rollout buffer."""
        return self._ppo_update(
            self.preview_buffer, self.preview_net, self.preview_optimizer
        )

    def _ppo_update(
        self, buffer: RolloutBuffer, network: nn.Module, optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """Generic PPO update on a buffer/network pair."""
        if len(buffer) == 0:
            return {}

        cfg = self.config

        # Compute returns and advantages
        returns, advantages = buffer.compute_returns_and_advantages(
            cfg.gamma, cfg.gae_lambda, last_value=0.0
        )

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for epoch in range(cfg.ppo_epochs):
            for batch in buffer.get_batches(cfg.batch_size, returns, advantages):
                (obs, actions, masks, old_log_probs,
                 batch_returns, batch_advantages) = batch

                obs_t = torch.FloatTensor(obs).to(self.device)
                actions_t = torch.LongTensor(actions).to(self.device)
                masks_t = torch.FloatTensor(masks).to(self.device)
                old_lp_t = torch.FloatTensor(old_log_probs).to(self.device)
                returns_t = torch.FloatTensor(batch_returns).to(self.device)
                adv_t = torch.FloatTensor(batch_advantages).to(self.device)

                _, new_lp, entropy, values = network.get_action_and_value(
                    obs_t, masks_t, actions_t
                )

                # Policy loss (clipped PPO)
                ratio = torch.exp(new_lp - old_lp_t)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, returns_t)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss
                        + cfg.value_coef * value_loss
                        + cfg.entropy_coef * entropy_loss)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), cfg.max_grad_norm)
                optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_batches += 1

        buffer.clear()
        self.total_updates += 1

        if num_batches == 0:
            return {}

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "entropy": total_entropy / num_batches,
        }

    def get_state_dict(self) -> dict:
        """Get full agent state for checkpointing."""
        return {
            "agent_id": self.agent_id,
            "battle_net": self.battle_net.state_dict(),
            "preview_net": self.preview_net.state_dict(),
            "battle_optimizer": self.battle_optimizer.state_dict(),
            "preview_optimizer": self.preview_optimizer.state_dict(),
            "total_battles": self.total_battles,
            "total_updates": self.total_updates,
            "wins": self.wins,
            "losses": self.losses,
        }

    def load_state_dict(self, state: dict):
        """Load agent state from checkpoint."""
        self.agent_id = state.get("agent_id", self.agent_id)
        self.battle_net.load_state_dict(state["battle_net"])
        self.preview_net.load_state_dict(state["preview_net"])
        if "battle_optimizer" in state:
            self.battle_optimizer.load_state_dict(state["battle_optimizer"])
        if "preview_optimizer" in state:
            self.preview_optimizer.load_state_dict(state["preview_optimizer"])
        self.total_battles = state.get("total_battles", 0)
        self.total_updates = state.get("total_updates", 0)
        self.wins = state.get("wins", 0)
        self.losses = state.get("losses", 0)

    def load_weights_only(self, state: dict):
        """Load only network weights (for frozen league opponents)."""
        self.battle_net.load_state_dict(state["battle_net"])
        self.preview_net.load_state_dict(state["preview_net"])

    def set_eval(self):
        self.battle_net.eval()
        self.preview_net.eval()

    def set_train(self):
        self.battle_net.train()
        self.preview_net.train()
