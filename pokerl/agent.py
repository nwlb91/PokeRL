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

    def to_tensors(self, device: torch.device):
        """Pre-stack all data into tensors once (avoid repeated conversions)."""
        n = len(self.steps)
        obs = np.array([self.steps[i].obs for i in range(n)])
        actions = np.array([self.steps[i].action for i in range(n)], dtype=np.int64)
        masks = np.array([self.steps[i].action_mask for i in range(n)])
        log_probs = np.array([self.steps[i].log_prob for i in range(n)], dtype=np.float32)
        old_values = np.array([self.steps[i].value for i in range(n)], dtype=np.float32)
        return (
            torch.from_numpy(obs).to(device),
            torch.from_numpy(actions).to(device),
            torch.from_numpy(masks).to(device),
            torch.from_numpy(log_probs).to(device),
            torch.from_numpy(old_values).to(device),
        )


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
            uncertainty_heads=config.uncertainty_heads,
            uncertainty_weight=config.uncertainty_weight,
        ).to(self.device)

        # Team preview network
        self.preview_net = TeamPreviewNet(
            obs_size=TEAM_PREVIEW_OBS_SIZE,
            hidden_size=config.hidden_size // 2,
            num_leads=config.team_preview_action_size,
            uncertainty_heads=config.uncertainty_heads,
            uncertainty_weight=config.uncertainty_weight,
        ).to(self.device)

        # Try torch.compile for PyTorch 2.0+ (requires triton, not available on Windows)
        if hasattr(torch, "compile") and config.device != "cpu":
            try:
                import triton  # noqa: F401
                self.battle_net = torch.compile(self.battle_net)
                self.preview_net = torch.compile(self.preview_net)
            except Exception:
                pass  # graceful fallback (triton not available)

        # Optimizers
        self.battle_optimizer = optim.Adam(
            self.battle_net.parameters(), lr=config.lr
        )
        self.preview_optimizer = optim.Adam(
            self.preview_net.parameters(), lr=config.lr
        )

        # LR schedulers
        self.battle_lr_scheduler = self._create_lr_scheduler(self.battle_optimizer)
        self.preview_lr_scheduler = self._create_lr_scheduler(self.preview_optimizer)

        # Rollout buffers
        self.battle_buffer = RolloutBuffer()
        self.preview_buffer = RolloutBuffer()

        # Training stats
        self.total_battles = 0
        self.total_updates = 0
        self.wins = 0
        self.losses = 0

    def _create_lr_scheduler(self, optimizer: optim.Optimizer):
        """Create an LR scheduler based on config."""
        cfg = self.config
        if cfg.lr_schedule == "cosine":
            # Use warm restarts for infinite training compatibility.
            # T_0 = estimated updates per entropy_anneal_battles cycle,
            # falling back to a reasonable default.
            updates_per_battle = 1.0 / max(1, cfg.rollout_steps // cfg.batch_size)
            t_0 = max(10, int(10000 * updates_per_battle))
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=t_0, eta_min=cfg.lr_min,
            )
        elif cfg.lr_schedule == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=10, factor=0.5,
                min_lr=cfg.lr_min,
            )
        # "constant" — no scheduler
        return None

    def step_lr_scheduler(self, metric: Optional[float] = None):
        """Step the LR schedulers after a PPO update."""
        for sched in (self.battle_lr_scheduler, self.preview_lr_scheduler):
            if sched is None:
                continue
            if isinstance(sched, optim.lr_scheduler.ReduceLROnPlateau):
                if metric is not None:
                    sched.step(metric)
            else:
                sched.step()

    def select_battle_action(
        self, obs: np.ndarray, action_mask: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Select a battle action using the policy network.

        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)

            logits, value = self.battle_net(obs_t, mask_t, deterministic=deterministic)

            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
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
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)

            logits, value = self.preview_net(obs_t, mask_t, deterministic=deterministic)

            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

        return (action.item(), log_prob.item(), value.squeeze().item())

    def update_battle_policy(self, entropy_coef_override: Optional[float] = None) -> Dict[str, float]:
        """Run PPO update on battle rollout buffer."""
        return self._ppo_update(
            self.battle_buffer, self.battle_net, self.battle_optimizer,
            entropy_coef_override=entropy_coef_override,
        )

    def update_preview_policy(self, entropy_coef_override: Optional[float] = None,
                              ppo_epochs_override: Optional[int] = None) -> Dict[str, float]:
        """Run PPO update on team preview rollout buffer."""
        return self._ppo_update(
            self.preview_buffer, self.preview_net, self.preview_optimizer,
            entropy_coef_override=entropy_coef_override,
            ppo_epochs_override=ppo_epochs_override,
        )

    def _ppo_update(
        self, buffer: RolloutBuffer, network: nn.Module, optimizer: optim.Optimizer,
        entropy_coef_override: Optional[float] = None,
        ppo_epochs_override: Optional[int] = None,
    ) -> Dict[str, float]:
        """Generic PPO update on a buffer/network pair."""
        if len(buffer) == 0:
            return {}

        cfg = self.config
        n = len(buffer)

        # Compute returns and advantages
        returns, advantages = buffer.compute_returns_and_advantages(
            cfg.gamma, cfg.gae_lambda, last_value=0.0
        )

        # Explained variance: how well the critic predicts returns
        values_arr = np.array([step.value for step in buffer.steps], dtype=np.float32)
        var_returns = np.var(returns)
        explained_var = (
            1.0 - np.var(returns - values_arr) / var_returns
            if var_returns > 1e-8 else 0.0
        )

        # Mean episode return
        episode_returns = []
        current_return = 0.0
        for step in buffer.steps:
            current_return += step.reward
            if step.done:
                episode_returns.append(current_return)
                current_return = 0.0
        mean_ep_return = float(np.mean(episode_returns)) if episode_returns else 0.0

        # Normalize advantages
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Pre-convert ALL data to GPU tensors once
        all_obs, all_actions, all_masks, all_old_lp, all_old_values = buffer.to_tensors(self.device)
        all_returns = torch.from_numpy(returns).to(self.device)
        all_advantages = torch.from_numpy(advantages).to(self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        n_epochs = ppo_epochs_override if ppo_epochs_override is not None else cfg.ppo_epochs
        for epoch in range(n_epochs):
            indices = torch.randperm(n, device=self.device)

            for start in range(0, n, cfg.batch_size):
                end = min(start + cfg.batch_size, n)
                idx = indices[start:end]

                # Slice pre-converted tensors (no CPU→GPU transfer)
                obs_t = all_obs[idx]
                actions_t = all_actions[idx]
                masks_t = all_masks[idx]
                old_lp_t = all_old_lp[idx]
                old_val_t = all_old_values[idx]
                returns_t = all_returns[idx]
                adv_t = all_advantages[idx]

                _, new_lp, entropy, values = network.get_action_and_value(
                    obs_t, masks_t, actions_t, detach_uncertainty=True
                )

                # Policy loss (clipped PPO)
                ratio = torch.exp(new_lp - old_lp_t)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped to prevent destructive critic updates)
                values_clipped = old_val_t + torch.clamp(
                    values - old_val_t, -cfg.clip_eps, cfg.clip_eps
                )
                value_loss = torch.max(
                    F.mse_loss(values, returns_t),
                    F.mse_loss(values_clipped, returns_t),
                )

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                ent_coef = entropy_coef_override if entropy_coef_override is not None else cfg.entropy_coef
                loss = (policy_loss
                        + cfg.value_coef * value_loss
                        + ent_coef * entropy_loss)

                optimizer.zero_grad(set_to_none=True)
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

        stats = {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "entropy": total_entropy / num_batches,
            "explained_variance": explained_var,
            "mean_episode_return": mean_ep_return,
        }

        # Log ensemble uncertainty if enabled
        if hasattr(network, 'mean_ensemble_uncertainty') and network.num_uncertainty_heads > 1:
            sample_n = min(256, n)
            sample_idx = torch.randperm(n, device=self.device)[:sample_n]
            stats["mean_uncertainty"] = network.mean_ensemble_uncertainty(
                all_obs[sample_idx], all_masks[sample_idx]
            )

        return stats

    def get_state_dict(self) -> dict:
        """Get full agent state for checkpointing."""
        state = {
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
        if self.battle_lr_scheduler is not None:
            state["battle_lr_scheduler"] = self.battle_lr_scheduler.state_dict()
        if self.preview_lr_scheduler is not None:
            state["preview_lr_scheduler"] = self.preview_lr_scheduler.state_dict()
        return state

    def load_state_dict(self, state: dict):
        """Load agent state from checkpoint.

        Uses strict=False so old checkpoints without ensemble heads
        still load correctly (new heads keep their random init).
        """
        self.agent_id = state.get("agent_id", self.agent_id)
        self.battle_net.load_state_dict(state["battle_net"], strict=False)
        self.preview_net.load_state_dict(state["preview_net"], strict=False)
        if "battle_optimizer" in state:
            self.battle_optimizer.load_state_dict(state["battle_optimizer"])
        if "preview_optimizer" in state:
            self.preview_optimizer.load_state_dict(state["preview_optimizer"])
        self.total_battles = state.get("total_battles", 0)
        self.total_updates = state.get("total_updates", 0)
        self.wins = state.get("wins", 0)
        self.losses = state.get("losses", 0)
        if "battle_lr_scheduler" in state and self.battle_lr_scheduler is not None:
            self.battle_lr_scheduler.load_state_dict(state["battle_lr_scheduler"])
        if "preview_lr_scheduler" in state and self.preview_lr_scheduler is not None:
            self.preview_lr_scheduler.load_state_dict(state["preview_lr_scheduler"])

    def load_weights_only(self, state: dict):
        """Load only network weights (for frozen league opponents)."""
        self.battle_net.load_state_dict(state["battle_net"], strict=False)
        self.preview_net.load_state_dict(state["preview_net"], strict=False)

    def set_eval(self):
        self.battle_net.eval()
        self.preview_net.eval()

    def set_train(self):
        self.battle_net.train()
        self.preview_net.train()
