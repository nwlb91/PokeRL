"""PPO Agent for Pokemon battles.

Manages:
  - Battle policy (PolicyValueNet) for in-battle decisions
  - Team preview policy (TeamPreviewNet) for lead selection
  - Rollout buffer and PPO update logic
  - Win probability reward shaping integration
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from pokerl.config import Config
from pokerl.models import PolicyValueNet, RecurrentPolicyValueNet, TeamPreviewNet
from pokerl.features import (
    BATTLE_OBS_SIZE, BATTLE_OBS_SIZE_EXTENDED,
    TEAM_PREVIEW_OBS_SIZE, TEAM_PREVIEW_OBS_SIZE_EXTENDED,
)


@dataclass
class RolloutStep:
    obs: np.ndarray
    action: int
    action_mask: np.ndarray
    log_prob: float
    value: float
    reward: float
    done: bool
    matchup_context: Optional[np.ndarray] = None


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
        obs = np.stack([s.obs for s in self.steps])
        actions = np.array([s.action for s in self.steps], dtype=np.int64)
        masks = np.stack([s.action_mask for s in self.steps])
        log_probs = np.array([s.log_prob for s in self.steps], dtype=np.float32)
        old_values = np.array([s.value for s in self.steps], dtype=np.float32)

        # Matchup context (None if not used)
        has_ctx = self.steps[0].matchup_context is not None
        if has_ctx:
            ctx = np.stack([s.matchup_context for s in self.steps])
            ctx_t = torch.from_numpy(ctx).to(device)
        else:
            ctx_t = None

        return (
            torch.from_numpy(obs).to(device),
            torch.from_numpy(actions).to(device),
            torch.from_numpy(masks).to(device),
            torch.from_numpy(log_probs).to(device),
            torch.from_numpy(old_values).to(device),
            ctx_t,
        )


class PPOAgent:
    """PPO agent with separate battle and team preview networks."""

    def __init__(self, config: Config, agent_id: str = "agent"):
        self.config = config
        self.agent_id = agent_id
        self.device = torch.device(config.device)

        # Determine observation sizes
        battle_obs_size = BATTLE_OBS_SIZE_EXTENDED if config.team_sheet_obs else BATTLE_OBS_SIZE
        preview_obs_size = TEAM_PREVIEW_OBS_SIZE_EXTENDED if config.team_sheet_obs else TEAM_PREVIEW_OBS_SIZE

        # Battle policy network — recurrent or feedforward
        net_cls = RecurrentPolicyValueNet if config.use_lstm else PolicyValueNet
        net_kwargs = dict(
            obs_size=battle_obs_size,
            action_size=config.action_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            uncertainty_heads=config.uncertainty_heads,
            uncertainty_weight=config.uncertainty_weight,
            matchup_context_size=1 if config.matchup_conditioned_value else 0,
            q_head_enabled=config.q_head_enabled,
        )
        if config.use_lstm:
            net_kwargs["lstm_hidden_size"] = config.lstm_hidden_size
        self.battle_net = net_cls(**net_kwargs).to(self.device)
        self.use_lstm = config.use_lstm

        # Team preview network (always feedforward — one decision per battle)
        self.preview_net = TeamPreviewNet(
            obs_size=preview_obs_size,
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
                logger.info("torch.compile enabled for battle and preview networks")
            except (ImportError, RuntimeError) as e:
                logger.debug("torch.compile unavailable, using eager mode: %s", e)

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
        """Create an LR scheduler based on config, with optional linear warmup."""
        cfg = self.config

        # Estimate warmup length in optimizer steps.
        # Each PPO update processes ~rollout_steps / batch_size mini-batches,
        # and one update fires per ~rollout_steps / avg_episode_len battles.
        # Approximate: 1 update per battle (conservative).
        warmup_steps = max(1, cfg.lr_warmup_battles) if cfg.lr_warmup_battles > 0 else 0

        if cfg.lr_schedule == "cosine":
            # Use warm restarts for infinite training compatibility.
            # T_0 = estimated updates per entropy_anneal_battles cycle,
            # falling back to a reasonable default.
            updates_per_battle = 1.0 / max(1, cfg.rollout_steps // cfg.batch_size)
            t_0 = max(10, int(10000 * updates_per_battle))
            main_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=t_0, eta_min=cfg.lr_min,
            )
        elif cfg.lr_schedule == "reduce_on_plateau":
            main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=10, factor=0.5,
                min_lr=cfg.lr_min,
            )
        else:
            # "constant" — no main scheduler
            main_scheduler = None

        if warmup_steps > 0 and main_scheduler is not None:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-2, end_factor=1.0,
                total_iters=warmup_steps,
            )
            return optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps],
            )

        return main_scheduler

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
        self, obs: np.ndarray, action_mask: np.ndarray, deterministic: bool = False,
        matchup_context: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        hidden_state=None,
    ):
        """Select a battle action using the policy network.

        Args:
            obs: Battle observation.
            action_mask: Binary mask of legal actions.
            deterministic: If True, select the greedy action.
            matchup_context: Optional matchup conditioning for value head.
            temperature: Logit temperature for exploration.
            hidden_state: LSTM hidden state tuple (h, c) or None.

        Returns:
            (action, log_prob, value, new_hidden_state) if LSTM enabled
            (action, log_prob, value) otherwise
        """
        with torch.inference_mode():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
            ctx_t = None
            if matchup_context is not None:
                ctx_t = torch.from_numpy(matchup_context).unsqueeze(0).to(self.device)

            if self.use_lstm:
                logits, value, q_values, new_hidden = self.battle_net(
                    obs_t, mask_t, hidden_state=hidden_state,
                    deterministic=deterministic, matchup_context=ctx_t,
                )
            else:
                logits, value, q_values = self.battle_net(
                    obs_t, mask_t, deterministic=deterministic, matchup_context=ctx_t,
                )
                new_hidden = None

            # Build raw policy distribution (log_prob must come from this
            # to match what PPO recomputes in _ppo_update via get_action_and_value)
            raw_dist = torch.distributions.Categorical(logits=logits)

            # Blend policy logits with Q-values for search (affects sampling only)
            sampling_logits = logits
            if q_values is not None and not deterministic:
                sampling_logits = logits + self.config.search_weight * q_values

            # Apply temperature scaling for novelty-driven exploration
            if temperature != 1.0 and not deterministic:
                sampling_logits = sampling_logits / temperature

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                sampling_dist = torch.distributions.Categorical(logits=sampling_logits)
                action = sampling_dist.sample()

            # Log prob from raw policy -- matches what PPO recomputes
            log_prob = raw_dist.log_prob(action)

        if self.use_lstm:
            return (action.item(), log_prob.item(), value.squeeze().item(), new_hidden)
        return (action.item(), log_prob.item(), value.squeeze().item())

    def select_preview_action(
        self, obs: np.ndarray, mask: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Select a lead at team preview.

        Returns:
            (action, log_prob, value)
        """
        with torch.inference_mode():
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
        all_obs, all_actions, all_masks, all_old_lp, all_old_values, all_ctx = buffer.to_tensors(self.device)
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
                ctx_t = all_ctx[idx] if all_ctx is not None else None

                result = network.get_action_and_value(
                    obs_t, masks_t, actions_t, detach_uncertainty=True,
                    matchup_context=ctx_t,
                )
                # Unpack — Q-head returns 5 values, otherwise 4
                if len(result) == 5:
                    _, new_lp, entropy, values, q_values = result
                else:
                    _, new_lp, entropy, values = result
                    q_values = None

                # Policy loss (clipped PPO)
                ratio = torch.exp(new_lp - old_lp_t)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped to prevent destructive critic updates)
                values_clipped = old_val_t + torch.clamp(
                    values - old_val_t, -cfg.clip_eps, cfg.clip_eps
                )
                vl_unclipped = (values - returns_t) ** 2
                vl_clipped = (values_clipped - returns_t) ** 2
                value_loss = 0.5 * torch.max(vl_unclipped, vl_clipped).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Q-head loss: learn Q(s, a) ≈ return for the taken action
                q_loss = torch.tensor(0.0, device=self.device)
                if q_values is not None:
                    q_taken = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                    q_loss = ((q_taken - returns_t) ** 2).mean()

                # Total loss
                ent_coef = entropy_coef_override if entropy_coef_override is not None else cfg.entropy_coef
                loss = (policy_loss
                        + cfg.value_coef * value_loss
                        + ent_coef * entropy_loss
                        + cfg.q_value_coef * q_loss)

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
        # Optimizer/scheduler state may be incompatible if the network
        # architecture changed between checkpoint and current config.
        # Fall back to fresh optimizer state when that happens.
        if "battle_optimizer" in state:
            try:
                self.battle_optimizer.load_state_dict(state["battle_optimizer"])
            except (ValueError, RuntimeError):
                logger.warning(
                    "Battle optimizer state incompatible with current model — "
                    "using fresh optimizer (momentum/schedule will be reset)"
                )
        if "preview_optimizer" in state:
            try:
                self.preview_optimizer.load_state_dict(state["preview_optimizer"])
            except (ValueError, RuntimeError):
                logger.warning(
                    "Preview optimizer state incompatible with current model — "
                    "using fresh optimizer (momentum/schedule will be reset)"
                )
        self.total_battles = state.get("total_battles", 0)
        self.total_updates = state.get("total_updates", 0)
        self.wins = state.get("wins", 0)
        self.losses = state.get("losses", 0)
        if "battle_lr_scheduler" in state and self.battle_lr_scheduler is not None:
            try:
                self.battle_lr_scheduler.load_state_dict(state["battle_lr_scheduler"])
            except (ValueError, RuntimeError, KeyError):
                logger.warning("Battle LR scheduler state incompatible — using fresh scheduler")
        if "preview_lr_scheduler" in state and self.preview_lr_scheduler is not None:
            try:
                self.preview_lr_scheduler.load_state_dict(state["preview_lr_scheduler"])
            except (ValueError, RuntimeError, KeyError):
                logger.warning("Preview LR scheduler state incompatible — using fresh scheduler")

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
