"""Global configuration for PokeRL training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # --- Teams ---
    team1_path: str = "teams/team1.txt"
    team2_path: str = "teams/team2.txt"
    battle_format: str = "gen9nationaldexmonotype"

    # --- Network architecture ---
    hidden_size: int = 256
    num_layers: int = 3

    # --- PPO hyperparameters ---
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    rollout_steps: int = 256

    # --- Win probability estimator ---
    wp_hidden_size: int = 128
    wp_lr: float = 1e-3
    wp_buffer_size: int = 50000
    wp_batch_size: int = 256
    wp_train_interval: int = 10  # train every N battles
    wp_reward_weight: float = 0.5  # weight of WP-shaped reward vs sparse
    ko_reward_weight: float = 0.15  # weight for KO differential in terminal reward
    damage_reward_weight: float = 0.1  # weight for damage differential in terminal reward
    survival_reward_per_turn: float = 0.005  # small per-turn bonus for staying alive

    # --- Evaluation ---
    greedy_eval_interval: int = 200       # run greedy eval every N battles (0 to disable)
    greedy_eval_battles: int = 10         # number of deterministic battles per eval
    plateau_metric: str = "greedy_wr"     # metric for plateau detection: "greedy_wr", "train_wr", or "explained_variance"

    # --- AlphaStar League ---
    league_size: int = 20  # max agents in the league
    checkpoint_interval: int = 50  # battles between checkpoints
    pfsp_temperature: float = 0.1  # temperature for PFSP opponent sampling
    main_agent_fraction: float = 0.5  # fraction of games vs latest opponent
    exploiter_fraction: float = 0.35  # fraction of games via PFSP
    self_play_fraction: float = 0.15  # fraction of self-play games

    # --- League admission & pruning ---
    league_min_size: int = 4  # min agents before admission gate activates
    league_admit_win_rate_delta: float = 0.05  # min |ΔWR| to trigger admission
    league_admit_param_novelty: float = 0.05  # min relative param distance for admission
    league_prune_interval: int = 200  # battles between pruning passes
    league_prune_similarity: float = 0.02  # relative param distance below which agents are redundant

    # --- Uncertainty-weighted exploration ---
    uncertainty_heads: int = 1           # ensemble policy heads (1 = disabled, >1 = enabled)
    uncertainty_weight: float = 0.5      # logit bonus scaling for per-action uncertainty

    # --- Training ---
    total_battles: int = 100000
    num_parallel_battles: int = 1
    device: str = "cpu"

    # --- Checkpoints ---
    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    resume_path: Optional[str] = None

    # --- Showdown server ---
    server_url: str = "localhost"
    server_port: int = 8000

    @property
    def gen(self) -> int:
        for i in range(1, 10):
            if self.battle_format.startswith(f"gen{i}"):
                return i
        return 9

    @property
    def num_gimmicks(self) -> int:
        g = self.gen
        if g <= 5:
            return 0
        elif g == 6:
            return 1  # mega
        elif g == 7:
            return 2  # mega + z
        elif g == 8:
            return 3  # mega + z + dynamax
        else:
            return 4  # mega + z + dynamax + tera

    @property
    def action_size(self) -> int:
        """6 switches + 4 moves * (1 + num_gimmicks)."""
        return 6 + 4 * (1 + self.num_gimmicks)

    @property
    def team_preview_action_size(self) -> int:
        """6 possible leads."""
        return 6
