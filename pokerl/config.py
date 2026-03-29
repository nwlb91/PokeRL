"""Global configuration for PokeRL training."""

from dataclasses import dataclass
from typing import Optional


def _check_positive(name: str, value, allow_zero: bool = False):
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")


def _check_range(name: str, value, lo: float, hi: float):
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")


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
    lr_schedule: str = "cosine"  # "constant", "cosine", or "reduce_on_plateau"
    lr_min: float = 1e-5
    lr_warmup_battles: int = 1000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01  # used when entropy_anneal_battles == 0
    entropy_coef_start: float = 0.05
    entropy_coef_end: float = 0.005
    entropy_anneal_battles: int = 50000  # 0 = use static entropy_coef
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    rollout_steps: int = 256
    preview_rollout_steps: int = 32    # separate (lower) threshold for team preview updates
    preview_ppo_epochs: int = 8        # more epochs to extract signal from sparse preview data

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

    # --- Matchup-aware reward balancing ---
    matchup_reward_scaling: bool = True       # enable win-rate-aware reward rescaling
    reward_wr_ema_alpha: float = 0.01         # EMA smoothing for per-team win rate tracking
    underdog_shaping_boost: float = 3.0       # max multiplier on KO/damage weights for underdog
    underdog_wr_threshold: float = 0.3        # WR below which underdog boost activates
    matchup_conditioned_value: bool = True    # feed team WR EMA to value head as extra input

    # --- Evaluation ---
    greedy_eval_interval: int = 1000      # run greedy eval every N battles (0 to disable)
    greedy_eval_battles: int = 50         # number of evaluation battles per eval
    plateau_metric: str = "greedy_wr"     # metric for plateau detection: "greedy_wr", "train_wr", or "explained_variance"

    # --- Best-model tracking & regression protection ---
    best_model_tracking: bool = True
    regression_rollback_enabled: bool = False  # rollback on regression (harmful in self-play)
    regression_threshold: float = 0.08    # WR drop below best that triggers rollback
    regression_eval_window: int = 3       # consecutive bad evals before rollback

    # --- Baseline evaluation (absolute skill measure) ---
    baseline_eval_enabled: bool = True    # evaluate current agent vs frozen initial snapshot
    baseline_eval_interval: int = 0       # 0 = reuse greedy_eval_interval
    baseline_eval_battles: int = 50       # number of battles per baseline eval per team
    baseline_promotion_margin: float = 0.05  # WR improvement over baseline-vs-baseline needed for promotion

    # --- AlphaStar League ---
    league_size: int = 20  # max agents in the league
    checkpoint_interval: int = 50  # battles between checkpoints
    pfsp_temperature: float = 0.1  # temperature for PFSP opponent sampling
    main_agent_fraction: float = 0.5  # fraction of games vs latest opponent
    pfsp_fraction: float = 0.35  # fraction of games via PFSP
    self_play_fraction: float = 0.15  # fraction of self-play games

    # --- League admission & pruning ---
    league_min_size: int = 4  # min agents before admission gate activates
    league_admit_win_rate_delta: float = 0.05  # min |ΔWR| to trigger admission
    league_admit_param_novelty: float = 0.05  # min relative param distance for admission
    league_prune_interval: int = 200  # battles between pruning passes
    league_prune_similarity: float = 0.02  # relative param distance below which agents are redundant
    league_prune_stale_margin: float = 0.20  # WR margin above team average at which a league agent is stale
    payoff_decay: float = 0.995  # EMA decay factor for payoff matrix records

    # --- Uncertainty-weighted exploration ---
    uncertainty_heads: int = 3           # ensemble policy heads (1 = disabled, >1 = enabled)
    uncertainty_weight: float = 0.5      # logit bonus scaling for per-action uncertainty

    # --- Training ---
    total_battles: int = 100000
    infinite_training: bool = False       # ignore total_battles, train forever
    num_parallel_battles: int = 4
    battle_timeout: float = 300.0  # seconds; timeout per battle_against call
    device: str = "cpu"

    # --- Plateau response ---
    plateau_action: str = "entropy_bump"  # "entropy_bump", "noise_inject", or "none"
    plateau_entropy_bump: float = 0.03    # temporary entropy increase on plateau
    plateau_bump_duration: int = 2000     # battles to maintain the bump

    # --- Cloud / headless ---
    headless: bool = False
    log_file: str = ""

    # --- Checkpoints ---
    checkpoint_dir: str = "checkpoints"
    resume: bool = False
    resume_path: Optional[str] = None
    fresh_start: bool = False  # load weights from checkpoint but reset schedules & exploration

    # --- Showdown server ---
    server_url: str = "localhost"
    server_port: int = 8000

    def __post_init__(self):
        _check_positive("lr", self.lr)
        _check_positive("lr_min", self.lr_min, allow_zero=True)
        _check_range("gamma", self.gamma, 0.0, 1.0)
        _check_range("gae_lambda", self.gae_lambda, 0.0, 1.0)
        _check_range("clip_eps", self.clip_eps, 0.0, 1.0)
        _check_positive("entropy_coef", self.entropy_coef, allow_zero=True)
        _check_positive("value_coef", self.value_coef)
        _check_positive("max_grad_norm", self.max_grad_norm)
        _check_positive("ppo_epochs", self.ppo_epochs)
        _check_positive("batch_size", self.batch_size)
        _check_positive("rollout_steps", self.rollout_steps)
        _check_positive("hidden_size", self.hidden_size)
        _check_positive("num_layers", self.num_layers)
        _check_positive("total_battles", self.total_battles)
        _check_positive("battle_timeout", self.battle_timeout)
        _check_range("main_agent_fraction", self.main_agent_fraction, 0.0, 1.0)
        _check_range("pfsp_fraction", self.pfsp_fraction, 0.0, 1.0)
        _check_range("self_play_fraction", self.self_play_fraction, 0.0, 1.0)
        if self.lr_schedule not in ("constant", "cosine", "reduce_on_plateau"):
            raise ValueError(
                f"lr_schedule must be 'constant', 'cosine', or 'reduce_on_plateau', "
                f"got {self.lr_schedule!r}"
            )

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
