#!/usr/bin/env python3
"""PokeRL Training Script.

Trains RL agents to battle Pokemon teams against each other using:
  - PPO with action masking for battle decisions
  - Learned team preview (lead selection)
  - Win probability estimator for reward shaping
  - AlphaStar League-style training to prevent blind spots

Usage:
    python train.py --team1 teams/team1.txt --team2 teams/team2.txt
    python train.py --resume  # resume from latest checkpoint
    python train.py --resume-path checkpoints/checkpoint_001000.pt
"""

import argparse
import asyncio
import logging
import sys

from pokerl.config import Config
from pokerl.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Pokemon battle RL agents with AlphaStar League training"
    )

    # Team files
    parser.add_argument(
        "--team1", default="teams/team1.txt",
        help="Path to team 1 file (Showdown format)"
    )
    parser.add_argument(
        "--team2", default="teams/team2.txt",
        help="Path to team 2 file (Showdown format)"
    )

    # Battle format
    parser.add_argument(
        "--format", default="gen9nationaldexmonotype",
        help="Battle format (default: gen9nationaldexmonotype)"
    )

    # Training parameters
    parser.add_argument("--total-battles", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--num-parallel-battles", type=int, default=4,
                        help="Concurrent battles via poke-env (default: 4)")

    # PPO
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)

    # Win probability & reward shaping
    parser.add_argument("--wp-reward-weight", type=float, default=0.5,
                        help="Weight of WP-shaped reward (0=sparse only, 1=WP only)")
    parser.add_argument("--ko-reward-weight", type=float, default=0.15,
                        help="Weight of KO differential in terminal reward")
    parser.add_argument("--damage-reward-weight", type=float, default=0.1,
                        help="Weight of damage differential in terminal reward")
    parser.add_argument("--survival-reward", type=float, default=0.005,
                        help="Per-turn survival bonus for mid-battle steps")

    # Matchup-aware reward balancing
    parser.add_argument("--no-matchup-reward-scaling", action="store_true",
                        help="Disable matchup-aware reward rescaling for underdog teams")
    parser.add_argument("--reward-wr-ema-alpha", type=float, default=0.01,
                        help="EMA smoothing for per-team win rate tracking")
    parser.add_argument("--underdog-shaping-boost", type=float, default=3.0,
                        help="Max multiplier on KO/damage weights for underdog teams")
    parser.add_argument("--underdog-wr-threshold", type=float, default=0.3,
                        help="WR below which underdog shaping boost activates")
    parser.add_argument("--no-matchup-conditioned-value", action="store_true",
                        help="Disable matchup-conditioned value head")

    # AlphaStar League
    parser.add_argument("--league-size", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--pfsp-temperature", type=float, default=0.1)

    # Checkpointing
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--resume-path", default=None,
                        help="Resume from specific checkpoint file")

    # Server
    parser.add_argument("--server-url", default="localhost")
    parser.add_argument("--server-port", type=int, default=8000)

    # Uncertainty-weighted exploration
    parser.add_argument("--uncertainty-heads", type=int, default=3,
                        help="Ensemble policy heads for uncertainty exploration (1=disabled)")
    parser.add_argument("--uncertainty-weight", type=float, default=0.5,
                        help="Logit bonus scaling for per-action uncertainty")

    # Entropy scheduling
    parser.add_argument("--entropy-start", type=float, default=0.05,
                        help="Initial entropy coefficient for annealing")
    parser.add_argument("--entropy-end", type=float, default=0.005,
                        help="Floor entropy coefficient for annealing")
    parser.add_argument("--entropy-anneal-battles", type=int, default=50000,
                        help="Battles over which to anneal entropy (0=static)")

    # Learning rate scheduling
    parser.add_argument("--lr-schedule", default="cosine",
                        choices=["constant", "cosine", "reduce_on_plateau"],
                        help="Learning rate schedule")
    parser.add_argument("--lr-min", type=float, default=1e-5,
                        help="Minimum learning rate")
    parser.add_argument("--lr-warmup-battles", type=int, default=1000,
                        help="LR warmup period in battles")

    # Best-model tracking
    parser.add_argument("--no-best-model-tracking", action="store_true",
                        help="Disable best-model checkpointing and regression rollback")
    parser.add_argument("--regression-threshold", type=float, default=0.08,
                        help="WR drop below best that triggers rollback")
    parser.add_argument("--regression-eval-window", type=int, default=3,
                        help="Consecutive bad evals before rollback")
    parser.add_argument("--regression-rollback", action="store_true",
                        help="Enable rollback to best model on sustained regression")

    # Infinite training & plateau response
    parser.add_argument("--infinite", action="store_true",
                        help="Train indefinitely (ignore --total-battles)")
    parser.add_argument("--plateau-action", default="entropy_bump",
                        choices=["entropy_bump", "noise_inject", "none"],
                        help="Action to take on learning plateau")
    parser.add_argument("--plateau-entropy-bump", type=float, default=0.03,
                        help="Temporary entropy increase on plateau")
    parser.add_argument("--plateau-bump-duration", type=int, default=2000,
                        help="Battles to maintain entropy bump")

    # Device
    parser.add_argument("--device", default="cpu",
                        help="Device for training (cpu/cuda)")

    # Logging & cloud
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--headless", action="store_true",
                        help="Headless mode for cloud/server deployment")
    parser.add_argument("--log-file", default="",
                        help="Path to log file (enables file logging)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.getLogger().addHandler(file_handler)

    # Build config
    config = Config(
        team1_path=args.team1,
        team2_path=args.team2,
        battle_format=args.format,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        lr_warmup_battles=args.lr_warmup_battles,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        entropy_coef_start=args.entropy_start,
        entropy_coef_end=args.entropy_end,
        entropy_anneal_battles=args.entropy_anneal_battles,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
        num_parallel_battles=args.num_parallel_battles,
        ppo_epochs=args.ppo_epochs,
        wp_reward_weight=args.wp_reward_weight,
        ko_reward_weight=args.ko_reward_weight,
        damage_reward_weight=args.damage_reward_weight,
        survival_reward_per_turn=args.survival_reward,
        matchup_reward_scaling=not args.no_matchup_reward_scaling,
        reward_wr_ema_alpha=args.reward_wr_ema_alpha,
        underdog_shaping_boost=args.underdog_shaping_boost,
        underdog_wr_threshold=args.underdog_wr_threshold,
        matchup_conditioned_value=not args.no_matchup_conditioned_value,
        best_model_tracking=not args.no_best_model_tracking,
        regression_threshold=args.regression_threshold,
        regression_eval_window=args.regression_eval_window,
        regression_rollback_enabled=args.regression_rollback,
        league_size=args.league_size,
        checkpoint_interval=args.checkpoint_interval,
        pfsp_temperature=args.pfsp_temperature,
        uncertainty_heads=args.uncertainty_heads,
        uncertainty_weight=args.uncertainty_weight,
        total_battles=args.total_battles,
        infinite_training=args.infinite,
        device=args.device,
        plateau_action=args.plateau_action,
        plateau_entropy_bump=args.plateau_entropy_bump,
        plateau_bump_duration=args.plateau_bump_duration,
        headless=args.headless,
        log_file=args.log_file,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume or args.resume_path is not None,
        resume_path=args.resume_path,
        server_url=args.server_url,
        server_port=args.server_port,
    )

    log = logging.getLogger(__name__)
    log.info(
        f"Config: format={config.battle_format}, gen={config.gen}, "
        f"action_size={config.action_size}, gimmicks={config.num_gimmicks}"
    )
    if config.uncertainty_heads > 1:
        log.info(
            f"Uncertainty-weighted exploration: "
            f"heads={config.uncertainty_heads}, weight={config.uncertainty_weight}"
        )

    # Create trainer and run
    trainer = Trainer(config)

    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
