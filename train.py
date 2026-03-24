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

    # PPO
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)

    # Win probability
    parser.add_argument("--wp-reward-weight", type=float, default=0.5,
                        help="Weight of WP-shaped reward (0=sparse only, 1=WP only)")

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

    # Device
    parser.add_argument("--device", default="cpu",
                        help="Device for training (cpu/cuda)")

    # Logging
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def main():
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Build config
    config = Config(
        team1_path=args.team1,
        team2_path=args.team2,
        battle_format=args.format,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        wp_reward_weight=args.wp_reward_weight,
        league_size=args.league_size,
        checkpoint_interval=args.checkpoint_interval,
        pfsp_temperature=args.pfsp_temperature,
        total_battles=args.total_battles,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume or args.resume_path is not None,
        resume_path=args.resume_path,
        server_url=args.server_url,
        server_port=args.server_port,
    )

    logging.getLogger(__name__).info(
        f"Config: format={config.battle_format}, gen={config.gen}, "
        f"action_size={config.action_size}, gimmicks={config.num_gimmicks}"
    )

    # Create trainer and run
    trainer = Trainer(config)

    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
