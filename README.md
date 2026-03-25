# PokeRL - Pokemon Battle Reinforcement Learning

Train RL agents to battle two Pokemon teams against each other using PPO with AlphaStar League-style training.

## Features

- **Comprehensive battle state embedding** (1032-dim): encodes active Pokemon stats/types/status/boosts, all 4 moves with properties, bench Pokemon, opponent info, weather, terrain, side conditions, hazards, and battle mechanics flags
- **Full action space** (26 actions for gen9): 6 switches + 4 moves × 5 gimmick options (normal, mega evolve, z-move, dynamax, terastallize), with illegal action masking
- **Learned team preview**: separate neural network for lead selection (not random)
- **Win probability estimator**: auxiliary model providing dense reward shaping via win probability deltas between turns
- **AlphaStar League training**: maintains a pool of frozen agent checkpoints with Prioritized Fictitious Self-Play (PFSP) opponent selection to prevent strategy collapse and blind spots
- **Resumable training**: periodic checkpoints save all agent states, league, and win probability estimator

## Requirements

- Python 3.9+
- A running [Pokemon Showdown](https://github.com/smogon/pokemon-showdown) server
- Dependencies: `pip install -r requirements.txt`

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start a Pokemon Showdown server** (in a separate terminal):
   ```bash
   git clone https://github.com/smogon/pokemon-showdown.git
   cd pokemon-showdown
   npm install
   node pokemon-showdown start --no-security
   ```

3. **Add your teams** in Showdown format to `teams/team1.txt` and `teams/team2.txt`

4. **Train:**
   ```bash
   python train.py --team1 teams/team1.txt --team2 teams/team2.txt --format gen9nationaldexmonotype
   ```

5. **Resume training:**
   ```bash
   python train.py --resume
   # or from a specific checkpoint:
   python train.py --resume-path checkpoints/checkpoint_001000.pt
   ```

## Architecture

### Action Space (gen9)

| Action Range | Description |
|---|---|
| 0-5 | Switch to team slot 0-5 |
| 6-9 | Use move 1-4 |
| 10-13 | Use move 1-4 + Mega Evolve |
| 14-17 | Use move 1-4 + Z-Move |
| 18-21 | Use move 1-4 + Dynamax |
| 22-25 | Use move 1-4 + Terastallize |

Illegal actions are masked before sampling. Action space adapts to generation (e.g., gen6 has only mega evolve, gen7 adds z-moves, etc.).

### Observation Space (1032 features)

- Active Pokemon (69): HP, types, status, boosts, base stats, level, flags
- Move details (4×37=148): power, type, category, accuracy, PP, effects
- Full team bench (6×55=330): HP, types, status, stats, fainted
- Opponent active + team (69+330=399): same encoding with known info
- Field state (72): weather, terrain, side conditions (both sides)
- Battle flags (14): gimmick availability, force switch, turn count, etc.

### AlphaStar League

The training system maintains a pool of frozen agent checkpoints. Each battle, the opponent is selected by:
- **50%**: play against the latest opposing agent
- **35%**: PFSP selection (prioritize opponents the agent loses to)
- **15%**: self-play against own historical checkpoints

### Win Probability Estimator

A separate network predicts win probability from battle state. The reward signal blends:
- Sparse terminal reward (+1 win / -1 loss)
- Dense WP delta reward (change in predicted win probability)

Controlled by `--wp-reward-weight` (0 = sparse only, 1 = WP only, default 0.5).

## CLI Arguments

```
--team1, --team2          Team files (Showdown format)
--format                  Battle format (default: gen9nationaldexmonotype)
--total-battles           Total training battles (default: 100000)
--lr                      Learning rate (default: 3e-4)
--hidden-size             Network hidden size (default: 256)
--checkpoint-interval     Battles between checkpoints (default: 50)
--league-size             Max agents in league pool (default: 20)
--wp-reward-weight        Win probability reward weight (default: 0.5)
--device                  Training device (default: cpu)
--resume                  Resume from latest checkpoint
--resume-path             Resume from specific checkpoint
```

## Project Structure

```
PokeRL/
├── train.py                  # Main training script
├── requirements.txt
├── teams/
│   ├── team1.txt             # Team 1 (Showdown format)
│   └── team2.txt             # Team 2 (Showdown format)
├── pokerl/
│   ├── config.py             # Training configuration
│   ├── features.py           # Battle state feature extraction
│   ├── actions.py            # Action space + masking
│   ├── models.py             # Neural network architectures
│   ├── agent.py              # PPO agent implementation
│   ├── win_probability.py    # Win probability estimator
│   ├── league.py             # AlphaStar League system
│   ├── checkpoint.py         # Checkpoint save/load
│   ├── env.py                # poke-env integration (RLPlayer)
│   └── trainer.py            # Training orchestrator
└── checkpoints/              # Saved checkpoints (gitignored)
```
