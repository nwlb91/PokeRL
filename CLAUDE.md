# PokeRL — Project Documentation

This document describes the architecture, features, and conventions of the PokeRL project for future contributors and LLMs.

## Overview

PokeRL trains reinforcement learning agents to play Pokemon battles using PPO with AlphaStar League-style self-play. Two fixed teams battle each other, with agents learning both lead selection (team preview) and in-battle decisions (moves and switches).

## Entry Points

- **`gui.py`** — Tkinter GUI for training, evaluation, challenges, and server management. The primary way to run training.
- **`train.py`** — CLI training script with argparse. All config options are exposed as CLI flags.
- **`pokerl/trainer.py`** — Core training orchestrator. Both GUI and CLI instantiate `Trainer` and call `trainer.train()`.

## Architecture

### Observation Encoding (`pokerl/features.py`)

Two observation modes controlled by `Config.team_sheet_obs`:

**Standard (1032 dims)** — `BATTLE_OBS_SIZE`:
- Active pokemon (69): HP, types(40), status(7), boosts(7), base_stats(6), level, flags
- Active moves (4 x 37 = 148): power, type(20), category(3), accuracy, priority, PP, STAB, effects
- Team bench (6 x 55 = 330): HP, types, status, base_stats, fainted
- Opponent active (69) + opponent team (6 x 55 = 330)
- Weather(9) + fields(15) + side conditions(24+24) + battle flags(14)

**Extended with team sheets (2808 dims)** — `BATTLE_OBS_SIZE_EXTENDED`:
- Same as standard but bench pokemon use 203 dims each (55 base + 4 x 37 moves)
- Moves for bench pokemon come from pre-parsed team files (`pokerl/teamsheet.py`)
- Useful when both players know exact sets (the teams never change)

Team preview observations: `TEAM_PREVIEW_OBS_SIZE` (664) or `TEAM_PREVIEW_OBS_SIZE_EXTENDED` (2440).

### Action Space (`pokerl/actions.py`)

Discrete, generation-adaptive. For gen9: 26 actions.
- Indices 0-5: Switch to team slot
- Indices 6-9: Use move 1-4
- Indices 10-13: Move + Mega Evolve
- Indices 14-17: Move + Z-Move
- Indices 18-21: Move + Dynamax
- Indices 22-25: Move + Terastallize

Illegal actions are masked at the logit level: `logits + action_mask.log().clamp(min=-1e8)`.

### Neural Networks (`pokerl/models.py`)

**`BattleNetwork`** — Shared MLP backbone (Linear → LayerNorm → ReLU) x num_layers.

**`PolicyValueNet`** — Feedforward actor-critic for battle decisions.
- Backbone → policy head (with optional ensemble uncertainty heads) + value head (with optional matchup conditioning) + optional Q-head.
- `forward()` returns `(logits, value, q_values)`. Q-values are `None` when `q_head_enabled=False`.
- `get_action_and_value()` returns 4 values normally, 5 when Q-head is enabled.

**`RecurrentPolicyValueNet`** — LSTM variant of PolicyValueNet.
- Backbone → LSTM → policy/value/Q heads.
- `forward()` returns `(logits, value, q_values, new_hidden_state)`.
- `forward_sequence()` processes a full episode for efficient PPO recomputation.
- `get_action_and_value()` is a compatibility wrapper (uses zero hidden state).

**`TeamPreviewNet`** — Separate small network for lead selection. Always feedforward.

**`WinProbabilityNet`** — Auxiliary network predicting P(win) from battle state for reward shaping.

**`RNDTargetNet` / `RNDPredictorNet`** — Random Network Distillation networks for state novelty.

### PPO Agent (`pokerl/agent.py`)

**`PPOAgent`** manages:
- Battle network (feedforward or recurrent, chosen by `Config.use_lstm`)
- Team preview network (always feedforward)
- Separate optimizers and LR schedulers for each
- Rollout buffers for battle and team preview
- PPO update logic with clipped surrogate objective, clipped value loss, entropy bonus, and optional Q-head TD loss

**`RolloutBuffer`** stores `RolloutStep` dataclasses with obs, action, mask, log_prob, value, reward, done, and optional matchup_context.

**`select_battle_action()`** accepts optional `temperature` (for RND adaptive exploration) and `hidden_state` (for LSTM). Returns 3 values (feedforward) or 4 values (LSTM, includes new hidden state).

### Environment (`pokerl/env.py`)

**`RLPlayer`** extends poke-env's `Player`:
- `teampreview()` → lead selection via TeamPreviewNet
- `choose_move()` → action selection via battle network
- Per-battle state isolation via `_BattleEpisodeState` (supports concurrent battles)
- `finalize_episode()` flushes buffered steps to shared rollout buffer after reward shaping

External modules are attached by the trainer:
- `player.rnd` — RND exploration module
- `player.our_team_moves` / `player.opp_team_moves` — pre-parsed team move data
- `player.matchup_context` — team win rate EMA for value head conditioning

### Training Loop (`pokerl/trainer.py`)

**`Trainer`** orchestrates:
1. Alternating training: agent1 and agent2 take turns as the training agent
2. Opponent selection: 50% live opponent, 35% PFSP from league, 15% self-play
3. Episode finalization: WP reward shaping → RND intrinsic rewards → terminal reward scaling
4. PPO updates when rollout buffer is full
5. RND predictor training alongside PPO updates
6. Periodic: checkpoints, league snapshots, greedy evaluation, baseline evaluation, plateau detection
7. Elo scoreboard evaluation every `elo_eval_interval` battles

### Elo Scoreboard (`pokerl/scoreboard.py`)

Separate per-team leaderboards using **Bradley-Terry MLE** ratings:
- Team1 scoreboard: current agent1 (team1) vs checkpoint agent2 (team2)
- Team2 scoreboard: current agent2 (team2) vs checkpoint agent1 (team1)
- Initial checkpoint pinned at rating 1000; new entries added only if stronger than all existing
- All match results persisted to JSONL files (`checkpoints/elo_matches_team{1,2}.jsonl`) for resume
- Bradley-Terry iterative fixed-point: `r_i = W_i / sum_j(N_ij / (r_i + r_j))`, first player pinned at 1000

### AlphaStar League (`pokerl/league.py`)

Maintains a pool of frozen agent checkpoints with:
- PFSP opponent selection (prioritize opponents the agent loses to)
- Admission gates: BT rating strength, win-record exploitation, or parameter novelty
- `WinLossTracker`: raw W/L records between agent pairs for league eligibility
- Pruning: similarity-based and win-record staleness (agents with no winning matchup removed)
- Payoff matrix tracking with EMA decay

### Reward System

Terminal reward (`_compute_scaled_reward`):
- Base: +1/-1 scaled by team win rate EMA (rare wins amplified, expected losses dampened)
- KO differential bonus (weight: 0.15)
- Damage differential bonus (weight: 0.1)
- Underdog boost for teams with WR < 0.3

Per-step reward (`_apply_reward_shaping`):
- Win probability delta (potential-based reward shaping, weight: 0.5)
- Survival bonus per turn (0.005)
- RND intrinsic novelty reward (when enabled, annealed coefficient)

## Feature Flags

All advanced features are opt-in via `Config` (and exposed in GUI under "Advanced Features"):

| Feature | Config Flag | Default | Description |
|---------|------------|---------|-------------|
| Team sheet obs | `team_sheet_obs` | False | Encode full movesets for all bench pokemon (1032 → 2808 dims) |
| LSTM recurrence | `use_lstm` | False | Add LSTM between backbone and heads for cross-turn reasoning |
| LSTM hidden size | `lstm_hidden_size` | 256 | LSTM hidden state dimension |
| Q-head search | `q_head_enabled` | False | Action-value head for value-guided action selection |
| Search weight | `search_weight` | 1.0 | Blend weight of Q-values with policy logits |
| Q-value loss coef | `q_value_coef` | 0.25 | Q-head loss coefficient during PPO training |
| RND exploration | `rnd_enabled` | False | Random Network Distillation for state novelty rewards |
| RND coefficient | `rnd_coef` | 0.1 | Intrinsic reward scaling (anneals to `rnd_coef_end`) |
| RND adaptive temp | `rnd_adaptive_temp` | True | Scale action temperature by state novelty |
| Uncertainty heads | `uncertainty_heads` | 3 | Ensemble policy heads for uncertainty-weighted exploration |
| Matchup conditioning | `matchup_conditioned_value` | True | Feed team WR EMA to value head |
| Elo eval interval | `elo_eval_interval` | 1000 | Battles between Elo scoreboard evaluations |
| Elo eval games | `elo_eval_games` | 20 | Games per matchup during Elo evaluation |

## File Map

```
PokeRL/
├── gui.py                    # Tkinter GUI (training, eval, challenges, server)
├── train.py                  # CLI training entry point
├── CLAUDE.md                 # This file — project documentation
├── README.md                 # User-facing README
├── requirements.txt
├── teams/
│   ├── team1.txt             # Team 1 in Showdown paste format
│   └── team2.txt             # Team 2 in Showdown paste format
├── pokerl/
│   ├── config.py             # Config dataclass with all hyperparameters
│   ├── features.py           # Observation encoding (1032 or 2808 dims)
│   ├── actions.py            # Action space, masking, action↔order conversion
│   ├── models.py             # All neural networks (PolicyValueNet, RecurrentPolicyValueNet,
│   │                         #   TeamPreviewNet, WinProbabilityNet, RND nets)
│   ├── agent.py              # PPOAgent, RolloutBuffer, PPO update logic
│   ├── env.py                # RLPlayer (poke-env integration), episode management
│   ├── trainer.py            # Training orchestrator (battles, PPO, league, eval)
│   ├── league.py             # AlphaStar League (PFSP, admission, pruning, payoff)
│   ├── win_probability.py    # WP estimator for reward shaping
│   ├── rnd.py                # RND exploration (RunningMeanStd, RNDExploration)
│   ├── scoreboard.py         # Elo scoreboard with Bradley-Terry ratings
│   ├── teamsheet.py          # Showdown team file parser
│   ├── checkpoint.py         # Checkpoint save/load manager
│   └── plateau.py            # Plateau detection and response
├── tests/
│   ├── test_config.py        # Config validation tests
│   ├── test_features.py      # Observation encoding tests
│   ├── test_actions.py       # Action space tests
│   ├── test_env.py           # Environment tests
│   ├── test_rnd.py           # RND exploration tests
│   ├── test_teamsheet.py     # Team sheet parser + extended obs tests
│   └── test_lstm_qhead.py    # LSTM and Q-head tests
└── checkpoints/              # Saved training checkpoints (gitignored)
```

## Conventions

- **Backward compatibility**: All new features default to `False`/disabled. Old checkpoints load via `strict=False`.
- **Per-battle state isolation**: Each concurrent battle has its own `_BattleEpisodeState` to prevent data interleaving.
- **Reward shaping flow**: WP deltas → RND intrinsic → survival bonus → terminal reward. Applied in `Trainer._apply_reward_shaping()` and `Trainer._compute_scaled_reward()`.
- **Observation sizes are constants**: `BATTLE_OBS_SIZE`, `BATTLE_OBS_SIZE_EXTENDED`, etc. defined at module level in `features.py`. Networks accept `obs_size` as a constructor parameter.
- **Q-head return value**: `get_action_and_value()` returns 4 or 5 values depending on whether Q-head is enabled. Callers must check `len(result)`.
- **LSTM hidden state**: Managed per-battle in `_BattleEpisodeState.lstm_hidden`. Reset implicitly when a new episode state is created. `select_battle_action()` returns 3 or 4 values depending on `use_lstm`.
- **Tests**: Run with `pytest tests/`. All features have unit tests. Tests don't require a Showdown server.

## When Making Changes

1. **Adding a new config parameter**: Add to `Config` dataclass in `config.py`, add validation in `__post_init__`, expose in GUI's `_build_training_tab()` and `_make_config()`, and add to CLI in `train.py` if needed. Update this document.
2. **Changing observation size**: Update the relevant constant in `features.py`. All networks accept `obs_size` as a parameter. Also update `WinProbabilityEstimator` and `RNDExploration` which read the obs size from config.
3. **Adding a new network**: Add to `models.py`. Follow the `get_action_and_value()` interface pattern. Wire into `PPOAgent` in `agent.py`.
4. **Modifying rewards**: Terminal rewards in `Trainer._compute_scaled_reward()`, per-step shaping in `Trainer._apply_reward_shaping()`.
5. **Adding a new exploration method**: Follow the RND pattern — create a module, add config flags, integrate into `_apply_reward_shaping` (reward) and/or `env.py:choose_move` (action selection).
