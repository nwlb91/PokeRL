"""AlphaStar League-style training system.

Implements a simplified version of AlphaStar's league training:

1. **Main Agents**: Two actively training agents (one per team). They play
   against each other and against historical league opponents.

2. **League**: A pool of frozen agent checkpoints. As main agents improve,
   snapshots are added to the league *only if they pass admission criteria*
   (sufficient novelty or performance change).

3. **Payoff Matrix**: Tracks win rates between all pairs of agents in the
   league, used for Prioritized Fictitious Self-Play (PFSP).

4. **PFSP Opponent Selection**: When choosing a league opponent, we
   prioritize opponents that the current agent struggles against, preventing
   blind spots.

5. **Pruning**: Periodically removes redundant agents — those that are
   superseded by similar-but-stronger agents — keeping the league lean and
   relevant.

Opponent selection distribution per battle:
  - main_agent_fraction: play against the other team's latest agent
  - exploiter_fraction: play against a PFSP-selected historical opponent
  - self_play_fraction: play against own team's historical checkpoint
"""

import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pokerl.agent import PPOAgent
from pokerl.config import Config

logger = logging.getLogger(__name__)


class LeagueAgent:
    """A frozen agent snapshot in the league.

    Weights are lazy-loaded from disk on first access and can be evicted
    via :meth:`unload` to reclaim RAM when the league grows large.
    """

    def __init__(self, agent_id: str, team_id: int, state_dict: Optional[dict],
                 checkpoint_path: str, battle_count: int,
                 win_rate_at_snapshot: float = 0.5):
        self.agent_id = agent_id
        self.team_id = team_id  # 0 or 1
        self._state_dict: Optional[dict] = state_dict
        self.checkpoint_path = checkpoint_path
        self.battle_count = battle_count
        self.win_rate_at_snapshot = win_rate_at_snapshot
        self.selection_count = 0  # how often selected as opponent

    @property
    def state_dict(self) -> dict:
        """Lazy-load weights from disk if not already cached."""
        if self._state_dict is None:
            self._state_dict = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=True
            )
        return self._state_dict

    def unload(self):
        """Free cached weights to reclaim RAM."""
        self._state_dict = None


class PayoffMatrix:
    """Tracks win rates between agent pairs."""

    def __init__(self):
        # (agent_id_a, agent_id_b) -> (wins_a, total_games)
        self._records: Dict[Tuple[str, str], Tuple[int, int]] = {}

    def record_result(self, agent_a: str, agent_b: str, a_won: bool):
        """Record a game result."""
        key = (agent_a, agent_b)
        wins, total = self._records.get(key, (0, 0))
        self._records[key] = (wins + int(a_won), total + 1)

        # Also record from b's perspective
        key_b = (agent_b, agent_a)
        wins_b, total_b = self._records.get(key_b, (0, 0))
        self._records[key_b] = (wins_b + int(not a_won), total_b + 1)

    def get_win_rate(self, agent_a: str, agent_b: str) -> float:
        """Get win rate of agent_a against agent_b."""
        key = (agent_a, agent_b)
        wins, total = self._records.get(key, (0, 0))
        if total == 0:
            return 0.5  # unknown matchup
        return wins / total

    def get_state_dict(self) -> dict:
        return {"records": dict(self._records)}

    def load_state_dict(self, state: dict):
        self._records = {
            tuple(k) if isinstance(k, list) else k: tuple(v)
            for k, v in state["records"].items()
        }


class League:
    """AlphaStar-style league for training diverse agents.

    Agents are admitted based on novelty (parameter divergence from existing
    members) or performance change (win-rate shift since the last snapshot).
    Redundant agents — those superseded by similar-but-stronger members — are
    periodically pruned so the league stays lean and relevant.
    """

    def __init__(self, config: Config):
        self.config = config
        self.agents: List[LeagueAgent] = []  # frozen snapshots
        self.payoff = PayoffMatrix()
        self.checkpoint_dir = Path(config.checkpoint_dir) / "league"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._next_id = 0
        self._last_prune_battle = 0

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def add_agent(self, agent: PPOAgent, team_id: int,
                  current_win_rate: float = 0.5) -> Optional[str]:
        """Snapshot a training agent and add it to the league *if* it passes
        the admission gate.

        Admission is granted when any of the following hold:
        - The team has fewer agents than ``league_min_size // 2`` (cold start).
        - The main agent's win rate has shifted by at least
          ``league_admit_win_rate_delta`` since the last same-team admission.
        - The agent's parameters are at least ``league_admit_param_novelty``
          (relative L2) away from every existing same-team member.

        Returns:
            The league agent's ID if admitted, ``None`` otherwise.
        """
        team_agents = self.get_team_agents(team_id)

        # Always admit during cold start
        if len(team_agents) < self.config.league_min_size // 2:
            return self._admit_agent(agent, team_id, current_win_rate)

        # Gate 1: Win-rate delta
        last_wr = team_agents[-1].win_rate_at_snapshot
        wr_delta = abs(current_win_rate - last_wr)
        if wr_delta >= self.config.league_admit_win_rate_delta:
            return self._admit_agent(agent, team_id, current_win_rate)

        # Gate 2: Parameter novelty
        candidate_state = agent.get_state_dict()
        min_dist = self._min_param_distance(candidate_state, team_agents)
        if min_dist >= self.config.league_admit_param_novelty:
            return self._admit_agent(agent, team_id, current_win_rate)

        logger.debug(
            f"League admission denied for team {team_id}: "
            f"wr_delta={wr_delta:.4f} (need {self.config.league_admit_win_rate_delta}), "
            f"param_novelty={min_dist:.4f} (need {self.config.league_admit_param_novelty})"
        )
        return None

    def _admit_agent(self, agent: PPOAgent, team_id: int,
                     win_rate: float) -> str:
        """Unconditionally add an agent snapshot to the league."""
        agent_id = f"league_{team_id}_{self._next_id:04d}"
        self._next_id += 1

        # Save checkpoint
        checkpoint_path = str(self.checkpoint_dir / f"{agent_id}.pt")
        state = agent.get_state_dict()
        torch.save(state, checkpoint_path)

        league_agent = LeagueAgent(
            agent_id=agent_id,
            team_id=team_id,
            state_dict=state,
            checkpoint_path=checkpoint_path,
            battle_count=agent.total_battles,
            win_rate_at_snapshot=win_rate,
        )
        self.agents.append(league_agent)

        # Hard cap — use quality-based trimming
        if len(self.agents) > self.config.league_size:
            self._trim_league()

        # Unload older agents' cached weights to reclaim RAM.
        # Keep only the two most recent per team loaded.
        self._evict_old_weights()

        logger.info(
            f"Agent {agent_id} admitted to league "
            f"(wr={win_rate:.1%}, league_size={len(self.agents)})"
        )
        return agent_id

    # ------------------------------------------------------------------
    # Opponent selection
    # ------------------------------------------------------------------

    def select_opponent(
        self, current_agent_id: str, current_team_id: int
    ) -> Optional[Tuple[LeagueAgent, str]]:
        """Select a league opponent using the mixed strategy.

        Returns:
            A (LeagueAgent, selection_type) tuple, where selection_type is one
            of "main", "pfsp", or "self_play".  Returns None if league is empty.
        """
        opponent_team_id = 1 - current_team_id
        opponents = [a for a in self.agents if a.team_id == opponent_team_id]

        if not opponents:
            return None

        roll = random.random()

        if roll < self.config.main_agent_fraction:
            selected = opponents[-1]
            kind = "main"
        elif roll < self.config.main_agent_fraction + self.config.exploiter_fraction:
            selected = self._pfsp_select(current_agent_id, opponents)
            kind = "pfsp"
        else:
            own_agents = [a for a in self.agents if a.team_id == current_team_id]
            if own_agents:
                selected = random.choice(own_agents)
                kind = "self_play"
            else:
                selected = opponents[-1]
                kind = "main"

        selected.selection_count += 1
        return selected, kind

    def select_pfsp_opponent(
        self, current_agent_id: str, current_team_id: int
    ) -> Optional[LeagueAgent]:
        """Select a PFSP opponent from the opposing team."""
        opponent_team_id = 1 - current_team_id
        opponents = [a for a in self.agents if a.team_id == opponent_team_id]
        if not opponents:
            return None
        selected = self._pfsp_select(current_agent_id, opponents)
        selected.selection_count += 1
        return selected

    def select_self_play_opponent(
        self, current_team_id: int
    ) -> Optional[LeagueAgent]:
        """Select a random historical agent from the same team."""
        own_agents = [a for a in self.agents if a.team_id == current_team_id]
        if not own_agents:
            return None
        selected = random.choice(own_agents)
        selected.selection_count += 1
        return selected

    def _pfsp_select(
        self, current_agent_id: str, opponents: List[LeagueAgent]
    ) -> LeagueAgent:
        """Prioritized Fictitious Self-Play opponent selection.

        Agents that the current agent has a LOWER win rate against are
        sampled with HIGHER probability (to fix weaknesses).

        Uses softmax with negative win rates:
            p(opponent) ∝ exp(-win_rate / temperature)
        """
        if len(opponents) == 1:
            return opponents[0]

        temp = self.config.pfsp_temperature
        win_rates = np.array([
            self.payoff.get_win_rate(current_agent_id, opp.agent_id)
            for opp in opponents
        ])

        # Higher weight for opponents we lose to more
        weights = np.exp(-win_rates / temp)
        # Also add a small uniform component for exploration
        weights = 0.9 * weights + 0.1 * np.ones_like(weights)
        probs = weights / weights.sum()

        idx = np.random.choice(len(opponents), p=probs)
        return opponents[idx]

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def maybe_prune(self, current_battle_count: int):
        """Run a pruning pass if enough battles have elapsed."""
        if (current_battle_count - self._last_prune_battle
                < self.config.league_prune_interval):
            return
        self._last_prune_battle = current_battle_count
        self._prune_redundant()

    def _prune_redundant(self):
        """Remove agents superseded by similar-but-stronger league members.

        For each pair of same-team agents whose relative parameter distance is
        below ``league_prune_similarity``, the lower-quality agent (by
        ``_agent_quality_score``) is removed.  The most recent agent per team
        is always protected.
        """
        if len(self.agents) <= self.config.league_min_size:
            return

        # Protect most recent per team
        protected = self._protected_agent_ids()
        to_remove: set = set()

        for team_id in range(2):
            team_agents = [a for a in self.agents if a.team_id == team_id]

            for agent in team_agents:
                if agent.agent_id in protected or agent.agent_id in to_remove:
                    continue

                for other in team_agents:
                    if other.agent_id == agent.agent_id or other.agent_id in to_remove:
                        continue

                    dist = self._param_distance(agent.state_dict, other.state_dict)
                    if dist < self.config.league_prune_similarity:
                        # Similar pair — remove the weaker one
                        if self._agent_quality_score(agent) <= self._agent_quality_score(other):
                            to_remove.add(agent.agent_id)
                            break

        # Never prune below minimum size
        max_removable = len(self.agents) - self.config.league_min_size
        if len(to_remove) > max_removable:
            # Keep the lowest-scored ones in to_remove, drop the rest
            scored = sorted(to_remove, key=lambda aid: self._score_by_id(aid))
            to_remove = set(scored[:max_removable])

        if to_remove:
            self.agents = [a for a in self.agents if a.agent_id not in to_remove]
            logger.info(
                f"Pruned {len(to_remove)} redundant league agents "
                f"(remaining: {len(self.agents)})"
            )

    def _agent_quality_score(self, agent: LeagueAgent) -> float:
        """Score an agent's overall quality (higher = more valuable).

        Combines:
        - Strength: win rate at time of snapshot.
        - Recency: newer agents are likelier to represent better strategies.
        - Utility: agents selected more often by PFSP are more useful.
        """
        strength = agent.win_rate_at_snapshot
        recency = agent.battle_count / max(1, self._max_battle_count())
        utility = math.log1p(agent.selection_count) / 10.0
        return 0.4 * strength + 0.4 * recency + 0.2 * utility

    def _score_by_id(self, agent_id: str) -> float:
        """Look up quality score by agent ID."""
        for a in self.agents:
            if a.agent_id == agent_id:
                return self._agent_quality_score(a)
        return 0.0

    def _max_battle_count(self) -> int:
        if not self.agents:
            return 1
        return max(a.battle_count for a in self.agents)

    # ------------------------------------------------------------------
    # Hard-cap trimming (quality-based)
    # ------------------------------------------------------------------

    def _trim_league(self):
        """Remove lowest-quality agents when the league exceeds its size cap.

        Unlike the old spacing-based approach, this scores every agent and
        removes the weakest ones (the most recent agent per team is always
        protected).
        """
        max_size = self.config.league_size
        protected = self._protected_agent_ids()

        # Score non-protected agents
        scored = [
            (self._agent_quality_score(a), a.agent_id)
            for a in self.agents
            if a.agent_id not in protected
        ]
        scored.sort()  # lowest score first

        n_to_remove = len(self.agents) - max_size
        to_remove = set(aid for _, aid in scored[:n_to_remove])

        self.agents = [a for a in self.agents if a.agent_id not in to_remove]

    def _evict_old_weights(self, keep_per_team: int = 2):
        """Unload cached weights for all but the *keep_per_team* most recent
        agents per team, freeing RAM while keeping frequently-used agents hot."""
        for team_id in range(2):
            team_agents = [a for a in self.agents if a.team_id == team_id]
            # Most recent are at the end of the list
            for agent in team_agents[:-keep_per_team]:
                agent.unload()

    def _protected_agent_ids(self) -> set:
        """IDs of the most recent agent per team (never pruned/trimmed)."""
        latest: Dict[int, str] = {}
        for agent in reversed(self.agents):
            if agent.team_id not in latest:
                latest[agent.team_id] = agent.agent_id
        return set(latest.values())

    # ------------------------------------------------------------------
    # Parameter-space utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_params(state_dict: dict) -> torch.Tensor:
        """Flatten all tensor parameters into a single vector."""
        tensors = []
        for key in sorted(state_dict.keys()):
            val = state_dict[key]
            if isinstance(val, torch.Tensor):
                tensors.append(val.flatten().float())
        if not tensors:
            return torch.zeros(1)
        return torch.cat(tensors)

    @classmethod
    def _param_distance(cls, state_a: dict, state_b: dict) -> float:
        """Relative L2 distance between two parameter sets.

        Computed as ``||a - b|| / ((||a|| + ||b||) / 2)``, giving a
        scale-invariant value in roughly [0, 2].
        """
        flat_a = cls._flatten_params(state_a)
        flat_b = cls._flatten_params(state_b)
        if flat_a.shape != flat_b.shape:
            return 2.0  # incompatible = maximally different
        diff_norm = torch.norm(flat_a - flat_b).item()
        scale = (torch.norm(flat_a).item() + torch.norm(flat_b).item()) / 2 + 1e-8
        return diff_norm / scale

    @classmethod
    def _min_param_distance(cls, candidate_state: dict,
                            agents: List[LeagueAgent]) -> float:
        """Minimum relative param distance from *candidate_state* to any agent."""
        if not agents:
            return float('inf')
        return min(
            cls._param_distance(candidate_state, a.state_dict)
            for a in agents
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_team_agents(self, team_id: int) -> List[LeagueAgent]:
        """Get all league agents for a specific team."""
        return [a for a in self.agents if a.team_id == team_id]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_state_dict(self) -> dict:
        return {
            "next_id": self._next_id,
            "last_prune_battle": self._last_prune_battle,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "team_id": a.team_id,
                    "checkpoint_path": a.checkpoint_path,
                    "battle_count": a.battle_count,
                    "win_rate_at_snapshot": a.win_rate_at_snapshot,
                    "selection_count": a.selection_count,
                }
                for a in self.agents
            ],
            "payoff": self.payoff.get_state_dict(),
        }

    def load_state_dict(self, state: dict):
        self._next_id = state["next_id"]
        self._last_prune_battle = state.get("last_prune_battle", 0)
        self.payoff.load_state_dict(state["payoff"])

        self.agents = []
        for agent_data in state["agents"]:
            cp_path = agent_data["checkpoint_path"]
            if os.path.exists(cp_path):
                # Weights are lazy-loaded on first access via LeagueAgent.state_dict
                league_agent = LeagueAgent(
                    agent_id=agent_data["agent_id"],
                    team_id=agent_data["team_id"],
                    state_dict=None,
                    checkpoint_path=cp_path,
                    battle_count=agent_data["battle_count"],
                    win_rate_at_snapshot=agent_data.get("win_rate_at_snapshot", 0.5),
                )
                league_agent.selection_count = agent_data.get("selection_count", 0)
                self.agents.append(league_agent)
