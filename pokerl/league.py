"""AlphaStar League-style training system.

Implements a simplified version of AlphaStar's league training:

1. **Main Agents**: Two actively training agents (one per team). They play
   against each other and against historical league opponents.

2. **League**: A pool of frozen agent checkpoints. Snapshots are admitted
   when they demonstrate *relative strength* (outperform same-team average),
   *exploit* the opposing team's best agent better than the team average,
   or offer *novel parameters* while at least matching the team average.

3. **Payoff Matrix**: Tracks EMA (exponential moving average) win rates
   between all pairs of agents, used for Prioritized Fictitious Self-Play
   (PFSP).  The EMA ensures recent performance is weighted more heavily
   than stale history.

4. **PFSP Opponent Selection**: When choosing a league opponent, we
   prioritize opponents that the current agent struggles against, preventing
   blind spots.

5. **Pruning**: Periodically removes redundant agents (superseded by
   similar-but-stronger members) and stale agents (dominated by all live
   agents) — unless a stale agent uniquely exploits another league member.

Opponent selection distribution per battle:
  - main_agent_fraction: play against the other team's latest (live) agent
  - pfsp_fraction: play against a PFSP-selected historical opponent
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

# Allow numpy scalars in torch.load(weights_only=True).
torch.serialization.add_safe_globals([np._core.multiarray.scalar])


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
        self.is_best = False  # tagged best agents are protected from pruning

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
    """Tracks win rates between agent pairs using exponential moving average.

    Each (agent_a, agent_b) pair stores ``(ema_win_rate, total_games)``.
    The EMA gives more weight to recent results, so PFSP selection reflects
    the current agent's strengths and weaknesses rather than stale history.
    """

    def __init__(self, decay: float = 0.995):
        # (agent_id_a, agent_id_b) -> (ema_win_rate, total_games)
        self._records: Dict[Tuple[str, str], Tuple[float, int]] = {}
        self.decay = decay

    def record_result(self, agent_a: str, agent_b: str, a_won: bool):
        """Record a game result, updating EMAs for both perspectives."""
        self._update_ema(agent_a, agent_b, float(a_won))
        # Enforce symmetry: reverse direction is always the complement
        # so that ema(a,b) + ema(b,a) == 1.0 after every update.
        fwd_key = (agent_a, agent_b)
        rev_key = (agent_b, agent_a)
        fwd_ema, _ = self._records[fwd_key]
        _, rev_total = self._records.get(rev_key, (0.5, 0))
        self._records[rev_key] = (1.0 - fwd_ema, rev_total + 1)

    def _update_ema(self, agent_a: str, agent_b: str, outcome: float):
        key = (agent_a, agent_b)
        ema, total = self._records.get(key, (0.5, 0))
        ema = self.decay * ema + (1.0 - self.decay) * outcome
        self._records[key] = (ema, total + 1)

    def get_win_rate(self, agent_a: str, agent_b: str) -> float:
        """Get EMA win rate of agent_a against agent_b."""
        key = (agent_a, agent_b)
        ema, total = self._records.get(key, (0.5, 0))
        if total == 0:
            return 0.5  # unknown matchup
        return ema

    def get_state_dict(self) -> dict:
        return {"records": dict(self._records), "decay": self.decay}

    def load_state_dict(self, state: dict):
        self.decay = state.get("decay", self.decay)
        self._records = {
            tuple(k) if isinstance(k, list) else k: (float(v[0]), int(v[1]))
            for k, v in state["records"].items()
        }


class League:
    """AlphaStar-style league for training diverse agents.

    Agents are admitted when they outperform their team's historical average
    (relative strength), exploit the opposing best agent better than the
    team average, or offer novel parameters while at least matching the team
    average.  All thresholds are relative so asymmetric matchups don't bias
    admission.  Redundant agents (parameter-similar to a stronger member) and
    stale agents (dominated by all live agents well above average, unless
    they uniquely exploit another league member) are periodically pruned.
    """

    def __init__(self, config: Config):
        self.config = config
        self.agents: List[LeagueAgent] = []  # frozen snapshots
        self.payoff = PayoffMatrix(decay=config.payoff_decay)
        self.checkpoint_dir = Path(config.checkpoint_dir) / "league"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._next_id = 0
        self._last_prune_battle = 0
        self.live_agent_ids: List[str] = []  # set by trainer for staleness checks

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def add_agent(self, agent: PPOAgent, team_id: int,
                  current_win_rate: float = 0.5,
                  exploit_win_rate: Optional[float] = None) -> Optional[str]:
        """Snapshot a training agent and add it to the league *if* it passes
        the admission gate.

        All thresholds are **relative** to the team's own history so that
        asymmetric matchups (where one team has a structural WR advantage)
        don't bias admission.

        Admission is granted when any of the following hold:
        - The team has fewer agents than ``league_min_size // 2`` (cold start).
        - The agent is relatively strong: its win rate exceeds the mean WR
          of existing same-team league agents by ``league_admit_win_rate_delta``.
        - The agent is a relatively strong exploiter: its WR against the
          opposing team's best agent exceeds the team's mean exploit WR by
          ``league_admit_win_rate_delta``.
        - The agent's parameters are novel (relative L2 ≥
          ``league_admit_param_novelty``) AND its win rate is at least as
          high as the team's mean WR.

        Args:
            agent: The live training agent to snapshot.
            team_id: Which team this agent belongs to (0 or 1).
            current_win_rate: The agent's recent overall win rate.
            exploit_win_rate: The agent's payoff win rate vs the opposing
                team's best league agent, if one exists.  ``None`` when there
                is no best agent to compare against.

        Returns:
            The league agent's ID if admitted, ``None`` otherwise.
        """
        team_agents = self.get_team_agents(team_id)

        # Always admit during cold start
        if len(team_agents) < self.config.league_min_size // 2:
            return self._admit_agent(agent, team_id, current_win_rate)

        delta = self.config.league_admit_win_rate_delta
        mean_wr = self._team_mean_win_rate(team_agents)

        # Gate 1: Relative strength — agent outperforms its team's average
        if current_win_rate >= mean_wr + delta:
            return self._admit_agent(agent, team_id, current_win_rate)

        # Gate 2: Relative exploitation — agent exploits the best opposing
        # agent better than its team's historical average against that agent
        if exploit_win_rate is not None:
            mean_exploit = self._team_mean_exploit_wr(team_agents)
            if exploit_win_rate >= mean_exploit + delta:
                return self._admit_agent(agent, team_id, current_win_rate)

        # Gate 3: Parameter novelty with a relative strength floor
        if current_win_rate >= mean_wr:
            candidate_state = agent.get_state_dict()
            min_dist = self._min_param_distance(candidate_state, team_agents)
            if min_dist >= self.config.league_admit_param_novelty:
                return self._admit_agent(agent, team_id, current_win_rate)
        else:
            min_dist = 0.0

        logger.debug(
            f"League admission denied for team {team_id}: "
            f"wr={current_win_rate:.4f} (team_mean={mean_wr:.4f}, "
            f"need +{delta}), exploit_wr={exploit_win_rate}, "
            f"param_novelty={min_dist:.4f} (need {self.config.league_admit_param_novelty})"
        )
        return None

    def _team_mean_win_rate(self, team_agents: List[LeagueAgent]) -> float:
        """Mean win_rate_at_snapshot for a list of team agents."""
        if not team_agents:
            return 0.5
        return sum(a.win_rate_at_snapshot for a in team_agents) / len(team_agents)

    def _team_mean_exploit_wr(self, team_agents: List[LeagueAgent]) -> float:
        """Mean payoff WR of *team_agents* against the opposing team's best agent.

        Returns 0.5 when there is insufficient data.
        """
        # Find the opposing best agent
        if not team_agents:
            return 0.5
        opp_team = 1 - team_agents[0].team_id
        best_agents = [a for a in self.agents if a.team_id == opp_team and a.is_best]
        if not best_agents:
            return 0.5
        best = best_agents[-1]

        wrs = []
        for a in team_agents:
            _, total = self.payoff._records.get(
                (a.agent_id, best.agent_id), (0.5, 0)
            )
            if total > 0:
                wrs.append(self.payoff.get_win_rate(a.agent_id, best.agent_id))
        return sum(wrs) / len(wrs) if wrs else 0.5

    def _live_mean_wr_vs_team(self, live_id: str, team_id: int,
                               exclude: set) -> float:
        """Mean payoff WR of a live agent against all league agents of *team_id*.

        Used to compute the baseline for relative staleness checks.
        """
        team_agents = [
            a for a in self.agents
            if a.team_id == team_id and a.agent_id not in exclude
        ]
        if not team_agents:
            return 0.5
        wrs = [
            self.payoff.get_win_rate(live_id, a.agent_id)
            for a in team_agents
        ]
        return sum(wrs) / len(wrs)

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

    def add_best_agent(self, agent: PPOAgent, team_id: int,
                       win_rate: float) -> str:
        """Add a confirmed-best agent to the league, bypassing the admission gate.

        Best agents are tagged so they are protected from pruning, ensuring the
        strongest known version of each team is always available as a PFSP target.
        """
        agent_id = f"best_{team_id}_{self._next_id:04d}"
        self._next_id += 1

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
        league_agent.is_best = True

        # Remove any previous best agent for this team to avoid accumulation
        self.agents = [
            a for a in self.agents
            if not (a.is_best and a.team_id == team_id)
        ]
        self.agents.append(league_agent)

        if len(self.agents) > self.config.league_size:
            self._trim_league()
        self._evict_old_weights()

        logger.info(
            f"Best agent {agent_id} added to league "
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
            of "main" or "pfsp".  Returns None if no opposing-team agents
            exist in the league.
        """
        opponent_team_id = 1 - current_team_id
        opponents = [a for a in self.agents if a.team_id == opponent_team_id]

        if not opponents:
            return None

        roll = random.random()

        if roll < self.config.main_agent_fraction:
            selected = opponents[-1]
            kind = "main"
        else:
            selected = self._pfsp_select(current_agent_id, opponents)
            kind = "pfsp"

        selected.selection_count += 1
        return selected, kind

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
        """Remove agents that are redundant or stale.

        Two pruning passes:

        1. **Similarity**: For each pair of same-team agents whose relative
           parameter distance is below ``league_prune_similarity``, the
           lower-quality agent is removed.

        2. **Staleness**: Agents that both live agents beat by at least
           ``league_prune_stale_margin`` above their average WR against the
           agent's team are removed — *unless* the agent is the strongest
           exploiter of some other league member (it uniquely provides
           training signal).

        The most recent agent per team and ``is_best``-tagged agents are
        always protected.
        """
        if len(self.agents) <= self.config.league_min_size:
            return

        # Protect most recent per team
        protected = self._protected_agent_ids()
        to_remove: set = set()

        # Pass 1: Similarity-based pruning
        for team_id in range(2):
            team_agents = [a for a in self.agents if a.team_id == team_id]

            for agent in team_agents:
                if agent.agent_id in protected or agent.agent_id in to_remove or agent.is_best:
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

        # Pass 2: Staleness-based pruning (relative to team average)
        # An agent is stale when every live agent beats it by a large margin
        # above the live agent's average WR against that team.
        if self.live_agent_ids:
            margin = self.config.league_prune_stale_margin
            for agent in self.agents:
                if (agent.agent_id in protected or agent.is_best
                        or agent.agent_id in to_remove):
                    continue

                # Check if all live agents dominate this league agent
                # relative to their average WR against the agent's team
                dominated = True
                for live_id in self.live_agent_ids:
                    mean_wr = self._live_mean_wr_vs_team(
                        live_id, agent.team_id, exclude=to_remove,
                    )
                    wr_vs_agent = self.payoff.get_win_rate(
                        live_id, agent.agent_id,
                    )
                    if wr_vs_agent < mean_wr + margin:
                        dominated = False
                        break

                if not dominated:
                    continue

                # Protect if this agent uniquely exploits another league member
                if self._is_unique_exploiter(agent, to_remove):
                    logger.debug(
                        f"Stale agent {agent.agent_id} kept: uniquely "
                        f"exploits another league member"
                    )
                    continue

                to_remove.add(agent.agent_id)
                logger.debug(
                    f"Stale agent {agent.agent_id} marked for removal "
                    f"(dominated by all live agents, margin={margin})"
                )

        # Never prune below minimum size
        max_removable = len(self.agents) - self.config.league_min_size
        if len(to_remove) > max_removable:
            # Keep the lowest-scored ones in to_remove, drop the rest
            scored = sorted(to_remove, key=lambda aid: self._score_by_id(aid))
            to_remove = set(scored[:max_removable])

        if to_remove:
            self.agents = [a for a in self.agents if a.agent_id not in to_remove]
            logger.info(
                f"Pruned {len(to_remove)} redundant/stale league agents "
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

    def _is_unique_exploiter(self, candidate: LeagueAgent,
                             already_removing: set) -> bool:
        """Return True if *candidate* has the highest payoff win rate against
        any other league agent (i.e. it uniquely exploits someone).

        An agent that is the best counter to another league member still
        provides valuable training signal even if the live agents dominate it.
        """
        remaining = [
            a for a in self.agents
            if a.agent_id != candidate.agent_id
            and a.agent_id not in already_removing
        ]
        if not remaining:
            return False

        for target in remaining:
            candidate_wr = self.payoff.get_win_rate(
                candidate.agent_id, target.agent_id
            )
            # Check if any other remaining agent beats this target more
            best_other_wr = max(
                (self.payoff.get_win_rate(other.agent_id, target.agent_id)
                 for other in remaining
                 if other.agent_id != target.agent_id),
                default=0.0,
            )
            if candidate_wr > best_other_wr and candidate_wr > 0.5:
                return True
        return False

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

        # Score non-protected agents (best agents are also protected)
        scored = [
            (self._agent_quality_score(a), a.agent_id)
            for a in self.agents
            if a.agent_id not in protected and not a.is_best
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
            "live_agent_ids": self.live_agent_ids,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "team_id": a.team_id,
                    "checkpoint_path": a.checkpoint_path,
                    "battle_count": a.battle_count,
                    "win_rate_at_snapshot": a.win_rate_at_snapshot,
                    "selection_count": a.selection_count,
                    "is_best": a.is_best,
                }
                for a in self.agents
            ],
            "payoff": self.payoff.get_state_dict(),
        }

    def load_state_dict(self, state: dict):
        self._next_id = state["next_id"]
        self._last_prune_battle = state.get("last_prune_battle", 0)
        self.live_agent_ids = state.get("live_agent_ids", [])
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
                league_agent.is_best = agent_data.get("is_best", False)
                self.agents.append(league_agent)
