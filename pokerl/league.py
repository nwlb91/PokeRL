"""AlphaStar League-style training system.

Implements a simplified version of AlphaStar's league training:

1. **Main Agents**: Two actively training agents (one per team). They play
   against each other and against historical league opponents.

2. **League**: A pool of frozen agent checkpoints. As main agents improve,
   periodic snapshots are added to the league.

3. **Payoff Matrix**: Tracks win rates between all pairs of agents in the
   league, used for Prioritized Fictitious Self-Play (PFSP).

4. **PFSP Opponent Selection**: When choosing a league opponent, we
   prioritize opponents that the current agent struggles against, preventing
   blind spots.

Opponent selection distribution per battle:
  - main_agent_fraction: play against the other team's latest agent
  - exploiter_fraction: play against a PFSP-selected historical opponent
  - self_play_fraction: play against own team's historical checkpoint
"""

import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pokerl.agent import PPOAgent
from pokerl.config import Config


class LeagueAgent:
    """A frozen agent snapshot in the league."""

    def __init__(self, agent_id: str, team_id: int, state_dict: dict,
                 checkpoint_path: str, battle_count: int):
        self.agent_id = agent_id
        self.team_id = team_id  # 0 or 1
        self.state_dict = state_dict
        self.checkpoint_path = checkpoint_path
        self.battle_count = battle_count


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
    """AlphaStar-style league for training diverse agents."""

    def __init__(self, config: Config):
        self.config = config
        self.agents: List[LeagueAgent] = []  # frozen snapshots
        self.payoff = PayoffMatrix()
        self.checkpoint_dir = Path(config.checkpoint_dir) / "league"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._next_id = 0

    def add_agent(self, agent: PPOAgent, team_id: int) -> str:
        """Snapshot a training agent and add it to the league.

        Args:
            agent: The agent to snapshot.
            team_id: Which team this agent plays (0 or 1).

        Returns:
            The league agent's ID.
        """
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
        )
        self.agents.append(league_agent)

        # Trim league if too large (keep most recent + diverse set)
        if len(self.agents) > self.config.league_size:
            self._trim_league()

        return agent_id

    def select_opponent(
        self, current_agent_id: str, current_team_id: int
    ) -> Optional[LeagueAgent]:
        """Select a league opponent using the mixed strategy.

        The opponent is selected from the OTHER team's agents (so team1 agents
        fight team2 agents).

        Returns:
            A LeagueAgent to use as opponent, or None if league is empty.
        """
        opponent_team_id = 1 - current_team_id
        opponents = [a for a in self.agents if a.team_id == opponent_team_id]

        if not opponents:
            return None

        roll = random.random()

        if roll < self.config.main_agent_fraction:
            # Play against the latest opponent
            return opponents[-1]
        elif roll < self.config.main_agent_fraction + self.config.exploiter_fraction:
            # PFSP: prioritize hard opponents
            return self._pfsp_select(current_agent_id, opponents)
        else:
            # Self-play: play against own team's historical checkpoint
            own_agents = [a for a in self.agents if a.team_id == current_team_id]
            if own_agents:
                return random.choice(own_agents)
            return opponents[-1]

    def _pfsp_select(
        self, current_agent_id: str, opponents: List[LeagueAgent]
    ) -> LeagueAgent:
        """Prioritized Fictitious Self-Play opponent selection.

        Agents that the current agent has a LOWER win rate against are
        sampled with HIGHER probability (to fix weaknesses).

        Uses softmax with negative win rates:
            p(opponent) \u221d exp(-win_rate / temperature)
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

    def _trim_league(self):
        """Remove oldest agents when league exceeds max size.

        Keeps the most recent agent per team and a diverse set based on
        spacing throughout training history.
        """
        max_size = self.config.league_size

        # Always keep the latest agent for each team
        latest = {}
        for agent in reversed(self.agents):
            if agent.team_id not in latest:
                latest[agent.team_id] = agent

        # Keep evenly spaced agents from history
        keep = set(a.agent_id for a in latest.values())
        remaining = [a for a in self.agents if a.agent_id not in keep]

        # Evenly sample from remaining
        n_to_keep = max_size - len(keep)
        if n_to_keep > 0 and remaining:
            step = max(1, len(remaining) // n_to_keep)
            for i in range(0, len(remaining), step):
                keep.add(remaining[i].agent_id)
                if len(keep) >= max_size:
                    break

        self.agents = [a for a in self.agents if a.agent_id in keep]

    def get_team_agents(self, team_id: int) -> List[LeagueAgent]:
        """Get all league agents for a specific team."""
        return [a for a in self.agents if a.team_id == team_id]

    def get_state_dict(self) -> dict:
        return {
            "next_id": self._next_id,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "team_id": a.team_id,
                    "checkpoint_path": a.checkpoint_path,
                    "battle_count": a.battle_count,
                }
                for a in self.agents
            ],
            "payoff": self.payoff.get_state_dict(),
        }

    def load_state_dict(self, state: dict):
        self._next_id = state["next_id"]
        self.payoff.load_state_dict(state["payoff"])

        self.agents = []
        for agent_data in state["agents"]:
            cp_path = agent_data["checkpoint_path"]
            if os.path.exists(cp_path):
                saved_state = torch.load(cp_path, map_location="cpu", weights_only=True)
                league_agent = LeagueAgent(
                    agent_id=agent_data["agent_id"],
                    team_id=agent_data["team_id"],
                    state_dict=saved_state,
                    checkpoint_path=cp_path,
                    battle_count=agent_data["battle_count"],
                )
                self.agents.append(league_agent)
