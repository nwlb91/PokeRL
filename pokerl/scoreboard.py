"""Elo-style scoreboard using Bradley-Terry model.

Maintains separate per-team leaderboards that track model strength over
training. Match results are persisted to JSONL files so the leaderboard
can be reconstructed on resume.

Bradley-Terry MLE:
    r_i = W_i / sum_j(N_ij / (r_i + r_j))
    First player is pinned at 1000 after each iteration.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ScoreboardEntry:
    """A single entry on the leaderboard."""
    name: str
    battle_count: int  # games trained at time of snapshot
    rating: float = 1000.0
    checkpoint_path: str = ""


class Scoreboard:
    """Per-team Elo leaderboard using Bradley-Terry ratings.

    Each team has its own scoreboard. Team1's scoreboard evaluates
    agent1 (team1) vs checkpoint agent2 (team2), and vice versa.
    """

    def __init__(self, team_id: int, checkpoint_dir: str,
                 eval_interval: int = 1000, eval_games: int = 20):
        self.team_id = team_id
        self.entries: List[ScoreboardEntry] = []
        self.checkpoint_dir = Path(checkpoint_dir)
        self.match_log_path = self.checkpoint_dir / f"elo_matches_team{team_id + 1}.jsonl"
        # (player_a, player_b) -> [a_wins, b_wins]
        self.matches: Dict[Tuple[str, str], List[int]] = {}
        self.eval_interval = eval_interval
        self.eval_games = eval_games

    def add_initial(self, name: str, battle_count: int,
                    checkpoint_path: str) -> None:
        """Add the first entry, pinned at rating 1000."""
        entry = ScoreboardEntry(
            name=name,
            battle_count=battle_count,
            rating=1000.0,
            checkpoint_path=checkpoint_path,
        )
        self.entries = [entry]
        logger.info(
            "Scoreboard team%d: initial entry '%s' added at rating 1000",
            self.team_id + 1, name,
        )

    def record_match(self, player_a: str, player_b: str,
                     a_won: bool) -> None:
        """Record a single game result and append to the log file."""
        key = (player_a, player_b)
        if key not in self.matches:
            self.matches[key] = [0, 0]
        if a_won:
            self.matches[key][0] += 1
        else:
            self.matches[key][1] += 1
        self._append_to_log(player_a, player_b, a_won)

    def _record_match_no_log(self, player_a: str, player_b: str,
                             a_won: bool) -> None:
        """Record a match result without writing to the log file.

        Used during log replay to avoid re-writing entries.
        """
        key = (player_a, player_b)
        if key not in self.matches:
            self.matches[key] = [0, 0]
        if a_won:
            self.matches[key][0] += 1
        else:
            self.matches[key][1] += 1

    def compute_ratings(self) -> Dict[str, float]:
        """Compute Bradley-Terry MLE ratings from all accumulated matches.

        The first entry is always pinned at 1000. Returns a dict mapping
        player name to rating.
        """
        names = [e.name for e in self.entries]
        if len(names) <= 1:
            return {names[0]: 1000.0} if names else {}

        # Collect all player names that appear in match data
        all_players = set(names)
        for (a, b) in self.matches:
            all_players.add(a)
            all_players.add(b)

        # Only rate players that are on the leaderboard
        players = [n for n in all_players if n in set(names)]
        if not players:
            return {}

        # Ensure pinned player is first
        pinned = names[0]
        if pinned in players:
            players.remove(pinned)
            players.insert(0, pinned)

        ratings = {p: 1000.0 for p in players}

        # Precompute wins and game counts per pair
        wins: Dict[str, float] = {p: 0.0 for p in players}
        games: Dict[Tuple[str, str], int] = {}

        for (a, b), (a_wins, b_wins) in self.matches.items():
            if a not in ratings or b not in ratings:
                continue
            total = a_wins + b_wins
            if total == 0:
                continue
            wins[a] += a_wins
            wins[b] += b_wins
            games[(a, b)] = games.get((a, b), 0) + total
            games[(b, a)] = games.get((b, a), 0) + total

        # Iterative fixed-point updates
        for iteration in range(200):
            max_change = 0.0
            new_ratings = dict(ratings)

            for p in players:
                if p == pinned:
                    continue
                w = wins.get(p, 0.0)
                if w == 0:
                    new_ratings[p] = 1.0  # minimal rating for zero-win players
                    continue

                denom = 0.0
                for q in players:
                    if q == p:
                        continue
                    n_pq = games.get((p, q), 0)
                    if n_pq == 0:
                        continue
                    denom += n_pq / (ratings[p] + ratings[q])

                if denom > 0:
                    new_ratings[p] = w / denom
                else:
                    new_ratings[p] = ratings[p]

            # Rescale so pinned player = 1000
            scale = 1000.0 / max(new_ratings.get(pinned, 1000.0), 1e-8)
            for p in players:
                new_ratings[p] *= scale

            for p in players:
                max_change = max(max_change, abs(new_ratings[p] - ratings[p]))
            ratings = new_ratings

            if max_change < 0.1:
                break

        return ratings

    def should_add(self, current_rating: float) -> bool:
        """Return True if current_rating exceeds all existing entries."""
        if not self.entries:
            return True
        return current_rating > max(e.rating for e in self.entries)

    def add_entry(self, name: str, battle_count: int,
                  checkpoint_path: str) -> None:
        """Add a new entry and recompute all ratings."""
        entry = ScoreboardEntry(
            name=name,
            battle_count=battle_count,
            checkpoint_path=checkpoint_path,
        )
        self.entries.append(entry)
        self._update_ratings()
        logger.info(
            "Scoreboard team%d: '%s' added (rating=%.1f, games_trained=%d, "
            "leaderboard_size=%d)",
            self.team_id + 1, name, entry.rating, battle_count,
            len(self.entries),
        )

    def _update_ratings(self) -> None:
        """Recompute ratings for all entries from match data."""
        ratings = self.compute_ratings()
        for entry in self.entries:
            if entry.name in ratings:
                entry.rating = ratings[entry.name]
        # First entry always stays at 1000
        if self.entries:
            self.entries[0].rating = 1000.0

    def _append_to_log(self, player_a: str, player_b: str,
                       a_won: bool) -> None:
        """Append a single match result to the JSONL log file."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        record = {"a": player_a, "b": player_b, "winner": "a" if a_won else "b"}
        with open(self.match_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load_from_log(self) -> bool:
        """Reconstruct matches and leaderboard from the JSONL log file.

        Returns True if data was loaded, False if no log exists.
        """
        if not self.match_log_path.exists():
            return False

        self.matches.clear()
        players_seen: List[str] = []  # ordered by first appearance

        with open(self.match_log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                a = record["a"]
                b = record["b"]
                a_won = record["winner"] == "a"
                self._record_match_no_log(a, b, a_won)
                if a not in players_seen:
                    players_seen.append(a)
                if b not in players_seen:
                    players_seen.append(b)

        if not players_seen:
            return False

        # Reconstruct leaderboard entries from checkpoint directory
        # The first player seen is always the initial entry
        self.entries = []
        for name in players_seen:
            # Try to find the checkpoint
            ckpt_path = self._find_checkpoint(name)
            entry = ScoreboardEntry(
                name=name,
                battle_count=self._extract_battle_count(name),
                checkpoint_path=ckpt_path,
            )
            self.entries.append(entry)

        # Now determine which entries actually belong on the leaderboard
        # by replaying the "add only if strongest" logic
        self._rebuild_leaderboard(players_seen)

        logger.info(
            "Scoreboard team%d: loaded %d entries from log (%d total matches)",
            self.team_id + 1, len(self.entries),
            sum(a + b for a, b in self.matches.values()),
        )
        return True

    def _rebuild_leaderboard(self, players_seen: List[str]) -> None:
        """Rebuild the leaderboard by replaying entry decisions.

        Walk through players in order. The first is always added. Each
        subsequent player is added only if its rating exceeds all current
        entries at that point.
        """
        if not players_seen:
            return

        candidates = list(players_seen)
        if not candidates:
            return

        # Start with just the first entry
        self.entries = [ScoreboardEntry(
            name=candidates[0],
            battle_count=self._extract_battle_count(candidates[0]),
            rating=1000.0,
            checkpoint_path=self._find_checkpoint(candidates[0]),
        )]

        for name in candidates[1:]:
            # Temporarily add to compute rating
            test_entry = ScoreboardEntry(
                name=name,
                battle_count=self._extract_battle_count(name),
                checkpoint_path=self._find_checkpoint(name),
            )
            self.entries.append(test_entry)
            ratings = self.compute_ratings()

            candidate_rating = ratings.get(name, 0.0)
            max_existing = max(
                ratings.get(e.name, 0.0)
                for e in self.entries if e.name != name
            )

            if candidate_rating > max_existing:
                # Keep it — update all ratings
                for e in self.entries:
                    if e.name in ratings:
                        e.rating = ratings[e.name]
                if self.entries:
                    self.entries[0].rating = 1000.0
            else:
                # Remove it
                self.entries = [e for e in self.entries if e.name != name]

    def _find_checkpoint(self, name: str) -> str:
        """Find checkpoint path for a given entry name."""
        if name == "initial":
            path = self.checkpoint_dir / "initial_elo.pt"
            if path.exists():
                return str(path)
        # Try numbered checkpoint pattern
        path = self.checkpoint_dir / f"{name}.pt"
        if path.exists():
            return str(path)
        # Try standard checkpoint naming
        bc = self._extract_battle_count(name)
        if bc > 0:
            path = self.checkpoint_dir / f"checkpoint_{bc:06d}.pt"
            if path.exists():
                return str(path)
        return ""

    @staticmethod
    def _extract_battle_count(name: str) -> int:
        """Extract battle count from entry name like 'ckpt_001000'."""
        if name == "initial":
            return 0
        try:
            # Handle "ckpt_NNNNNN" format
            parts = name.split("_")
            for part in reversed(parts):
                if part.isdigit():
                    return int(part)
        except (ValueError, IndexError):
            pass
        return 0

    def get_state_for_dashboard(self) -> List[dict]:
        """Return leaderboard as a list of dicts for JSON serialization."""
        # Sort by rating descending
        sorted_entries = sorted(self.entries, key=lambda e: e.rating,
                                reverse=True)
        return [
            {
                "rank": i + 1,
                "name": e.name,
                "rating": round(e.rating, 1),
                "battle_count": e.battle_count,
            }
            for i, e in enumerate(sorted_entries)
        ]

    def get_current_rating(self, current_name: str) -> float:
        """Compute the rating for a candidate (current model) without adding it.

        The candidate must already have match data recorded under
        *current_name*.
        """
        # Temporarily add entry, compute, then remove
        temp = ScoreboardEntry(name=current_name, battle_count=0)
        self.entries.append(temp)
        ratings = self.compute_ratings()
        self.entries.pop()
        return ratings.get(current_name, 0.0)
