"""Tests for the Elo scoreboard with Bradley-Terry ratings."""

import json
import tempfile
from pathlib import Path

import pytest

from pokerl.scoreboard import Scoreboard, ScoreboardEntry


class TestBradleyTerry:
    """Test Bradley-Terry rating computation."""

    def _make_scoreboard(self, tmpdir):
        return Scoreboard(team_id=0, checkpoint_dir=str(tmpdir))

    def test_single_entry_stays_at_1000(self, tmp_path):
        sb = self._make_scoreboard(tmp_path)
        sb.add_initial("initial", 0, str(tmp_path / "initial.pt"))
        ratings = sb.compute_ratings()
        assert ratings["initial"] == 1000.0

    def test_two_players_dominant(self, tmp_path):
        """Player B beats player A 18-2 → B should have higher rating."""
        sb = self._make_scoreboard(tmp_path)
        sb.entries = [
            ScoreboardEntry("A", 0, 1000.0),
            ScoreboardEntry("B", 100, 1000.0),
        ]
        for _ in range(18):
            sb._record_match_no_log("B", "A", a_won=True)
        for _ in range(2):
            sb._record_match_no_log("A", "B", a_won=True)

        ratings = sb.compute_ratings()
        assert ratings["A"] == 1000.0  # pinned
        assert ratings["B"] > ratings["A"]

    def test_equal_players_equal_ratings(self, tmp_path):
        sb = self._make_scoreboard(tmp_path)
        sb.entries = [
            ScoreboardEntry("A", 0, 1000.0),
            ScoreboardEntry("B", 100, 1000.0),
        ]
        for _ in range(50):
            sb._record_match_no_log("A", "B", a_won=True)
            sb._record_match_no_log("A", "B", a_won=False)

        ratings = sb.compute_ratings()
        assert abs(ratings["A"] - ratings["B"]) < 10.0

    def test_three_players_transitive(self, tmp_path):
        """A < B < C → ratings should reflect that ordering."""
        sb = self._make_scoreboard(tmp_path)
        sb.entries = [
            ScoreboardEntry("A", 0, 1000.0),
            ScoreboardEntry("B", 100, 1000.0),
            ScoreboardEntry("C", 200, 1000.0),
        ]
        # B beats A 15-5
        for _ in range(15):
            sb._record_match_no_log("B", "A", a_won=True)
        for _ in range(5):
            sb._record_match_no_log("A", "B", a_won=True)
        # C beats B 15-5
        for _ in range(15):
            sb._record_match_no_log("C", "B", a_won=True)
        for _ in range(5):
            sb._record_match_no_log("B", "C", a_won=True)
        # C beats A 18-2
        for _ in range(18):
            sb._record_match_no_log("C", "A", a_won=True)
        for _ in range(2):
            sb._record_match_no_log("A", "C", a_won=True)

        ratings = sb.compute_ratings()
        assert ratings["A"] == 1000.0
        assert ratings["B"] > ratings["A"]
        assert ratings["C"] > ratings["B"]

    def test_pinned_first_player(self, tmp_path):
        """First player always stays at 1000 regardless of results."""
        sb = self._make_scoreboard(tmp_path)
        sb.entries = [
            ScoreboardEntry("first", 0, 1000.0),
            ScoreboardEntry("second", 100, 1000.0),
        ]
        # second dominates first
        for _ in range(19):
            sb._record_match_no_log("second", "first", a_won=True)
        sb._record_match_no_log("first", "second", a_won=True)

        ratings = sb.compute_ratings()
        assert ratings["first"] == 1000.0
        assert ratings["second"] > 1000.0


class TestScoreboardEntryManagement:
    """Test leaderboard add/should_add logic."""

    def test_should_add_when_stronger(self, tmp_path):
        sb = Scoreboard(team_id=0, checkpoint_dir=str(tmp_path))
        sb.entries = [ScoreboardEntry("A", 0, 1000.0)]
        assert sb.should_add(1100.0) is True

    def test_should_not_add_when_weaker(self, tmp_path):
        sb = Scoreboard(team_id=0, checkpoint_dir=str(tmp_path))
        sb.entries = [ScoreboardEntry("A", 0, 1000.0)]
        assert sb.should_add(900.0) is False

    def test_should_not_add_when_equal(self, tmp_path):
        sb = Scoreboard(team_id=0, checkpoint_dir=str(tmp_path))
        sb.entries = [ScoreboardEntry("A", 0, 1000.0)]
        assert sb.should_add(1000.0) is False

    def test_add_entry_recomputes_ratings(self, tmp_path):
        sb = Scoreboard(team_id=0, checkpoint_dir=str(tmp_path))
        sb.add_initial("A", 0, str(tmp_path / "a.pt"))

        # Record matches showing B is stronger
        for _ in range(15):
            sb._record_match_no_log("B", "A", a_won=True)
        for _ in range(5):
            sb._record_match_no_log("A", "B", a_won=True)

        sb.add_entry("B", 100, str(tmp_path / "b.pt"))
        assert len(sb.entries) == 2
        assert sb.entries[0].rating == 1000.0  # A stays pinned
        assert sb.entries[1].rating > 1000.0  # B rated higher


class TestJSONLPersistence:
    """Test match log save and load."""

    def test_record_and_load(self, tmp_path):
        sb = Scoreboard(team_id=0, checkpoint_dir=str(tmp_path))
        sb.add_initial("A", 0, str(tmp_path / "a.pt"))

        sb.record_match("A", "B", a_won=True)
        sb.record_match("A", "B", a_won=False)
        sb.record_match("A", "B", a_won=True)

        assert sb.matches[("A", "B")] == [2, 1]

        # Verify file contents
        lines = sb.match_log_path.read_text().strip().split("\n")
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first == {"a": "A", "b": "B", "winner": "a"}

    def test_load_from_log_reconstructs(self, tmp_path):
        # Create a log file manually
        log_path = tmp_path / "elo_matches_team1.jsonl"
        # Create a dummy checkpoint file (just needs to exist)
        (tmp_path / "initial_elo.pt").write_bytes(b"dummy")

        records = [
            {"a": "initial", "b": "current_100", "winner": "b"},
            {"a": "initial", "b": "current_100", "winner": "b"},
            {"a": "initial", "b": "current_100", "winner": "a"},
        ]
        with open(log_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        sb = Scoreboard(team_id=0, checkpoint_dir=str(tmp_path))
        loaded = sb.load_from_log()
        assert loaded is True
        assert len(sb.entries) >= 1  # at least initial
        assert sb.entries[0].name == "initial"

    def test_get_state_for_dashboard(self, tmp_path):
        sb = Scoreboard(team_id=0, checkpoint_dir=str(tmp_path))
        sb.entries = [
            ScoreboardEntry("A", 0, 1000.0),
            ScoreboardEntry("B", 100, 1500.0),
            ScoreboardEntry("C", 200, 1200.0),
        ]
        state = sb.get_state_for_dashboard()
        assert len(state) == 3
        assert state[0]["rank"] == 1
        assert state[0]["name"] == "B"
        assert state[0]["rating"] == 1500.0
        assert state[1]["name"] == "C"
        assert state[2]["name"] == "A"


class TestGetCurrentRating:
    """Test computing a candidate's rating without adding it."""

    def test_current_rating_computation(self, tmp_path):
        sb = Scoreboard(team_id=0, checkpoint_dir=str(tmp_path))
        sb.entries = [ScoreboardEntry("A", 0, 1000.0)]

        # Record matches: current beats A 15-5
        for _ in range(15):
            sb._record_match_no_log("current_test", "A", a_won=True)
        for _ in range(5):
            sb._record_match_no_log("A", "current_test", a_won=True)

        rating = sb.get_current_rating("current_test")
        assert rating > 1000.0
        # Entry should not be permanently added
        assert len(sb.entries) == 1
