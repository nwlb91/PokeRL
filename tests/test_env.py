"""Tests for environment utilities."""

import os
import tempfile

import pytest

from pokerl.env import load_team


class TestLoadTeam:
    def test_load_valid_team(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Pikachu @ Light Ball\nAbility: Static\n")
            f.flush()
            team = load_team(f.name)
        os.unlink(f.name)
        assert "Pikachu" in team

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Team file not found"):
            load_team("/nonexistent/path/team.txt")

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
        try:
            with pytest.raises(ValueError, match="empty"):
                load_team(f.name)
        finally:
            os.unlink(f.name)

    def test_directory_not_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="not a file"):
                load_team(tmpdir)
