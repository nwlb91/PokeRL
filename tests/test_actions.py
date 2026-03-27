"""Tests for action space management."""

import numpy as np

from pokerl.actions import team_preview_to_order
from pokerl.config import Config


class TestTeamPreviewOrder:
    """Test team_preview_to_order string generation."""

    def test_lead_first(self):
        """Lead should be the first number in the order string."""

        class FakeBattle:
            team = {f"mon{i}": None for i in range(6)}

        for lead in range(6):
            order_str = team_preview_to_order(lead, FakeBattle())
            assert order_str.startswith("/team ")
            digits = order_str.split(" ")[1]
            assert digits[0] == str(lead + 1)

    def test_all_team_members_present(self):
        """All team members should appear exactly once."""

        class FakeBattle:
            team = {f"mon{i}": None for i in range(6)}

        for lead in range(6):
            order_str = team_preview_to_order(lead, FakeBattle())
            digits = order_str.split(" ")[1]
            assert sorted(digits) == ["1", "2", "3", "4", "5", "6"]


class TestConfigActionSize:
    """Test action_size computation across generations."""

    def test_action_sizes_increase_with_gen(self):
        sizes = []
        for gen in range(1, 10):
            config = Config(battle_format=f"gen{gen}ou")
            sizes.append(config.action_size)
        # Action size should be non-decreasing with generation
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1]

    def test_all_gens_have_switches_and_moves(self):
        for gen in range(1, 10):
            config = Config(battle_format=f"gen{gen}ou")
            # At minimum: 6 switches + 4 moves = 10
            assert config.action_size >= 10
