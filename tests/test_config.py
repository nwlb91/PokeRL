"""Tests for Config hyperparameter validation."""

import pytest
from pokerl.config import Config


class TestConfigDefaults:
    """Default config should be valid."""

    def test_default_config_is_valid(self):
        config = Config()
        assert config.lr > 0
        assert config.action_size > 0

    def test_gen_property(self):
        config = Config(battle_format="gen9nationaldexmonotype")
        assert config.gen == 9

    def test_gen_property_other_gens(self):
        for g in range(1, 10):
            config = Config(battle_format=f"gen{g}ou")
            assert config.gen == g

    def test_action_size_gen9(self):
        config = Config(battle_format="gen9ou")
        # 6 switches + 4 moves * (1 + 4 gimmicks) = 26
        assert config.action_size == 26

    def test_action_size_gen5(self):
        config = Config(battle_format="gen5ou")
        # 6 switches + 4 moves * (1 + 0 gimmicks) = 10
        assert config.action_size == 10


class TestConfigValidation:
    """Config should reject invalid hyperparameters."""

    def test_negative_lr(self):
        with pytest.raises(ValueError, match="lr"):
            Config(lr=-1e-4)

    def test_zero_lr(self):
        with pytest.raises(ValueError, match="lr"):
            Config(lr=0)

    def test_gamma_out_of_range(self):
        with pytest.raises(ValueError, match="gamma"):
            Config(gamma=1.5)

    def test_gamma_negative(self):
        with pytest.raises(ValueError, match="gamma"):
            Config(gamma=-0.1)

    def test_clip_eps_out_of_range(self):
        with pytest.raises(ValueError, match="clip_eps"):
            Config(clip_eps=2.0)

    def test_invalid_lr_schedule(self):
        with pytest.raises(ValueError, match="lr_schedule"):
            Config(lr_schedule="invalid")

    def test_valid_lr_schedules(self):
        for sched in ("constant", "cosine", "reduce_on_plateau"):
            config = Config(lr_schedule=sched)
            assert config.lr_schedule == sched

    def test_zero_hidden_size(self):
        with pytest.raises(ValueError, match="hidden_size"):
            Config(hidden_size=0)

    def test_zero_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            Config(batch_size=0)

    def test_negative_entropy_coef(self):
        with pytest.raises(ValueError, match="entropy_coef"):
            Config(entropy_coef=-0.01)

    def test_zero_entropy_coef_allowed(self):
        config = Config(entropy_coef=0.0)
        assert config.entropy_coef == 0.0

    def test_fractions_out_of_range(self):
        with pytest.raises(ValueError, match="main_agent_fraction"):
            Config(main_agent_fraction=1.5)
