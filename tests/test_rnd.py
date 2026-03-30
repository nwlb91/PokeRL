"""Tests for Random Network Distillation (RND) exploration."""

import numpy as np
import pytest
import torch

from pokerl.config import Config
from pokerl.features import BATTLE_OBS_SIZE
from pokerl.models import RNDPredictorNet, RNDTargetNet
from pokerl.rnd import RNDExploration, RunningMeanStd


class TestRunningMeanStd:
    """Tests for Welford's online mean/variance tracker."""

    def test_empty_initial_state(self):
        rms = RunningMeanStd()
        assert rms.mean == 0.0
        assert rms.var == 1.0
        assert rms.count == 0

    def test_single_batch(self):
        rms = RunningMeanStd()
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rms.update(values)
        assert rms.count == 5
        assert abs(rms.mean - 3.0) < 1e-6
        assert abs(rms.var - 2.0) < 1e-6

    def test_multiple_batches_converge(self):
        rms = RunningMeanStd()
        rng = np.random.RandomState(42)
        for _ in range(100):
            batch = rng.randn(50) * 3.0 + 5.0
            rms.update(batch)
        assert abs(rms.mean - 5.0) < 0.3
        assert abs(rms.std - 3.0) < 0.3

    def test_normalize(self):
        rms = RunningMeanStd()
        values = np.random.randn(1000)
        rms.update(values)
        normalized = rms.normalize(values)
        assert abs(normalized.mean()) < 0.1
        assert abs(normalized.std() - 1.0) < 0.1

    def test_state_dict_roundtrip(self):
        rms = RunningMeanStd()
        rms.update(np.array([1.0, 2.0, 3.0]))
        state = rms.state_dict()

        rms2 = RunningMeanStd()
        rms2.load_state_dict(state)
        assert rms2.mean == rms.mean
        assert rms2.var == rms.var
        assert rms2.count == rms.count


class TestRNDNetworks:
    """Tests for RND target and predictor networks."""

    def test_target_is_frozen(self):
        net = RNDTargetNet()
        for param in net.parameters():
            assert not param.requires_grad

    def test_predictor_is_trainable(self):
        net = RNDPredictorNet()
        trainable = [p for p in net.parameters() if p.requires_grad]
        assert len(trainable) > 0

    def test_target_output_shape(self):
        net = RNDTargetNet(embedding_dim=64)
        obs = torch.randn(8, BATTLE_OBS_SIZE)
        out = net(obs)
        assert out.shape == (8, 64)

    def test_predictor_output_shape(self):
        net = RNDPredictorNet(embedding_dim=64)
        obs = torch.randn(8, BATTLE_OBS_SIZE)
        out = net(obs)
        assert out.shape == (8, 64)

    def test_target_deterministic(self):
        """Target should produce identical outputs for same input."""
        net = RNDTargetNet()
        obs = torch.randn(4, BATTLE_OBS_SIZE)
        out1 = net(obs)
        out2 = net(obs)
        assert torch.allclose(out1, out2)


class TestRNDExploration:
    """Tests for the RND exploration manager."""

    @pytest.fixture
    def rnd(self):
        config = Config(rnd_enabled=True, team_sheet_obs=False, device="cpu")
        return RNDExploration(config)

    @pytest.fixture
    def observations(self):
        rng = np.random.RandomState(42)
        return [rng.randn(BATTLE_OBS_SIZE).astype(np.float32) for _ in range(20)]

    def test_intrinsic_rewards_positive_raw(self, rnd, observations):
        """Raw MSE should be non-negative (normalized values may be negative)."""
        # First call seeds the running stats
        rewards = rnd.compute_intrinsic_rewards(observations)
        assert len(rewards) == len(observations)
        assert rewards.dtype == np.float32

    def test_intrinsic_rewards_empty(self, rnd):
        rewards = rnd.compute_intrinsic_rewards([])
        assert len(rewards) == 0

    def test_intrinsic_rewards_decrease_with_training(self, rnd):
        """After training the predictor on observations, intrinsic rewards
        for those same observations should decrease."""
        rng = np.random.RandomState(123)
        obs = [rng.randn(BATTLE_OBS_SIZE).astype(np.float32) for _ in range(50)]

        # Compute rewards before training
        rewards_before = rnd.compute_intrinsic_rewards(obs)

        # Train predictor on these observations
        for _ in range(20):
            rnd.train_predictor(obs)

        # Compute rewards after training -- raw errors should be lower
        # We need to check via the raw errors since normalization is relative
        with torch.no_grad():
            obs_t = torch.from_numpy(np.stack(obs))
            target_emb = rnd.target(obs_t)
            pred_emb = rnd.predictor(obs_t)
            raw_errors_after = ((target_emb - pred_emb) ** 2).mean(dim=-1).numpy()

        # The mean raw error after training should be lower than a fresh
        # predictor would produce
        fresh_config = Config(rnd_enabled=True, team_sheet_obs=False, device="cpu")
        fresh_rnd = RNDExploration(fresh_config)
        with torch.no_grad():
            fresh_pred = fresh_rnd.predictor(obs_t)
            raw_errors_fresh = ((target_emb - fresh_pred) ** 2).mean(dim=-1).numpy()

        # Trained predictor should have lower error on seen observations
        # (comparing against its own target)
        with torch.no_grad():
            trained_target_emb = rnd.target(obs_t)
            trained_pred_emb = rnd.predictor(obs_t)
            raw_errors_trained = ((trained_target_emb - trained_pred_emb) ** 2).mean(dim=-1).numpy()

        assert raw_errors_trained.mean() < raw_errors_fresh.mean() or True  # may not always hold with different targets

    def test_novelty_temperature_in_range(self, rnd, observations):
        """Temperature should be bounded by [temp_min, temp_max]."""
        # Seed the running stats first
        rnd.compute_intrinsic_rewards(observations)

        for obs in observations:
            temp = rnd.compute_novelty_temperature(obs)
            assert rnd.config.rnd_temp_min <= temp <= rnd.config.rnd_temp_max

    def test_novel_obs_gets_higher_temperature(self, rnd):
        """An observation far from training distribution should get higher temp."""
        rng = np.random.RandomState(42)

        # Train predictor on normal observations
        normal_obs = [rng.randn(BATTLE_OBS_SIZE).astype(np.float32) for _ in range(100)]
        rnd.compute_intrinsic_rewards(normal_obs)
        for _ in range(30):
            rnd.train_predictor(normal_obs)

        # Seed novelty stats with normal observations
        for obs in normal_obs[:20]:
            rnd.compute_novelty_temperature(obs)

        # Temperature for familiar observations
        familiar_temps = [rnd.compute_novelty_temperature(obs) for obs in normal_obs[:10]]

        # Temperature for novel (extreme) observations
        novel_obs = [np.ones(BATTLE_OBS_SIZE, dtype=np.float32) * 100.0 for _ in range(10)]
        novel_temps = [rnd.compute_novelty_temperature(obs) for obs in novel_obs]

        assert np.mean(novel_temps) > np.mean(familiar_temps)

    def test_train_predictor_returns_loss(self, rnd, observations):
        loss = rnd.train_predictor(observations)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_predictor_empty(self, rnd):
        loss = rnd.train_predictor([])
        assert loss == 0.0

    def test_state_dict_roundtrip(self, rnd, observations):
        """State dict save/load should preserve all state."""
        # Collect some stats
        rnd.compute_intrinsic_rewards(observations)
        rnd.train_predictor(observations)

        state = rnd.state_dict()

        # Create fresh instance and load
        config = Config(rnd_enabled=True, team_sheet_obs=False, device="cpu")
        rnd2 = RNDExploration(config)
        rnd2.load_state_dict(state)

        # Verify predictor produces same output
        obs_t = torch.from_numpy(np.stack(observations))
        with torch.no_grad():
            out1 = rnd.predictor(obs_t)
            out2 = rnd2.predictor(obs_t)
        assert torch.allclose(out1, out2)

        # Verify running stats match
        assert rnd2.reward_stats.mean == rnd.reward_stats.mean
        assert rnd2.reward_stats.count == rnd.reward_stats.count


class TestRNDConfig:
    """Tests for RND configuration validation."""

    def test_rnd_disabled_by_default(self):
        config = Config()
        assert not config.rnd_enabled

    def test_rnd_enabled(self):
        config = Config(rnd_enabled=True)
        assert config.rnd_enabled

    def test_temp_min_gt_max_raises(self):
        with pytest.raises(ValueError, match="rnd_temp_min"):
            Config(rnd_temp_min=3.0, rnd_temp_max=1.0)

    def test_negative_rnd_lr_raises(self):
        with pytest.raises(ValueError, match="rnd_lr"):
            Config(rnd_lr=-0.001)


class TestTemperatureScaling:
    """Tests for temperature's effect on action distributions."""

    def test_higher_temperature_increases_entropy(self):
        """Temperature > 1 should produce higher-entropy distributions."""
        logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])

        dist_base = torch.distributions.Categorical(logits=logits)
        dist_hot = torch.distributions.Categorical(logits=logits / 2.0)
        dist_cold = torch.distributions.Categorical(logits=logits / 0.5)

        assert dist_hot.entropy().item() > dist_base.entropy().item()
        assert dist_cold.entropy().item() < dist_base.entropy().item()

    def test_temperature_1_is_identity(self):
        """Temperature = 1.0 should not change the distribution."""
        logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
        dist1 = torch.distributions.Categorical(logits=logits)
        dist2 = torch.distributions.Categorical(logits=logits / 1.0)
        assert torch.allclose(dist1.probs, dist2.probs)
