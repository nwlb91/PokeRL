"""Tests for LSTM recurrence and Q-head search."""

import numpy as np
import pytest
import torch

from pokerl.config import Config
from pokerl.features import BATTLE_OBS_SIZE
from pokerl.models import PolicyValueNet, RecurrentPolicyValueNet


class TestRecurrentPolicyValueNet:
    """Tests for the LSTM-based policy-value network."""

    @pytest.fixture
    def net(self):
        return RecurrentPolicyValueNet(
            obs_size=BATTLE_OBS_SIZE,
            action_size=26,
            hidden_size=64,
            num_layers=2,
            lstm_hidden_size=64,
        )

    def test_forward_output_shapes(self, net):
        obs = torch.randn(4, BATTLE_OBS_SIZE)
        mask = torch.ones(4, 26)
        logits, value, q_values, hidden = net(obs, mask)
        assert logits.shape == (4, 26)
        assert value.shape == (4, 1)
        assert q_values is None  # q_head not enabled
        assert hidden[0].shape == (1, 4, 64)  # h
        assert hidden[1].shape == (1, 4, 64)  # c

    def test_hidden_state_changes_output(self, net):
        """Different hidden states should produce different outputs."""
        obs = torch.randn(1, BATTLE_OBS_SIZE)
        mask = torch.ones(1, 26)
        logits1, _, _, h1 = net(obs, mask)
        logits2, _, _, h2 = net(obs, mask, hidden_state=h1)
        # Second call uses updated hidden state -> different logits
        assert not torch.allclose(logits1, logits2)

    def test_initial_hidden(self, net):
        h, c = net.initial_hidden(batch_size=3)
        assert h.shape == (1, 3, 64)
        assert c.shape == (1, 3, 64)
        assert (h == 0).all()
        assert (c == 0).all()

    def test_forward_sequence(self, net):
        """Test processing a full episode sequence."""
        seq_len = 10
        obs_seq = torch.randn(seq_len, BATTLE_OBS_SIZE)
        mask_seq = torch.ones(seq_len, 26)
        logits, values, q_values = net.forward_sequence(obs_seq, mask_seq)
        assert logits.shape == (seq_len, 26)
        assert values.shape == (seq_len,)
        assert q_values is None

    def test_get_action_and_value_compat(self, net):
        """Test the compatibility method works like PolicyValueNet."""
        obs = torch.randn(4, BATTLE_OBS_SIZE)
        mask = torch.ones(4, 26)
        result = net.get_action_and_value(obs, mask)
        assert len(result) == 4  # action, log_prob, entropy, value
        action, log_prob, entropy, value = result
        assert action.shape == (4,)
        assert value.shape == (4,)


class TestRecurrentWithQHead:
    """Tests for LSTM + Q-head combination."""

    @pytest.fixture
    def net(self):
        return RecurrentPolicyValueNet(
            obs_size=BATTLE_OBS_SIZE,
            action_size=26,
            hidden_size=64,
            num_layers=2,
            lstm_hidden_size=64,
            q_head_enabled=True,
        )

    def test_q_values_returned(self, net):
        obs = torch.randn(4, BATTLE_OBS_SIZE)
        mask = torch.ones(4, 26)
        logits, value, q_values, hidden = net(obs, mask)
        assert q_values is not None
        assert q_values.shape == (4, 26)

    def test_q_values_in_sequence(self, net):
        obs_seq = torch.randn(10, BATTLE_OBS_SIZE)
        mask_seq = torch.ones(10, 26)
        logits, values, q_values = net.forward_sequence(obs_seq, mask_seq)
        assert q_values is not None
        assert q_values.shape == (10, 26)

    def test_get_action_and_value_returns_5(self, net):
        obs = torch.randn(4, BATTLE_OBS_SIZE)
        mask = torch.ones(4, 26)
        result = net.get_action_and_value(obs, mask)
        assert len(result) == 5  # action, log_prob, entropy, value, q_values


class TestPolicyValueNetQHead:
    """Tests for Q-head on the feedforward PolicyValueNet."""

    def test_q_head_disabled_by_default(self):
        net = PolicyValueNet(obs_size=BATTLE_OBS_SIZE, action_size=26)
        assert net.q_head is None

    def test_q_head_enabled(self):
        net = PolicyValueNet(obs_size=BATTLE_OBS_SIZE, action_size=26,
                             q_head_enabled=True)
        assert net.q_head is not None

    def test_forward_returns_q_values(self):
        net = PolicyValueNet(obs_size=BATTLE_OBS_SIZE, action_size=26,
                             q_head_enabled=True)
        obs = torch.randn(4, BATTLE_OBS_SIZE)
        mask = torch.ones(4, 26)
        logits, value, q_values = net(obs, mask)
        assert q_values is not None
        assert q_values.shape == (4, 26)

    def test_forward_no_q_head(self):
        net = PolicyValueNet(obs_size=BATTLE_OBS_SIZE, action_size=26)
        obs = torch.randn(4, BATTLE_OBS_SIZE)
        mask = torch.ones(4, 26)
        logits, value, q_values = net(obs, mask)
        assert q_values is None

    def test_q_values_masked(self):
        net = PolicyValueNet(obs_size=BATTLE_OBS_SIZE, action_size=26,
                             q_head_enabled=True)
        obs = torch.randn(1, BATTLE_OBS_SIZE)
        mask = torch.zeros(1, 26)
        mask[0, :3] = 1.0  # only 3 legal actions
        _, _, q_values = net(obs, mask)
        # Illegal actions should have very negative Q-values
        assert q_values[0, 3:].max().item() < -1e6

    def test_get_action_and_value_returns_5(self):
        net = PolicyValueNet(obs_size=BATTLE_OBS_SIZE, action_size=26,
                             q_head_enabled=True)
        obs = torch.randn(4, BATTLE_OBS_SIZE)
        mask = torch.ones(4, 26)
        result = net.get_action_and_value(obs, mask)
        assert len(result) == 5


class TestLSTMConfig:
    """Tests for LSTM and Q-head configuration."""

    def test_lstm_enabled_by_default(self):
        config = Config()
        assert config.use_lstm

    def test_q_head_enabled_by_default(self):
        config = Config()
        assert config.q_head_enabled

    def test_lstm_enabled(self):
        config = Config(use_lstm=True)
        assert config.use_lstm
        assert config.lstm_hidden_size == 256

    def test_q_head_config(self):
        config = Config(q_head_enabled=True, search_weight=2.0)
        assert config.q_head_enabled
        assert config.search_weight == 2.0
