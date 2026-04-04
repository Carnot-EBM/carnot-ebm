"""Tests for ARM-EBM bijection: per-token energy extraction from LLM logits.

Spec coverage: REQ-INFER-010, SCENARIO-INFER-011, SCENARIO-INFER-012,
SCENARIO-INFER-013
"""

from __future__ import annotations

import math

import pytest

from carnot.inference import (
    TokenEnergyAnalysis,
    analyze_token_energy,
    compute_sequence_energy,
    extract_token_rewards,
    extract_token_rewards_from_logprobs,
    identify_hallucination_tokens,
)


class TestExtractTokenRewards:
    """Tests for extract_token_rewards with full logit vectors.

    Spec: REQ-INFER-010, SCENARIO-INFER-011
    """

    def test_bijection_equals_logprob(self) -> None:
        """REQ-INFER-010: reward = logit(chosen) - logsumexp = log P(chosen).

        The core bijection identity: the reward at each position must exactly
        equal the log-probability of the chosen token.
        """
        # Simple 3-token vocabulary at one position.
        # logits = [2.0, 1.0, 0.5], chosen = token 0 (logit 2.0)
        logits = [[2.0, 1.0, 0.5]]
        token_ids = [0]

        rewards = extract_token_rewards(logits, token_ids)

        # Manual computation: logsumexp([2.0, 1.0, 0.5])
        lse = math.log(math.exp(2.0) + math.exp(1.0) + math.exp(0.5))
        expected = 2.0 - lse  # = log P(token 0)
        assert len(rewards) == 1
        assert rewards[0] == pytest.approx(expected, abs=1e-5)

    def test_confident_token_has_high_reward(self) -> None:
        """SCENARIO-INFER-011: high logit relative to others -> near-zero reward.

        When one token dominates the logit distribution, its reward (log-prob)
        should be close to 0 (meaning probability close to 1).
        """
        # Token 0 has logit 10, others have logit 0 -> token 0 is ~certain.
        logits = [[10.0, 0.0, 0.0, 0.0]]
        token_ids = [0]

        rewards = extract_token_rewards(logits, token_ids)

        # reward should be close to 0 (log of ~1.0)
        assert rewards[0] > -0.01

    def test_uncertain_token_has_low_reward(self) -> None:
        """SCENARIO-INFER-011: low logit relative to others -> negative reward.

        When the chosen token is unlikely (low logit relative to alternatives),
        its reward should be very negative.
        """
        # Token 3 has logit 0, but tokens 0-2 all have logit 5 -> token 3 is
        # very unlikely.
        logits = [[5.0, 5.0, 5.0, 0.0]]
        token_ids = [3]

        rewards = extract_token_rewards(logits, token_ids)

        # reward should be very negative (log of a small probability)
        assert rewards[0] < -3.0

    def test_multi_position_sequence(self) -> None:
        """REQ-INFER-010: works across multiple positions in a sequence."""
        logits = [
            [3.0, 1.0],  # position 0: choose token 0 (high logit)
            [1.0, 3.0],  # position 1: choose token 1 (high logit)
            [1.0, 3.0],  # position 2: choose token 0 (LOW logit)
        ]
        token_ids = [0, 1, 0]

        rewards = extract_token_rewards(logits, token_ids)

        assert len(rewards) == 3
        # Positions 0 and 1: chose the dominant token -> high reward
        assert rewards[0] > -1.0
        assert rewards[1] > -1.0
        # Position 2: chose the minority token -> low reward
        assert rewards[2] < -1.0

    def test_length_mismatch_raises(self) -> None:
        """REQ-INFER-010: mismatched logits and token_ids raises ValueError."""
        with pytest.raises(ValueError, match="logits has 2 positions"):
            extract_token_rewards([[1.0], [2.0]], [0])

    def test_uniform_logits_give_log_one_over_vocab(self) -> None:
        """REQ-INFER-010: uniform logits -> reward = log(1/vocab_size).

        When all logits are equal, every token has equal probability 1/V,
        so reward = log(1/V). This is a useful sanity check.
        """
        vocab_size = 100
        logits = [[1.0] * vocab_size]
        token_ids = [42]

        rewards = extract_token_rewards(logits, token_ids)

        expected = math.log(1.0 / vocab_size)
        assert rewards[0] == pytest.approx(expected, abs=1e-5)


class TestExtractTokenRewardsFromLogprobs:
    """Tests for the API-friendly logprob-based reward extraction.

    Spec: REQ-INFER-010
    """

    def test_identity_function(self) -> None:
        """REQ-INFER-010: logprobs ARE rewards (bijection identity)."""
        logprobs = [-0.1, -0.5, -2.3, -0.01]
        rewards = extract_token_rewards_from_logprobs(logprobs)
        assert rewards == logprobs

    def test_returns_new_list(self) -> None:
        """REQ-INFER-010: returns a new list, not a reference to input."""
        logprobs = [-0.1, -0.5]
        rewards = extract_token_rewards_from_logprobs(logprobs)
        rewards[0] = 999.0
        assert logprobs[0] == -0.1  # original unmodified

    def test_empty_input(self) -> None:
        """REQ-INFER-010: empty input returns empty list."""
        assert extract_token_rewards_from_logprobs([]) == []


class TestComputeSequenceEnergy:
    """Tests for sequence energy computation.

    Spec: REQ-INFER-010, SCENARIO-INFER-013
    """

    def test_energy_is_negative_sum(self) -> None:
        """SCENARIO-INFER-013: energy = -sum(rewards)."""
        rewards = [-0.1, -0.2, -0.3]
        energy = compute_sequence_energy(rewards)
        assert energy == pytest.approx(0.6, abs=1e-9)

    def test_confident_sequence_has_low_energy(self) -> None:
        """SCENARIO-INFER-013: confident sequence -> low energy."""
        confident = [-0.01, -0.02, -0.01]
        uncertain = [-3.0, -4.0, -5.0]
        assert compute_sequence_energy(confident) < compute_sequence_energy(uncertain)

    def test_empty_sequence_has_zero_energy(self) -> None:
        """REQ-INFER-010: empty sequence -> energy = 0."""
        assert compute_sequence_energy([]) == 0.0

    def test_perfect_confidence_has_zero_energy(self) -> None:
        """SCENARIO-INFER-013: all rewards = 0 -> energy = 0."""
        assert compute_sequence_energy([0.0, 0.0, 0.0]) == 0.0


class TestIdentifyHallucinationTokens:
    """Tests for hallucination token detection.

    Spec: REQ-INFER-010, SCENARIO-INFER-012
    """

    def test_flags_uncertain_tokens(self) -> None:
        """SCENARIO-INFER-012: tokens below threshold are flagged."""
        rewards = [-0.1, -3.0, -0.2, -5.0, -0.05]
        indices = identify_hallucination_tokens(rewards, threshold=-2.0)
        assert indices == [1, 3]

    def test_no_flags_when_all_confident(self) -> None:
        """SCENARIO-INFER-012: confident tokens are not flagged."""
        rewards = [-0.1, -0.2, -0.3]
        indices = identify_hallucination_tokens(rewards, threshold=-2.0)
        assert indices == []

    def test_custom_threshold(self) -> None:
        """REQ-INFER-010: threshold is configurable."""
        rewards = [-0.5, -1.5, -2.5]
        # Strict threshold: only flag very uncertain tokens
        assert identify_hallucination_tokens(rewards, threshold=-3.0) == []
        # Loose threshold: flag mildly uncertain tokens
        assert identify_hallucination_tokens(rewards, threshold=-1.0) == [1, 2]

    def test_empty_input(self) -> None:
        """REQ-INFER-010: empty input returns empty list."""
        assert identify_hallucination_tokens([]) == []


class TestAnalyzeTokenEnergy:
    """Tests for the full analysis pipeline.

    Spec: REQ-INFER-010, SCENARIO-INFER-011, SCENARIO-INFER-012,
    SCENARIO-INFER-013
    """

    def test_full_pipeline(self) -> None:
        """REQ-INFER-010: full analysis returns all derived quantities."""
        logprobs = [-0.1, -0.3, -4.5, -0.2, -0.1]
        analysis = analyze_token_energy(logprobs, threshold=-2.0)

        assert isinstance(analysis, TokenEnergyAnalysis)
        assert analysis.token_rewards == logprobs
        assert analysis.sequence_energy == pytest.approx(5.2, abs=1e-9)
        assert analysis.hallucination_indices == [2]
        assert analysis.mean_reward == pytest.approx(-1.04, abs=1e-9)
        assert analysis.min_reward == pytest.approx(-4.5, abs=1e-9)

    def test_empty_sequence(self) -> None:
        """REQ-INFER-010: empty sequence produces zero-valued analysis."""
        analysis = analyze_token_energy([])
        assert analysis.token_rewards == []
        assert analysis.sequence_energy == 0.0
        assert analysis.hallucination_indices == []
        assert analysis.mean_reward == 0.0
        assert analysis.min_reward == 0.0

    def test_bijection_consistency(self) -> None:
        """SCENARIO-INFER-011: rewards from logits match rewards from logprobs.

        The bijection identity means extract_token_rewards(logits, ids) should
        produce the same values as extract_token_rewards_from_logprobs(logprobs)
        when logprobs = log P(id | context).
        """
        # Construct logits and compute rewards both ways.
        logits = [[2.0, 1.0, 0.5], [0.0, 3.0, 1.0]]
        token_ids = [0, 1]

        rewards_from_logits = extract_token_rewards(logits, token_ids)

        # Compute logprobs manually for the same tokens.
        import math

        logprobs = []
        for logit_vec, tid in zip(logits, token_ids):
            lse = math.log(sum(math.exp(l) for l in logit_vec))
            logprobs.append(logit_vec[tid] - lse)

        rewards_from_logprobs = extract_token_rewards_from_logprobs(logprobs)

        # They should match (the bijection identity).
        for r1, r2 in zip(rewards_from_logits, rewards_from_logprobs):
            assert r1 == pytest.approx(r2, abs=1e-5)
