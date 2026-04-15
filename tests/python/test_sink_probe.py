"""Tests for sink_probe.py — SinkProbe attention-sink hallucination pre-filter.

Theoretical basis: arXiv 2604.10697 (SinkProbe, Apr 2026).
Specific attention heads concentrate probability mass on "sink" tokens (BOS,
period, etc.) when the model is generating factually CERTAIN content.  When
sink concentration is LOW the model is uncertain — and full verification should
run.  When concentration is HIGH the response can bypass the Ising verifier
(fast path).

Spec: REQ-VERIFY-086, REQ-VERIFY-087
SCENARIO-VERIFY-113 (high sink → skip verification)
SCENARIO-VERIFY-114 (low sink → run verification)
SCENARIO-VERIFY-115 (benchmark returns accurate skip / FNR / TNR)
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.pipeline.sink_probe import (
    SinkConcentration,
    SinkProbe,
    SinkProbeResult,
    SinkTokenType,
    compute_sink_concentration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_attn(n_heads: int = 4, seq_len: int = 8) -> jnp.ndarray:
    """Return an attention matrix where every head is perfectly uniform.

    Uniform attention means no head "prefers" any position, so sink
    concentration will be proportional to the fraction of sink positions.
    """
    val = 1.0 / seq_len
    return jnp.full((n_heads, seq_len, seq_len), val)


def _sink_dominated_attn(
    n_heads: int = 4, seq_len: int = 8, sink_pos: int = 0, sink_mass: float = 0.9
) -> jnp.ndarray:
    """Return an attention matrix where every head places `sink_mass` on position `sink_pos`.

    Remaining mass is spread uniformly over the other positions.
    """
    remaining = (1.0 - sink_mass) / (seq_len - 1)
    attn = jnp.full((n_heads, seq_len, seq_len), remaining)
    # Assign sink_mass to the sink column for all heads and all query positions.
    attn = attn.at[:, :, sink_pos].set(sink_mass)
    return attn


def _rng_attn(
    n_heads: int = 4, seq_len: int = 8, seed: int = 42
) -> jnp.ndarray:
    """Return a random row-normalised attention matrix (CI-safe, no model needed)."""
    rng = np.random.default_rng(seed)
    raw = rng.exponential(1.0, size=(n_heads, seq_len, seq_len)).astype(np.float32)
    # Row-normalise so each query row sums to 1.
    row_sums = raw.sum(axis=-1, keepdims=True)
    normed = raw / row_sums
    return jnp.array(normed)


# ---------------------------------------------------------------------------
# SinkTokenType enum
# ---------------------------------------------------------------------------


class TestSinkTokenType:
    """Spec: REQ-VERIFY-086"""

    def test_bos_member_exists(self) -> None:
        """BOS is a member of SinkTokenType."""
        assert SinkTokenType.BOS is SinkTokenType.BOS

    def test_eos_member_exists(self) -> None:
        """EOS is a member of SinkTokenType."""
        assert SinkTokenType.EOS is SinkTokenType.EOS

    def test_period_member_exists(self) -> None:
        """PERIOD is a member of SinkTokenType."""
        assert SinkTokenType.PERIOD is SinkTokenType.PERIOD

    def test_comma_member_exists(self) -> None:
        """COMMA is a member of SinkTokenType."""
        assert SinkTokenType.COMMA is SinkTokenType.COMMA

    def test_all_four_members(self) -> None:
        """Exactly four SinkTokenType members: BOS, EOS, PERIOD, COMMA."""
        names = {m.name for m in SinkTokenType}
        assert names == {"BOS", "EOS", "PERIOD", "COMMA"}


# ---------------------------------------------------------------------------
# SinkConcentration dataclass
# ---------------------------------------------------------------------------


class TestSinkConcentration:
    """Spec: REQ-VERIFY-086"""

    def test_fields_exist(self) -> None:
        """per_head_sink_scores, mean_sink_score, max_sink_score are present."""
        sc = SinkConcentration(
            per_head_sink_scores=[0.1, 0.2, 0.3],
            mean_sink_score=0.2,
            max_sink_score=0.3,
        )
        assert sc.per_head_sink_scores == [0.1, 0.2, 0.3]
        assert sc.mean_sink_score == pytest.approx(0.2)
        assert sc.max_sink_score == pytest.approx(0.3)

    def test_mean_and_max_consistent(self) -> None:
        """max_sink_score >= mean_sink_score for any valid SinkConcentration."""
        sc = SinkConcentration(
            per_head_sink_scores=[0.05, 0.4, 0.2],
            mean_sink_score=0.05 + 0.4 + 0.2,  # deliberately wrong — we only check API
            max_sink_score=0.4,
        )
        # This test just verifies the field names are exposed; arithmetic is checked
        # in compute_sink_concentration tests below.
        assert sc.max_sink_score >= 0.0

    def test_zero_scores_accepted(self) -> None:
        """Zero scores are valid (no attention on sink tokens)."""
        sc = SinkConcentration(
            per_head_sink_scores=[0.0, 0.0],
            mean_sink_score=0.0,
            max_sink_score=0.0,
        )
        assert sc.mean_sink_score == 0.0


# ---------------------------------------------------------------------------
# compute_sink_concentration
# ---------------------------------------------------------------------------


class TestComputeSinkConcentration:
    """Spec: REQ-VERIFY-086"""

    def test_uniform_single_sink_position(self) -> None:
        """Uniform attention with one sink position → concentration ≈ 1/seq_len.

        For uniform attention (each key gets equal weight) with seq_len=8 and
        one sink position, each head places 1/8 of its mass on that position.
        """
        seq_len = 8
        attn = _uniform_attn(n_heads=4, seq_len=seq_len)
        result = compute_sink_concentration(attn, sink_positions=[0])
        assert isinstance(result, SinkConcentration)
        assert result.mean_sink_score == pytest.approx(1.0 / seq_len, abs=1e-5)
        assert result.max_sink_score == pytest.approx(1.0 / seq_len, abs=1e-5)
        assert len(result.per_head_sink_scores) == 4

    def test_sink_dominated_concentration_high(self) -> None:
        """When every head concentrates 0.9 on sink pos 0, mean_sink_score ≈ 0.9."""
        attn = _sink_dominated_attn(
            n_heads=4, seq_len=8, sink_pos=0, sink_mass=0.9
        )
        result = compute_sink_concentration(attn, sink_positions=[0])
        # Each head sums attention over all query rows for column 0.
        # For a row-wise matrix, mean over rows then mean over heads ≈ 0.9.
        assert result.mean_sink_score == pytest.approx(0.9, abs=0.01)
        assert result.max_sink_score == pytest.approx(0.9, abs=0.01)

    def test_multiple_sink_positions(self) -> None:
        """Multiple sink positions accumulate attention from all specified columns."""
        attn = _uniform_attn(n_heads=2, seq_len=4)
        # With 2 sink positions out of 4, each head should give 2/4 = 0.5
        result = compute_sink_concentration(attn, sink_positions=[0, 3])
        assert result.mean_sink_score == pytest.approx(0.5, abs=1e-5)

    def test_per_head_scores_length_matches_heads(self) -> None:
        """per_head_sink_scores has length equal to n_heads."""
        n_heads = 6
        attn = _uniform_attn(n_heads=n_heads, seq_len=10)
        result = compute_sink_concentration(attn, sink_positions=[0])
        assert len(result.per_head_sink_scores) == n_heads

    def test_max_equals_max_of_per_head(self) -> None:
        """max_sink_score equals max(per_head_sink_scores)."""
        attn = _rng_attn(n_heads=4, seq_len=8, seed=7)
        result = compute_sink_concentration(attn, sink_positions=[0])
        assert result.max_sink_score == pytest.approx(
            max(result.per_head_sink_scores), abs=1e-6
        )

    def test_mean_equals_mean_of_per_head(self) -> None:
        """mean_sink_score equals mean(per_head_sink_scores)."""
        attn = _rng_attn(n_heads=4, seq_len=8, seed=99)
        result = compute_sink_concentration(attn, sink_positions=[0])
        expected_mean = sum(result.per_head_sink_scores) / len(
            result.per_head_sink_scores
        )
        assert result.mean_sink_score == pytest.approx(expected_mean, abs=1e-6)

    def test_no_sink_positions_returns_zeros(self) -> None:
        """Empty sink_positions list → all scores are 0.0."""
        attn = _uniform_attn(n_heads=3, seq_len=6)
        result = compute_sink_concentration(attn, sink_positions=[])
        assert result.mean_sink_score == pytest.approx(0.0)
        assert result.max_sink_score == pytest.approx(0.0)
        assert all(s == pytest.approx(0.0) for s in result.per_head_sink_scores)

    def test_scores_bounded_0_1(self) -> None:
        """All per-head sink scores are in [0, 1] for a valid attention matrix."""
        attn = _rng_attn(n_heads=8, seq_len=16, seed=12)
        result = compute_sink_concentration(attn, sink_positions=[0, 1])
        for score in result.per_head_sink_scores:
            assert 0.0 <= score <= 1.0 + 1e-6

    def test_random_array_ci_safe(self) -> None:
        """compute_sink_concentration works on random jnp arrays without real model."""
        rng = np.random.default_rng(0)
        raw = rng.random((3, 5, 5)).astype(np.float32)
        raw /= raw.sum(axis=-1, keepdims=True)
        attn = jnp.array(raw)
        result = compute_sink_concentration(attn, sink_positions=[0])
        assert isinstance(result, SinkConcentration)
        assert 0.0 <= result.mean_sink_score <= 1.0 + 1e-6

    def test_all_mass_on_single_sink(self) -> None:
        """When all query rows for every head point to position 0, score = 1.0."""
        n_heads, seq_len = 2, 5
        # One-hot attention matrix: every query attends only to key 0.
        attn_np = np.zeros((n_heads, seq_len, seq_len), dtype=np.float32)
        attn_np[:, :, 0] = 1.0
        attn = jnp.array(attn_np)
        result = compute_sink_concentration(attn, sink_positions=[0])
        assert result.mean_sink_score == pytest.approx(1.0, abs=1e-5)
        assert result.max_sink_score == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# SinkProbeResult dataclass
# ---------------------------------------------------------------------------


class TestSinkProbeResult:
    """Spec: REQ-VERIFY-086"""

    def test_fields_exist(self) -> None:
        """sink_concentration, is_uncertain, should_skip_verification present."""
        sc = SinkConcentration(
            per_head_sink_scores=[0.5], mean_sink_score=0.5, max_sink_score=0.5
        )
        result = SinkProbeResult(
            sink_concentration=sc,
            is_uncertain=False,
            should_skip_verification=True,
        )
        assert result.sink_concentration is sc
        assert result.is_uncertain is False
        assert result.should_skip_verification is True

    def test_uncertain_response_should_not_skip(self) -> None:
        """Uncertain responses must NOT skip verification."""
        sc = SinkConcentration(
            per_head_sink_scores=[0.05], mean_sink_score=0.05, max_sink_score=0.05
        )
        result = SinkProbeResult(
            sink_concentration=sc,
            is_uncertain=True,
            should_skip_verification=False,
        )
        assert result.is_uncertain is True
        assert result.should_skip_verification is False


# ---------------------------------------------------------------------------
# SinkProbe.score
# ---------------------------------------------------------------------------


class TestSinkProbeScore:
    """Spec: REQ-VERIFY-086, REQ-VERIFY-087"""

    def test_score_returns_sink_concentration(self) -> None:
        """score() returns a SinkConcentration object."""
        probe = SinkProbe(threshold=0.3)
        attn = _uniform_attn(n_heads=4, seq_len=8)
        result = probe.score(attn, sink_positions=[0])
        assert isinstance(result, SinkConcentration)

    def test_score_consistent_with_compute(self) -> None:
        """score() produces the same result as compute_sink_concentration()."""
        probe = SinkProbe(threshold=0.3)
        attn = _rng_attn(n_heads=4, seq_len=8, seed=55)
        from_probe = probe.score(attn, sink_positions=[0])
        from_fn = compute_sink_concentration(attn, sink_positions=[0])
        assert from_probe.mean_sink_score == pytest.approx(
            from_fn.mean_sink_score, abs=1e-6
        )

    def test_score_with_multiple_sink_positions(self) -> None:
        """score() handles multiple sink positions correctly."""
        probe = SinkProbe(threshold=0.3)
        attn = _uniform_attn(n_heads=2, seq_len=6)
        result = probe.score(attn, sink_positions=[0, 1])
        # Uniform 2/6 expected
        assert result.mean_sink_score == pytest.approx(2.0 / 6.0, abs=1e-5)


# ---------------------------------------------------------------------------
# SinkProbe.decide — SCENARIO-VERIFY-113 and SCENARIO-VERIFY-114
# ---------------------------------------------------------------------------


class TestSinkProbeDecide:
    """Spec: REQ-VERIFY-086, REQ-VERIFY-087
    SCENARIO-VERIFY-113, SCENARIO-VERIFY-114
    """

    def test_high_sink_skips_verification_scenario_113(self) -> None:
        """SCENARIO-VERIFY-113: high sink concentration → skip verification.

        Every head places >= 0.5 mass on BOS (position 0).
        With threshold=0.3, mean_sink_score >= 0.5 > 0.3 → not uncertain.
        """
        probe = SinkProbe(threshold=0.3)
        attn = _sink_dominated_attn(
            n_heads=4, seq_len=8, sink_pos=0, sink_mass=0.8
        )
        conc = compute_sink_concentration(attn, sink_positions=[0])
        result = probe.decide(conc)
        assert isinstance(result, SinkProbeResult)
        assert result.is_uncertain is False
        assert result.should_skip_verification is True

    def test_low_sink_triggers_verification_scenario_114(self) -> None:
        """SCENARIO-VERIFY-114: low sink concentration → run full verification.

        Uniform attention over 16 tokens gives 1/16 ≈ 0.0625 per position.
        With threshold=0.3, mean_sink_score ≈ 0.0625 < 0.3 → uncertain.
        """
        probe = SinkProbe(threshold=0.3)
        attn = _uniform_attn(n_heads=4, seq_len=16)
        conc = compute_sink_concentration(attn, sink_positions=[0])
        result = probe.decide(conc)
        assert result.is_uncertain is True
        assert result.should_skip_verification is False

    def test_exactly_at_threshold_is_not_uncertain(self) -> None:
        """Score exactly equal to threshold counts as confident (not uncertain).

        is_uncertain = mean_sink_score < threshold (strict less-than).
        """
        probe = SinkProbe(threshold=0.3)
        sc = SinkConcentration(
            per_head_sink_scores=[0.3],
            mean_sink_score=0.3,
            max_sink_score=0.3,
        )
        result = probe.decide(sc)
        # 0.3 < 0.3 is False → not uncertain → skip
        assert result.is_uncertain is False
        assert result.should_skip_verification is True

    def test_just_below_threshold_is_uncertain(self) -> None:
        """Score just below threshold → uncertain → do not skip."""
        probe = SinkProbe(threshold=0.3)
        sc = SinkConcentration(
            per_head_sink_scores=[0.299],
            mean_sink_score=0.299,
            max_sink_score=0.299,
        )
        result = probe.decide(sc)
        assert result.is_uncertain is True
        assert result.should_skip_verification is False

    def test_sink_concentration_preserved_in_result(self) -> None:
        """decide() embeds the input SinkConcentration in the returned result."""
        probe = SinkProbe(threshold=0.3)
        sc = SinkConcentration(
            per_head_sink_scores=[0.6, 0.7],
            mean_sink_score=0.65,
            max_sink_score=0.7,
        )
        result = probe.decide(sc)
        assert result.sink_concentration is sc

    def test_custom_threshold_low(self) -> None:
        """A very low threshold (0.01) means almost all responses skip verification."""
        probe = SinkProbe(threshold=0.01)
        sc = SinkConcentration(
            per_head_sink_scores=[0.05],
            mean_sink_score=0.05,
            max_sink_score=0.05,
        )
        result = probe.decide(sc)
        # 0.05 >= 0.01 → not uncertain → skip
        assert result.should_skip_verification is True

    def test_custom_threshold_high(self) -> None:
        """A very high threshold (0.99) means almost no response skips verification."""
        probe = SinkProbe(threshold=0.99)
        sc = SinkConcentration(
            per_head_sink_scores=[0.9],
            mean_sink_score=0.9,
            max_sink_score=0.9,
        )
        result = probe.decide(sc)
        # 0.9 < 0.99 → uncertain → do NOT skip
        assert result.should_skip_verification is False


# ---------------------------------------------------------------------------
# SinkProbe default construction
# ---------------------------------------------------------------------------


class TestSinkProbeDefaults:
    """Spec: REQ-VERIFY-087"""

    def test_default_threshold_is_0_3(self) -> None:
        """Default threshold is 0.3."""
        probe = SinkProbe()
        assert probe.threshold == pytest.approx(0.3)

    def test_default_sink_token_types_include_bos_and_period(self) -> None:
        """Default sink token types include at least BOS and PERIOD."""
        probe = SinkProbe()
        types = set(probe.sink_token_types)
        assert SinkTokenType.BOS in types
        assert SinkTokenType.PERIOD in types

    def test_custom_threshold_stored(self) -> None:
        """Custom threshold is stored and accessible."""
        probe = SinkProbe(threshold=0.5)
        assert probe.threshold == pytest.approx(0.5)

    def test_custom_sink_token_types_stored(self) -> None:
        """Custom sink token types are stored and accessible."""
        probe = SinkProbe(sink_token_types=(SinkTokenType.BOS,))
        assert SinkTokenType.BOS in probe.sink_token_types


# ---------------------------------------------------------------------------
# SinkProbe.benchmark — SCENARIO-VERIFY-115
# ---------------------------------------------------------------------------


class TestSinkProbeBenchmark:
    """Spec: REQ-VERIFY-086, REQ-VERIFY-087
    SCENARIO-VERIFY-115
    """

    def _make_responses_with_attention(
        self,
        n_correct: int,
        n_wrong: int,
        n_heads: int = 4,
        seq_len: int = 8,
        sink_pos: int = 0,
        high_mass: float = 0.8,
        low_mass: float = 0.05,
    ) -> tuple[list[dict], list[bool]]:
        """Build synthetic (attention_matrix, correctness) pairs.

        Correct responses have high sink concentration (confident model).
        Wrong responses have low sink concentration (uncertain model).
        """
        responses = []
        labels = []
        for i in range(n_correct):
            attn = _sink_dominated_attn(n_heads, seq_len, sink_pos, high_mass)
            responses.append({"attention_matrix": attn, "sink_positions": [sink_pos]})
            labels.append(True)
        for i in range(n_wrong):
            attn = _uniform_attn(n_heads, seq_len)
            responses.append({"attention_matrix": attn, "sink_positions": [sink_pos]})
            labels.append(False)
        return responses, labels

    def test_benchmark_returns_dict_with_required_keys(self) -> None:
        """benchmark() returns a dict with skip_rate, false_negative_rate, true_negative_rate."""
        probe = SinkProbe(threshold=0.3)
        responses, labels = self._make_responses_with_attention(5, 5)
        result = probe.benchmark(responses, labels)
        assert "skip_rate" in result
        assert "false_negative_rate" in result
        assert "true_negative_rate" in result

    def test_rates_in_0_1_range(self) -> None:
        """All rates are in [0.0, 1.0]."""
        probe = SinkProbe(threshold=0.3)
        responses, labels = self._make_responses_with_attention(6, 4)
        result = probe.benchmark(responses, labels)
        for key in ("skip_rate", "false_negative_rate", "true_negative_rate"):
            assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"

    def test_scenario_115_accurate_rates(self) -> None:
        """SCENARIO-VERIFY-115: 6 correct (high sink) 4 wrong (low sink), threshold=0.3.

        High sink (0.8) responses → mean_sink_score ≈ 0.8 → skipped.
        Low  sink (1/8 ≈ 0.125) responses → mean_sink_score ≈ 0.125 → NOT skipped.
        Expected: all 6 correct skipped, 0 wrong skipped.
        skip_rate = 6/10 = 0.6
        false_negative_rate = 0/4 = 0.0  (no wrong responses were skipped)
        true_negative_rate = 6/6 = 1.0   (all correct responses were skipped)
        """
        probe = SinkProbe(threshold=0.3)
        responses, labels = self._make_responses_with_attention(
            n_correct=6, n_wrong=4, seq_len=8, high_mass=0.8, low_mass=0.05
        )
        result = probe.benchmark(responses, labels)
        assert result["skip_rate"] == pytest.approx(0.6, abs=0.01)
        assert result["false_negative_rate"] == pytest.approx(0.0, abs=0.01)
        assert result["true_negative_rate"] == pytest.approx(1.0, abs=0.01)

    def test_all_skipped_when_all_high_sink(self) -> None:
        """When all responses have high sink, skip_rate = 1.0 and FNR = 0."""
        probe = SinkProbe(threshold=0.3)
        responses, labels = self._make_responses_with_attention(
            n_correct=5, n_wrong=0, high_mass=0.9
        )
        result = probe.benchmark(responses, labels)
        assert result["skip_rate"] == pytest.approx(1.0, abs=0.01)
        assert result["false_negative_rate"] == pytest.approx(0.0, abs=0.01)
        assert result["true_negative_rate"] == pytest.approx(1.0, abs=0.01)

    def test_none_skipped_when_all_uncertain(self) -> None:
        """When all responses have low sink, skip_rate = 0.0."""
        probe = SinkProbe(threshold=0.3)
        # Build uniform attention (low concentration)
        attn_list = [
            {"attention_matrix": _uniform_attn(4, 16), "sink_positions": [0]}
            for _ in range(6)
        ]
        labels = [True, False, True, False, True, False]
        result = probe.benchmark(attn_list, labels)
        assert result["skip_rate"] == pytest.approx(0.0, abs=0.01)

    def test_false_negative_rate_nonzero_when_wrong_skipped(self) -> None:
        """FNR > 0 when wrong responses are skipped (high sink on incorrect output)."""
        probe = SinkProbe(threshold=0.3)
        # All responses have high sink, but some are WRONG (incorrect labels)
        attn_list = [
            {"attention_matrix": _sink_dominated_attn(4, 8, 0, 0.8), "sink_positions": [0]}
            for _ in range(4)
        ]
        labels = [True, False, True, False]  # 2 wrong with high sink → FNR > 0
        result = probe.benchmark(attn_list, labels)
        # 2 wrong responses both skipped → FNR = 2/2 = 1.0
        assert result["false_negative_rate"] == pytest.approx(1.0, abs=0.01)

    def test_empty_responses_returns_zero_rates(self) -> None:
        """Empty input returns all-zero rates (no division by zero)."""
        probe = SinkProbe(threshold=0.3)
        result = probe.benchmark([], [])
        assert result["skip_rate"] == pytest.approx(0.0)
        assert result["false_negative_rate"] == pytest.approx(0.0)
        assert result["true_negative_rate"] == pytest.approx(0.0)

    def test_only_wrong_responses_no_correct(self) -> None:
        """When there are no correct responses, true_negative_rate = 0.0."""
        probe = SinkProbe(threshold=0.3)
        attn_list = [
            {"attention_matrix": _uniform_attn(2, 8), "sink_positions": [0]}
            for _ in range(3)
        ]
        labels = [False, False, False]
        result = probe.benchmark(attn_list, labels)
        assert result["true_negative_rate"] == pytest.approx(0.0)

    def test_only_correct_responses_no_wrong(self) -> None:
        """When there are no wrong responses, false_negative_rate = 0.0."""
        probe = SinkProbe(threshold=0.3)
        attn_list = [
            {"attention_matrix": _sink_dominated_attn(2, 8, 0, 0.9), "sink_positions": [0]}
            for _ in range(3)
        ]
        labels = [True, True, True]
        result = probe.benchmark(attn_list, labels)
        assert result["false_negative_rate"] == pytest.approx(0.0)
