"""Tests for three_tier_pipeline.py — three-tier verification pipeline benchmark.

Covers:
  - ThreeTierPipelineResult dataclass fields
  - ThreeTierPipeline.verify() routing through each tier
  - ThreeTierPipeline.benchmark() metrics calculation
  - build_three_tier_artifact() schema and serialization
  - Tier 0c (NUP Probe v4) and Tier 0d (HallucinationBasinDetector) wiring
  - CI-safe: all tests run on CPU with synthetic data, no real LLM required

Spec: REQ-VERIFY-088, REQ-VERIFY-111, REQ-VERIFY-112
SCENARIO-VERIFY-116 (SinkProbe fast-path routing)
SCENARIO-VERIFY-117 (benchmark skip rate and accuracy metrics)
SCENARIO-VERIFY-146 (NUP Probe v4 clears on low score)
SCENARIO-VERIFY-147 (NUP Probe v4 absent — no regression)
SCENARIO-VERIFY-148 (basin detector clears on low risk)
"""

from __future__ import annotations

import math
import time

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.models.eorm import EORMModel
from carnot.pipeline.hallucination_basin import HallucinationBasinDetector
from carnot.pipeline.nup_probe_v4 import NUPProbeV4
from carnot.pipeline.sink_probe import SinkProbe
from carnot.pipeline.three_tier_pipeline import (
    ThreeTierPipeline,
    ThreeTierPipelineResult,
    build_three_tier_artifact,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_eorm(seed: int = 0) -> EORMModel:
    """Build a tiny EORM model for CI (embed_dim=32 so tests are fast)."""
    import jax.random as jr

    key = jr.PRNGKey(seed)
    return EORMModel(
        embed_dim=32,
        n_heads=2,
        n_layers=1,
        max_seq_len=64,
        vocab_size=256,
        key=key,
    )


def _uniform_attn(n_heads: int = 4, seq_len: int = 8) -> jnp.ndarray:
    """Uniform attention — sink concentration = 1/seq_len per sink position."""
    return jnp.full((n_heads, seq_len, seq_len), 1.0 / seq_len)


def _high_sink_attn(
    n_heads: int = 4,
    seq_len: int = 8,
    sink_mass: float = 0.9,
) -> jnp.ndarray:
    """Attention where every head routes `sink_mass` to position 0 (BOS)."""
    remaining = (1.0 - sink_mass) / max(seq_len - 1, 1)
    attn = jnp.full((n_heads, seq_len, seq_len), remaining)
    attn = attn.at[:, :, 0].set(sink_mass)
    return attn


def _ising_stub_correct(response: str, question: str) -> tuple[bool, float]:
    """Stub ising pipeline that always returns (True, 0.0)."""
    return (True, 0.0)


def _ising_stub_wrong(response: str, question: str) -> tuple[bool, float]:
    """Stub ising pipeline that always returns (False, 2.0)."""
    return (False, 2.0)


def _ising_stub_energy(energy: float):
    """Return an ising stub that returns (True, energy)."""
    def _stub(response: str, question: str) -> tuple[bool, float]:
        return (True, energy)
    return _stub


def _make_pipeline(
    sink_threshold: float = 0.3,
    eorm_threshold: float = 0.5,
    ising_fn=_ising_stub_correct,
    eorm_seed: int = 0,
) -> ThreeTierPipeline:
    """Build a CI-safe ThreeTierPipeline with small EORM and stub Ising."""
    return ThreeTierPipeline(
        sink_probe=SinkProbe(threshold=sink_threshold),
        eorm_model=_make_eorm(seed=eorm_seed),
        ising_pipeline=ising_fn,
        sink_threshold=sink_threshold,
        eorm_threshold=eorm_threshold,
    )


# ---------------------------------------------------------------------------
# ThreeTierPipelineResult — dataclass structure
# ---------------------------------------------------------------------------


class TestThreeTierPipelineResult:
    """Spec: REQ-VERIFY-088"""

    def test_fields_are_floats(self):
        """All numeric fields must be floats."""
        r = ThreeTierPipelineResult(
            skip_rate_sink_probe=0.3,
            skip_rate_eorm=0.2,
            total_skip_rate=0.5,
            fn_rate=0.05,
            throughput_qps=100.0,
            ising_calls_saved_pct=50.0,
            inference_mode="cpu_synthetic",
        )
        assert isinstance(r.skip_rate_sink_probe, float)
        assert isinstance(r.skip_rate_eorm, float)
        assert isinstance(r.total_skip_rate, float)
        assert isinstance(r.fn_rate, float)
        assert isinstance(r.throughput_qps, float)
        assert isinstance(r.ising_calls_saved_pct, float)

    def test_inference_mode_field(self):
        """inference_mode is a string label."""
        r = ThreeTierPipelineResult(
            skip_rate_sink_probe=0.0,
            skip_rate_eorm=0.0,
            total_skip_rate=0.0,
            fn_rate=0.0,
            throughput_qps=1.0,
            ising_calls_saved_pct=0.0,
            inference_mode="live_gpu",
        )
        assert r.inference_mode == "live_gpu"

    def test_ising_calls_saved_pct_relationship(self):
        """ising_calls_saved_pct should equal total_skip_rate * 100."""
        r = ThreeTierPipelineResult(
            skip_rate_sink_probe=0.2,
            skip_rate_eorm=0.15,
            total_skip_rate=0.35,
            fn_rate=0.02,
            throughput_qps=500.0,
            ising_calls_saved_pct=35.0,
            inference_mode="cpu_synthetic",
        )
        assert r.ising_calls_saved_pct == pytest.approx(r.total_skip_rate * 100, abs=1e-6)


# ---------------------------------------------------------------------------
# ThreeTierPipeline construction
# ---------------------------------------------------------------------------


class TestThreeTierPipelineConstruction:
    """Spec: REQ-VERIFY-088"""

    def test_attributes_stored(self):
        """Constructor stores all parameters as attributes."""
        sp = SinkProbe(threshold=0.25)
        eorm = _make_eorm()
        pipeline = ThreeTierPipeline(
            sink_probe=sp,
            eorm_model=eorm,
            ising_pipeline=_ising_stub_correct,
            sink_threshold=0.25,
            eorm_threshold=0.6,
        )
        assert pipeline.sink_probe is sp
        assert pipeline.eorm_model is eorm
        assert pipeline.ising_pipeline is _ising_stub_correct
        assert pipeline.sink_threshold == 0.25
        assert pipeline.eorm_threshold == 0.6

    def test_default_thresholds(self):
        """Default thresholds are 0.3 for SinkProbe and 0.5 for EORM."""
        pipeline = _make_pipeline()
        assert pipeline.sink_threshold == 0.3
        assert pipeline.eorm_threshold == 0.5


# ---------------------------------------------------------------------------
# ThreeTierPipeline.verify() — Tier 1 (SinkProbe)
# ---------------------------------------------------------------------------


class TestVerifyTierSinkProbe:
    """SCENARIO-VERIFY-116: high-sink responses routed to fast path."""

    def test_high_sink_clears_via_sink_probe(self):
        """High sink concentration → tier_used='sink_probe', verified=True."""
        pipeline = _make_pipeline(sink_threshold=0.3)
        attn = _high_sink_attn(sink_mass=0.9)
        verified, tier_used, energy = pipeline.verify(
            "The answer is 42.",
            attention_matrix=attn,
            question="What is the answer?",
        )
        assert tier_used == "sink_probe"
        assert verified is True
        assert energy >= 0.3  # the mean_sink_score

    def test_high_sink_attn_returns_sink_score_as_energy(self):
        """Energy returned by SinkProbe tier is the mean_sink_score."""
        pipeline = _make_pipeline(sink_threshold=0.3)
        attn = _high_sink_attn(sink_mass=0.9)
        _verified, tier_used, energy = pipeline.verify("r", attention_matrix=attn)
        assert tier_used == "sink_probe"
        # mean_sink_score ≈ 0.9 for 4 heads all routing 0.9 to position 0
        assert energy == pytest.approx(0.9, abs=1e-5)

    def test_above_threshold_clears(self):
        """mean_sink_score above sink_threshold clears via SinkProbe (>= is inclusive)."""
        pipeline = _make_pipeline(sink_threshold=0.3, eorm_threshold=-999.0)
        # sink_mass=0.5 gives mean_sink_score≈0.5 which is clearly >= 0.3
        attn = _high_sink_attn(sink_mass=0.5)
        _verified, tier_used, energy = pipeline.verify("r", attention_matrix=attn)
        assert tier_used == "sink_probe"
        assert energy == pytest.approx(0.5, abs=1e-4)

    def test_sink_probe_not_called_when_attn_none(self):
        """When attention_matrix=None, Tier 1 is skipped entirely."""
        # Use eorm_threshold=-999.0 so EORM never clears, ensuring Ising is reached.
        # If SinkProbe were called with None, it would error; it must be bypassed.
        pipeline = _make_pipeline(sink_threshold=0.0, eorm_threshold=-999.0)
        _verified, tier_used, _energy = pipeline.verify("r", attention_matrix=None)
        assert tier_used == "ising"

    def test_low_sink_falls_through_to_next_tier(self):
        """Low sink concentration → SinkProbe does NOT clear → proceeds to EORM/Ising."""
        # Use eorm_threshold=-999.0 so EORM never clears; response must reach Ising.
        pipeline = _make_pipeline(sink_threshold=0.9, eorm_threshold=-999.0)
        # uniform attn → mean_sink_score ≈ 1/seq_len = 0.125 < 0.9
        attn = _uniform_attn(seq_len=8)
        _verified, tier_used, _energy = pipeline.verify("r", attention_matrix=attn)
        assert tier_used == "ising"


# ---------------------------------------------------------------------------
# ThreeTierPipeline.verify() — Tier 2 (EORM)
# ---------------------------------------------------------------------------


class TestVerifyTierEORM:
    """Spec: REQ-VERIFY-088"""

    def test_low_eorm_energy_clears_at_tier2(self):
        """If EORM energy < eorm_threshold, tier_used='eorm'."""
        # Use eorm_threshold=999.0 so EORM always clears (energy < 999).
        pipeline = _make_pipeline(
            sink_threshold=0.99,  # high — uniform attn will miss
            eorm_threshold=999.0,
        )
        attn = _uniform_attn(seq_len=8)
        _verified, tier_used, energy = pipeline.verify(
            "response",
            attention_matrix=attn,
            question="question",
        )
        assert tier_used == "eorm"
        assert isinstance(energy, float)

    def test_eorm_tier_verified_is_true(self):
        """EORM clear always sets verified=True."""
        pipeline = _make_pipeline(sink_threshold=0.99, eorm_threshold=999.0)
        verified, tier_used, _energy = pipeline.verify("r", attention_matrix=_uniform_attn())
        assert tier_used == "eorm"
        assert verified is True

    def test_eorm_energy_returned_as_energy(self):
        """The energy field for EORM tier is the raw EORM scalar."""
        pipeline = _make_pipeline(sink_threshold=0.99, eorm_threshold=999.0)
        _v, tier_used, energy = pipeline.verify("r", attention_matrix=_uniform_attn())
        assert tier_used == "eorm"
        assert math.isfinite(energy)

    def test_no_attn_eorm_clears(self):
        """Without attention_matrix, if EORM clears, tier_used='eorm'."""
        pipeline = _make_pipeline(eorm_threshold=999.0)
        _v, tier_used, _e = pipeline.verify("r", attention_matrix=None)
        assert tier_used == "eorm"


# ---------------------------------------------------------------------------
# ThreeTierPipeline.verify() — Tier 3 (Ising)
# ---------------------------------------------------------------------------


class TestVerifyTierIsing:
    """Spec: REQ-VERIFY-088"""

    def test_ising_reached_when_both_upper_tiers_fail(self):
        """High sink_threshold + high eorm_threshold → response reaches Ising."""
        pipeline = _make_pipeline(
            sink_threshold=0.99,
            eorm_threshold=-999.0,  # nothing clears (energy always >= -999)
            ising_fn=_ising_stub_correct,
        )
        attn = _uniform_attn()
        _v, tier_used, energy = pipeline.verify("r", attention_matrix=attn)
        assert tier_used == "ising"
        assert energy == 0.0  # stub returns 0.0

    def test_ising_verified_true_when_stub_says_correct(self):
        """Ising stub returning True propagates as verified=True."""
        pipeline = _make_pipeline(sink_threshold=0.99, eorm_threshold=-999.0)
        verified, tier_used, _e = pipeline.verify("r", attention_matrix=_uniform_attn())
        assert tier_used == "ising"
        assert verified is True

    def test_ising_verified_false_when_stub_says_wrong(self):
        """Ising stub returning False propagates as verified=False."""
        pipeline = _make_pipeline(
            sink_threshold=0.99,
            eorm_threshold=-999.0,
            ising_fn=_ising_stub_wrong,
        )
        verified, tier_used, energy = pipeline.verify("r", attention_matrix=_uniform_attn())
        assert tier_used == "ising"
        assert verified is False
        assert energy == 2.0  # stub returns 2.0

    def test_ising_energy_from_stub(self):
        """Energy from Ising tier matches stub return value."""
        target_energy = 3.14
        pipeline = _make_pipeline(
            sink_threshold=0.99,
            eorm_threshold=-999.0,
            ising_fn=_ising_stub_energy(target_energy),
        )
        _v, tier_used, energy = pipeline.verify("r", attention_matrix=_uniform_attn())
        assert tier_used == "ising"
        assert energy == pytest.approx(target_energy, abs=1e-6)

    def test_no_attn_ising_fallback_when_eorm_threshold_negative(self):
        """No attention_matrix + negative eorm_threshold → Ising."""
        pipeline = _make_pipeline(eorm_threshold=-999.0, ising_fn=_ising_stub_correct)
        _v, tier_used, _e = pipeline.verify("r", attention_matrix=None)
        assert tier_used == "ising"


# ---------------------------------------------------------------------------
# ThreeTierPipeline.benchmark() — empty input
# ---------------------------------------------------------------------------


class TestBenchmarkEmpty:
    """Spec: REQ-VERIFY-088"""

    def test_empty_responses_returns_zeros(self):
        """Empty input returns a zero-filled result."""
        pipeline = _make_pipeline()
        result = pipeline.benchmark([], [])
        assert result.skip_rate_sink_probe == 0.0
        assert result.skip_rate_eorm == 0.0
        assert result.total_skip_rate == 0.0
        assert result.fn_rate == 0.0
        assert result.throughput_qps == 0.0
        assert result.ising_calls_saved_pct == 0.0

    def test_empty_result_inference_mode(self):
        """inference_mode is preserved in the empty result."""
        pipeline = _make_pipeline()
        result = pipeline.benchmark([], [], inference_mode="live_gpu")
        assert result.inference_mode == "live_gpu"


# ---------------------------------------------------------------------------
# ThreeTierPipeline.benchmark() — skip rates
# ---------------------------------------------------------------------------


class TestBenchmarkSkipRates:
    """SCENARIO-VERIFY-117: benchmark metrics correctness."""

    def _make_responses_high_sink(self, n: int) -> list[dict]:
        """n responses that will be cleared by SinkProbe (high sink attn)."""
        attn = _high_sink_attn(sink_mass=0.9)
        return [
            {"response": f"resp_{i}", "question": "q", "attention_matrix": attn}
            for i in range(n)
        ]

    def _make_responses_no_attn_low_eorm(self, n: int) -> list[dict]:
        """n responses with no attention_matrix; cleared by EORM (eorm_threshold=999)."""
        return [
            {"response": f"resp_{i}", "question": "q", "attention_matrix": None}
            for i in range(n)
        ]

    def _make_responses_reach_ising(self, n: int) -> list[dict]:
        """n responses that reach Ising: uniform attn (low sink) + high eorm threshold."""
        attn = _uniform_attn()
        return [
            {"response": f"resp_{i}", "question": "q", "attention_matrix": attn}
            for i in range(n)
        ]

    def test_all_cleared_by_sink_probe(self):
        """When all responses have high sink: skip_rate_sink_probe=1.0."""
        pipeline = _make_pipeline(sink_threshold=0.3)
        responses = self._make_responses_high_sink(10)
        labels = [True] * 10
        result = pipeline.benchmark(responses, labels)
        assert result.skip_rate_sink_probe == pytest.approx(1.0, abs=1e-6)
        assert result.total_skip_rate == pytest.approx(1.0, abs=1e-6)
        assert result.ising_calls_saved_pct == pytest.approx(100.0, abs=1e-6)

    def test_all_cleared_by_eorm(self):
        """When eorm_threshold=999 and no attn: skip_rate_eorm=1.0."""
        pipeline = _make_pipeline(sink_threshold=0.99, eorm_threshold=999.0)
        responses = self._make_responses_no_attn_low_eorm(10)
        labels = [True] * 10
        result = pipeline.benchmark(responses, labels)
        assert result.skip_rate_eorm == pytest.approx(1.0, abs=1e-6)
        assert result.skip_rate_sink_probe == pytest.approx(0.0, abs=1e-6)
        assert result.total_skip_rate == pytest.approx(1.0, abs=1e-6)

    def test_all_reach_ising(self):
        """When no tier clears: total_skip_rate=0.0."""
        pipeline = _make_pipeline(sink_threshold=0.99, eorm_threshold=-999.0)
        responses = self._make_responses_reach_ising(10)
        labels = [True] * 10
        result = pipeline.benchmark(responses, labels)
        assert result.total_skip_rate == pytest.approx(0.0, abs=1e-6)
        assert result.skip_rate_sink_probe == pytest.approx(0.0, abs=1e-6)
        assert result.skip_rate_eorm == pytest.approx(0.0, abs=1e-6)

    def test_mixed_tiers(self):
        """5 high-sink (cleared by T1) + 5 no-attn (cleared by EORM) = total_skip_rate=1.0."""
        pipeline = _make_pipeline(sink_threshold=0.3, eorm_threshold=999.0)
        responses = (
            self._make_responses_high_sink(5)
            + self._make_responses_no_attn_low_eorm(5)
        )
        labels = [True] * 10
        result = pipeline.benchmark(responses, labels)
        assert result.skip_rate_sink_probe == pytest.approx(0.5, abs=1e-6)
        assert result.skip_rate_eorm == pytest.approx(0.5, abs=1e-6)
        assert result.total_skip_rate == pytest.approx(1.0, abs=1e-6)

    def test_partial_skip(self):
        """4/10 high-sink cleared + 6/10 reach Ising → skip_rate_sink_probe=0.4."""
        pipeline = _make_pipeline(sink_threshold=0.3, eorm_threshold=-999.0)
        attn_high = _high_sink_attn(sink_mass=0.9)
        attn_low = _uniform_attn()
        responses = (
            [{"response": f"r{i}", "question": "q", "attention_matrix": attn_high} for i in range(4)]
            + [{"response": f"r{i}", "question": "q", "attention_matrix": attn_low} for i in range(6)]
        )
        labels = [True] * 10
        result = pipeline.benchmark(responses, labels)
        assert result.skip_rate_sink_probe == pytest.approx(0.4, abs=1e-6)
        assert result.total_skip_rate == pytest.approx(0.4, abs=1e-6)


# ---------------------------------------------------------------------------
# ThreeTierPipeline.benchmark() — false-negative rate
# ---------------------------------------------------------------------------


class TestBenchmarkFNRate:
    """Spec: REQ-VERIFY-088"""

    def test_no_wrong_responses_fn_rate_zero(self):
        """When all ground_truth=True, fn_rate=0.0 (no wrong responses exist)."""
        pipeline = _make_pipeline(sink_threshold=0.3)
        responses = [
            {"response": "r", "question": "q", "attention_matrix": _high_sink_attn()}
            for _ in range(10)
        ]
        labels = [True] * 10
        result = pipeline.benchmark(responses, labels)
        assert result.fn_rate == pytest.approx(0.0, abs=1e-6)

    def test_wrong_responses_cleared_by_sink_probe_counted_as_fn(self):
        """Wrong responses cleared by SinkProbe are false negatives."""
        pipeline = _make_pipeline(sink_threshold=0.3)
        attn_high = _high_sink_attn(sink_mass=0.9)
        # 3 wrong responses that SinkProbe will incorrectly clear
        responses = [
            {"response": "r", "question": "q", "attention_matrix": attn_high}
            for _ in range(3)
        ]
        labels = [False, False, False]
        result = pipeline.benchmark(responses, labels)
        # All 3 wrong, all 3 cleared → fn_rate=1.0
        assert result.fn_rate == pytest.approx(1.0, abs=1e-6)

    def test_wrong_responses_reaching_ising_not_fn(self):
        """Wrong responses that reach Ising are NOT false negatives."""
        # High threshold ensures responses reach Ising
        pipeline = _make_pipeline(sink_threshold=0.99, eorm_threshold=-999.0)
        attn_low = _uniform_attn()
        responses = [
            {"response": "r", "question": "q", "attention_matrix": attn_low}
            for _ in range(5)
        ]
        labels = [False] * 5
        result = pipeline.benchmark(responses, labels)
        # All reach Ising → fn_rate=0.0
        assert result.fn_rate == pytest.approx(0.0, abs=1e-6)

    def test_mixed_fn_rate(self):
        """2 wrong cleared + 3 wrong reaching Ising → fn_rate=0.4."""
        pipeline = _make_pipeline(sink_threshold=0.3, eorm_threshold=-999.0)
        attn_high = _high_sink_attn(sink_mass=0.9)
        attn_low = _uniform_attn()
        responses = (
            [{"response": "r", "question": "q", "attention_matrix": attn_high} for _ in range(2)]
            + [{"response": "r", "question": "q", "attention_matrix": attn_low} for _ in range(3)]
        )
        labels = [False] * 5
        result = pipeline.benchmark(responses, labels)
        assert result.fn_rate == pytest.approx(0.4, abs=1e-6)

    def test_wrong_cleared_by_eorm_counted_as_fn(self):
        """Wrong responses cleared by EORM are also false negatives."""
        pipeline = _make_pipeline(sink_threshold=0.99, eorm_threshold=999.0)
        responses = [
            {"response": "r", "question": "q", "attention_matrix": None}
            for _ in range(4)
        ]
        labels = [False] * 4
        result = pipeline.benchmark(responses, labels)
        # All cleared by EORM (no attn, threshold=999) → fn_rate=1.0
        assert result.fn_rate == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# ThreeTierPipeline.benchmark() — throughput and inference_mode
# ---------------------------------------------------------------------------


class TestBenchmarkThroughput:
    """Spec: REQ-VERIFY-088"""

    def test_throughput_qps_positive(self):
        """throughput_qps must be > 0 for a non-empty input."""
        pipeline = _make_pipeline()
        responses = [
            {"response": "r", "question": "q", "attention_matrix": None}
            for _ in range(20)
        ]
        labels = [True] * 20
        result = pipeline.benchmark(responses, labels)
        assert result.throughput_qps > 0.0

    def test_inference_mode_default(self):
        """Default inference_mode is 'cpu_synthetic'."""
        pipeline = _make_pipeline()
        responses = [{"response": "r", "question": "q", "attention_matrix": None}]
        result = pipeline.benchmark(responses, [True])
        assert result.inference_mode == "cpu_synthetic"

    def test_inference_mode_override(self):
        """Custom inference_mode is preserved."""
        pipeline = _make_pipeline()
        responses = [{"response": "r", "question": "q", "attention_matrix": None}]
        result = pipeline.benchmark(responses, [True], inference_mode="live_gpu")
        assert result.inference_mode == "live_gpu"

    def test_ising_calls_saved_pct_formula(self):
        """ising_calls_saved_pct == total_skip_rate * 100."""
        pipeline = _make_pipeline(sink_threshold=0.3, eorm_threshold=-999.0)
        attn_high = _high_sink_attn()
        responses = [
            {"response": "r", "question": "q", "attention_matrix": attn_high}
            for _ in range(10)
        ]
        labels = [True] * 10
        result = pipeline.benchmark(responses, labels)
        assert result.ising_calls_saved_pct == pytest.approx(
            result.total_skip_rate * 100, abs=1e-6
        )


# ---------------------------------------------------------------------------
# build_three_tier_artifact()
# ---------------------------------------------------------------------------


class TestBuildThreeTierArtifact:
    """Spec: REQ-VERIFY-088"""

    def _make_result(self) -> ThreeTierPipelineResult:
        return ThreeTierPipelineResult(
            skip_rate_sink_probe=0.3,
            skip_rate_eorm=0.2,
            total_skip_rate=0.5,
            fn_rate=0.04,
            throughput_qps=150.0,
            ising_calls_saved_pct=50.0,
            inference_mode="cpu_synthetic",
        )

    def test_schema_tag(self):
        """Artifact must have schema='carnot.three_tier_benchmark.v1'."""
        artifact = build_three_tier_artifact(self._make_result())
        assert artifact["schema"] == "carnot.three_tier_benchmark.v1"

    def test_all_fields_present(self):
        """All ThreeTierPipelineResult fields appear in the artifact."""
        artifact = build_three_tier_artifact(self._make_result())
        for field in [
            "skip_rate_sink_probe",
            "skip_rate_eorm",
            "total_skip_rate",
            "fn_rate",
            "throughput_qps",
            "ising_calls_saved_pct",
            "inference_mode",
        ]:
            assert field in artifact, f"Missing field: {field}"

    def test_values_match_result(self):
        """Artifact values must match the input ThreeTierPipelineResult."""
        r = self._make_result()
        artifact = build_three_tier_artifact(r)
        assert artifact["skip_rate_sink_probe"] == r.skip_rate_sink_probe
        assert artifact["skip_rate_eorm"] == r.skip_rate_eorm
        assert artifact["total_skip_rate"] == r.total_skip_rate
        assert artifact["fn_rate"] == r.fn_rate
        assert artifact["throughput_qps"] == r.throughput_qps
        assert artifact["ising_calls_saved_pct"] == r.ising_calls_saved_pct
        assert artifact["inference_mode"] == r.inference_mode

    def test_artifact_is_json_serialisable(self):
        """All values in the artifact must be JSON-serialisable."""
        import json

        artifact = build_three_tier_artifact(self._make_result())
        # Should not raise
        serialised = json.dumps(artifact)
        parsed = json.loads(serialised)
        assert parsed["schema"] == "carnot.three_tier_benchmark.v1"

    def test_artifact_returns_dict(self):
        """build_three_tier_artifact returns a plain dict."""
        artifact = build_three_tier_artifact(self._make_result())
        assert isinstance(artifact, dict)


# ---------------------------------------------------------------------------
# Integration: verify() return type guarantees
# ---------------------------------------------------------------------------


class TestVerifyReturnTypes:
    """Spec: REQ-VERIFY-088 — verify() return type contract."""

    @pytest.mark.parametrize("attn", [None, _uniform_attn(), _high_sink_attn()])
    def test_verify_returns_tuple_of_three(self, attn):
        """verify() always returns a 3-tuple."""
        pipeline = _make_pipeline()
        result = pipeline.verify("response text", attention_matrix=attn, question="q")
        assert len(result) == 3

    @pytest.mark.parametrize("attn", [None, _uniform_attn(), _high_sink_attn()])
    def test_verify_verified_is_bool(self, attn):
        """First element of verify() result is bool."""
        pipeline = _make_pipeline()
        verified, _t, _e = pipeline.verify("r", attention_matrix=attn)
        assert isinstance(verified, bool)

    @pytest.mark.parametrize("attn", [None, _uniform_attn(), _high_sink_attn()])
    def test_verify_tier_is_string(self, attn):
        """Second element of verify() result is a non-empty string."""
        pipeline = _make_pipeline()
        _v, tier_used, _e = pipeline.verify("r", attention_matrix=attn)
        assert isinstance(tier_used, str)
        assert tier_used in ("sink_probe", "eorm", "ising")

    @pytest.mark.parametrize("attn", [None, _uniform_attn(), _high_sink_attn()])
    def test_verify_energy_is_float(self, attn):
        """Third element of verify() result is a finite float."""
        pipeline = _make_pipeline()
        _v, _t, energy = pipeline.verify("r", attention_matrix=attn)
        assert isinstance(energy, float)
        assert math.isfinite(energy)

    def test_verify_empty_response_and_question(self):
        """verify() handles empty strings without error."""
        pipeline = _make_pipeline()
        verified, tier_used, energy = pipeline.verify("", attention_matrix=None, question="")
        assert tier_used in ("sink_probe", "eorm", "ising")

    def test_verify_numpy_attn_matrix(self):
        """verify() accepts numpy arrays (not just jnp) for attention_matrix."""
        pipeline = _make_pipeline(sink_threshold=0.3)
        attn_np = np.full((4, 8, 8), 0.9)  # high sink, will be cleared
        # Clamp rows to sum=1: set column 0 to 0.9, rest equally share 0.1
        remaining = 0.1 / 7
        attn_np[:, :, :] = remaining
        attn_np[:, :, 0] = 0.9
        verified, tier_used, energy = pipeline.verify("r", attention_matrix=attn_np)
        assert tier_used == "sink_probe"
        assert verified is True


# ---------------------------------------------------------------------------
# Tier 0c — NUP Probe v4 wiring
# ---------------------------------------------------------------------------


def _make_nup_low_score() -> NUPProbeV4:
    """NUPProbeV4 instance whose score() always returns -1.0 (below default threshold 0.0).

    We train it so incorrect steps get higher energy than correct steps, then
    use a known correct-looking string that scores low.  Simpler: reset weights to
    all-negative so dot(weights, features) is always negative.
    """
    probe = NUPProbeV4(energy_dim=8, random_seed=0)
    # Force all weights to a large negative value so score() always returns << 0.
    probe._weights = [-10.0] * probe.energy_dim
    probe._bias = 0.0
    return probe


def _make_nup_high_score() -> NUPProbeV4:
    """NUPProbeV4 instance whose score() always returns a large positive value.

    Force weights all-positive so any non-trivial response gets a high score.
    """
    probe = NUPProbeV4(energy_dim=8, random_seed=0)
    probe._weights = [10.0] * probe.energy_dim
    probe._bias = 0.0
    return probe


class TestTier0cNUPProbeV4:
    """Spec: REQ-VERIFY-111, SCENARIO-VERIFY-146, SCENARIO-VERIFY-147"""

    def test_nup_probe_none_does_not_add_tier0c(self):
        """SCENARIO-VERIFY-147: nup_probe_v4=None → pipeline unchanged, no Tier 0c step.

        With a threshold that prevents EORM from clearing, response reaches Ising.
        If Tier 0c were somehow triggered with probe=None, it would short-circuit
        to "nup_probe_v4" instead of "ising" — this confirms it does not.
        """
        pipeline = _make_pipeline(sink_threshold=0.99, eorm_threshold=-999.0)
        _v, tier_used, _e = pipeline.verify("hello world", attention_matrix=None)
        assert tier_used == "ising"

    def test_nup_probe_low_score_clears_before_tier1(self):
        """SCENARIO-VERIFY-146: probe score ≤ threshold → tier_used='nup_probe_v4', verified=True."""
        nup = _make_nup_low_score()  # score always << 0
        pipeline = ThreeTierPipeline(
            sink_probe=SinkProbe(threshold=0.3),
            eorm_model=_make_eorm(),
            ising_pipeline=_ising_stub_wrong,  # would return False if reached
            sink_threshold=0.99,
            eorm_threshold=-999.0,
            nup_probe_v4=nup,
            nup_probe_threshold=0.0,
        )
        verified, tier_used, energy = pipeline.verify(
            "The answer is 42.", attention_matrix=None
        )
        assert tier_used == "nup_probe_v4"
        assert verified is True
        assert energy <= 0.0

    def test_nup_probe_high_score_falls_through(self):
        """When NUP score > threshold, Tier 0c does NOT clear — pipeline continues."""
        nup = _make_nup_high_score()  # score always >> 0
        pipeline = ThreeTierPipeline(
            sink_probe=SinkProbe(threshold=0.3),
            eorm_model=_make_eorm(),
            ising_pipeline=_ising_stub_correct,
            sink_threshold=0.99,
            eorm_threshold=-999.0,
            nup_probe_v4=nup,
            nup_probe_threshold=0.0,
        )
        _v, tier_used, _e = pipeline.verify("hello", attention_matrix=None)
        # High NUP score → not cleared by Tier 0c → falls through to Ising
        assert tier_used == "ising"

    def test_tier0c_skip_count_increments_in_benchmark(self):
        """benchmark() increments tier0c_skip_count for each Tier 0c early exit."""
        nup = _make_nup_low_score()
        pipeline = ThreeTierPipeline(
            sink_probe=SinkProbe(threshold=0.3),
            eorm_model=_make_eorm(),
            ising_pipeline=_ising_stub_correct,
            nup_probe_v4=nup,
            nup_probe_threshold=0.0,
        )
        responses = [{"response": "r", "question": "q", "attention_matrix": None} for _ in range(5)]
        result = pipeline.benchmark(responses, [True] * 5)
        assert result.tier0c_skip_count == 5

    def test_tier0c_skip_count_zero_when_probe_absent(self):
        """tier0c_skip_count stays 0 when nup_probe_v4=None."""
        pipeline = _make_pipeline()
        responses = [{"response": "r", "question": "q", "attention_matrix": None}]
        result = pipeline.benchmark(responses, [True])
        assert result.tier0c_skip_count == 0


# ---------------------------------------------------------------------------
# Tier 0d — HallucinationBasinDetector wiring
# ---------------------------------------------------------------------------


def _quadratic_energy(x: jnp.ndarray) -> float:
    """Simple quadratic energy proxy for CI tests: E(x) = sum(x^2)."""
    return float(jnp.sum(x ** 2))


def _make_basin_pipeline(basin_threshold: float = 0.5) -> ThreeTierPipeline:
    """Build a pipeline with a basin detector wired in."""
    detector = HallucinationBasinDetector(
        energy_fn=_quadratic_energy,
        n_perturbations=4,
        threshold=basin_threshold,
        perturbation_scale=0.01,  # tiny noise → deep basin for all states
    )
    return ThreeTierPipeline(
        sink_probe=SinkProbe(threshold=0.99),
        eorm_model=_make_eorm(),
        ising_pipeline=_ising_stub_correct,
        sink_threshold=0.99,
        eorm_threshold=-999.0,
        basin_detector=detector,
        basin_threshold=basin_threshold,
    )


class TestTier0dBasinDetector:
    """Spec: REQ-VERIFY-112, SCENARIO-VERIFY-148"""

    def test_basin_detector_skipped_when_hidden_states_none(self):
        """When hidden_states=None, Tier 0d is bypassed entirely (CI-safe).

        With sink_threshold=0.99 and eorm_threshold=-999, response reaches Ising.
        If Tier 0d ran without hidden_states it would error or short-circuit; this
        confirms it is skipped correctly.
        """
        pipeline = _make_basin_pipeline()
        _v, tier_used, _e = pipeline.verify("r", attention_matrix=None, hidden_states=None)
        assert tier_used == "ising"

    def test_basin_detector_low_risk_clears(self):
        """SCENARIO-VERIFY-148: low basin_risk_score ≤ threshold → tier_used='basin_detector'."""
        pipeline = _make_basin_pipeline(basin_threshold=0.9)  # wide threshold → always clears
        hidden = jnp.zeros((3, 8))  # zero hidden states → deep basin (risk ≈ 0.5 when depth=0)
        verified, tier_used, energy = pipeline.verify(
            "r", attention_matrix=None, hidden_states=hidden
        )
        assert tier_used == "basin_detector"
        assert verified is True

    def test_tier0d_skip_count_increments_in_benchmark(self):
        """benchmark() increments tier0d_skip_count for each Tier 0d early exit."""
        pipeline = _make_basin_pipeline(basin_threshold=0.9)
        hidden = jnp.zeros((2, 4))
        responses = [
            {"response": "r", "question": "q", "attention_matrix": None, "hidden_states": hidden}
            for _ in range(4)
        ]
        result = pipeline.benchmark(responses, [True] * 4)
        assert result.tier0d_skip_count == 4

    def test_tier0d_skip_count_zero_when_no_hidden_states(self):
        """tier0d_skip_count stays 0 when no hidden_states supplied in responses."""
        pipeline = _make_basin_pipeline(basin_threshold=0.9)
        responses = [{"response": "r", "question": "q", "attention_matrix": None}]
        result = pipeline.benchmark(responses, [True])
        assert result.tier0d_skip_count == 0

    def test_both_tiers_wired_clears_at_0c_when_nup_low(self):
        """When both 0c and 0d are wired and NUP score is low, response clears at 0c.

        0d is never reached because 0c fires first.
        """
        nup = _make_nup_low_score()
        detector = HallucinationBasinDetector(
            energy_fn=_quadratic_energy, n_perturbations=4, threshold=0.5, perturbation_scale=0.01
        )
        pipeline = ThreeTierPipeline(
            sink_probe=SinkProbe(threshold=0.99),
            eorm_model=_make_eorm(),
            ising_pipeline=_ising_stub_correct,
            sink_threshold=0.99,
            eorm_threshold=-999.0,
            nup_probe_v4=nup,
            nup_probe_threshold=0.0,
            basin_detector=detector,
            basin_threshold=0.9,
        )
        hidden = jnp.zeros((2, 4))
        _v, tier_used, _e = pipeline.verify(
            "The sky is blue.", attention_matrix=None, hidden_states=hidden
        )
        assert tier_used == "nup_probe_v4"

    def test_both_tiers_wired_nup_high_basin_low_clears_at_0d(self):
        """When NUP score is high (0c misses) but basin risk is low (0d clears)."""
        nup = _make_nup_high_score()  # NUP won't clear
        detector = HallucinationBasinDetector(
            energy_fn=_quadratic_energy, n_perturbations=4, threshold=0.5, perturbation_scale=0.01
        )
        pipeline = ThreeTierPipeline(
            sink_probe=SinkProbe(threshold=0.99),
            eorm_model=_make_eorm(),
            ising_pipeline=_ising_stub_correct,
            sink_threshold=0.99,
            eorm_threshold=-999.0,
            nup_probe_v4=nup,
            nup_probe_threshold=0.0,
            basin_detector=detector,
            basin_threshold=0.9,  # wide threshold → basin always clears
        )
        hidden = jnp.zeros((2, 4))
        _v, tier_used, _e = pipeline.verify(
            "some text", attention_matrix=None, hidden_states=hidden
        )
        assert tier_used == "basin_detector"
