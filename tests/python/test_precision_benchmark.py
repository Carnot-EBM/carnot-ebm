"""Tests for python/carnot/pipeline/precision_benchmark.py.

Covers:
- PipelineVariant enum members and values
- compute_signed_improvement (positive, negative, zero — no clamping)
- PrecisionStackResult dataclass construction and asdict serialization
- build_precision_benchmark_artifact:
    - schema
    - headline_result with positive signed_improvement → headline_label
    - headline_result with negative signed_improvement → no headline_label
    - inference_mode resolution (single, empty, mixed)
    - honest_verdict for simulated mode
    - empty results list
    - FULL_STACK on non-headline model (no headline_result populated)

Spec: REQ-BENCH-003, SCENARIO-BENCH-007, SCENARIO-BENCH-008, SCENARIO-BENCH-009
"""

from __future__ import annotations

import dataclasses

import pytest

from carnot.pipeline.precision_benchmark import (
    PipelineVariant,
    PrecisionStackResult,
    build_precision_benchmark_artifact,
    compute_signed_improvement,
)


# ---------------------------------------------------------------------------
# PipelineVariant
# ---------------------------------------------------------------------------


class TestPipelineVariant:
    """REQ-BENCH-003: PipelineVariant enum members."""

    def test_all_five_members_exist(self):
        """SCENARIO-BENCH-007: All five ablation conditions must be present."""
        members = {v.name for v in PipelineVariant}
        assert members == {
            "BASELINE",
            "CONFIDENCE_ONLY",
            "CONFIDENCE_ADAPTIVE",
            "CONFIDENCE_ADAPTIVE_VERGE",
            "FULL_STACK",
        }

    def test_string_values(self):
        """PipelineVariant members have expected string values for JSON serialization."""
        assert PipelineVariant.BASELINE.value == "baseline"
        assert PipelineVariant.CONFIDENCE_ONLY.value == "confidence_only"
        assert PipelineVariant.CONFIDENCE_ADAPTIVE.value == "confidence_adaptive"
        assert PipelineVariant.CONFIDENCE_ADAPTIVE_VERGE.value == "confidence_adaptive_verge"
        assert PipelineVariant.FULL_STACK.value == "full_stack"

    def test_is_str_enum(self):
        """PipelineVariant inherits from str so enum values compare equal to plain strings."""
        assert PipelineVariant.BASELINE == "baseline"
        assert PipelineVariant.FULL_STACK == "full_stack"


# ---------------------------------------------------------------------------
# compute_signed_improvement
# ---------------------------------------------------------------------------


class TestComputeSignedImprovement:
    """REQ-BENCH-003, SCENARIO-BENCH-007: compute_signed_improvement."""

    def test_positive_improvement(self):
        """SCENARIO-BENCH-007: 0.65 - 0.50 = 0.15 (positive improvement)."""
        result = compute_signed_improvement(0.50, 0.65)
        assert abs(result - 0.15) < 1e-9

    def test_negative_improvement_no_clamping(self):
        """SCENARIO-BENCH-007: negative result is preserved — no clamping."""
        result = compute_signed_improvement(0.65, 0.60)
        assert abs(result - (-0.05)) < 1e-9

    def test_zero_improvement(self):
        """When stack equals baseline, signed_improvement is exactly 0."""
        result = compute_signed_improvement(0.50, 0.50)
        assert result == 0.0

    def test_large_positive(self):
        """Accepts values outside [0,1] without clamping."""
        result = compute_signed_improvement(0.0, 1.0)
        assert result == 1.0

    def test_large_negative(self):
        """Accepts large negative values without clamping."""
        result = compute_signed_improvement(1.0, 0.0)
        assert result == -1.0

    def test_returns_float(self):
        """Return type is float."""
        result = compute_signed_improvement(0.3, 0.4)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# PrecisionStackResult dataclass
# ---------------------------------------------------------------------------


class TestPrecisionStackResult:
    """REQ-BENCH-003, SCENARIO-BENCH-007: PrecisionStackResult dataclass."""

    def _make_result(self, **overrides) -> PrecisionStackResult:
        defaults = dict(
            model_id="Gemma4-E4B-it",
            n_questions=200,
            baseline_accuracy=0.50,
            precision_stack_accuracy=0.60,
            signed_improvement=0.10,
            pipeline_variant=PipelineVariant.FULL_STACK,
            inference_mode="simulated",
        )
        defaults.update(overrides)
        return PrecisionStackResult(**defaults)

    def test_construction_with_required_fields(self):
        """SCENARIO-BENCH-007: dataclass constructs with all required fields."""
        r = self._make_result()
        assert r.model_id == "Gemma4-E4B-it"
        assert r.n_questions == 200
        assert r.baseline_accuracy == 0.50
        assert r.precision_stack_accuracy == 0.60
        assert r.signed_improvement == 0.10
        assert r.pipeline_variant == PipelineVariant.FULL_STACK
        assert r.inference_mode == "simulated"

    def test_default_counter_fields(self):
        """Optional counter fields default to 0."""
        r = self._make_result()
        assert r.n_violations_found == 0
        assert r.n_repairs_attempted == 0
        assert r.n_repairs_improved == 0
        assert r.n_repairs_broken == 0

    def test_counter_fields_settable(self):
        """Counter fields can be set explicitly."""
        r = self._make_result(
            n_violations_found=15,
            n_repairs_attempted=8,
            n_repairs_improved=5,
            n_repairs_broken=3,
        )
        assert r.n_violations_found == 15
        assert r.n_repairs_attempted == 8
        assert r.n_repairs_improved == 5
        assert r.n_repairs_broken == 3

    def test_asdict_serializable(self):
        """SCENARIO-BENCH-007: dataclasses.asdict() produces a dict."""
        r = self._make_result()
        d = dataclasses.asdict(r)
        assert isinstance(d, dict)
        assert d["model_id"] == "Gemma4-E4B-it"
        assert d["signed_improvement"] == 0.10
        # pipeline_variant is an enum; asdict keeps enum value as-is
        assert d["pipeline_variant"] == PipelineVariant.FULL_STACK

    def test_negative_signed_improvement_stored(self):
        """Negative signed_improvement is stored without modification."""
        r = self._make_result(signed_improvement=-0.05)
        assert r.signed_improvement == -0.05

    def test_all_pipeline_variants_accepted(self):
        """Any PipelineVariant value is accepted in the dataclass."""
        for variant in PipelineVariant:
            r = self._make_result(pipeline_variant=variant)
            assert r.pipeline_variant == variant


# ---------------------------------------------------------------------------
# build_precision_benchmark_artifact
# ---------------------------------------------------------------------------


def _make_result(model_id: str, variant: PipelineVariant, signed_improvement: float,
                 inference_mode: str = "simulated") -> PrecisionStackResult:
    return PrecisionStackResult(
        model_id=model_id,
        n_questions=200,
        baseline_accuracy=0.50,
        precision_stack_accuracy=0.50 + signed_improvement,
        signed_improvement=signed_improvement,
        pipeline_variant=variant,
        inference_mode=inference_mode,
    )


class TestBuildPrecisionBenchmarkArtifact:
    """REQ-BENCH-003, SCENARIO-BENCH-008, SCENARIO-BENCH-009."""

    def test_schema_field(self):
        """SCENARIO-BENCH-008: artifact precision_schema is 'carnot.precision_benchmark.v1'."""
        artifact = build_precision_benchmark_artifact([
            _make_result("Gemma4-E4B-it", PipelineVariant.FULL_STACK, 0.08),
        ])
        assert artifact["precision_schema"] == "carnot.precision_benchmark.v1"

    def test_headline_result_positive_improvement(self):
        """SCENARIO-BENCH-008: positive signed_improvement → headline_label set."""
        artifact = build_precision_benchmark_artifact([
            _make_result("Gemma4-E4B-it", PipelineVariant.FULL_STACK, 0.08),
        ])
        hr = artifact["headline_result"]
        assert hr["signed_improvement"] == pytest.approx(0.08)
        assert hr["headline_label"] == "first_positive_live_it_result"

    def test_headline_result_negative_no_label(self):
        """SCENARIO-BENCH-008: negative signed_improvement → no headline_label."""
        artifact = build_precision_benchmark_artifact([
            _make_result("Gemma4-E4B-it", PipelineVariant.FULL_STACK, -0.03),
        ])
        hr = artifact["headline_result"]
        assert hr["signed_improvement"] == pytest.approx(-0.03)
        assert "headline_label" not in hr

    def test_headline_result_zero_no_label(self):
        """Zero signed_improvement → no headline_label (must be strictly positive)."""
        artifact = build_precision_benchmark_artifact([
            _make_result("Gemma4-E4B-it", PipelineVariant.FULL_STACK, 0.0),
        ])
        assert "headline_label" not in artifact["headline_result"]

    def test_headline_result_model_id_field(self):
        """headline_result carries model_id and pipeline_variant as string."""
        artifact = build_precision_benchmark_artifact([
            _make_result("Gemma4-E4B-it", PipelineVariant.FULL_STACK, 0.05),
        ])
        hr = artifact["headline_result"]
        assert hr["model_id"] == "Gemma4-E4B-it"
        assert hr["pipeline_variant"] == "full_stack"

    def test_non_headline_model_does_not_set_headline(self):
        """FULL_STACK result for non-Gemma4 model does not become headline."""
        artifact = build_precision_benchmark_artifact([
            _make_result("Qwen3.5-0.8B", PipelineVariant.FULL_STACK, 0.10),
        ])
        assert artifact["headline_result"] == {}

    def test_baseline_variant_not_headline(self):
        """BASELINE variant for Gemma4-E4B-it is not the headline result."""
        artifact = build_precision_benchmark_artifact([
            _make_result("Gemma4-E4B-it", PipelineVariant.BASELINE, 0.10),
        ])
        assert artifact["headline_result"] == {}

    def test_empty_results_list(self):
        """Empty results list → empty headline_result, unknown inference_mode."""
        artifact = build_precision_benchmark_artifact([])
        assert artifact["headline_result"] == {}
        assert artifact["inference_mode"] == "unknown"
        assert artifact["all_results"] == []

    def test_inference_mode_simulated(self):
        """SCENARIO-BENCH-009: all simulated results → inference_mode='simulated'."""
        results = [
            _make_result("Gemma4-E4B-it", PipelineVariant.BASELINE, 0.0, "simulated"),
            _make_result("Gemma4-E4B-it", PipelineVariant.FULL_STACK, 0.05, "simulated"),
        ]
        artifact = build_precision_benchmark_artifact(results)
        assert artifact["inference_mode"] == "simulated"

    def test_honest_verdict_simulated_only(self):
        """SCENARIO-BENCH-009: simulated mode adds honest_verdict='simulated_only'."""
        results = [_make_result("Gemma4-E4B-it", PipelineVariant.FULL_STACK, 0.05, "simulated")]
        artifact = build_precision_benchmark_artifact(results)
        assert artifact["honest_verdict"] == "simulated_only"

    def test_no_honest_verdict_for_live_gpu(self):
        """live_gpu inference_mode does NOT add honest_verdict."""
        results = [_make_result("Gemma4-E4B-it", PipelineVariant.FULL_STACK, 0.05, "live_gpu")]
        artifact = build_precision_benchmark_artifact(results)
        assert "honest_verdict" not in artifact

    def test_inference_mode_live_gpu(self):
        """All live_gpu results → inference_mode='live_gpu'."""
        results = [
            _make_result("Gemma4-E4B-it", PipelineVariant.BASELINE, 0.0, "live_gpu"),
            _make_result("Gemma4-E4B-it", PipelineVariant.FULL_STACK, 0.05, "live_gpu"),
        ]
        artifact = build_precision_benchmark_artifact(results)
        assert artifact["inference_mode"] == "live_gpu"
        assert "honest_verdict" not in artifact

    def test_inference_mode_mixed(self):
        """Mixed inference modes → inference_mode='mixed'."""
        results = [
            _make_result("Gemma4-E4B-it", PipelineVariant.BASELINE, 0.0, "simulated"),
            _make_result("Gemma4-E4B-it", PipelineVariant.FULL_STACK, 0.05, "live_gpu"),
        ]
        artifact = build_precision_benchmark_artifact(results)
        assert artifact["inference_mode"] == "mixed"

    def test_all_results_serialized(self):
        """all_results contains one entry per input with pipeline_variant as string."""
        results = [
            _make_result("Gemma4-E4B-it", PipelineVariant.BASELINE, 0.0),
            _make_result("Gemma4-E4B-it", PipelineVariant.FULL_STACK, 0.05),
            _make_result("Qwen3.5-0.8B", PipelineVariant.BASELINE, 0.0),
        ]
        artifact = build_precision_benchmark_artifact(results)
        assert len(artifact["all_results"]) == 3
        # pipeline_variant should be string, not enum
        for entry in artifact["all_results"]:
            assert isinstance(entry["pipeline_variant"], str)

    def test_first_full_stack_gemma_wins_headline(self):
        """When multiple FULL_STACK Gemma4 results exist, the first is headline."""
        results = [
            _make_result("Gemma4-E4B-it", PipelineVariant.FULL_STACK, 0.10),
            _make_result("Gemma4-E4B-it", PipelineVariant.FULL_STACK, 0.20),
        ]
        artifact = build_precision_benchmark_artifact(results)
        assert artifact["headline_result"]["signed_improvement"] == pytest.approx(0.10)

    def test_multiple_models_all_variants(self):
        """Full 5-variant × 2-model scenario produces 10 all_results entries."""
        results = []
        for model in ["Gemma4-E4B-it", "Qwen3.5-0.8B"]:
            for variant in PipelineVariant:
                results.append(_make_result(model, variant, 0.02))
        artifact = build_precision_benchmark_artifact(results)
        assert len(artifact["all_results"]) == 10
