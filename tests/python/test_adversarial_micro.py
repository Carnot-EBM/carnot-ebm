"""Tests for MicroAdversarialResult and build_micro_adversarial_artifact.

100% coverage of:
  - MicroAdversarialResult dataclass construction and field types.
  - build_micro_adversarial_artifact: all honest_verdict paths
      ('blocked', 'improvement_positive', 'degradation_positive', 'neutral').
  - robustness_claim logic: True only when repair_improvement_pct > 0 AND
    adversarial_drop_pct > 5 for at least one model.
  - _micro_result_to_dict serialization (tested indirectly via artifact output).
  - Multi-model aggregation (avg_adversarial_drop_pct, avg_repair_improvement_pct,
    headline_result = model with highest repair_improvement_pct).

Spec: REQ-BENCH-011, SCENARIO-BENCH-029, SCENARIO-BENCH-030
"""

from __future__ import annotations

import pytest

from carnot.pipeline.adversarial_gsm8k import (
    MicroAdversarialResult,
    build_micro_adversarial_artifact,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    model_id: str = "TestModel",
    n_questions: int = 50,
    standard_accuracy: float = 0.80,
    adversarial_accuracy: float = 0.70,
    repaired_accuracy: float = 0.75,
    adversarial_drop_pct: float = 10.0,
    repair_improvement_pct: float = 5.0,
    inference_mode: str = "live_gpu",
) -> MicroAdversarialResult:
    return MicroAdversarialResult(
        model_id=model_id,
        n_questions=n_questions,
        standard_accuracy=standard_accuracy,
        adversarial_accuracy=adversarial_accuracy,
        repaired_accuracy=repaired_accuracy,
        adversarial_drop_pct=adversarial_drop_pct,
        repair_improvement_pct=repair_improvement_pct,
        inference_mode=inference_mode,
    )


# ---------------------------------------------------------------------------
# MicroAdversarialResult construction
# ---------------------------------------------------------------------------


class TestMicroAdversarialResult:
    def test_fields_stored(self) -> None:
        r = _make_result()
        assert r.model_id == "TestModel"
        assert r.n_questions == 50
        assert r.standard_accuracy == pytest.approx(0.80)
        assert r.adversarial_accuracy == pytest.approx(0.70)
        assert r.repaired_accuracy == pytest.approx(0.75)
        assert r.adversarial_drop_pct == pytest.approx(10.0)
        assert r.repair_improvement_pct == pytest.approx(5.0)
        assert r.inference_mode == "live_gpu"

    def test_simulated_mode(self) -> None:
        r = _make_result(inference_mode="simulated")
        assert r.inference_mode == "simulated"

    def test_negative_drop_pct_allowed(self) -> None:
        # adversarial can sometimes be easier — no clamping
        r = _make_result(adversarial_drop_pct=-2.0)
        assert r.adversarial_drop_pct == pytest.approx(-2.0)

    def test_negative_repair_improvement_allowed(self) -> None:
        # repair can regress — no clamping
        r = _make_result(repair_improvement_pct=-3.0)
        assert r.repair_improvement_pct == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# build_micro_adversarial_artifact — blocked paths
# ---------------------------------------------------------------------------


class TestBuildMicroAdversarialArtifactBlocked:
    def test_empty_results_blocked(self) -> None:
        art = build_micro_adversarial_artifact([])
        assert art["honest_verdict"] == "blocked"
        assert art["robustness_claim"] is False
        assert art["inference_mode"] == "blocked"
        assert art["n_models"] == 0
        assert art["per_model_results"] == []
        assert art["headline_result"] is None

    def test_simulated_mode_blocked(self) -> None:
        r = _make_result(inference_mode="simulated")
        art = build_micro_adversarial_artifact([r])
        assert art["honest_verdict"] == "blocked"
        assert art["robustness_claim"] is False
        assert art["inference_mode"] == "blocked"

    def test_mixed_modes_blocked(self) -> None:
        # one live_gpu + one simulated → blocked (any non-live triggers it)
        r1 = _make_result(model_id="A", inference_mode="live_gpu")
        r2 = _make_result(model_id="B", inference_mode="simulated")
        art = build_micro_adversarial_artifact([r1, r2])
        assert art["honest_verdict"] == "blocked"

    def test_blocked_includes_per_model_results(self) -> None:
        r = _make_result(inference_mode="simulated")
        art = build_micro_adversarial_artifact([r])
        assert len(art["per_model_results"]) == 1
        assert art["per_model_results"][0]["model_id"] == "TestModel"

    def test_schema_always_present(self) -> None:
        art = build_micro_adversarial_artifact([])
        assert art["schema"] == "carnot.adversarial_micro.v1"


# ---------------------------------------------------------------------------
# build_micro_adversarial_artifact — improvement_positive
# ---------------------------------------------------------------------------


class TestBuildMicroAdversarialArtifactImprovement:
    def test_single_model_improvement_positive(self) -> None:
        r = _make_result(repair_improvement_pct=5.0, adversarial_drop_pct=10.0)
        art = build_micro_adversarial_artifact([r])
        assert art["honest_verdict"] == "improvement_positive"
        assert art["inference_mode"] == "live_gpu"

    def test_improvement_positive_even_if_small(self) -> None:
        r = _make_result(repair_improvement_pct=0.001, adversarial_drop_pct=1.0)
        art = build_micro_adversarial_artifact([r])
        assert art["honest_verdict"] == "improvement_positive"

    def test_multi_model_one_improves(self) -> None:
        r1 = _make_result(model_id="A", repair_improvement_pct=-2.0, adversarial_drop_pct=5.0)
        r2 = _make_result(model_id="B", repair_improvement_pct=3.0, adversarial_drop_pct=8.0)
        art = build_micro_adversarial_artifact([r1, r2])
        assert art["honest_verdict"] == "improvement_positive"


# ---------------------------------------------------------------------------
# build_micro_adversarial_artifact — degradation_positive
# ---------------------------------------------------------------------------


class TestBuildMicroAdversarialArtifactDegradation:
    def test_degradation_positive_no_improvement(self) -> None:
        r = _make_result(repair_improvement_pct=0.0, adversarial_drop_pct=8.0)
        art = build_micro_adversarial_artifact([r])
        assert art["honest_verdict"] == "degradation_positive"

    def test_degradation_positive_negative_improvement(self) -> None:
        r = _make_result(repair_improvement_pct=-1.0, adversarial_drop_pct=3.0)
        art = build_micro_adversarial_artifact([r])
        assert art["honest_verdict"] == "degradation_positive"


# ---------------------------------------------------------------------------
# build_micro_adversarial_artifact — neutral
# ---------------------------------------------------------------------------


class TestBuildMicroAdversarialArtifactNeutral:
    def test_neutral_no_drop_no_improvement(self) -> None:
        # extractor fully robust: no drop, repair not needed
        r = _make_result(repair_improvement_pct=0.0, adversarial_drop_pct=0.0)
        art = build_micro_adversarial_artifact([r])
        assert art["honest_verdict"] == "neutral"

    def test_neutral_negative_drop_zero_improvement(self) -> None:
        r = _make_result(repair_improvement_pct=0.0, adversarial_drop_pct=-2.0)
        art = build_micro_adversarial_artifact([r])
        assert art["honest_verdict"] == "neutral"


# ---------------------------------------------------------------------------
# robustness_claim logic
# ---------------------------------------------------------------------------


class TestRobustnessClaim:
    def test_robustness_claim_true(self) -> None:
        # repair_improvement_pct > 0 AND adversarial_drop_pct > 5
        r = _make_result(repair_improvement_pct=4.0, adversarial_drop_pct=10.0)
        art = build_micro_adversarial_artifact([r])
        assert art["robustness_claim"] is True

    def test_robustness_claim_false_drop_too_small(self) -> None:
        # improvement but drop <= 5 → claim not satisfied
        r = _make_result(repair_improvement_pct=4.0, adversarial_drop_pct=4.9)
        art = build_micro_adversarial_artifact([r])
        assert art["robustness_claim"] is False

    def test_robustness_claim_false_no_improvement(self) -> None:
        r = _make_result(repair_improvement_pct=0.0, adversarial_drop_pct=20.0)
        art = build_micro_adversarial_artifact([r])
        assert art["robustness_claim"] is False

    def test_robustness_claim_requires_both_conditions(self) -> None:
        r = _make_result(repair_improvement_pct=0.0, adversarial_drop_pct=0.0)
        art = build_micro_adversarial_artifact([r])
        assert art["robustness_claim"] is False

    def test_robustness_claim_multi_model_any_qualifies(self) -> None:
        r1 = _make_result(model_id="A", repair_improvement_pct=0.0, adversarial_drop_pct=20.0)
        r2 = _make_result(model_id="B", repair_improvement_pct=5.0, adversarial_drop_pct=10.0)
        art = build_micro_adversarial_artifact([r1, r2])
        assert art["robustness_claim"] is True

    def test_robustness_claim_blocked_always_false(self) -> None:
        r = _make_result(inference_mode="simulated", repair_improvement_pct=10.0, adversarial_drop_pct=20.0)
        art = build_micro_adversarial_artifact([r])
        assert art["robustness_claim"] is False


# ---------------------------------------------------------------------------
# Multi-model aggregation and headline_result
# ---------------------------------------------------------------------------


class TestMultiModelAggregation:
    def test_avg_drop_two_models(self) -> None:
        r1 = _make_result(model_id="A", adversarial_drop_pct=10.0, repair_improvement_pct=2.0)
        r2 = _make_result(model_id="B", adversarial_drop_pct=6.0, repair_improvement_pct=1.0)
        art = build_micro_adversarial_artifact([r1, r2])
        assert art["avg_adversarial_drop_pct"] == pytest.approx(8.0)

    def test_avg_improvement_two_models(self) -> None:
        r1 = _make_result(model_id="A", repair_improvement_pct=4.0)
        r2 = _make_result(model_id="B", repair_improvement_pct=2.0)
        art = build_micro_adversarial_artifact([r1, r2])
        assert art["avg_repair_improvement_pct"] == pytest.approx(3.0)

    def test_headline_result_is_best_model(self) -> None:
        r1 = _make_result(model_id="A", repair_improvement_pct=2.0)
        r2 = _make_result(model_id="B", repair_improvement_pct=8.0)
        art = build_micro_adversarial_artifact([r1, r2])
        assert art["headline_result"]["model_id"] == "B"

    def test_headline_result_serialized_fields(self) -> None:
        r = _make_result()
        art = build_micro_adversarial_artifact([r])
        hl = art["headline_result"]
        assert "model_id" in hl
        assert "n_questions" in hl
        assert "standard_accuracy" in hl
        assert "adversarial_accuracy" in hl
        assert "repaired_accuracy" in hl
        assert "adversarial_drop_pct" in hl
        assert "repair_improvement_pct" in hl
        assert "inference_mode" in hl

    def test_n_models_field(self) -> None:
        r1 = _make_result(model_id="A")
        r2 = _make_result(model_id="B")
        art = build_micro_adversarial_artifact([r1, r2])
        assert art["n_models"] == 2

    def test_per_model_results_count(self) -> None:
        results = [_make_result(model_id=f"M{i}") for i in range(3)]
        art = build_micro_adversarial_artifact(results)
        assert len(art["per_model_results"]) == 3
