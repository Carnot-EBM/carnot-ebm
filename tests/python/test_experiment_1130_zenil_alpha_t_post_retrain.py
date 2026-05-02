"""Tests for Exp 1130 — post-retrain Zenil alpha_t measurement.

These tests are intentionally CPU-only. They cover the parts that make the
experiment auditable: the Exp1077-compatible alpha_t calculation, low-energy
threshold calibration, cached SOTA row normalization, and the required artifact
schema.

Spec: REQ-FR11-1130, SCENARIO-FR11-1130, SCENARIO-FR11-1131.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval.zenil_alpha_post_retrain import (
    REQUIRED_ARTIFACT_FIELDS,
    EvaluationExample,
    build_exp1130_artifact,
    calibrate_low_energy_threshold,
    load_cached_sota_examples,
    measure_alpha_t_against_temperature,
    pearson_corr,
)


def test_calibrate_low_energy_threshold_prefers_correct_low_energy() -> None:
    """REQ-FR11-1130: verifier verdicts must preserve correct < incorrect ordering."""
    energies = [0.10, 0.20, 0.80, 0.90]
    labels = [1, 1, 0, 0]

    threshold = calibrate_low_energy_threshold(energies, labels)

    assert 0.20 < threshold < 0.80


def test_alpha_t_uses_exp1077_temperature_disagreement_formula() -> None:
    """REQ-FR11-1130: alpha_t is comparable to the Exp1077 0.38 baseline."""
    examples = [
        EvaluationExample("a", "q", "short", 1, 1),
        EvaluationExample("b", "q", "medium response", 2, 1),
        EvaluationExample("c", "q", "a much longer response", 3, 0),
        EvaluationExample("d", "q", "tiny", 4, 0),
    ]
    # Low energy predicts a/b correct. The length baseline predicts b/c correct.
    result = measure_alpha_t_against_temperature(
        examples=examples,
        energy_scores=[0.10, 0.20, 0.80, 0.90],
        energy_threshold=0.50,
    )

    assert result.alpha_t == pytest.approx(0.5)
    assert result.n_disagreements == 2
    assert result.verifier_verdicts == ["correct", "correct", "incorrect", "incorrect"]
    assert result.temperature_verdicts == ["incorrect", "correct", "correct", "incorrect"]


def test_pearson_corr_negated_energy_is_positive_for_good_verifier() -> None:
    """REQ-FR11-1130: diagnostics expose grounding correlation with label direction fixed."""
    labels = [1, 1, 0, 0]
    energies = [0.1, 0.2, 0.8, 0.9]

    assert pearson_corr([-e for e in energies], labels) > 0.95


def test_load_cached_sota_examples_normalizes_exp1077_rows(tmp_path: Path) -> None:
    """SCENARIO-FR11-1131: cached Qwen3.6 SOTA rows remain usable as fallback."""
    cache = tmp_path / "fr11.jsonl"
    rows = [
        {
            "question_id": "q0",
            "question": "What is 2+2?",
            "response": "2 + 2 = 4. Answer: 4",
            "correct_answer": 4,
            "correct": True,
            "model": "Qwen3.6-35B-A3B",
        },
        {
            "question_id": "small",
            "prompt": "ignored",
            "completion": "ignored",
            "correct_answer": 1,
            "is_correct": False,
            "model_name": "Qwen/Qwen3.5-0.8B",
        },
    ]
    cache.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    examples = load_cached_sota_examples(cache, n_examples=50)

    assert len(examples) == 1
    assert examples[0].example_id == "q0"
    assert examples[0].label == 1
    assert examples[0].response.endswith("Answer: 4")


def test_build_artifact_classifies_alpha_improvement() -> None:
    """SCENARIO-FR11-1130: improved alpha_t gets the required verdict."""
    artifact = build_exp1130_artifact(
        alpha_t_post_retrain=0.52,
        verifier_auroc_used=0.977419,
        n_evaluation_examples=50,
        inference_mode="cached",
        measurement_complete=True,
        fr11_logged=False,
        verifier_ground_truth_corr=0.41,
        thinkprm_ground_truth_corr=0.22,
        alpha_t_method="exp1077_temperature_disagreement",
        score_summary={"mean_energy": 0.3},
        examples_path="data/fr11_zenil_distill_v2.jsonl",
    )

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"missing required field: {field}"
    assert artifact["alpha_t_prior"] == 0.38
    assert artifact["alpha_t_improved"] is True
    assert artifact["honest_verdict"] == "alpha_t_improved"
    assert artifact["zenil_alpha_t_post_retrain_measured"] is True


def test_build_artifact_marks_incomplete_measurement() -> None:
    """REQ-FR11-1130: incomplete scoring must not masquerade as a degradation."""
    artifact = build_exp1130_artifact(
        alpha_t_post_retrain=0.0,
        verifier_auroc_used=0.977419,
        n_evaluation_examples=0,
        inference_mode="cached",
        measurement_complete=False,
        fr11_logged=False,
        verifier_ground_truth_corr=0.0,
        thinkprm_ground_truth_corr=0.0,
        alpha_t_method="exp1077_temperature_disagreement",
        score_summary={},
        examples_path="",
    )

    assert artifact["honest_verdict"] == "measurement_incomplete"
    assert artifact["alpha_t_improved"] is False


def test_generated_artifact_has_required_schema_when_present() -> None:
    """REQ-FR11-1130: generated deliverable must keep the required schema stable."""
    path = Path("results/experiment_1130_zenil_alpha_t_post_retrain.json")
    if not path.exists():
        pytest.skip("Exp1130 artifact has not been generated yet.")
    artifact = json.loads(path.read_text(encoding="utf-8"))

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"missing required field: {field}"
    assert artifact["n_evaluation_examples"] >= 1
    assert artifact["inference_mode"] in {"live_gpu", "cached"}
    assert artifact["honest_verdict"] in {
        "alpha_t_improved",
        "alpha_t_unchanged",
        "alpha_t_degraded",
        "measurement_incomplete",
    }
