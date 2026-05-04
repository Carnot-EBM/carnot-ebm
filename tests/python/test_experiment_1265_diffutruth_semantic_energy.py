"""Tests for Exp 1265 DiffuTruth semantic energy comparison.

Spec: REQ-VERIFY-1265, SCENARIO-VERIFY-1265
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import diffutruth_semantic_energy as exp


def test_semantic_energy_removes_last_sentence_for_req1265_proxy() -> None:
    """REQ-VERIFY-1265-2: reconstruction removes the response's last sentence."""

    response = "Stable facts stay low. Hallucinated tail becomes high energy."

    assert exp.reconstruct_response(response) == "Stable facts stay low"
    assert exp.reconstruct_response("One sentence only") == "One sentence only"
    assert exp.semantic_energy("One sentence only") == 0.0
    assert 0.0 < exp.semantic_energy(response) < 1.0


def test_tie_aware_auroc_handles_req1265_defined_and_undefined_inputs() -> None:
    """REQ-VERIFY-1265-4: AUROC is tie-aware and returns 0.5 when undefined."""

    assert exp.tie_aware_auroc([0, 1], [0.1, 0.9]) == pytest.approx(1.0)
    assert exp.tie_aware_auroc([0, 1], [0.9, 0.1]) == pytest.approx(0.0)
    assert exp.tie_aware_auroc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)
    assert exp.tie_aware_auroc([1, 1], [0.1, 0.2]) == pytest.approx(0.5)


def test_load_fover_pairs_maps_incorrect_to_positive_hallucination(tmp_path: Path) -> None:
    """REQ-VERIFY-1265-1: is_correct false becomes the positive hallucination label."""

    corpus_path = tmp_path / "fover_corpus_v5.json"
    corpus_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {"response": "Correct answer.", "is_correct": True},
                    {"response": "Wrong answer. Unsupported tail.", "is_correct": False},
                    {"response": "Missing label defaults correct."},
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = exp.load_fover_pairs(corpus_path, limit=2)

    assert [row.label for row in rows] == [0, 1]
    assert [row.response for row in rows] == [
        "Correct answer.",
        "Wrong answer. Unsupported tail.",
    ]


def test_load_carnot_baseline_prefers_auroc_auc_then_semenergy(tmp_path: Path) -> None:
    """REQ-VERIFY-1265-3: Carnot baseline extraction supports known artifact keys."""

    baseline_path = tmp_path / "experiment_1096_semenergy_probe_v1.json"

    baseline_path.write_text(json.dumps({"semenergy_auroc": 0.948187}), encoding="utf-8")
    assert exp.load_carnot_baseline_auroc(baseline_path) == pytest.approx(0.948187)

    baseline_path.write_text(
        json.dumps({"auc": 0.61, "semenergy_auroc": 0.948187}),
        encoding="utf-8",
    )
    assert exp.load_carnot_baseline_auroc(baseline_path) == pytest.approx(0.61)

    baseline_path.write_text(
        json.dumps({"auroc": 0.72, "auc": 0.61, "semenergy_auroc": 0.948187}),
        encoding="utf-8",
    )
    assert exp.load_carnot_baseline_auroc(baseline_path) == pytest.approx(0.72)


def test_run_experiment_writes_required_diffutruth_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1265: runner writes the required Exp 1265 artifact."""

    fover_path = tmp_path / "fover_corpus_v5.json"
    baseline_path = tmp_path / "experiment_1096_semenergy_probe_v1.json"
    output_path = tmp_path / "experiment_1265_diffutruth_vs_carnot_baseline.json"
    fover_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {"response": "The answer is 4.", "is_correct": True},
                    {
                        "response": "The answer is 4. Unsupported claim says 99 99 99.",
                        "is_correct": False,
                    },
                    {"response": "The answer is 5.", "is_correct": True},
                    {
                        "response": "The answer is 5. False extra trail repeats 42 42.",
                        "is_correct": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_path.write_text(json.dumps({"semenergy_auroc": 0.948187}), encoding="utf-8")

    artifact = exp.run_experiment(
        fover_path=fover_path,
        carnot_baseline_path=baseline_path,
        output_path=output_path,
        limit=4,
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact == persisted
    assert exp.REQUIRED_ARTIFACT_FIELDS <= set(persisted)
    assert persisted["experiment"] == "1265_diffutruth_vs_carnot_baseline"
    assert persisted["status"] == "complete"
    assert persisted["n_pairs"] == 4
    assert persisted["diffutruth_semantic_energy_auroc"] == pytest.approx(1.0)
    assert persisted["carnot_semenergy_probe_auroc"] == pytest.approx(0.948187)
    assert persisted["diffutruth_fever_paper_auroc"] == pytest.approx(0.725)
    assert persisted["carnot_beats_diffutruth_paper"] is True
    assert persisted["diffutruth_comparison_measured"] is True
    assert persisted["honest_verdict"] == "diffutruth_fover_1.000_carnot_0.948"
