"""Tests for Exp 1367 DiffuTruth energy-of-falsehood FoVer probe.

Spec: REQ-VERIFY-1367, SCENARIO-VERIFY-1367
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import diffutruth_falsehood_probe as exp


def test_write_in_progress_artifact_for_req1367(tmp_path: Path) -> None:
    """REQ-VERIFY-1367-1: the workflow writes an in-progress artifact first."""

    output_path = tmp_path / "experiment_1367_diffutruth_energy_of_falsehood_probe.json"

    artifact = exp.write_in_progress_artifact(output_path)
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact == persisted
    assert persisted["status"] == "in_progress"
    assert persisted["honest_verdict"] == "in_progress"


def test_perturbation_and_reconstruction_energy_separate_stability() -> None:
    """REQ-VERIFY-1367-3/4: unstable claims keep more corruption and score higher."""

    stable = exp.FoVerClaimCase(
        case_id="stable",
        question="What is 2 + 2?",
        response="First compute 2 + 2 = 4. Therefore the answer is 4.",
        label=0,
    )
    unstable = exp.FoVerClaimCase(
        case_id="unstable",
        question="What is 2 + 2?",
        response="The answer is 42.",
        label=1,
    )

    stable_score = exp.score_case(stable, seed=1367)
    unstable_score = exp.score_case(unstable, seed=1367)

    assert stable_score.corrupted_text != stable.response
    assert unstable_score.corrupted_text != unstable.response
    assert stable_score.semantic_similarity > unstable_score.semantic_similarity
    assert unstable_score.diffutruth_energy > stable_score.diffutruth_energy


def test_build_artifact_computes_required_metrics_and_viability() -> None:
    """REQ-VERIFY-1367-5/6/7: metrics, correlations, and gate are deterministic."""

    cases = [
        exp.FoVerClaimCase("c0", "", "stable a", 0),
        exp.FoVerClaimCase("h0", "", "unstable a", 1),
        exp.FoVerClaimCase("c1", "", "stable b", 0),
        exp.FoVerClaimCase("h1", "", "unstable b", 1),
    ]
    scores = [
        exp.ScoredDiffuTruthCase("c0", 0, "x", "x", 0.9, 0.1, 0.9, 2, 0.0),
        exp.ScoredDiffuTruthCase("h0", 1, "x", "x", 0.1, 0.9, 0.1, 2, 1.0),
        exp.ScoredDiffuTruthCase("c1", 0, "x", "x", 0.8, 0.2, 0.8, 2, 0.0),
        exp.ScoredDiffuTruthCase("h1", 1, "x", "x", 0.2, 0.8, 0.2, 2, 1.0),
    ]

    artifact = exp.build_artifact(
        cases,
        scores,
        ising_scores=[0.1, 0.9, 0.2, 0.8],
        kan_scores=[0.2, 0.7, 0.3, 0.6],
        corpus_path="fixture.json",
        run_date="20260505",
    )

    assert exp.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["corpus_cases_used"] == 4
    assert artifact["diffutruth_energy_delta_mean"] == pytest.approx(0.7)
    assert artifact["detection_auroc_proxy"] == pytest.approx(1.0)
    assert artifact["hallucination_energy_rate"] == pytest.approx(1.0)
    assert artifact["ising_correlation"] > 0.9
    assert artifact["kan_correlation"] > 0.9
    assert artifact["viable_as_complement"] is True


def test_run_experiment_writes_complete_req1367_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1367: runner writes the complete required artifact."""

    corpus_path = tmp_path / "fover_corpus_v5.json"
    output_path = tmp_path / "experiment_1367_diffutruth_energy_of_falsehood_probe.json"
    corpus_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "question_index": 0,
                        "question": "What is 2 + 2?",
                        "response": "First compute 2 + 2 = 4. Therefore the answer is 4.",
                        "is_correct": True,
                    },
                    {
                        "question_index": 1,
                        "question": "What is 3 + 3?",
                        "response": "First compute 3 + 3 = 6. Therefore the answer is 6.",
                        "is_correct": True,
                    },
                    {
                        "question_index": 2,
                        "question": "What is 2 + 2?",
                        "response": "The answer is 42.",
                        "is_correct": False,
                    },
                    {
                        "question_index": 3,
                        "question": "What is 3 + 3?",
                        "response": "The answer is 99.",
                        "is_correct": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    artifact = exp.run_experiment(
        corpus_path=corpus_path,
        output_path=output_path,
        limit=4,
        use_kan_adapter=False,
        run_date="20260505",
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert artifact == persisted
    assert persisted["status"] == "complete"
    assert persisted["corpus_cases_used"] == 4
    assert persisted["perturbation_method"]
    assert persisted["reconstruction_method"]
    assert isinstance(persisted["honest_verdict"], str)
