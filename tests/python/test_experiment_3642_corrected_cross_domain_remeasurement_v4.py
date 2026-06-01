"""Tests for Exp 3642 corrected cross-domain remeasurement.

Spec: REQ-VERIFY-3642, SCENARIO-VERIFY-3642.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify.corrected_cross_domain_remeasurement_v4 import (
    REQUIRED_ARTIFACT_FIELDS,
    build_artifact,
    score_fact_rows,
    validate_artifact,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _seed_cross_domain_fixture(
    root: Path,
    *,
    facts_validated: bool,
    code_verifiers_fire: bool,
) -> tuple[list[dict], list[dict]]:
    results = root / "results"
    data = root / "data"

    facts_rows = [
        {
            "answer": "Paris",
            "evidence_passage": "Paris is the capital city of France.",
            "is_hallucination": 0,
            "model_confidence": 0.7,
            "question": "What is the capital of France?",
        },
        {
            "answer": "Lyon",
            "evidence_passage": "Lyon is a city in France, while Paris is the capital.",
            "is_hallucination": 0,
            "model_confidence": 0.3,
            "question": "Name a French city in the evidence.",
        },
        {
            "answer": "Berlin",
            "evidence_passage": "Paris is the capital city of France.",
            "is_hallucination": 1,
            "model_confidence": 0.8,
            "question": "What is the capital of France?",
        },
        {
            "answer": "France",
            "evidence_passage": "France is a country in Europe.",
            "is_hallucination": 1,
            "model_confidence": 0.4,
            "question": "What country is mentioned?",
        },
    ]
    code_rows = [
        {"candidate_code": "def f():\n    return 1\n", "label": True, "task_id": "a"},
        {"candidate_code": "def g():\n    return 2\n", "label": True, "task_id": "b"},
        {"candidate_code": "def h(:\n    return 3\n", "label": False, "task_id": "c"},
        {"candidate_code": "def i():\n    pass\n", "label": False, "task_id": "d"},
    ]
    _write_jsonl(data / "facts.jsonl", facts_rows)
    _write_jsonl(data / "code.jsonl", code_rows)
    _write_json(
        results / "experiment_3640_build_factual_corpus_v3.json",
        {
            "corpus_path_used": "data/facts.jsonl",
            "facts_corpus_validated": facts_validated,
            "confidence_baseline_auroc_on_corpus": 0.5,
        },
    )
    _write_json(
        results / "experiment_3641_code_corpus_verifiers_fire_transfer_v3.json",
        {
            "code_corpus_path": "data/code.jsonl",
            "code_verifiers_fire": code_verifiers_fire,
        },
    )
    _write_json(
        results / "experiment_2837_fover_memory_leakage_v3.json",
        {"condition_a_production_auroc_mean": 0.9131335999999999},
    )
    return facts_rows, code_rows


@pytest.mark.parametrize(
    (
        "facts_validated",
        "code_verifiers_fire",
        "code_scores",
        "facts_scores",
        "expected_verdict",
        "expected_positive_control",
        "expected_at_least_one",
        "expected_code_generalizes",
        "expected_facts_generalize",
    ),
    [
        (
            True,
            True,
            [0.1, 0.8, 0.7, 0.9],
            [0.1, 0.8, 0.7, 0.9],
            "complete: verifier_value_generalizes_beyond_math_329_null_was_artifact",
            True,
            True,
            True,
            True,
        ),
        (
            False,
            True,
            [0.1, 0.8, 0.7, 0.9],
            [0.1, 0.8, 0.7, 0.9],
            "complete: verifier_value_generalizes_to_code_not_facts_partial_scope",
            False,
            True,
            True,
            False,
        ),
        (
            True,
            False,
            [0.1, 0.8, 0.7, 0.9],
            [0.1, 0.8, 0.7, 0.9],
            "complete: verifier_value_generalizes_to_facts_not_code_partial_scope",
            False,
            True,
            False,
            True,
        ),
        (
            True,
            True,
            [0.4, 0.6, 0.3, 0.5],
            [0.4, 0.6, 0.3, 0.5],
            "complete: verifier_value_math_only_EARNED_against_valid_positive_control_scoped_limitation",
            True,
            True,
            False,
            False,
        ),
    ],
)
def test_exp3642_parametrizes_honest_row_outcomes(
    tmp_path: Path,
    facts_validated: bool,
    code_verifiers_fire: bool,
    code_scores: list[float],
    facts_scores: list[float],
    expected_verdict: str,
    expected_positive_control: bool,
    expected_at_least_one: bool,
    expected_code_generalizes: bool,
    expected_facts_generalize: bool,
) -> None:
    """SCENARIO-VERIFY-3642: ran/blocked rows drive the verdict honestly."""

    _seed_cross_domain_fixture(
        tmp_path,
        facts_validated=facts_validated,
        code_verifiers_fire=code_verifiers_fire,
    )
    confidence_scores = [0.3, 0.7, 0.2, 0.6]
    artifact = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=2.0,
        n_bootstrap=20,
        score_overrides={
            "code": {
                "ensemble_scores": code_scores,
                "confidence_scores": confidence_scores,
            },
            "facts": {
                "ensemble_scores": facts_scores,
                "confidence_scores": confidence_scores,
            },
        },
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert type(artifact["positive_control_valid"]) is bool
    assert type(artifact["at_least_one_nonmath_row_ran"]) is bool
    assert artifact["positive_control_valid"] is expected_positive_control
    assert artifact["at_least_one_nonmath_row_ran"] is expected_at_least_one
    assert artifact["code_generalizes"] is expected_code_generalizes
    assert artifact["facts_generalize"] is expected_facts_generalize
    assert set(artifact["generalization_table"]) == {"math", "code", "facts"}
    assert artifact["math_ensemble_auroc"] == 0.9131
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])

    for domain in ("code", "facts"):
        row = artifact["generalization_table"][domain]
        if row["ran_or_blocked"] == "ran":
            assert row["ensemble_auroc"]["point"] is not None
            assert row["confidence_auroc"]["ci95"] is not None
            assert row["delta"]["ci95"] is not None
            assert row["class_balance"]["positive_errors"] == 2
            assert row["class_balance"]["negative_correct"] == 2
        else:
            assert row["ensemble_auroc"] is None
            assert row["blocked_reason"].startswith("blocked_")


def test_exp3642_facts_verifier_sees_only_model_answer_and_evidence() -> None:
    """REQ-VERIFY-3642: grounding energy has no gold-answer or label input path."""

    class SpyVerifier:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def verify(self, answer: str, context: str) -> float:
            self.calls.append((answer, context))
            assert "SECRET_GOLD" not in answer
            assert "SECRET_GOLD" not in context
            return 0.25

    rows = [
        {
            "answer": "model answer",
            "evidence_passage": "evidence passage",
            "gold_answer": "SECRET_GOLD",
            "is_hallucination": 1,
            "model_confidence": 0.2,
        }
    ]
    verifier = SpyVerifier()
    scores = score_fact_rows(rows, verifier=verifier)
    assert scores == [0.25]
    assert verifier.calls == [("model answer", "evidence passage")]


def test_exp3642_validate_rejects_wrapped_control_booleans(tmp_path: Path) -> None:
    """REQ-VERIFY-3642: downstream gates require bare JSON booleans."""

    _seed_cross_domain_fixture(tmp_path, facts_validated=True, code_verifiers_fire=True)
    artifact = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=2.0,
        n_bootstrap=20,
        score_overrides={
            "code": {
                "ensemble_scores": [0.1, 0.8, 0.7, 0.9],
                "confidence_scores": [0.3, 0.7, 0.2, 0.6],
            },
            "facts": {
                "ensemble_scores": [0.1, 0.8, 0.7, 0.9],
                "confidence_scores": [0.3, 0.7, 0.2, 0.6],
            },
        },
    )
    artifact["positive_control_valid"] = {"value": True}
    with pytest.raises(ValueError, match="positive_control_valid"):
        validate_artifact(artifact)
