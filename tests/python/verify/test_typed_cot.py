"""Tests for Exp 2396 Typed CoT verifier.

Spec: REQ-TIER28-002, SCENARIO-TIER28-002
"""

from __future__ import annotations

import json

from carnot.verify.typed_cot import (
    StepType,
    TypedCoTVerifier,
    build_experiment_artifact,
    extract_cot_text,
    validate_experiment_artifact,
)


def test_classify_step_uses_curry_howard_surface_types() -> None:
    """REQ-TIER28-002-1: heuristic surface patterns map to proof-step types."""

    verifier = TypedCoTVerifier()

    assert verifier.classify_step("The premise is true.", 0, 3) is StepType.PROPOSITION
    assert verifier.classify_step("Because the premise is true, the claim follows.", 1, 3) is StepType.INFERENCE
    assert verifier.classify_step("Therefore the answer follows.", 2, 3) is StepType.CONCLUSION


def test_verify_text_scores_dependency_consistency() -> None:
    """SCENARIO-TIER28-002: proposition -> inference -> conclusion type-checks."""

    result = TypedCoTVerifier().verify_text(
        "The premise is available. Because the premise is available, the claim follows. "
        "Therefore the conclusion is supported."
    )

    assert result["typed_cot_score"] == 1.0
    assert [step["type"] for step in result["typed_steps"]] == [
        "Proposition",
        "Inference",
        "Conclusion",
    ]
    assert all(step["type_checks"] for step in result["typed_steps"])


def test_conclusion_before_inference_fails_dependency_check() -> None:
    """REQ-TIER28-002-3: conclusions require an earlier inference."""

    result = TypedCoTVerifier().verify_text(
        "The premise is available. Therefore the answer follows."
    )

    assert result["typed_cot_score"] == 0.5
    assert result["typed_steps"][1]["type"] == "Conclusion"
    assert result["typed_steps"][1]["type_checks"] is False


def test_extract_cot_text_prefers_explicit_reasoning_fields() -> None:
    """REQ-TIER28-002-4: telemetry extraction records the source CoT field."""

    text, field_name = extract_cot_text(
        {
            "chain_of_thought": "The premise holds. Therefore the result follows.",
            "reasoning": "unused",
            "response_text": "unused",
        }
    )

    assert text.startswith("The premise")
    assert field_name == "chain_of_thought"


def test_build_experiment_artifact_uses_realistic_manifest_shape(tmp_path) -> None:
    """REQ-TIER28-002-4: artifact contains required fields for telemetry runs."""

    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "case_id": "correct",
            "correctness_label": "correct",
            "response_text": "<think>The premise holds. Because it holds, the result follows. Therefore answer 1 follows.</think>\n1",
        },
        {
            "case_id": "incorrect",
            "correctness_label": "incorrect",
            "response_text": "<think>Therefore answer 0 follows.</think>\n0",
        },
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    artifact = build_experiment_artifact(manifest_path=manifest, n_eval_examples=2)

    assert artifact["typed_cot_validated"] is True
    assert artifact["n_eval_examples"] == 2
    assert artifact["random_seed"] == 42
    assert artifact["cot_fields_found"] == ["response_text"]
    assert artifact["honest_verdict"].startswith("complete:")
    validate_experiment_artifact(artifact, expected_n_eval_examples=2)
