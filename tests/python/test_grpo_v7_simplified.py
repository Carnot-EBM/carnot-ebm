"""Tests for Exp 1247 GRPO v7 simplified VPS-only artifact helpers.

Spec: REQ-LEARN-1247, SCENARIO-LEARN-1247, SCENARIO-LEARN-1248.
"""

from __future__ import annotations

import json

import pytest

from carnot.training.grpo_v7_simplified import (
    ALLOWED_GRPO_V7_DEVICES,
    REQUIRED_GRPO_V7_ARTIFACT_FIELDS,
    build_grpo_v7_simplified_artifact,
    derive_grpo_v7_honest_verdict,
    extract_exp1220_accuracy,
    validate_grpo_v7_artifact,
    write_grpo_v7_simplified_artifact,
)


def test_build_artifact_has_required_vps_only_schema() -> None:
    """REQ-LEARN-1247-1/-2/-3: schema is VPS-only with 20 train / 30 eval."""
    artifact = build_grpo_v7_simplified_artifact(
        baseline_accuracy=0.8,
        final_accuracy=0.95,
        device_used="cpu",
        fallback_used=False,
    )

    for field in REQUIRED_GRPO_V7_ARTIFACT_FIELDS:
        assert field in artifact, f"missing required field: {field}"
    assert artifact["grpo_v7_ran"] is True
    assert artifact["training_mode"] == "vps_only"
    assert artifact["verifier_type"] == "vps_only"
    assert artifact["n_training_questions"] == 20
    assert artifact["n_eval_questions"] == 30
    assert artifact["improvement_pp"] == pytest.approx(15.0)
    assert artifact["honest_verdict"] == "grpo_v7_improvement_pp_15.0"


def test_allowed_devices_match_req_1247() -> None:
    """REQ-LEARN-1247-4: allowed device values are exactly the spec tokens."""
    assert ALLOWED_GRPO_V7_DEVICES == {"cuda:0", "cpu", "llama_cpp", "fallback"}


def test_extract_exp1220_accuracy_prefers_vps_training_fields() -> None:
    """SCENARIO-LEARN-1247: Exp 1220 VPS fields map to baseline/final."""
    baseline, final = extract_exp1220_accuracy(
        {
            "grpo_vps_fraction_correct_before": 0.8,
            "grpo_vps_fraction_correct_after": 0.95,
        }
    )
    assert baseline == pytest.approx(0.8)
    assert final == pytest.approx(0.95)


def test_extract_exp1220_accuracy_accepts_legacy_field_names() -> None:
    """REQ-LEARN-1247-5: replay also accepts baseline/final aliases."""
    baseline, final = extract_exp1220_accuracy(
        {"baseline_accuracy": 0.2, "final_accuracy": 0.3}
    )
    assert baseline == pytest.approx(0.2)
    assert final == pytest.approx(0.3)


def test_extract_exp1220_accuracy_requires_complete_pair() -> None:
    """REQ-LEARN-1247-5: replay source must contain both accuracy values."""
    with pytest.raises(KeyError, match="Exp 1220 accuracy fields"):
        extract_exp1220_accuracy({"grpo_vps_fraction_correct_before": 0.8})


def test_negative_delta_verdict_is_honest() -> None:
    """SCENARIO-LEARN-1248: negative improvement is accepted honestly."""
    artifact = build_grpo_v7_simplified_artifact(
        baseline_accuracy=0.6,
        final_accuracy=0.5,
        device_used="cpu",
        fallback_used=False,
    )
    assert artifact["improvement_pp"] == pytest.approx(-10.0)
    assert artifact["honest_verdict"] == "grpo_v7_negative_delta"


def test_missing_runtime_without_replay_maps_to_gpu_missing() -> None:
    """REQ-LEARN-1247-6: no run and no replay source maps to gpu_missing."""
    verdict = derive_grpo_v7_honest_verdict(
        improvement_pp=0.0,
        grpo_v7_ran=False,
    )
    assert verdict == "grpo_v7_gpu_missing"


def test_build_artifact_rejects_non_vps_mode() -> None:
    """REQ-LEARN-1247-2: training_mode must remain vps_only."""
    with pytest.raises(ValueError, match="training_mode"):
        build_grpo_v7_simplified_artifact(
            baseline_accuracy=0.5,
            final_accuracy=0.5,
            device_used="cpu",
            fallback_used=False,
            training_mode="fspo_vps",
        )


def test_build_artifact_rejects_unknown_device() -> None:
    """REQ-LEARN-1247-4: device_used must be one of the allowed tokens."""
    with pytest.raises(ValueError, match="device_used"):
        build_grpo_v7_simplified_artifact(
            baseline_accuracy=0.5,
            final_accuracy=0.5,
            device_used="dualgpu",
            fallback_used=False,
        )


def test_validate_artifact_rejects_missing_required_fields() -> None:
    """REQ-LEARN-1247-1: required fields are checked before write."""
    with pytest.raises(AssertionError, match="missing required fields"):
        validate_grpo_v7_artifact({"training_mode": "vps_only"})


def test_validate_artifact_rejects_wrong_training_mode() -> None:
    """REQ-LEARN-1247-2: validator enforces vps_only after construction."""
    artifact = build_grpo_v7_simplified_artifact(
        baseline_accuracy=0.5,
        final_accuracy=0.5,
        device_used="cpu",
        fallback_used=False,
    )
    artifact["training_mode"] = "fspo_vps"
    with pytest.raises(AssertionError, match="training_mode"):
        validate_grpo_v7_artifact(artifact)


def test_write_fallback_artifact_from_exp1220(tmp_path) -> None:
    """SCENARIO-LEARN-1247: fallback replay writes the stable v7 schema."""
    exp1220 = tmp_path / "experiment_1220.json"
    exp1220.write_text(
        json.dumps(
            {
                "grpo_vps_fraction_correct_before": 0.8,
                "grpo_vps_fraction_correct_after": 0.95,
            }
        )
    )
    artifact_path = tmp_path / "experiment_1247.json"

    artifact = write_grpo_v7_simplified_artifact(
        artifact_path=artifact_path,
        exp1220_path=exp1220,
        device_used="fallback",
        fallback_used=True,
    )

    written = json.loads(artifact_path.read_text())
    assert written == artifact
    assert written["baseline_accuracy"] == pytest.approx(0.8)
    assert written["final_accuracy"] == pytest.approx(0.95)
    assert written["improvement_pp"] == pytest.approx(15.0)
    assert written["device_used"] == "fallback"
    assert written["fallback_used"] is True
    assert written["honest_verdict"] == "grpo_v7_improvement_pp_15.0"
