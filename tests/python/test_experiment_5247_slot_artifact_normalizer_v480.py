"""Tests for Exp 5247 strict SLOT artifact normalizer.

Spec refs: REQ-REPORT-5247, SCENARIO-REPORT-5247-SAFE-REPAIR,
SCENARIO-REPORT-5247-UNSAFE-REJECTION,
SCENARIO-REPORT-5247-REPRESENTATIVE-479.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5247_slot_artifact_normalizer_v480 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
REPRESENTATIVE_PATHS = (
    REPO / "results/experiment_5235_adversarial_qa_null_tautology_calibration_v479.json",
    REPO / "results/experiment_5236_gap4_clean_status_after_qa_calibration_v479.json",
    REPO / "results/experiment_5241_arc_gated_live_patch_attempt_v479.json",
)


def _wrap(value: Any, principle: str = "fixture principle") -> dict[str, Any]:
    return {"value": value, "principle": principle}


def _receipt_wrap(field: str, value: Any) -> dict[str, Any]:
    return {"value": value, "principle": mod.FIELD_PRINCIPLES[field]}


def _repair_kinds(result: mod.NormalizationResult) -> set[str]:
    return {str(row["kind"]) for row in result.safe_repairs}


def _rejection_kinds(result: mod.NormalizationResult) -> set[str]:
    return {str(row["kind"]) for row in result.unsafe_rejections}


def test_req_report_5247_spec_declares_strict_normalizer_contract() -> None:
    """REQ-REPORT-5247: OpenSpec anchors the normalizer and receipt fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5247") : spec.index("### REQ-REPORT-5162")]

    for marker in (
        "REQ-REPORT-5247",
        "SCENARIO-REPORT-5247-SAFE-REPAIR",
        "SCENARIO-REPORT-5247-UNSAFE-REJECTION",
        "SCENARIO-REPORT-5247-REPRESENTATIVE-479",
        str(mod.RESULT_RELATIVE_PATH),
        "cached_fixture_replay_no_llm",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_report_5247_safe_repairs_are_copy_null_or_unwrap_only() -> None:
    """SCENARIO-REPORT-5247-SAFE-REPAIR: safe repairs are non-evidentiary."""

    payload = {
        "honest_verdict": _wrap("complete: fixture"),
        "inference_substrate": _wrap(mod.INFERENCE_SUBSTRATE),
        "field_principles": {
            "honest_verdict": "terminal verdict principle",
            "inference_substrate": "substrate principle",
            "model_specs": "nullable because no model was invoked",
            "artifact_normalizer_ready": "gate principle",
        },
        "gate_receipts": {
            "artifact_normalizer_ready": _wrap(True, "fixture gate already measured")
        },
    }

    result = mod.normalize_artifact(
        payload,
        nullable_fields=("model_specs",),
        gate_fields=("artifact_normalizer_ready",),
        required_principle_fields=(
            "honest_verdict",
            "inference_substrate",
            "model_specs",
            "artifact_normalizer_ready",
        ),
    )

    assert result.ready_for_gated_consumers is True
    assert result.normalized["honest_verdict"] == "complete: fixture"
    assert result.normalized["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert result.normalized["model_specs"] is None
    assert result.normalized["artifact_normalizer_ready"] is True
    assert _repair_kinds(result) == {
        "top_level_wrapper_unwrapped",
        "missing_explicit_null_added",
        "unambiguous_gate_boolean_extracted",
    }
    assert result.unsafe_rejections == []


def test_scenario_report_5247_unsafe_missing_evidence_is_rejected() -> None:
    """SCENARIO-REPORT-5247-UNSAFE-REJECTION: evidence is never synthesized."""

    payload = {
        "honest_verdict": "complete: live-model win fixture",
        "inference_substrate": "live_llm_inference",
        "duration_s": 1.0,
        "gate_a": {"acceptance_gate_passed": True},
        "gate_b": {"acceptance_gate_passed": False},
        "field_principles": {"honest_verdict": "terminal verdict principle"},
    }

    result = mod.normalize_artifact(
        payload,
        gate_fields=("acceptance_gate_passed",),
        required_principle_fields=("honest_verdict", "inference_substrate", "duration_s"),
    )

    assert result.ready_for_gated_consumers is False
    assert "acceptance_gate_passed" not in result.normalized
    assert {
        "conflicting_gate_boolean",
        "missing_principle",
        "duration_too_short",
        "missing_methodology_receipt",
    }.issubset(_rejection_kinds(result))
    assert "model_specs" not in result.normalized
    assert "random_seed" not in result.normalized
    assert "reproducibility_checksum" not in result.normalized


def test_req_report_5247_nonboolean_gate_and_missing_substrate_fail_closed() -> None:
    """REQ-REPORT-5247: ambiguous gate and substrate shapes are unsafe."""

    payload = {
        "honest_verdict": "complete: fixture",
        "field_principles": {
            "honest_verdict": "terminal verdict principle",
            "inference_substrate": "substrate principle",
            "artifact_normalizer_ready": "gate principle",
        },
        "nested": {"artifact_normalizer_ready": "true"},
    }

    result = mod.normalize_artifact(
        payload,
        gate_fields=("artifact_normalizer_ready",),
        required_principle_fields=("honest_verdict", "inference_substrate"),
    )

    assert result.ready_for_gated_consumers is False
    assert "artifact_normalizer_ready" not in result.normalized
    assert {"missing_inference_substrate", "nonboolean_gate_value"}.issubset(
        _rejection_kinds(result)
    )


def test_req_report_5247_alternate_principles_lists_and_duration_receipts() -> None:
    """REQ-REPORT-5247: strict checks cover alternate safe and unsafe shapes."""

    list_gate_payload = {
        "honest_verdict": "complete: fixture",
        "honest_verdict_principle": "terminal verdict sibling principle",
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "inference_substrate_principle": "substrate sibling principle",
        "field_principles": {"artifact_normalizer_ready": {"principle": "gate principle"}},
        "rows": [{"artifact_normalizer_ready": True}],
    }
    list_gate = mod.normalize_artifact(
        list_gate_payload,
        gate_fields=("artifact_normalizer_ready",),
        required_principle_fields=(
            "honest_verdict",
            "inference_substrate",
            "artifact_normalizer_ready",
        ),
    )
    assert list_gate.ready_for_gated_consumers is True
    assert list_gate.normalized["artifact_normalizer_ready"] is True

    top_level_nonbool = mod.normalize_artifact(
        {
            "honest_verdict": "complete: fixture",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "artifact_normalizer_ready": "yes",
            "field_principles": {
                "honest_verdict": "terminal verdict principle",
                "inference_substrate": "substrate principle",
            },
        },
        gate_fields=("artifact_normalizer_ready", "missing_gate"),
        required_principle_fields=("honest_verdict", "inference_substrate"),
    )
    assert {"nonboolean_gate_value", "missing_gate_boolean"}.issubset(
        _rejection_kinds(top_level_nonbool)
    )

    missing_duration = mod.normalize_artifact(
        {
            "honest_verdict": "complete: live fixture",
            "inference_substrate": "live_llm_inference",
            "model_specs": [{"hf_id": "fixture-GGUF"}],
            "random_seed": 1,
            "reproducibility_checksum": "sha256:" + "1" * 64,
            "field_principles": {
                "honest_verdict": "terminal verdict principle",
                "inference_substrate": "substrate principle",
            },
        },
        required_principle_fields=("honest_verdict", "inference_substrate"),
    )
    assert "missing_duration_receipt" in _rejection_kinds(missing_duration)


def test_scenario_report_5247_representative_artifacts_are_classified_without_mutation() -> None:
    """SCENARIO-REPORT-5247-REPRESENTATIVE-479: .479 artifacts are read-only inputs."""

    before_bytes = {path: path.read_bytes() for path in REPRESENTATIVE_PATHS}

    classifications = mod.classify_representative_artifacts(REPRESENTATIVE_PATHS)

    assert set(classifications) == {path.name for path in REPRESENTATIVE_PATHS}
    assert classifications[REPRESENTATIVE_PATHS[0].name]["source_flagged_adversarial"] is True
    assert classifications[REPRESENTATIVE_PATHS[1].name]["normalizer_ready_for_gates"] is False
    assert classifications[REPRESENTATIVE_PATHS[2].name]["normalizer_ready_for_gates"] is False
    assert "DURATION_TOO_SHORT" in classifications[REPRESENTATIVE_PATHS[2].name][
        "source_corrigendum_kinds"
    ]
    assert before_bytes == {path: path.read_bytes() for path in REPRESENTATIVE_PATHS}


def test_req_report_5247_classification_rejects_non_object_json(tmp_path: Path) -> None:
    """REQ-REPORT-5247: only JSON object artifacts are normalizable."""

    path = tmp_path / "list.json"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a JSON object"):
        mod.classify_artifact_path(path)


def test_req_report_5247_builds_and_validates_terminal_receipt() -> None:
    """REQ-REPORT-5247: receipt exposes required fields for Exp5248 gating."""

    classifications = {
        "fixture.json": {
            "normalizer_ready_for_gates": False,
            "normalization_rejections": ["source_flagged_adversarial"],
        }
    }
    artifact = mod.build_artifact(
        representative_classifications=classifications,
        tests_run=[{"command": "pytest fixture", "outcome": "PASS"}],
        duration_s=0.25,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "ready for gated consumers" in artifact["honest_verdict"]["value"]
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["artifact_normalizer_ready"] is True
    assert artifact["artifact_normalizer_ready_principle"] == mod.FIELD_PRINCIPLES[
        "artifact_normalizer_ready"
    ]
    assert artifact["duration_policy_preserved"]["value"] is True
    assert artifact["conductor_modified"]["value"] is False


def test_req_report_5247_validation_rejects_schema_breaks() -> None:
    """REQ-REPORT-5247: malformed receipts fail closed."""

    artifact = mod.build_artifact(
        representative_classifications={},
        tests_run=[{"command": "pytest fixture", "outcome": "PASS"}],
        duration_s=0.25,
    )

    with pytest.raises(ValueError, match="artifact_normalizer_ready"):
        mod.validate_artifact(artifact | {"artifact_normalizer_ready": {"value": True}})
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(
            artifact | {"inference_substrate": _receipt_wrap("inference_substrate", "aggregation_from_upstream_artifacts")}
        )
    with pytest.raises(ValueError, match="conductor_modified"):
        mod.validate_artifact(artifact | {"conductor_modified": _receipt_wrap("conductor_modified", True)})
    with pytest.raises(ValueError, match="safe_repairs_supported"):
        mod.validate_artifact(artifact | {"safe_repairs_supported": []})
    with pytest.raises(ValueError, match="principle mismatch"):
        mod.validate_artifact(artifact | {"safe_repairs_supported": _wrap(["bad"])})
    with pytest.raises(ValueError, match="tests_run"):
        mod.validate_artifact(artifact | {"tests_run": []})
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(artifact | {"reproducibility_checksum": "sha256:bad"})


def test_req_report_5247_validation_rejects_additional_receipt_breaks() -> None:
    """REQ-REPORT-5247: validation guards every required receipt field."""

    artifact = mod.build_artifact(
        representative_classifications={},
        tests_run=[{"command": "pytest fixture", "outcome": "PASS"}],
        duration_s=0.25,
    )

    missing_schema = dict(artifact)
    missing_schema.pop("schema")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_schema)
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(artifact | {"field_principles": {}})
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(artifact | {"honest_verdict": _receipt_wrap("honest_verdict", "not terminal")})
    with pytest.raises(ValueError, match="gated-consumer readiness"):
        mod.validate_artifact(artifact | {"honest_verdict": _receipt_wrap("honest_verdict", "complete: okay")})
    with pytest.raises(ValueError, match="artifact_normalizer_ready_principle"):
        mod.validate_artifact(artifact | {"artifact_normalizer_ready_principle": "wrong"})
    with pytest.raises(ValueError, match="duration_policy_preserved"):
        mod.validate_artifact(
            artifact | {"duration_policy_preserved": _receipt_wrap("duration_policy_preserved", False)}
        )
    with pytest.raises(ValueError, match="safe_repairs_supported"):
        mod.validate_artifact(
            artifact | {"safe_repairs_supported": _receipt_wrap("safe_repairs_supported", [])}
        )
    with pytest.raises(ValueError, match="tests_run rows"):
        mod.validate_artifact(artifact | {"tests_run": [{"command": "pytest fixture"}]})


def test_req_report_5247_write_artifact_outputs_valid_json(tmp_path: Path) -> None:
    """REQ-REPORT-5247: write_artifact emits the requested JSON receipt."""

    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: fixture",
                "inference_substrate": mod.INFERENCE_SUBSTRATE,
                "field_principles": {
                    "honest_verdict": "terminal verdict principle",
                    "inference_substrate": "substrate principle",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.write_artifact(
        output_path=output,
        representative_paths=(source,),
        tests_run=[{"command": "pytest fixture", "outcome": "PASS"}],
        duration_s=0.25,
    )

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == artifact
    mod.validate_artifact(written)


def test_req_report_5247_repository_artifact_matches_schema() -> None:
    """REQ-REPORT-5247: checked-in Exp5247 result artifact remains valid."""

    if not RESULT_PATH.exists():
        pytest.skip("Exp5247 artifact not written yet")
    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    mod.validate_artifact(artifact)
