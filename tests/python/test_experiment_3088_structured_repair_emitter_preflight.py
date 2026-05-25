"""Tests for Exp 3088 structured repair emitter preflight.

Spec refs: REQ-REPORT-3088, SCENARIO-REPORT-3088.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import structured_repair_emitter_preflight_3088 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "research-reporting" / "spec.md"
TEST_PATH = Path("tests/python/test_experiment_3088_structured_repair_emitter_preflight.py")


def test_req_report_3088_spec_and_schema_contract_are_anchored() -> None:
    """REQ-REPORT-3088: the preflight has an OpenSpec and schema contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    schema = mod.load_repair_candidate_schema(REPO_ROOT)

    assert "REQ-REPORT-3088" in spec
    assert "SCENARIO-REPORT-3088" in spec
    assert mod.ARTIFACT_FILENAME in spec
    assert mod.SCHEMA_REL_PATH.as_posix() in spec
    assert (REPO_ROOT / mod.SCHEMA_REL_PATH).is_file()
    assert schema["title"] == "CarnotStructuredRepairCandidateV1"
    assert schema["required"] == list(mod.REPAIR_CANDIDATE_REQUIRED_FIELDS)
    assert schema["properties"]["verifier_authority"]["enum"] == [
        "deterministic_tests",
        "exact_solver",
        "exact_verifier",
        "blocked_unavailable",
    ]


def test_req_report_3088_cached_payloads_parse_and_invalid_payloads_reject() -> None:
    """REQ-REPORT-3088: cached valid payloads parse and invalid payloads reject."""

    valid_payloads = mod.cached_valid_payloads()
    invalid_payloads = mod.cached_invalid_payloads()
    summary = mod.run_parser_validation(valid_payloads, invalid_payloads)

    assert summary["parser_validation_count"] == len(valid_payloads) >= 2
    assert summary["invalid_payload_rejection_count"] == len(invalid_payloads) >= 6
    assert summary["accepted_invalid_count"] == 0
    assert summary["rejected_valid_count"] == 0

    for payload in valid_payloads:
        result = mod.parse_repair_payload_text(json.dumps(payload, sort_keys=True))
        assert result.valid is True
        assert result.payload == payload
        assert result.errors == []

    by_name = {case.name: case for case in invalid_payloads}
    malformed = mod.parse_repair_payload_text(by_name["malformed_json"].raw_text)
    missing = mod.parse_repair_payload_text(by_name["missing_patch"].raw_text)
    extra = mod.parse_repair_payload_text(by_name["extra_field"].raw_text)
    wrong_enum = mod.parse_repair_payload_text(by_name["wrong_verifier_authority"].raw_text)

    assert malformed.failure_class == "json_decode_error"
    assert missing.failure_class == "missing_required_field"
    assert extra.failure_class == "extra_property"
    assert wrong_enum.failure_class == "enum_violation"
    assert set(summary["syntax_failure_classes_for_exp3089"]) >= {
        "json_decode_error",
        "missing_required_field",
        "wrong_type",
        "enum_violation",
        "extra_property",
        "invalid_task_intent_hash",
        "empty_patch",
        "unchecked_semantic_drift",
    }


def test_scenario_report_3088_artifact_declares_fallback_and_no_live_claim(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-3088: artifact readiness is cache-only and fallback-explicit."""

    artifact = mod.build_artifact(
        root=REPO_ROOT,
        tests_run=[TEST_PATH.as_posix()],
        import_checker=lambda _name: False,
        started_s=10.0,
        now_s=12.5,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["structured_generation_ready"] is True
    assert artifact["grammar_or_schema_path"] == mod.SCHEMA_REL_PATH.as_posix()
    assert artifact["parser_validation_count"] >= 2
    assert artifact["invalid_payload_rejection_count"] >= 6
    assert artifact["structured_library_available"] is False
    assert artifact["fallback_contract_used"] is True
    assert artifact["blocked_library_missing"] is False
    assert artifact["dependency_probe"]["xgrammar"]["available"] is False
    assert artifact["dependency_probe"]["llguidance"]["available"] is False
    assert artifact["dependency_probe"]["repo_native_fallback"]["available"] is True
    assert TEST_PATH.as_posix() in artifact["tests_added_or_reused"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")

    assert any(
        row["path"] == mod.EXP3074_REL_PATH.as_posix() and row["present"] is True
        for row in artifact["source_artifacts"]
    )
    assert any(
        row["path"] == mod.EXP3075_REL_PATH.as_posix() and row["required"] is False
        for row in artifact["source_artifacts"]
    )
    assert artifact["inference_substrate"] == {
        "mode": "cached_payload_schema_preflight",
        "cached_examples_only": True,
        "live_llm_inference": False,
        "local_gguf_inference": False,
        "model_load_attempted": False,
        "fresh_repair_generation": False,
        "fresh_verifier_scoring": False,
        "fresh_solver_execution": False,
        "repair_quality_claimed": False,
        "conductor_invoked": False,
    }

    output_path = tmp_path / mod.ARTIFACT_FILENAME
    written = mod.run_experiment(
        output_path=output_path,
        root=REPO_ROOT,
        tests_run=[TEST_PATH.as_posix()],
        import_checker=lambda _name: False,
        started_s=10.0,
        now_s=12.5,
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == written
    assert written["structured_generation_ready"] is True
    mod.validate_artifact(written)


def test_req_report_3088_artifact_validation_fails_closed() -> None:
    """REQ-REPORT-3088: artifact validation rejects drift and live-claim leakage."""

    artifact = mod.build_artifact(
        root=REPO_ROOT,
        tests_run=[TEST_PATH.as_posix()],
        import_checker=lambda _name: False,
        started_s=1.0,
        now_s=2.0,
    )

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="grammar_or_schema_path"):
        mod.validate_artifact(artifact | {"grammar_or_schema_path": "missing.json"})
    with pytest.raises(ValueError, match="parser_validation_count"):
        mod.validate_artifact(artifact | {"parser_validation_count": 0})
    with pytest.raises(ValueError, match="invalid_payload_rejection_count"):
        mod.validate_artifact(artifact | {"invalid_payload_rejection_count": 0})
    with pytest.raises(ValueError, match="live_llm_inference"):
        mod.validate_artifact(
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"live_llm_inference": True}
            }
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(artifact | {"honest_verdict": "ready"})


def test_req_report_3088_validation_edges_and_blocked_relative_write(tmp_path: Path) -> None:
    """REQ-REPORT-3088: malformed nested payloads and blocked fallback rows are explicit."""

    base = mod.cached_valid_payloads()[0]
    invalid_case = mod.cached_invalid_payloads()[0]

    assert invalid_case.to_dict()["expected_failure_class"] == "json_decode_error"
    assert mod.validate_repair_payload(["not", "object"]) == ["not_json_object:$ expected object"]
    assert mod.validate_repair_payload(base | {"task_id": 7}) == ["wrong_type:$.task_id"]
    assert mod.validate_repair_payload(base | {"task_id": ""}) == ["empty_string:$.task_id"]
    assert mod.validate_repair_payload(base | {"task_intent_hash": 7}) == [
        "wrong_type:$.task_intent_hash"
    ]
    assert mod.validate_repair_payload(base | {"patch": 7}) == ["wrong_type:$.patch"]
    assert mod.validate_repair_payload(base | {"behavioral_tests": []}) == [
        "wrong_type:$.behavioral_tests"
    ]
    assert mod.validate_repair_payload(base | {"behavioral_tests": ["bad"]}) == [
        "wrong_type:$.behavioral_tests[0]"
    ]
    assert mod.validate_repair_payload(
        base
        | {
            "behavioral_tests": [
                {"name": "unit", "command": ".venv/bin/pytest", "expected": "maybe"}
            ]
        }
    ) == ["enum_violation:$.behavioral_tests[0].expected"]
    assert mod.validate_repair_payload(base | {"semantic_drift_checks": "none"}) == [
        "wrong_type:$.semantic_drift_checks"
    ]
    assert mod.validate_repair_payload(base | {"semantic_drift_checks": []}) == [
        "unchecked_semantic_drift:$.semantic_drift_checks"
    ]
    assert mod.validate_repair_payload(base | {"semantic_drift_checks": ["bad"]}) == [
        "wrong_type:$.semantic_drift_checks[0]"
    ]
    assert mod.validate_repair_payload(
        base
        | {
            "semantic_drift_checks": [
                {"name": "intent", "authority": "self_graded", "must_pass": True}
            ]
        }
    ) == ["enum_violation:$.semantic_drift_checks[0].authority"]

    with pytest.raises(ValueError, match="repair_quality_claimed"):
        mod.validate_artifact(
            mod.build_artifact(root=REPO_ROOT, import_checker=lambda _name: False)
            | {
                "inference_substrate": mod._inference_substrate()
                | {"repair_quality_claimed": True}
            }
        )
    assert (
        mod._honest_verdict(
            ready=False,
            fallback_used=True,
            validation={"parser_validation_count": 0, "invalid_payload_rejection_count": 0},
        )
        == "blocked_preflight_contract_invalid"
    )

    blocked = mod.run_experiment(
        output_path=Path("results") / "blocked-exp3088.json",
        root=tmp_path,
        import_checker=lambda _name: False,
        started_s=1.0,
        now_s=1.0,
    )
    assert blocked["structured_generation_ready"] is False
    assert (tmp_path / "results" / "blocked-exp3088.json").is_file()
