"""Tests for the Exp 1427 repair-executor rejection ledger.

Spec: REQ-VERIFY-1427, SCENARIO-VERIFY-1427
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.pipeline.repair_rejection_ledger import (
    REJECTION_CLASSES,
    build_experiment_1427_artifact,
    build_rejection_ledger,
    classify_repair_result,
    repair_v2_acceptance_contract,
)


def _repair_result(case_id: str, fallback_reason: str, validation_result: dict) -> dict:
    return {
        "accepted": False,
        "attempted": True,
        "case_id": case_id,
        "fallback_reason": fallback_reason,
        "local_model_used": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "runtime_s": 0.25,
        "validation_result": validation_result,
    }


def test_classify_repair_result_distinguishes_required_rejection_classes() -> None:
    """REQ-VERIFY-1427: every repair v2 rejection class has a distinct label."""

    missing = classify_repair_result(
        "exp1414",
        _repair_result(
            "missing",
            "schema_validation_failed",
            {"error": "invalid JSON repair output: Expecting value: line 1 column 1 (char 0)"},
        ),
    )
    malformed = classify_repair_result(
        "exp1414",
        _repair_result(
            "malformed",
            "schema_validation_failed",
            {"error": "invalid JSON repair output: Expecting ',' delimiter: line 7 column 1"},
        ),
    )
    prompt = classify_repair_result(
        "exp1414",
        _repair_result(
            "prompt",
            "schema_validation_failed",
            {"error": "unexpected repair output field(s): ['analysis']"},
        ),
    )
    validator = classify_repair_result(
        "exp1419",
        _repair_result(
            "validator",
            "validation_failed",
            {
                "constraint_passed": False,
                "fallback_reason": "no_validator_injected",
                "repair_required": True,
                "semantic_result": "REPAIR_HINT",
            },
        ),
    )
    semantic = classify_repair_result(
        "exp1419",
        _repair_result(
            "semantic",
            "validation_failed",
            {"constraint_passed": False, "repair_required": True, "semantic_result": "UNSAT"},
        ),
    )
    timeout = classify_repair_result(
        "exp1419",
        _repair_result("timeout", "timeout", {"error": "timed out after 30s"}),
    )

    assert missing.rejection_class == "missing_output"
    assert missing.rejection_reason == "missing_output_or_nonjson_response"
    assert "raw_model_output" in missing.missing_evidence
    assert malformed.rejection_class == "schema_failure"
    assert malformed.rejection_reason == "malformed_json_schema_failure"
    assert prompt.rejection_class == "prompt_noncompliance"
    assert validator.rejection_class == "validator_mismatch"
    assert semantic.rejection_class == "semantic_failure"
    assert timeout.rejection_class == "timeout"
    assert {entry.rejection_class for entry in [missing, malformed, prompt, validator, semantic, timeout]} == set(
        REJECTION_CLASSES
    )


def test_build_rejection_ledger_counts_observed_reasons_and_zero_classes() -> None:
    """SCENARIO-VERIFY-1427: rejected candidates are counted exactly once."""

    artifacts = {
        "exp1414": {
            "repair_results": [
                _repair_result(
                    "a",
                    "schema_validation_failed",
                    {
                        "error": (
                            "invalid JSON repair output: Expecting value: "
                            "line 1 column 1 (char 0)"
                        )
                    },
                ),
                _repair_result(
                    "b",
                    "validation_failed",
                    {
                        "constraint_passed": False,
                        "fallback_reason": "no_validator_injected",
                        "repair_required": True,
                        "semantic_result": "REPAIR_HINT",
                    },
                ),
            ]
        },
        "exp1419": {
            "repair_results": [
                {
                    "accepted": True,
                    "attempted": True,
                    "case_id": "accepted-control",
                    "validation_result": {"constraint_passed": True, "semantic_result": "SAT"},
                },
                _repair_result(
                    "c",
                    "schema_validation_failed",
                    {"error": "invalid JSON repair output: Expecting ',' delimiter"},
                ),
            ]
        },
    }

    ledger = build_rejection_ledger(artifacts)

    assert ledger["cases_analyzed"] == 3
    assert ledger["unique_cases_analyzed"] == 3
    assert ledger["accepted_candidates_seen"] == 1
    assert ledger["top_rejection_reason"] == "missing_output_or_nonjson_response"
    assert ledger["rejection_reason_counts"] == {
        "malformed_json_schema_failure": 1,
        "missing_output_or_nonjson_response": 1,
        "validator_mismatch_no_validator_injected": 1,
    }
    assert ledger["rejection_class_counts"]["prompt_noncompliance"] == 0
    assert ledger["rejection_class_counts"]["timeout"] == 0
    assert len(ledger["ledger_entries"]) == 3


def test_repair_v2_contract_requires_schema_first_and_nonzero_gate() -> None:
    """REQ-VERIFY-1427: repair v2 cannot repeat the scale run without a positive gate."""

    contract = repair_v2_acceptance_contract()

    assert contract["schema_validation_before_semantic_validation"] is True
    assert contract["record_rejection_reason_for_every_candidate"] is True
    assert contract["nonzero_validated_repair_success_gate_required"] is True
    assert "corrected_certificate" in contract["required_schema_fields"]
    assert set(REJECTION_CLASSES).issubset(contract["rejection_classes"])


def test_build_experiment_1427_artifact_reads_source_files(tmp_path: Path) -> None:
    """REQ-VERIFY-1427: terminal artifact fields are derived from source artifacts."""

    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_1414_certificate_llm_repair_executor_v1.json").write_text(
        json.dumps(
            {
                "repair_results": [
                    _repair_result(
                        "156",
                        "validation_failed",
                        {
                            "constraint_passed": False,
                            "fallback_reason": "no_validator_injected",
                            "repair_required": True,
                            "semantic_result": "REPAIR_HINT",
                        },
                    )
                ]
            }
        )
    )
    (results / "experiment_1419_fullscale_pipeline_v3_repair_executor.json").write_text(
        json.dumps(
            {
                "repair_results": [
                    _repair_result(
                        "160",
                        "schema_validation_failed",
                        {
                            "error": (
                                "invalid JSON repair output: Expecting value: "
                                "line 1 column 1 (char 0)"
                            )
                        },
                    ),
                    _repair_result(
                        "161",
                        "schema_validation_failed",
                        {
                            "error": (
                                "invalid JSON repair output: Expecting value: "
                                "line 1 column 1 (char 0)"
                            )
                        },
                    ),
                ]
            }
        )
    )

    artifact = build_experiment_1427_artifact(tmp_path, run_date="20260506")

    assert artifact["status"] == "complete"
    assert artifact["rejection_ledger_path"] == "docs/research/repair_executor_rejection_ledger_v1.md"
    assert artifact["rejection_ledger_complete"] is True
    assert artifact["cases_analyzed"] == 3
    assert artifact["top_rejection_reason"] == "missing_output_or_nonjson_response"
    assert artifact["repair_v2_contract_ready"] is True
    assert artifact["nonzero_repair_gate_required"] is True
    assert artifact["honest_verdict"].startswith("complete_rejection_ledger")
