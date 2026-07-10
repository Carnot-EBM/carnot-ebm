"""Tests for Exp5525 SOTA schema failure taxonomy.

Spec refs: REQ-VERIFY-5525, SCENARIO-VERIFY-5525.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5512_structured_output_positive_control as positive
from carnot import experiment_5525_sota_schema_failure_taxonomy as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5525_sota_schema_failure_taxonomy.py")


def _schema_invalid_placeholder_row() -> dict:
    row = positive.classify_candidate_payload(
        {"candidate_id": "...", "claimed_exact_validator_verdict": "..."}
    )
    row["model_hf_id"] = positive.MANDATED_HEADLINE_MODEL_IDS[0]
    return row


def _live_artifact(path: Path) -> Path:
    row = _schema_invalid_placeholder_row()
    diagnostics = [
        {
            "resource": "llama_cpp_gpu_offload",
            "available": True,
            "detail": "injected nvidia-smi memory delta 21228.000 MB",
            "model_path": str(path.parent / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
            "n_gpu_layers": -1,
        }
    ]
    artifact = {
        "model_specs": [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": positive.MANDATED_HEADLINE_MODEL_IDS[0],
                "headline_eligible": True,
                "local_model_present": True,
                "model_path": diagnostics[0]["model_path"],
                "preferred_quant": "Q4_K_M",
            }
        ],
        "headline_models_used": [positive.MANDATED_HEADLINE_MODEL_IDS[0]],
        "runtime_status": {
            "llama_cpp_cuda_available": True,
            "gpu_offload_verified": True,
            "gpu_memory_delta_mb": 21228.0,
            "offload_diagnostics": diagnostics,
        },
        "offload_diagnostics": diagnostics,
        "model_runs": [
            {
                "model_hf_id": positive.MANDATED_HEADLINE_MODEL_IDS[0],
                "model_file": diagnostics[0]["model_path"],
                "quant": "Q4_K_M",
                "llama_cpp_binding": "llama_cpp.Llama.create_completion",
                "llama_cpp_command": None,
                "n_gpu_layers": -1,
                "gpu_memory_delta_mb": 21048.0,
                "wall_time_s": 28.18,
                "prompt_tokens": 1985,
                "completion_tokens": mod.DEFAULT_LIVE_MAX_TOKENS,
                "raw_output_preview": (
                    "The user wants me to follow the final answer shape "
                    "{\"candidate_rows\": [...], \"proof_claims\": "
                    "[{\"candidate_id\": \"...\", "
                    "\"claimed_exact_validator_verdict\": \"...\"}]}"
                ),
                "parse_failures": [],
                "candidate_rows": [row],
                "missing_instance_ids": [
                    "claim_infeasible_negative_control",
                    "claim_safety_conflict",
                    "claim_support_preference",
                ],
                "runtime_error": None,
            }
        ],
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def test_req_verify_5525_spec_declares_taxonomy_contract() -> None:
    """REQ-VERIFY-5525: OpenSpec anchors categories, fields, and local GGUF rules."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5525") : spec.index("### REQ-VERIFY-5501")
    ]

    assert "SCENARIO-VERIFY-5525" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert "AutoTokenizer.from_pretrained" in section
    for category in mod.FAILURE_CATEGORIES:
        assert f"`{category}`" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_5525_fixture_rows_reuse_live_parser_path() -> None:
    """SCENARIO-VERIFY-5525: deterministic fixtures still parse before live claims."""

    rows = mod.build_fixture_diagnostic_rows()

    assert len(rows) == 3
    assert all(row["row_source"] == "fixture" for row in rows)
    assert all(row["first_failure"] is None for row in rows)
    assert all(row["schema_valid"] is True for row in rows)
    assert all(row["exact_validator_correct"] is True for row in rows)
    assert {row["parser_backend"] for row in rows} == {mod.PARSER_BACKEND}
    assert {row["prompt_prefix_hash"] for row in rows} == {mod.prompt_prefix_hash()}


def test_req_verify_5525_builds_live_failure_taxonomy_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-5525: live malformed and missing rows are classified visibly."""

    live_path = _live_artifact(tmp_path / "experiment_5513.json")
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        live_artifact_path=live_path,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["model_specs"][0]["hf_id"] == positive.MANDATED_HEADLINE_MODEL_IDS[0]
    assert artifact["smoke_models_used"] == [positive.MANDATED_HEADLINE_MODEL_IDS[0]]
    assert artifact["fixture_rows_checked"] == 3
    assert artifact["live_rows_checked"] == 4
    assert artifact["failure_taxonomy_counts"]["prompt_contract_miss"] == 1
    assert artifact["failure_taxonomy_counts"]["semantic_candidate_absent"] == 3
    assert artifact["grammar_runtime_available"] is True
    assert artifact["grammar_mask_applied"] is False
    assert artifact["truncation_detected"] is True
    assert artifact["json_extraction_success_rate"] == pytest.approx(1.0)
    assert artifact["schema_validity_rate"] == pytest.approx(0.0)
    assert artifact["exact_validator_handoff_ready"] is False
    assert artifact["gpu_offload_evidence"]["gpu_offload_verified"] is True
    assert artifact["sota_schema_failure_taxonomy_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["prompt_prefix_hashes"] == [mod.prompt_prefix_hash()]

    live_rows = artifact["diagnostic_rows"]["live"]
    emitted = next(row for row in live_rows if row["row_source"] == "live_emitted")
    assert emitted["first_failure"] == "prompt_contract_miss"
    assert emitted["max_tokens"] == mod.DEFAULT_LIVE_MAX_TOKENS
    assert emitted["truncation_marker"] == "completion_tokens_reached_max_tokens"
    assert emitted["output_byte_length"] > 0
    assert emitted["exact_validator_target"] is None
    missing = [row for row in live_rows if row["row_source"] == "live_missing"]
    assert {row["first_failure"] for row in missing} == {"semantic_candidate_absent"}

    mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("row", "context", "expected"),
    [
        ({"runtime_error": "boom"}, {}, "runtime_unavailable"),
        ({}, {"grammar_runtime_available": False}, "grammar_runtime_unavailable"),
        (
            {"schema_valid": False, "schema_errors": ["$.x is not allowed"]},
            {"grammar_runtime_available": True, "grammar_mask_applied": False},
            "grammar_mask_not_applied",
        ),
        (
            {"parse_status": "no_json_payload"},
            {"grammar_runtime_available": True, "grammar_mask_applied": True, "truncated": True},
            "max_tokens_truncation",
        ),
        (
            {"parse_status": "no_json_payload"},
            {"grammar_runtime_available": True, "grammar_mask_applied": True},
            "json_extraction_failure",
        ),
        (
            {"schema_valid": False, "schema_errors": ["$.premises is required"]},
            {"grammar_runtime_available": True, "grammar_mask_applied": True},
            "required_field_missing",
        ),
        (
            {"schema_valid": False, "schema_errors": ["$.extra is not allowed"]},
            {"grammar_runtime_available": True, "grammar_mask_applied": True},
            "json_schema_invalid",
        ),
        (
            {
                "parseable": True,
                "schema_valid": True,
                "exact_validator_correct": False,
                "exact_validator_verdict": "soft_suboptimal",
            },
            {"grammar_runtime_available": True, "grammar_mask_applied": True},
            "exact_validator_mismatch",
        ),
        (
            {"parse_status": "missing_candidate_row"},
            {"grammar_runtime_available": True, "grammar_mask_applied": True},
            "semantic_candidate_absent",
        ),
    ],
)
def test_req_verify_5525_first_failure_classifier_edges(
    row: dict,
    context: dict,
    expected: str,
) -> None:
    """REQ-VERIFY-5525: every taxonomy bucket has an explicit classifier path."""

    assert mod.classify_first_failure(row, context) == expected


def test_req_verify_5525_validation_and_no_live_artifact_branch(tmp_path: Path) -> None:
    """REQ-VERIFY-5525: missing live evidence is blocked and validation fails closed."""

    artifact = mod.run(
        result_path=tmp_path / "missing_live.json",
        live_artifact_path=tmp_path / "absent_5513.json",
    )

    assert artifact["live_rows_checked"] == 1
    assert artifact["failure_taxonomy_counts"]["runtime_unavailable"] == 1
    assert artifact["sota_schema_failure_taxonomy_ready"] is False
    assert artifact["honest_verdict"].startswith("blocked:")

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    bad_substrate["reproducibility_checksum"] = mod.payload_checksum(bad_substrate)
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_count = deepcopy(artifact)
    bad_count["failure_taxonomy_counts"].pop("runtime_unavailable")
    bad_count["reproducibility_checksum"] = mod.payload_checksum(bad_count)
    with pytest.raises(ValueError, match="failure_taxonomy_counts"):
        mod.validate_artifact(bad_count)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)
