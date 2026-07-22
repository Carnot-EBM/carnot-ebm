"""Tests for Exp5811 Exp5799 event/provenance audit.

Spec refs: REQ-REPORT-5811, SCENARIO-REPORT-5811-ROW-REPLAY,
SCENARIO-REPORT-5811-GPU-RECEIPTS, SCENARIO-REPORT-5811-PRODUCER-REPAIR.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from carnot import experiment_5811_exp5799_event_provenance_audit as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"


def test_req_report_5811_spec_declares_audit_contract() -> None:
    """REQ-REPORT-5811: OpenSpec names the audit fields and principles."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5811") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-REPORT-5811",
        "SCENARIO-REPORT-5811-ROW-REPLAY",
        "SCENARIO-REPORT-5811-GPU-RECEIPTS",
        "SCENARIO-REPORT-5811-PRODUCER-REPAIR",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle=true`",
        "`canary_evidence_ready_score=1.0`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_5811_row_replay_reconstructs_events() -> None:
    """SCENARIO-REPORT-5811-ROW-REPLAY: raw predicates close over rows."""

    artifact = mod.build_artifact(
        root=REPO,
        duration_s=12.5,
        test_commands=["audit"],
        test_exit_codes={"audit": 0},
    )

    mod.validate_artifact(artifact)
    matrix = artifact["overlapping_event_matrix"]
    taxonomy = artifact["exclusive_primary_failure_taxonomy"]
    reconstruction = artifact["per_model_mode_reconstruction"]

    assert artifact["status"] == "complete"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["canary_evidence_ready_score"] == 1.0
    assert artifact["original_files_unchanged"] is True
    assert artifact["preconditions_checked"]["row_replay"]["row_count"] == 120
    assert artifact["preconditions_checked"]["row_replay"]["unique_cell_count"] == 120
    assert matrix["denominator"] == 120
    assert matrix["event_counts"] == {
        "parser_failure": 93,
        "truncation": 93,
        "empty_final_content": 91,
        "exact_wrong_answer": 6,
        "invalid_candidate": 2,
        "timeout": 0,
        "stop_collision": 0,
    }
    assert matrix["pairwise_overlap_counts"]["parser_failure&truncation"] == 93
    assert matrix["pairwise_overlap_counts"]["parser_failure&empty_final_content"] == 91
    assert matrix["pairwise_overlap_counts"]["truncation&empty_final_content"] == 91
    assert matrix["pairwise_overlap_counts"]["invalid_candidate&truncation"] == 2
    assert "legitimate co-occurrence" in matrix["tautology_flag_resolution"]
    assert taxonomy["denominator"] == 120
    assert taxonomy["primary_counts"] == {
        "empty_final_content": 91,
        "truncation": 2,
        "exact_wrong_answer": 6,
        "valid_exact_output": 21,
    }
    assert taxonomy["total_matches_denominator"] is True
    assert reconstruction["overall"]["exact_label_coverage"] == 1.0
    assert reconstruction["overall"]["independent_unit_count"] == 12
    assert reconstruction["overall"]["duration_from_rows_s"] > 0.0
    assert reconstruction["overall"]["row_file_sha256_matches_declared"] is True


def test_scenario_report_5811_model_mode_gpu_and_methodology_receipts() -> None:
    """SCENARIO-REPORT-5811-GPU-RECEIPTS: resume receipts stay unqualified."""

    artifact = mod.build_artifact(
        root=REPO,
        duration_s=12.5,
        test_commands=["audit"],
        test_exit_codes={"audit": 0},
    )

    per_mode = artifact["per_model_mode_reconstruction"]["models"]
    gpu = artifact["gpu_provenance_reconciliation"]
    gaps = artifact["methodology_gap_matrix"]

    qwen_embedded = per_mode[mod.QWEN_ID]["modes"][
        "qwen3-6-35b-a3b:embedded_template_final_sentinel_192"
    ]
    gemma31_selected = per_mode[mod.GEMMA31_ID]["modes"][
        "gemma-4-31b-it:reasoning_disabled_final_sentinel_128"
    ]
    assert qwen_embedded["event_counts"]["invalid_candidate"] == 2
    assert qwen_embedded["primary_failure_counts"] == {
        "empty_final_content": 22,
        "truncation": 2,
    }
    assert gemma31_selected["event_counts"]["exact_wrong_answer"] == 6
    assert gemma31_selected["primary_failure_counts"] == {
        "exact_wrong_answer": 6,
        "valid_exact_output": 18,
    }
    assert gpu["mode_receipts"][
        "unsloth/Qwen3.6-35B-A3B-GGUF::qwen3-6-35b-a3b:embedded_template_final_sentinel_192"
    ]["classification"] == "authenticated"
    assert gpu["mode_receipts"][
        "unsloth/gemma-4-31B-it-GGUF::gemma-4-31b-it:reasoning_disabled_final_sentinel_128"
    ]["classification"] == "resume_only"
    assert gpu["top_level_receipts"][mod.GEMMA31_ID]["classification"] == "resume_only"
    assert gpu["original_answer_channel_qualification_uses_resume_only"] is True
    assert gpu["audit_answer_channel_qualified_models"] == []
    assert gpu["unauthenticated_receipt_used_to_qualify_model"] is False
    assert gaps["original_duration_s"]["status"] == "absent"
    assert gaps["runtime_logs"]["status"] == "missing"
    assert gaps["original_test_exit_codes"]["status"] == "absent"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE


def test_scenario_report_5811_blocked_when_rows_do_not_replay() -> None:
    """SCENARIO-REPORT-5811-ROW-REPLAY: row drift fails closed."""

    rows = mod.read_jsonl(REPO / mod.EXP5799_ROWS_RELATIVE_PATH)
    tampered_rows = deepcopy(rows)
    tampered_rows[0]["raw_response_text"] = "tampered"

    artifact = mod.build_artifact(
        root=REPO,
        rows_override=tampered_rows,
        duration_s=12.5,
        test_commands=["audit"],
        test_exit_codes={"audit": 1},
    )

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["canary_evidence_ready_score"] == 0.0
    assert artifact["preconditions_checked"]["row_replay"]["ok"] is False
    assert "raw_response_sha256" in artifact["preconditions_checked"]["blocked_reasons"]


def test_scenario_report_5811_row_replay_failure_reasons_are_explicit(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-5811-ROW-REPLAY: each row/hash mismatch names its blocker."""

    source_artifact = json.loads((REPO / mod.EXP5799_ARTIFACT_RELATIVE_PATH).read_text())
    rows = mod.read_jsonl(REPO / mod.EXP5799_ROWS_RELATIVE_PATH)
    rows_path = REPO / mod.EXP5799_ROWS_RELATIVE_PATH

    for mutate_artifact, mutate_rows, expected in (
        (
            lambda artifact: artifact.update({"row_file_sha256": "sha256:" + "0" * 64}),
            lambda items: items,
            "row_file_sha256",
        ),
        (lambda artifact: artifact, lambda items: items + [deepcopy(items[0])], "duplicate canary cell"),
        (
            lambda artifact: artifact,
            lambda items: [dict(items[0], row_hash=mod.sha256_text("bad"))] + items[1:],
            "row_hash",
        ),
        (
            lambda artifact: artifact["raw_response_receipts"].pop(mod._row_key(rows[0])),
            lambda items: items,
            "missing raw_response_receipt",
        ),
        (
            lambda artifact: artifact["raw_response_receipts"][mod._row_key(rows[0])].update(
                {"prompt_hash": mod.sha256_text("bad")}
            ),
            lambda items: items,
            "prompt_hash",
        ),
        (
            lambda artifact: artifact["raw_response_receipts"].update(
                {"extra::cell": next(iter(artifact["raw_response_receipts"].values()))}
            ),
            lambda items: items,
            "row receipt set",
        ),
        (
            lambda artifact: artifact["mode_execution_matrix"][0].update({"row_count": 0}),
            lambda items: items,
            "declared mode row counts",
        ),
    ):
        artifact = deepcopy(source_artifact)
        changed_rows = mutate_rows(deepcopy(rows))
        mutate_artifact(artifact)
        try:
            mod._assert_row_replay(changed_rows, artifact, rows_path)
        except mod.AuditReplayError as exc:
            assert str(exc) == expected
        else:  # pragma: no cover - the assertion above must fire.
            raise AssertionError(expected)

    temp_rows = tmp_path / "rows.jsonl"
    temp_rows.write_text("", encoding="utf-8")
    artifact = deepcopy(source_artifact)
    artifact["row_file_sha256"] = mod.sha256_file(temp_rows)
    try:
        mod._assert_row_replay(rows, artifact, temp_rows)
    except mod.AuditReplayError as exc:
        assert str(exc) == "row_file_sha256"


def test_scenario_report_5811_gpu_classifier_covers_missing_and_inconsistent() -> None:
    """SCENARIO-REPORT-5811-GPU-RECEIPTS: receipt classes are not inferred loosely."""

    inconsistent_receipt = {
        "model_hf_id": mod.QWEN_ID,
        "mode_id": "bad-mode",
        "cuda_offload_authenticated": True,
        "n_gpu_layers_offloaded": 0,
        "gpu_memory_before_mb": 0,
        "gpu_memory_peak_mb": 0,
        "llama_cpp_build_info": {"cuda_backend": True},
    }
    synthetic = {
        "model_runtime_receipts": {
            mod.QWEN_ID: {"mode_runtime_receipts": {"bad-mode": inconsistent_receipt}}
        },
        "gpu_offload_receipts": {},
        "selected_transport_by_model": {
            mod.QWEN_ID: {"mode_id": "bad-mode"},
        },
    }

    assert mod.classify_runtime_receipt(None) == "missing"
    assert mod.classify_runtime_receipt(inconsistent_receipt) == "inconsistent"
    reconciled = mod._gpu_reconciliation(synthetic)

    assert reconciled["mode_receipts"][f"{mod.QWEN_ID}::bad-mode"]["classification"] == (
        "inconsistent"
    )
    assert reconciled["model_receipts"][mod.QWEN_ID]["classification"] == "inconsistent"
    assert reconciled["model_receipts"][mod.GEMMA31_ID]["classification"] == "missing"
    assert reconciled["audit_answer_channel_qualified_models"] == []

    authenticated_receipt = dict(
        inconsistent_receipt,
        cuda_offload_authenticated=True,
        n_gpu_layers_offloaded=4,
        gpu_memory_peak_mb=1024,
    )
    authenticated = deepcopy(synthetic)
    authenticated["model_runtime_receipts"][mod.QWEN_ID]["mode_runtime_receipts"][
        "bad-mode"
    ] = authenticated_receipt

    assert mod._gpu_reconciliation(authenticated)["audit_answer_channel_qualified_models"] == [
        mod.QWEN_ID
    ]


def test_scenario_report_5811_write_preserves_original_inputs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5811-PRODUCER-REPAIR: companion writes leave inputs sealed."""

    before = mod.immutable_input_hashes(REPO)
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name

    artifact = mod.build_and_write_artifact(
        root=REPO,
        result_path=result_path,
        duration_s=12.5,
        test_commands=["audit"],
        test_exit_codes={"audit": 0},
    )

    after = mod.immutable_input_hashes(REPO)
    written = json.loads(result_path.read_text(encoding="utf-8"))

    assert written == artifact
    assert before == after
    assert artifact["immutable_input_hashes"]["before"] == before
    assert artifact["immutable_input_hashes"]["after"] == after
    assert artifact["producer_repairs_and_tests"]["historical_files_mutated"] is False
