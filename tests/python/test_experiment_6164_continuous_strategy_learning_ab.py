"""Tests for Exp6164 continuous strategy learning A/B.

Spec refs: REQ-LEARN-6164, REQ-LEARN-6164-1, REQ-LEARN-6164-2,
REQ-LEARN-6164-3, REQ-LEARN-6164-4, REQ-LEARN-6164-5, REQ-LEARN-6164-6,
REQ-LEARN-6164-7, REQ-LEARN-6164-8, REQ-LEARN-6164-9,
REQ-LEARN-6164-10, SCENARIO-LEARN-6164-BLOCKED,
SCENARIO-LEARN-6164-MATCHED, SCENARIO-LEARN-6164-TRANSACTION,
SCENARIO-LEARN-6164-READY.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6164_continuous_strategy_learning_ab as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/self-learning/spec.md"


def _passing_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _ready_exp6162() -> dict[str, Any]:
    return {
        "status": "complete_positive",
        "prospective_admission_replication_ready_score": 1.0,
        "honest_verdict": "complete_positive: fixture pass",
        "per_model_and_conjunctive_gate_matrix": {
            "conjunctive_pass": True,
            "by_model": {model["hf_id"]: {"model_pass": True} for model in mod.MODEL_SPECS},
        },
        "policy_manifest_path_hash_and_contents": {
            "contents_hash": mod.sha256_text("fixture-policy")
        },
    }


def _ready_exp6163() -> dict[str, Any]:
    return {
        "status": "complete_ready",
        "certified_strategy_store_scaleup_ready_score": 1.0,
        "honest_verdict": "complete_ready: fixture scaleup pass",
        "schema_abi_verdict_receipt": {
            "schema_version": "carnot.exp6163.strategy_store.v1",
            "schema_valid": True,
            "abi_valid": True,
            "verdict_passed": True,
            "bounded_state": True,
        },
        "strategy_store_scaleup_gate_matrix": {"all_gates_passed": True},
    }


def _fake_model_runner(
    model_specs: list[dict[str, Any]],
    arm_names: tuple[str, ...],
    event_count: int,
) -> dict[str, Any]:
    records = []
    for index, spec in enumerate(model_specs):
        records.append(
            {
                "name": spec["name"],
                "hf_id": spec["hf_id"],
                "role": spec["role"],
                "gpu": index,
                "resolved_path": f"/tmp/{mod.model_slug(spec['hf_id'])}.gguf",
                "revision": f"fixture-revision-{index}",
                "quantization": spec["quantization"],
                "sha256": mod.sha256_text(spec["hf_id"]),
                "loader": "llama_cpp.Llama",
                "native_chat": True,
                "actual_offload": "cuda",
            }
        )
    return {
        "resolved_records": records,
        "embedded_tokenizer_receipts": {
            "all_loaded": True,
            "chat_template_present": True,
            "cuda_runtime_seen": True,
            "worker_pids": [616400, 616401],
            "lifecycle": "released",
            "model_load_count": len(model_specs),
            "tokenizer_load_count": len(model_specs),
            "cuda_context_count": len(model_specs),
            "gpu_worker_count": len(model_specs),
        },
        "runtime_fingerprints_before": {
            spec["hf_id"]: mod.sha256_text(spec["hf_id"] + ":before") for spec in model_specs
        },
        "runtime_fingerprints_after": {
            spec["hf_id"]: mod.sha256_text(spec["hf_id"] + ":before") for spec in model_specs
        },
        "arm_invocation_counts": {
            arm: {spec["hf_id"]: event_count for spec in model_specs} for arm in arm_names
        },
        "cleanup": {
            "workers_released": True,
            "cuda_contexts_released": True,
            "orphan_task_owned_pid_count": 0,
        },
    }


def _run_qualified(tmp_path: Path, *, write: bool = False) -> dict[str, Any]:
    return mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        exp6162_artifact=_ready_exp6162(),
        exp6163_artifact=_ready_exp6163(),
        model_runner=_fake_model_runner,
        test_exit_codes=_passing_exit_codes(),
        duration_s=2.5,
        write=write,
    )


def test_req_6164_spec_declares_continuous_strategy_learning_contract() -> None:
    """REQ-LEARN-6164: OpenSpec owns the mandatory artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("## REQ-LEARN-6164") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-6164-1",
        "REQ-LEARN-6164-2",
        "REQ-LEARN-6164-3",
        "REQ-LEARN-6164-4",
        "REQ-LEARN-6164-5",
        "REQ-LEARN-6164-6",
        "REQ-LEARN-6164-7",
        "REQ-LEARN-6164-8",
        "REQ-LEARN-6164-9",
        "REQ-LEARN-6164-10",
        "SCENARIO-LEARN-6164-BLOCKED",
        "SCENARIO-LEARN-6164-MATCHED",
        "SCENARIO-LEARN-6164-TRANSACTION",
        "SCENARIO-LEARN-6164-READY",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.MODEL_SPECS[0]["hf_id"],
        mod.MODEL_SPECS[1]["hf_id"],
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_6164_missing_exp6163_blocks_before_model_load(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6164-BLOCKED: failed prerequisite writes zero-load artifact."""

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        exp6162_artifact=_ready_exp6162(),
        exp6163_path=tmp_path / "missing_exp6163.json",
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.0,
        write=True,
    )

    assert (tmp_path / mod.RESULT_RELATIVE_PATH.name).is_file()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert "self-learning did not execute" in artifact["honest_verdict"]
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["mandatory_artifact_written"] is True
    assert artifact["continuous_strategy_learning_ready_score"] == 0.0
    assert artifact["blocked_before_model_load_receipt"]["blocked"] is True
    assert artifact["blocked_before_model_load_receipt"]["all_invocation_counts_zero"] is True
    assert artifact["blocked_before_model_load_receipt"]["invocation_counts"] == (
        mod.ZERO_MODEL_INVOCATION_COUNTS
    )
    assert artifact["MODEL_SPECS"] == mod.MODEL_SPECS
    assert artifact["model_specs"] == mod.MODEL_SPECS
    assert all("Qwen3.5" not in spec["hf_id"] for spec in artifact["MODEL_SPECS"])
    assert all("gemma-4-E4B" not in spec["hf_id"] for spec in artifact["MODEL_SPECS"])
    assert mod.validate_artifact(artifact) is True


def test_req_6164_prerequisite_recompute_fails_closed_on_bad_6162(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6164-1/2: Exp6162 and Exp6163 are recomputed conjunctively."""

    bad_6162 = _ready_exp6162()
    bad_6162["prospective_admission_replication_ready_score"] = 0.0
    artifact = mod.run(
        result_path=tmp_path / "blocked_6162.json",
        exp6162_artifact=bad_6162,
        exp6163_artifact=_ready_exp6163(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.1,
    )

    receipt = artifact["prerequisite_gate_receipts"]
    assert receipt["all_passed"] is False
    assert receipt["exp6162"]["ready"] is False
    assert receipt["exp6163"]["ready"] is True
    assert "exp6162_not_ready" in receipt["blocked_reasons"]
    assert artifact["blocked_before_model_load_receipt"]["all_invocation_counts_zero"] is True
    assert artifact["preconditions_checked"]["exp6160_rows_hashed_count"] == 2
    assert artifact["preconditions_checked"]["model_ids"] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]
    assert mod.validate_artifact(artifact) is True


def test_scenario_6164_qualified_run_matches_four_arms_and_transactions(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6164-MATCHED/TRANSACTION: qualified run is isolated."""

    artifact = _run_qualified(tmp_path, write=True)

    assert artifact["status"] == "complete_positive"
    assert artifact["honest_verdict"].startswith("complete_positive:")
    assert artifact["continuous_strategy_learning_ready_score"] == 1.0
    assert artifact["blocked_before_model_load_receipt"]["blocked"] is False
    assert artifact["arm_definitions_and_resource_matching"]["arm_names"] == list(mod.ARM_NAMES)
    assert artifact["arm_definitions_and_resource_matching"]["all_arms_matched"] is True
    assert (
        artifact["chronological_event_order_and_decision_snapshot_receipts"][
            "current_label_visible_before_decision_count"
        ]
        == 0
    )
    assert (
        artifact["chronological_event_order_and_decision_snapshot_receipts"][
            "same_decision_write_count"
        ]
        == 0
    )
    assert (
        artifact["exact_post_outcome_commit_abort_quarantine_receipts"][
            "all_commits_after_exact_outcome"
        ]
        is True
    )
    assert artifact["exact_post_outcome_commit_abort_quarantine_receipts"]["quarantine_count"] > 0
    assert (
        artifact["duplicate_reordered_rollback_restart_eviction_and_state_bytes"]["idempotent"]
        is True
    )
    assert (
        artifact["duplicate_reordered_rollback_restart_eviction_and_state_bytes"][
            "bounded_state_ok"
        ]
        is True
    )
    assert artifact["model_weight_immutability_receipt"]["all_unchanged"] is True
    assert (
        artifact["acquisition_analysis_duration_and_cleanup_receipts"]["cleanup"][
            "workers_released"
        ]
        is True
    )
    assert mod.validate_artifact(artifact) is True
    assert json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text()) == artifact


def test_req_6164_ready_score_is_conjunctive_per_model_safety_and_weights(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6164-8/10: pooled success cannot mask model or safety failure."""

    artifact = _run_qualified(tmp_path)
    assert mod.ready_score(artifact) == 1.0

    bad_ci = deepcopy(artifact)
    model_id = mod.MODEL_SPECS[0]["hf_id"]
    bad_ci["per_model_family_partition_future_utility_accuracy_regret_and_grouped_intervals"][
        "by_model"
    ][model_id]["future_known"]["decision_calibrated_minus_no_memory_ci95"][0] = 0.0
    bad_ci["continuous_strategy_learning_ready_score"] = mod.ready_score(bad_ci)
    bad_ci["status"] = mod.status(bad_ci)
    bad_ci["honest_verdict"] = mod.honest_verdict(bad_ci)
    bad_ci["reproducibility_checksum"] = mod.reproducibility_checksum(bad_ci)
    assert bad_ci["continuous_strategy_learning_ready_score"] == 0.0
    assert bad_ci["status"] == "complete_null"
    assert bad_ci["honest_verdict"].startswith("complete_null:")
    assert "per_model_positive_lower_ci_not_met" in mod.missing_verifier_gaps(bad_ci)
    assert mod.validate_artifact(bad_ci) is True

    bad_safety = deepcopy(artifact)
    bad_safety["protected_retention_forgetting_safety_abstention_and_poison_metrics"][
        "safety_regression_count"
    ] = 1
    bad_safety["continuous_strategy_learning_ready_score"] = mod.ready_score(bad_safety)
    assert bad_safety["continuous_strategy_learning_ready_score"] == 0.0
    assert "safety_regression" in mod.missing_verifier_gaps(bad_safety)

    bad_weight = deepcopy(artifact)
    bad_weight["model_weight_immutability_receipt"]["all_unchanged"] = False
    bad_weight["continuous_strategy_learning_ready_score"] = mod.ready_score(bad_weight)
    assert bad_weight["continuous_strategy_learning_ready_score"] == 0.0
    assert "model_weight_immutability_failed" in mod.missing_verifier_gaps(bad_weight)

    retired = deepcopy(artifact)
    retired["repeated_null_retirement_receipt"] = {"retire": True}
    retired["retirement_triggered"] = mod.retirement_triggered(retired)
    retired["status"] = mod.status(retired)
    retired["honest_verdict"] = mod.honest_verdict(retired)
    retired["reproducibility_checksum"] = mod.reproducibility_checksum(retired)
    assert retired["status"] == "retired"
    assert retired["honest_verdict"].startswith("retired:")


def test_req_6164_validation_checksum_and_blocked_invocation_edges(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6164-2/3/10: schema validation rejects hidden bypasses."""

    artifact = mod.run(
        result_path=tmp_path / "blocked.json",
        exp6162_artifact=_ready_exp6162(),
        exp6163_path=tmp_path / "missing.json",
        test_exit_codes=_passing_exit_codes(),
    )
    assert artifact["duration_s"] >= 0.0

    missing = dict(artifact)
    missing.pop("mandatory_artifact_written")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(missing)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    bad_invocation = deepcopy(artifact)
    bad_invocation["blocked_before_model_load_receipt"]["invocation_counts"]["model_load_count"] = 1
    bad_invocation["reproducibility_checksum"] = mod.reproducibility_checksum(bad_invocation)
    with pytest.raises(ValueError, match="blocked_before_model_load_receipt"):
        mod.validate_artifact(bad_invocation)

    bad_model = deepcopy(artifact)
    bad_model["MODEL_SPECS"][0]["hf_id"] = "Qwen/Qwen3.5-0.8B"
    bad_model["reproducibility_checksum"] = mod.reproducibility_checksum(bad_model)
    with pytest.raises(ValueError, match="MODEL_SPECS"):
        mod.validate_artifact(bad_model)

    bad_continuous = deepcopy(artifact)
    bad_continuous["continuous_self_learning_task"] = False
    bad_continuous["reproducibility_checksum"] = mod.reproducibility_checksum(bad_continuous)
    with pytest.raises(ValueError, match="continuous_self_learning_task"):
        mod.validate_artifact(bad_continuous)

    bad_mandatory = deepcopy(artifact)
    bad_mandatory["mandatory_artifact_written"] = False
    bad_mandatory["reproducibility_checksum"] = mod.reproducibility_checksum(bad_mandatory)
    with pytest.raises(ValueError, match="mandatory_artifact_written"):
        mod.validate_artifact(bad_mandatory)

    bad_mirror = deepcopy(artifact)
    bad_mirror["model_specs"] = []
    bad_mirror["reproducibility_checksum"] = mod.reproducibility_checksum(bad_mirror)
    with pytest.raises(ValueError, match="model_specs"):
        mod.validate_artifact(bad_mirror)

    bad_zero_flag = deepcopy(artifact)
    bad_zero_flag["blocked_before_model_load_receipt"]["all_invocation_counts_zero"] = False
    bad_zero_flag["reproducibility_checksum"] = mod.reproducibility_checksum(bad_zero_flag)
    with pytest.raises(ValueError, match="blocked_before_model_load_receipt"):
        mod.validate_artifact(bad_zero_flag)

    bad_score = deepcopy(artifact)
    bad_score["continuous_strategy_learning_ready_score"] = 1.0
    bad_score["reproducibility_checksum"] = mod.reproducibility_checksum(bad_score)
    with pytest.raises(ValueError, match="continuous_strategy_learning_ready_score"):
        mod.validate_artifact(bad_score)

    bad_status = deepcopy(artifact)
    bad_status["status"] = "complete_positive"
    bad_status["reproducibility_checksum"] = mod.reproducibility_checksum(bad_status)
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "blocked: wrong"
    bad_verdict["reproducibility_checksum"] = mod.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)

    bad_provenance_type = deepcopy(artifact)
    bad_provenance_type["field_provenance"] = []
    bad_provenance_type["reproducibility_checksum"] = mod.reproducibility_checksum(
        bad_provenance_type
    )
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)

    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    bad_provenance["reproducibility_checksum"] = mod.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance)

    runtime_blocked = mod.run(
        result_path=tmp_path / "runtime_blocked.json",
        exp6162_artifact=_ready_exp6162(),
        exp6163_artifact=_ready_exp6163(),
        test_exit_codes=_passing_exit_codes(),
        duration_s=0.0,
    )
    assert runtime_blocked["status"] == "blocked"
    assert "model_runner_not_supplied" in runtime_blocked["honest_verdict"]

    blocked_after_gate = deepcopy(artifact)
    blocked_after_gate["prerequisite_gate_receipts"]["all_passed"] = True
    blocked_after_gate["prerequisite_gate_receipts"]["blocked_reasons"] = []
    blocked_after_gate["blocked_before_model_load_receipt"]["blocked_reasons"] = ["runtime_block"]
    blocked_after_gate["status"] = mod.status(blocked_after_gate)
    blocked_after_gate["honest_verdict"] = mod.honest_verdict(blocked_after_gate)
    assert blocked_after_gate["status"] == "blocked"
    assert "runtime_block" in blocked_after_gate["honest_verdict"]

    object_path = tmp_path / "object.json"
    object_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
    list_path = tmp_path / "list.json"
    list_path.write_text(json.dumps([]), encoding="utf-8")
    assert mod.load_json(tmp_path / "missing.json") == {}
    assert mod.load_json(object_path) == {"ok": True}
    with pytest.raises(ValueError, match="did not contain"):
        mod.load_json(list_path)
    assert mod.model_slug("unsloth/Qwen3.6-35B-A3B-GGUF") == "qwen3_6_35b_a3b"
