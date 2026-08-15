"""Tests for Exp6456 corrupt-feedback held-restart CSL replication.

Spec refs: REQ-LEARN-6456, SCENARIO-LEARN-6456-SPEC,
SCENARIO-LEARN-6456-MODELS, SCENARIO-LEARN-6456-HELD-STREAM,
SCENARIO-LEARN-6456-RESTARTS, SCENARIO-LEARN-6456-PATH-CORRUPTION,
SCENARIO-LEARN-6456-QUARANTINE-ROLLBACK, SCENARIO-LEARN-6456-ROWS,
SCENARIO-LEARN-6456-ATTACKS, SCENARIO-LEARN-6456-READY.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6456_corrupt_feedback_held_restart_csl_replication as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + ".gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((model_id + "\nExp6456 fixture GGUF bytes\n").encode("utf-8"))
        paths[model_id] = path
    return paths


def _cached_pair(paths: dict[str, Path], calls: list[dict[str, Any]]):
    def resolve(
        *,
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        calls.append(
            {
                "gpu_indices": gpu_indices,
                "preferred_quant": preferred_quant,
                "model_indices": model_indices,
            }
        )
        ordered = (
            (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[2])
            if model_indices is None
            else (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[1])
        )
        return [
            {
                "name": mod.MODEL_TEMPLATE_BY_ID[model_id]["name"],
                "hf_id": model_id,
                "gpu": gpu,
                "model_path": str(paths[model_id]),
            }
            for gpu, model_id in zip(gpu_indices, ordered, strict=True)
        ]

    return resolve


def _tokenizer(path: str) -> tuple[bool, str]:
    return True, f"embedded tokenizer fixture for {Path(path).name}"


def _host_ok(
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[dict[str, Any]],
    upstream_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"resource": "verifier_bounded_csl_ready_score", "available": True, "detail": "1.0"},
        {"resource": "authenticated_upstream_state_and_receipts", "available": True, "detail": "fixture"},
        {"resource": "rtx_3090_gpu_count", "available": True, "detail": "2 fixture RTX 3090 GPUs"},
        {"resource": "mandatory_gguf_files", "available": True, "detail": str(len(model_specs))},
        {"resource": "embedded_gguf_tokenizers", "available": True, "detail": "fixture tokenizers"},
        {"resource": "exact_local_checkers", "available": True, "detail": "fixture checkers"},
        {
            "resource": "fresh_held_paths",
            "available": not result_path.exists() and not (data_dir / "raw_outputs").exists(),
            "detail": "fresh fixture paths",
        },
        {"resource": "disk_space", "available": True, "detail": "fixture disk"},
        {"resource": "wall_time_budget", "available": True, "detail": "fixture wall time"},
        {"resource": "sealed_held_stream", "available": True, "detail": "fixture stream"},
        {"resource": "sealed_corruption_schedule", "available": True, "detail": "fixture schedule"},
    ]


def _host_blocked(
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[dict[str, Any]],
    upstream_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = _host_ok(
        result_path=result_path,
        data_dir=data_dir,
        model_specs=model_specs,
        upstream_artifact=upstream_artifact,
    )
    rows[2] = {"resource": "rtx_3090_gpu_count", "available": False, "detail": "only one GPU"}
    return rows


def _restart_probe():
    calls: list[dict[str, Any]] = []

    def probe(
        *,
        state_path: Path,
        expected_head: str,
        model: str,
        arm: str,
        session_id: int,
    ) -> dict[str, Any]:
        calls.append(
            {
                "state_path": state_path,
                "expected_head": expected_head,
                "model": model,
                "arm": arm,
                "session_id": session_id,
            }
        )
        child_pid = 50000 + len(calls)
        return {
            "parent_pid": 40000,
            "child_pid": child_pid,
            "session_id": session_id,
            "model": model,
            "arm": arm,
            "child_start_time": f"2026-08-15T00:{len(calls):02d}:00Z",
            "parent_start_time": "2026-08-15T00:00:00Z",
            "exit_code": 0,
            "state_path": str(state_path),
            "expected_head": expected_head,
            "recovered_head": expected_head,
            "transaction_ancestry_valid": True,
            "head_hash_valid": True,
            "recovered_from_disk": True,
            "inherited_memory_state_visible": False,
        }

    return probe


def _artifact(tmp_path: Path, *, write: bool = True) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    return mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "exp6456-data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        precondition_func=_host_ok,
        restart_probe_func=_restart_probe(),
        duration_s=75.0,
        test_exit_codes={command: 0 for command in mod.DEFAULT_TEST_COMMANDS},
        write=write,
    )


def test_req_learn_6456_spec_declares_fields_and_scenarios() -> None:
    """REQ-LEARN-6456: OpenSpec owns the Exp6456 artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-LEARN-6456") : text.index("REQ-LEARN-6444")]
    normalized = " ".join(section.split())

    for marker in (
        "SCENARIO-LEARN-6456-SPEC",
        "SCENARIO-LEARN-6456-MODELS",
        "SCENARIO-LEARN-6456-HELD-STREAM",
        "SCENARIO-LEARN-6456-RESTARTS",
        "SCENARIO-LEARN-6456-PATH-CORRUPTION",
        "SCENARIO-LEARN-6456-QUARANTINE-ROLLBACK",
        "SCENARIO-LEARN-6456-ROWS",
        "SCENARIO-LEARN-6456-ATTACKS",
        "SCENARIO-LEARN-6456-READY",
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
    ):
        assert marker in section
    assert "transport-corrupted checker response SHALL NOT be authoritative" in normalized
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
    for condition in mod.READINESS_CONDITIONS:
        key = f"csl_safety_replication_ready_score:{condition}"
        assert key in mod.FIELD_PRINCIPLES
        assert " ".join(mod.FIELD_PRINCIPLES[key].split()) in normalized


def test_scenario_learn_6456_models_use_cached_sota_and_embedded_tokenizers(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-6456-MODELS: models resolve through cached GGUF helpers."""

    calls: list[dict[str, Any]] = []
    paths = _model_paths(tmp_path)
    resolved = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, calls),
        tokenizer_func=_tokenizer,
    )

    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": mod.PREFERRED_QUANT, "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": mod.PREFERRED_QUANT, "model_indices": (0, 2)},
    ]
    assert [row["hf_id"] for row in resolved["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert resolved["autotokenizer_usage_count"] == 0
    assert resolved["all_resolved"] is True
    assert all(row["tokenizer_source"] == mod.TOKENIZER_SOURCE for row in resolved["MODEL_SPECS"])
    assert all(
        row["tokenizer_loadable"]
        for row in resolved["model_and_embedded_tokenizer_hashes"]["rows"]
    )


def test_scenario_learn_6456_held_stream_disjoint_and_restarts(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6456-HELD-STREAM and RESTARTS: fresh rows reload from disk."""

    artifact = _artifact(tmp_path)
    manifest = artifact["sealed_held_stream_corruption_and_analysis_manifest"]
    freshness = artifact["path_nonexistence_freshness_and_disjointness_receipts"]
    restarts = artifact["process_restart_and_pid_receipts"]

    assert manifest["held_unit_count"] == len(mod.MANDATED_MODEL_IDS) * mod.UNITS_PER_MODEL
    assert manifest["session_count"] == mod.SESSION_COUNT
    assert manifest["units_per_model"] == mod.UNITS_PER_MODEL
    assert manifest["corruption_event_count"] == len(mod.MANDATED_MODEL_IDS) * mod.SESSION_COUNT
    assert manifest["protected_case_count"] == len(mod.MANDATED_MODEL_IDS) * mod.UNITS_PER_MODEL
    assert freshness["problem_overlap_with_exp6455_count"] == 0
    assert freshness["problem_overlap_with_exp6432_count"] == 0
    assert freshness["raw_hash_overlap_with_exp6455_count"] == 0
    assert freshness["raw_hash_overlap_with_exp6432_count"] == 0
    assert freshness["all_fresh_and_disjoint"] is True

    expected_restart_rows = len(mod.MANDATED_MODEL_IDS) * len(mod.ARMS) * mod.SESSION_COUNT
    assert restarts["session_restart_count"] == expected_restart_rows
    assert restarts["restart_recovery_rate"] == 1.0
    assert restarts["inherited_state_visible_count"] == 0
    assert all(row["child_pid"] != row["parent_pid"] for row in restarts["rows"])
    assert all(row["exit_code"] == 0 for row in restarts["rows"])


def test_scenario_learn_6456_corruption_quarantine_rollback(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6456-PATH-CORRUPTION: corrupt transport is not authoritative."""

    artifact = _artifact(tmp_path)
    rows = artifact["per_unit_rows"]["rows"]
    corruption = artifact["corruption_detection_and_path_receipts"]
    rollback = artifact["tombstone_rollback_and_resurrection_results"]
    quarantine = artifact["quarantine_precision_and_recall"]

    corrupt_rows = [row for row in rows if row["corrupt_event"]["scheduled"]]
    corrupt_count = len(mod.MANDATED_MODEL_IDS) * mod.SESSION_COUNT
    assert len(corrupt_rows) == corrupt_count
    assert corruption["scheduled_corrupt_event_count"] == corrupt_count
    assert corruption["detected_corrupt_event_count"] == corrupt_count
    assert corruption["non_authoritative_checker_response_count"] == corrupt_count
    assert quarantine["precision"] == 1.0
    assert quarantine["recall"] == 1.0
    assert rollback["tombstone_count"] == corrupt_count
    assert rollback["rollback_success_count"] == corrupt_count
    assert rollback["corrupt_update_resurrection_count"] == 0

    assert all(row["quarantine"]["quarantined"] for row in corrupt_rows)
    assert all(row["tombstone"]["written"] for row in corrupt_rows)
    assert all(row["rollback"]["restored_last_good_head"] for row in corrupt_rows)
    assert all(row["update"]["admitted"] is False for row in corrupt_rows)
    assert all(row["checker_response"]["authoritative"] is False for row in corrupt_rows)
    assert all(row["path_receipts"]["validation"]["accepted"] is False for row in corrupt_rows)
    assert all("stage_hash_mismatch:checker_transport" in row["path_receipts"]["validation"]["errors"] for row in corrupt_rows)
    assert artifact["protected_retention"]["protected_release_count"] == 0


def test_scenario_learn_6456_rows_recompute_and_ready(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6456-ROWS and READY: readiness recomputes from rows."""

    artifact = _artifact(tmp_path)
    rows = artifact["per_unit_rows"]["rows"]

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])
    assert artifact["per_unit_rows"]["row_count"] == (
        len(mod.MANDATED_MODEL_IDS) * mod.UNITS_PER_MODEL * len(mod.ARMS)
    )
    assert artifact["aggregate_row_recomputation"]["matches_reported"] is True
    assert artifact["checker_calls_tokens_and_timing"]["checker_call_count"] == len(rows)
    assert artifact["frozen_clean_and_governed_outcomes_by_model"]["all_models_have_eligible_rows"] is True
    assert artifact["future_exact_yield_delta"]["clean_minus_frozen"] > 0.0
    assert artifact["future_exact_yield_delta"]["governed_minus_frozen"] > 0.0
    assert artifact["future_exact_yield_delta"]["governed_within_tolerance"] is True
    assert artifact["negative_transfer_and_forgetting"]["negative_transfer_count"] == 0
    assert artifact["protected_retention"]["regression_count"] == 0
    assert artifact["false_accepts_and_abstentions"]["false_accept_count"] == 0
    assert artifact["transaction_ancestry_and_restart_recovery"]["all_transaction_ancestry_valid"] is True
    assert artifact["effects_and_uncertainty_over_distinct_held_units"]["distinct_held_unit_count"] == (
        len(mod.MANDATED_MODEL_IDS) * mod.UNITS_PER_MODEL
    )
    assert artifact["verifier_is_oracle"]["value"] is True
    assert "transport_corrupted_checker_response" in artifact["verifier_is_oracle"]["false_for"]
    assert artifact["csl_safety_replication_ready_score"] == 1.0
    assert artifact["status"] == "success_ready"
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) is True


def test_scenario_learn_6456_attacks_and_validation_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-6456-ATTACKS: unsafe corrupt-feedback mutations fail closed."""

    artifact = _artifact(tmp_path)
    attacks = artifact["attack_matrix"]

    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_critical_attacks_fail_closed"] is True
    assert attacks["readiness_promoted_attack_count"] == 0

    mutations = [
        ("required_fields", lambda data: data.pop("field_principles")),
        ("checksum", lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad")),
        ("aggregate_row_mismatch", lambda data: data["aggregate_row_recomputation"].__setitem__("matches_reported", False)),
        ("future_exact_yield_delta", lambda data: data["future_exact_yield_delta"].__setitem__("clean_minus_frozen", 0.0)),
        ("freshness", lambda data: data["path_nonexistence_freshness_and_disjointness_receipts"].__setitem__("all_fresh_and_disjoint", False)),
        ("missed_corruption", lambda data: data["corruption_detection_and_path_receipts"].__setitem__("detected_corrupt_event_count", 0)),
        ("quarantine_false_positive", lambda data: data["quarantine_precision_and_recall"].__setitem__("false_positive_count", 1)),
        ("corrupt_update_resurrection", lambda data: data["tombstone_rollback_and_resurrection_results"].__setitem__("corrupt_update_resurrection_count", 1)),
        ("restart_recovery", lambda data: data["process_restart_and_pid_receipts"].__setitem__("restart_recovery_rate", 0.5)),
        ("duration", lambda data: data.__setitem__("duration_s", 1.0)),
        ("attack_matrix", lambda data: data["attack_matrix"].__setitem__("all_critical_attacks_fail_closed", False)),
        ("verifier_is_oracle", lambda data: data["verifier_is_oracle"]["false_for"].__setitem__("transport_corrupted_checker_response", True)),
    ]
    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {"checksum", "required_fields"}:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_learn_6456_blocked_preconditions_write_terminal_artifact(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6456: blocked preconditions still write a terminal artifact."""

    paths = _model_paths(tmp_path / "models")
    artifact = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "blocked-data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        precondition_func=_host_blocked,
        restart_probe_func=_restart_probe(),
        duration_s=0.01,
        test_exit_codes={},
        write=True,
    )

    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "blocked_preconditions"
    assert artifact["blocked_reason"] == "rtx_3090_gpu_count"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["gate_check_summary"]["failed_check_count"] == 1
    assert artifact["per_unit_rows"]["row_count"] == 0
    assert artifact["csl_safety_replication_ready_score"] == 0.0
    assert mod.validate_artifact(artifact) is True

    unresolved = mod.run(
        date=mod.RUN_DATE,
        result_path=tmp_path / "unresolved.json",
        data_dir=tmp_path / "unresolved-data",
        cached_pair_func=lambda **_: [],
        tokenizer_func=_tokenizer,
        precondition_func=_host_ok,
        restart_probe_func=_restart_probe(),
        duration_s=0.01,
        test_exit_codes={},
        write=False,
    )
    assert unresolved["status"] == "blocked_preconditions"
    assert "model_not_resolved" in unresolved["blocked_reason"]
    assert mod.validate_artifact(unresolved) is True

    unexpected_date = mod.run(
        date="20260816",
        result_path=tmp_path / "unexpected-date.json",
        data_dir=tmp_path / "unexpected-date-data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        precondition_func=_host_ok,
        restart_probe_func=_restart_probe(),
        duration_s=0.01,
        test_exit_codes={},
        write=False,
    )
    assert unexpected_date["status"] == "blocked_preconditions"
    assert "unexpected_date:20260816" in unexpected_date["blocked_reason"]


def test_req_learn_6456_helper_edges_cover_safety_findings(tmp_path: Path) -> None:
    """REQ-LEARN-6456: helper edge paths remain deterministic and covered."""

    assert mod.sha256_file(tmp_path / "missing.gguf") is None
    assert mod.preconditions_pass([{"available": True}, {"available": True}]) is True
    assert mod.preconditions_pass([{"available": True}, {"available": False}]) is False
    assert mod._load_upstream(tmp_path)["present"] is False
    assert mod._ci95([1.0]) == [1.0, 1.0]

    artifact = _artifact(tmp_path / "write-false", write=False)
    assert artifact["device_and_runner_receipts"]["raw_pool_receipts"][0]["present"] is False
    assert mod.validate_artifact(artifact) is True

    bad_rows = deepcopy(artifact["per_unit_rows"]["rows"])
    bad_rows[1]["head_before"] = "sha256:bad"
    assert mod._transaction_ancestry_valid(bad_rows) is False

    blocked = deepcopy(artifact)
    blocked["status"] = "blocked_preconditions"
    assert mod._critical_findings(blocked) == []

    bad = deepcopy(artifact)
    bad["aggregate_row_recomputation"]["matches_reported"] = False
    bad["future_exact_yield_delta"]["clean_minus_frozen"] = 0.0
    bad["path_nonexistence_freshness_and_disjointness_receipts"]["all_fresh_and_disjoint"] = False
    bad["corruption_detection_and_path_receipts"]["detected_corrupt_event_count"] = 0
    bad["quarantine_precision_and_recall"]["false_positive_count"] = 1
    bad["tombstone_rollback_and_resurrection_results"]["corrupt_update_resurrection_count"] = 1
    bad["process_restart_and_pid_receipts"]["restart_recovery_rate"] = 0.0
    bad["duration_s"] = 1.0
    bad["attack_matrix"]["all_critical_attacks_fail_closed"] = False
    bad["verifier_is_oracle"]["false_for"]["transport_corrupted_checker_response"] = True
    bad["protected_retention"]["protected_release_count"] = 1
    bad["device_and_runner_receipts"]["cpu_fallback_count"] = 1
    assert {row["kind"] for row in mod._critical_findings(bad)} == {
        "aggregate_row_mismatch",
        "attack_matrix",
        "corrupt_update_resurrection",
        "cpu_fallback",
        "duration",
        "freshness",
        "future_exact_yield_delta",
        "missed_corruption",
        "protected_release",
        "quarantine_false_positive",
        "restart_recovery",
        "verifier_is_oracle",
    }
