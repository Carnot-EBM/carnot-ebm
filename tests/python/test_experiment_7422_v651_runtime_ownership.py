"""Tests for the no-model runtime capacity and ownership audit.

Spec refs: REQ-VERIFY-7422 and SCENARIO-VERIFY-7422-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from carnot import experiment_7422_v651_runtime_ownership as mod
from carnot import gpu_lease_phase_journal as lease_api
from carnot.reporting.experiment_7303_validation_scope import REQUIRED_CHECK_NAMES


def _gpu(uuid: str, *, pid: int | None = None, known: bool = True) -> dict[str, object]:
    """Build one private device observation without querying host hardware."""

    return {
        "gpu_index": int(uuid[-1]),
        "gpu_uuid": uuid,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "gpu_utilization_pct": 0,
        "gpu_memory_total_mb": 24576,
        "gpu_memory_used_mb": 4 if pid is None else 12000,
        "gpu_memory_free_mb": 24572 if pid is None else 12576,
        "pid": pid,
        "gpu_process_memory_mb": 0 if pid is None else 11996,
        "proc_exists": known if pid is not None else None,
        "start_time_ticks": 1234 if pid is not None and known else None,
        "ownership_classification": ("idle" if pid is None else "conflicting"),
        "matching_lease_id": None,
        "ownership_evidence_errors": ([] if pid is None else ["current_canonical_lease_missing"]),
    }


def _receipts() -> list[dict[str, object]]:
    """Build one passing receipt for each frozen affected and terminal check."""

    return [
        {
            "name": name,
            "command_argv": [name],
            "command_environment": {},
            "scope": "fixture",
            "exit_code": 0,
            "duration_s": 0.01,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": "sha256:" + "c" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in (*REQUIRED_CHECK_NAMES, *mod.TERMINAL_CHECK_NAMES)
    ]


def _lease_row() -> dict[str, object]:
    """Represent a completed no-model lease lifecycle for artifact tests."""

    return {
        "row_kind": "lease_lifecycle",
        "device_uuid": "GPU-0",
        "lease_id": "lease:fixture",
        "task_id": mod.TASK_ID,
        "owner_pid": os.getpid(),
        "owner_pid_start_ticks": lease_api.proc_start_ticks(os.getpid()),
        "acquired_monotonic_ns": 100,
        "child_readback_monotonic_ns": 110,
        "released_monotonic_ns": 120,
        "fresh_child_readback_passed": True,
        "release_passed": True,
        "release_duration_s": 0.01,
        "terminal_phase": "terminal_blocked",
        "signals_sent": [],
    }


def _artifact() -> dict[str, object]:
    """Build one deterministic terminal artifact through production reducers."""

    artifact = mod.build_artifact_for_test(
        capacity_rows=mod.build_private_capacity_rows(),
        lease_rows=[_lease_row()],
        receipts=_receipts(),
    )
    artifact["reproducibility_checksum"] = mod.artifact_checksum(artifact)
    return artifact


def test_req_verify_7422_spec_and_no_model_contract_exist() -> None:
    """REQ-VERIFY-7422 fixes the contract before runtime implementation."""

    text = mod.SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-7422" in text
    assert "SCENARIO-VERIFY-7422-CAPACITY" in text
    assert "SCENARIO-VERIFY-7422-OWNERSHIP" in text
    assert mod.MODEL_SPECS == []
    assert mod.INFERENCE_SUBSTRATE_CLASS == "no_model_load"
    assert mod.EXECUTION_VENUE == "host"
    assert all(value == 0 for value in mod.zero_counts().values())


def test_scenario_verify_7422_capacity_reproduces_exact_two_slot_failure() -> None:
    """SCENARIO-VERIFY-7422-CAPACITY retains the old failure and repaired result."""

    rows = {row["fixture_id"]: row for row in mod.build_private_capacity_rows()}
    assert rows["zero_free_devices"]["available_capacity"] == 0
    assert rows["zero_free_devices"]["capacity_predicate_passed"] is False
    assert rows["one_free_device"]["available_capacity"] == 1
    assert rows["one_free_device"]["legacy_dictionary_equality_passed"] is True
    assert rows["two_free_devices"]["available_capacity"] == 2
    assert rows["two_free_devices"]["legacy_expected"] == {
        "query_ok": True,
        "minimum_rtx3090_slots": 1,
    }
    assert rows["two_free_devices"]["legacy_observed"] == {
        "query_ok": True,
        "minimum_rtx3090_slots": 2,
    }
    assert rows["two_free_devices"]["legacy_dictionary_equality_passed"] is False
    assert rows["two_free_devices"]["capacity_predicate_passed"] is True
    assert rows["two_free_devices"]["passed"] is True
    assert rows["failed_inventory"]["query_ok"] is False
    assert rows["failed_inventory"]["passed"] is False


def test_scenario_verify_7422_fail_closed_busy_unknown_and_stale_rows() -> None:
    """SCENARIO-VERIFY-7422-FAIL-CLOSED keeps busy and unknown devices unavailable."""

    busy = mod.reduce_capacity_row(
        "one_busy_device",
        [_gpu("GPU-0", pid=50), _gpu("GPU-1")],
        [],
        [{"returncode": 0}, {"returncode": 0}],
    )
    assert busy["available_gpu_uuids"] == ["GPU-1"]
    assert busy["available_capacity"] == 1
    assert busy["observed_owner_state"][0]["ownership_classification"] == "conflicting"

    unknown = mod.reduce_capacity_row(
        "unknown_process_state",
        [_gpu("GPU-0", pid=50, known=False)],
        [],
        [{"returncode": 0}, {"returncode": 0}],
    )
    assert unknown["available_gpu_uuids"] == []
    assert unknown["capacity_predicate_passed"] is False

    stale = {
        "device_uuid": "GPU-0",
        "readable": True,
        "canonical": True,
        "released": False,
        "fresh": True,
        "owner_live": False,
        "owner_pid": 50,
        "owner_start_ticks": 1,
        "lease_id": "lease:stale",
    }
    recoverable = mod.reduce_capacity_row(
        "stale_owner_identity",
        [_gpu("GPU-0")],
        [stale],
        [{"returncode": 0}, {"returncode": 0}],
    )
    assert recoverable["available_gpu_uuids"] == ["GPU-0"]
    assert recoverable["observed_owner_state"][0]["recovery_eligible"] is True

    live = {**stale, "owner_live": True, "lease_id": "lease:live"}
    blocked = mod.reduce_capacity_row(
        "live_owner",
        [_gpu("GPU-0")],
        [live],
        [{"returncode": 0}, {"returncode": 0}],
    )
    assert blocked["available_gpu_uuids"] == []
    assert blocked["observed_owner_state"][0]["recovery_eligible"] is False


def test_scenario_verify_7422_ownership_race_stale_recovery_and_bounded_release(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7422-OWNERSHIP uses the shipped owner-bound lease protocol."""

    lease = lease_api.GpuLease.acquire(
        runtime_dir=tmp_path,
        task_id=mod.TASK_ID,
        device_uuid="GPU-race",
        expected_model="path-prerequisite-no-model-load",
        vram_before_mb=4,
        ttl_s=5.0,
    )
    with pytest.raises(lease_api.LeaseBusy):
        lease_api.GpuLease.acquire(
            runtime_dir=tmp_path,
            task_id="contender",
            device_uuid="GPU-race",
            expected_model="path-prerequisite-no-model-load",
            vram_before_mb=4,
            ttl_s=5.0,
        )
    receipt = mod.complete_no_model_lease(lease)
    assert receipt["release_passed"] is True
    assert receipt["release_duration_s"] <= mod.RELEASE_TIMEOUT_S
    assert receipt["signals_sent"] == []

    stale = lease_api.GpuLease.acquire(
        runtime_dir=tmp_path,
        task_id="stale-owner",
        device_uuid="GPU-stale",
        expected_model="path-prerequisite-no-model-load",
        vram_before_mb=4,
        ttl_s=5.0,
    )
    stale_document = deepcopy(stale.document)
    stale.close()
    stale_document["owner"]["pid_start_ticks"] += 1
    stale_document["checksum"] = lease_api.journal_checksum(stale_document)
    lease_api.write_json_atomic(stale.journal_path, stale_document)
    recovered = lease_api.GpuLease.acquire(
        runtime_dir=tmp_path,
        task_id=mod.TASK_ID,
        device_uuid="GPU-stale",
        expected_model="path-prerequisite-no-model-load",
        vram_before_mb=4,
        ttl_s=5.0,
    )
    assert recovered.owner_receipt()["recovery"]["performed"] is True
    assert recovered.owner_receipt()["recovery"]["signals_sent"] == []
    mod.complete_no_model_lease(recovered)


def test_scenario_verify_7422_no_model_current_receipt_is_zero() -> None:
    """SCENARIO-VERIFY-7422-NO-MODEL never turns resource proof into inference."""

    receipt = mod.build_no_model_current_receipt(100, 200, [])
    assert receipt["MODEL_SPECS"] == []
    assert receipt["model_invoked"] is False
    assert receipt["inference_substrate_class"] == "no_model_load"
    assert receipt["execution_venue"] == "host"
    assert receipt["small_ebm_training"] == {"performed": False}
    assert receipt["invocation_counts"] == mod.zero_counts()


def test_scenario_verify_7422_validation_manifest_is_exact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7422-TERMINAL freezes only affected files and eight checks."""

    commands = mod.build_validation_plan(mod.REPO_ROOT, tmp_path / "private")
    assert [row.name for row in commands] == list(REQUIRED_CHECK_NAMES)
    assert mod.validate_validation_plan(mod.REPO_ROOT, commands) == []
    assert "full_python_suite" not in {row.name for row in commands}
    focused = next(row for row in commands if row.name == "focused_pytest")
    assert set(mod.VALIDATION_MANIFEST.test_paths) <= set(focused.argv)
    coverage = next(row for row in commands if row.name == "changed_module_coverage_report")
    assert dict(coverage.command_environment)["COVERAGE_FILE"].endswith("/.coverage")


def test_scenario_verify_7422_terminal_artifact_reduces_independently() -> None:
    """SCENARIO-VERIFY-7422-TERMINAL rejects capacity, ownership, and checksum drift."""

    artifact = _artifact()
    assert mod.independent_reduce_artifact(artifact) == []
    assert mod.validate_artifact(artifact, require_terminal=True) == []
    assert artifact["runtime_ownership_ready_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_")

    changed = deepcopy(artifact)
    changed["capacity_rows"][2]["available_capacity"] = 0
    assert "capacity_rows_mismatch" in mod.independent_reduce_artifact(changed)

    changed = deepcopy(artifact)
    changed["lease_rows"][0]["fresh_child_readback_passed"] = False
    assert "lease_lifecycle_not_ready" in mod.independent_reduce_artifact(changed)

    changed = deepcopy(artifact)
    changed["model_invoked"] = True
    changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
    assert "model_invoked_invalid" in mod.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum_mismatch" in mod.validate_artifact(changed)


def test_req_verify_7422_blocked_artifact_names_exact_capacity_state() -> None:
    """REQ-VERIFY-7422 reports genuine contention without partial or promotion claims."""

    checks = [
        mod.gate_row(
            "minimum_available_rtx3090_capacity",
            "nvidia-smi_and_gpu_lease_journal",
            "available_rtx3090_slots",
            ">=",
            1,
            0,
        )
    ]
    blocked = mod.build_blocked_artifact(
        checks,
        capacity_rows=mod.build_private_capacity_rows(),
        owner_state=[{"device_uuid": "GPU-0", "owner_pid": 99, "state": "busy"}],
    )
    assert blocked["status"] == "blocked_precondition"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_minimum_available_rtx3090_capacity"
    assert blocked["runtime_ownership_ready_score"] == 0
    assert blocked["gate_check_summary"]["expected_value"] == 1
    assert blocked["gate_check_summary"]["observed_value"] == 0


def test_req_verify_7422_source_preconditions_and_cli_date() -> None:
    """REQ-VERIFY-7422 authenticates source bytes and fixes the execution date."""

    checks, context = mod.collect_preconditions(mod.REPO_ROOT)
    assert checks
    assert all(row["passed"] for row in checks)
    assert context["model_path_prerequisite"]["weights_opened"] is False
    assert mod._date_argument("20260919") == "20260919"
    with pytest.raises(ValueError, match="date must be 20260919"):
        mod._date_argument("20260918")


def test_req_verify_7422_helpers_reject_malformed_values(tmp_path: Path) -> None:
    """REQ-VERIFY-7422 keeps malformed inputs explicit in cold readers."""

    assert mod.load_object(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("[", encoding="utf-8")
    assert mod.load_object(bad) == {}
    assert mod.validate_artifact([]) == ["artifact_not_object"]
    assert mod.independent_reduce_artifact({}) == [
        "raw_row_manifest_invalid",
        "capacity_rows_invalid",
    ]
    assert mod.independent_reduce_artifact({"raw_row_manifest": [], "capacity_rows": []}) == [
        "lease_rows_invalid"
    ]
    assert mod.compare("in", "null", ["null", "blocked"]) is True
    with pytest.raises(ValueError, match="unsupported_operator"):
        mod.compare("!=", 1, 2)
    failed = mod.gate_row("bad", "upstream", "field", "!=", 1, 2)
    assert failed["passed"] is False
    assert failed["operator"] == "!="

    output = tmp_path / "atomic.json"
    mod.atomic_json(output, {"complete": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"complete": True}


def test_req_verify_7422_independent_reducer_rejects_each_boundary() -> None:
    """REQ-VERIFY-7422 independently rejects malformed raw and current evidence."""

    artifact = _artifact()
    malformed = deepcopy(artifact)
    malformed["capacity_rows"][0].pop("fixture_id")
    errors = mod.independent_reduce_artifact(malformed)
    assert any(error.startswith("capacity_row_invalid:0:KeyError") for error in errors)
    assert "capacity_rows_mismatch" in errors

    reordered = deepcopy(artifact)
    reordered["capacity_rows"][0], reordered["capacity_rows"][1] = (
        reordered["capacity_rows"][1],
        reordered["capacity_rows"][0],
    )
    assert "private_capacity_fixtures_mismatch" in mod.independent_reduce_artifact(reordered)

    no_events = deepcopy(artifact)
    no_events["current_invocation_events"] = "bad"
    assert "current_invocation_events_invalid" in mod.independent_reduce_artifact(no_events)

    bad_counts = deepcopy(artifact)
    bad_counts["invocation_counts"]["model_loads_attempted"] = 1
    assert "invocation_counts_mismatch" in mod.independent_reduce_artifact(bad_counts)

    assert mod._lease_lifecycle_passed([]) is False


def test_req_verify_7422_validator_rejects_closed_contract_mutations() -> None:
    """REQ-VERIFY-7422 cold validation covers identity, class, and receipt gates."""

    artifact = _artifact()
    cases = {
        "missing_required_field:capacity_rows": {"capacity_rows": None},
        "identity_invalid": {"schema": "wrong"},
        "run_identity_invalid": {"milestone": "wrong"},
        "model_specs_invalid": {"MODEL_SPECS": ["wrong"]},
        "invocation_counts_invalid": {
            "invocation_counts": {**mod.zero_counts(), "model_loads_attempted": 1}
        },
        "inference_substrate_class_invalid": {"inference_substrate_class": "model_load"},
        "execution_venue_invalid": {"execution_venue": "board"},
        "random_seed_invalid": {"random_seed": 1},
        "promotion_score_invalid": {"promotion_score": 1},
        "small_ebm_training_invalid": {"small_ebm_training": {"performed": True}},
        "verdict_class_invalid": {"verdict_class": "ready"},
        "honest_verdict_prefix_invalid": {"honest_verdict": "ready"},
    }
    for expected, update in cases.items():
        changed = deepcopy(artifact)
        if expected.startswith("missing_required_field"):
            changed.pop("capacity_rows")
        else:
            changed.update(update)
        changed["reproducibility_checksum"] = mod.artifact_checksum(changed)
        assert expected in mod.validate_artifact(changed)

    blocked = mod.build_blocked_artifact([mod.gate_row("capacity", "host", "count", ">=", 1, 0)])
    blocked["honest_verdict"] = "wrong"
    blocked["reproducibility_checksum"] = mod.artifact_checksum(blocked)
    assert "blocked_verdict_prefix_invalid" in mod.validate_artifact(blocked)

    missing_receipts = deepcopy(artifact)
    missing_receipts["validation_receipts"] = []
    missing_receipts["runtime_ownership_ready_score"] = 0
    missing_receipts["reproducibility_checksum"] = mod.artifact_checksum(missing_receipts)
    assert "required_validation_receipts_invalid" in mod.validate_artifact(
        missing_receipts, require_terminal=True
    )


def test_req_verify_7422_raw_hash_and_terminal_command_helpers(tmp_path: Path) -> None:
    """REQ-VERIFY-7422 binds raw rows and declares all fresh-process readers."""

    raw = mod._raw_manifest(
        tmp_path,
        tmp_path / "raw",
        {"capacity_rows": [{"fixture": True}], "lease_rows": [{"lease": True}]},
    )
    assert [row["field"] for row in raw] == ["capacity_rows", "lease_rows"]
    assert all(row["sha256"].startswith("sha256:") for row in raw)

    artifact = _artifact()
    artifact["raw_row_manifest"] = mod._raw_manifest(
        tmp_path,
        tmp_path / "artifact-raw",
        {
            "capacity_rows": artifact["capacity_rows"],
            "lease_rows": artifact["lease_rows"],
        },
    )
    assert mod.independent_reduce_artifact(artifact, root=tmp_path) == []
    changed_raw = tmp_path / artifact["raw_row_manifest"][0]["path"]
    changed_raw.write_text("{}\n", encoding="utf-8")
    assert "raw_row_hash_mismatch:0" in mod.independent_reduce_artifact(artifact, root=tmp_path)

    malformed = deepcopy(artifact)
    malformed["raw_row_manifest"] = [None]
    assert "raw_row_reference_invalid:0" in mod.independent_reduce_artifact(
        malformed, root=tmp_path
    )
    missing = deepcopy(artifact)
    missing["raw_row_manifest"][0]["path"] = "missing.json"
    assert "raw_row_path_missing:0" in mod.independent_reduce_artifact(missing, root=tmp_path)
    bad_field = deepcopy(artifact)
    bad_field["raw_row_manifest"][1]["field"] = "other"
    assert "raw_row_field_invalid:1" in mod.independent_reduce_artifact(bad_field, root=tmp_path)
    payload_drift = deepcopy(artifact)
    payload_path = tmp_path / payload_drift["raw_row_manifest"][1]["path"]
    mod.atomic_json(payload_path, {"lease_rows": []})
    payload_drift["raw_row_manifest"][1]["sha256"] = mod.sha256_file(payload_path)
    assert "raw_row_payload_mismatch:1" in mod.independent_reduce_artifact(
        payload_drift, root=tmp_path
    )

    source = mod._source_hashes(
        mod.REPO_ROOT,
        {
            "source_hashes": {"input": "sha256:test"},
            "model_path_prerequisite": {"weights_opened": False},
        },
        raw,
        {"path": "sidecar.json", "sha256": "sha256:test", "scope": "historical_model_receipts"},
    )
    assert source["input"] == "sha256:test"
    assert source[mod.MODULE_PATH.as_posix()].startswith("sha256:")
    assert source["model_file_metadata"]["weights_opened"] is False

    commands = mod._terminal_commands(mod.REPO_ROOT, tmp_path / "candidate.json")
    assert [row.name for row in commands] == list(mod.TERMINAL_CHECK_NAMES)
    assert commands[0].argv[2] == mod.WRAPPER_PATH.as_posix()
    assert "--strict" in commands[-1].argv
