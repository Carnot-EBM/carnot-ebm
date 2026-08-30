"""Tests for the owned-lease memory branch admission contract.

Spec refs: REQ-CL-6773, SCENARIO-CL-6773-STREAM,
SCENARIO-CL-6773-MODEL-CONTRACT, SCENARIO-CL-6773-BLOCKED,
REQ-INFRA-6773, SCENARIO-INFRA-6773-LEASE,
SCENARIO-INFRA-6773-SEQUENTIAL, SCENARIO-INFRA-6773-NO-PREEMPTION,
REQ-REPORT-6773, SCENARIO-REPORT-6773-LIVE, and
SCENARIO-REPORT-6773-BLOCKED.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from carnot import experiment_6773_csl_owned_lease_contract as exp
from carnot import gpu_lease_phase_journal as lease_api


def _sha(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _model_records(tmp_path: Path) -> list[dict]:
    rows = []
    for index, planned in enumerate(exp.PLANNED_MODELS):
        path = tmp_path / planned["filename"]
        path.write_bytes(f"model-{index}".encode())
        rows.append(
            {
                "model_id": planned["model_id"],
                "role": planned["role"],
                "family": planned["family"],
                "quantization": "Q4_K_M",
                "revision": f"revision-{index}",
                "filename": planned["filename"],
                "model_path": str(path),
                "model_sha256": planned["expected_sha256"],
                "model_size_bytes": path.stat().st_size,
                "tokenizer": {
                    "source": "llama.cpp_embedded_gguf",
                    "loadable": True,
                    "detail": "fixture tokenizer",
                },
            }
        )
    return rows


def _stream_fixture() -> dict:
    orders = [
        {
            "order_id": f"order_{index}",
            "order_hash": f"sha256:{index:064x}",
            "event_ids": ["a01", "r01"],
        }
        for index in range(1, 7)
    ]
    return {
        "procedural_memory_stream_ready": True,
        "order_count": 6,
        "future_evidence_violations": 0,
        "stream_manifest": {
            "stream_hash": exp.EXPECTED_STREAM_HASH,
            "frozen_before_dry_replay": True,
            "orders": orders,
        },
        "capacity_contract": {
            "arms": {
                "detailed_trajectory": {
                    "storage_bytes": 32768,
                    "context_tokens": 256,
                    "top_k": 3,
                },
                "procedural_lesson": {
                    "storage_bytes": 32768,
                    "context_tokens": 256,
                    "top_k": 3,
                },
            },
            "storage_ceiling_bytes": 32768,
            "max_committed_bytes_by_arm": {
                "detailed_trajectory": 12288,
                "procedural_lesson": 12288,
            },
        },
        "read_only_episode_enforced": True,
        "transaction_schema": {
            "version": 1,
            "required_fields": list(exp.stream_mod.TRANSACTION_REQUIRED_FIELDS),
            "active_episode_policy": "read_only",
            "commit_timing": "after_exact_result_closes_episode",
        },
        "restart_receipts": [{"bytes_match": True, "hash_match": True}],
        "rollback_receipts": [{"inverse_patch_applied": True, "byte_identical": True}],
        "poison_fixture_receipts": [
            {
                "committed": False,
                "admission_reason": "reject_poison_exact_authority",
                "intended_admission_reason": "reject_poison_exact_authority",
                "state_hash": "sha256:parent",
                "parent_hash": "sha256:parent",
            }
        ],
        "representation_pair_receipts": [
            {
                "event_id": "a01",
                "representations": {
                    "procedural_lesson": {
                        "payload": {
                            "abstract_constraint": "Reject an index equal to the length.",
                            "applicability_scope": "python_collection",
                            "repair_procedure": "Use an exclusive upper bound.",
                        }
                    }
                },
            }
        ],
    }


def _devices() -> list[dict]:
    return [
        {
            "index": 0,
            "uuid": exp.EXPECTED_GPU_UUIDS[0],
            "name": "NVIDIA GeForce RTX 3090",
            "memory_total_mb": 24576,
            "memory_used_mb": 900,
            "memory_free_mb": 23100,
            "temperature_c": 60,
            "utilization_pct": 1,
            "active_compute_processes": [{"pid": 81, "used_memory_mb": 100}],
        },
        {
            "index": 1,
            "uuid": exp.EXPECTED_GPU_UUIDS[1],
            "name": "NVIDIA GeForce RTX 3090",
            "memory_total_mb": 24576,
            "memory_used_mb": 500,
            "memory_free_mb": 23500,
            "temperature_c": 65,
            "utilization_pct": 0,
            "active_compute_processes": [],
        },
    ]


def _preconditions(models: list[dict], fixture: dict, *, passed: bool = True) -> dict:
    stream_checks = exp.stream_contract_checks(
        fixture,
        source_artifact_sha256=exp.EXPECTED_SOURCE_ARTIFACT_SHA256,
        upstream_validator_errors=[],
    )
    checks = [
        {
            "check": "planning_date_matches",
            "expected": True,
            "observed": passed,
            "passed": passed,
        },
        {
            "check": "models_resolved",
            "expected": True,
            "observed": True,
            "passed": True,
        },
    ]
    return {
        "all_passed": passed,
        "checks": checks,
        "models": deepcopy(models),
        "stream_fixture": deepcopy(fixture),
        "stream_contract_checks": stream_checks,
        "source_artifact_sha256": exp.EXPECTED_SOURCE_ARTIFACT_SHA256,
        "device_inventory_before": _devices(),
        "device_selection_receipt": exp.rank_eligible_devices(_devices()),
        "ports": [46773, 46774],
        "llama_cpp": {
            "exists": True,
            "cuda_linked": True,
            "python_cuda_offload": True,
        },
        "resources": {"ram_available_bytes": 80 * 1024**3, "disk_free_bytes": 10**9},
    }


def _phase_history() -> list[dict]:
    return [
        {
            "phase": phase,
            "previous_phase": None if ordinal == 0 else exp.COMPLETE_PHASE_SEQUENCE[ordinal - 1],
            "monotonic_ns": 1000 + ordinal,
            "event_checksum": f"sha256:{ordinal:064x}",
        }
        for ordinal, phase in enumerate(exp.COMPLETE_PHASE_SEQUENCE)
    ]


def _gpu_receipt(model: dict, index: int, *, error: str | None = None) -> dict:
    owner_pid = 7000 + index
    row = {
        "model_record": deepcopy(model),
        "model_id": model["model_id"],
        "device": deepcopy(_devices()[1]),
        "unrelated_process_inventory": [{"pid": 81, "used_memory_mb": 100}],
        "lease_owner": {
            "task_id": f"exp6773-{index}",
            "device_uuid": exp.EXPECTED_GPU_UUIDS[1],
            "pid": owner_pid,
            "pid_start_ticks": 300 + index,
            "expected_model": model["model_path"],
            "signals_sent": [],
        },
        "phase_history": _phase_history(),
        "lease_release": {
            "released": True,
            "phase": "terminal_complete",
            "device_uuid": exp.EXPECTED_GPU_UUIDS[1],
            "pid": owner_pid,
            "pid_start_ticks": 300 + index,
            "signals_sent": [],
        },
        "gpu_layers": {"requested": -1, "offloaded": 65, "total": 65},
        "offload_full": True,
        "resident_owned_vram_mb": 18000 + index * 4000,
        "peak_owned_vram_mb": 18100 + index * 4000,
        "first_token_canary": {
            "fixture_event_id": "a01",
            "prompt_sha256": f"sha256:{71 + index:064x}",
            "first_token_observed": True,
            "completion_tokens": 1,
            "first_token_sha256": f"sha256:{81 + index:064x}",
            "bounded": True,
        },
        "worker_process": {
            "pid": owner_pid,
            "pid_start_ticks": 300 + index,
            "exit_code": 0,
            "absent_after_exit": True,
        },
        "vram_recovery": {
            "before_used_mb": 500,
            "after_used_mb": 520,
            "absolute_delta_mb": 20,
            "tolerance_mb": exp.VRAM_RECOVERY_TOLERANCE_MB,
            "owned_pid_present": False,
            "passed": True,
        },
        "duration_s": 12.5 + index,
        "unrelated_processes_signaled": [],
        "errors": [] if error is None else [error],
    }
    if error is not None:
        row["phase_history"][-1]["phase"] = "terminal_blocked"
        row["lease_release"]["phase"] = "terminal_blocked"
    row["receipt_sha256"] = exp.gpu_receipt_checksum(row)
    return row


def _ready_artifact(tmp_path: Path) -> dict:
    models = _model_records(tmp_path)
    fixture = _stream_fixture()
    preconditions = _preconditions(models, fixture)
    receipts = [_gpu_receipt(model, index) for index, model in enumerate(models)]
    return exp.build_artifact(
        date=exp.RUN_DATE,
        preconditions=preconditions,
        gpu_receipts=receipts,
        code_receipts={"module": f"sha256:{91:064x}"},
        started_ns=1_000_000_000,
        finished_ns=4_000_000_000,
    )


def test_req_cl_6773_exact_typed_model_schema_and_roles() -> None:
    """REQ-CL-6773 fixes both model identities and one record schema."""
    assert [row["model_id"] for row in exp.PLANNED_MODELS] == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    ]
    assert [row["role"] for row in exp.PLANNED_MODELS] == [
        "flagship_moe_acquisition_and_within_family",
        "flagship_dense_held_family_transfer",
    ]
    assert exp.MODEL_RECORD_FIELDS == {
        "model_id",
        "role",
        "family",
        "quantization",
        "revision",
        "filename",
        "model_path",
        "model_sha256",
        "model_size_bytes",
        "tokenizer",
    }
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)


def test_scenario_cl_6773_model_contract_rejects_missing_extra_and_mismatch(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-6773-MODEL-CONTRACT rejects identity drift."""
    models = _model_records(tmp_path)
    assert exp.model_record_errors(models[0], exp.PLANNED_MODELS[0]) == []

    missing = deepcopy(models[0])
    missing.pop("revision")
    assert "field_set" in exp.model_record_errors(missing, exp.PLANNED_MODELS[0])

    extra = deepcopy(models[0])
    extra["renamed_model"] = extra["model_id"]
    assert "field_set" in exp.model_record_errors(extra, exp.PLANNED_MODELS[0])

    changed = deepcopy(models[0])
    changed["family"] = "legacy"
    assert "family" in exp.model_record_errors(changed, exp.PLANNED_MODELS[0])

    changed = deepcopy(models[0])
    changed["tokenizer"]["source"] = "transformers"
    assert "tokenizer" in exp.model_record_errors(changed, exp.PLANNED_MODELS[0])


def test_req_cl_6773_resolution_uses_cached_pair_and_embedded_tokenizers(tmp_path: Path) -> None:
    """REQ-CL-6773 resolves exact Qwen and Gemma files before loading."""
    models = _model_records(tmp_path)
    cached_rows = [
        {
            "hf_id": row["model_id"],
            "model_path": row["model_path"],
            "name": row["family"],
            "gpu": index,
        }
        for index, row in enumerate(models)
    ]
    calls = []

    def resolver(**kwargs):
        calls.append(kwargs)
        return cached_rows

    rows = exp.resolve_model_specs(
        pair_resolver=resolver,
        tokenizer_probe=lambda path: (True, f"embedded:{Path(path).name}"),
        file_hasher=lambda path: next(
            row["model_sha256"] for row in models if row["model_path"] == str(path)
        ),
    )
    assert calls == [{"gpu_indices": (0, 1), "model_indices": (0, 2)}]
    assert [row["model_id"] for row in rows] == [row["model_id"] for row in models]
    assert all(row["tokenizer"]["loadable"] is True for row in rows)
    assert all(row["revision"] == "local-unversioned" for row in rows)

    unresolved = exp.resolve_model_specs(pair_resolver=lambda **_: None)
    assert [row["model_path"] for row in unresolved] == ["", ""]
    assert [row["model_sha256"] for row in unresolved] == ["missing", "missing"]


def test_req_cl_6773_resolution_helpers_reject_missing_identity_fields(
    tmp_path: Path,
) -> None:
    """REQ-CL-6773 covers snapshot revisions and fail-closed local identities."""
    assert exp.sha256_file(tmp_path / "absent.gguf") == "missing"
    snapshot = tmp_path / "snapshots" / "revision-abc" / "model.gguf"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"model")
    assert exp._revision_from_path(snapshot) == "revision-abc"

    model = _model_records(tmp_path)[0]
    model["model_path"] = ""
    model["model_size_bytes"] = 0
    errors = exp.model_record_errors(model, exp.PLANNED_MODELS[0])
    assert {"model_path", "model_size_bytes"} <= set(errors)


def test_req_cl_6773_prompt_and_json_input_fail_closed(tmp_path: Path, monkeypatch) -> None:
    """REQ-CL-6773 rejects oversized context and unreadable JSON inputs."""
    monkeypatch.setattr(exp, "CANARY_PROMPT_MAX_BYTES", 1)
    with pytest.raises(ValueError, match="byte budget"):
        exp.build_canary_prompt(_stream_fixture())
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    assert exp._load_json(malformed) == {}
    assert exp._load_json(tmp_path / "missing.json") == {}


def test_req_cl_6773_collects_every_precondition_before_loading(
    tmp_path: Path, monkeypatch
) -> None:
    """SCENARIO-CL-6773-BLOCKED records every resource observation first."""
    source = tmp_path / "source.json"
    source.write_text(json.dumps(_stream_fixture()))
    models = _model_records(tmp_path)
    inventory = {
        "devices": _devices(),
        "device_query": {"command": "device-query"},
        "process_query": {"command": "process-query"},
    }
    monkeypatch.setattr(exp, "sha256_file", lambda _: exp.EXPECTED_SOURCE_ARTIFACT_SHA256)
    receipt = exp.collect_preconditions(
        source_path=source,
        model_resolver=lambda: models,
        inventory_fn=lambda: inventory,
        llama_receipt_fn=lambda: {
            "exists": True,
            "executable": True,
            "cuda_linked": True,
            "python_cuda_offload": True,
        },
        port_picker=lambda count: list(range(46773, 46773 + count)),
        port_probe=lambda _: True,
        resource_fn=lambda _: {
            "ram_available_bytes": exp.RAM_AVAILABLE_FLOOR_BYTES,
            "disk_free_bytes": exp.DISK_FREE_FLOOR_BYTES,
        },
        stream_validator=lambda _: [],
    )
    assert receipt["all_passed"] is True
    assert receipt["device_selection_receipt"]["selected_device"]["index"] == 1
    assert receipt["device_inventory_commands"] == {
        "devices": {"command": "device-query"},
        "processes": {"command": "process-query"},
    }


def test_req_infra_6773_live_device_selector_refreshes_inventory(monkeypatch) -> None:
    """REQ-INFRA-6773 selects from a new two-device inventory before a load."""
    inventory = {
        "devices": _devices(),
        "device_query": {"exit_code": 0},
        "process_query": {"exit_code": 0},
    }
    monkeypatch.setattr(exp.infra, "nvidia_smi_inventory", lambda: inventory)
    selection = exp.select_device_before_load()
    assert selection["selected_device"]["index"] == 1
    assert selection["inventory_commands"] == {
        "devices": {"exit_code": 0},
        "processes": {"exit_code": 0},
    }


@pytest.mark.parametrize(
    ("mutate", "failed_check"),
    [
        (lambda row: row.update(procedural_memory_stream_ready=False), "stream_ready"),
        (lambda row: row.update(order_count=5), "order_count"),
        (lambda row: row["stream_manifest"].update(stream_hash="sha256:bad"), "stream_hash"),
        (
            lambda row: row["capacity_contract"]["arms"]["procedural_lesson"].update(top_k=9),
            "capacity_contract",
        ),
        (lambda row: row.update(read_only_episode_enforced=False), "read_only_episode"),
        (lambda row: row["transaction_schema"].update(version=2), "transaction_schema"),
        (lambda row: row.update(restart_receipts=[]), "restart_receipts"),
        (lambda row: row.update(rollback_receipts=[]), "rollback_receipts"),
        (lambda row: row.update(poison_fixture_receipts=[]), "poison_receipts"),
    ],
)
def test_scenario_cl_6773_stream_checks_fail_closed(mutate, failed_check) -> None:
    """SCENARIO-CL-6773-STREAM keeps each frozen contract check separate."""
    fixture = _stream_fixture()
    mutate(fixture)
    checks = exp.stream_contract_checks(
        fixture,
        source_artifact_sha256=exp.EXPECTED_SOURCE_ARTIFACT_SHA256,
        upstream_validator_errors=[],
    )
    by_name = {row["check"]: row for row in checks}
    assert by_name[failed_check]["passed"] is False


def test_scenario_cl_6773_stream_manifest_and_prompt_are_compact_and_stable() -> None:
    """SCENARIO-CL-6773-STREAM copies frozen identities, not the full stream."""
    fixture = _stream_fixture()
    manifest = exp.compact_stream_manifest(fixture)
    prompt = exp.build_canary_prompt(fixture)
    assert manifest["stream_hash"] == exp.EXPECTED_STREAM_HASH
    assert len(manifest["orders"]) == 6
    assert manifest["orders"][0]["event_ids"] == ["a01", "r01"]
    assert "Reject an index equal to the length" in prompt
    assert "Use an exclusive upper bound" in prompt
    assert len(prompt.encode()) <= exp.CANARY_PROMPT_MAX_BYTES


def test_req_infra_6773_phase_rows_and_receipt_validation(tmp_path: Path) -> None:
    """REQ-INFRA-6773 derives one row from every owned lease phase."""
    model = _model_records(tmp_path)[0]
    receipt = _gpu_receipt(model, 0)
    rows = exp.phase_rows_for_receipt(receipt)
    assert [row["phase"] for row in rows] == list(exp.COMPLETE_PHASE_SEQUENCE)
    assert all(row["row_kind"] == "model_phase" for row in rows)
    assert exp.gpu_receipt_errors(receipt, model) == []

    changed = deepcopy(receipt)
    changed["offload_full"] = False
    changed["receipt_sha256"] = exp.gpu_receipt_checksum(changed)
    assert "offload_full" in exp.gpu_receipt_errors(changed, model)

    changed = deepcopy(receipt)
    changed["unrelated_processes_signaled"] = [81]
    changed["receipt_sha256"] = exp.gpu_receipt_checksum(changed)
    assert "unrelated_processes_signaled" in exp.gpu_receipt_errors(changed, model)

    receipt["phase_history"].insert(0, "not-a-phase-row")
    assert len(exp.phase_rows_for_receipt(receipt)) == len(exp.COMPLETE_PHASE_SEQUENCE)


def test_req_infra_6773_receipt_validator_reports_each_failed_gate(tmp_path: Path) -> None:
    """REQ-INFRA-6773 names each independently invalid lifecycle field."""
    model = _model_records(tmp_path)[0]
    receipt = _gpu_receipt(model, 0)
    receipt["receipt_sha256"] = "sha256:bad"
    receipt["model_id"] = "wrong"
    receipt["device"] = {"uuid": "GPU-wrong"}
    receipt["lease_owner"] = {}
    receipt["phase_history"] = []
    receipt["lease_release"] = {}
    receipt["gpu_layers"] = {}
    receipt["offload_full"] = False
    receipt["resident_owned_vram_mb"] = 0
    receipt["peak_owned_vram_mb"] = 0
    receipt["first_token_canary"] = {}
    receipt["worker_process"] = {}
    receipt["vram_recovery"] = {}
    receipt["errors"] = ["failed"]
    assert {
        "receipt_sha256",
        "model_record",
        "device_uuid",
        "lease_owner",
        "phase_sequence",
        "lease_release",
        "offload_full",
        "resident_owned_vram_mb",
        "peak_owned_vram_mb",
        "first_token_canary",
        "worker_process",
        "vram_recovery",
        "errors",
    } <= set(exp.gpu_receipt_errors(receipt, model))

    invalid_model = deepcopy(model)
    invalid_model["model_path"] = ""
    assert "model_identity" in exp.gpu_receipt_errors(receipt, invalid_model)


def test_scenario_report_6773_live_artifact_is_row_derived(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6773-LIVE derives readiness from rows and receipts."""
    artifact = _ready_artifact(tmp_path)
    assert artifact["csl_live_preflight_ready"] is True
    assert artifact["live_model_invoked"] is True
    assert artifact["model_specs"] == artifact["models_used"]
    assert artifact["verdict_class"] == "positive"
    assert artifact["honest_verdict"].startswith("complete_")
    assert len([row for row in artifact["rows"] if row["row_kind"] == "model_phase"]) == 16
    assert len([row for row in artifact["rows"] if row["row_kind"] == "stream_contract"]) == len(
        exp.STREAM_CHECK_NAMES
    )
    assert exp.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda row: row.pop("model_specs"), "required_field_set"),
        (lambda row: row["field_principles"].pop("rows"), "field_principles"),
        (lambda row: row["model_specs"][0].update(family="wrong"), "model_specs"),
        (lambda row: row.update(models_used=[]), "models_used"),
        (lambda row: row.update(csl_live_preflight_ready=False), "csl_live_preflight_ready"),
        (lambda row: row["rows"].pop(), "rows"),
        (lambda row: row.update(reproducibility_checksum="sha256:bad"), "reproducibility_checksum"),
    ],
)
def test_req_report_6773_cold_validator_rejects_tampering(
    tmp_path: Path, mutate, expected: str
) -> None:
    """REQ-REPORT-6773 rejects copied or identity-mismatched fields."""
    artifact = _ready_artifact(tmp_path)
    mutate(artifact)
    assert expected in exp.validate_artifact(artifact)


def test_req_report_6773_cold_validator_covers_all_metadata_gates(tmp_path: Path) -> None:
    """REQ-REPORT-6773 rejects every non-derived metadata substitution."""
    artifact = _ready_artifact(tmp_path)
    artifact.update(
        schema="wrong",
        run_date="20260829",
        inference_substrate="cpu",
        duration_s=float("nan"),
        random_seed=0,
        verifier_is_oracle=True,
        verdict_class="invented",
        live_model_invoked=False,
        status="blocked",
        honest_verdict="complete_wrong",
        source_artifact_sha256="sha256:bad",
        reproducibility_checksum="sha256:bad",
    )
    artifact["lease_receipts"] = []
    artifact["teardown_receipts"] = []
    artifact["gate_check_summary"] = {}
    artifact["stream_manifest"] = {}
    assert {
        "schema",
        "run_date",
        "inference_substrate",
        "duration_s",
        "random_seed",
        "verifier_is_oracle",
        "verdict_class",
        "live_model_invoked",
        "status",
        "honest_verdict",
        "source_artifact_sha256",
        "lease_receipts",
        "teardown_receipts",
        "gate_check_summary",
        "stream_manifest",
        "reproducibility_checksum",
    } <= set(exp.validate_artifact(artifact))


def test_scenario_report_6773_blocked_artifact_keeps_complete_schema(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6773-BLOCKED retains the failed expected and observed values."""
    models = _model_records(tmp_path)
    fixture = _stream_fixture()
    preconditions = _preconditions(models, fixture, passed=False)
    artifact = exp.build_artifact(
        date=exp.RUN_DATE,
        preconditions=preconditions,
        gpu_receipts=[],
        code_receipts={"module": f"sha256:{91:064x}"},
        started_ns=10,
        finished_ns=20,
    )
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["models_used"] == []
    assert artifact["rows"] == exp.stream_rows(preconditions["stream_contract_checks"])
    assert artifact["csl_live_preflight_ready"] is False
    assert artifact["live_model_invoked"] is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "complete_blocked_csl_owned_lease_contract"
    assert artifact["gate_check_summary"]["failures"] == [
        {
            "check": "planning_date_matches",
            "expected": True,
            "observed": False,
        }
    ]
    assert exp.validate_artifact(artifact) == []


def test_req_infra_6773_partial_lifecycle_does_not_claim_readiness(tmp_path: Path) -> None:
    """REQ-INFRA-6773 closes admission after a failed second teardown."""
    models = _model_records(tmp_path)
    fixture = _stream_fixture()
    receipts = [_gpu_receipt(models[0], 0), _gpu_receipt(models[1], 1, error="teardown")]
    artifact = exp.build_artifact(
        date=exp.RUN_DATE,
        preconditions=_preconditions(models, fixture),
        gpu_receipts=receipts,
        code_receipts={"module": f"sha256:{91:064x}"},
        started_ns=10,
        finished_ns=20,
    )
    assert artifact["csl_live_preflight_ready"] is False
    assert artifact["verdict_class"] == "partial"
    assert artifact["models_used"] == models
    assert (
        "model_lifecycle:" + models[1]["model_id"]
        in artifact["gate_check_summary"]["failed_checks"]
    )


def test_req_report_6773_live_invocation_records_the_first_observed_token(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6773 records live use even when later lifecycle admission fails."""
    models = _model_records(tmp_path)
    fixture = _stream_fixture()
    receipt = _gpu_receipt(models[0], 0, error="recovery")
    artifact = exp.build_artifact(
        date=exp.RUN_DATE,
        preconditions=_preconditions(models, fixture),
        gpu_receipts=[receipt],
        code_receipts={"module": f"sha256:{91:064x}"},
        started_ns=10,
        finished_ns=20,
    )
    assert artifact["models_used"] == [models[0]]
    assert artifact["live_model_invoked"] is True
    assert artifact["csl_live_preflight_ready"] is False


def test_req_infra_6773_vram_recovery_receipt_uses_frozen_tolerance() -> None:
    """SCENARIO-INFRA-6773-SEQUENTIAL records process absence and VRAM return."""
    assert exp.build_vram_recovery_receipt(500, 1012, False)["passed"] is True
    assert exp.build_vram_recovery_receipt(500, 1013, False)["passed"] is False
    assert exp.build_vram_recovery_receipt(500, 500, True)["passed"] is False


def test_req_infra_6773_process_and_gpu_snapshots_are_owner_scoped(monkeypatch) -> None:
    """REQ-INFRA-6773 binds process identity and selected-device observations."""
    missing = exp._process_identity(-1)
    assert missing["pid"] == -1
    assert missing["executable"] == ""

    device = deepcopy(_devices()[0])
    monkeypatch.setattr(
        exp.infra,
        "nvidia_smi_inventory",
        lambda: {"devices": [device]},
    )
    snapshot = exp._gpu_snapshot(device["uuid"], 81)
    assert snapshot["owned_pid_present"] is True
    assert snapshot["owned_pid_vram_mb"] == 100


def test_req_infra_6773_terminalizer_handles_resident_and_nonterminal_leases() -> None:
    """REQ-INFRA-6773 terminalizes only through the legal phase journal."""
    device = _devices()[1]
    model = {"model_path": "/model.gguf"}
    lease = _FakeLease(model, device)
    lease.transition("admitted")
    lease.transition("loading")
    lease.transition("resident")
    released = exp._terminalize_lease(lease, True, {"memory_used_mb": 500})
    assert released["phase"] == "terminal_complete"

    class NonterminalLease:
        document = {"phase": "unknown"}

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    nonterminal = NonterminalLease()
    assert exp._terminalize_lease(nonterminal, False, {}) == {}
    assert nonterminal.closed is True


def test_req_infra_6773_parent_recovery_retries_until_owner_is_absent(monkeypatch) -> None:
    """SCENARIO-INFRA-6773-SEQUENTIAL waits for parent-observed VRAM recovery."""
    snapshots = iter(
        [
            {"memory_used_mb": 2000, "owned_pid_present": True, "observed_monotonic_ns": 1},
            {"memory_used_mb": 500, "owned_pid_present": False, "observed_monotonic_ns": 2},
        ]
    )
    monkeypatch.setattr(exp, "_gpu_snapshot", lambda *_: next(snapshots))
    monkeypatch.setattr(exp.time, "sleep", lambda _: None)
    receipt = exp._wait_parent_recovery("GPU-test", 99, 500, timeout_s=5)
    assert receipt["passed"] is True
    assert receipt["observed_monotonic_ns"] == 2


class _FakeLease:
    def __init__(self, model: dict, device: dict) -> None:
        self.model = model
        self.device = device
        self.phases = ["preflight"]
        self.document = {"phase": "preflight", "phase_history": _phase_history()[:-7]}

    def owner_receipt(self) -> dict:
        return {
            "task_id": "exp6773-worker",
            "device_uuid": self.device["uuid"],
            "pid": 9001,
            "pid_start_ticks": 411,
            "expected_model": self.model["model_path"],
            "signals_sent": [],
        }

    def transition(self, phase: str, **kwargs) -> None:
        del kwargs
        self.phases.append(phase)
        self.document["phase"] = phase
        if phase == "terminal_blocked":
            current = deepcopy(self.document["phase_history"])
            previous = current[-1]["phase"] if current else None
            current.append(
                {
                    "phase": phase,
                    "previous_phase": previous,
                    "monotonic_ns": 2000,
                    "event_checksum": f"sha256:{99:064x}",
                }
            )
            self.document["phase_history"] = current
            return
        self.document["phase_history"] = [
            row
            for row in _phase_history()
            if exp.COMPLETE_PHASE_SEQUENCE.index(row["phase"])
            <= exp.COMPLETE_PHASE_SEQUENCE.index(phase)
        ]

    def release(self) -> dict:
        return {
            "released": True,
            "phase": self.document["phase"],
            "device_uuid": self.device["uuid"],
            "pid": 9001,
            "pid_start_ticks": 411,
            "signals_sent": [],
        }

    def close(self) -> None:
        return None


class _FakeLlama:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.closed = False

    def create_completion(self, **kwargs) -> dict:
        assert kwargs["max_tokens"] == 1
        return {
            "choices": [{"text": "A"}],
            "usage": {"completion_tokens": 1},
        }

    def close(self) -> None:
        self.closed = True


def test_req_infra_6773_live_worker_owns_lease_and_runs_one_token(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6773-LEASE binds the live call to one owner and model."""
    model = _model_records(tmp_path)[0]
    device = _devices()[1]
    snapshots = iter(
        [
            {**device, "owned_pid_present": False, "owned_pid_vram_mb": 0},
            {
                **device,
                "memory_used_mb": 18500,
                "owned_pid_present": True,
                "owned_pid_vram_mb": 18000,
            },
            {
                **device,
                "memory_used_mb": 18600,
                "owned_pid_present": True,
                "owned_pid_vram_mb": 18100,
            },
            {**device, "memory_used_mb": 520, "owned_pid_present": False, "owned_pid_vram_mb": 0},
        ]
    )
    lease_box = {}

    def lease_factory(**kwargs):
        assert kwargs["expected_model"] == model["model_path"]
        lease_box["lease"] = _FakeLease(model, device)
        return lease_box["lease"]

    receipt = exp.run_live_model_worker(
        model,
        device,
        prompt="fixture prompt",
        lease_runtime_dir=tmp_path / "leases",
        llama_factory=_FakeLlama,
        lease_factory=lease_factory,
        snapshot_fn=lambda *_: next(snapshots),
        sleep_fn=lambda _: None,
    )
    assert receipt["first_token_canary"]["first_token_observed"] is True
    assert receipt["first_token_canary"]["completion_tokens"] == 1
    assert receipt["unrelated_processes_signaled"] == []
    assert lease_box["lease"].phases == list(exp.COMPLETE_PHASE_SEQUENCE)
    assert receipt["lease_release"]["released"] is True
    assert receipt["vram_recovery"]["passed"] is True


def test_req_infra_6773_live_worker_rechecks_device_before_acquiring_lease(
    tmp_path: Path,
) -> None:
    """REQ-INFRA-6773 refuses a device that lost eligibility before its load."""
    model = _model_records(tmp_path)[0]
    device = _devices()[1]
    changed = {
        **device,
        "memory_used_mb": 18_000,
        "memory_free_mb": 6_000,
        "owned_pid_present": False,
        "owned_pid_vram_mb": 0,
    }
    lease_called = False

    def lease_factory(**kwargs):
        nonlocal lease_called
        del kwargs
        lease_called = True
        raise AssertionError("an ineligible device must not acquire a lease")

    receipt = exp.run_live_model_worker(
        model,
        device,
        prompt="fixture prompt",
        lease_runtime_dir=tmp_path / "leases",
        llama_factory=_FakeLlama,
        lease_factory=lease_factory,
        snapshot_fn=lambda *_: deepcopy(changed),
        sleep_fn=lambda _: None,
    )
    assert lease_called is False
    assert receipt["errors"] == ["RuntimeError: selected_device_recheck_failed"]
    assert receipt["first_token_canary"]["first_token_observed"] is False


def test_req_infra_6773_live_worker_failure_releases_a_blocked_lease(tmp_path: Path) -> None:
    """REQ-INFRA-6773 records a load failure without inventing a token."""
    model = _model_records(tmp_path)[0]
    device = _devices()[1]
    lease = _FakeLease(model, device)

    class BrokenLlama:
        def __init__(self, **kwargs) -> None:
            del kwargs
            raise RuntimeError("load failed")

    snapshots = iter(
        [
            {**device, "owned_pid_present": False, "owned_pid_vram_mb": 0},
            {**device, "memory_used_mb": 500, "owned_pid_present": False, "owned_pid_vram_mb": 0},
        ]
    )
    receipt = exp.run_live_model_worker(
        model,
        device,
        prompt="fixture prompt",
        lease_runtime_dir=tmp_path / "leases",
        llama_factory=BrokenLlama,
        lease_factory=lambda **_: lease,
        snapshot_fn=lambda *_: next(snapshots),
        sleep_fn=lambda _: None,
    )
    assert receipt["first_token_canary"]["first_token_observed"] is False
    assert receipt["errors"] == ["RuntimeError: load failed"]
    assert receipt["lease_release"]["phase"] == "terminal_blocked"
    assert receipt["unrelated_processes_signaled"] == []


def test_req_infra_6773_live_worker_imports_llama_and_retries_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-INFRA-6773 uses the local binding and waits for one delayed teardown."""
    model = _model_records(tmp_path)[0]
    device = _devices()[1]
    lease = _FakeLease(model, device)
    snapshots = iter(
        [
            {**device, "owned_pid_present": False, "owned_pid_vram_mb": 0},
            {
                **device,
                "memory_used_mb": 18000,
                "owned_pid_present": True,
                "owned_pid_vram_mb": 17500,
            },
            {
                **device,
                "memory_used_mb": 18100,
                "owned_pid_present": True,
                "owned_pid_vram_mb": 17600,
            },
            {
                **device,
                "memory_used_mb": 18000,
                "owned_pid_present": True,
                "owned_pid_vram_mb": 17500,
            },
            {**device, "memory_used_mb": 500, "owned_pid_present": False, "owned_pid_vram_mb": 0},
        ]
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=_FakeLlama))
    receipt = exp.run_live_model_worker(
        model,
        device,
        prompt="fixture prompt",
        lease_runtime_dir=tmp_path / "leases",
        lease_factory=lambda **_: lease,
        snapshot_fn=lambda *_: next(snapshots),
        sleep_fn=lambda _: None,
    )
    assert receipt["errors"] == []
    assert receipt["vram_recovery"]["passed"] is True


@pytest.mark.parametrize("failure", ["residency", "token"])
def test_req_infra_6773_live_worker_rejects_missing_cuda_or_token(
    tmp_path: Path, failure: str
) -> None:
    """REQ-INFRA-6773 does not infer readiness from load alone."""
    model = _model_records(tmp_path)[0]
    device = _devices()[1]
    lease = _FakeLease(model, device)

    class NoTokenLlama(_FakeLlama):
        def create_completion(self, **kwargs) -> dict:
            del kwargs
            return {"choices": [], "usage": {"completion_tokens": 0}}

    if failure == "residency":
        factory = _FakeLlama
        snapshots = iter(
            [
                {**device, "owned_pid_present": False, "owned_pid_vram_mb": 0},
                {**device, "owned_pid_present": False, "owned_pid_vram_mb": 0},
                {**device, "owned_pid_present": False, "owned_pid_vram_mb": 0},
            ]
        )
    else:
        factory = NoTokenLlama
        snapshots = iter(
            [
                {**device, "owned_pid_present": False, "owned_pid_vram_mb": 0},
                {
                    **device,
                    "memory_used_mb": 18000,
                    "owned_pid_present": True,
                    "owned_pid_vram_mb": 17500,
                },
                {
                    **device,
                    "memory_used_mb": 18100,
                    "owned_pid_present": True,
                    "owned_pid_vram_mb": 17600,
                },
                {**device, "owned_pid_present": False, "owned_pid_vram_mb": 0},
            ]
        )
    receipt = exp.run_live_model_worker(
        model,
        device,
        prompt="fixture prompt",
        lease_runtime_dir=tmp_path / "leases",
        llama_factory=factory,
        lease_factory=lambda **_: lease,
        snapshot_fn=lambda *_: next(snapshots),
        sleep_fn=lambda _: None,
    )
    marker = (
        "owner_bound_cuda_residency_missing"
        if failure == "residency"
        else "first_token_not_observed"
    )
    assert marker in receipt["errors"][0]


def test_req_infra_6773_live_worker_records_teardown_and_lease_errors(tmp_path: Path) -> None:
    """REQ-INFRA-6773 preserves close and terminalization failures in the receipt."""
    model = _model_records(tmp_path)[0]
    device = _devices()[1]

    class BrokenCloseLlama(_FakeLlama):
        def close(self) -> None:
            raise RuntimeError("close failed")

    class BrokenUnloadLease(_FakeLease):
        def transition(self, phase: str, **kwargs) -> None:
            if phase == "unloading":
                raise lease_api.LeaseError("unload failed")
            super().transition(phase, **kwargs)

        def close(self) -> None:
            self.phases.append("closed")

    lease = BrokenUnloadLease(model, device)
    snapshots = iter(
        [
            {**device, "owned_pid_present": False, "owned_pid_vram_mb": 0},
            {
                **device,
                "memory_used_mb": 18000,
                "owned_pid_present": True,
                "owned_pid_vram_mb": 17500,
            },
            {
                **device,
                "memory_used_mb": 18100,
                "owned_pid_present": True,
                "owned_pid_vram_mb": 17600,
            },
            {**device, "owned_pid_present": False, "owned_pid_vram_mb": 0},
        ]
    )
    receipt = exp.run_live_model_worker(
        model,
        device,
        prompt="fixture prompt",
        lease_runtime_dir=tmp_path / "leases",
        llama_factory=BrokenCloseLlama,
        lease_factory=lambda **_: lease,
        snapshot_fn=lambda *_: next(snapshots),
        sleep_fn=lambda _: None,
    )
    assert any("unload failed" in error for error in receipt["errors"])
    assert any("close failed" in error for error in receipt["errors"])
    assert lease.phases[-1] == "closed"


def test_req_cl_6773_run_stops_on_preflight_and_runs_workers_sequentially(tmp_path: Path) -> None:
    """SCENARIO-CL-6773-BLOCKED and SCENARIO-INFRA-6773-SEQUENTIAL control execution."""
    models = _model_records(tmp_path)
    fixture = _stream_fixture()
    blocked_path = tmp_path / "blocked.json"
    calls = []
    blocked = exp.run(
        result_path=blocked_path,
        date=exp.RUN_DATE,
        preflight_fn=lambda: _preconditions(models, fixture, passed=False),
        worker_runner=lambda *args: calls.append(args),
        code_receipt_fn=lambda: {"module": f"sha256:{91:064x}"},
        clock=iter([10, 20]).__next__,
    )
    assert calls == []
    assert blocked_path.is_file()
    assert blocked["honest_verdict"] == "complete_blocked_csl_owned_lease_contract"

    ready_path = tmp_path / "ready.json"
    sequence = []

    def worker(model, device, prompt, runtime_dir):
        del device, prompt, runtime_dir
        index = [row["model_id"] for row in models].index(model["model_id"])
        sequence.append(model["model_id"])
        return _gpu_receipt(model, index)

    ready = exp.run(
        result_path=ready_path,
        date=exp.RUN_DATE,
        preflight_fn=lambda: _preconditions(models, fixture),
        device_selector=lambda: exp.rank_eligible_devices(_devices()),
        worker_runner=worker,
        code_receipt_fn=lambda: {"module": f"sha256:{91:064x}"},
        clock=iter([10, 20]).__next__,
    )
    assert sequence == [row["model_id"] for row in models]
    assert ready["csl_live_preflight_ready"] is True
    assert json.loads(ready_path.read_text())["model_specs"] == models


def test_req_infra_6773_run_reselects_the_least_used_device_before_each_model(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6773-SEQUENTIAL refreshes device choice after recovery."""
    models = _model_records(tmp_path)
    fixture = _stream_fixture()
    devices = _devices()
    selections = iter(
        [
            {"selected_device": deepcopy(devices[1])},
            {"selected_device": deepcopy(devices[0])},
        ]
    )
    observed_uuids = []

    def worker(model, device, prompt, runtime_dir):
        del prompt, runtime_dir
        observed_uuids.append(device["uuid"])
        index = [row["model_id"] for row in models].index(model["model_id"])
        receipt = _gpu_receipt(model, index)
        receipt["device"] = deepcopy(device)
        receipt["lease_owner"]["device_uuid"] = device["uuid"]
        receipt["lease_release"]["device_uuid"] = device["uuid"]
        receipt["receipt_sha256"] = exp.gpu_receipt_checksum(receipt)
        return receipt

    artifact = exp.run(
        result_path=tmp_path / "reselected.json",
        date=exp.RUN_DATE,
        preflight_fn=lambda: _preconditions(models, fixture),
        device_selector=lambda: next(selections),
        worker_runner=worker,
        code_receipt_fn=lambda: {"module": f"sha256:{91:064x}"},
        clock=iter([10, 20]).__next__,
    )
    assert observed_uuids == [devices[1]["uuid"], devices[0]["uuid"]]
    assert artifact["csl_live_preflight_ready"] is True


def test_req_infra_6773_run_stops_when_refreshed_devices_are_ineligible(
    tmp_path: Path,
) -> None:
    """REQ-INFRA-6773 retains a changed device gate instead of loading a model."""
    models = _model_records(tmp_path)
    fixture = _stream_fixture()
    workers = []
    selection = {"selected_device": None, "eligible_devices": []}
    artifact = exp.run(
        result_path=tmp_path / "device-blocked.json",
        date=exp.RUN_DATE,
        preflight_fn=lambda: _preconditions(models, fixture),
        device_selector=lambda: selection,
        worker_runner=lambda *args: workers.append(args),
        code_receipt_fn=lambda: {"module": f"sha256:{91:064x}"},
        clock=iter([10, 20]).__next__,
    )
    assert workers == []
    assert artifact["status"] == "partial"
    assert artifact["live_model_invoked"] is False
    device_failure = next(
        row
        for row in artifact["gate_check_summary"]["failures"]
        if str(row["check"]).startswith("device_recheck:")
    )
    assert device_failure["observed"] == selection


def test_req_cl_6773_run_stops_after_first_failed_lifecycle(tmp_path: Path) -> None:
    """REQ-CL-6773 never starts a second model after failed recovery."""
    models = _model_records(tmp_path)
    fixture = _stream_fixture()
    calls = []

    def worker(model, *_):
        calls.append(model["model_id"])
        return _gpu_receipt(model, 0, error="recovery")

    artifact = exp.run(
        result_path=tmp_path / "partial.json",
        date=exp.RUN_DATE,
        preflight_fn=lambda: _preconditions(models, fixture),
        device_selector=lambda: exp.rank_eligible_devices(_devices()),
        worker_runner=worker,
        code_receipt_fn=lambda: {"module": f"sha256:{91:064x}"},
        clock=iter([10, 20]).__next__,
    )
    assert calls == [models[0]["model_id"]]
    assert artifact["verdict_class"] == "partial"


def test_req_infra_6773_parent_worker_records_logs_and_recovery(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-INFRA-6773 parent-binds worker exit, offload logs, and recovery."""
    model = _model_records(tmp_path)[0]
    device = _devices()[1]
    expected = _gpu_receipt(model, 0)

    class Process:
        pid = 7000
        returncode = 0

        def communicate(self, timeout=None):
            assert timeout == 2
            return "worker output", "offloaded 65/65 layers to GPU"

        def poll(self):
            return 0

    monkeypatch.setattr(exp.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(exp.lease_api, "proc_start_ticks", lambda _: 300)
    monkeypatch.setattr(exp, "_load_json", lambda _: deepcopy(expected))
    monkeypatch.setattr(
        exp,
        "_wait_parent_recovery",
        lambda *_: exp.build_vram_recovery_receipt(500, 500, False),
    )
    receipt = exp.run_model_worker(model, device, "prompt", tmp_path / "runtime", timeout_s=2)
    assert receipt["worker_process"]["absent_after_exit"] is True
    assert receipt["offload_full"] is True
    assert receipt["gpu_layers"] == {"requested": -1, "offloaded": 65, "total": 65}


def test_req_infra_6773_parent_worker_timeout_emits_blocked_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    """REQ-INFRA-6773 timeout cleanup targets only the fresh worker session."""
    model = _model_records(tmp_path)[0]
    device = _devices()[1]

    class Process:
        pid = 8000
        returncode = -15

        def __init__(self) -> None:
            self.calls = 0

        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("worker", timeout)
            return "", "timed out"

        def poll(self):
            return -15

    process = Process()
    cleanup = {"term_sent": True, "unrelated_processes_signaled": []}
    monkeypatch.setattr(exp.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(exp.lease_api, "proc_start_ticks", lambda _: 400)
    monkeypatch.setattr(exp.infra, "_terminate_worker_group", lambda *_: cleanup)
    monkeypatch.setattr(exp, "_load_json", lambda _: {})
    monkeypatch.setattr(
        exp,
        "_wait_parent_recovery",
        lambda *_: exp.build_vram_recovery_receipt(500, 500, False),
    )
    receipt = exp.run_model_worker(model, device, "prompt", tmp_path / "runtime", timeout_s=1)
    assert receipt["errors"][0].startswith("worker_output_missing")
    assert receipt["worker_process"]["timeout_cleanup"] == cleanup
    assert receipt["unrelated_processes_signaled"] == []


def test_req_report_6773_write_and_cli_validation(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-6773 publishes atomically and validates without mutation."""
    artifact = _ready_artifact(tmp_path)
    path = tmp_path / "artifact.json"
    receipt = exp.write_artifact(path, artifact)
    assert receipt["atomic_rename"] is True
    assert receipt["sha256"] == _sha(path.read_bytes())
    assert exp.main(["--validate", "--result-path", str(path)]) == 0

    with pytest.raises(ValueError, match="planning date"):
        exp.main(["--date", "20260829", "--result-path", str(path)])

    monkeypatch.setattr(exp, "run", lambda **_: artifact)
    assert exp.main(["--date", exp.RUN_DATE, "--result-path", str(path)]) == 0


def test_req_report_6773_worker_entry_and_cli_errors(tmp_path: Path, monkeypatch) -> None:
    """REQ-REPORT-6773 worker and validator CLI branches fail closed."""
    model_path = tmp_path / "model.json"
    device_path = tmp_path / "device.json"
    prompt_path = tmp_path / "prompt.json"
    output_path = tmp_path / "output.json"
    model = {"model_id": "fixture"}
    device = {"uuid": "GPU-fixture"}
    model_path.write_text(json.dumps(model))
    device_path.write_text(json.dumps(device))
    prompt_path.write_text(json.dumps({"prompt": "canary"}))
    monkeypatch.setattr(
        exp,
        "run_live_model_worker",
        lambda observed_model, observed_device, **kwargs: {
            "model_id": observed_model["model_id"],
            "device_uuid": observed_device["uuid"],
            "prompt": kwargs["prompt"],
            "errors": [],
        },
    )
    assert exp._worker_entry(model_path, device_path, prompt_path, output_path, tmp_path) == 0
    assert json.loads(output_path.read_text())["prompt"] == "canary"

    monkeypatch.setattr(exp, "_worker_entry", lambda *args: 7)
    assert (
        exp.main(
            [
                "--worker",
                "--worker-model",
                str(model_path),
                "--worker-device",
                str(device_path),
                "--worker-prompt",
                str(prompt_path),
                "--worker-output",
                str(output_path),
            ]
        )
        == 7
    )
    with pytest.raises(SystemExit):
        exp.main(["--worker"])

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}")
    with pytest.raises(ValueError, match="invalid Exp6773"):
        exp.main(["--validate", "--result-path", str(invalid)])


def test_req_report_6773_write_rejects_invalid_artifact(tmp_path: Path) -> None:
    """REQ-REPORT-6773 refuses an invalid terminal JSON write."""
    artifact = _ready_artifact(tmp_path)
    artifact["model_specs"] = []
    with pytest.raises(ValueError, match="model_specs"):
        exp.write_artifact(tmp_path / "bad.json", artifact)


def test_req_infra_6773_gpu_receipt_checksum_ignores_only_itself(tmp_path: Path) -> None:
    """REQ-INFRA-6773 binds every lifecycle field into one receipt hash."""
    receipt = _gpu_receipt(_model_records(tmp_path)[0], 0)
    assert receipt["receipt_sha256"] == exp.gpu_receipt_checksum(receipt)
    changed = deepcopy(receipt)
    changed["peak_owned_vram_mb"] += 1
    assert changed["receipt_sha256"] != exp.gpu_receipt_checksum(changed)


def test_req_report_6773_code_receipts_bind_module_wrapper_and_test(tmp_path: Path) -> None:
    """REQ-REPORT-6773 records the code identities used by the checksum."""
    module = tmp_path / "module.py"
    wrapper = tmp_path / "wrapper.py"
    test = tmp_path / "test.py"
    module.write_text("module\n")
    wrapper.write_text("wrapper\n")
    test.write_text("test\n")
    receipt = exp.code_receipts((module, wrapper, test))
    assert list(receipt) == [str(module), str(wrapper), str(test)]
    assert receipt[str(module)] == _sha(b"module\n")


def test_req_infra_6773_worker_environment_is_single_gpu_and_local(tmp_path: Path) -> None:
    """REQ-INFRA-6773 exposes one selected physical GPU to the local worker."""
    model = _model_records(tmp_path)[0]
    env = exp.worker_environment({"PATH": "/bin"}, model, _devices()[1])
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["CARNOT_CSL_EXPECTED_GPU_UUID"] == exp.EXPECTED_GPU_UUIDS[1]
    assert env["CARNOT_CSL_EXPECTED_MODEL"] == model["model_path"]
    assert env["PATH"] == "/bin"
