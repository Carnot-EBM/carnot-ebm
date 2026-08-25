"""Tests for the independent Exp6584 three-family receipt audit."""

from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6584_three_family_source_receipt_audit as exp


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(_canonical(value).encode("utf-8"))


def _row_hash(row: dict[str, Any]) -> str:
    return _sha_json({key: value for key, value in row.items() if key != "row_hash"})


def _protocol(unit_count: int = 2) -> dict[str, Any]:
    units = []
    for index in range(unit_count):
        source = f"source bytes {index}"
        units.append(
            {
                "unit_id": f"unit-{index}",
                "fixture_id": f"fixture-{index}",
                "case_kind": "single_hop" if index == 0 else "unsupported",
                "split": "held" if index else "train",
                "exact_source_bytes": source,
                "source_bytes_sha256": _sha_bytes(source.encode()),
                "content_hash": _sha_json({"source": source, "index": index}),
            }
        )
    manifest: dict[str, Any] = {
        "schema": "carnot.v572.source_unit_manifest.v1",
        "bounded_unit_count": len(units),
        "units": units,
    }
    manifest["manifest_hash"] = _sha_json(manifest)
    prompt = "Use only the source. Return one JSON claim object."
    budget = {
        "max_prompt_tokens": 4096,
        "max_output_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    family_rows = []
    for family in exp.FAMILY_SPECS:
        family_rows.append(
            {
                "task_id": family["task_id"],
                "model_family": family["repository_id"],
                "seed": family["seed"],
                "prompt_sha256": _sha_bytes(prompt.encode()),
                "token_budget_hash": _sha_json(budget),
                "per_source_unit_timeout_s": 720,
                "task_timeout_s": 4200,
                "family_specific_prompt_allowed": False,
            }
        )
    return {
        "v572_source_method_ready_score": 1.0,
        "source_unit_manifest": manifest,
        "prompt_seed_budget_contract": {
            "family_neutral_prompt": prompt,
            "prompt_sha256": _sha_bytes(prompt.encode()),
            "token_budget": budget,
            "family_rows": family_rows,
            "one_family_task_mapping": {
                family["task_id"]: family["repository_id"] for family in exp.FAMILY_SPECS
            },
            "stop_rules": ["<|eot_id|>", "<stop>"],
            "timeout_s": 4200,
            "raw_before_derived_write_order": True,
            "failure_retention_required": True,
            "fresh_process_per_family": True,
        },
    }


def _json_response(label: str) -> bytes:
    return json.dumps(
        {
            "claim_id": label,
            "supported_spans": [label],
            "unsupported_reason": None,
            "release_action": "release",
        },
        separators=(",", ":"),
    ).encode()


def _write_family(
    root: Path,
    protocol: dict[str, Any],
    family: dict[str, Any],
    *,
    pid: int,
    responses: list[bytes] | None = None,
    stored_ready: float = 1.0,
) -> dict[str, Any]:
    artifact_path = root / family["artifact_path"]
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = root / "raw" / family["family_id"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    blob = root / "models" / f"{family['family_id']}.gguf"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(f"GGUF-{family['family_id']}".encode())
    gguf_hash = _sha_bytes(blob.read_bytes())
    command = [
        "llama-server",
        "--model",
        str(blob),
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "none",
    ]
    command_hash = _sha_json(command)
    units = protocol["source_unit_manifest"]["units"]
    prompt = protocol["prompt_seed_budget_contract"]["family_neutral_prompt"]
    responses = responses or [
        _json_response(f"{family['family_id']}-{i}") for i in range(len(units))
    ]
    rows = []
    checkpoints = []
    diagnostics = []
    raw_receipts = []
    for index, (unit, response) in enumerate(zip(units, responses, strict=True)):
        request = f"{prompt}\n\nSOURCE BYTES:\n{unit['exact_source_bytes']}".encode()
        components = [
            {"metric": "prompt_tokens", "quantity": 10 + index, "unit_cost": 1.0},
            {"metric": "response_tokens", "quantity": 5 + index, "unit_cost": 1.0},
            {"metric": "latency_s", "quantity": 0.25 + index, "unit_cost": 1.0},
        ]
        raw_row: dict[str, Any] = {
            "row_type": "raw_terminal_source_unit",
            "unit_id": unit["unit_id"],
            "fixture_id": unit["fixture_id"],
            "case_kind": unit["case_kind"],
            "split": unit["split"],
            "order_index": index,
            "source_manifest_hash": protocol["source_unit_manifest"]["manifest_hash"],
            "source_content_hash": unit["content_hash"],
            "source_bytes_b64": base64.b64encode(unit["exact_source_bytes"].encode()).decode(),
            "source_bytes_sha256": unit["source_bytes_sha256"],
            "prompt_sha256": _sha_bytes(prompt.encode()),
            "request_bytes_b64": base64.b64encode(request).decode(),
            "request_sha256": _sha_bytes(request),
            "repository_id": family["repository_id"],
            "revision": f"revision-{family['family_id']}",
            "gguf_sha256": gguf_hash,
            "gguf_blob_path": str(blob),
            "command_sha256": command_hash,
            "pid": pid,
            "cuda_device": 0,
            "offloaded_layers": 32,
            "seed": family["seed"],
            "attempt_count": 1,
            "retry_count": 0,
            "raw_response_bytes_b64": base64.b64encode(response).decode(),
            "raw_response_byte_count": len(response),
            "raw_response_sha256": _sha_bytes(response),
            "raw_api_response_sha256": _sha_json({"response": response.decode()}),
            "prompt_token_count": 10 + index,
            "response_token_count": 5 + index,
            "total_token_count": 15 + (2 * index),
            "latency_s": 0.25 + index,
            "stop_reason": "stop",
            "request_exit_code": 0,
            "stderr_sha256_at_terminal": _sha_bytes(f"stderr-{index}".encode()),
            "failure_flags": {
                "timeout": False,
                "malformed_output": False,
                "refusal": False,
                "empty_output": False,
                "no_claim": False,
                "process_failure": False,
            },
            "charged_cost_unit": "normalized_token_and_second_units",
            "charged_cost_components": components,
            "charged_cost": 15.25 + (3 * index),
            "raw_response_recorded_monotonic_ns": 1000 + (index * 100),
        }
        raw_row["row_hash"] = _row_hash(raw_row)
        checkpoint = raw_dir / f"{index:02d}-{raw_row['row_hash'][7:]}.json"
        checkpoint.write_text(_canonical(raw_row) + "\n", encoding="utf-8")
        checkpoint_hash = _sha_bytes(checkpoint.read_bytes())
        checkpoint_receipt = {
            "unit_id": unit["unit_id"],
            "order_index": index,
            "absolute_path": str(checkpoint),
            "raw_row_hash": raw_row["row_hash"],
            "checkpoint_sha256": checkpoint_hash,
            "written_monotonic_ns": 1050 + (index * 100),
            "atomic_replace": True,
        }
        diagnostic: dict[str, Any] = {
            "unit_id": unit["unit_id"],
            "order_index": index,
            "diagnostic_only": True,
            "raw_row_hash": raw_row["row_hash"],
            "parser_started_monotonic_ns": 1075 + (index * 100),
            "raw_before_parser": True,
            "claim_bearing": True,
            "parser_can_filter_rows": False,
        }
        diagnostic["row_hash"] = _row_hash(diagnostic)
        final = dict(raw_row)
        final.update(
            {
                "raw_checkpoint_path": str(checkpoint),
                "raw_checkpoint_sha256": checkpoint_hash,
                "raw_checkpoint_row_hash": raw_row["row_hash"],
                "parser_diagnostic_row_hash": diagnostic["row_hash"],
                "claim_bearing": True,
                "process_receipt": {
                    "pid": pid,
                    "parent_pid": pid - 1,
                    "started_monotonic_ns": 100,
                    "ended_monotonic_ns": 2000,
                    "exit_code": 0,
                    "normal_shutdown": True,
                    "worker_alive_after_exit": False,
                    "stdout_sha256": _sha_bytes(b""),
                    "stderr_sha256": _sha_bytes(b"server stderr"),
                },
            }
        )
        final["row_hash"] = _row_hash(final)
        rows.append(final)
        checkpoints.append(checkpoint_receipt)
        diagnostics.append(diagnostic)
        raw_receipts.append(
            {
                "unit_id": unit["unit_id"],
                "raw_response_sha256": raw_row["raw_response_sha256"],
                "raw_response_byte_count": len(response),
                "raw_bytes_present": True,
                "recoverable_path": str(checkpoint),
                "checkpoint_sha256": checkpoint_hash,
                "raw_before_parser": True,
            }
        )
    gpu_samples = [
        {
            "stage": "before",
            "device": {"memory_used_mb": 4, "utilization_pct": 0},
            "compute_processes": [],
        },
        {
            "stage": "during",
            "device": {"memory_used_mb": 2048, "utilization_pct": 80},
            "compute_processes": [{"pid": pid, "used_memory_mb": 2044}],
        },
        {
            "stage": "during",
            "device": {"memory_used_mb": 2050, "utilization_pct": 75},
            "compute_processes": [{"pid": pid, "used_memory_mb": 2046}],
        },
        {
            "stage": "after",
            "device": {"memory_used_mb": 4, "utilization_pct": 0},
            "compute_processes": [],
        },
    ]
    process = {
        "command": command,
        "os_command": command,
        "command_sha256": command_hash,
        "os_command_sha256": command_hash,
        "command_matches_os": True,
        "pid": pid,
        "parent_pid": pid - 1,
        "os_pid_verified": True,
        "os_parent_pid_verified": True,
        "fresh_process": True,
        "selected_blob_path": str(blob),
        "gguf_sha256": gguf_hash,
        "selected_gpu": 0,
        "cuda_visible_devices": "0",
        "offloaded_layers": 32,
        "gpu_samples": gpu_samples,
        "resident_model_families": [family["repository_id"]],
        "server_healthy": True,
        "http_status": 200,
        "started_monotonic_ns": 100,
        "ended_monotonic_ns": 2000,
        "shutdown_requested": True,
        "normal_shutdown": True,
        "exit_code": 0,
        "worker_alive_after_exit": False,
        "stdout_sha256": _sha_bytes(b""),
        "stderr_sha256": _sha_bytes(b"server stderr"),
        "evidence_mode": "measured",
        "signals_sent_to_unrelated_pids": [],
        "embedded_tokenizer": True,
    }
    unload = {
        "worker_pid": pid,
        "shutdown_requested": True,
        "normal_shutdown": True,
        "exit_code": 0,
        "worker_absent_from_proc": True,
        "worker_absent_from_nvidia_smi": True,
        "port_closed": True,
        "memory_delta_from_baseline_mb": 0,
        "recovery_tolerance_mb": 256,
        "no_task_worker_remains": True,
        "recovery_bounded": True,
        "signals_sent_to_unrelated_pids": [],
        "recovery_complete": True,
    }
    payload = {
        "schema": family["schema"],
        "task_id": family["task_id"],
        "status": "complete",
        "honest_verdict": f"complete_{family['family_id']}_source_shard",
        "verdict_class": None,
        family["readiness_field"]: stored_ready,
        "rows": rows,
        "raw_response_receipts": raw_receipts,
        "checkpoint_receipts": checkpoints,
        "parser_diagnostic_rows": diagnostics,
        "process_and_gpu_receipts": process,
        "unload_and_recovery_rows": [unload],
        "model_revision_and_hash_receipt": {
            "repository_id": family["repository_id"],
            "selected_blob_path": str(blob),
            "trusted_sha256": gguf_hash,
            "provenance": {
                "revision": f"revision-{family['family_id']}",
                "valid": True,
                "repository_id": family["repository_id"],
            },
            "content_metadata": {
                "architecture": family["architecture"],
                "tokenizer_metadata": {
                    "model": family["tokenizer_model"],
                    "token_count": 256000,
                    "chat_template_present": True,
                },
            },
        },
        "protected_files_unchanged": {"all_unchanged": True},
    }
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _repo(tmp_path: Path, *, unit_count: int = 2) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _protocol(unit_count)
    protocol_path = tmp_path / exp.PROTOCOL_RELATIVE_PATH
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    for protected in exp.PROTECTED_RELATIVE_PATHS:
        path = tmp_path / protected
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"protected {protected}\n", encoding="utf-8")
    payloads = {}
    for index, family in enumerate(exp.FAMILY_SPECS):
        payloads[family["family_id"]] = _write_family(tmp_path, protocol, family, pid=2000 + index)
    return protocol, payloads


def _inspector(path: Path) -> dict[str, Any]:
    family = next(family for family in exp.FAMILY_SPECS if family["family_id"] in path.name)
    return {
        "architecture": family["architecture"],
        "tokenizer_metadata": {
            "model": family["tokenizer_model"],
            "token_count": 256000,
            "chat_template_present": True,
        },
    }


def _build(tmp_path: Path, *, protected_before: dict[str, str] | None = None) -> dict[str, Any]:
    return exp.build_audit(
        tmp_path,
        duration_s=1.25,
        tests_run=[{"command": "focused", "exit_code": 0, "duration_s": 0.1}],
        protected_before=protected_before,
        gguf_inspector=_inspector,
    )


def _rewrite(root: Path, family: dict[str, Any], payload: dict[str, Any]) -> None:
    (root / family["artifact_path"]).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def test_spec_declares_exp6584_requirements_and_scenarios() -> None:
    """REQ-REPORT-6584: OpenSpec owns replay, merge, attacks, and atomic output."""

    text = (exp.REPO_ROOT / exp.SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-6584") :]
    for anchor in (
        "REQ-REPORT-6584-PRECONDITIONS",
        "REQ-REPORT-6584-REPLAY",
        "REQ-REPORT-6584-MERGE",
        "REQ-REPORT-6584-RAW-FIRST",
        "REQ-REPORT-6584-FAILURES",
        "REQ-REPORT-6584-DUPLICATES",
        "REQ-REPORT-6584-UNLOAD",
        "REQ-REPORT-6584-REDUCER",
        "REQ-REPORT-6584-RETIREMENT",
        "REQ-REPORT-6584-ATOMIC",
        "SCENARIO-REPORT-6584-MISSING",
        "SCENARIO-REPORT-6584-REPLAY",
        "SCENARIO-REPORT-6584-MERGE",
        "SCENARIO-REPORT-6584-ATTACKS",
        "SCENARIO-REPORT-6584-UNLOAD",
        "SCENARIO-REPORT-6584-ATOMIC",
    ):
        assert anchor in section


def test_hash_recovery_failure_and_cost_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-6584-REPLAY: byte, cost, and failure reducers are independent."""

    path = tmp_path / "bytes.bin"
    path.write_bytes(b"abc")
    assert exp.sha256_file(path) == exp.sha256_bytes(b"abc")
    assert exp.sha256_file(tmp_path / "missing") == "missing"
    row = {"value": 1}
    row["row_hash"] = exp.row_hash(row)
    assert exp.row_hash(row) == row["row_hash"]
    assert exp.recover_bytes({"value_bytes_b64": "YWJj"}, "value") == (b"abc", "inline_base64")
    assert exp.recover_bytes({"value_bytes_b64": "!"}, "value") == (None, "invalid_base64")
    assert (
        exp.cost_from_components(
            [{"quantity": 2, "unit_cost": 3}, {"quantity": 0.25, "unit_cost": 4}]
        )
        == 7.0
    )
    assert exp.classify_response(b"")["empty_output"] is True
    assert exp.classify_response(b"I cannot comply")["refusal"] is True
    assert exp.classify_response(b"Please provide the claim")["no_claim"] is True
    fenced = b'```json\n{"claim_id":"x","supported_spans":[],"unsupported_reason":null,"release_action":"release"}\n```'
    classified = exp.classify_response(fenced)
    assert classified["malformed_output"] is True
    assert classified["claim_bearing"] is True


def test_clean_three_family_replay_merges_rows_and_recomputes_every_aggregate(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6584-REPLAY/MERGE/UNLOAD: all raw evidence replays."""

    protocol, _ = _repo(tmp_path)
    artifact = _build(tmp_path)
    aggregate = artifact["aggregate_row_recomputation"]
    assert artifact["all_family_source_audit_ready_score"] == 1.0
    assert artifact["verdict_class"] is None
    assert len(artifact["rows"]) == 6
    assert len(artifact["family_coverage_rows"]) == 6
    assert len(artifact["unload_and_recovery_rows"]) == 3
    assert aggregate["expected_row_count"] == 6
    assert aggregate["observed_row_count"] == 6
    assert aggregate["replayed_row_count"] == 6
    assert aggregate["source_unit_coverage"] == 1.0
    assert aggregate["failure_row_count"] == 0
    assert aggregate["prompt_token_count"] == 63
    assert aggregate["response_token_count"] == 33
    assert aggregate["total_token_count"] == 96
    assert aggregate["latency_s"] == 4.5
    assert aggregate["charged_cost"] == 100.5
    assert aggregate["family_readiness_recomputation"] == {
        family["family_id"]: 1.0 for family in exp.FAMILY_SPECS
    }
    assert all(row["passed"] for row in artifact["duplicate_drift_and_substitution_rows"])
    assert all(row["retained"] for row in artifact["failure_retention_rows"])
    assert (
        artifact["preconditions_checked"]["source_manifest_hash"]
        == protocol["source_unit_manifest"]["manifest_hash"]
    )
    assert aggregate["protocol_comparison"]["observed_seed_values"] == [6581, 6582, 6583]
    assert aggregate["protocol_comparison"]["seed_values_identical"] is False
    assert artifact["reproducibility_checksum"] == exp.artifact_checksum(artifact)
    assert exp.validate_artifact(artifact) == []


def test_missing_and_blocked_family_names_exact_field_without_synthetic_rows(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6584-MISSING: missing family rows block and stay absent."""

    _, payloads = _repo(tmp_path)
    qwen = exp.FAMILY_SPECS[0]
    blocked = payloads[qwen["family_id"]]
    blocked["status"] = "blocked"
    blocked["verdict_class"] = "blocked"
    blocked["rows"] = []
    blocked["checkpoint_receipts"] = []
    blocked["parser_diagnostic_rows"] = []
    blocked["raw_response_receipts"] = []
    blocked[qwen["readiness_field"]] = 0.0
    _rewrite(tmp_path, qwen, blocked)
    artifact = _build(tmp_path)
    assert artifact["all_family_source_audit_ready_score"] == 0.0
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    first = artifact["gate_check_summary"]["first_failure"]
    assert first["field"] == f"{qwen['artifact_path']}.status"
    assert first["observed"] == "blocked"
    assert len(artifact["rows"]) == 4
    assert len(artifact["family_coverage_rows"]) == 6
    assert sum(not row["row_present"] for row in artifact["family_coverage_rows"]) == 2
    assert artifact["gate_check_summary"]["retirement"]["activated"] is False
    assert artifact["gate_check_summary"]["retirement"]["same_missing_chain_as_exp6577"] is False
    assert exp.validate_artifact(artifact) == []

    (tmp_path / qwen["artifact_path"]).unlink()
    missing = _build(tmp_path)
    first = missing["gate_check_summary"]["first_failure"]
    assert first == {
        "field": qwen["artifact_path"],
        "expected": "file_exists",
        "observed": "missing",
        "passed": False,
    }


@pytest.mark.parametrize(
    ("attack_id", "mutation"),
    [
        ("legacy_substitution", "legacy"),
        ("source_alias", "source"),
        ("prompt_drift", "prompt"),
        ("seed_drift", "seed"),
        ("duplicate_unit_id", "duplicate"),
        ("copied_output_across_families", "copied"),
        ("selective_retry", "retry"),
        ("hidden_row_drop", "drop"),
        ("null_only_rows", "null"),
        ("stale_pid", "pid"),
        ("zero_layer_offload", "offload"),
        ("missing_raw_path", "raw_path"),
        ("missing_unload", "unload"),
        ("reused_process", "process_reuse"),
        ("protected_drift", "protected"),
        ("readiness_contradicted_by_rows", "readiness"),
    ],
)
def test_every_attack_mutation_fails_closed(tmp_path: Path, attack_id: str, mutation: str) -> None:
    """SCENARIO-REPORT-6584-ATTACKS: each named invariant mutation closes."""

    _, payloads = _repo(tmp_path)
    family = exp.FAMILY_SPECS[0]
    payload = payloads[family["family_id"]]
    protected_before = None
    if mutation == "legacy":
        payload["rows"][0]["repository_id"] = "Qwen/Qwen3.5-0.8B"
    elif mutation == "source":
        payload["rows"][0]["source_bytes_b64"] = base64.b64encode(b"alias").decode()
    elif mutation == "prompt":
        payload["rows"][0]["prompt_sha256"] = _sha_bytes(b"drift")
    elif mutation == "seed":
        payload["rows"][0]["seed"] = -1
    elif mutation == "duplicate":
        payload["rows"][1]["unit_id"] = payload["rows"][0]["unit_id"]
    elif mutation == "copied":
        other = exp.FAMILY_SPECS[1]
        other_payload = payloads[other["family_id"]]
        other_payload["rows"][0]["raw_response_sha256"] = payload["rows"][0]["raw_response_sha256"]
        _rewrite(tmp_path, other, other_payload)
    elif mutation == "retry":
        payload["rows"][0]["attempt_count"] = 2
    elif mutation == "drop":
        payload["rows"].pop()
    elif mutation == "null":
        for row in payload["rows"]:
            response = b"Please provide the claim"
            row["raw_response_bytes_b64"] = base64.b64encode(response).decode()
            row["raw_response_sha256"] = _sha_bytes(response)
    elif mutation == "pid":
        payload["rows"][0]["pid"] += 99
    elif mutation == "offload":
        payload["process_and_gpu_receipts"]["offloaded_layers"] = 0
        for row in payload["rows"]:
            row["offloaded_layers"] = 0
    elif mutation == "raw_path":
        Path(payload["rows"][0]["raw_checkpoint_path"]).unlink()
    elif mutation == "unload":
        payload["unload_and_recovery_rows"] = []
    elif mutation == "process_reuse":
        other = exp.FAMILY_SPECS[1]
        other_payload = payloads[other["family_id"]]
        reused = payload["process_and_gpu_receipts"]["pid"]
        other_payload["process_and_gpu_receipts"]["pid"] = reused
        other_payload["unload_and_recovery_rows"][0]["worker_pid"] = reused
        for row in other_payload["rows"]:
            row["pid"] = reused
        _rewrite(tmp_path, other, other_payload)
    elif mutation == "protected":
        protected_before = exp.protected_hashes(tmp_path)
        (tmp_path / exp.PROTECTED_RELATIVE_PATHS[0]).write_text("drift\n", encoding="utf-8")
    elif mutation == "readiness":
        payload[family["readiness_field"]] = 0.0
    else:  # pragma: no cover - the parameter table owns every branch.
        raise AssertionError(mutation)
    _rewrite(tmp_path, family, payload)
    artifact = _build(tmp_path, protected_before=protected_before)
    attack = next(
        row
        for row in artifact["duplicate_drift_and_substitution_rows"]
        if row["attack_id"] == attack_id
    )
    assert attack["passed"] is False
    assert artifact["all_family_source_audit_ready_score"] == 0.0


def test_failure_retention_recomputes_raw_classes_and_keeps_denominator(tmp_path: Path) -> None:
    """REQ-REPORT-6584-FAILURES: no-claim and malformed rows stay visible."""

    protocol, _ = _repo(tmp_path)
    dense = exp.FAMILY_SPECS[1]
    moe = exp.FAMILY_SPECS[2]
    _write_family(
        tmp_path,
        protocol,
        dense,
        pid=2001,
        responses=[b"Please provide the claim", b"Please provide the claim"],
        stored_ready=0.0,
    )
    fenced = [
        b'```json\n{"claim_id":"x","supported_spans":[],"unsupported_reason":null,"release_action":"release"}\n```'
    ] * 2
    _write_family(tmp_path, protocol, moe, pid=2002, responses=fenced)
    artifact = _build(tmp_path)
    counts = artifact["aggregate_row_recomputation"]["failure_class_counts"]
    assert counts["no_claim"] == 2
    assert counts["malformed_output"] == 2
    assert artifact["aggregate_row_recomputation"]["failure_row_count"] == 4
    assert artifact["aggregate_row_recomputation"]["observed_row_count"] == 6
    assert all(row["retained"] for row in artifact["failure_retention_rows"])


def test_validator_atomic_writer_and_exact_required_fields(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6584-ATOMIC: validation precedes same-directory replace."""

    _repo(tmp_path)
    artifact = _build(tmp_path)
    target = tmp_path / "terminal.json"
    receipt = exp.atomic_write_json(target, artifact)
    assert receipt["atomic_replace"] is True
    assert receipt["sha256"] == exp.sha256_file(target)
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == artifact

    for mutation in (
        lambda value: value.pop("rows"),
        lambda value: value.__setitem__("inference_substrate", "live_llm_inference"),
        lambda value: value.__setitem__("verifier_is_oracle", False),
        lambda value: value.__setitem__("verdict_class", "positive"),
        lambda value: value.__setitem__("all_family_source_audit_ready_score", 0.0),
        lambda value: value.__setitem__("reproducibility_checksum", "sha256:bad"),
    ):
        bad = deepcopy(artifact)
        mutation(bad)
        assert exp.validate_artifact(bad)
    with pytest.raises(ValueError, match="artifact validation failed"):
        exp.atomic_write_json(target, {"bad": True})


def test_blocked_real_shape_has_required_principles_and_no_positive_science(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6584-REDUCER: blocked output remains complete and non-positive."""

    _, payloads = _repo(tmp_path)
    qwen = exp.FAMILY_SPECS[0]
    payload = payloads[qwen["family_id"]]
    payload["status"] = "blocked"
    payload["rows"] = []
    payload["checkpoint_receipts"] = []
    payload["parser_diagnostic_rows"] = []
    payload["raw_response_receipts"] = []
    payload[qwen["readiness_field"]] = 0.0
    _rewrite(tmp_path, qwen, payload)
    artifact = _build(tmp_path)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(artifact["field_provenance"])
    assert artifact["inference_substrate"] == "immutable_three_family_source_replay_no_llm"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["preconditions_checked"]["llm_calls_issued"] == 0
    assert artifact["preconditions_checked"]["model_inference_invoked"] is False
    assert artifact["status"].startswith("blocked_")
    assert "lineage=" in artifact["honest_verdict"]
    assert "coverage=" in artifact["honest_verdict"]
    assert "failure_retention=" in artifact["honest_verdict"]
    assert "unload=" in artifact["honest_verdict"]
    assert "merge=" in artifact["honest_verdict"]


def test_error_receipts_and_validator_branches_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-6584-ATOMIC: malformed evidence cannot evade validation."""

    assert exp.classify_response(b"```json\n{broken}\n```")["malformed_output"] is True
    assert exp.classify_response(b"\xff")["malformed_output"] is True

    broken_json = tmp_path / "broken.json"
    broken_json.write_text("{", encoding="utf-8")
    assert exp._read_json(broken_json)[1] == "unreadable:JSONDecodeError"
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp._read_json(list_json)[1] == "schema_not_object"

    _, payloads = _repo(tmp_path)
    family = exp.FAMILY_SPECS[0]

    def broken_inspector(_path: Path) -> dict[str, Any]:
        raise ValueError("bad metadata")

    metadata = exp._metadata_receipt(payloads[family["family_id"]], family, broken_inspector)
    assert metadata["passed"] is False
    assert metadata["inspect_error"] == "ValueError:bad metadata"
    raw_paths = exp._raw_path_preconditions({family["family_id"]: {"rows": [None]}})
    assert raw_paths == []

    artifact = _build(tmp_path)
    bad_prefix = deepcopy(artifact)
    bad_prefix["verdict_class"] = "blocked"
    bad_prefix["honest_verdict"] = "not_blocked"
    assert "blocked_verdict_prefix_missing" in exp.validate_artifact(bad_prefix)

    bad_principle = deepcopy(artifact)
    bad_principle["field_provenance"]["status"]["principle"] = "wrong"
    assert "field_principle_mismatch:status" in exp.validate_artifact(bad_principle)

    bad_container = deepcopy(artifact)
    bad_container["family_coverage_rows"] = {}
    assert "family_coverage_rows_not_list" in exp.validate_artifact(bad_container)

    bad_row = deepcopy(artifact)
    bad_row["failure_retention_rows"][0]["row_hash"] = "sha256:bad"
    assert "failure_retention_rows_row_hash:0" in exp.validate_artifact(bad_row)
