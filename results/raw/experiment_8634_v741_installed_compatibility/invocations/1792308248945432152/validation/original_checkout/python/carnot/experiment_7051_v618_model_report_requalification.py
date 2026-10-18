"""Requalify one owned CUDA llama.cpp model-report capture.

The artifact keeps server observations separate from derived filesystem facts.
It does not call the ARC identity validator or open an ARC game.

Spec: REQ-ARC-7051 and SCENARIO-ARC-7051-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any

from carnot.experiment_7039_v617_model_report_forensics import (
    analyze_identity_report,
    extract_quantization,
    gate_check_summary,
    gate_row,
    sha256_file,
    sha256_json,
)
from carnot.inference.sota_models import cached_sota_pair
from carnot.terminal_artifacts import payload_sha256


JsonDict = dict[str, Any]
EXPERIMENT_ID = 7051
SCHEMA = "carnot.exp7051.v618_model_report_requalification.v1"
RUN_DATE = "20260906"
RANDOM_SEED = 7_051_202_609_06
INFERENCE_SUBSTRATE = "live_llm_inference"
MANDATED_MODEL_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
MANDATED_MODEL_NAME = "Qwen3.6-35B-A3B"
MODEL_N_CTX = 4096
LIVE_INFERENCE_DURATION_FLOOR_S = 60.0
MINIMUM_OWNED_LIVE_INTERVAL_S = 75.0
DIAGNOSTIC_PROMPTS = (
    "/no_think\nReply with the single token OK.",
    "/no_think\nReply with the single token READY.",
)
IDENTITY_FIELDS = ("model_path", "model", "model_alias")
REPORT_CLASSIFICATIONS = frozenset(
    {"snapshot_alias", "resolved_blob", "direct_file", "conflicting", "unknown"}
)
EVIDENCE_STATUSES = frozenset({"supported", "contradicted", "unknown"})
CHECKSUM_EXCLUDED_FIELDS = (
    "reproducibility_checksum",
    "checksum_recomputation_rows",
)
CHECKSUM_CONTRACT = {
    "helper": "carnot.terminal_artifacts.payload_sha256",
    "canonical_json": "sort_keys_compact_ascii",
    "excluded_fields": list(CHECKSUM_EXCLUDED_FIELDS),
}

RESULT_RELATIVE_PATH = Path("results/experiment_7051_v618_model_report_requalification.json")
CHECKPOINT_RELATIVE_PATH = Path("results/checkpoints/experiment_7051/checkpoint.json")
EXP7039_RELATIVE_PATH = Path("results/experiment_7039_v617_model_report_forensics.json")
EXP7040_RELATIVE_PATH = Path("results/experiment_7040_v617_typed_identity_bridge.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_7051_v618_model_report_requalification.py")
SCRIPT_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_7051_v618_model_report_requalification.py"
)
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_7051_v618_model_report_requalification.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-agi/spec.md")
REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "MODEL_SPECS",
    "model_specs",
    "models_used",
    "selected_model_spec",
    "requested_model_path",
    "requested_model_filename",
    "requested_hf_id",
    "requested_revision",
    "launch_model_argument",
    "process_command_rows",
    "raw_server_props",
    "raw_identity_field_rows",
    "resolved_identity_field_rows",
    "file_hash_rows",
    "consistency_rows",
    "diagnostic_request_rows",
    "server_process_rows",
    "port_lease_rows",
    "gpu_lease_rows",
    "gpu_sample_rows",
    "phase_clock_rows",
    "cuda_layer_offload_confirmed",
    "owned_live_interval_s",
    "minimum_owned_live_interval_s",
    "cleanup_rows",
    "generation_request_count",
    "arc_action_count",
    "game_level_solve_claim",
    "checksum_contract",
    "checksum_recomputation_rows",
    "model_report_evidence_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why it is part of the evidence contract.",
    "preconditions_checked": "Explicit checks prevent missing resources from becoming synthetic evidence.",
    "inference_substrate": "The declared substrate distinguishes live model generation from offline analysis.",
    "duration_s": "Measured total time exposes an implausibly short live-model claim.",
    "source_artifact_hashes": "Source hashes reveal code or evidence drift after this capture.",
    "rows": "Summary rows make each terminal evidence gate independently countable.",
    "MODEL_SPECS": "The uppercase model list preserves the experiment selection contract.",
    "model_specs": "The lowercase model list supports repository methodology checks.",
    "models_used": "The invoked hub list prevents a fallback model from satisfying the task.",
    "selected_model_spec": "The selected specification binds model, snapshot, GPU, hash, and build.",
    "requested_model_path": "The requested path preserves the exact snapshot launch intent.",
    "requested_model_filename": "The filename records the selected GGUF before path resolution.",
    "requested_hf_id": "The hub ID prevents a same-named file from another repository from passing.",
    "requested_revision": "The revision binds the request to one immutable snapshot.",
    "launch_model_argument": "The raw launch argument records what the server command received.",
    "process_command_rows": "The process command independently confirms the launch argument and owner.",
    "raw_server_props": "The complete server response remains available before interpretation.",
    "raw_identity_field_rows": "Separate raw candidates expose missing and conflicting reports.",
    "resolved_identity_field_rows": "Derived rows resolve paths without replacing raw observations.",
    "file_hash_rows": "Content hashes compare every reachable model file with the selected bytes.",
    "consistency_rows": "Typed states keep unknown or conflicting facts from becoming support.",
    "diagnostic_request_rows": "Fixed responses prove that the loaded model generated genuine tokens.",
    "server_process_rows": "Server evidence binds process, port, build, model, and CUDA intent.",
    "port_lease_rows": "A port lease proves exclusive endpoint ownership through cleanup.",
    "gpu_lease_rows": "A GPU lease proves task authority over the selected device.",
    "gpu_sample_rows": "A process sample proves that the owned server used the leased GPU.",
    "phase_clock_rows": "Monotonic events prove token ordering and owned shutdown duration.",
    "cuda_layer_offload_confirmed": "True rules out CPU inference and zero-layer fallback.",
    "owned_live_interval_s": "The owned interval cannot be inflated by pre-launch model hashing.",
    "minimum_owned_live_interval_s": "A fixed 75-second floor repairs the short Exp7039 evidence.",
    "cleanup_rows": "Cleanup receipts prove that only owned resources were stopped and released.",
    "generation_request_count": "An exact count prevents skipped or repeated diagnostics.",
    "arc_action_count": "Zero proves the report capture did not act in an ARC game.",
    "game_level_solve_claim": "False prevents infrastructure evidence from becoming solve credit.",
    "checksum_contract": "The named helper and exclusions make checksum reproduction unambiguous.",
    "checksum_recomputation_rows": "A clean-reader receipt detects a stale terminal checksum.",
    "model_report_evidence_ready_score": "One means evidence validity, not model quality or belief value.",
    "random_seed": "Fixed seeds make the diagnostic requests reproducible.",
    "reproducibility_checksum": "A canonical terminal digest detects later included-field changes.",
    "gate_check_summary": "The first exact failure makes a blocked run actionable.",
    "verifier_is_oracle": "False states that this capture does not define ARC correctness.",
    "verdict_class": "A closed class gives the terminal state one machine-readable meaning.",
    "honest_verdict": "A class-consistent prefix prevents a block from reading as success.",
}

VERDICT_PREFIXES = {
    "positive": "complete_positive_",
    "circular_positive": "complete_circular_positive_",
    "null": "complete_null_",
    "blocked": "blocked_",
    "disqualified": "disqualified_",
    "partial": "partial_",
}


def _snapshot_revision(path: Path, hf_id: str) -> str | None:
    """Read a revision only from an exact Hugging Face snapshot path."""

    expected_root = "models--" + hf_id.replace("/", "--")
    if path.parent.parent.name != "snapshots" or path.parent.parent.parent.name != expected_root:
        return None
    return path.parent.name or None


def resolve_model_spec(
    cached_pair_fn: Callable[..., list[dict] | None] = cached_sota_pair,
    *,
    gpu_index: int,
    gpu_uuid: str,
    gpu_model: str,
    server_build: str,
) -> JsonDict | None:
    """Select the mandated Qwen snapshot only through ``cached_sota_pair``."""

    pair = cached_pair_fn(gpu_indices=(int(gpu_index), int(gpu_index)))
    if not pair:
        return None
    selected = next(
        (dict(row) for row in pair if row.get("hf_id") == MANDATED_MODEL_HF_ID),
        None,
    )
    if selected is None:
        return None
    path = Path(str(selected.get("model_path") or ""))
    if not path.is_file() or path.suffix.lower() != ".gguf":
        return None
    revision = _snapshot_revision(path.absolute(), MANDATED_MODEL_HF_ID)
    if revision is None:
        return None
    selected.update(
        {
            "name": MANDATED_MODEL_NAME,
            "hf_id": MANDATED_MODEL_HF_ID,
            "gpu": int(gpu_index),
            "gpu_uuid": str(gpu_uuid),
            "gpu_model": str(gpu_model),
            "model_path": str(path.absolute()),
            "model_filename": path.name,
            "revision": revision,
            "quantization": extract_quantization(path.name),
            "model_file_hash": sha256_file(path),
            "server_build": str(server_build),
            "resolved_via": "cached_sota_pair",
        }
    )
    return selected


def _checksum_projection(artifact: Mapping[str, Any]) -> JsonDict:
    """Remove only fields declared as self-referential checksum receipts."""

    return {
        key: deepcopy(value)
        for key, value in artifact.items()
        if key not in CHECKSUM_EXCLUDED_FIELDS
    }


def canonical_artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Use the repository canonical JSON helper on the declared projection."""

    return payload_sha256(_checksum_projection(artifact))


def clean_reader_checksum(raw_json: str) -> str:
    """Parse new JSON bytes before recomputing the canonical checksum."""

    value = json.loads(raw_json)
    if not isinstance(value, dict):
        raise ValueError("terminal artifact must be a JSON object")
    return canonical_artifact_checksum(value)


def _threshold_gate(check: str, minimum: float, observed: Any) -> JsonDict:
    """Record a numeric floor without replacing the exact observation."""

    passed = (
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and float(observed) >= minimum
    )
    return {
        "check": check,
        "expected_value": f">={minimum:.1f}",
        "observed_value": observed,
        "passed": passed,
        "terminal": True,
    }


def _empty_live_evidence() -> JsonDict:
    """Create every live field before a process or lease exists."""

    return {
        "launch_model_argument": None,
        "process_command_rows": [],
        "raw_server_props": {},
        "diagnostic_request_rows": [],
        "server_process_rows": [],
        "port_lease_rows": [],
        "gpu_lease_rows": [],
        "gpu_sample_rows": [],
        "phase_clock_rows": [],
        "owned_live_interval_s": 0.0,
        "cleanup_rows": [],
    }


def _status(left: Any, right: Any) -> str:
    if left is None or right is None:
        return "unknown"
    return "supported" if left == right else "contradicted"


def _base_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, str],
    model_specs: Sequence[Mapping[str, Any]],
    selected_model_spec: Mapping[str, Any] | None,
    live_evidence: Mapping[str, Any] | None,
) -> JsonDict:
    """Project raw and derived evidence into one complete terminal schema."""

    evidence = {**_empty_live_evidence(), **deepcopy(dict(live_evidence or {}))}
    selected = deepcopy(dict(selected_model_spec)) if selected_model_spec is not None else None
    report = (
        analyze_identity_report(evidence["raw_server_props"], selected)
        if selected is not None
        else {
            "raw_server_props": deepcopy(evidence["raw_server_props"]),
            "raw_identity_field_rows": [],
            "resolved_identity_field_rows": [],
            "file_hash_rows": [],
            "consistency_rows": [],
            "report_channel_classification": "unknown",
        }
    )
    process_rows = deepcopy(list(evidence["process_command_rows"]))
    process_argument = process_rows[0].get("model_argument") if process_rows else None
    requested_path = selected.get("model_path") if selected else None
    launch_argument = evidence["launch_model_argument"]
    consistency_rows = deepcopy(report["consistency_rows"])
    consistency_rows.extend(
        [
            {
                "check": "launch_argument_matches_requested_path",
                "status": _status(launch_argument, requested_path),
                "expected_value": requested_path,
                "observed_value": launch_argument,
                "terminal": True,
            },
            {
                "check": "process_argument_matches_launch_argument",
                "status": _status(process_argument, launch_argument),
                "expected_value": launch_argument,
                "observed_value": process_argument,
                "terminal": True,
            },
        ]
    )
    file_rows = deepcopy(report["file_hash_rows"])
    if selected is not None and requested_path:
        try:
            observed_hash = sha256_file(requested_path)
            resolved_path = str(Path(requested_path).resolve(strict=True))
        except OSError:
            observed_hash = None
            resolved_path = None
        expected_hash = selected.get("model_file_hash")
        matches = observed_hash is not None and observed_hash == expected_hash
        file_rows.insert(
            0,
            {
                "source_field": "selected_model_spec.model_path",
                "raw_path": requested_path,
                "resolved_path": resolved_path,
                "sha256": observed_hash,
                "selected_model_hash": expected_hash,
                "hash_matches_selected": matches,
                "status": "supported" if matches else "unknown",
                "terminal": True,
            },
        )
    diagnostics = deepcopy(list(evidence["diagnostic_request_rows"]))
    server_rows = deepcopy(list(evidence["server_process_rows"]))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": str(run_date),
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "rows": [],
        "MODEL_SPECS": deepcopy([dict(row) for row in model_specs]),
        "model_specs": deepcopy([dict(row) for row in model_specs]),
        "models_used": [selected["hf_id"]] if selected and diagnostics else [],
        "selected_model_spec": selected,
        "requested_model_path": requested_path,
        "requested_model_filename": selected.get("model_filename") if selected else None,
        "requested_hf_id": selected.get("hf_id") if selected else None,
        "requested_revision": selected.get("revision") if selected else None,
        "launch_model_argument": launch_argument,
        "process_command_rows": process_rows,
        "raw_server_props": deepcopy(report["raw_server_props"]),
        "raw_identity_field_rows": deepcopy(report["raw_identity_field_rows"]),
        "resolved_identity_field_rows": deepcopy(report["resolved_identity_field_rows"]),
        "file_hash_rows": file_rows,
        "consistency_rows": consistency_rows,
        "report_channel_classification": report["report_channel_classification"],
        "diagnostic_request_rows": diagnostics,
        "server_process_rows": server_rows,
        "port_lease_rows": deepcopy(list(evidence["port_lease_rows"])),
        "gpu_lease_rows": deepcopy(list(evidence["gpu_lease_rows"])),
        "gpu_sample_rows": deepcopy(list(evidence["gpu_sample_rows"])),
        "phase_clock_rows": deepcopy(list(evidence["phase_clock_rows"])),
        "cuda_layer_offload_confirmed": bool(
            server_rows and server_rows[0].get("cuda_layer_offload_confirmed") is True
        ),
        "owned_live_interval_s": float(evidence["owned_live_interval_s"]),
        "minimum_owned_live_interval_s": MINIMUM_OWNED_LIVE_INTERVAL_S,
        "cleanup_rows": deepcopy(list(evidence["cleanup_rows"])),
        "generation_request_count": len(diagnostics),
        "arc_action_count": 0,
        "game_level_solve_claim": False,
        "checksum_contract": deepcopy(CHECKSUM_CONTRACT),
        "checksum_recomputation_rows": [],
        "model_report_evidence_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_model_report_evidence:unclassified",
    }
    return artifact


def _phase_clocks_valid(artifact: Mapping[str, Any]) -> bool:
    rows = artifact.get("phase_clock_rows")
    if not isinstance(rows, list) or [row.get("event") for row in rows] != [
        "owned_live_start",
        "first_token",
        "last_token",
        "shutdown",
    ]:
        return False
    timestamps = [row.get("monotonic_ns") for row in rows]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in timestamps):
        return False
    if timestamps != sorted(timestamps):
        return False
    measured = (timestamps[-1] - timestamps[0]) / 1_000_000_000
    interval = artifact.get("owned_live_interval_s")
    return (
        isinstance(interval, (int, float))
        and not isinstance(interval, bool)
        and abs(float(interval) - measured) <= 0.001
    )


def _positive_gate_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Reduce every ownership and capture requirement to an exact check."""

    diagnostics = artifact.get("diagnostic_request_rows")
    diagnostics = diagnostics if isinstance(diagnostics, list) else []
    process_rows = artifact.get("process_command_rows")
    process_rows = process_rows if isinstance(process_rows, list) else []
    servers = artifact.get("server_process_rows")
    servers = servers if isinstance(servers, list) else []
    ports = artifact.get("port_lease_rows")
    ports = ports if isinstance(ports, list) else []
    leases = artifact.get("gpu_lease_rows")
    leases = leases if isinstance(leases, list) else []
    samples = artifact.get("gpu_sample_rows")
    samples = samples if isinstance(samples, list) else []
    cleanup = artifact.get("cleanup_rows")
    cleanup = cleanup if isinstance(cleanup, list) else []
    selected_rows = [
        row
        for row in artifact.get("file_hash_rows", [])
        if isinstance(row, Mapping) and row.get("source_field") == "selected_model_spec.model_path"
    ]
    reachable_rows = [
        row
        for row in artifact.get("resolved_identity_field_rows", [])
        if isinstance(row, Mapping) and row.get("reachable_file") is True
    ]
    return [
        _threshold_gate(
            "live_inference_duration_floor_s",
            LIVE_INFERENCE_DURATION_FLOOR_S,
            artifact.get("duration_s"),
        ),
        _threshold_gate(
            "owned_live_interval_floor_s",
            MINIMUM_OWNED_LIVE_INTERVAL_S,
            artifact.get("owned_live_interval_s"),
        ),
        gate_row(
            "launch_model_argument",
            artifact.get("requested_model_path"),
            artifact.get("launch_model_argument"),
        ),
        gate_row(
            "owned_process_command",
            True,
            len(process_rows) == 1 and process_rows[0].get("owned") is True,
        ),
        gate_row(
            "process_model_argument",
            artifact.get("launch_model_argument"),
            process_rows[0].get("model_argument") if process_rows else None,
        ),
        gate_row(
            "raw_props_captured",
            True,
            isinstance(artifact.get("raw_server_props"), Mapping)
            and bool(artifact.get("raw_server_props")),
        ),
        gate_row(
            "raw_identity_fields_captured",
            list(IDENTITY_FIELDS),
            [row.get("field") for row in artifact.get("raw_identity_field_rows", [])],
        ),
        gate_row(
            "reachable_identity_supported",
            True,
            bool(reachable_rows)
            and all(row.get("evidence_status") == "supported" for row in reachable_rows),
        ),
        gate_row(
            "selected_model_hash",
            True,
            len(selected_rows) == 1 and selected_rows[0].get("hash_matches_selected") is True,
        ),
        gate_row(
            "generation_request_count",
            len(DIAGNOSTIC_PROMPTS),
            artifact.get("generation_request_count"),
        ),
        gate_row(
            "diagnostic_requests_complete",
            True,
            len(diagnostics) == len(DIAGNOSTIC_PROMPTS)
            and all(
                row.get("completed") is True
                and isinstance(row.get("generated_tokens"), int)
                and row.get("generated_tokens") >= 1
                for row in diagnostics
            ),
        ),
        gate_row("phase_clock_order", True, _phase_clocks_valid(artifact)),
        gate_row(
            "owned_server",
            True,
            len(servers) == 1 and servers[0].get("owned") is True,
        ),
        gate_row(
            "owned_port_lease_released",
            True,
            len(ports) == 1 and ports[0].get("owned") is True and ports[0].get("released") is True,
        ),
        gate_row(
            "owned_gpu_lease_released",
            True,
            len(leases) == 1
            and leases[0].get("owned") is True
            and leases[0].get("released") is True,
        ),
        gate_row(
            "gpu_process_sample",
            True,
            len(samples) == 1
            and float(samples[0].get("pid_memory_mb", 0)) > 0
            and str(samples[0].get("gpu_uuid", "")).startswith("GPU-")
            and "RTX 3090" in str(samples[0].get("gpu_model", "")),
        ),
        gate_row("cuda_layer_offload", True, artifact.get("cuda_layer_offload_confirmed") is True),
        gate_row(
            "safe_cleanup",
            True,
            len(cleanup) == 1 and cleanup[0].get("passed") is True,
        ),
        gate_row("arc_action_count", 0, artifact.get("arc_action_count")),
        gate_row("game_level_solve_claim", False, artifact.get("game_level_solve_claim")),
    ]


def _set_terminal_state(artifact: JsonDict, checks: Sequence[Mapping[str, Any]]) -> None:
    """Set score and verdict only from terminal evidence checks."""

    summary = gate_check_summary(checks)
    ready = summary["passed"] is True
    artifact["gate_check_summary"] = summary
    artifact["model_report_evidence_ready_score"] = int(ready)
    artifact["verdict_class"] = "positive" if ready else "blocked"
    artifact["honest_verdict"] = (
        "complete_positive_model_report_evidence_requalified"
        if ready
        else f"blocked_model_report_evidence:{summary['failed_check']}"
    )


def _finish_checksum(artifact: JsonDict, checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Hash a complete candidate and confirm it through a new JSON reader."""

    checksum_check = gate_row("clean_reader_reproducibility_checksum", True, True)
    terminal_checks = [deepcopy(dict(row)) for row in checks] + [checksum_check]
    _set_terminal_state(artifact, terminal_checks)
    artifact["reproducibility_checksum"] = canonical_artifact_checksum(artifact)
    observed = clean_reader_checksum(json.dumps(artifact, sort_keys=True))
    passed = observed == artifact["reproducibility_checksum"]
    if not passed:  # pragma: no cover - the same deterministic helper runs on both sides
        terminal_checks[-1] = gate_row(
            "clean_reader_reproducibility_checksum",
            artifact["reproducibility_checksum"],
            observed,
        )
        _set_terminal_state(artifact, terminal_checks)
        artifact["reproducibility_checksum"] = canonical_artifact_checksum(artifact)
        observed = clean_reader_checksum(json.dumps(artifact, sort_keys=True))
    artifact["checksum_recomputation_rows"] = [
        {
            "reader": "clean_json_reader",
            "helper": CHECKSUM_CONTRACT["helper"],
            "expected_checksum": artifact["reproducibility_checksum"],
            "observed_checksum": observed,
            "passed": observed == artifact["reproducibility_checksum"],
            "terminal": True,
        }
    ]
    return artifact


def build_blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, str],
    model_specs: Sequence[Mapping[str, Any]],
    selected_model_spec: Mapping[str, Any] | None,
    live_evidence: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build a complete blocker that retains the first exact failed gate."""

    artifact = _base_artifact(
        run_date=run_date,
        duration_s=duration_s,
        preconditions=preconditions,
        source_artifact_hashes=source_artifact_hashes,
        model_specs=model_specs,
        selected_model_spec=selected_model_spec,
        live_evidence=live_evidence,
    )
    summary = gate_check_summary(preconditions)
    failed = summary["failed_check"] or "unknown_failure"
    artifact["rows"] = [
        {
            "check": "model_report_evidence_requalification",
            "passed": False,
            "failed_check": failed,
            "terminal": True,
        }
    ]
    checks = list(preconditions)
    if summary["passed"] is True:
        checks.append(gate_row("blocked_live_capture", True, False))
    return _finish_checksum(artifact, checks)


def build_positive_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, str],
    model_specs: Sequence[Mapping[str, Any]],
    selected_model_spec: Mapping[str, Any],
    live_evidence: Mapping[str, Any],
) -> JsonDict:
    """Build evidence readiness from one owned capture and its cleanup."""

    artifact = _base_artifact(
        run_date=run_date,
        duration_s=duration_s,
        preconditions=preconditions,
        source_artifact_hashes=source_artifact_hashes,
        model_specs=model_specs,
        selected_model_spec=selected_model_spec,
        live_evidence=live_evidence,
    )
    checks = [*list(preconditions), *_positive_gate_rows(artifact)]
    artifact["rows"] = [
        {
            "check": "raw_identity_fields",
            "row_count": len(artifact["raw_identity_field_rows"]),
            "passed": len(artifact["raw_identity_field_rows"]) == len(IDENTITY_FIELDS),
            "terminal": True,
        },
        {
            "check": "diagnostic_requests",
            "row_count": len(artifact["diagnostic_request_rows"]),
            "passed": len(artifact["diagnostic_request_rows"]) == len(DIAGNOSTIC_PROMPTS),
            "terminal": True,
        },
        {
            "check": "safe_cleanup",
            "row_count": len(artifact["cleanup_rows"]),
            "passed": bool(
                artifact["cleanup_rows"] and artifact["cleanup_rows"][0].get("passed") is True
            ),
            "terminal": True,
        },
    ]
    return _finish_checksum(artifact, checks)


def _checksum_receipt_valid(artifact: Mapping[str, Any]) -> bool:
    rows = artifact.get("checksum_recomputation_rows")
    checksum = artifact.get("reproducibility_checksum")
    return (
        isinstance(rows, list)
        and len(rows) == 1
        and rows[0]
        == {
            "reader": "clean_json_reader",
            "helper": CHECKSUM_CONTRACT["helper"],
            "expected_checksum": checksum,
            "observed_checksum": checksum,
            "passed": True,
            "terminal": True,
        }
    )


def validate_artifact(artifact: Any) -> list[str]:
    """Recompute terminal evidence without validating production ARC identity."""

    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    errors: list[str] = []
    if missing := sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)):
        errors.append(f"required_fields_missing:{missing}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_invalid")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    if artifact.get("arc_action_count") != 0 or artifact.get("game_level_solve_claim") is not False:
        errors.append("arc_non_action_contract_invalid")
    if artifact.get("minimum_owned_live_interval_s") != MINIMUM_OWNED_LIVE_INTERVAL_S:
        errors.append("minimum_owned_live_interval_invalid")
    if artifact.get("checksum_contract") != CHECKSUM_CONTRACT:
        errors.append("checksum_contract_invalid")
    if not _checksum_receipt_valid(artifact):
        errors.append("checksum_recomputation_rows_invalid")
    classification = artifact.get("report_channel_classification")
    if classification not in REPORT_CLASSIFICATIONS:
        errors.append("report_channel_classification_invalid")
    summary = artifact.get("gate_check_summary")
    summary_valid = isinstance(summary, Mapping) and {
        "passed",
        "failed_check",
        "expected_value",
        "observed_value",
        "checks",
    } <= set(summary)
    if not summary_valid:
        errors.append("gate_check_summary_invalid")
    verdict_class = str(artifact.get("verdict_class", ""))
    prefix = VERDICT_PREFIXES.get(verdict_class)
    verdict = artifact.get("honest_verdict")
    if prefix is None or not isinstance(verdict, str) or not verdict.startswith(prefix):
        errors.append("verdict_prefix_invalid")
    score = artifact.get("model_report_evidence_ready_score")
    if type(score) is not int or score not in (0, 1):
        errors.append("ready_score_invalid")

    if verdict_class == "positive":
        raw_props = artifact.get("raw_server_props")
        raw_rows = artifact.get("raw_identity_field_rows")
        resolved_rows = artifact.get("resolved_identity_field_rows")
        if not isinstance(raw_props, Mapping) or not raw_props:
            errors.append("raw_server_props_invalid")
        if not isinstance(raw_rows, list) or [row.get("field") for row in raw_rows] != list(
            IDENTITY_FIELDS
        ):
            errors.append("raw_identity_rows_invalid")
        elif isinstance(raw_props, Mapping) and any(
            row.get("raw_value") != raw_props.get(row["field"]) for row in raw_rows
        ):
            errors.append("raw_identity_rows_do_not_match_props")
        if not isinstance(resolved_rows, list) or [
            row.get("field") for row in resolved_rows
        ] != list(IDENTITY_FIELDS):
            errors.append("resolved_identity_rows_invalid")
        elif (
            isinstance(raw_rows, list)
            and len(raw_rows) == len(resolved_rows)
            and any(
                row.get("raw_value") != raw_rows[index].get("raw_value")
                for index, row in enumerate(resolved_rows)
            )
        ):
            errors.append("resolved_rows_do_not_retain_raw_values")
        status_rows = [
            *list(raw_rows if isinstance(raw_rows, list) else []),
            *list(resolved_rows if isinstance(resolved_rows, list) else []),
        ]
        if any(row.get("evidence_status") not in EVIDENCE_STATUSES for row in status_rows):
            errors.append("identity_evidence_status_invalid")
        file_rows = artifact.get("file_hash_rows")
        if (
            not isinstance(file_rows, list)
            or not file_rows
            or any(row.get("status") not in EVIDENCE_STATUSES for row in file_rows)
        ):
            errors.append("file_hash_rows_invalid")
        consistency = artifact.get("consistency_rows")
        if (
            not isinstance(consistency, list)
            or len(consistency) < 4
            or any(row.get("status") not in EVIDENCE_STATUSES for row in consistency)
        ):
            errors.append("consistency_rows_invalid")
        specs = artifact.get("MODEL_SPECS")
        selected = artifact.get("selected_model_spec")
        if (
            not isinstance(specs, list)
            or len(specs) != 1
            or artifact.get("model_specs") != specs
            or selected != specs[0]
        ):
            errors.append("model_specs_invalid")
        elif (
            selected.get("hf_id") != MANDATED_MODEL_HF_ID
            or selected.get("resolved_via") != "cached_sota_pair"
            or not str(selected.get("gpu_uuid", "")).startswith("GPU-")
            or not selected.get("server_build")
        ):
            errors.append("selected_model_spec_invalid")
        if not _phase_clocks_valid(artifact):
            errors.append("phase_clock_rows_invalid")
        live_errors = [
            row["check"] for row in _positive_gate_rows(artifact) if row["passed"] is not True
        ]
        errors.extend(f"positive_evidence_invalid:{name}" for name in live_errors)
        if score != 1 or not summary_valid or summary.get("passed") is not True:
            errors.append("positive_terminal_semantics_invalid")
    elif verdict_class == "blocked":
        if score != 0 or not summary_valid or summary.get("passed") is not False:
            errors.append("blocked_terminal_semantics_invalid")
        elif summary.get("failed_check") is None:
            errors.append("blocked_gate_failure_missing")
    else:
        errors.append("unsupported_terminal_class")

    checksum = artifact.get("reproducibility_checksum")
    if checksum != canonical_artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    else:
        try:
            recomputed = clean_reader_checksum(json.dumps(dict(artifact), sort_keys=True))
        except (TypeError, ValueError):
            recomputed = None
        if recomputed != checksum:
            errors.append("clean_reader_checksum_mismatch")
    return list(dict.fromkeys(errors))


def cleanup_owned_process(
    process: Any,
    *,
    owned_pid: int,
    terminate_timeout_s: float = 20.0,
) -> JsonDict:
    """Stop a process only when its current PID equals the recorded owner."""

    actual_pid = int(getattr(process, "pid", -1))
    ownership_match = actual_pid == int(owned_pid) and owned_pid > 0
    signals: list[str] = []
    if ownership_match and process.poll() is None:
        process.terminate()
        signals.append("terminate")
        try:
            process.wait(timeout=terminate_timeout_s)
        except subprocess.TimeoutExpired:
            process.kill()
            signals.append("kill")
            process.wait(timeout=terminate_timeout_s)
    exited = ownership_match and process.poll() is not None
    return {
        "owned_pid": int(owned_pid),
        "observed_pid": actual_pid,
        "ownership_match": ownership_match,
        "signals_sent_only_to_owned_pid": ownership_match,
        "signals_sent": signals,
        "process_exit_confirmed": exited,
        "process_reaped": exited,
        "terminal": True,
    }


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    """Validate, atomically publish, and re-read one terminal artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp7051 artifact: " + ";".join(errors))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(artifact), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        published = json.loads(temporary.read_text(encoding="utf-8"))
        published_errors = validate_artifact(published)
        if published_errors:
            raise ValueError("invalid Exp7051 published bytes: " + ";".join(published_errors))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> JsonDict:  # pragma: no cover - filesystem boundary
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _source_hashes(root: Path) -> dict[str, str]:  # pragma: no cover - filesystem boundary
    paths = (
        MODULE_RELATIVE_PATH,
        SCRIPT_RELATIVE_PATH,
        TEST_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
        EXP7039_RELATIVE_PATH,
        EXP7040_RELATIVE_PATH,
        Path("python/carnot/experiment_7039_v617_model_report_forensics.py"),
        Path("python/carnot/inference/sota_models.py"),
        Path("python/carnot/terminal_artifacts.py"),
        Path("scripts/experiment_template.py"),
    )
    return {
        relative.as_posix(): sha256_file(root / relative)
        for relative in paths
        if (root / relative).is_file()
    }


def _gpu_rows() -> list[JsonDict]:  # pragma: no cover - hardware boundary
    from carnot.agentic import arc_belief_shadow_live_trace as live_core

    return live_core._gpu_rows()


def _cuda_server_receipt() -> JsonDict:  # pragma: no cover - hardware boundary
    from carnot.agentic import arc_belief_shadow_live_trace as live_core

    return live_core._cuda_server_receipt()


def _unattributed_llama_server_rows() -> list[JsonDict]:  # pragma: no cover - process boundary
    rows: list[JsonDict] = []
    for proc_dir in sorted(Path("/proc").glob("[0-9]*"), key=lambda path: int(path.name)):
        try:
            comm = (proc_dir / "comm").read_text(encoding="utf-8").strip()
            if comm != "llama-server":
                continue
            raw = (proc_dir / "cmdline").read_bytes()
            argv = [part.decode("utf-8", "surrogateescape") for part in raw.split(b"\0") if part]
        except OSError:
            continue
        rows.append(
            {
                "pid": int(proc_dir.name),
                "comm": comm,
                "argv": argv,
                "command_line": " ".join(argv),
                "attributed_to_task": False,
                "terminal": True,
            }
        )
    return rows


def collect_preconditions(
    *, repo_root: Path, output_path: Path, checkpoint_path: Path
) -> JsonDict:  # pragma: no cover - live host boundary
    """Check every required resource before a lease or model launch."""

    checks: list[JsonDict] = []
    exp7039 = _load_json(repo_root / EXP7039_RELATIVE_PATH)
    exp7040 = _load_json(repo_root / EXP7040_RELATIVE_PATH)
    checks.extend(
        (
            gate_row("exp7039_evidence_readable", True, bool(exp7039)),
            gate_row("exp7040_evidence_readable", True, bool(exp7040)),
        )
    )
    foreign_servers = _unattributed_llama_server_rows()
    checks.append(gate_row("no_unattributed_llama_server", [], foreign_servers))
    gpu_candidates = [row for row in _gpu_rows() if row.get("idle") and row.get("supported")]
    gpu = gpu_candidates[0] if gpu_candidates else None
    checks.append(gate_row("idle_supported_rtx3090", True, gpu is not None))
    server = _cuda_server_receipt()
    can_resolve = bool(exp7039 and exp7040 and not foreign_servers and gpu is not None)
    spec = (
        resolve_model_spec(
            cached_sota_pair,
            gpu_index=int(gpu["index"]),
            gpu_uuid=str(gpu["gpu_uuid"]),
            gpu_model=str(gpu["gpu_model"]),
            server_build=str(server.get("version_output") or ""),
        )
        if can_resolve and gpu is not None
        else None
    )
    checks.append(gate_row("cached_mandated_gguf", True, spec is not None))
    if gpu is not None and spec is not None:
        required_mb = int(Path(spec["model_path"]).stat().st_size / (1024 * 1024)) + 2048
        available_mb = int(gpu["memory_free_mb"])
        checks.append(
            {
                "check": "adequate_free_vram_mb",
                "expected_value": f">={required_mb}",
                "observed_value": available_mb,
                "passed": available_mb >= required_mb,
                "terminal": True,
            }
        )
    else:
        checks.append(gate_row("adequate_free_vram_mb", "model_and_gpu_resolved", "unavailable"))
    checks.extend(
        (
            gate_row("cuda_llama_server_executable", True, server.get("exists") is True),
            gate_row("cuda_llama_server_library", True, server.get("cuda_enabled") is True),
            gate_row("cuda_llama_server_version", 0, server.get("version_returncode")),
            gate_row(
                "clean_stop_authority",
                True,
                callable(getattr(subprocess.Popen, "terminate", None)),
            ),
        )
    )
    for label, path in (
        ("result_path", output_path.parent),
        ("checkpoint_path", checkpoint_path.parent),
    ):
        path.mkdir(parents=True, exist_ok=True)
        checks.append(
            gate_row(f"writable_{label}", True, path.is_dir() and os.access(path, os.W_OK))
        )
    return {
        "checks": checks,
        "summary": gate_check_summary(checks),
        "gpu": gpu,
        "model_spec": spec,
        "server": server,
        "unattributed_server_rows": foreign_servers,
        "source_hashes": _source_hashes(repo_root),
    }


def _read_process_command(pid: int) -> JsonDict:  # pragma: no cover - process boundary
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    argv = [part.decode("utf-8", "surrogateescape") for part in raw.split(b"\0") if part]
    model_argument = None
    if "-m" in argv and argv.index("-m") + 1 < len(argv):
        model_argument = argv[argv.index("-m") + 1]
    return {
        "pid": int(pid),
        "argv": argv,
        "command_line": " ".join(argv),
        "model_argument": model_argument,
        "owned": True,
        "terminal": True,
    }


def _http_json(
    url: str, *, payload: Mapping[str, Any] | None = None
) -> JsonDict:  # pragma: no cover - HTTP boundary
    import urllib.request

    data = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(request, timeout=600 if data is not None else 10) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("server JSON response is not an object")
    return value


def _utc_now() -> str:  # pragma: no cover - live clock boundary
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _diagnostic_request(
    endpoint: str, prompt: str, request_index: int
) -> JsonDict:  # pragma: no cover - model boundary
    payload = {
        "prompt": prompt,
        "n_predict": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "seed": RANDOM_SEED + request_index - 1,
    }
    started_ns = time.monotonic_ns()
    started_wall = _utc_now()
    response = _http_json(endpoint + "/completion", payload=payload)
    finished_ns = time.monotonic_ns()
    finished_wall = _utc_now()
    timings = response.get("timings") if isinstance(response.get("timings"), Mapping) else {}
    generated = timings.get("predicted_n")
    if not isinstance(generated, int):
        generated = 1 if str(response.get("content", "")) else 0
    return {
        "request_index": int(request_index),
        "request_payload": payload,
        "response": response,
        "response_sha256": sha256_json(response),
        "requested_tokens": 1,
        "generated_tokens": int(generated),
        "completed": int(generated) >= 1,
        "monotonic_start_ns": started_ns,
        "monotonic_end_ns": finished_ns,
        "wall_clock_start": started_wall,
        "wall_clock_end": finished_wall,
        "terminal": True,
    }


def _gpu_process_sample(
    gpu: Mapping[str, Any], pid: int
) -> JsonDict:  # pragma: no cover - hardware boundary
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    pid_memory_mb = 0
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[0] == gpu["gpu_uuid"] and parts[1] == str(pid):
            pid_memory_mb = int(parts[2])
            break
    current = next(
        (row for row in _gpu_rows() if row["gpu_uuid"] == gpu["gpu_uuid"]),
        dict(gpu),
    )
    return {
        "pid": int(pid),
        "gpu_uuid": gpu["gpu_uuid"],
        "gpu_model": gpu["gpu_model"],
        "device": f"cuda:{gpu['index']}",
        "pid_memory_mb": pid_memory_mb,
        "memory_free_mb": current.get("memory_free_mb"),
        "utilization_pct": current.get("utilization_pct"),
        "sample_time_ns": time.monotonic_ns(),
        "terminal": True,
    }


def _port_is_free(port: int) -> bool:  # pragma: no cover - socket boundary
    import socket

    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", int(port))) != 0


def _finish_gpu_lease(
    lease: Any, *, success: bool, process_exit_confirmed: bool, gpu: Mapping[str, Any]
) -> JsonDict:  # pragma: no cover - ownership boundary
    from carnot import gpu_lease_phase_journal as lease_api

    phase = str(lease.document.get("phase"))
    if phase in {"resident", "inferencing"}:
        lease.transition("unloading")
        phase = "unloading"
    if phase == "unloading":
        current = next(
            (row for row in _gpu_rows() if row["gpu_uuid"] == gpu["gpu_uuid"]),
            gpu,
        )
        after_vram = int(current["memory_total_mb"]) - int(current["memory_free_mb"])
        lease.transition(
            "validating",
            vram_mb=after_vram,
            exit_code=0 if process_exit_confirmed else 1,
            unload_observed=process_exit_confirmed,
        )
        phase = "validating"
    if phase == "validating":
        lease.transition("terminal_complete" if success else "terminal_blocked")
    elif phase not in lease_api.TERMINAL_PHASES:
        lease.transition("terminal_blocked")
    return lease.release()


def execute_live_capture(
    *, preflight: Mapping[str, Any], checkpoint_path: Path, run_date: str, started: float
) -> JsonDict:  # pragma: no cover - owned live orchestration
    """Launch one owned server, capture diagnostics, and release its resources."""

    from carnot import gpu_lease_phase_journal as lease_api
    from carnot.agentic import arc_belief_shadow_live_trace as live_core
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    checks = [deepcopy(dict(row)) for row in preflight["checks"]]
    spec = deepcopy(dict(preflight["model_spec"]))
    gpu = deepcopy(dict(preflight["gpu"]))
    task_id = f"exp7051-report-requalification:{run_date}:{os.getpid()}"
    evidence = _empty_live_evidence()
    port_lease: Any = None
    gpu_lease: Any = None
    proposer: Any = None
    owned_pid = -1
    owned_start_ns: int | None = None
    owned_start_wall: str | None = None
    shutdown_ns: int | None = None
    shutdown_wall: str | None = None
    lease_release: JsonDict = {}
    process_cleanup: JsonDict = {
        "owned_pid": -1,
        "ownership_match": False,
        "signals_sent_only_to_owned_pid": False,
        "signals_sent": [],
        "process_exit_confirmed": True,
        "process_reaped": True,
        "terminal": True,
    }
    failure: tuple[str, Any, Any] | None = None
    old_env = {
        key: os.environ.get(key)
        for key in (
            "CARNOT_ARC_GENERATOR_CUDA_GPU",
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA",
            "CARNOT_ARC_N_CTX",
            "CARNOT_ARC_LLAMA_PARALLEL",
            "CARNOT_ARC_GENERATOR_SEED",
            "CARNOT_ARC_SERVER_LOG_DIR",
        )
    }
    try:
        late_servers = _unattributed_llama_server_rows()
        checks.append(gate_row("no_late_unattributed_llama_server", [], late_servers))
        if late_servers:
            raise RuntimeError("unattributed llama-server appeared after preflight")
        port_lease = live_core._PortLease(checkpoint_path.parent / "port-leases", task_id)
        checks.append(gate_row("owned_port_lease", True, True))
        gpu_lease = lease_api.GpuLease.acquire(
            runtime_dir=checkpoint_path.parent / "gpu-lease",
            task_id=task_id,
            device_uuid=str(gpu["gpu_uuid"]),
            expected_model=str(spec["model_path"]),
            vram_before_mb=int(gpu["memory_total_mb"]) - int(gpu["memory_free_mb"]),
            ttl_s=1800.0,
        )
        gpu_lease.transition("admitted")
        checks.append(gate_row("owned_gpu_lease", True, True))
        os.environ.update(
            {
                "CARNOT_ARC_GENERATOR_CUDA_GPU": str(gpu["index"]),
                "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
                "CARNOT_ARC_N_CTX": str(MODEL_N_CTX),
                "CARNOT_ARC_LLAMA_PARALLEL": "1",
                "CARNOT_ARC_GENERATOR_SEED": str(RANDOM_SEED),
                "CARNOT_ARC_SERVER_LOG_DIR": str(checkpoint_path.parent / "server-logs"),
            }
        )
        gpu_lease.transition("loading")
        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.6-35B-A3B",
            n_ctx=MODEL_N_CTX,
            max_tokens=1,
            timeout=600,
            port=int(port_lease.port),
            mtp=False,
            kv_quant="q8_0",
            n_gpu_layers=999,
            ffn_cpu_layers=0,
            model_path=str(spec["model_path"]),
            model_repository=MANDATED_MODEL_HF_ID,
            model_filename=str(spec["model_filename"]),
            model_revision=str(spec["revision"]),
            tries=1,
        )
        if proposer._ensure_server() is not True:
            raise RuntimeError("owned CUDA llama-server failed to start")
        process = proposer._proc
        if process is None or process.poll() is not None or proposer.port != port_lease.port:
            raise RuntimeError("server was reused, exited, or changed the leased port")
        owned_pid = int(process.pid)
        owned_start_ns = time.monotonic_ns()
        owned_start_wall = _utc_now()
        launch_argv = list(proposer.last_launch_argv)
        if "-m" not in launch_argv or launch_argv.index("-m") + 1 >= len(launch_argv):
            raise RuntimeError("owned server command has no launch model argument")
        launch_argument = launch_argv[launch_argv.index("-m") + 1]
        evidence["launch_model_argument"] = launch_argument
        evidence["process_command_rows"] = [_read_process_command(owned_pid)]
        raw_props = _http_json(proposer._url() + "/props")
        evidence["raw_server_props"] = raw_props
        current = next(row for row in _gpu_rows() if row["gpu_uuid"] == gpu["gpu_uuid"])
        resident_vram = int(current["memory_total_mb"]) - int(current["memory_free_mb"])
        gpu_lease.transition("resident", vram_mb=resident_vram)
        gpu_lease.transition("inferencing")
        diagnostics = [
            _diagnostic_request(proposer._url(), prompt, index)
            for index, prompt in enumerate(DIAGNOSTIC_PROMPTS, start=1)
        ]
        evidence["diagnostic_request_rows"] = diagnostics
        gpu_sample = _gpu_process_sample(gpu, owned_pid)
        evidence["gpu_sample_rows"] = [gpu_sample]
        cuda_offload = (
            "-ngl" in launch_argv
            and int(launch_argv[launch_argv.index("-ngl") + 1]) > 0
            and gpu_sample["pid_memory_mb"] > 0
            and preflight["server"].get("cuda_enabled") is True
            and Path(proposer.generator_server_path).resolve()
            == Path(preflight["server"]["path"]).resolve()
        )
        evidence["server_process_rows"] = [
            {
                "pid": owned_pid,
                "owned": True,
                "port": proposer.port,
                "endpoint": proposer._url(),
                "server_binary": proposer.generator_server_path,
                "server_binary_hash": sha256_file(proposer.generator_server_path),
                "server_build": preflight["server"].get("version_output"),
                "launch_argv": launch_argv,
                "launch_model_argument": launch_argument,
                "raw_server_props_sha256": sha256_json(raw_props),
                "cuda_layer_offload_confirmed": cuda_offload,
                "terminal": True,
            }
        ]
        checks.extend(
            (
                gate_row("owned_model_server", True, True),
                gate_row(
                    "diagnostic_requests_generated_tokens",
                    True,
                    all(row["completed"] is True for row in diagnostics),
                ),
                gate_row("cuda_layer_offload_confirmed", True, cuda_offload),
            )
        )
        if any(row["passed"] is not True for row in checks[-3:]):
            raise RuntimeError("live model, diagnostic, or CUDA evidence failed")
        while (
            time.monotonic_ns() - owned_start_ns
        ) / 1_000_000_000 < MINIMUM_OWNED_LIVE_INTERVAL_S:
            if process.poll() is not None:
                raise RuntimeError("owned CUDA llama-server exited during the live interval")
            remaining = MINIMUM_OWNED_LIVE_INTERVAL_S - (
                (time.monotonic_ns() - owned_start_ns) / 1_000_000_000
            )
            time.sleep(min(1.0, max(0.0, remaining)))
    except Exception as exc:  # noqa: BLE001 - exact failure becomes terminal evidence
        failure = (
            "live_report_capture",
            "one owned complete CUDA capture",
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        if proposer is not None and getattr(proposer, "_proc", None) is not None:
            process_cleanup = cleanup_owned_process(proposer._proc, owned_pid=owned_pid)
        shutdown_ns = time.monotonic_ns()
        shutdown_wall = _utc_now()
        process_exit_confirmed = process_cleanup.get("process_exit_confirmed") is True
        if gpu_lease is not None:
            try:
                lease_release = _finish_gpu_lease(
                    gpu_lease,
                    success=failure is None,
                    process_exit_confirmed=process_exit_confirmed,
                    gpu=gpu,
                )
            except Exception as exc:  # noqa: BLE001 - cleanup failure remains exact
                if failure is None:
                    failure = (
                        "gpu_lease_cleanup",
                        "released terminal lease",
                        f"{type(exc).__name__}: {exc}",
                    )
        port = int(port_lease.port) if port_lease is not None else -1
        if port_lease is not None:
            port_lease.release()
        port_released = port_lease is None or (port_lease.released and _port_is_free(port))
        lease_released = lease_release.get("released") is True
        cleanup_passed = bool(process_exit_confirmed and lease_released and port_released)
        evidence["port_lease_rows"] = [
            {
                "port": port,
                "lease_path": str(port_lease.path) if port_lease is not None else None,
                "owned": port_lease is not None,
                "released": port_released,
                "terminal": True,
            }
        ]
        evidence["gpu_lease_rows"] = [
            {
                "gpu_uuid": gpu["gpu_uuid"],
                "lease_id": lease_release.get("lease_id"),
                "owned": gpu_lease is not None,
                "released": lease_released,
                "phase": lease_release.get("phase"),
                "terminal": True,
            }
        ]
        evidence["cleanup_rows"] = [
            {
                **process_cleanup,
                "lease_released": lease_released,
                "port_released": port_released,
                "passed": cleanup_passed,
                "terminal": True,
            }
        ]
        diagnostics = evidence["diagnostic_request_rows"]
        if owned_start_ns is not None and shutdown_ns is not None:
            evidence["owned_live_interval_s"] = (shutdown_ns - owned_start_ns) / 1_000_000_000
            if diagnostics:
                evidence["phase_clock_rows"] = [
                    {
                        "event": "owned_live_start",
                        "monotonic_ns": owned_start_ns,
                        "wall_clock": owned_start_wall,
                        "terminal": True,
                    },
                    {
                        "event": "first_token",
                        "monotonic_ns": diagnostics[0]["monotonic_end_ns"],
                        "wall_clock": diagnostics[0]["wall_clock_end"],
                        "terminal": True,
                    },
                    {
                        "event": "last_token",
                        "monotonic_ns": diagnostics[-1]["monotonic_end_ns"],
                        "wall_clock": diagnostics[-1]["wall_clock_end"],
                        "terminal": True,
                    },
                    {
                        "event": "shutdown",
                        "monotonic_ns": shutdown_ns,
                        "wall_clock": shutdown_wall,
                        "terminal": True,
                    },
                ]
        if evidence["owned_live_interval_s"] < MINIMUM_OWNED_LIVE_INTERVAL_S and failure is None:
            failure = (
                "owned_live_interval_s",
                f">={MINIMUM_OWNED_LIVE_INTERVAL_S:.1f}",
                evidence["owned_live_interval_s"],
            )
        if not cleanup_passed and failure is None:
            failure = ("owned_cleanup", True, evidence["cleanup_rows"][0])
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    if failure is not None:
        checks.append(gate_row(failure[0], failure[1], failure[2]))
        return build_blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions=checks,
            source_artifact_hashes=preflight["source_hashes"],
            model_specs=[spec],
            selected_model_spec=spec,
            live_evidence=evidence,
        )
    return build_positive_artifact(
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        preconditions=checks,
        source_artifact_hashes=preflight["source_hashes"],
        model_specs=[spec],
        selected_model_spec=spec,
        live_evidence=evidence,
    )


def run(
    *,
    run_date: str,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> JsonDict:  # pragma: no cover - live orchestration boundary
    """Run preflight, one capture, validation, and atomic publication."""

    started = time.perf_counter()
    output = output_path or repo_root / RESULT_RELATIVE_PATH
    checkpoint = checkpoint_path or repo_root / CHECKPOINT_RELATIVE_PATH
    preflight = collect_preconditions(
        repo_root=repo_root,
        output_path=output,
        checkpoint_path=checkpoint,
    )
    if preflight["summary"]["passed"] is not True:
        spec = preflight.get("model_spec")
        artifact = build_blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions=preflight["checks"],
            source_artifact_hashes=preflight["source_hashes"],
            model_specs=[spec] if isinstance(spec, Mapping) else [],
            selected_model_spec=spec if isinstance(spec, Mapping) else None,
            live_evidence={"server_process_rows": preflight.get("unattributed_server_rows", [])},
        )
    else:
        artifact = execute_live_capture(
            preflight=preflight,
            checkpoint_path=checkpoint,
            run_date=run_date,
            started=started,
        )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("Exp7051 produced an invalid artifact: " + ";".join(errors))
    write_artifact(output, artifact)
    return artifact


def _date_argument(value: str) -> str:
    if re.fullmatch(r"\d{8}", value) is None:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD")
    return value


def _parser() -> argparse.ArgumentParser:  # pragma: no cover - CLI boundary
    parser = argparse.ArgumentParser(description="Requalify one owned llama.cpp model report")
    parser.add_argument("--date", type=_date_argument, default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT / CHECKPOINT_RELATIVE_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary
    """Write one stable terminal artifact and print its terminal state."""

    args = _parser().parse_args(argv)
    if args.output.is_file():
        existing = _load_json(args.output)
        recorded = existing.get("source_artifact_hashes")
        current = _source_hashes(REPO_ROOT)
        stable = isinstance(recorded, Mapping) and all(
            recorded.get(path) == digest for path, digest in current.items()
        )
        if (
            existing.get("execution_date") == args.date
            and stable
            and validate_artifact(existing) == []
        ):
            print(
                f"stable {args.output} model_report_evidence_ready_score="
                f"{existing['model_report_evidence_ready_score']}"
            )
            return 0
    artifact = run(
        run_date=args.date,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )
    print(
        f"wrote {args.output} model_report_evidence_ready_score="
        f"{artifact['model_report_evidence_ready_score']} "
        f"verdict={artifact['honest_verdict']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
