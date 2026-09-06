"""Capture one owned llama.cpp model report without taking an ARC action.

The report channel is evidence, not an identity validator. This module keeps
each raw ``/props`` value unchanged and derives path facts in separate rows.
The production ARC provenance validator is not called or changed.

Spec: REQ-ARC-7039 and SCENARIO-ARC-7039-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time
from typing import Any

from carnot.inference.sota_models import cached_sota_pair


JsonDict = dict[str, Any]
EXPERIMENT_ID = 7039
SCHEMA = "carnot.exp7039.v617_model_report_forensics.v1"
RUN_DATE = "20260906"
RANDOM_SEED = 7_039_202_609_06
INFERENCE_SUBSTRATE = "live_llm_inference"
MANDATED_MODEL_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
MANDATED_MODEL_NAME = "Qwen3.6-35B-A3B"
ONE_TOKEN_PROMPT = "/no_think\nReply with the single token OK."
MODEL_N_CTX = 4096
LIVE_INFERENCE_DURATION_FLOOR_S = 60.0
IDENTITY_FIELDS = ("model_path", "model", "model_alias")
REPORT_CLASSIFICATIONS = frozenset(
    {"snapshot_alias", "resolved_blob", "direct_file", "conflicting", "unknown"}
)
EVIDENCE_STATUSES = frozenset({"supported", "contradicted", "unknown"})
RESULT_RELATIVE_PATH = Path("results/experiment_7039_v617_model_report_forensics.json")
CHECKPOINT_RELATIVE_PATH = Path("results/checkpoints/experiment_7039/checkpoint.json")
EXP7032_RELATIVE_PATH = Path("results/experiment_7032_repaired_belief_shadow_live_trace.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_7039_v617_model_report_forensics.py")
SCRIPT_RELATIVE_PATH = Path(
    "scripts/experiments/experiment_7039_v617_model_report_forensics.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_7039_v617_model_report_forensics.py"
)
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
    "report_channel_classification",
    "one_token_probe_rows",
    "server_process_rows",
    "port_lease_rows",
    "gpu_lease_rows",
    "gpu_sample_rows",
    "cuda_layer_offload_confirmed",
    "cleanup_rows",
    "generation_request_count",
    "arc_action_count",
    "game_level_solve_claim",
    "arc_report_channel_forensics_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each required field states why the report capture needs it.",
    "preconditions_checked": "Exact gates stop missing hardware from becoming synthetic evidence.",
    "inference_substrate": "The substrate distinguishes real local generation from offline analysis.",
    "duration_s": "Measured wall time makes model load and inference auditable.",
    "source_artifact_hashes": "Source hashes bind the result to the code and prior evidence it used.",
    "rows": "Summary rows make evidence-completeness checks independently countable.",
    "MODEL_SPECS": "The uppercase field preserves the experiment model-selection contract.",
    "model_specs": "The lowercase field supports standard methodology audits.",
    "models_used": "The invoked-model list prevents a fallback model from satisfying the task.",
    "selected_model_spec": "The selected specification records one immutable model request.",
    "requested_model_path": "The snapshot path preserves the exact selected launch intent.",
    "requested_model_filename": "The filename records the requested GGUF name before resolution.",
    "requested_hf_id": "The hub ID distinguishes equal filenames from different repositories.",
    "requested_revision": "The revision binds the request to one cached snapshot.",
    "launch_model_argument": "The launch argument records what the server command received.",
    "process_command_rows": "The process command line independently checks the launch record.",
    "raw_server_props": "The complete response preserves the server report before interpretation.",
    "raw_identity_field_rows": "Separate raw fields expose omissions and contradictions.",
    "resolved_identity_field_rows": "Derived rows resolve paths without overwriting observations.",
    "file_hash_rows": "File hashes compare every reachable candidate with the selected bytes.",
    "consistency_rows": "Typed consistency states keep unknown facts from becoming passes.",
    "report_channel_classification": "A closed shape label makes the observed report reproducible.",
    "one_token_probe_rows": "One bounded response proves the loaded model executed inference.",
    "server_process_rows": "Server rows bind build, process, model, port, and CUDA intent.",
    "port_lease_rows": "Port rows prove the endpoint was owned and later released.",
    "gpu_lease_rows": "GPU rows prove the selected device belonged to this task.",
    "gpu_sample_rows": "A process sample proves the owned server used the leased GPU.",
    "cuda_layer_offload_confirmed": "True rules out a CPU or zero-layer fallback.",
    "cleanup_rows": "Cleanup rows prove only owned resources were stopped and released.",
    "generation_request_count": "An exact count prevents a skipped or repeated diagnostic.",
    "arc_action_count": "Zero proves the report capture did not act in an ARC game.",
    "game_level_solve_claim": "False prevents identity evidence from becoming solve credit.",
    "arc_report_channel_forensics_ready_score": "One means the evidence capture is complete and cleaned up.",
    "random_seed": "A fixed seed makes the one-token request reproducible.",
    "reproducibility_checksum": "A canonical digest detects later artifact changes.",
    "gate_check_summary": "The first exact failure makes a blocked run actionable.",
    "verifier_is_oracle": "False states that this report capture does not define ARC correctness.",
    "verdict_class": "A closed class gives the terminal result one machine-readable meaning.",
    "honest_verdict": "A class-consistent prefix prevents blockage from reading as success.",
}

VERDICT_PREFIXES = {
    "positive": "complete_positive_",
    "circular_positive": "complete_circular_positive_",
    "null": "complete_null_",
    "blocked": "blocked_",
    "disqualified": "disqualified_",
    "partial": "partial_",
}


def sha256_file(path: str | Path) -> str:
    """Hash one readable file with the project digest label."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash stable JSON bytes without changing the supplied object."""

    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the complete artifact except its self-referential checksum."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def gate_row(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record an exact check without replacing its observed value."""

    return {
        "check": str(check),
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
        "terminal": True,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Return the first exact failure while retaining every check."""

    rows = [dict(row) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "failed_check": None if failed is None else failed.get("check"),
        "expected_value": True if failed is None else failed.get("expected_value"),
        "observed_value": True if failed is None else failed.get("observed_value"),
        "checks": rows,
    }


def extract_quantization(filename: Any) -> str | None:
    """Read the selected quantization token from one GGUF filename."""

    name = str(filename or "").upper()
    for token in ("Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q3_K_M", "Q2_K"):
        if token in name:
            return token
    return None


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
) -> JsonDict | None:
    """Select the mandated Qwen snapshot only from ``cached_sota_pair()``."""

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
            "model_path": str(path.absolute()),
            "model_filename": path.name,
            "revision": revision,
            "quantization": extract_quantization(path.name),
            "model_file_hash": sha256_file(path),
            "resolved_via": "cached_sota_pair",
        }
    )
    return selected


def _raw_kind(present: bool, value: Any) -> str:
    if not present or value is None:
        return "missing"
    if not isinstance(value, str):
        return "non_string"
    if not value.strip():
        return "blank"
    return "absolute_path" if Path(value).is_absolute() else "relative"


def _is_content_blob(path: Path) -> bool:
    return path.parent.name == "blobs" and re.fullmatch(r"[0-9a-f]{64}", path.name) is not None


def analyze_identity_report(
    raw_server_props: Mapping[str, Any], selected_model_spec: Mapping[str, Any]
) -> JsonDict:
    """Preserve raw identity fields and derive path facts in separate rows.

    Only absolute paths are resolved. A missing or relative value stays
    unknown. A reachable file with different bytes is contradicted.
    """

    props = deepcopy(dict(raw_server_props))
    selected_hash = selected_model_spec.get("model_file_hash")
    raw_rows: list[JsonDict] = []
    resolved_rows: list[JsonDict] = []
    file_rows: list[JsonDict] = []
    for field in IDENTITY_FIELDS:
        present = field in props
        value = deepcopy(props.get(field))
        kind = _raw_kind(present, value)
        raw_row: JsonDict = {
            "field": field,
            "present": present,
            "raw_value": value,
            "raw_kind": kind,
            "evidence_status": "unknown",
            "terminal": True,
        }
        derived: JsonDict = {
            "field": field,
            "raw_value": value,
            "resolution_state": kind,
            "resolved_path": None,
            "reachable_file": False,
            "is_symlink": False,
            "raw_equals_resolved": None,
            "evidence_status": "unknown",
            "terminal": True,
        }
        if kind == "absolute_path":
            path = Path(value)
            derived["is_symlink"] = path.is_symlink()
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                derived["resolution_state"] = "unreachable"
            else:
                if resolved.is_file():
                    digest = sha256_file(resolved)
                    matches = isinstance(selected_hash, str) and digest == selected_hash
                    status = "supported" if matches else "contradicted"
                    derived.update(
                        {
                            "resolution_state": "resolved",
                            "resolved_path": str(resolved),
                            "reachable_file": True,
                            "raw_equals_resolved": str(path) == str(resolved),
                            "evidence_status": status,
                        }
                    )
                    raw_row["evidence_status"] = status
                    file_rows.append(
                        {
                            "source_field": field,
                            "raw_path": str(path),
                            "resolved_path": str(resolved),
                            "sha256": digest,
                            "selected_model_hash": selected_hash,
                            "hash_matches_selected": matches,
                            "status": status,
                            "terminal": True,
                        }
                    )
                else:
                    derived["resolution_state"] = "not_file"
        raw_rows.append(raw_row)
        resolved_rows.append(derived)

    reachable = [row for row in resolved_rows if row["reachable_file"] is True]
    any_contradicted = any(row["evidence_status"] == "contradicted" for row in reachable)
    resolved_paths = {row["resolved_path"] for row in reachable}
    hash_status = (
        "contradicted"
        if any_contradicted
        else ("supported" if reachable else "unknown")
    )
    path_status = (
        "contradicted"
        if len(resolved_paths) > 1
        else ("supported" if len(reachable) > 1 else "unknown")
    )
    consistency_rows = [
        {
            "check": "selected_model_hash_agreement",
            "status": hash_status,
            "observed_value": [row["evidence_status"] for row in reachable],
            "terminal": True,
        },
        {
            "check": "resolved_report_path_agreement",
            "status": path_status,
            "observed_value": sorted(str(path) for path in resolved_paths),
            "terminal": True,
        },
    ]
    if any_contradicted or len(resolved_paths) > 1:
        classification = "conflicting"
    else:
        supported = next(
            (row for row in resolved_rows if row["evidence_status"] == "supported"),
            None,
        )
        if supported is None:
            classification = "unknown"
        elif supported["is_symlink"] is True:
            classification = "snapshot_alias"
        elif _is_content_blob(Path(str(supported["resolved_path"]))):
            classification = "resolved_blob"
        else:
            classification = "direct_file"
    return {
        "raw_server_props": props,
        "raw_identity_field_rows": raw_rows,
        "resolved_identity_field_rows": resolved_rows,
        "file_hash_rows": file_rows,
        "consistency_rows": consistency_rows,
        "report_channel_classification": classification,
    }


def _empty_live_evidence() -> JsonDict:
    """Create every live evidence field before a process exists."""

    return {
        "launch_model_argument": None,
        "process_command_rows": [],
        "raw_server_props": {},
        "one_token_probe_rows": [],
        "server_process_rows": [],
        "port_lease_rows": [],
        "gpu_lease_rows": [],
        "gpu_sample_rows": [],
        "cleanup_rows": [],
    }


def _status(left: Any, right: Any) -> str:
    if left is None or right is None:
        return "unknown"
    return "supported" if left == right else "contradicted"


def _build_artifact_body(
    *,
    run_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, str],
    model_specs: Sequence[Mapping[str, Any]],
    selected_model_spec: Mapping[str, Any] | None,
    live_evidence: Mapping[str, Any] | None,
) -> JsonDict:
    """Project raw and derived evidence into the shared terminal schema."""

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
    process_model_argument = process_rows[0].get("model_argument") if process_rows else None
    requested_path = selected.get("model_path") if selected else None
    launch_argument = evidence["launch_model_argument"]
    consistency = deepcopy(report["consistency_rows"])
    consistency.extend(
        [
            {
                "check": "launch_argument_matches_requested_path",
                "status": _status(launch_argument, requested_path),
                "observed_value": launch_argument,
                "expected_value": requested_path,
                "terminal": True,
            },
            {
                "check": "process_argument_matches_launch_argument",
                "status": _status(process_model_argument, launch_argument),
                "observed_value": process_model_argument,
                "expected_value": launch_argument,
                "terminal": True,
            },
        ]
    )
    file_rows = deepcopy(report["file_hash_rows"])
    if selected is not None and requested_path:
        try:
            selected_observed_hash = sha256_file(requested_path)
        except OSError:
            selected_observed_hash = None
        selected_expected_hash = selected.get("model_file_hash")
        selected_matches = (
            selected_observed_hash is not None
            and selected_observed_hash == selected_expected_hash
        )
        file_rows.insert(
            0,
            {
                "source_field": "selected_model_spec.model_path",
                "raw_path": requested_path,
                "resolved_path": (
                    str(Path(requested_path).resolve()) if selected_observed_hash else None
                ),
                "sha256": selected_observed_hash,
                "selected_model_hash": selected_expected_hash,
                "hash_matches_selected": selected_matches,
                "status": "supported" if selected_matches else "unknown",
                "terminal": True,
            },
        )
    probes = deepcopy(list(evidence["one_token_probe_rows"]))
    server_rows = deepcopy(list(evidence["server_process_rows"]))
    gpu_rows = deepcopy(list(evidence["gpu_sample_rows"]))
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": str(run_date),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [dict(row) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": dict(source_artifact_hashes),
        "rows": [],
        "MODEL_SPECS": deepcopy([dict(row) for row in model_specs]),
        "model_specs": deepcopy([dict(row) for row in model_specs]),
        "models_used": [selected["hf_id"]] if selected and probes else [],
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
        "consistency_rows": consistency,
        "report_channel_classification": report["report_channel_classification"],
        "one_token_probe_rows": probes,
        "server_process_rows": server_rows,
        "port_lease_rows": deepcopy(list(evidence["port_lease_rows"])),
        "gpu_lease_rows": deepcopy(list(evidence["gpu_lease_rows"])),
        "gpu_sample_rows": gpu_rows,
        "cuda_layer_offload_confirmed": bool(
            server_rows and server_rows[0].get("cuda_layer_offload_confirmed") is True
        ),
        "cleanup_rows": deepcopy(list(evidence["cleanup_rows"])),
        "generation_request_count": len(probes),
        "arc_action_count": 0,
        "game_level_solve_claim": False,
        "arc_report_channel_forensics_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_arc_report_channel_forensics:unclassified",
    }


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
    """Build a schema-complete blocker with the first exact failed check."""

    artifact = _build_artifact_body(
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
    artifact.update(
        {
            "rows": [
                {
                    "check": "report_channel_forensics_capture",
                    "passed": False,
                    "failed_check": failed,
                    "terminal": True,
                }
            ],
            "gate_check_summary": summary,
            "verdict_class": "blocked",
            "honest_verdict": f"blocked_arc_report_channel_forensics:{failed}",
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def _positive_gate_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Reduce ownership and capture evidence into exact positive checks."""

    probes = artifact.get("one_token_probe_rows")
    probes = probes if isinstance(probes, list) else []
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
    return [
        gate_row(
            "launch_model_argument",
            artifact.get("requested_model_path"),
            artifact.get("launch_model_argument"),
        ),
        gate_row("owned_process_command", True, len(process_rows) == 1 and process_rows[0].get("owned") is True),
        gate_row("process_model_argument", artifact.get("launch_model_argument"), process_rows[0].get("model_argument") if process_rows else None),
        gate_row("raw_props_captured", True, isinstance(artifact.get("raw_server_props"), Mapping)),
        gate_row("selected_model_hash", True, len(selected_rows) == 1 and selected_rows[0].get("hash_matches_selected") is True),
        gate_row("generation_request_count", 1, artifact.get("generation_request_count")),
        gate_row(
            "live_inference_duration_floor_s",
            True,
            float(artifact.get("duration_s", 0.0)) >= LIVE_INFERENCE_DURATION_FLOOR_S,
        ),
        gate_row("one_token_probe_complete", True, len(probes) == 1 and probes[0].get("completed") is True and probes[0].get("generated_tokens") == 1),
        gate_row("owned_server", True, len(servers) == 1 and servers[0].get("owned") is True),
        gate_row("owned_port_lease_released", True, len(ports) == 1 and ports[0].get("owned") is True and ports[0].get("released") is True),
        gate_row("owned_gpu_lease_released", True, len(leases) == 1 and leases[0].get("owned") is True and leases[0].get("released") is True),
        gate_row("gpu_process_sample", True, len(samples) == 1 and float(samples[0].get("pid_memory_mb", 0)) > 0 and str(samples[0].get("gpu_uuid", "")).startswith("GPU-") and "RTX 3090" in str(samples[0].get("gpu_model", ""))),
        gate_row("cuda_layer_offload", True, artifact.get("cuda_layer_offload_confirmed") is True),
        gate_row("safe_cleanup", True, len(cleanup) == 1 and cleanup[0].get("passed") is True),
        gate_row("arc_action_count", 0, artifact.get("arc_action_count")),
        gate_row("game_level_solve_claim", False, artifact.get("game_level_solve_claim")),
    ]


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
    """Build a positive evidence-completeness artifact from one owned capture."""

    artifact = _build_artifact_body(
        run_date=run_date,
        duration_s=duration_s,
        preconditions=preconditions,
        source_artifact_hashes=source_artifact_hashes,
        model_specs=model_specs,
        selected_model_spec=selected_model_spec,
        live_evidence=live_evidence,
    )
    checks = [*list(preconditions), *_positive_gate_rows(artifact)]
    summary = gate_check_summary(checks)
    artifact.update(
        {
            "preconditions_checked": [dict(row) for row in checks],
            "rows": [
                {
                    "check": "raw_identity_fields",
                    "row_count": len(artifact["raw_identity_field_rows"]),
                    "passed": len(artifact["raw_identity_field_rows"]) == 3,
                    "terminal": True,
                },
                {
                    "check": "one_token_probe",
                    "row_count": len(artifact["one_token_probe_rows"]),
                    "passed": len(artifact["one_token_probe_rows"]) == 1,
                    "terminal": True,
                },
                {
                    "check": "safe_cleanup",
                    "row_count": len(artifact["cleanup_rows"]),
                    "passed": bool(artifact["cleanup_rows"] and artifact["cleanup_rows"][0].get("passed") is True),
                    "terminal": True,
                },
            ],
            "arc_report_channel_forensics_ready_score": int(summary["passed"] is True),
            "gate_check_summary": summary,
            "verdict_class": "positive" if summary["passed"] is True else "blocked",
            "honest_verdict": (
                "complete_positive_arc_report_channel_forensics_ready"
                if summary["passed"] is True
                else f"blocked_arc_report_channel_forensics:{summary['failed_check']}"
            ),
        }
    )
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Any) -> list[str]:
    """Validate terminal schema and evidence without validating model identity."""

    if not isinstance(artifact, Mapping):
        return ["artifact_object_required"]
    errors: list[str] = []
    if missing := sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)):
        errors.append(f"required_fields_missing:{missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles_invalid")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    if artifact.get("arc_action_count") != 0 or artifact.get("game_level_solve_claim") is not False:
        errors.append("arc_non_action_contract_invalid")
    classification = artifact.get("report_channel_classification")
    if classification not in REPORT_CLASSIFICATIONS:
        errors.append("report_channel_classification_invalid")
    summary = artifact.get("gate_check_summary")
    if not isinstance(summary, Mapping) or not {
        "passed", "failed_check", "expected_value", "observed_value", "checks"
    } <= set(summary):
        errors.append("gate_check_summary_invalid")
    verdict_class = str(artifact.get("verdict_class", ""))
    prefix = VERDICT_PREFIXES.get(verdict_class)
    verdict = artifact.get("honest_verdict")
    if prefix is None or not isinstance(verdict, str) or not verdict.startswith(prefix):
        errors.append("verdict_prefix_invalid")
    score = artifact.get("arc_report_channel_forensics_ready_score")
    if type(score) is not int or score not in (0, 1):
        errors.append("ready_score_invalid")
    raw_props = artifact.get("raw_server_props")
    raw_rows = artifact.get("raw_identity_field_rows")
    resolved_rows = artifact.get("resolved_identity_field_rows")
    if verdict_class == "positive":
        if not isinstance(raw_props, Mapping):
            errors.append("raw_server_props_invalid")
        if not isinstance(raw_rows, list) or [row.get("field") for row in raw_rows] != list(IDENTITY_FIELDS):
            errors.append("raw_identity_rows_invalid")
        elif isinstance(raw_props, Mapping) and any(
            row.get("raw_value") != raw_props.get(row["field"]) for row in raw_rows
        ):
            errors.append("raw_identity_rows_do_not_match_props")
        if not isinstance(resolved_rows, list) or [row.get("field") for row in resolved_rows] != list(IDENTITY_FIELDS):
            errors.append("resolved_identity_rows_invalid")
        elif isinstance(raw_rows, list) and any(
            row.get("raw_value") != raw_rows[index].get("raw_value")
            for index, row in enumerate(resolved_rows)
        ):
            errors.append("resolved_rows_do_not_retain_raw_values")
        status_rows = [
            *list(raw_rows if isinstance(raw_rows, list) else []),
            *list(resolved_rows if isinstance(resolved_rows, list) else []),
        ]
        if any(row.get("evidence_status") not in EVIDENCE_STATUSES for row in status_rows):
            errors.append("identity_evidence_status_invalid")
        consistency = artifact.get("consistency_rows")
        if not isinstance(consistency, list) or len(consistency) < 4 or any(
            row.get("status") not in EVIDENCE_STATUSES for row in consistency
        ):
            errors.append("consistency_rows_invalid")
        specs = artifact.get("MODEL_SPECS")
        selected = artifact.get("selected_model_spec")
        if not isinstance(specs, list) or len(specs) != 1 or artifact.get("model_specs") != specs or selected != specs[0]:
            errors.append("model_specs_invalid")
        elif selected.get("hf_id") != MANDATED_MODEL_HF_ID or selected.get("resolved_via") != "cached_sota_pair":
            errors.append("selected_model_spec_invalid")
        live_errors = [row["check"] for row in _positive_gate_rows(artifact) if row["passed"] is not True]
        errors.extend(f"positive_evidence_invalid:{name}" for name in live_errors)
        if score != 1 or not isinstance(summary, Mapping) or summary.get("passed") is not True:
            errors.append("positive_terminal_semantics_invalid")
    elif verdict_class == "blocked":
        if score != 0 or not isinstance(summary, Mapping) or summary.get("passed") is not False or summary.get("failed_check") is None:
            errors.append("blocked_terminal_semantics_invalid")
    else:
        errors.append("unsupported_terminal_class")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> None:
    """Validate and atomically publish one terminal artifact."""

    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("invalid Exp7039 artifact: " + ";".join(errors))
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
        EXP7032_RELATIVE_PATH,
        Path("python/carnot/agentic/arc_executable_world_model.py"),
        Path("python/carnot/gpu_lease_phase_journal.py"),
        Path("python/carnot/inference/sota_models.py"),
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


def _official_access_receipt(root: Path) -> JsonDict:  # pragma: no cover - network boundary
    from carnot.agentic import arc_belief_shadow_live_trace as live_core

    return live_core._live_access_receipt(root)


def collect_preconditions(
    *, repo_root: Path, output_path: Path, checkpoint_path: Path
) -> JsonDict:  # pragma: no cover - live host boundary
    """Check every required resource before a lease or model launch."""

    checks: list[JsonDict] = []
    exp7032 = _load_json(repo_root / EXP7032_RELATIVE_PATH)
    failure_text = json.dumps(exp7032.get("gate_check_summary", {}), sort_keys=True).lower()
    checks.extend(
        (
            gate_row("exp7032_evidence_readable", True, bool(exp7032)),
            gate_row("exp7032_alias_failure_readable", True, "alias" in failure_text),
        )
    )
    gpu_candidates = [row for row in _gpu_rows() if row["idle"] and row["supported"]]
    gpu = gpu_candidates[0] if gpu_candidates else None
    checks.append(gate_row("idle_supported_rtx3090", True, gpu is not None))
    spec = resolve_model_spec(cached_sota_pair, gpu_index=int(gpu["index"])) if gpu else None
    checks.append(gate_row("cached_mandated_gguf", True, spec is not None))
    if gpu and spec:
        required_mb = int(Path(spec["model_path"]).stat().st_size / (1024 * 1024)) + 2048
        enough = int(gpu["memory_free_mb"]) >= required_mb
        checks.append(
            gate_row(
                "adequate_free_vram_mb",
                f">={required_mb}",
                f">={required_mb}" if enough else int(gpu["memory_free_mb"]),
            )
        )
    else:
        checks.append(gate_row("adequate_free_vram_mb", "model_and_gpu_resolved", "unavailable"))
    server = _cuda_server_receipt()
    checks.extend(
        (
            gate_row("cuda_llama_server_executable", True, server.get("exists") is True),
            gate_row("cuda_llama_server_library", True, server.get("cuda_enabled") is True),
            gate_row("cuda_llama_server_version", 0, server.get("version_returncode")),
            gate_row("clean_stop_authority", True, callable(getattr(subprocess.Popen, "terminate", None))),
        )
    )
    for label, path in (("result_path", output_path.parent), ("checkpoint_path", checkpoint_path.parent)):
        path.mkdir(parents=True, exist_ok=True)
        checks.append(gate_row(f"writable_{label}", True, path.is_dir() and os.access(path, os.W_OK)))
    access = _official_access_receipt(repo_root)
    checks.extend(
        (
            gate_row("official_live_access", True, access.get("anonymous_access_available") is True),
            gate_row("official_catalog_readable_without_game_open", True, bool(access.get("catalog"))),
        )
    )
    return {
        "checks": checks,
        "summary": gate_check_summary(checks),
        "gpu": gpu,
        "model_spec": spec,
        "server": server,
        "access": access,
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


def _http_json(url: str, *, payload: Mapping[str, Any] | None = None) -> JsonDict:  # pragma: no cover - HTTP boundary
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


def _one_token_probe(endpoint: str) -> JsonDict:  # pragma: no cover - model boundary
    payload = {
        "prompt": ONE_TOKEN_PROMPT,
        "n_predict": 1,
        "temperature": 0.0,
        "ignore_eos": True,
        "seed": RANDOM_SEED,
    }
    started_ns = time.monotonic_ns()
    response = _http_json(endpoint + "/completion", payload=payload)
    finished_ns = time.monotonic_ns()
    timings = response.get("timings") if isinstance(response.get("timings"), Mapping) else {}
    generated = timings.get("predicted_n")
    if not isinstance(generated, int):
        generated = 1 if str(response.get("content", "")) else 0
    return {
        "request_payload": payload,
        "response": response,
        "requested_tokens": 1,
        "generated_tokens": int(generated),
        "completed": int(generated) == 1,
        "monotonic_start_ns": started_ns,
        "monotonic_end_ns": finished_ns,
        "terminal": True,
    }


def _gpu_process_sample(gpu: Mapping[str, Any], pid: int) -> JsonDict:  # pragma: no cover - hardware boundary
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
    current = next((row for row in _gpu_rows() if row["gpu_uuid"] == gpu["gpu_uuid"]), dict(gpu))
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
        current = next((row for row in _gpu_rows() if row["gpu_uuid"] == gpu["gpu_uuid"]), gpu)
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
    """Launch one owned server, capture one report, probe once, and clean up."""

    from carnot import gpu_lease_phase_journal as lease_api
    from carnot.agentic import arc_belief_shadow_live_trace as live_core
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    checks = [dict(row) for row in preflight["checks"]]
    spec = dict(preflight["model_spec"])
    gpu = dict(preflight["gpu"])
    task_id = f"exp7039-report-forensics:{run_date}:{os.getpid()}"
    evidence = _empty_live_evidence()
    port_lease: Any = None
    gpu_lease: Any = None
    proposer: Any = None
    owned_pid = -1
    process_exit_confirmed = False
    lease_release: JsonDict = {}
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
        proc = proposer._proc
        if proc is None or proc.poll() is not None or proposer.port != port_lease.port:
            raise RuntimeError("server was reused, exited, or changed the leased port")
        owned_pid = int(proc.pid)
        launch_argv = list(proposer.last_launch_argv)
        launch_argument = launch_argv[launch_argv.index("-m") + 1]
        evidence["launch_model_argument"] = launch_argument
        evidence["process_command_rows"] = [_read_process_command(owned_pid)]
        raw_props = _http_json(proposer._url() + "/props")
        evidence["raw_server_props"] = raw_props
        current = next(row for row in _gpu_rows() if row["gpu_uuid"] == gpu["gpu_uuid"])
        resident_vram = int(current["memory_total_mb"]) - int(current["memory_free_mb"])
        gpu_lease.transition("resident", vram_mb=resident_vram)
        gpu_lease.transition("inferencing")
        probe = _one_token_probe(proposer._url())
        evidence["one_token_probe_rows"] = [probe]
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
        while time.perf_counter() - started < LIVE_INFERENCE_DURATION_FLOOR_S:
            if proc.poll() is not None:
                raise RuntimeError("owned CUDA llama-server exited during the live evidence window")
            remaining = LIVE_INFERENCE_DURATION_FLOOR_S - (time.perf_counter() - started)
            time.sleep(min(1.0, max(0.0, remaining)))
        evidence["server_process_rows"][0]["live_observation_duration_s"] = (
            time.perf_counter() - started
        )
        checks.extend(
            (
                gate_row("owned_model_server", True, True),
                gate_row("one_token_probe_completed", True, probe["completed"] is True),
                gate_row("cuda_layer_offload_confirmed", True, cuda_offload),
            )
        )
        if any(row["passed"] is not True for row in checks[-3:]):
            raise RuntimeError("live model, one-token probe, or CUDA check failed")
    except Exception as exc:  # noqa: BLE001 - exact failure becomes artifact evidence
        failure = ("live_report_capture", "one owned complete CUDA capture", f"{type(exc).__name__}: {exc}")
    finally:
        if proposer is not None and getattr(proposer, "_proc", None) is not None:
            proc = proposer._proc
            if int(getattr(proc, "pid", -2)) == owned_pid and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            process_exit_confirmed = proc.poll() is not None
        elif owned_pid < 0:
            process_exit_confirmed = True
        if gpu_lease is not None:
            try:
                lease_release = _finish_gpu_lease(
                    gpu_lease,
                    success=failure is None,
                    process_exit_confirmed=process_exit_confirmed,
                    gpu=gpu,
                )
            except Exception as exc:  # noqa: BLE001 - cleanup failure is terminal evidence
                if failure is None:
                    failure = ("gpu_lease_cleanup", "released terminal lease", f"{type(exc).__name__}: {exc}")
        port = int(port_lease.port) if port_lease is not None else -1
        port_free = True if port < 0 else _port_is_free(port)
        if port_lease is not None:
            port_lease.release()
        port_released = port_lease is None or (port_lease.released and port_free)
        cleanup_passed = bool(
            process_exit_confirmed and lease_release.get("released") is True and port_released
        )
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
                "released": lease_release.get("released") is True,
                "phase": lease_release.get("phase"),
                "terminal": True,
            }
        ]
        evidence["cleanup_rows"] = [
            {
                "owned_pid": owned_pid,
                "signals_sent_only_to_owned_pid": True,
                "process_exit_confirmed": process_exit_confirmed,
                "process_reaped": process_exit_confirmed,
                "lease_released": lease_release.get("released") is True,
                "port_released": port_released,
                "passed": cleanup_passed,
                "terminal": True,
            }
        ]
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
    """Run preflight, one live capture, validation, and atomic output."""

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
        raise ValueError("Exp7039 produced an invalid artifact: " + ";".join(errors))
    write_artifact(output, artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:  # pragma: no cover - CLI boundary
    parser = argparse.ArgumentParser(description="Capture one owned llama.cpp model report")
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / CHECKPOINT_RELATIVE_PATH)
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
        if existing.get("execution_date") == args.date and stable and validate_artifact(existing) == []:
            print(
                f"stable {args.output} arc_report_channel_forensics_ready_score="
                f"{existing['arc_report_channel_forensics_ready_score']}"
            )
            return 0
    artifact = run(
        run_date=args.date,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )
    print(
        f"wrote {args.output} arc_report_channel_forensics_ready_score="
        f"{artifact['arc_report_channel_forensics_ready_score']} "
        f"verdict={artifact['honest_verdict']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())
