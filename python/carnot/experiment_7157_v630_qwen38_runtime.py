"""Produce one honest Qwen3.8 bounded-generation runtime receipt.

The receipt qualifies local transport only. It does not measure output quality
or verifier value. Every external gate finishes the artifact before exit.

Spec refs: REQ-VERIFY-7157 and SCENARIO-VERIFY-7157-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib import error, request

from carnot.experiment_6212_three_family_gguf_runtime_recovery import (
    read_gguf_metadata,
    resolve_native_llama_server,
    snapshot_revision,
)
from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.llama_server_supervisor import (
    NativeLlamaServerSupervisor,
    canonical_json,
    sha256_file,
    sha256_text,
    supervisor_contract,
)
from carnot.inference.sota_models import (
    LEGACY_COMPARATOR_GGUF_MODELS,
    SOTA_GGUF_MODELS,
    cached_current_model,
)
from carnot.paths import repo_root as find_repo_root


JsonDict = dict[str, Any]
RUN_DATE = "20260909"
RANDOM_SEED = 7_157_202_609_09
RESULT_PATH = Path("results/experiment_7157_v630_qwen38_runtime.json")
RAW_DIR = Path("results/raw/experiment_7157_v630_qwen38_runtime")
QWEN_MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QWEN_FILENAME = "Qwen3.8-27B-Q4_K_M.gguf"
PREFERRED_QUANT = "Q4_K_M"
LEGACY_COMPARATOR_IDS = tuple(row["hf_id"] for row in LEGACY_COMPARATOR_GGUF_MODELS)
CANARY_PROMPT = "Reply with exactly QWEN38_RUNTIME_OK."
BOUNDED_DURATION_FLOOR_S = 10.0
INFERENCE_SUBSTRATE = "live_llm_inference"
EXECUTION_VENUE = "host"

HEADLINE_MODEL_SPEC: JsonDict = {
    "name": "Qwen3.8-27B",
    "hf_id": QWEN_MODEL_ID,
    "gpu": None,
    "model_path": "",
    "selection_role": "current_headline",
    "preferred_quant": PREFERRED_QUANT,
    "remote_allowed": False,
    "resolution_method": "cached_current_model",
    "chat_template_source": "embedded_gguf",
}

READINESS_CHECKS = (
    "run_date",
    "source_paths",
    "output_path",
    "registry_cutover",
    "nvidia_smi",
    "idle_rtx_3090",
    "cuda_llama_runtime",
    "cached_qwen38_q4",
    "embedded_gguf_template",
    "model_load",
    "native_cuda_markers",
    "task_owned_vram",
    "bounded_generation",
    "task_owned_teardown",
    "bounded_duration_floor",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "MODEL_SPECS",
    "model_identity_rows",
    "registry_cutover_rows",
    "model_load_receipts",
    "generation_receipts",
    "gpu_telemetry_rows",
    "server_lease_rows",
    "qwen38_runtime_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Field-level reasons make the runtime claim auditable instead of ceremonial.",
    "status": "A terminal state prevents another partial_running artifact.",
    "preconditions_checked": "Resource receipts prevent a synthetic runtime result when GPU, cache, or binary support is absent.",
    "run_date": "The date binds the receipt to the 2026-09-09 mandate.",
    "inference_substrate": "Use live_llm_inference for the real Qwen3.8 bounded canary.",
    "inference_substrate_class": "Use model_bounded_generation, or blocked_no_run before invocation, because the canary has a small fixed token budget.",
    "execution_venue": "Host and GPU identity keep the runtime claim local and reproducible.",
    "duration_s": "Wall time must exceed the honest bounded-generation floor without pretending this is a full run.",
    "source_artifact_hashes": "Hashes bind the cutover to the exact old registry, template, and prior receipts.",
    "rows": "Typed registry, runtime, generation, and teardown rows support independent review.",
    "MODEL_SPECS": "The literal model specification proves every LLM call used unsloth/Qwen3.8-27B-GGUF.",
    "model_identity_rows": "Exact repository, filename, revision, bytes, and hash prevent model-label drift.",
    "registry_cutover_rows": "Cutover rows distinguish the mandate from preserved comparators.",
    "model_load_receipts": "Load receipts prove weights were actually opened before generation.",
    "generation_receipts": "Raw canary evidence proves the model emitted tokens rather than merely loading.",
    "gpu_telemetry_rows": "Task-owned VRAM and CUDA markers prevent CPU-only work from being labeled GPU inference.",
    "server_lease_rows": "PID ownership and teardown receipts prevent orphaned servers and unsafe cleanup.",
    "qwen38_runtime_ready_score": "The exact field gates later model science without asserting quality.",
    "random_seed": "A fixed decoding seed makes the canary reproducible.",
    "reproducibility_checksum": "The checksum detects registry, model, prompt, or binary drift.",
    "gate_check_summary": "A blocked artifact names the failed resource and its observed state.",
    "verifier_is_oracle": "False records that this task qualifies transport, not model correctness.",
    "verdict_class": "Use the closed enum positive | circular_positive | null | blocked | disqualified | partial for structural aggregation.",
    "honest_verdict": "The terminal text must say runtime ready, blocked, or disqualified without a quality claim.",
}

REQUIRED_SOURCE_PATHS = (
    Path("results/experiment_7153_v629_grounding_runtime.json"),
    Path("results/experiment_7154_v629_qwen_dual_side_grounding.json"),
    Path("scripts/experiment_template.py"),
    Path("python/carnot/inference/sota_models.py"),
    Path("python/carnot/inference/llama_server_supervisor.py"),
    Path("python/carnot/agentic/arc_executable_world_model.py"),
    Path("openspec/capabilities/verification/spec.md"),
    Path("openspec/capabilities/llm-ebm-inference/spec.md"),
)


def _required_source_hashes() -> JsonDict:
    """Hash the exact cutover sources and the two incident receipts."""

    root = Path(__file__).resolve().parents[2]
    return {
        str(path): sha256_file(root / path) if (root / path).is_file() else None
        for path in REQUIRED_SOURCE_PATHS
    }


REQUIRED_SOURCE_HASHES = _required_source_hashes()


def gate_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Keep the full comparison so a failed gate names observed state."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is not None:
        return deepcopy(dict(failed))
    return gate_row("all_readiness_gates", 1, 1, True)


def registry_cutover_rows() -> list[JsonDict]:
    """Separate the one mandate from old models kept for comparisons."""

    return [
        {
            **deepcopy(SOTA_GGUF_MODELS[0]),
            "selection_role": "current_headline",
        },
        *[
            {**deepcopy(row), "selection_role": "legacy_comparator"}
            for row in LEGACY_COMPARATOR_GGUF_MODELS
        ],
    ]


def typed_rows(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Add stable row types to each detailed receipt collection."""

    rows: list[JsonDict] = []
    for field, row_type in (
        ("registry_cutover_rows", "registry_cutover"),
        ("model_identity_rows", "model_identity"),
        ("model_load_receipts", "model_load"),
        ("generation_receipts", "generation"),
        ("gpu_telemetry_rows", "gpu_telemetry"),
        ("server_lease_rows", "server_lease"),
    ):
        rows.extend(
            {"row_type": row_type, **deepcopy(dict(row))} for row in artifact.get(field, [])
        )
    return rows


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every receipt field except the checksum itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    return sha256_text(canonical_json(payload))


def base_artifact(run_date: str) -> JsonDict:
    """Create every final field before an external resource check."""

    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "running",
        "preconditions_checked": [],
        "run_date": run_date,
        "inference_substrate": "no_inference",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "MODEL_SPECS": [deepcopy(HEADLINE_MODEL_SPEC)],
        "model_identity_rows": [],
        "registry_cutover_rows": registry_cutover_rows(),
        "model_load_receipts": [],
        "generation_receipts": [],
        "gpu_telemetry_rows": [],
        "server_lease_rows": [],
        "qwen38_runtime_ready_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_row("experiment_complete", True, False, False),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_qwen38_runtime_running",
    }
    artifact["rows"] = typed_rows(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def write_artifact(path: Path | str, artifact: Mapping[str, Any]) -> Path:
    """Write one atomic checkpoint so the schema survives interruption."""

    return atomic_write_json(path, dict(artifact), allow_override=False, sort_keys=True)


def initialize_artifact(path: Path | str, run_date: str) -> JsonDict:
    """Persist the complete no-run shape before checking the host."""

    artifact = base_artifact(run_date)
    write_artifact(path, artifact)
    return artifact


def _terminal_copy(
    artifact: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    result = deepcopy(dict(artifact))
    result["preconditions_checked"] = [deepcopy(dict(row)) for row in checks]
    result["duration_s"] = round(max(0.0, float(duration_s)), 6)
    result["gate_check_summary"] = _gate_summary(result["preconditions_checked"])
    return result


def finish_blocked(
    artifact: Mapping[str, Any],
    path: Path | str,
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    """Finish before invocation when one exact resource gate fails."""

    blocked = _terminal_copy(artifact, checks, duration_s=duration_s)
    failed = str(blocked["gate_check_summary"].get("check") or "unknown_gate")
    blocked.update(
        {
            "status": "blocked",
            "inference_substrate": "no_inference",
            "inference_substrate_class": "blocked_no_run",
            "qwen38_runtime_ready_score": 0,
            "verdict_class": "blocked",
            "honest_verdict": f"blocked_{failed}",
        }
    )
    blocked["rows"] = typed_rows(blocked)
    blocked["reproducibility_checksum"] = artifact_checksum(blocked)
    write_artifact(path, blocked)
    return blocked


def finish_disqualified(
    artifact: Mapping[str, Any],
    path: Path | str,
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:  # pragma: no cover - live failure boundary
    """Finish after invocation without relabeling attempted work as no-run."""

    result = _terminal_copy(artifact, checks, duration_s=duration_s)
    failed = str(result["gate_check_summary"].get("check") or "runtime_evidence")
    result.update(
        {
            "status": "disqualified",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": "model_bounded_generation",
            "qwen38_runtime_ready_score": 0,
            "verdict_class": "disqualified",
            "honest_verdict": f"disqualified_{failed}",
        }
    )
    result["rows"] = typed_rows(result)
    result["reproducibility_checksum"] = artifact_checksum(result)
    write_artifact(path, result)
    return result


def parse_canary_output(raw_output: str) -> JsonDict:
    """Preserve a minimal parse without treating content as correct."""

    text = raw_output.strip()
    return {"text": text, "nonempty": bool(text)}


def canary_payload() -> JsonDict:
    """Return the one fixed deterministic request with a small budget."""

    return {
        "messages": [
            {"role": "system", "content": "Follow the user instruction."},
            {"role": "user", "content": CANARY_PROMPT},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": RANDOM_SEED & 0x7FFFFFFF,
        "cache_prompt": False,
    }


def _model_spec_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    specs = list(artifact.get("MODEL_SPECS") or [])
    if len(specs) != 1 or specs[0].get("hf_id") != QWEN_MODEL_ID:
        errors.append("headline_model_spec_mismatch")
        return errors
    spec = specs[0]
    if Path(str(spec.get("model_path", ""))).name != QWEN_FILENAME:
        errors.append("exact_q4_model_path_mismatch")
    if spec.get("selection_role") != "current_headline":
        errors.append("headline_selection_role_mismatch")
    if spec.get("resolution_method") != "cached_current_model":
        errors.append("canonical_cache_resolution_missing")
    if spec.get("remote_allowed") is not False:
        errors.append("remote_fallback_enabled")
    if spec.get("chat_template_source") != "embedded_gguf":
        errors.append("embedded_template_source_missing")
    return errors


def _positive_evidence_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = _model_spec_errors(artifact)
    if artifact.get("source_artifact_hashes") != REQUIRED_SOURCE_HASHES:
        errors.append("source_artifact_hashes_mismatch")
    if artifact.get("registry_cutover_rows") != registry_cutover_rows():
        errors.append("registry_cutover_rows_mismatch")
    identities = list(artifact.get("model_identity_rows") or [])
    if len(identities) != 1:
        errors.append("model_identity_missing")
    else:
        identity = identities[0]
        if identity.get("repository") != QWEN_MODEL_ID:
            errors.append("model_repository_mismatch")
        if identity.get("filename") != QWEN_FILENAME:
            errors.append("model_filename_mismatch")
        if not identity.get("revision") or int(identity.get("size_bytes", 0) or 0) <= 0:
            errors.append("model_file_identity_incomplete")
        if not str(identity.get("sha256", "")).startswith("sha256:"):
            errors.append("model_hash_missing")
        if (
            identity.get("template_source") != "embedded_gguf"
            or identity.get("template_present") is not True
        ):
            errors.append("embedded_template_missing")
    loads = list(artifact.get("model_load_receipts") or [])
    if len(loads) != 1:
        errors.append("model_load_receipt_missing")
    else:
        load = loads[0]
        if load.get("model_id") != QWEN_MODEL_ID or load.get("health_ok") is not True:
            errors.append("model_load_unconfirmed")
        if load.get("pid_owned_by_task") is not True:
            errors.append("task_pid_ownership_missing")
        if load.get("requested_gpu_layers") != "all":
            errors.append("all_layer_request_missing")
        linkage = dict(load.get("binary_linkage") or {})
        if not all(
            linkage.get(field) is True
            for field in (
                "cuda_linkage_confirmed",
                "libggml_cuda_linked",
                "libcuda_linked",
            )
        ):
            errors.append("native_cuda_linkage_missing")
        markers = set(load.get("native_cuda_markers") or [])
        if not {"CUDA0", "CUDA : ARCHS"}.issubset(markers):
            errors.append("native_cuda_markers_missing")
        if int(load.get("task_owned_vram_delta_mb", 0) or 0) <= 0:
            errors.append("task_owned_vram_missing")
        server_log = str(load.get("server_log", ""))
        if load.get("server_log_sha256") != sha256_text(server_log):
            errors.append("server_log_hash_mismatch")
    generations = list(artifact.get("generation_receipts") or [])
    if len(generations) != 1:
        errors.append("generation_receipt_missing")
    else:
        generation = generations[0]
        raw_output = str(generation.get("raw_output", ""))
        raw_response = generation.get("raw_response", {})
        if generation.get("prompt") != CANARY_PROMPT or generation.get(
            "prompt_sha256"
        ) != sha256_text(CANARY_PROMPT):
            errors.append("prompt_hash_mismatch")
        payload = canary_payload()
        if generation.get("request") != payload or generation.get("request_sha256") != sha256_text(
            canonical_json(payload)
        ):
            errors.append("generation_request_mismatch")
        if not raw_output or generation.get("raw_output_sha256") != sha256_text(raw_output):
            errors.append("raw_output_missing_or_changed")
        if generation.get("parsed_output") != parse_canary_output(raw_output):
            errors.append("parsed_output_mismatch")
        if generation.get("raw_response_sha256") != sha256_text(canonical_json(raw_response)):
            errors.append("raw_response_hash_mismatch")
        if int(generation.get("completion_tokens", 0) or 0) <= 0:
            errors.append("completion_tokens_missing")
        if int(generation.get("total_tokens", 0) or 0) <= int(
            generation.get("completion_tokens", 0) or 0
        ):
            errors.append("prompt_tokens_missing")
        if float(generation.get("latency_s", 0.0) or 0.0) <= 0:
            errors.append("generation_latency_missing")
        if loads and generation.get("server_log_sha256") != loads[0].get("server_log_sha256"):
            errors.append("generation_log_link_mismatch")
    gpu_rows = list(artifact.get("gpu_telemetry_rows") or [])
    by_phase = {str(row.get("phase")): row for row in gpu_rows}
    if not {"before", "model_loaded", "after_generation", "after_teardown"}.issubset(by_phase):
        errors.append("gpu_phase_receipts_missing")
    else:
        loaded_gpu = by_phase["model_loaded"]
        after_gpu = by_phase["after_teardown"]
        if "RTX 3090" not in str(loaded_gpu.get("selected_gpu_name", "")):
            errors.append("selected_gpu_not_rtx_3090")
        if (
            loaded_gpu.get("task_pid_present") is not True
            or int(loaded_gpu.get("task_pid_memory_mb", 0) or 0) <= 0
        ):
            errors.append("task_owned_gpu_telemetry_missing")
        if (
            after_gpu.get("task_pid_present") is not False
            or int(after_gpu.get("task_pid_memory_mb", 0) or 0) != 0
        ):
            errors.append("task_owned_vram_not_released")
    leases = list(artifact.get("server_lease_rows") or [])
    if len(leases) != 1:
        errors.append("server_lease_receipt_missing")
    else:
        lease = leases[0]
        if (
            lease.get("owned_by_task") is not True
            or lease.get("recorded_identity_present") is not True
        ):
            errors.append("task_lease_ownership_missing")
        if lease.get("cleanup_bounded") is not True or lease.get("cleanup_leak_free") is not True:
            errors.append("task_cleanup_failed")
        if lease.get("pid_released") is not True:
            errors.append("task_pid_not_released")
        if lease.get("vram_released") is not True:
            errors.append("task_vram_not_released")
        if int(lease.get("unrelated_process_kill_count_delta", 0) or 0) != 0:
            errors.append("unrelated_process_signaled")
    if artifact.get("rows") != typed_rows(artifact):
        errors.append("typed_rows_mismatch")
    return list(dict.fromkeys(errors))


def finalize_artifact(
    artifact: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
) -> JsonDict:
    """Grant readiness only to complete bounded runtime and teardown evidence."""

    result = _terminal_copy(artifact, checks, duration_s=duration_s)
    result.update(
        {
            "status": "completed",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": "model_bounded_generation",
            "qwen38_runtime_ready_score": 1,
            "verdict_class": "positive",
            "honest_verdict": "positive_qwen38_runtime_ready_bounded_generation_only",
        }
    )
    result["rows"] = typed_rows(result)
    check_map = {
        str(row.get("check")): row.get("passed") for row in result["preconditions_checked"]
    }
    errors = [
        f"readiness_check_failed:{name}"
        for name in READINESS_CHECKS
        if check_map.get(name) is not True
    ]
    errors.extend(_positive_evidence_errors(result))
    if errors:
        raise ValueError(f"positive runtime evidence is incomplete: {errors}")
    result["reproducibility_checksum"] = artifact_checksum(result)
    return result


def _load_artifact(value: Mapping[str, Any] | str | Path | object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.is_file():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"__unreadable__": True}
        return loaded if isinstance(loaded, Mapping) else {"__not_object__": True}
    return {"__not_object__": True}


def validate_artifact(value: Mapping[str, Any] | str | Path | object) -> list[str]:
    """Cold-check terminal state, bounded scope, evidence links, and checksum."""

    artifact = _load_artifact(value)
    if artifact is None:
        return ["artifact_missing"]
    if artifact.get("__unreadable__"):
        return ["artifact_unreadable"]
    if artifact.get("__not_object__"):
        return ["artifact_not_object"]
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        return ["artifact_fields_mismatch"]
    errors: list[str] = []
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_mismatch")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    verdict = artifact.get("verdict_class")
    if verdict not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    if not str(artifact.get("honest_verdict", "")).startswith(f"{verdict}_"):
        errors.append("honest_verdict_prefix_mismatch")
    if "quality" in str(artifact.get("honest_verdict", "")).lower():
        errors.append("quality_claim_present")
    checks = list(artifact.get("preconditions_checked") or [])
    if artifact.get("gate_check_summary") != _gate_summary(checks):
        errors.append("gate_check_summary_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if verdict == "positive":
        if artifact.get("status") != "completed":
            errors.append("positive_status")
        if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
            errors.append("positive_inference_substrate")
        if artifact.get("inference_substrate_class") != "model_bounded_generation":
            errors.append("positive_substrate_class")
        if (
            type(artifact.get("qwen38_runtime_ready_score")) is not int
            or artifact.get("qwen38_runtime_ready_score") != 1
        ):
            errors.append("positive_readiness_score")
        if float(artifact.get("duration_s", 0.0) or 0.0) < BOUNDED_DURATION_FLOOR_S:
            errors.append("bounded_duration_floor")
        if artifact.get("gate_check_summary", {}).get("passed") is not True:
            errors.append("positive_gate_summary")
        errors.extend(_positive_evidence_errors(artifact))
    elif verdict == "blocked":
        if artifact.get("status") != "blocked":
            errors.append("blocked_status")
        if artifact.get("inference_substrate") != "no_inference":
            errors.append("blocked_inference_substrate")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class")
        if artifact.get("qwen38_runtime_ready_score") != 0:
            errors.append("blocked_readiness_score")
        if artifact.get("gate_check_summary", {}).get("passed") is not False:
            errors.append("blocked_gate_summary")
    elif verdict == "disqualified":
        if artifact.get("status") != "disqualified":
            errors.append("disqualified_status")
        if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
            errors.append("disqualified_inference_substrate")
        if artifact.get("inference_substrate_class") != "model_bounded_generation":
            errors.append("disqualified_substrate_class")
        if artifact.get("qwen38_runtime_ready_score") != 0:
            errors.append("disqualified_readiness_score")
        if artifact.get("gate_check_summary", {}).get("passed") is not False:
            errors.append("disqualified_gate_summary")
    else:
        errors.append("terminal_verdict_class_invalid")
    return list(dict.fromkeys(errors))


def _progress(phase: int, event: str, **fields: Any) -> None:  # pragma: no cover
    print(canonical_json({"phase": phase, "event": event, **fields}), flush=True)


def _run_subprocess(
    command: list[str], *, timeout_s: float = 15.0, phase: int
) -> JsonDict:  # pragma: no cover
    _progress(phase, "subprocess_start", command=command)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        row = {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "duration_s": time.perf_counter() - started,
        }
    except Exception as exc:
        row = {
            "returncode": 127,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": time.perf_counter() - started,
        }
    _progress(phase, "subprocess_end", command=command, returncode=row["returncode"])
    return row


def _gpu_snapshot(
    phase_name: str,
    *,
    phase: int,
    selected_gpu: int | None = None,
    task_pid: int | None = None,
) -> JsonDict:  # pragma: no cover
    gpu = _run_subprocess(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=15,
        phase=phase,
    )
    apps = _run_subprocess(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=15,
        phase=phase,
    )
    devices: list[JsonDict] = []
    uuid_to_index: dict[str, int] = {}
    for line in str(gpu.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 7:
            index = int(parts[0])
            uuid_to_index[parts[1]] = index
            devices.append(
                {
                    "index": index,
                    "uuid": parts[1],
                    "name": parts[2],
                    "utilization_pct": int(float(parts[3])),
                    "memory_total_mb": int(float(parts[4])),
                    "memory_used_mb": int(float(parts[5])),
                    "memory_free_mb": int(float(parts[6])),
                }
            )
    compute_apps: list[JsonDict] = []
    for line in str(apps.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4 and parts[1].isdigit():
            compute_apps.append(
                {
                    "gpu_uuid": parts[0],
                    "gpu_index": uuid_to_index.get(parts[0]),
                    "pid": int(parts[1]),
                    "process_name": parts[2],
                    "used_memory_mb": int(float(parts[3])),
                }
            )
    selected = next(
        (row for row in devices if row["index"] == selected_gpu),
        {},
    )
    task_apps = [row for row in compute_apps if row["pid"] == task_pid]
    return {
        "phase": phase_name,
        "query_ok": gpu.get("returncode") == 0 and bool(devices),
        "devices": devices,
        "compute_apps": compute_apps,
        "selected_gpu_index": selected_gpu,
        "selected_gpu_name": selected.get("name"),
        "selected_gpu_memory_used_mb": selected.get("memory_used_mb"),
        "selected_gpu_memory_free_mb": selected.get("memory_free_mb"),
        "selected_gpu_utilization_pct": selected.get("utilization_pct"),
        "task_pid": task_pid,
        "task_pid_memory_mb": sum(int(row["used_memory_mb"]) for row in task_apps),
        "task_pid_present": bool(task_apps),
        "command_receipts": {"gpu": gpu, "compute_apps": apps},
    }


def _idle_rtx_3090_indices(snapshot: Mapping[str, Any]) -> list[int]:  # pragma: no cover
    busy = {
        int(row["gpu_index"])
        for row in snapshot.get("compute_apps", [])
        if row.get("gpu_index") is not None
    }
    return [
        int(row["index"])
        for row in snapshot.get("devices", [])
        if "RTX 3090" in str(row.get("name", ""))
        and int(row["index"]) not in busy
        and int(row.get("memory_free_mb", 0) or 0) >= 20_000
    ]


def _binary_linkage(server_path: Path) -> JsonDict:  # pragma: no cover
    linked = _run_subprocess(["ldd", str(server_path)], timeout_s=15, phase=3)
    version = _run_subprocess([str(server_path), "--version"], timeout_s=15, phase=3)
    link_text = f"{linked.get('stdout', '')}\n{linked.get('stderr', '')}".lower()
    libggml = "libggml-cuda" in link_text
    libcuda = "libcuda.so" in link_text
    return {
        "server_path": str(server_path),
        "server_exists": server_path.is_file(),
        "ldd_receipt": linked,
        "version_receipt": version,
        "libggml_cuda_linked": libggml,
        "libcuda_linked": libcuda,
        "cuda_linkage_confirmed": linked.get("returncode") == 0 and libggml and libcuda,
    }


def _server_command(server: Path, model: Path, port: int) -> list[str]:  # pragma: no cover
    return [
        str(server),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "2048",
        "--n-gpu-layers",
        "all",
        "--split-mode",
        "none",
        "--parallel",
        "1",
        "--batch-size",
        "256",
        "--ubatch-size",
        "256",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--fit",
        "off",
        "--offline",
        "--jinja",
        "--reasoning",
        "off",
        "--no-webui",
        "--log-verbosity",
        "3",
    ]


def _wait_for_health(
    supervisor: NativeLlamaServerSupervisor,
    port: int,
    *,
    timeout_s: float,
) -> JsonDict:  # pragma: no cover
    started = time.perf_counter()
    deadline = time.monotonic() + timeout_s
    next_heartbeat = time.monotonic()
    attempts = 0
    last_error = "not_started"
    while time.monotonic() < deadline:
        attempts += 1
        if supervisor.proc and supervisor.proc.poll() is not None:
            return {
                "ok": False,
                "attempts": attempts,
                "classification": "early_exit",
                "last_error": last_error,
                "duration_s": time.perf_counter() - started,
            }
        try:
            with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                return {
                    "ok": response.status == 200,
                    "attempts": attempts,
                    "classification": "healthy",
                    "status": response.status,
                    "duration_s": time.perf_counter() - started,
                }
        except (OSError, error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if time.monotonic() >= next_heartbeat:
            _progress(
                5,
                "model_load_heartbeat",
                attempts=attempts,
                elapsed_s=round(time.perf_counter() - started, 3),
            )
            next_heartbeat = time.monotonic() + 60
        time.sleep(1)
    return {
        "ok": False,
        "attempts": attempts,
        "classification": "deadline_expired",
        "last_error": last_error,
        "duration_s": time.perf_counter() - started,
    }


def _request_canary(port: int) -> JsonDict:  # pragma: no cover
    payload = canary_payload()
    http_request = request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with request.urlopen(http_request, timeout=240) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = list(body.get("choices") or [{}])
    message = dict(choices[0].get("message") or {})
    raw_output = str(message.get("content") or message.get("reasoning_content") or "")
    usage = dict(body.get("usage") or {})
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    return {
        "body": body,
        "raw_output": raw_output,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(usage.get("total_tokens", 0) or 0) or prompt_tokens + completion_tokens,
        "latency_s": time.perf_counter() - started,
    }


def _launch_for_gpu(
    supervisor: NativeLlamaServerSupervisor, gpu_index: int
) -> JsonDict:  # pragma: no cover
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    try:
        return supervisor.launch()
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _run_canary(
    spec: Mapping[str, Any],
    *,
    server_path: Path,
    binary_linkage: Mapping[str, Any],
    selected_gpu: int,
    before_gpu: Mapping[str, Any],
    raw_dir: Path,
) -> tuple[JsonDict, JsonDict, list[JsonDict], JsonDict, str | None]:  # pragma: no cover
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    command = _server_command(server_path, Path(str(spec["model_path"])), port)
    contract = supervisor_contract(
        outer_deadline_s=900,
        health_timeout_s=480,
        token_timeout_s=240,
        cleanup_grace_s=30,
        kill_after_cleanup_timeout_s=10,
        retry_budget=0,
        endurance_interval_s=0,
        endurance_sample_count=1,
    )
    supervisor = NativeLlamaServerSupervisor(command, raw_dir, contract)
    identity: JsonDict = {}
    health: JsonDict = {"ok": False, "classification": "not_started"}
    response: JsonDict = {}
    telemetry: list[JsonDict] = []
    cleanup: JsonDict = {"action": "not_started", "bounded": True, "leak_free": True}
    failure: str | None = None
    startup_started = time.perf_counter()
    startup_duration_s = 0.0
    _progress(5, "model_load_start", model_id=QWEN_MODEL_ID, command=command)
    try:
        identity = _launch_for_gpu(supervisor, selected_gpu)
        health = _wait_for_health(supervisor, port, timeout_s=480)
        startup_duration_s = time.perf_counter() - startup_started
        _progress(
            5,
            "model_load_end",
            model_id=QWEN_MODEL_ID,
            pid=identity.get("pid"),
            health_ok=health.get("ok"),
        )
        telemetry.append(
            _gpu_snapshot(
                "model_loaded",
                phase=5,
                selected_gpu=selected_gpu,
                task_pid=int(identity.get("pid", -1)),
            )
        )
        if health.get("ok") is not True:
            raise RuntimeError(f"server health failed: {health.get('classification')}")
        _progress(5, "generation_start", model_id=QWEN_MODEL_ID)
        response = _request_canary(port)
        _progress(
            5,
            "generation_end",
            model_id=QWEN_MODEL_ID,
            completion_tokens=response.get("completion_tokens"),
        )
        telemetry.append(
            _gpu_snapshot(
                "after_generation",
                phase=5,
                selected_gpu=selected_gpu,
                task_pid=int(identity.get("pid", -1)),
            )
        )
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        _progress(5, "runtime_failure", error=failure)
    finally:
        _progress(6, "teardown_start", pid=identity.get("pid"))
        cleanup = supervisor.cleanup()
        if supervisor.proc is not None:
            try:
                supervisor.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        after = _gpu_snapshot(
            "after_teardown",
            phase=6,
            selected_gpu=selected_gpu,
            task_pid=int(identity.get("pid", -1)),
        )
        telemetry.append(after)
        _progress(
            6,
            "teardown_end",
            pid=identity.get("pid"),
            leak_free=cleanup.get("leak_free"),
            task_pid_present=after.get("task_pid_present"),
        )
    server_log = supervisor.stderr_tail(limit=2_000_000)
    server_log_hash = sha256_text(server_log)
    markers = [marker for marker in ("CUDA0", "CUDA : ARCHS") if marker in server_log]
    loaded = next((row for row in telemetry if row.get("phase") == "model_loaded"), {})
    task_vram = int(loaded.get("task_pid_memory_mb", 0) or 0)
    pid = int(identity.get("pid", -1))
    load_receipt = {
        "model_id": QWEN_MODEL_ID,
        "model_sha256": spec.get("sha256"),
        "loaded_path": spec.get("loaded_path"),
        "command": command,
        "pid": pid,
        "process_identity": identity,
        "pid_owned_by_task": identity.get("owned_by_task") is True,
        "health": health,
        "health_ok": health.get("ok") is True,
        "startup_duration_s": startup_duration_s,
        "binary_linkage": deepcopy(dict(binary_linkage)),
        "native_cuda_markers": markers,
        "requested_gpu_layers": "all",
        "selected_gpu_index": selected_gpu,
        "task_owned_vram_delta_mb": task_vram,
        "server_log_path": str(supervisor.log_path),
        "server_log": server_log,
        "server_log_sha256": server_log_hash,
        "error": failure,
    }
    body = response.get("body", {})
    raw_output = str(response.get("raw_output", ""))
    generation_receipt = {
        "model_id": QWEN_MODEL_ID,
        "prompt": CANARY_PROMPT,
        "prompt_sha256": sha256_text(CANARY_PROMPT),
        "request": canary_payload(),
        "request_sha256": sha256_text(canonical_json(canary_payload())),
        "raw_output": raw_output,
        "raw_output_sha256": sha256_text(raw_output),
        "parsed_output": parse_canary_output(raw_output),
        "raw_response": body,
        "raw_response_sha256": sha256_text(canonical_json(body)),
        "prompt_tokens": int(response.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(response.get("completion_tokens", 0) or 0),
        "total_tokens": int(response.get("total_tokens", 0) or 0),
        "latency_s": float(response.get("latency_s", 0.0) or 0.0),
        "gpu_snapshot_phase": "after_generation",
        "server_log_sha256": server_log_hash,
        "error": failure,
    }
    before_external = {
        int(row["pid"])
        for row in before_gpu.get("compute_apps", [])
        if int(row.get("pid", -1)) != pid
    }
    after_external = {
        int(row["pid"])
        for row in telemetry[-1].get("compute_apps", [])
        if int(row.get("pid", -1)) != pid
    }
    lease = {
        "pid": pid,
        "owned_by_task": identity.get("owned_by_task") is True,
        "recorded_identity_present": bool(identity),
        "cleanup_action": cleanup.get("action"),
        "cleanup_bounded": cleanup.get("bounded") is True,
        "cleanup_leak_free": cleanup.get("leak_free") is True,
        "signals_sent": deepcopy(cleanup.get("signals_sent", [])),
        "signaled_pid": pid if cleanup.get("signals_sent") else None,
        "pid_released": pid <= 0 or not Path(f"/proc/{pid}").exists(),
        "vram_released": telemetry[-1].get("task_pid_present") is False,
        "unrelated_process_kill_count_delta": int(
            cleanup.get("unrelated_process_kill_count_delta", 0) or 0
        ),
        "unrelated_pids_before": sorted(before_external),
        "unrelated_pids_after": sorted(after_external),
    }
    return load_receipt, generation_receipt, telemetry, lease, failure


def _checkpoint(path: Path, artifact: JsonDict) -> None:  # pragma: no cover
    artifact["rows"] = typed_rows(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    write_artifact(path, artifact)


def run_experiment(
    *, root: Path, run_date: str, result_path: Path, raw_dir: Path
) -> JsonDict:  # pragma: no cover
    """Check resources, invoke one owned server, and always finish terminal."""

    started = time.perf_counter()
    checks: list[JsonDict] = []
    _progress(0, "phase_start", name="schema_complete_first_write")
    artifact = initialize_artifact(result_path, run_date)
    _progress(0, "phase_end", name="schema_complete_first_write", path=str(result_path))

    _progress(1, "phase_start", name="date_sources_and_output")
    source_exists = {str(path): (root / path).is_file() for path in REQUIRED_SOURCE_PATHS}
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_observed = {
        "path": str(result_path),
        "exists_after_first_write": result_path.is_file(),
        "parent_writable": os.access(result_path.parent, os.W_OK),
        "raw_dir_writable": os.access(raw_dir, os.W_OK),
        "raw_dir_entries": sorted(path.name for path in raw_dir.iterdir()),
    }
    output_expected = {
        "path": str(root / RESULT_PATH),
        "exists_after_first_write": True,
        "parent_writable": True,
        "raw_dir_writable": True,
        "raw_dir_entries": [],
    }
    artifact["source_artifact_hashes"] = deepcopy(REQUIRED_SOURCE_HASHES)
    checks.extend(
        [
            gate_row("run_date", RUN_DATE, run_date, run_date == RUN_DATE),
            gate_row(
                "source_paths",
                {str(path): True for path in REQUIRED_SOURCE_PATHS},
                source_exists,
                all(source_exists.values()),
            ),
            gate_row(
                "output_path",
                output_expected,
                output_observed,
                output_observed == output_expected,
            ),
        ]
    )
    _checkpoint(result_path, artifact)
    _progress(
        1, "phase_end", name="date_sources_and_output", passed=all(row["passed"] for row in checks)
    )
    if any(row["passed"] is not True for row in checks):
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(2, "phase_start", name="registry_cutover")
    registry_rows = registry_cutover_rows()
    registry_ok = (
        len(SOTA_GGUF_MODELS) == 1
        and SOTA_GGUF_MODELS[0]["hf_id"] == QWEN_MODEL_ID
        and all(
            row["mandate_status"] == "legacy_comparator" for row in LEGACY_COMPARATOR_GGUF_MODELS
        )
    )
    checks.append(
        gate_row(
            "registry_cutover",
            {"current": [QWEN_MODEL_ID], "comparators": list(LEGACY_COMPARATOR_IDS)},
            {
                "current": [row["hf_id"] for row in SOTA_GGUF_MODELS],
                "comparators": [row["hf_id"] for row in LEGACY_COMPARATOR_GGUF_MODELS],
            },
            registry_ok,
        )
    )
    artifact["registry_cutover_rows"] = registry_rows
    _checkpoint(result_path, artifact)
    _progress(2, "phase_end", name="registry_cutover", passed=registry_ok)
    if not registry_ok:
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(3, "phase_start", name="gpu_and_cuda_runtime")
    before_gpu = _gpu_snapshot("before", phase=3)
    idle_indices = _idle_rtx_3090_indices(before_gpu)
    selected_gpu = idle_indices[0] if idle_indices else None
    if selected_gpu is not None:
        before_gpu = _gpu_snapshot("before", phase=3, selected_gpu=selected_gpu)
    artifact["gpu_telemetry_rows"] = [before_gpu]
    server_path = resolve_native_llama_server()
    linkage = _binary_linkage(server_path)
    checks.extend(
        [
            gate_row(
                "nvidia_smi", True, before_gpu.get("query_ok"), before_gpu.get("query_ok") is True
            ),
            gate_row(
                "idle_rtx_3090",
                {"minimum_count": 1},
                {"indices": idle_indices, "count": len(idle_indices)},
                bool(idle_indices),
            ),
            gate_row(
                "cuda_llama_runtime",
                {"server_exists": True, "cuda_linkage_confirmed": True},
                {
                    "server_path": str(server_path),
                    "server_exists": linkage["server_exists"],
                    "cuda_linkage_confirmed": linkage["cuda_linkage_confirmed"],
                },
                linkage["server_exists"] and linkage["cuda_linkage_confirmed"],
            ),
        ]
    )
    _checkpoint(result_path, artifact)
    gpu_cuda_ok = all(row["passed"] for row in checks)
    _progress(
        3, "phase_end", name="gpu_and_cuda_runtime", passed=gpu_cuda_ok, selected_gpu=selected_gpu
    )
    if not gpu_cuda_ok or selected_gpu is None:
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(4, "phase_start", name="exact_cached_model_identity")
    resolved = cached_current_model(gpu_index=selected_gpu, preferred_quant=PREFERRED_QUANT)
    cache_observed = {
        "resolved": resolved is not None,
        "hf_id": resolved.get("hf_id") if resolved else None,
        "path": resolved.get("model_path") if resolved else None,
        "filename": Path(str(resolved.get("model_path"))).name if resolved else None,
        "exists": Path(str(resolved.get("model_path"))).is_file() if resolved else False,
    }
    cache_ok = (
        resolved is not None
        and resolved.get("hf_id") == QWEN_MODEL_ID
        and Path(str(resolved.get("model_path"))).name == QWEN_FILENAME
        and Path(str(resolved.get("model_path"))).is_file()
    )
    checks.append(
        gate_row(
            "cached_qwen38_q4",
            {"hf_id": QWEN_MODEL_ID, "filename": QWEN_FILENAME, "exists": True},
            cache_observed,
            cache_ok,
        )
    )
    if not cache_ok or resolved is None:
        _progress(4, "phase_end", name="exact_cached_model_identity", passed=False)
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )
    model_path = Path(str(resolved["model_path"]))
    _progress(4, "benchmark_start", name="model_hash_and_embedded_template", path=str(model_path))
    metadata = dict(read_gguf_metadata(model_path))
    model_hash = sha256_file(model_path)
    _progress(4, "benchmark_end", name="model_hash_and_embedded_template", path=str(model_path))
    template_present = bool(metadata.get("chat_template_present"))
    identity = {
        "model_id": QWEN_MODEL_ID,
        "repository": QWEN_MODEL_ID,
        "filename": model_path.name,
        "model_path": str(model_path),
        "loaded_path": str(model_path.resolve()),
        "revision": snapshot_revision(model_path),
        "size_bytes": model_path.stat().st_size,
        "sha256": model_hash,
        "template_source": "embedded_gguf",
        "template_sha256": metadata.get("chat_template_sha256"),
        "template_present": template_present,
        "template_metadata_keys": list(metadata.get("metadata_keys", [])),
        "template_detail": metadata.get("tokenizer_detail"),
    }
    model_spec = {
        **deepcopy(HEADLINE_MODEL_SPEC),
        **deepcopy(resolved),
        "preferred_quant": PREFERRED_QUANT,
        "remote_allowed": False,
        "resolution_method": "cached_current_model",
        "loaded_path": identity["loaded_path"],
        "revision": identity["revision"],
        "size_bytes": identity["size_bytes"],
        "sha256": model_hash,
        "chat_template_source": "embedded_gguf",
        "chat_template_sha256": identity["template_sha256"],
    }
    artifact["MODEL_SPECS"] = [model_spec]
    artifact["model_identity_rows"] = [identity]
    checks.append(gate_row("embedded_gguf_template", True, template_present, template_present))
    _checkpoint(result_path, artifact)
    _progress(4, "phase_end", name="exact_cached_model_identity", passed=template_present)
    if not template_present:
        return finish_blocked(
            artifact, result_path, checks, duration_s=time.perf_counter() - started
        )

    _progress(5, "phase_start", name="bounded_generation")
    artifact["inference_substrate"] = INFERENCE_SUBSTRATE
    artifact["inference_substrate_class"] = "model_bounded_generation"
    _checkpoint(result_path, artifact)
    load, generation, telemetry, lease, failure = _run_canary(
        model_spec,
        server_path=server_path,
        binary_linkage=linkage,
        selected_gpu=selected_gpu,
        before_gpu=before_gpu,
        raw_dir=raw_dir,
    )
    artifact["model_load_receipts"] = [load]
    artifact["generation_receipts"] = [generation]
    artifact["gpu_telemetry_rows"].extend(telemetry)
    artifact["server_lease_rows"] = [lease]
    checks.extend(
        [
            gate_row("model_load", True, load["health_ok"], load["health_ok"] is True),
            gate_row(
                "native_cuda_markers",
                ["CUDA0", "CUDA : ARCHS"],
                load["native_cuda_markers"],
                {"CUDA0", "CUDA : ARCHS"}.issubset(load["native_cuda_markers"]),
            ),
            gate_row(
                "task_owned_vram",
                {"minimum_delta_mb": 1},
                {"delta_mb": load["task_owned_vram_delta_mb"]},
                int(load["task_owned_vram_delta_mb"]) > 0,
            ),
            gate_row(
                "bounded_generation",
                {"completion_tokens_min": 1, "error": None},
                {
                    "completion_tokens": generation["completion_tokens"],
                    "error": failure,
                },
                generation["completion_tokens"] > 0 and failure is None,
            ),
            gate_row(
                "task_owned_teardown",
                {"pid_released": True, "vram_released": True, "unrelated_signals": 0},
                {
                    "pid_released": lease["pid_released"],
                    "vram_released": lease["vram_released"],
                    "unrelated_signals": lease["unrelated_process_kill_count_delta"],
                },
                lease["pid_released"]
                and lease["vram_released"]
                and lease["unrelated_process_kill_count_delta"] == 0,
            ),
        ]
    )
    _checkpoint(result_path, artifact)
    _progress(
        5, "phase_end", name="bounded_generation", passed=all(row["passed"] for row in checks)
    )

    _progress(7, "phase_start", name="terminal_reduction")
    duration = time.perf_counter() - started
    checks.append(
        gate_row(
            "bounded_duration_floor",
            {"minimum_s": BOUNDED_DURATION_FLOOR_S},
            {"duration_s": duration},
            duration >= BOUNDED_DURATION_FLOOR_S,
        )
    )
    if any(row["passed"] is not True for row in checks):
        result = finish_disqualified(artifact, result_path, checks, duration_s=duration)
    else:
        try:
            result = finalize_artifact(artifact, checks, duration_s=duration)
            write_artifact(result_path, result)
        except ValueError as exc:
            checks.append(gate_row("terminal_evidence", [], [str(exc)], False))
            result = finish_disqualified(artifact, result_path, checks, duration_s=duration)
    _progress(
        7,
        "phase_end",
        name="terminal_reduction",
        status=result["status"],
        readiness=result["qwen38_runtime_ready_score"],
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args(argv)
    if args.validate is not None:
        _progress(8, "artifact_validator_start", path=str(args.validate))
        errors = validate_artifact(args.validate)
        _progress(8, "artifact_validator_end", valid=not errors)
        print(canonical_json({"valid": not errors, "errors": errors}), flush=True)
        return int(bool(errors))
    root = find_repo_root()
    result_path = args.result_path if args.result_path.is_absolute() else root / args.result_path
    try:
        result = run_experiment(
            root=root,
            run_date=args.date,
            result_path=result_path,
            raw_dir=root / RAW_DIR,
        )
    except Exception as exc:
        _progress(7, "unexpected_failure", error=f"{type(exc).__name__}: {exc}")
        if result_path.is_file():
            loaded = _load_artifact(result_path)
            artifact = (
                dict(loaded)
                if loaded and "__unreadable__" not in loaded
                else base_artifact(args.date)
            )
        else:
            artifact = base_artifact(args.date)
        checks = list(artifact.get("preconditions_checked") or [])
        checks.append(
            gate_row("unexpected_failure", "no_exception", f"{type(exc).__name__}: {exc}", False)
        )
        invoked = (
            bool(artifact.get("model_load_receipts"))
            or artifact.get("inference_substrate_class") == "model_bounded_generation"
        )
        result = (
            finish_disqualified(
                artifact, result_path, checks, duration_s=float(artifact.get("duration_s", 0) or 0)
            )
            if invoked
            else finish_blocked(
                artifact, result_path, checks, duration_s=float(artifact.get("duration_s", 0) or 0)
            )
        )
    _progress(8, "artifact_validator_start", path=str(result_path))
    errors = validate_artifact(result)
    _progress(8, "artifact_validator_end", valid=not errors)
    print(
        canonical_json(
            {
                "artifact": str(result_path),
                "valid": not errors,
                "errors": errors,
                "verdict_class": result.get("verdict_class"),
                "qwen38_runtime_ready_score": result.get("qwen38_runtime_ready_score"),
            }
        ),
        flush=True,
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
