#!/usr/bin/env python3
"""Exp 5337: clean SOTA GGUF runtime corrigendum receipt.

Spec refs: REQ-VERIFY-5337, SCENARIO-VERIFY-5337.

This module is a runtime-methodology corrigendum only. It replays the stable
native llama.cpp command from Exp 5324 for the flagship dense GGUF model and
requires a real live-inference duration of at least 60 seconds. It records
optional mandated-model probe blockers, but it never claims model quality,
answer accuracy, verifier quality, solver quality, benchmark improvement, or
memory usefulness.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5323_native_gguf_backend_flag_bisect_v486 as exp5323
from carnot import experiment_5324_runtime_receipt_stabilization_v486 as exp5324
from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
PreconditionsProvider = Callable[[], JsonDict]
RuntimeProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5337_sota_runtime_corrigendum_multimodel_v487"
MILESTONE = "2026.07.487"
RESULT_RELATIVE_PATH = Path("results/experiment_5337_sota_runtime_corrigendum_multimodel_v487.json")
SCHEMA = "carnot.experiment_5337.sota_runtime_corrigendum_multimodel.v487"
INFERENCE_SUBSTRATE = "live_llm_inference"
SPEC_REFS = ("REQ-VERIFY-5337", "SCENARIO-VERIFY-5337")

MANDATED_MODEL_SPECS = exp5323.MANDATED_MODEL_SPECS
PROMPT = exp5324.PROMPT
RANDOM_SEED = 5337
MIN_CLEAN_DURATION_S = 60.0
DEFAULT_REPEATS = 3
TERMINAL_PREFIXES = ("complete:", "blocked_")
MISSING_WRAPPED_VALUE = object()

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Traceability for the Exp5337 clean SOTA runtime corrigendum receipt.",
    "milestone": "Milestone accountability for the V487 runtime-corrigendum gate.",
    "status": "Machine-readable terminal state for downstream SOTA runtime gates.",
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether a "
        "clean unflagged-duration live GGUF receipt exists."
    ),
    "inference_substrate": (
        "Declares live_llm_inference so adversarial verification applies the correct "
        "60-second runtime floor to the cleaned receipt."
    ),
    "MODEL_SPECS": (
        "Records the three mandated SOTA GGUF model IDs, local file receipts, roles, "
        "and no-AutoTokenizer status."
    ),
    "preconditions_checked": (
        "Records GPU visibility, nvidia-smi, llama.cpp version, model file presence, "
        "free VRAM, offload evidence, and the exact Exp5324 command before live "
        "inference."
    ),
    "selected_backend_command": (
        "Preserves the Exp5324 stable command shape and records any bounded "
        "token-budget increase needed for clean duration."
    ),
    "runtime_corrigendum_receipt": (
        "Records the load, first-token, duration, token, GPU-memory, offload, "
        "stdout/stderr, and answer-separation evidence for the cleaned dense receipt."
    ),
    "multi_model_receipt_matrix": (
        "Separates completed, skipped, and blocked receipts for all mandated SOTA "
        "models without turning optional probes into quality claims."
    ),
    "tests_run": (
        "Commands run to validate the corrigendum module, artifact schema, new-code "
        "coverage, repository tests, and applicable runtime checks."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "MODEL_SPECS",
    "preconditions_checked",
    "selected_backend_command",
    "runtime_corrigendum_receipt",
    "multi_model_receipt_matrix",
    "methodology_duration_s",
    "sota_runtime_clean_receipt_ready",
    "runtime_unblocked_min_one_mandated",
    "quality_claim_permitted",
    "no_autotokenizer_used",
    "tests_run",
)
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "MODEL_SPECS",
    "preconditions_checked",
    "selected_backend_command",
    "runtime_corrigendum_receipt",
    "multi_model_receipt_matrix",
    "tests_run",
)


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _raw_or_wrapped_value(payload: Mapping[str, Any], field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _prior_stable_command(prior_artifact: Mapping[str, Any]) -> JsonDict | None:
    if _raw_or_wrapped_value(prior_artifact, "sota_runtime_unblocked_stable") is not True:
        return None
    candidate = _raw_or_wrapped_value(prior_artifact, "selected_backend_command")
    if not isinstance(candidate, Mapping):
        return None
    command = candidate.get("command")
    if not isinstance(command, list) or not command:
        return None
    if str(candidate.get("model_role") or "") != "flagship_dense":
        return None
    return dict(candidate)


def _resolve_model_specs(model_resolver: ModelResolver) -> JsonDict:
    return {
        str(spec["role"]): exp5323._resolve_model_spec(spec, model_resolver)
        for spec in MANDATED_MODEL_SPECS
    }


def _selected_model_spec(candidate: Mapping[str, Any] | None, model_specs: Mapping[str, Any]) -> JsonDict | None:
    if candidate is None:
        return None
    role = str(candidate.get("model_role") or "")
    selected = model_specs.get(role)
    return dict(selected) if isinstance(selected, Mapping) else None


def _candidate_command(candidate: Mapping[str, Any] | None) -> list[str] | None:
    if candidate is None:
        return None
    command = candidate.get("command")
    return list(command) if isinstance(command, list) and command else None


def _build_replay_variant(candidate: Mapping[str, Any]) -> JsonDict:
    command = list(candidate["command"])
    return {
        "name": str(candidate.get("backend_variant") or "exp5324-selected-command"),
        "backend_kind": str(candidate.get("backend_kind") or Path(command[0]).name),
        "command": command,
        "model_path": str(candidate.get("model_path") or ""),
        "context": int(candidate.get("context") or exp5323.DEFAULT_CONTEXT),
        "batch": int(candidate.get("batch") or exp5323.DEFAULT_BATCH),
        "ubatch": int(candidate.get("ubatch") or exp5323.DEFAULT_UBATCH),
        "gpu_layers": str(candidate.get("gpu_layers") or exp5323.DEFAULT_GPU_LAYERS),
        "split_mode": exp5323.DEFAULT_SPLIT_MODE,
        "tensor_split": candidate.get("tensor_split"),
        "prompt": str(candidate.get("prompt") or PROMPT),
        "n_predict": int(candidate.get("n_predict") or exp5323.N_PREDICT),
        "timeout_s": float(candidate.get("timeout_s") or exp5323.DEFAULT_TIMEOUT_S),
    }


def _stderr_summary(text: str) -> str:
    return "\n".join(str(text).strip().splitlines()[-12:])[-1200:]


def _stdout_summary(text: str) -> str:
    lines = str(text).strip().splitlines()
    return "\n".join(lines[-24:])[-2000:]


def answer_text_separable_from_thinking_text(text: str) -> bool:
    lowered = text.lower()
    thinking_positions = [
        pos
        for marker in ("[start thinking]", "<think>", "thinking")
        if (pos := lowered.find(marker)) >= 0
    ]
    final_positions = [
        pos
        for marker in ("[final answer]", "</think>", "\nanswer:", "final answer")
        if (pos := lowered.find(marker)) >= 0
    ]
    return bool(thinking_positions and final_positions and min(final_positions) > min(thinking_positions))


def _extract_final_answer_text(text: str) -> str | None:
    lowered = text.lower()
    for marker in ("[final answer]", "\nanswer:", "final answer"):
        idx = lowered.find(marker)
        if idx >= 0:
            return text[idx + len(marker) :].strip()[:1000] or None
    end_think = lowered.find("</think>")
    if end_think >= 0:
        return text[end_think + len("</think>") :].strip()[:1000] or None
    return None


def classify_clean_receipt(receipt: Mapping[str, Any]) -> str:
    timeout_class = str(receipt.get("timeout_class") or "")
    if timeout_class.startswith("timeout") or receipt.get("timed_out"):
        return "timeout"
    if timeout_class in {"llama_context_batch_assert", "native_llama_cpp_abort_signal"}:
        return "runtime_crash"
    if not receipt.get("completed_load_first_token_and_8_tokens"):
        return "generation_incomplete"
    if not receipt.get("offload_authenticated"):
        return "offload_not_authenticated"
    if timeout_class and timeout_class != "completed_no_timeout":
        return timeout_class
    return "ready"


def _receipt_ready(receipt: Mapping[str, Any]) -> bool:
    return classify_clean_receipt(receipt) == "ready"


def _normalise_replay_receipt(
    receipt: Mapping[str, Any],
    *,
    model_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    timeout_s: float,
    run_index: int,
) -> JsonDict:
    attempt = exp5324._normalise_replay_receipt(
        receipt,
        model_spec=model_spec,
        variant=variant,
        timeout_s=timeout_s,
        run_index=run_index,
    )
    stdout_tail = str(attempt.get("stdout_tail", ""))
    stderr_tail = str(attempt.get("stderr_tail", ""))
    attempt["stdout_summary"] = _stdout_summary(stdout_tail)
    attempt["stderr_summary"] = _stderr_summary(stderr_tail)
    attempt["answer_text_separable_from_thinking_text"] = answer_text_separable_from_thinking_text(
        stdout_tail
    )
    attempt["final_answer_text"] = _extract_final_answer_text(stdout_tail)
    return attempt


def _precondition_blockers(
    *,
    preconditions: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    selected_model: Mapping[str, Any] | None,
) -> list[str]:
    blockers = list(preconditions.get("blocked_preconditions", []))
    command = _candidate_command(candidate)
    if candidate is None or command is None:
        blockers.append("exp5324_stable_command_unavailable")
    if not preconditions.get("gpu_visible"):
        blockers.append("gpu_not_visible")
    if int(preconditions.get("free_vram_mb") or 0) <= 0:
        blockers.append("free_vram_unavailable")
    if candidate is not None:
        prior_delta = int(candidate.get("gpu_memory_delta_mb") or 0)
        if prior_delta > 0 and int(preconditions.get("free_vram_mb") or 0) < prior_delta:
            blockers.append("free_vram_below_exp5324_delta")
    if command is not None and not Path(str(command[0])).is_file():
        blockers.append("selected_binary_missing")
    if selected_model is None or selected_model.get("status") != "local_gguf_resolved":
        blockers.append("selected_model_file_missing")
    else:
        selected_path = str(selected_model.get("model_path") or "")
        candidate_path = str((candidate or {}).get("model_path") or "")
        if not selected_path or not Path(selected_path).is_file():
            blockers.append("selected_model_file_missing")
        elif candidate_path and selected_path != candidate_path:
            blockers.append("selected_command_model_path_drift")
    if candidate is not None and not preconditions.get("cuda_backend_evidence"):
        blockers.append("native_llama_cpp_cuda_evidence_missing")
    return list(dict.fromkeys(blockers))


def _selected_backend_command(candidate: Mapping[str, Any] | None, repeats: int) -> JsonDict | None:
    if candidate is None:
        return None
    selected = dict(candidate)
    selected["repeat_plan"] = {
        "strategy": "repeat_exp5324_stable_command",
        "requested_repeats": repeats,
        "minimum_total_duration_s": MIN_CLEAN_DURATION_S,
        "token_budget_per_repeat": int(candidate.get("n_predict") or exp5323.N_PREDICT),
        "token_budget_changed_from_exp5324": False,
        "command_shape_preserved": True,
    }
    return selected


def _optional_model_row(role: str, spec: Mapping[str, Any], *, attempt_optional_models: bool) -> JsonDict:
    if spec.get("status") != "local_gguf_resolved":
        return {
            "role": role,
            "hf_id": spec.get("hf_id"),
            "model_path": spec.get("model_path"),
            "status": "blocked",
            "attempted": False,
            "blocked_reason": "model_file_missing_or_metadata_unreadable",
            "quality_claim_permitted": False,
        }
    if not attempt_optional_models:
        return {
            "role": role,
            "hf_id": spec.get("hf_id"),
            "model_path": spec.get("model_path"),
            "status": "skipped",
            "attempted": False,
            "blocked_reason": "optional_probe_skipped_to_preserve_stable_dense_receipt_after_clean_gate",
            "quality_claim_permitted": False,
        }
    return {
        "role": role,
        "hf_id": spec.get("hf_id"),
        "model_path": spec.get("model_path"),
        "status": "skipped",
        "attempted": False,
        "blocked_reason": "optional_probe_hook_not_used_in_this_corrigendum_run",
        "quality_claim_permitted": False,
    }


def _dense_blocked_row(
    selected_model: Mapping[str, Any] | None,
    blockers: Sequence[str],
    receipts: Sequence[Mapping[str, Any]],
    methodology_duration_s: float,
) -> JsonDict:
    reason = "preconditions_blocked:" + ",".join(blockers) if blockers else "methodology_duration_below_60s"
    if receipts and not blockers and all(_receipt_ready(row) for row in receipts):
        reason = "methodology_duration_below_60s"
    elif receipts and not blockers:
        reason = "receipt_not_clean:" + ",".join(classify_clean_receipt(row) for row in receipts)
    return {
        "role": "flagship_dense",
        "hf_id": (selected_model or {}).get("hf_id", "unsloth/gemma-4-31B-it-GGUF"),
        "model_path": (selected_model or {}).get("model_path"),
        "status": "blocked",
        "attempted": bool(receipts),
        "repeat_count": len(receipts),
        "methodology_duration_s": methodology_duration_s,
        "blocked_reason": reason,
        "quality_claim_permitted": False,
        "receipts": list(receipts),
    }


def _build_dense_row(
    selected_model: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    methodology_duration_s: float,
) -> JsonDict:
    return {
        "role": "flagship_dense",
        "hf_id": selected_model.get("hf_id"),
        "model_path": selected_model.get("model_path"),
        "status": "clean_live_receipt_ready",
        "attempted": True,
        "repeat_count": len(receipts),
        "methodology_duration_s": methodology_duration_s,
        "blocked_reason": None,
        "completed_receipts": len([row for row in receipts if _receipt_ready(row)]),
        "offload_authenticated": all(bool(row.get("offload_authenticated")) for row in receipts),
        "answer_text_separable_from_thinking_text": any(
            bool(row.get("answer_text_separable_from_thinking_text")) for row in receipts
        ),
        "quality_claim_permitted": False,
        "receipts": list(receipts),
    }


def _receipt_blocked_reason(receipts: Sequence[Mapping[str, Any]], blockers: Sequence[str]) -> str | None:
    if blockers:
        return "preconditions_blocked:" + ",".join(blockers)
    if not receipts:
        return "no_dense_receipt_attempted"
    if not all(_receipt_ready(row) for row in receipts):
        return "receipt_not_clean:" + ",".join(classify_clean_receipt(row) for row in receipts)
    if sum(float(row.get("wall_clock_s") or 0.0) for row in receipts) < MIN_CLEAN_DURATION_S:
        return "methodology_duration_below_60s"
    return None


def _aggregate_corrigendum_receipt(
    *,
    selected_model: Mapping[str, Any] | None,
    selected_command: Mapping[str, Any] | None,
    receipts: Sequence[Mapping[str, Any]],
    methodology_duration_s: float,
    blockers: Sequence[str],
) -> JsonDict:
    clean = bool(
        selected_model
        and selected_command
        and len(receipts) >= DEFAULT_REPEATS
        and all(_receipt_ready(row) for row in receipts)
        and methodology_duration_s >= MIN_CLEAN_DURATION_S
    )
    first_token_times = [
        float(row["first_token_latency_s"])
        for row in receipts
        if isinstance(row.get("first_token_latency_s"), int | float)
    ]
    gpu_deltas = [int(row.get("gpu_memory_delta_mb") or 0) for row in receipts]
    stdout_summaries = [str(row.get("stdout_summary", "")) for row in receipts]
    stderr_summaries = [str(row.get("stderr_summary", "")) for row in receipts]
    command = list((selected_command or {}).get("command") or [])
    return {
        "model_role": (selected_command or {}).get("model_role"),
        "hf_id": (selected_model or {}).get("hf_id"),
        "model_path": (selected_model or {}).get("model_path"),
        "command": command,
        "context": (selected_command or {}).get("context"),
        "batch": (selected_command or {}).get("batch"),
        "ubatch": (selected_command or {}).get("ubatch"),
        "gpu_layers": (selected_command or {}).get("gpu_layers"),
        "split_mode": exp5323.DEFAULT_SPLIT_MODE,
        "prompt": (selected_command or {}).get("prompt"),
        "token_count": sum(int(row.get("generated_token_count") or 0) for row in receipts),
        "token_budget_per_repeat": (selected_command or {}).get("n_predict"),
        "timeout_s": (selected_command or {}).get("timeout_s"),
        "repeat_count": len(receipts),
        "methodology_duration_s": methodology_duration_s,
        "total_duration_s": methodology_duration_s,
        "first_token_time_s": first_token_times,
        "first_token_time_s_min": min(first_token_times) if first_token_times else None,
        "first_token_time_s_max": max(first_token_times) if first_token_times else None,
        "gpu_memory_delta_mb_max": max(gpu_deltas) if gpu_deltas else 0,
        "gpu_memory_receipts": [row.get("gpu_memory_receipts") for row in receipts],
        "offload_authenticated": bool(receipts) and all(
            bool(row.get("offload_authenticated")) for row in receipts
        ),
        "stdout_summary": "\n--- receipt ---\n".join(stdout_summaries)[-4000:],
        "stderr_summary": "\n--- receipt ---\n".join(stderr_summaries)[-2400:],
        "answer_text_separable_from_thinking_text": any(
            bool(row.get("answer_text_separable_from_thinking_text")) for row in receipts
        ),
        "final_answer_text": next(
            (row.get("final_answer_text") for row in receipts if row.get("final_answer_text")),
            None,
        ),
        "receipt_rows": list(receipts),
        "clean_receipt_ready": clean,
        "blocked_reason": _receipt_blocked_reason(receipts, blockers),
        "quality_claim_permitted": False,
    }


def default_runtime_probe(
    *,
    model_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    timeout_s: float,
    run_index: int,
) -> JsonDict:  # pragma: no cover - delegates to the live Exp5323 subprocess probe
    _ = run_index
    return exp5323.default_runtime_probe(
        model_spec=model_spec,
        variant=variant,
        timeout_s=timeout_s,
    )


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    prior_artifact_path: Path | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    preconditions_provider: PreconditionsProvider | None = None,
    runtime_probe: RuntimeProbe = default_runtime_probe,
    tests_run: Sequence[Any] | None = None,
    repeats: int = DEFAULT_REPEATS,
    attempt_optional_models: bool = False,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    artifact_path = artifact_path or root / RESULT_RELATIVE_PATH
    prior_artifact_path = prior_artifact_path or root / exp5324.RESULT_RELATIVE_PATH
    prior_artifact = _read_json(prior_artifact_path)
    candidate = _prior_stable_command(prior_artifact)
    selected_command = _selected_backend_command(candidate, repeats)

    preconditions_provider = preconditions_provider or (lambda: exp5323.collect_preconditions(root))
    preconditions = dict(preconditions_provider())
    model_specs = _resolve_model_specs(model_resolver)
    selected_model = _selected_model_spec(candidate, model_specs)
    command = _candidate_command(candidate)
    preconditions["exp5324_artifact_path"] = str(prior_artifact_path)
    preconditions["exp5324_stable_command_ready"] = candidate is not None
    preconditions["exact_exp5324_command"] = command
    preconditions["selected_backend_command"] = selected_command
    preconditions["selected_model_role"] = (candidate or {}).get("model_role") if candidate else None
    preconditions["selected_model_file_present"] = bool(
        selected_model and selected_model.get("model_path") and Path(str(selected_model["model_path"])).is_file()
    )
    preconditions["model_file_presence"] = {
        role: bool(spec.get("model_path") and Path(str(spec["model_path"])).is_file())
        for role, spec in model_specs.items()
    }
    preconditions["autotokenizer_used"] = False
    blockers = _precondition_blockers(
        preconditions=preconditions,
        candidate=candidate,
        selected_model=selected_model,
    )
    preconditions["blocked_preconditions"] = blockers

    receipts: list[JsonDict] = []
    if not blockers and candidate is not None and selected_model is not None:
        variant = _build_replay_variant(candidate)
        timeout_s = float(variant["timeout_s"])
        for run_index in range(1, max(1, repeats) + 1):
            raw_receipt = runtime_probe(
                model_spec=selected_model,
                variant=variant,
                timeout_s=timeout_s,
                run_index=run_index,
            )
            receipts.append(
                _normalise_replay_receipt(
                    raw_receipt,
                    model_spec=selected_model,
                    variant=variant,
                    timeout_s=timeout_s,
                    run_index=run_index,
                )
            )

    methodology_duration_s = round(sum(float(row.get("wall_clock_s") or 0.0) for row in receipts), 6)
    dense_ready = bool(
        selected_model
        and selected_command
        and len(receipts) >= DEFAULT_REPEATS
        and all(_receipt_ready(row) for row in receipts)
        and methodology_duration_s >= MIN_CLEAN_DURATION_S
    )
    status = "complete" if dense_ready else "blocked"
    if dense_ready:
        honest = "complete: sota_runtime_clean_receipt_ready=flagship_dense:llama-cli"
    else:
        honest = "blocked_sota_runtime_clean_receipt_ready_false: "
        honest += _receipt_blocked_reason(receipts, blockers) or "unknown_runtime_blocker"

    matrix: JsonDict = {}
    for role, spec in model_specs.items():
        if role == "flagship_dense":
            if dense_ready and selected_model is not None:
                matrix[role] = _build_dense_row(selected_model, receipts, methodology_duration_s)
            else:
                matrix[role] = _dense_blocked_row(selected_model, blockers, receipts, methodology_duration_s)
        else:
            matrix[role] = _optional_model_row(
                role,
                spec,
                attempt_optional_models=attempt_optional_models,
            )

    runtime_corrigendum_receipt = _aggregate_corrigendum_receipt(
        selected_model=selected_model,
        selected_command=selected_command,
        receipts=receipts,
        methodology_duration_s=methodology_duration_s,
        blockers=blockers,
    )

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", honest),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "MODEL_SPECS": _wrap("MODEL_SPECS", model_specs),
        "preconditions_checked": _wrap("preconditions_checked", preconditions),
        "selected_backend_command": _wrap("selected_backend_command", selected_command),
        "runtime_corrigendum_receipt": _wrap(
            "runtime_corrigendum_receipt",
            runtime_corrigendum_receipt,
        ),
        "multi_model_receipt_matrix": _wrap("multi_model_receipt_matrix", matrix),
        "methodology_duration_s": methodology_duration_s,
        "sota_runtime_clean_receipt_ready": dense_ready,
        "runtime_unblocked_min_one_mandated": dense_ready,
        "quality_claim_permitted": False,
        "no_autotokenizer_used": True,
        "tests_run": _wrap("tests_run", list(tests_run or [])),
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.perf_counter() - started, 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = exp5323.sha16(
        exp5323._stable_json(
            {
                "experiment_id": EXPERIMENT_ID,
                "selected_command": selected_command,
                "selected_model": selected_model,
                "receipts": receipts,
                "methodology_duration_s": methodology_duration_s,
                "ready": dense_ready,
                "seed": RANDOM_SEED,
            }
        )
    )
    validate_artifact(artifact)
    if write:
        exp5323.write_json(artifact_path, artifact)
    return artifact


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    value = artifact.get(field)
    if not isinstance(value, Mapping):
        return MISSING_WRAPPED_VALUE
    if value.get("principle") != FIELD_PRINCIPLES.get(field):
        return MISSING_WRAPPED_VALUE
    return value.get("value")


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    for field in WRAPPED_FIELDS:
        if field in artifact and _wrapped_value(artifact, field) is MISSING_WRAPPED_VALUE:
            errors.append(f"{field} must be principle-wrapped")
    if _wrapped_value(artifact, "experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if _wrapped_value(artifact, "milestone") != MILESTONE:
        errors.append("milestone mismatch")
    if _wrapped_value(artifact, "status") not in {"complete", "blocked"}:
        errors.append("status must be complete or blocked")
    honest = _wrapped_value(artifact, "honest_verdict")
    if not isinstance(honest, str) or not honest.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked_")
    if _wrapped_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")

    methodology_duration = artifact.get("methodology_duration_s")
    if not isinstance(methodology_duration, int | float):
        errors.append("methodology_duration_s must be numeric")
        methodology_duration = -1.0
    if not isinstance(artifact.get("sota_runtime_clean_receipt_ready"), bool):
        errors.append("sota_runtime_clean_receipt_ready must be a bare boolean")
    if not isinstance(artifact.get("runtime_unblocked_min_one_mandated"), bool):
        errors.append("runtime_unblocked_min_one_mandated must be a bare boolean")
    if artifact.get("quality_claim_permitted") is not False:
        errors.append("quality_claim_permitted must be bare false")
    if artifact.get("no_autotokenizer_used") is not True:
        errors.append("no_autotokenizer_used must be bare true")

    model_specs = _wrapped_value(artifact, "MODEL_SPECS")
    if not isinstance(model_specs, Mapping):
        errors.append("MODEL_SPECS must be an object")
    else:
        expected_roles = {str(spec["role"]) for spec in MANDATED_MODEL_SPECS}
        if set(model_specs) != expected_roles:
            errors.append("MODEL_SPECS roles mismatch")
        expected_hf = {str(spec["role"]): str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS}
        for role in expected_roles & set(model_specs):
            spec = model_specs[role]
            if spec.get("hf_id") != expected_hf[role]:
                errors.append("hf_id mismatch for mandated model role")
            if spec.get("autotokenizer_used") is not False:
                errors.append("autotokenizer_used must stay false")

    tests_run = _wrapped_value(artifact, "tests_run")
    runtime_receipt = _wrapped_value(artifact, "runtime_corrigendum_receipt")
    matrix = _wrapped_value(artifact, "multi_model_receipt_matrix")
    if tests_run is not MISSING_WRAPPED_VALUE and not isinstance(tests_run, list):
        errors.append("tests_run must be a list")
    if runtime_receipt is not MISSING_WRAPPED_VALUE and not isinstance(runtime_receipt, Mapping):
        errors.append("runtime_corrigendum_receipt must be an object")
    if matrix is not MISSING_WRAPPED_VALUE and not isinstance(matrix, Mapping):
        errors.append("multi_model_receipt_matrix must be an object")
    elif isinstance(matrix, Mapping):
        expected_roles = {str(spec["role"]) for spec in MANDATED_MODEL_SPECS}
        if set(matrix) != expected_roles:
            errors.append("multi_model_receipt_matrix roles mismatch")

    ready = artifact.get("sota_runtime_clean_receipt_ready")
    if ready is True:
        if _wrapped_value(artifact, "status") != "complete":
            errors.append("clean artifact must have complete status")
        if artifact.get("runtime_unblocked_min_one_mandated") is not True:
            errors.append("clean artifact must unblock at least one mandated model")
        if isinstance(methodology_duration, int | float) and methodology_duration < MIN_CLEAN_DURATION_S:
            errors.append("clean receipt cannot be ready below 60s")
        if not isinstance(runtime_receipt, Mapping) or runtime_receipt.get("clean_receipt_ready") is not True:
            errors.append("runtime_corrigendum_receipt must be clean when gate is true")
        if isinstance(matrix, Mapping):
            dense = matrix.get("flagship_dense")
            if not isinstance(dense, Mapping) or dense.get("status") != "clean_live_receipt_ready":
                errors.append("flagship_dense row must be clean when gate is true")
    elif ready is False:
        if _wrapped_value(artifact, "status") != "blocked":
            errors.append("blocked artifact must have blocked status")
        if artifact.get("runtime_unblocked_min_one_mandated") is not False:
            errors.append("blocked artifact must not unblock runtime")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError("; ".join(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--prior", type=Path, default=REPO_ROOT / exp5324.RESULT_RELATIVE_PATH)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--attempt-optional-models",
        action="store_true",
        help="Attempt optional mandated MoE probes. Default skips them to preserve dense receipt.",
    )
    parser.add_argument(
        "--tests-run-json",
        default="[]",
        help="JSON list of validation commands to embed in the artifact.",
    )
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=args.out,
        prior_artifact_path=args.prior,
        repeats=args.repeats,
        attempt_optional_models=args.attempt_optional_models,
        tests_run=json.loads(args.tests_run_json),
        write=True,
    )
    print(
        f"[exp5337] status={artifact['status']['value']} "
        f"clean={artifact['sota_runtime_clean_receipt_ready']} out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
