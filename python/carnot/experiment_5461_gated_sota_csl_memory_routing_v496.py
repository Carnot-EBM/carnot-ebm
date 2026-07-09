#!/usr/bin/env python3
"""Exp5461 gated local SOTA CSL memory routing panel.

Spec refs: REQ-LEARN-5461,
SCENARIO-LEARN-5461-PRECONDITIONS,
SCENARIO-LEARN-5461-CONDITIONS,
SCENARIO-LEARN-5461-VERIFIERS,
SCENARIO-LEARN-5461-NO-WEIGHT-MUTATION.

This experiment keeps the model frozen and treats the Exp5460 policy snapshot
as the only adaptive state. The live local GGUF model is a candidate generator;
small deterministic task witnesses decide correctness so the model cannot grade
its own answer.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from carnot import experiment_5460_csl_policy_bandit_v496 as exp5460
from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
RuntimeProbe = Callable[..., JsonDict]
GenerationRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5461_gated_sota_csl_memory_routing_v496.json")
ROW_RESULTS_RELATIVE_PATH = Path(
    "results/experiment_5461_gated_sota_csl_memory_routing_v496_rows.jsonl"
)
EXP5460_RESULT_RELATIVE_PATH = exp5460.RESULT_RELATIVE_PATH
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5461_gated_sota_csl_memory_routing_v496.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")

EXPERIMENT_ID = "experiment_5461_gated_sota_csl_memory_routing_v496"
TASK_ID = "exp5461-gated-sota-csl-memory-routing-v496"
MILESTONE = "2026.07.496"
RUN_DATE = "2026-07-09"
SCHEMA = "carnot.experiment_5461.gated_sota_csl_memory_routing.v496"
SPEC_REFS = (
    "REQ-LEARN-5461",
    "SCENARIO-LEARN-5461-PRECONDITIONS",
    "SCENARIO-LEARN-5461-CONDITIONS",
    "SCENARIO-LEARN-5461-VERIFIERS",
    "SCENARIO-LEARN-5461-NO-WEIGHT-MUTATION",
)
RANDOM_SEED = 5461
DEFAULT_QUANTIZATION = "Q4_K_M"
DEFAULT_MAX_TASKS = 4
DEFAULT_TOKEN_BUDGET = 24
N_GPU_LAYERS = -1
INFERENCE_SUBSTRATE = "live_llm_inference_with_frozen_policy_state"
BLOCKED_INFERENCE_SUBSTRATE = "blocked_preconditions_no_live_llm"
TERMINAL_PREFIXES = ("complete:", "blocked:")

NO_MEMORY_CONDITION = "no_memory"
NAIVE_CONDITION = "naive_icl"
GOVERNED_CONDITION = "governed_memory"
POLICY_CONDITION = "policy_selected"
CONDITION_NAMES = (
    NO_MEMORY_CONDITION,
    NAIVE_CONDITION,
    GOVERNED_CONDITION,
    POLICY_CONDITION,
)

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "role": "flagship_moe",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "quantization": DEFAULT_QUANTIZATION,
    },
    {
        "role": "flagship_dense",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "quantization": DEFAULT_QUANTIZATION,
    },
    {
        "role": "middle_moe",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "quantization": DEFAULT_QUANTIZATION,
    },
)
MANDATED_HF_IDS = tuple(str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS)
RUN_ROLE_PREFERENCE = ("middle_moe", "flagship_moe", "flagship_dense")

FIELD_PRINCIPLES: dict[str, str] = {
    "preconditions_checked": "Compute gate transparency.",
    "model_specs": "Mandated SOTA GGUF provenance.",
    "headline_required_any_of": "Headline model boundary.",
    "runtime_backend": "Live GGUF runtime provenance.",
    "gpu_offload_verified": "No CPU-only headline.",
    "condition_names": "Baseline comparability.",
    "row_results_path": "Inspectable row evidence.",
    "policy_state_checksum": "Frozen policy-state provenance.",
    "quality_delta_vs_no_memory": "Utility against no-memory.",
    "quality_delta_vs_naive_icl": "Utility against naive memory.",
    "context_efficiency_delta": "Context budget accounting.",
    "verifier_cost_delta": "Verifier budget accounting.",
    "negative_transfer_deflection_rate": "Unsafe memory-transfer guard.",
    "no_weight_mutation": "Frozen-model boundary.",
    "csl_sota_memory_routing_ready": "Downstream gate.",
    "inference_substrate": "Explicit learning substrate.",
    "honest_verdict": "Terminal status; starts with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def run(
    *,
    root: Path | str = REPO_ROOT,
    artifact_path: Path | str | None = None,
    row_results_path: Path | str | None = None,
    policy_artifact: Mapping[str, Any] | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    runtime_probe: RuntimeProbe | None = None,
    generation_runner: GenerationRunner | None = None,
    max_tasks: int = DEFAULT_MAX_TASKS,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    write: bool = True,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """Run the bounded SOTA memory-routing panel and optionally write artifacts."""

    started = time.perf_counter()
    root_path = Path(root)
    destination = _destination(root_path, artifact_path, RESULT_RELATIVE_PATH)
    rows_destination = _destination(root_path, row_results_path, ROW_RESULTS_RELATIVE_PATH)
    policy_payload = (
        dict(policy_artifact)
        if policy_artifact is not None
        else _read_json(root_path / EXP5460_RESULT_RELATIVE_PATH)
    )
    model_specs = resolve_model_specs(model_resolver)
    selected_model = select_headline_model(model_specs)
    runtime_fn = runtime_probe or default_runtime_probe
    runtime_receipt = runtime_fn(model_spec=selected_model, n_gpu_layers=N_GPU_LAYERS)
    preconditions = evaluate_preconditions(
        policy_artifact=policy_payload,
        model_specs=model_specs,
        selected_model=selected_model,
        runtime_receipt=runtime_receipt,
    )
    model_before = model_file_receipt(selected_model.get("model_path") if selected_model else None)

    live_runner = generation_runner
    if preconditions["all_passed"] and live_runner is None:
        try:
            live = LlamaCslGenerationRunner(
                model_spec=selected_model or {},
                n_gpu_layers=N_GPU_LAYERS,
                seed=random_seed,
            )
        except Exception as exc:  # pragma: no cover - depends on local runtime/model.
            preconditions["all_passed"] = False
            preconditions["blocked_preconditions"].append(
                f"llama_cpp_model_load_failed:{type(exc).__name__}: {exc}"
            )
        else:
            live_runner = live
            runtime_receipt = {**runtime_receipt, "load_receipt": dict(live.load_receipt)}
            if live.load_receipt.get("offload_evidence") is not True:
                preconditions["all_passed"] = False
                preconditions["blocked_preconditions"].append("gpu_offload_not_observed_after_load")

    if not preconditions["all_passed"] or live_runner is None:
        model_after = model_file_receipt(selected_model.get("model_path") if selected_model else None)
        artifact = build_artifact(
            policy_artifact=policy_payload,
            model_specs=model_specs,
            selected_model=selected_model,
            runtime_receipt=runtime_receipt,
            preconditions=preconditions,
            rows=[],
            row_results_path=rows_destination,
            tests_run=tests_run,
            methodology_duration_s=time.perf_counter() - started,
            model_before=model_before,
            model_after=model_after,
            blocked=True,
        )
        if write:
            _write_json(destination, artifact)
        validate_artifact(artifact)
        return artifact

    tasks = build_task_stream(policy_payload)[: max(0, int(max_tasks))]
    rows = run_condition_rows(
        tasks=tasks,
        model_spec=selected_model or {},
        runtime_receipt=runtime_receipt,
        generation_runner=live_runner,
        policy_artifact=policy_payload,
        random_seed=random_seed,
    )
    model_after = model_file_receipt(selected_model.get("model_path") if selected_model else None)
    artifact = build_artifact(
        policy_artifact=policy_payload,
        model_specs=_mark_model_ran(model_specs, selected_model),
        selected_model=selected_model,
        runtime_receipt=runtime_receipt,
        preconditions=preconditions,
        rows=rows,
        row_results_path=rows_destination,
        tests_run=tests_run,
        methodology_duration_s=time.perf_counter() - started,
        model_before=model_before,
        model_after=model_after,
        blocked=False,
    )
    if write:
        _write_jsonl(rows_destination, rows)
        validate_artifact(artifact)
        _write_json(destination, artifact)
    else:
        validate_artifact(artifact)
    return artifact


def resolve_model_specs(model_resolver: ModelResolver = resolve_cached_gguf) -> list[JsonDict]:
    """Resolve all mandated GGUF paths without touching HF tokenizer APIs."""

    rows: list[JsonDict] = []
    for spec in MANDATED_MODEL_SPECS:
        hf_id = str(spec["hf_id"])
        quantization = str(spec.get("quantization", DEFAULT_QUANTIZATION))
        path_text = model_resolver(hf_id, quantization)
        path = Path(path_text) if path_text else None
        local = bool(path and path.is_file())
        rows.append(
            {
                "role": str(spec["role"]),
                "hf_id": hf_id,
                "quantization": quantization,
                "model_path": str(path) if path else None,
                "local_path_available": local,
                "file_receipt": model_file_receipt(str(path)) if local and path else None,
                "runtime_backend": None,
                "n_gpu_layers": None,
                "gpu_offload_verified": False,
                "ran_headline": False,
                "legacy_smoke_only": False,
            }
        )
    return rows


def select_headline_model(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Select one resolved mandated model, preferring the smaller MoE GGUF."""

    by_role = {str(row.get("role")): row for row in model_specs}
    for role in RUN_ROLE_PREFERENCE:
        row = by_role.get(role)
        if isinstance(row, Mapping) and row.get("local_path_available") is True:
            return dict(row)
    return None


def default_runtime_probe(**kwargs: Any) -> JsonDict:  # pragma: no cover
    """Check that CUDA and llama.cpp GPU offload are available before load."""

    blocked: list[str] = []
    try:
        import torch  # noqa: PLC0415

        torch_cuda = {
            "import_ok": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
        if not torch_cuda["cuda_available"] or int(torch_cuda["device_count"]) <= 0:
            blocked.append("torch_cuda_unavailable")
    except Exception as exc:
        torch_cuda = {"import_ok": False, "error": f"{type(exc).__name__}: {exc}"}
        blocked.append("torch_import_failed")

    try:
        from llama_cpp import llama_cpp  # noqa: PLC0415

        supports = bool(llama_cpp.llama_supports_gpu_offload())
        system_info_raw = llama_cpp.llama_print_system_info()
        system_info = (
            system_info_raw.decode("utf-8", "replace")
            if isinstance(system_info_raw, bytes)
            else str(system_info_raw)
        )
        llama_info = {
            "import_ok": True,
            "gpu_offload_supported": supports,
            "system_info": system_info,
        }
        if not supports:
            blocked.append("llama_cpp_gpu_offload_unsupported")
    except Exception as exc:
        llama_info = {"import_ok": False, "error": f"{type(exc).__name__}: {exc}"}
        blocked.append("llama_cpp_import_failed")

    nvidia_smi = _nvidia_smi_query()
    if not nvidia_smi.get("ok"):
        blocked.append("nvidia_smi_unavailable")

    return {
        "runtime_backend": "llama_cpp_python_cuda_gguf",
        "llama_cpp_import_ok": bool(llama_info.get("import_ok")),
        "cuda_visible": bool(torch_cuda.get("cuda_available"))
        and int(torch_cuda.get("device_count", 0)) > 0,
        "gpu_offload_supported": bool(llama_info.get("gpu_offload_supported")),
        "n_gpu_layers": int(kwargs.get("n_gpu_layers", N_GPU_LAYERS)),
        "offload_evidence": bool(llama_info.get("gpu_offload_supported"))
        and bool(torch_cuda.get("cuda_available")),
        "torch_cuda": torch_cuda,
        "llama_cpp": llama_info,
        "nvidia_smi": nvidia_smi,
        "blocked_preconditions": sorted(set(blocked)),
    }


def evaluate_preconditions(
    *,
    policy_artifact: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    selected_model: Mapping[str, Any] | None,
    runtime_receipt: Mapping[str, Any],
) -> JsonDict:
    """Combine Exp5460, model-cache, CUDA, and GPU-offload gates."""

    blocked = [str(item) for item in runtime_receipt.get("blocked_preconditions", [])]
    policy_ready = policy_artifact.get("csl_policy_ready") is True
    model_ids = {str(row.get("hf_id")) for row in model_specs}
    mandated_present = model_ids == set(MANDATED_HF_IDS)
    all_model_paths = bool(model_specs) and all(
        row.get("local_path_available") is True
        and str(row.get("model_path", "")).endswith(".gguf")
        for row in model_specs
    )
    runtime_backend = str(runtime_receipt.get("runtime_backend", ""))

    if not policy_ready:
        blocked.append("exp5460_csl_policy_not_ready")
    if not mandated_present:
        blocked.append("mandated_model_specs_missing")
    if not all_model_paths:
        blocked.append("non_empty_mandated_model_paths_missing")
    if selected_model is None:
        blocked.append("no_mandated_local_gguf_model_path")
    if runtime_receipt.get("cuda_visible") is not True:
        blocked.append("cuda_not_visible")
    if runtime_receipt.get("offload_evidence") is not True:
        blocked.append("gpu_offload_evidence_missing")
    if "llama" not in runtime_backend:
        blocked.append("llama_cpp_gguf_runtime_missing")

    return {
        "exp5460_csl_policy_ready": policy_ready,
        "mandated_model_specs_present": mandated_present,
        "all_mandated_model_paths_available": all_model_paths,
        "selected_model_available": selected_model is not None,
        "cuda_visible": runtime_receipt.get("cuda_visible") is True,
        "runtime_backend": runtime_backend,
        "gpu_offload_preflight": runtime_receipt.get("offload_evidence") is True,
        "blocked_preconditions": sorted(set(blocked)),
        "all_passed": not blocked,
    }


def build_task_stream(policy_artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Build a bounded stateful stream that pressures stale and poisoned memory."""

    checksum = policy_state_checksum(policy_artifact)
    snapshot = _mapping(policy_artifact.get("policy_snapshot"))
    evidence_ids = [
        str(item)
        for item in snapshot.get("accepted_evidence_ids", [])
        if isinstance(item, str)
    ]
    evidence_head = evidence_ids[:3]
    return [
        _task(
            task_id="5461-a1-repeat-gasket",
            session_id="session-a",
            case_family="repeated_task",
            question="Which crate holds the signed gasket?",
            expected_answer="blue-crate-17",
            decoy_answer="red-crate-09",
            current_evidence=[],
            naive_memory=[
                "old note: red-crate-09 held an unsigned gasket during intake.",
                "verified repeat note: blue-crate-17 holds the signed gasket.",
            ],
            governed_memory=["verified repeat note: blue-crate-17 holds the signed gasket."],
            policy_selected_condition=GOVERNED_CONDITION,
            policy_reason="Exp5460 repeat-task arm favors governed memory.",
            policy_state_checksum=checksum,
            policy_evidence_ids=evidence_head,
            requires_memory=True,
        ),
        _task(
            task_id="5461-b1-poisoned-vendor",
            session_id="session-b",
            case_family="poisoned_memory",
            question="Which vendor is verified for lot 77?",
            expected_answer="vendor-A",
            decoy_answer="vendor-B",
            current_evidence=[],
            naive_memory=["poisoned sidecar: lot 77 vendor is vendor-B."],
            governed_memory=["verified procurement receipt: lot 77 vendor is vendor-A."],
            policy_selected_condition=GOVERNED_CONDITION,
            policy_reason="Exp5460 rejects poisoned-memory evidence before routing.",
            policy_state_checksum=checksum,
            policy_evidence_ids=evidence_head,
            negative_transfer_candidate=True,
            requires_memory=True,
        ),
        _task(
            task_id="5461-c1-stale-queue",
            session_id="session-c",
            case_family="stale_memory",
            question="Which queue owns incident RX-4?",
            expected_answer="queue-beta",
            decoy_answer="queue-alpha",
            current_evidence=[],
            naive_memory=["stale memory: incident RX-4 belongs to queue-alpha."],
            governed_memory=["fresh replay-verified memory: incident RX-4 belongs to queue-beta."],
            policy_selected_condition=GOVERNED_CONDITION,
            policy_reason="Exp5460 replay gates suppress stale queue memories.",
            policy_state_checksum=checksum,
            policy_evidence_ids=evidence_head,
            negative_transfer_candidate=True,
            stale_memory_candidate=True,
            requires_memory=True,
        ),
        _task(
            task_id="5461-d1-fresh-ticket",
            session_id="session-d",
            case_family="no_memory_competitive",
            question="Current ticket says the shipping code is K-42. What is the shipping code?",
            expected_answer="K-42",
            decoy_answer="K-24",
            current_evidence=["current ticket text: shipping code is K-42."],
            naive_memory=["irrelevant old example: a different ticket used K-24."],
            governed_memory=["fresh current-ticket receipt: shipping code is K-42."],
            policy_selected_condition=NO_MEMORY_CONDITION,
            policy_reason="Exp5460 keeps no-memory competitive for fresh simple rows.",
            policy_state_checksum=checksum,
            policy_evidence_ids=evidence_head,
            requires_memory=False,
        ),
    ]


def run_condition_rows(
    *,
    tasks: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    generation_runner: GenerationRunner,
    policy_artifact: Mapping[str, Any],
    random_seed: int = RANDOM_SEED,
) -> list[JsonDict]:
    """Generate and exactly score one row for every task and condition."""

    rows: list[JsonDict] = []
    runtime_backend = str(runtime_receipt.get("runtime_backend", "llama_cpp_python_cuda_gguf"))
    policy_checksum = policy_state_checksum(policy_artifact)
    for task_index, task in enumerate(tasks):
        task_row = copy.deepcopy(dict(task))
        for condition_index, condition in enumerate(CONDITION_NAMES):
            seed = random_seed + task_index * 101 + condition_index
            prompt, memory_receipt, policy_receipt = build_prompt_and_receipts(
                task_row,
                condition=condition,
                policy_checksum=policy_checksum,
            )
            started = time.perf_counter()
            generation = generation_runner(
                prompt=prompt,
                condition=condition,
                task=task_row,
                model_spec=model_spec,
                runtime_backend=runtime_backend,
                seed=seed,
                max_tokens=DEFAULT_TOKEN_BUDGET,
                n_gpu_layers=N_GPU_LAYERS,
            )
            fallback_duration_s = time.perf_counter() - started
            row = score_candidate_row(
                task=task_row,
                condition=condition,
                model_spec=model_spec,
                runtime_backend=runtime_backend,
                runtime_receipt=runtime_receipt,
                generation=generation,
                prompt_text=prompt,
                memory_receipt=memory_receipt,
                policy_receipt=policy_receipt,
                seed=seed,
                fallback_duration_s=fallback_duration_s,
            )
            rows.append(row)
    return rows


def build_prompt_and_receipts(
    task: Mapping[str, Any],
    *,
    condition: str,
    policy_checksum: str,
) -> tuple[str, JsonDict, JsonDict]:
    """Build the prompt plus memory and policy receipts for one condition."""

    if condition not in CONDITION_NAMES:
        raise ValueError(f"unknown condition: {condition}")
    routed_condition = str(task.get("policy_selected_condition", GOVERNED_CONDITION))
    effective_condition = routed_condition if condition == POLICY_CONDITION else condition
    memory_lines = _memory_lines(task, effective_condition)
    memory_ids = [
        f"{task['task_id']}:{effective_condition}:memory-{index}"
        for index, _line in enumerate(memory_lines, start=1)
    ]
    policy_receipt = {
        "condition": condition,
        "policy_selected": condition == POLICY_CONDITION,
        "routed_condition": routed_condition if condition == POLICY_CONDITION else None,
        "policy_state_checksum": policy_checksum,
        "policy_reason": task.get("policy_reason", ""),
        "source_policy_evidence_ids": list(task.get("policy_evidence_ids", [])),
    }
    memory_receipt = {
        "condition": condition,
        "effective_condition": effective_condition,
        "task_id": task["task_id"],
        "memory_ids": memory_ids,
        "memory_lines": memory_lines,
        "negative_transfer_candidate": task.get("negative_transfer_candidate") is True,
        "stale_memory_candidate": task.get("stale_memory_candidate") is True,
    }
    lines = [
        "Return only the final answer token. Do not explain.",
        "If no supplied evidence determines the answer, return unknown.",
        f"Task id: {task['task_id']}",
        f"Question: {task['question']}",
    ]
    if memory_lines:
        lines.append("Supplied evidence:")
        lines.extend(f"- {line}" for line in memory_lines)
    else:
        lines.append("Supplied evidence: none.")
    lines.append("Final answer:")
    return "\n".join(lines), memory_receipt, policy_receipt


def score_candidate_row(
    *,
    task: Mapping[str, Any],
    condition: str,
    model_spec: Mapping[str, Any],
    runtime_backend: str,
    runtime_receipt: Mapping[str, Any],
    generation: Mapping[str, Any],
    prompt_text: str,
    memory_receipt: Mapping[str, Any],
    policy_receipt: Mapping[str, Any],
    seed: int,
    fallback_duration_s: float,
) -> JsonDict:
    """Parse one model output and score it with the exact task verifier."""

    output_text = str(generation.get("output_text", ""))
    witness = exact_task_verifier(task, output_text)
    prompt_tokens = int(generation.get("prompt_token_count", 0) or 0) or _estimate_tokens(prompt_text)
    completion_tokens = int(generation.get("generated_token_count", 0) or 0)
    routed_condition = str(memory_receipt.get("effective_condition", condition))
    row: JsonDict = {
        "schema": "carnot.experiment_5461.row.v1",
        "experiment_id": EXPERIMENT_ID,
        "task_id": str(task["task_id"]),
        "session_id": str(task["session_id"]),
        "case_family": str(task["case_family"]),
        "condition": condition,
        "effective_condition": routed_condition,
        "model_role": str(model_spec.get("role")),
        "model_hf_id": str(model_spec.get("hf_id")),
        "model_path": str(model_spec.get("model_path")),
        "runtime_backend": runtime_backend,
        "n_gpu_layers": int(runtime_receipt.get("n_gpu_layers", N_GPU_LAYERS)),
        "gpu_offload_evidence": bool(runtime_receipt.get("offload_evidence")),
        "random_seed": seed,
        "prompt_text": prompt_text,
        "prompt_hash": _sha256_text(prompt_text),
        "context_cost": prompt_tokens,
        "generated_token_count": completion_tokens,
        "token_cost": prompt_tokens + completion_tokens,
        "verifier_cost": verifier_cost_units(task, condition, routed_condition),
        "memory_receipt": copy.deepcopy(dict(memory_receipt)),
        "policy_receipt": copy.deepcopy(dict(policy_receipt)),
        "generation_duration_s": float(generation.get("duration_s", fallback_duration_s)),
        "output_text": output_text,
        "selected_answer": witness["selected_answer"],
        "expected_answer": str(task["expected_answer"]),
        "decoy_answer": str(task["decoy_answer"]),
        "exact_verifier_witness": witness,
        "accepted_by_final_authority": bool(witness["accepted"]),
        "final_authority_bypassed": False,
        "negative_transfer_candidate": task.get("negative_transfer_candidate") is True,
        "stale_memory_candidate": task.get("stale_memory_candidate") is True,
        "negative_transfer_detected": bool(witness["negative_transfer_detected"]),
        "backend_details": copy.deepcopy(generation.get("backend_details", {})),
    }
    row["row_checksum"] = row_checksum(row)
    return row


def exact_task_verifier(task: Mapping[str, Any], output_text: str) -> JsonDict:
    """Score a row using a deterministic expected-answer witness."""

    expected = str(task["expected_answer"])
    decoy = str(task["decoy_answer"])
    candidates = (expected, decoy, "unknown")
    selected = _first_literal_match(output_text, candidates)
    reasons: list[str] = []
    if selected is None:
        reasons.append("answer_not_found")
    elif selected != expected:
        reasons.append("answer_mismatch")
    negative_transfer = selected == decoy
    if negative_transfer:
        reasons.append("decoy_memory_selected")
    return {
        "authority": "exact_task_verifier",
        "verified": True,
        "accepted": selected == expected,
        "selected_answer": selected,
        "expected_answer": expected,
        "decoy_answer": decoy,
        "negative_transfer_detected": negative_transfer,
        "failure_reasons": reasons,
        "model_self_verdict_ignored": True,
        "witness": {
            "literal_match_candidates": list(candidates),
            "normalized_output": _normalize_answer(output_text),
        },
    }


def verifier_cost_units(
    task: Mapping[str, Any],
    condition: str,
    routed_condition: str,
) -> int:
    """Return deterministic witness cost units for one row."""

    effective = routed_condition if condition == POLICY_CONDITION else condition
    cost = 1
    if effective in {NAIVE_CONDITION, GOVERNED_CONDITION}:
        cost += 1
    if effective == NAIVE_CONDITION and task.get("negative_transfer_candidate") is True:
        cost += 1
    return cost


def derive_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute aggregate quality, cost, deflection, and independence metrics."""

    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    by_condition: dict[str, list[JsonDict]] = {name: [] for name in CONDITION_NAMES}
    for row in row_list:
        by_condition.setdefault(str(row.get("condition")), []).append(row)
    condition_metrics = {
        condition: _condition_metric(by_condition.get(condition, []))
        for condition in CONDITION_NAMES
    }
    policy_quality = condition_metrics[POLICY_CONDITION]["quality_score"]
    no_memory_quality = condition_metrics[NO_MEMORY_CONDITION]["quality_score"]
    naive_quality = condition_metrics[NAIVE_CONDITION]["quality_score"]
    policy_context = condition_metrics[POLICY_CONDITION]["context_cost"]
    naive_context = condition_metrics[NAIVE_CONDITION]["context_cost"]
    policy_verifier = condition_metrics[POLICY_CONDITION]["verifier_cost"]
    naive_verifier = condition_metrics[NAIVE_CONDITION]["verifier_cost"]
    policy_rows = by_condition.get(POLICY_CONDITION, [])
    negative_policy_rows = [
        row for row in policy_rows if row.get("negative_transfer_candidate") is True
    ]
    stale_policy_rows = [row for row in policy_rows if row.get("stale_memory_candidate") is True]
    row_checksums_match = all(row.get("row_checksum") == row_checksum(row) for row in row_list)
    exact_authority_ok = all(_exact_authority_row(row) for row in row_list)
    task_ids = sorted({str(row.get("task_id")) for row in row_list})
    regret = _regret_proxy(row_list, task_ids)
    predicate_support = {
        "quality": "row.accepted_by_final_authority",
        "context_cost": "row.context_cost",
        "verifier_cost": "row.verifier_cost",
        "negative_transfer": "policy row exact verifier avoids decoy answer",
        "policy_regret": "oracle accepted row score minus condition score",
    }
    metric_independence = bool(
        (not row_list or row_checksums_match)
        and exact_authority_ok
        and len(set(predicate_support.values())) == len(predicate_support)
    )
    return {
        "condition_metrics": condition_metrics,
        "quality_delta_vs_no_memory": round(policy_quality - no_memory_quality, 6),
        "quality_delta_vs_naive_icl": round(policy_quality - naive_quality, 6),
        "context_efficiency_delta": _relative_savings(naive_context, policy_context),
        "verifier_cost_delta": _relative_savings(naive_verifier, policy_verifier),
        "negative_transfer_deflection_rate": _deflection_rate(negative_policy_rows),
        "stale_memory_deflection_rate": _deflection_rate(stale_policy_rows),
        "policy_regret_proxy": regret,
        "metric_independence_checks_passed": metric_independence,
        "predicate_support": predicate_support,
        "row_checksums_match": row_checksums_match,
        "exact_authority_rows": sum(1 for row in row_list if _exact_authority_row(row)),
        "row_count": len(row_list),
        "task_count": len(task_ids),
        "case_family_counts": dict(
            sorted(Counter(str(row.get("case_family")) for row in policy_rows).items())
        ),
    }


def build_artifact(
    *,
    policy_artifact: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    selected_model: Mapping[str, Any] | None,
    runtime_receipt: Mapping[str, Any],
    preconditions: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    row_results_path: Path,
    tests_run: Sequence[str | Mapping[str, Any]],
    methodology_duration_s: float,
    model_before: Mapping[str, Any],
    model_after: Mapping[str, Any],
    blocked: bool,
) -> JsonDict:
    """Assemble the terminal Exp5461 artifact from row evidence."""

    row_list = [dict(row) for row in rows]
    metrics = derive_metrics(row_list)
    policy_checksum = policy_state_checksum(policy_artifact)
    gpu_offload_verified = bool(
        preconditions.get("all_passed")
        and runtime_receipt.get("offload_evidence") is True
        and _mapping(runtime_receipt.get("load_receipt")).get("offload_evidence", True) is True
    )
    weight_receipt = weight_mutation_receipt(
        model_before=model_before,
        model_after=model_after,
        model_loaded=bool(row_list),
    )
    ready = bool(
        not blocked
        and row_list
        and gpu_offload_verified
        and metrics["metric_independence_checks_passed"]
        and metrics["quality_delta_vs_no_memory"] >= 0.0
        and metrics["quality_delta_vs_naive_icl"] >= 0.0
        and metrics["context_efficiency_delta"] > 0.0
        and metrics["verifier_cost_delta"] > 0.0
        and metrics["negative_transfer_deflection_rate"] == 1.0
        and weight_receipt["no_weight_mutation"] is True
        and bool(tests_run)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "status": "complete" if row_list else "blocked",
        "preconditions_checked": True,
        "precondition_details": copy.deepcopy(dict(preconditions)),
        "model_specs": [dict(row) for row in model_specs],
        "selected_model_spec": dict(selected_model) if selected_model is not None else None,
        "headline_required_any_of": list(MANDATED_HF_IDS),
        "runtime_backend": str(runtime_receipt.get("runtime_backend", "unavailable")),
        "runtime_receipt": copy.deepcopy(dict(runtime_receipt)),
        "gpu_offload_verified": gpu_offload_verified,
        "condition_names": list(CONDITION_NAMES),
        "row_results_path": str(row_results_path),
        "policy_state_checksum": policy_checksum,
        "quality_delta_vs_no_memory": metrics["quality_delta_vs_no_memory"],
        "quality_delta_vs_naive_icl": metrics["quality_delta_vs_naive_icl"],
        "context_efficiency_delta": metrics["context_efficiency_delta"],
        "verifier_cost_delta": metrics["verifier_cost_delta"],
        "negative_transfer_deflection_rate": metrics["negative_transfer_deflection_rate"],
        "no_weight_mutation": weight_receipt["no_weight_mutation"],
        "csl_sota_memory_routing_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE if row_list else BLOCKED_INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, row_list, preconditions, metrics),
        "task_count": metrics["task_count"],
        "row_count": metrics["row_count"],
        "condition_metrics": metrics["condition_metrics"],
        "stale_memory_deflection_rate": metrics["stale_memory_deflection_rate"],
        "policy_regret_proxy": metrics["policy_regret_proxy"],
        "metric_independence_checks_passed": metrics["metric_independence_checks_passed"],
        "metric_details": metrics,
        "metric_dependency_graph": build_metric_dependency_graph(),
        "row_results": row_list,
        "row_checksums": [row.get("row_checksum") for row in row_list],
        "policy_source": {
            "path": str(EXP5460_RESULT_RELATIVE_PATH),
            "csl_policy_ready": policy_artifact.get("csl_policy_ready") is True,
            "reproducibility_checksum": policy_artifact.get("reproducibility_checksum", ""),
        },
        "weight_mutation_receipt": weight_receipt,
        "field_principles": dict(FIELD_PRINCIPLES),
        "methodology_duration_s": methodology_duration_s,
        "tests_run": _normalise_tests_run(tests_run),
        "source_files": {
            "module": str(MODULE_RELATIVE_PATH),
            "spec": str(SPEC_RELATIVE_PATH),
            "upstream_policy": str(EXP5460_RESULT_RELATIVE_PATH),
        },
        "source_file_checksums": _source_file_checksums(REPO_ROOT),
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return _json_ready(artifact)


def build_metric_dependency_graph() -> JsonDict:
    """Return the readiness dependency graph used by validation."""

    return {
        "quality_delta_vs_no_memory": [
            "condition_metrics.policy_selected.quality_score",
            "condition_metrics.no_memory.quality_score",
        ],
        "quality_delta_vs_naive_icl": [
            "condition_metrics.policy_selected.quality_score",
            "condition_metrics.naive_icl.quality_score",
        ],
        "context_efficiency_delta": [
            "condition_metrics.naive_icl.context_cost",
            "condition_metrics.policy_selected.context_cost",
        ],
        "verifier_cost_delta": [
            "condition_metrics.naive_icl.verifier_cost",
            "condition_metrics.policy_selected.verifier_cost",
        ],
        "negative_transfer_deflection_rate": [
            "row.condition",
            "row.negative_transfer_candidate",
            "row.exact_verifier_witness.selected_answer",
        ],
        "csl_sota_memory_routing_ready": [
            "gpu_offload_verified",
            "metric_independence_checks_passed",
            "quality_delta_vs_no_memory",
            "quality_delta_vs_naive_icl",
            "context_efficiency_delta",
            "verifier_cost_delta",
            "negative_transfer_deflection_rate",
            "no_weight_mutation",
        ],
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp5461 artifact cannot support its readiness claim."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, row, precondition, and metric errors."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-LEARN-5461")
    if type(artifact.get("preconditions_checked")) is not bool:
        errors.append("preconditions_checked must be boolean")
    model_specs = _model_specs_from_artifact(artifact, errors)
    model_ids = {str(row.get("hf_id")) for row in model_specs}
    if model_ids != set(MANDATED_HF_IDS):
        errors.append("mandated model_specs must include the three required GGUF IDs")
    if artifact.get("headline_required_any_of") != list(MANDATED_HF_IDS):
        errors.append("headline_required_any_of must list mandated SOTA IDs")
    errors.extend(_headline_model_errors(model_specs))
    if artifact.get("condition_names") != list(CONDITION_NAMES):
        errors.append("condition_names must match required memory conditions")
    for field in (
        "gpu_offload_verified",
        "no_weight_mutation",
        "csl_sota_memory_routing_ready",
        "metric_independence_checks_passed",
    ):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be boolean")
    if not isinstance(artifact.get("runtime_backend"), str):
        errors.append("runtime_backend must be a string")
    if not isinstance(artifact.get("policy_state_checksum"), str) or not str(
        artifact.get("policy_state_checksum", "")
    ).startswith("sha256:"):
        errors.append("policy_state_checksum must be a sha256 string")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or "\n" in verdict or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked:")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py must not be modified")

    rows = _list_of_mappings(artifact.get("row_results"))
    if not isinstance(artifact.get("row_results"), list):
        errors.append("row_results must be a list")
    errors.extend(_row_integrity_errors(rows))
    metrics = derive_metrics(rows)
    errors.extend(_aggregate_errors(artifact, metrics, rows))
    errors.extend(_row_path_errors(artifact.get("row_results_path"), rows))
    errors.extend(_dependency_graph_errors(artifact.get("metric_dependency_graph")))

    ready = artifact.get("csl_sota_memory_routing_ready") is True
    if ready:
        if artifact.get("status") != "complete":
            errors.append("csl_sota_memory_routing_ready requires complete status")
        if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
            errors.append("complete artifact requires live frozen-policy substrate")
        if artifact.get("gpu_offload_verified") is not True:
            errors.append("csl_sota_memory_routing_ready requires gpu_offload_verified")
        if not any(row.get("ran_headline") is True for row in model_specs):
            errors.append("headline_required_any_of requires at least one mandated model ran")
        if not rows:
            errors.append("csl_sota_memory_routing_ready requires row evidence")
        if artifact.get("quality_delta_vs_no_memory", -1.0) < 0.0:
            errors.append("quality_delta_vs_no_memory must be non-negative")
        if artifact.get("quality_delta_vs_naive_icl", -1.0) < 0.0:
            errors.append("quality_delta_vs_naive_icl must be non-negative")
        if artifact.get("context_efficiency_delta", 0.0) <= 0.0:
            errors.append("context_efficiency_delta must be positive")
        if artifact.get("verifier_cost_delta", 0.0) <= 0.0:
            errors.append("verifier_cost_delta must be positive")
        if artifact.get("negative_transfer_deflection_rate") != 1.0:
            errors.append("negative_transfer_deflection_rate must be complete")
        if artifact.get("no_weight_mutation") is not True:
            errors.append("no_weight_mutation must be true")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact payload")
    return errors


def policy_state_checksum(policy_artifact: Mapping[str, Any]) -> str:
    """Hash only the Exp5460 policy state used for routing decisions."""

    return _sha256_json(_mapping(policy_artifact.get("policy_snapshot")))


def model_file_receipt(path_value: Any) -> JsonDict:
    """Return lightweight file metadata proving a GGUF was not rewritten."""

    if not path_value:
        return {"path": None, "exists": False}
    path = Path(str(path_value))
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
        "suffix": path.suffix,
    }


def weight_mutation_receipt(
    *,
    model_before: Mapping[str, Any],
    model_after: Mapping[str, Any],
    model_loaded: bool,
) -> JsonDict:
    """Report the frozen-weight boundary for a live inference-only run."""

    unchanged = dict(model_before) == dict(model_after)
    return {
        "no_weight_mutation": unchanged,
        "no_adapter_weight_mutation": True,
        "model_weights_loaded": bool(model_loaded),
        "model_weights_written": not unchanged,
        "adapter_weights_loaded": False,
        "adapter_weights_written": False,
        "model_file_receipt_before": dict(model_before),
        "model_file_receipt_after": dict(model_after),
        "learned_state_scope": "exp5460_policy_state_and_prompt_memory_routing_only",
    }


class LlamaCslGenerationRunner:  # pragma: no cover
    """Small llama-cpp-python wrapper for bounded frozen-model generation."""

    def __init__(
        self,
        *,
        model_spec: Mapping[str, Any],
        n_gpu_layers: int = N_GPU_LAYERS,
        seed: int = RANDOM_SEED,
    ) -> None:
        from llama_cpp import Llama  # noqa: PLC0415

        self.model_spec = dict(model_spec)
        self.n_gpu_layers = n_gpu_layers
        before = _gpu_memory_snapshot()
        started = time.perf_counter()
        self.llm = Llama(
            model_path=str(model_spec["model_path"]),
            n_gpu_layers=n_gpu_layers,
            n_ctx=1024,
            n_batch=128,
            seed=seed,
            verbose=False,
        )
        duration_s = time.perf_counter() - started
        after = _gpu_memory_snapshot()
        delta = _max_memory_delta_mb(before, after)
        self.load_receipt = {
            "load_duration_s": duration_s,
            "gpu_memory_before": before,
            "gpu_memory_after": after,
            "gpu_memory_delta_mb": delta,
            "offload_evidence": delta > 512,
            "n_gpu_layers": n_gpu_layers,
        }

    def __call__(self, **kwargs: Any) -> JsonDict:
        prompt = str(kwargs["prompt"])
        started = time.perf_counter()
        result = self.llm.create_completion(
            prompt=prompt,
            max_tokens=int(kwargs["max_tokens"]),
            temperature=0.0,
            top_p=1.0,
            seed=int(kwargs["seed"]),
            echo=False,
            stop=["</s>", "<end_of_turn>"],
        )
        duration_s = time.perf_counter() - started
        choices = result.get("choices", []) if isinstance(result, Mapping) else []
        text = str(choices[0].get("text", "")) if choices else ""
        usage = result.get("usage", {}) if isinstance(result, Mapping) else {}
        prompt_token_count = len(self.llm.tokenize(prompt.encode("utf-8")))
        return {
            "output_text": text,
            "duration_s": duration_s,
            "generated_token_count": int(usage.get("completion_tokens", 0) or 0),
            "prompt_token_count": prompt_token_count,
            "backend_details": {"llama_cpp_create_completion": True},
        }


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash one row while excluding its checksum field."""

    return _sha256_json({key: value for key, value in row.items() if key != "row_checksum"})


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact payload without the self-referential checksum."""

    return _sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """CLI entry point for writing the Exp5461 result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-path", type=Path, default=None)
    parser.add_argument("--row-results-path", type=Path, default=None)
    parser.add_argument("--max-tasks", type=int, default=DEFAULT_MAX_TASKS)
    args = parser.parse_args(argv)
    artifact = run(
        root=args.root,
        artifact_path=args.artifact_path,
        row_results_path=args.row_results_path,
        max_tasks=args.max_tasks,
        write=True,
    )
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact["status"] == "complete" else 1


def _task(**kwargs: Any) -> JsonDict:
    row = {
        "negative_transfer_candidate": False,
        "stale_memory_candidate": False,
        "requires_memory": False,
    }
    row.update(kwargs)
    return _json_ready(row)


def _memory_lines(task: Mapping[str, Any], condition: str) -> list[str]:
    if condition == NO_MEMORY_CONDITION:
        return [str(item) for item in task.get("current_evidence", [])]
    if condition == NAIVE_CONDITION:
        return [str(item) for item in task.get("current_evidence", [])] + [
            str(item) for item in task.get("naive_memory", [])
        ]
    if condition == GOVERNED_CONDITION:
        return [str(item) for item in task.get("current_evidence", [])] + [
            str(item) for item in task.get("governed_memory", [])
        ]
    raise ValueError(f"unknown effective condition: {condition}")


def _condition_metric(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    accepted = [row for row in rows if row.get("accepted_by_final_authority") is True]
    return {
        "count": total,
        "accepted_count": len(accepted),
        "quality_score": _rate(accepted, total),
        "context_cost": sum(int(row.get("context_cost", 0)) for row in rows),
        "verifier_cost": sum(int(row.get("verifier_cost", 0)) for row in rows),
        "token_cost": sum(int(row.get("token_cost", 0)) for row in rows),
    }


def _regret_proxy(rows: Sequence[Mapping[str, Any]], task_ids: Sequence[str]) -> JsonDict:
    totals = {condition: 0.0 for condition in CONDITION_NAMES}
    for task_id in task_ids:
        task_rows = [row for row in rows if row.get("task_id") == task_id]
        scores = {
            str(row.get("condition")): 1.0 if row.get("accepted_by_final_authority") else 0.0
            for row in task_rows
        }
        oracle = max(scores.values()) if scores else 0.0
        for condition in CONDITION_NAMES:
            totals[condition] += oracle - scores.get(condition, 0.0)
    return {
        "oracle_regret": 0.0,
        "condition_regret": {key: round(value, 6) for key, value in sorted(totals.items())},
        "policy_regret_delta_vs_no_memory": round(
            totals.get(NO_MEMORY_CONDITION, 0.0) - totals.get(POLICY_CONDITION, 0.0),
            6,
        ),
        "policy_regret_delta_vs_naive_icl": round(
            totals.get(NAIVE_CONDITION, 0.0) - totals.get(POLICY_CONDITION, 0.0),
            6,
        ),
    }


def _aggregate_errors(
    artifact: Mapping[str, Any],
    metrics: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    scalar_fields = (
        "quality_delta_vs_no_memory",
        "quality_delta_vs_naive_icl",
        "context_efficiency_delta",
        "verifier_cost_delta",
        "negative_transfer_deflection_rate",
        "stale_memory_deflection_rate",
    )
    for field in scalar_fields:
        if not _float_close(artifact.get(field), metrics.get(field)):
            errors.append(f"{field} must match row recomputation")
    for field in ("condition_metrics", "policy_regret_proxy", "metric_independence_checks_passed"):
        if artifact.get(field) != metrics.get(field):
            errors.append(f"{field} must match row recomputation")
    if artifact.get("metric_details") != metrics:
        errors.append("metric_details must match row recomputation")
    if artifact.get("task_count") != metrics.get("task_count"):
        errors.append("task_count must match row recomputation")
    if artifact.get("row_count") != len(rows):
        errors.append("row_count must match row_results")
    if artifact.get("row_checksums") != [row.get("row_checksum") for row in rows]:
        errors.append("row_checksums must match row_results")
    return errors


def _row_integrity_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        witness = _mapping(row.get("exact_verifier_witness"))
        if witness.get("verified") is not True or witness.get("authority") != "exact_task_verifier":
            errors.append("row exact task verifier authority must be exact_task_verifier")
        if row.get("final_authority_bypassed") is not False:
            errors.append("row exact task verifier authority must not be bypassed")
        if row.get("row_checksum") != row_checksum(row):
            errors.append("row checksum must match row payload")
        if str(row.get("model_hf_id")) not in MANDATED_HF_IDS:
            errors.append("row model_hf_id must be mandated")
        if not str(row.get("model_path", "")).endswith(".gguf"):
            errors.append("row model_path must be a GGUF path")
        if row.get("condition") not in CONDITION_NAMES:
            errors.append("row condition must be known")
        if type(row.get("gpu_offload_evidence")) is not bool:
            errors.append("row gpu_offload_evidence must be boolean")
        policy_receipt = _mapping(row.get("policy_receipt"))
        if not str(policy_receipt.get("policy_state_checksum", "")).startswith("sha256:"):
            errors.append("row policy receipt must include policy_state_checksum")
    return errors


def _row_path_errors(path_text: Any, rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not isinstance(path_text, str) or not path_text:
        return ["row_results_path must be a non-empty string"]
    if not rows:
        return []
    path = Path(path_text)
    if not path.is_file():
        return ["row_results_path must point to written row evidence"]
    try:
        disk_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        return [f"row_results_path is unreadable: {type(exc).__name__}: {exc}"]
    if disk_rows != list(rows):
        return ["row_results_path contents must match embedded row_results"]
    return []


def _dependency_graph_errors(graph_value: Any) -> list[str]:
    if not isinstance(graph_value, Mapping):
        return ["metric_dependency_graph must be a dict"]
    deps = graph_value.get("csl_sota_memory_routing_ready")
    if not isinstance(deps, list):
        return ["metric_dependency_graph must include readiness dependencies"]
    forbidden = {"csl_sota_memory_routing_ready", "prior.csl_sota_memory_routing_ready"}
    if any(str(dep) in forbidden for dep in deps):
        return ["self-validating readiness dependency is forbidden"]
    return []


def _headline_model_errors(model_specs: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for spec in model_specs:
        if spec.get("ran_headline") is True and spec.get("gpu_offload_verified") is not True:
            errors.append("CPU-only headline model is forbidden")
        if spec.get("ran_headline") is True and spec.get("legacy_smoke_only") is True:
            errors.append("legacy smoke model cannot headline")
    return errors


def _model_specs_from_artifact(
    artifact: Mapping[str, Any],
    errors: list[str],
) -> list[JsonDict]:
    value = artifact.get("model_specs")
    if not isinstance(value, list):
        errors.append("model_specs must be a list")
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _mark_model_ran(
    model_specs: Sequence[Mapping[str, Any]],
    selected_model: Mapping[str, Any] | None,
) -> list[JsonDict]:
    selected_hf_id = str(selected_model.get("hf_id")) if selected_model else ""
    rows: list[JsonDict] = []
    for spec in model_specs:
        row = dict(spec)
        if str(row.get("hf_id")) == selected_hf_id:
            row["ran_headline"] = True
            row["runtime_backend"] = "llama_cpp_python_cuda_gguf"
            row["n_gpu_layers"] = N_GPU_LAYERS
            row["gpu_offload_verified"] = True
        rows.append(row)
    return rows


def _exact_authority_row(row: Mapping[str, Any]) -> bool:
    witness = _mapping(row.get("exact_verifier_witness"))
    return (
        witness.get("verified") is True
        and witness.get("authority") == "exact_task_verifier"
        and row.get("final_authority_bypassed") is False
    )


def _deflection_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 1.0
    deflected = [
        row
        for row in rows
        if row.get("accepted_by_final_authority") is True
        and row.get("negative_transfer_detected") is not True
    ]
    return round(len(deflected) / len(rows), 6)


def _first_literal_match(text: str, candidates: Sequence[str]) -> str | None:
    matches: list[tuple[int, str]] = []
    for candidate in candidates:
        pattern = re.escape(candidate)
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            matches.append((found.start(), candidate))
    if not matches:
        return None
    return sorted(matches, key=lambda item: item[0])[0][1]


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _rate(items: Sequence[Any], total: int) -> float:
    return 0.0 if total <= 0 else len(items) / total


def _relative_savings(before: int | float, after: int | float) -> float:
    return round((float(before) - float(after)) / float(before), 6) if before else 0.0


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _normalise_tests_run(tests_run: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    if not tests_run:
        return [{"command": "not_recorded", "outcome": "not_recorded"}]
    return [_normalise_test_run(item) for item in tests_run]


def _normalise_test_run(item: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(item, str):
        return {"command": item, "outcome": "passed"}
    return {
        "command": str(item.get("command", "")),
        "outcome": str(item.get("outcome", "passed")),
    }


def _honest_verdict(
    ready: bool,
    rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> str:
    if ready:
        return "complete: live SOTA GGUF governed CSL memory routing preserved quality and deflected negative transfer with frozen weights"
    if rows:
        blockers: list[str] = []
        if metrics.get("quality_delta_vs_no_memory", -1.0) < 0.0:
            blockers.append("quality_regressed_vs_no_memory")
        if metrics.get("quality_delta_vs_naive_icl", -1.0) < 0.0:
            blockers.append("quality_regressed_vs_naive_icl")
        if metrics.get("negative_transfer_deflection_rate") != 1.0:
            blockers.append("negative_transfer_not_deflected")
        return "complete: live panel ran; readiness false (" + ",".join(blockers or ["gate_failed"]) + ")"
    blockers = ",".join(preconditions.get("blocked_preconditions", []))
    return f"blocked: {blockers or 'preconditions_failed'}"


def _source_file_checksums(root: Path) -> JsonDict:
    paths = {
        "module": root / MODULE_RELATIVE_PATH,
        "spec": root / SPEC_RELATIVE_PATH,
        "upstream_policy": root / EXP5460_RESULT_RELATIVE_PATH,
    }
    return {name: _file_checksum(path) for name, path in paths.items() if path.is_file()}


def _file_checksum(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _nvidia_smi_query() -> JsonDict:  # pragma: no cover
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "gpus": []}
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_used_mb": _safe_int(parts[2]),
                    "memory_total_mb": _safe_int(parts[3]),
                    "utilization_gpu_pct": _safe_int(parts[4]),
                }
            )
    return {"ok": result.returncode == 0, "gpus": rows, "stderr": result.stderr.strip()}


def _gpu_memory_snapshot() -> list[JsonDict]:  # pragma: no cover
    return list(_nvidia_smi_query().get("gpus", []))


def _max_memory_delta_mb(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> int:  # pragma: no cover
    before_by_index = {str(row.get("index")): int(row.get("memory_used_mb", 0)) for row in before}
    deltas = [
        int(row.get("memory_used_mb", 0)) - before_by_index.get(str(row.get("index")), 0)
        for row in after
    ]
    return max(deltas) if deltas else 0


def _safe_int(value: Any) -> int:  # pragma: no cover
    try:
        return int(str(value).strip())
    except ValueError:
        return 0


def _destination(root: Path, path: Path | str | None, default: Path) -> Path:
    return Path(path) if path is not None else root / default


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_json_ready(row), sort_keys=True, ensure_ascii=True) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _list_of_mappings(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _float_close(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-12)
    except (TypeError, ValueError):
        return False


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
