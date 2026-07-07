#!/usr/bin/env python3
"""Exp 5366 live grammar-budgeted SOTA structured-output protocol.

Spec refs: REQ-VERIFY-5366, SCENARIO-VERIFY-5366.

This runner is deliberately a runtime/protocol gate, not a model-quality
benchmark. It copies the Exp 5365 grammar-budget gate, refuses headline runs
without a non-retired GPU/offload path, uses only llama.cpp-compatible GGUF
loading, and measures whether a mandated local SOTA model can produce final
schema JSON without collapsing truncation, schema, and semantic failures.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback
from typing import Any

from carnot import experiment_5351_trigger_constrain_structured_protocol_v488 as exp5351
from carnot import experiment_5365_grammar_budget_protocol_preflight_v489 as exp5365
from carnot.inference.sota_models import resolve_cached_gguf


JsonDict = dict[str, Any]
ModelResolver = Callable[[str, str], str | None]
RuntimeProbe = Callable[..., JsonDict]
GenerationProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5366_live_grammar_budgeted_sota_protocol_v489"
MILESTONE = "2026.07.489"
RESULT_RELATIVE_PATH = Path("results/experiment_5366_live_grammar_budgeted_sota_protocol_v489.json")
SCHEMA = "carnot.experiment_5366.live_grammar_budgeted_sota_protocol.v489"
SPEC_REFS = ("REQ-VERIFY-5366", "SCENARIO-VERIFY-5366")
RANDOM_SEED = 5366
TERMINAL_PREFIXES = ("complete:", "blocked_")
DEFAULT_QUANTIZATION = "Q4_K_M"
MIN_PARSE_SUCCESS_RATE = 0.95
MIN_SCHEMA_SUCCESS_RATE = 0.90
MIN_FINAL_JSON_EXTRACTION_RATE = 0.95
MIN_CLEAN_METHODOLOGY_DURATION_S = 60.0

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
SELECTION_ROLE_ORDER = ("flagship_dense", "middle_moe", "flagship_moe")

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "grammar_budget_protocol_ready",
    "structured_protocol_clean",
    "MODEL_SPECS",
    "selected_model_spec",
    "inference_substrate",
    "gpu_or_offload_receipt",
    "no_autotokenizer_used",
    "prompt_count",
    "parse_success_rate",
    "schema_success_rate",
    "final_json_extraction_rate",
    "semantic_success_rate",
    "truncation_failure_rate",
    "unsafe_false_accepts",
    "completion_slack_min_tokens",
    "methodology_duration_s",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "complete only if live local SOTA inference ran; blocked if preconditions fail.",
    "grammar_budget_protocol_ready": "copied from Exp5365 gate source for audit.",
    "structured_protocol_clean": (
        "boolean gate for downstream constraint-tax work; true only if "
        "parse_success_rate>=0.95, schema_success_rate>=0.90, "
        "final_json_extraction_rate>=0.95, unsafe_false_accepts=0, and "
        "methodology_duration_s>=60."
    ),
    "MODEL_SPECS": "list containing the mandated local GGUF model specs considered for headline results.",
    "selected_model_spec": "exact model spec used for headline measurements.",
    "inference_substrate": "concrete runtime path, GPU/offload status, and GGUF loader family.",
    "gpu_or_offload_receipt": "machine-readable evidence that this was not the retired CPU-only headline path.",
    "no_autotokenizer_used": "must be true for GGUF repos.",
    "prompt_count": "number of live prompts evaluated.",
    "parse_success_rate": "fraction of responses with parseable JSON.",
    "schema_success_rate": "fraction of responses satisfying the schema.",
    "final_json_extraction_rate": "fraction with unambiguous final JSON extraction.",
    "semantic_success_rate": "fraction satisfying task semantics after schema validity.",
    "truncation_failure_rate": "fraction classified as token-budget truncation failures.",
    "unsafe_false_accepts": "count of invalid/unsafe outputs accepted as valid.",
    "completion_slack_min_tokens": "minimum JSON completion slack observed or inherited from Exp5365.",
    "methodology_duration_s": "live measurement duration excluding planning prose.",
    "honest_verdict": "one-line clean/block verdict.",
}


def field_provenance() -> dict[str, JsonDict]:
    """Return principle annotations for every required Exp 5366 field."""

    return {
        field: {
            "principle": principle,
            "satisfied_by": "Exp 5366 live grammar-budgeted SOTA protocol runner",
        }
        for field, principle in FIELD_PRINCIPLES.items()
    }


def default_model_specs_unresolved() -> list[JsonDict]:
    """Return mandated model rows before any local GGUF cache is trusted."""

    return [_unresolved_model_spec(spec) for spec in MANDATED_MODEL_SPECS]


def resolve_model_specs(model_resolver: ModelResolver = resolve_cached_gguf) -> list[JsonDict]:
    """Resolve all mandated GGUF specs without touching transformers or tokenizers."""

    rows: list[JsonDict] = []
    for spec in MANDATED_MODEL_SPECS:
        row = _unresolved_model_spec(spec)
        path_text = model_resolver(str(spec["hf_id"]), str(spec.get("quantization", DEFAULT_QUANTIZATION)))
        if not path_text:
            rows.append(row)
            continue
        path = Path(path_text)
        row["model_path"] = str(path)
        try:
            if not path.is_file():
                row["blocked_preconditions"] = ["model_file_missing"]
            else:
                row["file_receipts"] = _file_receipts(path)
                row["metadata"] = read_gguf_header(path)
                row["status"] = "local_gguf_resolved"
        except Exception as exc:
            row["status"] = "blocked_metadata_unreadable"
            row["blocked_preconditions"] = [f"metadata_unreadable:{type(exc).__name__}: {exc}"]
        rows.append(row)
    return rows


def select_headline_model(model_specs: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    """Select one locally resolved mandated model, preferring prior dense receipts."""

    by_role = {str(row.get("role")): row for row in model_specs}
    for role in SELECTION_ROLE_ORDER:
        row = by_role.get(role)
        if isinstance(row, Mapping) and row.get("status") == "local_gguf_resolved":
            return dict(row)
    return None


def protocol_settings_from_exp5365(exp5365_artifact: Mapping[str, Any]) -> JsonDict:
    """Inherit schema grammar, completion slack, and max-token budget from Exp 5365."""

    variant = dict(exp5351.DEFAULT_PROTOCOL_VARIANTS[0])
    cases = exp5365_artifact.get("completion_budget_cases")
    budgets = [
        int(case["max_tokens"])
        for case in (cases if isinstance(cases, list) else [])
        if isinstance(case, Mapping) and isinstance(case.get("max_tokens"), int)
    ]
    max_tokens = min(budgets) if budgets else int(variant.get("n_predict") or 1024)
    grammar_summary = exp5365_artifact.get("schema_grammar_summary")
    grammar_summary = dict(grammar_summary) if isinstance(grammar_summary, Mapping) else {}
    variant["n_predict"] = max_tokens
    variant["max_tokens"] = max_tokens
    variant["completion_slack_min_tokens"] = int(
        exp5365_artifact.get("completion_slack_min_tokens")
        if isinstance(exp5365_artifact.get("completion_slack_min_tokens"), int)
        else -1
    )
    variant["schema_grammar_summary"] = grammar_summary
    variant["grammar_string"] = _gbnf_for_exp5351_schema()
    variant["grammar_backend"] = "llama_cpp_gbnf_from_exp5365_schema_summary"
    return variant


def build_live_prompt(prompt_spec: Mapping[str, Any], variant: Mapping[str, Any]) -> str:
    """Build the live prompt while preserving Exp 5351 final-sentinel semantics."""

    return exp5351.build_protocol_prompt(prompt_spec, variant)


def score_live_output(
    *,
    prompt_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    output_text: str,
    completed: bool,
    timed_out: bool = False,
    generated_token_count: int | None = None,
) -> JsonDict:
    """Score one model response without merging truncation, schema, and semantics."""

    base = exp5351.score_protocol_output(
        prompt_spec=prompt_spec,
        variant=variant,
        output_text=output_text,
        completed=completed,
    )
    payload = base.get("parsed_object")
    expected = prompt_spec.get("expected")
    expected = expected if isinstance(expected, Mapping) else {}
    schema_success = bool(base.get("schema_success"))
    semantic_success = bool(
        schema_success
        and isinstance(payload, Mapping)
        and all(payload.get(key) == value for key, value in expected.items())
    )
    truncation_failure = _looks_like_truncation(
        output_text=output_text,
        variant=variant,
        completed=completed,
        timed_out=timed_out,
        generated_token_count=generated_token_count,
    )
    if truncation_failure:
        failure_class = "truncation"
    elif not base.get("parse_success"):
        failure_class = "parse"
    elif not schema_success:
        failure_class = "schema"
    elif not semantic_success:
        failure_class = "semantic"
    else:
        failure_class = "accepted"
    return {
        **base,
        "semantic_success": semantic_success,
        "truncation_failure": truncation_failure,
        "failure_class": failure_class,
    }


def summarize_scores(scores: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate prompt-row scores into protocol rates."""

    if not scores:
        return {
            "prompt_count": 0,
            "parse_success_rate": 0.0,
            "schema_success_rate": 0.0,
            "final_json_extraction_rate": 0.0,
            "semantic_success_rate": 0.0,
            "truncation_failure_rate": 0.0,
        }
    schema_denominator = sum(1 for row in scores if row.get("schema_success") is True)
    return {
        "prompt_count": len(scores),
        "parse_success_rate": _rate(scores, "parse_success"),
        "schema_success_rate": _rate(scores, "schema_success"),
        "final_json_extraction_rate": _rate(scores, "final_json_extraction_success"),
        "semantic_success_rate": (
            0.0 if schema_denominator == 0 else sum(1 for row in scores if row.get("semantic_success") is True) / schema_denominator
        ),
        "truncation_failure_rate": _rate(scores, "truncation_failure"),
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    artifact_path: Path | str | None = None,
    exp5365_path: Path | str | None = None,
    exp5365_artifact: Mapping[str, Any] | None = None,
    model_resolver: ModelResolver = resolve_cached_gguf,
    runtime_probe: RuntimeProbe = None,  # type: ignore[assignment]
    generation_probe: GenerationProbe = None,  # type: ignore[assignment]
    tests_run: Sequence[Any] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp 5366 or write a blocked artifact when preconditions fail."""

    started = time.perf_counter()
    root_path = Path(root)
    destination = Path(artifact_path) if artifact_path is not None else root_path / RESULT_RELATIVE_PATH
    if not destination.is_absolute():
        destination = root_path / destination
    gate_path = Path(exp5365_path) if exp5365_path is not None else root_path / exp5365.RESULT_RELATIVE_PATH
    gate = dict(exp5365_artifact or _load_json(gate_path))
    protocol_settings = protocol_settings_from_exp5365(gate)
    grammar_ready = bool(gate.get("grammar_budget_protocol_ready"))
    model_specs = default_model_specs_unresolved()
    selected_model: JsonDict | None = None
    runtime_receipt: JsonDict = {
        "gpu_visible": False,
        "gguf_runtime_available": False,
        "gguf_loader_family": "not_checked",
        "offload_evidence": False,
        "non_retired_gpu_or_offload_path": False,
        "blocked_preconditions": ["exp5365_grammar_budget_protocol_not_ready"],
    }
    generation_rows: list[JsonDict] = []
    blockers: list[str] = []
    runtime_probe = default_runtime_probe if runtime_probe is None else runtime_probe
    generation_probe = default_generation_probe if generation_probe is None else generation_probe

    if not grammar_ready:
        blockers.append("exp5365_grammar_budget_protocol_not_ready")
    else:
        model_specs = resolve_model_specs(model_resolver)
        selected_model = select_headline_model(model_specs)
        if selected_model is None:
            blockers.append("no_mandated_sota_gguf_resolved")
        runtime_receipt = _normalise_runtime_receipt(
            runtime_probe(
                model_specs=model_specs,
                selected_model_spec=selected_model,
                protocol_settings=protocol_settings,
            )
        )
        blockers.extend(str(item) for item in runtime_receipt.get("blocked_preconditions", ()))
        if not runtime_receipt.get("non_retired_gpu_or_offload_path"):
            blockers.append("non_retired_gpu_or_offload_path_unavailable")
        blockers = list(dict.fromkeys(blockers))
        runtime_receipt["blocked_preconditions"] = blockers
        if not blockers and selected_model is not None:
            generation_rows = _run_generation_rows(
                model_spec=selected_model,
                variant=protocol_settings,
                generation_probe=generation_probe,
            )

    scores = [dict(row["score"]) for row in generation_rows]
    metrics = summarize_scores(scores)
    prompt_duration_s = sum(float(row.get("wall_clock_s") or 0.0) for row in generation_rows)
    methodology_duration_s = round(
        max(prompt_duration_s, time.perf_counter() - started if generation_rows else 0.0),
        6,
    )
    if generation_rows and not any(row.get("completed") is True for row in generation_rows):
        blockers = list(dict.fromkeys([*blockers, "live_generation_failed"]))
        runtime_receipt["blocked_preconditions"] = blockers
    unsafe_false_accepts = (
        exp5351.unsafe_false_accept_count(protocol_settings, exp5351.DEFAULT_CALIBRATION_PROMPTS)
        if generation_rows
        else 0
    )
    structured_clean = _structured_protocol_clean(
        metrics=metrics,
        unsafe_false_accepts=unsafe_false_accepts,
        methodology_duration_s=methodology_duration_s,
    )
    live_ran = bool(
        generation_rows
        and not blockers
        and selected_model is not None
        and any(row.get("completed") is True for row in generation_rows)
    )
    status = "complete" if live_ran else "blocked"
    inference_substrate = _inference_substrate(
        runtime_receipt=runtime_receipt,
        selected_model=selected_model,
        live_ran=live_ran,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "status": status,
        "grammar_budget_protocol_ready": grammar_ready,
        "structured_protocol_clean": structured_clean,
        "MODEL_SPECS": model_specs,
        "selected_model_spec": selected_model if live_ran else None,
        "inference_substrate": inference_substrate,
        "gpu_or_offload_receipt": runtime_receipt,
        "no_autotokenizer_used": True,
        "prompt_count": metrics["prompt_count"],
        "parse_success_rate": metrics["parse_success_rate"],
        "schema_success_rate": metrics["schema_success_rate"],
        "final_json_extraction_rate": metrics["final_json_extraction_rate"],
        "semantic_success_rate": metrics["semantic_success_rate"],
        "truncation_failure_rate": metrics["truncation_failure_rate"],
        "unsafe_false_accepts": unsafe_false_accepts,
        "completion_slack_min_tokens": int(protocol_settings["completion_slack_min_tokens"]),
        "methodology_duration_s": methodology_duration_s,
        "honest_verdict": _honest_verdict(
            status=status,
            grammar_ready=grammar_ready,
            structured_clean=structured_clean,
            blockers=blockers,
        ),
        "protocol_settings": _artifact_protocol_settings(protocol_settings),
        "prompt_results": scores,
        "generation_receipts": generation_rows,
        "source_artifacts": [
            {
                "path": exp5365.RESULT_RELATIVE_PATH.as_posix(),
                "used": True,
                "grammar_budget_protocol_ready": grammar_ready,
            }
        ],
        "tests_run": list(tests_run or []),
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.perf_counter() - started, 6),
        "field_provenance": field_provenance(),
    }
    artifact["reproducibility_checksum"] = _sha16(
        _stable_json(
            {
                "experiment_id": EXPERIMENT_ID,
                "grammar_ready": grammar_ready,
                "model_specs": model_specs,
                "selected_model": artifact["selected_model_spec"],
                "prompt_results": scores,
                "seed": RANDOM_SEED,
            }
        )
    )
    validate_artifact(artifact)
    if write:
        _write_json(destination, artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and gate errors for the Exp 5366 artifact contract."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("status") not in {"complete", "blocked"}:
        errors.append("status must be complete or blocked")
    for field in ("grammar_budget_protocol_ready", "structured_protocol_clean"):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be boolean")
    if not _model_specs_cover_mandated(artifact.get("MODEL_SPECS")):
        errors.append("MODEL_SPECS must contain all mandated SOTA GGUF specs")
    selected = artifact.get("selected_model_spec")
    if selected is not None and (
        not isinstance(selected, Mapping) or selected.get("hf_id") not in MANDATED_HF_IDS
    ):
        errors.append("selected_model_spec must be null or one mandated model spec")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        errors.append("inference_substrate must be object")
        substrate = {}
    receipt = artifact.get("gpu_or_offload_receipt")
    if not isinstance(receipt, Mapping):
        errors.append("gpu_or_offload_receipt must be object")
    if artifact.get("no_autotokenizer_used") is not True:
        errors.append("no_autotokenizer_used must be true")
    if not _non_negative_int(artifact.get("prompt_count")):
        errors.append("prompt_count must be non-negative integer")
    for field in (
        "parse_success_rate",
        "schema_success_rate",
        "final_json_extraction_rate",
        "semantic_success_rate",
        "truncation_failure_rate",
    ):
        if not _rate_is_valid(artifact.get(field)):
            errors.append(f"{field} must be in [0, 1]")
    if not _non_negative_int(artifact.get("unsafe_false_accepts")):
        errors.append("unsafe_false_accepts must be non-negative integer")
    if not isinstance(artifact.get("completion_slack_min_tokens"), int):
        errors.append("completion_slack_min_tokens must be integer")
    if not isinstance(artifact.get("methodology_duration_s"), int | float):
        errors.append("methodology_duration_s must be numeric")
    honest = artifact.get("honest_verdict")
    if not isinstance(honest, str) or not honest.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked_")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping) or any(field not in provenance for field in REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    if artifact.get("status") == "complete" and substrate.get("live_local_sota_inference_ran") is not True:
        errors.append("complete status requires live local SOTA inference")
    if artifact.get("status") == "complete" and selected is None:
        errors.append("complete status requires selected_model_spec")
    if artifact.get("grammar_budget_protocol_ready") is False and artifact.get("status") == "complete":
        errors.append("complete status requires Exp5365 grammar budget readiness")
    if artifact.get("structured_protocol_clean") is True and not _structured_protocol_clean(
        metrics={
            "parse_success_rate": artifact.get("parse_success_rate"),
            "schema_success_rate": artifact.get("schema_success_rate"),
            "final_json_extraction_rate": artifact.get("final_json_extraction_rate"),
        },
        unsafe_false_accepts=artifact.get("unsafe_false_accepts"),
        methodology_duration_s=artifact.get("methodology_duration_s"),
    ):
        errors.append("structured_protocol_clean thresholds are not satisfied")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5366 artifact cannot support downstream gating."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError("; ".join(errors))


def default_runtime_probe(**kwargs: Any) -> JsonDict:  # pragma: no cover - host GPU probe
    """Probe CUDA/GPU visibility and llama.cpp GPU-offload support."""

    del kwargs
    nvidia_smi = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10.0,
    )
    raw_nvidia_smi = _run_command(["nvidia-smi"], timeout_s=10.0)
    llama_cpp = _llama_cpp_import_receipt()
    gpu_visible = bool(nvidia_smi.get("ok") and str(nvidia_smi.get("stdout") or "").strip())
    runtime_available = bool(llama_cpp.get("import_ok"))
    offload_supported = bool(llama_cpp.get("gpu_offload_supported"))
    blockers: list[str] = []
    if not gpu_visible:
        blockers.append("gpu_not_visible")
    if not runtime_available:
        blockers.append("llama_cpp_unavailable")
    if runtime_available and not offload_supported:
        blockers.append("llama_cpp_cpu_only")
    return {
        "gpu_visible": gpu_visible,
        "gguf_runtime_available": runtime_available,
        "gguf_loader_family": "llama.cpp/llama-cpp-python",
        "llama_cpp": llama_cpp,
        "nvidia_smi": nvidia_smi,
        "raw_nvidia_smi": raw_nvidia_smi,
        "llama_cpp_gpu_offload_supported": offload_supported,
        "offload_evidence": offload_supported,
        "non_retired_gpu_or_offload_path": bool(gpu_visible and runtime_available and offload_supported),
        "blocked_preconditions": blockers,
    }


def default_generation_probe(
    *,
    model_spec: Mapping[str, Any],
    prompt_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    prompt: str,
    max_tokens: int,
    run_index: int,
    timeout_s: float = 900.0,
) -> JsonDict:  # pragma: no cover - live local GGUF inference
    """Run one live llama.cpp Python GGUF generation with GPU memory receipts."""

    del prompt_spec, timeout_s
    from llama_cpp import Llama, LlamaGrammar  # noqa: PLC0415

    model_path = str(model_spec["model_path"])
    started = time.perf_counter()
    before = _gpu_memory_snapshot()
    llm = _cached_llama(model_path=model_path, seed=RANDOM_SEED + run_index)
    after_load = _gpu_memory_snapshot()
    grammar = LlamaGrammar.from_string(str(variant["grammar_string"]), verbose=False)
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        grammar=grammar,
    )
    after_generate = _gpu_memory_snapshot()
    output = _llama_response_text(response)
    delta = max(_total_gpu_used(after_load), _total_gpu_used(after_generate)) - _total_gpu_used(before)
    return {
        "completed": True,
        "timed_out": False,
        "returncode": 0,
        "stdout": output,
        "stderr": "",
        "wall_clock_s": round(time.perf_counter() - started, 6),
        "generated_token_count": exp5365.estimate_tokens(output),
        "gpu_memory_receipts": {
            "before": before,
            "after_load": after_load,
            "after_generate": after_generate,
            "max_memory_delta_mb": delta,
            "offload_evidence": delta > 128,
        },
    }


def read_gguf_header(model_path: str | Path) -> JsonDict:
    """Read enough GGUF metadata to reject pointer files before inference."""

    with Path(model_path).open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24:
        raise ValueError("truncated GGUF header")
    if header[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    version, tensor_count, metadata_kv_count = struct.unpack("<IQQ", header[4:24])
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version: {version}")
    return {
        "magic": "GGUF",
        "version": int(version),
        "tensor_count": int(tensor_count),
        "metadata_kv_count": int(metadata_kv_count),
    }


def _unresolved_model_spec(spec: Mapping[str, Any]) -> JsonDict:
    hf_id = str(spec["hf_id"])
    return {
        "role": str(spec["role"]),
        "hf_id": hf_id,
        "quantization": str(spec.get("quantization", DEFAULT_QUANTIZATION)),
        "cache_path": str(Path.home() / ".cache" / "huggingface" / "hub" / f"models--{hf_id.replace('/', '--')}"),
        "model_path": None,
        "status": "missing_local_gguf",
        "headline_eligible": True,
        "gguf_loader_family": "llama.cpp",
        "autotokenizer_used": False,
        "file_receipts": None,
        "metadata": None,
        "blocked_preconditions": [],
    }


def _file_receipts(path: Path) -> JsonDict:
    size = path.stat().st_size
    head = path.read_bytes()[: 1024 * 1024]
    return {
        "path": str(path),
        "size_bytes": size,
        "checksum_head_1m_sha256": hashlib.sha256(head).hexdigest(),
        "checksum_note": "full_sha256_skipped_for_large_file_head_1m_recorded",
    }


def _run_generation_rows(
    *,
    model_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    generation_probe: GenerationProbe,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for run_index, prompt_spec in enumerate(exp5351.DEFAULT_CALIBRATION_PROMPTS, start=1):
        prompt = build_live_prompt(prompt_spec, variant)
        try:
            raw = generation_probe(
                model_spec=model_spec,
                prompt_spec=prompt_spec,
                variant=variant,
                prompt=prompt,
                max_tokens=int(variant["max_tokens"]),
                run_index=run_index,
            )
        except Exception as exc:  # pragma: no cover - defensive live runtime path
            raw = {
                "completed": False,
                "timed_out": False,
                "returncode": None,
                "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                "wall_clock_s": 0.0,
                "generated_token_count": 0,
                "gpu_memory_receipts": {},
            }
        rows.append(
            _normalise_generation_row(
                raw,
                model_spec=model_spec,
                prompt_spec=prompt_spec,
                variant=variant,
                prompt=prompt,
                run_index=run_index,
            )
        )
    return rows


def _normalise_generation_row(
    raw: Mapping[str, Any],
    *,
    model_spec: Mapping[str, Any],
    prompt_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    prompt: str,
    run_index: int,
) -> JsonDict:
    stdout = str(raw.get("stdout") or "")
    stderr = str(raw.get("stderr") or "")
    generated_tokens = int(raw.get("generated_token_count") or exp5365.estimate_tokens(stdout))
    completed = bool(raw.get("completed")) and raw.get("returncode", 0) == 0 and not raw.get("timed_out")
    score = score_live_output(
        prompt_spec=prompt_spec,
        variant=variant,
        output_text=stdout,
        completed=completed,
        timed_out=bool(raw.get("timed_out")),
        generated_token_count=generated_tokens,
    )
    return {
        "run_index": run_index,
        "prompt_id": str(prompt_spec["prompt_id"]),
        "hf_id": str(model_spec["hf_id"]),
        "model_role": str(model_spec["role"]),
        "prompt_checksum": _sha16(prompt),
        "output_checksum": _sha16(stdout),
        "completed": completed,
        "timed_out": bool(raw.get("timed_out")),
        "returncode": raw.get("returncode"),
        "wall_clock_s": float(raw.get("wall_clock_s") or 0.0),
        "generated_token_count": generated_tokens,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-1000:],
        "gpu_memory_receipts": dict(raw.get("gpu_memory_receipts") or {}),
        "score": score,
    }


def _normalise_runtime_receipt(receipt: Mapping[str, Any]) -> JsonDict:
    out = dict(receipt)
    out.setdefault("gpu_visible", False)
    out.setdefault("gguf_runtime_available", False)
    out.setdefault("gguf_loader_family", "llama.cpp")
    out.setdefault("offload_evidence", False)
    out.setdefault("non_retired_gpu_or_offload_path", False)
    out.setdefault("blocked_preconditions", [])
    out["blocked_preconditions"] = list(out.get("blocked_preconditions") or [])
    return out


def _inference_substrate(
    *,
    runtime_receipt: Mapping[str, Any],
    selected_model: Mapping[str, Any] | None,
    live_ran: bool,
) -> JsonDict:
    return {
        "kind": "live_llm_inference" if live_ran else "blocked_preconditions",
        "loader_family": str(runtime_receipt.get("gguf_loader_family") or "llama.cpp"),
        "gguf_loader_family": str(runtime_receipt.get("gguf_loader_family") or "llama.cpp"),
        "gpu_or_offload_status": (
            "non_retired_gpu_or_offload_path"
            if runtime_receipt.get("non_retired_gpu_or_offload_path")
            else "blocked_or_cpu_only"
        ),
        "live_local_sota_inference_ran": live_ran,
        "selected_model_hf_id": None if selected_model is None else selected_model.get("hf_id"),
    }


def _artifact_protocol_settings(variant: Mapping[str, Any]) -> JsonDict:
    return {
        "variant_id": variant.get("variant_id"),
        "max_tokens": int(variant.get("max_tokens") or variant.get("n_predict") or 0),
        "completion_slack_min_tokens": int(variant.get("completion_slack_min_tokens") or -1),
        "sentinel": variant.get("sentinel"),
        "end_sentinel": variant.get("end_sentinel"),
        "grammar_backend": variant.get("grammar_backend"),
        "schema_grammar_summary": variant.get("schema_grammar_summary"),
    }


def _structured_protocol_clean(
    *,
    metrics: Mapping[str, Any],
    unsafe_false_accepts: Any,
    methodology_duration_s: Any,
) -> bool:
    return bool(
        _numeric(metrics.get("parse_success_rate")) >= MIN_PARSE_SUCCESS_RATE
        and _numeric(metrics.get("schema_success_rate")) >= MIN_SCHEMA_SUCCESS_RATE
        and _numeric(metrics.get("final_json_extraction_rate")) >= MIN_FINAL_JSON_EXTRACTION_RATE
        and unsafe_false_accepts == 0
        and _numeric(methodology_duration_s) >= MIN_CLEAN_METHODOLOGY_DURATION_S
    )


def _honest_verdict(
    *,
    status: str,
    grammar_ready: bool,
    structured_clean: bool,
    blockers: Sequence[str],
) -> str:
    if not grammar_ready:
        return "blocked_exp5365_grammar_budget_protocol_not_ready"
    if status == "blocked":
        first = str(blockers[0]) if blockers else "preconditions_failed"
        return f"blocked_preconditions: {first}"
    if structured_clean:
        return "complete: structured_protocol_clean=true"
    return "blocked_structured_protocol_clean_false: live SOTA inference ran but clean gate failed"


def _looks_like_truncation(
    *,
    output_text: str,
    variant: Mapping[str, Any],
    completed: bool,
    timed_out: bool,
    generated_token_count: int | None,
) -> bool:
    if timed_out:
        return True
    if generated_token_count is not None and generated_token_count >= int(variant.get("max_tokens") or variant.get("n_predict") or 0):
        return True
    sentinel = str(variant.get("sentinel") or "FINAL_JSON:")
    end_sentinel = str(variant.get("end_sentinel") or "END_FINAL_JSON")
    marker = output_text.rfind(sentinel)
    if marker < 0:
        return False
    segment = output_text[marker + len(sentinel) :]
    return bool(not completed and "{" in segment and end_sentinel not in segment)


def _model_specs_cover_mandated(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    return len(value) == len(MANDATED_HF_IDS) and {row.get("hf_id") for row in value if isinstance(row, Mapping)} == set(MANDATED_HF_IDS)


def _gbnf_for_exp5351_schema() -> str:
    return r'''
root ::= "FINAL_JSON:" ws "{" ws "\"answer\"" ws ":" ws string ws "," ws "\"facts\"" ws ":" ws array ws "," ws "\"id\"" ws ":" ws string ws "}" ws ("END_FINAL_JSON")?
array ::= "[" ws (string (ws "," ws string)*)? ws "]"
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" (["\\/bfnrt] | "u" hex hex hex hex))*
hex ::= [0-9a-fA-F]
ws ::= [ \t\n]*
'''.strip()


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return 0.0 if not rows else sum(1 for row in rows if row.get(key) is True) / len(rows)


def _rate_is_valid(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and 0.0 <= float(value) <= 1.0


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _numeric(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else -1.0


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha16(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _load_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def _run_command(command: Sequence[str], timeout_s: float = 10.0) -> JsonDict:  # pragma: no cover
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": list(command),
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": round(time.perf_counter() - started, 6),
        }
    except Exception as exc:
        return {
            "command": list(command),
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.perf_counter() - started, 6),
        }


def _llama_cpp_import_receipt() -> JsonDict:  # pragma: no cover
    spec = importlib.util.find_spec("llama_cpp")
    version: str | None
    try:
        version = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        version = None
    support: bool | None = None
    error: str | None = None
    if spec is not None:
        try:
            from llama_cpp import llama_cpp as backend  # noqa: PLC0415

            support = bool(getattr(backend, "llama_supports_gpu_offload", lambda: False)())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    return {
        "import_ok": spec is not None,
        "origin": spec.origin if spec else None,
        "version": version,
        "gpu_offload_supported": support,
        "gpu_offload_support_error": error,
    }


_LLAMA_CACHE: dict[str, Any] = {}


def _cached_llama(*, model_path: str, seed: int) -> Any:  # pragma: no cover
    if model_path not in _LLAMA_CACHE:
        from llama_cpp import Llama  # noqa: PLC0415

        _LLAMA_CACHE[model_path] = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=2048,
            seed=seed,
            verbose=False,
        )
    return _LLAMA_CACHE[model_path]


def _gpu_memory_snapshot() -> JsonDict:  # pragma: no cover
    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10.0,
    )
    gpus: list[JsonDict] = []
    for line in str(result.get("stdout") or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(float(parts[2])),
                    "memory_free_mb": int(float(parts[3])),
                    "utilization_gpu_pct": int(float(parts[4])),
                }
            )
        except ValueError:
            continue
    return {"ok": bool(result.get("ok")), "stderr": result.get("stderr"), "gpus": gpus}


def _total_gpu_used(snapshot: Mapping[str, Any]) -> int:  # pragma: no cover
    rows = snapshot.get("gpus")
    if not isinstance(rows, list):
        return 0
    return sum(int(row.get("memory_used_mb") or 0) for row in rows if isinstance(row, Mapping))


def _llama_response_text(response: Any) -> str:  # pragma: no cover
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            return str(choices[0].get("text") or "")
    return str(response)


def _load_tests_run_argument(value: str | None) -> list[Any]:  # pragma: no cover
    if not value:
        return []
    path = Path(value)
    text = path.read_text(encoding="utf-8") if path.exists() else value
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("--tests-run-json must decode to a list")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--exp5365", type=Path, default=REPO_ROOT / exp5365.RESULT_RELATIVE_PATH)
    parser.add_argument("--tests-run-json", default=None)
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=args.out,
        exp5365_path=args.exp5365,
        tests_run=_load_tests_run_argument(args.tests_run_json),
        write=True,
    )
    print(
        f"[exp5366] status={artifact['status']} "
        f"clean={artifact['structured_protocol_clean']} out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
