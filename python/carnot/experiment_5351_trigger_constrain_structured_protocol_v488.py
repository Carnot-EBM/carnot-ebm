#!/usr/bin/env python3
"""Exp 5351: trigger-then-constrain structured protocol calibration.

Spec refs: REQ-VERIFY-5351, SCENARIO-VERIFY-5351.

This module calibrates a formatting protocol only. It lets a mandated local
GGUF model produce any reasoning text it needs, then accepts only the final JSON
object after a sentinel and validates that object against a strict prompt
schema. The artifact records parser/schema behavior and runtime methodology,
not answer quality.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot import experiment_5337_sota_runtime_corrigendum_multimodel_v487 as exp5337
from carnot import experiment_5338_structured_output_protocol_calibration_v487 as exp5338


JsonDict = dict[str, Any]
GenerationProbe = Callable[..., JsonDict]
PreconditionsProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5351_trigger_constrain_structured_protocol_v488"
MILESTONE = "2026.07.488"
RESULT_RELATIVE_PATH = Path("results/experiment_5351_trigger_constrain_structured_protocol_v488.json")
SCHEMA = "carnot.experiment_5351.trigger_constrain_structured_protocol.v488"
INFERENCE_SUBSTRATE = "live_llm_inference"
SPEC_REFS = ("REQ-VERIFY-5351", "SCENARIO-VERIFY-5351")
RANDOM_SEED = 5351
MIN_CLEAN_METHODOLOGY_DURATION_S = 60.0
FREE_REASONING_TRIGGER = "FREE_REASONING_TRIGGER"
TERMINAL_PREFIXES = ("complete:", "blocked_")
MISSING_WRAPPED_VALUE = object()

EXPECTED_MODEL_IDS = exp5338.EXPECTED_MODEL_IDS
EXPECTED_ROLES = exp5338.EXPECTED_ROLES
EXPECTED_HF_BY_ROLE = exp5338.EXPECTED_HF_BY_ROLE

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Stable id ties the artifact to this roadmap task.",
    "milestone": (
        "Prevents `.487` flagged protocol evidence from being reused as `.488` evidence."
    ),
    "status": "Lets gates distinguish completed calibration from blocked runtime.",
    "honest_verdict": (
        "Terminal prefix `complete:` or `blocked_` prevents ambiguous downstream "
        "quality gates."
    ),
    "inference_substrate": (
        "Expected value is live_llm_inference when a GGUF model generated outputs."
    ),
    "MODEL_SPECS": (
        "Confirms every LLM task includes the mandated local SOTA GGUF model ids."
    ),
    "preconditions_checked": "Records GPU/backend/model checks before inference.",
    "selected_model_spec": "Identifies which mandated model actually supplied the receipt.",
    "protocol_variants": "Shows which trigger/constrain mechanisms were tested.",
    "tests_run": "Lists local checks that validated parser/schema behavior.",
}

REQUIRED_WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "MODEL_SPECS",
    "preconditions_checked",
    "selected_model_spec",
    "protocol_variants",
    "tests_run",
)
REQUIRED_ARTIFACT_FIELDS = (
    *REQUIRED_WRAPPED_FIELDS,
    "prompt_count",
    "parse_success_rate",
    "schema_success_rate",
    "final_json_extraction_rate",
    "unsafe_false_accepts",
    "methodology_duration_s",
    "structured_protocol_clean",
    "no_quality_claim",
    "no_autotokenizer_used",
)

DEFAULT_SCHEMA: JsonDict = {
    "type": "object",
    "required": ["id", "answer", "facts"],
    "properties": {
        "id": {"type": "string"},
        "answer": {"type": "string"},
        "facts": {"type": "array"},
    },
}

DEFAULT_CALIBRATION_PROMPTS: tuple[JsonDict, ...] = (
    {
        "prompt_id": "battery_duration_probe",
        "prompt": (
            "Protocol calibration only. Use id battery_probe. The Aster-9 battery "
            "ran 47 minutes under the amber-load test. Return answer 47 minutes."
        ),
        "expected": {"id": "battery_probe", "answer": "47 minutes"},
        "target_final_object": {
            "id": "battery_probe",
            "answer": "47 minutes",
            "facts": ["The Aster-9 battery ran 47 minutes under the amber-load test."],
        },
        "schema": DEFAULT_SCHEMA,
    },
    {
        "prompt_id": "code_word_probe",
        "prompt": (
            "Protocol calibration only. Use id code_probe. Fixture B-12 has code "
            "word orange. Return answer orange."
        ),
        "expected": {"id": "code_probe", "answer": "orange"},
        "target_final_object": {
            "id": "code_probe",
            "answer": "orange",
            "facts": ["Fixture B-12 has code word orange."],
        },
        "schema": DEFAULT_SCHEMA,
    },
    {
        "prompt_id": "route_probe",
        "prompt": (
            "Protocol calibration only. Use id route_probe. In route card R-3, the "
            "approved direction is north. Return answer north."
        ),
        "expected": {"id": "route_probe", "answer": "north"},
        "target_final_object": {
            "id": "route_probe",
            "answer": "north",
            "facts": ["In route card R-3, the approved direction is north."],
        },
        "schema": DEFAULT_SCHEMA,
    },
    {
        "prompt_id": "count_probe",
        "prompt": (
            "Protocol calibration only. Use id count_probe. The fixture contains "
            "three validated rows. Return answer 3."
        ),
        "expected": {"id": "count_probe", "answer": "3"},
        "target_final_object": {
            "id": "count_probe",
            "answer": "3",
            "facts": ["The fixture contains three validated rows."],
        },
        "schema": DEFAULT_SCHEMA,
    },
)

DEFAULT_PROTOCOL_VARIANTS: tuple[JsonDict, ...] = (
    {
        "variant_id": "trigger_then_final_json_sentinel_v1",
        "n_predict": 1024,
        "sentinel": "FINAL_JSON:",
        "end_sentinel": "END_FINAL_JSON",
        "free_reasoning_trigger_token": FREE_REASONING_TRIGGER,
        "final_json_sentinel": "FINAL_JSON:",
        "final_only_extraction": True,
        "explicit_final_only_sentinel": True,
        "strict_schema_validation": True,
        "parser_side_strips_llama_cpp_banners": True,
        "stop_sequences_requested": ["END_FINAL_JSON"],
        "stop_sequences_supported": False,
        "prompt_suffix": (
            f"Think freely after {FREE_REASONING_TRIGGER}. When ready, write exactly one "
            "final JSON object after FINAL_JSON:. Copy the target final object exactly. "
            "The answer value must be a quoted string and facts must be an array of "
            "strings. Do not put analysis or thinking text inside JSON values. End "
            "with END_FINAL_JSON."
        ),
    },
)

THINKING_MARKERS = exp5338.THINKING_MARKERS


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):  # pragma: no cover - defensive I/O
        return {}


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha16(text: str) -> str:
    return exp5337.exp5323.sha16(text)


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key) is True) / len(rows)


def strip_llama_cpp_banners(text: str) -> str:
    return exp5338.strip_llama_cpp_banners(text)


def _type_matches(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return False


def schema_errors(payload: Any, schema: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_type = str(schema.get("type") or "object")
    if not _type_matches(payload, expected_type):
        return [f"payload must be {expected_type}"]
    if not isinstance(payload, Mapping):
        return errors
    required = [str(key) for key in schema.get("required", ())]
    missing = [key for key in required if key not in payload]
    if missing:
        errors.append(f"missing required keys: {missing}")
    properties = schema.get("properties", {})
    if isinstance(properties, Mapping):
        for key, subschema in properties.items():
            if key in payload and isinstance(subschema, Mapping):
                subtype = str(subschema.get("type") or "")
                if subtype and not _type_matches(payload[key], subtype):
                    errors.append(f"{key} must be {subtype}")
    return errors


def _thinking_text_outside_final(
    output_text: str,
    extraction: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
) -> bool:
    return exp5338._thinking_text_outside_final(output_text, extraction, payload)


def score_protocol_output(
    *,
    prompt_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    output_text: str,
    completed: bool,
) -> JsonDict:
    extraction = exp5338.extract_final_json_payload(output_text, variant)
    payload = extraction["payload"]
    parse_success = completed and isinstance(payload, Mapping)
    final_json_extraction_success = parse_success and bool(extraction.get("final_marker_found"))
    schema_error_rows = schema_errors(payload, prompt_spec.get("schema", DEFAULT_SCHEMA))
    schema_success = parse_success and not schema_error_rows
    thinking_text_outside_final = _thinking_text_outside_final(output_text, extraction, payload)
    accepted = bool(
        completed
        and parse_success
        and final_json_extraction_success
        and schema_success
        and thinking_text_outside_final
    )
    return {
        "prompt_id": str(prompt_spec.get("prompt_id")),
        "variant_id": str(variant.get("variant_id")),
        "completed": bool(completed),
        "parse_success": bool(parse_success),
        "schema_success": bool(schema_success),
        "final_json_extraction_success": bool(final_json_extraction_success),
        "thinking_text_outside_final": bool(thinking_text_outside_final),
        "accepted_for_protocol": accepted,
        "unsafe_false_accept": False,
        "schema_errors": schema_error_rows,
        "parsed_keys": sorted(payload) if isinstance(payload, Mapping) else [],
        "parsed_object": dict(payload) if isinstance(payload, Mapping) else None,
    }


def unsafe_false_accept_count(
    variant: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
) -> int:
    if not prompts:
        return 0
    prompt = prompts[0]
    sentinel = str(variant.get("sentinel") or "FINAL_JSON:")
    adversarial = (
        '{"id":"draft","answer":"wrong","facts":[]}',
        f'{{"id":"draft","answer":"wrong","facts":[]}}\n{sentinel} {{not json}}',
        f'{sentinel} {{"id":"battery_probe","answer":"47 minutes"}}',
        f'{sentinel} {{"id":"battery_probe","answer":47,"facts":[]}}',
        f'{sentinel} {{"id":"battery_probe","answer":"[Start thinking] leaked","facts":[]}}',
    )
    return sum(
        1
        for text in adversarial
        if score_protocol_output(
            prompt_spec=prompt,
            variant=variant,
            output_text=text,
            completed=True,
        )["accepted_for_protocol"]
    )


def build_protocol_prompt(prompt_spec: Mapping[str, Any], variant: Mapping[str, Any]) -> str:
    target = prompt_spec.get("target_final_object")
    target_text = _stable_json(target) if isinstance(target, Mapping) else "{}"
    return (
        f"{prompt_spec['prompt']}\n"
        f"{variant['free_reasoning_trigger_token']}: reason if needed before the final object.\n"
        "Strict schema: id is a string, answer is a string, facts is an array of strings.\n"
        f"Target final object to copy exactly: {target_text}\n"
        f"{variant['prompt_suffix']}\n"
        f"Final format exactly: {variant['sentinel']} {{...}} {variant['end_sentinel']}"
    )


def command_for_protocol(
    command: Sequence[str],
    prompt: str,
    *,
    n_predict: int,
    seed: int,
    variant: Mapping[str, Any],
) -> list[str]:
    return exp5338.command_for_protocol(
        command,
        prompt,
        n_predict=n_predict,
        seed=seed,
        variant=variant,
    )


def default_generation_probe(
    *,
    prompt_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    command: Sequence[str],
    timeout_s: float,
    run_index: int,
) -> JsonDict:  # pragma: no cover - invokes live local llama.cpp
    return exp5338.default_generation_probe(
        prompt_spec=prompt_spec,
        variant=variant,
        command=command,
        timeout_s=timeout_s,
        run_index=run_index,
    )


def _estimate_generated_tokens(text: str) -> int:
    cleaned = strip_llama_cpp_banners(text)
    return len([chunk for chunk in cleaned.replace("{", " ").replace("}", " ").split() if chunk])


def _normalise_generation_receipt(
    raw: Mapping[str, Any],
    *,
    prompt_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    command: Sequence[str],
    timeout_s: float,
    run_index: int,
) -> JsonDict:
    stdout = str(raw.get("stdout") or "")
    stderr = str(raw.get("stderr") or "")
    completed = bool(raw.get("completed")) and raw.get("returncode") == 0 and not raw.get("timed_out")
    score = score_protocol_output(
        prompt_spec=prompt_spec,
        variant=variant,
        output_text=stdout,
        completed=completed,
    )
    return {
        "run_index": run_index,
        "prompt_id": str(prompt_spec["prompt_id"]),
        "variant_id": str(variant["variant_id"]),
        "command": list(command),
        "completed": completed,
        "timed_out": bool(raw.get("timed_out")),
        "returncode": raw.get("returncode"),
        "timeout_s": timeout_s,
        "wall_clock_s": float(raw.get("wall_clock_s") or 0.0),
        "generated_token_count": int(raw.get("generated_token_count") or _estimate_generated_tokens(stdout)),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-1000:],
        "output_checksum": _sha16(stdout),
        "score": score,
    }


def _variant_summary(
    variant: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    scores = [row["score"] for row in receipts]
    unsafe = unsafe_false_accept_count(variant, prompts)
    methodology_duration_s = round(sum(float(row.get("wall_clock_s") or 0.0) for row in receipts), 6)
    ready = bool(
        scores
        and len(scores) == len(prompts)
        and all(row.get("accepted_for_protocol") is True for row in scores)
        and unsafe == 0
        and methodology_duration_s >= MIN_CLEAN_METHODOLOGY_DURATION_S
    )
    return {
        "variant_id": str(variant["variant_id"]),
        "n_predict": int(variant["n_predict"]),
        "free_reasoning_trigger_token": str(variant["free_reasoning_trigger_token"]),
        "final_json_sentinel": str(variant["final_json_sentinel"]),
        "end_sentinel": str(variant["end_sentinel"]),
        "final_only_extraction": bool(variant["final_only_extraction"]),
        "strict_schema_validation": bool(variant["strict_schema_validation"]),
        "parser_side_strips_llama_cpp_banners": bool(
            variant["parser_side_strips_llama_cpp_banners"]
        ),
        "stop_sequences_requested": list(variant.get("stop_sequences_requested", ())),
        "stop_sequences_supported": bool(variant.get("stop_sequences_supported")),
        "prompt_count": len(scores),
        "parse_success_rate": _rate(scores, "parse_success"),
        "schema_success_rate": _rate(scores, "schema_success"),
        "final_json_extraction_rate": _rate(scores, "final_json_extraction_success"),
        "unsafe_false_accepts": unsafe,
        "methodology_duration_s": methodology_duration_s,
        "token_counts": {
            "generated_token_count": sum(int(row.get("generated_token_count") or 0) for row in receipts),
            "n_predict_per_prompt": int(variant["n_predict"]),
        },
        "command_lines": [list(row.get("command") or []) for row in receipts],
        "ready": ready,
        "prompt_results": list(scores),
        "generation_receipts": list(receipts),
    }


def _select_best_variant(variant_rows: Sequence[Mapping[str, Any]]) -> JsonDict | None:
    if not variant_rows:
        return None
    return dict(
        max(
            variant_rows,
            key=lambda row: (
                bool(row.get("ready")),
                float(row.get("schema_success_rate") or 0.0),
                float(row.get("final_json_extraction_rate") or 0.0),
                float(row.get("parse_success_rate") or 0.0),
            ),
        )
    )


def _default_model_specs() -> JsonDict:
    return {
        role: {
            "role": role,
            "hf_id": EXPECTED_HF_BY_ROLE[role],
            "model_path": None,
            "status": "unavailable_from_exp5337_preconditions",
            "autotokenizer_used": False,
        }
        for role in EXPECTED_ROLES
    }


def _selected_runtime_context(
    prior: Mapping[str, Any],
) -> tuple[JsonDict | None, JsonDict, JsonDict | None, list[str]]:
    selected_command, model_specs, selected_model, blockers = exp5338._selected_runtime_context(prior)
    if any(bool(spec.get("autotokenizer_used")) for spec in model_specs.values()):
        blockers.append("autotokenizer_used_for_gguf")
    return selected_command, model_specs, selected_model, list(dict.fromkeys(blockers))


def _run_command(command: Sequence[str], timeout_s: float = 5.0) -> JsonDict:  # pragma: no cover
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
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": list(command),
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "duration_s": round(time.perf_counter() - started, 6),
        }


def _parse_free_vram_mb(query_stdout: str) -> int:  # pragma: no cover
    total = 0
    for line in query_stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            try:
                total += int(parts[4])
            except ValueError:
                pass
    return total


def default_preconditions_probe(
    *,
    selected_command: Mapping[str, Any] | None,
    model_specs: Mapping[str, Any],
    selected_model: Mapping[str, Any] | None,
    prior_runtime_path: Path,
) -> JsonDict:  # pragma: no cover - probes host GPU and native llama.cpp
    command = list((selected_command or {}).get("command") or [])
    binary = command[0] if command else ""
    raw_nvidia_smi = _run_command(["nvidia-smi"], timeout_s=8.0)
    nvidia_smi = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=8.0,
    )
    llama_cpp_version = _run_command([binary, "--version"], timeout_s=8.0) if binary else {}
    return {
        "exp5337_runtime_artifact_path": str(prior_runtime_path),
        "exp5337_runtime_receipt_clean": True,
        "gpu_visible": bool(nvidia_smi.get("ok") and str(nvidia_smi.get("stdout") or "").strip()),
        "raw_nvidia_smi": raw_nvidia_smi,
        "nvidia_smi": nvidia_smi,
        "free_vram_mb": _parse_free_vram_mb(str(nvidia_smi.get("stdout") or "")),
        "llama_cpp_command": [binary] if binary else [],
        "llama_cpp_version": llama_cpp_version,
        "model_file_presence": {
            role: bool(spec.get("model_path") and Path(str(spec["model_path"])).is_file())
            for role, spec in model_specs.items()
        },
        "selected_model_file_present": bool(
            selected_model
            and selected_model.get("model_path")
            and Path(str(selected_model["model_path"])).is_file()
        ),
        "backend_cuda_evidence": "libggml-cuda" in json.dumps(selected_command, sort_keys=True),
        "blocked_preconditions": [],
    }


def _blocked_variant_rows(
    variants: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
    blockers: Sequence[str],
) -> list[JsonDict]:
    return [
        {
            "variant_id": str(variant["variant_id"]),
            "n_predict": int(variant["n_predict"]),
            "free_reasoning_trigger_token": str(variant["free_reasoning_trigger_token"]),
            "final_json_sentinel": str(variant["final_json_sentinel"]),
            "end_sentinel": str(variant["end_sentinel"]),
            "final_only_extraction": bool(variant["final_only_extraction"]),
            "strict_schema_validation": bool(variant["strict_schema_validation"]),
            "status": "skipped_preconditions",
            "ready": False,
            "blocked_reason": ",".join(blockers),
            "prompt_count": 0,
            "parse_success_rate": 0.0,
            "schema_success_rate": 0.0,
            "final_json_extraction_rate": 0.0,
            "unsafe_false_accepts": unsafe_false_accept_count(variant, prompts),
            "methodology_duration_s": 0.0,
            "token_counts": {"generated_token_count": 0, "n_predict_per_prompt": int(variant["n_predict"])},
            "command_lines": [],
            "prompt_results": [],
            "generation_receipts": [],
        }
        for variant in variants
    ]


def _prompt_count_blockers(prompts: Sequence[Mapping[str, Any]]) -> list[str]:
    return [] if 4 <= len(prompts) <= 6 else ["prompt_count_outside_4_to_6"]


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    prior_runtime_path: Path | None = None,
    protocol_variants: Sequence[Mapping[str, Any]] = DEFAULT_PROTOCOL_VARIANTS,
    calibration_prompts: Sequence[Mapping[str, Any]] = DEFAULT_CALIBRATION_PROMPTS,
    preconditions_probe: PreconditionsProbe = default_preconditions_probe,
    generation_probe: GenerationProbe = default_generation_probe,
    tests_run: Sequence[Any] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    artifact_path = artifact_path or root / RESULT_RELATIVE_PATH
    prior_runtime_path = prior_runtime_path or root / exp5337.RESULT_RELATIVE_PATH
    prior = _read_json(prior_runtime_path)
    selected_command, model_specs, selected_model, blockers = _selected_runtime_context(prior)
    if not model_specs:
        model_specs = _default_model_specs()

    prompt_blockers = _prompt_count_blockers(calibration_prompts)
    current_preconditions = dict(
        preconditions_probe(
            selected_command=selected_command,
            model_specs=model_specs,
            selected_model=selected_model,
            prior_runtime_path=prior_runtime_path,
        )
    )
    if not current_preconditions.get("gpu_visible"):
        blockers.append("current_gpu_not_visible")
    if current_preconditions.get("selected_model_file_present") is False:
        blockers.append("selected_model_file_missing")
    blockers.extend(str(item) for item in current_preconditions.get("blocked_preconditions", ()))
    blockers.extend(prompt_blockers)
    blockers = list(dict.fromkeys(blockers))
    current_preconditions["blocked_preconditions"] = blockers
    current_preconditions["no_autotokenizer_used"] = not any(
        bool(spec.get("autotokenizer_used")) for spec in model_specs.values()
    )

    variant_rows: list[JsonDict] = []
    if blockers or selected_command is None:
        variant_rows = _blocked_variant_rows(protocol_variants, calibration_prompts, blockers)
    else:
        base_command = selected_command["command"]
        timeout_s = float(selected_command.get("timeout_s") or 240.0)
        run_index = 0
        for variant in protocol_variants:
            receipts: list[JsonDict] = []
            for prompt_spec in calibration_prompts:
                run_index += 1
                prompt = build_protocol_prompt(prompt_spec, variant)
                command = command_for_protocol(
                    base_command,
                    prompt,
                    n_predict=int(variant["n_predict"]),
                    seed=RANDOM_SEED + run_index,
                    variant=variant,
                )
                raw = generation_probe(
                    prompt_spec=prompt_spec,
                    variant=variant,
                    command=command,
                    timeout_s=timeout_s,
                    run_index=run_index,
                )
                receipts.append(
                    _normalise_generation_receipt(
                        raw,
                        prompt_spec=prompt_spec,
                        variant=variant,
                        command=command,
                        timeout_s=timeout_s,
                        run_index=run_index,
                    )
                )
            variant_rows.append(_variant_summary(variant, receipts, calibration_prompts))

    best_variant = _select_best_variant(variant_rows)
    unsafe_false_accepts = sum(int(row.get("unsafe_false_accepts") or 0) for row in variant_rows)
    no_autotokenizer_used = bool(current_preconditions.get("no_autotokenizer_used"))
    top_parse = float((best_variant or {}).get("parse_success_rate") or 0.0)
    top_schema = float((best_variant or {}).get("schema_success_rate") or 0.0)
    top_final = float((best_variant or {}).get("final_json_extraction_rate") or 0.0)
    methodology_duration_s = float((best_variant or {}).get("methodology_duration_s") or 0.0)
    clean = bool(
        best_variant
        and best_variant.get("ready") is True
        and len(calibration_prompts) >= 4
        and len(calibration_prompts) <= 6
        and top_parse == 1.0
        and top_schema == 1.0
        and top_final == 1.0
        and unsafe_false_accepts == 0
        and methodology_duration_s >= MIN_CLEAN_METHODOLOGY_DURATION_S
        and no_autotokenizer_used
        and not blockers
    )
    status = "complete" if clean else "blocked"
    honest = (
        f"complete: structured_protocol_clean={best_variant['variant_id']}"
        if clean and best_variant
        else "blocked_structured_protocol_clean_false"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", honest),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "MODEL_SPECS": _wrap("MODEL_SPECS", dict(model_specs)),
        "preconditions_checked": _wrap("preconditions_checked", current_preconditions),
        "selected_model_spec": _wrap("selected_model_spec", selected_model),
        "protocol_variants": _wrap("protocol_variants", variant_rows),
        "prompt_count": len(calibration_prompts),
        "parse_success_rate": top_parse,
        "schema_success_rate": top_schema,
        "final_json_extraction_rate": top_final,
        "unsafe_false_accepts": unsafe_false_accepts,
        "methodology_duration_s": methodology_duration_s,
        "structured_protocol_clean": clean,
        "no_quality_claim": True,
        "no_autotokenizer_used": no_autotokenizer_used,
        "selected_variant_id": (best_variant or {}).get("variant_id"),
        "tests_run": _wrap("tests_run", list(tests_run or [])),
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.perf_counter() - started, 6),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _sha16(
        _stable_json(
            {
                "experiment_id": EXPERIMENT_ID,
                "selected_model": selected_model,
                "variant_rows": variant_rows,
                "clean": clean,
                "seed": RANDOM_SEED,
            }
        )
    )
    validate_artifact(artifact)
    if write:
        write_json(artifact_path, artifact)
    return artifact


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    value = artifact.get(field)
    if not isinstance(value, Mapping):
        return MISSING_WRAPPED_VALUE
    if value.get("principle") != FIELD_PRINCIPLES.get(field):
        return MISSING_WRAPPED_VALUE
    return value.get("value")


def _rate_is_valid(value: Any) -> bool:
    return isinstance(value, int | float) and 0.0 <= float(value) <= 1.0


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    for field in REQUIRED_WRAPPED_FIELDS:
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

    prompt_count = artifact.get("prompt_count")
    if not isinstance(prompt_count, int):
        errors.append("prompt_count must be a bare integer")
    for field in ("parse_success_rate", "schema_success_rate", "final_json_extraction_rate"):
        if not _rate_is_valid(artifact.get(field)):
            errors.append(f"{field} must be in [0, 1]")
    if not isinstance(artifact.get("unsafe_false_accepts"), int):
        errors.append("unsafe_false_accepts must be a bare integer")
    methodology_duration = artifact.get("methodology_duration_s")
    if not isinstance(methodology_duration, int | float):
        errors.append("methodology_duration_s must be numeric")
        methodology_duration = -1.0
    if not isinstance(artifact.get("structured_protocol_clean"), bool):
        errors.append("structured_protocol_clean must be a bare boolean")
    if artifact.get("no_quality_claim") is not True:
        errors.append("no_quality_claim must be bare true")
    if artifact.get("no_autotokenizer_used") is not True:
        errors.append("no_autotokenizer_used must be bare true")

    model_specs = _wrapped_value(artifact, "MODEL_SPECS")
    if not isinstance(model_specs, Mapping):
        errors.append("MODEL_SPECS must be an object")
    else:
        if set(model_specs) != set(EXPECTED_ROLES):
            errors.append("MODEL_SPECS roles mismatch")
        for role, hf_id in EXPECTED_HF_BY_ROLE.items():
            if role in model_specs and model_specs[role].get("hf_id") != hf_id:
                errors.append("hf_id mismatch for mandated model role")
            if role in model_specs and model_specs[role].get("autotokenizer_used") is not False:
                errors.append("autotokenizer_used must stay false")

    tests_run = _wrapped_value(artifact, "tests_run")
    selected_model = _wrapped_value(artifact, "selected_model_spec")
    protocol_variants = _wrapped_value(artifact, "protocol_variants")
    if tests_run is not MISSING_WRAPPED_VALUE and not isinstance(tests_run, list):
        errors.append("tests_run must be a list")
    if selected_model is not MISSING_WRAPPED_VALUE and selected_model is not None:
        if not isinstance(selected_model, Mapping):
            errors.append("selected_model_spec must be an object or null")
    if protocol_variants is not MISSING_WRAPPED_VALUE and not isinstance(protocol_variants, list):
        errors.append("protocol_variants must be a list")

    clean = artifact.get("structured_protocol_clean")
    unsafe = artifact.get("unsafe_false_accepts")
    if clean is True:
        if _wrapped_value(artifact, "status") != "complete":
            errors.append("clean artifact must have complete status")
        if not isinstance(prompt_count, int) or not 4 <= prompt_count <= 6:
            errors.append("prompt_count must be 4 to 6")
        if unsafe != 0:
            errors.append("unsafe_false_accepts must be zero when protocol is clean")
        if artifact.get("parse_success_rate") != 1.0:
            errors.append("clean artifact must have parse_success_rate 1.0")
        if artifact.get("schema_success_rate") != 1.0:
            errors.append("clean artifact must have schema_success_rate 1.0")
        if artifact.get("final_json_extraction_rate") != 1.0:
            errors.append("clean artifact must have final_json_extraction_rate 1.0")
        if isinstance(methodology_duration, int | float):
            if methodology_duration < MIN_CLEAN_METHODOLOGY_DURATION_S:
                errors.append("clean artifact cannot be ready below 60s")
        if isinstance(protocol_variants, list) and not any(row.get("ready") is True for row in protocol_variants):
            errors.append("clean artifact must include a ready protocol variant")
    elif clean is False:
        if _wrapped_value(artifact, "status") != "blocked":
            errors.append("blocked artifact must have blocked status")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise AssertionError("; ".join(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--prior-runtime", type=Path, default=REPO_ROOT / exp5337.RESULT_RELATIVE_PATH)
    parser.add_argument(
        "--tests-run-json",
        default="[]",
        help="JSON list of validation commands to embed in the artifact.",
    )
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=args.out,
        prior_runtime_path=args.prior_runtime,
        tests_run=json.loads(args.tests_run_json),
        write=True,
    )
    print(
        f"[exp5351] status={artifact['status']['value']} "
        f"clean={artifact['structured_protocol_clean']} out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
