#!/usr/bin/env python3
"""Exp 5338: local SOTA structured-output protocol calibration.

Spec refs: REQ-VERIFY-5338, SCENARIO-VERIFY-5338.

This module calibrates a formatting protocol only.  It asks the already-clean
Exp5337 local GGUF runtime for tiny deterministic JSON-shaped outputs, strips
llama.cpp console banners, extracts JSON only after an explicit final sentinel,
and records whether the final object is parseable with required keys.  It does
not score whether the model's facts or rewrites are good.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot import experiment_5325_theoria_rewrite_state_fixture_v486 as exp5325
from carnot import experiment_5337_sota_runtime_corrigendum_multimodel_v487 as exp5337


JsonDict = dict[str, Any]
GenerationProbe = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5338_structured_output_protocol_calibration_v487"
MILESTONE = "2026.07.487"
RESULT_RELATIVE_PATH = Path("results/experiment_5338_structured_output_protocol_calibration_v487.json")
SCHEMA = "carnot.experiment_5338.structured_output_protocol_calibration.v487"
INFERENCE_SUBSTRATE = "live_llm_inference"
SPEC_REFS = ("REQ-VERIFY-5338", "SCENARIO-VERIFY-5338")
RANDOM_SEED = 5338
TERMINAL_PREFIXES = ("complete:", "blocked_")
MISSING_WRAPPED_VALUE = object()

EXPECTED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
EXPECTED_ROLES = ("flagship_moe", "flagship_dense", "middle_moe")
EXPECTED_HF_BY_ROLE = dict(zip(EXPECTED_ROLES, EXPECTED_MODEL_IDS, strict=True))

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "Traceability for the Exp5338 structured-output protocol calibration receipt.",
    "milestone": "Milestone accountability for the V487 structured-output protocol gate.",
    "status": "Machine-readable terminal state for downstream structured-output gates.",
    "honest_verdict": (
        "Terminal verdict must start with complete: or blocked_ and state whether a "
        "parse-only structured-output protocol is ready."
    ),
    "inference_substrate": (
        "Declares live_llm_inference because Exp5338 calibrates the local SOTA GGUF "
        "generation protocol rather than replaying cached text."
    ),
    "MODEL_SPECS": (
        "Records the three mandated SOTA GGUF model IDs so calibration cannot silently "
        "substitute a legacy or smaller model."
    ),
    "preconditions_checked": (
        "Records Exp5337 runtime readiness, selected model/backend command, GPU "
        "visibility, and rewrite fixture availability before protocol calibration."
    ),
    "selected_model_spec": (
        "Binds protocol calibration outputs to the stable mandated model selected by Exp5337."
    ),
    "protocol_variants": (
        "Records each prompt, token-budget, sentinel, stop-sequence, extraction, and "
        "parser-stripping variant tested without scoring semantic quality."
    ),
    "tests_run": (
        "Commands run to validate the Exp5338 module, artifact schema, new-code "
        "coverage, repository tests, and applicable protocol checks."
    ),
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
    "parse_success_rate",
    "final_json_extraction_rate",
    "thinking_text_outside_final_rate",
    "unsafe_false_accepts",
    "structured_output_protocol_ready",
    "no_quality_claim",
)

DEFAULT_CALIBRATION_PROMPTS: tuple[JsonDict, ...] = (
    {
        "prompt_id": "protocol_fact_probe",
        "required_keys": ("id", "answer", "facts"),
        "prompt": (
            "Protocol calibration only. Return the Aster-9 battery duration from this "
            "sentence: The Aster-9 battery ran 47 minutes under the amber-load test."
        ),
    },
    {
        "prompt_id": "protocol_copy_probe",
        "required_keys": ("id", "answer", "facts"),
        "prompt": (
            "Protocol calibration only. Return code word orange for fixture B-12. "
            "Use id b12 and facts with code_word=orange."
        ),
    },
)

DEFAULT_PROTOCOL_VARIANTS: tuple[JsonDict, ...] = (
    {
        "variant_id": "final_sentinel_post_think_json_v1",
        "n_predict": 640,
        "sentinel": "FINAL_JSON:",
        "end_sentinel": "END_FINAL_JSON",
        "increased_token_budget": True,
        "explicit_final_only_sentinel": True,
        "post_think_json_extraction": True,
        "forbids_analysis_in_final": True,
        "parser_side_strips_llama_cpp_banners": True,
        "stop_sequences_requested": ["END_FINAL_JSON"],
        "stop_sequences_supported": False,
        "prompt_suffix": (
            "You may think first if the runtime emits a thinking transcript, but the final "
            "answer must be one compact JSON object only after FINAL_JSON:. Do not put "
            "analysis, scratch work, or thinking text inside any JSON value. End after "
            "END_FINAL_JSON."
        ),
    },
    {
        "variant_id": "final_only_no_analysis_wording_v1",
        "n_predict": 1024,
        "sentinel": "FINAL_JSON:",
        "end_sentinel": "END_FINAL_JSON",
        "increased_token_budget": True,
        "explicit_final_only_sentinel": True,
        "post_think_json_extraction": True,
        "forbids_analysis_in_final": True,
        "parser_side_strips_llama_cpp_banners": True,
        "stop_sequences_requested": ["END_FINAL_JSON"],
        "stop_sequences_supported": False,
        "prompt_suffix": (
            "Suppress analysis in the final object. The only machine-read part is the "
            "line beginning FINAL_JSON:. The object must contain every required key "
            "and no explanation fields."
        ),
    },
)

THINKING_MARKERS = ("[start thinking]", "<think>", "</think>", "thinking transcript", "scratch")


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> JsonDict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):  # pragma: no cover - defensive I/O path
        return {}


def _raw_or_wrapped_value(payload: Mapping[str, Any], field: str) -> Any:
    value = payload.get(field)
    if isinstance(value, Mapping) and "value" in value:
        return value.get("value")
    return value


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha16(text: str) -> str:
    return exp5337.exp5323.sha16(text)


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key) is True) / len(rows)


def strip_llama_cpp_banners(text: str) -> str:
    """Remove console chrome so JSON extraction sees model output, not runtime UI."""

    kept: list[str] = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if stripped.startswith(("Loading model", "available commands:", "/exit", "/regen")):
            continue
        if stripped.startswith(("/clear", "/read ", "/glob ", "build", "model", "modalities")):
            continue
        if stripped.startswith("> "):
            continue
        if stripped.startswith("[ Prompt:") and "Generation:" in stripped:
            continue
        if stripped == "Exiting...":
            continue
        if set(stripped) <= {"▄", "█", "▀", " ", "=", "-"}:
            continue
        kept.append(line)
    return "\n".join(kept)


def _json_object_from_segment(segment: str) -> tuple[JsonDict | None, str | None, int | None, int | None]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(segment):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(segment[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return dict(value), segment[index : index + end], index, index + end
    return None, None, None, None


def extract_final_json_payload(output_text: str, variant: Mapping[str, Any]) -> JsonDict:
    """Extract only the final sentinel JSON object from a model transcript."""

    cleaned = (
        strip_llama_cpp_banners(output_text)
        if variant.get("parser_side_strips_llama_cpp_banners")
        else str(output_text)
    )
    sentinel = str(variant.get("sentinel") or "")
    requires_sentinel = bool(variant.get("explicit_final_only_sentinel"))
    marker_index = cleaned.rfind(sentinel) if sentinel else -1
    if requires_sentinel and marker_index < 0:
        return {
            "payload": None,
            "object_text": None,
            "cleaned_text": cleaned,
            "final_marker_found": False,
            "json_start": None,
            "json_end": None,
        }
    search_start = marker_index + len(sentinel) if marker_index >= 0 else 0
    end_sentinel = str(variant.get("end_sentinel") or "")
    search_end = len(cleaned)
    if end_sentinel:
        end_index = cleaned.find(end_sentinel, search_start)
        if end_index >= 0:
            search_end = end_index
    segment = cleaned[search_start:search_end]
    payload, object_text, start, end = _json_object_from_segment(segment)
    return {
        "payload": payload,
        "object_text": object_text,
        "cleaned_text": cleaned,
        "final_marker_found": marker_index >= 0,
        "json_start": None if start is None else search_start + start,
        "json_end": None if end is None else search_start + end,
    }


def _contains_thinking_marker(value: Any) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in THINKING_MARKERS)
    if isinstance(value, Mapping):
        return any(_contains_thinking_marker(nested) for nested in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_thinking_marker(item) for item in value)
    return False


def _thinking_text_outside_final(
    output_text: str,
    extraction: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
) -> bool:
    if payload is None:
        return False
    if _contains_thinking_marker(payload):
        return False
    cleaned = str(extraction.get("cleaned_text") or output_text)
    json_start = extraction.get("json_start")
    lowered = cleaned.lower()
    positions = [lowered.find(marker) for marker in THINKING_MARKERS if lowered.find(marker) >= 0]
    if not positions:
        return True
    if not isinstance(json_start, int):
        return False
    return max(positions) < json_start


def score_protocol_output(
    *,
    prompt_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    output_text: str,
    completed: bool,
) -> JsonDict:
    extraction = extract_final_json_payload(output_text, variant)
    payload = extraction["payload"]
    required_keys = tuple(str(key) for key in prompt_spec.get("required_keys", ()))
    parse_success = completed and isinstance(payload, Mapping)
    final_json_extraction_success = parse_success and bool(extraction.get("final_marker_found"))
    schema_keys_present = parse_success and all(key in payload for key in required_keys)
    thinking_text_outside_final = _thinking_text_outside_final(output_text, extraction, payload)
    accepted = bool(
        completed
        and parse_success
        and final_json_extraction_success
        and schema_keys_present
        and thinking_text_outside_final
    )
    return {
        "prompt_id": str(prompt_spec.get("prompt_id")),
        "variant_id": str(variant.get("variant_id")),
        "completed": bool(completed),
        "parse_success": bool(parse_success),
        "final_json_extraction_success": bool(final_json_extraction_success),
        "schema_keys_present": bool(schema_keys_present),
        "thinking_text_outside_final": bool(thinking_text_outside_final),
        "accepted_for_protocol": accepted,
        "unsafe_false_accept": False,
        "required_keys": list(required_keys),
        "parsed_keys": sorted(payload) if isinstance(payload, Mapping) else [],
        "parsed_object": dict(payload) if isinstance(payload, Mapping) else None,
    }


def unsafe_false_accept_count(
    variant: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
) -> int:
    """Count parser accepts on deliberately invalid final-output shapes."""

    if not prompts:
        return 0
    prompt = prompts[0]
    sentinel = str(variant.get("sentinel") or "FINAL_JSON:")
    adversarial = (
        (
            '{"id":"draft","answer":"wrong","facts":{}}\n'
            f"{sentinel} {{not json}}"
        ),
        f'{sentinel} {{"id":"x","answer":"47"}}',
        f'{sentinel} {{"id":"x","answer":"[Start thinking] leaked","facts":{{}}}}',
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
    required = ", ".join(str(key) for key in prompt_spec.get("required_keys", ()))
    return (
        f"{prompt_spec['prompt']}\n"
        f"Required final JSON keys: {required}.\n"
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
    rewritten = list(command)

    def set_flag(flag: str, value: str) -> None:
        if flag in rewritten and rewritten.index(flag) + 1 < len(rewritten):
            rewritten[rewritten.index(flag) + 1] = value
        else:
            rewritten.extend([flag, value])

    set_flag("-p", prompt)
    set_flag("-n", str(n_predict))
    set_flag("--seed", str(seed))
    if variant.get("stop_sequences_supported"):
        for stop in variant.get("stop_sequences_requested", ()):
            rewritten.extend(["--reverse-prompt", str(stop)])
    return rewritten


def default_generation_probe(
    *,
    prompt_spec: Mapping[str, Any],
    variant: Mapping[str, Any],
    command: Sequence[str],
    timeout_s: float,
    run_index: int,
) -> JsonDict:  # pragma: no cover - invokes the live local llama.cpp subprocess
    _ = prompt_spec, variant, run_index
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
            "completed": result.returncode == 0,
            "timed_out": False,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "wall_clock_s": time.perf_counter() - started,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "completed": False,
            "timed_out": True,
            "returncode": None,
            "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "timeout") if isinstance(exc.stderr, str) else "timeout",
            "wall_clock_s": time.perf_counter() - started,
        }


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
    ready = bool(scores) and all(row.get("accepted_for_protocol") is True for row in scores) and unsafe == 0
    return {
        "variant_id": str(variant["variant_id"]),
        "n_predict": int(variant["n_predict"]),
        "sentinel": str(variant["sentinel"]),
        "end_sentinel": str(variant["end_sentinel"]),
        "increased_token_budget": bool(variant["increased_token_budget"]),
        "explicit_final_only_sentinel": bool(variant["explicit_final_only_sentinel"]),
        "post_think_json_extraction": bool(variant["post_think_json_extraction"]),
        "forbids_analysis_in_final": bool(variant["forbids_analysis_in_final"]),
        "parser_side_strips_llama_cpp_banners": bool(
            variant["parser_side_strips_llama_cpp_banners"]
        ),
        "stop_sequences_requested": list(variant.get("stop_sequences_requested", ())),
        "stop_sequences_supported": bool(variant.get("stop_sequences_supported")),
        "prompt_count": len(scores),
        "parse_success_rate": _rate(scores, "parse_success"),
        "final_json_extraction_rate": _rate(scores, "final_json_extraction_success"),
        "schema_key_presence_rate": _rate(scores, "schema_keys_present"),
        "thinking_text_outside_final_rate": _rate(scores, "thinking_text_outside_final"),
        "unsafe_false_accepts": unsafe,
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
                float(row.get("final_json_extraction_rate") or 0.0),
                float(row.get("parse_success_rate") or 0.0),
                float(row.get("schema_key_presence_rate") or 0.0),
            ),
        )
    )


def _selected_runtime_context(
    prior: Mapping[str, Any],
) -> tuple[JsonDict | None, JsonDict, JsonDict | None, list[str]]:
    blockers: list[str] = []
    status = _raw_or_wrapped_value(prior, "status")
    substrate = _raw_or_wrapped_value(prior, "inference_substrate")
    runtime_ready = prior.get("sota_runtime_clean_receipt_ready") is True
    if status != "complete" or substrate != "live_llm_inference" or not runtime_ready:
        blockers.append("exp5337_runtime_receipt_not_clean")

    selected_command = _raw_or_wrapped_value(prior, "selected_backend_command")
    if not isinstance(selected_command, Mapping) or not isinstance(selected_command.get("command"), list):
        blockers.append("selected_backend_command_missing")
        selected_command = None
    else:
        selected_command = dict(selected_command)

    model_specs = _raw_or_wrapped_value(prior, "MODEL_SPECS")
    if not isinstance(model_specs, Mapping):
        blockers.append("model_specs_missing_or_drift")
        model_specs = {}
    else:
        model_specs = {str(role): dict(spec) for role, spec in model_specs.items()}
        if set(model_specs) != set(EXPECTED_ROLES):
            blockers.append("model_specs_missing_or_drift")
        for role, hf_id in EXPECTED_HF_BY_ROLE.items():
            if role not in model_specs or model_specs[role].get("hf_id") != hf_id:
                blockers.append("model_specs_missing_or_drift")
                break

    selected_model = None
    selected_role = str((selected_command or {}).get("model_role") or "")
    if selected_role and selected_role in model_specs:
        selected_model = dict(model_specs[selected_role])
    else:
        blockers.append("selected_model_spec_missing")

    preconditions = _raw_or_wrapped_value(prior, "preconditions_checked")
    gpu_visible = isinstance(preconditions, Mapping) and preconditions.get("gpu_visible") is True
    if not gpu_visible:
        blockers.append("gpu_not_visible")

    command_list = list((selected_command or {}).get("command") or [])
    if command_list and not Path(str(command_list[0])).is_file():
        blockers.append("selected_binary_missing")
    model_path = str((selected_model or {}).get("model_path") or "")
    if not model_path or not Path(model_path).is_file():
        blockers.append("selected_model_file_missing")

    return selected_command, model_specs, selected_model, list(dict.fromkeys(blockers))


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


def _rewrite_fixture_ready(path: Path) -> tuple[bool, JsonDict]:
    payload = _read_json(path)
    ready = payload.get("rewrite_state_fixture_ready") is True
    return ready, {
        "rewrite_fixture_artifact_path": str(path),
        "rewrite_state_fixture_ready": ready,
        "rewrite_fixture_status": _raw_or_wrapped_value(payload, "status"),
        "rewrite_fixture_path": _raw_or_wrapped_value(payload, "fixture_path"),
    }


def _blocked_variant_rows(
    variants: Sequence[Mapping[str, Any]],
    blockers: Sequence[str],
) -> list[JsonDict]:
    return [
        {
            "variant_id": str(variant["variant_id"]),
            "n_predict": int(variant["n_predict"]),
            "sentinel": str(variant["sentinel"]),
            "end_sentinel": str(variant["end_sentinel"]),
            "status": "skipped_preconditions",
            "ready": False,
            "blocked_reason": ",".join(blockers),
            "prompt_count": 0,
            "parse_success_rate": 0.0,
            "final_json_extraction_rate": 0.0,
            "schema_key_presence_rate": 0.0,
            "thinking_text_outside_final_rate": 0.0,
            "unsafe_false_accepts": unsafe_false_accept_count(variant, DEFAULT_CALIBRATION_PROMPTS),
            "prompt_results": [],
            "generation_receipts": [],
        }
        for variant in variants
    ]


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    prior_runtime_path: Path | None = None,
    rewrite_fixture_artifact_path: Path | None = None,
    protocol_variants: Sequence[Mapping[str, Any]] = DEFAULT_PROTOCOL_VARIANTS,
    calibration_prompts: Sequence[Mapping[str, Any]] = DEFAULT_CALIBRATION_PROMPTS,
    generation_probe: GenerationProbe = default_generation_probe,
    tests_run: Sequence[Any] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    artifact_path = artifact_path or root / RESULT_RELATIVE_PATH
    prior_runtime_path = prior_runtime_path or root / exp5337.RESULT_RELATIVE_PATH
    rewrite_fixture_artifact_path = (
        rewrite_fixture_artifact_path or root / exp5325.RESULT_RELATIVE_PATH
    )
    prior = _read_json(prior_runtime_path)
    selected_command, model_specs, selected_model, blockers = _selected_runtime_context(prior)
    if not model_specs:
        model_specs = _default_model_specs()
    rewrite_ready, rewrite_status = _rewrite_fixture_ready(rewrite_fixture_artifact_path)
    if not rewrite_ready:
        blockers.append("rewrite_state_fixture_unavailable")
    blockers = list(dict.fromkeys(blockers))
    preconditions: JsonDict = {
        "exp5337_runtime_artifact_path": str(prior_runtime_path),
        "exp5337_runtime_receipt_clean": not any(
            blocker == "exp5337_runtime_receipt_not_clean" for blocker in blockers
        ),
        "selected_backend_command_present": selected_command is not None,
        "selected_model_role": (selected_command or {}).get("model_role"),
        "selected_model_file_present": bool(
            selected_model
            and selected_model.get("model_path")
            and Path(str(selected_model["model_path"])).is_file()
        ),
        "gpu_visible": "gpu_not_visible" not in blockers,
        **rewrite_status,
        "blocked_preconditions": blockers,
    }

    variant_rows: list[JsonDict] = []
    if blockers or selected_command is None:
        variant_rows = _blocked_variant_rows(protocol_variants, blockers)
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
    ready = bool(best_variant and best_variant.get("ready") is True and unsafe_false_accepts == 0)
    status = "complete" if ready else "blocked"
    honest = (
        f"complete: structured_output_protocol_ready={best_variant['variant_id']}"
        if ready and best_variant
        else "blocked_structured_output_protocol_ready_false"
    )
    top_parse = float((best_variant or {}).get("parse_success_rate") or 0.0)
    top_final = float((best_variant or {}).get("final_json_extraction_rate") or 0.0)
    top_thinking = float((best_variant or {}).get("thinking_text_outside_final_rate") or 0.0)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", honest),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "MODEL_SPECS": _wrap("MODEL_SPECS", dict(model_specs)),
        "preconditions_checked": _wrap("preconditions_checked", preconditions),
        "selected_model_spec": _wrap("selected_model_spec", selected_model),
        "protocol_variants": _wrap("protocol_variants", variant_rows),
        "parse_success_rate": top_parse,
        "final_json_extraction_rate": top_final,
        "thinking_text_outside_final_rate": top_thinking,
        "unsafe_false_accepts": unsafe_false_accepts,
        "structured_output_protocol_ready": ready,
        "no_quality_claim": True,
        "selected_variant_id": (best_variant or {}).get("variant_id"),
        "calibration_prompt_count": len(calibration_prompts),
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
                "ready": ready,
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

    for field in (
        "parse_success_rate",
        "final_json_extraction_rate",
        "thinking_text_outside_final_rate",
    ):
        if not _rate_is_valid(artifact.get(field)):
            errors.append(f"{field} must be in [0, 1]")
    if not isinstance(artifact.get("unsafe_false_accepts"), int):
        errors.append("unsafe_false_accepts must be a bare integer")
    if not isinstance(artifact.get("structured_output_protocol_ready"), bool):
        errors.append("structured_output_protocol_ready must be a bare boolean")
    if artifact.get("no_quality_claim") is not True:
        errors.append("no_quality_claim must be bare true")

    model_specs = _wrapped_value(artifact, "MODEL_SPECS")
    if not isinstance(model_specs, Mapping):
        errors.append("MODEL_SPECS must be an object")
    else:
        if set(model_specs) != set(EXPECTED_ROLES):
            errors.append("MODEL_SPECS roles mismatch")
        for role, hf_id in EXPECTED_HF_BY_ROLE.items():
            if role in model_specs and model_specs[role].get("hf_id") != hf_id:
                errors.append("hf_id mismatch for mandated model role")

    tests_run = _wrapped_value(artifact, "tests_run")
    if tests_run is not MISSING_WRAPPED_VALUE and not isinstance(tests_run, list):
        errors.append("tests_run must be a list")
    selected_model = _wrapped_value(artifact, "selected_model_spec")
    protocol_variants = _wrapped_value(artifact, "protocol_variants")
    if selected_model is not MISSING_WRAPPED_VALUE and selected_model is not None:
        if not isinstance(selected_model, Mapping):
            errors.append("selected_model_spec must be an object or null")
    if protocol_variants is not MISSING_WRAPPED_VALUE and not isinstance(protocol_variants, list):
        errors.append("protocol_variants must be a list")

    ready = artifact.get("structured_output_protocol_ready")
    unsafe = artifact.get("unsafe_false_accepts")
    if ready is True:
        if _wrapped_value(artifact, "status") != "complete":
            errors.append("ready artifact must have complete status")
        if unsafe != 0:
            errors.append("unsafe_false_accepts must be zero when protocol is ready")
        if isinstance(protocol_variants, list) and not any(row.get("ready") is True for row in protocol_variants):
            errors.append("ready artifact must include a ready protocol variant")
    elif ready is False:
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
        "--rewrite-fixture-artifact",
        type=Path,
        default=REPO_ROOT / exp5325.RESULT_RELATIVE_PATH,
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=len(DEFAULT_PROTOCOL_VARIANTS),
        help="Limit live calibration variants for bounded local runs.",
    )
    parser.add_argument(
        "--tests-run-json",
        default="[]",
        help="JSON list of validation commands to embed in the artifact.",
    )
    args = parser.parse_args(argv)
    artifact = run(
        artifact_path=args.out,
        prior_runtime_path=args.prior_runtime,
        rewrite_fixture_artifact_path=args.rewrite_fixture_artifact,
        protocol_variants=DEFAULT_PROTOCOL_VARIANTS[: max(1, args.max_variants)],
        tests_run=json.loads(args.tests_run_json),
        write=True,
    )
    print(
        f"[exp5338] status={artifact['status']['value']} "
        f"ready={artifact['structured_output_protocol_ready']} out={args.out}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
