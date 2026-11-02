"""Balanced live SOTA GGUF telemetry manifest for Exp 1480.

Exp 1468 proved that local mandated GGUF models can expose token text,
completion logprobs, top-k alternatives, and llama.cpp logits.  Exp 1473 then
showed that those signals are not publishable by themselves because length,
format, prompt family, and mechanical gates can explain the apparent effect.

This module builds the next artifact in that chain: a balanced manifest whose
labels and superficial baselines are recorded on every row before any downstream
diagnostic is allowed to claim a signal.  The manifest is intentionally local
and conservative.  It uses only the mandated GGUF registry/cache path for
headline rows and defaults `claim_allowed` to false unless live SOTA rows,
balanced labels, top-k/logit telemetry, and row-level confound baselines all
exist together.

Spec: REQ-INFER-SOTA-010,
      SCENARIO-INFER-SOTA-010-001,
      SCENARIO-INFER-SOTA-010-002
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import cached_sota_pair
from carnot.reporting.live_sota_repair_runtime_preflight import probe_gpu_state


DEFAULT_RUN_DATE = "20260507"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1480_live_sota_balanced_telemetry_v2.json")
DEFAULT_MANIFEST_PATH = Path("results/live_sota_balanced_telemetry_manifest_1480.jsonl")

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_SPECS: tuple[dict[str, str], ...] = (
    {"hf_id": MANDATED_MODEL_IDS[0], "role": "flagship_moe_telemetry_model"},
    {"hf_id": MANDATED_MODEL_IDS[1], "role": "flagship_dense_telemetry_model"},
    {"hf_id": MANDATED_MODEL_IDS[2], "role": "middle_moe_telemetry_model"},
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "telemetry_cases_requested",
    "telemetry_cases_completed",
    "balanced_label_counts",
    "topk_logprobs_available",
    "logits_available",
    "superficial_baselines_recorded",
    "telemetry_manifest_path",
    "models_used",
    "gpu_probe",
    "blockers",
    "claim_allowed",
    "honest_verdict",
)
SUPERFICIAL_BASELINE_FIELDS: tuple[str, ...] = (
    "response_length",
    "token_count",
    "json_valid",
    "schema_valid",
    "prompt_family",
    "answer_lexical_overlap",
    "model_family",
)

LOGPROBS_REQUESTED = 5
TOPK_MIN_ALTERNATIVES = 2
TOPK_MIN_CASE_COVERAGE = 0.8
MAX_TOKENS = 24

JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[JsonDict] | None]
GenerationFn = Callable[[JsonDict, "BalancedTelemetryCase"], "RawBalancedGeneration"]
LlamaImporter = Callable[[], tuple[bool, type[Any] | None, str | None]]
GpuProbeFn = Callable[[], JsonDict]
WriteJsonFn = Callable[[Path, JsonDict], None]


@dataclass(frozen=True)
class BalancedTelemetryCase:
    """One bounded prompt with an expected adversarial label bucket.

    The `intended_*` fields are not the scored labels; they document how the
    fixed prompt set was constructed.  The manifest still recomputes correctness
    and format validity from the actual model response so a model that refuses a
    forced wrong answer cannot accidentally be counted as balanced.
    """

    case_id: str
    family: str
    prompt: str
    expected_answer: str
    adversarial_wrong_answer: str
    intended_correct: bool
    intended_format_valid: bool


@dataclass(frozen=True)
class RawBalancedGeneration:
    """Raw runtime output before normalization into a manifest row.

    llama.cpp returns OpenAI-compatible completion dictionaries on most hosts,
    but tests and fallback probes may inject simpler objects.  Keeping the raw
    result separate from the normalized row makes missing telemetry explicit
    instead of silently treating response text as a full token trace.
    """

    response_text: str
    raw_response: JsonDict | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
    logprob_error: str | None = None
    logits_available: bool = False
    logits_shape: list[int] | None = None


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_path(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _completion_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return ""
    choices = result.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    first = choices[0]
    if isinstance(first.get("text"), str):
        return first["text"]
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return ""


def _extract_completion_telemetry(raw_response: JsonDict | None) -> JsonDict:
    response_text = _completion_text(raw_response)
    usage = raw_response.get("usage", {}) if isinstance(raw_response, dict) else {}
    completion_tokens = usage.get("completion_tokens")
    choices = raw_response.get("choices") if isinstance(raw_response, dict) else None
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    logprobs = first.get("logprobs") if isinstance(first, dict) else None
    if not isinstance(logprobs, dict):
        return {
            "response_text": response_text,
            "completion_tokens": completion_tokens if isinstance(completion_tokens, int) else 0,
            "token_texts": [],
            "token_logprobs": [],
            "top_logprobs": [],
        }

    top_logprobs: list[dict[str, float]] = []
    for top in logprobs.get("top_logprobs") or []:
        if not isinstance(top, dict):
            continue
        converted = {
            str(key): coerced
            for key, value in top.items()
            if (coerced := _coerce_float(value)) is not None
        }
        if converted:
            top_logprobs.append(converted)

    return {
        "response_text": response_text,
        "completion_tokens": completion_tokens if isinstance(completion_tokens, int) else 0,
        "token_texts": [
            str(token) for token in logprobs.get("tokens") or [] if token is not None
        ],
        "token_logprobs": [
            coerced
            for value in logprobs.get("token_logprobs") or []
            if (coerced := _coerce_float(value)) is not None
        ],
        "top_logprobs": top_logprobs,
    }


def _logits_summary(llm: Any) -> tuple[bool, list[int]]:
    scores = getattr(llm, "scores", None)
    shape = getattr(scores, "shape", None)
    if shape:
        return True, [int(part) for part in shape]
    if isinstance(scores, list) and scores and isinstance(scores[0], list):
        return True, [len(scores), len(scores[0])]
    eval_logits = getattr(llm, "eval_logits", None)
    if isinstance(eval_logits, list) and eval_logits and isinstance(eval_logits[0], list):
        return True, [len(eval_logits), len(eval_logits[0])]
    return False, []


def _default_llama_importer() -> tuple[bool, type[Any] | None, str | None]:  # pragma: no cover
    try:
        from llama_cpp import Llama  # noqa: PLC0415
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, Llama, None


def _close_llama(llm: Any) -> None:
    close = getattr(llm, "close", None)
    if callable(close):
        close()


def _task_prompt(family: str, statement: str) -> str:
    if family in {"fover_claim", "constraint_check"}:
        return f"{statement} Return 1 if true, 0 if false."
    return f"{statement} What is the final answer?"


def _case_prompt(
    *,
    family: str,
    statement: str,
    expected: str,
    wrong: str,
    correct: bool,
    format_valid: bool,
) -> str:
    task = _task_prompt(family, statement)
    if correct and format_valid:
        return f"{task} Return the final integer only."
    if not correct and format_valid:
        return (
            "Telemetry obedience check. Do not solve the task. "
            f"Return exactly this integer and no other text: {wrong}"
        )
    if correct and not format_valid:
        return (
            f"{task} Reply in one short sentence, not as a bare integer. "
            f"The sentence must include the answer {expected}."
        )
    return (
        "Telemetry obedience check. Do not solve the task. "
        "Reply in one short sentence, not as a bare integer. "
        f"The sentence must include the deliberately wrong answer {wrong}."
    )


def build_balanced_telemetry_cases() -> list[BalancedTelemetryCase]:
    """Return the fixed 36-case adversarially balanced prompt set.

    The prompts ask the model to repeat a candidate response.  That is
    deliberate: this experiment is not measuring mathematical skill.  It is
    measuring whether token telemetry can survive an audit when correctness and
    format labels are balanced instead of accidentally correlated with prompt
    family or answer shape.
    """

    stems = [
        ("fover_claim", "Verify claim: 2 + 5 = 7.", "1", "0"),
        ("arithmetic_word_problem", "Mia has 1 marble and gets 2 more.", "3", "4"),
        ("constraint_check", "Constraint x=2 satisfies x+3=5.", "1", "0"),
        ("fover_claim", "Verify claim: 4 * 3 = 11.", "0", "1"),
        ("arithmetic_word_problem", "A bus has 9 riders, 4 get off, and 6 get on.", "11", "10"),
        ("constraint_check", "Constraint x=4 satisfies 2*x=10.", "0", "1"),
        ("fover_claim", "Verify claim: 15 - 6 = 9.", "1", "0"),
        ("arithmetic_word_problem", "Noah buys 3 packs with 4 pencils each.", "12", "10"),
        ("constraint_check", "Constraint y=3 satisfies y*y=9.", "1", "0"),
    ]
    buckets = [
        ("correct_format_valid", True, True),
        ("incorrect_format_valid", False, True),
        ("correct_format_invalid", True, False),
        ("incorrect_format_invalid", False, False),
    ]
    cases: list[BalancedTelemetryCase] = []
    for bucket_name, correct, format_valid in buckets:
        for idx, (family, statement, expected, wrong) in enumerate(stems, start=1):
            prompt = _case_prompt(
                family=family,
                statement=statement,
                expected=expected,
                wrong=wrong,
                correct=correct,
                format_valid=format_valid,
            )
            cases.append(
                BalancedTelemetryCase(
                    case_id=f"balanced_{bucket_name}_{idx:03d}",
                    family=family,
                    prompt=prompt,
                    expected_answer=expected,
                    adversarial_wrong_answer=wrong,
                    intended_correct=correct,
                    intended_format_valid=format_valid,
                )
            )
    return cases


def _strip_think_wrappers(response_text: str) -> str:
    stripped = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    return stripped.strip("`").strip()


def _leading_integer_before_think(response_text: str) -> str | None:
    match = re.match(r"\s*([+-]?\d+)\s*(?:<think>|\n+\s*<think>)", response_text)
    return match.group(1) if match else None


def _extract_answer_text(response_text: str) -> str | None:
    if (leading := _leading_integer_before_think(response_text)) is not None:
        return leading
    cleaned = _strip_think_wrappers(response_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and parsed.get("answer") is not None:
        return str(parsed["answer"]).strip()
    phrase = re.search(r"answer\s*(?:is|:)\s*([+-]?\d+)", cleaned, flags=re.IGNORECASE)
    if phrase:
        return phrase.group(1)
    if re.fullmatch(r"[+-]?\d+", cleaned):
        return cleaned
    matches = re.findall(r"(?<!\d)[+-]?\d+(?!\d)", cleaned)
    return matches[-1] if matches else None


def _format_valid(response_text: str) -> bool:
    if _leading_integer_before_think(response_text) is not None:
        return True
    return bool(re.fullmatch(r"[+-]?\d+", _strip_think_wrappers(response_text)))


def _answer_lexical_overlap(response_text: str, expected_answer: str) -> float:
    return 1.0 if _extract_answer_text(response_text) == str(expected_answer).strip() else 0.0


def _json_valid(response_text: str) -> bool:
    try:
        json.loads(response_text)
    except json.JSONDecodeError:
        return False
    return True


def _evaluate_row_labels(case: BalancedTelemetryCase, response_text: str) -> dict[str, bool]:
    return {
        "correct": _answer_lexical_overlap(response_text, case.expected_answer) == 1.0,
        "format_valid": _format_valid(response_text),
    }


def _model_family(spec: Mapping[str, Any]) -> str:
    hf_id = str(spec.get("hf_id", "")).lower()
    if "qwen3.6-35b-a3b" in hf_id:
        return "qwen_moe"
    if "gemma-4-31b" in hf_id:
        return "gemma_dense"
    if "gemma-4-26b-a4b" in hf_id:
        return "gemma_moe"
    return "unknown_or_non_sota"


def _superficial_baselines(
    case: BalancedTelemetryCase,
    *,
    response_text: str,
    completion_tokens: int,
    spec: Mapping[str, Any],
) -> JsonDict:
    return {
        "response_length": len(response_text),
        "token_count": int(completion_tokens),
        "json_valid": _json_valid(response_text),
        "schema_valid": _format_valid(response_text),
        "prompt_family": case.family,
        "answer_lexical_overlap": _answer_lexical_overlap(response_text, case.expected_answer),
        "model_family": _model_family(spec),
    }


def _generate_with_llama(llm: Any, case: BalancedTelemetryCase) -> RawBalancedGeneration:
    started = time.monotonic()
    try:
        result = llm(
            case.prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            top_p=1.0,
            echo=False,
            stop=["</s>", "<eos>"],
            logprobs=LOGPROBS_REQUESTED,
        )
        logprob_error = None
    except TypeError as exc:
        result = llm(
            case.prompt,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            top_p=1.0,
            echo=False,
            stop=["</s>", "<eos>"],
        )
        logprob_error = f"logprobs_unavailable: {exc}"

    logits_available, logits_shape = _logits_summary(llm)
    telemetry = _extract_completion_telemetry(result if isinstance(result, dict) else None)
    return RawBalancedGeneration(
        response_text=telemetry["response_text"] or _completion_text(result),
        raw_response=result if isinstance(result, dict) else None,
        elapsed_seconds=max(time.monotonic() - started, 0.0),
        logprob_error=logprob_error,
        logits_available=logits_available,
        logits_shape=logits_shape,
    )


def _row_from_generation(
    spec: JsonDict,
    case: BalancedTelemetryCase,
    raw: RawBalancedGeneration,
    generation_source: str,
) -> JsonDict:
    telemetry = _extract_completion_telemetry(raw.raw_response)
    response_text = raw.response_text or telemetry["response_text"]
    completion_tokens = int(telemetry["completion_tokens"] or len(response_text.split()))
    token_logprobs = telemetry["token_logprobs"]
    top_logprobs = telemetry["top_logprobs"]
    topk_available = any(len(top) >= TOPK_MIN_ALTERNATIVES for top in top_logprobs)
    labels = _evaluate_row_labels(case, response_text)
    baselines = _superficial_baselines(
        case,
        response_text=response_text,
        completion_tokens=completion_tokens,
        spec=spec,
    )
    blocker_parts = [
        part
        for part in (
            raw.error,
            raw.logprob_error,
            None if response_text.strip() else "response_text_unavailable",
            None if token_logprobs else "token_logprobs_missing",
            None if topk_available else "topk_logprobs_missing",
            None if raw.logits_available else "logits_unavailable",
        )
        if part
    ]
    correctness_label = "correct" if labels["correct"] else "incorrect"
    format_label = "format_valid" if labels["format_valid"] else "format_invalid"
    return {
        "case_id": case.case_id,
        "family": case.family,
        "prompt": case.prompt,
        "expected_answer": case.expected_answer,
        "adversarial_wrong_answer": case.adversarial_wrong_answer,
        "intended_correct": case.intended_correct,
        "intended_format_valid": case.intended_format_valid,
        "correct": labels["correct"],
        "format_valid": labels["format_valid"],
        "correctness_label": correctness_label,
        "format_label": format_label,
        "known_verifier_label": 1 if labels["correct"] else 0,
        "model_name": spec.get("name"),
        "hf_id": spec.get("hf_id"),
        "gpu": spec.get("gpu"),
        "model_path": spec.get("model_path"),
        "model_family": baselines["model_family"],
        "response_text": response_text,
        "response_text_available": bool(response_text.strip()),
        "completion_tokens": completion_tokens,
        "token_texts": telemetry["token_texts"],
        "token_logprobs": token_logprobs,
        "token_logprobs_available": bool(token_logprobs),
        "top_logprobs": top_logprobs,
        "topk_alternatives_available": topk_available,
        "topk_position_count": len(top_logprobs),
        "logits_available": bool(raw.logits_available),
        "logits_shape": raw.logits_shape or [],
        "elapsed_seconds": round(raw.elapsed_seconds, 6),
        "error": raw.error,
        "logprob_error": raw.logprob_error,
        "generation_source": generation_source,
        "superficial_baselines": baselines,
        "blocker": "; ".join(blocker_parts) if blocker_parts else None,
    }


def _collect_with_generation_fn(
    spec: JsonDict,
    cases: Sequence[BalancedTelemetryCase],
    generation_fn: GenerationFn,
    generation_source: str,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for case in cases:
        try:
            raw = generation_fn(spec, case)
        except Exception as exc:
            raw = RawBalancedGeneration("", error=f"{type(exc).__name__}: {exc}")
        rows.append(_row_from_generation(spec, case, raw, generation_source))
    return rows


def _collect_with_llama(
    spec: JsonDict,
    cases: Sequence[BalancedTelemetryCase],
    *,
    llama_class: type[Any],
) -> list[JsonDict]:
    llm: Any | None = None
    try:
        llm = llama_class(
            model_path=spec["model_path"],
            n_gpu_layers=-1,
            main_gpu=int(spec.get("gpu") or 0),
            n_ctx=1024,
            seed=1480,
            logits_all=True,
            verbose=False,
        )
        return [
            _row_from_generation(
                spec,
                case,
                _generate_with_llama(llm, case),
                "live_sota_llamacpp",
            )
            for case in cases
        ]
    except Exception as exc:
        return [
            _row_from_generation(
                spec,
                case,
                RawBalancedGeneration("", error=f"{type(exc).__name__}: {exc}"),
                "live_sota_llamacpp",
            )
            for case in cases
        ]
    finally:
        if llm is not None:
            _close_llama(llm)


def _resolved_specs(raw_specs: list[JsonDict] | None) -> list[JsonDict]:
    if not isinstance(raw_specs, list):
        return []
    resolved: list[JsonDict] = []
    for spec in raw_specs:
        hf_id = spec.get("hf_id")
        model_path = spec.get("model_path")
        if hf_id not in MANDATED_MODEL_IDS or not model_path:
            continue
        resolved.append(
            {
                "name": spec.get("name"),
                "hf_id": hf_id,
                "gpu": spec.get("gpu", 0),
                "model_path": model_path,
            }
        )
    return resolved


def _select_probe_spec(specs: Sequence[JsonDict]) -> JsonDict | None:
    for preferred in MANDATED_MODEL_IDS:
        for spec in specs:
            if spec.get("hf_id") == preferred:
                return dict(spec)
    return None


def _label_counts(rows: Sequence[JsonDict]) -> JsonDict:
    if not rows:
        return {}
    counts = {
        "correct": 0,
        "incorrect": 0,
        "format_valid": 0,
        "format_invalid": 0,
        "correct_format_valid": 0,
        "incorrect_format_valid": 0,
        "correct_format_invalid": 0,
        "incorrect_format_invalid": 0,
    }
    for row in rows:
        correct = bool(row.get("correct"))
        format_valid = bool(row.get("format_valid"))
        counts["correct" if correct else "incorrect"] += 1
        counts["format_valid" if format_valid else "format_invalid"] += 1
        if correct and format_valid:
            counts["correct_format_valid"] += 1
        elif not correct and format_valid:
            counts["incorrect_format_valid"] += 1
        elif correct and not format_valid:
            counts["correct_format_invalid"] += 1
        else:
            counts["incorrect_format_invalid"] += 1
    return counts


def _labels_are_balanced(counts: Mapping[str, Any]) -> bool:
    keys = (
        "correct",
        "incorrect",
        "format_valid",
        "format_invalid",
        "correct_format_valid",
        "incorrect_format_valid",
        "correct_format_invalid",
        "incorrect_format_invalid",
    )
    return bool(counts) and all(int(counts.get(key) or 0) > 0 for key in keys)


def _superficial_baselines_complete(rows: Sequence[JsonDict]) -> bool:
    return bool(rows) and all(
        isinstance(row.get("superficial_baselines"), dict)
        and set(SUPERFICIAL_BASELINE_FIELDS) <= set(row["superficial_baselines"])
        for row in rows
    )


def _summarize_rows(rows: Sequence[JsonDict]) -> JsonDict:
    completed = [
        row
        for row in rows
        if not row.get("error") and bool(str(row.get("response_text") or "").strip())
    ]
    completed_count = len(completed)
    topk_rows = [row for row in completed if row.get("topk_alternatives_available")]
    topk_positions = sum(int(row.get("topk_position_count") or 0) for row in completed)
    topk_case_coverage = len(topk_rows) / completed_count if completed_count else 0.0
    return {
        "completed_count": completed_count,
        "topk_position_count": topk_positions,
        "topk_case_coverage": round(topk_case_coverage, 6),
        "topk_ready": (
            completed_count > 0
            and topk_case_coverage >= TOPK_MIN_CASE_COVERAGE
            and topk_positions >= completed_count
        ),
        "logits_available": any(bool(row.get("logits_available")) for row in completed),
        "label_counts": _label_counts(completed),
        "baselines_complete": _superficial_baselines_complete(completed),
    }


def _unique_blockers(values: Sequence[str | None]) -> list[str]:
    blockers: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        blockers.append(value)
    return blockers


def _base_artifact(*, project_root: Path, run_date: str, manifest_path: Path) -> JsonDict:
    return {
        "status": "complete",
        "artifact": "experiment_1480_live_sota_balanced_telemetry_v2",
        "schema_version": 1,
        "run_date": run_date,
        "project_root": str(project_root),
        "model_specs": [dict(spec) for spec in MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "telemetry_cases_requested": len(build_balanced_telemetry_cases()),
        "telemetry_cases_completed": 0,
        "balanced_label_counts": {},
        "topk_logprobs_available": False,
        "logits_available": False,
        "superficial_baselines_recorded": False,
        "telemetry_manifest_path": _display_path(project_root, manifest_path),
        "models_used": [],
        "resolved_model_specs": [],
        "gpu_probe": {},
        "manifest_summary": {
            "topk_position_count": 0,
            "topk_case_coverage": 0.0,
            "logprobs_requested": LOGPROBS_REQUESTED,
        },
        "blockers": [],
        "claim_allowed": False,
        "honest_verdict": "blocked_no_live_sota_balanced_telemetry",
    }


def _in_progress_artifact(*, project_root: Path, run_date: str, manifest_path: Path) -> JsonDict:
    artifact = _base_artifact(project_root=project_root, run_date=run_date, manifest_path=manifest_path)
    artifact["status"] = "in_progress"
    artifact["blockers"] = ["experiment_1480_live_sota_balanced_telemetry_in_progress"]
    artifact["honest_verdict"] = "in_progress"
    return artifact


def build_balanced_telemetry_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    llama_importer: LlamaImporter = _default_llama_importer,
    cases: Sequence[BalancedTelemetryCase] | None = None,
    generation_fn: GenerationFn | None = None,
    generation_source: str = "live_sota_llamacpp",
    gpu_probe_fn: GpuProbeFn = probe_gpu_state,
) -> JsonDict:
    """Build the terminal Exp 1480 artifact and row-level JSONL manifest.

    The function is dependency-injected so tests can exercise the balance and
    gating rules without loading a 20+ GB model.  Production runs leave
    `generation_fn=None`, which loads the selected mandated GGUF via llama.cpp.
    """

    root = Path(project_root)
    manifest = _resolve_path(root, manifest_path)
    selected_cases = list(cases) if cases is not None else build_balanced_telemetry_cases()
    artifact = _base_artifact(project_root=root, run_date=run_date, manifest_path=manifest)
    artifact["telemetry_cases_requested"] = len(selected_cases)
    artifact["gpu_probe"] = gpu_probe_fn()

    try:
        specs = _resolved_specs(cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M"))
    except Exception as exc:
        specs = []
        artifact["cached_sota_pair_error"] = f"{type(exc).__name__}: {exc}"
    artifact["resolved_model_specs"] = specs
    spec = _select_probe_spec(specs)
    if spec is None:
        artifact["blockers"] = _unique_blockers(
            ["cached_sota_pair_not_loadable", artifact.get("cached_sota_pair_error")]
        )
        _write_jsonl(manifest, [])
        return artifact

    artifact["models_used"] = [str(spec["hf_id"])]
    if generation_fn is None:
        import_ok, llama_class, import_error = llama_importer()
        artifact["llama_cpp_import_ok"] = import_ok
        artifact["llama_cpp_import_error"] = import_error
        if not import_ok or llama_class is None:
            rows = [
                _row_from_generation(
                    spec,
                    case,
                    RawBalancedGeneration("", error=import_error or "llama_cpp_import_failed"),
                    "live_sota_llamacpp",
                )
                for case in selected_cases
            ]
        else:
            rows = _collect_with_llama(spec, selected_cases, llama_class=llama_class)
    else:
        artifact["llama_cpp_import_ok"] = None
        artifact["llama_cpp_import_error"] = None
        rows = _collect_with_generation_fn(spec, selected_cases, generation_fn, generation_source)

    _write_jsonl(manifest, rows)
    summary = _summarize_rows(rows)
    live_used = summary["completed_count"] > 0 and generation_source == "live_sota_llamacpp"
    labels_balanced = _labels_are_balanced(summary["label_counts"])
    claim_allowed = (
        live_used
        and labels_balanced
        and bool(summary["topk_ready"])
        and bool(summary["logits_available"])
        and bool(summary["baselines_complete"])
    )
    blockers: list[str] = []
    if summary["completed_count"] == 0:
        blockers.append("no_live_sota_generation_completed")
    if not live_used:
        blockers.append("live_sota_model_inference_not_used")
    if not labels_balanced:
        blockers.append("balanced_labels_missing")
    if not summary["topk_ready"]:
        blockers.append("topk_logprobs_unavailable_or_insufficient")
    if not summary["logits_available"]:
        blockers.append("logits_unavailable_from_llamacpp_response")
    if not summary["baselines_complete"]:
        blockers.append("superficial_baselines_missing")
    row_error_blockers = [str(row.get("blocker")) for row in rows if row.get("error")]

    artifact["live_sota_model_inference_used"] = live_used
    artifact["telemetry_cases_completed"] = summary["completed_count"]
    artifact["balanced_label_counts"] = summary["label_counts"]
    artifact["topk_logprobs_available"] = summary["topk_ready"]
    artifact["logits_available"] = summary["logits_available"]
    artifact["superficial_baselines_recorded"] = summary["baselines_complete"]
    artifact["manifest_summary"] = {
        "topk_position_count": summary["topk_position_count"],
        "topk_case_coverage": summary["topk_case_coverage"],
        "logprobs_requested": LOGPROBS_REQUESTED,
    }
    artifact["claim_allowed"] = claim_allowed
    artifact["blockers"] = _unique_blockers([*blockers, *row_error_blockers])
    artifact["honest_verdict"] = (
        "balanced_live_sota_telemetry_ready"
        if claim_allowed
        else "balanced_live_sota_telemetry_claim_blocked"
        if live_used
        else "blocked_no_live_sota_balanced_telemetry"
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path = Path("."),
    run_date: str = DEFAULT_RUN_DATE,
    output_path: str | Path = DEFAULT_ARTIFACT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    llama_importer: LlamaImporter = _default_llama_importer,
    cases: Sequence[BalancedTelemetryCase] | None = None,
    generation_fn: GenerationFn | None = None,
    generation_source: str = "live_sota_llamacpp",
    gpu_probe_fn: GpuProbeFn = probe_gpu_state,
    write_json_fn: WriteJsonFn = _write_json,
) -> JsonDict:
    """Write the in-progress marker, final artifact, and JSONL manifest."""

    root = Path(project_root)
    output = _resolve_path(root, output_path)
    manifest = _resolve_path(root, manifest_path)
    write_json_fn(output, _in_progress_artifact(project_root=root, run_date=run_date, manifest_path=manifest))
    artifact = build_balanced_telemetry_artifact(
        project_root=root,
        run_date=run_date,
        manifest_path=manifest,
        cached_pair_fn=cached_pair_fn,
        llama_importer=llama_importer,
        cases=cases,
        generation_fn=generation_fn,
        generation_source=generation_source,
        gpu_probe_fn=gpu_probe_fn,
    )
    write_json_fn(output, artifact)
    return artifact


def _parse_args() -> argparse.Namespace:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-date", default=DEFAULT_RUN_DATE)
    parser.add_argument("--output-path", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI wrapper.
    args = _parse_args()
    run_experiment(
        project_root=args.project_root,
        run_date=args.run_date,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
    )


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()


__all__ = [
    "REQUIRED_ARTIFACT_FIELDS",
    "SUPERFICIAL_BASELINE_FIELDS",
    "BalancedTelemetryCase",
    "RawBalancedGeneration",
    "build_balanced_telemetry_artifact",
    "build_balanced_telemetry_cases",
    "run_experiment",
]
