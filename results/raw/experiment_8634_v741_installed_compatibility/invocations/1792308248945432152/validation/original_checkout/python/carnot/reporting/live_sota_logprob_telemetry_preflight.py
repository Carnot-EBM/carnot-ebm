"""Live SOTA GGUF logprob telemetry preflight for Exp 1468.

The Exp 1463 runtime repair proved that a mandated local SOTA GGUF can produce
real text through llama.cpp.  This follow-up answers the narrower question that
matters for HALT, Spilled Energy, and BEAVER-lite features: does the same local
path expose per-token telemetry, especially top-k alternatives, while producing
bounded FoVer/GSM8K-style generations?

Spec: REQ-INFER-SOTA-009,
      SCENARIO-INFER-SOTA-009-001,
      SCENARIO-INFER-SOTA-009-002
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import cached_sota_pair
from carnot.reporting.live_sota_repair_runtime_preflight import probe_gpu_state


DEFAULT_RUN_DATE = "20260507"
DEFAULT_ARTIFACT_PATH = Path(
    "results/experiment_1468_live_sota_logprob_telemetry_preflight.json"
)
DEFAULT_MANIFEST_PATH = Path("results/live_sota_telemetry_manifest_1468.jsonl")
MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_SPECS: tuple[dict[str, str], ...] = (
    {"hf_id": MANDATED_MODEL_IDS[0], "role": "flagship_moe_telemetry_probe"},
    {"hf_id": MANDATED_MODEL_IDS[1], "role": "flagship_dense_telemetry_probe"},
    {"hf_id": MANDATED_MODEL_IDS[2], "role": "middle_moe_telemetry_probe"},
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "telemetry_cases_requested",
    "telemetry_cases_completed",
    "topk_logprobs_available",
    "logits_available",
    "telemetry_manifest_path",
    "models_used",
    "gpu_probe",
    "blockers",
    "honest_verdict",
)
TOPK_MIN_CASE_COVERAGE = 0.8
TOPK_MIN_ALTERNATIVES = 2
LOGPROBS_REQUESTED = 5
MAX_TOKENS = 48

JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[JsonDict] | None]
GenerationFn = Callable[[JsonDict, "TelemetryCase"], "RawTelemetryGeneration"]
LlamaImporter = Callable[[], tuple[bool, type[Any] | None, str | None]]
GpuProbeFn = Callable[[], JsonDict]
WriteJsonFn = Callable[[Path, JsonDict], None]


@dataclass(frozen=True)
class TelemetryCase:
    """One bounded arithmetic or formal-verification-style telemetry prompt."""

    case_id: str
    family: str
    prompt: str
    expected_answer: str


@dataclass(frozen=True)
class RawTelemetryGeneration:
    """Raw generation result before it is normalized into a manifest row."""

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
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


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


def build_telemetry_cases() -> list[TelemetryCase]:
    """Return the fixed 12-case FoVer/GSM8K-style telemetry prompt set."""
    rows = [
        (
            "gsm8k_style",
            "Mia has 1 marble and gets 2 more. Answer with the final integer only.",
            "3",
        ),
        (
            "gsm8k_style",
            "A bus has 9 riders. 4 get off and 6 get on. Answer with the final integer only.",
            "11",
        ),
        (
            "gsm8k_style",
            "Noah buys 3 packs with 4 pencils each. Answer with the final integer only.",
            "12",
        ),
        (
            "gsm8k_style",
            "A jar has 18 coins split equally into 3 bags. Answer with the final integer only.",
            "6",
        ),
        (
            "gsm8k_style",
            "Lina read 7 pages on Monday and twice that on Tuesday. Answer with the final integer only.",
            "14",
        ),
        (
            "gsm8k_style",
            "A shelf starts with 20 books. 8 are borrowed and 5 returned. Answer with the final integer only.",
            "17",
        ),
        (
            "fover_style",
            "Verify the arithmetic claim: 2 + 5 = 7. Return 1 if true, 0 if false.",
            "1",
        ),
        (
            "fover_style",
            "Verify the arithmetic claim: 4 * 3 = 11. Return 1 if true, 0 if false.",
            "0",
        ),
        (
            "fover_style",
            "Constraint check: x=2 satisfies x+3=5. Return 1 if true, 0 if false.",
            "1",
        ),
        (
            "fover_style",
            "Constraint check: x=4 satisfies 2*x=10. Return 1 if true, 0 if false.",
            "0",
        ),
        (
            "fover_style",
            "A step says 15 - 6 = 9. Return 1 if the step is valid, else 0.",
            "1",
        ),
        (
            "fover_style",
            "A step says 21 / 7 = 4. Return 1 if the step is valid, else 0.",
            "0",
        ),
    ]
    return [
        TelemetryCase(
            case_id=f"fover_gsm8k_verified_{idx:03d}",
            family=family,
            prompt=prompt,
            expected_answer=answer,
        )
        for idx, (family, prompt, answer) in enumerate(rows, start=1)
    ]


def _completion_text(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result or "")
    choices = result.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    first = choices[0]
    if "text" in first:
        return str(first.get("text") or "")
    message = first.get("message") or {}
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return ""


def _extract_completion_telemetry(raw_response: JsonDict | None) -> JsonDict:
    """Extract OpenAI-style llama.cpp completion telemetry from a raw response."""
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


def _generate_with_llama(llm: Any, case: TelemetryCase) -> RawTelemetryGeneration:
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
    text = telemetry["response_text"] or _completion_text(result)
    return RawTelemetryGeneration(
        response_text=text,
        raw_response=result if isinstance(result, dict) else None,
        elapsed_seconds=max(time.monotonic() - started, 0.0),
        logprob_error=logprob_error,
        logits_available=logits_available,
        logits_shape=logits_shape,
    )


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


def _row_from_generation(
    spec: JsonDict,
    case: TelemetryCase,
    raw: RawTelemetryGeneration,
    generation_source: str,
) -> JsonDict:
    telemetry = _extract_completion_telemetry(raw.raw_response)
    response_text = raw.response_text or telemetry["response_text"]
    top_logprobs = telemetry["top_logprobs"]
    token_logprobs = telemetry["token_logprobs"]
    topk_available = any(len(top) >= TOPK_MIN_ALTERNATIVES for top in top_logprobs)
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
    return {
        "case_id": case.case_id,
        "family": case.family,
        "prompt": case.prompt,
        "expected_answer": case.expected_answer,
        "model_name": spec.get("name"),
        "hf_id": spec.get("hf_id"),
        "gpu": spec.get("gpu"),
        "model_path": spec.get("model_path"),
        "response_text": response_text,
        "response_text_available": bool(response_text.strip()),
        "completion_tokens": telemetry["completion_tokens"],
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
        "blocker": "; ".join(blocker_parts) if blocker_parts else None,
    }


def _collect_with_generation_fn(
    spec: JsonDict,
    cases: Sequence[TelemetryCase],
    generation_fn: GenerationFn,
    generation_source: str,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for case in cases:
        try:
            raw = generation_fn(spec, case)
        except Exception as exc:
            raw = RawTelemetryGeneration("", error=f"{type(exc).__name__}: {exc}")
        rows.append(_row_from_generation(spec, case, raw, generation_source))
    return rows


def _collect_with_llama(
    spec: JsonDict,
    cases: Sequence[TelemetryCase],
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
            seed=1468,
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
                RawTelemetryGeneration("", error=f"{type(exc).__name__}: {exc}"),
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
    return dict(specs[0]) if specs else None


def _summarize_rows(rows: Sequence[JsonDict]) -> JsonDict:
    completed = [
        row
        for row in rows
        if not row.get("error") and bool(str(row.get("response_text") or "").strip())
    ]
    completed_count = len(completed)
    topk_rows = [row for row in completed if row.get("topk_alternatives_available")]
    topk_positions = sum(int(row.get("topk_position_count") or 0) for row in completed)
    topk_case_coverage = (len(topk_rows) / completed_count) if completed_count else 0.0
    topk_ready = (
        completed_count > 0
        and topk_case_coverage >= TOPK_MIN_CASE_COVERAGE
        and topk_positions >= completed_count
    )
    return {
        "completed_count": completed_count,
        "topk_position_count": topk_positions,
        "topk_case_coverage": round(topk_case_coverage, 6),
        "topk_ready": topk_ready,
        "logits_available": any(bool(row.get("logits_available")) for row in completed),
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
        "artifact": "experiment_1468_live_sota_logprob_telemetry_preflight",
        "schema_version": 1,
        "run_date": run_date,
        "project_root": str(project_root),
        "model_specs": [dict(spec) for spec in MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "telemetry_cases_requested": len(build_telemetry_cases()),
        "telemetry_cases_completed": 0,
        "topk_logprobs_available": False,
        "logits_available": False,
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
        "honest_verdict": "blocked_no_live_sota_telemetry",
    }


def _in_progress_artifact(*, project_root: Path, run_date: str, manifest_path: Path) -> JsonDict:
    artifact = _base_artifact(project_root=project_root, run_date=run_date, manifest_path=manifest_path)
    artifact["status"] = "in_progress"
    artifact["blockers"] = ["experiment_1468_live_sota_logprob_telemetry_in_progress"]
    artifact["honest_verdict"] = "in_progress"
    return artifact


def build_telemetry_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    llama_importer: LlamaImporter = _default_llama_importer,
    cases: Sequence[TelemetryCase] | None = None,
    generation_fn: GenerationFn | None = None,
    generation_source: str = "injected",
    gpu_probe_fn: GpuProbeFn = probe_gpu_state,
) -> JsonDict:
    """Build the Exp 1468 artifact and write its per-case JSONL manifest."""
    root = Path(project_root)
    manifest = _resolve_path(root, manifest_path)
    selected_cases = list(cases) if cases is not None else build_telemetry_cases()
    artifact = _base_artifact(project_root=root, run_date=run_date, manifest_path=manifest)
    artifact["telemetry_cases_requested"] = len(selected_cases)
    artifact["gpu_probe"] = gpu_probe_fn()

    try:
        raw_specs = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
        specs = _resolved_specs(raw_specs)
    except Exception as exc:
        specs = []
        artifact["cached_sota_pair_error"] = f"{type(exc).__name__}: {exc}"

    artifact["resolved_model_specs"] = specs
    spec = _select_probe_spec(specs)
    if spec is None:
        artifact["blockers"] = _unique_blockers(
            [
                "cached_sota_pair_not_loadable",
                artifact.get("cached_sota_pair_error"),
            ]
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
                    RawTelemetryGeneration("", error=import_error or "llama_cpp_import_failed"),
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
    artifact["live_sota_model_inference_used"] = live_used
    artifact["telemetry_cases_completed"] = summary["completed_count"]
    artifact["topk_logprobs_available"] = summary["topk_ready"]
    artifact["logits_available"] = summary["logits_available"]
    artifact["manifest_summary"] = {
        "topk_position_count": summary["topk_position_count"],
        "topk_case_coverage": summary["topk_case_coverage"],
        "logprobs_requested": LOGPROBS_REQUESTED,
    }

    row_blockers = [str(row.get("blocker")) for row in rows if row.get("error")]
    blockers = []
    if summary["completed_count"] == 0:
        blockers.append("no_live_sota_generation_completed")
    if not summary["topk_ready"]:
        blockers.append("topk_logprobs_unavailable_or_insufficient")
    if not summary["logits_available"]:
        blockers.append("logits_unavailable_from_llamacpp_response")
    artifact["blockers"] = _unique_blockers([*blockers, *row_blockers])
    artifact["honest_verdict"] = (
        "live_sota_topk_telemetry_ready"
        if live_used and summary["topk_ready"]
        else "live_sota_text_only_telemetry_incomplete"
        if live_used
        else "blocked_no_live_sota_telemetry"
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path = ".",
    run_date: str = DEFAULT_RUN_DATE,
    output_path: str | Path = DEFAULT_ARTIFACT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    llama_importer: LlamaImporter = _default_llama_importer,
    cases: Sequence[TelemetryCase] | None = None,
    generation_fn: GenerationFn | None = None,
    generation_source: str = "live_sota_llamacpp",
    gpu_probe_fn: GpuProbeFn = probe_gpu_state,
    write_json_fn: WriteJsonFn = _write_json,
) -> JsonDict:
    """Write the bootstrap artifact, collect telemetry, then write final JSON."""
    root = Path(project_root)
    output = _resolve_path(root, output_path)
    manifest = _resolve_path(root, manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_fn(
        output,
        _in_progress_artifact(project_root=root, run_date=run_date, manifest_path=manifest),
    )
    artifact = build_telemetry_artifact(
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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date", default=DEFAULT_RUN_DATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parse_args(argv)
    run_experiment(run_date=args.run_date, output_path=args.output, manifest_path=args.manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
