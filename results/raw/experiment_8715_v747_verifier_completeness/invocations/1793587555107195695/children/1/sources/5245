"""Exp 1323 SOTA GGUF token-health prompt/runtime diagnostic.

The prior SOTA GGUF milestones proved that the mandated local pair can load,
but Exp 1311 also showed a dangerous pattern: many raw outputs were empty or
only one token long.  This module keeps the follow-up diagnostic small and
auditable.  It compares prompt shape and runtime settings before any larger
certificate experiment spends more local SOTA model time.

Spec: REQ-VERIFY-1323,
      SCENARIO-VERIFY-1323
"""

from __future__ import annotations

import gc
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import cached_sota_pair


DEFAULT_RUN_DATE = "20260505"
DEFAULT_ARTIFACT_PATH = Path(
    "results/experiment_1323_sota_gguf_token_health_prompt_runtime_diagnostic.json"
)
EXP1310_ARTIFACT_PATH = Path("results/experiment_1310_sota_gguf_llamacpp_smoke_load.json")
EXP1311_ARTIFACT_PATH = Path(
    "results/experiment_1311_sota_constraintbench_satquest_answer_stability.json"
)
EXP1312_ARTIFACT_PATH = Path(
    "results/experiment_1312_triggered_certificate_extraction_dccd_gbnf.json"
)
MANDATED_HEADLINE_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "models_used",
    "prompt_variants_tested",
    "generation_settings",
    "empty_or_one_token_rate",
    "min_tokens_recovered",
    "topk_logprob_available",
    "entropy_production_rate_available",
    "certificate_parse_delta_with_probe_gate",
    "recommended_certificate_runtime_settings",
    "headline_result_allowed",
    "honest_verdict",
)

CachedPairFn = Callable[..., list[dict[str, Any]] | None]
LlamaImporter = Callable[[], tuple[bool, type[Any] | None, str | None]]


@dataclass(frozen=True)
class PromptVariant:
    """One deterministic prompt/runtime setting in the Exp 1323 grid."""

    name: str
    prompt: str
    max_tokens: int
    stop: list[str]
    prompt_shape: str
    use_chat_template: bool = False
    certificate_shaped: bool = False
    grammar: str = "none"


@dataclass(frozen=True)
class RawProbeGeneration:
    """Raw result from one prompt variant before artifact normalization."""

    text: str
    token_count: int
    elapsed_seconds: float = 0.0
    raw_response: dict[str, Any] | None = None
    error: str | None = None
    logprob_error: str | None = None
    used_chat_template: bool = False


GenerationFn = Callable[[dict[str, Any], PromptVariant], RawProbeGeneration]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_prompt_variants() -> list[PromptVariant]:
    """Return the fixed prompt/runtime grid mandated for Exp 1323."""
    label_prompt = (
        "Return exactly one label: SAT, UNSAT, or UNKNOWN.\n"
        "Item cb_sat_schedule: Boolean x1 means task A is selected, x2 means task B "
        "is selected. Constraints: at least one task is selected; not both tasks are "
        "selected. Is the constraint set feasible?\n"
        "Final label:"
    )
    chat_prompt = (
        "Classify feasibility for this tiny Boolean constraint instance. Respond with "
        "a short reason and a final label from SAT, UNSAT, UNKNOWN. Instance: choose "
        "x1 or x2, but not both."
    )
    certificate_prompt = (
        "Produce a compact JSON certificate for this Boolean constraint instance. "
        "The JSON object must contain label, constraints, verifier, and rationale. "
        "Instance: choose x1 or x2, but not both. Expected verifier family: cnf."
    )
    return [
        PromptVariant(
            name="baseline_prompt",
            prompt=label_prompt,
            max_tokens=6,
            stop=["\n", "</s>", "<eos>"],
            prompt_shape="exp1311_bounded_label_completion",
        ),
        PromptVariant(
            name="chat_template_prompt",
            prompt=chat_prompt,
            max_tokens=32,
            stop=["</s>", "<eos>"],
            prompt_shape="llama_cpp_chat_template",
            use_chat_template=True,
        ),
        PromptVariant(
            name="no_stop_string_prompt",
            prompt=label_prompt,
            max_tokens=32,
            stop=[],
            prompt_shape="bounded_label_no_stop_strings",
        ),
        PromptVariant(
            name="larger_max_token_budget",
            prompt=label_prompt,
            max_tokens=64,
            stop=["</s>", "<eos>"],
            prompt_shape="bounded_label_larger_budget",
        ),
        PromptVariant(
            name="certificate_shaped_prompt",
            prompt=certificate_prompt,
            max_tokens=96,
            stop=["</s>", "<eos>"],
            prompt_shape="certificate_json_skeleton",
            certificate_shaped=True,
        ),
    ]


def _completion_text(result: Any) -> str:
    if not isinstance(result, dict):
        return str(result)
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


def _completion_token_count(result: Any, text: str, llm: Any) -> int:
    if isinstance(result, dict):
        completion_tokens = (result.get("usage") or {}).get("completion_tokens")
        if isinstance(completion_tokens, int):
            return max(0, completion_tokens)
    tokenize = getattr(llm, "tokenize", None)
    if callable(tokenize):
        try:
            return len(tokenize(text.encode("utf-8"), add_bos=False))
        except Exception:
            pass
    return len(text.split()) if text.strip() else 0


def _certificate_skeleton_available(text: str) -> bool:
    lowered = text.lower()
    return (
        "{" in text
        and "}" in text
        and "label" in lowered
        and "constraint" in lowered
        and "verifier" in lowered
    )


def _resolved_specs(raw_specs: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not isinstance(raw_specs, list) or len(raw_specs) != 2:
        return []
    resolved: list[dict[str, Any]] = []
    for spec in raw_specs:
        hf_id = spec.get("hf_id")
        model_path = spec.get("model_path")
        if hf_id not in MANDATED_HEADLINE_MODEL_IDS or not model_path:
            return []
        resolved.append(
            {
                "name": spec.get("name"),
                "hf_id": hf_id,
                "gpu": spec.get("gpu"),
                "model_path": model_path,
            }
        )
    return resolved


def _rate(flags: Sequence[bool]) -> float:
    return round(sum(1 for flag in flags if flag) / len(flags), 6) if flags else 0.0


def _raw_lengths(rows: Sequence[dict[str, Any]]) -> list[int]:
    return [len(str(row.get("raw_output") or "")) for row in rows]


def _summarize_prior_artifacts(project_root: Path) -> dict[str, Any]:
    exp1310 = _read_json(project_root / EXP1310_ARTIFACT_PATH)
    exp1311 = _read_json(project_root / EXP1311_ARTIFACT_PATH)
    exp1312 = _read_json(project_root / EXP1312_ARTIFACT_PATH)
    responses = exp1311.get("responses") if isinstance(exp1311.get("responses"), list) else []
    attempts = exp1312.get("attempts") if isinstance(exp1312.get("attempts"), list) else []
    smoke_rows = (
        exp1310.get("per_model_results")
        if isinstance(exp1310.get("per_model_results"), list)
        else []
    )

    return {
        "exp1310": {
            "artifact_found": bool(exp1310),
            "status": exp1310.get("status"),
            "models_used": exp1310.get("models_used") or [],
            "prompt_shapes": ["smoke_one_short_word_completion"],
            "token_budgets": [4],
            "stop_settings": [["</s>", "<eos>"]],
            "observed_token_counts": [row.get("token_count") for row in smoke_rows],
            "observed_raw_output_lengths": [],
            "empty_or_one_token_rate": _rate(
                [int(row.get("token_count") or 0) <= 1 for row in smoke_rows]
            ),
        },
        "exp1311": {
            "artifact_found": bool(exp1311),
            "status": exp1311.get("status"),
            "models_used": exp1311.get("models_used") or [],
            "prompt_shapes": ["bounded_label_completion"],
            "token_budgets": [6],
            "stop_settings": [["\n", "</s>", "<eos>"]],
            "observed_token_counts": [row.get("token_count") for row in responses],
            "observed_raw_output_lengths": _raw_lengths(responses),
            "empty_or_one_token_rate": _rate(
                [
                    int(row.get("token_count") or 0) <= 1
                    or not str(row.get("raw_output") or "").strip()
                    for row in responses
                ]
            ),
        },
        "exp1312": {
            "artifact_found": bool(exp1312),
            "status": exp1312.get("status"),
            "models_used": exp1312.get("models_used") or [],
            "prompt_shapes": sorted({str(row.get("path")) for row in attempts if row.get("path")}),
            "token_budgets": ["post_hoc_from_exp1311_outputs"],
            "stop_settings": ["not_recorded_in_artifact"],
            "observed_raw_output_lengths": [],
            "observed_prompt_chars": [row.get("prompt_chars") for row in attempts],
            "certificate_parse_rate": exp1312.get("certificate_parse_rate"),
        },
    }


def _extract_logprobs(raw_response: dict[str, Any] | None) -> dict[str, Any]:
    if not raw_response:
        return {"token_logprobs": [], "top_logprobs": []}
    choices = raw_response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {"token_logprobs": [], "top_logprobs": []}
    logprobs = choices[0].get("logprobs") or {}
    if not isinstance(logprobs, dict):
        return {"token_logprobs": [], "top_logprobs": []}
    return {
        "token_logprobs": [
            float(value)
            for value in logprobs.get("token_logprobs") or []
            if isinstance(value, int | float)
        ],
        "top_logprobs": [
            dict(top)
            for top in logprobs.get("top_logprobs") or []
            if isinstance(top, dict) and top
        ],
    }


def _entropy_by_position(top_logprobs: Sequence[dict[str, float]]) -> list[float]:
    entropies: list[float] = []
    for top_k in top_logprobs:
        values = [float(value) for value in top_k.values()]
        max_logprob = max(values)
        weights = [math.exp(value - max_logprob) for value in values]
        normalizer = sum(weights)
        probs = [weight / normalizer for weight in weights]
        entropies.append(-sum(prob * math.log(prob) for prob in probs if prob > 0.0))
    return entropies


def _row_from_generation(
    spec: dict[str, Any],
    variant: PromptVariant,
    raw: RawProbeGeneration,
    generation_source: str,
) -> dict[str, Any]:
    logprobs = _extract_logprobs(raw.raw_response)
    topk_entropy = _entropy_by_position(logprobs["top_logprobs"])
    text = raw.text or ""
    empty_or_one = raw.token_count <= 1 or not text.strip()
    return {
        "model_name": spec.get("name"),
        "hf_id": spec.get("hf_id"),
        "gpu": spec.get("gpu"),
        "variant": variant.name,
        "prompt_shape": variant.prompt_shape,
        "prompt_chars": len(variant.prompt),
        "max_tokens": variant.max_tokens,
        "stop": variant.stop,
        "grammar": variant.grammar,
        "use_chat_template": variant.use_chat_template,
        "used_chat_template": raw.used_chat_template,
        "raw_output": text,
        "raw_output_length": len(text),
        "token_count": raw.token_count,
        "empty_or_one_token": empty_or_one,
        "certificate_skeleton_available": (
            _certificate_skeleton_available(text) if variant.certificate_shaped else None
        ),
        "elapsed_seconds": round(raw.elapsed_seconds, 6),
        "error": raw.error,
        "logprob_error": raw.logprob_error,
        "token_logprobs": logprobs["token_logprobs"],
        "topk_entropy_by_position": [round(value, 6) for value in topk_entropy],
        "generation_source": generation_source,
    }


def _collect_with_generation_fn(
    specs: list[dict[str, Any]],
    variants: list[PromptVariant],
    generation_fn: GenerationFn,
    generation_source: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for variant in variants:
            try:
                raw = generation_fn(spec, variant)
            except Exception as exc:  # pragma: no cover - defensive around injected callables.
                raw = RawProbeGeneration("", 0, error=f"{type(exc).__name__}: {exc}")
            rows.append(_row_from_generation(spec, variant, raw, generation_source))
    return rows


def _import_llama_class() -> tuple[bool, type[Any] | None, str | None]:  # pragma: no cover
    try:
        from llama_cpp import Llama  # noqa: PLC0415
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, Llama, None


def _close_llama(llm: Any) -> None:  # pragma: no cover
    close = getattr(llm, "close", None)
    if callable(close):
        close()
    gc.collect()


def _call_llama_with_optional_logprobs(
    llm: Any, variant: PromptVariant
) -> tuple[dict[str, Any], str | None, bool]:  # pragma: no cover
    kwargs = {
        "max_tokens": variant.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stop": variant.stop,
    }
    if variant.use_chat_template and callable(getattr(llm, "create_chat_completion", None)):
        messages = [{"role": "user", "content": variant.prompt}]
        try:
            return (
                llm.create_chat_completion(messages=messages, logprobs=5, **kwargs),
                None,
                True,
            )
        except TypeError as exc:
            return (
                llm.create_chat_completion(messages=messages, **kwargs),
                f"logprobs_unavailable: {exc}",
                True,
            )
    if variant.use_chat_template:
        return {}, "create_chat_completion_missing", False
    try:
        return llm(variant.prompt, echo=False, logprobs=5, **kwargs), None, False
    except TypeError as exc:
        return llm(variant.prompt, echo=False, **kwargs), f"logprobs_unavailable: {exc}", False


def _generate_with_llama(llm: Any, variant: PromptVariant) -> RawProbeGeneration:  # pragma: no cover
    started = time.monotonic()
    result, logprob_error, used_chat_template = _call_llama_with_optional_logprobs(llm, variant)
    elapsed = max(time.monotonic() - started, 0.0)
    text = _completion_text(result)
    token_count = _completion_token_count(result, text, llm)
    return RawProbeGeneration(
        text=text,
        token_count=token_count,
        elapsed_seconds=elapsed,
        raw_response=result,
        logprob_error=logprob_error,
        used_chat_template=used_chat_template,
    )


def _collect_with_llama(
    specs: list[dict[str, Any]], variants: list[PromptVariant], *, llama_class: type[Any]
) -> list[dict[str, Any]]:  # pragma: no cover
    rows: list[dict[str, Any]] = []
    for spec in specs:
        llm: Any | None = None
        try:
            llm = llama_class(
                model_path=spec["model_path"],
                n_gpu_layers=-1,
                n_ctx=1024,
                seed=1323,
                main_gpu=int(spec["gpu"]),
                logits_all=True,
                verbose=False,
            )
            for variant in variants:
                raw = _generate_with_llama(llm, variant)
                rows.append(_row_from_generation(spec, variant, raw, "live_sota_llamacpp"))
        except Exception as exc:
            for variant in variants:
                raw = RawProbeGeneration("", 0, error=f"{type(exc).__name__}: {exc}")
                rows.append(_row_from_generation(spec, variant, raw, "live_sota_llamacpp"))
        finally:
            if llm is not None:
                _close_llama(llm)
    return rows


def _token_health_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    token_logprobs: list[float] = []
    entropies: list[float] = []
    logprob_errors = [row.get("logprob_error") for row in rows if row.get("logprob_error")]
    for row in rows:
        token_logprobs.extend(row.get("token_logprobs") or [])
        entropies.extend(row.get("topk_entropy_by_position") or [])

    if entropies:
        mean_entropy = sum(entropies) / len(entropies)
        return {
            "token_logprob_count": len(token_logprobs),
            "topk_position_count": len(entropies),
            "mean_token_logprob": (
                round(sum(token_logprobs) / len(token_logprobs), 6) if token_logprobs else None
            ),
            "entropy_production_rate": round(mean_entropy, 6),
            "missing_api_reason": None,
        }

    return {
        "token_logprob_count": len(token_logprobs),
        "topk_position_count": 0,
        "mean_token_logprob": (
            round(sum(token_logprobs) / len(token_logprobs), 6) if token_logprobs else None
        ),
        "entropy_production_rate": None,
        "missing_api_reason": (
            str(logprob_errors[0]) if logprob_errors else "llama_cpp_response_missing_top_logprobs"
        ),
    }


def _certificate_parse_delta(
    rows: Sequence[dict[str, Any]], prior_context: dict[str, Any]
) -> dict[str, Any]:
    cert_rows = [row for row in rows if row.get("variant") == "certificate_shaped_prompt"]
    skeleton_rate = _rate([bool(row.get("certificate_skeleton_available")) for row in cert_rows])
    baseline = prior_context.get("exp1312", {}).get("certificate_parse_rate")
    delta = round(skeleton_rate - float(baseline), 6) if isinstance(baseline, int | float) else None
    return {
        "baseline_exp1312_certificate_parse_rate": baseline,
        "probe_gate_certificate_skeleton_rate": skeleton_rate,
        "delta_proxy": delta,
        "measurement_note": "proxy_only_token_health_diagnostic_did_not_rerun_full_dccd_gbnf_parser",
    }


def _min_tokens_recovered(rows: Sequence[dict[str, Any]]) -> bool:
    return any(
        row.get("variant") == "certificate_shaped_prompt"
        and not row.get("error")
        and int(row.get("token_count") or 0) > 1
        and int(row.get("raw_output_length") or 0) > 1
        for row in rows
    )


def _recommended_settings(rows: Sequence[dict[str, Any]], recovered: bool) -> dict[str, Any]:
    if not recovered:
        return {
            "status": "blocked_until_certificate_shaped_prompt_emits_multi_token_output",
            "avoid_stop_strings": ["\n"],
            "grammar": "disable_until_min_token_health_passes",
        }
    cert_rows = [
        row
        for row in rows
        if row.get("variant") == "certificate_shaped_prompt"
        and not row.get("error")
        and int(row.get("token_count") or 0) > 1
    ]
    best = max(cert_rows, key=lambda row: int(row.get("token_count") or 0))
    return {
        "prompt_variant": best["variant"],
        "max_tokens": max(int(best["max_tokens"]), 96),
        "temperature": 0.0,
        "top_p": 1.0,
        "stop": best["stop"],
        "avoid_stop_strings": ["\n"],
        "grammar": "none_for_initial_health_gate_then_reenable_bounded_certificate_schema",
        "chat_template": bool(best["used_chat_template"]),
    }


def _generation_settings(variants: Sequence[PromptVariant]) -> dict[str, Any]:
    return {
        "seed": 1323,
        "temperature": 0.0,
        "top_p": 1.0,
        "n_ctx": 1024,
        "n_gpu_layers": -1,
        "logprobs_requested": 5,
        "variants": {
            variant.name: {
                "prompt_shape": variant.prompt_shape,
                "max_tokens": variant.max_tokens,
                "stop": variant.stop,
                "use_chat_template": variant.use_chat_template,
                "grammar": variant.grammar,
            }
            for variant in variants
        },
    }


def _base_artifact(*, project_root: Path, run_date: str) -> dict[str, Any]:
    variants = build_prompt_variants()
    return {
        "artifact": "experiment_1323_sota_gguf_token_health_prompt_runtime_diagnostic",
        "schema_version": 1,
        "run_date": run_date,
        "artifact_metadata": {
            "project_root": str(project_root),
            "run_date": run_date,
            "preferred_quant": "Q4_K_M",
            "gpu_indices": [0, 1],
        },
        "status": "complete",
        "models_used": [],
        "resolved_model_specs": [],
        "prompt_variants_tested": [variant.name for variant in variants],
        "generation_settings": _generation_settings(variants),
        "prior_artifact_context": {},
        "per_variant_results": [],
        "empty_or_one_token_rate": 0.0,
        "min_tokens_recovered": False,
        "topk_logprob_available": False,
        "entropy_production_rate_available": False,
        "token_health_summary": {
            "token_logprob_count": 0,
            "topk_position_count": 0,
            "mean_token_logprob": None,
            "entropy_production_rate": None,
            "missing_api_reason": "not_run",
        },
        "certificate_parse_delta_with_probe_gate": {
            "baseline_exp1312_certificate_parse_rate": None,
            "probe_gate_certificate_skeleton_rate": 0.0,
            "delta_proxy": None,
            "measurement_note": "not_run",
        },
        "recommended_certificate_runtime_settings": {},
        "headline_result_allowed": False,
        "honest_verdict": "not_run",
    }


def _blocked_artifact(
    artifact: dict[str, Any], *, reason: str, verdict: str, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    artifact.update(
        {
            "status": "blocked",
            "blocked_reason": reason,
            "headline_result_allowed": False,
            "honest_verdict": verdict,
        }
    )
    artifact["recommended_certificate_runtime_settings"] = {
        "status": "blocked",
        "blocked_reason": reason,
    }
    if extra:
        artifact.update(extra)
    return artifact


def build_token_health_artifact(
    *,
    project_root: str | Path,
    run_date: str,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    llama_importer: LlamaImporter = _import_llama_class,
    generation_fn: GenerationFn | None = None,
    generation_source: str = "injected",
) -> dict[str, Any]:
    """Build the Exp 1323 artifact, using live SOTA generation only when requested."""
    root = Path(project_root)
    artifact = _base_artifact(project_root=root, run_date=run_date)
    artifact["prior_artifact_context"] = _summarize_prior_artifacts(root)
    variants = build_prompt_variants()

    try:
        raw_specs = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    except Exception as exc:  # pragma: no cover - defensive around resolver host state.
        return _blocked_artifact(
            artifact,
            reason="cached_sota_pair_exception",
            verdict="blocked_cached_sota_pair_exception",
            extra={"cached_sota_pair_error": f"{type(exc).__name__}: {exc}"},
        )

    specs = _resolved_specs(raw_specs)
    artifact["resolved_model_specs"] = specs
    artifact["models_used"] = [str(spec["hf_id"]) for spec in specs]
    if not specs:
        return _blocked_artifact(
            artifact,
            reason="cached_sota_pair_not_loadable",
            verdict="blocked_cached_sota_pair_not_loadable",
        )

    if generation_fn is None:
        import_ok, llama_class, import_error = llama_importer()
        artifact["llama_cpp_import_ok"] = import_ok
        artifact["llama_cpp_import_error"] = import_error
        if not import_ok or llama_class is None:
            return _blocked_artifact(
                artifact,
                reason="llama_cpp_import_failed",
                verdict="blocked_llama_cpp_import_failed",
                extra={"llama_cpp_import_error": import_error},
            )
        rows = _collect_with_llama(specs, variants, llama_class=llama_class)
    else:
        artifact["llama_cpp_import_ok"] = None
        artifact["llama_cpp_import_error"] = None
        rows = _collect_with_generation_fn(specs, variants, generation_fn, generation_source)

    artifact["per_variant_results"] = rows
    artifact["empty_or_one_token_rate"] = _rate(
        [bool(row.get("empty_or_one_token")) for row in rows]
    )
    recovered = _min_tokens_recovered(rows)
    artifact["min_tokens_recovered"] = recovered
    summary = _token_health_summary(rows)
    artifact["token_health_summary"] = summary
    artifact["topk_logprob_available"] = summary["topk_position_count"] > 0
    artifact["entropy_production_rate_available"] = summary["entropy_production_rate"] is not None
    artifact["certificate_parse_delta_with_probe_gate"] = _certificate_parse_delta(
        rows, artifact["prior_artifact_context"]
    )
    artifact["recommended_certificate_runtime_settings"] = _recommended_settings(rows, recovered)
    generation_errors = sum(1 for row in rows if row.get("error"))
    artifact["headline_result_allowed"] = (
        generation_source == "live_sota_llamacpp"
        and recovered
        and generation_errors == 0
        and all(spec["hf_id"] in MANDATED_HEADLINE_MODEL_IDS for spec in specs)
    )
    artifact["honest_verdict"] = (
        "token_health_recovered_certificate_prompt_multitoken"
        if artifact["headline_result_allowed"]
        else "token_health_diagnostic_no_headline_certificate_recovery"
    )
    return artifact


def run_experiment(
    *,
    project_root: str | Path = ".",
    run_date: str = DEFAULT_RUN_DATE,
    output_path: str | Path = DEFAULT_ARTIFACT_PATH,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    llama_importer: LlamaImporter = _import_llama_class,
    generation_fn: GenerationFn | None = None,
    generation_source: str = "injected",
) -> dict[str, Any]:
    """Write the in-progress marker, then overwrite it with the final artifact."""
    root = Path(project_root)
    out = Path(output_path)
    _write_json(out, {"status": "in_progress", "run_date": run_date})
    artifact = build_token_health_artifact(
        project_root=root,
        run_date=run_date,
        cached_pair_fn=cached_pair_fn,
        llama_importer=llama_importer,
        generation_fn=generation_fn,
        generation_source=generation_source,
    )
    _write_json(out, artifact)
    return artifact


def main() -> None:  # pragma: no cover
    run_experiment(project_root=Path.cwd(), run_date=DEFAULT_RUN_DATE, generation_source="live_sota_llamacpp")


if __name__ == "__main__":  # pragma: no cover
    main()
