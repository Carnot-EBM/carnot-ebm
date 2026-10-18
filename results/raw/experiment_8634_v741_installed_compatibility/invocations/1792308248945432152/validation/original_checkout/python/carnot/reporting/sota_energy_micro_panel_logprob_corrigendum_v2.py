"""Exp 2875 SOTA energy/logprob micro-panel corrigendum v2.

**Researcher summary:**
    Exp 2870 proved that a mandated local GGUF path could be invoked, but the
    resulting rows were empty and lacked token logprobs.  This corrigendum keeps
    the scope deliberately small: it either records a tiny clean diagnostic
    panel with real telemetry, or it writes a terminal blocked artifact that
    says exactly which telemetry gate failed.

**Detailed explanation for engineers:**
    The runner treats Exp 2874 as the runtime gate because that artifact records
    the clean single-model llama.cpp/GPU provenance.  Before collecting panel
    rows, this module checks the selected model path, mandates one of the three
    approved headline GGUF IDs, and runs a tiny telemetry probe.  The panel is
    marked clean only when every row has non-empty text and every row carries
    token logprobs or an explicitly named substitute score such as entropy from
    llama.cpp ``logits_all`` scores.  This prevents the common failure mode
    where text-only generations accidentally become benchmark evidence.

Spec: REQ-INFER-SOTA-016,
      SCENARIO-INFER-SOTA-016-001,
      SCENARIO-INFER-SOTA-016-002
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carnot.metrics.spilled_energy import compute_spilled_energy


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]
PanelRunnerFn = Callable[..., list[JsonDict]]
TelemetryProbeFn = Callable[..., JsonDict]

OUTPUT_FILENAME = "experiment_2875_sota_energy_micro_panel_logprob_corrigendum_v2.json"
EXP2874_FILENAME = "experiment_2874_sota_runtime_clean_corrigendum_v4.json"
RUN_DATE = "20260522"
RANDOM_SEED = 2875
DEFAULT_N_PROMPTS = 6
DEFAULT_MANIFEST_PATHS: tuple[Path, ...] = (Path("data/eval_manifests/fever_20260522.jsonl"),)
MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MODEL_NAMES: dict[str, str] = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "Qwen3.6-35B-A3B",
    "unsloth/gemma-4-31B-it-GGUF": "Gemma4-31B-it",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "Gemma4-26B-A4B-it",
}
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "micro_panel_clean",
    "blocked_reason",
    "model_specs",
    "selected_model_hf_id",
    "selected_model_path",
    "preconditions_checked",
    "n_prompts",
    "n_nonempty_responses",
    "logprobs_available",
    "substitute_telemetry_used",
    "prompt_rows",
    "auroc_if_computable",
    "benchmark_claim_made",
    "random_seed",
    "tests_run",
    "field_principles",
    "run_date",
    "duration_s",
)


@dataclass(frozen=True)
class MicroPanelExample:
    """One local labeled row converted into a short, bounded prompt."""

    example_id: str
    dataset: str
    claim: str
    context: str
    expected_answer: str
    source_path: str

    def prompt_text(self) -> str:
        """Return a compact prompt whose answer space is exactly three labels."""
        context = " ".join(self.context.split())[:700]
        return (
            "Use the evidence to classify the claim.\n"
            "Reply with exactly one word: SUPPORTS, REFUTES, or UNKNOWN.\n\n"
            f"Evidence: {context}\n"
            f"Claim: {self.claim}\n"
            "One-word answer:"
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 2875 corrigendum."""

    repo_root: Path = Path(__file__).resolve().parents[3]
    output_path: Path | None = None
    exp2874_path: Path = Path("results") / EXP2874_FILENAME
    run_date: str = RUN_DATE
    n_prompts: int = DEFAULT_N_PROMPTS
    random_seed: int = RANDOM_SEED
    started_at: float | None = None
    clock: ClockFn = time.perf_counter
    manifest_paths: tuple[Path, ...] = DEFAULT_MANIFEST_PATHS
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        return self.repo_root / "results" / OUTPUT_FILENAME

    def resolved_exp2874_path(self) -> Path:
        return self.exp2874_path if self.exp2874_path.is_absolute() else self.repo_root / self.exp2874_path


def normalize_expected_answer(label_text: str) -> str:
    """Map existing FEVER-style labels into the panel answer vocabulary."""
    normalized = str(label_text).strip().upper()
    if normalized in {"SUPPORTS", "SUPPORTED", "SUPPORT"}:
        return "SUPPORTS"
    if normalized in {"REFUTES", "REFUTED", "REFUTE"}:
        return "REFUTES"
    if normalized in {"NOT ENOUGH INFO", "NEI", "UNKNOWN", "UNVERIFIABLE"}:
        return "UNKNOWN"
    return normalized


def classify_response(response_text: str) -> str | None:
    """Return the first supported label in a bounded model response."""
    text = str(response_text).strip().upper()
    if not text:
        return None
    first = re.split(r"[^A-Z]+", text, maxsplit=1)[0]
    if first in {"SUPPORTS", "REFUTES", "UNKNOWN"}:
        return first
    if text.startswith("NOT ENOUGH INFO"):
        return "UNKNOWN"
    return None


def compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    """Compute tie-aware AUROC, returning ``None`` when one class is absent."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not labels:
        return None
    positives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 0]
    if not positives or not negatives:
        return None
    if not all(math.isfinite(float(score)) for score in [*positives, *negatives]):
        raise ValueError("scores must be finite")
    wins = 0.0
    for positive in positives:
        wins += sum(1.0 for negative in negatives if positive > negative)
        wins += sum(0.5 for negative in negatives if positive == negative)
    return wins / (len(positives) * len(negatives))


def extract_completion_telemetry(raw_response: Mapping[str, Any] | None) -> JsonDict:
    """Normalize OpenAI-style llama.cpp completion telemetry."""
    choices = raw_response.get("choices") if isinstance(raw_response, Mapping) else None
    first = choices[0] if choices and isinstance(choices[0], Mapping) else {}
    usage = raw_response.get("usage", {}) if isinstance(raw_response, Mapping) else {}
    logprobs = first.get("logprobs") if isinstance(first, Mapping) else None
    response_text = _completion_text(raw_response)
    if not isinstance(logprobs, Mapping):
        return {
            "response_text": response_text,
            "completion_tokens": usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else 0,
            "tokens": [],
            "token_logprobs": [],
            "top_logprobs": [],
        }
    top_logprobs: list[dict[str, float]] = []
    for top in logprobs.get("top_logprobs") or []:
        if isinstance(top, Mapping):
            converted = {
                str(key): coerced
                for key, value in top.items()
                if (coerced := _coerce_float(value)) is not None
            }
            if converted:
                top_logprobs.append(converted)
    return {
        "response_text": response_text,
        "completion_tokens": usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else 0,
        "tokens": [str(token) for token in logprobs.get("tokens") or [] if token is not None],
        "token_logprobs": [
            coerced
            for value in logprobs.get("token_logprobs") or []
            if (coerced := _coerce_float(value)) is not None
        ],
        "top_logprobs": top_logprobs,
    }


def first_token_confidence(tokens: Any, token_logprobs: Sequence[float]) -> float | None:
    """Convert the first non-blank token logprob into probability space."""
    if not token_logprobs:
        return None
    if isinstance(tokens, list):
        for index, token in enumerate(tokens[: len(token_logprobs)]):
            if str(token).strip():
                return _confidence_from_logprob(token_logprobs[index])
    return _confidence_from_logprob(token_logprobs[0])


def logits_entropy_score(logits: Sequence[float]) -> float | None:
    """Return natural-log entropy for a finite logits vector."""
    numeric = [float(value) for value in logits if not isinstance(value, bool) and math.isfinite(float(value))]
    if not numeric:
        return None
    maximum = max(numeric)
    weights = [math.exp(value - maximum) for value in numeric]
    total = sum(weights)
    probabilities = [weight / total for weight in weights if weight > 0.0]
    return -sum(probability * math.log(probability) for probability in probabilities)


def select_micro_panel(
    repo_root: Path,
    *,
    n_prompts: int = DEFAULT_N_PROMPTS,
    manifest_paths: Sequence[Path] = DEFAULT_MANIFEST_PATHS,
) -> list[MicroPanelExample]:
    """Select a deterministic label-balanced tiny panel from local manifests."""
    buckets: dict[str, list[MicroPanelExample]] = {"SUPPORTS": [], "REFUTES": [], "UNKNOWN": []}
    for relative in manifest_paths:
        path = repo_root / relative
        if not path.is_file():
            continue
        for raw in _read_jsonl(path):
            expected = normalize_expected_answer(str(raw.get("label_text", "")))
            claim = str(raw.get("claim") or raw.get("question") or "").strip()
            context = str(raw.get("prompt") or raw.get("context") or "").strip()
            if expected in buckets and claim and context:
                buckets[expected].append(
                    MicroPanelExample(
                        example_id=str(raw.get("stable_id") or f"{path.name}:{sum(len(v) for v in buckets.values())}"),
                        dataset=str(raw.get("dataset") or "FEVER"),
                        claim=claim,
                        context=context,
                        expected_answer=expected,
                        source_path=str(relative),
                    )
                )
    selected: list[MicroPanelExample] = []
    max_bucket = max((len(values) for values in buckets.values()), default=0)
    for index in range(max_bucket):
        for label in ("SUPPORTS", "REFUTES", "UNKNOWN"):
            if index < len(buckets[label]):
                selected.append(buckets[label][index])
                if len(selected) >= n_prompts:
                    return selected
    return selected


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    telemetry_probe_fn: TelemetryProbeFn = None,
    panel_runner_fn: PanelRunnerFn = None,
    write: bool = True,
) -> JsonDict:
    """Build the Exp 2875 artifact and optionally write it to disk."""
    active = config or ExperimentConfig()
    started_at = active.start_time()
    exp2874 = _read_json(active.resolved_exp2874_path())
    selected_hf_id = str(exp2874.get("selected_model_hf_id") or "")
    selected_path = str(exp2874.get("selected_model_path") or "")
    model_specs = _model_specs_from_exp2874(exp2874)
    preconditions = _runtime_preconditions(exp2874, selected_hf_id, selected_path)
    blocked = _first_blocked_reason(preconditions)
    model_spec = {
        "name": MODEL_NAMES.get(selected_hf_id, selected_hf_id),
        "hf_id": selected_hf_id,
        "model_path": selected_path,
        "gpu": 0,
    }

    if blocked:
        artifact = _base_artifact(
            active,
            started_at,
            honest_verdict=blocked,
            blocked_reason=blocked,
            model_specs=model_specs,
            selected_model_hf_id=selected_hf_id,
            selected_model_path=selected_path,
            preconditions=preconditions,
        )
        return _maybe_write(active.resolved_output_path(), artifact, write)

    probe_fn = telemetry_probe_fn or _run_live_telemetry_probe
    probe = probe_fn(
        model_spec=model_spec,
        prompt="Reply with exactly one word: SUPPORTS.",
        random_seed=active.random_seed,
    )
    probe_ok = _row_has_logprobs(probe) or _row_has_substitute(probe)
    preconditions.append(
        {
            "resource": "llama_cpp_logprob_or_substitute_telemetry",
            "available": probe_ok,
            "detail": str(probe.get("telemetry_source") or probe.get("blocked_reason") or probe.get("error") or ""),
            "response_nonempty": bool(str(probe.get("response_text") or "").strip()),
        }
    )
    if not probe_ok:
        reason = str(probe.get("blocked_reason") or "blocked_logprobs_unavailable")
        artifact = _base_artifact(
            active,
            started_at,
            honest_verdict=reason,
            blocked_reason=reason,
            model_specs=model_specs,
            selected_model_hf_id=selected_hf_id,
            selected_model_path=selected_path,
            preconditions=preconditions,
        )
        return _maybe_write(active.resolved_output_path(), artifact, write)

    examples = select_micro_panel(
        active.repo_root,
        n_prompts=active.n_prompts,
        manifest_paths=active.manifest_paths,
    )
    preconditions.append(
        {
            "resource": "fixed_labeled_micro_panel",
            "available": bool(examples),
            "detail": f"selected={len(examples)} requested={active.n_prompts}",
        }
    )
    if not examples:
        reason = "blocked_insufficient_micro_panel_rows"
        artifact = _base_artifact(
            active,
            started_at,
            honest_verdict=reason,
            blocked_reason=reason,
            model_specs=model_specs,
            selected_model_hf_id=selected_hf_id,
            selected_model_path=selected_path,
            preconditions=preconditions,
        )
        return _maybe_write(active.resolved_output_path(), artifact, write)

    runner = panel_runner_fn or _run_live_panel
    generated = runner(model_spec=model_spec, examples=examples, random_seed=active.random_seed)
    prompt_rows = score_prompt_rows(examples, generated, selected_model_hf_id=selected_hf_id)
    artifact = _completed_artifact(
        active,
        started_at,
        model_specs=model_specs,
        selected_model_hf_id=selected_hf_id,
        selected_model_path=selected_path,
        preconditions=preconditions,
        prompt_rows=prompt_rows,
    )
    return _maybe_write(active.resolved_output_path(), artifact, write)


def score_prompt_rows(
    examples: Sequence[MicroPanelExample],
    generated_rows: Sequence[Mapping[str, Any]],
    *,
    selected_model_hf_id: str,
) -> list[JsonDict]:
    """Attach labels and telemetry-derived diagnostic scores to raw rows."""
    by_id = {str(row.get("example_id")): row for row in generated_rows}
    scored: list[JsonDict] = []
    for example in examples:
        raw = by_id.get(example.example_id, {})
        response = str(raw.get("response_text") or "").strip()
        predicted = classify_response(response)
        is_correct = predicted == example.expected_answer
        token_logprobs = _numeric_values(raw.get("token_logprobs"))
        substitute_score = _finite_or_none(raw.get("substitute_score"))
        uses_substitute = bool(raw.get("substitute_telemetry_used")) and substitute_score is not None and not token_logprobs
        spilled = compute_spilled_energy(token_logprobs) if token_logprobs else None
        telemetry_score = spilled if token_logprobs else substitute_score if uses_substitute else None
        model_hf_id = str(raw.get("model_hf_id") or selected_model_hf_id)
        scored.append(
            {
                "example_id": example.example_id,
                "dataset": example.dataset,
                "prompt_text": example.prompt_text(),
                "expected_answer": example.expected_answer,
                "response_text": response,
                "response_nonempty": bool(response),
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "hallucination_label": 0 if is_correct else 1,
                "model_hf_id": model_hf_id,
                "model_path": raw.get("model_path"),
                "headline_mandated_model": model_hf_id in MANDATED_MODEL_IDS,
                "tokens_generated": int(raw.get("tokens_generated") or 0),
                "tokens": list(raw.get("tokens") or []),
                "token_logprobs": token_logprobs,
                "top_logprobs": list(raw.get("top_logprobs") or []),
                "logprobs_available": bool(token_logprobs),
                "first_token_confidence": first_token_confidence(raw.get("tokens"), token_logprobs),
                "spilled_energy_score": spilled,
                "substitute_telemetry_used": uses_substitute,
                "substitute_telemetry_source": raw.get("substitute_telemetry_source"),
                "substitute_score": substitute_score if uses_substitute else None,
                "telemetry_score": telemetry_score,
                "telemetry_sufficient": (bool(token_logprobs) or uses_substitute) and model_hf_id in MANDATED_MODEL_IDS,
                "generation_error": raw.get("error"),
                "duration_s": raw.get("duration_s"),
                "source_path": example.source_path,
            }
        )
    return scored


def _completed_artifact(
    config: ExperimentConfig,
    started_at: float,
    *,
    model_specs: Sequence[Mapping[str, Any]],
    selected_model_hf_id: str,
    selected_model_path: str,
    preconditions: Sequence[Mapping[str, Any]],
    prompt_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    rows = [dict(row) for row in prompt_rows]
    nonempty_count = sum(1 for row in rows if row.get("response_nonempty"))
    all_nonempty = bool(rows) and nonempty_count == len(rows)
    all_telemetry = bool(rows) and all(bool(row.get("telemetry_sufficient")) for row in rows)
    logprobs_available = bool(rows) and all(bool(row.get("logprobs_available")) for row in rows)
    substitute_used = any(bool(row.get("substitute_telemetry_used")) for row in rows)
    clean = all_nonempty and all_telemetry
    if clean:
        blocked_reason = ""
    elif not all_nonempty:
        blocked_reason = "blocked_empty_responses"
    else:
        blocked_reason = "blocked_logprobs_unavailable"
    pairs = [
        (int(row["hallucination_label"]), float(row["telemetry_score"]))
        for row in rows
        if row.get("telemetry_score") is not None
    ]
    return _base_artifact(
        config,
        started_at,
        honest_verdict="micro_panel_clean_no_benchmark_claim" if clean else blocked_reason,
        blocked_reason=blocked_reason,
        model_specs=model_specs,
        selected_model_hf_id=selected_model_hf_id,
        selected_model_path=selected_model_path,
        preconditions=preconditions,
        prompt_rows=rows,
        n_nonempty_responses=nonempty_count,
        logprobs_available=logprobs_available,
        substitute_telemetry_used=substitute_used,
        auroc_if_computable=_auroc_from_pairs(pairs),
        micro_panel_clean=clean,
    )


def _base_artifact(
    config: ExperimentConfig,
    started_at: float,
    *,
    honest_verdict: str,
    blocked_reason: str,
    model_specs: Sequence[Mapping[str, Any]],
    selected_model_hf_id: str,
    selected_model_path: str,
    preconditions: Sequence[Mapping[str, Any]],
    prompt_rows: Sequence[Mapping[str, Any]] = (),
    n_nonempty_responses: int = 0,
    logprobs_available: bool = False,
    substitute_telemetry_used: bool = False,
    auroc_if_computable: float | None = None,
    micro_panel_clean: bool = False,
) -> JsonDict:
    duration = config.clock() - started_at
    rows = [dict(row) for row in prompt_rows]
    return {
        "honest_verdict": honest_verdict,
        "micro_panel_clean": micro_panel_clean,
        "blocked_reason": blocked_reason,
        "model_specs": [dict(spec) for spec in model_specs],
        "selected_model_hf_id": selected_model_hf_id,
        "selected_model_path": selected_model_path,
        "preconditions_checked": [dict(row) for row in preconditions],
        "n_prompts": len(rows),
        "n_nonempty_responses": int(n_nonempty_responses),
        "logprobs_available": bool(logprobs_available),
        "substitute_telemetry_used": bool(substitute_telemetry_used),
        "prompt_rows": rows,
        "auroc_if_computable": auroc_if_computable,
        "benchmark_claim_made": False,
        "random_seed": config.random_seed,
        "tests_run": list(config.tests_run),
        "field_principles": _field_principles(),
        "run_date": config.run_date,
        "duration_s": round(duration, 6),
        "artifact": "experiment_2875_sota_energy_micro_panel_logprob_corrigendum_v2",
        "schema_version": 1,
    }


def _runtime_preconditions(
    exp2874: Mapping[str, Any],
    selected_hf_id: str,
    selected_path: str,
) -> list[JsonDict]:
    clean = bool(exp2874.get("sota_runtime_clean"))
    mandated = selected_hf_id in MANDATED_MODEL_IDS
    path_exists = bool(selected_path) and Path(selected_path).is_file()
    gpu = bool(exp2874.get("llama_cpp_gpu_offload_verified"))
    return [
        {
            "resource": "exp2874_sota_runtime_clean",
            "available": clean,
            "detail": "sota_runtime_clean=true" if clean else str(exp2874.get("honest_verdict") or ""),
            "blocked_reason": "" if clean else "blocked_exp2874_sota_runtime_not_clean",
        },
        {
            "resource": "selected_mandated_sota_model",
            "available": mandated,
            "detail": selected_hf_id,
            "blocked_reason": "" if mandated else "blocked_selected_model_not_mandated",
        },
        {
            "resource": "selected_model_path_exists",
            "available": path_exists,
            "detail": selected_path,
            "blocked_reason": "" if path_exists else "blocked_selected_model_path_missing",
        },
        {
            "resource": "llama_cpp_gpu_offload",
            "available": gpu,
            "detail": "llama_cpp_gpu_offload_verified=true" if gpu else "llama_cpp_gpu_offload_verified=false",
            "blocked_reason": "" if gpu else "blocked_llama_cpp_gpu_offload",
        },
    ]


def _first_blocked_reason(preconditions: Sequence[Mapping[str, Any]]) -> str:
    for row in preconditions:
        if not row.get("available"):
            return str(row.get("blocked_reason") or "blocked_precondition_unavailable")
    return ""


def _model_specs_from_exp2874(exp2874: Mapping[str, Any]) -> list[JsonDict]:
    selected_hf_id = str(exp2874.get("selected_model_hf_id") or "")
    selected_path = str(exp2874.get("selected_model_path") or "")
    recorded_paths = {
        str(spec.get("hf_id")): str(spec.get("model_path") or "")
        for spec in exp2874.get("model_specs", [])
        if isinstance(spec, Mapping)
    }
    specs: list[JsonDict] = []
    for hf_id in MANDATED_MODEL_IDS:
        path = selected_path if hf_id == selected_hf_id else recorded_paths.get(hf_id, "")
        specs.append(
            {
                "name": MODEL_NAMES[hf_id],
                "hf_id": hf_id,
                "headline_candidate": True,
                "selected": hf_id == selected_hf_id,
                "model_path": path,
                "legacy_smoke_only": False,
            }
        )
    return specs


def _field_principles() -> JsonDict:
    return {
        "claim_boundary": "Tiny diagnostic micro-panel only; benchmark_claim_made is always false.",
        "runtime_gate": "Exp 2874 clean runtime, selected GGUF path, and llama.cpp GPU offload are preconditions.",
        "model_selection": "Headline rows must use only the three mandated SOTA GGUF IDs.",
        "telemetry_gate": "micro_panel_clean requires non-empty rows plus token logprobs or a documented substitute score on every row.",
        "label_boundary": "Scores compare only against existing local FEVER/FoVer/HaluEval-style labels.",
        "blocked_behavior": "Missing telemetry produces blocked_logprobs_unavailable instead of fabricated scores.",
    }


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _maybe_write(path: Path, artifact: JsonDict, write: bool) -> JsonDict:
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _completion_text(result: Mapping[str, Any] | None) -> str:
    if not isinstance(result, Mapping):
        return ""
    choices = result.get("choices") or []
    first = choices[0] if choices and isinstance(choices[0], Mapping) else {}
    if "text" in first:
        return str(first.get("text") or "")
    message = first.get("message")
    return str(message.get("content") or "") if isinstance(message, Mapping) else ""


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _numeric_values(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    return [coerced for value in values if (coerced := _coerce_float(value)) is not None]


def _finite_or_none(value: Any) -> float | None:
    return _coerce_float(value)


def _confidence_from_logprob(logprob: float) -> float:
    return max(0.0, min(1.0, math.exp(min(0.0, float(logprob)))))


def _row_has_logprobs(row: Mapping[str, Any]) -> bool:
    return bool(_numeric_values(row.get("token_logprobs"))) or bool(row.get("logprobs_available") and _numeric_values(row.get("token_logprobs")))


def _row_has_substitute(row: Mapping[str, Any]) -> bool:
    return bool(row.get("substitute_telemetry_used")) and _finite_or_none(row.get("substitute_score")) is not None


def _auroc_from_pairs(pairs: Sequence[tuple[int, float]]) -> float | None:
    if not pairs:
        return None
    return compute_auroc([label for label, _score in pairs], [score for _label, score in pairs])


def _run_live_telemetry_probe(  # pragma: no cover - live GPU path exercised by artifact run.
    *,
    model_spec: Mapping[str, Any],
    prompt: str,
    random_seed: int,
) -> JsonDict:
    from llama_cpp import Llama  # type: ignore[import]  # noqa: PLC0415

    llm = Llama(
        model_path=str(model_spec["model_path"]),
        n_ctx=512,
        n_batch=32,
        n_gpu_layers=-1,
        main_gpu=int(model_spec.get("gpu") or 0),
        logits_all=True,
        verbose=False,
    )
    try:
        return _generate_one(llm, model_spec, "telemetry-preflight", prompt, random_seed)
    finally:
        _close_llama(llm)


def _run_live_panel(  # pragma: no cover - live GPU path exercised by artifact run.
    *,
    model_spec: Mapping[str, Any],
    examples: Sequence[MicroPanelExample],
    random_seed: int,
) -> list[JsonDict]:
    from llama_cpp import Llama  # type: ignore[import]  # noqa: PLC0415

    llm = Llama(
        model_path=str(model_spec["model_path"]),
        n_ctx=1024,
        n_batch=64,
        n_gpu_layers=-1,
        main_gpu=int(model_spec.get("gpu") or 0),
        logits_all=True,
        verbose=False,
    )
    try:
        return [
            _generate_one(
                llm,
                model_spec,
                example.example_id,
                example.prompt_text(),
                random_seed + index,
            )
            for index, example in enumerate(examples)
        ]
    finally:
        _close_llama(llm)


def _generate_one(  # pragma: no cover - live GPU path exercised by artifact run.
    llm: Any,
    model_spec: Mapping[str, Any],
    example_id: str,
    prompt: str,
    seed: int,
) -> JsonDict:
    started = time.perf_counter()
    logprob_error = None
    try:
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=12,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            logprobs=5,
            stop=["</s>", "<eos>"],
        )
    except TypeError as exc:
        logprob_error = f"logprobs_unavailable: {exc}"
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=12,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            stop=["</s>", "<eos>"],
        )
    telemetry = extract_completion_telemetry(result if isinstance(result, Mapping) else None)
    substitute_score = _substitute_score_from_llm(llm)
    token_logprobs = telemetry["token_logprobs"]
    return {
        "example_id": example_id,
        "model_hf_id": model_spec.get("hf_id"),
        "model_path": model_spec.get("model_path"),
        "response_text": str(telemetry["response_text"]).strip(),
        "tokens_generated": int(telemetry["completion_tokens"] or len(telemetry["tokens"])),
        "duration_s": round(time.perf_counter() - started, 6),
        "tokens": telemetry["tokens"],
        "token_logprobs": token_logprobs,
        "top_logprobs": telemetry["top_logprobs"],
        "logprobs_available": bool(token_logprobs),
        "substitute_telemetry_used": substitute_score is not None and not token_logprobs,
        "substitute_telemetry_source": "llama_cpp_logits_entropy" if substitute_score is not None and not token_logprobs else None,
        "substitute_score": substitute_score,
        "telemetry_source": "llama_cpp_token_logprobs" if token_logprobs else "llama_cpp_logits_entropy" if substitute_score is not None else "llama_cpp_text_only",
        "error": logprob_error,
    }


def _substitute_score_from_llm(llm: Any) -> float | None:  # pragma: no cover - live runtime shape varies.
    for attr in ("scores", "eval_logits"):
        value = getattr(llm, attr, None)
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 2 and int(shape[-1]) > 0:
            row = value[-1]
            return logits_entropy_score([float(item) for item in row])
        if isinstance(value, list) and value and isinstance(value[-1], list):
            return logits_entropy_score(value[-1])
    return None


def _close_llama(llm: Any) -> None:  # pragma: no cover - live cleanup.
    close = getattr(llm, "close", None)
    if callable(close):
        close()


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI glue.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-date", default=RUN_DATE)
    parser.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
    parser.add_argument("--tests-run", action="append", default=[])
    args = parser.parse_args(argv)
    run_experiment(
        ExperimentConfig(
            repo_root=Path.cwd(),
            output_path=args.output,
            run_date=args.run_date,
            n_prompts=args.n_prompts,
            tests_run=args.tests_run,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI glue.
    raise SystemExit(main())
