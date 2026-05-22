"""Exp 2870 live SOTA GGUF cheap-logit micro-panel.

**Researcher summary:**
    This module runs a deliberately tiny live model panel after Exp 2862 has
    already proved the local SOTA GGUF runtime.  The goal is not a benchmark
    result.  It records whether first-token confidence and spilled-energy-style
    logprob signals are available cheaply before any expensive multi-sample
    decoding campaign.

**Detailed explanation for engineers:**
    The runner calls ``cached_sota_pair()`` first, because headline local-model
    work should use the shared SOTA resolver instead of hand-written model
    lists.  On this workstation Exp 2862 may still have only one mandated GGUF
    cached; in that case this module can fall back to Exp 2862's selected
    mandated model path while recording that the two-model pair was unavailable.
    Every metric is computed from generation telemetry returned by llama.cpp.
    If logprobs are missing, the artifact keeps the generated answers but marks
    the confidence/energy AUROCs as blocked rather than inventing proxy scores.

Spec: REQ-INFER-SOTA-014,
      SCENARIO-INFER-SOTA-014-001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from carnot.inference.sota_models import cached_sota_pair
from carnot.metrics.spilled_energy import compute_marginalized_energy, compute_spilled_energy


JsonDict = dict[str, Any]
CachedPairFn = Callable[..., list[dict[str, Any]] | None]
ClockFn = Callable[[], float]

OUTPUT_FILENAME = "experiment_2870_sota_energy_baseline_micro_panel_v1.json"
EXP2862_FILENAME = "experiment_2862_sota_runtime_cache_offload_resolver_v3.json"
RUN_DATE = "20260522"
RANDOM_SEED = 2870
DEFAULT_N_EXAMPLES = 12
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
DEFAULT_MANIFEST_PATHS: tuple[Path, ...] = (
    Path("data/eval_manifests/fever_20260522.jsonl"),
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "micro_panel_ready",
    "live_model_invoked",
    "model_specs",
    "models_used",
    "n_examples",
    "first_token_confidence_available",
    "spilled_energy_available",
    "first_token_confidence_auroc",
    "spilled_energy_auroc",
    "usable_response_count",
    "blocked_metrics",
    "sample_rows",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "field_principles",
    "run_date",
    "duration_s",
)


@dataclass(frozen=True)
class MicroPanelExample:
    """One deterministic local manifest row converted to a bounded prompt."""

    example_id: str
    dataset: str
    claim: str
    context: str
    expected_answer: str
    source_path: str

    def prompt_text(self) -> str:
        """Return a short FEVER-style label prompt for bounded greedy decoding."""
        context = self.context.replace("\n", " ")[:900]
        return (
            "Classify the claim using the context.\n"
            "Answer with exactly one token: SUPPORTS, REFUTES, or UNKNOWN.\n\n"
            f"Context: {context}\n"
            f"Claim: {self.claim}\n"
            "Answer:"
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for the Exp 2870 micro-panel."""

    repo_root: Path = Path(__file__).resolve().parents[3]
    output_path: Path | None = None
    run_date: str = RUN_DATE
    n_examples: int = DEFAULT_N_EXAMPLES
    random_seed: int = RANDOM_SEED
    started_at: float | None = None
    clock: ClockFn = time.perf_counter
    manifest_paths: tuple[Path, ...] = DEFAULT_MANIFEST_PATHS
    exp2862_path: Path = Path("results") / EXP2862_FILENAME

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def resolved_output_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path
        return self.repo_root / "results" / OUTPUT_FILENAME

    def resolved_exp2862_path(self) -> Path:
        return self.repo_root / self.exp2862_path


PanelRunnerFn = Callable[
    ...,
    list[JsonDict],
]


def normalize_expected_answer(label_text: str) -> str:
    """Map local FEVER labels into the constrained answer vocabulary."""
    normalized = str(label_text).strip().upper()
    if normalized in {"SUPPORTS", "SUPPORTED", "SUPPORT"}:
        return "SUPPORTS"
    if normalized in {"REFUTES", "REFUTED", "REFUTE"}:
        return "REFUTES"
    if normalized in {"NOT ENOUGH INFO", "NEI", "UNKNOWN", "UNVERIFIABLE"}:
        return "UNKNOWN"
    return normalized


def classify_response(response_text: str) -> str | None:
    """Return the first supported label found in a bounded model response."""
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
    """Compute tie-aware binary AUROC, returning None when a class is missing."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not labels:
        return None
    label_array = np.asarray(labels, dtype=np.int64)
    score_array = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    positive = score_array[label_array == 1]
    negative = score_array[label_array == 0]
    if positive.size == 0 or negative.size == 0:
        return None
    wins = 0.0
    for score in positive:
        wins += float(np.sum(score > negative))
        wins += 0.5 * float(np.sum(score == negative))
    return float(wins / (positive.size * negative.size))


def select_micro_panel(
    repo_root: Path,
    *,
    n_examples: int = DEFAULT_N_EXAMPLES,
    random_seed: int = RANDOM_SEED,
    manifest_paths: Sequence[Path] = DEFAULT_MANIFEST_PATHS,
) -> list[MicroPanelExample]:
    """Select a deterministic 10-20 row local FEVER-style micro-panel."""
    del random_seed
    buckets: dict[str, list[MicroPanelExample]] = {"SUPPORTS": [], "REFUTES": [], "UNKNOWN": []}
    for rel_path in manifest_paths:
        path = repo_root / rel_path
        if not path.is_file():
            continue
        for raw in _read_jsonl(path):
            expected = normalize_expected_answer(str(raw.get("label_text", "")))
            if expected not in buckets:
                continue
            claim = str(raw.get("claim") or raw.get("question") or "").strip()
            context = str(raw.get("prompt") or raw.get("context") or "").strip()
            if not claim or not context:
                continue
            buckets[expected].append(
                MicroPanelExample(
                    example_id=str(raw.get("stable_id") or f"{path.name}:{sum(len(v) for v in buckets.values())}"),
                    dataset=str(raw.get("dataset") or "FEVER"),
                    claim=claim,
                    context=context,
                    expected_answer=expected,
                    source_path=str(rel_path),
                )
            )

    selected: list[MicroPanelExample] = []
    max_bucket = max((len(rows) for rows in buckets.values()), default=0)
    for index in range(max_bucket):
        for label in ("SUPPORTS", "REFUTES", "UNKNOWN"):
            if index < len(buckets[label]):
                selected.append(buckets[label][index])
                if len(selected) >= n_examples:
                    return selected
    return selected


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    cached_pair_fn: CachedPairFn = cached_sota_pair,
    panel_runner_fn: PanelRunnerFn = None,
    write: bool = True,
) -> JsonDict:
    """Run the Exp 2870 micro-panel and optionally write its JSON artifact."""
    active = config or ExperimentConfig()
    started_at = active.start_time()
    runner = panel_runner_fn or _run_live_panel
    exp2862 = _load_exp2862(active.resolved_exp2862_path())
    exp2862_ready = bool(exp2862.get("sota_runtime_ready_v3"))

    preconditions: list[JsonDict] = [
        {
            "resource": "exp2862_sota_runtime_ready_v3",
            "available": exp2862_ready,
            "detail": active.resolved_exp2862_path().as_posix(),
        }
    ]
    if not exp2862_ready:
        artifact = _base_artifact(
            active,
            started_at,
            "blocked_exp2862_sota_runtime_not_ready",
            preconditions,
            blocked_metrics=["blocked_exp2862_sota_runtime_not_ready"],
        )
        if write:
            _write_artifact(active.resolved_output_path(), artifact)
        return artifact

    model_specs, pair_result = _resolve_model_specs(exp2862, cached_pair_fn)
    preconditions.append(
        {
            "resource": "cached_sota_pair",
            "available": bool(pair_result.get("returned_two_loadable_specs")),
            "detail": pair_result.get("result") if pair_result.get("error") is None else pair_result.get("error"),
        }
    )
    preconditions.append(
        {
            "resource": "mandated_sota_model_specs",
            "available": bool(model_specs),
            "detail": [spec.get("hf_id") for spec in model_specs],
        }
    )
    examples = select_micro_panel(
        active.repo_root,
        n_examples=active.n_examples,
        random_seed=active.random_seed,
        manifest_paths=active.manifest_paths,
    )
    preconditions.append(
        {
            "resource": "deterministic_micro_panel",
            "available": len(examples) >= min(active.n_examples, 10),
            "detail": f"selected={len(examples)} requested={active.n_examples}",
        }
    )
    if not model_specs:
        artifact = _base_artifact(
            active,
            started_at,
            "blocked_no_mandated_sota_model_spec",
            preconditions,
            blocked_metrics=["blocked_no_mandated_sota_model_spec"],
        )
    elif len(examples) < min(active.n_examples, 10):
        artifact = _base_artifact(
            active,
            started_at,
            "blocked_insufficient_micro_panel_rows",
            preconditions,
            blocked_metrics=["blocked_insufficient_micro_panel_rows"],
        )
    else:
        model = model_specs[0]
        generated = runner(
            model_spec=model,
            examples=examples,
            selected_python=str(exp2862.get("selected_python") or ""),
            env=dict(os.environ),
            random_seed=active.random_seed,
        )
        sample_rows = _score_rows(examples, generated)
        artifact = _completed_artifact(
            active,
            started_at,
            preconditions,
            model_specs,
            sample_rows,
            live_model_invoked=True,
        )

    if write:
        _write_artifact(active.resolved_output_path(), artifact)
    return artifact


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_exp2862(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_model_specs(
    exp2862: Mapping[str, Any],
    cached_pair_fn: CachedPairFn,
) -> tuple[list[JsonDict], JsonDict]:
    try:
        pair = cached_pair_fn(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
        pair_result: JsonDict = {
            "called": True,
            "result": pair,
            "error": None,
            "returned_two_loadable_specs": _loadable_pair(pair),
        }
    except Exception as exc:
        pair = None
        pair_result = {
            "called": True,
            "result": None,
            "error": f"{type(exc).__name__}: {exc}",
            "returned_two_loadable_specs": False,
        }
    if _loadable_pair(pair):
        return [
            {
                "name": str(spec.get("name") or MODEL_NAMES.get(str(spec["hf_id"]), spec["hf_id"])),
                "hf_id": str(spec["hf_id"]),
                "gpu": int(spec.get("gpu") or index),
                "model_path": str(spec["model_path"]),
                "source": "cached_sota_pair",
            }
            for index, spec in enumerate(pair or [])
        ], pair_result

    hf_id = str(exp2862.get("selected_model_hf_id") or "")
    model_path = str(exp2862.get("selected_model_path") or "")
    if hf_id in MANDATED_MODEL_IDS and model_path:
        return [
            {
                "name": MODEL_NAMES.get(hf_id, hf_id),
                "hf_id": hf_id,
                "gpu": 0,
                "model_path": model_path,
                "source": "exp2862_selected_model_fallback",
            }
        ], pair_result
    return [], pair_result


def _loadable_pair(pair: Any) -> bool:
    return bool(
        isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(spec, dict) and spec.get("hf_id") in MANDATED_MODEL_IDS and spec.get("model_path") for spec in pair)
    )


def _score_rows(
    examples: Sequence[MicroPanelExample],
    generated_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    by_id = {str(row.get("example_id")): row for row in generated_rows}
    scored: list[JsonDict] = []
    for example in examples:
        row = by_id.get(example.example_id, {})
        response = str(row.get("response_text") or "")
        predicted = classify_response(response)
        is_correct = bool(predicted == example.expected_answer)
        token_logprobs = _numeric_logprobs(row.get("token_logprobs"))
        first_confidence = _first_token_confidence(
            row.get("tokens"),
            token_logprobs,
        )
        spilled = compute_spilled_energy(token_logprobs) if token_logprobs else None
        marginalized = compute_marginalized_energy(token_logprobs) if token_logprobs else None
        scored.append(
            {
                "example_id": example.example_id,
                "dataset": example.dataset,
                "expected_answer": example.expected_answer,
                "predicted_answer": predicted,
                "is_correct": is_correct,
                "hallucination_label": 0 if is_correct else 1,
                "response_text": response,
                "model_hf_id": row.get("model_hf_id"),
                "tokens_generated": int(row.get("tokens_generated") or 0),
                "first_token_confidence": first_confidence,
                "spilled_energy": spilled,
                "marginalized_energy": marginalized,
                "logprobs_available": bool(token_logprobs),
                "logprobs_requested": bool(row.get("logprobs_requested")),
                "generation_error": row.get("error"),
                "source_path": example.source_path,
            }
        )
    return scored


def _numeric_logprobs(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    result: list[float] = []
    for value in values:
        if isinstance(value, int | float) and math.isfinite(float(value)):
            result.append(float(value))
    return result


def _first_token_confidence(tokens: Any, token_logprobs: Sequence[float]) -> float | None:
    if not token_logprobs:
        return None
    if isinstance(tokens, list):
        for index, token in enumerate(tokens[: len(token_logprobs)]):
            if str(token).strip():
                return _confidence_from_logprob(token_logprobs[index])
    return _confidence_from_logprob(token_logprobs[0])


def _confidence_from_logprob(logprob: float) -> float:
    return max(0.0, min(1.0, math.exp(min(0.0, float(logprob)))))


def _completed_artifact(
    config: ExperimentConfig,
    started_at: float,
    preconditions: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    sample_rows: list[JsonDict],
    *,
    live_model_invoked: bool,
) -> JsonDict:
    usable_rows = [row for row in sample_rows if str(row.get("response_text") or "").strip()]
    labels = [int(row["hallucination_label"]) for row in sample_rows]
    confidence_pairs = [
        (int(row["hallucination_label"]), 1.0 - float(row["first_token_confidence"]))
        for row in sample_rows
        if row.get("first_token_confidence") is not None
    ]
    spilled_pairs = [
        (int(row["hallucination_label"]), float(row["spilled_energy"]))
        for row in sample_rows
        if row.get("spilled_energy") is not None
    ]
    first_auc = _auroc_from_pairs(confidence_pairs)
    spilled_auc = _auroc_from_pairs(spilled_pairs)
    blocked_metrics: list[str] = []
    if not confidence_pairs and not spilled_pairs:
        blocked_metrics.append("blocked_logprobs_unavailable")
    elif len(set(labels)) < 2:
        blocked_metrics.append("blocked_auroc_undefined_label_variance")
    return _base_artifact(
        config,
        started_at,
        "micro_panel_complete_no_full_benchmark_claim",
        list(preconditions)
        + [
            {
                "resource": "generation_logprobs",
                "available": bool(confidence_pairs or spilled_pairs),
                "detail": f"rows_with_logprobs={max(len(confidence_pairs), len(spilled_pairs))}",
            }
        ],
        model_specs=list(model_specs),
        models_used=_models_used(sample_rows),
        n_examples=len(sample_rows),
        first_token_confidence_available=bool(confidence_pairs),
        spilled_energy_available=bool(spilled_pairs),
        first_token_confidence_auroc=first_auc,
        spilled_energy_auroc=spilled_auc,
        usable_response_count=len(usable_rows),
        blocked_metrics=blocked_metrics,
        sample_rows=sample_rows,
        live_model_invoked=live_model_invoked,
    )


def _auroc_from_pairs(pairs: Sequence[tuple[int, float]]) -> float | None:
    if not pairs:
        return None
    return compute_auroc([label for label, _score in pairs], [score for _label, score in pairs])


def _models_used(sample_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    used: list[str] = []
    for row in sample_rows:
        hf_id = str(row.get("model_hf_id") or "")
        if hf_id and hf_id not in used:
            used.append(hf_id)
    return used


def _base_artifact(
    config: ExperimentConfig,
    started_at: float,
    honest_verdict: str,
    preconditions: Sequence[Mapping[str, Any]],
    *,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    models_used: Sequence[str] | None = None,
    n_examples: int = 0,
    first_token_confidence_available: bool = False,
    spilled_energy_available: bool = False,
    first_token_confidence_auroc: float | None = None,
    spilled_energy_auroc: float | None = None,
    usable_response_count: int = 0,
    blocked_metrics: Sequence[str] = (),
    sample_rows: Sequence[Mapping[str, Any]] = (),
    live_model_invoked: bool = False,
) -> JsonDict:
    duration = config.clock() - started_at
    artifact: JsonDict = {
        "honest_verdict": honest_verdict,
        "micro_panel_ready": (
            honest_verdict == "micro_panel_complete_no_full_benchmark_claim"
            and usable_response_count > 0
        ),
        "live_model_invoked": live_model_invoked,
        "model_specs": [dict(spec) for spec in (model_specs or [])],
        "models_used": list(models_used or []),
        "n_examples": int(n_examples),
        "first_token_confidence_available": first_token_confidence_available,
        "spilled_energy_available": spilled_energy_available,
        "first_token_confidence_auroc": first_token_confidence_auroc,
        "spilled_energy_auroc": spilled_energy_auroc,
        "usable_response_count": int(usable_response_count),
        "blocked_metrics": list(blocked_metrics),
        "sample_rows": [dict(row) for row in sample_rows],
        "random_seed": config.random_seed,
        "reproducibility_checksum": "",
        "preconditions_checked": [dict(row) for row in preconditions],
        "field_principles": _field_principles(),
        "run_date": config.run_date,
        "duration_s": round(duration, 6),
        "artifact": "experiment_2870_sota_energy_baseline_micro_panel_v1",
        "schema_version": 1,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _field_principles() -> JsonDict:
    return {
        "claim_boundary": "Tiny live micro-panel only; no full-benchmark or headline AUROC claim.",
        "model_selection": "cached_sota_pair() is attempted first; Exp 2862 selected mandated GGUF is fallback evidence when only one model is cached.",
        "first_token_confidence_auroc": "Computed as AUROC of 1 - first-token confidence against hallucination_label=1.",
        "spilled_energy_auroc": "Computed from real token logprobs only; higher spilled energy predicts hallucination_label=1.",
        "blocked_metrics": "Missing logprobs produce blocked_logprobs_unavailable instead of proxy metric fabrication.",
        "duration_s": "Measured wall-clock duration; no sleep padding.",
    }


def _checksum(artifact: Mapping[str, Any]) -> str:
    payload = {
        "random_seed": artifact.get("random_seed"),
        "model_specs": artifact.get("model_specs"),
        "sample_rows": artifact.get("sample_rows"),
        "blocked_metrics": artifact.get("blocked_metrics"),
        "run_date": artifact.get("run_date"),
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _write_artifact(path: Path, artifact: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_live_panel(  # pragma: no cover - exercised only by the live experiment run.
    *,
    model_spec: Mapping[str, Any],
    examples: Sequence[MicroPanelExample],
    selected_python: str,
    env: Mapping[str, str],
    random_seed: int,
) -> list[JsonDict]:
    del selected_python, env
    from llama_cpp import Llama  # type: ignore[import]  # noqa: PLC0415

    llm = Llama(
        model_path=str(model_spec["model_path"]),
        n_ctx=1024,
        n_batch=64,
        n_ubatch=64,
        n_gpu_layers=-1,
        main_gpu=int(model_spec.get("gpu") or 0),
        logits_all=True,
        verbose=False,
    )
    rows: list[JsonDict] = []
    try:
        for index, example in enumerate(examples):
            started = time.perf_counter()
            try:
                out = llm.create_completion(
                    prompt=example.prompt_text(),
                    max_tokens=8,
                    temperature=0.0,
                    seed=random_seed + index,
                    logprobs=5,
                    stop=["\n"],
                )
                choice = out.get("choices", [{}])[0]
                logprobs = choice.get("logprobs") or {}
                text = str(choice.get("text") or "").strip()
                rows.append(
                    {
                        "example_id": example.example_id,
                        "model_hf_id": model_spec["hf_id"],
                        "model_path": model_spec["model_path"],
                        "response_text": text,
                        "tokens_generated": int(out.get("usage", {}).get("completion_tokens") or len(logprobs.get("tokens") or text.split())),
                        "duration_s": round(time.perf_counter() - started, 6),
                        "logprobs_requested": True,
                        "logprobs_available": bool(logprobs.get("token_logprobs")),
                        "tokens": logprobs.get("tokens") or [],
                        "token_logprobs": logprobs.get("token_logprobs") or [],
                        "top_logprobs": logprobs.get("top_logprobs") or [],
                    }
                )
            except Exception as exc:
                first_error = f"{type(exc).__name__}: {exc}"
                try:
                    out = llm.create_completion(
                        prompt=example.prompt_text(),
                        max_tokens=8,
                        temperature=0.0,
                        seed=random_seed + index,
                        stop=["\n"],
                    )
                    choice = out.get("choices", [{}])[0]
                    text = str(choice.get("text") or "").strip()
                    rows.append(
                        {
                            "example_id": example.example_id,
                            "model_hf_id": model_spec.get("hf_id"),
                            "model_path": model_spec.get("model_path"),
                            "response_text": text,
                            "tokens_generated": int(out.get("usage", {}).get("completion_tokens") or len(text.split())),
                            "duration_s": round(time.perf_counter() - started, 6),
                            "logprobs_requested": True,
                            "logprobs_available": False,
                            "error": f"logprobs_unavailable_retry_without_logprobs: {first_error}",
                        }
                    )
                except Exception as retry_exc:
                    rows.append(
                        {
                            "example_id": example.example_id,
                            "model_hf_id": model_spec.get("hf_id"),
                            "model_path": model_spec.get("model_path"),
                            "response_text": "",
                            "tokens_generated": 0,
                            "duration_s": round(time.perf_counter() - started, 6),
                            "logprobs_requested": True,
                            "logprobs_available": False,
                            "error": f"{first_error}; retry_failed={type(retry_exc).__name__}: {retry_exc}",
                        }
                    )
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()
    return rows


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI glue.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run-date", default=RUN_DATE)
    parser.add_argument("--n-examples", type=int, default=DEFAULT_N_EXAMPLES)
    args = parser.parse_args(argv)
    run_experiment(
        ExperimentConfig(
            repo_root=Path.cwd(),
            output_path=args.output,
            run_date=args.run_date,
            n_examples=args.n_examples,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI glue.
    raise SystemExit(main())
