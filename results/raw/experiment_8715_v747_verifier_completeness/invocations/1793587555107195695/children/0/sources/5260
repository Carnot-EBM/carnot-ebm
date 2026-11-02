"""Exp 2917 spilled-energy logit detector micro-panel.

This module builds a deliberately small diagnostic panel from local code
generation outcomes and local factuality rows, runs those rows through a local
mandated GGUF model, and records final-token logprob/logit-derived energy
features.  It does not train a detector and it never upgrades a benchmark or
matrix row by itself.

Spec: REQ-INFER-SOTA-018,
      SCENARIO-INFER-SOTA-018-001,
      SCENARIO-INFER-SOTA-018-002
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]
ClockFn = Callable[[], float]
CachedPairProvider = Callable[..., list[dict[str, Any]] | None]
MandatedModelResolver = Callable[[], list[dict[str, Any]]]
InferenceRunner = Callable[..., JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2917_spilled_energy_logit_detector_micro_panel_v1.json"
RUN_DATE = "20260523"
RANDOM_SEED = 2917
INFERENCE_SUBSTRATE = "live_llm_inference"
CLAIM_BOUNDARY = "diagnostic_only_no_benchmark_claim"
MIN_EXAMPLES = 24
MAX_EXAMPLES = 40
DEFAULT_TARGET_EXAMPLES = 24
DEFAULT_MAX_TOKENS = 8
EXP2910_REL_PATH = Path("results") / "experiment_2910_sota_code_generation_corrigendum_v2.json"
HALUEVAL_REL_PATH = Path("data") / "eval_manifests" / "halueval_20260522.jsonl"

MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "spilled_energy_micro_panel_ready",
    "benchmark_claim_made",
    "claim_boundary",
    "model_specs",
    "models_used",
    "cached_sota_pair_used",
    "examples",
    "random_seed",
    "logprob_or_logits_available",
    "spilled_energy_features",
    "separation_summary",
    "inference_substrate",
    "duration_s",
    "run_date",
)

FEATURE_NAMES: tuple[str, ...] = (
    "final_token_spilled_energy",
    "final_token_marginal_energy",
    "sequence_spilled_energy",
    "sequence_marginal_energy",
)


@dataclass(frozen=True)
class PanelExample:
    """One local source row converted into a bounded diagnostic prompt."""

    example_id: str
    source_type: str
    source_id: str
    source_path: str
    prompt: str
    source_output: str
    verification_label: str
    verification_label_int: int
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for Exp 2917.

    The defaults point at the real repository artifacts.  Tests inject a temp
    repo root, deterministic clock, and fake inference runner so the schema and
    feature math are exercised without loading a GGUF.
    """

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    code_artifact_path: Path = EXP2910_REL_PATH
    factuality_manifest_path: Path = HALUEVAL_REL_PATH
    run_date: str = RUN_DATE
    random_seed: int = RANDOM_SEED
    target_examples: int = DEFAULT_TARGET_EXAMPLES
    max_tokens: int = DEFAULT_MAX_TOKENS
    tests_run: Sequence[str] = field(default_factory=tuple)
    started_at: float | None = None
    clock: ClockFn = time.perf_counter

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def resolved_code_artifact_path(self) -> Path:
        return _repo_path(self.repo_root, self.code_artifact_path)

    def resolved_factuality_manifest_path(self) -> Path:
        return _repo_path(self.repo_root, self.factuality_manifest_path)


def write_experiment_artifact(
    config: ExperimentConfig | None = None,
    *,
    inference_runner: InferenceRunner = None,
    cached_pair_provider: CachedPairProvider = None,
    mandated_model_resolver: MandatedModelResolver = None,
) -> JsonDict:
    """Run Exp 2917 and persist the resulting artifact JSON."""

    active = config or ExperimentConfig()
    started_at = active.start_time()
    artifact = build_experiment_artifact(
        active,
        started_at=started_at,
        inference_runner=inference_runner or _default_inference_runner,
        cached_pair_provider=cached_pair_provider or _default_cached_sota_pair,
        mandated_model_resolver=mandated_model_resolver or _default_mandated_model_resolver,
    )
    output_path = active.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def build_experiment_artifact(
    config: ExperimentConfig,
    *,
    started_at: float,
    inference_runner: InferenceRunner,
    cached_pair_provider: CachedPairProvider,
    mandated_model_resolver: MandatedModelResolver,
) -> JsonDict:
    """Build the diagnostic artifact without training or benchmark promotion."""

    model_specs, cached_pair_used = _resolve_model_specs(
        cached_pair_provider=cached_pair_provider,
        mandated_model_resolver=mandated_model_resolver,
    )
    if not model_specs:
        return _artifact(
            config,
            started_at,
            honest_verdict="complete: blocked_no_mandated_sota_gguf_cached",
            ready=False,
            model_specs=[],
            models_used=[],
            cached_pair_used=False,
            examples=[],
            logprob_or_logits_available=False,
            spilled_energy_features=_unavailable_feature_summary("blocked_no_mandated_sota_gguf_cached"),
            separation_summary=_unavailable_separation_summary("blocked_no_mandated_sota_gguf_cached"),
            blocked_reason="blocked_no_mandated_sota_gguf_cached",
        )

    panel = build_panel(config)
    if len(panel) < MIN_EXAMPLES or len(panel) > MAX_EXAMPLES:
        reason = "blocked_insufficient_micro_panel_examples"
        return _artifact(
            config,
            started_at,
            honest_verdict=f"complete: {reason}",
            ready=False,
            model_specs=model_specs,
            models_used=[str(model_specs[0]["hf_id"])],
            cached_pair_used=cached_pair_used,
            examples=[],
            logprob_or_logits_available=False,
            spilled_energy_features=_unavailable_feature_summary(reason),
            separation_summary=_unavailable_separation_summary(reason),
            blocked_reason=reason,
        )

    model_spec = model_specs[0]
    scored: list[JsonDict] = []
    for index, example in enumerate(panel):
        generation = inference_runner(
            prompt=example.prompt,
            example=example,
            model_spec=model_spec,
            seed=config.random_seed + index,
            max_tokens=config.max_tokens,
        )
        row = score_example(example, generation, model_spec=model_spec)
        if not row["logprob_or_logits_available"]:
            reason = "blocked_logprob_runtime_unavailable"
            return _artifact(
                config,
                started_at,
                honest_verdict=f"complete: {reason}",
                ready=False,
                model_specs=model_specs,
                models_used=[str(model_spec["hf_id"])],
                cached_pair_used=cached_pair_used,
                examples=[],
                logprob_or_logits_available=False,
                spilled_energy_features=_unavailable_feature_summary(reason),
                separation_summary=_unavailable_separation_summary(reason),
                blocked_reason=reason,
            )
        scored.append(row)

    return _artifact(
        config,
        started_at,
        honest_verdict="complete: spilled_energy_micro_panel_diagnostic_ready",
        ready=True,
        model_specs=model_specs,
        models_used=[str(model_spec["hf_id"])],
        cached_pair_used=cached_pair_used,
        examples=scored,
        logprob_or_logits_available=True,
        spilled_energy_features=build_spilled_energy_features(scored),
        separation_summary=build_separation_summary(scored),
        blocked_reason="",
    )


def build_panel(config: ExperimentConfig) -> list[PanelExample]:
    """Build a 50/50 local panel from code-output and factuality examples."""

    target = min(MAX_EXAMPLES, max(MIN_EXAMPLES, int(config.target_examples)))
    code_target = target // 2
    factuality_target = target - code_target
    code_examples = _load_code_examples(config, limit=code_target)
    factuality_examples = _load_factuality_examples(config, limit=factuality_target)
    return [*code_examples, *factuality_examples]


def score_example(
    example: PanelExample,
    generation: Mapping[str, Any],
    *,
    model_spec: Mapping[str, Any],
) -> JsonDict:
    """Attach raw runtime output and energy features to one panel example."""

    features = compute_energy_features(generation)
    raw_response = str(generation.get("raw_response") or generation.get("response_text") or "")
    model_id = str(generation.get("model_id") or model_spec.get("hf_id") or "")
    top_logprobs = _top_logprob_rows(generation.get("top_logprobs"))
    token_logprobs = _numeric_values(generation.get("token_logprobs"))
    return {
        "example_id": example.example_id,
        "source_type": example.source_type,
        "source_id": example.source_id,
        "source_path": example.source_path,
        "source_output": example.source_output,
        "prompt_hash": _sha256_text(example.prompt),
        "raw_response": raw_response,
        "model_id": model_id,
        "model_path": str(model_spec.get("model_path") or ""),
        "token_count": features["token_count"],
        "logprob_or_logits_available": features["logprob_or_logits_available"],
        "token_logprob_count": len(token_logprobs),
        "top_logprobs_count": len(top_logprobs),
        "final_logits_count": len(_final_logits(generation)),
        "verification_label": example.verification_label,
        "verification_label_int": example.verification_label_int,
        "final_token_top1_probability": features["final_token_top1_probability"],
        "final_token_spilled_energy": features["final_token_spilled_energy"],
        "final_token_marginal_energy": features["final_token_marginal_energy"],
        "sequence_spilled_energy": features["sequence_spilled_energy"],
        "sequence_marginal_energy": features["sequence_marginal_energy"],
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "metadata": dict(example.metadata),
    }


def compute_energy_features(generation: Mapping[str, Any]) -> JsonDict:
    """Compute final-token and sequence-level spilled/marginal energies."""

    token_logprobs = _numeric_values(generation.get("token_logprobs"))
    top_rows = _top_logprob_rows(generation.get("top_logprobs"))
    final_distribution = _final_distribution(top_rows, generation)
    selected_final_prob = _selected_final_probability(token_logprobs)

    if final_distribution:
        top1_probability = max(final_distribution)
    elif selected_final_prob is not None:
        top1_probability = selected_final_prob
    else:
        top1_probability = None

    sequence_spilled = _mean([1.0 - math.exp(min(0.0, value)) for value in token_logprobs])
    sequence_marginal = -_mean(token_logprobs) if token_logprobs else None
    final_spilled = (1.0 - top1_probability) if top1_probability is not None else None
    final_marginal = (
        -math.log(max(1.0 - top1_probability, 1e-12))
        if top1_probability is not None
        else None
    )
    token_count = _token_count(generation, token_logprobs)
    available = bool(token_logprobs or top_rows or _final_logits(generation))
    return {
        "logprob_or_logits_available": available,
        "token_count": token_count,
        "final_token_top1_probability": top1_probability,
        "final_token_spilled_energy": final_spilled,
        "final_token_marginal_energy": final_marginal,
        "sequence_spilled_energy": sequence_spilled,
        "sequence_marginal_energy": sequence_marginal,
    }


def build_spilled_energy_features(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize feature availability and formulas without fitting anything."""

    return {
        "available": bool(rows),
        "detector_trained": False,
        "feature_names": list(FEATURE_NAMES),
        "feature_formulas": {
            "final_token_spilled_energy": "1 - max softmax(final_token_topk_logprobs_or_logits)",
            "final_token_marginal_energy": "-log(non_top1_final_token_probability_mass)",
            "sequence_spilled_energy": "mean(1 - exp(selected_token_logprob))",
            "sequence_marginal_energy": "-mean(selected_token_logprob)",
        },
        "per_label_means": _per_label_feature_means(rows),
    }


def build_separation_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute fixed diagnostic separations; no detector is trained."""

    summary: JsonDict = {
        "available": bool(rows),
        "detector_trained": False,
        "n_examples": len(rows),
        "label_counts": _label_counts(rows),
        "features": {},
    }
    for feature in FEATURE_NAMES:
        pairs = [
            (int(row["verification_label_int"]), float(row[feature]))
            for row in rows
            if _finite_or_none(row.get(feature)) is not None
        ]
        summary["features"][feature] = _feature_separation(pairs)
    return summary


def _artifact(
    config: ExperimentConfig,
    started_at: float,
    *,
    honest_verdict: str,
    ready: bool,
    model_specs: Sequence[Mapping[str, Any]],
    models_used: Sequence[str],
    cached_pair_used: bool,
    examples: Sequence[Mapping[str, Any]],
    logprob_or_logits_available: bool,
    spilled_energy_features: Mapping[str, Any],
    separation_summary: Mapping[str, Any],
    blocked_reason: str,
) -> JsonDict:
    return {
        "artifact": "experiment_2917_spilled_energy_logit_detector_micro_panel_v1",
        "schema_version": 1,
        "honest_verdict": honest_verdict,
        "spilled_energy_micro_panel_ready": bool(ready),
        "benchmark_claim_made": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "model_specs": [dict(spec) for spec in model_specs],
        "models_used": list(models_used),
        "cached_sota_pair_used": bool(cached_pair_used),
        "examples": [dict(row) for row in examples],
        "random_seed": int(config.random_seed),
        "logprob_or_logits_available": bool(logprob_or_logits_available),
        "spilled_energy_features": dict(spilled_energy_features),
        "separation_summary": dict(separation_summary),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0.0, config.clock() - started_at), 6),
        "run_date": config.run_date,
        "blocked_reason": blocked_reason,
        "tests_run": list(config.tests_run),
        "field_principles": _field_principles(),
    }


def _resolve_model_specs(
    *,
    cached_pair_provider: CachedPairProvider,
    mandated_model_resolver: MandatedModelResolver,
) -> tuple[list[JsonDict], bool]:
    pair = _usable_specs(cached_pair_provider(gpu_indices=(0, 1)))
    if pair:
        return pair, True
    return _usable_specs(mandated_model_resolver()), False


def _usable_specs(specs: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    usable: list[JsonDict] = []
    for spec in specs or ():
        hf_id = str(spec.get("hf_id") or "")
        model_path = str(spec.get("model_path") or "")
        if hf_id in MANDATED_MODEL_IDS and model_path and Path(model_path).is_file():
            usable.append(dict(spec))
    return usable


def _load_code_examples(config: ExperimentConfig, *, limit: int) -> list[PanelExample]:
    artifact = _read_json(config.resolved_code_artifact_path())
    rows = [row for row in artifact.get("candidate_results", []) if isinstance(row, Mapping)]
    buckets: dict[str, list[Mapping[str, Any]]] = {"verified": [], "hallucination_like": []}
    for row in rows:
        label = "verified" if bool(row.get("passed")) else "hallucination_like"
        buckets[label].append(row)
    selected = _balanced_rows(buckets, limit)
    examples: list[PanelExample] = []
    for index, row in enumerate(selected):
        label = "verified" if bool(row.get("passed")) else "hallucination_like"
        raw_response = str(row.get("raw_response") or _read_optional_text(config.repo_root, row.get("raw_response_path")))
        source_id = f"{row.get('corpus', 'code')}:{row.get('stable_id', index)}:{row.get('candidate_index', index)}"
        source_path = str(row.get("raw_response_path") or config.code_artifact_path)
        examples.append(
            PanelExample(
                example_id=f"code-{index:02d}-{_sha256_text(source_id)[:8]}",
                source_type="code",
                source_id=source_id,
                source_path=source_path,
                prompt=_code_prompt(raw_response),
                source_output=raw_response[:2000],
                verification_label=label,
                verification_label_int=0 if label == "verified" else 1,
                metadata={
                    "corpus": row.get("corpus"),
                    "stable_id": row.get("stable_id"),
                    "candidate_index": row.get("candidate_index"),
                    "source_prompt_sha256": row.get("prompt_sha256"),
                },
            )
        )
    return examples


def _load_factuality_examples(config: ExperimentConfig, *, limit: int) -> list[PanelExample]:
    rows = _read_jsonl(config.resolved_factuality_manifest_path())
    buckets: dict[str, list[Mapping[str, Any]]] = {"verified": [], "hallucination_like": []}
    for row in rows:
        label = "hallucination_like" if int(row.get("label", 1)) == 1 else "verified"
        buckets[label].append(row)
    selected = _balanced_rows(buckets, limit)
    examples: list[PanelExample] = []
    for index, row in enumerate(selected):
        label = "hallucination_like" if int(row.get("label", 1)) == 1 else "verified"
        source_output = str(row.get("candidate") or "")
        prompt = _factuality_prompt(
            prompt=str(row.get("prompt") or ""),
            candidate=source_output,
            reference=str(row.get("reference") or ""),
        )
        source_id = str(row.get("stable_id") or f"halueval-{index}")
        examples.append(
            PanelExample(
                example_id=f"factuality-{index:02d}-{_sha256_text(source_id)[:8]}",
                source_type="factuality",
                source_id=source_id,
                source_path=str(config.factuality_manifest_path),
                prompt=prompt,
                source_output=source_output[:2000],
                verification_label=label,
                verification_label_int=0 if label == "verified" else 1,
                metadata={"dataset": row.get("dataset"), "reference": row.get("reference")},
            )
        )
    return examples


def _balanced_rows(buckets: Mapping[str, Sequence[Mapping[str, Any]]], limit: int) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    max_bucket = max((len(rows) for rows in buckets.values()), default=0)
    for index in range(max_bucket):
        for label in ("verified", "hallucination_like"):
            rows = buckets.get(label, ())
            if index < len(rows):
                selected.append(rows[index])
                if len(selected) >= limit:
                    return selected
    return selected


def _code_prompt(raw_response: str) -> str:
    clipped = raw_response[:1800]
    return (
        "Inspect this generated Python candidate using the local test result as the target label.\n"
        "Reply with exactly VERIFIED or HALLUCINATION-LIKE.\n\n"
        f"Candidate code:\n{clipped}\n\nVerdict:"
    )


def _factuality_prompt(*, prompt: str, candidate: str, reference: str) -> str:
    return (
        "Inspect this factual answer candidate using the provided reference.\n"
        "Reply with exactly VERIFIED or HALLUCINATION-LIKE.\n\n"
        f"Question context:\n{prompt[:1200]}\n\n"
        f"Candidate answer: {candidate[:500]}\n"
        f"Reference answer: {reference[:500]}\n\nVerdict:"
    )


def _top_logprob_rows(value: Any) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not isinstance(value, list):
        return rows
    for item in value:
        if isinstance(item, Mapping):
            converted = {
                str(key): numeric
                for key, raw in item.items()
                if (numeric := _finite_or_none(raw)) is not None
            }
            if converted:
                rows.append(converted)
    return rows


def _final_distribution(top_rows: Sequence[Mapping[str, float]], generation: Mapping[str, Any]) -> list[float]:
    if top_rows:
        return _softmax_log_values(list(top_rows[-1].values()))
    logits = _final_logits(generation)
    return _softmax_log_values(logits) if logits else []


def _final_logits(generation: Mapping[str, Any]) -> list[float]:
    for key in ("final_logits", "logits"):
        values = generation.get(key)
        if isinstance(values, list):
            if values and isinstance(values[-1], list):
                return _numeric_values(values[-1])
            return _numeric_values(values)
    return []


def _selected_final_probability(token_logprobs: Sequence[float]) -> float | None:
    if not token_logprobs:
        return None
    return max(0.0, min(1.0, math.exp(min(0.0, float(token_logprobs[-1])))))


def _softmax_log_values(values: Sequence[float]) -> list[float]:
    numeric = [float(value) for value in values if math.isfinite(float(value))]
    if not numeric:
        return []
    maximum = max(numeric)
    weights = [math.exp(value - maximum) for value in numeric]
    total = sum(weights)
    return [weight / total for weight in weights]


def _token_count(generation: Mapping[str, Any], token_logprobs: Sequence[float]) -> int:
    explicit = generation.get("token_count") or generation.get("tokens_generated")
    if isinstance(explicit, int) and explicit >= 0:
        return explicit
    tokens = generation.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    return len(token_logprobs)


def _feature_separation(pairs: Sequence[tuple[int, float]]) -> JsonDict:
    verified = [score for label, score in pairs if label == 0]
    risky = [score for label, score in pairs if label == 1]
    return {
        "n_with_feature": len(pairs),
        "verified_mean": _mean(verified),
        "hallucination_like_mean": _mean(risky),
        "delta_hallucination_like_minus_verified": (
            _mean(risky) - _mean(verified) if verified and risky else None
        ),
        "auroc": _auroc([label for label, _ in pairs], [score for _, score in pairs]),
    }


def _per_label_feature_means(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    result: JsonDict = {}
    for label in ("verified", "hallucination_like"):
        label_rows = [row for row in rows if row.get("verification_label") == label]
        result[label] = {
            feature: _mean(
                [
                    float(row[feature])
                    for row in label_rows
                    if _finite_or_none(row.get(feature)) is not None
                ]
            )
            for feature in FEATURE_NAMES
        }
    return result


def _label_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "verified": sum(1 for row in rows if row.get("verification_label_int") == 0),
        "hallucination_like": sum(1 for row in rows if row.get("verification_label_int") == 1),
    }


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        wins += sum(1.0 for negative in negatives if positive > negative)
        wins += sum(0.5 for negative in negatives if positive == negative)
    return wins / (len(positives) * len(negatives))


def _unavailable_feature_summary(reason: str) -> JsonDict:
    return {
        "available": False,
        "detector_trained": False,
        "blocked_reason": reason,
        "feature_names": list(FEATURE_NAMES),
    }


def _unavailable_separation_summary(reason: str) -> JsonDict:
    return {
        "available": False,
        "detector_trained": False,
        "blocked_reason": reason,
        "n_examples": 0,
        "features": {},
    }


def _field_principles() -> JsonDict:
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "benchmark_claim_made": "Always false; this is a diagnostic micro-panel only.",
        "detector_training": "No detector is trained; separation metrics are fixed descriptive summaries.",
        "runtime_gate": "Missing token logprobs, top-k logprobs, and final-token logits blocks the artifact.",
        "model_selection": "cached_sota_pair(gpu_indices=(0, 1)) is attempted before single-model fallback.",
    }


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_optional_text(repo_root: Path, raw_path: Any) -> str:
    if not raw_path:
        return ""
    path = _repo_path(repo_root, Path(str(raw_path)))
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _numeric_values(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    return [numeric for value in values if (numeric := _finite_or_none(value)) is not None]


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _default_cached_sota_pair(**kwargs: Any) -> list[dict[str, Any]] | None:  # pragma: no cover
    from carnot.inference.sota_models import cached_sota_pair

    return cached_sota_pair(**kwargs)


def _default_mandated_model_resolver() -> list[dict[str, Any]]:  # pragma: no cover
    from carnot.inference.sota_models import SOTA_GGUF_MODELS, resolve_cached_gguf

    specs: list[dict[str, Any]] = []
    for index, model in enumerate(SOTA_GGUF_MODELS):
        model_path = resolve_cached_gguf(model["hf_id"])
        if model_path and Path(model_path).is_file():
            specs.append(
                {
                    "name": model["name"],
                    "hf_id": model["hf_id"],
                    "gpu": index,
                    "model_path": model_path,
                }
            )
    return specs


_LLAMA_CACHE: dict[str, Any] = {}


def _default_inference_runner(  # pragma: no cover
    *,
    prompt: str,
    example: PanelExample,
    model_spec: Mapping[str, Any],
    seed: int,
    max_tokens: int,
) -> JsonDict:
    del example
    from llama_cpp import Llama

    model_path = str(model_spec["model_path"])
    if model_path not in _LLAMA_CACHE:
        _LLAMA_CACHE[model_path] = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_batch=64,
            n_gpu_layers=-1,
            main_gpu=int(model_spec.get("gpu") or 0),
            logits_all=True,
            verbose=False,
        )
    llm = _LLAMA_CACHE[model_path]
    started = time.perf_counter()
    try:
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=int(max_tokens),
            temperature=0.0,
            top_p=1.0,
            seed=int(seed),
            logprobs=5,
            stop=["</s>", "<eos>"],
        )
    except TypeError:
        result = llm.create_completion(
            prompt=prompt,
            max_tokens=int(max_tokens),
            temperature=0.0,
            top_p=1.0,
            seed=int(seed),
            stop=["</s>", "<eos>"],
        )
    telemetry = _extract_completion_telemetry(result if isinstance(result, Mapping) else {})
    final_logits = (
        []
        if telemetry.get("token_logprobs") or telemetry.get("top_logprobs")
        else _runtime_final_logits(llm)
    )
    telemetry.update(
        {
            "duration_s": round(time.perf_counter() - started, 6),
            "model_id": model_spec.get("hf_id"),
            "final_logits": final_logits,
        }
    )
    return telemetry


def _extract_completion_telemetry(result: Mapping[str, Any]) -> JsonDict:  # pragma: no cover
    choices = result.get("choices") or []
    first = choices[0] if choices and isinstance(choices[0], Mapping) else {}
    usage = result.get("usage") if isinstance(result.get("usage"), Mapping) else {}
    logprobs = first.get("logprobs") if isinstance(first.get("logprobs"), Mapping) else {}
    text = str(first.get("text") or "")
    return {
        "raw_response": text.strip(),
        "token_count": usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else 0,
        "tokens": [str(token) for token in logprobs.get("tokens") or []],
        "token_logprobs": _numeric_values(logprobs.get("token_logprobs")),
        "top_logprobs": _top_logprob_rows(logprobs.get("top_logprobs")),
    }


def _runtime_final_logits(llm: Any) -> list[float]:  # pragma: no cover
    for attr in ("scores", "eval_logits"):
        value = getattr(llm, attr, None)
        if isinstance(value, list) and value:
            row = value[-1]
            return _numeric_values(row) if isinstance(row, list) else []
    return []


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--target-examples", type=int, default=DEFAULT_TARGET_EXAMPLES)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--tests-run", action="append", default=[])
    args = parser.parse_args(argv)
    write_experiment_artifact(
        ExperimentConfig(
            repo_root=Path.cwd(),
            output_path=args.output,
            target_examples=args.target_examples,
            max_tokens=args.max_tokens,
            tests_run=args.tests_run,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
