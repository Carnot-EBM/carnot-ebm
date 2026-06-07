"""Exp 3917 efficiency head-to-head verifier comparison.

This runner measures the operator-facing efficiency question directly: the
cheap energy verifier and the LLM-as-judge score the same labeled steps, then
the artifact reports accuracy parity and measured cost ratios. The LLM judge
uses the Exp 3915 robust GGUF generator path; both verifier timings use the Exp
3905 cost measurement helper.

Spec refs: REQ-VERIFY-3917, SCENARIO-VERIFY-3917,
SCENARIO-VERIFY-3917-BLOCKED.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from carnot.eval.fover_memory_leakage_v3 import _label_to_int
from carnot.verify.cost_instrumented_verification import (
    _step_text,
    measure_verification_cost,
    model_params_for_path,
    run_energy_verifier,
)
from carnot.verify.gguf_inference import (
    DEFAULT_PREFER_ORDER,
    generate as gguf_generate,
    load_gguf_generator,
)
from carnot.verify.reasoner_self_verification import (
    build_judge_prompt,
    parse_self_verification_response,
)


OUTPUT_REL_PATH = Path("results/experiment_3917_efficiency_head_to_head.json")
EXP3915_ARTIFACT_REL_PATH = Path("results/experiment_3915_robust_gguf_inference_harness.json")
EXP3905_ARTIFACT_REL_PATH = Path("results/experiment_3905_cost_instrumented_verify_harness.json")
EXP3884_ARTIFACT_REL_PATH = Path("results/experiment_3884_in_distribution_error_rich_corpus.json")
FOVER_CORPUS_REL_PATH = Path("data/fover_corpus_v4.json")
GGUF_HARNESS_MODULE_PATH = "python/carnot/verify/gguf_inference.py"
COST_HARNESS_MODULE_PATH = "python/carnot/verify/cost_instrumented_verification.py"
TITLE = "efficiency_head_to_head"
EXPERIMENT_ID = 3917
DEFAULT_RANDOM_SEED = 3917
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_MAX_TOKENS = 96
MIN_SOURCE_ITEMS = 200
DURATION_FLOOR_S = 60.0
INFERENCE_SUBSTRATE = "live_llm_inference:robust_gguf_judge_plus_exp3905_cost_harness_energy_verifier"

REQUIRED_FIELDS = {
    "energy_auroc",
    "llm_judge_auroc",
    "accuracy_parity",
    "cost_ratio_walltime",
    "cost_ratio_flops",
    "energy_per_item_ms",
    "llm_per_item_ms",
    "llm_judge_model_used",
    "n_items",
    "corpus_sources",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "random_seeds_used",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
    "honest_verdict",
}

FIELD_PRINCIPLES: dict[str, str] = {
    "energy_auroc": (
        "Accuracy of each verifier on the SAME labels - parity is energy within the LLM-judge CI."
    ),
    "llm_judge_auroc": (
        "Accuracy of each verifier on the SAME labels - parity is energy within the LLM-judge CI."
    ),
    "accuracy_parity": (
        "BARE BOOL - is the energy verifier's AUROC within the LLM-judge's CI95 (equally effective)."
    ),
    "cost_ratio_walltime": (
        "BARE FLOAT - llm_per_item_ms / energy_per_item_ms; the 'Nx cheaper' headline."
    ),
    "cost_ratio_flops": "BARE FLOAT - FLOP-based cost ratio; substrate-independent corroboration.",
    "energy_per_item_ms": "Per-item latency for each verifier - the latency story.",
    "llm_per_item_ms": "Per-item latency for each verifier - the latency story.",
    "llm_judge_model_used": (
        "Which GGUF the robust harness loaded for the judge (records fallback if used)."
    ),
    "n_items": (
        "Pre-Launch + Adversarial-Verify - real LLM-judge run over hundreds of items."
    ),
    "corpus_sources": (
        "Pre-Launch + Adversarial-Verify - records both labeled corpora used in the same run."
    ),
    "preconditions_checked": (
        "Pre-Launch + Adversarial-Verify - resource gates checked before scoring."
    ),
    "model_specs": "Pre-Launch + Adversarial-Verify - exact GGUF and runtime provenance.",
    "random_seed": "Pre-Launch + Adversarial-Verify - primary reproducibility seed.",
    "random_seeds_used": "Pre-Launch + Adversarial-Verify - all deterministic sampling seeds.",
    "reproducibility_checksum": "Pre-Launch + Adversarial-Verify - hash over inputs and scores.",
    "duration_s": (
        "Pre-Launch + Adversarial-Verify - live floor 60s on this full-corpus artifact."
    ),
    "inference_substrate": "Pre-Launch + Adversarial-Verify - actual inference runtime.",
}

WRAPPED_VALUE_FORBIDDEN_FIELDS = (
    "energy_auroc",
    "llm_judge_auroc",
    "accuracy_parity",
    "cost_ratio_walltime",
    "cost_ratio_flops",
    "energy_per_item_ms",
    "llm_per_item_ms",
    "llm_judge_model_used",
    "n_items",
    "duration_s",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource gate checked before the Exp 3917 run."""

    resource: str
    available: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"resource": self.resource, "available": self.available, "detail": self.detail}


@dataclass(frozen=True)
class CorpusBundle:
    """Labeled verifier items from Exp 3884 and FoVer in one fixed order."""

    items: tuple[dict[str, object], ...]
    labels: tuple[int, ...]
    corpus_sources: tuple[dict[str, object], ...]
    checksum: str

    @property
    def n_items(self) -> int:
        return len(self.items)


@dataclass(frozen=True)
class CostMeasurements:
    """Cost-harness measurements plus retained score vectors for bootstrap CIs."""

    energy_cost: dict[str, object]
    llm_cost: dict[str, object]
    energy_scores: tuple[float, ...]
    llm_scores: tuple[float, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime configuration for Exp 3917."""

    repo_root: Path
    output_path: Path | None = None
    random_seed: int = DEFAULT_RANDOM_SEED
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES
    max_tokens: int = DEFAULT_MAX_TOKENS
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    cuda_probe_timeout_s: int = 60
    fover_min_items: int = MIN_SOURCE_ITEMS
    n_ctx: int = 1024

    def resolved_output_path(self) -> Path:
        return self.output_path if self.output_path is not None else self.repo_root / OUTPUT_REL_PATH

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def venv_python(self) -> Path:
        return self.repo_root / ".venv" / "bin" / "python"


def _checksum(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = payload.get("items") or payload.get("rows") or payload.get("scores") or []
    else:
        candidates = []
    return [dict(row) for row in candidates if isinstance(row, dict)]


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _relative_to_repo(repo_root: Path, path: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:  # pragma: no cover - absolute paths outside the repo are provenance only.
        return path.as_posix()


def _label_to_error_int(value: object) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int) and value in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "incorrect", "error", "bad", "wrong", "true"}:
        return 1
    if text in {"0", "correct", "ok", "good", "right", "false"}:
        return 0
    return _label_to_int(value)


def _step_from_row(row: dict[str, Any], index: int) -> str:
    for key in ("step_text", "step", "text", "candidate_text", "reasoning_step"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"row {index} has no step text")


def _normalized_item(
    row: dict[str, Any],
    *,
    corpus_name: str,
    source_index: int,
) -> dict[str, object]:
    label = _label_to_error_int(row.get("gold_error", row.get("label")))
    return {
        "step_text": _step_from_row(row, source_index),
        "gold_error": label,
        "label": "incorrect" if label else "correct",
        "corpus_source": corpus_name,
        "source_index": source_index,
        "question_id": row.get("question_id"),
        "corpus_item_id": row.get("corpus_item_id"),
    }


def _source_summary(
    *,
    name: str,
    path: Path,
    rows: Sequence[dict[str, object]],
    repo_root: Path,
    selection: str,
) -> dict[str, object]:
    labels = [int(row["gold_error"]) for row in rows]
    return {
        "name": name,
        "path": _relative_to_repo(repo_root, path),
        "sha256": _sha256_file(path),
        "selection": selection,
        "n_items": len(rows),
        "n_incorrect": sum(labels),
        "n_correct": len(labels) - sum(labels),
    }


def _load_exp3884_items(repo_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    artifact_path = repo_root / EXP3884_ARTIFACT_REL_PATH
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{EXP3884_ARTIFACT_REL_PATH.as_posix()} is missing")
    artifact = _read_json(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("exp3884 artifact is not a JSON object")
    corpus_path = _resolve_repo_path(repo_root, str(artifact.get("corpus_path") or ""))
    rows = _json_rows(_read_json(corpus_path))
    items = [
        _normalized_item(row, corpus_name="exp3884_in_distribution", source_index=index)
        for index, row in enumerate(rows)
    ]
    source = _source_summary(
        name="exp3884_in_distribution",
        path=corpus_path,
        rows=items,
        repo_root=repo_root,
        selection="all_rows_from_exp3884_artifact",
    )
    source["artifact_path"] = EXP3884_ARTIFACT_REL_PATH.as_posix()
    source["artifact_sha256"] = _sha256_file(artifact_path)
    return items, source


def _load_fover_slice(
    repo_root: Path,
    *,
    random_seed: int,
    min_items: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    corpus_path = repo_root / FOVER_CORPUS_REL_PATH
    rows = _json_rows(_read_json(corpus_path))
    incorrect = [row for row in rows if _label_to_error_int(row.get("label")) == 1]
    correct = [row for row in rows if _label_to_error_int(row.get("label")) == 0]
    per_class = min(len(incorrect), len(correct))
    if per_class * 2 < min_items:
        raise ValueError(f"FoVer slice has {per_class * 2} balanced rows; required >= {min_items}")
    rng = random.Random(random_seed)
    rng.shuffle(incorrect)
    rng.shuffle(correct)
    selected_raw = [*incorrect[:per_class], *correct[:per_class]]
    rng.shuffle(selected_raw)
    items = [
        _normalized_item(row, corpus_name="fover_corpus_v4_slice", source_index=index)
        for index, row in enumerate(selected_raw)
    ]
    source = _source_summary(
        name="fover_corpus_v4_slice",
        path=corpus_path,
        rows=items,
        repo_root=repo_root,
        selection=f"seeded_balanced_all_{per_class}_incorrect_plus_{per_class}_correct",
    )
    return items, source


def load_labeled_corpora(
    repo_root: Path,
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    fover_min_items: int = MIN_SOURCE_ITEMS,
) -> CorpusBundle:
    """Load Exp 3884 and the deterministic FoVer slice for the head-to-head."""

    exp3884_items, exp3884_source = _load_exp3884_items(repo_root)
    fover_items, fover_source = _load_fover_slice(
        repo_root,
        random_seed=random_seed,
        min_items=fover_min_items,
    )
    items = tuple([*exp3884_items, *fover_items])
    labels = tuple(int(item["gold_error"]) for item in items)
    for source in (exp3884_source, fover_source):
        if int(source["n_items"]) < fover_min_items:
            raise ValueError(f"{source['name']} has fewer than {fover_min_items} items")
    checksum = _checksum(
        {
            "items": [
                {
                    "step_text_sha256": hashlib.sha256(
                        str(item["step_text"]).encode("utf-8")
                    ).hexdigest(),
                    "gold_error": item["gold_error"],
                    "corpus_source": item["corpus_source"],
                    "source_index": item["source_index"],
                }
                for item in items
            ],
            "sources": [exp3884_source, fover_source],
            "random_seed": random_seed,
        }
    )
    return CorpusBundle(
        items=items,
        labels=labels,
        corpus_sources=(exp3884_source, fover_source),
        checksum=checksum,
    )


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if int(label) == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both positive and negative labels")
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def bootstrap_ci95(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    seed: int,
    resamples: int,
) -> dict[str, float]:
    """Compute a deterministic bootstrap CI95 for AUROC."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must align")
    rng = random.Random(seed)
    n_items = len(labels)
    aucs: list[float] = []
    for _ in range(resamples):
        indices = [rng.randrange(n_items) for _ in range(n_items)]
        sampled_labels = [int(labels[index]) for index in indices]
        if len(set(sampled_labels)) < 2:
            continue
        sampled_scores = [float(scores[index]) for index in indices]
        aucs.append(_auroc(sampled_labels, sampled_scores))
    if not aucs:
        aucs.append(_auroc(labels, scores))
    aucs.sort()
    low_index = int(0.025 * (len(aucs) - 1))
    high_index = int(0.975 * (len(aucs) - 1))
    return {"low": float(aucs[low_index]), "high": float(aucs[high_index])}


def _score_digest(scores: Sequence[float]) -> str:
    return _checksum({"scores": [round(float(score), 12) for score in scores]})


def _llama_token_count(generator: Any, text: str, *, add_bos: bool) -> int:
    payload = text.encode("utf-8")
    tokenize = getattr(generator, "tokenize", None)
    if callable(tokenize):
        try:
            return len(tokenize(payload, add_bos=add_bos))
        except TypeError:  # pragma: no cover - compatibility for older llama.cpp builds.
            return len(tokenize(payload))
    return max(1, len(text.split()) + (1 if add_bos else 0))


def run_llm_judge_with_generator(
    items: Sequence[dict[str, object]],
    *,
    generator: object,
    model_specs: dict[str, object],
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, object]:
    """Score items with the Exp 3894 judge prompt through the robust generator."""

    scores: list[float] = []
    total_tokens = 0
    for item in items:
        prompt = build_judge_prompt(_step_text(item))
        prompt_tokens = _llama_token_count(generator, prompt, add_bos=True)
        response = gguf_generate(generator, prompt, max_tokens=max_tokens).strip()
        completion_tokens = _llama_token_count(generator, response, add_bos=False)
        scores.append(parse_self_verification_response(response).score)
        total_tokens += prompt_tokens + completion_tokens
    model_path = str(model_specs.get("gguf_path") or model_specs.get("model_path") or "")
    params = int(model_specs.get("parameter_count_for_flop_estimate") or model_params_for_path(model_path))
    return {
        "scores": scores,
        "est_tokens": total_tokens,
        "est_flops": 2 * params * total_tokens,
    }


def _scores_from_result(result: dict[str, object], n_expected: int) -> tuple[float, ...]:
    scores = tuple(float(score) for score in result.get("scores", ()))
    if len(scores) != n_expected:
        raise ValueError(f"expected {n_expected} scores, got {len(scores)}")
    return scores


def measure_head_to_head_costs(
    items: Sequence[dict[str, object]],
    *,
    generator: object,
    model_specs: dict[str, object],
    clock: Callable[[], float] = time.perf_counter,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> CostMeasurements:
    """Measure both verifier families through Exp 3905 cost instrumentation."""

    energy_scores: tuple[float, ...] = ()
    llm_scores: tuple[float, ...] = ()

    def measured_energy(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
        nonlocal energy_scores
        result = dict(run_energy_verifier(rows))
        energy_scores = _scores_from_result(result, len(rows))
        return result

    def measured_llm(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
        nonlocal llm_scores
        result = run_llm_judge_with_generator(
            rows,
            generator=generator,
            model_specs=model_specs,
            max_tokens=max_tokens,
        )
        llm_scores = _scores_from_result(result, len(rows))
        return result

    energy_cost = measure_verification_cost(measured_energy, items, "energy_verifier", clock=clock)
    llm_cost = measure_verification_cost(measured_llm, items, "llm_judge", clock=clock)
    return CostMeasurements(
        energy_cost=energy_cost,
        llm_cost=llm_cost,
        energy_scores=energy_scores,
        llm_scores=llm_scores,
    )


def _prefer_order_from_source(gguf_harness_source: dict[str, object]) -> list[str]:
    model_used = gguf_harness_source.get("model_used")
    prefer_order: list[str] = []
    if isinstance(model_used, str) and model_used:
        prefer_order.append(model_used)
    for model_name in DEFAULT_PREFER_ORDER:
        if model_name not in prefer_order:
            prefer_order.append(model_name)
    return prefer_order


def load_robust_generator(
    gguf_harness_source: dict[str, object],
    config: ExperimentConfig,
) -> tuple[object, dict[str, object]]:  # pragma: no cover - exercised by full live script.
    """Load the live LLM judge through the Exp 3915 robust harness."""

    n_gpu_layers = int(gguf_harness_source.get("n_gpu_layers_used") or -1)
    generator, meta = load_gguf_generator(
        prefer_order=_prefer_order_from_source(gguf_harness_source),
        n_ctx=config.n_ctx,
        max_n_gpu_layers=n_gpu_layers,
    )
    model_path = str(meta.get("gguf_path") or "")
    return generator, {
        **dict(meta),
        "loader": "carnot.verify.gguf_inference.load_gguf_generator",
        "source_exp3915_model_used": gguf_harness_source.get("model_used"),
        "source_exp3915_n_gpu_layers_used": gguf_harness_source.get("n_gpu_layers_used"),
        "n_ctx": config.n_ctx,
        "max_tokens": config.max_tokens,
        "parameter_count_for_flop_estimate": model_params_for_path(model_path),
    }


def _cost_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _accuracy_parity(energy_auroc: float, llm_ci95: dict[str, float]) -> bool:
    return float(llm_ci95["low"]) <= energy_auroc <= float(llm_ci95["high"])


def _classify_verdict(
    *,
    accuracy_parity: bool,
    cost_ratio_walltime: float,
    energy_auroc: float,
    llm_auroc: float,
) -> str:
    ratio = f"{cost_ratio_walltime:.2f}"
    energy = f"{energy_auroc:.4f}"
    llm = f"{llm_auroc:.4f}"
    if accuracy_parity and cost_ratio_walltime > 10.0:
        return (
            "complete: "
            f"efficiency_PARITY_AND_{ratio}x_CHEAPER_energy{energy}_llm{llm}_"
            "verifier_earns_its_place"
        )
    if cost_ratio_walltime > 10.0:
        return (
            "complete: "
            f"efficiency_CHEAPER_{ratio}x_but_NOT_PARITY_energy{energy}_llm{llm}_"
            "honest_partial"
        )
    return f"complete: efficiency_NOT_DECISIVELY_CHEAPER_ratio{ratio}_energy{energy}_llm{llm}"


def _per_corpus_results(
    bundle: CorpusBundle,
    measured: CostMeasurements,
    *,
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    offset = 0
    for source_index, source in enumerate(bundle.corpus_sources):
        n_source = int(source["n_items"])
        labels = bundle.labels[offset : offset + n_source]
        energy_scores = measured.energy_scores[offset : offset + n_source]
        llm_scores = measured.llm_scores[offset : offset + n_source]
        results.append(
            {
                "name": source["name"],
                "n_items": n_source,
                "energy_auroc": _auroc(labels, energy_scores),
                "energy_auroc_ci95": bootstrap_ci95(
                    labels,
                    energy_scores,
                    seed=config.random_seed + 101 + source_index,
                    resamples=config.bootstrap_resamples,
                ),
                "llm_judge_auroc": _auroc(labels, llm_scores),
                "llm_judge_auroc_ci95": bootstrap_ci95(
                    labels,
                    llm_scores,
                    seed=config.random_seed + 201 + source_index,
                    resamples=config.bootstrap_resamples,
                ),
            }
        )
        offset += n_source
    return results


def _per_item_results(bundle: CorpusBundle, measured: CostMeasurements) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "corpus_source": item.get("corpus_source", "fixture"),
            "source_index": item.get("source_index", index),
            "label": item.get("label", "incorrect" if int(item["gold_error"]) else "correct"),
            "gold_error": int(item["gold_error"]),
            "question_id": item.get("question_id"),
            "corpus_item_id": item.get("corpus_item_id"),
            "energy_score": float(measured.energy_scores[index]),
            "llm_judge_score": float(measured.llm_scores[index]),
        }
        for index, item in enumerate(bundle.items)
    ]


def build_artifact(
    *,
    config: ExperimentConfig,
    bundle: CorpusBundle,
    measured: CostMeasurements,
    preconditions_checked: Sequence[PreconditionCheck],
    model_specs: dict[str, object],
    gguf_harness_source: dict[str, object],
    cost_harness_source: dict[str, object],
) -> dict[str, object]:
    """Build the terminal Exp 3917 artifact from measured scores and costs."""

    started_at = config.start_time()
    finished_at = config.clock()
    duration_s = finished_at - started_at
    energy_auroc = float(measured.energy_cost["auroc"])
    llm_auroc = float(measured.llm_cost["auroc"])
    energy_ci95 = bootstrap_ci95(
        bundle.labels,
        measured.energy_scores,
        seed=config.random_seed + 11,
        resamples=config.bootstrap_resamples,
    )
    llm_ci95 = bootstrap_ci95(
        bundle.labels,
        measured.llm_scores,
        seed=config.random_seed + 17,
        resamples=config.bootstrap_resamples,
    )
    energy_ms = float(measured.energy_cost["per_item_wall_ms"])
    llm_ms = float(measured.llm_cost["per_item_wall_ms"])
    ratio_wall = _cost_ratio(llm_ms, energy_ms)
    ratio_flops = _cost_ratio(
        float(measured.llm_cost["est_flops"]),
        float(measured.energy_cost["est_flops"]),
    )
    if ratio_wall is None or ratio_flops is None:
        raise ValueError("cost ratios require positive energy denominator")
    parity = _accuracy_parity(energy_auroc, llm_ci95)
    verdict = _classify_verdict(
        accuracy_parity=parity,
        cost_ratio_walltime=ratio_wall,
        energy_auroc=energy_auroc,
        llm_auroc=llm_auroc,
    )
    checksum_payload = {
        "experiment": EXPERIMENT_ID,
        "bundle_checksum": bundle.checksum,
        "model_specs": model_specs,
        "gguf_harness_source": gguf_harness_source,
        "cost_harness_source": cost_harness_source,
        "random_seed": config.random_seed,
        "energy_cost": measured.energy_cost,
        "llm_cost": measured.llm_cost,
        "energy_scores_digest": _score_digest(measured.energy_scores),
        "llm_scores_digest": _score_digest(measured.llm_scores),
        "energy_ci95": energy_ci95,
        "llm_ci95": llm_ci95,
    }
    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "run_date": datetime.fromtimestamp(finished_at, tz=UTC).strftime("%Y%m%d"),
        "started_at": datetime.fromtimestamp(started_at, tz=UTC).isoformat().replace("+00:00", "Z"),
        "finished_at": datetime.fromtimestamp(finished_at, tz=UTC).isoformat().replace("+00:00", "Z"),
        "honest_verdict": verdict,
        "status": verdict,
        "energy_auroc": energy_auroc,
        "energy_auroc_ci95": energy_ci95,
        "llm_judge_auroc": llm_auroc,
        "llm_judge_auroc_ci95": llm_ci95,
        "accuracy_parity": parity,
        "cost_ratio_walltime": float(ratio_wall),
        "cost_ratio_flops": float(ratio_flops),
        "energy_per_item_ms": energy_ms,
        "llm_per_item_ms": llm_ms,
        "llm_judge_model_used": model_specs.get("model_used"),
        "n_items": bundle.n_items,
        "corpus_sources": list(bundle.corpus_sources),
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs,
        "gguf_harness_source": gguf_harness_source,
        "cost_harness_source": cost_harness_source,
        "random_seed": config.random_seed,
        "random_seeds_used": {
            "fover_slice": config.random_seed,
            "bootstrap_energy": config.random_seed + 11,
            "bootstrap_llm_judge": config.random_seed + 17,
        },
        "reproducibility_checksum": _checksum(checksum_payload),
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "energy_cost": measured.energy_cost,
        "llm_judge_cost": measured.llm_cost,
        "per_corpus_results": _per_corpus_results(bundle, measured, config=config),
        "per_item_results": _per_item_results(bundle, measured),
        "score_digests": {
            "energy_scores_sha256": _score_digest(measured.energy_scores),
            "llm_judge_scores_sha256": _score_digest(measured.llm_scores),
            "corpus_bundle_sha256": bundle.checksum,
        },
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
    model_specs: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a non-fabricated blocked artifact with empty headline metrics."""

    artifact: dict[str, object] = {
        "experiment": EXPERIMENT_ID,
        "title": TITLE,
        "honest_verdict": reason,
        "status": reason,
        "energy_auroc": None,
        "energy_auroc_ci95": None,
        "llm_judge_auroc": None,
        "llm_judge_auroc_ci95": None,
        "accuracy_parity": None,
        "cost_ratio_walltime": None,
        "cost_ratio_flops": None,
        "energy_per_item_ms": None,
        "llm_per_item_ms": None,
        "llm_judge_model_used": None,
        "n_items": 0,
        "corpus_sources": [],
        "preconditions_checked": [check.as_dict() for check in preconditions_checked],
        "model_specs": model_specs or {},
        "random_seed": DEFAULT_RANDOM_SEED,
        "random_seeds_used": {},
        "reproducibility_checksum": _checksum(
            {
                "experiment": EXPERIMENT_ID,
                "reason": reason,
                "preconditions_checked": [check.as_dict() for check in preconditions_checked],
                "model_specs": model_specs or {},
            }
        ),
        "duration_s": duration_s,
        "inference_substrate": "none_blocked_preflight",
        "energy_cost": None,
        "llm_judge_cost": None,
        "per_corpus_results": [],
        "per_item_results": [],
        "score_digests": {},
        "field_principles": FIELD_PRINCIPLES,
    }
    validate_artifact(artifact)
    return artifact


def _terminal_verdict(verdict: str) -> bool:
    return verdict.startswith(("complete:", "blocked_"))


def validate_artifact(artifact: dict[str, object]) -> None:
    """Validate required Exp 3917 fields and bare-scalar discipline."""

    missing = sorted(REQUIRED_FIELDS - artifact.keys())
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    if not _terminal_verdict(verdict):
        raise ValueError(f"honest_verdict lacks terminal prefix: {verdict}")
    for key in WRAPPED_VALUE_FORBIDDEN_FIELDS:
        if isinstance(artifact.get(key), dict):
            raise ValueError(f"{key} must not be a value/principle wrapper")
    if not isinstance(artifact["duration_s"], (int, float)):
        raise ValueError("duration_s must be a bare number")
    if len(str(artifact["reproducibility_checksum"])) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if verdict.startswith("blocked_"):
        if artifact["n_items"] != 0:
            raise ValueError("blocked artifacts must not claim item counts")
        return
    for key in (
        "energy_auroc",
        "llm_judge_auroc",
        "cost_ratio_walltime",
        "cost_ratio_flops",
        "energy_per_item_ms",
        "llm_per_item_ms",
    ):
        if not isinstance(artifact[key], float):
            raise ValueError(f"{key} must be a bare float")
    if not isinstance(artifact["accuracy_parity"], bool):
        raise ValueError("accuracy_parity must be a bare bool")
    if not isinstance(artifact["n_items"], int) or int(artifact["n_items"]) <= 0:
        raise ValueError("n_items must be a positive bare int")


def write_artifact(output_path: Path, artifact: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_exp3915_gguf_harness_source(repo_root: Path) -> dict[str, object]:
    """Load and validate the robust GGUF readiness artifact from Exp 3915."""

    artifact_path = repo_root / EXP3915_ARTIFACT_REL_PATH
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{EXP3915_ARTIFACT_REL_PATH.as_posix()} is missing")
    artifact = _read_json(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("exp3915 artifact is not a JSON object")
    if artifact.get("unit_test_passed") is not True:
        raise ValueError("exp3915 unit_test_passed is not true")
    smoke_tokens = int(artifact.get("smoke_tokens") or 0)
    if smoke_tokens <= 0:
        raise ValueError("exp3915 smoke_tokens must be >0")
    module_path = _resolve_repo_path(repo_root, str(artifact.get("harness_module_path") or ""))
    if not module_path.is_file():
        raise FileNotFoundError(f"exp3915 harness module missing: {module_path}")
    return {
        "artifact_path": EXP3915_ARTIFACT_REL_PATH.as_posix(),
        "artifact_sha256": _sha256_file(artifact_path),
        "harness_module_path": artifact.get("harness_module_path"),
        "harness_module_sha256": _sha256_file(module_path),
        "unit_test_passed": True,
        "smoke_tokens": smoke_tokens,
        "model_used": artifact.get("model_used"),
        "n_gpu_layers_used": artifact.get("n_gpu_layers_used"),
        "model_specs": artifact.get("model_specs", {}),
        "honest_verdict": artifact.get("honest_verdict"),
    }


def load_exp3905_cost_harness_source(repo_root: Path) -> dict[str, object]:
    """Load and validate the cost instrumentation artifact and importable module."""

    artifact_path = repo_root / EXP3905_ARTIFACT_REL_PATH
    if not artifact_path.is_file():
        raise FileNotFoundError(f"{EXP3905_ARTIFACT_REL_PATH.as_posix()} is missing")
    artifact = _read_json(artifact_path)
    if not isinstance(artifact, dict):
        raise ValueError("exp3905 artifact is not a JSON object")
    module_path = _resolve_repo_path(repo_root, str(artifact.get("harness_module_path") or COST_HARNESS_MODULE_PATH))
    if not module_path.is_file():
        raise FileNotFoundError(f"exp3905 harness module missing: {module_path}")
    importlib.import_module("carnot.verify.cost_instrumented_verification")
    return {
        "artifact_path": EXP3905_ARTIFACT_REL_PATH.as_posix(),
        "artifact_sha256": _sha256_file(artifact_path),
        "harness_module_path": artifact.get("harness_module_path", COST_HARNESS_MODULE_PATH),
        "harness_module_sha256": _sha256_file(module_path),
        "unit_test_passed": artifact.get("unit_test_passed"),
        "honest_verdict": artifact.get("honest_verdict"),
    }


def _probe_cuda_with_venv(config: ExperimentConfig) -> PreconditionCheck:
    try:
        proc = subprocess.run(
            [str(config.venv_python()), "-c", "import torch; assert torch.cuda.is_available()"],
            capture_output=True,
            text=True,
            timeout=config.cuda_probe_timeout_s,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - subprocess launch failures are environment-specific.
        return PreconditionCheck("cuda_available", False, repr(exc))
    detail = (proc.stdout or proc.stderr or f"returncode={proc.returncode}").strip()
    return PreconditionCheck("cuda_available", proc.returncode == 0, detail)


def probe_preconditions(
    config: ExperimentConfig,
    *,
    cuda_probe: Callable[[ExperimentConfig], PreconditionCheck] = _probe_cuda_with_venv,
) -> tuple[
    tuple[PreconditionCheck, ...],
    str | None,
    dict[str, object],
    dict[str, object] | None,
    dict[str, object] | None,
]:
    """Check hard resources before loading the full GGUF judge."""

    checks: list[PreconditionCheck] = [cuda_probe(config)]

    gguf_harness_source: dict[str, object] | None = None
    try:
        gguf_harness_source = load_exp3915_gguf_harness_source(config.repo_root)
        checks.append(
            PreconditionCheck(
                "exp3915_gguf_harness_ready",
                True,
                f"model={gguf_harness_source.get('model_used')} smoke_tokens={gguf_harness_source.get('smoke_tokens')}",
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("exp3915_gguf_harness_ready", False, repr(exc)))

    cost_harness_source: dict[str, object] | None = None
    try:
        cost_harness_source = load_exp3905_cost_harness_source(config.repo_root)
        checks.append(
            PreconditionCheck(
                "exp3905_cost_harness_importable",
                True,
                str(cost_harness_source.get("harness_module_path")),
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("exp3905_cost_harness_importable", False, repr(exc)))

    try:
        bundle = load_labeled_corpora(
            config.repo_root,
            random_seed=config.random_seed,
            fover_min_items=config.fover_min_items,
        )
        checks.append(
            PreconditionCheck(
                "labeled_corpora_ready",
                True,
                f"items={bundle.n_items} sources={len(bundle.corpus_sources)}",
            )
        )
    except Exception as exc:
        checks.append(PreconditionCheck("labeled_corpora_ready", False, repr(exc)))

    available = {check.resource: check.available for check in checks}
    blocked_reason = None
    if not available.get("cuda_available", False):
        blocked_reason = "blocked_no_cuda"
    elif not available.get("exp3915_gguf_harness_ready", False):
        blocked_reason = "blocked_upstream_gguf_harness_not_ready"
    elif not available.get("exp3905_cost_harness_importable", False):
        blocked_reason = "blocked_upstream_cost_harness_not_ready"
    elif not available.get("labeled_corpora_ready", False):
        blocked_reason = "blocked_labeled_corpora_not_ready"

    model_specs = {
        "prefer_order": _prefer_order_from_source(gguf_harness_source or {}),
        "source_exp3915_model_used": (gguf_harness_source or {}).get("model_used"),
        "source_exp3915_n_gpu_layers_used": (gguf_harness_source or {}).get("n_gpu_layers_used"),
        "max_tokens": config.max_tokens,
        "n_ctx": config.n_ctx,
    }
    return tuple(checks), blocked_reason, model_specs, gguf_harness_source, cost_harness_source


def run_experiment(config: ExperimentConfig | None = None, *, write: bool = True) -> dict[str, object]:
    """Run Exp 3917 end to end, or write a blocked artifact on failed gates."""

    config = config or ExperimentConfig(repo_root=Path(__file__).resolve().parents[3])
    started = config.start_time()
    active_config = replace(config, started_at=started)
    output_path = active_config.resolved_output_path()
    checks, blocked_reason, preflight_model_specs, gguf_source, cost_source = probe_preconditions(
        active_config
    )
    if blocked_reason is not None:
        artifact = build_blocked_artifact(
            reason=blocked_reason,
            preconditions_checked=checks,
            duration_s=active_config.clock() - started,
            model_specs=preflight_model_specs,
        )
        if write:
            write_artifact(output_path, artifact)
        return artifact

    bundle = load_labeled_corpora(
        active_config.repo_root,
        random_seed=active_config.random_seed,
        fover_min_items=active_config.fover_min_items,
    )
    generator, loaded_model_specs = load_robust_generator(gguf_source or {}, active_config)
    model_specs = {**preflight_model_specs, **loaded_model_specs}
    measured = measure_head_to_head_costs(
        bundle.items,
        generator=generator,
        model_specs=model_specs,
        max_tokens=active_config.max_tokens,
    )
    duration_s = active_config.clock() - started
    if duration_s < DURATION_FLOOR_S:
        checks = (
            *checks,
            PreconditionCheck(
                "llm_judge_duration_floor",
                False,
                f"duration_s={duration_s:.6f} < {DURATION_FLOOR_S:.1f}",
            ),
        )
        artifact = build_blocked_artifact(
            reason="blocked_llm_judge_not_invoked",
            preconditions_checked=checks,
            duration_s=duration_s,
            model_specs=model_specs,
        )
    else:
        artifact = build_artifact(
            config=active_config,
            bundle=bundle,
            measured=measured,
            preconditions_checked=checks,
            model_specs=model_specs,
            gguf_harness_source=gguf_source or {},
            cost_harness_source=cost_source or {},
        )
    if write:
        write_artifact(output_path, artifact)
    return artifact


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run_experiment(
        ExperimentConfig(repo_root=args.repo_root, output_path=args.output_path),
        write=True,
    )
    output_path = args.output_path if args.output_path is not None else args.repo_root / OUTPUT_REL_PATH
    print(f"{output_path.name} wrote {artifact['honest_verdict']}")
    return 0 if str(artifact["honest_verdict"]).startswith("complete:") else 1
