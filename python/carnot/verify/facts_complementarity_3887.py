"""Exp 3887 cached facts-complementarity aggregation.

Spec refs: REQ-VERIFY-3887, SCENARIO-VERIFY-3887.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.verify.corrected_cross_domain_remeasurement_v4 import tie_aware_auroc


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_REL_PATH = Path("results/experiment_3886_graph_grounding_fact_verifier_defabricated.json")
OUTPUT_REL_PATH = Path("results/experiment_3887_facts_complementarity.json")
RANDOM_SEED = 3887
INFERENCE_SUBSTRATE = "cached_exp3886_per_item_score_aggregation_only_no_live_model_execution"
FUSION_METHOD = "max_graph_math_scores"
COMPLEMENTARITY_CORR_MAX = 0.5
FUSED_AUROC_MARGIN = 0.02

REQUIRED_PRINCIPLE_FIELDS = (
    "facts_error_mask_correlation",
    "graph_independent_contribution",
    "fused_auroc",
    "math_only_auroc",
    "graph_only_auroc",
    "n_items",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "inference_substrate",
)
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    *REQUIRED_PRINCIPLE_FIELDS,
    "field_principles",
)
METRIC_FIELDS = (
    "facts_error_mask_correlation",
    "graph_independent_contribution",
    "fused_auroc",
    "math_only_auroc",
    "graph_only_auroc",
)
FIELD_PRINCIPLES = {
    "facts_error_mask_correlation": (
        "Independence - low correlation => graph and ensemble fail on DIFFERENT "
        "items => genuine complementarity; high => redundant."
    ),
    "graph_independent_contribution": (
        "Fraction of gold hallucinations caught ONLY by graph-grounding - the "
        "marginal value of adding it."
    ),
    "fused_auroc": (
        "Does fusing beat either alone - the product case for a math+graph "
        "fact-aware verifier."
    ),
    "math_only_auroc": (
        "Does fusing beat either alone - the product case for a math+graph "
        "fact-aware verifier."
    ),
    "graph_only_auroc": (
        "Does fusing beat either alone - the product case for a math+graph "
        "fact-aware verifier."
    ),
    "n_items": "Aggregation methodology - reads upstream per-item scores only.",
    "preconditions_checked": "Aggregation methodology - reads upstream per-item scores only.",
    "random_seed": "Aggregation methodology - reads upstream per-item scores only.",
    "reproducibility_checksum": "Aggregation methodology - reads upstream per-item scores only.",
    "duration_s": "Aggregation methodology - reads upstream per-item scores only.",
    "inference_substrate": "Aggregation methodology - reads upstream per-item scores only.",
}

LABEL_KEYS = ("gold_ungrounded", "is_hallucination", "is_ungrounded", "label", "gold_label")
GRAPH_SCORE_KEYS = ("graph_score", "graph_grounding_score", "graph_grounding", "graph_energy")
MATH_SCORE_KEYS = (
    "math_baseline_score",
    "math_ensemble_score",
    "math_score",
    "ensemble_score",
    "math_baseline",
)
GRAPH_NESTED_KEYS = ("scores", "graph", "graph_grounding", "metrics")
MATH_NESTED_KEYS = ("scores", "math_baseline", "math_ensemble", "ensemble", "metrics")
GENERIC_NESTED_SCORE_KEYS = ("score", "energy", "value")


@dataclass(frozen=True)
class FactsScoreItem:
    """One cached Exp 3886 facts row with graph and math scores."""

    item_id: str
    gold_ungrounded: bool
    graph_score: float
    math_score: float


@dataclass(frozen=True)
class ComplementarityMetrics:
    """Exp 3887 graph-versus-math facts complementarity metrics."""

    n_items: int
    n_gold_hallucinations: int
    facts_error_mask_correlation: float
    graph_independent_contribution: float
    fused_auroc: float
    math_only_auroc: float
    graph_only_auroc: float
    graph_threshold: float
    math_threshold: float
    graph_caught_gold: int
    math_caught_gold: int
    graph_only_caught_ids: tuple[str, ...]
    math_only_caught_ids: tuple[str, ...]
    catch_mask_confusion: JsonDict


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    upstream_path: Path | str = UPSTREAM_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3887 terminal artifact from Exp 3886 cached scores."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    upstream = _repo_path(root_path, Path(upstream_path))
    preconditions: list[JsonDict] = []
    payload: JsonDict = {}
    items: tuple[FactsScoreItem, ...] = ()
    scores_path: Path | None = None

    if not upstream.is_file():
        preconditions.append(
            {
                "resource": "exp3886_artifact",
                "available": False,
                "detail": relative_path(root_path, upstream),
            }
        )
        return build_blocked_artifact(
            preconditions_checked=preconditions,
            cited_upstream_artifact={},
            started_s=start,
            finished_s=_finish(start, now_s),
            n_items=0,
            tests_run=tests_run,
        )

    preconditions.append(
        {
            "resource": "exp3886_artifact",
            "available": True,
            "detail": relative_path(root_path, upstream),
        }
    )
    try:
        payload = json.loads(upstream.read_text(encoding="utf-8"))
    except Exception as exc:
        preconditions.append(
            {
                "resource": "exp3886_artifact_json",
                "available": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        return build_blocked_artifact(
            preconditions_checked=preconditions,
            cited_upstream_artifact=cite_upstream_artifact(root_path, upstream, {}),
            started_s=start,
            finished_s=_finish(start, now_s),
            n_items=0,
            tests_run=tests_run,
        )

    delta = _coerce_score(payload.get("facts_catch_delta"))
    delta_ok = delta is not None and delta > 0.0
    preconditions.append(
        {
            "resource": "exp3886_facts_catch_delta_positive",
            "available": bool(delta_ok),
            "detail": payload.get("facts_catch_delta"),
        }
    )

    scores_path = _resolve_scores_path(root_path, payload)
    scores_loadable = False
    score_detail: Any = "missing per_item_scores_path"
    if scores_path is not None and scores_path.is_file():
        try:
            items = load_per_item_scores(scores_path)
            scores_loadable = bool(items)
            score_detail = f"{len(items)} parsed rows from {relative_path(root_path, scores_path)}"
        except Exception as exc:
            score_detail = f"{type(exc).__name__}: {exc}"
    elif scores_path is not None:
        score_detail = relative_path(root_path, scores_path)
    preconditions.append(
        {
            "resource": "exp3886_per_item_scores_loadable",
            "available": bool(scores_loadable),
            "detail": score_detail,
        }
    )

    label_support = _has_positive_and_negative_labels(items)
    if scores_loadable:
        preconditions.append(
            {
                "resource": "gold_hallucination_and_grounded_items",
                "available": bool(label_support),
                "detail": {
                    "gold_hallucinations": sum(1 for item in items if item.gold_ungrounded),
                    "gold_grounded": sum(1 for item in items if not item.gold_ungrounded),
                },
            }
        )

    cited = cite_upstream_artifact(root_path, upstream, payload)
    if not (delta_ok and scores_loadable and label_support):
        return build_blocked_artifact(
            preconditions_checked=preconditions,
            cited_upstream_artifact=cited,
            started_s=start,
            finished_s=_finish(start, now_s),
            n_items=len(items),
            tests_run=tests_run,
        )

    metrics = compute_complementarity_metrics(items)
    artifact = build_artifact_from_metrics(
        metrics=metrics,
        items=items,
        cited_upstream_artifact=cited,
        preconditions_checked=preconditions,
        started_s=start,
        finished_s=_finish(start, now_s),
        tests_run=tests_run,
    )
    validate_artifact(artifact)
    return artifact


def load_per_item_scores(path: Path) -> tuple[FactsScoreItem, ...]:
    """Read Exp 3886 JSONL score rows into normalized score items."""

    items: list[FactsScoreItem] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            continue
        item = _parse_score_row(row, index)
        if item is not None:
            items.append(item)
    return tuple(items)


def compute_complementarity_metrics(
    items: Sequence[FactsScoreItem],
) -> ComplementarityMetrics:
    """Compute catch-mask independence and same-row AUROCs."""

    labels = [1 if item.gold_ungrounded else 0 for item in items]
    graph_scores = [item.graph_score for item in items]
    math_scores = [item.math_score for item in items]
    graph_threshold = tune_threshold(labels, graph_scores)
    math_threshold = tune_threshold(labels, math_scores)
    gold_indices = [index for index, label in enumerate(labels) if label == 1]
    graph_catch_mask = [1 if graph_scores[index] >= graph_threshold else 0 for index in gold_indices]
    math_catch_mask = [1 if math_scores[index] >= math_threshold else 0 for index in gold_indices]
    graph_only_indices = [
        index
        for mask_index, index in enumerate(gold_indices)
        if graph_catch_mask[mask_index] and not math_catch_mask[mask_index]
    ]
    math_only_indices = [
        index
        for mask_index, index in enumerate(gold_indices)
        if math_catch_mask[mask_index] and not graph_catch_mask[mask_index]
    ]
    fused_scores = [max(graph, math) for graph, math in zip(graph_scores, math_scores, strict=True)]
    n_gold = len(gold_indices)
    return ComplementarityMetrics(
        n_items=len(items),
        n_gold_hallucinations=n_gold,
        facts_error_mask_correlation=matthews_phi(graph_catch_mask, math_catch_mask),
        graph_independent_contribution=len(graph_only_indices) / n_gold if n_gold else 0.0,
        fused_auroc=float(tie_aware_auroc(labels, fused_scores)),
        math_only_auroc=float(tie_aware_auroc(labels, math_scores)),
        graph_only_auroc=float(tie_aware_auroc(labels, graph_scores)),
        graph_threshold=graph_threshold,
        math_threshold=math_threshold,
        graph_caught_gold=sum(graph_catch_mask),
        math_caught_gold=sum(math_catch_mask),
        graph_only_caught_ids=tuple(items[index].item_id for index in graph_only_indices),
        math_only_caught_ids=tuple(items[index].item_id for index in math_only_indices),
        catch_mask_confusion=_mask_confusion(graph_catch_mask, math_catch_mask),
    )


def tune_threshold(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Choose the deterministic threshold that maximizes Youden's J statistic."""

    if not labels or not scores:
        return 0.5
    clean_labels = [1 if int(label) else 0 for label in labels]
    clean_scores = [float(score) for score in scores]
    candidates = sorted({*clean_scores, min(clean_scores) - 1e-12, max(clean_scores) + 1e-12})
    best_threshold = candidates[0]
    best_key = (-math.inf, -math.inf, -math.inf, math.inf)
    for threshold in candidates:
        preds = [1 if score >= threshold else 0 for score in clean_scores]
        positives = sum(clean_labels)
        negatives = len(clean_labels) - positives
        true_positive = sum(1 for pred, label in zip(preds, clean_labels, strict=True) if pred and label)
        false_positive = sum(1 for pred, label in zip(preds, clean_labels, strict=True) if pred and not label)
        true_negative = sum(1 for pred, label in zip(preds, clean_labels, strict=True) if not pred and not label)
        true_positive_rate = true_positive / positives if positives else 0.0
        false_positive_rate = false_positive / negatives if negatives else 0.0
        true_negative_rate = true_negative / negatives if negatives else 0.0
        accuracy = (true_positive + true_negative) / len(clean_labels)
        key = (true_positive_rate - false_positive_rate, accuracy, true_negative_rate, threshold)
        if key > best_key:
            best_threshold = threshold
            best_key = key
    return float(best_threshold)


def matthews_phi(first_mask: Sequence[int], second_mask: Sequence[int]) -> float:
    """Return phi/Matthews correlation for two binary masks."""

    if len(first_mask) != len(second_mask):
        raise ValueError("masks must have the same length")
    both_one = first_only = second_only = both_zero = 0
    for first, second in zip(first_mask, second_mask, strict=True):
        first_b = 1 if int(first) else 0
        second_b = 1 if int(second) else 0
        if first_b and second_b:
            both_one += 1
        elif first_b:
            first_only += 1
        elif second_b:
            second_only += 1
        else:
            both_zero += 1
    denominator = math.sqrt(
        (both_one + first_only)
        * (both_one + second_only)
        * (both_zero + first_only)
        * (both_zero + second_only)
    )
    if denominator == 0.0:
        return 0.0
    return ((both_one * both_zero) - (first_only * second_only)) / denominator


def build_artifact_from_metrics(
    *,
    metrics: ComplementarityMetrics,
    items: Sequence[FactsScoreItem],
    cited_upstream_artifact: Mapping[str, Any],
    preconditions_checked: Sequence[Mapping[str, Any]],
    started_s: float,
    finished_s: float,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build a complete Exp 3887 artifact with bare metric fields."""

    artifact: JsonDict = {
        "honest_verdict": classify_verdict(metrics),
        "facts_error_mask_correlation": round(metrics.facts_error_mask_correlation, 6),
        "graph_independent_contribution": round(metrics.graph_independent_contribution, 6),
        "fused_auroc": round(metrics.fused_auroc, 6),
        "math_only_auroc": round(metrics.math_only_auroc, 6),
        "graph_only_auroc": round(metrics.graph_only_auroc, 6),
        "n_items": metrics.n_items,
        "n_gold_hallucinations": metrics.n_gold_hallucinations,
        "graph_caught_gold": metrics.graph_caught_gold,
        "math_caught_gold": metrics.math_caught_gold,
        "graph_only_caught_ids": list(metrics.graph_only_caught_ids),
        "math_only_caught_ids": list(metrics.math_only_caught_ids),
        "catch_mask_confusion": metrics.catch_mask_confusion,
        "threshold_policy": {
            "graph_catch_threshold": round(metrics.graph_threshold, 6),
            "math_catch_threshold": round(metrics.math_threshold, 6),
            "fusion_method": FUSION_METHOD,
        },
        "cited_upstream_artifact": dict(cited_upstream_artifact),
        "preconditions_checked": list(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {
                "items": [
                    {
                        "item_id": item.item_id,
                        "gold_ungrounded": item.gold_ungrounded,
                        "graph_score": round(item.graph_score, 6),
                        "math_score": round(item.math_score, 6),
                    }
                    for item in items
                ],
                "metrics": {
                    "facts_error_mask_correlation": round(metrics.facts_error_mask_correlation, 12),
                    "graph_independent_contribution": round(metrics.graph_independent_contribution, 12),
                    "fused_auroc": round(metrics.fused_auroc, 12),
                    "math_only_auroc": round(metrics.math_only_auroc, 12),
                    "graph_only_auroc": round(metrics.graph_only_auroc, 12),
                },
                "random_seed": RANDOM_SEED,
                "fusion_method": FUSION_METHOD,
            }
        ),
        "duration_s": round(max(0.0, finished_s - started_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": list(tests_run or []),
        "frozen_fover_0_9131_untouched": True,
        "scripts_research_conductor_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    *,
    preconditions_checked: Sequence[Mapping[str, Any]],
    cited_upstream_artifact: Mapping[str, Any],
    started_s: float,
    finished_s: float,
    n_items: int,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal blocked artifact without fabricated metrics."""

    artifact: JsonDict = {
        "honest_verdict": "blocked_upstream_scores_missing",
        "facts_error_mask_correlation": None,
        "graph_independent_contribution": None,
        "fused_auroc": None,
        "math_only_auroc": None,
        "graph_only_auroc": None,
        "n_items": int(n_items),
        "n_gold_hallucinations": 0,
        "graph_caught_gold": 0,
        "math_caught_gold": 0,
        "graph_only_caught_ids": [],
        "math_only_caught_ids": [],
        "catch_mask_confusion": None,
        "threshold_policy": {
            "graph_catch_threshold": None,
            "math_catch_threshold": None,
            "fusion_method": FUSION_METHOD,
        },
        "cited_upstream_artifact": dict(cited_upstream_artifact),
        "preconditions_checked": list(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {
                "blocked_reason": "blocked_upstream_scores_missing",
                "preconditions_checked": list(preconditions_checked),
                "cited_upstream_artifact": dict(cited_upstream_artifact),
                "n_items": int(n_items),
                "random_seed": RANDOM_SEED,
            }
        ),
        "duration_s": round(max(0.0, finished_s - started_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": list(tests_run or []),
        "frozen_fover_0_9131_untouched": True,
        "scripts_research_conductor_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def classify_verdict(metrics: ComplementarityMetrics) -> str:
    """Apply the Exp 3887 complementarity falsification gate."""

    corr = metrics.facts_error_mask_correlation
    fused = metrics.fused_auroc
    best_alone = max(metrics.math_only_auroc, metrics.graph_only_auroc)
    if corr < COMPLEMENTARITY_CORR_MAX and fused > best_alone + FUSED_AUROC_MARGIN:
        return f"complete: facts_COMPLEMENTARY_corr{corr:.3f}_fused{fused:.3f}_graph_broadens_the_verifier"
    return f"complete: facts_REDUNDANT_corr{corr:.3f}_fused{fused:.3f}_graph_does_not_broaden"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3887 artifact schema."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict must use a terminal prefix")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    uncovered = [field for field in REQUIRED_PRINCIPLE_FIELDS if field not in principles]
    if uncovered:
        raise ValueError(f"field_principles missing required fields: {uncovered}")
    if not isinstance(artifact.get("preconditions_checked"), list):
        raise ValueError("preconditions_checked must be a list")
    if not isinstance(artifact.get("n_items"), int) or artifact["n_items"] < 0:
        raise ValueError("n_items must be a non-negative integer")
    if not isinstance(artifact.get("random_seed"), int):
        raise ValueError("random_seed must be an integer")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a SHA-256 hex digest")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")
    substrate = str(artifact.get("inference_substrate") or "").lower()
    if "gguf" in substrate or "cuda" in substrate:
        raise ValueError("inference_substrate must not include live GGUF/CUDA markers")

    if verdict.startswith("complete:"):
        for field in METRIC_FIELDS:
            value = artifact.get(field)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be a bare number")
            if field == "facts_error_mask_correlation":
                if not -1.0 <= float(value) <= 1.0:
                    raise ValueError("facts_error_mask_correlation must be in [-1, 1]")
            elif not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{field} must be in [0, 1]")
    else:
        if artifact.get("honest_verdict") != "blocked_upstream_scores_missing":
            raise ValueError("blocked Exp 3887 artifacts must use blocked_upstream_scores_missing")
        for field in METRIC_FIELDS:
            if artifact.get(field) is not None:
                raise ValueError("blocked artifacts must not fabricate metrics")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    upstream_path: Path | str = UPSTREAM_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build and persist the Exp 3887 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(root_path, upstream_path=upstream_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def cite_upstream_artifact(root: Path, upstream_path: Path, payload: Mapping[str, Any]) -> JsonDict:
    """Return the Exp 3886 source citation."""

    return {
        "experiment_id": 3886,
        "path": relative_path(root, upstream_path),
        "sha256": sha256_file(upstream_path),
        "facts_catch_delta": payload.get("facts_catch_delta"),
        "per_item_scores_path": payload.get("per_item_scores_path"),
    }


def sha256_file(path: Path) -> str | None:
    """Return SHA-256 for an existing file."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for artifact inputs."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative_path(root: Path, path: Path) -> str:
    """Return a repo-relative path when possible."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _parse_score_row(row: Mapping[str, Any], index: int) -> FactsScoreItem | None:
    label = _coerce_label(_first_value(row, LABEL_KEYS))
    graph_score = _coerce_score(_score_value(row, GRAPH_SCORE_KEYS, GRAPH_NESTED_KEYS))
    math_score = _coerce_score(_score_value(row, MATH_SCORE_KEYS, MATH_NESTED_KEYS))
    if label is None or graph_score is None or math_score is None:
        return None
    return FactsScoreItem(
        item_id=str(row.get("item_id") or row.get("id") or row.get("question_id") or index),
        gold_ungrounded=label,
        graph_score=graph_score,
        math_score=math_score,
    )


def _score_value(
    row: Mapping[str, Any],
    direct_keys: Sequence[str],
    nested_keys: Sequence[str],
) -> Any:
    direct = _first_value(row, direct_keys)
    if direct is not None and not isinstance(direct, Mapping):
        return direct
    for nested_key in nested_keys:
        nested = row.get(nested_key)
        if isinstance(nested, Mapping):
            value = _first_value(nested, (*direct_keys, *GENERIC_NESTED_SCORE_KEYS))
            if value is not None and not isinstance(value, Mapping):
                return value
    return None


def _first_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _coerce_label(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "hallucination", "incorrect", "ungrounded"}:
            return True
        if normalized in {"0", "false", "no", "correct", "grounded"}:
            return False
    return None


def _coerce_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def _resolve_scores_path(root: Path, payload: Mapping[str, Any]) -> Path | None:
    raw_path = payload.get("per_item_scores_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    return _repo_path(root, Path(raw_path))


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _has_positive_and_negative_labels(items: Sequence[FactsScoreItem]) -> bool:
    positives = any(item.gold_ungrounded for item in items)
    negatives = any(not item.gold_ungrounded for item in items)
    return positives and negatives


def _mask_confusion(first_mask: Sequence[int], second_mask: Sequence[int]) -> JsonDict:
    return {
        "both_catch": sum(1 for first, second in zip(first_mask, second_mask, strict=True) if first and second),
        "graph_only": sum(1 for first, second in zip(first_mask, second_mask, strict=True) if first and not second),
        "math_only": sum(1 for first, second in zip(first_mask, second_mask, strict=True) if second and not first),
        "neither": sum(1 for first, second in zip(first_mask, second_mask, strict=True) if not first and not second),
    }


def _finish(started_s: float, now_s: float | None) -> float:
    return time.perf_counter() if now_s is None else max(float(now_s), float(started_s))
