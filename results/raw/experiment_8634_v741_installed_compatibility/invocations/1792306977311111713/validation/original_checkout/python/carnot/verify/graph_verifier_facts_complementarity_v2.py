"""Cached facts-domain complementarity audit for Exp 3863.

This module does not score a model and does not re-run the Exp 3862 graph
prototype. It reads the Exp 3862 JSON artifact, requires cached per-item facts
labels plus graph and math-ensemble scores, then measures whether the graph
verifier catches gold-ungrounded claims that the math-bound ensemble misses.

Spec: REQ-VERIFY-3863, SCENARIO-VERIFY-3863.
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


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_REL_PATH = Path(
    "results/experiment_3862_graph_grounding_fact_verifier_prototype_v2.json"
)
OUTPUT_REL_PATH = Path(
    "results/experiment_3863_graph_verifier_facts_complementarity_v2.json"
)
RANDOM_SEED = 3863
GRAPH_CATCH_THRESHOLD = 0.5
ENSEMBLE_CATCH_THRESHOLD = 0.5
MATERIAL_GRAPH_CATCH_MIN = 1
LOW_ERROR_MASK_CORRELATION_MAX = 0.35
INFERENCE_SUBSTRATE = (
    "cached_exp3862_json_scores_only_no_gpu "
    "(principle: consumes stored facts labels and verifier scores; no live LLM, "
    "no GGUF, no GPU scoring, and no frozen FoVer 0.9131 movement)."
)

REQUIRED_PRINCIPLE_FIELDS = (
    "honest_verdict",
    "graph_catches_ensemble_misses",
    "facts_error_mask_correlation",
    "union_facts_catch_rate",
    "extended_ensemble_recommended",
    "n_facts_items",
    "cited_upstream_artifacts",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "inference_substrate",
    "duration_s",
)

REQUIRED_ARTIFACT_FIELDS = REQUIRED_PRINCIPLE_FIELDS + (
    "field_principles",
    "ensemble_catches_graph_misses",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "principle: Terminal prefix records complementarity, redundancy, or the "
        "exact blocked resource."
    ),
    "graph_catches_ensemble_misses": (
        "principle: The graph verifier's independent facts contribution over "
        "the math ensemble."
    ),
    "facts_error_mask_correlation": (
        "principle: Decorrelation on the facts domain -- low => a genuinely "
        "additive verifier, not a redundant one."
    ),
    "union_facts_catch_rate": (
        "principle: Best-of-both on facts; the lift over the math-ensemble-alone "
        "is the value of adding the graph verifier."
    ),
    "extended_ensemble_recommended": (
        "principle: Bare bool -- true iff graph_catches_ensemble_misses material "
        "AND facts_error_mask_correlation low; the forward-path signal for the "
        "capstone."
    ),
    "n_facts_items": (
        "principle: Adversarial-Verify + Inference-Substrate methodology; "
        "verifier-scoring substrate."
    ),
    "cited_upstream_artifacts": (
        "principle: Adversarial-Verify + Inference-Substrate methodology; "
        "verifier-scoring substrate."
    ),
    "preconditions_checked": (
        "principle: Adversarial-Verify + Inference-Substrate methodology; "
        "verifier-scoring substrate."
    ),
    "random_seed": (
        "principle: Adversarial-Verify + Inference-Substrate methodology; "
        "verifier-scoring substrate."
    ),
    "reproducibility_checksum": (
        "principle: Adversarial-Verify + Inference-Substrate methodology; "
        "verifier-scoring substrate."
    ),
    "inference_substrate": (
        "principle: Adversarial-Verify + Inference-Substrate methodology; "
        "verifier-scoring substrate."
    ),
    "duration_s": (
        "principle: Adversarial-Verify + Inference-Substrate methodology; "
        "verifier-scoring substrate."
    ),
    "ensemble_catches_graph_misses": (
        "principle: Symmetric direction; exposes whether the math ensemble still "
        "covers facts misses unique to the graph verifier."
    ),
}

PER_ITEM_KEYS = (
    "per_item_scores",
    "facts_items",
    "facts_scores",
    "score_rows",
    "scored_items",
    "items",
    "rows",
)
GRAPH_SCORE_KEYS = (
    "graph_score",
    "graph_grounding",
    "graph_grounding_score",
    "graph_grounding_energy",
    "graph_energy",
)
ENSEMBLE_SCORE_KEYS = (
    "math_ensemble_score",
    "math_ensemble",
    "math_bound_ensemble_score",
    "ensemble_score",
    "math_score",
    "math_ensemble_energy",
)
LABEL_KEYS = (
    "gold_ungrounded",
    "is_hallucination",
    "is_ungrounded",
    "label",
    "gold_label",
)


@dataclass(frozen=True)
class FactsScoreItem:
    """One cached facts row with the two verifier scores needed by Exp 3863."""

    item_id: str
    gold_ungrounded: bool
    graph_score: float
    ensemble_score: float


@dataclass(frozen=True)
class ComplementarityMetrics:
    """Computed graph-versus-ensemble facts complementarity metrics."""

    n_facts_items: int
    n_gold_ungrounded: int
    graph_catches_ensemble_misses: int
    ensemble_catches_graph_misses: int
    graph_catch_rate: float
    ensemble_catch_rate: float
    union_facts_catch_rate: float
    union_lift_over_ensemble: float
    facts_error_mask_correlation: float
    extended_ensemble_recommended: bool
    error_mask_confusion: JsonDict
    graph_catches_ensemble_miss_ids: tuple[str, ...]
    ensemble_catches_graph_miss_ids: tuple[str, ...]


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    upstream_path: Path | str = UPSTREAM_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3863 terminal artifact from cached Exp 3862 scores."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    upstream = _repo_path(root_path, Path(upstream_path))
    cited = cite_upstream_artifact(root_path, upstream)
    preconditions: list[JsonDict] = []

    if not upstream.exists():
        return _blocked_from_preconditions(
            "missing exp3862 artifact with cached facts scores",
            preconditions
            + [
                {
                    "resource": "exp3862_artifact",
                    "available": False,
                    "detail": relative_path(root_path, upstream),
                }
            ],
            cited,
            start,
            now_s,
            tests_run,
        )

    preconditions.append(
        {
            "resource": "exp3862_artifact",
            "available": True,
            "detail": relative_path(root_path, upstream),
        }
    )
    try:
        payload = json.loads(upstream.read_text(encoding="utf-8"))
    except Exception as exc:
        return _blocked_from_preconditions(
            f"malformed exp3862 artifact: {type(exc).__name__}",
            preconditions
            + [
                {
                    "resource": "exp3862_json_parse",
                    "available": False,
                    "detail": str(exc),
                }
            ],
            cited,
            start,
            now_s,
            tests_run,
        )

    delta = _coerce_score(payload.get("facts_catch_delta"))
    delta_ok = delta is not None and delta > 0.0
    preconditions.append(
        {
            "resource": "exp3862_facts_catch_delta_positive",
            "available": bool(delta_ok),
            "detail": payload.get("facts_catch_delta"),
        }
    )
    if not delta_ok:
        return _blocked_from_preconditions(
            f"facts_catch_delta precondition failed: {payload.get('facts_catch_delta')}",
            preconditions,
            cited,
            start,
            now_s,
            tests_run,
        )

    items = parse_per_item_scores(payload)
    preconditions.append(
        {
            "resource": "exp3862_per_item_scores",
            "available": bool(items),
            "detail": f"{len(items)} parsed per-item graph/math facts rows",
        }
    )
    if not items:
        return _blocked_from_preconditions(
            "per-item graph + ensemble facts scores unavailable in exp3862 artifact",
            preconditions,
            cited,
            start,
            now_s,
            tests_run,
        )

    n_gold_ungrounded = sum(1 for item in items if item.gold_ungrounded)
    n_gold_grounded = len(items) - n_gold_ungrounded
    enough_label_support = n_gold_ungrounded > 0 and n_gold_grounded > 0
    preconditions.append(
        {
            "resource": "gold_ungrounded_and_grounded_items",
            "available": bool(enough_label_support),
            "detail": {
                "gold_ungrounded": n_gold_ungrounded,
                "gold_grounded": n_gold_grounded,
            },
        }
    )
    if not enough_label_support:
        return _blocked_from_preconditions(
            "gold_ungrounded_and_grounded_items required for facts error-mask correlation",
            preconditions,
            cited,
            start,
            now_s,
            tests_run,
        )

    metrics = compute_complementarity_metrics(items)
    finished = time.perf_counter() if now_s is None else float(now_s)
    artifact = build_artifact_from_metrics(
        metrics=metrics,
        items=items,
        cited_upstream_artifacts=cited,
        preconditions_checked=preconditions,
        started_s=start,
        finished_s=finished,
        tests_run=tests_run,
    )
    validate_artifact(artifact)
    return artifact


def parse_per_item_scores(payload: Mapping[str, Any]) -> tuple[FactsScoreItem, ...]:
    """Extract valid cached per-item score rows from likely Exp 3862 shapes."""

    rows = _candidate_rows(payload)
    items: list[FactsScoreItem] = []
    for index, row in enumerate(rows):
        label = _coerce_label(_first_value(row, LABEL_KEYS))
        graph_score = _coerce_score(_score_value(row, GRAPH_SCORE_KEYS))
        ensemble_score = _coerce_score(_score_value(row, ENSEMBLE_SCORE_KEYS))
        if label is None or graph_score is None or ensemble_score is None:
            continue
        items.append(
            FactsScoreItem(
                item_id=str(
                    row.get("item_id")
                    or row.get("id")
                    or row.get("question_id")
                    or row.get("row_id")
                    or index
                ),
                gold_ungrounded=label,
                graph_score=graph_score,
                ensemble_score=ensemble_score,
            )
        )
    return tuple(items)


def compute_complementarity_metrics(
    items: Sequence[FactsScoreItem],
    *,
    graph_threshold: float = GRAPH_CATCH_THRESHOLD,
    ensemble_threshold: float = ENSEMBLE_CATCH_THRESHOLD,
) -> ComplementarityMetrics:
    """Compute facts catch sets and phi/Matthews error-mask correlation."""

    labels = [1 if item.gold_ungrounded else 0 for item in items]
    graph_preds = [1 if item.graph_score >= graph_threshold else 0 for item in items]
    ensemble_preds = [1 if item.ensemble_score >= ensemble_threshold else 0 for item in items]

    gold_indices = [idx for idx, label in enumerate(labels) if label == 1]
    n_gold = len(gold_indices)
    graph_caught = {
        idx for idx in gold_indices if graph_preds[idx] == 1
    }
    ensemble_caught = {
        idx for idx in gold_indices if ensemble_preds[idx] == 1
    }
    graph_only = tuple(idx for idx in gold_indices if idx in graph_caught and idx not in ensemble_caught)
    ensemble_only = tuple(idx for idx in gold_indices if idx in ensemble_caught and idx not in graph_caught)
    union_caught = graph_caught | ensemble_caught

    graph_error_mask = [1 if pred != label else 0 for pred, label in zip(graph_preds, labels, strict=True)]
    ensemble_error_mask = [
        1 if pred != label else 0
        for pred, label in zip(ensemble_preds, labels, strict=True)
    ]
    correlation = matthews_phi(graph_error_mask, ensemble_error_mask)
    confusion = _mask_confusion(graph_error_mask, ensemble_error_mask)
    graph_only_count = len(graph_only)
    recommended = bool(
        graph_only_count >= MATERIAL_GRAPH_CATCH_MIN
        and correlation <= LOW_ERROR_MASK_CORRELATION_MAX
    )
    graph_rate = len(graph_caught) / n_gold if n_gold else 0.0
    ensemble_rate = len(ensemble_caught) / n_gold if n_gold else 0.0
    union_rate = len(union_caught) / n_gold if n_gold else 0.0
    return ComplementarityMetrics(
        n_facts_items=len(items),
        n_gold_ungrounded=n_gold,
        graph_catches_ensemble_misses=graph_only_count,
        ensemble_catches_graph_misses=len(ensemble_only),
        graph_catch_rate=graph_rate,
        ensemble_catch_rate=ensemble_rate,
        union_facts_catch_rate=union_rate,
        union_lift_over_ensemble=union_rate - ensemble_rate,
        facts_error_mask_correlation=correlation,
        extended_ensemble_recommended=recommended,
        error_mask_confusion=confusion,
        graph_catches_ensemble_miss_ids=tuple(items[idx].item_id for idx in graph_only),
        ensemble_catches_graph_miss_ids=tuple(items[idx].item_id for idx in ensemble_only),
    )


def matthews_phi(first_mask: Sequence[int], second_mask: Sequence[int]) -> float:
    """Return the phi/Matthews correlation for two binary masks."""

    if len(first_mask) != len(second_mask):
        raise ValueError("masks must have the same length")
    both_one = 0
    first_only = 0
    second_only = 0
    both_zero = 0
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
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
    preconditions_checked: Sequence[Mapping[str, Any]],
    started_s: float,
    finished_s: float,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build a complete Exp 3863 artifact from computed metrics."""

    artifact: JsonDict = {
        "status": "complete",
        "honest_verdict": classify_verdict(metrics),
        "graph_catches_ensemble_misses": metrics.graph_catches_ensemble_misses,
        "ensemble_catches_graph_misses": metrics.ensemble_catches_graph_misses,
        "facts_error_mask_correlation": round(
            float(metrics.facts_error_mask_correlation), 6
        ),
        "union_facts_catch_rate": round(float(metrics.union_facts_catch_rate), 6),
        "extended_ensemble_recommended": bool(metrics.extended_ensemble_recommended),
        "n_facts_items": metrics.n_facts_items,
        "n_gold_ungrounded_items": metrics.n_gold_ungrounded,
        "graph_facts_catch_rate": round(float(metrics.graph_catch_rate), 6),
        "math_ensemble_facts_catch_rate": round(float(metrics.ensemble_catch_rate), 6),
        "union_lift_over_math_ensemble": round(
            float(metrics.union_lift_over_ensemble), 6
        ),
        "graph_catches_ensemble_miss_ids": list(metrics.graph_catches_ensemble_miss_ids),
        "ensemble_catches_graph_miss_ids": list(metrics.ensemble_catches_graph_miss_ids),
        "facts_error_mask_confusion": metrics.error_mask_confusion,
        "threshold_policy": {
            "graph_catch_threshold": GRAPH_CATCH_THRESHOLD,
            "math_ensemble_catch_threshold": ENSEMBLE_CATCH_THRESHOLD,
            "material_graph_catch_min": MATERIAL_GRAPH_CATCH_MIN,
            "low_error_mask_correlation_max": LOW_ERROR_MASK_CORRELATION_MAX,
        },
        "cited_upstream_artifacts": list(cited_upstream_artifacts),
        "preconditions_checked": list(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {
                "items": [
                    {
                        "item_id": item.item_id,
                        "gold_ungrounded": item.gold_ungrounded,
                        "graph_score": round(item.graph_score, 6),
                        "ensemble_score": round(item.ensemble_score, 6),
                    }
                    for item in items
                ],
                "metrics": {
                    "graph_catches_ensemble_misses": metrics.graph_catches_ensemble_misses,
                    "facts_error_mask_correlation": round(
                        metrics.facts_error_mask_correlation, 12
                    ),
                    "union_facts_catch_rate": round(metrics.union_facts_catch_rate, 12),
                    "extended_ensemble_recommended": metrics.extended_ensemble_recommended,
                },
                "random_seed": RANDOM_SEED,
                "threshold_policy": {
                    "graph": GRAPH_CATCH_THRESHOLD,
                    "ensemble": ENSEMBLE_CATCH_THRESHOLD,
                },
            }
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0.0, finished_s - started_s), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "model_specs": {
            "source": "Exp 3862 cached per-item graph and math-ensemble scores",
            "graph_score_orientation": "higher means predicted ungrounded",
            "math_ensemble_score_orientation": "higher means predicted ungrounded",
        },
        "tests_run": list(tests_run or []),
        "frozen_fover_0_9131_untouched": True,
        "scripts_research_conductor_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def classify_verdict(metrics: ComplementarityMetrics) -> str:
    """Map the complementarity gate to the required terminal verdict."""

    corr_text = f"{metrics.facts_error_mask_correlation:.3f}"
    catches = metrics.graph_catches_ensemble_misses
    if metrics.extended_ensemble_recommended:
        return (
            "complete: graph_verifier_COMPLEMENTARY_"
            f"catches{catches}_corr{corr_text}_extended_ensemble_recommended"
        )
    if catches < MATERIAL_GRAPH_CATCH_MIN:
        detail = f"low_independent_catch{catches}"
    elif metrics.facts_error_mask_correlation > LOW_ERROR_MASK_CORRELATION_MAX:
        detail = f"high_corr{corr_text}"
    else:
        detail = "boundary"
    return f"complete: graph_verifier_REDUNDANT_with_math_ensemble_on_facts_{detail}"


def build_blocked_artifact(
    *,
    reason: str,
    blocked_detail: str,
    preconditions_checked: Sequence[Mapping[str, Any]],
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
    started_s: float,
    finished_s: float,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build a terminal blocked artifact without fabricated complementarity."""

    artifact: JsonDict = {
        "status": "blocked",
        "honest_verdict": reason,
        "blocked_detail": blocked_detail,
        "graph_catches_ensemble_misses": 0,
        "ensemble_catches_graph_misses": 0,
        "facts_error_mask_correlation": None,
        "union_facts_catch_rate": None,
        "extended_ensemble_recommended": False,
        "n_facts_items": 0,
        "n_gold_ungrounded_items": 0,
        "graph_facts_catch_rate": None,
        "math_ensemble_facts_catch_rate": None,
        "union_lift_over_math_ensemble": None,
        "graph_catches_ensemble_miss_ids": [],
        "ensemble_catches_graph_miss_ids": [],
        "facts_error_mask_confusion": None,
        "threshold_policy": {
            "graph_catch_threshold": GRAPH_CATCH_THRESHOLD,
            "math_ensemble_catch_threshold": ENSEMBLE_CATCH_THRESHOLD,
            "material_graph_catch_min": MATERIAL_GRAPH_CATCH_MIN,
            "low_error_mask_correlation_max": LOW_ERROR_MASK_CORRELATION_MAX,
        },
        "cited_upstream_artifacts": list(cited_upstream_artifacts),
        "preconditions_checked": list(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": reproducibility_checksum(
            {
                "blocked_reason": reason,
                "blocked_detail": blocked_detail,
                "preconditions_checked": list(preconditions_checked),
                "cited_upstream_artifacts": list(cited_upstream_artifacts),
                "random_seed": RANDOM_SEED,
            }
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(0.0, finished_s - started_s), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "model_specs": {
            "source": "Exp 3862 cached per-item graph and math-ensemble scores required",
            "blocked": True,
        },
        "tests_run": list(tests_run or []),
        "frozen_fover_0_9131_untouched": True,
        "scripts_research_conductor_modified": False,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3863 artifact contract."""

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
    for field in REQUIRED_PRINCIPLE_FIELDS:
        if "principle" not in str(principles[field]).lower():
            raise ValueError(f"field_principles.{field} must include a principle note")
    if type(artifact.get("extended_ensemble_recommended")) is not bool:
        raise ValueError("extended_ensemble_recommended must be a bare bool")
    if not isinstance(artifact.get("n_facts_items"), int) or artifact["n_facts_items"] < 0:
        raise ValueError("n_facts_items must be a non-negative integer")
    if not isinstance(artifact.get("preconditions_checked"), list):
        raise ValueError("preconditions_checked must be a list")
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        raise ValueError("cited_upstream_artifacts must be a list")
    if not isinstance(artifact.get("random_seed"), int):
        raise ValueError("random_seed must be an integer")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a SHA-256 hex digest")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or float(duration) < 0.0:
        raise ValueError("duration_s must be a non-negative number")

    if verdict.startswith("complete:"):
        if not isinstance(artifact.get("graph_catches_ensemble_misses"), int):
            raise ValueError("graph_catches_ensemble_misses must be an integer")
        corr = artifact.get("facts_error_mask_correlation")
        if not isinstance(corr, (int, float)) or not -1.0 <= float(corr) <= 1.0:
            raise ValueError("facts_error_mask_correlation must be in [-1, 1]")
        union = artifact.get("union_facts_catch_rate")
        if not isinstance(union, (int, float)) or not 0.0 <= float(union) <= 1.0:
            raise ValueError("union_facts_catch_rate must be in [0, 1]")
    else:
        if artifact.get("honest_verdict") != "blocked_graph_prototype_unavailable":
            raise ValueError("blocked Exp 3863 artifacts must use blocked_graph_prototype_unavailable")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    upstream_path: Path | str = UPSTREAM_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build and persist the Exp 3863 artifact."""

    root_path = Path(root)
    output = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(
        root_path,
        upstream_path=upstream_path,
        tests_run=tests_run,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def cite_upstream_artifact(root: Path, upstream_path: Path) -> list[JsonDict]:
    """Return the Exp 3862 citation with SHA-256 when the file exists."""

    if not upstream_path.exists():
        return []
    try:
        payload = json.loads(upstream_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    return [
        {
            "experiment_id": 3862,
            "path": relative_path(root, upstream_path),
            "sha256": sha256_file(upstream_path),
            "facts_catch_delta": payload.get("facts_catch_delta"),
            "n_facts_items": payload.get("n_facts_items"),
        }
    ]


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 digest for an existing file."""

    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reproducibility_checksum(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest for the analyzed cached inputs."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def relative_path(root: Path, path: Path) -> str:
    """Return a repo-relative display path when possible."""

    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _blocked_from_preconditions(
    detail: str,
    preconditions_checked: Sequence[Mapping[str, Any]],
    cited_upstream_artifacts: Sequence[Mapping[str, Any]],
    started_s: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    finished = time.perf_counter() if now_s is None else float(now_s)
    return build_blocked_artifact(
        reason="blocked_graph_prototype_unavailable",
        blocked_detail=detail,
        preconditions_checked=preconditions_checked,
        cited_upstream_artifacts=cited_upstream_artifacts,
        started_s=started_s,
        finished_s=finished,
        tests_run=tests_run,
    )


def _candidate_rows(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    for key in PER_ITEM_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return tuple(row for row in value if isinstance(row, Mapping))
    for value in payload.values():
        if isinstance(value, Mapping):
            rows = _candidate_rows(value)
            if rows:
                return rows
    return ()


def _first_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _score_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    direct = _first_value(row, keys)
    if direct is not None and not isinstance(direct, Mapping):
        return direct
    for nested_key in ("scores", "score", "metrics"):
        nested = row.get(nested_key)
        if isinstance(nested, Mapping):
            nested_value = _first_value(nested, keys)
            if nested_value is not None:
                return nested_value
    graph_nested = row.get("graph")
    if isinstance(graph_nested, Mapping) and keys == GRAPH_SCORE_KEYS:
        nested_value = _first_value(graph_nested, ("score", "energy", "grounding_score"))
        if nested_value is not None:
            return nested_value
    ensemble_nested = row.get("ensemble") or row.get("math_ensemble")
    if isinstance(ensemble_nested, Mapping) and keys == ENSEMBLE_SCORE_KEYS:
        nested_value = _first_value(ensemble_nested, ("score", "energy"))
        if nested_value is not None:
            return nested_value
    return None


def _coerce_label(value: Any) -> bool | None:
    if type(value) is bool:
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "ungrounded", "hallucination", "hallucinated", "incorrect"}:
            return True
        if lowered in {"0", "false", "no", "grounded", "supported", "correct"}:
            return False
    return None


def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return score


def _mask_confusion(first_mask: Sequence[int], second_mask: Sequence[int]) -> JsonDict:
    both_error = 0
    graph_only = 0
    ensemble_only = 0
    both_correct = 0
    for first, second in zip(first_mask, second_mask, strict=True):
        first_b = bool(first)
        second_b = bool(second)
        if first_b and second_b:
            both_error += 1
        elif first_b:
            graph_only += 1
        elif second_b:
            ensemble_only += 1
        else:
            both_correct += 1
    return {
        "both_error": both_error,
        "graph_only": graph_only,
        "ensemble_only": ensemble_only,
        "both_correct": both_correct,
    }


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path
