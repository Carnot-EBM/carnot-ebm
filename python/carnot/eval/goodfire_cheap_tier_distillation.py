"""Exp 1145 Goodfire cheap-tier threshold distillation.

Spec: REQ-VERIFY-1145, SCENARIO-VERIFY-1145.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from carnot.eval.halluguard_cascade_router_v3 import (
    EMBEDDING_DISTANCE_THRESHOLD,
    ENTROPY_THRESHOLD,
    HashingTextEmbedder,
    TextEmbeddingModel,
    cosine_distance,
    entropy_proxy,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXEMPLAR_PATH = REPO_ROOT / "data" / "llm_failure_exemplars.jsonl"
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus_v4.json"
DEFAULT_EXP1132_PATH = REPO_ROOT / "results" / "experiment_1132_goodfire_exemplar_cascade_tp.json"
DEFAULT_EXP1143_PATH = REPO_ROOT / "results" / "experiment_1143_halluguard_cascade_router_v3.json"
DEFAULT_RESULT_PATH = (
    REPO_ROOT / "results" / "experiment_1145_goodfire_cheap_tier_distillation.json"
)

DEFAULT_THINKPRM_THRESHOLD = 0.372
DEFAULT_SEMENERGY_THRESHOLD = -0.5
DEFAULT_FOVER_CORRECT_N = 100

REQUIRED_ARTIFACT_FIELDS = [
    "n_exemplars",
    "thinkprm_tp_before",
    "semenergy_tp_before",
    "combined_cheap_tp_before",
    "thinkprm_threshold_adjusted",
    "semenergy_threshold_adjusted",
    "combined_cheap_tp_after",
    "false_positive_rate_after",
    "cheap_tier_tp_rate_improved",
    "honest_verdict",
]
ALLOWED_HONEST_VERDICTS = {
    "cheap_tier_calibrated_tp_improved",
    "calibration_no_improvement",
    "threshold_trade_off_fp_increase",
    "honest_negative",
}


@dataclass
class CheapTierScore:
    """One row of cheap-tier scores and HalluGuard context features."""

    id: str
    category: str
    text: str
    thinkprm_score: float
    semenergy_score: float
    entropy_proxy: float
    embedding_distance: float
    query_text: str = ""


@dataclass
class ThresholdPolicy:
    """Cheap-tier threshold policy, optionally gated by a HalluGuard feature."""

    thinkprm_default_threshold: float = DEFAULT_THINKPRM_THRESHOLD
    semenergy_default_threshold: float = DEFAULT_SEMENERGY_THRESHOLD
    thinkprm_adjusted_threshold: float | None = None
    semenergy_adjusted_threshold: float | None = None
    thinkprm_feature_gate: str | None = None
    semenergy_feature_gate: str | None = None
    entropy_threshold: float = ENTROPY_THRESHOLD
    embedding_distance_threshold: float = EMBEDDING_DISTANCE_THRESHOLD
    thinkprm_optimal_threshold: float = DEFAULT_THINKPRM_THRESHOLD
    semenergy_optimal_threshold: float = DEFAULT_SEMENERGY_THRESHOLD


def load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load local JSON array or JSONL rows for experiment corpora."""

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return list(json.loads(text))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def find_tp_maximizing_threshold(scores: Sequence[float], *, direction: str) -> float:
    """Return the highest threshold that catches all positive calibration scores.

    Exp 1145 calibrates on a positive-only Goodfire set because k=5 is the
    ground-truth bug label for every exemplar.  For `>=` tiers, the tightest
    all-positive threshold is the minimum score.  For strict `>` tiers, the
    threshold must sit just below the minimum score.
    """

    values = [float(score) for score in scores]
    if not values:
        if direction == "ge":
            return DEFAULT_THINKPRM_THRESHOLD
        if direction == "gt":
            return DEFAULT_SEMENERGY_THRESHOLD
        raise ValueError(f"unknown threshold direction: {direction}")
    minimum = min(values)
    if direction == "ge":
        return float(minimum)
    if direction == "gt":
        return float(math.nextafter(minimum, -math.inf))
    raise ValueError(f"unknown threshold direction: {direction}")


def score_exemplar_rows(
    rows: Sequence[dict[str, Any]],
    *,
    think_probe: Any,
    semenergy_probe: Any,
    text_keys: Sequence[str],
) -> list[CheapTierScore]:
    """Score rows with the Exp 1132 cheap-tier conventions."""

    scored: list[CheapTierScore] = []
    for idx, row in enumerate(rows, start=1):
        text = _first_text(row, text_keys)
        query = _first_text(row, ("prompt", "question", "query", "problem"))
        scored.append(
            CheapTierScore(
                id=str(row.get("id", f"row_{idx:03d}")),
                category=str(row.get("category", "unknown")),
                text=text,
                query_text=query,
                thinkprm_score=_score_thinkprm(think_probe, text),
                semenergy_score=_score_semenergy(semenergy_probe, text),
                entropy_proxy=entropy_proxy(text),
                embedding_distance=0.0,
            )
        )
    return scored


def attach_embedding_distances(
    goodfire_scores: Sequence[CheapTierScore],
    correct_scores: Sequence[CheapTierScore],
    *,
    embedding_model: TextEmbeddingModel | None = None,
) -> str:
    """Attach HalluGuard-style query embedding distances to score rows."""

    if not goodfire_scores and not correct_scores:
        return "none"
    model = embedding_model or HashingTextEmbedder()
    backend = model.__class__.__name__
    centroid_texts = [_embedding_text(row) for row in correct_scores]
    if not centroid_texts:
        for row in [*goodfire_scores, *correct_scores]:
            row.embedding_distance = 1.0
        return backend

    correct_vectors = _normalized_embeddings(model, centroid_texts)
    centroid = correct_vectors.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm > 0.0:
        centroid = centroid / centroid_norm

    for row in [*goodfire_scores, *correct_scores]:
        vector = _normalized_embeddings(model, [_embedding_text(row)])[0]
        row.embedding_distance = cosine_distance(vector, centroid)
    return backend


def summarize_halluguard_miss_features(
    scores: Sequence[CheapTierScore],
    exp1143_artifact: dict[str, Any],
) -> dict[str, Any]:
    """Summarize which HalluGuard feature explains default cheap-tier misses."""

    entropy_threshold = float(exp1143_artifact.get("entropy_threshold", ENTROPY_THRESHOLD))
    distance_threshold = float(
        exp1143_artifact.get("embedding_distance_threshold", EMBEDDING_DISTANCE_THRESHOLD)
    )
    misses = [
        row
        for row in scores
        if not _combined_flag(row, ThresholdPolicy(entropy_threshold=entropy_threshold))
    ]
    if not misses:
        return {
            "halluguard_features_explain_goodfire_failures": bool(
                exp1143_artifact.get("halluguard_features_explain_goodfire_failures", False)
            ),
            "cheap_tier_miss_count": 0,
            "entropy_threshold": entropy_threshold,
            "embedding_distance_threshold": distance_threshold,
            "entropy_proxy_miss_flag_rate": 0.0,
            "embedding_distance_miss_flag_rate": 0.0,
            "dominant_halluguard_feature": "none",
        }

    entropy_rate = _rate(row.entropy_proxy >= entropy_threshold for row in misses)
    distance_rate = _rate(row.embedding_distance >= distance_threshold for row in misses)
    dominant = "entropy_proxy" if entropy_rate >= distance_rate else "embedding_distance"
    return {
        "halluguard_features_explain_goodfire_failures": bool(
            exp1143_artifact.get("halluguard_features_explain_goodfire_failures", False)
        ),
        "cheap_tier_miss_count": len(misses),
        "entropy_threshold": entropy_threshold,
        "embedding_distance_threshold": distance_threshold,
        "entropy_proxy_miss_flag_rate": round(entropy_rate, 6),
        "embedding_distance_miss_flag_rate": round(distance_rate, 6),
        "dominant_halluguard_feature": dominant,
    }


def calibrate_policy(
    goodfire_scores: Sequence[CheapTierScore],
    feature_summary: dict[str, Any],
) -> ThresholdPolicy:
    """Find per-tier optimal thresholds and apply the HalluGuard-consistent one."""

    think_threshold = find_tp_maximizing_threshold(
        [row.thinkprm_score for row in goodfire_scores],
        direction="ge",
    )
    sem_threshold = find_tp_maximizing_threshold(
        [row.semenergy_score for row in goodfire_scores],
        direction="gt",
    )
    policy = ThresholdPolicy(
        entropy_threshold=float(feature_summary.get("entropy_threshold", ENTROPY_THRESHOLD)),
        embedding_distance_threshold=float(
            feature_summary.get("embedding_distance_threshold", EMBEDDING_DISTANCE_THRESHOLD)
        ),
        thinkprm_optimal_threshold=think_threshold,
        semenergy_optimal_threshold=sem_threshold,
    )
    if not feature_summary.get("halluguard_features_explain_goodfire_failures", False):
        return policy

    dominant = feature_summary.get("dominant_halluguard_feature")
    if dominant == "entropy_proxy":
        policy.thinkprm_adjusted_threshold = think_threshold
        policy.thinkprm_feature_gate = "entropy_proxy"
    elif dominant == "embedding_distance":
        policy.semenergy_adjusted_threshold = sem_threshold
        policy.semenergy_feature_gate = "embedding_distance"
    return policy


def evaluate_policy(
    scores: Sequence[CheapTierScore],
    policy: ThresholdPolicy,
) -> dict[str, Any]:
    """Evaluate cheap-tier OR-logic under a threshold policy."""

    think_flags = [_thinkprm_flag(row, policy) for row in scores]
    sem_flags = [_semenergy_flag(row, policy) for row in scores]
    combined_flags = [think or sem for think, sem in zip(think_flags, sem_flags, strict=True)]
    return {
        "thinkprm_tp_rate": round(_rate(think_flags), 6),
        "semenergy_tp_rate": round(_rate(sem_flags), 6),
        "combined_tp_rate": round(_rate(combined_flags), 6),
        "combined_flags": combined_flags,
        "thinkprm_flags": think_flags,
        "semenergy_flags": sem_flags,
    }


def build_exp1145_artifact(
    *,
    goodfire_scores: Sequence[CheapTierScore],
    correct_scores: Sequence[CheapTierScore],
    feature_summary: dict[str, Any],
    policy: ThresholdPolicy,
    exp1132_artifact: dict[str, Any],
    exp1143_artifact: dict[str, Any],
    duration_s: float,
    fover_correct_n: int,
    embedding_backend: str = "HashingTextEmbedder",
) -> dict[str, Any]:
    """Build the stable Exp 1145 result artifact."""

    default_policy = ThresholdPolicy(
        entropy_threshold=policy.entropy_threshold,
        embedding_distance_threshold=policy.embedding_distance_threshold,
    )
    before = evaluate_policy(goodfire_scores, default_policy)
    after = evaluate_policy(goodfire_scores, policy)
    correct_before = evaluate_policy(correct_scores, default_policy)
    correct_after = evaluate_policy(correct_scores, policy)
    before_category = _category_rates(goodfire_scores, before["combined_flags"])
    after_category = _category_rates(goodfire_scores, after["combined_flags"])
    improved = after["combined_tp_rate"] > before["combined_tp_rate"]
    false_positive_increased = correct_after["combined_tp_rate"] > (
        correct_before["combined_tp_rate"] + 1e-12
    )

    artifact = {
        "experiment": 1145,
        "schema": "goodfire_cheap_tier_distillation_v1",
        "run_date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "started_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 2),
        "n_exemplars": int(len(goodfire_scores)),
        "fover_correct_examples": int(fover_correct_n),
        "thinkprm_tp_before": _artifact_rate(
            exp1132_artifact,
            "tier_0a_thinkprm",
            before["thinkprm_tp_rate"],
        ),
        "semenergy_tp_before": _artifact_rate(
            exp1132_artifact,
            "tier_0c_semenergy",
            before["semenergy_tp_rate"],
        ),
        "combined_cheap_tp_before": before["combined_tp_rate"],
        "thinkprm_threshold_adjusted": policy.thinkprm_adjusted_threshold is not None,
        "semenergy_threshold_adjusted": policy.semenergy_adjusted_threshold is not None,
        "combined_cheap_tp_after": after["combined_tp_rate"],
        "false_positive_rate_before": correct_before["combined_tp_rate"],
        "false_positive_rate_after": correct_after["combined_tp_rate"],
        "cheap_tier_tp_rate_improved": improved,
        "honest_verdict": _honest_verdict(
            feature_summary=feature_summary,
            improved=improved,
            false_positive_increased=false_positive_increased,
        ),
        "halluguard_features_added": list(exp1143_artifact.get("halluguard_features_added", [])),
        "halluguard_features_explain_goodfire_failures": bool(
            exp1143_artifact.get("halluguard_features_explain_goodfire_failures", False)
        ),
        "dominant_halluguard_feature": feature_summary["dominant_halluguard_feature"],
        "entropy_proxy_miss_flag_rate": feature_summary["entropy_proxy_miss_flag_rate"],
        "embedding_distance_miss_flag_rate": feature_summary["embedding_distance_miss_flag_rate"],
        "cheap_tier_miss_count_before": feature_summary["cheap_tier_miss_count"],
        "thinkprm_default_threshold": DEFAULT_THINKPRM_THRESHOLD,
        "thinkprm_optimal_threshold": round(policy.thinkprm_optimal_threshold, 6),
        "thinkprm_threshold_after": round(
            policy.thinkprm_adjusted_threshold
            if policy.thinkprm_adjusted_threshold is not None
            else DEFAULT_THINKPRM_THRESHOLD,
            6,
        ),
        "thinkprm_feature_gate": policy.thinkprm_feature_gate,
        "semenergy_default_threshold": DEFAULT_SEMENERGY_THRESHOLD,
        "semenergy_optimal_threshold": round(policy.semenergy_optimal_threshold, 6),
        "semenergy_threshold_after": round(
            policy.semenergy_adjusted_threshold
            if policy.semenergy_adjusted_threshold is not None
            else DEFAULT_SEMENERGY_THRESHOLD,
            6,
        ),
        "semenergy_feature_gate": policy.semenergy_feature_gate,
        "entropy_threshold": policy.entropy_threshold,
        "embedding_distance_threshold": policy.embedding_distance_threshold,
        "thinkprm_tp_after": after["thinkprm_tp_rate"],
        "semenergy_tp_after": after["semenergy_tp_rate"],
        "per_category_cheap_tp_before": before_category,
        "per_category_cheap_tp_after": after_category,
        "category_improvement_summary": _category_improvement_summary(
            before_category,
            after_category,
        ),
        "embedding_backend": embedding_backend,
        "note": (
            "Exp1145 distills the k=5 Goodfire signal into cheap-tier threshold "
            "calibration using Exp1143 HalluGuard features as a gate."
        ),
    }
    return artifact


def run_experiment(
    *,
    exemplar_path: Path = DEFAULT_EXEMPLAR_PATH,
    fover_path: Path = DEFAULT_FOVER_PATH,
    exp1132_path: Path = DEFAULT_EXP1132_PATH,
    exp1143_path: Path = DEFAULT_EXP1143_PATH,
    result_path: Path = DEFAULT_RESULT_PATH,
    think_probe: Any | None = None,
    semenergy_probe: Any | None = None,
    embedding_model: TextEmbeddingModel | None = None,
    fover_correct_n: int = DEFAULT_FOVER_CORRECT_N,
) -> dict[str, Any]:
    """Run Exp 1145 and write its result artifact."""

    t0 = time.time()
    if think_probe is None:
        from carnot.verify.spilled_energy import SpilledEnergyDetector

        think_probe = SpilledEnergyDetector()
    if semenergy_probe is None:
        from carnot.verify.semenergy_probe import SemEnergyProbe

        semenergy_probe = SemEnergyProbe()

    exp1132_artifact = json.loads(exp1132_path.read_text(encoding="utf-8"))
    exp1143_artifact = json.loads(exp1143_path.read_text(encoding="utf-8"))
    goodfire_rows = load_json_or_jsonl(exemplar_path)
    correct_rows = _select_correct_fover_rows(load_json_or_jsonl(fover_path), fover_correct_n)

    goodfire_scores = score_exemplar_rows(
        goodfire_rows,
        think_probe=think_probe,
        semenergy_probe=semenergy_probe,
        text_keys=("buggy_response", "response", "completion", "answer"),
    )
    correct_scores = score_exemplar_rows(
        correct_rows,
        think_probe=think_probe,
        semenergy_probe=semenergy_probe,
        text_keys=("step_text", "correct_response", "response", "completion"),
    )
    embedding_backend = attach_embedding_distances(
        goodfire_scores,
        correct_scores,
        embedding_model=embedding_model,
    )
    feature_summary = summarize_halluguard_miss_features(goodfire_scores, exp1143_artifact)
    policy = calibrate_policy(goodfire_scores, feature_summary)
    artifact = build_exp1145_artifact(
        goodfire_scores=goodfire_scores,
        correct_scores=correct_scores,
        feature_summary=feature_summary,
        policy=policy,
        exp1132_artifact=exp1132_artifact,
        exp1143_artifact=exp1143_artifact,
        duration_s=time.time() - t0,
        fover_correct_n=len(correct_rows),
        embedding_backend=embedding_backend,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _first_text(row: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _score_thinkprm(probe: Any, text: str) -> float:
    if hasattr(probe, "spill_score"):
        return float(probe.spill_score(text))
    return float(probe.score(text))


def _score_semenergy(probe: Any, text: str) -> float:
    if hasattr(probe, "score_response_proxy"):
        return float(probe.score_response_proxy(text))
    return float(probe.score(text))


def _embedding_text(row: CheapTierScore) -> str:
    return row.query_text or row.text


def _normalized_embeddings(model: TextEmbeddingModel, texts: Sequence[str]) -> np.ndarray:
    vectors = model.encode(
        list(texts),
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _select_correct_fover_rows(rows: Sequence[dict[str, Any]], n_rows: int) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("label") == "correct"][:n_rows]


def _rate(flags: Sequence[bool] | Any) -> float:
    values = list(flags)
    if not values:
        return 0.0
    return float(sum(bool(flag) for flag in values) / len(values))


def _thinkprm_flag(row: CheapTierScore, policy: ThresholdPolicy) -> bool:
    default_flag = row.thinkprm_score >= policy.thinkprm_default_threshold
    if policy.thinkprm_adjusted_threshold is None:
        return default_flag
    return default_flag or (
        _feature_gate_active(row, policy.thinkprm_feature_gate, policy)
        and row.thinkprm_score >= policy.thinkprm_adjusted_threshold
    )


def _semenergy_flag(row: CheapTierScore, policy: ThresholdPolicy) -> bool:
    default_flag = row.semenergy_score > policy.semenergy_default_threshold
    if policy.semenergy_adjusted_threshold is None:
        return default_flag
    return default_flag or (
        _feature_gate_active(row, policy.semenergy_feature_gate, policy)
        and row.semenergy_score > policy.semenergy_adjusted_threshold
    )


def _combined_flag(row: CheapTierScore, policy: ThresholdPolicy) -> bool:
    return _thinkprm_flag(row, policy) or _semenergy_flag(row, policy)


def _feature_gate_active(
    row: CheapTierScore,
    gate: str | None,
    policy: ThresholdPolicy,
) -> bool:
    if gate is None:
        return True
    if gate == "entropy_proxy":
        return row.entropy_proxy >= policy.entropy_threshold
    if gate == "embedding_distance":
        return row.embedding_distance >= policy.embedding_distance_threshold
    raise ValueError(f"unknown HalluGuard feature gate: {gate}")


def _category_rates(
    scores: Sequence[CheapTierScore],
    flags: Sequence[bool],
) -> dict[str, float]:
    categories = sorted({row.category for row in scores})
    result: dict[str, float] = {}
    for category in categories:
        category_flags = [
            flag for row, flag in zip(scores, flags, strict=True) if row.category == category
        ]
        result[category] = round(_rate(category_flags), 6)
    return result


def _category_improvement_summary(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, Any]:
    categories = sorted(before)
    improved = sum(after[category] > before[category] for category in categories)
    no_worse = sum(after[category] >= before[category] for category in categories)
    return {
        "categories_total": len(categories),
        "categories_improved": int(improved),
        "categories_no_worse": int(no_worse),
        "improvement_persisted_across_categories": bool(
            categories and no_worse == len(categories) and improved > 0
        ),
    }


def _artifact_rate(
    exp1132_artifact: dict[str, Any],
    tier_name: str,
    fallback: float,
) -> float:
    per_tier = exp1132_artifact.get("per_tier_tp_rate", {})
    if tier_name not in per_tier:
        return round(float(fallback), 6)
    return round(float(per_tier[tier_name]), 6)


def _honest_verdict(
    *,
    feature_summary: dict[str, Any],
    improved: bool,
    false_positive_increased: bool,
) -> str:
    if not feature_summary.get("halluguard_features_explain_goodfire_failures", False):
        return "honest_negative"
    if not improved:
        return "calibration_no_improvement"
    if false_positive_increased:
        return "threshold_trade_off_fp_increase"
    return "cheap_tier_calibrated_tp_improved"
