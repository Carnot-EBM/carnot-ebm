"""Exp 4256 ARC oracle-distinct provenance leak audit.

Spec refs: REQ-VERIFY-4256, SCENARIO-VERIFY-4256.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression

from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244
from carnot.reporting import arc_set_encoder_beats_vote_4245 as exp4245


RANDOM_SEED = 4256
BOOTSTRAP_RESAMPLES = 2000
OUTPUT_REL = Path("results/experiment_4256_arc_oracle_distinct_leak_audit.json")
POOL_BUILD_REL = Path("results/experiment_4243_arc_candidate_pool_grow.json")
POOL_REL = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
SET_ENCODER_BUILD_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_build.json")
SET_ENCODER_MODEL_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_model.json")
EXP4245_REL = Path("results/experiment_4245_arc_set_encoder_beats_vote.json")
INFERENCE_SUBSTRATE = "cached_grown_arc_pool_cpu_provenance_blind_set_encoder"
BLOCKED_PROVENANCE_VERDICT = "blocked_arc_provenance_unrecoverable"
FULL_FEATURE_NAMES = tuple(exp4244.FEATURE_NAMES)
ORIGIN_HIGH_WEIGHT_TOP_N = 12
ORIGIN_HIGH_WEIGHT_MIN_FRACTION_OF_MAX = 0.25

PROVENANCE_BLIND_ALLOWLIST = (
    "vote_weight",
    "self_consistency_margin",
    "vote_weight_rank_fraction",
    "cell_confidence_mean",
    "cell_confidence_margin",
    "cell_confidence_rank_fraction",
    "set_vote_mean",
    "set_vote_max",
    "set_vote_std",
    "set_confidence_mean",
    "set_confidence_max",
    "set_confidence_std",
    "vote_weight_zscore",
    "cell_confidence_zscore",
    "modal_cell_agreement_frac",
)
MANUAL_STRIPPED_FEATURES = (
    "grid_height",
    "grid_width",
    "grid_cells",
    "grid_color_count",
    "grid_nonzero_frac",
    "grid_entropy",
    "program_length",
    "program_digit_fraction",
    "program_demo_fit",
    "program_n_calls",
    "set_candidate_count",
    "set_entropy_mean",
    "set_entropy_max",
    "set_entropy_std",
    "set_cells_mean",
    "set_cells_max",
    "set_cells_std",
    "grid_entropy_zscore",
    "grid_cells_zscore",
    "grid_duplicate_count",
    "grid_duplicate_frac",
    "shape_family_count",
    "shape_family_frac",
    "shape_vote_frac",
    "is_modal_shape",
    "palette_family_count",
    "palette_family_frac",
    "palette_vote_frac",
    "is_modal_palette",
    "same_shape_as_input",
    "area_delta_from_input_frac",
)
SPEC_REFS = ["REQ-VERIFY-4256", "SCENARIO-VERIFY-4256"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A surviving provenance-blind win AND a collapse (the win was leak) "
        "are BOTH COMPLETE and decision-grade."
    ),
    "win_survives_provenance_blind": (
        "BARE bool: A4/B1 gate on this raw value (gated-fields-must-be-bare); true iff the "
        "beats-vote delta>0 AND CI95-excl-0 on provenance-BLIND features -- the de-leaked headline."
    ),
    "origin_probe_auroc": (
        "BARE float: AUROC of a classifier predicting candidate ORIGIN (induced-vs-sampled) "
        "from the .393 features -- high value flags that origin is encoded in the features."
    ),
    "origin_correctness_corr": (
        "BARE float: correlation between candidate origin and is_correct -- the leak signature "
        "is high origin_probe_auroc AND high origin_correctness_corr (the verifier could win by "
        "detecting provenance)."
    ),
    "provenance_blind_delta": (
        "set_encoder@1 - vote@1 with origin-encoding features removed -- the de-leaked "
        "oracle-distinct lift; compare to the .393 +0.4423."
    ),
    "provenance_blind_ci95": (
        "Task-level bootstrap CI95 of the provenance-blind delta -- excluding 0 is what makes "
        "the de-leaked win real."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the de-leaked verifier scores content WITHOUT executing demos; only "
        "this keeps an A4/B1 build headline/gate-eligible."
    ),
    "random_seed": "Determinism precondition; the held-out split + bootstrap must be reproducible.",
    "reproducibility_checksum": (
        "Hash of the pool + provenance join + feature set; lets a third party re-run the audit."
    ),
    "model_specs": (
        "The origin-probe + provenance-blind feature partition (which features were stripped and why); "
        "required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "win_survives_provenance_blind",
    "origin_probe_auroc",
    "origin_correctness_corr",
    "provenance_blind_delta",
    "provenance_blind_ci95",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)


class BlockedRun(RuntimeError):
    """Expected audit precondition failure that still writes a terminal artifact."""


@dataclass(frozen=True)
class AuditRow:
    task_id: str
    candidate_id: str
    candidate_index: int
    correct: bool
    origin_induced: bool
    source_kinds: tuple[str, ...]
    features: dict[str, float]
    vote_weight: float


@dataclass(frozen=True)
class AuditCorpus:
    rows: list[AuditRow]
    grown_rows: list[exp4244.GrownPoolRow]
    pool_artifact_path: Path
    pool_artifact_sha256: str
    upstream_checksum: str
    source_kind_counts: dict[str, int]
    task_n: int
    candidate_n: int
    positive_candidate_n: int


@dataclass(frozen=True)
class BlindTrainingReport:
    auroc: float
    scores: dict[str, float]


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _resolve_pool_path(repo_root: Path) -> Path:
    build_path = repo_root / POOL_BUILD_REL
    if build_path.exists():
        build = _read_json(build_path)
        rel = build.get("pool_artifact_path")
        if isinstance(rel, str) and rel:
            candidate = Path(rel)
            return candidate if candidate.is_absolute() else repo_root / candidate
    return repo_root / POOL_REL


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else 0.0
    return 0.0


def _candidate_features(candidate: dict[str, Any]) -> dict[str, float]:
    raw = candidate.get("features")
    if not isinstance(raw, dict):
        raw = {}
    return {name: _as_float(raw.get(name)) for name in FULL_FEATURE_NAMES}


def _source_kinds(candidate: dict[str, Any]) -> tuple[str, ...]:
    raw = candidate.get("source_kinds")
    if not isinstance(raw, list) or not raw:
        raise BlockedRun(BLOCKED_PROVENANCE_VERDICT)
    kinds = tuple(str(item) for item in raw if isinstance(item, str) and item)
    if not kinds:
        raise BlockedRun(BLOCKED_PROVENANCE_VERDICT)
    return kinds


def load_audit_corpus(repo_root: Path | str = Path(".")) -> AuditCorpus:
    """SCENARIO-VERIFY-4256: load candidates with induced-vs-sampled provenance."""

    root = Path(repo_root)
    pool_path = _resolve_pool_path(root)
    try:
        with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise BlockedRun(BLOCKED_PROVENANCE_VERDICT) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise BlockedRun(BLOCKED_PROVENANCE_VERDICT)

    rows: list[AuditRow] = []
    grown_rows: list[exp4244.GrownPoolRow] = []
    source_counts: Counter[str] = Counter()
    for task in payload["tasks"]:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "")
        candidates = task.get("candidates")
        if not task_id or not isinstance(candidates, list):
            continue
        for fallback_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            source_kinds = _source_kinds(candidate)
            source_counts.update(source_kinds)
            features = _candidate_features(candidate)
            candidate_id = str(candidate.get("candidate_id") or f"{task_id}::candidate{fallback_index}")
            candidate_index = int(candidate.get("candidate_index", fallback_index))
            correct = candidate.get("is_correct") is True
            row = AuditRow(
                task_id=task_id,
                candidate_id=candidate_id,
                candidate_index=candidate_index,
                correct=correct,
                origin_induced="induced_pred_grid" in source_kinds,
                source_kinds=source_kinds,
                features=features,
                vote_weight=features["vote_weight"],
            )
            rows.append(row)
            grown_rows.append(
                exp4244.GrownPoolRow(
                    task_id=task_id,
                    candidate_id=candidate_id,
                    candidate_index=candidate_index,
                    correct=correct,
                    features=features,
                    vote_weight=features["vote_weight"],
                )
            )
    if not rows or not any(row.origin_induced for row in rows):
        raise BlockedRun(BLOCKED_PROVENANCE_VERDICT)
    return AuditCorpus(
        rows=rows,
        grown_rows=grown_rows,
        pool_artifact_path=pool_path.resolve(),
        pool_artifact_sha256=_sha256_file(pool_path),
        upstream_checksum=str(payload.get("reproducibility_checksum") or ""),
        source_kind_counts=dict(sorted(source_counts.items())),
        task_n=int(payload.get("task_n") or len({row.task_id for row in rows})),
        candidate_n=int(payload.get("candidate_n") or len(rows)),
        positive_candidate_n=int(payload.get("positive_candidate_n") or sum(row.correct for row in rows)),
    )


def load_reference_folds(
    repo_root: Path | str,
    corpus: AuditCorpus,
) -> tuple[list[set[str]], str]:
    """Load Exp 4244's task-held-out folds; fall back to the same deterministic split."""

    root = Path(repo_root)
    task_ids = {row.task_id for row in corpus.rows}
    model_path = root / SET_ENCODER_MODEL_REL
    build_path = root / SET_ENCODER_BUILD_REL
    if build_path.exists():
        try:
            build = _read_json(build_path)
            learned = build.get("learned_verifier_path")
            if isinstance(learned, str) and learned:
                candidate = Path(learned)
                model_path = candidate if candidate.is_absolute() else root / candidate
        except Exception:
            model_path = root / SET_ENCODER_MODEL_REL
    if model_path.exists():
        try:
            model = _read_json(model_path)
            raw_folds = model.get("set_encoder_oof", {}).get("fold_task_ids", [])
            folds = [set(str(task_id) for task_id in fold) for fold in raw_folds if isinstance(fold, list)]
            if folds and set().union(*folds) == task_ids:
                return folds, "exp4244_set_encoder_oof.fold_task_ids"
        except Exception:
            pass
    return exp4244.split_task_folds(corpus.grown_rows, random_seed=4244), "reconstructed_exp4244_split"


def _standardizer(rows: list[AuditRow], feature_names: tuple[str, ...]) -> tuple[list[float], list[float]]:
    means = [
        sum(row.features.get(name, 0.0) for row in rows) / float(len(rows))
        for name in feature_names
    ]
    scales = []
    for name, mean in zip(feature_names, means, strict=True):
        variance = sum((row.features.get(name, 0.0) - mean) ** 2 for row in rows) / float(len(rows))
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def _standardized_vectors(
    rows: list[AuditRow],
    feature_names: tuple[str, ...],
    means: list[float],
    scales: list[float],
) -> list[list[float]]:
    return [
        [
            (row.features.get(name, 0.0) - means[index]) / scales[index]
            for index, name in enumerate(feature_names)
        ]
        for row in rows
    ]


def _pearson(xs: list[float], ys: list[float]) -> float:
    if not xs or len(xs) != len(ys):
        return 0.0
    x_mean = sum(xs) / float(len(xs))
    y_mean = sum(ys) / float(len(ys))
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / float(len(xs))
    x_var = sum((x - x_mean) ** 2 for x in xs) / float(len(xs))
    y_var = sum((y - y_mean) ** 2 for y in ys) / float(len(ys))
    denom = math.sqrt(x_var * y_var)
    return cov / denom if denom else 0.0


def origin_probe_report(
    corpus: AuditCorpus,
    folds: list[set[str]],
    *,
    random_seed: int,
) -> dict[str, Any]:
    """REQ-VERIFY-4256: predict induced origin from the original .393 features."""

    scores: dict[str, float] = {}
    coef_totals = [0.0 for _ in FULL_FEATURE_NAMES]
    fitted_fold_n = 0
    for fold_index, heldout in enumerate(folds):
        train_rows = [row for row in corpus.rows if row.task_id not in heldout]
        test_rows = [row for row in corpus.rows if row.task_id in heldout]
        labels = [int(row.origin_induced) for row in train_rows]
        if len(set(labels)) < 2:
            base = sum(labels) / float(len(labels)) if labels else 0.0
            scores.update({row.candidate_id: base for row in test_rows})
            continue
        means, scales = _standardizer(train_rows, FULL_FEATURE_NAMES)
        clf = LogisticRegression(
            random_state=random_seed + fold_index,
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
        )
        clf.fit(_standardized_vectors(train_rows, FULL_FEATURE_NAMES, means, scales), labels)
        probabilities = clf.predict_proba(_standardized_vectors(test_rows, FULL_FEATURE_NAMES, means, scales))[:, 1]
        scores.update(
            {row.candidate_id: float(score) for row, score in zip(test_rows, probabilities, strict=True)}
        )
        for index, weight in enumerate(clf.coef_[0]):
            coef_totals[index] += abs(float(weight))
        fitted_fold_n += 1
    origin_labels = [row.origin_induced for row in corpus.rows]
    correctness_labels = [row.correct for row in corpus.rows]
    origin_scores = [scores[row.candidate_id] for row in corpus.rows]
    avg_abs_coef = [
        coef / float(fitted_fold_n)
        for coef in coef_totals
    ] if fitted_fold_n else [0.0 for _ in FULL_FEATURE_NAMES]
    max_coef = max(avg_abs_coef) if avg_abs_coef else 0.0
    floor = max_coef * ORIGIN_HIGH_WEIGHT_MIN_FRACTION_OF_MAX
    ranked = sorted(
        zip(FULL_FEATURE_NAMES, avg_abs_coef, strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    high_weight = [
        name
        for index, (name, coef) in enumerate(ranked)
        if coef > 0.0 and (index < ORIGIN_HIGH_WEIGHT_TOP_N or coef >= floor)
    ]
    positive_n = sum(correctness_labels)
    induced_positive_n = sum(row.correct and row.origin_induced for row in corpus.rows)
    return {
        "origin_probe_auroc": _round_metric(exp4244._auroc(origin_labels, origin_scores)),
        "origin_correctness_corr": _round_metric(
            _pearson([float(value) for value in origin_labels], [float(value) for value in correctness_labels])
        ),
        "induced_origin_positive_fraction": _round_metric(
            induced_positive_n / float(positive_n) if positive_n else 0.0
        ),
        "origin_probe_high_weight_features": high_weight,
        "origin_probe_avg_abs_coefficients": {
            name: _round_metric(coef)
            for name, coef in ranked
            if coef > 0.0
        },
    }


def provenance_blind_feature_partition(origin_report: dict[str, Any]) -> dict[str, Any]:
    high_weight = set(origin_report.get("origin_probe_high_weight_features", []))
    allow = set(PROVENANCE_BLIND_ALLOWLIST)
    manual = set(MANUAL_STRIPPED_FEATURES)
    retained = [
        feature
        for feature in FULL_FEATURE_NAMES
        if feature in allow and feature not in high_weight and feature not in manual
    ]
    if not retained:
        retained = ["vote_weight"]
    stripped = [feature for feature in FULL_FEATURE_NAMES if feature not in retained]
    strip_reasons = {}
    for feature in stripped:
        if feature in high_weight:
            reason = "origin_probe_high_weight"
        elif feature in manual:
            reason = "manual_shape_palette_program_duplicate_or_noncontent"
        else:  # pragma: no cover - all current Exp 4244 features are allowlisted or manually stripped.
            reason = "outside_provenance_blind_content_whitelist"
        strip_reasons[feature] = reason
    return {
        "retained_features": retained,
        "stripped_features": stripped,
        "strip_reasons": strip_reasons,
    }


@contextmanager
def _patched_feature_names(feature_names: tuple[str, ...]):
    old_feature_names = exp4244.FEATURE_NAMES
    exp4244.FEATURE_NAMES = tuple(feature_names)
    try:
        yield
    finally:
        exp4244.FEATURE_NAMES = old_feature_names


def _train_blind_set_encoder_oof(
    corpus: AuditCorpus,
    folds: list[set[str]],
    *,
    feature_names: tuple[str, ...],
    random_seed: int,
    training_epochs: int = exp4244.DEFAULT_TRAINING_EPOCHS,
    hidden_dim: int = exp4244.DEFAULT_HIDDEN_DIM,
    lr: float = exp4244.DEFAULT_LR,
) -> BlindTrainingReport:
    with _patched_feature_names(feature_names):
        report = exp4244.train_oof_set_encoder(
            corpus.grown_rows,
            folds=folds,
            random_seed=random_seed,
            bootstrap_n=0,
            hidden_dim=hidden_dim,
            training_epochs=training_epochs,
            lr=lr,
        )
    return BlindTrainingReport(
        auroc=_round_metric(report.auroc),
        scores={row.candidate_id: float(row.score) for row in report.rows},
    )


def measure_provenance_blind_gate(
    corpus: AuditCorpus,
    scores: dict[str, float],
    *,
    random_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    candidates = [
        exp4245.ScoredArcCandidate(
            task_id=row.task_id,
            candidate_id=row.candidate_id,
            candidate_index=row.candidate_index,
            vote_weight=row.vote_weight,
            correct=row.correct,
            set_encoder_score=float(scores[row.candidate_id]),
            set_encoder_train_task_excluded=True,
            fold=0,
            features=row.features,
        )
        for row in corpus.rows
        if row.candidate_id in scores
    ]
    pool = exp4245.HeldoutPool(
        candidates=candidates,
        candidate_pool_path=corpus.pool_artifact_path,
        candidate_pool_sha256=corpus.pool_artifact_sha256,
        learned_verifier_path=Path("provenance_blind_set_encoder"),
        learned_verifier_sha256="",
        score_source="exp4256_provenance_blind_set_encoder_oof_scores",
        model_specs={},
        dropped_task_n=0,
        dropped_candidate_n=len(corpus.rows) - len(candidates),
    )
    metrics = exp4245._measure_pool(
        pool,
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
        margin_threshold=exp4245.MARGIN_TRIGGER_THRESHOLD,
    )
    ci95 = metrics["set_encoder_minus_vote_ci95"]
    delta = metrics["set_encoder_minus_vote_delta"]
    return {
        "provenance_blind_delta": delta,
        "provenance_blind_ci95": ci95,
        "win_survives_provenance_blind": bool(delta > 0.0 and ci95[0] > 0.0),
        "provenance_blind_pass_rates": metrics["pass_rates"],
        "provenance_blind_task_rows": metrics["task_rows"],
        "provenance_blind_set_encoder_auroc": _round_metric(
            exp4244._auroc([row.correct for row in corpus.rows], [scores[row.candidate_id] for row in corpus.rows])
        ),
        "oracle_at_k": metrics["oracle_at_k"],
        "headroom_exists": metrics["headroom_exists"],
    }


def _load_exp4245_summary(repo_root: Path) -> dict[str, Any]:
    path = repo_root / EXP4245_REL
    return _read_json(path) if path.exists() else {}


def _model_specs(
    *,
    corpus: AuditCorpus,
    fold_source: str,
    origin_report: dict[str, Any],
    partition: dict[str, Any],
    blind_report: BlindTrainingReport,
    exp4245_summary: dict[str, Any],
    training_epochs: int,
    hidden_dim: int,
) -> dict[str, Any]:
    return {
        "audit_method": (
            "Behavior/provenance check: predict candidate origin from .393 features, then retrain "
            "a task-held-out DeepSets scorer using only provenance-blind content signals."
        ),
        "origin_definition": (
            "induced_origin is true iff source_kinds contains induced_pred_grid; gold_flag-only "
            "candidates remain sampled/gold-pool origin and are reported in the source breakdown."
        ),
        "origin_probe": {
            "classifier": "task_held_out_standardized_logistic_regression_balanced",
            "feature_set": list(FULL_FEATURE_NAMES),
            "high_weight_rule": (
                f"top_{ORIGIN_HIGH_WEIGHT_TOP_N}_standardized_abs_coefficients_or_at_least_"
                f"{ORIGIN_HIGH_WEIGHT_MIN_FRACTION_OF_MAX:.2f}_of_max"
            ),
            "high_weight_features": list(origin_report["origin_probe_high_weight_features"]),
        },
        "provenance_blind_feature_partition": partition,
        "provenance_blind_set_encoder": {
            "architecture": "deepsets_pooled_context_set_encoder",
            "feature_set": list(partition["retained_features"]),
            "fold_source": fold_source,
            "hidden_dim": int(hidden_dim),
            "training_epochs": int(training_epochs),
            "oof_auroc": blind_report.auroc,
        },
        "source_kind_breakdown": corpus.source_kind_counts,
        "original_exp4245": {
            "set_encoder_minus_vote_delta": exp4245_summary.get("set_encoder_minus_vote_delta"),
            "set_encoder_minus_vote_ci95": exp4245_summary.get("set_encoder_minus_vote_ci95"),
            "verifier_is_oracle": exp4245_summary.get("verifier_is_oracle"),
        },
    }


def reproducibility_checksum(
    *,
    corpus: AuditCorpus,
    folds: list[set[str]],
    origin_report: dict[str, Any],
    partition: dict[str, Any],
    random_seed: int,
) -> str:
    payload = {
        "folds": [sorted(fold) for fold in folds],
        "origin_probe_high_weight_features": origin_report["origin_probe_high_weight_features"],
        "pool_artifact_sha256": corpus.pool_artifact_sha256,
        "provenance_source_kind_counts": corpus.source_kind_counts,
        "random_seed": int(random_seed),
        "retained_features": partition["retained_features"],
        "stripped_features": partition["stripped_features"],
        "upstream_checksum": corpus.upstream_checksum,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_artifact(
    reason: str,
    *,
    random_seed: int,
    checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4256_arc_oracle_distinct_leak_audit",
        "schema": "carnot.arc_oracle_distinct_leak_audit_4256.v1",
        "status": "complete",
        "honest_verdict": reason,
        "headline_outcome": "arc_provenance_audit_blocked",
        "win_survives_provenance_blind": False,
        "origin_probe_auroc": 0.0,
        "origin_correctness_corr": 0.0,
        "induced_origin_positive_fraction": 0.0,
        "provenance_blind_delta": 0.0,
        "provenance_blind_ci95": [0.0, 0.0],
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": {
            "status": "blocked",
            "blocked_reason": reason,
            "provenance_recovery": "source_kinds_missing_or_empty",
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "bootstrap_resamples": 0,
        "duration_s": round(duration_s, 6),
        "adversarial_verify": {"status": "pending"},
    }


def _complete_artifact(
    *,
    corpus: AuditCorpus,
    origin_report: dict[str, Any],
    blind_metrics: dict[str, Any],
    model_specs: dict[str, Any],
    checksum: str,
    random_seed: int,
    bootstrap_resamples: int,
    duration_s: float,
) -> dict[str, Any]:
    survives = blind_metrics["win_survives_provenance_blind"]
    verdict = (
        "complete: arc_set_encoder_win_survives_provenance_blind_audit"
        if survives
        else "complete: arc_set_encoder_win_collapses_under_provenance_blind_audit"
    )
    return {
        "experiment": "experiment_4256_arc_oracle_distinct_leak_audit",
        "schema": "carnot.arc_oracle_distinct_leak_audit_4256.v1",
        "status": "complete",
        "honest_verdict": verdict,
        "headline_outcome": (
            "arc_provenance_blind_win_survives"
            if survives
            else "arc_provenance_blind_win_collapses"
        ),
        "win_survives_provenance_blind": survives,
        "origin_probe_auroc": origin_report["origin_probe_auroc"],
        "origin_correctness_corr": origin_report["origin_correctness_corr"],
        "induced_origin_positive_fraction": origin_report["induced_origin_positive_fraction"],
        "provenance_blind_delta": blind_metrics["provenance_blind_delta"],
        "provenance_blind_ci95": blind_metrics["provenance_blind_ci95"],
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": model_specs,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "bootstrap_resamples": int(bootstrap_resamples),
        "candidate_count": corpus.candidate_n,
        "held_out_task_n": corpus.task_n,
        "positive_candidate_n": corpus.positive_candidate_n,
        "provenance_blind_pass_rates": blind_metrics["provenance_blind_pass_rates"],
        "provenance_blind_task_rows": blind_metrics["provenance_blind_task_rows"],
        "provenance_blind_set_encoder_auroc": blind_metrics["provenance_blind_set_encoder_auroc"],
        "oracle_at_k": blind_metrics["oracle_at_k"],
        "headroom_exists": blind_metrics["headroom_exists"],
        "duration_s": round(duration_s, 6),
        "adversarial_verify": {"status": "pending"},
    }


def _run_adversarial_verify(repo_root: Path, artifact_path: Path) -> dict[str, Any]:  # pragma: no cover
    proc = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "adversarial_verify.py"), "--json", str(artifact_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        payload = {"stdout": proc.stdout, "stderr": proc.stderr}
    payload["returncode"] = proc.returncode
    return payload


def _clean_adversarial_report(report: dict[str, Any]) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []
    for item in report.get("reports", []):
        if isinstance(item, dict):
            flags.extend(flag for flag in item.get("flags", []) if isinstance(flag, dict))
    circular_clean = not any(flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM" for flag in flags)
    return {
        "status": "clean" if not flags else "flagged",
        "circular_moat_overclaim_clean": circular_clean,
        "flag_count": len(flags),
        "flags": flags,
        "returncode": int(report.get("returncode", 0) or 0),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "complete_", "blocked_")):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if type(artifact["win_survives_provenance_blind"]) is not bool:
        raise ValueError("win_survives_provenance_blind must be a bare bool")
    for field in ("origin_probe_auroc", "origin_correctness_corr", "provenance_blind_delta"):
        if isinstance(artifact[field], bool) or not isinstance(artifact[field], (int, float)):
            raise ValueError(f"{field} must be a bare float")
    ci95 = artifact["provenance_blind_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ci95)
    ):
        raise ValueError("provenance_blind_ci95 must be a two-number ci95")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4256")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4256")


def _blocked_checksum(reason: str, random_seed: int) -> str:
    raw = json.dumps({"random_seed": random_seed, "reason": reason}, sort_keys=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    training_epochs: int = exp4244.DEFAULT_TRAINING_EPOCHS,
    hidden_dim: int = exp4244.DEFAULT_HIDDEN_DIM,
    lr: float = exp4244.DEFAULT_LR,
    adversarial_runner: Any | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    try:
        corpus = load_audit_corpus(root)
        folds, fold_source = load_reference_folds(root, corpus)
        origin_report = origin_probe_report(corpus, folds, random_seed=random_seed)
        partition = provenance_blind_feature_partition(origin_report)
        blind_report = _train_blind_set_encoder_oof(
            corpus,
            folds,
            feature_names=tuple(partition["retained_features"]),
            random_seed=random_seed,
            training_epochs=training_epochs,
            hidden_dim=hidden_dim,
            lr=lr,
        )
        blind_metrics = measure_provenance_blind_gate(
            corpus,
            blind_report.scores,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        checksum = reproducibility_checksum(
            corpus=corpus,
            folds=folds,
            origin_report=origin_report,
            partition=partition,
            random_seed=random_seed,
        )
        model_specs = _model_specs(
            corpus=corpus,
            fold_source=fold_source,
            origin_report=origin_report,
            partition=partition,
            blind_report=blind_report,
            exp4245_summary=_load_exp4245_summary(root),
            training_epochs=training_epochs,
            hidden_dim=hidden_dim,
        )
        artifact = _complete_artifact(
            corpus=corpus,
            origin_report=origin_report,
            blind_metrics=blind_metrics,
            model_specs=model_specs,
            checksum=checksum,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as exc:
        reason = str(exc) or BLOCKED_PROVENANCE_VERDICT
        artifact = _blocked_artifact(
            reason,
            random_seed=random_seed,
            checksum=_blocked_checksum(reason, random_seed),
            duration_s=time.perf_counter() - start,
        )
    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_report = (
        adversarial_runner(output_path)
        if adversarial_runner is not None
        else _run_adversarial_verify(root, output_path)
    )
    artifact["adversarial_verify"] = _clean_adversarial_report(raw_report)
    validate_artifact(artifact)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - exercised by the result entrypoint.
    repo_root = Path(__file__).resolve().parents[3]
    print(json.dumps(run(repo_root), indent=2, sort_keys=True))
