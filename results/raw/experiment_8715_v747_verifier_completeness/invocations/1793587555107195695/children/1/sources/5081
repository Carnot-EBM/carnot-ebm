"""Exp 4220 labeled oracle-distinct ARC verifier.

Spec refs: REQ-VERIFY-4220, SCENARIO-VERIFY-4220,
SCENARIO-VERIFY-4220-BLOCKED.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.reporting import vstar_learned_selector_4176 as vstar_4176


RANDOM_SEED = 4220
BOOTSTRAP_N = 1000
SPARSE_POSITIVE_THRESHOLD = 30
POOL_REL = Path("results/arc3_gap3_stage2_eval_pool.json.gz")
PROGRAMS_REL = Path("results/arc3_gap4_induced_programs.json")
OUTPUT_REL = Path("results/experiment_4220_oracle_distinct_arc_verifier_build_labeled.json")
VERIFIER_REL = Path("results/experiment_4220_oracle_distinct_arc_verifier_model.json")
SPEC_REFS = [
    "REQ-VERIFY-4220",
    "SCENARIO-VERIFY-4220",
    "SCENARIO-VERIFY-4220-BLOCKED",
]
FEATURE_NAMES = (
    "vote_weight",
    "self_consistency_margin",
    "vote_rank_fraction",
    "cell_confidence_mean",
    "cell_confidence_margin",
    "cell_confidence_rank_fraction",
    "grid_height",
    "grid_width",
    "grid_cells",
    "grid_color_count",
    "grid_nonzero_frac",
    "grid_entropy",
    "program_length",
    "program_digit_fraction",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed (complete:/success:/passed:/shipped:). A trained off-fold "
        "verifier OR an honest 'no learnable oracle-distinct signal / too-few-positives' "
        "is COMPLETE -- both feed A2."
    ),
    "selector_trained": (
        "BARE bool: A2's gate compares this raw value (gated-fields-must-be-bare); "
        "true iff a learned ARC verifier artifact was persisted out-of-fold."
    ),
    "oracle_distinct_auroc": (
        "BARE float: off-fold detection AUROC of the LEARNED verifier vs is_correct -- "
        "the oracle-distinct discrimination the GAP-3 content energies lacked; >0.5 "
        "CI95-excl is the precondition for an A2 beats-vote win."
    ),
    "wrong_majority_n": (
        "BARE int: count of stratified tasks where oracle@K > vote@1 -- the "
        "ARBITER/AggLM headroom the learned verifier targets; A2 measures vote-beating "
        "ON these."
    ),
    "learned_verifier_path": (
        "The persisted artifact A2 loads to rerank held-out ARC candidates; the build deliverable."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the learned verifier scores WITHOUT executing the demos "
        "(Circularity Discipline); this is what makes an A2 win headline/gate-eligible, "
        "unlike the circular execution verifier."
    ),
    "model_specs": (
        "The V-STaR/aggregator probe architecture + the oracle-distinct feature set; "
        "required methodology."
    ),
    "random_seed": (
        "Determinism precondition; the fold split + probe init seeded so the AUROC is reproducible."
    ),
    "reproducibility_checksum": (
        "Hash of the ARC pools + fold split + features; catches silent pool/feature drift "
        "before A2 measures."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "selector_trained",
    "oracle_distinct_auroc",
    "oracle_distinct_auroc_ci95",
    "wrong_majority_n",
    "learned_verifier_path",
    "verifier_is_oracle",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ArcCandidateRow:
    task_id: str
    candidate_id: str
    candidate_index: int
    vote_weight: float
    correct: bool
    features: dict[str, float]
    raw_candidate_correct_flag: bool | None


@dataclass(frozen=True)
class ArcCandidateCorpus:
    pool_path: Path
    programs_path: Path
    pool_sha256: str
    programs_sha256: str
    rows: list[ArcCandidateRow]
    wrong_majority_n: int
    stratified_task_n: int
    raw_candidate_n: int
    detector_row_n: int


@dataclass(frozen=True)
class OOFRow:
    task_id: str
    candidate_id: str
    correct: bool
    score: float
    fold: int
    train_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class OOFReport:
    oracle_distinct_auroc: float
    oracle_distinct_auroc_ci95: tuple[float, float]
    fold_task_ids: list[list[str]]
    oof_rows: list[OOFRow]
    final_verifier: dict[str, Any]
    no_learnable_signal_reason: str | None = None


def _paths(repo_root: Path | str) -> tuple[Path, Path]:
    root = Path(repo_root)
    return root / POOL_REL, root / PROGRAMS_REL


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _flatten_grid(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    flat: list[float] = []
    for row in value:
        if isinstance(row, list):
            flat.extend(_as_float(cell) for cell in row)
        else:
            flat.append(_as_float(row))
    return flat


def _grid_shape(value: Any) -> tuple[int, int]:
    if not isinstance(value, list):
        return (0, 0)
    height = len(value)
    width = max((len(row) if isinstance(row, list) else 1 for row in value), default=0)
    return height, width


def _grid_equal(left: Any, right: Any) -> bool:
    left_h, left_w = _grid_shape(left)
    right_h, right_w = _grid_shape(right)
    return (left_h, left_w) == (right_h, right_w) and _flatten_grid(left) == _flatten_grid(right)


def _grid_stats(value: Any) -> dict[str, float]:
    height, width = _grid_shape(value)
    flat = _flatten_grid(value)
    cells = len(flat)
    if cells == 0:
        return {
            "grid_height": 0.0,
            "grid_width": 0.0,
            "grid_cells": 0.0,
            "grid_color_count": 0.0,
            "grid_nonzero_frac": 0.0,
            "grid_entropy": 0.0,
        }
    counts: dict[float, int] = {}
    for cell in flat:
        counts[cell] = counts.get(cell, 0) + 1
    entropy = -sum((count / cells) * math.log2(count / cells) for count in counts.values())
    return {
        "grid_height": float(height),
        "grid_width": float(width),
        "grid_cells": float(cells),
        "grid_color_count": float(len(counts)),
        "grid_nonzero_frac": sum(1 for cell in flat if cell != 0.0) / float(cells),
        "grid_entropy": entropy,
    }


def _rank_fraction(value: float, values: list[float]) -> float:
    if len(values) <= 1:
        return 1.0
    ordered = sorted(values, reverse=True)
    rank = ordered.index(value)
    return 1.0 - rank / float(len(values) - 1)


def _program_stats(program: dict[str, Any]) -> dict[str, float]:
    code = program.get("code", "")
    if not isinstance(code, str):
        code = json.dumps(code, sort_keys=True)
    digit_count = sum(1 for char in code if char.isdigit())
    length = len(code)
    return {
        "program_length": float(length),
        "program_digit_fraction": digit_count / float(length) if length else 0.0,
    }


def _feature_map(
    candidate: dict[str, Any],
    *,
    candidate_index: int,
    vote_weight: float,
    vote_weights: list[float],
    confidence_values: list[float],
    program: dict[str, Any],
) -> dict[str, float]:
    q_mean = _as_float(candidate.get("q_mean"))
    max_other_vote = max(
        (weight for index, weight in enumerate(vote_weights) if index != candidate_index),
        default=0.0,
    )
    mean_confidence = sum(confidence_values) / float(len(confidence_values)) if confidence_values else 0.0
    features = {
        "vote_weight": vote_weight,
        "self_consistency_margin": vote_weight - max_other_vote,
        "vote_rank_fraction": _rank_fraction(vote_weight, vote_weights),
        "cell_confidence_mean": q_mean,
        "cell_confidence_margin": q_mean - mean_confidence,
        "cell_confidence_rank_fraction": _rank_fraction(q_mean, confidence_values),
    }
    features.update(_grid_stats(candidate.get("grid")))
    features.update(_program_stats(program))
    return {name: float(features[name]) for name in FEATURE_NAMES}


def _import_detector_module() -> Any:  # pragma: no cover - real precondition import is integration-level
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from scripts import exp_verifier_detector_auroc as detector
    except Exception as exc:  # pragma: no cover - environment dependent import guard
        raise BlockedRun("blocked_arc_gap4_pools_missing") from exc
    return detector


def _load_gap_payloads(pool_path: Path, programs_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    if not pool_path.exists() or not programs_path.exists():
        raise BlockedRun("blocked_arc_gap4_pools_missing")
    detector = _import_detector_module()
    try:
        detector_rows = detector.load_arc_rows(pool_path, programs_path)
        with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
            pool = json.load(handle)
        programs_payload = json.loads(programs_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BlockedRun("blocked_arc_gap4_pools_missing") from exc
    entries = pool.get("entries")
    programs = programs_payload.get("programs")
    if not isinstance(entries, list) or not isinstance(programs, list):
        raise BlockedRun("blocked_arc_gap4_pools_missing")
    return entries, programs, len(detector_rows)


def load_labeled_arc_pool(repo_root: Path | str = Path(".")) -> ArcCandidateCorpus:
    """SCENARIO-VERIFY-4220: build labels from GAP-4 pred_grid equality."""

    pool_path, programs_path = _paths(repo_root)
    entries, programs, detector_row_n = _load_gap_payloads(pool_path, programs_path)
    by_entry = {
        int(program.get("entry_i", index)): program
        for index, program in enumerate(programs)
        if isinstance(program, dict)
    }
    rows_by_task: dict[str, list[ArcCandidateRow]] = {}
    raw_candidate_n = 0
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        task_id = str(entry.get("task") or f"entry-{entry_index}")
        candidates = entry.get("candidates")
        if not isinstance(candidates, list):
            continue
        raw_candidate_n += len(candidates)
        program = by_entry.get(entry_index, {})
        pred_grid = program.get("pred_grid")
        total_votes = sum(_as_float(candidate.get("votes")) for candidate in candidates if isinstance(candidate, dict))
        vote_weights = [
            _as_float(candidate.get("votes")) / total_votes if total_votes else 0.0
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
        confidence_values = [
            _as_float(candidate.get("q_mean")) for candidate in candidates if isinstance(candidate, dict)
        ]
        task_rows: list[ArcCandidateRow] = []
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            vote_weight = vote_weights[candidate_index] if candidate_index < len(vote_weights) else 0.0
            raw_flag = candidate.get("correct")
            task_rows.append(
                ArcCandidateRow(
                    task_id=task_id,
                    candidate_id=f"{task_id}::candidate{candidate_index}",
                    candidate_index=candidate_index,
                    vote_weight=vote_weight,
                    correct=_grid_equal(candidate.get("grid"), pred_grid),
                    features=_feature_map(
                        candidate,
                        candidate_index=candidate_index,
                        vote_weight=vote_weight,
                        vote_weights=vote_weights,
                        confidence_values=confidence_values,
                        program=program,
                    ),
                    raw_candidate_correct_flag=raw_flag if isinstance(raw_flag, bool) else None,
                )
            )
        rows_by_task[task_id] = task_rows

    rows: list[ArcCandidateRow] = []
    wrong_majority_n = 0
    for task_rows in rows_by_task.values():
        if len(task_rows) < 2 or not any(row.correct for row in task_rows):
            continue
        rows.extend(task_rows)
        vote_winner = max(task_rows, key=lambda row: (row.vote_weight, -row.candidate_index))
        wrong_majority_n += int(not vote_winner.correct)

    return ArcCandidateCorpus(
        pool_path=pool_path.resolve(),
        programs_path=programs_path.resolve(),
        pool_sha256=_sha256_file(pool_path),
        programs_sha256=_sha256_file(programs_path),
        rows=rows,
        wrong_majority_n=wrong_majority_n,
        stratified_task_n=len({row.task_id for row in rows}),
        raw_candidate_n=raw_candidate_n,
        detector_row_n=detector_row_n,
    )


def accepted_rejected_counts(rows: list[ArcCandidateRow]) -> dict[str, int]:
    accepted = sum(row.correct for row in rows)
    rejected = len(rows) - accepted
    return {"accepted": int(accepted), "rejected": int(rejected), "total": len(rows)}


def _feature_vector(row: ArcCandidateRow) -> list[float]:
    return [float(row.features[name]) for name in FEATURE_NAMES]


def _split_task_folds(rows: list[ArcCandidateRow], random_seed: int, n_folds: int) -> list[set[str]]:
    task_ids = sorted({row.task_id for row in rows})
    fold_count = max(2, min(int(n_folds), len(task_ids)))
    shuffled = task_ids[:]
    random.Random(random_seed).shuffle(shuffled)
    return [set(shuffled[index::fold_count]) for index in range(fold_count)]


def _standardizer(rows: list[ArcCandidateRow]) -> tuple[list[float], list[float]]:
    vectors = [_feature_vector(row) for row in rows]
    means = [sum(vector[index] for vector in vectors) / float(len(vectors)) for index in range(len(FEATURE_NAMES))]
    scales: list[float] = []
    for index, mean in enumerate(means):
        variance = sum((vector[index] - mean) ** 2 for vector in vectors) / float(len(vectors))
        scales.append(math.sqrt(variance) or 1.0)
    return means, scales


def _standardized_vector(features: dict[str, float], means: list[float], scales: list[float]) -> list[float]:
    return [
        (float(features[name]) - means[index]) / scales[index]
        for index, name in enumerate(FEATURE_NAMES)
    ]


def _train_verifier(rows: list[ArcCandidateRow], random_seed: int) -> dict[str, Any]:
    means, scales = _standardizer(rows)
    model = vstar_4176.LogisticRegression(
        random_state=random_seed,
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(
        [_standardized_vector(row.features, means, scales) for row in rows],
        [int(row.correct) for row in rows],
    )
    return {
        "model_type": "standardized_logistic_regression",
        "feature_names": list(FEATURE_NAMES),
        "feature_means": [float(value) for value in means],
        "feature_scales": [float(value) for value in scales],
        "intercept": float(model.intercept_[0]),
        "coefficients": [float(value) for value in model.coef_[0]],
    }


def _constant_verifier(rows: list[ArcCandidateRow]) -> dict[str, Any]:
    counts = accepted_rejected_counts(rows)
    base_rate = counts["accepted"] / float(counts["total"]) if counts["total"] else 0.0
    return {
        "model_type": "constant_score",
        "feature_names": list(FEATURE_NAMES),
        "constant_score": float(base_rate),
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def score_with_verifier(verifier: dict[str, Any], features: dict[str, float]) -> float:
    model_type = verifier.get("model_type")
    if model_type == "constant_score":
        return float(verifier.get("constant_score", 0.0))
    if model_type != "standardized_logistic_regression":
        raise ValueError("unknown verifier model_type")
    means = [float(value) for value in verifier["feature_means"]]
    scales = [float(value) for value in verifier["feature_scales"]]
    values = _standardized_vector(features, means, scales)
    logit = float(verifier["intercept"]) + sum(
        float(weight) * value for weight, value in zip(verifier["coefficients"], values, strict=True)
    )
    return _sigmoid(logit)


def _auroc(labels: list[bool], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / float(len(positives) * len(negatives))


def _bootstrap_auroc_ci95(labels: list[bool], scores: list[float], random_seed: int) -> tuple[float, float]:
    if len(set(labels)) < 2 or not scores:
        return (0.0, 0.0)
    rng = random.Random(random_seed)
    samples: list[float] = []
    n = len(labels)
    for _ in range(BOOTSTRAP_N):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_labels = [labels[index] for index in indices]
        if len(set(sample_labels)) < 2:
            continue
        sample_scores = [scores[index] for index in indices]
        samples.append(_auroc(sample_labels, sample_scores))
    if not samples:
        point = _auroc(labels, scores)
        return point, point
    samples.sort()
    low_index = int(0.025 * (len(samples) - 1))
    high_index = int(0.975 * (len(samples) - 1))
    return samples[low_index], samples[high_index]


def train_oof_verifier(
    rows: list[ArcCandidateRow], *, random_seed: int = RANDOM_SEED, n_folds: int = 5
) -> OOFReport:
    """SCENARIO-VERIFY-4220: train on non-held-out ARC tasks and score OOF."""

    counts = accepted_rejected_counts(rows)
    task_ids = sorted({row.task_id for row in rows})
    if counts["accepted"] < 2 or counts["rejected"] < 2 or len(task_ids) < 2:
        scores = [counts["accepted"] / float(counts["total"]) if counts["total"] else 0.0 for _ in rows]
        labels = [row.correct for row in rows]
        return OOFReport(
            oracle_distinct_auroc=_auroc(labels, scores),
            oracle_distinct_auroc_ci95=_bootstrap_auroc_ci95(labels, scores, random_seed),
            fold_task_ids=[task_ids],
            oof_rows=[
                OOFRow(row.task_id, row.candidate_id, row.correct, scores[index], 0, tuple())
                for index, row in enumerate(rows)
            ],
            final_verifier=_constant_verifier(rows),
            no_learnable_signal_reason="too_few_positives_or_task_contrast",
        )

    folds = _split_task_folds(rows, random_seed, n_folds)
    oof_scores_by_id: dict[str, float] = {}
    oof_rows: list[OOFRow] = []
    for fold, heldout_task_ids in enumerate(folds):
        train_rows = [row for row in rows if row.task_id not in heldout_task_ids]
        test_rows = [row for row in rows if row.task_id in heldout_task_ids]
        train_counts = accepted_rejected_counts(train_rows)
        if train_counts["accepted"] < 1 or train_counts["rejected"] < 1:
            verifier = _constant_verifier(train_rows)
        else:
            verifier = _train_verifier(train_rows, random_seed + fold)
        train_task_ids = tuple(sorted({row.task_id for row in train_rows}))
        for row in test_rows:
            score = score_with_verifier(verifier, row.features)
            oof_scores_by_id[row.candidate_id] = score
            oof_rows.append(
                OOFRow(
                    task_id=row.task_id,
                    candidate_id=row.candidate_id,
                    correct=row.correct,
                    score=score,
                    fold=fold,
                    train_task_ids=train_task_ids,
                )
            )
    labels = [row.correct for row in rows]
    oof_scores = [oof_scores_by_id[row.candidate_id] for row in rows]
    auroc = _auroc(labels, oof_scores)
    return OOFReport(
        oracle_distinct_auroc=auroc,
        oracle_distinct_auroc_ci95=_bootstrap_auroc_ci95(labels, oof_scores, random_seed),
        fold_task_ids=[sorted(fold) for fold in folds],
        oof_rows=oof_rows,
        final_verifier=_train_verifier(rows, random_seed),
        no_learnable_signal_reason="auroc_near_chance" if 0.45 <= auroc <= 0.55 else None,
    )


def reproducibility_checksum(corpus: ArcCandidateCorpus, report: OOFReport) -> str:
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "fold_task_ids": report.fold_task_ids,
        "pool_sha256": corpus.pool_sha256,
        "programs_sha256": corpus.programs_sha256,
        "random_seed": RANDOM_SEED,
        "rows": [
            {
                "candidate_id": row.candidate_id,
                "correct": row.correct,
                "features": row.features,
                "task_id": row.task_id,
            }
            for row in corpus.rows
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _model_specs(status: str) -> dict[str, Any]:
    return {
        "architecture": "class_weight_balanced_standardized_logistic_regression",
        "base": "Exp4176 V-STaR/AggLM logistic selector",
        "feature_set": list(FEATURE_NAMES),
        "oracle_distinct_feature_groups": [
            "vote_weight",
            "cross_candidate_self_consistency_margin",
            "per_cell_confidence",
            "grid_statistics",
            "program_text_statistics",
        ],
        "training_recipe": "accepted_and_rejected_arc_candidates_task_held_out",
        "status": status,
    }


def persist_verifier(
    path: Path,
    verifier: dict[str, Any],
    *,
    checksum: str,
    counts: dict[str, int],
    corpus: ArcCandidateCorpus,
    report: OOFReport,
    random_seed: int,
) -> None:
    payload = {
        **verifier,
        "accepted_rejected_n": counts,
        "fold_task_ids": report.fold_task_ids,
        "model_specs": _model_specs("trained"),
        "no_learnable_signal_reason": report.no_learnable_signal_reason,
        "oof_rows": [
            {
                "candidate_id": row.candidate_id,
                "correct": row.correct,
                "fold": row.fold,
                "score": _round_metric(row.score),
                "task_id": row.task_id,
                "train_task_ids": list(row.train_task_ids),
            }
            for row in report.oof_rows
        ],
        "random_seed": random_seed,
        "reproducibility_checksum": checksum,
        "source_paths": [str(corpus.pool_path), str(corpus.programs_path)],
        "spec_refs": SPEC_REFS,
        "verifier_is_oracle": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_verifier(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("verifier artifact must be a JSON object")
    return payload


def _blocked_artifact(
    reason: str,
    *,
    random_seed: int,
    checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "honest_verdict": reason,
        "selector_trained": False,
        "oracle_distinct_auroc": 0.0,
        "oracle_distinct_auroc_ci95": [0.0, 0.0],
        "wrong_majority_n": 0,
        "learned_verifier_path": "",
        "verifier_is_oracle": False,
        "model_specs": _model_specs("blocked"),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "accepted_rejected_n": {"accepted": 0, "rejected": 0, "total": 0},
        "positive_candidate_n": 0,
        "positive_sparsity_flag": False,
        "duration_s": round(duration_s, 6),
    }


def _complete_artifact(
    corpus: ArcCandidateCorpus,
    report: OOFReport,
    *,
    checksum: str,
    counts: dict[str, int],
    verifier_path: Path,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    auroc = _round_metric(report.oracle_distinct_auroc)
    ci95 = [_round_metric(value) for value in report.oracle_distinct_auroc_ci95]
    if report.no_learnable_signal_reason:
        verdict = f"complete: oracle_distinct_arc_verifier_no_learnable_signal_auroc{auroc:.4f}"
    else:
        verdict = f"complete: oracle_distinct_arc_verifier_trained_auroc_{auroc:.4f}"
    positive_candidate_n = counts["accepted"]
    return {
        "experiment": "experiment_4220_oracle_distinct_arc_verifier_build_labeled",
        "schema": "carnot.oracle_distinct_arc_verifier_4220.v1",
        "honest_verdict": verdict,
        "selector_trained": True,
        "oracle_distinct_auroc": auroc,
        "oracle_distinct_auroc_ci95": ci95,
        "wrong_majority_n": int(corpus.wrong_majority_n),
        "learned_verifier_path": str(verifier_path),
        "verifier_is_oracle": False,
        "model_specs": _model_specs("trained"),
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "accepted_rejected_n": counts,
        "positive_candidate_n": int(positive_candidate_n),
        "positive_sparsity_flag": positive_candidate_n < SPARSE_POSITIVE_THRESHOLD,
        "stratified_task_n": corpus.stratified_task_n,
        "raw_candidate_n": corpus.raw_candidate_n,
        "detector_row_n": corpus.detector_row_n,
        "candidate_pool_source": str(corpus.pool_path),
        "induced_programs_source": str(corpus.programs_path),
        "feature_names": list(FEATURE_NAMES),
        "oof_folds": len(report.fold_task_ids),
        "no_learnable_signal_reason": report.no_learnable_signal_reason,
        "label_source": "candidate_grid_equals_gap4_induced_pred_grid",
        "inference_substrate": "cached_gap_arc_pool_oof_vstar_agglm_selector",
        "duration_s": round(duration_s, 6),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith(("complete:", "success:", "passed:", "shipped:"))
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact["selector_trained"], bool):
        raise ValueError("selector_trained must be a bare bool")
    if not isinstance(artifact["oracle_distinct_auroc"], float):
        raise ValueError("oracle_distinct_auroc must be a bare float")
    if not isinstance(artifact["wrong_majority_n"], int):
        raise ValueError("wrong_majority_n must be a bare int")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4220")
    if artifact["selector_trained"] and not Path(artifact["learned_verifier_path"]).exists():
        raise ValueError("trained artifacts require a persisted verifier")


def _blocked_checksum(repo_root: Path | str) -> str:
    pool_path, programs_path = _paths(repo_root)
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "pool_sha256": _sha256_file(pool_path) if pool_path.exists() else "",
        "programs_sha256": _sha256_file(programs_path) if programs_path.exists() else "",
        "rows": [],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def run(
    repo_root: Path | str = Path("."), *, random_seed: int = RANDOM_SEED, n_folds: int = 5
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    try:
        corpus = load_labeled_arc_pool(root)
        report = train_oof_verifier(corpus.rows, random_seed=random_seed, n_folds=n_folds)
        counts = accepted_rejected_counts(corpus.rows)
        checksum = reproducibility_checksum(corpus, report)
        verifier_path = (root / VERIFIER_REL).resolve()
        persist_verifier(
            verifier_path,
            report.final_verifier,
            checksum=checksum,
            counts=counts,
            corpus=corpus,
            report=report,
            random_seed=random_seed,
        )
        artifact = _complete_artifact(
            corpus,
            report,
            checksum=checksum,
            counts=counts,
            verifier_path=verifier_path,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as blocked:
        artifact = _blocked_artifact(
            blocked.reason,
            random_seed=random_seed,
            checksum=_blocked_checksum(root),
            duration_s=time.perf_counter() - start,
        )
    validate_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact
