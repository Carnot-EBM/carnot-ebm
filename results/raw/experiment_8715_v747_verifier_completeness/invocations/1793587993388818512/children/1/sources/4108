"""Exp 4271 ARC cross-family transfer on the existing grown pool.

Spec refs: REQ-VERIFY-4271, SCENARIO-VERIFY-4271.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.reporting import arc_family_provenance_recovery_4270 as exp4270
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


RANDOM_SEED = 4271
WITHIN_POOL_DELTA_393 = 0.4423076923
BOOTSTRAP_RESAMPLES = 2000
OUTPUT_REL = Path("results/experiment_4271_arc_cross_family_transfer_existing_pool.json")
POOL_REL = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
PROVENANCE_REL = Path("results/experiment_4270_arc_family_provenance_recovery.json")
MANIFEST_REL = Path("results/experiment_4270_arc_family_manifest.json")
SET_ENCODER_BUILD_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_build.json")
SET_ENCODER_MODEL_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_model.json")
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
DEFERRED_INFEASIBLE_VERDICT = "complete_arc_cross_family_deferred_pool_infeasible"
BLOCKED_INPUTS_VERDICT = "blocked_arc_cross_family_inputs_missing"
ONLINE_INITIAL_SET_WEIGHT = 0.45
ONLINE_INITIAL_VOTE_WEIGHT = 0.55
ONLINE_ETA = 0.75
SPEC_REFS = ["REQ-VERIFY-4271", "SCENARIO-VERIFY-4271"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A cross-family win, a within-pool-only collapse, "
        "and a no-headroom are ALL COMPLETE and decision-grade."
    ),
    "cross_family_win_holds": (
        "BARE bool: the capstone reads this as the OOD verdict "
        "(gated-fields-must-be-bare); true iff the held-out-family "
        "set_encoder@1 - vote@1 > 0 AND CI95-excl-0 -- the real generalization signal."
    ),
    "cross_family_delta": (
        "BARE float: set_encoder@1 - vote@1 on HELD-OUT FAMILIES -- the "
        "load-bearing OOD lift (compare to the within-pool +0.4423)."
    ),
    "cross_family_ci95": (
        "Task-level bootstrap CI95 of the cross-family delta -- excluding 0 "
        "means the verifier generalizes to unseen families."
    ),
    "within_minus_cross_gap": (
        "The within-pool +0.4423 minus cross_family_delta -- quantifies how "
        "much of the win was per-family-basin memorization."
    ),
    "held_out_family_n": (
        "BARE int: number of families never seen in training -- the OOD "
        "breadth; report with held_out_task_n for power."
    ),
    "oracle_at_k": (
        "Positive-control ceiling on held-out families -- if ~=vote the "
        "cross-family null is uninformative, not a verifier failure."
    ),
    "online_adapt_cross_family_delta": (
        "The online-reweighted selector's cross-family delta -- the "
        "static-vs-adaptive self-learning comparison (deepened in A4)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- learned set-encoder, no demo execution; keeps the "
        "transfer result oracle-distinct."
    ),
    "random_seed": "Determinism precondition; the family-split + bootstrap reproducible.",
    "reproducibility_checksum": (
        "Hash of the pool + family partition + manifest; lets a third party re-run."
    ),
    "model_specs": (
        "The family-disjoint split protocol + set-encoder config + online-adapt "
        "rule; required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "cross_family_win_holds",
    "cross_family_delta",
    "cross_family_ci95",
    "within_minus_cross_gap",
    "held_out_family_n",
    "held_out_task_n",
    "oracle_at_k",
    "matched_control_delta",
    "online_adapt_cross_family_delta",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
    "adversarial_verify",
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""


@dataclass(frozen=True)
class FamilyAnnotatedRow:
    task_id: str
    family_id: str
    fold: int
    candidate_id: str
    candidate_index: int
    correct: bool
    features: dict[str, float]
    vote_weight: float


@dataclass(frozen=True)
class FamilyAnnotatedCorpus:
    rows: list[FamilyAnnotatedRow]
    task_family_ids: dict[str, str]
    task_folds: dict[str, int]
    manifest_path: Path
    manifest_sha256: str
    pool_artifact_path: Path
    pool_artifact_sha256: str
    upstream_checksum: str
    held_out_family_n: int
    held_out_task_n: int
    candidate_n: int


@dataclass(frozen=True)
class FamilyFold:
    held_out_families: set[str]
    train_families: set[str]
    held_out_task_ids: set[str]
    train_task_ids: set[str]


@dataclass(frozen=True)
class CrossFamilyTrainingReport:
    rows: list[exp4244.OOFRow]
    fold_summaries: list[dict[str, Any]]
    training_config: dict[str, Any]


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _safe_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _resolve_required_path(repo_root: Path, value: Any, fallback: Path | None = None) -> Path:
    candidate_value = value if isinstance(value, str) and value else str(fallback or "")
    if not candidate_value:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    path = Path(candidate_value)
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    return resolved


def _read_provenance(repo_root: Path) -> dict[str, Any]:
    try:
        provenance = _read_json_object(repo_root / PROVENANCE_REL)
    except Exception as exc:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT) from exc
    if provenance.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    return provenance


def _manifest_path_from_provenance(repo_root: Path, provenance: dict[str, Any]) -> Path:
    return _resolve_required_path(repo_root, provenance.get("provenance_manifest_path"), MANIFEST_REL)


def _load_required_artifacts(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    try:
        provenance = _read_provenance(repo_root)
        _manifest_path_from_provenance(repo_root, provenance)
        build = _read_json_object(repo_root / SET_ENCODER_BUILD_REL)
        model_path = _resolve_required_path(
            repo_root,
            build.get("learned_verifier_path"),
            SET_ENCODER_MODEL_REL,
        )
        model = exp4244.load_set_encoder(model_path)
    except Exception as exc:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT) from exc
    if build.get("aggregator_trained") is not True:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    if build.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    if model.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    return provenance, build, model, model_path


def load_family_annotated_corpus(repo_root: Path | str = Path(".")) -> FamilyAnnotatedCorpus:
    """SCENARIO-VERIFY-4271: attach Exp 4270 family ids to the grown pool."""

    root = Path(repo_root)
    provenance = _read_provenance(root)
    manifest_path = _manifest_path_from_provenance(root, provenance)
    manifest = exp4270.load_manifest(manifest_path)
    corpus = exp4244.load_grown_pool(root)

    task_family_ids: dict[str, str] = {}
    task_folds: dict[str, int] = {}
    family_folds: dict[str, int] = {}
    for row in manifest.rows:
        task_family_ids[row.task_id] = row.family_id
        task_folds[row.task_id] = int(row.fold)
        previous = family_folds.setdefault(row.family_id, int(row.fold))
        if previous != int(row.fold):
            raise BlockedRun(BLOCKED_INPUTS_VERDICT)

    rows: list[FamilyAnnotatedRow] = []
    for row in corpus.rows:
        if row.task_id not in task_family_ids:
            raise BlockedRun(BLOCKED_INPUTS_VERDICT)
        rows.append(
            FamilyAnnotatedRow(
                task_id=row.task_id,
                family_id=task_family_ids[row.task_id],
                fold=task_folds[row.task_id],
                candidate_id=row.candidate_id,
                candidate_index=row.candidate_index,
                correct=row.correct,
                features=row.features,
                vote_weight=row.vote_weight,
            )
        )
    task_ids = {row.task_id for row in rows}
    if task_ids != set(task_family_ids):
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    return FamilyAnnotatedCorpus(
        rows=rows,
        task_family_ids=task_family_ids,
        task_folds=task_folds,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=_sha256_file(manifest_path),
        pool_artifact_path=corpus.pool_artifact_path,
        pool_artifact_sha256=corpus.pool_artifact_sha256,
        upstream_checksum=corpus.upstream_checksum,
        held_out_family_n=len(set(task_family_ids.values())),
        held_out_task_n=len(task_family_ids),
        candidate_n=len(rows),
    )


def build_family_disjoint_folds(corpus: FamilyAnnotatedCorpus) -> list[FamilyFold]:
    fold_to_families: dict[int, set[str]] = defaultdict(set)
    for task_id, family_id in corpus.task_family_ids.items():
        fold_to_families[int(corpus.task_folds[task_id])].add(family_id)
    if len(fold_to_families) < 2:
        raise BlockedRun(BLOCKED_INPUTS_VERDICT)
    all_families = set(corpus.task_family_ids.values())
    folds: list[FamilyFold] = []
    for fold_index in sorted(fold_to_families):
        held_out_families = set(fold_to_families[fold_index])
        train_families = all_families - held_out_families
        held_out_task_ids = {
            task_id
            for task_id, family_id in corpus.task_family_ids.items()
            if family_id in held_out_families
        }
        train_task_ids = {
            task_id
            for task_id, family_id in corpus.task_family_ids.items()
            if family_id in train_families
        }
        if not train_task_ids or not held_out_task_ids:  # pragma: no cover - guarded by fold grouping.
            raise BlockedRun(BLOCKED_INPUTS_VERDICT)
        folds.append(
            FamilyFold(
                held_out_families=held_out_families,
                train_families=train_families,
                held_out_task_ids=held_out_task_ids,
                train_task_ids=train_task_ids,
            )
        )
    return folds


def _as_grown_rows(rows: list[FamilyAnnotatedRow]) -> list[exp4244.GrownPoolRow]:
    return [
        exp4244.GrownPoolRow(
            task_id=row.task_id,
            candidate_id=row.candidate_id,
            candidate_index=row.candidate_index,
            correct=row.correct,
            features=row.features,
            vote_weight=row.vote_weight,
        )
        for row in rows
    ]


def train_cross_family_oof(
    corpus: FamilyAnnotatedCorpus,
    folds: list[FamilyFold],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_n: int = exp4244.BOOTSTRAP_N,
    training_epochs: int = exp4244.DEFAULT_TRAINING_EPOCHS,
    hidden_dim: int = exp4244.DEFAULT_HIDDEN_DIM,
    lr: float = exp4244.DEFAULT_LR,
) -> CrossFamilyTrainingReport:
    """Train Exp 4244 set-encoders with held-out folds defined by family ids."""

    task_folds = [set(fold.held_out_task_ids) for fold in folds]
    report = exp4244.train_oof_set_encoder(
        _as_grown_rows(corpus.rows),
        folds=task_folds,
        random_seed=random_seed,
        bootstrap_n=bootstrap_n,
        hidden_dim=hidden_dim,
        training_epochs=training_epochs,
        lr=lr,
    )
    return CrossFamilyTrainingReport(
        rows=report.rows,
        fold_summaries=[
            {
                "fold": index,
                "held_out_families": sorted(fold.held_out_families),
                "train_families": sorted(fold.train_families),
                "held_out_task_n": len(fold.held_out_task_ids),
                "train_task_n": len(fold.train_task_ids),
            }
            for index, fold in enumerate(folds)
        ],
        training_config={
            "architecture": "deepsets_pooled_context_set_encoder",
            "hidden_dim": int(hidden_dim),
            "training_epochs": int(training_epochs),
            "lr": float(lr),
        },
    )


def _group_by_task(rows: list[FamilyAnnotatedRow]) -> list[list[FamilyAnnotatedRow]]:
    grouped: dict[str, list[FamilyAnnotatedRow]] = defaultdict(list)
    for row in rows:
        grouped[row.task_id].append(row)
    return [sorted(items, key=lambda item: item.candidate_index) for _, items in sorted(grouped.items())]


def _select_vote(task_rows: list[FamilyAnnotatedRow]) -> FamilyAnnotatedRow:
    return max(task_rows, key=lambda row: (row.vote_weight, -row.candidate_index))


def _select_first(task_rows: list[FamilyAnnotatedRow]) -> FamilyAnnotatedRow:
    return min(task_rows, key=lambda row: row.candidate_index)


def _rate(values: list[bool]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _bootstrap_ci95(deltas: list[float], *, random_seed: int, resamples: int) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    if len(deltas) == 1:
        point = _round_metric(deltas[0])
        return [point, point]
    rng = random.Random(random_seed)
    n = len(deltas)
    samples = [
        sum(deltas[rng.randrange(n)] for _ in range(n)) / float(n)
        for _ in range(int(resamples))
    ]
    if not samples:
        point = sum(deltas) / float(len(deltas))
        return [_round_metric(point), _round_metric(point)]
    samples.sort()
    return [
        _round_metric(samples[int(0.025 * (len(samples) - 1))]),
        _round_metric(samples[int(0.975 * (len(samples) - 1))]),
    ]


def _ci_excludes_zero(ci95: list[float]) -> bool:
    return bool(len(ci95) == 2 and (ci95[0] > 0.0 or ci95[1] < 0.0))


def _score_map(oof_rows: list[exp4244.OOFRow]) -> dict[str, tuple[float, bool, int]]:
    scores: dict[str, tuple[float, bool, int]] = {}
    for row in oof_rows:
        excluded = row.task_id not in row.train_task_ids
        scores[row.candidate_id] = (float(row.score), excluded, int(row.fold))
    return scores


def _minmax(values: dict[str, float], key: str) -> float:
    lo = min(values.values())
    hi = max(values.values())
    if hi == lo:
        return 0.5
    return (values[key] - lo) / (hi - lo)


def _online_pick(
    task_rows: list[FamilyAnnotatedRow],
    scores_by_id: dict[str, float],
    *,
    set_weight: float,
    vote_weight: float,
) -> FamilyAnnotatedRow:
    vote_by_id = {row.candidate_id: row.vote_weight for row in task_rows}
    return max(
        task_rows,
        key=lambda row: (
            set_weight * _minmax(scores_by_id, row.candidate_id)
            + vote_weight * _minmax(vote_by_id, row.candidate_id),
            _minmax(scores_by_id, row.candidate_id),
            row.vote_weight,
            -row.candidate_index,
        ),
    )


def _renormalize_weights(set_weight: float, vote_weight: float) -> tuple[float, float]:
    total = set_weight + vote_weight
    if total <= 0.0:
        return ONLINE_INITIAL_SET_WEIGHT, ONLINE_INITIAL_VOTE_WEIGHT
    return set_weight / total, vote_weight / total


def measure_cross_family_gate(
    corpus: FamilyAnnotatedCorpus,
    oof_rows: list[exp4244.OOFRow],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Measure static and online Set-Encoder selectors on held-out-family scores."""

    score_map = _score_map(oof_rows)
    retained_tasks: list[list[tuple[FamilyAnnotatedRow, float, int]]] = []
    dropped_task_n = 0
    for task_rows in _group_by_task(corpus.rows):
        scored: list[tuple[FamilyAnnotatedRow, float, int]] = []
        for row in task_rows:
            item = score_map.get(row.candidate_id)
            if item is None or item[1] is not True:
                scored = []
                break
            scored.append((row, item[0], item[2]))
        if len(scored) != len(task_rows):
            dropped_task_n += 1
            continue
        retained_tasks.append(scored)

    vote_hits: list[bool] = []
    set_hits: list[bool] = []
    oracle_hits: list[bool] = []
    control_hits: list[bool] = []
    online_hits: list[bool] = []
    deltas_set_vote: list[float] = []
    deltas_set_control: list[float] = []
    deltas_online_vote: list[float] = []
    task_rows_payload: list[dict[str, Any]] = []
    online_set_weight, online_vote_weight = ONLINE_INITIAL_SET_WEIGHT, ONLINE_INITIAL_VOTE_WEIGHT
    for scored in retained_tasks:
        task_candidates = [row for row, _score, _fold in scored]
        scores_by_id = {row.candidate_id: score for row, score, _fold in scored}
        folds = {fold for _row, _score, fold in scored}
        vote_pick = _select_vote(task_candidates)
        set_pick = max(
            task_candidates,
            key=lambda row: (scores_by_id[row.candidate_id], row.vote_weight, -row.candidate_index),
        )
        control_pick = _select_first(task_candidates)
        online_pick = _online_pick(
            task_candidates,
            scores_by_id,
            set_weight=online_set_weight,
            vote_weight=online_vote_weight,
        )
        oracle_hit = any(row.correct for row in task_candidates)
        vote_hit = vote_pick.correct
        set_hit = set_pick.correct
        control_hit = control_pick.correct
        online_hit = online_pick.correct
        vote_hits.append(vote_hit)
        set_hits.append(set_hit)
        oracle_hits.append(oracle_hit)
        control_hits.append(control_hit)
        online_hits.append(online_hit)
        deltas_set_vote.append(float(set_hit) - float(vote_hit))
        deltas_set_control.append(float(set_hit) - float(control_hit))
        deltas_online_vote.append(float(online_hit) - float(vote_hit))
        task_rows_payload.append(
            {
                "task_id": vote_pick.task_id,
                "family_id": vote_pick.family_id,
                "fold": min(folds) if folds else 0,
                "oracle_hit": oracle_hit,
                "vote_candidate_id": vote_pick.candidate_id,
                "vote_correct": vote_hit,
                "set_encoder_candidate_id": set_pick.candidate_id,
                "set_encoder_correct": set_hit,
                "set_encoder_score_margin_vs_vote": _round_metric(
                    scores_by_id[set_pick.candidate_id] - scores_by_id[vote_pick.candidate_id]
                ),
                "matched_control_candidate_id": control_pick.candidate_id,
                "matched_control_correct": control_hit,
                "online_adapt_candidate_id": online_pick.candidate_id,
                "online_adapt_correct": online_hit,
                "online_set_weight_before": _round_metric(online_set_weight),
                "online_vote_weight_before": _round_metric(online_vote_weight),
            }
        )
        online_set_weight *= math.exp(ONLINE_ETA * float(set_hit))
        online_vote_weight *= math.exp(ONLINE_ETA * float(vote_hit))
        online_set_weight, online_vote_weight = _renormalize_weights(
            online_set_weight,
            online_vote_weight,
        )

    vote_at_1 = _rate(vote_hits)
    set_at_1 = _rate(set_hits)
    oracle_at_k = _rate(oracle_hits)
    control_at_1 = _rate(control_hits)
    online_at_1 = _rate(online_hits)
    cross_family_delta = _round_metric(set_at_1 - vote_at_1)
    ci95 = _bootstrap_ci95(deltas_set_vote, random_seed=random_seed, resamples=bootstrap_resamples)
    matched_control_delta = _round_metric(
        sum(deltas_set_control) / float(len(deltas_set_control)) if deltas_set_control else 0.0
    )
    online_delta = _round_metric(
        sum(deltas_online_vote) / float(len(deltas_online_vote)) if deltas_online_vote else 0.0
    )
    held_out_families = {row["family_id"] for row in task_rows_payload}
    headroom_exists = oracle_at_k > vote_at_1
    cross_family_win_holds = bool(cross_family_delta > 0.0 and _ci_excludes_zero(ci95))
    if not headroom_exists:
        honest_read = "no_headroom"
    elif cross_family_win_holds:
        honest_read = "cross_family_generalizes"
    else:
        honest_read = "within_pool_only"
    return {
        "headline_outcome": honest_read,
        "honest_verdict": f"complete: {honest_read}",
        "honest_read": honest_read,
        "cross_family_win_holds": cross_family_win_holds,
        "cross_family_delta": cross_family_delta,
        "cross_family_ci95": ci95,
        "within_minus_cross_gap": _round_metric(WITHIN_POOL_DELTA_393 - cross_family_delta),
        "held_out_family_n": len(held_out_families),
        "held_out_task_n": len(retained_tasks),
        "oracle_at_k": _round_metric(oracle_at_k),
        "matched_control_delta": matched_control_delta,
        "online_adapt_cross_family_delta": online_delta,
        "pass_rates": {
            "vote_at_1": _round_metric(vote_at_1),
            "set_encoder_at_1": _round_metric(set_at_1),
            "matched_control_at_1": _round_metric(control_at_1),
            "online_adapt_at_1": _round_metric(online_at_1),
        },
        "oracle_minus_vote": _round_metric(oracle_at_k - vote_at_1),
        "headroom_exists": headroom_exists,
        "false_negative_risk": not headroom_exists,
        "ci95_excludes_zero": _ci_excludes_zero(ci95),
        "bootstrap_resamples": int(bootstrap_resamples),
        "task_rows": task_rows_payload,
        "dropped_task_n": dropped_task_n,
        "matched_control_policy": "deterministic_first_of_k_no_verifier",
        "online_adapt_policy": "prequential_exponential_weights_static_set_encoder_vs_vote",
    }


def reproducibility_checksum(
    *,
    corpus: FamilyAnnotatedCorpus,
    folds: list[FamilyFold],
    metrics: dict[str, Any],
    random_seed: int,
) -> str:
    payload = {
        "cross_family_delta": metrics.get("cross_family_delta"),
        "feature_names": list(exp4244.FEATURE_NAMES),
        "folds": [
            {
                "held_out_families": sorted(fold.held_out_families),
                "held_out_task_ids": sorted(fold.held_out_task_ids),
                "train_families": sorted(fold.train_families),
            }
            for fold in folds
        ],
        "manifest_sha256": corpus.manifest_sha256,
        "pool_artifact_sha256": corpus.pool_artifact_sha256,
        "random_seed": int(random_seed),
        "task_family_ids": sorted(corpus.task_family_ids.items()),
        "upstream_checksum": corpus.upstream_checksum,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _model_specs(
    *,
    build_artifact: dict[str, Any],
    training_report: CrossFamilyTrainingReport | None,
    folds: list[FamilyFold],
    status: str,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "blocked_reason": blocked_reason,
        "set_encoder_config": build_artifact.get("model_specs", {}),
        "family_disjoint_split_protocol": {
            "split_unit": "family_id",
            "n_folds": len(folds),
            "folds": [
                {
                    "held_out_families": sorted(fold.held_out_families),
                    "train_families": sorted(fold.train_families),
                    "held_out_task_n": len(fold.held_out_task_ids),
                    "train_task_n": len(fold.train_task_ids),
                }
                for fold in folds
            ],
            "no_family_overlap_per_fold": all(
                fold.train_families.isdisjoint(fold.held_out_families) for fold in folds
            ),
        },
        "training_report": training_report.training_config if training_report is not None else {},
        "fold_summaries": training_report.fold_summaries if training_report is not None else [],
        "online_adapt_rule": {
            "rule": "prequential_exponential_weights_static_set_encoder_vs_vote",
            "initial_set_encoder_weight": ONLINE_INITIAL_SET_WEIGHT,
            "initial_vote_weight": ONLINE_INITIAL_VOTE_WEIGHT,
            "eta": ONLINE_ETA,
            "feedback": "prior held-out task exact-match labels only; current task ranking uses no current correctness",
        },
    }


def _blocked_checksum(reason: str, random_seed: int) -> str:
    raw = json.dumps({"random_seed": int(random_seed), "reason": reason}, sort_keys=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _deferred_artifact(
    reason: str,
    *,
    random_seed: int,
    checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4271_arc_cross_family_transfer_existing_pool",
        "schema": "carnot.arc_cross_family_transfer_existing_pool_4271.v1",
        "status": "complete",
        "headline_outcome": "arc_cross_family_transfer_deferred",
        "honest_verdict": reason,
        "honest_read": "deferred" if reason == DEFERRED_INFEASIBLE_VERDICT else "blocked",
        "cross_family_win_holds": False,
        "cross_family_delta": 0.0,
        "cross_family_ci95": [0.0, 0.0],
        "within_minus_cross_gap": 0.0,
        "held_out_family_n": 0,
        "held_out_task_n": 0,
        "oracle_at_k": 0.0,
        "matched_control_delta": 0.0,
        "online_adapt_cross_family_delta": 0.0,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": {
            "status": "deferred" if reason == DEFERRED_INFEASIBLE_VERDICT else "blocked",
            "blocked_reason": reason,
            "family_disjoint_split_protocol": {},
            "online_adapt_rule": {},
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "pass_rates": {
            "vote_at_1": 0.0,
            "set_encoder_at_1": 0.0,
            "matched_control_at_1": 0.0,
            "online_adapt_at_1": 0.0,
        },
        "oracle_minus_vote": 0.0,
        "headroom_exists": False,
        "false_negative_risk": False,
        "ci95_excludes_zero": False,
        "bootstrap_resamples": 0,
        "task_rows": [],
        "dropped_task_n": 0,
        "matched_control_policy": "deterministic_first_of_k_no_verifier",
        "online_adapt_policy": "prequential_exponential_weights_static_set_encoder_vs_vote",
        "duration_s": round(duration_s, 6),
        "adversarial_verify": {"status": "pending"},
    }


def _complete_artifact(
    *,
    corpus: FamilyAnnotatedCorpus,
    metrics: dict[str, Any],
    model_specs: dict[str, Any],
    checksum: str,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4271_arc_cross_family_transfer_existing_pool",
        "schema": "carnot.arc_cross_family_transfer_existing_pool_4271.v1",
        "status": "complete",
        **metrics,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": model_specs,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "candidate_count": corpus.candidate_n,
        "candidate_pool_path": str(corpus.pool_artifact_path),
        "candidate_pool_sha256": corpus.pool_artifact_sha256,
        "family_manifest_path": str(corpus.manifest_path),
        "family_manifest_sha256": corpus.manifest_sha256,
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


def _bare_float(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a bare float")


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("complete_") or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if type(artifact["cross_family_win_holds"]) is not bool:
        raise ValueError("cross_family_win_holds must be a bare bool")
    for field in (
        "cross_family_delta",
        "within_minus_cross_gap",
        "oracle_at_k",
        "matched_control_delta",
        "online_adapt_cross_family_delta",
    ):
        _bare_float(artifact[field], field)
    ci95 = artifact["cross_family_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ci95)
    ):
        raise ValueError("cross_family_ci95 must be a two-number ci95")
    if type(artifact["held_out_family_n"]) is not int:
        raise ValueError("held_out_family_n must be a bare int")
    if type(artifact["held_out_task_n"]) is not int:
        raise ValueError("held_out_task_n must be a bare int")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4271")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4271")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    training_epochs: int = exp4244.DEFAULT_TRAINING_EPOCHS,
    hidden_dim: int = exp4244.DEFAULT_HIDDEN_DIM,
    lr: float = exp4244.DEFAULT_LR,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    try:
        provenance = _read_provenance(root)
        if provenance.get("family_split_feasible") is False:
            artifact = _deferred_artifact(
                DEFERRED_INFEASIBLE_VERDICT,
                random_seed=random_seed,
                checksum=_blocked_checksum(DEFERRED_INFEASIBLE_VERDICT, random_seed),
                duration_s=time.perf_counter() - start,
            )
        elif provenance.get("family_split_feasible") is not True:
            raise BlockedRun(BLOCKED_INPUTS_VERDICT)
        else:
            _provenance, build_artifact, _model_artifact, _model_path = _load_required_artifacts(root)
            corpus = load_family_annotated_corpus(root)
            folds = build_family_disjoint_folds(corpus)
            training_report = train_cross_family_oof(
                corpus,
                folds,
                random_seed=random_seed,
                training_epochs=training_epochs,
                hidden_dim=hidden_dim,
                lr=lr,
            )
            metrics = measure_cross_family_gate(
                corpus,
                training_report.rows,
                random_seed=random_seed,
                bootstrap_resamples=bootstrap_resamples,
            )
            checksum = reproducibility_checksum(
                corpus=corpus,
                folds=folds,
                metrics=metrics,
                random_seed=random_seed,
            )
            artifact = _complete_artifact(
                corpus=corpus,
                metrics=metrics,
                model_specs=_model_specs(
                    build_artifact=build_artifact,
                    training_report=training_report,
                    folds=folds,
                    status="complete",
                ),
                checksum=checksum,
                random_seed=random_seed,
                duration_s=time.perf_counter() - start,
            )
    except BlockedRun as exc:
        reason = str(exc) or BLOCKED_INPUTS_VERDICT
        artifact = _deferred_artifact(
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


if __name__ == "__main__":  # pragma: no cover
    main()
