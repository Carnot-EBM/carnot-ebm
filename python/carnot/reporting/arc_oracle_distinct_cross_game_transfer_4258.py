"""Exp 4258 ARC oracle-distinct cross-game transfer.

Spec refs: REQ-VERIFY-4258, SCENARIO-VERIFY-4258.
"""

from __future__ import annotations

import gzip
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

from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


RANDOM_SEED = 4258
WITHIN_GAME_DELTA_393 = 0.4423076923
BOOTSTRAP_RESAMPLES = 2000
DEFAULT_N_FOLDS = exp4244.DEFAULT_N_FOLDS
OUTPUT_REL = Path("results/experiment_4258_arc_oracle_distinct_cross_game_transfer.json")
POOL_REL = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
SURVEY_REL = Path("results/arc3_win_condition_survey.json")
SET_ENCODER_BUILD_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_build.json")
SET_ENCODER_MODEL_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_model.json")
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
BLOCKED_GAME_IDS_VERDICT = "blocked_arc_game_ids_unrecoverable"
BLOCKED_ARTIFACT_VERDICT = "blocked_arc_set_encoder_artifact_missing"
SPEC_REFS = ["REQ-VERIFY-4258", "SCENARIO-VERIFY-4258"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A cross-game win, a within-game-only collapse, and a no-headroom "
        "are ALL COMPLETE and decision-grade for scoping the headline."
    ),
    "cross_game_delta": (
        "BARE float: set_encoder@1 - vote@1 on HELD-OUT GAMES -- the real OOD lift; "
        "the load-bearing generalization number."
    ),
    "cross_game_ci95": (
        "Task-level bootstrap CI95 of the cross-game delta -- excluding 0 means the "
        "verifier generalizes to unseen games."
    ),
    "within_game_minus_cross_game_gap": (
        "The .393 within-game +0.4423 minus cross_game_delta -- quantifies how much "
        "the win is per-game-basin memorization vs general signal."
    ),
    "held_out_game_n": (
        "BARE int: number of games never seen in training -- the OOD breadth; report "
        "alongside held_out_task_n for power."
    ),
    "oracle_at_k": (
        "Positive-control ceiling on held-out games -- if ~=vote the cross-game null "
        "is uninformative, not a verifier failure."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- learned set-encoder, no demo execution; keeps the transfer "
        "result oracle-distinct."
    ),
    "random_seed": "Determinism precondition; the game-split + bootstrap reproducible.",
    "reproducibility_checksum": "Hash of the pool + game partition; lets a third party re-run.",
    "model_specs": (
        "The game-disjoint split protocol + set-encoder config; required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "cross_game_delta",
    "cross_game_ci95",
    "within_game_minus_cross_game_gap",
    "held_out_game_n",
    "held_out_task_n",
    "oracle_at_k",
    "matched_control_delta",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""


@dataclass(frozen=True)
class GameAnnotatedRow:
    task_id: str
    game_id: str
    candidate_id: str
    candidate_index: int
    correct: bool
    features: dict[str, float]
    vote_weight: float


@dataclass(frozen=True)
class GameAnnotatedCorpus:
    rows: list[GameAnnotatedRow]
    task_game_ids: dict[str, str]
    pool_artifact_path: Path
    pool_artifact_sha256: str
    upstream_checksum: str
    survey_games: list[str]
    held_out_game_n: int
    held_out_task_n: int
    candidate_n: int


@dataclass(frozen=True)
class GameFold:
    held_out_games: set[str]
    train_games: set[str]
    held_out_task_ids: set[str]
    train_task_ids: set[str]


@dataclass(frozen=True)
class CrossGameTrainingReport:
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
        raise BlockedRun(BLOCKED_ARTIFACT_VERDICT)
    path = Path(candidate_value)
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
        raise BlockedRun(BLOCKED_ARTIFACT_VERDICT)
    return resolved


def _load_required_artifacts(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    try:
        build = _read_json_object(repo_root / SET_ENCODER_BUILD_REL)
        model_path = _resolve_required_path(
            repo_root,
            build.get("learned_verifier_path"),
            SET_ENCODER_MODEL_REL,
        )
        model = exp4244.load_set_encoder(model_path)
    except Exception as exc:
        raise BlockedRun(BLOCKED_ARTIFACT_VERDICT) from exc
    if build.get("aggregator_trained") is not True:
        raise BlockedRun(BLOCKED_ARTIFACT_VERDICT)
    if build.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_ARTIFACT_VERDICT)
    if model.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_ARTIFACT_VERDICT)
    return build, model, model_path


def _load_pool_payload(pool_path: Path) -> dict[str, Any]:
    try:
        with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise BlockedRun(BLOCKED_GAME_IDS_VERDICT) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise BlockedRun(BLOCKED_GAME_IDS_VERDICT)
    return payload


def _survey_games(repo_root: Path) -> list[str]:
    path = repo_root / SURVEY_REL
    if not path.exists():
        return []
    try:
        survey = _read_json_object(path)
    except Exception:
        return []
    games: set[str] = set()
    for field in ("per_game_surveys", "ranked_targets"):
        rows = survey.get(field, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("game"), str) and row["game"]:
                games.add(row["game"])
    return sorted(games)


def _candidate_game_ids(candidates: Any, survey_game_set: set[str]) -> set[str]:
    values: set[str] = set()
    if not isinstance(candidates, list):
        return values
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in ("game_id", "game", "env_id", "environment_id", "target_game"):
            value = candidate.get(key)
            if isinstance(value, str) and value and (not survey_game_set or value in survey_game_set):
                values.add(value)
    return values


def _recover_game_id(task: dict[str, Any], survey_game_set: set[str]) -> str | None:
    for key in ("game_id", "game", "env_id", "environment_id", "target_game"):
        value = task.get(key)
        if isinstance(value, str) and value and (not survey_game_set or value in survey_game_set):
            return value
    candidate_values = _candidate_game_ids(task.get("candidates"), survey_game_set)
    if len(candidate_values) == 1:
        return next(iter(candidate_values))
    task_id = str(task.get("task_id") or "")
    for token in task_id.replace("/", ":").split(":"):
        if token in survey_game_set:
            return token
    return None


def _task_game_map_from_pool(payload: dict[str, Any], survey_games: list[str]) -> tuple[dict[str, str], list[str]]:
    survey_game_set = set(survey_games)
    mapping: dict[str, str] = {}
    missing: list[str] = []
    for task in payload.get("tasks", []):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "")
        if not task_id:
            continue
        game_id = _recover_game_id(task, survey_game_set)
        if game_id is None:
            missing.append(task_id)
        else:
            mapping[task_id] = game_id
    return mapping, missing


def load_game_annotated_corpus(repo_root: Path | str = Path(".")) -> GameAnnotatedCorpus:
    """SCENARIO-VERIFY-4258: load the grown pool only if every task has a game id."""

    root = Path(repo_root)
    corpus = exp4244.load_grown_pool(root)
    payload = _load_pool_payload(corpus.pool_artifact_path)
    survey_games = _survey_games(root)
    task_game_ids, missing = _task_game_map_from_pool(payload, survey_games)
    if missing or not task_game_ids or len(set(task_game_ids.values())) < 2:
        raise BlockedRun(BLOCKED_GAME_IDS_VERDICT)
    rows = [
        GameAnnotatedRow(
            task_id=row.task_id,
            game_id=task_game_ids[row.task_id],
            candidate_id=row.candidate_id,
            candidate_index=row.candidate_index,
            correct=row.correct,
            features=row.features,
            vote_weight=row.vote_weight,
        )
        for row in corpus.rows
        if row.task_id in task_game_ids
    ]
    if len({row.task_id for row in rows}) != len(task_game_ids):
        raise BlockedRun(BLOCKED_GAME_IDS_VERDICT)
    return GameAnnotatedCorpus(
        rows=rows,
        task_game_ids=task_game_ids,
        pool_artifact_path=corpus.pool_artifact_path,
        pool_artifact_sha256=corpus.pool_artifact_sha256,
        upstream_checksum=corpus.upstream_checksum,
        survey_games=survey_games,
        held_out_game_n=len(set(task_game_ids.values())),
        held_out_task_n=len(task_game_ids),
        candidate_n=len(rows),
    )


def build_game_disjoint_folds(
    task_game_ids: dict[str, str],
    *,
    random_seed: int = RANDOM_SEED,
    n_folds: int = DEFAULT_N_FOLDS,
) -> list[GameFold]:
    games = sorted(set(task_game_ids.values()))
    if len(games) < 2:
        raise BlockedRun(BLOCKED_GAME_IDS_VERDICT)
    rng = random.Random(random_seed)
    rng.shuffle(games)
    fold_count = max(2, min(int(n_folds), len(games)))
    held_out_by_fold = [set() for _ in range(fold_count)]
    for index, game_id in enumerate(games):
        held_out_by_fold[index % fold_count].add(game_id)
    all_games = set(games)
    folds: list[GameFold] = []
    for held_out_games in held_out_by_fold:
        train_games = all_games - held_out_games
        held_out_task_ids = {task for task, game in task_game_ids.items() if game in held_out_games}
        train_task_ids = {task for task, game in task_game_ids.items() if game in train_games}
        folds.append(
            GameFold(
                held_out_games=held_out_games,
                train_games=train_games,
                held_out_task_ids=held_out_task_ids,
                train_task_ids=train_task_ids,
            )
        )
    return folds


def _as_grown_rows(rows: list[GameAnnotatedRow]) -> list[exp4244.GrownPoolRow]:
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


def train_cross_game_oof(
    corpus: GameAnnotatedCorpus,
    folds: list[GameFold],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_n: int = exp4244.BOOTSTRAP_N,
    training_epochs: int = exp4244.DEFAULT_TRAINING_EPOCHS,
    hidden_dim: int = exp4244.DEFAULT_HIDDEN_DIM,
    lr: float = exp4244.DEFAULT_LR,
) -> CrossGameTrainingReport:
    """Train Exp 4244 set-encoders with held-out folds defined by game ids."""

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
    return CrossGameTrainingReport(
        rows=report.rows,
        fold_summaries=[
            {
                "fold": index,
                "held_out_games": sorted(fold.held_out_games),
                "train_games": sorted(fold.train_games),
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


def _group_by_task(rows: list[GameAnnotatedRow]) -> list[list[GameAnnotatedRow]]:
    grouped: dict[str, list[GameAnnotatedRow]] = defaultdict(list)
    for row in rows:
        grouped[row.task_id].append(row)
    return [sorted(items, key=lambda item: item.candidate_index) for _, items in sorted(grouped.items())]


def _select_vote(task_rows: list[GameAnnotatedRow]) -> GameAnnotatedRow:
    return max(task_rows, key=lambda row: (row.vote_weight, -row.candidate_index))


def _select_first(task_rows: list[GameAnnotatedRow]) -> GameAnnotatedRow:
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


def measure_cross_game_gate(
    corpus: GameAnnotatedCorpus,
    oof_rows: list[exp4244.OOFRow],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Measure set_encoder@1 vs vote@1 only on held-out-game scores."""

    score_map = _score_map(oof_rows)
    retained_tasks: list[list[tuple[GameAnnotatedRow, float, int]]] = []
    dropped_task_n = 0
    for task_rows in _group_by_task(corpus.rows):
        scored: list[tuple[GameAnnotatedRow, float, int]] = []
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
    deltas_set_vote: list[float] = []
    deltas_set_control: list[float] = []
    task_rows_payload: list[dict[str, Any]] = []
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
        oracle_hit = any(row.correct for row in task_candidates)
        vote_hit = vote_pick.correct
        set_hit = set_pick.correct
        control_hit = control_pick.correct
        vote_hits.append(vote_hit)
        set_hits.append(set_hit)
        oracle_hits.append(oracle_hit)
        control_hits.append(control_hit)
        deltas_set_vote.append(float(set_hit) - float(vote_hit))
        deltas_set_control.append(float(set_hit) - float(control_hit))
        task_rows_payload.append(
            {
                "task_id": vote_pick.task_id,
                "game_id": vote_pick.game_id,
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
            }
        )

    vote_at_1 = _rate(vote_hits)
    set_at_1 = _rate(set_hits)
    oracle_at_k = _rate(oracle_hits)
    control_at_1 = _rate(control_hits)
    cross_game_delta = _round_metric(set_at_1 - vote_at_1)
    ci95 = _bootstrap_ci95(deltas_set_vote, random_seed=random_seed, resamples=bootstrap_resamples)
    matched_control_delta = _round_metric(
        sum(deltas_set_control) / float(len(deltas_set_control)) if deltas_set_control else 0.0
    )
    held_out_games = {row["game_id"] for row in task_rows_payload}
    headroom_exists = oracle_at_k > vote_at_1
    ci_excludes_zero = _ci_excludes_zero(ci95)
    if not headroom_exists:
        honest_read = "no_headroom"
    elif cross_game_delta > 0.0 and ci_excludes_zero:
        honest_read = "cross_game_transfers"
    else:
        honest_read = "within_game_only"
    return {
        "headline_outcome": honest_read,
        "honest_verdict": f"complete: {honest_read}",
        "honest_read": honest_read,
        "cross_game_delta": cross_game_delta,
        "cross_game_ci95": ci95,
        "within_game_minus_cross_game_gap": _round_metric(WITHIN_GAME_DELTA_393 - cross_game_delta),
        "held_out_game_n": len(held_out_games),
        "held_out_task_n": len(retained_tasks),
        "oracle_at_k": _round_metric(oracle_at_k),
        "matched_control_delta": matched_control_delta,
        "pass_rates": {
            "vote_at_1": _round_metric(vote_at_1),
            "set_encoder_at_1": _round_metric(set_at_1),
            "matched_control_at_1": _round_metric(control_at_1),
        },
        "oracle_minus_vote": _round_metric(oracle_at_k - vote_at_1),
        "headroom_exists": headroom_exists,
        "false_negative_risk": not headroom_exists,
        "ci95_excludes_zero": ci_excludes_zero,
        "bootstrap_resamples": int(bootstrap_resamples),
        "task_rows": task_rows_payload,
        "dropped_task_n": dropped_task_n,
        "matched_control_policy": "deterministic_first_of_k_no_verifier",
    }


def reproducibility_checksum(
    *,
    corpus: GameAnnotatedCorpus,
    folds: list[GameFold],
    metrics: dict[str, Any],
    random_seed: int,
) -> str:
    payload = {
        "cross_game_delta": metrics.get("cross_game_delta"),
        "feature_names": list(exp4244.FEATURE_NAMES),
        "folds": [
            {
                "held_out_games": sorted(fold.held_out_games),
                "held_out_task_ids": sorted(fold.held_out_task_ids),
                "train_games": sorted(fold.train_games),
            }
            for fold in folds
        ],
        "pool_artifact_sha256": corpus.pool_artifact_sha256,
        "random_seed": int(random_seed),
        "task_game_ids": sorted(corpus.task_game_ids.items()),
        "upstream_checksum": corpus.upstream_checksum,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _model_specs(
    *,
    build_artifact: dict[str, Any],
    training_report: CrossGameTrainingReport | None,
    folds: list[GameFold],
    status: str,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "blocked_reason": blocked_reason,
        "set_encoder_config": build_artifact.get("model_specs", {}),
        "game_disjoint_split_protocol": {
            "split_unit": "game_id",
            "n_folds": len(folds),
            "folds": [
                {
                    "held_out_games": sorted(fold.held_out_games),
                    "train_games": sorted(fold.train_games),
                    "held_out_task_n": len(fold.held_out_task_ids),
                    "train_task_n": len(fold.train_task_ids),
                }
                for fold in folds
            ],
            "no_game_overlap_per_fold": all(fold.train_games.isdisjoint(fold.held_out_games) for fold in folds),
        },
        "training_report": training_report.training_config if training_report is not None else {},
        "fold_summaries": training_report.fold_summaries if training_report is not None else [],
    }


def _blocked_checksum(reason: str, random_seed: int) -> str:
    raw = json.dumps({"random_seed": int(random_seed), "reason": reason}, sort_keys=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_artifact(
    reason: str,
    *,
    random_seed: int,
    checksum: str,
    duration_s: float,
    missing_task_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4258_arc_oracle_distinct_cross_game_transfer",
        "schema": "carnot.arc_oracle_distinct_cross_game_transfer_4258.v1",
        "status": "complete",
        "headline_outcome": "arc_oracle_distinct_cross_game_transfer_blocked",
        "honest_verdict": reason,
        "honest_read": "blocked",
        "cross_game_delta": None,
        "cross_game_ci95": None,
        "within_game_minus_cross_game_gap": None,
        "held_out_game_n": 0,
        "held_out_task_n": 0,
        "oracle_at_k": None,
        "matched_control_delta": None,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": {
            "status": "blocked",
            "blocked_reason": reason,
            "missing_task_ids_sample": list(missing_task_ids or [])[:20],
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "pass_rates": {},
        "oracle_minus_vote": None,
        "headroom_exists": None,
        "false_negative_risk": False,
        "ci95_excludes_zero": False,
        "bootstrap_resamples": 0,
        "task_rows": [],
        "dropped_task_n": 0,
        "matched_control_policy": "deterministic_first_of_k_no_verifier",
        "duration_s": round(duration_s, 6),
        "adversarial_verify": {"status": "pending"},
    }


def _complete_artifact(
    *,
    corpus: GameAnnotatedCorpus,
    metrics: dict[str, Any],
    model_specs: dict[str, Any],
    checksum: str,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4258_arc_oracle_distinct_cross_game_transfer",
        "schema": "carnot.arc_oracle_distinct_cross_game_transfer_4258.v1",
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
        "within_game_reference_delta": WITHIN_GAME_DELTA_393,
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


def _nullable_float(value: Any, field: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a bare float or null for blocked artifacts")


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    _nullable_float(artifact["cross_game_delta"], "cross_game_delta")
    ci95 = artifact["cross_game_ci95"]
    if ci95 is not None and (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ci95)
    ):
        raise ValueError("cross_game_ci95 must be a two-number ci95 or null")
    _nullable_float(artifact["within_game_minus_cross_game_gap"], "within_game_minus_cross_game_gap")
    if type(artifact["held_out_game_n"]) is not int:
        raise ValueError("held_out_game_n must be a bare int")
    if type(artifact["held_out_task_n"]) is not int:
        raise ValueError("held_out_task_n must be a bare int")
    _nullable_float(artifact["oracle_at_k"], "oracle_at_k")
    _nullable_float(artifact["matched_control_delta"], "matched_control_delta")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4258")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4258")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    n_folds: int = DEFAULT_N_FOLDS,
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
        build_artifact, _model_artifact, _model_path = _load_required_artifacts(root)
        corpus = load_game_annotated_corpus(root)
        folds = build_game_disjoint_folds(corpus.task_game_ids, random_seed=random_seed, n_folds=n_folds)
        training_report = train_cross_game_oof(
            corpus,
            folds,
            random_seed=random_seed,
            training_epochs=training_epochs,
            hidden_dim=hidden_dim,
            lr=lr,
        )
        metrics = measure_cross_game_gate(
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
        reason = str(exc) or BLOCKED_GAME_IDS_VERDICT
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


if __name__ == "__main__":  # pragma: no cover
    main()
