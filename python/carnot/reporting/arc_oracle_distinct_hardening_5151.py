"""Exp 5151 ARC Set-Encoder oracle-distinct hardening.

Spec refs: REQ-VERIFY-5151, SCENARIO-VERIFY-5151,
SCENARIO-VERIFY-5151-CROSS-GAME-BLOCKED,
SCENARIO-VERIFY-5151-UPSTREAM-BLOCKED.
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

from carnot.reporting import arc_oracle_distinct_cross_game_transfer_4258 as exp4258
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


RANDOM_SEED = 5151
DEFAULT_RANDOM_SEEDS = [5151, 5152, 5153, 5154, 5155]
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_N_FOLDS = exp4244.DEFAULT_N_FOLDS
EXP4245_REL = Path("results/experiment_4245_arc_set_encoder_beats_vote.json")
POOL_REL = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
SET_ENCODER_BUILD_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_build.json")
OUTPUT_REL = Path("results/experiment_5151_arc_oracle_distinct_hardening_v472.json")
INFERENCE_SUBSTRATE = "cached_grown_arc_pool_cpu_multiseed_set_encoder_hardening"
SOLVE_PROVENANCE = "development_proxy"
BLOCKED_UPSTREAM_VERDICT = "blocked_upstream_artifact_missing"
BLOCKED_GAME_IDS_VERDICT = "blocked_arc_game_ids_unrecoverable"
SPEC_REFS = [
    "REQ-VERIFY-5151",
    "SCENARIO-VERIFY-5151",
    "SCENARIO-VERIFY-5151-CROSS-GAME-BLOCKED",
    "SCENARIO-VERIFY-5151-UPSTREAM-BLOCKED",
]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ AND state plainly whether "
        "the +44pp win survives hardening or not -- do not bury a null in qualifiers."
    ),
    "multiseed_delta_ci95": (
        "A single-seed CI is not a real interval; multiseed pooling is the minimum bar "
        "before this result can inform a scaling decision (DiffusionGemma, exp5152)."
    ),
    "leak_audit_passed": (
        "An unaudited held-out split cannot distinguish a real win from a leaked one."
    ),
    "cross_game_replication_delta": (
        "A win on one pool that vanishes on a second is a corpus artifact, not a general finding."
    ),
    "verifier_is_oracle": (
        "Must stay oracle-distinct (per the Circularity Discipline) to count toward the moat thesis."
    ),
    "solve_provenance": (
        "This is offline pool-scoring on already-generated candidates, not a live hidden-game solve."
    ),
    "inference_substrate": "Declare accurately per the Inference-Substrate Declaration Discipline.",
    "reproducibility_checksum": (
        "Hash of the fixed candidate pool, row-level leak audit, multiseed splits, exact test, "
        "and cross-game status so downstream scaling gates can detect drift."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "multiseed_delta_ci95",
    "leak_audit_passed",
    "cross_game_replication_delta",
    "verifier_is_oracle",
    "solve_provenance",
    "inference_substrate",
    "random_seeds_used",
    "per_seed_results",
    "cluster_bootstrap_delta_ci95",
    "exact_test_discordant_wins",
    "exact_test_discordant_losses",
    "exact_test_p_value",
    "cross_game_blocked_reason",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)


class BlockedRun(RuntimeError):
    """Expected missing-artifact path that still writes a terminal artifact."""


@dataclass(frozen=True)
class CleanCandidate:
    task_id: str
    candidate_id: str
    candidate_index: int
    vote_weight: float
    correct: bool


@dataclass(frozen=True)
class SeedHardeningResult:
    random_seed: int
    auroc: float
    delta: float
    held_out_task_n: int
    vote_at_1: float
    set_encoder_at_1: float
    oracle_at_k: float
    fold_task_ids: list[list[str]]


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


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BlockedRun(BLOCKED_UPSTREAM_VERDICT) from exc
    if not isinstance(payload, dict):
        raise BlockedRun(BLOCKED_UPSTREAM_VERDICT)
    return payload


def _read_gzip_json_object(path: Path) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise BlockedRun(BLOCKED_UPSTREAM_VERDICT) from exc
    if not isinstance(payload, dict):
        raise BlockedRun(BLOCKED_UPSTREAM_VERDICT)
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_required_path(repo_root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise BlockedRun(BLOCKED_UPSTREAM_VERDICT)
    path = Path(value)
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
        raise BlockedRun(BLOCKED_UPSTREAM_VERDICT)
    return resolved


def _load_required_inputs(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    exp4245_path = repo_root / EXP4245_REL
    pool_path = repo_root / POOL_REL
    if not exp4245_path.exists() or not pool_path.exists():
        raise BlockedRun(BLOCKED_UPSTREAM_VERDICT)
    exp4245 = _read_json_object(exp4245_path)
    pool_payload = _read_gzip_json_object(pool_path)
    build = _read_json_object(repo_root / SET_ENCODER_BUILD_REL)
    if build.get("aggregator_trained") is not True or build.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_UPSTREAM_VERDICT)
    model_path = _resolve_required_path(repo_root, build.get("learned_verifier_path"))
    model = _read_json_object(model_path)
    if (
        model.get("verifier_is_oracle") is not False
        or exp4245.get("verifier_is_oracle") is not False
    ):
        raise BlockedRun(BLOCKED_UPSTREAM_VERDICT)
    if not isinstance(pool_payload.get("tasks"), list):
        raise BlockedRun(BLOCKED_UPSTREAM_VERDICT)
    return exp4245, pool_payload, model, model_path


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _t_critical_975(df: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.16,
        14: 2.145,
        15: 2.131,
        16: 2.12,
        17: 2.11,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.08,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.06,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    return table.get(max(1, int(df)), 1.96)


def _multiseed_ci95(deltas: list[float]) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    if len(deltas) == 1:
        point = _round_metric(deltas[0])
        return [point, point]
    mean = _mean(deltas)
    variance = sum((delta - mean) ** 2 for delta in deltas) / float(len(deltas) - 1)
    half_width = _t_critical_975(len(deltas) - 1) * math.sqrt(variance) / math.sqrt(len(deltas))
    return [_round_metric(mean - half_width), _round_metric(mean + half_width)]


def _cluster_bootstrap_ci95(
    deltas: list[float], *, random_seed: int, resamples: int
) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    if len(deltas) == 1:
        point = _round_metric(deltas[0])
        return [point, point]
    rng = random.Random(random_seed)
    n = len(deltas)
    samples = [
        sum(deltas[rng.randrange(n)] for _ in range(n)) / float(n) for _ in range(int(resamples))
    ]
    if not samples:
        point = _mean(deltas)
        return [_round_metric(point), _round_metric(point)]
    samples.sort()
    return [
        _round_metric(samples[int(0.025 * (len(samples) - 1))]),
        _round_metric(samples[int(0.975 * (len(samples) - 1))]),
    ]


def _ci_excludes_zero(ci95: list[float] | None) -> bool:
    return bool(ci95 and len(ci95) == 2 and (ci95[0] > 0.0 or ci95[1] < 0.0))


def _two_sided_binomial_p(wins: int, losses: int) -> float:
    n = int(wins) + int(losses)
    if n <= 0:
        return 1.0
    tail = min(int(wins), int(losses))
    probability = 2.0 * sum(math.comb(n, index) for index in range(tail + 1)) / float(2**n)
    return min(1.0, probability)


def _clean_candidates_from_rows(rows: list[exp4244.GrownPoolRow]) -> list[CleanCandidate]:
    return [
        CleanCandidate(
            task_id=row.task_id,
            candidate_id=row.candidate_id,
            candidate_index=row.candidate_index,
            vote_weight=row.vote_weight,
            correct=row.correct,
        )
        for row in rows
    ]


def _group_clean_candidates(candidates: list[CleanCandidate]) -> list[list[CleanCandidate]]:
    grouped: dict[str, list[CleanCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.task_id].append(candidate)
    return [
        sorted(task_candidates, key=lambda item: item.candidate_index)
        for _, task_candidates in sorted(grouped.items())
    ]


def _task_metrics_from_scores(
    candidates: list[CleanCandidate],
    score_map: dict[str, tuple[float, bool, int]],
) -> dict[str, Any]:
    vote_hits: list[bool] = []
    set_hits: list[bool] = []
    oracle_hits: list[bool] = []
    retained_candidates = 0
    for task_candidates in _group_clean_candidates(candidates):
        scored: list[tuple[CleanCandidate, float]] = []
        for candidate in task_candidates:
            item = score_map.get(candidate.candidate_id)
            if item is None or item[1] is not True:
                scored = []
                break
            scored.append((candidate, item[0]))
        if len(scored) != len(task_candidates):
            continue
        retained_candidates += len(scored)
        vote_pick = max(task_candidates, key=lambda item: (item.vote_weight, -item.candidate_index))
        set_pick, _set_score = max(
            scored,
            key=lambda item: (item[1], item[0].vote_weight, -item[0].candidate_index),
        )
        vote_hits.append(vote_pick.correct)
        set_hits.append(set_pick.correct)
        oracle_hits.append(any(candidate.correct for candidate in task_candidates))
    vote_rate = _mean([float(hit) for hit in vote_hits])
    set_rate = _mean([float(hit) for hit in set_hits])
    return {
        "delta": _round_metric(set_rate - vote_rate),
        "held_out_task_n": len(vote_hits),
        "candidate_count": retained_candidates,
        "vote_at_1": _round_metric(vote_rate),
        "set_encoder_at_1": _round_metric(set_rate),
        "oracle_at_k": _round_metric(_mean([float(hit) for hit in oracle_hits])),
    }


def _measure_seed_oof(
    rows: list[exp4244.GrownPoolRow],
    oof_rows: list[exp4244.OOFRow],
) -> dict[str, Any]:
    score_map = {
        row.candidate_id: (float(row.score), row.task_id not in row.train_task_ids, int(row.fold))
        for row in oof_rows
    }
    return _task_metrics_from_scores(_clean_candidates_from_rows(rows), score_map)


def _train_seed_hardening(
    corpus: exp4244.GrownPoolCorpus,
    *,
    random_seed: int,
    n_folds: int,
    bootstrap_n: int,
    training_epochs: int,
    hidden_dim: int,
    lr: float,
) -> SeedHardeningResult:
    folds = exp4244.split_task_folds(corpus.rows, random_seed=random_seed, n_folds=n_folds)
    report = exp4244.train_oof_set_encoder(
        corpus.rows,
        folds=folds,
        random_seed=random_seed,
        bootstrap_n=bootstrap_n,
        hidden_dim=hidden_dim,
        training_epochs=training_epochs,
        lr=lr,
    )
    metrics = _measure_seed_oof(corpus.rows, report.rows)
    return SeedHardeningResult(
        random_seed=int(random_seed),
        auroc=_round_metric(report.auroc),
        delta=_round_metric(metrics["delta"]),
        held_out_task_n=int(metrics["held_out_task_n"]),
        vote_at_1=_round_metric(metrics["vote_at_1"]),
        set_encoder_at_1=_round_metric(metrics["set_encoder_at_1"]),
        oracle_at_k=_round_metric(metrics["oracle_at_k"]),
        fold_task_ids=report.fold_task_ids,
    )


def _strings_from(value: Any) -> set[str]:
    strings: set[str] = set()
    if isinstance(value, str) and value:
        strings.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            strings.update(_strings_from(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            strings.update(_strings_from(item))
    return strings


def _task_surrogates(pool_payload: dict[str, Any]) -> dict[str, dict[str, set[str]]]:
    surrogates: dict[str, dict[str, set[str]]] = {}
    for task in pool_payload.get("tasks", []):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "")
        if not task_id:
            continue
        raw_task_id = str(task.get("raw_task_id") or "")
        task_tokens = {task_id}
        if raw_task_id:
            task_tokens.add(raw_task_id)
        candidate_tokens: set[str] = set()
        gold_tokens: set[str] = set()
        candidates = task.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("candidate_id") or "")
            grid_hash = str(candidate.get("candidate_grid_hash") or "")
            grid = candidate.get("grid")
            grid_text = (
                json.dumps(grid, sort_keys=True, separators=(",", ":")) if grid is not None else ""
            )
            for token in (candidate_id, grid_hash, grid_text):
                if token:
                    candidate_tokens.add(token)
                    if candidate.get("is_correct") is True:
                        gold_tokens.add(token)
        surrogates[task_id] = {
            "task": task_tokens,
            "candidate": candidate_tokens,
            "gold": gold_tokens,
        }
    return surrogates


def _row_training_signal(row: dict[str, Any]) -> set[str]:
    signal: set[str] = set()
    for key in (
        "train_task_ids",
        "train_candidate_ids",
        "train_candidate_grid_hashes",
        "train_gold_grid_hashes",
        "training_signal",
    ):
        signal.update(_strings_from(row.get(key)))
    return signal


def row_level_leak_audit(
    pool_payload: dict[str, Any], model_artifact: dict[str, Any]
) -> dict[str, Any]:
    """SCENARIO-VERIFY-5151: audit OOF training-signal rows for held-out overlap."""

    surrogates = _task_surrogates(pool_payload)
    rows = model_artifact.get("set_encoder_oof", {}).get("rows", [])
    if not isinstance(rows, list):
        rows = []
    task_collisions: list[dict[str, str]] = []
    candidate_collisions: list[dict[str, str]] = []
    gold_collisions: list[dict[str, str]] = []
    excluded_count = 0
    scored_count = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("task_id") or "")
        candidate_id = str(row.get("candidate_id") or "")
        if task_id not in surrogates:
            continue
        scored_count += 1
        signal = _row_training_signal(row)
        train_task_ids = {
            str(item) for item in row.get("train_task_ids", []) if isinstance(item, str)
        }
        if task_id not in train_task_ids:
            excluded_count += 1
        for token in sorted(surrogates[task_id]["task"] & signal):
            task_collisions.append(
                {"task_id": task_id, "candidate_id": candidate_id, "token": token}
            )
        for token in sorted(surrogates[task_id]["candidate"] & signal):
            candidate_collisions.append(
                {"task_id": task_id, "candidate_id": candidate_id, "token": token}
            )
        for token in sorted(surrogates[task_id]["gold"] & signal):
            gold_collisions.append(
                {"task_id": task_id, "candidate_id": candidate_id, "token": token}
            )
    passed = (
        scored_count > 0
        and excluded_count == scored_count
        and not task_collisions
        and not candidate_collisions
        and not gold_collisions
    )
    return {
        "passed": passed,
        "scored_oof_row_count": scored_count,
        "task_excluded_row_count": excluded_count,
        "held_out_task_n": len(surrogates),
        "task_id_collision_count": len(task_collisions),
        "candidate_signal_collision_count": len(candidate_collisions),
        "gold_signal_collision_count": len(gold_collisions),
        "task_id_collisions_sample": task_collisions[:10],
        "candidate_signal_collisions_sample": candidate_collisions[:10],
        "gold_signal_collisions_sample": gold_collisions[:10],
        "training_signal_fields_audited": [
            "train_task_ids",
            "train_candidate_ids",
            "train_candidate_grid_hashes",
            "train_gold_grid_hashes",
            "training_signal",
        ],
    }


def _task_deltas_from_exp4245(exp4245_artifact: dict[str, Any]) -> tuple[list[float], int, int]:
    deltas: list[float] = []
    wins = 0
    losses = 0
    rows = exp4245_artifact.get("task_rows", [])
    if not isinstance(rows, list):
        return deltas, wins, losses
    for row in rows:
        if not isinstance(row, dict):
            continue
        vote = row.get("vote_correct") is True
        set_encoder = row.get("set_encoder_correct") is True
        delta = float(set_encoder) - float(vote)
        deltas.append(delta)
        if set_encoder and not vote:
            wins += 1
        elif vote and not set_encoder:
            losses += 1
    return deltas, wins, losses


def _run_cross_game_check(
    repo_root: Path,
    *,
    random_seed: int,
    n_folds: int,
    bootstrap_resamples: int,
    training_epochs: int,
    hidden_dim: int,
    lr: float,
) -> dict[str, Any]:
    try:
        exp4258._load_required_artifacts(repo_root)
        corpus = exp4258.load_game_annotated_corpus(repo_root)
        folds = exp4258.build_game_disjoint_folds(
            corpus.task_game_ids, random_seed=random_seed, n_folds=n_folds
        )
        report = exp4258.train_cross_game_oof(
            corpus,
            folds,
            random_seed=random_seed,
            training_epochs=training_epochs,
            hidden_dim=hidden_dim,
            lr=lr,
        )
        metrics = exp4258.measure_cross_game_gate(
            corpus,
            report.rows,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
    except exp4258.BlockedRun as exc:
        reason = str(exc) or BLOCKED_GAME_IDS_VERDICT
        return {
            "cross_game_replication_delta": None,
            "cross_game_replication_ci95": None,
            "cross_game_blocked_reason": reason,
            "cross_game_honest_read": "blocked",
            "held_out_game_n": 0,
        }
    return {
        "cross_game_replication_delta": metrics["cross_game_delta"],
        "cross_game_replication_ci95": metrics["cross_game_ci95"],
        "cross_game_blocked_reason": None,
        "cross_game_honest_read": metrics["honest_read"],
        "held_out_game_n": metrics["held_out_game_n"],
    }


def _hardening_axes(
    *,
    multiseed_passed: bool,
    leak_audit_passed: bool,
    exact_test_passed: bool,
    cross_game: dict[str, Any],
) -> dict[str, str]:
    if cross_game["cross_game_blocked_reason"] is not None:
        cross_status = "blocked"
    elif _safe_float(cross_game["cross_game_replication_delta"]) > 0.0 and _ci_excludes_zero(
        cross_game.get("cross_game_replication_ci95")
    ):
        cross_status = "passed"
    else:
        cross_status = "failed"
    return {
        "multiseed": "passed" if multiseed_passed else "failed",
        "leak_audit": "passed" if leak_audit_passed else "failed",
        "exact_test": "passed" if exact_test_passed else "failed",
        "cross_game": cross_status,
    }


def _model_specs(
    *,
    training_epochs: int,
    hidden_dim: int,
    n_folds: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "protocol": "exp4245_hardening_multiseed_leak_audit_exact_test_cross_game_check",
        "set_encoder_architecture": "deepsets_pooled_context_set_encoder",
        "fixed_candidate_pool": str(POOL_REL),
        "random_seeds": list(DEFAULT_RANDOM_SEEDS),
        "n_folds": int(n_folds),
        "training_epochs": int(training_epochs),
        "hidden_dim": int(hidden_dim),
        "bootstrap_resamples": int(bootstrap_resamples),
        "oracle_distinct_inference": True,
        "score_inputs": "candidate features and cross-candidate set context only; no gold labels at inference",
    }


def reproducibility_checksum(
    *,
    pool_sha256: str,
    model_sha256: str,
    seed_results: list[SeedHardeningResult],
    leak_audit: dict[str, Any],
    cluster_ci95: list[float],
    exact_p: float,
    cross_game: dict[str, Any],
    random_seeds: list[int],
) -> str:
    payload = {
        "cluster_bootstrap_delta_ci95": cluster_ci95,
        "cross_game": cross_game,
        "exact_test_p_value": exact_p,
        "leak_audit": leak_audit,
        "model_sha256": model_sha256,
        "pool_sha256": pool_sha256,
        "random_seeds_used": random_seeds,
        "seed_results": [
            {
                "auroc": result.auroc,
                "delta": result.delta,
                "fold_task_ids": result.fold_task_ids,
                "random_seed": result.random_seed,
            }
            for result in seed_results
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_checksum(reason: str, random_seed: int) -> str:
    raw = json.dumps({"random_seed": int(random_seed), "reason": reason}, sort_keys=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_artifact(
    reason: str,
    *,
    random_seed: int,
    checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_5151_arc_oracle_distinct_hardening_v472",
        "schema": "carnot.arc_oracle_distinct_hardening_5151.v1",
        "status": "complete",
        "headline_outcome": "arc_set_encoder_hardening_blocked",
        "honest_verdict": reason,
        "multiseed_delta_ci95": [0.0, 0.0],
        "leak_audit_passed": False,
        "cross_game_replication_delta": None,
        "cross_game_replication_ci95": None,
        "verifier_is_oracle": False,
        "solve_provenance": SOLVE_PROVENANCE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": int(random_seed),
        "random_seeds_used": [],
        "per_seed_results": [],
        "mean_delta": 0.0,
        "cluster_bootstrap_delta_ci95": [0.0, 0.0],
        "exact_test_discordant_wins": 0,
        "exact_test_discordant_losses": 0,
        "exact_test_p_value": 1.0,
        "exact_test_passes_min6_rule": False,
        "cross_game_blocked_reason": None,
        "leak_audit": {"passed": False, "blocked_reason": reason},
        "hardening_axes": {
            "multiseed": "blocked",
            "leak_audit": "blocked",
            "exact_test": "blocked",
            "cross_game": "blocked",
        },
        "acceptance_gate": False,
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "model_specs": {"status": "blocked", "blocked_reason": reason},
        "duration_s": round(duration_s, 6),
        "adversarial_verify": {"status": "pending"},
    }


def _complete_artifact(
    *,
    exp4245_artifact: dict[str, Any],
    seed_results: list[SeedHardeningResult],
    leak_audit: dict[str, Any],
    task_deltas: list[float],
    wins: int,
    losses: int,
    cross_game: dict[str, Any],
    checksum: str,
    random_seed: int,
    random_seeds: list[int],
    bootstrap_resamples: int,
    training_epochs: int,
    hidden_dim: int,
    n_folds: int,
    duration_s: float,
) -> dict[str, Any]:
    per_seed_deltas = [_round_metric(result.delta) for result in seed_results]
    multiseed_ci95 = _multiseed_ci95(per_seed_deltas)
    mean_delta = _round_metric(_mean(per_seed_deltas))
    cluster_ci95 = _cluster_bootstrap_ci95(
        task_deltas,
        random_seed=random_seed,
        resamples=bootstrap_resamples,
    )
    exact_p = _two_sided_binomial_p(wins, losses)
    exact_pass = bool(wins >= 6 and losses == 0 and exact_p < 0.05)
    multiseed_pass = bool(mean_delta > 0.0 and _ci_excludes_zero(multiseed_ci95))
    axes = _hardening_axes(
        multiseed_passed=multiseed_pass,
        leak_audit_passed=leak_audit["passed"],
        exact_test_passed=exact_pass,
        cross_game=cross_game,
    )
    survives = all(status == "passed" for status in axes.values())
    failed_axes = [axis for axis, status in axes.items() if status != "passed"]
    if survives:
        verdict = (
            "success_arc_set_encoder_win_survives_hardening: +44pp win survives "
            "multiseed, leak-audit, exact-test, and cross-game checks"
        )
        headline = "arc_set_encoder_win_survives_hardening"
    else:
        verdict = (
            "complete_arc_set_encoder_win_not_hardened: +44pp win does not fully "
            f"survive hardening; unresolved_axes={','.join(failed_axes)}"
        )
        headline = "arc_set_encoder_win_not_hardened"
    return {
        "experiment": "experiment_5151_arc_oracle_distinct_hardening_v472",
        "schema": "carnot.arc_oracle_distinct_hardening_5151.v1",
        "status": "complete",
        "headline_outcome": headline,
        "honest_verdict": verdict,
        "original_exp4245_delta": _round_metric(
            _safe_float(exp4245_artifact.get("set_encoder_minus_vote_delta"))
        ),
        "original_exp4245_ci95": exp4245_artifact.get("set_encoder_minus_vote_ci95", [0.0, 0.0]),
        "multiseed_delta_ci95": multiseed_ci95,
        "multiseed_ci_method": "student_t_mean_ci_over_seed_deltas",
        "leak_audit_passed": bool(leak_audit["passed"]),
        "cross_game_replication_delta": cross_game["cross_game_replication_delta"],
        "cross_game_replication_ci95": cross_game["cross_game_replication_ci95"],
        "cross_game_blocked_reason": cross_game["cross_game_blocked_reason"],
        "cross_game_honest_read": cross_game["cross_game_honest_read"],
        "held_out_game_n": int(cross_game["held_out_game_n"]),
        "verifier_is_oracle": False,
        "solve_provenance": SOLVE_PROVENANCE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seed": int(random_seed),
        "random_seeds_used": list(random_seeds),
        "n_seeds": len(seed_results),
        "per_seed_results": [
            {
                "random_seed": result.random_seed,
                "auroc": _round_metric(result.auroc),
                "delta": _round_metric(result.delta),
                "held_out_task_n": result.held_out_task_n,
                "vote_at_1": _round_metric(result.vote_at_1),
                "set_encoder_at_1": _round_metric(result.set_encoder_at_1),
                "oracle_at_k": _round_metric(result.oracle_at_k),
            }
            for result in seed_results
        ],
        "per_seed_deltas": per_seed_deltas,
        "mean_delta": mean_delta,
        "sign_flip_seeds": [result.random_seed for result in seed_results if result.delta <= 0.0],
        "cluster_bootstrap_delta_ci95": cluster_ci95,
        "cluster_bootstrap_resamples": int(bootstrap_resamples),
        "exact_test_discordant_wins": int(wins),
        "exact_test_discordant_losses": int(losses),
        "exact_test_p_value": _round_metric(exact_p),
        "exact_test_passes_min6_rule": exact_pass,
        "leak_audit": leak_audit,
        "hardening_axes": axes,
        "acceptance_gate": True,
        "reproducibility_checksum": checksum,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "model_specs": _model_specs(
            training_epochs=training_epochs,
            hidden_dim=hidden_dim,
            n_folds=n_folds,
            bootstrap_resamples=bootstrap_resamples,
        ),
        "duration_s": round(duration_s, 6),
        "adversarial_verify": {"status": "pending"},
    }


def _run_adversarial_verify(
    repo_root: Path, artifact_path: Path
) -> dict[str, Any]:  # pragma: no cover
    proc = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "adversarial_verify.py"),
            "--json",
            str(artifact_path),
        ],
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
        raise ValueError(f"{field} must be a bare float or null")


def _validate_ci95(value: Any, field: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise ValueError(f"{field} must be a two-number ci95")


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(
        ("complete:", "complete_", "success:", "success_", "blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    _validate_ci95(artifact["multiseed_delta_ci95"], "multiseed_delta_ci95")
    _validate_ci95(artifact["cluster_bootstrap_delta_ci95"], "cluster_bootstrap_delta_ci95")
    if type(artifact["leak_audit_passed"]) is not bool:
        raise ValueError("leak_audit_passed must be a bare bool")
    _nullable_float(artifact["cross_game_replication_delta"], "cross_game_replication_delta")
    if artifact["cross_game_replication_ci95"] is not None:
        _validate_ci95(artifact["cross_game_replication_ci95"], "cross_game_replication_ci95")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if artifact["solve_provenance"] != SOLVE_PROVENANCE:
        raise ValueError("solve_provenance must be development_proxy")
    if not isinstance(artifact["inference_substrate"], str) or not artifact["inference_substrate"]:
        raise ValueError("inference_substrate must be a non-empty string")
    if not isinstance(artifact["random_seeds_used"], list) or any(
        type(value) is not int for value in artifact["random_seeds_used"]
    ):
        raise ValueError("random_seeds_used must be a list of bare ints")
    if not isinstance(artifact["per_seed_results"], list):
        raise ValueError("per_seed_results must be a list")
    for field in ("exact_test_discordant_wins", "exact_test_discordant_losses"):
        if type(artifact[field]) is not int:
            raise ValueError(f"{field} must be a bare int")
    if isinstance(artifact["exact_test_p_value"], bool) or not isinstance(
        artifact["exact_test_p_value"], (int, float)
    ):
        raise ValueError("exact_test_p_value must be a bare float")
    if artifact["cross_game_blocked_reason"] is not None and not isinstance(
        artifact["cross_game_blocked_reason"], str
    ):
        raise ValueError("cross_game_blocked_reason must be a string or null")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-5151")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-5151")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    random_seeds: list[int] | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_n: int = exp4244.BOOTSTRAP_N,
    training_epochs: int = exp4244.DEFAULT_TRAINING_EPOCHS,
    hidden_dim: int = exp4244.DEFAULT_HIDDEN_DIM,
    lr: float = exp4244.DEFAULT_LR,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    seeds = list(random_seeds or DEFAULT_RANDOM_SEEDS)
    try:
        if len(seeds) < 5:
            raise BlockedRun("blocked_arc_hardening_requires_at_least_5_seeds")
        exp4245_artifact, pool_payload, model_artifact, model_path = _load_required_inputs(root)
        corpus = exp4244.load_grown_pool(root)
        pool_sha256 = _sha256_file(root / POOL_REL)
        model_sha256 = _sha256_file(model_path)
        seed_results = [
            _train_seed_hardening(
                corpus,
                random_seed=seed,
                n_folds=n_folds,
                bootstrap_n=bootstrap_n,
                training_epochs=training_epochs,
                hidden_dim=hidden_dim,
                lr=lr,
            )
            for seed in seeds
        ]
        leak_audit = row_level_leak_audit(pool_payload, model_artifact)
        task_deltas, wins, losses = _task_deltas_from_exp4245(exp4245_artifact)
        cluster_ci95 = _cluster_bootstrap_ci95(
            task_deltas,
            random_seed=random_seed,
            resamples=bootstrap_resamples,
        )
        exact_p = _two_sided_binomial_p(wins, losses)
        cross_game = _run_cross_game_check(
            root,
            random_seed=random_seed,
            n_folds=n_folds,
            bootstrap_resamples=bootstrap_resamples,
            training_epochs=training_epochs,
            hidden_dim=hidden_dim,
            lr=lr,
        )
        checksum = reproducibility_checksum(
            pool_sha256=pool_sha256,
            model_sha256=model_sha256,
            seed_results=seed_results,
            leak_audit=leak_audit,
            cluster_ci95=cluster_ci95,
            exact_p=exact_p,
            cross_game=cross_game,
            random_seeds=seeds,
        )
        artifact = _complete_artifact(
            exp4245_artifact=exp4245_artifact,
            seed_results=seed_results,
            leak_audit=leak_audit,
            task_deltas=task_deltas,
            wins=wins,
            losses=losses,
            cross_game=cross_game,
            checksum=checksum,
            random_seed=random_seed,
            random_seeds=seeds,
            bootstrap_resamples=bootstrap_resamples,
            training_epochs=training_epochs,
            hidden_dim=hidden_dim,
            n_folds=n_folds,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as exc:
        reason = str(exc) or BLOCKED_UPSTREAM_VERDICT
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
