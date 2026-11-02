"""Exp 4257 ARC oracle-distinct multi-seed replication.

Spec refs: REQ-VERIFY-4257, SCENARIO-VERIFY-4257.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


RANDOM_SEED = 4257
DEFAULT_REPLICATION_SEEDS = [4257, 4258, 4259, 4260, 4261]
DEFAULT_N_FOLDS = exp4244.DEFAULT_N_FOLDS
DEFAULT_BOOTSTRAP_N = exp4244.BOOTSTRAP_N
POOL_REL = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
SET_ENCODER_BUILD_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_build.json")
SET_ENCODER_MODEL_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_model.json")
SINGLE_SEED_WIN_REL = Path("results/experiment_4245_arc_set_encoder_beats_vote.json")
OUTPUT_REL = Path("results/experiment_4257_arc_oracle_distinct_multiseed_replication.json")
INFERENCE_SUBSTRATE = "cached_grown_arc_pool_cpu_multiseed_set_encoder"
BLOCKED_GROWN_POOL_VERDICT = "blocked_arc_grown_pool_missing"
BLOCKED_ARTIFACT_VERDICT = "blocked_arc_set_encoder_artifact_missing"
SPEC_REFS = ["REQ-VERIFY-4257", "SCENARIO-VERIFY-4257"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A clean multi-seed replication AND an honest fragility finding "
        "are BOTH COMPLETE and decision-grade."
    ),
    "oracle_distinct_win_replicates": (
        "BARE bool: A4/B1 gate on this raw value (gated-fields-must-be-bare); true iff "
        "mean cross-seed delta>0 AND CI excl 0 AND the independent re-score lands within "
        "the .393 CI."
    ),
    "per_seed_deltas": (
        "List of set_encoder@1 - vote@1 per seed -- exposes single-seed fragility; "
        "a sign-flip on any seed is a fragility flag."
    ),
    "mean_delta": (
        "BARE float: mean cross-seed delta -- the replicated lift (compare to the single-seed +0.4423)."
    ),
    "cross_seed_ci95": (
        "Cross-seed CI of the delta -- excluding 0 distinguishes a robust win from a single-seed fluke."
    ),
    "independent_rescore_delta": (
        "set_encoder@1 - vote@1 recomputed off the persisted artifact via a SECOND code path -- "
        "the independent-reproducer check (G2-style)."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- learned set-encoder, no demo execution; keeps the replication oracle-distinct."
    ),
    "random_seeds_used": (
        "The >=5 seeds -- determinism + power; lets a third party re-run every fold."
    ),
    "reproducibility_checksum": "Hash of the pool + per-seed splits; catches silent drift.",
    "model_specs": (
        "The set-encoder config + the multi-seed + independent-rescore protocol; required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "oracle_distinct_win_replicates",
    "per_seed_deltas",
    "mean_delta",
    "cross_seed_ci95",
    "independent_rescore_delta",
    "verifier_is_oracle",
    "random_seeds_used",
    "reproducibility_checksum",
    "model_specs",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""


@dataclass(frozen=True)
class CleanRoomCandidate:
    task_id: str
    candidate_id: str
    candidate_index: int
    vote_weight: float
    correct: bool


@dataclass(frozen=True)
class CleanRoomPool:
    candidates: list[CleanRoomCandidate]
    pool_artifact_path: Path
    pool_artifact_sha256: str
    upstream_checksum: str


@dataclass(frozen=True)
class SeedReplicationResult:
    random_seed: int
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


def _resolve_required_path(repo_root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise BlockedRun(BLOCKED_ARTIFACT_VERDICT)
    path = Path(value)
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.exists():
        raise BlockedRun(BLOCKED_ARTIFACT_VERDICT)
    return resolved


def _load_required_artifacts(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    pool_path = repo_root / POOL_REL
    if not pool_path.exists():
        raise BlockedRun(BLOCKED_GROWN_POOL_VERDICT)
    try:
        build = _read_json_object(repo_root / SET_ENCODER_BUILD_REL)
        model_path = _resolve_required_path(
            repo_root,
            build.get("learned_verifier_path") or str(SET_ENCODER_MODEL_REL),
        )
        model = _read_json_object(model_path)
        single_seed = _read_json_object(repo_root / SINGLE_SEED_WIN_REL)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BlockedRun(BLOCKED_ARTIFACT_VERDICT) from exc
    if build.get("aggregator_trained") is not True:
        raise BlockedRun(BLOCKED_ARTIFACT_VERDICT)
    if build.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_ARTIFACT_VERDICT)
    if model.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_ARTIFACT_VERDICT)
    if single_seed.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_ARTIFACT_VERDICT)
    return build, model, single_seed, model_path


def _load_clean_room_pool(repo_root: Path) -> CleanRoomPool:
    pool_path = repo_root / POOL_REL
    if not pool_path.exists():
        raise BlockedRun(BLOCKED_GROWN_POOL_VERDICT)
    try:
        with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise BlockedRun(BLOCKED_GROWN_POOL_VERDICT) from exc
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list):
        raise BlockedRun(BLOCKED_GROWN_POOL_VERDICT)
    candidates: list[CleanRoomCandidate] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "")
        raw_candidates = task.get("candidates")
        if not task_id or not isinstance(raw_candidates, list):
            continue
        for fallback_index, candidate in enumerate(raw_candidates):
            if not isinstance(candidate, dict):
                continue
            features = candidate.get("features")
            if not isinstance(features, dict):
                features = {}
            candidate_index = _safe_int(candidate.get("candidate_index", fallback_index))
            candidates.append(
                CleanRoomCandidate(
                    task_id=task_id,
                    candidate_id=str(candidate.get("candidate_id") or f"{task_id}::candidate{fallback_index}"),
                    candidate_index=candidate_index,
                    vote_weight=_safe_float(features.get("vote_weight", candidate.get("votes"))),
                    correct=candidate.get("is_correct") is True,
                )
            )
    return CleanRoomPool(
        candidates=candidates,
        pool_artifact_path=pool_path.resolve(),
        pool_artifact_sha256=_sha256_file(pool_path),
        upstream_checksum=str(payload.get("reproducibility_checksum") or ""),
    )


def _load_oof_scores(model: dict[str, Any]) -> dict[str, tuple[float, bool, int]]:
    scores: dict[str, tuple[float, bool, int]] = {}
    rows = model.get("set_encoder_oof", {}).get("rows", [])
    if not isinstance(rows, list):
        return scores
    for row in rows:
        if not isinstance(row, dict):
            continue
        candidate_id = row.get("candidate_id")
        task_id = row.get("task_id")
        if not isinstance(candidate_id, str) or not isinstance(task_id, str):
            continue
        train_task_ids = row.get("train_task_ids", [])
        excluded = isinstance(train_task_ids, list) and task_id not in train_task_ids
        scores[candidate_id] = (_safe_float(row.get("score")), excluded, _safe_int(row.get("fold")))
    return scores


def _group_clean_candidates(
    candidates: list[CleanRoomCandidate],
) -> list[list[CleanRoomCandidate]]:
    grouped: dict[str, list[CleanRoomCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.task_id].append(candidate)
    return [
        sorted(task_candidates, key=lambda item: item.candidate_index)
        for _, task_candidates in sorted(grouped.items())
    ]


def _task_metrics_from_scores(
    candidates: list[CleanRoomCandidate],
    score_map: dict[str, tuple[float, bool, int]],
) -> dict[str, Any]:
    set_hits: list[bool] = []
    vote_hits: list[bool] = []
    oracle_hits: list[bool] = []
    retained_candidates = 0
    task_rows: list[dict[str, Any]] = []
    for task_candidates in _group_clean_candidates(candidates):
        scored = []
        for candidate in task_candidates:
            score_item = score_map.get(candidate.candidate_id)
            if score_item is None or score_item[1] is not True:
                scored = []
                break
            scored.append((candidate, score_item[0]))
        if len(scored) != len(task_candidates):
            continue
        retained_candidates += len(scored)
        vote_pick = max(task_candidates, key=lambda item: (item.vote_weight, -item.candidate_index))
        set_pick, set_score = max(
            scored,
            key=lambda item: (item[1], item[0].vote_weight, -item[0].candidate_index),
        )
        oracle_hit = any(candidate.correct for candidate in task_candidates)
        vote_hits.append(vote_pick.correct)
        set_hits.append(set_pick.correct)
        oracle_hits.append(oracle_hit)
        task_rows.append(
            {
                "task_id": vote_pick.task_id,
                "oracle_hit": oracle_hit,
                "vote_candidate_id": vote_pick.candidate_id,
                "vote_correct": vote_pick.correct,
                "set_encoder_candidate_id": set_pick.candidate_id,
                "set_encoder_correct": set_pick.correct,
                "set_encoder_score": _round_metric(set_score),
            }
        )
    vote_rate = _mean([float(hit) for hit in vote_hits])
    set_rate = _mean([float(hit) for hit in set_hits])
    return {
        "independent_rescore_delta": _round_metric(set_rate - vote_rate),
        "pass_rates": {
            "vote_at_1": _round_metric(vote_rate),
            "set_encoder_at_1": _round_metric(set_rate),
        },
        "oracle_at_k": _round_metric(_mean([float(hit) for hit in oracle_hits])),
        "held_out_task_n": len(vote_hits),
        "candidate_count": retained_candidates,
        "task_rows": task_rows,
        "score_source": "clean_room_persisted_set_encoder_oof_rows",
    }


def independent_rescore_persisted_artifact(repo_root: Path | str = Path(".")) -> dict[str, Any]:
    """SCENARIO-VERIFY-4257: independently re-score persisted pool/model rows."""

    root = Path(repo_root)
    _build, model, _single_seed, _model_path = _load_required_artifacts(root)
    pool = _load_clean_room_pool(root)
    return {
        **_task_metrics_from_scores(pool.candidates, _load_oof_scores(model)),
        "candidate_pool_path": str(pool.pool_artifact_path),
        "candidate_pool_sha256": pool.pool_artifact_sha256,
        "pool_reproducibility_checksum": pool.upstream_checksum,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _measure_seed_oof(
    rows: list[exp4244.GrownPoolRow],
    oof_rows: list[exp4244.OOFRow],
) -> dict[str, Any]:
    candidates = [
        CleanRoomCandidate(
            task_id=row.task_id,
            candidate_id=row.candidate_id,
            candidate_index=row.candidate_index,
            vote_weight=row.vote_weight,
            correct=row.correct,
        )
        for row in rows
    ]
    score_map = {
        row.candidate_id: (float(row.score), row.task_id not in row.train_task_ids, int(row.fold))
        for row in oof_rows
    }
    return _task_metrics_from_scores(candidates, score_map)


def _train_seed_replication(
    corpus: exp4244.GrownPoolCorpus,
    *,
    random_seed: int,
    n_folds: int,
    bootstrap_n: int,
    training_epochs: int,
    hidden_dim: int,
    lr: float,
) -> SeedReplicationResult:
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
    return SeedReplicationResult(
        random_seed=int(random_seed),
        delta=_round_metric(metrics["independent_rescore_delta"]),
        held_out_task_n=int(metrics["held_out_task_n"]),
        vote_at_1=_round_metric(metrics["pass_rates"]["vote_at_1"]),
        set_encoder_at_1=_round_metric(metrics["pass_rates"]["set_encoder_at_1"]),
        oracle_at_k=_round_metric(metrics["oracle_at_k"]),
        fold_task_ids=report.fold_task_ids,
    )


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


def _cross_seed_ci95(deltas: list[float]) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    if len(deltas) == 1:
        point = _round_metric(deltas[0])
        return [point, point]
    mean = _mean(deltas)
    variance = sum((delta - mean) ** 2 for delta in deltas) / float(len(deltas) - 1)
    se = math.sqrt(variance) / math.sqrt(len(deltas))
    half_width = _t_critical_975(len(deltas) - 1) * se
    return [_round_metric(mean - half_width), _round_metric(mean + half_width)]


def _ci_excludes_zero(ci95: list[float]) -> bool:
    return bool(len(ci95) == 2 and (ci95[0] > 0.0 or ci95[1] < 0.0))


def reproducibility_checksum(
    *,
    corpus: exp4244.GrownPoolCorpus,
    seed_results: list[SeedReplicationResult],
    independent_report: dict[str, Any],
    random_seeds: list[int],
) -> str:
    payload = {
        "feature_names": list(exp4244.FEATURE_NAMES),
        "independent_rescore_delta": independent_report.get("independent_rescore_delta"),
        "pool_artifact_sha256": corpus.pool_artifact_sha256,
        "random_seeds_used": list(random_seeds),
        "seed_folds": [
            {
                "delta": result.delta,
                "fold_task_ids": result.fold_task_ids,
                "random_seed": result.random_seed,
            }
            for result in seed_results
        ],
        "upstream_checksum": corpus.upstream_checksum,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _model_specs(
    *,
    build_artifact: dict[str, Any],
    seed_results: list[SeedReplicationResult],
    independent_report: dict[str, Any],
    random_seeds: list[int],
    n_folds: int,
    training_epochs: int,
    hidden_dim: int,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "set_encoder_config": build_artifact.get("model_specs", {}),
        "multi_seed_protocol": {
            "random_seeds_used": list(random_seeds),
            "n_folds": int(n_folds),
            "training_epochs": int(training_epochs),
            "hidden_dim": int(hidden_dim),
            "task_split_unit": "task_id",
            "seed_count": len(seed_results),
            "out_of_fold": True,
        },
        "independent_rescore_protocol": {
            "path": "direct_gzip_json_pool_plus_persisted_oof_rows",
            "imports_exp4245_gate_code": False,
            "held_out_task_n": int(independent_report.get("held_out_task_n", 0)),
            "score_source": independent_report.get("score_source", ""),
        },
    }


def _blocked_checksum(reason: str, random_seed: int) -> str:
    raw = json.dumps({"random_seed": random_seed, "reason": reason}, sort_keys=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_artifact(
    reason: str,
    *,
    random_seed: int,
    checksum: str,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4257_arc_oracle_distinct_multiseed_replication",
        "schema": "carnot.arc_oracle_distinct_multiseed_replication_4257.v1",
        "status": "complete",
        "headline_outcome": "arc_oracle_distinct_multiseed_replication_blocked",
        "honest_verdict": reason,
        "oracle_distinct_win_replicates": False,
        "per_seed_deltas": [],
        "mean_delta": 0.0,
        "cross_seed_ci95": [0.0, 0.0],
        "independent_rescore_delta": 0.0,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "random_seeds_used": [],
        "reproducibility_checksum": checksum,
        "model_specs": {"status": "blocked", "blocked_reason": reason},
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_seeds": 0,
        "sign_flip_seeds": [],
        "fragility_flags": [],
        "cross_seed_ci95_excludes_zero": False,
        "independent_rescore_within_4245_ci": False,
        "single_seed_4245_ci95": [0.0, 0.0],
        "duration_s": round(duration_s, 6),
        "adversarial_verify": {"status": "pending"},
    }


def _complete_artifact(
    *,
    seed_results: list[SeedReplicationResult],
    independent_report: dict[str, Any],
    single_seed_artifact: dict[str, Any],
    model_specs: dict[str, Any],
    checksum: str,
    random_seed: int,
    random_seeds: list[int],
    duration_s: float,
) -> dict[str, Any]:
    deltas = [_round_metric(result.delta) for result in seed_results]
    mean_delta = _round_metric(_mean(deltas))
    ci95 = _cross_seed_ci95(deltas)
    single_seed_ci95 = [
        _round_metric(_safe_float(value))
        for value in single_seed_artifact.get("set_encoder_minus_vote_ci95", [0.0, 0.0])[:2]
    ]
    independent_delta = _round_metric(independent_report.get("independent_rescore_delta", 0.0))
    in_single_seed_ci = (
        len(single_seed_ci95) == 2 and single_seed_ci95[0] <= independent_delta <= single_seed_ci95[1]
    )
    sign_flip_seeds = [result.random_seed for result in seed_results if result.delta <= 0.0]
    ci_excludes_zero = _ci_excludes_zero(ci95)
    replicates = bool(mean_delta > 0.0 and ci_excludes_zero and in_single_seed_ci)
    honest_verdict = (
        "complete: arc_oracle_distinct_win_replicates_multiseed"
        if replicates
        else "complete: arc_oracle_distinct_win_fragile_multiseed"
    )
    return {
        "experiment": "experiment_4257_arc_oracle_distinct_multiseed_replication",
        "schema": "carnot.arc_oracle_distinct_multiseed_replication_4257.v1",
        "status": "complete",
        "headline_outcome": (
            "arc_oracle_distinct_win_replicates_multiseed"
            if replicates
            else "arc_oracle_distinct_win_fragile_multiseed"
        ),
        "honest_verdict": honest_verdict,
        "oracle_distinct_win_replicates": replicates,
        "per_seed_deltas": deltas,
        "mean_delta": mean_delta,
        "cross_seed_ci95": ci95,
        "independent_rescore_delta": independent_delta,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "random_seeds_used": list(random_seeds),
        "reproducibility_checksum": checksum,
        "model_specs": model_specs,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "n_seeds": len(seed_results),
        "per_seed_results": [
            {
                "random_seed": result.random_seed,
                "delta": _round_metric(result.delta),
                "held_out_task_n": result.held_out_task_n,
                "vote_at_1": _round_metric(result.vote_at_1),
                "set_encoder_at_1": _round_metric(result.set_encoder_at_1),
                "oracle_at_k": _round_metric(result.oracle_at_k),
            }
            for result in seed_results
        ],
        "sign_flip_seeds": sign_flip_seeds,
        "fragility_flags": [
            f"seed_{seed}_delta_non_positive"
            for seed in sign_flip_seeds
        ],
        "cross_seed_ci95_excludes_zero": ci_excludes_zero,
        "cross_seed_ci_method": "student_t_mean_ci_over_seed_deltas",
        "independent_rescore_within_4245_ci": in_single_seed_ci,
        "single_seed_4245_ci95": single_seed_ci95,
        "independent_rescore_pass_rates": independent_report.get("pass_rates", {}),
        "independent_rescore_oracle_at_k": independent_report.get("oracle_at_k", 0.0),
        "independent_rescore_held_out_task_n": independent_report.get("held_out_task_n", 0),
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
    if type(artifact["oracle_distinct_win_replicates"]) is not bool:
        raise ValueError("oracle_distinct_win_replicates must be a bare bool")
    if not isinstance(artifact["per_seed_deltas"], list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in artifact["per_seed_deltas"]
    ):
        raise ValueError("per_seed_deltas must be a list of numbers")
    for field in ("mean_delta", "independent_rescore_delta"):
        if isinstance(artifact[field], bool) or not isinstance(artifact[field], (int, float)):
            raise ValueError(f"{field} must be a bare float")
    ci95 = artifact["cross_seed_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ci95)
    ):
        raise ValueError("cross_seed_ci95 must be a two-number ci95")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if not isinstance(artifact["random_seeds_used"], list) or any(
        type(value) is not int for value in artifact["random_seeds_used"]
    ):
        raise ValueError("random_seeds_used must be a list of bare ints")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4257")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4257")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    random_seeds: list[int] | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    bootstrap_n: int = DEFAULT_BOOTSTRAP_N,
    training_epochs: int = exp4244.DEFAULT_TRAINING_EPOCHS,
    hidden_dim: int = exp4244.DEFAULT_HIDDEN_DIM,
    lr: float = exp4244.DEFAULT_LR,
    adversarial_runner: Any | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    seeds = list(random_seeds or DEFAULT_REPLICATION_SEEDS)
    try:
        if len(seeds) < 5:
            raise BlockedRun("blocked_arc_multiseed_requires_at_least_5_seeds")
        build_artifact, _model_artifact, single_seed_artifact, _model_path = _load_required_artifacts(root)
        corpus = exp4244.load_grown_pool(root)
        independent_report = independent_rescore_persisted_artifact(root)
        seed_results = [
            _train_seed_replication(
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
        model_specs = _model_specs(
            build_artifact=build_artifact,
            seed_results=seed_results,
            independent_report=independent_report,
            random_seeds=seeds,
            n_folds=n_folds,
            training_epochs=training_epochs,
            hidden_dim=hidden_dim,
        )
        checksum = reproducibility_checksum(
            corpus=corpus,
            seed_results=seed_results,
            independent_report=independent_report,
            random_seeds=seeds,
        )
        artifact = _complete_artifact(
            seed_results=seed_results,
            independent_report=independent_report,
            single_seed_artifact=single_seed_artifact,
            model_specs=model_specs,
            checksum=checksum,
            random_seed=random_seed,
            random_seeds=seeds,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as exc:
        reason = str(exc) or BLOCKED_GROWN_POOL_VERDICT
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
