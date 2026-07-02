"""Exp 5160 oracle-distinct cross-corpus closure.

Spec refs: REQ-VERIFY-5160, SCENARIO-VERIFY-5160,
SCENARIO-VERIFY-5160-UPSTREAM-BLOCKED.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244


RANDOM_SEED = 5160
DEFAULT_RANDOM_SEEDS = [5160, 5161, 5162, 5163, 5164]
DEFAULT_BOOTSTRAP_N = 0
DEFAULT_N_FOLDS = exp4244.DEFAULT_N_FOLDS
DEFAULT_TRAINING_EPOCHS = exp4244.DEFAULT_TRAINING_EPOCHS
DEFAULT_HIDDEN_DIM = exp4244.DEFAULT_HIDDEN_DIM
DEFAULT_LR = exp4244.DEFAULT_LR
EXP5151_REL = Path("results/experiment_5151_arc_oracle_distinct_hardening_v472.json")
ORIGINAL_POOL_REL = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
PREFERRED_SECOND_POOL_REL = Path("results/arc3_gap4_arc2_eval_pool.json.gz")
FALLBACK_SECOND_POOL_REL = Path("results/experiment_4291_arcgen_cross_generator_pool.json.gz")
OUTPUT_REL = Path("results/experiment_5160_oracle_distinct_cross_corpus_closure_v473.json")
INFERENCE_SUBSTRATE = "cached_disjoint_candidate_pool_cpu_multiseed_set_encoder_cross_corpus"
SOLVE_PROVENANCE = "development_proxy"
BLOCKED_UPSTREAM_VERDICT = "blocked_upstream_artifact_missing"
BLOCKED_SECOND_POOL_VERDICT = "complete_cross_corpus_replication_blocked_no_disjoint_second_pool"
SCHEMA = "carnot.oracle_distinct_cross_corpus_closure_5160.v1"
SPEC_REFS = [
    "REQ-VERIFY-5160",
    "SCENARIO-VERIFY-5160",
    "SCENARIO-VERIFY-5160-UPSTREAM-BLOCKED",
]
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
RECOMMENDATIONS = {"keep_gated", "ungate_now"}

FIELD_PRINCIPLES = {
    "game_id_misnomer_confirmed": (
        "The precise, code-verified fact this task exists to establish before choosing a fix path."
    ),
    "second_pool_source": (
        "The disjoint second-pool path actually used after disqualifying overlapping candidates."
    ),
    "second_pool_leak_audit_passed": (
        "A second-pool win is not evidence if task ids, candidate content, or gold-label "
        "surrogates overlap the original pool."
    ),
    "cross_corpus_delta": (
        "set_encoder@1 - vote@1 on the second corpus -- the corrected transfer axis after "
        "the game-id misnomer is verified."
    ),
    "cross_corpus_delta_ci95": (
        "The decisive number: does the +44pp win replicate off the original pool, CI95 excluding 0?"
    ),
    "cross_corpus_replication_passed": (
        "BARE bool: true iff the corrected cross-corpus delta is positive, CI95 excludes zero, "
        "and the leak audit passes."
    ),
    "diffusiongemma_gate_updated_recommendation": (
        "Feeds directly into whether DiffusionGemma scaling is queued next milestone."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- learned set-encoder, no demo execution; keeps the closure oracle-distinct."
    ),
    "solve_provenance": (
        "This is offline pool-scoring on cached candidate pools, not a live hidden-game solve."
    ),
    "random_seeds_used": "The >=5 seeds used for the corrected replication protocol.",
    "reproducibility_checksum": (
        "Hash of the schema inspection, selected-pool overlap audit, per-seed results, and recommendation."
    ),
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ AND state plainly whether the "
        "win survives cross-corpus replication."
    ),
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "honest_verdict",
    "game_id_misnomer_confirmed",
    "second_pool_source",
    "second_pool_leak_audit_passed",
    "cross_corpus_delta",
    "cross_corpus_delta_ci95",
    "cross_corpus_replication_passed",
    "diffusiongemma_gate_updated_recommendation",
    "verifier_is_oracle",
    "solve_provenance",
    "random_seeds_used",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
    "acceptance_gate",
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""


@dataclass(frozen=True)
class SeedReplicationResult:
    random_seed: int
    auroc: float
    delta: float
    held_out_task_n: int
    candidate_count: int
    vote_at_1: float
    set_encoder_at_1: float
    oracle_at_k: float
    task_deltas: list[float]
    oof_rows: list[exp4244.OOFRow]


@dataclass(frozen=True)
class SecondPoolSelection:
    corpus: exp4244.GrownPoolCorpus
    source_rel: Path
    source_sha256: str
    source_kind: str
    classic_arc_static_puzzle_pool: bool
    preferred_audit: dict[str, Any]
    adapter: dict[str, Any]


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _safe_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


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


def _grid_hash(grid: Any) -> str:
    if grid is None:
        return ""
    raw = json.dumps(grid, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalized_tasks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = payload.get("tasks")
    if isinstance(tasks, list):
        return [task for task in tasks if isinstance(task, dict)]
    entries = payload.get("entries")
    if isinstance(entries, list):
        normalized = []
        source_id = str(payload.get("experiment") or "entries_pool")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            raw_task_id = str(entry.get("task") or "")
            candidates = []
            for candidate_index, candidate in enumerate(entry.get("candidates") or []):
                if not isinstance(candidate, dict):
                    continue
                candidates.append(
                    {
                        "candidate_id": f"{source_id}:{raw_task_id}:{index}::candidate{candidate_index}",
                        "candidate_index": candidate_index,
                        "features": {},
                        "grid": candidate.get("grid"),
                        "is_correct": candidate.get("correct") is True,
                        "votes": candidate.get("votes", 0.0),
                    }
                )
            normalized.append(
                {
                    "task_id": f"{source_id}:{raw_task_id}:{index}",
                    "raw_task_id": raw_task_id,
                    "source_id": source_id,
                    "candidates": candidates,
                }
            )
        return normalized
    return []


def _pool_signature(payload: dict[str, Any]) -> dict[str, set[str]]:
    signature = {
        "task_ids": set(),
        "raw_task_ids": set(),
        "candidate_ids": set(),
        "candidate_grid_hashes": set(),
        "gold_grid_hashes": set(),
    }
    for task in _normalized_tasks(payload):
        task_id = str(task.get("task_id") or "")
        raw_task_id = str(task.get("raw_task_id") or task_id)
        if task_id:
            signature["task_ids"].add(task_id)
        if raw_task_id:
            signature["raw_task_ids"].add(raw_task_id)
        for candidate in task.get("candidates") or []:
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("candidate_id") or "")
            grid_hash = str(candidate.get("candidate_grid_hash") or _grid_hash(candidate.get("grid")))
            if candidate_id:
                signature["candidate_ids"].add(candidate_id)
            if grid_hash:
                signature["candidate_grid_hashes"].add(grid_hash)
                if candidate.get("is_correct") is True:
                    signature["gold_grid_hashes"].add(grid_hash)
    return signature


def _overlap_count(left: set[str], right: set[str]) -> tuple[int, list[str]]:
    sample = sorted(left & right)[:10]
    return len(left & right), sample


def audit_pool_overlap(original_payload: dict[str, Any], second_payload: dict[str, Any]) -> dict[str, Any]:
    """SCENARIO-VERIFY-5160: audit row-level overlap before using a second pool."""

    original = _pool_signature(original_payload)
    second = _pool_signature(second_payload)
    task_count, task_sample = _overlap_count(original["task_ids"], second["task_ids"])
    raw_count, raw_sample = _overlap_count(original["raw_task_ids"], second["raw_task_ids"])
    candidate_count, candidate_sample = _overlap_count(
        original["candidate_ids"], second["candidate_ids"]
    )
    grid_count, grid_sample = _overlap_count(
        original["candidate_grid_hashes"], second["candidate_grid_hashes"]
    )
    gold_count, gold_sample = _overlap_count(
        original["gold_grid_hashes"], second["gold_grid_hashes"]
    )
    disjoint = not any((task_count, raw_count, candidate_count, grid_count, gold_count))
    return {
        "disjoint": disjoint,
        "task_id_collision_count": task_count,
        "task_id_collisions_sample": task_sample,
        "raw_task_id_collision_count": raw_count,
        "raw_task_id_collisions_sample": raw_sample,
        "candidate_id_collision_count": candidate_count,
        "candidate_id_collisions_sample": candidate_sample,
        "candidate_grid_hash_collision_count": grid_count,
        "candidate_grid_hash_collisions_sample": grid_sample,
        "gold_grid_hash_collision_count": gold_count,
        "gold_grid_hash_collisions_sample": gold_sample,
    }


def inspect_original_pool_schema(pool_payload: dict[str, Any]) -> dict[str, Any]:
    """REQ-VERIFY-5160: verify whether Exp 4243 has a real game grouping."""

    tasks = _normalized_tasks(pool_payload)
    key_union = sorted({key for task in tasks for key in task})
    game_fields = [
        field for field in ("arc_game_id", "game_id", "domain_id", "family_id", "level_id") if field in key_union
    ]
    grouping_fields = [
        field for field in ("source_id", "source_kind", "source_kinds") if field in key_union
    ]
    raw_task_ids = [str(task.get("raw_task_id") or "") for task in tasks]
    task_ids = [str(task.get("task_id") or "") for task in tasks]
    raw_hex = bool(raw_task_ids) and all(re.fullmatch(r"[0-9a-fA-F]{8}", item) for item in raw_task_ids)
    source_histogram = Counter(str(task.get("source_id") or "") for task in tasks if task.get("source_id"))
    return {
        "game_id_misnomer_confirmed": bool(raw_hex and not game_fields),
        "raw_task_id_format": "8_hex_static_arc_puzzle_id" if raw_hex else "other_or_mixed",
        "task_id_examples": task_ids[:10],
        "raw_task_id_examples": raw_task_ids[:10],
        "game_id_fields_present": game_fields,
        "grouping_fields_present": grouping_fields,
        "source_id_histogram": dict(sorted(source_histogram.items())),
        "task_schema_keys": key_union,
        "interpretation": (
            "static_arc_puzzle_pool_no_game_grouping"
            if raw_hex and not game_fields
            else "recoverable_grouping_or_nonstandard_schema"
        ),
    }


def _candidate_features(candidate: dict[str, Any]) -> dict[str, float]:
    raw = candidate.get("features")
    if not isinstance(raw, dict):
        raw = {}
    features = {name: _safe_float(raw.get(name)) for name in exp4244.FEATURE_NAMES}
    if features.get("vote_weight", 0.0) == 0.0:
        features["vote_weight"] = _safe_float(candidate.get("vote_weight", candidate.get("votes")))
    return features


def _build_second_corpus(
    repo_root: Path,
    *,
    original_payload: dict[str, Any],
    second_payload: dict[str, Any],
    source_rel: Path,
    source_kind: str,
    classic_arc_static_puzzle_pool: bool,
    preferred_audit: dict[str, Any],
) -> SecondPoolSelection:
    original = _pool_signature(original_payload)
    rows: list[exp4244.GrownPoolRow] = []
    dropped_overlap_task_n = 0
    dropped_overlap_candidate_n = 0
    dropped_too_small_task_n = 0
    selected_task_n = 0
    wrong_majority_n = 0
    positive_candidate_n = 0
    for task in _normalized_tasks(second_payload):
        task_id = str(task.get("task_id") or "")
        raw_task_id = str(task.get("raw_task_id") or task_id)
        if task_id in original["task_ids"] or raw_task_id in original["raw_task_ids"]:
            dropped_overlap_task_n += 1
            continue
        task_rows: list[exp4244.GrownPoolRow] = []
        for fallback_index, candidate in enumerate(task.get("candidates") or []):
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("candidate_id") or f"{task_id}::candidate{fallback_index}")
            grid_hash = str(candidate.get("candidate_grid_hash") or _grid_hash(candidate.get("grid")))
            is_correct = candidate.get("is_correct") is True
            if (
                candidate_id in original["candidate_ids"]
                or grid_hash in original["candidate_grid_hashes"]
                or (is_correct and grid_hash in original["gold_grid_hashes"])
            ):
                dropped_overlap_candidate_n += 1
                continue
            features = _candidate_features(candidate)
            task_rows.append(
                exp4244.GrownPoolRow(
                    task_id=task_id,
                    candidate_id=candidate_id,
                    candidate_index=int(candidate.get("candidate_index", fallback_index)),
                    correct=is_correct,
                    features=features,
                    vote_weight=features["vote_weight"],
                )
            )
        if len(task_rows) < 2:
            dropped_too_small_task_n += 1
            continue
        selected_task_n += 1
        positive_candidate_n += sum(1 for row in task_rows if row.correct)
        vote_pick = max(task_rows, key=lambda row: (row.vote_weight, -row.candidate_index))
        if any(row.correct for row in task_rows) and not vote_pick.correct:
            wrong_majority_n += 1
        rows.extend(sorted(task_rows, key=lambda row: row.candidate_index))
    if not rows:
        raise BlockedRun(BLOCKED_SECOND_POOL_VERDICT)
    pool_path = (repo_root / source_rel).resolve()
    corpus = exp4244.GrownPoolCorpus(
        rows=rows,
        pool_artifact_path=pool_path,
        pool_artifact_sha256=_sha256_file(pool_path),
        upstream_checksum=str(second_payload.get("reproducibility_checksum") or ""),
        held_out_task_n=selected_task_n,
        wrong_majority_n=wrong_majority_n,
        positive_candidate_n=positive_candidate_n,
    )
    return SecondPoolSelection(
        corpus=corpus,
        source_rel=source_rel,
        source_sha256=corpus.pool_artifact_sha256,
        source_kind=source_kind,
        classic_arc_static_puzzle_pool=classic_arc_static_puzzle_pool,
        preferred_audit=preferred_audit,
        adapter={
            "adapter": "exp4244_feature_schema_from_task_candidate_pool",
            "dropped_overlap_task_n": dropped_overlap_task_n,
            "dropped_overlap_candidate_n": dropped_overlap_candidate_n,
            "dropped_too_small_task_n": dropped_too_small_task_n,
            "selected_task_n": selected_task_n,
            "selected_candidate_n": len(rows),
            "wrong_majority_n": wrong_majority_n,
            "positive_candidate_n": positive_candidate_n,
            "source_kind": source_kind,
        },
    )


def select_second_pool(repo_root: Path, original_payload: dict[str, Any]) -> SecondPoolSelection:
    preferred_payload = _read_gzip_json_object(repo_root / PREFERRED_SECOND_POOL_REL)
    preferred_audit = audit_pool_overlap(original_payload, preferred_payload)
    if preferred_audit["disjoint"]:  # pragma: no cover - current checked-in GAP-4 pool overlaps.
        return _build_second_corpus(
            repo_root,
            original_payload=original_payload,
            second_payload=preferred_payload,
            source_rel=PREFERRED_SECOND_POOL_REL,
            source_kind="classic_arc2_preferred",
            classic_arc_static_puzzle_pool=True,
            preferred_audit=preferred_audit,
        )
    fallback_payload = _read_gzip_json_object(repo_root / FALLBACK_SECOND_POOL_REL)
    return _build_second_corpus(
        repo_root,
        original_payload=original_payload,
        second_payload=fallback_payload,
        source_rel=FALLBACK_SECOND_POOL_REL,
        source_kind="arcgen_non_degenerate_cross_generator",
        classic_arc_static_puzzle_pool=False,
        preferred_audit=preferred_audit,
    )


def _rows_by_task(rows: list[exp4244.GrownPoolRow]) -> list[list[exp4244.GrownPoolRow]]:
    grouped: dict[str, list[exp4244.GrownPoolRow]] = defaultdict(list)
    for row in rows:
        grouped[row.task_id].append(row)
    return [sorted(items, key=lambda row: row.candidate_index) for _, items in sorted(grouped.items())]


def _measure_oof(
    rows: list[exp4244.GrownPoolRow], oof_rows: list[exp4244.OOFRow]
) -> dict[str, Any]:
    score_map = {
        row.candidate_id: (float(row.score), row.task_id not in row.train_task_ids)
        for row in oof_rows
    }
    vote_hits: list[bool] = []
    set_hits: list[bool] = []
    oracle_hits: list[bool] = []
    task_deltas: list[float] = []
    retained_candidate_n = 0
    for task_rows in _rows_by_task(rows):
        scored = [(row, score_map.get(row.candidate_id)) for row in task_rows]
        if any(item is None or item[1] is not True for _, item in scored):
            continue
        retained_candidate_n += len(scored)
        vote_pick = max(task_rows, key=lambda row: (row.vote_weight, -row.candidate_index))
        set_pick = max(
            task_rows,
            key=lambda row: (score_map[row.candidate_id][0], row.vote_weight, -row.candidate_index),
        )
        vote_hits.append(vote_pick.correct)
        set_hits.append(set_pick.correct)
        oracle_hits.append(any(row.correct for row in task_rows))
        task_deltas.append(float(set_pick.correct) - float(vote_pick.correct))
    vote_at_1 = _mean([float(hit) for hit in vote_hits])
    set_at_1 = _mean([float(hit) for hit in set_hits])
    return {
        "delta": _round_metric(set_at_1 - vote_at_1),
        "held_out_task_n": len(vote_hits),
        "candidate_count": retained_candidate_n,
        "vote_at_1": _round_metric(vote_at_1),
        "set_encoder_at_1": _round_metric(set_at_1),
        "oracle_at_k": _round_metric(_mean([float(hit) for hit in oracle_hits])),
        "task_deltas": task_deltas,
    }


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
    metrics = _measure_oof(corpus.rows, report.rows)
    return SeedReplicationResult(
        random_seed=int(random_seed),
        auroc=_round_metric(report.auroc),
        delta=_round_metric(metrics["delta"]),
        held_out_task_n=int(metrics["held_out_task_n"]),
        candidate_count=int(metrics["candidate_count"]),
        vote_at_1=_round_metric(metrics["vote_at_1"]),
        set_encoder_at_1=_round_metric(metrics["set_encoder_at_1"]),
        oracle_at_k=_round_metric(metrics["oracle_at_k"]),
        task_deltas=[float(delta) for delta in metrics["task_deltas"]],
        oof_rows=list(report.rows),
    )


def _t_critical_975(df: int) -> float:
    return {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(max(1, int(df)), 1.96)


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


def _ci_excludes_zero(ci95: list[float]) -> bool:
    return bool(len(ci95) == 2 and (ci95[0] > 0.0 or ci95[1] < 0.0))


def _second_signature_from_rows(rows: list[exp4244.GrownPoolRow]) -> dict[str, set[str]]:
    return {
        "task_ids": {row.task_id for row in rows},
        "candidate_ids": {row.candidate_id for row in rows},
    }


def second_pool_leak_audit(
    original_payload: dict[str, Any],
    selection: SecondPoolSelection,
    seed_results: list[SeedReplicationResult],
) -> dict[str, Any]:
    original = _pool_signature(original_payload)
    second = _second_signature_from_rows(selection.corpus.rows)
    task_count, task_sample = _overlap_count(original["task_ids"], second["task_ids"])
    candidate_count, candidate_sample = _overlap_count(
        original["candidate_ids"], second["candidate_ids"]
    )
    filtered_grid_collisions = int(selection.adapter["dropped_overlap_candidate_n"])
    heldout_train_collisions = []
    oof_row_count = 0
    for result in seed_results:
        for row in result.oof_rows:
            oof_row_count += 1
            if row.task_id in row.train_task_ids:
                heldout_train_collisions.append(
                    {"random_seed": result.random_seed, "task_id": row.task_id, "candidate_id": row.candidate_id}
                )
    passed = not any((task_count, candidate_count, heldout_train_collisions))
    return {
        "passed": passed,
        "task_id_collision_count": task_count,
        "task_id_collisions_sample": task_sample,
        "candidate_id_collision_count": candidate_count,
        "candidate_id_collisions_sample": candidate_sample,
        "candidate_grid_hash_collision_count": 0,
        "gold_grid_hash_collision_count": 0,
        "heldout_training_task_collision_count": len(heldout_train_collisions),
        "heldout_training_task_collisions_sample": heldout_train_collisions[:10],
        "scored_oof_row_count": oof_row_count,
        "adapter_filtered_overlap_candidate_n": filtered_grid_collisions,
        "training_signal_fields_audited": ["train_task_ids", "candidate_ids", "candidate_grid_hashes", "gold_grid_hashes"],
    }


def reproducibility_checksum(artifact_without_checksum: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact_without_checksum.items()
        if key not in {"reproducibility_checksum", "adversarial_verify", "duration_s"}
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_artifact(reason: str, *, random_seed: int, duration_s: float) -> dict[str, Any]:
    artifact = {
        "schema": SCHEMA,
        "experiment": "experiment_5160_oracle_distinct_cross_corpus_closure_v473",
        "status": "blocked" if reason.startswith("blocked_") else "complete",
        "headline_outcome": reason,
        "honest_verdict": reason,
        "game_id_misnomer_confirmed": False,
        "second_pool_source": "",
        "second_pool_leak_audit_passed": False,
        "cross_corpus_delta": 0.0,
        "cross_corpus_delta_ci95": [0.0, 0.0],
        "cross_corpus_replication_passed": False,
        "diffusiongemma_gate_updated_recommendation": "keep_gated",
        "verifier_is_oracle": False,
        "solve_provenance": SOLVE_PROVENANCE,
        "random_seed": int(random_seed),
        "random_seeds_used": [],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": False,
        "schema_inspection": {},
        "preferred_second_pool_audit": {},
        "second_pool_adapter": {},
        "second_pool_leak_audit": {"passed": False, "blocked_reason": reason},
        "model_specs": {"status": "blocked", "blocked_reason": reason},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "adversarial_verify": {"status": "pending"},
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _complete_artifact(
    *,
    schema_inspection: dict[str, Any],
    selection: SecondPoolSelection,
    seed_results: list[SeedReplicationResult],
    leak_audit: dict[str, Any],
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    per_seed_deltas = [_round_metric(result.delta) for result in seed_results]
    cross_delta = _round_metric(_mean(per_seed_deltas))
    cross_ci95 = _multiseed_ci95(per_seed_deltas)
    passed = bool(cross_delta > 0.0 and _ci_excludes_zero(cross_ci95) and leak_audit["passed"])
    recommendation = "ungate_now" if passed else "keep_gated"
    headline = (
        "arc_set_encoder_win_survives_cross_corpus_replication"
        if passed
        else "arc_set_encoder_win_does_not_survive_cross_corpus_replication"
    )
    prefix = "success" if passed else "complete"
    artifact = {
        "schema": SCHEMA,
        "experiment": "experiment_5160_oracle_distinct_cross_corpus_closure_v473",
        "status": "complete",
        "headline_outcome": headline,
        "honest_verdict": (
            f"{prefix}_{headline}: set-encoder-vs-vote win "
            f"{'survives' if passed else 'does not survive'} corrected cross-corpus replication"
        ),
        "game_id_misnomer_confirmed": bool(schema_inspection["game_id_misnomer_confirmed"]),
        "second_pool_source": str(selection.source_rel),
        "second_pool_leak_audit_passed": bool(leak_audit["passed"]),
        "cross_corpus_delta": cross_delta,
        "cross_corpus_delta_ci95": cross_ci95,
        "cross_corpus_replication_passed": passed,
        "diffusiongemma_gate_updated_recommendation": recommendation,
        "verifier_is_oracle": False,
        "solve_provenance": SOLVE_PROVENANCE,
        "random_seed": int(random_seed),
        "random_seeds_used": [result.random_seed for result in seed_results],
        "n_seeds": len(seed_results),
        "per_seed_deltas": per_seed_deltas,
        "per_seed_results": [
            {
                "random_seed": result.random_seed,
                "auroc": _round_metric(result.auroc),
                "delta": _round_metric(result.delta),
                "held_out_task_n": result.held_out_task_n,
                "candidate_count": result.candidate_count,
                "vote_at_1": _round_metric(result.vote_at_1),
                "set_encoder_at_1": _round_metric(result.set_encoder_at_1),
                "oracle_at_k": _round_metric(result.oracle_at_k),
            }
            for result in seed_results
        ],
        "pass_rates": {
            "vote_at_1": _round_metric(_mean([result.vote_at_1 for result in seed_results])),
            "set_encoder_at_1": _round_metric(_mean([result.set_encoder_at_1 for result in seed_results])),
            "oracle_at_k": _round_metric(_mean([result.oracle_at_k for result in seed_results])),
        },
        "held_out_task_n": selection.corpus.held_out_task_n,
        "candidate_count": len(selection.corpus.rows),
        "preferred_second_pool_audit": selection.preferred_audit,
        "second_pool_adapter": selection.adapter,
        "second_pool_classic_arc_static_puzzle": selection.classic_arc_static_puzzle_pool,
        "second_pool_source_kind": selection.source_kind,
        "second_pool_sha256": selection.source_sha256,
        "second_pool_leak_audit": leak_audit,
        "schema_inspection": schema_inspection,
        "model_specs": {
            "status": "complete",
            "protocol": "exp5151_deepsets_pooled_context_set_encoder_on_disjoint_second_pool",
            "set_encoder_architecture": "deepsets_pooled_context_set_encoder",
            "n_folds": DEFAULT_N_FOLDS,
            "training_epochs": DEFAULT_TRAINING_EPOCHS,
            "hidden_dim": DEFAULT_HIDDEN_DIM,
            "random_seeds": [result.random_seed for result in seed_results],
            "preferred_second_pool": str(PREFERRED_SECOND_POOL_REL),
            "selected_second_pool": str(selection.source_rel),
            "selected_pool_note": (
                "Preferred GAP-4 ARC-2 pool overlaps Exp 4243; selected fallback is disjoint "
                "ARC-GEN non-degenerate cross-generator pool."
            ),
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 6),
        "adversarial_verify": {"status": "pending"},
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


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


def _validate_ci95(value: Any, field: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise ValueError(f"{field} must be a two-number ci95")


def _validate_float(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a bare float")


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")
    for field in (
        "game_id_misnomer_confirmed",
        "second_pool_leak_audit_passed",
        "cross_corpus_replication_passed",
    ):
        if type(artifact[field]) is not bool:
            raise ValueError(f"{field} must be a bare bool")
    _validate_float(artifact["cross_corpus_delta"], "cross_corpus_delta")
    _validate_ci95(artifact["cross_corpus_delta_ci95"], "cross_corpus_delta_ci95")
    if artifact["diffusiongemma_gate_updated_recommendation"] not in RECOMMENDATIONS:
        raise ValueError("diffusiongemma_gate_updated_recommendation has invalid recommendation")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if artifact["solve_provenance"] != SOLVE_PROVENANCE:
        raise ValueError("solve_provenance must be development_proxy")
    if not isinstance(artifact["random_seeds_used"], list) or any(
        type(seed) is not int for seed in artifact["random_seeds_used"]
    ):
        raise ValueError("random_seeds_used must be a list of bare ints")
    checksum = artifact["reproducibility_checksum"]
    if not isinstance(checksum, str) or not checksum.startswith("sha256:") or len(checksum) != 71:
        raise ValueError("reproducibility_checksum must be sha256-prefixed")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles drifted")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs drifted")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    random_seeds: list[int] | None = None,
    n_folds: int = DEFAULT_N_FOLDS,
    bootstrap_n: int = DEFAULT_BOOTSTRAP_N,
    training_epochs: int = DEFAULT_TRAINING_EPOCHS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    lr: float = DEFAULT_LR,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    seeds = list(random_seeds or DEFAULT_RANDOM_SEEDS)
    try:
        exp5151 = _read_json_object(root / EXP5151_REL)
        original_payload = _read_gzip_json_object(root / ORIGINAL_POOL_REL)
        if exp5151.get("verifier_is_oracle") is not False or not isinstance(original_payload.get("tasks"), list):
            raise BlockedRun(BLOCKED_UPSTREAM_VERDICT)
        if len(seeds) < 5:
            raise BlockedRun(BLOCKED_SECOND_POOL_VERDICT)
        schema_inspection = inspect_original_pool_schema(original_payload)
        selection = select_second_pool(root, original_payload)
        seed_results = [
            _train_seed_replication(
                selection.corpus,
                random_seed=seed,
                n_folds=n_folds,
                bootstrap_n=bootstrap_n,
                training_epochs=training_epochs,
                hidden_dim=hidden_dim,
                lr=lr,
            )
            for seed in seeds
        ]
        leak_audit = second_pool_leak_audit(original_payload, selection, seed_results)
        artifact = _complete_artifact(
            schema_inspection=schema_inspection,
            selection=selection,
            seed_results=seed_results,
            leak_audit=leak_audit,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as exc:
        reason = str(exc) or BLOCKED_UPSTREAM_VERDICT
        artifact = _blocked_artifact(reason, random_seed=random_seed, duration_s=time.perf_counter() - start)
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


def main() -> None:  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[3]
    print(json.dumps(run(repo_root), indent=2, sort_keys=True))
