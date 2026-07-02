"""Exp 5171 harden Exp 5160 cross-corpus Set-Encoder result to n>=30.

Spec refs: REQ-VERIFY-5171, SCENARIO-VERIFY-5171,
SCENARIO-VERIFY-5171-UPSTREAM-BLOCKED, SCENARIO-VERIFY-5171-INSUFFICIENT-POOL.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244
from carnot.reporting import arcgen_cross_generator_nondegenerate_4291 as exp4291
from carnot.reporting import oracle_distinct_cross_corpus_closure_5160 as exp5160


RANDOM_SEED = 5171
DEFAULT_RANDOM_SEEDS = [5171, 5172, 5173, 5174, 5175]
DEFAULT_GENERATOR_LIMIT = 10
DEFAULT_TASKS_PER_GENERATOR = 3
DEFAULT_CANDIDATES_PER_TASK = exp4291.DEFAULT_CANDIDATES_PER_TASK
DEFAULT_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_N_FOLDS = exp5160.DEFAULT_N_FOLDS
DEFAULT_TRAINING_EPOCHS = exp5160.DEFAULT_TRAINING_EPOCHS
DEFAULT_HIDDEN_DIM = exp5160.DEFAULT_HIDDEN_DIM
DEFAULT_LR = exp5160.DEFAULT_LR
MIN_HELD_OUT_TASK_N = 30
EXP5160_REL = Path("results/experiment_5160_oracle_distinct_cross_corpus_closure_v473.json")
EXP5160_SECOND_POOL_REL = Path("results/experiment_4291_arcgen_cross_generator_pool.json.gz")
ORIGINAL_POOL_REL = exp5160.ORIGINAL_POOL_REL
OUTPUT_REL = Path("results/experiment_5171_harden_set_encoder_cross_corpus_n30_v474.json")
SCHEMA = "carnot.harden_set_encoder_cross_corpus_n30_5171.v1"
SOLVE_PROVENANCE = "development_proxy"
INFERENCE_SUBSTRATE = "procedural_arcgen_extension_cpu_set_encoder_cross_corpus"
BLOCKED_UPSTREAM_VERDICT = "blocked_upstream_artifact_missing"
BLOCKED_INSUFFICIENT_POOL_VERDICT = "blocked_insufficient_disjoint_pool_size"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
SPEC_REFS = [
    "REQ-VERIFY-5171",
    "SCENARIO-VERIFY-5171",
    "SCENARIO-VERIFY-5171-UPSTREAM-BLOCKED",
    "SCENARIO-VERIFY-5171-INSUFFICIENT-POOL",
]

FIELD_PRINCIPLES = {
    "held_out_task_n": (
        "Must be >=30 to satisfy CLAUDE.md's CLT floor for percentage-point delta "
        "claims -- the exact quantity this task exists to increase."
    ),
    "cross_corpus_delta_n30": "set_encoder@1 - vote@1 on the expanded n>=30 second corpus.",
    "cross_corpus_delta_ci95_n30": (
        "The decisive number: does the win survive at proper statistical power, "
        "CI95 excluding 0?"
    ),
    "variance_is_genuine": (
        "exp5160's n=24 result was identical across all 5 seeds -- this field "
        "explicitly checks whether that is a real property of the domain or a "
        "too-small/too-easy-set artifact."
    ),
    "leak_audit_passed_on_expanded_set": (
        "A leak-audit that passed on the n=24 subset does not automatically cover "
        "newly-added items -- must be re-verified, not assumed."
    ),
    "gate_passed": (
        "n>=30 AND CI95 excludes 0 AND consistent direction/magnitude with exp5160 "
        "-- feeds exp5173's gated_on check directly. Do not redefine the threshold "
        "post hoc."
    ),
    "verifier_is_oracle": "Must remain the bare bool false for the oracle-distinct ARC verifier claim.",
    "solve_provenance": "Offline scoring over a static candidate pool, not a live hidden-game solve.",
    "random_seeds_used": "The >=5 seeds used for the expanded n>=30 replication protocol.",
    "inference_substrate": (
        "Declare accurately per the Inference-Substrate Declaration Discipline -- "
        "verifier_ensemble_against_cached_candidates if scoring an existing pool, "
        "live_llm_inference if new candidates had to be generated to reach n>=30."
    ),
    "reproducibility_checksum": (
        "Hash of the expanded ARC-GEN source description, leak audit, per-seed "
        "results, n>=30 CI, and gate decision."
    ),
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ AND state plainly "
        "whether the gate passed at n>=30 -- this directly determines whether "
        "exp5173 runs this milestone."
    ),
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "honest_verdict",
    "held_out_task_n",
    "cross_corpus_delta_n30",
    "cross_corpus_delta_ci95_n30",
    "variance_is_genuine",
    "leak_audit_passed_on_expanded_set",
    "gate_passed",
    "verifier_is_oracle",
    "solve_provenance",
    "random_seeds_used",
    "inference_substrate",
    "reproducibility_checksum",
    "field_principles",
    "spec_refs",
)

SeedReplicationResult = exp5160.SeedReplicationResult


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""


@dataclass(frozen=True)
class Preconditions:
    exp5160_artifact: dict[str, Any]
    original_payload: dict[str, Any]
    source_pool_payload: dict[str, Any]
    source_pool_rel: Path
    source_pool_sha256: str
    source_pool_task_n: int


@dataclass(frozen=True)
class ExpandedArcgenSelection:
    corpus: exp4244.GrownPoolCorpus
    tasks: list[dict[str, Any]]
    source_pool_task_n: int
    source_pool_sha256: str
    expanded_pool_sha256: str
    selected_arcgen_task_ids: list[str]
    generator_limit: int
    tasks_per_generator: int
    candidates_per_task: int
    dropped_overlap_task_n: int
    dropped_overlap_candidate_n: int
    dropped_too_small_task_n: int
    generator_failures: list[str]
    arcgen_path: str
    arcgen_commit: str | None


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


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


def _sha256_json(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resolve_source_path(repo_root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise BlockedRun(BLOCKED_UPSTREAM_VERDICT)
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _arcgen_commit(arcgen_path: Path) -> str | None:
    if not (arcgen_path / ".git").exists():
        return None
    proc = subprocess.run(  # pragma: no cover - exercised only against a real checkout.
        ["git", "-C", str(arcgen_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None  # pragma: no cover


def _load_preconditions(repo_root: Path) -> Preconditions:
    exp5160_artifact = _read_json_object(repo_root / EXP5160_REL)
    if exp5160_artifact.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_UPSTREAM_VERDICT)
    source_pool_rel = Path(str(exp5160_artifact.get("second_pool_source") or ""))
    source_pool_path = _resolve_source_path(repo_root, exp5160_artifact.get("second_pool_source"))
    source_pool_payload = _read_gzip_json_object(source_pool_path)
    original_payload = _read_gzip_json_object(repo_root / ORIGINAL_POOL_REL)
    source_task_n = exp5160_artifact.get("held_out_task_n")
    if type(source_task_n) is not int or source_task_n <= 0:
        source_task_n = len(exp5160._normalized_tasks(source_pool_payload))
    return Preconditions(
        exp5160_artifact=exp5160_artifact,
        original_payload=original_payload,
        source_pool_payload=source_pool_payload,
        source_pool_rel=source_pool_rel,
        source_pool_sha256=_sha256_file(source_pool_path),
        source_pool_task_n=source_task_n,
    )


def _candidate_grid_hash(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_grid_hash") or exp5160._grid_hash(candidate.get("grid")))


def _filtered_task_payload(
    *,
    task_payload: dict[str, Any],
    rows: list[Any],
    original_signature: dict[str, set[str]],
) -> tuple[dict[str, Any] | None, list[exp4244.GrownPoolRow], int]:
    kept_candidates: list[dict[str, Any]] = []
    kept_rows: list[exp4244.GrownPoolRow] = []
    dropped_candidate_n = 0
    for row in rows:
        candidate = task_payload["candidates"][row.candidate_index]
        grid_hash = _candidate_grid_hash(candidate)
        if (
            row.candidate_id in original_signature["candidate_ids"]
            or grid_hash in original_signature["candidate_grid_hashes"]
            or (row.correct and grid_hash in original_signature["gold_grid_hashes"])
        ):
            dropped_candidate_n += 1
            continue
        kept_candidates.append({**candidate, "candidate_grid_hash": grid_hash})
        kept_rows.append(
            exp4244.GrownPoolRow(
                task_id=row.task_id,
                candidate_id=row.candidate_id,
                candidate_index=row.candidate_index,
                correct=bool(row.correct),
                features=row.features,
                vote_weight=float(row.vote_weight),
            )
        )
    if len(kept_rows) < 2:
        return None, [], dropped_candidate_n
    vote_pick = max(kept_rows, key=lambda item: (item.vote_weight, -item.candidate_index))
    filtered_payload = {
        **task_payload,
        "candidate_count": len(kept_candidates),
        "candidates": kept_candidates,
        "oracle_present": any(row.correct for row in kept_rows),
        "vote_top_candidate_id": vote_pick.candidate_id,
        "vote_top_correct": bool(vote_pick.correct),
        "wrong_majority": bool(not vote_pick.correct),
    }
    return filtered_payload, kept_rows, dropped_candidate_n


def build_expanded_arcgen_selection(
    repo_root: Path | str,
    *,
    original_payload: dict[str, Any],
    source_pool_task_n: int,
    source_pool_sha256: str,
    arcgen_task_catalog: dict[str, tuple[Any, Any]] | None = None,
    generator_limit: int = DEFAULT_GENERATOR_LIMIT,
    tasks_per_generator: int = DEFAULT_TASKS_PER_GENERATOR,
    candidates_per_task: int = DEFAULT_CANDIDATES_PER_TASK,
    random_seed: int = exp4291.RANDOM_SEED,
) -> ExpandedArcgenSelection:
    """SCENARIO-VERIFY-5171: build same-source ARC-GEN n>=30 extension."""

    root = Path(repo_root)
    arcgen_path = root / exp4291.ARCGEN_REL
    try:
        catalog = (
            arcgen_task_catalog
            if arcgen_task_catalog is not None
            else exp4291.load_arcgen_task_catalog(arcgen_path)
        )
    except Exception as exc:  # pragma: no cover - real checkout availability varies by machine.
        raise BlockedRun(BLOCKED_INSUFFICIENT_POOL_VERDICT) from exc
    selected = sorted(catalog.items())[: max(0, int(generator_limit))]
    original_signature = exp5160._pool_signature(original_payload)
    rows: list[exp4244.GrownPoolRow] = []
    task_payloads: list[dict[str, Any]] = []
    dropped_overlap_task_n = 0
    dropped_overlap_candidate_n = 0
    dropped_too_small_task_n = 0
    generator_failures: list[str] = []
    generated_task_ordinal = 0

    for generator_index, (arcgen_task_id, task_info) in enumerate(selected):
        generator = task_info[0]
        for instance_index in range(max(1, int(tasks_per_generator))):
            random.seed(int(random_seed) + generator_index * 1009 + instance_index)
            try:
                example = generator()
            except Exception:  # pragma: no cover - covered by ARC-GEN's own builder tests.
                generator_failures.append(str(arcgen_task_id))
                break
            if not isinstance(example, dict) or not isinstance(example.get("output"), list):
                generator_failures.append(str(arcgen_task_id))  # pragma: no cover
                break  # pragma: no cover
            generated_rows, task_payload = exp4291._task_rows_from_example(
                source_task_id=str(arcgen_task_id),
                generator_index=generator_index,
                instance_index=instance_index,
                task_ordinal=generated_task_ordinal,
                example=example,
                candidates_per_task=int(candidates_per_task),
            )
            generated_task_ordinal += 1
            task_id = str(task_payload.get("task_id") or "")
            raw_task_id = str(task_payload.get("raw_task_id") or task_id)
            if task_id in original_signature["task_ids"] or raw_task_id in original_signature["raw_task_ids"]:
                dropped_overlap_task_n += 1
                continue
            filtered_task, filtered_rows, dropped_candidates = _filtered_task_payload(
                task_payload=task_payload,
                rows=generated_rows,
                original_signature=original_signature,
            )
            dropped_overlap_candidate_n += dropped_candidates
            if filtered_task is None:
                dropped_too_small_task_n += 1
                continue
            task_payloads.append(filtered_task)
            rows.extend(filtered_rows)

    expanded_payload = {
        "schema": "carnot.arcgen_cross_generator_pool_5171.v1",
        "source_parent": str(EXP5160_SECOND_POOL_REL),
        "source_parent_sha256": source_pool_sha256,
        "random_seed": int(random_seed),
        "source_kind": "arcgen",
        "generator_limit": int(generator_limit),
        "tasks_per_generator": int(tasks_per_generator),
        "candidates_per_task": int(candidates_per_task),
        "tasks": task_payloads,
    }
    expanded_sha256 = _sha256_json(expanded_payload)
    corpus = exp4244.GrownPoolCorpus(
        rows=rows,
        pool_artifact_path=(root / OUTPUT_REL).resolve(),
        pool_artifact_sha256=expanded_sha256,
        upstream_checksum="sha256:" + expanded_sha256,
        held_out_task_n=len(task_payloads),
        wrong_majority_n=sum(int(task.get("wrong_majority") is True) for task in task_payloads),
        positive_candidate_n=sum(int(row.correct) for row in rows),
    )
    return ExpandedArcgenSelection(
        corpus=corpus,
        tasks=task_payloads,
        source_pool_task_n=int(source_pool_task_n),
        source_pool_sha256=source_pool_sha256,
        expanded_pool_sha256=expanded_sha256,
        selected_arcgen_task_ids=[str(task_id) for task_id, _ in selected],
        generator_limit=int(generator_limit),
        tasks_per_generator=int(tasks_per_generator),
        candidates_per_task=int(candidates_per_task),
        dropped_overlap_task_n=dropped_overlap_task_n,
        dropped_overlap_candidate_n=dropped_overlap_candidate_n,
        dropped_too_small_task_n=dropped_too_small_task_n,
        generator_failures=sorted(set(generator_failures)),
        arcgen_path=str(arcgen_path.resolve()),
        arcgen_commit=_arcgen_commit(arcgen_path),
    )


def _selection_for_5160_audit(selection: ExpandedArcgenSelection) -> exp5160.SecondPoolSelection:
    return exp5160.SecondPoolSelection(
        corpus=selection.corpus,
        source_rel=Path("external/ARC-GEN"),
        source_sha256=selection.expanded_pool_sha256,
        source_kind="arcgen_non_degenerate_cross_generator_n30_extension",
        classic_arc_static_puzzle_pool=False,
        preferred_audit={},
        adapter={
            "adapter": "exp4291_arcgen_non_degenerate_extension_to_exp4244_feature_schema",
            "dropped_overlap_task_n": selection.dropped_overlap_task_n,
            "dropped_overlap_candidate_n": selection.dropped_overlap_candidate_n,
            "dropped_too_small_task_n": selection.dropped_too_small_task_n,
            "selected_task_n": selection.corpus.held_out_task_n,
            "selected_candidate_n": len(selection.corpus.rows),
            "source_kind": "arcgen_non_degenerate_cross_generator_n30_extension",
        },
    )


def expanded_leak_audit(
    original_payload: dict[str, Any],
    selection: ExpandedArcgenSelection,
    seed_results: list[SeedReplicationResult],
) -> dict[str, Any]:
    """REQ-VERIFY-5171: rerun the leak audit on the expanded scored set."""

    base = exp5160.second_pool_leak_audit(
        original_payload,
        _selection_for_5160_audit(selection),
        seed_results,
    )
    original = exp5160._pool_signature(original_payload)
    expanded = exp5160._pool_signature({"tasks": selection.tasks})
    raw_count, raw_sample = exp5160._overlap_count(original["raw_task_ids"], expanded["raw_task_ids"])
    grid_count, grid_sample = exp5160._overlap_count(
        original["candidate_grid_hashes"], expanded["candidate_grid_hashes"]
    )
    gold_count, gold_sample = exp5160._overlap_count(
        original["gold_grid_hashes"], expanded["gold_grid_hashes"]
    )
    passed = bool(base["passed"] and raw_count == 0 and grid_count == 0 and gold_count == 0)
    return {
        **base,
        "passed": passed,
        "raw_task_id_collision_count": raw_count,
        "raw_task_id_collisions_sample": raw_sample,
        "candidate_grid_hash_collision_count": grid_count,
        "candidate_grid_hash_collisions_sample": grid_sample,
        "gold_grid_hash_collision_count": gold_count,
        "gold_grid_hash_collisions_sample": gold_sample,
        "adapter_filtered_overlap_task_n": selection.dropped_overlap_task_n,
        "adapter_filtered_overlap_candidate_n": selection.dropped_overlap_candidate_n,
        "adapter_dropped_too_small_task_n": selection.dropped_too_small_task_n,
    }


def _task_delta_means(seed_results: list[SeedReplicationResult]) -> list[float]:
    populated = [result.task_deltas for result in seed_results if result.task_deltas]
    if not populated:
        return []
    task_n = min(len(items) for items in populated)
    return [
        _mean([float(items[index]) for items in populated])
        for index in range(task_n)
    ]


def _task_bootstrap_ci95(
    task_deltas: list[float], *, random_seed: int, resamples: int
) -> list[float]:
    if not task_deltas:
        return [0.0, 0.0]
    if len(task_deltas) == 1:
        point = _round_metric(task_deltas[0])
        return [point, point]
    rng = random.Random(int(random_seed))
    n = len(task_deltas)
    samples = [
        sum(task_deltas[rng.randrange(n)] for _ in range(n)) / float(n)
        for _ in range(max(1, int(resamples)))
    ]
    samples.sort()
    return [
        _round_metric(samples[int(0.025 * (len(samples) - 1))]),
        _round_metric(samples[int(0.975 * (len(samples) - 1))]),
    ]


def _variance_is_genuine(task_deltas: list[float], ci95: list[float]) -> bool:
    unique = {round(float(delta), 12) for delta in task_deltas}
    return bool(len(unique) > 1 and len(ci95) == 2 and float(ci95[1]) > float(ci95[0]))


def _ci_excludes_zero(ci95: list[float]) -> bool:
    return bool(len(ci95) == 2 and (ci95[0] > 0.0 or ci95[1] < 0.0))


def _consistent_with_exp5160(exp5160_artifact: dict[str, Any], delta: float) -> bool:
    original_delta = exp5160._safe_float(exp5160_artifact.get("cross_corpus_delta"))
    tolerance = max(0.2, abs(original_delta) * 0.5)
    return bool(original_delta > 0.0 and delta > 0.0 and abs(delta - original_delta) <= tolerance)


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


def reproducibility_checksum(artifact_without_checksum: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact_without_checksum.items()
        if key not in {"reproducibility_checksum", "adversarial_verify", "duration_s"}
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _expanded_pool_summary(selection: ExpandedArcgenSelection | None) -> dict[str, Any]:
    if selection is None:
        return {"usable_task_n": 0, "usable_candidate_n": 0}
    return {
        "source_pool": str(EXP5160_SECOND_POOL_REL),
        "source_pool_task_n": selection.source_pool_task_n,
        "source_pool_sha256": selection.source_pool_sha256,
        "expanded_pool_sha256": selection.expanded_pool_sha256,
        "usable_task_n": selection.corpus.held_out_task_n,
        "usable_candidate_n": len(selection.corpus.rows),
        "selected_arcgen_task_ids": selection.selected_arcgen_task_ids,
        "generator_limit": selection.generator_limit,
        "tasks_per_generator": selection.tasks_per_generator,
        "candidates_per_task": selection.candidates_per_task,
        "dropped_overlap_task_n": selection.dropped_overlap_task_n,
        "dropped_overlap_candidate_n": selection.dropped_overlap_candidate_n,
        "dropped_too_small_task_n": selection.dropped_too_small_task_n,
        "generator_failures": selection.generator_failures,
        "arcgen_path": selection.arcgen_path,
        "arcgen_commit": selection.arcgen_commit,
        "construction_policy": "exp4291_mixed_vote_winning_wrong_majority_and_no_oracle_tasks",
    }


def _blocked_artifact(
    reason: str,
    *,
    random_seed: int,
    duration_s: float,
    selection: ExpandedArcgenSelection | None = None,
) -> dict[str, Any]:
    artifact = {
        "schema": SCHEMA,
        "experiment": "experiment_5171_harden_set_encoder_cross_corpus_n30_v474",
        "status": "blocked",
        "headline_outcome": reason,
        "honest_verdict": reason,
        "held_out_task_n": selection.corpus.held_out_task_n if selection is not None else 0,
        "cross_corpus_delta_n30": 0.0,
        "cross_corpus_delta_ci95_n30": [0.0, 0.0],
        "variance_is_genuine": False,
        "leak_audit_passed_on_expanded_set": False,
        "gate_passed": False,
        "verifier_is_oracle": False,
        "solve_provenance": SOLVE_PROVENANCE,
        "random_seed": int(random_seed),
        "random_seeds_used": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "expanded_pool": _expanded_pool_summary(selection),
        "expanded_leak_audit": {"passed": False, "blocked_reason": reason},
        "per_seed_results": [],
        "per_seed_deltas": [],
        "task_delta_summary": {"mean": 0.0, "unique_values": []},
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "adversarial_verify": {"status": "pending"},
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _complete_artifact(
    *,
    exp5160_artifact: dict[str, Any],
    selection: ExpandedArcgenSelection,
    seed_results: list[SeedReplicationResult],
    leak_audit: dict[str, Any],
    random_seed: int,
    bootstrap_resamples: int,
    duration_s: float,
) -> dict[str, Any]:
    per_seed_deltas = [_round_metric(result.delta) for result in seed_results]
    task_deltas = _task_delta_means(seed_results)
    delta = _round_metric(_mean(per_seed_deltas))
    ci95 = _task_bootstrap_ci95(
        task_deltas,
        random_seed=random_seed,
        resamples=bootstrap_resamples,
    )
    variance_is_genuine = _variance_is_genuine(task_deltas, ci95)
    seed_delta_variance_is_zero = len({round(delta_item, 12) for delta_item in per_seed_deltas}) <= 1
    consistent = _consistent_with_exp5160(exp5160_artifact, delta)
    gate_passed = bool(
        selection.corpus.held_out_task_n >= MIN_HELD_OUT_TASK_N
        and _ci_excludes_zero(ci95)
        and consistent
        and leak_audit["passed"]
    )
    headline = (
        "arc_set_encoder_cross_corpus_gate_passed_n30"
        if gate_passed
        else "arc_set_encoder_cross_corpus_gate_not_passed_n30"
    )
    prefix = "success" if gate_passed else "complete"
    artifact = {
        "schema": SCHEMA,
        "experiment": "experiment_5171_harden_set_encoder_cross_corpus_n30_v474",
        "status": "complete",
        "headline_outcome": headline,
        "honest_verdict": (
            f"{prefix}_{headline}: gate "
            f"{'passed' if gate_passed else 'did not pass'} at n>=30"
        ),
        "held_out_task_n": selection.corpus.held_out_task_n,
        "cross_corpus_delta_n30": delta,
        "cross_corpus_delta_ci95_n30": ci95,
        "variance_is_genuine": variance_is_genuine,
        "seed_delta_variance_is_zero": seed_delta_variance_is_zero,
        "leak_audit_passed_on_expanded_set": bool(leak_audit["passed"]),
        "gate_passed": gate_passed,
        "consistent_with_exp5160": consistent,
        "exp5160_reference": {
            "artifact": str(EXP5160_REL),
            "held_out_task_n": exp5160_artifact.get("held_out_task_n"),
            "cross_corpus_delta": exp5160_artifact.get("cross_corpus_delta"),
            "cross_corpus_delta_ci95": exp5160_artifact.get("cross_corpus_delta_ci95"),
            "second_pool_source": exp5160_artifact.get("second_pool_source"),
            "verifier_is_oracle": exp5160_artifact.get("verifier_is_oracle"),
        },
        "verifier_is_oracle": False,
        "solve_provenance": SOLVE_PROVENANCE,
        "random_seed": int(random_seed),
        "random_seeds_used": [result.random_seed for result in seed_results],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "expanded_pool": _expanded_pool_summary(selection),
        "expanded_leak_audit": leak_audit,
        "per_seed_deltas": per_seed_deltas,
        "per_seed_delta_ci95": exp5160._multiseed_ci95(per_seed_deltas),
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
        "task_delta_summary": {
            "mean": _round_metric(_mean(task_deltas)),
            "unique_values": sorted({_round_metric(delta_item) for delta_item in task_deltas}),
            "task_bootstrap_resamples": int(bootstrap_resamples),
            "task_delta_n": len(task_deltas),
        },
        "model_specs": {
            "status": "complete",
            "protocol": "exp5160_deepsets_pooled_context_set_encoder_on_expanded_arcgen_second_pool",
            "set_encoder_architecture": "deepsets_pooled_context_set_encoder",
            "n_folds": DEFAULT_N_FOLDS,
            "training_epochs": DEFAULT_TRAINING_EPOCHS,
            "hidden_dim": DEFAULT_HIDDEN_DIM,
            "random_seeds": [result.random_seed for result in seed_results],
            "source_extension": "exp4291_arcgen_non_degenerate_generator_limit_10_tasks_per_generator_3",
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": round(float(duration_s), 6),
        "adversarial_verify": {"status": "pending"},
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _validate_float(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be a bare float")


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
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if type(artifact["held_out_task_n"]) is not int:
        raise ValueError("held_out_task_n must be a bare int")
    _validate_float(artifact["cross_corpus_delta_n30"], "cross_corpus_delta_n30")
    _validate_ci95(artifact["cross_corpus_delta_ci95_n30"], "cross_corpus_delta_ci95_n30")
    for field in ("variance_is_genuine", "leak_audit_passed_on_expanded_set", "gate_passed"):
        if type(artifact[field]) is not bool:
            raise ValueError(f"{field} must be a bare bool")
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


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    random_seeds: list[int] | None = None,
    generator_limit: int = DEFAULT_GENERATOR_LIMIT,
    tasks_per_generator: int = DEFAULT_TASKS_PER_GENERATOR,
    candidates_per_task: int = DEFAULT_CANDIDATES_PER_TASK,
    n_folds: int = DEFAULT_N_FOLDS,
    bootstrap_n: int = exp5160.DEFAULT_BOOTSTRAP_N,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    training_epochs: int = DEFAULT_TRAINING_EPOCHS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    lr: float = DEFAULT_LR,
    arcgen_task_catalog: dict[str, tuple[Any, Any]] | None = None,
    adversarial_runner: Any | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    seeds = list(random_seeds or DEFAULT_RANDOM_SEEDS)
    try:
        preconditions = _load_preconditions(root)
        selection = build_expanded_arcgen_selection(
            root,
            original_payload=preconditions.original_payload,
            source_pool_task_n=preconditions.source_pool_task_n,
            source_pool_sha256=preconditions.source_pool_sha256,
            arcgen_task_catalog=arcgen_task_catalog,
            generator_limit=generator_limit,
            tasks_per_generator=tasks_per_generator,
            candidates_per_task=candidates_per_task,
        )
        if selection.corpus.held_out_task_n < MIN_HELD_OUT_TASK_N or len(seeds) < 5:
            artifact = _blocked_artifact(
                BLOCKED_INSUFFICIENT_POOL_VERDICT,
                random_seed=random_seed,
                duration_s=time.perf_counter() - start,
                selection=selection,
            )
        else:
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
            leak_audit = expanded_leak_audit(preconditions.original_payload, selection, seed_results)
            artifact = _complete_artifact(
                exp5160_artifact=preconditions.exp5160_artifact,
                selection=selection,
                seed_results=seed_results,
                leak_audit=leak_audit,
                random_seed=random_seed,
                bootstrap_resamples=bootstrap_resamples,
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


_train_seed_replication = exp5160._train_seed_replication


def main() -> None:  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[3]
    print(json.dumps(run(repo_root), indent=2, sort_keys=True))
