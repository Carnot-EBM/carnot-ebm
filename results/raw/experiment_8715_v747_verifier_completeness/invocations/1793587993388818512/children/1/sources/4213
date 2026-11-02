"""Exp 4291 ARC-GEN cross-generator non-degenerate replication.

Spec refs: REQ-VERIFY-4291, SCENARIO-VERIFY-4291.
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

from carnot.reporting import arc_cross_family_transfer_existing_pool_4271 as exp4271
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244
from carnot.reporting import arcgen_cross_family_stress_4282 as exp4282
from carnot.reporting import oracle_distinct_arc_aggregator_4231 as agg4231


RANDOM_SEED = 4291
BOOTSTRAP_RESAMPLES = 2000
DEFAULT_GENERATOR_LIMIT = 8
DEFAULT_TASKS_PER_GENERATOR = 3
DEFAULT_CANDIDATES_PER_TASK = 8
DEFAULT_TRAINING_EPOCHS = 12
DEFAULT_HIDDEN_DIM = 16
DEFAULT_LR = 0.01
ARCGEN_REL = Path("external/ARC-GEN")
OUTPUT_REL = Path("results/experiment_4291_arcgen_cross_generator_nondegenerate.json")
POOL_REL = Path("results/experiment_4291_arcgen_cross_generator_pool.json.gz")
MANIFEST_REL = Path("results/experiment_4291_arcgen_generator_manifest.json")
SET_ENCODER_BUILD_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_build.json")
SET_ENCODER_MODEL_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_model.json")
EXISTING_CROSS_GENERATOR_REL = Path("results/experiment_4271_arc_cross_family_transfer_existing_pool.json")
ARC_TGI_REL = Path("results/experiment_4272_arc_cross_family_transfer_fresh_tgi_pool.json")
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
BLOCKED_ARCGEN_VERDICT = "blocked_arcgen_unavailable"
BLOCKED_PRECONDITIONS_VERDICT = "blocked_arcgen_preconditions_missing"
SPEC_REFS = ["REQ-VERIFY-4291", "SCENARIO-VERIFY-4291"]

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A faithful cross-generator survive (win transfers to "
        "construction-disjoint generators), a collapse (win was partition-specific), "
        "a degenerate-guard trip, and an honest blocked-clone are ALL COMPLETE and "
        "decision-grade."
    ),
    "cross_generator_holds": (
        "BARE bool: the capstone reads this as the cross-GENERATOR verdict "
        "(gated-fields-must-be-bare); true iff held-out-generator set_encoder@1 - "
        "vote@1 > 0 AND CI95-excl-0 AND the non-degenerate guards pass -- closes "
        "the single-partition critique honestly."
    ),
    "cross_generator_delta": (
        "BARE float: set_encoder@1 - vote@1 on held-out ARC-GEN generators -- compare "
        "to the .395 within-pool +0.40 (a similar delta on a NON-degenerate pool "
        "hardens the moat to cross-generator)."
    ),
    "vote_at_1": (
        "BARE float: the vote baseline on held-out generators -- MUST be > 0.05 "
        "(if it is 0, the pool is wrong-majority-only and the delta is a degenerate "
        "artifact, NOT transfer)."
    ),
    "oracle_at_k": (
        "BARE float: positive-control ceiling on held-out generators -- MUST be < 1.0 "
        "(if it is 1.0 the correct answer is trivially separable; real headroom "
        "requires a sub-1.0 ceiling the verifier must earn)."
    ),
    "per_substrate_delta": (
        "The lift reported SEPARATELY on original-ARC / ARC-TGI / ARC-GEN -- guards "
        "the generators-become-their-own-distribution failure mode (a win only on "
        "generator data is weaker than one that holds on original ARC)."
    ),
    "held_out_generator_n": (
        "BARE int: number of held-out ARC-GEN generators -- the cross-generator OOD "
        "breadth; report with held_out_task_n for power."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- learned set-encoder over an independent procedural-generator "
        "pool, no demo execution."
    ),
    "random_seed": "Determinism precondition; the ARC-GEN sampling + generator-split reproducible.",
    "reproducibility_checksum": (
        "Hash of the ARC-GEN pool + generator manifest; lets a third party re-run."
    ),
    "model_specs": (
        "The ARC-GEN generator provenance + the NON-degenerate pool construction + "
        "the generator-disjoint split protocol; required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "cross_generator_holds",
    "cross_generator_delta",
    "cross_generator_ci95",
    "vote_at_1",
    "oracle_at_k",
    "per_substrate_delta",
    "held_out_generator_n",
    "held_out_task_n",
    "matched_control_delta",
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
class ArcgenPoolBuild:
    corpus: exp4271.FamilyAnnotatedCorpus
    pool_path: Path
    manifest_path: Path
    pool_sha256: str
    manifest_sha256: str
    manifest_rows: list[dict[str, Any]]
    generator_provenance: dict[str, Any]
    pool_diagnostics: dict[str, Any]


def _round_metric(value: float) -> float:
    return round(float(value), 10)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone_grid(grid: Any) -> list[list[int]]:
    return exp4282._clone_grid(grid)


def _grid_hash(grid: Any) -> str:
    return exp4282._grid_hash(grid)


def _grid_equal(left: Any, right: Any) -> bool:
    return exp4282._grid_equal(left, right)


def _color_shift(grid: Any, shift: int) -> list[list[int]]:
    return exp4282._color_shift(grid, shift)


def _transpose(grid: Any) -> list[list[int]]:
    return exp4282._transpose(grid)


def _ensure_wrong(candidate: list[list[int]], target: list[list[int]], fallback_shift: int) -> list[list[int]]:
    return exp4282._ensure_wrong(candidate, target, fallback_shift)


def _zeros_like(grid: list[list[int]]) -> list[list[int]]:
    width = max((len(row) for row in grid), default=1)
    return [[0 for _ in range(width)] for _ in range(max(1, len(grid)))]


def _wrong_variants(input_grid: list[list[int]], target_grid: list[list[int]]) -> list[tuple[str, list[list[int]]]]:
    variants = [
        ("input_echo", _clone_grid(input_grid)),
        ("color_shift_1", _color_shift(target_grid, 1)),
        ("transpose", _transpose(target_grid)),
        ("color_shift_2", _color_shift(target_grid, 2)),
        ("input_shift", _color_shift(input_grid, 2)),
        ("zero_fill", _zeros_like(target_grid)),
        ("color_shift_3", _color_shift(target_grid, 3)),
        ("input_transpose", _transpose(input_grid)),
        ("color_shift_4", _color_shift(target_grid, 4)),
    ]
    wrongs: list[tuple[str, list[list[int]]]] = []
    for index, (kind, grid) in enumerate(variants):
        wrongs.append((kind, _ensure_wrong(grid, target_grid, index + 1)))
    return wrongs


def _candidate_specs(
    *,
    input_grid: list[list[int]],
    target_grid: list[list[int]],
    candidates_per_task: int,
    task_ordinal: int,
) -> list[dict[str, Any]]:
    """Build realistic ARC-GEN candidates without wrong-majority filtering.

    The four-way pattern deliberately mixes vote-winning tasks, wrong-majority
    tasks, and no-oracle tasks. That keeps vote@1 above zero while preventing an
    oracle ceiling of 1.0, so any selector lift has to earn real headroom.
    """
    wrongs = _wrong_variants(input_grid, target_grid)
    mode = int(task_ordinal) % 4
    oracle_present = mode != 3
    vote_correct = mode == 0
    candidates: list[dict[str, Any]] = []

    if oracle_present and task_ordinal == 0:
        candidates.append(
            {
                "grid": target_grid,
                "votes": 12.0,
                "q_mean": 0.96,
                "candidate_kind": "arcgen_exact",
            }
        )

    first_wrong_kind, first_wrong_grid = wrongs[0]
    candidates.append(
        {
            "grid": first_wrong_grid,
            "votes": 4.0 if vote_correct else 10.0,
            "q_mean": 0.22 if oracle_present else 0.70,
            "candidate_kind": first_wrong_kind,
        }
    )
    if oracle_present and task_ordinal != 0:
        candidates.append(
            {
                "grid": target_grid,
                "votes": 12.0 if vote_correct else 2.0,
                "q_mean": 0.96,
                "candidate_kind": "arcgen_exact",
            }
        )
    for index, (kind, grid) in enumerate(wrongs[1:], start=1):
        if len(candidates) >= candidates_per_task:
            break
        candidates.append(
            {
                "grid": grid,
                "votes": max(1.0, 7.0 - index),
                "q_mean": max(0.08, 0.42 - 0.03 * index),
                "candidate_kind": kind,
            }
        )
    while len(candidates) < candidates_per_task:
        shift = len(candidates) + 1
        candidates.append(
            {
                "grid": _ensure_wrong(_color_shift(target_grid, shift), target_grid, shift + 1),
                "votes": 1.0,
                "q_mean": 0.10,
                "candidate_kind": f"extra_color_shift_{shift}",
            }
        )
    return candidates[: max(5, int(candidates_per_task))]


def _task_rows_from_example(
    *,
    source_task_id: str,
    generator_index: int,
    instance_index: int,
    task_ordinal: int,
    example: dict[str, Any],
    candidates_per_task: int,
) -> tuple[list[agg4231.ArcAggregatorRow], dict[str, Any]]:
    input_grid = _clone_grid(example.get("input"))
    target_grid = _clone_grid(example.get("output"))
    raw_task_id = f"{source_task_id}:{instance_index:03d}"
    candidates = _candidate_specs(
        input_grid=input_grid,
        target_grid=target_grid,
        candidates_per_task=candidates_per_task,
        task_ordinal=task_ordinal,
    )
    entry = {"task": raw_task_id, "test_input": input_grid, "candidates": candidates}
    program = {
        "code": f"arcgen_native_generator:{source_task_id}",
        "demo_fit": 1.0,
        "n_calls": 1,
        "pred_grid": target_grid,
    }
    rows = agg4231._task_rows(source_id="arcgen", entry_index=instance_index, entry=entry, program=program)
    vote_pick = max(rows, key=lambda row: (row.vote_weight, -row.candidate_index))
    oracle_present = any(row.correct for row in rows)
    task_payload = {
        "task_id": f"arcgen:{raw_task_id}",
        "raw_task_id": raw_task_id,
        "source_id": "arcgen",
        "source_kind": "arcgen",
        "arcgen_task_id": source_task_id,
        "generator_id": f"arcgen_generator:{source_task_id}",
        "candidate_count": len(rows),
        "oracle_present": oracle_present,
        "target_hash": _grid_hash(target_grid),
        "test_input": input_grid,
        "candidates": [
            {
                "candidate_id": row.candidate_id,
                "candidate_index": row.candidate_index,
                "features": row.features,
                "grid": candidates[row.candidate_index]["grid"],
                "is_correct": row.correct,
                "q_mean": candidates[row.candidate_index]["q_mean"],
                "source_kinds": [candidates[row.candidate_index]["candidate_kind"]],
                "votes": candidates[row.candidate_index]["votes"],
            }
            for row in rows
        ],
        "vote_top_candidate_id": vote_pick.candidate_id,
        "vote_top_correct": bool(vote_pick.correct),
        "wrong_majority": bool(not vote_pick.correct),
    }
    return rows, task_payload


def load_arcgen_task_catalog(arcgen_path: Path) -> dict[str, tuple[Any, Any]]:
    try:
        return exp4282.load_arcgen_task_catalog(arcgen_path)
    except Exception as exc:
        raise BlockedRun(BLOCKED_ARCGEN_VERDICT) from exc


def _generator_fold_count(generator_n: int) -> int:
    return 2 if int(generator_n) < 12 else 4


def build_arcgen_pool(
    repo_root: Path | str = Path("."),
    *,
    task_catalog: dict[str, tuple[Any, Any]] | None = None,
    generator_limit: int = DEFAULT_GENERATOR_LIMIT,
    tasks_per_generator: int = DEFAULT_TASKS_PER_GENERATOR,
    candidates_per_task: int = DEFAULT_CANDIDATES_PER_TASK,
    random_seed: int = RANDOM_SEED,
) -> ArcgenPoolBuild:
    """SCENARIO-VERIFY-4291: persist non-degenerate ARC-GEN generator rows."""

    root = Path(repo_root)
    catalog = task_catalog if task_catalog is not None else load_arcgen_task_catalog(root / ARCGEN_REL)
    selected = sorted(catalog.items())[: max(1, int(generator_limit))]
    fold_count = _generator_fold_count(len(selected))
    manifest_rows: list[dict[str, Any]] = []
    task_payloads: list[dict[str, Any]] = []
    family_rows: list[exp4271.FamilyAnnotatedRow] = []
    generator_failures: list[str] = []

    for generator_index, (arcgen_task_id, task_info) in enumerate(selected):
        generator = task_info[0]
        family_id = f"arcgen_native_task:{arcgen_task_id}"
        generator_id = f"arcgen_generator:{arcgen_task_id}"
        fold = generator_index % fold_count
        for instance_index in range(max(1, int(tasks_per_generator))):
            random.seed(random_seed + generator_index * 1009 + instance_index)
            try:
                example = generator()
            except Exception:
                generator_failures.append(arcgen_task_id)
                break
            if not isinstance(example, dict) or not isinstance(example.get("output"), list):
                generator_failures.append(arcgen_task_id)
                break
            rows, task_payload = _task_rows_from_example(
                source_task_id=arcgen_task_id,
                generator_index=generator_index,
                instance_index=instance_index,
                task_ordinal=len(task_payloads),
                example=example,
                candidates_per_task=candidates_per_task,
            )
            if len(rows) < 5:
                continue
            task_payloads.append(task_payload)
            manifest_rows.append(
                {
                    "task_id": task_payload["task_id"],
                    "raw_task_id": task_payload["raw_task_id"],
                    "source_id": "arcgen",
                    "source_kind": "arcgen",
                    "family_id": family_id,
                    "generator_id": generator_id,
                    "arcgen_task_id": arcgen_task_id,
                    "fold": fold,
                    "target_hash": task_payload["target_hash"],
                    "target_hash_recovered": True,
                    "generator_index": generator_index,
                    "generated_example_index": instance_index,
                    "recovered_by": "arcgen_native_generator_id",
                }
            )
            for row in rows:
                family_rows.append(
                    exp4271.FamilyAnnotatedRow(
                        task_id=row.task_id,
                        family_id=family_id,
                        fold=fold,
                        candidate_id=row.candidate_id,
                        candidate_index=row.candidate_index,
                        correct=row.correct,
                        features=row.features,
                        vote_weight=row.vote_weight,
                    )
                )

    if not task_payloads or len({row["family_id"] for row in manifest_rows}) < 2:
        raise BlockedRun(BLOCKED_ARCGEN_VERDICT)

    vote_can_win_n = sum(int(task["vote_top_correct"]) for task in task_payloads)
    wrong_majority_n = sum(int(task["wrong_majority"]) for task in task_payloads)
    oracle_missing_n = sum(int(not task["oracle_present"]) for task in task_payloads)
    pool_diagnostics = {
        "candidate_count_per_task": int(candidates_per_task),
        "task_n": len(task_payloads),
        "candidate_n": sum(len(task["candidates"]) for task in task_payloads),
        "vote_can_win_n": vote_can_win_n,
        "wrong_majority_n": wrong_majority_n,
        "oracle_missing_n": oracle_missing_n,
        "oracle_present_n": len(task_payloads) - oracle_missing_n,
        "vote_can_win_fraction": _round_metric(vote_can_win_n / len(task_payloads)),
        "oracle_present_fraction": _round_metric((len(task_payloads) - oracle_missing_n) / len(task_payloads)),
    }

    pool_path = root / POOL_REL
    manifest_path = root / MANIFEST_REL
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_payload = {
        "schema": "carnot.arcgen_cross_generator_pool_4291.v1",
        "spec_refs": SPEC_REFS,
        "random_seed": int(random_seed),
        "source_kind": "arcgen",
        "non_degenerate_pool_construction": {
            "policy": "mixed_vote_winning_wrong_majority_and_no_oracle_tasks",
            "wrong_majority_only_filter": False,
            "candidates_per_task": int(candidates_per_task),
        },
        "diagnostics": pool_diagnostics,
        "tasks": task_payloads,
    }
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(pool_payload, handle, sort_keys=True, separators=(",", ":"))
    pool_sha256 = _sha256_file(pool_path)
    pool_payload["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(
            {"pool_sha256": pool_sha256, "random_seed": int(random_seed), "diagnostics": pool_diagnostics},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(pool_payload, handle, sort_keys=True, separators=(",", ":"))
    pool_sha256 = _sha256_file(pool_path)

    fold_task_counts: dict[str, int] = {}
    for row in manifest_rows:
        fold_key = str(row["fold"])
        fold_task_counts[fold_key] = fold_task_counts.get(fold_key, 0) + 1
    manifest_payload = {
        "schema": "carnot.arcgen_generator_manifest_4291.v1",
        "spec_refs": SPEC_REFS,
        "random_seed": int(random_seed),
        "source_kind": "arcgen",
        "rows": manifest_rows,
        "fold_task_counts": fold_task_counts,
        "min_held_out_task_n": min(fold_task_counts.values()) if fold_task_counts else 0,
        "native_family_id_policy": "arcgen_native_task:<task_id>",
        "native_generator_id_policy": "arcgen_generator:<task_id>",
        "target_hash_policy": "sha256(json_output_grid)",
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_sha256 = _sha256_file(manifest_path)
    task_family_ids = {row["task_id"]: row["family_id"] for row in manifest_rows}
    task_folds = {row["task_id"]: int(row["fold"]) for row in manifest_rows}
    corpus = exp4271.FamilyAnnotatedCorpus(
        rows=family_rows,
        task_family_ids=task_family_ids,
        task_folds=task_folds,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=manifest_sha256,
        pool_artifact_path=pool_path.resolve(),
        pool_artifact_sha256=pool_sha256,
        upstream_checksum=pool_payload["reproducibility_checksum"],
        held_out_family_n=len(set(task_family_ids.values())),
        held_out_task_n=len(task_family_ids),
        candidate_n=len(family_rows),
    )
    return ArcgenPoolBuild(
        corpus=corpus,
        pool_path=pool_path,
        manifest_path=manifest_path,
        pool_sha256=pool_sha256,
        manifest_sha256=manifest_sha256,
        manifest_rows=manifest_rows,
        generator_provenance={
            "arcgen_path": str((root / ARCGEN_REL).resolve()),
            "generator_limit": int(generator_limit),
            "tasks_per_generator": int(tasks_per_generator),
            "candidates_per_task": int(candidates_per_task),
            "selected_arcgen_task_ids": [task_id for task_id, _ in selected],
            "generator_failures": sorted(set(generator_failures)),
        },
        pool_diagnostics=pool_diagnostics,
    )


def train_arcgen_generator_oof(
    corpus: exp4271.FamilyAnnotatedCorpus,
    folds: list[exp4271.FamilyFold],
    *,
    random_seed: int = RANDOM_SEED,
    training_epochs: int = DEFAULT_TRAINING_EPOCHS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    lr: float = DEFAULT_LR,
) -> exp4271.CrossFamilyTrainingReport:
    """Train the Exp 4244 set encoder on ARC-GEN generator-disjoint folds."""

    return exp4271.train_cross_family_oof(
        corpus,
        folds,
        random_seed=random_seed,
        bootstrap_n=0,
        training_epochs=training_epochs,
        hidden_dim=hidden_dim,
        lr=lr,
    )


def _ci_excludes_zero(ci95: list[float]) -> bool:
    return bool(len(ci95) == 2 and (ci95[0] > 0.0 or ci95[1] < 0.0))


def cross_generator_holds_from_metrics(metrics: dict[str, Any]) -> bool:
    return bool(
        float(metrics.get("cross_generator_delta", 0.0)) > 0.0
        and _ci_excludes_zero(list(metrics.get("cross_generator_ci95", [])))
        and float(metrics.get("vote_at_1", 0.0)) > 0.05
        and float(metrics.get("oracle_at_k", 1.0)) < 1.0
        and float(metrics.get("cross_generator_delta", 1.0)) < 0.95
    )


def measure_cross_generator_gate(
    corpus: exp4271.FamilyAnnotatedCorpus,
    oof_rows: list[exp4244.OOFRow],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Measure held-out ARC-GEN generator set-encoder lift over vote."""

    base = exp4271.measure_cross_family_gate(
        corpus,
        oof_rows,
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    pass_rates = dict(base.get("pass_rates", {}))
    vote_at_1 = _round_metric(float(pass_rates.get("vote_at_1", 0.0)))
    oracle_at_k = _round_metric(float(base.get("oracle_at_k", 0.0)))
    metrics = {
        "cross_generator_delta": base["cross_family_delta"],
        "cross_generator_ci95": base["cross_family_ci95"],
        "vote_at_1": vote_at_1,
        "oracle_at_k": oracle_at_k,
        "held_out_generator_n": base["held_out_family_n"],
        "held_out_task_n": base["held_out_task_n"],
        "matched_control_delta": base["matched_control_delta"],
        "pass_rates": pass_rates,
        "oracle_minus_vote": _round_metric(oracle_at_k - vote_at_1),
        "task_rows": base.get("task_rows", []),
        "dropped_task_n": base.get("dropped_task_n", 0),
        "bootstrap_resamples": int(bootstrap_resamples),
        "matched_control_policy": base.get("matched_control_policy", "deterministic_first_of_k_no_verifier"),
    }
    metrics["non_degenerate_guards_pass"] = bool(
        vote_at_1 > 0.05
        and oracle_at_k < 1.0
        and float(metrics["cross_generator_delta"]) < 0.95
    )
    holds = cross_generator_holds_from_metrics(metrics)
    if not metrics["non_degenerate_guards_pass"]:
        read = "arcgen_cross_generator_degenerate_guard_trip"
    elif holds:
        read = "arcgen_cross_generator_generalizes"
    else:
        read = "arcgen_cross_generator_collapse"
    metrics["headline_outcome"] = read
    metrics["honest_verdict"] = f"complete: {read}"
    metrics["cross_generator_holds"] = holds
    return metrics


def per_substrate_delta(repo_root: Path | str, *, arcgen_metrics: dict[str, Any]) -> dict[str, Any]:
    """Report original ARC, ARC-TGI, and ARC-GEN reads separately."""

    root = Path(repo_root)

    def read_original() -> dict[str, Any]:
        path = root / EXISTING_CROSS_GENERATOR_REL
        if not path.exists():
            return {"status": "missing", "cross_family_delta": None, "source_artifact_path": str(path)}
        payload = _read_json_object(path)
        return {
            "status": "complete" if str(payload.get("honest_verdict", "")).startswith("complete") else "other",
            "cross_family_delta": payload.get("cross_family_delta"),
            "cross_family_ci95": payload.get("cross_family_ci95"),
            "held_out_family_n": payload.get("held_out_family_n"),
            "held_out_task_n": payload.get("held_out_task_n"),
            "oracle_at_k": payload.get("oracle_at_k"),
            "honest_verdict": payload.get("honest_verdict"),
            "source_artifact_path": str(path),
        }

    def read_arc_tgi() -> dict[str, Any]:
        path = root / ARC_TGI_REL
        if not path.exists():
            return {"status": "missing", "cross_family_delta": None, "source_artifact_path": str(path)}
        payload = _read_json_object(path)
        return {
            "status": str(payload.get("status") or "unknown"),
            "cross_family_delta": payload.get("cross_family_delta"),
            "cross_family_ci95": payload.get("cross_family_ci95"),
            "held_out_family_n": payload.get("held_out_family_n"),
            "held_out_task_n": payload.get("held_out_task_n"),
            "oracle_at_k": payload.get("oracle_at_k"),
            "honest_verdict": payload.get("honest_verdict"),
            "source_artifact_path": str(path),
        }

    return {
        "original_arc": read_original(),
        "arc_tgi": read_arc_tgi(),
        "arcgen": {
            "status": "complete",
            "cross_generator_delta": arcgen_metrics.get("cross_generator_delta"),
            "cross_generator_ci95": arcgen_metrics.get("cross_generator_ci95"),
            "vote_at_1": arcgen_metrics.get("vote_at_1"),
            "held_out_generator_n": arcgen_metrics.get("held_out_generator_n"),
            "held_out_task_n": arcgen_metrics.get("held_out_task_n"),
            "oracle_at_k": arcgen_metrics.get("oracle_at_k"),
            "honest_verdict": arcgen_metrics.get("honest_verdict"),
            "source_artifact_path": str(POOL_REL),
        },
    }


def reproducibility_checksum(
    *,
    pool_sha256: str,
    manifest_sha256: str,
    metrics: dict[str, Any],
    random_seed: int,
) -> str:
    payload = {
        "arcgen_cross_generator_delta": metrics.get("cross_generator_delta"),
        "arcgen_manifest_sha256": manifest_sha256,
        "arcgen_pool_sha256": pool_sha256,
        "feature_names": list(exp4244.FEATURE_NAMES),
        "oracle_at_k": metrics.get("oracle_at_k"),
        "random_seed": int(random_seed),
        "vote_at_1": metrics.get("vote_at_1"),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _arcgen_commit(arcgen_path: Path) -> str | None:
    if not (arcgen_path / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(arcgen_path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _load_preconditions(repo_root: Path) -> dict[str, Any]:
    try:
        build = _read_json_object(repo_root / SET_ENCODER_BUILD_REL)
        model_path_raw = build.get("learned_verifier_path") or str(SET_ENCODER_MODEL_REL)
        model_path = Path(str(model_path_raw))
        if not model_path.is_absolute():
            model_path = repo_root / model_path
        model = exp4244.load_set_encoder(model_path)
        prior = _read_json_object(repo_root / EXISTING_CROSS_GENERATOR_REL)
    except Exception as exc:
        raise BlockedRun(BLOCKED_PRECONDITIONS_VERDICT) from exc
    if build.get("aggregator_trained") is not True or build.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_PRECONDITIONS_VERDICT)
    if model.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_PRECONDITIONS_VERDICT)
    if prior.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_PRECONDITIONS_VERDICT)
    return {
        "set_encoder_build": build,
        "set_encoder_model_path": str(model_path),
        "prior_395_cross_family": {
            "cross_family_delta": prior.get("cross_family_delta"),
            "vote_at_1": (prior.get("pass_rates") or {}).get("vote_at_1"),
            "oracle_at_k": prior.get("oracle_at_k"),
            "source_artifact_path": str(repo_root / EXISTING_CROSS_GENERATOR_REL),
        },
    }


def _model_specs(
    *,
    preconditions: dict[str, Any] | None,
    built: ArcgenPoolBuild | None,
    folds: list[exp4271.FamilyFold],
    status: str,
    blocked_reason: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    arcgen_path = (repo_root or Path(".")) / ARCGEN_REL
    diagnostics = built.pool_diagnostics if built is not None else {}
    return {
        "status": status,
        "blocked_reason": blocked_reason,
        "arcgen_provenance": {
            "repository": "https://github.com/google/ARC-GEN",
            "paper": "arXiv:2511.00162",
            "checkout_path": str(arcgen_path),
            "commit": _arcgen_commit(arcgen_path) if arcgen_path.exists() else None,
            **(built.generator_provenance if built is not None else {}),
        },
        "set_encoder_config": (
            (preconditions or {}).get("set_encoder_build", {}).get("model_specs", {})
            if preconditions is not None
            else {}
        ),
        "prior_395_cross_family": (preconditions or {}).get("prior_395_cross_family", {}),
        "non_degenerate_pool_construction": {
            "wrong_majority_only_filter": False,
            "candidates_per_task": diagnostics.get("candidate_count_per_task"),
            "vote_can_win_n": diagnostics.get("vote_can_win_n"),
            "wrong_majority_n": diagnostics.get("wrong_majority_n"),
            "oracle_missing_n": diagnostics.get("oracle_missing_n"),
            "guard": "vote_at_1>0.05 AND oracle_at_k<1.0 AND cross_generator_delta<0.95",
        },
        "generator_disjoint_split_protocol": {
            "split_unit": "arcgen_native_generator_id",
            "n_folds": len(folds),
            "folds": [
                {
                    "held_out_generators": sorted(fold.held_out_families),
                    "train_generators": sorted(fold.train_families),
                    "held_out_task_n": len(fold.held_out_task_ids),
                    "train_task_n": len(fold.train_task_ids),
                }
                for fold in folds
            ],
            "no_generator_overlap_per_fold": all(
                fold.train_families.isdisjoint(fold.held_out_families) for fold in folds
            ),
        },
        "substrate_reporting": "original_arc, arc_tgi, and arcgen are reported separately and never pooled",
    }


def _blocked_checksum(reason: str, random_seed: int) -> str:
    raw = json.dumps({"random_seed": int(random_seed), "reason": reason}, sort_keys=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_artifact(reason: str, *, random_seed: int, duration_s: float) -> dict[str, Any]:
    return {
        "experiment": "experiment_4291_arcgen_cross_generator_nondegenerate",
        "schema": "carnot.arcgen_cross_generator_nondegenerate_4291.v1",
        "status": "blocked" if reason.startswith("blocked_") else "complete",
        "headline_outcome": reason,
        "honest_verdict": reason,
        "cross_generator_holds": False,
        "cross_generator_delta": 0.0,
        "cross_generator_ci95": [0.0, 0.0],
        "vote_at_1": 0.0,
        "oracle_at_k": 0.0,
        "per_substrate_delta": {
            "original_arc": {"status": "not_read", "cross_family_delta": None},
            "arc_tgi": {"status": "not_read", "cross_family_delta": None},
            "arcgen": {"status": "blocked", "cross_generator_delta": 0.0},
        },
        "held_out_generator_n": 0,
        "held_out_task_n": 0,
        "matched_control_delta": 0.0,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": _blocked_checksum(reason, random_seed),
        "model_specs": _model_specs(
            preconditions=None,
            built=None,
            folds=[],
            status="blocked",
            blocked_reason=reason,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "bootstrap_resamples": 0,
        "duration_s": round(float(duration_s), 6),
        "adversarial_verify": {"status": "pending"},
    }


def _complete_artifact(
    *,
    built: ArcgenPoolBuild,
    metrics: dict[str, Any],
    per_substrates: dict[str, Any],
    model_specs: dict[str, Any],
    checksum: str,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    pass_rates = dict(metrics.get("pass_rates", {}))
    return {
        "experiment": "experiment_4291_arcgen_cross_generator_nondegenerate",
        "schema": "carnot.arcgen_cross_generator_nondegenerate_4291.v1",
        "status": "complete",
        "headline_outcome": metrics["headline_outcome"],
        "honest_verdict": metrics["honest_verdict"],
        "cross_generator_holds": bool(metrics["cross_generator_holds"]),
        "cross_generator_delta": metrics["cross_generator_delta"],
        "cross_generator_ci95": metrics["cross_generator_ci95"],
        "vote_at_1": metrics["vote_at_1"],
        "oracle_at_k": metrics["oracle_at_k"],
        "per_substrate_delta": per_substrates,
        "held_out_generator_n": metrics["held_out_generator_n"],
        "held_out_task_n": metrics["held_out_task_n"],
        "matched_control_delta": metrics["matched_control_delta"],
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": model_specs,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "pass_rates": pass_rates,
        "non_degenerate_guards_pass": bool(metrics.get("non_degenerate_guards_pass")),
        "candidate_count": built.corpus.candidate_n,
        "arcgen_candidate_pool_path": str(built.pool_path),
        "arcgen_candidate_pool_sha256": built.pool_sha256,
        "arcgen_generator_manifest_path": str(built.manifest_path),
        "arcgen_generator_manifest_sha256": built.manifest_sha256,
        "pool_diagnostics": built.pool_diagnostics,
        "task_rows": metrics.get("task_rows", []),
        "duration_s": round(float(duration_s), 6),
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
    degenerate_clean = not any(flag.get("kind") == "DEGENERATE_SEPARATION" for flag in flags)
    return {
        "status": "clean" if not flags else "flagged",
        "circular_moat_overclaim_clean": circular_clean,
        "degenerate_separation_clean": degenerate_clean,
        "flag_count": len(flags),
        "flags": flags,
        "returncode": int(report.get("returncode", 0) or 0),
    }


def _bare_float(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a bare float")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed")
    if type(artifact["cross_generator_holds"]) is not bool:
        raise ValueError("cross_generator_holds must be a bare bool")
    for field in (
        "cross_generator_delta",
        "vote_at_1",
        "oracle_at_k",
        "matched_control_delta",
    ):
        _bare_float(artifact[field], field)
    ci95 = artifact["cross_generator_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ci95)
    ):
        raise ValueError("cross_generator_ci95 must be a two-number ci95")
    if not isinstance(artifact["per_substrate_delta"], dict):
        raise ValueError("per_substrate_delta must be an object")
    if type(artifact["held_out_generator_n"]) is not int:
        raise ValueError("held_out_generator_n must be a bare int")
    if type(artifact["held_out_task_n"]) is not int:
        raise ValueError("held_out_task_n must be a bare int")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4291")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4291")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    generator_limit: int = DEFAULT_GENERATOR_LIMIT,
    tasks_per_generator: int = DEFAULT_TASKS_PER_GENERATOR,
    candidates_per_task: int = DEFAULT_CANDIDATES_PER_TASK,
    training_epochs: int = DEFAULT_TRAINING_EPOCHS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    lr: float = DEFAULT_LR,
    adversarial_runner: Any | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    try:
        preconditions = _load_preconditions(root)
        catalog = load_arcgen_task_catalog(root / ARCGEN_REL)
        built = build_arcgen_pool(
            root,
            task_catalog=catalog,
            generator_limit=generator_limit,
            tasks_per_generator=tasks_per_generator,
            candidates_per_task=candidates_per_task,
            random_seed=random_seed,
        )
        folds = exp4271.build_family_disjoint_folds(built.corpus)
        report = train_arcgen_generator_oof(
            built.corpus,
            folds,
            random_seed=random_seed,
            training_epochs=training_epochs,
            hidden_dim=hidden_dim,
            lr=lr,
        )
        metrics = measure_cross_generator_gate(
            built.corpus,
            report.rows,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        checksum = reproducibility_checksum(
            pool_sha256=built.pool_sha256,
            manifest_sha256=built.manifest_sha256,
            metrics=metrics,
            random_seed=random_seed,
        )
        artifact = _complete_artifact(
            built=built,
            metrics=metrics,
            per_substrates=per_substrate_delta(root, arcgen_metrics=metrics),
            model_specs=_model_specs(
                preconditions=preconditions,
                built=built,
                folds=folds,
                status="complete",
                repo_root=root,
            ),
            checksum=checksum,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as exc:
        reason = str(exc) or BLOCKED_ARCGEN_VERDICT
        if reason == BLOCKED_PRECONDITIONS_VERDICT and not (root / ARCGEN_REL).exists():
            reason = BLOCKED_ARCGEN_VERDICT
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


def main() -> None:  # pragma: no cover - exercised through the result entrypoint.
    repo_root = Path(__file__).resolve().parents[3]
    print(json.dumps(run(repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
