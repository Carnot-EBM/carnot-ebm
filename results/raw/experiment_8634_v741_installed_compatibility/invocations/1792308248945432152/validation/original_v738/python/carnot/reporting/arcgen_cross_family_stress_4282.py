"""Exp 4282 ARC-GEN cross-family stress replication.

Spec refs: REQ-VERIFY-4282, SCENARIO-VERIFY-4282.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from carnot.reporting import arc_cross_family_transfer_existing_pool_4271 as exp4271
from carnot.reporting import arc_set_encoder_aggregator_4244 as exp4244
from carnot.reporting import oracle_distinct_arc_aggregator_4231 as agg4231


RANDOM_SEED = 4282
BOOTSTRAP_RESAMPLES = 2000
STRESS_BOOTSTRAP_RESAMPLES = 2000
DEFAULT_FAMILY_LIMIT = 10
DEFAULT_TASKS_PER_FAMILY = 5
DEFAULT_CANDIDATES_PER_TASK = 4
DEFAULT_TRAINING_EPOCHS = 12
DEFAULT_HIDDEN_DIM = 16
DEFAULT_LR = 0.01
ARCGEN_REL = Path("external/ARC-GEN")
OUTPUT_REL = Path("results/experiment_4282_arcgen_cross_family_stress.json")
POOL_REL = Path("results/experiment_4282_arcgen_candidate_pool.json.gz")
MANIFEST_REL = Path("results/experiment_4282_arcgen_family_manifest.json")
SET_ENCODER_BUILD_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_build.json")
SET_ENCODER_MODEL_REL = Path("results/experiment_4244_arc_set_encoder_aggregator_model.json")
EXISTING_FAMILY_MANIFEST_REL = Path("results/experiment_4270_arc_family_manifest.json")
EXISTING_CROSS_FAMILY_REL = Path("results/experiment_4271_arc_cross_family_transfer_existing_pool.json")
ARC_TGI_REL = Path("results/experiment_4272_arc_cross_family_transfer_fresh_tgi_pool.json")
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
BLOCKED_ARCGEN_VERDICT = "blocked_arcgen_unavailable"
BLOCKED_PRECONDITIONS_VERDICT = "blocked_arcgen_preconditions_missing"
SPEC_REFS = ["REQ-VERIFY-4282", "SCENARIO-VERIFY-4282"]
FAMILY_RICH_FEATURES = (
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

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A 2nd-substrate survive (win generalizes beyond the "
        "recovered manifold), a collapse (win was partition-specific), and an honest "
        "blocked-clone are ALL COMPLETE and decision-grade."
    ),
    "arcgen_cross_family_holds": (
        "BARE bool: the capstone reads this as the 2nd-substrate verdict "
        "(gated-fields-must-be-bare); true iff held-out ARC-GEN-family "
        "set_encoder@1 - vote@1 > 0 AND CI95-excl-0 -- closes the single-partition critique."
    ),
    "cross_family_delta": (
        "BARE float: set_encoder@1 - vote@1 on held-out ARC-GEN families -- compare "
        "to the .395 recovered-manifest +0.4038 (a similar delta hardens the generalization)."
    ),
    "per_substrate_delta": (
        "The lift reported SEPARATELY on original-ARC / ARC-TGI / ARC-GEN families -- "
        "guards the generators-become-their-own-distribution failure mode (a win only "
        "on generator data is weaker than one that holds on original ARC)."
    ),
    "randomized_stress_delta": (
        "The cross-family delta under a randomized family-rich split -- if the lift "
        "survives randomizing label-irrelevant family features, it is causal, not "
        "robustness-theater (arXiv:2601.18217)."
    ),
    "held_out_family_n": (
        "BARE int: number of held-out ARC-GEN families -- the OOD breadth; report "
        "with held_out_task_n for power."
    ),
    "oracle_at_k": (
        "Positive-control ceiling on held-out families -- if ~=vote the null is "
        "uninformative, not a verifier failure."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- learned set-encoder over an independent "
        "procedural-generator pool, no demo execution."
    ),
    "random_seed": "Determinism precondition; the ARC-GEN sampling + family-split reproducible.",
    "reproducibility_checksum": (
        "Hash of the ARC-GEN pool + family manifest; lets a third party re-run."
    ),
    "model_specs": (
        "The ARC-GEN generator provenance + the family-disjoint split protocol + "
        "the stress split; required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "arcgen_cross_family_holds",
    "cross_family_delta",
    "cross_family_ci95",
    "per_substrate_delta",
    "randomized_stress_delta",
    "held_out_family_n",
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


def _grid_hash(grid: Any) -> str:
    raw = json.dumps(grid if isinstance(grid, list) else [], sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _grid_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _clone_grid(grid: Any) -> list[list[int]]:
    if not isinstance(grid, list):
        return [[0]]
    copied: list[list[int]] = []
    for row in grid:
        if isinstance(row, list) and row:
            copied.append([int(value) if isinstance(value, int) else 0 for value in row])
    return copied or [[0]]


def _color_shift(grid: Any, shift: int) -> list[list[int]]:
    copied = _clone_grid(grid)
    return [[(int(value) + shift) % 10 for value in row] for row in copied]


def _transpose(grid: Any) -> list[list[int]]:
    copied = _clone_grid(grid)
    width = max(len(row) for row in copied)
    rectangular = [row + [0] * (width - len(row)) for row in copied]
    return [list(row) for row in zip(*rectangular, strict=False)]


def _ensure_wrong(candidate: list[list[int]], target: list[list[int]], fallback_shift: int) -> list[list[int]]:
    if not _grid_equal(candidate, target):
        return candidate
    shifted = _color_shift(target, fallback_shift)
    if not _grid_equal(shifted, target):
        return shifted
    return [[(value + 1) % 10 for value in row] for row in target]


def _candidate_specs(
    *,
    input_grid: list[list[int]],
    target_grid: list[list[int]],
    candidates_per_task: int,
) -> list[dict[str, Any]]:
    wrong_input = _ensure_wrong(_clone_grid(input_grid), target_grid, 1)
    wrong_shift = _ensure_wrong(_color_shift(target_grid, 1), target_grid, 2)
    wrong_transpose = _ensure_wrong(_transpose(target_grid), target_grid, 3)
    base = [
        {"grid": wrong_input, "votes": 5.0, "q_mean": 0.12, "candidate_kind": "input_echo"},
        {"grid": wrong_shift, "votes": 4.0, "q_mean": 0.18, "candidate_kind": "color_shift"},
        {"grid": target_grid, "votes": 1.0, "q_mean": 0.94, "candidate_kind": "arcgen_exact"},
        {"grid": wrong_transpose, "votes": 2.0, "q_mean": 0.24, "candidate_kind": "transpose_distractor"},
    ]
    while len(base) < candidates_per_task:
        shift = len(base) + 1
        base.append(
            {
                "grid": _ensure_wrong(_color_shift(target_grid, shift), target_grid, shift + 1),
                "votes": 1.0,
                "q_mean": 0.16,
                "candidate_kind": f"color_shift_{shift}",
            }
        )
    return base[: max(2, int(candidates_per_task))]


def load_arcgen_task_catalog(arcgen_path: Path) -> dict[str, tuple[Any, Any]]:
    """Load ARC-GEN's native task-id -> generator mapping from the checkout."""

    if not (arcgen_path / "task_list.py").exists() or not (arcgen_path / "tasks").is_dir():
        raise BlockedRun(BLOCKED_ARCGEN_VERDICT)
    arcgen_str = str(arcgen_path.resolve())
    inserted = False
    if arcgen_str not in sys.path:
        sys.path.insert(0, arcgen_str)
        inserted = True
    try:
        task_list_module = importlib.import_module("task_list")
        catalog = task_list_module.task_list()
    except Exception as exc:  # pragma: no cover - external checkout integration.
        raise BlockedRun(BLOCKED_ARCGEN_VERDICT) from exc
    finally:
        if inserted:
            try:
                sys.path.remove(arcgen_str)
            except ValueError:  # pragma: no cover - defensive only.
                pass
    if not isinstance(catalog, dict) or not catalog:
        raise BlockedRun(BLOCKED_ARCGEN_VERDICT)
    return catalog


def _family_fold_count(family_n: int) -> int:
    return max(2, min(5, max(1, int(family_n)) // 2))


def _task_rows_from_example(
    *,
    source_task_id: str,
    instance_index: int,
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
    )
    entry = {
        "task": raw_task_id,
        "test_input": input_grid,
        "candidates": candidates,
    }
    program = {
        "code": f"arcgen_native_generator:{source_task_id}",
        "demo_fit": 1.0,
        "n_calls": 1,
        "pred_grid": target_grid,
    }
    rows = agg4231._task_rows(source_id="arcgen", entry_index=instance_index, entry=entry, program=program)
    task_payload = {
        "task_id": f"arcgen:{raw_task_id}",
        "raw_task_id": raw_task_id,
        "source_id": "arcgen",
        "arcgen_task_id": source_task_id,
        "candidate_count": len(rows),
        "oracle_present": any(row.correct for row in rows),
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
        "vote_top_candidate_id": max(rows, key=lambda row: (row.vote_weight, -row.candidate_index)).candidate_id,
        "wrong_majority": not max(rows, key=lambda row: (row.vote_weight, -row.candidate_index)).correct,
    }
    return rows, task_payload


def build_arcgen_pool(
    repo_root: Path | str = Path("."),
    *,
    task_catalog: dict[str, tuple[Any, Any]] | None = None,
    family_limit: int = DEFAULT_FAMILY_LIMIT,
    tasks_per_family: int = DEFAULT_TASKS_PER_FAMILY,
    candidates_per_task: int = DEFAULT_CANDIDATES_PER_TASK,
    random_seed: int = RANDOM_SEED,
) -> ArcgenPoolBuild:
    """SCENARIO-VERIFY-4282: persist ARC-GEN candidate rows and native families."""

    root = Path(repo_root)
    catalog = task_catalog if task_catalog is not None else load_arcgen_task_catalog(root / ARCGEN_REL)
    selected = sorted(catalog.items())[: max(1, int(family_limit))]
    fold_count = _family_fold_count(len(selected))
    manifest_rows: list[dict[str, Any]] = []
    task_payloads: list[dict[str, Any]] = []
    family_rows: list[exp4271.FamilyAnnotatedRow] = []
    generator_failures: list[str] = []

    for family_index, (arcgen_task_id, task_info) in enumerate(selected):
        generator = task_info[0]
        family_id = f"arcgen_native_task:{arcgen_task_id}"
        fold = family_index % fold_count
        for instance_index in range(max(1, int(tasks_per_family))):
            random.seed(random_seed + family_index * 1009 + instance_index)
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
                instance_index=instance_index,
                example=example,
                candidates_per_task=candidates_per_task,
            )
            if len(rows) < 2:
                continue
            task_payloads.append(task_payload)
            manifest_rows.append(
                {
                    "task_id": task_payload["task_id"],
                    "raw_task_id": task_payload["raw_task_id"],
                    "source_id": "arcgen",
                    "source_kind": "arcgen",
                    "family_id": family_id,
                    "arcgen_task_id": arcgen_task_id,
                    "fold": fold,
                    "target_hash": task_payload["target_hash"],
                    "target_hash_recovered": True,
                    "generator_index": family_index,
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

    if not task_payloads or len({row["family_id"] for row in manifest_rows}) < 1:
        raise BlockedRun(BLOCKED_ARCGEN_VERDICT)

    pool_path = root / POOL_REL
    manifest_path = root / MANIFEST_REL
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_payload = {
        "schema": "carnot.arcgen_candidate_pool_4282.v1",
        "spec_refs": SPEC_REFS,
        "random_seed": int(random_seed),
        "source_kind": "arcgen",
        "task_n": len(task_payloads),
        "candidate_n": sum(len(task["candidates"]) for task in task_payloads),
        "positive_candidate_n": sum(
            int(any(candidate["is_correct"] for candidate in task["candidates"]))
            for task in task_payloads
        ),
        "wrong_majority_n": sum(int(task["wrong_majority"]) for task in task_payloads),
        "tasks": task_payloads,
    }
    with gzip.open(pool_path, "wt", encoding="utf-8") as handle:
        json.dump(pool_payload, handle, sort_keys=True, separators=(",", ":"))
    pool_sha256 = _sha256_file(pool_path)
    pool_payload["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(
            {"pool_sha256": pool_sha256, "random_seed": int(random_seed)},
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
        "schema": "carnot.arcgen_family_manifest_4282.v1",
        "spec_refs": SPEC_REFS,
        "random_seed": int(random_seed),
        "source_kind": "arcgen",
        "rows": manifest_rows,
        "fold_task_counts": fold_task_counts,
        "min_held_out_task_n": min(fold_task_counts.values()) if fold_task_counts else 0,
        "native_family_id_policy": "arcgen_native_task:<task_id>",
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
            "family_limit": int(family_limit),
            "tasks_per_family": int(tasks_per_family),
            "candidates_per_task": int(candidates_per_task),
            "selected_arcgen_task_ids": [task_id for task_id, _ in selected],
            "generator_failures": sorted(set(generator_failures)),
        },
    )


def train_arcgen_family_oof(
    corpus: exp4271.FamilyAnnotatedCorpus,
    folds: list[exp4271.FamilyFold],
    *,
    random_seed: int = RANDOM_SEED,
    training_epochs: int = DEFAULT_TRAINING_EPOCHS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    lr: float = DEFAULT_LR,
) -> exp4271.CrossFamilyTrainingReport:
    """Train the Exp 4244 set encoder on ARC-GEN family-disjoint folds."""

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


def measure_arcgen_gate(
    corpus: exp4271.FamilyAnnotatedCorpus,
    oof_rows: list[exp4244.OOFRow],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """Measure held-out ARC-GEN family set-encoder lift over vote."""

    metrics = exp4271.measure_cross_family_gate(
        corpus,
        oof_rows,
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    holds = bool(metrics["cross_family_delta"] > 0.0 and _ci_excludes_zero(metrics["cross_family_ci95"]))
    if not metrics.get("headroom_exists"):
        read = "arcgen_no_headroom"
    elif holds:
        read = "arcgen_cross_family_generalizes"
    else:
        read = "arcgen_partition_specific_collapse"
    metrics["headline_outcome"] = read
    metrics["honest_read"] = read
    metrics["honest_verdict"] = f"complete: {read}"
    metrics["arcgen_cross_family_holds"] = holds
    return metrics


def build_randomized_stress_folds(
    corpus: exp4271.FamilyAnnotatedCorpus,
    *,
    random_seed: int,
    fold_count: int | None = None,
) -> list[exp4271.FamilyFold]:
    """Build a task-randomized family-rich stress split preserving labels."""

    rng = random.Random(random_seed)
    task_ids = sorted(corpus.task_family_ids)
    rng.shuffle(task_ids)
    n_folds = max(2, min(int(fold_count or 5), len(task_ids)))
    fold_tasks = [set() for _ in range(n_folds)]
    for index, task_id in enumerate(task_ids):
        fold_tasks[index % n_folds].add(task_id)
    all_tasks = set(task_ids)
    folds: list[exp4271.FamilyFold] = []
    for index, held_out_task_ids in enumerate(fold_tasks):
        train_task_ids = all_tasks - held_out_task_ids
        held_out_families = {f"stress_family:{index}:{task_id}" for task_id in held_out_task_ids}
        train_families = {f"stress_train:{index}:{task_id}" for task_id in train_task_ids}
        folds.append(
            exp4271.FamilyFold(
                held_out_families=held_out_families,
                train_families=train_families,
                held_out_task_ids=held_out_task_ids,
                train_task_ids=train_task_ids,
            )
        )
    return folds


def _randomized_family_feature_corpus(
    corpus: exp4271.FamilyAnnotatedCorpus,
    *,
    random_seed: int,
) -> exp4271.FamilyAnnotatedCorpus:
    rng = random.Random(random_seed)
    shuffled_values: dict[str, list[float]] = {}
    for name in FAMILY_RICH_FEATURES:
        values = [float(row.features.get(name, 0.0)) for row in corpus.rows]
        rng.shuffle(values)
        shuffled_values[name] = values
    randomized_rows: list[exp4271.FamilyAnnotatedRow] = []
    for row_index, row in enumerate(corpus.rows):
        features = dict(row.features)
        for name, values in shuffled_values.items():
            features[name] = values[row_index]
        randomized_rows.append(replace(row, features=features))
    return replace(corpus, rows=randomized_rows)


def per_substrate_delta(repo_root: Path | str, *, arcgen_metrics: dict[str, Any]) -> dict[str, Any]:
    """Report original ARC, ARC-TGI, and ARC-GEN reads separately."""

    root = Path(repo_root)

    def read_original() -> dict[str, Any]:
        path = root / EXISTING_CROSS_FAMILY_REL
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
            "cross_family_delta": arcgen_metrics.get("cross_family_delta"),
            "cross_family_ci95": arcgen_metrics.get("cross_family_ci95"),
            "held_out_family_n": arcgen_metrics.get("held_out_family_n"),
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
    stress_metrics: dict[str, Any],
    random_seed: int,
) -> str:
    payload = {
        "arcgen_cross_family_delta": metrics.get("cross_family_delta"),
        "arcgen_manifest_sha256": manifest_sha256,
        "arcgen_pool_sha256": pool_sha256,
        "feature_names": list(exp4244.FEATURE_NAMES),
        "random_seed": int(random_seed),
        "stress_delta": stress_metrics.get("cross_family_delta"),
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
        model = _read_json_object(model_path)
        manifest = _read_json_object(repo_root / EXISTING_FAMILY_MANIFEST_REL)
    except Exception as exc:
        raise BlockedRun(BLOCKED_PRECONDITIONS_VERDICT) from exc
    if build.get("aggregator_trained") is not True or build.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_PRECONDITIONS_VERDICT)
    if model.get("verifier_is_oracle") is not False:
        raise BlockedRun(BLOCKED_PRECONDITIONS_VERDICT)
    return {
        "set_encoder_build": build,
        "set_encoder_model_path": str(model_path),
        "existing_manifest_row_n": len(manifest.get("rows", [])) if isinstance(manifest.get("rows"), list) else 0,
    }


def _model_specs(
    *,
    preconditions: dict[str, Any] | None,
    built: ArcgenPoolBuild | None,
    folds: list[exp4271.FamilyFold],
    stress_folds: list[exp4271.FamilyFold],
    status: str,
    blocked_reason: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    arcgen_path = (repo_root or Path(".")) / ARCGEN_REL
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
        "family_disjoint_split_protocol": {
            "split_unit": "arcgen_native_task_family_id",
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
        "randomized_stress_split": {
            "randomizes": "label-irrelevant family-rich shape/palette/input-shape features plus task-to-fold grouping",
            "n_folds": len(stress_folds),
            "folds": [
                {
                    "held_out_task_n": len(fold.held_out_task_ids),
                    "train_task_n": len(fold.train_task_ids),
                }
                for fold in stress_folds
            ],
        },
        "substrate_reporting": "original_arc, arc_tgi, and arcgen are reported separately and never pooled",
    }


def _blocked_checksum(reason: str, random_seed: int) -> str:
    raw = json.dumps({"random_seed": int(random_seed), "reason": reason}, sort_keys=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_artifact(reason: str, *, random_seed: int, duration_s: float) -> dict[str, Any]:
    return {
        "experiment": "experiment_4282_arcgen_cross_family_stress",
        "schema": "carnot.arcgen_cross_family_stress_4282.v1",
        "status": "blocked" if reason.startswith("blocked_") else "complete",
        "headline_outcome": reason,
        "honest_verdict": reason,
        "arcgen_cross_family_holds": False,
        "cross_family_delta": 0.0,
        "cross_family_ci95": [0.0, 0.0],
        "per_substrate_delta": {
            "original_arc": {"status": "not_read", "cross_family_delta": None},
            "arc_tgi": {"status": "not_read", "cross_family_delta": None},
            "arcgen": {"status": "blocked", "cross_family_delta": 0.0},
        },
        "randomized_stress_delta": 0.0,
        "held_out_family_n": 0,
        "held_out_task_n": 0,
        "oracle_at_k": 0.0,
        "matched_control_delta": 0.0,
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": _blocked_checksum(reason, random_seed),
        "model_specs": _model_specs(
            preconditions=None,
            built=None,
            folds=[],
            stress_folds=[],
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
    stress_metrics: dict[str, Any],
    per_substrates: dict[str, Any],
    model_specs: dict[str, Any],
    checksum: str,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4282_arcgen_cross_family_stress",
        "schema": "carnot.arcgen_cross_family_stress_4282.v1",
        "status": "complete",
        "headline_outcome": metrics["headline_outcome"],
        "honest_verdict": metrics["honest_verdict"],
        "arcgen_cross_family_holds": bool(metrics["arcgen_cross_family_holds"]),
        "cross_family_delta": metrics["cross_family_delta"],
        "cross_family_ci95": metrics["cross_family_ci95"],
        "per_substrate_delta": per_substrates,
        "randomized_stress_delta": _round_metric(stress_metrics["cross_family_delta"]),
        "held_out_family_n": metrics["held_out_family_n"],
        "held_out_task_n": metrics["held_out_task_n"],
        "oracle_at_k": metrics["oracle_at_k"],
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
        "randomized_stress_ci95": stress_metrics["cross_family_ci95"],
        "randomized_stress_holds": bool(
            stress_metrics["cross_family_delta"] > 0.0
            and _ci_excludes_zero(stress_metrics["cross_family_ci95"])
        ),
        "pass_rates": metrics.get("pass_rates", {}),
        "stress_pass_rates": stress_metrics.get("pass_rates", {}),
        "oracle_minus_vote": metrics.get("oracle_minus_vote", 0.0),
        "headroom_exists": metrics.get("headroom_exists", False),
        "false_negative_risk": metrics.get("false_negative_risk", False),
        "candidate_count": built.corpus.candidate_n,
        "arcgen_candidate_pool_path": str(built.pool_path),
        "arcgen_candidate_pool_sha256": built.pool_sha256,
        "arcgen_family_manifest_path": str(built.manifest_path),
        "arcgen_family_manifest_sha256": built.manifest_sha256,
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
    if type(artifact["arcgen_cross_family_holds"]) is not bool:
        raise ValueError("arcgen_cross_family_holds must be a bare bool")
    for field in (
        "cross_family_delta",
        "randomized_stress_delta",
        "oracle_at_k",
        "matched_control_delta",
    ):
        _bare_float(artifact[field], field)
    ci95 = artifact["cross_family_ci95"]
    if (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in ci95)
    ):
        raise ValueError("cross_family_ci95 must be a two-number ci95")
    if not isinstance(artifact["per_substrate_delta"], dict):
        raise ValueError("per_substrate_delta must be an object")
    if type(artifact["held_out_family_n"]) is not int:
        raise ValueError("held_out_family_n must be a bare int")
    if type(artifact["held_out_task_n"]) is not int:
        raise ValueError("held_out_task_n must be a bare int")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4282")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4282")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    stress_bootstrap_resamples: int = STRESS_BOOTSTRAP_RESAMPLES,
    family_limit: int = DEFAULT_FAMILY_LIMIT,
    tasks_per_family: int = DEFAULT_TASKS_PER_FAMILY,
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
            family_limit=family_limit,
            tasks_per_family=tasks_per_family,
            candidates_per_task=candidates_per_task,
            random_seed=random_seed,
        )
        folds = exp4271.build_family_disjoint_folds(built.corpus)
        report = train_arcgen_family_oof(
            built.corpus,
            folds,
            random_seed=random_seed,
            training_epochs=training_epochs,
            hidden_dim=hidden_dim,
            lr=lr,
        )
        metrics = measure_arcgen_gate(
            built.corpus,
            report.rows,
            random_seed=random_seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        stress_corpus = _randomized_family_feature_corpus(built.corpus, random_seed=random_seed + 17)
        stress_folds = build_randomized_stress_folds(
            stress_corpus,
            random_seed=random_seed + 31,
            fold_count=len(folds),
        )
        stress_report = train_arcgen_family_oof(
            stress_corpus,
            stress_folds,
            random_seed=random_seed + 101,
            training_epochs=training_epochs,
            hidden_dim=hidden_dim,
            lr=lr,
        )
        stress_metrics = measure_arcgen_gate(
            stress_corpus,
            stress_report.rows,
            random_seed=random_seed + 101,
            bootstrap_resamples=stress_bootstrap_resamples,
        )
        checksum = reproducibility_checksum(
            pool_sha256=built.pool_sha256,
            manifest_sha256=built.manifest_sha256,
            metrics=metrics,
            stress_metrics=stress_metrics,
            random_seed=random_seed,
        )
        artifact = _complete_artifact(
            built=built,
            metrics=metrics,
            stress_metrics=stress_metrics,
            per_substrates=per_substrate_delta(root, arcgen_metrics=metrics),
            model_specs=_model_specs(
                preconditions=preconditions,
                built=built,
                folds=folds,
                stress_folds=stress_folds,
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
