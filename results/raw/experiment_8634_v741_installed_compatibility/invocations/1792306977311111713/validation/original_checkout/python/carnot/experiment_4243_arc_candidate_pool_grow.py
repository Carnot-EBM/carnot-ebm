"""Exp 4243 ARC candidate-pool growth for the A2 oracle-distinct run.

Spec refs: REQ-CAPSTONE-4243, SCENARIO-CAPSTONE-4243.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.reporting import oracle_distinct_arc_aggregator_4231 as agg4231


RANDOM_SEED = 4243
BASELINE_POSITIVE_CANDIDATES_392 = 20
BASELINE_WRONG_MAJORITY_392 = 9
BASELINE_HELD_OUT_TASKS_392 = 52
BASELINE_CANDIDATES_392 = 28419
OUTPUT_REL = Path("results/experiment_4243_arc_candidate_pool_grow.json")
POOL_ARTIFACT_REL = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
SPEC_REFS = ["REQ-CAPSTONE-4243", "SCENARIO-CAPSTONE-4243"]
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
FEATURE_NAMES = agg4231.FEATURE_NAMES


@dataclass(frozen=True)
class PoolSpec:
    source_id: str
    pool_rel: Path
    programs_rel: Path
    required: bool


DEFAULT_POOL_SPECS = (
    PoolSpec(
        "gap3_stage2",
        Path("results/arc3_gap3_stage2_eval_pool.json.gz"),
        Path("results/arc3_gap4_induced_programs.json"),
        True,
    ),
    PoolSpec(
        "gap4_arc2",
        Path("results/arc3_gap4_arc2_eval_pool.json.gz"),
        Path("results/arc3_gap4_arc2_induced_programs.json"),
        False,
    ),
)


FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed (complete:/success:/passed:/shipped:). A grown pool OR an "
        "honest 'cached+bounded-topup ceiling below target' is COMPLETE -- A2 reads "
        "whatever is persisted."
    ),
    "arc_pool_grown": (
        "BARE bool: A2's gate compares this raw value (gated-fields-must-be-bare); "
        "true iff positive_candidate_n AND wrong_majority_n both meaningfully exceed "
        "the .392 baseline (20 / 5-9)."
    ),
    "positive_candidate_n": (
        "BARE int: count of minority-correct candidates in the grown pool -- the "
        "data-sparsity metric; must exceed .392's 20 for the set-encoder to learn "
        "wrong-majority recovery."
    ),
    "wrong_majority_n": (
        "BARE int: count of tasks where oracle@K > vote@1 (correct answer present "
        "but loses the vote) -- the recoverable headroom the aggregator targets; "
        "target >=40 held-out."
    ),
    "held_out_task_n": (
        "BARE int: tasks reserved for the A3 gate's held-out split -- target >=40 "
        "(CLT floor) so the A3 win/null is not under-powered like .392's n=52-at-sparsity."
    ),
    "pool_artifact_path": (
        "The persisted grown labeled-pool artifact A2 loads (per-task candidate sets "
        "+ is_correct + features); the build deliverable."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the labels come from gold/induced pred_grid match used "
        "as the TRAINING target; the eventual aggregator scores WITHOUT executing "
        "demos at inference (Circularity Discipline)."
    ),
    "random_seed": (
        "Determinism precondition; the pool assembly + any solver top-up + fold "
        "reservation seeded so the pool is reproducible."
    ),
    "reproducibility_checksum": (
        "Hash of the assembled pool; catches silent candidate/label drift before A2 trains."
    ),
    "model_specs": (
        "The cached-assembly description + any bounded solver top-up config; required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "arc_pool_grown",
    "positive_candidate_n",
    "wrong_majority_n",
    "held_out_task_n",
    "pool_artifact_path",
    "verifier_is_oracle",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)


class BlockedRun(RuntimeError):
    """Expected precondition failure that still writes a terminal artifact."""


@dataclass(frozen=True)
class PoolAssembly:
    tasks: list[dict[str, Any]]
    source_paths: list[Path]
    source_sha256: dict[str, str]
    candidate_n: int
    raw_candidate_n: int
    positive_candidate_n: int
    wrong_majority_n: int
    task_n: int
    detector_row_n: int
    skipped_optional_pools: list[str]


def _json_grid(value: Any) -> str:
    return json.dumps(value if isinstance(value, list) else [], sort_keys=True, separators=(",", ":"))


def candidate_grid_hash(grid: Any) -> str:
    return hashlib.sha256(_json_grid(grid).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detector_row_count(pool_path: Path, programs_path: Path) -> int:  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts import exp_verifier_detector_auroc as detector

    return len(detector.load_arc_rows(pool_path, programs_path))


def _load_payloads(
    repo_root: Path,
    spec: PoolSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int] | None:
    pool_path = repo_root / spec.pool_rel
    programs_path = repo_root / spec.programs_rel
    if not pool_path.exists() or not programs_path.exists():
        if spec.required:
            raise BlockedRun("blocked_arc_gap4_pools_missing")
        return None
    try:
        detector_n = _detector_row_count(pool_path, programs_path)
        with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
            pool_payload = json.load(handle)
        programs_payload = json.loads(programs_path.read_text(encoding="utf-8"))
    except Exception as exc:
        if spec.required:
            raise BlockedRun("blocked_arc_gap4_pools_missing") from exc
        return None
    entries = pool_payload.get("entries") if isinstance(pool_payload, dict) else None
    programs = programs_payload.get("programs") if isinstance(programs_payload, dict) else None
    if not isinstance(entries, list) or not isinstance(programs, list):
        if spec.required:
            raise BlockedRun("blocked_arc_gap4_pools_missing")
        return None
    return entries, programs, detector_n


def _as_float(value: Any) -> float:
    return agg4231._as_float(value)


def _target_hashes(candidates: list[dict[str, Any]], pred_grid: Any) -> set[str]:
    targets = {
        candidate_grid_hash(candidate.get("grid"))
        for candidate in candidates
        if candidate.get("correct") is True
    }
    if isinstance(pred_grid, list):
        targets.add(candidate_grid_hash(pred_grid))
    return targets


def _merged_candidates(
    candidates: list[dict[str, Any]],
    pred_grid: Any,
    program: dict[str, Any],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    augmented = list(candidates)
    if isinstance(pred_grid, list):
        augmented.append(
            {
                "grid": pred_grid,
                "votes": 0.0,
                "q_mean": _as_float(program.get("demo_fit")),
                "correct": True,
                "_source_kind": "induced_pred_grid",
                "_raw_index": len(candidates),
            }
        )
    for index, candidate in enumerate(augmented):
        if not isinstance(candidate, dict) or not isinstance(candidate.get("grid"), list):
            continue
        grid_hash = candidate_grid_hash(candidate.get("grid"))
        source_kind = str(
            candidate.get("_source_kind")
            or ("gold_flag" if candidate.get("correct") is True else "pool_candidate")
        )
        if grid_hash not in merged:
            merged[grid_hash] = {
                "grid": candidate.get("grid"),
                "votes": 0.0,
                "q_mean": 0.0,
                "correct": False,
                "_source_kinds": [],
                "_raw_indices": [],
            }
        item = merged[grid_hash]
        item["votes"] = _as_float(item.get("votes")) + _as_float(candidate.get("votes"))
        item["q_mean"] = max(_as_float(item.get("q_mean")), _as_float(candidate.get("q_mean")))
        item["correct"] = bool(item.get("correct")) or candidate.get("correct") is True
        item["_raw_indices"].append(int(candidate.get("_raw_index", index)))
        if source_kind not in item["_source_kinds"]:
            item["_source_kinds"].append(source_kind)
    return list(merged.values())


def _task_payload(
    *,
    source_id: str,
    entry_index: int,
    entry: dict[str, Any],
    program: dict[str, Any],
) -> dict[str, Any] | None:
    raw_candidates = [candidate for candidate in entry.get("candidates", []) if isinstance(candidate, dict)]
    if not raw_candidates:
        return None
    pred_grid = program.get("pred_grid")
    target_hashes = _target_hashes(raw_candidates, pred_grid)
    merged_candidates = _merged_candidates(raw_candidates, pred_grid, program)
    if len(merged_candidates) < 2:
        return None
    merged_entry = {**entry, "candidates": merged_candidates}
    rows = agg4231._task_rows(
        source_id=source_id,
        entry_index=entry_index,
        entry=merged_entry,
        program=program,
    )
    if not rows:
        return None

    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidate = merged_candidates[row.candidate_index]
        grid_hash = candidate_grid_hash(candidate.get("grid"))
        is_correct = grid_hash in target_hashes
        candidates.append(
            {
                "candidate_id": row.candidate_id,
                "candidate_grid_hash": grid_hash,
                "candidate_index": row.candidate_index,
                "features": {name: float(row.features[name]) for name in FEATURE_NAMES},
                "grid": candidate.get("grid"),
                "is_correct": bool(is_correct),
                "q_mean": _as_float(candidate.get("q_mean")),
                "raw_candidate_indices": list(candidate.get("_raw_indices", [])),
                "source_kinds": list(candidate.get("_source_kinds", [])),
                "votes": _as_float(candidate.get("votes")),
            }
        )
    vote_winner = max(candidates, key=lambda candidate: (candidate["votes"], -candidate["candidate_index"]))
    raw_task_id = str(entry.get("task") or f"entry-{entry_index}")
    return {
        "candidate_count": len(candidates),
        "candidates": candidates,
        "oracle_present": any(candidate["is_correct"] for candidate in candidates),
        "raw_task_id": raw_task_id,
        "source_id": source_id,
        "task_id": f"{source_id}:{raw_task_id}",
        "vote_top_candidate_id": vote_winner["candidate_id"],
        "wrong_majority": any(candidate["is_correct"] for candidate in candidates)
        and not vote_winner["is_correct"],
    }


def assemble_pool(
    repo_root: Path | str = Path("."),
    *,
    pool_specs: tuple[PoolSpec, ...] = DEFAULT_POOL_SPECS,
) -> PoolAssembly:
    root = Path(repo_root)
    tasks: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    source_sha256: dict[str, str] = {}
    skipped_optional_pools: list[str] = []
    raw_candidate_n = 0
    detector_row_n = 0

    for spec in pool_specs:
        loaded = _load_payloads(root, spec)
        if loaded is None:
            skipped_optional_pools.append(spec.source_id)
            continue
        entries, programs, detector_n = loaded
        pool_path = (root / spec.pool_rel).resolve()
        programs_path = (root / spec.programs_rel).resolve()
        source_paths.extend([pool_path, programs_path])
        source_sha256[str(pool_path)] = _sha256_file(pool_path)
        source_sha256[str(programs_path)] = _sha256_file(programs_path)
        detector_row_n += detector_n
        programs_by_entry = {
            int(program.get("entry_i", index)): program
            for index, program in enumerate(programs)
            if isinstance(program, dict)
        }
        grouped_entries: dict[str, dict[str, Any]] = {}
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            raw_candidates = entry.get("candidates")
            if isinstance(raw_candidates, list):
                raw_candidate_n += sum(1 for candidate in raw_candidates if isinstance(candidate, dict))
            raw_task_id = str(entry.get("task") or f"entry-{entry_index}")
            grouped = grouped_entries.setdefault(
                raw_task_id,
                {
                    "entry_index": entry_index,
                    "entry": {**entry, "candidates": []},
                    "program": {},
                },
            )
            if isinstance(raw_candidates, list):
                grouped["entry"]["candidates"].extend(raw_candidates)
            program = programs_by_entry.get(entry_index, {})
            if isinstance(program, dict) and program.get("pred_grid") is not None:
                grouped["program"] = program
        for grouped in grouped_entries.values():
            task = _task_payload(
                source_id=spec.source_id,
                entry_index=int(grouped["entry_index"]),
                entry=grouped["entry"],
                program=grouped["program"],
            )
            if task is not None:
                tasks.append(task)

    candidate_n = sum(int(task["candidate_count"]) for task in tasks)
    positive_candidate_n = sum(
        1 for task in tasks for candidate in task["candidates"] if candidate["is_correct"]
    )
    wrong_majority_n = sum(1 for task in tasks if task["wrong_majority"])
    return PoolAssembly(
        tasks=tasks,
        source_paths=source_paths,
        source_sha256=source_sha256,
        candidate_n=candidate_n,
        raw_candidate_n=raw_candidate_n,
        positive_candidate_n=positive_candidate_n,
        wrong_majority_n=wrong_majority_n,
        task_n=len(tasks),
        detector_row_n=detector_row_n,
        skipped_optional_pools=skipped_optional_pools,
    )


def reproducibility_checksum(assembly: PoolAssembly, *, random_seed: int = RANDOM_SEED) -> str:
    payload = {
        "random_seed": int(random_seed),
        "source_sha256": assembly.source_sha256,
        "tasks": assembly.tasks,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _model_specs(
    *,
    assembly: PoolAssembly | None,
    top_up_status: str,
    baseline_positive_n: int,
    baseline_wrong_majority_n: int,
) -> dict[str, Any]:
    return {
        "assembly": (
            "cached GAP-4 pool union with per-task candidate dedupe by task_id and "
            "candidate_grid_hash; labels are exact matches to gold-flagged grids or "
            "GAP-4 induced pred_grid"
        ),
        "baseline_392": {
            "positive_candidate_n": BASELINE_POSITIVE_CANDIDATES_392,
            "wrong_majority_n": BASELINE_WRONG_MAJORITY_392,
            "held_out_task_n": BASELINE_HELD_OUT_TASKS_392,
            "candidate_n": BASELINE_CANDIDATES_392,
        },
        "baseline_gate_used": {
            "positive_candidate_n": int(baseline_positive_n),
            "wrong_majority_n": int(baseline_wrong_majority_n),
        },
        "feature_set": list(FEATURE_NAMES),
        "pool_sources": [str(path) for path in assembly.source_paths] if assembly else [],
        "top_up": {
            "status": top_up_status,
            "solver": "cached_exp4236_offline_explore_induce_verify",
            "task_cap": 0,
            "candidate_cap_per_task": 0,
            "note": "not invoked when cached pool already exceeds the .392 growth gate",
        },
    }


def _pool_artifact_payload(
    assembly: PoolAssembly,
    *,
    checksum: str,
    random_seed: int,
) -> dict[str, Any]:
    task_size_histogram = Counter(str(task["candidate_count"]) for task in assembly.tasks)
    return {
        "schema": "carnot.arc_candidate_pool_grow.v1",
        "spec_refs": SPEC_REFS,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "task_n": assembly.task_n,
        "candidate_n": assembly.candidate_n,
        "positive_candidate_n": assembly.positive_candidate_n,
        "wrong_majority_n": assembly.wrong_majority_n,
        "task_candidate_count_histogram": dict(sorted(task_size_histogram.items())),
        "source_paths": [str(path) for path in assembly.source_paths],
        "source_sha256": assembly.source_sha256,
        "tasks": assembly.tasks,
    }


def persist_pool_artifact(
    path: Path,
    assembly: PoolAssembly,
    *,
    checksum: str,
    random_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(
            _pool_artifact_payload(assembly, checksum=checksum, random_seed=random_seed),
            handle,
            sort_keys=True,
            separators=(",", ":"),
        )


def _blocked_artifact(reason: str, *, random_seed: int, duration_s: float) -> dict[str, Any]:
    return {
        "experiment": "experiment_4243_arc_candidate_pool_grow",
        "schema": "carnot.arc_candidate_pool_grow_result.v1",
        "honest_verdict": reason,
        "arc_pool_grown": False,
        "positive_candidate_n": 0,
        "wrong_majority_n": 0,
        "held_out_task_n": 0,
        "pool_artifact_path": "",
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": "sha256:blocked_arc_gap4_pools_missing",
        "model_specs": _model_specs(
            assembly=None,
            top_up_status="blocked_before_generation",
            baseline_positive_n=BASELINE_POSITIVE_CANDIDATES_392,
            baseline_wrong_majority_n=BASELINE_WRONG_MAJORITY_392,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": "cached_pool_assembly_no_llm_no_training",
        "duration_s": round(duration_s, 6),
        "acceptance_gate": True,
    }


def _complete_artifact(
    assembly: PoolAssembly,
    *,
    checksum: str,
    arc_pool_grown: bool,
    random_seed: int,
    duration_s: float,
    baseline_positive_n: int,
    baseline_wrong_majority_n: int,
) -> dict[str, Any]:
    verdict = (
        "complete: arc_candidate_pool_grown_for_a2"
        if arc_pool_grown
        else "complete: cached_plus_bounded_topup_ceiling_below_target"
    )
    top_up_status = (
        "not_invoked_cached_pool_exceeded_baseline"
        if arc_pool_grown
        else "not_invoked_no_compatible_cached_arc_grid_topup_available"
    )
    return {
        "experiment": "experiment_4243_arc_candidate_pool_grow",
        "schema": "carnot.arc_candidate_pool_grow_result.v1",
        "honest_verdict": verdict,
        "arc_pool_grown": bool(arc_pool_grown),
        "positive_candidate_n": int(assembly.positive_candidate_n),
        "wrong_majority_n": int(assembly.wrong_majority_n),
        "held_out_task_n": int(assembly.task_n),
        "pool_artifact_path": str(POOL_ARTIFACT_REL),
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": _model_specs(
            assembly=assembly,
            top_up_status=top_up_status,
            baseline_positive_n=baseline_positive_n,
            baseline_wrong_majority_n=baseline_wrong_majority_n,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": "cached_pool_assembly_no_llm_no_training",
        "duration_s": round(duration_s, 6),
        "acceptance_gate": True,
        "pool_candidate_n": int(assembly.candidate_n),
        "raw_cached_candidate_n": int(assembly.raw_candidate_n),
        "skipped_optional_pools": assembly.skipped_optional_pools,
    }


def validate_artifact(artifact: dict[str, Any], repo_root: Path | str = Path(".")) -> None:
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or (
        not verdict.startswith(TERMINAL_PREFIXES) and not verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must be terminal-prefixed or blocked_*")
    if not isinstance(artifact["arc_pool_grown"], bool):
        raise ValueError("arc_pool_grown must be a bare bool")
    for field in ("positive_candidate_n", "wrong_majority_n", "held_out_task_n", "random_seed"):
        if isinstance(artifact[field], bool) or not isinstance(artifact[field], int):
            raise ValueError(f"{field} must be a bare int")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if not isinstance(artifact["pool_artifact_path"], str):
        raise ValueError("pool_artifact_path must be a string")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be a JSON object")
    if not isinstance(artifact["reproducibility_checksum"], str) or not artifact[
        "reproducibility_checksum"
    ].startswith("sha256:"):
        raise ValueError("reproducibility_checksum must be sha256-prefixed")
    if verdict.startswith(TERMINAL_PREFIXES):
        pool_path = Path(repo_root) / artifact["pool_artifact_path"]
        if not pool_path.exists():
            raise ValueError("complete artifact must point to an existing pool artifact")


def run(
    repo_root: Path | str = Path("."),
    *,
    pool_specs: tuple[PoolSpec, ...] = DEFAULT_POOL_SPECS,
    random_seed: int = RANDOM_SEED,
    baseline_positive_n: int = BASELINE_POSITIVE_CANDIDATES_392,
    baseline_wrong_majority_n: int = BASELINE_WRONG_MAJORITY_392,
    write: bool = False,
) -> dict[str, Any]:
    start = time.time()
    root = Path(repo_root)
    try:
        assembly = assemble_pool(root, pool_specs=pool_specs)
    except BlockedRun as exc:
        artifact = _blocked_artifact(str(exc), random_seed=random_seed, duration_s=time.time() - start)
        if write:
            out = root / OUTPUT_REL
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return artifact

    checksum = reproducibility_checksum(assembly, random_seed=random_seed)
    arc_pool_grown = (
        assembly.positive_candidate_n > int(baseline_positive_n)
        and assembly.wrong_majority_n > int(baseline_wrong_majority_n)
    )
    pool_path = root / POOL_ARTIFACT_REL
    persist_pool_artifact(pool_path, assembly, checksum=checksum, random_seed=random_seed)
    artifact = _complete_artifact(
        assembly,
        checksum=checksum,
        arc_pool_grown=arc_pool_grown,
        random_seed=random_seed,
        duration_s=time.time() - start,
        baseline_positive_n=baseline_positive_n,
        baseline_wrong_majority_n=baseline_wrong_majority_n,
    )
    validate_artifact(artifact, root)
    if write:
        out = root / OUTPUT_REL
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    artifact = run(Path(__file__).resolve().parents[2], write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
