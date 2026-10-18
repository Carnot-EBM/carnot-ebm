"""Exp 4270 ARC family provenance recovery.

Spec refs: REQ-VERIFY-4270, SCENARIO-VERIFY-4270.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


RANDOM_SEED = 4270
POOL_REL = Path("results/experiment_4243_arc_candidate_pool_grow_pool.json.gz")
OUTPUT_REL = Path("results/experiment_4270_arc_family_provenance_recovery.json")
MANIFEST_REL = Path("results/experiment_4270_arc_family_manifest.json")
SURVEY_REL = Path("results/arc3_win_condition_survey.json")
BLOCKED_SOURCE_TAXONOMY_VERDICT = "blocked_arc_source_taxonomy_unavailable"
SPEC_REFS = ["REQ-VERIFY-4270", "SCENARIO-VERIFY-4270"]
INFERENCE_SUBSTRATE = "cached_pool_provenance_recovery_no_llm_no_training"
MIN_HELD_OUT_TASK_N = 10
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    pool_rel: Path
    programs_rel: Path
    required: bool = False


DEFAULT_SOURCE_SPECS = (
    SourceSpec(
        "gap3_stage2",
        Path("results/arc3_gap3_stage2_eval_pool.json.gz"),
        Path("results/arc3_gap4_induced_programs.json"),
        required=False,
    ),
    SourceSpec(
        "gap4_arc2",
        Path("results/arc3_gap4_arc2_eval_pool.json.gz"),
        Path("results/arc3_gap4_arc2_induced_programs.json"),
        required=False,
    ),
)

DEFAULT_TAXONOMY_RELS = (
    Path("results/arc_tgi_family_taxonomy.json"),
    Path("results/arc_tgi_task_families.json"),
    Path("data/arc_tgi_family_taxonomy.json"),
    Path("data/arc_tgi_task_families.json"),
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A recovered manifest (feasible) AND an honest "
        "infeasible-pool finding are BOTH COMPLETE and decision-grade (feasible -> "
        "A2 tests existing; infeasible -> A3 builds fresh)."
    ),
    "family_split_feasible": (
        "BARE bool: A2 gates on ==true, A3 on ==false, A4 on ==true "
        "(gated-fields-must-be-bare); true iff a family-disjoint split with >=4 "
        "families and >=~10 tasks per held-out fold exists on the existing pool."
    ),
    "distinct_family_n": (
        "BARE int: number of distinct task-families recovered -- the OOD breadth; "
        "<4 means the existing pool cannot support a powered cross-family test."
    ),
    "per_family_task_count": (
        "Histogram of tasks per family -- exposes family concentration (the reason "
        "exp4258's pool may be untestable cross-family)."
    ),
    "provenance_manifest_path": (
        "Path to the first-class per-row {source_kind, family_id, game_id, fold, "
        "target_hash} manifest -- the Reliability Gap artifact every downstream "
        "cross-family claim must trace to."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- this is a data-provenance recovery for the learned "
        "selector, not an executable oracle."
    ),
    "random_seed": "Determinism precondition; the fold assignment must be reproducible.",
    "reproducibility_checksum": (
        "Hash of the pool + join sources + manifest; lets a third party re-derive "
        "the provenance."
    ),
    "model_specs": (
        "The join sources (ARC-TGI taxonomy / 25-game survey / source pools) + the "
        "fallback-unit policy; required methodology."
    ),
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "family_split_feasible",
    "distinct_family_n",
    "per_family_task_count",
    "provenance_manifest_path",
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
class SourceJoin:
    source_id: str
    raw_task_id: str
    entry_count: int
    candidate_count: int
    pred_hashes: tuple[str, ...]


@dataclass(frozen=True)
class TaxonomyRecord:
    family_id: str
    game_id: str | None
    source_path: str


@dataclass(frozen=True)
class ManifestRow:
    task_id: str
    raw_task_id: str
    source_id: str
    source_kind: str
    family_id: str
    game_id: str | None
    fold: int
    target_hash: str
    recovered_by: str
    target_hash_recovered: bool
    source_join_found: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "raw_task_id": self.raw_task_id,
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "family_id": self.family_id,
            "game_id": self.game_id,
            "fold": int(self.fold),
            "target_hash": self.target_hash,
            "recovered_by": self.recovered_by,
            "target_hash_recovered": bool(self.target_hash_recovered),
            "source_join_found": bool(self.source_join_found),
        }


@dataclass(frozen=True)
class Manifest:
    rows: list[ManifestRow]
    source_paths: list[Path]
    source_sha256: dict[str, str]
    taxonomy_paths: list[Path]
    survey_path: Path | None
    fold_task_counts: dict[str, int]
    max_feasible_fold_count: int
    min_held_out_task_n: int
    fallback_rows: list[str]
    target_hash_unavailable_rows: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "carnot.arc_family_manifest.v1",
            "spec_refs": SPEC_REFS,
            "random_seed": RANDOM_SEED,
            "rows": [row.to_dict() for row in self.rows],
            "source_paths": [str(path) for path in self.source_paths],
            "source_sha256": self.source_sha256,
            "taxonomy_paths": [str(path) for path in self.taxonomy_paths],
            "survey_path": str(self.survey_path) if self.survey_path else None,
            "fold_task_counts": self.fold_task_counts,
            "max_feasible_fold_count": int(self.max_feasible_fold_count),
            "min_held_out_task_n": int(self.min_held_out_task_n),
            "fallback_rows": list(self.fallback_rows),
            "target_hash_unavailable_rows": list(self.target_hash_unavailable_rows),
        }


def _json_grid(value: Any) -> str:
    return json.dumps(value if isinstance(value, list) else [], sort_keys=True, separators=(",", ":"))


def _grid_hash(value: Any) -> str:
    return hashlib.sha256(_json_grid(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_missing_token(source_id: str, raw_task_id: str) -> str:
    raw = f"{source_id}:{raw_task_id}".encode()
    return "unavailable:" + hashlib.sha256(raw).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _load_pool_payload(pool_path: Path) -> dict[str, Any]:
    try:
        with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise BlockedRun(BLOCKED_SOURCE_TAXONOMY_VERDICT) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise BlockedRun(BLOCKED_SOURCE_TAXONOMY_VERDICT)
    return payload


def _program_pred_hash(program: dict[str, Any]) -> str | None:
    value = program.get("pred_hash")
    if isinstance(value, str) and value:
        return value
    pred_grid = program.get("pred_grid")
    if isinstance(pred_grid, list):
        return _grid_hash(pred_grid)
    return None


def _load_source_joins(
    repo_root: Path,
    source_specs: tuple[SourceSpec, ...],
) -> tuple[dict[tuple[str, str], SourceJoin], list[Path], dict[str, str], list[str]]:
    joins: dict[tuple[str, str], SourceJoin] = {}
    source_paths: list[Path] = []
    source_sha256: dict[str, str] = {}
    missing_required: list[str] = []
    for spec in source_specs:
        pool_path = repo_root / spec.pool_rel
        programs_path = repo_root / spec.programs_rel
        if not pool_path.exists() or not programs_path.exists():
            if spec.required:
                missing_required.append(spec.source_id)
            continue
        try:
            with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
                pool_payload = json.load(handle)
            programs_payload = _read_json_object(programs_path)
        except Exception:
            if spec.required:
                missing_required.append(spec.source_id)
            continue
        entries = pool_payload.get("entries") if isinstance(pool_payload, dict) else None
        programs = programs_payload.get("programs")
        if not isinstance(entries, list) or not isinstance(programs, list):
            if spec.required:
                missing_required.append(spec.source_id)
            continue
        resolved_pool = pool_path.resolve()
        resolved_programs = programs_path.resolve()
        source_paths.extend([resolved_pool, resolved_programs])
        source_sha256[str(resolved_pool)] = _sha256_file(resolved_pool)
        source_sha256[str(resolved_programs)] = _sha256_file(resolved_programs)

        pred_by_task: dict[str, set[str]] = {}
        for program in programs:
            if not isinstance(program, dict):
                continue
            raw_task_id = str(program.get("task") or "")
            if not raw_task_id:
                continue
            pred_hash = _program_pred_hash(program)
            if pred_hash:
                pred_by_task.setdefault(raw_task_id, set()).add(pred_hash)

        counts: dict[str, dict[str, int]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            raw_task_id = str(entry.get("task") or "")
            if not raw_task_id:
                continue
            candidates = entry.get("candidates")
            item = counts.setdefault(raw_task_id, {"entry_count": 0, "candidate_count": 0})
            item["entry_count"] += 1
            if isinstance(candidates, list):
                item["candidate_count"] += sum(1 for candidate in candidates if isinstance(candidate, dict))

        for raw_task_id, item in counts.items():
            joins[(spec.source_id, raw_task_id)] = SourceJoin(
                source_id=spec.source_id,
                raw_task_id=raw_task_id,
                entry_count=item["entry_count"],
                candidate_count=item["candidate_count"],
                pred_hashes=tuple(sorted(pred_by_task.get(raw_task_id, set()))),
            )
    return joins, source_paths, source_sha256, missing_required


def _iter_taxonomy_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("families", "tasks", "rows", "task_families", "generators"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    for key, value in payload.items():
        if isinstance(value, str):
            rows.append({"task_id": key, "family_id": value})
        elif isinstance(value, dict):
            rows.append({"task_id": key, **value})
    return rows


def _first_string(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _load_taxonomy(
    repo_root: Path,
    taxonomy_rels: tuple[Path, ...],
) -> tuple[dict[tuple[str, str], TaxonomyRecord], dict[str, TaxonomyRecord], list[Path]]:
    keyed: dict[tuple[str, str], TaxonomyRecord] = {}
    by_raw: dict[str, TaxonomyRecord] = {}
    paths: list[Path] = []
    for rel in taxonomy_rels:
        path = repo_root / rel
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        resolved = path.resolve()
        paths.append(resolved)
        for row in _iter_taxonomy_rows(payload):
            raw_task_id = _first_string(row, ("raw_task_id", "task_id", "arc_task_id", "source_task_id"))
            family_id = _first_string(row, ("family_id", "task_family_id", "generator_id", "family"))
            if not raw_task_id or not family_id:
                continue
            source_id = _first_string(row, ("source_id", "source", "dataset"))
            record = TaxonomyRecord(
                family_id=family_id,
                game_id=_first_string(row, ("game_id", "game", "env_id", "environment_id")),
                source_path=str(resolved),
            )
            by_raw.setdefault(raw_task_id, record)
            if source_id:
                keyed[(source_id, raw_task_id)] = record
    return keyed, by_raw, paths


def _survey_games(repo_root: Path) -> tuple[set[str], Path | None]:
    path = repo_root / SURVEY_REL
    if not path.exists():
        return set(), None
    try:
        payload = _read_json_object(path)
    except Exception:
        return set(), path.resolve()
    games: set[str] = set()
    for field in ("per_game_surveys", "ranked_targets"):
        rows = payload.get(field)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                value = row.get("game")
                if isinstance(value, str) and value:
                    games.add(value)
    return games, path.resolve()


def _candidate_strings(candidates: Any, keys: tuple[str, ...]) -> set[str]:
    values: set[str] = set()
    if not isinstance(candidates, list):
        return values
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in keys:
            value = candidate.get(key)
            if isinstance(value, str) and value:
                values.add(value)
    return values


def _direct_task_string(task: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    value = _first_string(task, keys)
    if value:
        return value
    values = _candidate_strings(task.get("candidates"), keys)
    if len(values) == 1:
        return next(iter(values))
    return None


def _recover_game_id(
    task: dict[str, Any],
    taxonomy: TaxonomyRecord | None,
    survey_games: set[str],
) -> str | None:
    direct = _direct_task_string(task, ("game_id", "game", "env_id", "environment_id", "target_game"))
    if direct and (not survey_games or direct in survey_games):
        return direct
    if taxonomy and taxonomy.game_id:
        return taxonomy.game_id
    task_id = str(task.get("task_id") or "")
    for token in task_id.replace("/", ":").split(":"):
        if token in survey_games:
            return token
    return direct if direct else None


def _recover_family(
    task: dict[str, Any],
    raw_task_id: str,
    taxonomy: TaxonomyRecord | None,
) -> tuple[str, str]:
    direct = _direct_task_string(task, ("family_id", "task_family_id", "generator_id", "family"))
    if direct:
        return direct, "pool_metadata"
    if taxonomy:
        return taxonomy.family_id, "arc_tgi_taxonomy"
    return f"original_arc_task:{raw_task_id}", "original_arc_task_fallback"


def _candidate_hash(candidate: dict[str, Any]) -> str | None:
    value = candidate.get("candidate_grid_hash")
    if isinstance(value, str) and value:
        return value
    grid = candidate.get("grid")
    if isinstance(grid, list):
        return _grid_hash(grid)
    return None


def _recover_target_hash(
    task: dict[str, Any],
    join: SourceJoin | None,
    *,
    source_id: str,
    raw_task_id: str,
) -> tuple[str, str, bool]:
    candidates = task.get("candidates")
    positive_hashes: set[str] = set()
    positive_source_kinds: set[str] = set()
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict) or candidate.get("is_correct") is not True:
                continue
            value = _candidate_hash(candidate)
            if value:
                positive_hashes.add(value)
            source_kinds = candidate.get("source_kinds")
            if isinstance(source_kinds, list):
                positive_source_kinds.update(str(kind) for kind in source_kinds)
    if positive_hashes:
        ordered = sorted(positive_hashes)
        target_hash = ordered[0] if len(ordered) == 1 else "multi:" + hashlib.sha256(
            json.dumps(ordered, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        induced_hashes = set(join.pred_hashes) if join else set()
        source_kind = (
            "induced"
            if "induced_pred_grid" in positive_source_kinds or bool(positive_hashes & induced_hashes)
            else "sampled"
        )
        return target_hash, source_kind, True
    if join and join.pred_hashes:
        return join.pred_hashes[0], "induced", True
    return _hash_missing_token(source_id, raw_task_id), "sampled", False


def _recover_rows(
    tasks: list[dict[str, Any]],
    joins: dict[tuple[str, str], SourceJoin],
    taxonomy_keyed: dict[tuple[str, str], TaxonomyRecord],
    taxonomy_by_raw: dict[str, TaxonomyRecord],
    survey_games: set[str],
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or f"task-{index}")
        source_id = str(task.get("source_id") or task_id.split(":", 1)[0])
        raw_task_id = str(task.get("raw_task_id") or task_id.split(":", 1)[-1])
        join = joins.get((source_id, raw_task_id))
        taxonomy = taxonomy_keyed.get((source_id, raw_task_id)) or taxonomy_by_raw.get(raw_task_id)
        family_id, recovered_by = _recover_family(task, raw_task_id, taxonomy)
        target_hash, source_kind, target_recovered = _recover_target_hash(
            task,
            join,
            source_id=source_id,
            raw_task_id=raw_task_id,
        )
        rows.append(
            ManifestRow(
                task_id=task_id,
                raw_task_id=raw_task_id,
                source_id=source_id,
                source_kind=source_kind,
                family_id=family_id,
                game_id=_recover_game_id(task, taxonomy, survey_games),
                fold=-1,
                target_hash=target_hash,
                recovered_by=recovered_by,
                target_hash_recovered=target_recovered,
                source_join_found=join is not None,
            )
        )
    return rows


def _family_task_counts(rows: list[ManifestRow]) -> dict[str, int]:
    counts = Counter(row.family_id for row in rows)
    return dict(sorted(counts.items()))


def _assign_family_folds(
    per_family_task_count: dict[str, int],
    *,
    random_seed: int,
    min_held_out_task_n: int,
) -> tuple[dict[str, int], dict[str, int], int, bool, str]:
    if not per_family_task_count:
        return {}, {}, 0, False, "no family units recovered"
    distinct_family_n = len(per_family_task_count)
    total_task_n = sum(per_family_task_count.values())
    if distinct_family_n < 4:
        reason = f"only {distinct_family_n} distinct family units recovered (<4)"
        return {family: 0 for family in per_family_task_count}, {"0": total_task_n}, 0, False, reason
    max_possible = min(distinct_family_n, total_task_n // max(1, int(min_held_out_task_n)))
    if max_possible < 2:
        reason = f"no held-out fold has fewer than {min_held_out_task_n} tasks avoidable at task_n={total_task_n}"
        return {family: 0 for family in per_family_task_count}, {"0": total_task_n}, 0, False, reason

    rng = random.Random(random_seed)
    families = list(per_family_task_count)
    rng.shuffle(families)
    families.sort(key=lambda family: (-per_family_task_count[family], family))

    best_assignment: dict[str, int] = {}
    best_counts: dict[str, int] = {}
    best_fold_count = 0
    for fold_count in range(max_possible, 1, -1):
        fold_loads = [0 for _ in range(fold_count)]
        assignment: dict[str, int] = {}
        for family in families:
            fold = min(range(fold_count), key=lambda idx: (fold_loads[idx], idx))
            assignment[family] = fold
            fold_loads[fold] += per_family_task_count[family]
        if all(load >= min_held_out_task_n for load in fold_loads):
            best_assignment = assignment
            best_counts = {str(index): int(load) for index, load in enumerate(fold_loads)}
            best_fold_count = fold_count
            break
    if not best_assignment:
        fold_loads = [0, 0]
        assignment = {}
        for family in families:
            fold = min(range(2), key=lambda idx: (fold_loads[idx], idx))
            assignment[family] = fold
            fold_loads[fold] += per_family_task_count[family]
        best_assignment = assignment
        best_counts = {str(index): int(load) for index, load in enumerate(fold_loads)}
        reason = f"at least one held-out fold has fewer than {min_held_out_task_n} tasks"
        return best_assignment, best_counts, 0, False, reason
    return best_assignment, best_counts, best_fold_count, True, ""


def build_manifest(
    repo_root: Path | str = Path("."),
    *,
    source_specs: tuple[SourceSpec, ...] = DEFAULT_SOURCE_SPECS,
    taxonomy_rels: tuple[Path, ...] = DEFAULT_TAXONOMY_RELS,
    random_seed: int = RANDOM_SEED,
    min_held_out_task_n: int = MIN_HELD_OUT_TASK_N,
) -> tuple[Manifest, bool, str]:
    root = Path(repo_root)
    pool_path = root / POOL_REL
    payload = _load_pool_payload(pool_path)
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise BlockedRun(BLOCKED_SOURCE_TAXONOMY_VERDICT)
    joins, source_paths, source_sha256, missing_required = _load_source_joins(root, source_specs)
    taxonomy_keyed, taxonomy_by_raw, taxonomy_paths = _load_taxonomy(root, taxonomy_rels)
    survey_games, survey_path = _survey_games(root)
    if missing_required:
        raise BlockedRun(BLOCKED_SOURCE_TAXONOMY_VERDICT)
    if not joins and not taxonomy_keyed and not taxonomy_by_raw:
        raise BlockedRun(BLOCKED_SOURCE_TAXONOMY_VERDICT)

    pool_resolved = pool_path.resolve()
    source_paths = [pool_resolved, *source_paths, *taxonomy_paths]
    source_sha256[str(pool_resolved)] = _sha256_file(pool_resolved)
    if survey_path and survey_path.exists():
        source_paths.append(survey_path)
        source_sha256[str(survey_path)] = _sha256_file(survey_path)

    rows = _recover_rows(tasks, joins, taxonomy_keyed, taxonomy_by_raw, survey_games)
    per_family = _family_task_counts(rows)
    fold_by_family, fold_task_counts, max_feasible_fold_count, feasible, reason = _assign_family_folds(
        per_family,
        random_seed=random_seed,
        min_held_out_task_n=min_held_out_task_n,
    )
    folded_rows = [replace(row, fold=int(fold_by_family.get(row.family_id, 0))) for row in rows]
    fallback_rows = [
        row.task_id for row in folded_rows if row.recovered_by == "original_arc_task_fallback"
    ]
    target_missing = [row.task_id for row in folded_rows if not row.target_hash_recovered]
    manifest = Manifest(
        rows=folded_rows,
        source_paths=sorted({path.resolve() for path in source_paths}, key=lambda p: str(p)),
        source_sha256=source_sha256,
        taxonomy_paths=sorted({path.resolve() for path in taxonomy_paths}, key=lambda p: str(p)),
        survey_path=survey_path,
        fold_task_counts=fold_task_counts,
        max_feasible_fold_count=max_feasible_fold_count,
        min_held_out_task_n=int(min_held_out_task_n),
        fallback_rows=fallback_rows,
        target_hash_unavailable_rows=target_missing,
    )
    return manifest, feasible, reason


def load_manifest(path: Path) -> Manifest:
    payload = _read_json_object(path)
    rows = [
        ManifestRow(
            task_id=str(row["task_id"]),
            raw_task_id=str(row["raw_task_id"]),
            source_id=str(row["source_id"]),
            source_kind=str(row["source_kind"]),
            family_id=str(row["family_id"]),
            game_id=row.get("game_id") if isinstance(row.get("game_id"), str) else None,
            fold=int(row["fold"]),
            target_hash=str(row["target_hash"]),
            recovered_by=str(row["recovered_by"]),
            target_hash_recovered=bool(row.get("target_hash_recovered")),
            source_join_found=bool(row.get("source_join_found")),
        )
        for row in payload.get("rows", [])
        if isinstance(row, dict)
    ]
    return Manifest(
        rows=rows,
        source_paths=[Path(path) for path in payload.get("source_paths", [])],
        source_sha256=dict(payload.get("source_sha256", {})),
        taxonomy_paths=[Path(path) for path in payload.get("taxonomy_paths", [])],
        survey_path=Path(payload["survey_path"]) if payload.get("survey_path") else None,
        fold_task_counts={str(k): int(v) for k, v in payload.get("fold_task_counts", {}).items()},
        max_feasible_fold_count=int(payload.get("max_feasible_fold_count", 0)),
        min_held_out_task_n=int(payload.get("min_held_out_task_n", MIN_HELD_OUT_TASK_N)),
        fallback_rows=list(payload.get("fallback_rows", [])),
        target_hash_unavailable_rows=list(payload.get("target_hash_unavailable_rows", [])),
    )


def persist_manifest(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def reproducibility_checksum(
    *,
    source_paths: list[Path],
    manifest: Manifest,
    random_seed: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"random_seed": int(random_seed)}, sort_keys=True).encode("utf-8"))
    for path in sorted({Path(path) for path in source_paths}, key=lambda p: str(p)):
        digest.update(str(path).encode("utf-8"))
        if not path.exists():
            digest.update(b"\0MISSING\0")
            continue
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    manifest_payload = manifest.to_dict()
    manifest_payload.pop("source_sha256", None)
    raw = json.dumps(manifest_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest.update(raw.encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _model_specs(
    *,
    manifest: Manifest | None,
    status: str,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "blocked_reason": blocked_reason,
        "join_sources": {
            "grown_pool": str(POOL_REL),
            "source_pools": [
                {
                    "source_id": spec.source_id,
                    "pool_path": str(spec.pool_rel),
                    "programs_path": str(spec.programs_rel),
                }
                for spec in DEFAULT_SOURCE_SPECS
            ],
            "arc_tgi_taxonomy_paths_consulted": [str(path) for path in DEFAULT_TAXONOMY_RELS],
            "arc3_survey": str(SURVEY_REL),
            "available_source_paths": [str(path) for path in manifest.source_paths] if manifest else [],
        },
        "fallback_policy": {
            "family_id": "Use checked-in ARC-TGI family metadata when available; otherwise use original_arc_task:<raw_task_id> as the honest disjoint unit.",
            "game_id": "Use pool/taxonomy/survey game metadata when available; otherwise leave null because original ARC tasks do not carry ARC-AGI-3 game ids.",
            "target_hash": "Use correct candidate grid hash, then induced pred_hash, else unavailable:<sha256(source_id:raw_task_id)>.",
        },
        "fold_policy": {
            "split_unit": "family_id",
            "min_distinct_families": 4,
            "min_held_out_task_n": manifest.min_held_out_task_n if manifest else MIN_HELD_OUT_TASK_N,
            "max_feasible_fold_count": manifest.max_feasible_fold_count if manifest else 0,
            "fold_task_counts": manifest.fold_task_counts if manifest else {},
        },
    }


def _blocked_checksum(reason: str, random_seed: int) -> str:
    raw = json.dumps({"random_seed": int(random_seed), "reason": reason}, sort_keys=True)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _blocked_artifact(reason: str, *, random_seed: int, duration_s: float) -> dict[str, Any]:
    return {
        "experiment": "experiment_4270_arc_family_provenance_recovery",
        "schema": "carnot.arc_family_provenance_recovery_4270.v1",
        "status": "complete",
        "honest_verdict": reason,
        "family_split_feasible": False,
        "distinct_family_n": 0,
        "per_family_task_count": {},
        "provenance_manifest_path": "",
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": _blocked_checksum(reason, random_seed),
        "model_specs": _model_specs(manifest=None, status="blocked", blocked_reason=reason),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "adversarial_verify": {"status": "pending"},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "infeasible_reason": reason,
        "fallback_rows_sample": [],
        "diversity_audit": {
            "max_feasible_fold_count": 0,
            "fold_task_counts": {},
        },
    }


def _complete_artifact(
    *,
    manifest: Manifest,
    feasible: bool,
    infeasible_reason: str,
    checksum: str,
    random_seed: int,
    duration_s: float,
) -> dict[str, Any]:
    per_family = _family_task_counts(manifest.rows)
    distinct_family_n = len(per_family)
    verdict = (
        "complete: arc_family_manifest_recovered_existing_pool_feasible"
        if feasible
        else "complete: arc_family_manifest_recovered_existing_pool_infeasible"
    )
    return {
        "experiment": "experiment_4270_arc_family_provenance_recovery",
        "schema": "carnot.arc_family_provenance_recovery_4270.v1",
        "status": "complete",
        "honest_verdict": verdict,
        "family_split_feasible": bool(feasible),
        "distinct_family_n": int(distinct_family_n),
        "per_family_task_count": per_family,
        "provenance_manifest_path": str(MANIFEST_REL),
        "verifier_is_oracle": False,
        "random_seed": int(random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": _model_specs(manifest=manifest, status="complete"),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "acceptance_gate": True,
        "adversarial_verify": {"status": "pending"},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration_s, 6),
        "infeasible_reason": "" if feasible else infeasible_reason,
        "fallback_rows_sample": manifest.fallback_rows[:20],
        "diversity_audit": {
            "max_feasible_fold_count": int(manifest.max_feasible_fold_count),
            "fold_task_counts": manifest.fold_task_counts,
            "min_held_out_task_n": int(manifest.min_held_out_task_n),
        },
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
    return {
        "status": "clean" if not flags else "flagged",
        "flag_count": len(flags),
        "flags": flags,
        "returncode": int(report.get("returncode", 0) or 0),
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith(TERMINAL_PREFIXES) or verdict == BLOCKED_SOURCE_TAXONOMY_VERDICT
    ):
        raise ValueError("honest_verdict must be terminal-prefixed or the blocked source taxonomy verdict")
    if type(artifact["family_split_feasible"]) is not bool:
        raise ValueError("family_split_feasible must be a bare bool")
    if type(artifact["distinct_family_n"]) is not int:
        raise ValueError("distinct_family_n must be a bare int")
    if not isinstance(artifact["per_family_task_count"], dict):
        raise ValueError("per_family_task_count must be a histogram")
    for family, count in artifact["per_family_task_count"].items():
        if not isinstance(family, str) or type(count) is not int:
            raise ValueError("per_family_task_count must be a histogram of string family to int count")
    if not isinstance(artifact["provenance_manifest_path"], str):
        raise ValueError("provenance_manifest_path must be a string")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle must be the bare bool false")
    if type(artifact["random_seed"]) is not int:
        raise ValueError("random_seed must be a bare int")
    checksum = artifact["reproducibility_checksum"]
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        raise ValueError("reproducibility_checksum must be a sha256 string")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be a dict")
    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles do not match REQ-VERIFY-4270")
    if artifact["spec_refs"] != SPEC_REFS:
        raise ValueError("spec_refs do not match REQ-VERIFY-4270")


def run(
    repo_root: Path | str = Path("."),
    *,
    random_seed: int = RANDOM_SEED,
    min_held_out_task_n: int = MIN_HELD_OUT_TASK_N,
    source_specs: tuple[SourceSpec, ...] = DEFAULT_SOURCE_SPECS,
    taxonomy_rels: tuple[Path, ...] = DEFAULT_TAXONOMY_RELS,
    adversarial_runner: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    root = Path(repo_root)
    output_path = root / OUTPUT_REL
    manifest_path = root / MANIFEST_REL
    try:
        manifest, feasible, infeasible_reason = build_manifest(
            root,
            source_specs=source_specs,
            taxonomy_rels=taxonomy_rels,
            random_seed=random_seed,
            min_held_out_task_n=min_held_out_task_n,
        )
        persist_manifest(manifest_path, manifest)
        checksum = reproducibility_checksum(
            source_paths=manifest.source_paths + [manifest_path],
            manifest=manifest,
            random_seed=random_seed,
        )
        artifact = _complete_artifact(
            manifest=manifest,
            feasible=feasible,
            infeasible_reason=infeasible_reason,
            checksum=checksum,
            random_seed=random_seed,
            duration_s=time.perf_counter() - start,
        )
    except BlockedRun as exc:
        reason = str(exc) or BLOCKED_SOURCE_TAXONOMY_VERDICT
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


def main() -> None:  # pragma: no cover - exercised by the result entrypoint.
    repo_root = Path(__file__).resolve().parents[3]
    print(json.dumps(run(repo_root), indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
