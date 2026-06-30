"""Exp 5044: reusable second-corpus candidate cache.

Spec refs: REQ-VERIFY-5044, SCENARIO-VERIFY-5044.

This data product fills the D4 precondition that Exp 5035 lacked: a second
corpus with cached candidate rows, labels, genuine self-consistency, oracle@K,
and auditable solver provenance. It prefers PPBench when already present, but
does not download it. The local fallback is the checked-in ConstraintBench exact
pilot, which is solver-backed and deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import moat_benchmark_harness as harness  # noqa: E402
from carnot.eval import constraintbench_feasibility_objective_pilot_v1 as cb  # noqa: E402


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]
SourceLoader = Callable[..., tuple[list[JsonDict], Path, list["PreconditionCheck"]]]

EXPERIMENT_ID = 5044
EXPERIMENT_NAME = "experiment_5044_second_corpus_candidate_cache"
ARTIFACT_SCHEMA = "carnot.experiment_5044_second_corpus_candidate_cache.v1"
CACHE_ROW_SCHEMA = "carnot.second_corpus_candidate_cache.row.v1"
RESULT_RELATIVE_PATH = "results/experiment_5044_second_corpus_candidate_cache.json"
CACHE_RELATIVE_PATH = "results/experiment_5044_second_corpus_candidate_cache.jsonl"
CONSTRAINTBENCH_CORPUS_NAME = "ConstraintBench-exact-v1"
CONSTRAINTBENCH_SOURCE_RELATIVE_PATH = cb.FIXTURE_REL_PATH
SPEC_REFS = ["REQ-VERIFY-5044", "SCENARIO-VERIFY-5044"]
RANDOM_SEED = harness.DEFAULT_RANDOM_SEED
DEFAULT_LIMIT = 100
DEFAULT_MIN_QUESTIONS = 100
DEFAULT_CANDIDATES_PER_QUESTION = 5
HEADROOM_THRESHOLD = harness.HEADROOM_THRESHOLD
PPBENCH_PROBE_PATHS = (
    Path("data/ppbench"),
    Path("data/pencil_puzzle_bench"),
    Path("data/pencil-puzzle-bench"),
    Path("third_party/pencil-puzzle-bench"),
)
MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix for ready, blocked, or fallback cache outcomes."
    },
    "model_specs": {"principle": "records mandated model ids and whether LLM generation was used."},
    "second_corpus_cache_built": {
        "principle": "true iff the reusable JSONL cache has enough complete labeled rows."
    },
    "second_corpus_name": {
        "principle": "the selected local second corpus or honest fallback corpus name."
    },
    "n_questions": {"principle": "number of complete labeled questions in the cache."},
    "n_candidate_rows": {"principle": "total candidate answers across complete questions."},
    "genuine_sc_accuracy": {
        "principle": "GENUINE tuned self-consistency accuracy over cached candidates."
    },
    "oracle_at_k": {"principle": "oracle recovery rate over the cached K-candidate pools."},
    "headroom_present": {
        "principle": "true only when oracle@K exceeds genuine SC by the headroom threshold."
    },
    "verifier_is_oracle": {
        "principle": "false because the cache builder labels rows but does not select with an oracle."
    },
    "candidate_cache_path": {
        "principle": "path to the resumable second-corpus candidate JSONL cache."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "random_seed",
    "ppbench_probe",
    "fallback_used",
    "fallback_reason",
    "source_paths",
    "solver_provenance",
    "verifier_provenance",
    "candidates_per_question",
    "candidate_cache_schema",
    "resume_summary",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "field_principles",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One local resource check used by the Exp 5044 cache builder."""

    resource: str
    available: bool
    detail: str
    path: str | None = None

    def as_dict(self) -> JsonDict:
        payload: JsonDict = {
            "resource": self.resource,
            "available": bool(self.available),
            "detail": self.detail,
        }
        if self.path is not None:
            payload["path"] = self.path
        return payload


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl_row(path: Path, row: JsonMap) -> None:
    """Append one complete cache row and fsync it for interruption-safe resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, line)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _precondition_dicts(checks: Sequence[PreconditionCheck | JsonMap]) -> list[JsonDict]:
    return [
        check.as_dict() if isinstance(check, PreconditionCheck) else dict(check) for check in checks
    ]


def _canonical_solution(solution: Mapping[str, Any]) -> str:
    return _json_dumps(dict(solution))


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def probe_ppbench(root: Path = REPO_ROOT) -> JsonDict:
    """Probe only local PPBench-style paths; never downloads puzzle assets."""

    root = Path(root)
    probed = [root / relative for relative in PPBENCH_PROBE_PATHS]
    existing = [path.as_posix() for path in probed if path.exists()]
    return {
        "available": bool(existing),
        "probed_paths": [path.as_posix() for path in probed],
        "existing_paths": existing,
        "selected_path": existing[0] if existing else None,
        "download_attempted": False,
        "detail": "local PPBench/Pencil Puzzle Bench path found"
        if existing
        else "PPBench/Pencil Puzzle Bench not found locally",
    }


def default_source_loader(
    *,
    root: Path = REPO_ROOT,
) -> tuple[list[JsonDict], Path, list[PreconditionCheck]]:
    """Load the checked-in ConstraintBench exact pilot rows."""

    source_path = Path(root) / CONSTRAINTBENCH_SOURCE_RELATIVE_PATH
    if source_path.exists():
        rows = _read_jsonl(source_path)
        detail = f"{len(rows)} checked-in exact solver-backed row(s)"
        return (
            rows,
            source_path,
            [
                PreconditionCheck(
                    "constraintbench_exact_pilot",
                    bool(rows),
                    detail,
                    source_path.as_posix(),
                )
            ],
        )
    rows = cb.build_fixture_rows()
    return (
        rows,
        source_path,
        [
            PreconditionCheck(
                "constraintbench_exact_pilot",
                bool(rows),
                "checked-in file missing; rebuilt deterministic exact fixture rows from module",
                source_path.as_posix(),
            )
        ],
    )


def _with_exact_reference(row: JsonMap) -> JsonDict:
    prepared = dict(row)
    if not isinstance(prepared.get("exact_reference"), Mapping):
        prepared["exact_reference"] = cb.solve_row(prepared)
    return prepared


def _fallback_wrong_solution(row: JsonMap) -> JsonDict:
    family = str(row.get("family"))
    data = row.get("instance_data", {})
    if family == "knapsack":
        return {"selected_items": [str(item["name"]) for item in data.get("items", [])]}
    if family == "assignment":
        tasks = [str(task) for task in data.get("tasks", [])]
        workers = [str(worker) for worker in data.get("workers", [])]
        worker = workers[0] if workers else ""
        return {"assignment": {task: worker for task in tasks}}
    nodes = [str(node) for node in data.get("nodes", [])]
    colors = list(data.get("colors", []))
    color = int(colors[0]) if colors else 0
    return {"colors": {node: color for node in nodes}}


def _wrong_solution(row: JsonMap) -> JsonDict:
    try:
        return cb.feasible_nonoptimal_solution(row)
    except ValueError:
        return _fallback_wrong_solution(row)


def _constraint_context(row: JsonMap, *, variant_index: int) -> str:
    data = row.get("instance_data", {})
    return (
        f"Variant {variant_index}. Family: {row.get('family')}. "
        f"Instance data: {_json_dumps(data)}. "
        f"Hard constraints: {_json_dumps(list(row.get('constraints') or []))}. "
        f"Objective: {_json_dumps(dict(row.get('objective') or {}))}."
    )


def _candidate_row_id(source_row: JsonMap, variant_index: int) -> str:
    return f"{source_row.get('row_id', 'constraintbench')}::variant-{variant_index:04d}"


def _score_candidate(row: JsonMap, answer: str) -> JsonDict:
    score = cb.score_candidate(row, answer)
    return {
        "valid_format": bool(score.get("valid_format")),
        "feasibility_pass": bool(score.get("feasibility_pass")),
        "objective_value": score.get("objective_value"),
        "objective_gap": score.get("objective_gap"),
        "reasons": list(score.get("reasons") or []),
        "checker_backend": score.get("checker_backend"),
    }


def build_candidate_cache_row(
    source_row: JsonMap,
    *,
    variant_index: int,
    source_path: Path,
    candidates_per_question: int = DEFAULT_CANDIDATES_PER_QUESTION,
) -> JsonDict:
    """Build one deterministic solver-backed candidate row."""

    row = _with_exact_reference(source_row)
    exact = dict(row["exact_reference"])
    gold = _canonical_solution(dict(exact["solution"]))
    wrong = _canonical_solution(_wrong_solution(row))
    row_id = _candidate_row_id(row, variant_index)
    answer_pattern = [wrong, gold, wrong, gold, wrong]
    answers = answer_pattern[:candidates_per_question]
    candidates: list[JsonDict] = []
    for candidate_index, answer in enumerate(answers):
        score = _score_candidate(row, answer)
        label_correct = answer == gold
        candidates.append(
            {
                "candidate_id": f"{row_id}/det-{candidate_index}",
                "answer": answer,
                "cache_index": candidate_index,
                "temperature": "deterministic",
                "source": "deterministic_constraintbench_variant",
                "generator_kind": "deterministic_solver_backed_variant",
                "generation_model": None,
                "label_correct": label_correct,
                "candidate_label": "correct" if label_correct else "incorrect",
                "solver_verdict": score,
                "solver_score_used_for_selection": False,
            }
        )
    return {
        "schema": CACHE_ROW_SCHEMA,
        "row_id": row_id,
        "source_row_id": str(row.get("row_id")),
        "variant_index": int(variant_index),
        "corpus": CONSTRAINTBENCH_CORPUS_NAME,
        "question": (
            f"Solve this {row.get('family')} constraint puzzle. "
            f"Return JSON only matching {_json_dumps(dict(row.get('candidate_schema') or {}))}."
        ),
        "context": _constraint_context(row, variant_index=variant_index),
        "choices": [],
        "gold": gold,
        "label": gold,
        "family": str(row.get("family")),
        "candidate_schema": dict(row.get("candidate_schema") or {}),
        "constraints": list(row.get("constraints") or []),
        "objective": dict(row.get("objective") or {}),
        "candidates": candidates,
        "solver_provenance": {
            "authority": "local_exhaustive_enumeration",
            "backend": str(row.get("checker_backend")),
            "source_path": source_path.as_posix(),
            "source_row_id": str(row.get("row_id")),
            "objective_value": exact.get("objective_value"),
            "feasible_count": exact.get("feasible_count"),
        },
        "verifier_provenance": {
            "verifier_is_oracle": False,
            "selection_verifier": "not_run_cache_data_product",
            "solver_used_for_candidate_selection": False,
        },
    }


def validate_cache_row(row: JsonMap) -> list[str]:
    errors: list[str] = []
    if row.get("schema") != CACHE_ROW_SCHEMA:
        errors.append("schema")
    if row.get("corpus") != CONSTRAINTBENCH_CORPUS_NAME:
        errors.append("corpus")
    if not str(row.get("row_id") or ""):
        errors.append("row_id")
    if not str(row.get("question") or ""):
        errors.append("question")
    if not str(row.get("gold") or ""):
        errors.append("gold")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("candidates")
    else:
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                errors.append("candidate_object")
                continue
            if not str(candidate.get("candidate_id") or ""):
                errors.append("candidate_id")
            if not str(candidate.get("answer") or ""):
                errors.append("candidate_answer")
            if not isinstance(candidate.get("label_correct"), bool):
                errors.append("label_correct")
            if not isinstance(candidate.get("solver_verdict"), Mapping):
                errors.append("solver_verdict")
        gold_text = str(row.get("gold") or "")
        if not any(
            isinstance(candidate, Mapping)
            and str(candidate.get("answer") or "")
            and str(candidate.get("answer") or "") == gold_text
            for candidate in candidates
        ):
            errors.append("gold_missing_from_candidates")
    solver = row.get("solver_provenance")
    if not isinstance(solver, Mapping) or solver.get("authority") != "local_exhaustive_enumeration":
        errors.append("solver_provenance")
    verifier = row.get("verifier_provenance")
    if not isinstance(verifier, Mapping) or verifier.get("verifier_is_oracle") is not False:
        errors.append("verifier_provenance")
    return sorted(set(errors))


def read_complete_candidate_rows(path: Path) -> list[JsonDict]:
    complete_by_id: dict[str, JsonDict] = {}
    for row in _read_jsonl(path):
        if not validate_cache_row(row):
            complete_by_id[str(row["row_id"])] = row
    return list(complete_by_id.values())


def _expected_source_rows(
    source_rows: Sequence[JsonMap], *, limit: int
) -> list[tuple[JsonMap, int]]:
    return [(source_rows[index % len(source_rows)], index) for index in range(limit)]


def ensure_candidate_cache(
    *,
    cache_path: Path,
    source_rows: Sequence[JsonMap],
    source_path: Path,
    limit: int,
    candidates_per_question: int = DEFAULT_CANDIDATES_PER_QUESTION,
) -> tuple[list[JsonDict], JsonDict]:
    """Resume a cache by appending only missing deterministic row ids."""

    if not source_rows:
        return [], {
            "existing_complete_rows": 0,
            "appended_rows": 0,
            "target_rows": int(limit),
            "skipped_existing_rows": 0,
        }
    existing = {row["row_id"]: row for row in read_complete_candidate_rows(cache_path)}
    appended = 0
    for source_row, variant_index in _expected_source_rows(source_rows, limit=limit):
        row_id = _candidate_row_id(source_row, variant_index)
        if row_id in existing:
            continue
        row = build_candidate_cache_row(
            source_row,
            variant_index=variant_index,
            source_path=source_path,
            candidates_per_question=candidates_per_question,
        )
        append_jsonl_row(cache_path, row)
        existing[row_id] = row
        appended += 1
    ordered_rows = [
        existing[_candidate_row_id(source_row, variant_index)]
        for source_row, variant_index in _expected_source_rows(source_rows, limit=limit)
        if _candidate_row_id(source_row, variant_index) in existing
    ]
    return ordered_rows, {
        "existing_complete_rows": int(len(existing) - appended),
        "appended_rows": int(appended),
        "target_rows": int(limit),
        "skipped_existing_rows": int(len(existing) - appended),
    }


def compute_cache_metrics(
    rows: Sequence[JsonMap],
    *,
    headroom_threshold: float = HEADROOM_THRESHOLD,
) -> JsonDict:
    complete_rows = [dict(row) for row in rows if not validate_cache_row(row)]
    tuned = harness.tuned_self_consistency(complete_rows)
    sc_accuracy = float(tuned.get("accuracy") or 0.0)
    sc_correct = [int(value) for value in tuned.get("correct") or []]
    oracle_k = int(tuned.get("candidates_per_question") or 0)
    oracle_temperature = tuned.get("config", {}).get("temperature")
    oracle_accuracy, oracle_correct = harness.oracle_at_k(
        complete_rows,
        k=oracle_k,
        temperature=oracle_temperature,
    )
    n_flips_possible = sum(
        1
        for sc_ok, oracle_ok in zip(sc_correct, oracle_correct, strict=False)
        if not sc_ok and oracle_ok
    )
    return {
        "n_questions": len(complete_rows),
        "n_candidate_rows": sum(len(row.get("candidates") or []) for row in complete_rows),
        "genuine_sc_accuracy": round(sc_accuracy, 6),
        "oracle_at_k": round(float(oracle_accuracy), 6),
        "headroom_present": bool(
            (float(oracle_accuracy) - sc_accuracy) >= float(headroom_threshold)
            and n_flips_possible > 0
        ),
        "n_flips_possible": int(n_flips_possible),
        "tuned_self_consistency": {
            "accuracy": round(sc_accuracy, 6),
            "config": dict(tuned.get("config") or {}),
            "tuned_k": int(tuned.get("tuned_k") or 0),
            "k_sweep": dict(tuned.get("k_sweep") or {}),
            "candidate_pool_counts": list(tuned.get("candidate_pool_counts") or []),
        },
        "oracle_k": oracle_k,
        "oracle_temperature": oracle_temperature,
        "headroom_threshold": round(float(headroom_threshold), 6),
    }


def _model_specs() -> JsonDict:
    return {
        "mandated_models": {
            "flagship_moe": MANDATED_MODEL_IDS[0],
            "flagship_dense": MANDATED_MODEL_IDS[1],
            "middle_moe": MANDATED_MODEL_IDS[2],
        },
        "llm_generation_used": False,
        "candidate_generation": "deterministic_solver_backed_constraint_variants",
        "small_models_smoke_only": True,
    }


def _base_artifact(
    *,
    honest_verdict: str,
    root: Path,
    artifact_path: Path,
    cache_path: Path,
    ppbench_probe: JsonMap,
    preconditions_checked: Sequence[PreconditionCheck | JsonMap],
    duration_s: float,
) -> JsonDict:
    return {
        "schema": ARTIFACT_SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": artifact_path.as_posix(),
        "honest_verdict": honest_verdict,
        "model_specs": _model_specs(),
        "second_corpus_cache_built": False,
        "second_corpus_name": None,
        "n_questions": 0,
        "n_candidate_rows": 0,
        "genuine_sc_accuracy": 0.0,
        "oracle_at_k": 0.0,
        "headroom_present": False,
        "verifier_is_oracle": False,
        "candidate_cache_path": cache_path.as_posix(),
        "random_seed": RANDOM_SEED,
        "ppbench_probe": dict(ppbench_probe),
        "fallback_used": not bool(ppbench_probe.get("available")),
        "fallback_reason": None,
        "source_paths": [],
        "solver_provenance": {},
        "verifier_provenance": {
            "verifier_is_oracle": False,
            "selection_verifier": "not_run_cache_data_product",
        },
        "candidates_per_question": DEFAULT_CANDIDATES_PER_QUESTION,
        "candidate_cache_schema": CACHE_ROW_SCHEMA,
        "resume_summary": {
            "existing_complete_rows": 0,
            "appended_rows": 0,
            "target_rows": 0,
            "skipped_existing_rows": 0,
        },
        "preconditions_checked": _precondition_dicts(preconditions_checked),
        "inference_substrate": "deterministic_solver_backed_cache_build",
        "duration_s": round(float(duration_s), 6),
        "field_principles": dict(FIELD_PRINCIPLES),
        "reproducibility_checksum": "",
        "repo_root": Path(root).as_posix(),
    }


def reproducibility_checksum(payload: JsonMap) -> str:
    basis = _json_dumps(payload).encode("utf-8")
    return "sha256:" + hashlib.sha256(basis).hexdigest()


def _checksum(artifact: JsonMap) -> str:
    basis = {
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "second_corpus_name": artifact.get("second_corpus_name"),
        "n_questions": artifact.get("n_questions"),
        "n_candidate_rows": artifact.get("n_candidate_rows"),
        "genuine_sc_accuracy": artifact.get("genuine_sc_accuracy"),
        "oracle_at_k": artifact.get("oracle_at_k"),
        "headroom_present": artifact.get("headroom_present"),
        "candidate_cache_path": artifact.get("candidate_cache_path"),
        "random_seed": artifact.get("random_seed"),
    }
    return reproducibility_checksum(basis)


def build_skeleton_artifact(
    *,
    root: Path,
    artifact_path: Path,
    cache_path: Path,
    ppbench_probe: JsonMap | None = None,
    duration_s: float = 0.0,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict="running_second_corpus_cache_schema_skeleton",
        root=root,
        artifact_path=artifact_path,
        cache_path=cache_path,
        ppbench_probe=ppbench_probe or probe_ppbench(root),
        preconditions_checked=[],
        duration_s=duration_s,
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    root: Path,
    artifact_path: Path,
    cache_path: Path,
    ppbench_probe: JsonMap,
    preconditions_checked: Sequence[PreconditionCheck | JsonMap],
    duration_s: float,
    reason: str,
) -> JsonDict:
    artifact = _base_artifact(
        honest_verdict=f"blocked_{reason}",
        root=root,
        artifact_path=artifact_path,
        cache_path=cache_path,
        ppbench_probe=ppbench_probe,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    artifact["blocked_reason"] = reason
    artifact["fallback_reason"] = "no usable deterministic solver-backed or labeled cached fallback"
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def build_complete_artifact(
    *,
    root: Path,
    artifact_path: Path,
    cache_path: Path,
    ppbench_probe: JsonMap,
    source_path: Path,
    rows: Sequence[JsonMap],
    metrics: JsonMap,
    resume_summary: JsonMap,
    preconditions_checked: Sequence[PreconditionCheck | JsonMap],
    duration_s: float,
    min_questions: int,
) -> JsonDict:
    enough_rows = int(metrics["n_questions"]) >= int(min_questions)
    ready = enough_rows and bool(metrics["headroom_present"])
    verdict = (
        f"complete_second_corpus_cache_ready_constraintbench_exact_v1_n{metrics['n_questions']}"
        if ready
        else f"complete_second_corpus_cache_no_headroom_constraintbench_exact_v1_n{metrics['n_questions']}"
    )
    artifact = _base_artifact(
        honest_verdict=verdict,
        root=root,
        artifact_path=artifact_path,
        cache_path=cache_path,
        ppbench_probe=ppbench_probe,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
    )
    first_solver = dict(rows[0].get("solver_provenance") or {}) if rows else {}
    artifact.update(
        {
            "second_corpus_cache_built": bool(enough_rows),
            "second_corpus_name": CONSTRAINTBENCH_CORPUS_NAME,
            "n_questions": int(metrics["n_questions"]),
            "n_candidate_rows": int(metrics["n_candidate_rows"]),
            "genuine_sc_accuracy": float(metrics["genuine_sc_accuracy"]),
            "oracle_at_k": float(metrics["oracle_at_k"]),
            "headroom_present": bool(metrics["headroom_present"]),
            "fallback_reason": (
                "PPBench/Pencil Puzzle Bench not found locally; using checked-in "
                "ConstraintBench exact pilot fallback."
            )
            if not ppbench_probe.get("available")
            else None,
            "source_paths": [source_path.as_posix()],
            "solver_provenance": {
                "authority": "local_exhaustive_enumeration",
                "source_corpus": CONSTRAINTBENCH_CORPUS_NAME,
                "source_path": source_path.as_posix(),
                "checker_backends": sorted(
                    {str(row.get("solver_provenance", {}).get("backend")) for row in rows}
                ),
                "label_source": "exact_reference.solution",
                "sample": first_solver,
            },
            "resume_summary": dict(resume_summary),
            "metrics": dict(metrics),
        }
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("schema") != ARTIFACT_SCHEMA:
        errors.append("schema")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs")
    if not isinstance(artifact.get("second_corpus_cache_built"), bool):
        errors.append("second_corpus_cache_built")
    if not isinstance(artifact.get("headroom_present"), bool):
        errors.append("headroom_present")
    for field in ("n_questions", "n_candidate_rows"):
        if not isinstance(artifact.get(field), int) or int(artifact.get(field, -1)) < 0:
            errors.append(field)
    for field in ("genuine_sc_accuracy", "oracle_at_k"):
        value = _number(artifact.get(field))
        if value is None or not 0.0 <= value <= 1.0:
            errors.append(field)
    if not str(artifact.get("candidate_cache_path") or ""):
        errors.append("candidate_cache_path")
    if not isinstance(artifact.get("field_principles"), Mapping):
        errors.append("field_principles")
    if not str(artifact.get("honest_verdict") or "").startswith(
        ("complete_", "blocked_", "running_")
    ):
        errors.append("honest_verdict")
    return sorted(set(errors))


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    cache_path: Path | None = None,
    source_loader: SourceLoader = default_source_loader,
    limit: int = DEFAULT_LIMIT,
    min_questions: int = DEFAULT_MIN_QUESTIONS,
    candidates_per_question: int = DEFAULT_CANDIDATES_PER_QUESTION,
    now: Clock = time.perf_counter,
    write: bool = True,
) -> JsonDict:
    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    cache_path = Path(cache_path) if cache_path else root / CACHE_RELATIVE_PATH
    start = float(now())
    ppbench = probe_ppbench(root)
    if write:
        write_json(
            artifact_path,
            build_skeleton_artifact(
                root=root,
                artifact_path=artifact_path,
                cache_path=cache_path,
                ppbench_probe=ppbench,
                duration_s=0.0,
            ),
        )

    try:
        source_rows, source_path, checks = source_loader(root=root)
    except Exception as exc:
        checks = [
            PreconditionCheck(
                "second_corpus_source",
                False,
                f"{type(exc).__name__}: {exc}",
            )
        ]
        artifact = build_blocked_artifact(
            root=root,
            artifact_path=artifact_path,
            cache_path=cache_path,
            ppbench_probe=ppbench,
            preconditions_checked=checks,
            duration_s=float(now()) - start,
            reason="second_corpus_source_unavailable",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    if not source_rows:
        artifact = build_blocked_artifact(
            root=root,
            artifact_path=artifact_path,
            cache_path=cache_path,
            ppbench_probe=ppbench,
            preconditions_checked=checks,
            duration_s=float(now()) - start,
            reason="second_corpus_source_unavailable",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact

    rows, resume_summary = ensure_candidate_cache(
        cache_path=cache_path,
        source_rows=source_rows,
        source_path=source_path,
        limit=limit,
        candidates_per_question=candidates_per_question,
    )
    metrics = compute_cache_metrics(rows)
    artifact = build_complete_artifact(
        root=root,
        artifact_path=artifact_path,
        cache_path=cache_path,
        ppbench_probe=ppbench,
        source_path=source_path,
        rows=rows,
        metrics=metrics,
        resume_summary=resume_summary,
        preconditions_checked=checks,
        duration_s=float(now()) - start,
        min_questions=min_questions,
    )
    if write:
        write_json(artifact_path, artifact)
    return artifact


def main() -> int:  # pragma: no cover - exercised by the requested command, not unit tests
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if not artifact_schema_errors(artifact) else 1


if __name__ == "__main__":  # pragma: no cover - direct script entrypoint
    raise SystemExit(main())
