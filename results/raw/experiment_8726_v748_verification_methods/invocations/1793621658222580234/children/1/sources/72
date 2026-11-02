"""Exp 4087 GAP-5 certification precision-rescue sweep.

Spec refs: REQ-LEARN-4087, SCENARIO-LEARN-4087,
SCENARIO-LEARN-4087-FAIL.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # REQ-LEARN-4087-1: reuse the existing GAP-5 safe transform compiler.
    from carnot.agentic.gap5_cross_example_selector import safe_transform_from_code

    _SAFE_TRANSFORM_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - exercised only when the dependency is absent
    safe_transform_from_code = None  # type: ignore[assignment]
    _SAFE_TRANSFORM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_FILENAME = "experiment_4087_certification_precision_rescue.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_ARC1_POOL = Path("results/arc3_gap3_stage2_eval_pool.json.gz")
DEFAULT_ARC2_POOL = Path("results/arc3_gap4_arc2_eval_pool.json.gz")
DEFAULT_ARC1_PROGRAMS = Path("results/arc3_gap4_induced_programs.json")
DEFAULT_ARC2_PROGRAMS = Path("results/arc3_gap4_arc2_induced_programs.json")
DEFAULT_ENSEMBLES = (
    Path("results/arc3_gap4_arc2_consistency_ensemble.json"),
    Path("results/arc3_gap4_arc2_chain_ensemble.json"),
)
DEFAULT_VERIFIER_GAPS = REPO_ROOT / "ops" / "verifier_gaps.md"
INFERENCE_SUBSTRATE = "offline_saved_gap4_program_replay_precision_rescue"
RANDOM_SEED = 4087
PRECISION_THRESHOLD = 0.85
RECALL_THRESHOLD = 0.20
TERMINAL_PREFIXES = ("complete:", "success:", "blocked_")
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "precision_rescue_succeeded",
    "best_certified_precision",
    "best_op_point_recall",
    "frontier",
    "n_tasks_scored",
    "n_codex_calls",
    "random_seed",
    "reproducibility_checksum",
    "inference_substrate",
)
_GAP5_START = "<!-- exp4087-gap5:start -->"
_GAP5_END = "<!-- exp4087-gap5:end -->"
_COLOR_PERMUTATIONS = (
    tuple(range(10)),
    (0, 2, 3, 4, 5, 6, 7, 8, 9, 1),
    (0, 9, 8, 7, 6, 5, 4, 3, 2, 1),
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One offline resource check recorded before replay."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TaskRecord:
    """One ARC task with public demos, test input, and held-out gold labels."""

    task_id: str
    pool_name: str
    demos: list[dict[str, Any]]
    test_input: Any
    gold_outputs: list[Any]

    @property
    def task_key(self) -> str:
        return f"{self.pool_name}:{self.task_id}"


@dataclass(frozen=True)
class CandidateProgram:
    """Saved transform program from a cached GAP-4 pool."""

    task_key: str
    task_id: str
    pool_name: str
    program_id: str
    code: str
    source: str


@dataclass(frozen=True)
class CandidateReplay:
    """Deterministic replay outcome for one saved program."""

    task_key: str
    task_id: str
    pool_name: str
    program_id: str
    source: str
    code_hash: str
    compile_ok: bool
    demo_perfect: bool
    augmentation_invariant: bool
    prediction_hash: str | None
    prediction: Any
    test_gold: bool
    min_hamming_energy: float
    error: str


def _load_json_or_gzip(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _grid_array(grid: Any) -> np.ndarray:
    return np.asarray(grid, dtype=np.int64)


def _grid_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    left_arr = _grid_array(left)
    right_arr = _grid_array(right)
    return left_arr.shape == right_arr.shape and bool(np.array_equal(left_arr, right_arr))


def _grid_hash(grid: Any) -> str | None:
    if grid is None:
        return None
    arr = _grid_array(grid)
    payload = repr(tuple(arr.shape)).encode("ascii") + arr.tobytes()
    return hashlib.sha1(payload).hexdigest()


def _code_hash(code: str) -> str:
    return hashlib.sha1(code.strip().encode("utf-8")).hexdigest()


def _call_transform(fn: Any, grid: Any) -> tuple[Any, str]:
    try:
        out = fn(grid)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if out is None:
        return None, "transform returned None"
    return _grid_array(out).astype(int).tolist(), ""


def _cell_disagreement(prediction: Any, expected: Any) -> float:
    if prediction is None:
        return 1.0
    pred = _grid_array(prediction)
    exp = _grid_array(expected)
    if pred.shape != exp.shape or pred.size == 0:
        return 1.0
    return float(np.mean(pred != exp))


def _d4_variants(grid: Any) -> list[list[list[int]]]:
    arr = _grid_array(grid)
    variants = []
    for turns in range(4):
        rotated = np.rot90(arr, turns)
        variants.append(rotated.astype(int).tolist())
        variants.append(np.fliplr(rotated).astype(int).tolist())
    return variants


def _permute_colors(grid: Any, permutation: Sequence[int]) -> list[list[int]]:
    arr = _grid_array(grid)
    table = np.asarray(permutation, dtype=np.int64)
    return table[arr].astype(int).tolist()


def _augmented_demo_pairs(demo: Mapping[str, Any]) -> list[tuple[Any, Any]]:
    pairs: list[tuple[Any, Any]] = []
    for permutation in _COLOR_PERMUTATIONS:
        input_grid = _permute_colors(demo["input"], permutation)
        output_grid = _permute_colors(demo["output"], permutation)
        for aug_input, aug_output in zip(_d4_variants(input_grid), _d4_variants(output_grid), strict=True):
            pairs.append((aug_input, aug_output))
    return pairs


def _augmentation_invariant(fn: Any, demos: Sequence[Mapping[str, Any]]) -> bool:
    if fn is None or not demos:
        return False
    for demo in demos:
        for aug_input, aug_output in _augmented_demo_pairs(demo):
            prediction, _error = _call_transform(fn, aug_input)
            if not _grid_equal(prediction, aug_output):
                return False
    return True


def replay_program(task: TaskRecord, program: CandidateProgram) -> CandidateReplay:
    """REQ-LEARN-4087-2/3/5: replay one saved transform without oracle-guided filtering."""

    fn = safe_transform_from_code(program.code) if safe_transform_from_code is not None else None
    errors: list[str] = []
    if fn is None:
        errors.append(_SAFE_TRANSFORM_IMPORT_ERROR or "compile_failed")
    demo_energies: list[float] = []
    demo_matches: list[bool] = []
    if fn is not None:
        for index, demo in enumerate(task.demos):
            prediction, error = _call_transform(fn, demo.get("input"))
            if error:
                errors.append(f"demo{index}:{error}")
            energy = _cell_disagreement(prediction, demo.get("output"))
            demo_energies.append(energy)
            demo_matches.append(energy == 0.0)
        test_prediction, test_error = _call_transform(fn, task.test_input)
        if test_error:
            errors.append(f"test:{test_error}")
    else:
        test_prediction = None
    demo_perfect = bool(task.demos and demo_matches and all(demo_matches))
    invariant = bool(demo_perfect and _augmentation_invariant(fn, task.demos))
    test_gold = bool(any(_grid_equal(test_prediction, gold) for gold in task.gold_outputs))
    return CandidateReplay(
        task_key=task.task_key,
        task_id=task.task_id,
        pool_name=task.pool_name,
        program_id=program.program_id,
        source=program.source,
        code_hash=_code_hash(program.code),
        compile_ok=fn is not None,
        demo_perfect=demo_perfect,
        augmentation_invariant=invariant,
        prediction_hash=_grid_hash(test_prediction),
        prediction=test_prediction,
        test_gold=test_gold,
        min_hamming_energy=round(min(demo_energies), 4) if demo_energies else 1.0,
        error="; ".join(errors),
    )


def replay_dataset(
    tasks: Mapping[str, TaskRecord],
    programs_by_task: Mapping[str, Sequence[CandidateProgram]],
) -> list[CandidateReplay]:
    rows: list[CandidateReplay] = []
    for task_key in sorted(tasks):
        task = tasks[task_key]
        for program in sorted(programs_by_task.get(task_key, ()), key=lambda item: item.program_id):
            rows.append(replay_program(task, program))
    return rows


def _certified_for_task(
    rows: Sequence[CandidateReplay],
    *,
    require_demo_perfect: bool,
    require_invariance: bool,
    min_agreement: int,
    tau: float | None,
) -> list[CandidateReplay]:
    buckets: dict[str, list[CandidateReplay]] = defaultdict(list)
    for row in rows:
        if row.prediction_hash is None:
            continue
        if require_demo_perfect and not row.demo_perfect:
            continue
        if require_invariance and not row.augmentation_invariant:
            continue
        if tau is not None and row.min_hamming_energy > tau + 1e-12:
            continue
        buckets[str(row.prediction_hash)].append(row)
    certified = []
    for hash_value in sorted(buckets):
        bucket = sorted(buckets[hash_value], key=lambda item: item.program_id)
        if len(bucket) >= min_agreement:
            certified.append(bucket[0])
    return certified


def _score_frontier_point(
    rows_by_task: Mapping[str, Sequence[CandidateReplay]],
    *,
    n_tasks_scored: int,
    filter_stack: str,
    threshold: str,
    require_demo_perfect: bool,
    require_invariance: bool,
    min_agreement: int,
    tau: float | None,
) -> dict[str, object]:
    certified = 0
    gold_predictions = 0
    gold_tasks = 0
    for rows in rows_by_task.values():
        selected_rows = _certified_for_task(
            rows,
            require_demo_perfect=require_demo_perfect,
            require_invariance=require_invariance,
            min_agreement=min_agreement,
            tau=tau,
        )
        certified += len(selected_rows)
        gold_predictions += sum(int(row.test_gold) for row in selected_rows)
        gold_tasks += int(any(row.test_gold for row in selected_rows))
    precision = gold_predictions / certified if certified else 0.0
    recall = gold_tasks / n_tasks_scored if n_tasks_scored else 0.0
    return {
        "filter_stack": filter_stack,
        "threshold": threshold,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "n_certified": int(certified),
    }


def _tau_values(rows: Sequence[CandidateReplay]) -> list[float]:
    values = {0.0, 1.0}
    values.update(round(float(row.min_hamming_energy), 4) for row in rows)
    return sorted(values)


def build_frontier(rows: Sequence[CandidateReplay], *, n_tasks_scored: int) -> list[dict[str, object]]:
    """REQ-LEARN-4087-4/5/6: sweep deterministic precision-recall operating points."""

    rows_by_task: dict[str, list[CandidateReplay]] = defaultdict(list)
    for row in rows:
        rows_by_task[row.task_key].append(row)
    max_agreement = max((len(task_rows) for task_rows in rows_by_task.values()), default=1)
    frontier: list[dict[str, object]] = []

    def add(
        filter_stack: str,
        threshold: str,
        *,
        require_demo_perfect: bool = True,
        require_invariance: bool = False,
        min_agreement: int = 1,
        tau: float | None = None,
    ) -> None:
        frontier.append(
            _score_frontier_point(
                rows_by_task,
                n_tasks_scored=n_tasks_scored,
                filter_stack=filter_stack,
                threshold=threshold,
                require_demo_perfect=require_demo_perfect,
                require_invariance=require_invariance,
                min_agreement=min_agreement,
                tau=tau,
            )
        )

    add("demo_perfect", "k=1")
    add("demo_perfect+invariance", "required", require_invariance=True)
    for k_value in range(1, max_agreement + 1):
        add("k_of_n_agreement", f"k={k_value}", min_agreement=k_value)
        add(
            "demo_perfect+invariance+agreement",
            f"k={k_value}",
            require_invariance=True,
            min_agreement=k_value,
        )
    for tau in _tau_values(rows):
        add(
            "graded_min_hamming",
            f"tau={tau:.4f}",
            require_demo_perfect=False,
            tau=tau,
        )
    for k_value in range(1, max_agreement + 1):
        for tau in _tau_values(rows):
            add(
                "demo_perfect+invariance+agreement+min_hamming",
                f"k={k_value},tau={tau:.4f}",
                require_invariance=True,
                min_agreement=k_value,
                tau=tau,
            )
    return frontier


def _best_point(frontier: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    candidates = [row for row in frontier if float(row.get("recall", 0.0)) >= RECALL_THRESHOLD]
    candidates = candidates or list(frontier)
    if not candidates:
        return {
            "filter_stack": "none",
            "threshold": "none",
            "precision": 0.0,
            "recall": 0.0,
            "n_certified": 0,
        }
    return dict(
        max(
            candidates,
            key=lambda row: (
                float(row.get("precision", 0.0)),
                float(row.get("recall", 0.0)),
                int(row.get("n_certified", 0)),
                str(row.get("filter_stack", "")),
                str(row.get("threshold", "")),
            ),
        )
    )


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    checksum_fields = {
        key: artifact[key]
        for key in REQUIRED_ARTIFACT_FIELDS
        if key != "reproducibility_checksum"
    }
    encoded = json.dumps(checksum_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "Terminal-prefix outcome for the offline precision rescue gate.",
        "precision_rescue_succeeded": "Bare bool gate for whether Phase B RFT may proceed.",
        "best_certified_precision": "Headline certified precision reached by the selected operating point.",
        "best_op_point_recall": "Recall paired with the headline precision point.",
        "frontier": "Full deterministic precision-recall curve for independent threshold selection.",
        "n_tasks_scored": "Number of held-out ARC tasks replayed from cached pools.",
        "n_codex_calls": "Must be 0 because this is offline saved-program replay.",
        "random_seed": "Fixed seed recorded for reproducible ordering and checksums.",
        "reproducibility_checksum": "SHA-256 over the stable required artifact fields.",
        "inference_substrate": "Declares offline replay rather than live generation or GPU inference.",
    }


def build_artifact(
    frontier: Sequence[Mapping[str, Any]],
    *,
    n_tasks_scored: int,
    preconditions_checked: Sequence[PreconditionCheck] = (),
) -> dict[str, Any]:
    success_points = [
        row
        for row in frontier
        if float(row.get("precision", 0.0)) >= PRECISION_THRESHOLD
        and float(row.get("recall", 0.0)) >= RECALL_THRESHOLD
    ]
    rescue_succeeded = bool(success_points)
    best = _best_point(success_points or frontier)
    best_precision = round(float(best["precision"]), 4)
    best_recall = round(float(best["recall"]), 4)
    if rescue_succeeded:
        verdict = f"complete: precision_rescue_succeeded_best_{best_precision:.4f}_at_recall_{best_recall:.4f}"
    else:
        verdict = (
            f"complete: precision_rescue_FAILED_max_{best_precision:.4f}_"
            "verifier_as_reward_arc_precision_bounded"
        )
    artifact: dict[str, Any] = {
        "experiment": "experiment_4087_certification_precision_rescue",
        "schema": "carnot.experiment_4087_certification_precision_rescue.v1",
        "honest_verdict": verdict,
        "precision_rescue_succeeded": rescue_succeeded,
        "best_certified_precision": best_precision,
        "best_op_point_recall": best_recall,
        "best_operating_point": best,
        "frontier": [dict(row) for row in frontier],
        "n_tasks_scored": int(n_tasks_scored),
        "n_codex_calls": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": _field_principles(),
        "preconditions_checked": [check.to_dict() for check in preconditions_checked],
        "spec_refs": ["REQ-LEARN-4087", "SCENARIO-LEARN-4087", "SCENARIO-LEARN-4087-FAIL"],
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - builder fields are schema-valid by construction
        raise ValueError("; ".join(errors))
    return artifact


def build_blocked_artifact(
    blocker: str,
    *,
    preconditions_checked: Sequence[PreconditionCheck],
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "experiment": "experiment_4087_certification_precision_rescue",
        "schema": "carnot.experiment_4087_certification_precision_rescue.v1",
        "honest_verdict": blocker,
        "precision_rescue_succeeded": False,
        "best_certified_precision": 0.0,
        "best_op_point_recall": 0.0,
        "best_operating_point": {
            "filter_stack": "none",
            "threshold": "blocked",
            "precision": 0.0,
            "recall": 0.0,
            "n_certified": 0,
        },
        "frontier": [],
        "n_tasks_scored": 0,
        "n_codex_calls": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": _field_principles(),
        "preconditions_checked": [check.to_dict() for check in preconditions_checked],
        "spec_refs": ["REQ-LEARN-4087"],
    }
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - builder fields are schema-valid by construction
        raise ValueError("; ".join(errors))
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")
    if not isinstance(artifact.get("precision_rescue_succeeded"), bool):
        errors.append("precision_rescue_succeeded must be a bare bool")
    for field in ("best_certified_precision", "best_op_point_recall"):
        if not isinstance(artifact.get(field), float) or isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare float")
    for field in ("n_tasks_scored", "n_codex_calls", "random_seed"):
        if not isinstance(artifact.get(field), int) or isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare int")
    if artifact.get("n_codex_calls") != 0:
        errors.append("n_codex_calls must be 0 for offline replay")
    frontier = artifact.get("frontier")
    if not isinstance(frontier, list):
        errors.append("frontier must be a list")
    else:
        required = {"filter_stack", "threshold", "precision", "recall", "n_certified"}
        for row in frontier:
            if not isinstance(row, Mapping) or not required.issubset(row):
                errors.append("frontier entries must include filter_stack, threshold, precision, recall, n_certified")
                break
    if not isinstance(artifact.get("reproducibility_checksum"), str):
        errors.append("reproducibility_checksum must be a string")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare offline replay precision rescue")
    return errors


def _check_loadable(path: Path, resource: str) -> PreconditionCheck:
    if not path.exists() or path.stat().st_size <= 0:
        return PreconditionCheck(resource, False, f"missing or empty: {path}")
    try:
        payload = _load_json_or_gzip(path)
    except Exception as exc:
        return PreconditionCheck(resource, False, f"{type(exc).__name__}: {exc}")
    if isinstance(payload, Mapping) and not payload:
        return PreconditionCheck(resource, False, f"empty object: {path}")
    return PreconditionCheck(resource, True, f"loaded {path}")


def check_preconditions(*, repo_root: str | Path = REPO_ROOT) -> tuple[list[PreconditionCheck], str | None]:
    root = Path(repo_root)
    checks = [
        _check_loadable(root / DEFAULT_ARC1_PROGRAMS, "arc1_induced_programs"),
        _check_loadable(root / DEFAULT_ARC2_PROGRAMS, "arc2_induced_programs"),
        _check_loadable(root / DEFAULT_ARC1_POOL, "arc1_eval_pool"),
        _check_loadable(root / DEFAULT_ARC2_POOL, "arc2_eval_pool"),
    ]
    checks.append(
        PreconditionCheck(
            "safe_transform_from_code",
            callable(safe_transform_from_code),
            "importable via gap5_cross_example_selector" if callable(safe_transform_from_code) else _SAFE_TRANSFORM_IMPORT_ERROR,
        )
    )
    missing = next((check for check in checks if not check.available), None)
    return checks, f"blocked_{missing.resource}" if missing else None


def load_arc_pool(path: str | Path, *, pool_name: str) -> dict[str, TaskRecord]:
    payload = _load_json_or_gzip(Path(path))
    entries = payload.get("entries") if isinstance(payload, Mapping) else payload
    if not isinstance(entries, list):
        raise ValueError(f"{path} does not contain an entries list")
    tasks: dict[str, TaskRecord] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        task_id = str(entry.get("task") or entry.get("task_id") or "")
        gold_outputs = [
            candidate.get("grid")
            for candidate in entry.get("candidates", [])
            if isinstance(candidate, Mapping) and candidate.get("correct") is True and "grid" in candidate
        ]
        if task_id and gold_outputs:
            record = TaskRecord(
                task_id=task_id,
                pool_name=pool_name,
                demos=list(entry.get("demos") or []),
                test_input=entry.get("test_input"),
                gold_outputs=gold_outputs,
            )
            tasks[record.task_key] = record
    if not tasks:
        raise ValueError(f"{path} contains no usable ARC tasks with gold labels")
    return tasks


def load_induced_programs(path: str | Path, *, pool_name: str) -> dict[str, list[CandidateProgram]]:
    payload = _load_json_or_gzip(Path(path))
    rows = payload.get("programs") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a programs list")
    programs: dict[str, list[CandidateProgram]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("code"), str):
            continue
        task_id = str(row.get("task") or row.get("task_id") or "")
        if not task_id:
            continue
        task_key = f"{pool_name}:{task_id}"
        programs[task_key].append(
            CandidateProgram(
                task_key=task_key,
                task_id=task_id,
                pool_name=pool_name,
                program_id=f"{pool_name}:induced:{task_id}:{index}",
                code=str(row["code"]),
                source=str(row.get("source") or Path(path).stem),
            )
        )
    return dict(programs)


def load_ensemble_programs(path: str | Path, *, pool_name: str = "arc2") -> dict[str, list[CandidateProgram]]:
    payload = _load_json_or_gzip(Path(path))
    if not isinstance(payload, Mapping):
        return {}
    if isinstance(payload.get("part_b_agreement"), Mapping):
        task_rows = payload["part_b_agreement"].get("per_task", [])
    else:
        task_rows = payload.get("per_task", [])
    programs: dict[str, list[CandidateProgram]] = defaultdict(list)
    for task_row in task_rows if isinstance(task_rows, list) else []:
        if not isinstance(task_row, Mapping):
            continue
        task_id = str(task_row.get("task") or "")
        samples = task_row.get("samples") or task_row.get("arms") or []
        if not task_id or not isinstance(samples, list):
            continue
        task_key = f"{pool_name}:{task_id}"
        for index, sample in enumerate(samples):
            if not isinstance(sample, Mapping) or not isinstance(sample.get("code"), str):
                continue
            source = str(sample.get("source") or f"{Path(path).stem}:{index}")
            programs[task_key].append(
                CandidateProgram(
                    task_key=task_key,
                    task_id=task_id,
                    pool_name=pool_name,
                    program_id=f"{pool_name}:ensemble:{Path(path).stem}:{task_id}:{source}:{index}",
                    code=str(sample["code"]),
                    source=source,
                )
            )
    return dict(programs)


def _merge_programs(
    target: dict[str, list[CandidateProgram]],
    incoming: Mapping[str, Sequence[CandidateProgram]],
) -> None:
    for task_key, rows in incoming.items():
        seen = {_code_hash(row.code) for row in target.get(task_key, [])}
        for row in rows:
            digest = _code_hash(row.code)
            if digest in seen:
                continue
            target.setdefault(task_key, []).append(row)
            seen.add(digest)


def load_default_dataset(
    *,
    repo_root: str | Path = REPO_ROOT,
) -> tuple[dict[str, TaskRecord], dict[str, list[CandidateProgram]]]:
    root = Path(repo_root)
    tasks = {}
    tasks.update(load_arc_pool(root / DEFAULT_ARC1_POOL, pool_name="arc1"))
    tasks.update(load_arc_pool(root / DEFAULT_ARC2_POOL, pool_name="arc2"))
    programs: dict[str, list[CandidateProgram]] = {}
    _merge_programs(programs, load_induced_programs(root / DEFAULT_ARC1_PROGRAMS, pool_name="arc1"))
    _merge_programs(programs, load_induced_programs(root / DEFAULT_ARC2_PROGRAMS, pool_name="arc2"))
    for ensemble in DEFAULT_ENSEMBLES:
        path = root / ensemble
        if path.exists():
            _merge_programs(programs, load_ensemble_programs(path, pool_name="arc2"))
    scoped_tasks = {task_key: task for task_key, task in tasks.items() if programs.get(task_key)}
    scoped_programs = {task_key: rows for task_key, rows in programs.items() if task_key in scoped_tasks}
    if not scoped_tasks:
        raise ValueError("no cached programs match recoverable ARC tasks")
    return scoped_tasks, scoped_programs


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_gap5_precision_rescue_entry(path: str | Path, artifact: Mapping[str, Any]) -> None:
    gaps_path = Path(path)
    text = gaps_path.read_text(encoding="utf-8") if gaps_path.exists() else ""
    if _GAP5_START in text:
        return
    status = "precision_rescue_succeeded" if artifact["precision_rescue_succeeded"] else "precision_rescue_failed"
    entry = (
        f"{_GAP5_START}\n"
        "#### Exp 4087 certification precision-rescue update for GAP-5\n"
        f"- status: {status}\n"
        f"- evidence: `results/{RESULT_FILENAME}`; "
        f"best_certified_precision={artifact['best_certified_precision']:.4f}; "
        f"best_op_point_recall={artifact['best_op_point_recall']:.4f}; "
        f"n_tasks_scored={artifact['n_tasks_scored']}; n_codex_calls={artifact['n_codex_calls']}.\n"
        f"- outcome: {artifact['honest_verdict']}.\n"
        "- implication: GAP-5 demo-underdetermination remains the Phase B RFT gate unless the "
        "recorded frontier contains a qualifying precision/recall operating point.\n"
        f"{_GAP5_END}\n"
    )
    gaps_path.parent.mkdir(parents=True, exist_ok=True)
    gaps_path.write_text(text.rstrip() + "\n\n" + entry, encoding="utf-8")


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path | None = None,
    verifier_gaps_path: str | Path = DEFAULT_VERIFIER_GAPS,
    update_verifier_gaps: bool = True,
) -> dict[str, Any]:
    checks, blocker = check_preconditions(repo_root=repo_root)
    if blocker:
        artifact = build_blocked_artifact(blocker, preconditions_checked=checks)
    else:
        tasks, programs = load_default_dataset(repo_root=repo_root)
        rows = replay_dataset(tasks, programs)
        frontier = build_frontier(rows, n_tasks_scored=len(tasks))
        artifact = build_artifact(frontier, n_tasks_scored=len(tasks), preconditions_checked=checks)
        artifact["n_programs_replayed"] = len(rows)
        artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - run_experiment only receives builder-valid artifacts
        raise ValueError("; ".join(errors))
    destination = Path(output_path) if output_path is not None else Path(repo_root) / "results" / RESULT_FILENAME
    _write_json(destination, artifact)
    if update_verifier_gaps:
        append_gap5_precision_rescue_entry(verifier_gaps_path, artifact)
    return artifact
