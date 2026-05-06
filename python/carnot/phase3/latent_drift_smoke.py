"""EBRM-style latent trajectory drift smoke test for Phase 3/Kona planning.

The diagnostic is intentionally tiny and deterministic.  It models the failure
reported in arXiv 2603.28248 at smoke-test scale: a continuous planner can lower
its own latent energy while moving away from the support region where the decoder
was trained.  Carnot needs this measurement before treating lower latent energy
as a reliable proxy for better decoded reasoning.

Spec refs: REQ-KONA-033, SCENARIO-KONA-033.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

RUN_DATE = "20260506"
RESULT_PATH = Path("results/experiment_1417_ebrm_latent_trajectory_drift_smoke.json")
DEFAULT_SUPPORT_RADIUS = 0.75
DEFAULT_PLANNING_STEPS = 12
DEFAULT_PLANNING_LR = 0.20
DEFAULT_ANCHOR_WEIGHT = 0.02
FALLBACK_ASSIGNMENT = (False, False)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "latent_drift_smoke_complete",
    "task_family",
    "energy_monotone",
    "accuracy_before_planning",
    "accuracy_after_planning",
    "accuracy_delta_after_planning",
    "latent_drift_norm",
    "dual_path_decoder_required",
    "anchoring_required",
    "honest_verdict",
)
ASSIGNMENT_ANCHORS = {
    (False, False): np.array([-1.0, -1.0], dtype=np.float64),
    (False, True): np.array([-1.0, 1.0], dtype=np.float64),
    (True, False): np.array([1.0, -1.0], dtype=np.float64),
    (True, True): np.array([1.0, 1.0], dtype=np.float64),
}


@dataclass(frozen=True)
class CNFTask:
    """A two-variable CNF row with a unique satisfying assignment."""

    task_id: str
    clauses: tuple[tuple[int, ...], ...]
    target_assignment: tuple[bool, bool]

    def satisfied_by(self, assignment: tuple[bool, bool]) -> bool:
        """Return whether the decoded assignment satisfies all CNF clauses."""
        return all(
            any(
                assignment[abs(literal) - 1] if literal > 0 else not assignment[abs(literal) - 1]
                for literal in clause
            )
            for clause in self.clauses
        )


@dataclass(frozen=True)
class DecodeResult:
    """Decoder output plus whether the latent was inside decoder support."""

    assignment: tuple[bool, bool]
    supported: bool
    distance_to_support: float


@dataclass(frozen=True)
class PlanningResult:
    """One planned latent trajectory from encoder output ``h_x`` to final ``z_T``."""

    h_x: np.ndarray
    z_t: np.ndarray
    energy_trace: tuple[float, ...]
    latent_drift_norm: float


@dataclass(frozen=True)
class SmokeMetrics:
    """Measured Exp 1417 smoke-test fields before JSON serialization."""

    status: str
    latent_drift_smoke_complete: bool
    task_family: str
    energy_monotone: bool
    accuracy_before_planning: float
    accuracy_after_planning: float
    accuracy_delta_after_planning: float
    latent_drift_norm: float
    dual_path_decoder_required: bool
    anchoring_required: bool
    honest_verdict: str
    energy_trace: tuple[float, ...]
    support_radius: float
    planned_support_fraction: float
    n_tasks: int


def build_tiny_cnf_tasks() -> tuple[CNFTask, ...]:
    """Build four exact-assignment CNF tasks over variables x1 and x2."""
    tasks: list[CNFTask] = []
    for index, assignment in enumerate(ASSIGNMENT_ANCHORS):
        clauses = tuple(
            ((variable_index + 1) if value else -(variable_index + 1),)
            for variable_index, value in enumerate(assignment)
        )
        tasks.append(
            CNFTask(
                task_id=f"cnf_{index}",
                clauses=clauses,
                target_assignment=assignment,
            )
        )
    return tuple(tasks)


def encode_task(task: CNFTask) -> np.ndarray:
    """Return the deterministic encoder output ``h_x`` for one CNF task."""
    return ASSIGNMENT_ANCHORS[task.target_assignment].copy()


def decode_latent(
    z: Iterable[float], support_radius: float = DEFAULT_SUPPORT_RADIUS
) -> DecodeResult:
    """Decode ``z`` only when it remains close enough to a known support anchor."""
    z_arr = np.asarray(tuple(z), dtype=np.float64)
    nearest_assignment, nearest_distance = min(
        (
            (assignment, float(np.linalg.norm(z_arr - anchor)))
            for assignment, anchor in ASSIGNMENT_ANCHORS.items()
        ),
        key=lambda row: (row[1], row[0]),
    )
    supported = nearest_distance <= support_radius
    assignment = nearest_assignment if supported else FALLBACK_ASSIGNMENT
    return DecodeResult(
        assignment=assignment,
        supported=bool(supported),
        distance_to_support=nearest_distance,
    )


def direct_decode_dataset(tasks: Iterable[CNFTask]) -> list[DecodeResult]:
    """Decode every encoder output without latent planning."""
    return [decode_latent(encode_task(task)) for task in tasks]


def planning_energy(
    z: Iterable[float],
    h_x: Iterable[float],
    anchor_weight: float = DEFAULT_ANCHOR_WEIGHT,
) -> float:
    """Energy that prefers a low-norm latent while weakly anchoring to ``h_x``."""
    z_arr = np.asarray(tuple(z), dtype=np.float64)
    h_arr = np.asarray(tuple(h_x), dtype=np.float64)
    compatibility = float(np.dot(z_arr, z_arr))
    anchor = float(anchor_weight * np.dot(z_arr - h_arr, z_arr - h_arr))
    return compatibility + anchor


def planning_gradient(
    z: Iterable[float],
    h_x: Iterable[float],
    anchor_weight: float = DEFAULT_ANCHOR_WEIGHT,
) -> np.ndarray:
    """Analytic gradient of the smoke-test planning energy."""
    z_arr = np.asarray(tuple(z), dtype=np.float64)
    h_arr = np.asarray(tuple(h_x), dtype=np.float64)
    return 2.0 * z_arr + (2.0 * anchor_weight * (z_arr - h_arr))


def plan_latent(
    h_x: Iterable[float],
    n_steps: int = DEFAULT_PLANNING_STEPS,
    lr: float = DEFAULT_PLANNING_LR,
    anchor_weight: float = DEFAULT_ANCHOR_WEIGHT,
) -> PlanningResult:
    """Run deterministic latent descent from encoder output to planned ``z_T``."""
    h_arr = np.asarray(tuple(h_x), dtype=np.float64)
    z_arr = h_arr.copy()
    trace = [planning_energy(z_arr, h_arr, anchor_weight)]
    for _ in range(n_steps):
        z_arr = z_arr - (lr * planning_gradient(z_arr, h_arr, anchor_weight))
        trace.append(planning_energy(z_arr, h_arr, anchor_weight))
    return PlanningResult(
        h_x=h_arr,
        z_t=z_arr,
        energy_trace=tuple(float(value) for value in trace),
        latent_drift_norm=float(np.linalg.norm(z_arr - h_arr)),
    )


def planned_decode_dataset(tasks: Iterable[CNFTask]) -> list[DecodeResult]:
    """Plan from each encoder output, then decode the final planned latent."""
    return [decode_latent(plan_latent(encode_task(task)).z_t) for task in tasks]


def accuracy_for_decodes(tasks: Iterable[CNFTask], decodes: Iterable[DecodeResult]) -> float:
    """Compute CNF satisfaction accuracy for a list of decoder outputs."""
    pairs = tuple(zip(tasks, decodes, strict=True))
    return float(sum(task.satisfied_by(decode.assignment) for task, decode in pairs) / len(pairs))


def mean_energy_trace(results: Iterable[PlanningResult]) -> tuple[float, ...]:
    """Average per-step planning energy across all trajectories."""
    rows = tuple(results)
    return tuple(
        float(np.mean([row.energy_trace[index] for row in rows]))
        for index in range(len(rows[0].energy_trace))
    )


def energy_trace_is_monotone(trace: Iterable[float], tolerance: float = 1e-12) -> bool:
    """Return true when each planning step does not increase mean energy."""
    values = tuple(float(value) for value in trace)
    return all(right <= left + tolerance for left, right in zip(values, values[1:]))


def mean_latent_drift_norm(results: Iterable[PlanningResult]) -> float:
    """Average ``||z_T - h_x||`` across planned trajectories."""
    rows = tuple(results)
    return float(np.mean([row.latent_drift_norm for row in rows]))


def run_smoke() -> SmokeMetrics:
    """Run direct decode and planned decode on the same tiny CNF dataset."""
    tasks = build_tiny_cnf_tasks()
    planning_results = [plan_latent(encode_task(task)) for task in tasks]
    direct_decodes = direct_decode_dataset(tasks)
    planned_decodes = [decode_latent(result.z_t) for result in planning_results]
    energy_trace = mean_energy_trace(planning_results)
    accuracy_before = accuracy_for_decodes(tasks, direct_decodes)
    accuracy_after = accuracy_for_decodes(tasks, planned_decodes)
    accuracy_delta = accuracy_after - accuracy_before
    energy_monotone = energy_trace_is_monotone(energy_trace)
    energy_lowered = energy_trace[-1] < energy_trace[0]
    latent_drift = mean_latent_drift_norm(planning_results)
    planned_support_fraction = float(
        sum(decode.supported for decode in planned_decodes) / len(tasks)
    )
    dual_path_required = bool(energy_lowered and accuracy_delta <= 0.0)
    anchoring_required = bool(latent_drift > DEFAULT_SUPPORT_RADIUS)
    verdict = (
        "energy_down_accuracy_down_off_decoder_support"
        if energy_lowered and accuracy_delta < 0.0 and anchoring_required
        else "no_ebrm_support_shift_detected"
    )
    return SmokeMetrics(
        status="complete",
        latent_drift_smoke_complete=True,
        task_family="synthetic_two_variable_cnf",
        energy_monotone=energy_monotone,
        accuracy_before_planning=accuracy_before,
        accuracy_after_planning=accuracy_after,
        accuracy_delta_after_planning=accuracy_delta,
        latent_drift_norm=latent_drift,
        dual_path_decoder_required=dual_path_required,
        anchoring_required=anchoring_required,
        honest_verdict=verdict,
        energy_trace=energy_trace,
        support_radius=DEFAULT_SUPPORT_RADIUS,
        planned_support_fraction=planned_support_fraction,
        n_tasks=len(tasks),
    )


def build_artifact() -> dict[str, Any]:
    """Build the complete Exp 1417 JSON artifact from measured smoke metrics."""
    metrics = run_smoke()
    artifact: dict[str, Any] = {
        "schema": "carnot.phase3.ebrm_latent_drift_smoke.v1",
        "experiment": "1417_ebrm_latent_trajectory_drift_smoke",
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-KONA-033", "SCENARIO-KONA-033"],
        "status": metrics.status,
        "latent_drift_smoke_complete": metrics.latent_drift_smoke_complete,
        "task_family": metrics.task_family,
        "energy_monotone": metrics.energy_monotone,
        "accuracy_before_planning": metrics.accuracy_before_planning,
        "accuracy_after_planning": metrics.accuracy_after_planning,
        "accuracy_delta_after_planning": metrics.accuracy_delta_after_planning,
        "latent_drift_norm": metrics.latent_drift_norm,
        "dual_path_decoder_required": metrics.dual_path_decoder_required,
        "anchoring_required": metrics.anchoring_required,
        "honest_verdict": metrics.honest_verdict,
        "energy_trace": list(metrics.energy_trace),
        "decoder_support_radius": metrics.support_radius,
        "planned_support_fraction": metrics.planned_support_fraction,
        "n_tasks": metrics.n_tasks,
    }
    return artifact


def write_experiment_artifact(path: Path | str = RESULT_PATH) -> dict[str, Any]:
    """Persist the complete Exp 1417 smoke artifact."""
    artifact = build_artifact()
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact
