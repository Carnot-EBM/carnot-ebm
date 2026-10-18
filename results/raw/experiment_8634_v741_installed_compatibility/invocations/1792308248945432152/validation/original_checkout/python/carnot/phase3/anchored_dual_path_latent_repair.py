"""Anchored dual-path latent repair smoke test for Phase 3/Kona planning.

Exp 1417 showed the dangerous case: a latent planner can lower its internal
energy while drifting outside the decoder support and hurting decoded accuracy.
This module keeps that same tiny CNF benchmark, then adds the smallest repair
guardrail worth testing: stronger anchoring to the encoder state plus a decoded
quality check before accepting a lower-energy latent candidate.

Spec refs: REQ-KONA-034, SCENARIO-KONA-034.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from carnot.phase3.latent_drift_smoke import (
    CNFTask,
    DEFAULT_PLANNING_LR,
    DEFAULT_PLANNING_STEPS,
    DEFAULT_SUPPORT_RADIUS,
    DecodeResult,
    PlanningResult,
    accuracy_for_decodes,
    build_tiny_cnf_tasks,
    decode_latent,
    direct_decode_dataset,
    encode_task,
    energy_trace_is_monotone,
    mean_energy_trace,
    mean_latent_drift_norm,
    plan_latent,
    planning_energy,
    planning_gradient,
)

RUN_DATE = "20260506"
RESULT_PATH = Path("results/experiment_1436_anchored_dual_path_latent_repair_v1.json")
ANCHORED_ANCHOR_WEIGHT = 1.0
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "anchoring_applied",
    "dual_path_decoder_stub",
    "energy_monotone",
    "accuracy_before_planning",
    "accuracy_after_planning",
    "accuracy_delta_after_planning",
    "latent_drift_norm",
    "off_support_rate",
    "anchored_repair_viable",
    "honest_verdict",
)


@dataclass(frozen=True)
class CandidateDecision:
    """Decision record for one proposed latent step."""

    accepted: bool
    energy_lowered: bool
    current_quality: float
    candidate_quality: float
    current_energy: float
    candidate_energy: float


@dataclass(frozen=True)
class AnchoredPlanningResult(PlanningResult):
    """One accepted anchored trajectory and its repair-gate diagnostics."""

    anchoring_applied: bool
    dual_path_decoder_stub: bool
    rejected_candidates: int


@dataclass(frozen=True)
class DescentMetrics:
    """Aggregated metrics for one descent policy on the tiny benchmark."""

    energy_monotone: bool
    accuracy_before_planning: float
    accuracy_after_planning: float
    accuracy_delta_after_planning: float
    latent_drift_norm: float
    off_support_rate: float
    energy_trace: tuple[float, ...]


@dataclass(frozen=True)
class RepairComparisonMetrics(DescentMetrics):
    """Exp 1436 measured fields plus the raw-descent baseline."""

    status: str
    anchoring_applied: bool
    dual_path_decoder_stub: bool
    anchored_repair_viable: bool
    honest_verdict: str
    raw: DescentMetrics
    n_tasks: int
    support_radius: float
    task_family: str


def _decode_quality(task: CNFTask, decode: DecodeResult) -> float:
    return float(task.satisfied_by(decode.assignment))


def off_support_rate(decodes: Iterable[DecodeResult]) -> float:
    """Return the fraction of decodes that left the decoder support region."""
    rows = tuple(decodes)
    return float(sum(not row.supported for row in rows) / len(rows))


def _metrics_from_results(
    tasks: tuple[CNFTask, ...],
    planning_results: tuple[PlanningResult, ...],
) -> DescentMetrics:
    direct_decodes = direct_decode_dataset(tasks)
    planned_decodes = tuple(decode_latent(result.z_t) for result in planning_results)
    energy_trace = mean_energy_trace(planning_results)
    accuracy_before = accuracy_for_decodes(tasks, direct_decodes)
    accuracy_after = accuracy_for_decodes(tasks, planned_decodes)
    return DescentMetrics(
        energy_monotone=energy_trace_is_monotone(energy_trace),
        accuracy_before_planning=accuracy_before,
        accuracy_after_planning=accuracy_after,
        accuracy_delta_after_planning=accuracy_after - accuracy_before,
        latent_drift_norm=mean_latent_drift_norm(planning_results),
        off_support_rate=off_support_rate(planned_decodes),
        energy_trace=energy_trace,
    )


def run_raw_descent_metrics() -> DescentMetrics:
    """Measure the unguarded Exp 1417-compatible latent descent baseline."""
    tasks = build_tiny_cnf_tasks()
    planning_results = tuple(plan_latent(encode_task(task)) for task in tasks)
    return _metrics_from_results(tasks, planning_results)


def anchored_dual_path_acceptance(
    task: CNFTask,
    current_z: Iterable[float],
    candidate_z: Iterable[float],
    anchor_weight: float = ANCHORED_ANCHOR_WEIGHT,
) -> CandidateDecision:
    """Accept a candidate only when energy drops and decoded quality does not."""
    h_x = encode_task(task)
    current_arr = np.asarray(tuple(current_z), dtype=np.float64)
    candidate_arr = np.asarray(tuple(candidate_z), dtype=np.float64)
    current_energy = planning_energy(current_arr, h_x, anchor_weight)
    candidate_energy = planning_energy(candidate_arr, h_x, anchor_weight)
    current_quality = _decode_quality(task, decode_latent(current_arr))
    candidate_quality = _decode_quality(task, decode_latent(candidate_arr))
    energy_lowered = candidate_energy <= current_energy + 1e-12
    accepted = bool(energy_lowered and candidate_quality >= current_quality)
    return CandidateDecision(
        accepted=accepted,
        energy_lowered=bool(energy_lowered),
        current_quality=current_quality,
        candidate_quality=candidate_quality,
        current_energy=float(current_energy),
        candidate_energy=float(candidate_energy),
    )


def plan_anchored_dual_path_latent(
    task: CNFTask,
    n_steps: int = DEFAULT_PLANNING_STEPS,
    lr: float = DEFAULT_PLANNING_LR,
    anchor_weight: float = ANCHORED_ANCHOR_WEIGHT,
) -> AnchoredPlanningResult:
    """Plan with a stronger anchor and reject decoded-quality regressions."""
    h_arr = encode_task(task)
    z_arr = h_arr.copy()
    trace = [planning_energy(z_arr, h_arr, anchor_weight)]
    rejected = 0
    for _ in range(n_steps):
        candidate = z_arr - (lr * planning_gradient(z_arr, h_arr, anchor_weight))
        decision = anchored_dual_path_acceptance(
            task,
            z_arr,
            candidate,
            anchor_weight=anchor_weight,
        )
        if decision.accepted:
            z_arr = candidate
        else:
            rejected += 1
        trace.append(planning_energy(z_arr, h_arr, anchor_weight))
    return AnchoredPlanningResult(
        h_x=h_arr,
        z_t=z_arr,
        energy_trace=tuple(float(value) for value in trace),
        latent_drift_norm=float(np.linalg.norm(z_arr - h_arr)),
        anchoring_applied=True,
        dual_path_decoder_stub=True,
        rejected_candidates=rejected,
    )


def run_comparison() -> RepairComparisonMetrics:
    """Compare raw descent with anchored dual-path descent on one benchmark."""
    tasks = build_tiny_cnf_tasks()
    raw = run_raw_descent_metrics()
    anchored_results = tuple(plan_anchored_dual_path_latent(task) for task in tasks)
    anchored = _metrics_from_results(tasks, anchored_results)
    viable = bool(
        anchored.accuracy_delta_after_planning >= 0.0
        and anchored.off_support_rate < raw.off_support_rate
    )
    verdict = (
        "anchored_dual_path_repair_viable"
        if viable
        else "anchored_dual_path_repair_not_viable"
    )
    return RepairComparisonMetrics(
        status="complete",
        anchoring_applied=True,
        dual_path_decoder_stub=True,
        energy_monotone=anchored.energy_monotone,
        accuracy_before_planning=anchored.accuracy_before_planning,
        accuracy_after_planning=anchored.accuracy_after_planning,
        accuracy_delta_after_planning=anchored.accuracy_delta_after_planning,
        latent_drift_norm=anchored.latent_drift_norm,
        off_support_rate=anchored.off_support_rate,
        energy_trace=anchored.energy_trace,
        anchored_repair_viable=viable,
        honest_verdict=verdict,
        raw=raw,
        n_tasks=len(tasks),
        support_radius=DEFAULT_SUPPORT_RADIUS,
        task_family="synthetic_two_variable_cnf",
    )


def _metrics_to_artifact(metrics: DescentMetrics) -> dict[str, Any]:
    return {
        "energy_monotone": metrics.energy_monotone,
        "accuracy_before_planning": metrics.accuracy_before_planning,
        "accuracy_after_planning": metrics.accuracy_after_planning,
        "accuracy_delta_after_planning": metrics.accuracy_delta_after_planning,
        "latent_drift_norm": metrics.latent_drift_norm,
        "off_support_rate": metrics.off_support_rate,
        "energy_trace": list(metrics.energy_trace),
    }


def build_artifact() -> dict[str, Any]:
    """Build the complete Exp 1436 JSON artifact from measured smoke metrics."""
    metrics = run_comparison()
    artifact: dict[str, Any] = {
        "schema": "carnot.phase3.anchored_dual_path_latent_repair.v1",
        "experiment": "1436_anchored_dual_path_latent_repair_v1",
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-KONA-034", "SCENARIO-KONA-034"],
        "status": metrics.status,
        "anchoring_applied": metrics.anchoring_applied,
        "dual_path_decoder_stub": metrics.dual_path_decoder_stub,
        "energy_monotone": metrics.energy_monotone,
        "accuracy_before_planning": metrics.accuracy_before_planning,
        "accuracy_after_planning": metrics.accuracy_after_planning,
        "accuracy_delta_after_planning": metrics.accuracy_delta_after_planning,
        "latent_drift_norm": metrics.latent_drift_norm,
        "off_support_rate": metrics.off_support_rate,
        "anchored_repair_viable": metrics.anchored_repair_viable,
        "honest_verdict": metrics.honest_verdict,
        "energy_trace": list(metrics.energy_trace),
        "raw_descent": _metrics_to_artifact(metrics.raw),
        "decoder_support_radius": metrics.support_radius,
        "task_family": metrics.task_family,
        "n_tasks": metrics.n_tasks,
    }
    return artifact


def write_experiment_artifact(path: Path | str = RESULT_PATH) -> dict[str, Any]:
    """Persist the Exp 1436 artifact after first writing the bootstrap state."""
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"status": "in_progress"}, indent=2) + "\n")
    artifact = build_artifact()
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact
