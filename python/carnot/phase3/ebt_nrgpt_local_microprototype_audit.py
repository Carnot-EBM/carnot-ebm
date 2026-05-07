"""Exp 1450 EBT/NRGPT-style local energy-convergence microprototype.

The audit asks a deliberately small question: do existing Carnot traces expose
an iterative "think until energy flattens" signal that is worth comparing with
the anchored latent repair result from Exp 1436?  This module keeps the answer
CPU-local and trace-grounded.  It reuses the FoVer trace loader and the existing
reasoning embedding path, then applies a tiny quadratic energy descent around
the local correct-trace centroid.  Lower energy is reported only as a smoke
signal, not as decoded accuracy or a Phase-3 scale claim.

Spec refs: REQ-KONA-035, SCENARIO-KONA-035.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.core.energy import AutoGradMixin
from carnot.data.fover import FoVerDataset, FoVerItem
from carnot.inference.reasoning_energy import text_to_reasoning_embedding

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260507"
RESULT_PATH = PROJECT_ROOT / "results" / "experiment_1450_ebt_nrgpt_local_microprototype_audit.json"
ANCHORED_REFERENCE_PATH = (
    PROJECT_ROOT / "results" / "experiment_1436_anchored_dual_path_latent_repair_v1.json"
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "energy_convergence_probe_complete",
    "traces_evaluated",
    "baseline_energy_delta",
    "anchored_repair_energy_delta_reference",
    "convergence_steps_median",
    "scale_recommendation",
    "commands_run",
    "honest_verdict",
)
SCALE_RECOMMENDATIONS = ("retire", "keep_smoke_only", "scale_future_milestone")


@dataclass(frozen=True)
class LocalTraceEnergy(AutoGradMixin):
    """Quadratic trace energy used only for the local convergence comparator.

    The target is the centroid of local correct FoVer trace embeddings.  The
    anchor keeps each descent path tied to its starting trace so the probe tests
    EBT/NRGPT-style iterative inference rather than unconstrained teleporting to
    a memorized centroid.
    """

    target: jax.Array
    anchor: jax.Array
    anchor_weight: float = 0.05

    @property
    def input_dim(self) -> int:
        return int(self.target.shape[0])

    def energy(self, x: jax.Array) -> jax.Array:
        target_gap = x - self.target
        anchor_gap = x - self.anchor
        return jnp.sum(target_gap * target_gap) + self.anchor_weight * jnp.sum(
            anchor_gap * anchor_gap
        )


@dataclass(frozen=True)
class TraceConvergenceResult:
    """One trace's iterative energy-minimization measurement."""

    trace_id: str
    label: int
    energy_trace: tuple[float, ...]
    convergence_step: float


@dataclass(frozen=True)
class ConvergenceSummary:
    """Aggregate smoke metrics used by the Exp 1450 artifact."""

    energy_convergence_probe_complete: bool
    traces_evaluated: int
    baseline_energy_delta: float
    convergence_steps_median: float
    results: tuple[TraceConvergenceResult, ...]


def load_smoke_traces(
    *,
    path: str | Path | None = None,
    rows: Iterable[dict[str, Any]] | None = None,
    max_traces: int = 8,
) -> tuple[FoVerItem, ...]:
    """Load the tiny local FoVer trace sample for the smoke audit."""
    dataset = FoVerDataset(path=path, rows=rows)
    return tuple(dataset)[:max_traces]


def _embedding_matrix(traces: Sequence[FoVerItem], vocab_size: int) -> np.ndarray:
    return np.stack(
        [
            np.asarray(
                text_to_reasoning_embedding(trace.text, vocab_size=vocab_size), dtype=np.float64
            )
            for trace in traces
        ],
        axis=0,
    )


def _convergence_step(trace: Sequence[float], tolerance: float) -> float:
    deltas = (abs(right - left) for left, right in zip(trace, trace[1:]))
    return float(
        next(
            (index for index, delta in enumerate(deltas, start=1) if delta <= tolerance),
            len(trace) - 1,
        )
    )


def _descend_energy(
    embedding: np.ndarray,
    target: np.ndarray,
    *,
    max_steps: int,
    step_size: float,
) -> tuple[float, ...]:
    state = jnp.asarray(embedding)
    energy_model = LocalTraceEnergy(target=jnp.asarray(target), anchor=state)
    trace = [float(energy_model.energy(state))]
    for _ in range(max_steps):
        state = state - step_size * energy_model.grad_energy(state)
        trace.append(float(energy_model.energy(state)))
    return tuple(trace)


def run_energy_convergence_probe(
    traces: Sequence[FoVerItem],
    *,
    max_steps: int = 12,
    step_size: float = 0.2,
    tolerance: float = 1e-6,
    vocab_size: int = 64,
) -> ConvergenceSummary:
    """Run the EBT/NRGPT-style local energy descent over FoVer traces."""
    embeddings = _embedding_matrix(traces, vocab_size)
    labels = np.asarray([trace.label for trace in traces], dtype=np.int64)
    target = embeddings[labels == 1].mean(axis=0)
    results: list[TraceConvergenceResult] = []
    for index, (embedding, label) in enumerate(zip(embeddings, labels, strict=True)):
        energy_trace = _descend_energy(
            embedding,
            target,
            max_steps=max_steps,
            step_size=step_size,
        )
        results.append(
            TraceConvergenceResult(
                trace_id=f"fover_trace_{index}",
                label=int(label),
                energy_trace=energy_trace,
                convergence_step=_convergence_step(energy_trace, tolerance),
            )
        )
    deltas = [row.energy_trace[-1] - row.energy_trace[0] for row in results]
    baseline_energy_delta = float(np.mean(deltas))
    complete = bool(
        results and baseline_energy_delta < 0.0 and all(delta <= 0.0 for delta in deltas)
    )
    return ConvergenceSummary(
        energy_convergence_probe_complete=complete,
        traces_evaluated=len(results),
        baseline_energy_delta=baseline_energy_delta,
        convergence_steps_median=float(np.median([row.convergence_step for row in results])),
        results=tuple(results),
    )


def anchored_reference_energy_delta(path: str | Path = ANCHORED_REFERENCE_PATH) -> float:
    """Read Exp 1436's anchored repair energy delta for the comparator field."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    trace = [float(value) for value in payload["energy_trace"]]
    return float(trace[-1] - trace[0])


def choose_scale_recommendation(
    *,
    traces_evaluated: int,
    baseline_energy_delta: float,
    anchored_repair_energy_delta_reference: float,
    convergence_steps_median: float,
    decoded_quality_evidence: bool,
) -> str:
    """Gate scale-up using smoke metrics instead of treating energy as accuracy."""
    if traces_evaluated <= 0 or baseline_energy_delta >= 0.0 or convergence_steps_median <= 0.0:
        return "retire"
    if (
        decoded_quality_evidence
        and traces_evaluated >= 30
        and baseline_energy_delta <= anchored_repair_energy_delta_reference
        and convergence_steps_median <= 8.0
    ):
        return "scale_future_milestone"
    return "keep_smoke_only"


def _honest_verdict(scale_recommendation: str) -> str:
    return {
        "retire": "energy_convergence_probe_not_viable_retire",
        "keep_smoke_only": "energy_converges_but_no_decoded_quality_claim_keep_smoke_only",
        "scale_future_milestone": "energy_converges_with_quality_gate_scale_future_milestone",
    }[scale_recommendation]


def build_artifact(
    *,
    rows: Iterable[dict[str, Any]] | None = None,
    anchored_reference_path: str | Path = ANCHORED_REFERENCE_PATH,
    commands_run: Sequence[str] = (),
    max_traces: int = 8,
) -> dict[str, Any]:
    """Build the complete Exp 1450 JSON artifact from measured smoke metrics."""
    traces = load_smoke_traces(rows=rows, max_traces=max_traces)
    summary = run_energy_convergence_probe(traces)
    anchored_delta = anchored_reference_energy_delta(anchored_reference_path)
    recommendation = choose_scale_recommendation(
        traces_evaluated=summary.traces_evaluated,
        baseline_energy_delta=summary.baseline_energy_delta,
        anchored_repair_energy_delta_reference=anchored_delta,
        convergence_steps_median=summary.convergence_steps_median,
        decoded_quality_evidence=False,
    )
    return {
        "schema": "carnot.phase3.ebt_nrgpt_local_microprototype_audit.v1",
        "experiment": "1450_ebt_nrgpt_local_microprototype_audit",
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-KONA-035", "SCENARIO-KONA-035"],
        "status": "complete",
        "energy_convergence_probe_complete": summary.energy_convergence_probe_complete,
        "traces_evaluated": summary.traces_evaluated,
        "baseline_energy_delta": summary.baseline_energy_delta,
        "anchored_repair_energy_delta_reference": anchored_delta,
        "convergence_steps_median": summary.convergence_steps_median,
        "scale_recommendation": recommendation,
        "commands_run": list(commands_run),
        "honest_verdict": _honest_verdict(recommendation),
        "decoded_quality_evidence": False,
        "source_trace_family": "fover_corpus",
        "per_trace_energy_deltas": [
            row.energy_trace[-1] - row.energy_trace[0] for row in summary.results
        ],
        "per_trace_convergence_steps": [row.convergence_step for row in summary.results],
    }


def write_experiment_artifact(
    path: str | Path = RESULT_PATH,
    *,
    rows: Iterable[dict[str, Any]] | None = None,
    anchored_reference_path: str | Path = ANCHORED_REFERENCE_PATH,
    commands_run: Sequence[str] = (),
) -> dict[str, Any]:
    """Persist Exp 1450 after first writing the required bootstrap artifact."""
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"status": "in_progress"}, indent=2) + "\n")
    artifact = build_artifact(
        rows=rows,
        anchored_reference_path=anchored_reference_path,
        commands_run=commands_run,
    )
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact
