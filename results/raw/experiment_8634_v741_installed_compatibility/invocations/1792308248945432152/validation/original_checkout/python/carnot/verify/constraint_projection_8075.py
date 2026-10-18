"""REQ-REPORT-8075: bounded CPU correction of supplied linear constraints.

Known labels define these half-spaces. Feasibility says nothing about new labels.
The extra coefficient stays one so the frozen calibration intercept is included.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
Array = NDArray[np.float64]
TOL = 1e-8
CAPACITY = 64


def calibrated(basis: Array, coefficients: Any, calibration: Any) -> tuple[Array, Array]:
    """Use the stored intercept/slope order to preserve actual decision logits."""
    intercept, slope = calibration
    return np.column_stack((slope * basis, np.full(len(basis), intercept))), np.append(
        np.asarray(coefficients, dtype=float), 1.0
    )


def constraint(phi: Any, initial: Any, y: int, source_id: str) -> Json:
    """The minimum threshold keeps the original head feasible by construction."""
    x, w0 = np.asarray(phi, dtype=float), np.asarray(initial, dtype=float)
    if x.ndim != 1 or x.shape != w0.shape:
        raise ValueError("constraint_shape")
    if not np.isfinite(x).all() or not np.isfinite(w0).all():
        raise ValueError("nonfinite_constraint")
    if y not in (0, 1):
        raise ValueError("constraint_label")
    sign = 2 * y - 1
    return dict(
        source_id=source_id,
        normal=(sign * x).tolist(),
        rhs=min(float(sign * (x @ w0)), 0.0 if y else float(np.log(9))),
    )


class Memory:
    """Atomically persist released labels so restarts cannot apply them twice.

    Only 64 active constraints are kept; historical receipts remain in the audit
    journal. A late arrival uses its original slot rather than its arrival time.
    """

    def __init__(self, path: Path, initial: Any):
        self.path = path
        self.state: Json = dict(initial=list(initial), events=[], rows=[])
        if path.exists():
            value = json.loads(path.read_text())
            if canonical_hash(value["state"]) != value["sha256"]:
                raise ValueError("memory_hash")
            if value["state"]["initial"] != list(initial):
                raise ValueError("memory_identity")
            for event in value["state"]["events"]:
                self._apply(self.state, event["receipt"])
            if self.state != value["state"]:
                raise ValueError("memory_replay")

    @property
    def rows(self) -> list[Json]:
        """Return a copy so callers cannot alter the durable memory silently."""
        return deepcopy(self.state["rows"])

    @staticmethod
    def _apply(state: Json, receipt: Json) -> str:
        """Replay exact releases and record both insertion and capacity eviction."""
        if (
            receipt["role"] != "update"
            or not receipt["eligible"]
            or receipt["release_slot"] < 0
            or receipt["observed_slot"] < receipt["release_slot"]
        ):
            raise ValueError("release_contract")
        for event in state["events"]:
            if event["receipt"]["source_id"] == receipt["source_id"]:
                if event["receipt"] != receipt:
                    raise ValueError("duplicate_conflict")
                return "duplicate"
        row = dict(
            constraint(receipt["phi"], state["initial"], receipt["y"], receipt["source_id"]),
            release_slot=receipt["release_slot"],
            release_receipt=deepcopy(receipt),
        )
        ordered = sorted([*state["rows"], row], key=lambda r: (r["release_slot"], r["source_id"]))
        evicted = ordered[:-CAPACITY]
        state["rows"] = ordered[-CAPACITY:]
        state["events"].append(
            dict(
                receipt=deepcopy(receipt), addition=row, evictions=[r["source_id"] for r in evicted]
            )
        )
        return "added"

    def add(self, receipt: Json) -> str:
        """Write the complete next state before changing the in-memory state."""
        staged = deepcopy(self.state)
        status = self._apply(staged, receipt)
        if status != "duplicate":
            atomic_json(self.path, dict(state=staged, sha256=canonical_hash(staged)))
            self.state = staged
        return status


def residuals(point: Array, matrix: Array, rhs: Array, low: Array, high: Array) -> Array:
    """Check every row and box face, including after a fallback or final step."""
    return np.concatenate(
        (
            np.maximum(rhs - matrix @ point, 0),
            np.maximum(low - point, 0),
            np.maximum(point - high, 0),
        )
    )


def project(
    proposal: Any,
    rows: list[Json],
    initial: Any,
    incumbent: Any,
    *,
    seed: int,
    budget: int = 256,
    frozen_last: bool = False,
) -> Json:
    """Sample eight rows, correct the worst, then validate the complete set.

    Each selected row consumes one bounded attempt, including zero corrections.
    A fallback is separately validated; it never turns an unfinished proposal
    into a claimed feasible correction.
    """
    began = time.perf_counter_ns()
    w0, raw, previous = [np.asarray(v, dtype=float) for v in (initial, proposal, incumbent)]
    if not 0 <= budget <= 256 or len(rows) > CAPACITY:
        raise ValueError("projection_contract")
    if w0.ndim != 1 or not w0.size or raw.shape != w0.shape or previous.shape != w0.shape:
        raise ValueError("projection_shape")
    if not np.isfinite(w0).all():
        raise ValueError("nonfinite_initial")
    matrix = np.asarray([r["normal"] for r in rows], dtype=float).reshape(len(rows), len(w0))
    rhs = np.asarray([r["rhs"] for r in rows], dtype=float)
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ValueError("nonfinite_constraint")
    low, high = w0 - 0.5, w0 + 0.5
    if frozen_last:
        low[-1] = high[-1] = 1.0
    norms = np.sum(matrix * matrix, axis=1)
    full_checks = 0

    def full(value: Array) -> Array:
        """Count full dot products where they execute, including fallback checks."""
        nonlocal full_checks
        full_checks += 1
        return residuals(value, matrix, rhs, low, high)

    if np.any((norms == 0) & (rhs > TOL)):
        raise ValueError("zero_norm_violated")
    if np.max(full(w0)) > TOL:
        raise ValueError("initial_infeasible")
    point = np.clip(raw, low, high)
    steps: list[Json] = []
    rng = np.random.default_rng(seed)
    termination = "nonfinite_proposal"
    if np.isfinite(raw).all():
        termination = "budget_exhausted"
        for _ in range(budget):
            if np.max(full(point)) <= TOL:
                break
            sample = rng.choice(len(rows), min(8, len(rows)), replace=False).tolist()
            chosen = min(
                sample, key=lambda i: (-float(rhs[i] - matrix[i] @ point), rows[i]["source_id"])
            )
            violation = max(0.0, float(rhs[chosen] - matrix[chosen] @ point))
            before = point.copy()
            if violation > 0:
                point = np.clip(point + violation * matrix[chosen] / norms[chosen], low, high)
            steps.append(
                dict(
                    sample=sample,
                    chosen=chosen,
                    violation=violation,
                    before=before.tolist(),
                    after=point.tolist(),
                )
            )
        if np.max(full(point)) <= TOL:
            termination = "residual"
    candidate_residuals = full(point)
    candidate_feasible = bool(np.isfinite(point).all() and np.max(candidate_residuals) <= TOL)
    fallback = "none"
    fallback_began = time.perf_counter_ns()
    if not candidate_feasible:
        if np.isfinite(previous).all() and np.max(full(previous)) <= TOL:
            point, fallback = previous.copy(), "incumbent"
        else:
            point, fallback = w0.copy(), "initial"
    fallback_ns = time.perf_counter_ns() - fallback_began
    final = full(point)
    return dict(
        point=point.tolist(),
        proposal=raw.tolist(),
        candidate_feasible=candidate_feasible,
        feasible=bool(np.isfinite(point).all() and np.max(final) <= TOL),
        termination=termination,
        fallback=fallback,
        projection_steps=len(steps),
        projection_rows=steps,
        candidate_residuals=candidate_residuals.tolist(),
        residuals=final.tolist(),
        max_residual=float(np.max(final)),
        distance=float(np.linalg.norm(point - raw)),
        cost=dict(
            total_ns=time.perf_counter_ns() - began,
            fallback_ns=fallback_ns,
            full_residual_checks=full_checks,
            row_dot_products=full_checks * len(rows) + sum(len(r["sample"]) + 1 for r in steps),
            row_norm_evaluations=len(rows),
            coefficient_writes=(
                1 + sum(r["violation"] > 0 for r in steps) + int(fallback != "none")
            )
            * len(w0),
            cpu_calls=1,
            rust_calls=0,
            gpu_calls=0,
        ),
    )
