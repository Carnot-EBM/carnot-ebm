"""REQ-VERIFY-8085: local responses from public centers and released labels.

Gaussian energies are exactly logistic probabilities. This numerical identity
qualifies a component; supplied fixture labels cannot establish future benefit.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit
from scipy.spatial.distance import pdist

from carnot.experiment_8021_v695_typed_decision_test import action as action, loss
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import fresh_feedback_8064 as fresh

Json = dict[str, Any]
Array = NDArray[np.float64]
LIMITS: Json = dict(
    initial=16, additions=12, per_opportunity=4, opportunities=[64, 128, 192], update_rows=64
)


def matrix(value: Any) -> Array:
    """Reject invalid public geometry before any label can influence a center."""
    x = np.asarray(value, dtype=float)
    if x.ndim != 2 or x.shape[1] != 9 or not np.isfinite(x).all():
        raise ValueError("nonfinite_or_shape")
    return x


def initialize(values: Any, source_ids: list[str]) -> Json:
    """Freeze scaling, width and a label-free reserved center order from fit."""
    x = matrix(values)
    if len(x) < 28 or len(source_ids) != len(x) or len(set(source_ids)) != len(x):
        raise ValueError("fit_source_identity")
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1
    scaled = (x - mean) / std
    distances = pdist(scaled)
    positive = distances[distances > 0]
    sigma = float(np.median(positive)) if len(positive) else 1.0
    first = min(range(len(x)), key=lambda i: hashlib.sha256(source_ids[i].encode()).hexdigest())
    selected = [first]
    while len(selected) < 28:
        remaining = [i for i in range(len(x)) if i not in selected]
        selected.append(
            min(
                remaining,
                key=lambda i: (
                    -float(np.min(np.sum((scaled[selected] - scaled[i]) ** 2, axis=1))),
                    source_ids[i],
                ),
            )
        )
    records = [
        dict(
            source_id=source_ids[i],
            x=scaled[i].tolist(),
            feedback_origin="fit_public",
            identity=canonical_hash([source_ids[i], scaled[i].tolist()]),
        )
        for i in selected
    ]
    state = dict(
        geometry=dict(mean=mean.tolist(), std=std.tolist(), sigma=sigma),
        centers=records[:16],
        reserved=records[16:],
        coefficients=[0.0] * 17,
        version=0,
        opportunities=[],
        consumed=[],
    )
    state["commit_hash"] = canonical_hash(state)
    return state


def design(state: Json, values: Any) -> Array:
    """Keep fit geometry fixed while adding only public Gaussian columns."""
    raw = matrix(values)
    g = state["geometry"]
    z = (raw - np.asarray(g["mean"])) / np.asarray(g["std"])
    centers = np.asarray([r["x"] for r in state["centers"]])
    squared = np.sum((z[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return np.column_stack((np.ones(len(z)), np.exp(-squared / (2 * g["sigma"] ** 2))))


def predict(state: Json, values: Any) -> Array:
    """Normalize E0=0 and E1=-f without sampling or changing generator weights."""
    phi = design(state, values)
    theta = np.asarray(state["coefficients"])
    if theta.shape != (phi.shape[1],) or not np.isfinite(theta).all():
        raise ValueError("stale_coefficient_shape")
    logits = phi @ theta
    weights = np.exp(
        np.column_stack((np.zeros(len(logits)), logits)) - np.maximum(0, logits)[:, None]
    )
    return np.asarray(weights[:, 1] / weights.sum(1))


def objective(theta: Array, phi: Array, y: Array, ridge: float) -> tuple[float, Array]:
    """Penalize every coefficient so both learned arms optimize the same loss."""
    logits = phi @ theta
    return float(
        np.mean(np.logaddexp(0, logits) - y * logits) + ridge * (theta @ theta) / 2
    ), np.asarray(phi.T @ (expit(logits) - y) / len(y) + ridge * theta)


def solve(phi: Array, y: Array, initial: Array, ridge: float = 0.01) -> Json:
    """A bounded convex solve records real costs and rejects unfinished results."""
    if (
        not len(y)
        or phi.shape != (len(y), len(initial))
        or not np.isfinite(phi).all()
        or not np.isfinite(initial).all()
        or not np.isin(y, [0, 1]).all()
        or ridge <= 0
    ):
        raise ValueError("optimizer_operands")
    began = time.monotonic()

    def deadline(theta: Array) -> None:
        """Check elapsed time at each solver boundary instead of implying convergence."""
        if time.monotonic() - began > 600:
            raise TimeoutError("optimizer_600s")

    result = minimize(
        lambda theta: objective(theta, phi, y, ridge),
        initial,
        jac=True,
        method="L-BFGS-B",
        callback=deadline,
        options=dict(maxiter=256, gtol=1e-9, ftol=1e-15),
    )
    value, gradient = objective(result.x, phi, y, ridge)
    norm = float(np.max(np.abs(gradient)))
    if (
        not result.success
        or norm > 1e-7
        or not np.isfinite(result.x).all()
        or not np.isfinite(value)
    ):
        raise ValueError("optimizer_unfinished")
    return dict(
        coefficients=result.x.tolist(),
        objective=value,
        gradient_inf=norm,
        iterations=int(result.nit),
        duration_s=time.monotonic() - began,
        ridge=ridge,
        maxiter=256,
        deadline_s=600,
    )


def released(rows: list[Json], slot: int, role: str) -> list[Json]:
    """Labels are available only in eligible receipts at their original release."""
    if len({r["source_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate_source_ids")
    if not rows or any(
        r.get("role") != role
        or r.get("eligible") is not True
        or type(r.get("y")) is not int
        or r["y"] not in (0, 1)
        or not 0 <= r["release_slot"] <= r["observed_slot"] <= slot
        for r in rows
    ):
        raise ValueError("feedback_contract")
    matrix([r["x"] for r in rows])
    return sorted(deepcopy(rows), key=lambda r: (r["release_slot"], r["source_id"]))[-64:]


def candidate(state: Json, feedback: list[Json], slot: int, arm: str) -> Json:
    """Match capacity and optimization while varying only center provenance."""
    if arm not in {"fixed_center", "feedback_grown"}:
        raise ValueError("candidate_arm")
    if slot not in LIMITS["opportunities"] or slot in state["opportunities"]:
        raise ValueError("opportunity_used_or_invalid")
    if len(state["centers"]) > 24 or len(state["coefficients"]) != len(state["centers"]) + 1:
        raise ValueError("dictionary_overflow_or_coefficients")
    rows = released(feedback, slot, "update")
    existing = {r["source_id"] for r in state["centers"]}
    errors = sorted(
        [
            r
            for r in rows
            if r["source_id"] not in existing and (loss(r["issued_action"], r["y"]) or 0) > 0
        ],
        key=lambda r: (-float(loss(r["issued_action"], r["y"]) or 0), r["source_id"]),
    )[:4]
    if len(errors) != 4:
        raise ValueError("feedback_error_support")
    proposed = deepcopy(state)
    offset = len(state["centers"]) - 16
    additions = deepcopy(state["reserved"][offset : offset + 4])
    if arm == "feedback_grown":
        g = state["geometry"]
        additions = [
            dict(
                source_id=r["source_id"],
                x=((np.asarray(r["x"]) - g["mean"]) / g["std"]).tolist(),
                feedback_origin=deepcopy(r),
                identity=canonical_hash(
                    [r["source_id"], ((np.asarray(r["x"]) - g["mean"]) / g["std"]).tolist()]
                ),
            )
            for r in errors
        ]
    proposed["centers"] += additions
    if arm == "fixed_center":
        proposed["centers"] = deepcopy(state["centers"][:16] + state["reserved"][: offset + 4])
    theta = np.pad(np.asarray(state["coefficients"]), (0, 4))
    fit = solve(
        design(proposed, [r["x"] for r in rows]),
        np.asarray([r["y"] for r in rows], dtype=float),
        theta,
    )
    proposed.update(
        coefficients=fit["coefficients"],
        solve=fit,
        update_rows=rows,
        candidate_slot=slot,
        base_version=state["version"],
        base_hash=canonical_hash(state),
        arm=arm,
    )
    proposed["proposal_hash"] = canonical_hash(proposed)
    return proposed


def save(path: Path, state: Json) -> None:
    """Reuse the atomic writer and sync its directory before exposing new memory."""
    records = [
        dict(center=r, coefficient_version=state["version"], commit_hash=state["commit_hash"])
        for r in state["centers"]
    ]
    atomic_json(path, dict(state=state, sha256=canonical_hash(state), dictionary_records=records))
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load(path: Path) -> Json:
    """Cold reads require both dictionary identity and coefficient version binding."""
    value = json.loads(path.read_text())
    state: Json = value["state"]
    records = [
        dict(center=r, coefficient_version=state["version"], commit_hash=state["commit_hash"])
        for r in state["centers"]
    ]
    if (
        value["sha256"] != canonical_hash(state)
        or value["dictionary_records"] != records
        or len(state["coefficients"]) != len(state["centers"]) + 1
    ):
        raise ValueError("dictionary_hash_or_stale_coefficients")
    return state


def commit(path: Path, state: Json, proposed: Json, admissions: list[Json]) -> Json:
    """Consume fresh labels once and persist the dictionary with its coefficients."""
    if (
        proposed["base_version"] != state["version"]
        or proposed["base_hash"] != canonical_hash(state)
        or load(path) != state
    ):
        raise ValueError("stale_coefficients")
    unsigned = {key: value for key, value in proposed.items() if key != "proposal_hash"}
    if proposed["proposal_hash"] != canonical_hash(unsigned):
        raise ValueError("dictionary_proposal_hash")
    slot = proposed["candidate_slot"]
    if len(admissions) != 12 or any(
        r["source_id"] in state["consumed"] or r["release_slot"] <= slot for r in admissions
    ):
        raise ValueError("admission_not_fresh12")
    rows = released(admissions, max(r["observed_slot"] for r in admissions), "admission")
    phi = design(proposed, [r["x"] for r in rows])
    current = np.pad(np.asarray(state["coefficients"]), (0, 4))
    initial = np.zeros(len(current))
    checks, alpha = fresh.guard(
        dict(calibration=[0.0, 1.0]),
        current,
        np.asarray(proposed["coefficients"]),
        initial,
        phi,
        np.asarray([r["y"] for r in rows], dtype=float),
    )
    result = deepcopy(state)
    if alpha is not None:
        result.update(
            centers=deepcopy(proposed["centers"]),
            coefficients=(
                current + alpha * (np.asarray(proposed["coefficients"]) - current)
            ).tolist(),
        )
    result.update(
        version=state["version"] + 1,
        opportunities=[*state["opportunities"], slot],
        consumed=[*state["consumed"], *[r["source_id"] for r in rows]],
        last_commit=dict(
            proposal_hash=proposed["proposal_hash"], admissions=rows, checks=checks, alpha=alpha
        ),
    )
    result["commit_hash"] = canonical_hash(
        {key: value for key, value in result.items() if key != "commit_hash"}
    )
    save(path, result)
    return result
