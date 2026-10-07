"""REQ-VERIFY-8237: retain actual small-head fits without touching a generator.

The frozen V712 core owns objectives, folds and budgets. These receipts expose
its arithmetic and source membership so a later audit can reproduce selection.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import decision_margin_8234 as frozen
from carnot.verify import evidence_energy_8154 as base
from carnot.verify import restricted_action_rule_8207 as rule

Json = dict[str, Any]
SOLVE = base.solve


def numerical_checks() -> Json:
    """A derivative check detects a wrong weighting rule before any real fitting."""
    phi = np.array([[1.0, -2.0], [1.0, 0.3], [1.0, 1.2]])
    theta, y, w = np.array([0.2, -0.3]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 3.0, 5.0])
    gradient = frozen.objective(theta, phi, y, 0.01, w)[1]
    finite = [
        (
            frozen.objective(theta + np.eye(2)[i] * 1e-6, phi, y, 0.01, w)[0]
            - frozen.objective(theta - np.eye(2)[i] * 1e-6, phi, y, 0.01, w)[0]
        )
        / 2e-6
        for i in range(2)
    ]
    a, b = (
        frozen.objective(theta, phi, y, 0.01, np.ones(3)),
        frozen.BASE_OBJECTIVE(theta, phi, y, 0.01),
    )
    error = max(abs(a[0] - b[0]), float(np.max(np.abs(a[1] - b[1]))))
    return dict(
        analytic_gradient=gradient.tolist(),
        finite_difference_gradient=finite,
        gradient_error=float(np.max(np.abs(gradient - finite))),
        uniform_error=error,
        passed=error <= 1e-10 and np.allclose(gradient, finite, atol=1e-8, rtol=0),
    )


def signature(value: Any) -> Any:
    """Replay compares learned arithmetic while each invocation keeps its own clocks."""
    if isinstance(value, dict):
        return {
            k: signature(v)
            for k, v in value.items()
            if k
            not in {
                "duration_s",
                "started_monotonic_ns",
                "ended_monotonic_ns",
                "started_wall_ns",
                "checkpoint_path",
            }
        }
    if isinstance(value, list):
        return [signature(v) for v in value]
    return value


def fit(rows: list[Json], roles: Json, checkpoints: Path) -> Json:
    """Save each solve before continuing so an interrupted batch can reuse its work."""
    if any(r["role"] not in ["head_fit", "temperature_fit", "calibration"] for r in rows):
        raise ValueError("future_target")
    checkpoints.mkdir(parents=True, exist_ok=True)
    bound = canonical_hash([rows, roles])
    receipts: list[Json] = []

    def observed(phi: base.Array, y: base.Array, ridge: float, deadline: float) -> Json:
        """Observe the active weighted objective without changing the optimizer options."""
        path = checkpoints / f"{len(receipts):03d}.json"
        key = canonical_hash([bound, phi.tolist(), y.tolist(), ridge])
        start, wall = time.monotonic_ns(), time.time_ns()
        if path.exists():
            saved = json.loads(path.read_bytes())
            if saved["input_sha256"] != key:
                raise ValueError("checkpoint_input")
            result = saved["receipt"]
        else:
            consumed = sum(r["duration_s"] for r in receipts)
            result = SOLVE(phi, y, ridge, min(deadline, time.monotonic() + 600 - consumed))
            result.update(
                initial_gradient=base.objective(np.zeros(phi.shape[1]), phi, y, ridge)[1].tolist(),
                final_gradient=base.objective(np.asarray(result["weights"]), phi, y, ridge)[
                    1
                ].tolist(),
                parameter_hashes=dict(
                    before=canonical_hash([0.0] * phi.shape[1]),
                    after=canonical_hash(result["weights"]),
                ),
                started_monotonic_ns=start,
                ended_monotonic_ns=time.monotonic_ns(),
                started_wall_ns=wall,
            )
            atomic_json(path, dict(input_sha256=key, receipt=result))
        receipts.append(deepcopy(result))
        return dict(result)

    with patch.object(base, "solve", observed):
        fitted = frozen.train(rows, roles)
    training = [r for r in rows if r["unit_id"] in fitted["heads"][0]["fit_ids"]]
    for record in fitted["fit_fold_rows"]:
        ids = record["geometry"]["fit_source_ids"]
        record.update(
            fit_source_ids=ids,
            held_source_ids=[
                r["source_cluster_id"] for r in training if r["source_cluster_id"] not in ids
            ],
            source_weight_rows=[
                dict(
                    unit_id=r["unit_id"],
                    source_cluster_id=r["source_cluster_id"],
                    held=r["source_cluster_id"] not in ids,
                    weight=1.0
                    if record["arm"].endswith("_uniform")
                    else frozen.margin_weight(r["p0"], r["baseline_action"]),
                )
                for r in training
            ],
        )
    for head in fitted["heads"]:
        head["parameter_hashes"] = head["solve_receipt"]["parameter_hashes"]
        temp = [r for r in rows if r["unit_id"] in head["temperature_ids"]]
        z = (
            rule.design(head["basis"], np.asarray([r["x"] for r in temp]), head["geometry"])
            @ np.asarray(head["weights"])
            / head["temperature"]
        )
        labels = np.asarray([r["y"] for r in temp])
        head["temperature_receipt"] = dict(
            success=True,
            objective="unweighted_log_loss",
            value=float(np.mean(np.logaddexp(0, z) - labels * z)),
            source_ids=[r["source_cluster_id"] for r in temp],
            temperature=head["temperature"],
        )
    return dict(fitted, optimizer_receipts=receipts, input_sha256=bound)


def score(head: Json, query: Json) -> Json:
    """Use only public features and the original permission to return a typed action."""
    if set(query) - {"x", "p0", "baseline_action"}:
        raise ValueError("evaluator_label")
    if (
        query["x"] is not None and (len(query["x"]) != 16 or not np.isfinite(query["x"]).all())
    ) or (
        query["p0"] is not None and (not math.isfinite(query["p0"]) or not 0 <= query["p0"] <= 1)
    ):
        raise ValueError("public_input")
    p = None
    if query["x"] is not None and query["p0"] is not None:
        phi = rule.design(head["basis"], np.asarray([query["x"]]), head["geometry"])[0]
        z = (
            math.fsum(float(a) * float(b) for a, b in zip(phi, head["weights"], strict=True))
            / head["temperature"]
        )
        p = min(1 - 1e-6, max(1e-6, float(expit(z))))
    good, bad = (-math.log1p(-p), -math.log(p)) if p is not None else (None, None)
    error = abs(rule.probability(good, bad, 1) - p) if p is not None else 0.0
    return dict(
        p_bad=p,
        p=p,
        action=rule.action(p, query["baseline_action"]),
        energy_good=good,
        energy_bad=bad,
        energy_probability_error=error,
    )
