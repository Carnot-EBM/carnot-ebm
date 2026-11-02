"""REQ-VERIFY-8183: reuse the qualified radial fit with frozen source information.

Each energy is a calibrated logistic score on a fixed Gaussian design. This
module changes its inputs and tune policy, never the original fitted controls.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import evidence_energy_8154 as base
from carnot.verify import evidence_energy_fit_8154 as reporting

Json = dict[str, Any]
Array = base.Array
CONTROL_ARMS = ["scalar_span", "linear12", "radial16"]
NEW_ARMS = ["local_evidence_radial16", "local_max", "local_feature_ablation"]
ARMS = [*NEW_ARMS, *CONTROL_ARMS, "equivalent_logistic", "always_escalate"]
CONFIG = dict(base.CONFIG, seed=70783, feature_count=16, policy_tie_tolerance=1e-12)
BASE_GEOMETRY, BASE_DESIGN = base.geometry, base.design


def validate_rows(rows: list[Json]) -> None:
    """Reject future targets and identity leaks before fit geometry sees any row.

    The explicit row schema excludes gold-derived features. Targets remain a
    separate binary operand, while unavailable feature vectors stay unavailable.
    """
    allowed = {
        "unit_id",
        "source_cluster_id",
        "role",
        "x",
        "y",
        "status",
        "exclusion_reason",
        "historical_paired_control",
        "slot",
    }
    if any(r["role"] not in ("fit", "tune") for r in rows):
        raise ValueError("future_role")
    if len({r["source_cluster_id"] for r in rows}) != len(rows) or len(
        {r["unit_id"] for r in rows}
    ) != len(rows):
        raise ValueError("source_leak")
    if any(
        set(r) - allowed
        or (r["x"] is not None and (len(r["x"]) != 16 or not np.isfinite(r["x"]).all()))
        for r in rows
    ):
        raise ValueError("feature_schema")
    fit = np.asarray([r["x"] for r in rows if r["role"] == "fit" and r["status"] == "completed"])
    if not len(fit) or np.max(np.std(fit[:, 12:], axis=0)) == 0:
        raise ValueError("degenerate_features")


def geometry(x: Array, ids: list[str]) -> Json:
    """Use the same label-blind centers and width for sixteen or twelve inputs.

    The ablation gets its own fit-only twelve-dimensional geometry so removing
    local signals also removes their influence on center selection and width.
    """
    return dict(BASE_GEOMETRY(x, ids), ablation_geometry=BASE_GEOMETRY(x[:, :12], ids))


def design(arm: str, x: Array, g: Json) -> Array:
    """Keep each arm's information fixed when constructing its prediction basis."""
    if arm == "local_evidence_radial16":
        return BASE_DESIGN("radial16", x, g)
    if arm == "local_feature_ablation":
        return BASE_DESIGN("radial16", x[:, :12], g["ablation_geometry"])
    if arm == "local_max":
        p = np.clip(x[:, 13], 1e-6, 1 - 1e-6)
        return np.column_stack((np.ones(len(x)), np.log(p / (1 - p))))
    return BASE_DESIGN(arm, x[:, :12], g)


def decision(p: float | None, thresholds: list[float]) -> str:
    """Threshold ties and missing signals escalate under the sealed typed policy."""
    return (
        "escalate"
        if p is None or thresholds[0] - 1e-12 <= p <= thresholds[1] + 1e-12
        else "accept"
        if p < thresholds[0]
        else "reject"
    )


def select_comparator(costs: Json) -> str:
    """Listed order resolves an exact tie in original V705 tune cost."""
    return min(CONTROL_ARMS, key=lambda a: costs[a])


def tune_policy(rows: list[Json], h: Json) -> Json:
    """Minimize observed tune cost with missing slots charged as escalations.

    Candidate boundaries come only from tune probabilities. Equal-cost policies
    prefer the widest escalation interval, then the lower acceptance boundary.
    """
    tune = [r for r in rows if r["role"] == "tune"]
    available = [r for r in tune if r["status"] == "completed" and r["y"] in (0, 1)]
    phi = design(h["arm"], np.asarray([r["x"] for r in available]), h["geometry"])
    b, a = h["calibration"]
    ps = base.expit(b + a * (phi @ np.asarray(h["weights"])))
    candidates = sorted({0.0, 1.0, *map(float, ps)})
    options = []
    for low in candidates:
        for high in candidates:
            if low <= high:
                total = sum(
                    base.loss(decision(float(p), [low, high]), r["y"])
                    for p, r in zip(ps, available, strict=True)
                ) + 0.5 * (len(tune) - len(available))
                options.append((total, -(high - low), low, high))
    cost, _, low, high = min(options)
    return dict(
        thresholds=[low, high],
        typed_cost=cost / len(tune),
        role="tune",
        intended_count=len(tune),
        missing_action="escalate",
        tie_rule="escalate",
    )


def train(rows: list[Json], controls: list[Json], protocol: Json, raw: Path) -> Json:
    """Fit new heads, preserve original controls and seal before reserved capture."""
    validate_rows(rows)
    if [h["arm"] for h in controls] != CONTROL_ARMS:
        raise ValueError("missing_arm")

    def fit_progress(phase: str, completed: int = 0, pending: int = 0) -> None:
        """Correct inherited fold counts for the three heads trained in this run."""
        if phase.startswith("fold_completed_"):
            pending = len(NEW_ARMS) * len(base.RIDGES) * CONFIG["folds"] - completed
        print(f"[exp8183] phase={phase} completed={completed} pending={pending}", flush=True)

    with (
        patch.object(base, "FITTED_ARMS", NEW_ARMS),
        patch.object(base, "geometry", geometry),
        patch.object(base, "design", design),
        patch.object(base, "progress", fit_progress),
    ):
        fitted = base.train(rows, raw)
    for h in fitted["heads"]:
        if "calibration" in h:
            fit_progress("before_benchmark_policy_" + h["arm"])
            h["policy"] = tune_policy(rows, h)
            b, a = h["calibration"]
            theta = np.asarray(h["weights"]) * a
            theta[0] += b
            h["equivalent_logistic_weights"] = theta.tolist()
            fit_progress("after_benchmark_policy_" + h["arm"], 1, 0)
    copied = deepcopy(controls)
    for h in copied:
        b, a = h["calibration"]
        theta = np.asarray(h["weights"]) * a
        theta[0] += b
        h["equivalent_logistic_weights"] = theta.tolist()
        h["imported_control"] = True
    fitted["heads"] += copied
    fitted.update(
        comparator_id=select_comparator(protocol["H1"]["original_tune_costs"]),
        cost_matrix=protocol["decision_costs"],
        feature_schema=protocol["features"],
        prediction_code="sentence_energy_8183.design/evaluate",
        reserved_outcomes_opened=False,
    )
    atomic_json(raw / "frozen_heads.json", fitted)
    return fitted


def evaluate(rows: list[Json], heads: list[Json]) -> Json:
    """Reduce treatment and original control predictions with their own masks.

    Missing local evidence causes escalation without erasing an available V705
    prediction. All source slots remain visible, and repeated arms add no units.
    """
    historical = [r["historical_paired_control"] for r in rows]
    originals = [h for h in heads if h["arm"] in CONTROL_ARMS]
    trained = [h for h in heads if h["arm"] in NEW_ARMS]
    with patch.object(base, "ARMS", CONTROL_ARMS):
        old = base.evaluate(historical, originals)
    with (
        patch.object(base, "ARMS", [*NEW_ARMS, "always_escalate"]),
        patch.object(base, "design", design),
    ):
        new = base.evaluate(rows, trained)
    lookup = {h["arm"]: h for h in trained}
    parity = []
    for r in new["rows"]:
        if r["arm"] in lookup:
            r["action"] = decision(r["p"], lookup[r["arm"]]["policy"]["thresholds"])
        if r["y"] in (0, 1):
            r.update(
                numerator=base.loss(r["action"], r["y"]),
                denominator=1,
                coverage=int(r["action"] != "escalate"),
                false_accept=int(r["action"] == "accept" and r["y"] == 1),
            )
    for p in new["energy_logistic_parity_rows"]:
        h = lookup[p["arm"]]
        row = next(r for r in rows if r["unit_id"] == p["unit_id"])
        phi = design(h["arm"], np.asarray([row["x"]]), h["geometry"])
        lp = float(base.expit((phi @ np.asarray(h["equivalent_logistic_weights"]))[0]))
        ed, ld = (
            decision(prob, h["policy"]["thresholds"]) for prob in (p["energy_probability"], lp)
        )
        parity.append(
            dict(
                p,
                logistic_probability=lp,
                energy_action=ed,
                logistic_action=ld,
                passed=abs(p["energy_probability"] - lp) <= 1e-10 and ed == ld,
            )
        )
    equivalent = [
        dict(r, arm="equivalent_logistic") for r in new["rows"] if r["arm"] == NEW_ARMS[0]
    ]
    for r in old["rows"]:
        if not r["denominator"] and r["y"] in (0, 1):
            r.update(numerator=0.5, denominator=1)
    for p in old["energy_logistic_parity_rows"]:
        h = next(h for h in originals if h["arm"] == p["arm"])
        row = next(r for r in historical if r["unit_id"] == p["unit_id"])
        phi = BASE_DESIGN(h["arm"], np.asarray([row["x"]]), h["geometry"])
        b, a = h["calibration"]
        theta = np.asarray(h["weights"]) * a
        theta[0] += b
        lp = float(base.expit((phi @ theta)[0]))
        p.update(
            logistic_probability=lp,
            logistic_action=base.action(lp),
            passed=abs(p["energy_probability"] - lp) <= 1e-10
            and p["energy_action"] == base.action(lp),
        )
    records = [*new["rows"], *old["rows"], *equivalent]
    with patch.object(reporting, "energy", __import__(__name__, fromlist=["ARMS"])):
        summaries = reporting.independent_reduce(records)
    return dict(
        rows=records,
        tuning_rows=[r for r in records if r["role"] == "tune"],
        energy_logistic_parity_rows=[*parity, *old["energy_logistic_parity_rows"]],
        development_metrics=summaries,
    )
