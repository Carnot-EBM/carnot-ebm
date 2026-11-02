"""REQ-VERIFY-7997: frozen prediction with no target or optimizer interface.

Only public numeric inputs enter this package. Evaluator targets live in the
reporting layer and cannot affect coefficients or stream probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit, logit  # type: ignore[import-untyped]

from carnot.verify import sparse_energy_7996 as sparse
from carnot.verify import qwen_energy_calibration_7972 as scalar

Json = dict[str, Any]
ARMS = ("spline", "logistic", "mlp", "scalar", "raw_q")
CONFIG = dict(
    arms=list(ARMS),
    primary_seed=17,
    temperatures=[1.0, 0.5, 2.0],
    objective="calibration_primary_seed_brier",
    coefficient_refit_steps=0,
    costs=dict(accept="5*y", reject="1-y", escalate=0.25),
    draws=10000,
    minimum_groups=192,
    per_class=20,
    random_seed=69397,
    uncertainty_unit="normalized_source_group",
    primary_controls=["logistic", "mlp"],
    benefit_gates=dict(
        minimum_cost_gain=0.02,
        lower_95_bound_above=0.0,
        adjusted_p_below=0.05,
        maximum_extra_false_accepts=0,
        minimum_automation=0.50,
        maximum_brier_degradation_upper_bound=0.01,
    ),
)


def action(p: float | None) -> str:
    """Strict thresholds give escalation priority when expected costs tie."""
    return (
        "accept"
        if p is not None and p < 0.05
        else ("reject" if p is not None and p > 0.75 else "escalate")
    )


def transform(p: float, temperature: float) -> float:
    """Scale log odds without moving exact zero and one probabilities."""
    return p if p in (0.0, 1.0) else float(expit(logit(p) / temperature))


def predictions(heads: Json, sources: list[Json]) -> list[Json]:
    """Save every arm, seed and temperature before a target can be opened."""
    rows, seen = [], set()
    for i, source in enumerate(sources):
        if set(source) != {"family_id", "source_cluster_id", "q", "features", "status"}:
            raise ValueError("predictor_fields")
        if source["family_id"] in seen:
            raise ValueError("duplicate")
        seen.add(source["family_id"])
        q, features = source["q"], source["features"]
        if q is not None and (not np.isfinite(q) or not 0 <= q <= 1):
            raise ValueError("probability")
        if features is not None and (len(features) != 8 or not np.isfinite(features).all()):
            raise ValueError("features")
        usable = q is not None and features is not None and source["status"] == "completed"
        for arm in ARMS:
            arm_heads = heads[arm] if arm != "raw_q" else [dict(seed=17)]
            for index, original in enumerate(arm_heads):
                seed = original.get("seed", index)
                if usable:
                    h = dict(original, temperature=1.0)
                    x = np.array([[q, *features]], dtype=float)
                    p = (
                        float(q)
                        if arm == "raw_q"
                        else float(
                            scalar.predict(h, np.array([q]))[0]
                            if arm == "scalar"
                            else sparse.predict(h, x)[0]
                        )
                    )
                    identity = float(sparse.sigmoid_predict(h, x)[0]) if arm == "spline" else None
                else:
                    p, identity = None, None
                for t in CONFIG["temperatures"]:
                    rows.append(
                        dict(
                            source,
                            arm=arm,
                            seed=seed,
                            primary=index == 0,
                            probability=transform(p, t) if p is not None else None,
                            identity_probability=transform(identity, t)
                            if identity is not None
                            else None,
                            temperature=t,
                            eligibility=usable,
                            failure_status=not usable and source["status"] != "excluded",
                            censor_status=source["status"] == "censored",
                        )
                    )
        if i % 32 == 0:
            print(f"[exp7997] predictions_completed={i + 1}/{len(sources)}", flush=True)
    return rows


def select(rows: list[Json], policies: Json) -> list[Json]:
    """Selecting already sealed candidates never uses a stream target."""
    return [dict(r) for r in rows if r["temperature"] == policies[r["arm"]]["temperature"]]
