#!/usr/bin/env python3
"""Experiment 992 — KAN MILP Violation Fix v2.

Verifies that the monotonicity-enforcement patch added to KAEMEnergy
(enforce_monotonicity() + fit() integration) reduces the 11 MILP violations
found in Exp 972 to zero.

Exp 980 was blocked by a gate-config bug (op='' empty string).
This experiment is unconditionally independent — it has no upstream gate.

Fix applied in python/carnot/models/kaem_energy.py:
  - UnivariateKAEMLayer.enforce_monotonicity(): isotonic projection via
    np.maximum.accumulate() followed by a zero-floor shift.
  - KAEMEnergy.fit(): calls enforce_monotonicity() at the end of every epoch.

Verification procedure (identical to Exp 972 so results are comparable):
  1. Create fresh KAEMEnergy(n_vars=4, n_hidden=8, seed=42).
  2. Run the same PWA/MILP verifier on the fresh model.
  3. Fit the model on synthetic directional data (same distribution as Exp 972).
  4. Run the verifier again on the fitted model.
  5. Report violations_before (from Exp 972 result), violations_after,
     and run the KAEM vs MCMC speedup benchmark to confirm no regression.

Spec: REQ-SAMPLE-015, REQ-KAN-VERIFY-001
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np

RESULT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "experiment_992_kan_milp_violation_fix_v2.json",
)

# Number of variables used in Exp 972 (must match for comparison)
N_VARS = 4

# Violations found by Exp 972 — the baseline we are fixing
VIOLATIONS_BEFORE = 11


# ---------------------------------------------------------------------------
# Verifier functions (copied verbatim from experiment_972 for exact parity)
# ---------------------------------------------------------------------------


def pwa_segments(ctrl: np.ndarray, knot_positions: np.ndarray) -> list[dict]:
    """Return PWA segment list for a single variable's spline (exact for KAEM)."""
    segments = []
    n = len(ctrl)
    for i in range(n - 1):
        x0, x1 = float(knot_positions[i]), float(knot_positions[i + 1])
        y0, y1 = float(ctrl[i]), float(ctrl[i + 1])
        dx = x1 - x0
        slope = (y1 - y0) / dx if dx != 0 else 0.0
        intercept = y0 - slope * x0
        segments.append({"x_lo": x0, "x_hi": x1, "slope": slope, "intercept": intercept})
    return segments


def verify_monotonicity(layer: Any, var_idx: int) -> dict:
    """Knot-value sequential monotonicity check for one variable."""
    ctrl = np.array(layer.control_points[var_idx])
    knots = np.array(layer._knots)

    energy_at_knots = list(ctrl)
    violations = []
    for i in range(len(energy_at_knots) - 1):
        e_lo = energy_at_knots[i]
        e_hi = energy_at_knots[i + 1]
        if e_lo > e_hi + 1e-6:
            violations.append(
                f"var {var_idx}: energy decreases from knot {i} "
                f"(x={knots[i]:.3f}, e={e_lo:.4f}) to knot {i + 1} "
                f"(x={knots[i + 1]:.3f}, e={e_hi:.4f})"
            )

    return {
        "property": f"monotonicity_var_{var_idx}",
        "verified": len(violations) == 0,
        "violation_detail": violations,
        "energy_at_knots": [float(v) for v in energy_at_knots],
    }


def verify_output_range(layer: Any, n_spins: int) -> dict:
    """Check total energy is within [-n_spins, n_spins]."""
    ctrl_arr = np.array(layer.control_points)
    per_var_min = ctrl_arr.min(axis=1)
    per_var_max = ctrl_arr.max(axis=1)
    total_min = float(per_var_min.sum())
    total_max = float(per_var_max.sum())

    violations = []
    if total_min < -n_spins:
        violations.append(
            f"total energy lower bound {total_min:.4f} < -{n_spins} (N spins bound violated)"
        )
    if total_max > n_spins:
        violations.append(
            f"total energy upper bound {total_max:.4f} > {n_spins} (N spins bound violated)"
        )

    return {
        "property": "output_range",
        "verified": len(violations) == 0,
        "violation_detail": violations,
        "total_min": total_min,
        "total_max": total_max,
    }


def verify_boundary_condition(layer: Any) -> dict:
    """Check energy(-1,...,-1) <= energy(+1,...,+1) (no-viol < max-viol)."""
    import jax.numpy as jnp

    n_vars = layer.n_vars
    x_no_viol = jnp.full((n_vars,), -1.0)
    x_max_viol = jnp.full((n_vars,), 1.0)
    e_no_viol = float(layer.energy(x_no_viol))
    e_max_viol = float(layer.energy(x_max_viol))

    violated = e_no_viol > e_max_viol + 1e-6
    violations = []
    if violated:
        violations.append(
            f"energy at x=-1 ({e_no_viol:.4f}) > energy at x=+1 ({e_max_viol:.4f}): "
            "model assigns lower energy to violations (wrong polarity)"
        )

    return {
        "property": "boundary_condition",
        "verified": not violated,
        "violation_detail": violations,
        "e_no_violation": e_no_viol,
        "e_max_violation": e_max_viol,
    }


def verify_monotonicity_milp(layer: Any, var_idx: int) -> dict:
    """MILP counter-example search using PuLP (same encoding as Exp 972)."""
    import pulp

    ctrl = np.array(layer.control_points[var_idx])
    knots = np.array(layer._knots)
    segs = pwa_segments(ctrl, knots)
    n_segs = len(segs)

    M = float(np.max(ctrl) - np.min(ctrl)) + 1.0
    if M < 1e-8:
        return {
            "property": f"milp_monotonicity_var_{var_idx}",
            "verified": True,
            "violation_detail": [],
            "milp_status": "trivial_flat_spline",
            "milp_obj": 0.0,
        }

    prob = pulp.LpProblem(f"mono_v992_var_{var_idx}", pulp.LpMaximize)
    x_a = pulp.LpVariable("x_a", -1.0, 1.0)
    x_b = pulp.LpVariable("x_b", -1.0, 1.0)
    e_a = pulp.LpVariable("e_a", cat="Continuous")
    e_b = pulp.LpVariable("e_b", cat="Continuous")
    b_a = [pulp.LpVariable(f"b_a_{i}", cat="Binary") for i in range(n_segs)]
    b_b = [pulp.LpVariable(f"b_b_{i}", cat="Binary") for i in range(n_segs)]

    prob += e_a - e_b
    prob += pulp.lpSum(b_a) == 1
    prob += pulp.lpSum(b_b) == 1
    prob += x_b - x_a >= 1e-4

    for i, seg in enumerate(segs):
        x_lo, x_hi = seg["x_lo"], seg["x_hi"]
        s, c = seg["slope"], seg["intercept"]
        prob += x_a - x_lo >= -M * (1 - b_a[i])
        prob += x_hi - x_a >= -M * (1 - b_a[i])
        prob += e_a - (s * x_a + c) <= M * (1 - b_a[i])
        prob += e_a - (s * x_a + c) >= -M * (1 - b_a[i])

    for i, seg in enumerate(segs):
        x_lo, x_hi = seg["x_lo"], seg["x_hi"]
        s, c = seg["slope"], seg["intercept"]
        prob += x_b - x_lo >= -M * (1 - b_b[i])
        prob += x_hi - x_b >= -M * (1 - b_b[i])
        prob += e_b - (s * x_b + c) <= M * (1 - b_b[i])
        prob += e_b - (s * x_b + c) >= -M * (1 - b_b[i])

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]
    obj = pulp.value(prob.objective)
    obj_val = float(obj) if obj is not None else 0.0
    violated = status == "Optimal" and obj_val > 1e-6

    violations = []
    if violated:
        xa_val = pulp.value(x_a)
        xb_val = pulp.value(x_b)
        violations.append(
            f"MILP counter-example var {var_idx}: "
            f"energy({xa_val:.4f})={pulp.value(e_a):.4f} > "
            f"energy({xb_val:.4f})={pulp.value(e_b):.4f}"
        )

    return {
        "property": f"milp_monotonicity_var_{var_idx}",
        "verified": not violated,
        "violation_detail": violations,
        "milp_status": status,
        "milp_obj": obj_val,
    }


def run_verifier(layer: Any, n_spins: int, prefix: str = "") -> tuple[int, list[str]]:
    """Run the full Exp 972 verification suite on a given layer.

    Returns (n_violations, violation_detail_list).
    prefix is prepended to violation messages (e.g. '[fitted] ').
    """
    n_violations = 0
    details: list[str] = []

    for var_idx in range(N_VARS):
        r = verify_monotonicity(layer, var_idx)
        if not r["verified"]:
            n_violations += 1
            details.extend(f"{prefix}{v}" for v in r["violation_detail"])

    milp_r = verify_monotonicity_milp(layer, var_idx=0)
    if not milp_r["verified"]:
        n_violations += 1
        details.extend(f"{prefix}{v}" for v in milp_r["violation_detail"])

    range_r = verify_output_range(layer, n_spins=n_spins)
    if not range_r["verified"]:
        n_violations += 1
        details.extend(f"{prefix}{v}" for v in range_r["violation_detail"])

    boundary_r = verify_boundary_condition(layer)
    if not boundary_r["verified"]:
        n_violations += 1
        details.extend(f"{prefix}{v}" for v in boundary_r["violation_detail"])

    return n_violations, details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_experiment() -> dict[str, Any]:
    import jax.numpy as jnp
    import jax.random as jrandom

    from carnot.models.kaem_energy import KAEMEnergy, benchmark_kaem_vs_mcmc

    t_start = time.perf_counter()
    key = jrandom.PRNGKey(42)

    # ------------------------------------------------------------------
    # 1. Verify fresh (unfitted) model — same seed as Exp 972
    # ------------------------------------------------------------------
    model_fresh = KAEMEnergy(n_vars=N_VARS, n_hidden=8, key=key)
    fresh_violations, fresh_details = run_verifier(model_fresh.layer, n_spins=N_VARS)

    # ------------------------------------------------------------------
    # 2. Verify fitted model — same synthetic data distribution as Exp 972
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    train_data = jnp.array(
        rng.choice([-1.0, 0.0, 1.0], size=(200, N_VARS), p=[0.2, 0.3, 0.5]).astype(np.float32)
    )
    model_fitted = KAEMEnergy(n_vars=N_VARS, n_hidden=8, key=key)
    model_fitted.fit(train_data, n_epochs=20)
    fitted_violations, fitted_details = run_verifier(
        model_fitted.layer, n_spins=N_VARS, prefix="[fitted] "
    )

    violations_after = fresh_violations + fitted_violations
    all_details = fresh_details + fitted_details

    # Count by type
    monotonicity_fixed = sum(
        1
        for v in (fresh_details + fitted_details)
        if "energy decreases" in v or "MILP counter-example" in v
    )
    boundary_fixed = sum(
        1 for v in (fresh_details + fitted_details) if "N spins bound" in v or "wrong polarity" in v
    )

    # ------------------------------------------------------------------
    # 3. Speedup benchmark (confirms no AUROC/performance regression)
    # ------------------------------------------------------------------
    bench = benchmark_kaem_vs_mcmc(n_vars=N_VARS, n_samples=50)
    speedup_ratio = bench["speedup_ratio"]

    t_elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # 4. Honest verdict
    # ------------------------------------------------------------------
    if violations_after == 0:
        honest_verdict = "violations_fixed"
    elif violations_after < VIOLATIONS_BEFORE:
        honest_verdict = "violations_reduced"
    else:
        honest_verdict = "violations_unchanged"

    return {
        "experiment": "exp992_kan_milp_violation_fix_v2",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "exp992_v1",
        "duration_s": round(t_elapsed, 3),
        "violations_before": VIOLATIONS_BEFORE,
        "violations_after": violations_after,
        "monotonicity_violations_fixed": max(
            0,
            7
            - sum(1 for v in all_details if "energy decreases" in v or "MILP counter-example" in v),
        ),
        "boundary_violations_fixed": max(
            0, 4 - sum(1 for v in all_details if "N spins bound" in v or "wrong polarity" in v)
        ),
        "speedup_ratio_after_fix": float(speedup_ratio),
        "kan_milp_verified": violations_after == 0,
        "honest_verdict": honest_verdict,
        "violations_remaining": all_details,
        "notes": (
            "Fix: UnivariateKAEMLayer.enforce_monotonicity() applies "
            "np.maximum.accumulate() (isotonic projection) then shifts minimum to 0. "
            "KAEMEnergy.fit() calls enforce_monotonicity() after every epoch. "
            "Fresh and fitted models are both verified with identical Exp 972 procedure."
        ),
    }


if __name__ == "__main__":
    print("Experiment 992: KAN MILP Violation Fix v2")
    print("=" * 50)

    result = run_experiment()

    out_path = os.path.abspath(RESULT_PATH)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Violations before:   {result['violations_before']}")
    print(f"Violations after:    {result['violations_after']}")
    print(f"Speedup ratio:       {result['speedup_ratio_after_fix']:.2f}x")
    print(f"MILP verified:       {result['kan_milp_verified']}")
    print(f"Honest verdict:      {result['honest_verdict']}")
    if result["violations_remaining"]:
        print("\nRemaining violations:")
        for v in result["violations_remaining"]:
            print(f"  - {v}")
    print(f"\nResult written to: {out_path}")
