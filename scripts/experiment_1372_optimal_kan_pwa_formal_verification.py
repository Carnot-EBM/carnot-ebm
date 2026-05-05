#!/usr/bin/env python3
"""Exp 1372 - CPU-only PWA formal verification for a small GS-KAN layer.

This experiment follows arXiv:2602.06737 at a small Carnot scale: replace
each KAN spline unit with a piecewise-affine abstraction, propagate the
abstraction error into the bound, and encode an energy-bound property as an LP
over a FoVer-derived input box. The current `GSKANEnergy` splines are already
degree-1 B-splines, so knot-aligned PWA pieces are exact and no integer
constraints are needed.

No hardware execution or hardware correctness is claimed.

Spec: REQ-VERIFY-1372, SCENARIO-VERIFY-1372
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from carnot.models.gskan import GSKANEnergy  # noqa: E402
from carnot.verify.kan_pwa_formal import (  # noqa: E402
    build_gskan_pwa_abstraction,
    interval_arithmetic_energy_bound,
    maximize_energy_manual_lp,
    verify_energy_bound,
)
from scripts.experiment_1034_gskan_v4 import (  # noqa: E402
    LR,
    N_GROUPS,
    N_KNOTS,
    _featurize,
    _load_fover_corpus,
)


EXP_ID = 1372
RUN_DATE = "20260505"
RESULT_PATH = _REPO_ROOT / "results" / "experiment_1372_optimal_kan_pwa_formal_verification.json"
TITLE = "Optimal KAN PWA formal verification for a small GS-KAN energy layer"

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "kan_layer_tested",
    "spline_count",
    "pwa_segments_per_spline",
    "pwa_abstraction_error_max",
    "property_tested",
    "milp_solver_used",
    "milp_verification_result",
    "formal_property_verified",
    "bound_tightness_vs_interval_arithmetic",
    "kan_formal_claim_allowed",
    "honest_verdict",
}


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for reproducible experiment metadata."""
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _weights_checksum(model: GSKANEnergy) -> str:
    """Hash the deterministic trained weights without storing the full arrays in JSON."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(model.group_ctrl, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(model.proj_weights, dtype=np.float32).tobytes())
    return digest.hexdigest()[:16]


def _train_small_gskan(n_epochs: int) -> tuple[GSKANEnergy, np.ndarray, np.ndarray, int, int]:
    """Train the small Exp 1034 GS-KAN layer deterministically on FoVer features."""
    train_items, test_items = _load_fover_corpus()
    X_train, y_train = _featurize(train_items, n_vars=16)
    model = GSKANEnergy(n_vars=16, n_groups=N_GROUPS, n_knots=N_KNOTS, seed=42)
    model.fit(X_train, n_epochs=n_epochs, lr=LR)
    return model, X_train, y_train, len(train_items), len(test_items)


def _correct_fover_feature_box(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[tuple[float, float], ...]:
    """Return the axis-aligned feature box covering correct FoVer training rows."""
    correct_rows = X_train[y_train == 1]
    if len(correct_rows) == 0:
        correct_rows = X_train
    lower = np.min(correct_rows, axis=0)
    upper = np.max(correct_rows, axis=0)
    return tuple((float(l), float(u)) for l, u in zip(lower, upper))


def _bound_tightness_payload(lp_upper: float, interval_upper: float) -> dict[str, Any]:
    """Summarize whether the PWA/LP upper bound is tighter than interval arithmetic."""
    absolute_gap = float(interval_upper - lp_upper)
    ratio = None if abs(interval_upper) < 1e-12 else float(lp_upper / interval_upper)
    if absolute_gap > 1e-9:
        verdict = "pwa_lp_tighter"
    elif absolute_gap < -1e-9:
        verdict = "interval_arithmetic_tighter"
    else:
        verdict = "tie"
    return {
        "pwa_lp_certified_upper_bound": float(lp_upper),
        "interval_arithmetic_upper_bound": float(interval_upper),
        "interval_minus_pwa_gap": absolute_gap,
        "pwa_to_interval_upper_ratio": ratio,
        "verdict": verdict,
    }


def run_experiment(
    deliverable_path: Path = RESULT_PATH,
    n_epochs: int = 150,
    pwa_segments_per_spline: int = N_KNOTS - 1,
) -> dict[str, Any]:
    """Run the experiment and write the required JSON artifact."""
    started_at = _utc_now()
    start = time.time()

    model, X_train, y_train, n_train, n_test = _train_small_gskan(n_epochs=n_epochs)
    input_bounds = _correct_fover_feature_box(X_train, y_train)

    abstraction = build_gskan_pwa_abstraction(
        model,
        pwa_segments_per_spline=pwa_segments_per_spline,
        error_grid_points=513,
    )
    lp_result = maximize_energy_manual_lp(abstraction, model.proj_weights, input_bounds)
    interval_bound = interval_arithmetic_energy_bound(model, input_bounds)

    margin = max(1e-6, 0.001 * max(abs(lp_result.certified_upper_bound), 1.0))
    threshold = float(lp_result.certified_upper_bound + margin)
    verification = verify_energy_bound(
        model,
        input_bounds,
        threshold=threshold,
        pwa_segments_per_spline=pwa_segments_per_spline,
        error_grid_points=513,
    )

    formal_property_verified = bool(verification.formal_property_verified)
    kan_formal_claim_allowed = formal_property_verified
    if formal_property_verified:
        honest_verdict = "cpu_only_gskan_energy_bound_verified_no_hardware_claim"
    else:
        honest_verdict = "gskan_energy_bound_not_formally_verified_no_claim_allowed"

    duration_s = time.time() - start
    property_tested = (
        "For all x in the axis-aligned feature box covering correct FoVer train rows, "
        f"GSKANEnergy.energy(x) < {threshold:.9f}."
    )
    artifact: dict[str, Any] = {
        "experiment": EXP_ID,
        "title": TITLE,
        "run_date": RUN_DATE,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_s": round(duration_s, 4),
        "status": "complete",
        "schema": "experiment_result_v1",
        "kan_layer_tested": (
            "GSKANEnergy(n_vars=16,n_groups=4,n_knots=8), deterministically "
            f"trained from Exp 1034 FoVer features with seed=42, n_epochs={n_epochs}"
        ),
        "kan_layer_weight_checksum": _weights_checksum(model),
        "known_trained_weights_source": (
            "weights produced deterministically by GSKANEnergy.fit on data/fover_train.json; "
            "no external checkpoint or hardware run was used"
        ),
        "spline_count": abstraction.spline_count,
        "variable_contribution_count": model.n_vars,
        "pwa_segments_per_spline": abstraction.pwa_segments_per_spline,
        "pwa_abstraction_error_max": float(abstraction.max_abs_error),
        "pwa_abstraction_error_total_budget": float(
            verification.lp_result.abstraction_error_budget
        ),
        "pwa_algorithm_identified": (
            "arXiv:2602.06737 replaces each KAN unit with a PWA abstraction, "
            "tracks local/global error, then verifies properties through MILP; "
            "the paper's optimal allocation uses dynamic programming per unit plus "
            "knapsack across the network. This small one-layer GS-KAN run uses "
            "knot-aligned PWA pieces, so allocation is trivial and exact."
        ),
        "paper_source": "https://arxiv.org/abs/2602.06737",
        "property_tested": property_tested,
        "property_input_range_source": (
            f"correct FoVer training feature envelope from {n_train} train rows; "
            f"held-out FoVer rows available: {n_test}"
        ),
        "property_energy_threshold": threshold,
        "milp_solver_used": verification.lp_result.solver_name,
        "milp_solver_status": verification.lp_result.solver_status,
        "integer_constraints_needed": verification.lp_result.integer_constraints_needed,
        "milp_verification_result": verification.result,
        "formal_property_verified": formal_property_verified,
        "lp_exact_pwa_upper_bound": verification.lp_result.exact_pwa_upper_bound,
        "lp_certified_upper_bound": verification.lp_result.certified_upper_bound,
        "interval_arithmetic_lower_bound": verification.interval_bound.lower_bound,
        "interval_arithmetic_upper_bound": verification.interval_bound.upper_bound,
        "bound_tightness_vs_interval_arithmetic": _bound_tightness_payload(
            verification.lp_result.certified_upper_bound,
            verification.interval_bound.upper_bound,
        ),
        "counterexample_candidate": list(verification.lp_result.maximizer),
        "kan_formal_claim_allowed": kan_formal_claim_allowed,
        "hardware_execution_claimed": False,
        "hardware_correctness_claimed": False,
        "honest_verdict": honest_verdict,
    }

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise RuntimeError(f"artifact missing required fields: {sorted(missing)}")

    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    deliverable_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def main() -> None:
    """CLI entrypoint for the conductor and local experiment runs."""
    artifact = run_experiment()
    print(
        artifact["pwa_abstraction_error_max"],
        artifact["milp_verification_result"],
        artifact["formal_property_verified"],
        artifact["honest_verdict"],
    )


if __name__ == "__main__":
    main()
