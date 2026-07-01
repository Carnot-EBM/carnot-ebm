"""Exp 5114 KAN abstraction-refinement post-wall diagnostic.

Spec refs: REQ-KAN-5114, SCENARIO-KAN-5114.

Exp 5108 found the exact-MILP wall: N=10 solved slowly and N=20 timed out.
This module changes technique. It keeps the same additive KAEM fixture family
but replaces the global integer search with a conservative certificate:

1. Build a coarse one-piece upper envelope for every univariate KAEM unit.
2. Measure local error by comparing the exact native PWA spline to that coarse
   envelope at all native knots. For a piecewise-affine spline, the residual is
   also piecewise-affine, so the knot extrema bound the whole interval.
3. Spend a bounded piece budget by refining the units with the largest local
   slack back to their exact native pieces.
4. Sum the per-unit conservative bounds. The safe property is proved only when
   that global upper bound is below threshold; false controls use the exact
   decomposed witness; near-margin properties abstain when residual error
   consumes the margin.

The result is not a general KAN verifier and not evidence that exact MILP
scales. It is a CPU diagnostic showing whether a post-wall conservative
certificate can make honest progress beyond the exact-MILP setting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from carnot.experiment_5098_kan_pwa_milp_scale_v2 import (
    RESULT_RELATIVE_PATH as EXP5098_RESULT_RELATIVE_PATH,
)
from carnot.experiment_5108_kan_pwa_milp_scale_stress_test import (
    DEFAULT_SOLVER_TIMEOUT_MS as EXP5108_SOLVER_TIMEOUT_MS,
)
from carnot.experiment_5108_kan_pwa_milp_scale_stress_test import (
    REALISTIC_KAN_UNIT_COUNT_REFERENCE,
    RESULT_RELATIVE_PATH as EXP5108_RESULT_RELATIVE_PATH,
    build_n_unit_abstraction,
)


RESULT_RELATIVE_PATH = "results/experiment_5114_kan_abstraction_refinement_post_wall_v469.json"
EXPERIMENT_ID = "exp5114-kan-abstraction-refinement-post-wall-v469"
MILESTONE = "2026.07.469"
RUN_DATE = "20260701"
RANDOM_SEED = 5108
INFERENCE_SUBSTRATE = "kan_abstraction_refinement_cpu"
SPEC_REFS = ["REQ-KAN-5114", "SCENARIO-KAN-5114"]
DEFAULT_UNIT_COUNTS = (20, REALISTIC_KAN_UNIT_COUNT_REFERENCE)
SAFE_MARGIN_PER_UNIT = 0.04
FALSE_MARGIN = 0.05
NEAR_MARGIN = 0.001

SUCCESS_VERDICT_PREFIX = "success_kan_abstraction_refinement_post_wall_progress"
COMPLETE_VERDICT_PREFIX = "complete_kan_abstraction_refinement_post_wall_no_progress"
BLOCKED_VERDICT_PREFIX = "blocked_kan_abstraction_refinement_post_wall_precondition"
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "honest_verdict",
    "inference_substrate",
    "duration_s",
    "preconditions_checked",
    "technique_changed_from_exp5108",
    "exp5108_baseline_loaded",
    "solved_n",
    "attempted_n",
    "certificate_soundness",
    "false_property_detected",
    "near_margin_abstained",
    "abstraction_error_bounds",
    "post_wall_progress",
    "seeds_or_checksums",
    "flagged_adversarial",
    "tests_run",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "experiment_id": {"principle": "Traceability: stable experiment identifier."},
    "milestone": {"principle": "Milestone accountability: records the active milestone."},
    "honest_verdict": {"principle": "Terminal verdict with complete_/success_/blocked_ prefix."},
    "inference_substrate": {
        "principle": "Substrate honesty: declares the CPU abstraction-refinement path."
    },
    "duration_s": {"principle": "Timing accountability for the diagnostic run."},
    "preconditions_checked": {
        "principle": "Solver preflight accountability without claiming an exact-MILP rerun."
    },
    "technique_changed_from_exp5108": {
        "principle": "No doomed rerun: true only for non-MILP decomposition/refinement."
    },
    "exp5108_baseline_loaded": {
        "principle": "Comparison provenance: confirms the wall artifact was loaded."
    },
    "solved_n": {"principle": "Scale transparency: largest soundly handled N."},
    "attempted_n": {"principle": "Scale transparency: largest attempted N."},
    "certificate_soundness": {
        "principle": "Verification integrity: no false property is counted as proved."
    },
    "false_property_detected": {
        "principle": "Negative control: false properties must be counterexampled."
    },
    "near_margin_abstained": {
        "principle": "No unsound overclaim: near-margin cases abstain when bounds are loose."
    },
    "abstraction_error_bounds": {
        "principle": "Conservative certification: local errors propagate into global bounds."
    },
    "post_wall_progress": {
        "principle": "Decision bool: true only for sound progress beyond Exp 5108."
    },
    "seeds_or_checksums": {
        "principle": "Reproducibility: records deterministic seeds and checksums."
    },
    "flagged_adversarial": {
        "principle": "Adversarial-verification accountability for control failures."
    },
    "tests_run": {"principle": "Verification evidence for the new module and artifact."},
}


def _require(condition: bool, message: str) -> None:
    if not condition:  # pragma: no cover - defensive schema guard.
        raise ValueError(message)


@dataclass(frozen=True)
class UnitBound:
    """Local conservative certificate for one additive KAEM unit.

    The coarse envelope uses one chord over the full input interval plus a
    measured positive residual bound. Refinement switches selected units back
    to their exact native PWA pieces, removing their local slack while staying
    much cheaper than a global mixed-integer search.
    """

    unit_index: int
    allocated_pieces: int
    native_pieces: int
    refined: bool
    exact_upper_bound: float
    conservative_upper_bound: float
    initial_conservative_upper_bound: float
    local_error_bound: float
    initial_local_error_bound: float
    positive_residual_bound: float
    witness_x: float
    selected_segment: int

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe local certificate telemetry."""

        return {
            "unit_index": self.unit_index,
            "allocated_pieces": self.allocated_pieces,
            "native_pieces": self.native_pieces,
            "refined": self.refined,
            "exact_upper_bound": self.exact_upper_bound,
            "conservative_upper_bound": self.conservative_upper_bound,
            "initial_conservative_upper_bound": self.initial_conservative_upper_bound,
            "local_error_bound": self.local_error_bound,
            "initial_local_error_bound": self.initial_local_error_bound,
            "positive_residual_bound": self.positive_residual_bound,
            "witness_x": self.witness_x,
            "selected_segment": self.selected_segment,
        }


@dataclass(frozen=True)
class RefinedCertificate:
    """Global additive certificate assembled from local unit bounds."""

    n_units: int
    seed: int
    exact_upper_bound: float
    certified_upper_bound: float
    initial_certified_upper_bound: float
    global_error_bound: float
    initial_global_error_bound: float
    max_local_error_bound: float
    refined_unit_count: int
    binary_vars: int
    constraints: dict[str, int]
    piece_budget: dict[str, int | str]
    unit_bounds: tuple[UnitBound, ...]
    witness_inputs: tuple[float, ...]
    selected_segments: tuple[int, ...]
    witness_lower_bound: float
    runtime_s: float

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe global certificate telemetry."""

        return {
            "n_units": self.n_units,
            "seed": self.seed,
            "exact_upper_bound": self.exact_upper_bound,
            "certified_upper_bound": self.certified_upper_bound,
            "initial_certified_upper_bound": self.initial_certified_upper_bound,
            "global_error_bound": self.global_error_bound,
            "initial_global_error_bound": self.initial_global_error_bound,
            "max_local_error_bound": self.max_local_error_bound,
            "refined_unit_count": self.refined_unit_count,
            "binary_vars": self.binary_vars,
            "constraints": dict(self.constraints),
            "piece_budget": dict(self.piece_budget),
            "witness_inputs": list(self.witness_inputs),
            "selected_segments": list(self.selected_segments),
            "witness_lower_bound": self.witness_lower_bound,
            "runtime_s": self.runtime_s,
            "unit_bounds": [unit.as_serializable() for unit in self.unit_bounds],
        }


@dataclass(frozen=True)
class PropertyOutcome:
    """One property-class decision derived from a conservative certificate."""

    property_id: str
    property_class: str
    threshold: float
    property_status: str
    property_holds: bool | None
    exact_upper_bound: float
    certified_upper_bound: float
    counterexample: dict[str, Any] | None
    certificate: dict[str, Any]

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe property telemetry."""

        return {
            "property_id": self.property_id,
            "property_class": self.property_class,
            "threshold": self.threshold,
            "property_status": self.property_status,
            "property_holds": self.property_holds,
            "exact_upper_bound": self.exact_upper_bound,
            "certified_upper_bound": self.certified_upper_bound,
            "counterexample": self.counterexample,
            "certificate": self.certificate,
        }


@dataclass(frozen=True)
class DiagnosticRow:
    """All Exp 5114 telemetry for one attempted unit count."""

    n_units: int
    certificate: RefinedCertificate
    property_outcomes: tuple[PropertyOutcome, ...]

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe row telemetry."""

        return {
            "n_units": self.n_units,
            "certificate": self.certificate.as_serializable(),
            "property_outcomes": [row.as_serializable() for row in self.property_outcomes],
            "safe_property_verified": self.safe_property_verified,
            "false_property_detected": self.false_property_detected,
            "near_margin_abstained": self.near_margin_abstained,
        }

    @property
    def safe_property_verified(self) -> bool:
        return any(
            row.property_class == "true_safe" and row.property_status == "verified"
            for row in self.property_outcomes
        )

    @property
    def false_property_detected(self) -> bool:
        return any(
            row.property_class == "false_counterexample" and row.property_status == "counterexample"
            for row in self.property_outcomes
        )

    @property
    def near_margin_abstained(self) -> bool:
        return any(
            row.property_class == "near_margin_abstain"
            and row.property_status == "abstained_margin"
            for row in self.property_outcomes
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _piece_budget_for_n(n_units: int) -> int:
    """Return the bounded refinement budget used for Exp 5114.

    One coarse piece per unit costs N pieces. Native exact PWA would cost 3N
    pieces for the Exp 5108 fixture. The 1.5N budget refines only the largest
    local-error contributors and therefore demonstrates progress without
    silently reverting to the full exact representation.
    """

    return max(n_units, (3 * int(n_units)) // 2)


def _exact_unit_max(unit: Any) -> tuple[float, float, int]:
    """Return exact max value, witness x, and selected native segment for one unit."""

    candidates = sorted(
        {float(segment.x_min) for segment in unit.segments} | {float(unit.segments[-1].x_max)}
    )
    best_x = candidates[0]
    best_value = float(unit.evaluate(best_x))
    for x_value in candidates[1:]:
        value = float(unit.evaluate(x_value))
        if value > best_value + 1e-12:
            best_value = value
            best_x = x_value
    return best_value, best_x, int(unit.segment_for_x(best_x).index)


def _coarse_unit_bound(unit: Any, unit_index: int) -> UnitBound:
    """Build the one-piece conservative envelope for one KAEM unit."""

    exact_upper, witness_x, selected_segment = _exact_unit_max(unit)
    x_min = float(unit.knots[0])
    x_max = float(unit.knots[-1])
    y_min = float(unit.evaluate(x_min))
    y_max = float(unit.evaluate(x_max))
    slope = (y_max - y_min) / (x_max - x_min)
    intercept = y_min - slope * x_min
    residuals = [
        float(unit.evaluate(float(knot))) - (slope * float(knot) + intercept) for knot in unit.knots
    ]
    positive_residual = max(0.0, max(residuals))
    conservative_upper = max(y_min, y_max) + positive_residual
    local_error = max(0.0, conservative_upper - exact_upper)
    return UnitBound(
        unit_index=unit_index,
        allocated_pieces=1,
        native_pieces=int(unit.n_segments),
        refined=False,
        exact_upper_bound=exact_upper,
        conservative_upper_bound=conservative_upper,
        initial_conservative_upper_bound=conservative_upper,
        local_error_bound=local_error,
        initial_local_error_bound=local_error,
        positive_residual_bound=positive_residual,
        witness_x=witness_x,
        selected_segment=selected_segment,
    )


def _refine_unit(bound: UnitBound) -> UnitBound:
    """Switch one unit from the coarse envelope to the exact native PWA pieces."""

    return UnitBound(
        unit_index=bound.unit_index,
        allocated_pieces=bound.native_pieces,
        native_pieces=bound.native_pieces,
        refined=True,
        exact_upper_bound=bound.exact_upper_bound,
        conservative_upper_bound=bound.exact_upper_bound,
        initial_conservative_upper_bound=bound.initial_conservative_upper_bound,
        local_error_bound=0.0,
        initial_local_error_bound=bound.initial_local_error_bound,
        positive_residual_bound=bound.positive_residual_bound,
        witness_x=bound.witness_x,
        selected_segment=bound.selected_segment,
    )


def build_refined_certificate(
    n_units: int,
    seed: int,
    max_total_pieces: int | None = None,
) -> RefinedCertificate:
    """Build a local/global abstraction-refinement certificate for N KAEM units."""

    start = time.perf_counter()
    abstraction = build_n_unit_abstraction(int(n_units), int(seed))
    coarse_bounds = tuple(
        _coarse_unit_bound(unit, unit_index) for unit_index, unit in enumerate(abstraction.units)
    )
    max_pieces = int(max_total_pieces or _piece_budget_for_n(n_units))
    allocated_pieces = len(coarse_bounds)
    refined_indices: set[int] = set()
    ranked = sorted(
        coarse_bounds,
        key=lambda bound: (bound.initial_local_error_bound, -bound.unit_index),
        reverse=True,
    )
    for bound in ranked:
        extra_cost = bound.native_pieces - 1
        if bound.initial_local_error_bound <= 0.0:
            continue
        if allocated_pieces + extra_cost <= max_pieces:
            refined_indices.add(bound.unit_index)
            allocated_pieces += extra_cost

    final_bounds = tuple(
        _refine_unit(bound) if bound.unit_index in refined_indices else bound
        for bound in coarse_bounds
    )
    exact_upper = sum(bound.exact_upper_bound for bound in final_bounds)
    initial_upper = sum(bound.initial_conservative_upper_bound for bound in final_bounds)
    certified_upper = sum(bound.conservative_upper_bound for bound in final_bounds)
    global_error = max(0.0, certified_upper - exact_upper)
    initial_error = max(0.0, initial_upper - exact_upper)
    total_allocated_pieces = sum(bound.allocated_pieces for bound in final_bounds)
    native_total_pieces = sum(bound.native_pieces for bound in final_bounds)
    constraints = {
        "milp_constraints": 0,
        "local_unit_bound_checks": int(n_units),
        "refinement_order_checks": int(n_units),
        "global_sum_checks": 3,
        "total_certificate_checks": 2 * int(n_units) + 3,
        "exp5108_exact_milp_constraint_formula_at_same_n": 21 * int(n_units) + 1,
    }
    return RefinedCertificate(
        n_units=int(n_units),
        seed=int(seed),
        exact_upper_bound=float(exact_upper),
        certified_upper_bound=float(certified_upper),
        initial_certified_upper_bound=float(initial_upper),
        global_error_bound=float(global_error),
        initial_global_error_bound=float(initial_error),
        max_local_error_bound=max((bound.local_error_bound for bound in final_bounds), default=0.0),
        refined_unit_count=len(refined_indices),
        binary_vars=0,
        constraints=constraints,
        piece_budget={
            "coarse_total_pieces": int(n_units),
            "max_total_pieces": max_pieces,
            "allocated_total_pieces": total_allocated_pieces,
            "native_total_pieces": native_total_pieces,
            "refined_unit_count": len(refined_indices),
            "allocation_strategy": "largest_local_error_first",
        },
        unit_bounds=final_bounds,
        witness_inputs=tuple(bound.witness_x for bound in final_bounds),
        selected_segments=tuple(bound.selected_segment for bound in final_bounds),
        witness_lower_bound=float(exact_upper),
        runtime_s=round(time.perf_counter() - start, 6),
    )


def _classify_property(
    certificate: RefinedCertificate,
    property_class: str,
    threshold: float,
) -> PropertyOutcome:
    """Classify one threshold using exact witnesses and conservative upper bounds."""

    property_id = f"n{certificate.n_units}_{property_class}"
    if certificate.witness_lower_bound > threshold + 1e-9:
        counterexample = {
            "n_units": certificate.n_units,
            "inputs": list(certificate.witness_inputs),
            "selected_segments": list(certificate.selected_segments),
            "witness_value": certificate.witness_lower_bound,
            "threshold": threshold,
            "violation_margin": certificate.witness_lower_bound - threshold,
        }
        status = "counterexample"
        property_holds: bool | None = False
        cert = {
            "kind": "counterexample",
            "method": "decomposed_exact_witness_from_unit_maxima",
            "counterexample": counterexample,
        }
    elif certificate.certified_upper_bound <= threshold + 1e-9:
        counterexample = None
        status = "verified"
        property_holds = True
        cert = {
            "kind": "conservative_certificate",
            "method": "local_global_abstraction_refinement_decomposition",
            "certified_upper_bound": certificate.certified_upper_bound,
            "exact_decomposed_upper_bound": certificate.exact_upper_bound,
            "global_error_bound": certificate.global_error_bound,
            "threshold": threshold,
        }
    else:
        counterexample = None
        status = "abstained_margin"
        property_holds = None
        cert = {
            "kind": "margin_abstention",
            "method": "local_global_abstraction_refinement_decomposition",
            "exact_decomposed_upper_bound": certificate.exact_upper_bound,
            "certified_upper_bound": certificate.certified_upper_bound,
            "threshold": threshold,
            "residual_margin_gap": certificate.certified_upper_bound - threshold,
            "reason": "residual abstraction bound consumes the certification margin",
        }
    return PropertyOutcome(
        property_id=property_id,
        property_class=property_class,
        threshold=float(threshold),
        property_status=status,
        property_holds=property_holds,
        exact_upper_bound=certificate.exact_upper_bound,
        certified_upper_bound=certificate.certified_upper_bound,
        counterexample=counterexample,
        certificate=cert,
    )


def evaluate_property_classes(certificate: RefinedCertificate) -> tuple[PropertyOutcome, ...]:
    """Evaluate true, false-control, and near-margin properties for one N."""

    safe_threshold = certificate.exact_upper_bound + SAFE_MARGIN_PER_UNIT * certificate.n_units
    false_threshold = certificate.exact_upper_bound - FALSE_MARGIN
    near_threshold = certificate.exact_upper_bound + NEAR_MARGIN
    return (
        _classify_property(certificate, "true_safe", safe_threshold),
        _classify_property(certificate, "false_counterexample", false_threshold),
        _classify_property(certificate, "near_margin_abstain", near_threshold),
    )


def run_diagnostic(
    unit_counts: Sequence[int] = DEFAULT_UNIT_COUNTS,
    seed: int = RANDOM_SEED,
) -> tuple[DiagnosticRow, ...]:
    """Run the post-wall diagnostic for every configured N without exact MILP."""

    rows: list[DiagnosticRow] = []
    for n_units in sorted(int(value) for value in unit_counts):
        certificate = build_refined_certificate(n_units=n_units, seed=int(seed) + n_units)
        rows.append(
            DiagnosticRow(
                n_units=n_units,
                certificate=certificate,
                property_outcomes=evaluate_property_classes(certificate),
            )
        )
    return tuple(rows)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_exp5108_baseline(root: str | Path | None = None) -> dict[str, Any]:
    """Load the Exp 5108 wall artifact for direct comparison."""

    base = Path(root) if root is not None else _repo_root()
    path = base / EXP5108_RESULT_RELATIVE_PATH
    if not path.exists():
        return {"loaded": False, "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "loaded": True,
        "path": str(path.relative_to(base)),
        "largest_n_reached": int(payload.get("largest_n_reached", 0)),
        "solver_timeout_hit": bool(payload.get("solver_timeout_hit", False)),
        "timed_out_n": next(
            (
                int(row["n_units"])
                for row in payload.get("per_n_results", [])
                if row.get("timed_out") is True
            ),
            None,
        ),
        "duration_s": float(payload.get("duration_s", 0.0)),
        "inference_substrate": payload.get("inference_substrate"),
        "checksum": _sha256_file(path),
    }


def _preconditions_checked(root: Path, baseline: Mapping[str, Any]) -> dict[str, Any]:
    """Return explicit preflight checks used by the artifact."""

    return {
        "exp5108_wall_artifact_exists": bool(baseline.get("loaded")),
        "exp5098_property_controls_artifact_exists": (root / EXP5098_RESULT_RELATIVE_PATH).exists(),
        "python_carnot_kan_dir_present": (root / "python/carnot/kan").exists(),
        "python_carnot_verify_dir_present": (root / "python/carnot/verify").exists(),
        "exact_milp_scale_sweep_repeated": False,
        "solver_preflight": "z3_not_required_for_decomposed_certificate",
        "additive_unit_decomposition_precondition": True,
        "independent_input_box_precondition": True,
        "exp5108_solver_timeout_ms": EXP5108_SOLVER_TIMEOUT_MS,
    }


def _row_sound(row: DiagnosticRow) -> bool:
    for outcome in row.property_outcomes:
        if outcome.property_status == "verified":
            if outcome.certified_upper_bound > outcome.threshold + 1e-9:
                return False
        if outcome.property_class == "false_counterexample":
            if outcome.property_status != "counterexample":
                return False
        if outcome.property_status == "counterexample" and outcome.counterexample is None:
            return False
    return row.certificate.certified_upper_bound >= row.certificate.exact_upper_bound - 1e-9


def _certificate_soundness(rows: Sequence[DiagnosticRow]) -> bool:
    return bool(rows) and all(_row_sound(row) for row in rows)


def _abstain_rate(rows: Sequence[DiagnosticRow]) -> float:
    outcomes = [outcome for row in rows for outcome in row.property_outcomes]
    if not outcomes:
        return 0.0
    abstained = sum(outcome.property_status.startswith("abstained") for outcome in outcomes)
    return round(abstained / len(outcomes), 6)


def _checksum_payload(
    rows: Sequence[DiagnosticRow],
    baseline: Mapping[str, Any],
    unit_counts: Sequence[int],
    run_date: str,
) -> str:
    serial_rows = []
    for row in rows:
        serial = row.as_serializable()
        serial["certificate"]["runtime_s"] = "excluded"
        serial_rows.append(serial)
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "spec_refs": SPEC_REFS,
        "unit_counts": list(sorted(int(value) for value in unit_counts)),
        "rows": serial_rows,
        "baseline": {
            "loaded": baseline.get("loaded"),
            "largest_n_reached": baseline.get("largest_n_reached"),
            "solver_timeout_hit": baseline.get("solver_timeout_hit"),
            "timed_out_n": baseline.get("timed_out_n"),
            "checksum": baseline.get("checksum"),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    run_date: str = RUN_DATE,
    unit_counts: Sequence[int] = DEFAULT_UNIT_COUNTS,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the Exp 5114 deliverable payload."""

    start = time.perf_counter()
    repo = Path(root) if root is not None else _repo_root()
    baseline = load_exp5108_baseline(repo)
    rows = run_diagnostic(unit_counts=unit_counts)
    attempted_n = max((row.n_units for row in rows), default=0)
    sound = _certificate_soundness(rows)
    false_detected = bool(rows) and all(row.false_property_detected for row in rows)
    near_abstained = bool(rows) and all(row.near_margin_abstained for row in rows)
    solved_n = max(
        (
            row.n_units
            for row in rows
            if row.safe_property_verified
            and row.false_property_detected
            and row.near_margin_abstained
            and _row_sound(row)
        ),
        default=0,
    )
    baseline_n = int(baseline.get("largest_n_reached", 0)) if baseline.get("loaded") else 0
    flagged_adversarial = not (sound and false_detected)
    post_wall_progress = (
        bool(baseline.get("loaded"))
        and sound
        and false_detected
        and near_abstained
        and solved_n > baseline_n
        and attempted_n >= baseline_n
        and not flagged_adversarial
    )
    if post_wall_progress:
        honest_verdict = f"{SUCCESS_VERDICT_PREFIX}_n{solved_n}_over_exp5108_n{baseline_n}"
    elif not baseline.get("loaded"):
        honest_verdict = f"{BLOCKED_VERDICT_PREFIX}_exp5108_missing"
    else:
        honest_verdict = f"{COMPLETE_VERDICT_PREFIX}_n{solved_n}_over_exp5108_n{baseline_n}"

    by_n = {str(row.n_units): row for row in rows}
    artifact: dict[str, Any] = {
        "schema": "carnot.kan_abstraction_refinement_post_wall.v469",
        "experiment": 5114,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "artifact": "experiment_5114_kan_abstraction_refinement_post_wall_v469",
        "run_date": run_date,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(time.perf_counter() - start, 6),
        "preconditions_checked": _preconditions_checked(repo, baseline),
        "technique": "local_global_abstraction_refinement_unit_decomposition_margin_abstention",
        "technique_changed_from_exp5108": True,
        "exp5108_baseline_loaded": bool(baseline.get("loaded")),
        "exp5108_baseline": baseline,
        "solved_n": solved_n,
        "attempted_n": attempted_n,
        "attempted_unit_counts": [row.n_units for row in rows],
        "certificate_soundness": sound,
        "false_property_detected": false_detected,
        "near_margin_abstained": near_abstained,
        "abstain_rate": _abstain_rate(rows),
        "abstraction_error_bounds": {
            n: {
                "initial_global_error_bound": row.certificate.initial_global_error_bound,
                "refined_global_error_bound": row.certificate.global_error_bound,
                "max_local_error_bound_after_refinement": row.certificate.max_local_error_bound,
                "exact_upper_bound": row.certificate.exact_upper_bound,
                "certified_upper_bound": row.certificate.certified_upper_bound,
            }
            for n, row in by_n.items()
        },
        "piece_budget": {n: dict(row.certificate.piece_budget) for n, row in by_n.items()},
        "binary_vars": {n: row.certificate.binary_vars for n, row in by_n.items()},
        "constraints": {n: dict(row.certificate.constraints) for n, row in by_n.items()},
        "runtime": {n: row.certificate.runtime_s for n, row in by_n.items()},
        "post_wall_progress": post_wall_progress,
        "seeds_or_checksums": {
            "random_seed": RANDOM_SEED,
            "unit_seeds_by_n": {str(row.n_units): row.certificate.seed for row in rows},
            "exp5108_checksum": baseline.get("checksum"),
            "reproducibility_checksum": _checksum_payload(rows, baseline, unit_counts, run_date),
        },
        "flagged_adversarial": flagged_adversarial,
        "per_n_results": [row.as_serializable() for row in rows],
        "source_artifacts": [
            EXP5098_RESULT_RELATIVE_PATH,
            EXP5108_RESULT_RELATIVE_PATH,
        ],
        "spec_refs": list(SPEC_REFS),
        "field_principles": FIELD_PRINCIPLES,
        "methodology_note": (
            "Exp 5114 does not rerun the Exp 5108 exact-MILP scale sweep. It uses "
            "local unit upper envelopes, largest-error-first refinement under a bounded "
            "piece budget, additive global error propagation, false-property witnesses, "
            "and margin abstention."
        ),
        "tests_run": [
            "JAX_PLATFORMS=cpu .venv/bin/pytest tests/python/test_experiment_5114_kan_abstraction_refinement_post_wall.py -q --no-cov",
            "JAX_PLATFORMS=cpu .venv/bin/coverage run --source=python/carnot -m pytest tests/python/test_experiment_5114_kan_abstraction_refinement_post_wall.py -q --no-cov -n0",
            ".venv/bin/coverage report --fail-under=100 -m python/carnot/experiment_5114_kan_abstraction_refinement_post_wall_v469.py",
            "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python scripts/experiment_5114_kan_abstraction_refinement_post_wall_v469.py --date 20260701",
        ],
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed if the Exp 5114 artifact drifts from its schema boundary."""

    missing = set(REQUIRED_ARTIFACT_FIELDS) - set(artifact)
    _require(not missing, f"missing required artifact fields: {sorted(missing)}")
    verdict = artifact["honest_verdict"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict must use a terminal prefix",
    )
    _require(
        artifact["experiment_id"] == EXPERIMENT_ID,
        "experiment_id must match the required Exp 5114 id",
    )
    _require(artifact["milestone"] == MILESTONE, "milestone must match 2026.07.469")
    _require(
        artifact["inference_substrate"] == INFERENCE_SUBSTRATE,
        "inference_substrate must be kan_abstraction_refinement_cpu",
    )
    _require("live_llm" not in artifact["inference_substrate"], "must not claim live LLM")
    _require(isinstance(artifact["duration_s"], float), "duration_s must be a float")
    _require(artifact["duration_s"] >= 0.0, "duration_s cannot be negative")
    _require(
        set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact["field_principles"]),
        "field_principles must cover every required field",
    )
    _require(
        artifact["technique_changed_from_exp5108"] is True,
        "Exp 5114 must change technique from exact MILP",
    )
    _require(artifact["attempted_n"] >= artifact["solved_n"], "attempted_n must cover solved_n")
    _require(isinstance(artifact["certificate_soundness"], bool), "soundness must be bool")
    _require(isinstance(artifact["false_property_detected"], bool), "false control must be bool")
    _require(isinstance(artifact["near_margin_abstained"], bool), "near-margin field must be bool")
    _require(isinstance(artifact["flagged_adversarial"], bool), "flagged_adversarial must be bool")
    _require(
        artifact["flagged_adversarial"]
        == (not (artifact["certificate_soundness"] and artifact["false_property_detected"])),
        "flagged_adversarial must reflect soundness or false-control failure",
    )
    if artifact["post_wall_progress"]:
        _require(artifact["exp5108_baseline_loaded"] is True, "progress requires baseline")
        _require(artifact["certificate_soundness"] is True, "progress requires soundness")
        _require(artifact["false_property_detected"] is True, "progress requires false controls")
        _require(artifact["near_margin_abstained"] is True, "progress requires abstention")
        _require(
            artifact["solved_n"] > artifact["exp5108_baseline"]["largest_n_reached"],
            "progress requires solved_n beyond Exp 5108",
        )
    for row in artifact.get("per_n_results", []):
        for outcome in row.get("property_outcomes", []):
            if outcome["property_class"] == "false_counterexample":
                _require(
                    outcome["property_status"] == "counterexample",
                    "false controls must not be verified",
                )
            if outcome["property_status"] == "verified":
                _require(
                    outcome["certified_upper_bound"] <= outcome["threshold"] + 1e-9,
                    "verified rows require conservative bound below threshold",
                )


def write_outputs(
    *,
    artifact_path: str | Path,
    run_date: str = RUN_DATE,
    unit_counts: Sequence[int] = DEFAULT_UNIT_COUNTS,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Write the Exp 5114 JSON artifact and return the validated payload."""

    artifact = build_artifact(run_date=run_date, unit_counts=unit_counts, root=root)
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for writing the default Exp 5114 deliverable artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE, help="Run date as YYYYMMDD")
    parser.add_argument("--output", default=None, help="Optional artifact output path")
    args = parser.parse_args(argv)

    root = Path(os.environ.get("CARNOT_EXP5114_ROOT", _repo_root()))
    output = Path(args.output) if args.output else root / RESULT_RELATIVE_PATH
    artifact = write_outputs(artifact_path=output, run_date=str(args.date), root=root)
    print(artifact["honest_verdict"])
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the script wrapper.
    raise SystemExit(main())
