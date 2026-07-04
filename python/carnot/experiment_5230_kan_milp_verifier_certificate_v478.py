"""Exp 5230 tiny KAEM PWA/MILP verifier certificate pilot.

Spec refs: REQ-KAN-5230, SCENARIO-KAN-5230.

This module ties a real Carnot KAEM energy layer to a tiny certificate. It does
not implement GRS-KAN or a broad KAN verifier. The useful boundary is smaller:
export `UnivariateKAEMLayer`'s existing linear-interpolation splines through the
PWA helpers used by Exp 5080/5091, then certify two properties on a small input
box with deterministic CPU checks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp

from carnot.experiment_5080_kan_pwa_milp_bridge_v466 import (
    KAN_COMPONENT_PATH,
    build_pwa_abstraction,
)
from carnot.experiment_5091_kan_pwa_milp_scale_v467 import MultiUnitPWAAbstraction
from carnot.models.kaem_energy import UnivariateKAEMLayer


JsonDict = dict[str, Any]

RUN_DATE = "20260704"
EXPERIMENT_ID = "exp5230-kan-milp-verifier-certificate-v478"
SCHEMA = "carnot.experiment_5230.kan_milp_verifier_certificate.v478"
RESULT_RELATIVE_PATH = Path("results/experiment_5230_kan_milp_verifier_certificate_v478.json")
TARGET_MODULE = KAN_COMPONENT_PATH
INFERENCE_SUBSTRATE = "deterministic_pwa_milp_certificate"
SPEC_REFS = ("REQ-KAN-5230", "SCENARIO-KAN-5230")
RANDOM_SEED = 5230
INPUT_BOX = ((-0.25, 0.5), (-0.25, 0.5))
UNSAFE_DECISION_THRESHOLD = 0.7
CONTROL_POINTS = (
    (0.0, 0.1, 0.3, 0.6),
    (0.0, 0.05, 0.2, 0.4),
)
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")

FIELD_PRINCIPLES = {
    "kan_certificate_produced": (
        "True only when monotonicity and no-unsafe-decision checks are both verified."
    ),
    "certificate_path": "Path to the written certificate artifact, or null when blocked.",
    "target_module": "The real Carnot KAEM/KAN-style module being abstracted.",
    "properties_checked": "Bounded properties checked by deterministic PWA/MILP evidence.",
    "bound_tightness": (
        "Safety threshold minus certified maximum energy for the no-unsafe-decision check."
    ),
    "reused_existing_helpers": (
        "True only when the pilot reuses the existing Exp 5080/5091 PWA abstraction helpers."
    ),
    "tests_run": "Commands run for this certificate pilot, with pass/fail status.",
    "inference_substrate": "Must be deterministic_pwa_milp_certificate.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state whether a certificate was produced."
    ),
}
REQUIRED_WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@dataclass(frozen=True)
class PropertyCertificate:
    """One bounded property result in the tiny KAEM certificate."""

    property_id: str
    method: str
    verified: bool
    threshold: float | None
    certified_upper_bound: float | None
    bound_tightness: float | None
    witness_inputs: tuple[float, ...] | None
    min_slope: float | None
    solver_status: str
    details: JsonDict

    def as_serializable(self) -> JsonDict:
        return {
            "property_id": self.property_id,
            "method": self.method,
            "verified": self.verified,
            "threshold": self.threshold,
            "certified_upper_bound": self.certified_upper_bound,
            "bound_tightness": self.bound_tightness,
            "witness_inputs": list(self.witness_inputs) if self.witness_inputs else None,
            "min_slope": self.min_slope,
            "solver_status": self.solver_status,
            "details": self.details,
        }


@dataclass(frozen=True)
class CertificateResult:
    """All evidence needed to decide whether Exp 5230 produced a certificate."""

    produced: bool
    solver_available: bool
    solver_status: str
    blocked_reason: str | None
    bound_tightness: float | None
    abstraction: MultiUnitPWAAbstraction
    property_results: tuple[PropertyCertificate, ...]


def wrap_field(field: str, value: Any) -> JsonDict:
    """Return the repository's principle-wrapped artifact field shape."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def detect_solver() -> str:
    """Return the deterministic MILP-compatible backend available locally."""

    return "z3" if importlib.util.find_spec("z3") is not None else ""


def build_tiny_certificate_layer() -> UnivariateKAEMLayer:
    """Create the real KAEM layer fixture used for the certificate."""

    layer = UnivariateKAEMLayer(n_vars=2, n_knots=4, key=jax.random.PRNGKey(RANDOM_SEED))
    layer.control_points = jnp.array(CONTROL_POINTS, dtype=jnp.float32)
    return layer


def build_certificate_abstraction(layer: UnivariateKAEMLayer) -> MultiUnitPWAAbstraction:
    """Export each KAEM variable with the existing Exp 5080 PWA helper."""

    units = tuple(build_pwa_abstraction(layer, variable_index=index) for index in range(layer.n_vars))
    return MultiUnitPWAAbstraction(
        component_path=TARGET_MODULE,
        units=units,
        local_error_budget=0.0,
        global_error_budget=0.0,
    )


def _segment_overlaps(segment: Any, lo: float, hi: float) -> bool:
    return segment.x_max >= lo - 1e-12 and segment.x_min <= hi + 1e-12


def monotonicity_certificate(abstraction: MultiUnitPWAAbstraction) -> PropertyCertificate:
    """Check bounded monotonicity by inspecting PWA slopes on the tiny box."""

    slope_rows: list[JsonDict] = []
    for unit_index, unit in enumerate(abstraction.units):
        lo, hi = INPUT_BOX[unit_index]
        for segment in unit.segments:
            if _segment_overlaps(segment, lo, hi):
                slope_rows.append(
                    {
                        "unit_index": unit_index,
                        "segment_index": segment.index,
                        "x_min": segment.x_min,
                        "x_max": segment.x_max,
                        "slope": segment.slope,
                    }
                )
    min_slope = min(row["slope"] for row in slope_rows)
    verified = min_slope >= -1e-12
    return PropertyCertificate(
        property_id="bounded_monotonicity",
        method="pwa_slope_inspection",
        verified=verified,
        threshold=0.0,
        certified_upper_bound=None,
        bound_tightness=min_slope,
        witness_inputs=None,
        min_slope=min_slope,
        solver_status="not_needed_slope_inspection",
        details={
            "input_box": [list(bounds) for bounds in INPUT_BOX],
            "overlapping_slopes": slope_rows,
            "statement": "All active PWA slopes over the tiny box are nonnegative.",
        },
    )


def _z3_float(value: Any) -> float:
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        return float(numerator) / float(denominator)
    if text.endswith("?"):  # pragma: no cover - Z3 decimal approximation fallback.
        text = text[:-1]
    return float(text)


def _real(z3: Any, value: float) -> Any:
    return z3.RealVal(repr(float(value)))


def _blocked_unsafe_certificate() -> PropertyCertificate:
    return PropertyCertificate(
        property_id="no_unsafe_decision",
        method="z3_mixed_integer_pwa_box_bound",
        verified=False,
        threshold=UNSAFE_DECISION_THRESHOLD,
        certified_upper_bound=None,
        bound_tightness=None,
        witness_inputs=None,
        min_slope=None,
        solver_status="blocked_solver_dependency",
        details={
            "blocked_reason": "blocked_kan_pwa_milp_solver_unavailable",
            "missing_prerequisite": "python package 'z3'",
        },
    )


def no_unsafe_decision_certificate(
    abstraction: MultiUnitPWAAbstraction,
    *,
    solver_name: str | None = None,
) -> PropertyCertificate:
    """Maximize PWA energy on the input box and certify it stays below threshold."""

    selected_solver = detect_solver() if solver_name is None else solver_name
    if selected_solver != "z3":
        return _blocked_unsafe_certificate()

    z3 = importlib.import_module("z3")
    optimizer = z3.Optimize()
    xs = [z3.Real(f"x_{index}") for index in range(abstraction.input_dimension)]
    ys = [z3.Real(f"unit_energy_{index}") for index in range(abstraction.input_dimension)]
    total_energy = z3.Real("total_energy")
    selected_flag_groups: list[list[Any]] = []
    constraint_count = 0
    big_m = _real(z3, 10.0)

    def add_constraints(*constraints: Any) -> None:
        nonlocal constraint_count
        optimizer.add(*constraints)
        constraint_count += len(constraints)

    for unit_index, unit in enumerate(abstraction.units):
        x = xs[unit_index]
        y = ys[unit_index]
        lo, hi = INPUT_BOX[unit_index]
        flags = [
            z3.Int(f"exp5230_unit_{unit_index}_segment_{segment.index}")
            for segment in unit.segments
        ]
        selected_flag_groups.append(flags)

        add_constraints(x >= _real(z3, lo), x <= _real(z3, hi), z3.Sum(flags) == 1)
        for flag, segment in zip(flags, unit.segments):
            flag_real = z3.ToReal(flag)
            slack = big_m * (_real(z3, 1.0) - flag_real)
            affine_value = _real(z3, segment.slope) * x + _real(z3, segment.intercept)
            add_constraints(
                flag >= 0,
                flag <= 1,
                x >= _real(z3, segment.x_min) - slack,
                x <= _real(z3, segment.x_max) + slack,
                y - affine_value <= slack,
                affine_value - y <= slack,
            )

    add_constraints(total_energy == z3.Sum(ys))
    objective = optimizer.maximize(total_energy)
    status = optimizer.check()
    if status != z3.sat:  # pragma: no cover - retained for honest solver-status reporting.
        return PropertyCertificate(
            property_id="no_unsafe_decision",
            method="z3_mixed_integer_pwa_box_bound",
            verified=False,
            threshold=UNSAFE_DECISION_THRESHOLD,
            certified_upper_bound=None,
            bound_tightness=None,
            witness_inputs=None,
            min_slope=None,
            solver_status=str(status),
            details={"constraint_count": constraint_count, "solver_status": str(status)},
        )

    model = optimizer.model()
    certified_upper = _z3_float(objective.value())
    witness_inputs = tuple(_z3_float(model.eval(x, model_completion=True)) for x in xs)
    selected_segments = tuple(
        next(
            segment_index
            for segment_index, flag in enumerate(flags)
            if _z3_float(model.eval(flag, model_completion=True)) > 0.5
        )
        for flags in selected_flag_groups
    )
    bound_tightness = UNSAFE_DECISION_THRESHOLD - certified_upper
    verified = bound_tightness >= -1e-12
    return PropertyCertificate(
        property_id="no_unsafe_decision",
        method="z3_mixed_integer_pwa_box_bound",
        verified=verified,
        threshold=UNSAFE_DECISION_THRESHOLD,
        certified_upper_bound=certified_upper,
        bound_tightness=bound_tightness,
        witness_inputs=witness_inputs,
        min_slope=None,
        solver_status="optimal",
        details={
            "input_box": [list(bounds) for bounds in INPUT_BOX],
            "selected_segments": list(selected_segments),
            "constraint_count": constraint_count,
            "statement": (
                "For all x in the tiny input box, additive KAEM energy remains "
                f"below unsafe threshold {UNSAFE_DECISION_THRESHOLD}."
            ),
        },
    )


def run_certificate_checks(solver_name: str | None = None) -> CertificateResult:
    """Run the bounded PWA/MILP certificate checks for Exp 5230."""

    abstraction = build_certificate_abstraction(build_tiny_certificate_layer())
    monotone = monotonicity_certificate(abstraction)
    unsafe = no_unsafe_decision_certificate(abstraction, solver_name=solver_name)
    produced = monotone.verified and unsafe.verified
    solver_available = unsafe.solver_status != "blocked_solver_dependency"
    blocked_reason = None
    if not produced:
        blocked_reason = (
            "blocked_kan_pwa_milp_solver_unavailable"
            if not solver_available
            else "certificate_property_not_verified"
        )
    return CertificateResult(
        produced=produced,
        solver_available=solver_available,
        solver_status=unsafe.solver_status,
        blocked_reason=blocked_reason,
        bound_tightness=unsafe.bound_tightness if produced else None,
        abstraction=abstraction,
        property_results=(monotone, unsafe),
    )


def _honest_verdict(produced: bool, blocked_reason: str | None) -> str:
    if produced:
        return "success: tiny KAEM PWA/MILP certificate produced for bounded monotonicity and no unsafe decision"
    return f"complete: no KAEM PWA/MILP certificate produced; {blocked_reason}"


def _checksum_payload(result: CertificateResult) -> str:
    payload = {
        "input_box": INPUT_BOX,
        "produced": result.produced,
        "property_results": [row.as_serializable() for row in result.property_results],
        "random_seed": RANDOM_SEED,
        "spec_refs": SPEC_REFS,
        "target_module": TARGET_MODULE,
        "threshold": UNSAFE_DECISION_THRESHOLD,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    solver_name: str | None = None,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the validated Exp 5230 certificate artifact."""

    start = time.perf_counter()
    result = run_certificate_checks(solver_name=solver_name)
    measured_duration = round(time.perf_counter() - start, 6) if duration_s is None else duration_s
    property_rows = [row.as_serializable() for row in result.property_results]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "duration_s": measured_duration,
        "kan_certificate_produced": wrap_field("kan_certificate_produced", result.produced),
        "certificate_path": wrap_field(
            "certificate_path",
            str(RESULT_RELATIVE_PATH) if result.produced else None,
        ),
        "target_module": wrap_field("target_module", TARGET_MODULE),
        "properties_checked": wrap_field("properties_checked", property_rows),
        "bound_tightness": wrap_field("bound_tightness", result.bound_tightness),
        "reused_existing_helpers": wrap_field("reused_existing_helpers", True),
        "tests_run": wrap_field("tests_run", list(tests_run or [])),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": wrap_field(
            "honest_verdict",
            _honest_verdict(result.produced, result.blocked_reason),
        ),
        "blocked_reason": result.blocked_reason,
        "solver_available": result.solver_available,
        "solver_status": result.solver_status,
        "certificate": {
            "target_module": TARGET_MODULE,
            "input_box": [list(bounds) for bounds in INPUT_BOX],
            "pwa_abstraction": result.abstraction.as_serializable(),
            "property_results": property_rows,
            "limits": [
                "tiny deterministic KAEM fixture only",
                "no trained-network soundness claim",
                "no broad KAN verification claim",
                "no GRS-KAN implementation claim",
                "no hardware or live LLM inference claim",
            ],
        },
        "field_principles": FIELD_PRINCIPLES,
        "methodology_note": (
            "This is a small deterministic certificate for one KAEM PWA fixture, "
            "not broad KAN verification. It reuses existing Carnot PWA helpers and "
            "keeps the claim to bounded monotonicity plus a no-unsafe-decision box bound."
        ),
        "random_seed": RANDOM_SEED,
        "source_helpers": [
            "carnot.experiment_5080_kan_pwa_milp_bridge_v466.build_pwa_abstraction",
            "carnot.experiment_5091_kan_pwa_milp_scale_v467.MultiUnitPWAAbstraction",
        ],
        "spec_refs": list(SPEC_REFS),
    }
    artifact["reproducibility_checksum"] = _checksum_payload(result)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 5230 artifact and fail closed on schema drift."""

    for field in REQUIRED_WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
        _require(wrapped.get("principle") == FIELD_PRINCIPLES[field], f"{field} principle drift")
        _require("value" in wrapped, f"{field} missing value")

    produced = artifact["kan_certificate_produced"]["value"]
    certificate_path = artifact["certificate_path"]["value"]
    target_module = artifact["target_module"]["value"]
    properties_checked = artifact["properties_checked"]["value"]
    bound_tightness = artifact["bound_tightness"]["value"]
    reused_existing_helpers = artifact["reused_existing_helpers"]["value"]
    substrate = artifact["inference_substrate"]["value"]
    verdict = artifact["honest_verdict"]["value"]

    _require(substrate == INFERENCE_SUBSTRATE, "inference_substrate must be deterministic_pwa_milp_certificate")
    _require(target_module == TARGET_MODULE, "target_module must be UnivariateKAEMLayer")
    _require(reused_existing_helpers is True, "reused_existing_helpers must be true")
    _require(isinstance(properties_checked, list), "properties_checked must be a list")
    _require(
        {"bounded_monotonicity", "no_unsafe_decision"}
        == {str(row["property_id"]) for row in properties_checked},
        "properties_checked must contain the two tiny certificate properties",
    )
    _require(isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), "honest_verdict prefix")
    _require("live LLM" in artifact["methodology_note"] or "live_llm" not in substrate, "methodology note/substrate mismatch")
    _require("broad KAN verification" in artifact["methodology_note"], "must state broad KAN limit")

    if produced:
        _require(certificate_path == str(RESULT_RELATIVE_PATH), "certificate_path must point to deliverable")
        _require(bound_tightness is not None and bound_tightness > 0.0, "bound_tightness must be positive")
        _require(artifact["blocked_reason"] is None, "produced certificate cannot have blocked_reason")
        _require(artifact["solver_status"] == "optimal", "produced certificate requires optimal solver")
        _require(all(row["verified"] is True for row in properties_checked), "all properties must verify")
    else:
        _require(certificate_path is None, "blocked artifact cannot claim certificate_path")
        _require(bound_tightness is None, "blocked artifact cannot claim bound_tightness")
        _require(artifact["blocked_reason"] is not None, "blocked artifact must name prerequisite")
        _require("no KAEM PWA/MILP certificate produced" in verdict, "blocked verdict must say no certificate")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    solver_name: str | None = None,
) -> JsonDict:
    """Write the Exp 5230 certificate artifact and return the payload."""

    artifact = build_artifact(
        solver_name=solver_name,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
    )
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - exercised through unit-level writer tests.
    artifact = write_outputs()
    print(artifact["honest_verdict"]["value"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
