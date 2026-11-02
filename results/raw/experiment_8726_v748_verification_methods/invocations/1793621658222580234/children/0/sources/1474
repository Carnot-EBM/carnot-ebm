"""Exp 5102: direct HUBO/p-spin energy versus QUBO gadgets.

Spec refs: REQ-VERIFY-5102, SCENARIO-VERIFY-5102.

The experiment keeps the correctness authority deliberately simple: every tiny
high-order parity CSP is solved by exact CPU enumeration. Direct HUBO uses one
p-spin parity term per high-order clause. The QUBO arm quadratizes the same
binary parity polynomial with auxiliary AND gadgets, then enumerates all
original and auxiliary variables to prove that the gadget optimum projects
back to the same original assignments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
Polynomial = dict[tuple[str, ...], Fraction]

RESULT_RELATIVE_PATH = "results/experiment_5102_hubo_pspin_direct_energy_v468.json"
RUN_DATE = "20260701"
RANDOM_SEED = 5102
INSTANCE_FAMILY = "tiny_high_order_xorsat_parity_v1"
INFERENCE_SUBSTRATE = "exact_enumeration_cpu"
SUCCESS_VERDICT = "success_hubo_pspin_direct_encoding_reduces_gadget_blowup"
NO_ADVANTAGE_VERDICT = "complete_hubo_pspin_no_advantage_over_qubo_gadgets"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "instance_family",
    "exact_optima_verified",
    "hubo_variable_counts",
    "qubo_variable_counts",
    "auxiliary_variable_blowup",
    "coupling_density_hubo",
    "coupling_density_qubo",
    "energy_scale_distortion",
    "direct_hubo_advantage",
    "hardware_mapping_notes",
    "flagged_adversarial",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal prefix; success only when exact optima match and direct HUBO uses fewer "
        "variables than QUBO gadgets."
    ),
    "duration_s": "Measured local CPU time for deterministic encoding, exact enumeration, and JSON assembly.",
    "inference_substrate": "Must be exact_enumeration_cpu unless another exact substrate is proven.",
    "instance_family": "Declared deterministic tiny high-order parity CSP family.",
    "exact_optima_verified": "True only when direct and QUBO optima and projected optimum assignments match for every instance.",
    "hubo_variable_counts": "Original variable counts for direct high-order HUBO/p-spin encodings.",
    "qubo_variable_counts": "Original plus auxiliary variable counts for quadratized pairwise QUBO gadgets.",
    "auxiliary_variable_blowup": "Auxiliary variables introduced only by quadratization, reported per instance and in aggregate.",
    "coupling_density_hubo": "Direct high-order hyperedge density over the clause degrees used by each instance.",
    "coupling_density_qubo": "Pairwise QUBO coupler density over all original plus auxiliary variables.",
    "energy_scale_distortion": "Coefficient-scale ratio from QUBO penalties versus native direct p-spin coefficients.",
    "direct_hubo_advantage": "True only when exactness holds and direct HUBO avoids the QUBO auxiliary-variable blowup.",
    "hardware_mapping_notes": "Maps the result to native high-order/p-spin hardware versus pairwise QUBO hardware costs.",
    "flagged_adversarial": "False only when substrate, exactness, projection, and metric consistency checks pass.",
}


@dataclass(frozen=True)
class ParityClause:
    """A k-XORSAT clause over zero-based binary variable indices."""

    variables: tuple[int, ...]
    parity: int


@dataclass(frozen=True)
class HighOrderCspInstance:
    """One tiny high-order Boolean CSP instance with parity constraints."""

    instance_id: str
    family: str
    n_vars: int
    clauses: tuple[ParityClause, ...]
    description: str


@dataclass(frozen=True)
class HuboEncoding:
    """Direct p-spin parity energy over original variables only."""

    instance: HighOrderCspInstance
    constant: Fraction
    terms: dict[tuple[int, ...], Fraction]


@dataclass(frozen=True)
class QuboEncoding:
    """Pairwise binary QUBO after quadratizing high-order parity terms."""

    instance: HighOrderCspInstance
    polynomial: Polynomial
    original_variables: tuple[str, ...]
    all_variables: tuple[str, ...]
    auxiliary_definitions: dict[str, tuple[str, str]]
    penalty_strength: Fraction
    source_binary_polynomial: Polynomial


@dataclass(frozen=True)
class EnumerationResult:
    """Exact optimum energy and all original assignments reaching it."""

    optimum_energy: Fraction
    optimal_assignments: tuple[tuple[int, ...], ...]
    duration_s: float


@dataclass(frozen=True)
class QuboEnumerationResult:
    """Exact QUBO optimum and all projected original assignments reaching it."""

    optimum_energy: Fraction
    projected_assignments: tuple[tuple[int, ...], ...]
    full_assignments: tuple[dict[str, int], ...]
    duration_s: float


def build_instance_family() -> tuple[HighOrderCspInstance, ...]:
    """Return a deterministic tiny family with satisfiable and frustrated parity CSPs."""

    return (
        HighOrderCspInstance(
            instance_id="xor3_chain_sat",
            family=INSTANCE_FAMILY,
            n_vars=4,
            clauses=(
                ParityClause((0, 1, 2), 1),
                ParityClause((1, 2, 3), 0),
                ParityClause((0, 2, 3), 1),
            ),
            description="Three overlapping 3-XOR clauses with nonempty exact optimum set.",
        ),
        HighOrderCspInstance(
            instance_id="xor3_xor4_frustrated",
            family=INSTANCE_FAMILY,
            n_vars=4,
            clauses=(
                ParityClause((0, 1, 2), 0),
                ParityClause((0, 1, 3), 0),
                ParityClause((0, 2, 3), 0),
                ParityClause((1, 2, 3), 0),
                ParityClause((0, 1, 2, 3), 1),
            ),
            description="Four 3-XOR equations force even 4-parity, while a 4-XOR clause asks for odd parity.",
        ),
        HighOrderCspInstance(
            instance_id="xor4_planted_overlap",
            family=INSTANCE_FAMILY,
            n_vars=5,
            clauses=(
                ParityClause((0, 1, 2, 3), 1),
                ParityClause((0, 1, 3, 4), 0),
                ParityClause((1, 2, 3, 4), 1),
            ),
            description="Overlapping 4-XOR clauses with a planted-compatible optimum.",
        ),
    )


def build_hubo_encoding(instance: HighOrderCspInstance) -> HuboEncoding:
    """Encode parity clauses as direct p-spin terms over bipolar spins."""

    constant = Fraction(0, 1)
    terms: dict[tuple[int, ...], Fraction] = {}
    for clause in instance.clauses:
        _validate_clause(instance, clause)
        target_spin_product = Fraction(1 if clause.parity == 0 else -1, 1)
        constant += Fraction(1, 2)
        term = tuple(sorted(clause.variables))
        terms[term] = terms.get(term, Fraction(0, 1)) - target_spin_product * Fraction(1, 2)
        if terms[term] == 0:
            del terms[term]
    return HuboEncoding(instance=instance, constant=constant, terms=terms)


def build_qubo_gadget_encoding(instance: HighOrderCspInstance) -> QuboEncoding:
    """Expand parity energy to binary HUBO and reduce high-degree terms with AND gadgets."""

    source = _binary_parity_polynomial(instance)
    penalty_strength = sum(abs(coefficient) for coefficient in source.values()) + 1
    polynomial: Polynomial = {}
    auxiliary_by_pair: dict[tuple[str, str], str] = {}
    auxiliary_definitions: dict[str, tuple[str, str]] = {}

    def add_term(term: Sequence[str], coefficient: Fraction) -> None:
        canonical = tuple(sorted(term))
        polynomial[canonical] = polynomial.get(canonical, Fraction(0, 1)) + coefficient
        if polynomial[canonical] == 0:
            del polynomial[canonical]

    def ensure_auxiliary(left: str, right: str) -> str:
        pair = tuple(sorted((left, right)))
        if pair in auxiliary_by_pair:
            return auxiliary_by_pair[pair]
        auxiliary = f"aux_{len(auxiliary_by_pair)}"
        auxiliary_by_pair[pair] = auxiliary
        auxiliary_definitions[auxiliary] = pair
        add_term(pair, penalty_strength)
        add_term((left, auxiliary), -2 * penalty_strength)
        add_term((right, auxiliary), -2 * penalty_strength)
        add_term((auxiliary,), 3 * penalty_strength)
        return auxiliary

    for term, coefficient in sorted(source.items(), key=lambda item: (len(item[0]), item[0])):
        if len(term) <= 2:
            add_term(term, coefficient)
            continue
        reduced = list(term)
        while len(reduced) > 2:
            auxiliary = ensure_auxiliary(reduced[0], reduced[1])
            reduced = [auxiliary, *reduced[2:]]
        add_term(reduced, coefficient)

    original_variables = tuple(f"x_{index}" for index in range(instance.n_vars))
    all_variables = original_variables + tuple(sorted(auxiliary_definitions))
    return QuboEncoding(
        instance=instance,
        polynomial=polynomial,
        original_variables=original_variables,
        all_variables=all_variables,
        auxiliary_definitions=auxiliary_definitions,
        penalty_strength=penalty_strength,
        source_binary_polynomial=source,
    )


def enumerate_hubo(encoding: HuboEncoding) -> EnumerationResult:
    """Enumerate all original assignments for the direct HUBO energy."""

    started = time.perf_counter()
    best_energy: Fraction | None = None
    best_assignments: list[tuple[int, ...]] = []
    for assignment in itertools.product((0, 1), repeat=encoding.instance.n_vars):
        energy = evaluate_hubo(encoding, assignment)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_assignments = [tuple(assignment)]
        elif energy == best_energy:
            best_assignments.append(tuple(assignment))
    return EnumerationResult(
        optimum_energy=best_energy if best_energy is not None else Fraction(0, 1),
        optimal_assignments=tuple(sorted(best_assignments)),
        duration_s=round(time.perf_counter() - started, 6),
    )


def enumerate_qubo(encoding: QuboEncoding) -> QuboEnumerationResult:
    """Enumerate all original and auxiliary assignments for the QUBO energy."""

    started = time.perf_counter()
    best_energy: Fraction | None = None
    full_assignments: list[dict[str, int]] = []
    projected_assignments: set[tuple[int, ...]] = set()
    for bits in itertools.product((0, 1), repeat=len(encoding.all_variables)):
        assignment = dict(zip(encoding.all_variables, bits, strict=True))
        energy = evaluate_qubo(encoding.polynomial, assignment)
        projected = tuple(assignment[var] for var in encoding.original_variables)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            full_assignments = [assignment]
            projected_assignments = {projected}
        elif energy == best_energy:
            full_assignments.append(assignment)
            projected_assignments.add(projected)
    return QuboEnumerationResult(
        optimum_energy=best_energy if best_energy is not None else Fraction(0, 1),
        projected_assignments=tuple(sorted(projected_assignments)),
        full_assignments=tuple(full_assignments),
        duration_s=round(time.perf_counter() - started, 6),
    )


def energy_by_projection(encoding: HuboEncoding) -> dict[tuple[int, ...], Fraction]:
    """Return direct HUBO energy for every original assignment."""

    return {
        tuple(assignment): evaluate_hubo(encoding, assignment)
        for assignment in itertools.product((0, 1), repeat=encoding.instance.n_vars)
    }


def best_qubo_energy_by_projection(encoding: QuboEncoding) -> dict[tuple[int, ...], Fraction]:
    """Return the best QUBO energy over auxiliaries for every original assignment."""

    best: dict[tuple[int, ...], Fraction] = {}
    for bits in itertools.product((0, 1), repeat=len(encoding.all_variables)):
        assignment = dict(zip(encoding.all_variables, bits, strict=True))
        projected = tuple(assignment[var] for var in encoding.original_variables)
        energy = evaluate_qubo(encoding.polynomial, assignment)
        if projected not in best or energy < best[projected]:
            best[projected] = energy
    return dict(sorted(best.items()))


def evaluate_hubo(encoding: HuboEncoding, assignment: Sequence[int]) -> Fraction:
    """Evaluate direct p-spin energy for one binary assignment."""

    if len(assignment) != encoding.instance.n_vars:
        raise ValueError("assignment length does not match instance variables")
    energy = encoding.constant
    for variables, coefficient in encoding.terms.items():
        product = 1
        for variable in variables:
            product *= 1 if assignment[variable] == 0 else -1
        energy += coefficient * product
    return energy


def evaluate_qubo(polynomial: Mapping[tuple[str, ...], Fraction], assignment: Mapping[str, int]) -> Fraction:
    """Evaluate a binary QUBO polynomial on a complete variable assignment."""

    total = Fraction(0, 1)
    for term, coefficient in polynomial.items():
        product = 1
        for variable in term:
            product *= int(assignment[variable])
        total += coefficient * product
    return total


def compare_instance(instance: HighOrderCspInstance) -> JsonDict:
    """Build both encodings, enumerate them exactly, and return one metric row."""

    hubo = build_hubo_encoding(instance)
    qubo = build_qubo_gadget_encoding(instance)
    hubo_result = enumerate_hubo(hubo)
    qubo_result = enumerate_qubo(qubo)
    projection_match = energy_by_projection(hubo) == best_qubo_energy_by_projection(qubo)
    assignments_match = hubo_result.optimal_assignments == qubo_result.projected_assignments
    optima_match = hubo_result.optimum_energy == qubo_result.optimum_energy
    hubo_coupling_count = _hubo_coupling_count(hubo)
    qubo_coupling_count = _qubo_coupling_count(qubo)
    hubo_density = _hubo_density(hubo)
    qubo_density = _qubo_density(qubo)
    distortion = _energy_scale_distortion(hubo, qubo)

    return {
        "instance_id": instance.instance_id,
        "description": instance.description,
        "n_clauses": len(instance.clauses),
        "clause_degrees": [len(clause.variables) for clause in instance.clauses],
        "direct_optimum_energy": _json_number(hubo_result.optimum_energy),
        "qubo_optimum_energy": _json_number(qubo_result.optimum_energy),
        "direct_optimal_assignments": _json_assignments(hubo_result.optimal_assignments),
        "projected_qubo_optimal_assignments": _json_assignments(qubo_result.projected_assignments),
        "exact_optima_verified": bool(optima_match and assignments_match and projection_match),
        "projection_energy_match_for_all_assignments": projection_match,
        "hubo_variable_count": instance.n_vars,
        "qubo_variable_count": len(qubo.all_variables),
        "auxiliary_variable_count": len(qubo.auxiliary_definitions),
        "auxiliary_definitions": {
            key: list(value) for key, value in sorted(qubo.auxiliary_definitions.items())
        },
        "hubo_coupling_count": hubo_coupling_count,
        "qubo_coupling_count": qubo_coupling_count,
        "coupling_density_hubo": hubo_density,
        "coupling_density_qubo": qubo_density,
        "qubo_penalty_strength": _json_number(qubo.penalty_strength),
        "energy_scale_distortion": distortion,
        "enumeration_time_s": {
            "hubo": hubo_result.duration_s,
            "qubo": qubo_result.duration_s,
        },
    }


def run(duration_s: float | None = None) -> JsonDict:
    """Run the full Exp 5102 comparison and return the terminal artifact."""

    started = time.perf_counter()
    family = build_instance_family()
    rows = [compare_instance(instance) for instance in family]
    exact_optima_verified = all(row["exact_optima_verified"] for row in rows)
    direct_hubo_advantage = bool(
        exact_optima_verified
        and sum(row["hubo_variable_count"] for row in rows)
        < sum(row["qubo_variable_count"] for row in rows)
        and sum(row["auxiliary_variable_count"] for row in rows) > 0
    )
    flagged_adversarial = not exact_optima_verified or INFERENCE_SUBSTRATE != "exact_enumeration_cpu"
    honest_verdict = SUCCESS_VERDICT if direct_hubo_advantage and not flagged_adversarial else NO_ADVANTAGE_VERDICT
    elapsed = round(time.perf_counter() - started, 6) if duration_s is None else duration_s
    artifact: JsonDict = {
        "schema": "carnot.experiment_5102_hubo_pspin_direct_energy.v468",
        "experiment_id": 5102,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "duration_s": elapsed,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "instance_family": INSTANCE_FAMILY,
        "exact_optima_verified": exact_optima_verified,
        "hubo_variable_counts": _variable_counts(rows, "hubo_variable_count"),
        "qubo_variable_counts": _variable_counts(rows, "qubo_variable_count"),
        "auxiliary_variable_blowup": _auxiliary_blowup(rows),
        "coupling_density_hubo": _density_summary(rows, "coupling_density_hubo", "hubo_coupling_count"),
        "coupling_density_qubo": _density_summary(rows, "coupling_density_qubo", "qubo_coupling_count"),
        "energy_scale_distortion": _distortion_summary(rows),
        "direct_hubo_advantage": direct_hubo_advantage,
        "hardware_mapping_notes": (
            "Direct HUBO keeps native high-order p-spin parity terms over original variables. "
            "QUBO maps to pairwise hardware but pays auxiliary-variable, coupler-density, and "
            "large penalty-coefficient costs."
        ),
        "flagged_adversarial": flagged_adversarial,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-5102", "SCENARIO-VERIFY-5102"],
        "instance_results": rows,
        "exact_authority": "exhaustive_binary_enumeration",
        "methodology_note": (
            "All optima are proven by enumerating every binary assignment; no stochastic sampler, "
            "LLM judge, or hardware backend is used."
        ),
    }
    artifact["reproducibility_checksum"] = _sha256_json(
        {
            "instance_family": artifact["instance_family"],
            "instance_results": artifact["instance_results"],
            "exact_optima_verified": artifact["exact_optima_verified"],
        }
    )
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5102 artifact violates the terminal contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required fields: {missing}")
    verdict = str(artifact["honest_verdict"])
    _require(
        verdict.startswith(SUCCESS_VERDICT) or verdict.startswith(NO_ADVANTAGE_VERDICT),
        "honest_verdict must use the Exp5102 terminal prefix",
    )
    _require(
        isinstance(artifact["duration_s"], (int, float)) and artifact["duration_s"] >= 0.0,
        "duration_s must be nonnegative",
    )
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate")
    _require("llm" not in str(artifact["inference_substrate"]).lower(), "inference_substrate")
    _require(artifact["instance_family"] == INSTANCE_FAMILY, "instance_family")
    _require(artifact["exact_optima_verified"] is True, "exact_optima_verified")
    _require(artifact["direct_hubo_advantage"] is True, "direct_hubo_advantage")
    _require(artifact["flagged_adversarial"] is False, "flagged_adversarial")
    for field in ("hubo_variable_counts", "qubo_variable_counts"):
        _require(_valid_count_summary(artifact[field]), field)
    blowup = artifact["auxiliary_variable_blowup"]
    _require(
        isinstance(blowup, Mapping)
        and blowup.get("total_auxiliary_variables", 0) > 0
        and blowup.get("mean_auxiliary_variables", 0) > 0,
        "auxiliary_variable_blowup",
    )
    _require(_valid_density_summary(artifact["coupling_density_hubo"]), "coupling_density_hubo")
    _require(_valid_density_summary(artifact["coupling_density_qubo"]), "coupling_density_qubo")
    distortion = artifact["energy_scale_distortion"]
    _require(
        isinstance(distortion, Mapping)
        and distortion.get("max_qubo_to_hubo_coefficient_ratio", 0) > 1.0,
        "energy_scale_distortion",
    )
    notes = str(artifact["hardware_mapping_notes"])
    _require("native high-order" in notes and "QUBO" in notes, "hardware_mapping_notes")
    principles = artifact.get("field_principles")
    _require(
        isinstance(principles, Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS).issubset(principles),
        "field_principles",
    )
    rows = artifact.get("instance_results", [])
    _require(isinstance(rows, list) and bool(rows), "instance_results")
    _require(all(row.get("exact_optima_verified") is True for row in rows), "instance_results")


def write_artifact(root: str | Path | None = None, output_path: str | Path | None = None) -> JsonDict:
    """Run the experiment and write the Exp 5102 terminal JSON artifact."""

    repo_root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    destination = Path(output_path) if output_path is not None else repo_root / RESULT_RELATIVE_PATH
    artifact = run()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:
    """CLI entrypoint used by the conductor and tests."""

    root = Path(os.environ.get("CARNOT_EXP5102_ROOT", Path(__file__).resolve().parents[2]))
    write_artifact(root=root)
    return 0


def _validate_clause(instance: HighOrderCspInstance, clause: ParityClause) -> None:
    _require(clause.parity in (0, 1), "parity must be 0 or 1")
    _require(len(clause.variables) >= 3, "Exp 5102 clauses must be high-order")
    _require(len(set(clause.variables)) == len(clause.variables), "clause variables must be unique")
    _require(all(0 <= variable < instance.n_vars for variable in clause.variables), "clause variable out of range")


def _binary_parity_polynomial(instance: HighOrderCspInstance) -> Polynomial:
    polynomial: Polynomial = {}
    for clause in instance.clauses:
        _validate_clause(instance, clause)
        variables = tuple(f"x_{index}" for index in clause.variables)
        target_spin_product = Fraction(1 if clause.parity == 0 else -1, 1)
        for degree in range(len(variables) + 1):
            for subset in itertools.combinations(variables, degree):
                coefficient = -target_spin_product * Fraction((-2) ** degree, 2)
                if degree == 0:
                    coefficient += Fraction(1, 2)
                polynomial[tuple(sorted(subset))] = polynomial.get(tuple(sorted(subset)), Fraction(0, 1)) + coefficient
                if polynomial[tuple(sorted(subset))] == 0:
                    del polynomial[tuple(sorted(subset))]
    return dict(sorted(polynomial.items(), key=lambda item: (len(item[0]), item[0])))


def _hubo_coupling_count(encoding: HuboEncoding) -> int:
    return sum(1 for term in encoding.terms if len(term) >= 3)


def _qubo_coupling_count(encoding: QuboEncoding) -> int:
    return sum(1 for term in encoding.polynomial if len(term) == 2)


def _hubo_density(encoding: HuboEncoding) -> float:
    degrees = sorted({len(term) for term in encoding.terms if len(term) >= 3})
    possible = sum(math.comb(encoding.instance.n_vars, degree) for degree in degrees)
    return round(_hubo_coupling_count(encoding) / possible, 6) if possible else 0.0


def _qubo_density(encoding: QuboEncoding) -> float:
    possible = math.comb(len(encoding.all_variables), 2)
    return round(_qubo_coupling_count(encoding) / possible, 6) if possible else 0.0


def _energy_scale_distortion(hubo: HuboEncoding, qubo: QuboEncoding) -> float:
    hubo_max = max(abs(coefficient) for coefficient in hubo.terms.values())
    qubo_max = max(abs(coefficient) for term, coefficient in qubo.polynomial.items() if term)
    return round(float(qubo_max / hubo_max), 6)


def _variable_counts(rows: Sequence[Mapping[str, Any]], field: str) -> JsonDict:
    by_instance = {str(row["instance_id"]): int(row[field]) for row in rows}
    values = list(by_instance.values())
    return {
        "by_instance": by_instance,
        "total": sum(values),
        "mean": round(sum(values) / len(values), 6),
        "max": max(values),
    }


def _auxiliary_blowup(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_instance = {
        str(row["instance_id"]): int(row["auxiliary_variable_count"]) for row in rows
    }
    ratios = {
        str(row["instance_id"]): round(row["qubo_variable_count"] / row["hubo_variable_count"], 6)
        for row in rows
    }
    values = list(by_instance.values())
    return {
        "auxiliary_variables_by_instance": by_instance,
        "qubo_to_hubo_variable_ratio_by_instance": ratios,
        "total_auxiliary_variables": sum(values),
        "mean_auxiliary_variables": round(sum(values) / len(values), 6),
        "mean_qubo_to_hubo_variable_ratio": round(
            sum(ratios.values()) / len(ratios),
            6,
        ),
    }


def _density_summary(rows: Sequence[Mapping[str, Any]], density_field: str, count_field: str) -> JsonDict:
    by_instance = {str(row["instance_id"]): float(row[density_field]) for row in rows}
    counts = {str(row["instance_id"]): int(row[count_field]) for row in rows}
    values = list(by_instance.values())
    return {
        "by_instance": by_instance,
        "coupling_counts_by_instance": counts,
        "mean": round(sum(values) / len(values), 6),
        "max": round(max(values), 6),
    }


def _distortion_summary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_instance = {str(row["instance_id"]): float(row["energy_scale_distortion"]) for row in rows}
    values = list(by_instance.values())
    return {
        "by_instance": by_instance,
        "mean_qubo_to_hubo_coefficient_ratio": round(sum(values) / len(values), 6),
        "max_qubo_to_hubo_coefficient_ratio": round(max(values), 6),
        "definition": "max_abs_qubo_coefficient_after_gadgets / max_abs_direct_hubo_p_spin_coefficient",
    }


def _valid_count_summary(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("by_instance"), Mapping)
        and value.get("total", 0) > 0
        and value.get("mean", 0) > 0
    )


def _valid_density_summary(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("by_instance"), Mapping)
        and isinstance(value.get("coupling_counts_by_instance"), Mapping)
        and 0.0 <= float(value.get("mean", -1.0)) <= 1.0
        and 0.0 <= float(value.get("max", -1.0)) <= 1.0
    )


def _json_assignments(assignments: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(assignment) for assignment in assignments]


def _json_number(value: Fraction) -> int | float:
    if value.denominator == 1:
        return value.numerator
    return round(float(value), 6)


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
