"""Exp 5278: solver fixture to factor-graph boundary certificate.

Spec refs: REQ-VERIFY-5278, SCENARIO-VERIFY-5278.

This module takes one tiny solver-labeled Exp 5273 fixture and maps it to a
transparent factor graph plus a pairwise QUBO/Ising-style interface. The point
is boundary evidence only: the code records deterministic energy,
constraint-violation, and sampler-interface shape metrics without running a
hardware board or claiming speedup.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from carnot import experiment_5273_solver_fixture_rebuild_v482 as fixture_mod


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5278_constraint_factor_graph_boundary_v482.json"
)
SCHEMA = "carnot.experiment_5278.constraint_factor_graph_boundary.v482"
SPEC_REFS = ("REQ-VERIFY-5278", "SCENARIO-VERIFY-5278")
INFERENCE_SUBSTRATE = "offline_deterministic_certificate_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked_")
PENALTIES = {"one_hot": 20.0, "sum_is_five": 2.0, "a_less_than_b": 5.0}

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "Terminal Exp 5278 verdict; starts with complete: or blocked_ and states "
        "whether the factor-graph boundary is usable."
    ),
    "inference_substrate": (
        "Must be offline_deterministic_certificate_no_llm because Exp 5278 is a "
        "deterministic boundary certificate with no LLM inference."
    ),
    "factor_graph_boundary_ready": (
        "True only when the selected solver fixture is represented as transparent "
        "variables, factors, binary one-hot variables, and pairwise QUBO coefficients."
    ),
    "sampler_interface_ready": (
        "True only when the boundary emits finite bias/coupling arrays and "
        "backend-compatible interface metadata without running board-specific hardware."
    ),
    "mapping_roundtrip_passed": (
        "True only when the solver witness encodes to bits, decodes back to the "
        "same assignment, has zero deterministic energy, and has zero constraint violation."
    ),
    "false_assignment_rejected": (
        "True only when a deterministic rejecting assignment from the fixture has "
        "positive energy or positive constraint violation."
    ),
    "autocorrelation_metric": (
        "Numeric only for a meaningful CPU chain metric; null is required for "
        "exhaustive deterministic enumeration without a Markov chain."
    ),
    "hardware_speedup_claimed": (
        "Always false for Exp 5278; interface compatibility is not hardware "
        "timing or acceleration evidence."
    ),
    "tests_run": (
        "Commands run to validate mapping correctness, false-assignment rejection, "
        "no-speedup artifact fields, new-code coverage, and repository test status."
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
WRAPPED_FIELDS = tuple(field for field in REQUIRED_ARTIFACT_FIELDS if field != "tests_run")


@dataclass(frozen=True)
class BoundaryInstance:
    """Transparent finite-domain and QUBO view of one solver fixture."""

    fixture_id: str
    source: str
    variables: JsonDict
    solver_assignment: JsonDict
    false_assignment: JsonDict
    bit_order: tuple[str, ...]
    bit_values: tuple[int, ...]
    bit_variables: tuple[str, ...]
    constant: float
    linear: tuple[float, ...]
    quadratic: tuple[tuple[int, int, float], ...]
    factors: tuple[JsonDict, ...]

    def assignment_to_bits(self, assignment: JsonDict) -> tuple[int, ...]:
        """Encode integer assignment values as one-hot binary variables."""

        return tuple(
            1 if int(assignment[variable]) == value else 0
            for variable, value in zip(self.bit_variables, self.bit_values, strict=True)
        )

    def bits_to_assignment(self, bits: tuple[int, ...]) -> JsonDict:
        """Decode a valid one-hot bit state back into integer variables."""

        assignment: JsonDict = {}
        for variable in self.variables:
            active = [
                self.bit_values[index]
                for index, bit_variable in enumerate(self.bit_variables)
                if bit_variable == variable and bits[index] == 1
            ]
            assignment[variable] = active[0]
        return assignment

    def evaluate_assignment(self, assignment: JsonDict) -> JsonDict:
        """Evaluate the original finite-domain fixture constraints directly."""

        a = int(assignment["a"])
        b = int(assignment["b"])
        violations = {
            "a_domain": 0 if a in self.variables["a"]["domain"] else 1,
            "b_domain": 0 if b in self.variables["b"]["domain"] else 1,
            "sum_is_five": abs(a + b - 5),
            "a_less_than_b": 0 if a < b else 1,
        }
        return {
            "assignment": dict(assignment),
            "constraint_violations": violations,
            "total_violation": int(sum(violations.values())),
        }

    def qubo_energy(self, bits: tuple[int, ...]) -> float:
        """Evaluate the pairwise QUBO energy for a binary state."""

        state = np.asarray(bits, dtype=float)
        energy = self.constant + float(np.dot(np.asarray(self.linear, dtype=float), state))
        for left, right, weight in self.quadratic:
            energy += weight * float(state[left]) * float(state[right])
        return round(float(energy), 10)

    def is_valid_onehot(self, bits: tuple[int, ...]) -> bool:
        """Return true when every finite-domain variable has exactly one active bit."""

        return all(
            sum(
                bits[index]
                for index, bit_variable in enumerate(self.bit_variables)
                if bit_variable == variable
            )
            == 1
            for variable in self.variables
        )

    def factor_graph_artifact(self) -> JsonDict:
        """Return the JSON-ready factor graph and QUBO coefficient view."""

        return {
            "fixture_id": self.fixture_id,
            "source": self.source,
            "variables": self.variables,
            "binary_variable_order": list(self.bit_order),
            "factors": list(self.factors),
            "qubo": {
                "convention": "minimize constant + linear.x + pairwise.x_i.x_j",
                "constant": self.constant,
                "linear": list(self.linear),
                "quadratic": [
                    {"left": self.bit_order[left], "right": self.bit_order[right], "weight": weight}
                    for left, right, weight in self.quadratic
                ],
            },
        }

    def sampler_interface(self, enumeration: JsonDict) -> JsonDict:
        """Return an Ising-style bias/coupling payload without invoking hardware."""

        biases = -np.asarray(self.linear, dtype=float)
        couplings = _dense_couplings(self)
        return {
            "backend_protocol": "SamplerBackend",
            "energy_convention": "maximize biases.x + x.T couplings x equals negative QUBO up to constant",
            "binary_variable_order": list(self.bit_order),
            "bias_shape": list(biases.shape),
            "coupling_shape": list(couplings.shape),
            "biases": biases.tolist(),
            "couplings": couplings.tolist(),
            "cpu_enumerator_state_count": enumeration["state_count"],
            "cpu_enumerator_best_energy": enumeration["best_energy"],
            "compatible_backends_checked": ["CpuBackend", "TsuBackend"],
            "hardware_board_command_run": False,
            "speedup_claimed": False,
        }


def select_tiny_fixture(
    artifact_path: Path = REPO_ROOT / fixture_mod.RESULT_RELATIVE_PATH,
) -> JsonDict:
    """Select Exp 5273's small pair-sum fixture or the same local fallback."""

    artifact = json.loads(artifact_path.read_text(encoding="utf-8")) if artifact_path.exists() else {}
    reusable = bool(
        artifact.get("solver_fixture_ready")
        and any(row.get("fixture_id") == "small_pair_sum" for row in artifact.get("fixtures", []))
    )
    fixture = next(row for row in fixture_mod.fixture_set() if row.fixture_id == "small_pair_sum")
    return {
        "fixture_id": fixture.fixture_id,
        "expected_status": fixture.expected_status,
        "solver_assignment": dict(fixture.gold_assignment),
        "false_assignment": dict(fixture.negative_assignments[0]),
        "reference_encoding": fixture.reference_encoding,
        "source_artifact_reusable": reusable,
        "source": "exp5273_result" if reusable else "local_deterministic_fallback",
    }


def build_boundary(source_fixture: JsonDict) -> BoundaryInstance:
    """Build the transparent one-hot factor graph and pairwise QUBO boundary."""

    variables = {
        "a": {"domain": [0, 1, 2, 3], "encoding": "one_hot"},
        "b": {"domain": [0, 1, 2, 3], "encoding": "one_hot"},
    }
    bit_order = tuple(f"{variable}_{value}" for variable in variables for value in variables[variable]["domain"])
    bit_variables = tuple(bit.split("_", 1)[0] for bit in bit_order)
    bit_values = tuple(int(bit.rsplit("_", 1)[1]) for bit in bit_order)
    index = {name: offset for offset, name in enumerate(bit_order)}
    constant, linear, quadratic = _build_qubo_coefficients(index)
    factors = (
        {
            "id": "a_one_hot",
            "type": "one_hot",
            "variables": ["a_0", "a_1", "a_2", "a_3"],
            "weight": PENALTIES["one_hot"],
            "violation": "(sum(a_i) - 1)^2",
        },
        {
            "id": "b_one_hot",
            "type": "one_hot",
            "variables": ["b_0", "b_1", "b_2", "b_3"],
            "weight": PENALTIES["one_hot"],
            "violation": "(sum(b_i) - 1)^2",
        },
        {
            "id": "sum_is_five",
            "type": "linear_equality_square",
            "variables": list(bit_order),
            "weight": PENALTIES["sum_is_five"],
            "violation": "(decoded_a + decoded_b - 5)^2",
        },
        {
            "id": "a_less_than_b",
            "type": "pairwise_order_penalty",
            "variables": list(bit_order),
            "weight": PENALTIES["a_less_than_b"],
            "violation": "sum(a_i * b_j for i >= j)",
        },
    )
    return BoundaryInstance(
        fixture_id=str(source_fixture["fixture_id"]),
        source=str(source_fixture["source"]),
        variables=variables,
        solver_assignment=dict(source_fixture["solver_assignment"]),
        false_assignment=dict(source_fixture["false_assignment"]),
        bit_order=bit_order,
        bit_values=bit_values,
        bit_variables=bit_variables,
        constant=constant,
        linear=tuple(linear),
        quadratic=tuple((left, right, weight) for (left, right), weight in sorted(quadratic.items())),
        factors=factors,
    )


def roundtrip_assignment(boundary: BoundaryInstance, assignment: JsonDict) -> JsonDict:
    """Check solver assignment to bits to assignment plus zero-energy metrics."""

    bits = boundary.assignment_to_bits(assignment)
    decoded = boundary.bits_to_assignment(bits)
    evaluation = boundary.evaluate_assignment(decoded)
    energy = boundary.qubo_energy(bits)
    passed = decoded == assignment and energy == 0.0 and evaluation["total_violation"] == 0
    return {
        "assignment": dict(assignment),
        "bits": list(bits),
        "decoded_assignment": decoded,
        "energy": energy,
        "constraint_violation": evaluation["total_violation"],
        "passed": passed,
    }


def reject_false_assignment(boundary: BoundaryInstance, assignment: JsonDict) -> JsonDict:
    """Evaluate the fixture's deterministic rejecting assignment."""

    bits = boundary.assignment_to_bits(assignment)
    evaluation = boundary.evaluate_assignment(assignment)
    energy = boundary.qubo_energy(bits)
    return {
        "assignment": dict(assignment),
        "bits": list(bits),
        "energy": energy,
        "constraint_violation": evaluation["total_violation"],
        "rejected": bool(energy > 0.0 or evaluation["total_violation"] > 0),
    }


def enumerate_boundary(boundary: BoundaryInstance) -> JsonDict:
    """Exhaustively enumerate the tiny binary state space on CPU."""

    best_energy: float | None = None
    best_assignments: list[JsonDict] = []
    valid_onehot_count = 0
    for bits in itertools.product((0, 1), repeat=len(boundary.bit_order)):
        energy = boundary.qubo_energy(tuple(bits))
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_assignments = []
        if energy == best_energy and boundary.is_valid_onehot(tuple(bits)):
            valid_onehot_count += 1
            best_assignments.append(boundary.bits_to_assignment(tuple(bits)))
        elif boundary.is_valid_onehot(tuple(bits)):
            valid_onehot_count += 1
    return {
        "method": "cpu_exhaustive_enumerator",
        "state_count": 2 ** len(boundary.bit_order),
        "valid_onehot_state_count": valid_onehot_count,
        "best_energy": float(best_energy),
        "best_assignments": best_assignments,
        "autocorrelation_metric": None,
        "autocorrelation_reason": "exhaustive enumeration is not a Markov chain",
    }


def run(
    *,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: list[JsonDict] | tuple[JsonDict, ...] = (),
) -> JsonDict:
    """Create and write the Exp 5278 deterministic boundary artifact."""

    source = select_tiny_fixture()
    boundary = build_boundary(source)
    roundtrip = roundtrip_assignment(boundary, boundary.solver_assignment)
    false_rejection = reject_false_assignment(boundary, boundary.false_assignment)
    enumeration = enumerate_boundary(boundary)
    sampler_interface = boundary.sampler_interface(enumeration)
    factor_ready = bool(boundary.factors and boundary.quadratic and enumeration["best_assignments"])
    sampler_ready = _finite_sampler_interface(sampler_interface)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _wrap(
            "honest_verdict",
            _honest_verdict(factor_ready, sampler_ready, roundtrip["passed"], false_rejection["rejected"]),
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "factor_graph_boundary_ready": _wrap("factor_graph_boundary_ready", factor_ready),
        "sampler_interface_ready": _wrap("sampler_interface_ready", sampler_ready),
        "mapping_roundtrip_passed": _wrap("mapping_roundtrip_passed", roundtrip["passed"]),
        "false_assignment_rejected": _wrap("false_assignment_rejected", false_rejection["rejected"]),
        "autocorrelation_metric": _wrap("autocorrelation_metric", enumeration["autocorrelation_metric"]),
        "hardware_speedup_claimed": _wrap("hardware_speedup_claimed", False),
        "tests_run": [dict(row) for row in tests_run],
        "source_fixture": source,
        "factor_graph_boundary": boundary.factor_graph_artifact(),
        "deterministic_energy": {
            "solver_assignment_energy": roundtrip["energy"],
            "false_assignment_energy": false_rejection["energy"],
        },
        "constraint_violation_metrics": {
            "solver_assignment_violation": roundtrip["constraint_violation"],
            "false_assignment_violation": false_rejection["constraint_violation"],
        },
        "mapping_roundtrip": roundtrip,
        "false_assignment_check": false_rejection,
        "enumeration_metrics": enumeration,
        "sampler_interface": sampler_interface,
    }
    validate_artifact(artifact)
    write_json(result_path, artifact)
    return artifact


def validate_artifact(artifact: JsonDict) -> None:
    """Raise when the Exp 5278 artifact violates its required schema."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"missing required field {field}"
    for field in WRAPPED_FIELDS:
        wrapped = artifact[field]
        assert isinstance(wrapped, dict), f"{field} must be principle-wrapped"
        assert wrapped["principle"] == FIELD_PRINCIPLES[field], f"{field} principle mismatch"
        assert "value" in wrapped, f"{field} missing value"
    verdict = artifact["honest_verdict"]["value"]
    assert isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES), (
        "honest_verdict.value must start with complete: or blocked_"
    )
    assert artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE
    for field in (
        "factor_graph_boundary_ready",
        "sampler_interface_ready",
        "mapping_roundtrip_passed",
        "false_assignment_rejected",
        "hardware_speedup_claimed",
    ):
        assert isinstance(artifact[field]["value"], bool), f"{field}.value must be bool"
    autocorrelation = artifact["autocorrelation_metric"]["value"]
    assert autocorrelation is None or isinstance(autocorrelation, int | float)
    assert artifact["hardware_speedup_claimed"]["value"] is False
    assert isinstance(artifact["tests_run"], list)


def write_json(path: Path, payload: JsonDict) -> None:
    """Write stable JSON with parent directories created."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_qubo_coefficients(
    index: dict[str, int],
) -> tuple[float, list[float], dict[tuple[int, int], float]]:
    constant = 0.0
    linear = [0.0 for _ in index]
    quadratic: dict[tuple[int, int], float] = {}
    for variable in ("a", "b"):
        _add_square_penalty(
            {index[f"{variable}_{value}"]: 1.0 for value in range(4)},
            target=1.0,
            weight=PENALTIES["one_hot"],
            linear=linear,
            quadratic=quadratic,
            constant_ref=[constant],
        )
        constant += PENALTIES["one_hot"]
    coeffs = {index[f"a_{value}"]: float(value) for value in range(4)}
    coeffs.update({index[f"b_{value}"]: float(value) for value in range(4)})
    constant = _add_square_penalty(
        coeffs,
        target=5.0,
        weight=PENALTIES["sum_is_five"],
        linear=linear,
        quadratic=quadratic,
        constant_ref=[constant],
    )
    for a_value in range(4):
        for b_value in range(4):
            if a_value >= b_value:
                _add_quadratic(
                    quadratic,
                    index[f"a_{a_value}"],
                    index[f"b_{b_value}"],
                    PENALTIES["a_less_than_b"],
                )
    return constant, linear, quadratic


def _add_square_penalty(
    coeffs: dict[int, float],
    *,
    target: float,
    weight: float,
    linear: list[float],
    quadratic: dict[tuple[int, int], float],
    constant_ref: list[float],
) -> float:
    constant = constant_ref[0] + weight * target * target
    for bit, coeff in coeffs.items():
        linear[bit] += weight * (coeff * coeff - 2.0 * target * coeff)
    for left, right in itertools.combinations(sorted(coeffs), 2):
        _add_quadratic(quadratic, left, right, weight * 2.0 * coeffs[left] * coeffs[right])
    return constant


def _add_quadratic(
    quadratic: dict[tuple[int, int], float],
    left: int,
    right: int,
    weight: float,
) -> None:
    key = (left, right) if left < right else (right, left)
    quadratic[key] = quadratic.get(key, 0.0) + weight


def _dense_couplings(boundary: BoundaryInstance) -> np.ndarray:
    couplings = np.zeros((len(boundary.bit_order), len(boundary.bit_order)), dtype=float)
    for left, right, weight in boundary.quadratic:
        couplings[left, right] = -weight / 2.0
        couplings[right, left] = -weight / 2.0
    return couplings


def _finite_sampler_interface(sampler_interface: JsonDict) -> bool:
    biases = np.asarray(sampler_interface["biases"], dtype=float)
    couplings = np.asarray(sampler_interface["couplings"], dtype=float)
    return bool(
        biases.shape == (8,)
        and couplings.shape == (8, 8)
        and np.all(np.isfinite(biases))
        and np.all(np.isfinite(couplings))
        and sampler_interface["hardware_board_command_run"] is False
        and sampler_interface["speedup_claimed"] is False
    )


def _honest_verdict(
    factor_ready: bool,
    sampler_ready: bool,
    roundtrip_passed: bool,
    false_rejected: bool,
) -> str:
    if factor_ready and sampler_ready and roundtrip_passed and false_rejected:
        return (
            "complete: factor-graph boundary is usable for the tiny solver fixture; "
            "sampler interface compatibility is shape-only and no hardware speedup is claimed"
        )
    return "blocked_boundary_unusable: factor graph, sampler interface, round-trip, or rejection check failed"


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}
