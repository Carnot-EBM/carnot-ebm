"""Cold-audit serialized Exp6751 compiler outputs with separate evaluators.

This module does not import the compiler under test. It rebuilds each compiled
conditional from serialized factor data and compiler parameters. A separate
path enumerates every bounded trajectory. A direct sampler estimates the same
trajectory distance from likelihood ratios and frozen seeds.

Spec: REQ-HW-6766, REQ-HW-6766-PRECONDITIONS,
REQ-HW-6766-INDEPENDENCE, REQ-HW-6766-EXACT,
REQ-HW-6766-SAMPLER, REQ-HW-6766-ROWS,
REQ-HW-6766-CIRCULARITY, REQ-HW-6766-COMPLETION,
REQ-HW-6766-BOUNDARY, SCENARIO-HW-6766-COLD-REPRODUCTION,
SCENARIO-HW-6766-CIRCULAR-REFINEMENT,
SCENARIO-HW-6766-FAIL-CLOSED, REQ-REPORT-6766,
SCENARIO-REPORT-6766-ATOMIC, SCENARIO-REPORT-6766-BLOCKED.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import platform
import random
import re
import struct
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = Path("results/experiment_6751_thermalizer_factor_trajectory_fidelity.json")
RESULT_PATH = Path("results/experiment_6766_thermalizer_independent_trajectory_audit.json")
MODULE_PATH = Path("python/carnot/experiment_6766_thermalizer_independent_trajectory_audit.py")
COMPILER_MODULE_PATH = Path(
    "python/carnot/experiment_6751_thermalizer_factor_trajectory_fidelity.py"
)
SCRIPT_PATH = Path(
    "scripts/experiments/experiment_6766_thermalizer_independent_trajectory_audit.py"
)
HARDWARE_SPEC_PATH = Path("openspec/capabilities/hardware/spec.md")
REPORTING_SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")

EXPERIMENT_ID = "experiment_6766_thermalizer_independent_trajectory_audit"
SCHEMA_VERSION = "carnot.experiment_6766.thermalizer_independent_trajectory_audit.v1"
INFERENCE_SUBSTRATE = "local simulator/compiler cold audit; no physical TSU"
CLAIM_BOUNDARY = "simulator-only; no speed, power, X0, Z1, FPGA, or physical-hardware claim"
METHODS = ("independent_factor", "context_matched", "trajectory_refinement")
DEPTHS = (1, 2, 4, 8)
PRECISIONS = ("binary32", "fixed_q3_4")
EVALUATOR_PATHS = ("exact_enumerator", "direct_sampler")
MAXIMUM_PATHS = 20_000
NORMALIZATION_TOLERANCE = 1.0e-12
METRIC_TOLERANCE = 1.0e-12
DEFAULT_SAMPLES_PER_ROW = 4096

FORBIDDEN_SOURCE_FUNCTIONS = (
    "conditional_kl",
    "enumerate_trajectory_distribution",
    "total_variation",
    "context_input_weights",
    "training_context_weights",
    "candidate_bank",
    "_fit_arm",
    "build_rows",
    "derive_aggregates",
)

SPEC_REFS = [
    "REQ-HW-6766",
    "REQ-HW-6766-PRECONDITIONS",
    "REQ-HW-6766-INDEPENDENCE",
    "REQ-HW-6766-EXACT",
    "REQ-HW-6766-SAMPLER",
    "REQ-HW-6766-ROWS",
    "REQ-HW-6766-CIRCULARITY",
    "REQ-HW-6766-COMPLETION",
    "REQ-HW-6766-BOUNDARY",
    "SCENARIO-HW-6766-COLD-REPRODUCTION",
    "SCENARIO-HW-6766-CIRCULAR-REFINEMENT",
    "SCENARIO-HW-6766-FAIL-CLOSED",
    "REQ-REPORT-6766",
    "SCENARIO-REPORT-6766-ATOMIC",
    "SCENARIO-REPORT-6766-BLOCKED",
]


class AuditInputError(ValueError):
    """Report serialized evidence whose probability semantics are ambiguous."""


class AuditBoundError(RuntimeError):
    """Stop exact work before an input can exceed the frozen finite bound."""


def canonical_json_text(value: Any) -> str:
    """Encode finite JSON in one stable form for every evidence digest."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AuditInputError("evidence must be finite canonical JSON") from exc


def bytes_digest(value: bytes) -> str:
    """Prefix SHA-256 values so a digest cannot look like source evidence."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def json_digest(value: Any) -> str:
    """Bind one typed JSON value without depending on file formatting."""

    return bytes_digest(canonical_json_text(value).encode("utf-8"))


def file_digest(path: Path) -> str:
    """Keep a missing file distinct from a present empty file."""

    return bytes_digest(path.read_bytes()) if path.is_file() else "missing"


def receipt_digest(receipt: Mapping[str, Any]) -> str:
    """Hash a receipt after removing its self-referential digest field."""

    return json_digest({key: value for key, value in receipt.items() if key != "receipt_sha256"})


def audit_row_digest(row: Mapping[str, Any]) -> str:
    """Hash an audit row without including its own digest."""

    return json_digest({key: value for key, value in row.items() if key != "row_sha256"})


def find_spec_anchors(text: str) -> list[str]:
    """Find stable requirement identifiers for test-to-spec coverage."""

    return sorted(set(re.findall(r"(?:REQ|SCENARIO)-[A-Z0-9-]+", text)))


def _rounded(value: float) -> float:
    """Keep JSON stable while retaining more precision than any audit gate."""

    result = float(f"{float(value):.15g}")
    return 0.0 if result == 0.0 else result


def _binary_factor() -> JsonDict:
    """Serialize the binary target and feature tensor without compiler code."""

    tensor = []
    for input_value in range(2):
        row = []
        for output_value in range(2):
            input_spin = -1.0 if input_value == 0 else 1.0
            output_spin = -1.0 if output_value == 0 else 1.0
            row.append([output_spin, input_spin * output_spin])
        tensor.append(row)
    return {
        "factor_id": "binary_sticky_transition",
        "factor_kind": "binary",
        "topology_id": "sparse_ebm:binary_sticky_transition",
        "categories": ["zero", "one"],
        "target_conditional": [[0.98, 0.02], [0.38, 0.62]],
        "parameter_names": ["output_bias", "input_output_coupler"],
        "feature_tensor": tensor,
        "parameter_bound": 0.75,
        "contexts": {
            "binary_sticky_transition:left_heavy": [0.92, 0.08],
            "binary_sticky_transition:right_heavy": [0.08, 0.92],
        },
    }


def _categorical_factor() -> JsonDict:
    """Serialize the three-category target and sparse feature tensor."""

    tensor = []
    for input_value in range(3):
        row = []
        for output_value in range(3):
            row.append(
                [
                    1.0 if output_value == 0 else 0.0,
                    1.0 if output_value == 1 else 0.0,
                    1.0 if output_value == input_value else 0.0,
                    1.0 if output_value == (input_value + 1) % 3 else 0.0,
                ]
            )
        tensor.append(row)
    return {
        "factor_id": "categorical_ring_transition",
        "factor_kind": "categorical",
        "topology_id": "sparse_ebm:categorical_ring_transition",
        "categories": ["red", "green", "blue"],
        "target_conditional": [
            [0.90, 0.08, 0.02],
            [0.15, 0.25, 0.60],
            [0.55, 0.10, 0.35],
        ],
        "parameter_names": [
            "output_red_bias",
            "output_green_bias",
            "same_category_coupler",
            "forward_ring_coupler",
        ],
        "feature_tensor": tensor,
        "parameter_bound": 1.0,
        "contexts": {
            "categorical_ring_transition:left_heavy": [0.86, 0.10, 0.04],
            "categorical_ring_transition:right_heavy": [0.04, 0.10, 0.86],
        },
    }


def serialized_factors() -> list[JsonDict]:
    """Return isolated JSON factors that the evaluator may consume."""

    return deepcopy([_binary_factor(), _categorical_factor()])


def _probability_vector(values: Sequence[Any], size: int, label: str) -> list[float]:
    """Validate one finite normalized vector before probability arithmetic."""

    if len(values) != size:
        raise AuditInputError(f"{label} has the wrong size")
    vector = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in vector):
        raise AuditInputError(f"{label} must be finite and nonnegative")
    if abs(math.fsum(vector) - 1.0) > NORMALIZATION_TOLERANCE:
        raise AuditInputError(f"{label} must normalize")
    return vector


def _transition_matrix(values: Sequence[Sequence[Any]], label: str) -> list[list[float]]:
    """Validate one finite row-stochastic square matrix."""

    size = len(values)
    if size < 2 or any(len(row) != size for row in values):
        raise AuditInputError(f"{label} must be a square matrix")
    return [_probability_vector(row, size, f"{label} row") for row in values]


def _float32(value: float) -> float:
    """Round one scalar through IEEE-754 binary32 without NumPy."""

    return float(struct.unpack("<f", struct.pack("<f", float(value)))[0])


def build_compiled_conditional(
    factor: Mapping[str, Any], parameters: Sequence[Any], precision: str
) -> list[list[float]]:
    """Normalize serialized sparse logits with an independent implementation."""

    names = list(factor["parameter_names"])
    if len(parameters) != len(names):
        raise AuditInputError("compiled parameter count does not match factor capacity")
    if precision not in PRECISIONS:
        raise AuditInputError(f"unknown precision: {precision}")
    values = [float(value) for value in parameters]
    if any(not math.isfinite(value) for value in values):
        raise AuditInputError("compiled parameters must be finite")
    if precision == "binary32":
        values = [_float32(value) for value in values]
    tensor = factor["feature_tensor"]
    size = len(factor["categories"])
    if (
        len(tensor) != size
        or any(len(row) != size for row in tensor)
        or any(len(features) != len(values) for row in tensor for features in row)
    ):
        raise AuditInputError("feature tensor does not match the serialized factor")

    conditional = []
    for input_row in tensor:
        logits = []
        for features in input_row:
            if precision == "binary32":
                accumulator = _float32(0.0)
                for feature, parameter in zip(features, values, strict=True):
                    term = _float32(_float32(float(feature)) * parameter)
                    accumulator = _float32(accumulator + term)
                logits.append(accumulator)
            else:
                logits.append(
                    math.fsum(
                        float(feature) * parameter
                        for feature, parameter in zip(features, values, strict=True)
                    )
                )
        shift = max(logits)
        weights = [math.exp(value - shift) for value in logits]
        normalizer = math.fsum(weights)
        conditional.append([value / normalizer for value in weights])
    return conditional


def enumerate_path_law(
    initial: Sequence[Any],
    transition: Sequence[Sequence[Any]],
    depth: int,
    *,
    maximum_paths: int = MAXIMUM_PATHS,
) -> dict[tuple[int, ...], float]:
    """Enumerate each complete path with code that is separate from Exp6751."""

    if depth <= 0:
        raise AuditInputError("trajectory depth must be positive")
    matrix = _transition_matrix(transition, "transition")
    start = _probability_vector(initial, len(matrix), "initial distribution")
    path_count = len(matrix) ** (depth + 1)
    if path_count > maximum_paths:
        raise AuditBoundError(f"trajectory path count {path_count} exceeds {maximum_paths}")
    law: dict[tuple[int, ...], float] = {}
    for path in product(range(len(matrix)), repeat=depth + 1):
        probability = start[path[0]]
        for step in range(depth):
            probability *= matrix[path[step]][path[step + 1]]
        law[path] = probability
    return law


def _path_law_digest(law: Mapping[tuple[int, ...], float]) -> str:
    """Rebuild the documented little-endian path and probability receipt."""

    path_bytes = b"".join(struct.pack("<h", value) for path in law for value in path)
    probability_bytes = b"".join(struct.pack("<d", probability) for probability in law.values())
    return bytes_digest(path_bytes + probability_bytes)


def law_total_variation(
    left: Mapping[tuple[int, ...], float], right: Mapping[tuple[int, ...], float]
) -> float:
    """Compute total variation only when both exact laws share support."""

    if left.keys() != right.keys():
        raise AuditInputError("trajectory laws must have the same support")
    return _rounded(0.5 * math.fsum(abs(left[path] - right[path]) for path in left))


def conditional_kl_metrics(
    target: Sequence[Sequence[Any]],
    compiled: Sequence[Sequence[Any]],
    initial: Sequence[Any],
    depth: int,
) -> JsonDict:
    """Compute per-input KL and target-visitation weighting from first principles."""

    target_matrix = _transition_matrix(target, "target conditional")
    compiled_matrix = _transition_matrix(compiled, "compiled conditional")
    if len(target_matrix) != len(compiled_matrix):
        raise AuditInputError("conditional matrices must have the same size")
    marginal = _probability_vector(initial, len(target_matrix), "context initial")
    accumulated = [0.0] * len(target_matrix)
    for _step in range(depth):
        accumulated = [left + right for left, right in zip(accumulated, marginal, strict=True)]
        marginal = [
            math.fsum(
                marginal[input_value] * target_matrix[input_value][output_value]
                for input_value in range(len(target_matrix))
            )
            for output_value in range(len(target_matrix))
        ]
    weights = [value / depth for value in accumulated]
    per_input = []
    for target_row, compiled_row in zip(target_matrix, compiled_matrix, strict=True):
        if any(value <= 0.0 for value in compiled_row):
            raise AuditInputError("compiled conditional must have full support")
        per_input.append(
            math.fsum(
                target_value * math.log(target_value / compiled_value)
                for target_value, compiled_value in zip(target_row, compiled_row, strict=True)
                if target_value > 0.0
            )
        )
    return {
        "input_weights": [_rounded(value) for value in weights],
        "conditional_kl_by_input": [_rounded(value) for value in per_input],
        "conditional_kl": _rounded(
            math.fsum(weight * value for weight, value in zip(weights, per_input, strict=True))
        ),
    }


def _draw_category(probabilities: Sequence[float], generator: random.Random) -> int:
    """Draw one category directly from a serialized categorical law."""

    threshold = generator.random()
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if threshold < cumulative:
            return index
    return len(probabilities) - 1


def sample_trajectory_tv(
    initial: Sequence[Any],
    target: Sequence[Sequence[Any]],
    compiled: Sequence[Sequence[Any]],
    depth: int,
    *,
    seed: int,
    samples: int,
) -> JsonDict:
    """Estimate TV from target samples without calling the exact enumerator."""

    if samples < 2:
        raise AuditInputError("direct sample count must be at least two")
    target_matrix = _transition_matrix(target, "target conditional")
    compiled_matrix = _transition_matrix(compiled, "compiled conditional")
    if depth <= 0:
        raise AuditInputError("trajectory depth must be positive")
    start = _probability_vector(initial, len(target_matrix), "context initial")
    generator = random.Random(int(seed))
    contributions = []
    for _sample in range(samples):
        state = _draw_category(start, generator)
        likelihood_ratio = 1.0
        for _step in range(depth):
            next_state = _draw_category(target_matrix[state], generator)
            target_probability = target_matrix[state][next_state]
            if target_probability <= 0.0:
                raise AuditInputError("target sampler requires full support")
            likelihood_ratio *= compiled_matrix[state][next_state] / target_probability
            state = next_state
        contributions.append(max(1.0 - likelihood_ratio, 0.0))
    estimate = math.fsum(contributions) / samples
    variance = math.fsum((value - estimate) ** 2 for value in contributions) / (samples - 1)
    standard_error = math.sqrt(variance / samples)
    radius = 2.5758293035489004 * standard_error
    return {
        "api_path": (
            "carnot.experiment_6766_thermalizer_independent_trajectory_audit.sample_trajectory_tv"
        ),
        "sample_count": samples,
        "sample_seed": int(seed),
        "sampled_trajectory_tv": _rounded(estimate),
        "standard_error": _rounded(standard_error),
        "ci99_low": _rounded(max(0.0, estimate - radius)),
        "ci99_high": _rounded(min(1.0, estimate + radius)),
    }


def local_call_names(tree: ast.AST) -> list[str]:
    """List local and attribute call names for the isolation test."""

    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    return sorted(set(names))


def _imports(tree: ast.AST) -> list[str]:
    """List imported module names without importing those modules."""

    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return sorted(set(names))


def _defined_functions(tree: ast.AST) -> list[str]:
    """List local function names for provenance, not code reuse."""

    return sorted(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )


def dependency_graph_receipt(source: Mapping[str, Any]) -> JsonDict:
    """Record module hashes and prove that no compiler callable is imported."""

    compiler_path = REPO_ROOT / COMPILER_MODULE_PATH
    evaluator_path = REPO_ROOT / MODULE_PATH
    compiler_tree = ast.parse(compiler_path.read_text(encoding="utf-8"))
    evaluator_tree = ast.parse(evaluator_path.read_text(encoding="utf-8"))
    compiler_imports = _imports(compiler_tree)
    evaluator_imports = _imports(evaluator_tree)
    evaluator_imports_compiler = any("experiment_6751" in name for name in evaluator_imports)
    compiler_imports_evaluator = any("experiment_6766" in name for name in compiler_imports)
    compiler_declared = (
        source.get("compiler_provenance", {}).get("internal", {}).get("module_sha256", "missing")
    )
    receipt: JsonDict = {
        "compiler_module_path": COMPILER_MODULE_PATH.as_posix(),
        "evaluator_module_path": MODULE_PATH.as_posix(),
        "compiler_declared_module_sha256": compiler_declared,
        "compiler_current_module_sha256": file_digest(compiler_path),
        "evaluator_module_sha256": file_digest(evaluator_path),
        "compiler_source_matches_declared_hash": file_digest(compiler_path) == compiler_declared,
        "same_module_sha256": file_digest(compiler_path) == file_digest(evaluator_path),
        "compiler_imports": compiler_imports,
        "evaluator_imports": evaluator_imports,
        "evaluator_imports_compiler_module": evaluator_imports_compiler,
        "compiler_imports_evaluator_module": compiler_imports_evaluator,
        "compiler_to_evaluator_dependency_edge": compiler_imports_evaluator,
        "evaluator_to_compiler_dependency_edge": evaluator_imports_compiler,
        "shared_callable_objects": [],
        "same_named_local_functions": sorted(
            set(_defined_functions(compiler_tree)) & set(_defined_functions(evaluator_tree))
        ),
        "forbidden_source_calls_found": sorted(
            set(FORBIDDEN_SOURCE_FUNCTIONS) & set(local_call_names(evaluator_tree))
        ),
        "method_objectives": deepcopy(
            source.get("frozen_config", {}).get("selection_objectives", {})
        ),
        "methods_consuming_exact_evaluator_outcome": ["trajectory_refinement"],
    }
    receipt["receipt_sha256"] = receipt_digest(receipt)
    return receipt


def _gate(check: str, expected: Any, observed: Any) -> JsonDict:
    """Keep expected and observed evidence on every failed precondition."""

    return {"check": check, "expected": expected, "observed": observed}


def _expected_source_grid(source: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    """Build the planned source grid from isolated factors and frozen seeds."""

    bundles = source.get("random_seed", {}).get("bundles", [])
    return {
        (factor["factor_id"], context_id, method, depth, precision, bundle["seed_bundle_id"])
        for factor in serialized_factors()
        for context_id in factor["contexts"]
        for method in METHODS
        for depth in DEPTHS
        for precision in PRECISIONS
        for bundle in bundles
    }


def check_source_preconditions(source: Mapping[str, Any]) -> JsonDict:
    """Check all frozen source receipts before any cold measurement runs."""

    required = {
        "rows",
        "frozen_config",
        "topology_receipts",
        "precision_receipts",
        "random_seed",
        "compiler_provenance",
        "reproducibility_checksum",
    }
    missing = sorted(required - set(source))
    if missing:
        return {
            "gate_check_summary": [
                _gate("required_source_fields", {"missing": []}, {"missing": missing})
            ],
            "topology_mismatches": [],
            "precision_mismatches": [],
            "maximum_planned_path_count": 0,
        }

    errors = []
    topology_mismatches = []
    precision_mismatches = []
    topology = {row.get("factor_id"): row for row in source["topology_receipts"]}
    for factor in serialized_factors():
        observed = topology.get(factor["factor_id"])
        expected_fields = {
            "factor_id": factor["factor_id"],
            "factor_kind": factor["factor_kind"],
            "topology_id": factor["topology_id"],
            "categories": factor["categories"],
            "parameter_names": factor["parameter_names"],
            "factor_capacity": len(factor["parameter_names"]),
            "target_conditional_sha256": json_digest(factor["target_conditional"]),
        }
        mismatch = {
            key: {"expected": value, "observed": None if observed is None else observed.get(key)}
            for key, value in expected_fields.items()
            if observed is None or observed.get(key) != value
        }
        if observed is not None and observed.get("receipt_sha256") != receipt_digest(observed):
            mismatch["receipt_sha256"] = {
                "expected": receipt_digest(observed),
                "observed": observed.get("receipt_sha256"),
            }
        if mismatch:
            topology_mismatches.append({"factor_id": factor["factor_id"], "fields": mismatch})
    if topology_mismatches:
        errors.append(
            _gate(
                "topology_receipts",
                {"mismatch_count": 0},
                {"mismatch_count": len(topology_mismatches)},
            )
        )

    precision_by_id = {row.get("precision_id"): row for row in source["precision_receipts"]}
    for precision_id in PRECISIONS:
        observed = precision_by_id.get(precision_id)
        if observed is None:
            precision_mismatches.append({"precision_id": precision_id, "reason": "missing"})
        elif observed.get("receipt_sha256") != receipt_digest(observed):
            precision_mismatches.append(
                {
                    "precision_id": precision_id,
                    "expected_receipt_sha256": receipt_digest(observed),
                    "observed_receipt_sha256": observed.get("receipt_sha256"),
                }
            )
    if precision_mismatches:
        errors.append(
            _gate(
                "precision_receipts",
                {"mismatch_count": 0},
                {"mismatch_count": len(precision_mismatches)},
            )
        )

    rows = source["rows"] if isinstance(source["rows"], list) else []
    observed_grid = {
        (
            row.get("factor_id"),
            row.get("context_id"),
            row.get("arm"),
            row.get("depth"),
            row.get("precision"),
            row.get("seed_bundle_id"),
        )
        for row in rows
    }
    expected_grid = _expected_source_grid(source)
    invalid_row_hashes = [
        row.get("row_id")
        for row in rows
        if row.get("row_sha256")
        != json_digest({key: value for key, value in row.items() if key != "row_sha256"})
    ]
    required_row_fields = {
        "compiled_parameters",
        "context_initial",
        "target_trajectory_sha256",
        "compiled_trajectory_sha256",
        "topology_receipt_sha256",
        "factor_seed",
        "train_seed",
        "trajectory_seed",
    }
    incomplete_rows = [row.get("row_id") for row in rows if not required_row_fields <= set(row)]
    if (
        len(rows) != len(expected_grid)
        or observed_grid != expected_grid
        or invalid_row_hashes
        or incomplete_rows
    ):
        errors.append(
            _gate(
                "complete_source_row_grid",
                {"row_count": len(expected_grid), "invalid_hash_count": 0},
                {
                    "row_count": len(rows),
                    "unique_grid_count": len(observed_grid),
                    "invalid_hash_count": len(invalid_row_hashes),
                    "incomplete_row_count": len(incomplete_rows),
                },
            )
        )

    maximum_planned = max(
        len(factor["categories"]) ** (depth + 1)
        for factor in serialized_factors()
        for depth in DEPTHS
    )
    declared_maximum = source["frozen_config"].get("maximum_enumerated_trajectory_states")
    if (
        not isinstance(declared_maximum, int)
        or maximum_planned > declared_maximum
        or maximum_planned > MAXIMUM_PATHS
    ):
        errors.append(
            _gate(
                "bounded_exact_enumeration",
                {"maximum_planned_path_count": maximum_planned, "hard_limit": MAXIMUM_PATHS},
                {"declared_limit": declared_maximum},
            )
        )

    dependency = dependency_graph_receipt(source)
    if (
        dependency["evaluator_imports_compiler_module"]
        or dependency["forbidden_source_calls_found"]
        or dependency["same_module_sha256"]
    ):
        errors.append(
            _gate(
                "independent_evaluator_isolation",
                {"compiler_import": False, "forbidden_calls": [], "same_hash": False},
                {
                    "compiler_import": dependency["evaluator_imports_compiler_module"],
                    "forbidden_calls": dependency["forbidden_source_calls_found"],
                    "same_hash": dependency["same_module_sha256"],
                },
            )
        )
    return {
        "gate_check_summary": errors,
        "topology_mismatches": topology_mismatches,
        "precision_mismatches": precision_mismatches,
        "maximum_planned_path_count": maximum_planned,
    }


def _normalization_error(matrix: Sequence[Sequence[float]]) -> float:
    """Return the largest row-mass error for one conditional matrix."""

    return max(abs(math.fsum(row) - 1.0) for row in matrix)


def _sampler_seed(row: Mapping[str, Any]) -> int:
    """Derive one stable sampler seed from the frozen trajectory seed and row."""

    token = f"{row['row_id']}:{row['trajectory_seed']}:direct_sampler".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def evaluate_source_rows(source: Mapping[str, Any], *, samples_per_row: int) -> list[JsonDict]:
    """Emit exact and direct-sampler rows for each serialized compiler row."""

    factors = {factor["factor_id"]: factor for factor in serialized_factors()}
    audit_rows = []
    for source_row in source["rows"]:
        factor = factors[source_row["factor_id"]]
        target = factor["target_conditional"]
        compiled = build_compiled_conditional(
            factor, source_row["compiled_parameters"], source_row["precision"]
        )
        initial = source_row["context_initial"]
        depth = int(source_row["depth"])
        target_law = enumerate_path_law(initial, target, depth)
        compiled_law = enumerate_path_law(initial, compiled, depth)
        exact_tv = law_total_variation(target_law, compiled_law)
        kl = conditional_kl_metrics(target, compiled, initial, depth)
        normalization = {
            "target_conditional": _rounded(_normalization_error(target)),
            "compiled_conditional": _rounded(_normalization_error(compiled)),
            "target_trajectory": _rounded(abs(math.fsum(target_law.values()) - 1.0)),
            "compiled_trajectory": _rounded(abs(math.fsum(compiled_law.values()) - 1.0)),
        }
        common: JsonDict = {
            "source_row_id": source_row["row_id"],
            "source_row_sha256": source_row["row_sha256"],
            "factor_id": source_row["factor_id"],
            "factor_kind": source_row["factor_kind"],
            "context_id": source_row["context_id"],
            "context_initial": list(source_row["context_initial"]),
            "method": source_row["arm"],
            "precision": source_row["precision"],
            "topology_id": source_row["topology_id"],
            "topology_receipt_sha256": source_row["topology_receipt_sha256"],
            "depth": depth,
            "seed_bundle_id": source_row["seed_bundle_id"],
            "factor_seed": source_row["factor_seed"],
            "train_seed": source_row["train_seed"],
            "trajectory_seed": source_row["trajectory_seed"],
            "compiled_parameters": list(source_row["compiled_parameters"]),
            "normalization_error": normalization,
            "maximum_normalization_error": _rounded(max(normalization.values())),
            "evaluator_distinct": True,
        }
        exact_row = {
            **common,
            "row_id": f"{source_row['row_id']}::exact_enumerator",
            "evaluator_path": "exact_enumerator",
            "evaluator_api_path": (
                "carnot.experiment_6766_thermalizer_independent_trajectory_audit.enumerate_path_law"
            ),
            "conditional_kl_by_input": {
                category: value
                for category, value in zip(
                    factor["categories"], kl["conditional_kl_by_input"], strict=True
                )
            },
            "conditional_kl": kl["conditional_kl"],
            "trajectory_tv": exact_tv,
            "sampled_trajectory_tv": None,
            "sample_standard_error": None,
            "sample_ci99": None,
            "sample_count": 0,
            "sample_seed": None,
            "exact_trajectory_tv_reference": None,
            "source_conditional_kl_delta": _rounded(
                kl["conditional_kl"] - float(source_row["conditional_kl"])
            ),
            "source_trajectory_tv_delta": _rounded(exact_tv - float(source_row["trajectory_tv"])),
            "target_trajectory_receipt_matches": (
                _path_law_digest(target_law) == source_row["target_trajectory_sha256"]
            ),
            "compiled_trajectory_receipt_matches": (
                _path_law_digest(compiled_law) == source_row["compiled_trajectory_sha256"]
            ),
            "mechanism_consumes_evaluator_outcome": (source_row["arm"] == "trajectory_refinement"),
        }
        exact_row["row_sha256"] = audit_row_digest(exact_row)
        audit_rows.append(exact_row)

        sample = sample_trajectory_tv(
            initial,
            target,
            compiled,
            depth,
            seed=_sampler_seed(source_row),
            samples=samples_per_row,
        )
        sample_row = {
            **common,
            "row_id": f"{source_row['row_id']}::direct_sampler",
            "evaluator_path": "direct_sampler",
            "evaluator_api_path": sample["api_path"],
            "conditional_kl_by_input": None,
            "conditional_kl": None,
            "trajectory_tv": None,
            "sampled_trajectory_tv": sample["sampled_trajectory_tv"],
            "sample_standard_error": sample["standard_error"],
            "sample_ci99": [sample["ci99_low"], sample["ci99_high"]],
            "sample_count": sample["sample_count"],
            "sample_seed": sample["sample_seed"],
            "exact_trajectory_tv_reference": exact_tv,
            "source_conditional_kl_delta": None,
            "source_trajectory_tv_delta": None,
            "target_trajectory_receipt_matches": None,
            "compiled_trajectory_receipt_matches": None,
            "mechanism_consumes_evaluator_outcome": False,
        }
        sample_row["row_sha256"] = audit_row_digest(sample_row)
        audit_rows.append(sample_row)
    return audit_rows


def _mean_interval(values: Sequence[float], critical: float) -> JsonDict:
    """Return a row-derived mean and symmetric interval."""

    count = len(values)
    mean = math.fsum(values) / count
    variance = math.fsum((value - mean) ** 2 for value in values) / (count - 1)
    standard_error = math.sqrt(variance / count)
    radius = critical * standard_error
    return {
        "count": count,
        "mean": _rounded(mean),
        "standard_error": _rounded(standard_error),
        "interval_low": _rounded(mean - radius),
        "interval_high": _rounded(mean + radius),
    }


def _paired_rows(exact_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Pair each refinement row with its matched independent-factor row."""

    index = {
        (
            row["factor_id"],
            row["context_id"],
            row["precision"],
            row["topology_id"],
            row["depth"],
            row["seed_bundle_id"],
            row["method"],
        ): float(row["trajectory_tv"])
        for row in exact_rows
    }
    result = []
    for method in ("context_matched", "trajectory_refinement"):
        for depth_value in (*DEPTHS, "all"):
            deltas = []
            for key, independent_value in index.items():
                if key[-1] != "independent_factor":
                    continue
                if depth_value != "all" and key[4] != depth_value:
                    continue
                deltas.append(independent_value - index[(*key[:-1], method)])
            critical = 2.131449545559323 if len(deltas) == 16 else 1.998340542520741
            interval = _mean_interval(deltas, critical)
            result.append(
                {
                    "method": method,
                    "depth": depth_value,
                    "pair_count": interval["count"],
                    "mean_independent_minus_method_tv": interval["mean"],
                    "standard_error": interval["standard_error"],
                    "ci95_low": interval["interval_low"],
                    "ci95_high": interval["interval_high"],
                    "interval_excludes_zero": (
                        interval["interval_low"] > 0.0 or interval["interval_high"] < 0.0
                    ),
                    "improved_pair_count": sum(delta > 0.0 for delta in deltas),
                    "worsened_pair_count": sum(delta < 0.0 for delta in deltas),
                    "tied_pair_count": sum(delta == 0.0 for delta in deltas),
                    "interval_method": "paired_t_95",
                }
            )
    return result


def reduce_audit_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Derive every public metric only from retained cold-audit rows."""

    exact_rows = [row for row in rows if row["evaluator_path"] == "exact_enumerator"]
    sampled_rows = [row for row in rows if row["evaluator_path"] == "direct_sampler"]
    conditional_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    trajectory_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in exact_rows:
        conditional_groups[
            (
                row["factor_id"],
                row["method"],
                row["precision"],
                row["topology_id"],
                row["depth"],
                row["seed_bundle_id"],
            )
        ].append(float(row["conditional_kl"]))
        trajectory_groups[(row["method"], row["depth"])].append(float(row["trajectory_tv"]))
    conditional = []
    for key, values in sorted(conditional_groups.items()):
        interval = _mean_interval(values, 12.706204736432095)
        conditional.append(
            {
                "factor_id": key[0],
                "method": key[1],
                "precision": key[2],
                "topology_id": key[3],
                "depth": key[4],
                "seed_bundle_id": key[5],
                "context_count": interval["count"],
                "mean_conditional_kl": interval["mean"],
                "ci95_low": interval["interval_low"],
                "ci95_high": interval["interval_high"],
            }
        )
    trajectory = []
    for key, values in sorted(trajectory_groups.items()):
        interval = _mean_interval(values, 2.131449545559323)
        trajectory.append(
            {
                "method": key[0],
                "depth": key[1],
                "row_count": interval["count"],
                "mean_trajectory_tv": interval["mean"],
                "ci95_low": interval["interval_low"],
                "ci95_high": interval["interval_high"],
                "minimum_trajectory_tv": _rounded(min(values)),
                "maximum_trajectory_tv": _rounded(max(values)),
            }
        )
    paired = _paired_rows(exact_rows)
    absolute_errors = [
        abs(float(row["sampled_trajectory_tv"]) - float(row["exact_trajectory_tv_reference"]))
        for row in sampled_rows
    ]
    coverage_count = sum(
        float(row["sample_ci99"][0])
        <= float(row["exact_trajectory_tv_reference"])
        <= float(row["sample_ci99"][1])
        for row in sampled_rows
    )
    crosscheck = {
        "evaluator_path": "direct_sampler",
        "api_path": (
            "carnot.experiment_6766_thermalizer_independent_trajectory_audit.sample_trajectory_tv"
        ),
        "planned_row_count": 192,
        "observed_row_count": len(sampled_rows),
        "samples_per_row": sorted({row["sample_count"] for row in sampled_rows}),
        "total_sample_count": sum(int(row["sample_count"]) for row in sampled_rows),
        "mean_absolute_tv_error": _rounded(math.fsum(absolute_errors) / len(absolute_errors)),
        "maximum_absolute_tv_error": _rounded(max(absolute_errors)),
        "exact_in_ci99_count": coverage_count,
        "exact_in_ci99_rate": _rounded(coverage_count / len(sampled_rows)),
        "external_sampler_invoked": False,
        "external_sampler": None,
        "external_sampler_note": (
            "The audit uses Python random.Random directly. Torx and THRML are not invoked."
        ),
        "passed": (
            len(sampled_rows) == 192
            and math.fsum(absolute_errors) / len(absolute_errors) <= 0.10
            and coverage_count / len(sampled_rows) >= 0.85
        ),
    }
    return {
        "conditional_kl_by_factor": conditional,
        "trajectory_tv_by_depth": trajectory,
        "paired_trajectory_deltas": paired,
        "direct_sampler_crosscheck": crosscheck,
    }


def _compiler_provenance(source: Mapping[str, Any], source_sha256: str) -> JsonDict:
    """Keep serialized compiler identity separate from current worktree state."""

    declared = deepcopy(source.get("compiler_provenance", {}).get("internal", {}))
    declared_hash = declared.get("module_sha256", "missing")
    current_hash = file_digest(REPO_ROOT / COMPILER_MODULE_PATH)
    return {
        "source_artifact_path": SOURCE_PATH.as_posix(),
        "source_artifact_sha256": source_sha256,
        "identity": declared.get("identity", "missing"),
        "declared_module_path": declared.get("module_path", COMPILER_MODULE_PATH.as_posix()),
        "declared_module_sha256": declared_hash,
        "current_module_sha256": current_hash,
        "current_source_matches_declared_hash": current_hash == declared_hash,
        "source_snapshot_available_at_declared_hash": current_hash == declared_hash,
        "serialized_outputs_consumed": [
            "compiled_parameters",
            "topology_receipt_sha256",
            "precision",
            "seed_bundle_id",
            "trajectory_receipts",
        ],
        "authority_note": (
            "The audit measures frozen serialized outputs. A current-source hash mismatch stays "
            "visible and is not rewritten as compiler identity."
        ),
    }


def _evaluator_provenance() -> JsonDict:
    """Name both independent evaluator paths and their local code hash."""

    return {
        "identity": "carnot.exp6766.python_stdlib_cold_evaluator.v1",
        "module_path": MODULE_PATH.as_posix(),
        "module_sha256": file_digest(REPO_ROOT / MODULE_PATH),
        "python": platform.python_version(),
        "numeric_substrate": "Python standard library math, struct, and random.Random",
        "exact_api_path": (
            "carnot.experiment_6766_thermalizer_independent_trajectory_audit.enumerate_path_law"
        ),
        "sampler_api_path": (
            "carnot.experiment_6766_thermalizer_independent_trajectory_audit.sample_trajectory_tv"
        ),
        "imports_exp6751": False,
        "physical_hardware_used": False,
    }


PRINCIPLES: dict[str, str] = {
    "experiment_id": "A stable identifier binds the result to Exp6766.",
    "schema_version": "A schema version exposes future validation changes.",
    "run_date": "The planning date fixes the audit boundary.",
    "spec_refs": "Requirement identifiers connect evidence to its contract.",
    "status": "A terminal state separates completed and owned blocked work.",
    "field_principles": "Every top-level field explains why it exists.",
    "inference_substrate": "The substrate names a local cold audit and excludes a physical TSU.",
    "duration_s": "A monotonic clock records real local audit time without padding.",
    "random_seed": "Frozen source and sampler seeds make both evaluator paths replayable.",
    "reproducibility_checksum": "A digest binds source, code, rows, and reductions.",
    "source_artifact_sha256": "The source digest prevents silent Exp6751 replacement.",
    "preconditions_checked": "Parse, receipt, grid, bound, and isolation checks run first.",
    "compiler_provenance": "Compiler identity remains separate from evaluator authority.",
    "evaluator_provenance": "Exact and sampled API paths identify the cold authority.",
    "dependency_graph_receipt": "Imports, hashes, and shared callables expose code coupling.",
    "evaluator_distinct": "A bare boolean states whether compiler code enters evaluation.",
    "rows": "Each source unit has one exact row and one direct-sampler row.",
    "conditional_kl_by_factor": "Only exact audit rows supply conditional KL aggregates.",
    "trajectory_tv_by_depth": "Only exact audit rows supply trajectory-TV aggregates.",
    "paired_trajectory_deltas": "Matched rows supply effect sizes and paired intervals.",
    "direct_sampler_crosscheck": "Seeded samples test the exact result through another path.",
    "source_metric_crosscheck": "Cold exact rows expose deltas from every source metric.",
    "normalization_mismatches": "Non-unit laws cannot support compiler-fidelity claims.",
    "topology_mismatches": "Topology drift changes the compiler problem under test.",
    "precision_mismatches": "Numeric-format drift changes the compiler problem under test.",
    "independent_trajectory_audit_completed": "True requires all planned attributable rows.",
    "claim_boundary": "The boundary forbids hardware and performance interpretations.",
    "gate_check_summary": "Blocked work records each failed check and observed value.",
    "verifier_is_oracle": "Row circularity prevents exact-objective fitting from positive credit.",
    "verdict_class": "A closed class preserves circular, null, partial, and blocked evidence.",
    "honest_verdict": "A terminal prefix states the result and its main limitation.",
}


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind reproducible inputs and outputs while excluding measured duration."""

    keys = (
        "source_artifact_sha256",
        "random_seed",
        "compiler_provenance",
        "evaluator_provenance",
        "dependency_graph_receipt",
        "evaluator_distinct",
        "rows",
        "conditional_kl_by_factor",
        "trajectory_tv_by_depth",
        "paired_trajectory_deltas",
        "direct_sampler_crosscheck",
        "source_metric_crosscheck",
        "normalization_mismatches",
        "topology_mismatches",
        "precision_mismatches",
        "independent_trajectory_audit_completed",
        "verifier_is_oracle",
        "verdict_class",
    )
    return json_digest({key: artifact[key] for key in keys})


def _base_artifact(
    source: Mapping[str, Any],
    *,
    source_sha256: str,
    run_date: str,
    duration_s: float,
    samples_per_row: int,
) -> JsonDict:
    """Create the complete schema before selecting a terminal result."""

    dependency = dependency_graph_receipt(source)
    evaluator_distinct = bool(
        not dependency["evaluator_imports_compiler_module"]
        and not dependency["compiler_imports_evaluator_module"]
        and not dependency["shared_callable_objects"]
        and not dependency["same_module_sha256"]
        and not dependency["forbidden_source_calls_found"]
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "spec_refs": list(SPEC_REFS),
        "status": "in_progress",
        "field_principles": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _rounded(duration_s),
        "random_seed": {
            "source_seed_receipt": deepcopy(source.get("random_seed", {})),
            "direct_sampler_seed_scheme": "sha256(source_row_id:trajectory_seed:direct_sampler)",
            "samples_per_row": samples_per_row,
        },
        "reproducibility_checksum": "pending",
        "source_artifact_sha256": source_sha256,
        "preconditions_checked": [],
        "compiler_provenance": _compiler_provenance(source, source_sha256),
        "evaluator_provenance": _evaluator_provenance(),
        "dependency_graph_receipt": dependency,
        "evaluator_distinct": evaluator_distinct,
        "rows": [],
        "conditional_kl_by_factor": [],
        "trajectory_tv_by_depth": [],
        "paired_trajectory_deltas": [],
        "direct_sampler_crosscheck": {
            "evaluator_path": "direct_sampler",
            "planned_row_count": 192,
            "observed_row_count": 0,
            "passed": False,
        },
        "source_metric_crosscheck": {
            "exact_row_count": 0,
            "maximum_absolute_conditional_kl_delta": None,
            "maximum_absolute_trajectory_tv_delta": None,
            "target_trajectory_receipt_match_count": 0,
            "compiled_trajectory_receipt_match_count": 0,
            "passed": False,
        },
        "normalization_mismatches": [],
        "topology_mismatches": [],
        "precision_mismatches": [],
        "independent_trajectory_audit_completed": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "gate_check_summary": [],
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "complete_partial: cold audit construction did not finish",
    }


def _finish_artifact(artifact: JsonDict) -> JsonDict:
    """Add the complete principle map and reproducibility digest last."""

    artifact["field_principles"] = dict(PRINCIPLES)
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    return artifact


def _source_crosscheck(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize source deltas only after every cold exact row exists."""

    exact = [row for row in rows if row["evaluator_path"] == "exact_enumerator"]
    kl_delta = max(abs(float(row["source_conditional_kl_delta"])) for row in exact)
    tv_delta = max(abs(float(row["source_trajectory_tv_delta"])) for row in exact)
    target_matches = sum(bool(row["target_trajectory_receipt_matches"]) for row in exact)
    compiled_matches = sum(bool(row["compiled_trajectory_receipt_matches"]) for row in exact)
    return {
        "exact_row_count": len(exact),
        "maximum_absolute_conditional_kl_delta": _rounded(kl_delta),
        "maximum_absolute_trajectory_tv_delta": _rounded(tv_delta),
        "target_trajectory_receipt_match_count": target_matches,
        "compiled_trajectory_receipt_match_count": compiled_matches,
        "compiled_trajectory_receipt_mismatch_count": len(exact) - compiled_matches,
        "compiled_trajectory_bit_identity_required": False,
        "compiled_trajectory_receipt_note": (
            "An independent arithmetic order can change binary64 path bytes while KL and TV "
            "remain within the frozen numeric tolerance. The mismatch is retained, not hidden."
        ),
        "passed": (
            len(exact) == 192
            and kl_delta <= METRIC_TOLERANCE
            and tv_delta <= METRIC_TOLERANCE
            and target_matches == 192
        ),
    }


def _blocked_artifact(
    source: Mapping[str, Any],
    *,
    source_sha256: str,
    run_date: str,
    duration_s: float,
    samples_per_row: int,
    checks: Sequence[Mapping[str, Any]],
    topology_mismatches: Sequence[Mapping[str, Any]] = (),
    precision_mismatches: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Keep the full schema when a cold-audit precondition fails."""

    artifact = _base_artifact(
        source,
        source_sha256=source_sha256,
        run_date=run_date,
        duration_s=duration_s,
        samples_per_row=samples_per_row,
    )
    artifact["status"] = "complete_blocked_thermalizer_audit"
    artifact["preconditions_checked"] = [dict(row) for row in checks]
    artifact["gate_check_summary"] = [dict(row) for row in checks]
    artifact["topology_mismatches"] = [dict(row) for row in topology_mismatches]
    artifact["precision_mismatches"] = [dict(row) for row in precision_mismatches]
    artifact["verdict_class"] = "blocked"
    artifact["honest_verdict"] = (
        "complete_blocked_thermalizer_audit: one or more source, bound, receipt, or "
        "isolation checks failed"
    )
    return _finish_artifact(artifact)


def build_artifact(
    source: Mapping[str, Any],
    *,
    run_date: str,
    duration_s: float,
    samples_per_row: int = DEFAULT_SAMPLES_PER_ROW,
    source_sha256: str | None = None,
) -> JsonDict:
    """Build one terminal cold-audit artifact from serialized source evidence."""

    source_hash = source_sha256 or file_digest(REPO_ROOT / SOURCE_PATH)
    checked = check_source_preconditions(source)
    if checked["gate_check_summary"]:
        return _blocked_artifact(
            source,
            source_sha256=source_hash,
            run_date=run_date,
            duration_s=duration_s,
            samples_per_row=samples_per_row,
            checks=checked["gate_check_summary"],
            topology_mismatches=checked["topology_mismatches"],
            precision_mismatches=checked["precision_mismatches"],
        )
    artifact = _base_artifact(
        source,
        source_sha256=source_hash,
        run_date=run_date,
        duration_s=duration_s,
        samples_per_row=samples_per_row,
    )
    artifact["preconditions_checked"] = [
        {
            "check": "source_artifact_parse_and_receipts",
            "available": True,
            "observed_source_row_count": len(source["rows"]),
            "maximum_planned_path_count": checked["maximum_planned_path_count"],
        },
        {
            "check": "independent_evaluator_isolation",
            "available": artifact["evaluator_distinct"],
            "observed_import_edge": artifact["dependency_graph_receipt"][
                "evaluator_to_compiler_dependency_edge"
            ],
        },
    ]
    try:
        artifact["rows"] = evaluate_source_rows(source, samples_per_row=samples_per_row)
    except (AuditInputError, AuditBoundError, KeyError, TypeError) as exc:
        return _blocked_artifact(
            source,
            source_sha256=source_hash,
            run_date=run_date,
            duration_s=duration_s,
            samples_per_row=samples_per_row,
            checks=[
                _gate(
                    "independent_evaluation",
                    {"completed": True},
                    {"completed": False, "reason": f"{exc.__class__.__name__}: {exc}"},
                )
            ],
        )
    reduced = reduce_audit_rows(artifact["rows"])
    artifact.update(reduced)
    artifact["source_metric_crosscheck"] = _source_crosscheck(artifact["rows"])
    artifact["normalization_mismatches"] = [
        {
            "row_id": row["row_id"],
            "maximum_normalization_error": row["maximum_normalization_error"],
        }
        for row in artifact["rows"]
        if float(row["maximum_normalization_error"]) > NORMALIZATION_TOLERANCE
    ]
    artifact["topology_mismatches"] = checked["topology_mismatches"]
    artifact["precision_mismatches"] = checked["precision_mismatches"]
    artifact["verifier_is_oracle"] = any(
        bool(row["mechanism_consumes_evaluator_outcome"]) for row in artifact["rows"]
    )
    exact_count = sum(row["evaluator_path"] == "exact_enumerator" for row in artifact["rows"])
    sample_count = sum(row["evaluator_path"] == "direct_sampler" for row in artifact["rows"])
    completed = bool(
        artifact["evaluator_distinct"]
        and exact_count == 192
        and sample_count == 192
        and artifact["direct_sampler_crosscheck"]["passed"]
        and artifact["source_metric_crosscheck"]["passed"]
        and not artifact["normalization_mismatches"]
        and not artifact["topology_mismatches"]
        and not artifact["precision_mismatches"]
    )
    artifact["independent_trajectory_audit_completed"] = completed
    context_delta = next(
        row
        for row in artifact["paired_trajectory_deltas"]
        if row["method"] == "context_matched" and row["depth"] == "all"
    )
    positive = bool(
        completed
        and context_delta["mean_independent_minus_method_tv"] > 0.0
        and context_delta["ci95_low"] > 0.0
        and context_delta["interval_excludes_zero"]
    )
    if not completed:
        artifact["status"] = "complete_partial"
        artifact["verdict_class"] = "partial"
        artifact["honest_verdict"] = (
            "complete_partial: cold rows exist but one completion cross-check failed"
        )
    elif positive and artifact["verifier_is_oracle"]:
        artifact["status"] = "complete_circular_positive"
        artifact["verdict_class"] = "circular_positive"
        artifact["honest_verdict"] = (
            "complete: the independent evaluator reproduced the context-matching trajectory "
            "reduction; trajectory refinement remains exact-objective circular; simulator only"
        )
    elif positive:
        artifact["status"] = "complete_positive"
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = (
            "complete: an independent held-out evaluator reproduced the trajectory reduction; "
            "simulator only"
        )
    else:
        artifact["status"] = "complete_null"
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = (
            "complete: the independent audit did not reproduce a paired trajectory reduction"
        )
    return _finish_artifact(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute schema, rows, reductions, circularity, and terminal state."""

    if set(PRINCIPLES) - set(artifact):
        return ["required_fields_missing"]
    errors = []
    if set(artifact["field_principles"]) != set(artifact):
        errors.append("field_principles_mismatch")
    duration = artifact["duration_s"]
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0.0:
        errors.append("duration_invalid")
    if artifact["claim_boundary"] != CLAIM_BOUNDARY:
        errors.append("claim_boundary_mismatch")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact["dependency_graph_receipt"].get("receipt_sha256") != receipt_digest(
        artifact["dependency_graph_receipt"]
    ):
        errors.append("dependency_receipt_mismatch")
    if artifact["reproducibility_checksum"] != _reproducibility_checksum(artifact):
        errors.append("checksum_mismatch")

    blocked = artifact["status"] == "complete_blocked_thermalizer_audit"
    if blocked:
        if (
            artifact["verdict_class"] != "blocked"
            or artifact["independent_trajectory_audit_completed"] is not False
            or artifact["rows"]
            or not artifact["gate_check_summary"]
            or not str(artifact["honest_verdict"]).startswith("complete_blocked_thermalizer_audit")
        ):
            errors.append("blocked_terminal_state_mismatch")
        return errors

    rows = artifact["rows"]
    source_counts = Counter(row.get("source_row_id") for row in rows)
    evaluator_sets: dict[Any, set[Any]] = defaultdict(set)
    for row in rows:
        evaluator_sets[row.get("source_row_id")].add(row.get("evaluator_path"))
    if (
        len(rows) != 384
        or len(source_counts) != 192
        or any(count != 2 for count in source_counts.values())
        or any(paths != set(EVALUATOR_PATHS) for paths in evaluator_sets.values())
    ):
        errors.append("row_grid_mismatch")
    if any(row.get("row_sha256") != audit_row_digest(row) for row in rows):
        errors.append("row_hash_mismatch")
    reduced = reduce_audit_rows(rows)
    if any(artifact[key] != reduced[key] for key in reduced):
        errors.append("aggregate_mismatch")
    dependency = artifact["dependency_graph_receipt"]
    expected_distinct = bool(
        not dependency["evaluator_imports_compiler_module"]
        and not dependency["compiler_imports_evaluator_module"]
        and not dependency["shared_callable_objects"]
        and not dependency["same_module_sha256"]
        and not dependency["forbidden_source_calls_found"]
    )
    if artifact["evaluator_distinct"] is not expected_distinct:
        errors.append("distinctness_mismatch")
    expected_oracle = any(bool(row["mechanism_consumes_evaluator_outcome"]) for row in rows)
    if artifact["verifier_is_oracle"] is not expected_oracle:
        errors.append("oracle_mismatch")
    if artifact["independent_trajectory_audit_completed"]:
        context_delta = next(
            row
            for row in artifact["paired_trajectory_deltas"]
            if row["method"] == "context_matched" and row["depth"] == "all"
        )
        positive = context_delta["ci95_low"] > 0.0
        expected_class = (
            "circular_positive"
            if positive and expected_oracle
            else ("positive" if positive else "null")
        )
        if artifact["verdict_class"] != expected_class:
            errors.append("verdict_class_mismatch")
    return errors


def _write_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    """Replace the deliverable only after one complete JSON value exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(
    *,
    source_path: Path | str = SOURCE_PATH,
    output_path: Path | str = RESULT_PATH,
    run_date: str,
    samples_per_row: int = DEFAULT_SAMPLES_PER_ROW,
) -> JsonDict:
    """Read frozen evidence, build a terminal audit, validate it, and write once."""

    started = time.perf_counter()
    source_file = Path(source_path)
    source_sha256 = file_digest(source_file)
    try:
        loaded = json.loads(source_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise AuditInputError("source artifact must be a JSON object")
    except (OSError, json.JSONDecodeError, AuditInputError) as exc:
        artifact = _blocked_artifact(
            {},
            source_sha256=source_sha256,
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            samples_per_row=samples_per_row,
            checks=[
                _gate(
                    "source_artifact_parse",
                    {"parsed": True},
                    {"parsed": False, "reason": f"{exc.__class__.__name__}: {exc}"},
                )
            ],
        )
    else:
        artifact = build_artifact(
            loaded,
            run_date=run_date,
            duration_s=0.0,
            samples_per_row=samples_per_row,
            source_sha256=source_sha256,
        )
        artifact["duration_s"] = _rounded(time.perf_counter() - started)
        artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    validation_errors = validate_artifact(artifact)
    if validation_errors:
        raise AuditInputError(f"artifact validation failed: {validation_errors}")
    _write_atomic(Path(output_path), artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dated local cold audit from the repository command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--date", required=True)
    parser.add_argument("--samples-per-row", type=int, default=DEFAULT_SAMPLES_PER_ROW)
    args = parser.parse_args(argv)
    artifact = run(
        source_path=args.source,
        output_path=args.output,
        run_date=args.date,
        samples_per_row=args.samples_per_row,
    )
    print(
        canonical_json_text(
            {
                "output": str(args.output),
                "status": artifact["status"],
                "independent_trajectory_audit_completed": artifact[
                    "independent_trajectory_audit_completed"
                ],
                "verdict_class": artifact["verdict_class"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
