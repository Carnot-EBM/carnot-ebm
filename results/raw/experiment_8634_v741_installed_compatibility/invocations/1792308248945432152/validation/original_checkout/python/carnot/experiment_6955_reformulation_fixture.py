"""Build a deterministic exact fixture for optimization reformulations.

Spec refs: REQ-VERIFY-6955 and SCENARIO-VERIFY-6955-*.

The fixture separates a proposed semantic mapping from exact authority. Z3
searches symbolic counterexamples. A separate Python engine enumerates every
point in each finite universe. Their agreement certifies only this local
fixture; it does not measure an LLM or a learned verifier.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import permutations, product
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

try:
    import z3
except ImportError:  # pragma: no cover - the blocked-artifact path handles this environment.
    z3 = None  # type: ignore[assignment]


JsonDict = dict[str, Any]
Z3Prover = Callable[[Mapping[str, Any]], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
RESULT_PATH = Path("results/experiment_6955_reformulation_fixture.json")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_6955_reformulation_fixture_corpus.json")
MODULE_PATH = Path("python/carnot/experiment_6955_reformulation_fixture.py")
TEST_PATH = Path("tests/python/test_experiment_6955_reformulation_fixture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6955_reformulation_fixture.py")
CONSTRAINT_INTERFACE_PATHS = (
    Path("python/carnot/verify/constraint.py"),
    Path("python/carnot/verify/python_types.py"),
    Path("python/carnot/verify/__init__.py"),
)

INFERENCE_SUBSTRATE = "deterministic_z3_and_bounded_enumeration_fixture_no_llm"
MAPPING_SCHEMA_VERSION = "carnot.reformulation_mapping.v1"
FORMULATION_SCHEMA_VERSION = "carnot.bounded_optimization_formulation.v1"
CHECKPOINT_SCHEMA_VERSION = "carnot.reformulation_fixture_checkpoint.v1"
RANDOM_SEED = 695520260903
EXPECTED_ROW_COUNT = 120
MAX_UNIVERSE_SIZE = 7
Z3_TIMEOUT_MS = 2_000

FAMILIES = (
    "bounded_integer_linear",
    "boolean_cardinality",
    "bounded_piecewise_linear",
)
SPLITS = ("train", "calibration", "held_out", "prospective_family")
TEMPLATE_SPLITS = {
    0: "train",
    1: "train",
    2: "calibration",
    3: "held_out",
    4: "prospective_family",
}
HARD_NEGATIVE_EDITS = (
    "missing_bound",
    "strictness_change",
    "infeasible_domain",
    "duplicate_variable",
    "objective_scale",
    "objective_sign",
)
OBJECTIVE_SCALES = ("1", "2", "1/2", "-1", "-2")

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "pair_rows",
    "formulation_rows",
    "mapping_rows",
    "family_rows",
    "label_rows",
    "hard_negative_rows",
    "feasibility_witness_rows",
    "objective_order_rows",
    "z3_rows",
    "enumeration_rows",
    "authority_agreement_rows",
    "split_rows",
    "isomorphism_rows",
    "fresh_process_replay_rows",
    "corpus_checkpoint_path",
    "random_seed",
    "reproducibility_checksum",
    "reformulation_fixture_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "A reason per required field makes the evidence contract auditable.",
    "preconditions_checked": "Fail-closed checks prevent partial local tools from creating labels.",
    "inference_substrate": "The fixed declaration proves that no LLM supplied fixture authority.",
    "duration_s": "Measured wall time proves that construction and replay executed.",
    "source_artifact_hashes": "Source hashes bind the result to its exact contract and engines.",
    "rows": "Complete per-pair rows let consistency checks recompute the verdict.",
    "pair_rows": "One row per pair preserves labels, identities, witnesses, and edits.",
    "formulation_rows": "Source and target bytes make every optimization problem replayable.",
    "mapping_rows": "Versioned mapping bytes expose every accepted or rejected correspondence.",
    "family_rows": "Family counts prove that all three bounded problem classes are present.",
    "label_rows": "Label counts prove the precommitted 72 to 48 class balance.",
    "hard_negative_rows": "One named edit per negative makes the failure mechanism falsifiable.",
    "feasibility_witness_rows": "Exact assignments show domain preservation or its failure.",
    "objective_order_rows": "Assignment pairs expose preserved or reversed optimization order.",
    "z3_rows": "SMT statuses record the independent symbolic counterexample search.",
    "enumeration_rows": "Finite counts record the separate exhaustive semantic calculation.",
    "authority_agreement_rows": "Per-pair parity prevents aggregates from hiding disagreement.",
    "split_rows": "Template and identity rows expose any normalized cross-split leakage.",
    "isomorphism_rows": "Exhaustive name canonicalization detects small-variable isomorphs.",
    "fresh_process_replay_rows": "Child-process receipts prove reconstruction from serialized data.",
    "corpus_checkpoint_path": "The durable checkpoint is the sole input to fresh-process replay.",
    "random_seed": "One fixed seed controls every generated name and ordering decision.",
    "reproducibility_checksum": "A timing-free digest detects scientific-content drift.",
    "reformulation_fixture_ready_score": "A binary row-derived gate opens only after all checks pass.",
    "gate_check_summary": "Expected and observed values make every failure actionable.",
    "verifier_is_oracle": "True limits exact authority to conformance of this synthetic fixture.",
    "verdict_class": "The circular class prevents fixture conformance from becoming model evidence.",
    "honest_verdict": "A terminal prefix lets automation classify the completed run safely.",
}

_MAPPING_KEYS = {"schema_version", "variables", "domain_clauses", "objective", "claimed_relation"}
_VARIABLE_MAPPING_KEYS = {"source", "target", "scale", "offset"}
_DOMAIN_CLAUSE_KEYS = {
    "source",
    "target",
    "source_lower",
    "source_upper",
    "target_lower",
    "target_upper",
}
_OBJECTIVE_MAPPING_KEYS = {
    "source_direction",
    "target_direction",
    "scale",
    "offset",
}
_FORMULATION_KEYS = {"schema_version", "variables", "constraints", "objective"}
_VARIABLE_KEYS = {"name", "kind", "universe", "domain"}
_DOMAIN_KEYS = {"lower", "upper"}
_CONSTRAINT_KEYS = {"terms", "op", "rhs"}
_OBJECTIVE_KEYS = {"direction", "expression"}


class MappingSchemaError(ValueError):
    """Report one deterministic mapping-schema rejection code."""


class FormulationSchemaError(ValueError):
    """Report one deterministic bounded-formulation rejection code."""


def canonical_json(value: Any) -> bytes:
    """Return stable JSON bytes used for all scientific identities."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return a labeled SHA-256 digest."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_path(path: Path) -> str | None:
    """Hash one file while preserving a missing input as an explicit null."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def as_fraction(value: Any) -> Fraction:
    """Parse one exact rational and reject binary floating-point inputs."""

    if isinstance(value, (bool, float)):
        raise ValueError(f"non_exact_rational:{value!r}")
    try:
        return Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid_rational:{value!r}") from exc


def fraction_text(value: Any) -> str:
    """Render one rational in its unique numerator or numerator/denominator form."""

    exact = as_fraction(value)
    return (
        str(exact.numerator) if exact.denominator == 1 else f"{exact.numerator}/{exact.denominator}"
    )


def _require_keys(actual: Mapping[str, Any], expected: set[str], reason: str) -> None:
    """Reject missing and extra JSON keys in a stable lexical order."""

    missing = sorted(expected - set(actual))
    extra = sorted(set(actual) - expected)
    if missing or extra:
        raise MappingSchemaError(f"{reason}:missing={missing}:extra={extra}")


def _variable_names(formulation: Mapping[str, Any]) -> list[str]:
    """Read declared variable names without trusting their uniqueness."""

    return [str(row.get("name")) for row in formulation.get("variables", [])]


def validate_formulation(formulation: Mapping[str, Any]) -> JsonDict:
    """Validate the finite formulation syntax required by both exact engines."""

    if set(formulation) != _FORMULATION_KEYS:
        raise FormulationSchemaError("formulation_keys")
    if formulation["schema_version"] != FORMULATION_SCHEMA_VERSION:
        raise FormulationSchemaError("formulation_schema_version")
    variables = formulation["variables"]
    if not isinstance(variables, list) or not variables:
        raise FormulationSchemaError("variables_nonempty")
    names: list[str] = []
    canonical_variables: list[JsonDict] = []
    for variable in variables:
        if not isinstance(variable, Mapping) or set(variable) != _VARIABLE_KEYS:
            raise FormulationSchemaError("variable_keys")
        name = str(variable["name"])
        kind = str(variable["kind"])
        if kind not in {"integer", "boolean"}:
            raise FormulationSchemaError("variable_kind")
        universe = list(variable["universe"])
        if not universe or len(universe) > MAX_UNIVERSE_SIZE or len(universe) != len(set(universe)):
            raise FormulationSchemaError("finite_unique_universe")
        if kind == "boolean" and universe != [False, True]:
            raise FormulationSchemaError("boolean_universe")
        if kind == "integer" and any(
            isinstance(item, bool) or not isinstance(item, int) for item in universe
        ):
            raise FormulationSchemaError("integer_universe")
        domain = variable["domain"]
        if not isinstance(domain, Mapping) or set(domain) != _DOMAIN_KEYS:
            raise FormulationSchemaError("domain_keys")
        lower = domain["lower"]
        upper = domain["upper"]
        if kind == "boolean" and (lower is not None or upper is not None):
            raise FormulationSchemaError("boolean_domain_bounds")
        for bound in (lower, upper):
            if bound is not None:
                as_fraction(bound)
        names.append(name)
        canonical_variables.append(
            {
                "name": name,
                "kind": kind,
                "universe": universe,
                "domain": {
                    "lower": None if lower is None else fraction_text(lower),
                    "upper": None if upper is None else fraction_text(upper),
                },
            }
        )
    if len(names) != len(set(names)):
        raise FormulationSchemaError("duplicate_formulation_variable")
    name_set = set(names)
    canonical_constraints: list[JsonDict] = []
    for constraint in formulation["constraints"]:
        if not isinstance(constraint, Mapping) or set(constraint) != _CONSTRAINT_KEYS:
            raise FormulationSchemaError("constraint_keys")
        if constraint["op"] not in {"<=", ">=", "==", "<", ">"}:
            raise FormulationSchemaError("constraint_operator")
        if not isinstance(constraint["terms"], Mapping) or not set(constraint["terms"]) <= name_set:
            raise FormulationSchemaError("constraint_variables")
        canonical_constraints.append(
            {
                "terms": {
                    name: fraction_text(coefficient)
                    for name, coefficient in sorted(constraint["terms"].items())
                },
                "op": str(constraint["op"]),
                "rhs": fraction_text(constraint["rhs"]),
            }
        )
    objective = formulation["objective"]
    if not isinstance(objective, Mapping) or set(objective) != _OBJECTIVE_KEYS:
        raise FormulationSchemaError("objective_keys")
    if objective["direction"] not in {"min", "max"}:
        raise FormulationSchemaError("objective_direction")
    expression = _canonical_expression(objective["expression"], name_set)
    return {
        "schema_version": FORMULATION_SCHEMA_VERSION,
        "variables": canonical_variables,
        "constraints": canonical_constraints,
        "objective": {"direction": objective["direction"], "expression": expression},
    }


def _canonical_linear(expression: Mapping[str, Any], names: set[str]) -> JsonDict:
    """Canonicalize one affine expression used by linear and piecewise objectives."""

    if set(expression) != {"terms", "constant"}:
        raise FormulationSchemaError("linear_expression_keys")
    if not isinstance(expression["terms"], Mapping) or not set(expression["terms"]) <= names:
        raise FormulationSchemaError("objective_variables")
    return {
        "terms": {
            name: fraction_text(coefficient)
            for name, coefficient in sorted(expression["terms"].items())
        },
        "constant": fraction_text(expression["constant"]),
    }


def _canonical_expression(expression: Any, names: set[str]) -> JsonDict:
    """Canonicalize a linear expression or an exact max/min of affine pieces."""

    if not isinstance(expression, Mapping) or "kind" not in expression:
        raise FormulationSchemaError("objective_expression")
    if expression["kind"] == "linear":
        if set(expression) != {"kind", "terms", "constant"}:
            raise FormulationSchemaError("linear_objective_keys")
        linear = _canonical_linear(
            {"terms": expression["terms"], "constant": expression["constant"]}, names
        )
        return {"kind": "linear", **linear}
    if expression["kind"] == "piecewise_linear":
        if set(expression) != {"kind", "aggregation", "pieces"}:
            raise FormulationSchemaError("piecewise_objective_keys")
        if expression["aggregation"] not in {"max", "min"}:
            raise FormulationSchemaError("piecewise_aggregation")
        pieces = expression["pieces"]
        if not isinstance(pieces, list) or len(pieces) < 2:
            raise FormulationSchemaError("piecewise_pieces")
        return {
            "kind": "piecewise_linear",
            "aggregation": expression["aggregation"],
            "pieces": [_canonical_linear(piece, names) for piece in pieces],
        }
    raise FormulationSchemaError("objective_expression_kind")


def canonical_mapping(
    mapping: Mapping[str, Any],
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> JsonDict:
    """Validate and canonicalize the strict v1 source-to-target mapping."""

    _require_keys(mapping, _MAPPING_KEYS, "mapping_keys")
    if mapping["schema_version"] != MAPPING_SCHEMA_VERSION:
        raise MappingSchemaError("mapping_schema_version")
    variable_rows = mapping["variables"]
    if not isinstance(variable_rows, list):
        raise MappingSchemaError("variable_mappings_list")
    canonical_variables: list[JsonDict] = []
    for row in variable_rows:
        if not isinstance(row, Mapping):
            raise MappingSchemaError("variable_mapping_object")
        _require_keys(row, _VARIABLE_MAPPING_KEYS, "variable_mapping_keys")
        scale = as_fraction(row["scale"])
        if scale == 0:
            raise MappingSchemaError("zero_variable_scale")
        canonical_variables.append(
            {
                "source": str(row["source"]),
                "target": str(row["target"]),
                "scale": fraction_text(scale),
                "offset": fraction_text(row["offset"]),
            }
        )
    sources = [row["source"] for row in canonical_variables]
    targets = [row["target"] for row in canonical_variables]
    if len(sources) != len(set(sources)):
        raise MappingSchemaError("duplicate_source_variable")
    if len(targets) != len(set(targets)):
        raise MappingSchemaError("duplicate_target_variable")
    source_names = set(_variable_names(source))
    target_names = set(_variable_names(target))
    if set(sources) != source_names:
        raise MappingSchemaError("source_variable_coverage")
    if set(targets) != target_names:
        raise MappingSchemaError("target_variable_coverage")
    domain_rows = mapping["domain_clauses"]
    if not isinstance(domain_rows, list):
        raise MappingSchemaError("domain_clauses_list")
    canonical_domains: list[JsonDict] = []
    for row in domain_rows:
        if not isinstance(row, Mapping):
            raise MappingSchemaError("domain_clause_object")
        _require_keys(row, _DOMAIN_CLAUSE_KEYS, "domain_clause_keys")
        canonical_domains.append(
            {
                "source": str(row["source"]),
                "target": str(row["target"]),
                "source_lower": _optional_fraction_text(row["source_lower"]),
                "source_upper": _optional_fraction_text(row["source_upper"]),
                "target_lower": _optional_fraction_text(row["target_lower"]),
                "target_upper": _optional_fraction_text(row["target_upper"]),
            }
        )
    if {(row["source"], row["target"]) for row in canonical_domains} != {
        (row["source"], row["target"]) for row in canonical_variables
    }:
        raise MappingSchemaError("domain_clause_coverage")
    objective = mapping["objective"]
    if not isinstance(objective, Mapping):
        raise MappingSchemaError("objective_mapping_object")
    _require_keys(objective, _OBJECTIVE_MAPPING_KEYS, "objective_mapping_keys")
    if objective["source_direction"] not in {"min", "max"} or objective["target_direction"] not in {
        "min",
        "max",
    }:
        raise MappingSchemaError("mapping_objective_direction")
    objective_scale = as_fraction(objective["scale"])
    if objective_scale == 0:
        raise MappingSchemaError("zero_objective_scale")
    if mapping["claimed_relation"] not in {"equivalent", "non_equivalent"}:
        raise MappingSchemaError("claimed_relation")
    return {
        "schema_version": MAPPING_SCHEMA_VERSION,
        "variables": sorted(canonical_variables, key=lambda row: row["source"]),
        "domain_clauses": sorted(canonical_domains, key=lambda row: row["source"]),
        "objective": {
            "source_direction": objective["source_direction"],
            "target_direction": objective["target_direction"],
            "scale": fraction_text(objective_scale),
            "offset": fraction_text(objective["offset"]),
        },
        "claimed_relation": mapping["claimed_relation"],
    }


def _optional_fraction_text(value: Any) -> str | None:
    """Canonicalize an optional exact domain bound."""

    return None if value is None else fraction_text(value)


def _linear_expression(terms: Mapping[str, Any], constant: Any = 0) -> JsonDict:
    """Create one JSON-safe exact linear objective expression."""

    return {
        "kind": "linear",
        "terms": {name: fraction_text(value) for name, value in terms.items()},
        "constant": fraction_text(constant),
    }


def _source_formulation(family: str, template: int, ordinal: int) -> JsonDict:
    """Build one bounded source whose boundary and objective both have headroom."""

    if family == "boolean_cardinality":
        count = 3 + template % 2
        names = [f"b{index}" for index in range(count)]
        cap = 1 + (template + ordinal) % 2
        variables = [
            {
                "name": name,
                "kind": "boolean",
                "universe": [False, True],
                "domain": {"lower": None, "upper": None},
            }
            for name in names
        ]
        constraints = [
            {"terms": {name: "1" for name in names}, "op": "<=", "rhs": str(cap)},
            {"terms": {name: "1" for name in names}, "op": "<=", "rhs": str(count + template)},
        ]
        objective = {
            "direction": "min",
            "expression": _linear_expression(
                {name: 1 + (index + ordinal + template) % 3 for index, name in enumerate(names)}
            ),
        }
    else:
        names = ["x0", "x1"]
        upper = [2 + template % 2, 2 + ordinal % 2]
        variables = [
            {
                "name": name,
                "kind": "integer",
                "universe": list(range(-1, bound + 2)),
                "domain": {"lower": "0", "upper": str(bound)},
            }
            for name, bound in zip(names, upper, strict=True)
        ]
        cap = 1 + (template + ordinal) % 2
        constraints = [
            {"terms": {"x0": "1", "x1": "1"}, "op": "<=", "rhs": str(cap)},
            {
                "terms": {"x0": "1"},
                "op": "<=",
                "rhs": str(upper[0] + template + 1),
            },
        ]
        if family == "bounded_piecewise_linear":
            objective = {
                "direction": "min",
                "expression": {
                    "kind": "piecewise_linear",
                    "aggregation": "max",
                    "pieces": [
                        {
                            "terms": {"x0": "1", "x1": str(1 + template % 2)},
                            "constant": str(ordinal % 2),
                        },
                        {"terms": {"x0": "-1", "x1": "1"}, "constant": str(template % 3)},
                    ],
                },
            }
        else:
            objective = {
                "direction": "min",
                "expression": _linear_expression(
                    {"x0": 1 + ordinal % 2, "x1": 2 + template % 2}, template % 2
                ),
            }
    return validate_formulation(
        {
            "schema_version": FORMULATION_SCHEMA_VERSION,
            "variables": variables,
            "constraints": constraints,
            "objective": objective,
        }
    )


def _mapping_rows(
    source: Mapping[str, Any], target_names: Sequence[str], template: int, ordinal: int
) -> list[JsonDict]:
    """Choose a bijective affine renaming with small integer offsets."""

    rows: list[JsonDict] = []
    for index, (variable, target_name) in enumerate(
        zip(source["variables"], target_names, strict=True)
    ):
        is_boolean = variable["kind"] == "boolean"
        scale = 1 if is_boolean or (template + ordinal + index) % 2 == 0 else -1
        offset = 0 if is_boolean else ((template + ordinal + index) % 3) - 1
        rows.append(
            {
                "source": variable["name"],
                "target": target_name,
                "scale": str(scale),
                "offset": str(offset),
            }
        )
    return rows


def _mapped_bound(bound: Any, scale: Fraction, offset: Fraction) -> str | None:
    """Map one optional variable bound through an invertible affine transform."""

    return None if bound is None else fraction_text(scale * as_fraction(bound) + offset)


def _transform_linear(
    expression: Mapping[str, Any],
    variable_rows: Sequence[Mapping[str, Any]],
    multiplier: Fraction = Fraction(1),
    add_constant: Fraction = Fraction(0),
) -> JsonDict:
    """Substitute source variables into target names using exact rational algebra."""

    by_source = {row["source"]: row for row in variable_rows}
    terms: dict[str, Fraction] = {}
    constant = as_fraction(expression["constant"])
    for source_name, coefficient_value in expression["terms"].items():
        row = by_source[source_name]
        coefficient = as_fraction(coefficient_value)
        scale = as_fraction(row["scale"])
        offset = as_fraction(row["offset"])
        terms[row["target"]] = multiplier * coefficient / scale
        constant -= coefficient * offset / scale
    constant = multiplier * constant + add_constant
    return {
        "terms": {name: fraction_text(value) for name, value in terms.items()},
        "constant": fraction_text(constant),
    }


def _transform_expression(
    expression: Mapping[str, Any],
    variable_rows: Sequence[Mapping[str, Any]],
    scale: Fraction,
    offset: Fraction,
) -> JsonDict:
    """Transform a linear or piecewise objective while preserving exact values."""

    if expression["kind"] == "linear":
        linear = _transform_linear(expression, variable_rows, scale, offset)
        return {"kind": "linear", **linear}
    pieces = [
        _transform_linear(piece, variable_rows, scale, offset) for piece in expression["pieces"]
    ]
    aggregation = expression["aggregation"]
    if scale < 0:
        aggregation = "min" if aggregation == "max" else "max"
    return {"kind": "piecewise_linear", "aggregation": aggregation, "pieces": pieces}


def _derive_equivalent_target(
    source: Mapping[str, Any],
    variable_rows: Sequence[Mapping[str, Any]],
    objective_scale: Fraction,
    objective_offset: Fraction,
) -> JsonDict:
    """Derive the exact target by symbolic substitution, not by sampled behavior."""

    row_by_source = {row["source"]: row for row in variable_rows}
    target_variables: list[JsonDict] = []
    domain_clauses: list[JsonDict] = []
    for variable in source["variables"]:
        row = row_by_source[variable["name"]]
        scale = as_fraction(row["scale"])
        offset = as_fraction(row["offset"])
        universe = [
            bool(value) if variable["kind"] == "boolean" else int(scale * value + offset)
            for value in variable["universe"]
        ]
        lower = _mapped_bound(variable["domain"]["lower"], scale, offset)
        upper = _mapped_bound(variable["domain"]["upper"], scale, offset)
        if lower is not None and upper is not None and as_fraction(lower) > as_fraction(upper):
            lower, upper = upper, lower
        target_variables.append(
            {
                "name": row["target"],
                "kind": variable["kind"],
                "universe": sorted(universe),
                "domain": {"lower": lower, "upper": upper},
            }
        )
        domain_clauses.append(
            {
                "source": variable["name"],
                "target": row["target"],
                "source_lower": variable["domain"]["lower"],
                "source_upper": variable["domain"]["upper"],
                "target_lower": lower,
                "target_upper": upper,
            }
        )
    target_constraints: list[JsonDict] = []
    for constraint in source["constraints"]:
        transformed = _transform_linear(
            {"terms": constraint["terms"], "constant": "0"}, variable_rows
        )
        target_constraints.append(
            {
                "terms": transformed["terms"],
                "op": constraint["op"],
                "rhs": fraction_text(
                    as_fraction(constraint["rhs"]) - as_fraction(transformed["constant"])
                ),
            }
        )
    source_direction = source["objective"]["direction"]
    target_direction = (
        source_direction if objective_scale > 0 else ("max" if source_direction == "min" else "min")
    )
    target = validate_formulation(
        {
            "schema_version": FORMULATION_SCHEMA_VERSION,
            "variables": target_variables,
            "constraints": target_constraints,
            "objective": {
                "direction": target_direction,
                "expression": _transform_expression(
                    source["objective"]["expression"],
                    variable_rows,
                    objective_scale,
                    objective_offset,
                ),
            },
        }
    )
    return {"target": target, "domain_clauses": domain_clauses}


def _negate_expression(expression: JsonDict) -> JsonDict:
    """Apply the single conceptual edit that negates a complete objective."""

    if expression["kind"] == "linear":
        return {
            "kind": "linear",
            "terms": {
                name: fraction_text(-as_fraction(value))
                for name, value in expression["terms"].items()
            },
            "constant": fraction_text(-as_fraction(expression["constant"])),
        }
    return {
        "kind": "piecewise_linear",
        "aggregation": "min" if expression["aggregation"] == "max" else "max",
        "pieces": [
            {
                "terms": {
                    name: fraction_text(-as_fraction(value))
                    for name, value in piece["terms"].items()
                },
                "constant": fraction_text(-as_fraction(piece["constant"])),
            }
            for piece in expression["pieces"]
        ],
    }


def _change_first_objective_coefficient(expression: JsonDict) -> None:
    """Change one coefficient so the declared objective scale becomes false."""

    linear = expression if expression["kind"] == "linear" else expression["pieces"][0]
    name = sorted(linear["terms"])[0]
    linear["terms"][name] = fraction_text(as_fraction(linear["terms"][name]) + 1)


def _apply_hard_negative(pair: JsonDict, edit: str) -> None:
    """Apply exactly one predeclared semantic edit to an equivalent base pair."""

    target = pair["target"]
    mapping = pair["mapping"]
    if edit == "missing_bound":
        variable = next(row for row in target["variables"] if row["kind"] == "integer")
        variable_mapping = next(
            row for row in mapping["variables"] if row["target"] == variable["name"]
        )
        # A negative affine scale swaps lower and upper bounds. Remove the target
        # clause that came from the source lower bound so x=-1 stays a witness.
        bound = "lower" if as_fraction(variable_mapping["scale"]) > 0 else "upper"
        variable["domain"][bound] = None
    elif edit == "strictness_change":
        target["constraints"][0]["op"] = "<"
    elif edit == "infeasible_domain":
        variable = target["variables"][0]
        if variable["kind"] == "boolean":
            target["constraints"][0]["rhs"] = "-1"
        else:
            variable["domain"]["lower"] = str(max(variable["universe"]) + 1)
    elif edit == "duplicate_variable":
        mapping["variables"][1]["target"] = mapping["variables"][0]["target"]
    elif edit == "objective_scale":
        _change_first_objective_coefficient(target["objective"]["expression"])
    elif edit == "objective_sign":
        target["objective"]["expression"] = _negate_expression(target["objective"]["expression"])
    else:  # pragma: no cover - generation draws only from the frozen edit roster.
        raise ValueError(f"unknown_hard_negative_edit:{edit}")
    mapping["claimed_relation"] = "non_equivalent"


def _negative_positions() -> set[tuple[int, int]]:
    """Return the frozen 16-of-40 negative positions for each family."""

    return {
        (template, ordinal)
        for template in range(5)
        for ordinal in range(8)
        if ordinal >= (4 if template == 4 else 5)
    }


def generate_pairs(seed: int) -> list[JsonDict]:
    """Generate exactly 120 deterministic pairs from 15 frozen templates."""

    rng = random.Random(seed)
    pairs: list[JsonDict] = []
    negative_index = 0
    for family_index, family in enumerate(FAMILIES):
        for template in range(5):
            generator_template = f"{family}.template_{template}"
            for ordinal in range(8):
                source = _source_formulation(family, template, ordinal)
                name_pool = [
                    f"v{family_index}_{template}_{ordinal}_{index}"
                    for index in range(len(source["variables"]))
                ]
                rng.shuffle(name_pool)
                variable_rows = _mapping_rows(source, name_pool, template, ordinal)
                objective_scale = as_fraction(
                    OBJECTIVE_SCALES[(template * 8 + ordinal) % len(OBJECTIVE_SCALES)]
                )
                objective_offset = Fraction((template + ordinal) % 3 - 1)
                derived = _derive_equivalent_target(
                    source, variable_rows, objective_scale, objective_offset
                )
                target = derived["target"]
                mapping: JsonDict = {
                    "schema_version": MAPPING_SCHEMA_VERSION,
                    "variables": variable_rows,
                    "domain_clauses": derived["domain_clauses"],
                    "objective": {
                        "source_direction": source["objective"]["direction"],
                        "target_direction": target["objective"]["direction"],
                        "scale": fraction_text(objective_scale),
                        "offset": fraction_text(objective_offset),
                    },
                    "claimed_relation": "equivalent",
                }
                pair: JsonDict = {
                    "pair_id": f"{family_index}-{template}-{ordinal}",
                    "family": family,
                    "difficulty": "standard",
                    "generator_template": generator_template,
                    "template_index": template,
                    "split": TEMPLATE_SPLITS[template],
                    "expected_label": "equivalent",
                    "hard_negative_edit": None,
                    "hard_negative_edit_count": 0,
                    "source": source,
                    "target": target,
                    "mapping": mapping,
                }
                if (template, ordinal) in _negative_positions():
                    if family == "boolean_cardinality":
                        boolean_edits = (
                            "strictness_change",
                            "infeasible_domain",
                            "duplicate_variable",
                            "objective_scale",
                            "objective_sign",
                        )
                        edit = boolean_edits[negative_index % len(boolean_edits)]
                    else:
                        edit = HARD_NEGATIVE_EDITS[negative_index % len(HARD_NEGATIVE_EDITS)]
                    pair["expected_label"] = "non_equivalent"
                    pair["difficulty"] = "hard"
                    pair["hard_negative_edit"] = edit
                    pair["hard_negative_edit_count"] = 1
                    _apply_hard_negative(pair, edit)
                    negative_index += 1
                pairs.append(pair)
    return pairs


def _assignment_dict(names: Sequence[str], values: Sequence[Any]) -> JsonDict:
    """Create a stable assignment ordered by formulation declaration."""

    return dict(zip(names, values, strict=True))


def _all_assignments(formulation: Mapping[str, Any]) -> list[JsonDict]:
    """Enumerate every point in the declared finite universe."""

    names = [variable["name"] for variable in formulation["variables"]]
    universes = [variable["universe"] for variable in formulation["variables"]]
    return [_assignment_dict(names, values) for values in product(*universes)]


def _compare(left: Fraction, op: str, right: Fraction) -> bool:
    """Evaluate one exact relation without converting to floating point."""

    return {
        "<=": left <= right,
        ">=": left >= right,
        "==": left == right,
        "<": left < right,
        ">": left > right,
    }[op]


def _linear_value(expression: Mapping[str, Any], assignment: Mapping[str, Any]) -> Fraction:
    """Evaluate one affine expression exactly, treating booleans as zero or one."""

    return as_fraction(expression["constant"]) + sum(
        as_fraction(coefficient) * int(assignment[name])
        for name, coefficient in expression["terms"].items()
    )


def objective_value(formulation: Mapping[str, Any], assignment: Mapping[str, Any]) -> Fraction:
    """Evaluate a linear or piecewise-linear objective exactly."""

    expression = formulation["objective"]["expression"]
    if expression["kind"] == "linear":
        return _linear_value(expression, assignment)
    values = [_linear_value(piece, assignment) for piece in expression["pieces"]]
    return max(values) if expression["aggregation"] == "max" else min(values)


def is_feasible(formulation: Mapping[str, Any], assignment: Mapping[str, Any]) -> bool:
    """Check semantic bounds and constraints for one universe assignment."""

    for variable in formulation["variables"]:
        value = int(assignment[variable["name"]])
        lower = variable["domain"]["lower"]
        upper = variable["domain"]["upper"]
        if lower is not None and value < as_fraction(lower):
            return False
        if upper is not None and value > as_fraction(upper):
            return False
    for constraint in formulation["constraints"]:
        left = sum(
            as_fraction(coefficient) * int(assignment[name])
            for name, coefficient in constraint["terms"].items()
        )
        if not _compare(left, constraint["op"], as_fraction(constraint["rhs"])):
            return False
    return True


def _feasible_assignments(formulation: Mapping[str, Any]) -> list[JsonDict]:
    """Return all feasible assignments in deterministic product order."""

    return [
        assignment
        for assignment in _all_assignments(formulation)
        if is_feasible(formulation, assignment)
    ]


def _assignment_key(
    formulation: Mapping[str, Any], assignment: Mapping[str, Any]
) -> tuple[Any, ...]:
    """Convert an assignment to a hashable tuple in declaration order."""

    return tuple(assignment[variable["name"]] for variable in formulation["variables"])


def _map_assignment(assignment: Mapping[str, Any], mapping: Mapping[str, Any]) -> JsonDict:
    """Apply one validated affine mapping to a source assignment."""

    result: JsonDict = {}
    for row in mapping["variables"]:
        value = int(assignment[row["source"]])
        mapped = as_fraction(row["scale"]) * value + as_fraction(row["offset"])
        result[row["target"]] = (
            bool(mapped) if isinstance(assignment[row["source"]], bool) else int(mapped)
        )
    return result


def _order_holds(direction: str, left: Fraction, right: Fraction) -> bool:
    """Interpret objective order according to minimization or maximization."""

    return left <= right if direction == "min" else left >= right


def _json_fraction(value: Fraction) -> str:
    """Serialize an exact witness value without precision loss."""

    return fraction_text(value)


def prove_pair_with_enumerator(pair: Mapping[str, Any]) -> JsonDict:
    """Prove or refute one pair by exhaustive finite-domain enumeration."""

    pair_id = str(pair["pair_id"])
    source = validate_formulation(pair["source"])
    target = validate_formulation(pair["target"])
    source_feasible = _feasible_assignments(source)
    target_feasible = _feasible_assignments(target)
    base_witness = {
        "source": source_feasible[0] if source_feasible else None,
        "target": target_feasible[0] if target_feasible else None,
    }
    try:
        mapping = canonical_mapping(pair["mapping"], source, target)
    except MappingSchemaError as exc:
        return {
            "pair_id": pair_id,
            "engine": "python_exhaustive_bounded_enumerator_v1",
            "status": "schema_rejected",
            "label": "non_equivalent",
            "source_universe_count": len(_all_assignments(source)),
            "target_universe_count": len(_all_assignments(target)),
            "source_feasible_count": len(source_feasible),
            "target_feasible_count": len(target_feasible),
            "domain_preserved": False,
            "objective_affine_preserved": False,
            "objective_order_preserved": False,
            "feasibility_witness": base_witness,
            "objective_order_witness": None,
            "counterexample": {"kind": "mapping_schema", "reason": str(exc)},
        }
    mapped_source = [
        (_map_assignment(assignment, mapping), assignment) for assignment in source_feasible
    ]
    mapped_keys = {
        _assignment_key(target, mapped): source_assignment
        for mapped, source_assignment in mapped_source
    }
    target_keys = {
        _assignment_key(target, assignment): assignment for assignment in target_feasible
    }
    domain_preserved = set(mapped_keys) == set(target_keys)
    counterexample: JsonDict | None = None
    if not domain_preserved:
        source_only = sorted(set(mapped_keys) - set(target_keys))
        target_only = sorted(set(target_keys) - set(mapped_keys))
        if source_only:
            key = source_only[0]
            counterexample = {
                "kind": "domain",
                "direction": "mapped_source_only",
                "source": mapped_keys[key],
                "target": _assignment_dict(
                    [variable["name"] for variable in target["variables"]], key
                ),
            }
        else:
            key = target_only[0]
            counterexample = {
                "kind": "domain",
                "direction": "target_only",
                "source": None,
                "target": target_keys[key],
            }
    scale = as_fraction(mapping["objective"]["scale"])
    offset = as_fraction(mapping["objective"]["offset"])
    common: list[tuple[JsonDict, JsonDict, Fraction, Fraction]] = []
    affine_preserved = True
    for mapped, source_assignment in mapped_source:
        if _assignment_key(target, mapped) not in target_keys:
            continue
        source_value = objective_value(source, source_assignment)
        target_value = objective_value(target, mapped)
        common.append((source_assignment, mapped, source_value, target_value))
        if target_value != scale * source_value + offset:
            affine_preserved = False
            if counterexample is None:
                counterexample = {
                    "kind": "objective_affine",
                    "source": source_assignment,
                    "target": mapped,
                    "source_value": _json_fraction(source_value),
                    "target_value": _json_fraction(target_value),
                    "expected_target_value": _json_fraction(scale * source_value + offset),
                }
            break
    order_preserved = True
    order_witness: JsonDict | None = None
    for left, right in product(common, repeat=2):
        source_order = _order_holds(source["objective"]["direction"], left[2], right[2])
        target_order = _order_holds(target["objective"]["direction"], left[3], right[3])
        witness = {
            "source_left": left[0],
            "source_right": right[0],
            "target_left": left[1],
            "target_right": right[1],
            "source_values": [_json_fraction(left[2]), _json_fraction(right[2])],
            "target_values": [_json_fraction(left[3]), _json_fraction(right[3])],
            "source_order": source_order,
            "target_order": target_order,
        }
        if order_witness is None and left[2] != right[2]:
            order_witness = witness
        if source_order != target_order:
            order_preserved = False
            order_witness = witness
            if counterexample is None:
                counterexample = {"kind": "objective_order", **witness}
            break
    if order_witness is None and common:
        left = common[0]
        order_witness = {
            "source_left": left[0],
            "source_right": left[0],
            "target_left": left[1],
            "target_right": left[1],
            "source_values": [_json_fraction(left[2]), _json_fraction(left[2])],
            "target_values": [_json_fraction(left[3]), _json_fraction(left[3])],
            "source_order": True,
            "target_order": True,
        }
    equivalent = domain_preserved and affine_preserved and order_preserved
    return {
        "pair_id": pair_id,
        "engine": "python_exhaustive_bounded_enumerator_v1",
        "status": "proved" if equivalent else "counterexample",
        "label": "equivalent" if equivalent else "non_equivalent",
        "source_universe_count": len(_all_assignments(source)),
        "target_universe_count": len(_all_assignments(target)),
        "source_feasible_count": len(source_feasible),
        "target_feasible_count": len(target_feasible),
        "domain_preserved": domain_preserved,
        "objective_affine_preserved": affine_preserved,
        "objective_order_preserved": order_preserved,
        "feasibility_witness": base_witness,
        "objective_order_witness": order_witness,
        "counterexample": counterexample,
    }


def _z3_number(value: Any) -> Any:
    """Create a Z3 rational without a floating-point conversion."""

    exact = as_fraction(value)
    return z3.Q(exact.numerator, exact.denominator)


def _z3_variables(formulation: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Create typed Z3 symbols for one formulation copy."""

    return {
        variable["name"]: (
            z3.Bool(f"{prefix}_{variable['name']}")
            if variable["kind"] == "boolean"
            else z3.Int(f"{prefix}_{variable['name']}")
        )
        for variable in formulation["variables"]
    }


def _z3_numeric(symbol: Any) -> Any:
    """Convert a Boolean cardinality symbol to exact zero-or-one arithmetic."""

    return z3.If(symbol, z3.IntVal(1), z3.IntVal(0)) if z3.is_bool(symbol) else symbol


def _z3_universe(formulation: Mapping[str, Any], symbols: Mapping[str, Any]) -> Any:
    """Encode only the finite enumeration universe, separate from semantic bounds."""

    clauses = []
    for variable in formulation["variables"]:
        symbol = symbols[variable["name"]]
        if variable["kind"] == "integer":
            clauses.append(z3.Or(*[symbol == value for value in variable["universe"]]))
    return z3.And(*clauses) if clauses else z3.BoolVal(True)


def _z3_linear(expression: Mapping[str, Any], symbols: Mapping[str, Any]) -> Any:
    """Encode one exact affine expression in Z3."""

    constant = _z3_number(expression.get("constant", "0"))
    return constant + sum(
        _z3_number(coefficient) * _z3_numeric(symbols[name])
        for name, coefficient in expression["terms"].items()
    )


def _z3_relation(left: Any, op: str, right: Any) -> Any:
    """Translate the closed constraint-operator roster to Z3."""

    return {
        "<=": left <= right,
        ">=": left >= right,
        "==": left == right,
        "<": left < right,
        ">": left > right,
    }[op]


def _z3_semantics(formulation: Mapping[str, Any], symbols: Mapping[str, Any]) -> Any:
    """Encode semantic bounds and constraints, excluding the universe wrapper."""

    clauses = []
    for variable in formulation["variables"]:
        symbol = _z3_numeric(symbols[variable["name"]])
        lower = variable["domain"]["lower"]
        upper = variable["domain"]["upper"]
        if lower is not None:
            clauses.append(symbol >= _z3_number(lower))
        if upper is not None:
            clauses.append(symbol <= _z3_number(upper))
    for constraint in formulation["constraints"]:
        clauses.append(
            _z3_relation(
                _z3_linear({"terms": constraint["terms"], "constant": "0"}, symbols),
                constraint["op"],
                _z3_number(constraint["rhs"]),
            )
        )
    return z3.And(*clauses) if clauses else z3.BoolVal(True)


def _z3_feasible(formulation: Mapping[str, Any], symbols: Mapping[str, Any]) -> Any:
    """Encode complete bounded feasibility for one formulation."""

    return z3.And(_z3_universe(formulation, symbols), _z3_semantics(formulation, symbols))


def _z3_objective(formulation: Mapping[str, Any], symbols: Mapping[str, Any]) -> Any:
    """Encode a linear or exact piecewise-linear objective in Z3."""

    expression = formulation["objective"]["expression"]
    if expression["kind"] == "linear":
        return _z3_linear(expression, symbols)
    values = [_z3_linear(piece, symbols) for piece in expression["pieces"]]
    current = values[0]
    for value in values[1:]:
        if expression["aggregation"] == "max":
            current = z3.If(current >= value, current, value)
        else:
            current = z3.If(current <= value, current, value)
    return current


def _z3_mapping(
    mapping: Mapping[str, Any], source: Mapping[str, Any], target: Mapping[str, Any]
) -> Any:
    """Encode all affine variable equations for one source-target copy."""

    clauses = []
    for row in mapping["variables"]:
        source_symbol = _z3_numeric(source[row["source"]])
        target_symbol = _z3_numeric(target[row["target"]])
        clauses.append(
            target_symbol == _z3_number(row["scale"]) * source_symbol + _z3_number(row["offset"])
        )
    return z3.And(*clauses)


def _solver_status(*clauses: Any) -> tuple[str, str | None]:
    """Run one bounded proof obligation and retain any unknown explanation."""

    solver = z3.Solver()
    solver.set(timeout=Z3_TIMEOUT_MS)
    solver.add(*clauses)
    status = solver.check()
    if status == z3.sat:
        return "sat", None
    if status == z3.unsat:
        return "unsat", None
    return "unknown", solver.reason_unknown()


def _z3_better(direction: str, left: Any, right: Any) -> Any:
    """Encode weak objective preference in the declared direction."""

    return left <= right if direction == "min" else left >= right


def _z3_duplicate_mapping_status(mapping: Mapping[str, Any]) -> str:
    """Ask Z3 for the structural duplicate witness used by malformed negatives."""

    targets = [str(row.get("target")) for row in mapping.get("variables", [])]
    identities = {name: index for index, name in enumerate(sorted(set(targets)))}
    solver = z3.Solver()
    solver.add(z3.Not(z3.Distinct(*[z3.IntVal(identities[name]) for name in targets])))
    return str(solver.check())


def prove_pair_with_z3(pair: Mapping[str, Any]) -> JsonDict:
    """Prove or refute one pair with independent symbolic counterexample queries."""

    pair_id = str(pair["pair_id"])
    if z3 is None:
        return z3_failure_row(pair_id, "unknown", "z3_unavailable")
    source = validate_formulation(pair["source"])
    target = validate_formulation(pair["target"])
    try:
        mapping = canonical_mapping(pair["mapping"], source, target)
    except MappingSchemaError as exc:
        return {
            **z3_failure_row(pair_id, "schema_rejected", str(exc)),
            "label": "non_equivalent",
            "mapping_bijection_status": _z3_duplicate_mapping_status(pair["mapping"]),
        }
    s = _z3_variables(source, f"s_{pair_id.replace('-', '_')}")
    t = _z3_variables(target, f"t_{pair_id.replace('-', '_')}")
    map_clause = _z3_mapping(mapping, s, t)
    domain_status, domain_reason = _solver_status(
        _z3_universe(source, s),
        _z3_universe(target, t),
        map_clause,
        z3.Xor(_z3_semantics(source, s), _z3_semantics(target, t)),
    )
    source_objective = _z3_objective(source, s)
    target_objective = _z3_objective(target, t)
    affine_status, affine_reason = _solver_status(
        _z3_feasible(source, s),
        _z3_feasible(target, t),
        map_clause,
        target_objective
        != _z3_number(mapping["objective"]["scale"]) * source_objective
        + _z3_number(mapping["objective"]["offset"]),
    )
    s2 = _z3_variables(source, f"s2_{pair_id.replace('-', '_')}")
    t2 = _z3_variables(target, f"t2_{pair_id.replace('-', '_')}")
    order_status, order_reason = _solver_status(
        _z3_feasible(source, s),
        _z3_feasible(target, t),
        _z3_mapping(mapping, s, t),
        _z3_feasible(source, s2),
        _z3_feasible(target, t2),
        _z3_mapping(mapping, s2, t2),
        _z3_better(
            source["objective"]["direction"],
            _z3_objective(source, s),
            _z3_objective(source, s2),
        )
        != _z3_better(
            target["objective"]["direction"],
            _z3_objective(target, t),
            _z3_objective(target, t2),
        ),
    )
    statuses = (domain_status, affine_status, order_status)
    unknown_reasons = [reason for reason in (domain_reason, affine_reason, order_reason) if reason]
    if "unknown" in statuses:
        overall_status = (
            "timeout"
            if any("timeout" in reason.lower() for reason in unknown_reasons)
            else "unknown"
        )
        label = None
    elif statuses == ("unsat", "unsat", "unsat"):
        overall_status = "proved"
        label = "equivalent"
    else:
        overall_status = "counterexample"
        label = "non_equivalent"
    return {
        "pair_id": pair_id,
        "engine": "z3_symbolic_reformulation_checker_v1",
        "status": overall_status,
        "label": label,
        "domain_counterexample_status": domain_status,
        "objective_affine_status": affine_status,
        "objective_order_status": order_status,
        "unknown_reasons": unknown_reasons,
        "mapping_bijection_status": "unsat",
    }


def z3_failure_row(pair_id: str, status: str, reason: str | None = None) -> JsonDict:
    """Build a complete Z3 row when symbolic proof cannot run normally."""

    return {
        "pair_id": pair_id,
        "engine": "z3_symbolic_reformulation_checker_v1",
        "status": status,
        "label": None,
        "domain_counterexample_status": "not_run",
        "objective_affine_status": "not_run",
        "objective_order_status": "not_run",
        "unknown_reasons": [reason] if reason else [],
        "mapping_bijection_status": "not_run",
    }


def authority_agreement_row(
    pair: Mapping[str, Any], enumeration: Mapping[str, Any], z3_row: Mapping[str, Any]
) -> JsonDict:
    """Compare independent row labels and assign a fail-closed quarantine reason."""

    z3_status = str(z3_row["status"])
    if z3_status in {"unknown", "timeout"}:
        reason = f"z3_{z3_status}"
    elif enumeration["label"] != z3_row["label"]:
        reason = "authority_disagreement"
    elif enumeration["label"] != pair["expected_label"]:
        reason = "expected_label_disagreement"
    else:
        reason = None
    return {
        "pair_id": pair["pair_id"],
        "expected_label": pair["expected_label"],
        "enumeration_label": enumeration["label"],
        "z3_label": z3_row["label"],
        "authorities_agree": reason is None,
        "quarantined": reason is not None,
        "reason": reason,
    }


def formulation_hash(formulation: Mapping[str, Any]) -> str:
    """Hash the validated canonical formulation bytes."""

    return sha256_bytes(canonical_json(validate_formulation(formulation)))


def mapping_hash(mapping: Mapping[str, Any]) -> str:
    """Hash proposed mapping bytes even when the proposal is intentionally invalid."""

    return sha256_bytes(canonical_json(mapping))


def pair_hash(pair: Mapping[str, Any]) -> str:
    """Hash all serialized semantic inputs for one pair."""

    payload = {
        "source": validate_formulation(pair["source"]),
        "target": validate_formulation(pair["target"]),
        "mapping": pair["mapping"],
        "family": pair["family"],
        "expected_label": pair["expected_label"],
        "hard_negative_edit": pair["hard_negative_edit"],
    }
    return sha256_bytes(canonical_json(payload))


def _rename_formulation(formulation: Mapping[str, Any], rename: Mapping[str, str]) -> JsonDict:
    """Rename all variables for one isomorphism-canonicalization candidate."""

    renamed = deepcopy(formulation)
    for variable in renamed["variables"]:
        variable["name"] = rename[variable["name"]]
    for constraint in renamed["constraints"]:
        constraint["terms"] = {rename[name]: value for name, value in constraint["terms"].items()}
    expression = renamed["objective"]["expression"]
    linear_rows = [expression] if expression["kind"] == "linear" else expression["pieces"]
    for linear in linear_rows:
        linear["terms"] = {rename[name]: value for name, value in linear["terms"].items()}
    renamed["variables"] = sorted(renamed["variables"], key=lambda row: row["name"])
    return validate_formulation(renamed)


def formulation_isomorphism_hash(formulation: Mapping[str, Any]) -> str:
    """Find the lexical minimum across every variable-name permutation."""

    canonical = validate_formulation(formulation)
    names = [variable["name"] for variable in canonical["variables"]]
    candidates = []
    for ordering in permutations(names):
        rename = {name: f"v{index}" for index, name in enumerate(ordering)}
        candidates.append(canonical_json(_rename_formulation(canonical, rename)))
    return sha256_bytes(min(candidates))


def normalized_pair_hash(pair: Mapping[str, Any]) -> str:
    """Hash order-normalized pair content without IDs, templates, or split names."""

    payload = {
        "source": formulation_isomorphism_hash(pair["source"]),
        "target": formulation_isomorphism_hash(pair["target"]),
        "expected_label": pair["expected_label"],
        "hard_negative_edit": pair["hard_negative_edit"],
    }
    return sha256_bytes(canonical_json(payload))


def pair_isomorphism_hash(pair: Mapping[str, Any]) -> str:
    """Bind formulation isomorphism to affine coefficients and objective relation."""

    variable_shape = sorted(
        (str(row.get("scale")), str(row.get("offset"))) for row in pair["mapping"]["variables"]
    )
    payload = {
        "normalized_pair_hash": normalized_pair_hash(pair),
        "variable_shape": variable_shape,
        "objective": pair["mapping"]["objective"],
    }
    return sha256_bytes(canonical_json(payload))


def _replay_row(pair: Mapping[str, Any]) -> JsonDict:
    """Recompute all replay-sensitive identities and witnesses from one pair."""

    enumeration = prove_pair_with_enumerator(pair)
    z3_row = prove_pair_with_z3(pair)
    row = {
        "pair_id": pair["pair_id"],
        "source_hash": formulation_hash(pair["source"]),
        "target_hash": formulation_hash(pair["target"]),
        "mapping_hash": mapping_hash(pair["mapping"]),
        "pair_hash": pair_hash(pair),
        "label": enumeration["label"] if enumeration["label"] == z3_row["label"] else None,
        "feasibility_witness": enumeration["feasibility_witness"],
        "objective_order_witness": enumeration["objective_order_witness"],
        "counterexample": enumeration["counterexample"],
        "generator_template": pair["generator_template"],
        "split": TEMPLATE_SPLITS[int(pair["template_index"])],
    }
    row["replay_hash"] = sha256_bytes(canonical_json(row))
    return row


def replay_checkpoint(path: Path) -> list[JsonDict]:
    """Read only serialized pairs and recompute their complete replay receipts."""

    checkpoint = json.loads(Path(path).read_text(encoding="utf-8"))
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint_schema_version")
    pairs = checkpoint.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("checkpoint_pairs")
    return [_replay_row(pair) for pair in pairs]


def _write_json(path: Path, value: Any) -> None:
    """Write canonical pretty JSON through a same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fresh_process_replay(repo_root: Path, checkpoint_path: Path) -> list[JsonDict]:
    """Launch a clean interpreter that receives only the checkpoint path."""

    output_path = checkpoint_path.with_suffix(".replay.json")
    environment = dict(os.environ)
    python_root = str(repo_root / "python")
    environment["PYTHONPATH"] = python_root + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_6955_reformulation_fixture",
        "--replay-checkpoint",
        str(checkpoint_path),
        "--replay-output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fresh_process_replay_failed:{completed.stderr[-500:]}")
    rows = json.loads(output_path.read_text(encoding="utf-8"))
    output_path.unlink()
    if not isinstance(rows, list):
        raise RuntimeError("fresh_process_replay_not_rows")
    return rows


def _gate_check(name: str, expected: Any, observed: Any) -> JsonDict:
    """Build one exact expected-versus-observed gate row."""

    return {
        "check": name,
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep every gate row and surface the first deterministic failure."""

    failed = [row for row in checks if not row["passed"]]
    first = failed[0] if failed else None
    return {
        "checks": [dict(row) for row in checks],
        "passed": not failed,
        "failed_check": first["check"] if first else None,
        "expected": first["expected"] if first else "all checks pass",
        "observed": first["observed"] if first else "all checks pass",
    }


def _z3_version() -> str | None:
    """Return the installed Z3 version without making import failure fatal."""

    return z3.get_version_string() if z3 is not None else None


def check_preconditions(repo_root: Path, checkpoint_path: Path) -> JsonDict:
    """Check Z3, exact arithmetic, seeded replay, interfaces, and checkpoint writes."""

    root = Path(repo_root)
    checkpoint = Path(checkpoint_path)
    interface_hashes = {str(path): sha256_path(root / path) for path in CONSTRAINT_INTERFACE_PATHS}
    exact_observed = fraction_text(Fraction(1, 3) + Fraction(2, 3))
    seeded_observed = [random.Random(RANDOM_SEED).randrange(10**6) for _ in range(2)]
    seeded_expected = [random.Random(RANDOM_SEED).randrange(10**6) for _ in range(2)]
    writable = False
    writable_error: str | None = None
    try:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=checkpoint.parent, prefix=".exp6955-", delete=True):
            writable = True
    except OSError as exc:
        writable_error = type(exc).__name__
    checks = {
        "z3_available": {
            "passed": _z3_version() is not None,
            "expected": {"available": True},
            "observed": {"available": _z3_version() is not None, "version": _z3_version()},
        },
        "exact_rational_arithmetic": {
            "passed": exact_observed == "1",
            "expected": "1",
            "observed": exact_observed,
        },
        "deterministic_seed_control": {
            "passed": seeded_observed == seeded_expected,
            "expected": seeded_expected,
            "observed": seeded_observed,
        },
        "current_constraint_interfaces": {
            "passed": all(interface_hashes.values()),
            "expected": {name: "sha256:*" for name in interface_hashes},
            "observed": interface_hashes,
        },
        "writable_checkpoints": {
            "passed": writable,
            "expected": {"writable": True},
            "observed": {
                "writable": writable,
                "path": str(checkpoint),
                "error": writable_error,
            },
        },
    }
    return {**checks, "all_passed": all(row["passed"] for row in checks.values())}


def source_artifact_hashes(repo_root: Path) -> JsonDict:
    """Bind the artifact to the spec, implementation, tests, wrapper, and interfaces."""

    paths = {
        "spec": SPEC_PATH,
        "module": MODULE_PATH,
        "tests": TEST_PATH,
        "wrapper": WRAPPER_PATH,
        **{
            f"constraint_interface_{index}": path
            for index, path in enumerate(CONSTRAINT_INTERFACE_PATHS)
        },
    }
    return {
        name: {"path": str(path), "sha256": sha256_path(Path(repo_root) / path)}
        for name, path in paths.items()
    }


def _empty_rows() -> JsonDict:
    """Return every required row family for a complete blocked artifact."""

    return {
        field: []
        for field in REQUIRED_ARTIFACT_FIELDS
        if field == "rows" or field.endswith("_rows")
    }


def build_blocked_artifact(
    *,
    date: str,
    repo_root: Path,
    checkpoint_path: Path,
    preconditions: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Build the complete fail-closed artifact required when a precondition fails."""

    checks = [
        {
            "check": name,
            "expected": row["expected"],
            "observed": row["observed"],
            "passed": bool(row["passed"]),
        }
        for name, row in preconditions.items()
        if isinstance(row, Mapping) and {"expected", "observed", "passed"} <= set(row)
    ]
    artifact: JsonDict = {
        "schema": "carnot.exp6955.reformulation_fixture.v1",
        "experiment_id": 6955,
        "run_date": date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": dict(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": source_artifact_hashes(repo_root),
        **_empty_rows(),
        "quarantine_rows": [],
        "corpus_checkpoint_path": str(checkpoint_path),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "reformulation_fixture_ready_score": 0,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_reformulation_fixture",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _mapping_row(pair: Mapping[str, Any]) -> JsonDict:
    """Record canonical mapping content or its deterministic rejection."""

    try:
        canonical = canonical_mapping(pair["mapping"], pair["source"], pair["target"])
        valid = True
        rejection = None
    except MappingSchemaError as exc:
        canonical = deepcopy(pair["mapping"])
        valid = False
        rejection = str(exc)
    return {
        "pair_id": pair["pair_id"],
        "schema_version": pair["mapping"].get("schema_version"),
        "schema_valid": valid,
        "rejection": rejection,
        "mapping": canonical,
        "mapping_hash": mapping_hash(pair["mapping"]),
    }


def _collision_splits(rows: Sequence[Mapping[str, Any]], hash_field: str) -> dict[str, set[str]]:
    """Group any identity by the splits where it appears."""

    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[str(row[hash_field])].add(str(row["split"]))
    return grouped


def build_artifact(
    *,
    date: str,
    repo_root: Path = REPO_ROOT,
    checkpoint_path: Path | None = None,
    z3_prover: Z3Prover = prove_pair_with_z3,
) -> JsonDict:
    """Build, independently certify, serialize, and fresh-process replay the corpus."""

    started_ns = time.perf_counter_ns()
    root = Path(repo_root)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else root / CHECKPOINT_PATH
    preconditions = check_preconditions(root, checkpoint)
    if not preconditions["all_passed"]:
        duration = max((time.perf_counter_ns() - started_ns) / 1_000_000_000, 0.000001)
        return build_blocked_artifact(
            date=date,
            repo_root=root,
            checkpoint_path=checkpoint,
            preconditions=preconditions,
            duration_s=round(duration, 6),
        )
    pairs = generate_pairs(RANDOM_SEED)
    checkpoint_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "random_seed": RANDOM_SEED,
        "pairs": pairs,
    }
    _write_json(checkpoint, checkpoint_payload)
    enumeration_rows = [prove_pair_with_enumerator(pair) for pair in pairs]
    z3_rows = [z3_prover(pair) for pair in pairs]
    agreement_rows = [
        authority_agreement_row(pair, enumeration, z3_row)
        for pair, enumeration, z3_row in zip(pairs, enumeration_rows, z3_rows, strict=True)
    ]
    pair_rows: list[JsonDict] = []
    formulation_rows: list[JsonDict] = []
    mapping_rows = [_mapping_row(pair) for pair in pairs]
    split_rows: list[JsonDict] = []
    isomorphism_rows: list[JsonDict] = []
    for pair, enumeration, z3_row, agreement in zip(
        pairs, enumeration_rows, z3_rows, agreement_rows, strict=True
    ):
        source_hash = formulation_hash(pair["source"])
        target_hash = formulation_hash(pair["target"])
        current_pair_hash = pair_hash(pair)
        normalized_hash = normalized_pair_hash(pair)
        isomorphism_hash = pair_isomorphism_hash(pair)
        formulation_rows.extend(
            [
                {
                    "pair_id": pair["pair_id"],
                    "side": "source",
                    "formulation": pair["source"],
                    "formulation_hash": source_hash,
                },
                {
                    "pair_id": pair["pair_id"],
                    "side": "target",
                    "formulation": pair["target"],
                    "formulation_hash": target_hash,
                },
            ]
        )
        pair_rows.append(
            {
                "pair_id": pair["pair_id"],
                "family": pair["family"],
                "difficulty": pair["difficulty"],
                "generator_template": pair["generator_template"],
                "split": pair["split"],
                "claimed_relation": pair["mapping"]["claimed_relation"],
                "expected_label": pair["expected_label"],
                "enumeration_label": enumeration["label"],
                "z3_label": z3_row["label"],
                "hard_negative_edit": pair["hard_negative_edit"],
                "hard_negative_edit_count": pair["hard_negative_edit_count"],
                "source_hash": source_hash,
                "target_hash": target_hash,
                "mapping_hash": mapping_hash(pair["mapping"]),
                "pair_hash": current_pair_hash,
                "feasibility_witness": enumeration["feasibility_witness"],
                "objective_order_witness": enumeration["objective_order_witness"],
                "counterexample": enumeration["counterexample"],
                "authorities_agree": agreement["authorities_agree"],
                "quarantined": agreement["quarantined"],
            }
        )
        split_rows.append(
            {
                "pair_id": pair["pair_id"],
                "generator_template": pair["generator_template"],
                "split": pair["split"],
                "normalized_hash": normalized_hash,
                "isomorphism_hash": isomorphism_hash,
            }
        )
        isomorphism_rows.append(
            {
                "pair_id": pair["pair_id"],
                "split": pair["split"],
                "normalized_hash": normalized_hash,
                "isomorphism_hash": isomorphism_hash,
            }
        )
    normalized_splits = _collision_splits(split_rows, "normalized_hash")
    isomorphic_splits = _collision_splits(split_rows, "isomorphism_hash")
    for row in split_rows:
        row["normalized_hash_crosses_split"] = len(normalized_splits[row["normalized_hash"]]) > 1
        row["isomorphism_hash_crosses_split"] = len(isomorphic_splits[row["isomorphism_hash"]]) > 1
    for row in isomorphism_rows:
        row["crosses_split"] = (
            len(normalized_splits[row["normalized_hash"]]) > 1
            or len(isomorphic_splits[row["isomorphism_hash"]]) > 1
        )
    parent_replay = [_replay_row(pair) for pair in pairs]
    child_replay = _fresh_process_replay(root, checkpoint)
    child_by_id = {row["pair_id"]: row for row in child_replay}
    replay_rows = [
        {
            **row,
            "replay_matches": child_by_id.get(row["pair_id"]) == row,
            "fresh_process": True,
        }
        for row in parent_replay
    ]
    family_rows = [
        {
            "family": family,
            "pair_count": sum(row["family"] == family for row in pair_rows),
            "equivalent_count": sum(
                row["family"] == family and row["expected_label"] == "equivalent"
                for row in pair_rows
            ),
            "non_equivalent_count": sum(
                row["family"] == family and row["expected_label"] == "non_equivalent"
                for row in pair_rows
            ),
        }
        for family in FAMILIES
    ]
    label_rows = [
        {"label": label, "pair_count": sum(row["expected_label"] == label for row in pair_rows)}
        for label in ("equivalent", "non_equivalent")
    ]
    quarantine_rows = [row for row in agreement_rows if row["quarantined"]]
    hard_negative_rows = [
        {
            "pair_id": row["pair_id"],
            "family": row["family"],
            "edit": row["hard_negative_edit"],
            "edit_count": row["hard_negative_edit_count"],
            "counterexample": row["counterexample"],
        }
        for row in pair_rows
        if row["expected_label"] == "non_equivalent"
    ]
    feasibility_rows = [
        {
            "pair_id": row["pair_id"],
            "expected_label": row["expected_label"],
            "witness": enumeration["feasibility_witness"],
            "domain_preserved": enumeration["domain_preserved"],
            "counterexample": enumeration["counterexample"]
            if not enumeration["domain_preserved"]
            else None,
        }
        for row, enumeration in zip(pair_rows, enumeration_rows, strict=True)
    ]
    objective_rows = [
        {
            "pair_id": row["pair_id"],
            "expected_label": row["expected_label"],
            "objective_affine_preserved": enumeration["objective_affine_preserved"],
            "objective_order_preserved": enumeration["objective_order_preserved"],
            "witness": enumeration["objective_order_witness"],
            "counterexample": enumeration["counterexample"]
            if not enumeration["objective_order_preserved"]
            else None,
        }
        for row, enumeration in zip(pair_rows, enumeration_rows, strict=True)
    ]
    gate_checks = [
        _gate_check("pair_count", EXPECTED_ROW_COUNT, len(pair_rows)),
        _gate_check(
            "label_counts",
            {"equivalent": 72, "non_equivalent": 48},
            dict(Counter(row["expected_label"] for row in pair_rows)),
        ),
        _gate_check(
            "family_coverage", sorted(FAMILIES), sorted({row["family"] for row in pair_rows})
        ),
        _gate_check("split_coverage", sorted(SPLITS), sorted({row["split"] for row in pair_rows})),
        _gate_check(
            "authority_agreement_count",
            EXPECTED_ROW_COUNT,
            sum(row["authorities_agree"] for row in agreement_rows),
        ),
        _gate_check("quarantine_count", 0, len(quarantine_rows)),
        _gate_check(
            "normalized_cross_split_count",
            0,
            sum(row["normalized_hash_crosses_split"] for row in split_rows),
        ),
        _gate_check(
            "isomorphism_cross_split_count",
            0,
            sum(row["isomorphism_hash_crosses_split"] for row in split_rows),
        ),
        _gate_check(
            "fresh_process_replay_count",
            EXPECTED_ROW_COUNT,
            sum(row["replay_matches"] for row in replay_rows),
        ),
        _gate_check(
            "hard_negative_single_edit_count",
            48,
            sum(row["edit_count"] == 1 for row in hard_negative_rows),
        ),
    ]
    summary = gate_summary(gate_checks)
    ready = int(summary["passed"])
    duration = max((time.perf_counter_ns() - started_ns) / 1_000_000_000, 0.000001)
    artifact: JsonDict = {
        "schema": "carnot.exp6955.reformulation_fixture.v1",
        "experiment_id": 6955,
        "run_date": date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration, 6),
        "source_artifact_hashes": source_artifact_hashes(root),
        "rows": deepcopy(pair_rows),
        "pair_rows": pair_rows,
        "formulation_rows": formulation_rows,
        "mapping_rows": mapping_rows,
        "family_rows": family_rows,
        "label_rows": label_rows,
        "hard_negative_rows": hard_negative_rows,
        "feasibility_witness_rows": feasibility_rows,
        "objective_order_rows": objective_rows,
        "z3_rows": z3_rows,
        "enumeration_rows": enumeration_rows,
        "authority_agreement_rows": agreement_rows,
        "split_rows": split_rows,
        "isomorphism_rows": isomorphism_rows,
        "fresh_process_replay_rows": replay_rows,
        "quarantine_rows": quarantine_rows,
        "corpus_checkpoint_path": str(
            checkpoint.relative_to(root) if checkpoint.is_relative_to(root) else checkpoint
        ),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "reformulation_fixture_ready_score": ready,
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": "circular_positive" if ready else "blocked",
        "honest_verdict": (
            "complete_circular_positive_reformulation_fixture_conforms"
            if ready
            else "blocked_reformulation_fixture"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash scientific content while excluding wall-clock duration and the digest itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return sha256_bytes(canonical_json(payload))


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject a terminal artifact whose rows do not support its declared verdict."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing_required_fields:{missing}")
    missing_principles = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact["field_principles"]))
    if missing_principles:
        raise ValueError(f"missing_field_principles:{missing_principles}")
    score = artifact["reformulation_fixture_ready_score"]
    if score not in {0, 1}:
        raise ValueError("invalid_ready_score")
    if bool(score) != bool(artifact["gate_check_summary"]["passed"]):
        raise ValueError("ready_gate_disagreement")
    if score and len(artifact["pair_rows"]) != EXPECTED_ROW_COUNT:
        raise ValueError("ready_pair_count")
    if score and artifact["verdict_class"] != "circular_positive":
        raise ValueError("ready_verdict_class")
    if not score and artifact["verdict_class"] != "blocked":
        raise ValueError("blocked_verdict_class")
    honest = str(artifact["honest_verdict"])
    if score and not honest.startswith("complete_circular_positive"):
        raise ValueError("honest_verdict_not_circular_terminal")
    if not score and not honest.startswith("blocked"):
        raise ValueError("honest_verdict_not_blocked_terminal")
    if payload_checksum(artifact) != artifact["reproducibility_checksum"]:
        raise ValueError("reproducibility_checksum_mismatch")


def run(
    *,
    date: str,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> JsonDict:
    """Build and validate the fixture, then write its required artifact."""

    root = Path(repo_root)
    output = Path(output_path) if output_path is not None else root / RESULT_PATH
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else root / CHECKPOINT_PATH
    artifact = build_artifact(date=date, repo_root=root, checkpoint_path=checkpoint)
    validate_artifact(artifact)
    _write_json(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Expose corpus construction and the private child-process replay mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260903")
    parser.add_argument("--replay-checkpoint", type=Path)
    parser.add_argument("--replay-output", type=Path)
    args = parser.parse_args(argv)
    if args.replay_checkpoint is not None:
        if args.replay_output is None:
            parser.error("--replay-output is required with --replay-checkpoint")
        _write_json(args.replay_output, replay_checkpoint(args.replay_checkpoint))
        return 0
    artifact = run(date=args.date)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "pair_count": len(artifact["pair_rows"]),
                "reformulation_fixture_ready_score": artifact["reformulation_fixture_ready_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the required wrapper.
    raise SystemExit(main())
