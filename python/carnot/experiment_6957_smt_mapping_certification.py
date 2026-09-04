"""Certify every frozen Exp6956 reformulation proposal with two exact engines.

Spec refs: REQ-VERIFY-6957 and SCENARIO-VERIFY-6957-*.

The model output is evidence under test, never authority. Z3 and a separate
finite enumerator decide the relation. The module consumes the frozen Exp6956
parse result as-is and keeps every failed proposal in the fixed denominator.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from fractions import Fraction
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_6955_reformulation_fixture as fixture_exp


JsonDict = dict[str, Any]
Certifier = Callable[[Mapping[str, Any]], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
BANK_PATH = Path("results/experiment_6956_three_family_reformulation_bank.json")
FIXTURE_PATH = Path("results/experiment_6955_reformulation_fixture.json")
FIXTURE_CHECKPOINT_PATH = Path(
    "results/checkpoints/experiment_6955_reformulation_fixture_corpus.json"
)
REPLAY_CHECKPOINT_PATH = Path(
    "results/checkpoints/experiment_6957_smt_mapping_certification_inputs.json"
)
RESULT_PATH = Path("results/experiment_6957_smt_mapping_certification.json")
MODULE_PATH = Path("python/carnot/experiment_6957_smt_mapping_certification.py")
TEST_PATH = Path("tests/python/test_experiment_6957_smt_mapping_certification.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6957_smt_mapping_certification.py")

SCHEMA_VERSION = "carnot.exp6957.smt_mapping_certification.v1"
REPLAY_SCHEMA_VERSION = "carnot.exp6957.smt_mapping_replay.v1"
INFERENCE_SUBSTRATE = "frozen_sota_proposals_z3_and_exact_enumeration"
RANDOM_SEED = 695720260903
EXPECTED_PROPOSAL_COUNT = 162
BOOTSTRAP_RESAMPLES = 10_000
Z3_TIMEOUT_MS = 2_000
TERMINAL_ENGINE_STATUSES = {
    "proved",
    "counterexample",
    "parse_rejected",
    "schema_rejected",
    "timeout",
    "unknown",
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "proposal_rows",
    "parse_rows",
    "schema_rows",
    "cross_feasibility_rows",
    "variable_coverage_rows",
    "objective_direction_rows",
    "objective_order_rows",
    "z3_rows",
    "enumeration_rows",
    "authority_agreement_rows",
    "witness_rows",
    "counterexample_rows",
    "model_rows",
    "family_rows",
    "difficulty_rows",
    "confidence_rows",
    "rationale_rows",
    "calibration_rows",
    "baseline_rows",
    "paired_metric_rows",
    "confidence_interval_rows",
    "fresh_process_replay_rows",
    "random_seed",
    "reproducibility_checksum",
    "smt_certification_run_complete_score",
    "sota_mapping_positive_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason per field makes the evidence contract auditable.",
    "preconditions_checked": "Fail-closed checks prevent partial inputs or engines from becoming evidence.",
    "inference_substrate": "The fixed declaration separates exact certification from model generation.",
    "duration_s": "Measured wall time shows that both engines and replay executed.",
    "source_artifact_hashes": "Hashes bind conclusions to the frozen bank, fixture, code, and tests.",
    "rows": "The full proposal surface lets independent checks recompute every headline.",
    "proposal_rows": "One terminal outcome per frozen key preserves the fixed denominator.",
    "parse_rows": "Frozen parse outcomes expose malformed output without later extraction.",
    "schema_rows": "Strict schema decisions prevent repaired or incomplete maps from admission.",
    "cross_feasibility_rows": "Forward and reverse checks detect lost or invented feasible points.",
    "variable_coverage_rows": "Explicit rosters detect missing and non-bijective correspondences.",
    "objective_direction_rows": "Direction and scale-sign checks prevent optimization reversal.",
    "objective_order_rows": "Global pair checks detect rank changes, including broken ties.",
    "z3_rows": "SMT proof obligations provide symbolic external label authority.",
    "enumeration_rows": "Exhaustive finite search supplies an implementation-independent authority.",
    "authority_agreement_rows": "Per-row parity quarantines disagreements instead of averaging them away.",
    "witness_rows": "Exact assignments make successful and diagnostic claims replayable.",
    "counterexample_rows": "Exact failures identify why a proposed relation is not certified.",
    "model_rows": "Per-model metrics expose family-specific strengths and failures.",
    "family_rows": "Per-problem-family metrics prevent composition effects from hiding errors.",
    "difficulty_rows": "Difficulty strata show whether hard negatives change conclusions.",
    "confidence_rows": "Self-reported confidence remains visible only as a monitoring signal.",
    "rationale_rows": "Rationale presence can be monitored without receiving authority.",
    "calibration_rows": "Calibration and discrimination measure whether advisory signals predict error.",
    "baseline_rows": "A frozen syntax-only comparator gives the exact checker a paired reference.",
    "paired_metric_rows": "Attempt-level differences preserve pairing for uncertainty estimates.",
    "confidence_interval_rows": "Paired CI95 bounds prevent noisy point gains from passing the gate.",
    "fresh_process_replay_rows": "Child-process hashes detect hidden parent state and serialization drift.",
    "random_seed": "One fixed seed makes bootstrap and ordering decisions repeatable.",
    "reproducibility_checksum": "A timing-free digest detects scientific-content drift.",
    "smt_certification_run_complete_score": "The binary score opens only for 162 terminal replayed rows.",
    "sota_mapping_positive_score": "The positive gate requires replicated paired gains and low false acceptance.",
    "gate_check_summary": "Expected and observed values make blocked or partial runs actionable.",
    "verifier_is_oracle": "False states that external solvers label the evaluated mapping method.",
    "verdict_class": "A closed class separates positive, null, partial, and blocked evidence.",
    "honest_verdict": "A stable prefix lets automation classify the scientific outcome.",
}


def canonical_json(value: Any) -> bytes:
    """Return deterministic JSON bytes for hashes and equality checks."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return the repository spelling of one SHA-256 digest."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str | None:
    """Hash a file or preserve its absence as an explicit null."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record one exact precondition comparison."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": expected == observed,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return every failed check with the values needed to diagnose it."""

    return [
        {
            "failed_check": row["check"],
            "expected_value": row["expected_value"],
            "observed_value": row["observed_value"],
        }
        for row in checks
        if row.get("passed") is not True
    ]


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON through a same-directory replacement so replay sees whole rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _opposite(direction: str) -> str:
    """Return the only valid reversed optimization direction."""

    return "max" if direction == "min" else "min"


def _direction_valid(
    mapping: Mapping[str, Any], source: Mapping[str, Any], target: Mapping[str, Any]
) -> bool:
    """Check declarations and the exact scale-sign direction rule."""

    objective = mapping["objective"]
    source_direction = source["objective"]["direction"]
    target_direction = target["objective"]["direction"]
    scale = fixture_exp.as_fraction(objective["scale"])
    expected_target = source_direction if scale > 0 else _opposite(source_direction)
    return (
        objective["source_direction"] == source_direction
        and objective["target_direction"] == target_direction
        and target_direction == expected_target
    )


def _variable_coverage(
    attempt_key: str,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    mapping: Any,
) -> JsonDict:
    """Describe source and target roster coverage even for malformed mappings."""

    source_required = [str(row.get("name")) for row in source.get("variables", [])]
    target_required = [str(row.get("name")) for row in target.get("variables", [])]
    variable_rows = mapping.get("variables", []) if isinstance(mapping, Mapping) else []
    if not isinstance(variable_rows, list):
        variable_rows = []
    source_observed = [str(row.get("source")) for row in variable_rows if isinstance(row, Mapping)]
    target_observed = [str(row.get("target")) for row in variable_rows if isinstance(row, Mapping)]
    source_complete = sorted(source_observed) == sorted(source_required)
    target_complete = sorted(target_observed) == sorted(target_required)
    source_unique = len(source_observed) == len(set(source_observed))
    target_unique = len(target_observed) == len(set(target_observed))
    return {
        "attempt_key": attempt_key,
        "source_required": source_required,
        "source_observed": source_observed,
        "target_required": target_required,
        "target_observed": target_observed,
        "source_complete": source_complete,
        "target_complete": target_complete,
        "source_unique": source_unique,
        "target_unique": target_unique,
        "coverage_complete": source_complete
        and target_complete
        and source_unique
        and target_unique,
    }


def _candidate_state(attempt: Mapping[str, Any]) -> tuple[JsonDict, JsonDict, JsonDict | None]:
    """Apply the frozen parse boundary, then the stricter exact-engine schema."""

    key = str(attempt["attempt_key"])
    parse = attempt.get("parse") if isinstance(attempt.get("parse"), Mapping) else {}
    parsed = parse.get("parsed_candidate")
    mapping = parsed.get("mapping") if isinstance(parsed, Mapping) else None
    parse_row = {
        "attempt_key": key,
        "json_valid": parse.get("json_valid") is True,
        "bank_schema_valid": parse.get("schema_valid") is True,
        "failure_reason": parse.get("failure_reason"),
        "parse_failure": parse.get("json_valid") is not True,
        "candidate_hash": sha256_bytes(canonical_json(parsed)) if parsed is not None else None,
    }
    coverage = _variable_coverage(
        key, attempt["source_formulation"], attempt["target_formulation"], mapping
    )
    reason: str | None = None
    canonical: JsonDict | None = None
    if parse.get("json_valid") is not True:
        reason = str(parse.get("failure_reason") or "parse_rejected")
    elif parse.get("schema_valid") is not True:
        reason = str(parse.get("failure_reason") or "bank_schema_rejected")
    else:
        try:
            source = fixture_exp.validate_formulation(attempt["source_formulation"])
            target = fixture_exp.validate_formulation(attempt["target_formulation"])
            canonical = fixture_exp.canonical_mapping(mapping, source, target)
        except (
            fixture_exp.MappingSchemaError,
            fixture_exp.FormulationSchemaError,
            ValueError,
        ) as exc:
            reason = str(exc)
    schema_row = {
        "attempt_key": key,
        "schema_valid": canonical is not None,
        "failure_reason": reason,
        "mapping_hash": sha256_bytes(canonical_json(mapping)) if mapping is not None else None,
        "frozen_candidate_unchanged": True,
    }
    return parse_row, schema_row, canonical


def engine_failure_row(pair_id: str, engine: str, status: str, reason: str) -> JsonDict:
    """Build a terminal engine row for parse, schema, timeout, or unknown outcomes."""

    return {
        "attempt_key": pair_id,
        "engine": engine,
        "status": status,
        "terminal": status in TERMINAL_ENGINE_STATUSES,
        "label": None,
        "failure_reason": reason,
        "forward_feasible": None,
        "reverse_feasible": None,
        "variable_coverage_complete": None,
        "objective_direction_valid": None,
        "objective_affine_preserved": None,
        "objective_order_preserved": None,
        "tie_count": 0,
        "source_feasible_count": None,
        "target_feasible_count": None,
        "witnesses": {},
        "counterexamples": {},
        "unknown_reasons": [reason] if status in {"timeout", "unknown"} else [],
    }


def _exact_text(value: Any) -> Any:
    """Serialize exact rational witness values without losing booleans or integers."""

    if isinstance(value, bool):
        return value
    exact = fixture_exp.as_fraction(value)
    if exact.denominator == 1:
        return exact.numerator
    return fixture_exp.fraction_text(exact)


def _json_assignment(assignment: Mapping[str, Any]) -> JsonDict:
    """Convert one exact assignment to stable JSON values."""

    return {name: _exact_text(value) for name, value in assignment.items()}


def _coerce_assignment(
    formulation: Mapping[str, Any], exact_values: Mapping[str, Fraction]
) -> JsonDict | None:
    """Admit only exact values of the target variable type and finite universe."""

    result: JsonDict = {}
    for variable in formulation["variables"]:
        name = variable["name"]
        value = exact_values[name]
        if variable["kind"] == "boolean":
            if value not in {Fraction(0), Fraction(1)}:
                return None
            typed: Any = bool(value)
        else:
            if value.denominator != 1:
                return None
            typed = int(value)
        if typed not in variable["universe"]:
            return None
        result[name] = typed
    return result


def _numeric_fraction(value: Any) -> Fraction:
    """Convert Boolean cardinality values to exact zero or one before arithmetic."""

    return Fraction(int(value)) if isinstance(value, bool) else fixture_exp.as_fraction(value)


def _forward_values(
    assignment: Mapping[str, Any], mapping: Mapping[str, Any]
) -> dict[str, Fraction]:
    """Apply the proposed affine rows without truncating rational results."""

    return {
        row["target"]: fixture_exp.as_fraction(row["scale"])
        * _numeric_fraction(assignment[row["source"]])
        + fixture_exp.as_fraction(row["offset"])
        for row in mapping["variables"]
    }


def _reverse_values(
    assignment: Mapping[str, Any], mapping: Mapping[str, Any]
) -> dict[str, Fraction]:
    """Invert each nonzero affine row exactly."""

    return {
        row["source"]: (
            _numeric_fraction(assignment[row["target"]]) - fixture_exp.as_fraction(row["offset"])
        )
        / fixture_exp.as_fraction(row["scale"])
        for row in mapping["variables"]
    }


def _order_holds(direction: str, left: Fraction, right: Fraction) -> bool:
    """Evaluate weak preference in the formulation's declared direction."""

    return left <= right if direction == "min" else left >= right


def certify_with_enumerator(pair: Mapping[str, Any]) -> JsonDict:
    """Exhaust every bounded assignment and every feasible objective pair."""

    key = str(pair["pair_id"])
    source = fixture_exp.validate_formulation(pair["source"])
    target = fixture_exp.validate_formulation(pair["target"])
    mapping = fixture_exp.canonical_mapping(pair["mapping"], source, target)
    source_feasible = fixture_exp._feasible_assignments(source)
    target_feasible = fixture_exp._feasible_assignments(target)
    witnesses: JsonDict = {
        "source_feasible": _json_assignment(source_feasible[0]) if source_feasible else None,
        "target_feasible": _json_assignment(target_feasible[0]) if target_feasible else None,
    }
    counterexamples: JsonDict = {}
    mapped_rows: list[tuple[JsonDict, JsonDict]] = []
    for source_assignment in source_feasible:
        exact_target = _forward_values(source_assignment, mapping)
        target_assignment = _coerce_assignment(target, exact_target)
        if target_assignment is None or not fixture_exp.is_feasible(target, target_assignment):
            counterexamples.setdefault(
                "forward_feasibility",
                {
                    "source": _json_assignment(source_assignment),
                    "mapped_target": _json_assignment(exact_target),
                },
            )
        else:
            mapped_rows.append((source_assignment, target_assignment))
    for target_assignment in target_feasible:
        exact_source = _reverse_values(target_assignment, mapping)
        source_assignment = _coerce_assignment(source, exact_source)
        if source_assignment is None or not fixture_exp.is_feasible(source, source_assignment):
            counterexamples.setdefault(
                "reverse_feasibility",
                {
                    "target": _json_assignment(target_assignment),
                    "inverse_source": _json_assignment(exact_source),
                },
            )
    forward_feasible = "forward_feasibility" not in counterexamples
    reverse_feasible = "reverse_feasibility" not in counterexamples
    direction_valid = _direction_valid(mapping, source, target)
    if not direction_valid:
        counterexamples["objective_direction"] = {
            "source_declared": mapping["objective"]["source_direction"],
            "source_actual": source["objective"]["direction"],
            "target_declared": mapping["objective"]["target_direction"],
            "target_actual": target["objective"]["direction"],
            "scale": mapping["objective"]["scale"],
        }
    objective_scale = fixture_exp.as_fraction(mapping["objective"]["scale"])
    objective_offset = fixture_exp.as_fraction(mapping["objective"]["offset"])
    valued_rows: list[tuple[JsonDict, JsonDict, Fraction, Fraction]] = []
    for source_assignment, target_assignment in mapped_rows:
        source_value = fixture_exp.objective_value(source, source_assignment)
        target_value = fixture_exp.objective_value(target, target_assignment)
        valued_rows.append((source_assignment, target_assignment, source_value, target_value))
        expected = objective_scale * source_value + objective_offset
        if target_value != expected:
            counterexamples.setdefault(
                "objective_affine",
                {
                    "source": _json_assignment(source_assignment),
                    "target": _json_assignment(target_assignment),
                    "source_value": fixture_exp.fraction_text(source_value),
                    "target_value": fixture_exp.fraction_text(target_value),
                    "expected_target_value": fixture_exp.fraction_text(expected),
                },
            )
    affine_preserved = "objective_affine" not in counterexamples
    tie_count = 0
    order_witness: JsonDict | None = None
    for left, right in product(valued_rows, repeat=2):
        source_order = _order_holds(source["objective"]["direction"], left[2], right[2])
        target_order = _order_holds(target["objective"]["direction"], left[3], right[3])
        if left[2] == right[2]:
            tie_count += 1
        current = {
            "source_left": _json_assignment(left[0]),
            "source_right": _json_assignment(right[0]),
            "target_left": _json_assignment(left[1]),
            "target_right": _json_assignment(right[1]),
            "source_values": [
                fixture_exp.fraction_text(left[2]),
                fixture_exp.fraction_text(right[2]),
            ],
            "target_values": [
                fixture_exp.fraction_text(left[3]),
                fixture_exp.fraction_text(right[3]),
            ],
            "source_order": source_order,
            "target_order": target_order,
        }
        if order_witness is None:
            order_witness = current
        if source_order != target_order:
            counterexamples.setdefault("objective_order", current)
            break
    if order_witness is not None:
        witnesses["objective_order"] = order_witness
    order_preserved = "objective_order" not in counterexamples
    equivalent = (
        forward_feasible
        and reverse_feasible
        and direction_valid
        and affine_preserved
        and order_preserved
    )
    return {
        "attempt_key": key,
        "engine": "python_exhaustive_bounded_enumerator_v2",
        "status": "proved" if equivalent else "counterexample",
        "terminal": True,
        "label": "equivalent" if equivalent else "non_equivalent",
        "failure_reason": None,
        "forward_feasible": forward_feasible,
        "reverse_feasible": reverse_feasible,
        "variable_coverage_complete": True,
        "objective_direction_valid": direction_valid,
        "objective_affine_preserved": affine_preserved,
        "objective_order_preserved": order_preserved,
        "tie_count": tie_count,
        "source_feasible_count": len(source_feasible),
        "target_feasible_count": len(target_feasible),
        "witnesses": witnesses,
        "counterexamples": counterexamples,
        "unknown_reasons": [],
    }


def _model_assignment(symbols: Mapping[str, Any], model: Any) -> JsonDict:
    """Read exact Boolean and integer values from one satisfying Z3 model."""

    result: JsonDict = {}
    for name, symbol in symbols.items():
        value = model.eval(symbol, model_completion=True)
        if fixture_exp.z3.is_bool(symbol):
            result[name] = fixture_exp.z3.is_true(value)
        else:
            result[name] = value.as_long()
    return result


def _symbol_domains(
    formulation: Mapping[str, Any], symbols: Mapping[str, Any]
) -> list[tuple[Any, list[Any]]]:
    """Return each free Z3 symbol with its finite values in lexical proof order."""

    return [
        (symbols[variable["name"]], sorted(variable["universe"]))
        for variable in formulation["variables"]
    ]


def _z3_query(
    *clauses: Any, lex_domains: Sequence[tuple[Any, Sequence[Any]]] = ()
) -> tuple[str, Any | None, str | None]:
    """Run one obligation and select the lexical first finite-domain witness."""

    solver = fixture_exp.z3.Solver()
    solver.set(timeout=Z3_TIMEOUT_MS)
    solver.add(*clauses)
    status = solver.check()
    if status == fixture_exp.z3.sat:
        for symbol, values in lex_domains:
            selected = None
            for value in values:
                solver.push()
                solver.add(symbol == value)
                feasible = solver.check() == fixture_exp.z3.sat
                solver.pop()
                if feasible:
                    selected = value
                    break
            if selected is not None:
                solver.add(symbol == selected)
        solver.check()
        return "sat", solver.model(), None
    if status == fixture_exp.z3.unsat:
        return "unsat", None, None
    reason = solver.reason_unknown()
    return ("timeout" if "timeout" in reason.lower() else "unknown"), None, reason


def certify_with_z3(pair: Mapping[str, Any]) -> JsonDict:
    """Search symbolic counterexamples for every frozen certificate obligation."""

    key = str(pair["pair_id"])
    if fixture_exp.z3 is None:  # pragma: no cover - preflight blocks this host state.
        return engine_failure_row(key, "z3", "unknown", "z3_unavailable")
    source = fixture_exp.validate_formulation(pair["source"])
    target = fixture_exp.validate_formulation(pair["target"])
    mapping = fixture_exp.canonical_mapping(pair["mapping"], source, target)
    stem = hashlib.sha256(key.encode()).hexdigest()[:12]
    source_symbols = fixture_exp._z3_variables(source, f"s_{stem}")
    target_symbols = fixture_exp._z3_variables(target, f"t_{stem}")
    source_feasible = fixture_exp._z3_feasible(source, source_symbols)
    target_feasible = fixture_exp._z3_feasible(target, target_symbols)
    map_clause = fixture_exp._z3_mapping(mapping, source_symbols, target_symbols)
    witnesses: JsonDict = {}
    counterexamples: JsonDict = {}
    unknown_reasons: list[str] = []

    forward_status, model, reason = _z3_query(
        source_feasible,
        fixture_exp.z3.Not(
            fixture_exp.z3.Exists(
                list(target_symbols.values()), fixture_exp.z3.And(map_clause, target_feasible)
            )
        ),
        lex_domains=_symbol_domains(source, source_symbols),
    )
    if reason:
        unknown_reasons.append(reason)
    if forward_status == "sat":
        source_assignment = _model_assignment(source_symbols, model)
        counterexamples["forward_feasibility"] = {
            "source": source_assignment,
            "mapped_target": _json_assignment(_forward_values(source_assignment, mapping)),
        }

    reverse_status, model, reason = _z3_query(
        target_feasible,
        fixture_exp.z3.Not(
            fixture_exp.z3.Exists(
                list(source_symbols.values()), fixture_exp.z3.And(map_clause, source_feasible)
            )
        ),
        lex_domains=_symbol_domains(target, target_symbols),
    )
    if reason:
        unknown_reasons.append(reason)
    if reverse_status == "sat":
        target_assignment = _model_assignment(target_symbols, model)
        counterexamples["reverse_feasibility"] = {
            "target": target_assignment,
            "inverse_source": _json_assignment(_reverse_values(target_assignment, mapping)),
        }

    direction_valid = _direction_valid(mapping, source, target)
    if not direction_valid:
        counterexamples["objective_direction"] = {
            "source_declared": mapping["objective"]["source_direction"],
            "source_actual": source["objective"]["direction"],
            "target_declared": mapping["objective"]["target_direction"],
            "target_actual": target["objective"]["direction"],
            "scale": mapping["objective"]["scale"],
        }
    source_objective = fixture_exp._z3_objective(source, source_symbols)
    target_objective = fixture_exp._z3_objective(target, target_symbols)
    affine_status, model, reason = _z3_query(
        source_feasible,
        target_feasible,
        map_clause,
        target_objective
        != fixture_exp._z3_number(mapping["objective"]["scale"]) * source_objective
        + fixture_exp._z3_number(mapping["objective"]["offset"]),
        lex_domains=_symbol_domains(source, source_symbols)
        + _symbol_domains(target, target_symbols),
    )
    if reason:
        unknown_reasons.append(reason)
    if affine_status == "sat":
        counterexamples["objective_affine"] = {
            "source": _model_assignment(source_symbols, model),
            "target": _model_assignment(target_symbols, model),
        }

    source_symbols_2 = fixture_exp._z3_variables(source, f"s2_{stem}")
    target_symbols_2 = fixture_exp._z3_variables(target, f"t2_{stem}")
    source_left = fixture_exp._z3_objective(source, source_symbols)
    source_right = fixture_exp._z3_objective(source, source_symbols_2)
    target_left = fixture_exp._z3_objective(target, target_symbols)
    target_right = fixture_exp._z3_objective(target, target_symbols_2)
    order_status, model, reason = _z3_query(
        source_feasible,
        target_feasible,
        map_clause,
        fixture_exp._z3_feasible(source, source_symbols_2),
        fixture_exp._z3_feasible(target, target_symbols_2),
        fixture_exp._z3_mapping(mapping, source_symbols_2, target_symbols_2),
        fixture_exp._z3_better(source["objective"]["direction"], source_left, source_right)
        != fixture_exp._z3_better(target["objective"]["direction"], target_left, target_right),
        lex_domains=_symbol_domains(source, source_symbols)
        + _symbol_domains(target, target_symbols)
        + _symbol_domains(source, source_symbols_2)
        + _symbol_domains(target, target_symbols_2),
    )
    if reason:
        unknown_reasons.append(reason)
    if order_status == "sat":
        counterexamples["objective_order"] = {
            "source_left": _model_assignment(source_symbols, model),
            "source_right": _model_assignment(source_symbols_2, model),
            "target_left": _model_assignment(target_symbols, model),
            "target_right": _model_assignment(target_symbols_2, model),
        }

    query_statuses = (forward_status, reverse_status, affine_status, order_status)
    nondecision = next(
        (status for status in query_statuses if status in {"timeout", "unknown"}), None
    )
    forward_feasible = forward_status == "unsat"
    reverse_feasible = reverse_status == "unsat"
    affine_preserved = affine_status == "unsat"
    order_preserved = order_status == "unsat"
    if nondecision is not None:
        status = nondecision
        label = None
    else:
        equivalent = (
            forward_feasible
            and reverse_feasible
            and direction_valid
            and affine_preserved
            and order_preserved
        )
        status = "proved" if equivalent else "counterexample"
        label = "equivalent" if equivalent else "non_equivalent"
    if not counterexamples:
        witnesses["proof_obligations"] = {
            "forward": forward_status,
            "reverse": reverse_status,
            "objective_affine": affine_status,
            "objective_order": order_status,
        }
    return {
        "attempt_key": key,
        "engine": "z3_symbolic_reformulation_checker_v2",
        "status": status,
        "terminal": status in TERMINAL_ENGINE_STATUSES,
        "label": label,
        "failure_reason": nondecision,
        "forward_feasible": forward_feasible if nondecision is None else None,
        "reverse_feasible": reverse_feasible if nondecision is None else None,
        "variable_coverage_complete": True,
        "objective_direction_valid": direction_valid,
        "objective_affine_preserved": affine_preserved if nondecision is None else None,
        "objective_order_preserved": order_preserved if nondecision is None else None,
        "tie_count": 0,
        "source_feasible_count": None,
        "target_feasible_count": None,
        "witnesses": witnesses,
        "counterexamples": counterexamples,
        "unknown_reasons": unknown_reasons,
        "query_statuses": {
            "forward_feasibility": forward_status,
            "reverse_feasibility": reverse_status,
            "objective_affine": affine_status,
            "objective_order": order_status,
        },
    }


def authority_agreement_row(
    attempt_key: str, enumeration: Mapping[str, Any], z3_row: Mapping[str, Any]
) -> JsonDict:
    """Require both labels and every comparable obligation to agree per proposal."""

    if z3_row["status"] in {"timeout", "unknown"}:
        agree = False
        reason = f"z3_{z3_row['status']}"
    elif enumeration["status"] in {"timeout", "unknown"}:
        agree = False
        reason = f"enumerator_{enumeration['status']}"
    elif enumeration["status"] in {"parse_rejected", "schema_rejected"}:
        agree = z3_row["status"] == enumeration["status"] and z3_row.get(
            "failure_reason"
        ) == enumeration.get("failure_reason")
        reason = None if agree else "authority_disagreement"
    else:
        fields = (
            "label",
            "forward_feasible",
            "reverse_feasible",
            "variable_coverage_complete",
            "objective_direction_valid",
            "objective_affine_preserved",
            "objective_order_preserved",
        )
        agree = all(enumeration.get(field) == z3_row.get(field) for field in fields)
        reason = None if agree else "authority_disagreement"
    terminal = (
        enumeration.get("status") in TERMINAL_ENGINE_STATUSES
        and z3_row.get("status") in TERMINAL_ENGINE_STATUSES
    )
    relation = (
        enumeration.get("label")
        if agree and enumeration.get("status") in {"proved", "counterexample"}
        else None
    )
    return {
        "attempt_key": attempt_key,
        "enumeration_status": enumeration.get("status"),
        "z3_status": z3_row.get("status"),
        "enumeration_label": enumeration.get("label"),
        "z3_label": z3_row.get("label"),
        "certified_relation": relation,
        "authorities_agree": agree,
        "quarantined": not agree,
        "reason": reason,
        "terminal": terminal,
    }


def _engine_evidence_rows(
    attempt_key: str, engine_row: Mapping[str, Any], evidence_field: str
) -> list[JsonDict]:
    """Flatten named witness or counterexample evidence without dropping engine identity."""

    evidence = engine_row.get(evidence_field, {})
    if not isinstance(evidence, Mapping):
        return []
    return [
        {
            "attempt_key": attempt_key,
            "engine": engine_row["engine"],
            "kind": kind,
            "evidence": deepcopy(value),
        }
        for kind, value in evidence.items()
        if value is not None
    ]


def certify_proposal(
    attempt: Mapping[str, Any],
    *,
    z3_certifier: Certifier = certify_with_z3,
    enumeration_certifier: Certifier = certify_with_enumerator,
) -> JsonDict:
    """Certify one frozen candidate and return every per-attempt evidence projection."""

    key = str(attempt["attempt_key"])
    parse_row, schema_row, mapping = _candidate_state(attempt)
    coverage = _variable_coverage(
        key,
        attempt["source_formulation"],
        attempt["target_formulation"],
        (
            attempt["parse"]["parsed_candidate"].get("mapping")
            if isinstance(attempt.get("parse"), Mapping)
            and isinstance(attempt["parse"].get("parsed_candidate"), Mapping)
            else None
        ),
    )
    if mapping is None:
        status = "parse_rejected" if parse_row["parse_failure"] else "schema_rejected"
        reason = str(schema_row["failure_reason"])
        enumeration = engine_failure_row(key, "enumerator", status, reason)
        z3_row = engine_failure_row(key, "z3", status, reason)
    else:
        pair = {
            "pair_id": key,
            "source": attempt["source_formulation"],
            "target": attempt["target_formulation"],
            "mapping": mapping,
        }
        enumeration = enumeration_certifier(pair)
        z3_row = z3_certifier(pair)
    agreement = authority_agreement_row(key, enumeration, z3_row)
    certified = agreement["certified_relation"]
    canonical_relation = attempt.get("canonical_relation")
    exact_correct = certified is not None and certified == canonical_relation
    false_acceptance = certified == "equivalent" and canonical_relation == "non_equivalent"
    false_rejection = certified == "non_equivalent" and canonical_relation == "equivalent"
    parse = attempt.get("parse") if isinstance(attempt.get("parse"), Mapping) else {}
    parsed = parse.get("parsed_candidate")
    claimed = None
    if isinstance(parsed, Mapping) and isinstance(parsed.get("mapping"), Mapping):
        claimed = parsed["mapping"].get("claimed_relation")
    baseline_relation = claimed if parse.get("schema_valid") is True else None
    baseline_correct = baseline_relation is not None and baseline_relation == canonical_relation
    confidence = parse.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        confidence = None
    rationale = parse.get("rationale") if isinstance(parse.get("rationale"), str) else None
    proposal_row = {
        "attempt_key": key,
        "hf_id": attempt.get("hf_id"),
        "model_family": attempt.get("model_family"),
        "pair_id": attempt.get("pair_id"),
        "problem_family": attempt.get("problem_family"),
        "prompt_variant_id": attempt.get("prompt_variant_id"),
        "difficulty": attempt.get("difficulty"),
        "canonical_relation": canonical_relation,
        "candidate_claimed_relation": claimed,
        "certified_relation": certified,
        "relation_classification": certified or "unclassified",
        "claim_matches_certificate": certified is not None and claimed == certified,
        "exact_mapping_correct": exact_correct,
        "false_acceptance": false_acceptance,
        "false_rejection": false_rejection,
        "parse_failure": parse_row["parse_failure"],
        "schema_failure": not schema_row["schema_valid"] and not parse_row["parse_failure"],
        "timeout": z3_row["status"] == "timeout" or enumeration["status"] == "timeout",
        "unknown": z3_row["status"] == "unknown" or enumeration["status"] == "unknown",
        "quarantined": agreement["quarantined"],
        "terminal": agreement["terminal"],
        "confidence": confidence,
        "rationale_present": bool(rationale and rationale.strip()),
        "baseline_relation": baseline_relation,
        "baseline_correct": baseline_correct,
        "raw_sha256": attempt.get("raw_sha256"),
        "candidate_hash": parse_row["candidate_hash"],
    }
    cross_row = {
        "attempt_key": key,
        "enumeration_forward_feasible": enumeration["forward_feasible"],
        "enumeration_reverse_feasible": enumeration["reverse_feasible"],
        "z3_forward_feasible": z3_row["forward_feasible"],
        "z3_reverse_feasible": z3_row["reverse_feasible"],
        "authorities_agree": (
            enumeration["forward_feasible"] == z3_row["forward_feasible"]
            and enumeration["reverse_feasible"] == z3_row["reverse_feasible"]
        ),
    }
    direction_row = {
        "attempt_key": key,
        "enumeration_direction_valid": enumeration["objective_direction_valid"],
        "z3_direction_valid": z3_row["objective_direction_valid"],
        "direction_valid": enumeration["objective_direction_valid"]
        if enumeration["objective_direction_valid"] == z3_row["objective_direction_valid"]
        else None,
    }
    order_row = {
        "attempt_key": key,
        "enumeration_objective_affine_preserved": enumeration["objective_affine_preserved"],
        "z3_objective_affine_preserved": z3_row["objective_affine_preserved"],
        "enumeration_objective_order_preserved": enumeration["objective_order_preserved"],
        "z3_objective_order_preserved": z3_row["objective_order_preserved"],
        "tie_count": enumeration["tie_count"],
    }
    confidence_row = {
        "attempt_key": key,
        "confidence": confidence,
        "exact_mapping_correct": exact_correct,
        "error": not exact_correct,
        "top_confidence": confidence is not None and confidence >= 0.9,
        "self_report_only": True,
    }
    rationale_row = {
        "attempt_key": key,
        "rationale": rationale,
        "rationale_present": proposal_row["rationale_present"],
        "exact_mapping_correct": exact_correct,
        "self_report_only": True,
    }
    baseline_row = {
        "attempt_key": key,
        "baseline": "frozen_syntax_only_claimed_relation",
        "baseline_relation": baseline_relation,
        "canonical_relation": canonical_relation,
        "baseline_correct": baseline_correct,
    }
    paired_row = {
        "attempt_key": key,
        "model_family": attempt.get("model_family"),
        "exact_mapping_correct": exact_correct,
        "baseline_correct": baseline_correct,
        "paired_accuracy_delta": int(exact_correct) - int(baseline_correct),
    }
    return {
        "proposal_row": proposal_row,
        "parse_row": parse_row,
        "schema_row": schema_row,
        "cross_feasibility_row": cross_row,
        "variable_coverage_row": coverage,
        "objective_direction_row": direction_row,
        "objective_order_row": order_row,
        "z3_row": z3_row,
        "enumeration_row": enumeration,
        "authority_agreement_row": agreement,
        "witness_rows": _engine_evidence_rows(key, enumeration, "witnesses")
        + _engine_evidence_rows(key, z3_row, "witnesses"),
        "counterexample_rows": _engine_evidence_rows(key, enumeration, "counterexamples")
        + _engine_evidence_rows(key, z3_row, "counterexamples"),
        "confidence_row": confidence_row,
        "rationale_row": rationale_row,
        "baseline_row": baseline_row,
        "paired_metric_row": paired_row,
    }


def synthetic_metric_row(
    key: str,
    canonical_relation: str,
    certified_relation: str | None,
    *,
    parse_failure: bool,
) -> JsonDict:
    """Create a small complete outcome row for metric reducer tests."""

    return {
        "attempt_key": key,
        "canonical_relation": canonical_relation,
        "certified_relation": certified_relation,
        "exact_mapping_correct": certified_relation == canonical_relation,
        "false_acceptance": certified_relation == "equivalent"
        and canonical_relation == "non_equivalent",
        "false_rejection": certified_relation == "non_equivalent"
        and canonical_relation == "equivalent",
        "parse_failure": parse_failure,
        "schema_failure": False,
        "timeout": False,
        "unknown": False,
        "quarantined": False,
        "terminal": True,
        "confidence": None,
        "rationale_present": False,
    }


def _safe_rate(numerator: int, denominator: int) -> float | None:
    """Return a measured rate or null when its denominator is empty."""

    return numerator / denominator if denominator else None


def metric_row(group_field: str, group_value: str, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce exact relation outcomes for one declared reporting group."""

    total = len(rows)
    correct = sum(row.get("exact_mapping_correct") is True for row in rows)
    false_acceptances = sum(row.get("false_acceptance") is True for row in rows)
    false_rejections = sum(row.get("false_rejection") is True for row in rows)
    canonical_negative = sum(row.get("canonical_relation") == "non_equivalent" for row in rows)
    canonical_positive = sum(row.get("canonical_relation") == "equivalent" for row in rows)
    result: JsonDict = {
        group_field: group_value,
        "proposal_count": total,
        "terminal_count": sum(row.get("terminal") is True for row in rows),
        "exact_mapping_correct_count": correct,
        "exact_mapping_accuracy": _safe_rate(correct, total),
        "certified_equivalent_count": sum(
            row.get("certified_relation") == "equivalent" for row in rows
        ),
        "certified_non_equivalent_count": sum(
            row.get("certified_relation") == "non_equivalent" for row in rows
        ),
        "unclassified_count": sum(row.get("certified_relation") is None for row in rows),
        "parse_failure_count": sum(row.get("parse_failure") is True for row in rows),
        "schema_failure_count": sum(row.get("schema_failure") is True for row in rows),
        "false_acceptance_count": false_acceptances,
        "false_acceptance_rate": _safe_rate(false_acceptances, canonical_negative),
        "false_rejection_count": false_rejections,
        "false_rejection_rate": _safe_rate(false_rejections, canonical_positive),
        "timeout_count": sum(row.get("timeout") is True for row in rows),
        "unknown_count": sum(row.get("unknown") is True for row in rows),
        "quarantine_count": sum(row.get("quarantined") is True for row in rows),
    }
    signal = calibration_row(group_field, group_value, rows)
    for field in (
        "confidence_count",
        "confidence_coverage",
        "confidence_ece",
        "confidence_brier",
        "confidence_auroc",
        "top_confidence_count",
        "top_confidence_error_rate",
        "rationale_present_count",
        "rationale_present_accuracy",
        "rationale_absent_count",
        "rationale_absent_accuracy",
        "rationale_accuracy_delta",
        "rationale_presence_auroc",
    ):
        result[field] = signal[field]
    return result


def tie_aware_auroc(rows: Sequence[tuple[float, bool]]) -> float | None:
    """Compute pairwise AUROC and award tied positive-negative scores half credit."""

    positives = [score for score, label in rows if label]
    negatives = [score for score, label in rows if not label]
    if not positives or not negatives:
        return None
    credit = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                credit += 1.0
            elif positive == negative:
                credit += 0.5
    return credit / (len(positives) * len(negatives))


def calibration_row(
    group_field: str, group_value: str, rows: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Measure advisory confidence and rationale signals without changing labels."""

    confidence_pairs = [
        (float(row["confidence"]), row.get("exact_mapping_correct") is True)
        for row in rows
        if isinstance(row.get("confidence"), (int, float))
        and not isinstance(row.get("confidence"), bool)
    ]
    brier = (
        sum((score - int(correct)) ** 2 for score, correct in confidence_pairs)
        / len(confidence_pairs)
        if confidence_pairs
        else None
    )
    ece = None
    if confidence_pairs:
        weighted_error = 0.0
        for index in range(5):
            lower = index / 5
            upper = (index + 1) / 5
            members = [
                (score, correct)
                for score, correct in confidence_pairs
                if lower <= score <= upper and (index == 4 or score < upper)
            ]
            if members:
                mean_confidence = sum(score for score, _ in members) / len(members)
                mean_accuracy = sum(correct for _, correct in members) / len(members)
                weighted_error += len(members) * abs(mean_confidence - mean_accuracy)
        ece = weighted_error / len(confidence_pairs)
    top = [(score, correct) for score, correct in confidence_pairs if score >= 0.9]
    rationale_present = [row for row in rows if row.get("rationale_present") is True]
    rationale_absent = [row for row in rows if row.get("rationale_present") is not True]
    present_accuracy = _safe_rate(
        sum(row.get("exact_mapping_correct") is True for row in rationale_present),
        len(rationale_present),
    )
    absent_accuracy = _safe_rate(
        sum(row.get("exact_mapping_correct") is True for row in rationale_absent),
        len(rationale_absent),
    )
    rationale_signal = [
        (
            1.0 if row.get("rationale_present") is True else 0.0,
            row.get("exact_mapping_correct") is True,
        )
        for row in rows
    ]
    return {
        group_field: group_value,
        "proposal_count": len(rows),
        "confidence_count": len(confidence_pairs),
        "confidence_coverage": _safe_rate(len(confidence_pairs), len(rows)),
        "confidence_ece": ece,
        "confidence_brier": brier,
        "confidence_auroc": tie_aware_auroc(confidence_pairs),
        "top_confidence_count": len(top),
        "top_confidence_error_rate": _safe_rate(sum(not correct for _, correct in top), len(top)),
        "rationale_present_count": len(rationale_present),
        "rationale_present_accuracy": present_accuracy,
        "rationale_absent_count": len(rationale_absent),
        "rationale_absent_accuracy": absent_accuracy,
        "rationale_accuracy_delta": (
            present_accuracy - absent_accuracy
            if present_accuracy is not None and absent_accuracy is not None
            else None
        ),
        "rationale_presence_auroc": tie_aware_auroc(rationale_signal),
        "signals_are_authority": False,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    """Select one deterministic nearest-rank bootstrap percentile."""

    ordered = sorted(values)
    index = int(probability * (len(ordered) - 1))
    return ordered[index]


def paired_bootstrap_interval(
    differences: Sequence[int], *, seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> JsonDict:
    """Return a deterministic paired percentile interval for binary accuracy deltas."""

    if not differences:
        return {"mean_delta": None, "ci95_lower": None, "ci95_upper": None}
    mean = sum(differences) / len(differences)
    if all(value == differences[0] for value in differences):
        constant = float(differences[0])
        return {"mean_delta": mean, "ci95_lower": constant, "ci95_upper": constant}
    rng = random.Random(seed)
    means = [
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(resamples)
    ]
    return {
        "mean_delta": mean,
        "ci95_lower": _percentile(means, 0.025),
        "ci95_upper": _percentile(means, 0.975),
    }


def reduce_positive_score(
    complete_score: int,
    confidence_intervals: Sequence[Mapping[str, Any]],
    pooled_false_acceptance_rate: float | None,
) -> int:
    """Apply the preregistered two-model paired-gain and false-acceptance gate."""

    improving = sum(
        isinstance(row.get("ci95_lower"), (int, float)) and row["ci95_lower"] > 0
        for row in confidence_intervals
    )
    return int(
        complete_score == 1
        and improving >= 2
        and pooled_false_acceptance_rate is not None
        and pooled_false_acceptance_rate < 0.01
    )


def _group_rows(
    rows: Sequence[Mapping[str, Any]], source_field: str, output_field: str
) -> list[JsonDict]:
    """Build stable metric rows for each observed value of one grouping field."""

    return [
        metric_row(
            output_field,
            value,
            [row for row in rows if str(row.get(source_field)) == value],
        )
        for value in sorted({str(row.get(source_field)) for row in rows})
    ]


def _calibration_groups(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Report monitoring signals overall and across every required grouping."""

    result = [calibration_row("scope", "overall", rows)]
    for source, output in (
        ("model_family", "model_family"),
        ("problem_family", "family"),
        ("prompt_variant_id", "prompt_variant_id"),
        ("difficulty", "difficulty"),
    ):
        for value in sorted({str(row.get(source)) for row in rows}):
            result.append(
                calibration_row(
                    output,
                    value,
                    [row for row in rows if str(row.get(source)) == value],
                )
            )
    return result


def _certificate_hash(bundle: Mapping[str, Any]) -> str:
    """Hash the exact per-proposal evidence used by fresh-process replay."""

    fields = (
        "proposal_row",
        "parse_row",
        "schema_row",
        "cross_feasibility_row",
        "variable_coverage_row",
        "objective_direction_row",
        "objective_order_row",
        "z3_row",
        "enumeration_row",
        "authority_agreement_row",
        "baseline_row",
        "paired_metric_row",
    )
    return sha256_bytes(canonical_json({field: bundle[field] for field in fields}))


def certify_inputs(inputs: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Run both authorities and derive every row and headline from serialized inputs."""

    bundles = [certify_proposal(row) for row in inputs]

    def rows_for(field: str) -> list[JsonDict]:
        return [deepcopy(bundle[field]) for bundle in bundles]

    proposal_rows = rows_for("proposal_row")
    model_rows = _group_rows(proposal_rows, "model_family", "model_family")
    family_rows = _group_rows(proposal_rows, "problem_family", "family")
    difficulty_rows = _group_rows(proposal_rows, "difficulty", "difficulty")
    prompt_rows = _group_rows(proposal_rows, "prompt_variant_id", "prompt_variant_id")
    overall_rows = [metric_row("scope", "overall", proposal_rows)]
    paired_rows = rows_for("paired_metric_row")
    model_families = sorted({str(row.get("model_family")) for row in paired_rows})
    confidence_intervals = []
    for index, model_family in enumerate(model_families):
        members = [row for row in paired_rows if row.get("model_family") == model_family]
        interval = paired_bootstrap_interval(
            [int(row["paired_accuracy_delta"]) for row in members],
            seed=RANDOM_SEED + index,
        )
        confidence_intervals.append(
            {
                "model_family": model_family,
                "paired_count": len(members),
                "resamples": BOOTSTRAP_RESAMPLES,
                **interval,
            }
        )
    pooled_far = overall_rows[0]["false_acceptance_rate"]
    terminal_complete = int(
        len(proposal_rows) == EXPECTED_PROPOSAL_COUNT
        and all(row["terminal"] for row in proposal_rows)
        and all(bundle["z3_row"]["terminal"] for bundle in bundles)
        and all(bundle["enumeration_row"]["terminal"] for bundle in bundles)
        and all(bundle["authority_agreement_row"]["terminal"] for bundle in bundles)
    )
    positive = reduce_positive_score(terminal_complete, confidence_intervals, pooled_far)
    result: JsonDict = {
        "rows": deepcopy(proposal_rows),
        "proposal_rows": proposal_rows,
        "parse_rows": rows_for("parse_row"),
        "schema_rows": rows_for("schema_row"),
        "cross_feasibility_rows": rows_for("cross_feasibility_row"),
        "variable_coverage_rows": rows_for("variable_coverage_row"),
        "objective_direction_rows": rows_for("objective_direction_row"),
        "objective_order_rows": rows_for("objective_order_row"),
        "z3_rows": rows_for("z3_row"),
        "enumeration_rows": rows_for("enumeration_row"),
        "authority_agreement_rows": rows_for("authority_agreement_row"),
        "witness_rows": [deepcopy(row) for bundle in bundles for row in bundle["witness_rows"]],
        "counterexample_rows": [
            deepcopy(row) for bundle in bundles for row in bundle["counterexample_rows"]
        ],
        "model_rows": model_rows,
        "family_rows": family_rows,
        "difficulty_rows": difficulty_rows,
        "prompt_variant_rows": prompt_rows,
        "overall_metric_rows": overall_rows,
        "confidence_rows": rows_for("confidence_row"),
        "rationale_rows": rows_for("rationale_row"),
        "calibration_rows": _calibration_groups(proposal_rows),
        "baseline_rows": rows_for("baseline_row"),
        "paired_metric_rows": paired_rows,
        "confidence_interval_rows": confidence_intervals,
        "terminal_complete_score": terminal_complete,
        "positive_score": positive,
        "pooled_false_acceptance_rate": pooled_far,
        "certificate_hashes": {
            str(bundle["proposal_row"]["attempt_key"]): _certificate_hash(bundle)
            for bundle in bundles
        },
    }
    headline = {
        "overall_metric_rows": result["overall_metric_rows"],
        "model_rows": result["model_rows"],
        "family_rows": result["family_rows"],
        "difficulty_rows": result["difficulty_rows"],
        "prompt_variant_rows": result["prompt_variant_rows"],
        "confidence_interval_rows": result["confidence_interval_rows"],
        "terminal_complete_score": terminal_complete,
        "positive_score": positive,
        "pooled_false_acceptance_rate": pooled_far,
    }
    result["headline_hash"] = sha256_bytes(canonical_json(headline))
    return result


def serialize_attempt_input(attempt: Mapping[str, Any]) -> JsonDict:
    """Keep only frozen fields needed to replay one certificate and its metrics."""

    fields = (
        "attempt_key",
        "hf_id",
        "model_family",
        "pair_id",
        "problem_family",
        "prompt_variant_id",
        "raw_sha256",
        "source_formulation",
        "target_formulation",
        "parse",
        "canonical_relation",
        "difficulty",
    )
    return {field: deepcopy(attempt.get(field)) for field in fields}


def deserialize_fixture_pair(*, equivalent: bool) -> JsonDict:
    """Return one generated fixture pair for small replay-interface tests."""

    label = "equivalent" if equivalent else "non_equivalent"
    return next(
        deepcopy(pair)
        for pair in fixture_exp.generate_pairs(fixture_exp.RANDOM_SEED)
        if pair["expected_label"] == label
    )


def _raw_roster_valid(bank: Mapping[str, Any]) -> JsonDict:
    """Validate all frozen raw hashes and one-to-one attempt identities."""

    attempt_rows = bank.get("attempt_rows", [])
    raw_rows = bank.get("raw_output_rows", [])
    if not isinstance(attempt_rows, list) or not isinstance(raw_rows, list):
        return {"count": 0, "unique": False, "hashes_match": False, "terminal": False}
    raw_by_key = {str(row.get("attempt_key")): row for row in raw_rows if isinstance(row, Mapping)}
    hashes_match = True
    for row in attempt_rows:
        if not isinstance(row, Mapping):
            hashes_match = False
            continue
        key = str(row.get("attempt_key"))
        raw = raw_by_key.get(key)
        if raw is None:
            hashes_match = False
            continue
        text = str(row.get("raw_text", ""))
        expected_hash = sha256_bytes(text.encode())
        if row.get("raw_sha256") != expected_hash or raw.get("raw_sha256") != expected_hash:
            hashes_match = False
        if raw.get("raw_text") != row.get("raw_text"):
            hashes_match = False
    keys = [str(row.get("attempt_key")) for row in attempt_rows if isinstance(row, Mapping)]
    return {
        "count": len(attempt_rows),
        "raw_count": len(raw_rows),
        "unique": len(keys) == len(set(keys)) == len(raw_by_key),
        "hashes_match": hashes_match,
        "terminal": all(
            isinstance(row, Mapping) and row.get("terminal") is True for row in attempt_rows
        ),
    }


def _fixture_binding_valid(
    bank: Mapping[str, Any], fixture: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> JsonDict:
    """Check that bank formulations exactly match their checkpoint pair bytes."""

    pairs = checkpoint.get("pairs", [])
    pair_by_id = {str(row.get("pair_id")): row for row in pairs if isinstance(row, Mapping)}
    matched = 0
    for attempt in bank.get("attempt_rows", []):
        if not isinstance(attempt, Mapping):
            continue
        pair = pair_by_id.get(str(attempt.get("pair_id")))
        if pair is None:
            continue
        if canonical_json(attempt.get("source_formulation")) == canonical_json(
            pair.get("source")
        ) and canonical_json(attempt.get("target_formulation")) == canonical_json(
            pair.get("target")
        ):
            matched += 1
    return {
        "fixture_ready_score": fixture.get("reformulation_fixture_ready_score"),
        "fixture_pair_count": len(pairs) if isinstance(pairs, list) else 0,
        "matched_attempt_count": matched,
    }


def _writable(path: Path) -> JsonDict:
    """Probe the replay directory without changing the requested checkpoint."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".exp6957-", delete=True):
            pass
        return {"writable": True, "error": None}
    except OSError as exc:  # pragma: no cover - exercised only on a broken host filesystem.
        return {"writable": False, "error": type(exc).__name__}


def collect_preconditions(
    *,
    bank: Mapping[str, Any],
    fixture: Mapping[str, Any],
    fixture_checkpoint: Mapping[str, Any],
    bank_path: Path,
    fixture_path: Path,
    fixture_checkpoint_path: Path,
    replay_checkpoint_path: Path,
) -> list[JsonDict]:
    """Check frozen inputs, exact authorities, and replay storage before labels load."""

    raw = _raw_roster_valid(bank)
    binding = _fixture_binding_valid(bank, fixture, fixture_checkpoint)
    source_hashes = bank.get("source_artifact_hashes", {})
    source_hashes_complete = (
        isinstance(source_hashes, Mapping)
        and bool(source_hashes)
        and all(
            isinstance(value, str) and value.startswith("sha256:")
            for value in source_hashes.values()
        )
    )
    bank_fixture_hash = source_hashes.get("fixture_artifact") if source_hashes_complete else None
    bank_checkpoint_hash = (
        source_hashes.get("fixture_checkpoint") if source_hashes_complete else None
    )
    enumerator_status: str | None = None
    pairs = fixture_checkpoint.get("pairs", [])
    if isinstance(pairs, list) and pairs:
        try:
            enumerator_status = certify_with_enumerator(pairs[0])["status"]
        except (
            KeyError,
            TypeError,
            ValueError,
        ):  # pragma: no cover - malformed fixtures fail closed.
            enumerator_status = "error"
    writable = _writable(replay_checkpoint_path)
    return [
        gate_check(
            "reformulation_bank_complete_score", 1, bank.get("reformulation_bank_complete_score")
        ),
        gate_check(
            "frozen_attempt_roster",
            {
                "count": EXPECTED_PROPOSAL_COUNT,
                "raw_count": EXPECTED_PROPOSAL_COUNT,
                "unique": True,
                "hashes_match": True,
                "terminal": True,
            },
            raw,
        ),
        gate_check("reformulation_fixture_ready_score", 1, binding["fixture_ready_score"]),
        gate_check(
            "exact_fixture_pair_count",
            fixture_exp.EXPECTED_ROW_COUNT,
            binding["fixture_pair_count"],
        ),
        gate_check(
            "bank_fixture_attempt_binding",
            EXPECTED_PROPOSAL_COUNT,
            binding["matched_attempt_count"],
        ),
        gate_check("fixture_artifact_hash", bank_fixture_hash, sha256_path(fixture_path)),
        gate_check(
            "fixture_checkpoint_hash", bank_checkpoint_hash, sha256_path(fixture_checkpoint_path)
        ),
        gate_check("bank_source_artifact_hashes_complete", True, source_hashes_complete),
        gate_check("z3_available", True, fixture_exp.z3 is not None),
        gate_check(
            "bounded_enumerator_support",
            True,
            enumerator_status in {"proved", "counterexample"},
        ),
        gate_check("writable_replay_checkpoint", True, writable["writable"]),
        gate_check("bank_artifact_present", True, bank_path.is_file()),
    ]


def source_artifact_hashes(
    repo_root: Path, bank_path: Path, fixture_path: Path, fixture_checkpoint_path: Path
) -> JsonDict:
    """Bind the result to frozen inputs and all code that interprets them."""

    paths = {
        "bank_artifact": bank_path,
        "fixture_artifact": fixture_path,
        "fixture_checkpoint": fixture_checkpoint_path,
        "verification_spec": repo_root / SPEC_PATH,
        "module": repo_root / MODULE_PATH,
        "test": repo_root / TEST_PATH,
        "wrapper": repo_root / WRAPPER_PATH,
        "fixture_module": repo_root / "python/carnot/experiment_6955_reformulation_fixture.py",
        "bank_module": repo_root
        / "python/carnot/experiment_6956_three_family_reformulation_bank.py",
    }
    return {name: {"path": str(path), "sha256": sha256_path(path)} for name, path in paths.items()}


def _empty_row_fields() -> JsonDict:
    """Return every required row family for a complete blocked artifact."""

    return {
        field: []
        for field in REQUIRED_ARTIFACT_FIELDS
        if field == "rows" or field.endswith("_rows")
    }


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable scientific content while excluding duration and the digest itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return sha256_bytes(canonical_json(payload))


def build_blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
) -> JsonDict:
    """Build the full fail-closed schema when any prerequisite is absent."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": 6957,
        "run_date": run_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 9),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        **_empty_row_fields(),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "smt_certification_run_complete_score": 0,
        "sota_mapping_positive_score": 0,
        "gate_check_summary": gate_summary(preconditions_checked),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_smt_mapping_certification",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _build_serialized_inputs(
    bank: Mapping[str, Any], fixture_checkpoint: Mapping[str, Any]
) -> list[JsonDict]:
    """Attach canonical relations only after all certification rules and gates are fixed."""

    pair_by_id = {
        str(row["pair_id"]): row for row in fixture_checkpoint["pairs"] if isinstance(row, Mapping)
    }
    inputs = []
    for attempt in bank["attempt_rows"]:
        pair = pair_by_id[str(attempt["pair_id"])]
        row = {
            **serialize_attempt_input(attempt),
            "canonical_relation": pair["expected_label"],
            "difficulty": pair["difficulty"],
        }
        inputs.append(row)
    return inputs


def replay_checkpoint(path: Path) -> JsonDict:
    """Read serialized proposal inputs and recompute exact certificates and headlines."""

    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ValueError("replay_schema_version")
    inputs = checkpoint.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("replay_inputs")
    return certify_inputs(inputs)


def _fresh_process_replay(repo_root: Path, checkpoint_path: Path) -> JsonDict:
    """Launch a clean interpreter whose only scientific input is the checkpoint."""

    output_path = checkpoint_path.with_suffix(".replay.json")
    environment = dict(os.environ)
    python_root = str(repo_root / "python")
    environment["PYTHONPATH"] = python_root + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    command = [
        sys.executable,
        "-m",
        "carnot.experiment_6957_smt_mapping_certification",
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
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:  # pragma: no cover - a child crash is an integration failure.
        raise RuntimeError(f"fresh_process_replay_failed:{completed.stderr[-500:]}")
    result = json.loads(output_path.read_text(encoding="utf-8"))
    output_path.unlink()
    return result


def _read_json(path: Path) -> JsonDict:
    """Read one required JSON object with a stable type boundary."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def _completion_score(data: Mapping[str, Any], replay_rows: Sequence[Mapping[str, Any]]) -> int:
    """Require every declared terminal row and every fresh replay match."""

    per_attempt_fields = (
        "proposal_rows",
        "parse_rows",
        "schema_rows",
        "cross_feasibility_rows",
        "variable_coverage_rows",
        "objective_direction_rows",
        "objective_order_rows",
        "z3_rows",
        "enumeration_rows",
        "authority_agreement_rows",
        "baseline_rows",
        "paired_metric_rows",
    )
    return int(
        all(len(data.get(field, [])) == EXPECTED_PROPOSAL_COUNT for field in per_attempt_fields)
        and data.get("terminal_complete_score") == 1
        and len(replay_rows) == EXPECTED_PROPOSAL_COUNT
        and all(row.get("replay_matches") is True for row in replay_rows)
        and all(row.get("headline_metrics_match") is True for row in replay_rows)
    )


def _outcome(complete: int, positive: int) -> tuple[str, str]:
    """Map the two independent gates to the closed terminal verdict vocabulary."""

    if not complete:
        return "partial", "partial_smt_mapping_certification"
    if positive:
        return "positive", "complete_positive_sota_mapping_certification"
    return "null", "complete_null_sota_mapping_certification"


def build_from_paths(
    *,
    run_date: str,
    repo_root: Path = REPO_ROOT,
    bank_path: Path | None = None,
    fixture_path: Path | None = None,
    fixture_checkpoint_path: Path | None = None,
    replay_checkpoint_path: Path | None = None,
) -> JsonDict:
    """Preflight, certify, replay, and reduce the frozen proposal bank."""

    started = time.perf_counter()
    root = Path(repo_root)
    bank_file = Path(bank_path) if bank_path is not None else root / BANK_PATH
    fixture_file = Path(fixture_path) if fixture_path is not None else root / FIXTURE_PATH
    fixture_checkpoint_file = (
        Path(fixture_checkpoint_path)
        if fixture_checkpoint_path is not None
        else root / FIXTURE_CHECKPOINT_PATH
    )
    replay_file = (
        Path(replay_checkpoint_path)
        if replay_checkpoint_path is not None
        else root / REPLAY_CHECKPOINT_PATH
    )
    hashes = source_artifact_hashes(root, bank_file, fixture_file, fixture_checkpoint_file)
    try:
        bank = _read_json(bank_file)
        fixture = _read_json(fixture_file)
        fixture_checkpoint = _read_json(fixture_checkpoint_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        checks = [
            gate_check(
                "frozen_input_readable",
                {"readable": True},
                {"readable": False, "error": type(exc).__name__},
            )
        ]
        return build_blocked_artifact(
            run_date=run_date,
            duration_s=max(time.perf_counter() - started, 0.000001),
            preconditions_checked=checks,
            source_artifact_hashes=hashes,
        )
    checks = collect_preconditions(
        bank=bank,
        fixture=fixture,
        fixture_checkpoint=fixture_checkpoint,
        bank_path=bank_file,
        fixture_path=fixture_file,
        fixture_checkpoint_path=fixture_checkpoint_file,
        replay_checkpoint_path=replay_file,
    )
    if any(row["passed"] is not True for row in checks):
        return build_blocked_artifact(
            run_date=run_date,
            duration_s=max(time.perf_counter() - started, 0.000001),
            preconditions_checked=checks,
            source_artifact_hashes=hashes,
        )
    inputs = _build_serialized_inputs(bank, fixture_checkpoint)
    replay_payload = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "random_seed": RANDOM_SEED,
        "inputs": inputs,
    }
    write_json_atomic(replay_file, replay_payload)
    parent = certify_inputs(inputs)
    child = _fresh_process_replay(root, replay_file)
    replay_rows = [
        {
            "attempt_key": row["attempt_key"],
            "parent_certificate_hash": parent["certificate_hashes"].get(row["attempt_key"]),
            "child_certificate_hash": child["certificate_hashes"].get(row["attempt_key"]),
            "replay_matches": parent["certificate_hashes"].get(row["attempt_key"])
            == child["certificate_hashes"].get(row["attempt_key"]),
            "parent_headline_hash": parent["headline_hash"],
            "child_headline_hash": child["headline_hash"],
            "headline_metrics_match": parent["headline_hash"] == child["headline_hash"],
            "fresh_process": True,
        }
        for row in parent["proposal_rows"]
    ]
    complete = _completion_score(parent, replay_rows)
    positive = reduce_positive_score(
        complete,
        parent["confidence_interval_rows"],
        parent["pooled_false_acceptance_rate"],
    )
    verdict_class, honest_verdict = _outcome(complete, positive)
    excluded_parent_fields = {
        "terminal_complete_score",
        "positive_score",
        "pooled_false_acceptance_rate",
        "certificate_hashes",
        "headline_hash",
    }
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": 6957,
        "run_date": run_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(time.perf_counter() - started, 0.000001), 9),
        "source_artifact_hashes": hashes,
        **{key: value for key, value in parent.items() if key not in excluded_parent_fields},
        "fresh_process_replay_rows": replay_rows,
        "replay_checkpoint_path": str(
            replay_file.relative_to(root) if replay_file.is_relative_to(root) else replay_file
        ),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "smt_certification_run_complete_score": complete,
        "sota_mapping_positive_score": positive,
        "gate_check_summary": [],
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute schema, terminal coverage, score, verdict, and stable digest checks."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_ARTIFACT_FIELDS) <= set(principles):
        errors.append("field_principles_incomplete")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    verdict_class = artifact.get("verdict_class")
    blocked = verdict_class == "blocked"
    if not blocked:
        for field in (
            "proposal_rows",
            "parse_rows",
            "schema_rows",
            "cross_feasibility_rows",
            "variable_coverage_rows",
            "objective_direction_rows",
            "objective_order_rows",
            "z3_rows",
            "enumeration_rows",
            "authority_agreement_rows",
            "baseline_rows",
            "paired_metric_rows",
            "fresh_process_replay_rows",
        ):
            if len(artifact.get(field, [])) != EXPECTED_PROPOSAL_COUNT:
                errors.append(field.removesuffix("s") + "_count")
        terminal_complete = int(
            len(artifact.get("proposal_rows", [])) == EXPECTED_PROPOSAL_COUNT
            and all(row.get("terminal") is True for row in artifact.get("proposal_rows", []))
            and len(artifact.get("fresh_process_replay_rows", [])) == EXPECTED_PROPOSAL_COUNT
            and all(
                row.get("replay_matches") is True and row.get("headline_metrics_match") is True
                for row in artifact.get("fresh_process_replay_rows", [])
            )
        )
        if artifact.get("smt_certification_run_complete_score") != terminal_complete:
            errors.append("completion_score_mismatch")
        overall = metric_row("scope", "overall", artifact.get("proposal_rows", []))
        intervals = artifact.get("confidence_interval_rows", [])
        expected_positive = reduce_positive_score(
            terminal_complete, intervals, overall["false_acceptance_rate"]
        )
        if artifact.get("sota_mapping_positive_score") != expected_positive:
            errors.append("positive_score_mismatch")
    elif (
        artifact.get("smt_certification_run_complete_score") != 0
        or artifact.get("sota_mapping_positive_score") != 0
    ):
        errors.append("blocked_scores_nonzero")
    score = artifact.get("smt_certification_run_complete_score")
    positive = artifact.get("sota_mapping_positive_score")
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict_class == "blocked" and not verdict.startswith("blocked_"):
        errors.append("honest_verdict_prefix_mismatch")
    if verdict_class == "partial" and not verdict.startswith("partial_"):
        errors.append("honest_verdict_prefix_mismatch")
    if verdict_class in {"positive", "null"} and not verdict.startswith("complete_"):
        errors.append("honest_verdict_prefix_mismatch")
    if positive == 1 and verdict_class != "positive":
        errors.append("positive_verdict_class_mismatch")
    if score == 1 and positive == 0 and verdict_class != "null":
        errors.append("null_verdict_class_mismatch")
    if payload_checksum(artifact) != artifact.get("reproducibility_checksum"):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def run(
    *,
    date: str,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    replay_checkpoint_path: Path | None = None,
) -> JsonDict:
    """Build, validate, and write the required terminal artifact."""

    root = Path(repo_root)
    output = Path(output_path) if output_path is not None else root / RESULT_PATH
    artifact = build_from_paths(
        run_date=date,
        repo_root=root,
        replay_checkpoint_path=replay_checkpoint_path,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact_validation:" + ",".join(errors))
    write_json_atomic(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Expose the required run command and private fresh-process replay mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260903")
    parser.add_argument("--replay-checkpoint", type=Path)
    parser.add_argument("--replay-output", type=Path)
    args = parser.parse_args(argv)
    if args.replay_checkpoint is not None:
        if args.replay_output is None:
            parser.error("--replay-output is required with --replay-checkpoint")
        write_json_atomic(args.replay_output, replay_checkpoint(args.replay_checkpoint))
        return 0
    artifact = run(date=args.date)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "proposal_count": len(artifact["proposal_rows"]),
                "smt_certification_run_complete_score": artifact[
                    "smt_certification_run_complete_score"
                ],
                "sota_mapping_positive_score": artifact["sota_mapping_positive_score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the required wrapper.
    raise SystemExit(main())
