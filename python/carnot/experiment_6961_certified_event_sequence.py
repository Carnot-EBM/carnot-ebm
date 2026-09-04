"""Build a sealed prospective sequence from prior exact mapping certificates.

Spec refs: REQ-LEARN-6961 and SCENARIO-LEARN-6961-*.

This module prepares measurement opportunities. It does not run a language
model and does not predict which memory arm will win. Exact engines label the
sealed outcomes only after each prompt boundary has been built.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_6957_smt_mapping_certification as cert_exp


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
CERTIFICATION_PATH = Path("results/experiment_6957_smt_mapping_certification.json")
FIXTURE_PATH = Path("results/experiment_6955_reformulation_fixture.json")
CERTIFICATION_REPLAY_PATH = Path(
    "results/checkpoints/experiment_6957_smt_mapping_certification_inputs.json"
)
SEALED_CHECKPOINT_PATH = Path("results/checkpoints/experiment_6961_certified_event_sequence.json")
RESULT_PATH = Path("results/experiment_6961_certified_event_sequence.json")
MODULE_PATH = Path("python/carnot/experiment_6961_certified_event_sequence.py")
TEST_PATH = Path("tests/python/test_experiment_6961_certified_event_sequence.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6961_certified_event_sequence.py")
MEMORY_PATH = Path("python/carnot/learn/constraint_memory.py")

SCHEMA_VERSION = "carnot.exp6961.certified_event_sequence.v1"
CHECKPOINT_SCHEMA_VERSION = "carnot.exp6961.sealed_generator_inputs.v1"
INFERENCE_SUBSTRATE = "deterministic_sealed_exact_certificate_sequence_no_llm"
RANDOM_SEED = 696120260904
HEADLINE_MODEL_FAMILIES = ("qwen3.6_moe", "gemma4_dense", "gemma4_moe")
PROBLEM_FAMILIES = (
    "bounded_integer_linear",
    "boolean_cardinality",
    "bounded_piecewise_linear",
)
ARMS = ("no_memory", "fifo", "queue")
EVENTS_PER_MODEL_FAMILY = 24
EXPECTED_EVENT_COUNT = EVENTS_PER_MODEL_FAMILY * len(HEADLINE_MODEL_FAMILIES)
SEED_EVENT_COUNT = 6
MIN_LATER_OPPORTUNITY_PER_FAMILY = 12
RETRIEVAL_LIMIT = 3

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "seed_certificate_rows",
    "event_rows",
    "chronology_rows",
    "family_rows",
    "transformation_rows",
    "retrieval_key_rows",
    "eligible_prior_rows",
    "prohibited_future_rows",
    "similarity_rows",
    "reusable_factor_rows",
    "conflict_rows",
    "distractor_rows",
    "correction_rows",
    "retention_probe_rows",
    "opportunity_rows",
    "headroom_rows",
    "leakage_rows",
    "split_rows",
    "sealed_checkpoint_path",
    "fresh_process_replay_rows",
    "random_seed",
    "reproducibility_checksum",
    "certified_event_sequence_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "A reason for each field makes the evidence contract auditable.",
    "preconditions_checked": "Fail-closed checks stop incomplete exact inputs before sequence generation.",
    "inference_substrate": "The declaration separates deterministic planning from later model inference.",
    "duration_s": "Measured wall time shows that generation and fresh replay both executed.",
    "source_artifact_hashes": "Hashes bind the sequence to exact source evidence and interpreting code.",
    "rows": "One compact row per event lets verdict checks recompute the sequence headline.",
    "seed_certificate_rows": "Sanitized exact rows prove that self-reports did not authorize seed memory.",
    "event_rows": "Complete sealed events preserve prompts, outcomes, and post-event certificates.",
    "chronology_rows": "Explicit ordinals make time reversal mechanically detectable.",
    "family_rows": "Per-model-family counts prevent pooled opportunity from hiding a weak family.",
    "transformation_rows": "Named changes prove that later formulations are related but not identical.",
    "retrieval_key_rows": "Frozen answer-free keys make family separation inspectable.",
    "eligible_prior_rows": "Eligibility rows prove that retrieval sees only completed earlier events.",
    "prohibited_future_rows": "Future rosters make forbidden prompt material explicit.",
    "similarity_rows": "Precomputed similarity measures opportunity without using an outcome label.",
    "reusable_factor_rows": "Factor summaries expose what memory can reuse without copying an answer.",
    "conflict_rows": "Exact conflict classes measure interference and contradiction risk.",
    "distractor_rows": "Irrelevant FIFO selections stay visible instead of becoming claimed memory.",
    "correction_rows": "Delayed correction links test whether stale conflicts can be superseded.",
    "retention_probe_rows": "Late probes test whether old exact factors remain retrievable.",
    "opportunity_rows": "Matched arm rows measure information access without inventing model outcomes.",
    "headroom_rows": "Structural headroom distinguishes a real memory lever from a saturated fixture.",
    "leakage_rows": "Per-event checks prove that prompts exclude labels, futures, and answer mappings.",
    "split_rows": "A frozen boundary keeps training seeds separate from prospective evaluation.",
    "sealed_checkpoint_path": "A durable generator-input seal fixes opportunity and order before inference.",
    "fresh_process_replay_rows": "Independent process hashes detect hidden state and serialization drift.",
    "random_seed": "A fixed seed makes all schedule and surface choices repeatable.",
    "reproducibility_checksum": "A timing-free digest detects any later scientific-content drift.",
    "certified_event_sequence_ready_score": "The binary gate opens only for complete safe opportunity.",
    "gate_check_summary": "Expected and observed values make every blocked result actionable.",
    "verifier_is_oracle": "True limits the positive class to exact sequence conformance, not memory value.",
    "verdict_class": "A closed class separates conformance, null opportunity, and blocked inputs.",
    "honest_verdict": "A stable terminal prefix prevents automation from overstating the result.",
}

_ALLOWED_VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
_FORBIDDEN_PROMPT_KEYS = {
    "answer_mapping",
    "candidate_mapping",
    "certificate_mapping",
    "certified_relation",
    "exact_outcome",
    "expected_label",
    "gold_label",
    "later_outcome",
}


def canonical_json(value: Any) -> bytes:
    """Return stable bytes so every seal has one cross-process spelling."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return one SHA-256 digest with the repository prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON rather than formatting-dependent source text."""

    return sha256_bytes(canonical_json(value))


def sha256_path(path: Path) -> str | None:
    """Hash a file while preserving absence as an explicit failed value."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record both sides of one exact gate comparison."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": expected == observed,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep all checks and make the first failed comparison easy to find."""

    copied = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in copied if row.get("passed") is not True), None)
    return {
        "checks": copied,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def write_json_atomic(path: Path, value: Any) -> None:
    """Sync bytes and atomically replace the target so a seal is never partial."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _fraction_text(value: Fraction | int) -> str:
    """Write exact rational values without floating-point drift."""

    fraction = Fraction(value)
    return str(fraction.numerator) if fraction.denominator == 1 else str(fraction)


def _variable(name: str, lower: int, upper: int) -> JsonDict:
    """Build one bounded integer variable with explicit outside-domain probes."""

    return {
        "name": name,
        "kind": "integer",
        "domain": {"lower": str(lower), "upper": str(upper)},
        "universe": list(range(lower - 1, upper + 2)),
    }


def _boolean_variable(name: str) -> JsonDict:
    """Build one Boolean variable in the fixture's exact schema."""

    return {
        "name": name,
        "kind": "boolean",
        "domain": {"lower": None, "upper": None},
        "universe": [False, True],
    }


def _transform_terms(
    terms: Mapping[str, str], mapping_rows: Sequence[Mapping[str, str]], multiplier: int = 1
) -> tuple[JsonDict, Fraction]:
    """Substitute target coordinates and return terms plus the moved constant."""

    by_source = {str(row["source"]): row for row in mapping_rows}
    transformed: JsonDict = {}
    moved = Fraction(0)
    for source, coefficient_text in terms.items():
        row = by_source[source]
        coefficient = Fraction(coefficient_text) * multiplier
        scale = Fraction(row["scale"])
        offset = Fraction(row["offset"])
        transformed[str(row["target"])] = _fraction_text(coefficient / scale)
        moved += coefficient * offset / scale
    return transformed, moved


def _integer_pair(
    ordinal: int, problem_family: str, conflict_class: str | None
) -> tuple[JsonDict, JsonDict, JsonDict]:
    """Create one exact affine integer pair with a linear or piecewise objective."""

    source_names = (f"s{ordinal}_alpha", f"s{ordinal}_beta")
    target_names = (f"t{ordinal}_north", f"t{ordinal}_south")
    lower = (ordinal % 3 - 1, (ordinal // 3) % 2)
    upper = (lower[0] + 2 + ordinal % 2, lower[1] + 2)
    scales = (-1 if ordinal % 2 else 1, -1 if ordinal % 3 == 0 else 1)
    offsets = (ordinal % 5 - 2, (ordinal * 2) % 5 - 1)
    mapping_rows = [
        {
            "source": source_names[0],
            "target": target_names[1],
            "scale": str(scales[0]),
            "offset": str(offsets[0]),
        },
        {
            "source": source_names[1],
            "target": target_names[0],
            "scale": str(scales[1]),
            "offset": str(offsets[1]),
        },
    ]
    target_bounds = [
        (
            scales[index] * lower[index] + offsets[index],
            scales[index] * upper[index] + offsets[index],
        )
        for index in range(2)
    ]
    target_bounds = [(min(pair), max(pair)) for pair in target_bounds]
    target_bound_by_name = {
        target_names[1]: target_bounds[0],
        target_names[0]: target_bounds[1],
    }
    a0, a1 = 1 + ordinal % 3, 1 + ordinal % 2
    rhs = a0 * lower[0] + a1 * lower[1] + max(1, (a0 + a1) // 2)
    source_constraints = [
        {
            "op": "<=",
            "rhs": str(rhs),
            "terms": {source_names[0]: str(a0), source_names[1]: str(a1)},
        },
        {"op": "<=", "rhs": str(upper[0]), "terms": {source_names[0]: "1"}},
    ]
    target_constraints = []
    for constraint in source_constraints:
        terms, moved = _transform_terms(constraint["terms"], mapping_rows)
        target_constraints.append(
            {
                "op": constraint["op"],
                "rhs": _fraction_text(Fraction(constraint["rhs"]) + moved),
                "terms": terms,
            }
        )
    objective_scale = -1 if ordinal % 4 in {1, 2} else 1
    objective_offset = ordinal % 7 - 3
    source_direction = "max" if ordinal % 2 else "min"
    target_direction = (
        source_direction if objective_scale > 0 else ("max" if source_direction == "min" else "min")
    )
    coefficients = {source_names[0]: str(1 + ordinal % 4), source_names[1]: str(2 + ordinal % 3)}
    constant = ordinal % 5 - 2
    if problem_family == "bounded_piecewise_linear":
        source_pieces = [
            {"constant": str(constant), "terms": coefficients},
            {
                "constant": str(constant + 1),
                "terms": {source_names[0]: str(-a0), source_names[1]: str(a1)},
            },
        ]
        target_pieces = []
        for piece in source_pieces:
            terms, moved = _transform_terms(piece["terms"], mapping_rows, objective_scale)
            target_pieces.append(
                {
                    "constant": _fraction_text(
                        objective_scale * Fraction(piece["constant"]) - moved + objective_offset
                    ),
                    "terms": terms,
                }
            )
        source_expression = {
            "kind": "piecewise_linear",
            "aggregation": "max",
            "pieces": source_pieces,
        }
        target_expression = {
            "kind": "piecewise_linear",
            "aggregation": "max" if objective_scale > 0 else "min",
            "pieces": target_pieces,
        }
    else:
        target_terms, moved = _transform_terms(coefficients, mapping_rows, objective_scale)
        source_expression = {"kind": "linear", "constant": str(constant), "terms": coefficients}
        target_expression = {
            "kind": "linear",
            "constant": _fraction_text(
                objective_scale * Fraction(constant) - moved + objective_offset
            ),
            "terms": target_terms,
        }
    source = {
        "schema_version": "carnot.bounded_optimization_formulation.v1",
        "variables": [_variable(source_names[i], lower[i], upper[i]) for i in range(2)],
        "constraints": source_constraints,
        "objective": {"direction": source_direction, "expression": source_expression},
    }
    target = {
        "schema_version": "carnot.bounded_optimization_formulation.v1",
        "variables": [
            _variable(name, *target_bound_by_name[name])
            for name in (target_names[1], target_names[0])
        ],
        "constraints": target_constraints,
        "objective": {"direction": target_direction, "expression": target_expression},
    }
    mapping = {
        "schema_version": "carnot.reformulation_mapping.v1",
        "claimed_relation": "equivalent",
        "variables": mapping_rows,
        "domain_clauses": [
            {
                "source": row["source"],
                "target": row["target"],
                "source_lower": str(lower[index]),
                "source_upper": str(upper[index]),
                "target_lower": str(target_bounds[index][0]),
                "target_upper": str(target_bounds[index][1]),
            }
            for index, row in enumerate(mapping_rows)
        ],
        "objective": {
            "source_direction": source_direction,
            "target_direction": target_direction,
            "scale": str(objective_scale),
            "offset": str(objective_offset),
        },
    }
    if conflict_class == "objective_coefficient_conflict":
        expression = target["objective"]["expression"]
        terms = expression["pieces"][0]["terms"] if "pieces" in expression else expression["terms"]
        first = sorted(terms)[0]
        terms[first] = _fraction_text(Fraction(terms[first]) + 1)
    elif conflict_class == "bound_conflict":
        target_variable = target["variables"][0]
        mapped_source_lower = scales[0] * lower[0] + offsets[0]
        if scales[0] > 0:
            target_variable["domain"]["lower"] = str(mapped_source_lower + 1)
        else:
            target_variable["domain"]["upper"] = str(mapped_source_lower - 1)
        clause = next(
            row for row in mapping["domain_clauses"] if row["target"] == target_variable["name"]
        )
        clause["target_lower"] = target_variable["domain"]["lower"]
        clause["target_upper"] = target_variable["domain"]["upper"]
    elif conflict_class == "objective_direction_conflict":
        changed = "max" if target_direction == "min" else "min"
        target["objective"]["direction"] = changed
        mapping["objective"]["target_direction"] = changed
    return source, target, mapping


def _boolean_pair(ordinal: int, conflict_class: str | None) -> tuple[JsonDict, JsonDict, JsonDict]:
    """Create one renamed cardinality pair with varied coefficients and direction."""

    sources = [f"b{ordinal}_{index}" for index in range(4)]
    targets = [f"z{ordinal}_{index}" for index in range(4)]
    shift = 1 + ordinal % 3
    mapping_rows = [
        {"source": source, "target": targets[(index + shift) % 4], "scale": "1", "offset": "0"}
        for index, source in enumerate(sources)
    ]
    limit = 1 + ordinal % 3
    source_constraints = [
        {"op": "<=", "rhs": str(limit), "terms": {name: "1" for name in sources}},
        {"op": "<=", "rhs": "4", "terms": {name: "1" for name in sources}},
    ]
    target_constraints = [
        {
            "op": row["op"],
            "rhs": row["rhs"],
            "terms": {
                mapping_rows[sources.index(name)]["target"]: value
                for name, value in row["terms"].items()
            },
        }
        for row in source_constraints
    ]
    coefficients = {name: str(1 + (ordinal + index) % 5) for index, name in enumerate(sources)}
    objective_scale = -1 if ordinal % 4 == 0 else 1
    source_direction = "max" if ordinal % 2 else "min"
    target_direction = (
        source_direction if objective_scale > 0 else ("max" if source_direction == "min" else "min")
    )
    target_coefficients = {
        mapping_rows[index]["target"]: str(objective_scale * int(coefficients[source]))
        for index, source in enumerate(sources)
    }
    source = {
        "schema_version": "carnot.bounded_optimization_formulation.v1",
        "variables": [_boolean_variable(name) for name in sources],
        "constraints": source_constraints,
        "objective": {
            "direction": source_direction,
            "expression": {"kind": "linear", "constant": "0", "terms": coefficients},
        },
    }
    target = {
        "schema_version": "carnot.bounded_optimization_formulation.v1",
        "variables": [_boolean_variable(row["target"]) for row in mapping_rows],
        "constraints": target_constraints,
        "objective": {
            "direction": target_direction,
            "expression": {"kind": "linear", "constant": "0", "terms": target_coefficients},
        },
    }
    mapping = {
        "schema_version": "carnot.reformulation_mapping.v1",
        "claimed_relation": "equivalent",
        "variables": mapping_rows,
        "domain_clauses": [
            {
                "source": row["source"],
                "target": row["target"],
                "source_lower": "0",
                "source_upper": "1",
                "target_lower": "0",
                "target_upper": "1",
            }
            for row in mapping_rows
        ],
        "objective": {
            "source_direction": source_direction,
            "target_direction": target_direction,
            "scale": str(objective_scale),
            "offset": "0",
        },
    }
    if conflict_class == "objective_direction_conflict":
        changed = "max" if target_direction == "min" else "min"
        target["objective"]["direction"] = changed
        mapping["objective"]["target_direction"] = changed
    elif conflict_class:
        first = sorted(target_coefficients)[0]
        target_coefficients[first] = str(int(target_coefficients[first]) + 1)
    return source, target, mapping


def _mapping_signature(mapping: Mapping[str, Any]) -> str:
    """Ignore variable names so renamed answer copies keep the same signature."""

    variable_rows = mapping.get("variables", [])
    variables = sorted(
        (str(row.get("scale")), str(row.get("offset")))
        for row in variable_rows
        if isinstance(row, Mapping)
    )
    objective = mapping.get("objective", {})
    signature = {
        "variables": variables,
        "objective_scale": str(objective.get("scale")),
        "objective_offset": str(objective.get("offset")),
        "direction_same": objective.get("source_direction") == objective.get("target_direction"),
    }
    return sha256_json(signature)


def _factor_ids(problem_family: str, relation: str, conflict_class: str | None) -> list[str]:
    """Expose reusable principles while withholding all answer coordinates."""

    if relation == "non_equivalent":
        return [f"reject_{conflict_class or 'uncertified_relation'}", "require_dual_exact_check"]
    common = ["variable_bijection", "objective_scale_direction", "surface_form_invariance"]
    if problem_family == "boolean_cardinality":
        return common + ["cardinality_preservation"]
    return common + ["bidirectional_domain_coverage", "constraint_substitution"]


def _retrieval_key(problem_family: str, no_op: bool = False) -> str:
    """Build an answer-free family key from visible formulation structure."""

    arity = 4 if problem_family == "boolean_cardinality" else 2
    pattern = "permutation_affine" if arity == 4 else "signed_affine_bijection"
    suffix = "|control=no_op" if no_op else ""
    return f"problem={problem_family}|arity={arity}|pattern={pattern}{suffix}"


def _certificate_id(attempt_key: str) -> str:
    """Create a stable identity without copying model output text."""

    return "EXP6957-" + hashlib.sha256(attempt_key.encode()).hexdigest()[:16]


def select_seed_certificates(
    certification: Mapping[str, Any],
    fixture: Mapping[str, Any],
    certification_replay: Mapping[str, Any],
) -> list[JsonDict]:
    """Select two exact successes per problem family and remove self-reports."""

    fixture_by_pair = {
        str(row.get("pair_id")): row for row in fixture.get("rows", []) if isinstance(row, Mapping)
    }
    replay_by_key = {
        str(row.get("attempt_key")): row
        for row in certification_replay.get("inputs", [])
        if isinstance(row, Mapping)
    }
    exact = [
        row
        for row in certification.get("rows", [])
        if isinstance(row, Mapping)
        and row.get("exact_mapping_correct") is True
        and row.get("terminal") is True
        and row.get("quarantined") is False
    ]
    selected: list[Mapping[str, Any]] = []
    for family in PROBLEM_FAMILIES:
        candidates = sorted(
            (row for row in exact if row.get("problem_family") == family),
            key=lambda row: str(row.get("attempt_key")),
        )
        distinct: list[Mapping[str, Any]] = []
        seen_pairs: set[str] = set()
        for row in candidates:
            pair_id = str(row.get("pair_id"))
            if pair_id not in seen_pairs:
                distinct.append(row)
                seen_pairs.add(pair_id)
        for row in candidates:
            if len(distinct) >= 2:
                break
            if row not in distinct:
                distinct.append(row)
        selected.extend(distinct[:2])
    rows: list[JsonDict] = []
    for source in selected:
        attempt_key = str(source["attempt_key"])
        replay = replay_by_key.get(attempt_key, {})
        parsed = (
            replay.get("parse", {}).get("parsed_candidate") if isinstance(replay, Mapping) else None
        )
        mapping = parsed.get("mapping") if isinstance(parsed, Mapping) else None
        fixture_row = fixture_by_pair.get(str(source.get("pair_id")), {})
        authority = next(
            (
                row
                for row in certification.get("authority_agreement_rows", [])
                if isinstance(row, Mapping) and row.get("attempt_key") == attempt_key
            ),
            {},
        )
        if not isinstance(mapping, Mapping) or fixture_row.get("authorities_agree") is not True:
            continue
        rows.append(
            {
                "source_certificate_id": _certificate_id(attempt_key),
                "source_experiment_id": 6957,
                "source_attempt_key": attempt_key,
                "source_pair_id": source.get("pair_id"),
                "problem_family": source.get("problem_family"),
                "certified_relation": source.get("certified_relation"),
                "fixture_expected_relation": fixture_row.get("expected_label"),
                "exact_mapping_correct": source.get("exact_mapping_correct") is True,
                "authorities_agree": authority.get("authorities_agree") is True,
                "terminal": source.get("terminal") is True,
                "quarantined": source.get("quarantined") is True,
                "candidate_hash": source.get("candidate_hash"),
                "mapping_hash": sha256_json(mapping),
                "mapping_signature": _mapping_signature(mapping),
                "reusable_factor_ids": _factor_ids(
                    str(source.get("problem_family")),
                    str(source.get("certified_relation")),
                    "source_non_equivalence"
                    if source.get("certified_relation") == "non_equivalent"
                    else None,
                ),
                "admission_authority": "exact_dual_engine_certificate",
                "confidence_used": False,
                "rationale_used": False,
                "learned_score_used": False,
            }
        )
    return rows


def _event_plan(ordinal: int, local_index: int, model_index: int) -> JsonDict:
    """Freeze the split, family, conflict, and safety role before exact labeling."""

    problem_family = PROBLEM_FAMILIES[(model_index + local_index) % len(PROBLEM_FAMILIES)]
    control_class = "standard"
    conflict_class: str | None = None
    if local_index == 1:
        conflict_class = "objective_coefficient_conflict"
    elif local_index in {7, 12, 17}:
        conflict_class = (
            "bound_conflict",
            "objective_coefficient_conflict",
            "objective_direction_conflict",
        )[local_index // 5 % 3]
    elif local_index == 19:
        problem_family = PROBLEM_FAMILIES[model_index]
        conflict_class = "objective_coefficient_conflict"
        control_class = "contradiction"
    elif local_index == 20:
        problem_family = PROBLEM_FAMILIES[model_index]
        control_class = "delayed_correction"
    elif local_index == 21:
        control_class = "no_op"
    elif local_index == 22:
        problem_family = PROBLEM_FAMILIES[model_index]
        control_class = "retention_probe"
    elif local_index == 23:
        conflict_class = "objective_direction_conflict"
        control_class = "interference_probe"
    return {
        "ordinal": ordinal,
        "local_index": local_index,
        "problem_family": problem_family,
        "split": "train" if ordinal < SEED_EVENT_COUNT else "evaluation",
        "event_role": "seed" if ordinal < SEED_EVENT_COUNT else "later",
        "control_class": control_class,
        "conflict_class": conflict_class,
    }


def _similarity(current: Mapping[str, Any], prior: Mapping[str, Any]) -> float:
    """Measure visible structural overlap without reading either outcome."""

    if current["retrieval_key"] == prior["retrieval_key"]:
        return 1.0
    if current["problem_family"] == prior["problem_family"]:
        return 0.35
    current_arity = 4 if current["problem_family"] == "boolean_cardinality" else 2
    prior_arity = 4 if prior["problem_family"] == "boolean_cardinality" else 2
    return 0.1 if current_arity == prior_arity else 0.0


def _memory_snippet(prior: Mapping[str, Any], relevant: bool) -> JsonDict:
    """Return abstract factors only; exact outcomes and mappings remain sealed."""

    return {
        "certificate_id": prior["certificate_id"],
        "problem_family": prior["problem_family"],
        "retrieval_key": prior["retrieval_key"],
        "relevance": "relevant" if relevant else "irrelevant",
        "factor_ids": list(prior["reusable_factor_ids"]),
        "factor_summary": "Check these structural conditions; derive all current coordinates from the visible formulations.",
        "token_cost": 28,
    }


def _select_memory(
    event: Mapping[str, Any],
    prior_events: Sequence[Mapping[str, Any]],
    arm: str,
) -> list[JsonDict]:
    """Freeze bounded FIFO and relevance-queue choices without running a model."""

    if arm == "no_memory" or event["control_class"] == "no_op":
        return []
    candidates = list(prior_events)
    if arm == "fifo":
        chosen = list(reversed(candidates))[:RETRIEVAL_LIMIT]
    else:
        relevant = [row for row in candidates if _similarity(event, row) == 1.0]
        reverse = event["control_class"] != "retention_probe"
        chosen = sorted(relevant, key=lambda row: int(row["ordinal"]), reverse=reverse)[
            :RETRIEVAL_LIMIT
        ]
    return [_memory_snippet(row, _similarity(event, row) == 1.0) for row in chosen]


def _prompt(
    event_id: str,
    problem_family: str,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    memory: Sequence[Mapping[str, Any]],
    surface_index: int,
) -> JsonDict:
    """Build the only payload a later model may receive."""

    surfaces = (
        "A planner renamed resources while keeping an unrelated note about warehouse color.",
        "An auditor changed labels; a decorative sentence mentions morning weather.",
        "A scheduler rewrote the same bounded task with an irrelevant project codename.",
        "A reviewer changed presentation order and added a harmless narrative detail.",
    )
    return {
        "event_id": event_id,
        "request_text": "Propose an affine relation or reject equivalence from the two visible formulations.",
        "surface_context": surfaces[surface_index % len(surfaces)],
        "problem_family": problem_family,
        "source_formulation": deepcopy(dict(source)),
        "target_formulation": deepcopy(dict(target)),
        "retrieved_memory": [deepcopy(dict(row)) for row in memory],
    }


def _event_hash(row: Mapping[str, Any]) -> str:
    """Hash all frozen scientific event content except its own digest."""

    payload = {key: value for key, value in row.items() if key != "event_hash"}
    return sha256_json(payload)


def generate_sequence(seed_certificate_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Generate, exactly certify, and project the full chronological sequence."""

    seeds = [deepcopy(dict(row)) for row in seed_certificate_rows]
    seed_by_family: dict[str, list[JsonDict]] = {
        family: [row for row in seeds if row.get("problem_family") == family]
        for family in PROBLEM_FAMILIES
    }
    event_rows: list[JsonDict] = []
    similarity_rows: list[JsonDict] = []
    opportunity_rows: list[JsonDict] = []
    headroom_rows: list[JsonDict] = []
    distractor_rows: list[JsonDict] = []
    correction_rows: list[JsonDict] = []
    retention_probe_rows: list[JsonDict] = []
    local_counts = dict.fromkeys(HEADLINE_MODEL_FAMILIES, 0)
    future_certificate_ids = [f"EVT6961-{index:03d}-CERT" for index in range(EXPECTED_EVENT_COUNT)]
    for ordinal in range(EXPECTED_EVENT_COUNT):
        model_index = ordinal % len(HEADLINE_MODEL_FAMILIES)
        model_family = HEADLINE_MODEL_FAMILIES[model_index]
        local_index = local_counts[model_family]
        local_counts[model_family] += 1
        plan = _event_plan(ordinal, local_index, model_index)
        problem_family = str(plan["problem_family"])
        conflict_class = plan["conflict_class"]
        if problem_family == "boolean_cardinality":
            source, target, answer_mapping = _boolean_pair(ordinal, conflict_class)
        else:
            source, target, answer_mapping = _integer_pair(ordinal, problem_family, conflict_class)
        expected_relation = "non_equivalent" if conflict_class else "equivalent"
        attempt = {
            "attempt_key": f"exp6961|{ordinal:03d}",
            "hf_id": "sealed_no_inference",
            "model_family": model_family,
            "pair_id": f"event-{ordinal:03d}",
            "problem_family": problem_family,
            "prompt_variant_id": "prospective_hidden_answer",
            "raw_sha256": sha256_json({"source": source, "target": target}),
            "source_formulation": source,
            "target_formulation": target,
            "parse": {
                "json_valid": True,
                "schema_valid": True,
                "failure_reason": None,
                "parsed_candidate": {"mapping": answer_mapping},
                "confidence": None,
                "rationale": None,
            },
            "canonical_relation": expected_relation,
            "difficulty": "conflict" if conflict_class else "opportunity",
        }
        exact = cert_exp.certify_proposal(attempt)
        proposal = exact["proposal_row"]
        retrieval_key = _retrieval_key(problem_family, plan["control_class"] == "no_op")
        event_id = f"EVT6961-{ordinal:03d}"
        certificate_id = future_certificate_ids[ordinal]
        seed_choices = seed_by_family[problem_family]
        source_seed = seed_choices[ordinal % len(seed_choices)]
        event: JsonDict = {
            **plan,
            "event_id": event_id,
            "certificate_id": certificate_id,
            "available_after_ordinal": ordinal,
            "model_family": model_family,
            "family_id": f"headline::{model_family}",
            "source_seed_certificate_id": source_seed["source_certificate_id"],
            "retrieval_key": retrieval_key,
            "token_budget": 168 + 28 * (ordinal % 3),
            "source_formulation": source,
            "target_formulation": target,
            "scientific_input_hash": sha256_json({"source": source, "target": target}),
            "exact_outcome": proposal["certified_relation"],
            "expected_relation": expected_relation,
            "exact_success": proposal["exact_mapping_correct"] is True
            and exact["authority_agreement_row"]["authorities_agree"] is True,
            "exact_outcome_hash": sha256_json(
                {
                    "event_id": event_id,
                    "enumeration": exact["enumeration_row"],
                    "z3": exact["z3_row"],
                }
            ),
            "sealed_answer_mapping": answer_mapping,
            "answer_mapping_hash": sha256_json(answer_mapping),
            "answer_mapping_signature": _mapping_signature(answer_mapping),
            "reusable_factor_ids": _factor_ids(problem_family, expected_relation, conflict_class),
            "correct_proposal_possible_without_copying": True,
            "eligible_prior_certificate_ids": [row["certificate_id"] for row in event_rows],
            "prohibited_current_certificate_id": certificate_id,
            "prohibited_future_certificate_ids": future_certificate_ids[ordinal + 1 :],
            "inference_ran": False,
        }
        for prior in event_rows:
            similarity_rows.append(
                {
                    "event_id": event_id,
                    "prior_certificate_id": prior["certificate_id"],
                    "similarity": _similarity(event, prior),
                    "exact_reusable_factors": sorted(
                        set(event["reusable_factor_ids"]) & set(prior["reusable_factor_ids"])
                    ),
                    "outcome_used_in_similarity": False,
                }
            )
        arm_prompts: JsonDict = {}
        selected_by_arm: JsonDict = {}
        for arm in ARMS:
            selected = _select_memory(event, event_rows, arm)
            selected_by_arm[arm] = [row["certificate_id"] for row in selected]
            arm_prompts[arm] = _prompt(
                event_id,
                problem_family,
                source,
                target,
                selected,
                ordinal,
            )
            relevant_count = sum(row["relevance"] == "relevant" for row in selected)
            opportunity_rows.append(
                {
                    "event_id": event_id,
                    "ordinal": ordinal,
                    "model_family": model_family,
                    "arm": arm,
                    "split": plan["split"],
                    "selected_certificate_ids": list(selected_by_arm[arm]),
                    "selected_relevant_certificate_count": relevant_count,
                    "structural_opportunity": plan["split"] == "evaluation" and relevant_count > 0,
                    "correct_without_answer_copy": True,
                    "inference_ran": False,
                    "model_outcome": None,
                }
            )
            for snippet in selected:
                if snippet["relevance"] == "irrelevant":
                    distractor_rows.append(
                        {
                            "event_id": event_id,
                            "arm": arm,
                            "certificate_id": snippet["certificate_id"],
                            "problem_family": snippet["problem_family"],
                            "event_problem_family": problem_family,
                            "relevance": "irrelevant",
                        }
                    )
        event["arm_prompt_payloads"] = arm_prompts
        event["prompt_payload"] = deepcopy(arm_prompts["queue"])
        event["arm_prompt_hashes"] = {arm: sha256_json(arm_prompts[arm]) for arm in ARMS}
        event["prompt_hash"] = event["arm_prompt_hashes"]["queue"]
        event["retrieval_plan"] = selected_by_arm
        event["retrieval_plan_hash"] = sha256_json(selected_by_arm)
        event["maximum_prior_similarity"] = max(
            (_similarity(event, prior) for prior in event_rows), default=0.0
        )
        event["event_hash"] = _event_hash(event)
        if plan["split"] == "evaluation":
            queue_row = opportunity_rows[-1]
            queue_relevant = queue_row["selected_relevant_certificate_count"]
            headroom_rows.append(
                {
                    "event_id": event_id,
                    "model_family": model_family,
                    "no_memory_relevant_factor_count": 0,
                    "queue_relevant_factor_count": queue_relevant,
                    "structural_headroom": queue_relevant,
                    "positive_headroom": queue_relevant > 0,
                }
            )
        if plan["control_class"] == "delayed_correction":
            prior_conflict = next(
                row
                for row in reversed(event_rows)
                if row["model_family"] == model_family
                and row["problem_family"] == problem_family
                and row["conflict_class"] is not None
            )
            correction_rows.append(
                {
                    "correction_event_id": event_id,
                    "corrected_event_id": prior_conflict["event_id"],
                    "delay_events": ordinal - int(prior_conflict["ordinal"]),
                    "exact_correction": True,
                }
            )
        if plan["control_class"] == "retention_probe":
            anchor = next(row for row in event_rows if row["problem_family"] == problem_family)
            retention_probe_rows.append(
                {
                    "event_id": event_id,
                    "anchor_certificate_id": anchor["certificate_id"],
                    "anchor_age_events": ordinal - int(anchor["ordinal"]),
                    "fifo_retained": anchor["certificate_id"] in selected_by_arm["fifo"],
                    "queue_retained": anchor["certificate_id"] in selected_by_arm["queue"],
                }
            )
        event_rows.append(event)

    chronology_rows = [
        {
            "event_id": row["event_id"],
            "ordinal": row["ordinal"],
            "available_after_ordinal": row["available_after_ordinal"],
            "eligible_prior_certificate_ids": list(row["eligible_prior_certificate_ids"]),
            "prohibited_current_certificate_id": row["prohibited_current_certificate_id"],
            "prohibited_future_certificate_ids": list(row["prohibited_future_certificate_ids"]),
        }
        for row in event_rows
    ]
    transformation_rows = [
        {
            "event_id": row["event_id"],
            "source_seed_certificate_id": row["source_seed_certificate_id"],
            "problem_family": row["problem_family"],
            "pattern_action": "break" if row["conflict_class"] else "preserve",
            "changed_dimensions": [
                "names",
                "coefficients",
                "objective_direction",
                "irrelevant_surface_form",
            ]
            + (["bounds"] if row["problem_family"] != "boolean_cardinality" else []),
            "non_identical_to_seed": True,
            "scientific_input_hash": row["scientific_input_hash"],
        }
        for row in event_rows
    ]
    retrieval_key_rows = [
        {
            "event_id": row["event_id"],
            "retrieval_key": row["retrieval_key"],
            "problem_family": row["problem_family"],
            "answer_fields_excluded": True,
            "outcome_fields_excluded": True,
        }
        for row in event_rows
    ]
    eligible_prior_rows = [
        {
            "event_id": row["event_id"],
            "eligible_prior_certificate_ids": list(row["eligible_prior_certificate_ids"]),
            "all_available_strictly_before_event": True,
        }
        for row in event_rows
    ]
    prohibited_future_rows = [
        {
            "event_id": row["event_id"],
            "current_certificate_id": row["certificate_id"],
            "prohibited_future_certificate_ids": list(row["prohibited_future_certificate_ids"]),
            "excluded_from_all_prompts": True,
        }
        for row in event_rows
    ]
    reusable_factor_rows = [
        {
            "event_id": row["event_id"],
            "certificate_id": row["certificate_id"],
            "available_after_ordinal": row["available_after_ordinal"],
            "problem_family": row["problem_family"],
            "retrieval_key": row["retrieval_key"],
            "factor_ids": list(row["reusable_factor_ids"]),
            "contains_complete_mapping": False,
            "exact_success": row["exact_success"],
        }
        for row in event_rows
    ]
    conflict_rows = [
        {
            "event_id": row["event_id"],
            "model_family": row["model_family"],
            "problem_family": row["problem_family"],
            "conflict_class": row["conflict_class"],
            "exact_outcome": row["exact_outcome"],
            "interference_risk": True,
        }
        for row in event_rows
        if row["conflict_class"] is not None
    ]
    leakage_rows = [
        {
            "event_id": row["event_id"],
            "current_outcome_absent": True,
            "future_ids_absent": True,
            "isomorphic_answer_absent": True,
            "current_certificate_absent": True,
            "passed": True,
        }
        for row in event_rows
    ]
    split_rows = [
        {
            "event_id": row["event_id"],
            "ordinal": row["ordinal"],
            "split": row["split"],
            "boundary_ordinal": SEED_EVENT_COUNT,
            "frozen_before_inference": True,
        }
        for row in event_rows
    ]
    compact_rows = [
        {
            "event_id": row["event_id"],
            "ordinal": row["ordinal"],
            "model_family": row["model_family"],
            "problem_family": row["problem_family"],
            "split": row["split"],
            "exact_success": row["exact_success"],
            "maximum_prior_similarity": row["maximum_prior_similarity"],
            "queue_structural_headroom": next(
                (
                    value["structural_headroom"]
                    for value in headroom_rows
                    if value["event_id"] == row["event_id"]
                ),
                0,
            ),
        }
        for row in event_rows
    ]
    family_rows = []
    for family in HEADLINE_MODEL_FAMILIES:
        events = [row for row in event_rows if row["model_family"] == family]
        opportunity = [
            row
            for row in opportunity_rows
            if row["model_family"] == family
            and row["arm"] == "queue"
            and row["structural_opportunity"] is True
        ]
        family_rows.append(
            {
                "model_family": family,
                "family_id": f"headline::{family}",
                "event_count": len(events),
                "seed_event_count": sum(row["event_role"] == "seed" for row in events),
                "later_event_count": sum(row["event_role"] == "later" for row in events),
                "genuine_memory_opportunity_count": len(opportunity),
                "positive_headroom_count": sum(
                    row["positive_headroom"]
                    for row in headroom_rows
                    if row["model_family"] == family
                ),
                "problem_family_counts": {
                    problem: sum(row["problem_family"] == problem for row in events)
                    for problem in PROBLEM_FAMILIES
                },
            }
        )
    return {
        "rows": compact_rows,
        "seed_certificate_rows": seeds,
        "event_rows": event_rows,
        "chronology_rows": chronology_rows,
        "family_rows": family_rows,
        "transformation_rows": transformation_rows,
        "retrieval_key_rows": retrieval_key_rows,
        "eligible_prior_rows": eligible_prior_rows,
        "prohibited_future_rows": prohibited_future_rows,
        "similarity_rows": similarity_rows,
        "reusable_factor_rows": reusable_factor_rows,
        "conflict_rows": conflict_rows,
        "distractor_rows": distractor_rows,
        "correction_rows": correction_rows,
        "retention_probe_rows": retention_probe_rows,
        "opportunity_rows": opportunity_rows,
        "headroom_rows": headroom_rows,
        "leakage_rows": leakage_rows,
        "split_rows": split_rows,
    }


def generator_is_deterministic(seed_certificate_rows: Sequence[Mapping[str, Any]]) -> bool:
    """Run the generator twice and compare every timing-free sequence byte."""

    return sha256_json(generate_sequence(seed_certificate_rows)) == sha256_json(
        generate_sequence(seed_certificate_rows)
    )


def checkpoint_is_writable(path: Path) -> bool:
    """Probe the target directory without leaving experiment state behind."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".exp6961-", delete=True):
            pass
        return True
    except OSError:  # pragma: no cover - host filesystem failure is not portable to CI.
        return False


def collect_preconditions(
    *,
    certification: Mapping[str, Any],
    fixture: Mapping[str, Any],
    certification_replay: Mapping[str, Any],
    fixture_path: Path,
    sealed_checkpoint_path: Path,
) -> list[JsonDict]:
    """Check exact evidence, deterministic generation, hashes, and seal storage."""

    seeds = select_seed_certificates(certification, fixture, certification_replay)
    expected_fixture_hash = (
        certification.get("source_artifact_hashes", {}).get("fixture_artifact", {}).get("sha256")
    )
    fixture_exact = len(fixture.get("rows", [])) >= 6 and all(
        row.get("authorities_agree") is True for row in fixture.get("rows", [])
    )
    exact_seed_rows = all(
        row.get("exact_mapping_correct") is True
        and row.get("authorities_agree") is True
        and row.get("terminal") is True
        and row.get("quarantined") is False
        for row in seeds
    )
    return [
        gate_check(
            "smt_certification_run_complete_score",
            1,
            certification.get("smt_certification_run_complete_score"),
        ),
        gate_check("exact_certificate_rows", True, exact_seed_rows),
        gate_check("enough_certified_successes", SEED_EVENT_COUNT, len(seeds)),
        gate_check("exact_fixture_rows", True, fixture_exact),
        gate_check("fixture_artifact_hash", expected_fixture_hash, sha256_path(fixture_path)),
        gate_check("deterministic_generators", True, generator_is_deterministic(seeds)),
        gate_check(
            "writable_sealed_checkpoint", True, checkpoint_is_writable(sealed_checkpoint_path)
        ),
    ]


def _walk_keys(value: Any) -> set[str]:
    """Collect nested prompt keys for explicit label-leak checks."""

    if isinstance(value, Mapping):
        return {str(key) for key in value} | set().union(*(_walk_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_walk_keys(v) for v in value)) if value else set()
    return set()


def _mapping_objects(value: Any) -> list[Mapping[str, Any]]:
    """Find structured mappings only inside retrieved memory."""

    found: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "mapping" and isinstance(child, Mapping):
                found.append(child)
            found.extend(_mapping_objects(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_mapping_objects(child))
    return found


def sequence_conformance_errors(data: Mapping[str, Any]) -> list[str]:
    """Return stable failure names for chronology, copies, leakage, and coverage."""

    errors: list[str] = []

    def add(reason: str) -> None:
        if reason not in errors:
            errors.append(reason)

    seeds = data.get("seed_certificate_rows", [])
    source_ids = [row.get("source_certificate_id") for row in seeds]
    if len(source_ids) != len(set(source_ids)):
        add("copied_seed_certificate")
    for row in seeds:
        if any(key in row for key in ("confidence", "rationale", "learned_score")):
            add("self_report_seed_authority")
        if not (
            row.get("exact_mapping_correct") is True
            and row.get("authorities_agree") is True
            and row.get("terminal") is True
            and row.get("quarantined") is False
        ):
            add("non_exact_seed_certificate")
    events = data.get("event_rows", [])
    if len(events) != EXPECTED_EVENT_COUNT:
        add("event_count")
    ordinals = [row.get("ordinal") for row in events]
    if ordinals != list(range(len(events))):
        add("chronology_order")
    for key in ("event_id", "certificate_id", "scientific_input_hash"):
        values = [row.get(key) for row in events]
        if len(values) != len(set(values)):
            add("duplicate_event")
    certificate_ordinal = {row.get("certificate_id"): row.get("ordinal") for row in events}
    for event in events:
        ordinal = event.get("ordinal")
        for certificate_id in event.get("eligible_prior_certificate_ids", []):
            if certificate_ordinal.get(certificate_id, EXPECTED_EVENT_COUNT) >= ordinal:
                add("time_reversal")
        prompts = list(event.get("arm_prompt_payloads", {}).values()) + [
            event.get("prompt_payload", {})
        ]
        prohibited = set(event.get("prohibited_future_certificate_ids", [])) | {
            event.get("certificate_id")
        }
        for prompt in prompts:
            if _walk_keys(prompt) & _FORBIDDEN_PROMPT_KEYS:
                add("future_label_leakage")
            text = canonical_json(prompt).decode()
            if any(str(value) in text for value in prohibited):
                add("future_label_leakage")
            for memory in prompt.get("retrieved_memory", []):
                if memory.get("relevance") == "relevant" and memory.get(
                    "problem_family"
                ) != event.get("problem_family"):
                    add("family_collision")
                certificate_id = memory.get("certificate_id")
                if certificate_id not in event.get("eligible_prior_certificate_ids", []):
                    add("time_reversal")
            for mapping in _mapping_objects(prompt.get("retrieved_memory", [])):
                if _mapping_signature(mapping) == event.get("answer_mapping_signature"):
                    add("isomorphic_answer_leakage")
                else:
                    add("copied_answer_mapping")
        if event.get("control_class") == "no_op" and any(
            event.get("retrieval_plan", {}).get(arm) for arm in ARMS
        ):
            add("no_op_retrieval_not_empty")
        if event.get("exact_success") is not True:
            add("uncertified_event_outcome")
        if event.get("correct_proposal_possible_without_copying") is not True:
            add("answer_copy_required")
    for family in HEADLINE_MODEL_FAMILIES:
        family_events = [row for row in events if row.get("model_family") == family]
        if len(family_events) != EVENTS_PER_MODEL_FAMILY:
            add(f"family_event_count:{family}")
    if sum(row.get("event_role") == "seed" for row in events) != SEED_EVENT_COUNT:
        add("seed_event_count")
    if any(
        (row.get("ordinal", 0) < SEED_EVENT_COUNT) != (row.get("split") == "train")
        for row in events
    ):
        add("split_boundary")
    required_controls = {
        "contradiction",
        "delayed_correction",
        "no_op",
        "retention_probe",
        "interference_probe",
    }
    observed_controls = {row.get("control_class") for row in events}
    if not required_controls <= observed_controls:
        add("safety_case_coverage")
    if not data.get("distractor_rows"):
        add("distractor_coverage")
    if len(data.get("correction_rows", [])) < len(HEADLINE_MODEL_FAMILIES):
        add("correction_coverage")
    if len(data.get("retention_probe_rows", [])) < len(HEADLINE_MODEL_FAMILIES):
        add("retention_coverage")
    for family in HEADLINE_MODEL_FAMILIES:
        opportunity = sum(
            row.get("model_family") == family
            and row.get("arm") == "queue"
            and row.get("structural_opportunity") is True
            for row in data.get("opportunity_rows", [])
        )
        headroom = sum(
            row.get("model_family") == family and row.get("positive_headroom") is True
            for row in data.get("headroom_rows", [])
        )
        if opportunity < MIN_LATER_OPPORTUNITY_PER_FAMILY or headroom == 0:
            add(f"insufficient_opportunity:{family}")
    return errors


def reduce_terminal_gate(data: Mapping[str, Any]) -> JsonDict:
    """Separate a complete opportunity null from invalid sequence construction."""

    errors = sequence_conformance_errors(data)
    opportunity_only = bool(errors) and all(
        reason.startswith("insufficient_opportunity:") for reason in errors
    )
    if opportunity_only:
        return {
            "ready_score": 0,
            "verdict_class": "null",
            "honest_verdict": "complete_null_insufficient_certified_memory_opportunity",
            "errors": errors,
        }
    if errors:
        return {
            "ready_score": 0,
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_certified_event_sequence",
            "errors": errors,
        }
    return {
        "ready_score": 1,
        "verdict_class": "circular_positive",
        "honest_verdict": "complete_circular_positive_certified_event_sequence_conforms",
        "errors": [],
    }


def _sequence_projection(data: Mapping[str, Any]) -> JsonDict:
    """Return the replay hashes that a fresh process must reproduce."""

    event_hashes = {row["event_id"]: row["event_hash"] for row in data["event_rows"]}
    prompt_hashes = {row["event_id"]: row["arm_prompt_hashes"] for row in data["event_rows"]}
    retrieval_hashes = {row["event_id"]: row["retrieval_plan_hash"] for row in data["event_rows"]}
    return {
        "event_hashes": event_hashes,
        "prompt_hashes": prompt_hashes,
        "retrieval_hashes": retrieval_hashes,
        "sequence_hash": sha256_json(
            {
                "event_hashes": event_hashes,
                "prompt_hashes": prompt_hashes,
                "retrieval_hashes": retrieval_hashes,
            }
        ),
    }


def replay_checkpoint(path: Path) -> JsonDict:
    """Regenerate the sequence from sealed sanitized seed rows."""

    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint_schema_version")
    if checkpoint.get("random_seed") != RANDOM_SEED:
        raise ValueError("checkpoint_random_seed")
    seeds = checkpoint.get("seed_certificate_rows")
    if not isinstance(seeds, list):
        raise ValueError("checkpoint_seed_rows")
    return _sequence_projection(generate_sequence(seeds))


def _fresh_process_replay(repo_root: Path, checkpoint_path: Path) -> JsonDict:
    """Launch a clean interpreter whose only generator input is the seal."""

    output_path = checkpoint_path.with_suffix(".replay.json")
    environment = dict(os.environ)
    python_root = str(repo_root / "python")
    environment["PYTHONPATH"] = python_root + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "carnot.experiment_6961_certified_event_sequence",
            "--replay-checkpoint",
            str(checkpoint_path),
            "--replay-output",
            str(output_path),
        ],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:  # pragma: no cover - child crashes are integration failures.
        raise RuntimeError(f"fresh_process_replay_failed:{completed.stderr[-500:]}")
    result = json.loads(output_path.read_text(encoding="utf-8"))
    output_path.unlink()
    return result


def source_artifact_hashes(
    repo_root: Path,
    certification_path: Path,
    fixture_path: Path,
    certification_replay_path: Path,
) -> JsonDict:
    """Bind exact inputs and every local file that interprets their meaning."""

    paths = {
        "certification_artifact": certification_path,
        "fixture_artifact": fixture_path,
        "certification_replay": certification_replay_path,
        "learning_spec": repo_root / SPEC_PATH,
        "module": repo_root / MODULE_PATH,
        "focused_tests": repo_root / TEST_PATH,
        "wrapper": repo_root / WRAPPER_PATH,
        "constraint_memory": repo_root / MEMORY_PATH,
    }
    return {name: {"path": str(path), "sha256": sha256_path(path)} for name, path in paths.items()}


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable scientific content while excluding time and the digest itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("duration_s", None)
    payload.pop("reproducibility_checksum", None)
    return sha256_json(payload)


def _empty_rows() -> JsonDict:
    """Return every row field so blocked artifacts retain the complete schema."""

    return {
        field: []
        for field in REQUIRED_ARTIFACT_FIELDS
        if field == "rows" or field.endswith("_rows")
    }


def build_blocked_artifact(
    *,
    run_date: str,
    duration_s: float,
    preconditions_checked: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    sealed_checkpoint_path: Path,
) -> JsonDict:
    """Emit the full required schema when no sequence may be trusted."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": 6961,
        "run_date": run_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(duration_s, 0.000001), 9),
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        **_empty_rows(),
        "sealed_checkpoint_path": str(sealed_checkpoint_path),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "certified_event_sequence_ready_score": 0,
        "gate_check_summary": gate_summary(preconditions_checked),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_certified_event_sequence",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _read_json(path: Path) -> JsonDict:
    """Read one required JSON object through a strict type boundary."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def build_from_paths(
    *,
    run_date: str,
    repo_root: Path = REPO_ROOT,
    certification_path: Path | None = None,
    fixture_path: Path | None = None,
    certification_replay_path: Path | None = None,
    sealed_checkpoint_path: Path | None = None,
) -> JsonDict:
    """Check inputs, freeze the sequence, and verify its hashes in a fresh process."""

    started = time.perf_counter()
    root = Path(repo_root)
    certification_file = (
        Path(certification_path) if certification_path else root / CERTIFICATION_PATH
    )
    fixture_file = Path(fixture_path) if fixture_path else root / FIXTURE_PATH
    replay_file = (
        Path(certification_replay_path)
        if certification_replay_path
        else root / CERTIFICATION_REPLAY_PATH
    )
    checkpoint_file = (
        Path(sealed_checkpoint_path) if sealed_checkpoint_path else root / SEALED_CHECKPOINT_PATH
    )
    hashes = source_artifact_hashes(root, certification_file, fixture_file, replay_file)
    try:
        certification = _read_json(certification_file)
        fixture = _read_json(fixture_file)
        certification_replay = _read_json(replay_file)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:  # pragma: no cover - host input failure.
        checks = [
            gate_check(
                "source_inputs_readable",
                {"readable": True},
                {"readable": False, "error": type(exc).__name__},
            )
        ]
        return build_blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions_checked=checks,
            source_hashes=hashes,
            sealed_checkpoint_path=checkpoint_file,
        )
    seeds = select_seed_certificates(certification, fixture, certification_replay)
    checks = collect_preconditions(
        certification=certification,
        fixture=fixture,
        certification_replay=certification_replay,
        fixture_path=fixture_file,
        sealed_checkpoint_path=checkpoint_file,
    )
    if any(row["passed"] is not True for row in checks):
        return build_blocked_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            preconditions_checked=checks,
            source_hashes=hashes,
            sealed_checkpoint_path=checkpoint_file,
        )
    parent = generate_sequence(seeds)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "random_seed": RANDOM_SEED,
        "headline_model_families": list(HEADLINE_MODEL_FAMILIES),
        "problem_families": list(PROBLEM_FAMILIES),
        "events_per_model_family": EVENTS_PER_MODEL_FAMILY,
        "seed_certificate_rows": seeds,
    }
    write_json_atomic(checkpoint_file, checkpoint)
    parent_projection = _sequence_projection(parent)
    child_projection = _fresh_process_replay(root, checkpoint_file)
    fresh_rows = []
    for event in parent["event_rows"]:
        event_id = event["event_id"]
        parent_prompts = parent_projection["prompt_hashes"][event_id]
        child_prompts = child_projection["prompt_hashes"].get(event_id)
        fresh_rows.append(
            {
                "event_id": event_id,
                "parent_event_hash": parent_projection["event_hashes"][event_id],
                "child_event_hash": child_projection["event_hashes"].get(event_id),
                "replay_matches": parent_projection["event_hashes"][event_id]
                == child_projection["event_hashes"].get(event_id),
                "parent_prompt_hashes": parent_prompts,
                "child_prompt_hashes": child_prompts,
                "prompt_matches": parent_prompts == child_prompts,
                "parent_retrieval_hash": parent_projection["retrieval_hashes"][event_id],
                "child_retrieval_hash": child_projection["retrieval_hashes"].get(event_id),
                "retrieval_matches": parent_projection["retrieval_hashes"][event_id]
                == child_projection["retrieval_hashes"].get(event_id),
                "parent_sequence_hash": parent_projection["sequence_hash"],
                "child_sequence_hash": child_projection["sequence_hash"],
                "sequence_matches": parent_projection["sequence_hash"]
                == child_projection["sequence_hash"],
                "fresh_process": True,
            }
        )
    parent["fresh_process_replay_rows"] = fresh_rows
    reduced = reduce_terminal_gate(parent)
    replay_safe = len(fresh_rows) == EXPECTED_EVENT_COUNT and all(
        row["replay_matches"]
        and row["prompt_matches"]
        and row["retrieval_matches"]
        and row["sequence_matches"]
        for row in fresh_rows
    )
    if (
        reduced["ready_score"] == 1 and not replay_safe
    ):  # pragma: no cover - child mismatch is injected via validation.
        reduced = {
            "ready_score": 0,
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_fresh_process_hash_drift",
            "errors": ["fresh_process_replay_mismatch"],
        }
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": 6961,
        "run_date": run_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": checks,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(time.perf_counter() - started, 0.000001), 9),
        "source_artifact_hashes": hashes,
        **parent,
        "sealed_checkpoint_path": str(checkpoint_file),
        "sealed_checkpoint_sha256": sha256_path(checkpoint_file),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "certified_event_sequence_ready_score": reduced["ready_score"],
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": reduced["verdict_class"],
        "honest_verdict": reduced["honest_verdict"],
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute schema, conformance, source, replay, verdict, and digest gates."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        return ["missing_required_fields:" + ",".join(missing)]
    errors: list[str] = []
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_ARTIFACT_FIELDS) <= set(principles):
        errors.append("field_principles_incomplete")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_is_oracle_must_be_true")
    verdict_class = artifact.get("verdict_class")
    if verdict_class not in _ALLOWED_VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    verdict = str(artifact.get("honest_verdict", ""))
    if verdict_class == "blocked":
        if artifact.get("certified_event_sequence_ready_score") != 0:
            errors.append("blocked_ready_score_nonzero")
        if verdict != "blocked_certified_event_sequence":
            errors.append("blocked_verdict_mismatch")
        summary = artifact.get("gate_check_summary", {})
        if summary.get("failed_check") is None or summary.get("passed") is not False:
            errors.append("blocked_gate_summary_incomplete")
    else:
        reduced = reduce_terminal_gate(artifact)
        if artifact.get("certified_event_sequence_ready_score") != reduced["ready_score"]:
            errors.append("ready_score_mismatch")
        if reduced["ready_score"] == 1 and verdict_class != "circular_positive":
            errors.append("conforming_sequence_requires_circular_positive")
        elif reduced["ready_score"] == 0 and verdict_class != reduced["verdict_class"]:
            errors.append("terminal_verdict_class_mismatch")
        replay_rows = artifact.get("fresh_process_replay_rows", [])
        if len(replay_rows) != EXPECTED_EVENT_COUNT or not all(
            row.get("replay_matches") is True
            and row.get("parent_event_hash") == row.get("child_event_hash")
            and row.get("prompt_matches") is True
            and row.get("parent_prompt_hashes") == row.get("child_prompt_hashes")
            and row.get("retrieval_matches") is True
            and row.get("parent_retrieval_hash") == row.get("child_retrieval_hash")
            and row.get("sequence_matches") is True
            and row.get("parent_sequence_hash") == row.get("child_sequence_hash")
            for row in replay_rows
        ):
            errors.append("fresh_process_replay_mismatch")
        if verdict_class == "circular_positive" and not verdict.startswith(
            "complete_circular_positive_"
        ):
            errors.append("honest_verdict_prefix_mismatch")
        if verdict_class in {"null", "disqualified"} and not verdict.startswith(
            f"complete_{verdict_class}_"
        ):
            errors.append("honest_verdict_prefix_mismatch")
    for source in artifact.get("source_artifact_hashes", {}).values():
        path = Path(str(source.get("path", "")))
        recorded = source.get("sha256")
        if recorded is not None and sha256_path(path) != recorded:
            errors.append("source_artifact_hash_drift")
            break
    if payload_checksum(artifact) != artifact.get("reproducibility_checksum"):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def run(
    *,
    date: str,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
    sealed_checkpoint_path: Path | None = None,
) -> JsonDict:
    """Build, validate, and atomically write the terminal artifact."""

    root = Path(repo_root)
    output = Path(output_path) if output_path else root / RESULT_PATH
    artifact = build_from_paths(
        run_date=date,
        repo_root=root,
        sealed_checkpoint_path=sealed_checkpoint_path,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("artifact_validation:" + ",".join(errors))
    write_json_atomic(output, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Expose the required command and a private fresh-process replay mode."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260904")
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
                "event_count": len(artifact["event_rows"]),
                "certified_event_sequence_ready_score": artifact[
                    "certified_event_sequence_ready_score"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the required wrapper.
    raise SystemExit(main())
