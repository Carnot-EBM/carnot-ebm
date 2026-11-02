"""Certify every frozen candidate and select one schedule on calibration rows.

Z3 and bounded enumeration decide exact outcomes. The code freezes a schedule
before it opens held-out labels. Exact labels remain an oracle, so a favorable
result is circular evidence rather than an oracle-distinct verifier result.

Spec refs: REQ-VERIFY-6976 and SCENARIO-VERIFY-6976-*.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import time
from typing import Any

from carnot import experiment_6955_reformulation_fixture as fixture_exp
from carnot import experiment_6957_smt_mapping_certification as certificate_exp
from carnot import experiment_6975_delayed_constraint_candidate_bank as bank_exp


JsonDict = dict[str, Any]
Certifier = Callable[[Mapping[str, Any]], JsonDict]

EXPERIMENT_ID = 6976
SCHEMA_VERSION = "carnot.exp6976.exact_candidate_certification.v1"
RUN_DATE = "20260904"
RANDOM_SEED = 6_976_202_609_04
INFERENCE_SUBSTRATE = "deterministic_z3_and_bounded_enumeration_certification"
EXPECTED_CANDIDATE_COUNT = 108
BOOTSTRAP_RESAMPLES = 10_000
SCHEDULE_ORDER = ("direct", "trigger_switched", "draft_conditioned")
EXPECTED_BANK_SHA256 = "sha256:4a9faf7223d174729091248f8cb763cc6b76e481abe69bdadc19adfe7dac5455"
EXPECTED_FIXTURE_SHA256 = bank_exp.EXPECTED_EXP6967_SHA256
EXPECTED_SCHEDULE_HASHES = {
    "direct": "sha256:f52f3a25bc97b9f3fa285241d519050b1b84697296a8351cb14d09d2e931816a",
    "trigger_switched": "sha256:b57629bd80241a86448993b1c0d916cee79791e637d007917e1d875d6766004e",
    "draft_conditioned": "sha256:97c47e59fc07f35a89433a7b95bea518239ca376a91d51cef27d272e8bec69dc",
}
EXPECTED_SPLIT_HASH = "sha256:17670d86c28ad886be806fa6b0e610b42058f430fffe034fb789742c889db9fb"
RESULT_PATH = Path("results/experiment_6976_exact_candidate_certification.json")
SOURCE_PATHS = {
    "bank": Path("results/experiment_6975_delayed_constraint_candidate_bank.json"),
    "fixture": Path("results/experiment_6967_certified_error_headroom_fixture.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "per_candidate_rows",
    "parser_outcome_rows",
    "exact_witness_rows",
    "solver_agreement_rows",
    "calibration_metric_rows",
    "policy_selection_rows",
    "selected_policy",
    "selected_policy_hash",
    "label_opening_rows",
    "heldout_metric_rows",
    "per_group_results",
    "paired_schedule_delta_rows",
    "heldout_headroom_rows",
    "heldout_headroom_group_count",
    "candidate_certification_complete_score",
    "selected_policy_ready_score",
    "selected_policy_positive_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "schema": "A versioned schema lets later checks reject incompatible evidence.",
    "experiment_id": "A stable identity prevents another run from supplying these results.",
    "run_date": "The fixed date makes changes to the execution protocol visible.",
    "field_principles": "A reason for each field makes the evidence contract reviewable.",
    "preconditions_checked": "Exact gates stop changed inputs from becoming certified evidence.",
    "inference_substrate": "The substrate states that deterministic solvers, not an LLM, made labels.",
    "duration_s": "Wall time shows that parsing and exact certification executed.",
    "source_artifact_hashes": "Hashes bind every conclusion to frozen candidates and labels.",
    "rows": "The full terminal denominator lets downstream checks ignore headlines.",
    "per_candidate_rows": "One row per candidate preserves failures and exact outcomes.",
    "parser_outcome_rows": "Parser parity prevents later extraction or repair from changing admission.",
    "exact_witness_rows": "Witnesses and counterexamples make each solver decision auditable.",
    "solver_agreement_rows": "Per-candidate parity prevents one exact engine from certifying itself.",
    "calibration_metric_rows": "Calibration-only metrics expose the data used to select the schedule.",
    "policy_selection_rows": "Rank rows prove that success, parsing, and fixed order selected the policy.",
    "selected_policy": "The frozen policy records the decision made before held-out labels opened.",
    "selected_policy_hash": "A content hash prevents policy changes after held-out assessment.",
    "label_opening_rows": "Opening receipts enforce calibration-first and one held-out access.",
    "heldout_metric_rows": "Held-out metrics measure every schedule and the frozen policy once.",
    "per_group_results": "Model-pair rows keep all three candidates in each headroom decision.",
    "paired_schedule_delta_rows": "Paired differences compare schedules on identical candidate groups.",
    "heldout_headroom_rows": "Mixed valid and invalid groups demonstrate selectable candidate variation.",
    "heldout_headroom_group_count": "A row-derived count prevents a target from becoming a result.",
    "candidate_certification_complete_score": "One requires 108 terminal exact outcomes and solver parity.",
    "selected_policy_ready_score": "One means selection froze before all held-out rows terminated.",
    "selected_policy_positive_score": "One records a held-out point gain without claiming oracle independence.",
    "random_seed": "One seed makes paired bootstrap intervals reproducible.",
    "reproducibility_checksum": "A timing-free digest detects changes to scientific content.",
    "gate_check_summary": "Expected and observed values make blocked runs actionable.",
    "verifier_is_oracle": "True states that exact labels participate in selection and assessment.",
    "verdict_class": "A closed class keeps circular evidence distinct from a verifier win.",
    "honest_verdict": "A class-consistent prefix gives automation a stable terminal state.",
}

DECISION_STATUSES = {"proved", "counterexample"}
EXACT_TERMINAL_STATUSES = {
    "proved",
    "counterexample",
    "parse_rejected",
    "schema_rejected",
}
ALL_TERMINAL_STATUSES = EXACT_TERMINAL_STATUSES | {"timeout", "unknown", "exception"}
AGREEMENT_FIELDS = (
    "label",
    "forward_feasible",
    "reverse_feasible",
    "variable_coverage_complete",
    "objective_direction_valid",
    "objective_affine_preserved",
    "objective_order_preserved",
)


class CertificationError(RuntimeError):
    """Name a fail-closed certification or policy contract violation."""


def canonical_json(value: Any) -> bytes:
    """Return stable JSON bytes for scientific hashes and equality checks."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    """Return one SHA-256 digest with the repository's explicit prefix."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one JSON-compatible value without whitespace or key-order drift."""

    return sha256_bytes(canonical_json(value))


def sha256_path(path: Path) -> str | None:
    """Hash one file and preserve absence as an explicit null value."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def gate_check(check: str, expected: Any, observed: Any) -> JsonDict:
    """Record both sides of one exact precondition comparison."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": expected == observed,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep every failed gate with its expected and observed values."""

    return [
        {
            "failed_check": row.get("check"),
            "expected_value": row.get("expected_value"),
            "observed_value": row.get("observed_value"),
        }
        for row in checks
        if row.get("passed") is not True
    ]


def _read_object(path: Path) -> JsonDict:
    """Read one required JSON object through a strict type boundary."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_object_required:{path}")
    return value


def load_inputs(repo_root: Path) -> JsonDict:
    """Load the frozen candidate bank and the sealed exact fixture."""

    return {name: _read_object(repo_root / path) for name, path in SOURCE_PATHS.items()}


def source_artifact_hashes(repo_root: Path) -> JsonDict:
    """Bind the result to inputs, exact engines, parser, spec, tests, and command."""

    paths = {
        "experiment_6975": SOURCE_PATHS["bank"],
        "experiment_6967": SOURCE_PATHS["fixture"],
        "experiment_6957": Path("results/experiment_6957_smt_mapping_certification.json"),
        "experiment_6959": Path("results/experiment_6959_certified_energy_selection.json"),
        "frozen_parser": Path("python/carnot/experiment_6975_delayed_constraint_candidate_bank.py"),
        "exact_certifier": Path("python/carnot/experiment_6957_smt_mapping_certification.py"),
        "module": Path("python/carnot/experiment_6976_exact_candidate_certification.py"),
        "test": Path("tests/python/test_experiment_6976_exact_candidate_certification.py"),
        "spec": Path("openspec/capabilities/verification/spec.md"),
        "wrapper": Path("scripts/experiments/experiment_6976_exact_candidate_certification.py"),
    }
    return {name: sha256_path(repo_root / path) for name, path in paths.items()}


def _raw_roster_observation(bank: Mapping[str, Any]) -> JsonDict:
    """Rebuild all raw identities instead of trusting saved headline counts."""

    attempts = bank.get("per_attempt_rows", [])
    raw_rows = bank.get("raw_output_rows", [])
    if not isinstance(attempts, list) or not isinstance(raw_rows, list):
        return {"attempt_count": 0, "raw_count": 0, "unique": False, "hashes_match": False}
    raw_by_key = {str(row.get("attempt_key")): row for row in raw_rows if isinstance(row, Mapping)}
    attempt_keys = [str(row.get("attempt_key")) for row in attempts if isinstance(row, Mapping)]
    hashes_match = len(raw_by_key) == len(raw_rows)
    for row in attempts:
        if not isinstance(row, Mapping):
            hashes_match = False
            continue
        key = str(row.get("attempt_key"))
        raw = raw_by_key.get(key, {})
        text = str(row.get("candidate_raw_text", ""))
        digest = bank_exp.sha256_text(text)
        hashes_match = hashes_match and row.get("candidate_raw_sha256") == digest
        hashes_match = hashes_match and raw.get("candidate_raw_sha256") == digest
        hashes_match = hashes_match and raw.get("candidate_raw_text") == text
    return {
        "attempt_count": len(attempts),
        "raw_count": len(raw_rows),
        "unique": len(attempt_keys) == len(set(attempt_keys)) == EXPECTED_CANDIDATE_COUNT,
        "hashes_match": bool(hashes_match),
    }


def _fixture_binding_observation(bank: Mapping[str, Any], fixture: Mapping[str, Any]) -> JsonDict:
    """Check selected prompts, sealed labels, witnesses, and source certificates."""

    selected = bank.get("selected_pair_rows", [])
    prompt_rows = fixture.get("prompt_visible_rows", [])
    witnesses = [
        row
        for row in fixture.get("exact_witness_rows", [])
        if isinstance(row, Mapping) and row.get("subject_kind") == "slice_pair"
    ]
    solver_rows = fixture.get("solver_agreement_rows", [])
    manifests_by_key = {
        (split, str(row.get("pair_id"))): row
        for split in ("calibration", "heldout")
        for row in fixture.get(f"{split}_rows", [])
        if isinstance(row, Mapping)
    }
    prompts_by_key = {
        (str(row.get("split")), str(row.get("pair_id"))): row
        for row in prompt_rows
        if isinstance(row, Mapping)
    }
    witnesses_by_key = {(str(row.get("split")), str(row.get("pair_id"))): row for row in witnesses}
    solver_by_key = {
        (str(row.get("split")), str(row.get("pair_id"))): row
        for row in solver_rows
        if isinstance(row, Mapping)
    }
    selected_keys = {
        (str(row.get("split")), str(row.get("pair_id")))
        for row in selected
        if isinstance(row, Mapping)
    }
    bound = len(selected) == len(selected_keys) == 12 and all(
        key in manifests_by_key
        and key in prompts_by_key
        and key in witnesses_by_key
        and key in solver_by_key
        for key in selected_keys
    )
    for row in selected:
        if not isinstance(row, Mapping):
            bound = False
            continue
        key = (str(row.get("split")), str(row.get("pair_id")))
        manifest = manifests_by_key.get(key, {})
        prompt = prompts_by_key.get(key, {})
        witness = witnesses_by_key.get(key, {})
        solver = solver_by_key.get(key, {})
        expected = witness.get("expected_label")
        agreement = witness.get("agreement", {})
        bound = bound and row.get("prompt_record_hash") == manifest.get("prompt_record_hash")
        bound = bound and prompt.get("prompt_record_hash") == manifest.get("prompt_record_hash")
        bound = bound and witness.get("source_certificate_hash") == manifest.get(
            "source_certificate_hash"
        )
        bound = bound and agreement.get("authorities_agree") is True
        bound = bound and witness.get("enumeration", {}).get("label") == expected
        bound = bound and witness.get("z3", {}).get("label") == expected
        bound = bound and solver.get("authorities_agree") is True
        bound = bound and solver.get("labels_match_expected") is True
    return {
        "selected_pair_count": len(selected),
        "sealed_witness_count": sum(key in witnesses_by_key for key in selected_keys),
        "all_bindings_match": bool(bound),
    }


def check_preconditions(repo_root: Path, inputs: Mapping[str, Any]) -> list[JsonDict]:
    """Check all frozen evidence and exact engines before certification."""

    bank = inputs.get("bank", {})
    fixture = inputs.get("fixture", {})
    bank = bank if isinstance(bank, Mapping) else {}
    fixture = fixture if isinstance(fixture, Mapping) else {}
    schedule_hashes = {
        str(row.get("schedule_id")): bank_exp.sha256_text(str(row.get("schedule_text", "")))
        for row in bank.get("schedule_rows", [])
        if isinstance(row, Mapping)
    }
    selected = bank.get("selected_pair_rows", [])
    computed_split = bank_exp.compute_split_hash(selected) if isinstance(selected, list) else None
    return [
        gate_check(
            "candidate_bank_file_hash",
            EXPECTED_BANK_SHA256,
            sha256_path(repo_root / SOURCE_PATHS["bank"]),
        ),
        gate_check(
            "exp6967_file_hash",
            EXPECTED_FIXTURE_SHA256,
            sha256_path(repo_root / SOURCE_PATHS["fixture"]),
        ),
        gate_check("candidate_bank_complete_score", 1, bank.get("candidate_bank_complete_score")),
        gate_check(
            "raw_candidate_roster",
            {
                "attempt_count": EXPECTED_CANDIDATE_COUNT,
                "raw_count": EXPECTED_CANDIDATE_COUNT,
                "unique": True,
                "hashes_match": True,
            },
            _raw_roster_observation(bank),
        ),
        gate_check("schedule_hashes", EXPECTED_SCHEDULE_HASHES, bank.get("schedule_hashes")),
        gate_check("schedule_hash_replay", EXPECTED_SCHEDULE_HASHES, schedule_hashes),
        gate_check("split_hash", EXPECTED_SPLIT_HASH, bank.get("split_hash")),
        gate_check("split_hash_replay", EXPECTED_SPLIT_HASH, computed_split),
        gate_check(
            "bank_exp6967_source_hash",
            EXPECTED_FIXTURE_SHA256,
            bank.get("source_artifact_hashes", {}).get("experiment_6967"),
        ),
        gate_check(
            "sealed_label_and_witness_bindings",
            {
                "selected_pair_count": 12,
                "sealed_witness_count": 12,
                "all_bindings_match": True,
            },
            _fixture_binding_observation(bank, fixture),
        ),
        gate_check("z3_available", True, fixture_exp.z3 is not None),
        gate_check(
            "bounded_enumerator_available", True, callable(certificate_exp.certify_with_enumerator)
        ),
    ]


def _adapt_mapping(raw: Mapping[str, Any], pair: Mapping[str, Any]) -> JsonDict:
    """Adapt ConstraintIR fields without changing any model-supplied value."""

    source = pair["source_formulation"]
    target = pair["target_formulation"]
    source_domains = {str(row["name"]): row["domain"] for row in source["variables"]}
    target_domains = {str(row["name"]): row["domain"] for row in target["variables"]}
    source_direction = str(source["objective"]["direction"])
    declared_direction = raw["objective_map"]["direction"]
    target_direction = (
        source_direction
        if declared_direction == "same"
        else ("max" if source_direction == "min" else "min")
    )
    variable_rows = deepcopy(raw["variable_map"])
    return {
        "schema_version": fixture_exp.MAPPING_SCHEMA_VERSION,
        "variables": variable_rows,
        "domain_clauses": [
            {
                "source": row["source"],
                "target": row["target"],
                "source_lower": source_domains.get(str(row["source"]), {}).get("lower"),
                "source_upper": source_domains.get(str(row["source"]), {}).get("upper"),
                "target_lower": target_domains.get(str(row["target"]), {}).get("lower"),
                "target_upper": target_domains.get(str(row["target"]), {}).get("upper"),
            }
            for row in variable_rows
        ],
        "objective": {
            "source_direction": source_direction,
            "target_direction": target_direction,
            "scale": raw["objective_map"]["scale"],
            "offset": raw["objective_map"]["offset"],
        },
        "claimed_relation": "equivalent",
    }


def parse_raw_candidate(
    attempt: Mapping[str, Any], pair: Mapping[str, Any]
) -> tuple[JsonDict, JsonDict | None]:
    """Reparse exact raw text with the frozen parser and record strict parity."""

    raw_text = str(attempt.get("candidate_raw_text", ""))
    stored = deepcopy(attempt.get("parser_diagnostic"))
    reparsed = bank_exp.parse_syntax(raw_text)
    parse_success = reparsed.get("constraintir_shape_valid") is True
    mapping = _adapt_mapping(json.loads(raw_text), pair) if parse_success else None
    return (
        {
            "attempt_key": str(attempt.get("attempt_key")),
            "raw_sha256": attempt.get("candidate_raw_sha256"),
            "raw_sha256_matches": attempt.get("candidate_raw_sha256")
            == bank_exp.sha256_text(raw_text),
            "stored_diagnostic": stored,
            "reparsed_diagnostic": reparsed,
            "parser_parity": stored == reparsed,
            "json_valid": reparsed.get("json_valid") is True,
            "parse_success": parse_success,
            "parse_reason": reparsed.get("syntax_reason"),
            "terminal": True,
        },
        mapping,
    )


def _engine_terminal_row(engine: str, status: str, reason: str) -> JsonDict:
    """Build one complete engine row for a rejection or nondecision."""

    return {
        "engine": engine,
        "status": status,
        "terminal": status in ALL_TERMINAL_STATUSES,
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
        "unknown_reasons": [reason] if status in {"timeout", "unknown", "exception"} else [],
    }


def _safe_certify(certifier: Certifier, engine: str, pair: Mapping[str, Any]) -> JsonDict:
    """Turn solver exceptions into explicit terminal outcomes instead of omissions."""

    try:
        row = deepcopy(certifier(pair))
    except TimeoutError as exc:
        return _engine_terminal_row(engine, "timeout", str(exc) or "solver_timeout")
    except Exception as exc:  # noqa: BLE001 - exact exception evidence is part of the artifact.
        return _engine_terminal_row(engine, "exception", f"{type(exc).__name__}:{exc}")
    status = row.get("status")
    if status not in ALL_TERMINAL_STATUSES:
        return _engine_terminal_row(engine, "exception", f"invalid_engine_status:{status}")
    row["terminal"] = True
    return row


def _solver_agreement(
    attempt_key: str, enumeration: Mapping[str, Any], z3_row: Mapping[str, Any]
) -> JsonDict:
    """Compare every exact obligation without treating nondecisions as success."""

    enum_status = enumeration.get("status")
    z3_status = z3_row.get("status")
    status_agreement = enum_status == z3_status
    relation_agreement = enumeration.get("label") == z3_row.get("label")
    satisfiability_agreement = all(
        enumeration.get(field) == z3_row.get(field)
        for field in ("forward_feasible", "reverse_feasible")
    )
    mapping_direction_agreement = enumeration.get("objective_direction_valid") == z3_row.get(
        "objective_direction_valid"
    )
    solution_space_agreement = all(
        enumeration.get(field) == z3_row.get(field)
        for field in ("forward_feasible", "reverse_feasible", "variable_coverage_complete")
    )
    optimum_agreement = all(
        enumeration.get(field) == z3_row.get(field)
        for field in (
            "objective_direction_valid",
            "objective_affine_preserved",
            "objective_order_preserved",
        )
    )
    if enum_status in {"parse_rejected", "schema_rejected"}:
        all_required = status_agreement and enumeration.get("failure_reason") == z3_row.get(
            "failure_reason"
        )
    elif enum_status in DECISION_STATUSES and z3_status in DECISION_STATUSES:
        all_required = all(
            enumeration.get(field) == z3_row.get(field) for field in AGREEMENT_FIELDS
        )
    else:
        all_required = status_agreement and enumeration.get("failure_reason") == z3_row.get(
            "failure_reason"
        )
    certified_relation = (
        enumeration.get("label")
        if all_required and enum_status in DECISION_STATUSES and z3_status in DECISION_STATUSES
        else None
    )
    return {
        "attempt_key": attempt_key,
        "enumeration_status": enum_status,
        "z3_status": z3_status,
        "status_agreement": status_agreement,
        "relation_agreement": relation_agreement,
        "satisfiability_agreement": satisfiability_agreement,
        "optimum_agreement": optimum_agreement,
        "mapping_direction_agreement": mapping_direction_agreement,
        "solution_space_agreement": solution_space_agreement,
        "all_required_agreement": bool(all_required),
        "certified_relation": certified_relation,
        "terminal": enumeration.get("terminal") is True and z3_row.get("terminal") is True,
    }


def _combined_outcome(values: Sequence[Any]) -> str:
    """Reduce exact Boolean obligations without hiding absent evidence."""

    if not values or all(value is None for value in values):
        return "not_evaluated"
    if any(value is False for value in values):
        return "failed"
    if all(value is True for value in values):
        return "passed"
    return "unresolved"


def certify_candidate(
    attempt: Mapping[str, Any],
    pair: Mapping[str, Any],
    *,
    z3_certifier: Certifier = certificate_exp.certify_with_z3,
    enumeration_certifier: Certifier = certificate_exp.certify_with_enumerator,
) -> JsonDict:
    """Certify one frozen candidate while keeping each failure surface separate."""

    parser_row, proposed_mapping = parse_raw_candidate(attempt, pair)
    canonical_mapping: JsonDict | None = None
    schema_reason: str | None = None
    if not parser_row["json_valid"]:
        schema_outcome = "not_evaluated"
        schema_reason = parser_row["parse_reason"]
    elif proposed_mapping is None:
        schema_outcome = "rejected"
        schema_reason = parser_row["parse_reason"]
    else:
        try:
            source = fixture_exp.validate_formulation(pair["source_formulation"])
            target = fixture_exp.validate_formulation(pair["target_formulation"])
            canonical_mapping = fixture_exp.canonical_mapping(proposed_mapping, source, target)
        except (
            fixture_exp.MappingSchemaError,
            fixture_exp.FormulationSchemaError,
            ValueError,
        ) as exc:
            schema_outcome = "rejected"
            schema_reason = str(exc)
        else:
            schema_outcome = "valid"
    if canonical_mapping is None:
        status = "parse_rejected" if not parser_row["json_valid"] else "schema_rejected"
        enumeration = _engine_terminal_row("enumerator", status, str(schema_reason))
        z3_row = _engine_terminal_row("z3", status, str(schema_reason))
    else:
        engine_pair = {
            "pair_id": str(attempt["attempt_key"]),
            "source": pair["source_formulation"],
            "target": pair["target_formulation"],
            "mapping": canonical_mapping,
        }
        enumeration = _safe_certify(
            enumeration_certifier, "python_exhaustive_bounded_enumerator_v1", engine_pair
        )
        z3_row = _safe_certify(z3_certifier, "z3_symbolic_reformulation_checker_v2", engine_pair)
    agreement = _solver_agreement(str(attempt["attempt_key"]), enumeration, z3_row)
    domain_values = [
        enumeration.get(field)
        for field in ("forward_feasible", "reverse_feasible", "variable_coverage_complete")
    ] + [
        z3_row.get(field)
        for field in ("forward_feasible", "reverse_feasible", "variable_coverage_complete")
    ]
    direction_values = [
        enumeration.get("objective_direction_valid"),
        z3_row.get("objective_direction_valid"),
    ]
    order_values = [
        enumeration.get("objective_order_preserved"),
        z3_row.get("objective_order_preserved"),
    ]
    optimum_values = direction_values + [
        enumeration.get("objective_affine_preserved"),
        z3_row.get("objective_affine_preserved"),
        *order_values,
    ]
    statuses = {"enumeration": enumeration.get("status"), "z3": z3_row.get("status")}
    candidate_row = {
        "attempt_key": str(attempt["attempt_key"]),
        "ordinal": attempt.get("ordinal"),
        "hf_id": attempt.get("hf_id"),
        "pair_id": attempt.get("pair_id"),
        "split": attempt.get("split"),
        "formulation_family": attempt.get("formulation_family"),
        "schedule_id": attempt.get("schedule_id"),
        "raw_sha256": attempt.get("candidate_raw_sha256"),
        "parse_outcome": "parsed" if parser_row["json_valid"] else "rejected",
        "parse_success": parser_row["parse_success"],
        "parse_reason": parser_row["parse_reason"],
        "parser_parity": parser_row["parser_parity"],
        "schema_outcome": schema_outcome,
        "schema_reason": schema_reason,
        "domain_correspondence_outcome": _combined_outcome(domain_values),
        "satisfiability_outcome": _combined_outcome(domain_values),
        "objective_direction_outcome": _combined_outcome(direction_values),
        "objective_order_outcome": _combined_outcome(order_values),
        "optimum_outcome": _combined_outcome(optimum_values),
        "solution_space_equivalence_outcome": _combined_outcome(domain_values),
        "timeout_outcome": {engine: status == "timeout" for engine, status in statuses.items()},
        "exception_outcome": {engine: status == "exception" for engine, status in statuses.items()},
        "unknown_outcome": {engine: status == "unknown" for engine, status in statuses.items()},
        "generation_exception_outcome": bool(
            attempt.get("exception_type") or attempt.get("call_status") == "exception"
        ),
        "enumeration_status": statuses["enumeration"],
        "z3_status": statuses["z3"],
        "authorities_agree": agreement["all_required_agreement"],
        "certified_relation": agreement["certified_relation"],
        "expected_label": None,
        "exact_semantic_success": None,
        "terminal": agreement["terminal"],
    }
    witness_row = {
        "attempt_key": str(attempt["attempt_key"]),
        "pair_id": attempt.get("pair_id"),
        "enumeration_status": enumeration.get("status"),
        "enumeration_witnesses": deepcopy(enumeration.get("witnesses", {})),
        "enumeration_counterexamples": deepcopy(enumeration.get("counterexamples", {})),
        "z3_status": z3_row.get("status"),
        "z3_witnesses": deepcopy(z3_row.get("witnesses", {})),
        "z3_counterexamples": deepcopy(z3_row.get("counterexamples", {})),
        "enumeration_engine_row": deepcopy(enumeration),
        "z3_engine_row": deepcopy(z3_row),
        "terminal": agreement["terminal"],
    }
    return {
        "candidate_row": candidate_row,
        "parser_outcome_row": parser_row,
        "exact_witness_row": witness_row,
        "solver_agreement_row": agreement,
    }


class SealedLabelVault:
    """Expose each label split once and require a frozen policy for held-out data."""

    def __init__(
        self, fixture: Mapping[str, Any], *, allowed_pair_ids: set[str] | None = None
    ) -> None:
        self._fixture = fixture
        self._allowed_pair_ids = allowed_pair_ids
        self._opened: set[str] = set()
        self.opening_rows: list[JsonDict] = []

    def open(self, split: str, *, selected_policy_hash: str | None = None) -> dict[str, str]:
        """Open one sealed split once, after the required causal boundary."""

        if split in self._opened:
            raise CertificationError(f"labels_already_opened:{split}")
        if split == "heldout" and not selected_policy_hash:
            raise CertificationError("selected_policy_hash_required")
        labels = {
            str(row["pair_id"]): str(row["expected_label"])
            for row in self._fixture.get("exact_witness_rows", [])
            if isinstance(row, Mapping)
            and row.get("subject_kind") == "slice_pair"
            and row.get("split") == split
            and (
                self._allowed_pair_ids is None or str(row.get("pair_id")) in self._allowed_pair_ids
            )
        }
        self._opened.add(split)
        self.opening_rows.append(
            {
                "split": split,
                "opening_sequence": len(self.opening_rows) + 1,
                "selected_policy_hash": selected_policy_hash,
                "label_count": len(labels),
            }
        )
        return labels


def apply_labels(
    rows: Sequence[Mapping[str, Any]], split: str, labels: Mapping[str, str]
) -> list[JsonDict]:
    """Attach one opened label split and leave every other split sealed."""

    result = deepcopy(list(rows))
    for row in result:
        if row.get("split") != split:
            continue
        pair_id = str(row.get("pair_id"))
        if pair_id not in labels:
            raise CertificationError(f"opened_label_missing:{split}:{pair_id}")
        row["expected_label"] = labels[pair_id]
        row["exact_semantic_success"] = (
            row.get("authorities_agree") is True
            and row.get("certified_relation") is not None
            and row.get("certified_relation") == labels[pair_id]
        )
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a measured rate and preserve an empty denominator as null."""

    return numerator / denominator if denominator else None


def select_schedule(
    calibration_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], list[JsonDict], JsonDict, str]:
    """Rank schedules with calibration labels only and freeze one policy hash."""

    if any(row.get("split") != "calibration" for row in calibration_rows):
        raise CertificationError("calibration_rows_only")
    metrics: list[JsonDict] = []
    for order, schedule in enumerate(SCHEDULE_ORDER):
        members = [row for row in calibration_rows if row.get("schedule_id") == schedule]
        if any(not isinstance(row.get("exact_semantic_success"), bool) for row in members):
            raise CertificationError("calibration_label_not_opened")
        exact_count = sum(row.get("exact_semantic_success") is True for row in members)
        parse_count = sum(row.get("parse_success") is True for row in members)
        metrics.append(
            {
                "split": "calibration",
                "schedule_id": schedule,
                "candidate_count": len(members),
                "exact_success_count": exact_count,
                "exact_success_rate": _rate(exact_count, len(members)),
                "parse_success_count": parse_count,
                "parse_success_rate": _rate(parse_count, len(members)),
                "fixed_schedule_order": order,
                "label_opening_sequence": 1,
                "terminal": all(row.get("terminal") is True for row in members),
            }
        )
    ranked_metrics = sorted(
        metrics,
        key=lambda row: (
            -int(row["exact_success_count"]),
            -int(row["parse_success_count"]),
            int(row["fixed_schedule_order"]),
        ),
    )
    ranking = [
        {
            **deepcopy(row),
            "rank": rank,
            "selected": rank == 1,
            "selection_frozen": True,
            "selection_uses_heldout_labels": False,
            "rank_rule": "exact_success_then_parse_success_then_fixed_schedule_order",
        }
        for rank, row in enumerate(ranked_metrics, start=1)
    ]
    winner = ranking[0]
    policy = {
        "schedule_id": winner["schedule_id"],
        "rank_rule": winner["rank_rule"],
        "calibration_candidate_count": winner["candidate_count"],
        "calibration_exact_success_count": winner["exact_success_count"],
        "calibration_parse_success_count": winner["parse_success_count"],
        "fixed_schedule_order": list(SCHEDULE_ORDER),
        "selection_label_split": "calibration",
        "selection_frozen": True,
        "heldout_labels_opened_during_selection": False,
    }
    return metrics, ranking, policy, sha256_json(policy)


def _grouped(
    rows: Sequence[Mapping[str, Any]], field: str
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """Return lexical groups for one required reporting dimension."""

    values = sorted({str(row.get(field)) for row in rows})
    return [(value, [row for row in rows if str(row.get(field)) == value]) for value in values]


def _metric_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    schedule: str,
    dimension: str,
    value: str,
    role: str,
) -> JsonDict:
    """Reduce exact held-out success for one schedule and one grouping value."""

    count = len(rows)
    success = sum(row.get("exact_semantic_success") is True for row in rows)
    return {
        "split": "heldout",
        "metric_role": role,
        "schedule_id": schedule,
        "dimension": dimension,
        "value": value,
        "candidate_count": count,
        "exact_success_count": success,
        "exact_success_rate": _rate(success, count),
        "label_opening_sequence": 2,
        "terminal": all(row.get("terminal") is True for row in rows),
    }


def build_heldout_metric_rows(
    heldout_rows: Sequence[Mapping[str, Any]], selected_schedule: str | None
) -> list[JsonDict]:
    """Report every schedule and the frozen policy at all required dimensions."""

    if any(row.get("split") != "heldout" for row in heldout_rows):
        raise CertificationError("heldout_rows_only")
    dimensions = (
        ("overall", None),
        ("model", "hf_id"),
        ("formulation_family", "formulation_family"),
        ("pair", "pair_id"),
    )
    result: list[JsonDict] = []
    for schedule in SCHEDULE_ORDER:
        schedule_rows = [row for row in heldout_rows if row.get("schedule_id") == schedule]
        for dimension, field in dimensions:
            groups = [("all", schedule_rows)] if field is None else _grouped(schedule_rows, field)
            result.extend(
                _metric_row(
                    members,
                    schedule=schedule,
                    dimension=dimension,
                    value=value,
                    role="schedule",
                )
                for value, members in groups
            )
    if selected_schedule is not None:
        policy_rows = [row for row in heldout_rows if row.get("schedule_id") == selected_schedule]
        for dimension, field in dimensions:
            groups = [("all", policy_rows)] if field is None else _grouped(policy_rows, field)
            result.extend(
                _metric_row(
                    members,
                    schedule=selected_schedule,
                    dimension=dimension,
                    value=value,
                    role="selected_policy",
                )
                for value, members in groups
            )
    return result


def build_per_group_results(
    heldout_rows: Sequence[Mapping[str, Any]], selected_schedule: str | None
) -> list[JsonDict]:
    """Build one three-schedule row for each held-out model and pair."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in heldout_rows:
        if row.get("split") != "heldout":
            raise CertificationError("heldout_rows_only")
        grouped[(str(row.get("hf_id")), str(row.get("pair_id")))].append(row)
    result: list[JsonDict] = []
    for (model, pair_id), members in sorted(grouped.items()):
        by_schedule = {str(row.get("schedule_id")): row for row in members}
        if len(members) != len(SCHEDULE_ORDER) or set(by_schedule) != set(SCHEDULE_ORDER):
            raise CertificationError(f"schedule_roster_mismatch:{model}:{pair_id}")
        outcomes = {
            schedule: by_schedule[schedule].get("exact_semantic_success")
            for schedule in SCHEDULE_ORDER
        }
        if any(not isinstance(value, bool) for value in outcomes.values()):
            raise CertificationError(f"heldout_label_not_opened:{model}:{pair_id}")
        valid = sum(value is True for value in outcomes.values())
        invalid = len(SCHEDULE_ORDER) - valid
        has_headroom = valid > 0 and invalid > 0
        result.append(
            {
                "group_id": sha256_json({"hf_id": model, "pair_id": pair_id}),
                "hf_id": model,
                "pair_id": pair_id,
                "formulation_family": members[0].get("formulation_family"),
                "candidate_count": len(SCHEDULE_ORDER),
                "schedule_outcomes": outcomes,
                "valid_candidate_count": valid,
                "invalid_candidate_count": invalid,
                "has_heldout_headroom": has_headroom,
                "within_group_exact_headroom": int(has_headroom),
                "selected_schedule": selected_schedule,
                "selected_schedule_success": outcomes.get(selected_schedule),
                "oracle_at_k": int(valid > 0),
                "oracle_used_for_selection": False,
                "terminal": all(row.get("terminal") is True for row in members),
            }
        )
    return result


def headroom_rows(groups: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Project only measured groups that contain valid and invalid candidates."""

    return [deepcopy(dict(row)) for row in groups if row.get("has_heldout_headroom") is True]


def heldout_headroom_group_count(groups: Sequence[Mapping[str, Any]]) -> int:
    """Count headroom decisions from per-group evidence and no target value."""

    return sum(row.get("has_heldout_headroom") is True for row in groups)


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return one deterministic nearest-index bootstrap percentile."""

    ordered = sorted(values)
    return ordered[int(probability * (len(ordered) - 1))]


def _paired_interval(
    differences: Sequence[tuple[str, int]], *, seed: int, resamples: int
) -> tuple[float, float, float]:
    """Bootstrap pair IDs while retaining every model row within each pair."""

    by_pair: dict[str, list[int]] = defaultdict(list)
    for pair_id, difference in differences:
        by_pair[pair_id].append(difference)
    pair_ids = sorted(by_pair)
    mean = sum(value for _, value in differences) / len(differences)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        selected = [rng.choice(pair_ids) for _ in pair_ids]
        values = [value for pair_id in selected for value in by_pair[pair_id]]
        samples.append(sum(values) / len(values))
    return mean, _percentile(samples, 0.025), _percentile(samples, 0.975)


def paired_schedule_deltas(
    heldout_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = RANDOM_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[JsonDict]:
    """Compare every schedule pair on matching held-out model-pair groups."""

    groups = build_per_group_results(heldout_rows, selected_schedule=None)
    result: list[JsonDict] = []
    for index, (schedule_a, schedule_b) in enumerate(combinations(SCHEDULE_ORDER, 2)):
        differences = [
            (
                str(row["pair_id"]),
                int(row["schedule_outcomes"][schedule_a])
                - int(row["schedule_outcomes"][schedule_b]),
            )
            for row in groups
        ]
        mean, lower, upper = _paired_interval(differences, seed=seed + index, resamples=resamples)
        result.append(
            {
                "comparison": f"{schedule_a}_minus_{schedule_b}",
                "schedule_a": schedule_a,
                "schedule_b": schedule_b,
                "paired_candidate_count": len(differences),
                "paired_pair_count": len({pair_id for pair_id, _ in differences}),
                "wins": sum(value > 0 for _, value in differences),
                "losses": sum(value < 0 for _, value in differences),
                "ties": sum(value == 0 for _, value in differences),
                "mean_delta": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "bootstrap_unit": "pair_id",
                "bootstrap_resamples": resamples,
                "terminal": True,
            }
        )
    return result


def candidate_completion_score(
    rows: Sequence[Mapping[str, Any]], agreement_rows: Sequence[Mapping[str, Any]]
) -> int:
    """Require all candidates to have exact terminal outcomes and engine parity."""

    agreement_by_key = {str(row.get("attempt_key")): row for row in agreement_rows}
    keys = [str(row.get("attempt_key")) for row in rows]
    complete = (
        len(rows) == len(agreement_rows) == EXPECTED_CANDIDATE_COUNT
        and len(set(keys)) == EXPECTED_CANDIDATE_COUNT
        and set(keys) == set(agreement_by_key)
        and all(
            row.get("terminal") is True
            and row.get("parser_parity") is True
            and row.get("enumeration_status") in EXACT_TERMINAL_STATUSES
            and row.get("z3_status") in EXACT_TERMINAL_STATUSES
            and agreement_by_key[str(row.get("attempt_key"))].get("all_required_agreement") is True
            for row in rows
        )
    )
    return int(complete)


def policy_ready_score(
    policy: Mapping[str, Any] | None,
    policy_hash: str | None,
    selection_rows: Sequence[Mapping[str, Any]],
    heldout_rows: Sequence[Mapping[str, Any]],
) -> int:
    """Accept a frozen policy decision, including a deliberately frozen null."""

    expected_hash = sha256_json(policy) if policy is not None else sha256_json(None)
    frozen = (
        policy_hash == expected_hash
        and len(selection_rows) == len(SCHEDULE_ORDER)
        and all(row.get("selection_frozen") is True for row in selection_rows)
    )
    heldout_terminal = bool(heldout_rows) and all(
        row.get("terminal") is True for row in heldout_rows
    )
    return int(frozen and heldout_terminal)


def policy_positive_score(
    policy: Mapping[str, Any] | None,
    heldout_metrics: Sequence[Mapping[str, Any]],
    headroom_count: int,
    ready_score: int,
) -> int:
    """Record a strict held-out point gain while preserving oracle circularity."""

    if ready_score != 1 or policy is None or headroom_count <= 0:
        return 0
    overall = {
        str(row.get("schedule_id")): int(row.get("exact_success_count", 0))
        for row in heldout_metrics
        if row.get("metric_role") == "schedule" and row.get("dimension") == "overall"
    }
    selected = str(policy.get("schedule_id"))
    return int(
        selected in overall
        and all(
            overall[selected] > value for schedule, value in overall.items() if schedule != selected
        )
    )


def verdict_for_scores(
    complete_score: int,
    ready_score: int,
    positive_score: int,
    agreement_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Map exact completion and policy evidence to the closed verdict vocabulary."""

    if complete_score == ready_score == 1:
        if positive_score == 1:
            return "circular_positive", "complete_circular_exact_candidate_headroom"
        return "null", "complete_null_exact_candidate_headroom"
    if any(row.get("all_required_agreement") is False for row in agreement_rows):
        return "disqualified", "complete_disqualified_exact_candidate_certification"
    return "partial", "partial_exact_candidate_certification"


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable scientific content while excluding duration and the digest itself."""

    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    )


def _empty_surfaces() -> JsonDict:
    """Return every row surface needed by a diagnostic blocked artifact."""

    return {
        field: []
        for field in (
            "rows",
            "per_candidate_rows",
            "parser_outcome_rows",
            "exact_witness_rows",
            "solver_agreement_rows",
            "calibration_metric_rows",
            "policy_selection_rows",
            "label_opening_rows",
            "heldout_metric_rows",
            "per_group_results",
            "paired_schedule_delta_rows",
            "heldout_headroom_rows",
        )
    }


def build_blocked_artifact(
    *,
    date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    """Build the complete fail-closed schema when any precondition is absent."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(list(checks)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        **_empty_surfaces(),
        "selected_policy": None,
        "selected_policy_hash": None,
        "heldout_headroom_group_count": 0,
        "candidate_certification_complete_score": 0,
        "selected_policy_ready_score": 0,
        "selected_policy_positive_score": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_exact_candidate_certification",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _pair_index(bank: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Index the twelve frozen public pairs and reject duplicate IDs."""

    rows = bank.get("selected_pair_rows", [])
    result = {str(row["pair_id"]): deepcopy(row) for row in rows}
    if len(result) != len(rows):
        raise CertificationError("duplicate_selected_pair_id")
    return result


def build_certified_artifact(
    *,
    bank: Mapping[str, Any],
    fixture: Mapping[str, Any],
    date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    """Certify label-free candidates, freeze selection, then assess held-out rows."""

    pairs = _pair_index(bank)
    bundles: list[JsonDict] = []
    attempts = sorted(bank["per_attempt_rows"], key=lambda row: int(row.get("ordinal", -1)))
    for attempt in attempts:
        pair = pairs.get(str(attempt.get("pair_id")))
        if pair is None:
            raise CertificationError(f"selected_pair_missing:{attempt.get('pair_id')}")
        bundles.append(certify_candidate(attempt, pair))
    base_rows = [row["candidate_row"] for row in bundles]
    vault = SealedLabelVault(fixture, allowed_pair_ids=set(pairs))
    calibration_labels = vault.open("calibration")
    labeled_rows = apply_labels(base_rows, "calibration", calibration_labels)
    calibration_rows = [row for row in labeled_rows if row.get("split") == "calibration"]
    calibration_metrics, selection_rows, policy, policy_hash = select_schedule(calibration_rows)
    heldout_labels = vault.open("heldout", selected_policy_hash=policy_hash)
    labeled_rows = apply_labels(labeled_rows, "heldout", heldout_labels)
    heldout_rows = [row for row in labeled_rows if row.get("split") == "heldout"]
    selected_schedule = str(policy["schedule_id"]) if policy is not None else None
    heldout_metrics = build_heldout_metric_rows(heldout_rows, selected_schedule)
    groups = build_per_group_results(heldout_rows, selected_schedule)
    headroom = headroom_rows(groups)
    headroom_count = heldout_headroom_group_count(groups)
    paired = paired_schedule_deltas(heldout_rows)
    for row in paired:
        row["selected_schedule_in_comparison"] = selected_schedule in {
            row["schedule_a"],
            row["schedule_b"],
        }
    complete_score = candidate_completion_score(
        labeled_rows, [row["solver_agreement_row"] for row in bundles]
    )
    ready_score = policy_ready_score(policy, policy_hash, selection_rows, heldout_rows)
    positive_score = policy_positive_score(policy, heldout_metrics, headroom_count, ready_score)
    verdict_class, honest_verdict = verdict_for_scores(
        complete_score,
        ready_score,
        positive_score,
        [bundle["solver_agreement_row"] for bundle in bundles],
    )
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(list(checks)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": deepcopy(labeled_rows),
        "per_candidate_rows": deepcopy(labeled_rows),
        "parser_outcome_rows": [row["parser_outcome_row"] for row in bundles],
        "exact_witness_rows": [row["exact_witness_row"] for row in bundles],
        "solver_agreement_rows": [row["solver_agreement_row"] for row in bundles],
        "calibration_metric_rows": calibration_metrics,
        "policy_selection_rows": selection_rows,
        "selected_policy": policy,
        "selected_policy_hash": policy_hash,
        "label_opening_rows": deepcopy(vault.opening_rows),
        "heldout_metric_rows": heldout_metrics,
        "per_group_results": groups,
        "paired_schedule_delta_rows": paired,
        "heldout_headroom_rows": headroom,
        "heldout_headroom_group_count": headroom_count,
        "candidate_certification_complete_score": complete_score,
        "selected_policy_ready_score": ready_score,
        "selected_policy_positive_score": positive_score,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": [],
        "verifier_is_oracle": True,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_from_paths(repo_root: Path, *, date: str = RUN_DATE) -> JsonDict:
    """Preflight frozen files and build either certification or blocked evidence."""

    started = time.monotonic()
    hashes = source_artifact_hashes(repo_root)
    try:
        inputs = load_inputs(repo_root)
    except (OSError, json.JSONDecodeError, ValueError):
        inputs = {"bank": {}, "fixture": {}}
    checks = check_preconditions(repo_root, inputs)
    if any(row.get("passed") is not True for row in checks):
        return build_blocked_artifact(
            date=date,
            duration_s=time.monotonic() - started,
            checks=checks,
            source_hashes=hashes,
        )
    artifact = build_certified_artifact(
        bank=inputs["bank"],
        fixture=inputs["fixture"],
        date=date,
        duration_s=0.0,
        checks=checks,
        source_hashes=hashes,
    )
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _validate_required(artifact: Mapping[str, Any]) -> None:
    """Check required fields, principles, bare values, and fixed declarations."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"required_fields_missing:{missing}")
    principles = artifact.get("field_principles", {})
    if not isinstance(principles, Mapping) or any(
        not isinstance(principles.get(field), str) or not str(principles.get(field)).strip()
        for field in REQUIRED_ARTIFACT_FIELDS
    ):
        raise ValueError("field_principles_incomplete")
    for field in (
        "heldout_headroom_group_count",
        "candidate_certification_complete_score",
        "selected_policy_ready_score",
        "selected_policy_positive_score",
    ):
        if type(artifact.get(field)) is not int:
            raise ValueError(f"not_bare_int:{field}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_oracle_declaration_mismatch")
    if artifact.get("verdict_class") == "positive":
        raise ValueError("oracle_distinct_positive_forbidden")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Recompute all row-derived scores, metrics, verdicts, and stable hashes."""

    _validate_required(artifact)
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum_mismatch")
    if artifact.get("verdict_class") == "blocked":
        if (
            not artifact.get("gate_check_summary")
            or artifact.get("honest_verdict") != "blocked_exact_candidate_certification"
            or any(
                artifact.get(field) != 0
                for field in (
                    "candidate_certification_complete_score",
                    "selected_policy_ready_score",
                    "selected_policy_positive_score",
                )
            )
        ):
            raise ValueError("blocked_artifact_mismatch")
        return
    rows = artifact.get("per_candidate_rows", [])
    if artifact.get("rows") != rows:
        raise ValueError("rows_projection_mismatch")
    if len(rows) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("candidate_row_count_mismatch")
    if artifact.get("selected_policy_hash") != sha256_json(artifact.get("selected_policy")):
        raise ValueError("selected_policy_hash_mismatch")
    calibration = [row for row in rows if row.get("split") == "calibration"]
    heldout = [row for row in rows if row.get("split") == "heldout"]
    metrics, ranking, policy, policy_hash = select_schedule(calibration)
    if artifact.get("calibration_metric_rows") != metrics:
        raise ValueError("calibration_metric_rows_mismatch")
    if (
        artifact.get("policy_selection_rows") != ranking
        or artifact.get("selected_policy") != policy
    ):
        raise ValueError("policy_selection_rows_mismatch")
    selected_schedule = str(policy["schedule_id"])
    expected_heldout_metrics = build_heldout_metric_rows(heldout, selected_schedule)
    if artifact.get("heldout_metric_rows") != expected_heldout_metrics:
        raise ValueError("heldout_metric_rows_mismatch")
    groups = build_per_group_results(heldout, selected_schedule)
    if artifact.get("per_group_results") != groups:
        raise ValueError("per_group_results_mismatch")
    expected_headroom = headroom_rows(groups)
    if artifact.get("heldout_headroom_rows") != expected_headroom:
        raise ValueError("heldout_headroom_rows_mismatch")
    expected_count = heldout_headroom_group_count(groups)
    if artifact.get("heldout_headroom_group_count") != expected_count:
        raise ValueError("heldout_headroom_group_count_mismatch")
    expected_paired = paired_schedule_deltas(heldout)
    for row in expected_paired:
        row["selected_schedule_in_comparison"] = selected_schedule in {
            row["schedule_a"],
            row["schedule_b"],
        }
    if artifact.get("paired_schedule_delta_rows") != expected_paired:
        raise ValueError("paired_schedule_delta_rows_mismatch")
    complete = candidate_completion_score(rows, artifact.get("solver_agreement_rows", []))
    ready = policy_ready_score(
        policy, policy_hash, artifact.get("policy_selection_rows", []), heldout
    )
    positive = policy_positive_score(policy, expected_heldout_metrics, expected_count, ready)
    if artifact.get("candidate_certification_complete_score") != complete:
        raise ValueError("candidate_certification_complete_score_mismatch")
    if artifact.get("selected_policy_ready_score") != ready:
        raise ValueError("selected_policy_ready_score_mismatch")
    if artifact.get("selected_policy_positive_score") != positive:
        raise ValueError("selected_policy_positive_score_mismatch")
    expected_class = (
        "circular_positive"
        if complete == ready == positive == 1
        else ("null" if complete == ready == 1 else artifact.get("verdict_class"))
    )
    if artifact.get("verdict_class") != expected_class:
        raise ValueError("verdict_class_mismatch")
    if expected_class == "circular_positive" and not str(
        artifact.get("honest_verdict", "")
    ).startswith("complete_"):
        raise ValueError("honest_verdict_mismatch")
    opening_rows = artifact.get("label_opening_rows")
    if opening_rows != [
        {
            "split": "calibration",
            "opening_sequence": 1,
            "selected_policy_hash": None,
            "label_count": 6,
        },
        {
            "split": "heldout",
            "opening_sequence": 2,
            "selected_policy_hash": policy_hash,
            "label_count": 6,
        },
    ]:
        raise ValueError("label_opening_rows_mismatch")


def write_json_atomic(path: Path, value: Any) -> None:
    """Replace one JSON file atomically so readers never see partial evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(*, date: str, repo_root: Path, output_path: Path | None = None) -> JsonDict:
    """Build, validate, and write the requested terminal artifact."""

    artifact = build_from_paths(repo_root, date=date)
    validate_artifact(artifact)
    write_json_atomic(output_path or repo_root / RESULT_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Expose the registered experiment command and optional isolated output path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args(argv)
    artifact = run(date=args.date, repo_root=args.repo_root, output_path=args.output_path)
    print(
        json.dumps(
            {
                "candidate_certification_complete_score": artifact[
                    "candidate_certification_complete_score"
                ],
                "selected_policy_ready_score": artifact["selected_policy_ready_score"],
                "selected_policy_positive_score": artifact["selected_policy_positive_score"],
                "heldout_headroom_group_count": artifact["heldout_headroom_group_count"],
                "verdict_class": artifact["verdict_class"],
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution boundary.
    raise SystemExit(main())
