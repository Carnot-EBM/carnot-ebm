#!/usr/bin/env python3
"""Exp5443 deterministic verifier-potential prefix fixture.

Spec refs: REQ-SAFE-5443, SCENARIO-SAFE-5443.

The SOTA decoding pilot needs cheap signals it can query while a structured
answer is still incomplete.  This module builds those signals as deterministic
fixture data rather than model output: prefix potentials can suggest promising
partial rows, but every completed row is still judged by an exact final
verifier.  That separation is the important safety boundary because a prefix
can look locally useful and still finish as an invalid structured answer.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5443_verifier_potential_prefix_fixture_v495.json")
EXPERIMENT_ID = "experiment_5443_verifier_potential_prefix_fixture_v495"
TASK_ID = "exp5443-v495-verifier-potential-prefix-fixture"
MILESTONE = "2026.07.495"
RUN_DATE = "2026-07-08"
SCHEMA = "carnot.experiment_5443.verifier_potential_prefix_fixture.v495"
SPEC_REFS = ("REQ-SAFE-5443", "SCENARIO-SAFE-5443")
RANDOM_SEED = 5443
INFERENCE_SUBSTRATE = "deterministic_verifier_fixture_no_llm"
TERMINAL_PREFIXES = ("complete:", "blocked:")
FINAL_VERIFIER_COST_UNITS = 5

REQUIRED_CONSTRAINT_FAMILIES = (
    "schema_only_trap",
    "semantic_contradiction",
    "unreachable_tool_action",
    "arithmetic_finite_domain",
    "ontology_triple_update",
    "api_call_witness",
    "benign",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "fixture_count": "fixture coverage.",
    "constraint_family_counts": "taxonomy coverage.",
    "prefix_potential_functions": "reproducible scoring definition.",
    "exact_final_authority": "no learned-score certificate.",
    "prefix_final_disagreement_cases": "detects misleading partial scores.",
    "reward_evaluation_budget": "generation guidance cost accounting.",
    "row_provenance_checksum": "row-level reproducibility.",
    "reproducibility_checksum": "artifact reproducibility.",
    "metric_independence_checks_passed": "tautology prevention.",
    "verifier_potential_fixture_ready": "downstream gate.",
    "inference_substrate": "no hidden live model inference.",
    "honest_verdict": 'terminal status; start with "complete:" or "blocked:".',
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)

PREFIX_POTENTIAL_FUNCTIONS: tuple[JsonDict, ...] = (
    {
        "function_id": "schema_key_coverage",
        "cost_units": 1,
        "score_definition": (
            "score is present required top-level keys divided by required keys; "
            "accept only when all required keys are present"
        ),
        "decision_semantics": "accept means prefix guidance only, never final safety.",
        "unknown_prefix_policy": "abstain with neutral score 0.0 when no required key is present.",
        "monotone": True,
        "monotone_scope": "required-key coverage over additive prefixes only.",
    },
    {
        "function_id": "semantic_pair_consistency",
        "cost_units": 2,
        "score_definition": (
            "accept when object and negated_object are both present and different; "
            "reject when both are present and equal"
        ),
        "decision_semantics": "accept means the visible pair is consistent only.",
        "unknown_prefix_policy": "abstain with neutral score 0.0 until both semantic fields exist.",
        "monotone": False,
        "monotone_scope": "later fields can introduce a contradiction.",
    },
    {
        "function_id": "action_allowlist_potential",
        "cost_units": 2,
        "score_definition": "accept when the visible tool name is in the deterministic allowlist.",
        "decision_semantics": "accept does not prove state reachability.",
        "unknown_prefix_policy": "abstain with neutral score 0.0 when tool is absent.",
        "monotone": True,
        "monotone_scope": "tool-name membership is stable once the tool field is emitted.",
    },
    {
        "function_id": "finite_domain_bounds",
        "cost_units": 2,
        "score_definition": (
            "accept when all declared finite-domain fields visible so far are integers "
            "inside their row-specific bounds"
        ),
        "decision_semantics": "accept does not prove cross-field arithmetic relations.",
        "unknown_prefix_policy": "abstain with neutral score 0.0 when no domain field is visible.",
        "monotone": False,
        "monotone_scope": "later numeric fields can violate the finite-domain relation.",
    },
    {
        "function_id": "ontology_shape_potential",
        "cost_units": 2,
        "score_definition": (
            "accept when subject, predicate, and object are visible and match the "
            "row ontology type signature"
        ),
        "decision_semantics": "accept does not prove acyclic ontology update closure.",
        "unknown_prefix_policy": "abstain with neutral score 0.0 until triple fields exist.",
        "monotone": False,
        "monotone_scope": "later updates can create a cycle or type conflict.",
    },
    {
        "function_id": "api_shape_potential",
        "cost_units": 2,
        "score_definition": "accept when method and path match an allowed deterministic API call.",
        "decision_semantics": "accept does not prove witness signature validity.",
        "unknown_prefix_policy": "abstain with neutral score 0.0 until method and path exist.",
        "monotone": False,
        "monotone_scope": "later witness fields can invalidate the call.",
    },
)


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str | None = None,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build the Exp5443 artifact and optionally write it to disk."""

    root_path = Path(root)
    destination = _destination(root_path, result_path)
    artifact = build_artifact(tests_run=tests_run)
    if write:
        _write_json(destination, artifact)
    return artifact


def build_artifact(
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Assemble the terminal fixture artifact from deterministic rows."""

    rows = build_fixture_rows()
    metrics = derive_metrics(rows)
    budget = derive_reward_budget(rows)
    ready = bool(
        metrics["fixture_count"] > 0
        and set(REQUIRED_CONSTRAINT_FAMILIES).issubset(metrics["constraint_family_counts"])
        and metrics["exact_final_rows"] == metrics["fixture_count"]
        and metrics["prefix_final_disagreement_cases"] > 0
        and metrics["metric_independence_checks_passed"]
        and budget["per_row"]
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "status": "complete" if ready else "blocked",
        "fixture_count": metrics["fixture_count"],
        "constraint_family_counts": metrics["constraint_family_counts"],
        "prefix_potential_functions": [dict(row) for row in PREFIX_POTENTIAL_FUNCTIONS],
        "exact_final_authority": True,
        "prefix_final_disagreement_cases": metrics["prefix_final_disagreement_cases"],
        "reward_evaluation_budget": budget,
        "row_provenance_checksum": "0" * 64,
        "reproducibility_checksum": "0" * 64,
        "metric_independence_checks_passed": metrics["metric_independence_checks_passed"],
        "verifier_potential_fixture_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, metrics),
        "fixture_rows": rows,
        "row_checksums": [row["row_checksum"] for row in rows],
        "fixture_checksums": [row["fixture_checksum"] for row in rows],
        "metric_details": metrics,
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": _normalise_tests_run(tests_run),
        "research_conductor_modified": False,
    }
    artifact["row_provenance_checksum"] = row_provenance_checksum(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def build_fixture_rows() -> list[JsonDict]:
    """Return scored fixture rows with exact final checks and checksums attached."""

    rows: list[JsonDict] = []
    for fixture in _base_fixtures():
        row = copy.deepcopy(fixture)
        row["fixture_checksum"] = fixture_checksum(row)
        row["prefixes"] = [_score_prefix(row, prefix) for prefix in row.get("prefixes", [])]
        row["exact_final_verdict"] = exact_final_verdict(row)
        row["accepted_by_final_authority"] = bool(row["exact_final_verdict"]["accepted"])
        row["reward_evaluation_budget"] = derive_row_budget(row)
        row["row_checksum"] = row_checksum(row)
        rows.append(row)
    return rows


def derive_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute terminal aggregate metrics from row predicates only."""

    row_list = [dict(row) for row in rows if isinstance(row, Mapping)]
    family_counts = dict(
        sorted(Counter(str(row.get("constraint_family")) for row in row_list).items())
    )
    final_rejected_ids = [
        str(row["row_id"]) for row in row_list if _exact_final_accepted(row) is False
    ]
    disagreement_ids = [
        str(row["row_id"])
        for row in row_list
        if _exact_final_accepted(row) is False and _has_accepted_prefix(row)
    ]
    exact_final_rows = sum(
        1 for row in row_list if _mapping(row.get("exact_final_verdict")).get("verified") is True
    )
    row_checksums_match = [row.get("row_checksum") for row in row_list] == [
        row_checksum(row) for row in row_list
    ]
    fixture_checksums_match = [row.get("fixture_checksum") for row in row_list] == [
        fixture_checksum(row) for row in row_list
    ]
    predicate_support = {
        "fixture_count": "count every scored fixture row",
        "constraint_family_counts": "group rows by constraint_family",
        "final_rejected": "exact_final_verdict.accepted is false",
        "prefix_final_disagreement": (
            "exact_final_verdict.accepted is false and any prefix accepted_by_potential"
        ),
        "reward_budget": "sum deterministic potential and exact-final evaluation costs",
    }
    metric_independence = bool(
        row_list
        and row_checksums_match
        and fixture_checksums_match
        and len(set(predicate_support.values())) == len(predicate_support)
        and disagreement_ids != final_rejected_ids
    )
    return {
        "fixture_count": len(row_list),
        "constraint_family_counts": family_counts,
        "exact_final_rows": exact_final_rows,
        "final_rejected_row_ids": final_rejected_ids,
        "prefix_final_disagreement_row_ids": disagreement_ids,
        "prefix_final_disagreement_cases": len(disagreement_ids),
        "row_checksums_match": row_checksums_match,
        "fixture_checksums_match": fixture_checksums_match,
        "predicate_support": predicate_support,
        "metric_independence_checks_passed": metric_independence,
    }


def derive_reward_budget(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute reward-evaluation cost per row and per accepted prefix."""

    per_row: list[JsonDict] = []
    per_accepted_prefix: list[JsonDict] = []
    for row in rows:
        row_budget = derive_row_budget(row)
        per_row.append(row_budget)
        for prefix in row.get("prefixes", []):
            if isinstance(prefix, Mapping) and prefix.get("accepted_by_potential") is True:
                per_accepted_prefix.append(
                    {
                        "row_id": row["row_id"],
                        "fixture_id": row["fixture_id"],
                        "prefix_id": prefix["prefix_id"],
                        "accepted_functions": list(prefix["accepted_functions"]),
                        "cost_units": prefix["reward_evaluation_cost_units"],
                    }
                )
    potential_evaluations = sum(row["potential_evaluations"] for row in per_row)
    final_evaluations = sum(row["final_verifier_evaluations"] for row in per_row)
    return {
        "cost_unit": "one deterministic verifier-potential or exact-final check invocation",
        "potential_function_cost_units": {
            row["function_id"]: row["cost_units"] for row in PREFIX_POTENTIAL_FUNCTIONS
        },
        "exact_final_verifier_cost_units": FINAL_VERIFIER_COST_UNITS,
        "total_potential_evaluations": potential_evaluations,
        "total_final_verifications": final_evaluations,
        "accepted_prefix_count": len(per_accepted_prefix),
        "total_cost_units": sum(row["total_cost_units"] for row in per_row),
        "per_row": per_row,
        "per_accepted_prefix": per_accepted_prefix,
    }


def derive_row_budget(row: Mapping[str, Any]) -> JsonDict:
    """Summarize deterministic potential and final-verifier costs for one row."""

    prefixes = [prefix for prefix in row.get("prefixes", []) if isinstance(prefix, Mapping)]
    prefix_cost = sum(int(prefix.get("reward_evaluation_cost_units", 0)) for prefix in prefixes)
    accepted_prefixes = [
        prefix for prefix in prefixes if prefix.get("accepted_by_potential") is True
    ]
    accepted_prefix_cost = sum(
        int(prefix.get("reward_evaluation_cost_units", 0)) for prefix in accepted_prefixes
    )
    potential_evaluations = sum(len(prefix.get("potential_evaluations", [])) for prefix in prefixes)
    return {
        "row_id": row["row_id"],
        "fixture_id": row["fixture_id"],
        "prefix_count": len(prefixes),
        "potential_evaluations": potential_evaluations,
        "accepted_prefix_count": len(accepted_prefixes),
        "prefix_cost_units": prefix_cost,
        "accepted_prefix_cost_units": accepted_prefix_cost,
        "final_verifier_evaluations": 1,
        "final_verifier_cost_units": FINAL_VERIFIER_COST_UNITS,
        "total_cost_units": prefix_cost + FINAL_VERIFIER_COST_UNITS,
    }


def exact_final_verdict(row: Mapping[str, Any]) -> JsonDict:
    """Run deterministic final checks over a completed row."""

    final_output = _mapping(row.get("final_output"))
    family = str(row.get("constraint_family"))
    reasons: list[str] = []
    if not _schema_exact(row, final_output):
        reasons.append("schema_exact_failed")
    family_reason = _family_exact_reason(row, final_output, family)
    if family_reason:
        reasons.append(family_reason)
    accepted = not reasons
    return {
        "verified": True,
        "authority": "exact_final_verifier",
        "accepted": accepted,
        "failure_reasons": reasons,
        "verifiers_run": ["schema_exact", f"{family}_exact"],
        "overrides_prefix_potential": bool(_has_accepted_prefix(row) and not accepted),
    }


def fixture_checksum(row: Mapping[str, Any]) -> str:
    """Hash the fixture definition excluding computed scores and checksums."""

    payload = {
        "row_id": row.get("row_id"),
        "fixture_id": row.get("fixture_id"),
        "constraint_family": row.get("constraint_family"),
        "required_keys": row.get("required_keys"),
        "allowed_keys": row.get("allowed_keys"),
        "domain_fields": row.get("domain_fields"),
        "finite_domain": row.get("finite_domain"),
        "ontology": row.get("ontology"),
        "api": row.get("api"),
        "final_output": row.get("final_output"),
        "prefixes": [
            {"prefix_id": prefix.get("prefix_id"), "fields": prefix.get("fields")}
            for prefix in row.get("prefixes", [])
            if isinstance(prefix, Mapping)
        ],
    }
    return _sha256_json(payload)


def row_checksum(row: Mapping[str, Any]) -> str:
    """Hash a fully scored row while excluding the row checksum itself."""

    payload = {key: value for key, value in row.items() if key != "row_checksum"}
    return _sha256_json(payload)


def row_provenance_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash row-level reproducibility material and potential definitions."""

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "fixture_count": artifact.get("fixture_count"),
        "constraint_family_counts": artifact.get("constraint_family_counts"),
        "prefix_potential_functions": artifact.get("prefix_potential_functions"),
        "row_checksums": artifact.get("row_checksums"),
        "fixture_checksums": artifact.get("fixture_checksums"),
        "reward_evaluation_budget": artifact.get("reward_evaluation_budget"),
    }
    return _sha256_json(payload)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact payload without the self-referential checksum."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return _sha256_json(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp5443 artifact cannot support the downstream gate."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, row-recomputation, and budget validation errors."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles must match REQ-SAFE-5443")
    rows = artifact.get("fixture_rows")
    if not isinstance(rows, list):
        errors.append("fixture_rows must be a list")
        rows = []
    if not _bare_non_negative_int(artifact.get("fixture_count")):
        errors.append("fixture_count must be a non-negative integer")
    if not isinstance(artifact.get("constraint_family_counts"), Mapping):
        errors.append("constraint_family_counts must be a dict")
    if artifact.get("prefix_potential_functions") != list(PREFIX_POTENTIAL_FUNCTIONS):
        errors.append("prefix_potential_functions must match deterministic definitions")
    if artifact.get("exact_final_authority") is not True:
        errors.append("exact_final_authority must be true")
    if not _bare_non_negative_int(artifact.get("prefix_final_disagreement_cases")):
        errors.append("prefix_final_disagreement_cases must be a non-negative integer")
    if not isinstance(artifact.get("reward_evaluation_budget"), Mapping):
        errors.append("reward_evaluation_budget must be a dict")
    if not _sha256_text(artifact.get("row_provenance_checksum")):
        errors.append("row_provenance_checksum must be a sha256 hex string")
    if not _sha256_text(artifact.get("reproducibility_checksum")):
        errors.append("reproducibility_checksum must be a sha256 hex string")
    for field in (
        "metric_independence_checks_passed",
        "verifier_potential_fixture_ready",
    ):
        if type(artifact.get(field)) is not bool:
            errors.append(f"{field} must be boolean")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be deterministic_verifier_fixture_no_llm")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or "\n" in verdict or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked:")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py must not be modified")
    if isinstance(rows, list):
        errors.extend(_row_integrity_errors(rows))
        errors.extend(_aggregate_recompute_errors(artifact, rows))
    if artifact.get("row_provenance_checksum") != row_provenance_checksum(artifact):
        errors.append("row_provenance_checksum must match row provenance")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum must match artifact payload")
    if artifact.get("verifier_potential_fixture_ready") is True:
        if artifact.get("exact_final_authority") is not True:
            errors.append("verifier_potential_fixture_ready requires exact final authority")
        if artifact.get("metric_independence_checks_passed") is not True:
            errors.append("verifier_potential_fixture_ready requires metric independence")
        if int(artifact.get("prefix_final_disagreement_cases", 0)) <= 0:
            errors.append("verifier_potential_fixture_ready requires disagreement cases")
        if not set(REQUIRED_CONSTRAINT_FAMILIES).issubset(
            set(artifact.get("constraint_family_counts", {}))
        ):
            errors.append("verifier_potential_fixture_ready requires all fixture families")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for producing the Exp5443 result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=None)
    args = parser.parse_args(argv)
    artifact = run(root=args.root, result_path=args.result_path, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact["verifier_potential_fixture_ready"] else 1


def _base_fixtures() -> list[JsonDict]:
    return [
        {
            "row_id": "5443-001",
            "fixture_id": "schema_extra_admin_override",
            "constraint_family": "schema_only_trap",
            "required_keys": ["kind", "payload"],
            "allowed_keys": ["kind", "payload"],
            "prefixes": [
                {"prefix_id": "5443-001:empty", "fields": {}},
                {
                    "prefix_id": "5443-001:required-keys",
                    "fields": {"kind": "claim", "payload": {"answer": "yes"}},
                },
            ],
            "final_output": {
                "kind": "claim",
                "payload": {"answer": "yes"},
                "admin_override": True,
            },
        },
        {
            "row_id": "5443-002",
            "fixture_id": "schema_missing_payload",
            "constraint_family": "schema_only_trap",
            "required_keys": ["kind", "payload"],
            "allowed_keys": ["kind", "payload"],
            "prefixes": [{"prefix_id": "5443-002:empty", "fields": {}}],
            "final_output": {"kind": "claim"},
        },
        {
            "row_id": "5443-003",
            "fixture_id": "semantic_open_and_not_open",
            "constraint_family": "semantic_contradiction",
            "required_keys": ["subject", "relation", "object"],
            "allowed_keys": ["subject", "relation", "object", "negated_object"],
            "prefixes": [
                {"prefix_id": "5443-003:empty", "fields": {}},
                {
                    "prefix_id": "5443-003:visible-claim",
                    "fields": {
                        "subject": "door-17",
                        "relation": "state",
                        "object": "open",
                    },
                },
            ],
            "final_output": {
                "subject": "door-17",
                "relation": "state",
                "object": "open",
                "negated_object": "open",
            },
        },
        {
            "row_id": "5443-004",
            "fixture_id": "unreachable_cancel_locked_order",
            "constraint_family": "unreachable_tool_action",
            "required_keys": ["tool"],
            "allowed_keys": ["tool", "order_state", "lock_active"],
            "action_constraints": {
                "allowed_tools": ["cancel_order", "ship_order", "lookup_inventory"],
                "reachable_states": ["paid"],
            },
            "prefixes": [
                {"prefix_id": "5443-004:empty", "fields": {}},
                {"prefix_id": "5443-004:tool-name", "fields": {"tool": "cancel_order"}},
            ],
            "final_output": {
                "tool": "cancel_order",
                "order_state": "paid",
                "lock_active": True,
            },
        },
        {
            "row_id": "5443-005",
            "fixture_id": "arithmetic_wrong_sum",
            "constraint_family": "arithmetic_finite_domain",
            "required_keys": ["x", "y"],
            "allowed_keys": ["x", "y", "sum"],
            "domain_fields": ["x", "y", "sum"],
            "finite_domain": {"min": 0, "max": 9},
            "prefixes": [
                {"prefix_id": "5443-005:empty", "fields": {}},
                {"prefix_id": "5443-005:operands", "fields": {"x": 2, "y": 3}},
            ],
            "final_output": {"x": 2, "y": 3, "sum": 6},
        },
        {
            "row_id": "5443-006",
            "fixture_id": "ontology_cycle_update",
            "constraint_family": "ontology_triple_update",
            "required_keys": ["subject", "predicate", "object"],
            "allowed_keys": ["subject", "predicate", "object", "also_parent"],
            "ontology": {
                "predicate": "part_of",
                "object_type": "assembly",
                "type_map": {"bolt-7": "part", "assembly-1": "assembly"},
            },
            "prefixes": [
                {"prefix_id": "5443-006:empty", "fields": {}},
                {
                    "prefix_id": "5443-006:triple",
                    "fields": {
                        "subject": "bolt-7",
                        "predicate": "part_of",
                        "object": "assembly-1",
                    },
                },
            ],
            "final_output": {
                "subject": "bolt-7",
                "predicate": "part_of",
                "object": "assembly-1",
                "also_parent": "bolt-7",
            },
        },
        {
            "row_id": "5443-007",
            "fixture_id": "api_refund_bad_witness",
            "constraint_family": "api_call_witness",
            "required_keys": ["method", "path"],
            "allowed_keys": ["method", "path", "witness"],
            "api": {
                "allowed_calls": [["POST", "/orders/42/refund"]],
                "expected_signature": "sig:refund:42:approved",
            },
            "prefixes": [
                {"prefix_id": "5443-007:empty", "fields": {}},
                {
                    "prefix_id": "5443-007:method-path",
                    "fields": {"method": "POST", "path": "/orders/42/refund"},
                },
            ],
            "final_output": {
                "method": "POST",
                "path": "/orders/42/refund",
                "witness": {"signature": "sig:refund:42:forged", "nonce_reused": True},
            },
        },
        {
            "row_id": "5443-008",
            "fixture_id": "benign_inventory_claim",
            "constraint_family": "benign",
            "required_keys": ["claim_key", "claim_value"],
            "allowed_keys": ["claim_key", "claim_value", "evidence"],
            "prefixes": [
                {"prefix_id": "5443-008:empty", "fields": {}},
                {
                    "prefix_id": "5443-008:claim",
                    "fields": {"claim_key": "bolt_count", "claim_value": 4},
                },
            ],
            "final_output": {
                "claim_key": "bolt_count",
                "claim_value": 4,
                "evidence": {"bolt_count": 4},
            },
        },
    ]


def _score_prefix(row: Mapping[str, Any], prefix: Mapping[str, Any]) -> JsonDict:
    fields = _mapping(prefix.get("fields"))
    evaluations = [_evaluate_potential(spec, row, fields) for spec in PREFIX_POTENTIAL_FUNCTIONS]
    accepted_functions = [
        evaluation["function_id"]
        for evaluation in evaluations
        if evaluation["decision"] == "accept"
    ]
    cost_units = sum(int(evaluation["cost_units"]) for evaluation in evaluations)
    return {
        "prefix_id": prefix["prefix_id"],
        "fields": copy.deepcopy(fields),
        "potential_evaluations": evaluations,
        "accepted_by_potential": bool(accepted_functions),
        "accepted_functions": accepted_functions,
        "reward_evaluation_cost_units": cost_units,
    }


def _evaluate_potential(
    spec: Mapping[str, Any],
    row: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> JsonDict:
    function_id = str(spec["function_id"])
    if function_id == "schema_key_coverage":
        score, decision, evidence = _schema_key_potential(row, fields)
    elif function_id == "semantic_pair_consistency":
        score, decision, evidence = _semantic_pair_potential(fields)
    elif function_id == "action_allowlist_potential":
        score, decision, evidence = _action_allowlist_potential(row, fields)
    elif function_id == "finite_domain_bounds":
        score, decision, evidence = _finite_domain_potential(row, fields)
    elif function_id == "ontology_shape_potential":
        score, decision, evidence = _ontology_shape_potential(row, fields)
    else:
        score, decision, evidence = _api_shape_potential(row, fields)
    return {
        "function_id": function_id,
        "score": score,
        "decision": decision,
        "evidence": evidence,
        "cost_units": spec["cost_units"],
        "monotone": spec["monotone"],
    }


def _schema_key_potential(
    row: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> tuple[float, str, JsonDict]:
    required = list(row.get("required_keys", []))
    present = [key for key in required if key in fields]
    if not present:
        return 0.0, "abstain", {"present_required_keys": []}
    score = len(present) / len(required)
    decision = "accept" if len(present) == len(required) else "abstain"
    return score, decision, {"present_required_keys": present}


def _semantic_pair_potential(fields: Mapping[str, Any]) -> tuple[float, str, JsonDict]:
    if "object" not in fields or "negated_object" not in fields:
        return 0.0, "abstain", {"visible_pair": False}
    consistent = fields["object"] != fields["negated_object"]
    return (
        0.75 if consistent else 0.0,
        "accept" if consistent else "reject",
        {"visible_pair": True, "consistent": consistent},
    )


def _action_allowlist_potential(
    row: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> tuple[float, str, JsonDict]:
    if "tool" not in fields:
        return 0.0, "abstain", {"tool_visible": False}
    allowed = set(_mapping(row.get("action_constraints")).get("allowed_tools", []))
    tool_allowed = fields["tool"] in allowed
    return (
        0.6 if tool_allowed else 0.0,
        "accept" if tool_allowed else "reject",
        {"tool_visible": True, "tool_allowed": tool_allowed},
    )


def _finite_domain_potential(
    row: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> tuple[float, str, JsonDict]:
    domain_fields = [key for key in row.get("domain_fields", []) if key in fields]
    if not domain_fields:
        return 0.0, "abstain", {"domain_fields_visible": []}
    bounds = _mapping(row.get("finite_domain"))
    lower = int(bounds.get("min", 0))
    upper = int(bounds.get("max", 9))
    in_domain = all(
        isinstance(fields[key], int) and lower <= int(fields[key]) <= upper for key in domain_fields
    )
    all_visible = set(row.get("domain_fields", [])).issubset(fields)
    decision = "accept" if in_domain and all_visible else "abstain"
    return (
        0.7 if in_domain else 0.0,
        decision if in_domain else "reject",
        {"domain_fields_visible": domain_fields, "in_domain": in_domain},
    )


def _ontology_shape_potential(
    row: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> tuple[float, str, JsonDict]:
    if not {"subject", "predicate", "object"}.issubset(fields):
        return 0.0, "abstain", {"triple_visible": False}
    ontology = _mapping(row.get("ontology"))
    type_map = _mapping(ontology.get("type_map"))
    shape_ok = fields["predicate"] == ontology.get("predicate") and type_map.get(
        fields["object"]
    ) == ontology.get("object_type")
    return (
        0.7 if shape_ok else 0.0,
        "accept" if shape_ok else "reject",
        {"triple_visible": True, "shape_ok": shape_ok},
    )


def _api_shape_potential(
    row: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> tuple[float, str, JsonDict]:
    if "method" not in fields or "path" not in fields:
        return 0.0, "abstain", {"method_path_visible": False}
    api = _mapping(row.get("api"))
    allowed_calls = {tuple(call) for call in api.get("allowed_calls", [])}
    call_allowed = (fields["method"], fields["path"]) in allowed_calls
    return (
        0.7 if call_allowed else 0.0,
        "accept" if call_allowed else "reject",
        {"method_path_visible": True, "call_allowed": call_allowed},
    )


def _schema_exact(row: Mapping[str, Any], final_output: Mapping[str, Any]) -> bool:
    keys = set(final_output)
    required = set(row.get("required_keys", []))
    allowed = set(row.get("allowed_keys", []))
    return required.issubset(keys) and keys.issubset(allowed)


def _family_exact_reason(
    row: Mapping[str, Any],
    final_output: Mapping[str, Any],
    family: str,
) -> str | None:
    if family == "semantic_contradiction" and (
        final_output.get("object") == final_output.get("negated_object")
    ):
        return "semantic_contradiction_detected"
    if family == "unreachable_tool_action":
        return _action_exact_reason(row, final_output)
    if family == "arithmetic_finite_domain":
        return _arithmetic_exact_reason(row, final_output)
    if family == "ontology_triple_update":
        return _ontology_exact_reason(row, final_output)
    if family == "api_call_witness":
        return _api_exact_reason(row, final_output)
    if family == "benign":
        return _benign_exact_reason(final_output)
    return None


def _action_exact_reason(row: Mapping[str, Any], final_output: Mapping[str, Any]) -> str | None:
    constraints = _mapping(row.get("action_constraints"))
    if final_output.get("tool") not in set(constraints.get("allowed_tools", [])):
        return "tool_not_allowed"
    if final_output.get("order_state") not in set(constraints.get("reachable_states", [])):
        return "order_state_unreachable"
    if final_output.get("lock_active") is True:
        return "lock_blocks_tool_action"
    return None


def _arithmetic_exact_reason(row: Mapping[str, Any], final_output: Mapping[str, Any]) -> str | None:
    bounds = _mapping(row.get("finite_domain"))
    lower = int(bounds.get("min", 0))
    upper = int(bounds.get("max", 9))
    domain_fields = list(row.get("domain_fields", []))
    in_domain = all(
        isinstance(final_output.get(key), int) and lower <= int(final_output[key]) <= upper
        for key in domain_fields
    )
    if not in_domain:
        return "finite_domain_failed"
    if final_output.get("sum") != final_output.get("x") + final_output.get("y"):
        return "arithmetic_relation_failed"
    return None


def _ontology_exact_reason(row: Mapping[str, Any], final_output: Mapping[str, Any]) -> str | None:
    ontology = _mapping(row.get("ontology"))
    type_map = _mapping(ontology.get("type_map"))
    if final_output.get("predicate") != ontology.get("predicate"):
        return "ontology_predicate_failed"
    if type_map.get(final_output.get("object")) != ontology.get("object_type"):
        return "ontology_object_type_failed"
    if final_output.get("also_parent") == final_output.get("subject"):
        return "ontology_cycle_detected"
    return None


def _api_exact_reason(row: Mapping[str, Any], final_output: Mapping[str, Any]) -> str | None:
    api = _mapping(row.get("api"))
    allowed_calls = {tuple(call) for call in api.get("allowed_calls", [])}
    if (final_output.get("method"), final_output.get("path")) not in allowed_calls:
        return "api_call_not_allowed"
    witness = _mapping(final_output.get("witness"))
    if witness.get("signature") != api.get("expected_signature"):
        return "api_witness_signature_failed"
    if witness.get("nonce_reused") is True:
        return "api_witness_nonce_reused"
    return None


def _benign_exact_reason(final_output: Mapping[str, Any]) -> str | None:
    evidence = _mapping(final_output.get("evidence"))
    if evidence.get(final_output.get("claim_key")) != final_output.get("claim_value"):
        return "benign_evidence_mismatch"
    return None


def _row_integrity_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for row in rows:
        if row.get("fixture_checksum") != fixture_checksum(row):
            errors.append("fixture_checksums must match fixture definitions")
        expected_verdict = exact_final_verdict(row)
        if row.get("exact_final_verdict") != expected_verdict:
            errors.append("exact final verdict must match recomputation")
        expected_budget = derive_row_budget(row)
        if row.get("reward_evaluation_budget") != expected_budget:
            errors.append("reward_evaluation_budget row entry must match recomputation")
        if row.get("row_checksum") != row_checksum(row):
            errors.append("row_checksums must match scored rows")
    return errors


def _aggregate_recompute_errors(
    artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    errors: list[str] = []
    metrics = derive_metrics(rows)
    budget = derive_reward_budget(rows)
    for field in (
        "fixture_count",
        "constraint_family_counts",
        "prefix_final_disagreement_cases",
        "metric_independence_checks_passed",
    ):
        if artifact.get(field) != metrics.get(field):
            errors.append(f"{field} must match row recomputation")
    if artifact.get("metric_details") != metrics:
        errors.append("metric_details must match row recomputation")
    if artifact.get("reward_evaluation_budget") != budget:
        errors.append("reward_evaluation_budget must match row recomputation")
    if artifact.get("row_checksums") != [row.get("row_checksum") for row in rows]:
        errors.append("row_checksums must match fixture_rows")
    if artifact.get("fixture_checksums") != [row.get("fixture_checksum") for row in rows]:
        errors.append("fixture_checksums must match fixture_rows")
    return errors


def _has_accepted_prefix(row: Mapping[str, Any]) -> bool:
    return any(
        isinstance(prefix, Mapping) and prefix.get("accepted_by_potential") is True
        for prefix in row.get("prefixes", [])
    )


def _exact_final_accepted(row: Mapping[str, Any]) -> bool:
    verdict = _mapping(row.get("exact_final_verdict"))
    return bool(verdict.get("accepted"))


def _honest_verdict(ready: bool, metrics: Mapping[str, Any]) -> str:
    if ready:
        return "complete: deterministic verifier-potential prefix fixture ready"
    missing = sorted(
        set(REQUIRED_CONSTRAINT_FAMILIES) - set(_mapping(metrics.get("constraint_family_counts")))
    )
    if missing:
        return f"blocked: missing fixture families {missing}"
    return "blocked: verifier-potential fixture readiness checks failed"


def _normalise_tests_run(value: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    rows = [_normalise_test_run(row) for row in value]
    return rows or [
        {
            "command": (
                ".venv/bin/pytest tests/python/"
                "test_experiment_5443_verifier_potential_prefix_fixture_v495.py -q"
            ),
            "outcome": "not_recorded",
        }
    ]


def _normalise_test_run(value: str | Mapping[str, Any]) -> JsonDict:
    if isinstance(value, Mapping):
        return {
            "command": str(value.get("command", "")),
            "outcome": str(value.get("outcome", "not_recorded")),
        }
    return {"command": str(value), "outcome": "not_recorded"}


def _destination(root: Path, result_path: Path | str | None) -> Path:
    if result_path is None:
        return root / RESULT_RELATIVE_PATH
    path = Path(result_path)
    return path if path.is_absolute() else root / path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _bare_non_negative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
