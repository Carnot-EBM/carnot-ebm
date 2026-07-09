#!/usr/bin/env python3
"""Exp5458 deterministic minimal-core claim repair.

Spec refs: REQ-VERIFY-5458, SCENARIO-VERIFY-5458.

This module turns two V495 deterministic failure fixtures into a small
core-guided repair panel. The important boundary is that a core ID can propose
a repair, but it cannot certify the repair. The completed candidate is always
sent back through the exact Exp5443 verifier or the Exp5445 AST/KB witness
before any accept is counted.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5443_verifier_potential_prefix_fixture_v495 as vp5443
from carnot import experiment_5445_static_ast_kb_witness_constraints_v495 as astkb5445


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5458_minimal_core_claim_repair_v496.json")
EXPERIMENT_ID = "experiment_5458_minimal_core_claim_repair_v496"
TASK_ID = "exp5458-v496-minimal-core-claim-repair"
MILESTONE = "2026.07.496"
RUN_DATE = "2026-07-09"
SCHEMA = "carnot.experiment_5458.minimal_core_claim_repair.v496"
SPEC_REFS = ("REQ-VERIFY-5458", "SCENARIO-VERIFY-5458", "REQ-SAFE-5443", "REQ-CODE-5445")
RANDOM_SEED = 5458
INFERENCE_SUBSTRATE = "deterministic_solver_core_repair_no_llm"
SOURCE_ARTIFACTS = (
    vp5443.RESULT_RELATIVE_PATH,
    astkb5445.RESULT_RELATIVE_PATH,
)
SELECTED_VP_ROW_IDS = ("5443-001", "5443-003", "5443-005", "5443-007")
SELECTED_AST_ROW_IDS = (
    "fixture.nonexistent_json_method",
    "fixture.wrong_module_alias",
    "fixture.imported_symbol_missing",
    "fixture.argument_intent_mismatch",
)
TERMINAL_PREFIXES = ("complete:", "blocked:")

FIELD_PRINCIPLES: dict[str, str] = {
    "source_artifacts": "lists the exact Exp5443 and Exp5445 inputs.",
    "repair_case_count": "bounded failed-row coverage.",
    "minimal_core_success_rate": (
        "fraction of selected failures with a deterministic minimal repair core."
    ),
    "core_constraint_id_count": "number of distinct stable core IDs used for repairs.",
    "repaired_accept_rate_after_exact_recheck": "repairs accepted only after exact recheck.",
    "unrepaired_reject_rate": "original failed rows still reject under exact authority.",
    "exact_final_authority": "exact verifier or AST/KB witness is the only acceptance authority.",
    "row_provenance_checksum": "source-row and core reproducibility.",
    "minimal_core_repair_ready": "downstream gate for deterministic core-guided repair.",
    "inference_substrate": "no hidden LLM or generated diagnosis.",
    "honest_verdict": "terminal status; start with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def load_source_artifacts(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load and validate the two deterministic source artifacts."""

    root_path = Path(root)
    vp_artifact = _read_json(root_path / vp5443.RESULT_RELATIVE_PATH)
    ast_artifact = _read_json(root_path / astkb5445.RESULT_RELATIVE_PATH)
    vp5443.validate_artifact(vp_artifact)
    astkb5445.validate_artifact(ast_artifact)
    return {
        "verifier_potential": vp_artifact,
        "ast_kb_witness": ast_artifact,
    }


def select_repair_cases(source_artifacts: Mapping[str, Any]) -> list[JsonDict]:
    """Select bounded failed rows and attach deterministic minimal cores."""

    vp_rows = {
        str(row["row_id"]): row
        for row in source_artifacts["verifier_potential"]["fixture_rows"]
    }
    ast_rows = {
        str(row["row_id"]): row for row in source_artifacts["ast_kb_witness"]["witness_rows"]
    }
    cases: list[JsonDict] = []
    for row_id in SELECTED_VP_ROW_IDS:
        cases.append(
            _build_case(
                case_id=f"exp5443:{row_id}",
                substrate="verifier_potential",
                source_artifact=str(vp5443.RESULT_RELATIVE_PATH),
                source_row=vp_rows[row_id],
                encoded_constraints=_encode_vp_constraints(vp_rows[row_id]),
                original_candidate=copy.deepcopy(vp_rows[row_id]["final_output"]),
                recheck_context=_vp_recheck_context(vp_rows[row_id]),
                source_row_checksum=str(vp_rows[row_id]["row_checksum"]),
            )
        )
    for row_id in SELECTED_AST_ROW_IDS:
        cases.append(
            _build_case(
                case_id=f"exp5445:{row_id}",
                substrate="ast_kb_witness",
                source_artifact=str(astkb5445.RESULT_RELATIVE_PATH),
                source_row=ast_rows[row_id],
                encoded_constraints=_encode_ast_constraints(ast_rows[row_id]),
                original_candidate={"source": ast_rows[row_id]["source"]},
                recheck_context=_ast_recheck_context(ast_rows[row_id]),
                source_row_checksum=str(ast_rows[row_id]["witness_checksum"]),
            )
        )
    return cases


def derive_minimal_core(case: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the smallest repair core whose generated candidate passes recheck."""

    core = [
        constraint["constraint_id"]
        for constraint in case["encoded_constraints"]
        if constraint["satisfied"] is False and constraint.get("repair_action")
    ]
    core = sorted(core)
    for constraint_id in list(core):
        trial = [item for item in core if item != constraint_id]
        candidate = _apply_core_repairs(case, trial)
        if recheck_candidate(case, candidate)["accepted"] is True:
            core = trial
    return tuple(core)


def generate_repair_hypothesis(
    case: Mapping[str, Any],
    core_constraint_ids: Sequence[str],
) -> JsonDict:
    """Generate one repair candidate from exactly the minimal core IDs."""

    normalized_core = tuple(sorted(str(item) for item in core_constraint_ids))
    minimal_core = tuple(case["minimal_core_ids"])
    if normalized_core != minimal_core:
        raise ValueError(
            f"core IDs must match minimal core for {case['case_id']}: {minimal_core}"
        )
    candidate = _apply_core_repairs(case, normalized_core)
    actions = [
        _repair_action_for_constraint(case, constraint_id)
        for constraint_id in normalized_core
    ]
    hypothesis = {
        "hypothesis_id": _sha256_json(
            {
                "case_id": case["case_id"],
                "core_constraint_ids": normalized_core,
                "candidate": candidate,
            }
        ),
        "generated_from": "minimal_core_ids_only",
        "core_constraint_ids": list(normalized_core),
        "repair_actions": actions,
        "candidate": candidate,
    }
    return hypothesis


def recheck_candidate(case: Mapping[str, Any], candidate: Mapping[str, Any]) -> JsonDict:
    """Run the original exact authority over a candidate."""

    if case["substrate"] == "verifier_potential":
        row = copy.deepcopy(case["recheck_context"])
        row["final_output"] = copy.deepcopy(candidate)
        verdict = vp5443.exact_final_verdict(row)
        return {
            "authority": verdict["authority"],
            "accepted": verdict["accepted"],
            "failure_reasons": list(verdict["failure_reasons"]),
            "verifiers_run": list(verdict["verifiers_run"]),
        }
    source = str(candidate["source"])
    context = case["recheck_context"]
    fixture = astkb5445.AstKbFixture(
        row_id=str(context["row_id"]),
        fixture_family=str(context["fixture_family"]),
        api_family=str(context["api_family"]),
        source=source,
        expected_outcome="accept",
        intent=str(context["intent"]),
        expected_call_fqns=tuple(context["expected_call_fqns"]),
        metric_tags=("valid_call",),
    )
    row = astkb5445.evaluate_fixture(
        fixture,
        kb=astkb5445.ApiKnowledgeBase.from_fallback_metadata(),
    )
    return {
        "authority": "ast_kb_witness",
        "accepted": row["accepted"],
        "failure_reasons": list(row["reject_reasons"]),
        "witness_checksum": row["witness_checksum"],
        "call_sites": row["fully_qualified_call_sites"],
    }


def summarize_repair_attempt(
    case: Mapping[str, Any],
    hypothesis: Mapping[str, Any],
) -> JsonDict:
    """Recheck a generated repair and summarize whether exact authority accepted it."""

    recheck = recheck_candidate(case, hypothesis["candidate"])
    return {
        "case_id": case["case_id"],
        "hypothesis_id": hypothesis.get("hypothesis_id", ""),
        "accepted_after_exact_recheck": recheck["accepted"] is True,
        "exact_recheck": recheck,
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5458 artifact from selected source failures."""

    source_artifacts = load_source_artifacts(root)
    cases = select_repair_cases(source_artifacts)
    metrics = _derive_artifact_metrics(cases)
    ready = bool(
        cases
        and metrics["minimal_core_success_rate"] == 1.0
        and metrics["repaired_accept_rate_after_exact_recheck"] == 1.0
        and metrics["unrepaired_reject_rate"] == 1.0
        and {case["substrate"] for case in cases}
        == {"verifier_potential", "ast_kb_witness"}
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [str(path) for path in SOURCE_ARTIFACTS],
        "repair_case_count": metrics["repair_case_count"],
        "minimal_core_success_rate": metrics["minimal_core_success_rate"],
        "core_constraint_id_count": metrics["core_constraint_id_count"],
        "repaired_accept_rate_after_exact_recheck": metrics[
            "repaired_accept_rate_after_exact_recheck"
        ],
        "unrepaired_reject_rate": metrics["unrepaired_reject_rate"],
        "exact_final_authority": True,
        "row_provenance_checksum": row_provenance_checksum(cases),
        "minimal_core_repair_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            "complete: deterministic minimal-core repairs accepted after exact recheck"
            if ready
            else "blocked: minimal-core repair readiness checks failed"
        ),
        "repair_cases": cases,
        "field_principles": dict(FIELD_PRINCIPLES),
        "tests_run": _normalise_tests_run(tests_run),
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the Exp5458 artifact."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    if write:
        output_path = Path(result_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def row_provenance_checksum(cases: Sequence[Mapping[str, Any]]) -> str:
    """Hash the source rows, stable core IDs, hypotheses, and exact outcomes."""

    payload = [
        {
            "case_id": case["case_id"],
            "source_artifact": case["source_artifact"],
            "source_row_id": case["source_row_id"],
            "source_row_checksum": case["source_row_checksum"],
            "minimal_core_ids": list(case["minimal_core_ids"]),
            "hypothesis_id": case["repair_hypothesis"]["hypothesis_id"],
            "unrepaired_accepted": case["unrepaired_exact_recheck"]["accepted"],
            "repaired_accepted": case["repaired_exact_recheck"]["accepted"],
        }
        for case in cases
    ]
    return _sha256_json(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise if the artifact can no longer support the Exp5458 claim."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, provenance, and exact-authority validation errors."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    cases = artifact.get("repair_cases")
    if not isinstance(cases, list):
        errors.append("repair_cases must be a list")
        cases = []
    metrics = _derive_artifact_metrics(cases)
    for field in (
        "repair_case_count",
        "minimal_core_success_rate",
        "core_constraint_id_count",
        "repaired_accept_rate_after_exact_recheck",
        "unrepaired_reject_rate",
    ):
        if artifact.get(field) != metrics[field]:
            errors.append(f"{field} must match repair case recomputation")
    if artifact.get("source_artifacts") != [str(path) for path in SOURCE_ARTIFACTS]:
        errors.append("source_artifacts mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("exact_final_authority") is not True:
        errors.append("exact_final_authority must be true")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with complete: or blocked:")
    if artifact.get("row_provenance_checksum") != row_provenance_checksum(cases):
        errors.append("row_provenance_checksum mismatch")
    if artifact.get("reproducibility_checksum") != _artifact_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py must not be modified")
    errors.extend(_case_integrity_errors(cases))
    if artifact.get("minimal_core_repair_ready") is True:
        if not cases:
            errors.append("minimal_core_repair_ready requires repair cases")
        for field in (
            "minimal_core_success_rate",
            "repaired_accept_rate_after_exact_recheck",
            "unrepaired_reject_rate",
        ):
            if artifact.get(field) != 1.0:
                errors.append(f"minimal_core_repair_ready requires {field}=1.0")
        if artifact.get("exact_final_authority") is not True:
            errors.append("minimal_core_repair_ready requires exact authority")
    elif not isinstance(artifact.get("minimal_core_repair_ready"), bool):
        errors.append("minimal_core_repair_ready must be boolean")
    return errors


def _build_case(
    *,
    case_id: str,
    substrate: str,
    source_artifact: str,
    source_row: Mapping[str, Any],
    encoded_constraints: Sequence[Mapping[str, Any]],
    original_candidate: Mapping[str, Any],
    recheck_context: Mapping[str, Any],
    source_row_checksum: str,
) -> JsonDict:
    if recheck_candidate(
        {
            "case_id": case_id,
            "substrate": substrate,
            "recheck_context": recheck_context,
        },
        original_candidate,
    )["accepted"]:
        raise ValueError(f"selected source row is not failed: {case_id}")
    base_case: JsonDict = {
        "case_id": case_id,
        "substrate": substrate,
        "source_artifact": source_artifact,
        "source_row_id": str(source_row["row_id"]),
        "source_row_checksum": source_row_checksum,
        "original_candidate": copy.deepcopy(original_candidate),
        "recheck_context": copy.deepcopy(dict(recheck_context)),
        "encoded_constraints": [copy.deepcopy(dict(item)) for item in encoded_constraints],
    }
    base_case["satisfied_constraint_ids"] = [
        item["constraint_id"] for item in base_case["encoded_constraints"] if item["satisfied"]
    ]
    base_case["violated_constraint_ids"] = [
        item["constraint_id"]
        for item in base_case["encoded_constraints"]
        if item["satisfied"] is False
    ]
    base_case["minimal_core_ids"] = list(derive_minimal_core(base_case))
    base_case["minimality_evidence"] = _minimality_evidence(base_case)
    base_case["unrepaired_exact_recheck"] = recheck_candidate(base_case, original_candidate)
    base_case["repair_hypothesis"] = generate_repair_hypothesis(
        base_case,
        base_case["minimal_core_ids"],
    )
    repaired = summarize_repair_attempt(base_case, base_case["repair_hypothesis"])
    base_case["repaired_exact_recheck"] = repaired["exact_recheck"]
    base_case["accepted_after_exact_recheck"] = repaired["accepted_after_exact_recheck"]
    return base_case


def _encode_vp_constraints(row: Mapping[str, Any]) -> list[JsonDict]:
    final = row["final_output"]
    required = set(row.get("required_keys", []))
    allowed = set(row.get("allowed_keys", []))
    constraints = [
        _constraint(
            f"vp:{row['row_id']}:schema:required_keys_present",
            required.issubset(final),
            {"op": "fill_missing_required_keys"},
        ),
        _constraint(
            f"vp:{row['row_id']}:schema:allowed_keys_only",
            set(final).issubset(allowed),
            {"op": "remove_disallowed_keys"},
        ),
    ]
    family = row["constraint_family"]
    if family == "semantic_contradiction":
        constraints.append(
            _constraint(
                f"vp:{row['row_id']}:semantic:no_equal_object_negated_object",
                final.get("object") != final.get("negated_object"),
                {"op": "separate_negated_object"},
            )
        )
    if family == "arithmetic_finite_domain":
        constraints.append(
            _constraint(
                f"vp:{row['row_id']}:arithmetic:sum_matches_operands",
                final.get("sum") == final.get("x") + final.get("y"),
                {"op": "set_sum_from_operands"},
            )
        )
    if family == "api_call_witness":
        api = row["api"]
        witness = final["witness"]
        constraints.extend(
            [
                _constraint(
                    f"vp:{row['row_id']}:api:call_allowed",
                    [final.get("method"), final.get("path")] in api["allowed_calls"],
                    None,
                ),
                _constraint(
                    f"vp:{row['row_id']}:api:signature_matches",
                    witness.get("signature") == api["expected_signature"],
                    {"op": "set_api_signature"},
                ),
                _constraint(
                    f"vp:{row['row_id']}:api:nonce_fresh",
                    witness.get("nonce_reused") is not True,
                    {"op": "set_api_nonce_fresh"},
                ),
            ]
        )
    return constraints


def _encode_ast_constraints(row: Mapping[str, Any]) -> list[JsonDict]:
    constraints = [
        _constraint(
            f"astkb:{row['row_id']}:ast:parse_ok",
            row["ast_parse_ok"] is True,
            None,
        )
    ]
    missing_import_fqns = {
        check["fully_qualified_name"]
        for check in row["imported_symbol_checks"]
        if check["exists"] is False
    }
    for check in row["imported_symbol_checks"]:
        fqn = check["fully_qualified_name"]
        constraints.append(
            _constraint(
                f"astkb:{row['row_id']}:imported_symbol_exists:{fqn}",
                check["exists"] is True,
                {"op": "rewrite_to_expected_call"} if check["exists"] is False else None,
            )
        )
    for call, lookup in zip(
        row["fully_qualified_call_sites"],
        row["kb_lookup_results"],
        strict=True,
    ):
        fqn = call["fqn"]
        constraints.append(
            _constraint(
                f"astkb:{row['row_id']}:call_exists:{fqn}",
                lookup["exists"] is True,
                (
                    {"op": "rewrite_to_expected_call"}
                    if lookup["exists"] is False and fqn not in missing_import_fqns
                    else None
                ),
            )
        )
    intent = row["semantic_intent"]
    all_calls_exist = all(result["exists"] is True for result in row["kb_lookup_results"])
    constraints.append(
        _constraint(
            f"astkb:{row['row_id']}:intent_matches:{intent['intent']}",
            intent["matched"] is True,
            {"op": "rewrite_to_expected_call"}
            if intent["matched"] is False and all_calls_exist
            else None,
        )
    )
    return constraints


def _constraint(
    constraint_id: str,
    satisfied: bool,
    repair_action: Mapping[str, Any] | None,
) -> JsonDict:
    row = {
        "constraint_id": constraint_id,
        "satisfied": bool(satisfied),
    }
    if repair_action is not None:
        row["repair_action"] = dict(repair_action)
    return row


def _apply_core_repairs(
    case: Mapping[str, Any],
    core_constraint_ids: Sequence[str],
) -> JsonDict:
    candidate = copy.deepcopy(case["original_candidate"])
    for constraint_id in sorted(core_constraint_ids):
        action = _repair_action_for_constraint(case, constraint_id)
        op = action["op"]
        if op == "remove_disallowed_keys":
            allowed = set(case["recheck_context"].get("allowed_keys", []))
            candidate = {key: value for key, value in candidate.items() if key in allowed}
        elif op == "separate_negated_object":
            candidate["negated_object"] = f"not_{candidate['object']}"
        elif op == "set_sum_from_operands":
            candidate["sum"] = candidate["x"] + candidate["y"]
        elif op == "set_api_signature":
            candidate["witness"]["signature"] = case["recheck_context"]["api"][
                "expected_signature"
            ]
        elif op == "set_api_nonce_fresh":
            candidate["witness"]["nonce_reused"] = False
        elif op == "rewrite_to_expected_call":
            candidate = {"source": _source_for_expected_call(case)}
        elif op == "fill_missing_required_keys":
            for key in case["recheck_context"].get("required_keys", []):
                candidate.setdefault(key, {})
    return candidate


def _repair_action_for_constraint(
    case: Mapping[str, Any],
    constraint_id: str,
) -> JsonDict:
    for constraint in case["encoded_constraints"]:
        if constraint["constraint_id"] == constraint_id and constraint.get("repair_action"):
            action = dict(constraint["repair_action"])
            action["constraint_id"] = constraint_id
            return action
    raise ValueError(f"no deterministic repair action for constraint {constraint_id}")


def _source_for_expected_call(case: Mapping[str, Any]) -> str:
    context = case["recheck_context"]
    expected = str(context["expected_call_fqns"][0])
    module_name, _, symbol = expected.partition(".")
    original_source = str(case["original_candidate"]["source"])
    call_segment = str(context["first_call_source_segment"])
    arguments = call_segment[call_segment.index("(") + 1 : call_segment.rindex(")")]
    result_name = "result" if "result =" in original_source else "value"
    return f"import {module_name}\n{result_name} = {module_name}.{symbol}({arguments})\n"


def _minimality_evidence(case: Mapping[str, Any]) -> list[JsonDict]:
    evidence = []
    for constraint_id in case["minimal_core_ids"]:
        remaining = [item for item in case["minimal_core_ids"] if item != constraint_id]
        recheck = recheck_candidate(case, _apply_core_repairs(case, remaining))
        evidence.append(
            {
                "removed_constraint_id": constraint_id,
                "remaining_core_ids": remaining,
                "accepted_without_constraint": recheck["accepted"],
                "authority": recheck["authority"],
            }
        )
    return evidence


def _vp_recheck_context(row: Mapping[str, Any]) -> JsonDict:
    keys = (
        "row_id",
        "fixture_id",
        "constraint_family",
        "required_keys",
        "allowed_keys",
        "action_constraints",
        "domain_fields",
        "finite_domain",
        "ontology",
        "api",
        "prefixes",
    )
    return {key: copy.deepcopy(row[key]) for key in keys if key in row}


def _ast_recheck_context(row: Mapping[str, Any]) -> JsonDict:
    first_call = row["fully_qualified_call_sites"][0]
    return {
        "row_id": row["row_id"],
        "fixture_family": row["fixture_family"],
        "api_family": row["api_family"],
        "intent": row["semantic_intent"]["intent"],
        "expected_call_fqns": list(row["semantic_intent"]["expected_call_fqns"]),
        "first_call_source_segment": first_call["source_segment"],
    }


def _derive_artifact_metrics(cases: Sequence[Mapping[str, Any]]) -> JsonDict:
    repair_case_count = len(cases)
    core_ids = {
        core_id for case in cases for core_id in case.get("minimal_core_ids", [])
    }
    return {
        "repair_case_count": repair_case_count,
        "minimal_core_success_rate": _rate(
            sum(1 for case in cases if case.get("minimal_core_ids")),
            repair_case_count,
        ),
        "core_constraint_id_count": len(core_ids),
        "repaired_accept_rate_after_exact_recheck": _rate(
            sum(
                1
                for case in cases
                if case.get("accepted_after_exact_recheck") is True
                and case.get("repaired_exact_recheck", {}).get("accepted") is True
            ),
            repair_case_count,
        ),
        "unrepaired_reject_rate": _rate(
            sum(
                1
                for case in cases
                if case.get("unrepaired_exact_recheck", {}).get("accepted") is False
            ),
            repair_case_count,
        ),
    }


def _case_integrity_errors(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    for case in cases:
        if tuple(case.get("minimal_core_ids", [])) != derive_minimal_core(case):
            errors.append(f"{case.get('case_id')} minimal_core_ids mismatch")
        if case.get("unrepaired_exact_recheck") != recheck_candidate(
            case,
            case["original_candidate"],
        ):
            errors.append(f"{case.get('case_id')} unrepaired exact recheck mismatch")
        hypothesis = generate_repair_hypothesis(case, case.get("minimal_core_ids", []))
        if case.get("repair_hypothesis") != hypothesis:
            errors.append(f"{case.get('case_id')} repair_hypothesis mismatch")
        attempt = summarize_repair_attempt(case, hypothesis)
        if case.get("repaired_exact_recheck") != attempt["exact_recheck"]:
            errors.append(f"{case.get('case_id')} repaired exact recheck mismatch")
        if case.get("accepted_after_exact_recheck") != attempt["accepted_after_exact_recheck"]:
            errors.append(f"{case.get('case_id')} exact recheck acceptance mismatch")
    return errors


def _normalise_tests_run(value: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for row in value:
        if isinstance(row, Mapping):
            rows.append(
                {
                    "command": str(row.get("command", "")),
                    "outcome": str(row.get("outcome", "recorded")),
                }
            )
        else:
            rows.append({"command": str(row), "outcome": "recorded"})
    return rows


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return _sha256_json(payload)


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(root=args.root, result_path=args.result_path, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact["minimal_core_repair_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
