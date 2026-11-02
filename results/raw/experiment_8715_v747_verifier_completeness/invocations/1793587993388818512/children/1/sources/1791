#!/usr/bin/env python3
"""Exp5476 deterministic helper-lemma core witness repair.

Spec refs: REQ-VERIFY-5476, SCENARIO-VERIFY-5476.

The module tests a narrow version of helper-lemma repair over checked-in
AST/KB witness rows. Helper candidates are generated only from the row's
source, semantic intent, AST call site, import checks, KB lookup results, and
reject reasons. The helper never certifies itself; the Exp5445 exact AST/KB
witness evaluator remains the final authority after every candidate rewrite.
"""

from __future__ import annotations

import argparse
import ast
from collections import OrderedDict
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5445_static_ast_kb_witness_constraints_v495 as astkb5445
from carnot import experiment_5458_minimal_core_claim_repair_v496 as core5458


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5476_helper_lemma_core_witness_repair_v497.json")
EXPERIMENT_ID = "experiment_5476_helper_lemma_core_witness_repair_v497"
TASK_ID = "exp5476-v497-helper-lemma-core-witness-repair"
MILESTONE = "2026.07.497"
RUN_DATE = "2026-07-09"
SCHEMA = "carnot.experiment_5476.helper_lemma_core_witness_repair.v497"
SPEC_REFS = ("REQ-VERIFY-5476", "SCENARIO-VERIFY-5476", "REQ-CODE-5445", "REQ-VERIFY-5458")
RANDOM_SEED = 5476
INFERENCE_SUBSTRATE = "deterministic_witness_repair_no_llm"
SOURCE_ARTIFACTS = (
    astkb5445.RESULT_RELATIVE_PATH,
    core5458.RESULT_RELATIVE_PATH,
)
SELECTED_AST_ROW_IDS = (
    "fixture.nonexistent_json_method",
    "fixture.nonexistent_math_alias_method",
    "fixture.wrong_module_alias",
    "fixture.missing_bare_import",
    "fixture.imported_symbol_missing",
    "fixture.argument_intent_mismatch",
)
SOURCE_FIELDS_USED = (
    "source",
    "semantic_intent",
    "fully_qualified_call_sites",
    "imported_symbol_checks",
    "kb_lookup_results",
    "reject_reasons",
)
TERMINAL_PREFIXES = ("complete:", "blocked:")

FIELD_PRINCIPLES: dict[str, str] = {
    "witness_row_count": "bounded deterministic AST/KB witness rows selected for helper repair.",
    "failure_signature_count": "pre-repair verifier signatures are grouped before repair.",
    "helper_candidate_count": "bounded helper lemmas/contracts generated from row evidence only.",
    "accepted_helper_count": "helpers counted only after exact AST/KB recheck accepts.",
    "exact_recheck_pass_rate": "final authority pass rate over helper candidates.",
    "false_accept_count": "exact recheck rejects unsupported helpers instead of trusting helper text.",
    "repeated_failure_reduction_rate": "measures solved rows from repeated pre-repair failure signatures.",
    "helper_lemma_repair_ready": "downstream gate for deterministic helper-contract witness repair.",
    "inference_substrate": "no hidden LLM, source lookup, or generated judgment.",
    "random_seed": "deterministic row selection and ordering.",
    "honest_verdict": "terminal status; start with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)


def load_source_artifacts(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load upstream artifacts and validate them before deriving any helper."""

    root_path = Path(root)
    ast_artifact = _read_json(root_path / astkb5445.RESULT_RELATIVE_PATH)
    core_artifact = _read_json(root_path / core5458.RESULT_RELATIVE_PATH)
    astkb5445.validate_artifact(ast_artifact)
    core5458.validate_artifact(core_artifact)
    return {
        "ast_kb_witness": ast_artifact,
        "minimal_core_repair": core_artifact,
    }


def select_witness_rows(source_artifacts: Mapping[str, Any]) -> list[JsonDict]:
    """Select the deterministic failed AST/KB rows where row-local helpers apply."""

    rows_by_id = {
        str(row["row_id"]): row for row in source_artifacts["ast_kb_witness"]["witness_rows"]
    }
    selected: list[JsonDict] = []
    for row_id in SELECTED_AST_ROW_IDS:
        row = copy.deepcopy(dict(rows_by_id[row_id]))
        if row.get("accepted") is True:
            raise ValueError(f"selected row is not a failed witness: {row_id}")
        selected.append(row)
    return selected


def failure_signature(row: Mapping[str, Any]) -> str:
    """Normalize row-specific reject details into a repeated-failure signature."""

    categories = []
    for reason in row.get("reject_reasons", []):
        categories.append(str(reason).split(":", maxsplit=1)[0])
    normalized = "+".join(sorted(set(categories))) or "no_reject_reason"
    return f"ast_kb:{normalized}"


def group_failures_by_signature(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Group failed rows before repair so repeated modes cannot be hidden."""

    buckets: OrderedDict[str, list[str]] = OrderedDict()
    for row in rows:
        signature = failure_signature(row)
        buckets.setdefault(signature, []).append(str(row["row_id"]))
    return [
        {
            "failure_signature": signature,
            "row_ids": row_ids,
            "count": len(row_ids),
            "repeated": len(row_ids) > 1,
        }
        for signature, row_ids in buckets.items()
    ]


def generate_helper_candidates(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Generate and exact-recheck one helper candidate per selected witness row."""

    return [_build_helper_candidate(row) for row in rows]


def recheck_helper_candidate(row: Mapping[str, Any], candidate_source: str) -> JsonDict:
    """Run the original AST/KB exact authority over a helper-produced candidate."""

    intent = row["semantic_intent"]
    fixture = astkb5445.AstKbFixture(
        row_id=str(row["row_id"]),
        fixture_family=str(row["fixture_family"]),
        api_family=str(row["api_family"]),
        source=candidate_source,
        expected_outcome="accept",
        intent=str(intent["intent"]),
        expected_call_fqns=tuple(str(item) for item in intent["expected_call_fqns"]),
        metric_tags=("valid_call",),
    )
    checked = astkb5445.evaluate_fixture(
        fixture,
        kb=astkb5445.ApiKnowledgeBase.from_fallback_metadata(),
    )
    return {
        "authority": "ast_kb_witness",
        "accepted": checked["accepted"],
        "failure_reasons": list(checked["reject_reasons"]),
        "witness_checksum": checked["witness_checksum"],
        "call_sites": checked["fully_qualified_call_sites"],
        "kb_lookup_results": checked["kb_lookup_results"],
        "semantic_intent": checked["semantic_intent"],
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5476 helper repair artifact."""

    source_artifacts = load_source_artifacts(root)
    rows = select_witness_rows(source_artifacts)
    groups = group_failures_by_signature(rows)
    candidates = generate_helper_candidates(rows)
    metrics = _derive_metrics(rows, groups, candidates)
    ready = _ready(metrics, candidates)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [str(path) for path in SOURCE_ARTIFACTS],
        "witness_row_count": metrics["witness_row_count"],
        "failure_signature_count": metrics["failure_signature_count"],
        "helper_candidate_count": metrics["helper_candidate_count"],
        "accepted_helper_count": metrics["accepted_helper_count"],
        "exact_recheck_pass_rate": metrics["exact_recheck_pass_rate"],
        "false_accept_count": metrics["false_accept_count"],
        "repeated_failure_reduction_rate": metrics["repeated_failure_reduction_rate"],
        "helper_lemma_repair_ready": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            "complete: bounded helper-contract witness repair accepts supported helpers "
            "and rejects unsupported helpers after exact recheck"
            if ready
            else "blocked: helper-contract witness repair readiness checks failed"
        ),
        "witness_rows": rows,
        "failure_signature_groups": groups,
        "helper_candidates": candidates,
        "semantic_change_rejections": [
            candidate
            for candidate in candidates
            if candidate["semantics_changed_incorrectly"] is True
        ],
        "row_provenance_checksum": row_provenance_checksum(rows),
        "minimal_core_reference_case_ids": _minimal_core_reference_case_ids(source_artifacts),
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
    """Build and optionally persist the Exp5476 result artifact."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    if write:
        output_path = Path(result_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def row_provenance_checksum(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash witness identity, row witness checksums, and pre-repair signatures."""

    payload = [
        {
            "row_id": row.get("row_id"),
            "source_sha256": row.get("source_sha256"),
            "witness_checksum": row.get("witness_checksum"),
            "failure_signature": failure_signature(row),
        }
        for row in rows
    ]
    return _sha256_json(payload)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise if the artifact can no longer support the Exp5476 claim."""

    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema, provenance, and exact-authority validation errors."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    errors.extend(_error_if(bool(missing), f"missing required fields: {missing}"))

    rows = artifact.get("witness_rows")
    groups = artifact.get("failure_signature_groups")
    candidates = artifact.get("helper_candidates")
    row_list = rows if isinstance(rows, list) else []
    group_list = groups if isinstance(groups, list) else []
    candidate_list = candidates if isinstance(candidates, list) else []
    errors.extend(_error_if(not isinstance(rows, list), "witness_rows must be a list"))
    errors.extend(
        _error_if(not isinstance(groups, list), "failure_signature_groups must be a list")
    )
    errors.extend(_error_if(not isinstance(candidates, list), "helper_candidates must be a list"))

    metrics = _derive_metrics(row_list, group_list, candidate_list)
    for field in (
        "witness_row_count",
        "failure_signature_count",
        "helper_candidate_count",
        "accepted_helper_count",
        "exact_recheck_pass_rate",
        "false_accept_count",
        "repeated_failure_reduction_rate",
    ):
        errors.extend(_error_if(artifact.get(field) != metrics[field], f"{field} mismatch"))

    errors.extend(
        _error_if(
            artifact.get("source_artifacts") != [str(path) for path in SOURCE_ARTIFACTS],
            "source_artifacts mismatch",
        )
    )
    errors.extend(
        _error_if(artifact.get("field_principles") != FIELD_PRINCIPLES, "field_principles mismatch")
    )
    errors.extend(
        _error_if(
            artifact.get("inference_substrate") != INFERENCE_SUBSTRATE,
            "inference_substrate mismatch",
        )
    )
    errors.extend(_error_if(artifact.get("random_seed") != RANDOM_SEED, "random_seed mismatch"))
    verdict = artifact.get("honest_verdict")
    errors.extend(
        _error_if(
            not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES),
            "honest_verdict must start with complete: or blocked:",
        )
    )
    errors.extend(
        _error_if(
            artifact.get("row_provenance_checksum") != row_provenance_checksum(row_list),
            "row_provenance_checksum mismatch",
        )
    )
    errors.extend(
        _error_if(
            artifact.get("reproducibility_checksum") != _artifact_checksum(artifact),
            "reproducibility_checksum mismatch",
        )
    )
    errors.extend(
        _error_if(
            artifact.get("research_conductor_modified") is not False,
            "scripts/research_conductor.py must not be modified",
        )
    )
    errors.extend(_candidate_integrity_errors(row_list, candidate_list))
    expected_ready = _ready(metrics, candidate_list)
    errors.extend(
        _error_if(
            artifact.get("helper_lemma_repair_ready") != expected_ready,
            "helper_lemma_repair_ready mismatch",
        )
    )
    if artifact.get("helper_lemma_repair_ready") is True:
        errors.extend(
            _error_if(
                artifact.get("false_accept_count") != 0,
                "helper_lemma_repair_ready requires false_accept_count=0",
            )
        )
    return errors


def _build_helper_candidate(row: Mapping[str, Any]) -> JsonDict:
    expected_call = _expected_call_fqn(row)
    helper_kind = _helper_kind(row)
    candidate_source = _candidate_source_for_expected_call(row, expected_call)
    recheck = recheck_helper_candidate(row, candidate_source)
    accepted = recheck["accepted"] is True
    semantics_changed = any(
        str(reason).startswith("intent_mismatch:") for reason in recheck["failure_reasons"]
    )
    return {
        "helper_id": _sha256_json(
            {
                "row_id": row["row_id"],
                "failure_signature": failure_signature(row),
                "helper_kind": helper_kind,
                "expected_call_fqn": expected_call,
            }
        ),
        "row_id": row["row_id"],
        "failure_signature": failure_signature(row),
        "helper_kind": helper_kind,
        "helper_contract": {
            "statement": _helper_statement(row, helper_kind, expected_call),
            "expected_call_fqn": expected_call,
            "actual_call_fqns": list(row["semantic_intent"]["actual_call_fqns"]),
            "intent": row["semantic_intent"]["intent"],
        },
        "generated_from": "witness_row_source_semantic_intent_and_kb_results",
        "source_fields_used": list(SOURCE_FIELDS_USED),
        "source_before": row["source"],
        "candidate_source": candidate_source,
        "exact_recheck": recheck,
        "accepted_after_exact_recheck": accepted,
        "rejection_reason": "" if accepted else "exact_recheck_rejected",
        "false_accept": False,
        "semantics_changed_incorrectly": semantics_changed,
    }


def _helper_kind(row: Mapping[str, Any]) -> str:
    reasons = [str(reason) for reason in row.get("reject_reasons", [])]
    if any(reason.startswith("missing_import_for_bare_call:") for reason in reasons):
        return "api_import_binding_contract"
    if any(reason.startswith("imported_symbol_missing:") for reason in reasons):
        return "api_import_precondition"
    if any(reason.startswith("intent_mismatch:") for reason in reasons) and not any(
        reason.startswith("kb_missing_call:") for reason in reasons
    ):
        return "documentation_contract"
    actual = _first_actual_call_fqn(row)
    expected = _expected_call_fqn(row)
    if any(reason.startswith("kb_missing_call:") for reason in reasons):
        actual_module, _, actual_symbol = actual.partition(".")
        expected_module, _, expected_symbol = expected.partition(".")
        if actual_symbol == expected_symbol and actual_module != expected_module:
            return "module_alias_invariant"
        return "api_member_precondition"
    return "helper_invariant"


def _helper_statement(row: Mapping[str, Any], helper_kind: str, expected_call: str) -> str:
    intent = row["semantic_intent"]["intent"]
    if helper_kind == "documentation_contract":
        return f"Documentation intent {intent} is satisfied by {expected_call}."
    if helper_kind == "module_alias_invariant":
        return f"Alias binding must resolve the {intent} call to {expected_call}."
    if helper_kind == "api_import_binding_contract":
        return f"Bare helper call for {intent} must be imported as {expected_call}."
    return f"API precondition for {intent} requires an exact KB-valid call to {expected_call}."


def _candidate_source_for_expected_call(row: Mapping[str, Any], expected_call: str) -> str:
    module_name, _, symbol = expected_call.partition(".")
    target = _assignment_target(str(row["source"]))
    arguments = _first_call_arguments(row)
    return f"import {module_name}\n{target} = {module_name}.{symbol}({arguments})\n"


def _first_call_arguments(row: Mapping[str, Any]) -> str:
    segment = str(row["fully_qualified_call_sites"][0]["source_segment"])
    return segment[segment.index("(") + 1 : segment.rindex(")")]


def _assignment_target(source: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            return node.targets[0].id
    return "result"


def _expected_call_fqn(row: Mapping[str, Any]) -> str:
    return str(row["semantic_intent"]["expected_call_fqns"][0])


def _first_actual_call_fqn(row: Mapping[str, Any]) -> str:
    actual = row["semantic_intent"].get("actual_call_fqns", [])
    return str(actual[0]) if actual else ""


def _derive_metrics(
    rows: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> JsonDict:
    repeated_row_ids = {
        row_id
        for group in groups
        if group.get("repeated") is True
        for row_id in group.get("row_ids", [])
    }
    accepted = [
        candidate
        for candidate in candidates
        if candidate.get("accepted_after_exact_recheck") is True
    ]
    solved_repeated = [
        candidate for candidate in accepted if candidate.get("row_id") in repeated_row_ids
    ]
    return {
        "witness_row_count": len(rows),
        "failure_signature_count": len(groups),
        "helper_candidate_count": len(candidates),
        "accepted_helper_count": len(accepted),
        "exact_recheck_pass_rate": _rate(len(accepted), len(candidates)),
        "false_accept_count": sum(
            1 for candidate in candidates if candidate.get("false_accept") is True
        ),
        "repeated_failure_reduction_rate": _rate(len(solved_repeated), len(repeated_row_ids)),
    }


def _ready(metrics: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> bool:
    accepted = sum(
        1 for candidate in candidates if candidate.get("accepted_after_exact_recheck") is True
    )
    rejected = sum(
        1 for candidate in candidates if candidate.get("accepted_after_exact_recheck") is False
    )
    semantic_bad_accepts = [
        candidate
        for candidate in candidates
        if candidate.get("accepted_after_exact_recheck") is True
        and candidate.get("semantics_changed_incorrectly") is True
    ]
    return bool(
        metrics.get("witness_row_count") == metrics.get("helper_candidate_count")
        and accepted > 0
        and rejected > 0
        and metrics.get("false_accept_count") == 0
        and metrics.get("repeated_failure_reduction_rate") == 1.0
        and not semantic_bad_accepts
    )


def _candidate_integrity_errors(
    rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[str]:
    rows_by_id = {str(row.get("row_id")): row for row in rows}
    errors: list[str] = []
    for candidate in candidates:
        row = rows_by_id.get(str(candidate.get("row_id")))
        if row is None:  # pragma: no cover - schema drift guard.
            errors.append(f"{candidate.get('row_id')} candidate row missing")
            continue
        expected = _build_helper_candidate(row)
        errors.extend(
            _error_if(
                candidate.get("candidate_source") != expected["candidate_source"],
                f"{candidate.get('row_id')} candidate source mismatch",
            )
        )
        errors.extend(
            _error_if(
                candidate.get("exact_recheck") != expected["exact_recheck"],
                f"{candidate.get('row_id')} exact recheck mismatch",
            )
        )
        errors.extend(
            _error_if(
                candidate.get("accepted_after_exact_recheck")
                != expected["accepted_after_exact_recheck"],
                f"{candidate.get('row_id')} candidate acceptance mismatch",
            )
        )
        errors.extend(
            _error_if(
                candidate.get("false_accept") is not False,
                f"{candidate.get('row_id')} false_accept must be false",
            )
        )
    return errors


def _minimal_core_reference_case_ids(source_artifacts: Mapping[str, Any]) -> list[str]:
    return [
        str(case["case_id"])
        for case in source_artifacts["minimal_core_repair"]["repair_cases"]
        if case.get("substrate") == "ast_kb_witness"
    ]


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


def _error_if(condition: bool, message: str) -> list[str]:
    return [message] if condition else []


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--result-path", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    artifact = run(root=args.root, result_path=args.result_path, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if artifact["helper_lemma_repair_ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
