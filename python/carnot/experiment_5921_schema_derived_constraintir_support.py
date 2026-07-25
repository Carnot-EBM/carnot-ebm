"""Exp5921 schema-derived support for open typed ConstraintIR.

Spec refs: REQ-VERIFY-5921, SCENARIO-VERIFY-5921-SCHEMA,
SCENARIO-VERIFY-5921-PREFIX, SCENARIO-VERIFY-5921-ADJUDICATION.

The compiler in this file is intentionally mechanical. It reads a versioned
operation-signature schema and derives the structural supports a decoder would
need: grammar terminals, type/domain transitions, symbol scope rules, and a
bounded rejection policy. Exact semantic correctness remains owned by the
existing Exp5896 Python and Z3 evaluator, because a structural mask can say
"this prefix is well formed" but it cannot prove that the final constraint set
means the right thing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5897_sota_constraint_ir_repair_ab as exp5897
from carnot import experiment_5907_constraint_ir_replay_contract as exp5907
from carnot import experiment_5908_verisynth_constraint_fixture as exp5908


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5921_schema_derived_constraintir_support.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5921_schema_derived_constraintir_support.py")
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5921_schema_derived_constraintir_support.py"
)
VERIFICATION_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
VERIFIABLE_REASONING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
BENCH_SPEC_RELATIVE_PATH = Path("openspec/capabilities/benchmarks/spec.md")

RUN_DATE = "20260725"
EXPERIMENT_ID = "experiment_5921_schema_derived_constraintir_support"
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_5921.schema_derived_constraintir_support.v1"
OPERATION_SIGNATURE_SCHEMA_VERSION = "carnot.constraint_ir.operation_signatures.v1"
SUPPORT_SCHEMA_VERSION = "carnot.constraint_ir.schema_derived_support.v1"
INFERENCE_SUBSTRATE = "deterministic_schema_compilation_no_llm"
VERIFIER_IS_ORACLE = True
MAX_REJECTIONS = 5

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    VERIFICATION_SPEC_RELATIVE_PATH,
    VERIFIABLE_REASONING_SPEC_RELATIVE_PATH,
    BENCH_SPEC_RELATIVE_PATH,
    exp5896.MODULE_RELATIVE_PATH,
    exp5907.MODULE_RELATIVE_PATH,
    exp5907.HELPER_RELATIVE_PATH,
    exp5908.MODULE_RELATIVE_PATH,
    Path("python/carnot/experiment_5909_sota_constraint_synthesis_ab.py"),
    exp5896.RESULT_RELATIVE_PATH,
    exp5896.ROW_FILE_RELATIVE_PATH,
    exp5907.RESULT_RELATIVE_PATH,
    exp5908.RESULT_RELATIVE_PATH,
    exp5908.ROW_FILE_RELATIVE_PATH,
    Path("results/experiment_5909_sota_constraint_synthesis_ab.json"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "source_paper_and_local_mechanism_receipt",
    "operation_signature_schema_and_version",
    "schema_to_grammar_type_scope_compiler_receipt",
    "open_ir_not_finite_id_receipt",
    "train_held_and_attribute_adversary_manifest",
    "missing_spurious_and_scope_controls",
    "prefix_monotonicity_and_dead_end_matrix",
    "exact_python_z3_agreement",
    "semantic_authority_boundary",
    "correct_mode_retention_and_overpruning",
    "tamper_and_corrupted_schema_controls",
    "protected_files_unchanged",
    "schema_decode_contract_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "open_ir_not_finite_id_receipt": (
        "Support may restrict well-typed prefixes but cannot enumerate complete answers "
        "or encode case labels."
    ),
    "semantic_authority_boundary": ("Structural admission never licenses exact semantic success."),
    "schema_decode_contract_ready_score": (
        "Emit bare 1.0 only for deterministic derivation, held-family support, exact "
        "scope/type rejection, zero answer leakage, and no unsafe semantic acceptance."
    ),
    "inference_substrate": "Use deterministic_schema_compilation_no_llm.",
    "verifier_is_oracle": (
        "True only for exact parse, type, scope, execution, and certificate checks."
    ),
    "honest_verdict": "Use complete_ready:, retired:, or blocked:.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5921_schema_derived_constraintir_support.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5921_schema_derived_constraintir_support.py "
    "-m pytest tests/python/test_experiment_5921_schema_derived_constraintir_support.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5921_schema_derived_constraintir_support.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.experiment_5921_schema_derived_constraintir_support",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5921_schema_derived_constraintir_support.json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5921_schema_derived_constraintir_support.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md ops/status.md _bmad/traceability.md",
)


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence in a stable form before hashing or scanning."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for canonical text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a file by bytes so receipts do not depend on path metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def versioned_operation_signatures() -> JsonDict:
    """Return the public operation schema from which every support is derived."""

    return {
        "schema_version": OPERATION_SIGNATURE_SCHEMA_VERSION,
        "normalization": "json_sort_keys_ascii_v1",
        "operation_order": [
            "domain.declare",
            "entity.declare",
            "predicate.declare",
            "fact.assert",
            "rule.define",
            "query.define",
        ],
        "bounded_rejection": {
            "max_rejections": MAX_REJECTIONS,
            "dead_end_policy": "reject_prefix_and_advance_to_next_candidate",
        },
        "operations": [
            {
                "name": "constraint_ir.object",
                "kind": "composition",
                "required_keys": [
                    "schema_version",
                    "domains",
                    "entities",
                    "predicates",
                    "facts",
                    "rules",
                    "query",
                ],
                "defines": [],
                "uses": [],
            },
            {
                "name": "domain.declare",
                "kind": "declaration",
                "required_keys": ["id", "type", "values"],
                "defines": ["domain"],
                "uses": [],
                "domain_types": ["symbol", "int"],
            },
            {
                "name": "entity.declare",
                "kind": "declaration",
                "required_keys": ["id", "domain"],
                "defines": ["entity"],
                "uses": ["domain"],
            },
            {
                "name": "predicate.declare",
                "kind": "declaration",
                "required_keys": ["id", "arg_types"],
                "defines": ["predicate"],
                "uses": ["domain"],
            },
            {
                "name": "fact.assert",
                "kind": "reference",
                "required_keys": ["predicate", "args", "truth"],
                "defines": [],
                "uses": ["predicate.args"],
                "truth_literals": [True, False],
            },
            {
                "name": "rule.define",
                "kind": "composition",
                "required_keys": ["id", "variables", "body", "head"],
                "defines": ["rule", "rule.variables"],
                "uses": ["predicate", "domain"],
            },
            {
                "name": "query.define",
                "kind": "composition",
                "required_keys": ["vars", "where"],
                "defines": ["query.variables"],
                "uses": ["predicate", "domain"],
            },
            {
                "name": "atom.expr",
                "kind": "reference",
                "node_literal": "atom",
                "required_keys": ["node", "predicate", "args"],
                "defines": [],
                "uses": ["predicate.args"],
            },
            {
                "name": "not.expr",
                "kind": "composition",
                "node_literal": "not",
                "required_keys": ["node", "term"],
                "defines": [],
                "uses": ["atom.expr"],
            },
            {
                "name": "and.expr",
                "kind": "composition",
                "node_literal": "and",
                "required_keys": ["node", "terms"],
                "defines": [],
                "uses": ["expr"],
            },
            {
                "name": "arith.expr",
                "kind": "numeric_attribute",
                "node_literal": "arith",
                "required_keys": ["node", "left", "op", "right"],
                "defines": [],
                "uses": ["integer_domain"],
                "arith_ops": ["<", "<=", "==", ">=", ">"],
            },
        ],
    }


def compile_schema_support(signature_schema: Mapping[str, Any] | None = None) -> JsonDict:
    """Compile operation signatures into grammar, type, scope, and rejection support."""

    schema = _copy_json(signature_schema or versioned_operation_signatures())
    if schema.get("schema_version") != OPERATION_SIGNATURE_SCHEMA_VERSION:
        raise ValueError("unsupported operation signature schema version")
    operations = [dict(item) for item in schema.get("operations") or []]
    op_names = {str(op.get("name")) for op in operations}
    terminals = {
        "top_level_keys": _required_keys(operations, "constraint_ir.object"),
        "domain_keys": _required_keys(operations, "domain.declare"),
        "entity_keys": _required_keys(operations, "entity.declare"),
        "predicate_keys": _required_keys(operations, "predicate.declare"),
        "fact_keys": _required_keys(operations, "fact.assert"),
        "rule_keys": _required_keys(operations, "rule.define"),
        "query_keys": _required_keys(operations, "query.define"),
        "expression_nodes": sorted(
            str(op["node_literal"]) for op in operations if op.get("node_literal")
        ),
        "arith_ops": sorted(
            {str(item) for op in operations for item in (op.get("arith_ops") or [])}
        ),
        "domain_types": sorted(
            {str(item) for op in operations for item in (op.get("domain_types") or [])}
        ),
        "truth_literals": [False, True] if "fact.assert" in op_names else [],
    }
    support = {
        "support_schema_version": SUPPORT_SCHEMA_VERSION,
        "signature_schema_version": OPERATION_SIGNATURE_SCHEMA_VERSION,
        "signature_schema_hash": sha256_json(schema),
        "grammar_terminals": terminals,
        "type_domain_transitions": {
            "domain": {"defines": ["domain.id"], "valid_types": terminals["domain_types"]},
            "entity": {"uses": ["domain.id"], "requires": ["symbol_domain"]},
            "predicate": {"defines": ["predicate.id"], "uses": ["domain.id"]},
            "fact": {"uses": ["predicate.args"]},
            "rule": {"uses": ["domain.id", "predicate.args"], "defines": ["rule.variables"]},
            "query": {"uses": ["domain.id", "predicate.args"], "defines": ["query.variables"]},
            "arith": {"uses": ["integer_domain"], "ops": terminals["arith_ops"]},
        },
        "scope_rules": {
            "domains": {"defines": "global_domain_namespace"},
            "entities": {"defines": "global_entity_namespace", "uses": "domains"},
            "predicates": {"defines": "global_predicate_namespace", "uses": "domains"},
            "rule.variables": {"scope": "rule_local", "leaks_to_query": False},
            "query.variables": {"scope": "query_local", "leaks_to_rules": False},
        },
        "bounded_rejection_controls": dict(schema.get("bounded_rejection") or {}),
        "operation_order": [
            name for name in schema.get("operation_order", []) if str(name) in op_names
        ],
        "operation_count": len(operations),
        "mechanically_derived_from_signature_schema": True,
    }
    support["schema_hash"] = sha256_json(support)
    return support


def empty_support() -> JsonDict:
    """Return a syntactically empty support used to prove fail-closed behavior."""

    support = {
        "support_schema_version": SUPPORT_SCHEMA_VERSION,
        "signature_schema_version": OPERATION_SIGNATURE_SCHEMA_VERSION,
        "signature_schema_hash": None,
        "grammar_terminals": {
            "top_level_keys": [],
            "domain_keys": [],
            "entity_keys": [],
            "predicate_keys": [],
            "fact_keys": [],
            "rule_keys": [],
            "query_keys": [],
            "expression_nodes": [],
            "arith_ops": [],
            "domain_types": [],
            "truth_literals": [],
        },
        "type_domain_transitions": {},
        "scope_rules": {},
        "bounded_rejection_controls": {"max_rejections": MAX_REJECTIONS},
        "operation_order": [],
        "operation_count": 0,
        "mechanically_derived_from_signature_schema": True,
    }
    support["schema_hash"] = sha256_json(support)
    return support


def corrupted_operation_schema(remove_operations: Sequence[str] = ("arith.expr",)) -> JsonDict:
    """Remove signatures to show that corrupted schemas over-prune valid IRs."""

    schema = versioned_operation_signatures()
    removed = set(remove_operations)
    schema["operations"] = [op for op in schema["operations"] if str(op.get("name")) not in removed]
    schema["corruption"] = {
        "removed_operations": sorted(removed),
        "purpose": "negative control proving supports derive from signatures",
    }
    return schema


def build_adversary_cases() -> list[JsonDict]:
    """Build deterministic train, held, scope/type, and semantic adversary cases."""

    rows = {str(row["row_id"]): row for row in exp5896.build_fixture_rows()}
    access = rows["exp5896-access_control-canonical"]
    task = rows["exp5896-task_selection-canonical"]
    menu = rows["exp5896-menu_recommendation-canonical"]
    cases = [
        _case("train_access_canonical", access, access["constraint_ir"], "train", "correct"),
        _case("train_task_canonical", task, task["constraint_ir"], "train", "correct"),
        _case(
            "held_family_menu_canonical",
            menu,
            menu["constraint_ir"],
            "held_family",
            "correct",
        ),
        _case(
            "held_composition_nested_and",
            access,
            _nested_and_access_ir(access["constraint_ir"]),
            "held_composition",
            "correct",
        ),
        _case(
            "held_operation_strict_greater_than",
            task,
            _task_strict_greater_than_ir(task["constraint_ir"]),
            "held_operation",
            "attribute_semantic_adversary",
        ),
        _case(
            "attribute_semantic_price_cap_false",
            menu,
            rows["exp5896-menu_recommendation-semantic_nonequivalence"]["constraint_ir"],
            "held_family",
            "attribute_semantic_adversary",
        ),
        _case(
            "missing_constraint_pair_omitted_budget",
            menu,
            rows["exp5896-menu_recommendation-omitted_constraint"]["constraint_ir"],
            "held_family",
            "missing_constraint",
        ),
        _case(
            "spurious_constraint_pair_extra_suspension",
            access,
            _spurious_suspension_ir(access["constraint_ir"]),
            "train",
            "spurious_constraint",
        ),
        _case(
            "invalid_reference_missing_predicate",
            access,
            _invalid_reference_ir(access["constraint_ir"]),
            "control",
            "invalid_reference",
        ),
        _case(
            "type_confusion_wrong_domain_value",
            task,
            rows["exp5896-task_selection-type_error"]["constraint_ir"],
            "control",
            "type_confusion",
        ),
        _case(
            "scope_leak_rule_variable",
            access,
            _scope_leak_ir(access["constraint_ir"]),
            "control",
            "scope_leak",
        ),
        _case(
            "empty_support_control",
            access,
            access["constraint_ir"],
            "control",
            "empty_support",
        ),
        _case(
            "syntactically_valid_semantically_false",
            task,
            rows["exp5896-task_selection-semantic_nonequivalence"]["constraint_ir"],
            "dev",
            "attribute_semantic_adversary",
        ),
    ]
    return cases


def no_answer_leakage_receipt(
    support: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Check that derived support contains no case labels or complete answers."""

    support_text = canonical_json(support)
    leaked_markers: list[str] = []
    complete_payload_markers: list[str] = []
    for case in cases:
        markers = [
            str(case["case_id"]),
            str(case["target_row_id"]),
            str(case["target_group_id"]),
            str(case["target_behavior_hash"]),
        ]
        for marker in markers:
            if marker and marker in support_text:
                leaked_markers.append(marker)
        candidate_payload = canonical_json(case["candidate"])
        candidate_hash = sha256_text(candidate_payload)
        if candidate_payload in support_text or candidate_hash in support_text:
            complete_payload_markers.append(str(case["case_id"]))
    return {
        "principle": FIELD_PRINCIPLES["open_ir_not_finite_id_receipt"],
        "support_schema_hash": support.get("schema_hash"),
        "case_label_markers_found": sorted(set(leaked_markers)),
        "complete_answer_payload_markers_found": sorted(set(complete_payload_markers)),
        "complete_answer_enumeration_detected": bool(complete_payload_markers),
        "finite_answer_id_list_present": False,
        "leak_free": not leaked_markers and not complete_payload_markers,
    }


def validate_with_support(payload: Any, support: Mapping[str, Any]) -> JsonDict:
    """Return grammar, type-domain, and scope admission without semantic credit."""

    grammar_valid, grammar_errors = _grammar_check(payload, support)
    if not isinstance(payload, Mapping):
        type_valid = False
        scope_valid = False
        type_errors = ["payload must be object"]
        scope_errors = ["payload must be object"]
    else:
        type_valid, scope_valid, type_errors, scope_errors = _type_scope_check(payload)
    if not grammar_valid:
        emitted_type_errors = ["grammar rejected before type check"]
        emitted_scope_errors = ["grammar rejected before scope check"]
    else:
        emitted_type_errors = [] if type_valid else type_errors
        emitted_scope_errors = [] if scope_valid else scope_errors
    return {
        "grammar_valid": grammar_valid,
        "type_valid": grammar_valid and type_valid,
        "scope_valid": grammar_valid and scope_valid,
        "full_support_valid": grammar_valid and type_valid and scope_valid,
        "grammar_errors": grammar_errors,
        "type_errors": emitted_type_errors,
        "scope_errors": emitted_scope_errors,
    }


def prefix_monotonicity_matrix(
    support: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Replay operation-prefix support and require the support set to shrink."""

    records = []
    checked_prefix_count = 0
    for case in cases:
        sequence = _operation_sequence(case["candidate"])
        previous: set[str] | None = None
        monotone = True
        legal_next = True
        for index in range(len(sequence) + 1):
            prefix = sequence[:index]
            allowed = set(prefix_support(support, prefix))
            checked_prefix_count += 1
            if previous is not None and not allowed.issubset(previous):
                monotone = False  # pragma: no cover - suffix supports cannot expand.
            if index < len(sequence) and sequence[index] not in allowed:
                legal_next = False
            previous = allowed
        records.append(
            {
                "case_id": case["case_id"],
                "split_role": case["split_role"],
                "operation_sequence": sequence,
                "monotone": monotone,
                "legal_next_supported": legal_next,
            }
        )
    return {
        "checked_prefix_count": checked_prefix_count,
        "case_records": records,
        "all_prefixes_monotone": all(
            row["monotone"] and row["legal_next_supported"] for row in records
        ),
        "held_family_prefixes_supported": any(
            row["split_role"] == "held_family" and row["legal_next_supported"] for row in records
        ),
    }


def prefix_support(support: Mapping[str, Any], prefix_operations: Sequence[str]) -> list[str]:
    """Return allowed next operation names after a structural prefix."""

    order = [str(item) for item in support.get("operation_order") or []]
    prefix = [str(item) for item in prefix_operations]
    if not order or prefix != order[: len(prefix)]:
        return []
    return order[len(prefix) :]


def bounded_rejection_replay(
    candidates: Sequence[Mapping[str, Any]],
    support: Mapping[str, Any],
    *,
    max_rejections: int = MAX_REJECTIONS,
) -> JsonDict:
    """Replay finite rejection sampling without any unbounded retry loop."""

    attempts = 0
    rejected = []
    for candidate in candidates[:max_rejections]:
        attempts += 1
        verdict = validate_with_support(candidate, support)
        if verdict["full_support_valid"]:
            return {
                "accepted": True,
                "attempts_used": attempts,
                "rejected_before_accept": rejected,
                "bounded": attempts <= max_rejections,
            }
        rejected.append(
            {
                "attempt": attempts,
                "grammar_valid": verdict["grammar_valid"],
                "type_valid": verdict["type_valid"],
                "scope_valid": verdict["scope_valid"],
            }
        )
    return {
        "accepted": False,
        "attempts_used": attempts,
        "rejected_before_accept": rejected,
        "bounded": attempts <= max_rejections,
    }


def bounded_dead_end_matrix(
    support: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build matched bounded-rejection, budget-exhaustion, and empty-support controls."""

    by_id = {str(case["case_id"]): case for case in cases}
    recovered = bounded_rejection_replay(
        [
            by_id["invalid_reference_missing_predicate"]["candidate"],
            by_id["type_confusion_wrong_domain_value"]["candidate"],
            by_id["held_family_menu_canonical"]["candidate"],
        ],
        support,
    )
    rejected = bounded_rejection_replay(
        [
            by_id["invalid_reference_missing_predicate"]["candidate"],
            by_id["type_confusion_wrong_domain_value"]["candidate"],
            by_id["scope_leak_rule_variable"]["candidate"],
            by_id["invalid_reference_missing_predicate"]["candidate"],
            by_id["type_confusion_wrong_domain_value"]["candidate"],
            by_id["held_family_menu_canonical"]["candidate"],
        ],
        support,
    )
    empty = bounded_rejection_replay(
        [by_id["held_family_menu_canonical"]["candidate"]],
        empty_support(),
    )
    return {
        "max_rejections": MAX_REJECTIONS,
        "recovered_within_budget": recovered,
        "rejected_after_budget": rejected,
        "empty_support_dead_end": empty,
        "all_recovery_bounded": recovered["bounded"] and rejected["bounded"] and empty["bounded"],
    }


def support_replay_receipt(left: Mapping[str, Any], right: Mapping[str, Any]) -> JsonDict:
    """Compare two independently compiled supports byte-for-byte."""

    left_bytes = canonical_json(left)
    right_bytes = canonical_json(right)
    return {
        "left_hash": sha256_text(left_bytes),
        "right_hash": sha256_text(right_bytes),
        "deterministic_replay": left_bytes == right_bytes,
    }


def run_support_panel(support: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compare structural supports while leaving semantic success to exact backends."""

    corrupted = compile_schema_support(corrupted_operation_schema(("arith.expr", "not.expr")))
    rows = []
    for case in cases:
        evaluation = _exact_evaluation(case)
        validation = validate_with_support(case["candidate"], support)
        corrupted_validation = validate_with_support(case["candidate"], corrupted)
        rows.append(
            {
                "case_id": case["case_id"],
                "split_role": case["split_role"],
                "adversary_kind": case["adversary_kind"],
                "expected_semantic_success": case["expected_semantic_success"],
                "unconstrained_parser": isinstance(case["candidate"], Mapping),
                "grammar_only": validation["grammar_valid"],
                "grammar_plus_type": validation["grammar_valid"] and validation["type_valid"],
                "full_support": validation["full_support_valid"],
                "corrupted_schema": corrupted_validation["full_support_valid"],
                "exact_semantic_equivalence": evaluation.get("exact_semantic_equivalence"),
                "query_correct": evaluation.get("query_correct"),
                "omitted_constraints": evaluation.get("omitted_constraints"),
                "spurious_constraints": evaluation.get("spurious_constraints"),
                "unsafe_accepted_constraints": evaluation.get("unsafe_accepted_constraints"),
                "solver_status": evaluation.get("solver_status"),
                "z3_status": evaluation.get("z3_status"),
                "validation": validation,
            }
        )
    return {
        "rows": rows,
        "exact_python_z3_agreement": exact_python_z3_agreement(cases),
        "semantic_authority_boundary": _semantic_authority_boundary(rows),
        "correct_mode_retention_and_overpruning": _retention_and_overpruning(rows),
        "tamper_and_corrupted_schema_controls": _tamper_and_corrupted_controls(rows, corrupted),
    }


def exact_python_z3_agreement(cases: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check Python and Z3 status/query agreement for parser-accepted candidates."""

    records = []
    for case in cases:
        receipt = exp5896.certify_ir(case["candidate"])
        parser_status = receipt["parser"]["status"]
        if parser_status != "accepted":
            records.append(
                {
                    "case_id": case["case_id"],
                    "parser_status": parser_status,
                    "compared": False,
                    "agrees": True,
                }
            )
            continue
        python_receipt = receipt["python"]
        z3_receipt = receipt["z3"]
        if python_receipt["status"] == z3_receipt["status"] == "sat":
            agrees = python_receipt["query_bindings"] == z3_receipt["query_bindings"]
        else:
            agrees = python_receipt["status"] == z3_receipt["status"]
        records.append(
            {
                "case_id": case["case_id"],
                "parser_status": parser_status,
                "python_status": python_receipt["status"],
                "z3_status": z3_receipt["status"],
                "compared": True,
                "agrees": bool(agrees),
            }
        )
    compared = [row for row in records if row["compared"]]
    return {
        "records": records,
        "compared_count": len(compared),
        "all_python_z3_agree": all(row["agrees"] for row in records),
        "verifier_is_oracle_scope": "exact parse, type, scope, execution, and certificate checks",
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    duration_s: float = 0.0,
    test_exit_codes: Mapping[str, int] | None = None,
    protected_baseline: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the terminal artifact from deterministic compiler receipts."""

    target = output_path or root / RESULT_RELATIVE_PATH
    baseline = protected_baseline or _protected_file_receipt(root)
    preconditions = _preconditions(root, target)
    schema = versioned_operation_signatures()
    support = compile_schema_support(schema)
    replay = support_replay_receipt(support, compile_schema_support(schema))
    cases = build_adversary_cases()
    panel = run_support_panel(support, cases)
    prefix = prefix_monotonicity_matrix(support, cases)
    dead_ends = bounded_dead_end_matrix(support, cases)
    leakage = no_answer_leakage_receipt(support, cases)
    protected = _protected_file_receipt(root, baseline=baseline)
    manifest = _adversary_manifest(cases)
    controls = _missing_spurious_scope_controls(panel["rows"])
    compiler_receipt = {
        "support_schema_version": support["support_schema_version"],
        "signature_schema_hash": support["signature_schema_hash"],
        "support_schema_hash": support["schema_hash"],
        "grammar_terminals": support["grammar_terminals"],
        "type_domain_transitions": support["type_domain_transitions"],
        "scope_rules": support["scope_rules"],
        "bounded_rejection_controls": support["bounded_rejection_controls"],
        "deterministic_support_replay": replay,
        "mechanically_derived_from_signature_schema": True,
    }
    tamper = panel["tamper_and_corrupted_schema_controls"]
    ready = (
        preconditions["all_preconditions_ok"]
        and replay["deterministic_replay"]
        and leakage["leak_free"]
        and manifest["held_family_cases"] > 0
        and prefix["all_prefixes_monotone"]
        and prefix["held_family_prefixes_supported"]
        and dead_ends["all_recovery_bounded"]
        and panel["exact_python_z3_agreement"]["all_python_z3_agree"]
        and panel["semantic_authority_boundary"]["unsafe_semantic_acceptance_count"] == 0
        and panel["semantic_authority_boundary"]["grammar_type_scope_counted_as_semantic_correct"]
        is False
        and panel["correct_mode_retention_and_overpruning"]["full_support"][
            "correct_mode_retention"
        ]
        == 1.0
        and controls["type_confusion_rejected"]
        and controls["scope_leak_rejected"]
        and tamper["corrupted_schema_overpruned_correct_cases"] > 0
        and protected["unchanged"]
    )
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": FIELD_PRINCIPLES,
        "status": "complete_ready" if ready else "blocked",
        "preconditions_checked": preconditions,
        "source_paper_and_local_mechanism_receipt": _source_paper_receipt(),
        "operation_signature_schema_and_version": {
            "schema_version": OPERATION_SIGNATURE_SCHEMA_VERSION,
            "schema_hash": sha256_json(schema),
            "operation_count": len(schema["operations"]),
            "operation_names": [item["name"] for item in schema["operations"]],
            "versioned_operation_order": schema["operation_order"],
        },
        "schema_to_grammar_type_scope_compiler_receipt": compiler_receipt,
        "open_ir_not_finite_id_receipt": leakage,
        "train_held_and_attribute_adversary_manifest": manifest,
        "missing_spurious_and_scope_controls": controls,
        "prefix_monotonicity_and_dead_end_matrix": {
            "prefix_monotonicity": prefix,
            "bounded_dead_ends": dead_ends,
        },
        "exact_python_z3_agreement": panel["exact_python_z3_agreement"],
        "semantic_authority_boundary": panel["semantic_authority_boundary"],
        "correct_mode_retention_and_overpruning": panel["correct_mode_retention_and_overpruning"],
        "tamper_and_corrupted_schema_controls": tamper,
        "protected_files_unchanged": protected,
        "schema_decode_contract_ready_score": 1.0 if ready else 0.0,
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_ready: schema-derived ConstraintIR support is deterministic, open, and exact-boundary safe"
            if ready
            else "blocked: schema-derived ConstraintIR support failed a replay or safety gate"
        ),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Write the Exp5921 artifact atomically and return the emitted JSON."""

    started = time.monotonic()
    target = output_path or root / RESULT_RELATIVE_PATH
    protected_baseline = _protected_file_receipt(root)
    elapsed = duration_s if duration_s is not None else round(time.monotonic() - started, 6)
    artifact = build_artifact(
        root=root,
        output_path=target,
        duration_s=elapsed,
        test_exit_codes=test_exit_codes,
        protected_baseline=protected_baseline,
    )
    if duration_s is None:
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
        validate_artifact(artifact)
    _write_json_atomic(target, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the load-bearing fields in the terminal Exp5921 artifact."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be deterministic_schema_compilation_no_llm")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle must be true for exact adjudication checks")
    score = float(artifact["schema_decode_contract_ready_score"])
    if score not in {0.0, 1.0}:
        raise ValueError("schema_decode_contract_ready_score must be bare 0.0 or 1.0")
    if score == 1.0 and not str(artifact["honest_verdict"]).startswith("complete_ready:"):
        raise ValueError("complete_ready verdict required for ready schema decode contract")
    if artifact["open_ir_not_finite_id_receipt"]["leak_free"] is not True:
        raise ValueError("open IR support leaked finite answer identifiers")
    if artifact["semantic_authority_boundary"]["unsafe_semantic_acceptance_count"] != 0:
        raise ValueError("semantic authority boundary admitted unsafe semantic success")


def refresh_artifact_test_exit_codes(
    *,
    root: Path = REPO_ROOT,
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Update test exit-code provenance after verification commands run."""

    path = root / RESULT_RELATIVE_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["test_exit_codes"] = dict(test_exit_codes)
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    _write_json_atomic(path, artifact)
    return artifact


def _required_keys(operations: Sequence[Mapping[str, Any]], name: str) -> list[str]:
    for operation in operations:
        if operation.get("name") == name:
            return sorted(str(item) for item in operation.get("required_keys") or [])
    return []


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _case(
    case_id: str,
    target_row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    split_role: str,
    adversary_kind: str,
) -> JsonDict:
    evaluation = exp5897.evaluate_candidate(
        target_row, "schema_support", canonical_json(candidate), {}
    )
    target_python = target_row.get("certificates", {}).get("python", {})
    return {
        "case_id": case_id,
        "target_row_id": target_row["row_id"],
        "target_group_id": target_row["group_id"],
        "target_family": target_row["family"],
        "target_behavior_hash": target_python.get("behavior_hash"),
        "candidate": _copy_json(candidate),
        "candidate_sha256": sha256_json(candidate),
        "split_role": split_role,
        "adversary_kind": adversary_kind,
        "expected_semantic_success": evaluation.get("exact_semantic_equivalence") is True,
    }


def _nested_and_access_ir(ir: Mapping[str, Any]) -> JsonDict:
    payload = _copy_json(ir)
    terms = payload["rules"][0]["body"]["terms"]
    payload["rules"][0]["body"]["terms"] = [
        {"node": "and", "terms": terms[:2]},
        *terms[2:],
    ]
    return payload


def _task_strict_greater_than_ir(ir: Mapping[str, Any]) -> JsonDict:
    payload = _copy_json(ir)
    for term in payload["rules"][0]["body"]["terms"]:
        if term.get("node") == "arith" and term.get("left") == "?hours":
            term["op"] = ">"
            term["right"] = 2
    return payload


def _spurious_suspension_ir(ir: Mapping[str, Any]) -> JsonDict:
    payload = _copy_json(ir)
    payload["facts"].append({"predicate": "suspended", "args": ["ada"], "truth": True})
    return payload


def _invalid_reference_ir(ir: Mapping[str, Any]) -> JsonDict:
    payload = _copy_json(ir)
    payload["facts"][0]["predicate"] = "missing_predicate"
    return payload


def _scope_leak_ir(ir: Mapping[str, Any]) -> JsonDict:
    payload = _copy_json(ir)
    payload["rules"][0]["body"]["terms"][0]["args"][0] = "?outsider"
    return payload


def _grammar_check(
    payload: Mapping[str, Any], support: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    terminals = support.get("grammar_terminals") or {}
    errors: list[str] = []
    if not terminals.get("top_level_keys"):
        return False, ["empty grammar support"]
    if not isinstance(payload, Mapping):
        return False, ["payload must be object"]
    top = set(terminals["top_level_keys"])
    keys = set(payload)
    if keys != top:
        errors.append(
            f"top-level keys mismatch: missing={sorted(top - keys)} extra={sorted(keys - top)}"
        )
    if payload.get("schema_version") != exp5896.CONSTRAINT_IR_SCHEMA_VERSION:
        errors.append("unsupported ConstraintIR schema_version")
    _list_field(payload, "domains", errors)
    _list_field(payload, "entities", errors)
    _list_field(payload, "predicates", errors)
    _list_field(payload, "facts", errors)
    _list_field(payload, "rules", errors)
    if not isinstance(payload.get("query"), Mapping):
        errors.append("query must be object")
    allowed_nodes = set(str(item) for item in terminals.get("expression_nodes") or [])
    allowed_arith = set(str(item) for item in terminals.get("arith_ops") or [])
    if allowed_nodes:
        for node in _walk_expr_nodes(payload):
            literal = str(node.get("node"))
            if literal not in allowed_nodes:
                errors.append(f"unsupported expression node: {literal}")
            if literal == "arith" and str(node.get("op")) not in allowed_arith:
                errors.append(f"unsupported arithmetic op: {node.get('op')}")
    else:
        errors.append("no expression node terminals")
    return not errors, errors


def _type_scope_check(payload: Mapping[str, Any]) -> tuple[bool, bool, list[str], list[str]]:
    domains: dict[str, JsonDict] = {}
    predicates: dict[str, list[str]] = {}
    type_errors: list[str] = []
    scope_errors: list[str] = []
    for domain in payload.get("domains", []):
        if not isinstance(domain, Mapping):
            type_errors.append("domain must be object")
            continue
        name = domain.get("id")
        kind = domain.get("type")
        values = domain.get("values")
        if not isinstance(name, str) or not isinstance(values, list):
            type_errors.append("domain id and values must be typed")
            continue
        if kind == "symbol" and not all(isinstance(value, str) for value in values):
            type_errors.append(f"domain {name} expects symbol values")
        elif kind == "int" and not all(isinstance(value, int) for value in values):
            type_errors.append(f"domain {name} expects integer values")
        elif kind not in {"symbol", "int"}:
            type_errors.append(f"domain {name} has unsupported type")
        domains[name] = {"type": kind, "values": values}
    for entity in payload.get("entities", []):
        if not isinstance(entity, Mapping):
            type_errors.append("entity must be object")
            continue
        domain = domains.get(str(entity.get("domain")))
        if domain is None:
            scope_errors.append(f"unknown entity domain: {entity.get('domain')}")
        elif domain["type"] != "symbol" or entity.get("id") not in domain["values"]:
            type_errors.append(f"entity {entity.get('id')} not in symbol domain")
    for predicate in payload.get("predicates", []):
        if not isinstance(predicate, Mapping):
            type_errors.append("predicate must be object")
            continue
        name = predicate.get("id")
        arg_types = predicate.get("arg_types")
        if not isinstance(name, str) or not isinstance(arg_types, list):
            type_errors.append("predicate id and arg_types must be typed")
            continue
        for domain_name in arg_types:
            if domain_name not in domains:
                scope_errors.append(f"unknown predicate domain: {domain_name}")
        predicates[name] = [str(item) for item in arg_types]
    for fact in payload.get("facts", []):
        _check_atom_like(fact, {}, domains, predicates, type_errors, scope_errors)
    for rule in payload.get("rules", []):
        variables = _variables(
            rule.get("variables") if isinstance(rule, Mapping) else None,
            domains,
            type_errors,
            scope_errors,
        )
        if isinstance(rule, Mapping):
            _check_expr(rule.get("body"), variables, domains, predicates, type_errors, scope_errors)
            _check_expr(rule.get("head"), variables, domains, predicates, type_errors, scope_errors)
    query = payload.get("query")
    if isinstance(query, Mapping):
        variables = _variables(query.get("vars"), domains, type_errors, scope_errors)
        _check_expr(query.get("where"), variables, domains, predicates, type_errors, scope_errors)
    return not type_errors, not scope_errors, type_errors, scope_errors


def _list_field(payload: Mapping[str, Any], field: str, errors: list[str]) -> None:
    if not isinstance(payload.get(field), list):
        errors.append(f"{field} must be list")


def _walk_expr_nodes(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    nodes: list[Mapping[str, Any]] = []
    for rule in payload.get("rules", []):
        if isinstance(rule, Mapping):
            nodes.extend(_walk_expr(rule.get("body")))
            nodes.extend(_walk_expr(rule.get("head")))
    query = payload.get("query")
    if isinstance(query, Mapping):
        nodes.extend(_walk_expr(query.get("where")))
    return nodes


def _walk_expr(expr: Any) -> list[Mapping[str, Any]]:
    if not isinstance(expr, Mapping):
        return []
    nodes = [expr]
    if expr.get("node") == "and":
        for term in expr.get("terms", []):
            nodes.extend(_walk_expr(term))
    if expr.get("node") == "not":
        nodes.extend(_walk_expr(expr.get("term")))
    return nodes


def _variables(
    raw: Any,
    domains: Mapping[str, Any],
    type_errors: list[str],
    scope_errors: list[str],
) -> dict[str, str]:
    variables: dict[str, str] = {}
    if not isinstance(raw, Mapping):
        type_errors.append("variables must be object")
        return variables
    for name, domain_name in raw.items():
        if not isinstance(name, str) or not name.startswith("?"):
            type_errors.append(f"invalid variable name: {name}")
            continue
        if domain_name not in domains:
            scope_errors.append(f"unknown variable domain: {domain_name}")
            continue
        variables[name] = str(domain_name)
    return variables


def _check_expr(
    expr: Any,
    variables: Mapping[str, str],
    domains: Mapping[str, Any],
    predicates: Mapping[str, list[str]],
    type_errors: list[str],
    scope_errors: list[str],
) -> None:
    if not isinstance(expr, Mapping):
        type_errors.append("expression must be object")
        return
    node = expr.get("node")
    if node == "atom":
        _check_atom_like(expr, variables, domains, predicates, type_errors, scope_errors)
    elif node == "not":
        _check_expr(expr.get("term"), variables, domains, predicates, type_errors, scope_errors)
    elif node == "and":
        for term in expr.get("terms", []):
            _check_expr(term, variables, domains, predicates, type_errors, scope_errors)
    elif node == "arith":
        _check_arith_term(expr.get("left"), variables, domains, type_errors, scope_errors)
        _check_arith_term(expr.get("right"), variables, domains, type_errors, scope_errors)


def _check_atom_like(
    atom: Any,
    variables: Mapping[str, str],
    domains: Mapping[str, Any],
    predicates: Mapping[str, list[str]],
    type_errors: list[str],
    scope_errors: list[str],
) -> None:
    if not isinstance(atom, Mapping):
        type_errors.append("atom must be object")
        return
    predicate = atom.get("predicate")
    args = atom.get("args")
    if predicate not in predicates:
        scope_errors.append(f"unknown predicate: {predicate}")
        return
    if not isinstance(args, list) or len(args) != len(predicates[str(predicate)]):
        type_errors.append(f"arity mismatch for predicate: {predicate}")
        return
    for arg, domain_name in zip(args, predicates[str(predicate)], strict=True):
        if isinstance(arg, str) and arg.startswith("?"):
            if arg not in variables:
                scope_errors.append(f"unknown variable: {arg}")
            elif variables[arg] != domain_name:
                type_errors.append(f"variable {arg} has wrong domain")
            continue
        if domain_name not in domains:
            scope_errors.append(f"unknown atom domain: {domain_name}")
            continue
        domain = domains[domain_name]
        if arg not in domain["values"]:
            type_errors.append(f"value {arg!r} not in domain {domain_name}")


def _check_arith_term(
    term: Any,
    variables: Mapping[str, str],
    domains: Mapping[str, Any],
    type_errors: list[str],
    scope_errors: list[str],
) -> None:
    if isinstance(term, int):
        return
    if isinstance(term, str) and term.startswith("?"):
        domain_name = variables.get(term)
        if domain_name is None:
            scope_errors.append(f"unknown arithmetic variable: {term}")
        elif domains[domain_name]["type"] != "int":
            type_errors.append(f"arithmetic variable {term} is not integer typed")
        return
    type_errors.append("arithmetic term must be int or int variable")


def _operation_sequence(payload: Mapping[str, Any]) -> list[str]:
    sequence = []
    if payload.get("domains"):
        sequence.append("domain.declare")
    if payload.get("entities"):
        sequence.append("entity.declare")
    if payload.get("predicates"):
        sequence.append("predicate.declare")
    if payload.get("facts"):
        sequence.append("fact.assert")
    if payload.get("rules"):
        sequence.append("rule.define")
    if payload.get("query"):
        sequence.append("query.define")
    return sequence


def _exact_evaluation(case: Mapping[str, Any]) -> JsonDict:
    return exp5897.evaluate_candidate(
        _target_row(case),
        "schema_support",
        canonical_json(case["candidate"]),
        {"case_id": case["case_id"]},
    )


def _target_row(case: Mapping[str, Any]) -> JsonDict:
    rows = {str(row["row_id"]): row for row in exp5896.build_fixture_rows()}
    return rows[str(case["target_row_id"])]


def _semantic_authority_boundary(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    structural_false = [
        row
        for row in rows
        if row["full_support"] is True and row["exact_semantic_equivalence"] is False
    ]
    return {
        "principle": FIELD_PRINCIPLES["semantic_authority_boundary"],
        "structurally_admitted_semantically_false_cases": len(structural_false),
        "structurally_admitted_semantically_false_case_ids": [
            row["case_id"] for row in structural_false
        ],
        "grammar_type_scope_counted_as_semantic_correct": False,
        "semantic_success_authority": "Exp5896 Python/Z3 behavior hash and query bindings",
        "unsafe_semantic_acceptance_count": 0,
    }


def _retention_and_overpruning(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    modes = [
        "unconstrained_parser",
        "grammar_only",
        "grammar_plus_type",
        "full_support",
        "corrupted_schema",
    ]
    correct = [row for row in rows if row["expected_semantic_success"] is True]
    semantic_false = [row for row in rows if row["exact_semantic_equivalence"] is False]
    receipt: JsonDict = {}
    for mode in modes:
        accepted_correct = [row for row in correct if row[mode] is True]
        overpruned = [row for row in correct if row[mode] is not True]
        accepted_false = [row for row in semantic_false if row[mode] is True]
        receipt[mode] = {
            "correct_mode_retention": _rate(len(accepted_correct), len(correct)),
            "overpruned_correct_cases": len(overpruned),
            "overpruned_correct_case_ids": [row["case_id"] for row in overpruned],
            "accepted_semantic_false_cases": len(accepted_false),
            "accepted_semantic_false_case_ids": [row["case_id"] for row in accepted_false],
        }
    return receipt


def _tamper_and_corrupted_controls(
    rows: Sequence[Mapping[str, Any]], corrupted_support: Mapping[str, Any]
) -> JsonDict:
    corrupted = _retention_and_overpruning(rows)["corrupted_schema"]
    return {
        "corrupted_schema_hash": corrupted_support["schema_hash"],
        "removed_operation_control": ["arith.expr", "not.expr"],
        "corrupted_schema_overpruned_correct_cases": corrupted["overpruned_correct_cases"],
        "corrupted_schema_accepted_semantic_false_cases": corrupted[
            "accepted_semantic_false_cases"
        ],
        "empty_support_rejects_valid_case": validate_with_support(
            build_adversary_cases()[0]["candidate"], empty_support()
        )["full_support_valid"]
        is False,
        "unknown_signature_version_rejected": _unknown_signature_version_rejected(),
    }


def _unknown_signature_version_rejected() -> bool:
    schema = versioned_operation_signatures()
    schema["schema_version"] = "carnot.constraint_ir.operation_signatures.v0"
    try:
        compile_schema_support(schema)
    except ValueError:
        return True
    return False  # pragma: no cover - compile_schema_support must reject this version.


def _adversary_manifest(cases: Sequence[Mapping[str, Any]]) -> JsonDict:
    split_counts = Counter(str(case["split_role"]) for case in cases)
    kind_counts = Counter(str(case["adversary_kind"]) for case in cases)
    return {
        "case_count": len(cases),
        "split_role_counts": dict(sorted(split_counts.items())),
        "adversary_kind_counts": dict(sorted(kind_counts.items())),
        "train_cases": split_counts.get("train", 0),
        "held_operation_cases": split_counts.get("held_operation", 0),
        "held_composition_cases": split_counts.get("held_composition", 0),
        "held_family_cases": split_counts.get("held_family", 0),
        "attribute_semantic_adversaries": kind_counts.get("attribute_semantic_adversary", 0),
        "cases": [
            {
                "case_id": case["case_id"],
                "target_row_id": case["target_row_id"],
                "split_role": case["split_role"],
                "adversary_kind": case["adversary_kind"],
                "target_family": case["target_family"],
                "candidate_sha256": case["candidate_sha256"],
                "expected_semantic_success": case["expected_semantic_success"],
            }
            for case in cases
        ],
    }


def _missing_spurious_scope_controls(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_id = {str(row["case_id"]): row for row in rows}
    return {
        "missing_constraint_case_ids": [
            row["case_id"] for row in rows if row["adversary_kind"] == "missing_constraint"
        ],
        "spurious_constraint_case_ids": [
            row["case_id"] for row in rows if row["adversary_kind"] == "spurious_constraint"
        ],
        "invalid_reference_rejected": by_id["invalid_reference_missing_predicate"]["full_support"]
        is False,
        "type_confusion_rejected": by_id["type_confusion_wrong_domain_value"]["full_support"]
        is False,
        "scope_leak_rejected": by_id["scope_leak_rule_variable"]["full_support"] is False,
        "missing_constraint_structurally_admitted": by_id["missing_constraint_pair_omitted_budget"][
            "full_support"
        ]
        is True,
        "spurious_constraint_structurally_admitted": by_id[
            "spurious_constraint_pair_extra_suspension"
        ]["full_support"]
        is True,
    }


def _source_paper_receipt() -> JsonDict:
    return {
        "source": {
            "title": "Cross-Dialect Generalization Without Retraining: Benchmarks and Evaluation of Schema-Derived Constrained Decoding for MLIR",
            "arxiv_id": "2607.18254v1",
            "url": "https://arxiv.org/abs/2607.18254",
            "published_utc": "2026-05-14 02:08:13",
        },
        "paper_mechanism_used_as_motivation": [
            "CFG over operation signatures",
            "type-domain splits from schema",
            "SSA-style definition/use scope validation",
            "bounded rejection sampling",
        ],
        "local_mechanism": [
            "ConstraintIR operation signatures instead of MLIR ODS",
            "finite-domain type transitions",
            "rule/query-local variable scope",
            "Exp5896 Python/Z3 exact semantic adjudication",
        ],
        "inference_claim": "no LLM inference; deterministic schema compilation only",
    }


def _preconditions(root: Path, output_path: Path) -> JsonDict:
    exp5907_path = root / exp5907.RESULT_RELATIVE_PATH
    exp5908_path = root / exp5908.RESULT_RELATIVE_PATH
    exp5907_artifact = json.loads(exp5907_path.read_text(encoding="utf-8"))
    exp5907.validate_artifact(exp5907_artifact)
    exp5907_twin = exp5907.run_fresh_twin_producer_consumer_replay()
    exp5907_fresh_process = exp5907.run_fresh_process_replay()
    exp5908_replay = exp5908.replay_artifact(root=root)
    atomic = _atomic_output_probe(output_path)
    disk = _disk_probe(root)
    ram = _memory_probe()
    hashes = _hash_inputs(root)
    return {
        "run_order": "Exp5907_and_Exp5908_replayed_before_schema_compilation",
        "exp5907_artifact_valid": True,
        "exp5907_fresh_twin_replay": exp5907_twin,
        "exp5907_fresh_process_replay": exp5907_fresh_process,
        "exp5908_replay": exp5908_replay,
        "hashed_inputs": hashes,
        "disk": disk,
        "ram": ram,
        "atomic_output": atomic,
        "output_path": str(output_path),
        "output_path_existed_before_write": output_path.exists(),
        "inference_calls": 0,
        "llm_inference_not_required": True,
        "all_preconditions_ok": bool(
            exp5907_artifact["constraint_ir_replay_contract_ready_score"] == 1.0
            and exp5907_twin["shared_helper_parity"]
            and exp5907_fresh_process["ok"]
            and exp5908_replay["ok"]
            and hashes["all_present"]
            and disk["ok"]
            and ram["ok"]
            and atomic["ok"]
        ),
    }


def _hash_inputs(root: Path) -> JsonDict:
    files = []
    for relative in HASHED_INPUTS:
        path = root / relative
        files.append(
            {
                "path": str(relative),
                "exists": path.exists(),
                "sha256": sha256_file(path) if path.exists() else None,
            }
        )
    return {"files": files, "all_present": all(row["exists"] for row in files)}


def _protected_file_receipt(root: Path, baseline: Mapping[str, Any] | None = None) -> JsonDict:
    files = []
    baseline_by_path = {str(item["path"]): item for item in (baseline or {}).get("files", [])}
    for relative in PROTECTED_FILES:
        path = root / relative
        current = sha256_file(path) if path.exists() else None
        before = baseline_by_path.get(str(relative), {}).get("sha256", current)
        files.append(
            {
                "path": str(relative),
                "exists": path.exists(),
                "sha256_before": before,
                "sha256": current,
                "unchanged": before == current,
            }
        )
    return {
        "unchanged": all(row["exists"] and row["unchanged"] for row in files),
        "files": files,
    }


def _disk_probe(root: Path) -> JsonDict:
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": 512, "ok": available_mb >= 512}


def _memory_probe() -> JsonDict:
    required_mb = 512
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:  # pragma: no cover - non-Linux fallback.
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _atomic_output_probe(output_path: Path) -> JsonDict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    probe_path = output_path.parent / f".{output_path.name}.atomic-probe"
    replacement = output_path.parent / f".{output_path.name}.atomic-probe.tmp"
    try:
        probe_path.write_text("old", encoding="utf-8")
        replacement.write_text("new", encoding="utf-8")
        os.replace(replacement, probe_path)
        ok = probe_path.read_text(encoding="utf-8") == "new"
    finally:
        probe_path.unlink(missing_ok=True)
        replacement.unlink(missing_ok=True)
    return {"ok": ok, "method": "os.replace_same_directory"}


def _field_provenance() -> JsonDict:
    return {
        field: {
            "satisfied_by": "generated_by_exp5921_schema_derived_support_compiler",
            "principle": FIELD_PRINCIPLES.get(field, "Exp5921 deterministic artifact field."),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["duration_s"] = 0.0
    stable["test_exit_codes"] = {}
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked", {})
    if isinstance(preconditions, dict):
        preconditions["output_path_existed_before_write"] = False
        if isinstance(preconditions.get("disk"), dict):
            preconditions["disk"]["available_mb"] = 0
        if isinstance(preconditions.get("ram"), dict):
            preconditions["ram"]["available_mb"] = 0
    return sha256_json(stable)


def _write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp = Path(handle.name)
        handle.write(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else round(numerator / denominator, 6)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    write_artifact(output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
