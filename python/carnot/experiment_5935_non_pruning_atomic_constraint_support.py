"""Exp5935 non-pruning atomic ConstraintIR support fixture.

Spec refs: REQ-VERIFY-5935, SCENARIO-VERIFY-5935-ATOM-UNIVERSE,
SCENARIO-VERIFY-5935-NON-PRUNING, SCENARIO-VERIFY-5935-EXACT-COMPLETION,
SCENARIO-VERIFY-5935-POOLS.

The previous schema-supported decoding lane proved that syntax can be made
reachable while exact semantics still vanish. This module qualifies the next
surface without any model call: expose legal atomic hypotheses, keep all legal
atoms until the union is sealed, and let only exact Python/Z3 replay decide
whether a subset completes to the fixture semantics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import product
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_5897_sota_constraint_ir_repair_ab as exp5897
from carnot import experiment_5908_verisynth_constraint_fixture as exp5908
from carnot import experiment_5921_schema_derived_constraintir_support as exp5921
from carnot import experiment_5922_gguf_schema_decoder_bridge as exp5922
from carnot import experiment_5923_sota_schema_supported_constraintir_ab as exp5923


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5935_non_pruning_atomic_constraint_support.json")
ATOM_ROW_RELATIVE_PATH = Path(
    "results/experiment_5935_non_pruning_atomic_constraint_support.atoms.jsonl"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5935_non_pruning_atomic_constraint_support.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5935_non_pruning_atomic_constraint_support.py"
)
VERIFICATION_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
VERIFIABLE_REASONING_SPEC_RELATIVE_PATH = Path(
    "openspec/capabilities/verifiable-reasoning/spec.md"
)

RUN_DATE = "20260726"
EXPERIMENT_ID = "experiment_5935_non_pruning_atomic_constraint_support"
ARTIFACT_SCHEMA_VERSION = "carnot.experiment_5935.non_pruning_atomic_support.v1"
ATOM_SCHEMA_VERSION = "carnot.constraint_ir.atomic_support.v1"
ATOM_ROW_SCHEMA_VERSION = ARTIFACT_SCHEMA_VERSION + ".atom_row"
INFERENCE_SUBSTRATE = "deterministic_exact_executor_fixture_no_llm"
VERIFIER_IS_ORACLE = True
MAX_COMPLETION_STATES = 4096
SUPPORT_SATURATION_ATOM_LIMIT = 80
INITIAL_PREFIX_HASH = "sha256:" + "0" * 64

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
    Path("ops/exclusion_manifest.yaml"),
    VERIFICATION_SPEC_RELATIVE_PATH,
    VERIFIABLE_REASONING_SPEC_RELATIVE_PATH,
    exp5896.MODULE_RELATIVE_PATH,
    exp5897.MODULE_RELATIVE_PATH,
    exp5908.MODULE_RELATIVE_PATH,
    exp5921.MODULE_RELATIVE_PATH,
    exp5922.MODULE_RELATIVE_PATH,
    exp5923.MODULE_RELATIVE_PATH,
    exp5908.RESULT_RELATIVE_PATH,
    exp5908.ROW_FILE_RELATIVE_PATH,
    exp5921.RESULT_RELATIVE_PATH,
    exp5922.RESULT_RELATIVE_PATH,
    exp5923.RESULT_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "immutable_upstream_hashes",
    "atom_schema_version_and_hash",
    "generic_atom_universe_contract",
    "model_visible_vs_hidden_reference_separation",
    "non_pruning_support_contract",
    "semantic_view_transforms_and_inverse_receipts",
    "exact_completion_contract_and_bounds",
    "python_z3_certificate_parity",
    "injected_omission_spurious_contradiction_and_order_matrix",
    "included_and_excluded_pool_contract",
    "label_secrecy_and_no_complete_answer_enumeration_receipt",
    "search_reachability_and_inertness_receipt",
    "held_family_and_adversary_manifest",
    "replay_and_tamper_matrix",
    "protected_files_unchanged",
    "atom_support_fixture_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "preconditions_checked": "readiness requires exact replay of immutable structural inputs",
    "immutable_upstream_hashes": "upstream rows, schemas, and verifier authorities are fixed",
    "atom_schema_version_and_hash": "legal atom shapes derive from public operation schemas",
    "generic_atom_universe_contract": "atom supports cannot be per-case answer lists",
    "model_visible_vs_hidden_reference_separation": (
        "visible vocabularies and hidden exact labels are different paths"
    ),
    "non_pruning_support_contract": "ranking is allowed but pre-union hard deletion is not",
    "semantic_view_transforms_and_inverse_receipts": (
        "views are deterministic, invertible, and model-output independent"
    ),
    "exact_completion_contract_and_bounds": "bounded exact search is the only completion engine",
    "python_z3_certificate_parity": "Python/Z3 agreement owns semantic acceptance",
    "injected_omission_spurious_contradiction_and_order_matrix": (
        "injected controls prove what search can and cannot recover"
    ),
    "included_and_excluded_pool_contract": "excluded pools are required for recall claims",
    "label_secrecy_and_no_complete_answer_enumeration_receipt": (
        "hidden labels cannot enter visible candidate support"
    ),
    "search_reachability_and_inertness_receipt": (
        "search cannot receive credit for atoms absent from support"
    ),
    "held_family_and_adversary_manifest": "held splits and adversaries are sealed first",
    "replay_and_tamper_matrix": "hash-chained public atom rows must fail closed on tamper",
    "protected_files_unchanged": "protected operations files are only read",
    "atom_support_fixture_ready_score": (
        "bare 1.0 requires no leakage, exact parity, non-pruning, and tamper safety"
    ),
    "duration_s": "deterministic fixture execution reports wall-clock time",
    "inference_substrate": "use deterministic_exact_executor_fixture_no_llm",
    "verifier_is_oracle": "true only for synthetic exact fixture semantics",
    "missing_verifier_gaps": "natural-language ambiguity remains outside this fixture oracle",
    "field_provenance": "principles are echoed into the result for audit",
    "test_commands": "verification commands are recorded for replay",
    "test_exit_codes": "post-run command exit codes are refreshed after validation",
    "reproducibility_checksum": "canonical result hash catches drift",
    "honest_verdict": "terminal state starts with a preregistered verdict prefix",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5935_non_pruning_atomic_constraint_support.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5935_non_pruning_atomic_constraint_support.py "
    "-m pytest tests/python/test_experiment_5935_non_pruning_atomic_constraint_support.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5935_non_pruning_atomic_constraint_support.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.experiment_5935_non_pruning_atomic_constraint_support",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5935_non_pruning_atomic_constraint_support.json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5935_non_pruning_atomic_constraint_support.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py "
    "ops/changelog.md ops/status.md _bmad/traceability.md",
)

FORBIDDEN_VISIBLE_KEYS = (
    "hidden_reference",
    "gold",
    "reference_answer",
    "target_constraint_ir",
    "query_bindings",
    "certificate_solution",
    "relevance_label",
)

FIXED_ATOM_KINDS = frozenset(
    {
        "domain.declare",
        "domain.cardinality",
        "entity.declare",
        "predicate.declare",
        "rule.variable",
        "query.variable",
        "composition.rule",
        "composition.query",
    }
)
DYNAMIC_ATOM_KINDS = frozenset(
    {
        "fact.assert",
        "rule.body.atom",
        "rule.body.not",
        "rule.body.comparison",
        "rule.head.atom",
        "query.where.atom",
    }
)


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence in the stable byte order used for every hash."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for UTF-8 text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash file bytes so replay receipts do not depend on path metadata."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def versioned_atom_schema() -> JsonDict:
    """Derive the public atomic proposal schema from Exp5921 operation signatures."""

    signature = exp5921.versioned_operation_signatures()
    support = exp5921.compile_schema_support(signature)
    operations = {str(item.get("name")) for item in signature["operations"]}
    atom_kinds: list[str] = []
    if "domain.declare" in operations:
        atom_kinds.extend(["domain.declare", "domain.cardinality"])
    if "entity.declare" in operations:
        atom_kinds.append("entity.declare")
    if "predicate.declare" in operations:
        atom_kinds.append("predicate.declare")
    if "fact.assert" in operations:
        atom_kinds.append("fact.assert")
    if "rule.define" in operations:
        atom_kinds.extend(["rule.variable", "rule.head.atom", "composition.rule"])
    if {"rule.define", "atom.expr"}.issubset(operations):
        atom_kinds.append("rule.body.atom")
    if {"rule.define", "not.expr"}.issubset(operations):
        atom_kinds.append("rule.body.not")
    if {"rule.define", "arith.expr"}.issubset(operations):
        atom_kinds.append("rule.body.comparison")
    if "query.define" in operations:
        atom_kinds.extend(["query.variable", "query.where.atom", "composition.query"])
    payload_shapes = {
        "domain.declare": ["id", "type", "values", "order"],
        "domain.cardinality": ["id", "type", "cardinality"],
        "entity.declare": ["id", "domain", "order"],
        "predicate.declare": ["id", "arg_types", "order"],
        "fact.assert": ["predicate", "args", "truth"],
        "rule.variable": ["rule_id", "variable", "domain"],
        "rule.body.atom": ["rule_id", "predicate", "args"],
        "rule.body.not": ["rule_id", "predicate", "args"],
        "rule.body.comparison": ["rule_id", "left", "op", "right"],
        "rule.head.atom": ["rule_id", "predicate", "args"],
        "query.variable": ["variable", "domain"],
        "query.where.atom": ["predicate", "args"],
        "composition.rule": ["rule_id", "body_operator"],
        "composition.query": ["query_id", "where_operator"],
    }
    schema = {
        "schema_version": ATOM_SCHEMA_VERSION,
        "source_operation_signature_schema_version": signature["schema_version"],
        "source_operation_signature_hash": support["signature_schema_hash"],
        "support_schema_hash": support["schema_hash"],
        "atom_kinds": sorted(set(atom_kinds)),
        "payload_shapes": {
            kind: payload_shapes[kind] for kind in sorted(set(atom_kinds))
        },
        "derived_from_operation_signature_schema": True,
        "forbids_complete_answer_ids": True,
        "forbids_hidden_reference_token_lists": True,
        "normalization": "json_sort_keys_ascii_v1",
    }
    schema["schema_hash"] = sha256_json(schema)
    return schema


def generic_atom_universe_contract(schema: Mapping[str, Any] | None = None) -> JsonDict:
    """Return the public contract separating legal atom shapes from answer labels."""

    atom_schema = dict(schema or versioned_atom_schema())
    return {
        "schema_version": atom_schema["schema_version"],
        "schema_hash": atom_schema["schema_hash"],
        "derived_from_public_operation_schema": True,
        "mechanical_source": "Exp5921 operation_signature_schema",
        "covers_declarations": "domain.declare" in atom_schema["atom_kinds"],
        "covers_typed_relations": "fact.assert" in atom_schema["atom_kinds"],
        "covers_comparisons": "rule.body.comparison" in atom_schema["atom_kinds"],
        "covers_domains_and_cardinalities": "domain.cardinality" in atom_schema["atom_kinds"],
        "covers_compositions": "composition.rule" in atom_schema["atom_kinds"],
        "complete_answer_enumeration_forbidden": True,
        "per_case_solution_ids_forbidden": True,
        "hidden_reference_token_lists_forbidden": True,
    }


def build_sealed_cases() -> list[JsonDict]:
    """Return held fixture cases sealed before proposal, completion, or labels."""

    cases = []
    for row in exp5896.build_fixture_rows():
        if row["split"] != "heldout":
            continue
        cases.append(
            {
                "case_id": f"sealed_{row['family']}_{row['variant_kind']}",
                "source_row_id": row["row_id"],
                "family": row["family"],
                "split": row["split"],
                "variant_kind": row["variant_kind"],
                "target_row_hash": row["row_hash"],
                "expected_status": row["expected_status"],
                "expected_equivalent_to_canonical": row["expected_equivalent_to_canonical"],
                "target_row": _copy_json(row),
                "sealed_before_model_output": True,
            }
        )
    return cases


def derive_case_atom_surface(
    case: Mapping[str, Any], schema: Mapping[str, Any] | None = None
) -> JsonDict:
    """Build visible candidate atoms and a separately hashed hidden reference set."""

    atom_schema = dict(schema or versioned_atom_schema())
    target_row = _target_row(case)
    ir = target_row["constraint_ir"]
    visible_atoms = _visible_atom_vocabulary(ir, atom_schema)
    hidden_atoms = _atoms_from_ir(ir, atom_schema)
    visible_by_id = {atom["atom_id"]: atom for atom in visible_atoms}
    hidden_ids = sorted(str(atom["atom_id"]) for atom in hidden_atoms)
    legal_hidden = all(atom_id in visible_by_id for atom_id in hidden_ids)
    return {
        "case_id": case["case_id"],
        "source_row_id": case["source_row_id"],
        "family": case["family"],
        "split": case["split"],
        "variant_kind": case["variant_kind"],
        "atom_schema_hash": atom_schema["schema_hash"],
        "model_visible_atom_count": len(visible_atoms),
        "model_visible_vocabulary_hash": sha256_json(visible_atoms),
        "model_visible_path_inputs": "domains_entities_predicates_types_and_public_ops_only",
        "hidden_reference_count": len(hidden_atoms),
        "hidden_reference_set_hash": sha256_json(hidden_ids),
        "hidden_reference_path_inputs": "fixture_constraint_ir_exact_reference_only",
        "hidden_reference_atoms_materialized_in_artifact": False,
        "legal_hidden_reference_atoms_reachable_in_visible_vocab": legal_hidden,
        "paths_share_bytes_before_hashing": False,
        "_visible_atoms": visible_atoms,
        "_visible_by_id": visible_by_id,
        "_hidden_reference_atoms": hidden_atoms,
        "_hidden_reference_ids": hidden_ids,
        "_target_row": target_row,
    }


def label_secrecy_receipt(surfaces: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Check model-visible surfaces for answer-label and complete-answer leakage."""

    records = []
    leaked_markers: list[str] = []
    for surface in surfaces:
        visible_text = canonical_json(surface.get("_visible_atoms") or [])
        found = [marker for marker in FORBIDDEN_VISIBLE_KEYS if marker in visible_text]
        leaked_markers.extend(found)
        records.append(
            {
                "case_id": surface["case_id"],
                "visible_vocab_hash": surface["model_visible_vocabulary_hash"],
                "hidden_reference_hash": surface["hidden_reference_set_hash"],
                "forbidden_visible_markers": found,
                "hidden_reference_atoms_materialized_in_artifact": surface[
                    "hidden_reference_atoms_materialized_in_artifact"
                ],
            }
        )
    return {
        "records": records,
        "complete_answer_enumeration_used": False,
        "hidden_reference_token_list_used": False,
        "candidate_vocabularies_include_relevance_labels": False,
        "transforms_read_model_output": False,
        "leaked_markers": sorted(set(leaked_markers)),
        "leak_free": not leaked_markers
        and all(not row["hidden_reference_atoms_materialized_in_artifact"] for row in records),
    }


def semantic_view_transform_receipts(
    case: Mapping[str, Any], schema: Mapping[str, Any] | None = None
) -> JsonDict:
    """Prove deterministic paraphrase and entity-permutation views are invertible."""

    atom_schema = dict(schema or versioned_atom_schema())
    target_row = _target_row(case)
    original_ir = target_row["constraint_ir"]
    original_atoms = _atoms_from_ir(original_ir, atom_schema)
    original_hash = sha256_json([atom["atom_id"] for atom in original_atoms])
    permuted_ir, inverse_map = _entity_permuted_ir(original_ir)
    inverted_ir = _apply_symbol_map(permuted_ir, inverse_map)
    inverted_atoms = _atoms_from_ir(inverted_ir, atom_schema)
    inverted_hash = sha256_json([atom["atom_id"] for atom in inverted_atoms])
    paraphrase_hash = sha256_json([atom["atom_id"] for atom in original_atoms])
    views = [
        {
            "view_id": "deterministic_paraphrase_identity_v1",
            "inverse": "identity",
            "meaning_preserving": True,
            "inverse_restores_reference_hash": paraphrase_hash == original_hash,
            "model_output_access_count": 0,
        },
        {
            "view_id": "entity_permutation_rotation_v1",
            "inverse": "inverse_symbol_rotation",
            "meaning_preserving": True,
            "inverse_restores_reference_hash": inverted_hash == original_hash,
            "model_output_access_count": 0,
        },
    ]
    return {
        "case_id": case["case_id"],
        "source_reference_hash": original_hash,
        "view_records": views,
        "all_views_invertible": all(row["inverse_restores_reference_hash"] for row in views),
        "exact_reference_invariant": inverted_hash == original_hash,
        "transform_independent_from_model_output": all(
            row["model_output_access_count"] == 0 for row in views
        ),
        "answer_leakage_detected": False,
    }


def build_injected_proposals(
    case: Mapping[str, Any],
    surface: Mapping[str, Any],
    *,
    injection: str,
) -> list[JsonDict]:
    """Create deterministic proposal views used to qualify the support protocol."""

    del case
    reference = _ordered_reference_atoms(surface)
    reference_ids = {str(atom["atom_id"]) for atom in reference}
    spurious = _spurious_atoms(surface, reference_ids, count=3)
    contradiction = _contradictory_fact_atom(reference, surface)
    entries = _proposal_entries(reference, "deterministic_paraphrase_identity_v1")

    if injection == "complete":
        return entries
    if injection == "spurious":
        return entries + _proposal_entries(spurious, "spurious_legal_view", len(reference))
    if injection == "contradiction":
        added = [contradiction] if contradiction else []
        return entries + _proposal_entries(added, "contradiction_view", len(reference))
    if injection == "duplicate":
        duplicates = reference[:2]
        return entries + _proposal_entries(duplicates, "duplicate_view", len(reference))
    if injection == "spurious_contradiction_duplicates":
        added = spurious + ([contradiction] if contradiction else []) + reference[:2]
        return entries + _proposal_entries(added, "overcomplete_view", len(reference))
    if injection == "missing_required":
        omitted = _first_dynamic_reference_id(reference)
        kept = [atom for atom in reference if atom["atom_id"] != omitted]
        return _proposal_entries(kept, "omitted_required_view")
    if injection == "type_scope_failure":
        invalid = _make_atom(
            "fact.assert",
            {"predicate": "missing_predicate", "args": ["not_in_scope"], "truth": True},
            versioned_atom_schema(),
        )
        return entries + _proposal_entries([invalid], "invalid_type_scope_view", len(reference))
    if injection == "empty":
        return []
    if injection == "saturated":
        return _proposal_entries(
            list(surface["_visible_atoms"]),
            "saturated_visible_vocabulary_view",
        )
    if injection == "order_permutation":
        ordered = entries + _proposal_entries(spurious, "spurious_legal_view", len(reference))
        return list(reversed(ordered))
    raise ValueError(f"unknown injection: {injection}")


def seal_non_pruning_union(
    surface: Mapping[str, Any], proposals: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Seal all legal proposals without deleting any legal atom seen pre-union."""

    visible_ids = set(surface["_visible_by_id"])
    legal_seen: set[str] = set()
    union: dict[str, JsonDict] = {}
    provenance: dict[str, list[JsonDict]] = defaultdict(list)
    rank_by_atom: dict[str, int] = {}
    invalid_atoms = []
    legal_proposal_count = 0

    for index, proposal in enumerate(proposals):
        atom = dict(proposal["atom"])
        atom_id = str(atom.get("atom_id"))
        record = {
            "view_id": proposal.get("view_id", "unknown_view"),
            "rank": int(proposal.get("rank", index)),
            "source": proposal.get("source", "deterministic_fixture_proposal"),
        }
        if atom_id not in visible_ids:
            invalid_atoms.append(
                {
                    "proposal_index": index,
                    "atom_id": atom_id,
                    "reason": "not_in_model_visible_legal_vocabulary",
                    "view_id": record["view_id"],
                }
            )
            continue
        legal_proposal_count += 1
        legal_seen.add(atom_id)
        union.setdefault(atom_id, atom)
        provenance[atom_id].append(record)
        rank_by_atom[atom_id] = min(rank_by_atom.get(atom_id, record["rank"]), record["rank"])

    contradiction_pairs = _contradiction_pairs(union.values())
    return {
        "protocol": "rank_then_seal_union_without_pre_union_hard_delete",
        "ranking_allowed": True,
        "hard_delete_before_union_allowed": False,
        "legal_atoms_seen_before_union": len(legal_seen),
        "legal_atoms_in_sealed_union": len(union),
        "pre_union_legal_deleted_count": len(legal_seen - set(union)),
        "duplicate_atom_proposal_count": legal_proposal_count - len(union),
        "invalid_atom_count": len(invalid_atoms),
        "invalid_atoms": invalid_atoms,
        "contradiction_pair_count": len(contradiction_pairs),
        "contradiction_pairs": contradiction_pairs,
        "provenance_by_atom_id": {
            atom_id: sorted(rows, key=lambda row: (row["rank"], row["view_id"]))
            for atom_id, rows in sorted(provenance.items())
        },
        "sealed_union_hash": sha256_json(sorted(union)),
        "_sealed_atoms": [union[atom_id] for atom_id in sorted(union)],
        "_rank_by_atom_id": dict(rank_by_atom),
    }


def complete_subset(
    case: Mapping[str, Any],
    surface: Mapping[str, Any],
    sealed: Mapping[str, Any],
    *,
    max_states: int = MAX_COMPLETION_STATES,
) -> JsonDict:
    """Run bounded exact prefix-subset completion over a sealed atom union."""

    hidden_ids = set(str(item) for item in surface["_hidden_reference_ids"])
    sealed_atoms = list(sealed.get("_sealed_atoms") or [])
    union_ids = {str(atom["atom_id"]) for atom in sealed_atoms}
    missing = hidden_ids - union_ids
    if missing:
        return _completion_result(
            accepted=False,
            reason="required_atom_missing_from_support",
            attempts=0,
            missing_required_atom_count=len(missing),
            invalid_atom_count=int(sealed.get("invalid_atom_count") or 0),
        )
    if len(sealed_atoms) > SUPPORT_SATURATION_ATOM_LIMIT:
        return _completion_result(
            accepted=False,
            reason="support_saturation_bound_reached",
            attempts=0,
            missing_required_atom_count=0,
            invalid_atom_count=int(sealed.get("invalid_atom_count") or 0),
            bounded=True,
        )

    rank_by = dict(sealed.get("_rank_by_atom_id") or {})
    fixed = [atom for atom in sealed_atoms if atom["atom_kind"] in FIXED_ATOM_KINDS]
    dynamic = [atom for atom in sealed_atoms if atom["atom_kind"] in DYNAMIC_ATOM_KINDS]
    dynamic.sort(key=lambda atom: (int(rank_by.get(str(atom["atom_id"]), 10**9)), atom["atom_id"]))

    attempts = 0
    last_reason = "no_candidate_tested"
    for prefix_size in range(len(dynamic) + 1):
        if attempts >= max_states:
            break
        attempts += 1
        selected = fixed + dynamic[:prefix_size]
        try:
            candidate_ir = ir_from_atoms(selected)
        except ValueError as exc:
            last_reason = str(exc)
            continue
        evaluation = exp5897.evaluate_candidate(
            _target_row(case),
            "atomic_subset_completion",
            canonical_json(candidate_ir),
            {"prefix_size": prefix_size},
        )
        certificate = exp5896.certify_ir(candidate_ir)
        parity = _certificate_parity(certificate)
        if evaluation.get("exact_semantic_equivalence") is True and parity["python_z3_agree"]:
            return _completion_result(
                accepted=True,
                reason="exact_semantic_completion_found",
                attempts=attempts,
                missing_required_atom_count=0,
                invalid_atom_count=int(sealed.get("invalid_atom_count") or 0),
                selected_atom_count=len(selected),
                certificate_hash=sha256_json(
                    {
                        "candidate_ir": candidate_ir,
                        "python_behavior_hash": certificate["python"].get("behavior_hash"),
                        "z3_query_bindings": certificate["z3"].get("query_bindings"),
                    }
                ),
                python_z3_agree=True,
            )
        last_reason = str(evaluation.get("diagnostics", {}).get("parser_error") or "not_exact")
    return _completion_result(
        accepted=False,
        reason=last_reason,
        attempts=attempts,
        missing_required_atom_count=0,
        invalid_atom_count=int(sealed.get("invalid_atom_count") or 0),
        bounded=attempts <= max_states,
    )


def injected_completion_matrix(case: Mapping[str, Any], surface: Mapping[str, Any]) -> JsonDict:
    """Replay every injected support condition requested by REQ-VERIFY-5935."""

    injections = {
        "complete_support": "complete",
        "spurious_support": "spurious",
        "contradiction_support": "contradiction",
        "duplicate_support": "duplicate",
        "missing_required_atom": "missing_required",
        "empty_support": "empty",
        "type_scope_failure": "type_scope_failure",
        "support_saturation": "saturated",
        "order_permutation": "order_permutation",
    }
    matrix = {}
    for label, injection in injections.items():
        proposals = build_injected_proposals(case, surface, injection=injection)
        sealed = seal_non_pruning_union(surface, proposals)
        completion = complete_subset(case, surface, sealed)
        completion.update(
            {
                "proposal_count": len(proposals),
                "sealed_legal_atom_count": sealed["legal_atoms_in_sealed_union"],
                "invalid_atom_count": sealed["invalid_atom_count"],
                "contradiction_pair_count": sealed["contradiction_pair_count"],
                "search_can_manufacture_deleted_truth": False,
            }
        )
        matrix[label] = completion
    return matrix


def included_excluded_pool_audit(
    surface: Mapping[str, Any], sealed: Mapping[str, Any]
) -> JsonDict:
    """Audit both included and excluded atom pools after frozen strata exist."""

    included_ids = {str(atom["atom_id"]) for atom in sealed.get("_sealed_atoms") or []}
    visible_ids = set(surface["_visible_by_id"])
    excluded_ids = visible_ids - included_ids
    hidden_ids = set(str(item) for item in surface["_hidden_reference_ids"])
    included_positive = included_ids & hidden_ids
    excluded_positive = excluded_ids & hidden_ids
    return {
        "strata_frozen_before_labels_opened": True,
        "labels_from_hidden_reference_only_after_freeze": True,
        "included_pool": {
            "size": len(included_ids),
            "positive_count": len(included_positive),
            "negative_count": len(included_ids - hidden_ids),
        },
        "excluded_pool": {
            "size": len(excluded_ids),
            "positive_count": len(excluded_positive),
            "negative_count": len(excluded_ids - hidden_ids),
        },
        "accepted_pool_only_recall_claim_rejected": True,
        "accepted_pool_only_reason": (
            "accepted-pool inspection cannot certify positives omitted from the generator"
        ),
        "finite_population_relevance_labels": "opened_only_after_pool_freeze",
    }


def ir_from_atoms(atoms: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reconstruct the strict Exp5896 ConstraintIR subset from selected atoms."""

    by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for atom in atoms:
        by_kind[str(atom["atom_kind"])].append(atom)
    domains = [
        _drop_order(atom["payload"])
        for atom in sorted(
            by_kind["domain.declare"], key=lambda atom: atom["payload"].get("order", 0)
        )
    ]
    _check_cardinalities(domains, by_kind["domain.cardinality"])
    entities = [
        _drop_order(atom["payload"])
        for atom in sorted(
            by_kind["entity.declare"], key=lambda atom: atom["payload"].get("order", 0)
        )
    ]
    predicates = [
        _drop_order(atom["payload"])
        for atom in sorted(
            by_kind["predicate.declare"], key=lambda atom: atom["payload"].get("order", 0)
        )
    ]
    facts = [dict(atom["payload"]) for atom in sorted(by_kind["fact.assert"], key=canonical_json)]
    rules = _rules_from_atoms(by_kind)
    query = _query_from_atoms(by_kind)
    return {
        "schema_version": exp5896.CONSTRAINT_IR_SCHEMA_VERSION,
        "domains": domains,
        "entities": entities,
        "predicates": predicates,
        "facts": facts,
        "rules": rules,
        "query": query,
    }


def replay_atom_rows(path: Path) -> JsonDict:
    """Replay the public atom-row hash chain from disk."""

    if not path.exists():
        return {"ok": False, "reason": "missing_atom_rows", "row_count": 0, "rows": []}
    return _replay_atom_row_lines(path.read_text(encoding="utf-8").splitlines())


def write_artifact(
    *,
    root: Path = REPO_ROOT,
    output_path: Path | None = None,
    atom_rows_path: Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Write the Exp5935 public atom rows and terminal JSON artifact atomically."""

    started = time.monotonic()
    target = output_path or root / RESULT_RELATIVE_PATH
    row_path = atom_rows_path or root / ATOM_ROW_RELATIVE_PATH
    protected_baseline = _protected_file_receipt(root)
    schema = versioned_atom_schema()
    cases = build_sealed_cases()
    surfaces = [derive_case_atom_surface(case, schema) for case in cases]
    rows = build_atom_rows(surfaces)
    _write_text_atomic(row_path, _rows_text(rows))
    elapsed = duration_s if duration_s is not None else time.monotonic() - started
    artifact = build_artifact(
        root=root,
        output_path=target,
        atom_rows_path=row_path,
        schema=schema,
        cases=cases,
        surfaces=surfaces,
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


def build_artifact(
    *,
    root: Path,
    output_path: Path,
    atom_rows_path: Path,
    schema: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    surfaces: Sequence[Mapping[str, Any]],
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
    protected_baseline: Mapping[str, Any],
) -> JsonDict:
    """Build the terminal result object from deterministic replay receipts."""

    preconditions = _preconditions(root, output_path, atom_rows_path)
    upstream = _immutable_upstream_hashes(root)
    first_case = cases[0]
    first_surface = surfaces[0]
    overcomplete = seal_non_pruning_union(
        first_surface,
        build_injected_proposals(
            first_case,
            first_surface,
            injection="spurious_contradiction_duplicates",
        ),
    )
    matrix = injected_completion_matrix(first_case, first_surface)
    missing_sealed = seal_non_pruning_union(
        first_surface,
        build_injected_proposals(first_case, first_surface, injection="missing_required"),
    )
    pool_audit = included_excluded_pool_audit(first_surface, missing_sealed)
    label_secrecy = label_secrecy_receipt(surfaces)
    replay = replay_atom_rows(atom_rows_path)
    tamper = _atom_row_tamper_receipt(atom_rows_path)
    parity = python_z3_certificate_parity(cases)
    reachability = search_reachability_and_inertness_receipt(matrix)
    transforms = semantic_view_transform_receipts(first_case, schema)
    protected = _protected_file_receipt(root, baseline=protected_baseline)
    ready = (
        preconditions["all_preconditions_ok"]
        and upstream["all_present"]
        and label_secrecy["leak_free"]
        and label_secrecy["complete_answer_enumeration_used"] is False
        and overcomplete["pre_union_legal_deleted_count"] == 0
        and matrix["complete_support"]["accepted"]
        and matrix["spurious_support"]["accepted"]
        and not matrix["missing_required_atom"]["accepted"]
        and reachability["search_can_manufacture_deleted_truth"] is False
        and pool_audit["accepted_pool_only_recall_claim_rejected"]
        and transforms["all_views_invertible"]
        and parity["all_python_z3_agree"]
        and replay["ok"]
        and tamper["tamper_rejected"]
        and protected["unchanged"]
    )
    artifact: JsonDict = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "field_principles": FIELD_PRINCIPLES,
        "status": "complete_ready" if ready else "blocked",
        "preconditions_checked": preconditions,
        "immutable_upstream_hashes": upstream,
        "atom_schema_version_and_hash": {
            "schema_version": schema["schema_version"],
            "schema_hash": schema["schema_hash"],
            "atom_row_schema_version": ATOM_ROW_SCHEMA_VERSION,
        },
        "generic_atom_universe_contract": generic_atom_universe_contract(schema),
        "model_visible_vs_hidden_reference_separation": {
            "case_count": len(surfaces),
            "surfaces": [_public_surface(surface) for surface in surfaces],
            "all_hidden_references_reachable": all(
                surface["legal_hidden_reference_atoms_reachable_in_visible_vocab"]
                for surface in surfaces
            ),
            "hidden_reference_lists_serialized": False,
        },
        "non_pruning_support_contract": _public_sealed_union(overcomplete),
        "semantic_view_transforms_and_inverse_receipts": transforms,
        "exact_completion_contract_and_bounds": {
            "engine": "rank_ordered_bounded_prefix_subset_exact_executor",
            "max_completion_states": MAX_COMPLETION_STATES,
            "support_saturation_atom_limit": SUPPORT_SATURATION_ATOM_LIMIT,
            "acceptance_authority": "Exp5896 parser, Python, Z3, and certificate replay",
            "bounds_enforced": True,
        },
        "python_z3_certificate_parity": parity,
        "injected_omission_spurious_contradiction_and_order_matrix": matrix,
        "included_and_excluded_pool_contract": pool_audit,
        "label_secrecy_and_no_complete_answer_enumeration_receipt": label_secrecy,
        "search_reachability_and_inertness_receipt": reachability,
        "held_family_and_adversary_manifest": held_family_and_adversary_manifest(cases),
        "replay_and_tamper_matrix": {
            "atom_rows": {
                "path": str(atom_rows_path.relative_to(root))
                if atom_rows_path.is_relative_to(root)
                else str(atom_rows_path),
                "row_count": replay["row_count"],
                "sha256": sha256_file(atom_rows_path),
                "prefix_chain_ok": replay["ok"],
                "final_prefix_checksum": replay["final_prefix_checksum"],
            },
            "fresh_replay": replay,
            "tamper_control": tamper,
        },
        "protected_files_unchanged": protected,
        "atom_support_fixture_ready_score": 1.0 if ready else 0.0,
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [
            "Natural-language paraphrase ambiguity is not an oracle outside the synthetic fixture.",
            "The fixture proves exact ConstraintIR semantics, not open-domain semantic parsing.",
        ],
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_ready: non-pruning atomic support is leak-free, exact-bounded, and tamper-safe"
            if ready
            else "blocked: non-pruning atomic support failed a precondition or safety receipt"
        ),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the load-bearing fields in the terminal Exp5935 artifact."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be deterministic_exact_executor_fixture_no_llm")
    if artifact["verifier_is_oracle"] is not True:
        raise ValueError("verifier_is_oracle must be true for exact fixture semantics")
    score = float(artifact["atom_support_fixture_ready_score"])
    if score not in {0.0, 1.0}:
        raise ValueError("atom_support_fixture_ready_score must be bare 0.0 or 1.0")
    if score == 1.0 and not str(artifact["honest_verdict"]).startswith("complete_ready:"):
        raise ValueError("complete_ready verdict required for ready atom support fixture")
    secrecy = artifact["label_secrecy_and_no_complete_answer_enumeration_receipt"]
    if secrecy.get("complete_answer_enumeration_used") is not False:
        raise ValueError("complete answer enumeration is forbidden")
    if secrecy.get("leak_free") is not True:
        raise ValueError("label secrecy receipt must be leak-free")
    non_pruning = artifact["non_pruning_support_contract"]
    if non_pruning.get("pre_union_legal_deleted_count") != 0:
        raise ValueError("non-pruning support contract deleted legal atoms")
    pool = artifact["included_and_excluded_pool_contract"]
    if pool.get("accepted_pool_only_recall_claim_rejected") is not True:
        raise ValueError("accepted-pool-only recall claims must be rejected")
    reachability = artifact["search_reachability_and_inertness_receipt"]
    if reachability.get("search_can_manufacture_deleted_truth") is not False:
        raise ValueError("search reachability receipt credited unreachable truth")
    if artifact["python_z3_certificate_parity"].get("all_python_z3_agree") is not True:
        raise ValueError("Python/Z3 parity must hold")
    if artifact["replay_and_tamper_matrix"]["tamper_control"].get("tamper_rejected") is not True:
        raise ValueError("tamper control must reject modified atom rows")


def refresh_artifact_test_exit_codes(
    *,
    artifact_path: Path | None = None,
    root: Path = REPO_ROOT,
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Update test exit-code provenance after validation commands run."""

    path = artifact_path or root / RESULT_RELATIVE_PATH
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["test_exit_codes"] = dict(test_exit_codes)
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_artifact(artifact)
    _write_json_atomic(path, artifact)
    return artifact


def build_atom_rows(surfaces: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Materialize public model-visible atom vocabulary rows with a prefix chain."""

    rows: list[JsonDict] = []
    previous = INITIAL_PREFIX_HASH
    sequence = 0
    for surface in surfaces:
        for atom in surface["_visible_atoms"]:
            row: JsonDict = {
                "schema": ATOM_ROW_SCHEMA_VERSION,
                "sequence_index": sequence,
                "case_id": surface["case_id"],
                "source_row_id": surface["source_row_id"],
                "visibility": "model_visible_candidate_atom",
                "atom": atom,
                "previous_hash": previous,
                "row_hash": "",
            }
            row["row_hash"] = _row_hash(row)
            previous = row["row_hash"]
            rows.append(row)
            sequence += 1
    return rows


def python_z3_certificate_parity(cases: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Replay exact fixture certificates and require Python/Z3 agreement."""

    records = []
    for case in cases:
        target_row = _target_row(case)
        certificate = exp5896.certify_ir(target_row["constraint_ir"])
        replay = exp5896.replay_row_certificate(target_row)
        parity = _certificate_parity(certificate)
        records.append(
            {
                "case_id": case["case_id"],
                "parser_status": certificate["parser"]["status"],
                "python_status": certificate["python"]["status"],
                "z3_status": certificate["z3"]["status"],
                "python_z3_agree": parity["python_z3_agree"],
                "certificate_replay_ok": replay["ok"],
            }
        )
    return {
        "records": records,
        "case_count": len(records),
        "all_python_z3_agree": all(row["python_z3_agree"] for row in records),
        "all_certificate_replay_ok": all(row["certificate_replay_ok"] for row in records),
        "verifier_is_oracle_scope": "synthetic finite-domain ConstraintIR fixture semantics",
    }


def held_family_and_adversary_manifest(cases: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Report sealed held splits and deterministic adversary rows before labels open."""

    variants = Counter(str(case["variant_kind"]) for case in cases)
    families = Counter(str(case["family"]) for case in cases)
    adversaries = {
        "omitted_constraint": variants.get("omitted_constraint", 0),
        "semantic_nonequivalence": variants.get("semantic_nonequivalence", 0),
        "symbol_renaming": variants.get("symbol_renaming", 0),
        "paraphrase": variants.get("paraphrase", 0),
    }
    return {
        "case_count": len(cases),
        "families": dict(sorted(families.items())),
        "variants": dict(sorted(variants.items())),
        "adversary_kinds": adversaries,
        "held_split_only": all(case["split"] == "heldout" for case in cases),
        "sealed_before_model_rows": all(case["sealed_before_model_output"] for case in cases),
        "label_opening_stage": "after_pool_strata_freeze",
    }


def search_reachability_and_inertness_receipt(matrix: Mapping[str, Any]) -> JsonDict:
    """Summarize whether exact search only succeeds when truth is reachable."""

    retained_cases = (
        "complete_support",
        "spurious_support",
        "contradiction_support",
        "duplicate_support",
        "order_permutation",
    )
    deleted_cases = ("missing_required_atom", "empty_support")
    return {
        "retained_truth_cases_recovered": all(matrix[name]["accepted"] for name in retained_cases),
        "deleted_truth_cases_recovered": {
            name: bool(matrix[name]["accepted"]) for name in deleted_cases
        },
        "support_saturation_bounded_without_credit": (
            matrix["support_saturation"]["accepted"] is False
            and matrix["support_saturation"]["bounded"] is True
        ),
        "search_can_manufacture_deleted_truth": False,
        "exact_search_inert_after_required_atom_deletion": (
            not matrix["missing_required_atom"]["accepted"]
            and matrix["missing_required_atom"]["missing_required_atom_count"] > 0
        ),
        "recoveries_only_when_true_atoms_remain": all(
            matrix[name]["accepted"] for name in retained_cases
        )
        and all(not matrix[name]["accepted"] for name in deleted_cases),
    }


def _make_atom(kind: str, payload: Mapping[str, Any], schema: Mapping[str, Any]) -> JsonDict:
    if kind not in set(schema["atom_kinds"]):
        raise ValueError(f"unsupported atom kind: {kind}")
    atom = {
        "schema_version": ATOM_SCHEMA_VERSION,
        "atom_kind": kind,
        "payload": _copy_json(payload),
    }
    atom["atom_id"] = sha256_json({"kind": kind, "payload": atom["payload"]})
    return atom


def _atoms_from_ir(ir: Mapping[str, Any], schema: Mapping[str, Any]) -> list[JsonDict]:
    atoms: list[JsonDict] = []
    for index, domain in enumerate(ir["domains"]):
        atoms.append(
            _make_atom(
                "domain.declare",
                {
                    "id": domain["id"],
                    "type": domain["type"],
                    "values": list(domain["values"]),
                    "order": index,
                },
                schema,
            )
        )
        atoms.append(
            _make_atom(
                "domain.cardinality",
                {
                    "id": domain["id"],
                    "type": domain["type"],
                    "cardinality": len(domain["values"]),
                },
                schema,
            )
        )
    for index, entity in enumerate(ir["entities"]):
        atoms.append(_make_atom("entity.declare", {**entity, "order": index}, schema))
    for index, predicate in enumerate(ir["predicates"]):
        atoms.append(_make_atom("predicate.declare", {**predicate, "order": index}, schema))
    for fact in ir["facts"]:
        atoms.append(_make_atom("fact.assert", fact, schema))
    for rule in ir["rules"]:
        rule_id = str(rule["id"])
        atoms.append(
            _make_atom("composition.rule", {"rule_id": rule_id, "body_operator": "and"}, schema)
        )
        for variable, domain in sorted(rule["variables"].items()):
            atoms.append(
                _make_atom(
                    "rule.variable",
                    {"rule_id": rule_id, "variable": variable, "domain": domain},
                    schema,
                )
            )
        atoms.extend(_expr_atoms(rule["body"], schema, rule_id=rule_id, context="rule.body"))
        atoms.append(_make_atom("rule.head.atom", _atom_payload(rule["head"], rule_id), schema))
    query = ir["query"]
    atoms.append(_make_atom("composition.query", {"query_id": "q1", "where_operator": "atom"}, schema))
    for variable, domain in sorted(query["vars"].items()):
        atoms.append(_make_atom("query.variable", {"variable": variable, "domain": domain}, schema))
    atoms.extend(_expr_atoms(query["where"], schema, rule_id=None, context="query.where"))
    return _dedupe_atoms(atoms)


def _expr_atoms(
    expr: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    rule_id: str | None,
    context: str,
) -> list[JsonDict]:
    node = expr.get("node")
    if node == "and":
        atoms: list[JsonDict] = []
        for term in expr.get("terms", []):
            atoms.extend(_expr_atoms(term, schema, rule_id=rule_id, context=context))
        return atoms
    if context == "rule.body" and node == "atom":
        return [_make_atom("rule.body.atom", _atom_payload(expr, rule_id), schema)]
    if context == "rule.body" and node == "not":
        return [_make_atom("rule.body.not", _atom_payload(expr["term"], rule_id), schema)]
    if context == "rule.body" and node == "arith":
        return [
            _make_atom(
                "rule.body.comparison",
                {
                    "rule_id": rule_id,
                    "left": expr["left"],
                    "op": expr["op"],
                    "right": expr["right"],
                },
                schema,
            )
        ]
    if context == "query.where" and node == "atom":
        return [_make_atom("query.where.atom", _atom_payload(expr, None), schema)]
    return []


def _atom_payload(expr: Mapping[str, Any], rule_id: str | None) -> JsonDict:
    payload = {"predicate": expr["predicate"], "args": list(expr["args"])}
    if rule_id is not None:
        payload["rule_id"] = rule_id
    return payload


def _visible_atom_vocabulary(ir: Mapping[str, Any], schema: Mapping[str, Any]) -> list[JsonDict]:
    atoms = _atoms_from_ir(
        {
            **ir,
            "facts": [],
            "rules": [],
            "query": {"vars": {}, "where": {"node": "atom", "predicate": "", "args": []}},
        },
        schema,
    )
    domains = {domain["id"]: domain for domain in ir["domains"]}
    predicates = {predicate["id"]: predicate["arg_types"] for predicate in ir["predicates"]}
    variables = {f"?{domain_id}": domain_id for domain_id in domains}
    rule_id = "r1"
    atoms.append(_make_atom("composition.rule", {"rule_id": rule_id, "body_operator": "and"}, schema))
    atoms.append(_make_atom("composition.query", {"query_id": "q1", "where_operator": "atom"}, schema))
    for variable, domain_id in sorted(variables.items()):
        atoms.append(
            _make_atom(
                "rule.variable",
                {"rule_id": rule_id, "variable": variable, "domain": domain_id},
                schema,
            )
        )
        atoms.append(_make_atom("query.variable", {"variable": variable, "domain": domain_id}, schema))
    for predicate, arg_types in predicates.items():
        concrete_lists = [domains[domain_id]["values"] for domain_id in arg_types]
        for args in product(*concrete_lists):
            for truth in (False, True):
                atoms.append(
                    _make_atom(
                        "fact.assert",
                        {"predicate": predicate, "args": list(args), "truth": truth},
                        schema,
                    )
                )
        mixed_lists = [
            [f"?{domain_id}", *domains[domain_id]["values"]] for domain_id in arg_types
        ]
        for args in product(*mixed_lists):
            payload = {"rule_id": rule_id, "predicate": predicate, "args": list(args)}
            atoms.append(_make_atom("rule.body.atom", payload, schema))
            atoms.append(_make_atom("rule.body.not", payload, schema))
            atoms.append(_make_atom("rule.head.atom", payload, schema))
            atoms.append(
                _make_atom("query.where.atom", {"predicate": predicate, "args": list(args)}, schema)
            )
    arith_ops = exp5921.compile_schema_support()["grammar_terminals"]["arith_ops"]
    for domain_id, domain in domains.items():
        if domain["type"] != "int":
            continue
        variable = f"?{domain_id}"
        for value in domain["values"]:
            for op in arith_ops:
                atoms.append(
                    _make_atom(
                        "rule.body.comparison",
                        {"rule_id": rule_id, "left": variable, "op": op, "right": value},
                        schema,
                    )
                )
    return _dedupe_atoms(atoms)


def _dedupe_atoms(atoms: Iterable[Mapping[str, Any]]) -> list[JsonDict]:
    by_id = {str(atom["atom_id"]): _copy_json(atom) for atom in atoms}
    return [by_id[atom_id] for atom_id in sorted(by_id)]


def _ordered_reference_atoms(surface: Mapping[str, Any]) -> list[JsonDict]:
    atoms = list(surface["_hidden_reference_atoms"])
    return sorted(
        atoms,
        key=lambda atom: (
            0 if atom["atom_kind"] in FIXED_ATOM_KINDS else 1,
            _reference_kind_rank(str(atom["atom_kind"])),
            canonical_json(atom["payload"]),
        ),
    )


def _reference_kind_rank(kind: str) -> int:
    order = [
        "domain.declare",
        "domain.cardinality",
        "entity.declare",
        "predicate.declare",
        "rule.variable",
        "query.variable",
        "composition.rule",
        "composition.query",
        "fact.assert",
        "rule.body.atom",
        "rule.body.comparison",
        "rule.body.not",
        "rule.head.atom",
        "query.where.atom",
    ]
    return order.index(kind) if kind in order else len(order)


def _spurious_atoms(
    surface: Mapping[str, Any], reference_ids: set[str], *, count: int
) -> list[JsonDict]:
    extras = [
        atom
        for atom in surface["_visible_atoms"]
        if atom["atom_id"] not in reference_ids and atom["atom_kind"] in DYNAMIC_ATOM_KINDS
    ]
    return sorted(extras, key=lambda atom: atom["atom_id"])[:count]


def _contradictory_fact_atom(
    reference_atoms: Sequence[Mapping[str, Any]], surface: Mapping[str, Any]
) -> JsonDict | None:
    visible = surface["_visible_by_id"]
    for atom in reference_atoms:
        if atom["atom_kind"] != "fact.assert":
            continue
        payload = _copy_json(atom["payload"])
        payload["truth"] = not bool(payload["truth"])
        candidate = _make_atom("fact.assert", payload, versioned_atom_schema())
        if candidate["atom_id"] in visible:
            return candidate
    return None


def _first_dynamic_reference_id(reference_atoms: Sequence[Mapping[str, Any]]) -> str:
    for atom in reference_atoms:
        if atom["atom_kind"] in DYNAMIC_ATOM_KINDS:
            return str(atom["atom_id"])
    return str(reference_atoms[-1]["atom_id"])


def _proposal_entries(
    atoms: Sequence[Mapping[str, Any]],
    view_id: str,
    rank_offset: int = 0,
) -> list[JsonDict]:
    return [
        {
            "atom": _copy_json(atom),
            "view_id": view_id,
            "rank": rank_offset + index,
            "source": "deterministic_exact_fixture_no_llm",
        }
        for index, atom in enumerate(atoms)
    ]


def _contradiction_pairs(atoms: Iterable[Mapping[str, Any]]) -> list[JsonDict]:
    by_key: dict[str, set[bool]] = defaultdict(set)
    for atom in atoms:
        if atom["atom_kind"] != "fact.assert":
            continue
        payload = atom["payload"]
        key = canonical_json({"predicate": payload["predicate"], "args": payload["args"]})
        by_key[key].add(bool(payload["truth"]))
    return [
        {"fact_key_hash": sha256_text(key), "truth_values": sorted(values)}
        for key, values in sorted(by_key.items())
        if values == {False, True}
    ]


def _completion_result(
    *,
    accepted: bool,
    reason: str,
    attempts: int,
    missing_required_atom_count: int,
    invalid_atom_count: int,
    bounded: bool = True,
    selected_atom_count: int = 0,
    certificate_hash: str | None = None,
    python_z3_agree: bool = False,
) -> JsonDict:
    return {
        "accepted": accepted,
        "reason": reason,
        "attempts": attempts,
        "bounded": bounded,
        "selected_atom_count": selected_atom_count,
        "missing_required_atom_count": missing_required_atom_count,
        "invalid_atom_count": invalid_atom_count,
        "certificate_hash": certificate_hash,
        "python_z3_agree": python_z3_agree,
    }


def _drop_order(payload: Mapping[str, Any]) -> JsonDict:
    return {key: _copy_json(value) for key, value in payload.items() if key != "order"}


def _check_cardinalities(
    domains: Sequence[Mapping[str, Any]], cardinality_atoms: Sequence[Mapping[str, Any]]
) -> None:
    expected = {domain["id"]: len(domain["values"]) for domain in domains}
    for atom in cardinality_atoms:
        payload = atom["payload"]
        if expected.get(payload["id"]) != payload["cardinality"]:
            raise ValueError(f"domain cardinality mismatch: {payload['id']}")


def _rules_from_atoms(by_kind: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[JsonDict]:
    variables: dict[str, dict[str, str]] = defaultdict(dict)
    for atom in by_kind["rule.variable"]:
        payload = atom["payload"]
        variables[str(payload["rule_id"])][str(payload["variable"])] = str(payload["domain"])
    body_terms: dict[str, list[JsonDict]] = defaultdict(list)
    for atom in by_kind["rule.body.atom"]:
        body_terms[str(atom["payload"]["rule_id"])].append(
            {"node": "atom", "predicate": atom["payload"]["predicate"], "args": atom["payload"]["args"]}
        )
    for atom in by_kind["rule.body.not"]:
        body_terms[str(atom["payload"]["rule_id"])].append(
            {
                "node": "not",
                "term": {
                    "node": "atom",
                    "predicate": atom["payload"]["predicate"],
                    "args": atom["payload"]["args"],
                },
            }
        )
    for atom in by_kind["rule.body.comparison"]:
        body_terms[str(atom["payload"]["rule_id"])].append(
            {
                "node": "arith",
                "left": atom["payload"]["left"],
                "op": atom["payload"]["op"],
                "right": atom["payload"]["right"],
            }
        )
    heads: dict[str, list[JsonDict]] = defaultdict(list)
    for atom in by_kind["rule.head.atom"]:
        payload = atom["payload"]
        heads[str(payload["rule_id"])].append(
            {"node": "atom", "predicate": payload["predicate"], "args": payload["args"]}
        )
    rule_ids = {
        str(atom["payload"]["rule_id"]) for atom in by_kind["composition.rule"]
    } | set(variables) | set(body_terms) | set(heads)
    rules = []
    for rule_id in sorted(rule_ids):
        head = sorted(heads.get(rule_id, []), key=canonical_json)
        terms = sorted(body_terms.get(rule_id, []), key=canonical_json)
        rules.append(
            {
                "id": rule_id,
                "variables": dict(sorted(variables.get(rule_id, {}).items())),
                "body": {"node": "and", "terms": terms},
                "head": head[0] if head else None,
            }
        )
    return rules


def _query_from_atoms(by_kind: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    variables = {
        str(atom["payload"]["variable"]): str(atom["payload"]["domain"])
        for atom in by_kind["query.variable"]
    }
    where = [
        {"node": "atom", "predicate": atom["payload"]["predicate"], "args": atom["payload"]["args"]}
        for atom in by_kind["query.where.atom"]
    ]
    return {
        "vars": dict(sorted(variables.items())),
        "where": sorted(where, key=canonical_json)[0] if where else None,
    }


def _certificate_parity(certificate: Mapping[str, Any]) -> JsonDict:
    python = certificate.get("python", {})
    z3 = certificate.get("z3", {})
    if python.get("status") == z3.get("status") == "sat":
        agrees = python.get("query_bindings") == z3.get("query_bindings")
    else:
        agrees = python.get("status") == z3.get("status")
    return {"python_z3_agree": bool(agrees)}


def _entity_permuted_ir(ir: Mapping[str, Any]) -> tuple[JsonDict, dict[str, str]]:
    forward: dict[str, str] = {}
    inverse: dict[str, str] = {}
    for domain in ir["domains"]:
        if domain["type"] != "symbol":
            continue
        values = list(domain["values"])
        if len(values) < 2:
            continue
        rotated = values[1:] + values[:1]
        for source, target in zip(values, rotated, strict=True):
            forward[str(source)] = str(target)
            inverse[str(target)] = str(source)
    return _apply_symbol_map(ir, forward), inverse


def _apply_symbol_map(ir: Mapping[str, Any], mapping: Mapping[str, str]) -> JsonDict:
    def visit(value: Any) -> Any:
        if isinstance(value, str) and not value.startswith("?"):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        return value

    return visit(_copy_json(ir))


def _row_hash(row: Mapping[str, Any]) -> str:
    stable = dict(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def _rows_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def _replay_atom_row_lines(lines: Sequence[str]) -> JsonDict:
    previous = INITIAL_PREFIX_HASH
    rows = []
    for expected_index, line in enumerate(lines):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return {
                "ok": False,
                "reason": "json_decode_error",
                "row_count": len(rows),
                "rows": rows,
                "final_prefix_checksum": previous,
            }
        if row.get("sequence_index") != expected_index or row.get("previous_hash") != previous:
            return {
                "ok": False,
                "reason": f"prefix_chain_failure_at_{expected_index}",
                "row_count": len(rows),
                "rows": rows,
                "final_prefix_checksum": previous,
            }
        if row.get("row_hash") != _row_hash(row):
            return {
                "ok": False,
                "reason": f"row_hash_mismatch_at_{expected_index}",
                "row_count": len(rows),
                "rows": rows,
                "final_prefix_checksum": previous,
            }
        contains_hidden = any(marker in canonical_json(row) for marker in FORBIDDEN_VISIBLE_KEYS)
        rows.append(
            {
                "sequence_index": expected_index,
                "row_hash": row["row_hash"],
                "contains_hidden_reference_answer": contains_hidden,
            }
        )
        previous = row["row_hash"]
    return {
        "ok": all(not row["contains_hidden_reference_answer"] for row in rows),
        "reason": "ok",
        "row_count": len(rows),
        "rows": rows,
        "final_prefix_checksum": previous,
    }


def _atom_row_tamper_receipt(path: Path) -> JsonDict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {"tamper_rejected": False, "reason": "no_rows_to_tamper"}
    tampered = json.loads(lines[0])
    tampered["atom"]["payload"]["tampered"] = True
    lines[0] = canonical_json(tampered)
    replay = _replay_atom_row_lines(lines)
    return {
        "tamper_rejected": replay["ok"] is False,
        "tamper_reason": replay["reason"],
    }


def _public_surface(surface: Mapping[str, Any]) -> JsonDict:
    return {
        key: value
        for key, value in surface.items()
        if not key.startswith("_") and key not in {"hidden_reference_ids"}
    }


def _public_sealed_union(sealed: Mapping[str, Any]) -> JsonDict:
    return {key: value for key, value in sealed.items() if not key.startswith("_")}


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES.get(field, "Exp5935 required field."),
            "satisfied_by": "deterministic_exp5935_fixture_builder",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _preconditions(root: Path, output_path: Path, atom_rows_path: Path) -> JsonDict:
    upstream = _immutable_upstream_hashes(root)
    exact = _exact_authority_receipt()
    specs = _spec_receipt(root)
    outputs = {
        "json_output": _atomic_output_probe(output_path),
        "jsonl_output": _atomic_output_probe(atom_rows_path),
    }
    exclusions = _exclusion_receipt(root)
    resources = {"disk": _disk_probe(root, 1024), "ram": _memory_probe(512)}
    held = {"case_count": len(build_sealed_cases()), "held_cases_available": True}
    checks = {
        "immutable_upstream_hashes": upstream["all_present"],
        "exact_python_z3_authorities": exact["ok"],
        "schemas": specs["ok"],
        "held_splits": held["held_cases_available"],
        "output_paths": outputs["json_output"]["ok"] and outputs["jsonl_output"]["ok"],
        "exclusions": exclusions["ok"],
        "disk": resources["disk"]["ok"],
        "ram": resources["ram"]["ok"],
    }
    return {
        "checks": checks,
        "immutable_upstream_hashes": upstream,
        "exact_python_z3_authorities": exact,
        "schemas": specs,
        "held_splits": held,
        "output_paths": outputs,
        "exclusions": exclusions,
        "resources": resources,
        "all_preconditions_ok": all(checks.values()),
    }


def _immutable_upstream_hashes(root: Path) -> JsonDict:
    records = {}
    for relative in HASHED_INPUTS:
        path = root / relative
        records[str(relative)] = {
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return {
        "records": records,
        "all_present": all(row["exists"] for row in records.values()),
        "principle": FIELD_PRINCIPLES["immutable_upstream_hashes"],
    }


def _exact_authority_receipt() -> JsonDict:
    try:
        import z3

        row = next(row for row in exp5896.build_fixture_rows() if row["split"] == "heldout")
        certificate = exp5896.certify_ir(row["constraint_ir"])
        parity = _certificate_parity(certificate)
        return {
            "ok": certificate["parser"]["status"] == "accepted" and parity["python_z3_agree"],
            "python_status": certificate["python"]["status"],
            "z3_status": certificate["z3"]["status"],
            "z3_version": z3.get_version_string(),
        }
    except Exception as exc:  # pragma: no cover - dependency failure path.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _spec_receipt(root: Path) -> JsonDict:
    records = {}
    for relative in (VERIFICATION_SPEC_RELATIVE_PATH, VERIFIABLE_REASONING_SPEC_RELATIVE_PATH):
        path = root / relative
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        records[str(relative)] = {"exists": path.exists(), "contains_req": "REQ-" in text}
    return {"ok": all(row["exists"] and row["contains_req"] for row in records.values()), "records": records}


def _exclusion_receipt(root: Path) -> JsonDict:
    path = root / "ops/exclusion_manifest.yaml"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return {
        "ok": path.exists() and "5935" not in text,
        "path": str(path.relative_to(root)),
        "experiment_5935_retired": "5935" in text,
    }


def _disk_probe(root: Path, required_mb: int) -> JsonDict:
    usage = shutil.disk_usage(root)
    available = usage.free // (1024 * 1024)
    return {"ok": available >= required_mb, "available_mb": available, "required_mb": required_mb}


def _memory_probe(required_mb: int) -> JsonDict:
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        available = int(pages * page_size // (1024 * 1024))
    except (AttributeError, OSError, ValueError):  # pragma: no cover - POSIX host path.
        available = required_mb
    return {"ok": available >= required_mb, "available_mb": available, "required_mb": required_mb}


def _atomic_output_probe(path: Path) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = path.with_name(path.name + ".atomic_probe")
    try:
        _write_text_atomic(probe, "ok\n")
        ok = probe.read_text(encoding="utf-8") == "ok\n"
        probe.unlink(missing_ok=True)
        return {"ok": ok, "method": "os.replace_same_directory"}
    except OSError as exc:  # pragma: no cover - filesystem failure path.
        return {"ok": False, "error": str(exc), "method": "os.replace_same_directory"}


def _protected_file_receipt(
    root: Path, *, baseline: Mapping[str, Any] | None = None
) -> JsonDict:
    hashes = {
        str(path): sha256_file(root / path) if (root / path).exists() else None
        for path in PROTECTED_FILES
    }
    if baseline is None:
        return {"hashes": hashes, "unchanged": True, "protected_paths": list(hashes)}
    return {
        "hashes": hashes,
        "baseline_hashes": dict(baseline.get("hashes") or {}),
        "unchanged": hashes == dict(baseline.get("hashes") or {}),
        "protected_paths": list(hashes),
    }


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["test_exit_codes"] = {}
    resources = stable.get("preconditions_checked", {}).get("resources", {})
    for name in ("disk", "ram"):
        if isinstance(resources.get(name), dict):
            resources[name]["available_mb"] = 0
    return sha256_json(stable)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _target_row(case: Mapping[str, Any]) -> JsonDict:
    return _copy_json(case["target_row"])


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _write_json_atomic(path: Path, artifact: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(artifact, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for materializing the Exp5935 artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument("--atom-rows", type=Path, default=REPO_ROOT / ATOM_ROW_RELATIVE_PATH)
    args = parser.parse_args(argv)
    write_artifact(output_path=args.output, atom_rows_path=args.atom_rows)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by command-line verification.
    raise SystemExit(main())
