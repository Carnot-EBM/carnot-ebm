"""Tests for Exp5935 non-pruning atomic ConstraintIR support.

Spec refs: REQ-VERIFY-5935, SCENARIO-VERIFY-5935-ATOM-UNIVERSE,
SCENARIO-VERIFY-5935-NON-PRUNING, SCENARIO-VERIFY-5935-EXACT-COMPLETION,
SCENARIO-VERIFY-5935-POOLS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5935_non_pruning_atomic_constraint_support as exp5935


def _held_case() -> dict[str, object]:
    return exp5935.build_sealed_cases()[0]


def test_atom_schema_and_visible_hidden_paths_are_generic_and_leak_free() -> None:
    # REQ-VERIFY-5935, SCENARIO-VERIFY-5935-ATOM-UNIVERSE
    schema = exp5935.versioned_atom_schema()
    contract = exp5935.generic_atom_universe_contract(schema)
    case = _held_case()
    surface = exp5935.derive_case_atom_surface(case, schema)
    leakage = exp5935.label_secrecy_receipt([surface])

    assert schema["schema_version"] == exp5935.ATOM_SCHEMA_VERSION
    assert schema["derived_from_operation_signature_schema"] is True
    assert schema["schema_hash"].startswith("sha256:")
    assert {
        "domain.declare",
        "domain.cardinality",
        "entity.declare",
        "predicate.declare",
        "fact.assert",
        "rule.body.atom",
        "rule.body.not",
        "rule.body.comparison",
        "rule.head.atom",
        "query.where.atom",
        "composition.rule",
        "composition.query",
    }.issubset(set(schema["atom_kinds"]))
    assert contract["derived_from_public_operation_schema"] is True
    assert contract["complete_answer_enumeration_forbidden"] is True
    assert surface["model_visible_vocabulary_hash"] != surface["hidden_reference_set_hash"]
    assert surface["hidden_reference_atoms_materialized_in_artifact"] is False
    assert surface["hidden_reference_count"] > 0
    assert surface["model_visible_atom_count"] >= surface["hidden_reference_count"]
    assert leakage["leak_free"] is True
    assert leakage["complete_answer_enumeration_used"] is False


def test_non_pruning_union_and_transforms_preserve_legal_atoms() -> None:
    # REQ-VERIFY-5935, SCENARIO-VERIFY-5935-NON-PRUNING
    schema = exp5935.versioned_atom_schema()
    case = _held_case()
    surface = exp5935.derive_case_atom_surface(case, schema)
    transforms = exp5935.semantic_view_transform_receipts(case, schema)
    proposals = exp5935.build_injected_proposals(
        case,
        surface,
        injection="spurious_contradiction_duplicates",
    )
    sealed = exp5935.seal_non_pruning_union(surface, proposals)

    assert transforms["all_views_invertible"] is True
    assert transforms["exact_reference_invariant"] is True
    assert transforms["transform_independent_from_model_output"] is True
    assert transforms["answer_leakage_detected"] is False
    assert sealed["pre_union_legal_deleted_count"] == 0
    assert sealed["legal_atoms_seen_before_union"] == sealed["legal_atoms_in_sealed_union"]
    assert sealed["duplicate_atom_proposal_count"] > 0
    assert sealed["invalid_atom_count"] == 0
    assert sealed["contradiction_pair_count"] > 0
    assert all(
        provenance
        for provenance in sealed["provenance_by_atom_id"].values()
    )


def test_exact_completion_matrix_reachability_and_order_invariance() -> None:
    # REQ-VERIFY-5935, SCENARIO-VERIFY-5935-EXACT-COMPLETION
    case = _held_case()
    schema = exp5935.versioned_atom_schema()
    surface = exp5935.derive_case_atom_surface(case, schema)
    matrix = exp5935.injected_completion_matrix(case, surface)

    assert matrix["complete_support"]["accepted"] is True
    assert matrix["spurious_support"]["accepted"] is True
    assert matrix["contradiction_support"]["accepted"] is True
    assert matrix["duplicate_support"]["accepted"] is True
    assert matrix["order_permutation"]["accepted"] is True
    assert matrix["order_permutation"]["certificate_hash"] == matrix["spurious_support"][
        "certificate_hash"
    ]
    assert matrix["missing_required_atom"]["accepted"] is False
    assert matrix["missing_required_atom"]["missing_required_atom_count"] == 1
    assert matrix["missing_required_atom"]["search_can_manufacture_deleted_truth"] is False
    assert matrix["empty_support"]["accepted"] is False
    assert matrix["type_scope_failure"]["invalid_atom_count"] > 0
    assert matrix["support_saturation"]["accepted"] is False
    assert matrix["support_saturation"]["bounded"] is True


def test_included_excluded_audit_rejects_accepted_pool_only_recall() -> None:
    # REQ-VERIFY-5935, SCENARIO-VERIFY-5935-POOLS
    case = _held_case()
    schema = exp5935.versioned_atom_schema()
    surface = exp5935.derive_case_atom_surface(case, schema)
    proposals = exp5935.build_injected_proposals(case, surface, injection="missing_required")
    sealed = exp5935.seal_non_pruning_union(surface, proposals)
    audit = exp5935.included_excluded_pool_audit(surface, sealed)

    assert audit["strata_frozen_before_labels_opened"] is True
    assert audit["included_pool"]["positive_count"] < surface["hidden_reference_count"]
    assert audit["excluded_pool"]["positive_count"] > 0
    assert audit["accepted_pool_only_recall_claim_rejected"] is True
    assert audit["labels_from_hidden_reference_only_after_freeze"] is True


def test_artifact_required_fields_replay_tamper_validation_and_stability(
    tmp_path: Path,
) -> None:
    # REQ-VERIFY-5935, SCENARIO-VERIFY-5935-POOLS
    output_path = tmp_path / exp5935.RESULT_RELATIVE_PATH.name
    atom_rows_path = tmp_path / exp5935.ATOM_ROW_RELATIVE_PATH.name
    artifact = exp5935.write_artifact(
        output_path=output_path,
        atom_rows_path=atom_rows_path,
        duration_s=0.0,
    )
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    replay = exp5935.replay_atom_rows(atom_rows_path)

    assert loaded == artifact
    assert set(exp5935.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["atom_support_fixture_ready_score"] == 1.0
    assert artifact["inference_substrate"] == "deterministic_exact_executor_fixture_no_llm"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["included_and_excluded_pool_contract"][
        "accepted_pool_only_recall_claim_rejected"
    ] is True
    assert replay["ok"] is True
    assert replay["row_count"] == artifact["replay_and_tamper_matrix"]["atom_rows"]["row_count"]
    exp5935.validate_artifact(artifact)

    tampered_lines = atom_rows_path.read_text(encoding="utf-8").splitlines()
    first_row = json.loads(tampered_lines[0])
    first_row["atom"]["payload"]["tampered"] = True
    tampered_lines[0] = json.dumps(first_row, sort_keys=True)
    tampered_path = tmp_path / "tampered.atoms.jsonl"
    tampered_path.write_text("\n".join(tampered_lines) + "\n", encoding="utf-8")
    assert exp5935.replay_atom_rows(tampered_path)["ok"] is False

    second = exp5935.write_artifact(
        output_path=output_path,
        atom_rows_path=atom_rows_path,
        duration_s=0.0,
    )
    assert second["reproducibility_checksum"] == artifact["reproducibility_checksum"]

    refreshed = exp5935.refresh_artifact_test_exit_codes(
        artifact_path=output_path,
        test_exit_codes={"focused": 0, "coverage": 0},
    )
    assert refreshed["test_exit_codes"] == {"focused": 0, "coverage": 0}

    for key, value, message in [
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("atom_support_fixture_ready_score", 0.5, "atom_support_fixture_ready_score"),
        ("honest_verdict", "complete_partial: wrong", "complete_ready"),
    ]:
        broken = json.loads(json.dumps(artifact))
        broken[key] = value
        with pytest.raises(ValueError, match=message):
            exp5935.validate_artifact(broken)

    missing = dict(artifact)
    del missing["generic_atom_universe_contract"]
    with pytest.raises(ValueError, match="missing required fields"):
        exp5935.validate_artifact(missing)

    leaky = json.loads(json.dumps(artifact))
    leaky["label_secrecy_and_no_complete_answer_enumeration_receipt"][
        "complete_answer_enumeration_used"
    ] = True
    with pytest.raises(ValueError, match="complete answer enumeration"):
        exp5935.validate_artifact(leaky)


def test_defensive_edges_for_replay_completion_and_validation(tmp_path: Path) -> None:
    # REQ-VERIFY-5935, SCENARIO-VERIFY-5935-EXACT-COMPLETION
    schema = exp5935.versioned_atom_schema()
    case = _held_case()
    surface = exp5935.derive_case_atom_surface(case, schema)
    sealed = exp5935.seal_non_pruning_union(
        surface,
        exp5935.build_injected_proposals(case, surface, injection="complete"),
    )

    with pytest.raises(ValueError, match="unknown injection"):
        exp5935.build_injected_proposals(case, surface, injection="unknown")
    with pytest.raises(ValueError, match="unsupported atom kind"):
        exp5935._make_atom("bad.kind", {}, schema)

    bounded = exp5935.complete_subset(case, surface, sealed, max_states=0)
    assert bounded["accepted"] is False
    assert bounded["bounded"] is True
    assert exp5935.replay_atom_rows(tmp_path / "missing.jsonl")["reason"] == "missing_atom_rows"
    assert exp5935._replay_atom_row_lines([""])["row_count"] == 0
    assert exp5935._replay_atom_row_lines(["{"])["reason"] == "json_decode_error"

    rows = exp5935.build_atom_rows([surface])
    broken = json.loads(exp5935.canonical_json(rows[0]))
    broken["previous_hash"] = "sha256:wrong"
    assert exp5935._replay_atom_row_lines([json.dumps(broken)])[
        "reason"
    ].startswith("prefix_chain_failure")

    empty_rows_path = tmp_path / "empty.jsonl"
    empty_rows_path.write_text("", encoding="utf-8")
    assert exp5935._atom_row_tamper_receipt(empty_rows_path)["tamper_rejected"] is False

    fixed_only = [
        atom for atom in surface["_hidden_reference_atoms"] if atom["atom_kind"] in exp5935.FIXED_ATOM_KINDS
    ]
    assert exp5935._first_dynamic_reference_id(fixed_only) == fixed_only[-1]["atom_id"]
    assert exp5935._contradictory_fact_atom([], surface) is None
    assert exp5935._expr_atoms({"node": "xor"}, schema, rule_id="r1", context="rule.body") == []

    bad_cardinality = json.loads(json.dumps(fixed_only))
    for atom in bad_cardinality:
        if atom["atom_kind"] == "domain.cardinality":
            atom["payload"]["cardinality"] += 1
            break
    with pytest.raises(ValueError, match="domain cardinality mismatch"):
        exp5935.ir_from_atoms(bad_cardinality)
    malformed_sealed = json.loads(json.dumps(sealed))
    malformed_sealed["_sealed_atoms"] = json.loads(json.dumps(sealed["_sealed_atoms"]))
    for atom in malformed_sealed["_sealed_atoms"]:
        if atom["atom_kind"] == "domain.cardinality":
            atom["payload"]["cardinality"] += 1
            break
    malformed_completion = exp5935.complete_subset(case, surface, malformed_sealed)
    assert malformed_completion["accepted"] is False
    assert "domain cardinality mismatch" in malformed_completion["reason"]

    tiny_ir = {
        "domains": [{"id": "one", "type": "symbol", "values": ["solo"]}],
        "entities": [{"id": "solo", "domain": "one"}],
        "predicates": [],
        "facts": [],
        "rules": [],
        "query": {"vars": {}, "where": {"node": "atom", "predicate": "p", "args": []}},
    }
    permuted, inverse = exp5935._entity_permuted_ir(tiny_ir)
    assert permuted == tiny_ir
    assert inverse == {}

    out = tmp_path / "main.json"
    atom_rows = tmp_path / "main.atoms.jsonl"
    assert exp5935.main(["--output", str(out), "--atom-rows", str(atom_rows)]) == 0
    assert out.exists()

    artifact = json.loads(out.read_text(encoding="utf-8"))
    validation_mutations = [
        (
            ("label_secrecy_and_no_complete_answer_enumeration_receipt", "leak_free"),
            False,
            "label secrecy",
        ),
        (("non_pruning_support_contract", "pre_union_legal_deleted_count"), 1, "non-pruning"),
        (
            ("included_and_excluded_pool_contract", "accepted_pool_only_recall_claim_rejected"),
            False,
            "accepted-pool",
        ),
        (
            ("search_reachability_and_inertness_receipt", "search_can_manufacture_deleted_truth"),
            True,
            "search reachability",
        ),
        (("python_z3_certificate_parity", "all_python_z3_agree"), False, "Python/Z3"),
        (("replay_and_tamper_matrix", "tamper_control", "tamper_rejected"), False, "tamper"),
    ]
    for path, value, message in validation_mutations:
        mutated = json.loads(json.dumps(artifact))
        cursor = mutated
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            exp5935.validate_artifact(mutated)
