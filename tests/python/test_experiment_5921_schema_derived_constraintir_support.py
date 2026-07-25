"""Tests for Exp5921 schema-derived ConstraintIR support.

Spec refs: REQ-VERIFY-5921, SCENARIO-VERIFY-5921-SCHEMA,
SCENARIO-VERIFY-5921-PREFIX, SCENARIO-VERIFY-5921-ADJUDICATION.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5921_schema_derived_constraintir_support as exp5921


def test_operation_signature_compiler_derives_support_without_answer_ids() -> None:
    # REQ-VERIFY-5921, SCENARIO-VERIFY-5921-SCHEMA
    schema = exp5921.versioned_operation_signatures()
    support = exp5921.compile_schema_support(schema)
    cases = exp5921.build_adversary_cases()
    leakage = exp5921.no_answer_leakage_receipt(support, cases)

    assert schema["schema_version"] == exp5921.OPERATION_SIGNATURE_SCHEMA_VERSION
    assert support["schema_hash"].startswith("sha256:")
    assert {"schema_version", "domains", "facts", "rules", "query"}.issubset(
        set(support["grammar_terminals"]["top_level_keys"])
    )
    assert {"atom", "and", "not", "arith"}.issubset(
        set(support["grammar_terminals"]["expression_nodes"])
    )
    assert support["type_domain_transitions"]["fact"]["uses"] == ["predicate.args"]
    assert support["scope_rules"]["rule.variables"]["scope"] == "rule_local"
    assert leakage["leak_free"] is True
    assert leakage["complete_answer_enumeration_detected"] is False


def test_full_support_rejects_type_scope_and_recovers_dead_ends() -> None:
    # REQ-VERIFY-5921, SCENARIO-VERIFY-5921-PREFIX
    support = exp5921.compile_schema_support()
    cases = {case["case_id"]: case for case in exp5921.build_adversary_cases()}

    valid = exp5921.validate_with_support(cases["held_family_menu_canonical"]["candidate"], support)
    invalid_ref = exp5921.validate_with_support(
        cases["invalid_reference_missing_predicate"]["candidate"], support
    )
    type_confusion = exp5921.validate_with_support(
        cases["type_confusion_wrong_domain_value"]["candidate"], support
    )
    scope_leak = exp5921.validate_with_support(
        cases["scope_leak_rule_variable"]["candidate"], support
    )
    empty_support = exp5921.validate_with_support(
        cases["held_family_menu_canonical"]["candidate"], exp5921.empty_support()
    )
    bounded = exp5921.bounded_dead_end_matrix(support, list(cases.values()))

    assert valid["full_support_valid"] is True
    assert invalid_ref["grammar_valid"] is True
    assert invalid_ref["scope_valid"] is False
    assert type_confusion["grammar_valid"] is True
    assert type_confusion["type_valid"] is False
    assert scope_leak["scope_valid"] is False
    assert "unknown variable" in " ".join(scope_leak["scope_errors"])
    assert empty_support["grammar_valid"] is False
    assert bounded["all_recovery_bounded"] is True
    assert bounded["recovered_within_budget"]["accepted"] is True
    assert bounded["rejected_after_budget"]["attempts_used"] == exp5921.MAX_REJECTIONS


def test_prefix_monotonicity_and_support_replay_are_deterministic() -> None:
    # REQ-VERIFY-5921, SCENARIO-VERIFY-5921-PREFIX
    support = exp5921.compile_schema_support()
    cases = exp5921.build_adversary_cases()
    matrix = exp5921.prefix_monotonicity_matrix(support, cases)
    first = exp5921.compile_schema_support()
    second = exp5921.compile_schema_support()

    assert matrix["all_prefixes_monotone"] is True
    assert matrix["held_family_prefixes_supported"] is True
    assert matrix["checked_prefix_count"] > 0
    assert first == second
    assert exp5921.support_replay_receipt(first, second)["deterministic_replay"] is True


def test_exact_adjudication_keeps_structural_and_semantic_authority_separate() -> None:
    # REQ-VERIFY-5921, SCENARIO-VERIFY-5921-ADJUDICATION
    support = exp5921.compile_schema_support()
    cases = exp5921.build_adversary_cases()
    panel = exp5921.run_support_panel(support, cases)

    boundary = panel["semantic_authority_boundary"]
    retention = panel["correct_mode_retention_and_overpruning"]
    agreement = panel["exact_python_z3_agreement"]

    assert boundary["structurally_admitted_semantically_false_cases"] > 0
    assert boundary["unsafe_semantic_acceptance_count"] == 0
    assert boundary["grammar_type_scope_counted_as_semantic_correct"] is False
    assert agreement["all_python_z3_agree"] is True
    assert retention["full_support"]["correct_mode_retention"] == 1.0
    assert retention["full_support"]["overpruned_correct_cases"] == 0
    assert retention["corrupted_schema"]["overpruned_correct_cases"] > 0
    assert retention["grammar_only"]["accepted_semantic_false_cases"] > 0


def test_write_artifact_required_fields_validation_and_stability(tmp_path: Path) -> None:
    # REQ-VERIFY-5921, SCENARIO-VERIFY-5921-SCHEMA, SCENARIO-VERIFY-5921-ADJUDICATION
    output_path = tmp_path / "experiment_5921_schema_derived_constraintir_support.json"
    artifact = exp5921.write_artifact(output_path=output_path, duration_s=0.0)
    loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(exp5921.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["inference_substrate"] == "deterministic_schema_compilation_no_llm"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["schema_decode_contract_ready_score"] == 1.0
    assert artifact["open_ir_not_finite_id_receipt"]["leak_free"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["test_exit_codes"] == {}
    exp5921.validate_artifact(artifact)

    second = exp5921.write_artifact(output_path=output_path, duration_s=0.0)
    assert second["reproducibility_checksum"] == artifact["reproducibility_checksum"]

    for key, value, message in [
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("verifier_is_oracle", False, "verifier_is_oracle"),
        ("schema_decode_contract_ready_score", 0.5, "schema_decode"),
        ("honest_verdict", "blocked: bad", "complete_ready"),
    ]:
        broken = dict(artifact)
        broken[key] = value
        with pytest.raises(ValueError, match=message):
            exp5921.validate_artifact(broken)

    missing = dict(artifact)
    del missing["semantic_authority_boundary"]
    with pytest.raises(ValueError, match="missing required fields"):
        exp5921.validate_artifact(missing)


def test_defensive_fail_closed_branches_are_reported(tmp_path: Path) -> None:
    # REQ-VERIFY-5921, SCENARIO-VERIFY-5921-PREFIX, SCENARIO-VERIFY-5921-ADJUDICATION
    support = exp5921.compile_schema_support()
    cases = exp5921.build_adversary_cases()
    valid = json.loads(json.dumps(cases[0]["candidate"]))

    leaky_support = dict(support)
    leaky_support["case_label"] = cases[0]["case_id"]
    leaky_support["complete_payload_hash"] = exp5921.sha256_text(
        exp5921.canonical_json(cases[0]["candidate"])
    )
    leakage = exp5921.no_answer_leakage_receipt(leaky_support, cases[:1])
    assert leakage["leak_free"] is False
    assert leakage["complete_answer_enumeration_detected"] is True

    assert exp5921.prefix_support(support, ["query.define"]) == []
    broken_prefix = exp5921.prefix_monotonicity_matrix(
        {"operation_order": ["domain.declare"]},
        [{"case_id": "bad", "split_role": "held_family", "candidate": valid}],
    )
    assert broken_prefix["all_prefixes_monotone"] is False

    rows = {row["row_id"]: row for row in exp5921.exp5896.build_fixture_rows()}
    unsat_case = {
        "case_id": "unsat",
        "candidate": rows["exp5896-access_control-unsat_ir"]["constraint_ir"],
    }
    assert exp5921.exact_python_z3_agreement([unsat_case])["all_python_z3_agree"] is True

    bad_payloads: list[tuple[object, str]] = [
        ([], "payload must be object"),
        ({key: value for key, value in valid.items() if key != "facts"}, "top-level keys"),
        ({**valid, "schema_version": "bad"}, "unsupported ConstraintIR"),
        ({**valid, "facts": "bad"}, "facts must be list"),
        ({**valid, "query": []}, "query must be object"),
    ]
    bad_node = json.loads(json.dumps(valid))
    bad_node["query"]["where"] = {"node": "or", "terms": []}
    bad_payloads.append((bad_node, "unsupported expression node"))

    bad_op = json.loads(json.dumps(rows["exp5896-task_selection-canonical"]["constraint_ir"]))
    bad_op["rules"][0]["body"]["terms"][2]["op"] = "!="
    bad_payloads.append((bad_op, "unsupported arithmetic op"))

    node_empty_support = json.loads(json.dumps(support))
    node_empty_support["grammar_terminals"]["expression_nodes"] = []
    assert (
        "no expression node terminals"
        in exp5921.validate_with_support(valid, node_empty_support)["grammar_errors"]
    )

    for payload, message in bad_payloads:
        assert message in " ".join(
            exp5921.validate_with_support(payload, support)["grammar_errors"]
        )

    type_scope_payloads: list[tuple[dict[str, object], str]] = []
    domain_not_object = json.loads(json.dumps(valid))
    domain_not_object["domains"][0] = 7
    type_scope_payloads.append((domain_not_object, "domain must be object"))

    missing_domain_values = json.loads(json.dumps(valid))
    del missing_domain_values["domains"][0]["values"]
    type_scope_payloads.append((missing_domain_values, "domain id and values"))

    symbol_bad_value = json.loads(json.dumps(valid))
    symbol_bad_value["domains"][0]["values"][0] = 1
    type_scope_payloads.append((symbol_bad_value, "expects symbol values"))

    int_bad_value = json.loads(
        json.dumps(rows["exp5896-task_selection-canonical"]["constraint_ir"])
    )
    int_bad_value["domains"][1]["values"][0] = "one"
    type_scope_payloads.append((int_bad_value, "expects integer values"))

    bad_domain_kind = json.loads(json.dumps(valid))
    bad_domain_kind["domains"][0]["type"] = "float"
    type_scope_payloads.append((bad_domain_kind, "unsupported type"))

    entity_not_object = json.loads(json.dumps(valid))
    entity_not_object["entities"][0] = 7
    type_scope_payloads.append((entity_not_object, "entity must be object"))

    entity_unknown_domain = json.loads(json.dumps(valid))
    entity_unknown_domain["entities"][0]["domain"] = "missing"
    type_scope_payloads.append((entity_unknown_domain, "unknown entity domain"))

    entity_wrong_type = json.loads(
        json.dumps(rows["exp5896-task_selection-canonical"]["constraint_ir"])
    )
    entity_wrong_type["entities"].append({"id": "1", "domain": "hours"})
    type_scope_payloads.append((entity_wrong_type, "not in symbol domain"))

    predicate_not_object = json.loads(json.dumps(valid))
    predicate_not_object["predicates"][0] = 7
    type_scope_payloads.append((predicate_not_object, "predicate must be object"))

    predicate_bad_shape = json.loads(json.dumps(valid))
    predicate_bad_shape["predicates"][0]["arg_types"] = "bad"
    type_scope_payloads.append((predicate_bad_shape, "predicate id and arg_types"))

    predicate_unknown_domain = json.loads(json.dumps(valid))
    predicate_unknown_domain["predicates"][0]["arg_types"] = ["missing"]
    type_scope_payloads.append((predicate_unknown_domain, "unknown predicate domain"))

    atom_not_object = json.loads(json.dumps(valid))
    atom_not_object["facts"][0] = 7
    type_scope_payloads.append((atom_not_object, "atom must be object"))

    atom_arity = json.loads(json.dumps(valid))
    atom_arity["facts"][0]["args"] = ["ada"]
    type_scope_payloads.append((atom_arity, "arity mismatch"))

    rule_variables_bad = json.loads(json.dumps(valid))
    rule_variables_bad["rules"][0]["variables"] = []
    type_scope_payloads.append((rule_variables_bad, "variables must be object"))

    rule_variable_name_bad = json.loads(json.dumps(valid))
    rule_variable_name_bad["rules"][0]["variables"] = {"who": "person"}
    type_scope_payloads.append((rule_variable_name_bad, "invalid variable name"))

    rule_variable_domain_bad = json.loads(json.dumps(valid))
    rule_variable_domain_bad["rules"][0]["variables"] = {"?who": "missing"}
    type_scope_payloads.append((rule_variable_domain_bad, "unknown variable domain"))

    expression_not_object = json.loads(json.dumps(valid))
    expression_not_object["rules"][0]["body"] = "bad"
    type_scope_payloads.append((expression_not_object, "expression must be object"))

    variable_wrong_domain = json.loads(json.dumps(valid))
    variable_wrong_domain["rules"][0]["body"]["terms"][0]["args"][0] = "?dept"
    type_scope_payloads.append((variable_wrong_domain, "wrong domain"))

    unknown_arith = json.loads(
        json.dumps(rows["exp5896-task_selection-canonical"]["constraint_ir"])
    )
    unknown_arith["rules"][0]["body"]["terms"][2]["left"] = "?missing"
    type_scope_payloads.append((unknown_arith, "unknown arithmetic variable"))

    non_int_arith = json.loads(
        json.dumps(rows["exp5896-task_selection-canonical"]["constraint_ir"])
    )
    non_int_arith["rules"][0]["body"]["terms"][2]["left"] = "?task"
    type_scope_payloads.append((non_int_arith, "not integer typed"))

    bad_arith_literal = json.loads(
        json.dumps(rows["exp5896-task_selection-canonical"]["constraint_ir"])
    )
    bad_arith_literal["rules"][0]["body"]["terms"][2]["right"] = "two"
    type_scope_payloads.append((bad_arith_literal, "arithmetic term"))

    for payload, message in type_scope_payloads:
        verdict = exp5921.validate_with_support(payload, support)
        assert message in " ".join(verdict["type_errors"] + verdict["scope_errors"])

    output_path = tmp_path / exp5921.RESULT_RELATIVE_PATH
    no_duration = exp5921.write_artifact(output_path=output_path)
    refreshed = exp5921.refresh_artifact_test_exit_codes(
        root=tmp_path, test_exit_codes={"focused": 0}
    )
    assert no_duration["duration_s"] >= 0.0
    assert refreshed["test_exit_codes"] == {"focused": 0}

    for mutate in ("leakage", "unsafe"):
        broken = json.loads(json.dumps(refreshed))
        if mutate == "leakage":
            broken["open_ir_not_finite_id_receipt"]["leak_free"] = False
            message = "leaked"
        else:
            broken["semantic_authority_boundary"]["unsafe_semantic_acceptance_count"] = 1
            message = "semantic authority"
        with pytest.raises(ValueError, match=message):
            exp5921.validate_artifact(broken)

    assert exp5921._required_keys([], "missing") == []
