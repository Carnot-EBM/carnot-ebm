"""Tests for Exp5896 typed ConstraintIR exact fixture.

Spec refs: REQ-BENCH-5896, SCENARIO-BENCH-5896-SCHEMA,
SCENARIO-BENCH-5896-CERTIFICATES, SCENARIO-BENCH-5896-LEAKAGE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896


def _rows_by_id(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["row_id"]): row for row in rows}


# REQ-BENCH-5896, SCENARIO-BENCH-5896-SCHEMA
def test_parser_typechecker_rejects_unknown_ambiguous_and_type_error_constructs() -> None:
    valid = exp5896.make_access_control_ir()
    parsed = exp5896.parse_constraint_ir(valid)

    assert parsed.schema_version == exp5896.CONSTRAINT_IR_SCHEMA_VERSION
    assert [domain.name for domain in parsed.domains] == ["person", "department"]
    assert "eligible" in parsed.predicates

    unknown = copy.deepcopy(valid)
    unknown["backend_hint"] = "z3"
    with pytest.raises(exp5896.ConstraintIRValidationError, match="unknown top-level"):
        exp5896.parse_constraint_ir(unknown)

    ambiguous = copy.deepcopy(valid)
    ambiguous["rules"][0]["body"] = {
        "node": "and",
        "terms": [],
        "predicate": "works_in",
        "args": ["who", "dept"],
    }
    with pytest.raises(exp5896.ConstraintIRValidationError, match="unknown fields"):
        exp5896.parse_constraint_ir(ambiguous)

    bad_type = exp5896.make_task_selection_ir()
    bad_type["facts"][0]["args"][0] = "red"
    with pytest.raises(exp5896.ConstraintIRValidationError, match="not in domain"):
        exp5896.parse_constraint_ir(bad_type)

    unsupported = copy.deepcopy(valid)
    unsupported["query"]["where"] = {"node": "or", "terms": []}
    with pytest.raises(exp5896.ConstraintIRValidationError, match="unsupported expression"):
        exp5896.parse_constraint_ir(unsupported)


# REQ-BENCH-5896, SCENARIO-BENCH-5896-SCHEMA
def test_parser_typechecker_fail_closed_guard_matrix() -> None:
    base = exp5896.make_access_control_ir()

    invalid_payloads: list[tuple[object, str]] = [
        ([], "must be an object"),
        ({key: value for key, value in base.items() if key != "facts"}, "missing top-level"),
        ({**base, "schema_version": "wrong"}, "unsupported schema_version"),
        ({**base, "domains": "bad"}, "domains must be a list"),
    ]

    empty_domain_id = copy.deepcopy(base)
    empty_domain_id["domains"][0]["id"] = ""
    invalid_payloads.append((empty_domain_id, "domain id"))

    duplicate_domain = copy.deepcopy(base)
    duplicate_domain["domains"][1]["id"] = "person"
    invalid_payloads.append((duplicate_domain, "duplicate domain"))

    unsupported_domain = copy.deepcopy(base)
    unsupported_domain["domains"][0]["type"] = "float"
    invalid_payloads.append((unsupported_domain, "unsupported domain type"))

    empty_values = copy.deepcopy(base)
    empty_values["domains"][0]["values"] = []
    invalid_payloads.append((empty_values, "finite and non-empty"))

    duplicate_values = copy.deepcopy(base)
    duplicate_values["domains"][0]["values"] = ["ada", "ada"]
    invalid_payloads.append((duplicate_values, "values must be unique"))

    symbol_type = copy.deepcopy(base)
    symbol_type["domains"][0]["values"] = ["ada", 1]
    invalid_payloads.append((symbol_type, "expects string values"))

    int_type = exp5896.make_task_selection_ir()
    int_type["domains"][1]["values"] = [1, "two"]
    invalid_payloads.append((int_type, "expects integer values"))

    bad_entity_type = copy.deepcopy(base)
    bad_entity_type["entities"][0]["id"] = 7
    invalid_payloads.append((bad_entity_type, "entity id and domain"))

    unknown_entity_domain = copy.deepcopy(base)
    unknown_entity_domain["entities"][0]["domain"] = "missing"
    invalid_payloads.append((unknown_entity_domain, "unknown entity domain"))

    int_entity = exp5896.make_task_selection_ir()
    int_entity["entities"].append({"id": "1", "domain": "hours"})
    invalid_payloads.append((int_entity, "entities may only inhabit symbol domains"))

    missing_entity = copy.deepcopy(base)
    missing_entity["entities"][0]["id"] = "zoe"
    invalid_payloads.append((missing_entity, "not in domain"))

    bad_predicate_id = copy.deepcopy(base)
    bad_predicate_id["predicates"][0]["id"] = ""
    invalid_payloads.append((bad_predicate_id, "predicate id"))

    duplicate_predicate = copy.deepcopy(base)
    duplicate_predicate["predicates"][1]["id"] = "works_in"
    invalid_payloads.append((duplicate_predicate, "duplicate predicate"))

    unknown_predicate_domain = copy.deepcopy(base)
    unknown_predicate_domain["predicates"][0]["arg_types"] = ["missing"]
    invalid_payloads.append((unknown_predicate_domain, "unknown predicate domain"))

    bad_fact_truth = copy.deepcopy(base)
    bad_fact_truth["facts"][0]["truth"] = "true"
    invalid_payloads.append((bad_fact_truth, "fact truth"))

    bad_rule_id = copy.deepcopy(base)
    bad_rule_id["rules"][0]["id"] = ""
    invalid_payloads.append((bad_rule_id, "rule id"))

    bad_var_name = copy.deepcopy(base)
    bad_var_name["rules"][0]["variables"] = {"who": "person"}
    invalid_payloads.append((bad_var_name, "variable names must start"))

    bad_var_domain = copy.deepcopy(base)
    bad_var_domain["rules"][0]["variables"] = {"?who": "missing"}
    invalid_payloads.append((bad_var_domain, "unknown domain"))

    empty_and = copy.deepcopy(base)
    empty_and["rules"][0]["body"] = {"node": "and", "terms": []}
    invalid_payloads.append((empty_and, "requires at least one term"))

    bad_arith_op = exp5896.make_task_selection_ir()
    bad_arith_op["rules"][0]["body"]["terms"][2]["op"] = "!="
    invalid_payloads.append((bad_arith_op, "unsupported arithmetic op"))

    unknown_arith_var = exp5896.make_task_selection_ir()
    unknown_arith_var["rules"][0]["body"]["terms"][2]["left"] = "?missing"
    invalid_payloads.append((unknown_arith_var, "unknown arithmetic variable"))

    non_int_arith_var = exp5896.make_task_selection_ir()
    non_int_arith_var["rules"][0]["body"]["terms"][2]["left"] = "?task"
    invalid_payloads.append((non_int_arith_var, "not integer typed"))

    bad_arith_term = exp5896.make_task_selection_ir()
    bad_arith_term["rules"][0]["body"]["terms"][2]["right"] = "two"
    invalid_payloads.append((bad_arith_term, "arithmetic terms"))

    bad_head = copy.deepcopy(base)
    bad_head["rules"][0]["head"] = {"node": "not", "term": bad_head["rules"][0]["head"]}
    invalid_payloads.append((bad_head, "must be atom"))

    unknown_predicate = copy.deepcopy(base)
    unknown_predicate["facts"][0]["predicate"] = "missing"
    invalid_payloads.append((unknown_predicate, "unknown predicate"))

    arity_mismatch = copy.deepcopy(base)
    arity_mismatch["facts"][0]["args"] = ["ada"]
    invalid_payloads.append((arity_mismatch, "arity mismatch"))

    unknown_variable = copy.deepcopy(base)
    unknown_variable["rules"][0]["body"]["terms"][0]["args"][0] = "?missing"
    invalid_payloads.append((unknown_variable, "unknown variable"))

    wrong_variable_domain = copy.deepcopy(base)
    wrong_variable_domain["rules"][0]["body"]["terms"][0]["args"][0] = "?dept"
    invalid_payloads.append((wrong_variable_domain, "wrong domain"))

    recursive = copy.deepcopy(base)
    recursive["rules"][0]["body"]["terms"].append(
        {"node": "atom", "predicate": "eligible", "args": ["?who"]}
    )
    invalid_payloads.append((recursive, "recursive or multi-stage"))

    for payload, message in invalid_payloads:
        with pytest.raises(exp5896.ConstraintIRValidationError, match=message):
            exp5896.parse_constraint_ir(payload)  # type: ignore[arg-type]

    assert exp5896._compare_ints(1, "<", 2) is True
    assert exp5896._compare_ints(2, "==", 2) is True
    assert exp5896._compare_ints(3, ">", 2) is True
    assert exp5896._canonical_value("outside", {}) == "outside"


# REQ-BENCH-5896, SCENARIO-BENCH-5896-CERTIFICATES
def test_python_z3_compilers_agree_and_replay_certificates() -> None:
    rows = exp5896.build_fixture_rows()

    valid_rows = [row for row in rows if row["expected_status"] == "valid"]
    assert valid_rows
    for row in valid_rows:
        cert = row["certificates"]
        assert cert["parser"]["status"] == "accepted"
        assert cert["python"]["status"] == "sat"
        assert cert["z3"]["status"] == "sat"
        assert cert["cross_backend_agreement"]["agrees"] is True
        replay = exp5896.replay_row_certificate(row)
        assert replay["ok"] is True

    row_map = _rows_by_id(rows)
    unsat = row_map["exp5896-access_control-unsat_ir"]
    assert unsat["expected_status"] == "unsat"
    assert unsat["certificates"]["python"]["status"] == "unsat"
    assert unsat["certificates"]["z3"]["status"] == "unsat"
    assert unsat["certificates"]["counterexample"]["kind"] == "contradictory_fact"

    negative_first = exp5896.make_access_control_ir()
    negative_first["facts"].insert(
        0, {"predicate": "approved", "args": ["cardiology"], "truth": False}
    )
    assert exp5896.certify_ir(negative_first)["python"]["status"] == "unsat"

    derived_conflict = exp5896.make_access_control_ir()
    derived_conflict["facts"].append({"predicate": "eligible", "args": ["ada"], "truth": False})
    conflict_cert = exp5896.certify_ir(derived_conflict)
    assert conflict_cert["python"]["status"] == "unsat"
    assert conflict_cert["counterexample"]["kind"] == "derived_negative_conflict"


# REQ-BENCH-5896, SCENARIO-BENCH-5896-LEAKAGE
def test_semantic_equivalence_uses_behavior_not_surface_text() -> None:
    rows = exp5896.build_fixture_rows()
    row_map = _rows_by_id(rows)

    canonical = row_map["exp5896-access_control-canonical"]
    paraphrase = row_map["exp5896-access_control-paraphrase"]
    renamed = row_map["exp5896-access_control-symbol_renaming"]
    permuted = row_map["exp5896-access_control-order_permutation"]
    omitted = row_map["exp5896-access_control-omitted_constraint"]
    nonequivalent = row_map["exp5896-access_control-semantic_nonequivalence"]

    assert canonical["natural_language"] != paraphrase["natural_language"]
    assert renamed["constraint_ir"] != canonical["constraint_ir"]
    assert permuted["constraint_ir"] != canonical["constraint_ir"]

    for equivalent in (paraphrase, renamed, permuted):
        assert equivalent["semantic_equivalence"]["equivalent_to_canonical"] is True
        assert (
            equivalent["semantic_equivalence"]["canonical_behavior_hash"]
            == canonical["semantic_equivalence"]["behavior_hash"]
        )

    assert omitted["semantic_equivalence"]["equivalent_to_canonical"] is False
    assert nonequivalent["semantic_equivalence"]["equivalent_to_canonical"] is False
    assert omitted["certificates"]["python"]["query_bindings"] != canonical["certificates"]["python"][
        "query_bindings"
    ]


# REQ-BENCH-5896, SCENARIO-BENCH-5896-LEAKAGE
def test_groups_do_not_cross_splits_and_controls_are_balanced() -> None:
    rows = exp5896.build_fixture_rows()
    artifact = exp5896.build_artifact(rows, root=exp5896.REPO_ROOT, duration_s=0.0)

    leakage = artifact["split_and_group_leakage_receipts"]
    assert leakage["group_cross_split_count"] == 0
    assert leakage["splits"] == {"train": 8, "dev": 6, "heldout": 6}
    assert leakage["heldout_valid_rows"] >= 3
    assert leakage["heldout_control_rows"] >= 2

    controls = artifact["invalid_unsat_and_nonequivalence_controls"]
    assert controls["invalid_ir"] == 1
    assert controls["unsat_ir"] == 1
    assert controls["type_error"] == 1
    assert controls["omitted_constraint"] == 3
    assert controls["semantic_nonequivalence"] == 3


# REQ-BENCH-5896, SCENARIO-BENCH-5896-CERTIFICATES
def test_write_fixture_writes_required_artifact_fields_and_stable_rows(tmp_path: Path) -> None:
    artifact = exp5896.write_fixture(
        root=tmp_path,
        duration_s=0.0,
        test_exit_codes={
            ".venv/bin/pytest tests/python/test_experiment_5896_typed_constraint_ir_fixture.py -q --no-cov -n 0": 0
        },
    )

    result_path = tmp_path / exp5896.RESULT_RELATIVE_PATH
    row_path = tmp_path / exp5896.ROW_FILE_RELATIVE_PATH
    assert result_path.exists()
    assert row_path.exists()

    loaded = json.loads(result_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in row_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert set(exp5896.REQUIRED_ARTIFACT_FIELDS).issubset(loaded)
    assert loaded["status"] == "ready"
    assert loaded["typed_constraint_ir_fixture_ready_score"] == 1.0
    assert loaded["inference_substrate"] == "deterministic_exact_solver_labeled_dataset_no_llm"
    assert loaded["verifier_is_oracle"] is True
    assert loaded["honest_verdict"].startswith("ready:")
    assert loaded["row_file_receipt"]["row_count"] == len(rows) == 20
    assert loaded["row_file_receipt"]["sha256"] == exp5896.sha256_file(row_path)
    assert exp5896.replay_artifact(root=tmp_path)["ok"] is True

    second = exp5896.write_fixture(root=tmp_path, duration_s=0.0)
    assert second["reproducibility_checksum"] == artifact["reproducibility_checksum"]
    assert row_path.read_text(encoding="utf-8").splitlines() == [
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for row in exp5896.build_fixture_rows()
    ]


# REQ-BENCH-5896, SCENARIO-BENCH-5896-CERTIFICATES
def test_replay_detects_row_file_drift(tmp_path: Path) -> None:
    exp5896.write_fixture(root=tmp_path, duration_s=0.0)
    row_path = tmp_path / exp5896.ROW_FILE_RELATIVE_PATH
    row_path.write_text(row_path.read_text(encoding="utf-8") + "\n{}", encoding="utf-8")

    with pytest.raises(exp5896.ConstraintIRReplayError, match="row file hash"):
        exp5896.replay_artifact(root=tmp_path)


# REQ-BENCH-5896, SCENARIO-BENCH-5896-CERTIFICATES
def test_replay_detects_content_certificate_and_checksum_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exp5896.write_fixture(root=tmp_path, duration_s=0.0)
    result_path = tmp_path / exp5896.RESULT_RELATIVE_PATH
    row_path = tmp_path / exp5896.ROW_FILE_RELATIVE_PATH

    rows = [
        json.loads(line)
        for line in row_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["natural_language"] = "tampered but hash receipt updated"
    row_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) for row in rows)
        + "\n",
        encoding="utf-8",
    )
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    artifact["row_file_receipt"]["sha256"] = exp5896.sha256_file(row_path)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(exp5896.ConstraintIRReplayError, match="content"):
        exp5896.replay_artifact(root=tmp_path)

    exp5896.write_fixture(root=tmp_path, duration_s=0.0)
    monkeypatch.setattr(exp5896, "replay_row_certificate", lambda row: {"ok": False})
    with pytest.raises(exp5896.ConstraintIRReplayError, match="certificate replay"):
        exp5896.replay_artifact(root=tmp_path)

    monkeypatch.undo()
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    artifact["reproducibility_checksum"] = "sha256:bad"
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(exp5896.ConstraintIRReplayError, match="checksum"):
        exp5896.replay_artifact(root=tmp_path)
