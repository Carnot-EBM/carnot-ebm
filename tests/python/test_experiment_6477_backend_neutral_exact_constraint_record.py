"""Tests for Exp6477 backend-neutral exact constraint records.

Spec refs: REQ-VERIFY-6477, SCENARIO-VERIFY-6477-SCHEMA,
SCENARIO-VERIFY-6477-BACKEND-PARITY, SCENARIO-VERIFY-6477-ATTACKS,
SCENARIO-VERIFY-6477-ROWS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6477_backend_neutral_exact_constraint_record as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_tests() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _with_checksum(artifact: dict[str, object]) -> dict[str, object]:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def test_req_verify_6477_spec_declares_fields_and_scenarios() -> None:
    """REQ-VERIFY-6477: OpenSpec owns the exact-record contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-6477") :]
    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-VERIFY-6477-SCHEMA",
        "SCENARIO-VERIFY-6477-BACKEND-PARITY",
        "SCENARIO-VERIFY-6477-ATTACKS",
        "SCENARIO-VERIFY-6477-ROWS",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_verify_6477_schema_rejects_unsupported_records() -> None:
    """SCENARIO-VERIFY-6477-SCHEMA: unsupported semantics fail closed."""

    schema = mod.constraint_record_schema_and_hash()
    assert schema["schema_version"] == mod.RECORD_SCHEMA_VERSION
    assert schema["schema_sha256"].startswith("sha256:")
    assert "all_different" in schema["supported_constraint_kinds"]

    cases = {case.case_id: case for case in mod.immutable_cases()}
    assert mod.validate_record(cases["sat_linear_all_different"]) == []

    duplicate = deepcopy(cases["sat_linear_all_different"])
    duplicate.constraints = (
        duplicate.constraints[0],
        mod.ConstraintSpec(
            constraint_id=duplicate.constraints[0].constraint_id,
            expr=duplicate.constraints[1].expr,
        ),
    )
    assert "duplicate_constraint_id:c_sum_eq_two" in mod.validate_record(duplicate)

    invalid_rows = mod.build_unsupported_operation_rows()
    assert {row["operation_id"] for row in invalid_rows} == {
        "unsupported_nonlinear_multiply",
        "ambiguous_integer_to_boolean",
        "duplicate_constraint_ids",
        "overflow_domain",
        "unknown_variable",
        "duplicate_objective_ids",
    }
    assert all(row["rejected_before_translation"] is True for row in invalid_rows)
    assert all(row["backend_result_trusted"] is False for row in invalid_rows)

    with pytest.raises(mod.UnsupportedRecordError):
        mod.z3_backend_solve(mod.unsupported_record_fixtures()["ambiguous_integer_to_boolean"])


def test_scenario_verify_6477_validator_edge_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-6477-SCHEMA: validation branches name exact failures."""

    assert mod.and_(mod.bool_var("b")).op == "and"
    assert mod._compare(2, "gt", 1) is True
    with pytest.raises(mod.UnsupportedRecordError):
        mod._compare(1, "bad", 0)
    with pytest.raises(mod.UnsupportedRecordError):
        mod.eval_bool(mod.BoolExpr("linear_compare"), {"x": 0})
    with pytest.raises(mod.UnsupportedRecordError):
        mod.eval_bool(mod.BoolExpr("xor"), {"x": 0})
    with pytest.raises(mod.UnsupportedRecordError):
        mod._z3_bool(mod.BoolExpr("xor"), {"x": mod.z3.Int("x")})

    bool_record = mod.ConstraintRecord(
        case_id="bool-domain-check",
        case_kind="edge",
        seed=1,
        variables=(mod.FiniteDomainVar("b", 0, 1, kind="bool"),),
        constraints=(mod.ConstraintSpec("c_b", mod.bool_var("b")),),
    )
    assert mod.assignment_domain_valid(bool_record, {}) is False
    assert mod.assignment_domain_valid(bool_record, {"b": 2}) is False
    assert mod.assignment_domain_valid(bool_record, {"b": 1}) is True
    malformed_bool_domain = mod.ConstraintRecord(
        case_id="malformed-bool-domain",
        case_kind="edge",
        seed=11,
        variables=(mod.FiniteDomainVar("b", 0, 2, kind="bool"),),
        constraints=(mod.ConstraintSpec("c_b", mod.bool_var("b")),),
    )
    assert mod.assignment_domain_valid(malformed_bool_domain, {"b": 2}) is False

    protected_case = {case.case_id: case for case in mod.immutable_cases()}[
        "protected_clause_unsat"
    ]
    assert mod.protected_violations(protected_case, {"p": 0, "q": 0}) == [
        "c_protected_or"
    ]

    z3_and_gt = mod.ConstraintRecord(
        case_id="z3-and-gt",
        case_kind="edge",
        seed=2,
        variables=(mod.FiniteDomainVar("x", 0, 2), mod.FiniteDomainVar("b", 0, 1, kind="bool")),
        constraints=(
            mod.ConstraintSpec(
                "c_and_gt",
                mod.and_(mod.cmp(mod.lin({"x": 1}), "gt", 0), mod.bool_var("b")),
            ),
        ),
        objective_terms=(mod.ObjectiveTerm("o_x", mod.lin({"x": 1}), 1),),
    )
    assert mod.z3_backend_solve(z3_and_gt)["row"]["selected_assignment"] == {
        "x": 1,
        "b": 1,
    }

    invalid = mod.ConstraintRecord(
        case_id="edge-invalid",
        case_kind="edge",
        seed=3,
        schema_version="bad",
        variables=(
            mod.FiniteDomainVar("", 0, 0, kind="float", role="bad"),
            mod.FiniteDomainVar("", True, 1),
            mod.FiniteDomainVar("empty", 2, 1),
            mod.FiniteDomainVar("bad_bool", 0, 2, kind="bool"),
        ),
        constraints=(
            mod.ConstraintSpec("", "not-a-bool"),  # type: ignore[arg-type]
            mod.ConstraintSpec("c_bad_weight", mod.BoolExpr("xor"), weight="bad"),  # type: ignore[arg-type]
            mod.ConstraintSpec("c_zero_weight", mod.BoolExpr("and"), weight=0),
            mod.ConstraintSpec(
                "c_bad_compare",
                mod.BoolExpr(
                    "linear_compare",
                    expr=mod.lin({"empty": 1.5, "missing": mod.EXACT_INT_BOUND + 1}, mod.EXACT_INT_BOUND + 1),  # type: ignore[arg-type]
                    compare_op="between",
                    rhs=True,  # type: ignore[arg-type]
                ),
            ),
            mod.ConstraintSpec(
                "c_rhs_overflow",
                mod.BoolExpr(
                    "linear_compare",
                    expr=mod.lin({"empty": 1}),
                    compare_op="eq",
                    rhs=mod.EXACT_INT_BOUND + 1,
                ),
            ),
            mod.ConstraintSpec("c_bool_missing", mod.bool_var("missing_bool")),
            mod.ConstraintSpec("c_all_diff_bad", mod.all_different("missing_only")),
            mod.ConstraintSpec("c_bad_not", mod.BoolExpr("not")),
        ),
        objective_terms=(
            mod.ObjectiveTerm("", mod.lin({"empty": 1}), "bad"),  # type: ignore[arg-type]
            mod.ObjectiveTerm("o_zero", mod.lin({"empty": 1}), 0),
            mod.ObjectiveTerm("o_bad_expr", None, 1),  # type: ignore[arg-type]
        ),
    )
    errors = set(mod.validate_record(invalid))
    assert {
        "unsupported_schema_version:bad",
        "empty_variable_id",
        "unsupported_variable_kind:float",
        "unsupported_variable_role:bad",
        "non_integer_domain:",
        "empty_domain:empty",
        "invalid_bool_domain:bad_bool",
        "empty_constraint_id",
        "unsupported_boolean_expression",
        "unsupported_boolean_op:xor",
        "and_requires_children",
        "non_integer_constraint_weight:c_bad_weight",
        "nonpositive_constraint_weight:c_zero_weight",
        "unsupported_compare_op:between",
        "non_integer_compare_rhs",
        "overflow_linear_constant",
        "unknown_variable:missing",
        "non_integer_linear_coefficient",
        "overflow_linear_coefficient",
        "overflow_compare_rhs",
        "unknown_variable:missing_bool",
        "all_different_needs_two_variables",
        "unknown_variable:missing_only",
        "not_requires_one_child",
        "empty_objective_id",
        "non_integer_objective_weight:",
        "zero_objective_weight:o_zero",
        "unsupported_nonlinear_multiply",
    } <= errors

    duplicate_vars = mod.ConstraintRecord(
        case_id="duplicate-vars",
        case_kind="edge",
        seed=33,
        variables=(mod.FiniteDomainVar("x", 0, 1), mod.FiniteDomainVar("x", 0, 1)),
        constraints=(mod.ConstraintSpec("c_x", mod.cmp(mod.lin({"x": 1}), "ge", 0)),),
    )
    assert "duplicate_variable_id:x" in mod.validate_record(duplicate_vars)

    overflow_coefficient = mod.ConstraintRecord(
        case_id="overflow-coeff",
        case_kind="edge",
        seed=4,
        variables=(mod.FiniteDomainVar("x", 0, 1),),
        constraints=(
            mod.ConstraintSpec(
                "c_overflow_coeff",
                mod.cmp(mod.lin({"x": mod.EXACT_INT_BOUND + 1}), "eq", 0),
            ),
        ),
    )
    assert "overflow_linear_coefficient" in mod.validate_record(overflow_coefficient)

    state_budget = mod.ConstraintRecord(
        case_id="state-budget",
        case_kind="edge",
        seed=5,
        variables=(mod.FiniteDomainVar("x", 0, 300), mod.FiniteDomainVar("y", 0, 300)),
        constraints=(mod.ConstraintSpec("c_x", mod.cmp(mod.lin({"x": 1}), "ge", 0)),),
    )
    assert "exhaustive_state_budget_exceeded" in mod.validate_record(state_budget)

    assert mod._status(0.0, {"all_gates_passed": False}) == (
        "blocked_exact_constraint_record_parity"
    )
    assert mod._honest_verdict("blocked_exact_constraint_record_parity").startswith(
        "complete_blocked:"
    )
    with pytest.raises(KeyError):
        mod._case_by_id([], "missing")

    incomplete = mod.recompute_aggregates_from_rows(
        [mod.evaluate_case(mod.immutable_cases()[0])["backend_rows"][0]]
    )
    assert incomplete["all_backend_pairs_complete"] is False
    assert incomplete["exact_constraint_record_ready_score_from_rows"] == 0.0

    with monkeypatch.context() as mp:
        mp.setattr(
            mod,
            "_gate_check_summary",
            lambda **_: {
                "checks": {"forced": False},
                "all_gates_passed": False,
                "failed_gates": ["forced"],
                "mismatch_rows": ["forced"],
            },
        )
        blocked = mod.build_artifact(
            root=REPO,
            run_date="20260821",
            duration_s=0.25,
            tests_run=_passing_tests(),
        )
    assert blocked["status"] == "blocked_exact_constraint_record_parity"
    assert blocked["exact_constraint_record_ready_score"] == 0.0


def test_scenario_verify_6477_backends_match_on_immutable_cases() -> None:
    """SCENARIO-VERIFY-6477-BACKEND-PARITY: Z3 and exhaustive replay agree."""

    cases = mod.immutable_cases()
    kinds = {case.case_kind for case in cases}
    assert {
        "satisfiable",
        "unsatisfiable",
        "negated",
        "auxiliary_variable",
        "protected_clause",
        "boundary_value",
        "random_seed",
    } <= kinds
    assert len([case for case in cases if case.seed in mod.RANDOM_CASE_SEEDS]) >= 5

    saw_protected_violation = False
    saw_negated_unsat = False
    saw_auxiliary = False
    for case in cases:
        result = mod.evaluate_case(case)
        assert result["satisfiability_match"] is True
        assert result["witness_validity_match"] is True
        assert result["violation_set_match"] is True
        assert result["protected_violation_match"] is True
        assert result["objective_value_match"] is True
        assert result["scalar_energy_match"] is True
        assert {row["backend"] for row in result["backend_rows"]} == {
            "z3",
            "exhaustive",
        }
        for row in result["backend_rows"]:
            assert row["record_hash"] == case.record_hash()
            assert row["source_constraint_ids"] == [c.constraint_id for c in case.constraints]
            assert row["domain_assignment_valid"] is True
            if row["satisfiable"]:
                assert row["witness_valid"] is True
                assert row["violated_constraint_ids"] == []
        z3_receipts = result["translation_receipts"]["z3"]["constraint_receipts"]
        assert {receipt["constraint_id"] for receipt in z3_receipts} == {
            c.constraint_id for c in case.constraints
        }
        if case.case_id == "protected_clause_unsat":
            saw_protected_violation = True
            assert result["backend_rows"][0]["protected_violations"] == ["c_protected_or"]
        if case.case_id == "negated_unsat":
            saw_negated_unsat = True
            assert result["backend_rows"][0]["satisfiable"] is False
        if case.case_id == "auxiliary_link_sat":
            saw_auxiliary = True
            assert "aux" in result["backend_rows"][0]["selected_assignment"]

    assert saw_protected_violation
    assert saw_negated_unsat
    assert saw_auxiliary


def test_scenario_verify_6477_attack_matrix_catches_translation_risks() -> None:
    """SCENARIO-VERIFY-6477-ATTACKS: known translation risks fail closed."""

    attack_matrix = mod.build_attack_matrix(mod.immutable_cases())
    assert {row["attack_id"] for row in attack_matrix["rows"]} == set(mod.ATTACK_IDS)
    assert attack_matrix["all_attacks_detected"] is True
    assert attack_matrix["false_accept_count"] == 0

    by_id = {row["attack_id"]: row for row in attack_matrix["rows"]}
    assert by_id["dropped_negation"]["semantic_mismatch"] is True
    assert by_id["domain_widening"]["semantic_mismatch"] is True
    assert by_id["integer_to_boolean_coercion"]["rejected_before_translation"] is True
    assert by_id["auxiliary_variable_leakage"]["semantic_mismatch"] is True
    assert by_id["overflow"]["rejected_before_translation"] is True
    assert by_id["duplicate_constraint_ids"]["rejected_before_translation"] is True
    assert by_id["objective_sign_reversal"]["objective_mismatch"] is True
    assert by_id["matching_totals_different_violation_sets"]["scalar_energy_equal"] is True
    assert by_id["matching_totals_different_violation_sets"]["violation_sets_equal"] is False


def test_scenario_verify_6477_artifact_rows_recompute_and_validate(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6477-ROWS: readiness derives from row data only."""

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260821",
        duration_s=0.25,
        tests_run=_passing_tests(),
    )
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["exact_constraint_record_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["aggregate_row_recomputation"] == mod.recompute_aggregates_from_rows(
        artifact["per_unit_rows"]
    )
    assert artifact["satisfiability_parity"]["all_match"] is True
    assert artifact["witness_validity_parity"]["all_match"] is True
    assert artifact["violation_set_parity"]["all_match"] is True
    assert all(row["release_authority"] is False for row in artifact["scalar_violation_energy_rows"])
    assert set(artifact["field_provenance"]) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.validate_artifact(artifact) == []

    bad = deepcopy(artifact)
    bad["exact_constraint_record_ready_score"] = 0.0
    assert "exact_constraint_record_ready_score mismatch" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["per_unit_rows"] = bad["per_unit_rows"][:-1]
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["inference_substrate"] = "live_llm_inference"
    assert "inference_substrate mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["verifier_is_oracle"] = False
    assert "verifier_is_oracle must be true within declared finite-domain record" in (
        mod.validate_artifact(_with_checksum(bad))
    )

    bad = deepcopy(artifact)
    bad["field_principles"] = {}
    assert "missing field_principles entry: status" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["field_provenance"] = {}
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["honest_verdict"] = "done"
    assert "honest_verdict lacks required terminal prefix" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    del bad["status"]
    assert "missing required field: status" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["satisfiability_parity"] = {}
    assert "satisfiability_parity mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["witness_validity_parity"] = {}
    assert "witness_validity_parity mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["violation_set_parity"] = {}
    assert "violation_set_parity mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["attack_matrix"]["all_attacks_detected"] = False
    assert "attack matrix must detect every attack" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["unsupported_operation_rows"][0]["backend_result_trusted"] = True
    assert "unsupported operation row trusted a backend" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["protected_files_unchanged"]["unchanged"] = False
    assert "protected files changed" in mod.validate_artifact(_with_checksum(bad))

    path = tmp_path / "artifact.json"
    mod.write_artifact(artifact, path)
    assert json.loads(path.read_text(encoding="utf-8")) == artifact


def test_req_verify_6477_cli_and_dependency_edges(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6477: CLI writes and validates the terminal artifact."""

    original_version = mod.metadata.version

    def fake_version(name: str) -> str:
        if name == "missing-package":
            raise mod.metadata.PackageNotFoundError(name)
        return original_version(name)

    with monkeypatch.context() as mp:
        mp.setattr(mod.metadata, "version", fake_version)
        assert mod._package_version("missing-package") == "not_installed"

    result = tmp_path / "experiment_6477.json"
    artifact = mod.run(
        date="20260821",
        result_path=result,
        test_exit_codes=_passing_tests(),
    )
    assert json.loads(result.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"

    assert mod.main(["--date", "20260821", "--result-path", str(result)]) == 0
    written = json.loads(result.read_text(encoding="utf-8"))
    assert written["status"] == "complete"

    assert mod.main(["--validate", "--result-path", str(result)]) == 0
    validate_out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert validate_out["ok"] is True

    missing = tmp_path / "missing.json"
    assert mod.main(["--validate", "--result-path", str(missing)]) == 1
    missing_out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert missing_out["ok"] is False
    assert missing_out["errors"] == ["artifact missing"]
