"""Tests for Exp6478 identifiable held exact-energy selection.

Spec refs: REQ-VERIFY-6478, SCENARIO-VERIFY-6478-GATES,
SCENARIO-VERIFY-6478-PRECOMMITMENT, SCENARIO-VERIFY-6478-MATCHED-SELECTION,
SCENARIO-VERIFY-6478-ROWS, SCENARIO-VERIFY-6478-ATTACKS.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6478_identifiable_held_exact_energy_selection as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _passing_tests() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _with_checksum(artifact: dict[str, object]) -> dict[str, object]:
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)
    return artifact


def test_req_verify_6478_spec_declares_fields_and_scenarios() -> None:
    """REQ-VERIFY-6478: OpenSpec owns the held selector contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-6478") :]
    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-VERIFY-6478-GATES",
        "SCENARIO-VERIFY-6478-PRECOMMITMENT",
        "SCENARIO-VERIFY-6478-MATCHED-SELECTION",
        "SCENARIO-VERIFY-6478-ROWS",
        "SCENARIO-VERIFY-6478-ATTACKS",
        mod.INFERENCE_SUBSTRATE,
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES


def test_scenario_verify_6478_gates_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6478-GATES: failed upstream gates block held opening."""

    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / mod.UPSTREAM_IDENTIFIABILITY_RELATIVE_PATH.name).write_text(
        json.dumps({"protocol_identifying_score": 0.0}),
        encoding="utf-8",
    )
    (result_dir / mod.UPSTREAM_CONSTRAINT_RELATIVE_PATH.name).write_text(
        json.dumps({"exact_constraint_record_ready_score": 1.0}),
        encoding="utf-8",
    )

    artifact = mod.build_artifact(
        root=tmp_path,
        run_date="20260821",
        duration_s=0.25,
        tests_run=_passing_tests(),
    )
    assert artifact["status"] == "blocked_gate_check_failed"
    assert artifact["held_exact_energy_selection_ready_score"] == 0.0
    assert artifact["per_unit_rows"] == []
    assert artifact["gate_check_summary"]["all_gates_passed"] is False
    failed = artifact["gate_check_summary"]["failed_checks"][0]
    assert failed["field"] == "protocol_identifying_score"
    assert failed["operator"] == "=="
    assert failed["expected_value"] == 1.0
    assert failed["observed_value"] == 0.0
    assert artifact["honest_verdict"].startswith("complete_blocked:")
    assert mod.validate_artifact(artifact) == []


def test_scenario_verify_6478_precommitment_seals_solver_grounded_candidates() -> None:
    """SCENARIO-VERIFY-6478-PRECOMMITMENT: candidate bytes and labels are sealed."""

    dev_units = mod.development_units()
    held_units = mod.held_units()
    assert {unit["pattern"] for unit in held_units} >= {
        "protected_clause",
        "negation",
        "objective_conflict",
    }

    candidate_manifest = mod.build_candidate_manifest(dev_units + held_units, mod.SELECTION_SEEDS)
    precommitment = mod.build_precommitment(
        development_units=dev_units,
        held_units=held_units,
        candidate_manifest=candidate_manifest,
    )
    assert precommitment["opened_held_results_after_precommitment"] is True
    assert precommitment["candidate_manifest_hash"].startswith("sha256:")
    assert precommitment["exact_label_hash"].startswith("sha256:")
    assert precommitment["protected_weight_hash"].startswith("sha256:")
    assert precommitment["analysis_plan_hash"].startswith("sha256:")
    assert precommitment["development_weight_tuning"]["held_units_used"] == 0

    held_rows = [row for row in candidate_manifest["rows"] if row["split"] == "held"]
    assert held_rows
    assert all(row["candidate_bytes_sha256"].startswith("sha256:") for row in held_rows)
    assert all(row["exact_label_hash"].startswith("sha256:") for row in held_rows)
    by_rank = {
        row["candidate_rank"]: row
        for row in held_rows
        if row["unit_id"] == held_units[0]["unit_id"]
    }
    assert by_rank[0]["exact_success"] is False
    assert by_rank[1]["exact_success"] is True

    receipt = mod.protocol_recheck_receipt(candidate_manifest)
    assert receipt["identifying"] is True
    assert receipt["api"] == (
        "carnot.experiment_6474_protocol_identifiability_and_receipt_preflight.audit_support"
    )


def test_scenario_verify_6478_matched_selection_and_reduction() -> None:
    """SCENARIO-VERIFY-6478-MATCHED-SELECTION: arms share candidates and work."""

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260821",
        duration_s=0.25,
        tests_run=_passing_tests(),
    )
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["held_exact_energy_selection_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["aggregate_row_recomputation"] == mod.recompute_aggregates_from_rows(
        artifact["per_unit_rows"]
    )

    by_arm = {row["arm"]: row for row in artifact["exact_success_by_arm"]["rows"]}
    assert (
        by_arm["exact_energy"]["exact_success_count"]
        > by_arm["first_candidate"]["exact_success_count"]
    )
    assert (
        by_arm["exact_energy"]["exact_success_count"]
        > by_arm["shuffled_energy"]["exact_success_count"]
    )
    assert len({row["total_work_units"] for row in by_arm.values()}) == 1

    comparisons = {
        row["right_arm"]: row
        for row in artifact["paired_effects_and_intervals"]["comparisons"]
        if row["left_arm"] == "exact_energy"
    }
    assert comparisons["first_candidate"]["paired_gain"] > 0
    assert comparisons["first_candidate"]["ci_95"][0] > 0
    assert comparisons["shuffled_energy"]["paired_gain"] > 0
    assert comparisons["shuffled_energy"]["ci_95"][0] > 0

    flips = artifact["harmful_flips_and_recovered_failures"]
    assert flips["vs_first_candidate"]["harmful_flip_count"] == 0
    assert flips["vs_shuffled_energy"]["harmful_flip_count"] == 0
    assert flips["vs_first_candidate"]["recovered_failure_count"] > 0
    assert artifact["protected_clause_results"]["protected_regression_count"] == 0

    selected = [row for row in artifact["per_unit_rows"] if row["selected_by_arm"]]
    assert selected
    keys = {(row["unit_id"], row["seed"]) for row in selected}
    for unit_id, seed in keys:
        group = [
            row
            for row in artifact["per_unit_rows"]
            if row["unit_id"] == unit_id and row["seed"] == seed
        ]
        work_by_arm = mod.work_totals_by_arm(group)
        assert set(work_by_arm) == set(mod.ARMS)
        assert len(set(work_by_arm.values())) == 1

    assert mod.validate_artifact(artifact) == []


def test_scenario_verify_6478_attacks_and_validator_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-6478-ATTACKS: attacks and tampered summaries fail."""

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260821",
        duration_s=0.25,
        tests_run=_passing_tests(),
    )
    attacks = artifact["attack_matrix"]
    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_attacks_detected"] is True
    by_id = {row["attack_id"]: row for row in attacks["rows"]}
    assert by_id["held_leakage"]["uses_held_exact_labels"] is True
    assert by_id["result_dependent_weights"]["precommitment_hash_changed"] is True
    assert by_id["shuffled_labels"]["label_mismatch_count"] > 0
    assert by_id["tie_manipulation"]["tie_broken_by_hash"] is True
    assert by_id["energy_sign_reversal"]["selected_exact_success"] is False
    assert by_id["matched_totals_different_protected_violations"]["same_work_total"] is True
    assert by_id["aggregate_mismatch"]["stored_matches_rows"] is False

    bad = deepcopy(artifact)
    bad["held_exact_energy_selection_ready_score"] = 0.0
    assert "held_exact_energy_selection_ready_score mismatch" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["per_unit_rows"] = bad["per_unit_rows"][:-1]
    assert "aggregate_row_recomputation mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["exact_success_by_arm"] = {"rows": []}
    assert "exact_success_by_arm mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["attack_matrix"]["all_attacks_detected"] = False
    assert "attack matrix must detect every attack" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["inference_substrate"] = "live_llm_inference"
    assert "inference_substrate mismatch" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["verifier_is_oracle"] = False
    assert "verifier_is_oracle must be true for exact backend and row arithmetic" in (
        mod.validate_artifact(_with_checksum(bad))
    )

    bad = deepcopy(artifact)
    bad["field_principles"] = {}
    assert "missing field_principles entry: status" in mod.validate_artifact(_with_checksum(bad))

    bad = deepcopy(artifact)
    bad["field_provenance"] = {}
    assert "field_provenance must cover exactly required fields" in mod.validate_artifact(
        _with_checksum(bad)
    )

    bad = deepcopy(artifact)
    bad["protected_files_unchanged"]["unchanged"] = False
    assert "protected files changed" in mod.validate_artifact(_with_checksum(bad))

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

    with monkeypatch.context() as mp:
        mp.setattr(
            mod.metadata,
            "version",
            lambda name: (_ for _ in ()).throw(mod.metadata.PackageNotFoundError(name)),
        )
        assert mod._package_version("missing-package") == "not_installed"


def test_req_verify_6478_fail_closed_edge_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-6478: local fail-closed helpers cover malformed edges."""

    missing_gates = mod.check_upstream_gates(tmp_path)
    assert missing_gates["all_gates_passed"] is False
    assert all(row["path_exists"] is False for row in missing_gates["checks"])
    assert mod._status(0.0, {"upstream_gates_passed": False}) == ("blocked_gate_check_failed")
    assert mod._status(0.0, {"upstream_gates_passed": True, "all_gates_passed": True}) == (
        "complete_null"
    )
    assert mod._honest_verdict("complete_null").startswith("complete_null:")

    unsat = mod.exp6477.ConstraintRecord(
        case_id="unsat-edge",
        case_kind="edge",
        seed=1,
        variables=(mod.exp6477.FiniteDomainVar("x", 0, 0),),
        constraints=(mod.exp6477.ConstraintSpec("c_bad", mod._linear_cmp({"x": 1}, "gt", 0)),),
    )
    with pytest.raises(ValueError, match="unit must be satisfiable"):
        mod._valid_assignment(unsat)

    all_valid = mod.exp6477.ConstraintRecord(
        case_id="all-valid-edge",
        case_kind="edge",
        seed=2,
        variables=(mod.exp6477.FiniteDomainVar("x", 0, 0),),
        constraints=(mod.exp6477.ConstraintSpec("c_ok", mod._linear_cmp({"x": 1}, "eq", 0)),),
    )
    with pytest.raises(ValueError, match="unit has no invalid perturbation"):
        mod._invalid_assignments(all_valid, {"x": 0})

    one_invalid = mod.exp6477.ConstraintRecord(
        case_id="one-invalid-edge",
        case_kind="edge",
        seed=3,
        variables=(mod.exp6477.FiniteDomainVar("x", 0, 1),),
        constraints=(mod.exp6477.ConstraintSpec("c_x", mod._linear_cmp({"x": 1}, "eq", 1)),),
    )
    assert len(mod._candidate_assignments(one_invalid)) == 4

    rows = mod.build_candidate_manifest(mod.held_units()[:1], mod.SELECTION_SEEDS)["rows"][:4]
    with pytest.raises(ValueError, match="unknown arm"):
        mod.select_candidate(rows, "unknown")

    assert (
        mod._matched_candidate_sets([{"row_type": "attack"}])["candidate_set_mismatch_count"] == 0
    )
    assert (
        mod._no_headroom_and_ties(
            [
                {
                    "row_type": "candidate_selection",
                    "unit_id": "u",
                    "seed": 1,
                    "arm": "exact_energy",
                    "candidate_id": "a",
                    "exact_success": False,
                    "selected_by_arm": True,
                    "tie_group_size": 2,
                    "tie_candidate_ids": ["a", "b"],
                },
                {
                    "row_type": "candidate_selection",
                    "unit_id": "u",
                    "seed": 1,
                    "arm": "first_candidate",
                    "candidate_id": "b",
                    "exact_success": False,
                    "selected_by_arm": False,
                    "tie_group_size": 0,
                    "tie_candidate_ids": [],
                },
            ]
        )["no_headroom_count"]
        == 1
    )
    assert (
        mod._protected_results(
            [
                {
                    "row_type": "candidate_selection",
                    "unit_id": "u",
                    "seed": 1,
                    "arm": "exact_energy",
                    "pattern": "protected_clause",
                    "selected_by_arm": True,
                    "exact_success": True,
                    "protected_violations": [],
                },
                {
                    "row_type": "candidate_selection",
                    "unit_id": "u",
                    "seed": 1,
                    "arm": "first_candidate",
                    "pattern": "protected_clause",
                    "selected_by_arm": True,
                    "exact_success": False,
                    "protected_violations": ["c_protected_or"],
                },
            ]
        )["protected_regression_count"]
        == 0
    )

    with monkeypatch.context() as mp:
        mp.setattr(
            mod,
            "_terminal_gate_summary",
            lambda **_: {
                "checks": [],
                "all_gates_passed": False,
                "upstream_gates_passed": True,
                "failed_checks": [{"field": "forced"}],
            },
        )
        null_artifact = mod.build_artifact(
            root=REPO,
            run_date="20260821",
            duration_s=0.25,
            tests_run=_passing_tests(),
        )
    assert null_artifact["status"] == "complete_null"
    assert null_artifact["held_exact_energy_selection_ready_score"] == 0.0

    artifact = mod.build_artifact(
        root=REPO,
        run_date="20260821",
        duration_s=0.25,
        tests_run=_passing_tests(),
    )
    bad = deepcopy(artifact)
    bad["gate_check_summary"]["all_gates_passed"] = False
    assert "gate_check_summary must pass for complete status" in mod.validate_artifact(
        _with_checksum(bad)
    )


def test_req_verify_6478_run_write_and_validate_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-VERIFY-6478: CLI writes and validates the terminal artifact."""

    result = tmp_path / "experiment_6478.json"
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
