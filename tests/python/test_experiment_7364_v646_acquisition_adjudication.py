"""Tests for REQ-REPORT-7364 archived acquisition adjudication."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7364_v646_acquisition_adjudication as exp


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def source_artifact() -> dict[str, Any]:
    """Load the immutable historical record used by each reconciliation test."""

    return json.loads((ROOT / exp.SOURCE_ARTIFACT_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def source_rows() -> list[dict[str, Any]]:
    """Load the separate row sidecar so tests do not trust the inline copy."""

    payload = json.loads((ROOT / exp.SOURCE_ROWS_PATH).read_text(encoding="utf-8"))
    return payload["rows"]


@pytest.fixture(scope="module")
def measured_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Build one cold archive copy for artifact and validator assertions."""

    raw_dir = tmp_path_factory.mktemp("exp7364-cold")
    return exp.build_artifact(ROOT, raw_dir)


def _passing_receipt(name: str) -> dict[str, object]:
    return {
        "name": name,
        "command": f"command {name}",
        "command_argv": ["command", name],
        "scope": "test",
        "exit_code": 0,
        "duration_s": 0.1,
        "log_path": f"{name}.log",
        "log_sha256": "sha256:" + "0" * 64,
        "passed": True,
        "timed_out": False,
        "output_tail": "ok",
    }


def test_scenario_report_7364_gates_require_exact_exp7358_fields() -> None:
    """SCENARIO-REPORT-7364-GATES rejects each ineligible producer state."""

    producer = json.loads((ROOT / exp.VALIDATION_CONTRACT_PATH).read_text(encoding="utf-8"))
    assert all(row["passed"] for row in exp.check_upstream_contract(producer, True, False))

    mutations = (
        ("validation_contract_ready_score", 0),
        ("verdict_class", "disqualified"),
        ("flagged_adversarial", True),
    )
    for field, value in mutations:
        changed = deepcopy(producer)
        changed[field] = value
        failed = [
            row for row in exp.check_upstream_contract(changed, True, False) if not row["passed"]
        ]
        assert any(row["artifact_field"] == field for row in failed)

    assert exp.check_upstream_contract({}, False, False)[0]["observed"] == "missing"
    assert any(not row["passed"] for row in exp.check_upstream_contract(producer, True, True))


def test_scenario_report_7364_gates_block_without_dependent_rows() -> None:
    """SCENARIO-REPORT-7364-GATES makes external absence terminal and row-free."""

    checks = exp.check_upstream_contract({}, False, False)
    artifact = exp.build_blocked_artifact(checks, {}, 0.25)
    assert artifact["status"] == "blocked_acquisition_adjudication_precondition"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["paired_cost_rows"] == []
    assert artifact["validation_receipts"] == []
    assert artifact["acquisition_adjudication_complete_score"] == 0
    assert artifact["gate_check_summary"]["first_failure"]["artifact_field"] == "path"
    assert exp.validate_artifact(artifact, require_current_validation=False) == []


def test_scenario_report_7364_reconcile_enumerates_all_archived_units(
    source_artifact: dict[str, Any], source_rows: list[dict[str, Any]]
) -> None:
    """SCENARIO-REPORT-7364-RECONCILE counts each panel, arm, and development row."""

    signatures = exp.evidence_signatures(source_rows)
    result = exp.reconcile_archived_evidence(source_artifact, source_rows, signatures)
    assert result["errors"] == []
    assert result["row_accounting"] == {
        "evaluation_rows": 90,
        "evaluation_contexts": 30,
        "challenge_rows": 36,
        "challenge_contexts": 12,
        "development_records": 12,
        "arms_per_context": 3,
    }
    assert len(result["paired_cost_rows"]) == 42
    assert all(set(row["arms"]) == set(exp.ARMS) for row in result["paired_cost_rows"])
    assert sum(row["included_in_scientific_interval"] for row in result["paired_cost_rows"]) == 30
    reduction = result["independent_reduction"]
    assert reduction["complete_cost_ratio_ci95"]["upper"] == pytest.approx(
        source_artifact["independent_reduction"]["paired_context_clustered_ci95"]["upper"]
    )
    assert reduction["complete_cost_gate_passed"] is False
    assert reduction["returned_infeasible_count"] == 0
    assert reduction["finite_bias_coverage"] == reduction["conservative_coverage"]
    assert reduction["finite_bias_utility"] == pytest.approx(reduction["conservative_utility"])


def test_scenario_report_7364_mutations_name_changed_invariant(
    source_artifact: dict[str, Any], source_rows: list[dict[str, Any]]
) -> None:
    """SCENARIO-REPORT-7364-MUTATIONS catches cost, paid-query, and identity drift."""

    signatures = exp.evidence_signatures(source_rows)
    cases = (
        ("full_cost_s", lambda rows: rows[0].__setitem__("full_cost_s", 99.0), "complete_costs"),
        (
            "query_count",
            lambda rows: rows[1].__setitem__("query_count", rows[1]["query_count"] - 1),
            "paid_queries",
        ),
        (
            "context_id",
            lambda rows: rows[2].__setitem__("context_id", "changed-context"),
            "context_ids",
        ),
    )
    for _name, mutate, expected in cases:
        changed = deepcopy(source_rows)
        mutate(changed)
        result = exp.reconcile_archived_evidence(source_artifact, changed, signatures)
        assert expected in result["errors"]

    missing_arm = deepcopy(source_rows[:-1])
    result = exp.reconcile_archived_evidence(source_artifact, missing_arm, signatures)
    assert {"row_copy", "challenge_rows", "arm_sets"} <= set(result["errors"])


def test_scenario_report_7364_discrepancy_traces_broad_receipt(
    source_artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-7364-DISCREPANCY keeps the timed-out required command."""

    discrepancy = exp.diagnose_original_validation(source_artifact)
    assert discrepancy["supported"] is True
    assert discrepancy["original_required_checks_passed"] is False
    assert discrepancy["scoped_error_lists"] == {
        "missing_required_commands": [],
        "failed_required_commands": [],
        "duplicate_required_commands": [],
    }
    broad = discrepancy["causal_receipt"]
    assert broad["name"] == "full_python_suite"
    assert broad["exit_code"] == -15
    assert broad["timed_out"] is True
    assert broad["passed"] is False
    assert broad["affects_required_checks"] is True

    missing = deepcopy(source_artifact)
    missing["repository_health"].pop("current_observation")
    assert exp.diagnose_original_validation(missing)["supported"] is False


def test_scenario_report_7364_authority_bounds_exact_and_approximate_feedback() -> None:
    """SCENARIO-REPORT-7364-AUTHORITY limits support to the archived vocabulary."""

    rows = exp.build_authority_support_rows()
    assert {row["constraint_family"] for row in rows} == {
        "archived_finite_pairwise_relations",
        "cca_pairwise_separation",
        "cca_global_capacity",
        "higher_order_constraints",
        "t_oracle_fastca_finite_bias",
        "neural_oracle_approximation",
    }
    exact = next(
        row for row in rows if row["constraint_family"] == "archived_finite_pairwise_relations"
    )
    assert exact["authority_source"] == "exact_private_executor_boolean_feedback"
    assert exact["supported"] is True
    assert exact["actual_vocabulary"] == list(exp.FINITE_RELATION_VOCABULARY)
    assert all(row["claim_extends_beyond_actual_vocabulary"] is False for row in rows)
    assert (
        next(row for row in rows if row["constraint_family"] == "higher_order_constraints")[
            "supported"
        ]
        is False
    )
    assert (
        next(row for row in rows if row["constraint_family"] == "neural_oracle_approximation")[
            "authority_source"
        ]
        == "approximate_learned_labels_not_used_by_exp7351"
    )


def test_req_report_7364_hash_copies_every_source_file(
    measured_artifact: dict[str, Any], tmp_path: Path
) -> None:
    """REQ-REPORT-7364 binds every copied artifact, row file, sidecar, and receipt."""

    inventory = measured_artifact["archive_copy_inventory"]
    expected_count = 1 + len(list((ROOT / exp.SOURCE_RAW_DIR).rglob("*")))
    expected_count -= len([p for p in (ROOT / exp.SOURCE_RAW_DIR).rglob("*") if not p.is_file()])
    assert len(inventory) == expected_count
    assert all(row["source_sha256"] == row["copy_sha256"] for row in inventory)
    assert any(
        row["source_path"].endswith("validation/full_suite/00_full_python_suite.log")
        for row in inventory
    )
    assert measured_artifact["archive_copy_complete"] is True

    one_source = ROOT / exp.SOURCE_ARTIFACT_PATH
    copied = exp.hash_copy_files(ROOT, [one_source], tmp_path)
    assert len(copied) == 1
    assert copied[0]["source_sha256"] == copied[0]["copy_sha256"]


def test_scenario_report_7364_terminal_keeps_null_science_and_disqualification(
    measured_artifact: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-7364-TERMINAL separates complete accounting from value."""

    artifact = deepcopy(measured_artifact)
    artifact["validation_receipts"] = [
        _passing_receipt(name) for name in (*exp.REQUIRED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]
    artifact["required_checks_passed"] = True
    artifact["flagged_adversarial"] = False
    exp.apply_terminal_state(artifact, require_terminal=True)
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert artifact["status"] == "complete_acquisition_adjudication_null"
    assert artifact["verdict_class"] == "null"
    assert artifact["acquisition_adjudication_complete_score"] == 1
    assert artifact["acquisition_readiness_score"] == 0
    assert artifact["acquisition_value_score"] == 0
    assert artifact["promotion_score"] == 0
    assert artifact["original_disposition"]["verdict_class"] == "disqualified"
    assert artifact["unchanged_method_disposition"]["retired"] is True
    assert exp.validate_artifact(artifact, require_current_validation=True) == []

    failed = deepcopy(artifact)
    failed["validation_receipts"][0]["passed"] = False
    exp.apply_terminal_state(failed, require_terminal=True)
    assert failed["verdict_class"] == "disqualified"
    assert failed["acquisition_adjudication_complete_score"] == 0


def test_req_report_7364_validator_rejects_identity_inference_and_evidence_mutations(
    measured_artifact: dict[str, Any],
) -> None:
    """REQ-REPORT-7364 rejects fabricated identity, inference, scores, and checksums."""

    artifact = deepcopy(measured_artifact)
    exp.apply_terminal_state(artifact, require_terminal=False)
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    assert exp.validate_artifact(artifact, require_current_validation=False) == []
    mutations = (
        ("schema", "wrong", "identity"),
        ("MODEL_SPECS", ["model"], "model_contract"),
        ("model_invoked", True, "model_contract"),
        ("invocation_counts", {}, "invocation_counts"),
        ("inference_substrate_class", "gpu", "substrate_class"),
        ("execution_venue", "board", "venue"),
        ("verdict_class", "unknown", "verdict_class"),
        ("promotion_score", 1, "promotion"),
        ("original_disposition", {"verdict_class": "positive"}, "original_disposition"),
    )
    for field, value, expected in mutations:
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected in exp.validate_artifact(changed, require_current_validation=False)

    changed = deepcopy(artifact)
    changed["rows"][0]["full_cost_s"] = 99.0
    assert "reconciliation" in exp.validate_artifact(changed, require_current_validation=False)
    assert "reproducibility_checksum" in exp.validate_artifact(
        changed, require_current_validation=False
    )


def test_req_report_7364_uses_exp7358_scoped_command_boundary(tmp_path: Path) -> None:
    """REQ-REPORT-7364 builds only the explicit affected Exp7358 command plan."""

    commands = exp.scoped_command_plan(ROOT, tmp_path / "private")
    assert {row.name for row in commands} == set(exp.REQUIRED_CHECK_NAMES)
    joined = "\n".join(" ".join(row.argv) for row in commands)
    assert exp.TEST_PATH.as_posix() in joined
    assert "pytest tests/python -q" not in joined
    assert exp.validate_scoped_command_plan(ROOT, commands) == []
    assert any("--basetemp=" in argument for row in commands for argument in row.argv)


def test_req_report_7364_runner_and_cli_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, measured_artifact: dict[str, Any]
) -> None:
    """REQ-REPORT-7364 keeps runners bounded and the executable entrypoint thin."""

    commands = exp.scoped_command_plan(ROOT, tmp_path / "private")
    fake_rows = [_passing_receipt(command.name) for command in commands]
    fake_rows[0]["resolved_imports"] = {
        "carnot.experiment_7364_v646_acquisition_adjudication": str(ROOT / exp.MODULE_PATH)
    }
    monkeypatch.setattr(
        exp.validation_contract, "run_categorized_commands", lambda *a, **k: fake_rows
    )
    affected = exp.run_affected_validation(ROOT, tmp_path / "raw")
    assert affected["required_checks_passed"] is True

    monkeypatch.setattr(
        exp.validation_scope,
        "run_commands",
        lambda *a, **k: [_passing_receipt(command.name) for command in a[1]],
    )
    terminal = exp.run_terminal_validation(ROOT, tmp_path / "candidate.json", tmp_path / "raw")
    assert [row["name"] for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)

    with pytest.raises(SystemExit, match="requires --date"):
        exp.main(["--date", "wrong"])

    validate_path = tmp_path / "validate.json"
    validate_path.write_text("{}", encoding="utf-8")
    assert exp.main(["--date", exp.RUN_DATE, "--validate", str(validate_path)]) == 1

    blocked = exp.build_blocked_artifact(exp.check_upstream_contract({}, False, False), {}, 0.1)
    monkeypatch.setattr(
        exp, "collect_preconditions", lambda root: (blocked["preconditions_checked"], {})
    )
    blocked_root = tmp_path / "blocked-root"
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(blocked_root)]) == 0
    assert (blocked_root / exp.RESULT_PATH).is_file()

    artifact = deepcopy(measured_artifact)
    monkeypatch.setattr(
        exp, "collect_preconditions", lambda root: (artifact["preconditions_checked"], {})
    )
    monkeypatch.setattr(exp, "build_artifact", lambda root, raw: deepcopy(artifact))
    monkeypatch.setattr(
        exp,
        "run_affected_validation",
        lambda root, raw: {
            "validation_receipts": [_passing_receipt(name) for name in exp.REQUIRED_CHECK_NAMES],
            "required_checks_passed": True,
            "missing_required_commands": [],
            "failed_required_commands": [],
            "duplicate_required_commands": [],
        },
    )
    monkeypatch.setattr(
        exp,
        "run_terminal_validation",
        lambda root, candidate, raw: [_passing_receipt(name) for name in exp.TERMINAL_CHECK_NAMES],
    )
    success_root = tmp_path / "success-root"
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(success_root)]) == 0
    written = json.loads((success_root / exp.RESULT_PATH).read_text(encoding="utf-8"))
    assert written["acquisition_adjudication_complete_score"] == 1

    failed_terminal = [_passing_receipt(name) for name in exp.TERMINAL_CHECK_NAMES]
    failed_terminal[1]["passed"] = False
    failed_terminal[1]["exit_code"] = 1
    monkeypatch.setattr(
        exp, "run_terminal_validation", lambda root, candidate, raw: failed_terminal
    )
    flagged_root = tmp_path / "flagged-root"
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(flagged_root)]) == 0
    flagged = json.loads((flagged_root / exp.RESULT_PATH).read_text(encoding="utf-8"))
    assert flagged["flagged_adversarial"] is True
    assert flagged["verdict_class"] == "disqualified"

    monkeypatch.setattr(
        exp,
        "run_terminal_validation",
        lambda root, candidate, raw: [_passing_receipt(name) for name in exp.TERMINAL_CHECK_NAMES],
    )
    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(exp.AdjudicationError, match="terminal_artifact_invalid:forced"):
        exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path / "invalid-root")])


def test_req_report_7364_fail_closed_utility_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_artifact: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> None:
    """REQ-REPORT-7364 covers malformed bytes, missing receipts, and invalid plans."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    assert exp._load_object(non_object) == {}
    assert exp.validate_artifact([], require_current_validation=False) == ["artifact_object"]

    external_source = tmp_path / "outside.bin"
    external_source.write_bytes(b"archive")
    copied = exp.hash_copy_files(ROOT, [external_source], tmp_path / "external-copy")
    assert copied[0]["source_path"].startswith("external/000_")

    bad_development = deepcopy(source_artifact)
    bad_development["development_controls"]["rows"].pop()
    reconciled = exp.reconcile_archived_evidence(
        bad_development, source_rows, exp.evidence_signatures(source_rows)
    )
    assert "development_records" in reconciled["errors"]

    missing_receipt = deepcopy(source_artifact)
    missing_receipt["validation_receipts"].pop()
    errors = exp._receipt_integrity(ROOT, missing_receipt)
    assert "historical_receipt_inventory" in errors
    changed_log = deepcopy(source_artifact)
    changed_log["validation_receipts"][0]["log_sha256"] = "sha256:" + "f" * 64
    assert "historical_receipt_log:worktree_imports" in exp._receipt_integrity(ROOT, changed_log)

    blocked_checks = exp.check_upstream_contract({}, False, False)
    monkeypatch.setattr(exp, "collect_preconditions", lambda root: (blocked_checks, {}))
    built = exp.build_artifact(ROOT, tmp_path / "blocked-build")
    assert built["verdict_class"] == "blocked"
    exp.apply_terminal_state(built, require_terminal=True)
    assert built["verdict_class"] == "blocked"

    monkeypatch.setattr(
        exp.validation_contract,
        "validate_command_plan",
        lambda *args, **kwargs: ["forced_plan_error"],
    )
    with pytest.raises(exp.AdjudicationError, match="invalid_scoped_command_plan"):
        exp.scoped_command_plan(ROOT, tmp_path / "bad-plan")


def test_req_report_7364_unsupported_historical_receipt_blocks_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7364 names absent historical proof without clearing history."""

    original = exp.diagnose_original_validation

    def unsupported(value: dict[str, Any]) -> dict[str, Any]:
        result = original(value)
        result["supported"] = False
        return result

    monkeypatch.setattr(exp, "diagnose_original_validation", unsupported)
    artifact = exp.build_artifact(ROOT, tmp_path / "unsupported-receipt")
    assert "original_validation_receipt" in artifact["reconciliation_errors"]
    assert artifact["acquisition_adjudication_complete_score"] == 0
