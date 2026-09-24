"""Behavior tests for the V665 terminal reconciliation.

Spec refs: REQ-REPORT-7628 and SCENARIO-REPORT-7628-CUSTODY,
SCENARIO-REPORT-7628-BRANCHES, SCENARIO-REPORT-7628-CLASSIFY,
SCENARIO-REPORT-7628-RETIREMENT, SCENARIO-REPORT-7628-TERMINAL.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from carnot import experiment_7615_v665_contract_methods as contract
from carnot import experiment_7628_v665_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def authority() -> dict:
    """Load the real V665 contract once for all custody checks."""

    return capstone.load_authority(ROOT)


@pytest.fixture(scope="module")
def dispositions(authority: dict) -> list[dict]:
    """Collect the real roster without requiring planned missing producers."""

    return capstone.collect_milestone_dispositions(ROOT, authority["tasks"])


@pytest.fixture(scope="module")
def artifact() -> dict:
    """Build a deterministic test candidate without launching subprocesses."""

    return capstone.build_artifact_for_test(ROOT)


def test_exact_authority_and_fourteen_custody_rows(
    authority: dict, dispositions: list[dict]
) -> None:
    """REQ-REPORT-7628 / SCENARIO-REPORT-7628-CUSTODY."""

    assert authority["selected_roadmap_path"] == "research-roadmap.yaml"
    assert authority["comparison_passed"] is True
    assert [row["task_id"] for row in dispositions] == list(contract.EXPECTED_TASK_IDS)
    assert [row["order"] for row in dispositions] == list(range(1, 15))
    assert capstone.disposition_counts(dispositions) == {
        "terminal_producer": 7,
        "conductor_pre_gate": 3,
        "missing_work": 3,
        "current_self": 1,
    }
    assert dispositions[3]["actual_path"] == "results/experiment_7618_fit_evidence.json"
    assert dispositions[6]["custody_kind"] == "missing_work"
    assert dispositions[-1]["custody_kind"] == "current_self"
    assert all("terminal_reader_results" in row for row in dispositions)


def test_independent_branch_reduction_uses_rows() -> None:
    """REQ-REPORT-7628 / SCENARIO-REPORT-7628-BRANCHES."""

    summary = capstone.reduce_evidence_summary(ROOT)
    assert summary["syntax_readiness"]["ready"] is False
    assert summary["calibration_benefit"]["observed"] is None
    assert summary["semantic_evidence_dependence"]["observed"] is None
    assert summary["retained_delayed_learning"]["observed"] is None
    assert summary["freshness"]["observed"] is False

    arc = summary["observational_arc_support"]
    assert arc["observed_independent_games"] == 6
    assert arc["proposed_redirects"] == 20
    assert arc["actual_firings"] == 0
    assert arc["recomputed_from_rows"] is True

    speed = summary["deployment_speed"]
    assert speed["independent_blocks"] == 120
    assert speed["python_over_direct_native_estimate"] == pytest.approx(7.8270027462)
    assert speed["python_over_direct_native_lower95"] == pytest.approx(7.2843077593)
    assert speed["benefit_gate_passed"] is True
    assert speed["nfr_10x_met"] is False
    assert speed["recomputed_from_rows"] is True


def test_external_absence_is_blocked_and_never_partial(artifact: dict) -> None:
    """REQ-REPORT-7628 / SCENARIO-REPORT-7628-CLASSIFY."""

    assert artifact["honest_verdict"] == "complete_blocked_required_v665_external_evidence"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["status"] == "complete"
    assert artifact["flagged_adversarial"] is False
    assert artifact["capstone_complete_score"] == 1
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["planned_inference_substrate_class"] == "aggregation"
    assert artifact["actual_inference_substrate_class"] == "aggregation"
    assert capstone.classify_terminal(True, True, False) == (
        "complete_blocked_required_v665_external_evidence",
        "blocked",
    )
    assert capstone.classify_terminal(False, False, False)[1] == "partial"
    assert all(
        {"check", "upstream", "path", "field", "operator", "expected", "observed"} <= row.keys()
        for row in artifact["gate_check_summary"]["failed_checks"]
    )


def test_retirement_scope_and_open_prd_gaps_are_literal(artifact: dict) -> None:
    """REQ-REPORT-7628 / SCENARIO-REPORT-7628-RETIREMENT."""

    by_name = {row["hypothesis"]: row for row in artifact["next_decisions"]}
    assert by_name["schema_transport"]["decision"] == "change"
    assert by_name["evidence_semantics"]["decision"] == "keep"
    assert by_name["guarded_delayed_learning"]["decision"] == "keep"
    assert by_name["arc_supervisor_refinement"]["decision"] == "retire"
    assert by_name["native_boundary_speed"]["decision"] == "keep"
    assert artifact["prior_failure_disposition"]["retired_scope"] == (
        "unchanged_v664_capstone_accounting_retry"
    )
    assert artifact["prior_failure_disposition"]["scientific_hypothesis_retired"] is False
    assert [row["gap"] for row in artifact["remaining_prd_gaps"]] == [
        "decision_evidence",
        "causal_retained_learning",
        "total_deployment_cost",
    ]


def test_artifact_has_required_fields_and_separate_gates(artifact: dict) -> None:
    """REQ-REPORT-7628 requires auditable field and gate principles."""

    required = set(capstone.REQUIRED_ARTIFACT_FIELDS)
    assert not required.difference(artifact)
    assert set(artifact["acceptance_gate_results"]) == {
        "validity",
        "readiness",
        "benefit",
        "retention",
        "freshness",
    }
    assert all("principle" in gate for gate in artifact["acceptance_gate_results"].values())
    assert set(artifact).issubset(artifact["field_principles"])
    assert len(artifact["milestone_dispositions"]) == 14
    assert len(artifact["rows"]) == 14
    assert artifact["sample_size_budget"]["milestone_tasks"]["observed"] == 14
    assert artifact["publication_gates"]["claim_boundary"] == (
        "historical_fover_eligibility_only_not_a_v665_claim"
    )
    self_source = next(
        row for row in artifact["source_artifact_hashes"] if row["task_id"] == "exp7628-capstone"
    )
    assert self_source == {
        "task_id": "exp7628-capstone",
        "source_kind": "current_self",
        "planned_path": "results/experiment_7628_v665_capstone.json",
        "actual_path": None,
        "sha256": None,
        "exists": False,
    }
    assert artifact["submitted_externally"] is False


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("drop_disposition", "milestone_disposition_count"),
        ("missing_producer", "custody_kind_mismatch:exp7627-native-cost"),
        ("positive_over_block", "terminal_classification_mismatch"),
        ("native_ratio", "evidence_summary_mismatch"),
        ("partial_over_external", "terminal_classification_mismatch"),
    ],
)
def test_cold_validation_rejects_required_mutations(
    artifact: dict, mutation: str, expected_error: str
) -> None:
    """REQ-REPORT-7628 / SCENARIO-REPORT-7628-TERMINAL."""

    changed = capstone.mutate_for_test(deepcopy(artifact), mutation)
    assert expected_error in capstone.validate_artifact(changed, root=ROOT, require_terminal=False)


def test_clean_test_artifact_cold_reduces(artifact: dict) -> None:
    """REQ-REPORT-7628 exact fourteen-disposition replay stays stable."""

    assert capstone.validate_artifact(artifact, root=ROOT, require_terminal=False) == []
    assert capstone.independent_reduce(artifact, root=ROOT) == []
    assert capstone.date_argument("20260924") == "20260924"
    assert capstone.root_argument(str(ROOT)) == ROOT
    with pytest.raises(ValueError):
        capstone.date_argument("20260923")
    with pytest.raises(ValueError):
        capstone.root_argument(".")


def test_defensive_reducer_inputs_fail_closed(
    artifact: dict, authority: dict, dispositions: list[dict], tmp_path: Path, monkeypatch
) -> None:
    """REQ-REPORT-7628 malformed private inputs never become evidence."""

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        capstone.load_json(scalar)

    monkeypatch.setattr(
        contract,
        "resolve_v665_roadmap",
        lambda _root: (ROOT / "research-roadmap.yaml", {"tasks": "bad"}, []),
    )
    monkeypatch.setattr(contract, "compare_contract_authorities", lambda *_args: {"passed": False})
    with pytest.raises(ValueError, match="task list"):
        capstone.load_authority(ROOT)
    monkeypatch.undo()

    bad_tasks = deepcopy(authority["tasks"])
    bad_tasks[0], bad_tasks[1] = bad_tasks[1], bad_tasks[0]
    with pytest.raises(ValueError, match="authority order"):
        capstone.collect_milestone_dispositions(ROOT, bad_tasks)
    with pytest.raises(ValueError, match="Exp7625 rows"):
        capstone._arc_reduction({})
    with pytest.raises(ValueError, match="Exp7627 paired"):
        capstone._native_reduction({})
    with pytest.raises(ValueError, match="unknown mutation"):
        capstone.mutate_for_test(deepcopy(artifact), "not-a-mutation")

    assert capstone.classify_terminal(True, False, True)[1] == "positive"
    assert capstone.classify_terminal(True, False, False)[1] == "null"
    assert capstone.validate_artifact([], root=ROOT) == ["artifact_object_required"]

    rebuilt = capstone.build_artifact(
        ROOT,
        authority,
        dispositions,
        capstone.reduce_evidence_summary(ROOT),
        artifact["publication_gates"],
        artifact["preconditions_checked"],
        terminal_reader_outcomes={"reader": {"passed": True}},
    )
    assert rebuilt["milestone_dispositions"][-1]["terminal_reader_results"] == {
        "reader": {"passed": True}
    }


def test_validator_reports_exact_shape_and_provenance_failures(artifact: dict, monkeypatch) -> None:
    """SCENARIO-REPORT-7628-TERMINAL validates stored operands by name."""

    changed = deepcopy(artifact)
    changed["milestone_dispositions"][0], changed["milestone_dispositions"][1] = (
        changed["milestone_dispositions"][1],
        changed["milestone_dispositions"][0],
    )
    assert "milestone_disposition_order" in capstone.validate_artifact(
        changed, root=ROOT, require_terminal=False
    )

    monkeypatch.setattr(
        capstone, "load_authority", lambda _root: (_ for _ in ()).throw(ValueError())
    )
    assert "authority_reload_failed:ValueError" in capstone.validate_artifact(
        artifact, root=ROOT, require_terminal=False
    )
    monkeypatch.undo()

    cases = []
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = "bad"
    cases.append((changed, "source_artifact_hashes_shape"))
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"][0] = "bad"
    cases.append((changed, "source_artifact_hash_shape"))
    changed = deepcopy(artifact)
    missing = next(row for row in changed["source_artifact_hashes"] if row["actual_path"] is None)
    missing["exists"] = True
    cases.append((changed, f"missing_source_state:{missing['task_id']}"))
    changed = deepcopy(artifact)
    present = next(row for row in changed["source_artifact_hashes"] if row["actual_path"])
    present["sha256"] = "sha256:bad"
    cases.append((changed, f"source_hash_mismatch:{present['actual_path']}"))
    changed = deepcopy(artifact)
    del changed["rows"]
    cases.append((changed, "required_field_missing:rows"))
    changed = deepcopy(artifact)
    changed["gate_check_summary"] = {}
    cases.append((changed, "gate_check_summary_mismatch"))
    changed = deepcopy(artifact)
    changed["acceptance_gate_results"] = {}
    cases.append((changed, "acceptance_gate_results_mismatch"))
    changed = deepcopy(artifact)
    changed["sample_size_budget"] = {}
    cases.append((changed, "sample_size_budget_mismatch"))
    changed = deepcopy(artifact)
    changed["remaining_prd_gaps"] = []
    cases.append((changed, "remaining_prd_gaps_mismatch"))
    changed = deepcopy(artifact)
    changed["field_principles"] = {}
    cases.append((changed, "field_principles_incomplete"))
    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["wrong"]
    cases.append((changed, "current_model_provenance_mismatch"))
    changed = deepcopy(artifact)
    changed["inference_substrate_class"] = "no_model_load"
    cases.append((changed, "inference_substrate_class_mismatch"))
    changed = deepcopy(artifact)
    changed["execution_venue"] = "gpu"
    cases.append((changed, "execution_venue_mismatch"))
    changed = deepcopy(artifact)
    changed["submitted_externally"] = True
    cases.append((changed, "external_submission_mismatch"))
    for candidate, wanted in cases:
        assert wanted in capstone.validate_artifact(candidate, root=ROOT, require_terminal=False)

    assert capstone._terminal_receipts_pass(artifact) is False
    assert capstone._terminal_receipts_pass({"terminal_reader_outcomes": "bad"}) is False
    assert "terminal_reader_receipts_incomplete" in capstone.validate_artifact(
        artifact, root=ROOT, require_terminal=True
    )
    changed = deepcopy(artifact)
    changed["terminal_reader_outcomes"] = {
        name: {"passed": True, "exit_code": 0, "log_sha256": "sha256:test"}
        for name in capstone.TERMINAL_READER_NAMES
    }
    assert capstone._terminal_receipts_pass(changed) is True

    monkeypatch.setattr(
        capstone, "reduce_evidence_summary", lambda _root: (_ for _ in ()).throw(ValueError())
    )
    assert "evidence_reduction_failed:ValueError" in capstone.validate_artifact(
        artifact, root=ROOT, require_terminal=False
    )


def test_validation_and_terminal_command_plans(artifact: dict, tmp_path: Path) -> None:
    """REQ-REPORT-7628 freezes scoped commands and task-specific E2E controls."""

    commands = capstone.build_validation_commands(ROOT, tmp_path)
    assert [row.name for row in commands] == list(capstone.validation_scope.REQUIRED_CHECK_NAMES)
    assert all(
        "tests/python" not in row.argv or capstone.TEST_PATH.as_posix() in row.argv
        for row in commands
    )

    candidate = tmp_path / "candidate.json"
    terminal = capstone._terminal_commands(ROOT, candidate)
    assert [row.name for row in terminal] == list(capstone.TERMINAL_READER_NAMES)
    receipts = [
        {
            "name": row.name,
            "passed": True,
            "exit_code": 0,
            "log_path": f"/tmp/{row.name}.log",
            "log_sha256": f"sha256:{row.name}",
        }
        for row in terminal
    ]
    assert set(capstone._outcomes(receipts)) == set(capstone.TERMINAL_READER_NAMES)
    assert capstone._task_specific_e2e(deepcopy(artifact), ROOT) == []


def test_e2e_self_checks_report_broken_guards(artifact: dict, monkeypatch) -> None:
    """SCENARIO-REPORT-7628-TERMINAL fails when its own controls stop firing."""

    monkeypatch.setattr(capstone, "validate_artifact", lambda *_args, **_kwargs: [])
    errors = capstone._task_specific_e2e(deepcopy(artifact), ROOT)
    assert len([row for row in errors if row.startswith("mutation_not_rejected")]) == 4
    monkeypatch.setattr(
        capstone,
        "classify_terminal",
        lambda *_args: ("complete_null_v665_no_independent_benefit", "null"),
    )
    errors = capstone._task_specific_e2e(deepcopy(artifact), ROOT)
    assert "external_block_classification" in errors
    assert "owned_partial_classification" in errors
