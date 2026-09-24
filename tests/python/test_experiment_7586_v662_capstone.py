"""Tests for REQ-REPORT-7586 and SCENARIO-REPORT-7586-*.

The tests reduce tracked evidence without writing the terminal result. Writer
tests use ``tmp_path`` so historical artifacts remain unchanged.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import carnot.experiment_7586_v662_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def artifact() -> dict[str, object]:
    """Build one deterministic complete candidate for read-only assertions."""

    return capstone.build_artifact_for_test()


def test_contract_is_exact_fourteen_row_v662_authority() -> None:
    """REQ-REPORT-7586 binds the active V662 authority and contract."""

    contract = capstone.load_contract(ROOT)
    assert contract["comparison_passed"] is True
    assert contract["selected_roadmap_path"] == "research-roadmap.yaml"
    assert [task["id"] for task in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)


def test_inventory_distinguishes_pre_gate_and_absent_evidence(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7586-INVENTORY keeps every custody state literal."""

    rows = artifact["task_dispositions"]
    assert isinstance(rows, list) and len(rows) == 14
    assert artifact["rows"] == rows
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["exp7582-arc-panel-a"]["disposition"] == "absent_external_or_pre_gated"
    assert by_id["exp7582-arc-panel-a"]["evidence_state"] == "conductor_gate_blocked"
    assert by_id["exp7582-arc-panel-a"]["evidence_path"] == (
        "results/experiment_7582_arc_panel_a.json"
    )
    assert by_id["exp7583-arc-panel-b"]["disposition"] == "absent_external_or_pre_gated"
    assert by_id["exp7583-arc-panel-b"]["evidence_state"] == "missing"
    assert by_id["exp7583-arc-panel-b"]["artifact_sha256"] is None
    assert by_id["exp7586-capstone"]["disposition"] == "valid_terminal"
    assert by_id["exp7586-capstone"]["artifact_path"] is None


def test_inventory_preserves_source_verdicts_and_substrates(
    artifact: dict[str, object],
) -> None:
    """REQ-REPORT-7586 inherits evidence only by path and exact byte hash."""

    rows = artifact["task_dispositions"]
    assert isinstance(rows, list)
    by_id = {row["task_id"]: row for row in rows}
    assert by_id["exp7577-proper-loss-evaluation"]["honest_verdict"].startswith("complete_null_")
    assert by_id["exp7581-arc-bounded-canary"]["verdict_class"] == "blocked"
    assert by_id["exp7581-arc-bounded-canary"]["inference_substrate_class"] == ("blocked_no_run")
    assert by_id["exp7585-portable-service"]["verdict_class"] == "positive"
    assert by_id["exp7585-portable-service"]["artifact_sha256"].startswith("sha256:")
    assert all(row["claim_scope"] == "descriptive_reuse" for row in rows[:-1])


def test_four_branch_conclusions_never_promote_each_other(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7586-BRANCHES keeps independent outcomes separate."""

    branches = {row["branch"]: row for row in artifact["branch_conclusions"]}
    assert set(branches) == {
        "static_proper_loss",
        "delayed_learning_and_retention",
        "live_verifier_support_and_plan_execution",
        "rust_service_and_board_continuity",
    }
    assert branches["static_proper_loss"]["validity"] is True
    assert branches["static_proper_loss"]["benefit"] is False
    assert branches["static_proper_loss"]["verdict_class"] == "null"
    assert branches["delayed_learning_and_retention"]["benefit"] is False
    assert branches["delayed_learning_and_retention"]["retention_passed"] is False
    arc = branches["live_verifier_support_and_plan_execution"]
    assert arc["validity"] is False
    assert arc["benefit"] == "not_measured"
    assert arc["semantic_null_claimed"] is False
    assert arc["required_mechanism_change"]
    service = branches["rust_service_and_board_continuity"]
    assert service["validity"] is True
    assert service["benefit"] is True
    assert service["whole_service_speedup"]["warm"]["lower95"] > 1.0
    assert len(service["board_rows"]) == 3
    assert service["promotes_other_branches"] is False


@pytest.mark.parametrize(
    ("affected", "terminal", "states", "expected"),
    [
        (True, False, [], "partial"),
        (False, False, [], "disqualified"),
        (True, True, [{"evidence_state": "invalid"}], "disqualified"),
        (True, True, [{"evidence_state": "missing"}], "blocked"),
        (True, True, [{"evidence_state": "terminal", "verdict_class": "null"}], "null"),
    ],
)
def test_terminal_precedence(
    affected: bool, terminal: bool, states: list[dict[str, object]], expected: str
) -> None:
    """REQ-REPORT-7586 reserves partial for unfinished owned validation."""

    reduced = capstone.classify_terminal(
        states, affected_complete=affected, terminal_complete=terminal
    )
    assert reduced["verdict_class"] == expected


def test_retirement_is_narrow_and_arc_absence_does_not_retire(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-REPORT-7586-RETIREMENT closes only a repeated construction."""

    decisions = {row["branch"]: row for row in artifact["retirement_decisions"]}
    proper = decisions["static_proper_loss"]
    assert proper["decision"] == "retire"
    assert proper["scope"] == "v662_frozen_nine_knot_proper_loss_construction"
    assert proper["literal_current_verdict"] == (
        "complete_null_independent_static_and_learning_audit"
    )
    assert proper["explicit_addressed_cause"]
    arc = decisions["live_verifier_support_and_plan_execution"]
    assert arc["decision"] == "defer"
    assert arc["hypothesis_retired"] is False
    assert "E2E-009" in arc["exact_next_prerequisite"]
    assert "E2E-013" in arc["exact_next_prerequisite"]


def test_complete_artifact_has_zero_call_and_publication_contract(
    artifact: dict[str, object],
) -> None:
    """REQ-REPORT-7586 separates accounting completion from research benefit."""

    assert artifact["experiment_id"] == "exp7586-capstone"
    assert artifact["milestone"] == "2026.09.662"
    assert artifact["run_date"] == "20260924"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["planned_inference_substrate_class"] == "aggregation"
    assert artifact["capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "complete_blocked_required_v662_external_evidence"
    assert artifact["positive_claim"] is False
    publication = artifact["publication_gates"]
    assert publication["paper_ready"] is all(
        publication["gates"][name]["pass"] is True for name in ("G1", "G2", "G3", "G4")
    )
    assert publication["unmet_gates"] == [
        name for name in ("G1", "G2", "G3", "G4") if publication["gates"][name]["pass"] is not True
    ]
    assert artifact["publication_performed"] is False


def test_gate_summary_names_exact_external_failures(artifact: dict[str, object]) -> None:
    """REQ-REPORT-7586 gives every blocked verdict complete operands."""

    summary = artifact["gate_check_summary"]
    assert summary["passed"] is False
    failures = summary["failed_checks"]
    assert any(row["upstream"] == "exp7582-arc-panel-a" for row in failures)
    assert any(row["upstream"] == "exp7583-arc-panel-b" for row in failures)
    for row in failures:
        assert {"check", "upstream", "path", "field", "op", "expected", "observed"} <= set(row)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("task", "task_dispositions_invalid"),
        ("branch", "branch_conclusions_invalid"),
        ("retirement", "retirement_decisions_invalid"),
        ("checksum", "reproducibility_checksum_invalid"),
        ("principles", "field_principles_invalid"),
    ],
)
def test_cold_validation_rejects_mutations(
    artifact: dict[str, object], mutation: str, error: str
) -> None:
    """SCENARIO-REPORT-7586-VALIDATION rejects altered reductions."""

    changed = deepcopy(artifact)
    if mutation == "task":
        changed["task_dispositions"].pop(0)
    elif mutation == "branch":
        changed["branch_conclusions"][0]["benefit"] = True
    elif mutation == "retirement":
        changed["retirement_decisions"][0]["scope"] = "all_proper_loss"
    elif mutation == "checksum":
        changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    else:
        changed["field_principles"].pop("rows")
    assert error in capstone.validate_artifact(changed)


def test_fresh_file_reduction_and_validation_plan(
    artifact: dict[str, object], tmp_path: Path
) -> None:
    """SCENARIO-REPORT-7586-VALIDATION supports bounded cold replay."""

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    loaded = json.loads(candidate.read_text(encoding="utf-8"))
    assert capstone.validate_artifact(loaded) == []
    assert capstone.independent_reduce(loaded) == []
    commands = capstone.build_validation_plan(ROOT, tmp_path / "private")
    assert capstone.validate_validation_plan(ROOT, commands) == []
    focused = next(row for row in commands if row.name == "focused_pytest")
    assert capstone.TEST_PATH.as_posix() in focused.argv
    assert "tests/python" not in focused.argv
    assert "-n" in focused.argv and "0" in focused.argv and "--no-cov" in focused.argv


def test_date_root_and_mapping_boundaries_fail_closed(tmp_path: Path) -> None:
    """REQ-REPORT-7586 binds the date, root, and JSON object shape."""

    assert capstone.date_argument("20260924") == "20260924"
    with pytest.raises(ValueError, match="20260924"):
        capstone.date_argument("20260923")
    assert capstone.root_argument(str(ROOT)) == ROOT
    with pytest.raises(ValueError, match="repository root"):
        capstone.root_argument(str(tmp_path))
    assert capstone.validate_artifact([]) == ["artifact_mapping_required"]
    assert capstone.validate_artifact({})[0].startswith("missing_required_field:")


def test_all_fields_and_acceptance_gates_have_principles(
    artifact: dict[str, object],
) -> None:
    """REQ-REPORT-7586 records why each field and gate exists."""

    assert set(artifact["field_principles"]) == set(artifact) - {
        "field_principles",
        "reproducibility_checksum",
    }
    assert all(row["principle"] for row in artifact["acceptance_gate_results"])
    assert artifact["verifier_is_oracle"] is True


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("identity", "identity_invalid"),
        ("terminal", "terminal_identity_invalid"),
        ("model", "model_contract_invalid"),
        ("substrate", "substrate_invalid"),
        ("claim", "current_claim_contract_invalid"),
        ("source_hash", "source_hash_mismatch"),
        ("continuation", "continuation_rows_invalid"),
        ("prior", "prior_failure_rows_invalid"),
        ("terminal_reduction", "terminal_reduction_invalid"),
        ("acceptance", "acceptance_gates_invalid"),
        ("budget", "sample_size_budget_invalid"),
        ("score", "capstone_score_invalid"),
        ("publication", "publication_gates_invalid"),
    ],
)
def test_cold_validation_rejects_contract_mutations(
    artifact: dict[str, object], mutation: str, error: str
) -> None:
    """SCENARIO-REPORT-7586-VALIDATION checks every terminal reduction."""

    changed = deepcopy(artifact)
    if mutation == "identity":
        changed["milestone"] = "2026.09.000"
    elif mutation == "terminal":
        changed["honest_verdict"] = "blocked_without_terminal_prefix"
    elif mutation == "model":
        changed["MODEL_SPECS"] = [{"hf_id": "invented/current"}]
    elif mutation == "substrate":
        changed["inference_substrate_class"] = "model_full_generation"
    elif mutation == "claim":
        changed["positive_claim"] = True
    elif mutation == "source_hash":
        changed["source_artifact_hashes"][0]["sha256"] = "sha256:" + "0" * 64
    elif mutation == "continuation":
        changed["continuation_rows"] = []
    elif mutation == "prior":
        changed["prior_failure_rows"] = []
    elif mutation == "terminal_reduction":
        changed["status"] = "complete_null_wrong"
    elif mutation == "acceptance":
        changed["acceptance_gate_results"] = []
    elif mutation == "budget":
        changed["sample_size_budget"]["planned"] = 13
    elif mutation == "score":
        changed["capstone_complete_score"] = 0
    else:
        changed["publication_gates"]["paper_ready"] = not changed["publication_gates"][
            "paper_ready"
        ]
    assert error in capstone.validate_artifact(changed)


def test_source_reader_defensive_boundaries(tmp_path: Path) -> None:
    """REQ-REPORT-7586 disqualifies malformed or unauthenticated bytes."""

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        capstone.load_json(scalar)

    assert (
        capstone._producer_failure("task", "path", {"flagged_adversarial": True})["field"]
        == "flagged_adversarial"
    )
    assert (
        capstone._producer_failure("task", "path", {"verdict_class": "disqualified"})["field"]
        == "verdict_class"
    )
    assert capstone._producer_failure("task", "path", {})["field"] == "required_validation"
    assert capstone._producer_receipts_pass({}) is False
    assert capstone._producer_receipts_pass({"validation_receipts": []}) is False

    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    task = {"id": "exp9999-private", "deliverable": "bad.json"}
    source = capstone.load_producer(tmp_path, task)
    assert source["evidence_state"] == "invalid"
    assert source["verdict_class"] == "disqualified"
    disposition = capstone._disposition(source, task, 1)
    assert disposition["disposition"] == "disqualified"


def test_hash_failure_and_owned_validation_boundaries(
    artifact: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7586 fails closed on malformed hashes and owned receipts."""

    assert capstone._source_hashes_match({}, ROOT) is False
    assert capstone._source_hashes_match({"source_artifact_hashes": ["bad"]}, ROOT) is False
    absent = {"source_artifact_hashes": [{"path": "absent", "sha256": "sha256:0"}]}
    assert capstone._source_hashes_match(absent, ROOT) is False

    failures = capstone.failure_rows(
        {"comparison_passed": False, "selected_roadmap_path": "roadmap"},
        {},
        {"required_checks_passed": False, "terminal_validation_passed": False},
    )
    assert [row["check"] for row in failures] == [
        "contract_authorities_agree",
        "current_required_checks_passed",
        "current_terminal_validation_passed",
    ]

    incomplete = deepcopy(artifact)
    incomplete["validation_receipts"] = [
        row
        for row in incomplete["validation_receipts"]
        if row["name"] != "declared_entrypoint_cold_replay"
    ]
    assert "terminal_validation_incomplete" in capstone.validate_artifact(
        incomplete, require_terminal=True
    )

    monkeypatch.setattr(
        capstone, "load_contract", lambda _root: (_ for _ in ()).throw(ValueError())
    )
    assert "independent_reduction_failed" in capstone.validate_artifact(artifact)
