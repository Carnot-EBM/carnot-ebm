"""Tests for REQ-REPORT-7572 and SCENARIO-REPORT-7572-*.

The tests use repository evidence for read-only reductions. They never call the
public writer against the tracked result path.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import carnot.experiment_7572_v661_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


def test_contract_and_inventory_match_exact_v661_authorities() -> None:
    """REQ-REPORT-7572 keeps all thirteen ordered tasks."""

    contract = capstone.load_contract(ROOT)
    assert contract["comparison_passed"] is True
    assert contract["selected_roadmap_path"] == "research-roadmap.yaml"
    assert [row["id"] for row in contract["tasks"]] == list(capstone.EXPECTED_TASK_IDS)

    evidence = capstone.collect_evidence(ROOT, contract["tasks"])
    assert list(evidence) == list(capstone.EXPECTED_TASK_IDS[:-1])
    assert (
        evidence["exp7568-continuous-recalibration"]["evidence_state"] == "conductor_gate_blocked"
    )
    assert evidence["exp7568-continuous-recalibration"]["gate_check_summary"] == {
        "check": "conductor_pre_gate",
        "upstream": "exp7561-recalibration-prototype",
        "path": str(ROOT / "results/experiment_7561_v661_recalibration_prototype.json"),
        "field": "recalibration_ready_score",
        "op": "==",
        "expected": 1,
        "observed": 0,
        "passed": False,
    }


def test_dispositions_preserve_flags_and_do_not_read_future_self() -> None:
    """SCENARIO-REPORT-7572-INVENTORY preserves absence and invalidity."""

    artifact = capstone.build_artifact_for_test()
    rows = artifact["task_dispositions"]
    assert len(rows) == 13
    assert artifact["rows"] == rows
    assert rows[-1]["task_id"] == "exp7572-capstone"
    assert rows[-1]["artifact_path"] is None
    assert rows[-1]["evidence_state"] == "current_terminal"

    by_id = {row["task_id"]: row for row in rows}
    assert by_id["exp7561-recalibration-prototype"]["verdict_class"] == "disqualified"
    assert by_id["exp7568-continuous-recalibration"]["producer_unstarted"] is True
    assert by_id["exp7570-arc-live-lineage"]["flagged_adversarial"] is True
    assert by_id["exp7570-arc-live-lineage"]["excluded_from_scientific_metrics"] is True


def test_branch_conclusions_keep_claims_independent() -> None:
    """SCENARIO-REPORT-7572-BRANCHES blocks cross-branch promotion."""

    artifact = capstone.build_artifact_for_test()
    branches = {row["branch"]: row for row in artifact["branch_conclusions"]}

    source = branches["source_decisions"]
    assert source["claims_qualified_score"] == 1
    assert source["probability_improvement_score"] == 0
    assert source["decision_cost_improvement_score"] == 0
    assert source["verifier_is_oracle"] is True
    assert source["oracle_distinct_benefit_claimed"] is False

    learning = branches["persistent_learning"]
    assert learning["claims_qualified_score"] == 0
    assert learning["freshness_measured"] is False
    assert learning["retention_measured"] is False
    assert learning["restart_measured"] is False

    arc = branches["arc_runtime_lineage"]
    assert arc["source_collection_dependency"] is False
    assert arc["causal_suppression_efficacy_claimed"] is False
    assert arc["new_solve_credit_claimed"] is False
    assert arc["official_score_claimed"] is False
    assert arc["excluded_from_scientific_metrics"] is True

    portability = branches["portable_calibration"]
    assert portability["rust_parity"] is None
    assert portability["complete_service_costs"]["status"].startswith("not_measured")
    assert len(portability["board_rows"]) == 3


@pytest.mark.parametrize(
    ("affected", "terminal", "states", "expected"),
    [
        (True, False, [], "partial"),
        (False, False, [], "disqualified"),
        (True, True, [{"evidence_state": "invalid"}], "disqualified"),
        (True, True, [{"evidence_state": "conductor_gate_blocked"}], "blocked"),
        (True, True, [{"evidence_state": "terminal", "verdict_class": "null"}], "null"),
    ],
)
def test_terminal_precedence(
    affected: bool, terminal: bool, states: list[dict[str, object]], expected: str
) -> None:
    """REQ-REPORT-7572 reserves partial for unfinished owned work."""

    result = capstone.classify_terminal(
        states, affected_complete=affected, terminal_complete=terminal
    )
    assert result["verdict_class"] == expected


def test_artifact_has_required_no_model_and_publication_contract() -> None:
    """REQ-REPORT-7572 separates completion from benefit and publication."""

    artifact = capstone.build_artifact_for_test()
    assert artifact["experiment_id"] == "exp7572-capstone"
    assert artifact["milestone"] == "2026.09.661"
    assert artifact["run_date"] == "20260923"
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["execution_venue"] == "host"
    assert artifact["capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"].startswith("complete_disqualified_")
    publication = artifact["publication_gates"]
    assert publication["paper_ready"] is all(
        publication["gates"][name]["pass"] is True for name in ("G1", "G2", "G3", "G4")
    )
    assert publication["unmet_gates"] == [
        name for name in ("G1", "G2", "G3", "G4") if publication["gates"][name]["pass"] is not True
    ]
    assert artifact["publication_performed"] is False


def test_continuation_and_retirement_require_exact_changed_conditions() -> None:
    """SCENARIO-REPORT-7572-CONTINUATION does not retire resource blocks."""

    artifact = capstone.build_artifact_for_test()
    continuations = {row["branch"]: row for row in artifact["continuation_rows"]}
    assert set(continuations) == {
        "source_decisions",
        "persistent_learning",
        "arc_runtime_lineage",
        "portable_calibration",
    }
    assert all(row["decision"] in {"continue", "defer", "retire"} for row in continuations.values())
    assert all(row["reopen_when"] for row in continuations.values())
    assert continuations["source_decisions"]["decision"] == "continue"
    assert continuations["persistent_learning"]["decision"] == "defer"
    assert artifact["retirement_rows"] == []
    prior = artifact["prior_failure_rows"]
    assert len(prior) == 1
    assert prior[0]["retire_if_same_verdict"] is True
    assert prior[0]["exact_text_match"] is False
    assert prior[0]["retirement_triggered"] is False


def test_sample_budget_and_gate_summary_keep_missing_distinct() -> None:
    """REQ-REPORT-7572 never treats missing producer evidence as zero."""

    artifact = capstone.build_artifact_for_test()
    budget = artifact["sample_size_budget"]
    assert budget["planned"] == budget["attempted"] == budget["completed"] == 13
    assert budget["underlying_producer_work"]["planned_before_capstone"] == 12
    assert budget["underlying_producer_work"]["gate_blocked_without_run"] == 1
    summary = artifact["gate_check_summary"]
    assert summary["passed"] is False
    assert summary["first_failure"]["upstream"] == "exp7561-recalibration-prototype"
    assert summary["first_failure"]["field"]
    assert "expected" in summary["first_failure"]
    assert "observed" in summary["first_failure"]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("task", "task_dispositions_invalid"),
        ("branch", "branch_conclusions_invalid"),
        ("checksum", "reproducibility_checksum_invalid"),
        ("principles", "field_principles_invalid"),
    ],
)
def test_cold_validation_rejects_reduction_mutations(mutation: str, error: str) -> None:
    """SCENARIO-REPORT-7572-VALIDATION rejects altered terminal evidence."""

    artifact = capstone.build_artifact_for_test()
    changed = deepcopy(artifact)
    if mutation == "task":
        changed["task_dispositions"].pop(0)
    elif mutation == "branch":
        changed["branch_conclusions"][0]["benefit_score"] = 1
    elif mutation == "checksum":
        changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    else:
        changed["field_principles"].pop("rows")
    assert error in capstone.validate_artifact(changed)


def test_valid_artifact_reduces_in_a_fresh_file(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7572-VALIDATION supports cold read-only replay."""

    artifact = capstone.build_artifact_for_test()
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    loaded = json.loads(candidate.read_text(encoding="utf-8"))
    assert capstone.validate_artifact(loaded) == []
    assert capstone.independent_reduce(loaded) == []


def test_validation_plan_is_private_and_scoped(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7572-VALIDATION forbids an unscoped suite."""

    commands = capstone.build_validation_plan(ROOT, tmp_path)
    assert capstone.validate_validation_plan(ROOT, commands) == []
    names = [row.name for row in commands]
    assert names == list(capstone.REQUIRED_CHECK_NAMES)
    focused = next(row for row in commands if row.name == "focused_pytest")
    assert "tests/python" not in focused.argv
    assert capstone.TEST_PATH.as_posix() in focused.argv
    assert "-n" in focused.argv and "0" in focused.argv
    assert "--no-cov" in focused.argv
    coverage = next(row for row in commands if row.name == "changed_module_coverage")
    assert any(str(tmp_path) in arg for arg in coverage.argv)


def test_date_and_mapping_boundaries_fail_closed() -> None:
    """REQ-REPORT-7572 binds the frozen date and JSON object shape."""

    assert capstone.date_argument("20260923") == "20260923"
    with pytest.raises(ValueError, match="20260923"):
        capstone.date_argument("20260924")
    assert capstone.validate_artifact([]) == ["artifact_mapping_required"]
    assert capstone.validate_artifact({})[0].startswith("missing_required_field:")


def test_all_fields_and_gates_have_principles() -> None:
    """REQ-REPORT-7572 records why each field and gate exists."""

    artifact = capstone.build_artifact_for_test()
    assert set(artifact["field_principles"]) == set(artifact) - {
        "field_principles",
        "reproducibility_checksum",
    }
    assert all(gate["principle"] for gate in artifact["acceptance_gate_results"])
    scores = {
        key: value
        for key, value in artifact.items()
        if key.endswith(("_ready_score", "_complete_score"))
    }
    assert scores
    assert all(type(value) is int and value in {0, 1} for value in scores.values())


def test_defensive_input_readers_reject_missing_and_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7572 treats unreadable evidence as missing or invalid."""

    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        capstone.load_json(not_object)

    monkeypatch.setattr(
        capstone,
        "resolve_v661_roadmap",
        lambda _root: (tmp_path / "roadmap.yaml", {"tasks": None}, []),
    )
    monkeypatch.setattr(capstone, "compare_contract_authorities", lambda _text, _roadmap: {})
    (tmp_path / capstone.DESIGN_PATH).parent.mkdir(parents=True)
    (tmp_path / capstone.DESIGN_PATH).write_text("design", encoding="utf-8")
    with pytest.raises(ValueError, match="task list"):
        capstone.load_contract(tmp_path)

    task = {"id": "exp9999-missing", "deliverable": "results/missing.json"}
    assert capstone.load_producer(tmp_path, task)["evidence_state"] == "missing"
    malformed = tmp_path / "results/malformed.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{", encoding="utf-8")
    bad_task = {"id": "exp9999-bad", "deliverable": "results/malformed.json"}
    assert capstone.load_producer(tmp_path, bad_task)["evidence_state"] == "invalid"
    assert capstone._required_receipts_pass({}) is False
    assert capstone._producer_failure("x", "p", {})["field"] == "required_validation"


def test_publication_reader_and_source_hashes_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-7572 does not infer publication or custody on reader failure."""

    class Completed:
        stdout = '{"gates": []}'
        returncode = 0

    monkeypatch.setattr(capstone.subprocess, "run", lambda *args, **kwargs: Completed())
    publication = capstone.evaluate_publication_gates(tmp_path)
    assert publication["paper_ready"] is False
    assert publication["unmet_gates"] == ["G1", "G2", "G3", "G4"]

    assert capstone._source_hashes_match({}, tmp_path) is False
    assert capstone._source_hashes_match({"source_artifact_hashes": [None]}, tmp_path) is False
    assert (
        capstone._source_hashes_match(
            {"source_artifact_hashes": [{"path": "missing", "sha256": None}]}, tmp_path
        )
        is True
    )
    assert (
        capstone._source_hashes_match(
            {"source_artifact_hashes": [{"path": "missing", "sha256": "sha256:bad"}]}, tmp_path
        )
        is False
    )


def test_failure_reduction_and_receipt_shape_cover_invalid_controls() -> None:
    """REQ-REPORT-7572 preserves contract and owned-validation failures."""

    failures = capstone.failure_rows(
        {"comparison_passed": False, "selected_roadmap_path": "x"},
        {},
        {"required_checks_passed": False, "terminal_validation_passed": False},
    )
    assert [row["check"] for row in failures] == [
        "contract_authorities_agree",
        "current_required_checks_passed",
        "current_terminal_validation_passed",
    ]
    assert capstone._receipt_set_passed(None, capstone.REQUIRED_CHECK_NAMES) is False


@pytest.mark.parametrize(
    ("field", "replacement", "expected"),
    [
        ("schema", "wrong", "identity_invalid"),
        ("honest_verdict", "wrong", "terminal_identity_invalid"),
        ("model_invoked", True, "model_contract_invalid"),
        ("execution_venue", "host_cpu", "substrate_invalid"),
        ("positive_claim", True, "current_claim_contract_invalid"),
        ("source_artifact_hashes", [], "source_hash_mismatch"),
        ("prior_failure_rows", [], "prior_failure_rows_invalid"),
        ("retirement_rows", [{"wrong": True}], "retirement_rows_invalid"),
        ("continuation_rows", [], "continuation_rows_invalid"),
        ("status", "wrong", "terminal_reduction_invalid"),
        ("acceptance_gate_results", [], "acceptance_gates_invalid"),
        ("sample_size_budget", {}, "sample_size_budget_invalid"),
        ("capstone_complete_score", 0, "capstone_score_invalid"),
        ("publication_gates", {}, "publication_gates_invalid"),
    ],
)
def test_each_cold_validation_guard_rejects_mutation(
    field: str, replacement: object, expected: str
) -> None:
    """SCENARIO-REPORT-7572-VALIDATION proves every reduction guard bites."""

    artifact = capstone.build_artifact_for_test()
    artifact[field] = replacement
    assert expected in capstone.validate_artifact(artifact)


def test_terminal_requirement_and_independent_exception_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-REPORT-7572 distinguishes unfinished owned replay from bad reduction."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.collect_evidence(ROOT, contract["tasks"])
    affected_only = [
        row for row in capstone._passing_receipts() if row["name"] in capstone.REQUIRED_CHECK_NAMES
    ]
    partial = capstone.build_artifact(
        ROOT,
        contract,
        evidence,
        {
            "required_checks_passed": True,
            "terminal_validation_passed": False,
            "validation_receipts": affected_only,
        },
        started_at_utc="2026-09-23T00:00:00+00:00",
        completed_at_utc="2026-09-23T00:00:00+00:00",
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
        duration_s=0.0,
        phase_spans=[],
    )
    assert "terminal_validation_incomplete" in capstone.validate_artifact(
        partial, require_terminal=True
    )

    good = capstone.build_artifact_for_test()
    monkeypatch.setattr(
        capstone, "load_contract", lambda _root: (_ for _ in ()).throw(ValueError())
    )
    assert "independent_reduction_failed" in capstone.validate_artifact(good)
