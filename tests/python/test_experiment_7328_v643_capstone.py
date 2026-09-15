"""Behavior tests for the V643 capstone evidence reducer."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7328_v643_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def repository_state() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load the current contract and exact producer files once."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_evidence(ROOT, contract["tasks"])
    return contract, evidence


def test_contract_has_thirteen_exact_rows_and_self_tail(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-CONTRACT."""

    contract, _ = repository_state
    assert contract["passed"] is True
    assert contract["selected_yaml_path"] == "research-roadmap.yaml"
    assert [row["unit_id"] for row in contract["contract_rows"]] == list(capstone.EXPECTED_TASK_IDS)
    assert contract["tasks"][-1]["id"] == "exp7328-capstone"
    assert contract["tasks"][-1].get("gated_on") is None


def test_contract_mutation_is_disqualified(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-CONTRACT."""

    contract, _ = repository_state
    changed = deepcopy(contract["yaml_document"])
    changed["tasks"][4]["title"] += " changed"
    evaluation = capstone.evaluate_contract(contract["markdown_text"], changed)
    assert evaluation["passed"] is False
    assert evaluation["contract_rows"][4]["failures"] == ["title"]


def test_exact_evidence_paths_validate_and_keep_external_block(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-EVIDENCE."""

    _, evidence = repository_state
    assert list(evidence) == list(capstone.EXPECTED_TASK_IDS[:-1])
    assert all(
        row["selected_evidence_path"] == row["declared_deliverable_path"]
        for row in evidence.values()
    )
    assert all(row["artifact_sha256"].startswith("sha256:") for row in evidence.values())
    assert all(row["producer_validation_errors"] == [] for row in evidence.values())
    assert evidence["exp7321-batch-measurement"]["disposition_class"] == "null"
    assert evidence["exp7327-board-continuity"]["disposition_class"] == "blocked"
    assert evidence["exp7327-board-continuity"]["authenticated"] is True


def test_disqualified_score_one_and_quarantine_fail_closed() -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-EVIDENCE."""

    payload = {
        "status": "complete",
        "milestone": capstone.MILESTONE,
        "experiment_id": "exp7317-batch-harness",
        "verdict_class": "disqualified",
        "batch_harness_ready_score": 1,
        "honest_verdict": "complete_disqualified_fixture",
    }
    classified = capstone.classify_payload(
        "exp7317-batch-harness", payload, validator_errors=[], quarantined=False
    )
    assert classified == (True, "disqualified", False)
    quarantined = capstone.classify_payload(
        "exp7317-batch-harness", payload, validator_errors=[], quarantined=True
    )
    assert quarantined == (True, "quarantined", False)


def test_gate_replay_rejects_bad_class_even_when_score_is_one() -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-EVIDENCE."""

    task = {
        "id": "consumer",
        "gated_on": [
            {
                "upstream": "producer",
                "artifact_field": "ready_score",
                "op": "==",
                "value": 1,
            }
        ],
    }
    evidence = {
        "producer": {
            "payload": {"status": "complete", "ready_score": 1},
            "selected_evidence_path": "producer.json",
            "artifact_sha256": "sha256:abc",
            "disposition_class": "disqualified",
            "authenticated": True,
            "quarantined": False,
        }
    }
    row = capstone.replay_gates([task], evidence)[0]
    assert row["outcome"] == "disqualified"
    evidence["producer"]["disposition_class"] = "null"
    assert capstone.replay_gates([task], evidence)[0]["outcome"] == "passed"
    evidence["producer"]["payload"] = {"status": "complete"}
    assert capstone.replay_gates([task], evidence)[0]["outcome"] == "missing_field"


def test_claim_matrix_recomputes_five_separate_boundaries(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-CLAIMS."""

    _, evidence = repository_state
    rows = capstone.build_claim_matrix(evidence)
    assert [row["claim"] for row in rows] == list(capstone.CLAIM_NAMES)
    classes = {row["claim"]: row["verdict_class"] for row in rows}
    assert classes == {
        "source_cost": "null",
        "live_tool_causality": "null",
        "structural_learning": "circular_positive",
        "rust_software_parity": "null",
        "board_evidence": "blocked",
    }
    by_claim = {row["claim"]: row for row in rows}
    assert by_claim["source_cost"]["metrics"]["speedup_vs_direct"] == pytest.approx(
        0.3329221984008092
    )
    assert by_claim["live_tool_causality"]["metrics"]["complete_chains"] == 0
    assert by_claim["structural_learning"]["metrics"]["query_ratio_vs_reset"] == pytest.approx(
        0.14898810929994488
    )
    assert by_claim["rust_software_parity"]["metrics"]["parity_mismatches"] == 0
    assert by_claim["rust_software_parity"]["metrics"]["ten_x_cost_gate"] is False
    assert by_claim["board_evidence"]["metrics"]["blocked_boards"] == ["GateMate"]
    assert all(row["promotes_scientific_efficacy"] is False for row in rows)


def test_terminal_state_is_complete_but_names_exact_board_block(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-BLOCKED."""

    _, evidence = repository_state
    terminal = capstone.terminal_state(evidence, capstone.build_claim_matrix(evidence), True)
    assert terminal["status"] == "complete"
    assert terminal["verdict_class"] == "blocked"
    assert terminal["honest_verdict"].startswith("blocked_exp7327_board_continuity")
    failure = terminal["gate_check_summary"]["failures"][0]
    assert failure["upstream"] == "exp7327-board-continuity"
    assert failure["check"] == "gatemate_changed_physical_state_receipt"
    assert failure["artifact_field"] == "receipt_date/operator_authored/provenance/changed_field"
    assert failure["observed_value"]["accepted_receipt_count"] == 0


def test_missing_required_evidence_is_not_reduced_as_zero(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-BLOCKED."""

    _, evidence = repository_state
    missing = deepcopy(evidence)
    missing["exp7322-batch-audit"].update(
        {
            "selected_evidence_path": None,
            "artifact_sha256": None,
            "authenticated": False,
            "disposition_class": "missing",
            "payload": {},
        }
    )
    rows = capstone.build_claim_matrix(missing)
    source = next(row for row in rows if row["claim"] == "source_cost")
    assert source["metric"] is None
    assert source["verdict_class"] == "blocked"
    terminal = capstone.terminal_state(missing, rows, True)
    failure = next(
        row
        for row in terminal["gate_check_summary"]["failures"]
        if row["upstream"] == "exp7322-batch-audit"
    )
    assert failure["artifact_field"] == "declared_deliverable_or_canonical_block"
    assert failure["observed_value"] is None


def test_prior_verdict_bytes_and_v642_retirements_remain_exact(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-RETIREMENTS."""

    contract, evidence = repository_state
    terminal = capstone.terminal_state(evidence, capstone.build_claim_matrix(evidence), True)
    repeats = capstone.prior_failure_rows(contract["tasks"], evidence, terminal)
    assert repeats
    assert all(isinstance(row["prior_honest_verdict_bytes"], str) for row in repeats)
    assert all(
        row["exact_repeat"]
        == (row["prior_honest_verdict_bytes"] == row["current_honest_verdict_bytes"])
        for row in repeats
    )

    decisions = capstone.next_branch_decisions(evidence, ROOT)
    assert [row["branch"] for row in decisions] == list(capstone.DECISION_BRANCHES)
    by_branch = {row["branch"]: row for row in decisions}
    assert by_branch["source_cost"]["action"] == "retire"
    assert by_branch["structural_learning"]["action"] == "retain"
    assert by_branch["rust_software_parity"]["action"] == "implement"
    assert by_branch["board_evidence"]["action"] == "blocked_pending_prerequisite"
    assert "longest-consistent-suffix" in by_branch["v642_suffix_retirement"]["scope"]
    assert by_branch["v642_storage_retirement"]["action"] == "retire"
    assert by_branch["v642_storage_retirement"]["evidence"]["storage_sweep_retired"] is True


def test_publication_gate_is_exact_and_does_not_authorize_actions() -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-PUBLICATION."""

    publication, receipt = capstone.run_publication_gate(ROOT)
    assert publication["paper_ready"] is True
    assert publication["unmet_gates"] == []
    assert list(publication["gates"]) == ["G1", "G2", "G3", "G4"]
    assert receipt["exit_code"] == 0
    assert receipt["passed"] is True
    assert receipt["log_sha256"].startswith("sha256:")


def _passing_validation() -> dict[str, object]:
    receipts = [
        {
            "name": name,
            "command": f"check {name}",
            "scope": "explicit",
            "exit_code": 0,
            "duration_s": 0.01,
            "log_sha256": "sha256:" + "a" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in capstone.REQUIRED_SCOPED_CHECKS
    ]
    return {
        "validation_receipts": receipts,
        "required_checks_passed": True,
        "missing_required_commands": [],
        "failed_required_commands": [],
        "duplicate_required_commands": [],
        "repository_health": {
            "status": "degraded_open",
            "incident_open": True,
            "historical_failures": [],
            "historical_failure_count": 0,
            "unresolved_collection_error_observation_count": 0,
            "affects_required_checks": False,
        },
    }


def test_artifact_build_and_cold_replay_detect_mutation(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-ARTIFACT."""

    contract, evidence = repository_state
    publication, publication_receipt = capstone.run_publication_gate(ROOT)
    artifact = capstone.build_artifact(
        root=ROOT,
        run_date=capstone.RUN_DATE,
        contract=contract,
        evidence=evidence,
        publication_gate=publication,
        publication_receipt=publication_receipt,
        validation=_passing_validation(),
        started_at="2026-09-15T00:00:00+00:00",
        completed_at="2026-09-15T00:00:01+00:00",
        duration_s=1.0,
        phase_spans=[
            {
                "phase": "test",
                "start_s": 0.0,
                "end_s": 1.0,
                "duration_s": 1.0,
                "units": 13,
                "checkpoint": None,
                "pending_operations": [],
            }
        ],
    )
    assert capstone.validate_artifact(artifact, root=ROOT, replay=True) == []
    assert artifact["capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "blocked"
    assert artifact["capstone_readiness_score"] == 0
    assert artifact["capstone_promotion_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == capstone.ZERO_INVOCATION_COUNTS
    assert artifact["publication_performed"] is False
    assert artifact["upload_performed"] is False
    assert artifact["submission_performed"] is False
    assert artifact["production_default_changed"] is False

    changed = deepcopy(artifact)
    changed["claim_matrix"][0]["metric"] = 99
    assert "claim_matrix" in capstone.validate_artifact(changed, root=ROOT, replay=False)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum" in capstone.validate_artifact(
        changed, root=ROOT, replay=False
    )


def test_failed_affected_validation_disqualifies_current_result(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-BLOCKED."""

    contract, evidence = repository_state
    publication, receipt = capstone.run_publication_gate(ROOT)
    validation = _passing_validation()
    validation["required_checks_passed"] = False
    validation["failed_required_commands"] = ["focused_pytest"]
    artifact = capstone.build_artifact(
        root=ROOT,
        run_date=capstone.RUN_DATE,
        contract=contract,
        evidence=evidence,
        publication_gate=publication,
        publication_receipt=receipt,
        validation=validation,
        started_at="2026-09-15T00:00:00+00:00",
        completed_at="2026-09-15T00:00:01+00:00",
        duration_s=1.0,
        phase_spans=[],
    )
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"].startswith("complete_disqualified_")
    assert artifact["capstone_complete_score"] == 1


def test_atomic_json_round_trip_and_date_validation(tmp_path: Path) -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-ARTIFACT."""

    target = tmp_path / "nested" / "artifact.json"
    capstone.atomic_write_json(target, {"value": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
    assert capstone.date_argument("20260915") == "20260915"
    with pytest.raises(ValueError, match="YYYYMMDD"):
        capstone.date_argument("2026-09-15")


def test_fail_closed_helpers_cover_missing_and_malformed_inputs(tmp_path: Path) -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-EVIDENCE."""

    nonobject = tmp_path / "list.json"
    nonobject.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        capstone.read_json(nonobject)
    with pytest.raises(ValueError, match="no selected V643"):
        capstone.load_contract(tmp_path)

    malformed = {
        "status": "running",
        "milestone": capstone.MILESTONE,
        "experiment_id": "exp7317-batch-harness",
        "verdict_class": "unknown",
    }
    assert capstone.classify_payload(
        "exp7317-batch-harness",
        malformed,
        validator_errors=["bad"],
        quarantined=False,
    ) == (False, "disqualified", False)
    malformed["status"] = "complete"
    assert capstone.classify_payload(
        "exp7317-batch-harness", malformed, validator_errors=[], quarantined=False
    ) == (True, "disqualified", False)

    task = {
        "id": "exp7317-batch-harness",
        "deliverable": "results/declared-missing.json",
    }
    canonical = tmp_path / capstone.common.canonical_gate_block_path(task["id"])
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes((ROOT / "results/experiment_7317_v643_batch_harness.json").read_bytes())
    selected = capstone.load_evidence(tmp_path, task, {})
    assert selected["evidence_source"] == "canonical_conductor_block"
    canonical.unlink()
    missing = capstone.load_evidence(tmp_path, task, {})
    assert missing["evidence_source"] == "missing"
    assert missing["selected_evidence_path"] is None


def test_gate_replay_covers_missing_quarantine_and_blocked() -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-EVIDENCE."""

    task = {
        "id": "consumer",
        "gated_on": [{"upstream": "producer", "artifact_field": "ready", "op": "==", "value": 1}],
    }
    base = {
        "payload": {"status": "complete", "ready": 1},
        "selected_evidence_path": "producer.json",
        "artifact_sha256": "sha256:abc",
        "disposition_class": "null",
        "authenticated": True,
        "quarantined": False,
    }
    outcomes = []
    for change in (
        {"selected_evidence_path": None},
        {"quarantined": True},
        {"disposition_class": "blocked"},
    ):
        source = deepcopy(base)
        source.update(change)
        outcomes.append(capstone.replay_gates([task], {"producer": source})[0]["outcome"])
    assert outcomes == ["missing_file", "quarantined", "blocked"]


def test_terminal_helpers_cover_generic_and_nonblocked_states() -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-BLOCKED."""

    source = {
        "selected_evidence_path": "producer.json",
        "payload": {"gate_check_summary": {}},
        "disposition_class": "blocked",
    }
    generic = capstone._external_failure(source, "producer")
    assert generic["check"] == "producer_terminal_class"

    evidence = {"producer": source}
    circular = capstone.terminal_state(
        evidence,
        [{"producer": "producer", "verdict_class": "circular_positive"}],
        True,
    )
    assert circular["verdict_class"] == "circular_positive"
    null = capstone.terminal_state(
        evidence,
        [{"producer": "producer", "verdict_class": "null"}],
        True,
    )
    assert null["verdict_class"] == "null"


def test_validation_and_terminal_command_helpers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-REPORT-7328; SCENARIO-REPORT-7328-ARTIFACT."""

    assert capstone._receipt_set_passes({}) is False
    assert capstone.validate_artifact([], root=ROOT) == ["artifact_mapping"]
    assert capstone._phase_row("unit", 1.0, 2.5, 3, None)["duration_s"] == 1.5

    captured: list[capstone.scoped.CommandSpec] = []

    def fake_run(
        root: Path, commands: list[capstone.scoped.CommandSpec], *, log_dir: Path
    ) -> list[dict[str, object]]:
        assert root == ROOT
        assert log_dir == tmp_path / "terminal_validation"
        captured.extend(commands)
        return []

    monkeypatch.setattr(capstone.scoped, "run_commands", fake_run)
    assert capstone._run_terminal_commands(ROOT, tmp_path / "candidate.json", tmp_path) == []
    assert [row.name for row in captured] == [
        "independent_terminal_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
