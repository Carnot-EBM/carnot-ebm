"""Behavior tests for the V642 capstone evidence reducer."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import yaml

from carnot import experiment_7315_v642_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def repository_state() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Load the real roster and evidence once for independent behavior checks."""

    contract = capstone.load_contract(ROOT)
    evidence = capstone.load_repository_payloads(ROOT, contract["tasks"])
    return contract, evidence


def test_selected_roster_keeps_markdown_disagreement_visible(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7315; SCENARIO-REPORT-7315-ROSTER."""

    contract, _ = repository_state
    assert contract["task_ids"] == list(capstone.EXPECTED_TASK_IDS)
    assert contract["roadmap_path"] == "research-roadmap.yaml"
    assert contract["yaml_milestone"] == capstone.MILESTONE
    assert contract["markdown_milestone"] == "2026.09.641"
    assert contract["contract_agrees"] is False


def test_contract_selection_falls_back_and_rejects_wrong_authorities(tmp_path: Path) -> None:
    """REQ-REPORT-7315; SCENARIO-REPORT-7315-ROSTER."""

    with pytest.raises(ValueError, match="no selected V642"):
        capstone.load_contract(tmp_path)

    staged = tmp_path / capstone.STAGED_ROADMAP_PATH
    design = tmp_path / capstone.DESIGN_PATH
    design.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / capstone.ACTIVE_ROADMAP_PATH, staged)
    shutil.copyfile(ROOT / capstone.DESIGN_PATH, design)
    assert capstone.load_contract(tmp_path)["roadmap_path"] == str(capstone.STAGED_ROADMAP_PATH)

    roadmap = yaml.safe_load(staged.read_text(encoding="utf-8"))
    roadmap["tasks"] = list(reversed(roadmap["tasks"]))
    active = tmp_path / capstone.ACTIVE_ROADMAP_PATH
    active.write_text(yaml.safe_dump(roadmap, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="task order"):
        capstone.load_contract(tmp_path)


def test_dispositions_keep_gate_block_absence_and_disqualification_distinct(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7315; SCENARIO-REPORT-7315-ROSTER."""

    _, evidence = repository_state
    assert evidence["exp7306-batch-fixture"]["payload"]["batch_fixture_ready_score"] == 1
    assert evidence["exp7306-batch-fixture"]["disposition_class"] == "disqualified"
    assert evidence["exp7307-batch-canary"]["disposition_class"] == "blocked"
    assert evidence["exp7308-batch-measurement"]["evidence_source"] == "conductor_gate_block"
    assert evidence["exp7308-batch-measurement"]["disposition_class"] == "blocked"
    assert evidence["exp7309-batch-audit"]["selected_evidence_path"] is None
    assert evidence["exp7309-batch-audit"]["disposition_class"] == "absent"


def test_malformed_gate_blocks_and_quarantines_fail_closed() -> None:
    """REQ-REPORT-7315; SCENARIO-REPORT-7315-SCORES."""

    errors = capstone._conductor_block_errors(
        "exp7308-batch-measurement",
        {"schema": "wrong", "status": "complete", "experiment": 1},
    )
    assert errors == [
        "conductor_block_lifecycle",
        "conductor_block_identity",
        "conductor_block_gate",
    ]
    assert (
        capstone._disposition_class(
            {"status": "complete", "verdict_class": "positive"}, {"quarantined": True}
        )
        == "quarantined"
    )


def test_gate_replay_distinguishes_missing_quarantine_field_and_value() -> None:
    """REQ-REPORT-7315; SCENARIO-REPORT-7315-SCORES."""

    tasks = [
        {
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
    ]
    base = {
        "task_id": "producer",
        "declared_deliverable_path": "producer.json",
        "selected_evidence_path": "producer.json",
        "payload": {"status": "complete", "ready_score": 1},
        "quarantine_state": {"quarantined": False},
        "disposition_class": "null",
    }
    outcomes: list[str] = []
    for change in (
        {"selected_evidence_path": None},
        {"quarantine_state": {"quarantined": True}},
        {"payload": {"status": "complete"}},
        {"payload": {"status": "complete", "ready_score": 0}},
    ):
        source = deepcopy(base)
        source.update(change)
        outcomes.append(capstone.replay_gates(tasks, {"producer": source})[0]["outcome"])
    assert outcomes == ["missing_file", "quarantined", "missing_field", "value_mismatch"]


def test_required_scores_bind_numeric_values_to_source_acceptability(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7315; SCENARIO-REPORT-7315-SCORES."""

    _, evidence = repository_state
    rows = capstone.audit_score_rows(evidence)
    observed = {(row["task_id"], row["artifact_field"]): row for row in rows}
    assert observed[("exp7305-arc-selfparse", "arc_capture_complete_score")]["observed_value"] == 1
    assert observed[("exp7305-arc-selfparse", "arc_tool_use_score")]["observed_value"] == 0
    assert observed[("exp7309-batch-audit", "batch_audit_complete_score")]["observed_value"] is None
    assert (
        observed[("exp7309-batch-audit", "batch_audit_complete_score")]["source_authenticated"]
        is False
    )
    assert observed[("exp7312-factor-audit", "factor_promotion_score")]["observed_value"] == 0
    assert (
        observed[("exp7313-cost-envelope", "cost_envelope_complete_score")]["observed_value"] == 1
    )
    assert (
        observed[("exp7314-board-continuity", "board_continuity_complete_score")]["observed_value"]
        == 1
    )


def test_terminal_reduction_blocks_exact_missing_science_without_hiding_nulls(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7315; SCENARIO-REPORT-7315-SCORES."""

    _, evidence = repository_state
    branches = capstone.recompute_branch_rows(evidence)
    terminal = capstone.terminal_state(evidence, branches)
    classes = {row["branch"]: row["verdict_class"] for row in branches}
    assert classes == {
        "arc_tool_reachability": "null",
        "batched_source_value": "blocked",
        "factor_learning": "null",
        "durable_state_cost_context": "null",
        "board_context": "circular_positive",
    }
    assert terminal["status"] == "blocked"
    assert terminal["verdict_class"] == "blocked"
    assert terminal["honest_verdict"].startswith("blocked_")
    failures = terminal["gate_check_summary"]["failures"]
    assert any(
        row["upstream"] == "exp7309-batch-audit"
        and row["artifact_field"] == "declared_deliverable_or_canonical_block"
        and row["observed_value"] is None
        and row["expected_value"] == "terminal evidence"
        for row in failures
    )
    assert any(
        row["upstream"] == "exp7312-factor-audit"
        and row["artifact_field"] == "factor_promotion_score"
        and row["observed_value"] == 0
        and row["terminal_blocking"] is False
        for row in failures
    )


def test_complete_authenticated_science_with_failed_value_gates_is_null(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7315; SCENARIO-REPORT-7315-NULL."""

    _, evidence = repository_state
    complete = deepcopy(evidence)
    complete["exp7309-batch-audit"] = deepcopy(complete["exp7312-factor-audit"])
    complete["exp7309-batch-audit"].update(
        {
            "task_id": "exp7309-batch-audit",
            "selected_evidence_path": "synthetic-complete-audit.json",
            "authenticated": True,
            "accepted_for_positive_claim": True,
            "disposition_class": "null",
            "producer_validation_errors": [],
            "quarantine_state": {"quarantined": False},
        }
    )
    complete["exp7309-batch-audit"]["payload"] = {
        "status": "complete",
        "verdict_class": "null",
        "batch_audit_complete_score": 1,
        "batch_promotion_score": 0,
        "rows": [],
    }
    branches = capstone.recompute_branch_rows(complete)
    terminal = capstone.terminal_state(complete, branches)
    assert terminal["status"] == "complete"
    assert terminal["verdict_class"] == "null"
    assert terminal["honest_verdict"].startswith("complete_null")


def test_independent_value_can_be_positive_but_tool_use_stays_circular(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7315; SCENARIO-REPORT-7315-NULL."""

    _, evidence = repository_state
    complete = deepcopy(evidence)
    complete["exp7309-batch-audit"] = deepcopy(complete["exp7312-factor-audit"])
    complete["exp7309-batch-audit"].update(
        {
            "task_id": "exp7309-batch-audit",
            "selected_evidence_path": "synthetic-complete-audit.json",
            "authenticated": True,
            "accepted_for_positive_claim": True,
            "disposition_class": "positive",
            "producer_validation_errors": [],
            "quarantine_state": {"quarantined": False},
        }
    )
    complete["exp7309-batch-audit"]["payload"] = {
        "status": "complete",
        "verdict_class": "positive",
        "verifier_is_oracle": False,
        "batch_audit_complete_score": 1,
        "batch_promotion_score": 1,
        "rows": [],
    }
    complete["exp7305-arc-selfparse"]["payload"]["arc_tool_use_score"] = 1
    branches = capstone.recompute_branch_rows(complete)
    classes = {row["branch"]: row["verdict_class"] for row in branches}
    assert classes["arc_tool_reachability"] == "circular_positive"
    assert classes["batched_source_value"] == "positive"
    terminal = capstone.terminal_state(complete, branches)
    assert terminal["verdict_class"] == "positive"
    assert terminal["honest_verdict"].startswith("complete_positive")

    complete["exp7309-batch-audit"]["payload"]["batch_promotion_score"] = None
    branches = capstone.recompute_branch_rows(complete)
    assert branches[1]["verdict_class"] == "disqualified"


@pytest.mark.parametrize(
    ("change", "failed_check"),
    [
        ({"quarantine_state": {"quarantined": True}}, "required_evidence_not_quarantined"),
        ({"disposition_class": "disqualified"}, "required_evidence_not_disqualified"),
        (
            {"disposition_class": "blocked", "payload_status": "blocked"},
            "required_evidence_terminal_complete",
        ),
        (
            {"authenticated": False, "producer_validation_errors": ["bad"]},
            "required_evidence_authentic",
        ),
    ],
)
def test_required_evidence_class_precedes_numeric_score(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
    change: dict[str, object],
    failed_check: str,
) -> None:
    """REQ-REPORT-7315; SCENARIO-REPORT-7315-SCORES."""

    _, evidence = repository_state
    changed = deepcopy(evidence)
    source = changed["exp7305-arc-selfparse"]
    payload_status = change.pop("payload_status", None)
    source.update(change)
    if payload_status is not None:
        source["payload"]["status"] = payload_status
    terminal = capstone.terminal_state(changed, capstone.recompute_branch_rows(changed))
    assert any(
        row["upstream"] == "exp7305-arc-selfparse" and row["failed_check"] == failed_check
        for row in terminal["gate_check_summary"]["failures"]
    )


def test_retirement_uses_all_prior_fields_and_preserves_exact_scope(
    repository_state: tuple[dict[str, object], dict[str, dict[str, object]]],
) -> None:
    """REQ-REPORT-7315; SCENARIO-REPORT-7315-RETIREMENT."""

    contract, evidence = repository_state
    rows = capstone.retirement_decisions(contract["tasks"], evidence)
    by_task = {row["task_id"]: row for row in rows if row["task_id"] != "v641-preservation"}
    factor = by_task["exp7312-factor-audit"]
    assert factor["decision"] == "retire_exact_scope"
    assert factor["scope"] == evidence["exp7312-factor-audit"]["payload"]["retirement_scope"]
    assert factor["producer_retirement_signal"] is True
    assert all(
        {"experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"} <= set(prior)
        for row in by_task.values()
        for prior in row["prior_retirement_signals"]
    )
    assert any(
        row["task_id"] == "v641-preservation"
        and row["decision"] == "remain_retired"
        and row["retry_current_mechanism"] is False
        for row in rows
    )


def test_atomic_artifact_reloads_and_detects_bound_row_mutation(tmp_path: Path) -> None:
    """REQ-REPORT-7315; SCENARIO-REPORT-7315-REPLAY."""

    output = tmp_path / "experiment_7315_v642_capstone.json"
    checkpoint = tmp_path / "checkpoints" / output.name
    raw_dir = tmp_path / "raw"
    log_path = tmp_path / "validation.log"
    log_path.write_text("focused validation passed\n", encoding="utf-8")
    receipt_path = tmp_path / "validation_receipts.json"
    receipt_path.write_text(
        json.dumps(
            {
                "receipts": [
                    {
                        "name": "focused_pytest",
                        "command": "pytest focused",
                        "scope": "new V642 behavior",
                        "exit_code": 0,
                        "duration_s": 1.0,
                        "log_path": str(log_path),
                        "log_sha256": capstone.sha256(log_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    artifact = capstone.build_artifact(
        ROOT,
        capstone.RUN_DATE,
        output,
        checkpoint,
        raw_dir=raw_dir,
        validation_receipt_path=receipt_path,
    )
    assert output.is_file()
    assert checkpoint.is_file()
    assert artifact["capstone_complete_score"] == 1
    assert len(artifact["task_dispositions"]) == 14
    assert artifact["validation_receipts"][0]["scope"] == "new V642 behavior"
    assert capstone.validate_artifact(artifact, root=ROOT) == []

    changed = deepcopy(artifact)
    changed["audit_score_rows"][0]["observed_value"] = 99
    changed["reproducibility_checksum"] = capstone.artifact_checksum(changed)
    assert "audit_score_rows" in capstone.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    changed["source_artifact_hashes"] = []
    changed["reproducibility_checksum"] = capstone.artifact_checksum(changed)
    assert "source_artifact_hashes" in capstone.validate_artifact(changed, root=ROOT)

    changed = deepcopy(artifact)
    first_hash = next(iter(changed["source_artifact_hashes"].values()))
    first_hash["sha256"] = "sha256:" + "0" * 64
    changed["reproducibility_checksum"] = capstone.artifact_checksum(changed)
    assert "source_artifact_hashes" in capstone.validate_artifact(changed, root=ROOT)

    with pytest.raises(ValueError, match="run date must be"):
        capstone.build_artifact(ROOT, "wrong", output, checkpoint, raw_dir=raw_dir)

    assert capstone._load_validation_receipts(ROOT, None) == []


def test_read_json_rejects_non_object_root(tmp_path: Path) -> None:
    """REQ-REPORT-7315; SCENARIO-REPORT-7315-REPLAY."""

    path = tmp_path / "array.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON root is not an object"):
        capstone.read_json(path)
