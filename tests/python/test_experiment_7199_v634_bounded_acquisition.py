"""Tests for bounded exact version-space acquisition.

Spec refs: REQ-CL-7199 and SCENARIO-CL-7199-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7199_v634_bounded_acquisition as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


def public(event_id: str, value: int, family: str = "modular_equals") -> dict[str, object]:
    """Build one learner-visible event without evaluator fields."""

    return {
        "event_id": event_id,
        "family_id": family,
        "numeric_value": value,
        "public_input": f"family={family};value={value}",
    }


def observe(
    controller: exp.VersionSpaceController,
    event_id: str,
    value: int,
    label: str,
    role: str,
    index: int,
) -> dict[str, object]:
    """Release one requested label through the public controller API."""

    return controller.observe(
        public(event_id, value),
        observed_label=label,
        role=role,
        request_index=index,
        release_index=index,
    )


def freeze_modular_singleton(controller: exp.VersionSpaceController) -> None:
    """Use three distinct support labels to freeze parameter seven."""

    observe(controller, "support-accept", 7, "accept", "support", 1)
    observe(controller, "support-reject-8", 8, "reject", "support", 2)
    observe(controller, "support-reject-9", 9, "reject", "support", 3)


def commit_modular_singleton(controller: exp.VersionSpaceController) -> None:
    """Release eight later validation labels for the frozen candidate."""

    freeze_modular_singleton(controller)
    for offset in range(8):
        value = 7 if offset % 2 == 0 else 10 + offset
        label = "accept" if value == 7 else "reject"
        observe(controller, f"validation-{offset}", value, label, "validation", 4 + offset)


def test_contract_and_role_partition_are_frozen() -> None:
    """REQ-CL-7199: Fixed arms, cells, fields, and public roles match the spec."""

    assert exp.ARMS == (
        "warmup_frozen",
        "fifo_admission",
        "random_admission",
        "priority_admission",
        "all_information_oracle",
    )
    assert exp.CAPACITIES == (1, 4, 16)
    assert exp.DELAY_SCHEDULES == ("constant_0", "constant_4", "constant_16", "burst")
    assert exp.PRIMARY_CELL == {"capacity": 4, "delay_schedule": "burst"}
    assert exp.MODEL_SPECS == []
    assert exp.MODEL_INVOKED is False
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)

    block = [public(f"event-{index}", index) for index in range(4)]
    roles = exp.partition_roles(block)
    assert sorted(roles.values()) == ["support", "support", "validation", "validation"]
    assert roles == exp.partition_roles(deepcopy(block))
    changed = deepcopy(block)
    changed[0]["numeric_value"] = 99
    assert roles != exp.partition_roles(changed)


def test_majority_tie_and_empty_prediction_contract() -> None:
    """SCENARIO-CL-7199-ROLE-SEPARATION: Vote ties reject and empty sets abstain."""

    controller = exp.VersionSpaceController()
    state = controller.families["modular_equals"]
    state.hypotheses = {7}
    assert controller.predict(public("one", 7))[0] == "accept"
    state.hypotheses = {7, 8}
    assert controller.predict(public("tie", 7))[0] == "reject"
    state.hypotheses = set()
    assert controller.predict(public("empty", 7))[0] == "abstain"


def test_support_freezes_then_later_validation_commits_without_prediction_change() -> None:
    """SCENARIO-CL-7199-COMMIT: Fresh support and later validation persist a singleton."""

    controller = exp.VersionSpaceController()
    early = observe(controller, "early-validation", 7, "accept", "validation", 0)
    assert early["operation"] == "validation_archived_before_freeze"
    freeze_modular_singleton(controller)
    state = controller.families["modular_equals"]
    assert state.hypotheses == {7}
    assert state.candidate_parameter == 7
    assert state.freeze_release_index == 3
    assert state.validation_ids == []
    prediction_before = controller.predict(public("future", 7))[0]

    updates = []
    for offset in range(8):
        value = 7 if offset % 2 == 0 else 10 + offset
        label = "accept" if value == 7 else "reject"
        updates.append(
            observe(controller, f"later-validation-{offset}", value, label, "validation", 4 + offset)
        )
    state = controller.families["modular_equals"]
    assert updates[-1]["operation"] == "commit_template"
    assert state.committed_template["parameter"] == 7
    assert state.committed_template["predicate"] == "modular_equals(parameter=7)"
    assert len(state.validation_ids) == 8
    assert controller.predict(public("future", 7))[0] == prediction_before
    assert updates[-1]["prediction_changed_by_commit"] is False


def test_validation_failure_resets_without_fitting_replacement() -> None:
    """SCENARIO-CL-7199-ROLE-SEPARATION: Failed validation resets but never eliminates."""

    controller = exp.VersionSpaceController()
    freeze_modular_singleton(controller)
    update = observe(controller, "bad-validation", 7, "reject", "validation", 4)
    state = controller.families["modular_equals"]
    assert update["operation"] == "candidate_rejected_reset"
    assert state.epoch == 1
    assert state.hypotheses == set(range(33))
    assert state.support_ids == []
    assert state.validation_ids == []
    assert state.candidate_parameter is None
    assert 7 in state.hypotheses


def test_committed_contradiction_revokes_and_reopens_with_fresh_support() -> None:
    """SCENARIO-CL-7199-REVOCATION: Empty support evidence keeps rollback lineage."""

    controller = exp.VersionSpaceController()
    commit_modular_singleton(controller)
    old_hash = controller.state_hash()
    update = observe(controller, "counterexample", 7, "reject", "support", 20)
    state = controller.families["modular_equals"]
    assert update["operation"] == "revoke_reset_support"
    assert update["rollback_hash"] == old_hash
    assert state.epoch == 1
    assert state.committed_template is None
    assert state.hypotheses == set(range(33)) - {7}
    assert state.support_ids == ["counterexample"]
    assert state.validation_ids == []
    assert len(state.superseded_templates) == 1
    assert state.superseded_templates[0]["parameter"] == 7


def test_selectors_use_only_public_votes_and_one_tie_rule() -> None:
    """SCENARIO-CL-7199-MATCHED-ADMISSION: Only the admission choice differs."""

    controller = exp.VersionSpaceController()
    block = [public(f"event-{index}", value) for index, value in enumerate((7, 8, 9, 10))]
    ranks = exp.seeded_tie_ranks(123, 4, block)
    assert exp.select_request(block, "fifo_admission", ranks, controller) == block[0]
    assert exp.select_request(block, "random_admission", ranks, controller) == min(
        block, key=lambda row: ranks[str(row["event_id"])]
    )
    chosen = exp.select_request(block, "priority_admission", ranks, controller)
    scores = {str(row["event_id"]): controller.predict(row)[1] for row in block}
    assert scores[str(chosen["event_id"])] == max(scores.values())
    with pytest.raises(ValueError, match="unsupported_admission_arm"):
        exp.select_request(block, "unknown", ranks, controller)


def test_single_seed_cell_panel_preserves_chronology_roles_and_bounds() -> None:
    """SCENARIO-CL-7199-MATCHED-ADMISSION: A real upstream replay stays bounded."""

    views = exp.load_upstream_views(REPO_ROOT, exp.DEFAULT_UPSTREAM_ARTIFACT_PATH)
    seed = exp.STREAM_SEEDS[0]
    panel = exp.run_acquisition_panel(
        views,
        seeds=(seed,),
        cells=((4, "burst"),),
    )
    assert len(panel.rows) == len(exp.ARMS)
    assert len(panel.decision_rows) == exp.EVENTS_PER_SEED * len(exp.ARMS)
    assert {row["arm"] for row in panel.rows} == set(exp.ARMS)
    assert all(row["max_memory_bytes"] <= exp.MEMORY_BYTE_BUDGET for row in panel.rows)
    assert all(row["max_pending"] <= 4 for row in panel.rows)
    assert all(row["pending_eviction_count"] == 0 for row in panel.rows)
    assert all(row["prediction_index"] < row["outcome_score_index"] for row in panel.decision_rows)
    assert all(row["error"] == 1 for row in panel.decision_rows if row["abstention"] == 1)
    assert all(row["future_delay_visible_to_selector"] is False for row in panel.pending_queue_rows)
    assert all(row["role_frozen_before_label"] is True for row in panel.validation_access_rows)
    assert exp.panel_conformance_errors(panel, seeds=(seed,), cells=((4, "burst"),)) == []


def test_seed_bootstrap_and_primary_gate_are_fixed() -> None:
    """SCENARIO-CL-7199-METRICS: Seed-paired intervals control the primary verdict."""

    left = {seed: 0.1 for seed in exp.STREAM_SEEDS}
    right = {seed: 0.3 for seed in exp.STREAM_SEEDS}
    estimate = exp.paired_seed_bootstrap(left, right, seed=exp.RANDOM_SEED)
    assert estimate == {"difference": -0.2, "ci95_low": -0.2, "ci95_high": -0.2}

    comparisons = []
    for control in ("warmup_frozen", "fifo_admission", "random_admission"):
        comparisons.extend(
            [
                {
                    "capacity": 4,
                    "delay_schedule": "burst",
                    "arm": "priority_admission",
                    "control": control,
                    "window": "prospective",
                    "metric": "error_rate",
                    "difference": -0.1,
                    "ci95_high": -0.01,
                },
                {
                    "capacity": 4,
                    "delay_schedule": "burst",
                    "arm": "priority_admission",
                    "control": control,
                    "window": "prospective",
                    "metric": "false_accept_rate",
                    "difference": 0.0,
                    "ci95_high": 0.0,
                },
                {
                    "capacity": 4,
                    "delay_schedule": "burst",
                    "arm": "priority_admission",
                    "control": control,
                    "window": "recurrence",
                    "metric": "error_rate",
                    "difference": 0.01,
                    "ci95_high": 0.01,
                },
            ]
        )
    assert exp.score_primary_gate(comparisons, violation_count=0) == (1, 1)
    changed = deepcopy(comparisons)
    changed[0]["ci95_high"] = 0.0
    assert exp.score_primary_gate(changed, violation_count=0)[0] == 0
    assert exp.score_primary_gate(comparisons, violation_count=1) == (0, 0)


def test_preconditions_parse_real_files_and_reject_quarantine(tmp_path: Path) -> None:
    """SCENARIO-CL-7199-PRECONDITIONS: Exact gates run before stream loading."""

    checks, upstream = exp.collect_preconditions(
        REPO_ROOT,
        exp.DEFAULT_UPSTREAM_ARTIFACT_PATH,
        exp.ExperimentPaths.under(tmp_path),
    )
    assert all(row["passed"] for row in checks)
    gate = next(row for row in checks if row["check"] == "upstream_completion_gate")
    assert gate["expected_value"] == gate["observed_value"] == 1
    prior_null = next(row for row in checks if row["check"] == "known_prior_null_not_promoted")
    assert prior_null["passed"] is True
    assert upstream["stream_capacity_ready_score"] == 1

    quarantined = deepcopy(upstream)
    quarantined["flagged_adversarial"] = True
    fixture = tmp_path / "quarantined.json"
    fixture.write_text(json.dumps(quarantined), encoding="utf-8")
    failed, _ = exp.collect_preconditions(
        REPO_ROOT,
        fixture,
        exp.ExperimentPaths.under(tmp_path / "blocked"),
    )
    quarantine_gate = next(row for row in failed if row["check"] == "upstream_not_quarantined")
    assert quarantine_gate["passed"] is False
    assert quarantine_gate["observed_value"] is True


def test_blocked_artifact_is_terminal_diagnostic_and_row_free(tmp_path: Path) -> None:
    """SCENARIO-CL-7199-PRECONDITIONS: Missing evidence creates a blocked result."""

    root = tmp_path / "missing"
    paths = exp.ExperimentPaths.under(tmp_path / "output")
    checks, upstream = exp.collect_preconditions(root, Path("missing.json"), paths)
    artifact = exp.build_blocked_artifact(
        checks,
        upstream,
        repo_root=root,
        upstream_artifact_path=Path("missing.json"),
        paths=paths,
        run_date=exp.RUN_DATE,
        duration_s=0.01,
    )
    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["acquisition_run_complete_score"] == 0
    assert artifact["rows"] == artifact["decision_rows"] == artifact["update_rows"] == []
    assert artifact["gate_check_summary"]["failed_check"]
    assert exp.validate_artifact(artifact) == []
    assert exp.validate_artifact({})[0].startswith("missing_fields:")


def test_terminal_score_keeps_complete_null_separate() -> None:
    """SCENARIO-CL-7199-TERMINAL: Complete panels can have a zero value score."""

    assert exp.terminal_classification(completion_score=1, value_score=0) == (
        "complete",
        "null",
        "complete_null: bounded acquisition did not pass the frozen primary-cell gate",
    )
    assert exp.terminal_classification(completion_score=1, value_score=1)[1] == "positive"
    assert exp.terminal_classification(completion_score=0, value_score=0)[1] == "disqualified"


def test_command_refuses_invalid_artifact_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7199: The command validates before the terminal atomic write."""

    monkeypatch.setattr(exp, "build_and_seal", lambda *_args, **_kwargs: {"verdict_class": "null"})
    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["forced_error"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced_error"):
        exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path)])
    assert not exp.ExperimentPaths.under(tmp_path).artifact.exists()
