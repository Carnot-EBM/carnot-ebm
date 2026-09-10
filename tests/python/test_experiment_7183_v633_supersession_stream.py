"""Tests for the immutable delayed-feedback supersession stream.

Spec refs: REQ-CL-7183 and SCENARIO-CL-7183-*.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from carnot import experiment_7183_v633_supersession_stream as exp


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def views() -> exp.StreamViews:
    """Build the frozen views once because tests do not mutate shared evidence."""

    return exp.build_stream_views()


@pytest.fixture()
def sealed_artifact(tmp_path: Path) -> dict[str, object]:
    """Write private seals so unit tests never replace research result files."""

    return exp.build_and_seal(
        REPO_ROOT,
        exp.StreamPaths.under(tmp_path),
        run_date=exp.RUN_DATE,
        duration_s=1.25,
    )


def test_frozen_contract_and_principles_are_complete() -> None:
    """REQ-CL-7183: Counts, treatments, splits, and field reasons are fixed."""

    assert exp.EXPECTED_EVENT_COUNT == 240
    assert exp.EXPECTED_ARM_ROW_COUNT == 960
    assert exp.FEEDBACK_DELAY == 3
    assert exp.CORRUPTION_COUNT == 24
    assert exp.REGIMES == (
        "stable",
        "superseded_rule",
        "recurrence",
        "conflicting_poisoned_feedback",
    )
    assert len(exp.FAMILIES) == 6
    assert len(exp.ADAPTATION_FAMILIES) == 4
    assert len(exp.TRANSFER_FAMILIES) == 2
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) == set(exp.FIELD_PRINCIPLES)


def test_stream_has_frozen_regimes_families_and_new_numeric_entities(
    views: exp.StreamViews,
) -> None:
    """REQ-CL-7183: The stream contains the full frozen chronological panel."""

    assert len(views.events) == 240
    assert [row["chronology_index"] for row in views.events] == list(range(240))
    assert [sum(row["regime_id"] == regime for row in views.truths) for regime in exp.REGIMES] == [
        60,
        60,
        60,
        60,
    ]
    assert {row["family_id"] for row in views.decisions} == set(exp.FAMILIES)
    assert len({row["entity_id"] for row in views.decisions}) == 240
    assert all(isinstance(row["numeric_value"], int) for row in views.decisions)
    assert exp.stream_conformance_errors(views) == []


def test_decision_view_excludes_truth_regime_and_supersession_fields(
    views: exp.StreamViews,
) -> None:
    """SCENARIO-CL-7183-CHRONOLOGY: Decisions contain no hidden outcome fields."""

    assert exp.leakage_errors(views.decisions) == []
    assert all(
        exp._nested_keys(row).isdisjoint(exp.FORBIDDEN_DECISION_FIELDS) for row in views.decisions
    )
    assert all(row["decision_index"] == index for index, row in enumerate(views.decisions))


def test_feedback_is_released_after_exactly_three_events(
    views: exp.StreamViews,
) -> None:
    """SCENARIO-CL-7183-CHRONOLOGY: Feedback is never current-decision input."""

    assert all(
        row["release_index"] == row["decision_index"] + exp.FEEDBACK_DELAY for row in views.feedback
    )
    for availability in views.availability:
        decision_index = availability["decision_index"]
        expected = [
            row["event_id"] for row in views.feedback if row["release_index"] <= decision_index
        ]
        assert availability["released_feedback_event_ids"] == expected
        assert all(
            event_id != views.decisions[decision_index]["event_id"]
            for event_id in availability["released_feedback_event_ids"]
        )


def test_supersession_uses_new_versions_and_signed_revocations(
    views: exp.StreamViews,
) -> None:
    """SCENARIO-CL-7183-SUPERSESSION: Changed and recurrent rules revoke sources."""

    assert len(views.revocations) == 12
    for family in exp.FAMILIES:
        family_rows = [row for row in views.revocations if row["family_id"] == family]
        assert [(row["revoked_version"], row["replacement_version"]) for row in family_rows] == [
            ("v1", "v2"),
            ("v2", "v3"),
        ]
    assert all(exp.verify_revocation_receipt(row) for row in views.revocations)
    assert all(
        row["release_index"] == row["trigger_decision_index"] + 3 for row in views.revocations
    )


def test_poison_selection_has_exact_contradictory_witnesses(
    views: exp.StreamViews,
) -> None:
    """SCENARIO-CL-7183-POISON: All 24 corrupt labels have exact audit witnesses."""

    corrupted_truth = [row for row in views.truths if row["feedback_corrupted"]]
    assert len(corrupted_truth) == 24
    assert {row["regime_id"] for row in corrupted_truth} == {"conflicting_poisoned_feedback"}
    assert len(views.corruption_witnesses) == 24
    assert all(row["contradiction_confirmed"] for row in views.corruption_witnesses)
    feedback = {row["event_id"]: row for row in views.feedback}
    for row in views.corruption_witnesses:
        assert row["observed_feedback_label"] == feedback[row["event_id"]]["observed_label"]
        assert row["observed_feedback_label"] != row["exact_label"]
        assert exp.replay_corruption_witness(row)


def test_independent_evaluator_agrees_on_every_exact_label(
    views: exp.StreamViews,
) -> None:
    """REQ-CL-7183: Each evaluator-truth row has independent exact agreement."""

    assert len(views.exact_agreements) == 240
    assert all(row["agreement"] for row in views.exact_agreements)
    assert all(exp.replay_truth(row) for row in views.truths)


def test_validation_commit_transfer_and_audit_partitions_are_disjoint(
    views: exp.StreamViews,
) -> None:
    """SCENARIO-CL-7183-HELDOUT: Evaluation labels cannot select updates."""

    heldout = views.manifests["heldout"]
    validation_ids = {
        event_id
        for event_ids in heldout["rolling_validation_event_ids"].values()
        for event_id in event_ids
    }
    support_ids = {
        event_id
        for event_ids in heldout["commit_support_event_ids"].values()
        for event_id in event_ids
    }
    assert validation_ids.isdisjoint(support_ids)
    assert heldout["transfer_labels_available_for_commit_selection"] is False
    assert heldout["final_audit_labels_available_for_commit_selection"] is False
    truth = {row["event_id"]: row for row in views.truths}
    for row in views.availability:
        index = row["decision_index"]
        for event_id in row["commit_support_event_ids"] + row["rolling_validation_event_ids"]:
            source = truth[event_id]
            assert source["family_role"] == "adaptation"
            assert source["decision_index"] < 180
            assert source["feedback_release_index"] <= index
        assert not set(row["commit_support_event_ids"]) & set(row["rolling_validation_event_ids"])


def test_recurrence_and_transfer_cells_are_complete(views: exp.StreamViews) -> None:
    """REQ-CL-7183: Later forgetting and transfer evaluators have all family cells."""

    assert len(views.recurrence) == 12
    assert {row["family_id"] for row in views.recurrence} == set(exp.FAMILIES)
    assert all(row["recurrent_rule_matches_stable_rule"] for row in views.recurrence)
    assert len(views.transfer) == 80
    for family in exp.TRANSFER_FAMILIES:
        for regime in exp.REGIMES:
            assert (
                sum(
                    row["family_id"] == family and row["regime_id"] == regime
                    for row in views.transfer
                )
                == 10
            )


def test_four_arm_materializations_are_identical_except_treatment(
    views: exp.StreamViews,
) -> None:
    """SCENARIO-CL-7183-MATCHED-ARMS: Arms share rows, views, and charges."""

    assert len(views.arm_rows) == 960
    for event_id in [row["event_id"] for row in views.events]:
        rows = [row for row in views.arm_rows if row["event_id"] == event_id]
        assert [row["arm"] for row in rows] == list(exp.ARMS)
        projections = [{key: value for key, value in row.items() if key != "arm"} for row in rows]
        assert projections.count(projections[0]) == 4
        assert all(row["charged_memory_bytes"] == 4096 for row in rows)
        assert all(row["record_inspection_slots"] == 2 for row in rows)
        assert all(row["template_limit"] == 8 for row in rows)


def test_preconditions_record_expected_observed_upstream_and_field(tmp_path: Path) -> None:
    """SCENARIO-CL-7183-PRECONDITIONS: Every gate comparison is diagnostic."""

    checks = exp.collect_preconditions(REPO_ROOT, exp.StreamPaths.under(tmp_path))
    assert all(row["passed"] for row in checks)
    assert all(
        {"check", "upstream", "field", "expected_value", "observed_value", "passed"} <= set(row)
        for row in checks
    )
    gate = next(row for row in checks if row["check"] == "same_milestone_upstream_gates")
    assert gate["expected_value"] == gate["observed_value"] == []


def test_blocked_artifact_is_schema_complete_and_names_failed_gate(tmp_path: Path) -> None:
    """SCENARIO-CL-7183-PRECONDITIONS: Missing sources produce a terminal no-run."""

    paths = exp.StreamPaths.under(tmp_path)
    checks = exp.collect_preconditions(tmp_path / "missing", paths)
    artifact = exp.build_blocked_artifact(checks, paths, run_date=exp.RUN_DATE, duration_s=0.1)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["stream_ready_score"] == 0
    assert artifact["rows"] == []
    assert artifact["gate_check_summary"]["failed_check"]
    assert artifact["gate_check_summary"]["upstream"]
    assert artifact["gate_check_summary"]["field"]
    assert exp.validate_artifact(artifact) == []


def test_complete_artifact_recomputes_ready_and_all_file_seals(
    sealed_artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7183-READINESS: Rows and immutable bytes own readiness."""

    assert sealed_artifact["status"] == "complete"
    assert sealed_artifact["stream_ready_score"] == 1
    assert sealed_artifact["verdict_class"] == "circular_positive"
    assert "no learning" in str(sealed_artifact["honest_verdict"])
    assert exp.validate_artifact(sealed_artifact, repo_root=REPO_ROOT, check_files=True) == []


def test_artifact_mutations_fail_readiness_validation(
    sealed_artifact: dict[str, object],
) -> None:
    """SCENARIO-CL-7183-READINESS: Leakage, row loss, and false gates reject."""

    leaked = deepcopy(sealed_artifact)
    leaked["event_rows"][0]["decision"]["regime_id"] = "stable"
    leaked["reproducibility_checksum"] = exp.reproducibility_checksum(leaked)
    assert "stream_conformance:decision_leakage" in exp.validate_artifact(leaked)

    shortened = deepcopy(sealed_artifact)
    shortened["rows"].pop()
    shortened["reproducibility_checksum"] = exp.reproducibility_checksum(shortened)
    assert "rows_mismatch" in exp.validate_artifact(shortened)

    false_gate = deepcopy(sealed_artifact)
    false_gate["stream_ready_score"] = 0
    false_gate["reproducibility_checksum"] = exp.reproducibility_checksum(false_gate)
    assert "stream_ready_score_mismatch" in exp.validate_artifact(false_gate)


def test_immutable_writer_rejects_different_bytes(tmp_path: Path) -> None:
    """SCENARIO-CL-7183-READINESS: An existing seal cannot change silently."""

    path = tmp_path / "seal.json"
    first = exp.write_immutable_json(path, {"value": 1})
    assert exp.write_immutable_json(path, {"value": 1}) == first
    with pytest.raises(exp.ImmutableSealError, match="immutable_seal_mismatch"):
        exp.write_immutable_json(path, {"value": 2})


def test_defensive_path_and_validation_failures_are_closed(
    tmp_path: Path, views: exp.StreamViews, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7183-PRECONDITIONS: Local failures cannot look complete."""

    defaults = exp.StreamPaths.defaults()
    assert defaults.artifact == exp.DEFAULT_ARTIFACT_PATH
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")
    assert exp._path_writable(parent_file / "child.json") is False
    assert exp.validate_artifact({})[0].startswith("missing_fields:")
    assert "event_count" in exp.stream_conformance_errors(replace(views, events=[]))

    blocked = exp.build_and_seal(
        tmp_path / "missing-root",
        exp.StreamPaths.under(tmp_path / "blocked"),
        run_date=exp.RUN_DATE,
    )
    assert blocked["verdict_class"] == "blocked"

    destination = tmp_path / "atomic.json"
    monkeypatch.setattr(exp.os, "replace", lambda _source, _target: None)
    exp.write_json_atomic(destination, {"complete": True})
    assert not destination.exists()


def test_command_writes_terminal_artifact_and_sealed_views(tmp_path: Path) -> None:
    """REQ-CL-7183: The executable path publishes one validated terminal result."""

    paths = exp.StreamPaths.under(tmp_path)
    assert exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path)]) == 0
    artifact = json.loads(paths.artifact.read_text(encoding="utf-8"))
    assert artifact["stream_ready_score"] == 1
    assert exp.validate_artifact(artifact, repo_root=REPO_ROOT, check_files=True) == []


def test_command_rejects_an_invalid_in_memory_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7183-READINESS: Command validation fails before final publication."""

    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["forced_error"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced_error"):
        exp.main(["--date", exp.RUN_DATE, "--output-root", str(tmp_path)])
