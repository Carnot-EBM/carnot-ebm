"""Tests for REQ-CL-7183 and its immutable supersession-stream scenarios."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import carnot.experiment_7183_v633_supersession_stream as exp
import pytest


def test_req_cl_7183_builds_the_frozen_stream() -> None:
    """REQ-CL-7183 fixes event, regime, family, split, and poison coverage."""

    events = exp.build_events()

    assert len(events) == 240
    assert [row["chronology_index"] for row in events] == list(range(240))
    assert {
        name: sum(row["regime_id"] == name for row in events)
        for name in exp.REGIME_IDS
    } == {name: 60 for name in exp.REGIME_IDS}
    assert {row["family_id"] for row in events} == set(exp.FAMILY_IDS)
    assert all(sum(row["family_id"] == family for row in events) == 40 for family in exp.FAMILY_IDS)
    assert len(exp.ADAPTATION_FAMILIES) == 4
    assert len(exp.TRANSFER_FAMILIES) == 2
    assert sum(row["corrupted_feedback"] for row in events) == 24
    assert all(row["exact_label"] == row["independent_exact_label"] for row in events)


def test_scenario_cl_7183_chronology_and_heldout_isolation() -> None:
    """SCENARIO-CL-7183-CHRONOLOGY and HELDOUT prevent decision-time leakage."""

    events = exp.build_events()
    decisions = exp.decision_view(events)
    schedule = exp.feedback_schedule(events)
    availability = exp.availability_matrix(events)
    forbidden = {
        "exact_label",
        "independent_exact_label",
        "regime_id",
        "future_feedback",
        "supersession_flag",
        "corrupted_feedback",
        "revocation_receipt",
    }

    assert all(not (exp.nested_keys(row) & forbidden) for row in decisions)
    assert all(row["feedback_release_index"] == row["decision_index"] + 3 for row in schedule)
    for row in availability:
        index = row["decision_index"]
        assert all(int(event_id.rsplit("-", 1)[1]) <= index - 3 for event_id in row["available_feedback_event_ids"])
        assert set(row["commit_support_event_ids"]).isdisjoint(row["validation_event_ids"])
        assert row["current_feedback_available"] is False
    manifest = exp.heldout_manifest(events)
    assert manifest["audit_segment"] == {"start": 180, "stop": 240}
    assert set(manifest["transfer_families"]) == set(exp.TRANSFER_FAMILIES)
    assert all(not row["commit_selection_allowed"] for row in schedule if row["decision_index"] >= 180)
    assert all(
        not row["commit_selection_allowed"]
        for row in schedule
        if row["family_id"] in exp.TRANSFER_FAMILIES
    )


def test_scenario_cl_7183_supersession_and_poison_witnesses() -> None:
    """SCENARIO-CL-7183-SUPERSESSION and POISON retain exact audit evidence."""

    events = exp.build_events()
    feedback = exp.feedback_view(events)
    truth = exp.evaluator_truth_view(events)
    revocations = [row for row in feedback if row["revocation_receipt"] is not None]
    corrupt = [row for row in truth if row["corrupted_feedback"]]

    assert len(revocations) == 12
    assert {(row["revoked_source_version"], row["activated_source_version"]) for row in revocations} == {
        ("v1", "v2"),
        ("v2", "v3"),
    }
    assert len(corrupt) == 24
    feedback_by_id = {row["event_id"]: row for row in feedback}
    for row in corrupt:
        assert feedback_by_id[row["event_id"]]["observed_label"] is not row["exact_label"]
        assert row["contradictory_witness"]["observed_feedback_label"] is not row["exact_label"]
        assert exp.replay_truth(row["family_id"], row["source_version"], row["numeric_values"]) == row["exact_label"]


def test_scenario_cl_7183_matched_arm_materialization() -> None:
    """SCENARIO-CL-7183-MATCHED-ARMS gives each arm identical stream resources."""

    events = exp.build_events()
    rows = exp.arm_materializations(events)

    assert len(rows) == 960
    for event in events:
        matched = [row for row in rows if row["event_id"] == event["event_id"]]
        assert [row["arm_id"] for row in matched] == list(exp.ARM_IDS)
        comparable = [{key: value for key, value in row.items() if key != "arm_id"} for row in matched]
        assert comparable.count(comparable[0]) == 4
        assert all(row["memory_byte_budget"] == 4096 for row in matched)
        assert all(row["inspection_budget"] == 2 for row in matched)


def test_scenario_cl_7183_artifact_round_trip_and_mutations(tmp_path: Path) -> None:
    """SCENARIO-CL-7183-READINESS recomputes files, seals, score, and verdict."""

    paths = exp.StreamPaths.under(tmp_path)
    artifact = exp.build_and_seal(exp.REPO_ROOT, paths, run_date="20260910", duration_s=0.25)

    assert artifact["status"] == "complete"
    assert artifact["stream_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert len(artifact["event_rows"]) == 240
    assert len(artifact["rows"]) == 960
    assert exp.validate_artifact(artifact, repo_root=exp.REPO_ROOT, check_files=True) == []
    assert all(path.exists() for path in paths.evidence_paths())

    changed = deepcopy(artifact)
    changed["decision_rows"][0]["decision_input"]["regime_id"] = "leak"
    changed["reproducibility_checksum"] = exp.payload_checksum(changed)
    assert "decision_view_leakage" in exp.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["rows"].pop()
    changed["reproducibility_checksum"] = exp.payload_checksum(changed)
    assert "arm_row_count_mismatch" in exp.validate_artifact(changed)


def test_scenario_cl_7183_seals_are_immutable_and_replay_is_stable(tmp_path: Path) -> None:
    """SCENARIO-CL-7183-READINESS rejects changed bytes and reproduces all view hashes."""

    paths = exp.StreamPaths.under(tmp_path)
    first = exp.build_and_seal(exp.REPO_ROOT, paths, run_date="20260910", duration_s=0.25)
    second = exp.build_and_seal(exp.REPO_ROOT, paths, run_date="20260910", duration_s=0.50)

    assert first["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert exp.replay_projection(exp.build_events()) == {
        "stream_hash": first["stream_hash"],
        "decision_view_hash": first["decision_view_hash"],
        "feedback_view_hash": first["feedback_view_hash"],
        "evaluator_truth_view_hash": first["evaluator_truth_view_hash"],
        "availability_matrix_hash": first["availability_matrix_hash"],
    }
    with pytest.raises(exp.ImmutableSealError):
        exp.write_immutable_jsonl(paths.decisions, [{"changed": True}])


def test_scenario_cl_7183_blocked_artifact_is_terminal(tmp_path: Path) -> None:
    """SCENARIO-CL-7183-PRECONDITIONS records a failed external gate exactly."""

    paths = exp.StreamPaths.under(tmp_path)
    checks = [
        exp.gate_check(
            "required_source_bytes",
            "python/carnot/missing.py",
            "missing",
            False,
            upstream="repository",
            field="source_bytes",
        )
    ]
    artifact = exp.build_blocked_artifact(checks, paths, run_date="20260910", duration_s=0.1)

    assert artifact["status"] == "blocked"
    assert artifact["stream_ready_score"] == 0
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["failed_check"] == "required_source_bytes"
    assert artifact["gate_check_summary"]["upstream"] == "repository"
    assert artifact["gate_check_summary"]["field"] == "source_bytes"
    assert exp.validate_artifact(artifact) == []


def test_req_cl_7183_command_writes_the_requested_artifact(tmp_path: Path) -> None:
    """REQ-CL-7183 exercises the file-to-parser-to-gate entrypoint path."""

    output = tmp_path / "result.json"
    evidence = tmp_path / "evidence"
    command = [
        sys.executable,
        str(exp.REPO_ROOT / "scripts/experiments/experiment_7183_v633_supersession_stream.py"),
        "--date",
        "20260910",
        "--output",
        str(output),
        "--evidence-root",
        str(evidence),
    ]
    completed = subprocess.run(command, cwd=exp.REPO_ROOT, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert "PHASE 0 START" in completed.stdout
    assert "PHASE 7 END" in completed.stdout
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["run_date"] == "20260910"
    assert artifact["stream_ready_score"] == 1
    assert exp.validate_artifact(artifact, repo_root=exp.REPO_ROOT, check_files=True) == []
