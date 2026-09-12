"""Verify prospective bounded-coverage learning for REQ-CL-7254.

The tests use the sealed fixture or caller-owned temporary output paths. They
never rewrite the repository's evidence while they exercise the full writer.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7253_v638_coverage_memory as exp7253
from carnot import experiment_7254_v638_coverage_learning as exp7254
from carnot.memory import transactional_constraint_memory as transactional


def _synthetic_summary_rows() -> list[dict[str, object]]:
    """Build 32 paired streams whose treatment passes every efficacy gate."""

    values = {
        "frozen_warmup": (0.40, 0.10, 0.35),
        "reset_relearn": (0.36, 0.09, 0.31),
        "destructive_update": (0.34, 0.08, 0.30),
        "fifo_archive_shuffled": (0.32, 0.08, 0.28),
        "fifo_archive_aligned": (0.30, 0.08, 0.27),
        "coverage_archive_shuffled": (0.28, 0.07, 0.25),
        "coverage_archive_aligned": (0.20, 0.05, 0.21),
        "oracle_positive_control": (0.0, 0.0, 0.0),
    }
    rows: list[dict[str, object]] = []
    for seed in range(32):
        for arm in exp7253.ARMS:
            future, false_accept, recurrence = values[arm]
            rows.append(
                {
                    "stream_id": f"prospective-{seed + 1:02d}",
                    "seed": seed,
                    "arm": arm,
                    "future_error_rate": future,
                    "false_accept_rate": false_accept,
                    "recurrence_error_rate": recurrence,
                }
            )
    return rows


def test_paths_and_preconditions_authenticate_fixture(tmp_path: Path) -> None:
    """REQ-CL-7254: exact readiness and immutable contracts gate CPU work."""

    paths = exp7254.ExperimentPaths.under(tmp_path)
    checks, hashes, upstream = exp7254.collect_preconditions(exp7254.REPO_ROOT, paths)

    assert exp7254.gate_summary(checks)["passed"] is True
    assert upstream["coverage_fixture_ready_score"] == 1
    assert hashes[str(exp7254.REPO_ROOT / exp7254.DEFAULT_UPSTREAM_ARTIFACT)] == (
        exp7254.EXPECTED_UPSTREAM_SHA256
    )
    observed = {row["check"]: row for row in checks}
    assert observed["exp7253_controller_contract"]["passed"] is True
    assert observed["exp7253_arm_contract"]["passed"] is True
    assert observed["exp7253_stream_contract"]["passed"] is True
    assert paths.prequential_rows.parent.name == "experiment_7254"


def test_missing_upstream_is_terminal_blocked(tmp_path: Path) -> None:
    """SCENARIO-CL-7254-PRECONDITIONS: external absence emits no rows."""

    paths = exp7254.ExperimentPaths.under(tmp_path / "out")
    missing = tmp_path / "missing.json"
    artifact = exp7254.build_and_seal(
        exp7254.REPO_ROOT,
        paths,
        stream_ids=("prospective-01",),
        upstream_path=missing,
        progress=True,
    )

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["coverage_run_complete_score"] == 0
    assert artifact["gate_check_summary"]["failed_checks"]
    assert (
        exp7254.validate_artifact(
            artifact,
            repo_root=exp7254.REPO_ROOT,
            expected_stream_ids=("prospective-01",),
        )
        == []
    )


def test_prequential_single_stream_reduces_actual_commits(tmp_path: Path) -> None:
    """SCENARIO-CL-7254-PREQUENTIAL: commits affect only later decisions."""

    upstream = json.loads(
        (exp7254.REPO_ROOT / exp7254.DEFAULT_UPSTREAM_ARTIFACT).read_text(encoding="utf-8")
    )
    views = exp7254.load_stream_views(exp7254.REPO_ROOT, upstream)
    panel = exp7254.run_learning_panel(
        views,
        state_root=tmp_path / "state",
        stream_ids=("prospective-01",),
    )

    assert len(panel.prequential_rows) == exp7253.EVENTS_PER_STREAM * len(exp7253.ARMS)
    assert len(panel.rows) == len(exp7253.ARMS)
    assert exp7254.prequential_row_errors(panel.prequential_rows) == []
    committed = [row for row in panel.prequential_rows if row["commit_applied"]]
    assert committed
    assert all(row["commit_parent_hash"] and row["commit_child_hash"] for row in committed)
    assert all(row["commit_parent_hash"] != row["commit_child_hash"] for row in committed)
    assert panel.cost_rows
    assert panel.maximum_memory_bytes <= exp7253.MEMORY_CAPS["total_bytes"]
    assert {row["arm"] for row in panel.rows} == set(exp7253.ARMS)


def test_raw_reducer_rejects_leakage_and_bad_commit_hashes(tmp_path: Path) -> None:
    """SCENARIO-CL-7254-REDUCTION: raw chronology is independent authority."""

    upstream = json.loads(
        (exp7254.REPO_ROOT / exp7254.DEFAULT_UPSTREAM_ARTIFACT).read_text(encoding="utf-8")
    )
    views = exp7254.load_stream_views(exp7254.REPO_ROOT, upstream)
    panel = exp7254.run_learning_panel(
        views,
        state_root=tmp_path / "state",
        stream_ids=("prospective-02",),
    )
    raw_path = tmp_path / "rows.jsonl"
    raw_path.write_bytes(exp7253._jsonl_bytes(panel.prequential_rows))

    assert exp7254.independent_reduce(raw_path) == panel.rows
    changed = deepcopy(panel.prequential_rows)
    changed[0]["held_out_label_visible_to_controller"] = True
    assert "future_label_leakage" in exp7254.prequential_row_errors(changed)
    commit = next(row for row in changed if row["commit_applied"])
    commit["commit_child_hash"] = None
    assert "commit_hashes" in exp7254.prequential_row_errors(changed)
    with pytest.raises(ValueError, match="raw_rows_unavailable"):
        exp7254.independent_reduce(tmp_path / "absent.jsonl")


def test_bootstrap_gates_and_terminal_classification() -> None:
    """SCENARIO-CL-7254-CAUSAL: all strict paired gates remain independent."""

    rows = _synthetic_summary_rows()
    comparisons = exp7254.build_comparison_rows(rows, draws=100)
    causal = {
        "valid_reactivation_count": 3,
        "later_changed_decision_after_reactivation_count": 8,
        "pre_release_difference_count": 0,
        "prospective_shuffle_selection_change_count": 5,
        "oracle_headroom_event_count": 50,
        "constraint_addition_count": 20,
        "constraint_deactivation_count": 4,
    }
    gates = exp7254.score_acceptance_gates(comparisons, causal)

    assert len(comparisons) == len(exp7254.COMPARISON_SPECS)
    assert all(row["bootstrap_draws"] == 100 for row in comparisons)
    assert all(row["pass"] is True for row in gates.values())
    positive = exp7254.classify_result(gates, run_complete=True)
    assert positive["coverage_learning_value_score"] == 1
    assert positive["verdict_class"] == "circular_positive"

    failed = deepcopy(gates)
    failed["positive_aligned_vs_fifo_value"]["pass"] = False
    null = exp7254.classify_result(failed, run_complete=True)
    assert null["coverage_learning_value_score"] == 0
    assert null["verdict_class"] == "null"
    partial = exp7254.classify_result(gates, run_complete=False)
    assert partial["verdict_class"] == "partial"


def test_cost_targets_report_measured_results() -> None:
    """SCENARIO-CL-7254-COST: target gaps use observed operation costs."""

    rows = [
        {"operation": "lookup", "cost_ns": 800},
        {"operation": "lookup", "cost_ns": 1200},
        {"operation": "update", "cost_ns": 500},
        {"operation": "serialization", "cost_ns": 100_000},
        {"operation": "durable_commit", "cost_ns": 2_000_000},
    ]
    summary = exp7254.cost_summary(rows, maximum_memory_bytes=4096)

    assert summary["tier_1_update_target"]["passed"] is True
    assert summary["tier_2_lookup_target"]["passed"] is True
    assert summary["operations"]["durable_commit"]["p95_ns"] == 2_000_000
    assert summary["hardware_acceleration_100x"]["passed"] is False
    assert summary["bounded_bitset_call_graph"] is True


def test_e2e_restart_rejection_and_rollback(tmp_path: Path) -> None:
    """SCENARIO-CL-7254-E2E: restart and rollback retain exact state."""

    rows = exp7254.run_e2e_controls(tmp_path)

    assert {row["control"] for row in rows} == {
        "durable_restart_decision_parity",
        "rejected_update_preserves_bytes",
        "rollback_restores_parent",
    }
    assert all(row["passed"] is True for row in rows)


def test_build_validate_receipts_and_atomic_write(tmp_path: Path) -> None:
    """SCENARIO-CL-7254-TERMINAL: completion and value remain separate."""

    paths = exp7254.ExperimentPaths.under(tmp_path)
    artifact = exp7254.build_and_seal(
        exp7254.REPO_ROOT,
        paths,
        stream_ids=("prospective-03",),
        bootstrap_draws=100,
        progress=True,
    )

    assert artifact["status"] == "complete"
    assert artifact["coverage_run_complete_score"] == 1
    assert artifact["sample_size_budget"]["completed_arm_event_rows"] == 8192
    assert artifact["verdict_class"] in {"null", "circular_positive"}
    assert (
        exp7254.validate_artifact(
            artifact,
            repo_root=exp7254.REPO_ROOT,
            expected_stream_ids=("prospective-03",),
            check_files=True,
        )
        == []
    )

    receipt = {
        "command": "pytest focused",
        "exit_code": 0,
        "classification": "passed",
        "log_sha256": transactional.sha256_json("passed"),
    }
    attached = exp7254.attach_validation_receipts(artifact, [receipt])
    assert attached["validation_receipts"] == [receipt]
    assert attached["reproducibility_checksum"] != artifact["reproducibility_checksum"]
    exp7254.write_artifact(
        paths.artifact,
        attached,
        repo_root=exp7254.REPO_ROOT,
        expected_stream_ids=("prospective-03",),
    )
    assert json.loads(paths.artifact.read_text(encoding="utf-8"))["experiment_id"] == 7254

    malformed = deepcopy(attached)
    malformed["MODEL_SPECS"] = [{"name": "forbidden"}]
    assert "model_invocation" in exp7254.validate_artifact(
        malformed,
        expected_stream_ids=("prospective-03",),
    )
    with pytest.raises(ValueError, match="validation_receipt_schema"):
        exp7254.attach_validation_receipts(artifact, [{}])
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp7254.write_artifact(
            paths.artifact,
            malformed,
            expected_stream_ids=("prospective-03",),
        )

    unfinished = deepcopy(attached)
    unfinished["status"] = "in_progress"
    unfinished["reproducibility_checksum"] = exp7254.reproducibility_checksum(unfinished)
    assert "status" in exp7254.validate_artifact(
        unfinished,
        expected_stream_ids=("prospective-03",),
    )


def test_cli_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-CL-7254: the entrypoint keeps fixed date and delegates orchestration."""

    assert exp7254._parse_args(["--date", "20260912", "--output-root", str(tmp_path)]).date == (
        "20260912"
    )
    with pytest.raises(SystemExit, match="run_date_must_be_20260912"):
        exp7254.main(["--date", "20260911"])

    artifact = {"status": "blocked"}
    monkeypatch.setattr(exp7254, "build_and_seal", lambda *args, **kwargs: artifact)
    monkeypatch.setattr(exp7254, "validate_artifact", lambda *args, **kwargs: [])
    written: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        exp7254,
        "write_artifact",
        lambda path, value, **kwargs: written.append((path, value)),
    )
    assert exp7254.main(["--date", "20260912", "--output-root", str(tmp_path)]) == 0
    assert written == [(exp7254.ExperimentPaths.under(tmp_path).artifact, artifact)]

    monkeypatch.setattr(exp7254, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
        exp7254.main(["--date", "20260912", "--output-root", str(tmp_path)])


def test_defensive_failures_are_exercised(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-CL-7254: malformed inputs and failed controls stop publication."""

    assert (
        exp7254.ExperimentPaths.defaults().artifact == exp7254.REPO_ROOT / exp7254.DEFAULT_ARTIFACT
    )
    non_object = tmp_path / "non-object.jsonl"
    non_object.write_text("[]\n", encoding="utf-8")
    assert exp7254._read_jsonl(non_object) == []

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw_row_validation_failed"):
        exp7254.independent_reduce(invalid)

    upstream = json.loads(
        (exp7254.REPO_ROOT / exp7254.DEFAULT_UPSTREAM_ARTIFACT).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(exp7253, "stream_conformance_errors", lambda *args: ["forced"])
    with pytest.raises(ValueError, match="exp7253_stream_conformance:forced"):
        exp7254.load_stream_views(exp7254.REPO_ROOT, upstream)
    monkeypatch.undo()

    views = exp7254.load_stream_views(exp7254.REPO_ROOT, upstream)
    with pytest.raises(ValueError, match="incomplete_stream:missing"):
        exp7254.run_learning_panel(
            views,
            state_root=tmp_path / "missing-state",
            stream_ids=("missing",),
        )

    panel = exp7254.run_learning_panel(
        views,
        state_root=tmp_path / "state",
        stream_ids=("prospective-04",),
        progress=True,
    )
    monkeypatch.setattr(exp7254, "prequential_row_errors", lambda rows: ["forced"])
    with pytest.raises(ValueError, match="prequential_conformance:forced"):
        exp7254.run_learning_panel(
            views,
            state_root=tmp_path / "bad-state",
            stream_ids=("prospective-05",),
        )
    monkeypatch.undo()

    paths = exp7254.ExperimentPaths.under(tmp_path / "build")
    monkeypatch.setattr(exp7254, "load_stream_views", lambda *args: views)
    monkeypatch.setattr(exp7254, "run_learning_panel", lambda *args, **kwargs: panel)
    monkeypatch.setattr(
        exp7254,
        "run_e2e_controls",
        lambda *args: [{"control": "forced", "passed": False}],
    )
    with pytest.raises(ValueError, match="e2e_control_failed"):
        exp7254.build_and_seal(
            exp7254.REPO_ROOT,
            paths,
            stream_ids=("prospective-04",),
            bootstrap_draws=10,
        )

    monkeypatch.setattr(
        exp7254,
        "run_e2e_controls",
        lambda *args: [{"control": f"control-{index}", "passed": True} for index in range(3)],
    )
    monkeypatch.setattr(exp7254, "independent_reduce", lambda *args: [{"forced": True}])
    with pytest.raises(ValueError, match="independent_reducer_mismatch"):
        exp7254.build_and_seal(
            exp7254.REPO_ROOT,
            paths,
            stream_ids=("prospective-04",),
            bootstrap_draws=10,
        )

    monkeypatch.setattr(exp7254, "independent_reduce", lambda *args: panel.rows)
    monkeypatch.setattr(exp7254, "validate_artifact", lambda *args, **kwargs: ["forced"])
    with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
        exp7254.build_and_seal(
            exp7254.REPO_ROOT,
            paths,
            stream_ids=("prospective-04",),
            bootstrap_draws=10,
        )
