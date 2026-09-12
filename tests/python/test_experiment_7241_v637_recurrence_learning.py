"""Tests for prospective recurrence learning value.

Spec refs: REQ-CL-7241 and SCENARIO-CL-7241-*.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pytest

from carnot import experiment_7240_v637_recurrence_fixture as exp7240
from carnot import experiment_7241_v637_recurrence_learning as exp7241


@pytest.fixture(scope="module")
def sealed_views() -> exp7240.StreamViews:
    """Load the real separated fixture once for the focused tests."""

    upstream = exp7241._load_object(exp7241.REPO_ROOT / exp7241.DEFAULT_UPSTREAM_ARTIFACT)
    return exp7241.load_stream_views(exp7241.REPO_ROOT, upstream)


@pytest.fixture(scope="module")
def one_stream_panel(sealed_views: exp7240.StreamViews) -> exp7241.LearningPanel:
    """Exercise every arm on one complete stream without repeating the CPU replay."""

    return exp7241.run_learning_panel(
        sealed_views,
        stream_ids=("stream-01",),
        progress=True,
    )


def test_preconditions_authenticate_exact_fixture_and_quarantine(tmp_path: Path) -> None:
    """REQ-CL-7241 / SCENARIO-CL-7241-PRECONDITIONS."""

    assert exp7241.unwrap_principled({"principle": "why", "value": 3}) == 3
    ordinary = {"principle": "why", "value": 3, "evidence": "keep"}
    assert exp7241.unwrap_principled(ordinary) is ordinary

    paths = exp7241.ExperimentPaths.under(tmp_path)
    checks, hashes, upstream = exp7241.collect_preconditions(exp7241.REPO_ROOT, paths)
    assert exp7241.gate_summary(checks)["passed"] is True
    upstream_path = exp7241.REPO_ROOT / exp7241.DEFAULT_UPSTREAM_ARTIFACT
    assert hashes[str(upstream_path)] == exp7241.EXPECTED_UPSTREAM_SHA256
    assert upstream["recurrence_fixture_ready_score"] == 1
    assert exp7241.ExperimentPaths.defaults().artifact == (
        exp7241.REPO_ROOT / exp7241.DEFAULT_ARTIFACT
    )

    changed = deepcopy(upstream)
    changed["status"] = "blocked"
    with patch.object(exp7241, "_load_object", return_value=changed):
        failed, _, _ = exp7241.collect_preconditions(exp7241.REPO_ROOT, paths)
    assert exp7241.gate_summary(failed)["failed_check"] == "exp7240_status"

    with patch.object(
        exp7241,
        "quarantine_state",
        return_value={"quarantined": True, "matches": ["test quarantine"]},
    ):
        quarantined, _, _ = exp7241.collect_preconditions(exp7241.REPO_ROOT, paths)
    assert exp7241.gate_summary(quarantined)["passed"] is False


def test_real_stream_views_are_sealed_and_conform(sealed_views: exp7240.StreamViews) -> None:
    """REQ-CL-7241 / SCENARIO-CL-7241-PRECONDITIONS and -CHRONOLOGY."""

    assert exp7240.stream_conformance_errors(sealed_views) == []
    assert len(sealed_views.public) == exp7240.STREAM_COUNT * exp7240.EVENTS_PER_STREAM
    assert exp7240.public_leakage_errors(sealed_views.public) == []
    assert set(sealed_views.public[0]) == set(sealed_views.manifest["public_fields"])
    assert "exact_label" not in sealed_views.manifest["controller_input_fields"]

    malformed = exp7241.REPO_ROOT / "results/streams/experiment_7240/missing.jsonl"
    assert exp7241._read_jsonl(malformed) == []
    assert exp7241._sha256_path(malformed) is None


def test_panel_retains_prospective_rows_costs_and_state(
    one_stream_panel: exp7241.LearningPanel,
) -> None:
    """REQ-CL-7241 / SCENARIO-CL-7241-CHRONOLOGY, -PANEL, -METRICS, -STATE."""

    panel = one_stream_panel
    assert len(panel.decision_rows) == exp7240.EVENTS_PER_STREAM * len(exp7240.ARMS)
    assert len(panel.paired_seed_rows) == len(exp7240.ARMS)
    assert len(panel.state_entries) == len(exp7240.ARMS)
    assert (
        exp7241.panel_conformance_errors(
            panel,
            expected_stream_ids=("stream-01",),
        )
        == []
    )

    decision = panel.decision_rows[0]
    assert decision["prediction_receipt_completed_ns"] <= decision["label_accessed_ns"]
    assert decision["query_receipt_completed_ns"] <= decision["label_accessed_ns"]
    assert decision["controller_input_fields"] == ["event_id", "family_id", "numeric_value"]
    assert decision["held_out_label_visible_to_controller"] is False
    assert decision["full_denominator_error"] == max(
        decision["classification_error"], decision["abstention"]
    )
    assert decision["prediction_receipt_sha256"].startswith("sha256:")
    assert decision["label_release_receipt_sha256"].startswith("sha256:")

    operation_kinds = {row["operation"] for row in panel.operation_receipts}
    assert {"query", "delayed_delivery", "validation_and_state_write"} <= operation_kinds
    assert all(row["cost_ns"] >= 0 for row in panel.operation_receipts)
    assert all(str(row["source_sha256"]).startswith("sha256:") for row in panel.operation_receipts)
    assert all(entry["final_state_bytes_b64"] for entry in panel.state_entries)
    assert panel.causal_summary["pre_release_difference_count"] == 0

    latency = exp7241.latency_summary(panel, total_cpu_time_s=1.25)
    assert latency["total_cpu_time_s"] == 1.25
    assert latency["operations"]["lookup_ns"]["count"] == len(panel.decision_rows)
    assert latency["hardware_target_x"] == 100.0
    assert latency["hardware_target_met"] is False


def _synthetic_seed_rows() -> list[dict[str, object]]:
    """Build paired stream rows with clear positive learning headroom."""

    rows: list[dict[str, object]] = []
    for seed in range(32):
        for arm in exp7240.ARMS:
            future = 0.10 if arm == "validation_selected_archive" else 0.30
            recurrence = 0.05 if arm == "validation_selected_archive" else 0.25
            false_accept = 0.01 if arm == "validation_selected_archive" else 0.04
            rows.append(
                {
                    "stream_id": f"stream-{seed + 1:02d}",
                    "seed": seed,
                    "arm": arm,
                    "future_error_rate": future,
                    "false_accept_rate": false_accept,
                    "abstention_rate": 0.0,
                    "recurrence_error_rate": recurrence,
                    "valid_reactivation_count": int(arm == "validation_selected_archive"),
                    "later_changed_decision_count": int(arm == "validation_selected_archive"),
                    "pre_release_difference_count": 0,
                    "archive_hit_count": int(arm == "validation_selected_archive"),
                    "archive_hit_valid_count": int(arm == "validation_selected_archive"),
                }
            )
    return rows


def test_paired_bootstrap_and_frozen_gates_are_deterministic() -> None:
    """REQ-CL-7241 / SCENARIO-CL-7241-BOOTSTRAP and -GATES."""

    rows = _synthetic_seed_rows()
    first = exp7241.build_comparison_rows(rows, draws=100)
    second = exp7241.build_comparison_rows(rows, draws=100)
    assert first == second
    assert len(first) == 7
    assert all(row["independent_unit"] == "stream_seed" for row in first)
    assert all(row["bootstrap_draws"] == 100 for row in first)

    causal = {
        "valid_reactivation_count": 32,
        "later_changed_decision_count": 32,
        "pre_release_difference_count": 0,
        "positive_control_changed_decision_count": 32,
    }
    gates = exp7241.score_acceptance_gates(first, causal)
    assert all(row["pass"] is True for row in gates.values())
    classification = exp7241.classify_result(gates, run_complete=True)
    assert classification["recurrence_learning_value_score"] == 1
    assert classification["verdict_class"] == "circular_positive"

    no_headroom = dict(causal, later_changed_decision_count=0)
    null_gates = exp7241.score_acceptance_gates(first, no_headroom)
    null_result = exp7241.classify_result(null_gates, run_complete=True)
    assert null_result["recurrence_learning_value_score"] == 0
    assert null_result["verdict_class"] == "null"
    assert "inconclusive" in null_result["honest_verdict"]
    assert null_result["family_retired"] is False


def test_e2e_rejected_updates_and_byte_identical_rollback(tmp_path: Path) -> None:
    """REQ-CL-7241 / SCENARIO-CL-7241-STATE E2E-007."""

    rows = exp7241.run_e2e_controls(tmp_path)
    assert {row["control"] for row in rows} == {
        "rejected_stale_parent",
        "byte_identical_rollback",
        "fresh_process_restore",
    }
    assert all(row["passed"] is True for row in rows)
    rollback = next(row for row in rows if row["control"] == "byte_identical_rollback")
    assert rollback["byte_identical"] is True


def test_build_one_stream_artifact_and_blocked_contract(
    tmp_path: Path,
) -> None:
    """REQ-CL-7241 / SCENARIO-CL-7241-NULL and -TERMINAL."""

    paths = exp7241.ExperimentPaths.under(tmp_path)
    artifact = exp7241.build_and_seal(
        exp7241.REPO_ROOT,
        paths,
        stream_ids=("stream-01",),
        bootstrap_draws=100,
        progress=True,
    )
    assert artifact["status"] == "complete"
    assert artifact["recurrence_run_complete_score"] == 1
    assert artifact["recurrence_learning_value_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["sample_size_budget"]["completed_arm_event_rows"] == 6_144
    assert (
        exp7241.validate_artifact(
            artifact,
            repo_root=exp7241.REPO_ROOT,
            expected_stream_ids=("stream-01",),
            check_files=True,
        )
        == []
    )
    exp7241.write_artifact(
        paths.artifact,
        artifact,
        repo_root=exp7241.REPO_ROOT,
        expected_stream_ids=("stream-01",),
    )
    assert json.loads(paths.artifact.read_text(encoding="utf-8"))["status"] == "complete"

    changed = deepcopy(artifact)
    changed["rows"] = []
    assert "paired_seed_rows" in exp7241.validate_artifact(
        changed,
        expected_stream_ids=("stream-01",),
    )

    checks, hashes, upstream = exp7241.collect_preconditions(exp7241.REPO_ROOT, paths)
    checks[0] = dict(checks[0], passed=False, observed_value=False)
    blocked = exp7241.build_blocked_artifact(checks, hashes, upstream, paths)
    assert blocked["status"] == "blocked"
    assert blocked["rows"] == []
    assert blocked["inference_substrate"] == "blocked_no_run"
    assert (
        exp7241.validate_artifact(
            blocked,
            expected_stream_ids=("stream-01",),
        )
        == []
    )


def test_command_entrypoints_delegate_and_reject_wrong_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7241 / SCENARIO-CL-7241-TERMINAL command boundary."""

    with pytest.raises(SystemExit, match="run_date_must_be_20260912"):
        exp7241.main(["--date", "20260911"])

    with (
        patch.object(exp7241, "build_and_seal", return_value={}) as build,
        patch.object(exp7241, "validate_artifact", return_value=[]),
        patch.object(exp7241, "write_artifact") as write,
    ):
        assert exp7241.main(["--date", exp7241.RUN_DATE, "--output-root", str(tmp_path)]) == 0
        assert exp7241.main(["--date", exp7241.RUN_DATE]) == 0
    assert build.call_count == 2
    assert write.call_count == 2

    wrapper = exp7241.REPO_ROOT / "scripts/experiments/experiment_7241_v637_recurrence_learning.py"
    monkeypatch.setattr(sys, "argv", [str(wrapper), "--date", exp7241.RUN_DATE])
    with patch.object(exp7241, "main", return_value=0) as delegated:
        with pytest.raises(SystemExit) as stopped:
            runpy.run_path(str(wrapper), run_name="__main__")
    assert stopped.value.code == 0
    delegated.assert_called_once_with()


def test_malformed_evidence_empty_statistics_and_conformance_errors(
    tmp_path: Path,
    one_stream_panel: exp7241.LearningPanel,
) -> None:
    """REQ-CL-7241 / SCENARIO-CL-7241-PRECONDITIONS, -METRICS negative paths."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp7241._load_object(malformed) == {}
    assert exp7241.load_stream_views(exp7241.REPO_ROOT, {"stream_receipts": []}) == (
        exp7240.StreamViews([], [], [], {})
    )
    assert exp7241._percentile([], 0.5) == 0.0
    assert exp7241._bootstrap_interval([], draws=10, salt="empty") == {
        "estimate": 0.0,
        "ci95": [0.0, 0.0],
    }

    broken = replace(one_stream_panel, decision_rows=[])
    assert "decision_row_count" in exp7241.panel_conformance_errors(
        broken,
        expected_stream_ids=("stream-01",),
    )

    paths = exp7241.ExperimentPaths.under(tmp_path)
    with patch.object(exp7240, "reproducibility_checksum", side_effect=TypeError("bad")):
        checks, _, _ = exp7241.collect_preconditions(exp7241.REPO_ROOT, paths)
    assert next(row for row in checks if row["check"] == "exp7240_checksum")["passed"] is False


def test_build_block_and_fail_closed_stage_boundaries(
    tmp_path: Path,
    sealed_views: exp7240.StreamViews,
    one_stream_panel: exp7241.LearningPanel,
) -> None:
    """REQ-CL-7241 / SCENARIO-CL-7241-PRECONDITIONS and -TERMINAL failures."""

    paths = exp7241.ExperimentPaths.under(tmp_path / "blocked")
    checks, hashes, upstream = exp7241.collect_preconditions(exp7241.REPO_ROOT, paths)
    failed_checks = [dict(row) for row in checks]
    failed_checks[0] = dict(failed_checks[0], passed=False, observed_value=False)
    with patch.object(
        exp7241,
        "collect_preconditions",
        return_value=(failed_checks, hashes, upstream),
    ):
        blocked = exp7241.build_and_seal(
            exp7241.REPO_ROOT,
            paths,
            stream_ids=("stream-01",),
            bootstrap_draws=10,
            progress=True,
        )
    assert blocked["status"] == "blocked"

    base_patches = (
        patch.object(exp7241, "collect_preconditions", return_value=(checks, hashes, upstream)),
        patch.object(exp7241, "load_stream_views", return_value=sealed_views),
    )
    with (
        base_patches[0],
        base_patches[1],
        patch.object(exp7240, "stream_conformance_errors", return_value=["bad"]),
    ):
        with pytest.raises(ValueError, match="exp7240_stream_conformance"):
            exp7241.build_and_seal(
                exp7241.REPO_ROOT,
                exp7241.ExperimentPaths.under(tmp_path / "stream"),
                stream_ids=("stream-01",),
                bootstrap_draws=10,
            )

    with (
        patch.object(exp7241, "collect_preconditions", return_value=(checks, hashes, upstream)),
        patch.object(exp7241, "load_stream_views", return_value=sealed_views),
        patch.object(exp7241, "run_learning_panel", return_value=one_stream_panel),
        patch.object(exp7241, "panel_conformance_errors", return_value=["bad"]),
    ):
        with pytest.raises(ValueError, match="learning_panel_conformance"):
            exp7241.build_and_seal(
                exp7241.REPO_ROOT,
                exp7241.ExperimentPaths.under(tmp_path / "panel"),
                stream_ids=("stream-01",),
                bootstrap_draws=10,
            )

    with (
        patch.object(exp7241, "collect_preconditions", return_value=(checks, hashes, upstream)),
        patch.object(exp7241, "load_stream_views", return_value=sealed_views),
        patch.object(exp7241, "run_learning_panel", return_value=one_stream_panel),
        patch.object(exp7241, "run_e2e_controls", return_value=[{"passed": False}]),
    ):
        with pytest.raises(ValueError, match="e2e_control_failed"):
            exp7241.build_and_seal(
                exp7241.REPO_ROOT,
                exp7241.ExperimentPaths.under(tmp_path / "e2e"),
                stream_ids=("stream-01",),
                bootstrap_draws=10,
            )

    with (
        patch.object(exp7241, "collect_preconditions", return_value=(checks, hashes, upstream)),
        patch.object(exp7241, "load_stream_views", return_value=sealed_views),
        patch.object(exp7241, "run_learning_panel", return_value=one_stream_panel),
        patch.object(exp7241, "validate_artifact", return_value=["bad"]),
    ):
        with pytest.raises(ValueError, match="artifact_validation_failed"):
            exp7241.build_and_seal(
                exp7241.REPO_ROOT,
                exp7241.ExperimentPaths.under(tmp_path / "artifact"),
                stream_ids=("stream-01",),
                bootstrap_draws=10,
            )


def test_write_and_main_reject_invalid_terminal_artifacts(tmp_path: Path) -> None:
    """REQ-CL-7241 / SCENARIO-CL-7241-TERMINAL fail-closed publication."""

    with pytest.raises(ValueError, match="artifact_validation_failed"):
        exp7241.write_artifact(tmp_path / "bad.json", {})
    with (
        patch.object(exp7241, "build_and_seal", return_value={}),
        patch.object(exp7241, "validate_artifact", return_value=["bad"]),
    ):
        with pytest.raises(ValueError, match="artifact_validation_failed"):
            exp7241.main(["--date", exp7241.RUN_DATE, "--output-root", str(tmp_path)])
