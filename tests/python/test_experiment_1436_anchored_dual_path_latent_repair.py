"""Tests for Exp 1436 anchored dual-path latent repair smoke diagnostic.

Spec refs: REQ-KONA-034, SCENARIO-KONA-034.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.phase3.anchored_dual_path_latent_repair import (
    REQUIRED_ARTIFACT_FIELDS,
    anchored_dual_path_acceptance,
    build_artifact,
    plan_anchored_dual_path_latent,
    run_comparison,
    run_raw_descent_metrics,
    write_experiment_artifact,
)
from carnot.phase3.latent_drift_smoke import (
    DEFAULT_SUPPORT_RADIUS,
    accuracy_for_decodes,
    build_tiny_cnf_tasks,
    decode_latent,
    direct_decode_dataset,
    encode_task,
    planning_energy,
)


def test_raw_descent_reproduces_exp1417_support_drift() -> None:
    """SCENARIO-KONA-034: raw descent retains the Exp 1417 failure baseline."""
    raw = run_raw_descent_metrics()

    assert raw.accuracy_before_planning == pytest.approx(1.0)
    assert raw.accuracy_after_planning == pytest.approx(0.25)
    assert raw.accuracy_delta_after_planning == pytest.approx(-0.75)
    assert raw.off_support_rate == pytest.approx(1.0)
    assert raw.latent_drift_norm > DEFAULT_SUPPORT_RADIUS
    assert raw.energy_monotone is True


def test_dual_path_gate_rejects_lower_energy_candidate_with_worse_decode() -> None:
    """REQ-KONA-034: decoded quality is checked before accepting lower energy."""
    task = build_tiny_cnf_tasks()[-1]
    current_z = encode_task(task)
    candidate_z = [0.4, 0.4]
    decision = anchored_dual_path_acceptance(task, current_z, candidate_z)

    assert planning_energy(candidate_z, current_z) < planning_energy(current_z, current_z)
    assert decision.energy_lowered is True
    assert decision.current_quality == pytest.approx(1.0)
    assert decision.candidate_quality == pytest.approx(0.0)
    assert decision.accepted is False


def test_anchored_dual_path_keeps_accuracy_and_reduces_off_support() -> None:
    """SCENARIO-KONA-034: anchoring repairs support drift on the tiny benchmark."""
    metrics = run_comparison()

    assert metrics.anchoring_applied is True
    assert metrics.dual_path_decoder_stub is True
    assert metrics.raw.off_support_rate == pytest.approx(1.0)
    assert metrics.off_support_rate == pytest.approx(0.0)
    assert metrics.off_support_rate < metrics.raw.off_support_rate
    assert metrics.accuracy_before_planning == pytest.approx(1.0)
    assert metrics.accuracy_after_planning == pytest.approx(1.0)
    assert metrics.accuracy_delta_after_planning == pytest.approx(0.0)
    assert metrics.latent_drift_norm < DEFAULT_SUPPORT_RADIUS
    assert metrics.energy_monotone is True
    assert metrics.anchored_repair_viable is True
    assert metrics.honest_verdict == "anchored_dual_path_repair_viable"


def test_anchored_planning_uses_same_tasks_as_direct_decode() -> None:
    """REQ-KONA-034: direct and anchored planned decodes share one dataset."""
    tasks = build_tiny_cnf_tasks()
    direct_decodes = direct_decode_dataset(tasks)
    planned = [plan_anchored_dual_path_latent(task) for task in tasks]
    planned_decodes = [decode_latent(row.z_t) for row in planned]

    assert len(planned) == len(direct_decodes) == len(tasks)
    assert all(row.anchoring_applied for row in planned)
    assert all(row.dual_path_decoder_stub for row in planned)
    assert accuracy_for_decodes(tasks, planned_decodes) == accuracy_for_decodes(
        tasks, direct_decodes
    )


def test_anchored_planner_counts_rejected_quality_regressions() -> None:
    """REQ-KONA-034: rejected candidates remain visible in trajectory diagnostics."""
    task = build_tiny_cnf_tasks()[-1]
    planned = plan_anchored_dual_path_latent(task, anchor_weight=0.02)

    assert planned.rejected_candidates > 0
    assert decode_latent(planned.z_t).supported is True


def test_build_artifact_contains_required_exp1436_fields() -> None:
    """SCENARIO-KONA-034: artifact schema contains every required field."""
    artifact = build_artifact()

    assert artifact["schema"] == "carnot.phase3.anchored_dual_path_latent_repair.v1"
    assert artifact["experiment"] == "1436_anchored_dual_path_latent_repair_v1"
    assert artifact["run_date"] == "20260506"
    assert artifact["spec_refs"] == ["REQ-KONA-034", "SCENARIO-KONA-034"]
    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete"
    assert artifact["anchoring_applied"] is True
    assert artifact["dual_path_decoder_stub"] is True
    assert artifact["anchored_repair_viable"] is True
    assert artifact["raw_descent"]["off_support_rate"] > artifact["off_support_rate"]
    json.dumps(artifact)


def test_write_experiment_artifact_round_trips_json(tmp_path: Path) -> None:
    """REQ-KONA-034: writer persists the measured complete artifact."""
    result_path = tmp_path / "experiment_1436.json"
    artifact = write_experiment_artifact(result_path)

    loaded = json.loads(result_path.read_text())
    assert loaded == artifact
    assert loaded["status"] == "complete"
