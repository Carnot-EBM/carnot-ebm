"""Tests for Exp 1417 EBRM latent trajectory drift smoke diagnostic.

Spec refs: REQ-KONA-033, SCENARIO-KONA-033.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.phase3.latent_drift_smoke import (
    DEFAULT_SUPPORT_RADIUS,
    REQUIRED_ARTIFACT_FIELDS,
    accuracy_for_decodes,
    build_artifact,
    build_tiny_cnf_tasks,
    decode_latent,
    direct_decode_dataset,
    encode_task,
    energy_trace_is_monotone,
    mean_energy_trace,
    mean_latent_drift_norm,
    plan_latent,
    planned_decode_dataset,
    run_smoke,
    write_experiment_artifact,
)


def test_direct_decode_solves_tiny_cnf_dataset() -> None:
    """REQ-KONA-033: direct decode uses encoder support anchors."""
    tasks = build_tiny_cnf_tasks()
    direct_decodes = direct_decode_dataset(tasks)

    assert len(tasks) == 4
    assert all(decode.supported for decode in direct_decodes)
    assert [decode.assignment for decode in direct_decodes] == [
        task.target_assignment for task in tasks
    ]
    assert accuracy_for_decodes(tasks, direct_decodes) == pytest.approx(1.0)


def test_planning_lowers_energy_while_decoded_accuracy_regresses() -> None:
    """SCENARIO-KONA-033: monotone lower energy can still drift off decoder support."""
    tasks = build_tiny_cnf_tasks()
    planning_results = [plan_latent(encode_task(task)) for task in tasks]
    direct_decodes = direct_decode_dataset(tasks)
    planned_decodes = planned_decode_dataset(tasks)
    energy_trace = mean_energy_trace(planning_results)

    assert energy_trace_is_monotone(energy_trace) is True
    assert energy_trace[-1] < energy_trace[0]
    assert accuracy_for_decodes(tasks, planned_decodes) < accuracy_for_decodes(
        tasks, direct_decodes
    )
    assert mean_latent_drift_norm(planning_results) > DEFAULT_SUPPORT_RADIUS
    assert all(not decode.supported for decode in planned_decodes)


def test_decoder_marks_off_support_latents_with_fallback_assignment() -> None:
    """REQ-KONA-033: decoder support radius makes latent distribution shift visible."""
    tasks = build_tiny_cnf_tasks()
    anchor_decode = decode_latent(encode_task(tasks[-1]))
    off_support_decode = decode_latent([0.0, 0.0])

    assert anchor_decode.supported is True
    assert anchor_decode.assignment == tasks[-1].target_assignment
    assert off_support_decode.supported is False
    assert off_support_decode.assignment == (False, False)
    assert off_support_decode.distance_to_support > DEFAULT_SUPPORT_RADIUS


def test_run_smoke_sets_dual_path_and_anchoring_gates() -> None:
    """REQ-KONA-033: verdict gates follow measured energy, accuracy, and drift."""
    metrics = run_smoke()

    assert metrics.status == "complete"
    assert metrics.latent_drift_smoke_complete is True
    assert metrics.task_family == "synthetic_two_variable_cnf"
    assert metrics.energy_monotone is True
    assert metrics.accuracy_before_planning == pytest.approx(1.0)
    assert metrics.accuracy_after_planning == pytest.approx(0.25)
    assert metrics.accuracy_delta_after_planning == pytest.approx(-0.75)
    assert metrics.latent_drift_norm > DEFAULT_SUPPORT_RADIUS
    assert metrics.dual_path_decoder_required is True
    assert metrics.anchoring_required is True
    assert metrics.honest_verdict == "energy_down_accuracy_down_off_decoder_support"


def test_build_artifact_contains_required_exp1417_fields() -> None:
    """SCENARIO-KONA-033: artifact schema contains every required field."""
    artifact = build_artifact()

    assert artifact["schema"] == "carnot.phase3.ebrm_latent_drift_smoke.v1"
    assert artifact["experiment"] == "1417_ebrm_latent_trajectory_drift_smoke"
    assert artifact["run_date"] == "20260506"
    assert artifact["spec_refs"] == ["REQ-KONA-033", "SCENARIO-KONA-033"]
    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete"
    assert artifact["latent_drift_smoke_complete"] is True
    assert artifact["energy_monotone"] is True
    assert artifact["dual_path_decoder_required"] is True
    assert artifact["anchoring_required"] is True
    assert artifact["energy_trace"][0] > artifact["energy_trace"][-1]
    json.dumps(artifact)


def test_write_experiment_artifact_round_trips_json(tmp_path: Path) -> None:
    """REQ-KONA-033: writer persists the complete measured artifact."""
    result_path = tmp_path / "experiment_1417.json"
    artifact = write_experiment_artifact(result_path)

    loaded = json.loads(result_path.read_text())
    assert loaded == artifact
    assert loaded["status"] == "complete"
