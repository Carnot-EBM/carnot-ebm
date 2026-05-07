"""Tests for Exp 1450 EBT/NRGPT local energy-convergence microprototype.

Spec refs: REQ-KONA-035, SCENARIO-KONA-035.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from carnot.phase3.ebt_nrgpt_local_microprototype_audit import (
    LocalTraceEnergy,
    REQUIRED_ARTIFACT_FIELDS,
    anchored_reference_energy_delta,
    build_artifact,
    choose_scale_recommendation,
    load_smoke_traces,
    run_energy_convergence_probe,
    write_experiment_artifact,
)


def _rows() -> list[dict[str, object]]:
    return [
        {"step_text": "2 + 3 = 7, therefore 5 * 2 = 3", "label": "incorrect"},
        {"step_text": "2 + 3 = 5, therefore 5 * 2 = 10", "label": "correct"},
        {"step_text": "4 * 2 = 8, so 8 - 3 = 5", "label": "correct"},
        {"step_text": "4 * 2 = 9, so 9 - 3 = 1", "label": "incorrect"},
    ]


def test_local_probe_energy_converges_on_existing_trace_rows() -> None:
    """SCENARIO-KONA-035: local trace embeddings expose energy convergence."""
    traces = load_smoke_traces(rows=_rows(), max_traces=4)
    summary = run_energy_convergence_probe(traces, max_steps=8, step_size=0.2)

    assert summary.energy_convergence_probe_complete is True
    assert summary.traces_evaluated == 4
    assert summary.baseline_energy_delta < 0.0
    assert summary.convergence_steps_median >= 1.0
    assert all(row.energy_trace[-1] <= row.energy_trace[0] for row in summary.results)


def test_anchored_reference_delta_is_loaded_from_exp1436_trace(tmp_path: Path) -> None:
    """REQ-KONA-035: the comparator reads the anchored repair energy reference."""
    reference_path = tmp_path / "experiment_1436.json"
    reference_path.write_text(json.dumps({"energy_trace": [2.0, 1.25, 1.0]}))

    assert anchored_reference_energy_delta(reference_path) == pytest.approx(-1.0)


def test_local_trace_energy_reports_input_dim() -> None:
    """REQ-KONA-035: the local energy model follows the EnergyFunction shape API."""
    energy = LocalTraceEnergy(target=jnp.zeros(3), anchor=jnp.ones(3))

    assert energy.input_dim == 3


def test_scale_recommendation_keeps_smoke_only_without_quality_evidence() -> None:
    """REQ-KONA-035: lower energy alone is not enough to scale Phase-3 work."""
    recommendation = choose_scale_recommendation(
        traces_evaluated=4,
        baseline_energy_delta=-0.5,
        anchored_repair_energy_delta_reference=-1.0,
        convergence_steps_median=3.0,
        decoded_quality_evidence=False,
    )

    assert recommendation == "keep_smoke_only"


def test_scale_recommendation_can_retire_or_scale_when_gates_change() -> None:
    """REQ-KONA-035: the recommendation is derived from measured gates."""
    retire = choose_scale_recommendation(
        traces_evaluated=0,
        baseline_energy_delta=0.0,
        anchored_repair_energy_delta_reference=-1.0,
        convergence_steps_median=0.0,
        decoded_quality_evidence=False,
    )
    scale = choose_scale_recommendation(
        traces_evaluated=32,
        baseline_energy_delta=-1.2,
        anchored_repair_energy_delta_reference=-1.0,
        convergence_steps_median=6.0,
        decoded_quality_evidence=True,
    )

    assert retire == "retire"
    assert scale == "scale_future_milestone"


def test_build_artifact_contains_required_exp1450_fields(tmp_path: Path) -> None:
    """SCENARIO-KONA-035: artifact schema contains every required field."""
    reference_path = tmp_path / "experiment_1436.json"
    reference_path.write_text(json.dumps({"energy_trace": [2.0, 1.0]}))

    artifact = build_artifact(
        rows=_rows(),
        anchored_reference_path=reference_path,
        commands_run=(
            "pytest tests/python/test_experiment_1450_ebt_nrgpt_local_microprototype_audit.py -q",
        ),
        max_traces=4,
    )

    assert artifact["schema"] == "carnot.phase3.ebt_nrgpt_local_microprototype_audit.v1"
    assert artifact["experiment"] == "1450_ebt_nrgpt_local_microprototype_audit"
    assert artifact["run_date"] == "20260507"
    assert artifact["spec_refs"] == ["REQ-KONA-035", "SCENARIO-KONA-035"]
    assert set(REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["status"] == "complete"
    assert artifact["energy_convergence_probe_complete"] is True
    assert artifact["traces_evaluated"] == 4
    assert artifact["baseline_energy_delta"] < 0.0
    assert artifact["anchored_repair_energy_delta_reference"] == pytest.approx(-1.0)
    assert artifact["scale_recommendation"] == "keep_smoke_only"
    assert "smoke" in artifact["honest_verdict"]
    json.dumps(artifact)


def test_write_experiment_artifact_round_trips_json(tmp_path: Path) -> None:
    """REQ-KONA-035: writer persists the measured complete artifact."""
    result_path = tmp_path / "experiment_1450.json"
    reference_path = tmp_path / "experiment_1436.json"
    reference_path.write_text(json.dumps({"energy_trace": [2.0, 1.0]}))

    artifact = write_experiment_artifact(
        result_path,
        rows=_rows(),
        anchored_reference_path=reference_path,
    )

    loaded = json.loads(result_path.read_text())
    assert loaded == artifact
    assert loaded["status"] == "complete"
