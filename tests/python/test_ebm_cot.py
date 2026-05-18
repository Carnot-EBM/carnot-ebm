"""Tests for Exp 2358 EBM-CoT consistency calibration.

Spec coverage: REQ-VERIFY-2358, SCENARIO-VERIFY-2358.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.verify.ebm_cot import (
    EbmCotCalibrator,
    build_experiment_2358_artifact,
    build_synthetic_cot_corpus,
    consistency_auroc,
    validate_experiment_2358_artifact,
    write_experiment_2358_artifact,
)


def test_req_verify_2358_energy_detects_adjacent_negated_claims() -> None:
    calibrator = EbmCotCalibrator(seed=42)
    consistent = [
        "Step 1: Claim: the switch is closed.",
        "Step 2: Claim: the switch is closed, so current flows.",
        "Step 3: Claim: current flows, so the bulb is lit.",
    ]
    inconsistent = [
        "Step 1: Claim: the switch is closed.",
        "Step 2: Claim: the switch is not closed, so current does not flow.",
        "Step 3: Claim: current does not flow, so the bulb is dark.",
    ]

    assert calibrator.energy(consistent) == 0.0
    assert calibrator.energy(inconsistent) > calibrator.energy(consistent)


def test_req_verify_2358_calibrate_returns_one_energy_per_trace() -> None:
    traces, _labels = build_synthetic_cot_corpus(random_seed=42)
    calibrator = EbmCotCalibrator(seed=42)

    energies = calibrator.calibrate(traces)

    assert len(energies) == len(traces)
    assert all(isinstance(energy, float) for energy in energies)
    assert all(energy >= 0.0 for energy in energies)


def test_scenario_verify_2358_synthetic_corpus_passes_auroc_gate() -> None:
    traces, labels = build_synthetic_cot_corpus(random_seed=42)
    calibrator = EbmCotCalibrator(seed=42)

    energies = calibrator.calibrate(traces)
    auroc = consistency_auroc(labels, energies)
    consistent_mean = sum(e for e, label in zip(energies, labels) if label == 1) / 25
    inconsistent_mean = sum(e for e, label in zip(energies, labels) if label == 0) / 25

    assert len(traces) == 50
    assert sum(labels) == 25
    assert auroc >= 0.60
    assert consistent_mean < inconsistent_mean


def test_req_verify_2358_langevin_refine_reduces_inconsistent_energy() -> None:
    traces, labels = build_synthetic_cot_corpus(random_seed=42)
    inconsistent_trace = next(trace for trace, label in zip(traces, labels) if label == 0)
    calibrator = EbmCotCalibrator(seed=42)

    before = calibrator.energy(inconsistent_trace)
    refined = calibrator.langevin_refine(inconsistent_trace, n_steps=50)
    after = calibrator.energy(refined)

    assert after <= before
    assert before - after > 0.0
    assert refined is not inconsistent_trace


def test_scenario_verify_2358_writes_terminal_artifact(tmp_path: Path) -> None:
    output = tmp_path / "experiment_2358_ebm_cot.json"

    artifact = write_experiment_2358_artifact(output_path=output)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    validate_experiment_2358_artifact(artifact)
    assert artifact["n_traces"] == 50
    assert artifact["random_seed"] == 42
    assert artifact["ebm_cot_validated"] is True
    assert artifact["ebm_cot_auroc"] >= 0.60
    assert artifact["energy_reduction_mean"] >= 0.0
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_2358_artifact_builder_is_deterministic() -> None:
    first = build_experiment_2358_artifact(random_seed=42)
    second = build_experiment_2358_artifact(random_seed=42)

    assert first == second
    assert first["ebm_cot_validated"] == (first["ebm_cot_auroc"] >= 0.60)
