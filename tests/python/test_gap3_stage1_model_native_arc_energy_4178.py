"""Tests for Exp 4178 GAP-3 Stage-1 latent scoring.

Spec: REQ-VERIFY-4178, SCENARIO-VERIFY-4178.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.research import gap3_stage1_model_native_arc_energy_4178 as gap3
from carnot.research.gap3_stage1_model_native_arc_energy_4178 import (
    CandidateTable,
    Stage1Config,
    build_artifact,
    load_candidate_table,
    write_experiment_artifact,
)


def _toy_table(n_tasks: int = 6, candidates_per_task: int = 6, latent_dim: int = 8) -> CandidateTable:
    """Small table where latent correctness is separable but vote misses half the tasks."""

    rng = np.random.default_rng(123)
    rows = []
    task_idx = []
    votes = []
    correct = []
    q_mean = []
    committed_probe = []
    for task in range(n_tasks):
        for cand in range(candidates_per_task):
            is_gold = cand == 0
            z = np.zeros(latent_dim, dtype=np.float32)
            z[0] = 3.0 if is_gold else -2.5
            z[1] = 0.1 * task
            z[2] = rng.normal(0.0, 0.03)
            if not is_gold:
                z[3] = float(cand) / candidates_per_task
            rows.append(z)
            task_idx.append(task)
            correct.append(is_gold)
            if task % 2 == 0:
                votes.append(5 if is_gold else 20 - cand)
            else:
                votes.append(20 if is_gold else 10 - cand)
            q_mean.append(float(z[0]))
            committed_probe.append(float(-z[0]))

    return CandidateTable(
        z_mean=np.asarray(rows, dtype=np.float32),
        task_idx=np.asarray(task_idx, dtype=np.int32),
        votes=np.asarray(votes, dtype=np.int32),
        q_mean=np.asarray(q_mean, dtype=np.float64),
        probe=np.asarray(committed_probe, dtype=np.float64),
        correct=np.asarray(correct, dtype=bool),
    )


def _fast_config() -> Stage1Config:
    return Stage1Config(
        random_seed=4178,
        pca_components=3,
        bootstrap_resamples=80,
        permutation_resamples=3,
        logistic_max_iter=300,
    )


def test_build_artifact_reports_required_schema_and_gates():
    artifact = build_artifact(_toy_table(), _fast_config(), npz_path=Path("toy.npz"))

    required = {
        "honest_verdict",
        "pass2_energy_vs_vote",
        "candidate_auroc",
        "coverage_fraction",
        "headroom_capture_fraction",
        "adversarial_checks",
        "random_seed",
        "reproducibility_checksum",
    }
    assert required <= set(artifact)
    assert required <= set(artifact["field_principles"])
    assert artifact["honest_verdict"].startswith(("complete:", "success:"))
    assert artifact["pass2_energy_vs_vote"] > 0.0
    assert artifact["candidate_auroc"] > 0.70
    assert artifact["coverage_fraction"] == 1.0
    assert artifact["adversarial_checks"]["A4_bootstrap_ci95"]["ci95"]


def test_oracle_scrub_audit_keeps_held_out_scores_identical():
    artifact = build_artifact(_toy_table(), _fast_config(), npz_path=Path("toy.npz"))

    audit = artifact["adversarial_checks"]["A3_oracle_leak_audit"]
    assert audit["held_out_label_scrub_max_abs_diff"] == 0.0
    assert audit["passed"] is True
    assert artifact["preconditions_checked"]["no_gpu_used"] is True


def test_write_experiment_artifact_blocks_when_dump_missing(tmp_path: Path):
    output_path = tmp_path / "experiment_4178_gap3_stage1_model_native_arc_energy.json"

    written = write_experiment_artifact(
        npz_path=tmp_path / "missing.npz",
        output_path=output_path,
        config=_fast_config(),
    )
    artifact = json.loads(written.read_text())

    assert written == output_path
    assert artifact["honest_verdict"] == "blocked_stage1_latent_dump_missing"
    assert artifact["preconditions_checked"]["npz_exists"] is False


def test_load_candidate_table_rejects_invalid_width(tmp_path: Path):
    bad_path = tmp_path / "bad.npz"
    np.savez(
        bad_path,
        z_mean=np.zeros((2, 7), dtype=np.float32),
        task_idx=np.array([0, 0], dtype=np.int32),
        votes=np.array([1, 2], dtype=np.int32),
        q_mean=np.array([0.0, 1.0], dtype=np.float64),
        probe=np.array([0.0, 1.0], dtype=np.float64),
        correct=np.array([True, False]),
    )

    with pytest.raises(ValueError, match="latent width"):
        load_candidate_table(bad_path)


def test_valid_npz_write_and_precondition_validation_branches(tmp_path: Path):
    table = _toy_table(latent_dim=512)
    valid_path = tmp_path / "valid.npz"
    np.savez(
        valid_path,
        z_mean=table.z_mean,
        task_idx=table.task_idx,
        votes=table.votes,
        q_mean=table.q_mean,
        probe=table.probe,
        correct=table.correct,
    )
    output_path = tmp_path / "artifact.json"

    written = write_experiment_artifact(valid_path, output_path, _fast_config())
    artifact = json.loads(written.read_text())

    assert load_candidate_table(valid_path).z_mean.shape[1] == 512
    assert artifact["preconditions_checked"]["npz_exists"] is True
    assert artifact["reproducibility_checksum"]

    missing_key_path = tmp_path / "missing-key.npz"
    np.savez(missing_key_path, z_mean=np.zeros((1, 512), dtype=np.float32))
    with pytest.raises(ValueError, match="missing keys"):
        load_candidate_table(missing_key_path)

    with pytest.raises(ValueError, match="does not match"):
        CandidateTable(
            z_mean=np.zeros((2, 4), dtype=np.float32),
            task_idx=np.array([0]),
            votes=np.array([1, 2]),
            q_mean=np.array([0.0, 1.0]),
            probe=np.array([0.0, 1.0]),
            correct=np.array([True, False]),
        ).validate()
    with pytest.raises(ValueError, match="2-D"):
        CandidateTable(
            z_mean=np.zeros(2, dtype=np.float32),
            task_idx=np.array([0, 0]),
            votes=np.array([1, 2]),
            q_mean=np.array([0.0, 1.0]),
            probe=np.array([0.0, 1.0]),
            correct=np.array([True, False]),
        ).validate()
    with pytest.raises(ValueError, match="no correct"):
        CandidateTable(
            z_mean=np.zeros((2, 4), dtype=np.float32),
            task_idx=np.array([0, 0]),
            votes=np.array([1, 2]),
            q_mean=np.array([0.0, 1.0]),
            probe=np.array([0.0, 1.0]),
            correct=np.array([False, False]),
        ).validate()


def test_basis_audit_and_defensive_metric_branches():
    table = _toy_table()
    config = _fast_config()
    folds = gap3._fit_oof_fold_features(table, config)

    in_sample = gap3._in_sample_report(table, config, "model_native_basis_pca_gold_mahalanobis")
    audit = gap3._held_out_label_scrub_audit(
        folds,
        table.correct,
        config,
        "model_native_basis_pca_gold_mahalanobis",
        len(table.correct),
    )

    assert in_sample["coverage_fraction"] == 1.0
    assert audit["held_out_label_scrub_max_abs_diff"] == 0.0
    assert np.isnan(gap3._basis_gold_mahalanobis(np.zeros((2, 2)), np.array([False, False]), np.zeros((1, 2)))[0])
    assert np.isnan(gap3._logistic_probe_energy(np.zeros((2, 2)), np.array([True, True]), np.zeros((1, 2)), config)[0])
    assert np.isnan(gap3._binary_auc(np.array([False, False]), np.array([0.1, 0.2])))

    one_class_table = CandidateTable(
        z_mean=np.zeros((3, 2), dtype=np.float32),
        task_idx=np.array([0, 0, 1]),
        votes=np.array([1, 2, 3]),
        q_mean=np.zeros(3),
        probe=np.zeros(3),
        correct=np.array([False, False, False]),
    )
    assert gap3._within_task_auc(one_class_table, np.zeros(3))["macro"] != gap3._within_task_auc(
        one_class_table, np.zeros(3)
    )["macro"]
    assert gap3._chance_pass2_for_true_gold(one_class_table) == 0.0


def test_json_helpers_and_terminal_verdict_branches():
    assert gap3._verdict(False, "energy", 0.0, 0.4, 0.4).startswith("complete:")
    assert gap3._r(True) is True
    sentinel = object()
    assert gap3._r(sentinel) is sentinel
    assert gap3._r(float("nan")) is None
    assert gap3._json_ready(np.array([1, 2])) == [1, 2]
    assert gap3._json_ready(np.float64(1.25)) == 1.25
    assert gap3._json_ready(float("inf")) is None
