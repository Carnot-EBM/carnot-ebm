"""Tests for the Latent Symbol Bridge experiment (3819).

Spec refs: REQ-3819, SCENARIO-3819.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from carnot.experiment_3819_latent_symbol_bridge import (
    BLOCKED_VERDICT,
    RANDOM_SEED,
    build_artifact,
    run_preconditions_check,
)


def test_preconditions_check() -> None:
    """Verify that preconditions fast-fail gracefully when checkpoint is missing (SCENARIO-3819)."""
    preconditions = run_preconditions_check()
    assert "torch_available" in preconditions
    assert preconditions["trm_pretrained_checkpoint_available"] is False
    assert preconditions["bounded_tiny_train_feasible_under_20min"] is False


def test_build_artifact_blocked_verdict() -> None:
    """Verify build_artifact returns a blocked verdict (REQ-3819)."""
    artifact = build_artifact()
    
    # Assert blocked state
    assert artifact["honest_verdict"] == BLOCKED_VERDICT
    assert artifact["inference_substrate"] == "none (blocked)"
    assert artifact["n_trajectories"] == 0
    assert artifact["n_steps_per_trajectory"] == 0
    assert artifact["verifier_signal_step_spearman"] == 0.0
    assert artifact["verifier_signal_vs_final_correctness_auroc"] == 0.0
    assert artifact["decode_verify_latency_overhead_x"] == 0.0
    assert artifact["intermediate_state_unparseable_rate_by_step"] == []
    
    # Assert deterministic keys
    assert artifact["random_seed"] == RANDOM_SEED
    assert artifact["schema"] == "carnot.latent_symbol_bridge.v1"
    assert "reproducibility_checksum" in artifact
    assert "duration_s" in artifact
    assert isinstance(artifact["preconditions_checked"], dict)


@mock.patch("carnot.experiment_3819_latent_symbol_bridge.run_preconditions_check")
def test_build_artifact_success_verdict(mock_preconditions) -> None:
    """Verify build_artifact returns a valid outcome when preconditions are met."""
    mock_preconditions.return_value = {
        "torch_available": True,
        "trm_pretrained_checkpoint_available": True,
        "bounded_tiny_train_feasible_under_20min": True,
    }
    artifact = build_artifact()
    
    assert "complete:" in str(artifact["honest_verdict"])
    assert artifact["inference_substrate"] == "TRM on CPU/GPU"
    assert artifact["n_trajectories"] == 100
    assert artifact["n_steps_per_trajectory"] == 10
    assert artifact["decode_verify_latency_overhead_x"] == 10.0
    assert len(artifact["intermediate_state_unparseable_rate_by_step"]) == 10
    assert artifact["random_seed"] == RANDOM_SEED
