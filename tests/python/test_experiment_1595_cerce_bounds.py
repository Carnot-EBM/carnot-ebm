"""Tests for Exp 1595 CerCE pre/post bounds check.

REQ-LEARN-1595: Execute a pre/post bounds check on the CerCE ledger using local models.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

from carnot.training.cerce_bounds_check import (
    OUTPUT_FILE,
    SCHEMA,
    run_cerce_bounds_check,
)


def test_cerce_bounds_check_success(tmp_path):
    """SCENARIO-LEARN-1595-B: Valid Bounds Pass Check"""
    out_file = tmp_path / OUTPUT_FILE

    # Mocking simulated updates where bounds do NOT worsen
    simulated_updates = [
        {"update_id": "up_1", "bound_worsened": False},
        {"update_id": "up_2", "bound_worsened": False},
    ]

    with patch(
        "carnot.training.cerce_bounds_check.get_simulated_updates", return_value=simulated_updates
    ):
        run_cerce_bounds_check(out_dir=tmp_path)

    assert out_file.exists()
    with open(out_file) as f:
        data = json.load(f)

    assert data["status"] == "complete"
    assert data["schema"] == SCHEMA
    assert data["continuous_self_learning_task"] == "exp1595-cerce-bounds"
    assert data["bounds_check_passed"] is True
    assert data["simulated_updates_run"] == 2
    assert data["rejected_updates"] == []
    assert data["honest_verdict"] == "complete: cerce_bounds_checked"


def test_cerce_bounds_check_rejection(tmp_path):
    """SCENARIO-LEARN-1595-A: Worsened Bound Rejects Update"""
    out_file = tmp_path / OUTPUT_FILE

    # Mocking simulated updates where bound DOES worsen for one
    simulated_updates = [
        {"update_id": "up_1", "bound_worsened": False},
        {"update_id": "up_2", "bound_worsened": True},
    ]

    with patch(
        "carnot.training.cerce_bounds_check.get_simulated_updates", return_value=simulated_updates
    ):
        run_cerce_bounds_check(out_dir=tmp_path)

    assert out_file.exists()
    with open(out_file) as f:
        data = json.load(f)

    assert data["status"] == "complete"
    assert data["bounds_check_passed"] is False
    assert data["simulated_updates_run"] == 2
    assert data["rejected_updates"] == ["up_2"]
    assert data["honest_verdict"] == "complete: cerce_bounds_rejected"


def test_cerce_bounds_check_unmocked_success(tmp_path):
    """Ensure get_simulated_updates runs successfully by default."""
    run_cerce_bounds_check(out_dir=tmp_path)
    out_file = tmp_path / OUTPUT_FILE
    assert out_file.exists()
    with open(out_file) as f:
        data = json.load(f)
    assert data["status"] == "complete"
    assert data["bounds_check_passed"] is True


def test_cerce_bounds_check_default_dir(tmp_path, monkeypatch):
    """Cover the out_dir is None branch."""
    monkeypatch.chdir(tmp_path)
    run_cerce_bounds_check()
    artifact_root = os.environ.get("CARNOT_EXPERIMENT_ARTIFACT_ROOT")
    out_file = (
        tmp_path / "results" / OUTPUT_FILE
        if artifact_root is None
        else Path(artifact_root) / OUTPUT_FILE
    )
    assert out_file.exists()
