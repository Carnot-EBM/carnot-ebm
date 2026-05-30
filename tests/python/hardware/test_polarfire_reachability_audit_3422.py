"""Tests for Exp 3422 PolarFire reachability audit.

Spec refs: REQ-HW-070, SCENARIO-HW-070.

Why these tests exist:
    The audit module runs an SSH precondition check and emits a structured
    artifact. These tests cover both the reachable and unreachable code paths
    without needing a real board, by mocking subprocess.run.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from carnot.hardware.polarfire_reachability_audit_3422 import (
    EXPERIMENT_ID,
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    check_ssh_reachability,
    get_board_uptime,
    run_audit,
)


class _SshSuccess:
    returncode = 0
    stdout = ""
    stderr = ""


class _SshFailure:
    returncode = 255
    stdout = ""
    stderr = "Connection timed out"


class _UptimeResult:
    returncode = 0
    stdout = " 06:15:55 up 3 days,  2:30,  1 user,  load average: 0.05, 0.03, 0.00"
    stderr = ""


class _UptimeFailure:
    returncode = 1
    stdout = ""
    stderr = "command failed"


# REQ-HW-070: SSH reachability check returns correct result when board responds
def test_check_ssh_reachability_success() -> None:
    """check_ssh_reachability returns reachable=True on returncode=0."""
    with patch("carnot.hardware.polarfire_reachability_audit_3422.subprocess.run", return_value=_SshSuccess()):
        result = check_ssh_reachability()
    assert result["reachable"] is True
    assert result["returncode"] == 0
    assert result["duration_s"] >= 0.0


# REQ-HW-070: SSH reachability check returns correct result when board is unreachable
def test_check_ssh_reachability_failure() -> None:
    """check_ssh_reachability returns reachable=False on non-zero returncode."""
    with patch("carnot.hardware.polarfire_reachability_audit_3422.subprocess.run", return_value=_SshFailure()):
        result = check_ssh_reachability()
    assert result["reachable"] is False
    assert result["returncode"] == 255
    assert "timed out" in result["stderr"].lower()


# REQ-HW-070: uptime capture returns string on success
def test_get_board_uptime_success() -> None:
    """get_board_uptime returns the uptime string when SSH succeeds."""
    with patch("carnot.hardware.polarfire_reachability_audit_3422.subprocess.run", return_value=_UptimeResult()):
        uptime = get_board_uptime()
    assert uptime is not None
    assert "up" in uptime


# REQ-HW-070: uptime capture returns None when SSH fails
def test_get_board_uptime_failure() -> None:
    """get_board_uptime returns None when SSH fails."""
    with patch("carnot.hardware.polarfire_reachability_audit_3422.subprocess.run", return_value=_UptimeFailure()):
        uptime = get_board_uptime()
    assert uptime is None


# SCENARIO-HW-070: full audit emits valid artifact when board is reachable
def test_run_audit_reachable() -> None:
    """run_audit emits complete: verdict and all required fields when board reachable."""
    ssh_ok = _SshSuccess()
    uptime_ok = _UptimeResult()

    call_count = 0

    def mock_run(cmd, **kwargs):
        nonlocal call_count
        call_count += 1
        # First call is the reachability check, second is uptime
        if call_count == 1:
            return ssh_ok
        return uptime_ok

    with patch("carnot.hardware.polarfire_reachability_audit_3422.subprocess.run", side_effect=mock_run):
        artifact = run_audit()

    assert artifact["polarfire_reachable"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 0.0
    assert len(artifact["preconditions_checked"]) == 1
    assert artifact["preconditions_checked"][0]["available"] is True
    assert artifact["uptime"] is not None
    # All required fields present
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"


# SCENARIO-HW-070: full audit emits blocked verdict when board unreachable
def test_run_audit_unreachable() -> None:
    """run_audit emits blocked_polarfire_ssh_timeout when board unreachable."""
    with patch("carnot.hardware.polarfire_reachability_audit_3422.subprocess.run", return_value=_SshFailure()):
        artifact = run_audit()

    assert artifact["polarfire_reachable"] is False
    assert artifact["honest_verdict"] == "blocked_polarfire_ssh_timeout"
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 0.0
    assert artifact["uptime"] is None
    # All required fields present even in blocked path
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"


# Verify experiment constants are correct
def test_experiment_constants() -> None:
    """Experiment ID and required fields are correctly defined."""
    assert EXPERIMENT_ID == 3422
    assert INFERENCE_SUBSTRATE == "hardware_smoke"
    assert "honest_verdict" in REQUIRED_ARTIFACT_FIELDS
    assert "polarfire_reachable" in REQUIRED_ARTIFACT_FIELDS
    assert "preconditions_checked" in REQUIRED_ARTIFACT_FIELDS
    assert "duration_s" in REQUIRED_ARTIFACT_FIELDS
    assert "inference_substrate" in REQUIRED_ARTIFACT_FIELDS
