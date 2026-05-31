"""Tests for Exp 3558 PolarFire reachability audit v13.

Spec refs: REQ-HW-070, SCENARIO-HW-070.

Why these tests exist:
    The audit module runs an SSH precondition check and emits a structured
    artifact. These tests cover both the reachable and unreachable code paths
    without needing a real board, by mocking subprocess.run.

    v13 change: experiment_id is 3558, random_seed is 20260601.
"""

from __future__ import annotations

from unittest.mock import patch

from carnot.hardware.polarfire_reachability_audit_3558 import (
    EXPERIMENT_ID,
    INFERENCE_SUBSTRATE,
    RANDOM_SEED,
    REQUIRED_ARTIFACT_FIELDS,
    check_ssh_reachability,
    get_board_uptime_seconds,
    get_board_uptime_str,
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


class _UptimeSecondsResult:
    returncode = 0
    stdout = "345600"
    stderr = ""


class _UptimeSecondsFailure:
    returncode = 1
    stdout = ""
    stderr = "command failed"


class _UptimeSecondsBadValue:
    returncode = 0
    stdout = "not_an_int"
    stderr = ""


class _UptimeStrResult:
    returncode = 0
    stdout = " 09:42:11 up 4 days,  0:00,  0 users,  load average: 0.01, 0.00, 0.00"
    stderr = ""


class _UptimeStrFailure:
    returncode = 1
    stdout = ""
    stderr = "command failed"


# REQ-HW-070: SSH reachability check returns correct result when board responds
def test_check_ssh_reachability_success() -> None:
    """check_ssh_reachability returns reachable=True on returncode=0."""
    with patch("carnot.hardware.polarfire_reachability_audit_3558.subprocess.run", return_value=_SshSuccess()):
        result = check_ssh_reachability()
    assert result["reachable"] is True
    assert result["returncode"] == 0
    assert result["duration_s"] >= 0.0


# REQ-HW-070: SSH reachability check returns correct result when board is unreachable
def test_check_ssh_reachability_failure() -> None:
    """check_ssh_reachability returns reachable=False on non-zero returncode."""
    with patch("carnot.hardware.polarfire_reachability_audit_3558.subprocess.run", return_value=_SshFailure()):
        result = check_ssh_reachability()
    assert result["reachable"] is False
    assert result["returncode"] == 255
    assert "timed out" in result["stderr"].lower()


# REQ-HW-070: uptime capture returns int on success
def test_get_board_uptime_seconds_success() -> None:
    """get_board_uptime_seconds returns the uptime int when SSH succeeds."""
    with patch("carnot.hardware.polarfire_reachability_audit_3558.subprocess.run", return_value=_UptimeSecondsResult()):
        uptime = get_board_uptime_seconds()
    assert uptime == 345600


# REQ-HW-070: uptime capture returns None when SSH fails
def test_get_board_uptime_seconds_failure() -> None:
    """get_board_uptime_seconds returns None when SSH fails."""
    with patch("carnot.hardware.polarfire_reachability_audit_3558.subprocess.run", return_value=_UptimeSecondsFailure()):
        uptime = get_board_uptime_seconds()
    assert uptime is None


def test_get_board_uptime_seconds_bad_value() -> None:
    """get_board_uptime_seconds returns None when stdout is not an int."""
    with patch("carnot.hardware.polarfire_reachability_audit_3558.subprocess.run", return_value=_UptimeSecondsBadValue()):
        uptime = get_board_uptime_seconds()
    assert uptime is None


def test_get_board_uptime_str_success() -> None:
    """get_board_uptime_str returns the uptime string when SSH succeeds."""
    with patch("carnot.hardware.polarfire_reachability_audit_3558.subprocess.run", return_value=_UptimeStrResult()):
        uptime = get_board_uptime_str()
    assert uptime is not None
    assert "up" in uptime


def test_get_board_uptime_str_failure() -> None:
    """get_board_uptime_str returns None when SSH fails."""
    with patch("carnot.hardware.polarfire_reachability_audit_3558.subprocess.run", return_value=_UptimeStrFailure()):
        uptime = get_board_uptime_str()
    assert uptime is None


# SCENARIO-HW-070: full audit emits valid artifact when board is reachable
def test_run_audit_reachable() -> None:
    """run_audit emits complete: verdict and all required fields when board reachable."""
    call_count = 0

    def mock_run(cmd, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _SshSuccess()
        elif call_count == 2:
            return _UptimeSecondsResult()
        return _UptimeStrResult()

    with patch("carnot.hardware.polarfire_reachability_audit_3558.subprocess.run", side_effect=mock_run):
        artifact = run_audit()

    assert artifact["polarfire_ssh_reachable"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 0.0
    assert len(artifact["preconditions_checked"]) == 1
    assert artifact["preconditions_checked"][0]["available"] is True
    assert artifact["uptime_seconds"] == 345600
    assert artifact["uptime_str"] is not None
    assert artifact["distinct_fields_assert_passed"] is True
    assert artifact["random_seed"] == RANDOM_SEED
    assert artifact["experiment_id"] == EXPERIMENT_ID
    # v13: continuity_note must reference exp3536
    assert "3536" in artifact["continuity_note"]
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"


# SCENARIO-HW-070: full audit emits complete: blocked verdict when board unreachable
def test_run_audit_unreachable() -> None:
    """run_audit emits 'complete: blocked_polarfire_ssh_timeout' when board unreachable."""
    with patch("carnot.hardware.polarfire_reachability_audit_3558.subprocess.run", return_value=_SshFailure()):
        artifact = run_audit()

    assert artifact["polarfire_ssh_reachable"] is False
    assert artifact["honest_verdict"] == "complete: blocked_polarfire_ssh_timeout"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 0.0
    assert artifact["uptime_seconds"] is None
    assert artifact["uptime_str"] is None
    assert artifact["distinct_fields_assert_passed"] is True
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"


# Verify experiment constants are correct for v13
def test_experiment_constants() -> None:
    """Experiment ID and required fields are correctly defined for v13."""
    assert EXPERIMENT_ID == 3558
    assert RANDOM_SEED == 20260601
    assert INFERENCE_SUBSTRATE == "hardware_smoke"
    assert "honest_verdict" in REQUIRED_ARTIFACT_FIELDS
    assert "polarfire_ssh_reachable" in REQUIRED_ARTIFACT_FIELDS
    assert "uptime_seconds" in REQUIRED_ARTIFACT_FIELDS
    assert "continuity_confirmed" in REQUIRED_ARTIFACT_FIELDS
    assert "distinct_fields_assert_passed" in REQUIRED_ARTIFACT_FIELDS
    assert "preconditions_checked" in REQUIRED_ARTIFACT_FIELDS
    assert "duration_s" in REQUIRED_ARTIFACT_FIELDS
    assert "inference_substrate" in REQUIRED_ARTIFACT_FIELDS
    assert "random_seed" in REQUIRED_ARTIFACT_FIELDS
    assert "reproducibility_checksum" in REQUIRED_ARTIFACT_FIELDS


# REQ-HW-070: custom host parameter is forwarded correctly
def test_check_ssh_reachability_custom_host() -> None:
    """check_ssh_reachability accepts and uses a custom host name."""
    captured_cmd: list[str] = []

    def mock_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return _SshSuccess()

    with patch("carnot.hardware.polarfire_reachability_audit_3558.subprocess.run", side_effect=mock_run):
        check_ssh_reachability(host="my-custom-host")

    assert "my-custom-host" in captured_cmd


# REQ-HW-070: run_audit forwards custom host to all SSH calls
def test_run_audit_custom_host() -> None:
    """run_audit passes the custom host to all SSH calls."""
    captured_cmds: list[list[str]] = []

    def mock_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        if len(captured_cmds) == 1:
            return _SshSuccess()
        elif len(captured_cmds) == 2:
            return _UptimeSecondsResult()
        return _UptimeStrResult()

    with patch("carnot.hardware.polarfire_reachability_audit_3558.subprocess.run", side_effect=mock_run):
        run_audit(host="test-board")

    assert all("test-board" in cmd for cmd in captured_cmds)
