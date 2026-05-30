"""Tests for Exp 3443 GateMate detect + toolchain continuity audit.

Spec refs: REQ-HW-106, SCENARIO-HW-106.

Why these tests exist:
    The audit module runs toolchain presence checks and a board-detect command,
    then emits a structured artifact. These tests cover all three outcome paths
    (toolchain missing, board absent, board detected) without real hardware,
    using monkeypatching of shutil.which and subprocess.run.

    All honest_verdict values must start with 'complete:' per CLAUDE.md
    Verdict Terminal-Prefix Discipline.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from carnot.hardware.gatemate_detect_3443 import (
    EXPERIMENT_ID,
    EXPECTED_IDCODE,
    GATEMATE_MANUFACTURER,
    INFERENCE_SUBSTRATE,
    REQUIRED_ARTIFACT_FIELDS,
    check_toolchain,
    detect_board,
    run_audit,
)


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class _DetectSuccess:
    """Simulates openFPGALoader --detect finding the GateMate board."""
    returncode = 0
    stdout = (
        "Jtag frequency : requested 6000000 Hz -> real 6000000 Hz\n"
        "index 0:\n"
        "\tidcode 0x20000001\n"
        "\tmanufacturer colognechip\n"
        "\tfamily GateMate Series\n"
        "\tmodel  GM1Ax\n"
        "\tirlength 6\n"
    )
    stderr = ""


class _DetectNotFound:
    """openFPGALoader ran but found no JTAG device (board absent)."""
    returncode = 1
    stdout = ""
    stderr = "No JTAG device found"


def _which_all(name: str) -> str:
    """Fake shutil.which that returns a path for every tool."""
    return f"/opt/oss-cad-suite/bin/{name}"


def _which_none(_name: str) -> None:
    """Fake shutil.which that returns None for every tool (toolchain absent)."""
    return None


def _which_missing_himbaechel(name: str) -> str | None:
    """Fake shutil.which where nextpnr-himbaechel is absent."""
    if name == "nextpnr-himbaechel":
        return None
    return f"/opt/oss-cad-suite/bin/{name}"


# ---------------------------------------------------------------------------
# check_toolchain tests
# ---------------------------------------------------------------------------

# REQ-HW-106: toolchain check reports all tools present when which returns paths
def test_check_toolchain_all_present() -> None:
    """check_toolchain returns available=True for all tools when which succeeds."""
    with patch("carnot.hardware.gatemate_detect_3443.shutil.which", side_effect=_which_all):
        results = check_toolchain()
    assert len(results) == 3
    assert all(r["available"] for r in results)
    resources = [r["resource"] for r in results]
    assert "toolchain:yosys" in resources
    assert "toolchain:nextpnr-himbaechel" in resources
    assert "toolchain:openFPGALoader" in resources


# REQ-HW-106: toolchain check reports all tools missing when which returns None
def test_check_toolchain_all_missing() -> None:
    """check_toolchain returns available=False for all tools when which returns None."""
    with patch("carnot.hardware.gatemate_detect_3443.shutil.which", side_effect=_which_none):
        results = check_toolchain()
    assert all(not r["available"] for r in results)
    assert all(r["path_or_none"] is None for r in results)


# REQ-HW-106: toolchain check correctly reflects partial absence
def test_check_toolchain_partial_missing() -> None:
    """check_toolchain marks nextpnr-himbaechel absent when which returns None for it."""
    with patch("carnot.hardware.gatemate_detect_3443.shutil.which", side_effect=_which_missing_himbaechel):
        results = check_toolchain()
    by_resource = {r["resource"]: r for r in results}
    assert by_resource["toolchain:nextpnr-himbaechel"]["available"] is False
    assert by_resource["toolchain:yosys"]["available"] is True
    assert by_resource["toolchain:openFPGALoader"]["available"] is True


# ---------------------------------------------------------------------------
# detect_board tests
# ---------------------------------------------------------------------------

# REQ-HW-106: detect returns board_detected=True when idcode matches
def test_detect_board_success() -> None:
    """detect_board returns board_detected=True and correct idcode on success."""
    with patch("carnot.hardware.gatemate_detect_3443.subprocess.run", return_value=_DetectSuccess()):
        result = detect_board()
    assert result["board_detected"] is True
    assert result["idcode"] == EXPECTED_IDCODE.lower()
    assert result["returncode"] == 0
    assert result["duration_s"] >= 0.0
    assert GATEMATE_MANUFACTURER in (result["manufacturer"] or "").lower()


# REQ-HW-106: detect returns board_detected=False when JTAG finds no device
def test_detect_board_absent() -> None:
    """detect_board returns board_detected=False when openFPGALoader finds no device."""
    with patch("carnot.hardware.gatemate_detect_3443.subprocess.run", return_value=_DetectNotFound()):
        result = detect_board()
    assert result["board_detected"] is False
    assert result["returncode"] == 1


# REQ-HW-106: detect returns board_detected=False when openFPGALoader not on PATH
def test_detect_board_tool_not_found() -> None:
    """detect_board returns board_detected=False when openFPGALoader raises FileNotFoundError."""
    with patch("carnot.hardware.gatemate_detect_3443.subprocess.run", side_effect=FileNotFoundError):
        result = detect_board()
    assert result["board_detected"] is False
    assert result["returncode"] == -1
    assert "not found" in result["raw_output"].lower()


# ---------------------------------------------------------------------------
# run_audit tests
# ---------------------------------------------------------------------------

# SCENARIO-HW-106: audit emits 'complete: blocked_gatemate_toolchain_missing'
def test_run_audit_toolchain_missing() -> None:
    """run_audit emits blocked_gatemate_toolchain_missing when toolchain absent."""
    with patch("carnot.hardware.gatemate_detect_3443.shutil.which", side_effect=_which_none):
        artifact = run_audit()

    assert artifact["honest_verdict"] == "complete: blocked_gatemate_toolchain_missing"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gatemate_board_detected"] is False
    assert artifact["toolchain_ok"] is False
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 0.0
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"


# SCENARIO-HW-106: audit emits 'complete: blocked_gatemate_board_unreachable'
def test_run_audit_board_absent() -> None:
    """run_audit emits blocked_gatemate_board_unreachable when board not detected."""
    with (
        patch("carnot.hardware.gatemate_detect_3443.shutil.which", side_effect=_which_all),
        patch("carnot.hardware.gatemate_detect_3443.subprocess.run", return_value=_DetectNotFound()),
    ):
        artifact = run_audit()

    assert artifact["honest_verdict"] == "complete: blocked_gatemate_board_unreachable"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gatemate_board_detected"] is False
    assert artifact["toolchain_ok"] is True
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 0.0
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"


# SCENARIO-HW-106: audit emits success verdict when toolchain present and board detected
def test_run_audit_board_detected() -> None:
    """run_audit emits 'complete: gatemate ...' when toolchain OK and board detected."""
    with (
        patch("carnot.hardware.gatemate_detect_3443.shutil.which", side_effect=_which_all),
        patch("carnot.hardware.gatemate_detect_3443.subprocess.run", return_value=_DetectSuccess()),
    ):
        artifact = run_audit()

    assert artifact["honest_verdict"].startswith("complete:")
    assert "blocked" not in artifact["honest_verdict"]
    assert artifact["gatemate_board_detected"] is True
    assert artifact["toolchain_ok"] is True
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 0.0
    # Continuity note should reference the prior experiment
    assert "3432" in artifact["continuity_note"]
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"


# REQ-HW-106: preconditions_checked includes all toolchain checks
def test_run_audit_preconditions_include_toolchain() -> None:
    """run_audit always includes one entry per toolchain binary in preconditions_checked."""
    with (
        patch("carnot.hardware.gatemate_detect_3443.shutil.which", side_effect=_which_all),
        patch("carnot.hardware.gatemate_detect_3443.subprocess.run", return_value=_DetectSuccess()),
    ):
        artifact = run_audit()

    resources = [p["resource"] for p in artifact["preconditions_checked"]]
    assert "toolchain:yosys" in resources
    assert "toolchain:nextpnr-himbaechel" in resources
    assert "toolchain:openFPGALoader" in resources
    assert "gatemate_board_detect" in resources


# Verify module constants
def test_experiment_constants() -> None:
    """Experiment ID and required fields are correctly defined."""
    assert EXPERIMENT_ID == 3443
    assert INFERENCE_SUBSTRATE == "hardware_smoke"
    assert "honest_verdict" in REQUIRED_ARTIFACT_FIELDS
    assert "gatemate_board_detected" in REQUIRED_ARTIFACT_FIELDS
    assert "preconditions_checked" in REQUIRED_ARTIFACT_FIELDS
    assert "duration_s" in REQUIRED_ARTIFACT_FIELDS
    assert "inference_substrate" in REQUIRED_ARTIFACT_FIELDS
    assert EXPECTED_IDCODE == "0x20000001"
