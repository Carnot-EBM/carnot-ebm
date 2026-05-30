"""Tests for Exp 3466 GateMate detect + toolchain continuity audit v3.

Spec refs: REQ-HW-106, SCENARIO-HW-106.

Why these tests exist:
    The audit module runs toolchain presence checks and a board-detect command,
    then emits a structured artifact. These tests cover all three outcome paths
    (toolchain missing, board absent, board detected) without real hardware,
    using monkeypatching of shutil.which and subprocess.run.

    A dedicated test verifies the TAUTOLOGY fix: the artifact must NOT contain
    both a top-level ``experiment`` and ``experiment_id`` field with identical
    numeric values (the failure mode of exp3443/v1, carried forward as a guard).

    All honest_verdict values must start with 'complete:' per CLAUDE.md
    Verdict Terminal-Prefix Discipline.
"""

from __future__ import annotations

from unittest.mock import patch

from carnot.hardware.gatemate_detect_3466 import (
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
# Fakes
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
    return f"/opt/oss-cad-suite/bin/{name}"


def _which_none(_name: str) -> None:
    return None


def _which_missing_himbaechel(name: str) -> str | None:
    if name == "nextpnr-himbaechel":
        return None
    return f"/opt/oss-cad-suite/bin/{name}"


# ---------------------------------------------------------------------------
# check_toolchain
# ---------------------------------------------------------------------------

# REQ-HW-106
def test_check_toolchain_all_present() -> None:
    """check_toolchain returns available=True for every tool when which succeeds."""
    with patch("carnot.hardware.gatemate_detect_3466.shutil.which", side_effect=_which_all):
        results = check_toolchain()
    assert len(results) == 3
    assert all(r["available"] for r in results)
    resources = {r["resource"] for r in results}
    assert "toolchain:yosys" in resources
    assert "toolchain:nextpnr-himbaechel" in resources
    assert "toolchain:openFPGALoader" in resources


# REQ-HW-106
def test_check_toolchain_all_missing() -> None:
    """check_toolchain returns available=False for every tool when which returns None."""
    with patch("carnot.hardware.gatemate_detect_3466.shutil.which", side_effect=_which_none):
        results = check_toolchain()
    assert all(not r["available"] for r in results)
    assert all(r["path_or_none"] is None for r in results)


# REQ-HW-106
def test_check_toolchain_partial_missing() -> None:
    """check_toolchain marks nextpnr-himbaechel absent when which returns None for it."""
    with patch("carnot.hardware.gatemate_detect_3466.shutil.which", side_effect=_which_missing_himbaechel):
        results = check_toolchain()
    by_resource = {r["resource"]: r for r in results}
    assert by_resource["toolchain:nextpnr-himbaechel"]["available"] is False
    assert by_resource["toolchain:yosys"]["available"] is True
    assert by_resource["toolchain:openFPGALoader"]["available"] is True


# ---------------------------------------------------------------------------
# detect_board
# ---------------------------------------------------------------------------

# REQ-HW-106
def test_detect_board_success() -> None:
    """detect_board returns board_detected=True and correct idcode on success."""
    with patch("carnot.hardware.gatemate_detect_3466.subprocess.run", return_value=_DetectSuccess()):
        result = detect_board()
    assert result["board_detected"] is True
    assert result["idcode"] == EXPECTED_IDCODE.lower()
    assert result["returncode"] == 0
    assert result["duration_s"] >= 0.0
    assert GATEMATE_MANUFACTURER in (result["manufacturer"] or "").lower()


# REQ-HW-106
def test_detect_board_absent() -> None:
    """detect_board returns board_detected=False when openFPGALoader finds no device."""
    with patch("carnot.hardware.gatemate_detect_3466.subprocess.run", return_value=_DetectNotFound()):
        result = detect_board()
    assert result["board_detected"] is False
    assert result["returncode"] == 1


# REQ-HW-106
def test_detect_board_tool_not_found() -> None:
    """detect_board returns board_detected=False when openFPGALoader raises FileNotFoundError."""
    with patch("carnot.hardware.gatemate_detect_3466.subprocess.run", side_effect=FileNotFoundError):
        result = detect_board()
    assert result["board_detected"] is False
    assert result["returncode"] == -1
    assert "not found" in result["raw_output"].lower()


# ---------------------------------------------------------------------------
# run_audit
# ---------------------------------------------------------------------------

# SCENARIO-HW-106
def test_run_audit_toolchain_missing() -> None:
    """run_audit emits blocked_gatemate_toolchain_missing when toolchain absent."""
    with patch("carnot.hardware.gatemate_detect_3466.shutil.which", side_effect=_which_none):
        artifact = run_audit()

    assert artifact["honest_verdict"] == "complete: blocked_gatemate_toolchain_missing"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gatemate_board_detected"] is False
    assert artifact["toolchain_present"] is False
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 0.0
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"


# SCENARIO-HW-106
def test_run_audit_board_absent() -> None:
    """run_audit emits blocked_gatemate_board_unreachable when board not detected."""
    with (
        patch("carnot.hardware.gatemate_detect_3466.shutil.which", side_effect=_which_all),
        patch("carnot.hardware.gatemate_detect_3466.subprocess.run", return_value=_DetectNotFound()),
    ):
        artifact = run_audit()

    assert artifact["honest_verdict"] == "complete: blocked_gatemate_board_unreachable"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gatemate_board_detected"] is False
    assert artifact["toolchain_present"] is True
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 0.0
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"


# SCENARIO-HW-106
def test_run_audit_board_detected() -> None:
    """run_audit emits 'complete: gatemate ...' when toolchain OK and board detected."""
    with (
        patch("carnot.hardware.gatemate_detect_3466.shutil.which", side_effect=_which_all),
        patch("carnot.hardware.gatemate_detect_3466.subprocess.run", return_value=_DetectSuccess()),
    ):
        artifact = run_audit()

    assert artifact["honest_verdict"].startswith("complete:")
    assert "blocked" not in artifact["honest_verdict"]
    assert artifact["gatemate_board_detected"] is True
    assert artifact["toolchain_present"] is True
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["duration_s"] >= 0.0
    assert "exp3454" in artifact["continuity_note"]
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"


# REQ-HW-106
def test_run_audit_preconditions_include_toolchain() -> None:
    """run_audit always includes one entry per toolchain binary in preconditions_checked."""
    with (
        patch("carnot.hardware.gatemate_detect_3466.shutil.which", side_effect=_which_all),
        patch("carnot.hardware.gatemate_detect_3466.subprocess.run", return_value=_DetectSuccess()),
    ):
        artifact = run_audit()

    resources = {p["resource"] for p in artifact["preconditions_checked"]}
    assert "toolchain:yosys" in resources
    assert "toolchain:nextpnr-himbaechel" in resources
    assert "toolchain:openFPGALoader" in resources
    assert "gatemate_board_detect" in resources


# REQ-HW-106: run_audit does NOT return experiment_id (avoids TAUTOLOGY with wrapper)
def test_run_audit_does_not_return_experiment_id() -> None:
    """run_audit must NOT include experiment_id to avoid TAUTOLOGY when wrapped."""
    with patch("carnot.hardware.gatemate_detect_3466.shutil.which", side_effect=_which_none):
        artifact = run_audit()
    # If the module returned experiment_id and the wrapper also added experiment_id
    # (or experiment), we'd get two identical numeric fields → TAUTOLOGY flag.
    assert "experiment_id" not in artifact
    assert "experiment" not in artifact


# REQ-HW-106: the final envelope (as the script would produce) must not
# contain both 'experiment' and 'experiment_id' as numeric top-level fields.
def test_envelope_has_single_numeric_identifier() -> None:
    """The script wrapper must produce at most one top-level numeric experiment ID."""
    with patch("carnot.hardware.gatemate_detect_3466.shutil.which", side_effect=_which_none):
        artifact = run_audit()

    # Simulate what the script does: add only experiment_id, not experiment.
    envelope = {"experiment_id": EXPERIMENT_ID, "title": "...", **artifact}

    numeric_fields = {k for k, v in envelope.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    id_fields = {f for f in numeric_fields if "experiment" in f.lower()}
    assert len(id_fields) <= 1, f"Duplicate experiment ID fields: {id_fields}"


# REQ-HW-106: module constants are correct
def test_experiment_constants() -> None:
    """Experiment ID and required fields are correctly defined."""
    assert EXPERIMENT_ID == 3466
    assert INFERENCE_SUBSTRATE == "hardware_smoke"
    assert "honest_verdict" in REQUIRED_ARTIFACT_FIELDS
    assert "gatemate_board_detected" in REQUIRED_ARTIFACT_FIELDS
    assert "preconditions_checked" in REQUIRED_ARTIFACT_FIELDS
    assert "duration_s" in REQUIRED_ARTIFACT_FIELDS
    assert "inference_substrate" in REQUIRED_ARTIFACT_FIELDS
    assert "toolchain_present" in REQUIRED_ARTIFACT_FIELDS
    assert EXPECTED_IDCODE == "0x20000001"
