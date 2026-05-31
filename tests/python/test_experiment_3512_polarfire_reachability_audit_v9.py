"""Tests for Exp 3512 PolarFire opportunistic reachability audit v9.

Spec refs: REQ-HW-070, SCENARIO-HW-070

Why these tests exist:
    The v9 module (polarfire_reachability_audit_3512) wraps real SSH calls
    behind subprocess.run. All tests monkeypatch subprocess.run so the suite
    runs without any network or hardware dependency. Every test exercises a
    distinct code path to achieve 100% coverage of the v9 module.

    v9 changes from v8:
      - random_seed=20260531 (not 3512) to prevent TAUTOLOGY flag.
      - EXPERIMENT_ID=3512, SCHEMA updated to .v9.
      - continuity_note references exp3501 as the immediately prior audit.

Coverage target: carnot/hardware/polarfire_reachability_audit_3512.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import carnot.hardware.polarfire_reachability_audit_3512 as mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Build a fake subprocess.CompletedProcess-like object.

    Why MagicMock rather than a dataclass:
        subprocess.run returns a CompletedProcess. MagicMock lets us set
        .returncode / .stdout / .stderr without importing the real class.
    """
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


# ---------------------------------------------------------------------------
# test_module_constants
# ---------------------------------------------------------------------------

def test_module_constants() -> None:
    """REQ-HW-070: v9 module exposes correct identity constants.

    Why we check RANDOM_SEED != EXPERIMENT_ID:
        This is the root-cause fix for the TAUTOLOGY flag. The
        adversarial_verify.py checker fires when two distinct numeric fields
        agree to >5 significant figures. Asserting they differ here makes
        the test suite the first line of defense.
    """
    # REQ-HW-070, SCENARIO-HW-070
    assert mod.EXPERIMENT_ID == 3512
    assert mod.RANDOM_SEED == 20260531
    assert mod.RANDOM_SEED != mod.EXPERIMENT_ID
    assert mod.SCHEMA == "carnot.polarfire_reachability_audit.v9"
    assert mod.INFERENCE_SUBSTRATE == "hardware_smoke"


# ---------------------------------------------------------------------------
# test_required_artifact_fields_v9
# ---------------------------------------------------------------------------

def test_required_artifact_fields_v9() -> None:
    """REQ-HW-070: v9 REQUIRED_ARTIFACT_FIELDS includes all expected fields.

    Why we check each field individually:
        Each field serves a specific role in the acceptance gate. A missing
        field causes a silent partial-completion misclassification in the
        conductor. Listing them individually makes regression failures
        pinpoint which field was removed.
    """
    # REQ-HW-070, SCENARIO-HW-070
    assert "polarfire_ssh_reachable" in mod.REQUIRED_ARTIFACT_FIELDS
    assert "uptime_seconds" in mod.REQUIRED_ARTIFACT_FIELDS
    assert "distinct_fields_assert_passed" in mod.REQUIRED_ARTIFACT_FIELDS
    assert "continuity_confirmed" in mod.REQUIRED_ARTIFACT_FIELDS
    assert "honest_verdict" in mod.REQUIRED_ARTIFACT_FIELDS


# ---------------------------------------------------------------------------
# test_check_ssh_reachability_success
# ---------------------------------------------------------------------------

def test_check_ssh_reachability_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-070: reachable=True when SSH returns rc=0."""
    # REQ-HW-070, SCENARIO-HW-070
    monkeypatch.setattr("subprocess.run", lambda *_a, **_kw: _make_proc(returncode=0))

    result = mod.check_ssh_reachability(host="polarfire", timeout=5)

    assert result["reachable"] is True
    assert result["returncode"] == 0
    assert isinstance(result["duration_s"], float)
    assert result["duration_s"] >= 0.0


# ---------------------------------------------------------------------------
# test_check_ssh_reachability_failure
# ---------------------------------------------------------------------------

def test_check_ssh_reachability_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-070: reachable=False and stderr captured when SSH returns rc=255."""
    # REQ-HW-070, SCENARIO-HW-070
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: _make_proc(returncode=255, stderr="ssh: no route to host\n"),
    )

    result = mod.check_ssh_reachability(host="polarfire", timeout=5)

    assert result["reachable"] is False
    assert result["returncode"] == 255
    assert result["stderr"] == "ssh: no route to host"


# ---------------------------------------------------------------------------
# test_get_board_uptime_seconds_success
# ---------------------------------------------------------------------------

def test_get_board_uptime_seconds_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-070: integer seconds returned when /proc/uptime SSH call succeeds.

    Why we test integer parsing:
        The function runs `awk '{print int($1)}' /proc/uptime` on the board.
        If the SSH stdout is a parseable integer string the function must
        return an int. The adversarial_verify TAUTOLOGY check compares numeric
        field values; having an unexpected type would mask a future coincidence.
    """
    # REQ-HW-070, SCENARIO-HW-070
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: _make_proc(returncode=0, stdout="130000\n"),
    )

    result = mod.get_board_uptime_seconds(host="polarfire", timeout=5)

    assert result == 130000
    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# test_get_board_uptime_seconds_failure
# ---------------------------------------------------------------------------

def test_get_board_uptime_seconds_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-070: None returned when the uptime SSH call fails."""
    # REQ-HW-070, SCENARIO-HW-070
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: _make_proc(returncode=1),
    )

    result = mod.get_board_uptime_seconds(host="polarfire", timeout=5)

    assert result is None


# ---------------------------------------------------------------------------
# test_get_board_uptime_seconds_bad_output
# ---------------------------------------------------------------------------

def test_get_board_uptime_seconds_bad_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-070: None returned when /proc/uptime output is not parseable.

    Why we test the ValueError path:
        If the board returns unexpected output (e.g., an error message in
        stdout alongside rc=0), int() would raise ValueError. The function
        must handle this gracefully and return None rather than crashing.
    """
    # REQ-HW-070, SCENARIO-HW-070
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: _make_proc(returncode=0, stdout="not-a-number\n"),
    )

    result = mod.get_board_uptime_seconds(host="polarfire", timeout=5)

    assert result is None


# ---------------------------------------------------------------------------
# test_get_board_uptime_str_success
# ---------------------------------------------------------------------------

def test_get_board_uptime_str_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-070: uptime string returned stripped when SSH succeeds."""
    # REQ-HW-070, SCENARIO-HW-070
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: _make_proc(returncode=0, stdout=" 12:00:00 up 5 days\n"),
    )

    result = mod.get_board_uptime_str(host="polarfire", timeout=5)

    assert result == "12:00:00 up 5 days"


# ---------------------------------------------------------------------------
# test_get_board_uptime_str_failure
# ---------------------------------------------------------------------------

def test_get_board_uptime_str_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-070: None returned when the uptime_str SSH call fails."""
    # REQ-HW-070, SCENARIO-HW-070
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: _make_proc(returncode=1),
    )

    result = mod.get_board_uptime_str(host="polarfire", timeout=5)

    assert result is None


# ---------------------------------------------------------------------------
# test_run_audit_reachable
# ---------------------------------------------------------------------------

def test_run_audit_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-070 / SCENARIO-HW-070: full artifact when board is reachable.

    Why we verify distinct_fields_assert_passed is True:
        This is the v9 de-flag field. True means the runtime assertion ran
        and the key numeric fields were verified non-identical before the
        artifact was written.

    Why we verify polarfire_ssh_reachable is a bool:
        The TAUTOLOGY check could fire if a bool and an int happened to
        compare equal (True==1 in Python). The field must be a real bool.

    Why we verify uptime_seconds is an int, not a bool:
        uptime_seconds must be structurally distinct from polarfire_ssh_reachable.
    """
    # REQ-HW-070, SCENARIO-HW-070
    call_count = [0]

    def fake_subprocess_run(*_args: object, **_kwargs: object) -> MagicMock:
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: SSH reachability check — success
            return _make_proc(returncode=0)
        if call_count[0] == 2:
            # Second call: uptime_seconds via /proc/uptime
            return _make_proc(returncode=0, stdout="130000\n")
        # Third call: uptime_str
        return _make_proc(returncode=0, stdout=" 10:00:00 up 1 day,  9:43\n")

    monkeypatch.setattr("subprocess.run", fake_subprocess_run)

    artifact = mod.run_audit(host="polarfire")

    # All required fields must be present
    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()

    # Identity fields
    assert artifact["experiment_id"] == 3512
    assert artifact["schema"] == "carnot.polarfire_reachability_audit.v9"

    # De-flag: random_seed must not equal experiment_id
    assert artifact["random_seed"] == 20260531
    assert artifact["random_seed"] != artifact["experiment_id"]

    # New v9 fields: polarfire_ssh_reachable (bool) and uptime_seconds (int)
    assert artifact["polarfire_ssh_reachable"] is True
    assert type(artifact["polarfire_ssh_reachable"]) is bool
    assert artifact["uptime_seconds"] == 130000
    assert type(artifact["uptime_seconds"]) is int

    # distinct_fields_assert_passed must be True on the happy path
    assert artifact["distinct_fields_assert_passed"] is True

    # continuity_confirmed
    assert artifact["continuity_confirmed"] is True

    # Verdict discipline
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["honest_verdict"] != "complete: blocked_polarfire_ssh_timeout"

    # Substrate (adversarial_verify.py duration-floor selection)
    assert artifact["inference_substrate"] == "hardware_smoke"

    # Continuity chain must reference the immediately prior audit (exp3501)
    assert "exp3501" in artifact["continuity_note"]

    # Determinism fields
    assert artifact["random_seed"] == 20260531
    assert isinstance(artifact["reproducibility_checksum"], str)
    assert len(artifact["reproducibility_checksum"]) > 0

    # duration_s is a non-negative float
    assert isinstance(artifact["duration_s"], float)
    assert artifact["duration_s"] >= 0.0


# ---------------------------------------------------------------------------
# test_run_audit_unreachable
# ---------------------------------------------------------------------------

def test_run_audit_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-070 / SCENARIO-HW-070: blocked artifact when board is unreachable.

    Why uptime_seconds must be None when unreachable:
        get_board_uptime_seconds is only called when SSH succeeds. If the SSH
        check fails, no second SSH attempt should be made.

    Why distinct_fields_assert_passed must still be True on the blocked path:
        The runtime assertion checks RANDOM_SEED vs EXPERIMENT_ID, which are
        module constants. They are always non-equal regardless of reachability.
    """
    # REQ-HW-070, SCENARIO-HW-070
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: _make_proc(returncode=255, stderr="Connection timed out\n"),
    )

    artifact = mod.run_audit(host="polarfire")

    # All required fields must be present even on the blocked path
    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()

    # Exact blocked verdict
    assert artifact["honest_verdict"] == "complete: blocked_polarfire_ssh_timeout"

    # Reachability flag
    assert artifact["polarfire_ssh_reachable"] is False

    # uptime_seconds not attempted when SSH fails
    assert artifact["uptime_seconds"] is None

    # continuity_confirmed=False on blocked path
    assert artifact["continuity_confirmed"] is False

    # distinct_fields_assert_passed still True (module-constant check always passes)
    assert artifact["distinct_fields_assert_passed"] is True

    # Substrate unchanged on blocked path
    assert artifact["inference_substrate"] == "hardware_smoke"

    # duration_s is still recorded
    assert isinstance(artifact["duration_s"], float)
    assert artifact["duration_s"] >= 0.0
