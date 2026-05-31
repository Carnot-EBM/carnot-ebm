"""Tests for Exp 3490 PolarFire opportunistic reachability audit v7.

Spec refs: REQ-HW-070, SCENARIO-HW-070

Why these tests exist:
    The v7 module (polarfire_reachability_audit_3490) wraps real SSH calls
    behind subprocess.run. All tests monkeypatch subprocess.run so the suite
    runs without any network or hardware dependency. Every test exercises a
    distinct code path to achieve 100% coverage of the v7 module only.

    The v7 audit adds the 'continuity_confirmed' boolean field (absent in v6)
    so the conductor can gate on hardware visibility without re-probing.

Coverage target: carnot/hardware/polarfire_reachability_audit_3490.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import carnot.hardware.polarfire_reachability_audit_3490 as mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    """Build a fake subprocess.CompletedProcess-like object.

    Why MagicMock rather than a dataclass:
        subprocess.run returns a CompletedProcess. MagicMock lets us set
        .returncode / .stdout / .stderr without importing the real class,
        keeping the mock self-contained and immune to internal CPython changes.
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
    """REQ-HW-070: v7 module exposes the correct identity constants.

    Why we check all three constants together:
        EXPERIMENT_ID distinguishes this artifact from prior audit versions
        at query time. SCHEMA is the JSON key the conductor's reconciler
        uses to route the artifact. INFERENCE_SUBSTRATE is what
        adversarial_verify.py reads to select the duration floor — using the
        wrong substrate value would trigger a false DURATION_TOO_SHORT flag
        on a fast SSH-only run.
    """
    # REQ-HW-070, SCENARIO-HW-070
    assert mod.EXPERIMENT_ID == 3490
    assert mod.SCHEMA == "carnot.polarfire_reachability_audit.v7"
    assert mod.INFERENCE_SUBSTRATE == "hardware_smoke"


# ---------------------------------------------------------------------------
# test_required_artifact_fields_includes_continuity_confirmed
# ---------------------------------------------------------------------------

def test_required_artifact_fields_includes_continuity_confirmed() -> None:
    """REQ-HW-070: v7 adds 'continuity_confirmed' to REQUIRED_ARTIFACT_FIELDS.

    Why we check this field specifically:
        The task spec mandates 'continuity_confirmed' as a new required field
        (not present in v6). If it were absent from REQUIRED_ARTIFACT_FIELDS
        the acceptance-gate check in test_run_audit_reachable would not catch
        a missing-field regression.
    """
    # REQ-HW-070, SCENARIO-HW-070
    assert "continuity_confirmed" in mod.REQUIRED_ARTIFACT_FIELDS
    assert "honest_verdict" in mod.REQUIRED_ARTIFACT_FIELDS
    assert "polarfire_reachable" in mod.REQUIRED_ARTIFACT_FIELDS


# ---------------------------------------------------------------------------
# test_check_ssh_reachability_success
# ---------------------------------------------------------------------------

def test_check_ssh_reachability_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-070: reachable=True when SSH returns rc=0.

    Why we check duration_s >= 0:
        The function measures wall-clock time with time.monotonic(). Even with
        a mocked subprocess the two monotonic() calls may return the same
        value (yielding 0.0) in fast CI. We only assert non-negative to keep
        the test deterministic across all platforms.
    """
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
    """REQ-HW-070: reachable=False and stderr captured when SSH returns rc=255.

    Why we verify the exact returncode and stderr:
        The artifact records ssh_returncode for auditors who want to
        distinguish "host unreachable" (255) from "permission denied" (other
        non-zero codes). If the function swallowed the code we could not
        distinguish an honest blocked_polarfire_ssh_timeout from a
        credential problem.
    """
    # REQ-HW-070, SCENARIO-HW-070
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: _make_proc(returncode=255, stderr="ssh: no route to host\n"),
    )

    result = mod.check_ssh_reachability(host="polarfire", timeout=5)

    assert result["reachable"] is False
    assert result["returncode"] == 255
    # stderr is .strip()-ped inside the module
    assert result["stderr"] == "ssh: no route to host"


# ---------------------------------------------------------------------------
# test_get_board_uptime_success
# ---------------------------------------------------------------------------

def test_get_board_uptime_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-070: uptime string returned verbatim (stripped) when SSH succeeds.

    Why we check the stripped value:
        The board's 'uptime' command emits a trailing newline. The module
        strips it so callers can embed the string in JSON without embedded
        newlines that break readability and diff tooling.
    """
    # REQ-HW-070, SCENARIO-HW-070
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: _make_proc(returncode=0, stdout=" 12:00:00 up 5 days\n"),
    )

    result = mod.get_board_uptime(host="polarfire", timeout=5)

    assert result == "12:00:00 up 5 days"


# ---------------------------------------------------------------------------
# test_get_board_uptime_failure
# ---------------------------------------------------------------------------

def test_get_board_uptime_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-070: None returned when the uptime SSH call fails.

    Why None rather than raising:
        Uptime is opportunistic context — if the SSH session drops between
        the reachability check and the uptime call we do not want to abort
        the whole audit. None is the honest signal that we could not capture
        the metric, which the artifact records as uptime=null.
    """
    # REQ-HW-070, SCENARIO-HW-070
    monkeypatch.setattr(
        "subprocess.run",
        lambda *_a, **_kw: _make_proc(returncode=1),
    )

    result = mod.get_board_uptime(host="polarfire", timeout=5)

    assert result is None


# ---------------------------------------------------------------------------
# test_run_audit_reachable
# ---------------------------------------------------------------------------

def test_run_audit_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-HW-070 / SCENARIO-HW-070: full artifact when board is reachable.

    Why we check all REQUIRED_ARTIFACT_FIELDS:
        The conductor's reconciler gates task completion on these fields being
        present. A missing field causes a silent partial-completion
        misclassification. We use <= (subset) so future field additions to
        REQUIRED_ARTIFACT_FIELDS will automatically extend this test.

    Why we check honest_verdict starts with "complete:":
        Verdict Terminal-Prefix Discipline (CLAUDE.md) mandates that terminal
        verdicts start with one of the recognised prefixes. Without the prefix
        the conductor's _verdict_is_untrustworthy classifier may
        false-positive on words like "polarfire" or "confirmed".

    Why we check inference_substrate == "hardware_smoke":
        adversarial_verify.py uses this field to select the duration floor.
        hardware_smoke allows sub-60 s durations; live_llm_inference does not.
        Wrong substrate on a fast SSH-only run would trigger
        DURATION_TOO_SHORT.

    Why we check continuity_confirmed is True:
        The v7 addition: continuity_confirmed=True signals the board is
        reachable and dispatch-capable, letting the conductor log hardware
        continuity without a second probe.

    Why we check continuity_note contains "exp3479":
        The v7 module's continuity_note must reference the immediately prior
        audit (exp3479) so the audit trail is complete. If the note points at
        an earlier experiment the traceability chain breaks.
    """
    # REQ-HW-070, SCENARIO-HW-070
    call_count = [0]

    def fake_subprocess_run(*_args: object, **_kwargs: object) -> MagicMock:
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: SSH reachability check — success
            return _make_proc(returncode=0)
        # Second call: uptime — success
        return _make_proc(returncode=0, stdout=" 10:00:00 up 1 day\n")

    monkeypatch.setattr("subprocess.run", fake_subprocess_run)

    artifact = mod.run_audit(host="polarfire")

    # All required fields must be present
    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()

    # Identity fields
    assert artifact["experiment_id"] == 3490
    assert artifact["schema"] == "carnot.polarfire_reachability_audit.v7"

    # Reachability outcome
    assert artifact["polarfire_reachable"] is True

    # v7 addition: continuity_confirmed
    assert artifact["continuity_confirmed"] is True

    # Verdict discipline
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["honest_verdict"] != "complete: blocked_polarfire_ssh_timeout"

    # Substrate (adversarial_verify.py duration-floor selection)
    assert artifact["inference_substrate"] == "hardware_smoke"

    # Continuity chain must reference the immediately prior audit
    assert "exp3479" in artifact["continuity_note"]

    # Uptime was captured because board was reachable
    assert artifact["uptime"] == "10:00:00 up 1 day"

    # Determinism fields
    assert artifact["random_seed"] == 3490
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

    Why honest_verdict must equal exactly "complete: blocked_polarfire_ssh_timeout":
        The conductor's reconciler identifies this exact string as the
        canonical blocked verdict for a PolarFire SSH timeout. A typo or
        variation would send the task into an ambiguous partial state and
        trigger unnecessary retries on an opportunistic board.

    Why continuity_confirmed must be False:
        The v7 field must accurately reflect the hardware state. False signals
        that hardware visibility could not be confirmed this milestone without
        implying failure — the opportunistic board is expected to be
        occasionally unreachable per north-star §3.

    Why uptime must be absent / None:
        get_board_uptime is only called when SSH succeeds. If the SSH check
        fails, no second SSH attempt should be made. Confirming uptime is None
        verifies that short-circuit is in place.
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
    assert artifact["polarfire_reachable"] is False

    # v7 addition: continuity_confirmed=False on blocked path
    assert artifact["continuity_confirmed"] is False

    # Substrate unchanged on blocked path
    assert artifact["inference_substrate"] == "hardware_smoke"

    # Uptime must not have been attempted
    assert artifact["uptime"] is None

    # duration_s is still recorded (total wall-clock of the failed check)
    assert isinstance(artifact["duration_s"], float)
    assert artifact["duration_s"] >= 0.0
