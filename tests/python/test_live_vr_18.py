"""Tests for Experiment 656: Live VR Attempt #18 (Structured Equation Forcing).

Tests cover:
- Gate check logic: a closed gate produces a blocked artifact (SCENARIO-VERIFY-202)
- CI stub path: gate open but CARNOT_FORCE_LIVE unset produces ci_stub artifact (SCENARIO-VERIFY-203)

These tests do NOT require a live GPU.  They exercise the gate and stub branches
of scripts/experiment_656_live_vr_attempt_18.py by calling its helper functions
and the main() entrypoint with the relevant environment variables controlled.

Spec: REQ-VERIFY-150, SCENARIO-VERIFY-202, SCENARIO-VERIFY-203
"""

import json
import os
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Path wiring: ensure the repo root is on sys.path so `scripts` is importable.
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Helper: write a temporary gate file with the given gate_open value.
# ---------------------------------------------------------------------------


def _write_gate_file(path: str, gate_open: bool) -> None:
    """Write a minimal Exp 655 gate result to *path*."""
    payload = {
        "experiment": 655,
        "status": "success",
        "gate_open": gate_open,
        "gate_threshold": 0.30,
        "gate_version": "v3",
        "honest_verdict": "gate_open_vr18_authorized" if gate_open else "gate_closed_vr18_blocked",
    }
    with open(path, "w") as fh:
        json.dump(payload, fh)


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-202: Gate closed → blocked artifact, exit 0
# ---------------------------------------------------------------------------


def test_gate_closed_writes_blocked_artifact(monkeypatch, tmp_path):
    """REQ-VERIFY-150-1: When gate_open=False, main() writes a blocked artifact.

    Why: The gate guards against running VR when the structured-recall signal
    is below threshold.  Running VR on a closed gate would produce misleading
    metrics and continue a failed strategy.
    """
    # Write a closed gate file.
    gate_path = tmp_path / "experiment_655_ensemble_gate_v3.json"
    deliverable_path = tmp_path / "experiment_656_live_vr_attempt_18.json"
    _write_gate_file(str(gate_path), gate_open=False)

    # Patch the module-level constants so the script uses our temp files.
    import scripts.experiment_656_live_vr_attempt_18 as exp656  # noqa: PLC0415
    monkeypatch.setattr(exp656, "GATE_FILE", str(gate_path))
    monkeypatch.setattr(exp656, "DELIVERABLE", str(deliverable_path))
    monkeypatch.setattr(exp656, "_REPO_ROOT", str(tmp_path))

    # Run main() — should NOT raise, should NOT call sys.exit.
    exp656.main()

    assert deliverable_path.exists(), "Blocked artifact must be written"
    artifact = json.loads(deliverable_path.read_text())

    # REQ-VERIFY-150-1: status must be 'blocked', gate_open must be False.
    assert artifact["status"] == "blocked"
    assert artifact["gate_open"] is False
    assert artifact["honest_verdict"] == "vr18_blocked_gate_closed"
    assert artifact["retro_033_attempt"] == 18


def test_gate_file_absent_writes_blocked_artifact(monkeypatch, tmp_path):
    """REQ-VERIFY-150-1: When gate file is absent, treat as gate closed."""
    missing_gate = str(tmp_path / "nonexistent_655.json")
    deliverable_path = tmp_path / "experiment_656_live_vr_attempt_18.json"

    import scripts.experiment_656_live_vr_attempt_18 as exp656  # noqa: PLC0415
    monkeypatch.setattr(exp656, "GATE_FILE", missing_gate)
    monkeypatch.setattr(exp656, "DELIVERABLE", str(deliverable_path))
    monkeypatch.setattr(exp656, "_REPO_ROOT", str(tmp_path))

    exp656.main()

    artifact = json.loads(deliverable_path.read_text())
    assert artifact["status"] == "blocked"
    assert artifact["gate_open"] is False


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-203: Gate open, CARNOT_FORCE_LIVE unset → CI stub artifact
# ---------------------------------------------------------------------------


def test_ci_stub_when_gate_open_no_force_live(monkeypatch, tmp_path):
    """REQ-VERIFY-150-2: Gate open but CARNOT_FORCE_LIVE unset → ci_stub artifact.

    Why: In CI (no GPU), we must not load any model.  The stub path produces a
    deterministic artifact so the conductor can see that the code path ran.
    """
    gate_path = tmp_path / "experiment_655_ensemble_gate_v3.json"
    deliverable_path = tmp_path / "experiment_656_live_vr_attempt_18.json"
    _write_gate_file(str(gate_path), gate_open=True)

    import scripts.experiment_656_live_vr_attempt_18 as exp656  # noqa: PLC0415
    monkeypatch.setattr(exp656, "GATE_FILE", str(gate_path))
    monkeypatch.setattr(exp656, "DELIVERABLE", str(deliverable_path))
    monkeypatch.setattr(exp656, "_REPO_ROOT", str(tmp_path))

    # Ensure CARNOT_FORCE_LIVE is NOT set.
    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)

    exp656.main()

    assert deliverable_path.exists(), "CI stub artifact must be written"
    artifact = json.loads(deliverable_path.read_text())

    # REQ-VERIFY-150-2: gate_open=True, inference_mode must be ci_stub_gpu_required.
    assert artifact["gate_open"] is True
    assert artifact["inference_mode"] == "ci_stub_gpu_required"
    assert artifact["status"] == "success"
    assert artifact["retro_033_attempt"] == 18
    assert artifact["using_structured_forcing"] is True


# ---------------------------------------------------------------------------
# _load_gate helper tests
# ---------------------------------------------------------------------------


def test_load_gate_parses_gate_open_true(tmp_path):
    """REQ-VERIFY-150-1: _load_gate() returns dict with gate_open=True."""
    import scripts.experiment_656_live_vr_attempt_18 as exp656  # noqa: PLC0415

    path = str(tmp_path / "gate.json")
    _write_gate_file(path, gate_open=True)

    monkeypatch_attr = None
    # Call _load_gate directly using the path via temporary override.
    original = exp656.GATE_FILE
    exp656.GATE_FILE = path
    try:
        result = exp656._load_gate()
    finally:
        exp656.GATE_FILE = original

    assert result.get("gate_open") is True


def test_load_gate_returns_empty_dict_when_file_absent(tmp_path):
    """REQ-VERIFY-150-1: _load_gate() returns {} for missing file (treats as closed)."""
    import scripts.experiment_656_live_vr_attempt_18 as exp656  # noqa: PLC0415

    original = exp656.GATE_FILE
    exp656.GATE_FILE = str(tmp_path / "no_such_file.json")
    try:
        result = exp656._load_gate()
    finally:
        exp656.GATE_FILE = original

    assert result == {}
    # An empty dict means gate_open falsy → blocked.
    assert not bool(result.get("gate_open", False))


# ---------------------------------------------------------------------------
# _load_live_pairs helper test
# ---------------------------------------------------------------------------


def test_load_live_pairs_returns_empty_when_file_absent(tmp_path):
    """REQ-VERIFY-150-3: _load_live_pairs returns [] if live_pairs file is absent."""
    import scripts.experiment_656_live_vr_attempt_18 as exp656  # noqa: PLC0415

    original = exp656.LIVE_PAIRS_FILE
    exp656.LIVE_PAIRS_FILE = str(tmp_path / "no_pairs.json")
    try:
        result = exp656._load_live_pairs(25)
    finally:
        exp656.LIVE_PAIRS_FILE = original

    assert result == []


def test_load_live_pairs_truncates_to_n(tmp_path):
    """REQ-VERIFY-150-3: _load_live_pairs returns at most n entries."""
    import scripts.experiment_656_live_vr_attempt_18 as exp656  # noqa: PLC0415

    pairs_path = tmp_path / "pairs.json"
    pairs = [{"question": f"q{i}", "response": f"r{i}"} for i in range(30)]
    pairs_path.write_text(json.dumps(pairs))

    original = exp656.LIVE_PAIRS_FILE
    exp656.LIVE_PAIRS_FILE = str(pairs_path)
    try:
        result = exp656._load_live_pairs(10)
    finally:
        exp656.LIVE_PAIRS_FILE = original

    assert len(result) == 10
