"""Tests for scripts/experiment_672_kv260_dfx_fix.py.

Spec: REQ-VERIFY-083, REQ-INFRA-007
SCENARIO: SCENARIO-INFRA-011
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.experiment_672_kv260_dfx_fix as mod
from scripts.experiment_672_kv260_dfx_fix import (
    EXP_ID,
    TITLE,
    DELIVERABLE,
    _diagnose_failure,
    _try_method,
    main,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_exp_id() -> None:
    """EXP_ID must be 672.  Spec: REQ-VERIFY-083"""
    assert EXP_ID == 672


def test_deliverable_path() -> None:
    """DELIVERABLE must point to the expected results file.  Spec: REQ-VERIFY-083"""
    assert DELIVERABLE == "results/experiment_672_kv260_dfx_fix.json"


# ---------------------------------------------------------------------------
# blocked artifact when CARNOT_KV260_BITFILE not set
# ---------------------------------------------------------------------------


def test_blocked_artifact_when_bitfile_not_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When CARNOT_KV260_BITFILE is unset, main() writes a blocked artifact and exits 0.

    Spec: REQ-INFRA-007 — env-gate: missing environment produces blocked status.
    Why: we must not fail the conductor run just because the board isn't wired up.
    """
    monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)

    main()

    out = tmp_path / DELIVERABLE
    assert out.exists(), "Deliverable must be written even for blocked status"
    data = json.loads(out.read_text())
    assert data["status"] == "blocked"
    assert data["honest_verdict"] == "blocked_bitfile_not_configured"
    assert data["methods_tried"] == []


def test_blocked_artifact_when_bitfile_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When CARNOT_KV260_BITFILE points to a nonexistent file, writes blocked artifact.

    Spec: REQ-INFRA-007
    """
    monkeypatch.setenv("CARNOT_KV260_BITFILE", str(tmp_path / "nonexistent.bit"))
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)

    main()

    out = tmp_path / DELIVERABLE
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["status"] == "blocked"
    assert data["honest_verdict"] == "blocked_bitfile_not_configured"


# ---------------------------------------------------------------------------
# methods_tried list is populated when bitfile exists
# ---------------------------------------------------------------------------


def test_methods_tried_populated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the bitfile exists, methods_tried must be a non-empty list.

    Spec: REQ-VERIFY-083 — artifact schema completeness.
    Why: an empty methods_tried when a bitfile was present means the experiment
    silently skipped the loading logic, which is a bug not a blocked state.
    """
    # Create a dummy bitfile so the env-gate passes.
    bitfile = tmp_path / "carnot_ising_v2_n64.bit.bin"
    bitfile.write_bytes(b"\x00" * 64)

    monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)

    main()

    out = tmp_path / DELIVERABLE
    assert out.exists()
    data = json.loads(out.read_text())
    assert isinstance(data["methods_tried"], list)
    assert len(data["methods_tried"]) >= 1, "At least one method must have been attempted"


# ---------------------------------------------------------------------------
# honest_verdict is always present and non-None
# ---------------------------------------------------------------------------


def test_honest_verdict_not_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """honest_verdict must always be set to a non-None string.

    Spec: REQ-VERIFY-083
    Why: the conductor reads honest_verdict to classify experiment outcomes;
    a None value silently breaks downstream retrospective slicing.
    """
    bitfile = tmp_path / "carnot_ising_v2_n64.bit.bin"
    bitfile.write_bytes(b"\x00" * 64)

    monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)

    main()

    out = tmp_path / DELIVERABLE
    data = json.loads(out.read_text())
    assert data["honest_verdict"] is not None
    assert isinstance(data["honest_verdict"], str)
    assert len(data["honest_verdict"]) > 0


# ---------------------------------------------------------------------------
# honest_verdict prefix matches status
# ---------------------------------------------------------------------------


def test_honest_verdict_matches_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """honest_verdict prefix must be consistent with status.

    blocked -> blocked_* or dfx_protocol_diagnosed_*
    partial -> dfx_method_found_*
    Spec: REQ-VERIFY-083
    """
    bitfile = tmp_path / "carnot_ising_v2_n64.bit.bin"
    bitfile.write_bytes(b"\x00" * 64)

    monkeypatch.setenv("CARNOT_KV260_BITFILE", str(bitfile))
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)

    main()

    out = tmp_path / DELIVERABLE
    data = json.loads(out.read_text())

    status = data["status"]
    verdict = data["honest_verdict"]

    if status == "partial":
        assert verdict.startswith("dfx_method_found_"), (
            f"partial status must have dfx_method_found_* verdict, got: {verdict}"
        )
    else:
        assert verdict.startswith("blocked_") or verdict.startswith("dfx_protocol_diagnosed_"), (
            f"blocked status must have blocked_* or dfx_protocol_diagnosed_* verdict, got: {verdict}"
        )


# ---------------------------------------------------------------------------
# _diagnose_failure: protocol_error recognised
# ---------------------------------------------------------------------------


def test_diagnose_protocol_error() -> None:
    """Timeout/not-responding stderr maps to protocol_error diagnosis.

    Spec: REQ-INFRA-007 — root-cause diagnosis must name the category.
    Why: the operator needs to know to restart dfx-mgr, not fix file permissions.
    """
    methods = [
        {"method_name": "dfx_mgr_client", "stderr": "Timeout, server 192.168.51.98 not responding.", "exit_code": 1},
        {"method_name": "fpgautil", "stderr": "fpgautil: command not found", "exit_code": -1},
        {"method_name": "dd_xdevcfg", "stderr": "dd: failed to open '/dev/xdevcfg': No such file or directory", "exit_code": 1},
        {"method_name": "sysfs_firmware_copy", "stderr": "cp: cannot create regular file '/lib/firmware/x.bit.bin': Permission denied", "exit_code": 1},
    ]
    diag = _diagnose_failure(methods)
    assert diag == "protocol_error", f"Expected protocol_error, got: {diag}"


def test_diagnose_permission_denied() -> None:
    """Permission denied stderr maps to permission_denied diagnosis.

    Spec: REQ-INFRA-007
    """
    methods = [
        {"method_name": "dfx_mgr_client", "stderr": "permission denied", "exit_code": 1},
        {"method_name": "fpgautil", "stderr": "operation not permitted", "exit_code": 1},
        {"method_name": "dd_xdevcfg", "stderr": "permission denied writing to /dev/xdevcfg", "exit_code": 1},
        {"method_name": "sysfs_firmware_copy", "stderr": "Permission denied", "exit_code": 1},
    ]
    diag = _diagnose_failure(methods)
    assert diag == "permission_denied", f"Expected permission_denied, got: {diag}"


# ---------------------------------------------------------------------------
# _try_method: handles missing command gracefully
# ---------------------------------------------------------------------------


def test_try_method_missing_command() -> None:
    """_try_method returns success=False and records stderr when command does not exist.

    Spec: REQ-INFRA-007
    Why: FileNotFoundError from subprocess must not propagate as an uncaught
    exception — it must be caught and recorded so methods_tried stays complete.
    """
    result = _try_method("nonexistent_tool", ["/usr/bin/this_tool_does_not_exist_carnot"])
    assert result["success"] is False
    assert result["exit_code"] == -1
    assert "not found" in result["stderr"].lower() or len(result["stderr"]) > 0


# ---------------------------------------------------------------------------
# Required schema fields present in artifact
# ---------------------------------------------------------------------------


def test_required_schema_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """All REQUIRED_RESULT_FIELDS must be present in the written artifact.

    Spec: REQ-VERIFY-083
    """
    from scripts.experiment_template import REQUIRED_RESULT_FIELDS

    monkeypatch.delenv("CARNOT_KV260_BITFILE", raising=False)
    monkeypatch.setenv("CARNOT_REPO_ROOT", str(tmp_path))
    (tmp_path / "results").mkdir(parents=True, exist_ok=True)

    main()

    out = tmp_path / DELIVERABLE
    data = json.loads(out.read_text())
    for field in REQUIRED_RESULT_FIELDS:
        assert field in data, f"Required field '{field}' missing from artifact"
