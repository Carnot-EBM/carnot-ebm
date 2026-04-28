"""
Tests for scripts/conductor_supervisor.py.

Covers the four required scenarios:
  1. Supervisor starts, writes PID file, stops on SIGTERM.
  2. Heartbeat alert fires when heartbeat file is stale.
  3. Orphan detection works when pgrep finds extra PIDs.
  4. Log-handle reset creates archive and truncates working log.

All tests run without a real conductor process and without LLM calls.
File I/O is redirected to tmp_path so no ops/ directory is touched.

Spec: REQ-INFRA-080 (conductor supervisor watchdog).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone, UTC
from pathlib import Path
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Path fixup — allow importing the script directly without installation.
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import conductor_supervisor as sup  # noqa: E402  (after sys.path fixup)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_paths(tmp_path: Path) -> dict[str, Path]:
    """
    Redirect all module-level path constants to tmp_path so tests are isolated.
    Returns a dict of the patched paths.
    """
    ops = tmp_path / "ops"
    ops.mkdir()
    results = tmp_path / "results"
    results.mkdir()
    return {
        "ops": ops,
        "results": results,
        "heartbeat": ops / "conductor-heartbeat.json",
        "state": ops / "conductor-state.json",
        "alerts": ops / "supervisor-alerts.json",
        "pid": ops / "conductor-supervisor.pid",
        "log": ops / "conductor-log.md",
    }


def _patch_paths(monkeypatch: pytest.MonkeyPatch, paths: dict[str, Path]) -> None:
    monkeypatch.setattr(sup, "OPS_DIR", paths["ops"])
    monkeypatch.setattr(sup, "RESULTS_DIR", paths["results"])
    monkeypatch.setattr(sup, "HEARTBEAT_FILE", paths["heartbeat"])
    monkeypatch.setattr(sup, "STATE_FILE", paths["state"])
    monkeypatch.setattr(sup, "ALERTS_FILE", paths["alerts"])
    monkeypatch.setattr(sup, "PID_FILE", paths["pid"])
    monkeypatch.setattr(sup, "LOG_FILE", paths["log"])


def _read_alerts(paths: dict[str, Path]) -> list[dict]:
    alerts_file = paths["alerts"]
    if not alerts_file.exists():
        return []
    return [json.loads(line) for line in alerts_file.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Test 1: Supervisor starts, writes PID file, stops on SIGTERM
# ---------------------------------------------------------------------------


def test_supervisor_starts_writes_pid_and_stops_on_sigterm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    REQ: Supervisor writes PID file on startup and removes it on SIGTERM.

    We run the supervisor in a subprocess so we can send a real SIGTERM and
    verify the PID file lifecycle without interfering with the test runner's
    own signal handling.
    """
    pid_file = tmp_path / "ops" / "conductor-supervisor.pid"
    alerts_file = tmp_path / "ops" / "supervisor-alerts.json"
    (tmp_path / "ops").mkdir()

    env = os.environ.copy()
    # Point the supervisor at the tmp_path ops dir by monkey-patching the
    # module-level constants via environment — we do this by running with
    # a wrapper that overrides the paths before calling run_supervisor.
    wrapper_script = tmp_path / "run_sup.py"
    wrapper_script.write_text(f"""
import sys
sys.path.insert(0, {str(SCRIPTS_DIR)!r})
import conductor_supervisor as sup
from pathlib import Path
sup.OPS_DIR = Path({str(tmp_path / "ops")!r})
sup.RESULTS_DIR = Path({str(tmp_path / "results")!r})
sup.HEARTBEAT_FILE = sup.OPS_DIR / "conductor-heartbeat.json"
sup.STATE_FILE = sup.OPS_DIR / "conductor-state.json"
sup.ALERTS_FILE = sup.OPS_DIR / "supervisor-alerts.json"
sup.PID_FILE = sup.OPS_DIR / "conductor-supervisor.pid"
sup.LOG_FILE = sup.OPS_DIR / "conductor-log.md"
(tmp_results := sup.RESULTS_DIR).mkdir(exist_ok=True)
sup.run_supervisor(dry_run=False, once=False)
""")

    proc = subprocess.Popen(
        [sys.executable, str(wrapper_script)],
        env=env,
    )

    # Wait for the PID file to appear (up to 5 s).
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if pid_file.exists():
            break
        time.sleep(0.05)

    assert pid_file.exists(), "PID file should be written on startup"
    written_pid = int(pid_file.read_text().strip())
    assert written_pid == proc.pid

    # Send SIGTERM and wait for clean exit.
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)

    assert proc.returncode == 0, f"Supervisor should exit 0 on SIGTERM, got {proc.returncode}"
    assert not pid_file.exists(), "PID file should be removed on exit"

    # Supervisor should have written at least a started + stopped alert.
    alerts = [json.loads(l) for l in alerts_file.read_text().splitlines() if l.strip()]
    alert_types = [a["alert_type"] for a in alerts]
    assert "supervisor_started" in alert_types
    assert "supervisor_stopped" in alert_types


# ---------------------------------------------------------------------------
# Test 2: Heartbeat alert fires when heartbeat file is stale
# ---------------------------------------------------------------------------


def test_heartbeat_alert_fires_when_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    REQ: When conductor-heartbeat.json last_beat is older than HEARTBEAT_STALE_S,
    a heartbeat_stale alert is written to the alerts file.
    """
    paths = _fresh_paths(tmp_path)
    _patch_paths(monkeypatch, paths)

    # Write a heartbeat that is 200 seconds old (well past the 90 s threshold).
    stale_time = datetime.now(UTC) - timedelta(seconds=200)
    paths["heartbeat"].write_text(json.dumps({"last_beat": stale_time.isoformat()}))

    # Suppress the log-handle reset side-effect by mocking _reset_log_handle.
    with mock.patch.object(sup, "_reset_log_handle"):
        result = sup.check_heartbeat(dry_run=False)

    assert result is False, "check_heartbeat should return False on stale heartbeat"
    alerts = _read_alerts(paths)
    stale_alerts = [a for a in alerts if a["alert_type"] == "heartbeat_stale"]
    assert stale_alerts, "Should have written a heartbeat_stale alert"
    assert "200" in stale_alerts[0]["detail"] or "1" in stale_alerts[0]["detail"]


def test_heartbeat_ok_when_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Complementary check: a fresh heartbeat (< 90 s old) returns True with no alert.
    """
    paths = _fresh_paths(tmp_path)
    _patch_paths(monkeypatch, paths)

    fresh_time = datetime.now(UTC) - timedelta(seconds=10)
    paths["heartbeat"].write_text(json.dumps({"last_beat": fresh_time.isoformat()}))

    result = sup.check_heartbeat(dry_run=False)

    assert result is True
    alerts = _read_alerts(paths)
    stale = [a for a in alerts if a["alert_type"] == "heartbeat_stale"]
    assert not stale, "No stale alert expected for fresh heartbeat"


# ---------------------------------------------------------------------------
# Test 3: Orphan detection works when pgrep finds extra PIDs
# ---------------------------------------------------------------------------


def test_orphan_detection_reaps_extra_pids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    REQ: When pgrep returns PIDs that are not the conductor's claimed PID,
    the supervisor sends SIGTERM and writes an orphan_reaped alert.
    """
    paths = _fresh_paths(tmp_path)
    _patch_paths(monkeypatch, paths)

    claimed_pid = 99999  # Fake claimed PID that does NOT appear in pgrep output.
    orphan_pid = 12345  # Extra PID that pgrep returns.

    paths["state"].write_text(json.dumps({"pid": claimed_pid}))

    # Mock pgrep to return both the claimed PID and an orphan.
    monkeypatch.setattr(sup, "_find_conductor_pids", lambda: [claimed_pid, orphan_pid])

    # Mock _terminate_pid to avoid actually sending signals.
    terminate_calls: list[int] = []

    def _fake_terminate(pid: int, dry_run: bool = False) -> str:
        terminate_calls.append(pid)
        return f"[mock] SIGTERM sent to pid={pid}"

    monkeypatch.setattr(sup, "_terminate_pid", _fake_terminate)

    reaped = sup.check_orphans(dry_run=False)

    assert reaped == [orphan_pid], f"Expected orphan_pid in reaped list, got {reaped}"
    assert orphan_pid in terminate_calls, "Should have called _terminate_pid on orphan"
    assert claimed_pid not in terminate_calls, "Should NOT have terminated the claimed PID"

    alerts = _read_alerts(paths)
    orphan_alerts = [a for a in alerts if a["alert_type"] == "orphan_reaped"]
    assert orphan_alerts, "Should have written an orphan_reaped alert"


# ---------------------------------------------------------------------------
# Test 4: Log-handle reset creates archive and truncates working log
# ---------------------------------------------------------------------------


def test_log_handle_reset_archives_and_truncates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    REQ: When _reset_log_handle is called, the conductor log content is
    copied to an archive file and the original log is truncated to zero bytes.
    """
    paths = _fresh_paths(tmp_path)
    _patch_paths(monkeypatch, paths)

    original_content = "| 2026-04-28 | Exp 1027 | OK | supervisor shipped\n"
    paths["log"].write_text(original_content)

    sup._reset_log_handle(dry_run=False)

    # The live log should now be empty.
    assert paths["log"].read_text() == "", "Live log should be truncated to empty"

    # An archive file should exist in ops/ with the original content.
    archive_files = list(paths["ops"].glob("conductor-log-archive-*.md"))
    assert len(archive_files) == 1, "Exactly one archive file should have been created"
    assert archive_files[0].read_text() == original_content

    # An alert should have been written.
    alerts = _read_alerts(paths)
    reset_alerts = [a for a in alerts if a["alert_type"] == "log_handle_reset"]
    assert reset_alerts, "Should have written a log_handle_reset alert"
