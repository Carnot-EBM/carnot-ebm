"""Tests for the janitor's dead-conductor restart (REQ-CONDUCTOR-RESTART-1).

Origin: 2026-08-22 — no conductor ran for 4h39m+ while the janitor logged
"not alive — skipping" every 31 minutes; the deliberate-stop intent lived
only in ops/status.md prose, which no machine reads.

These tests run the TRACKED janitor copy (ops/systemd/orphan-cleanup.sh)
with every path overridden to tmp_path and a stub systemctl that records
its argv. CARNOT_JANITOR_SKIP_SWEEPS=1 keeps the maintenance blocks (real
sentinel / stop authority / sweeps) from ever running against the live box.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JANITOR = REPO / "ops" / "systemd" / "orphan-cleanup.sh"


def _stub_systemctl(tmp_path: Path, is_active_output: str = "inactive", start_rc: int = 0) -> Path:
    """A systemctl stand-in that records argv and answers is-active."""
    calls = tmp_path / "systemctl_calls.log"
    stub = tmp_path / "systemctl"
    stub.write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> "{calls}"\n'
        'if [ "$2" = "is-active" ]; then\n'
        f'  printf "%s\\n" "{is_active_output}"\n'
        "  exit 0\n"
        "fi\n"
        f"exit {start_rc}\n"
    )
    stub.chmod(0o755)
    return calls


def _run_janitor(tmp_path: Path, *, heartbeat: dict | None, hold: bool = False,
                 hold_age_min: int = 0, is_active: str = "inactive") -> tuple[str, str, str]:
    """Run the janitor; returns (janitor log, systemctl calls, conductor log)."""
    hb = tmp_path / "heartbeat.json"
    if heartbeat is not None:
        hb.write_text(json.dumps(heartbeat))
    hold_path = tmp_path / "conductor-hold"
    if hold:
        hold_path.write_text("test hold\n")
        if hold_age_min:
            old = hold_path.stat().st_mtime - hold_age_min * 60
            os.utime(hold_path, (old, old))
    calls = _stub_systemctl(tmp_path, is_active_output=is_active)
    env = dict(os.environ)
    env.update(
        {
            "CARNOT_JANITOR_HEARTBEAT": str(hb),
            "CARNOT_JANITOR_LOG": str(tmp_path / "janitor.log"),
            "CARNOT_JANITOR_HOLD": str(hold_path),
            "CARNOT_JANITOR_SYSTEMCTL": str(tmp_path / "systemctl"),
            "CARNOT_JANITOR_CONDUCTOR_LOG": str(tmp_path / "conductor-log.md"),
            "CARNOT_JANITOR_SKIP_SWEEPS": "1",
            "CARNOT_JANITOR_SKIP_REAP": "1",
        }
    )
    subprocess.run(["bash", str(JANITOR)], env=env, timeout=60, check=True)

    def read(name: str) -> str:
        p = tmp_path / name
        return p.read_text() if p.exists() else ""

    return read("janitor.log"), calls.read_text() if calls.exists() else "", read("conductor-log.md")


def test_dead_conductor_no_hold_starts_the_service(tmp_path):
    """SCENARIO-CONDUCTOR-RESTART-1-DEAD-NO-HOLD."""
    log, calls, clog = _run_janitor(tmp_path, heartbeat={"pid": 999999999})
    assert "start carnot-conductor.service" in calls
    assert "started carnot-conductor.service" in log
    assert "JANITOR: conductor auto-start" in clog
    assert "REQ-CONDUCTOR-RESTART-1" in clog


def test_dead_conductor_with_hold_is_respected(tmp_path):
    """SCENARIO-CONDUCTOR-RESTART-1-HOLD-RESPECTED."""
    log, calls, clog = _run_janitor(tmp_path, heartbeat={"pid": 999999999}, hold=True)
    assert "start carnot-conductor.service" not in calls
    assert "hold marker present" in log
    assert "auto-start" not in clog


def test_stale_hold_escalates_a_warn_once_per_day(tmp_path):
    """Rule 3: a hold older than 48h WARNs durably; deduped by day."""
    log, calls, clog = _run_janitor(
        tmp_path, heartbeat={"pid": 999999999}, hold=True, hold_age_min=49 * 60
    )
    assert "start carnot-conductor.service" not in calls
    assert "conductor hold stale" in clog
    assert clog.count("conductor hold stale") == 1


def test_missing_heartbeat_counts_as_dead(tmp_path):
    log, calls, _ = _run_janitor(tmp_path, heartbeat=None)
    assert "no heartbeat file" in log
    assert "start carnot-conductor.service" in calls


def test_live_conductor_is_not_restarted(tmp_path):
    """A live pid (this test's own) means no liveness action at all."""
    log, calls, clog = _run_janitor(tmp_path, heartbeat={"pid": os.getpid()})
    assert "start" not in calls
    assert "auto-start" not in clog


def test_unit_already_active_is_left_to_systemd(tmp_path):
    """Heartbeat pid dead but unit active = systemd mid-restart or a stale
    heartbeat; starting would race the owner."""
    log, calls, _ = _run_janitor(tmp_path, heartbeat={"pid": 999999999}, is_active="active")
    assert "systemd owns it" in log
    assert "start carnot-conductor.service" not in calls


def test_unreadable_systemd_state_does_nothing(tmp_path):
    """Rule 4 fail direction: absence of information is not permission."""
    log, calls, _ = _run_janitor(tmp_path, heartbeat={"pid": 999999999}, is_active="")
    assert "doing nothing" in log
    assert "start carnot-conductor.service" not in calls


def test_live_and_repo_janitor_copies_match():
    """The live janitor (~/.carnot) drifted from the tracked copy once
    already (the tracked copy was missing the sentinel block entirely).
    When the live copy exists on this box, the two must be byte-identical;
    on a box without it (fresh clone / CI), the tracked copy must at least
    exist and be non-empty. Conditional assertion, not a skip."""
    live = Path.home() / ".carnot" / "orphan-cleanup.sh"
    tracked = JANITOR.read_bytes()
    assert tracked, "tracked janitor copy is empty"
    if live.exists():
        assert live.read_bytes() == tracked, (
            "ops/systemd/orphan-cleanup.sh differs from ~/.carnot/orphan-cleanup.sh; "
            "sync them — the tracked copy is the source of truth"
        )
