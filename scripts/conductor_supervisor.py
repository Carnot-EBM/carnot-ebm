#!/usr/bin/env python3
"""conductor_supervisor.py — external watchdog for research_conductor.py.

No LLM calls. Pure file/process inspection + structured-state reconciliation.
Runs alongside the conductor as a standalone daemon.

Failure modes it catches:
  - Conductor heartbeat stale (log handle severed or conductor wedged)
  - Orphan subagent processes accumulating after conductor restarts
  - Claimed-running experiment with no artifact after 1.5x estimated wall time
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone, UTC
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

HEARTBEAT_FILE = PROJECT_ROOT / "ops" / "conductor-heartbeat.json"
STATE_FILE = PROJECT_ROOT / "ops" / "conductor-state.json"
CONDUCTOR_LOG = PROJECT_ROOT / "ops" / "conductor-log.md"
ALERTS_FILE = PROJECT_ROOT / "ops" / "supervisor-alerts.json"
PID_FILE = PROJECT_ROOT / "ops" / "conductor-supervisor.pid"

# Intervals
HEARTBEAT_POLL_S = 30
ORPHAN_POLL_S = 60
STATE_RECONCILE_S = 120

# Alert thresholds
HEARTBEAT_STALE_S = 90
DEFAULT_WALL_TIME_MIN = 30  # fallback if task doesn't advertise estimated_wall_time_min


def _now_z() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_z(ts: str) -> datetime:
    """Parse a Z-suffix ISO timestamp as returned by the conductor."""
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _write_alert(alert_type: str, detail: str) -> None:
    """Append one JSONL record to the alerts file. Uses O_APPEND for safety."""
    record = json.dumps({"timestamp": _now_z(), "alert_type": alert_type, "detail": detail})
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_FILE, "a") as fh:
        fh.write(record + "\n")
        os.fsync(fh.fileno())


def _read_json_file(path: Path) -> dict | None:
    """Read a JSON file; return None on any error."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. HEARTBEAT MONITOR
# ---------------------------------------------------------------------------


def check_heartbeat() -> bool:
    """Return True if heartbeat is healthy, False if stale/missing."""
    if not HEARTBEAT_FILE.exists():
        _write_alert(
            "heartbeat_missing",
            f"Heartbeat file {HEARTBEAT_FILE} does not exist",
        )
        reset_log_handle()
        return False

    data = _read_json_file(HEARTBEAT_FILE)
    if data is None:
        _write_alert("heartbeat_unreadable", f"Could not parse {HEARTBEAT_FILE}")
        return False

    last_beat_str = data.get("last_beat", "")
    try:
        last_beat = _parse_z(last_beat_str)
    except ValueError:
        _write_alert(
            "heartbeat_bad_timestamp",
            f"Cannot parse last_beat={last_beat_str!r}",
        )
        return False

    now = datetime.now(UTC)
    age_s = (now - last_beat).total_seconds()
    if age_s > HEARTBEAT_STALE_S:
        _write_alert(
            "heartbeat_stale",
            f"last_beat={last_beat_str} is {age_s:.0f}s ago (threshold={HEARTBEAT_STALE_S}s)",
        )
        reset_log_handle()
        return False

    return True


# ---------------------------------------------------------------------------
# 2. ORPHAN REAPER
# ---------------------------------------------------------------------------


def _pgrep_research_conductor() -> list[int]:
    """Return PIDs of processes matching 'research_conductor'."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "research_conductor"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [int(p) for p in result.stdout.split() if p.strip()]
    except Exception:
        return []


def _kill_pid_gracefully(pid: int) -> None:
    """Send SIGTERM; wait 5 s; SIGKILL if still alive."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(10):
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return  # process exited cleanly
    # Still alive after 5 s — escalate
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def reap_orphans() -> None:
    """Terminate any research_conductor processes that aren't the legitimate conductor."""
    state = _read_json_file(STATE_FILE)
    legitimate_pid: int | None = state.get("pid") if state else None

    all_pids = _pgrep_research_conductor()
    # Also exclude ourselves in case script name collides
    my_pid = os.getpid()

    for pid in all_pids:
        if pid in (legitimate_pid, my_pid):
            continue
        _write_alert(
            "orphan_reaped",
            f"Sending SIGTERM to orphan pid={pid} (legitimate_conductor_pid={legitimate_pid})",
        )
        _kill_pid_gracefully(pid)


# ---------------------------------------------------------------------------
# 3. STATE RECONCILER
# ---------------------------------------------------------------------------


def reconcile_state() -> None:
    """Check that the conductor's claimed experiment has a result artifact."""
    state = _read_json_file(STATE_FILE)
    if state is None:
        return  # can't reconcile without state

    exp_id = state.get("current_experiment")
    start_time_str = state.get("started_at") or state.get("start_time")
    wall_time_min = state.get("estimated_wall_time_min", DEFAULT_WALL_TIME_MIN)

    if not exp_id or not start_time_str:
        return  # nothing claimed; nothing to check

    # Check for any artifact matching results/experiment_<id>_*.json
    pattern = str(PROJECT_ROOT / "results" / f"experiment_{exp_id}_*.json")
    artifacts = glob.glob(pattern)

    try:
        start_time = _parse_z(start_time_str)
    except ValueError:
        return

    now = datetime.now(UTC)
    age_s = (now - start_time).total_seconds()
    threshold_s = float(wall_time_min) * 60 * 1.5

    if not artifacts and age_s > threshold_s:
        _write_alert(
            "stale_experiment",
            (
                f"Conductor claims exp={exp_id} started {age_s:.0f}s ago "
                f"but no artifact found at {pattern}; "
                f"threshold={threshold_s:.0f}s"
            ),
        )


# ---------------------------------------------------------------------------
# 4. LOG-HANDLE RESET
# ---------------------------------------------------------------------------


def reset_log_handle() -> None:
    """Archive the conductor log and truncate it so new writes start fresh."""
    if not CONDUCTOR_LOG.exists():
        return
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    archive = CONDUCTOR_LOG.parent / f"conductor-log-archive-{ts}.md"
    try:
        archive.write_text(CONDUCTOR_LOG.read_text())
        CONDUCTOR_LOG.write_text("")  # truncate
        _write_alert(
            "log_handle_reset",
            f"Archived conductor log to {archive.name} and truncated {CONDUCTOR_LOG.name}",
        )
    except Exception as exc:
        _write_alert("log_handle_reset_failed", str(exc))


# ---------------------------------------------------------------------------
# 5. PID FILE
# ---------------------------------------------------------------------------


def _write_pid_file() -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def _check_existing_supervisor() -> bool:
    """Return True if another supervisor is already running."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # signal 0 just checks existence
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        return False


def _remove_pid_file() -> None:
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# 6. GRACEFUL SHUTDOWN
# ---------------------------------------------------------------------------

_running = True


def _handle_sigterm(signum, frame):  # noqa: ANN001
    global _running
    _write_alert("supervisor_shutdown", f"Received signal {signum}; shutting down")
    _running = False


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------


def run(dry_run: bool = False) -> int:
    """Main supervisor loop. Returns 0 on clean exit."""
    if _check_existing_supervisor():
        print("supervisor already running", file=sys.stderr)
        return 1

    _write_pid_file()
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    _write_alert("supervisor_started", f"pid={os.getpid()} dry_run={dry_run}")

    last_heartbeat_check = 0.0
    last_orphan_check = 0.0
    last_reconcile_check = 0.0

    if dry_run:
        # In dry-run mode, run each check once and exit cleanly.
        check_heartbeat()
        reap_orphans()
        reconcile_state()
        _write_alert("supervisor_dry_run_complete", "all checks ran; exiting")
        _remove_pid_file()
        return 0

    try:
        while _running:
            now = time.monotonic()

            if now - last_heartbeat_check >= HEARTBEAT_POLL_S:
                check_heartbeat()
                last_heartbeat_check = now

            if now - last_orphan_check >= ORPHAN_POLL_S:
                reap_orphans()
                last_orphan_check = now

            if now - last_reconcile_check >= STATE_RECONCILE_S:
                reconcile_state()
                last_reconcile_check = now

            time.sleep(5)
    finally:
        _remove_pid_file()

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Carnot conductor supervisor daemon")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all checks once and exit (for smoke-testing)",
    )
    args = parser.parse_args()
    sys.exit(run(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
