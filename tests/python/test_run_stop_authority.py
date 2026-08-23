"""Tests for scripts/run_stop_authority.py — the action half of the sentinel.

REQ-CONDUCTOR-AUTHORITY-1: orphan llama-server reap, every condition
"nothing owns this", fail-toward-not-killing on any unverifiable check.
REQ-CONDUCTOR-AUTHORITY-2: dead-tier run stop only when armed; yes/no
packet when disarmed; row evidence alone never stops a run.

All filesystem interaction goes through tmp_path: fake /proc trees, fake
out files, fake server logs, explicit log/state/arm paths. No test touches
tracked state, and no test sends a real signal — the signaler is always a
recorder.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
from datetime import UTC, datetime, timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_AUTHORITY = _REPO / "scripts" / "run_stop_authority.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_stop_authority", _AUTHORITY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


A = _load()

_NOW = datetime(2026, 8, 23, 12, 0, 0, tzinfo=UTC)
_CLK = os.sysconf("SC_CLK_TCK")
_BTIME = 1_000_000_000


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _mk_proc(proc_root, pid, cmdline, environ=None, ppid=None, cgroup=None, start_epoch=None):
    d = proc_root / str(pid)
    d.mkdir(parents=True)
    (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in cmdline) + b"\0")
    if environ is not None:
        (d / "environ").write_bytes(
            b"\0".join(f"{k}={v}".encode() for k, v in environ.items()) + b"\0"
        )
    if ppid is not None:
        (d / "status").write_text(f"Name:\tx\nPPid:\t{ppid}\n")
    if cgroup is not None:
        (d / "cgroup").write_text(cgroup)
    if start_epoch is not None:
        ticks = int((start_epoch - _BTIME) * _CLK)
        # Real /proc layout after "pid (comm) ": state is index 0 and
        # starttime is index 19 (field 22 overall).
        fields = ["S"] + ["0"] * 49
        fields[19] = str(ticks)
        (d / "stat").write_text(f"{pid} (x) " + " ".join(fields))
    return d


def _mk_proc_root(tmp_path) -> Path:
    proc_root = tmp_path / "proc"
    proc_root.mkdir(exist_ok=True)
    (proc_root / "stat").write_text(f"cpu 0 0 0\nbtime {_BTIME}\n")
    return proc_root


def _mk_orphan_server(
    proc_root,
    *,
    pid=9001,
    port=8993,
    environ=None,
    ppid=1,
    cgroup="0::/user.slice/user-1000.slice/session-1.scope\n",
    age_s=3 * 3600,
):
    _mk_proc(
        proc_root,
        pid,
        ["/usr/bin/llama-server", "-m", "/models/Qwen3.8-27B-Q4_K_M.gguf", "--port", str(port)],
        environ={} if environ is None else environ,
        ppid=ppid,
        cgroup=cgroup,
        start_epoch=_NOW.timestamp() - age_s,
    )


_INVALID_ROW = {
    "llm_enabled": True,
    "llm_tier_operational": True,
    "generator_healthy_after": False,
    "server_storm_suspected": False,
    "llm_on_row_valid": False,
    "llm": {"responses": 0, "calls": 3, "errors": 3, "content_failures": 0},
}
_VALID_ROW = {
    "llm_enabled": True,
    "llm_tier_operational": True,
    "generator_healthy_after": True,
    "server_storm_suspected": False,
    "llm_on_row_valid": True,
    "llm": {"responses": 5, "calls": 6, "errors": 1, "content_failures": 0},
}

_INCIDENT_LOG_LINES = (
    "ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 629440768\n"
    "graph_reserve: failed to allocate compute buffers\n"
    "failed to create context with model 'Qwen3.8-27B-Q4_K_M.gguf'\n"
)


def _mk_run(
    tmp_path,
    proc_root,
    *,
    pid=4242,
    port=8995,
    rows=None,
    out_exists=True,
    environ_extra=None,
    age_s=2 * 3600,
    failing_log=True,
    log_age_s=None,
):
    out = tmp_path / f"rows_{pid}.json"
    if out_exists and rows is not None:
        out.write_text(json.dumps({"rows": rows}))
    slogs = tmp_path / f"slogs_{pid}"
    env = {"CARNOT_ARC_SERVER_LOG_DIR": str(slogs)}
    env.update(environ_extra or {})
    _mk_proc(
        proc_root,
        pid,
        [
            "python",
            "scripts/arc_scored_path_lever_harness.py",
            "--out",
            str(out),
            "--port",
            str(port),
        ],
        environ=env,
        start_epoch=_NOW.timestamp() - age_s,
    )
    if failing_log:
        log_dir = slogs / "carnot_llama_server_logs"
        log_dir.mkdir(parents=True)
        log = log_dir / f"llama_server_p{port}_1.log"
        log.write_text(_INCIDENT_LOG_LINES)
        mtime = _NOW.timestamp() - (log_age_s if log_age_s is not None else 60)
        os.utime(log, (mtime, mtime))
    return out


class _KillRecorder:
    """Records signals; makes the target vanish after SIGTERM so the grace
    poll exits on its first liveness probe."""

    def __init__(self):
        self.sent = []
        self._dead = set()

    def __call__(self, pid, sig):
        if sig == 0:
            if pid in self._dead:
                raise ProcessLookupError(pid)
            return
        self.sent.append((pid, sig))
        if sig == signal.SIGTERM:
            self._dead.add(pid)


def _ss_no_traffic(args):
    return ""  # ss ran, found nothing: no connections / no listener


def _ss_listener_and_conns(args):
    return "ESTAB 0 0 127.0.0.1:x 127.0.0.1:y\n"


def _scan(tmp_path, proc_root, **kw):
    defaults = dict(
        proc_root=proc_root,
        ss_runner=_ss_no_traffic,
        conductor_log=tmp_path / "conductor-log.md",
        known_issues=tmp_path / "known-issues.md",
        state_path=tmp_path / "state.json",
        arm_file=tmp_path / "armed",  # absent by default: DISARMED
        heartbeat_path=tmp_path / "heartbeat.json",  # absent: no conductor pid
        signaler=_KillRecorder(),
        sleeper=lambda s: None,
        now=_NOW,
    )
    defaults.update(kw)
    return A.run_scan(**defaults), defaults["signaler"]


def _seed_persistence(tmp_path, fingerprint, minutes_ago=26):
    first = (_NOW - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp_path / "state.json").write_text(
        json.dumps({"candidates": {fingerprint: first}, "written": {}})
    )


# ---------------------------------------------------------------------------
# REQ-CONDUCTOR-AUTHORITY-1 — orphan reap
# ---------------------------------------------------------------------------


def test_orphan_first_sighting_records_but_does_not_act(tmp_path):
    """SCENARIO-CONDUCTOR-AUTHORITY-1-FIRST-SIGHTING."""
    proc_root = _mk_proc_root(tmp_path)
    _mk_orphan_server(proc_root)
    summary, killer = _scan(tmp_path, proc_root)
    assert summary["orphan_candidates"] == 1
    assert summary["actionable"] == 0
    assert killer.sent == []
    state = json.loads((tmp_path / "state.json").read_text())
    assert any(fp.startswith("ORPHAN|9001|") for fp in state["candidates"])


def test_orphan_reaped_after_persistence(tmp_path):
    """SCENARIO-CONDUCTOR-AUTHORITY-1-ORPHAN-REAP: second sighting 26 min
    later -> SIGTERM, durable actor line, known-issues packet."""
    proc_root = _mk_proc_root(tmp_path)
    _mk_orphan_server(proc_root)
    _seed_persistence(tmp_path, "ORPHAN|9001|8993")
    summary, killer = _scan(tmp_path, proc_root)
    assert summary["actionable"] == 1
    assert (9001, signal.SIGTERM) in killer.sent
    log = (tmp_path / "conductor-log.md").read_text()
    assert "STOP-AUTHORITY: ORPHAN_SERVER_REAPED" in log
    issues = (tmp_path / "known-issues.md").read_text()
    assert "reaped an orphaned llama-server" in issues
    assert "REQ-CONDUCTOR-AUTHORITY-1" in issues


def test_orphan_with_referenced_port_not_touched(tmp_path):
    """SCENARIO-CONDUCTOR-AUTHORITY-1-REFERENCED-PORT: a live harness holds
    the port on its cmdline (the legitimate reparented-server shape)."""
    proc_root = _mk_proc_root(tmp_path)
    _mk_orphan_server(proc_root)
    _mk_proc(proc_root, 5000, ["python", "harness.py", "--port", "8993"])
    _seed_persistence(tmp_path, "ORPHAN|9001|8993")
    summary, killer = _scan(tmp_path, proc_root)
    assert summary["orphan_candidates"] == 0
    assert killer.sent == []


def test_orphan_ss_unavailable_fails_toward_not_killing(tmp_path):
    """SCENARIO-CONDUCTOR-AUTHORITY-1-SS-UNAVAILABLE."""
    proc_root = _mk_proc_root(tmp_path)
    _mk_orphan_server(proc_root)
    _seed_persistence(tmp_path, "ORPHAN|9001|8993")
    summary, killer = _scan(tmp_path, proc_root, ss_runner=lambda args: None)
    assert summary["orphan_candidates"] == 0
    assert killer.sent == []
    assert any("ss unavailable" in n for n in summary["notes"])


def test_orphan_with_established_connection_not_touched(tmp_path):
    proc_root = _mk_proc_root(tmp_path)
    _mk_orphan_server(proc_root)
    _seed_persistence(tmp_path, "ORPHAN|9001|8993")
    summary, killer = _scan(tmp_path, proc_root, ss_runner=_ss_listener_and_conns)
    assert summary["orphan_candidates"] == 0
    assert killer.sent == []


def test_orphan_younger_than_threshold_not_touched(tmp_path):
    proc_root = _mk_proc_root(tmp_path)
    _mk_orphan_server(proc_root, age_s=3600)  # 1h < 2h floor
    _seed_persistence(tmp_path, "ORPHAN|9001|8993")
    summary, killer = _scan(tmp_path, proc_root)
    assert summary["orphan_candidates"] == 0
    assert killer.sent == []


def test_orphan_environ_unreadable_fails_toward_not_killing(tmp_path):
    proc_root = _mk_proc_root(tmp_path)
    _mk_orphan_server(proc_root, environ=None)
    # remove the environ file entirely: unreadable, not empty
    (proc_root / "9001" / "environ").unlink()
    _seed_persistence(tmp_path, "ORPHAN|9001|8993")
    summary, killer = _scan(tmp_path, proc_root)
    assert summary["orphan_candidates"] == 0
    assert killer.sent == []
    assert any("environ unreadable" in n for n in summary["notes"])


def test_orphan_allow_env_opts_out(tmp_path):
    proc_root = _mk_proc_root(tmp_path)
    _mk_orphan_server(proc_root, environ={A.ALLOW_ENV: "1"})
    _seed_persistence(tmp_path, "ORPHAN|9001|8993")
    summary, killer = _scan(tmp_path, proc_root)
    assert summary["orphan_candidates"] == 0
    assert killer.sent == []


def test_systemd_service_server_not_a_candidate(tmp_path):
    proc_root = _mk_proc_root(tmp_path)
    _mk_orphan_server(proc_root, cgroup="0::/system.slice/llama.service\n")
    summary, killer = _scan(tmp_path, proc_root)
    assert summary["orphan_candidates"] == 0


# ---------------------------------------------------------------------------
# REQ-CONDUCTOR-AUTHORITY-2 — run stop
# ---------------------------------------------------------------------------


def test_disarmed_qualifying_run_emits_packet_not_kill(tmp_path):
    """SCENARIO-CONDUCTOR-AUTHORITY-2-DISARMED-PACKET: the yes/no packet
    carries the exact kill, arm, and opt-out instructions."""
    proc_root = _mk_proc_root(tmp_path)
    _mk_run(tmp_path, proc_root, rows=[_INVALID_ROW] * 3)
    _seed_persistence(tmp_path, f"RUNSTOP|4242|{tmp_path}/rows_4242.json")
    summary, killer = _scan(tmp_path, proc_root)
    assert summary["run_candidates"] == 1
    assert summary["packets"] == 1
    assert killer.sent == []
    issues = (tmp_path / "known-issues.md").read_text()
    assert "kill 4242" in issues
    assert "stop-authority-armed" in issues
    assert A.ALLOW_ENV in issues
    log = (tmp_path / "conductor-log.md").read_text()
    assert "STOP_CANDIDATE_AWAITING_OPERATOR" in log


def test_armed_qualifying_run_is_stopped_with_its_server(tmp_path):
    """SCENARIO-CONDUCTOR-AUTHORITY-2-INCIDENT-REPLAY (synthetic bytes;
    the real-bytes replay is below): armed + persistent -> TERM the
    harness and its port-matched server, with a durable actor packet."""
    proc_root = _mk_proc_root(tmp_path)
    _mk_run(tmp_path, proc_root, rows=[_INVALID_ROW] * 3)
    _mk_proc(  # the run's own server, alive, same port
        proc_root,
        9100,
        ["/usr/bin/llama-server", "-m", "/m.gguf", "--port", "8995"],
        ppid=1,
        cgroup="0::/user.slice/x\n",
        start_epoch=_NOW.timestamp() - 3600,
    )
    _seed_persistence(tmp_path, f"RUNSTOP|4242|{tmp_path}/rows_4242.json")
    (tmp_path / "armed").write_text("armed for test\n")
    summary, killer = _scan(tmp_path, proc_root)
    assert (4242, signal.SIGTERM) in killer.sent
    assert (9100, signal.SIGTERM) in killer.sent
    log = (tmp_path / "conductor-log.md").read_text()
    assert "STOP-AUTHORITY: INVALID_RUN_STOPPED" in log
    issues = (tmp_path / "known-issues.md").read_text()
    assert "ARMED" in issues


def test_rows_only_never_stops_a_run(tmp_path):
    """SCENARIO-CONDUCTOR-AUTHORITY-2-ROWS-ONLY: all-invalid rows with a
    HEALTHY server (no failure log, live listener) is the baseline25
    shape — an efficiency judgment, not a validity call. No candidate."""
    proc_root = _mk_proc_root(tmp_path)
    _mk_run(tmp_path, proc_root, rows=[_INVALID_ROW] * 3, failing_log=False)
    _seed_persistence(tmp_path, f"RUNSTOP|4242|{tmp_path}/rows_4242.json")
    (tmp_path / "armed").write_text("armed\n")
    summary, killer = _scan(tmp_path, proc_root, ss_runner=_ss_listener_and_conns)
    assert summary["run_candidates"] == 0
    assert killer.sent == []


def test_valid_rows_with_failing_log_not_a_candidate(tmp_path):
    proc_root = _mk_proc_root(tmp_path)
    _mk_run(tmp_path, proc_root, rows=[_VALID_ROW] * 3)
    _seed_persistence(tmp_path, f"RUNSTOP|4242|{tmp_path}/rows_4242.json")
    (tmp_path / "armed").write_text("armed\n")
    summary, killer = _scan(tmp_path, proc_root)
    assert summary["run_candidates"] == 0
    assert killer.sent == []


def test_one_invalid_row_below_floor_not_a_candidate(tmp_path):
    """One invalid row is a wasted cell the harness tolerates by design
    (and the real baseline25 first attempt had exactly one)."""
    proc_root = _mk_proc_root(tmp_path)
    _mk_run(tmp_path, proc_root, rows=[_INVALID_ROW])
    _seed_persistence(tmp_path, f"RUNSTOP|4242|{tmp_path}/rows_4242.json")
    (tmp_path / "armed").write_text("armed\n")
    summary, killer = _scan(tmp_path, proc_root)
    assert summary["run_candidates"] == 0
    assert killer.sent == []


def test_allow_env_opts_a_run_out(tmp_path):
    """SCENARIO-CONDUCTOR-AUTHORITY-2-ALLOW-ENV."""
    proc_root = _mk_proc_root(tmp_path)
    _mk_run(tmp_path, proc_root, rows=[_INVALID_ROW] * 3, environ_extra={A.ALLOW_ENV: "1"})
    _seed_persistence(tmp_path, f"RUNSTOP|4242|{tmp_path}/rows_4242.json")
    (tmp_path / "armed").write_text("armed\n")
    summary, killer = _scan(tmp_path, proc_root)
    assert summary["run_candidates"] == 0
    assert killer.sent == []


def test_stale_server_log_is_not_evidence(tmp_path):
    """A failure line from an EARLIER run on the same port must not stop
    this run: log older than the run's start is excluded."""
    proc_root = _mk_proc_root(tmp_path)
    _mk_run(tmp_path, proc_root, rows=[_INVALID_ROW] * 3, age_s=3600, log_age_s=7200)
    _seed_persistence(tmp_path, f"RUNSTOP|4242|{tmp_path}/rows_4242.json")
    (tmp_path / "armed").write_text("armed\n")
    summary, killer = _scan(tmp_path, proc_root, ss_runner=_ss_listener_and_conns)
    assert summary["run_candidates"] == 0
    assert killer.sent == []


def test_no_rows_after_thirty_minutes_with_dead_port_is_a_candidate(tmp_path):
    """The supab3 arm=on shape: an hour old, zero rows ever written, server
    gone (no listener). Dead-port evidence + no-rows row evidence."""
    proc_root = _mk_proc_root(tmp_path)
    _mk_run(tmp_path, proc_root, rows=None, out_exists=False, age_s=3600, failing_log=False)
    summary, killer = _scan(tmp_path, proc_root)  # first sighting: record only
    assert summary["run_candidates"] == 1
    assert summary["actionable"] == 0
    assert killer.sent == []


def test_first_sighting_of_run_candidate_does_not_act(tmp_path):
    proc_root = _mk_proc_root(tmp_path)
    _mk_run(tmp_path, proc_root, rows=[_INVALID_ROW] * 3)
    (tmp_path / "armed").write_text("armed\n")
    summary, killer = _scan(tmp_path, proc_root)
    assert summary["run_candidates"] == 1
    assert summary["actionable"] == 0
    assert killer.sent == []


def test_conductor_pid_is_protected(tmp_path):
    """REQ-CONDUCTOR-AUTHORITY-1 rule 4: never signal the conductor, even
    if discovery somehow matches it."""
    proc_root = _mk_proc_root(tmp_path)
    _mk_run(tmp_path, proc_root, rows=[_INVALID_ROW] * 3)
    (tmp_path / "heartbeat.json").write_text(json.dumps({"pid": 4242}))
    _seed_persistence(tmp_path, f"RUNSTOP|4242|{tmp_path}/rows_4242.json")
    (tmp_path / "armed").write_text("armed\n")
    summary, killer = _scan(tmp_path, proc_root)
    assert killer.sent == []
    assert summary["actionable"] == 0


def test_armed_action_dedupes_across_scans(tmp_path):
    proc_root = _mk_proc_root(tmp_path)
    _mk_run(tmp_path, proc_root, rows=[_INVALID_ROW] * 3)
    _seed_persistence(tmp_path, f"RUNSTOP|4242|{tmp_path}/rows_4242.json")
    (tmp_path / "armed").write_text("armed\n")
    summary1, killer1 = _scan(tmp_path, proc_root)
    assert len(summary1["actions"]) == 1
    # same world, next scan: the written-key dedupe holds
    killer2 = _KillRecorder()
    summary2, _ = _scan(tmp_path, proc_root, signaler=killer2)
    assert killer2.sent == []
    assert len(summary2["actions"]) == 0


def test_dry_run_writes_and_kills_nothing(tmp_path):
    proc_root = _mk_proc_root(tmp_path)
    _mk_orphan_server(proc_root)
    _seed_persistence(tmp_path, "ORPHAN|9001|8993")
    state_before = (tmp_path / "state.json").read_text()
    summary, killer = _scan(tmp_path, proc_root, dry_run=True)
    assert summary["actionable"] == 1
    assert killer.sent == []
    assert not (tmp_path / "conductor-log.md").exists()
    assert (tmp_path / "state.json").read_text() == state_before


def test_receipt_always_advances(tmp_path):
    """The state file is the authority's receipt: a clean scan still
    rewrites last_scan_utc (a dead authority must be visible)."""
    proc_root = _mk_proc_root(tmp_path)
    summary, _ = _scan(tmp_path, proc_root)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["last_scan_utc"] == _NOW.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_incident_replay_real_supab3_bytes(tmp_path):
    """REQ-CONDUCTOR-AUTHORITY-2 origin replay with the REAL incident rows
    when the session job dir still holds them; the synthetic-shape replay
    above keeps this covered after the job dir is cleaned."""
    real = Path("/home/ianblenke/.claude/jobs/ad0c053d/tmp/supab3/rows_off.json")
    if not real.exists():
        rows = [_INVALID_ROW] * 3  # job dir cleaned: fall back to the shape
    else:
        rows = json.loads(real.read_text())["rows"]
    proc_root = _mk_proc_root(tmp_path)
    out = tmp_path / "rows_4242.json"
    out.write_text(json.dumps({"rows": rows}))
    slogs = tmp_path / "slogs_4242"
    log_dir = slogs / "carnot_llama_server_logs"
    log_dir.mkdir(parents=True)
    log = log_dir / "llama_server_p8995_1.log"
    log.write_text(_INCIDENT_LOG_LINES)
    _mk_proc(
        proc_root,
        4242,
        ["python", "scripts/arc_scored_path_lever_harness.py", "--out", str(out), "--port", "8995"],
        environ={"CARNOT_ARC_SERVER_LOG_DIR": str(slogs)},
        start_epoch=_NOW.timestamp() - 4 * 3600,
    )
    summary, killer = _scan(tmp_path, proc_root)
    assert summary["run_candidates"] == 1
    assert killer.sent == []  # first sighting: never an action
    state = json.loads((tmp_path / "state.json").read_text())
    assert any(fp.startswith("RUNSTOP|4242|") for fp in state["candidates"])
    # And the candidate carries BOTH halves of the evidence, verbatim:
    sentinel = A.load_sentinel()
    lint = sentinel.load_liveness_lint()
    cands = A.evaluate_run_stop_candidates(
        sentinel, lint, proc_root=proc_root, now_s=_NOW.timestamp(), ss_runner=_ss_no_traffic
    )
    assert len(cands) == 1
    assert "3/3 LLM-on rows invalid" in cands[0]["row_evidence"]
    assert "failed to allocate" in cands[0]["server_evidence"]


def test_sentinel_source_still_has_no_kill_path():
    """The division of labor holds: the kill primitive lives HERE, and the
    sentinel stays read-only (its own source test also asserts this)."""
    sentinel_src = (_REPO / "scripts" / "conductor_run_sentinel.py").read_text()
    for token in ("os.kill", "killpg", "send_signal", "SIGKILL", "SIGTERM", ".terminate("):
        assert token not in sentinel_src
    authority_src = _AUTHORITY.read_text()
    assert "SIGTERM" in authority_src and "SIGKILL" in authority_src
