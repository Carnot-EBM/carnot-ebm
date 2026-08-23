#!/usr/bin/env python3
"""Stop authority: the ACTION half of the run sentinel's findings.

WHY THIS EXISTS (REQ-CONDUCTOR-AUTHORITY-1/2). On 2026-08-22 the sentinel
correctly escalated a run whose every LLM-on row was invalid — and the run
burned 1h33m more before a human killed it, then a sibling arm launched into
the same broken GPU and OOMed for another 1h08m. The same day an orphaned
llama-server held 20.7 GB with PPID 1 and zero connections until a human
killed it: the sentinel never kills by design, and the janitor's kill filter
only matches `python3`/`pytest`. Detection was machine; action was human.
This tool owns the action, under evidence gates measured before shipping.

DIVISION OF LABOR. The sentinel READS and escalates (kill-free, enforced by
a source test). The janitor SCHEDULES (30-minute systemd timer) and keeps
its own python3/pytest reaping. This authority ACTS, on two candidate
classes only:

  * ORPHAN llama-server reap (default ON): a server nothing can be using —
    PPID 1, no systemd service cgroup, port referenced by no live process,
    no established TCP connection, older than 2 hours, seen on two scans
    at least 25 minutes apart. Every condition is "nothing owns this".
  * DEAD-TIER run stop (default OFF, armed by ~/.carnot/stop-authority-
    armed): a live ARC harness run whose LLM-on rows are ALL invalid (the
    sentinel's own CRITICAL row shape) or which has written no rows in 30+
    minutes, AND whose own server log carries an unambiguous failure line
    (or whose declared port has no listener), seen on two scans 25+ minutes
    apart. Row evidence alone never stops a run: the baseline25 case
    (invalid rows, healthy generator, thinking misconfigured) was a human
    efficiency call. Validity is machine; efficiency stays human.
    When DISARMED, a qualifying candidate emits a yes/no packet instead:
    the evidence plus the exact stop / arm / opt-out commands.

MEASURED BEFORE BUILT (2026-08-23): the row half of the run-stop predicate
fires on 0 of 18,539 recorded results/** artifacts (413 with LLM-on rows)
and on exactly the two true incident row files (supab2 3/3, supab3 3/3).
Any widening of a stop predicate requires a new recorded sweep first
(REQ-CONDUCTOR-AUTHORITY-2 rule 5).

FAIL DIRECTION, per check — every inability to verify fails toward NOT
killing, and says so in the state notes:
  * `ss` cannot run            -> assume a connection exists; no kill
  * environ unreadable         -> assume the opt-out is set; no kill
  * /proc entry vanishes       -> skip that pid (races are normal)
  * state file unreadable      -> fresh state; persistence restarts (delay)
  * kill returns EPERM         -> escalate WARN (someone else owns it)

Every action writes a durable actor line (conductor-log row + known-issues
section): an unexplained dead process is its own incident class here
(2026-08-09 exit-143, sender never identified), and this project's own
reapers must never add to it. The state file records `last_scan_utc` on
EVERY run — the authority's receipt, same contract as the sentinel's.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Operator arm marker for the run-stop action (REQ-CONDUCTOR-AUTHORITY-2
# rule 2). Its ABSENCE is the shipped default: packets, never kills.
ARM_FILE = Path.home() / ".carnot" / "stop-authority-armed"

# Opt-out env var: a run or server that legitimately looks dead (a
# diagnostic DESIGNED to observe server death) sets this and is never a
# candidate (REQ-CONDUCTOR-AUTHORITY-2 rule 1e).
ALLOW_ENV = "CARNOT_STOP_AUTHORITY_ALLOW"

# A candidate must be observed on a prior scan at least this much earlier
# before any action. Two janitor cycles are 30 minutes apart; 25 minutes
# means one full cycle of persistence, and a fast conductor loop invoking
# this more often cannot rush a kill.
PERSISTENCE_MIN_S = 25 * 60

# Orphan servers must be older than this. Matches the janitor's own
# 2-hour orphan threshold for python workers.
ORPHAN_MIN_AGE_S = 2 * 3600

# A run with NO parseable rows at all becomes a candidate only after this
# age (the supab3 arm=on shape: 1h08m, zero rows, server OOM at load).
NO_ROWS_MIN_AGE_S = 30 * 60

# Grace between SIGTERM and SIGKILL.
KILL_GRACE_S = 10.0

# A written escalation/action fingerprint re-arms after this long, same
# semantics as the sentinel's dedupe.
STATE_REARM_DAYS = 14


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_sentinel():
    """The sentinel owns discovery, row evaluation, and log formats.
    Importing them keeps one pattern list per concept (the duplicated-list
    bug class the QA discipline names)."""
    return _load_script("conductor_run_sentinel", "conductor_run_sentinel.py")


# ---------------------------------------------------------------------------
# host probes (injectable for tests)
# ---------------------------------------------------------------------------


def run_ss(args: list[str]) -> str | None:
    """ss wrapper; None on any failure (callers must fail toward no-kill)."""
    try:
        proc = subprocess.run(
            ["ss", *args], capture_output=True, text=True, timeout=15, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def port_has_established_conn(port: int, ss_runner=run_ss) -> bool | None:
    """True/False when ss answered; None when it could not (unverifiable)."""
    out = ss_runner(["-Htn", "state", "established", f"( sport = :{port} )"])
    if out is None:
        return None
    return bool(out.strip())


def port_has_listener(port: int, ss_runner=run_ss) -> bool | None:
    out = ss_runner(["-Hltn", f"( sport = :{port} )"])
    if out is None:
        return None
    return bool(out.strip())


def proc_start_epoch(pid: int, proc_root: Path = Path("/proc")) -> float | None:
    """Process start time as a unix epoch, from /proc; None on any failure.

    /proc/<pid>/stat field 22 is starttime in clock ticks since boot; boot
    time comes from /proc/stat `btime`. The comm field may contain spaces
    and parens, so split on the LAST ') '.
    """
    try:
        stat = (proc_root / str(pid) / "stat").read_text()
        after_comm = stat.rsplit(") ", 1)[1].split()
        starttime_ticks = int(after_comm[19])
        btime = None
        for line in (proc_root / "stat").read_text().splitlines():
            if line.startswith("btime"):
                btime = int(line.split()[1])
                break
        if btime is None:
            return None
        return btime + starttime_ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return None


def read_environ_strict(pid: int, proc_root: Path = Path("/proc")) -> dict[str, str] | None:
    """Environ as a dict, or None when UNREADABLE. The sentinel's _environ
    returns {} for both empty and unreadable; this tool must tell them
    apart, because an unverifiable opt-out fails toward NOT killing."""
    try:
        raw = (proc_root / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    env: dict[str, str] = {}
    for chunk in raw.decode("utf-8", "replace").split("\0"):
        if "=" in chunk:
            key, _, value = chunk.partition("=")
            env[key] = value
    return env


def protected_pids(heartbeat_path: Path | None = None) -> set[int]:
    """PIDs this authority never signals: itself and the conductor."""
    pids = {os.getpid()}
    heartbeat_path = heartbeat_path or REPO / "ops" / "conductor-heartbeat.json"
    try:
        doc = json.loads(heartbeat_path.read_text(encoding="utf-8"))
        pid = doc.get("pid")
        if isinstance(pid, int):
            pids.add(pid)
    except (OSError, ValueError):
        pass  # unreadable heartbeat: the conductor pid is unknown, not 0
    return pids


# ---------------------------------------------------------------------------
# candidate evaluation (pure given probes)
# ---------------------------------------------------------------------------


def evaluate_orphan_candidates(
    sentinel,
    *,
    proc_root: Path,
    now_s: float,
    ss_runner=run_ss,
    notes: list[str] | None = None,
) -> list[dict]:
    """Orphan llama-servers meeting EVERY static reap condition
    (REQ-CONDUCTOR-AUTHORITY-1 rule 1a-1e; persistence 1f is applied by
    the caller against state)."""
    notes = notes if notes is not None else []
    candidates = []
    for server in sentinel.discover_llama_servers(proc_root):
        pid = server["pid"]
        scope = f"llama-server pid {pid} port {server.get('port')}"
        # 1a: reparented to init, and not a systemd service.
        if server.get("ppid") != 1 or "/system.slice/" in server.get("cgroup", ""):
            continue
        # 1e first (cheapest veto): opt-out env. Unreadable environ fails
        # toward NOT killing — we cannot prove the opt-out is absent.
        env = read_environ_strict(pid, proc_root)
        if env is None:
            notes.append(f"{scope}: environ unreadable; opt-out unverifiable; no action")
            continue
        if env.get(ALLOW_ENV) == "1":
            continue
        # 1b: no live process references the port (sentinel's own check —
        # one implementation of the concept, never a second copy).
        port = server.get("port")
        if port is not None and sentinel._port_referenced_elsewhere(port, pid, proc_root):
            continue
        # 1c: no established connection. None = ss failed = unverifiable
        # = assume a connection exists (fail toward NOT killing).
        if port is not None:
            conn = port_has_established_conn(port, ss_runner)
            if conn is None:
                notes.append(f"{scope}: ss unavailable; connections unverifiable; no action")
                continue
            if conn:
                continue
        # 1d: age. Unknown start time is unverifiable -> no action.
        start = proc_start_epoch(pid, proc_root)
        if start is None:
            notes.append(f"{scope}: start time unreadable; age unverifiable; no action")
            continue
        if now_s - start < ORPHAN_MIN_AGE_S:
            continue
        candidates.append(
            {
                "kind": "orphan_server",
                "pid": pid,
                "port": port,
                "age_s": int(now_s - start),
                "model_path": server.get("model_path"),
                "evidence": (
                    f"ppid=1, non-service cgroup, port {port} referenced by no live "
                    f"process, no established connections, age {int((now_s - start) / 60)} min"
                ),
            }
        )
    return candidates


def evaluate_run_stop_candidates(
    sentinel,
    lint,
    *,
    proc_root: Path,
    now_s: float,
    ss_runner=run_ss,
    notes: list[str] | None = None,
) -> list[dict]:
    """Live runs meeting the dead-tier predicate (REQ-CONDUCTOR-AUTHORITY-2
    rule 1a-1c, 1e; persistence 1d is applied by the caller)."""
    notes = notes if notes is not None else []
    candidates = []
    for run in sentinel.discover_live_runs(proc_root):
        pid = run["pid"]
        out_path = Path(run["out_path"])
        scope = f"run pid {pid} {out_path}"
        # 1e: opt-out. Unreadable environ -> unverifiable -> no action.
        env = read_environ_strict(pid, proc_root)
        if env is None:
            notes.append(f"{scope}: environ unreadable; opt-out unverifiable; no action")
            continue
        if env.get(ALLOW_ENV) == "1":
            continue
        start = proc_start_epoch(pid, proc_root)
        if start is None:
            notes.append(f"{scope}: start time unreadable; no action")
            continue
        # 1b row evidence: the sentinel's OWN evaluator (one row concept).
        # CRITICAL on CONSECUTIVE_INVALID_LLM_ON_ROWS means every LLM-on
        # row invalid with at least two rows — exactly the incident shape.
        doc, reason = sentinel.read_out_file(out_path, now_s)
        row_evidence = None
        if doc is not None:
            rows = sentinel.extract_rows(doc, lint)
            for finding in sentinel.evaluate_rows(rows, lint):
                if (
                    finding["code"] == "CONSECUTIVE_INVALID_LLM_ON_ROWS"
                    and finding["severity"] == "CRITICAL"
                ):
                    row_evidence = finding["detail"]
        elif reason in ("missing", "stale_unparseable") and now_s - start >= NO_ROWS_MIN_AGE_S:
            # The supab3 arm=on shape: an hour old, zero rows ever written.
            row_evidence = f"no parseable rows after {int((now_s - start) / 60)} min ({reason})"
        if not row_evidence:
            continue
        # 1c server evidence — REQUIRED. Rows alone never stop a run
        # (validity is machine; efficiency stays human).
        server_evidence = None
        port = run.get("port")
        log_dir = run.get("server_log_dir")
        if log_dir:
            for log_path in sentinel.find_server_logs(Path(log_dir), port):
                try:
                    # Only logs at least as new as the run: a failure line
                    # from an earlier run on the same port is not evidence
                    # against THIS run.
                    if log_path.stat().st_mtime < start - 60:
                        continue
                    hits = sentinel.scan_server_log_text(
                        log_path.read_text(encoding="utf-8", errors="replace")
                    )
                except OSError:
                    continue
                if hits:
                    server_evidence = f"{log_path.name}: {hits[0][:100]}"
                    break
        if server_evidence is None and port is not None:
            listener = port_has_listener(port, ss_runner)
            # None = ss failed = unverifiable = treat as listener present
            # (no evidence; fail toward NOT killing).
            if listener is False:
                server_evidence = f"declared --port {port} has no listener"
        if not server_evidence:
            continue
        candidates.append(
            {
                "kind": "run_stop",
                "pid": pid,
                "port": port,
                "out_path": str(out_path),
                "age_s": int(now_s - start),
                "row_evidence": row_evidence,
                "server_evidence": server_evidence,
                "evidence": f"rows: {row_evidence}; server: {server_evidence}",
            }
        )
    return candidates


def _fingerprint(candidate: dict) -> str:
    if candidate["kind"] == "orphan_server":
        return f"ORPHAN|{candidate['pid']}|{candidate.get('port')}"
    return f"RUNSTOP|{candidate['pid']}|{candidate.get('out_path')}"


# ---------------------------------------------------------------------------
# persistence + state (the authority's receipt)
# ---------------------------------------------------------------------------


def load_state(state_path: Path) -> dict:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(state, dict):
            state.setdefault("candidates", {})
            state.setdefault("written", {})
            return state
    except (OSError, json.JSONDecodeError):
        pass
    # Unreadable state = fresh state: persistence restarts, which only
    # DELAYS an action (fail toward not killing).
    return {"candidates": {}, "written": {}}


def write_state(state_path: Path, state: dict) -> None:
    """Atomic: the state file is the authority's receipt."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, state_path)


def apply_persistence(state: dict, candidates: list[dict], now: datetime) -> list[dict]:
    """Split candidates into (actionable now) per REQ rule 1f/1d: seen on a
    prior scan at least PERSISTENCE_MIN_S earlier. Candidates absent this
    scan are pruned, so a recovered condition resets its clock."""
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    seen_now = {}
    actionable = []
    for candidate in candidates:
        fp = _fingerprint(candidate)
        first_seen = state["candidates"].get(fp, now_iso)
        seen_now[fp] = first_seen
        try:
            first_dt = datetime.strptime(first_seen, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        except ValueError:
            seen_now[fp] = now_iso
            continue
        if (now - first_dt).total_seconds() >= PERSISTENCE_MIN_S:
            candidate["first_seen"] = first_seen
            actionable.append(candidate)
    state["candidates"] = seen_now
    return actionable


def _dedupe_ok(state: dict, key: str, now: datetime) -> bool:
    prior = state["written"].get(key)
    if not prior:
        return True
    try:
        prior_dt = datetime.strptime(prior, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return True  # unparseable stamp -> rewrite rather than stay silent
    return (now - prior_dt).total_seconds() >= STATE_REARM_DAYS * 86400


# ---------------------------------------------------------------------------
# action + durable records
# ---------------------------------------------------------------------------


def armed(arm_file: Path = ARM_FILE) -> bool:
    return arm_file.exists()


def kill_with_grace(
    pid: int, *, signaler=os.kill, sleeper=time.sleep, grace_s: float = KILL_GRACE_S
) -> str:
    """SIGTERM, wait, SIGKILL if still alive. Returns a short outcome word."""
    try:
        signaler(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_gone"
    except PermissionError:
        return "eperm"
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        sleeper(0.5)
        try:
            signaler(pid, 0)
        except ProcessLookupError:
            return "terminated"
        except PermissionError:
            return "eperm"
    try:
        signaler(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "terminated"
    except PermissionError:
        return "eperm"
    return "killed"


def _append_packet(known_issues: Path, title: str, body: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    with open(known_issues, "a", encoding="utf-8") as fh:
        fh.write(f"\n## OPERATOR-ATTENTION {stamp}: {title}\n\n{body}\n")


def run_stop_packet(candidate: dict) -> str:
    """The yes/no packet (REQ-CONDUCTOR-AUTHORITY-2 rule 3): evidence plus
    the exact commands. The human decision becomes yes/no, never an
    investigation."""
    pid = candidate["pid"]
    return (
        f"Stop-authority candidate (DISARMED — no action taken).\n\n"
        f"Run: pid {pid}, out {candidate.get('out_path')}, "
        f"age {candidate['age_s'] // 60} min, first seen {candidate.get('first_seen')}.\n"
        f"Row evidence: {candidate['row_evidence']}\n"
        f"Server evidence: {candidate['server_evidence']}\n\n"
        f"YES (stop it now, by hand):   kill {pid}\n"
        f"YES (always, arm the authority): "
        f'echo "$(date -u +%FT%TZ) armed by operator" > {ARM_FILE}\n'
        f"NO (this run is legitimate): relaunch it with {ALLOW_ENV}=1 in its "
        f"environment, or ignore this packet — the authority never acts while "
        f"disarmed.\n\n"
        f"Written by scripts/run_stop_authority.py (REQ-CONDUCTOR-AUTHORITY-2)."
    )


def run_scan(
    *,
    proc_root: Path = Path("/proc"),
    ss_runner=run_ss,
    conductor_log: Path | None = None,
    known_issues: Path | None = None,
    state_path: Path | None = None,
    arm_file: Path | None = None,
    heartbeat_path: Path | None = None,
    signaler=os.kill,
    sleeper=time.sleep,
    now: datetime | None = None,
    dry_run: bool = False,
) -> dict:
    """One scan-decide-act pass. Returns a summary; writes durable records."""
    sentinel = load_sentinel()
    lint = sentinel.load_liveness_lint()
    conductor_log = conductor_log or REPO / "ops" / "conductor-log.md"
    known_issues = known_issues or REPO / "ops" / "known-issues.md"
    state_path = state_path or REPO / "ops" / ".stop_authority_state.json"
    arm_file = arm_file or ARM_FILE
    now = now or datetime.now(UTC)
    now_s = now.timestamp()
    notes: list[str] = []
    protected = protected_pids(heartbeat_path)

    orphans = evaluate_orphan_candidates(
        sentinel, proc_root=proc_root, now_s=now_s, ss_runner=ss_runner, notes=notes
    )
    runs = evaluate_run_stop_candidates(
        sentinel, lint, proc_root=proc_root, now_s=now_s, ss_runner=ss_runner, notes=notes
    )
    candidates = [c for c in orphans + runs if c["pid"] not in protected]

    state = load_state(state_path)
    actionable = apply_persistence(state, candidates, now)
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    actions: list[dict] = []
    packets = 0

    for candidate in actionable:
        fp = _fingerprint(candidate)
        if candidate["kind"] == "orphan_server":
            # Default-ON: every condition says nothing owns this server.
            key = f"act|{fp}"
            if not _dedupe_ok(state, key, now):
                continue
            outcome = (
                "dry_run"
                if dry_run
                else kill_with_grace(candidate["pid"], signaler=signaler, sleeper=sleeper)
            )
            detail = f"pid {candidate['pid']}: {candidate['evidence']} -> {outcome}"
            if not dry_run:
                sentinel._append_conductor_log_row(
                    conductor_log, "STOP-AUTHORITY: ORPHAN_SERVER_REAPED", "WARN", detail
                )
                _append_packet(
                    known_issues,
                    "stop authority reaped an orphaned llama-server",
                    f"{detail}\n\nEvery reap condition and its value:\n"
                    f"{candidate['evidence']}.\nActor: scripts/run_stop_authority.py "
                    f"(REQ-CONDUCTOR-AUTHORITY-1). If this kill was wrong, set "
                    f"{ALLOW_ENV}=1 in the server's environment at launch.",
                )
                state["written"][key] = now_iso
            actions.append({"fingerprint": fp, "outcome": outcome, **candidate})
        else:  # run_stop
            if armed(arm_file):
                key = f"act|{fp}"
                if not _dedupe_ok(state, key, now):
                    continue
                if dry_run:
                    outcome = "dry_run"
                    server_outcome = "dry_run"
                else:
                    outcome = kill_with_grace(candidate["pid"], signaler=signaler, sleeper=sleeper)
                    # Also stop the run's own llama-server: a dead-tier run's
                    # server is either already broken or serving nothing.
                    server_outcome = "no_server_found"
                    for server in sentinel.discover_llama_servers(proc_root):
                        if (
                            candidate.get("port") is not None
                            and server.get("port") == candidate["port"]
                            and server["pid"] not in protected
                        ):
                            server_outcome = kill_with_grace(
                                server["pid"], signaler=signaler, sleeper=sleeper
                            )
                            break
                detail = (
                    f"pid {candidate['pid']} -> {outcome}; server -> {server_outcome}; "
                    f"{candidate['evidence']}"
                )
                if not dry_run:
                    sentinel._append_conductor_log_row(
                        conductor_log, "STOP-AUTHORITY: INVALID_RUN_STOPPED", "BLOCK", detail
                    )
                    _append_packet(
                        known_issues,
                        "stop authority stopped a dead-tier run (ARMED)",
                        f"{detail}\n\nFirst seen {candidate.get('first_seen')}; acted "
                        f"{now_iso}. Rows already written are preserved in "
                        f"{candidate.get('out_path')}; the loss bound is the in-flight "
                        f"cell.\nActor: scripts/run_stop_authority.py "
                        f"(REQ-CONDUCTOR-AUTHORITY-2 rule 4).",
                    )
                    state["written"][key] = now_iso
                actions.append(
                    {
                        "fingerprint": fp,
                        "outcome": outcome,
                        "server_outcome": server_outcome,
                        **candidate,
                    }
                )
            else:
                key = f"packet|{fp}"
                if not _dedupe_ok(state, key, now):
                    continue
                if not dry_run:
                    sentinel._append_conductor_log_row(
                        conductor_log,
                        "STOP-AUTHORITY: STOP_CANDIDATE_AWAITING_OPERATOR",
                        "BLOCK",
                        f"pid {candidate['pid']}: {candidate['evidence']}",
                    )
                    _append_packet(
                        known_issues,
                        "stop-authority candidate awaiting yes/no",
                        run_stop_packet(candidate),
                    )
                    state["written"][key] = now_iso
                packets += 1

    state["last_scan_utc"] = now_iso
    state["last_scan_notes"] = notes
    if not dry_run:
        write_state(state_path, state)
    return {
        "orphan_candidates": len(orphans),
        "run_candidates": len(runs),
        "actionable": len(actionable),
        "actions": actions,
        "packets": packets,
        "notes": notes,
        "armed": armed(arm_file),
        "last_scan_utc": now_iso,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true", help="evaluate + print; write nothing, kill nothing"
    )
    parser.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)
    parser.add_argument("--state", default=None)
    parser.add_argument("--conductor-log", default=None)
    parser.add_argument("--known-issues", default=None)
    args = parser.parse_args(argv)
    summary = run_scan(
        proc_root=Path(args.proc_root),
        state_path=Path(args.state) if args.state else None,
        conductor_log=Path(args.conductor_log) if args.conductor_log else None,
        known_issues=Path(args.known_issues) if args.known_issues else None,
        dry_run=args.dry_run,
    )
    print(
        f"[stop-authority] armed={summary['armed']} orphans={summary['orphan_candidates']} "
        f"runs={summary['run_candidates']} actionable={summary['actionable']} "
        f"actions={len(summary['actions'])} packets={summary['packets']}"
    )
    for action in summary["actions"]:
        print(f"  ACTION {action['fingerprint']}: {action['outcome']}")
    for note in summary["notes"]:
        print(f"  note: {note}")
    return 2 if summary["actionable"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
