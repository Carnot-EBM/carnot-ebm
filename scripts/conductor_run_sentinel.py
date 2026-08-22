#!/usr/bin/env python3
"""In-flight run sentinel: read the validity signals live runs already write.

WHY THIS EXISTS (REQ-CONDUCTOR-SENTINEL-1/2/3). On 2026-08-22 a supervisor
A/B ran ~2.5 hours looking healthy: harness alive, 3 of 3 rows written, no
runner-log errors. Every row was stamped `llm_on_row_valid: false`. The
llama-server had died mid-run ("ggml_gallocr_reserve_n_impl: failed to
allocate CUDA0 buffer"), a stranded 3,744 MiB VRAM fragment starved later
loads, and a naive read of the rows said "the supervisor makes things worse"
— a wrong conclusion about the exact mechanism under evaluation. Only a
human reading the stamp prevented it. The unifying defect: the system WRITES
validity signals and never READS them while the run is alive. This sentinel
is the reader.

WHAT IT READS, and from where:

  * Live runs, from /proc: a python process whose cmdline names an ARC
    harness script (`scripts/arc_*.py`) and carries `--out <path>.json`.
    /proc/<pid>/cmdline and /proc/<pid>/environ are the trustworthy sources;
    a launcher's own claims are not consulted.
  * Row validity, via the SAME evaluator the post-hoc gate uses:
    `check_row()` imported from scripts/arc_llm_on_liveness_lint.py. No
    second pattern list — a duplicated list drifts narrower than its concept.
  * llama-server stderr logs, located via CARNOT_ARC_SERVER_LOG_DIR from the
    run's environ (fallback: the system temp dir), under
    carnot_llama_server_logs/. Allocation failures there are unambiguous.
  * GPU state, from nvidia-smi: per-GPU used memory versus the sum of its
    compute apps (a large gap is a stranded allocation), plus llama-server
    ownership and loaded-model checks from /proc.

WHAT IT NEVER DOES: kill anything. A false stop is worse than a slow human.
The janitor (~/.carnot/orphan-cleanup.sh) owns process reaping; this sentinel
only escalates. Escalations go to ops/conductor-log.md (tracked — journald
retention on this host is hours) and, for CRITICAL findings, to
ops/known-issues.md, deduplicated through ops/.run_sentinel_state.json.

FAIL DIRECTION, per check (fail closed means "say so", not "guess"):
  * out file unparseable + fresh mtime  -> mid-write race, skip this cycle
  * out file unparseable + stale mtime  -> finding (closed)
  * nvidia-smi cannot run               -> gpu_check_unavailable finding (closed)
  * live-pin import fails               -> pin_check_unavailable finding (closed)
  * server log absent                   -> no finding (a server may be
                                           externally managed; stated open)
  * /proc entry vanishes mid-scan       -> skip that pid (races are normal)

The state file records `last_scan_utc` on EVERY run. That is this monitor's
own receipt: a sentinel that stops running becomes visible, instead of being
the next silent-but-trusted layer.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Unambiguous llama-server failure lines. Matched case-insensitively against
# the server's own stderr — these are the exact families from the 2026-08-22
# incident log plus the abort/OOM lines the same subsystem emits. This list
# gates a CRITICAL escalation only, never a kill.
SERVER_FAILURE_PATTERNS = (
    "failed to allocate",
    "ggml_gallocr_reserve",
    "failed to create context",
    "cuda error",
    "out of memory",
    "ggml_abort",
)

# Per-GPU slack between memory.used and the sum of compute-app memory before
# the gap counts as a stranded allocation. Measured on the healthy live box
# 2026-08-22: 4 MiB (GPU 0) and 15 MiB (GPU 1) unaccounted. The incident
# fragment was 3,744 MiB. 1024 MiB sits far from both.
STRANDED_VRAM_SLACK_MIB = 1024

# LLM-on rows carrying FAIL findings in a row before escalation. One invalid
# row is a wasted cell (the harness tolerates blips by design); two in a row
# is a run burning budget on a dead tier.
CONSECUTIVE_INVALID_THRESHOLD = 2

# A finding fingerprint older than this re-arms, so a condition that persists
# for weeks re-escalates rather than staying silent forever.
STATE_REARM_DAYS = 14

# An unparseable out file younger than this is a normal whole-file-rewrite
# race (the harness rewrites the file after every row), not a finding.
MIDWRITE_FRESH_S = 60.0


def _now_utc() -> datetime:
    return datetime.now(UTC)


def load_liveness_lint():
    """Import scripts/arc_llm_on_liveness_lint.py (scripts/ is not a package)."""
    path = REPO / "scripts" / "arc_llm_on_liveness_lint.py"
    spec = importlib.util.spec_from_file_location("arc_llm_on_liveness_lint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# /proc discovery
# ---------------------------------------------------------------------------

_ARC_HARNESS_RE = re.compile(r"(?:^|/)arc_[\w.]*\.py$")


def _read_proc_file(path: Path) -> bytes | None:
    """Read a /proc file; None when the pid vanished or is unreadable."""
    try:
        return path.read_bytes()
    except OSError:
        return None


def _cmdline(proc_dir: Path) -> list[str]:
    raw = _read_proc_file(proc_dir / "cmdline")
    if not raw:
        return []
    return [a for a in raw.decode("utf-8", "replace").split("\0") if a]


def _environ(proc_dir: Path) -> dict[str, str]:
    raw = _read_proc_file(proc_dir / "environ")
    if not raw:
        return {}
    env: dict[str, str] = {}
    for chunk in raw.decode("utf-8", "replace").split("\0"):
        if "=" in chunk:
            key, _, value = chunk.partition("=")
            env[key] = value
    return env


def _ppid(proc_dir: Path) -> int | None:
    raw = _read_proc_file(proc_dir / "status")
    if not raw:
        return None
    match = re.search(rb"^PPid:\s*(\d+)", raw, re.M)
    return int(match.group(1)) if match else None


def _cgroup(proc_dir: Path) -> str:
    raw = _read_proc_file(proc_dir / "cgroup")
    return raw.decode("utf-8", "replace") if raw else ""


def _flag_value(cmdline: list[str], flag: str) -> str | None:
    """Value of `--flag value` or `--flag=value` from an argv list."""
    for i, arg in enumerate(cmdline):
        if arg == flag and i + 1 < len(cmdline):
            return cmdline[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def _proc_dirs(proc_root: Path) -> list[Path]:
    try:
        return [d for d in proc_root.iterdir() if d.name.isdigit()]
    except OSError:
        return []


def discover_live_runs(proc_root: Path = Path("/proc")) -> list[dict]:
    """Find live ARC harness runs: python + scripts/arc_*.py + --out.

    Scope, stated: this covers the ARC harness family, which by convention
    takes `--out <rows json>`. A run that writes rows some other way is out
    of scope for the row detector (the post-hoc lint still covers it at
    commit time).
    """
    runs = []
    for proc_dir in _proc_dirs(proc_root):
        cmdline = _cmdline(proc_dir)
        if not cmdline or "python" not in Path(cmdline[0]).name:
            continue
        if not any(_ARC_HARNESS_RE.search(arg) for arg in cmdline):
            continue
        out = _flag_value(cmdline, "--out")
        if not out:
            continue
        # A relative --out means relative to the RUN's cwd, not this
        # sentinel's — /proc/<pid>/cwd is the trustworthy source. On
        # failure keep the raw path; a wrong guess would be worse
        # (adversarial-review finding 4, 2026-08-22).
        if not Path(out).is_absolute():
            try:
                out = str((proc_dir / "cwd").readlink() / out)
            except OSError:
                pass
        port = _flag_value(cmdline, "--port")
        env = _environ(proc_dir)
        runs.append(
            {
                "pid": int(proc_dir.name),
                "cmdline": cmdline,
                "out_path": out,
                "port": int(port) if port and port.isdigit() else None,
                "server_log_dir": env.get("CARNOT_ARC_SERVER_LOG_DIR"),
            }
        )
    return runs


def discover_llama_servers(proc_root: Path = Path("/proc")) -> list[dict]:
    """Every live llama-server, with the facts the health checks need."""
    servers = []
    for proc_dir in _proc_dirs(proc_root):
        cmdline = _cmdline(proc_dir)
        if not cmdline or Path(cmdline[0]).name != "llama-server":
            continue
        port = _flag_value(cmdline, "--port")
        servers.append(
            {
                "pid": int(proc_dir.name),
                "cmdline": cmdline,
                "model_path": _flag_value(cmdline, "-m") or _flag_value(cmdline, "--model"),
                "port": int(port) if port and port.isdigit() else None,
                "ppid": _ppid(proc_dir),
                "cgroup": _cgroup(proc_dir),
            }
        )
    return servers


def _port_referenced_elsewhere(port: int, server_pid: int, proc_root: Path) -> bool:
    """Does any OTHER live process name this port on its cmdline or environ?

    A llama-server reparented to init is not an orphan while a live client
    still points at its port (observed live 2026-08-22: the A/B harness holds
    `--port 8995` while the server's original parent shell is long gone).
    """
    needle = str(port)
    for proc_dir in _proc_dirs(proc_root):
        if proc_dir.name == str(server_pid):
            continue
        cmdline = _cmdline(proc_dir)
        if not cmdline:
            continue
        if _flag_value(cmdline, "--port") == needle:
            return True
        if any(f":{needle}" in arg for arg in cmdline):
            return True
        env = _environ(proc_dir)
        for key, value in env.items():
            if ("LLAMA" in key.upper() or "GENERATOR" in key.upper()) and needle in value:
                return True
    return False


# ---------------------------------------------------------------------------
# Detector A: live-run row validity
# ---------------------------------------------------------------------------


def read_out_file(path: Path, now_s: float | None = None):
    """Read a harness out file. Returns (doc, reason).

    reason values: "" ok; "missing"; "midwrite" (unparseable but fresh —
    the harness rewrites the whole file after every row, so this is a
    normal race, skipped this cycle); "stale_unparseable" (unparseable and
    old — a real finding).
    """
    now_s = time.time() if now_s is None else now_s
    try:
        text = path.read_text(encoding="utf-8")
        doc = json.loads(text)
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        try:
            fresh = now_s - path.stat().st_mtime < MIDWRITE_FRESH_S
        except OSError:
            fresh = False
        return None, "midwrite" if fresh else "stale_unparseable"
    return doc, ""


def extract_rows(doc, lint) -> list:
    """Rows from a parsed out doc, in document order.

    Top-level `rows` is the harness convention; when it is absent the
    fallback is the liveness lint's OWN any-depth row walker, so a
    nested/per-cell corpus shape (the 2026-07 corpora use `cells[i].row`)
    is not invisible. Adversarial-review finding 2026-08-22: the first
    ship read top-level `rows` only — a row locator narrower than the
    lint's row concept.
    """
    if isinstance(doc, dict):
        rows = doc.get("rows")
        if isinstance(rows, list):
            return rows
    return [row for _, row in lint.walk_rows(doc)]


def evaluate_rows(rows: list, lint) -> list[dict]:
    """Findings from the ordered row list of one live run.

    Validity comes from lint.check_row (primitives, not the derived stamp).
    The streak is counted over the LLM-on subsequence: in a --both run the
    llm-off arms interleave and must not mask a dying LLM tier.
    """
    findings: list[dict] = []
    llm_on_total = 0
    invalid_total = 0
    witness_missing = 0
    streak = 0
    max_streak = 0
    example_codes: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not lint._is_row(row) or not lint._claims_llm_on(row):
            continue
        llm_on_total += 1
        row_findings = lint.check_row(row)
        codes = [f.get("code") for f in row_findings]
        if any(c in lint.FAIL_CODES for c in codes):
            invalid_total += 1
            streak += 1
            max_streak = max(max_streak, streak)
            for c in codes:
                if c in lint.FAIL_CODES and c not in example_codes:
                    example_codes.append(c)
        else:
            streak = 0
        if any(c in ("WITNESS_MISSING", "WITNESS_UNAVAILABLE") for c in codes):
            witness_missing += 1
    if max_streak >= CONSECUTIVE_INVALID_THRESHOLD:
        # Every LLM-on row invalid = the whole run measures a dead tier (the
        # 2026-08-22 incident shape) -> CRITICAL. A contained streak -> WARN.
        all_invalid = invalid_total == llm_on_total
        findings.append(
            {
                "code": "CONSECUTIVE_INVALID_LLM_ON_ROWS",
                "severity": "CRITICAL" if all_invalid else "WARN",
                "detail": (
                    f"{invalid_total}/{llm_on_total} LLM-on rows invalid "
                    f"(max streak {max_streak}; codes {example_codes[:3]})"
                ),
            }
        )
    if witness_missing:
        findings.append(
            {
                "code": "ROW_WITNESS_MISSING",
                "severity": "WARN",
                "detail": (
                    f"{witness_missing}/{llm_on_total} LLM-on rows carry no auditable "
                    "liveness witness — absent is not valid"
                ),
            }
        )
    return findings


# ---------------------------------------------------------------------------
# Detector B: llama-server stderr logs
# ---------------------------------------------------------------------------


def find_server_logs(
    log_dir: Path,
    port: int | None = None,
    max_age_h: float = 24.0,
    now_s: float | None = None,
) -> list[Path]:
    """Recent llama-server stderr logs under a CARNOT_ARC_SERVER_LOG_DIR.

    The proposer writes `<dir>/carnot_llama_server_logs/llama_server_p{port}_
    {ts}.log` (arc_executable_world_model.py). Both the nested dir and the
    dir itself are searched, so a caller may pass either level.
    """
    now_s = time.time() if now_s is None else now_s
    pattern = f"llama_server_p{port}_*.log" if port else "llama_server_p*.log"
    candidates: list[Path] = []
    for base in (log_dir / "carnot_llama_server_logs", log_dir):
        try:
            candidates.extend(base.glob(pattern))
        except OSError:
            continue
    fresh = []
    for path in candidates:
        try:
            if now_s - path.stat().st_mtime <= max_age_h * 3600:
                fresh.append(path)
        except OSError:
            continue
    return sorted(set(fresh))


def scan_server_log_text(text: str) -> list[str]:
    """Unambiguous failure lines from a server stderr log (max 3 returned)."""
    hits = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(pattern in lowered for pattern in SERVER_FAILURE_PATTERNS):
            hits.append(line.strip())
            if len(hits) >= 3:
                break
    return hits


def evaluate_server_logs(log_paths: list[Path]) -> list[dict]:
    findings = []
    for path in log_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits = scan_server_log_text(text)
        if hits:
            findings.append(
                {
                    "code": "SERVER_LOG_FAILURE",
                    "severity": "CRITICAL",
                    "detail": f"{path.name}: {hits[0][:120]}",
                    "log_path": str(path),
                }
            )
    return findings


# ---------------------------------------------------------------------------
# Detector C: GPU / resource health
# ---------------------------------------------------------------------------


def run_nvidia_smi(args: list[str]) -> str | None:
    """nvidia-smi wrapper; None on any failure (the caller reports, not guesses)."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def gpu_snapshot(runner=run_nvidia_smi) -> dict | None:
    """GPUs + compute apps, or None when nvidia-smi is unavailable."""
    gpu_csv = runner(
        ["--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"]
    )
    if gpu_csv is None:
        return None
    apps_csv = runner(
        ["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"]
    )
    if apps_csv is None:
        return None
    gpus = []
    for line in gpu_csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "used_mib": int(parts[2]),
                    "total_mib": int(parts[3]),
                }
            )
    apps = []
    for line in apps_csv.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            apps.append({"uuid": parts[0], "pid": int(parts[1]), "used_mib": int(parts[2])})
    return {"gpus": gpus, "apps": apps}


def evaluate_gpu_snapshot(snapshot: dict, slack_mib: int = STRANDED_VRAM_SLACK_MIB) -> list[dict]:
    """Stranded-VRAM findings: used memory no compute app accounts for."""
    findings = []
    for gpu in snapshot.get("gpus", []):
        accounted = sum(a["used_mib"] for a in snapshot.get("apps", []) if a["uuid"] == gpu["uuid"])
        unaccounted = gpu["used_mib"] - accounted
        if unaccounted > slack_mib:
            findings.append(
                {
                    "code": "STRANDED_VRAM",
                    "severity": "WARN",
                    "detail": (
                        f"GPU {gpu['index']}: {unaccounted} MiB used with no owning "
                        f"compute app (used {gpu['used_mib']}/{gpu['total_mib']} MiB)"
                    ),
                    "gpu_index": gpu["index"],
                }
            )
    return findings


def evaluate_llama_servers(
    servers: list[dict],
    live_pin: str | None,
    proc_root: Path = Path("/proc"),
) -> list[dict]:
    """Orphaned-server and wrong-model findings.

    Orphan = reparented to init (PPid 1) outside /system.slice/ with no other
    live process referencing its port. Reparenting alone is NOT enough: the
    live A/B's server outlives its launcher shell legitimately while the
    harness still points at its port.
    """
    findings = []
    for server in servers:
        pid = server["pid"]
        if server.get("ppid") == 1 and "/system.slice/" not in server.get("cgroup", ""):
            port = server.get("port")
            referenced = (
                _port_referenced_elsewhere(port, pid, proc_root) if port is not None else False
            )
            if not referenced:
                findings.append(
                    {
                        "code": "ORPHANED_LLAMA_SERVER",
                        "severity": "WARN",
                        "detail": (
                            f"pid {pid} (port {port}) reparented to init, no live "
                            "process references its port; VRAM has no owner"
                        ),
                        "pid": pid,
                    }
                )
        model_path = server.get("model_path") or ""
        if live_pin and model_path and live_pin not in Path(model_path).name:
            findings.append(
                {
                    "code": "WRONG_MODEL_LOADED",
                    "severity": "WARN",
                    "detail": (
                        f"pid {pid} serves {Path(model_path).name!r}, live pin is "
                        f"{live_pin!r} — the loaded model is not the intended one"
                    ),
                    "pid": pid,
                }
            )
    return findings


def load_live_pin() -> tuple[str | None, str | None]:
    """(live pin, error). The pin has ONE home; a failed import is reported,
    never silently skipped (a check that cannot run must say so)."""
    try:
        sys.path.insert(0, str(REPO / "python"))
        from carnot.agentic.arc_executable_world_model import (
            ARC_LIVE_GENERATOR_REPO_SUBSTR,
        )

        return ARC_LIVE_GENERATOR_REPO_SUBSTR, None
    except Exception as exc:  # noqa: BLE001 — any import failure is the finding
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Class D: durable, deduplicated escalation
# ---------------------------------------------------------------------------


def _fingerprint(finding: dict, scope: str) -> str:
    # None-checked, not truthiness: GPU index 0 is a real identity and
    # `or`-chaining would drop it (the falsy-zero class the QA discipline
    # names; adversarial-review finding 6, 2026-08-22).
    extra = ""
    for key in ("log_path", "gpu_index", "pid"):
        value = finding.get(key)
        if value is not None:
            extra = str(value)
            break
    return f"{finding['code']}|{scope}|{extra}"


def load_state(state_path: Path) -> dict:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(state, dict):
            state.setdefault("escalated", {})
            return state
    except (OSError, json.JSONDecodeError):
        pass
    return {"escalated": {}}


def write_state(state_path: Path, state: dict) -> None:
    """Atomic write: the state file is this sentinel's receipt, so a torn
    write must never look like a missing scan."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, state_path)


def _append_conductor_log_row(conductor_log: Path, task: str, status: str, detail: str) -> None:
    """Exact log_step row format from research_conductor.py — one table, one
    format, so operator tooling that reads the log needs no second parser."""
    timestamp = _now_utc().strftime("%Y-%m-%d %H:%M UTC")
    entry = f"| {timestamp} | {task[:50]} | {status} | {detail[:80]} |\n"
    if not conductor_log.exists():
        header = (
            "# Research Conductor Log\n\n"
            "| Timestamp | Task | Status | Details |\n"
            "|-----------|------|--------|---------|\n"
        )
        conductor_log.write_text(header + entry, encoding="utf-8")
    else:
        with open(conductor_log, "a", encoding="utf-8") as fh:
            fh.write(entry)


def _append_known_issue(known_issues: Path, finding: dict, scope: str) -> None:
    stamp = _now_utc().strftime("%Y-%m-%d")
    with open(known_issues, "a", encoding="utf-8") as fh:
        fh.write(
            f"\n## OPERATOR-ATTENTION {stamp}: run sentinel {finding['code']}\n\n"
            f"Scope: {scope}\n\n{finding['detail']}\n\n"
            f"Written by scripts/conductor_run_sentinel.py "
            f"(REQ-CONDUCTOR-SENTINEL-3). The sentinel never kills work; "
            f"triage and clear by hand.\n"
        )


def escalate(
    findings: list[tuple[str, dict]],
    *,
    conductor_log: Path,
    known_issues: Path,
    state_path: Path,
    dry_run: bool = False,
    notes: list[str] | None = None,
) -> dict:
    """Write deduplicated durable escalations; ALWAYS advance the receipt.

    `findings` is a list of (scope, finding) pairs, scope being the stable
    identity of what was scanned (an out path, a pid, a gpu index).
    `notes` are non-finding observations (mid-write skips, missing out
    files); they land in the state file per SCENARIO-CONDUCTOR-SENTINEL-1-
    MIDWRITE-RACE so a skipped run leaves a durable trace, not a stdout
    line nobody keeps.
    """
    state = load_state(state_path)
    now = _now_utc()
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    rearm_s = STATE_REARM_DAYS * 86400
    written = 0
    skipped = 0
    for scope, finding in findings:
        fp = _fingerprint(finding, scope)
        prior = state["escalated"].get(fp)
        if prior:
            try:
                prior_dt = datetime.strptime(prior, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
                if (now - prior_dt).total_seconds() < rearm_s:
                    skipped += 1
                    continue
            except ValueError:
                pass  # unparseable stamp -> re-escalate rather than stay silent
        status = "BLOCK" if finding["severity"] == "CRITICAL" else "WARN"
        if not dry_run:
            _append_conductor_log_row(
                conductor_log,
                f"OPERATOR-ATTENTION: {finding['code']}",
                status,
                f"{scope}: {finding['detail']}",
            )
            if finding["severity"] == "CRITICAL":
                _append_known_issue(known_issues, finding, scope)
            state["escalated"][fp] = now_iso
        written += 1
    state["last_scan_utc"] = now_iso
    state["last_scan_notes"] = list(notes or [])
    if not dry_run:
        write_state(state_path, state)
    return {"written": written, "deduplicated": skipped, "last_scan_utc": now_iso}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_scan(
    *,
    proc_root: Path = Path("/proc"),
    gpu_runner=run_nvidia_smi,
    pin_loader=load_live_pin,
    conductor_log: Path | None = None,
    known_issues: Path | None = None,
    state_path: Path | None = None,
    dry_run: bool = False,
    log_max_age_h: float = 24.0,
    default_log_dir: Path | None = None,
) -> dict:
    """One full scan pass. Returns a summary dict; never signals a process.

    default_log_dir: where the standalone server-log sweep looks when no
    live run names a log dir. Defaults to the system temp dir (the
    proposer's own fallback); tests pass a tmp_path so a shared /tmp on a
    busy box can never leak real findings into a hermetic test.
    """
    conductor_log = conductor_log or REPO / "ops" / "conductor-log.md"
    known_issues = known_issues or REPO / "ops" / "known-issues.md"
    state_path = state_path or REPO / "ops" / ".run_sentinel_state.json"
    default_log_dir = default_log_dir or Path(tempfile.gettempdir())
    lint = load_liveness_lint()
    findings: list[tuple[str, dict]] = []
    notes: list[str] = []

    # A + B: live runs -> rows + their server logs.
    runs = discover_live_runs(proc_root)
    seen_log_dirs: set[str] = set()
    for run in runs:
        out_path = Path(run["out_path"])
        doc, reason = read_out_file(out_path)
        scope = f"pid {run['pid']} {out_path}"
        if reason == "midwrite":
            notes.append(f"{scope}: mid-write race, skipped this cycle")
        elif reason == "missing":
            # Absent is not zero: a run whose out file never appears is a
            # run nothing can supervise — say so rather than skip silently.
            notes.append(f"{scope}: out file missing, run unsupervisable this cycle")
        elif reason == "stale_unparseable":
            findings.append(
                (
                    scope,
                    {
                        "code": "OUT_FILE_STALE_UNPARSEABLE",
                        "severity": "WARN",
                        "detail": "out file unparseable and older than the rewrite window",
                    },
                )
            )
        else:
            rows = extract_rows(doc, lint)
            if rows:
                findings.extend((scope, f) for f in evaluate_rows(rows, lint))
            else:
                notes.append(f"{scope}: no rows found in out file yet")
        log_dir = run.get("server_log_dir") or tempfile.gettempdir()
        seen_log_dirs.add(log_dir)
        logs = find_server_logs(Path(log_dir), run.get("port"), max_age_h=log_max_age_h)
        findings.extend((scope, f) for f in evaluate_server_logs(logs))

    # B, standalone sweep: recent server logs in the default temp location
    # catch a server whose run already exited (the stranded-fragment shape).
    if str(default_log_dir) not in seen_log_dirs:
        logs = find_server_logs(default_log_dir, None, max_age_h=log_max_age_h)
        findings.extend((f"log sweep {default_log_dir}", f) for f in evaluate_server_logs(logs))

    # C: GPU health.
    snapshot = gpu_snapshot(gpu_runner)
    day_bucket = _now_utc().strftime("%Y-%m-%d")
    if snapshot is None:
        findings.append(
            (
                f"host {day_bucket}",
                {
                    "code": "GPU_CHECK_UNAVAILABLE",
                    "severity": "WARN",
                    "detail": "nvidia-smi could not run; GPU health is UNKNOWN, not clean",
                },
            )
        )
    else:
        findings.extend(("host", f) for f in evaluate_gpu_snapshot(snapshot))

    servers = discover_llama_servers(proc_root)
    live_pin, pin_error = pin_loader()
    if pin_error:
        findings.append(
            (
                f"host {day_bucket}",
                {
                    "code": "PIN_CHECK_UNAVAILABLE",
                    "severity": "WARN",
                    "detail": f"live-pin import failed ({pin_error[:60]}); "
                    "wrong-model check is UNKNOWN, not clean",
                },
            )
        )
    findings.extend(("host", f) for f in evaluate_llama_servers(servers, live_pin, proc_root))

    summary = escalate(
        findings,
        conductor_log=conductor_log,
        known_issues=known_issues,
        state_path=state_path,
        dry_run=dry_run,
        notes=notes,
    )
    summary.update(
        {
            "runs_seen": len(runs),
            "servers_seen": len(servers),
            "findings": [{"scope": s, **f} for s, f in findings],
            "notes": notes,
        }
    )
    return summary


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print findings; write nothing")
    parser.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)
    parser.add_argument("--state", default=None, help="state/receipt JSON path")
    parser.add_argument("--conductor-log", default=None)
    parser.add_argument("--known-issues", default=None)
    parser.add_argument("--log-age-hours", type=float, default=24.0)
    args = parser.parse_args(argv)
    summary = run_scan(
        proc_root=Path(args.proc_root),
        conductor_log=Path(args.conductor_log) if args.conductor_log else None,
        known_issues=Path(args.known_issues) if args.known_issues else None,
        state_path=Path(args.state) if args.state else None,
        dry_run=args.dry_run,
        log_max_age_h=args.log_age_hours,
    )
    print(
        f"[run-sentinel] runs={summary['runs_seen']} servers={summary['servers_seen']} "
        f"findings={len(summary['findings'])} escalated={summary['written']} "
        f"deduplicated={summary['deduplicated']}"
    )
    for finding in summary["findings"]:
        print(
            f"  {finding['severity']:8} {finding['code']}: {finding['scope']}: {finding['detail']}"
        )
    for note in summary["notes"]:
        print(f"  note: {note}")
    # Exit 0 clean, 2 findings (escalated, not an error), never a kill path.
    return 2 if summary["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
