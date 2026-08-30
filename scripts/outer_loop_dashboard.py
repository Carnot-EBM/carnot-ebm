#!/usr/bin/env python3
"""One-screen status of everything the outer loop is responsible for.

WHY A SCRIPT AND NOT PROSE (2026-08-30). The outer loop reports status hourly, and the report
drifted from a compact scannable block into paragraphs -- which the operator noticed and asked to
have reverted. A script fixes that permanently: the same fields in the same order every hour, so
a reader compares two reports at a glance instead of re-reading two essays.

It also removes a whole class of error this session kept hitting. Elapsed time is COMPUTED here,
never estimated, because an agent with no internal clock reliably reports days-old events as
months old. Process liveness is read from /proc rather than assumed. Nothing here is recalled.

Read-only by construction: it opens files and asks the OS about processes. It writes nothing.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(*args: str) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=15, cwd=str(REPO))
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def days_since(stamp: str, now: datetime | None = None) -> int | None:
    """Whole days between an ISO date and now. The antidote to "a while back".

    Returns None for anything unparseable rather than guessing, because a wrong age is worse
    than an absent one -- it is the wrong age that gets repeated as fact.
    """
    now = now or datetime.now(UTC)
    try:
        when = datetime.fromisoformat(stamp.strip()[:19])
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return (now - when).days


def pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def conductor_state() -> dict:
    pid_s = _run("systemctl", "--user", "show", "carnot-conductor", "-p", "MainPID", "--value")
    active = _run("systemctl", "--user", "is-active", "carnot-conductor")
    pid = int(pid_s) if pid_s.isdigit() and pid_s != "0" else None
    children = 0
    if pid and pid_alive(pid):
        children = len([c for c in _run("pgrep", "-P", str(pid)).split() if c.strip()])
    roadmap = REPO / "research-roadmap.yaml"
    milestone = "?"
    if roadmap.exists():
        m = re.search(r"^milestone:\s*(\S+)", roadmap.read_text(), re.M)
        milestone = m.group(1) if m else "?"
    return {"active": active, "pid": pid, "children": children, "milestone": milestone}


def outcome_mix(day: str) -> dict[str, int]:
    """Conductor outcomes for one YYYY-MM-DD, counted from the log rather than recalled."""
    log = REPO / "ops" / "conductor-log.md"
    if not log.exists():
        return {}
    counts: dict[str, int] = {}
    for line in log.read_text(errors="replace").splitlines():
        if not line.startswith(f"| {day}"):
            continue
        m = re.search(r"\|\s(OK|FAIL|FLAGGED|GATE_BLOCK|BLOCK|WARN)\s\|", line)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    return counts


def flag_states(names: list[str]) -> dict[str, str]:
    """Flag-ledger states. An `unevaluated` flag is shipped-but-unproven, which is the
    single most useful thing to keep in view: it is work that cannot pay off yet."""
    path = REPO / "ops" / "arc_flag_ledger.yaml"
    if not path.exists():
        return {}
    text = path.read_text(errors="replace")
    out: dict[str, str] = {}
    for name in names:
        m = re.search(rf"^\s+{re.escape(name)}:\s*$(.*?)(?=^\s+\w+:\s*$)", text, re.M | re.S)
        block = m.group(1) if m else ""
        s = re.search(r"state:\s*(\S+)", block)
        out[name] = s.group(1) if s else "?"
    return out


def gpu_rows() -> list[str]:
    out = _run(
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader",
    )
    return [r.strip() for r in out.splitlines() if r.strip()]


def render(jobs: list[tuple[str, int, Path | None]] | None = None) -> str:
    now = datetime.now(UTC)
    L: list[str] = []
    L.append(f"OUTER LOOP  {now:%Y-%m-%d %H:%M:%S}Z  ({now:%A})")
    L.append("=" * 64)

    c = conductor_state()
    alive = "alive" if c["pid"] and pid_alive(c["pid"]) else "DOWN"
    L.append(
        f"conductor   {c['active']}/{alive}  pid {c['pid']}  milestone {c['milestone']}  "
        f"children {c['children']}"
    )
    mix = outcome_mix(f"{now:%Y-%m-%d}")
    if mix:
        L.append("  today     " + "  ".join(f"{k}={v}" for k, v in sorted(mix.items())))

    rows = gpu_rows()
    if rows:
        L.append("gpu         " + " | ".join(rows))

    for name, pid, receipt in jobs or []:
        if pid_alive(pid):
            L.append(f"job         {name}: alive pid {pid}")
        elif receipt and receipt.exists():
            try:
                r = json.loads(receipt.read_text())
                L.append(
                    f"job         {name}: KILLED by {r.get('signal_name')} "
                    f"after {r.get('elapsed_s')}s -- progress {r.get('progress')}"
                )
            except (OSError, json.JSONDecodeError):
                L.append(f"job         {name}: gone; receipt unreadable at {receipt}")
        else:
            L.append(
                f"job         {name}: gone, NO receipt "
                f"(SIGKILL, OOM, or kernel event -- REQ-INFRA-6830)"
            )

    flags = flag_states(
        [
            "CARNOT_ARC_INDUCE_TOOL_LOOP",
            "CARNOT_ARC_INDUCE_CANDIDATE_TOOLS",
            "CARNOT_ARC_SUPERVISOR_TOOL_ARM",
            "CARNOT_ARC_TRAJECTORY_SUPERVISOR",
        ]
    )
    if flags:
        unproven = [k for k, v in flags.items() if v == "unevaluated"]
        L.append(f"flags       {len(unproven)}/{len(flags)} shipped-but-unevaluated")
        for k in unproven:
            L.append(f"              {k}")

    head = _run("git", "log", "-1", "--format=%h %ad %s", "--date=format:%m-%d %H:%M")
    L.append(f"head        {head[:70]}")
    return "\n".join(L)


def main(argv: list[str]) -> int:
    jobs: list[tuple[str, int, Path | None]] = []
    for spec in argv[1:]:
        # name:pid[:receipt]
        parts = spec.split(":")
        if len(parts) >= 2 and parts[1].isdigit():
            jobs.append((parts[0], int(parts[1]), Path(parts[2]) if len(parts) > 2 else None))
    print(render(jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
