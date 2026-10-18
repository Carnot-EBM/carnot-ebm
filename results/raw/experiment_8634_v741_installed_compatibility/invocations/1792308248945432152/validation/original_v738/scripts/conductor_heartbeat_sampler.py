#!/usr/bin/env python3
"""Sample the conductor's kernel state once a minute (REQ-CONDUCTOR-WCHAN-1).

WHY THIS EXISTS (2026-09-03). The conductor twice went ~60 minutes with no log
line and no heartbeat, then recovered on its own. The recorded remedy —
`py-spy dump` during the gap — turned out to be unexecutable: py-spy needs
ptrace privileges this environment does not have, and nothing else was
recording, so both gaps closed with "no cause named".

WHAT THIS RECORDS, all readable without any privilege:

- `/proc/<pid>/wchan`   — the kernel function the process is blocked in
- `/proc/<pid>/status`  — the `State:` line (S sleeping, D disk wait, R running)
- a `pgrep -P` child count

One line per minute is enough to tell "blocked in select() waiting on a silent
child that existed the whole hour" from "sleeping with zero children", which is
exactly the distinction the two gap post-mortems could not make: state was only
ever observed AFTER recovery.

WHERE SAMPLES GO. `~/.carnot/heartbeat_samples/YYYYMMDD.jsonl` — deliberately
OUTSIDE the repo. A tracked, ever-growing sample file would churn git status
and be swept into conductor checkpoint commits; these are rolling diagnostics,
not the research record. Old daily files are cheap to delete.

WIRED as the systemd user timer `carnot-heartbeat-sampler.timer` running
`--once` per minute, so it keeps sampling regardless of which sessions exist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

SAMPLES_DIR = Path.home() / ".carnot" / "heartbeat_samples"


def conductor_pid() -> int | None:
    """The conductor's MainPID per systemd, or None when it is not running."""
    try:
        out = subprocess.run(
            ["systemctl", "--user", "show", "carnot-conductor", "-p", "MainPID", "--value"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return int(out) if out.isdigit() and out != "0" else None


def child_count(pid: int) -> int | None:
    """Direct children per pgrep -P. None when pgrep itself failed."""
    try:
        proc = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    # pgrep exits 1 for "no matches", which is a real answer (zero children).
    if proc.returncode not in (0, 1):
        return None
    return len([line for line in proc.stdout.split() if line.strip()])


def take_sample(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
    count_children=child_count,
) -> dict:
    """One (timestamp, wchan, state, children) record. Never raises.

    A vanished pid or unreadable proc entry is recorded as an error marker,
    not skipped — a gap in the sample file would be exactly the ambiguity
    this sampler exists to remove.
    """
    sample: dict = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pid": pid,
    }
    try:
        sample["wchan"] = (proc_root / str(pid) / "wchan").read_text(errors="replace").strip()
    except OSError as exc:
        sample["wchan"] = f"unreadable:{type(exc).__name__}"
    try:
        status = (proc_root / str(pid) / "status").read_text(errors="replace")
        state = next(
            (
                line.split(":", 1)[1].strip()
                for line in status.splitlines()
                if line.startswith("State:")
            ),
            None,
        )
        sample["state"] = state if state is not None else "no_state_line"
    except OSError as exc:
        sample["state"] = f"unreadable:{type(exc).__name__}"
    children = count_children(pid)
    sample["children"] = children if children is not None else "pgrep_failed"
    return sample


def append_sample(sample: dict, samples_dir: Path = SAMPLES_DIR) -> Path:
    samples_dir.mkdir(parents=True, exist_ok=True)
    path = samples_dir / f"{datetime.now(UTC):%Y%m%d}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(sample, sort_keys=True) + "\n")
    return path


def run_once(samples_dir: Path = SAMPLES_DIR) -> int:
    pid = conductor_pid()
    if pid is None:
        # Still a record: "the conductor was down at this minute" is data too.
        append_sample(
            {
                "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "pid": None,
                "note": "conductor not running per systemd",
            },
            samples_dir,
        )
        return 0
    append_sample(take_sample(pid), samples_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="take one sample and exit")
    parser.add_argument("--loop", action="store_true", help="sample every --interval seconds")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    args = parser.parse_args(argv)
    if args.loop:
        while True:  # pragma: no cover - the timer path is --once; loop is manual use
            run_once(args.samples_dir)
            time.sleep(max(1, args.interval))
    return run_once(args.samples_dir)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
