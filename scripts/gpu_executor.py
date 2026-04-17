#!/usr/bin/env python3
"""GPU Executor — runs experiments that the conductor's subagent couldn't finish.

Companion to ``scripts/research_conductor.py``.

**The problem this solves:**
    The conductor's subagent is bounded to ~45-60 min per experiment. Experiments
    that legitimately need longer (live GPU benchmarks on hundreds of questions,
    training runs, adversarial sweeps) get committed as "scaffolding-only"
    deliverables with ``honest_verdict == 'live_benchmark_needs_human_triggered_run'``.
    Those scripts exist and are tested, but nobody runs them.

    This executor is that "nobody". It picks up scaffolding-only deliverables,
    runs the corresponding script, lets the script overwrite the scaffolding
    with live results.

**Deadlock-proofing by design:**
    1. *File-path ownership.* This executor only writes to paths the conductor
       doesn't write to mid-run.  Specifically, it runs experiment scripts
       directly; those scripts write their own canonical deliverable JSONs,
       overwriting the scaffolding in place.  The conductor's subagent writes
       scaffolding BEFORE this executor picks up the work; there is no
       temporal overlap.
    2. *GPU resource guard.* Before each experiment, check_dual_gpu_health()
       verifies GPU0 temp < 80C and GPU1 is either idle or actively computing
       (not zombie). On unhealthy state: sleep 60s, retry up to 3 times, then
       skip to the next experiment. No indefinite waits.
    3. *Single git writer.* This executor does NOT push. It commits locally
       with a [gpu-executor] tag; the conductor's next iteration picks up the
       new commit and pushes it via its normal git_commit_and_push path.
    4. *Process lock.* A pid lock at /tmp/carnot_gpu_executor.lock prevents
       two executors from running concurrently on the same machine.

**Usage:**
    # One-shot: run any scaffolding-only deliverables in the current milestone
    JAX_PLATFORMS=cpu .venv/bin/python scripts/gpu_executor.py

    # Continuous: wake every 5 min and check for new scaffolding
    JAX_PLATFORMS=cpu .venv/bin/python scripts/gpu_executor.py --loop

    # Dry run: print what would be executed, don't run
    JAX_PLATFORMS=cpu .venv/bin/python scripts/gpu_executor.py --dry-run

**Exit codes:**
    0 — at least one experiment was attempted (succeeded or cleanly failed)
    1 — another executor instance holds the lock
    2 — no scaffolding-only experiments to run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = Path("/tmp/carnot_gpu_executor.lock")
POLL_INTERVAL_S = 300  # 5 min between scans in --loop mode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [gpu-executor] %(levelname)s %(message)s",
)
logger = logging.getLogger("gpu_executor")


# ---------------------------------------------------------------------------
# Process lock
# ---------------------------------------------------------------------------


def acquire_lock() -> bool:
    """Return True if we acquired the lock; False if another instance has it.

    Uses O_EXCL so two concurrent acquire_lock() calls can never both succeed.
    """
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing_pid = _read_lock_pid()
        if existing_pid and not _pid_alive(existing_pid):
            logger.warning("Stale lock from pid %s — clearing", existing_pid)
            LOCK_PATH.unlink(missing_ok=True)
            return acquire_lock()
        return False
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)
    return True


def release_lock() -> None:
    LOCK_PATH.unlink(missing_ok=True)


def _read_lock_pid() -> int | None:
    try:
        return int(LOCK_PATH.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# Experiment discovery
# ---------------------------------------------------------------------------


SCAFFOLDING_VERDICTS = {
    "live_benchmark_needs_human_triggered_run",
    "scaffolding_only",
}


def find_scaffolding_experiments() -> list[dict[str, Any]]:
    """Scan the current milestone for experiments with scaffolding-only deliverables.

    Returns a list of dicts with keys: id, script_path, deliverable_path.
    """
    roadmap_path = PROJECT_ROOT / "research-roadmap.yaml"
    if not roadmap_path.exists():
        return []
    roadmap = yaml.safe_load(roadmap_path.read_text())
    candidates = []
    for task in roadmap.get("tasks", []):
        deliverable = PROJECT_ROOT / task["deliverable"]
        if not deliverable.exists():
            continue
        try:
            data = json.loads(deliverable.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        verdict = data.get("honest_verdict") or data.get("status")
        if verdict not in SCAFFOLDING_VERDICTS and data.get("status") != "scaffolding_only":
            continue
        script_path = data.get("scaffolding_complete", {}).get("script_path")
        if not script_path:
            script_path = f"scripts/{task['id'].replace('-', '_')}.py"
        script_full = PROJECT_ROOT / script_path
        if not script_full.exists():
            logger.warning("Skip %s: script %s not found", task["id"], script_path)
            continue
        candidates.append({
            "id": task["id"],
            "title": task.get("title", task["id"]),
            "script_path": str(script_full),
            "deliverable_path": str(deliverable),
        })
    return candidates


# ---------------------------------------------------------------------------
# GPU health guard
# ---------------------------------------------------------------------------


def wait_for_healthy_gpu(max_retries: int = 3, retry_delay_s: int = 60) -> bool:
    """Return True if GPU is healthy enough to start an experiment.

    Healthy means: GPU0 temperature < 80C AND GPU1 is either idle or computing
    (not in the RETRO-025 zombie pattern where VRAM is allocated but util is 0).

    Bounded retry: up to max_retries*retry_delay_s total wait. Never blocks
    indefinitely — this prevents deadlock on persistently-bad GPU state.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "python"))
    from carnot.pipeline.dual_gpu_health import check_dual_gpu_health

    for attempt in range(max_retries):
        health = check_dual_gpu_health()
        if not health.temperature_warning and not health.gpu1_is_zombie:
            return True
        reason = []
        if health.temperature_warning:
            reason.append(f"GPU0 temp {health.gpu0_temp_c}C")
        if health.gpu1_is_zombie:
            reason.append(
                f"GPU1 zombie ({health.gpu1_vram_mb}MB / {health.gpu1_util_pct}%)"
            )
        logger.warning(
            "GPU unhealthy (attempt %d/%d): %s — sleeping %ds",
            attempt + 1, max_retries, "; ".join(reason), retry_delay_s,
        )
        time.sleep(retry_delay_s)
    return False


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------


def run_experiment(exp: dict[str, Any]) -> bool:
    """Run the experiment script directly. Script writes its own deliverable.

    No turn limits, no wall-clock timeout — this is the point of the executor.
    The experiment's own ExperimentTimeoutWatchdog (if present) is the only
    time bound. Long-running scripts (e.g. 3-hour training) run to completion.
    """
    logger.info("Running %s: %s", exp["id"], exp["title"])
    logger.info("  script: %s", exp["script_path"])
    logger.info("  deliverable: %s", exp["deliverable_path"])

    start = time.monotonic()
    env = {**os.environ}
    env.setdefault("CARNOT_FORCE_LIVE", "1")
    env.setdefault("JAX_PLATFORMS", "cpu")

    result = subprocess.run(
        [".venv/bin/python", exp["script_path"]],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    elapsed = time.monotonic() - start

    if result.returncode != 0:
        logger.error(
            "%s failed (exit %d, %.1f min)",
            exp["id"], result.returncode, elapsed / 60,
        )
        return False

    logger.info("%s completed in %.1f min", exp["id"], elapsed / 60)

    deliverable = Path(exp["deliverable_path"])
    if not deliverable.exists():
        logger.error("%s: script succeeded but deliverable not written", exp["id"])
        return False

    try:
        data = json.loads(deliverable.read_text())
        verdict = data.get("honest_verdict") or data.get("status")
        if verdict in SCAFFOLDING_VERDICTS:
            logger.warning(
                "%s: deliverable still marked %r — script may not have upgraded to live",
                exp["id"], verdict,
            )
    except (json.JSONDecodeError, OSError):
        logger.warning("%s: could not re-read deliverable", exp["id"])

    _commit_locally(exp)
    return True


def _commit_locally(exp: dict[str, Any]) -> None:
    """Commit this experiment's live result locally. Do not push."""
    msg = (
        f"[gpu-executor] Exp {exp['id']}: live results from long-running executor\n"
        f"\nScript: {exp['script_path']}\n"
        f"Deliverable: {exp['deliverable_path']}\n"
        "\nWritten by scripts/gpu_executor.py; conductor will push on next iteration.\n"
    )
    subprocess.run(["git", "add", exp["deliverable_path"]], cwd=str(PROJECT_ROOT))
    result = subprocess.run(
        ["git", "commit", "-m", msg, "--", exp["deliverable_path"]],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True,
    )
    if result.returncode == 0:
        logger.info("Committed %s (not pushed)", exp["id"])
    else:
        logger.warning("Commit failed for %s: %s", exp["id"], result.stderr[:200])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def one_pass(dry_run: bool = False) -> int:
    """Run a single pass over all scaffolding-only experiments. Return count attempted."""
    candidates = find_scaffolding_experiments()
    if not candidates:
        logger.info("No scaffolding-only experiments found in current milestone")
        return 0

    logger.info("Found %d scaffolding-only experiments:", len(candidates))
    for exp in candidates:
        logger.info("  %s — %s", exp["id"], exp["title"])

    if dry_run:
        return len(candidates)

    if not wait_for_healthy_gpu():
        logger.error("GPU not healthy; aborting this pass")
        return 0

    attempted = 0
    for exp in candidates:
        if not wait_for_healthy_gpu():
            logger.warning("GPU unhealthy mid-pass; stopping early")
            break
        run_experiment(exp)
        attempted += 1
    return attempted


def main() -> int:
    parser = argparse.ArgumentParser(description="Carnot GPU Executor")
    parser.add_argument("--loop", action="store_true",
                        help="Scan every 5 min for new scaffolding")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be run; do not execute")
    args = parser.parse_args()

    if not acquire_lock():
        logger.error("Another gpu_executor instance holds the lock — exiting")
        return 1

    def _cleanup(*_: Any) -> None:
        release_lock()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        while True:
            count = one_pass(dry_run=args.dry_run)
            if not args.loop:
                return 0 if count > 0 else 2
            logger.info("Pass complete; sleeping %ds", POLL_INTERVAL_S)
            time.sleep(POLL_INTERVAL_S)
    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
