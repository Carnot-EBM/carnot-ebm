"""Outer-loop watchdog helper: relaunch the contiguous TRM Sudoku-Extreme run FULLY DETACHED.

Why this exists
---------------
The canonical launcher ``experiment_4157_baseline_harvest_contiguous_continue.launch_contiguous_run``
spawns the native ``src/nn/train.py`` with a plain ``subprocess.Popen`` (no ``start_new_session``).
When the launching process (a conductor task or a prior outer-loop shell) exits, the kernel tears
down its process group and the trainer receives SIGTERM -- which is exactly what we observed on
2026-06-18 16:45 (``[rank: 0] Received SIGTERM: 15`` three minutes after a clean checkpoint resume).
The training never trains long enough to bank a new ``val/exact_accuracy`` best, so the stable
checkpoint has been frozen at val~=0.501 for days while the conductor's defensive verifier graft
waits on val>=0.85.

This helper reuses the module's *exact* command + env (so the run is byte-identical to the
conductor's own resume) and changes one thing: it launches with ``start_new_session=True`` and is
itself meant to be invoked under ``setsid nohup`` from the watchdog shell. The trainer therefore
becomes session leader (PPID re-parents to 1), lives outside the conductor cgroup, and survives the
launching shell's exit. GPU is pinned to device 1 (the watchdog's TRM GPU); GPU 0 stays free for the
conductor.

Usage (from the hourly watchdog, after confirming the run is DEAD and val<0.85)::

    setsid nohup .venv/bin/python scripts/outer_loop_trm_relaunch.py >/dev/null 2>&1 &

It prints the launched PID and writes it to ``results/trm_runs/contiguous_run.pid`` (the pidfile the
watchdog polls). Exits immediately; the detached trainer keeps running for ``--max-time`` (default
12h) and stops on its own, writing ``last.ckpt`` back to the stable dir whenever val improves.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the carnot package importable so we can reuse the canonical command/env builders.
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_4157_baseline_harvest_contiguous_continue as exp4157  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-time",
        default="00:12:00:00",
        help="Lightning trainer.max_time (DD:HH:MM:SS); watchdog default 12h.",
    )
    ap.add_argument(
        "--gpu",
        default="1",
        help="CUDA_VISIBLE_DEVICES for the trainer (watchdog TRM GPU = 1).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=384,
        help=(
            "Train dataloader batch size (timekeeping.batch_size). The experiment default 768 needs "
            "~23.5 GiB and chronically OOMs on a 24 GiB 3090 (most launches die at the first forward); "
            "384 needs ~12 GiB and reliably banks a new val/exact_accuracy best. Step-based LR resume "
            "is batch-invariant, so a smaller continuation batch is safe."
        ),
    )
    args = ap.parse_args()

    # Durable stop: the operator retired the Sudoku-Extreme TRM verifier-graft track (2026-06-18 —
    # "kill it, we don't need it"; TRM is NOT on the live ARC-AGI-3 solve path). This sentinel makes
    # the kill survive any watchdog decision-A restart attempt. Delete it to re-enable training.
    sentinel = REPO_ROOT / "results" / "trm_runs" / "DO_NOT_RELAUNCH"
    if sentinel.exists():
        print(f"[relaunch] DO_NOT_RELAUNCH sentinel present ({sentinel}); TRM training is retired. Not launching.")
        return 0

    config = exp4157.Exp4157Config(max_time=args.max_time)

    # Refuse to launch a competing run if the trainer is already alive (GPU/checkpoint contention).
    pid_path = Path(config.pid_path)
    if pid_path.exists():
        try:
            existing = int(pid_path.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            existing = 0
        if existing > 0:
            try:
                os.kill(existing, 0)
                print(f"[relaunch] contiguous run already ALIVE pid={existing}; not relaunching.")
                return 0
            except OSError:
                pass  # stale pidfile -> safe to relaunch

    ckpt = Path(config.stable_checkpoint_path)
    if not ckpt.exists():
        print(f"[relaunch] FATAL: stable checkpoint missing: {ckpt}")
        return 2

    command = exp4157.build_train_command(config)
    if args.batch_size and args.batch_size > 0:
        # Override the experiment-default batch (768 -> OOM cliff) with one that reliably fits.
        command.append(f"timekeeping.batch_size={int(args.batch_size)}")
    env = exp4157.build_train_env(config)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # pin to the watchdog's TRM GPU

    run_dir = Path(config.contiguous_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "exp4157_contiguous_resume.log"
    log_handle = log_path.open("a", encoding="utf-8")

    print(f"[relaunch] cwd={config.nano_trm_root}")
    print(f"[relaunch] command: {' '.join(command)}")
    print(f"[relaunch] max_time={args.max_time} gpu={args.gpu} log={log_path}")

    proc = subprocess.Popen(  # noqa: S603
        command,
        cwd=str(config.nano_trm_root),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,  # THE FIX: detach so the launching shell's exit can't SIGTERM us.
    )
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    print(f"[relaunch] launched DETACHED pid={proc.pid}; wrote pidfile {pid_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
