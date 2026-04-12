#!/usr/bin/env python3
"""GPU resource monitor for Carnot research.

Monitors GPU utilization, detects zombie processes, and suggests
optimizations. Designed to run as a background loop via the conductor
or manually via CLI.

Usage:
    # One-shot status report
    python scripts/gpu_monitor.py

    # Continuous monitoring (every 30s)
    python scripts/gpu_monitor.py --loop --interval 30

    # Kill zombie GPU processes (processes idle >10min using >1GB VRAM)
    python scripts/gpu_monitor.py --kill-zombies

    # JSON output for programmatic use
    python scripts/gpu_monitor.py --json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("gpu-monitor")

# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class GPUInfo:
    """Snapshot of a single GPU's state."""
    index: int
    name: str
    total_mb: int
    used_mb: int
    free_mb: int
    utilization_pct: int
    temperature_c: int


@dataclass
class GPUProcess:
    """A process using GPU memory."""
    pid: int
    gpu_index: int
    used_mb: int
    command: str
    cpu_time_seconds: float = 0.0
    wall_time_seconds: float = 0.0
    is_zombie: bool = False


@dataclass
class MonitorReport:
    """Full GPU monitoring report."""
    timestamp: str
    gpus: list[GPUInfo] = field(default_factory=list)
    processes: list[GPUProcess] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    total_gpu_memory_mb: int = 0
    total_used_mb: int = 0
    total_free_mb: int = 0


# ── nvidia-smi Queries ───────────────────────────────────────────────────────


def _run_smi(query: str, fmt: str = "csv,noheader,nounits") -> list[str]:
    """Run an nvidia-smi query and return lines of output."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", f"--format={fmt}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _run_smi_compute(query: str, fmt: str = "csv,noheader,nounits") -> list[str]:
    """Run an nvidia-smi compute-apps query."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-compute-apps={query}", f"--format={fmt}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def get_gpu_info() -> list[GPUInfo]:
    """Query all GPUs and return their current state."""
    lines = _run_smi("index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu")
    gpus = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            gpus.append(GPUInfo(
                index=int(parts[0]),
                name=parts[1],
                total_mb=int(parts[2]),
                used_mb=int(parts[3]),
                free_mb=int(parts[4]),
                utilization_pct=int(parts[5]),
                temperature_c=int(parts[6]),
            ))
    return gpus


def get_gpu_processes() -> list[GPUProcess]:
    """Query all processes using GPU memory."""
    lines = _run_smi_compute("pid,gpu_uuid,used_memory,name")
    # Map GPU UUIDs to indices
    uuid_lines = _run_smi("index,gpu_uuid")
    uuid_to_idx: dict[str, int] = {}
    for line in uuid_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            uuid_to_idx[parts[1]] = int(parts[0])

    processes: list[GPUProcess] = []
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            pid = int(parts[0])
            gpu_idx = uuid_to_idx.get(parts[1], -1)
            used_mb = int(parts[2])
            command = parts[3]

            # Get process timing from /proc
            cpu_time = 0.0
            wall_time = 0.0
            try:
                stat = Path(f"/proc/{pid}/stat").read_text().split()
                # Fields 13 (utime) and 14 (stime) in clock ticks
                utime = int(stat[13])
                stime = int(stat[14])
                cpu_time = (utime + stime) / 100.0  # Assuming 100 Hz
                # Field 21 (starttime) in clock ticks since boot
                starttime = int(stat[21])
                uptime = float(Path("/proc/uptime").read_text().split()[0])
                wall_time = uptime - (starttime / 100.0)
            except (FileNotFoundError, IndexError, ValueError):
                pass

            processes.append(GPUProcess(
                pid=pid,
                gpu_index=gpu_idx,
                used_mb=used_mb,
                command=command,
                cpu_time_seconds=round(cpu_time, 1),
                wall_time_seconds=round(wall_time, 1),
            ))

    return processes


# ── Analysis ─────────────────────────────────────────────────────────────────


def detect_zombies(processes: list[GPUProcess], idle_threshold_min: float = 10.0,
                   min_vram_mb: int = 1024) -> list[GPUProcess]:
    """Detect GPU processes that appear idle but hold significant VRAM.

    A process is considered a zombie if:
    - It uses >= min_vram_mb of GPU memory
    - Its CPU time / wall time ratio is < 1% (barely using CPU)
    - It has been running for > idle_threshold_min minutes
    """
    zombies = []
    for proc in processes:
        if proc.used_mb < min_vram_mb:
            continue
        if proc.wall_time_seconds < idle_threshold_min * 60:
            continue
        # Check CPU/wall ratio — zombie if barely using CPU
        if proc.wall_time_seconds > 0:
            cpu_ratio = proc.cpu_time_seconds / proc.wall_time_seconds
            if cpu_ratio < 0.01:  # Less than 1% CPU usage
                proc.is_zombie = True
                zombies.append(proc)
    return zombies


def generate_suggestions(gpus: list[GPUInfo], processes: list[GPUProcess]) -> list[str]:
    """Generate optimization suggestions based on current GPU state."""
    suggestions = []

    # Check if models could run in parallel
    used_gpus = set()
    free_gpus = set()
    for gpu in gpus:
        if gpu.used_mb > 1024:
            used_gpus.add(gpu.index)
        else:
            free_gpus.add(gpu.index)

    if len(used_gpus) == 1 and len(free_gpus) >= 1:
        suggestions.append(
            f"Only GPU {list(used_gpus)[0]} is in use. GPU {list(free_gpus)[0]} is idle. "
            "Consider running a second model in parallel for 2x throughput."
        )

    if len(used_gpus) == 0 and len(gpus) >= 2:
        suggestions.append(
            "Both GPUs are idle. Use DualGPURunner to run Qwen on GPU 0 "
            "and Gemma on GPU 1 simultaneously."
        )

    # Check for memory waste
    for gpu in gpus:
        if gpu.used_mb > 0 and gpu.utilization_pct == 0:
            suggestions.append(
                f"GPU {gpu.index} has {gpu.used_mb}MB allocated but 0% utilization. "
                "Process may be idle — consider freeing memory."
            )

    # Check for large single-GPU processes that could be split
    for proc in processes:
        if proc.used_mb > 20000:  # >20GB on one GPU
            suggestions.append(
                f"PID {proc.pid} uses {proc.used_mb}MB on GPU {proc.gpu_index}. "
                "Consider device_map='auto' to split across both GPUs."
            )

    # Check for batching opportunity
    single_inference = [p for p in processes if "python" in p.command.lower()
                        and p.used_mb > 500]
    if len(single_inference) == 1:
        suggestions.append(
            "Single Python inference process detected. Batch multiple questions "
            "per forward pass (8-16) for 4-8x throughput improvement."
        )

    return suggestions


# ── Report Generation ────────────────────────────────────────────────────────


def generate_report() -> MonitorReport:
    """Generate a complete GPU monitoring report."""
    gpus = get_gpu_info()
    processes = get_gpu_processes()
    zombies = detect_zombies(processes)

    warnings = []
    for z in zombies:
        warnings.append(
            f"ZOMBIE: PID {z.pid} on GPU {z.gpu_index} using {z.used_mb}MB "
            f"VRAM but only {z.cpu_time_seconds}s CPU in {z.wall_time_seconds:.0f}s "
            f"wall time ({z.command})"
        )

    suggestions = generate_suggestions(gpus, processes)

    return MonitorReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        gpus=gpus,
        processes=processes,
        warnings=warnings,
        suggestions=suggestions,
        total_gpu_memory_mb=sum(g.total_mb for g in gpus),
        total_used_mb=sum(g.used_mb for g in gpus),
        total_free_mb=sum(g.free_mb for g in gpus),
    )


def format_report(report: MonitorReport) -> str:
    """Format a report as human-readable text."""
    lines = [f"GPU Monitor Report — {report.timestamp}", "=" * 60]

    # GPU summary
    for gpu in report.gpus:
        pct_used = (gpu.used_mb / gpu.total_mb * 100) if gpu.total_mb > 0 else 0
        lines.append(
            f"  GPU {gpu.index}: {gpu.name} — "
            f"{gpu.used_mb}/{gpu.total_mb}MB ({pct_used:.0f}%) — "
            f"Util: {gpu.utilization_pct}% — Temp: {gpu.temperature_c}C"
        )

    # Processes
    if report.processes:
        lines.append(f"\nProcesses ({len(report.processes)}):")
        for proc in report.processes:
            zombie = " [ZOMBIE]" if proc.is_zombie else ""
            lines.append(
                f"  PID {proc.pid} on GPU {proc.gpu_index}: "
                f"{proc.used_mb}MB — CPU: {proc.cpu_time_seconds}s / "
                f"Wall: {proc.wall_time_seconds:.0f}s — "
                f"{proc.command[:50]}{zombie}"
            )

    # Warnings
    if report.warnings:
        lines.append(f"\nWarnings ({len(report.warnings)}):")
        for w in report.warnings:
            lines.append(f"  ⚠ {w}")

    # Suggestions
    if report.suggestions:
        lines.append(f"\nSuggestions ({len(report.suggestions)}):")
        for s in report.suggestions:
            lines.append(f"  → {s}")

    return "\n".join(lines)


def kill_zombies(report: MonitorReport, dry_run: bool = True) -> list[int]:
    """Kill zombie GPU processes. Returns list of killed PIDs."""
    killed = []
    for proc in report.processes:
        if not proc.is_zombie:
            continue
        if dry_run:
            logger.info("DRY RUN: Would kill PID %d (%s, %dMB on GPU %d)",
                        proc.pid, proc.command, proc.used_mb, proc.gpu_index)
        else:
            import signal
            import os
            try:
                os.kill(proc.pid, signal.SIGTERM)
                logger.info("Killed PID %d (%s, freed ~%dMB on GPU %d)",
                            proc.pid, proc.command, proc.used_mb, proc.gpu_index)
                killed.append(proc.pid)
            except ProcessLookupError:
                pass
            except PermissionError:
                logger.warning("Cannot kill PID %d — permission denied", proc.pid)
    return killed


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU resource monitor for Carnot")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=30, help="Loop interval in seconds")
    parser.add_argument("--kill-zombies", action="store_true", help="Kill zombie GPU processes")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be killed")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [gpu-monitor] %(message)s")

    while True:
        report = generate_report()

        if args.json:
            print(json.dumps(asdict(report), indent=2, default=str))
        else:
            print(format_report(report))

        if args.kill_zombies:
            killed = kill_zombies(report, dry_run=args.dry_run)
            if killed:
                print(f"\nKilled {len(killed)} zombie processes: {killed}")
            elif not args.dry_run:
                print("\nNo zombies found.")

        if not args.loop:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
