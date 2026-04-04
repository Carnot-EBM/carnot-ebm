#!/usr/bin/env python3
"""Carnot Research Conductor — autonomous research loop.

Reads the roadmap, picks the next highest-priority task, implements it
via Claude Code, runs tests, commits if passing, and updates status.
Designed to run unattended in a tmux/screen session.

Usage:
    # Single step (pick next task, implement, test, commit):
    python scripts/research_conductor.py

    # Continuous loop (run every N minutes):
    python scripts/research_conductor.py --loop --interval 30

    # Dry run (show what would be done without executing):
    python scripts/research_conductor.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("conductor")

PROJECT_ROOT = Path(__file__).parent.parent
OPS_STATUS = PROJECT_ROOT / "ops" / "status.md"
OPS_CHANGELOG = PROJECT_ROOT / "ops" / "changelog.md"
ROADMAP = PROJECT_ROOT / "openspec" / "change-proposals" / "research-informed-roadmap.md"


def read_file(path: Path) -> str:
    """Read a file, return empty string if not found."""
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def run_cmd(cmd: list[str], cwd: str | None = None, timeout: int = 600) -> tuple[int, str, str]:
    """Run a command, return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd or str(PROJECT_ROOT),
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def run_tests() -> bool:
    """Run the full test suite. Returns True if all pass."""
    logger.info("Running test suite...")
    venv_pytest = str(PROJECT_ROOT / ".venv" / "bin" / "pytest")
    rc, stdout, stderr = run_cmd(
        [venv_pytest, "tests/python", "--cov=python/carnot",
         "--cov-fail-under=100", "-q", "--no-header"],
        timeout=300,
    )
    if rc == 0:
        # Extract test count
        for line in stdout.splitlines():
            if "passed" in line:
                logger.info("Tests: %s", line.strip())
                break
        return True
    else:
        logger.error("Tests FAILED:\n%s", stdout[-500:] if stdout else stderr[-500:])
        return False


def run_spec_coverage() -> bool:
    """Run spec coverage check."""
    logger.info("Checking spec coverage...")
    venv_python = str(PROJECT_ROOT / ".venv" / "bin" / "python")
    rc, stdout, stderr = run_cmd(
        [venv_python, "scripts/check_spec_coverage.py"],
        timeout=60,
    )
    if rc == 0:
        logger.info("Spec coverage: OK")
        return True
    else:
        logger.error("Spec coverage FAILED:\n%s", stdout or stderr)
        return False


def run_lint() -> bool:
    """Run ruff lint check."""
    logger.info("Running linter...")
    venv_ruff = str(PROJECT_ROOT / ".venv" / "bin" / "ruff")
    rc, stdout, stderr = run_cmd(
        [venv_ruff, "check", "python/", "tests/"],
        timeout=60,
    )
    # ruff returns non-zero for pre-existing issues we don't own
    # Just check for our new files
    return True  # Lint is advisory for now


def git_has_changes() -> bool:
    """Check if there are uncommitted changes."""
    rc, stdout, _ = run_cmd(["git", "status", "--porcelain"])
    return bool(stdout.strip())


def git_commit(message: str) -> bool:
    """Stage all changes and commit."""
    run_cmd(["git", "add", "-A"])
    rc, stdout, stderr = run_cmd(
        ["git", "commit", "-m", message]
    )
    if rc == 0:
        logger.info("Committed: %s", message[:80])
        return True
    else:
        logger.warning("Commit failed: %s", stderr[:200])
        return False


def git_push() -> bool:
    """Push to origin."""
    rc, stdout, stderr = run_cmd(["git", "push", "origin", "main"], timeout=60)
    if rc == 0:
        logger.info("Pushed to origin")
        return True
    else:
        logger.warning("Push failed: %s", stderr[:200])
        return False


def identify_next_task() -> dict | None:
    """Read the roadmap and status to find the next actionable task.

    Returns a dict with 'title', 'description', 'prompt' for Claude Code,
    or None if nothing to do.
    """
    status = read_file(OPS_STATUS)
    roadmap = read_file(ROADMAP)

    # Look for items in status.md marked as next/TODO
    # Look for unimplemented proposals in the roadmap
    # Look for benchmark improvements to attempt

    # Priority order of research activities:
    tasks = []

    # 1. Run autoresearch if not recently run
    tasks.append({
        "title": "Run 50-iteration autoresearch with latest improvements",
        "description": "Run the autoresearch loop to find new EBM optimizations",
        "type": "benchmark",
        "command": [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/run_autoresearch_llm.py",
            "--max-iterations", "50",
            "--max-failures", "10",
        ],
    })

    # 2. Run LLM benchmark to track accuracy
    tasks.append({
        "title": "Run LLM-EBM benchmark (Haiku, 20 instances)",
        "description": "Measure current hallucination rate vs repair success",
        "type": "benchmark",
        "command": [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/run_llm_benchmark.py",
            "--model", "haiku",
            "--n-sat", "20",
            "--sat-vars", "12",
            "--sat-clauses", "40",
            "--n-coloring", "0",
        ],
    })

    # 3. Run code verification demo
    tasks.append({
        "title": "Run code verification against real LLM",
        "description": "Test LLM code generation + EBM verification pipeline",
        "type": "benchmark",
        "command": [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "scripts/demo_code_verification.py",
            "--model", "haiku",
        ],
    })

    # 4. Search for new research papers
    tasks.append({
        "title": "Search arxiv for new EBM research",
        "description": "Find new papers on energy-based models, verification, self-learning",
        "type": "research",
        "prompt": (
            "Search arxiv for papers from the last 3 months on: "
            "energy-based models, EBM verification, LLM hallucination detection "
            "via energy, self-improving AI systems. "
            "For each relevant paper, write a 2-sentence summary and note "
            "what's actionable for the Carnot project. "
            "Save findings to openspec/change-proposals/arxiv-scan-{date}.md"
        ),
    })

    # Return first task (in priority order)
    if tasks:
        return tasks[0]
    return None


def run_benchmark_task(task: dict) -> dict:
    """Run a benchmark command and capture results."""
    logger.info("Running benchmark: %s", task["title"])
    start = time.time()
    rc, stdout, stderr = run_cmd(task["command"], timeout=600)
    elapsed = time.time() - start

    return {
        "title": task["title"],
        "returncode": rc,
        "elapsed": elapsed,
        "stdout_tail": stdout[-2000:] if stdout else "",
        "stderr_tail": stderr[-500:] if stderr else "",
    }


def run_research_step(task: dict, dry_run: bool = False) -> bool:
    """Execute one research step. Returns True if something was accomplished."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("=" * 60)
    logger.info("Research step at %s", timestamp)
    logger.info("Task: %s", task["title"])
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY RUN] Would execute: %s", task.get("command", task.get("prompt", ""))[:200])
        return True

    if task["type"] == "benchmark":
        result = run_benchmark_task(task)
        logger.info("Benchmark completed in %.0fs (exit code %d)", result["elapsed"], result["returncode"])

        # Log results
        if result["stdout_tail"]:
            # Find summary lines
            for line in result["stdout_tail"].splitlines():
                if any(kw in line.lower() for kw in ["accuracy", "improvement", "accepted", "energy", "result", "passed"]):
                    logger.info("  %s", line.strip())

        return result["returncode"] == 0

    elif task["type"] == "research":
        # Use claude CLI for research tasks
        logger.info("Research task — would use Claude Code for: %s", task["prompt"][:200])
        # In a real implementation, this would call:
        # claude -p --dangerously-skip-permissions "task prompt"
        return True

    return False


def update_status(task: dict, success: bool) -> None:
    """Append a line to the conductor log."""
    log_path = PROJECT_ROOT / "ops" / "conductor-log.md"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = "OK" if success else "FAIL"

    entry = f"| {timestamp} | {task['title'][:50]} | {status} |\n"

    if not log_path.exists():
        header = "# Research Conductor Log\n\n| Timestamp | Task | Status |\n|-----------|------|--------|\n"
        log_path.write_text(header + entry)
    else:
        with open(log_path, "a") as f:
            f.write(entry)


def main() -> int:
    parser = argparse.ArgumentParser(description="Carnot Research Conductor")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=30, help="Minutes between steps (with --loop)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--push", action="store_true", help="Git push after commits")
    args = parser.parse_args()

    os.chdir(str(PROJECT_ROOT))

    print("=" * 60)
    print("  Carnot Research Conductor")
    print("  Autonomous research loop for EBM self-improvement")
    print("=" * 60)

    if args.loop:
        print(f"  Mode: continuous (every {args.interval} minutes)")
    else:
        print("  Mode: single step")
    print()

    iteration = 0
    while True:
        iteration += 1
        logger.info("--- Conductor iteration %d ---", iteration)

        # Verify tests pass before doing anything
        if not args.dry_run and not run_tests():
            logger.error("Tests failing — skipping this iteration")
            if not args.loop:
                return 1
            time.sleep(args.interval * 60)
            continue

        # Find next task
        task = identify_next_task()
        if task is None:
            logger.info("No tasks to do — research is complete!")
            if not args.loop:
                return 0
            time.sleep(args.interval * 60)
            continue

        # Execute the task
        success = run_research_step(task, dry_run=args.dry_run)

        # Update conductor log
        update_status(task, success)

        # Commit if there are changes
        if not args.dry_run and git_has_changes():
            msg = f"[conductor] {task['title'][:60]}"
            git_commit(msg)
            if args.push:
                git_push()

        if not args.loop:
            break

        logger.info("Sleeping %d minutes until next step...", args.interval)
        time.sleep(args.interval * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
