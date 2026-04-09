#!/usr/bin/env python3
"""Carnot Research Conductor — autonomous research via Claude Code.

Tasks are loaded from YAML files:
  research-roadmap.yaml   — pending experiments (processed in order)
  research-complete.yaml  — completed experiments (historical record)

Milestones use CalVer (YYYY.MM.seq) to show chronology.
See openspec/change-proposals/ for roadmap design docs.

Uses `claude -p` to actually implement research improvements, not just
run benchmarks. Each iteration: identify a gap → ask Claude to fix it →
verify tests pass → commit → push.

Usage:
    # Single research step:
    python scripts/research_conductor.py

    # Continuous loop:
    python scripts/research_conductor.py --loop --interval 30

    # Dry run:
    python scripts/research_conductor.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [conductor] %(message)s",
)
logger = logging.getLogger("conductor")

PROJECT_ROOT = Path(__file__).parent.parent
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
CONDUCTOR_LOG = PROJECT_ROOT / "ops" / "conductor-log.md"


def run_cmd(
    cmd: list[str],
    timeout: int = 600,
    input_text: str | None = None,
) -> tuple[int, str, str]:
    """Run a command, return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=timeout,
            input=input_text,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


def run_claude(prompt: str, max_turns: int = 20, timeout: int = 600) -> tuple[bool, str]:
    """Run claude -p with a research prompt. Returns (success, output).

    Uses --verbose and streams output live to the terminal so you can
    watch Claude working in real-time.
    """
    cmd = [
        CLAUDE_BIN, "-p",
        "--dangerously-skip-permissions",
        "--verbose",
        "--max-turns", str(max_turns),
    ]

    logger.info("Calling Claude Code (%d max turns)...", max_turns)

    # Set CARNOT_MODE=research so that .claude/settings.json hooks
    # (phase gate, task completion) bypass their checks. The research
    # conductor manages its own test/commit/revert cycle independently.
    env = {**os.environ, "CARNOT_MODE": "research"}

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for live viewing
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        # Send prompt and close stdin
        if proc.stdin:
            proc.stdin.write(prompt)
            proc.stdin.close()

        # Stream output live to terminal while capturing it
        output_lines = []
        while True:
            if proc.stdout is None:
                break
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                print(line, end="", flush=True)  # Live to terminal
                output_lines.append(line)

        proc.wait(timeout=timeout)
        full_output = "".join(output_lines)

        if proc.returncode != 0:
            logger.error("Claude failed (exit %d)", proc.returncode)
            return False, full_output[-500:]

        logger.info("Claude completed (exit 0)")
        return True, full_output[-2000:]

    except subprocess.TimeoutExpired:
        proc.kill()
        logger.error("Claude timed out after %ds", timeout)
        return False, "Timed out"
    except Exception as e:
        logger.error("Claude error: %s", e)
        return False, str(e)


def run_tests() -> tuple[bool, str]:
    """Run the full test suite. Returns (passed, summary)."""
    logger.info("Running test suite...")
    venv_pytest = str(PROJECT_ROOT / ".venv" / "bin" / "pytest")
    rc, stdout, stderr = run_cmd(
        [venv_pytest, "tests/python", "--cov=python/carnot",
         "--cov-fail-under=100", "-q", "--no-header"],
        timeout=300,
    )
    # Find the summary line
    summary = ""
    for line in (stdout or stderr).splitlines():
        if "passed" in line or "failed" in line:
            summary = line.strip()
            break
    return rc == 0, summary


def git_status() -> str:
    """Get git status summary."""
    _, stdout, _ = run_cmd(["git", "diff", "--stat"])
    return stdout.strip()


def git_has_changes() -> bool:
    """Check if there are uncommitted changes."""
    _, stdout, _ = run_cmd(["git", "status", "--porcelain"])
    return bool(stdout.strip())


def git_commit_and_push(message: str, push: bool = True) -> bool:
    """Stage, commit, and optionally push."""
    run_cmd(["git", "add", "-A"])
    rc, _, stderr = run_cmd(["git", "commit", "-m", message])
    if rc != 0:
        logger.warning("Commit failed: %s", stderr[:200])
        return False
    logger.info("Committed: %s", message[:80])
    if push:
        rc, _, stderr = run_cmd(["git", "push", "origin", "main"], timeout=60)
        if rc == 0:
            logger.info("Pushed to origin")
        else:
            logger.warning("Push failed: %s", stderr[:200])
    return True


def log_step(task: str, status: str, details: str = "") -> None:
    """Append to conductor log."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = f"| {timestamp} | {task[:50]} | {status} | {details[:80]} |\n"

    if not CONDUCTOR_LOG.exists():
        header = (
            "# Research Conductor Log\n\n"
            "| Timestamp | Task | Status | Details |\n"
            "|-----------|------|--------|---------|\n"
        )
        CONDUCTOR_LOG.write_text(header + entry)
    else:
        with open(CONDUCTOR_LOG, "a") as f:
            f.write(entry)


# ---------------------------------------------------------------------------
# Research task definitions — loaded from YAML
# ---------------------------------------------------------------------------

ROADMAP_FILE = PROJECT_ROOT / "research-roadmap.yaml"
COMPLETE_FILE = PROJECT_ROOT / "research-complete.yaml"
NEXT_ROADMAP_FILE = PROJECT_ROOT / "research-roadmap-next.yaml"
NEXT_ROADMAP_FILE = PROJECT_ROOT / "research-roadmap-next.yaml"


def load_research_tasks() -> list[dict]:
    """Load pending research tasks from research-roadmap.yaml.

    Falls back to an empty list if the file is missing or malformed.
    Each YAML task is converted to the dict format the conductor expects:
      {"id": str, "deliverable": str, "title": str, "prompt": str}
    """
    if not ROADMAP_FILE.exists():
        logger.warning("research-roadmap.yaml not found — no tasks to run")
        return []
    try:
        with open(ROADMAP_FILE) as f:
            data = yaml.safe_load(f)
        tasks = data.get("tasks", [])
        return [
            {
                "id": t["id"],
                "deliverable": t.get("deliverable", ""),
                "title": t["title"],
                "prompt": t["prompt"],
            }
            for t in tasks
        ]
    except Exception as e:
        logger.error("Failed to load research-roadmap.yaml: %s", e)
        return []


# Task list loaded lazily from YAML on first access.
# Re-reads if the file's mtime changes (no restart needed for YAML edits).
RESEARCH_TASKS: list[dict] = []
_tasks_loaded = False
_roadmap_mtime: float = 0


def _ensure_tasks_loaded():
    """Load tasks from YAML, re-reading if the file changed since last load."""
    global RESEARCH_TASKS, _tasks_loaded, _roadmap_mtime
    try:
        current_mtime = ROADMAP_FILE.stat().st_mtime if ROADMAP_FILE.exists() else 0
    except OSError:
        current_mtime = 0

    if not _tasks_loaded or current_mtime != _roadmap_mtime:
        RESEARCH_TASKS = load_research_tasks()
        _tasks_loaded = True
        _roadmap_mtime = current_mtime


MAX_FAILURES_PER_TASK = 3  # Skip task after this many consecutive failures



def _deliverable_exists(task: dict) -> bool:
    """Check if a task's deliverable file already exists in the repo.

    If the task has a "deliverable" key (a file path), check if it exists.
    This catches the case where the conductor built something but the log
    didn't record it as OK (e.g., coverage failure at commit time, power loss).
    """
    deliverable = task.get("deliverable")
    if not deliverable:
        return False
    path = PROJECT_ROOT / deliverable
    return path.exists()


def pick_next_task(completed_log: str) -> dict | None:
    """Pick the next task that hasn't been completed or failed too many times.

    Uses THREE signals to determine if a task is done:
    1. Conductor log says OK (explicit completion)
    2. Deliverable file exists (implicit completion — code was built)
    3. Failure count >= MAX (exhausted — skip it)
    """
    # Parse completed and failed task counts from log
    completed_titles = set()
    fail_counts: dict[str, int] = {}

    for line in completed_log.splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        title = parts[2].strip()
        status = parts[3].strip()

        if status == "OK":
            completed_titles.add(title)
            fail_counts[title] = 0  # Reset on success
        elif status in ("FAIL", "REVERT", "SKIP", "NOOP"):
            fail_counts[title] = fail_counts.get(title, 0) + 1

    # Ensure tasks are loaded from YAML
    _ensure_tasks_loaded()

    # Find first task not yet completed AND not failed too many times
    for task in RESEARCH_TASKS:
        title_prefix = task["title"][:50]

        # Signal 1: log says OK
        if title_prefix in completed_titles:
            continue

        # Signal 2: deliverable already exists (built but not logged)
        if _deliverable_exists(task):
            logger.info("Task '%s' deliverable exists — marking complete", title_prefix)
            log_step(title_prefix, "OK", "Deliverable already exists in repo")
            continue

        # Signal 3: too many failures
        if fail_counts.get(title_prefix, 0) >= MAX_FAILURES_PER_TASK:
            logger.warning("Skipping '%s' — failed %d times", title_prefix, fail_counts[title_prefix])
            continue

        return task

    # All tasks completed or exhausted — return None
    logger.info("All %d research tasks completed or exhausted. Nothing to do.", len(RESEARCH_TASKS))
    return None


def _load_roadmap_metadata() -> dict:
    """Load milestone metadata from the active roadmap YAML."""
    if not ROADMAP_FILE.exists():
        return {}
    try:
        with open(ROADMAP_FILE) as f:
            data = yaml.safe_load(f)
        return {
            "milestone": data.get("milestone", "unknown"),
            "milestone_title": data.get("milestone_title", ""),
            "milestone_doc": data.get("milestone_doc", ""),
        }
    except Exception:
        return {}


def _archive_current_milestone(push: bool = True) -> bool:
    """Archive the current milestone's tasks to research-complete.yaml.

    Reads the current roadmap, appends its tasks to the completed file,
    and clears the active roadmap. Returns True if successful.
    """
    if not ROADMAP_FILE.exists():
        return False

    try:
        with open(ROADMAP_FILE) as f:
            roadmap = yaml.safe_load(f)
    except Exception as e:
        logger.error("Failed to read roadmap for archival: %s", e)
        return False

    milestone = roadmap.get("milestone", "unknown")
    title = roadmap.get("milestone_title", "")
    tasks = roadmap.get("tasks", [])
    if not tasks:
        return False

    logger.info("Archiving milestone %s (%s) — %d tasks", milestone, title, len(tasks))

    # Build the completed milestone entry
    completed_entry = {
        "id": milestone,
        "title": title,
        "doc": roadmap.get("milestone_doc", ""),
        "completed": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "finding": "See conductor log for per-experiment results.",
        "tasks": [
            {
                "id": t["id"],
                "title": t["title"],
                "deliverable": t.get("deliverable", ""),
                "result": "OK (conductor)",
            }
            for t in tasks
        ],
    }

    # Append to research-complete.yaml
    try:
        if COMPLETE_FILE.exists():
            with open(COMPLETE_FILE) as f:
                complete_data = yaml.safe_load(f) or {}
        else:
            complete_data = {"milestones": []}

        milestones = complete_data.get("milestones", [])
        milestones.append(completed_entry)
        complete_data["milestones"] = milestones

        with open(COMPLETE_FILE, "w") as f:
            f.write("# Carnot Research — Completed Experiments\n")
            f.write("# Tasks moved here from research-roadmap.yaml after successful completion.\n")
            f.write("# Ordered chronologically by completion date.\n\n")
            yaml.dump(complete_data, f, default_flow_style=False, sort_keys=False, width=120)

        logger.info("Archived %d tasks to research-complete.yaml", len(tasks))
    except Exception as e:
        logger.error("Failed to archive milestone: %s", e)
        return False

    return True


def _activate_next_roadmap(push: bool = True) -> bool:
    """Swap research-roadmap-next.yaml into research-roadmap.yaml.

    If a next roadmap exists, it becomes the active roadmap. The old
    roadmap should already be archived via _archive_current_milestone().
    Returns True if a new roadmap was activated.
    """
    if not NEXT_ROADMAP_FILE.exists():
        logger.info("No research-roadmap-next.yaml found — nothing to activate")
        return False

    try:
        # Validate the next roadmap is well-formed
        with open(NEXT_ROADMAP_FILE) as f:
            next_data = yaml.safe_load(f)
        next_tasks = next_data.get("tasks", [])
        if not next_tasks:
            logger.warning("research-roadmap-next.yaml has no tasks — skipping")
            return False

        next_milestone = next_data.get("milestone", "unknown")
        logger.info(
            "Activating next roadmap: milestone %s (%d tasks)",
            next_milestone, len(next_tasks),
        )

        # Swap: next -> active, delete next
        shutil.copy2(NEXT_ROADMAP_FILE, ROADMAP_FILE)
        NEXT_ROADMAP_FILE.unlink()

        # Reset task cache so next iteration loads the new tasks
        global RESEARCH_TASKS, _tasks_loaded
        RESEARCH_TASKS = []
        _tasks_loaded = False

        # Commit the milestone transition
        run_cmd(["git", "add", "research-roadmap.yaml", "research-complete.yaml"])
        run_cmd(["git", "add", "--force", "research-roadmap-next.yaml"])  # In case it's gitignored
        msg = (
            f"[conductor] Activate milestone {next_milestone}\n\n"
            f"Archived previous milestone, activated {next_milestone}.\n\n"
            "Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
        )
        run_cmd(["git", "commit", "-m", msg])
        if push:
            run_cmd(["git", "push", "origin", "main"], timeout=60)

        log_step(f"Milestone {next_milestone} activated", "OK", f"{len(next_tasks)} tasks queued")
        return True

    except Exception as e:
        logger.error("Failed to activate next roadmap: %s", e)
        return False


def _plan_next_milestone(push: bool = True) -> bool:
    """Ask Claude to plan the next research milestone.

    When all current tasks are done AND no research-roadmap-next.yaml exists,
    this function asks Claude to analyze completed work, the PRD, and the
    architecture to propose the next milestone with a full set of experiment
    tasks in research-roadmap-next.yaml format.

    Returns True if a next roadmap was successfully created.
    """
    if NEXT_ROADMAP_FILE.exists():
        logger.info("research-roadmap-next.yaml already exists — skipping planning")
        return False

    current = _load_roadmap_metadata()
    current_milestone = current.get("milestone", "unknown")

    logger.info("=" * 60)
    logger.info("PLANNING NEXT MILESTONE (current: %s)", current_milestone)
    logger.info("=" * 60)

    planning_prompt = f"""You are the research planning agent for the Carnot EBM framework in {PROJECT_ROOT}.
Read CLAUDE.md for project context and code style requirements.

ALL TASKS IN THE CURRENT MILESTONE ({current_milestone}) HAVE COMPLETED.
Your job: plan the NEXT research milestone.

READ THESE FILES FIRST (in order):
1. _bmad/prd.md — long-term vision and requirements
2. _bmad/architecture.md — current architecture
3. ops/status.md — what's working, what's next
4. ops/changelog.md — recent work
5. research-complete.yaml — all completed experiments and findings
6. research-roadmap.yaml — the milestone that just finished
7. openspec/change-proposals/ — all previous roadmap docs (v2-v8+)
8. ops/conductor-log.md — per-experiment results

THEN:
1. Identify the 3 biggest gaps between current state and PRD vision
2. Determine the natural next experiments based on completed work
3. Design 10-14 experiments across 3-4 phases
4. Use Qwen3.5-0.8B and google/gemma-4-E4B-it as the target LLM models
   (latest small SoTA — do NOT propose older models like Llama/Phi)

CREATE TWO FILES:

FILE 1: openspec/change-proposals/research-roadmap-v{{NEXT_VERSION}}.md
- Full milestone design doc following the v7/v8 format
- Include: what previous milestone proved, architecture diagram, phase descriptions,
  dependency graph, hardware requirements, what's deferred

FILE 2: research-roadmap-next.yaml
- Full YAML with all experiment tasks in conductor execution order
- Follow the EXACT format of research-roadmap.yaml:
  milestone, milestone_title, milestone_doc, tasks (id, milestone, deliverable, title, prompt)
- Each prompt must include: CONTEXT, EXISTING CODE TO READ FIRST, TASK, CONCRETE STEPS
- End each prompt with: Run command, "Do NOT push. Do NOT modify scripts/research_conductor.py."
- Use {{project_root}} and {{date}} as placeholders in prompts

IMPORTANT:
- Do NOT modify research-roadmap.yaml (the conductor manages this)
- Do NOT modify scripts/research_conductor.py
- Do NOT push
- CalVer milestones: increment the seq number (e.g., 2026.04.3 -> 2026.04.4, or 2026.05.1 if month changes)
- Each experiment must have a clear deliverable file path
- Experiments should be ordered so dependencies are met (earlier experiments first)
"""

    success, output = run_claude(planning_prompt, max_turns=50, timeout=1200)

    if not success:
        logger.error("Planning agent failed: %s", output[:200])
        log_step("Plan next milestone", "FAIL", f"Claude error: {output[:60]}")
        return False

    # Verify the planning agent created the next roadmap file
    if not NEXT_ROADMAP_FILE.exists():
        logger.warning("Planning agent ran but didn't create research-roadmap-next.yaml")
        log_step("Plan next milestone", "FAIL", "No research-roadmap-next.yaml produced")
        return False

    # Validate the YAML is well-formed
    try:
        with open(NEXT_ROADMAP_FILE) as f:
            next_data = yaml.safe_load(f)
        next_tasks = next_data.get("tasks", [])
        next_milestone = next_data.get("milestone", "unknown")
        logger.info(
            "Planning agent created milestone %s with %d tasks",
            next_milestone, len(next_tasks),
        )
    except Exception as e:
        logger.error("Planning agent produced invalid YAML: %s", e)
        NEXT_ROADMAP_FILE.unlink(missing_ok=True)
        log_step("Plan next milestone", "FAIL", f"Invalid YAML: {e}")
        return False

    # Commit the planned roadmap
    if git_has_changes():
        # Guard: don't let planning modify conductor or active roadmap
        for guarded in ["scripts/research_conductor.py", "research-roadmap.yaml"]:
            _, gdiff, _ = run_cmd(["git", "diff", "--name-only", "--", guarded])
            if gdiff.strip():
                logger.warning("Planning agent modified %s — reverting", guarded)
                run_cmd(["git", "checkout", "--", guarded])

        run_cmd(["git", "add", "-A"])
        msg = (
            f"[conductor] Plan next milestone: {next_milestone}\n\n"
            f"Planning agent proposed {len(next_tasks)} experiments.\n"
            f"Stored in research-roadmap-next.yaml for activation.\n\n"
            "Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
        )
        git_commit_and_push(msg, push=push)

    log_step(f"Plan milestone {next_milestone}", "OK", f"{len(next_tasks)} tasks proposed")
    return True


def _load_roadmap_metadata() -> dict:
    """Load milestone metadata from the active roadmap YAML."""
    if not ROADMAP_FILE.exists():
        return {}
    try:
        with open(ROADMAP_FILE) as f:
            data = yaml.safe_load(f)
        return {
            "milestone": data.get("milestone", "unknown"),
            "milestone_title": data.get("milestone_title", ""),
            "milestone_doc": data.get("milestone_doc", ""),
        }
    except Exception:
        return {}


def _archive_current_milestone(push: bool = True) -> bool:
    """Archive the current milestone's tasks to research-complete.yaml.

    Reads the current roadmap, appends its tasks to the completed file,
    and clears the active roadmap. Returns True if successful.
    """
    if not ROADMAP_FILE.exists():
        return False

    try:
        with open(ROADMAP_FILE) as f:
            roadmap = yaml.safe_load(f)
    except Exception as e:
        logger.error("Failed to read roadmap for archival: %s", e)
        return False

    milestone = roadmap.get("milestone", "unknown")
    title = roadmap.get("milestone_title", "")
    tasks = roadmap.get("tasks", [])
    if not tasks:
        return False

    logger.info("Archiving milestone %s (%s) — %d tasks", milestone, title, len(tasks))

    completed_entry = {
        "id": milestone,
        "title": title,
        "doc": roadmap.get("milestone_doc", ""),
        "completed": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "finding": "See conductor log for per-experiment results.",
        "tasks": [
            {
                "id": t["id"],
                "title": t["title"],
                "deliverable": t.get("deliverable", ""),
                "result": "OK (conductor)",
            }
            for t in tasks
        ],
    }

    try:
        if COMPLETE_FILE.exists():
            with open(COMPLETE_FILE) as f:
                complete_data = yaml.safe_load(f) or {}
        else:
            complete_data = {"milestones": []}

        milestones = complete_data.get("milestones", [])
        milestones.append(completed_entry)
        complete_data["milestones"] = milestones

        with open(COMPLETE_FILE, "w") as f:
            f.write("# Carnot Research — Completed Experiments\n")
            f.write("# Tasks moved here from research-roadmap.yaml after successful completion.\n")
            f.write("# Ordered chronologically by completion date.\n\n")
            yaml.dump(complete_data, f, default_flow_style=False, sort_keys=False, width=120)

        logger.info("Archived %d tasks to research-complete.yaml", len(tasks))
    except Exception as e:
        logger.error("Failed to archive milestone: %s", e)
        return False

    return True


def _activate_next_roadmap(push: bool = True) -> bool:
    """Swap research-roadmap-next.yaml into research-roadmap.yaml.

    If a next roadmap exists, it becomes the active roadmap. The old
    roadmap should already be archived via _archive_current_milestone().
    Returns True if a new roadmap was activated.
    """
    if not NEXT_ROADMAP_FILE.exists():
        logger.info("No research-roadmap-next.yaml found — nothing to activate")
        return False

    try:
        with open(NEXT_ROADMAP_FILE) as f:
            next_data = yaml.safe_load(f)
        next_tasks = next_data.get("tasks", [])
        if not next_tasks:
            logger.warning("research-roadmap-next.yaml has no tasks — skipping")
            return False

        next_milestone = next_data.get("milestone", "unknown")
        logger.info(
            "Activating next roadmap: milestone %s (%d tasks)",
            next_milestone, len(next_tasks),
        )

        shutil.copy2(NEXT_ROADMAP_FILE, ROADMAP_FILE)
        NEXT_ROADMAP_FILE.unlink()

        # Reset task cache so next iteration loads the new tasks
        global RESEARCH_TASKS, _tasks_loaded
        RESEARCH_TASKS = []
        _tasks_loaded = False

        run_cmd(["git", "add", "research-roadmap.yaml", "research-complete.yaml"])
        msg = (
            f"[conductor] Activate milestone {next_milestone}\n\n"
            f"Archived previous milestone, activated {next_milestone}.\n\n"
            "Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
        )
        run_cmd(["git", "commit", "-m", msg])
        if push:
            run_cmd(["git", "push", "origin", "main"], timeout=60)

        log_step(f"Milestone {next_milestone} activated", "OK", f"{len(next_tasks)} tasks queued")
        return True

    except Exception as e:
        logger.error("Failed to activate next roadmap: %s", e)
        return False


def _plan_next_milestone(push: bool = True) -> bool:
    """Ask Claude to plan the next research milestone.

    When all current tasks are done AND no research-roadmap-next.yaml exists,
    this function asks Claude to analyze completed work and propose the next
    milestone with a full set of experiment tasks.

    Returns True if a next roadmap was successfully created.
    """
    if NEXT_ROADMAP_FILE.exists():
        logger.info("research-roadmap-next.yaml already exists — skipping planning")
        return False

    current = _load_roadmap_metadata()
    current_milestone = current.get("milestone", "unknown")

    logger.info("=" * 60)
    logger.info("PLANNING NEXT MILESTONE (current: %s)", current_milestone)
    logger.info("=" * 60)

    planning_prompt = (
        f"You are the research planning agent for the Carnot EBM framework in {PROJECT_ROOT}.\n"
        f"Read CLAUDE.md for project context and code style requirements.\n\n"
        f"ALL TASKS IN THE CURRENT MILESTONE ({current_milestone}) HAVE COMPLETED.\n"
        f"Your job: plan the NEXT research milestone.\n\n"
        f"READ THESE FILES FIRST (in order):\n"
        f"1. research-program.md — HIGH-LEVEL GOALS AND PRIORITIES (start here)\n"
        f"2. _bmad/prd.md — long-term vision and requirements\n"
        f"3. _bmad/architecture.md — current architecture\n"
        f"4. ops/status.md — what's working, what's next\n"
        f"5. ops/changelog.md — recent work\n"
        f"6. research-complete.yaml — all completed experiments and findings\n"
        f"7. research-roadmap.yaml — the milestone that just finished\n"
        f"8. openspec/change-proposals/ — all previous roadmap docs\n"
        f"9. ops/conductor-log.md — per-experiment results\n"
        f"10. research-references.md — technologies and ideas to consider\n\n"
        f"THEN:\n"
        f"1. Identify the 3 biggest gaps between current state and PRD vision\n"
        f"2. Determine the natural next experiments based on completed work\n"
        f"3. Design 10-14 experiments across 3-4 phases\n"
        f"4. Use Qwen3.5-0.8B and google/gemma-4-E4B-it as the target LLM models\n"
        f"   (latest small SoTA — do NOT propose older models)\n\n"
        f"CREATE TWO FILES:\n\n"
        f"FILE 1: openspec/change-proposals/research-roadmap-vNEXT.md\n"
        f"- Full milestone design doc following the v7/v8 format\n"
        f"- Include: what previous milestone proved, architecture diagram,\n"
        f"  phase descriptions, dependency graph, hardware requirements\n\n"
        f"FILE 2: research-roadmap-next.yaml\n"
        f"- Full YAML with all experiment tasks in conductor execution order\n"
        f"- Follow the EXACT format of research-roadmap.yaml:\n"
        f"  milestone, milestone_title, milestone_doc, tasks\n"
        f"- Each prompt must include: CONTEXT, EXISTING CODE TO READ FIRST,\n"
        f"  TASK, CONCRETE STEPS\n"
        f"- End each prompt with: Run command, 'Do NOT push. Do NOT modify "
        f"scripts/research_conductor.py.'\n"
        f"- Use {{project_root}} and {{date}} as placeholders in prompts\n\n"
        f"IMPORTANT:\n"
        f"- Do NOT modify research-roadmap.yaml\n"
        f"- Do NOT modify scripts/research_conductor.py\n"
        f"- Do NOT push\n"
        f"- CalVer milestones: increment the seq number\n"
        f"- Each experiment must have a clear deliverable file path\n"
    )

    success, output = run_claude(planning_prompt, max_turns=50, timeout=1200)

    if not success:
        logger.error("Planning agent failed: %s", output[:200])
        log_step("Plan next milestone", "FAIL", f"Claude error: {output[:60]}")
        return False

    if not NEXT_ROADMAP_FILE.exists():
        logger.warning("Planning agent ran but didn't create research-roadmap-next.yaml")
        log_step("Plan next milestone", "FAIL", "No research-roadmap-next.yaml produced")
        return False

    try:
        with open(NEXT_ROADMAP_FILE) as f:
            next_data = yaml.safe_load(f)
        next_tasks = next_data.get("tasks", [])
        next_milestone = next_data.get("milestone", "unknown")
        logger.info(
            "Planning agent created milestone %s with %d tasks",
            next_milestone, len(next_tasks),
        )
    except Exception as e:
        logger.error("Planning agent produced invalid YAML: %s", e)
        NEXT_ROADMAP_FILE.unlink(missing_ok=True)
        log_step("Plan next milestone", "FAIL", f"Invalid YAML: {e}")
        return False

    if git_has_changes():
        for guarded in ["scripts/research_conductor.py", "research-roadmap.yaml"]:
            _, gdiff, _ = run_cmd(["git", "diff", "--name-only", "--", guarded])
            if gdiff.strip():
                logger.warning("Planning agent modified %s — reverting", guarded)
                run_cmd(["git", "checkout", "--", guarded])

        run_cmd(["git", "add", "-A"])
        msg = (
            f"[conductor] Plan next milestone: {next_milestone}\n\n"
            f"Planning agent proposed {len(next_tasks)} experiments.\n"
            f"Stored in research-roadmap-next.yaml for activation.\n\n"
            "Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
        )
        git_commit_and_push(msg, push=push)

    log_step(f"Plan milestone {next_milestone}", "OK", f"{len(next_tasks)} tasks proposed")
    return True


def research_step(push: bool = True, dry_run: bool = False) -> bool:
    """Execute one research step. Returns True if progress was made."""
    timestamp = datetime.now(timezone.utc)

    # Read conductor log
    log_content = ""
    if CONDUCTOR_LOG.exists():
        log_content = CONDUCTOR_LOG.read_text()

    # Pick task
    task = pick_next_task(log_content)
    if task is None:
        # All tasks in current milestone are done.
        # Try to transition to the next milestone.
        logger.info("All tasks in current milestone complete.")

        if NEXT_ROADMAP_FILE.exists():
            # A next roadmap is ready — archive current and activate it
            logger.info("Found research-roadmap-next.yaml — transitioning milestones")
            _archive_current_milestone(push=push)
            activated = _activate_next_roadmap(push=push)
            if activated:
                logger.info("New milestone activated — will pick up tasks next iteration")
                return True  # Progress was made (milestone transition)
            else:
                logger.error("Failed to activate next roadmap")
                return False
        else:
            # No next roadmap — ask planning agent to create one
            logger.info("No research-roadmap-next.yaml — launching planning agent")
            if dry_run:
                logger.info("[DRY RUN] Would launch planning agent for next milestone")
                return False
            planned = _plan_next_milestone(push=push)
            if planned:
                # Planning created the file; next iteration will activate it
                logger.info("Planning complete — will activate next iteration")
                return True
            else:
                logger.info("Planning did not produce a next roadmap")
                return False

    logger.info("=" * 60)
    logger.info("RESEARCH STEP: %s", task["title"])
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY RUN] Would run Claude with prompt:")
        logger.info("  %s", task["prompt"][:200].format(
            project_root=PROJECT_ROOT,
            date=timestamp.strftime("%Y%m%d"),
        ))
        return True

    # Clean any dirty state from previous interrupted runs
    # But preserve conductor-log.md (it's modified every iteration)
    if git_has_changes():
        # Check if the ONLY change is conductor-log.md (normal operation)
        _, porcelain, _ = run_cmd(["git", "diff", "--name-only"])
        changed_files = [f.strip() for f in porcelain.splitlines() if f.strip()]
        if changed_files == ["ops/conductor-log.md"]:
            # Just the log — commit it and continue
            run_cmd(["git", "add", "ops/conductor-log.md"])
            run_cmd(["git", "commit", "-m", "[conductor] Update conductor log"])
        else:
            # Real dirty state — revert everything except the log
            logger.warning("Dirty working tree detected — reverting to clean state")
            run_cmd(["git", "checkout", "--", "."])
            run_cmd(["git", "clean", "-fd", "--exclude=.coverage*"])

    # Run tests first — ensure clean state
    tests_ok, test_summary = run_tests()
    if not tests_ok:
        # Self-heal: ask Claude to fix the pre-existing test failures
        # before attempting the research task. This prevents getting stuck
        # in a loop where every iteration SKIPs because tests are broken.
        logger.warning("Pre-flight tests failing — attempting self-heal")
        logger.warning("Failure: %s", test_summary[:300])

        heal_prompt = (
            f"You are working on the Carnot EBM framework in {PROJECT_ROOT}.\n\n"
            f"The test suite is failing BEFORE any research work. This is a pre-existing "
            f"issue that must be fixed before the research conductor can proceed.\n\n"
            f"Test output:\n{test_summary}\n\n"
            f"TASK: Fix the failing tests so the full suite passes with 100%% coverage.\n"
            f"Common causes:\n"
            f"- New code added without tests (coverage < 100%%)\n"
            f"- A test assertion that no longer matches reality\n"
            f"- A missing file or dependency\n"
            f"- A CSS/HTML check that doesn't match the current docs\n\n"
            f"STEPS:\n"
            f"1. Read the test failure output above carefully\n"
            f"2. Identify the root cause (not just the symptom)\n"
            f"3. Fix it — add tests, fix assertions, update docs, etc.\n"
            f"4. Run: JAX_PLATFORMS=cpu .venv/bin/pytest tests/python --tb=short -q\n"
            f"5. Verify 0 failures and 100%% coverage\n"
            f"6. Do NOT push. Do NOT modify scripts/research_conductor.py or "
            f"research-roadmap.yaml."
        )

        MAX_HEAL_ATTEMPTS = 2
        healed = False
        for heal_attempt in range(MAX_HEAL_ATTEMPTS):
            logger.info("Self-heal attempt %d/%d", heal_attempt + 1, MAX_HEAL_ATTEMPTS)
            heal_ok, heal_output = run_claude(heal_prompt, max_turns=30, timeout=600)
            if not heal_ok:
                logger.error("Claude failed during self-heal: %s", heal_output[:200])
                break

            # Guard: don't let heal modify conductor or roadmap
            for guarded in ["scripts/research_conductor.py", "research-roadmap.yaml"]:
                _, gdiff, _ = run_cmd(["git", "diff", "--name-only", "--", guarded])
                if gdiff.strip():
                    logger.warning("Self-heal modified %s — reverting", guarded)
                    run_cmd(["git", "checkout", "--", guarded])

            tests_ok, test_summary = run_tests()
            if tests_ok:
                logger.info("Self-heal succeeded: %s", test_summary)
                # Commit the fix
                if git_has_changes():
                    git_commit_and_push(
                        "[conductor] Self-heal: fix pre-existing test failures\n\n"
                        "Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
                        push=push,
                    )
                healed = True
                break
            else:
                logger.warning("Self-heal attempt %d failed: %s", heal_attempt + 1, test_summary[:200])
                # Update the prompt with new failure info for next attempt
                heal_prompt = (
                    f"You are working on the Carnot EBM framework in {PROJECT_ROOT}.\n\n"
                    f"Previous self-heal attempt did not fully fix the tests.\n\n"
                    f"Current test output:\n{test_summary}\n\n"
                    f"Fix the remaining failures. 100%% coverage required.\n"
                    f"Do NOT modify scripts/research_conductor.py or research-roadmap.yaml."
                )

        if not healed:
            # Revert any partial self-heal changes and abort
            if git_has_changes():
                run_cmd(["git", "checkout", "."])
                run_cmd(["git", "clean", "-fd", "--exclude=.coverage*"])
            logger.error("Self-heal failed after %d attempts — aborting", MAX_HEAL_ATTEMPTS)
            log_step(task["title"], "SKIP", f"Pre-tests failing, self-heal failed: {test_summary}")
            return False

    logger.info("Pre-check: %s", test_summary)

    # Format the prompt with project root
    prompt = task["prompt"].format(
        project_root=PROJECT_ROOT,
        date=timestamp.strftime("%Y%m%d"),
    )

    # Run Claude Code
    success, output = run_claude(prompt, max_turns=50, timeout=1200)

    if not success:
        logger.error("Claude failed: %s", output[:200])
        log_step(task["title"], "FAIL", f"Claude error: {output[:60]}")
        return False

    # Check if Claude made any changes
    if not git_has_changes():
        logger.info("Claude made no file changes")
        log_step(task["title"], "FAIL", "No file changes produced")
        return True

    # Show what changed
    diff = git_status()
    logger.info("Changes:\n%s", diff[:500])

    # Guard: never let Claude modify the conductor itself
    _, conductor_diff, _ = run_cmd(["git", "diff", "--name-only", "--", "scripts/research_conductor.py"])
    if conductor_diff.strip():
        logger.warning("Claude modified research_conductor.py — reverting that file")
        run_cmd(["git", "checkout", "--", "scripts/research_conductor.py"])

    # Run tests after changes — retry up to 2 times if tests fail
    MAX_FIX_ATTEMPTS = 2
    tests_ok, test_summary = run_tests()

    for fix_attempt in range(MAX_FIX_ATTEMPTS):
        if tests_ok:
            break
        logger.warning("Tests FAILED (attempt %d/%d): %s", fix_attempt + 1, MAX_FIX_ATTEMPTS, test_summary[:200])

        # Feed the test failure back to Claude to fix
        fix_prompt = (
            f"You are working on the Carnot EBM framework in {PROJECT_ROOT}.\n\n"
            f"Your previous changes caused test failures:\n{test_summary}\n\n"
            f"Fix the failing tests. Do NOT revert your changes — fix the code "
            f"so all tests pass with 100% coverage.\n"
            f"Do NOT modify scripts/research_conductor.py."
        )
        logger.info("Asking Claude to fix test failures...")
        fix_ok, fix_output = run_claude(fix_prompt, max_turns=30, timeout=600)
        if not fix_ok:
            logger.error("Claude failed to fix tests")
            break
        tests_ok, test_summary = run_tests()

    if not tests_ok:
        logger.error("Tests still failing after %d fix attempts — reverting", MAX_FIX_ATTEMPTS)
        run_cmd(["git", "checkout", "."])
        run_cmd(["git", "clean", "-fd", "--exclude=.coverage*"])
        log_step(task["title"], "REVERT", f"Post-tests failed: {test_summary}")
        return False

    logger.info("Post-check: %s", test_summary)

    # Commit and push
    commit_msg = (
        f"[conductor] {task['title']}\n\n"
        f"Automated research step by research conductor.\n"
        f"Task ID: {task['id']}\n\n"
        f"Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
    )
    git_commit_and_push(commit_msg, push=push)

    # Post-commit reconciliation: ask Claude to update docs for this experiment
    logger.info("Running post-commit documentation reconciliation...")
    reconcile_prompt = (
        f"You are working on the Carnot EBM framework in {PROJECT_ROOT}.\n\n"
        f"A research experiment was just completed and committed:\n"
        f"  Task: {task['title']}\n"
        f"  ID: {task['id']}\n\n"
        f"TASK: Update documentation to reflect this new work.\n"
        f"1. Add an entry to ops/changelog.md for today ({timestamp.strftime('%Y-%m-%d')})\n"
        f"2. Update ops/status.md if this adds new capabilities\n"
        f"3. Update _bmad/traceability.md if new REQ-*/SCENARIO-* were added\n"
        f"4. Do NOT remove existing content — only ADD\n"
        f"5. Do NOT modify scripts/research_conductor.py or research-roadmap.yaml\n"
        f"6. Keep changes minimal — just the doc updates for this experiment.\n"
    )
    recon_ok, recon_output = run_claude(reconcile_prompt, max_turns=15, timeout=300)
    if recon_ok and git_has_changes():
        # Guard protected files
        for guarded in ["scripts/research_conductor.py", "research-roadmap.yaml"]:
            _, gdiff, _ = run_cmd(["git", "diff", "--name-only", "--", guarded])
            if gdiff.strip():
                run_cmd(["git", "checkout", "--", guarded])
        git_commit_and_push(
            f"[conductor] Update docs for {task['title']}\n\n"
            f"Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
            push=push,
        )
        logger.info("Documentation reconciliation committed")
    else:
        logger.info("No doc updates needed (or reconciliation skipped)")

    log_step(task["title"], "OK", test_summary)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Carnot Research Conductor")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=30,
                        help="Minutes between steps (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done")
    parser.add_argument("--no-push", action="store_true",
                        help="Don't git push (just commit locally)")
    args = parser.parse_args()

    os.chdir(str(PROJECT_ROOT))

    print("=" * 60)
    print("  Carnot Research Conductor")
    print("  Autonomous research via Claude Code")
    print("=" * 60)
    print(f"  Claude: {CLAUDE_BIN}")
    print(f"  Project: {PROJECT_ROOT}")
    if args.loop:
        print(f"  Mode: continuous (every {args.interval} min)")
    else:
        print("  Mode: single step")
    print()

    iteration = 0
    while True:
        iteration += 1
        logger.info("--- Iteration %d ---", iteration)

        try:
            progress = research_step(
                push=not args.no_push,
                dry_run=args.dry_run,
            )
        except Exception:
            logger.exception("Unexpected error in research step")
            progress = False

        if not args.loop:
            return 0 if progress else 1

        logger.info("Sleeping %d minutes...", args.interval)
        time.sleep(args.interval * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
