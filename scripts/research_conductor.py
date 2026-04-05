#!/usr/bin/env python3
"""Carnot Research Conductor — autonomous research via Claude Code.

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
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
    """Run claude -p with a research prompt. Returns (success, output)."""
    cmd = [
        CLAUDE_BIN, "-p",
        "--dangerously-skip-permissions",
        "--output-format", "json",
        "--max-turns", str(max_turns),
    ]

    logger.info("Calling Claude Code (%d max turns)...", max_turns)
    rc, stdout, stderr = run_cmd(cmd, timeout=timeout, input_text=prompt)

    if rc != 0:
        logger.error("Claude failed (exit %d): %s", rc, stderr[:500])
        return False, stderr[:500]

    # Parse JSON output
    try:
        result = json.loads(stdout)
        output = result.get("result", stdout)
        cost = result.get("total_cost_usd", 0)
        turns = result.get("num_turns", 0)
        logger.info("Claude completed: %d turns, $%.4f", turns, cost)
        return True, output
    except json.JSONDecodeError:
        return True, stdout[:2000]


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
# Research task definitions
# ---------------------------------------------------------------------------

# Each task is a prompt that tells Claude Code what to do.
# Claude has full file access and can edit code, run tests, etc.

RESEARCH_TASKS = [
    {
        "id": "improve-repair-success",
        "title": "Improve SAT repair success rate",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.

CONTEXT: The Haiku LLM benchmark shows 60% SAT accuracy → 80% after gradient repair.
The remaining 20% fails because single-start repair gets stuck in local minima.

TASK: Modify scripts/run_llm_benchmark.py to use multi_start_repair (from
carnot.inference.multi_start) with n_starts=5 instead of single-start repair.
This should improve the repair success rate.

STEPS:
1. Read the current run_llm_benchmark.py
2. Read python/carnot/inference/multi_start.py to understand the API
3. Modify the SAT experiment in run_llm_sat_experiment() to use multi_start_repair
4. Run the full test suite: .venv/bin/pytest tests/python --cov=python/carnot --cov-fail-under=100
5. If tests pass, update ops/status.md with the change
6. Do NOT push to git (the conductor handles that)

IMPORTANT: Maintain 100% test coverage. Add tests if needed.""",
    },
    {
        "id": "property-test-integration",
        "title": "Integrate property testing into iterative refinement",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.

CONTEXT: We have property-based testing (carnot.verify.property_test) and iterative
refinement (iterative_refine_code in carnot.inference.llm_solver), but they're
not connected. The refinement loop uses simple test cases, missing edge cases.

TASK: Create a new function iterative_refine_with_properties() that combines
iterative_refine_code with property_test — after each LLM attempt, run both
specific test cases AND random property tests, feeding ALL failures back to the LLM.

STEPS:
1. Read python/carnot/inference/llm_solver.py (iterative_refine_code)
2. Read python/carnot/verify/property_test.py (property_test, format_violations_for_llm)
3. Add iterative_refine_with_properties() to llm_solver.py
4. Add tests in tests/python/test_inference_iterative_refine.py
5. Run full test suite: .venv/bin/pytest tests/python --cov=python/carnot --cov-fail-under=100
6. Maintain 100% coverage and spec coverage (every test references REQ-* or SCENARIO-*)""",
    },
    {
        "id": "harder-code-tasks",
        "title": "Find coding tasks that make LLMs fail",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.

CONTEXT: Current coding tasks (fibonacci, gcd, binary_search, etc.) are solved
perfectly by both Sonnet and Haiku. We need tasks where LLMs actually make mistakes
so the EBM verification adds value.

TASK: Add 5 new coding tasks to scripts/demo_code_verification.py that are likely
to trip up LLMs. Focus on:
- Stateful/mutable operations (in-place list manipulation with tricky indices)
- Off-by-one edge cases (ranges, boundaries, fencepost problems)
- Numerical precision (floating point comparison, rounding)
- Unusual specifications (non-standard sort orders, custom encodings)
- Property-based test cases that check invariants, not just specific values

STEPS:
1. Read scripts/demo_code_verification.py
2. Add 5 new tasks to TASKS_HARD with comprehensive test cases
3. Include at least one property-based test case per task
4. Run the demo with --model haiku to see if any tasks fail
5. Document which tasks Haiku fails on (if any)""",
    },
    {
        "id": "diffusion-vs-repair-benchmark",
        "title": "Compare diffusion generation vs LLM+repair on SAT",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.

CONTEXT: We have two ways to solve SAT:
1. LLM generates → EBM repairs (run_llm_sat_experiment)
2. Diffusion generates from noise (diffusion_generate_sat)
Neither has been compared head-to-head.

TASK: Write a benchmark script scripts/bench_diffusion_vs_llm.py that:
1. Generates 10 random SAT instances (12 vars, 40 clauses)
2. Solves each with diffusion (n_candidates=10, 100 steps)
3. Measures: success rate, energy, wall clock time
4. Compares against the known Haiku results (60% → 80%)
5. Prints a comparison table

STEPS:
1. Read python/carnot/inference/diffusion.py
2. Read python/carnot/inference/benchmark.py for instance generation
3. Write the benchmark script
4. Run it and report results""",
    },
    {
        "id": "arxiv-scan",
        "title": "Scan arxiv for new EBM research",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.

TASK: Search arxiv for papers from the last 3 months (2026) on:
- Energy-based models for code verification
- EBM + LLM integration
- Self-improving AI systems
- Hallucination detection via energy

For each relevant paper, write:
1. Title and arxiv ID
2. One-sentence summary
3. What's actionable for Carnot (specific module/function that could benefit)

Save to openspec/change-proposals/arxiv-scan-{date}.md

Use web search to find the papers.""",
    },
    {
        "id": "improve-code-embedding",
        "title": "Improve code embedding beyond bag-of-tokens",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.

CONTEXT: The current code_to_embedding() in python/carnot/verify/python_types.py
uses a simple bag-of-tokens frequency vector. This loses all structural information
(indentation, nesting, control flow). The learned code verifier's accuracy is
limited by this embedding quality.

TASK: Add an AST-based embedding that captures structural features:
- Number of function calls, loops, conditionals, returns
- Nesting depth statistics
- Variable count and reuse patterns
- Cyclomatic complexity approximation

STEPS:
1. Read python/carnot/verify/python_types.py (current code_to_embedding)
2. Add ast_code_to_embedding() using Python's ast module
3. Add tests comparing bag-of-tokens vs AST embedding discriminability
4. Run full test suite, maintain 100% coverage
5. Update the code verifier to optionally use the AST embedding""",
    },
]


def pick_next_task(completed_log: str) -> dict | None:
    """Pick the next task that hasn't been completed recently."""
    # Parse completed tasks from log
    recent_completed = set()
    for line in completed_log.splitlines():
        if "| OK |" in line:
            parts = line.split("|")
            if len(parts) >= 3:
                task_name = parts[2].strip()
                recent_completed.add(task_name)

    # Find first task not recently completed
    for task in RESEARCH_TASKS:
        if task["title"][:50] not in recent_completed:
            return task

    # All tasks completed — cycle back to first
    return RESEARCH_TASKS[0]


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
        logger.info("No tasks available")
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

    # Run tests first — ensure clean state
    tests_ok, test_summary = run_tests()
    if not tests_ok:
        logger.error("Tests failing before research step — aborting")
        log_step(task["title"], "SKIP", f"Pre-tests failing: {test_summary}")
        return False

    logger.info("Pre-check: %s", test_summary)

    # Format the prompt with project root
    prompt = task["prompt"].format(
        project_root=PROJECT_ROOT,
        date=timestamp.strftime("%Y%m%d"),
    )

    # Run Claude Code
    success, output = run_claude(prompt, max_turns=30, timeout=900)

    if not success:
        logger.error("Claude failed: %s", output[:200])
        log_step(task["title"], "FAIL", f"Claude error: {output[:60]}")
        return False

    # Check if Claude made any changes
    if not git_has_changes():
        logger.info("Claude made no file changes")
        log_step(task["title"], "NOOP", "No file changes")
        return True

    # Show what changed
    diff = git_status()
    logger.info("Changes:\n%s", diff[:500])

    # Run tests after changes
    tests_ok, test_summary = run_tests()
    if not tests_ok:
        logger.error("Tests FAILED after changes — reverting")
        run_cmd(["git", "checkout", "."])
        run_cmd(["git", "clean", "-fd"])
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
