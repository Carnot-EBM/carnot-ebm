#!/usr/bin/env python3
"""Carnot Research Conductor — autonomous research via a configurable CLI agent.

Tasks are loaded from YAML files:
  research-roadmap.yaml   — pending experiments (processed in order)
  research-complete.yaml  — completed experiments (historical record)

Milestones use CalVer (YYYY.MM.seq) to show chronology.
See openspec/change-proposals/ for roadmap design docs.

Uses the configured agent CLI (`claude`, `gemini`, `opencode`, or `codex`)
to actually implement research improvements, not just run benchmarks.
Each iteration: identify a gap → ask the agent to fix it → verify tests
pass → commit → push.

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
import concurrent.futures
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [conductor] %(message)s",
)
logger = logging.getLogger("conductor")

PROJECT_ROOT = Path(__file__).parent.parent

# Per-agent-type lookup tables. Module-level constants for the four supported
# CLI backends; per-task agent_type override (added 2026-04-29) reads from
# these instead of the process-startup AGENT_TYPE constant. See
# openspec/change-proposals/multi-agent-routing.md.
AGENT_BIN_BY_TYPE = {
    "claude": os.environ.get("CLAUDE_BIN", "claude"),
    "gemini": os.environ.get("GEMINI_BIN", "gemini"),
    "opencode": os.environ.get("OPENCODE_BIN", "opencode"),
    "codex": os.environ.get("CODEX_BIN", "codex"),
}
DEFAULT_MODEL_BY_TYPE = {
    "claude": "sonnet",
    "gemini": "gemini-3.1-pro-preview",
    "opencode": "opencode/big-pickle",
    "codex": "gpt-5.5",
}

# Flexible agent configuration: AGENT_TYPE can be 'claude', 'gemini', 'opencode', or 'codex'
RAW_AGENT_TYPE = os.environ.get("AGENT_TYPE", "claude").lower()
if RAW_AGENT_TYPE in AGENT_BIN_BY_TYPE:
    AGENT_TYPE = RAW_AGENT_TYPE
else:
    AGENT_TYPE = "claude"
AGENT_BIN = AGENT_BIN_BY_TYPE[AGENT_TYPE]
DEFAULT_MODEL = DEFAULT_MODEL_BY_TYPE[AGENT_TYPE]

AGENT_MODEL = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
# Per-role model overrides (none = fall through to AGENT_MODEL). Rationale:
# Opus is best for high-synthesis tasks (milestone planning, retrospective
# honest self-evaluation); Sonnet is the default for experiments and docs;
# Haiku handles simple post-commit reconciliation. See 2026-04-17 analysis
# of Claude Opus 4.7 system card for the three-tier gating decision.
AGENT_MODEL_PLANNER = os.environ.get("AGENT_MODEL_PLANNER")  # e.g. "opus"
AGENT_MODEL_RETRO = os.environ.get("AGENT_MODEL_RETRO")  # e.g. "opus"
# 2026-05-02 fix: per-role agent_type overrides. When AGENT_TYPE=codex
# globally (Anthropic-quota conservation), the planner role specifically
# stalls at 180s silence on 50-turn YAML generation (codex/gpt-5.5
# observed twice in .88→.89 transition). These overrides let operators
# pin the planner+retro to claude (small Anthropic burn ~$1-2 per
# milestone) while experiments stay on codex (bulk savings preserved).
AGENT_TYPE_PLANNER = os.environ.get("AGENT_TYPE_PLANNER")  # e.g. "claude"
AGENT_TYPE_RETRO = os.environ.get("AGENT_TYPE_RETRO")  # e.g. "claude"
CONDUCTOR_LOG = PROJECT_ROOT / "ops" / "conductor-log.md"
DOGFOOD_MEMORY_FILE = PROJECT_ROOT / "ops" / "dogfood-memory.json"
AGENT_DISPLAY_BY_TYPE = {
    "claude": "Claude Code",
    "gemini": "Gemini CLI",
    "opencode": "OpenCode CLI",
    "codex": "Codex CLI",
}
AGENT_DISPLAY = AGENT_DISPLAY_BY_TYPE[AGENT_TYPE]
AGENT_SIGNATURE_BY_TYPE = {
    "claude": "\n\nCo-Authored-By: Claude Code <noreply@anthropic.com>",
    "gemini": "\n\nCo-Authored-By: Gemini CLI <noreply@google.com>",
    "opencode": "\n\nCo-Authored-By: OpenCode CLI <noreply@opencode.ai>",
    "codex": "\n\nCo-Authored-By: Codex CLI <noreply@openai.com>",
}
AGENT_SIGNATURE = AGENT_SIGNATURE_BY_TYPE[AGENT_TYPE]


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


def with_agent_signature(message: str) -> str:
    """Append the configured agent signature to a commit message."""
    return message.strip() + AGENT_SIGNATURE


def _build_agent_command(
    prompt: str,
    max_turns: int,
    model_override: str | None = None,
    agent_type_override: str | None = None,
) -> tuple[list[str], str | None, str]:
    """Build the command, optional stdin payload, and log message.

    agent_type_override: per-task agent backend (claude/codex/gemini/opencode).
    Falls through to module-level AGENT_TYPE when None. See multi-agent-routing
    proposal: openspec/change-proposals/multi-agent-routing.md.
    """
    effective_agent_type = agent_type_override or AGENT_TYPE
    if effective_agent_type not in AGENT_BIN_BY_TYPE:
        # Defensive: unknown agent_type silently falls back to default rather
        # than crashing the conductor mid-iteration. The schema validator
        # catches typos at planner output time.
        logger.warning(
            "Unknown agent_type %r — falling back to module default %r",
            effective_agent_type,
            AGENT_TYPE,
        )
        effective_agent_type = AGENT_TYPE

    bin_path = AGENT_BIN_BY_TYPE[effective_agent_type]
    display = AGENT_DISPLAY_BY_TYPE[effective_agent_type]
    # Model resolution: explicit override > AGENT_MODEL (when type matches the
    # process default) > per-agent-type DEFAULT_MODEL_BY_TYPE.
    if model_override:
        model = model_override
    elif effective_agent_type == AGENT_TYPE:
        model = AGENT_MODEL
    else:
        model = DEFAULT_MODEL_BY_TYPE[effective_agent_type]

    # 2026-05-01 fix: model_override is meaningful only when agent_type
    # matches the override's vendor namespace. The .85 planner emitted
    # `model: opus` on tasks with `agent_type: codex` (exp1097, exp1098),
    # then codex CLI rejected the Anthropic model with HTTP 400:
    # "The 'opus' model is not supported when using Codex with a ChatGPT
    # account." Snap any cross-vendor model name to the agent_type's
    # default. Anthropic names (sonnet/opus/haiku/claude-*) only make
    # sense for agent_type=claude; gpt-/o1-/codex-* only for codex;
    # gemini-* only for gemini.
    _ANTHROPIC_NAMES = ("sonnet", "opus", "haiku")
    if effective_agent_type == "codex":
        if any(model.lower().startswith(p) for p in _ANTHROPIC_NAMES) or model.startswith(
            "claude-"
        ):
            logger.warning(
                "Cross-vendor model override ignored: agent_type=codex got model=%s; "
                "snapping to default %s",
                model,
                DEFAULT_MODEL_BY_TYPE["codex"],
            )
            model = DEFAULT_MODEL_BY_TYPE["codex"]
    elif effective_agent_type == "gemini":
        if any(model.lower().startswith(p) for p in _ANTHROPIC_NAMES) or model.startswith("gpt-"):
            logger.warning(
                "Cross-vendor model override ignored: agent_type=gemini got model=%s; "
                "snapping to default %s",
                model,
                DEFAULT_MODEL_BY_TYPE["gemini"],
            )
            model = DEFAULT_MODEL_BY_TYPE["gemini"]
    elif effective_agent_type == "claude":
        if model.startswith(("gpt-", "gemini-", "o1-")):
            logger.warning(
                "Cross-vendor model override ignored: agent_type=claude got model=%s; "
                "snapping to default %s",
                model,
                DEFAULT_MODEL_BY_TYPE["claude"],
            )
            model = DEFAULT_MODEL_BY_TYPE["claude"]

    if effective_agent_type == "gemini":
        return (
            [
                bin_path,
                "-p",
                prompt,
                "--yolo",
                "--model",
                model,
            ],
            None,
            f"Calling {display} (model: {model})...",
        )

    if effective_agent_type == "opencode":
        return (
            [
                bin_path,
                "run",
                "--dangerously-skip-permissions",
                "--model",
                model,
                "--dir",
                str(PROJECT_ROOT),
                prompt,
            ],
            None,
            f"Calling {display} (model: {model})...",
        )

    if effective_agent_type == "codex":
        # 2026-04-30 fix: removed the
        #   `-c model_providers.openai.stream_idle_timeout_ms=120000`
        # override that was killing every codex invocation in
        # milestones .82-.84 with "Error loading config.toml:
        # model_providers contains reserved". Newer codex CLI versions
        # (verified on 0.125.0) treat the `model_providers.*` key
        # namespace as reserved and reject `-c` overrides into it.
        # Direct invocation without this override works correctly:
        # tested with `codex exec --color never --model gpt-5.5
        # --ephemeral - <<< "What is 17+25?"` returning the expected
        # answer with full session metadata.
        #
        # The original "stall fix #2" intent (2-min stream idle
        # protection) is now redundant — the conductor's own
        # progress-aware wall-clock timeout (commit 9683ea5e) catches
        # silent stalls at the orchestrator layer regardless of the
        # codex CLI's internal timeout.
        return (
            [
                bin_path,
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--color",
                "never",
                "--model",
                model,
                "--cd",
                str(PROJECT_ROOT),
                "--ephemeral",  # Prevent session file accumulation (stall fix #3)
                "-c",
                # 2026-05-04: bumped 1200 → 7200 (120 min cap). The 1200s cap
                # was killing legitimate long-running tasks (Phase-5-D 70 min,
                # GRPO v6 90 min, Boltzmann CD training, etc.) before they
                # could write the terminal artifact, causing universal
                # artifact_not_updated_past_bootstrap failures across .96
                # and .97. The comment above already acknowledged the cap
                # was "redundant" given the conductor's own progress-aware
                # wall-clock timeout (5-min idle grace + 4x HARD_CAP),
                # which provides correct stall protection without
                # premature kills. Setting to 7200s gives codex
                # enough headroom for any reasonable experiment while
                # still letting the orchestrator-layer timeout catch
                # genuinely stuck processes via idle detection.
                "agents.job_max_runtime_seconds=7200",
                "-",
            ],
            prompt,
            f"Calling {display} (model: {model})...",
        )

    # Default path: claude
    return (
        [
            bin_path,
            "-p",
            "--dangerously-skip-permissions",
            "--verbose",
            "--max-turns",
            str(max_turns),
            "--model",
            model,
        ],
        prompt,
        f"Calling {display} ({max_turns} max turns, model: {model})...",
    )


def run_agent(
    prompt: str,
    max_turns: int = 20,
    timeout: int = 600,
    model_override: str | None = None,
    deliverable_path: str | None = None,
    agent_type_override: str | None = None,
) -> tuple[bool, str]:
    """Run the configured agent with a research prompt.

    Streams output live to the terminal.
    Args:
        model_override: Use a different model for this call (e.g., "haiku"
            for lightweight tasks like doc reconciliation).
        deliverable_path: Optional repo-relative path (e.g.
            ``results/experiment_453_vericot.json``) that this call is expected
            to produce.  When set, the stream loop polls the file every
            ``DELIVERABLE_POLL_SECS`` and, once it has existed with a stable
            size+mtime for ``DELIVERABLE_STABLE_SECS``, kills the subagent
            early and returns success.  This rescues the 40–55 min of
            post-deliverable polishing we were burning on every experiment
            that finishes its real work inside the first 10 min but keeps the
            subagent alive running extra tests until the 60-min wall-clock
            cap fires.  See observations around Exp 447/448 on 2026-04-18.
        agent_type_override: per-task agent backend (claude/codex/gemini/
            opencode). Falls through to module-level AGENT_TYPE when None.
            Multi-agent routing per
            openspec/change-proposals/multi-agent-routing.md.
    """
    cmd, stdin_text, msg = _build_agent_command(
        prompt, max_turns, model_override, agent_type_override
    )

    logger.info(msg)

    # Expose research mode so repo-local agent hooks can relax interactive
    # gates when the configured CLI supports them.
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
            # Put the subagent in its own process group so we can kill it plus
            # every descendant (pytest workers, python -c helpers, etc.) when
            # the wall-clock timeout fires. Without this, `proc.kill()` only
            # reaps the direct child — leaving pytest workers alive, holding
            # GPU memory, and poisoning the next pre-flight test run. Observed
            # in session 2026-04-17: 34 zombie pytest workers accumulated after
            # 8 hours of killed subagents, causing consecutive pre-flight
            # failures until manually cleared.
            start_new_session=True,
        )

        if stdin_text is not None and proc.stdin:
            proc.stdin.write(stdin_text)
            proc.stdin.close()
        elif proc.stdin:
            proc.stdin.close()

        # Stream output live with stall detection (Fix #1).
        # If no output is received for STALL_TIMEOUT seconds, kill the process.
        # This prevents infinite hangs when Codex stalls mid-stream.
        # Claude legitimately thinks for longer periods, so use a higher threshold.
        import select
        import signal

        # Stall detection: only for Codex (which has a known infinite-hang bug).
        # Claude doesn't stall — it either completes or hits the turn limit.
        # Setting to 0 disables the stall detector.
        # 2026-05-02 fix: STALL_TIMEOUT was using module-level AGENT_TYPE, which
        # ignored per-call agent_type_override. With AGENT_TYPE=codex globally
        # for quota conservation but AGENT_TYPE_PLANNER=claude override, the
        # planner ran on Sonnet but kept the codex 180s stall threshold. Sonnet
        # legitimately thinks longer than 180s for 50-turn YAML drafting, so
        # the planner kept getting killed. Resolve based on the effective agent
        # type for this call (claude → 0/disabled, codex/gemini → 600s).
        # 2026-05-04 fix: bumped codex from 180s → 600s. After flipping
        # AGENT_TYPE_PLANNER from claude to codex/gpt-5.5 for quota conservation,
        # the .100 planner stalled 11 successive times at 180s silence.
        # gpt-5.5 thinks longer between output bursts than older codex models,
        # especially during 50-turn YAML planning involving multi-step
        # research roadmap construction. The conductor's own progress-aware
        # WALL_CLOCK_TIMEOUT (5-min idle grace + 4x HARD_CAP) provides
        # adequate stall protection at the orchestrator layer; the codex-side
        # STALL_TIMEOUT only needs to catch the genuine infinite-hang bug.
        # 600s gives codex enough thinking room while still bounding hang
        # detection within ~10 min.
        _effective_for_stall = agent_type_override or AGENT_TYPE
        STALL_TIMEOUT = 0 if _effective_for_stall == "claude" else 600

        def _kill_subagent_group(reason: str) -> None:
            """Kill the subagent AND every descendant process in its process group.

            Critical: a bare ``proc.kill()`` only reaps the direct child. Pytest
            workers, python -c helpers, model-loading processes spawned by the
            subagent survive and hold GPU memory + CPU. After 8 hours of killed
            subagents on 2026-04-17, 34 zombie pytest workers accumulated and
            started poisoning the next iteration's pre-flight by competing for
            resources. Killing the process group (SIGKILL to -pgid) reaps them
            all in one call.
            """
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError) as exc:
                logger.warning(
                    "Process group kill for %s failed (%s) — falling back to proc.kill()",
                    reason,
                    exc,
                )
                proc.kill()

        # Wall-clock timeout: the Exp 425 ExperimentTimeoutWatchdog guards
        # inside experiment scripts, but run_agent spawns the Claude CLI which
        # itself can hang past its turn limit (observed Exp 426: 90+ min with
        # no output, still alive). This is the orchestrator-level counterpart.
        #
        # Resolution order (2026-04-29 fix — was previously a silent dead-code bug):
        #   1. Honor the explicit ``timeout`` parameter when > 0. All 5
        #      production call sites pass explicit values (600s for doc
        #      reconciliation + self-heal; 1200s for planning + research
        #      steps), so this is the active path.
        #   2. Fall back to the CARNOT_CONDUCTOR_TIMEOUT_MINUTES env var (60
        #      min default) when the caller passes 0 or omits the arg.
        #
        # Prior bug (caused 5+ stuck-Sonnet wedges on 2026-04-29): the
        # ``timeout`` parameter was ignored entirely; only the env-var
        # 60-minute default applied. Self-heal calls expected 10 min but
        # actually got 60 min, allowing pre-test self-heal to hang for
        # 49 min before manual intervention. Honoring the explicit
        # parameter caps doc-recon + self-heal at 10 min, planning +
        # research at 20 min — matching what the call sites intend.
        #
        # Configurable via CARNOT_CONDUCTOR_TIMEOUT_MINUTES (only when
        # explicit timeout is 0). Experiments that legitimately require
        # >20 min should split: subagent writes the script (under 20 min),
        # a separate long-running executor runs it.
        if timeout and timeout > 0:
            WALL_CLOCK_TIMEOUT = timeout
        else:
            WALL_CLOCK_TIMEOUT = int(os.environ.get("CARNOT_CONDUCTOR_TIMEOUT_MINUTES", "60")) * 60
        start_time = time.time()

        output_lines = []
        last_output_time = time.time()

        # Deliverable-watch state.  We remember the first (size, mtime) at
        # which the deliverable appeared and declare it "stable" once it
        # stops changing for DELIVERABLE_STABLE_SECS.  This gives the
        # subagent a short grace window to finish writing the file before
        # we kill it — avoids racing against half-written JSON.
        DELIVERABLE_POLL_SECS = 30
        DELIVERABLE_STABLE_SECS = 120
        deliverable_file = None
        if deliverable_path:
            deliverable_file = PROJECT_ROOT / deliverable_path
        deliverable_last_check = time.time()
        deliverable_stable_since: float | None = None
        deliverable_last_sig: tuple[int, float] | None = None

        while True:
            if proc.stdout is None:
                break

            # Use select for non-blocking read with timeout
            ready, _, _ = select.select([proc.stdout], [], [], 5.0)

            if ready:
                line = proc.stdout.readline()
                if not line and proc.poll() is not None:
                    break
                if line:
                    print(line, end="", flush=True)  # Live to terminal
                    output_lines.append(line)
                    last_output_time = time.time()
            else:
                # No output ready — check for stall
                if proc.poll() is not None:
                    break  # Process exited
                elapsed_silence = time.time() - last_output_time
                if STALL_TIMEOUT > 0 and elapsed_silence > STALL_TIMEOUT:
                    logger.warning(
                        "%s stalled — no output for %ds, killing process group",
                        AGENT_DISPLAY,
                        int(elapsed_silence),
                    )
                    _kill_subagent_group("stall")
                    proc.wait(timeout=10)
                    full_output = "".join(output_lines)
                    return (
                        False,
                        f"Stalled after {int(elapsed_silence)}s silence. Last output: {full_output[-300:]}",
                    )

            # Deliverable-watch: if the experiment has produced its expected
            # result file and it has been stable for DELIVERABLE_STABLE_SECS,
            # kill the subagent early.  The real work is done; further
            # turns are usually doc polishing or extra tests that the
            # conductor's own post-run reconciliation step will redo anyway.
            now = time.time()
            if (
                deliverable_file is not None
                and now - deliverable_last_check >= DELIVERABLE_POLL_SECS
            ):
                deliverable_last_check = now
                try:
                    st = deliverable_file.stat()
                    sig = (st.st_size, st.st_mtime)
                except FileNotFoundError:
                    sig = None
                    deliverable_stable_since = None
                    deliverable_last_sig = None
                if sig is not None:
                    # 2026-05-01 fix (Issue 3): require mtime > start_time
                    # before allowing the stable-deliverable kill. If the
                    # file pre-exists from a prior iteration (e.g., a
                    # status='blocked' artifact from DOOMED_RERUN_BLOCK),
                    # the new agent has not yet started writing — killing
                    # at the 60s mark on stale-but-unchanged file is a
                    # false positive. Empirical .85 incident: exp1090's
                    # Opus was killed after 2.5 min of unchanged stale
                    # blocked artifact, before it could write the new
                    # diagnostics_library_v1 deliverable.
                    if st.st_mtime <= start_time:
                        # Pre-existing stale artifact; agent hasn't
                        # written yet. Reset the stability tracker.
                        deliverable_stable_since = None
                        deliverable_last_sig = sig
                    elif sig == deliverable_last_sig:
                        # 2026-05-04 fix: the deliverable-watch was
                        # firing on STEP-0 skeleton files that the agent
                        # wrote at the start of its run and then did NOT
                        # re-touch while doing other work (reading code,
                        # running pytest, etc.). Conductor concluded
                        # "stable → done" and killed the agent before
                        # it could write the terminal artifact, causing
                        # universal artifact_not_updated_past_bootstrap
                        # failures across .96/.97/.98. Fix: parse the
                        # deliverable's status field; if it's still in
                        # _BOOTSTRAP_STATUSES (running/blocked/partial/
                        # in_progress), the agent isn't done — don't
                        # trigger early-kill, regardless of mtime
                        # stability.
                        bootstrap_only = False
                        try:
                            with deliverable_file.open(
                                "r", encoding="utf-8"
                            ) as _fh:
                                _payload = json.load(_fh)
                            if isinstance(_payload, dict):
                                _st_field = _payload.get("status")
                                if (
                                    isinstance(_st_field, str)
                                    and _st_field.lower()
                                    in _BOOTSTRAP_STATUSES
                                ):
                                    bootstrap_only = True
                        except (OSError, json.JSONDecodeError):
                            # Mid-write race or non-JSON artifact —
                            # treat as not-yet-finished to be safe.
                            bootstrap_only = True
                        if bootstrap_only:
                            # Don't reset the stability tracker — keep
                            # tracking so we eventually fall through to
                            # WALL_CLOCK_TIMEOUT logic. But never trigger
                            # the deliverable-watch early-kill on a
                            # bootstrap-only artifact.
                            pass
                        elif deliverable_stable_since is None:
                            deliverable_stable_since = now
                        elif now - deliverable_stable_since >= DELIVERABLE_STABLE_SECS:
                            logger.info(
                                "%s produced stable deliverable %s "
                                "(%.1f min elapsed) — killing subagent early",
                                AGENT_DISPLAY,
                                deliverable_path,
                                (now - start_time) / 60,
                            )
                            _kill_subagent_group("deliverable-stable")
                            try:
                                proc.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                pass
                            full_output = "".join(output_lines)
                            return True, full_output[-2000:]
                    else:
                        deliverable_last_sig = sig
                        deliverable_stable_since = now

            # Progress-aware wall-clock timeout (applies to all agents):
            # prevents the kind of 90+ min silent hang we saw on Exp 426 while
            # allowing legitimate long-running training runs (e.g. exp1057
            # Probe Ensemble — 20+ min Sonnet was killed mid-progress).
            #
            # Algorithm: the hard `WALL_CLOCK_TIMEOUT` is a *soft* cap. Once
            # exceeded, kill ONLY if the subagent has been silent for more
            # than IDLE_GRACE seconds. While output is fresh, extend the
            # budget. Backstop at HARD_CAP_MULTIPLIER × WALL_CLOCK_TIMEOUT
            # so a chatty-but-stuck process can't run forever.
            #
            # 2026-04-30: introduced after exp1057 Probe Ensemble v6 hit a
            # hard 1201s wall-clock kill mid-training despite making real
            # progress. The user wants long training runs to complete as
            # long as they're actually doing work.
            IDLE_GRACE = 300  # 5 min of silence before we consider the run stuck
            HARD_CAP_MULTIPLIER = 4  # absolute backstop relative to soft cap
            elapsed_total = now - start_time
            elapsed_silence = now - last_output_time
            soft_cap_hit = WALL_CLOCK_TIMEOUT > 0 and elapsed_total > WALL_CLOCK_TIMEOUT
            hard_cap_hit = (
                WALL_CLOCK_TIMEOUT > 0 and elapsed_total > HARD_CAP_MULTIPLIER * WALL_CLOCK_TIMEOUT
            )

            def _rescue_via_deliverable(reason: str, elapsed: float) -> tuple[bool, str] | None:
                """After a timeout-kill, give the experiment a 60s grace
                window to flush its deliverable, then validate it before
                accepting.

                Per "scientific method" discipline (2026-04-30 user
                directive): we don't accept partially-run experiments
                as total successes. A rescue is valid ONLY when:

                  1. Artifact exists and is fresh (mtime >= start_time)
                  2. JSON parses cleanly
                  3. status is NOT in _BOOTSTRAP_STATUSES (running,
                     blocked, partial, in_progress)
                  4. honest_verdict (if present) does NOT match any
                     _PARTIAL_TOKENS / _BLOCKED_TOKENS / _FAILED_TOKENS
                     pattern — these indicate the experiment ran but
                     did not satisfy its acceptance criteria

                A bootstrap-only artifact, a "partial_some_below_x"
                verdict, or a "blocked_*" verdict produces no rescue;
                the original timeout-FAIL stands.
                """
                if not deliverable_path:
                    return None
                rescue_path = PROJECT_ROOT / deliverable_path
                rescue_deadline = time.time() + 60.0
                # Reuse the verdict-token sets from the doc reconciler
                # so rescue policy can't drift from the milestone-retro
                # policy.
                try:
                    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
                    from in_process_doc_reconcile import (  # type: ignore[import-not-found]
                        _BLOCKED_TOKENS,
                        _FAILED_TOKENS,
                        _PARTIAL_TOKENS,
                    )

                    untrustworthy_tokens = _PARTIAL_TOKENS + _BLOCKED_TOKENS + _FAILED_TOKENS
                except ImportError:
                    untrustworthy_tokens = (
                        "partial",
                        "inverted",
                        "insufficient",
                        "no_improvement",
                        "still_wrong",
                        "no_delta",
                        "below",
                        "regression",
                        "negative",
                        "flat",
                        "plateau",
                        "collapsed",
                        "blocked",
                        "failed",
                        "timed_out",
                        "exception",
                        "tolerance_exceeded",
                        "marginal",
                        "incorrect",
                    )

                while time.time() < rescue_deadline:
                    if rescue_path.exists():
                        try:
                            mtime = rescue_path.stat().st_mtime
                        except OSError:
                            mtime = 0
                        if mtime >= start_time:
                            # Validate the artifact's status + verdict
                            try:
                                with rescue_path.open("r", encoding="utf-8") as fh:
                                    payload = json.load(fh)
                            except (OSError, json.JSONDecodeError) as exc:
                                logger.warning(
                                    "Rescue candidate %s present but unparseable (%s) — refusing",
                                    deliverable_path,
                                    exc,
                                )
                                return None

                            if not isinstance(payload, dict):
                                logger.warning(
                                    "Rescue candidate %s is not a JSON object — refusing",
                                    deliverable_path,
                                )
                                return None

                            status = payload.get("status")
                            if isinstance(status, str) and status.lower() in _BOOTSTRAP_STATUSES:
                                logger.warning(
                                    "Rescue refused: %s status=%r is bootstrap-only",
                                    deliverable_path,
                                    status,
                                )
                                return None

                            verdict = payload.get("honest_verdict")
                            if isinstance(verdict, str):
                                vlow = verdict.lower()
                                if any(tok in vlow for tok in untrustworthy_tokens):
                                    logger.warning(
                                        "Rescue refused: %s honest_verdict=%r is untrustworthy "
                                        "(matches partial/blocked/failed token)",
                                        deliverable_path,
                                        verdict,
                                    )
                                    return None

                            logger.info(
                                "%s rescued via deliverable post-%s "
                                "(elapsed %.1f min, status=%r, verdict=%r)",
                                AGENT_DISPLAY,
                                reason,
                                elapsed / 60,
                                status,
                                verdict,
                            )
                            full_output_local = "".join(output_lines)
                            return True, (
                                f"[rescued via deliverable after {reason}; "
                                f"status={status} verdict={verdict}] "
                                f"{full_output_local[-1500:]}"
                            )
                    time.sleep(2.0)
                return None

            if hard_cap_hit:
                # Backstop: even with output, don't run more than 4× the soft cap.
                logger.warning(
                    "%s exceeded HARD wall-clock cap (%d min, %d× soft), killing process group",
                    AGENT_DISPLAY,
                    (HARD_CAP_MULTIPLIER * WALL_CLOCK_TIMEOUT) // 60,
                    HARD_CAP_MULTIPLIER,
                )
                _kill_subagent_group("wall-clock-hard")
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                rescued = _rescue_via_deliverable("hard-cap", elapsed_total)
                if rescued is not None:
                    return rescued
                full_output = "".join(output_lines)
                return False, (
                    f"Hard wall-clock cap after {int(elapsed_total)}s. "
                    f"Last output: {full_output[-300:]}"
                )
            elif soft_cap_hit and elapsed_silence > IDLE_GRACE:
                # Past the soft cap and the subagent has gone quiet — kill.
                logger.warning(
                    "%s past soft wall-clock cap (%d min) AND silent %ds — killing",
                    AGENT_DISPLAY,
                    WALL_CLOCK_TIMEOUT // 60,
                    int(elapsed_silence),
                )
                _kill_subagent_group("wall-clock-idle")
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                rescued = _rescue_via_deliverable("idle-timeout", elapsed_total)
                if rescued is not None:
                    return rescued
                full_output = "".join(output_lines)
                return False, (
                    f"Wall-clock+idle timeout after {int(elapsed_total)}s "
                    f"({int(elapsed_silence)}s silence). "
                    f"Last output: {full_output[-300:]}"
                )
            # else: either still within soft cap, or past it but actively
            # producing output. Let the run continue.

        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _kill_subagent_group("wait-timeout")

        full_output = "".join(output_lines)

        if proc.returncode != 0:
            logger.error("%s failed (exit %d)", AGENT_DISPLAY, proc.returncode)
            return False, full_output[-500:]

        logger.info("%s completed (exit 0)", AGENT_DISPLAY)
        return True, full_output[-2000:]

    except subprocess.TimeoutExpired:
        try:
            import signal as _sig  # noqa: PLC0415

            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, _sig.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        logger.error("%s timed out after %ds", AGENT_DISPLAY, timeout)
        return False, "Timed out"
    except Exception as e:
        logger.error("%s error: %s", AGENT_DISPLAY, e)
        return False, str(e)


def preflight_gpu_reap() -> dict:
    """Run ExpandedGPUReaper live before each research step.

    Why this exists
    ---------------
    Zombie GPU processes are the #1 cause of consecutive experiment failures in
    this project's autoresearch loop.  Observed recurrences: 2026-04-17 (34
    zombie pytest workers after 8h), 2026-04-19 (Exp 528 CUDA OOM at 1.4s with
    22+ GiB pinned by 7 stale PIDs from Exp 524/527).  Running the reaper
    **live** before every ``research_step()`` prevents the next iteration from
    being poisoned by the previous iteration's orphans.

    Safety rails
    ------------
    * Configurable via ``CARNOT_CONDUCTOR_REAPER``: set to ``0`` to disable,
      ``dry_run`` to audit without killing (default: live).
    * Reaper itself enforces ``min_age_s=1800`` — freshly-spawned legitimate
      workers are never touched.
    * Reaper enforces subtree membership — if a GPU process is a descendant of
      the current conductor, it is skipped regardless of age.
    * Wrapped in a try/except: a reaper failure must NOT block the research
      step.  If nvidia-smi is missing or the reaper crashes, we log and move on.

    Returns
    -------
    dict with keys ``killed`` (list of pids), ``total_vram_freed_mb`` (int),
    ``honest_verdict`` (str), and ``skipped_reason`` (str | None) when the
    reaper did not run (e.g. disabled, CPU-only host).
    """
    mode = os.environ.get("CARNOT_CONDUCTOR_REAPER", "live").lower()
    if mode in ("0", "off", "disabled", "no", "false"):
        return {
            "honest_verdict": "reaper_disabled",
            "killed": [],
            "total_vram_freed_mb": 0,
            "skipped_reason": "env_disabled",
        }

    try:
        sys.path.insert(0, str(PROJECT_ROOT / "python"))
        from carnot.pipeline.expanded_gpu_reaper import (
            ExpandedGPUReaper,
            ExpandedGPUReaperConfig,
        )
    except ImportError as exc:
        logger.warning("GPU reaper import failed (%s) — skipping pre-flight reap", exc)
        return {
            "honest_verdict": "reaper_import_failed",
            "killed": [],
            "total_vram_freed_mb": 0,
            "skipped_reason": str(exc),
        }

    dry = mode in ("dry_run", "dry-run", "audit")
    try:
        cfg = ExpandedGPUReaperConfig(
            min_vram_mb=1024,
            min_age_s=1800,
            dry_run=dry,
        )
        result = ExpandedGPUReaper(cfg).reap()
    except Exception as exc:  # noqa: BLE001  — never block the loop on reaper errors
        logger.warning("GPU reaper failed (%s) — continuing without reap", exc)
        return {
            "honest_verdict": "reaper_exception",
            "killed": [],
            "total_vram_freed_mb": 0,
            "skipped_reason": str(exc),
        }

    if result.killed:
        logger.warning(
            "Pre-flight reaper killed %d stale GPU process(es), freed %d MiB",
            len(result.killed),
            result.total_vram_freed_mb,
        )
        for entry in result.killed:
            logger.warning(
                "  reaped pid=%s vram=%dMiB age=%ds name=%s",
                entry.get("pid"),
                entry.get("used_memory_mb", 0),
                entry.get("age_s", 0),
                entry.get("process_name", "?"),
            )
    else:
        logger.info("Pre-flight reaper: nothing to kill (verdict=%s)", result.honest_verdict)

    return {
        "honest_verdict": result.honest_verdict,
        "killed": [e.get("pid") for e in result.killed],
        "total_vram_freed_mb": result.total_vram_freed_mb,
        "skipped_reason": None,
    }


# 2026-05-01 fingerprint cache for pre-tests.
#
# Pre-tests dominate conductor wall time: ~17 min/task in the full-suite
# path, ~32 min in the smart-subset path. Multiplied across a ~13-task
# milestone that's ~3.7 hours of pure pre-test wall time per milestone,
# while the experiments themselves average 2-5 min each. In steady state
# most iterations do not change source code at all (the conductor is
# reading roadmap YAML, writing artifacts, committing JSON results — the
# test outcomes cannot have changed since the last green run).
#
# The cache short-circuits run_tests() when the fingerprint of all .py
# files under python/carnot/, tests/python/, scripts/ — plus
# pyproject.toml, Cargo.toml, uv.lock — matches the fingerprint at the
# last green pre-test for an equivalent or stronger mode. (A green
# full-suite satisfies a subset request; a green subset does not satisfy
# a full request.)
PRETEST_CACHE_FILE = PROJECT_ROOT / "ops" / ".pretest-cache.json"
PRETEST_FINGERPRINT_DIRS = ("python/carnot", "tests/python", "scripts")
PRETEST_FINGERPRINT_FILES = ("pyproject.toml", "Cargo.toml", "uv.lock")


def _compute_pretest_fingerprint() -> str:
    """Hash mtimes + sizes of files that could affect test outcomes.

    Two runs with the same fingerprint must produce the same test results
    (modulo flaky tests). Fingerprint changes when any tracked .py file
    is added, removed, modified, or when build manifest files change.
    """
    h = hashlib.sha256()
    for d in PRETEST_FINGERPRINT_DIRS:
        root = PROJECT_ROOT / d
        if not root.exists():
            continue
        for f in sorted(root.rglob("*.py")):
            try:
                stat = f.stat()
            except OSError:
                continue
            try:
                rel = f.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                continue
            h.update(f"{rel}:{stat.st_mtime_ns}:{stat.st_size}\n".encode())
    for fname in PRETEST_FINGERPRINT_FILES:
        f = PROJECT_ROOT / fname
        if not f.exists():
            continue
        try:
            stat = f.stat()
        except OSError:
            continue
        h.update(f"{fname}:{stat.st_mtime_ns}:{stat.st_size}\n".encode())
    return h.hexdigest()


def _load_pretest_cache() -> dict:
    """Load the pretest fingerprint cache. Returns empty dict on any error."""
    try:
        return json.loads(PRETEST_CACHE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_pretest_cache(fingerprint: str, summary: str, mode: str) -> None:
    """Persist the green-pre-test fingerprint so the next run can short-circuit."""
    payload = {
        "fingerprint": fingerprint,
        "summary": summary,
        "mode": mode,
        "saved_at": datetime.now(UTC).isoformat(),
    }
    try:
        PRETEST_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        PRETEST_CACHE_FILE.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        logger.warning("Failed to write pre-test cache: %s", exc)


def _pretest_cache_satisfies(mode: str, current_fp: str, cache: dict) -> bool:
    """Whether a cached green pre-test satisfies the requested mode.

    A green full-suite cache satisfies both subset and full requests.
    A green subset cache satisfies subset requests only — a full request
    must run a real full suite even if subset previously passed.
    """
    if cache.get("fingerprint") != current_fp:
        return False
    cached_mode = cache.get("mode")
    if mode == "subset":
        return cached_mode in ("full", "subset")
    if mode == "full":
        return cached_mode == "full"
    return False


def run_tests(full: bool = False) -> tuple[bool, str]:
    """Run tests. Uses smart subset by default, full suite when full=True.

    Smart subset: runs only core tests + tests for recently changed files.
    This takes ~30-60s instead of ~8 min for the full 2300+ test suite.
    Full suite is used for post-commit validation.

    2026-05-01: short-circuits when a fingerprint-cache hit indicates no
    test-relevant file has changed since the last green pre-test of the
    requested (or stronger) mode.
    """
    mode = "full" if full else "subset"
    current_fp = _compute_pretest_fingerprint()
    cache = _load_pretest_cache()
    if _pretest_cache_satisfies(mode, current_fp, cache):
        cached_summary = cache.get("summary", "(no summary)")
        cached_mode = cache.get("mode", "?")
        logger.info(
            "Pre-test SKIPPED — fingerprint %s matches last green %s (cached summary: %s)",
            current_fp[:12],
            cached_mode,
            cached_summary,
        )
        return True, f"cache hit: {cached_summary}"

    logger.info("Running test suite%s...", " (FULL)" if full else " (smart subset)")
    venv_pytest = str(PROJECT_ROOT / ".venv" / "bin" / "pytest")

    # Pre-tests are GATING, not live experiments. Strip CARNOT_FORCE_LIVE
    # from the pytest env so live-mode-only tests (e.g. GPU EP presence
    # assertions in test_experiment_259_onnxruntime_gpu.py) don't fail the
    # gate when the dev environment lacks the asserted hardware build.
    #
    # 2026-04-30 incident: conductor's startup env had CARNOT_FORCE_LIVE=1
    # which un-skipped test_cuda_ep_present_when_gpu_available, asserting
    # CUDAExecutionProvider on a ROCm/AMD dev machine. The single failure
    # caused exp1054 (KV260) to SKIP twice (02:26Z, 02:47Z) on what is
    # actually unrelated infrastructure.
    pretest_env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}

    def _pytest_run(cmd: list[str], timeout: int) -> tuple[int, str, str]:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=timeout,
                env=pretest_env,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as exc:
            return -1, "", str(exc)

    if full:
        # Full suite — used after successful experiment commit
        rc, stdout, stderr = _pytest_run(
            [
                venv_pytest,
                "tests/python",
                "-q",
                "--no-header",
                "-n",
                "0",
                "--no-cov",
                "-o",
                "addopts=",
            ],
            timeout=600,
        )
    else:
        # Smart subset: core tests + tests for recently changed Python files
        test_files = [
            "tests/python/test_pipeline_extract.py",
            "tests/python/test_docs.py",
            # test_cli.py excluded: PBT tests hang due to hypothesis/Docker
            # test_full_pipeline.py excluded: slow integration tests
        ]

        # Add tests for any recently modified source files
        try:
            _, diff_out, _ = run_cmd(["git", "diff", "--name-only", "HEAD~1"])
            changed = [f.strip() for f in diff_out.splitlines() if f.strip()]
            for f in changed:
                if f.startswith("python/carnot/") and f.endswith(".py"):
                    # Map source file to likely test file
                    module = f.replace("python/carnot/", "").replace("/", "_").replace(".py", "")
                    candidates = [
                        f"tests/python/test_{module}.py",
                        f"tests/python/test_{module.split('_')[0]}.py",
                    ]
                    for c in candidates:
                        if (PROJECT_ROOT / c).exists() and c not in test_files:
                            test_files.append(c)
                elif f.startswith("tests/python/") and f.endswith(".py"):
                    # Skip quarantine — those are tests we've explicitly
                    # excluded from default discovery (e.g., known to deadlock
                    # with the rest of the smart subset). See 2026-04-28 .80
                    # debug for the test_conductor_supervisor.py incident.
                    if "/quarantine/" in f:
                        continue
                    if f not in test_files:
                        test_files.append(f)
        except Exception:
            pass

        # Filter to files that actually exist
        existing = [f for f in test_files if (PROJECT_ROOT / f).exists()]
        if not existing:
            existing = ["tests/python/test_cli.py"]

        # Pre-flight timeout: 1200s (20 min). Originally 120s, then 300s.
        # 2026-04-30: bumped from 300s after the .84 smart-subset hit
        # 32 minutes wall time on a 29-file subset. The 5-min cap was
        # SIGKILLing pytest mid-run, producing empty SKIP messages and
        # blocking exp1078 (Position Paper v2) for two consecutive
        # iterations with no diagnostic output. The smart-subset has
        # grown organically as more test files land — the timeout needs
        # to scale with it.
        #
        # If the smart-subset exceeds 20 min, the planner should split
        # individual experiment scripts (the 30+ min cost is concentrated
        # in 1-2 slow training-loop tests, not the bulk of the suite).
        rc, stdout, stderr = _pytest_run(
            [venv_pytest]
            + existing
            + ["-q", "--no-header", "-n", "0", "--no-cov", "-o", "addopts="],
            timeout=1200,
        )

    # Find the summary line + capture failed/errored test names so the
    # self-heal path has more than a count to work with. Without this the
    # conductor logs only "551 passed, 7 errors" and the operator has no
    # way to identify which tests actually failed without a re-run.
    summary = ""
    failed_names: list[str] = []
    in_short_summary = False
    for line in (stdout or stderr).splitlines():
        stripped = line.strip()
        if not summary and ("passed" in line or "failed" in line):
            summary = stripped
        if stripped.startswith("=") and "short test summary" in stripped:
            in_short_summary = True
            continue
        if in_short_summary:
            if stripped.startswith("="):
                in_short_summary = False
                continue
            if stripped.startswith(("FAILED ", "ERROR ", "FAILED\t", "ERROR\t")):
                # Lines look like: "ERROR tests/python/test_x.py::test_y - reason"
                head = stripped.split(" - ", 1)[0]
                parts = head.split(None, 1)
                if len(parts) == 2:
                    failed_names.append(f"{parts[0]} {parts[1]}")
    success = rc == 0
    if success:
        # 2026-05-01 fix (Issue 4): persist the END-of-pretest fingerprint,
        # not the START fingerprint. If files changed during the pre-test
        # run (e.g., operator commits while the conductor was busy), the
        # START fingerprint is stale by the time the run finishes — saving
        # it causes the next iteration to cache-miss because the current
        # fingerprint reflects post-commit state. Recomputing here captures
        # the actual state we just verified green.
        end_fp = _compute_pretest_fingerprint()
        if end_fp != current_fp:
            logger.info(
                "Pre-test fingerprint changed during run (%s -> %s); caching END state",
                current_fp[:12],
                end_fp[:12],
            )
        _save_pretest_cache(end_fp, summary, mode)
    elif failed_names:
        # Log up to 10 failed/errored test ids so the journal records the
        # diagnostic detail. Operators can grep journalctl for these names
        # instead of having to re-run pytest themselves.
        logger.warning(
            "Pre-test failures (showing %d of %d): %s",
            min(10, len(failed_names)),
            len(failed_names),
            "; ".join(failed_names[:10]),
        )
    return success, summary


def git_status() -> str:
    """Get git status summary."""
    _, stdout, _ = run_cmd(["git", "diff", "--stat"])
    return stdout.strip()


def git_has_changes() -> bool:
    """Check if there are uncommitted changes."""
    _, stdout, _ = run_cmd(["git", "status", "--porcelain"])
    return bool(stdout.strip())


def git_commit_and_push(message: str, push: bool = True) -> bool:
    """Stage, commit, and optionally push.

    Conductor commits use --no-verify by design. Operator directive
    2026-05-03 19:48Z: "always committing and never reverting so that
    we fail forward and fix any problems rather than lose transient
    assets." pre-commit's `staged_files_only` plugin stashes unstaged
    changes before running hooks, then restores via `git apply` if any
    hook fails. When the restore patch fails to apply (base files have
    moved, etc.), unstaged work is silently lost. Tonight's session
    observed multiple losses: pyproject.toml --ignore additions,
    openspec/change-proposals/in-situ-training-phase5-derisking.md,
    multiple changelog entries — each had to be recreated from
    conversation memory, only succeeding because content was still
    present.

    Conductor commits are "preserve work as checkpoint" — the whole
    point is not to lose work. Running hooks that might fail and
    trigger stash-loss is precisely backwards. --no-verify skips the
    stash-restore cycle entirely. Hooks still run on:
      - operator commits via `git commit` directly
      - agent-spawned commits via run_agent (subprocess shell hooks fire)
      - CI / pre-merge gates (server-side enforcement)
    so verification coverage is preserved at the right boundaries.

    See ops/known-issues.md entry "CRITICAL — Pre-Commit
    `staged_files_only` is Causing Silent Data Loss" (2026-05-03 19:50Z).
    """
    full_message = with_agent_signature(message)

    run_cmd(["git", "add", "-A"])
    rc, _, stderr = run_cmd(["git", "commit", "--no-verify", "-m", full_message])
    if rc != 0:
        logger.warning("Commit failed: %s", stderr[:200])
        return False
    logger.info("Committed (--no-verify): %s", message.splitlines()[0][:80])
    if push:
        rc, _, stderr = run_cmd(["git", "push", "origin", "main"], timeout=60)
        if rc == 0:
            logger.info("Pushed to origin")
        else:
            logger.warning("Push failed: %s", stderr[:200])
    return True


###############################################################################
# Async doc-reconciliation
#
# When --async-doc-recon is set, the post-experiment doc reconciliation (the
# 1-2 min Haiku call, or a fall-through from a failed in-process attempt)
# runs in a background thread instead of blocking the main iteration loop.
# The conductor enters its inter-iteration sleep immediately after the
# experiment commit, and the doc-recon completes during that sleep.
#
# Single-worker executor: doc-recons across iterations remain serialised so
# we never have two threads racing to commit/push at once. If a prior
# recon hasn't finished by the next iteration's start, _await_pending_recon
# blocks on it before any git operations — preventing the "preserve
# uncommitted work" sweep at iteration start from accidentally grabbing
# the in-flight recon's diff.
###############################################################################

_recon_executor: concurrent.futures.ThreadPoolExecutor | None = None
_pending_recon_future: concurrent.futures.Future | None = None
_recon_state_lock = threading.Lock()


def _ensure_recon_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Lazy-init a single-worker executor for async doc reconciliation."""
    global _recon_executor
    with _recon_state_lock:
        if _recon_executor is None:
            _recon_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="doc-recon",
            )
        return _recon_executor


def _submit_async_recon(callable_fn) -> None:
    """Submit a doc-reconciliation callable to the background executor.

    If a previous recon is still pending, block on it first — this enforces
    sequential git operations across iterations and prevents the
    "preserve uncommitted work" sweep at the next iteration's start from
    accidentally swallowing the in-flight recon's diff.
    """
    global _pending_recon_future
    executor = _ensure_recon_executor()
    # Wait for any previous recon to settle before submitting a new one,
    # so the executor's queue depth never exceeds 1 in steady state.
    _await_pending_recon()
    with _recon_state_lock:
        _pending_recon_future = executor.submit(callable_fn)
    logger.info("Async doc-reconciliation submitted to background")


def _await_pending_recon(timeout: float = 600.0) -> None:
    """Wait for any in-flight async doc reconciliation to complete.

    Called at the start of every research_step() *before* any git operation
    so the next iteration's pre-flight reaper, smart-subset pre-check, and
    "preserve uncommitted work" checkpoint don't race with a still-running
    recon thread. Also called by _submit_async_recon to enforce sequential
    git operations.

    Failures or timeouts are logged but do not raise — the conductor
    continues. The recon's commit, if any, has already been pushed
    (or not) by the time we get here; the next iteration takes over.
    """
    global _pending_recon_future
    with _recon_state_lock:
        future = _pending_recon_future
        _pending_recon_future = None
    if future is None:
        return
    try:
        future.result(timeout=timeout)
        logger.info("Pending async doc-reconciliation completed")
    except concurrent.futures.TimeoutError:
        logger.warning(
            "Pending async doc-reconciliation timed out after %.0fs; "
            "the next iteration will proceed anyway",
            timeout,
        )
    except Exception:
        logger.exception(
            "Pending async doc-reconciliation raised; the next iteration will proceed anyway"
        )


def _shutdown_recon_executor(wait: bool = True, timeout: float = 600.0) -> None:
    """Drain any pending recon and shut the executor down. Called from main()."""
    global _recon_executor
    _await_pending_recon(timeout=timeout)
    with _recon_state_lock:
        if _recon_executor is not None:
            _recon_executor.shutdown(wait=wait)
            _recon_executor = None


def _check_auroc_anomaly(task: dict) -> None:
    """Detect suspicious AUROC values in a deliverable and page the operator.

    Background — exp995 and exp1003 both shipped pathological AUROC values
    (0.0 and 1.0 respectively) for ~24h before the inverted-sign bug was
    caught. Verdicts read directionally correct, but the underlying signals
    were anti-correlated, so headline narratives were silently wrong.
    Detecting AUROC == 0 / ≈0.001 / ≈0.999 / 1.0 at deliverable-write time
    converts a 24h silent regression into an instant operator page.

    Reads ``task["deliverable"]`` relative to ``PROJECT_ROOT``, extracts the
    top-level ``auroc`` field (if present), and appends one JSON-line record
    to ``ops/supervisor-alerts.json`` when the value is at a suspicious edge.
    Normal values (anything else) are silent.

    Spec: REQ-CONDUCTOR-AUROC-ANOMALY, SCENARIO-CONDUCTOR-AUROC-1, -2.
    """
    import json as _json

    deliverable = task.get("deliverable")
    if not deliverable:
        return
    try:
        path = PROJECT_ROOT / deliverable
        if not path.exists():
            return
        data = _json.loads(path.read_text())
    except (OSError, ValueError):
        return

    auroc = data.get("auroc")
    if auroc is None or not isinstance(auroc, (int, float)):
        return

    # Edge values that historically indicated sign-error bugs or trivial
    # data, not genuine separability. The 0.001 / 0.999 epsilons absorb
    # rounding noise without paging on legitimate near-perfect results.
    epsilon = 0.005
    is_anomaly = auroc <= epsilon or auroc >= 1.0 - epsilon
    if not is_anomaly:
        return

    alerts_path = PROJECT_ROOT / "ops" / "supervisor-alerts.json"
    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "alert_type": "AUROC_ANOMALY",
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task_id": task.get("id", "unknown"),
        "deliverable": deliverable,
        "auroc": auroc,
        "detail": (
            f"AUROC={auroc} for task {task.get('id', 'unknown')} is at a "
            f"suspicious edge value. Inspect the experiment for sign errors "
            f"(see exp995/exp1003 inverted-AUROC pattern) before trusting "
            f"the verdict."
        ),
    }
    with open(alerts_path, "a") as f:
        f.write(_json.dumps(record) + "\n")


def _classify_retirement(exp_id: str, verdict: str | None) -> str:
    """Classify a task's retirement cause as 'environmental' or 'merit'.

    Per the no-permanent-retirement-on-environmental-failures policy
    (openspec/change-proposals/no-permanent-retirement-on-environmental-failures.md):

    - **environmental** = the task didn't get a fair shot. Pre-tests
      were broken, the conductor hit max-turns, an upstream gate was
      missing, the GPU was unavailable, etc. These should be respawned
      with variance applied — the experiment hypothesis hasn't been
      tested yet.
    - **merit** = the task ran cleanly and the hypothesis didn't hold
      (below baseline, no improvement, regression). Don't auto-respawn;
      the planner has to decide whether to retry with different scope.

    Conservative default: empty/None verdict → "merit" (don't auto-respawn
    when we have no signal). Compound verdicts are matched by substring,
    case-insensitive.

    Spec: REQ-CONDUCTOR-RETIRE-CLASSIFY, SCENARIO-RETIRE-1, -2.
    """
    if not verdict or not isinstance(verdict, str):
        return "merit"
    vlow = verdict.lower()

    environmental_tokens = (
        "pre_tests_failing",
        "max_turns",
        "gate_block",
        "gate_check_failed",
        "blocked_no_live_gpu",
        "blocked_prereq",
        "scaffold_only",
        "blocked_gate_check_failed",
        "envpropagation",
        # Conductor log status codes that indicate environmental failure.
        # SKIP/FAIL alone are too ambiguous in isolation, but SKIP is
        # almost always environmental (pre-test self-heal failed) and
        # FAIL with the surrounding context tokens above is too.
        "skip",
        "fail: max_turns",
        "fail: pre",
    )
    if any(tok in vlow for tok in environmental_tokens):
        return "environmental"
    return "merit"


def log_step(task: str, status: str, details: str = "") -> None:
    """Append to conductor log."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
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
    Each YAML task is preserved as-is so downstream consumers see every
    field the planner emitted — `prior_failures` (failure-ledger
    discipline), `gated_on` (deliverable-gating), `max_turns`
    (per-task budget override), `depends_on` (cross-task ordering), and
    `milestone` (archive scope) all need to flow through. The
    historical cherry-pick that kept only id/deliverable/title/prompt
    silently disabled the failure-ledger pre-launch check (which
    queries `task["prior_failures"]`) — fixed 2026-04-26.
    """
    if not ROADMAP_FILE.exists():
        logger.warning("research-roadmap.yaml not found — no tasks to run")
        return []
    try:
        with open(ROADMAP_FILE) as f:
            data = yaml.safe_load(f)
        tasks = data.get("tasks", [])
        # Pass every YAML field through. The conductor's required fields
        # (id, title, prompt) are still validated up-front via the bare
        # lookups in pick_next_task / research_step.
        result: list[dict] = []
        for t in tasks:
            if "id" not in t or "title" not in t or "prompt" not in t:
                logger.warning(
                    "Skipping malformed task (missing id/title/prompt): %s",
                    t.get("id") or t.get("title") or "<unknown>",
                )
                continue
            result.append(dict(t))  # shallow copy preserves all fields
        return result
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


def compute_adaptive_sleep_min(iter_duration_s: float, interval_min: int) -> tuple[int, str]:
    """Pick an inter-iteration sleep duration based on how much work the
    iteration actually did.

    Three tiers:
      - **short (block/skip)** — iter_duration < 30 s. The iteration was a
        doomed-rerun block, a deliverable-already-exists skip, or another
        sub-second decision. No real Sonnet work, no GPU contention, no
        downstream service to settle. Sleep ~10 % of the configured
        interval (floor 1 min) so the loop stays responsive.
      - **medium (CPU experiment)** — 30 s ≤ iter_duration < 5 min. A real
        but lightweight experiment: CPU-only, fast Sonnet, pre-flight,
        retros. The 5-min cutoff matches the upstream LLM's prompt-cache
        TTL — sleeping past 5 min costs a cache miss, so anything below
        gets a proportionally shorter sleep. Sleep ~50 % of the interval
        (floor 2 min).
      - **long (GPU/planner)** — iter_duration ≥ 5 min. A heavyweight
        experiment (GPU inference, training, planner Sonnet 50-turn).
        Real downstream load on git remotes, GPU memory, model caches.
        Sleep the full configured interval — the safety margin is real.

    Returns ``(sleep_min, tier_label)``. ``sleep_min`` is always ≥ 1.

    Tuned against the .71 operational retro finding: ~80 min of the .71
    milestone's 110 sleep-minutes were spent on doomed-rerun blocks that
    finished in < 1 sec each. Adaptive sleep would have recovered most
    of that time without changing the cache or pacing characteristics
    of real experiments.
    """
    if iter_duration_s < 30:
        return max(1, interval_min // 10), "short (block/skip)"
    if iter_duration_s < 300:
        return max(2, interval_min // 2), "medium (CPU experiment)"
    return interval_min, "long (GPU/planner)"


_BOOTSTRAP_STATUSES = frozenset({"running", "blocked", "partial", "in_progress"})


def _verdict_is_untrustworthy(payload: dict) -> tuple[bool, str | None]:
    """Return (is_untrustworthy, verdict_string).

    An artifact whose ``honest_verdict`` matches any
    ``_PARTIAL_TOKENS`` / ``_BLOCKED_TOKENS`` / ``_FAILED_TOKENS``
    substring should not be cached as a completed task — the
    experiment ran but did not satisfy its acceptance criteria.

    Per user directive 2026-04-30: "we don't want to accept only
    partially run experiments as total successes. we need to be able
    to trust our experiments for the scientific method to work."

    Reuses the verdict-token sets from in_process_doc_reconcile so
    the cache policy can't drift from milestone-retro policy. Falls
    back to a hard-coded list if the import fails.
    """
    verdict = payload.get("honest_verdict") if isinstance(payload, dict) else None
    if not isinstance(verdict, str):
        return False, None
    vlow = verdict.lower()
    # 2026-05-01 fix (Issue 7): verdicts ending in `_honest_negative` are
    # deliberately-named confirmations that the hypothesis didn't pan out —
    # a valid scientific finding, not a partially-run experiment. The
    # earlier user directive (2026-04-30) was about not accepting partial
    # runs as success; an explicitly-honest-negative result is the
    # experiment running fully and reporting truthfully. Don't re-run
    # these. Empirical .85 incident: exp1099 RLVR-SSD reported
    # `no_improvement_honest_negative` because its corpus had been
    # pre-filtered to all-zero energies (Carnot energy filter
    # tautologically accepts everything when all scores are 0). The
    # finding is correct and reproducible; cycling it 3 more times via
    # the fail-cap is wasted wall time.
    # Broadened 2026-05-01 18:00Z: recognize `honest_negative` /
    # `honest_null` / `honest_neutral` ANYWHERE in verdict, not just
    # at the end. .86 incidents: exp1108 produced
    # `and_composition_still_not_viable` (no honest_* marker — needed
    # operator rename), exp1109 produced `kl_below_threshold_simulation_only`
    # (also no marker — needed operator rename), exp1110 produced
    # `honest_negative_non_degenerate` (marker present but in middle,
    # not suffix). The original endsWith check missed the middle case.
    # 2026-05-06 fix: `_retired` is a deliberate scientific verdict per
    # CLAUDE.md "Failed-Experiment Rerun Discipline" — the experiment ran
    # fully and concluded retirement is appropriate after 3 same-verdict
    # attempts. exp1393 GRPO v8 NGRPO shipped "no_improvement_all_unknown_retired"
    # but was flagged untrustworthy because "no_improvement" matched
    # _PARTIAL_TOKENS even though the verdict explicitly closed via
    # retire-discipline. Treat as terminal honest-finding.
    HONEST_FINDING_TOKENS = (
        "honest_negative",
        "honest_null",
        "honest_neutral",
        "_retired",
    )
    if any(tok in vlow for tok in HONEST_FINDING_TOKENS):
        return False, verdict
    # 2026-05-05 fix: agents commonly prefix terminal verdicts with explicit
    # status markers ("complete: ...", "success: ...", "passed: ..."). The
    # text after the prefix often contains nuance words that match
    # _PARTIAL_TOKENS as substrings (e.g., "complete: ... DSP feasibility
    # is still marginal" matches "marginal" → false positive partial).
    # exp1305 incident: agent shipped status=complete with honest_verdict
    # "complete: conservative replay policy is useful as an operator gate,
    # but DSP feasibility is still marginal and this is not a learned
    # general stop rule" — three retries logged FAIL bootstrap-only because
    # "marginal" matched _PARTIAL_TOKENS even though the verdict explicitly
    # opens with a terminal-success marker. Prefix-trust agent's explicit
    # terminal markers when present.
    # 2026-05-06 extension: agents also use underscore-separator format
    # like "complete_prm_guided_no_improvement_prototype_no_headline_claim"
    # (exp1430 retired) and "complete_mcmc_constrained_repair_..." and
    # "complete_repair_executor_no_successful_repairs". The underscore
    # variant is just as valid a terminal marker as the colon/space
    # variant — agents pick separator by template convention, not
    # by terminality intent.
    _TERMINAL_VERDICT_PREFIXES = (
        "complete:",
        "complete ",
        "complete_",
        "success:",
        "success ",
        "success_",
        "passed:",
        "passed ",
        "passed_",
        "shipped:",
        "shipped ",
        "shipped_",
    )
    if any(vlow.startswith(p) for p in _TERMINAL_VERDICT_PREFIXES):
        return False, verdict
    # 2026-05-07 fix: positive-context "blocked" patterns. Agents writing
    # adversarial-audit / defense / safety-test verdicts often produce
    # phrases like "telemetry_claim_blocked_adversarial_audit" (exp1473)
    # where "blocked" means "the audit successfully blocked an
    # unsupported claim" — a TERMINAL GOOD outcome, not a partial run.
    # Same pattern for Sakana-defense ("attack_blocked"), capability
    # firewalls ("escape_blocked"), input validation ("injection_blocked"),
    # and bound-checking experiments ("bound_blocked_at_threshold"). These
    # are good results that the substring "blocked" alone misclassifies.
    # Without this whitelist, exp1473 (Live Telemetry Adversarial Validity
    # Audit) was retired despite shipping status=complete with a real 8KB
    # artifact reporting the audit's success. Whitelist positive-context
    # blocking patterns to override the partial-token check.
    _POSITIVE_BLOCKED_PATTERNS = (
        "_claim_blocked",  # exp1473: claim was correctly blocked by audit
        "_attack_blocked",  # Sakana-style: attack blocked = defense worked
        "_audit_blocked",  # audit blocked an unsupported claim
        "_injection_blocked",  # input validation blocked injection
        "_escape_blocked",  # capability sandbox blocked escape
        "_violation_blocked",  # constraint violation blocked
        "_unsupported_blocked",  # unsupported pattern blocked
    )
    if any(p in vlow for p in _POSITIVE_BLOCKED_PATTERNS):
        return False, verdict
    # Issue 7 extension 2026-05-01 18:50Z: verdicts that carry both a
    # *progress* token (improved/gained/above_baseline) AND a
    # *threshold-miss* token (below/under/missed) describe an honest
    # negative on a strict acceptance gate — real progress that didn't
    # clear an ambitious threshold. exp1111 incident: ThinkPRM v2 went
    # AUROC 0.9885 → 0.9946 (+0.6 points, α_t = 0.3801 satisfies Zenil
    # convergence) but missed the strict 0.995 acceptance gate; verdict
    # `auroc_improved_below_995`. Cycling it 3 more times produces the
    # same number — the work is done, the gate is the issue. Rerun
    # discipline in CLAUDE.md says reruns must address a root cause;
    # there's nothing to address — improvement is real, gate is strict.
    _PROGRESS_TOKENS = ("improved", "improvement", "gained", "above_baseline")
    _MISS_TOKENS = ("below", "under_threshold", "missed_threshold", "missed_target")
    if any(p in vlow for p in _PROGRESS_TOKENS) and any(m in vlow for m in _MISS_TOKENS):
        return False, verdict
    # Issue 3 v2 fix 2026-05-02 12:35Z: explicit-acceptance-gate token
    # override. Some experiments encode acceptance compositionally with
    # "below" or "above" pointing at threshold values (which the partial-
    # token check would otherwise flag as failure):
    #   exp1156 sampler_kl_below_05_viable      (KL < 0.5  = passing)
    #   exp1157 calibrated_tp_above_80_fpr_below_30  (TP > 80, FPR < 30 = passing)
    # The Issue 7 extension above catches `improved + below` but not the
    # `viable + below` or `calibrated + below` patterns. Both verdicts had
    # genuinely-positive artifacts that the conductor retired despite real
    # success. The token list below is the explicit-acceptance whitelist
    # that overrides partial-detection. Three retirements in 24 hours
    # (exp1118 GRPO, exp1156 HMC sampler, exp1157 SECL calibration) drove
    # the fix.
    _EXPLICIT_ACCEPTANCE_TOKENS = (
        "_viable",  # exp1156: sampler_kl_below_05_viable
        "calibrated_",  # exp1157: calibrated_tp_above_80_fpr_below_30
        "_acceptance_met",  # explicit acceptance-met marker
        "_passes_gate",  # explicit passes-gate marker
        "_within_tolerance",  # within tolerance (positive)
        "_meets_target",  # explicit meets-target marker
    )
    # _NEGATIVE_VIABLE_PATTERNS: explicit negations that contain "_viable"
    # but mean "NOT viable". Without this guard, `not_viable`,
    # `still_not_viable`, etc. would match `_viable` in the acceptance
    # whitelist above and silently pass as success. Order matters: check
    # negations BEFORE applying the acceptance override.
    _NEGATIVE_VIABLE_PATTERNS = ("not_viable", "non_viable", "unviable")
    # Defensive guard: even if an acceptance token is present, refuse to
    # override partial-detection if the verdict ALSO contains a clearly-
    # negative context word. Catches hypothetical strings like
    # `non_viable_collapse`, `calibrated_but_collapsed`, etc.
    _CLEARLY_NEGATIVE_CONTEXT = (
        "collapse",
        "broken",
        "wedged",
        "diverged",
        "garbage",
        "useless",
        "stalled_out",
        "degraded",
        "regress",
    )
    has_acceptance = any(tok in vlow for tok in _EXPLICIT_ACCEPTANCE_TOKENS)
    has_negative_viable = any(p in vlow for p in _NEGATIVE_VIABLE_PATTERNS)
    has_clearly_negative = any(p in vlow for p in _CLEARLY_NEGATIVE_CONTEXT)
    if has_acceptance and not has_negative_viable and not has_clearly_negative:
        return False, verdict
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from in_process_doc_reconcile import (  # type: ignore[import-not-found]
            _BLOCKED_TOKENS,
            _FAILED_TOKENS,
            _PARTIAL_TOKENS,
        )

        tokens = _PARTIAL_TOKENS + _BLOCKED_TOKENS + _FAILED_TOKENS
    except ImportError:
        tokens = (
            "partial",
            "inverted",
            "insufficient",
            "no_improvement",
            "still_wrong",
            "no_delta",
            "below",
            "regression",
            "negative",
            "flat",
            "plateau",
            "collapsed",
            "blocked",
            "failed",
            "timed_out",
            "exception",
            "tolerance_exceeded",
            "marginal",
            "incorrect",
        )
    if any(tok in vlow for tok in tokens):
        return True, verdict
    return False, verdict


def _deliverable_exists(task: dict) -> bool:
    """Check if a task's deliverable file already exists *and is finished*.

    Background: the original implementation returned True for any file at the
    deliverable path, which caused the .80 milestone to wedge on 2026-04-29:
    Sonnet's "CRITICAL: write artifact FIRST" defensive pattern landed
    bootstrap-only artifacts (status=running, all-fields-False) and the
    fast-path then short-circuited every retry. Downstream gated tasks read
    `False` forever. This function now reads the JSON status field and
    refuses to fast-path artifacts whose status indicates incompletion.

    Rules:
      - file missing                          -> not done (False)
      - file exists, not JSON                 -> assume done (True; legacy)
      - JSON with status in BOOTSTRAP_STATUSES -> not done (False)
      - JSON with no status field             -> assume done (True; legacy)
      - any other JSON status (e.g. success)  -> done (True)

    Test coverage: tests/python/test_conductor_deliverable_status.py.
    Change proposal: openspec/change-proposals/conductor-fastpath-bootstrap-skip.md.
    """
    deliverable = task.get("deliverable")
    if not deliverable:
        return False
    path = PROJECT_ROOT / deliverable
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return True  # legacy / non-JSON deliverables: preserve old behavior
    status = payload.get("status") if isinstance(payload, dict) else None
    if isinstance(status, str) and status.lower() in _BOOTSTRAP_STATUSES:
        logger.info(
            "Deliverable %s exists but status=%r is bootstrap-only; not skipping",
            deliverable,
            status,
        )
        return False
    if isinstance(payload, dict):
        untrust, verdict = _verdict_is_untrustworthy(payload)
        if untrust:
            logger.info(
                "Deliverable %s exists but honest_verdict=%r is partial/blocked/failed; "
                "not skipping (will re-run rather than accept partial as success)",
                deliverable,
                verdict,
            )
            return False
    return True


def _artifact_is_finished(task: dict) -> bool:
    """Return True iff the task's artifact (if any) is NOT bootstrap-only.

    Used to re-validate a prior log OK: a task may have been logged "OK"
    because the conductor's pytest self-heal passed, even though Sonnet
    short-circuited and the artifact is still status=running. Trusting the
    log OK in that case poisons the cache forever — see fast-path bootstrap
    proposal.

    Tasks without a deliverable field (planning steps, retros, doc-only
    work) trivially return True so that the log OK alone is trusted.
    """
    deliverable = task.get("deliverable")
    if not deliverable:
        return True
    path = PROJECT_ROOT / deliverable
    if not path.exists():
        return True  # no artifact yet to poison the OK; trust the log
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return True  # legacy / non-JSON: preserve old behavior
    status = payload.get("status") if isinstance(payload, dict) else None
    if isinstance(status, str) and status.lower() in _BOOTSTRAP_STATUSES:
        logger.warning(
            "Prior log OK for task %r is poisoned: artifact %s status=%r; scheduling re-run.",
            task.get("title", task.get("id", "?"))[:50],
            deliverable,
            status,
        )
        return False
    if isinstance(payload, dict):
        untrust, verdict = _verdict_is_untrustworthy(payload)
        if untrust:
            logger.warning(
                "Prior log OK for task %r is poisoned: artifact %s "
                "honest_verdict=%r (partial/blocked/failed); scheduling re-run.",
                task.get("title", task.get("id", "?"))[:50],
                deliverable,
                verdict,
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Exclusion manifest — RETRO-067 wire-in (2026-04-20)
# ---------------------------------------------------------------------------
# Exp 575 (milestone 2026.04.44) built scripts/conductor_exclusion_manifest.json
# listing five experiments (308, 260, 309, 425, 410) that appeared in the
# slowest-5 for eight consecutive milestones (.37–.44), consuming ~385 min per
# milestone (cumulative ~45 hours).  Exp 589 (milestone 2026.04.45) wrote the
# ExclusionManifest library in python/carnot/pipeline/exclusion_manifest.py
# but could NOT wire it into this file because the conductor's own guard
# reverts any subagent edit to research_conductor.py.  This block is the
# human-authored wire-in that closes RETRO-067.
#
# Caching: the manifest is loaded once at module import and re-read only if
# the JSON file's mtime changes.  The per-task lookup is O(1) via the library's
# internal set.  Manifest errors NEVER block the conductor — a malformed or
# missing file degrades gracefully to "no exclusions."
import re

_EXCLUSION_MANIFEST: object | None = None
_EXCLUSION_MANIFEST_MTIME: float = 0.0
_EXPERIMENT_ID_RE = re.compile(r"^exp(\d+)[-_]")


def _ensure_exclusion_manifest_loaded() -> None:
    """Lazy-load (and mtime-refresh) the conductor exclusion manifest."""
    global _EXCLUSION_MANIFEST, _EXCLUSION_MANIFEST_MTIME
    manifest_path = PROJECT_ROOT / "scripts" / "conductor_exclusion_manifest.json"
    try:
        current_mtime = manifest_path.stat().st_mtime if manifest_path.exists() else 0.0
    except OSError:
        current_mtime = 0.0
    if _EXCLUSION_MANIFEST is not None and current_mtime == _EXCLUSION_MANIFEST_MTIME:
        return
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "python"))
        from carnot.pipeline.exclusion_manifest import ExclusionManifest

        em = ExclusionManifest(str(manifest_path))
        em.load()
        _EXCLUSION_MANIFEST = em
        _EXCLUSION_MANIFEST_MTIME = current_mtime
    except Exception as exc:  # noqa: BLE001  — never block the loop on manifest errors
        logger.warning(
            "Exclusion manifest load failed (%s) — proceeding without exclusions",
            exc,
        )
        _EXCLUSION_MANIFEST = None


def _task_is_excluded(task: dict) -> tuple[bool, str]:
    """Return ``(is_excluded, reason)`` for a task based on the manifest.

    Extracts the integer experiment ID from the task's ``id`` field (pattern
    ``exp<N>-...``) or from the ``title`` as a fallback.  If the ID can't be
    extracted or the manifest isn't loaded, returns ``(False, "no id")`` — err
    on the side of allowing the task to run.  Exclusion is a performance
    optimisation, not a safety gate.
    """
    _ensure_exclusion_manifest_loaded()
    if _EXCLUSION_MANIFEST is None:
        return False, "manifest unavailable"
    # Try task id first (preferred: "exp308-legacy-cleanup" → 308)
    match = _EXPERIMENT_ID_RE.match(task.get("id", ""))
    if not match:
        # Fallback: title like "Exp 308: ..." or "Experiment 308 — ..."
        title = task.get("title", "")
        match = re.search(r"(?:Exp|Experiment)\s+(\d+)", title)
    if not match:
        return False, "no id parsed"
    exp_id = int(match.group(1))
    try:
        if _EXCLUSION_MANIFEST.is_excluded(exp_id):  # type: ignore[attr-defined]
            return True, f"exp_id={exp_id} in manifest"
    except Exception as exc:  # noqa: BLE001
        logger.warning("Exclusion check failed (%s) — allowing task", exc)
        return False, "check failed"
    return False, "not excluded"


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

    # 2026-05-01 fix (Issue 1): scope fail-counting to the current milestone
    # only. Previously, .85's exp1096 SemEnergy Probe inherited 4 fails from
    # .84's exp1080 SemEnergy Probe (same title) and was retired before the
    # planner's prior_failures: block could even be evaluated. Same for
    # exp1097 N-Queens vs .84's exp1086. The right semantics: when the
    # planner re-proposes a previously-retired task in a new milestone,
    # the new attempt is a NEW experiment with possibly-new approach
    # (codified via prior_failures: addressed_by) and deserves its own
    # fail-count budget, not inheritance from .84.
    #
    # Mechanic: find the most recent "Milestone X activated" line and only
    # count fails that occurred AFTER that marker. Pre-activation fails
    # belong to a prior milestone's count and have already had their cap
    # consequences play out there.
    log_lines = completed_log.splitlines()
    activation_index = -1
    for i, line in enumerate(log_lines):
        if "Milestone" in line and "activated" in line:
            activation_index = i  # Keep the LAST activation line index
    fail_counting_lines = log_lines[activation_index + 1 :] if activation_index >= 0 else log_lines

    for line in fail_counting_lines:
        parts = line.split("|")
        if len(parts) < 4:
            continue
        title = parts[2].strip()
        status = parts[3].strip()

        if status == "OK":
            completed_titles.add(title)
            fail_counts[title] = 0  # Reset on success
        elif status in ("FAIL", "REVERT", "SKIP", "NOOP", "GATE_BLOCK", "DOOMED_RERUN_BLOCK"):
            # DOOMED_RERUN_BLOCK added 2026-04-30: when the failure-ledger
            # pre-launch check rejects a task because its YAML lacks
            # prior_failures matching a scope-similar prior failure, the
            # task can never run as written. Without counting this as a
            # fail, the conductor loops forever — observed in .84 with
            # exp1080 SemEnergy Probe v1 cycling 4+ blocks at ~10 min each.
            # Same structural fix as the GATE_BLOCK count from earlier
            # today (commit 4e46ede6) — the only "variance" available is
            # editing the YAML, which the conductor can't do, so we should
            # retire after MAX_FAILURES_PER_TASK rather than loop.
            # GATE_BLOCK added 2026-04-29: when an upstream task retires
            # (hits MAX_FAILURES) without producing the artifact a downstream
            # gate references, the downstream task gate-blocks indefinitely
            # on every iteration — pick_next_task keeps picking it because
            # GATE_BLOCK was previously not counted as a failure. Observed
            # in .81: exp1039 retired (3 SKIP/FAIL), exp1044 then
            # GATE_BLOCKed on exp1039.gate_coercion_fixed every iteration
            # for ~30 min, blocking exp1045+ from being picked. Counting
            # GATE_BLOCK as a failure mode lets MAX_FAILURES retire the
            # task after 3 consecutive gate-blocks, unblocking the cascade.
            fail_counts[title] = fail_counts.get(title, 0) + 1

    # Ensure tasks are loaded from YAML
    _ensure_tasks_loaded()

    # Pre-compute the set of retired upstream task IDs so downstream tasks
    # whose gates reference them can be skipped immediately rather than
    # GATE_BLOCK-retried 3 times each.
    #
    # Why this matters (2026-04-30 incident): exp1050-pretest-surgery
    # retired after 3 fails. Three downstream tasks gated on its
    # pre_tests_fixed artifact (exp1051, exp1052, exp1053) then each
    # GATE_BLOCK-cycled 3 times = 9 wasted iterations × ~10 min each
    # = ~90 min of pure overhead before the cascade settled. Velocity
    # collapsed to 0 successes per 2.5 hours in milestone .82.
    retired_task_ids: set[str] = set()
    for task in RESEARCH_TASKS:
        title_prefix = task["title"][:50].strip()
        if fail_counts.get(title_prefix, 0) >= MAX_FAILURES_PER_TASK:
            tid = task.get("id")
            if tid:
                retired_task_ids.add(tid)

    # Find first task not yet completed AND not failed too many times
    for task in RESEARCH_TASKS:
        # Strip to match .strip() applied to parsed log entries — otherwise
        # titles whose 50-char cut lands on whitespace (e.g. Exp 447) have
        # a trailing space in the lookup key and never match their fail rows,
        # which caused an infinite retry loop in milestone 2026.04.33.
        title_prefix = task["title"][:50].strip()

        # Signal 1: log says OK — but re-validate against artifact status. A
        # prior OK can be poisoned by Sonnet's "CRITICAL: write artifact FIRST"
        # bootstrap pattern: the conductor's pytest self-heal passed (logged
        # OK) but Sonnet hit max-turns before updating the artifact's status
        # field from "running" → "success". Without this guard, the task is
        # forever marked done while downstream gates read the bootstrap
        # `false` fields and GATE_BLOCK indefinitely. See
        # openspec/change-proposals/conductor-fastpath-bootstrap-skip.md.
        if title_prefix in completed_titles and _artifact_is_finished(task):
            continue

        # Signal 2: deliverable already exists (built but not logged)
        if _deliverable_exists(task):
            logger.info("Task '%s' deliverable exists — marking complete", title_prefix)
            log_step(title_prefix, "OK", "Deliverable already exists in repo")
            continue

        # Signal 3: exclusion manifest (RETRO-067 wire-in, 2026-04-20)
        # Five legacy experiments (308/260/309/425/410) are permanently
        # excluded because they consumed ~385 min/milestone for 8 consecutive
        # milestones with no research benefit.  See scripts/conductor_exclusion_manifest.json.
        excluded, reason = _task_is_excluded(task)
        if excluded:
            logger.info("Task '%s' excluded by manifest (%s) — skipping", title_prefix, reason)
            log_step(title_prefix, "OK", f"Excluded by manifest: {reason}")
            continue

        # Signal 4: too many failures
        if fail_counts.get(title_prefix, 0) >= MAX_FAILURES_PER_TASK:
            logger.warning(
                "Skipping '%s' — failed %d times", title_prefix, fail_counts[title_prefix]
            )
            continue

        # Signal 4.5: upstream gate target retired. Pre-emptively skip
        # rather than GATE_BLOCK-retrying 3 times. See retired_task_ids
        # comment block above for the .82 incident this prevents.
        gated_on = task.get("gated_on") or []
        retired_upstreams = [
            g.get("upstream")
            for g in gated_on
            if isinstance(g, dict) and g.get("upstream") in retired_task_ids
        ]
        if retired_upstreams:
            logger.warning(
                "Skipping '%s' — upstream(s) retired: %s",
                title_prefix,
                ", ".join(retired_upstreams),
            )
            log_step(
                title_prefix,
                "GATE_BLOCK",
                f"Pre-emptive skip: upstream retired ({', '.join(retired_upstreams)})",
            )
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
        "completed": datetime.now(UTC).strftime("%Y-%m-%d"),
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
            next_milestone,
            len(next_tasks),
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
        )
        run_cmd(["git", "commit", "-m", with_agent_signature(msg)])
        if push:
            run_cmd(["git", "push", "origin", "main"], timeout=60)

        log_step(f"Milestone {next_milestone} activated", "OK", f"{len(next_tasks)} tasks queued")
        return True

    except Exception as e:
        logger.error("Failed to activate next roadmap: %s", e)
        return False


def _update_docs_before_planning(push: bool = True) -> bool:
    """Update docs, technical report, and GitHub pages before planning.

    Runs before the planning agent to ensure documentation reflects the
    latest experiment results. The planning agent then reads up-to-date
    docs when designing the next milestone.
    """
    logger.info("=" * 60)
    logger.info("UPDATING DOCS BEFORE PLANNING")
    logger.info("=" * 60)

    doc_prompt = (
        f"You are working on the Carnot EBM framework in {PROJECT_ROOT}.\n"
        f"Read CLAUDE.md for project context.\n\n"
        f"TASK: Update ALL documentation to reflect the latest experiment results.\n"
        f"This runs BEFORE the planning agent, so docs must be current.\n\n"
        f"READ FIRST:\n"
        f"- ops/status.md — current experiment count and results\n"
        f"- ops/changelog.md — recent experiments\n"
        f"- research-complete.yaml — all completed milestones\n\n"
        f"UPDATE THESE FILES:\n"
        f"1. docs/index.html — update stats (experiment count, test count),\n"
        f"   results cards with latest numbers, capabilities if new ones added\n"
        f"2. README.md — update experiment count, key results table\n"
        f"3. docs/technical-report.md — update abstract AND header with latest\n"
        f"   experiment count, key results, milestone count. Add new sections\n"
        f"   for any major new findings not yet documented.\n"
        f"4. docs/technical-report.html — FULLY RE-RENDER from the updated\n"
        f"   technical-report.md. The HTML must match the markdown content.\n"
        f"   Keep the same dark theme CSS, nav bar, and footer styling.\n\n"
        f"RULES:\n"
        f"- Update numbers, results, and add new findings sections\n"
        f"- Keep changes minimal and focused on accuracy\n"
        f"- Do NOT modify scripts/research_conductor.py\n"
        f"- Do NOT push\n"
    )

    success, output = run_agent(doc_prompt, max_turns=30, timeout=600)

    if success and git_has_changes():
        run_cmd(["git", "add", "-A"])
        msg = with_agent_signature(
            "[conductor] Update docs before planning — sync with latest results"
        )
        run_cmd(["git", "commit", "-m", msg])
        if push:
            run_cmd(["git", "push", "origin", "main"], timeout=60)
        logger.info("Docs updated before planning")
        return True

    logger.info("No doc updates needed (or update failed)")
    return False


def _run_operational_retrospective(push: bool = True) -> bool:
    """Run an operational retrospective at milestone boundary.

    Evaluates HOW the milestone was executed, not just WHAT it produced.
    Identifies bottlenecks, resource waste, and process improvements.
    Feeds suggestions into the next milestone planning.

    This is Tier 1 self-learning applied to the research process itself:
    the system gets better at running experiments, not just at verification.
    """
    try:
        with open(ROADMAP_FILE) as f:
            roadmap = yaml.safe_load(f) or {}
        current = roadmap.get("milestone", "unknown")
    except Exception:
        current = "unknown"
    logger.info("=" * 60)
    logger.info("OPERATIONAL RETROSPECTIVE (milestone %s)", current)
    logger.info("=" * 60)

    # Gather timing data from git log
    try:
        _, git_log, _ = run_cmd(
            [
                "git",
                "log",
                "--format=%H %ai %s",
                "--grep=\\[conductor\\]",
                "--since=7 days ago",
            ]
        )
        experiment_times: list[dict] = []
        commits = git_log.strip().splitlines()
        prev_time = None
        for line in reversed(commits):
            parts = line.split(maxsplit=3)
            if len(parts) < 4:
                continue
            # Parse timestamp (format: 2026-04-12 08:35:46 -0400)
            try:
                ts_str = f"{parts[1]} {parts[2]}"
                from datetime import datetime as _dt

                ts = _dt.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                msg = parts[3] if len(parts) > 3 else ""
                if prev_time and "Exp " in msg:
                    duration_min = (ts - prev_time).total_seconds() / 60
                    experiment_times.append(
                        {
                            "experiment": msg[:80],
                            "duration_min": round(duration_min, 1),
                        }
                    )
                prev_time = ts
            except (ValueError, IndexError):
                continue

    except Exception:
        experiment_times = []

    # Gather GPU utilization data
    gpu_report_text = ""
    try:
        from gpu_monitor import format_report, generate_report

        gpu_report = generate_report()
        gpu_report_text = format_report(gpu_report)
    except Exception:
        gpu_report_text = "GPU monitor not available"

    # Build the retrospective prompt
    timing_summary = ""
    if experiment_times:
        total_min = sum(e["duration_min"] for e in experiment_times)
        slowest = sorted(experiment_times, key=lambda x: x["duration_min"], reverse=True)[:5]
        timing_summary = (
            f"Total milestone wall time: {total_min:.0f} minutes ({total_min / 60:.1f} hours)\n"
            f"Experiments completed: {len(experiment_times)}\n"
            f"Average per experiment: {total_min / len(experiment_times):.0f} minutes\n"
            f"Slowest experiments:\n"
        )
        for e in slowest:
            timing_summary += f"  - {e['duration_min']:.0f}min: {e['experiment']}\n"

    retro_prompt = (
        f"You are working on the Carnot EBM framework in {PROJECT_ROOT}.\n\n"
        f"TASK: Write an operational retrospective for milestone {current}.\n\n"
        f"STEP 0 (MANDATORY, FIRST): Immediately write a SKELETON artifact JSON to\n"
        f"   results/operational_retro_{current.replace('.', '_')}.json with:\n"
        f"     {{\n"
        f"       \"schema\": \"carnot.operational_retro.v63\",\n"
        f"       \"milestone\": \"{current}\",\n"
        f"       \"generated_at\": \"<current ISO-8601 UTC>\",\n"
        f"       \"retro_type\": \"operational_in_progress\",\n"
        f"       \"summary\": \"in progress — being filled in this turn\",\n"
        f"       \"slowest_experiments\": [],\n"
        f"       \"bottlenecks_identified\": [],\n"
        f"       \"improvements_suggested\": [],\n"
        f"       \"top_3_highest_leverage_actions\": [],\n"
        f"       \"meta_reflection\": \"\"\n"
        f"     }}\n"
        f"   This protects against turn-budget exhaustion: even if you run out\n"
        f"   of turns mid-analysis, the artifact exists at status='success'\n"
        f"   with whatever you completed. Then refine its contents in subsequent\n"
        f"   turns. The conductor's _artifact_is_finished check will accept the\n"
        f"   skeleton if you don't get to refine it; loss of detail is acceptable;\n"
        f"   loss of the entire artifact (status=running stuck) is not.\n\n"
        f"This is NOT about research results — it's about how EFFICIENTLY\n"
        f"the milestone was executed. Analyze bottlenecks and suggest\n"
        f"improvements for the next milestone.\n\n"
        f"TIMING DATA:\n{timing_summary}\n\n"
        f"GPU STATE:\n{gpu_report_text}\n\n"
        f"QUESTIONS TO ANSWER:\n"
        f"1. Which experiments took the longest and why?\n"
        f"2. Was GPU utilization efficient? (sequential vs parallel)\n"
        f"3. Were there zombie processes wasting resources?\n"
        f"4. Could any experiments have been parallelized?\n"
        f"5. Was the pre-flight test suite a bottleneck?\n"
        f"6. Were doc reconciliation passes efficient?\n"
        f"7. What tooling/infrastructure changes would speed up the next milestone?\n\n"
        f"EXISTING FILES TO READ:\n"
        f"- ops/metrics.md — session metrics\n"
        f"- ops/changelog.md — recent experiment log\n"
        f"- scripts/gpu_monitor.py — GPU resource monitor\n\n"
        f"DELIVERABLES:\n"
        f"1. Write results/operational_retro_{current.replace('.', '_')}.json with:\n"
        f"   - total_wall_time_minutes\n"
        f"   - experiments_completed\n"
        f"   - slowest_experiments (top 5)\n"
        f"   - bottlenecks_identified (list of strings)\n"
        f"   - improvements_suggested (list of strings)\n"
        f"   - estimated_time_savings_pct (how much faster next milestone could be)\n"
        f"2. Append a brief summary to ops/changelog.md\n"
        f"3. Append ONE row to the 'Completed Milestones' table in docs/roadmap.md\n"
        f"   for milestone {current}. Format: | {current} | <one-line theme> | <exp range> | <key breakthrough> |\n"
        f"   Do NOT delete or rewrite existing rows. Do NOT touch the 'Current Milestone' block\n"
        f"   (that updates at activation, not retro). If docs/roadmap.md does not exist or the\n"
        f"   table heading is missing, skip silently — do not create the file structure from scratch.\n"
        f"4. Do NOT modify scripts/research_conductor.py or research-roadmap.yaml.\n"
    )

    logger.info("Calling agent for operational retrospective...")
    # Retrospective benefits from Opus-class honest self-evaluation (anti-
    # sycophancy + anti-scheming training makes it less likely to paper over
    # failures). Set AGENT_MODEL_RETRO=opus to enable; defaults to Sonnet.
    # max_turns 15 → 60 (operator fix 2026-05-03 13:55Z): heavy retros (12+
    # experiments to read + cascade-pattern analysis + structured JSON write)
    # don't fit in 15-turn budget; .92 retro retired 3× and .93 retro is on
    # path to retire as artifact_not_updated_past_bootstrap. STEP 0 skeleton
    # write (added to retro_prompt above) is belt-and-braces against budget
    # exhaustion; longer max_turns is the suspenders.
    success, output = run_agent(
        retro_prompt,
        max_turns=60,
        model_override=AGENT_MODEL_RETRO,
        agent_type_override=AGENT_TYPE_RETRO,
    )

    if success:
        logger.info("Operational retrospective complete")
        if git_has_changes():
            git_commit_and_push(
                f"[conductor] Operational retrospective for milestone {current}", push=push
            )
        return True
    else:
        logger.warning("Operational retrospective failed — continuing")
        # Clean up any partial changes
        if git_has_changes():
            run_cmd(["git", "checkout", "."])
            run_cmd(["git", "clean", "-fd", "--exclude=.coverage*"])
        return False


def _plan_next_milestone(push: bool = True) -> bool:
    """Ask the configured agent to plan the next research milestone.

    When all current tasks are done AND no research-roadmap-next.yaml exists,
    this function asks the configured agent to analyze completed work, the PRD, and the
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
1. research-program.md — HIGH-LEVEL GOALS AND PRIORITIES (start here)
2. _bmad/prd.md — long-term vision and requirements
3. _bmad/architecture.md — current architecture
4. ops/status.md — what's working, what's next
5. ops/changelog.md — recent work
6. research-complete.yaml — all completed experiments and findings
7. research-roadmap.yaml — the milestone that just finished
8. openspec/change-proposals/ — all previous roadmap docs
9. ops/conductor-log.md — per-experiment results
10. research-references.md — technologies and ideas to consider
11. research-hardware-wishlist.md — available and desired hardware

THEN DO RESEARCH (arxiv + other sources):
Search these sources for recent work (2025-2026) relevant to Carnot:

PRIMARY — arxiv.org:
- Energy-Based Models for verification/reasoning
- Constraint satisfaction with neural networks
- Ising model applications in ML
- LLM hallucination detection and mitigation
- Kolmogorov-Arnold Networks
- Energy-guided decoding / constrained generation
- Hardware-accelerated sampling (FPGA, thermodynamic computing)
- Continual/online learning for constraint systems

SECONDARY — also check:
- OpenReview.net — NeurIPS/ICML/ICLR submissions on EBMs, constrained decoding
- extropic.ai/writing — Extropic TSU hardware updates
- Semantic Scholar — papers CITING our key references (EBT arxiv:2507.02092,
  ARM-EBM bijection arxiv:2512.15605)
- HuggingFace papers (huggingface.co/papers) — verification/hallucination work
- GitHub trending repos — new EBM/constraint/KAN implementations (Python+Rust)
- logicalintelligence.com — Kona architecture updates

Add any promising findings to research-references.md before designing experiments.
This research phase ensures we stay current and don't miss accelerating ideas.

THEN DESIGN THE MILESTONE:

MILESTONE NAMING: Use CalVer format YYYY.MM.XX where:
- YYYY = current year (2026)
- MM = current month (use the ACTUAL current month, NOT a projected future month)
- XX = sequential number starting from 01, incrementing within the month
- Example: if current month is April and last milestone was 2026.04.18,
  the next is 2026.04.19 (NOT 2026.05.xx or 2026.06.xx)
- Only increment MM when the calendar month actually changes

1. Identify the 3 biggest gaps between current state and PRD vision
2. Incorporate any promising arxiv findings as experiments
3. Determine the natural next experiments based on completed work
4. Design 10-14 experiments across 3-4 phases
5. Use Qwen3.5-0.8B and google/gemma-4-E4B-it as the target LLM models
   (latest small SoTA — do NOT propose older models like Llama/Phi)
6. Ensure at least one experiment targets continuous self-learning
   (see research-program.md "Continuous Self-Learning" section)

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

    # Planner benefits from Opus-class synthesis (big-context design of 12-13
    # coherent experiments). Set AGENT_MODEL_PLANNER=opus to enable; defaults to Sonnet.
    success, output = run_agent(
        planning_prompt,
        max_turns=50,
        timeout=1200,
        model_override=AGENT_MODEL_PLANNER,
        agent_type_override=AGENT_TYPE_PLANNER,
    )

    if not success:
        logger.error("Planning agent failed: %s", output[:200])
        log_step("Plan next milestone", "FAIL", f"{AGENT_DISPLAY} error: {output[:60]}")
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
            next_milestone,
            len(next_tasks),
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
        "completed": datetime.now(UTC).strftime("%Y-%m-%d"),
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
            next_milestone,
            len(next_tasks),
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
        )
        run_cmd(["git", "commit", "-m", with_agent_signature(msg)])
        if push:
            run_cmd(["git", "push", "origin", "main"], timeout=60)

        log_step(f"Milestone {next_milestone} activated", "OK", f"{len(next_tasks)} tasks queued")
        return True

    except Exception as e:
        logger.error("Failed to activate next roadmap: %s", e)
        return False


def _plan_next_milestone(push: bool = True) -> bool:
    """Ask the configured agent to plan the next research milestone.

    When all current tasks are done AND no research-roadmap-next.yaml exists,
    this function asks the configured agent to analyze completed work and propose the next
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
        f"10. research-references.md — technologies and ideas to consider\n"
        f"11. research-hardware-wishlist.md — available and desired hardware\n\n"
        f"THEN DO RESEARCH (arxiv + other sources):\n"
        f"Search these sources for recent work (2025-2026) relevant to Carnot:\n\n"
        f"PRIMARY — arxiv.org:\n"
        f"- Energy-Based Models for verification/reasoning\n"
        f"- Constraint satisfaction with neural networks\n"
        f"- Ising model applications in ML\n"
        f"- LLM hallucination detection and mitigation\n"
        f"- Kolmogorov-Arnold Networks\n"
        f"- Energy-guided decoding / constrained generation\n"
        f"- Hardware-accelerated sampling (FPGA, thermodynamic computing)\n"
        f"- Continual/online learning for constraint systems\n\n"
        f"SECONDARY — also check:\n"
        f"- OpenReview.net — NeurIPS/ICML/ICLR submissions on EBMs\n"
        f"- extropic.ai/writing — TSU hardware updates\n"
        f"- Semantic Scholar — papers citing EBT (2507.02092) and ARM-EBM (2512.15605)\n"
        f"- HuggingFace papers (huggingface.co/papers) — verification work\n"
        f"- GitHub trending — new EBM/constraint/KAN repos\n"
        f"- logicalintelligence.com — Kona architecture updates\n\n"
        f"Add any promising findings to research-references.md before designing "
        f"experiments. This ensures we stay current and don't miss ideas.\n\n"
        f"THEN DESIGN THE MILESTONE:\n"
        f"1. Identify the 3 biggest gaps between current state and PRD vision\n"
        f"2. Incorporate any promising arxiv findings as experiments\n"
        f"3. Determine the natural next experiments based on completed work\n"
        f"4. Design 10-14 experiments across 3-4 phases\n"
        f"5. Use the mandated SOTA local GGUF models (CLAUDE.md):\n"
        f"   - unsloth/Qwen3.6-35B-A3B-GGUF (flagship MoE)\n"
        f"   - unsloth/gemma-4-31B-it-GGUF (flagship dense)\n"
        f"   - unsloth/gemma-4-26B-A4B-it-GGUF (middle MoE)\n"
        f"   Each new experiment that needs an LLM MUST include at least one of\n"
        f"   these in its MODEL_SPECS. The legacy small models (Qwen3.5-0.8B,\n"
        f"   gemma-4-E4B-it) are acceptable ONLY as fast CPU smoke-tests, not as\n"
        f"   headline-result models — see scripts/experiment_template.py docstring\n"
        f"   for the cached_sota_pair() pattern.\n"
        f"6. Ensure at least one experiment targets continuous self-learning\n"
        f"   (see research-program.md 'Continuous Self-Learning' section)\n\n"
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
        f"OPTIONAL YAML FIELDS — POPULATE WHEN APPLICABLE:\n\n"
        f"The conductor reads two optional fields per task that activate runtime\n"
        f"speedups (see scripts/conductor_gates.py).  Populating them lets simple\n"
        f"experiments fail fast and downstream-gated experiments skip their\n"
        f"5-9 min Sonnet call entirely when the prerequisite verdict doesn't\n"
        f"satisfy the gate.\n\n"
        f"  gated_on: [optional list]\n"
        f"    Use when this task depends on a previous experiment's artifact\n"
        f"    field meeting a condition.  Example for a task whose title says\n"
        f"    'gated on Exp NNN delta>0':\n\n"
        f"        gated_on:\n"
        f"          - upstream: expNNN-the-prereq-task-id\n"
        f"            artifact_field: delta_overall\n"
        f"            op: '>'\n"
        f"            value: 0.0\n\n"
        f"    Supported ops: ==, !=, >, >=, <, <=, in, not_in, contains,\n"
        f"    not_contains.  All gates conjunctive (AND).\n"
        f"    Always populate this field when the task title says 'gated on\n"
        f"    Exp NNN ...' — translate the natural-language gate into a\n"
        f"    structured op/value pair.  Without the structured field, the\n"
        f"    Sonnet call still runs and the experiment script does the gate\n"
        f"    check internally — wasteful when we could skip the call.\n\n"
        f"  max_turns: [optional int, 1-100, default 50]\n"
        f"    Lower the budget for simple experiments.  Recommended values:\n"
        f"      - 20 — pure-diagnostic / no-new-source experiments (RETRO\n"
        f"             audits, retirement plans, retros, doc-only updates)\n"
        f"      - 30 — wiring / configuration / single-method-add experiments\n"
        f"      - 50 — default; full Sonnet budget for non-trivial work\n"
        f"    Do NOT increase above 50 unless the task genuinely needs deeper\n"
        f"    Sonnet thinking (Phase 3 architecture work, complex retros).\n\n"
        f"  model: [optional, 'sonnet' | 'opus', default sonnet via AGENT_MODEL]\n"
        f"    Pre-emptive Opus routing for tasks that consistently exhaust\n"
        f"    Sonnet's max-turns budget. The C+E (Sonnet→Opus) escalation\n"
        f"    pattern handles max-turns failures reactively, but tasks in the\n"
        f"    categories below should skip the wasted Sonnet attempt and route\n"
        f"    directly to Opus.\n\n"
        f"    Set `model: opus` (and increase max_turns to 100) for:\n"
        f"      - Hardware integration: FPGA bring-up, ROCm probes, KV260\n"
        f"        boards, dual-GPU work, nvidia-smi fallbacks\n"
        f"      - Schema / preflight infrastructure: schema-validation,\n"
        f"        manifest retirement, gate-cascade fixes, conductor patches\n"
        f"      - Multi-step coordination: tasks bundling (manifest retire +\n"
        f"        pretest fix + GPU probe) into a single experiment\n"
        f"      - Bootstrap-and-bail risk: any prompt instructing 'CRITICAL:\n"
        f"        write artifact FIRST' — these are the bootstrap-only\n"
        f"        artifact wedge class observed in milestone .80\n\n"
        f"    Empirically, ~10–15% of tasks per milestone need Opus rescue.\n"
        f"    Pre-classifying them avoids ~3 min of wasted Sonnet wall-clock\n"
        f"    per task plus the bootstrap-only artifact failure mode that\n"
        f"    cascade-blocks downstream experiments.\n\n"
        f"    Routine experiments (single-question evaluation, training\n"
        f"    loops with established pipelines, deliverable-already-exists\n"
        f"    fast-paths) keep the default Sonnet — those succeed >95%.\n\n"
        f"  agent_type: [optional, 'claude' | 'codex' | 'gemini' | 'opencode']\n"
        f"    Per-task agent backend selection (orthogonal to `model`). The\n"
        f"    conductor defaults to AGENT_TYPE=claude for synthesis-heavy\n"
        f"    work. Use this field to route specific task categories to their\n"
        f"    strongest backend. Multi-agent routing per\n"
        f"    openspec/change-proposals/multi-agent-routing.md.\n\n"
        f"    Set `agent_type: codex` + `model: gpt-5.5` for FORMULAIC CODE:\n"
        f"      - WOPR-games-gallery cartridges (Sudoku, Lights Out,\n"
        f"        N-Queens, Connect Four, Hex, Slitherlink, Hashi, etc.)\n"
        f"      - New verifier implementations (constraint encoding follows\n"
        f"        well-known patterns: Z3, SAT, graph coloring, etc.)\n"
        f"      - Test scaffolding (after Claude designs the module, use\n"
        f"        Codex to generate the comprehensive test suite)\n"
        f"      - PyO3 / Rust binding boilerplate\n"
        f"      - Sampler / MCMC implementations (well-documented Bayesian)\n"
        f"      - Dataset generation pipelines (FoVer expansion, Z3 labeling)\n\n"
        f"    Set `agent_type: gemini` + `model: gemini-3.1-pro-preview` for\n"
        f"    LONG-CONTEXT WORK (1M token window):\n"
        f"      - Failure-ledger pattern detection across milestone history\n"
        f"        (feed entire research-complete.yaml + conductor logs)\n"
        f"      - Architecture coherence audits (Phase-3..7 chain + outline)\n"
        f"      - Multi-paper literature synthesis (3-5 papers full text)\n"
        f"      - Multimodal verification (FPGA bitstream / oscilloscope —\n"
        f"        future)\n\n"
        f"    Keep DEFAULT (Claude, no agent_type) for SYNTHESIS / JUDGMENT:\n"
        f"      - Routine experiments (most tasks)\n"
        f"      - Retros / milestone-N analysis\n"
        f"      - Planning / roadmap design / hardware integration\n"
        f"      - Position paper drafting / multi-file coordination\n\n"
        f"    CAVEAT: Gemini Deep Think (the deeper extended-reasoning mode\n"
        f"    used for Phase-3 → Phase-7 architectural derivation) is NOT in\n"
        f"    the standard Gemini API as of 2026-04-29 — only via consumer\n"
        f"    Gemini app or early-access program. agent_type=gemini routes\n"
        f"    to standard Gemini API thinking, comparable to Claude extended\n"
        f"    thinking but NOT Deep Think.\n\n"
        f"    NOTE: When agent_type is set, C+E (Sonnet→Opus) escalation is\n"
        f"    skipped for that task because the escalation logic is\n"
        f"    Claude-specific (max-turns signal is a Claude-CLI output).\n\n"
        f"IMPORTANT:\n"
        f"- Do NOT modify research-roadmap.yaml\n"
        f"- Do NOT modify scripts/research_conductor.py\n"
        f"- Do NOT push\n"
        f"- CalVer milestones: increment the seq number\n"
        f"- Each experiment must have a clear deliverable file path\n"
    )

    # Planner benefits from Opus-class synthesis (big-context design of 12-13
    # coherent experiments). Set AGENT_MODEL_PLANNER=opus to enable; defaults to Sonnet.
    success, output = run_agent(
        planning_prompt,
        max_turns=50,
        timeout=1200,
        model_override=AGENT_MODEL_PLANNER,
        agent_type_override=AGENT_TYPE_PLANNER,
    )

    if not success:
        logger.error("Planning agent failed: %s", output[:200])
        log_step("Plan next milestone", "FAIL", f"{AGENT_DISPLAY} error: {output[:60]}")
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
            next_milestone,
            len(next_tasks),
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
        )
        git_commit_and_push(msg, push=push)

    log_step(f"Plan milestone {next_milestone}", "OK", f"{len(next_tasks)} tasks proposed")
    return True


def _load_dogfood_memory() -> dict:
    """Load persistent dogfood memory from disk.

    Tracks patterns across conductor restarts: which experiments fail,
    which brace patterns recur, which file types need extra attention.
    """
    if DOGFOOD_MEMORY_FILE.exists():
        try:
            return json.loads(DOGFOOD_MEMORY_FILE.read_text())
        except Exception:
            return {
                "patterns": {},
                "brace_fixes": 0,
                "code_violations": 0,
                "experiments_checked": 0,
            }
    return {"patterns": {}, "brace_fixes": 0, "code_violations": 0, "experiments_checked": 0}


def _save_dogfood_memory(memory: dict) -> None:
    """Persist dogfood memory to disk."""
    try:
        DOGFOOD_MEMORY_FILE.write_text(json.dumps(memory, indent=2))
    except Exception as e:
        logger.debug("DOGFOOD: Failed to save memory: %s", e)


def _dogfood_verify_generated_code() -> None:
    """Use Carnot's own pipeline to verify code the conductor just generated.

    This is the dogfooding step: we eat our own cooking. Runs checks
    on any new/modified Python files and persists learned patterns:

    1. Brace validation — catch {key} patterns in YAML prompts that will
       break .format(). This was our #1 recurring bug.
    2. CodeExtractor — static constraint extraction (types, bounds, returns)
       on new .py files. Catches issues pytest might miss.
    3. Z3ArithmeticExtractor — formal verification of any arithmetic claims
       in generated code comments/docstrings (zero false positives).
    4. LLMConstraintExtractor — LLM-based extraction for natural language
       claims in docstrings and comments (best precision, 1/91 FP).
    5. Constraint tracker update — record what we find for future learning.
    """
    try:
        # 1. Validate YAML prompt braces in roadmap files
        for yaml_file in ["research-roadmap.yaml", "research-roadmap-next.yaml"]:
            yaml_path = PROJECT_ROOT / yaml_file
            if not yaml_path.exists():
                continue
            try:
                import yaml as _yaml

                with open(yaml_path) as f:
                    data = _yaml.safe_load(f)
                # 2026-05-01 fix: only auto-fix when at least one prompt
                # actually fails .format(). Previously this code would
                # unconditionally rewrite every `{X}` in the YAML, even
                # when the prompts were syntactically valid — e.g., the
                # task instructions for exp1097 N-Queens (`Spin in {0,1}`),
                # exp1098 Potts (`q ∈ {0,1,2}`), and exp1099 RLVR-SSD
                # (`{question, response, correct, ...}`) describe sets
                # and dict literals respectively, NOT format placeholders,
                # and never needed fixing. The previous fix also converted
                # `{X}` to `(X)`, which is a category error: sets and
                # open-intervals are mathematically distinct objects.
                #
                # New behavior: gate rewrite on actual KeyError/ValueError
                # from .format(). If no prompt fails, leave the YAML
                # alone. If any prompt fails, ESCAPE the offending braces
                # by doubling them ({X} → {{X}}). After .format(), the
                # double-brace collapses back to a literal `{X}` — the
                # original semantics are preserved.
                had_brace_error = False
                for t in data.get("tasks", []):
                    prompt = t.get("prompt", "")
                    try:
                        prompt.format(project_root="/test", date="20260101")
                    except (KeyError, ValueError) as e:
                        had_brace_error = True
                        logger.warning(
                            "DOGFOOD: Brace error in %s task %s: %s — auto-fixing",
                            yaml_file,
                            t.get("id", "?"),
                            e,
                        )

                # Only rewrite the YAML if at least one prompt actually
                # fails .format(). Skip the rewrite otherwise.
                if had_brace_error:
                    raw = yaml_path.read_text()
                    import re as _re

                    def _fix_braces(m):
                        inner = m.group(1)
                        # Skip already-escaped braces (don't double-escape).
                        # Skip the two known format placeholders.
                        if inner in ("project_root", "date"):
                            return m.group(0)
                        return "{{" + inner + "}}"

                    # Only match single braces, not already-escaped doubles.
                    fixed_raw = _re.sub(r"(?<!\{)\{([^{}]+)\}(?!\})", _fix_braces, raw)
                    if fixed_raw != raw:
                        yaml_path.write_text(fixed_raw)
                        logger.info("DOGFOOD: Auto-fixed brace escaping in %s", yaml_file)
            except Exception as e:
                logger.debug("DOGFOOD: YAML check skipped: %s", e)

        # 2. Run CodeExtractor on new Python files
        _, new_files, _ = run_cmd(["git", "diff", "--name-only", "--diff-filter=A"])
        py_files = [
            f.strip()
            for f in new_files.splitlines()
            if f.strip().endswith(".py") and "test" not in f.lower()
        ]
        if py_files:
            try:
                from carnot.pipeline.extract import CodeExtractor

                extractor = CodeExtractor()
                for py_file in py_files[:5]:  # Limit to 5 files
                    path = PROJECT_ROOT / py_file
                    if not path.exists():
                        continue
                    code = path.read_text()
                    constraints = extractor.extract(code, domain="code")
                    violations = [c for c in constraints if c.metadata.get("satisfied") is False]
                    if violations:
                        logger.info(
                            "DOGFOOD: CodeExtractor found %d violations in %s",
                            len(violations),
                            py_file,
                        )
                        for v in violations[:3]:
                            logger.info("  - %s", v.description[:100])
            except ImportError:
                logger.debug("DOGFOOD: CodeExtractor not available, skipping")
            except Exception as e:
                logger.debug("DOGFOOD: CodeExtractor check failed: %s", e)

        # 3. Run Z3ArithmeticExtractor on new Python files
        z3_violations = 0
        if py_files:
            try:
                from carnot.pipeline.z3_extractor import Z3ArithmeticExtractor

                z3_ext = Z3ArithmeticExtractor()
                for py_file in py_files[:5]:
                    path = PROJECT_ROOT / py_file
                    if not path.exists():
                        continue
                    code = path.read_text()
                    z3_results = z3_ext.extract(code)
                    z3_bad = [r for r in z3_results if r.metadata.get("satisfied") is False]
                    if z3_bad:
                        z3_violations += len(z3_bad)
                        logger.info(
                            "DOGFOOD: Z3 found %d violations in %s",
                            len(z3_bad),
                            py_file,
                        )
                        for v in z3_bad[:3]:
                            logger.info("  - Z3: %s", v.description[:100])
            except ImportError:
                logger.debug("DOGFOOD: Z3ArithmeticExtractor not available")
            except Exception as e:
                logger.debug("DOGFOOD: Z3 check failed: %s", e)

        # 4. Run LLMConstraintExtractor on new Python files (docstrings only)
        llm_violations = 0
        if py_files:
            try:
                from carnot.pipeline.llm_extractor import LLMConstraintExtractor

                llm_ext = LLMConstraintExtractor()
                for py_file in py_files[:3]:  # Limit to 3 (LLM calls are slower)
                    path = PROJECT_ROOT / py_file
                    if not path.exists():
                        continue
                    code = path.read_text()
                    # Extract docstrings only to avoid false positives on code
                    import ast

                    try:
                        tree = ast.parse(code)
                        docstrings = []
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                                ds = ast.get_docstring(node)
                                if ds:
                                    docstrings.append(ds)
                        if docstrings:
                            combined = "\n".join(docstrings)
                            llm_results = llm_ext.extract(combined)
                            llm_bad = [
                                r for r in llm_results if r.metadata.get("satisfied") is False
                            ]
                            if llm_bad:
                                llm_violations += len(llm_bad)
                                logger.info(
                                    "DOGFOOD: LLM extractor found %d violations in %s",
                                    len(llm_bad),
                                    py_file,
                                )
                    except SyntaxError:
                        pass  # Skip files that don't parse
            except ImportError:
                logger.debug("DOGFOOD: LLMConstraintExtractor not available")
            except Exception as e:
                logger.debug("DOGFOOD: LLM extractor check failed: %s", e)

        # 5. Persist learned patterns to disk
        memory = _load_dogfood_memory()
        memory["experiments_checked"] = memory.get("experiments_checked", 0) + 1

        # Record brace fix pattern
        if "fixed_raw" in dir() and fixed_raw != raw:
            memory["brace_fixes"] = memory.get("brace_fixes", 0) + 1
            # Track which tasks had braces
            brace_tasks = memory.get("brace_fix_tasks", [])
            brace_tasks.append(yaml_file if "yaml_file" in dir() else "unknown")
            memory["brace_fix_tasks"] = brace_tasks[-50:]  # Keep last 50

        # Record code violations (all extractors combined)
        code_v = (
            len(
                [
                    c
                    for c in (constraints if "constraints" in dir() else [])
                    if c.metadata.get("satisfied") is False
                ]
            )
            if py_files
            else 0
        )
        total_violations = code_v + z3_violations + llm_violations
        memory["code_violations"] = memory.get("code_violations", 0) + total_violations
        memory["z3_violations"] = memory.get("z3_violations", 0) + z3_violations
        memory["llm_violations"] = memory.get("llm_violations", 0) + llm_violations

        _save_dogfood_memory(memory)
        logger.info(
            "DOGFOOD: Verification complete (total: %d checks, %d brace fixes, "
            "%d code/%d z3/%d llm violations)",
            memory["experiments_checked"],
            memory.get("brace_fixes", 0),
            code_v,
            z3_violations,
            llm_violations,
        )

    except Exception as e:
        # Never let dogfooding block the pipeline
        logger.debug("DOGFOOD: Skipped due to error: %s", e)


def research_step(
    push: bool = True,
    dry_run: bool = False,
    in_process_docs: bool = False,
    async_doc_recon: bool = False,
) -> bool:
    """Execute one research step. Returns True if progress was made.

    in_process_docs: when True, run the mechanical Python reconciler
    (scripts/in_process_doc_reconcile.py) in place of the Haiku doc
    reconciliation call. Saves ~1-2 min per iteration. Honest-verdict
    mapping is identical; freeform "research finding" prose is omitted.

    async_doc_recon: when True, the post-experiment Haiku doc reconciliation
    (or the in-process fallback when in-process raises) runs in a background
    thread so the iteration completes the moment the experiment commit is
    pushed. The next iteration's first action is to wait on any still-running
    recon. Saves 1-2 min per iteration on the Haiku path.
    """
    # CRITICAL: drain any prior async doc-reconciliation before touching git.
    # The "preserve uncommitted work" sweep below would otherwise grab the
    # in-flight recon's diff and attribute it to this iteration's checkpoint.
    _await_pending_recon()
    timestamp = datetime.now(UTC)

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
            # No next roadmap — first update docs, then plan next milestone.
            # This ensures documentation reflects latest results before
            # the planning agent reads them to design the next milestone.
            logger.info("Updating docs before planning next milestone...")
            if not dry_run:
                _update_docs_before_planning(push=push)

            # Run operational retrospective before planning
            logger.info("Running operational retrospective...")
            if not dry_run:
                _run_operational_retrospective(push=push)

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
        logger.info("[DRY RUN] Would run %s with prompt:", AGENT_DISPLAY)
        logger.info(
            "  %s",
            task["prompt"][:200].format(
                project_root=PROJECT_ROOT,
                date=timestamp.strftime("%Y%m%d"),
            ),
        )
        return True

    # Pre-gate check (cheap, ~50ms): if the task declares `gated_on:` in
    # the roadmap YAML and any prerequisite experiment's artifact fails
    # the gate condition, write a blocked artifact directly and skip the
    # 5-9 min Sonnet call entirely. Tasks without `gated_on` pass
    # vacuously, so the existing roadmap continues working unchanged.
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from conductor_gates import (  # type: ignore[import-not-found]
            evaluate_gates as _eval_gates,
        )
        from conductor_gates import (
            write_blocked_artifact as _write_blocked,
        )

        gate_check = _eval_gates(task, results_dir=PROJECT_ROOT / "results")
        if not gate_check.passed:
            logger.warning(
                "Pre-gate check FAILED: %s — writing blocked artifact and skipping Sonnet",
                gate_check.summary,
            )
            blocked_path = _write_blocked(
                task,
                gate_check,
                results_dir=PROJECT_ROOT / "results",
            )
            if blocked_path is not None:
                logger.info("Blocked artifact written: %s", blocked_path.name)
                # Commit the blocked artifact so the in-process reconciler
                # (or the Haiku doc-recon path) can pick it up downstream.
                if git_has_changes():
                    for guarded in [
                        "scripts/research_conductor.py",
                        "research-roadmap.yaml",
                    ]:
                        _, gdiff, _ = run_cmd(
                            ["git", "diff", "--name-only", "--", guarded],
                        )
                        if gdiff.strip():
                            run_cmd(["git", "checkout", "--", guarded])
                    git_commit_and_push(
                        f"[conductor] Pre-gate block: {task['title']}\n\n{gate_check.summary}\n",
                        push=push,
                    )
                log_step(task["title"], "GATE_BLOCK", gate_check.summary)
                return True
            # write_blocked returned None — task id wasn't parseable; fall
            # through to the Sonnet path so the experiment script's own
            # gate logic still runs.
            logger.warning(
                "Pre-gate detected failure but could not write blocked "
                "artifact; falling through to Sonnet"
            )
    except Exception:
        logger.exception("Pre-gate check raised; falling through to Sonnet (defensive)")

    # Failed-experiment rerun-discipline check (cheap, ~100ms): scans
    # results/ for prior failures whose scope matches this task's scope.
    # Per CLAUDE.md "Failed-Experiment Rerun Discipline": a rerun must
    # carry a `prior_failures:` field with experiment_id + verdict +
    # addressed_by + retire_if_same_verdict populated. Without it, refuse
    # to launch and write blocked_doomed_rerun_no_root_cause.
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from failure_ledger import (  # type: ignore[import-not-found]
            FailureLedger as _FailureLedger,
        )

        ledger = _FailureLedger.load_from_artifacts(PROJECT_ROOT)
        rerun_check = ledger.is_doomed_rerun(task)
        if rerun_check.blocked:
            logger.warning(
                "Failed-experiment rerun-discipline check FAILED: %s",
                rerun_check.reason,
            )
            # Reuse the conductor_gates write_blocked_artifact helper —
            # produces a properly-formed artifact with all required
            # fields. We synthesize a GateCheckResult-like object since
            # both check types end in the same downstream path.
            from conductor_gates import (  # type: ignore[import-not-found]
                GateCheckResult as _GateCheckResult,
            )
            from conductor_gates import (
                GateResult as _GateResult,
            )
            from conductor_gates import (
                write_blocked_artifact as _write_blocked,
            )

            synthetic_gate_check = _GateCheckResult(
                passed=False,
                gates_evaluated=[
                    _GateResult(
                        upstream=p.experiment_id,
                        artifact_field="rerun_discipline",
                        op="prior_failures_required",
                        expected="non-empty entry naming this prior",
                        actual=task.get("prior_failures"),
                        passed=False,
                        reason=f"{p.status_label}: {p.verdict[:80]}",
                    )
                    for p in rerun_check.matched_priors
                ],
                summary=rerun_check.reason,
            )
            blocked_path = _write_blocked(
                task,
                synthetic_gate_check,
                results_dir=PROJECT_ROOT / "results",
            )
            if blocked_path is not None:
                logger.info(
                    "Doomed-rerun blocked artifact written: %s",
                    blocked_path.name,
                )
                if git_has_changes():
                    for guarded in [
                        "scripts/research_conductor.py",
                        "research-roadmap.yaml",
                    ]:
                        _, gdiff, _ = run_cmd(
                            ["git", "diff", "--name-only", "--", guarded],
                        )
                        if gdiff.strip():
                            run_cmd(["git", "checkout", "--", guarded])
                    git_commit_and_push(
                        f"[conductor] Doomed-rerun block: {task['title']}\n\n"
                        f"{rerun_check.reason}\n",
                        push=push,
                    )
                log_step(task["title"], "DOOMED_RERUN_BLOCK", rerun_check.reason)
                return True
            logger.warning(
                "Doomed-rerun detected but could not write blocked artifact; "
                "falling through to Sonnet"
            )
    except Exception:
        logger.exception(
            "Failed-experiment rerun-discipline check raised; falling through to Sonnet (defensive)"
        )

    # Preserve any dirty state from previous interrupted runs by committing it.
    # Previous behavior (git checkout -- .) destroyed uncommitted experiment
    # deliverables when claude -p was killed mid-run. Now we commit everything
    # as a checkpoint so nothing is lost.
    if git_has_changes():
        _, porcelain, _ = run_cmd(["git", "diff", "--name-only"])
        _, untracked, _ = run_cmd(["git", "ls-files", "--others", "--exclude-standard"])
        changed = [f.strip() for f in porcelain.splitlines() if f.strip()]
        new_files = [f.strip() for f in untracked.splitlines() if f.strip()]
        all_dirty = changed + new_files

        # Filter out files we never want to commit
        skip = {".coverage", ".pytest_cache"}
        committable = [f for f in all_dirty if not any(f.startswith(s) for s in skip)]

        if committable:
            logger.info(
                "Committing %d dirty files as checkpoint (preserving work)", len(committable)
            )
            for f in committable:
                run_cmd(["git", "add", "--", f])
            msg = with_agent_signature(
                "[conductor] Checkpoint: preserve uncommitted work from interrupted run"
            )
            run_cmd(["git", "commit", "-m", msg])

    # Reap stale GPU processes BEFORE tests.  Pre-flight test runs themselves
    # can OOM if a prior experiment left zombie workers pinning VRAM.  This
    # also protects the downstream research experiment from starting in a
    # half-broken state.  See preflight_gpu_reap() for safety rails.
    preflight_gpu_reap()

    # Run tests first — ensure clean state
    tests_ok, test_summary = run_tests()
    if not tests_ok:
        # Self-heal: ask the configured agent to fix the pre-existing test failures
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

        # MAX_HEAL_ATTEMPTS = 2 → 0 (operator emergency 2026-05-03 ~11:45Z)
        # Reason: self-heal was burning ~40 min agent runtime per retiring task
        # (2 attempts × 600s × 2 cycles for heal+verify). Schema-drifted tests
        # in the suite (test_experiment_337/355/368/692/1033 and likely more)
        # cannot be repaired by codex/sonnet/opus within turn budget; the
        # heal loop is a quota-burn pattern.
        # When pre-tests fail, just SKIP the task and let MAX_FAILURES_PER_TASK
        # retire it cleanly. Test-suite repair is properly a dedicated milestone
        # task, not an in-loop auto-fix. See ops/known-issues.md "Schema-drifted
        # tests" entry.
        # To re-enable self-heal once tests are clean: change 0 back to 2.
        MAX_HEAL_ATTEMPTS = 0
        healed = False
        for heal_attempt in range(MAX_HEAL_ATTEMPTS):
            logger.info("Self-heal attempt %d/%d", heal_attempt + 1, MAX_HEAL_ATTEMPTS)
            heal_ok, heal_output = run_agent(heal_prompt, max_turns=30, timeout=600)
            if not heal_ok:
                logger.error("%s failed during self-heal: %s", AGENT_DISPLAY, heal_output[:200])
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
                        "[conductor] Self-heal: fix pre-existing test failures\n\n",
                        push=push,
                    )
                healed = True
                break
            else:
                logger.warning(
                    "Self-heal attempt %d failed: %s", heal_attempt + 1, test_summary[:200]
                )
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

    # GPU resource check — detect zombies and log suggestions
    try:
        from gpu_monitor import generate_report, kill_zombies

        gpu_report = generate_report()
        if gpu_report.warnings:
            for w in gpu_report.warnings:
                logger.warning("GPU: %s", w)
            # Auto-kill zombies to free GPU memory for the experiment
            killed = kill_zombies(gpu_report, dry_run=False)
            if killed:
                logger.info("GPU: Killed %d zombie processes, freed GPU memory", len(killed))
        if gpu_report.suggestions:
            for s in gpu_report.suggestions:
                logger.debug("GPU suggestion: %s", s)
    except Exception as e:
        logger.debug("GPU monitor skipped: %s", e)

    # Format the prompt with project root.
    # NOTE: Prompts in YAML must use {{...}} to escape literal braces.
    # The old auto-fixer was removed because it conflicted with proper escaping.
    raw_prompt = task["prompt"]
    try:
        prompt = raw_prompt.format(
            project_root=PROJECT_ROOT,
            date=timestamp.strftime("%Y%m%d"),
        )
    except (KeyError, IndexError, ValueError) as e:
        # 2026-05-01 fix: also catch ValueError ("unmatched '{' in format spec")
        # which fires when a prompt embeds LaTeX-like nested braces (e.g.
        # `\author{Ian \texttt{x@y}}`). exp1116 incident: planner Sonnet wrote
        # an arXiv-submission prompt with raw \author{...\texttt{...}}; raw
        # str.format() crashed the iteration before fallback could run. The
        # fallback's str-replace leaves other braces alone, which is what the
        # downstream agent actually wants for LaTeX prompts.
        logger.warning("Prompt format error in %s: %s — using raw prompt", task.get("id", "?"), e)
        prompt = raw_prompt.replace("{project_root}", str(PROJECT_ROOT)).replace(
            "{date}", timestamp.strftime("%Y%m%d")
        )

    # Inject focused AST context to save tokens.
    # Instead of the agent reading entire files, we extract class/function
    # signatures from files mentioned in the prompt's "EXISTING CODE TO READ"
    # section. This gives the agent the API surface without full file contents.
    try:
        import ast as _ast
        import re as _re2

        # Find file paths mentioned in the prompt
        file_refs = _re2.findall(
            r"(?:python/carnot/[^\s]+\.py|scripts/[^\s]+\.py)",
            prompt,
        )
        if file_refs:
            outlines: list[str] = []
            total_symbols = 0
            for ref in file_refs[:6]:  # Limit to 6 files
                path = PROJECT_ROOT / ref
                if not path.exists():
                    continue
                try:
                    tree = _ast.parse(path.read_text())
                    sigs: list[str] = []
                    for node in _ast.walk(tree):
                        if isinstance(node, _ast.ClassDef):
                            bases = (
                                ", ".join(
                                    _ast.dump(b) if not hasattr(b, "id") else b.id
                                    for b in node.bases
                                )
                                if node.bases
                                else ""
                            )
                            ds = _ast.get_docstring(node) or ""
                            first_line = ds.split("\n")[0][:100] if ds else ""
                            sigs.append(f"class {node.name}({bases}):  # {first_line}")
                            for item in node.body:
                                if isinstance(item, _ast.FunctionDef):
                                    args = ", ".join(a.arg for a in item.args.args)
                                    sigs.append(f"    def {item.name}({args})")
                        elif isinstance(node, _ast.FunctionDef) and node.col_offset == 0:
                            args = ", ".join(a.arg for a in node.args.args)
                            ds = _ast.get_docstring(node) or ""
                            first_line = ds.split("\n")[0][:100] if ds else ""
                            sigs.append(f"def {node.name}({args}):  # {first_line}")
                    if sigs:
                        outlines.append(f"# {ref}\n" + "\n".join(sigs))
                        total_symbols += len(sigs)
                except Exception:
                    pass
            if outlines:
                context_block = (
                    "\n\nCODE SIGNATURES (AST-extracted — read full files only if needed):\n\n"
                    + "\n\n".join(outlines)
                )
                prompt = prompt + context_block
                logger.info(
                    "Injected %d symbol signatures from %d files (AST outlines)",
                    total_symbols,
                    len(outlines),
                )
    except Exception as e:
        logger.debug("AST context injection skipped: %s", e)

    # Prepend mandatory workflow instructions for non-Claude agents.
    # Claude reads CLAUDE.md automatically; other agents need explicit
    # instructions to follow the spec-anchored development workflow.
    if AGENT_TYPE != "claude":
        workflow_preamble = (
            "MANDATORY WORKFLOW — you MUST follow these steps:\n"
            "1. Read AGENTS.md and CODEX.md (or OPENCODE.md) for repo instructions\n"
            "2. Spec first: ensure relevant openspec/capabilities/*/spec.md has REQ-*\n"
            "3. Write tests FIRST that reference REQ-* or SCENARIO-*\n"
            "4. Implement the code to satisfy the tests\n"
            "5. Run: .venv/bin/pytest tests/python -q — ALL tests must pass\n"
            "6. Verify 100% test coverage on new code\n"
            "7. Update ops/changelog.md and ops/status.md\n"
            "8. Do NOT skip tests. Do NOT skip coverage. Do NOT just run scripts.\n\n"
        )
        prompt = workflow_preamble + prompt

    # Stop-when-done postamble — applies to ALL agents.  Every task prompt in
    # the roadmap YAML builds toward writing a specific results/experiment_NNN_*.json
    # deliverable.  Once that file is written and the tests for the *new* code
    # pass, the research value is locked in.  Earlier milestones saw subagents
    # spend 40–55 min per run after the deliverable was already on disk,
    # iterating on 100% coverage, docstring polish, and ops/*.md touch-ups.
    # The conductor's post-commit Haiku reconciliation step handles those doc
    # updates in ~2 min, so further subagent turns are pure overhead.  This
    # postamble tells the subagent explicitly to stop after the deliverable
    # stabilises, and pairs with the deliverable-watch kill-switch in
    # run_agent() as a belt-and-braces solution.
    deliverable_hint = task.get("deliverable") or "results/experiment_*.json"
    stop_postamble = (
        "\n\n=== STOP-WHEN-DONE RULE ===\n"
        f"The point of this task is to produce a valid {deliverable_hint} with "
        "all required schema fields, plus the module and tests that back it up.\n"
        "Once (a) the deliverable JSON is written and stable, and (b) the tests "
        "you added pass, STOP immediately — do not keep iterating.  In particular:\n"
        "- Do NOT re-run the full test suite multiple times chasing 100% coverage "
        "across pre-existing code.  Cover only the code you added.\n"
        "- Do NOT polish docstrings beyond the verbose-layman baseline required "
        "by CLAUDE.md on your first pass.\n"
        "- Do NOT update ops/changelog.md, ops/status.md, or _bmad/traceability.md — "
        "the conductor runs a separate Haiku reconciliation step immediately after "
        "you exit that handles all doc/status/traceability updates.  Touching those "
        "files just creates a merge with the reconciler.\n"
        "- Do NOT perform a self-review revision cycle once the deliverable is valid.\n"
        "If you finish the real work inside 10 minutes, that is correct and expected — "
        "exit promptly.  The conductor rewards short, focused runs."
    )
    prompt = prompt + stop_postamble

    # Run the configured agent.
    # Per-experiment model override via YAML "model:" field — Opus for complex
    # Phase 3 / infrastructure work, Sonnet for routine scaffolding. Absence of
    # the field falls through to AGENT_MODEL (default Sonnet).
    task_model = task.get("model")
    # Per-experiment agent_type override via YAML "agent_type:" field. Routes
    # the task to a specific CLI backend (claude/codex/gemini/opencode)
    # regardless of the conductor's startup AGENT_TYPE. Multi-agent routing
    # per openspec/change-proposals/multi-agent-routing.md.
    task_agent_type = task.get("agent_type")
    # 2026-05-03 23:30Z operator directive: enforce codex on experiments when
    # weekly Claude quota is constrained. Operator at 85% with 2.5 days to
    # reset; the .95 planner Sonnet emitted agent_type:claude on 11/13 tasks
    # despite the 2026-05-02 codex-default memory directive (planner cannot
    # read user-memory). When CODEX_FORCE_EXPERIMENTS=1 is set, coerce
    # per-task claude → codex unless the task carries an explicit
    # `requires_claude: true` flag (reserved for tasks that genuinely need
    # Claude's tool ergonomics — e.g., complex multi-file refactors). The
    # planner and retro call sites at lines ~2361/2495/2873 are NOT affected
    # — those paths use AGENT_TYPE_PLANNER / AGENT_TYPE_RETRO env overrides
    # directly and bypass this per-task coercion.
    if (
        os.environ.get("CODEX_FORCE_EXPERIMENTS") == "1"
        and task_agent_type == "claude"
        and not task.get("requires_claude")
    ):
        logger.warning(
            "CODEX_FORCE_EXPERIMENTS=1: coercing task %r agent_type "
            "claude → codex (operator quota directive 2026-05-03 23:30Z; "
            "set requires_claude:true to bypass)",
            task.get("id", "?"),
        )
        task_agent_type = "codex"
    # Per-experiment max_turns hint via YAML "max_turns:" field. Default 50
    # mirrors the historical hard-coded value; simple experiments (CPU-only
    # retros, doc passes, configuration changes) can opt into a smaller budget
    # to free quota and shave wall time. Bounds-checked in select_max_turns.
    try:
        from conductor_gates import (
            select_max_turns as _select_max_turns,  # type: ignore[import-not-found]
        )

        task_max_turns = _select_max_turns(task)
    except ImportError:
        task_max_turns = 100
    success, output = run_agent(
        prompt,
        max_turns=task_max_turns,
        timeout=1200,
        model_override=task_model,
        deliverable_path=task.get("deliverable"),
        agent_type_override=task_agent_type,
    )

    # Tiered Opus escalation on max-turns failure (2026-04-28).
    # When the configured (Sonnet by default) agent exits because it ran
    # out of turns, the experiment is capacity-bound, not logic-bound:
    # it was making progress but the budget was too small. Retry once
    # with Opus and a 100-turn cap. Cost is ~5–10× per fail, but a single
    # recovered experiment unblocks downstream cascade — see the .79→.80
    # spiral where Preflight v29/v30 + FoVer Corpus v2 max-turns'd and
    # cascade-blocked 8+ downstream slots. Per-task opt-out via
    # `escalate_on_max_turns: false` in the YAML.
    #
    # NOTE on multi-agent: C+E escalation is Claude-specific (Sonnet→Opus).
    # When agent_type=codex or =gemini, we skip the escalation — those
    # backends have their own retry semantics and the "Reached max turns"
    # signal is a Claude-CLI-specific output string. Future work: per-agent
    # escalation policies (e.g. codex gpt-5.5 → gpt-5.5-extended-thinking).
    if (
        not success
        and task_model != "opus"
        and (task_agent_type or AGENT_TYPE) == "claude"
        and "Reached max turns" in output
        and task.get("escalate_on_max_turns", False)  # Default flipped True→False 2026-05-03 ~14:55Z (quota emergency: 76% used, 3d to reset). Opus-100 retry burns $2-5/escalation × 5-7 escalations/milestone = $10-35 of claude quota that we can't afford this week. Tasks that would have escalated now just FAIL — they retire normally, get re-proposed in next milestone with better task definition. Set escalate_on_max_turns: true on individual high-leverage tasks (Phase-4 anchors, paper-v6 critical) if needed. Re-flip to True after Wednesday noon reset.
    ):
        logger.warning(
            "%s hit max-turns (%d); escalating to Opus 100 turns",
            AGENT_DISPLAY,
            task_max_turns,
        )
        log_step(
            task["title"],
            "ESCALATE_OPUS",
            f"Sonnet max-turns at {task_max_turns}; retrying with Opus 100 turns",
        )
        success, output = run_agent(
            prompt,
            max_turns=100,
            timeout=1800,
            model_override="opus",
            deliverable_path=task.get("deliverable"),
        )

    # Opus-budget-extension on max-turns failure (2026-04-30).
    # When a task is pre-routed to Opus via differential agent routing
    # but the planner picked an under-budgeted max_turns (e.g. 50),
    # the C+E pattern above doesn't help — model is already opus, so
    # the FIRST escalation arm is skipped. The task hits max-turns and
    # FAILs with no further recovery attempt. Velocity collapse on
    # exp1050-pretest-surgery-respawn-queue (.82 first task, load-bearing)
    # produced 1 FAIL + 2 SKIPs before retiring.
    if (
        not success
        and task_model == "opus"
        and task_max_turns < 100
        and (task_agent_type or AGENT_TYPE) == "claude"
        and "Reached max turns" in output
        and task.get("escalate_on_max_turns", False)  # Default flipped True→False 2026-05-03 ~14:55Z (quota emergency: 76% used, 3d to reset). Opus-100 retry burns $2-5/escalation × 5-7 escalations/milestone = $10-35 of claude quota that we can't afford this week. Tasks that would have escalated now just FAIL — they retire normally, get re-proposed in next milestone with better task definition. Set escalate_on_max_turns: true on individual high-leverage tasks (Phase-4 anchors, paper-v6 critical) if needed. Re-flip to True after Wednesday noon reset.
    ):
        logger.warning(
            "Opus hit max-turns (%d) on pre-routed task; retrying with 100 turns",
            task_max_turns,
        )
        log_step(
            task["title"],
            "ESCALATE_OPUS_100",
            f"Opus max-turns at {task_max_turns}; retrying with 100 turns",
        )
        success, output = run_agent(
            prompt,
            max_turns=100,
            timeout=1800,
            model_override="opus",
            deliverable_path=task.get("deliverable"),
        )

    if not success:
        logger.error("%s failed: %s", AGENT_DISPLAY, output[:200])
        log_step(task["title"], "FAIL", f"{AGENT_DISPLAY} error: {output[:60]}")
        return False

    # Check if the agent made any changes
    if not git_has_changes():
        logger.info("%s made no file changes", AGENT_DISPLAY)
        log_step(task["title"], "FAIL", "No file changes produced")
        return True

    # Show what changed
    diff = git_status()
    logger.info("Changes:\n%s", diff[:500])

    # Guard: never let the agent modify the conductor itself
    _, conductor_diff, _ = run_cmd(
        ["git", "diff", "--name-only", "--", "scripts/research_conductor.py"]
    )
    if conductor_diff.strip():
        logger.warning("%s modified research_conductor.py — reverting that file", AGENT_DISPLAY)
        run_cmd(["git", "checkout", "--", "scripts/research_conductor.py"])

    # ── Dogfooding: use Carnot to verify generated code ──────────
    _dogfood_verify_generated_code()

    # Run tests after changes — progressive escalation across fix attempts.
    # Tier 1 (Haiku,  30 turns): cheapest — catches simple coverage gaps and
    #                            missing exports.  Most of the time this is enough.
    # Tier 2 (Sonnet, 50 turns): default model with a larger turn budget than
    #                            the research step itself.  Catches middle-
    #                            complexity logic bugs.  Raised from 30 to 50
    #                            after observing Exp 453/472/491 max-turns
    #                            failures on fix attempts that were otherwise
    #                            making steady progress.
    # Tier 3 (Opus,   30 turns): opt-in via CARNOT_ENABLE_OPUS_FIX=1.  Fires
    #                            only when the previous attempt exited via
    #                            max-turns (fix_ok=False), i.e. the agent was
    #                            genuinely running out of reasoning capacity,
    #                            not just producing bad code that stronger
    #                            reasoning would likely repeat.  The cost is
    #                            ~5–10× Sonnet in tokens but one success here
    #                            beats a broken-tests checkpoint that blocks
    #                            downstream pre-flights for the rest of the
    #                            milestone.
    OPUS_FIX_ENABLED = os.environ.get("CARNOT_ENABLE_OPUS_FIX") == "1"
    MAX_FIX_ATTEMPTS = 3 if OPUS_FIX_ENABLED else 2
    tests_ok, test_summary = run_tests(full=False)  # Use smart subset — full suite hangs serially

    prev_fix_ok = True  # Tracks whether the *previous* fix attempt exited cleanly
    for fix_attempt in range(MAX_FIX_ATTEMPTS):
        if tests_ok:
            break
        logger.warning(
            "Tests FAILED (attempt %d/%d): %s",
            fix_attempt + 1,
            MAX_FIX_ATTEMPTS,
            test_summary[:200],
        )

        # Feed the test failure back to the configured agent to fix
        fix_prompt = (
            f"You are working on the Carnot EBM framework in {PROJECT_ROOT}.\n\n"
            f"Your previous changes caused test failures:\n{test_summary}\n\n"
            f"Fix the failing tests. Do NOT revert your changes — fix the code "
            f"so all tests pass with 100% coverage.\n"
            f"Do NOT modify scripts/research_conductor.py."
        )
        # Pick the model and turn budget for this attempt.  See the tier
        # comment above this loop for rationale.
        if AGENT_TYPE != "claude":
            fix_model = None
            fix_max_turns = 30
        elif fix_attempt == 0:
            fix_model = "haiku"
            fix_max_turns = 30
        elif fix_attempt == 1:
            fix_model = None  # default model (sonnet)
            fix_max_turns = 50
        else:
            # fix_attempt == 2 and OPUS_FIX_ENABLED is True.  Only escalate to
            # Opus when the Sonnet attempt ran out of turns — that indicates
            # it was making progress but needed more capacity.  Otherwise
            # Opus is more likely to produce a fancier version of the same
            # broken patch.
            if not prev_fix_ok:
                fix_model = "opus"
                fix_max_turns = 30
                logger.info(
                    "Opus fix tier armed (prior Sonnet attempt exited "
                    "via max-turns — capacity-bound, not logic-bound)"
                )
            else:
                logger.info(
                    "Skipping Opus fix tier: prior attempt exited "
                    "cleanly but still produced failing tests, so "
                    "the problem is code quality not reasoning depth"
                )
                break
        logger.info(
            "Asking %s to fix test failures (tier %d, model=%s, turns=%d)...",
            AGENT_DISPLAY,
            fix_attempt + 1,
            fix_model or "default",
            fix_max_turns,
        )
        fix_ok, fix_output = run_agent(
            fix_prompt,
            max_turns=fix_max_turns,
            timeout=600,
            model_override=fix_model,
        )
        prev_fix_ok = fix_ok
        if not fix_ok:
            logger.error("%s failed to fix tests (attempt %d)", AGENT_DISPLAY, fix_attempt + 1)
            # Do NOT break here when opus escalation is enabled and we are
            # between Sonnet and Opus — the capacity-bound signal is exactly
            # what the Opus tier is designed to catch.
            if fix_attempt + 1 >= MAX_FIX_ATTEMPTS:
                break
            if not OPUS_FIX_ENABLED:
                break
            # Otherwise fall through to the next iteration so the Opus tier
            # gets its turn.  The model-selection block above uses prev_fix_ok
            # to decide whether Opus is actually warranted.
            continue
        tests_ok, test_summary = run_tests(
            full=False
        )  # Use smart subset — full suite hangs serially

    if not tests_ok:
        logger.error(
            "Tests still failing after %d fix attempts — committing as broken checkpoint",
            MAX_FIX_ATTEMPTS,
        )
        # Commit the broken state as a checkpoint instead of reverting.
        # This preserves experiment deliverables even when tests fail.
        if git_has_changes():
            run_cmd(["git", "add", "-A"])
            msg = with_agent_signature(f"[conductor] Checkpoint (tests failing): {task['title']}")
            run_cmd(["git", "commit", "-m", msg])
        log_step(task["title"], "FAIL", f"Post-tests failed: {test_summary}")
        return False

    logger.info("Post-check: %s", test_summary)

    # Commit and push
    commit_msg = (
        f"[conductor] {task['title']}\n\n"
        f"Automated research step by research conductor.\n"
        f"Task ID: {task['id']}\n\n"
    )
    git_commit_and_push(commit_msg, push=push)

    # Post-commit reconciliation: ask the configured agent to update docs.
    #
    # HONEST-VERDICT MAPPING (RETRO-063 fix, 2026-04-20)
    # --------------------------------------------------
    # Prior versions of this prompt let haiku invent its own status label by
    # looking at the commit message and inferring "success".  That produced a
    # systematic rubber-stamp pattern:
    #   Exp 544: honest_verdict='tolerance_exceeded' → haiku wrote "✅ Complete"
    #   Exp 556: honest_verdict='real_data_improvement' (AUC 1.0→1.0) → "retro_058_resolved=true"
    #   Exp 564: honest_verdict='retro_061_partial'   → "RETRO-061 closed"
    #   Exp 566: honest_verdict='loss_redesign_partial' (val_auc<0.5) → "RETRO-060 FIXED"
    # Each rubber-stamp misled downstream planning agents and polluted the spec
    # base with REQs for capabilities that did not actually work.  The fix is
    # mechanical: require the subagent to read the experiment artifact, extract
    # the `honest_verdict` field, and map it via the whitelist below.  No
    # interpretation — the artifact is the source of truth.
    logger.info("Running post-commit documentation reconciliation...")

    # In-process path: skip the Haiku call entirely and run the mechanical
    # Python reconciler. Saves ~1-2 min per iteration. The reconciler reads
    # the artifact, maps honest_verdict -> status label using the same
    # whitelist the Haiku prompt uses, and appends to ops/changelog.md
    # (always), _bmad/traceability.md (when new REQ-*/SCENARIO-* in
    # commit), and ops/status.md (only on a clear win with new REQ-*).
    if in_process_docs:
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
            from in_process_doc_reconcile import reconcile  # type: ignore[import-not-found]

            recon = reconcile(task)
            logger.info(
                "In-process docs: changelog=%s, status=%s, traceability_rows=%d, "
                "verdict=%r, label=%r%s",
                recon.changelog_appended,
                recon.status_appended,
                recon.traceability_rows_added,
                recon.verdict,
                recon.status_label,
                f", skipped={recon.skipped_reason}" if recon.skipped_reason else "",
            )
            if git_has_changes():
                for guarded in ["scripts/research_conductor.py", "research-roadmap.yaml"]:
                    _, gdiff, _ = run_cmd(["git", "diff", "--name-only", "--", guarded])
                    if gdiff.strip():
                        run_cmd(["git", "checkout", "--", guarded])
                git_commit_and_push(
                    f"[conductor] In-process docs for {task['title']}\n\n",
                    push=push,
                )
                logger.info("Documentation reconciliation committed (in-process)")
            else:
                logger.info("No doc changes from in-process reconciler")
            _log_experiment_completion(task, test_summary)
            return True
        except Exception:
            logger.exception("In-process doc reconciliation failed; falling back to Haiku")
            # Fall through to the Haiku path below.

    if async_doc_recon:
        # Submit the Haiku reconciliation to the background executor and
        # return immediately. The next iteration's _await_pending_recon at
        # the very top of research_step() blocks until this completes
        # *before* any git operation, so the iteration-start "preserve
        # uncommitted work" sweep can't grab the in-flight recon's diff.
        _submit_async_recon(lambda t=task, p=push, ts=timestamp: _run_haiku_doc_reconcile(t, p, ts))
        _log_experiment_completion(task, test_summary)
        return True

    _run_haiku_doc_reconcile(task, push, timestamp)
    _log_experiment_completion(task, test_summary)
    return True


def _log_experiment_completion(task: dict, test_summary: str) -> None:
    """Log a research step's completion, downgrading OK -> FAIL when the
    artifact is bootstrap-only (Sonnet bailed without updating it).

    Without this guard, the conductor counts a Sonnet bootstrap-and-bail run
    as OK based purely on pytest passing -- the .80 milestone wedge symptom
    observed 2026-04-29: exp1028 bootstrapped status='running', Sonnet exited
    cleanly, pytest passed, conductor logged OK. The artifact never reached
    pre_test_fixed=true and exp1030 GATE_BLOCKed forever on the false field.

    The downgrade lets MAX_FAILURES_PER_TASK kick in after 3 attempts so the
    burn loop terminates and the operator gets a visible signal in the log
    rather than a silent infinite OK cycle.
    """
    if not _artifact_is_finished(task):
        deliverable = task.get("deliverable", "<no deliverable>")
        log_step(
            task["title"],
            "FAIL",
            f"artifact_not_updated_past_bootstrap (deliverable={deliverable}); pytest: {test_summary}",
        )
        return
    log_step(task["title"], "OK", test_summary)


def _run_haiku_doc_reconcile(task: dict, push: bool, timestamp: datetime) -> None:
    """Spawn a Haiku Claude Code call to update ops/_bmad docs.

    Pulled out of research_step() so it can run either synchronously
    (default) or via the background executor for async doc-reconciliation.
    The body is the historical Haiku-reconciliation logic verbatim — see
    the long comment above the call site in research_step() for the
    honest-verdict mapping rationale (RETRO-063 fix, 2026-04-20).

    Side effects:
      - Calls run_agent (spawns Claude Code with model=haiku, max_turns=40,
        timeout=300s).
      - Reads/writes ops/changelog.md, ops/status.md, _bmad/traceability.md.
      - Issues a `git commit` and a `git push` if the agent produced a diff.

    Errors are logged via the logger; nothing is raised. This matches the
    original synchronous behaviour and keeps the background-thread variant
    crash-resistant — a failed recon must not crash the conductor.
    """
    reconcile_prompt = (
        f"You are working on the Carnot EBM framework in {PROJECT_ROOT}.\n\n"
        f"A research experiment was just completed and committed:\n"
        f"  Task: {task['title']}\n"
        f"  ID: {task['id']}\n\n"
        f"TASK: Make MINIMAL doc updates for this experiment. Be fast and HONEST.\n\n"
        f"STEP 0 (MANDATORY, BLOCKING) — Read the experiment artifact:\n"
        f"  Find the file results/experiment_<N>_*.json that was just written\n"
        f"  (the experiment ID in the task is `{task['id']}` — find the matching N).\n"
        f"  Extract the `honest_verdict` field VERBATIM.  Also note:\n"
        f"  - `status` (success/blocked/timed_out/...)\n"
        f"  - any `retro_*_closed` / `retro_*_resolved` / `retro_*_partial` flags\n"
        f"  - any AUC / TP / FP / signed_improvement / violation_rate numbers\n\n"
        f"STEP 1 — HONEST-VERDICT MAPPING (use EXACTLY this table; DO NOT improvise):\n"
        f"  honest_verdict contains 'partial' | 'inverted' | 'insufficient' |\n"
        f"    'neutral' | 'not_viable' | 'no_improvement' | 'tolerance_exceeded' |\n"
        f"    'marginal'          → status label MUST be ⚠️ Partial / Not Viable /\n"
        f"                          Research Finding.  NEVER ✅ Complete.\n"
        f"  honest_verdict contains 'blocked' | 'gpu_required' | 'required' |\n"
        f"    'synthesis_required' → status label MUST be ⚠️ Blocked.\n"
        f"  honest_verdict == 'timed_out' | 'exception' | 'failed' → status ❌ Failed.\n"
        f"  Only if honest_verdict is an unambiguous win ('complete', 'confirmed',\n"
        f"    'viable' with measured > threshold, 'closed', 'resolved', 'done')\n"
        f"    AND the artifact shows a real measured improvement matching the task\n"
        f"    goal, may you use ✅ Complete.\n"
        f"  A retro flag `retro_X_partial: true` MEANS the retro is NOT closed.\n"
        f"    Do not write 'RETRO-X closed' / 'RETRO-X resolved' / 'RETRO-X FIXED'\n"
        f"    when any `_partial: true` or `_resolved: false` or `_closed: false`\n"
        f"    flag is set in the artifact.\n\n"
        f"STEP 2 — Append to ops/changelog.md:\n"
        f"  Read the TAIL (last 20 lines), append 1 line for today "
        f"({timestamp.strftime('%Y-%m-%d')}).\n"
        f"  Include honest_verdict VERBATIM in your line.  Include the key number\n"
        f"  (AUC / TP / signed_improvement) if there is one.\n\n"
        f"STEP 3 — Append to ops/status.md ONLY if the experiment adds a new capability:\n"
        f"  Read the TAIL (last 30 lines of experiment table), append one row.\n"
        f"  The status column MUST match the mapping from STEP 1 exactly.\n\n"
        f"STEP 4 — Append to _bmad/traceability.md ONLY if new REQ-*/SCENARIO-* were added:\n"
        f"  Check if the commit diff shows new REQ/SCENARIO lines in spec.md files.\n"
        f"  If yes, append rows.  If no, skip this file entirely.\n"
        f"  CRITICAL: never mark traceability rows 'Implemented' when the source\n"
        f"  experiment's honest_verdict indicates partial / not-viable / no-improvement.\n"
        f"  Use 'Implemented-Partial' or 'Scaffolding' instead.\n\n"
        f"HARD RULES:\n"
        f"  - Do NOT remove existing content — only APPEND.\n"
        f"  - Do NOT modify scripts/research_conductor.py or research-roadmap.yaml.\n"
        f"  - Do NOT read entire files — only read the tail to find where to append.\n"
        f"  - Do NOT invent status labels.  The artifact's honest_verdict is the\n"
        f"    ONLY valid source of truth for your status claims.\n"
    )
    recon_model = "haiku" if AGENT_TYPE == "claude" else None
    try:
        recon_ok, _ = run_agent(
            reconcile_prompt,
            max_turns=40,
            timeout=300,
            model_override=recon_model,
        )
    except Exception:
        logger.exception("Haiku doc-reconciliation raised; skipping commit")
        return
    if recon_ok and git_has_changes():
        for guarded in ["scripts/research_conductor.py", "research-roadmap.yaml"]:
            _, gdiff, _ = run_cmd(["git", "diff", "--name-only", "--", guarded])
            if gdiff.strip():
                run_cmd(["git", "checkout", "--", guarded])
        git_commit_and_push(
            f"[conductor] Update docs for {task['title']}\n\n",
            push=push,
        )
        logger.info("Documentation reconciliation committed")
    else:
        logger.info("No doc updates needed (or reconciliation skipped)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Carnot Research Conductor")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument(
        "--interval", type=int, default=30, help="Minutes between steps (default: 30)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument(
        "--no-push", action="store_true", help="Don't git push (just commit locally)"
    )
    parser.add_argument(
        "--in-process-docs",
        action="store_true",
        help="Use the in-process Python doc reconciler "
        "(scripts/in_process_doc_reconcile.py) instead "
        "of the Haiku Claude Code call. Saves ~1-2 min "
        "per iteration; mechanical mapping only.",
    )
    parser.add_argument(
        "--async-doc-recon",
        action="store_true",
        help="Run the post-experiment Haiku doc reconciliation "
        "in a background thread. The conductor enters its "
        "inter-iteration sleep immediately after the "
        "experiment commit; the recon completes during "
        "sleep. Saves ~1-2 min per iteration on the Haiku "
        "path. Has no effect when --in-process-docs is set "
        "(in-process is already <100ms).",
    )
    parser.add_argument(
        "--adaptive-interval",
        action="store_true",
        help="Scale inter-iteration sleep to the iteration's "
        "actual duration. Short iterations (e.g. doomed-"
        "rerun blocks completing in <30 s) get a short "
        "sleep; mid-length iterations (CPU experiments "
        "in <5 min) get a medium sleep; long iterations "
        "(GPU experiments, planner runs) get the full "
        "--interval sleep. Saves ~30-40 min on milestones "
        "where blocks dominate (.71 retro flagged this). "
        "Off by default — fixed cadence is the safer "
        "baseline for stable autoresearch.",
    )
    args = parser.parse_args()

    os.chdir(str(PROJECT_ROOT))

    # RETRO-022 ROOT-CAUSE FIX: call env_autofix at conductor startup so that
    # CARNOT_FORCE_LIVE propagates into every subagent spawned by run_agent().
    # The env={**os.environ,...} dict built for Popen inherits from the
    # conductor process; setting it here means all children see it.
    # Previously each experiment script had to call apply_env_autofix() itself,
    # which only worked if the conductor had already given it CPU-mode hints.
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "python"))
        from carnot.pipeline.env_autofix import apply_env_autofix

        autofix = apply_env_autofix()
        if autofix.auto_fix_applied:
            logger.warning(
                "CARNOT_FORCE_LIVE auto-injected at conductor startup "
                "(gpu_detected=%s, final_env_value=%s)",
                autofix.gpu_detected,
                autofix.final_env_value,
            )
    except Exception as exc:
        logger.warning("env_autofix unavailable at startup: %s", exc)

    # Register an atexit handler so any in-flight async doc-reconciliation
    # gets a chance to finish (and push) before the conductor process exits.
    # SIGTERM-on-quota or KeyboardInterrupt will still drop in-flight recons,
    # but a normal exit path drains cleanly.
    import atexit as _atexit

    _atexit.register(_shutdown_recon_executor, wait=True, timeout=600.0)

    print("=" * 60)
    print("  Carnot Research Conductor")
    print(f"  Autonomous research via {AGENT_DISPLAY}")
    print("=" * 60)
    print(f"  Agent: {AGENT_TYPE} ({AGENT_BIN})")
    print(f"  Model: {AGENT_MODEL}")
    if AGENT_MODEL_PLANNER:
        print(f"  Model (planner override): {AGENT_MODEL_PLANNER}")
    if AGENT_MODEL_RETRO:
        print(f"  Model (retro override): {AGENT_MODEL_RETRO}")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  CARNOT_FORCE_LIVE: {os.environ.get('CARNOT_FORCE_LIVE', '<unset>')}")
    if args.loop:
        print(f"  Mode: continuous (every {args.interval} min)")
    else:
        print("  Mode: single step")

    # Load and display persistent dogfood memory
    memory = _load_dogfood_memory()
    if memory.get("experiments_checked", 0) > 0:
        print(
            f"  Dogfood memory: {memory['experiments_checked']} checks, "
            f"{memory.get('brace_fixes', 0)} brace fixes, "
            f"{memory.get('code_violations', 0)} code violations"
        )
    print()

    iteration = 0
    HEARTBEAT_FILE = PROJECT_ROOT / "ops" / "conductor-heartbeat.json"
    STATE_FILE = PROJECT_ROOT / "ops" / "conductor-state.json"

    def _write_heartbeat(phase: str, iter_n: int) -> None:
        """Write the heartbeat + state files the supervisor reads.

        Without these, conductor_supervisor.py treats the running
        conductor as an orphan and SIGTERMs it (observed 2026-04-28
        at 23:26Z, 23:41Z, 23:54Z, 23:57Z).

        Heartbeat format must use `%Y-%m-%dT%H:%M:%SZ` (no microseconds,
        Z suffix) — supervisor's strptime parser is strict.
        State file records the conductor's PID so the orphan reaper
        recognizes the legitimate conductor and doesn't SIGTERM it.
        """
        now_z = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
            HEARTBEAT_FILE.write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "iteration": iter_n,
                        "phase": phase,
                        "last_beat": now_z,
                    }
                )
            )
            STATE_FILE.write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "iteration": iter_n,
                        "phase": phase,
                        "started_at": now_z,
                    }
                )
            )
        except Exception:
            logger.exception("Heartbeat/state write failed (non-fatal)")

    while True:
        iteration += 1
        logger.info("--- Iteration %d ---", iteration)
        _write_heartbeat("iteration_start", iteration)

        iter_start = time.time()
        try:
            progress = research_step(
                push=not args.no_push,
                dry_run=args.dry_run,
                in_process_docs=args.in_process_docs,
                async_doc_recon=args.async_doc_recon,
            )
        except Exception:
            logger.exception("Unexpected error in research step")
            progress = False
        iter_duration = time.time() - iter_start

        if not args.loop:
            return 0 if progress else 1

        # Decide sleep duration. The default is `args.interval` minutes — a
        # fixed cadence. With --adaptive-interval the sleep scales to how
        # much real work the iteration did, measured by iter_duration. See
        # `compute_adaptive_sleep_min` for the tier definitions and rationale.
        if args.adaptive_interval:
            sleep_min, tier = compute_adaptive_sleep_min(iter_duration, args.interval)
            logger.info(
                "Adaptive sleep: %d min (%s) — iteration ran %.1fs",
                sleep_min,
                tier,
                iter_duration,
            )
        else:
            sleep_min = args.interval

        # Chunked sleep: survive background-harness hibernation. A single
        # time.sleep(1800) gets suspended by some schedulers and never resumes,
        # losing hours of wall clock. Chunking into 60s tick bursts lets the
        # OS scheduler re-page us promptly and lets us log progress so a stuck
        # sleep is visible from the output file.
        total_seconds = sleep_min * 60
        logger.info("Sleeping %d minutes (chunked 60s ticks)...", sleep_min)
        slept = 0
        while slept < total_seconds:
            chunk = min(60, total_seconds - slept)
            time.sleep(chunk)
            slept += chunk
            _write_heartbeat("sleeping", iteration)
            if slept % 300 == 0 and slept < total_seconds:
                logger.info("...sleeping, %d/%d min elapsed", slept // 60, sleep_min)
        logger.info("Sleep complete — resuming")

    return 0


if __name__ == "__main__":
    sys.exit(main())
