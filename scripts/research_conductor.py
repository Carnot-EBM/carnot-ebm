#!/usr/bin/env python3
"""Carnot Research Conductor — autonomous research via Claude Code.

RESEARCH_TASKS are organized by the v6 roadmap (constraint-based reasoning
via Ising/thrml). See openspec/change-proposals/research-roadmap-v6.md.

Current v6 task queue:
  Exp 48 — Code → constraint extraction
  Exp 49 — Natural language → constraint extraction
  Exp 51 — Learn constraint structure from LLM outputs
  Exp 52 — Transfer learned Ising across domains
  Exp 44 — Scheduling constraints
  Exp 42c — Fix QUBO carry chain propagation

Legacy tasks from roadmaps v2-v5 (activation-detection era) are kept at the
bottom of the list but will be skipped since their deliverables already exist.

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
# Research task definitions
# ---------------------------------------------------------------------------

# Each task is a prompt that tells Claude Code what to do.
# Claude has full file access and can edit code, run tests, etc.

# ── Research Roadmap v2: Phase 1 tasks ─────────────────────
# See openspec/change-proposals/research-roadmap-v2.md for full plan.
# These are ordered: each builds on the previous.

RESEARCH_TASKS = [
    # ══════════════════════════════════════════════════════════════════
    # Roadmap v6: Constraint-Based Reasoning via Ising/thrml
    # See: openspec/change-proposals/research-roadmap-v6.md
    #
    # Completed (not listed here):
    #   Exp 42b — Arithmetic QUBO (8/12)
    #   Exp 45  — Logical consistency (8/8)
    #   Exp 46b — Scale SAT to 5000 vars
    #   Exp 47  — LLM self-constraint extraction (10/10)
    #   Exp 50  — Learn SAT couplings via CD (89/100 perfect)
    # ══════════════════════════════════════════════════════════════════

    # ── Phase 2: LLM Constraint Extraction (continued) ────────────
    {
        "id": "v6-exp48-code-constraints",
        "deliverable": "scripts/experiment_48_code_constraints.py",
        "title": "Exp 48: Code → constraint extraction",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements (verbose docstrings, spec refs).

CONTEXT: Experiments 42b-47 validated constraint verification for arithmetic
and logic. The next step: automatically extract verifiable constraints from
LLM-generated CODE, not just natural language claims.

The idea: parse Python code into a constraint graph, then verify via Ising.
- Variable types → type constraints (x should be int, not str)
- Loop bounds → arithmetic constraints (i < len(arr))
- Function contracts → logical constraints (pre/post conditions)
- Return values → consistency constraints

EXISTING CODE TO READ FIRST:
- scripts/experiment_47_llm_self_constraints.py — the verification pipeline
- scripts/experiment_42b_arithmetic_qubo.py — QUBO arithmetic verification
- python/carnot/verify/constraint.py — ComposedEnergy constraint system
- scripts/experiment_45_logical_consistency.py — encode_claims_as_ising

TASK: Create scripts/experiment_48_code_constraints.py

CONCRETE STEPS:
1. Implement code_to_constraints(code: str) -> list[dict]:
   - Use Python's ast module to parse the code
   - Extract constraints:
     a. Type assertions: if annotation says int, constraint is "var is int"
     b. Arithmetic bounds: for i in range(n) → 0 <= i < n
     c. Return consistency: if function promises return type, verify
     d. Variable initialization: all used vars must be assigned
   - Return list of constraint dicts (same format as Exp 47)

2. Implement verify_code_constraints(code, constraints) -> dict:
   - For arithmetic constraints, verify via direct computation
   - For logical constraints, verify via Ising sampling using
     carnot.samplers.parallel_ising.ParallelIsingSampler (NOT thrml)
   - Return verdict with per-constraint details

3. Test on 8-10 scenarios:
   CORRECT code (should pass):
   - Simple function with type hints that satisfy contracts
   - Loop with correct bounds
   - Function returning correct type
   BUGGY code (should fail):
   - Off-by-one loop bound
   - Type mismatch (returns str when int expected)
   - Uninitialized variable used
   - Index out of bounds

4. Print results table with verdict (like Exp 47)
5. Use JAX_PLATFORMS=cpu compatible code (no thrml dependency)
6. Run the script and verify it works:
   JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_48_code_constraints.py
7. Do NOT push to git. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "v6-exp49-nl-constraints",
        "deliverable": "scripts/experiment_49_nl_constraints.py",
        "title": "Exp 49: Natural language → constraint extraction",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: Experiments 47-48 extract constraints from structured outputs (claims,
code). The final extraction challenge: extract verifiable claims from FREE TEXT
using natural language inference (NLI) patterns.

"The capital of France is Paris" → lookup constraint
"All mammals are warm-blooded" + "Whales are mammals" → "Whales are warm-blooded"

EXISTING CODE TO READ:
- scripts/experiment_47_llm_self_constraints.py — constraint verification pipeline
- scripts/experiment_45_logical_consistency.py — encode_claims_as_ising, count_violations
- carnot/samplers/parallel_ising.py — ParallelIsingSampler

TASK: Create scripts/experiment_49_nl_constraints.py

CONCRETE STEPS:
1. Implement claim_patterns — a set of regex/template extractors for common claim types:
   - "X is Y" → factual claim (X, is, Y)
   - "X is the Y of Z" → factual claim with relation
   - "If X then Y" → implication
   - "X and Y" → conjunction
   - "X or Y" → disjunction
   - "X but not Y" → exclusion
   - "All X are Y" + "Z is X" → entailment chain

2. Implement text_to_constraints(text: str, knowledge_base: dict) -> list[dict]:
   - Split text into sentences
   - Extract claims using patterns
   - Cross-reference factual claims against knowledge_base
   - Convert implications/conjunctions to Ising logical constraints

3. Implement verify_text_constraints(constraints) -> dict:
   - Factual claims: check against knowledge_base (True/False/Unknown)
   - Logical claims: verify consistency via Ising parallel sampler
   - Return overall verdict

4. Test on 10+ scenarios:
   CONSISTENT text (should pass):
   - "Paris is the capital of France. France is in Europe."
   - "If it rains, the ground is wet. It rained. The ground is wet."
   INCONSISTENT text (should fail):
   - "Sydney is the capital of Australia." (factually wrong)
   - "All birds fly. Penguins are birds. Penguins cannot fly."
   - "The meeting is on Monday. The meeting is on Tuesday." (contradiction)

5. Use parallel sampler (NOT thrml) for Ising verification
6. Run and verify: JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_49_nl_constraints.py
7. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    # ── Phase 3: thrml Training (continued) ──────────────────
    {
        "id": "v6-exp51-learn-from-llm",
        "deliverable": "scripts/experiment_51_learn_from_llm.py",
        "title": "Exp 51: Learn constraint structure from LLM outputs",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: Exp 50 showed we can learn Ising couplings from satisfying assignments
via Contrastive Divergence. Now: learn couplings from (correct, wrong) LLM
output pairs so the model learns WHAT MAKES AN ANSWER WRONG from data.

EXISTING CODE TO READ:
- scripts/experiment_50_learn_ising.py — CD training pipeline (train_ising_cd)
- scripts/experiment_47_llm_self_constraints.py — constraint types
- carnot/samplers/parallel_ising.py — ParallelIsingSampler
- scripts/experiment_45_logical_consistency.py — logical encoding

TASK: Create scripts/experiment_51_learn_from_llm.py

CONCRETE STEPS:
1. Create a dataset of (correct, wrong) answer pairs for arithmetic:
   - Correct: "3+4=7", "12+5=17", "100+23=123", etc.
   - Wrong: "3+4=8", "12+5=16", "100+23=124", etc. (common LLM errors)
   - Encode each as binary features: operand bits + result bits

2. Train an Ising model using Contrastive Divergence:
   - Positive phase: statistics from CORRECT answers
   - Negative phase: statistics from WRONG answers (not model samples)
   This is discriminative CD, not generative CD.
   ΔJ = -β(⟨s_i s_j⟩_correct - ⟨s_i s_j⟩_wrong)
   Δb = -β(⟨s_i⟩_correct - ⟨s_i⟩_wrong)

3. Evaluate: does the trained model assign lower energy to correct answers
   and higher energy to wrong answers? Test on HELD-OUT pairs not seen
   during training.

4. Compare to hand-coded QUBO from Exp 42b on the same test cases.

5. Test on 20+ arithmetic pairs (10 train, 10 test)
6. Run: JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_51_learn_from_llm.py
7. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "v6-exp52-transfer-ising",
        "deliverable": "scripts/experiment_52_transfer_ising.py",
        "title": "Exp 52: Transfer learned Ising across domains",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: Exp 50 learned Ising couplings for SAT. Exp 51 learns from LLM outputs.
Key question: do Ising models learned on one domain transfer to another?
Unlike activation EBMs (which showed ~50%% transfer), Ising constraints encode
STRUCTURAL rules that might generalize.

EXISTING CODE TO READ:
- scripts/experiment_50_learn_ising.py — CD training on SAT data
- scripts/experiment_51_learn_from_llm.py — discriminative CD
- scripts/experiment_39_thrml_sat.py — random_3sat, check_assignment
- carnot/samplers/parallel_ising.py — ParallelIsingSampler

TASK: Create scripts/experiment_52_transfer_ising.py

CONCRETE STEPS:
1. Train Ising models on THREE different SAT structures:
   a. Random 3-SAT at ratio 4.26 (phase transition)
   b. Random 3-SAT at ratio 3.0 (underconstrained, many solutions)
   c. Structured SAT (e.g., graph coloring encoded as SAT)
   Use n_vars=15 (small enough to find many satisfying assignments)

2. Test each trained model on ALL THREE domains:
   - Model trained on (a) tested on (a), (b), (c)
   - Model trained on (b) tested on (a), (b), (c)
   - Model trained on (c) tested on (a), (b), (c)
   For each: report mean SAT%% and best SAT%% from 100 samples

3. Compare to random baseline and to domain-specific models

4. Print 3x3 transfer matrix: rows = training domain, cols = test domain

5. Verdict: do models transfer? Is within-domain always best?
6. Run: JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_52_transfer_ising.py
7. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },

    # ── Phase 1 continued: remaining experiments ──────────────
    {
        "id": "v6-exp44-scheduling",
        "deliverable": "scripts/experiment_44_scheduling_constraints.py",
        "title": "Exp 44: Scheduling constraints",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: We've verified arithmetic (Exp 42b), logic (Exp 45), SAT (Exp 46b),
and code (Exp 48). Scheduling is a practical constraint domain where LLMs
often hallucinate: "Meeting A is at 2pm and Meeting B is at 2pm in the same
room" → conflict.

EXISTING CODE TO READ:
- scripts/experiment_45_logical_consistency.py — encode_claims_as_ising
- carnot/samplers/parallel_ising.py — ParallelIsingSampler
- scripts/experiment_39_thrml_sat.py — SAT encoding patterns

TASK: Create scripts/experiment_44_scheduling_constraints.py

CONCRETE STEPS:
1. Define scheduling constraint types:
   - Time slot exclusion: two events can't overlap in same resource
   - Ordering: event A must happen before event B
   - Duration: event takes N slots
   - Resource capacity: max K events at same time

2. Encode as Ising:
   - Variables: event_i_at_time_t (binary: is event i at time t?)
   - Exclusion: mutex coupling between event_i_at_t and event_j_at_t
   - Ordering: implication coupling
   - One-hot: each event must be at exactly one time (sum=1 constraint)

3. Test on 8+ scenarios:
   VALID schedules (should verify as consistent):
   - 3 meetings, 3 rooms, no conflicts
   - Sequential tasks with ordering constraints
   INVALID schedules (LLM-style errors):
   - Double-booked room
   - Meeting before its prerequisite
   - Same person in two places at once
   - Event scheduled outside available hours

4. Use ParallelIsingSampler for verification
5. Run: JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_44_scheduling_constraints.py
6. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "v6-exp42c-carry-chain-fix",
        "deliverable": "scripts/experiment_42c_arithmetic_carry_fix.py",
        "title": "Exp 42c: Fix QUBO carry chain propagation",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: Exp 42b achieved 8/12 correct arithmetic via QUBO encoding, but 4
cases fail (E=4.0) due to carry chain propagation. The SA solver gets stuck
in local minima where one carry bit is wrong. Cases: 12+5, 15+1, 15+9, 255+1.

EXISTING CODE TO READ (CRITICAL):
- scripts/experiment_42b_arithmetic_qubo.py — current QUBO encoding + SA solver
  Focus on addition_to_qubo() and the sa_solve() function.

TASK: Create scripts/experiment_42c_arithmetic_carry_fix.py

The approach: instead of general SA on the full QUBO, exploit the STRUCTURE
of addition. Carries propagate LEFT (from LSB to MSB). So solve bit-by-bit:

CONCRETE STEPS:
1. Implement propagation_solve(a, b, n_bits):
   - For bit 0: s[0] = a[0] XOR b[0], c[0] = a[0] AND b[0]
   - For bit i>0: s[i] = a[i] XOR b[i] XOR c[i-1], c[i] = MAJ(a[i],b[i],c[i-1])
   - This is just... computing addition. Which is the point: the Ising model's
     ground state IS the correct answer.

2. But the interesting version: verify a CLAIMED result.
   - Given a, b, claimed_r: check if claimed_r[i] matches the carry chain
   - If not: report which bit position has the first error
   - Use the QUBO energy per bit to identify exactly which constraints fail

3. Implement hybrid_verify(a, b, claimed):
   - Compute correct answer via propagation
   - If claimed != correct: use QUBO energy decomposition to show exactly
     which bit/carry is wrong (not just "wrong answer")
   - Return: dict with keys: correct, claimed, first_error_bit, per_bit_energy

4. Re-test ALL 12 cases from Exp 42b + add harder cases (up to 16-bit)
5. Should achieve 12/12 correct verification
6. Run: JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_42c_arithmetic_carry_fix.py
7. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    # ══════════════════════════════════════════════════════════════════
    # Legacy tasks below are from roadmaps v2-v5 (activation-detection era).
    # They are SUPERSEDED by the Ising/thrml constraint-based reasoning
    # direction above. Kept for reference but will be skipped by the
    # conductor since their deliverables mostly already exist.
    # ══════════════════════════════════════════════════════════════════
    {
        "id": "p1.5-hallucination-direction",
        "deliverable": "python/carnot/embeddings/hallucination_direction.py",
        "title": "Find hallucination direction in activation space",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: Given per-layer activations from correct and hallucinated outputs,
find the principal direction in activation space that distinguishes them.
This direction becomes the EBM's energy function.

TASK: Create python/carnot/embeddings/hallucination_direction.py

CONCRETE STEPS:
1. Implement find_hallucination_direction(
       correct_activations: list[jax.Array],
       hallucinated_activations: list[jax.Array],
   ) -> jax.Array:
   - Compute mean activation for correct and hallucinated sets
   - Difference vector = mean_hallucinated - mean_correct
   - Optionally: SVD to find top-k directions
   - Return the hallucination direction vector

2. Implement hallucination_energy(activation, direction) -> float:
   - Energy = dot(activation, direction) / norm(direction)
   - High projection = likely hallucination

3. Implement HallucinationDirectionConstraint(BaseConstraint):
   - Wraps hallucination_energy as a constraint for ComposedEnergy

4. Add tests with synthetic activations (no model needed):
   - Correct activations cluster away from hallucination direction
   - Energy is higher for hallucinated than correct
   - Reference REQ-INFER-014
5. Run full test suite, 100% coverage
6. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "p1.5-layer-targeted-ebm",
        "deliverable": "python/carnot/embeddings/layer_ebm.py",
        "title": "Train layer-targeted hallucination detector EBM",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: We can extract activations and find hallucination directions. Now train
a compact Gibbs model that monitors the 2-3 most informative layers (not all layers).

TASK: Create python/carnot/embeddings/layer_ebm.py

CONCRETE STEPS:
1. Implement identify_critical_layers(
       activations_correct: dict[int, list[jax.Array]],
       activations_hallucinated: dict[int, list[jax.Array]],
   ) -> list[int]:
   - For each layer, compute discrimination power (e.g., Fisher criterion)
   - Return top-3 layer indices

2. Implement train_layer_ebm(
       correct_activations: jax.Array,  # from critical layers only
       hallucinated_activations: jax.Array,
       config: LearnedVerifierConfig,
   ) -> GibbsModel:
   - Train via NCE on the concatenated critical-layer activations
   - Same training loop as train_sat_verifier

3. Implement LayerEBMVerifier combining extraction + direction + trained model

4. Add tests (synthetic data, no real model):
   - Critical layer identification works
   - Trained model discriminates correct vs hallucinated
5. Run full test suite, 100% coverage
6. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    # ── Phase 2: Energy-Based Transformer ──────────────────
    {
        "id": "p2-m2.1-minimal-ebt",
        "deliverable": "python/carnot/models/ebt.py",
        "title": "Implement minimal Energy-Based Transformer",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: The EBT paper (arxiv 2507.02092) shows transformers that output scalar
energy and do inference via gradient descent. We need a minimal JAX implementation.

TASK: Create python/carnot/models/ebt.py

CONCRETE STEPS:
1. Read python/carnot/models/gibbs.py and python/carnot/models/boltzmann.py for patterns
2. Implement EBTConfig: n_layers, d_model, n_heads, d_ff, max_seq_len
3. Implement EBTransformer(AutoGradMixin):
   - Input: concatenated (input_tokens, candidate_output_tokens) as integer sequence
   - Embedding layer: token embeddings + positional embeddings
   - N transformer layers with self-attention + FFN
   - Final: mean pool → linear → scalar energy
   - energy(x) where x is a 1-D integer array of token IDs
4. Use pure JAX (no flax/equinox) — manual attention:
   - Q, K, V projections
   - Scaled dot-product attention: softmax(QK^T / sqrt(d)) V
   - Multi-head: split dims, attend, concat, project
5. Add tests:
   - Model creation with various configs
   - energy() returns finite scalar
   - grad_energy() returns correct shape
   - Different inputs give different energies
6. Run full test suite, 100% coverage
7. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    # ── Infrastructure improvements (fill gaps between phases) ──
    {
        "id": "infra-property-refine",
        "deliverable": "tests/python/test_inference_iterative_refine_with_properties.py",
        "title": "Integrate property testing into iterative refinement",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: We have property-based testing (carnot.verify.property_test) and iterative
refinement (iterative_refine_code in carnot.inference.llm_solver), but they're
not connected.

TASK: Add iterative_refine_with_properties() to llm_solver.py that combines both:
after each LLM attempt, run specific test cases AND random property tests,
feeding ALL failures back to the LLM.

STEPS:
1. Read python/carnot/inference/llm_solver.py (iterative_refine_code)
2. Read python/carnot/verify/property_test.py (property_test, format_violations_for_llm)
3. Add the new function
4. Add tests referencing REQ-INFER-013
5. Run full test suite, 100% coverage
6. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "infra-arxiv-scan",
        "deliverable": "openspec/change-proposals/arxiv-scan-20260405.md",
        "title": "Scan arxiv for new EBM research",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.

TASK: Search the web for recent arxiv papers (2025-2026) on:
- Energy-based transformers
- JEPA and joint embedding architectures
- EBM for code generation or verification
- Self-supervised energy training
- Thermodynamic computing / analog EBM hardware

For each relevant paper found, write:
1. Title and arxiv ID
2. One-sentence summary
3. What's actionable for Carnot

Save to openspec/change-proposals/arxiv-scan-{date}.md
Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    # ── Phase 2.5: Productionize experiment findings ───────
    {
        "id": "p2.5-logprob-rejection-lib",
        "deliverable": "tests/python/test_inference_logprob_rejection.py",
        "title": "Productionize logprob rejection sampling",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: Experiment 13 proved logprob rejection sampling improves QA accuracy
by +10% (45% → 55%). This needs to be a reusable library function, not just
an experiment script.

TASK: Add logprob_rejection_sample() to python/carnot/inference/llm_solver.py

CONCRETE STEPS:
1. Read scripts/experiment_logprob_rejection.py for the working approach
2. Add to llm_solver.py:
   - logprob_rejection_sample(config, prompt, n_candidates=5, temperature=0.8)
     → (best_response, mean_logprob, all_candidates)
   - Uses the local model approach: generate N candidates with output_scores,
     compute per-token logprobs, select highest mean logprob
   - Falls back gracefully if transformers not installed
3. Add tests with mock model (don't require real model):
   - Test candidate selection picks highest logprob
   - Test with n_candidates=1 (degrades to single generation)
   - Reference REQ-INFER-008
4. Run full test suite, 100% coverage
5. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "p2.5-composite-scorer",
        "deliverable": "python/carnot/inference/composite_scorer.py",
        "title": "Productionize composite energy scorer",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: Experiment 14 proved composite scoring (logprob + structural tests)
is never worse than either alone. For code: 0% greedy → 30% with rejection.
For QA: logprobs dominate. The composite handles both.

TASK: Add CompositeEnergyScorer to python/carnot/inference/

CONCRETE STEPS:
1. Create python/carnot/inference/composite_scorer.py:
   - CompositeEnergyConfig: logprob_weight, structural_weight, test_failure_penalty
   - CompositeEnergyScorer:
     - score_candidate(code, logprob, test_cases) → float (lower is better)
     - score = -logprob_weight * mean_logprob + structural_weight * test_failure_penalty * n_failures
   - select_best(candidates: list[tuple[str, float, int]]) → best candidate
2. Wire into iterative_refine_code: use composite scoring to select among
   N candidates at each refinement iteration
3. Add tests:
   - Candidate with 0 failures + high logprob beats candidate with failures
   - Among equal-failure candidates, highest logprob wins
   - Weights are configurable
   - Reference REQ-INFER-013
4. Export from inference/__init__.py
5. Run full test suite, 100% coverage
6. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "p2.5-rejection-via-api-bridge",
        "deliverable": "tests/python/test_inference_llm_solver.py",
        "title": "Logprob rejection sampling via Claude API bridge",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: The logprob experiments used a local Qwen3 model. We need this to
work through the Claude API bridge too (for Sonnet/Haiku). The OpenAI API
supports logprobs=True parameter.

TASK: Add API-based logprob rejection sampling to llm_solver.py

CONCRETE STEPS:
1. Read the existing solve_sat_with_llm and iterative_refine_code
2. Add api_rejection_sample(config, messages, n_candidates=5):
   - Calls the LLM N times with temperature > 0
   - Requests logprobs=True in the API call
   - Extracts logprobs from the response
   - If logprobs not available (API doesn't support), fall back to random selection
   - Returns best candidate by mean logprob
3. Add tests with mocked OpenAI client:
   - Mock responses with logprobs field
   - Mock responses without logprobs (fallback)
   - Reference REQ-INFER-008
4. Run full test suite, 100% coverage
5. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "p2.5-ebt-training-real-data",
        "deliverable": "scripts/train_ebt_qa.py",
        "title": "Train EBT on real QA activations",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: We have a minimal EBT (python/carnot/models/ebt.py) and real QA
activation data from experiments. Train the EBT to score (question_embedding,
answer_embedding) pairs — low energy for correct, high for hallucinated.

TASK: Create scripts/train_ebt_qa.py

CONCRETE STEPS:
1. Read python/carnot/models/ebt.py for the EBT architecture
2. Read scripts/experiment_real_hallucination_detection.py for activation extraction
3. Write a training script that:
   a. Loads Qwen3-0.6B, generates answers for 100+ QA pairs
   b. Extracts activations, labels as correct/hallucinated
   c. Trains the EBT via optimization-through-training (P5):
      - Start from random answer embedding
      - Gradient descent on EBT energy for N steps
      - Loss = MSE between optimized embedding and real correct embedding
   d. Evaluates: does the trained EBT rank correct > hallucinated?
4. Save trained model via safetensors
5. Report accuracy on held-out test set
6. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "p2.5-experiment-dashboard",
        "deliverable": "scripts/generate_dashboard.py",
        "title": "Create experiment results dashboard",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: We have 14 experiments documented in ops/experiment-log.md but no
easy way to visualize trends or compare approaches.

TASK: Create a simple HTML dashboard for experiment results

CONCRETE STEPS:
1. Read ops/experiment-log.md for all experiment data
2. Create scripts/generate_dashboard.py that:
   - Parses the experiment log
   - Generates a static HTML file at ops/dashboard.html
   - Shows: experiment timeline, accuracy comparison bar chart,
     approach comparison table
   - Uses inline CSS/JS (no external dependencies)
   - Chart via simple SVG bars (no chart library needed)
3. Add to Makefile: make dashboard
4. Run it to generate the HTML
5. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    # ── Phase 3: In-Generation Activation Steering ─────────
    {
        "id": "p3-layer-navigator",
        "deliverable": "python/carnot/embeddings/layer_navigator.py",
        "title": "LayerNavigator: find most steerable layers",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: We have activation extraction (python/carnot/embeddings/activation_extractor.py)
and hallucination direction (hallucination_direction.py). Experiments showed that
scoring AFTER generation doesn't beat logprobs. The next step: find which layers
are most effective for STEERING during generation.

TASK: Create python/carnot/embeddings/layer_navigator.py

CONCRETE STEPS:
1. Read python/carnot/embeddings/activation_extractor.py
2. Implement LayerNavigator:
   - score_layer_steerability(model, tokenizer, qa_pairs, layer_idx) -> float
     For a given layer, measure how much adding/subtracting the hallucination
     direction at that layer changes the output. Higher change = more steerable.
     Method: run model on a question, hook into layer L, add alpha * direction
     to the hidden state, measure how much the output logits change.
   - find_best_layers(model, tokenizer, qa_pairs, n_layers=3) -> list[int]
     Score all layers, return the top-N most steerable ones.
3. Use register_forward_hook to inject activation modifications at specific layers.
   The hook: hidden_state = hidden_state + alpha * direction_vector (broadcast across sequence).
4. Add tests with a mock model (don't require real model download):
   - Test hook registration/removal
   - Test steerability scoring returns finite positive values
   - Test find_best_layers returns sorted layer indices
   - Reference REQ-INFER-015
5. Run full test suite, 100% coverage
6. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "p3-activation-steering",
        "deliverable": "python/carnot/embeddings/activation_steering.py",
        "title": "In-generation activation steering",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: LayerNavigator identifies the most steerable layers. Now use that
to STEER the model during generation: at each token step, hook into the
critical layers and subtract the hallucination direction from the hidden state.
This corrects hallucinations BEFORE the token is committed.

TASK: Create python/carnot/embeddings/activation_steering.py

CONCRETE STEPS:
1. Read layer_navigator.py for the hook-based approach
2. Implement SteeringConfig: layer_indices, direction, alpha (steering strength)
3. Implement steered_generate(model, tokenizer, prompt, config) -> str:
   - Register forward hooks on the critical layers
   - Each hook: hidden_state = hidden_state - alpha * hallucination_direction
   - This subtracts the hallucination direction at each forward pass
   - Run model.generate() with hooks active
   - Remove hooks after generation
   - Return the generated text
4. Implement calibrate_alpha(model, tokenizer, qa_pairs, layers, direction):
   - Try different alpha values (0.1, 0.5, 1.0, 2.0, 5.0)
   - For each: run steered generation on calibration QA
   - Pick the alpha that maximizes accuracy
5. Add tests with mock model:
   - Test hooks are registered and removed properly
   - Test steered output differs from unsteered output
   - Test alpha=0 gives same output as no steering
   - Reference REQ-INFER-015
6. Run full test suite, 100% coverage
7. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "p3-steering-experiment",
        "deliverable": "scripts/experiment_activation_steering.py",
        "title": "Run steering experiment on real model",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: We now have LayerNavigator + activation steering. Time to test on
the real Qwen3-0.6B model with the same QA pairs from experiments 8-14.

TASK: Create scripts/experiment_activation_steering.py

CONCRETE STEPS:
1. Read scripts/experiment_real_hallucination_detection.py for the QA pairs
2. Read python/carnot/embeddings/activation_steering.py
3. Write the experiment script:
   a. Load Qwen3-0.6B with output_hidden_states=True
   b. Run greedy baseline on 25 QA pairs
   c. Calibrate: find hallucination direction from correct vs wrong answers
   d. Find best layers via LayerNavigator (top 3)
   e. Calibrate alpha on a held-out set
   f. Run steered generation on test set
   g. Compare: greedy baseline vs steered accuracy
   h. Print results table and save to ops/experiment-log.md
4. The key metric: does steering DURING generation beat:
   - Greedy (experiment baseline)
   - Logprob rejection (experiment 13: +10%)
   - All activation-based post-hoc approaches (experiments 9-12: all negative)
5. This is THE critical experiment — if in-generation steering beats post-hoc
   scoring, it validates the entire Phase 1.5 architecture.
6. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "p3-contrastive-weight-steering",
        "deliverable": "python/carnot/embeddings/weight_steering.py",
        "title": "Contrastive Weight Steering without retraining",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: Activation steering modifies hidden states during generation.
Contrastive Weight Steering (CWS) goes further: permanently modify the
model's WEIGHTS to suppress the hallucination direction. No hooks needed
during inference — the weights themselves are corrected.

TASK: Create python/carnot/embeddings/weight_steering.py

CONCRETE STEPS:
1. Read activation_steering.py for the hook-based approach
2. Implement apply_cws(model, layer_idx, direction, alpha):
   - Modifies the layer's output projection weights in-place
   - W_new = W_old - alpha * direction @ direction.T / ||direction||^2
   - This projects out the hallucination direction from the weight matrix
   - No hooks needed during generation — weights are permanently modified
3. Implement revert_cws(model, layer_idx, original_weights):
   - Restores original weights
4. Implement steered_model(model, layers, direction, alpha) -> context manager:
   - Applies CWS on entry, reverts on exit
   - Usage: with steered_model(model, layers, dir, 1.0): model.generate(...)
5. Add tests:
   - CWS changes weights (not equal to original)
   - Revert restores original weights exactly
   - Context manager cleans up on exit
   - Reference REQ-INFER-015
6. Run full test suite, 100% coverage
7. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    # ── Phase 3.5: Concept Vectors (from Anthropic emotion research) ──
    {
        "id": "p3.5-concept-vectors",
        "deliverable": "python/carnot/embeddings/concept_vectors.py",
        "title": "Find hallucination concept vectors (multi-vector)",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: Anthropic's emotion vector research (anthropic.com/research/emotion-concepts-function)
found that LLMs have distinct activation vectors for different emotions, and steering with
the RIGHT specific vector (e.g., "desperate") is more effective than a generic direction.
Our single hallucination direction conflates uncertainty, confabulation, pattern-matching,
and memorization — which is why experiments 9-12 showed mixed results.

TASK: Create python/carnot/embeddings/concept_vectors.py

CONCRETE STEPS:
1. Read python/carnot/embeddings/hallucination_direction.py for the single-direction approach
2. Implement find_concept_vectors() that discovers MULTIPLE hallucination-related vectors:
   - Define concept prompts: generate text where the model is "certain about a fact",
     "uncertain and guessing", "confabulating/making things up", "reasoning step by step",
     "reciting memorized content"
   - For each concept: generate characteristic activations via model forward pass
   - Compute contrastive vectors between concept pairs
   - Return a dict of named concept vectors
3. Implement concept_energy(activation, concept_vectors) -> dict[str, float]:
   - Returns per-concept energy scores
   - "confabulation_energy", "uncertainty_energy", etc.
4. Implement best_concept_for_detection(concept_vectors, correct_acts, hallucinated_acts):
   - Test each concept vector's discriminative power
   - Return the concept that best separates correct from hallucinated
5. Add tests with synthetic data (no real model needed):
   - Multiple concept vectors are orthogonal-ish
   - Per-concept energy returns finite values
   - Reference REQ-INFER-016
6. Run full test suite, 100% coverage
7. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "p3.5-concept-steering-experiment",
        "deliverable": "scripts/experiment_concept_steering.py",
        "title": "Steer with concept-specific vectors on real model",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: We now have concept-specific vectors (confabulation, uncertainty, etc.)
instead of a single hallucination direction. The Anthropic research showed that
concept-specific steering beats generic steering.

TASK: Create scripts/experiment_concept_steering.py

CONCRETE STEPS:
1. Load Qwen3-0.6B (already installed: transformers + torch)
2. Generate concept vectors using find_concept_vectors() on the model
3. Run baseline QA (same 25 questions from experiment 8)
4. For each concept vector, try steering:
   a. Suppress confabulation vector during generation
   b. Amplify uncertainty vector (model should say "I don't know" more)
   c. Measure accuracy change vs baseline
5. Compare:
   - Baseline greedy
   - Generic hallucination direction steering
   - Confabulation-specific steering
   - Uncertainty-amplification steering
6. Report which concept vector gives the best improvement
7. Save results to ops/experiment-log.md
8. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    # ── Track A: Concept-Specific Vectors (v3 roadmap) ─────
    {
        "id": "a1-concept-prompting",
        "deliverable": "scripts/experiment_concept_specific_vectors.py",
        "title": "Generate concept-specific vectors via targeted prompting",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: Generic mean-difference hallucination direction (experiments 9-16) failed for
steering and rejection sampling. Anthropic's emotion research showed concept-specific
vectors via targeted prompting ARE causally effective. We need to prompt the model to
generate text in specific cognitive modes, not just compare correct vs wrong answers.

TASK: Create scripts/experiment_concept_specific_vectors.py

STEPS:
1. Load Qwen3-0.6B (already installed)
2. For each concept, generate 20 responses using targeted prompts:
   - "certain": "I am absolutely certain that [topic] is [fact]. This is well-established."
   - "uncertain": "I'm not sure about this, but I think [topic] might be [guess]..."
   - "confabulating": "Let me just make up a plausible-sounding answer about [topic]..."
   - "reasoning": "Let me think step by step about [topic]. First..."
   - "memorized": "As is commonly known, [topic] is [well-known fact]."
   Use 20 different topics (countries, animals, science facts).
3. Extract per-token activations from the LAST layer for each response
4. Compute contrastive vectors between concept pairs:
   - confabulation_dir = mean(confabulating) - mean(certain)
   - uncertainty_dir = mean(uncertain) - mean(certain)
5. Test each concept vector's ability to separate the 25 QA pairs from experiment 8
   (correct vs hallucinated answers)
6. Report: which concept direction best separates correct from hallucinated?
7. Compare against the generic mean-difference direction (experiment 8: 64%)
8. Handle bfloat16: cast to float32 for numpy/jax, cast back for model ops
9. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "a2-concept-steering",
        "deliverable": "scripts/experiment_concept_specific_steering.py",
        "title": "Steer with confabulation-specific vector",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: experiment_concept_specific_vectors.py identified which concept direction
best separates correct from hallucinated. Now test steering with THAT specific
direction (not the generic mean-difference that failed in experiments 15-16).

TASK: Create scripts/experiment_concept_specific_steering.py

STEPS:
1. Load the concept vectors from the previous experiment (or recompute)
2. Use the BEST concept direction (likely confabulation)
3. Hook into model layers during generation, subtract alpha * confabulation_direction
4. IMPORTANT: cast direction to float32 for arithmetic, then back to bfloat16:
   modified = (hidden.float() + alpha * direction.float()).to(hidden.dtype)
5. Try alphas: [0.1, 0.5, 1.0, 2.0, 5.0] on the 25 QA pairs
6. Compare against greedy baseline AND logprob rejection (+10%)
7. Report results and save to ops/experiment-log.md
8. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    # ── Track B: Per-Token Activation EBM ──────────────────
    {
        "id": "b1-token-activations",
        "deliverable": "scripts/collect_token_activations.py",
        "title": "Collect per-token activation dataset",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: Mean-pooled activations destroyed the token-level signal (experiments 9-12).
Logprobs work because they ARE per-token. We need per-token activation data to train
an EBM that operates at the token level.

TASK: Create scripts/collect_token_activations.py

STEPS:
1. Load Qwen3-0.6B
2. Use the 93 QA pairs from experiment_scaled_rejection_sampling.py
3. For each question: generate answer, get output_hidden_states=True
4. For each GENERATED token: save (token_id, last_layer_activation, is_correct)
   - A token is "correct" if the full answer is verified as correct
   - Save as a dict with jax arrays
5. Save to data/token_activations.npz (or .safetensors)
6. Report: total tokens collected, dimension, correct/wrong ratio
7. Target: 5000+ per-token examples
8. Handle bfloat16 → float32 conversion for numpy
9. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "b2-token-ebm",
        "deliverable": "scripts/experiment_per_token_ebm.py",
        "title": "Train Gibbs EBM on per-token activations",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: We have per-token activations from collect_token_activations.py.
Train a Gibbs model on individual token activations (5000+ examples) instead
of mean-pooled sequence activations (42 examples that overfit).

TASK: Create scripts/experiment_per_token_ebm.py

STEPS:
1. Load token activations from data/token_activations.npz
2. Split into train (80%) and test (20%)
3. Train Gibbs model [1024→128→32→1] via NCE on per-token activations
   (Use the training pattern from carnot.inference.learned_verifier)
4. Evaluate: does the trained EBM separate correct from wrong tokens?
5. Use the trained EBM for rejection sampling:
   - For each candidate answer, compute mean per-token EBM energy
   - Select candidate with lowest mean energy
6. Compare against logprob rejection (+10% from experiment 13)
7. Report: calibration accuracy, test accuracy, rejection sampling accuracy
8. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    # ── Track C: Scale Up ──────────────────────────────────
    {
        "id": "c1-large-dataset",
        "deliverable": "scripts/generate_qa_dataset.py",
        "title": "Generate 1000+ QA pairs programmatically",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: All activation experiments used 25-93 examples. At 1000+, overfitting
may not be a problem. We need a large QA dataset with known correct answers.

TASK: Create scripts/generate_qa_dataset.py

STEPS:
1. Generate 1000+ factual QA pairs from these categories:
   - Math: "What is A * B?" for random A, B (easy to verify)
   - Geography: capitals, populations (from a hardcoded list of 200+ countries)
   - Science: atomic numbers, chemical symbols (from periodic table)
   - Dates: historical events (from a hardcoded list of 100+ events)
   - General: "How many X does Y have?" (hardcoded facts)
2. For each pair, store: (question, expected_answer_substring)
3. Save to data/qa_dataset_1000.json
4. Verify: load and check a few examples
5. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    # ── Track D: Ship What Works ───────────────────────────
    {
        "id": "d1-mcp-server",
        "deliverable": "tools/verify-mcp/server.py",
        "title": "MCP server for code verification",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: The composite scorer (logprob + structural tests) works. Iterative
refinement with property testing works. Package these as an MCP server that
Claude Code can call automatically during code generation.

TASK: Create tools/verify-mcp/server.py

STEPS:
1. Create an MCP server using the MCP Python SDK (pip install mcp)
2. Expose two tools:
   - verify_code(code: str, func_name: str, test_cases: list) -> dict
     Runs structural tests, returns energy score and pass/fail details
   - verify_with_properties(code: str, func_name: str, properties: list) -> dict
     Runs property-based tests, returns violations
3. The server should be startable via: python tools/verify-mcp/server.py
4. Add an MCP config entry for .mcp.json
5. Test manually: start server, call verify_code with a simple function
6. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
    {
        "id": "d2-cli-tool",
        "deliverable": "scripts/carnot_cli.py",
        "title": "CLI tool for code verification",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: Package the verification pipeline as a simple CLI.

TASK: Create scripts/carnot_cli.py

STEPS:
1. CLI: carnot verify <file.py> --func <name> --test "input:expected"
2. Runs: structural tests + property-based tests
3. Reports: energy score, test results, violations
4. Exit code 0 = verified, 1 = violations found
5. Use argparse, no external CLI library needed
6. Do NOT push. Do NOT modify scripts/research_conductor.py.""",
    },
]


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
