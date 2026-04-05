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

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout for live viewing
            text=True,
            cwd=str(PROJECT_ROOT),
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
    # ── Phase 1: Learned Energy in Latent Space ────────────
    {
        "id": "p1-m1.1a-ast-embedding",
        "title": "Add AST-based code embedding",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements (verbose docstrings, spec refs, 100% coverage).

CONTEXT: The current code_to_embedding() in python/carnot/verify/python_types.py
uses a simple bag-of-tokens frequency vector (256-dim). This loses all structural
information. We need richer embeddings as a stepping stone to real model embeddings.

TASK: Add ast_code_to_embedding() that uses Python's ast module to extract structural features.

CONCRETE STEPS:
1. Read python/carnot/verify/python_types.py (see code_to_embedding)
2. Add a new function ast_code_to_embedding(code: str, feature_dim: int = 64) -> jax.Array
   Features to extract:
   - Number of: function defs, function calls, loops (for/while), conditionals (if/elif),
     returns, assignments, imports, try/except blocks
   - Nesting depth: max and mean
   - Variable count: unique names
   - Line count, AST node count
   - Cyclomatic complexity approximation (branches + 1)
   Normalize to [0,1] range, pad/truncate to feature_dim.
3. Add tests in tests/python/test_verify_python_types.py:
   - Correct shape, deterministic, different code → different embedding
   - AST embedding distinguishes correct from buggy code better than bag-of-tokens
   - Handle syntax errors gracefully (return zeros)
   All tests must reference REQ-CODE-002
4. Run: .venv/bin/pytest tests/python --cov=python/carnot --cov-fail-under=100
5. Run: .venv/bin/python scripts/check_spec_coverage.py
6. Do NOT push to git.""",
    },
    {
        "id": "p1-m1.1b-local-model-embeddings",
        "title": "Add local model embeddings via transformers",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: We need real semantic embeddings for code, not bag-of-tokens.
A small local model (like microsoft/codebert-base or Salesforce/codet5-small)
can provide 768-dim embeddings that capture meaning.

TASK: Create python/carnot/embeddings/__init__.py and python/carnot/embeddings/model_embeddings.py

CONCRETE STEPS:
1. Create the python/carnot/embeddings/ package
2. In model_embeddings.py implement:
   - ModelEmbeddingConfig: model_name, device, max_length
   - extract_embedding(code: str, config: ModelEmbeddingConfig) -> jax.Array
     Uses transformers library (lazy import) to get the [CLS] or mean-pooled
     last hidden state. Returns a jax.Array.
   - If transformers not installed, return None (graceful fallback)
3. Add tests with mock (don't require transformers to be installed):
   - Test config defaults
   - Test graceful fallback when transformers missing
   - Test with a mock model that returns a known tensor
4. Run full test suite, maintain 100% coverage
5. Do NOT push to git.""",
    },
    {
        "id": "p1-m1.2-jepa-energy",
        "title": "JEPA-style context prediction energy",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: EB-JEPA predicts missing context in embedding space, scored by energy.
We need an energy function that takes (context_embedding, prediction_embedding)
and returns a scalar: low if the prediction is a coherent continuation, high otherwise.

TASK: Create python/carnot/embeddings/jepa_energy.py

CONCRETE STEPS:
1. Read python/carnot/models/gibbs.py for the GibbsModel pattern
2. Implement ContextPredictionEnergy(AutoGradMixin):
   - Takes concatenated (context_emb, prediction_emb) as input
   - Uses a Gibbs-like network to output scalar energy
   - energy(concat(ctx, pred)) → scalar
3. Implement generate_jepa_training_data():
   - Take real Python functions, split into (first_half, second_half)
   - Embed each half (using ast_code_to_embedding for now)
   - Correct pairs = real (first, second) halves
   - Noise pairs = (first_half_of_A, second_half_of_B) shuffled
4. Train with NCE: correct pairs are data, shuffled pairs are noise
5. Add tests:
   - Training reduces NCE loss
   - Correct pairs get lower energy than shuffled pairs
6. Run full test suite, 100% coverage
7. Do NOT push.""",
    },
    {
        "id": "p1-m1.3-embedding-repair",
        "title": "Repair in embedding space",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: We can now score (context, prediction) embedding pairs with energy.
The next step: given a bad prediction embedding, use gradient descent on the
energy to IMPROVE it, then find the nearest real code to the repaired embedding.

TASK: Add embedding_repair() to python/carnot/embeddings/jepa_energy.py

CONCRETE STEPS:
1. Read the existing repair() in python/carnot/verify/constraint.py
2. Add embedding_repair(ctx_emb, pred_emb, energy_model, steps, step_size):
   - Runs gradient descent on pred_emb to minimize energy(ctx, pred)
   - Returns the repaired prediction embedding
3. Add nearest_code_match(repaired_emb, codebook_embs, codebook_texts):
   - Finds the codebook entry closest to repaired_emb (cosine similarity)
   - Returns the corresponding code text
4. Add tests:
   - Repair reduces energy
   - Nearest match finds the correct code from a small codebook
5. Run full test suite, 100% coverage
6. Do NOT push.""",
    },
    # ── Phase 1.5: LLM Activation Introspection ─────────────
    {
        "id": "p1.5-activation-extraction",
        "title": "Extract per-layer activations from local model",
        "prompt": """You are working on the Carnot EBM framework in {project_root}.
Read CLAUDE.md for code style requirements.

CONTEXT: We want to monitor the LLM's internal state during generation to
detect hallucination in real-time. First step: extract activations.

TASK: Create python/carnot/embeddings/activation_extractor.py

CONCRETE STEPS:
1. Create the file with:
   - ActivationConfig: model_name (default "Qwen/Qwen3-0.6B" or similar small model), device
   - extract_layer_activations(text, config) -> dict[int, jax.Array]
     Uses transformers library (lazy import) with register_forward_hook to
     capture hidden states at each layer.
     Returns layer_num -> activation_tensor mapping.
   - compute_activation_stats(activations) -> dict with per-layer:
     norm, direction_change (cosine distance from previous layer),
     entropy of attention weights
2. Handle missing transformers gracefully (return None)
3. Add tests with mocks (don't require actual model download):
   - Test config defaults
   - Test graceful fallback
   - Test compute_activation_stats with synthetic data
   - All tests reference REQ-INFER-014
4. Update python/carnot/embeddings/__init__.py with exports
5. Run full test suite, 100% coverage
6. Do NOT push.""",
    },
    {
        "id": "p1.5-hallucination-direction",
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
6. Do NOT push.""",
    },
    {
        "id": "p1.5-layer-targeted-ebm",
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
6. Do NOT push.""",
    },
    # ── Phase 2: Energy-Based Transformer ──────────────────
    {
        "id": "p2-m2.1-minimal-ebt",
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
7. Do NOT push.""",
    },
    # ── Infrastructure improvements (fill gaps between phases) ──
    {
        "id": "infra-property-refine",
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
6. Do NOT push.""",
    },
    {
        "id": "infra-arxiv-scan",
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
Do NOT push.""",
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
