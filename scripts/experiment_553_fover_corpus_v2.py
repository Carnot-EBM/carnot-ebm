#!/usr/bin/env python3
"""Experiment 553: FOVER Corpus v2 — merge all real pairs and enforce diversity gate.

**Researcher summary (RETRO-056, RETRO-058):**
    Exp 543 JEPA retrain produced AUC=0.444 (below random) because the training corpus
    had only 24 pairs with 88% carry violations.  Exps 556-558 and 561 are all gated on a
    diverse corpus of >=100 real pairs with Shannon entropy >= 1.5 bits.

    This experiment:
    1. Merges ALL available real pairs (Exps 442, 538, 551, 552).
    2. Audits diversity (Shannon entropy of constraint_type distribution).
    3. Balances the corpus by downsampling overrepresented carry violations.
    4. Writes results/fover_corpus_v2.json as the canonical training corpus for .42.

**Gate chain (in order; EVERY exit path writes the deliverable):**
    0. Zombie PIDs killed immediately (subprocess.run kill -9)
    1. apply_env_autofix()                     — normalise env before any imports
    2. ExperimentTimeoutWatchdog(553, 20)      — 20-minute hard cap (CPU-only, fast)
    3. merge_fover_sources()                   — load + deduplicate all sources
    4. compute_corpus_diversity() before balance
    5. balance_corpus(target_entropy=1.5)      — downsample carry violations
    6. compute_corpus_diversity() after balance
    7. AtomicResultWriter: results/fover_corpus_v2.json
    8. Build main artifact: schema='carnot.fover_corpus.v2'
    9. tmpl.assert_deliverable_written()       — FINAL LINE

Spec: REQ-DATA-003, REQ-DATA-004,
      SCENARIO-DATA-007, SCENARIO-DATA-008, SCENARIO-DATA-009
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0: Kill zombie PIDs FIRST — before any CUDA import.
# ---------------------------------------------------------------------------
import subprocess

subprocess.run(["kill", "-9"], capture_output=True)  # no specific PIDs; harmless call

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix() — must be called before any CUDA import.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------
import json  # noqa: E402

from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.fover_corpus import (  # noqa: E402
    balance_corpus,
    compute_corpus_diversity,
    merge_fover_sources,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 553
EXP_TITLE = "FOVER Corpus v2 Diversity"
DELIVERABLE = "results/experiment_553_fover_corpus_v2.json"
CORPUS_FILE = "results/fover_corpus_v2.json"

SOURCES = [
    "results/fover_labeled_steps_live.json",  # Exp 442: 57 step-level pairs
    "results/exp538_cot_pairs.json",           # Exp 538: 25 CoT pairs (indirect)
    "results/live_pairs_551.json",             # Exp 551: 50 entry-level pairs (may not exist)
    "results/live_pairs_552.json",             # Exp 552: 100 entry-level pairs
]


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------


def run_experiment() -> None:
    """Execute the FOVER Corpus v2 diversity merge and balance pipeline."""

    # Step 2: ExperimentTimeoutWatchdog — 20-minute hard cap (CPU-only, fast).
    watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=20)
    watchdog.start()

    # Step 3: ExperimentTemplate setup (creates dirs, wires DeliverableGuard).
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=EXP_TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    try:
        # Step 4: Merge all sources — deduplicated by (question, model_id).
        entries = merge_fover_sources(SOURCES)
        n_sources_merged = sum(1 for s in SOURCES if Path(s).exists())
        n_pairs_before = len(entries)

        # Step 5: Compute diversity BEFORE balancing.
        diversity_before = compute_corpus_diversity(entries)
        entropy_before = diversity_before["constraint_type_entropy"]
        carry_pct_before = diversity_before["carry_pct"]

        # Step 6: Balance corpus to target entropy >= 1.5 bits.
        balanced = balance_corpus(entries, target_entropy=1.5)
        n_pairs_after = len(balanced)

        # Step 7: Compute diversity AFTER balancing.
        diversity_after = compute_corpus_diversity(balanced)
        entropy_after = diversity_after["constraint_type_entropy"]
        carry_pct_after = diversity_after["carry_pct"]

        # Step 8: Write corpus file atomically.
        corpus_writer = AtomicResultWriter(CORPUS_FILE)
        corpus_data = [
            {
                "question": e.question,
                "response": e.response,
                "model_id": e.model_id,
                "is_correct": e.is_correct,
                "constraint_types": e.constraint_types,
                "cot_steps": e.cot_steps,
            }
            for e in balanced
        ]
        corpus_writer.write(corpus_data)

        # Step 9: Build main artifact.
        retro_058_data_ready = n_pairs_after >= 100 and entropy_after >= 1.5
        honest_verdict = "corpus_ready" if retro_058_data_ready else "corpus_partial"

        artifact = tmpl.build_result(
            {
                "schema": "carnot.fover_corpus.v2",
                "n_sources_merged": n_sources_merged,
                "n_pairs_before_balance": n_pairs_before,
                "n_pairs_after_balance": n_pairs_after,
                "constraint_type_entropy_before": entropy_before,
                "constraint_type_entropy_after": entropy_after,
                "carry_pct_before": carry_pct_before,
                "carry_pct_after": carry_pct_after,
                "corpus_file": CORPUS_FILE,
                "retro_058_data_ready": retro_058_data_ready,
                "honest_verdict": honest_verdict,
                "diversity_before": diversity_before,
                "diversity_after": diversity_after,
            },
            status="success",
        )

    except Exception as exc:
        artifact = tmpl.build_result(
            {
                "schema": "carnot.fover_corpus.v2",
                "error": str(exc),
                "honest_verdict": "corpus_partial",
                "retro_058_data_ready": False,
            },
            status="error",
        )

    # Write main artifact atomically.
    writer = AtomicResultWriter(DELIVERABLE)
    writer.write(artifact)

    watchdog.stop()

    # FINAL LINE — raises RuntimeError if deliverable is absent.
    tmpl.assert_deliverable_written()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment()
