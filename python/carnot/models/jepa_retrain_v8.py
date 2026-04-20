"""Helpers for JEPA Live Retrain v8 (Exp 543) — expanded FOVER corpus loading.

**Why this module exists:**
    Exp 543 trains the JEPA predictor on the expanded FOVER corpus produced by Exp 542.
    The key difference from v7 is the data source priority:

    1. fover_labeled_steps_expanded.json — Exp 542 merged corpus (preferred)
    2. fover_labeled_steps_live.json     — Exp 442 baseline corpus (fallback)
    3. synthetic                         — 100 deterministic pairs (CI fallback)

    Re-exports v6/v7 helpers so callers need only one import for training data utilities.

**Why lambda_reg=0.1 (vs 0.01 in v6/v7):**
    v7 used lambda_reg=0.01 on 57 pairs. The expanded corpus may have more diverse
    embeddings, so a stronger KL regularization (0.1) prevents the predictor from
    overfitting the training set while still fitting the signal in emb[0].

Spec: REQ-LEARN-056, REQ-LEARN-057, SCENARIO-LEARN-088, SCENARIO-LEARN-089
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from carnot.embeddings.jepa_retrain import ViolationPair
from carnot.models.jepa_retrain_v6 import (
    _load_pairs_from_file,
    compute_held_out_split,
    violation_pairs_to_trainer_dicts,
)

__all__ = [
    "load_v8_cot_corpus",
    "compute_held_out_split",
    "violation_pairs_to_trainer_dicts",
]

# Filenames relative to repo results/ directory.
_EXPANDED_FILENAME = "fover_labeled_steps_expanded.json"
_LIVE_FALLBACK_FILENAME = "fover_labeled_steps_live.json"

# Lambda regularization weight for JEPA v8 — stronger than v7 (0.01) to
# prevent overfitting when training on a more diverse expanded corpus.
LAMBDA_REG_V8: float = 0.1


def load_v8_cot_corpus(
    expanded_path: str,
    live_fallback_path: str,
) -> Tuple[List[ViolationPair], str]:
    """Load the best available CoT corpus for v8 training with cascading fallback.

    **Priority order (FR-11 honesty contract):**
        1. expanded_path — Exp 542 multi-source merged corpus (preferred).
           data_source label: 'live_fover_expanded'
        2. live_fallback_path — Exp 442 baseline 57-pair corpus (fallback).
           data_source label: 'live_fover_442'
        3. If both are missing or empty — return ([], 'synthetic').
           Caller must generate synthetic pairs.

    **Why separate paths vs. a list:**
        Each source has a distinct data_source label ('live_fover_expanded' vs
        'live_fover_442') that propagates into the experiment artifact.  Keeping
        them separate makes the label assignment unambiguous.

    Args:
        expanded_path: Absolute path to the Exp 542 expanded corpus JSON.
        live_fallback_path: Absolute path to the Exp 442 FOVER baseline JSON.

    Returns:
        (pairs, data_source) where:
            - pairs is a list of ViolationPair objects (possibly empty)
            - data_source is one of 'live_fover_expanded', 'live_fover_442', 'synthetic'

    Spec: REQ-LEARN-056, SCENARIO-LEARN-089
    """
    # --- Step 1: Try expanded corpus (Exp 542 output) ---
    expanded_pairs = _load_pairs_from_file(Path(expanded_path))
    if expanded_pairs:
        return expanded_pairs, "live_fover_expanded"

    # --- Step 2: Fall back to Exp 442 baseline FOVER corpus ---
    live_pairs = _load_pairs_from_file(Path(live_fallback_path))
    if live_pairs:
        return live_pairs, "live_fover_442"

    # --- Step 3: Nothing found — caller generates synthetic pairs ---
    return [], "synthetic"
