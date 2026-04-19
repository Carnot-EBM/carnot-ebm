"""Helpers for JEPA Live Retrain v7 (Exp 535) — data loading and corpus utilities.

**Why this module exists:**
    Exp 535 trains the JEPA predictor with the LeWorldModel two-term objective on
    the freshest available real CoT pairs: Exps 527/528 (live 100q/200q benchmarks).
    Two new helpers are introduced beyond the v6 interface:

    1. ``load_v7_cot_corpus`` — preferred/fallback loading with explicit data_source
       label.  Prefers exp527/exp528 over the Exp 442 FOVER fallback over synthetic.

    2. ``summarize_corpus`` — returns a summary dict (n_pairs, n_correct, n_incorrect,
       source_breakdown) so the artifact records corpus quality at a glance.

    Re-exports v6 helpers (compute_held_out_split, violation_pairs_to_trainer_dicts)
    so callers need only one import.

**Data source priority (FR-11 honesty contract):**
    live_exp527_528 > live_fover_442 > synthetic

    The ``data_source`` label propagates into the artifact so downstream tooling
    can determine whether FR-11 was satisfied with real data.

Spec: REQ-LEARN-049, REQ-LEARN-050, SCENARIO-LEARN-078, SCENARIO-LEARN-079
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from carnot.embeddings.jepa_retrain import ViolationPair
from carnot.models.jepa_retrain_v6 import (
    _load_pairs_from_file,
    compute_held_out_split,
    violation_pairs_to_trainer_dicts,
)

__all__ = [
    "load_v7_cot_corpus",
    "summarize_corpus",
    "compute_held_out_split",
    "violation_pairs_to_trainer_dicts",
]

# ---------------------------------------------------------------------------
# load_v7_cot_corpus
# ---------------------------------------------------------------------------


def load_v7_cot_corpus(
    preferred_paths: List[str],
    fallback_paths: List[str],
) -> Tuple[List[ViolationPair], str]:
    """Load the best available CoT corpus for v7 training, with cascading fallback.

    **Priority order (FR-11 honesty contract):**
        1. Try each path in preferred_paths in order (expected: exp527, exp528).
           If any of these files yield pairs, return data_source='live_exp527_528'.
        2. If preferred_paths yielded nothing, try each fallback_path in order.
           The first fallback that yields pairs gets data_source='live_fover_442'.
        3. If all fallbacks are empty, return ([], 'synthetic') — the caller is
           responsible for generating synthetic pairs when data_source=='synthetic'.

    **Why separate preferred/fallback rather than one merged list:**
        The FR-11 verdict distinguishes between the freshest live data (Exps 527/528)
        and the older fallback corpus (Exp 442 FOVER).  Keeping them separate lets
        the caller label the data_source precisely without inspecting file paths.

    **Why return data_source as a string rather than an enum:**
        Three possible values (live_exp527_528, live_fover_442, synthetic) are
        stable and match the v6 contract.  String is simpler and JSON-serialisable.

    Args:
        preferred_paths: Paths to try first (typically exp527, exp528 results).
            Each path is tried as-is (absolute) or relative to cwd.  Missing files
            are silently skipped.
        fallback_paths: Paths to try when all preferred_paths are empty.
            Typically [fover_labeled_steps_live.json, exp514_cot_pairs.json].

    Returns:
        (pairs, data_source) where:
            - pairs is a (possibly empty) list of ViolationPair objects
            - data_source is one of 'live_exp527_528', 'live_fover_442', 'synthetic'

    Spec: REQ-LEARN-049, SCENARIO-LEARN-078, SCENARIO-LEARN-079
    """
    # --- Step 1: Try preferred paths (Exps 527/528) ---
    preferred_pairs: List[ViolationPair] = []
    for path_str in preferred_paths:
        loaded = _load_pairs_from_file(Path(path_str))
        preferred_pairs.extend(loaded)

    if preferred_pairs:
        return preferred_pairs, "live_exp527_528"

    # --- Step 2: Try fallback paths (FOVER 442, exp514) ---
    fallback_pairs: List[ViolationPair] = []
    for path_str in fallback_paths:
        loaded = _load_pairs_from_file(Path(path_str))
        fallback_pairs.extend(loaded)

    if fallback_pairs:
        return fallback_pairs, "live_fover_442"

    # --- Step 3: Nothing found — caller must generate synthetic pairs ---
    return [], "synthetic"


# ---------------------------------------------------------------------------
# summarize_corpus
# ---------------------------------------------------------------------------


def summarize_corpus(pairs: List[ViolationPair]) -> Dict[str, object]:
    """Return a summary dict describing the composition of a ViolationPair corpus.

    **Why this helper:**
        The artifact should record corpus quality so reviewers can judge whether
        the training data was balanced and sufficient.  This summary is small and
        JSON-serialisable, so it can be embedded directly in the experiment artifact.

    **source_breakdown field:**
        Counts pairs by model_id.  Most real pairs come from a single model
        (e.g., 'fover_442', 'exp527'), so this quickly shows provenance.

    Args:
        pairs: List of ViolationPair objects (may be empty).

    Returns:
        Dict with keys:
            - n_pairs (int): total number of pairs
            - n_correct (int): pairs with has_violation=False
            - n_incorrect (int): pairs with has_violation=True
            - source_breakdown (Dict[str, int]): count per model_id

    Spec: REQ-LEARN-049
    """
    n_correct = sum(1 for p in pairs if not p.has_violation)
    n_incorrect = sum(1 for p in pairs if p.has_violation)

    source_breakdown: Dict[str, int] = {}
    for pair in pairs:
        mid = pair.model_id or "unknown"
        source_breakdown[mid] = source_breakdown.get(mid, 0) + 1

    return {
        "n_pairs": len(pairs),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "source_breakdown": source_breakdown,
    }
