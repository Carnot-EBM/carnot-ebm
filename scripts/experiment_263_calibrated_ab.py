"""Experiment 263: Calibrated A/B strategy branching.

Support module for test_experiment_263_calibration.py.
Provides corpus splitting and calibrated gate decision logic.

Spec: REQ-PRED-263-001, REQ-PRED-263-002, REQ-PRED-263-003
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


def split_corpus_by_case_id(
    corpus: list[dict[str, Any]],
    holdout_fraction: float = 0.2,
    seed: int = 263,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split corpus into train/holdout by case_id.

    Uses deterministic hash of case_id to ensure consistent splits.

    Args:
        corpus: List of corpus rows with 'case_id' field
        holdout_fraction: Target fraction for holdout set (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_corpus, holdout_corpus)
    """
    np.random.seed(seed)

    # Group rows by case_id
    case_id_to_rows = {}
    for row in corpus:
        case_id = row["case_id"]
        if case_id not in case_id_to_rows:
            case_id_to_rows[case_id] = []
        case_id_to_rows[case_id].append(row)

    # Deterministically assign each case_id to train or holdout
    case_ids = sorted(case_id_to_rows.keys())
    holdout_case_ids = set()

    for case_id in case_ids:
        # Use hash of case_id for determinism
        hash_val = int(hashlib.md5(case_id.encode()).hexdigest(), 16)
        if (hash_val % 100) < int(holdout_fraction * 100):
            holdout_case_ids.add(case_id)

    # Partition rows
    train_corpus = []
    holdout_corpus = []
    for row in corpus:
        if row["case_id"] in holdout_case_ids:
            holdout_corpus.append(row)
        else:
            train_corpus.append(row)

    return train_corpus, holdout_corpus


@dataclass(frozen=True)
class _Decision263:
    """Calibrated gate routing decision."""

    use_repair: bool
    reason: str
    fast_path_hit: bool = False


def _calibrated_gate_decision(
    case: Any,
    verifier: Any,
    calibration: Any,
    base_decision: Any,
) -> _Decision263:
    """Apply calibrated gate to decide repair routing.

    If calibration threshold is 1.0, all cases route to FAST_PATH
    (skip full verification).

    Args:
        case: ReplayCase object
        verifier: PredictiveVerifier instance
        calibration: IsotonicCalibration with threshold
        base_decision: Base decision from no-learning strategy

    Returns:
        _Decision263 with routing decision
    """
    # For threshold=1.0, everything routes FAST_PATH (skip full repair)
    if calibration.threshold >= 1.0:
        return _Decision263(
            use_repair=False,
            reason="calibrated_gate_fast_path",
            fast_path_hit=True,
        )

    # Default: follow base decision
    return _Decision263(
        use_repair=base_decision.use_repair,
        reason="calibrated_gate_neutral",
        fast_path_hit=False,
    )
