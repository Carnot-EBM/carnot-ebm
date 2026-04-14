"""Tier 1 self-learning trainer from live experimental traces (Exp 272).

**Researcher summary:**
    Trains the Tier 1 ConstraintTracker using ONLY real live traces from
    experiments 219, 220, and 221 — replacing the simulated error patterns
    from Exp 134. Exp 223 validated the architecture with live data (-86%
    false positives on held-out); Exp 272 formalises the training step,
    persisting the learned weights and producing a comparable results artifact.

**Detailed explanation for engineers:**
    This module wraps the existing ``build_tier1_live_retrain_payload``
    function from ``self_learning_replay`` (which was developed for Exp 224
    and contains the full, tested extraction pipeline) and adapts the output
    schema to experiment 272.

    The extraction pipeline uses ``verify_only`` runs (not ``verify_repair``)
    because that mode records the ``semantic_grounding.violations`` block with
    fine-grained taxonomy hints (e.g.
    ``question_grounding_failures:answer_target_mismatch``) that match the
    error-type strings in the held-out decision records.  Reusing the same
    extractor guarantees that trainer types align with evaluator types.

    The held-out split (final 25% of each experiment) is identical to Exp 223
    so the false-positive comparison is apples-to-apples.

Spec: REQ-LEARN-001, SCENARIO-LEARN-001, REQ-LEARN-002
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.pipeline.live_trace_memory import load_json
from carnot.pipeline.self_learning_replay import (
    HOLDOUT_FRACTION,
    TRACKER_MIN_PRECISION,
    TRACKER_MIN_SUPPORT,
    build_tier1_live_retrain_payload,
)
from carnot.pipeline.tracker import ConstraintTracker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Experiment number for this training run.
EXPERIMENT_NUMBER: int = 272

#: Run date stamp (matches the session that produced this artifact).
RUN_DATE: str = "20260413"

#: Default output path for the results JSON.
RESULT_OUTPUT: Path = Path("results/experiment_272_results.json")

#: Default output path for the saved ConstraintTracker weights.
WEIGHTS_OUTPUT: Path = Path("results/tier1_live_weights_272.json")


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_exp272_payload(
    *,
    exp219: dict[str, Any],
    exp220: dict[str, Any],
    exp221: dict[str, Any],
    exp223_reference: dict[str, Any],
    holdout_fraction: float = HOLDOUT_FRACTION,
    tracker_min_support: int = TRACKER_MIN_SUPPORT,
    tracker_min_precision: float = TRACKER_MIN_PRECISION,
) -> tuple[dict[str, Any], ConstraintTracker]:
    """Build the Exp 272 payload: live-only Tier 1 retrain with held-out evaluation.

    **Researcher summary:**
        Trains the Tier 1 ConstraintTracker on ONLY the live traces from
        Exp 219-221 (first 75% of each experiment), then evaluates on the
        held-out 25%.  Produces the results JSON and the trained weights.

    **Detailed explanation for engineers:**
        Delegates the heavy lifting to
        ``self_learning_replay.build_tier1_live_retrain_payload``, which
        contains the full extraction and evaluation pipeline.  This function
        then patches the experiment number and title fields so the result
        is recognisable as Exp 272 in the research record.

        Comparison columns in the output reference Exp 223 (the live-data
        validation experiment) and Exp 224 (the first explicit retrain).

    Spec: REQ-LEARN-001, SCENARIO-LEARN-001

    Args:
        exp219: Parsed experiment_219_results.json.
        exp220: Parsed experiment_220_results.json.
        exp221: Parsed experiment_221_results.json.
        exp223_reference: Parsed experiment_223_results.json (held-out ground truth).
        holdout_fraction: Fraction of each experiment held out (default 0.25).
        tracker_min_support: Minimum fired count before type is gated.
        tracker_min_precision: Minimum precision required to trust a type.

    Returns:
        Tuple of (results payload dict, trained ConstraintTracker).
    """
    payload, tracker = build_tier1_live_retrain_payload(
        exp219=exp219,
        exp220=exp220,
        exp221=exp221,
        exp223_reference=exp223_reference,
        holdout_fraction=holdout_fraction,
        tracker_min_support=tracker_min_support,
        tracker_min_precision=tracker_min_precision,
    )

    # Patch the experiment number and title to reflect Exp 272.
    payload["experiment"] = EXPERIMENT_NUMBER
    payload["run_date"] = RUN_DATE
    payload["title"] = (
        "Exp 272: Tier 1 self-learning retrained on live-only traces "
        "(Exp 219-221, no simulated data)"
    )

    # Update the comparison block to also reference Exp 224.
    comp = payload.get("comparison_to_exp223", {})
    comp["note"] = (
        "Exp 272 (this run) and Exp 224 both train on ONLY live traces from "
        "Exp 219-221 with an explicit separate training phase; weights are "
        "persisted.  Exp 223 tracker_only used the same live traces but in an "
        "online pass without saving weights.  Exp 272 is the conductor-assigned "
        "run number for this task."
    )
    payload["comparison_to_exp223"] = comp

    return payload, tracker


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------


def run(
    *,
    exp219_path: str | Path = "results/experiment_219_results.json",
    exp220_path: str | Path = "results/experiment_220_results.json",
    exp221_path: str | Path = "results/experiment_221_results.json",
    exp223_path: str | Path = "results/experiment_223_results.json",
    output_path: str | Path = RESULT_OUTPUT,
    weights_path: str | Path = WEIGHTS_OUTPUT,
    tracker_min_support: int = TRACKER_MIN_SUPPORT,
    tracker_min_precision: float = TRACKER_MIN_PRECISION,
) -> dict[str, Any]:
    """Load live traces, train Tier 1, evaluate, and write Exp 272 results.

    **Detailed explanation for engineers:**
        End-to-end entry point.  Resolves all paths relative to the current
        working directory, loads four JSON files, calls build_exp272_payload(),
        writes the results JSON and the trained tracker weights, then returns
        the results dict.

    Args:
        exp219_path: Path to experiment_219_results.json.
        exp220_path: Path to experiment_220_results.json.
        exp221_path: Path to experiment_221_results.json.
        exp223_path: Path to experiment_223_results.json (held-out decisions).
        output_path: Destination for experiment_272_results.json.
        weights_path: Destination for the trained tracker weights JSON.
        tracker_min_support: Minimum support before precision gating applies.
        tracker_min_precision: Precision threshold for repair suppression.

    Returns:
        The full results dict (also written to output_path).
    """
    exp219 = load_json(Path(exp219_path))
    exp220 = load_json(Path(exp220_path))
    exp221 = load_json(Path(exp221_path))
    exp223 = load_json(Path(exp223_path))

    payload, tracker = build_exp272_payload(
        exp219=exp219,
        exp220=exp220,
        exp221=exp221,
        exp223_reference=exp223,
        tracker_min_support=tracker_min_support,
        tracker_min_precision=tracker_min_precision,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    w_out = Path(weights_path)
    w_out.parent.mkdir(parents=True, exist_ok=True)
    tracker.save(str(w_out))

    return payload


__all__ = [
    "EXPERIMENT_NUMBER",
    "RESULT_OUTPUT",
    "RUN_DATE",
    "WEIGHTS_OUTPUT",
    "build_exp272_payload",
    "run",
]
