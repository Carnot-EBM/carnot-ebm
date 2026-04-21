#!/usr/bin/env python3
"""Experiment 635: AdapTrack Constrained Generation — in-generation backtrack repair.

**Context (arXiv 2510.17376 — AdapTrack):**
    AdapTrack backtracks during constrained decoding when the fraction of invalid
    next-token choices exceeds a threshold.  Key mathematical property: the output
    distribution is IDENTICAL to the model's own distribution under constraints —
    no output distortion.

    Applied to Carnot: when SymCodeVerifier detects an arithmetic violation at
    sentence k, backtrack to sentence k-1 and regenerate with a correction hint
    injected into the prompt.  The backtrack is proportional to the detection_score
    (high score = high confidence violation = always backtrack; low score = ambiguous
    = sometimes backtrack).

    This provides in-generation repair vs Carnot's current post-hoc VerifyRepairPipeline.

**What this experiment measures:**
    - 25 known-incorrect + 10 known-correct responses from live_pairs_578.json
    - adaptrack_tp: incorrect responses where >= 1 backtrack was triggered
    - adaptrack_fp: correct responses where >= 1 backtrack was triggered
    - adaptrack_recall  = adaptrack_tp / 25
    - adaptrack_fp_rate = adaptrack_fp / 10
    - interwhen_baseline = 0.12 (Exp 629 interwhen_recall_primary)
    - adaptrack_improves_recall = adaptrack_recall > interwhen_baseline

Spec: REQ-REPAIR-010, REQ-REPAIR-011,
      SCENARIO-REPAIR-020, SCENARIO-REPAIR-021, SCENARIO-REPAIR-022
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.adaptrack_repairer import AdapTrackRepairer
from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.interwhen_monitor import InterWhenMonitor
from carnot.pipeline.symcode_verifier import SymCodeVerifier
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus helper
# ---------------------------------------------------------------------------


def load_corpus(
    corpus_path: Path,
    n_incorrect: int = 25,
    n_correct: int = 10,
) -> tuple[list[str], list[str]]:
    """Load the first n_incorrect incorrect and n_correct correct responses.

    Args:
        corpus_path : Path to the JSON corpus file (live_pairs format).
        n_incorrect : Number of incorrect responses to collect.
        n_correct   : Number of correct responses to collect.

    Returns:
        (incorrect_responses, correct_responses) as string lists.
    """
    with corpus_path.open() as f:
        pairs = json.load(f)

    incorrect: list[str] = []
    correct: list[str] = []
    for record in pairs:
        resp = record.get("response", "")
        if not resp:
            continue
        if not record.get("is_correct", False) and len(incorrect) < n_incorrect:
            incorrect.append(resp)
        elif record.get("is_correct", False) and len(correct) < n_correct:
            correct.append(resp)
        if len(incorrect) >= n_incorrect and len(correct) >= n_correct:
            break

    return incorrect, correct


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 635: AdapTrack backtrack simulation on live GSM8K responses."""
    # MANDATORY: fix environment issues (ROCm/JAX platform conflicts, etc.)
    apply_env_autofix()

    # MANDATORY: 30-minute safety cutoff — kills process if experiment hangs
    _watchdog = ExperimentTimeoutWatchdog(635, timeout_minutes=30)

    tmpl = ExperimentTemplate(
        635,
        "AdapTrack Constrained Generation",
        "results/experiment_635_adaptrack_backtrack.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # CI stub — no LLM caller; SymCodeVerifier falls back to regex detection
    llm_caller = None
    verifier = SymCodeVerifier(llm_caller)
    monitor = InterWhenMonitor(verifier)
    repairer = AdapTrackRepairer(monitor, backtrack_threshold=0.5)

    # -----------------------------------------------------------------------
    # Load corpus
    # -----------------------------------------------------------------------
    corpus_path = _REPO_ROOT / "results" / "live_pairs_578.json"
    incorrect_responses, correct_responses = load_corpus(
        corpus_path, n_incorrect=25, n_correct=10
    )
    _log.info(
        "Loaded %d incorrect and %d correct responses",
        len(incorrect_responses),
        len(correct_responses),
    )

    # -----------------------------------------------------------------------
    # Load interwhen baseline from Exp 629
    # -----------------------------------------------------------------------
    baseline_path = _REPO_ROOT / "results" / "experiment_629_interwhen_diagnostic.json"
    interwhen_baseline = 0.12  # default if file unavailable
    if baseline_path.exists():
        with baseline_path.open() as f:
            b = json.load(f)
        interwhen_baseline = b.get("interwhen_recall_primary", 0.12)
    _log.info("interwhen_baseline = %.4f", interwhen_baseline)

    # -----------------------------------------------------------------------
    # Evaluate on incorrect responses (recall)
    # -----------------------------------------------------------------------
    adaptrack_tp_list: list[int] = []
    total_backtracks_incorrect: int = 0
    for resp in incorrect_responses:
        _, events = repairer.simulate_repair(resp)
        n_bt = sum(e.backtrack_triggered for e in events)
        total_backtracks_incorrect += n_bt
        adaptrack_tp_list.append(1 if n_bt > 0 else 0)

    adaptrack_recall = sum(adaptrack_tp_list) / len(incorrect_responses)

    # -----------------------------------------------------------------------
    # Evaluate on correct responses (false-positive rate)
    # -----------------------------------------------------------------------
    adaptrack_fp_list: list[int] = []
    total_backtracks_correct: int = 0
    for resp in correct_responses:
        _, events = repairer.simulate_repair(resp)
        n_bt = sum(e.backtrack_triggered for e in events)
        total_backtracks_correct += n_bt
        adaptrack_fp_list.append(1 if n_bt > 0 else 0)

    adaptrack_fp_rate = sum(adaptrack_fp_list) / len(correct_responses)

    # -----------------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------------
    total_responses = len(incorrect_responses) + len(correct_responses)
    total_backtracks = total_backtracks_incorrect + total_backtracks_correct
    mean_backtracks_per_response = total_backtracks / total_responses

    adaptrack_improves_recall = adaptrack_recall > interwhen_baseline
    honest_verdict = (
        "adaptrack_improves" if adaptrack_improves_recall else "adaptrack_comparable"
    )

    _log.info(
        "adaptrack_recall=%.4f  adaptrack_fp_rate=%.4f  "
        "interwhen_baseline=%.4f  improves=%s",
        adaptrack_recall,
        adaptrack_fp_rate,
        interwhen_baseline,
        adaptrack_improves_recall,
    )

    # -----------------------------------------------------------------------
    # Write deliverable
    # -----------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "n_incorrect": len(incorrect_responses),
            "n_correct": len(correct_responses),
            "adaptrack_recall": adaptrack_recall,
            "adaptrack_fp_rate": adaptrack_fp_rate,
            "interwhen_baseline": interwhen_baseline,
            "mean_backtracks_per_response": mean_backtracks_per_response,
            "adaptrack_improves_recall": adaptrack_improves_recall,
            "honest_verdict": honest_verdict,
        },
        status="success",
    )
    # Override schema field with canonical name after build_result auto-populates it.
    artifact["artifact_schema"] = "carnot.adaptrack_backtrack.v1"

    out_path = tmpl._output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        import json as _json
        _json.dump(artifact, f, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
