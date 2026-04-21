#!/usr/bin/env python3
"""Experiment 629: InterWhen Diagnostic Gate for VR attempt #16.

**Context (RETRO-070):**
    15 consecutive Verify-Repair (VR) attempts achieved 0% improvement because
    post-hoc extraction finds only 0-4% of arithmetic violations.  Exp 627
    benchmarked InterWhenMonitor on 25 incorrect + 10 correct responses and
    obtained recall=0.12, which is above the 0.04 baseline (retro_070_partial=True)
    but still below the 0.20 gate threshold (gate_open=False).

    This experiment runs InterWhenMonitor on an EXTENDED diagnostic set with
    higher statistical confidence: 50 incorrect + 20 correct from the combined
    live corpus (live_pairs_578.json + live_pairs_615.json).

    gate_open = interwhen_recall_primary >= 0.20 is REQUIRED before Exp 630
    (VR #16) may be scheduled.

**What this experiment measures:**
    - Primary gate set (25 incorrect + 10 correct): interwhen_tp_primary,
      interwhen_fp_primary, interwhen_recall_primary, interwhen_fp_rate_primary.
    - Extended set (50 incorrect + 20 correct, if available):
      interwhen_recall_extended — for statistical confidence only, not the gate.
    - gate_open: True iff interwhen_recall_primary >= 0.20.
    - prior_best_recall: 0.04 (post-hoc extraction baseline from Exp 617).
    - retro_070_partial: True iff interwhen_recall_primary > 0.04.
    - retro_070_resolved: True iff interwhen_recall_primary >= 0.20.

Spec: REQ-VERIFY-132, SCENARIO-VERIFY-171, SCENARIO-VERIFY-172
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repo root is on the import path when run directly as a script.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.interwhen_monitor import InterWhenMonitor
from carnot.pipeline.live_assertion import assert_live_or_ci_skip
from carnot.pipeline.symcode_verifier import SymCodeVerifier
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------


def load_corpus_pairs(
    corpus_paths: list[Path],
) -> tuple[list[str], list[str]]:
    """Load incorrect and correct responses from one or more live-pair JSON files.

    Each file must be a JSON list of dicts with 'response' and 'is_correct' fields.
    Responses from later files are appended after earlier files (no deduplication).

    Returns (incorrect_responses, correct_responses) as flat lists of strings.
    """
    incorrect: list[str] = []
    correct: list[str] = []
    for path in corpus_paths:
        if not path.exists():
            _log.warning("Corpus file not found, skipping: %s", path)
            continue
        with path.open() as f:
            pairs = json.load(f)
        for p in pairs:
            if not p.get("is_correct", True):
                incorrect.append(p["response"])
            else:
                correct.append(p["response"])
    return incorrect, correct


def run_monitor_on_set(
    monitor: InterWhenMonitor,
    incorrect_responses: list[str],
    correct_responses: list[str],
) -> tuple[int, int]:
    """Run InterWhenMonitor on a set of responses and return (tp, fp).

    tp: number of incorrect responses where any_violation() returns True.
    fp: number of correct responses where any_violation() returns True.

    This creates a FRESH monitor per call so violation state does not bleed
    between the primary and extended evaluation sets.
    """
    # Build a fresh monitor wrapping the same underlying verifier so that
    # the internal violations_detected list starts empty for each set.
    fresh = InterWhenMonitor(monitor.verifier, monitor.boundary_chars)
    tp = sum(1 for r in incorrect_responses if fresh.any_violation(r))
    fp = sum(1 for r in correct_responses if fresh.any_violation(r))
    return tp, fp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # Step 1: self-inject CARNOT_FORCE_LIVE if GPU present but var was absent.
    apply_env_autofix()

    # Step 2: skip in CI unless CARNOT_FORCE_LIVE=1 (no GPU needed but live
    # corpus is only meaningful with real LLM outputs — CI uses fixture data).
    assert_live_or_ci_skip()

    # Step 3: watchdog — kill after 40 minutes so the conductor does not hang.
    ExperimentTimeoutWatchdog(629, timeout_minutes=40)

    # Step 4: standard experiment scaffolding.
    tmpl = ExperimentTemplate(
        629,
        "InterWhen Diagnostic Gate",
        "results/experiment_629_interwhen_diagnostic.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # Step 5: build LLM caller — Qwen3.5-0.8B CPU if CARNOT_FORCE_LIVE=1,
    # else None (regex-only fallback).  The LLM provides code extraction for
    # SymCodeVerifier; regex fallback is sufficient for benchmarking purposes.
    force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
    llm_caller = None
    if force_live:
        _log.info("CARNOT_FORCE_LIVE=1: loading Qwen3.5-0.8B for live SymCode extraction")
        try:
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415

            _pipe = hf_pipeline(
                "text-generation",
                model="Qwen/Qwen3.5-0.8B",
                device="cpu",
                max_new_tokens=64,
            )

            def _llm(prompt: str) -> str:
                out = _pipe(prompt, return_full_text=False)
                return out[0]["generated_text"] if out else ""

            llm_caller = _llm
        except Exception as exc:
            _log.warning(
                "Could not load Qwen3.5-0.8B (%s); falling back to regex mode", exc
            )

    # Step 6: build verifier + monitor.
    verifier = SymCodeVerifier(llm_caller=llm_caller)
    monitor = InterWhenMonitor(verifier)

    # Step 7: load corpus.
    # Primary source: live_pairs_578.json (80 incorrect, 20 correct — most balanced).
    # Extended source: live_pairs_615.json for additional incorrect if needed.
    all_incorrect, all_correct = load_corpus_pairs(
        [
            _REPO_ROOT / "results" / "live_pairs_578.json",
            _REPO_ROOT / "results" / "live_pairs_615.json",
        ]
    )
    _log.info(
        "Combined corpus: %d incorrect, %d correct",
        len(all_incorrect),
        len(all_correct),
    )

    # Primary gate set: exactly 25 incorrect + 10 correct.
    primary_incorrect = all_incorrect[:25]
    primary_correct = all_correct[:10]

    # Extended set: 50 incorrect + 20 correct (if enough data available).
    extended_incorrect = all_incorrect[:50] if len(all_incorrect) >= 50 else []
    extended_correct = all_correct[:20] if len(all_correct) >= 20 else []

    _log.info(
        "Primary gate set: %d incorrect, %d correct",
        len(primary_incorrect),
        len(primary_correct),
    )
    _log.info(
        "Extended set: %d incorrect, %d correct",
        len(extended_incorrect),
        len(extended_correct),
    )

    # Step 8: run primary gate evaluation.
    _log.info("Running primary gate evaluation...")
    interwhen_tp_primary, interwhen_fp_primary = run_monitor_on_set(
        monitor, primary_incorrect, primary_correct
    )
    interwhen_recall_primary = interwhen_tp_primary / 25
    interwhen_fp_rate_primary = interwhen_fp_primary / 10

    _log.info(
        "Primary: tp=%d fp=%d recall=%.3f fp_rate=%.3f",
        interwhen_tp_primary,
        interwhen_fp_primary,
        interwhen_recall_primary,
        interwhen_fp_rate_primary,
    )

    # Step 9: run extended evaluation if data is available.
    interwhen_recall_extended: float | None = None
    if extended_incorrect:
        _log.info("Running extended evaluation...")
        ext_tp, _ext_fp = run_monitor_on_set(
            monitor, extended_incorrect, extended_correct
        )
        n_extended = len(extended_incorrect)
        interwhen_recall_extended = ext_tp / n_extended
        _log.info(
            "Extended: tp=%d/%d recall=%.3f",
            ext_tp,
            n_extended,
            interwhen_recall_extended,
        )

    # Step 10: gate decision and verdicts.
    gate_open = interwhen_recall_primary >= 0.20
    retro_070_partial = interwhen_recall_primary > 0.04
    retro_070_resolved = interwhen_recall_primary >= 0.20
    prior_best_recall = 0.04  # Exp 617 post-hoc extraction best.

    if gate_open:
        gate_note = "Exp 630 UNBLOCKED — schedule VR attempt #16"
        honest_verdict = "gate_open_vr_unblocked"
    else:
        gate_note = "Exp 630 GATED — do not schedule without gate_open=True"
        honest_verdict = "gate_closed_do_not_retry"

    _log.info(
        "Gate decision: gate_open=%s verdict=%s", gate_open, honest_verdict
    )

    # Step 11: write artifact.
    artifact = tmpl.build_result(
        {
            "result_schema": "carnot.interwhen_diagnostic.v1",
            "n_primary_incorrect": 25,
            "n_primary_correct": 10,
            "interwhen_tp_primary": interwhen_tp_primary,
            "interwhen_fp_primary": interwhen_fp_primary,
            "interwhen_recall_primary": round(interwhen_recall_primary, 4),
            "interwhen_fp_rate_primary": round(interwhen_fp_rate_primary, 4),
            "interwhen_recall_extended": (
                round(interwhen_recall_extended, 4)
                if interwhen_recall_extended is not None
                else None
            ),
            "prior_best_recall": prior_best_recall,
            "gate_open": gate_open,
            "gate_note": gate_note,
            "retro_070_partial": retro_070_partial,
            "retro_070_resolved": retro_070_resolved,
            "honest_verdict": honest_verdict,
        },
        status="success",
    )

    out_path = _REPO_ROOT / "results" / "experiment_629_interwhen_diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(artifact, f, indent=2)
    _log.info("Artifact written to %s", out_path)

    # Step 12: final assertion — deliverable must exist before exit.
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
