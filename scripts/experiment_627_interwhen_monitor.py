#!/usr/bin/env python3
"""Experiment 627: InterWhenMonitor Mid-Generation Violation Detection Benchmark.

**Context (RETRO-070):**
    15 consecutive Verify-Repair (VR) attempts achieved 0% improvement because
    post-hoc extraction of completed responses finds only 0–4% of violations.
    IT-tuned models bury arithmetic in natural language prose; a single pass over
    the finished response misses almost everything.

    arXiv 2602.11202 (Interwhen) showed +15 pp accuracy by checking intermediate
    solutions DURING generation.  This experiment benchmarks InterWhenMonitor —
    a sentence-boundary replay simulator that calls SymCodeVerifier at each
    boundary — on the live_pairs_578 corpus (25 incorrect + 10 correct responses).

**What this experiment measures:**
    - interwhen_recall: fraction of incorrect responses where InterWhenMonitor
      detects at least one violation (sentence-by-sentence replay).
    - interwhen_fp_rate: fraction of correct responses where InterWhenMonitor
      fires (false positives).
    - early_detection_rate: fraction of detected violations that were caught
      before the last sentence (the "early warning" benefit of mid-generation
      monitoring vs. post-hoc).
    - avg_sentences_before_detection: mean sentence index of first detected
      violation across all true-positive cases.
    - gate_open: True iff interwhen_recall >= 0.20 (minimum useful threshold
      before integration into the VR pipeline).

Spec: REQ-VERIFY-130, REQ-VERIFY-131,
      SCENARIO-VERIFY-168, SCENARIO-VERIFY-169, SCENARIO-VERIFY-170
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
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # Step 1: self-inject CARNOT_FORCE_LIVE if GPU is present but var was absent/falsy.
    apply_env_autofix()

    # Step 2: skip in CI unless explicitly unlocked (CARNOT_FORCE_LIVE=1).
    assert_live_or_ci_skip()

    # Step 3: watchdog — kill the process after 35 minutes so the conductor
    # does not hang indefinitely on a blocked LLM call.
    ExperimentTimeoutWatchdog(627, timeout_minutes=35)

    # Step 4: standard experiment scaffolding.
    tmpl = ExperimentTemplate(
        627,
        "InterWhenMonitor Mid-Generation",
        "results/experiment_627_interwhen_monitor.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # Step 5: build the LLM caller — Qwen3.5-0.8B CPU in live mode, None in CI.
    force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") not in ("0", "", "false", "False")
    llm_caller = None
    if force_live:
        _log.info("CARNOT_FORCE_LIVE=1: loading Qwen3.5-0.8B for live SymCode extraction")
        try:
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415

            _pipe = hf_pipeline(
                "text-generation",
                model="Qwen/Qwen3.5-0.8B",
                device=0,
                max_new_tokens=64,
            )

            def _llm(prompt: str) -> str:
                out = _pipe(prompt, return_full_text=False)
                return out[0]["generated_text"] if out else ""

            llm_caller = _llm
        except Exception as exc:
            _log.warning("Could not load Qwen3.5-0.8B (%s); falling back to regex mode", exc)

    # Step 6: build verifier + monitor.
    verifier = SymCodeVerifier(llm_caller=llm_caller)
    monitor = InterWhenMonitor(verifier)

    # Step 7: load corpus — 25 incorrect + 10 correct from live_pairs_578.json.
    corpus_path = _REPO_ROOT / "results" / "live_pairs_578.json"
    _log.info("Loading corpus from %s", corpus_path)
    with corpus_path.open() as f:
        all_pairs = json.load(f)

    incorrect_responses = [p["response"] for p in all_pairs if not p.get("is_correct", True)]
    correct_responses = [p["response"] for p in all_pairs if p.get("is_correct", True)]

    # Take exactly 25 incorrect and 10 correct (as specified).
    incorrect_responses = incorrect_responses[:25]
    correct_responses = correct_responses[:10]

    _log.info(
        "Corpus: %d incorrect, %d correct", len(incorrect_responses), len(correct_responses)
    )

    # Step 8: run mid-generation monitoring on all responses.
    # For each response, record whether any violation was detected and how early.
    incorrect_results: list[dict] = []
    for resp in incorrect_responses:
        violations = monitor.monitor_full_response(resp)
        n_sentences = len(monitor.split_at_boundaries(resp))
        detected = len(violations) > 0
        first_sentence_idx = violations[0].sentence_index if violations else None
        is_early = (
            first_sentence_idx is not None and first_sentence_idx < n_sentences - 1
        ) if n_sentences > 1 else False
        incorrect_results.append(
            {
                "detected": detected,
                "n_violations": len(violations),
                "first_sentence_idx": first_sentence_idx,
                "n_sentences": n_sentences,
                "is_early": is_early,
            }
        )

    correct_results: list[dict] = []
    for resp in correct_responses:
        violations = monitor.monitor_full_response(resp)
        correct_results.append({"detected": len(violations) > 0})

    # Step 9: compute metrics.
    interwhen_tp = sum(1 for r in incorrect_results if r["detected"])
    interwhen_fp = sum(1 for r in correct_results if r["detected"])
    interwhen_recall = interwhen_tp / 25
    interwhen_fp_rate = interwhen_fp / 10

    # Baseline: SymCodeVerifier post-hoc AUC from Exp 619.
    postcog_recall_baseline = 0.804

    # Early detection: among true positives, fraction where violation was
    # caught before the last sentence.
    tp_results = [r for r in incorrect_results if r["detected"]]
    early_count = sum(1 for r in tp_results if r["is_early"])
    early_detection_rate = early_count / len(tp_results) if tp_results else 0.0

    # Average sentence index of first detection across all true positives.
    detected_sentence_indices = [
        r["first_sentence_idx"] for r in tp_results if r["first_sentence_idx"] is not None
    ]
    avg_sentences_before_detection = (
        sum(detected_sentence_indices) / len(detected_sentence_indices)
        if detected_sentence_indices
        else 0.0
    )

    # Step 10: gate and verdict.
    gate_open = interwhen_recall >= 0.20
    retro_070_partial = interwhen_recall > 0.04
    if interwhen_recall >= 0.20:
        honest_verdict = "interwhen_breakthrough"
    elif interwhen_recall > 0.04:
        honest_verdict = "interwhen_improved"
    else:
        honest_verdict = "interwhen_no_improvement"

    _log.info(
        "Results: tp=%d fp=%d recall=%.3f fp_rate=%.3f early=%.3f avg_sent=%.1f verdict=%s",
        interwhen_tp,
        interwhen_fp,
        interwhen_recall,
        interwhen_fp_rate,
        early_detection_rate,
        avg_sentences_before_detection,
        honest_verdict,
    )

    # Step 11: write artifact.
    artifact = tmpl.build_result(
        {
            "schema": "carnot.interwhen_monitor.v1",
            "n_incorrect": 25,
            "n_correct": 10,
            "interwhen_tp": interwhen_tp,
            "interwhen_fp": interwhen_fp,
            "interwhen_recall": round(interwhen_recall, 4),
            "interwhen_fp_rate": round(interwhen_fp_rate, 4),
            "postcog_recall_baseline": postcog_recall_baseline,
            "early_detection_rate": round(early_detection_rate, 4),
            "avg_sentences_before_detection": round(avg_sentences_before_detection, 2),
            "gate_open": gate_open,
            "retro_070_partial": retro_070_partial,
            "honest_verdict": honest_verdict,
        },
        status="success",
    )

    out_path = _REPO_ROOT / "results" / "experiment_627_interwhen_monitor.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(artifact, f, indent=2)
    _log.info("Artifact written to %s", out_path)

    # Step 12: assert deliverable was written (final line).
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
