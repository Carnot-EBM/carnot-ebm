#!/usr/bin/env python3
"""Experiment 633: HERMES Verifier Adapter — CPU prototype of step-boundary feedback loop.

**Context (arXiv 2511.18760 — HERMES):**
    HERMES achieves 67% accuracy improvement on AIME'25 using a four-module loop:
    LLM generates step → translator formalizes → prover verifies →
    feedback injected before next step.  Verification runs ASYNCHRONOUSLY at step
    boundaries, not every token.

    This experiment implements HermesVerifierAdapter as a CPU prototype:
    - Translator: LLMAsExtractorV1 (extract arithmetic claims from each step)
    - Prover:     SymCodeVerifier (executable Python; no formal logic dependency)
    - Feedback:   correction hint injected when prover says 'violated'

    Baseline: v1_baseline_recall = 0.04 (post-hoc extraction, 15 VR attempts)
    Target: hermes_recall > 0.04 (hermes_improvement = True)

**What this experiment measures:**
    - 25 known-incorrect + 10 known-correct responses from live_pairs_578.json
    - hermes_tp: incorrect responses where any step has prover_verdict='violated'
    - hermes_fp: correct responses where any step has prover_verdict='violated'
    - hermes_recall  = hermes_tp / 25
    - hermes_fp_rate = hermes_fp / 10
    - hermes_improvement = hermes_recall > 0.04 (v1 baseline)

Spec: REQ-VERIFY-136, SCENARIO-VERIFY-178, SCENARIO-VERIFY-179
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

from carnot.extraction.llm_extractor_v1 import LLMAsExtractorV1
from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.hermes_adapter import HermesVerifierAdapter
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
    """Load the first n_incorrect and n_correct responses from a live-pairs JSON file.

    Each record in the JSON list must have 'response' (str) and 'is_correct' (bool)
    fields.  Records are iterated in order; incorrect and correct are collected
    separately until the requested counts are reached.

    Args:
        corpus_path : Path to the JSON corpus file.
        n_incorrect : Number of incorrect responses to return.
        n_correct   : Number of correct responses to return.

    Returns:
        (incorrect_responses, correct_responses) as string lists.
    """
    with corpus_path.open() as f:
        pairs = json.load(f)

    incorrect: list[str] = []
    correct: list[str] = []
    for p in pairs:
        response = p.get("response", "")
        if p.get("is_correct"):
            if len(correct) < n_correct:
                correct.append(response)
        else:
            if len(incorrect) < n_incorrect:
                incorrect.append(response)
        if len(incorrect) >= n_incorrect and len(correct) >= n_correct:
            break

    return incorrect, correct


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # Step 1: self-inject CARNOT_FORCE_LIVE if GPU is present but env var absent.
    apply_env_autofix()

    # Step 2: watchdog — kill after 30 minutes to prevent conductor hangs.
    ExperimentTimeoutWatchdog(633, timeout_minutes=30)

    # Step 3: standard experiment scaffolding.
    tmpl = ExperimentTemplate(
        633,
        "HERMES Verifier Adapter",
        "results/experiment_633_hermes_adapter.json",
        requires_gpu=False,
        repo_root=_REPO_ROOT,
    )
    tmpl.setup()

    # Step 4: CPU prototype always runs in regex mode (llm_caller=None).
    # Rationale: the HERMES step-boundary feedback loop is what we're testing,
    # not the LLM extraction quality.  The Qwen3.5-0.8B linear-attention
    # architecture causes ~36 sec/call latency even on RTX 3090 CUDA (Exp 633
    # diagnostic), making 350+ calls infeasible within the 30-min watchdog window.
    # Regex mode (StepSegmentEvalChain + SymCodeVerifier regex fallback) completes
    # in seconds and is the valid CPU baseline for the step-boundary pipeline.
    # Live LLM mode is deferred to a future experiment with a batching-optimised
    # caller.  CARNOT_FORCE_LIVE is honoured by apply_env_autofix() above but the
    # extractor/verifier are explicitly set to None here for CPU prototype speed.
    _log.info("CPU prototype mode: using regex-only extraction (llm_caller=None)")
    llm_caller = None

    # Step 5: construct the HermesVerifierAdapter.
    extractor = LLMAsExtractorV1(llm_caller=llm_caller)
    verifier = SymCodeVerifier(llm_caller=llm_caller)
    adapter = HermesVerifierAdapter(extractor=extractor, verifier=verifier)

    # Step 6: load corpus — 25 incorrect + 10 correct from live_pairs_578.json.
    corpus_path = _REPO_ROOT / "results" / "live_pairs_578.json"
    incorrect_responses, correct_responses = load_corpus(corpus_path, 25, 10)

    _log.info(
        "Loaded corpus: %d incorrect, %d correct",
        len(incorrect_responses),
        len(correct_responses),
    )

    # Step 7: run HERMES adapter on all responses and compute metrics.
    # hermes_violation is True for a response iff any step has prover_verdict='violated'.
    hermes_tp = 0
    hermes_fp = 0

    for i, response in enumerate(incorrect_responses):
        steps = adapter.process_response(response)
        if any(s.prover_verdict == "violated" for s in steps):
            hermes_tp += 1
        if (i + 1) % 5 == 0:
            _log.info("Incorrect responses processed: %d/25 (tp so far: %d)", i + 1, hermes_tp)

    for i, response in enumerate(correct_responses):
        steps = adapter.process_response(response)
        if any(s.prover_verdict == "violated" for s in steps):
            hermes_fp += 1
        if (i + 1) % 5 == 0:
            _log.info("Correct responses processed: %d/10 (fp so far: %d)", i + 1, hermes_fp)

    hermes_recall = hermes_tp / 25
    hermes_fp_rate = hermes_fp / 10

    _log.info(
        "HERMES: tp=%d fp=%d recall=%.4f fp_rate=%.4f",
        hermes_tp,
        hermes_fp,
        hermes_recall,
        hermes_fp_rate,
    )

    # Step 8: compute verdict.
    v1_baseline_recall = 0.04  # post-hoc extraction best (Exp 617)
    hermes_improvement = hermes_recall > v1_baseline_recall
    honest_verdict = "hermes_improved" if hermes_improvement else "hermes_no_improvement"

    _log.info(
        "Verdict: hermes_improvement=%s honest_verdict=%s",
        hermes_improvement,
        honest_verdict,
    )

    # Step 9: write artifact.
    artifact = tmpl.build_result(
        {
            "schema": "carnot.hermes_adapter.v1",
            "n_incorrect": 25,
            "n_correct": 10,
            "hermes_tp": hermes_tp,
            "hermes_fp": hermes_fp,
            "hermes_recall": round(hermes_recall, 4),
            "hermes_fp_rate": round(hermes_fp_rate, 4),
            "v1_baseline_recall": v1_baseline_recall,
            "hermes_improvement": hermes_improvement,
            "honest_verdict": honest_verdict,
        },
        status="success",
    )

    out_path = tmpl._output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(artifact, f, indent=2)
    _log.info("Artifact written to %s", out_path)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
