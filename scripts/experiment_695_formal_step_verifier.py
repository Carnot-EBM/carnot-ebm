#!/usr/bin/env python3
"""Experiment 695 — FormalStepVerifier vs EidokuCSP: Tier 2.8 Candidate Benchmark.

**Researcher summary:**
    SymCodeVerifier (Tier 2) checks arithmetic within each step independently but
    cannot detect contradictions that span multiple steps — e.g., a step that
    re-assigns a variable to a different value.  This experiment fills the Tier 2.8
    gap by comparing two multi-step formal verification approaches on the FoVer corpus:

    FormalStepVerifier (arXiv 2603.29500): step entailment via Z3 — each consecutive
    step pair is checked for Z3 consistency.  A chain is correct iff all transitions
    are consistent.

    EidokuCSP (arXiv 2512.20664): CSP variable-domain consistency — each step's
    variable assignments are extracted and global contradictions are detected.

    The benchmark computes AUC for each approach on fover_labeled_formal_v1.json
    and recommends the winner as the Tier 2.8 candidate.

**Gate chain (every exit path writes the deliverable):**
    1. ExperimentTimeoutWatchdog(695, timeout_minutes=30) — hard cap.
    2. Load fover_labeled_formal_v1.json — extract (steps, correct_label) pairs.
    3. SymCodeVerifier baseline: compute symcode_auc.
    4. FormalStepVerifier: compute formal_step_auc.
    5. EidokuCSP: compute eidoku_auc.
    6. Speed benchmark: time all three on 20 chains.
    7. Recommend tier_28_winner and honest_verdict.
    8. Write results/experiment_695_formal_step_verifier.json.
    9. tmpl.assert_deliverable_written() — FINAL LINE.

Spec: REQ-VERIFY-165, REQ-VERIFY-166,
      SCENARIO-VERIFY-217, SCENARIO-VERIFY-218, SCENARIO-VERIFY-219
"""

from __future__ import annotations

import json
import os
import sys
import time

# Ensure scripts/ and repo root are on the path for imports.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.experiment_template import ExperimentTemplate
from carnot.pipeline.atomic_writer import AtomicResultWriter
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.symcode_verifier import SymCodeVerifier
from carnot.pipeline.formal_step_verifier import FormalStepVerifier, EidokuCSP

_DELIVERABLE = "results/experiment_695_formal_step_verifier.json"
_CORPUS_PATH = "results/fover_labeled_formal_v1.json"
_SPEED_BENCHMARK_N = 20


def _compute_auc(scores: list[float], labels: list[bool]) -> float:
    """Compute ROC-AUC, returning 0.5 when only one class is present.

    ROC-AUC requires both positive and negative examples.  If the FoVer corpus
    has only one class (e.g. all step_correct=True), we cannot compute a meaningful
    AUC.  In that case we return 0.5 (random classifier baseline) and let the
    honest_verdict logic handle the degenerate case.

    Args:
        scores: List of float scores (higher = more likely incorrect).
        labels: List of bool ground-truth labels (True = correct chain).

    Returns:
        Float AUC in [0.0, 1.0], or 0.5 if only one class is present.
    """
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore[import]
        # sklearn expects: label=1 means "positive class" = incorrect.
        # Our convention: score is "probability of violation" (higher = more likely wrong).
        # labels are True=correct, False=incorrect, so we invert for sklearn.
        int_labels = [0 if lab else 1 for lab in labels]
        if len(set(int_labels)) < 2:
            # Degenerate: only one class present — AUC undefined, return 0.5.
            return 0.5
        return float(roc_auc_score(int_labels, scores))
    except ImportError:
        # sklearn not available — manual trapezoidal AUC.
        return _manual_auc(scores, labels)


def _manual_auc(scores: list[float], labels: list[bool]) -> float:
    """Compute ROC-AUC manually via trapezoidal rule when sklearn is absent.

    Sorts by score descending and sweeps the ROC curve.  Returns 0.5 when
    only one class is present (degenerate case).

    Args:
        scores: Float scores (higher = predicts incorrect).
        labels: Bool ground-truth (True = correct chain, False = incorrect).

    Returns:
        Float AUC or 0.5 for degenerate label sets.
    """
    # Invert labels: 1 = incorrect (positive class), 0 = correct.
    y = [0 if lab else 1 for lab in labels]
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5  # degenerate

    # Sort by descending score.
    paired = sorted(zip(scores, y), key=lambda x: -x[0])
    tp = fp = 0
    prev_tp = prev_fp = 0
    auc = 0.0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
        # Add trapezoid slice.
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
        prev_tp, prev_fp = tp, fp
    return auc / (n_pos * n_neg)


def _segment_steps(text: str) -> list[str]:
    """Split a step_text string into individual steps for multi-step analysis.

    Uses SymCodeVerifier.segment_steps to be consistent with the Tier 2 baseline.

    Args:
        text: Raw step_text from the FoVer corpus.

    Returns:
        List of non-empty step strings.
    """
    verifier = SymCodeVerifier(llm_caller=None)
    steps = verifier.segment_steps(text)
    # If segmentation produces nothing (empty text), return the original text as one step.
    return steps if steps else [text]


def main() -> None:
    """Run Exp 695: FormalStepVerifier vs EidokuCSP benchmark."""
    tmpl = ExperimentTemplate(
        exp_id=695,
        title="FormalStepVerifier vs EidokuCSP: Tier 2.8 Candidate Benchmark",
        deliverable=_DELIVERABLE,
    )
    tmpl.setup()

    result_path = os.path.join(_REPO_ROOT, _DELIVERABLE)

    writer = AtomicResultWriter(result_path)

    with ExperimentTimeoutWatchdog(695, timeout_minutes=30, result_path=result_path):
        # ------------------------------------------------------------------
        # Step 1: Load corpus
        # ------------------------------------------------------------------
        corpus_path = os.path.join(_REPO_ROOT, _CORPUS_PATH)
        with open(corpus_path) as f:
            corpus = json.load(f)

        pairs = corpus["pairs"]
        n_pairs = len(pairs)
        print(f"Loaded {n_pairs} FoVer pairs from {_CORPUS_PATH}")

        # Extract (steps_list, correct_label) from each pair.
        chains: list[list[str]] = []
        labels: list[bool] = []
        for pair in pairs:
            step_text = pair["step_text"]
            correct = bool(pair["step_correct"])
            steps = _segment_steps(step_text)
            chains.append(steps)
            labels.append(correct)

        n_unique_labels = len(set(labels))
        print(f"Label distribution: {sum(labels)} correct, {sum(1 for l in labels if not l)} incorrect")
        corpus_degenerate = n_unique_labels < 2

        # ------------------------------------------------------------------
        # Step 2: SymCodeVerifier baseline
        # ------------------------------------------------------------------
        symcode_verifier = SymCodeVerifier(llm_caller=None)
        symcode_scores: list[float] = []
        for chain in chains:
            response_text = " ".join(chain)
            score = symcode_verifier.detection_score(response_text)
            symcode_scores.append(score)

        symcode_auc = _compute_auc(symcode_scores, labels)
        print(f"SymCodeVerifier AUC: {symcode_auc:.4f}")

        # ------------------------------------------------------------------
        # Step 3: FormalStepVerifier
        # ------------------------------------------------------------------
        formal_verifier = FormalStepVerifier()
        formal_scores: list[float] = []
        for chain in chains:
            correct = formal_verifier.chain_correct(chain)
            # Score is "probability of violation": 0.0 = correct, 1.0 = incorrect.
            formal_scores.append(0.0 if correct else 1.0)

        formal_step_auc = _compute_auc(formal_scores, labels)
        print(f"FormalStepVerifier AUC: {formal_step_auc:.4f}")

        # ------------------------------------------------------------------
        # Step 4: EidokuCSP
        # ------------------------------------------------------------------
        eidoku = EidokuCSP()
        eidoku_scores: list[float] = []
        for chain in chains:
            consistent = eidoku.check_global_consistency(chain)
            # Score is "probability of violation": 0.0 = consistent, 1.0 = inconsistent.
            eidoku_scores.append(0.0 if consistent else 1.0)

        eidoku_auc = _compute_auc(eidoku_scores, labels)
        print(f"EidokuCSP AUC: {eidoku_auc:.4f}")

        # ------------------------------------------------------------------
        # Step 5: Speed benchmark (20 chains)
        # ------------------------------------------------------------------
        benchmark_chains = chains[:_SPEED_BENCHMARK_N]

        # SymCodeVerifier speed
        t0 = time.perf_counter()
        for chain in benchmark_chains:
            symcode_verifier.detection_score(" ".join(chain))
        symcode_ms = (time.perf_counter() - t0) * 1000 / len(benchmark_chains)

        # FormalStepVerifier speed
        t0 = time.perf_counter()
        for chain in benchmark_chains:
            formal_verifier.chain_correct(chain)
        formal_ms = (time.perf_counter() - t0) * 1000 / len(benchmark_chains)

        # EidokuCSP speed
        t0 = time.perf_counter()
        for chain in benchmark_chains:
            eidoku.check_global_consistency(chain)
        eidoku_ms = (time.perf_counter() - t0) * 1000 / len(benchmark_chains)

        ms_per_chain = {
            "symcode_verifier": round(symcode_ms, 3),
            "formal_step_verifier": round(formal_ms, 3),
            "eidoku_csp": round(eidoku_ms, 3),
        }
        print(f"Speed (ms/chain): {ms_per_chain}")

        # ------------------------------------------------------------------
        # Step 6: Recommend Tier 2.8 winner
        # ------------------------------------------------------------------
        if corpus_degenerate:
            # Cannot compute meaningful AUC when only one class is present.
            # Both approaches score 0.5 by convention; neither qualifies.
            tier_28_winner = "neither_below_threshold"
            honest_verdict = "tier_28_no_candidate"
            notes = (
                "FoVer corpus v1 contains only step_correct=True labels (all Z3 verdicts "
                "are 'unparseable' — single-step 'The answer is 42.' format with no "
                "arithmetic). AUC is undefined; reported as 0.5 (random baseline). "
                "A corpus with genuine violation examples is required for meaningful "
                "Tier 2.8 benchmarking."
            )
        elif formal_step_auc >= eidoku_auc and formal_step_auc >= 0.70:
            tier_28_winner = "formal_step_verifier"
            honest_verdict = "tier_28_candidate_identified"
            notes = ""
        elif eidoku_auc > formal_step_auc and eidoku_auc >= 0.70:
            tier_28_winner = "eidoku_csp"
            honest_verdict = "tier_28_candidate_identified"
            notes = ""
        else:
            tier_28_winner = "neither_below_threshold"
            honest_verdict = "tier_28_no_candidate"
            notes = (
                f"Neither approach reached AUC >= 0.70 threshold. "
                f"FormalStepVerifier: {formal_step_auc:.4f}, EidokuCSP: {eidoku_auc:.4f}."
            )

        print(f"Tier 2.8 winner: {tier_28_winner}")
        print(f"Honest verdict: {honest_verdict}")

        # ------------------------------------------------------------------
        # Step 7: Build and write artifact
        # ------------------------------------------------------------------
        artifact_data: dict = {
            "n_pairs": n_pairs,
            "corpus_path": _CORPUS_PATH,
            "corpus_degenerate": corpus_degenerate,
            "symcode_auc": round(symcode_auc, 6),
            "formal_step_auc": round(formal_step_auc, 6),
            "eidoku_auc": round(eidoku_auc, 6),
            "ms_per_chain": ms_per_chain,
            "tier_28_winner": tier_28_winner,
            "honest_verdict": honest_verdict,
        }
        if notes:
            artifact_data["notes"] = notes

        artifact = tmpl.build_result(artifact_data, status="success")
        writer.write(artifact)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
