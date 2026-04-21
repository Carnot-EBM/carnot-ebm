#!/usr/bin/env python3
"""Experiment 619: DSVD-SymCode Hybrid Verifier.

**Researcher summary (RETRO-069 partial resolution):**
    DSVD (Exp 587-604) achieved offline AUC=0.976 but live AUC=0.158 — completely
    failed on real Qwen3.5-0.8B outputs.  Root cause: hidden-state probing trained
    on synthetic data cannot generalise to live model distribution.

    This experiment tests the SymCode-style replacement (arXiv 2510.25975 + arXiv
    2602.11202 Interwhen architecture): instead of probing hidden states, the LLM
    generates executable Python for each CoT step, and we run it.  Code execution
    is model-agnostic and distribution-invariant — eval('47+28') == 75 regardless
    of how the model phrased the step.

    Gate condition:
        retro_069_partial  = symcode_live_auc > 0.158  (beats DSVD live baseline)
        retro_069_resolved = symcode_live_auc >= 0.50  (at or above chance-above level)

    In CI mode (CARNOT_FORCE_LIVE != '1' or no GPU): uses regex fallback.
    In live mode (CARNOT_FORCE_LIVE=1): uses Qwen3.5-0.8B CPU inference.

Spec: REQ-VERIFY-122, REQ-VERIFY-123,
      SCENARIO-VERIFY-160, SCENARIO-VERIFY-161, SCENARIO-VERIFY-162
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# apply_env_autofix MUST be called before any other imports that might touch JAX
# or GPU state.
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.live_assertion import assert_live_or_ci_skip  # noqa: E402

assert_live_or_ci_skip()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.symcode_verifier import SymCodeVerifier  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 619
EXP_TITLE = "DSVD-SymCode Hybrid Verifier"
DELIVERABLE = "results/experiment_619_dsvd_symcode.json"

# Live baseline from Exp 604: DSVD post-finetune live AUC.
DSVD_BASELINE_LIVE_AUC = 0.158

# Number of responses to sample from each class.
N_INCORRECT_TARGET = 25
N_CORRECT_TARGET = 10


# ---------------------------------------------------------------------------
# AUC computation (no sklearn dependency in CI)
# ---------------------------------------------------------------------------


def _compute_auc(scores: list[float], labels: list[int]) -> float:
    """Compute ROC AUC via trapezoidal integration over (FPR, TPR) pairs.

    Why manual: sklearn may not be available in all CI environments.  The
    trapezoidal AUC is equivalent to sklearn's roc_auc_score for binary labels.

    Args:
        scores: Detection scores (higher = more likely violation).
        labels: Binary labels (1 = incorrect/violation, 0 = correct).

    Returns:
        AUC in [0.0, 1.0].  Returns 0.5 if all labels are the same class
        (degenerate case — cannot rank positives against negatives).
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5  # degenerate: no ranking possible

    # Sort by score descending.
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])

    tpr_vals = [0.0]
    fpr_vals = [0.0]
    tp = 0
    fp = 0
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_vals.append(tp / n_pos)
        fpr_vals.append(fp / n_neg)

    # Trapezoidal integration.
    auc = 0.0
    for i in range(1, len(fpr_vals)):
        auc += (fpr_vals[i] - fpr_vals[i - 1]) * (tpr_vals[i] + tpr_vals[i - 1]) / 2.0
    return auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 619: SymCode verifier on live_pairs_578 corpus."""
    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30):
        tmpl = ExperimentTemplate(
            EXP_ID,
            EXP_TITLE,
            DELIVERABLE,
            requires_gpu=False,
        )
        tmpl.setup()
        writer = AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE))

        # Load live corpus from Exp 578.
        corpus_path = _REPO_ROOT / "results" / "live_pairs_578.json"
        if not corpus_path.exists():
            _log.error("live_pairs_578.json not found at %s", corpus_path)
            artifact = tmpl.build_result(
                {},
                status="blocked",
                block_reason="live_pairs_578.json missing",
            )
            writer.write(artifact)
            tmpl.assert_deliverable_written()
            return

        corpus: list[dict] = json.loads(corpus_path.read_text())
        incorrect_all = [r for r in corpus if not r.get("is_correct", True)]
        correct_all = [r for r in corpus if r.get("is_correct", False)]

        # Sample up to the target counts.
        incorrect_sample = incorrect_all[:N_INCORRECT_TARGET]
        correct_sample = correct_all[:N_CORRECT_TARGET]

        n_incorrect = len(incorrect_sample)
        n_correct = len(correct_sample)
        _log.info(
            "Loaded %d incorrect + %d correct responses from live_pairs_578.json",
            n_incorrect,
            n_correct,
        )

        # Determine live vs CI mode.
        import os

        is_ci = os.environ.get("CARNOT_IS_CI", "0") == "1"
        force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1" and not is_ci
        llm_caller = None

        if force_live:
            # Attempt to load Qwen3.5-0.8B on CPU for live code generation.
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]
                import torch  # type: ignore[import]

                _log.info("Loading Qwen3.5-0.8B for live SymCode generation…")
                tok = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen3.5-0.8B", trust_remote_code=False
                )
                model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen3.5-0.8B",
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=False,
                )
                model.eval()

                def _llm_caller(prompt: str) -> str:
                    inputs = tok(prompt, return_tensors="pt")
                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=32,
                            do_sample=False,
                            pad_token_id=tok.eos_token_id,
                        )
                    new_tokens = out[0][inputs["input_ids"].shape[1] :]
                    return tok.decode(new_tokens, skip_special_tokens=True).strip()

                llm_caller = _llm_caller
                _log.info("Qwen3.5-0.8B loaded — running in live mode")
            except Exception as exc:
                _log.warning("Could not load Qwen3.5-0.8B (%s); falling back to CI regex", exc)

        verifier = SymCodeVerifier(llm_caller=llm_caller)
        mode = "live_qwen" if llm_caller is not None else "ci_regex"
        _log.info("SymCodeVerifier mode: %s", mode)

        # Score all responses.
        def score_response(r: dict) -> float:
            return verifier.detection_score(r.get("response", ""))

        incorrect_scores = [score_response(r) for r in incorrect_sample]
        correct_scores = [score_response(r) for r in correct_sample]

        symcode_tp = sum(1 for s in incorrect_scores if s > 0.0)
        symcode_fp = sum(1 for s in correct_scores if s > 0.0)

        # Compute AUC.
        all_scores = incorrect_scores + correct_scores
        all_labels = [1] * n_incorrect + [0] * n_correct
        symcode_live_auc = _compute_auc(all_scores, all_labels)

        symcode_recall = symcode_tp / n_incorrect if n_incorrect > 0 else 0.0
        symcode_fp_rate = symcode_fp / n_correct if n_correct > 0 else 0.0

        retro_069_partial = symcode_live_auc > DSVD_BASELINE_LIVE_AUC
        retro_069_resolved = symcode_live_auc >= 0.50
        honest_verdict = (
            "symcode_beats_dsvd" if retro_069_partial else "symcode_no_improvement"
        )

        _log.info(
            "symcode_tp=%d  symcode_fp=%d  recall=%.3f  fp_rate=%.3f  auc=%.3f",
            symcode_tp,
            symcode_fp,
            symcode_recall,
            symcode_fp_rate,
            symcode_live_auc,
        )
        _log.info(
            "DSVD baseline live AUC=%.3f  retro_069_partial=%s  retro_069_resolved=%s",
            DSVD_BASELINE_LIVE_AUC,
            retro_069_partial,
            retro_069_resolved,
        )

        artifact = tmpl.build_result(
            {
                "schema": "carnot.dsvd_symcode.v1",
                "n_incorrect": n_incorrect,
                "n_correct": n_correct,
                "symcode_tp": symcode_tp,
                "symcode_fp": symcode_fp,
                "symcode_recall": symcode_recall,
                "symcode_fp_rate": symcode_fp_rate,
                "symcode_live_auc": symcode_live_auc,
                "dsvd_baseline_live_auc": DSVD_BASELINE_LIVE_AUC,
                "retro_069_partial": retro_069_partial,
                "retro_069_resolved": retro_069_resolved,
                "honest_verdict": honest_verdict,
                "verifier_mode": mode,
            },
            status="success",
        )
        writer.write(artifact)
        _log.info("Artifact written: %s", DELIVERABLE)

        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
