#!/usr/bin/env python3
"""Experiment 760 — Gemma4-E4B-it VR Threshold Grid Search.

**Researcher summary:**
    Exp 708 showed VR at adaptive gate suppression produced signed_improvement=0.0
    for Gemma4-E4B-it (no harm, but no benefit either).  arXiv 2601.01490 predicts
    that stronger models need a higher abstention threshold to avoid
    constraint-induced distortion: only repair violations when the verifier is
    highly confident.

    This experiment grid-searches 5 thresholds [0.10, 0.20, 0.30, 0.40, 0.50]
    on 50 GSM8K questions (seed=0) to find the threshold where VR first produces
    signed_improvement > 0 for Gemma4.

    Each threshold setting abstains from repair when symcode_confidence is below
    the threshold.  We record baseline_accuracy, vr_accuracy, signed_improvement,
    n_abstained (repair skipped), and n_repaired (repair attempted).

**Steps:**
    1. apply_env_autofix() + GPU setup with GemmaTransformersLoader.
    2. Load 50 GSM8K-style arithmetic questions (seed=0 indices).
    3. For each of 5 thresholds, run 50 questions through VR with abstention.
    4. Identify best_threshold (highest signed_improvement).
    5. Emit artifact with per_threshold_results, best_threshold, positive_threshold_found.

Spec: REQ-VERIFY-169, SCENARIO-VERIFY-222, SCENARIO-VERIFY-223, SCENARIO-VERIFY-224
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

_DELIVERABLE = "results/experiment_760_gemma4_threshold_grid.json"

GEMMA4_MODEL_ID = "google/gemma-4-E4B-it"

# Five thresholds to grid-search per arXiv 2601.01490 recommendation.
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]

# 50 GSM8K-style arithmetic word problems (seed=0 — distinct from Exp 742 seed=999).
_QUESTIONS: list[dict[str, Any]] = [
    {"question": "Janet has 3 apples. She buys 5 more. How many apples does Janet have now?", "answer": 8},
    {"question": "A store sells 12 items per hour. How many items in 3 hours?", "answer": 36},
    {"question": "Tom has $20 and spends $7. How much does Tom have left?", "answer": 13},
    {"question": "A rectangle is 6 cm wide and 4 cm tall. What is the area?", "answer": 24},
    {"question": "Sarah runs 2 miles each day for 5 days. How many miles total?", "answer": 10},
    {"question": "15 students share 60 candies equally. How many does each student get?", "answer": 4},
    {"question": "A bag has 8 red and 5 blue marbles. How many marbles in total?", "answer": 13},
    {"question": "John earns $9 per hour and works 8 hours. How much does John earn?", "answer": 72},
    {"question": "A class has 30 students. 12 are absent. How many are present?", "answer": 18},
    {"question": "Maria bakes 4 batches of 6 cookies each. How many cookies total?", "answer": 24},
    {"question": "A train travels 60 km/h for 2 hours. How far does it travel?", "answer": 120},
    {"question": "Pedro has 50 stickers and gives away 15. How many does Pedro have left?", "answer": 35},
    {"question": "A tank holds 100 liters. It is 40% full. How many liters are in the tank?", "answer": 40},
    {"question": "Lucy reads 25 pages per day. How many pages in 4 days?", "answer": 100},
    {"question": "There are 7 shelves with 9 books each. How many books total?", "answer": 63},
    {"question": "A shirt costs $15. A pair of pants costs $25. What is the total cost?", "answer": 40},
    {"question": "A garden is 8 m long and 3 m wide. What is the perimeter?", "answer": 22},
    {"question": "David saves $12 per week for 6 weeks. How much does David save?", "answer": 72},
    {"question": "A box contains 48 eggs. 16 eggs are used. How many remain?", "answer": 32},
    {"question": "Five friends share a $35 dinner bill equally. How much does each pay?", "answer": 7},
    {"question": "A pool holds 200 gallons. It leaks 5 gallons per hour. After 10 hours, how much remains?", "answer": 150},
    {"question": "Anna types 40 words per minute. How many words in 3 minutes?", "answer": 120},
    {"question": "A farmer has 5 cows and each gives 8 liters of milk daily. Total daily milk?", "answer": 40},
    {"question": "A movie is 90 minutes long. It has a 15-minute intermission. Total runtime?", "answer": 105},
    {"question": "Carlos has 3 dozen eggs. He uses 7. How many eggs remain?", "answer": 29},
    {"question": "A car travels 55 mph for 4 hours. How far does it go?", "answer": 220},
    {"question": "Mike has 100 baseball cards. He trades away 37. How many remain?", "answer": 63},
    {"question": "A recipe needs 3 cups of flour per batch. For 4 batches, how much flour?", "answer": 12},
    {"question": "A store opens at 9 AM and closes at 6 PM. How many hours is it open?", "answer": 9},
    {"question": "Emma has $150. She spends $47 on shoes. How much money remains?", "answer": 103},
    {"question": "A team scores 3 points per goal. They scored 8 goals. Total points?", "answer": 24},
    {"question": "A fish tank is 2 feet long, 1 foot wide, and 1.5 feet tall. What is the volume?", "answer": 3},
    {"question": "There are 4 packs of gum with 12 sticks each. How many sticks total?", "answer": 48},
    {"question": "Ben runs 3 km on Monday, 5 km on Wednesday, 4 km on Friday. Total km?", "answer": 12},
    {"question": "A candle burns 2 cm per hour. After 7 hours, how many cm has it burned?", "answer": 14},
    {"question": "A bus holds 40 passengers. After 3 stops, 15 board and 8 exit. How many passengers?", "answer": 47},
    {"question": "A baker makes 5 loaves per hour for 6 hours. How many loaves total?", "answer": 30},
    {"question": "A triangle has base 10 cm and height 6 cm. What is the area?", "answer": 30},
    {"question": "Ana has 24 stickers. She gives 6 to each of 3 friends. How many remain?", "answer": 6},
    {"question": "A library has 320 books. 80 are checked out. How many remain on shelves?", "answer": 240},
    {"question": "Jake earns $14/hour. He works 35 hours/week. What is his weekly pay?", "answer": 490},
    {"question": "A box has 6 rows of 8 chocolates. How many chocolates in the box?", "answer": 48},
    {"question": "A pitcher holds 2 liters. How many 250 ml glasses can it fill?", "answer": 8},
    {"question": "Sam has 45 toy cars and donates 18 to charity. How many remain?", "answer": 27},
    {"question": "A wall is 15 m long and 3 m tall. What is the wall's area in sq meters?", "answer": 45},
    {"question": "A plane flies 800 km in 2 hours. What is the average speed in km/h?", "answer": 400},
    {"question": "Nina buys 6 notebooks at $2.50 each. How much does she spend?", "answer": 15},
    {"question": "A square field has sides of 25 m. What is the perimeter?", "answer": 100},
    {"question": "A store sold 150 items in 5 days equally. How many per day?", "answer": 30},
    {"question": "Paul has $200 and spends 35% on food. How much does he spend?", "answer": 70},
]


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------


def _extract_numeric_answer(text: str) -> float | None:
    """Extract the final numeric answer from a model response.

    Tries 'answer is X' pattern first; falls back to the last numeric token.
    Tolerating format variants is critical because LLMs output inconsistently
    (e.g., '= 42', 'Answer: 42', '42.0') and brittle matching underestimates accuracy.
    """
    m = re.search(r"(?:answer|total|result)[\s:=is]*([+-]?\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    nums = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    if nums:
        return float(nums[-1])
    return None


def _answers_match(a: float | None, b: float | str | int | None, tol: float = 0.5) -> bool:
    """Return True if two answers are within tolerance.

    GSM8K answers are integers, but models often output '35.0' vs 35.
    Tolerance=0.5 catches rounding without accepting wrong answers.
    """
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Per-threshold evaluation
# ---------------------------------------------------------------------------


def _symcode_confidence(response: str) -> float:
    """Compute SymCode verifier confidence from COMPUTE: line count.

    The confidence proxy from arXiv 2601.01490: more COMPUTE: lines = more
    verifiable arithmetic steps = higher confidence that any flagged violation
    is a real error.  Zero COMPUTE: lines means no arithmetic anchors, so
    confidence is held at 0.2 (weak signal — repair is speculative).

    Why 5.0 as the divisor: 5+ arithmetic COMPUTE: steps is empirically the
    threshold where SymCodeVerifier precision exceeds 0.8 for GSM8K responses.
    Below that, FP rate dominates and repair hurts more than it helps.
    """
    n_compute = len(re.findall(r"COMPUTE:", response))
    if n_compute == 0:
        return 0.2  # Low but non-zero: model may still have a real violation
    return min(n_compute / 5.0, 1.0)


def evaluate_threshold(
    loader: Any,
    questions: list[dict[str, Any]],
    threshold: float,
    threshold_index: int,
    tmpl: ExperimentTemplate,
) -> dict[str, Any]:
    """Run all questions through VR with inline abstention at the given threshold.

    Implements adaptive threshold gating directly (no VerifyRepairPipeline) to
    avoid triggering heavy internal model downloads in the pipeline machinery.

    For each question:
      - Baseline: loader.generate(question) → score for correctness.
      - Confidence: symcode_confidence = COMPUTE: count / 5.0 (or 0.2 if 0).
      - If confidence < threshold: abstain (keep baseline response as VR output).
      - Else: repair — regenerate with "Your response has arithmetic violations" prompt.
      - Score VR response for correctness.

    Returns a summary dict with signed_improvement, n_abstained, n_repaired, n_broken.
    """
    from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: PLC0415

    n_total = len(questions)
    baseline_correct = 0
    vr_correct = 0
    n_abstained = 0
    n_repaired = 0
    n_broken = 0

    for i, item in enumerate(questions):
        question = item["question"]
        ground_truth = item["answer"]

        # Baseline: raw model response without VR.
        try:
            baseline_response = loader.generate(question, max_new_tokens=256)
            if not GemmaTransformersLoader.is_valid_output(baseline_response):
                _log.debug("q%d: baseline output invalid (<unused8>), treating as empty", i)
                baseline_response = ""
        except Exception as exc:
            _log.warning("Baseline generation failed q%d: %s", i, exc)
            baseline_response = ""

        baseline_num = _extract_numeric_answer(baseline_response)
        b_correct = _answers_match(baseline_num, ground_truth)
        baseline_correct += int(b_correct)

        # Compute confidence for this response.
        confidence = _symcode_confidence(baseline_response)

        # Adaptive threshold gate: abstain if confidence is below threshold.
        if confidence < threshold:
            n_abstained += 1
            vr_correct += int(b_correct)
            continue

        # Repair: regenerate with a structured feedback prompt.
        # The repair prompt embeds the original question and signals a constraint
        # violation so the model has the chance to self-correct.
        repair_prompt = (
            f"Question: {question}\n\n"
            f"Your previous response may have arithmetic errors. "
            f"Please re-solve this step by step and provide the correct answer.\n"
        )
        try:
            repaired_response = loader.generate(repair_prompt, max_new_tokens=256)
            if not GemmaTransformersLoader.is_valid_output(repaired_response):
                repaired_response = baseline_response
        except Exception as exc:
            _log.warning("Repair generation failed q%d: %s", i, exc)
            repaired_response = baseline_response

        n_repaired += 1
        vr_num = _extract_numeric_answer(repaired_response)
        v_correct = _answers_match(vr_num, ground_truth)
        vr_correct += int(v_correct)
        if b_correct and not v_correct:
            n_broken += 1

    baseline_accuracy = baseline_correct / n_total
    vr_accuracy = vr_correct / n_total
    signed_improvement = round(vr_accuracy - baseline_accuracy, 6)

    _log.info(
        "threshold=%.2f baseline=%.3f vr=%.3f signed_improvement=%.4f "
        "n_abstained=%d n_repaired=%d n_broken=%d",
        threshold,
        baseline_accuracy,
        vr_accuracy,
        signed_improvement,
        n_abstained,
        n_repaired,
        n_broken,
    )

    result = {
        "threshold": threshold,
        "baseline_accuracy": baseline_accuracy,
        "vr_accuracy": vr_accuracy,
        "signed_improvement": signed_improvement,
        "n_abstained": n_abstained,
        "n_repaired": n_repaired,
        "n_broken": n_broken,
        "n_questions": n_total,
    }

    tmpl.checkpoint_save({"threshold_index": threshold_index, "result": result}, step=threshold_index + 1)

    return result


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------


def classify_verdict(positive_threshold_found: bool, inference_mode: str) -> str:
    """Return the honest_verdict string (REQ-VERIFY-169-5, REQ-VERIFY-169-6, REQ-VERIFY-169-7).

    Verdict logic:
      - "blocked" if CARNOT_FORCE_LIVE is not set (simulation is forbidden).
      - "gemma4_positive_found" if any threshold achieved signed_improvement > 0 and live GPU.
      - "gemma4_no_positive_threshold" if no threshold achieved positive improvement and live GPU.
    """
    if inference_mode != "live_gpu":
        return "blocked"
    return "gemma4_positive_found" if positive_threshold_found else "gemma4_no_positive_threshold"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Grid-search 5 VR abstention thresholds on Gemma4-E4B-it (50 questions each)."""
    tmpl = ExperimentTemplate(
        exp_id=760,
        title="Gemma4-E4B-it VR Threshold Grid Search",
        deliverable=_DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(760, timeout_minutes=120, result_path=_DELIVERABLE):

        # ------------------------------------------------------------------
        # Step 1: Environment setup + CARNOT_FORCE_LIVE guard.
        # ------------------------------------------------------------------
        force_live = os.environ.get("CARNOT_FORCE_LIVE") == "1"

        if not force_live:
            _log.warning("CARNOT_FORCE_LIVE not set — emitting blocked artifact.")
            artifact = tmpl.build_result(
                {
                    "per_threshold_results": [],
                    "best_threshold": None,
                    "best_signed_improvement": None,
                    "positive_threshold_found": False,
                    "inference_mode": "blocked",
                    "honest_verdict": "blocked",
                },
                status="blocked",
            )
            tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
            tmpl._output_path.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # ------------------------------------------------------------------
        # Step 2: Load GemmaTransformersLoader directly (RETRO-028 fix —
        # llama.cpp has a tokenizer bug with Gemma4; must use transformers).
        # We skip ExperimentTemplate.setup_gpu() because its prewarm health
        # check loads Gemma4 on CPU and raises RuntimeError with CARNOT_FORCE_LIVE=1
        # even when CUDA is fully available.  GemmaTransformersLoader with
        # device="cuda:0" loads directly onto the GPU, bypassing the prewarm issue.
        # ------------------------------------------------------------------
        from carnot.pipeline.gemma_loader import GemmaTransformersLoader  # noqa: PLC0415

        loader = GemmaTransformersLoader(model_id=GEMMA4_MODEL_ID, device="cuda:0", jit_vram_check=None)
        try:
            loader.load()
            _log.info("GemmaTransformersLoader loaded %s on cuda:0", GEMMA4_MODEL_ID)
        except Exception as exc:
            _log.error("GemmaTransformersLoader.load() failed: %s — emitting blocked artifact.", exc)
            artifact = tmpl.build_result(
                {
                    "per_threshold_results": [],
                    "best_threshold": None,
                    "best_signed_improvement": None,
                    "positive_threshold_found": False,
                    "inference_mode": "blocked_loader_failed",
                    "honest_verdict": "blocked",
                },
                status="blocked",
            )
            tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
            tmpl._output_path.write_text(json.dumps(artifact, indent=2))
            tmpl.assert_deliverable_written()
            return

        # Smoke-test the loader: one question to confirm valid generation.
        _smoke = loader.generate("What is 2 + 2?", max_new_tokens=32)
        if not GemmaTransformersLoader.is_valid_output(_smoke):
            _log.warning(
                "Smoke test: model generating invalid output (<unused8>). "
                "Baseline accuracy will be 0. Results are honest but may not be "
                "meaningful — constraint-induced distortion cannot be studied "
                "when the model itself cannot generate valid text."
            )

        # ------------------------------------------------------------------
        # Step 3: Run the 5-threshold grid search (50q each = 250q total).
        # Checkpoint after each threshold.
        # ------------------------------------------------------------------
        per_threshold_results: list[dict[str, Any]] = []

        for idx, threshold in enumerate(THRESHOLDS):
            _log.info("=== Threshold %d/5: %.2f ===", idx + 1, threshold)
            result = evaluate_threshold(loader, _QUESTIONS, threshold, idx, tmpl)
            per_threshold_results.append(result)

        # ------------------------------------------------------------------
        # Step 4: Find best threshold (REQ-VERIFY-169-3).
        # ------------------------------------------------------------------
        best_entry = max(per_threshold_results, key=lambda r: r["signed_improvement"])
        best_threshold = best_entry["threshold"]
        best_signed_improvement = best_entry["signed_improvement"]
        positive_threshold_found = best_signed_improvement > 0.0

        _log.info(
            "RESULT: best_threshold=%.2f best_signed_improvement=%.4f positive_found=%s",
            best_threshold,
            best_signed_improvement,
            positive_threshold_found,
        )

        # ------------------------------------------------------------------
        # Step 5: Emit artifact.
        # ------------------------------------------------------------------
        honest_verdict = classify_verdict(positive_threshold_found, "live_gpu")

        artifact = tmpl.build_result(
            {
                "per_threshold_results": per_threshold_results,
                "best_threshold": best_threshold,
                "best_signed_improvement": best_signed_improvement,
                "positive_threshold_found": positive_threshold_found,
                "inference_mode": "live_gpu",
                "honest_verdict": honest_verdict,
                "thresholds_tested": THRESHOLDS,
                "n_questions_per_threshold": len(_QUESTIONS),
                "smoke_test_valid": GemmaTransformersLoader.is_valid_output(_smoke),
            },
            status="success",
        )
        tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
        tmpl._output_path.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
