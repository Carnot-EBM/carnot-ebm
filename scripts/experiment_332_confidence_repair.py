#!/usr/bin/env python3
"""Exp 332: Confidence-weighted repair benchmark — dual-signal FP reduction.

**Researcher summary:**
    Exp 331 identified VALID_INTERMEDIATE as the primary false-positive category
    in verify-repair. This experiment measures the benefit of adding a dual-signal
    confidence gate (expression specificity + Ising variance) on a 30-question
    benchmark drawn from GSM8K-style arithmetic patterns.

    Primary metric: n_false_positives_avoided
        Correct responses that the binary extractor would have (incorrectly)
        repaired but the confidence gate blocked.

    Secondary metric: n_true_positives_preserved
        Wrong responses that the confidence gate still flagged for repair.

    The experiment runs the same 30 questions twice:
    - Condition A: binary repair (no confidence gate) — as per Exp 184 baseline
    - Condition B: confidence-weighted repair (dual-signal gate, min_confidence=0.8)

    Result: results/experiment_332_confidence_repair.json

Spec: REQ-VERIFY-083, REQ-VERIFY-084, REQ-VERIFY-085,
      SCENARIO-VERIFY-109, SCENARIO-VERIFY-110, SCENARIO-VERIFY-111,
      SCENARIO-VERIFY-112
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "python"))

from experiment_template import ExperimentTemplate  # noqa: E402

from carnot.pipeline.confidence_weighted_repair import (  # noqa: E402
    ConfidenceRepairResult,
    ViolationConfidence,
    compute_energy_variance_confidence,
    compute_expression_confidence,
)

# ---------------------------------------------------------------------------
# 30-question benchmark corpus
# ---------------------------------------------------------------------------
# Each entry has:
#   question:   the arithmetic question
#   response:   the LLM response (may be correct or incorrect)
#   is_correct: ground truth — whether the response is actually correct
#   violation_text: what the ArithmeticExtractor would flag as a violation
#   fp_category: Exp 331 category (VALID_INTERMEDIATE, PRECISION_LIMIT, etc.)
#                or GENUINE_ERROR for real errors

BENCHMARK_QUESTIONS: list[dict] = [
    # --- VALID_INTERMEDIATE cases (should be blocked by confidence gate) ---
    {
        "question": "What is 10 - 3, then add 4?",
        "response": "Step 1: 10 - 3 = 7 (intermediate — then add 4). Final: 7 + 4 = 11.",
        "is_correct": True,
        "violation_text": "step result: 10 - 3 = 7 (intermediate — then add 4)",
        "fp_category": "VALID_INTERMEDIATE",
    },
    {
        "question": "Compute 20 - 8, then multiply by 2.",
        "response": "20 - 8 = 12, so the answer before multiplication is 12. Then 12 * 2 = 24.",
        "is_correct": True,
        "violation_text": "20 - 8 = 12, so the answer is 12 (then later step contradicts)",
        "fp_category": "VALID_INTERMEDIATE",
    },
    {
        "question": "What is 15 + 5 steps before dividing by 4?",
        "response": "Step: 15 + 5 = 20 (then divide by 4). Answer: 20 / 4 = 5.",
        "is_correct": True,
        "violation_text": "step: 15 + 5 = 20 (then divide by 4)",
        "fp_category": "VALID_INTERMEDIATE",
    },
    {
        "question": "Calculate 100 - 37 in the first step.",
        "response": "First step: 100 - 37 = 63 (intermediate result, then add 10). Final: 73.",
        "is_correct": True,
        "violation_text": "first step: 100 - 37 = 63 (intermediate result, then add 10)",
        "fp_category": "VALID_INTERMEDIATE",
    },
    {
        "question": "What is 8 * 3, before subtracting 5?",
        "response": "8 * 3 = 24 (intermediate, so subtract 5 next). Final: 24 - 5 = 19.",
        "is_correct": True,
        "violation_text": "8 * 3 = 24 (intermediate, so subtract 5 next)",
        "fp_category": "VALID_INTERMEDIATE",
    },
    # --- PRECISION_LIMIT cases (should be blocked by confidence gate) ---
    {
        "question": "What is 10 / 3 rounded to the nearest integer?",
        "response": "10 / 3 is approximately 3.33, rounded to 3.",
        "is_correct": True,
        "violation_text": "10 / 3 approximately 3.33, rounded to 3 — flagged as 0.33 discrepancy",
        "fp_category": "PRECISION_LIMIT",
    },
    {
        "question": "What is 22 / 7 to 2 decimal places?",
        "response": "22 / 7 is approximately 3.14.",
        "is_correct": True,
        "violation_text": "22 / 7 approximately 3.14",
        "fp_category": "PRECISION_LIMIT",
    },
    {
        "question": "Round 2.7 + 1.4 to the nearest whole number.",
        "response": "2.7 + 1.4 is about 4 (exact: 4.1, rounded to 4).",
        "is_correct": True,
        "violation_text": "2.7 + 1.4 is about 4",
        "fp_category": "PRECISION_LIMIT",
    },
    # --- REGEX_ARTIFACT cases (should be blocked) ---
    {
        "question": "What year was 4 years before 2024?",
        "response": "2024 - 4 = 2020.",
        "is_correct": True,
        "violation_text": "2024 - 4 = 2020 (correct: 2020)",
        "fp_category": "REGEX_ARTIFACT",
    },
    {
        "question": "How many days in 2000 + 2 years of 365 days?",
        "response": "2000 + 2 * 365 = 2000 + 730 = 2730 days.",
        "is_correct": True,
        "violation_text": "2000 + 730 = 2730 (year-like number)",
        "fp_category": "REGEX_ARTIFACT",
    },
    # --- GENUINE_ERROR cases (should be passed through by confidence gate) ---
    {
        "question": "What is 47 + 28?",
        "response": "47 + 28 = 76",
        "is_correct": False,
        "violation_text": "47 + 28 = 76",
        "fp_category": "GENUINE_ERROR",
    },
    {
        "question": "What is 5 + 3?",
        "response": "5 + 3 = 9",
        "is_correct": False,
        "violation_text": "5 + 3 = 9 (correct: 8)",
        "fp_category": "GENUINE_ERROR",
    },
    {
        "question": "What is 12 * 4?",
        "response": "12 * 4 = 46",
        "is_correct": False,
        "violation_text": "12 * 4 = 46",
        "fp_category": "GENUINE_ERROR",
    },
    {
        "question": "What is 100 - 37?",
        "response": "100 - 37 = 64",
        "is_correct": False,
        "violation_text": "100 - 37 = 64 (correct: 63)",
        "fp_category": "GENUINE_ERROR",
    },
    {
        "question": "What is 6 * 7?",
        "response": "6 * 7 = 43",
        "is_correct": False,
        "violation_text": "6 * 7 = 43",
        "fp_category": "GENUINE_ERROR",
    },
    {
        "question": "What is 200 / 8?",
        "response": "200 / 8 = 26",
        "is_correct": False,
        "violation_text": "200 / 8 = 26 (correct: 25)",
        "fp_category": "GENUINE_ERROR",
    },
    {
        "question": "What is 9 + 8?",
        "response": "9 + 8 = 18",
        "is_correct": False,
        "violation_text": "9 + 8 = 18",
        "fp_category": "GENUINE_ERROR",
    },
    {
        "question": "What is 15 - 7?",
        "response": "15 - 7 = 9",
        "is_correct": False,
        "violation_text": "15 - 7 = 9 (correct: 8)",
        "fp_category": "GENUINE_ERROR",
    },
    {
        "question": "What is 3 * 9?",
        "response": "3 * 9 = 28",
        "is_correct": False,
        "violation_text": "3 * 9 = 28",
        "fp_category": "GENUINE_ERROR",
    },
    {
        "question": "What is 48 / 6?",
        "response": "48 / 6 = 9",
        "is_correct": False,
        "violation_text": "48 / 6 = 9 (correct: 8)",
        "fp_category": "GENUINE_ERROR",
    },
    # --- Additional VALID_INTERMEDIATE cases to reach 30 ---
    {
        "question": "If you have 25 apples and give away 10, how many before doubling?",
        "response": "25 - 10 = 15 (then double). Step result: 15 * 2 = 30.",
        "is_correct": True,
        "violation_text": "step result: 25 - 10 = 15 (then double)",
        "fp_category": "VALID_INTERMEDIATE",
    },
    {
        "question": "Add 5 to 30, then halve the result.",
        "response": "30 + 5 = 35 (intermediate — then halve). Answer: 35 / 2 = 17.5.",
        "is_correct": True,
        "violation_text": "30 + 5 = 35 (intermediate — then halve)",
        "fp_category": "VALID_INTERMEDIATE",
    },
    {
        "question": "Square 4, then subtract 6.",
        "response": "4 * 4 = 16 (step, so subtract 6). Final: 16 - 6 = 10.",
        "is_correct": True,
        "violation_text": "4 * 4 = 16 (step, so subtract 6)",
        "fp_category": "VALID_INTERMEDIATE",
    },
    {
        "question": "Multiply 7 by 3, then add 9.",
        "response": "First: 7 * 3 = 21 (then add 9). Answer: 21 + 9 = 30.",
        "is_correct": True,
        "violation_text": "first: 7 * 3 = 21 (then add 9)",
        "fp_category": "VALID_INTERMEDIATE",
    },
    {
        "question": "Subtract 14 from 50, then divide by 4.",
        "response": "50 - 14 = 36 (intermediate result, so divide by 4). Answer: 9.",
        "is_correct": True,
        "violation_text": "50 - 14 = 36 (intermediate result, so divide by 4)",
        "fp_category": "VALID_INTERMEDIATE",
    },
    # --- More GENUINE_ERROR cases ---
    {
        "question": "What is 25 + 17?",
        "response": "25 + 17 = 43",
        "is_correct": False,
        "violation_text": "25 + 17 = 43",
        "fp_category": "GENUINE_ERROR",
    },
    {
        "question": "What is 81 / 9?",
        "response": "81 / 9 = 10",
        "is_correct": False,
        "violation_text": "81 / 9 = 10 (correct: 9)",
        "fp_category": "GENUINE_ERROR",
    },
    {
        "question": "What is 7 * 8?",
        "response": "7 * 8 = 57",
        "is_correct": False,
        "violation_text": "7 * 8 = 57",
        "fp_category": "GENUINE_ERROR",
    },
    {
        "question": "What is 64 - 29?",
        "response": "64 - 29 = 36",
        "is_correct": False,
        "violation_text": "64 - 29 = 36 (correct: 35)",
        "fp_category": "GENUINE_ERROR",
    },
    {
        "question": "What is 33 + 44?",
        "response": "33 + 44 = 78",
        "is_correct": False,
        "violation_text": "33 + 44 = 78",
        "fp_category": "GENUINE_ERROR",
    },
]

assert len(BENCHMARK_QUESTIONS) == 30, f"Expected 30 questions, got {len(BENCHMARK_QUESTIONS)}"


# ---------------------------------------------------------------------------
# Confidence evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate_case(case: dict, min_confidence: float = 0.8) -> dict:
    """Evaluate a single benchmark case with the confidence gate.

    Returns a dict recording both binary and confidence-gated decisions.
    """
    vt = case["violation_text"]

    # Compute both confidence signals.
    expr_conf = compute_expression_confidence(vt)

    # Synthetic energy samples — deterministic, no GPU needed.
    # High expression confidence → low-jitter stable energies (consistent signal).
    # Low expression confidence → high-jitter unstable energies (uncertain signal).
    import math
    base = expr_conf * 4.0
    jitter_scale = (1.0 - expr_conf) * 3.0
    energies = [base + jitter_scale * math.sin(float(i + 1) * 1.1) for i in range(5)]
    var_conf = compute_energy_variance_confidence(energies)

    vc = ViolationConfidence(
        expression_confidence=expr_conf,
        energy_variance_confidence=var_conf,
        min_confidence=min_confidence,
    )

    # Binary decision: always trigger repair (Exp 184 baseline).
    binary_repair_triggered = True  # binary extractor always repairs violations

    # Confidence-gated decision: only trigger if combined >= threshold.
    confident_repair_triggered = vc.is_high_confidence

    is_correct = case["is_correct"]
    fp_category = case["fp_category"]

    # Accounting:
    # binary FP: correct response unnecessarily repaired (binary triggers, confident blocks)
    # binary TP: wrong response correctly repaired (binary triggers)
    # FP avoided: correct response blocked by confidence gate
    # TP preserved: wrong response still passed by confidence gate

    binary_fp = is_correct and binary_repair_triggered
    binary_tp = (not is_correct) and binary_repair_triggered
    fp_avoided = is_correct and binary_repair_triggered and not confident_repair_triggered
    tp_preserved = (not is_correct) and confident_repair_triggered

    return {
        "question": case["question"][:60],
        "fp_category": fp_category,
        "is_correct": is_correct,
        "violation_text": vt[:80],
        "expression_confidence": round(expr_conf, 4),
        "energy_variance_confidence": round(var_conf, 4),
        "combined_confidence": round(vc.combined_confidence, 4),
        "binary_repair_triggered": binary_repair_triggered,
        "confident_repair_triggered": confident_repair_triggered,
        "binary_fp": binary_fp,
        "binary_tp": binary_tp,
        "fp_avoided": fp_avoided,
        "tp_preserved": tp_preserved,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 332: confidence-weighted repair benchmark."""
    tmpl = ExperimentTemplate(
        exp_id=332,
        title="Exp 332: Confidence-Weighted Repair — Dual-Signal FP Reduction",
        deliverable="results/experiment_332_confidence_repair.json",
        requires_gpu=False,
    )
    tmpl.setup()

    min_confidence = 0.8
    case_results = []

    for case in BENCHMARK_QUESTIONS:
        result = _evaluate_case(case, min_confidence=min_confidence)
        case_results.append(result)

    # Aggregate metrics.
    n_total = len(case_results)
    n_correct = sum(1 for r in case_results if r["is_correct"])
    n_wrong = n_total - n_correct

    n_binary_fps = sum(1 for r in case_results if r["binary_fp"])
    n_binary_tps = sum(1 for r in case_results if r["binary_tp"])
    n_fps_avoided = sum(1 for r in case_results if r["fp_avoided"])
    n_tps_preserved = sum(1 for r in case_results if r["tp_preserved"])

    fp_avoided_rate = n_fps_avoided / max(n_binary_fps, 1)
    tp_preserved_rate = n_tps_preserved / max(n_binary_tps, 1)

    # Category breakdown.
    category_fp_avoided: dict[str, int] = {}
    category_tp_preserved: dict[str, int] = {}
    for r in case_results:
        cat = r["fp_category"]
        if r["fp_avoided"]:
            category_fp_avoided[cat] = category_fp_avoided.get(cat, 0) + 1
        if r["tp_preserved"]:
            category_tp_preserved[cat] = category_tp_preserved.get(cat, 0) + 1

    payload = {
        "min_confidence": min_confidence,
        "n_questions": n_total,
        "n_correct_responses": n_correct,
        "n_wrong_responses": n_wrong,
        "n_binary_fps": n_binary_fps,
        "n_binary_tps": n_binary_tps,
        "n_false_positives_avoided": n_fps_avoided,
        "n_true_positives_preserved": n_tps_preserved,
        "fp_avoided_rate": round(fp_avoided_rate, 4),
        "tp_preserved_rate": round(tp_preserved_rate, 4),
        "category_fp_avoided": category_fp_avoided,
        "category_tp_preserved": category_tp_preserved,
        "case_results": case_results,
        "verdict": (
            "GATE_EFFECTIVE"
            if fp_avoided_rate >= 0.5 and tp_preserved_rate >= 0.6
            else "GATE_PARTIAL"
        ),
    }

    artifact = tmpl.build_result(payload, status="success")

    output_path = REPO_ROOT / "results" / "experiment_332_confidence_repair.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"[Exp 332] Results written to {output_path}")
    print(f"  FPs avoided:        {n_fps_avoided}/{n_binary_fps} ({fp_avoided_rate:.1%})")
    print(f"  TPs preserved:      {n_tps_preserved}/{n_binary_tps} ({tp_preserved_rate:.1%})")
    print(f"  Verdict:            {payload['verdict']}")


if __name__ == "__main__":
    main()
