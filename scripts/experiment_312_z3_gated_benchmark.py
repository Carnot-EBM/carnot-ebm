#!/usr/bin/env python3
"""Exp 312: Z3-gated repair benchmark — 30-question mini-benchmark.

**Researcher summary:**
    Exp 311 confirmed that NL2Z3Extractor has the lowest FP rate of the three
    extractors, making it the natural "first gate" in a two-stage pipeline.
    This experiment wires Z3 as that gate: Z3 SAT → skip Ising (cheap),
    Z3 UNSAT → full Ising + LLM repair (confident violation).

    Key metrics:
    - z3_gate_skip_rate:       fraction of questions where Z3 SAT skipped Ising
    - ising_trigger_rate:      fraction where Z3 UNSAT/unknown triggered Ising
    - net_accuracy_improvement: honest delta (may be 0 in CI mode — reported truthfully)

    In CI mode (no CARNOT_FORCE_LIVE), NL2Z3Extractor always returns "unknown"
    so all questions go through the Ising fallback path and skip_rate == 0.
    This is correct and expected — the gate only fires in production.

**Detailed explanation for engineers:**
    Corpus design (30 questions):
    - 15 "correct" responses: response text contains arithmetic that matches
      the expected answer.  Z3 SAT (or unknown in CI) → skip Ising.
    - 15 "incorrect" responses: response text contains a deliberate arithmetic
      error.  Z3 UNSAT (or unknown in CI) → trigger Ising repair.

    Each question is run through Z3GatedRepair.repair().  The pipeline uses
    VerifyRepairPipeline with no model (verify-only mode) — no GPU needed.

    Constraint:
    - z3_gate_skip_rate + ising_trigger_rate == 1.0 always (every question
      takes exactly one path).

    Output: results/experiment_312_z3_gated_results.json

Spec: REQ-REPAIR-010, REQ-REPAIR-011,
      SCENARIO-REPAIR-020, SCENARIO-REPAIR-021, SCENARIO-REPAIR-022,
      SCENARIO-REPAIR-023
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Benchmark corpus — 30 labeled questions (deterministic, no GPU)
# ---------------------------------------------------------------------------

# 15 correct responses: arithmetic is valid.
_CORRECT_RESPONSES: list[dict[str, str]] = [
    {
        "question": "A baker makes 12 muffins per batch. She bakes 4 batches. How many muffins does she have?",
        "response": "12 muffins per batch × 4 batches = 48 muffins total. The answer is 48.",
        "label": "correct",
    },
    {
        "question": "John earns $15 per hour. He works 8 hours. What does he earn?",
        "response": "15 × 8 = 120. John earns $120.",
        "label": "correct",
    },
    {
        "question": "A car travels 60 km/h for 3 hours. How far does it travel?",
        "response": "60 × 3 = 180. The car travels 180 km.",
        "label": "correct",
    },
    {
        "question": "There are 5 boxes with 7 apples each. How many apples in total?",
        "response": "5 × 7 = 35. There are 35 apples.",
        "label": "correct",
    },
    {
        "question": "A class has 30 students. 12 are absent. How many are present?",
        "response": "30 − 12 = 18. There are 18 students present.",
        "label": "correct",
    },
    {
        "question": "What is 144 divided by 12?",
        "response": "144 ÷ 12 = 12. The answer is 12.",
        "label": "correct",
    },
    {
        "question": "Sarah saves $25 per week for 10 weeks. How much has she saved?",
        "response": "25 × 10 = 250. Sarah has saved $250.",
        "label": "correct",
    },
    {
        "question": "A train travels at 90 km/h for 2.5 hours. What distance does it cover?",
        "response": "90 × 2.5 = 225. The train covers 225 km.",
        "label": "correct",
    },
    {
        "question": "If 3 pencils cost $0.75, what does 1 pencil cost?",
        "response": "0.75 ÷ 3 = 0.25. Each pencil costs $0.25.",
        "label": "correct",
    },
    {
        "question": "A tank holds 200 liters. It is 40% full. How many liters are in the tank?",
        "response": "200 × 0.40 = 80. There are 80 liters in the tank.",
        "label": "correct",
    },
    {
        "question": "What is 7 squared?",
        "response": "7 × 7 = 49. The answer is 49.",
        "label": "correct",
    },
    {
        "question": "A rectangle is 8 m wide and 5 m long. What is its area?",
        "response": "8 × 5 = 40. The area is 40 m².",
        "label": "correct",
    },
    {
        "question": "How many hours are in 3.5 days?",
        "response": "3.5 × 24 = 84. There are 84 hours in 3.5 days.",
        "label": "correct",
    },
    {
        "question": "A factory produces 500 units per day. How many in 6 days?",
        "response": "500 × 6 = 3000. The factory produces 3000 units.",
        "label": "correct",
    },
    {
        "question": "What is 15% of 200?",
        "response": "200 × 0.15 = 30. 15% of 200 is 30.",
        "label": "correct",
    },
]

# 15 incorrect responses: deliberate arithmetic errors.
_INCORRECT_RESPONSES: list[dict[str, str]] = [
    {
        "question": "A baker makes 12 muffins per batch. She bakes 4 batches. How many muffins does she have?",
        "response": "12 muffins per batch × 4 batches = 52 muffins total. The answer is 52.",
        "label": "incorrect",
    },
    {
        "question": "John earns $15 per hour. He works 8 hours. What does he earn?",
        "response": "15 × 8 = 115. John earns $115.",
        "label": "incorrect",
    },
    {
        "question": "A car travels 60 km/h for 3 hours. How far does it travel?",
        "response": "60 + 3 = 63. The car travels 63 km.",
        "label": "incorrect",
    },
    {
        "question": "There are 5 boxes with 7 apples each. How many apples in total?",
        "response": "5 + 7 = 12. There are 12 apples.",
        "label": "incorrect",
    },
    {
        "question": "A class has 30 students. 12 are absent. How many are present?",
        "response": "30 − 12 = 28. There are 28 students present.",
        "label": "incorrect",
    },
    {
        "question": "What is 144 divided by 12?",
        "response": "144 ÷ 12 = 11. The answer is 11.",
        "label": "incorrect",
    },
    {
        "question": "Sarah saves $25 per week for 10 weeks. How much has she saved?",
        "response": "25 + 10 = 35. Sarah has saved $35.",
        "label": "incorrect",
    },
    {
        "question": "A train travels at 90 km/h for 2.5 hours. What distance does it cover?",
        "response": "90 + 2.5 = 92.5. The train covers 92.5 km.",
        "label": "incorrect",
    },
    {
        "question": "If 3 pencils cost $0.75, what does 1 pencil cost?",
        "response": "0.75 ÷ 3 = 0.30. Each pencil costs $0.30.",
        "label": "incorrect",
    },
    {
        "question": "A tank holds 200 liters. It is 40% full. How many liters are in the tank?",
        "response": "200 × 0.40 = 90. There are 90 liters in the tank.",
        "label": "incorrect",
    },
    {
        "question": "What is 7 squared?",
        "response": "7 + 7 = 14. The answer is 14.",
        "label": "incorrect",
    },
    {
        "question": "A rectangle is 8 m wide and 5 m long. What is its area?",
        "response": "8 + 5 = 13. The area is 13 m².",
        "label": "incorrect",
    },
    {
        "question": "How many hours are in 3.5 days?",
        "response": "3.5 × 24 = 72. There are 72 hours in 3.5 days.",
        "label": "incorrect",
    },
    {
        "question": "A factory produces 500 units per day. How many in 6 days?",
        "response": "500 + 6 = 506. The factory produces 506 units.",
        "label": "incorrect",
    },
    {
        "question": "What is 15% of 200?",
        "response": "200 × 0.15 = 45. 15% of 200 is 45.",
        "label": "incorrect",
    },
]


def build_corpus() -> list[dict[str, str]]:
    """Build the 30-question labeled corpus (15 correct + 15 incorrect).

    **Detailed explanation for engineers:**
        The corpus is deterministic and synthetic so it runs in CI without
        GPU access.  Correct responses have arithmetic that Z3 can verify
        as SAT; incorrect responses have deliberate errors that Z3 returns
        UNSAT for (when CARNOT_FORCE_LIVE=1) or unknown for (in CI mode).

    Returns:
        List of 30 dicts with keys: question, response, label.
    """
    corpus = _CORRECT_RESPONSES + _INCORRECT_RESPONSES
    assert len(corpus) == 30, f"Expected 30 questions, got {len(corpus)}"
    n_correct = sum(1 for q in corpus if q["label"] == "correct")
    n_incorrect = sum(1 for q in corpus if q["label"] == "incorrect")
    assert n_correct >= 10, f"Need ≥10 correct, got {n_correct}"
    assert n_incorrect >= 10, f"Need ≥10 incorrect, got {n_incorrect}"
    return corpus


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(corpus: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Run Z3GatedRepair on every question in the corpus and collect results.

    **Detailed explanation for engineers:**
        Each question is run through Z3GatedRepair.repair().  The gate uses
        a fresh VerifyRepairPipeline instance with no loaded model (verify-only
        mode — no GPU needed, no LLM calls in CI).

        In CI mode (CARNOT_FORCE_LIVE not set), NL2Z3Extractor always returns
        "unknown", so ising_trigger_rate == 1.0 and skip_rate == 0.0.
        In live mode, the split depends on how many responses are SAT.

    Args:
        corpus: 30-question labeled list from build_corpus().

    Returns:
        List of per-question result dicts for the artifact.
    """
    from carnot.pipeline.z3_gated_repair import Z3GatedRepair
    from carnot.pipeline.verify_repair import VerifyRepairPipeline
    from carnot.pipeline.nl2z3_extractor import NL2Z3Extractor

    # Pipeline with no model — verify-only mode.
    pipeline = VerifyRepairPipeline(
        model=None,
        domains=["reasoning"],
        max_repairs=1,
        extractor=None,
        semantic_grounding_verifier=None,
        semantic_verifier_v2=None,
        timeout_seconds=30,
        memory=None,
    )

    extractor = NL2Z3Extractor()
    gate = Z3GatedRepair(
        nl2z3_extractor=extractor,
        ising_pipeline=pipeline,
        confidence_threshold=0.8,
    )

    rows: list[dict[str, Any]] = []
    for entry in corpus:
        question = entry["question"]
        response = entry["response"]
        label = entry["label"]

        result = gate.repair(question, response, domain="reasoning")

        rows.append(
            {
                "question": question,
                "label": label,
                "z3_status": result.z3_status,
                "ising_triggered": result.ising_triggered,
                "ising_violations": result.ising_violations,
                "repair_attempted": result.repair_attempted,
                "repaired": result.repaired,
                "improvement": result.improvement,
                "runtime_ms": round(result.runtime_ms, 3),
            }
        )

    return rows


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question rows into benchmark metrics.

    **Detailed explanation for engineers:**
        skip_rate and trigger_rate always sum to 1.0 because every question
        takes exactly one path.  net_accuracy_improvement is honest: 0 is
        valid and common in CI mode (no repair possible without a model).

    Args:
        rows: Per-question result dicts from run_benchmark().

    Returns:
        Metrics dict: z3_gate_skip_rate, ising_trigger_rate,
        net_accuracy_improvement, n_repaired, n_triggered, n_skipped.
    """
    n = len(rows)
    if n == 0:
        return {
            "z3_gate_skip_rate": 0.0,
            "ising_trigger_rate": 0.0,
            "net_accuracy_improvement": 0,
            "n_repaired": 0,
            "n_triggered": 0,
            "n_skipped": 0,
        }

    n_skipped = sum(1 for r in rows if not r["ising_triggered"])
    n_triggered = sum(1 for r in rows if r["ising_triggered"])
    n_repaired = sum(1 for r in rows if r["repaired"])

    skip_rate = n_skipped / n
    trigger_rate = n_triggered / n

    # Invariant: skip_rate + trigger_rate == 1.0
    assert abs(skip_rate + trigger_rate - 1.0) < 1e-9, (
        f"skip_rate ({skip_rate}) + trigger_rate ({trigger_rate}) != 1.0"
    )

    return {
        "z3_gate_skip_rate": round(skip_rate, 4),
        "ising_trigger_rate": round(trigger_rate, 4),
        "net_accuracy_improvement": n_repaired,
        "n_repaired": n_repaired,
        "n_triggered": n_triggered,
        "n_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the Exp 312 Z3-gated benchmark and write the artifact."""
    tmpl = ExperimentTemplate(
        exp_id=312,
        title="Z3-gated repair benchmark — 30-question mini-benchmark",
        deliverable="results/experiment_312_z3_gated_results.json",
        requires_gpu=False,
    )
    tmpl.setup()

    corpus = build_corpus()
    n_correct = sum(1 for q in corpus if q["label"] == "correct")
    n_incorrect = sum(1 for q in corpus if q["label"] == "incorrect")

    print(
        f"[Exp 312] Running Z3-gated repair on {len(corpus)} questions "
        f"({n_correct} correct, {n_incorrect} incorrect)"
    )

    start = time.monotonic()
    rows = run_benchmark(corpus)
    elapsed = time.monotonic() - start

    metrics = compute_metrics(rows)

    artifact = tmpl.build_result(
        {
            "n_questions": len(corpus),
            "n_correct_baseline": n_correct,
            "n_incorrect_baseline": n_incorrect,
            "z3_gate_skip_rate": metrics["z3_gate_skip_rate"],
            "ising_trigger_rate": metrics["ising_trigger_rate"],
            "net_accuracy_improvement": metrics["net_accuracy_improvement"],
            "n_repaired": metrics["n_repaired"],
            "n_triggered": metrics["n_triggered"],
            "n_skipped": metrics["n_skipped"],
            "benchmark_elapsed_s": round(elapsed, 3),
            "results": rows,
            "inference_mode": "CI (no live LLM)" if not __import__("os").environ.get("CARNOT_FORCE_LIVE") else "live",
        },
        status="success",
    )

    out_path = _REPO_ROOT / "results" / "experiment_312_z3_gated_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"[Exp 312] z3_gate_skip_rate     = {metrics['z3_gate_skip_rate']:.4f}")
    print(f"[Exp 312] ising_trigger_rate    = {metrics['ising_trigger_rate']:.4f}")
    print(f"[Exp 312] net_accuracy_improv   = {metrics['net_accuracy_improvement']}")
    print(f"[Exp 312] elapsed               = {elapsed:.2f}s")
    print(f"[Exp 312] artifact written to   {out_path}")


if __name__ == "__main__":
    main()
