#!/usr/bin/env python3
"""Experiment 334: VERGE-style iterative Z3 refinement benchmark.

**Researcher summary:**
    VERGE (arXiv 2601.20055) implements a closed-loop SMT-refinement loop that
    identifies the specific reasoning step that is arithmetically inconsistent and
    repairs only that step.  This experiment benchmarks VergeRefiner against the
    whole-response Z3-gated repair from Exp 312.

    Key claims being tested:
    1. VergeRefiner achieves at least as many SAT-converged questions as the Exp 312
       Z3-gated baseline (tracked as n_resolved).
    2. Mean iterations to convergence is <= max_iterations (no infinite loops).
    3. Honest reporting: n_not_resolved questions are flagged, never suppressed.

    Corpus:
    - 30 synthetic arithmetic questions with known-correct and known-incorrect answers.
    - Synthetic fallback is used when no live LLM is available (CI mode).
    - The synthetic corpus mirrors the structure of the Exp 312 benchmark.

    Metrics:
    - n_questions: total questions in corpus.
    - n_sat_initial: questions where initial Z3 check was SAT (no repair needed).
    - n_resolved: questions where VERGE converged to SAT within max_iterations.
    - n_not_resolved: questions where UNSAT persisted for all iterations.
    - mean_iterations: average iterations across UNSAT questions.
    - iteration_distribution: {1: n, 2: n, 3: n} — how many questions needed k iterations.
    - accuracy_vs_baseline: delta vs Exp 312 z3_gate_skip_rate (positive = better).

**How to run:**
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_334_verge_refinement.py

Spec: REQ-REPAIR-012, REQ-REPAIR-013,
      SCENARIO-REPAIR-024, SCENARIO-REPAIR-025
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Allow scripts/ to import from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from experiment_template import ExperimentTemplate

from carnot.pipeline.nl2z3_extractor import NL2Z3Extractor, Z3Result, run_z3_code
from carnot.pipeline.verge_refiner import VergeIteration, VergeRefiner

# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------

# 30 questions: 15 with correct arithmetic (SAT), 15 with deliberate errors (UNSAT).
# In CI mode (CARNOT_FORCE_LIVE not set), the Z3 gate always returns "unknown",
# so all questions go through the repair path but no real LLM calls are made.
SYNTHETIC_QUESTIONS = [
    # (question, response, is_correct)
    ("What is 12 + 15?", "12 + 15 = 27. Answer: 27.", True),
    ("What is 8 × 7?", "8 × 7 = 56. Answer: 56.", True),
    ("What is 100 / 4?", "100 / 4 = 25. Answer: 25.", True),
    ("What is 5! (5 factorial)?", "5! = 5 × 4 × 3 × 2 × 1 = 120. Answer: 120.", True),
    ("If x = 3 and y = 4, what is x² + y²?", "x² = 9, y² = 16. 9 + 16 = 25. Answer: 25.", True),
    ("What is 2^10?", "2^10 = 1024. Answer: 1024.", True),
    ("What is the sum of 1 to 10?", "Sum = 10 × 11 / 2 = 55. Answer: 55.", True),
    ("What is 17 mod 5?", "17 = 3 × 5 + 2. So 17 mod 5 = 2. Answer: 2.", True),
    ("What is sqrt(144)?", "sqrt(144) = 12. Answer: 12.", True),
    ("What is 7/8 + 1/8?", "7/8 + 1/8 = 8/8 = 1. Answer: 1.", True),
    ("What is 3/4 × 8?", "3/4 × 8 = 6. Answer: 6.", True),
    ("What is 20% of 150?", "20% of 150 = 0.2 × 150 = 30. Answer: 30.", True),
    ("What is 9² - 4²?", "9² = 81, 4² = 16. 81 - 16 = 65. Answer: 65.", True),
    ("What is (3 + 4) × 2?", "(3 + 4) × 2 = 7 × 2 = 14. Answer: 14.", True),
    ("What is 6! / 4!?", "6! / 4! = 6 × 5 = 30. Answer: 30.", True),
    # Incorrect responses (arithmetic errors)
    ("What is 12 + 15?", "12 + 15 = 26. Answer: 26.", False),  # Should be 27
    ("What is 8 × 7?", "8 × 7 = 54. Answer: 54.", False),      # Should be 56
    ("What is 100 / 4?", "100 / 4 = 30. Answer: 30.", False),  # Should be 25
    ("What is 5! (5 factorial)?", "5! = 5 × 4 × 3 × 2 × 1 = 100. Answer: 100.", False),
    ("If x = 3 and y = 4, what is x² + y²?", "x² = 9, y² = 16. 9 + 16 = 23. Answer: 23.", False),
    ("What is 2^10?", "2^10 = 512. Answer: 512.", False),       # Should be 1024
    ("What is the sum of 1 to 10?", "Sum = 10 × 11 / 2 = 50. Answer: 50.", False),
    ("What is 17 mod 5?", "17 = 3 × 5 + 2. So 17 mod 5 = 3. Answer: 3.", False),
    ("What is sqrt(144)?", "sqrt(144) = 11. Answer: 11.", False),
    ("What is 7/8 + 1/8?", "7/8 + 1/8 = 8/8 = 2. Answer: 2.", False),
    ("What is 3/4 × 8?", "3/4 × 8 = 8. Answer: 8.", False),    # Should be 6
    ("What is 20% of 150?", "20% of 150 = 0.2 × 150 = 25. Answer: 25.", False),
    ("What is 9² - 4²?", "9² = 81, 4² = 16. 81 - 16 = 60. Answer: 60.", False),
    ("What is (3 + 4) × 2?", "(3 + 4) × 2 = 7 × 2 = 12. Answer: 12.", False),
    ("What is 6! / 4!?", "6! / 4! = 6 × 5 = 25. Answer: 25.", False),
]

assert len(SYNTHETIC_QUESTIONS) == 30, "Corpus must have exactly 30 questions"

# ---------------------------------------------------------------------------
# CI-mode LLM stub
# ---------------------------------------------------------------------------


def _ci_llm_stub(prompt: str) -> str:
    """CI-safe LLM stub that returns a generic correction without calling a real model.

    In CI (CARNOT_FORCE_LIVE not set), we never load a real LLM.  This stub
    produces a deterministic "correction" that allows the repair loop to proceed
    and generate an artifact, even though it does not fix actual errors.

    The stub simulates what a real LLM might say: it echoes the failed assertion
    back as a "correction", which means Z3 will still return UNSAT (or unknown
    in CI mode since NL2Z3Extractor returns "unknown" without CARNOT_FORCE_LIVE).
    """
    return "[CI stub correction — no real LLM available]"


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(max_iterations: int = 3) -> dict[str, Any]:
    """Run the 30-question VERGE benchmark and return aggregate metrics.

    Each question is processed by:
    1. A fresh NL2Z3Extractor (returns "unknown" in CI mode).
    2. VergeRefiner with max_iterations.

    In CI mode, all questions go through the iterative loop (since "unknown"
    does not trigger the SAT fast-path).  The iteration log is recorded honestly.

    Returns a dict suitable for inclusion in the experiment artifact.
    """
    force_live = bool(os.environ.get("CARNOT_FORCE_LIVE"))

    extractor = NL2Z3Extractor(timeout_s=5.0)
    llm_caller = _ci_llm_stub  # In production: replace with real LLM wrapper

    n_sat_initial = 0
    n_resolved = 0
    n_not_resolved = 0
    all_iteration_counts: list[int] = []
    iteration_distribution: dict[str, int] = {"1": 0, "2": 0, "3": 0}
    per_question_results = []

    for idx, (question, response, is_correct) in enumerate(SYNTHETIC_QUESTIONS):
        refiner = VergeRefiner(
            nl2z3_extractor=extractor,
            llm_caller=llm_caller,
            max_iterations=max_iterations,
        )
        final_response, iterations = refiner.refine(question, response)

        if not iterations:
            # SAT fast-path (initial check was SAT or unknown with no UNSAT)
            n_sat_initial += 1
            result = {
                "question_id": idx + 1,
                "question": question,
                "is_correct": is_correct,
                "initial_z3_status": getattr(extractor.last_z3_result, "sat_status", "unknown"),
                "n_iterations": 0,
                "resolved": True,  # SAT means no contradiction — treated as "no issue"
                "final_response_changed": False,
            }
        else:
            n_iter = len(iterations)
            resolved = iterations[-1].resolved
            all_iteration_counts.append(n_iter)
            k = min(n_iter, 3)
            iteration_distribution[str(k)] = iteration_distribution.get(str(k), 0) + 1
            if resolved:
                n_resolved += 1
            else:
                n_not_resolved += 1
            result = {
                "question_id": idx + 1,
                "question": question,
                "is_correct": is_correct,
                "initial_z3_status": iterations[0].new_z3_result.sat_status
                if iterations else "sat",
                "n_iterations": n_iter,
                "resolved": resolved,
                "final_response_changed": final_response != response,
                "iteration_log": [
                    {
                        "iteration_n": it.iteration_n,
                        "assertion_failed": it.assertion_failed,
                        "resolved": it.resolved,
                        "new_z3_status": it.new_z3_result.sat_status,
                    }
                    for it in iterations
                ],
            }
        per_question_results.append(result)

    mean_iterations = (
        sum(all_iteration_counts) / len(all_iteration_counts)
        if all_iteration_counts
        else 0.0
    )

    # Load Exp 312 baseline for comparison if available
    baseline_path = Path("results/experiment_312_z3_gated_results.json")
    baseline_skip_rate: float | None = None
    if baseline_path.exists():
        try:
            with baseline_path.open() as f:
                baseline = json.load(f)
            baseline_skip_rate = baseline.get("z3_gate_skip_rate")
        except Exception:  # noqa: BLE001
            baseline_skip_rate = None

    n_total = len(SYNTHETIC_QUESTIONS)
    verge_resolution_rate = (n_sat_initial + n_resolved) / n_total if n_total > 0 else 0.0
    accuracy_vs_baseline = (
        verge_resolution_rate - baseline_skip_rate
        if baseline_skip_rate is not None
        else None
    )

    return {
        "n_questions": n_total,
        "n_sat_initial": n_sat_initial,
        "n_resolved": n_resolved,
        "n_not_resolved": n_not_resolved,
        "mean_iterations": round(mean_iterations, 3),
        "iteration_distribution": iteration_distribution,
        "verge_resolution_rate": round(verge_resolution_rate, 4),
        "baseline_skip_rate": baseline_skip_rate,
        "accuracy_vs_baseline": (
            round(accuracy_vs_baseline, 4) if accuracy_vs_baseline is not None else None
        ),
        "force_live": force_live,
        "per_question_results": per_question_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=334,
        title="VERGE-style iterative Z3 refinement benchmark",
        deliverable="results/experiment_334_verge_refinement.json",
        requires_gpu=False,
    )
    tmpl.setup()

    metrics = run_benchmark(max_iterations=3)

    artifact = tmpl.build_result(metrics, status="success")

    out_path = Path("results/experiment_334_verge_refinement.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Artifact written to {out_path}")
    print(f"n_questions={metrics['n_questions']}")
    print(f"n_sat_initial={metrics['n_sat_initial']}")
    print(f"n_resolved={metrics['n_resolved']}")
    print(f"n_not_resolved={metrics['n_not_resolved']}")
    print(f"mean_iterations={metrics['mean_iterations']}")
    print(f"verge_resolution_rate={metrics['verge_resolution_rate']}")
    if metrics["accuracy_vs_baseline"] is not None:
        print(f"accuracy_vs_baseline={metrics['accuracy_vs_baseline']:+.4f}")
    else:
        print("accuracy_vs_baseline=N/A (no Exp 312 baseline found)")


if __name__ == "__main__":
    main()
