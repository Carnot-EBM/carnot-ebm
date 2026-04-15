"""Experiment 336: CoTCircuitVerifier benchmark.

**Researcher summary:**
    Measures the TP and FP rates of CoTCircuitVerifier on a synthetic corpus of
    20 multi-step math responses — 10 with known broken links (incorrect
    reasoning chains) and 10 consistent chains (correct reasoning).

    Compares the FP rate against the ArithmeticExtractor and NL2Z3Extractor
    baselines from Exp 311 to show whether CRV adds complementary signal or
    overlaps with existing extractors.

**What we measure:**
    - TP rate: broken links flagged on responses with real chain errors
    - FP rate: broken links flagged on responses that are internally consistent
    - Comparison table against Exp 311 extractor baselines

Usage:
    JAX_PLATFORMS=cpu python scripts/experiment_336_cot_circuit_benchmark.py

Outputs:
    results/experiment_336_cot_circuit_results.json

Spec: REQ-EXTRACT-015, REQ-EXTRACT-016, SCENARIO-EXTRACT-031–035
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path so we can import carnot and scripts.
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

from carnot.pipeline.cot_circuit_verifier import CoTCircuitVerifier  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------

# Each entry: {"response": str, "is_correct": bool, "description": str}
# is_correct=True  → no chain errors; CRV should return 0 broken links (no FP)
# is_correct=False → broken chain; CRV should return ≥1 broken links (TP)

CORPUS: list[dict] = [
    # -----------------------------------------------------------------------
    # Correct responses (is_correct=True) — CRV should NOT flag these
    # -----------------------------------------------------------------------
    {
        "response": (
            "Step 1: Sarah starts with 24 apples.\n"
            "Step 2: She gives away 8. She now has 16.\n"
            "Step 3: She buys 5 more. Total: 21.\n"
            "Step 4: The answer is 21."
        ),
        "is_correct": True,
        "description": "apple subtraction/addition — consistent chain",
    },
    {
        "response": (
            "Step 1: The train travels at 60 mph.\n"
            "Step 2: The journey is 3 hours. Distance = 60 * 3 = 180 miles.\n"
            "Step 3: The answer is 180."
        ),
        "is_correct": True,
        "description": "distance = rate * time — consistent",
    },
    {
        "response": (
            "First, compute 5 * 6 = 30.\n"
            "Then, add 15 to get 45.\n"
            "Finally, the answer is 45."
        ),
        "is_correct": True,
        "description": "discourse markers — consistent arithmetic",
    },
    {
        "response": (
            "1. The factory produces 200 units per day.\n"
            "2. Over 5 days it produces 1000 units.\n"
            "3. The answer is 1000."
        ),
        "is_correct": True,
        "description": "numbered lines — consistent multiplication",
    },
    {
        "response": "The answer is 42.",
        "is_correct": True,
        "description": "single-step — no chain to break",
    },
    {
        "response": (
            "Step 1: There are 7 days in a week.\n"
            "Step 2: In 4 weeks there are 28 days.\n"
            "Step 3: In 52 weeks there are 364 days.\n"
            "Step 4: The answer is 364."
        ),
        "is_correct": True,
        "description": "week/day conversion — consistent chain",
    },
    {
        "response": (
            "First, note that 100 / 4 = 25.\n"
            "Then, 25 * 3 = 75.\n"
            "Finally, the answer is 75."
        ),
        "is_correct": True,
        "description": "fraction of a whole — consistent",
    },
    {
        "response": (
            "Step 1: Read the problem: 48 cookies divided equally among 6 children.\n"
            "Step 2: 48 / 6 = 8 cookies per child.\n"
            "Step 3: The answer is 8."
        ),
        "is_correct": True,
        "description": "division — consistent",
    },
    {
        "response": (
            "Step 1: Start with 50.\n"
            "Step 2: Subtract 13. Result: 37.\n"
            "Step 3: Add 22. Result: 59.\n"
            "Step 4: The final answer is 59."
        ),
        "is_correct": True,
        "description": "multi-op chain — consistent",
    },
    {
        "response": (
            "First, compute 9 squared = 81.\n"
            "Next, take the square root of 81 = 9.\n"
            "Finally, the answer is 9."
        ),
        "is_correct": True,
        "description": "square/sqrt round-trip — consistent",
    },
    # -----------------------------------------------------------------------
    # Incorrect responses (is_correct=False) — CRV should flag broken links
    # -----------------------------------------------------------------------
    {
        "response": (
            "Step 1: John earns $15 per hour.\n"
            "Step 2: He works 8 hours. Total: $110.\n"
            # Correct would be 15*8=120; step 2 says 110 — arithmetic error
            "Step 3: After tax (10%), he keeps $99.\n"
            # If step 2 produced 110, then 110*0.9=99 — internal consistency broken
            # because 110 was itself wrong, but step 3 uses 110*0.9=99 — consistent
            # with step 2. The error is in step 2 arithmetic, not carryover.
            "Step 4: The answer is $99."
        ),
        "is_correct": False,
        "description": "arithmetic error in step 2 — wrong product",
    },
    {
        "response": (
            "Step 1: The box has 30 red and 20 blue balls. Total = 50.\n"
            "Step 2: From step 1, we have 60 balls.\n"
            # step 1 output is 50, but step 2 reads it as 60 → broken link
            "Step 3: Removing 10, we have 50 balls."
        ),
        "is_correct": False,
        "description": "wrong carryover from step 1 (50→60) — broken link",
    },
    {
        "response": (
            "First, Alice has 45 marbles.\n"
            "Then, she gives away 15. She now has 25.\n"
            # 45-15=30, not 25 — arithmetic error within step
            "Finally, the answer is 25."
        ),
        "is_correct": False,
        "description": "subtraction error — 45-15≠25",
    },
    {
        "response": (
            "Step 1: Temperature in Celsius = 100.\n"
            "Step 2: From step 1, C = 100. F = (100 * 9/5) + 32 = 212.\n"
            "Step 3: From step 2, the temperature is 200 degrees F.\n"
            # step 2 output is 212, step 3 reads it as 200 → broken link
            "Step 4: The answer is 200."
        ),
        "is_correct": False,
        "description": "wrong carryover in step 3 (212→200) — broken link",
    },
    {
        "response": (
            "1. Buy 3 packs of 12 eggs each.\n"
            "2. Total eggs = 3 * 12 = 36.\n"
            "3. From step 2, we had 32 eggs. After using 10, we have 22 left.\n"
            # step 2 output is 36, step 3 reads it as 32 → broken link
            "4. The answer is 22."
        ),
        "is_correct": False,
        "description": "wrong carryover in step 3 (36→32) — broken link",
    },
    {
        "response": (
            "Step 1: Compute 8 * 7 = 56.\n"
            "Step 2: From step 1 (54), add 6 = 60.\n"
            # step 1 output is 56; step 2 reads it as 54 → broken link
            "Step 3: The answer is 60."
        ),
        "is_correct": False,
        "description": "wrong carryover in step 2 (56→54) — broken link",
    },
    {
        "response": (
            "First, note that 100 students, 60% pass = 60 students.\n"
            "Then, from the 55 passing students, 10 got A grades.\n"
            # First step says 60; second reads it as 55 → broken link
            "Finally, the fraction is 10/55 ≈ 0.18."
        ),
        "is_correct": False,
        "description": "wrong carryover (60→55) — broken link",
    },
    {
        "response": (
            "Step 1: Start with 200 km distance at 50 km/h → time = 4 hours.\n"
            "Step 2: From step 1 (3 hours), cost at $20/hr = $60.\n"
            # step 1 output is 4 hours; step 2 reads it as 3 → broken link
            "Step 3: The answer is $60."
        ),
        "is_correct": False,
        "description": "wrong carryover in step 2 (4→3 hours) — broken link",
    },
    {
        "response": (
            "1. First batch: 24 cookies.\n"
            "2. Second batch: 18 cookies. Total so far: 42.\n"
            "3. From step 2 we had 40 cookies. Add a third batch of 12 = 52.\n"
            # step 2 output is 42; step 3 reads it as 40 → broken link
            "4. The answer is 52."
        ),
        "is_correct": False,
        "description": "wrong carryover in step 3 (42→40) — broken link",
    },
    {
        "response": (
            "Step 1: A tank holds 500 liters.\n"
            "Step 2: It is 40% full: 0.4 * 500 = 200 liters.\n"
            "Step 3: From step 2 (180 liters), drain 50 → 130 liters remain.\n"
            # step 2 output is 200; step 3 reads it as 180 → broken link
            "Step 4: The answer is 130."
        ),
        "is_correct": False,
        "description": "wrong carryover in step 3 (200→180) — broken link",
    },
]


# ---------------------------------------------------------------------------
# Main benchmark logic
# ---------------------------------------------------------------------------


def run_benchmark(
    corpus: list[dict],
    tolerance: float = 0.01,
) -> dict:
    """Run CoTCircuitVerifier against the corpus and compute TP/FP rates.

    **Detailed explanation for engineers:**
        For each corpus entry:
        - Run CoTCircuitVerifier.extract() to get a list of ConstraintResult.
        - A "flagged" result means ≥1 broken link was detected.
        - For incorrect responses: flagged = True Positive (TP).
        - For correct responses:   flagged = True Negative (TN) is good;
                                   flagged = False Positive (FP) is bad.

    Returns:
        Dict with per-case results, TP rate, FP rate, and comparison table.
    """
    verifier = CoTCircuitVerifier(tolerance=tolerance)

    correct_cases = [c for c in corpus if c["is_correct"]]
    incorrect_cases = [c for c in corpus if not c["is_correct"]]

    per_case_results = []

    tp_count = 0
    fp_count = 0

    for entry in corpus:
        violations = verifier.extract("", entry["response"])
        flagged = len(violations) > 0

        if entry["is_correct"]:
            is_fp = flagged
            is_tp = False
            if is_fp:
                fp_count += 1
        else:
            is_tp = flagged
            is_fp = False
            if is_tp:
                tp_count += 1

        # Capture broken-link details for the artifact
        circuit = verifier.last_circuit
        broken_links = []
        if circuit:
            broken_links = [
                {
                    "downstream_step": link[0],
                    "upstream_step": link[1],
                    "expected": link[2],
                    "actual": link[3],
                }
                for link in circuit.broken_links
            ]

        per_case_results.append(
            {
                "description": entry["description"],
                "is_correct": entry["is_correct"],
                "flagged": flagged,
                "is_tp": is_tp,
                "is_fp": is_fp,
                "n_broken_links": len(broken_links),
                "broken_links": broken_links,
            }
        )

    n_correct = len(correct_cases)
    n_incorrect = len(incorrect_cases)

    tp_rate = tp_count / n_incorrect if n_incorrect > 0 else 0.0
    fp_rate = fp_count / n_correct if n_correct > 0 else 0.0

    # Load Exp 311 baseline for comparison (if available)
    exp311_path = PROJECT_ROOT / "results" / "experiment_311_extractor_benchmark.json"
    exp311_comparison: dict = {}
    if exp311_path.exists():
        try:
            with open(exp311_path) as f:
                exp311 = json.load(f)
            exp311_comparison = {
                "source": "experiment_311_extractor_benchmark.json",
                "extractors": exp311.get("extractors", []),
                "winner": exp311.get("winner", "unknown"),
                "note": "Exp 311 used different corpus; FP rates not directly comparable.",
            }
        except Exception as exc:  # noqa: BLE001
            exp311_comparison = {"error": str(exc)}

    return {
        "corpus_size": len(corpus),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "tp_count": tp_count,
        "fp_count": fp_count,
        "tp_rate": round(tp_rate, 4),
        "fp_rate": round(fp_rate, 4),
        "tolerance": tolerance,
        "per_case_results": per_case_results,
        "exp311_comparison": exp311_comparison,
        "interpretation": (
            "TP rate = fraction of incorrect responses where ≥1 broken link was detected. "
            "FP rate = fraction of correct responses incorrectly flagged. "
            "CRV is complementary to Z3: Z3 checks arithmetic correctness, "
            "CRV checks value-carryover consistency across reasoning steps."
        ),
    }


def main() -> None:
    """Entry point for Exp 336."""
    tmpl = ExperimentTemplate(
        336,
        "CoT Circuit Verifier benchmark — TP/FP rates on synthetic reasoning corpus",
        "results/experiment_336_cot_circuit_results.json",
        requires_gpu=False,
    )
    tmpl.setup()

    benchmark_results = run_benchmark(CORPUS, tolerance=0.01)

    artifact = tmpl.build_result(
        benchmark_results,
        status="success",
    )

    output_path = PROJECT_ROOT / "results" / "experiment_336_cot_circuit_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Exp 336 complete.")
    print(f"  TP rate: {benchmark_results['tp_rate']:.1%}  ({benchmark_results['tp_count']}/{benchmark_results['n_incorrect']} incorrect responses flagged)")
    print(f"  FP rate: {benchmark_results['fp_rate']:.1%}  ({benchmark_results['fp_count']}/{benchmark_results['n_correct']} correct responses incorrectly flagged)")
    print(f"  Results: {output_path}")


if __name__ == "__main__":
    main()
