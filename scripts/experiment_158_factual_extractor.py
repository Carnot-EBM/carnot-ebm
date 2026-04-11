#!/usr/bin/env python3
"""Experiment 158 — FactualExtractor: TruthfulQA Coverage + Accuracy Benchmark.

**Researcher summary:**
    Benchmarks FactualExtractor on 50 simulated TruthfulQA-style factual
    questions. Measures two metrics:

    1. **Constraint coverage**: fraction of questions for which FactualExtractor
       returns ≥1 ConstraintResult. Target: >30% of questions (Goal #3
       threshold). This metric works without network access (claims are
       extracted from text locally).

    2. **Constraint accuracy**: of constraints returned, what fraction are
       correctly satisfied=True for best answers and satisfied=False (or
       absent) for wrong answers? Requires Wikidata network access; degrades
       gracefully if unavailable.

**Methodology:**
    We use 50 TruthfulQA-style Q&A pairs with known-correct and known-wrong
    answers. For each question:
    - Run FactualExtractor on the "best answer" (should produce verified constraints)
    - Run FactualExtractor on a "wrong answer" (should produce contradicted constraints)

    Wikidata is queried for each claim triple. If network is unavailable, the
    benchmark reports coverage only (extraction part) and notes that accuracy
    measurement requires network access.

    This mirrors the methodology of Exp 157 (SpilledEnergyExtractor) but uses
    a KB-backed extractor rather than logit signals.

Run:
    JAX_PLATFORMS=cpu python scripts/experiment_158_factual_extractor.py

Output:
    Console report + results/experiment_158_results.json

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on the path.
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

from carnot.pipeline.factual_extractor import (
    _QID_CACHE,
    _CLAIM_CACHE,
    FactualExtractor,
    extract_claims,
)


# ---------------------------------------------------------------------------
# TruthfulQA-style question bank (50 questions with best and wrong answers)
# ---------------------------------------------------------------------------
# Each entry: (question, best_answer_sentence, wrong_answer_sentence)
# These are carefully constructed so that:
# - best_answer contains a verifiable true claim pattern
# - wrong_answer contains the same pattern with a false object
# ---------------------------------------------------------------------------

QUESTION_BANK: list[tuple[str, str, str]] = [
    # Geography — capitals
    (
        "What is the capital of France?",
        "Paris is the capital of France.",
        "London is the capital of France.",
    ),
    (
        "What is the capital of Germany?",
        "The capital of Germany is Berlin.",
        "The capital of Germany is Munich.",
    ),
    (
        "What is the capital of Japan?",
        "Tokyo is the capital of Japan.",
        "Osaka is the capital of Japan.",
    ),
    (
        "What is the capital of Australia?",
        "The capital of Australia is Canberra.",
        "The capital of Australia is Sydney.",
    ),
    (
        "What is the capital of Brazil?",
        "Brasilia is the capital of Brazil.",
        "Rio de Janeiro is the capital of Brazil.",
    ),
    (
        "What is the capital of Canada?",
        "Ottawa is the capital of Canada.",
        "Toronto is the capital of Canada.",
    ),
    (
        "What is the capital of Italy?",
        "Rome is the capital of Italy.",
        "Milan is the capital of Italy.",
    ),
    (
        "What is the capital of Spain?",
        "Madrid is the capital of Spain.",
        "Barcelona is the capital of Spain.",
    ),
    (
        "What is the capital of China?",
        "Beijing is the capital of China.",
        "Shanghai is the capital of China.",
    ),
    (
        "What is the capital of Russia?",
        "Moscow is the capital of Russia.",
        "Saint Petersburg is the capital of Russia.",
    ),
    # Geography — locations
    (
        "Where is the Eiffel Tower located?",
        "The Eiffel Tower is located in Paris.",
        "The Eiffel Tower is located in London.",
    ),
    (
        "Where is the Great Wall located?",
        "The Great Wall is located in China.",
        "The Great Wall is located in Japan.",
    ),
    (
        "Which country is Rome in?",
        "Rome is in Italy.",
        "Rome is in Spain.",
    ),
    (
        "Which country is Vienna in?",
        "Vienna is in Austria.",
        "Vienna is in Germany.",
    ),
    (
        "Which country is Amsterdam in?",
        "Amsterdam is in the Netherlands.",
        "Amsterdam is in Belgium.",
    ),
    # Birthplaces
    (
        "Where was Albert Einstein born?",
        "Albert Einstein was born in Ulm.",
        "Albert Einstein was born in Munich.",
    ),
    (
        "Where was Napoleon Bonaparte born?",
        "Napoleon Bonaparte was born in Corsica.",
        "Napoleon Bonaparte was born in Paris.",
    ),
    (
        "Where was Mozart born?",
        "Mozart was born in Salzburg.",
        "Mozart was born in Vienna.",
    ),
    # Currencies
    (
        "What is the currency of Japan?",
        "The currency of Japan is the yen.",
        "The currency of Japan is the dollar.",
    ),
    (
        "What is the currency of the United Kingdom?",
        "The currency of the United Kingdom is the pound.",
        "The currency of the United Kingdom is the euro.",
    ),
    (
        "What is the currency of Brazil?",
        "The currency of Brazil is the real.",
        "The currency of Brazil is the dollar.",
    ),
    (
        "What is the currency of India?",
        "The currency of India is the rupee.",
        "The currency of India is the dollar.",
    ),
    (
        "What is the currency of China?",
        "The currency of China is the yuan.",
        "The currency of China is the dollar.",
    ),
    # Official languages
    (
        "What is the official language of Brazil?",
        "The official language of Brazil is Portuguese.",
        "The official language of Brazil is Spanish.",
    ),
    (
        "What is the official language of Egypt?",
        "The official language of Egypt is Arabic.",
        "The official language of Egypt is English.",
    ),
    (
        "What is the official language of Mexico?",
        "The official language of Mexico is Spanish.",
        "The official language of Mexico is English.",
    ),
    # More capitals
    (
        "What is the capital of Egypt?",
        "The capital of Egypt is Cairo.",
        "The capital of Egypt is Alexandria.",
    ),
    (
        "What is the capital of Argentina?",
        "The capital of Argentina is Buenos Aires.",
        "The capital of Argentina is Córdoba.",
    ),
    (
        "What is the capital of South Korea?",
        "The capital of South Korea is Seoul.",
        "The capital of South Korea is Busan.",
    ),
    (
        "What is the capital of Turkey?",
        "The capital of Turkey is Ankara.",
        "The capital of Turkey is Istanbul.",
    ),
    (
        "What is the capital of India?",
        "The capital of India is New Delhi.",
        "The capital of India is Mumbai.",
    ),
    (
        "What is the capital of Mexico?",
        "The capital of Mexico is Mexico City.",
        "The capital of Mexico is Guadalajara.",
    ),
    (
        "What is the capital of the Netherlands?",
        "The capital of the Netherlands is Amsterdam.",
        "The capital of the Netherlands is Rotterdam.",
    ),
    (
        "What is the capital of Poland?",
        "The capital of Poland is Warsaw.",
        "The capital of Poland is Kraków.",
    ),
    (
        "What is the capital of Sweden?",
        "The capital of Sweden is Stockholm.",
        "The capital of Sweden is Gothenburg.",
    ),
    (
        "What is the capital of Norway?",
        "The capital of Norway is Oslo.",
        "The capital of Norway is Bergen.",
    ),
    (
        "What is the capital of Denmark?",
        "The capital of Denmark is Copenhagen.",
        "The capital of Denmark is Aarhus.",
    ),
    (
        "What is the capital of Finland?",
        "The capital of Finland is Helsinki.",
        "The capital of Finland is Tampere.",
    ),
    (
        "What is the capital of Portugal?",
        "The capital of Portugal is Lisbon.",
        "The capital of Portugal is Porto.",
    ),
    (
        "What is the capital of Greece?",
        "The capital of Greece is Athens.",
        "The capital of Greece is Thessaloniki.",
    ),
    (
        "What is the capital of Switzerland?",
        "The capital of Switzerland is Bern.",
        "The capital of Switzerland is Zurich.",
    ),
    (
        "What is the capital of Austria?",
        "The capital of Austria is Vienna.",
        "The capital of Austria is Salzburg.",
    ),
    (
        "What is the capital of Belgium?",
        "The capital of Belgium is Brussels.",
        "The capital of Belgium is Antwerp.",
    ),
    (
        "What is the capital of Hungary?",
        "The capital of Hungary is Budapest.",
        "The capital of Hungary is Debrecen.",
    ),
    (
        "What is the capital of Czech Republic?",
        "The capital of Czech Republic is Prague.",
        "The capital of Czech Republic is Brno.",
    ),
    (
        "What is the capital of Romania?",
        "The capital of Romania is Bucharest.",
        "The capital of Romania is Cluj.",
    ),
    (
        "What is the capital of Ukraine?",
        "The capital of Ukraine is Kyiv.",
        "The capital of Ukraine is Kharkiv.",
    ),
    (
        "What is the capital of Indonesia?",
        "The capital of Indonesia is Jakarta.",
        "The capital of Indonesia is Surabaya.",
    ),
    (
        "What is the capital of Thailand?",
        "The capital of Thailand is Bangkok.",
        "The capital of Thailand is Chiang Mai.",
    ),
    (
        "What is the capital of Saudi Arabia?",
        "The capital of Saudi Arabia is Riyadh.",
        "The capital of Saudi Arabia is Jeddah.",
    ),
]

# Ensure exactly 50 questions
assert len(QUESTION_BANK) == 50, f"Expected 50 questions, got {len(QUESTION_BANK)}"

TARGET_COVERAGE_PCT = 30.0  # Goal #3 threshold
NETWORK_TIMEOUT = 5.0


def check_network_available() -> bool:
    """Quick check if Wikidata SPARQL endpoint is reachable (≤5s timeout).

    Returns True if a simple HEAD request to Wikidata succeeds.
    Returns False on any error — used to decide whether to attempt KB lookups.
    """
    try:
        import requests

        requests.head("https://query.wikidata.org/", timeout=3.0)
        return True
    except Exception:
        return False


def run_benchmark(use_network: bool = True) -> dict:
    """Run the Exp 158 benchmark and return results dictionary.

    Args:
        use_network: If True, attempt Wikidata lookups (requires internet).
            If False, measure extraction-only coverage (no network calls).

    Returns:
        Results dictionary with coverage, accuracy, and per-question data.
    """
    # Use a short timeout in offline mode to fail fast without blocking.
    timeout = NETWORK_TIMEOUT if use_network else 0.1
    ext = FactualExtractor(timeout=timeout)

    per_question: list[dict] = []
    n_covered = 0          # questions with ≥1 constraint from best answer
    total_constraints = 0  # total constraints from best answers
    total_correct = 0      # constraints with satisfied=True (from best answers)
    total_false_constraints = 0   # constraints from wrong answers
    total_false_contradicted = 0  # wrong-answer constraints with satisfied=False

    print(f"Exp 158 — FactualExtractor: TruthfulQA Coverage + Accuracy")
    print(f"  N={len(QUESTION_BANK)} questions")
    print(f"  Network lookups: {'enabled' if use_network else 'disabled (offline mode)'}")
    print(f"  Timeout: {timeout}s")
    print()

    for i, (question, best_answer, wrong_answer) in enumerate(QUESTION_BANK):
        t0 = time.monotonic()

        # --- Best answer constraints ---
        best_results = ext.extract(best_answer, domain="factual")
        n_best = len(best_results)
        n_best_verified = sum(
            1 for r in best_results if r.metadata.get("verified") is True
        )

        # --- Wrong answer constraints ---
        false_results = ext.extract(wrong_answer, domain="factual")
        n_false = len(false_results)
        n_false_contradicted = sum(
            1 for r in false_results if r.metadata.get("verified") is False
        )

        covered = n_best >= 1
        if covered:
            n_covered += 1
        total_constraints += n_best
        total_correct += n_best_verified
        total_false_constraints += n_false
        total_false_contradicted += n_false_contradicted

        elapsed = time.monotonic() - t0

        per_question.append(
            {
                "question": question,
                "best_answer": best_answer,
                "covered": covered,
                "n_best_constraints": n_best,
                "n_best_verified": n_best_verified,
                "n_false_constraints": n_false,
                "n_false_contradicted": n_false_contradicted,
                "elapsed_s": round(elapsed, 3),
            }
        )

        if (i + 1) % 10 == 0:
            print(
                f"  Progress: {i + 1}/{len(QUESTION_BANK)}"
                f" — coverage so far: {n_covered}/{i + 1}"
                f" ({100.0 * n_covered / (i + 1):.1f}%)"
            )

    n_questions = len(QUESTION_BANK)
    coverage_pct = 100.0 * n_covered / n_questions
    accuracy_pct = (
        100.0 * total_correct / total_constraints if total_constraints > 0 else 0.0
    )
    coverage_target_met = coverage_pct >= TARGET_COVERAGE_PCT

    # Claims breakdown for covered questions
    n_extractable = sum(
        1 for r in per_question if _has_any_claim(r["best_answer"])
    )

    results_dict = {
        "experiment": "Exp 158 — FactualExtractor (Wikidata SPARQL)",
        "n_questions": n_questions,
        "n_covered": n_covered,
        "coverage_pct": coverage_pct,
        "target_coverage_pct": TARGET_COVERAGE_PCT,
        "coverage_target_met": coverage_target_met,
        "total_constraints": total_constraints,
        "total_correct": total_correct,
        "accuracy_pct": accuracy_pct,
        "total_false_constraints": total_false_constraints,
        "total_false_contradicted": total_false_contradicted,
        "n_extractable_questions": n_extractable,
        "network_used": use_network,
        "qid_cache_size": len(_QID_CACHE),
        "claim_cache_size": len(_CLAIM_CACHE),
        "per_question": per_question,
    }

    # Print summary
    print()
    print(f"Results:")
    print(
        f"  Coverage:       {n_covered}/{n_questions} = {coverage_pct:.1f}%"
        f"  (target >{TARGET_COVERAGE_PCT:.0f}%,"
        f" {'✓ MET' if coverage_target_met else '✗ NOT MET'})"
    )
    print(f"  Claims extracted from best answers: {total_constraints}")
    print(
        f"  Verified correct:                   {total_correct}"
        f"  ({accuracy_pct:.1f}%)"
    )
    print(f"  Wikidata QID cache size:  {len(_QID_CACHE)}")
    print(f"  Wikidata claim cache size:{len(_CLAIM_CACHE)}")
    print()

    if not use_network:
        print(
            "  NOTE: Running in offline mode. Accuracy metrics require Wikidata "
            "network access. Coverage measures extraction only."
        )

    # Print sample covered questions
    covered_qs = [r for r in per_question if r["covered"]][:5]
    if covered_qs:
        print(f"  Sample covered questions (first 5):")
        for q in covered_qs:
            print(
                f"    [{q['n_best_constraints']} constraint(s)] {q['question']}"
                f" ({q['elapsed_s']:.2f}s)"
            )

    return results_dict


def _has_any_claim(text: str) -> bool:
    """Return True if text contains at least one extractable claim triple."""
    return len(extract_claims(text)) > 0


def save_results(results_dict: dict) -> Path:
    """Save benchmark results to results/experiment_158_results.json."""
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "experiment_158_results.json"
    with out_path.open("w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return out_path


if __name__ == "__main__":
    t_start = time.monotonic()

    # Check network availability first (quick test, <3s).
    print("Checking Wikidata network availability... ", end="", flush=True)
    network_ok = check_network_available()
    print("OK" if network_ok else "OFFLINE (graceful degradation mode)")
    print()

    results = run_benchmark(use_network=network_ok)
    save_results(results)

    elapsed_total = time.monotonic() - t_start
    print(f"\nTotal elapsed: {elapsed_total:.1f}s")

    # Exit status
    if results["coverage_target_met"]:
        print(
            f"\nTarget coverage {TARGET_COVERAGE_PCT:.0f}% met:"
            f" {results['coverage_pct']:.1f}%"
        )
        sys.exit(0)
    else:
        print(
            f"\nWARNING: Coverage {results['coverage_pct']:.1f}%"
            f" < target {TARGET_COVERAGE_PCT:.0f}%"
        )
        sys.exit(1)
