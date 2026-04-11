#!/usr/bin/env python3
"""Experiment 172 — Global Consistency Checking Benchmark.

**Researcher summary:**
    Evaluates GlobalConsistencyChecker (Exp 172) on 20 synthetic multi-step
    reasoning chains: 10 chains that are locally consistent but globally
    inconsistent (each step passes ConstraintStateMachine's per-step check,
    but cross-step contradictions exist), and 10 fully consistent chains
    (all steps agree on all entities and claims).

    The core hypothesis from arxiv 2601.13600: local per-step verification
    misses global inconsistencies that pairwise comparison catches. This
    benchmark quantifies the detection gap.

    Results:
        Local-only detection rate on inconsistent chains:  expected 0%
            (local check sees verified=True for all steps)
        Global detection rate on inconsistent chains:      expected ~90-100%
            (text-level extraction catches numeric/arithmetic/factual contradictions)
        False positive rate on consistent chains:          expected 0%
            (no contradictions in text → checker returns consistent=True)

**Chain construction:**
    Each chain has 4 steps. The pipeline is mocked to always return
    verified=True (simulating local-pass behaviour). The output_text
    contains deliberate contradictions in the globally-inconsistent chains.

    Inconsistent chain types (3 contradiction types, 10 chains total):
        Type A — Numeric (4 chains):
            Step 1: "The widget costs $50."
            Steps 2-3: neutral content
            Step 4: "The widget costs $75."  ← contradicts step 1

        Type B — Arithmetic (3 chains):
            Step 1: "We computed 3 + 5 = 8."
            Steps 2-3: neutral content
            Step 4: "We know that 3 + 5 = 10 from earlier."  ← contradicts step 1

        Type C — Factual (3 chains):
            Step 1: "Paris is the capital of France."
            Steps 2-3: neutral content
            Step 4: "Berlin is the capital of France."  ← contradicts step 1

    Consistent chains (10):
        All 4 steps agree on entities/values, arithmetic, and factual claims.

**Key metrics:**
    For each chain:
    - local_detected: True iff ANY step has verification.verified == False
    - global_detected: True iff check_global_consistency().consistent == False

    Aggregate metrics:
    - local_detection_rate: fraction of inconsistent chains where local_detected
    - global_detection_rate: fraction of inconsistent chains where global_detected
    - false_positive_rate: fraction of consistent chains where global_detected

Run:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_172_global_consistency.py

Output:
    Prints per-chain results and summary statistics.
    Saves results to results/experiment_172_results.json.

Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from carnot.pipeline.consistency_checker import GlobalConsistencyChecker
from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.state_machine import ConstraintStateMachine
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Chain definitions
# ---------------------------------------------------------------------------


@dataclass
class ChainSpec:
    """Specification for one benchmark chain.

    **Detailed explanation for engineers:**
        Each chain has 4 steps. output_texts[i] is the text for step i.
        contradiction_type identifies which detection method should fire
        (or None for fully consistent chains).
        expected_consistent tells us whether the chain is expected to
        pass the global consistency check.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    chain_id: int
    chain_type: str  # "globally_inconsistent" | "fully_consistent"
    contradiction_type: str | None  # "numeric" | "arithmetic" | "factual" | None
    output_texts: list[str]
    expected_consistent: bool


def _build_numeric_chain(chain_id: int, wrong_value: int = 75) -> ChainSpec:
    """Build a chain where step 1 and step 4 contradict on numeric value.

    **Detailed explanation for engineers:**
        Both steps mention 'widget' with different costs. The pipeline sees
        verified=True for all steps (no local violation), but GlobalConsistency-
        Checker finds the numeric contradiction.

    Args:
        chain_id: Unique identifier for this chain in the benchmark.
        wrong_value: The contradicting value in step 4.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    return ChainSpec(
        chain_id=chain_id,
        chain_type="globally_inconsistent",
        contradiction_type="numeric",
        output_texts=[
            "The widget costs $50. This is the listed price in our catalogue.",
            "We need to verify the item specifications before proceeding.",
            "The specifications have been reviewed and are correct.",
            f"The widget costs ${wrong_value} based on the updated price list.",
        ],
        expected_consistent=False,
    )


def _build_arithmetic_chain(chain_id: int, wrong_result: int = 10) -> ChainSpec:
    """Build a chain where step 1 and step 4 contradict on arithmetic result.

    **Detailed explanation for engineers:**
        Step 1 claims "3 + 5 = 8" (correct). Step 4 claims "3 + 5 = {wrong}"
        (incorrect but passes local per-step verification since the pipeline
        is mocked). GlobalConsistencyChecker catches the arithmetic conflict.

    Args:
        chain_id: Unique identifier for this chain.
        wrong_result: The wrong arithmetic result asserted in step 4.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    return ChainSpec(
        chain_id=chain_id,
        chain_type="globally_inconsistent",
        contradiction_type="arithmetic",
        output_texts=[
            "We calculated 3 + 5 = 8 in the previous round.",
            "Moving forward with the intermediate analysis.",
            "The intermediate analysis supports our earlier findings.",
            f"Based on prior work, we know that 3 + 5 = {wrong_result}.",
        ],
        expected_consistent=False,
    )


def _build_factual_chain(chain_id: int) -> ChainSpec:
    """Build a chain where step 1 and step 4 contradict on a factual triple.

    **Detailed explanation for engineers:**
        Step 1 states "Paris is the capital of France." Step 4 states
        "Berlin is the capital of France." — a direct factual contradiction
        on (France, capital). GlobalConsistencyChecker uses extract_claims()
        (from Exp 158 FactualExtractor patterns) to detect this.

    Args:
        chain_id: Unique identifier for this chain.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    return ChainSpec(
        chain_id=chain_id,
        chain_type="globally_inconsistent",
        contradiction_type="factual",
        output_texts=[
            "Paris is the capital of France.",
            "Let us examine the European political landscape.",
            "The analysis considers multiple European countries.",
            "Berlin is the capital of France, per this document.",
        ],
        expected_consistent=False,
    )


def _build_consistent_chain(chain_id: int, chain_variant: int) -> ChainSpec:
    """Build a fully consistent 4-step chain (no cross-step contradictions).

    **Detailed explanation for engineers:**
        All 4 steps agree on all numeric values, arithmetic, and factual claims.
        This tests that GlobalConsistencyChecker does not raise false positives.
        chain_variant provides variety in the neutral text content so the 10
        consistent chains are not identical.

    Args:
        chain_id: Unique identifier for this chain.
        chain_variant: Integer 0-9 to diversify the neutral content.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    topic = [
        "shipping", "inventory", "pricing", "quality", "logistics",
        "supply", "demand", "production", "distribution", "returns",
    ][chain_variant % 10]
    return ChainSpec(
        chain_id=chain_id,
        chain_type="fully_consistent",
        contradiction_type=None,
        output_texts=[
            f"The {topic} cost is $42. We computed 2 + 3 = 5 for verification.",
            f"The {topic} process involves multiple steps, all reviewed.",
            f"Reviewing {topic}: cost confirmed at $42.",
            f"Final check: the {topic} cost remains $42. Also 2 + 3 = 5.",
        ],
        expected_consistent=True,
    )


def _build_all_chains() -> list[ChainSpec]:
    """Build all 20 benchmark chains.

    **Detailed explanation for engineers:**
        10 globally-inconsistent chains (4 numeric, 3 arithmetic, 3 factual) +
        10 fully-consistent chains. Total: 20 chains × 4 steps = 80 pipeline
        calls (all mocked, so fast and deterministic).

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    chains: list[ChainSpec] = []

    # 4 numeric chains with different wrong values
    for k, wrong_val in enumerate([75, 80, 60, 100]):
        chains.append(_build_numeric_chain(chain_id=k, wrong_value=wrong_val))

    # 3 arithmetic chains with different wrong results
    for k, wrong_res in enumerate([10, 9, 7], start=4):
        chains.append(_build_arithmetic_chain(chain_id=k, wrong_result=wrong_res))

    # 3 factual chains
    for k in range(3):
        chains.append(_build_factual_chain(chain_id=k + 7))

    # 10 consistent chains
    for k in range(10):
        chains.append(_build_consistent_chain(chain_id=k + 10, chain_variant=k))

    return chains


# ---------------------------------------------------------------------------
# Pipeline mock — always returns verified=True (simulates local pass)
# ---------------------------------------------------------------------------


def _make_always_passing_pipeline() -> VerifyRepairPipeline:
    """Create a mocked pipeline that always returns verified=True.

    **Detailed explanation for engineers:**
        This simulates a local per-step verification system that cannot detect
        cross-step contradictions — it sees each step in isolation and always
        confirms verified=True. This is the baseline against which
        GlobalConsistencyChecker is compared.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    pipeline = MagicMock(spec=VerifyRepairPipeline)
    pipeline.extract_constraints.return_value = []
    pipeline.verify.return_value = VerificationResult(
        verified=True,
        constraints=[],
        energy=0.0,
        violations=[],
    )
    return pipeline


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class ChainResult:
    """Result of running one chain through local and global checkers.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """

    chain_id: int
    chain_type: str
    contradiction_type: str | None
    expected_consistent: bool
    local_detected: bool
    global_detected: bool
    global_report_severity: str
    inconsistent_pairs: list[tuple[int, int, str, str]]
    latency_ms: float


def run_chain(chain: ChainSpec) -> ChainResult:
    """Run a single chain and evaluate local vs global detection.

    **Detailed explanation for engineers:**
        1. Creates a ConstraintStateMachine with a mocked pipeline.
        2. Feeds each step's output_text through machine.step().
        3. Local detection: True iff any step has verification.verified == False.
           (With the mock, this is always False — local check misses everything.)
        4. Global detection: True iff check_global_consistency().consistent == False.
        5. Returns a ChainResult capturing both detection outcomes and latency.

    Args:
        chain: The ChainSpec to evaluate.

    Returns:
        ChainResult with all detection outcomes.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    t0 = time.perf_counter()

    machine = ConstraintStateMachine(pipeline=_make_always_passing_pipeline())
    step_results = []
    for idx, output_text in enumerate(chain.output_texts):
        result = machine.step(f"Step {idx} question", output_text)
        step_results.append(result)

    # Local detection: any step has verified=False
    local_detected = any(not r.verification.verified for r in step_results)

    # Global detection: GlobalConsistencyChecker finds a contradiction
    checker = GlobalConsistencyChecker()
    report = checker.check(machine)
    global_detected = not report.consistent

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    return ChainResult(
        chain_id=chain.chain_id,
        chain_type=chain.chain_type,
        contradiction_type=chain.contradiction_type,
        expected_consistent=chain.expected_consistent,
        local_detected=local_detected,
        global_detected=global_detected,
        global_report_severity=report.severity,
        inconsistent_pairs=list(report.inconsistent_pairs),
        latency_ms=latency_ms,
    )


def run_benchmark() -> dict[str, Any]:
    """Run the full Exp 172 benchmark and return structured results.

    **Detailed explanation for engineers:**
        Builds all 20 chains, runs each, computes per-chain and aggregate
        metrics. Returns a dict suitable for JSON serialisation and saving
        to results/experiment_172_results.json.

        Key aggregate metrics:
        - local_detection_rate: fraction of inconsistent chains caught by local check
        - global_detection_rate: fraction of inconsistent chains caught by global check
        - false_positive_rate: fraction of consistent chains falsely flagged by global check
        - detection_improvement: global_rate - local_rate (the gap GlobalConsistency closes)

    Returns:
        Dict with "experiment", "date", "chains", and "summary" keys.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    print("=" * 60)
    print("Experiment 172 — Global Consistency Checking Benchmark")
    print("=" * 60)
    print(f"  20 chains × 4 steps, pipeline always-passes (mocked)")
    print(f"  Contradiction types: numeric (4), arithmetic (3), factual (3)")
    print(f"  Fully consistent chains: 10")
    print()

    chains = _build_all_chains()
    chain_results: list[ChainResult] = []

    for chain in chains:
        result = run_chain(chain)
        chain_results.append(result)
        marker = "✓" if result.global_detected == (not result.expected_consistent) else "✗"
        print(
            f"  Chain {result.chain_id:02d} [{result.chain_type[:10]:10s}]"
            f" ctype={str(result.contradiction_type):10s}"
            f" local={result.local_detected}"
            f" global={result.global_detected}"
            f" sev={result.global_report_severity:8s}"
            f" {marker}"
            f" {result.latency_ms:.1f}ms"
        )

    # Aggregate metrics
    inconsistent_results = [r for r in chain_results if not r.expected_consistent]
    consistent_results = [r for r in chain_results if r.expected_consistent]

    local_detected_count = sum(r.local_detected for r in inconsistent_results)
    global_detected_count = sum(r.global_detected for r in inconsistent_results)
    false_positives = sum(r.global_detected for r in consistent_results)

    n_inconsistent = len(inconsistent_results)
    n_consistent = len(consistent_results)

    local_rate = local_detected_count / n_inconsistent if n_inconsistent > 0 else 0.0
    global_rate = global_detected_count / n_inconsistent if n_inconsistent > 0 else 0.0
    fp_rate = false_positives / n_consistent if n_consistent > 0 else 0.0
    improvement = global_rate - local_rate

    # Per-type breakdown
    type_breakdown: dict[str, dict[str, Any]] = {}
    for ctype in ["numeric", "arithmetic", "factual"]:
        type_chains = [r for r in inconsistent_results if r.contradiction_type == ctype]
        if type_chains:
            type_detected = sum(r.global_detected for r in type_chains)
            type_breakdown[ctype] = {
                "n_chains": len(type_chains),
                "global_detected": type_detected,
                "detection_rate": type_detected / len(type_chains),
            }

    avg_latency = sum(r.latency_ms for r in chain_results) / len(chain_results)

    print()
    print("Summary:")
    print(f"  Inconsistent chains: {n_inconsistent}")
    print(f"  Consistent chains:   {n_consistent}")
    print(f"  Local detection rate:   {local_rate:.1%} ({local_detected_count}/{n_inconsistent})")
    print(f"  Global detection rate:  {global_rate:.1%} ({global_detected_count}/{n_inconsistent})")
    print(f"  False positive rate:    {fp_rate:.1%} ({false_positives}/{n_consistent})")
    print(f"  Detection improvement:  +{improvement:.1%}")
    print(f"  Avg latency per chain:  {avg_latency:.2f}ms")
    print()
    print("Per-type detection:")
    for ctype, stats in type_breakdown.items():
        print(
            f"  {ctype:12s}: {stats['global_detected']}/{stats['n_chains']}"
            f" = {stats['detection_rate']:.1%}"
        )
    print()

    # Build serialisable chain records
    chain_records = []
    for r in chain_results:
        chain_records.append({
            "chain_id": r.chain_id,
            "chain_type": r.chain_type,
            "contradiction_type": r.contradiction_type,
            "expected_consistent": r.expected_consistent,
            "local_detected": r.local_detected,
            "global_detected": r.global_detected,
            "global_report_severity": r.global_report_severity,
            "n_inconsistent_pairs": len(r.inconsistent_pairs),
            "inconsistent_pairs": [
                {"step_i": i, "step_j": j, "type": ctype, "description": desc}
                for i, j, ctype, desc in r.inconsistent_pairs
            ],
            "latency_ms": round(r.latency_ms, 3),
        })

    return {
        "experiment": "172_global_consistency",
        "date": "2026-04-11",
        "target_models": ["Qwen3.5-0.8B", "google/gemma-4-E4B-it"],
        "description": (
            "Benchmark GlobalConsistencyChecker (Exp 172) on 20 synthetic "
            "4-step chains: 10 locally-consistent globally-inconsistent, "
            "10 fully-consistent. Measures detection improvement over local "
            "per-step verification. Theoretical basis: arxiv 2601.13600."
        ),
        "chains": chain_records,
        "summary": {
            "n_chains_total": len(chain_results),
            "n_inconsistent_chains": n_inconsistent,
            "n_consistent_chains": n_consistent,
            "local_detection_rate": round(local_rate, 4),
            "global_detection_rate": round(global_rate, 4),
            "false_positive_rate": round(fp_rate, 4),
            "detection_improvement": round(improvement, 4),
            "avg_latency_ms": round(avg_latency, 3),
            "per_type_detection": type_breakdown,
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 172 and save results to results/experiment_172_results.json.

    Spec: REQ-VERIFY-001, SCENARIO-VERIFY-005
    """
    results = run_benchmark()

    output_path = Path(__file__).parent.parent / "results" / "experiment_172_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")

    # Exit with error if global detection rate is < 80% (sanity check)
    global_rate = results["summary"]["global_detection_rate"]
    if global_rate < 0.8:
        print(
            f"ERROR: global detection rate {global_rate:.1%} below 80% threshold.",
            file=sys.stderr,
        )
        sys.exit(1)

    fp_rate = results["summary"]["false_positive_rate"]
    if fp_rate > 0.1:
        print(
            f"ERROR: false positive rate {fp_rate:.1%} above 10% threshold.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Experiment 172 PASSED sanity checks.")


if __name__ == "__main__":
    main()
