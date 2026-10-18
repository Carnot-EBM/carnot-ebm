#!/usr/bin/env python3
"""Exp 833 — Constraint-Delta Root-Cause Diagnosis.

**Researcher summary:**
    Exp 821 showed delta_overall=0.0 across 3 sessions of 30 GSM8K questions despite
    the external field injection fix from Exp 819.  Three hypotheses:

    H1: VerifyRepairPipeline.verify() detects violations but never calls
        EmbeddingConstraintStore.store() — the write path is structurally absent.
    H2: store() is called but retrieval returns near-zero vectors (collapsed embeddings).
    H3: compute_energy_with_external_field is never called; legacy energy() path is used.

    This experiment instruments the live pipeline to determine which hypothesis holds.
    It does NOT change pipeline behaviour — it wraps key methods with call-counting
    shims and runs 5 synthetic GSM8K-style questions with KNOWN violations.

**Diagnosis methodology:**
    1. Wrap EmbeddingConstraintStore.store() to count calls (n_store_write_calls).
    2. Wrap EmbeddingConstraintStore.retrieve() to count calls and measure retrieved
       vector norm (mean_retrieved_vector_norm).
    3. Wrap IsingConstraintInjector.compute_energy_with_external_field() to count calls.
    4. Track calls to the legacy project_to_spin_bias() path (used for logging only
       in current code — n_legacy_energy_calls proxy).
    5. Run 5 test cases through VerifyRepairPipeline.verify() with the real objects.
    6. Examine counters to identify the broken stage.

**Decision tree:**
    - n_store_write_calls == 0           → H1 confirmed: write_path_missing
    - writes > 0, vector_norm < 0.01     → H2 confirmed: retrieval_returns_zeros
    - legacy_calls > 0, ext_field == 0   → H3 confirmed: external_field_not_called
    - all paths correct                  → pipeline_wiring_correct
    - none of the above                  → diagnosis_inconclusive

Spec: REQ-LEARN-048, REQ-LEARN-049, SCENARIO-LEARN-060
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

# Ensure project root on PYTHONPATH for imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Synthetic test data: 5 GSM8K-style questions with known violations.
# ground_truth_correct[i] == False means the answer contains an arithmetic error.
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, str]] = [
    (
        "What is 47 + 28?",
        "Step 1: 47 + 28 = 76. The answer is 76.",  # WRONG (should be 75)
    ),
    (
        "What is 100 - 37?",
        "Step 1: 100 - 37 = 63. The answer is 63.",  # correct
    ),
    (
        "What is 6 * 9?",
        "Step 1: 6 * 9 = 42. The answer is 42.",  # WRONG (should be 54)
    ),
    (
        "What is 15% of 200?",
        "Step 1: 15% of 200 = 0.15 * 200 = 30. The answer is 30.",  # correct
    ),
    (
        "What is 3^3?",
        "Step 1: 3^3 = 3 * 3 * 3 = 27. The answer is 27.",  # correct
    ),
]

# Whether each response is expected to be correct (True) or contain a violation (False).
GROUND_TRUTH_CORRECT: list[bool] = [False, True, False, True, True]


# ---------------------------------------------------------------------------
# Instrumentation helpers
# ---------------------------------------------------------------------------


def _instrument_store(store: Any, counters: dict[str, Any]) -> None:
    """Wrap store.store() and store.retrieve() with call-counting shims.

    Why in-place patching instead of subclassing:
        We want to use the real EmbeddingConstraintStore object that would be
        passed to verify() in production — subclassing would change the type and
        could bypass isinstance checks.  Replacing individual methods on the
        instance (not the class) is the minimal, non-invasive instrumentation.

    Args:
        store: An EmbeddingConstraintStore instance to instrument.
        counters: Dict that accumulates n_store_write_calls, n_store_retrieve_calls,
                  and retrieved_vector_norms (list of per-call mean norms).
    """
    original_store_method = store.store
    original_retrieve_method = store.retrieve

    def _counted_store(spo: Any) -> None:
        counters["n_store_write_calls"] += 1
        original_store_method(spo)

    def _counted_retrieve(query: str, top_k: int = 3) -> list[Any]:
        counters["n_store_retrieve_calls"] += 1
        results = original_retrieve_method(query, top_k)
        # Measure mean vector norm of retrieved embeddings to detect H2 (zero vectors).
        norms = []
        for spo in results:
            if spo.embedding is not None:
                norm = math.sqrt(sum(x * x for x in spo.embedding))
                norms.append(norm)
        if norms:
            counters["retrieved_vector_norms"].append(sum(norms) / len(norms))
        return results

    store.store = _counted_store
    store.retrieve = _counted_retrieve


def _instrument_injector(injector: Any, counters: dict[str, Any]) -> None:
    """Wrap IsingConstraintInjector methods with call-counting shims.

    Tracks:
        n_external_field_calls: calls to compute_energy_with_external_field().
        n_legacy_energy_calls:  calls to project_to_spin_bias() (proxy for the
                                logging-only path in the current verify() code).

    Why track project_to_spin_bias as a "legacy" proxy:
        In the current verify() code, project_to_spin_bias() is called inside a
        logging block but the result is never used for energy computation.
        So project_to_spin_bias calls > 0 AND external_field calls == 0 is the
        signature of H3 (external field path not wired up).

    Args:
        injector: An IsingConstraintInjector instance to instrument.
        counters: Dict that accumulates n_external_field_calls and n_legacy_energy_calls.
    """
    original_ext_field = injector.compute_energy_with_external_field
    original_spin_bias = injector.project_to_spin_bias

    def _counted_ext_field(J: Any, spins: Any, constraint_embeddings: Any) -> Any:
        counters["n_external_field_calls"] += 1
        return original_ext_field(J, spins, constraint_embeddings)

    def _counted_spin_bias(constraint_embeddings: Any) -> Any:
        counters["n_legacy_energy_calls"] += 1
        return original_spin_bias(constraint_embeddings)

    injector.compute_energy_with_external_field = _counted_ext_field
    injector.project_to_spin_bias = _counted_spin_bias


# ---------------------------------------------------------------------------
# Verdict computation
# ---------------------------------------------------------------------------


def compute_honest_verdict(counters: dict[str, Any]) -> str:
    """Map counter values to one of the five diagnosis verdicts.

    Decision tree (in priority order):
        1. write_path_missing       — n_store_write_calls == 0
           (store() was never called; constraint accumulation is structurally absent)
        2. retrieval_returns_zeros  — writes > 0 but mean retrieved vector norm < 0.01
           (store() was called but embeddings are near-zero; retrieval cannot discriminate)
        3. external_field_not_called — legacy_energy_calls > 0 AND external_field_calls == 0
           (bias is computed for logging but never applied via compute_energy_with_external_field)
        4. pipeline_wiring_correct  — write, retrieve, and external_field all > 0 with sane norms
        5. diagnosis_inconclusive   — none of the above

    Args:
        counters: The accumulated counter dict from _instrument_store and _instrument_injector.

    Returns:
        str verdict label.

    Spec: REQ-LEARN-048, REQ-LEARN-049
    """
    n_writes = counters["n_store_write_calls"]
    n_retrieves = counters["n_store_retrieve_calls"]
    norms = counters["retrieved_vector_norms"]
    mean_norm = sum(norms) / len(norms) if norms else 0.0
    n_ext = counters["n_external_field_calls"]
    n_legacy = counters["n_legacy_energy_calls"]

    if n_writes == 0:
        return "write_path_missing"
    if n_writes > 0 and mean_norm < 0.01:
        return "retrieval_returns_zeros"
    if n_legacy > 0 and n_ext == 0:
        return "external_field_not_called"
    if n_writes > 0 and n_retrieves > 0 and n_ext > 0:
        return "pipeline_wiring_correct"
    return "diagnosis_inconclusive"


def build_fix_recommendation(honest_verdict: str) -> str:
    """Return a one-sentence fix recommendation for the confirmed root cause.

    The recommendation is written for engineers who will implement the fix —
    it names the exact method to call and where in the source file to add it.

    Args:
        honest_verdict: The string returned by compute_honest_verdict().

    Returns:
        str fix recommendation.
    """
    _FIXES = {
        "write_path_missing": (
            "Add a call to embedding_constraint_store.store(violation_spo) for each "
            "violation in result.violations inside VerifyRepairPipeline.verify() "
            "(python/carnot/pipeline/verify_repair.py, ~line 1360, after the existing "
            "self._memory.record_pattern() block)."
        ),
        "retrieval_returns_zeros": (
            "Investigate EmbeddingConstraintStore._orthogonalize() — near-zero norms "
            "after orthogonalization indicate the Gram-Schmidt step is collapsing all "
            "embeddings to zero.  Check that the sentence_transformers fallback "
            "(ci_hash) produces non-zero vectors and that repeated calls are not "
            "projecting all content away."
        ),
        "external_field_not_called": (
            "Replace the logging-only project_to_spin_bias() call in verify() with a "
            "real call to ising_constraint_injector.compute_energy_with_external_field() "
            "and incorporate the returned E_total into the VerificationResult energy."
        ),
        "pipeline_wiring_correct": (
            "All pipeline stages are wired correctly.  Investigate whether delta=0.0 "
            "is caused by retrieval_auc being too low (constraints not discriminating) "
            "or the external field magnitude being too small to shift the energy ranking."
        ),
        "diagnosis_inconclusive": (
            "Add finer-grained logging to identify which stage is failing; the current "
            "counters did not match any known failure signature."
        ),
    }
    return _FIXES.get(honest_verdict, "Unknown verdict — no fix recommendation available.")


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_diagnosis(
    test_cases: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Run the constraint-delta root-cause diagnosis and return the result dict.

    This function is separate from main() so tests can call it directly without
    going through the ExperimentTemplate lifecycle.

    Args:
        test_cases: Optional override for the test cases.  Defaults to TEST_CASES.

    Returns:
        Dict with all required diagnosis fields.
    """
    from carnot.pipeline.embedding_constraint_store import EmbeddingConstraintStore
    from carnot.pipeline.ising_constraint_injector import IsingConstraintInjector
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    if test_cases is None:
        test_cases = TEST_CASES

    # Initialise counter dict — all counters start at zero.
    counters: dict[str, Any] = {
        "n_store_write_calls": 0,
        "n_store_retrieve_calls": 0,
        "retrieved_vector_norms": [],
        "n_external_field_calls": 0,
        "n_legacy_energy_calls": 0,
    }

    # Build real pipeline objects — no mocks so we diagnose the real wiring.
    constraint_store = EmbeddingConstraintStore()
    injector = IsingConstraintInjector(embedding_dim=384, n_spins=64)

    # Instrument with call-counting shims (non-invasive — only affects this instance).
    _instrument_store(constraint_store, counters)
    _instrument_injector(injector, counters)

    # Build a minimal pipeline — no model required for verify-only mode.
    pipeline = VerifyRepairPipeline(
        model=None,
        domains=["arithmetic"],
        max_repairs=0,
        extractor=None,
        semantic_grounding_verifier=None,
        semantic_verifier_v2=None,
        timeout_seconds=30,
        memory=None,
        template_library=None,
        session_memory=None,
        constraint_memory=None,
        nup_probe=None,
        nup_probe_threshold=0.5,
    )

    # Run all test cases through the instrumented pipeline.
    verify_results = []
    for question, response in test_cases:
        result = pipeline.verify(
            question=question,
            response=response,
            domain="arithmetic",
            embedding_constraint_store=constraint_store,
            ising_constraint_injector=injector,
        )
        verify_results.append(
            {
                "question": question,
                "response": response[:60],
                "verified": result.verified,
                "n_violations": len(result.violations),
                "violation_types": [v.constraint_type for v in result.violations],
            }
        )

    # Compute aggregated metrics from counters.
    norms = counters["retrieved_vector_norms"]
    mean_retrieved_vector_norm = sum(norms) / len(norms) if norms else 0.0

    honest_verdict = compute_honest_verdict(counters)
    fix_recommendation = build_fix_recommendation(honest_verdict)

    # Derive which hypotheses are confirmed for explainability.
    hypothesis_confirmed = _map_verdict_to_hypothesis(honest_verdict)

    return {
        "hypothesis_confirmed": hypothesis_confirmed,
        "n_store_write_calls": counters["n_store_write_calls"],
        "n_store_retrieve_calls": counters["n_store_retrieve_calls"],
        "mean_retrieved_vector_norm": mean_retrieved_vector_norm,
        "n_external_field_calls": counters["n_external_field_calls"],
        "n_legacy_energy_calls": counters["n_legacy_energy_calls"],
        "root_cause": honest_verdict,
        "fix_recommendation": fix_recommendation,
        "honest_verdict": honest_verdict,
        "verify_results": verify_results,
        "embedding_mode": constraint_store.embedding_mode,
    }


def _map_verdict_to_hypothesis(verdict: str) -> str:
    """Map a verdict string to its human-readable hypothesis label.

    Args:
        verdict: One of the five honest_verdict strings.

    Returns:
        str hypothesis label (H1, H2, H3, or descriptive string).
    """
    _MAP = {
        "write_path_missing": "H1",
        "retrieval_returns_zeros": "H2",
        "external_field_not_called": "H3",
        "pipeline_wiring_correct": "none_all_correct",
        "diagnosis_inconclusive": "unknown",
    }
    return _MAP.get(verdict, "unknown")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 833 with full ExperimentTemplate lifecycle.

    Writes results/experiment_833_constraint_delta_root_cause.json on success.
    """
    tmpl = ExperimentTemplate(
        exp_id=833,
        title="Constraint-Delta Root-Cause Diagnosis",
        deliverable="results/experiment_833_constraint_delta_root_cause.json",
        requires_gpu=False,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(833, timeout_minutes=30)
    watchdog.start()

    try:
        diagnosis = run_diagnosis()

        artifact = tmpl.build_result(
            diagnosis,
            status="success",
            decision_class="verify",
        )

        out_path = Path("results/experiment_833_constraint_delta_root_cause.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))

    finally:
        watchdog.stop()

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
