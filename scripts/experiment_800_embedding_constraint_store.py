#!/usr/bin/env python3
"""Exp 800: EmbeddingConstraintStore — SPO Format + Orthogonality Regularization.

**Researcher summary (RETRO-CONSTRAINT-ZERO-DELTA):**
    Exp 788 produced constraint_addition_delta=0.0.  Root cause (arXiv 2601.15313):
    scalar keyword-count encoding causes embedding collapse — semantically similar
    constraints become indistinguishable vectors, so retrieval always returns the
    same top-K regardless of query.

    Fix: encode constraints as SPO (Subject-Predicate-Object) triples using
    sentence-transformer embeddings (all-MiniLM-L6-v2, CPU), with orthogonality
    regularization (Gram-Schmidt projection) to force each constraint type into
    a distinct embedding subspace.

    This experiment validates the fix by measuring retrieval_auc on 5 constraint
    types (carry, sign, unit, comparison, causal) with 10 queries each.

**honest_verdict logic:**
    - "retrieval_auc_exceeds_target" if retrieval_auc > 0.70
    - "retrieval_auc_partial"        if retrieval_auc in [0.50, 0.70]
    - "retrieval_auc_below_chance"   if retrieval_auc < 0.50

Spec: REQ-LEARN-057, REQ-LEARN-058, REQ-LEARN-059, SCENARIO-LEARN-098
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# apply_env_autofix MUST be called before any JAX or CUDA import.
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.embedding_constraint_store import (  # noqa: E402
    ConstraintSPOTuple,
    EmbeddingConstraintStore,
)
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 800
TITLE = "EmbeddingConstraintStore — SPO Format + Orthogonality Regularization"
DELIVERABLE = "results/experiment_800_embedding_constraint_store.json"

_WATCHDOG = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=45)

tmpl = ExperimentTemplate(
    exp_id=EXP_ID,
    title=TITLE,
    deliverable=DELIVERABLE,
    requires_gpu=False,
)
tmpl.setup()


# ---------------------------------------------------------------------------
# Test query bank — 10 queries per constraint type (50 total)
# ---------------------------------------------------------------------------

def _build_queries() -> tuple[list[str], list[str]]:
    """Build 50 (query, label) pairs for retrieval_auc evaluation.

    Why 10 per type:
        10 queries × 5 types = 50 is enough to detect systematic retrieval
        failures (binomial 95% CI at 0.70 AUC: roughly ±13pp per type).
        These are realistic error descriptions drawn from GSM8K-style problems.

    Returns:
        (queries, labels) where labels parallel queries with source_violation_type.
    """
    carry_queries = [
        "the addition of 37 and 45 dropped the carry bit giving 72 instead of 82",
        "carry propagation was omitted in the tens column of the addition",
        "ripple carry from the ones digit was not propagated to hundreds",
        "the addition result is 10 less than expected because carry was lost",
        "binary carry out from bit 3 was discarded causing wrong sum",
        "carry between digits is missing in the intermediate computation",
        "addition dropped the carry flag producing an undercount",
        "the sum ignores carry from lower byte leading to truncated result",
        "arithmetic carry was not included in the final addition step",
        "carry digit omitted during column-by-column addition producing wrong total",
    ]
    sign_queries = [
        "5 minus negative 3 was computed as 2 instead of 8 due to sign error",
        "negative sign was flipped incorrectly when subtracting a negative number",
        "the sign of the result is wrong — subtraction of negation treated as addition",
        "sign inversion error: -(-x) was computed as -x",
        "the response gives a negative answer but the correct value is positive",
        "sign bit was negated incorrectly producing a wrong signed result",
        "subtracting a negative produced a sum less than the minuend",
        "double negation collapse was skipped giving the wrong sign",
        "sign error in the final step reversed positive and negative",
        "sign preservation violated: positive minus negative should be positive",
    ]
    unit_queries = [
        "10 km plus 500 m was added as 10.5 km without converting to common unit",
        "units were mixed: meters added to kilometers without conversion",
        "the answer gives km but the inputs include meters without normalization",
        "unit mismatch: feet and inches were added without converting to a common unit",
        "liters and milliliters mixed without unit conversion in the sum",
        "unit consistency violated: celsius and fahrenheit added directly",
        "weight in grams was added to kilograms without converting to grams first",
        "speed in mph and km/h were summed without unit normalization",
        "unit label error: the answer uses mixed units in the output expression",
        "unit conversion was skipped before combining distance measurements",
    ]
    comparison_queries = [
        "8 greater than 10 was marked as true — comparison direction was flipped",
        "the inequality direction is wrong: less-than was used where greater-than applies",
        "comparison result is reversed: the response says the smaller number is larger",
        "the ordering of two values is incorrect — the maximum was chosen incorrectly",
        "strict inequality was violated: the wrong side of the comparison was selected",
        "the response picked the smaller value but the problem requires the maximum",
        "comparison operator flipped: >= treated as <= in the conditional branch",
        "greater-than comparison produced the wrong boolean due to direction error",
        "the answer used less-than where the constraint requires greater-than",
        "inequality direction violated: the larger number was incorrectly labeled smaller",
    ]
    causal_queries = [
        "the conclusion does not follow from the preceding step in the chain",
        "step 3 assumes a result that contradicts the output of step 2",
        "causal entailment violated: the effect is stated before the cause is established",
        "intermediate step was skipped, breaking the causal chain",
        "step causality error: the final result contradicts the chain of reasoning",
        "the deduction at step 4 requires a fact not established in prior steps",
        "logical entailment chain is broken between steps 2 and 3",
        "the conclusion cannot follow from the premises as stated",
        "causal dependency violated: later step relies on unproven earlier claim",
        "step-by-step reasoning has a broken link between the third and fourth step",
    ]

    queries: list[str] = (
        carry_queries + sign_queries + unit_queries + comparison_queries + causal_queries
    )
    labels: list[str] = (
        ["carry"] * 10 + ["sign"] * 10 + ["unit"] * 10 + ["comparison"] * 10 + ["causal"] * 10
    )
    return queries, labels


def _run_store(apply_orthogonalization: bool) -> tuple[EmbeddingConstraintStore, float, float]:
    """Initialize and populate a store, then compute retrieval_auc.

    Args:
        apply_orthogonalization: When True, uses the standard EmbeddingConstraintStore
            (orthogonalization enabled by default).  When False, patches the store
            to skip orthogonalization so we can compare both modes.

    Returns:
        (store, retrieval_auc_score, elapsed_seconds)
    """
    t0 = time.monotonic()
    store = EmbeddingConstraintStore()

    if not apply_orthogonalization:
        # Monkey-patch _orthogonalize to identity (no projection) for comparison.
        # This simulates the Exp 788 failure mode — no orthogonalization.
        store._orthogonalize = lambda v: [  # type: ignore[method-assign]
            x / max(sum(xi * xi for xi in v) ** 0.5, 1e-12) for x in v
        ]

    # Bootstrap from the five canonical violation-type patterns.
    # This mirrors what would happen if CaseMemory had accumulated these patterns.
    store.from_casememory_patterns(
        {"carry": 4, "sign": 4, "unit": 4, "comparison": 4, "causal": 4}
    )

    queries, labels = _build_queries()
    auc = store.retrieval_auc(queries, labels)
    elapsed = time.monotonic() - t0
    return store, auc, elapsed


def main() -> None:
    """Run Exp 800: compare orthogonalized vs non-orthogonalized retrieval_auc."""
    print(f"[Exp {EXP_ID}] Running {TITLE}")

    store_ortho, auc_ortho, elapsed_ortho = _run_store(apply_orthogonalization=True)
    print(f"  Orthogonalized  retrieval_auc = {auc_ortho:.4f}  ({elapsed_ortho:.1f}s)")

    store_plain, auc_plain, elapsed_plain = _run_store(apply_orthogonalization=False)
    print(f"  No-ortho (plain) retrieval_auc = {auc_plain:.4f}  ({elapsed_plain:.1f}s)")

    auc_delta = auc_ortho - auc_plain
    embedding_mode = store_ortho.embedding_mode

    if auc_ortho > 0.70:
        honest_verdict = "retrieval_auc_exceeds_target"
    elif auc_ortho >= 0.50:
        honest_verdict = "retrieval_auc_partial"
    else:
        honest_verdict = "retrieval_auc_below_chance"

    print(f"  embedding_mode    = {embedding_mode}")
    print(f"  auc_delta (ortho - plain) = {auc_delta:.4f}")
    print(f"  honest_verdict    = {honest_verdict}")

    artifact = tmpl.build_result(
        {
            "retrieval_auc_orthogonalized": auc_ortho,
            "retrieval_auc_plain": auc_plain,
            "auc_delta": auc_delta,
            "embedding_mode": embedding_mode,
            "model_name": store_ortho.model_name,
            "n_constraint_types": 5,
            "n_queries_total": 50,
            "n_queries_per_type": 10,
            "honest_verdict": honest_verdict,
            "elapsed_ortho_s": round(elapsed_ortho, 3),
            "elapsed_plain_s": round(elapsed_plain, 3),
        },
        status="success",
    )
    Path(DELIVERABLE).write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
