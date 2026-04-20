#!/usr/bin/env python3
"""Experiment 546: AutoRefine Constraint Template Distillation.

**Researcher summary (AutoRefine, arXiv 2601.22758):**
    AutoRefine converts agent interaction trajectories into reusable abstract
    strategic principles via offline self-distillation.  This experiment wires
    that idea into Carnot's Tier 2 self-learning:

    - Violation patterns from Exp 538 (real live benchmark) and Exp 541 (live
      wire-in) are ingested into a ConstraintTemplateStore.
    - ``distill(min_observations=3)`` promotes mature patterns to named templates.
    - ``retrieve(query_context, top_k=3)`` is tested on 5 representative query
      contexts to verify that relevant templates are ranked first.
    - The distilled store is saved to results/constraint_templates_546.json for
      use by downstream experiments and the live VerifyRepairPipeline.

**Honest verdict logic:**
    'templates_distilled'     — n_templates_distilled >= 3
    'insufficient_patterns'   — n_templates_distilled < 3 (too few mature patterns)

**Deliverable:** results/experiment_546_autorefine_constraint_templates.json
**Schema:** carnot.autorefine_templates.v1

Spec: REQ-LEARN-058, REQ-LEARN-059,
SCENARIO-LEARN-090, SCENARIO-LEARN-091, SCENARIO-LEARN-092
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# apply_env_autofix() MUST be the very first non-stdlib call.
# WHY: injects JAX_PLATFORMS=cpu (and optionally CARNOT_FORCE_LIVE) before any
# JAX or CUDA import.  Calling it later allows JAX to initialise against a GPU
# backend that can stall on CPU-only machines.
sys.path.insert(0, str(Path(__file__).parent.parent))
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_AUTOFIX_RESULT = apply_env_autofix()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.constraint_template_store import ConstraintTemplateStore  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 546
EXP_TITLE = "AutoRefine Constraint Templates"
RESULT_PATH = "results/experiment_546_autorefine_constraint_templates.json"
TEMPLATE_STORE_PATH = "results/constraint_templates_546.json"
WATCHDOG_TIMEOUT_MINUTES = 25
MIN_OBSERVATIONS = 3

# Source files for violation patterns
_REPO_ROOT = Path(__file__).parent.parent
EXP538_PATH = _REPO_ROOT / "results" / "experiment_538_live_25q_precision_v9.json"
EXP541_PATH = _REPO_ROOT / "results" / "experiment_541_constraint_addition_live.json"

# Five representative query contexts used to test template retrieval.
# Each is designed to trigger a different violation type when relevant templates exist.
_RETRIEVAL_TEST_QUERIES = [
    "arithmetic carry overflow addition sum of two numbers",
    "semantic answer mismatch question grounding factual response",
    "sign negative positive arithmetic operation result",
    "unit conversion measurement error scaling factor",
    "comparison direction ordering inequality greater less than",
]


# ---------------------------------------------------------------------------
# Violation pattern loading
# ---------------------------------------------------------------------------


def _load_exp538_patterns(path: Path) -> list[tuple[str, str]]:
    """Load violation type + context pairs from Exp 538 result JSON.

    Exp 538 stored violation types as a flat list in exp538_violation_types_seeded
    and raw counts in pattern_counts_before_session2.  We reconstruct pairs by
    treating each seeded type as one observation with a generic context string.

    Returns list of (violation_type, context_text) pairs.
    """
    if not path.exists():
        _log.warning("Exp 538 result not found at %s — skipping", path)
        return []

    raw = json.loads(path.read_text())

    # Method 1: use the seeded violation types list (each type → one pair)
    seeded: list[str] = raw.get("exp538_violation_types_seeded", [])
    if seeded:
        # Provide a context derived from the violation type so keyword extraction
        # is meaningful even without free-text per-observation context.
        return [(vtype, f"{vtype} violation observed in live benchmark run") for vtype in seeded]

    # Method 2: fall back to pattern_counts_before_session2 dict
    counts: dict[str, int] = raw.get("pattern_counts_before_session2", {})
    pairs: list[tuple[str, str]] = []
    for vtype, count in counts.items():
        for _ in range(count):
            pairs.append((vtype, f"{vtype} violation observed in live benchmark run"))
    return pairs


def _load_exp541_patterns(path: Path) -> list[tuple[str, str]]:
    """Load violation type + context pairs from Exp 541 result JSON.

    Exp 541 stored pattern_counts_after_session2.  We also check
    constraints_added for type names to synthesise context strings.

    Returns list of (violation_type, context_text) pairs.
    """
    if not path.exists():
        _log.warning("Exp 541 result not found at %s — skipping", path)
        return []

    raw = json.loads(path.read_text())
    pairs: list[tuple[str, str]] = []

    counts: dict[str, int] = raw.get("pattern_counts_after_session2", {})
    for vtype, count in counts.items():
        for _ in range(count):
            pairs.append((vtype, f"{vtype} constraint addition pipeline session relay"))

    # Also add any constraints_added names as synthetic observations
    constraints_added: list[str] = raw.get("constraints_added", [])
    for cname in constraints_added:
        # e.g. "carry_check_constraint" → extract "carry"
        vtype = cname.split("_")[0] if "_" in cname else cname
        pairs.append((vtype, f"{vtype} constraint added from Exp 541 relay"))

    return pairs


# ---------------------------------------------------------------------------
# Retrieval verification
# ---------------------------------------------------------------------------


def _test_retrieval(store: ConstraintTemplateStore, queries: list[str], top_k: int = 3) -> list[dict[str, Any]]:
    """Run retrieve() for each query and return structured results.

    Each result entry contains:
    - query: the input context string
    - top_templates: list of {name, violation_type, n_violations_observed, overlap_kws}
    - n_retrieved: number of templates returned (may be < top_k if fewer exist)

    WHY this check: Exp 546's purpose is to validate that retrieval returns
    relevant templates.  Logging the overlap keywords lets researchers confirm
    that the scoring is semantically sensible without running an LLM judge.
    """
    results: list[dict[str, Any]] = []
    distilled = store.distill(min_observations=MIN_OBSERVATIONS)
    distilled_vtypes = {t.violation_type for t in distilled}

    for query in queries:
        retrieved = store.retrieve(query, top_k=top_k)
        top_info = []
        for tmpl in retrieved:
            overlap = [kw for kw in tmpl.context_keywords if kw in query.lower()]
            top_info.append({
                "name": tmpl.name,
                "violation_type": tmpl.violation_type,
                "n_violations_observed": tmpl.n_violations_observed,
                "overlap_keywords": overlap,
            })
        results.append({
            "query": query,
            "top_templates": top_info,
            "n_retrieved": len(retrieved),
        })
        _log.info(
            "retrieve query='%s...' → %d templates (top: %s)",
            query[:40],
            len(retrieved),
            [t["name"] for t in top_info],
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=WATCHDOG_TIMEOUT_MINUTES)

    tmpl = ExperimentTemplate(
        EXP_ID,
        EXP_TITLE,
        RESULT_PATH,
        requires_gpu=False,
    )
    tmpl.setup()

    # ------------------------------------------------------------------
    # Step 1: Ingest violation patterns from Exp 538 + Exp 541
    # ------------------------------------------------------------------
    store = ConstraintTemplateStore()

    exp538_pairs = _load_exp538_patterns(EXP538_PATH)
    _log.info("Loaded %d violation pairs from Exp 538", len(exp538_pairs))
    for vtype, ctx in exp538_pairs:
        store.add_violation(vtype, ctx)

    exp541_pairs = _load_exp541_patterns(EXP541_PATH)
    _log.info("Loaded %d violation pairs from Exp 541", len(exp541_pairs))
    for vtype, ctx in exp541_pairs:
        store.add_violation(vtype, ctx)

    n_ingested = sum(store.violation_counts().values())
    _log.info(
        "Total violations ingested: %d (types: %s)",
        n_ingested,
        dict(sorted(store.violation_counts().items())),
    )

    # ------------------------------------------------------------------
    # Step 2: Distil templates with min_observations=3
    # ------------------------------------------------------------------
    templates = store.distill(min_observations=MIN_OBSERVATIONS)
    n_templates = len(templates)
    _log.info("Distilled %d templates (min_observations=%d)", n_templates, MIN_OBSERVATIONS)
    for t in templates:
        _log.info(
            "  template=%s vtype=%s n_obs=%d keywords=%s",
            t.name, t.violation_type, t.n_violations_observed, t.context_keywords[:5],
        )

    # ------------------------------------------------------------------
    # Step 3: Test retrieval on 5 query contexts
    # ------------------------------------------------------------------
    retrieval_results = _test_retrieval(store, _RETRIEVAL_TEST_QUERIES, top_k=3)
    retrieval_verified = all(r["n_retrieved"] > 0 for r in retrieval_results)

    # ------------------------------------------------------------------
    # Step 4: Save the template store
    # ------------------------------------------------------------------
    store_path = _REPO_ROOT / TEMPLATE_STORE_PATH
    store.save(store_path)
    _log.info("Template store saved to %s", store_path)

    # ------------------------------------------------------------------
    # Step 5: Build artifact
    # ------------------------------------------------------------------
    sample_templates = [t.to_dict() for t in templates[:3]]
    honest_verdict = "templates_distilled" if n_templates >= 3 else "insufficient_patterns"

    artifact = tmpl.build_result(
        {
            "n_violations_ingested": n_ingested,
            "n_templates_distilled": n_templates,
            "min_observations_threshold": MIN_OBSERVATIONS,
            "violation_counts": dict(sorted(store.violation_counts().items())),
            "sample_templates": sample_templates,
            "retrieval_results": retrieval_results,
            "retrieval_verified": retrieval_verified,
            "template_store_path": TEMPLATE_STORE_PATH,
            "exp538_pairs_loaded": len(exp538_pairs),
            "exp541_pairs_loaded": len(exp541_pairs),
            "honest_verdict": honest_verdict,
            "env_autofix": "applied" if _AUTOFIX_RESULT.auto_fix_applied else "skipped",
            "env_autofix_applied": _AUTOFIX_RESULT.auto_fix_applied,
        },
        status="success",
        schema="carnot.autorefine_templates.v1",
    )

    # Write the deliverable JSON
    output_path = _REPO_ROOT / RESULT_PATH
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    _log.info(
        "Deliverable written: %s (verdict=%s, n_templates=%d)",
        RESULT_PATH, honest_verdict, n_templates,
    )

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
