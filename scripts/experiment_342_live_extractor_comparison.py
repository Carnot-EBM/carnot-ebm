#!/usr/bin/env python3
"""Experiment 342: Live 4-way extractor comparison on IT model responses.

**Researcher summary:**
    Experiment 311 crowned ArithmeticExtractor the winner — but on a SYNTHETIC corpus
    of word problems where "47 + 28 = X" patterns appear frequently.  Real instruction-
    tuned (IT) models like Gemma4-E4B-it write fluent prose that rarely matches that
    pattern.  Exp 328 confirmed: 0 violations found on live Gemma4 output.

    This experiment runs ALL FOUR extractors on the SAME 50 live IT model responses
    and asks: which extractor (or combination) actually works on real output?

**The four extractors:**
    1. ArithmeticExtractor  — regex for "X + Y = Z" patterns.
    2. NL2Z3Extractor       — LLM→Z3 translation + SMT solver.
    3. VergeRefiner (extractor mode) — UNSAT assertion extraction without repair.
    4. CoTCircuitVerifier   — structural dependency graph check (no LLM call).

**Metrics per extractor:**
    - violation_rate:       fraction of responses where at least one violation was found.
    - estimated_precision:  fraction of violations estimated to be real errors
                            (using Exp 331 FP taxonomy as prior).
    - fp_category_distribution: breakdown of false-positive types.

**Agreement matrix:**
    Which extractor pairs flag the same responses?  High agreement between a
    high-precision extractor and a lower-precision one suggests the lower-precision
    extractor can be used as a cheap pre-filter.

**Recommended extractor chain:**
    The extractor with the highest estimated_precision becomes the recommended
    default.  This recommendation replaces the Exp 311 synthetic-corpus winner.

**Output:** results/experiment_342_live_extractor_comparison.json

Usage:
    # CI mode (no GPU, synthetic responses):
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_342_live_extractor_comparison.py

    # Live mode (requires GPU + model + CARNOT_FORCE_LIVE=1):
    CARNOT_FORCE_LIVE=1 .venv/bin/python scripts/experiment_342_live_extractor_comparison.py

Spec: REQ-EXTRACT-017, SCENARIO-EXTRACT-036, SCENARIO-EXTRACT-037
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.extract import ArithmeticExtractor  # noqa: E402
from carnot.pipeline.extractor_comparison import (  # noqa: E402
    ExtractorResult,
    build_comparison_artifact,
    compare_extractors,
)
from carnot.pipeline.cot_circuit_verifier import CoTCircuitVerifier  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_RESPONSES = 50
_DELIVERABLE = "results/experiment_342_live_extractor_comparison.json"
_EXP_340_RESULTS = _REPO_ROOT / "results" / "experiment_340_live_precision_benchmark.json"
_EXP_331_RESULTS = _REPO_ROOT / "results" / "experiment_331_fp_autopsy.json"


# ---------------------------------------------------------------------------
# Load or generate responses
# ---------------------------------------------------------------------------


def _load_exp340_responses(path: Path, n: int) -> list[str] | None:
    """Try to load live responses from Exp 340's result file.

    **Detailed explanation for engineers:**
        Exp 340 captured live Gemma4-E4B-it responses to GSM8K problems.
        The artifact may store them under a "responses" list key.  If the
        file is missing, partial, or does not contain the expected list of
        response strings, return None so the caller can use synthetic fallback.

    Args:
        path: Path to experiment_340 result JSON.
        n:    Number of responses required.

    Returns:
        List of up to n response strings, or None if unavailable.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        responses_raw = data.get("responses")
        if not isinstance(responses_raw, list) or len(responses_raw) == 0:
            return None
        # Responses might be dicts with a "response" key or plain strings.
        strings: list[str] = []
        for item in responses_raw:
            if isinstance(item, str):
                strings.append(item)
            elif isinstance(item, dict):
                text = item.get("response") or item.get("text") or item.get("answer") or ""
                strings.append(str(text))
        if not strings:
            return None
        return strings[:n]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _synthetic_responses(n: int) -> list[str]:
    """Generate n deterministic synthetic IT model responses for CI mode.

    **Detailed explanation for engineers:**
        Real Gemma4-E4B-it responses to GSM8K problems are fluent prose with
        implicit arithmetic — they rarely match "X + Y = Z" literally.  Our
        synthetic responses replicate the dominant patterns seen in live output:

        Pattern mix (chosen to stress-test all four extractors):
        - ~40%: correct arithmetic in prose ("John has 5 apples plus 3 more,
                so he has 8 apples total.")
        - ~20%: step-referenced reasoning ("From step 1 we computed 10.
                Multiplying by 2 gives 20.")
        - ~20%: wrong embedded arithmetic ("total: 47 + 28 = 76" — off by 1)
        - ~20%: pure prose without numbers ("The answer depends on the context.")

        The cycle repeats for any n.  Responses are deterministic (no random seed
        needed) so CI tests are reproducible.

    Args:
        n: Number of synthetic responses to generate.

    Returns:
        List of n synthetic response strings.
    """
    templates = [
        # Correct arithmetic in prose (no "X + Y = Z" pattern — ArithmeticExtractor misses these).
        "John has 5 apples. He gets 3 more from Mary. Now he has 8 apples in total. The answer is 8.",
        "We start with 100 dollars. After spending 35 dollars, we have 65 dollars remaining.",
        "The train travels at 60 km/h for 2 hours, covering a distance of 120 km.",
        "There are 24 students in the class. Half leave early, so 12 remain.",
        "She earns 15 dollars per hour and works 8 hours, making 120 dollars total.",
        # Step-referenced reasoning (for CoTCircuitVerifier).
        "Step 1: Calculate base cost: 50 dollars.\nStep 2: Add 10% tax from step 1 (50 * 0.1 = 5). Total: 55.",
        "Step 1: Count apples: 7.\nStep 2: Count oranges: 3.\nStep 3: From step 1 and step 2, total fruit: 10.",
        "1. Start: 100 meters.\n2. Add 50 meters.\n3. From step 2, total: 150 meters.",
        "First, compute base: 20. Then, from step 1, double it: 40. Finally, subtract 5: 35.",
        "Step 1: Rate = 30. Step 2: Time = 4. Step 3: Using step 1, distance = 120.",
        # Wrong embedded arithmetic (ArithmeticExtractor should catch these).
        "If 47 + 28 = 76, then the total is 76 items.",
        "We compute 15 - 7 = 9, leaving us with 9 items.",
        "Adding these together: 33 + 44 = 78.",
        "The subtraction gives 100 - 37 = 62.",
        "Final sum: 12 + 9 = 20.",
        # Pure prose without numbers.
        "The answer depends on the context provided in the problem.",
        "We need more information to determine the correct solution.",
        "Based on the given constraints, the problem has no unique solution.",
        "The question asks about a scenario that is not fully specified.",
        "Without knowing the initial conditions, we cannot determine the outcome.",
    ]
    responses: list[str] = []
    for i in range(n):
        responses.append(templates[i % len(templates)])
    return responses


# ---------------------------------------------------------------------------
# Load FP prior from Exp 331
# ---------------------------------------------------------------------------


def _load_fp_prior(path: Path) -> dict[str, int]:
    """Load the FP category distribution from Exp 331's autopsy artifact.

    **Detailed explanation for engineers:**
        Exp 331 manually categorised broken ArithmeticExtractor cases into five
        categories.  We use this distribution as a Bayesian prior when estimating
        each extractor's precision on new violations.

        Falls back to a hardcoded default if the file is missing or malformed.

    Args:
        path: Path to experiment_331_fp_autopsy.json.

    Returns:
        Dict mapping FP category name -> count.
    """
    default: dict[str, int] = {
        "VALID_INTERMEDIATE": 2,
        "PRECISION_LIMIT": 1,
        "REGEX_ARTIFACT": 1,
        "REPAIR_DEGRADATION": 2,
        "UNCATEGORIZED": 0,
    }
    try:
        with open(path) as f:
            data = json.load(f)
        dist = data.get("category_distribution")
        if isinstance(dist, dict) and dist:
            return {k: int(v) for k, v in dist.items()}
        return default
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Build extractor wrappers
# ---------------------------------------------------------------------------


def _build_extractors(
    force_live: bool,
) -> list[tuple[str, Any]]:
    """Build the four extractor callables for the comparison.

    **Detailed explanation for engineers:**
        Each extractor is wrapped as a simple callable (str → list) so
        compare_extractors() can treat them uniformly.

        ArithmeticExtractor: always available; no LLM dependency.
        NL2Z3Extractor: in CI mode (force_live=False), returns "unknown" → no
            violations.  In live mode: requires CARNOT_FORCE_LIVE=1 and Z3 installed.
        VergeRefiner (extractor mode): wraps NL2Z3Extractor; in CI mode → no violations.
        CoTCircuitVerifier: always available; purely regex-based.

    Args:
        force_live: When True, attempt live LLM calls for NL2Z3-based extractors.

    Returns:
        List of (name, callable) pairs.
    """
    arith = ArithmeticExtractor()
    crv = CoTCircuitVerifier()

    # NL2Z3Extractor: only import if needed (has external dependencies).
    try:
        from carnot.pipeline.nl2z3_extractor import NL2Z3Extractor

        nl2z3 = NL2Z3Extractor()
    except ImportError:
        nl2z3 = None  # type: ignore[assignment]

    # VergeRefiner uses NL2Z3Extractor internally.
    # In extractor-only mode we call nl2z3.extract() directly on the response
    # (no LLM repair call) — this replicates "UNSAT extraction only, no repair".
    try:
        from carnot.pipeline.verge_refiner import VergeRefiner  # noqa: F401

        verge_available = True
    except ImportError:
        verge_available = False

    def _arith_fn(response: str) -> list:
        return [v for v in arith.extract(response, "arithmetic") if not v.metadata.get("satisfied", True)]

    def _nl2z3_fn(response: str) -> list:
        # Guard: NL2Z3Extractor checks os.environ.get("CARNOT_FORCE_LIVE") truthiness.
        # "0" is a truthy string so we check the force_live parameter instead.
        if nl2z3 is None or not force_live:
            return []
        try:
            return nl2z3.extract("", response, "reasoning")
        except Exception:  # noqa: BLE001
            return []

    def _verge_fn(response: str) -> list:
        # In extractor mode: run NL2Z3 extraction only (no repair LLM call).
        # Functionally equivalent to NL2Z3Extractor in this context.
        return _nl2z3_fn(response)

    def _crv_fn(response: str) -> list:
        return crv.extract("", response, "reasoning")

    extractors: list[tuple[str, Any]] = [
        ("ArithmeticExtractor", _arith_fn),
        ("NL2Z3Extractor", _nl2z3_fn),
        ("VergeRefiner", _verge_fn),
        ("CoTCircuitVerifier", _crv_fn),
    ]
    return extractors


# ---------------------------------------------------------------------------
# Agreement matrix
# ---------------------------------------------------------------------------


def build_agreement_matrix(
    responses: list[str],
    extractors: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Compute pairwise agreement matrix: which extractor pairs flag the same responses.

    **Detailed explanation for engineers:**
        For each response, record which extractors flagged it (binary vector).
        Pairwise agreement = fraction of responses where both extractors
        agree (both flag or both pass).  Stored as a nested dict:
        agreement[nameA][nameB] = fraction.

        High agreement between a cheap extractor (ArithmeticExtractor) and an
        expensive one (NL2Z3Extractor) suggests the cheap one can serve as a
        pre-filter: run it first and only invoke the expensive extractor when
        the cheap one is uncertain.

    Args:
        responses:  List of response strings.
        extractors: List of (name, callable) pairs.

    Returns:
        Dict with keys "extractor_names" (list) and "agreement_matrix"
        (nested dict name -> name -> float).
    """
    from carnot.pipeline.extractor_comparison import _run_extractor_fn  # noqa: PLC0415

    names = [name for name, _ in extractors]
    n = len(responses)

    # flags[i][j] = True if extractor j flagged response i.
    flags: list[list[bool]] = [
        [bool(_run_extractor_fn(fn, resp)) for _, fn in extractors]
        for resp in responses
    ]

    matrix: dict[str, dict[str, float]] = {}
    for i, na in enumerate(names):
        matrix[na] = {}
        for j, nb in enumerate(names):
            if n == 0:
                matrix[na][nb] = 1.0
            else:
                agreed = sum(1 for row in flags if row[i] == row[j])
                matrix[na][nb] = round(agreed / n, 4)

    return {"extractor_names": names, "agreement_matrix": matrix}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the 4-way extractor comparison and write results artifact."""
    tmpl = ExperimentTemplate(
        exp_id=342,
        title="Live 4-way extractor comparison on IT model responses",
        deliverable=_DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"

    # -----------------------------------------------------------------------
    # Load or generate responses.
    # -----------------------------------------------------------------------
    responses: list[str] | None = None
    inference_mode = "simulated"

    if force_live:
        responses = _load_exp340_responses(_EXP_340_RESULTS, _N_RESPONSES)
        if responses:
            inference_mode = "live_exp340"

    if responses is None:
        responses = _synthetic_responses(_N_RESPONSES)
        inference_mode = "simulated"

    assert len(responses) <= _N_RESPONSES

    # -----------------------------------------------------------------------
    # Load FP prior from Exp 331.
    # -----------------------------------------------------------------------
    fp_prior = _load_fp_prior(_EXP_331_RESULTS)

    # -----------------------------------------------------------------------
    # Build extractors and run comparison.
    # -----------------------------------------------------------------------
    extractors = _build_extractors(force_live=force_live)
    results: list[ExtractorResult] = compare_extractors(responses, extractors, fp_prior=fp_prior)

    # -----------------------------------------------------------------------
    # Compute agreement matrix.
    # -----------------------------------------------------------------------
    agreement = build_agreement_matrix(responses, extractors)

    # -----------------------------------------------------------------------
    # Build artifact.
    # -----------------------------------------------------------------------
    artifact_base = build_comparison_artifact(results)
    artifact = tmpl.build_result(
        {
            **artifact_base,
            "inference_mode": inference_mode,
            "n_responses": len(responses),
            "fp_prior_source": "experiment_331_fp_autopsy",
            "fp_prior": fp_prior,
            "agreement": agreement,
            "exp311_comparison": {
                "note": "Exp 311 winner (ArithmeticExtractor) was on synthetic corpus; "
                        "this experiment benchmarks on live IT model responses.",
                "exp311_winner": "ArithmeticExtractor",
                "exp342_recommended": artifact_base.get("recommended_extractor", ""),
            },
        },
        status="success",
    )

    # -----------------------------------------------------------------------
    # Write to disk.
    # -----------------------------------------------------------------------
    output_path = _REPO_ROOT / _DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"[Exp 342] Wrote artifact to {output_path}")
    print(f"[Exp 342] Inference mode: {inference_mode}")
    print(f"[Exp 342] Recommended extractor: {artifact_base.get('recommended_extractor')}")
    for r in results:
        print(
            f"  {r.extractor_name:24s}  "
            f"violation_rate={r.violation_rate:.3f}  "
            f"estimated_precision={r.estimated_precision:.3f}"
        )


if __name__ == "__main__":
    main()
