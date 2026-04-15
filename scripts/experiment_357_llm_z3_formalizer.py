#!/usr/bin/env python3
"""Experiment 357: LLM-guided Z3 formalization vs NL2Z3Extractor head-to-head.

**Researcher summary:**
    NL2Z3Extractor (Exp 310) was designed for plain chain-of-thought responses.
    When models produce instruction-tuned (IT) format output — markdown headers,
    numbered steps, boxed answers — the combined extraction + formalization task
    fails and Z3 receives malformed code (sat_status="error" or "unknown").

    LLMz3Formalizer (REQ-EXTRACT-019) separates these concerns: the LLM is only
    asked to translate numeric claims into z3.Solver().add() calls.  Inspired by
    arXiv 2601.04675, which showed 80% improvement in Z3 success rate from this
    kind of task decomposition.

    This experiment benchmarks both approaches on 20 synthetic IT-format responses
    with known arithmetic errors, measuring:
    - z3_success_rate: % of responses where Z3 produced SAT or UNSAT (not error/unknown)
    - fp_rate: false positives (flagged as UNSAT when arithmetic is actually correct)
    - tp_rate: true positives (flagged as UNSAT when arithmetic has a known error)
    - improvement_delta: llm_z3_success_rate - nl2z3_success_rate

**CI-safe mode:**
    When CARNOT_FORCE_LIVE is not set:
    - NL2Z3Extractor runs in its default CI mode → sat_status="unknown" for all
    - LLMz3Formalizer uses llm_caller=None → CI stub → z3_result="sat" for all
    - Both are honest about their mode in the artifact
    - nl2z3_success_rate = 0.0 (all unknown, CI guard)
    - llm_z3_success_rate = 1.0 (all sat, CI stub)
    - improvement_delta = 1.0 (CI stub always succeeds; not a real benchmark)
    - The artifact records inference_mode="simulated" so results are not headlined

**Output:** results/experiment_357_llm_z3_formalizer.json

Spec: REQ-EXTRACT-019, REQ-EXTRACT-020,
      SCENARIO-EXTRACT-039, SCENARIO-EXTRACT-040, SCENARIO-EXTRACT-041
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: ensure repo root is on sys.path so scripts.* and carnot.* resolve.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.llm_z3_formalizer import LLMz3Formalizer  # noqa: E402
from carnot.pipeline.nl2z3_extractor import NL2Z3Extractor  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 357
TITLE = "LLM-guided Z3 formalization vs NL2Z3Extractor head-to-head"
DELIVERABLE = "results/experiment_357_llm_z3_formalizer.json"
N_RESPONSES = 20

# ---------------------------------------------------------------------------
# Synthetic IT-format test corpus
# ---------------------------------------------------------------------------
# Each entry is (question, response, has_error).
# has_error=True means the response contains a deliberate arithmetic mistake.
# has_error=False means all arithmetic in the response is correct.
# All responses use IT-format: markdown, numbered steps, **bold**, boxed answers.

_SYNTHETIC_CORPUS: list[tuple[str, str, bool]] = [
    # === Correct responses (has_error=False) ===
    (
        "What is 15 + 27?",
        "## Solution\n\n**Step 1:** Add the ones digits: 5 + 7 = 12, carry 1.\n"
        "**Step 2:** Add the tens digits plus carry: 1 + 2 + 1 = 4.\n"
        "**Answer:** 15 + 27 = **42**",
        False,
    ),
    (
        "If a store sells 8 apples for $12, what is the price per apple?",
        "### Approach\n\n1. Total cost = $12\n2. Number of apples = 8\n"
        "3. Price per apple = 12 / 8 = **$1.50**\n\nThe answer is $1.50.",
        False,
    ),
    (
        "What is 100 - 37?",
        "**Step 1:** Borrow from tens: 10 - 7 = 3.\n"
        "**Step 2:** Tens: 9 - 3 = 6.\n"
        "**Result:** 100 - 37 = **63**",
        False,
    ),
    (
        "A car travels 60 mph for 3 hours. How far does it go?",
        "## Distance Calculation\n\n- Speed = 60 mph\n- Time = 3 hours\n"
        "- Distance = speed × time = 60 × 3 = **180 miles**",
        False,
    ),
    (
        "What is 7 × 8?",
        "### Multiplication\n\n7 × 8 = **56**\n\nThis is a basic multiplication fact.",
        False,
    ),
    (
        "Divide 144 by 12.",
        "**Solution:**\n1. 144 ÷ 12 = 12\n2. Verification: 12 × 12 = 144 ✓\n\n**Answer: 12**",
        False,
    ),
    (
        "What is 25% of 80?",
        "## Percentage Calculation\n\n25% = 0.25\n\n0.25 × 80 = **20**\n\nThe answer is 20.",
        False,
    ),
    (
        "If you have $50 and spend $23, how much remains?",
        "### Subtraction\n\n$50 - $23 = **$27**\n\nYou have $27 remaining.",
        False,
    ),
    (
        "What is 3 squared?",
        "**Step 1:** 3² = 3 × 3 = **9**\n\nThe answer is 9.",
        False,
    ),
    (
        "A rectangle has length 8 and width 5. What is the area?",
        "## Area Formula\n\nArea = length × width\nArea = 8 × 5 = **40 square units**",
        False,
    ),
    # === Erroneous responses (has_error=True) ===
    (
        "What is 13 + 29?",
        "## Solution\n\n**Step 1:** Add the ones digits: 3 + 9 = 12, carry 1.\n"
        "**Step 2:** Add tens plus carry: 1 + 2 + 1 = 4.\n"
        "**Answer:** 13 + 29 = **41**",  # Correct is 42
        True,
    ),
    (
        "If a store sells 6 apples for $9, what is the price per apple?",
        "### Approach\n\n1. Total cost = $9\n2. Number of apples = 6\n"
        "3. Price per apple = 9 / 6 = **$2.00**\n\nThe answer is $2.00.",  # Correct is $1.50
        True,
    ),
    (
        "What is 100 - 43?",
        "**Step 1:** 10 - 3 = 7.\n"
        "**Step 2:** Tens: 9 - 4 = 5.\n"
        "**Result:** 100 - 43 = **57**",  # Correct is 57 -- wait that IS correct
        False,  # Actually correct; the arithmetic checks out
    ),
    (
        "A car travels 55 mph for 4 hours. How far does it go?",
        "## Distance Calculation\n\n- Speed = 55 mph\n- Time = 4 hours\n"
        "- Distance = speed × time = 55 × 4 = **210 miles**",  # Correct is 220
        True,
    ),
    (
        "What is 9 × 7?",
        "### Multiplication\n\n9 × 7 = **65**\n\nThis is a basic multiplication fact.",  # Correct is 63
        True,
    ),
    (
        "Divide 132 by 12.",
        "**Solution:**\n1. 132 ÷ 12 = 10\n2. Verification: 10 × 12 = 120 ≠ 132\n\n**Answer: 10**",  # Correct is 11
        True,
    ),
    (
        "What is 20% of 90?",
        "## Percentage Calculation\n\n20% = 0.20\n\n0.20 × 90 = **16**\n\nThe answer is 16.",  # Correct is 18
        True,
    ),
    (
        "If you have $75 and spend $38, how much remains?",
        "### Subtraction\n\n$75 - $38 = **$47**\n\nYou have $47 remaining.",  # Correct is 37
        True,
    ),
    (
        "What is 4 squared?",
        "**Step 1:** 4² = 4 × 4 = **14**\n\nThe answer is 14.",  # Correct is 16
        True,
    ),
    (
        "A rectangle has length 9 and width 6. What is the area?",
        "## Area Formula\n\nArea = length × width\nArea = 9 × 6 = **52 square units**",  # Correct is 54
        True,
    ),
]

assert len(_SYNTHETIC_CORPUS) == N_RESPONSES, (
    f"Expected {N_RESPONSES} responses, got {len(_SYNTHETIC_CORPUS)}"
)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _run_nl2z3_benchmark(
    corpus: list[tuple[str, str, bool]],
) -> dict[str, Any]:
    """Run NL2Z3Extractor on the corpus and compute metrics.

    In CI mode (CARNOT_FORCE_LIVE not set), all results will be 'unknown'.
    Returns a metrics dict with success_rate, fp_rate, tp_rate, n_success.
    """
    extractor = NL2Z3Extractor()
    results: list[dict[str, Any]] = []

    for question, response, has_error in corpus:
        violations = extractor.extract(question, response, domain="reasoning")
        z3_status = (
            extractor.last_z3_result.sat_status
            if extractor.last_z3_result is not None
            else "unknown"
        )
        flagged = len(violations) > 0
        results.append(
            {
                "question": question[:60],
                "has_error": has_error,
                "z3_status": z3_status,
                "flagged": flagged,
            }
        )

    return _compute_metrics(results, extractor_name="NL2Z3Extractor")


def _run_llm_z3_benchmark(
    corpus: list[tuple[str, str, bool]],
    llm_caller: Any,
) -> dict[str, Any]:
    """Run LLMz3Formalizer on the corpus and compute metrics.

    When llm_caller is None, CI stub is used (all results will be 'sat').
    Returns a metrics dict with success_rate, fp_rate, tp_rate, n_success.
    """
    formalizer = LLMz3Formalizer(llm_caller=llm_caller, model_id="ci_stub")
    results: list[dict[str, Any]] = []

    for question, response, has_error in corpus:
        result = formalizer.formalize(question, response)
        flagged = result.z3_result == "unsat"
        results.append(
            {
                "question": question[:60],
                "has_error": has_error,
                "z3_status": result.z3_result,
                "formalization_mode": result.formalization_mode,
                "n_assertions": result.n_assertions,
                "flagged": flagged,
            }
        )

    return _compute_metrics(results, extractor_name="LLMz3Formalizer")


def _compute_metrics(
    results: list[dict[str, Any]],
    extractor_name: str,
) -> dict[str, Any]:
    """Compute z3_success_rate, fp_rate, tp_rate from a list of per-response results.

    Definitions:
    - z3_success_rate: fraction of responses where Z3 returned sat or unsat (not error/unknown)
    - tp (true positive): response has_error=True AND flagged=True
    - fp (false positive): response has_error=False AND flagged=True
    - fn (false negative): response has_error=True AND flagged=False
    - tn (true negative): response has_error=False AND flagged=False
    - tp_rate: tp / (tp + fn) — fraction of actual errors detected
    - fp_rate: fp / (fp + tn) — fraction of correct responses incorrectly flagged
    """
    n_total = len(results)
    n_success = sum(
        1 for r in results if r["z3_status"] in ("sat", "unsat")
    )

    n_error = sum(1 for r in results if r["has_error"])
    n_correct = n_total - n_error

    tp = sum(1 for r in results if r["has_error"] and r["flagged"])
    fp = sum(1 for r in results if not r["has_error"] and r["flagged"])
    fn = n_error - tp
    tn = n_correct - fp

    tp_rate = tp / n_error if n_error > 0 else 0.0
    fp_rate = fp / n_correct if n_correct > 0 else 0.0

    return {
        "extractor_name": extractor_name,
        "n_total": n_total,
        "n_success": n_success,
        "n_true_positives": tp,
        "n_false_positives": fp,
        "n_false_negatives": fn,
        "n_true_negatives": tn,
        "z3_success_rate": n_success / n_total if n_total > 0 else 0.0,
        "tp_rate": tp_rate,
        "fp_rate": fp_rate,
        "per_response": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Experiment 357: LLMz3Formalizer vs NL2Z3Extractor benchmark."""
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    _log.info("Experiment %d: %s", EXP_ID, TITLE)
    _log.info("Corpus: %d synthetic IT-format responses", N_RESPONSES)

    is_live = bool(os.environ.get("CARNOT_FORCE_LIVE"))
    inference_mode = "live_gpu" if is_live else "simulated"

    if is_live:
        _log.warning(
            "CARNOT_FORCE_LIVE=1 detected but LLMz3Formalizer CI stub used "
            "(no model loaded for Exp 357). Set llm_caller explicitly for live mode."
        )

    _log.info("Running NL2Z3Extractor benchmark...")
    nl2z3_metrics = _run_nl2z3_benchmark(_SYNTHETIC_CORPUS)

    _log.info("Running LLMz3Formalizer benchmark (CI stub)...")
    llm_z3_metrics = _run_llm_z3_benchmark(_SYNTHETIC_CORPUS, llm_caller=None)

    nl2z3_success = nl2z3_metrics["z3_success_rate"]
    llm_z3_success = llm_z3_metrics["z3_success_rate"]
    improvement_delta = llm_z3_success - nl2z3_success

    _log.info(
        "NL2Z3 success_rate=%.2f | LLMz3 success_rate=%.2f | delta=%.2f",
        nl2z3_success,
        llm_z3_success,
        improvement_delta,
    )
    _log.info(
        "LLMz3 fp_rate=%.2f, tp_rate=%.2f",
        llm_z3_metrics["fp_rate"],
        llm_z3_metrics["tp_rate"],
    )

    artifact = tmpl.build_result(
        {
            "schema": "carnot.llm_z3_formalizer.v1",
            "inference_mode": inference_mode,
            "n_responses": N_RESPONSES,
            "nl2z3_success_rate": nl2z3_success,
            "llm_z3_success_rate": llm_z3_success,
            "improvement_delta": improvement_delta,
            "n_true_positives": llm_z3_metrics["n_true_positives"],
            "n_false_positives": llm_z3_metrics["n_false_positives"],
            "fp_rate": llm_z3_metrics["fp_rate"],
            "tp_rate": llm_z3_metrics["tp_rate"],
            "nl2z3_metrics": {
                k: v for k, v in nl2z3_metrics.items() if k != "per_response"
            },
            "llm_z3_metrics": {
                k: v for k, v in llm_z3_metrics.items() if k != "per_response"
            },
            "note": (
                "CI stub mode: LLMz3Formalizer returns 'sat' for all responses. "
                "improvement_delta=1.0 reflects stub behavior, not live benchmark. "
                "NL2Z3Extractor returns 'unknown' for all in CI (CARNOT_FORCE_LIVE not set)."
            ) if not is_live else (
                "Live mode: LLMz3Formalizer used CI stub (no LLM loaded). "
                "For true live comparison, inject a live llm_caller."
            ),
        },
        status="success",
    )

    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(artifact, f, indent=2)

    _log.info("Artifact written to %s", out_path)
    _log.info("Experiment %d complete.", EXP_ID)


if __name__ == "__main__":
    main()
