#!/usr/bin/env python3
"""Experiment 564: CoACEExtractor Implementation — execution-based arithmetic violation detection.

**Researcher summary (RETRO-061 partial closure):**
    Exp 554 found that VeriCoT and VPRM both produce 0 true positives on 25 known-incorrect
    IT model responses.  Root cause: both check FORMAT (regex patterns, Z3 UNSAT) but IT
    models produce correct-format, wrong-arithmetic outputs.  Example: '47 + 28 = 76' passes
    Z3 and VPRM patterns — but 47+28=75, not 76.

    CoACE (Caco, arXiv 2510.04081): parse arithmetic equations from prose, eval() the LHS,
    compare to stated RHS.  eval('47+28')=75 != 76 → violation.  CPU-only, no GPU needed.

    This experiment validates CoACEExtractor on 20 hardcoded test cases (10 incorrect, 10 correct)
    and reports TP rate, FP rate, and whether RETRO-061 partial closure is achieved (TP > 0).

**Gate chain:**
    1. apply_env_autofix()
    2. ExperimentTimeoutWatchdog(564, timeout_minutes=20)
    3. ExperimentTemplate(564, ..., requires_gpu=False)
    4. Run CoACEExtractor on 20 test cases
    5. Compute TP/FP rates
    6. Write artifact
    7. tmpl.assert_deliverable_written()

Spec: REQ-EXTRACT-033, REQ-EXTRACT-034,
      SCENARIO-EXTRACT-061, SCENARIO-EXTRACT-062, SCENARIO-EXTRACT-063, SCENARIO-EXTRACT-064
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_watchdog = ExperimentTimeoutWatchdog(564, timeout_minutes=20)
_watchdog.start()

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

tmpl = ExperimentTemplate(
    exp_id=564,
    title="CoACEExtractor Implementation",
    deliverable="results/experiment_564_coace_extractor.json",
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Import CoACEExtractor after setup
# ---------------------------------------------------------------------------

from carnot.extraction.coace_extractor import CoACEExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# 20 hardcoded test cases
# ---------------------------------------------------------------------------

# 10 INCORRECT cases — each contains at least one arithmetic error.
# These are representative of the IT model prose CoT format from Exp 554.
INCORRECT_CASES = [
    {
        "id": "ic_01",
        "text": "We add 47 and 28 to get 76. So the sum is 76.",
        "note": "47+28=75, not 76",
    },
    {
        "id": "ic_02",
        "text": "Multiplying 15 by 4 gives us 55.",
        "note": "15*4=60, not 55",
    },
    {
        "id": "ic_03",
        "text": "We compute 100 / 5 = 25. The answer is 25.",
        "note": "100/5=20, not 25",
    },
    {
        "id": "ic_04",
        "text": "First, 13 + 9 = 21 items total.",
        "note": "13+9=22, not 21",
    },
    {
        "id": "ic_05",
        "text": "The subtraction gives 50 - 17 = 34.",
        "note": "50-17=33, not 34",
    },
    {
        "id": "ic_06",
        "text": "Step 3: 8 * 7 = 65",
        "note": "8*7=56, not 65",
    },
    {
        "id": "ic_07",
        "text": "Adding the costs: 120 + 85 = 204",
        "note": "120+85=205, not 204",
    },
    {
        "id": "ic_08",
        "text": "We divide 144 by 12 to get 13.",
        "note": "144/12=12, not 13",
    },
    {
        "id": "ic_09",
        "text": "The result is 25 + 36 = 60.",
        "note": "25+36=61, not 60",
    },
    {
        "id": "ic_10",
        "text": "So 99 - 44 = 56.",
        "note": "99-44=55, not 56",
    },
]

# 10 CORRECT cases — arithmetic is exact.
CORRECT_CASES = [
    {
        "id": "cc_01",
        "text": "We add 47 and 28 to get 75. So the sum is 75.",
        "note": "47+28=75 correct",
    },
    {
        "id": "cc_02",
        "text": "Multiplying 15 by 4 gives us 60.",
        "note": "15*4=60 correct",
    },
    {
        "id": "cc_03",
        "text": "We compute 100 / 5 = 20. The answer is 20.",
        "note": "100/5=20 correct",
    },
    {
        "id": "cc_04",
        "text": "13 + 9 = 22 items total.",
        "note": "13+9=22 correct",
    },
    {
        "id": "cc_05",
        "text": "The subtraction gives 50 - 17 = 33.",
        "note": "50-17=33 correct",
    },
    {
        "id": "cc_06",
        "text": "Step 3: 8 * 7 = 56",
        "note": "8*7=56 correct",
    },
    {
        "id": "cc_07",
        "text": "Adding the costs: 120 + 85 = 205",
        "note": "120+85=205 correct",
    },
    {
        "id": "cc_08",
        "text": "We divide 144 by 12 to get 12.",
        "note": "144/12=12 correct",
    },
    {
        "id": "cc_09",
        "text": "The result is 25 + 36 = 61.",
        "note": "25+36=61 correct",
    },
    {
        "id": "cc_10",
        "text": "So 99 - 44 = 55.",
        "note": "99-44=55 correct",
    },
]

# ---------------------------------------------------------------------------
# Run CoACEExtractor on all 20 cases
# ---------------------------------------------------------------------------

extractor = CoACEExtractor(tolerance=1e-6, min_confidence=0.5)

tp = 0  # violations found on incorrect cases (true positives)
fp = 0  # violations found on correct cases (false positives)

per_case_results = []

for case in INCORRECT_CASES:
    result = extractor.extract(case["text"])
    flagged = result.n_violations > 0
    if flagged:
        tp += 1
    per_case_results.append(
        {
            "id": case["id"],
            "label": "incorrect",
            "flagged": flagged,
            "n_violations": result.n_violations,
            "n_equations_found": result.n_equations_found,
            "note": case["note"],
        }
    )

for case in CORRECT_CASES:
    result = extractor.extract(case["text"])
    flagged = result.n_violations > 0
    if flagged:
        fp += 1
    per_case_results.append(
        {
            "id": case["id"],
            "label": "correct",
            "flagged": flagged,
            "n_violations": result.n_violations,
            "n_equations_found": result.n_equations_found,
            "note": case["note"],
        }
    )

tp_rate = tp / 10
fp_rate = fp / 10

if tp_rate > 0.5:
    honest_verdict = "extraction_viable"
elif tp_rate > 0:
    honest_verdict = "extraction_partial"
else:
    honest_verdict = "extraction_zero"

retro_061_partial = tp_rate > 0

# ---------------------------------------------------------------------------
# Write artifact
# ---------------------------------------------------------------------------

artifact = tmpl.build_result(
    {
        "schema": "carnot.coace_extractor.v1",
        "n_test_cases": 20,
        "n_incorrect_cases": 10,
        "n_correct_cases": 10,
        "tp": tp,
        "fp": fp,
        "tp_rate": tp_rate,
        "fp_rate": fp_rate,
        "retro_061_partial": retro_061_partial,
        "honest_verdict": honest_verdict,
        "per_case_results": per_case_results,
    },
    status="success",
    decision_class="detect",
)

tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
tmpl._output_path.write_text(json.dumps(artifact, indent=2))

print(f"Exp 564: tp_rate={tp_rate:.2f}, fp_rate={fp_rate:.2f}, verdict={honest_verdict}")
print(f"RETRO-061 partial closure: {retro_061_partial}")
print(f"Artifact: {tmpl._output_path}")

tmpl.assert_deliverable_written()
