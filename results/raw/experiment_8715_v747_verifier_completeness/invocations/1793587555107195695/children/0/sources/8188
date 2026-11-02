#!/usr/bin/env python3
"""Experiment 755: PSV Relapse Root-Cause Diagnosis via Three-Hypothesis Controlled Test.

**What this experiment does:**
    The PSV (self-play verify-repair) loop relapsed for the third consecutive milestone
    (Exp 753 retro: fp_rate_slope_new30=+0.00110). Three prior recovery attempts (Exps 697,
    737) each achieved temporary improvement that subsequently reversed. This pattern
    indicates an architecture problem, not a hyperparameter problem.

    This experiment runs a controlled diagnostic to identify which of three competing
    architectural hypotheses best explains the relapse:
      A (SRSA memory contamination): incorrect repairs corrupt the constraint signal
      B (PPSEBM coupling overwrite): self-play perturbs CD-learned coupling matrix
      C (Curriculum collapse): question diversity exhausted, overfitting to narrow pool

    Each hypothesis is tested in isolation with exactly ONE independent variable changed
    vs. the control. The primary hypothesis routes Exp 756 to the correct architectural fix.

**Data:** 100 synthetic GSM8K-style questions, CPU-only (no GPU, no real LLM required).
**Runtime:** < 5 seconds on any CPU.

Spec: REQ-PSV-013, SCENARIO-PSV-020
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add repo root to path so we can import both scripts/ and python/carnot/
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "python"))

from scripts.experiment_template import ExperimentTemplate
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.psv_diagnostic import PSVDiagnostic

DELIVERABLE = "results/experiment_755_psv_relapse_diagnosis.json"

# ---------------------------------------------------------------------------
# Synthetic GSM8K-style question generator
# ---------------------------------------------------------------------------


def _make_synthetic_questions(n: int = 100) -> list[str]:
    """Generate n synthetic arithmetic word problems to fill the PSV question pool.

    WHY synthetic instead of real GSM8K: this experiment is a CPU-only diagnostic
    run with no LLM. The question pool is used only as a size/diversity parameter
    for the simulation — the actual question text does not affect the fp_rate
    measurement (the simulation uses constraint_quality dynamics, not LLM outputs).

    Args:
        n: Number of questions to generate.

    Returns:
        List of synthetic question strings.
    """
    questions: list[str] = []
    for i in range(n):
        a, b, c = (i * 7 + 3) % 50, (i * 13 + 7) % 30, (i * 11 + 2) % 20
        questions.append(
            f"If a store has {a + 10} apples and sells {b + 1} per day, "
            f"how many remain after {c + 1} days? "
            f"Show your arithmetic step by step."
        )
    return questions


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=755,
        title="PSV Relapse Root-Cause Diagnosis (Three Hypotheses)",
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()
    tmpl.check_exclusion_manifest()

    with ExperimentTimeoutWatchdog(
        experiment_id=755,
        timeout_minutes=60,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        questions = _make_synthetic_questions(n=100)

        diagnostic = PSVDiagnostic(
            n_trials=len(questions),
            seed=42,
            n_steps=30,
        )

        result = diagnostic.diagnose()

        # Extract slopes for top-level artifact fields (per task spec)
        hyp_a_slope = result.evidence_dict["hypothesis_a"]["slope"]
        hyp_b_slope = result.evidence_dict["hypothesis_b"]["slope"]
        hyp_c_slope = result.evidence_dict["hypothesis_c"]["slope"]

        # Map primary_hypothesis to honest_verdict per task contract
        honest_verdict = result.primary_hypothesis

        artifact = tmpl.build_result(
            {
                "hypothesis_a_slope": hyp_a_slope,
                "hypothesis_b_slope": hyp_b_slope,
                "hypothesis_c_slope": hyp_c_slope,
                "hypothesis_a_confirmed": result.hypothesis_a_confirmed,
                "hypothesis_b_confirmed": result.hypothesis_b_confirmed,
                "hypothesis_c_confirmed": result.hypothesis_c_confirmed,
                "primary_hypothesis": result.primary_hypothesis,
                "evidence_dict": result.evidence_dict,
                "honest_verdict": honest_verdict,
                "n_questions": len(questions),
                "inference_mode": "cpu_synthetic",
            },
            status="success",
            decision_class="diagnose" if False else "detect",
        )

        # decision_class must be from DECISION_CLASSES — use "detect" as closest
        # (we are DETECTING the root cause of PSV relapse)
        output_path = _REPO_ROOT / DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
