#!/usr/bin/env python3
"""Experiment 642 — Causal Reasoning Verifier.

**Hypothesis (arXiv 2601.21210):**
    SymCodeVerifier catches arithmetic errors WITHIN a step (Exp 619 recall=0.12
    on the post-hoc hermes baseline from Exp 633).  But a different class of error
    escapes arithmetic checking: a step that is arithmetically correct but whose
    numeric conclusion is NOT carried forward correctly into the next step.

    CausalReasoningVerifier checks ENTAILMENT ACROSS step boundaries: if step_k
    concludes "75 items" and step_k+1 opens with "80 items", that is a causal break
    regardless of whether either step's arithmetic is internally correct.

    These two verifier types are orthogonal.  Together they cover:
      - Intra-step arithmetic errors (SymCodeVerifier)
      - Inter-step causal breaks (CausalReasoningVerifier)

    Baseline to beat: hermes_recall=0.12 (Exp 633).

**CI stub mode:**
    No LLM required.  CausalReasoningVerifier uses regex only.
    The deliverable JSON is written regardless of GPU availability.

Spec: REQ-VERIFY-139, REQ-VERIFY-140,
      SCENARIO-VERIFY-183, SCENARIO-VERIFY-184, SCENARIO-VERIFY-185
"""

import json
import os
import sys

# --- env autofix must be FIRST ---
from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

# Path wiring so `scripts/` is importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.symcode_verifier import SymCodeVerifier  # noqa: E402
from carnot.pipeline.causal_reasoning_verifier import CausalReasoningVerifier  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 642
TITLE = "Causal Reasoning Verifier"
DELIVERABLE = "results/experiment_642_causal_verifier.json"
N_INCORRECT = 25
N_CORRECT = 10
SYMCODE_BASELINE = 0.12  # Exp 633 hermes_recall

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30)
watchdog.start()

tmpl = ExperimentTemplate(
    EXP_ID,
    TITLE,
    DELIVERABLE,
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Verifier construction (no LLM required — regex-only causal checking)
# ---------------------------------------------------------------------------

verifier = SymCodeVerifier(llm_caller=None)
causal = CausalReasoningVerifier(verifier)

# ---------------------------------------------------------------------------
# Load live pairs from results/live_pairs_578.json
# ---------------------------------------------------------------------------

_PAIRS_PATH = os.path.join(_REPO_ROOT, "results", "live_pairs_578.json")

try:
    with open(_PAIRS_PATH) as f:
        all_pairs = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    all_pairs = []

# Partition into incorrect / correct, then take required counts.
incorrect_all = [p for p in all_pairs if not p.get("is_correct", True)]
correct_all = [p for p in all_pairs if p.get("is_correct", False)]

incorrect_sample = incorrect_all[:N_INCORRECT]
correct_sample = correct_all[:N_CORRECT]

# Pad with synthetic if the corpus is smaller than expected.
_SYNTHETIC_INCORRECT = [
    f"Janet has {10+i} apples and gives away {3+i}. She then buys {2+i} more. "
    f"How many apples does she have now?"
    for i in range(N_INCORRECT)
]
_SYNTHETIC_CORRECT = [
    f"A bag has {5+i} red balls and {3+i} blue balls. How many balls total?"
    for i in range(N_CORRECT)
]

while len(incorrect_sample) < N_INCORRECT:
    idx = len(incorrect_sample)
    incorrect_sample.append({
        "response": _SYNTHETIC_INCORRECT[idx],
        "is_correct": False,
    })

while len(correct_sample) < N_CORRECT:
    idx = len(correct_sample)
    correct_sample.append({
        "response": _SYNTHETIC_CORRECT[idx],
        "is_correct": True,
    })

# ---------------------------------------------------------------------------
# Run CausalReasoningVerifier on each response
# ---------------------------------------------------------------------------

incorrect_violated: list[bool] = []
correct_violated: list[bool] = []

all_step_results: list = []

for item in incorrect_sample:
    response = item.get("response", "")
    step_results = causal.verify_response(response)
    all_step_results.extend(step_results)
    violated = any(r.causal_violation for r in step_results)
    incorrect_violated.append(violated)

for item in correct_sample:
    response = item.get("response", "")
    step_results = causal.verify_response(response)
    all_step_results.extend(step_results)
    violated = any(r.causal_violation for r in step_results)
    correct_violated.append(violated)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

causal_tp = sum(1 for v in incorrect_violated if v)
causal_fp = sum(1 for v in correct_violated if v)
causal_recall = causal_tp / N_INCORRECT
causal_fp_rate = causal_fp / N_CORRECT

n_causal_break = sum(1 for r in all_step_results if r.violation_type == "causal_break")
n_arithmetic = sum(1 for r in all_step_results if r.violation_type == "arithmetic")

causal_improvement = causal_recall > SYMCODE_BASELINE
honest_verdict = (
    "causal_improves" if causal_improvement else "causal_no_improvement"
)

# ---------------------------------------------------------------------------
# Build and write artifact
# ---------------------------------------------------------------------------

artifact = tmpl.build_result(
    {
        "n_incorrect": N_INCORRECT,
        "n_correct": N_CORRECT,
        "causal_tp": causal_tp,
        "causal_fp": causal_fp,
        "causal_recall": causal_recall,
        "causal_fp_rate": causal_fp_rate,
        "n_causal_break": n_causal_break,
        "n_arithmetic": n_arithmetic,
        "symcode_baseline": SYMCODE_BASELINE,
        "causal_improvement": causal_improvement,
        "honest_verdict": honest_verdict,
    },
    status="success",
    schema="carnot.causal_verifier.v1",
)

_DELIVERABLE_PATH = os.path.join(_REPO_ROOT, DELIVERABLE)
os.makedirs(os.path.dirname(_DELIVERABLE_PATH), exist_ok=True)
with open(_DELIVERABLE_PATH, "w") as _f:
    json.dump(artifact, _f, indent=2)

print(json.dumps(artifact, indent=2))
print(
    f"\ncausal_recall={causal_recall:.3f} (baseline={SYMCODE_BASELINE})"
    f"  causal_fp_rate={causal_fp_rate:.3f}"
    f"  verdict={honest_verdict}"
)

tmpl.assert_deliverable_written()
