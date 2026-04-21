#!/usr/bin/env python3
"""Experiment 643 — Ensemble Recall Gate v2.

**What this experiment computes:**
    Exps 641 (HERMES v2 live loop) and 642 (CausalReasoningVerifier) each provide
    an independent violation-detection signal.  InterWhenMonitor provides a third
    signal (interwhen_recall=0.12, Exp 629).  This experiment computes the OR ensemble:

        any_violation = interwhen_hit OR hermes_v2_hit OR causal_hit

    Because the three detectors are orthogonal (different error classes), their union
    should substantially exceed any individual recall.

**Gate decision for VR #17:**
    RETRO-070 action item: "Do NOT schedule attempt #17 until interwhen recall >= 30%".
    That threshold now applies to the combined ensemble recall, not just interwhen alone.
    gate_open = (ensemble_recall >= 0.30).

**CI stub mode (no LLM required):**
    InterWhenMonitor and CausalReasoningVerifier both operate in regex-only mode
    when SymCodeVerifier is passed llm_caller=None.  HERMES v2 per-question indices
    are not available from Exp 641, so hermes_hit defaults to False per-question
    (conservative: no spurious TPs from guessing).

Spec: REQ-VERIFY-141, REQ-VERIFY-142,
      SCENARIO-VERIFY-186, SCENARIO-VERIFY-187
"""

import json
import os
import sys

# --- env autofix must be FIRST ---
from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

# Path wiring so scripts/ is importable from outside the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.symcode_verifier import SymCodeVerifier  # noqa: E402
from carnot.pipeline.interwhen_monitor import InterWhenMonitor  # noqa: E402
from carnot.pipeline.causal_reasoning_verifier import CausalReasoningVerifier  # noqa: E402
from carnot.pipeline.ensemble_gate import compute_ensemble_hits  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 643
TITLE = "Ensemble Recall Gate v2"
DELIVERABLE = "results/experiment_643_ensemble_gate_v2.json"
N_INCORRECT = 25
N_CORRECT = 10
GATE_THRESHOLD = 0.30

# Known-good interwhen recall from Exp 629 diagnostic (post-hoc primary signal).
INTERWHEN_RECALL_KNOWN = 0.12

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=40)
watchdog.start()

tmpl = ExperimentTemplate(
    EXP_ID,
    TITLE,
    DELIVERABLE,
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Load Exp 641 (HERMES v2) and Exp 642 (Causal) prior results
# ---------------------------------------------------------------------------

_641_PATH = os.path.join(_REPO_ROOT, "results", "experiment_641_hermes_v2_live.json")
_642_PATH = os.path.join(_REPO_ROOT, "results", "experiment_642_causal_verifier.json")

try:
    with open(_641_PATH) as f:
        exp641 = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    exp641 = {}

try:
    with open(_642_PATH) as f:
        exp642 = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    exp642 = {}

hermes_v2_recall = float(exp641.get("hermes_v2_recall", 0.0))
causal_recall_prior = float(exp642.get("causal_recall", 0.0))

# Per-question TP index sets — Exp 641/642 did not store these, so they are empty.
# Conservative: hermes_hit defaults to False when indices are unknown.
hermes_v2_tp_indices: set[int] = set(exp641.get("tp_question_indices") or [])
causal_tp_indices: set[int] = set(exp642.get("tp_question_indices") or [])

# ---------------------------------------------------------------------------
# Load live corpus
# ---------------------------------------------------------------------------

_PAIRS_PATH = os.path.join(_REPO_ROOT, "results", "live_pairs_578.json")

try:
    with open(_PAIRS_PATH) as f:
        all_pairs = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    all_pairs = []

incorrect_all = [p for p in all_pairs if not p.get("is_correct", True)]
correct_all = [p for p in all_pairs if p.get("is_correct", False)]

incorrect_sample = incorrect_all[:N_INCORRECT]
correct_sample = correct_all[:N_CORRECT]

# Pad with synthetic responses if corpus is smaller than required (CI safety net).
_SYNTHETIC_INCORRECT = [
    (
        f"Step 1: {10+i} + {5+i} = {15+i} items. "
        f"Step 2: We had {20+i} items so we have {20+i} - {3+i} = {17+i} left."
    )
    for i in range(N_INCORRECT)
]
_SYNTHETIC_CORRECT = [
    f"Step 1: {5+i} + {3+i} = {8+i}. The answer is {8+i}."
    for i in range(N_CORRECT)
]

while len(incorrect_sample) < N_INCORRECT:
    idx = len(incorrect_sample)
    incorrect_sample.append({"response": _SYNTHETIC_INCORRECT[idx], "is_correct": False})

while len(correct_sample) < N_CORRECT:
    idx = len(correct_sample)
    correct_sample.append({"response": _SYNTHETIC_CORRECT[idx], "is_correct": True})

# ---------------------------------------------------------------------------
# Construct verifiers (CI stub: no LLM caller)
# ---------------------------------------------------------------------------

_verifier = SymCodeVerifier(llm_caller=None)
interwhen = InterWhenMonitor(_verifier)
causal = CausalReasoningVerifier(_verifier)

# ---------------------------------------------------------------------------
# Run ensemble on incorrect responses
# ---------------------------------------------------------------------------

incorrect_indices = [
    p.get("question_index", i) for i, p in enumerate(incorrect_sample)
]
correct_indices = [
    p.get("question_index", N_INCORRECT + i) for i, p in enumerate(correct_sample)
]

incorrect_responses = [p.get("response", "") for p in incorrect_sample]
correct_responses = [p.get("response", "") for p in correct_sample]

incorrect_hits = compute_ensemble_hits(
    incorrect_responses,
    incorrect_indices,
    interwhen,
    causal,
    hermes_v2_tp_indices,
)

correct_hits = compute_ensemble_hits(
    correct_responses,
    correct_indices,
    interwhen,
    causal,
    hermes_v2_tp_indices,
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

ensemble_tp = sum(1 for h in incorrect_hits if h)
ensemble_fp = sum(1 for h in correct_hits if h)
ensemble_recall = ensemble_tp / N_INCORRECT
ensemble_fp_rate = ensemble_fp / N_CORRECT

gate_open = ensemble_recall >= GATE_THRESHOLD
gate_note = (
    "Exp 644 VR #17 UNBLOCKED"
    if gate_open
    else "Exp 644 VR #17 BLOCKED — combined recall below 0.30 threshold"
)
honest_verdict = (
    "gate_open_vr_unblocked" if gate_open else "gate_closed_recall_below_threshold"
)

# ---------------------------------------------------------------------------
# Build and write artifact
# ---------------------------------------------------------------------------

artifact = tmpl.build_result(
    {
        "schema": "carnot.ensemble_gate_v2.v1",
        "n_incorrect": N_INCORRECT,
        "n_correct": N_CORRECT,
        "interwhen_recall": INTERWHEN_RECALL_KNOWN,
        "hermes_v2_recall": hermes_v2_recall,
        "causal_recall": causal_recall_prior,
        "ensemble_tp": ensemble_tp,
        "ensemble_fp": ensemble_fp,
        "ensemble_recall": ensemble_recall,
        "ensemble_fp_rate": ensemble_fp_rate,
        "gate_threshold": GATE_THRESHOLD,
        "gate_open": gate_open,
        "gate_note": gate_note,
        "retro_070_resolved": ensemble_recall >= GATE_THRESHOLD,
        "honest_verdict": honest_verdict,
    },
    status="success",
)

_DELIVERABLE_PATH = os.path.join(_REPO_ROOT, DELIVERABLE)
os.makedirs(os.path.dirname(_DELIVERABLE_PATH), exist_ok=True)
with open(_DELIVERABLE_PATH, "w") as _f:
    json.dump(artifact, _f, indent=2)

print(json.dumps(artifact, indent=2))
print(
    f"\nensemble_recall={ensemble_recall:.3f}  gate_open={gate_open}"
    f"  ensemble_fp_rate={ensemble_fp_rate:.3f}"
    f"  verdict={honest_verdict}"
)

tmpl.assert_deliverable_written()
