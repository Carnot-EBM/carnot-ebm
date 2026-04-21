#!/usr/bin/env python3
"""Experiment 655 — Ensemble Recall Gate v3.

**What this experiment computes:**
    Gate v3 integrates four independent recall signals and decides whether
    to authorise VR live attempt #18 (RETRO-033).

    Weighted ensemble formula:
        ensemble_recall = (
            0.3 * symcode_recall      (SymCodeVerifier, Exp 619/630)
          + 0.4 * structured_recall   (HermesV2StructuredLoop, Exp 654)
          + 0.3 * causal_recall       (CausalReasoningVerifier, Exp 642)
        )
    Gate opens when ensemble_recall >= 0.30.

    hermes_v2_recall (Exp 641) is loaded for traceability but excluded from
    the ensemble weights — it has been 0.0 in every live run to date.

**Signal sources (prior experiments):**
    - symcode_recall:     results/experiment_619_dsvd_symcode.json  (0.12)
    - hermes_v2_recall:   results/experiment_641_hermes_v2_live.json (0.0)
    - structured_recall:  results/experiment_654_hermes_v2_structured.json (0.2)
    - causal_recall:      results/experiment_642_causal_verifier.json (0.36)

**Gate formula (v3):**
    ensemble_recall = 0.3*symcode + 0.4*structured + 0.3*causal
    gate_open = ensemble_recall >= 0.30

**No GPU required.**  All signals are loaded from prior experiment JSON files.

Spec: REQ-VERIFY-149, SCENARIO-VERIFY-200, SCENARIO-VERIFY-201
"""

import json
import os
import sys

# --- env autofix must be FIRST ---
from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.ensemble_gate_v3 import EnsembleRecallGateV3  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 655
TITLE = "Ensemble Recall Gate v3"
DELIVERABLE = "results/experiment_655_ensemble_gate_v3.json"
GATE_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=20)
watchdog.start()

tmpl = ExperimentTemplate(
    EXP_ID,
    TITLE,
    DELIVERABLE,
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Load recall signals from prior experiments
# ---------------------------------------------------------------------------


def _load_json(path: str) -> dict:
    """Load a JSON result file relative to repo root; return empty dict if absent."""
    abs_path = os.path.join(_REPO_ROOT, path)
    if not os.path.exists(abs_path):
        return {}
    with open(abs_path) as fh:
        return json.load(fh)


# symcode_recall: try Exp 619 first (DSVD symcode), fall back to Exp 630 (symcode v2)
_symcode_data = _load_json("results/experiment_619_dsvd_symcode.json")
if not _symcode_data:
    _symcode_data = _load_json("results/experiment_630_symcode_v2.json")
symcode_recall: float = float(
    _symcode_data.get("symcode_recall", _symcode_data.get("auc", 0.12))
)

# hermes_v2_recall: tracked for provenance but not included in ensemble weight
_hermes_data = _load_json("results/experiment_641_hermes_v2_live.json")
hermes_v2_recall: float = float(_hermes_data.get("hermes_v2_recall", 0.12))

# structured_recall: from HermesV2StructuredLoop (Exp 654)
_structured_data = _load_json("results/experiment_654_hermes_v2_structured.json")
structured_recall: float = float(
    _structured_data.get("hermes_v2_structured_recall", 0.0)
)

# causal_recall: from CausalReasoningVerifier (Exp 642)
_causal_data = _load_json("results/experiment_642_causal_verifier.json")
causal_recall: float = float(_causal_data.get("causal_recall", 0.12))

print(f"[655] symcode_recall={symcode_recall:.3f}")
print(f"[655] hermes_v2_recall={hermes_v2_recall:.3f} (tracked, not in ensemble)")
print(f"[655] structured_recall={structured_recall:.3f}")
print(f"[655] causal_recall={causal_recall:.3f}")

# ---------------------------------------------------------------------------
# Compute gate v3
# ---------------------------------------------------------------------------

gate = EnsembleRecallGateV3()
gate_result = gate.compute(
    symcode_recall=symcode_recall,
    hermes_v2_recall=hermes_v2_recall,
    structured_recall=structured_recall,
    causal_recall=causal_recall,
)

print(f"[655] ensemble_recall={gate_result.ensemble_recall:.4f}")
print(f"[655] gate_open={gate_result.gate_open}")

# Identify which signal is contributing least (gate_blocker when gate is closed)
signal_values = {
    "symcode": symcode_recall,
    "structured": structured_recall,
    "causal": causal_recall,
}
gate_blocker = min(signal_values, key=lambda k: signal_values[k])

if not gate_result.gate_open:
    print(
        f"[655] GATE CLOSED — lowest signal is '{gate_blocker}' "
        f"({signal_values[gate_blocker]:.3f}). "
        f"ensemble_recall={gate_result.ensemble_recall:.4f} < {GATE_THRESHOLD}"
    )
else:
    print(
        f"[655] GATE OPEN — VR attempt #18 authorised. "
        f"ensemble_recall={gate_result.ensemble_recall:.4f} >= {GATE_THRESHOLD}"
    )

# ---------------------------------------------------------------------------
# Build and write result artifact
# ---------------------------------------------------------------------------

honest_verdict = (
    "gate_open_vr18_authorized" if gate_result.gate_open else "gate_closed_vr18_blocked"
)

artifact = tmpl.build_result(
    {
        "schema": "carnot.ensemble_gate_v3.v1",
        "symcode_recall": gate_result.symcode_recall,
        "hermes_v2_recall": gate_result.hermes_v2_recall,
        "structured_recall": gate_result.structured_recall,
        "causal_recall": gate_result.causal_recall,
        "ensemble_recall": gate_result.ensemble_recall,
        "gate_open": gate_result.gate_open,
        "gate_threshold": GATE_THRESHOLD,
        "gate_version": gate_result.gate_version,
        "gate_blocker": gate_blocker if not gate_result.gate_open else None,
        "retro_033_attempt_18_authorized": gate_result.gate_open,
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
    f"\nensemble_recall={gate_result.ensemble_recall:.4f}  gate_open={gate_result.gate_open}"
    f"  verdict={honest_verdict}"
)

watchdog.stop()
tmpl.assert_deliverable_written()
