#!/usr/bin/env python3
"""Experiment 600: Milestone 2026.04.45 Operational Retrospective.

**Researcher summary:**
    Milestone 2026.04.45 ran Exps 589-599 under the theme
    "Recall Surgery Follow-Up: CoACE v3, DSVD Live Validation, and JEPA v12 Confirmation".

    Five pre-existing RETROs were the surgery targets for this milestone:

    RETRO-033 (Live 25q verify-repair precision — 13+ attempts) — Exp 594 (CoACEV3)
        and Exp 595 (DSVD) both blocked because gate_open=False.  The live recall
        threshold has not been crossed.  retro_033_resolved=False.  STILL BLOCKED.

    RETRO-038 (Live 200q Wilson CI — 9+ attempts) — Exp 596 blocked (gate closed).
        retro_038_resolved=False.  STILL BLOCKED.

    RETRO-066 (CoACE offline/live distribution gap) — Exp 591 measured
        v3_recall=0.04 on live data, which is LOWER than v2_recall=0.059.
        The v3 model did not fix the offline/live gap.  retro_066_resolved=False.
        STILL BLOCKED.

    RETRO-063 (JEPA v10/v11 overfitting risk) — Exp 593 retrained JEPA v12 on
        the full 100-pair corpus from Exp 578.  v12_val_auc=1.0, retro_063_validated=True.
        The AUC holds on a larger corpus, confirming the v11 result was not 9-pair
        overfitting.  VALIDATED (closure confirmed, not a new close).

    RETRO-067 (ExclusionManifest not wired) — Exp 589 wired the manifest into
        the conductor pick_next_task() path (npu_iron_available=False but the
        conductor exclusion check was the primary goal).

    Supporting experiments:
        Exp 590 (Live assertion): status=success, pipeline assertion passed.
        Exp 592 (DSVD live validation): dsvd_live_auc=0.586 < 0.80 threshold; gate_open=False.
        Exp 597 (FR-11 real violations v4): fr11_real_violations_confirmed=False (gate closed).
        Exp 598 (HISR + D-Wave): hisr_credit_assignment_correct=True, dwave_available=True.
            D-Wave Cloud access confirmed.  HISR credit assignment logic validated.
        Exp 599 (Vivado + GRPO NUP): vivado_status=not_installed, bitfile_built=None;
            nup_v5_auc=0.739 (improved from prior but below 0.80).

    Headline: ZERO RETROs CLOSED this milestone.  JEPA v12 overfitting was DISPROVEN
    (positive result), D-Wave Cloud access was CONFIRMED, but end-to-end accuracy
    improvement remains zero.  CoACE v3 recall on live data (4%) is WORSE than v2 (5.9%).
    The offline/live gap is the single critical unresolved bottleneck.

Spec: REQ-INFRA-058, REQ-INFRA-076, SCENARIO-INFRA-069, SCENARIO-INFRA-075
"""

from __future__ import annotations

# apply_env_autofix MUST be called first, before any other carnot import.
# This ensures CARNOT_FORCE_LIVE and related env vars are set correctly
# before any pipeline code reads them at import time.
from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate, _utc_now  # noqa: F401

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 600
TITLE = "Milestone 2026.04.45 Retrospective"
DELIVERABLE = "results/experiment_600_retro_2026_04_45.json"
MILESTONE = "2026.04.45"
SCHEMA = "carnot.operational_retro.v20"
PRIOR_MILESTONE_WALL_TIME_MINUTES = 4654.0  # from results/operational_retro_2026_04_44.json

# All 11 upstream experiment result files for milestone .45.
# Exp 600 (this retro) is #12 and is computed here, not loaded.
_MILESTONE_RESULTS = [
    ("589", "results/experiment_589_exclusion_manifest_wire_in.json"),
    ("590", "results/experiment_590_live_assertion.json"),
    ("591", "results/experiment_591_coace_v3_live.json"),
    ("592", "results/experiment_592_dsvd_live_val.json"),
    ("593", "results/experiment_593_jepa_v12_retrain.json"),
    ("594", "results/experiment_594_live_vr_coace_v3.json"),
    ("595", "results/experiment_595_live_vr_dsvd.json"),
    ("596", "results/experiment_596_live_200q_wilson.json"),
    ("597", "results/experiment_597_fr11_real_violations_v4.json"),
    ("598", "results/experiment_598_hisr_dwave.json"),
    ("599", "results/experiment_599_vivado_grpo_nup.json"),
]

# RETROs open at the START of milestone .45 (carry-forward from .44 retro).
# Used to compute retro_closure_rate: closed_this_milestone / open_at_start.
_RETROS_OPEN_AT_MILESTONE_START = [
    "RETRO-031",  # partial carry — closure status unverified
    "RETRO-033",  # live 25q verify-repair precision — 13+ attempts, still not closed
    "RETRO-038",  # live 200q VeriCoT+Wilson CI — 9+ attempts, still not closed
    "RETRO-049",  # NUP Probe v4 contrastive margin loss redesign
    "RETRO-057",  # LowRankKAEM energy accuracy outside tolerance
    "RETRO-060",  # JEPA architecturally anti-correlated (superseded by 063, monitoring)
    "RETRO-064",  # CoACE recall 5.9% on live data — pipeline accuracy lift undetectable
    "RETRO-065",  # RAPL unavailable — hardware energy calibration blocked
    "RETRO-066",  # CoACE offline/live distribution gap (opened in .44)
    "RETRO-067",  # ExclusionManifest built but conductor not wired (opened in .44)
]


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def _load_result(path_str: str) -> dict:
    """Load a JSON experiment result, returning empty dict if absent or corrupt.

    Producing a retro artifact even when an upstream experiment is missing allows
    the conductor to record a partial milestone rather than crashing the retrospective.
    Missing experiments increment n_not_run rather than stopping the retro.
    """
    path = _REPO_ROOT / path_str
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_retro() -> dict:
    """Load all .45 milestone results and compute aggregate retro metrics.

    Returns a dict with every v20 schema field, ready to pass to build_result().

    The v20 schema tracks eight boolean success criteria specific to milestone .45:
    retro_033_resolved, retro_038_resolved, retro_066_resolved, retro_063_validated,
    fr11_improved, dsvd_live_validated, npu_unblocked, fpga_progress.
    """
    results = {exp_id: _load_result(path) for exp_id, path in _MILESTONE_RESULTS}

    # --- Per-experiment status counts ----------------------------------------
    # n_experiments includes this retro script (600) as the 12th experiment.
    n_experiments = len(_MILESTONE_RESULTS) + 1  # +1 for this retro
    n_missing = sum(1 for r in results.values() if not r)
    n_experiments_run = n_experiments - n_missing
    n_not_run = n_missing

    # --- Wall time -----------------------------------------------------------
    # Sum duration_s for all upstream experiments.  600 (this retro) is near-zero
    # and is excluded to avoid a self-referential timing loop.
    total_wall_time_seconds = sum(r.get("duration_s", 0.0) for r in results.values())
    total_wall_time_minutes = round(total_wall_time_seconds / 60.0, 3)
    mean_time_min = round(total_wall_time_minutes / n_experiments, 3)

    # Wall-time comparison to prior milestone (.44 = 4654 min).
    # This milestone is dramatically faster because most experiments were blocked
    # (near-zero duration) rather than running full GPU inference.
    wall_time_vs_prior_delta_minutes = round(
        total_wall_time_minutes - PRIOR_MILESTONE_WALL_TIME_MINUTES, 3
    )

    # --- Success criteria evaluation -----------------------------------------

    # RETRO-033 (13th+ attempt): Live VR with CoACEV3 or DSVD shows signed_improvement > 0
    # Result: Both Exp 594 and Exp 595 blocked (gate_open=False), signed_improvement=0.
    retro_033_resolved: bool = any(
        r.get("retro_033_resolved", False) for r in [results["594"], results["595"]]
    )

    # RETRO-038 (9th+ attempt): Live 200q Wilson CI lower bound > 0.
    # Result: Exp 596 blocked (gate closed), wilson_lower_ci=null.
    retro_038_resolved: bool = bool(results["596"].get("retro_038_resolved", False))

    # RETRO-066: CoACE v3 achieves recall >= 0.20 on live outputs (Exp 591).
    # Result: v3_recall=0.04 — gate_open=False.  Even worse than v2's 0.059.
    # The new v3 architecture did NOT fix the offline/live distribution gap.
    retro_066_resolved: bool = bool(results["591"].get("retro_066_resolved", False))

    # RETRO-063 validation: JEPA v12 retrained on full 100-pair corpus achieves AUC > 0.5.
    # This validates that v11's AUC=1.0 was not 9-pair overfitting.
    # Result: v12_val_auc=1.0, retro_063_validated=True.  CONFIRMED non-overfitting.
    retro_063_validated: bool = bool(results["593"].get("retro_063_validated", False))

    # FR-11 real violations: Exp 597 confirmed real FR-11 violations exist.
    # Result: fr11_real_violations_confirmed=False (gate closed, no violations processed).
    fr11_improved: bool = bool(results["597"].get("fr11_real_violations_confirmed", False))

    # DSVD live validation: Exp 592 dsvd_live_auc >= 0.80 threshold.
    # Result: dsvd_live_auc=0.586 < 0.80.  DSVD does not meet live deployment threshold.
    dsvd_live_validated: bool = float(results["592"].get("dsvd_live_auc", 0.0)) >= 0.80

    # NPU unblocked: Exp 589 reports npu_iron_available or npu_ninja_available.
    # Result: both False — NPU hardware is still not present on this machine.
    npu_unblocked: bool = bool(results["589"].get("npu_iron_available", False)) or bool(
        results["589"].get("npu_ninja_available", False)
    )

    # FPGA progress: Exp 599 bitfile_built=True.
    # Result: bitfile_built=None (Vivado not installed).
    fpga_progress: bool = bool(results["599"].get("bitfile_built", False))

    # D-Wave Cloud confirmed (positive infrastructure result from Exp 598).
    # This is a new positive result: quantum annealer accessible for future experiments.
    dwave_available: bool = bool(results["598"].get("dwave_available", False))

    # HISR credit assignment correct (positive algorithmic result from Exp 598).
    hisr_credit_correct: bool = bool(results["598"].get("hisr_credit_assignment_correct", False))

    # --- RETRO closure rate --------------------------------------------------
    # RETROs closed this milestone: NONE.
    # retro_063_validated is a VALIDATION of a previously-closed RETRO, not a new close.
    # retro_067 (ExclusionManifest wiring) has partial progress via Exp 589, but the
    # conductor is not fully wired — not closed.
    n_closed_this_milestone = 0
    retro_closure_rate = round(
        n_closed_this_milestone / len(_RETROS_OPEN_AT_MILESTONE_START), 3
    )
    # +2 new RETROs opening this milestone (RETRO-068, RETRO-069)
    open_retro_count = len(_RETROS_OPEN_AT_MILESTONE_START) - n_closed_this_milestone + 2

    # --- Honest verdict ------------------------------------------------------
    # Determined by the highest-level success signal available.
    if retro_033_resolved:
        honest_verdict = "first_positive_achieved"
    elif retro_063_validated and dwave_available:
        honest_verdict = "infrastructure_progress_no_accuracy_gain"
    elif retro_066_resolved:
        honest_verdict = "recall_fixed_no_positive"
    else:
        honest_verdict = "recall_still_blocked_all_retros_open"

    # --- New RETRO items opening this milestone -------------------------------
    new_retro_items = [
        {
            "id": "RETRO-068",
            "title": "CoACE v3 recall 4% on live data — WORSE than v2's 5.9%",
            "opened_milestone": MILESTONE,
            "carry_count": 0,
            "description": (
                "Exp 591 measured CoACEV3 on 25 live production responses from Qwen3.5-0.8B "
                "and google/gemma-4-E4B-it.  Result: v3_recall=0.04 — gate_open=False.  "
                "This is LOWER than v2_recall=0.059 (Exp 581).  The v3 architecture was "
                "redesigned to close the offline/live gap (RETRO-066), but the change made "
                "things worse.  The live error distribution is fundamentally different from "
                "the offline training corpus (injected numeric errors vs. actual model "
                "reasoning failures).  The CoACE extractor must be trained on real live "
                "model output pairs, not synthetic errors.  Until this is done, all "
                "verify-repair and FR-11 experiments remain blocked."
            ),
            "priority": "critical",
        },
        {
            "id": "RETRO-069",
            "title": "DSVD live AUC 0.586 below 0.80 deployment threshold",
            "opened_milestone": MILESTONE,
            "carry_count": 0,
            "description": (
                "Exp 592 validated DSVD on live model outputs and measured dsvd_live_auc=0.586, "
                "which is below the minimum 0.80 threshold for deployment as a Tier-2.5 "
                "replacement for CoACE.  The offline result was dsvd_auc=0.976 (Exp 587), "
                "showing the same offline/live transfer failure pattern as CoACE.  "
                "DSVD was proposed as a fallback if CoACE remained blocked, but it faces "
                "the same root-cause distribution mismatch.  Both models require retraining "
                "on live corpus pairs before either can gate the verify-repair pipeline."
            ),
            "priority": "high",
        },
    ]

    # --- Open RETRO items carry-forward to .46 --------------------------------
    open_retro_items = [
        {
            "id": "RETRO-031",
            "title": "Partial carry — closure status unverified",
            "carry_count": ">=4",
            "action_required": "Verify closure in result files before .46 planning",
        },
        {
            "id": "RETRO-033",
            "title": "Live 25q verify-repair precision — 13+ attempts, still not closed",
            "carry_count": ">=13",
            "action_required": (
                "Blocked: gate_open=False (CoACE v3 recall=4%, DSVD live AUC=0.586).  "
                "Do not schedule attempt #14 until RETRO-066, RETRO-068, and RETRO-069 "
                "are resolved.  The recall threshold must be crossed before scheduling VR."
            ),
        },
        {
            "id": "RETRO-038",
            "title": "Live 200q VeriCoT+Wilson CI — 9+ attempts, still not closed",
            "carry_count": ">=9",
            "action_required": "Same root cause as RETRO-033. Block until recall > 30%.",
        },
        {
            "id": "RETRO-049",
            "title": "NUP Probe v4 contrastive margin loss redesign",
            "carry_count": ">=3",
            "action_required": (
                "nup_v5_auc=0.739 (Exp 599) — improved but below 0.80 threshold.  "
                "Schedule NUP v6 retrain on larger live corpus."
            ),
        },
        {
            "id": "RETRO-057",
            "title": "LowRankKAEM energy accuracy outside 5% tolerance",
            "carry_count": ">=3",
            "action_required": "Architectural redesign needed. Calibration layer approach exhausted.",
        },
        {
            "id": "RETRO-060",
            "title": "JEPA architecturally anti-correlated — superseded by RETRO-063",
            "carry_count": ">=3",
            "action_required": (
                "RETRO-063 validated by Exp 593: v12_val_auc=1.0 on full 100-pair corpus.  "
                "JEPA v12 is architecturally sound.  Close RETRO-060 tracking."
            ),
        },
        {
            "id": "RETRO-064",
            "title": "CoACE recall 5.9% on live data — pipeline accuracy lift undetectable",
            "carry_count": 2,
            "action_required": (
                "v3_recall=0.04 is WORSE than v2's 0.059 (RETRO-068).  "
                "Build live-corpus training set from Exp 578 pairs before any further retrain."
            ),
        },
        {
            "id": "RETRO-065",
            "title": "RAPL unavailable — hardware energy calibration blocked",
            "carry_count": 2,
            "action_required": "Need Intel RAPL or AMD Energy driver on test machine",
        },
        {
            "id": "RETRO-066",
            "title": "CoACE offline/live distribution gap — v3 made it worse",
            "carry_count": 1,
            "action_required": (
                "v3 recall (4%) is lower than v2 (5.9%).  Architecture changes alone cannot "
                "fix training distribution mismatch.  Must collect live (model_A, model_B) "
                "output pairs and retrain on those.  Exp 578 has 100 live pairs — use them."
            ),
        },
        {
            "id": "RETRO-067",
            "title": "ExclusionManifest not fully wired into conductor",
            "carry_count": 1,
            "action_required": (
                "Exp 589 made partial progress.  Verify conductor_consulted=True in next "
                "exclusion manifest wire-in run.  Expected savings: ~385 min/milestone."
            ),
        },
        {
            "id": "RETRO-068",
            "title": "CoACE v3 recall 4% on live data — WORSE than v2",
            "carry_count": 0,
            "action_required": (
                "Retrain CoACE on live model output pairs (Exp 578 corpus).  "
                "Do not release v4 until live recall >= 20% is confirmed before scheduling VR."
            ),
        },
        {
            "id": "RETRO-069",
            "title": "DSVD live AUC 0.586 below deployment threshold",
            "carry_count": 0,
            "action_required": (
                "Retrain DSVD on live corpus pairs and re-evaluate.  "
                "Gate DSVD deployment on live AUC >= 0.80."
            ),
        },
    ]

    # --- Top priorities for milestone .46 ------------------------------------
    top_priorities_for_46 = [
        (
            "Build live-corpus training set and retrain CoACE v4 on real model outputs "
            "(RETRO-066, RETRO-068).  Exp 578 collected 100 live pairs from Qwen3.5-0.8B "
            "and gemma-4-E4B-it.  Retrain CoACEV4 on these pairs rather than the synthetic "
            "GSM8K corpus.  Target: live recall >= 20% before scheduling any VR experiment.  "
            "This is the single highest-leverage action — it unblocks RETRO-033, RETRO-038, "
            "RETRO-064, RETRO-066, RETRO-068, RETRO-069, and the entire live verify-repair chain."
        ),
        (
            "Retrain DSVD on live corpus pairs to close the offline/live AUC gap (RETRO-069).  "
            "dsvd_live_auc=0.586 vs offline=0.976 is a 40-point gap — the same root cause as "
            "CoACE.  Use the same Exp 578 live corpus.  Both CoACEV4 and DSVD retrains can share "
            "the same data pipeline.  DSVD is the Tier-2.5 fallback if CoACEV4 does not improve; "
            "both must be retrained together.  Target: dsvd_live_auc >= 0.80."
        ),
        (
            "Leverage D-Wave Cloud access (Exp 598, dwave_available=True) for HISR-Dwave "
            "integration on a real verify-repair task.  The credit assignment logic is now "
            "validated (hisr_credit_assignment_correct=True).  This is the first infrastructure "
            "experiment that produced a genuine positive result this milestone.  Running a "
            "real HISR-DWave VR experiment would validate the quantum annealing path and "
            "provide a new signal for EBM energy minimisation beyond gradient descent."
        ),
    ]

    return {
        "schema": SCHEMA,
        "milestone": MILESTONE,
        "title": TITLE,
        # Aggregate counts
        "n_experiments_run": n_experiments_run,
        "n_not_run": n_not_run,
        # Wall time
        "total_wall_time_minutes": total_wall_time_minutes,
        "mean_time_min": mean_time_min,
        "wall_time_vs_prior_delta_minutes": wall_time_vs_prior_delta_minutes,
        # Success criteria
        "retro_033_resolved": retro_033_resolved,
        "retro_038_resolved": retro_038_resolved,
        "retro_066_resolved": retro_066_resolved,
        "retro_063_validated": retro_063_validated,
        "fr11_improved": fr11_improved,
        "dsvd_live_validated": dsvd_live_validated,
        "npu_unblocked": npu_unblocked,
        "fpga_progress": fpga_progress,
        # Bonus positive results
        "dwave_available": dwave_available,
        "hisr_credit_correct": hisr_credit_correct,
        # RETRO closure rate
        "retro_closure_rate": retro_closure_rate,
        "open_retro_count": open_retro_count,
        # Narrative
        "new_retro_items": new_retro_items,
        "open_retro_items": open_retro_items,
        "top_priorities_for_46": top_priorities_for_46,
        # Verdict
        "honest_verdict": honest_verdict,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Build and write the Exp 600 milestone retro artifact."""
    # Watchdog: abort if the retro script hangs (pure JSON computation, should
    # complete in <5 s; 30-minute limit is generous and defensive only).
    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30)  # noqa: F841

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    retro_data = compute_retro()

    artifact = tmpl.build_result(retro_data, status="success")
    # build_result() overwrites "schema" with a sorted key list.  Restore the
    # named schema identifier after the call so downstream consumers can assert
    # the correct schema version string.
    artifact["schema"] = SCHEMA
    artifact["env_autofix"] = True  # applied at module top via apply_env_autofix()

    deliverable_path = _REPO_ROOT / DELIVERABLE
    deliverable_path.write_text(json.dumps(artifact, indent=2))

    print(f"[Exp {EXP_ID}] Deliverable written: {DELIVERABLE}")
    print(f"[Exp {EXP_ID}] honest_verdict={artifact['honest_verdict']}")
    print(f"[Exp {EXP_ID}] retro_033_resolved={artifact['retro_033_resolved']}")
    print(f"[Exp {EXP_ID}] retro_038_resolved={artifact['retro_038_resolved']}")
    print(f"[Exp {EXP_ID}] retro_066_resolved={artifact['retro_066_resolved']}")
    print(f"[Exp {EXP_ID}] retro_063_validated={artifact['retro_063_validated']}")
    print(f"[Exp {EXP_ID}] dsvd_live_validated={artifact['dsvd_live_validated']}")
    print(f"[Exp {EXP_ID}] dwave_available={artifact['dwave_available']}")
    print(f"[Exp {EXP_ID}] fpga_progress={artifact['fpga_progress']}")
    print(f"[Exp {EXP_ID}] retro_closure_rate={artifact['retro_closure_rate']}")
    print(f"[Exp {EXP_ID}] total_wall_time_minutes={artifact['total_wall_time_minutes']}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
