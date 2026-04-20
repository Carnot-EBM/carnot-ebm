#!/usr/bin/env python3
"""Experiment 574: Milestone 2026.04.43 Operational Retrospective.

**Researcher summary:**
    Milestone 2026.04.43 ran Exps 563-574 under the theme
    "Root Cause Surgery — Execution-Based Extraction and PURE JEPA Recovery".

    Three pre-existing RETROs were the surgery targets:
      RETRO-062 (Live 50q A never collected) — Exp 563 re-attempted.  BLOCKED
        again: CARNOT_FORCE_LIVE was not set before session startup.
        n_pairs_collected=0.  Root cause persists.
      RETRO-061 (Extraction TP rate = 0) — Exp 564 shipped CoACEExtractor
        (execution-based, not format-based).  Exp 565 confirmed it achieves
        TP rate = 0.059 on 25 live responses. CLOSED: first non-zero TP rate
        in the extraction pipeline.  Precision is low (0.33), recall is low
        (0.06), but the gate is open.
      RETRO-060 (JEPA AUC below random) — Exps 566 (PURE margin) + 567
        (JEPA v10 retrain with pure_min_form loss).  Result: v10_auc=0.4444,
        still below the 0.5 random baseline.  PURE objective was insufficient;
        objective function change alone did not fix the anti-correlation.
        STILL BLOCKED.

    Supporting experiments:
      Exp 568 (KV260 bring-up v2): synthesis still required; fpga_alive=False.
      Exp 569 (live verify-repair with CoACE): signed_improvement=0.0 —
        pipeline accuracy unchanged (26%→26%).  Low recall means violations
        found but repaired correctly only 1/7 times.  No accuracy lift.
      Exp 570 (FR-11 real violations relay): fr11_real_violations_confirmed=True.
        The self-learning relay is processing real violations from CoACE.
      Exp 571 (HalluField Tier 0e): AUC = 0.974 on synthetic corpus.  Viable.
      Exp 572 (PRA EORM beam search): beam_improvement=0.50 on mock EORM.  Viable.
      Exp 573 (energy-per-token calibration): RAPL unavailable on this machine;
        calibration_viable=False.

    Headline: ONE RETRO CLOSED (RETRO-061 — extraction no longer silently blocked),
    but end-to-end accuracy improvement remains zero.  JEPA and live data collection
    still need architectural work.

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
from scripts.experiment_template import ExperimentTemplate, _utc_now

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 574
TITLE = "Milestone 2026.04.43 Retrospective"
DELIVERABLE = "results/experiment_574_retro_2026_04_43.json"
MILESTONE = "2026.04.43"
SCHEMA = "carnot.operational_retro.v18"

# .42 milestone baseline (from experiment_562_retro_2026_04_42.json) for comparison.
_PRIOR_MILESTONE_WALL_TIME_MIN = round(
    (0.0 + 0.005 + 0.005 + 0.0 + 175.0 + 144.0 + 63.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0) / 60,
    3,
)  # approximate .42 total (Exps 549-561)

# All 11 upstream experiment result files for milestone .43.
# Exp 574 (this retro) is #12 and is computed here, not loaded.
_MILESTONE_RESULTS = [
    ("563", "results/experiment_563_live_data_a_v2.json"),
    ("564", "results/experiment_564_coace_extractor.json"),
    ("565", "results/experiment_565_coace_live_diagnostic.json"),
    ("566", "results/experiment_566_jepa_pure_margin.json"),
    ("567", "results/experiment_567_jepa_v10_retrain.json"),
    ("568", "results/experiment_568_kv260_bringup_v2.json"),
    ("569", "results/experiment_569_live_vr_coace.json"),
    ("570", "results/experiment_570_fr11_real_violations.json"),
    ("571", "results/experiment_571_hallufield_tier0e.json"),
    ("572", "results/experiment_572_pra_eorm_beam_search.json"),
    ("573", "results/experiment_573_energy_per_token_calibration.json"),
]

# RETROs open at the START of milestone .43 (carry-forward from .42 retro).
# Used to compute retro_closure_rate: closed_this_milestone / open_at_start.
_RETROS_OPEN_AT_MILESTONE_START = [
    "RETRO-031",  # partial carry — closure status unverified
    "RETRO-033",  # live 25q precision — 10+ attempts, still not closed
    "RETRO-038",  # live 100q VeriCoT — 8+ attempts, still not closed
    "RETRO-049",  # NUP Probe v4 contrastive margin loss redesign
    "RETRO-056",  # JEPA AUC below random
    "RETRO-057",  # LowRankKAEM energy accuracy outside tolerance
    "RETRO-060",  # JEPA architecturally anti-correlated — objective redesign needed
    "RETRO-061",  # Extraction TP rate = 0
    "RETRO-062",  # Live 50q A unrun — questions 0-49 missing
]


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def _load_result(path_str: str) -> dict:
    """Load a JSON experiment result, returning empty dict if absent or corrupt.

    Producing a retro artifact even when an upstream experiment is missing allows
    the conductor to record a partial milestone rather than crashing the retrospective.
    Missing experiments increment n_missing rather than stopping the retro.
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
    """Load all .43 milestone results and compute aggregate retro metrics.

    Returns a dict with every v18 schema field, ready to pass to build_result().

    The v18 schema adds five new milestone-specific boolean success criteria
    (retro_062_resolved, retro_061_resolved, retro_060_resolved, fpga_alive,
    live_vr_positive, fr11_real_violations) compared to v17.
    """
    results = {exp_id: _load_result(path) for exp_id, path in _MILESTONE_RESULTS}

    # --- Per-experiment status counts ----------------------------------------
    # n_experiments includes this retro script (574) as the 12th experiment.
    n_experiments = len(_MILESTONE_RESULTS) + 1  # +1 for this retro
    n_completed = (
        sum(1 for r in results.values() if r.get("status") == "success") + 1
    )  # +1 for 574 itself
    n_timed_out = sum(1 for r in results.values() if r.get("status") == "timed_out")
    n_deferred_to_gpu = sum(
        1
        for r in results.values()
        if r.get("status") == "blocked" and r.get("inference_mode") == "gpu_required"
    )
    n_missing = sum(1 for r in results.values() if not r)

    # --- Wall time -----------------------------------------------------------
    # Sum duration_s for all upstream experiments.  574 (this retro) is near-zero
    # and is excluded to avoid a self-referential timing loop.
    total_wall_time_seconds = sum(r.get("duration_s", 0.0) for r in results.values())
    total_wall_time_minutes = round(total_wall_time_seconds / 60.0, 3)
    mean_time_min = round(total_wall_time_minutes / n_experiments, 3)

    # --- Success criteria evaluation -----------------------------------------

    # RETRO-062: Live 50q A collected >= 40 pairs (Exp 563).
    # Result: n_pairs_collected=0 because CARNOT_FORCE_LIVE was not set — still blocked.
    retro_062_resolved: bool = int(results["563"].get("n_pairs_collected", 0)) >= 40

    # RETRO-061: CoACEExtractor achieves TP rate > 0 on live IT-model outputs (Exp 565).
    # Result: coace_tp_rate=0.0588 — first non-zero TP rate in the extraction pipeline.
    retro_061_resolved: bool = bool(results["565"].get("retro_061_resolved", False))

    # RETRO-060: JEPA v10 with PURE objective achieves AUC > 0.5 (Exp 567).
    # Result: v10_auc=0.4444 — still inverted below random baseline.
    retro_060_resolved: bool = bool(results["567"].get("retro_060_resolved", False))

    # FPGA alive: KV260 hardware latency < 100 μs (Exp 568).
    # Result: synthesis not complete; hardware_latency_us=null; fpga_alive=False.
    _hw_latency = results["568"].get("hardware_latency_us")
    fpga_alive: bool = (
        _hw_latency is not None and float(_hw_latency) < 100.0
    )

    # Live verify-repair positive: signed_improvement > 0 (Exp 569).
    # Result: signed_improvement=0.0 — pipeline accuracy unchanged at 26%.
    live_vr_positive: bool = float(results["569"].get("signed_improvement", 0.0)) > 0.0

    # FR-11 real violations confirmed (Exp 570).
    # Result: fr11_real_violations_confirmed=True — relay processing real violations.
    fr11_real_violations: bool = bool(
        results["570"].get("fr11_real_violations_confirmed", False)
    )

    # --- RETRO closure rate --------------------------------------------------
    # RETRO-061 closed (coace_tp_rate > 0 confirmed by Exp 565).
    # RETRO-060: still blocked (v10_auc < 0.5).
    # RETRO-062: still blocked (n_pairs_collected = 0).
    n_closed_this_milestone = sum([
        retro_061_resolved,   # RETRO-061 closed
        # All others remain open
    ])
    retro_closure_rate = round(
        n_closed_this_milestone / len(_RETROS_OPEN_AT_MILESTONE_START), 3
    )

    # --- Honest verdict ------------------------------------------------------
    # root_cause_fixed requires BOTH retro_061 AND retro_060 resolved.
    # partial_fix requires at least one resolved.
    # both_still_blocked if neither.
    if retro_061_resolved and retro_060_resolved:
        honest_verdict = "root_cause_fixed"
    elif retro_061_resolved or retro_060_resolved:
        honest_verdict = "partial_fix"
    else:
        honest_verdict = "both_still_blocked"

    # --- New RETRO items opening this milestone -------------------------------
    new_retro_items = [
        {
            "id": "RETRO-063",
            "title": "JEPA v10 still inverted despite PURE objective — architectural redesign insufficient",
            "opened_milestone": MILESTONE,
            "carry_count": 0,
            "description": (
                "Exp 567 retrained JEPA v10 with the pure_min_form loss (PURE objective), "
                "the architectural change recommended after RETRO-060 root-cause analysis. "
                "Result: v10_auc=0.4444, still below the 0.5 random baseline. "
                "PURE objective alone is insufficient — the model still learns to invert the "
                "correctness signal. The predictor may need a fundamentally different "
                "architecture (contrastive energy margin with explicit positive/negative pair "
                "construction), or the training corpus needs quality filtering before the "
                "next retrain attempt."
            ),
            "priority": "critical",
        },
        {
            "id": "RETRO-064",
            "title": "Live verify-repair accuracy unchanged despite CoACE violations — low recall bottleneck",
            "opened_milestone": MILESTONE,
            "carry_count": 0,
            "description": (
                "Exp 569 ran the full verify-repair pipeline with CoACEExtractor over 50 "
                "GSM8K questions.  CoACE found 7 violations and applied 7 repairs, but "
                "signed_improvement=0.0 (accuracy 26%→26%).  Only 1/7 repairs improved the "
                "answer.  Root cause: coace_recall=0.059 (1 of 17 incorrect responses flagged). "
                "At this recall level the pipeline's expected accuracy lift is near-zero even "
                "with perfect repair.  CoACE recall must exceed ~30% before pipeline accuracy "
                "improvement is detectable over 50 questions."
            ),
            "priority": "high",
        },
        {
            "id": "RETRO-065",
            "title": "RAPL energy unavailable — hardware energy calibration blocked",
            "opened_milestone": MILESTONE,
            "carry_count": 0,
            "description": (
                "Exp 573 attempted to calibrate EORM energy scores against real hardware "
                "energy readings via RAPL (Running Average Power Limit), the Linux kernel "
                "interface for CPU/DRAM energy measurement.  Result: rapl_available=False — "
                "the /sys/class/powercap/intel-rapl path does not exist on the current machine "
                "(likely AMD CPU or RAPL not exposed in kernel).  calibration_viable=False. "
                "Hardware energy calibration requires either a machine with Intel RAPL support, "
                "an AMD equivalent (AMD Energy driver), or an external power meter."
            ),
            "priority": "medium",
        },
    ]

    # --- Open RETRO items carry-forward to .44 --------------------------------
    # All retros open at .43 start, minus RETRO-061 (now closed), plus new items.
    open_retro_items = [
        {
            "id": "RETRO-031",
            "title": "Partial carry — closure status unverified",
            "carry_count": ">=3",
            "action_required": "Verify closure in result files before .44 planning",
        },
        {
            "id": "RETRO-033",
            "title": "Live 25q precision benchmark — 10+ attempts, still not closed",
            "carry_count": ">=10",
            "action_required": (
                "CoACE recall=5.9% — cannot drive accuracy improvement. "
                "Block until RETRO-064 (recall improvement) is addressed."
            ),
        },
        {
            "id": "RETRO-038",
            "title": "Live 100q VeriCoT+VPRM — 8+ attempts, still not closed",
            "carry_count": ">=8",
            "action_required": "Same root cause as RETRO-033. Block until recall > 30%.",
        },
        {
            "id": "RETRO-049",
            "title": "NUP Probe v4 contrastive margin loss redesign",
            "carry_count": ">=2",
            "action_required": "Confirm closure status via Exp 530 result file inspection",
        },
        {
            "id": "RETRO-056",
            "title": "JEPA AUC below random — objective redesign insufficient (see RETRO-063)",
            "carry_count": 2,
            "action_required": "Superseded by RETRO-063 for tracking. Architectural change needed.",
        },
        {
            "id": "RETRO-057",
            "title": "LowRankKAEM energy accuracy outside 5% tolerance",
            "carry_count": 1,
            "action_required": "Architectural redesign needed. Calibration layer approach exhausted.",
        },
        {
            "id": "RETRO-060",
            "title": "JEPA architecturally anti-correlated — PURE objective did not fix it",
            "carry_count": 1,
            "action_required": "Superseded by RETRO-063 with deeper diagnosis. See RETRO-063.",
        },
        {
            "id": "RETRO-062",
            "title": "Live 50q A unrun — questions 0-49 missing from FOVER corpus",
            "carry_count": 1,
            "action_required": "Run as first experiment of .44 with pre-flight GPU gate check",
        },
        {
            "id": "RETRO-063",
            "title": "JEPA v10 inverted despite PURE objective",
            "carry_count": 0,
            "action_required": "Contrastive energy margin loss or quality-filtered corpus required",
        },
        {
            "id": "RETRO-064",
            "title": "CoACE recall 5.9% — pipeline accuracy improvement undetectable",
            "carry_count": 0,
            "action_required": "Improve CoACE recall to >30% before scheduling accuracy benchmarks",
        },
        {
            "id": "RETRO-065",
            "title": "RAPL unavailable — hardware energy calibration blocked",
            "carry_count": 0,
            "action_required": "Need Intel RAPL or AMD Energy driver on test machine",
        },
    ]

    # --- Top priorities for milestone .44 ------------------------------------
    top_priorities_for_44 = [
        (
            "Improve CoACE recall from 5.9% to >30% (RETRO-064). "
            "At current recall the pipeline accuracy lift is undetectable over 50 questions. "
            "Approach: expand the CoACEExtractor execution harness to cover more arithmetic "
            "patterns; add symbolic re-execution for multi-step chains. "
            "This is the single highest-leverage action — it unblocks RETRO-033, RETRO-038, "
            "and live verify-repair accuracy improvement."
        ),
        (
            "Redesign JEPA contrastive objective with explicit positive/negative pair "
            "construction (RETRO-063). PURE objective failed on two retrains. "
            "The contrastive energy margin approach pairs a correct CoT trace against an "
            "incorrect one and trains the predictor to assign lower energy to the correct trace. "
            "Do not schedule retrain attempt #4 until the objective function is replaced."
        ),
        (
            "Run Live 50q A as the FIRST experiment of .44 with CARNOT_FORCE_LIVE=1 pre-flight "
            "gate (RETRO-062). Two milestones in a row have blocked this collection. "
            "Add a hard abort in the conductor session startup if CARNOT_FORCE_LIVE is not set "
            "before any experiment with inference_mode=gpu_required."
        ),
    ]

    return {
        "schema": SCHEMA,
        "milestone": MILESTONE,
        "title": TITLE,
        # Aggregate counts
        "n_experiments": n_experiments,
        "n_completed": n_completed,
        "n_timed_out": n_timed_out,
        "n_deferred_to_gpu": n_deferred_to_gpu,
        "n_missing": n_missing,
        # Wall time
        "total_wall_time_minutes": total_wall_time_minutes,
        "mean_time_min": mean_time_min,
        # Success criteria
        "retro_062_resolved": retro_062_resolved,
        "retro_061_resolved": retro_061_resolved,
        "retro_060_resolved": retro_060_resolved,
        "fpga_alive": fpga_alive,
        "live_vr_positive": live_vr_positive,
        "fr11_real_violations": fr11_real_violations,
        # RETRO closure rate
        "retro_closure_rate": retro_closure_rate,
        # Narrative
        "new_retro_items": new_retro_items,
        "open_retro_items": open_retro_items,
        "top_priorities_for_44": top_priorities_for_44,
        # Verdict
        "honest_verdict": honest_verdict,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Build and write the Exp 574 milestone retro artifact."""
    # Watchdog: abort if the retro script hangs (pure JSON computation, should
    # complete in <5 s; 30-minute limit is generous and defensive only).
    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30)

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
    print(f"[Exp {EXP_ID}] retro_061_resolved={artifact['retro_061_resolved']}")
    print(f"[Exp {EXP_ID}] retro_060_resolved={artifact['retro_060_resolved']}")
    print(f"[Exp {EXP_ID}] retro_062_resolved={artifact['retro_062_resolved']}")
    print(f"[Exp {EXP_ID}] fpga_alive={artifact['fpga_alive']}")
    print(f"[Exp {EXP_ID}] live_vr_positive={artifact['live_vr_positive']}")
    print(f"[Exp {EXP_ID}] fr11_real_violations={artifact['fr11_real_violations']}")
    print(f"[Exp {EXP_ID}] retro_closure_rate={artifact['retro_closure_rate']}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
