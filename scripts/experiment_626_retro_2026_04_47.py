#!/usr/bin/env python3
"""Experiment 626: Milestone 2026.04.47 Operational Retrospective.

**Researcher summary:**
    Milestone 2026.04.47 ran Exps 614-625 under themes:
    "LLMAsExtractorV1, JEPA v13 CAPO Calibration, SymCode-DSVD, VR Attempt #15,
    MetaJuLS Adaptation, NUP v6 Cascade, TRUST Agents, KV260 Vivado v2, FR-11 Relay".

    Key questions answered:

    RETRO-070 (LLMAsExtractorV1 recall >= 20%): Exp 616 measured v1_recall=0.04 —
        gate_open=False.  retro_070_resolved=False.  STILL BLOCKED.  Architecture
        review required: LLMAsExtractorV1 in isolation cannot close the live/offline gap.

    RETRO-033 (15th attempt): Exp 620 status=blocked, gate_open=False,
        signed_improvement=0.0.  retro_033_resolved=False.  STILL BLOCKED.
        15 consecutive zero-positive VR attempts.  The gate (extractor recall < 20%)
        blocks scheduling attempt #16.

    RETRO-069 (SymCode-DSVD AUC >= 0.50): Exp 619 measured symcode_live_auc=0.804.
        retro_069_resolved=True.  CLOSED.  SymCode-DSVD beats the prior DSVD
        fine-tune (0.158) and far exceeds the 0.50 threshold.

    JEPA v13 calibration (ECE < 0.10): Exp 618 measured v13_ece=0.207 and
        calibration_improved=False.  jepa_v13_calibrated=False.  NOT MET.

    NUP v6 deployed in cascade: Exp 622 nup_v6_wired=True.  DONE.
        cascade_latency_ms=1.27 — well within real-time budget.

    FR-11 real violations: Exp 625 fr11_real_violations_confirmed=False.
        Gate still closed on live real violations.

    MetaJuLS adaptation: Exp 621 adaptation_effective=True.  Positive signal.

    Best extractor: Exp 617 timed out, best_extractor unknown from that run.
        Exp 623 TRUST Agents showed best_extractor='llm_v1' with trust_recall=0.0.

    DualGPU: Exp 614 gpu1_utilization_confirmed=False.  Six consecutive
        milestones without confirmation.  RETRO-071 opened.

    Headline: ONE RETRO CLOSED (RETRO-069 — SymCode-DSVD beats 0.50 AUC).
    NUP v6 cascade deployed. Two critical blockers remain: RETRO-033 (15 failed
    attempts) and RETRO-070 (extractor recall 4%). Architecture-level redesign
    is now mandatory — implement interwhen mid-generation monitor + ORACLE
    data elicitation for FOVER v5 training corpus.

Spec: REQ-INFRA-058, REQ-INFRA-076, SCENARIO-INFRA-069, SCENARIO-INFRA-075
"""

from __future__ import annotations

# apply_env_autofix MUST be called first, before any other carnot import.
# It injects CARNOT_FORCE_LIVE and related env vars so all downstream pipeline
# code reads them correctly at import time, not after.
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

EXP_ID = 626
TITLE = "Milestone 2026.04.47 Retrospective"
DELIVERABLE = "results/experiment_626_retro_2026_04_47.json"
MILESTONE = "2026.04.47"
SCHEMA = "carnot.operational_retro.v22"

# Prior milestone (.46) estimated wall time in minutes — from .46 retro (Exp 613).
# The experiment result showed total_wall_time_minutes=16.677 for the .46 experiment
# scripts themselves, but the conductor wall-clock comparison target from the task
# description is ~570 min (calendar time for the full milestone run).
PRIOR_MILESTONE_WALL_TIME_MINUTES = 570.0

# All 12 upstream experiment result files for milestone .47.
# Exp 626 (this retro) is #13 and is computed here, not loaded.
_MILESTONE_RESULTS = [
    ("614", "results/experiment_614_exclusion_manifest_dualgpu.json"),
    ("615", "results/experiment_615_live_corpus_v3.json"),
    ("616", "results/experiment_616_llm_extractor_v1.json"),
    ("617", "results/experiment_617_extractor_diagnostic_v5.json"),
    ("618", "results/experiment_618_jepa_v13_capo.json"),
    ("619", "results/experiment_619_dsvd_symcode.json"),
    ("620", "results/experiment_620_live_vr_attempt_15.json"),
    ("621", "results/experiment_621_metajuls_adaptation.json"),
    ("622", "results/experiment_622_nup_v6_cascade.json"),
    ("623", "results/experiment_623_trust_agents.json"),
    ("624", "results/experiment_624_kv260_vivado_v2.json"),
    ("625", "results/experiment_625_tier1_fr11_relay.json"),
]

# RETROs open at the START of milestone .47 (carry-forward from .46 retro, Exp 613).
# Used to compute retro_closure_rate: closed_this_milestone / open_at_start.
# Source: Exp 613 open_retro_items contained these 11 items.
_RETROS_OPEN_AT_MILESTONE_START = [
    "RETRO-031",
    "RETRO-033",
    "RETRO-038",
    "RETRO-057",
    "RETRO-060",
    "RETRO-064",
    "RETRO-065",
    "RETRO-066",
    "RETRO-068",
    "RETRO-069",
    "RETRO-070",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_result(path: str) -> dict:
    """Load a JSON experiment result file relative to _REPO_ROOT.

    Why this exists: each milestone retrospective must load many sibling
    experiment result files.  Missing files are treated as 'not run' rather
    than crashing the retro, because a partially-complete milestone is still
    worth summarising.

    Returns an empty dict if the file is missing or contains invalid JSON.
    """
    full = _REPO_ROOT / path
    if not full.exists():
        return {}
    try:
        return json.loads(full.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_retro() -> dict:
    """Load all .47 milestone results and compute aggregate retro metrics.

    Returns a dict with every v22 schema field, ready to pass to build_result().

    The v22 schema tracks nine boolean success criteria specific to milestone .47:
    retro_033_resolved, retro_069_resolved, retro_070_resolved,
    nup_v6_deployed, fr11_confirmed, jepa_v13_calibrated,
    adaptation_effective, dualgpu_confirmed, and best_extractor (string).
    """
    results = {exp_id: _load_result(path) for exp_id, path in _MILESTONE_RESULTS}

    # --- Per-experiment status counts -----------------------------------------
    # n_experiments includes this retro script (626) as the 13th experiment.
    n_experiments = len(_MILESTONE_RESULTS) + 1  # +1 for this retro
    n_missing = sum(1 for r in results.values() if not r)
    n_experiments_run = n_experiments - n_missing
    n_not_run = n_missing

    # --- Wall time ------------------------------------------------------------
    # Sum duration_s for all upstream experiments.  626 (this retro) is near-zero
    # and is excluded to avoid a self-referential timing loop.
    total_wall_time_seconds = sum(r.get("duration_s", 0.0) for r in results.values())
    total_wall_time_minutes = round(total_wall_time_seconds / 60.0, 3)
    mean_time_min = round(total_wall_time_minutes / n_experiments, 3)

    wall_time_vs_prior_delta_minutes = round(
        total_wall_time_minutes - PRIOR_MILESTONE_WALL_TIME_MINUTES, 3
    )

    # --- Success criteria evaluation ------------------------------------------

    # RETRO-070 (LLMAsExtractorV1 recall >= 20%): Exp 616 v1_recall=0.04, gate_open=False.
    # Result: retro_070_resolved=False.  STILL BLOCKED.
    retro_070_resolved: bool = bool(results["616"].get("retro_070_resolved", False))

    # RETRO-033 (15th attempt): Exp 620 status=blocked, signed_improvement=0.0.
    # Result: retro_033_resolved=False.  STILL BLOCKED.
    retro_033_resolved: bool = bool(results["620"].get("retro_033_resolved", False))

    # RETRO-069 (SymCode-DSVD AUC >= 0.50): Exp 619 symcode_live_auc=0.804.
    # Result: retro_069_resolved=True.  CLOSED.
    retro_069_resolved: bool = bool(results["619"].get("retro_069_resolved", False))

    # NUP v6 cascade deployment: Exp 622 nup_v6_wired=True.  DONE.
    nup_v6_deployed: bool = bool(results["622"].get("nup_v6_wired", False))

    # FR-11 real violations: Exp 625 fr11_real_violations_confirmed=False.
    # Gate still closed on live real violations.
    fr11_confirmed: bool = bool(results["625"].get("fr11_real_violations_confirmed", False))

    # JEPA v13 calibration (ECE < 0.10): Exp 618 calibration_improved=False.
    # v13_ece=0.207 — well above the 0.10 threshold.  NOT MET.
    jepa_v13_calibrated: bool = bool(results["618"].get("calibration_improved", False))

    # MetaJuLS adaptation: Exp 621 adaptation_effective=True.  Positive signal.
    adaptation_effective: bool = bool(results["621"].get("adaptation_effective", False))

    # Best extractor: Exp 617 timed out so best_extractor field is absent.
    # Fall back to 'unknown' as specified.
    best_extractor: str = str(results["617"].get("best_extractor", "unknown"))

    # DualGPU parallel forward-pass: Exp 614 gpu1_utilization_confirmed=False.
    # Six consecutive milestones without a positive confirmation.
    dualgpu_confirmed: bool = bool(results["614"].get("gpu1_utilization_confirmed", False))

    # Key metric floats for downstream analysis
    v1_recall: float = float(results["616"].get("v1_recall", 0.0))
    symcode_live_auc: float = float(results["619"].get("symcode_live_auc", 0.0))
    v13_ece: float = float(results["618"].get("v13_ece", 0.0))
    cascade_latency_ms: float = float(results["622"].get("cascade_latency_ms", 0.0))
    trust_recall: float = float(results["623"].get("trust_recall", 0.0))

    # KV260 FPGA: Exp 624 simulation_validated=True, synthesis_succeeded='not_attempted'.
    simulation_validated: bool = bool(results["624"].get("simulation_validated", False))

    # --- RETRO closure rate ---------------------------------------------------
    # RETROs closed this milestone: RETRO-069 only.
    n_closed_this_milestone = sum([retro_069_resolved])
    retro_closure_rate = round(
        n_closed_this_milestone / len(_RETROS_OPEN_AT_MILESTONE_START), 3
    )
    # New RETRO opening this milestone: RETRO-071 if dualgpu unconfirmed for 6 milestones.
    n_new_retros = 1 if not dualgpu_confirmed else 0
    open_retro_count = (
        len(_RETROS_OPEN_AT_MILESTONE_START) - n_closed_this_milestone + n_new_retros
    )

    # --- Honest verdict -------------------------------------------------------
    # Priority: first check the main blocker (VR), then positive closures.
    if retro_033_resolved:
        honest_verdict = "first_positive_vr_achieved"
    elif retro_069_resolved and nup_v6_deployed:
        honest_verdict = "symcode_closed_nup_deployed_recall_still_blocked"
    elif retro_069_resolved:
        honest_verdict = "symcode_closed_recall_still_blocked"
    else:
        honest_verdict = "no_retros_closed"

    # --- New RETRO items opening this milestone --------------------------------
    new_retro_items: list[dict] = []
    if not dualgpu_confirmed:
        new_retro_items.append(
            {
                "id": "RETRO-071",
                "title": "DualGPU parallel forward-pass unconfirmed — SIX consecutive milestones",
                "opened_milestone": MILESTONE,
                "carry_count": 0,
                "description": (
                    "Exp 614 measured gpu1_utilization_confirmed=False.  This is the sixth "
                    "consecutive milestone where GPU-1 parallel utilisation has not been "
                    "confirmed.  The exclusion manifest DualGPU verification experiment "
                    "ran successfully but did not confirm GPU-1 utilisation.  Root cause: "
                    "the benchmark script does not load a large enough model to saturate "
                    "both GPUs simultaneously.  Resolution requires a real multi-GPU "
                    "forward-pass benchmark with a model >= 13B parameters spread across "
                    "both GPUs, measuring sustained GPU-1 utilization > 70% during the "
                    "inference pass.  Block any DualGPU performance claims until confirmed."
                ),
                "priority": "medium",
            }
        )

    # --- Open RETRO items carry-forward to .48 --------------------------------
    open_retro_items = [
        {
            "id": "RETRO-031",
            "title": "Partial carry — closure status unverified",
            "carry_count": ">=5",
            "action_required": "Verify closure in result files before .48 planning",
        },
        {
            "id": "RETRO-033",
            "title": "Live 25q verify-repair precision — 15 attempts, still not closed",
            "carry_count": ">=15",
            "action_required": (
                "Blocked: gate_open=False (v1_recall=4%).  Fifteen zero-positive "
                "attempts confirm that no extractor variant can cross the live/offline gap "
                "without mid-generation monitoring.  Do NOT schedule attempt #16 until "
                "interwhen (arXiv 2602.11202) is integrated and extractor recall >= 20% "
                "on live data.  ORACLE data elicitation (arXiv 2603.21140) is required "
                "to build a FOVER v5 training corpus that matches live LLM output style."
            ),
        },
        {
            "id": "RETRO-038",
            "title": "Live 200q VeriCoT+Wilson CI — still not closed",
            "carry_count": ">=10",
            "action_required": "Same root cause as RETRO-033. Block until recall > 30%.",
        },
        {
            "id": "RETRO-057",
            "title": "LowRankKAEM energy accuracy outside 5% tolerance",
            "carry_count": ">=4",
            "action_required": "Architectural redesign needed. Calibration layer approach exhausted.",
        },
        {
            "id": "RETRO-060",
            "title": "JEPA architecturally anti-correlated — superseded by RETRO-063",
            "carry_count": ">=4",
            "action_required": (
                "RETRO-063 validated.  JEPA v13 OOD AUC=0.868 confirms architecture is sound.  "
                "Calibration (ECE) remains open — v13_ece=0.207 vs threshold 0.10.  "
                "Close RETRO-060 tracking; open calibration work under JEPA v14."
            ),
        },
        {
            "id": "RETRO-064",
            "title": "CoACE recall 4% on live data — pipeline accuracy lift undetectable",
            "carry_count": ">=4",
            "action_required": (
                "LLMAsExtractorV1 v1_recall=0.04 (Exp 616).  TRUST agents trust_recall=0.0 "
                "(Exp 623).  Neither extractor architecture closes the gap.  "
                "interwhen + ORACLE data elicitation required (RETRO-033 root cause)."
            ),
        },
        {
            "id": "RETRO-065",
            "title": "RAPL unavailable — hardware energy calibration blocked",
            "carry_count": ">=4",
            "action_required": "Need Intel RAPL or AMD Energy driver on test machine",
        },
        {
            "id": "RETRO-066",
            "title": "CoACE offline/live distribution gap — extractor redesign unresolved",
            "carry_count": ">=3",
            "action_required": (
                "LLM-as-extractor in isolation (v1_recall=4%) cannot close the gap.  "
                "ORACLE data elicitation (arXiv 2603.21140) is the only identified path "
                "to a training corpus that matches live output distribution."
            ),
        },
        {
            "id": "RETRO-068",
            "title": "LLMAsExtractorV1 live recall 4-6% — below 20% gate threshold",
            "carry_count": ">=2",
            "action_required": (
                "Gate recall >= 20%.  Current: v1_recall=0.04 (Exp 616), trust_recall=0.0 "
                "(Exp 623).  interwhen mid-generation monitor is the next architectural "
                "intervention (arXiv 2602.11202).  No more extractor-only tuning passes."
            ),
        },
        {
            "id": "RETRO-070",
            "title": "LLMAsExtractorV1 still below 20% recall — architecture review required",
            "carry_count": 1,
            "action_required": (
                "Exp 616 v1_recall=0.04 — gate_open=False.  Implement interwhen "
                "(arXiv 2602.11202) mid-generation monitor to intercept chain-of-thought "
                "violations before they are committed to the token stream.  This is the "
                "architectural prerequisite for closing RETRO-033 at attempt #16."
            ),
        },
    ]
    if not dualgpu_confirmed:
        open_retro_items.append(
            {
                "id": "RETRO-071",
                "title": "DualGPU parallel forward-pass unconfirmed for six consecutive milestones",
                "carry_count": 0,
                "action_required": (
                    "Run a real multi-GPU benchmark with a model >= 13B spread across both GPUs.  "
                    "Measure sustained GPU-1 utilization > 70% during inference.  "
                    "Block all DualGPU performance claims until confirmed."
                ),
            }
        )

    # --- Top priorities for milestone .48 -------------------------------------
    # Priority logic per task specification:
    # if retro_033_resolved → scale to 200q Wilson CI
    # elif retro_070_resolved → extractor recall crosses 20%, run VR attempt #16
    # else → both still open, implement interwhen + ORACLE
    if retro_033_resolved:
        vr_priority = (
            "Scale to 200q Wilson CI (RETRO-038 closure).  "
            "Run on live corpus with winning extractor configuration."
        )
    elif retro_070_resolved:
        vr_priority = (
            "Extractor recall crosses 20% gate — run VR attempt #16 immediately.  "
            "Gate on live recall >= 20% confirmed; schedule VR with winning extractor."
        )
    else:
        vr_priority = (
            "RETRO-070 still open — implement interwhen (arXiv 2602.11202) "
            "mid-generation monitor + ORACLE data elicitation (arXiv 2603.21140) "
            "for FOVER v5 training data.  Both interventions are required before "
            "scheduling VR attempt #16.  No more extractor-only tuning passes."
        )

    top_priorities_for_48 = [
        vr_priority,
        (
            "JEPA v14: train on ORACLE-labeled FOVER v5 corpus with LLMAsExtractorV1 labels.  "
            "Target: v14_ece < 0.10 (JEPA v13 ECE=0.207 failed this threshold).  "
            "Use ORACLE-labeled data to close the calibration gap that synthetic training "
            "data cannot bridge.  JEPA v14 is the prerequisite for a calibrated cascade."
        ),
        (
            "KV260 FPGA: human must install Vivado 2023.2 before .49.  "
            "Exp 624 confirmed simulation_validated=True — the synchronous Ising RTL is "
            "functionally correct.  Synthesis (not_attempted) is the only remaining blocker.  "
            "Install Vivado, run synthesis, capture resource utilisation and timing closure, "
            "and publish as the first Carnot FPGA hardware result."
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
        # Primary success criteria (v22 booleans)
        "retro_033_resolved": retro_033_resolved,
        "retro_069_resolved": retro_069_resolved,
        "retro_070_resolved": retro_070_resolved,
        "nup_v6_deployed": nup_v6_deployed,
        "fr11_confirmed": fr11_confirmed,
        "jepa_v13_calibrated": jepa_v13_calibrated,
        "adaptation_effective": adaptation_effective,
        "best_extractor": best_extractor,
        "dualgpu_confirmed": dualgpu_confirmed,
        # Key metric floats
        "v1_recall": v1_recall,
        "symcode_live_auc": symcode_live_auc,
        "v13_ece": v13_ece,
        "cascade_latency_ms": cascade_latency_ms,
        "trust_recall": trust_recall,
        "simulation_validated": simulation_validated,
        # RETRO closure rate
        "retro_closure_rate": retro_closure_rate,
        "open_retro_count": open_retro_count,
        # Narrative
        "new_retro_items": new_retro_items,
        "open_retro_items": open_retro_items,
        "top_priorities_for_48": top_priorities_for_48,
        # Verdict
        "honest_verdict": honest_verdict,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Build and write the Exp 626 milestone retro artifact."""
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
    print(f"[Exp {EXP_ID}] retro_069_resolved={artifact['retro_069_resolved']}")
    print(f"[Exp {EXP_ID}] retro_070_resolved={artifact['retro_070_resolved']}")
    print(f"[Exp {EXP_ID}] nup_v6_deployed={artifact['nup_v6_deployed']}")
    print(f"[Exp {EXP_ID}] fr11_confirmed={artifact['fr11_confirmed']}")
    print(f"[Exp {EXP_ID}] jepa_v13_calibrated={artifact['jepa_v13_calibrated']}")
    print(f"[Exp {EXP_ID}] best_extractor={artifact['best_extractor']}")
    print(f"[Exp {EXP_ID}] dualgpu_confirmed={artifact['dualgpu_confirmed']}")
    print(f"[Exp {EXP_ID}] retro_closure_rate={artifact['retro_closure_rate']}")
    print(f"[Exp {EXP_ID}] total_wall_time_minutes={artifact['total_wall_time_minutes']}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
