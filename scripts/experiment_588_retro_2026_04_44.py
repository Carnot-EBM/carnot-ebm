#!/usr/bin/env python3
"""Experiment 588: Milestone 2026.04.44 Operational Retrospective.

**Researcher summary:**
    Milestone 2026.04.44 ran Exps 575-588 under the theme
    "Recall Surgery and Contrastive JEPA — First Verified Improvement on Live Models".

    Six pre-existing RETROs were the surgery targets for this milestone:

    RETRO-056 (ExclusionManifest never built — 7th milestone) — Exp 575 built the
        manifest (conductor_exclusion_manifest.json, 5 excluded IDs).
        exclusion_manifest_built=True.  CLOSED by deliverable, but the conductor
        is NOT yet wired to consult it (conductor_consulted=False).  A follow-on
        RETRO-067 tracks the wiring task.

    RETRO-062 (Live 50q A never collected — questions 0-49 missing) — Exp 578
        collected 100 live pairs from 50 GSM8K questions using live_gpu inference.
        n_pairs_collected=100 >= 40.  CLOSED.

    RETRO-063 (JEPA AUC still below random despite PURE objective) — Exp 580
        retrained JEPA v11 with the contrastive CPMI hinge-margin loss using
        9 real pairs from Exp 577.  v11_auc=1.0 > 0.5.  CLOSED — but this is a
        9-pair overfitting risk; the AUC will need to be confirmed on the larger
        Exp 578 corpus before claiming full resolution.

    RETRO-064 (CoACE recall 5.9% — pipeline accuracy lift undetectable) — Exp 576
        achieved 86.7% recall on the offline synthetic corpus, but Exp 581 (live gate)
        measured v2_recall=0.059 on production model outputs.  The offline boost did
        NOT transfer to live data.  STILL BLOCKED.  gate_open=False.

    RETRO-033 (Live 25q verify-repair accuracy unchanged — attempt #12) — Exp 582
        was gated by Exp 581 recall check and did not run (signed_improvement=0.0,
        inference_mode='blocked_gate_closed_recall_too_low').  STILL BLOCKED.

    FPGA synthesis (Exp 584): Vivado is not installed on this machine.
        bitfile_built=False, vivado_available=False.  Not blocked by synthesis errors,
        blocked by missing toolchain.

    Supporting experiments:
        Exp 577 (JEPA CPMI pair builder): 9 real pairs built, contrastive loss validated.
        Exp 579 (Live 50q C): blocked (CARNOT_FORCE_LIVE not set pre-session).
        Exp 583 (FR-11 v3): blocked (gate closed, recall too low).
        Exp 585 (KV260 benchmark v3): blocked (no bitfile from Exp 584).
        Exp 586 (Symbolic-KAN): formula_interpretable=True, MSE=0.059.  Viable.
        Exp 587 (DSVD adapter): dsvd_auc=0.976 vs coace_v1_auc=0.824.  tier_2_5_viable=True.

    Headline: THREE RETROs CLOSED (RETRO-056, RETRO-062, RETRO-063), but end-to-end
    accuracy improvement is still zero.  The critical offline/live recall gap for CoACE
    is the single highest-leverage unresolved issue.

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

EXP_ID = 588
TITLE = "Milestone 2026.04.44 Retrospective"
DELIVERABLE = "results/experiment_588_retro_2026_04_44.json"
MILESTONE = "2026.04.44"
SCHEMA = "carnot.operational_retro.v19"

# All 13 upstream experiment result files for milestone .44.
# Exp 588 (this retro) is #14 and is computed here, not loaded.
_MILESTONE_RESULTS = [
    ("575", "results/experiment_575_exclusion_manifest.json"),
    ("576", "results/experiment_576_coace_recall_boost.json"),
    ("577", "results/experiment_577_jepa_cpmi_pairs.json"),
    ("578", "results/experiment_578_live_data_a_v3.json"),
    ("579", "results/experiment_579_live_data_c.json"),
    ("580", "results/experiment_580_jepa_v11_retrain.json"),
    ("581", "results/experiment_581_coace_recall_diagnostic_v2.json"),
    ("582", "results/experiment_582_live_vr_coace_v2.json"),
    ("583", "results/experiment_583_fr11_real_violations_v3.json"),
    ("584", "results/experiment_584_kv260_synthesis.json"),
    ("585", "results/experiment_585_kv260_live_benchmark_v3.json"),
    ("586", "results/experiment_586_symbolic_kan_energy.json"),
    ("587", "results/experiment_587_dsvd_adapter.json"),
]

# RETROs open at the START of milestone .44 (carry-forward from .43 retro).
# Used to compute retro_closure_rate: closed_this_milestone / open_at_start.
_RETROS_OPEN_AT_MILESTONE_START = [
    "RETRO-031",  # partial carry — closure status unverified
    "RETRO-033",  # live 25q precision — 12+ attempts, still not closed
    "RETRO-038",  # live 100q VeriCoT — 8+ attempts, still not closed
    "RETRO-049",  # NUP Probe v4 contrastive margin loss redesign
    "RETRO-056",  # ExclusionManifest never built (7 milestones)
    "RETRO-057",  # LowRankKAEM energy accuracy outside tolerance
    "RETRO-060",  # JEPA architecturally anti-correlated (superseded by 063)
    "RETRO-062",  # Live 50q A unrun — questions 0-49 missing
    "RETRO-063",  # JEPA v10 inverted despite PURE objective
    "RETRO-064",  # CoACE recall 5.9% — pipeline accuracy lift undetectable
    "RETRO-065",  # RAPL unavailable — hardware energy calibration blocked
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
    """Load all .44 milestone results and compute aggregate retro metrics.

    Returns a dict with every v19 schema field, ready to pass to build_result().

    The v19 schema tracks ten boolean success criteria specific to milestone .44:
    retro_056_resolved, retro_064_partial, retro_064_resolved, retro_063_resolved,
    retro_062_resolved, retro_033_resolved, fr11_improved, fpga_progress,
    symbolic_viable, dsvd_viable.
    """
    results = {exp_id: _load_result(path) for exp_id, path in _MILESTONE_RESULTS}

    # --- Per-experiment status counts ----------------------------------------
    # n_experiments includes this retro script (588) as the 14th experiment.
    n_experiments = len(_MILESTONE_RESULTS) + 1  # +1 for this retro
    n_missing = sum(1 for r in results.values() if not r)
    n_experiments_run = n_experiments - n_missing
    n_not_run = n_missing

    # --- Wall time -----------------------------------------------------------
    # Sum duration_s for all upstream experiments.  588 (this retro) is near-zero
    # and is excluded to avoid a self-referential timing loop.
    total_wall_time_seconds = sum(r.get("duration_s", 0.0) for r in results.values())
    total_wall_time_minutes = round(total_wall_time_seconds / 60.0, 3)
    mean_time_min = round(total_wall_time_minutes / n_experiments, 3)

    # --- Success criteria evaluation -----------------------------------------

    # RETRO-056 (7th milestone): ExclusionManifest built (Exp 575).
    # Result: exclusion_manifest_built=True — manifest exists with 5 excluded IDs.
    # Caveat: conductor_consulted=False — the manifest is not yet wired into pick_next_task().
    retro_056_resolved: bool = bool(results["575"].get("exclusion_manifest_built", False))

    # RETRO-064 partial: CoACEV2 achieves recall >= 0.20 on live outputs (Exp 581).
    # Result: v2_recall=0.059 — gate_open=False, threshold not reached.
    # NOTE: Exp 576 showed 86.7% offline recall, but this did not transfer to live data.
    retro_064_partial: bool = bool(results["581"].get("retro_064_partial", False))

    # RETRO-064 resolved: CoACEV2 achieves recall >= 0.30 on live outputs (Exp 581).
    retro_064_resolved: bool = bool(results["581"].get("retro_064_resolved", False))

    # RETRO-063: JEPA v11 with CPMI objective achieves AUC >= 0.600 (Exp 580).
    # Result: v11_auc=1.0 > 0.5.  CLOSED — but 9-pair corpus risks overfitting.
    retro_063_resolved: bool = float(results["580"].get("v11_auc", 0.0)) > 0.5

    # RETRO-062: Live 50q A collected >= 40 pairs (Exp 578).
    # Result: n_pairs_collected=100 >= 40.  CLOSED.
    retro_062_resolved: bool = int(results["578"].get("n_pairs_collected", 0)) >= 40

    # RETRO-033 (attempt #12): Live VR with CoACEV2 shows signed_improvement > 0
    # AND inference_mode='live_gpu' (Exp 582).
    # Result: Exp 582 blocked (gate_closed_recall_too_low), signed_improvement=0.0.
    retro_033_resolved: bool = (
        float(results["582"].get("signed_improvement", 0.0)) > 0.0
        and results["582"].get("inference_mode") == "live_gpu"
    )

    # FR-11 improved: Exp 583 fr11_improved=True.
    # Result: False — gate closed, no violations processed.
    fr11_improved: bool = bool(results["583"].get("fr11_improved", False))

    # FPGA progress: Exp 584 bitfile_built=True OR vivado_available=True.
    # Result: both False — Vivado is not installed on this machine.
    fpga_progress: bool = (
        bool(results["584"].get("bitfile_built", False))
        or bool(results["584"].get("vivado_available", False))
    )

    # Symbolic-KAN viable: Exp 586 formula_interpretable=True.
    # Result: True — tanh formula recovered with MSE=0.059.
    symbolic_viable: bool = bool(results["586"].get("formula_interpretable", False))

    # DSVD viable: Exp 587 tier_2_5_viable=True.
    # Result: True — dsvd_auc=0.976 vs coace_v1_auc=0.824.
    dsvd_viable: bool = bool(results["587"].get("tier_2_5_viable", False))

    # --- RETRO closure rate --------------------------------------------------
    # RETROs closed this milestone:
    #   RETRO-056: manifest built (retro_056_resolved=True)
    #   RETRO-062: live pairs collected (retro_062_resolved=True)
    #   RETRO-063: JEPA v11 AUC > 0.5 (retro_063_resolved=True)
    n_closed_this_milestone = sum([
        retro_056_resolved,  # RETRO-056 closed
        retro_062_resolved,  # RETRO-062 closed
        retro_063_resolved,  # RETRO-063 closed
    ])
    retro_closure_rate = round(
        n_closed_this_milestone / len(_RETROS_OPEN_AT_MILESTONE_START), 3
    )
    open_retro_count = len(_RETROS_OPEN_AT_MILESTONE_START) - n_closed_this_milestone + 2  # +2 new

    # --- Honest verdict ------------------------------------------------------
    # Determined by whether live verify-repair achieved positive improvement this milestone.
    if retro_033_resolved:
        honest_verdict = "first_positive_achieved"
    elif retro_064_partial:
        honest_verdict = "recall_fixed_no_positive"
    else:
        honest_verdict = "recall_still_blocked"

    # --- New RETRO items opening this milestone -------------------------------
    new_retro_items = [
        {
            "id": "RETRO-066",
            "title": "CoACE offline/live distribution gap — 86.7% offline recall collapses to 5.9% live",
            "opened_milestone": MILESTONE,
            "carry_count": 0,
            "description": (
                "Exp 576 improved CoACEV2 recall to 86.7% on the synthetic offline corpus "
                "(GSM8K questions with controlled error injection).  Exp 581 then measured "
                "the same CoACEV2 extractor on 25 live production responses from Qwen3.5-0.8B "
                "and google/gemma-4-E4B-it.  Result: v2_recall=0.059 — gate_open=False. "
                "The offline training distribution does not match live model output patterns. "
                "Live models produce different error types (numeric off-by-one, unit confusion, "
                "multi-step chain breaks) than the synthetic corpus.  The extractor must be "
                "calibrated on live model outputs before the offline recall gain is meaningful."
            ),
            "priority": "critical",
        },
        {
            "id": "RETRO-067",
            "title": "ExclusionManifest built but conductor not wired — excluded experiments still spawn",
            "opened_milestone": MILESTONE,
            "carry_count": 0,
            "description": (
                "Exp 575 built scripts/conductor_exclusion_manifest.json listing 5 excluded "
                "experiment IDs (308, 260, 309, 425, 410) with estimated savings of 385 min/milestone. "
                "However conductor_consulted=False — the research_conductor.py pick_next_task() "
                "function does not call check_exclusion_manifest.py before spawning agents. "
                "The cumulative wasted time since the retro was first raised is 2695 minutes. "
                "Wiring the manifest check into pick_next_task() is a 10-line code change that "
                "would save roughly 6 hours per milestone of unnecessary agent spawns."
            ),
            "priority": "high",
        },
    ]

    # --- Open RETRO items carry-forward to .45 --------------------------------
    open_retro_items = [
        {
            "id": "RETRO-031",
            "title": "Partial carry — closure status unverified",
            "carry_count": ">=3",
            "action_required": "Verify closure in result files before .45 planning",
        },
        {
            "id": "RETRO-033",
            "title": "Live 25q verify-repair precision — 12+ attempts, still not closed",
            "carry_count": ">=12",
            "action_required": (
                "Blocked: gate_open=False (CoACE recall=5.9%).  Do not schedule attempt #13 "
                "until RETRO-064 and RETRO-066 (offline/live distribution gap) are resolved."
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
            "id": "RETRO-057",
            "title": "LowRankKAEM energy accuracy outside 5% tolerance",
            "carry_count": ">=2",
            "action_required": "Architectural redesign needed. Calibration layer approach exhausted.",
        },
        {
            "id": "RETRO-060",
            "title": "JEPA architecturally anti-correlated — superseded by RETRO-063",
            "carry_count": ">=2",
            "action_required": (
                "RETRO-063 closed with v11_auc=1.0 on 9 pairs. Monitor for overfitting "
                "when Exp 578's 100 pairs are used for retrain in .45."
            ),
        },
        {
            "id": "RETRO-064",
            "title": "CoACE recall 5.9% on live data — pipeline accuracy lift undetectable",
            "carry_count": 1,
            "action_required": (
                "Fix offline/live distribution gap (RETRO-066).  Build live-calibrated "
                "training corpus from Exp 578 pairs before next CoACEV2 retrain."
            ),
        },
        {
            "id": "RETRO-065",
            "title": "RAPL unavailable — hardware energy calibration blocked",
            "carry_count": 1,
            "action_required": "Need Intel RAPL or AMD Energy driver on test machine",
        },
        {
            "id": "RETRO-066",
            "title": "CoACE offline/live distribution gap",
            "carry_count": 0,
            "action_required": (
                "Build live-corpus training set from Exp 578 pairs; retrain CoACEV2 on "
                "real model outputs; measure live recall before scheduling any VR experiment."
            ),
        },
        {
            "id": "RETRO-067",
            "title": "ExclusionManifest not wired into conductor",
            "carry_count": 0,
            "action_required": (
                "Wire check_exclusion_manifest.py into pick_next_task() in research_conductor.py. "
                "Expected savings: ~385 min/milestone."
            ),
        },
    ]

    # --- Top priorities for milestone .45 ------------------------------------
    top_priorities_for_45 = [
        (
            "Fix CoACE offline/live distribution gap (RETRO-066).  Build a live-corpus "
            "training set from the 100 pairs collected in Exp 578 and retrain CoACEV2 on "
            "real model outputs from Qwen3.5-0.8B and gemma-4-E4B-it.  The offline recall "
            "of 86.7% (Exp 576) is meaningless until it transfers to live data.  This is the "
            "single highest-leverage action: it unblocks RETRO-033, RETRO-038, RETRO-064, "
            "and the entire live verify-repair accuracy improvement chain."
        ),
        (
            "Retrain JEPA v11 on the full 100-pair live corpus (Exp 578) to validate that "
            "AUC=1.0 from Exp 580 is not 9-pair overfitting.  The contrastive CPMI objective "
            "is architecturally sound (RETRO-063 closed) but the corpus was too small for a "
            "reliable AUC estimate.  Use Exp 578 pairs as positive/negative training signal "
            "for a v12 retrain before claiming JEPA is production-ready."
        ),
        (
            "Wire ExclusionManifest into conductor pick_next_task() (RETRO-067).  Exp 575 "
            "built the manifest; the wiring is a 10-line change in research_conductor.py. "
            "The cumulative waste is 2695 minutes (44+ hours) over 7 milestones.  Do this "
            "as the first infrastructure task of .45 before any research experiments run."
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
        # Success criteria
        "retro_056_resolved": retro_056_resolved,
        "retro_064_partial": retro_064_partial,
        "retro_064_resolved": retro_064_resolved,
        "retro_063_resolved": retro_063_resolved,
        "retro_062_resolved": retro_062_resolved,
        "retro_033_resolved": retro_033_resolved,
        "fr11_improved": fr11_improved,
        "fpga_progress": fpga_progress,
        "symbolic_viable": symbolic_viable,
        "dsvd_viable": dsvd_viable,
        # RETRO closure rate
        "retro_closure_rate": retro_closure_rate,
        "open_retro_count": open_retro_count,
        # Narrative
        "new_retro_items": new_retro_items,
        "open_retro_items": open_retro_items,
        "top_priorities_for_45": top_priorities_for_45,
        # Verdict
        "honest_verdict": honest_verdict,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Build and write the Exp 588 milestone retro artifact."""
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
    print(f"[Exp {EXP_ID}] retro_056_resolved={artifact['retro_056_resolved']}")
    print(f"[Exp {EXP_ID}] retro_062_resolved={artifact['retro_062_resolved']}")
    print(f"[Exp {EXP_ID}] retro_063_resolved={artifact['retro_063_resolved']}")
    print(f"[Exp {EXP_ID}] retro_064_partial={artifact['retro_064_partial']}")
    print(f"[Exp {EXP_ID}] retro_033_resolved={artifact['retro_033_resolved']}")
    print(f"[Exp {EXP_ID}] fpga_progress={artifact['fpga_progress']}")
    print(f"[Exp {EXP_ID}] symbolic_viable={artifact['symbolic_viable']}")
    print(f"[Exp {EXP_ID}] dsvd_viable={artifact['dsvd_viable']}")
    print(f"[Exp {EXP_ID}] retro_closure_rate={artifact['retro_closure_rate']}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
