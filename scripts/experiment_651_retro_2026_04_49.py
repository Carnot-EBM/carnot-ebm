#!/usr/bin/env python3
"""Experiment 651: Milestone 2026.04.49 Operational Retrospective.

**Researcher summary:**
    Milestone 2026.04.49 ran Exps 640-650 under themes:
    "Pre-Flight Infra v2, HERMES v2 Live Loop, Causal Verifier, Ensemble Gate v2,
    Live VR Attempt #17, Tier1 FR-11 Relay, JEPA v14 Platt Scaling, OTV Verifier,
    Parallel Ising Inertia, DualGPU 13B Proof v2, KAEM Multilevel Sparse".

    Key questions answered for .49:

    RETRO-070 (ensemble recall >= 30%): Exp 643 ensemble_recall=0.36 — gate_open=True.
        retro_070_resolved=True.  CLOSED.  Causal verifier (Exp 642) contributed recall=0.36
        to the ensemble (interwhen=0.12 + hermes_v2=0.0 + causal=0.36 → ensemble=0.36).

    RETRO-033 (attempt #17): Exp 644 signed_improvement=0.0, retro_033_resolved=False.
        STILL BLOCKED despite gate finally opening.  Seventeen consecutive zero-positive
        VR attempts.  Root cause: CI stub mode produces no real violations to repair.
        Next attempt requires AST-level code extraction, not regex or LLM prompt.

    RETRO-071 (DualGPU 13B proof): Exp 649 status=blocked, dualgpu_proven=False.
        HuggingFace weights still not cached.  GPUs confirmed (2x RTX 3090, 48 GB).
        Action: huggingface-cli download Qwen/Qwen2.5-7B-Instruct before .50.

    RETRO-057 (KAEM sparse < 5% error): Exp 650 multilevel_sparse_vs_dense_error=13.01%.
        retro_057_resolved=False.  Multilevel approach made accuracy worse, not better
        (42.44 MAE vs 3.03 standard, 1.73 sparse-only).  Try top_k=1.0 dense-sparse baseline.

    JEPA v14 calibration (ECE < 0.10): Exp 646 ece_after=0.023, calibration_target_met=True.
        jepa_v14_calibrated=True.  CLOSED.  Platt scaling reduced ECE by 87.96%.
        RETRO-060 (JEPA calibration) is now closed.

    OTV verifier: Exp 647 otv_viable=False (otv_auc=0.5, eorm_auc=1.0).  Keep EORM as Tier 2.

    Exclusion manifest: Exp 640 conductor_consulted=True.  Manifest wired and working.

    FR-11 relay: Exp 645 fr11_real_violations_confirmed=False (semi-real-ensemble mode).

    Inertia sampler: Exp 648 inertia_faster=False.  No convergence speedup vs baseline.
        V3 RTL spec written for KV260.

    Headline: 2 RETROs closed (RETRO-070, RETRO-060).  Gate opened but VR #17 failed.
    JEPA calibrated via Platt scaling.  Causal verifier the key architectural breakthrough.

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

EXP_ID = 651
TITLE = "Milestone 2026.04.49 Retrospective"
DELIVERABLE = "results/experiment_651_retro_2026_04_49.json"
MILESTONE = "2026.04.49"
SCHEMA = "carnot.operational_retro.v24"

# Prior milestone (.48) wall time in minutes — from .48 retro Exp 639.
PRIOR_MILESTONE_WALL_TIME_MINUTES = 18.445

# All 11 upstream experiment result files for milestone .49.
# Exp 651 (this retro) is #12 and is computed here, not loaded.
_MILESTONE_RESULTS = [
    ("640", "results/experiment_640_preflght_infra.json"),
    ("641", "results/experiment_641_hermes_v2_live.json"),
    ("642", "results/experiment_642_causal_verifier.json"),
    ("643", "results/experiment_643_ensemble_gate_v2.json"),
    ("644", "results/experiment_644_live_vr_attempt_17.json"),
    ("645", "results/experiment_645_tier1_fr11_relay.json"),
    ("646", "results/experiment_646_jepa_v14_platt.json"),
    ("647", "results/experiment_647_otv_verifier.json"),
    ("648", "results/experiment_648_parallel_ising_inertia.json"),
    ("649", "results/experiment_649_dualgpu_13b_v2.json"),
    ("650", "results/experiment_650_kaem_multilevel_sparse.json"),
]

# RETROs open at the START of milestone .49 (carry-forward from .48 retro, Exp 639).
# Source: Exp 639 open_retro_items contained these 11 items.
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
    "RETRO-070",
    "RETRO-071",
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
    """Load all 11 upstream results and compute .49 retrospective metrics.

    Why this is a pure function: the test suite can monkeypatch _REPO_ROOT and
    call compute_retro() with controlled fake result files, verifying every
    boolean branch without touching the real filesystem or running actual experiments.
    """
    # --- Load all upstream results ----------------------------------------
    results = {}
    for exp_id_str, rel_path in _MILESTONE_RESULTS:
        results[exp_id_str] = _load_result(rel_path)

    exp640 = results["640"]
    exp641 = results["641"]
    exp642 = results["642"]
    exp643 = results["643"]
    exp644 = results["644"]
    exp645 = results["645"]
    exp646 = results["646"]
    exp647 = results["647"]
    exp648 = results["648"]
    exp649 = results["649"]
    exp650 = results["650"]

    # --- Primary success criteria (per task spec) -------------------------
    hermes_v2_recall: float = float(exp641.get("hermes_v2_recall", 0.0))
    ensemble_recall: float = float(exp643.get("ensemble_recall", 0.0))
    retro_070_resolved: bool = bool(exp643.get("retro_070_resolved", ensemble_recall >= 0.30))
    retro_033_resolved: bool = bool(exp644.get("retro_033_resolved", False))
    retro_071_resolved: bool = bool(exp649.get("retro_071_resolved", False))
    retro_057_resolved: bool = bool(exp650.get("retro_057_resolved", False))
    jepa_v14_calibrated: bool = bool(exp646.get("calibration_target_met", False))
    otv_viable: bool = bool(exp647.get("otv_viable", False))
    manifest_wired: bool = bool(exp640.get("conductor_consulted", False))
    fr11_confirmed: bool = bool(exp645.get("fr11_real_violations_confirmed", False))
    inertia_faster: bool = bool(exp648.get("inertia_faster", False))
    dualgpu_proven: bool = bool(exp649.get("dualgpu_proven", False))

    # Secondary metrics used in open_retro_items carry-forward
    ece_after: float = float(exp646.get("ece_after", 1.0))
    multilevel_sparse_vs_dense_error: float = float(
        exp650.get("multilevel_sparse_vs_dense_error", 1.0)
    )

    # --- Wall time aggregation -------------------------------------------
    # Sum durations of the 11 upstream experiments (not this retro itself).
    upstream_duration_s: float = sum(r.get("duration_s", 0.0) for r in results.values())
    # Retro itself is negligible (<1s) — we do not add a placeholder.
    total_wall_time_minutes: float = round(upstream_duration_s / 60.0, 3)
    n_experiments_run: int = len(_MILESTONE_RESULTS) + 1  # +1 for this retro
    n_not_run: int = sum(1 for r in results.values() if not r)
    # Adjust n_experiments_run downward for experiments that never ran.
    n_experiments_run = n_experiments_run - n_not_run
    mean_time_min: float = round(
        total_wall_time_minutes / max(n_experiments_run, 1), 3
    )

    # --- RETRO closure accounting ----------------------------------------
    # RETRO-070 closed if retro_070_resolved.
    # RETRO-060 closed if jepa_v14_calibrated (calibration was the action required).
    n_closed_this_milestone: int = sum([
        retro_070_resolved,
        jepa_v14_calibrated,  # closes RETRO-060
    ])
    n_new_retros: int = 0  # no new RETROs opened in .49

    open_retro_count: int = (
        len(_RETROS_OPEN_AT_MILESTONE_START) - n_closed_this_milestone + n_new_retros
    )
    retro_closure_rate: float = round(
        n_closed_this_milestone / len(_RETROS_OPEN_AT_MILESTONE_START), 3
    )

    # --- Honest verdict ---------------------------------------------------
    if retro_033_resolved:
        honest_verdict = "vr_17_succeeded_033_closed"
    elif retro_070_resolved and jepa_v14_calibrated:
        honest_verdict = "retro_070_closed_jepa_calibrated_vr17_blocked"
    elif retro_070_resolved:
        honest_verdict = "retro_070_closed_vr17_blocked"
    elif jepa_v14_calibrated:
        honest_verdict = "jepa_calibrated_all_vr_retros_carry"
    else:
        honest_verdict = "no_retros_closed"

    # --- Open RETRO carry-forward ----------------------------------------
    open_retro_items: list[dict] = []

    open_retro_items.append({
        "id": "RETRO-031",
        "title": "Partial carry — closure status unverified",
        "carry_count": ">=7",
        "action_required": "Verify closure in result files before .50 planning.",
    })

    if not retro_033_resolved:
        open_retro_items.append({
            "id": "RETRO-033",
            "title": "Live 25q verify-repair precision — 17 attempts, still not closed",
            "carry_count": ">=17",
            "action_required": (
                "RETRO-033 carry: attempt #17 failed. "
                "Upgrade gate to 35% recall before attempt #18. "
                "Explore AST-level code extraction (not regex or LLM prompt)."
            ),
        })

    open_retro_items.append({
        "id": "RETRO-038",
        "title": "Live 200q VeriCoT+Wilson CI — still not closed",
        "carry_count": ">=12",
        "action_required": "Same root cause as RETRO-033. Block until VR produces signed_improvement > 0.",
    })

    if not retro_057_resolved:
        open_retro_items.append({
            "id": "RETRO-057",
            "title": "LowRankKAEM energy accuracy outside 5% tolerance",
            "carry_count": ">=6",
            "action_required": (
                "RETRO-057 carry: try top_k=1.0 (dense sparse baseline). "
                f"Exp 650 multilevel_sparse_vs_dense_error={multilevel_sparse_vs_dense_error:.2f} "
                "— far outside 5% threshold.  Multilevel approach worsened accuracy."
            ),
        })

    # RETRO-060 is closed if jepa_v14_calibrated; otherwise carry.
    if not jepa_v14_calibrated:
        open_retro_items.append({
            "id": "RETRO-060",
            "title": "JEPA calibration ECE still above 0.10 threshold",
            "carry_count": ">=6",
            "action_required": (
                f"ece_after={ece_after:.4f} — try isotonic regression calibration "
                "(non-parametric, monotone). Platt scaling was insufficient."
            ),
        })

    open_retro_items.append({
        "id": "RETRO-064",
        "title": "CoACE recall 4% on live data — pipeline accuracy lift undetectable",
        "carry_count": ">=6",
        "action_required": (
            f"ensemble_recall={ensemble_recall:.2f} (Exp 643).  "
            "Causal verifier crossed the 30% gate.  RETRO-033 VR #17 still blocked by CI stub."
        ),
    })

    open_retro_items.append({
        "id": "RETRO-065",
        "title": "RAPL unavailable — hardware energy calibration blocked",
        "carry_count": ">=6",
        "action_required": "Need Intel RAPL or AMD Energy driver on test machine.",
    })

    open_retro_items.append({
        "id": "RETRO-066",
        "title": "CoACE offline/live distribution gap — extractor redesign unresolved",
        "carry_count": ">=5",
        "action_required": (
            "ORACLE data elicitation (arXiv 2603.21140) is the only identified path "
            "to a training corpus that matches live output distribution."
        ),
    })

    open_retro_items.append({
        "id": "RETRO-068",
        "title": "LLMAsExtractorV1 live recall 4-12% — below 20% gate threshold",
        "carry_count": ">=4",
        "action_required": (
            "Gate recall >= 20%.  Ensemble recall=0.36 (Exp 643) uses causal verifier; "
            "HERMES v2 alone recall=0.0 in Exp 641 (CI stub mode)."
        ),
    })

    if not retro_070_resolved:
        open_retro_items.append({
            "id": "RETRO-070",
            "title": "interwhen recall still below 20% — HERMES tool-augmented pipeline next",
            "carry_count": 3,
            "action_required": (
                "RETRO-070 carry: ensemble recall still below 30%. "
                "Try forced structured-format generation: prompt model to write equations explicitly."
            ),
        })

    if not retro_071_resolved:
        open_retro_items.append({
            "id": "RETRO-071",
            "title": "DualGPU 13B proof blocked by missing HF model weights",
            "carry_count": 2,
            "action_required": (
                "RETRO-071 carry: pre-cache Qwen2.5-7B-Instruct weights, retry. "
                "Run: huggingface-cli download Qwen/Qwen2.5-7B-Instruct "
                "--local-dir ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
            ),
        })

    new_retro_items: list[dict] = []

    # --- Top priorities for milestone .50 ---------------------------------
    top_priorities_for_50: list[str] = []

    if retro_033_resolved:
        top_priorities_for_50.append("Scale to 200q Wilson CI (RETRO-038).")
    elif retro_070_resolved:
        top_priorities_for_50.append("Run VR #17 immediately.")
    else:
        top_priorities_for_50.append(
            "Implement structured-format forcing: prompt Qwen3.5-0.8B to write "
            "explicit equations at each step. Then HERMES v2 can parse them exactly."
        )

    top_priorities_for_50.append(
        "FPGA: human must install Vivado 2023.2. "
        "TCL v2 ready (Exp 636). v3 RTL spec available (Exp 648)."
    )

    if not jepa_v14_calibrated:
        top_priorities_for_50.append(
            "Try isotonic regression calibration (non-parametric)."
        )
    else:
        top_priorities_for_50.append(
            "Deploy JEPA v14 + Platt in cascade. Measure wall-clock throughput."
        )

    return {
        "schema": SCHEMA,
        "milestone": MILESTONE,
        "n_experiments_run": n_experiments_run,
        "n_not_run": n_not_run,
        "total_wall_time_minutes": total_wall_time_minutes,
        "mean_time_min": mean_time_min,
        "retro_033_resolved": retro_033_resolved,
        "retro_057_resolved": retro_057_resolved,
        "retro_070_resolved": retro_070_resolved,
        "retro_071_resolved": retro_071_resolved,
        "jepa_v14_calibrated": jepa_v14_calibrated,
        "otv_viable": otv_viable,
        "manifest_wired": manifest_wired,
        "fr11_confirmed": fr11_confirmed,
        "inertia_faster": inertia_faster,
        "dualgpu_proven": dualgpu_proven,
        "hermes_v2_recall": hermes_v2_recall,
        "ensemble_recall": ensemble_recall,
        "retro_closure_rate": retro_closure_rate,
        "open_retro_count": open_retro_count,
        "new_retro_items": new_retro_items,
        "open_retro_items": open_retro_items,
        "top_priorities_for_50": top_priorities_for_50,
        "honest_verdict": honest_verdict,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the .49 retrospective: load results, compute metrics, write deliverable."""
    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30)  # noqa: F841

    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    retro = compute_retro()

    artifact = tmpl.build_result(retro, status="success")

    # Overwrite schema to v24 (build_result may set generic schema).
    artifact["schema"] = SCHEMA
    artifact["env_autofix"] = True

    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"[Exp {EXP_ID}] honest_verdict={artifact['honest_verdict']}")
    print(f"[Exp {EXP_ID}] retro_closure_rate={artifact['retro_closure_rate']}")
    print(f"[Exp {EXP_ID}] open_retro_count={artifact['open_retro_count']}")
    print(f"[Exp {EXP_ID}] retro_070_resolved={artifact['retro_070_resolved']}")
    print(f"[Exp {EXP_ID}] jepa_v14_calibrated={artifact['jepa_v14_calibrated']}")
    print(f"[Exp {EXP_ID}] ensemble_recall={artifact['ensemble_recall']}")
    print(f"[Exp {EXP_ID}] Deliverable: {DELIVERABLE}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
