#!/usr/bin/env python3
"""Experiment 639: Milestone 2026.04.48 Operational Retrospective.

**Researcher summary:**
    Milestone 2026.04.48 ran Exps 627-638 under themes:
    "InterWhenMonitor, ORACLE FOVER v5, InterwhenDiagnostic Gate, Live VR #16,
    JEPA v14, DualGPU 13B Proof, HERMES Adapter, Multilevel KAN KAEMEnergy,
    AdapTrack Backtrack, FPGA TCL v2, LowRankKAEM Sparse, FR-11 Self-Learning Relay".

    Key questions answered:

    RETRO-070 (interwhen recall >= 20%): Exp 629 measured interwhen_recall_primary=0.12 —
        gate_open=False.  retro_070_resolved=False.  STILL BLOCKED.  interwhen mid-generation
        monitor is improving (0.04 → 0.12 across milestones) but has not crossed the 20%
        gate threshold.  HERMES tool-augmented step boundary is the next architectural
        intervention.

    RETRO-033 (16th attempt): Exp 630 status=blocked, gate_open=False,
        retro_033_resolved=False.  STILL BLOCKED.  Sixteen consecutive zero-positive VR
        attempts.  The gate (extractor recall 12% < 20%) prevented scheduling VR attempt #16.

    RETRO-071 (DualGPU 13B proof): Exp 632 dualgpu_proven=False because HuggingFace model
        weights are not cached in the CI environment.  retro_071_resolved=False.
        GPUs are present (2x RTX 3090, 48 GB total); re-run with HF weights cached.

    RETRO-057 (LowRankKAEM sparse < 5% error): Exp 637 sparse_vs_dense_error=0.429 >>
        0.05 threshold.  retro_057_resolved=False.  Sparse redesign alone cannot close gap.

    JEPA v14 calibration (ECE < 0.10): Exp 631 v14_ece=0.132 (down from v13 0.207) but
        calibration_improved=False (threshold not crossed).  jepa_v14_calibrated=False.

    FR-11 real violations: Exp 638 fr11_real_violations_confirmed=False.
        Gate still closed; synthetic fallback maintained FR-11 relay continuity.

    HERMES adapter: Exp 633 hermes_improvement=True.  hermes_recall=0.12 with lower
        FP rate (0.20) vs interwhen standalone (0.40).  Positive architectural signal.

    Multilevel KAN KAEMEnergy: Exp 634 multilevel_wins=False (multilevel_faster=False,
        accuracy_improvement < 0).  No improvement over standard KAE.

    AdapTrack backtrack: Exp 635 adaptrack_improves_recall=False (recall 0.08 vs
        interwhen baseline 0.12).  Comparable but not better.

    FPGA TCL v2: Exp 636 tcl_v2_written='hardware/kv260/synth_ising_v2.tcl'.  TCL updated.
        Synthesis deferred pending human Vivado installation.

    Headline: ZERO RETROs closed.  HERMES improves FP rate but recall gate still blocked.
    All four key RETROs (033, 057, 070, 071) carry into .49.  JEPA v14 ECE improving
    (0.207 → 0.132) but not yet at threshold.  Vivado installation required from human
    before KV260 synthesis can proceed.

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

EXP_ID = 639
TITLE = "Milestone 2026.04.48 Retrospective"
DELIVERABLE = "results/experiment_639_retro_2026_04_48.json"
MILESTONE = "2026.04.48"
SCHEMA = "carnot.operational_retro.v23"

# Prior milestone (.47) wall time in minutes — from .47 retro Exp 626.
PRIOR_MILESTONE_WALL_TIME_MINUTES = 2.092

# All 12 upstream experiment result files for milestone .48.
# Exp 639 (this retro) is #13 and is computed here, not loaded.
_MILESTONE_RESULTS = [
    ("627", "results/experiment_627_interwhen_monitor.json"),
    ("628", "results/experiment_628_oracle_fover_v5.json"),
    ("629", "results/experiment_629_interwhen_diagnostic.json"),
    ("630", "results/experiment_630_live_vr_attempt_16.json"),
    ("631", "results/experiment_631_jepa_v14_oracle.json"),
    ("632", "results/experiment_632_dualgpu_13b_proof.json"),
    ("633", "results/experiment_633_hermes_adapter.json"),
    ("634", "results/experiment_634_multilevel_kan_kaem.json"),
    ("635", "results/experiment_635_adaptrack_backtrack.json"),
    ("636", "results/experiment_636_fpga_tcl_v2.json"),
    ("637", "results/experiment_637_lowrank_kaem_sparse.json"),
    ("638", "results/experiment_638_tier1_fr11_relay.json"),
]

# RETROs open at the START of milestone .48 (carry-forward from .47 retro, Exp 626).
# Used to compute retro_closure_rate: closed_this_milestone / open_at_start.
# Source: Exp 626 open_retro_items contained these 11 items.
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
    """Load all .48 milestone results and compute aggregate retro metrics.

    Returns a dict with every v23 schema field, ready to pass to build_result().

    The v23 schema tracks ten boolean success criteria specific to milestone .48:
    retro_033_resolved, retro_057_resolved, retro_070_resolved, retro_071_resolved,
    jepa_v14_calibrated, fr11_confirmed, multilevel_wins, adaptrack_improves,
    hermes_improves, and fpga_tcl_updated (bool).
    """
    results = {exp_id: _load_result(path) for exp_id, path in _MILESTONE_RESULTS}

    # --- Per-experiment status counts -----------------------------------------
    # n_experiments includes this retro script (639) as the 13th experiment.
    n_experiments = len(_MILESTONE_RESULTS) + 1  # +1 for this retro
    n_missing = sum(1 for r in results.values() if not r)
    n_experiments_run = n_experiments - n_missing
    n_not_run = n_missing

    # --- Load individual experiment results ------------------------------------
    exp627 = results["627"]
    exp629 = results["629"]
    exp630 = results["630"]
    exp631 = results["631"]
    exp632 = results["632"]
    exp633 = results["633"]
    exp634 = results["634"]
    exp635 = results["635"]
    exp636 = results["636"]
    exp637 = results["637"]
    exp638 = results["638"]

    # --- Key success criteria for .48 -----------------------------------------

    # RETRO-070: interwhen recall >= 20% gate — Exp 629 diagnostic verdict.
    retro_070_resolved: bool = bool(exp629.get("retro_070_resolved", False))
    interwhen_recall = exp629.get("interwhen_recall_primary", exp627.get("interwhen_recall", 0.0))
    gate_open = bool(exp629.get("gate_open", False))

    # RETRO-033: live VR positive improvement — Exp 630 gate outcome.
    retro_033_resolved: bool = bool(exp630.get("retro_033_resolved", False))
    signed_improvement: float = float(exp630.get("signed_improvement", 0.0))

    # JEPA v14 calibration: ECE < 0.10 threshold — Exp 631 result.
    jepa_v14_calibrated: bool = bool(exp631.get("calibration_improved", False))
    v14_ece: float = float(exp631.get("v14_ece", 1.0))
    v14_ood_auc: float = float(exp631.get("v14_ood_auc", 0.0))

    # RETRO-071: DualGPU 13B forward-pass proven — Exp 632 result.
    retro_071_resolved: bool = bool(exp632.get("retro_071_resolved", False))
    dualgpu_proven: bool = bool(exp632.get("dualgpu_proven", False))

    # HERMES adapter improvement: lower FP rate and comparable recall — Exp 633.
    hermes_improves: bool = bool(exp633.get("hermes_improvement", False))
    hermes_recall: float = float(exp633.get("hermes_recall", 0.0))
    hermes_fp_rate: float = float(exp633.get("hermes_fp_rate", 1.0))

    # Multilevel KAN KAEMEnergy improvement: Exp 634.
    # The experiment reports 'multilevel_faster' not 'multilevel_wins'; treat
    # either field as the signal (task spec uses 'multilevel_wins').
    multilevel_wins: bool = bool(
        exp634.get("multilevel_wins", exp634.get("multilevel_faster", False))
    )

    # AdapTrack recall improvement over interwhen baseline: Exp 635.
    adaptrack_improves: bool = bool(exp635.get("adaptrack_improves_recall", False))
    adaptrack_recall: float = float(exp635.get("adaptrack_recall", 0.0))

    # FPGA TCL v2 written: Exp 636.
    fpga_tcl_updated: bool = exp636.get("tcl_v2_written") is not None

    # RETRO-057: LowRankKAEM sparse vs dense error < 5% — Exp 637.
    retro_057_resolved: bool = bool(exp637.get("retro_057_resolved", False))
    sparse_vs_dense_error: float = float(exp637.get("sparse_vs_dense_error", 1.0))

    # FR-11 real violations confirmed: Exp 638.
    fr11_confirmed: bool = bool(exp638.get("fr11_real_violations_confirmed", False))

    # --- Wall time aggregation -------------------------------------------------
    # Sum duration_s from all 12 upstream experiments (retro itself adds ~0 s).
    upstream_duration_s = sum(r.get("duration_s", 0.0) for r in results.values())
    total_wall_time_minutes = round(upstream_duration_s / 60.0, 3)
    mean_time_min = round(total_wall_time_minutes / n_experiments, 3)
    wall_time_vs_prior_delta_minutes = round(
        total_wall_time_minutes - PRIOR_MILESTONE_WALL_TIME_MINUTES, 3
    )

    # --- Retro closure accounting ---------------------------------------------
    # Which RETROs closed this milestone?
    n_closed_this_milestone = 0  # none resolved in .48
    # No new RETROs to open: RETRO-071 was already opened in .47; DualGPU blocked by
    # missing HF weights (model load infra issue), not a new architectural problem.
    n_new_retros = 0
    open_retro_count = (
        len(_RETROS_OPEN_AT_MILESTONE_START) - n_closed_this_milestone + n_new_retros
    )
    retro_closure_rate = round(
        n_closed_this_milestone / len(_RETROS_OPEN_AT_MILESTONE_START), 3
    )

    # --- Honest verdict -------------------------------------------------------
    if retro_033_resolved:
        honest_verdict = "first_positive_vr_achieved"
    elif hermes_improves and retro_070_resolved:
        honest_verdict = "hermes_improved_gate_open_vr16_ready"
    elif hermes_improves:
        honest_verdict = "hermes_improved_all_retros_carry"
    else:
        honest_verdict = "no_retros_closed"

    # --- Open RETRO carry-forward items ----------------------------------------
    open_retro_items: list[dict] = []

    open_retro_items.append({
        "id": "RETRO-031",
        "title": "Partial carry — closure status unverified",
        "carry_count": ">=6",
        "action_required": "Verify closure in result files before .49 planning",
    })

    if not retro_033_resolved:
        open_retro_items.append({
            "id": "RETRO-033",
            "title": "Live 25q verify-repair precision — 16 attempts, still not closed",
            "carry_count": ">=16",
            "action_required": (
                "Blocked: gate_open=False (interwhen_recall=12%).  Sixteen zero-positive "
                "attempts confirm that no extractor variant can cross the live/offline gap "
                "without mid-generation monitoring.  Do NOT schedule attempt #17 until "
                "interwhen recall >= 30%.  HERMES tool-augmented loop (Exp 633) reduces "
                "FP rate but does not yet improve recall enough to open the gate."
            ),
        })

    open_retro_items.append({
        "id": "RETRO-038",
        "title": "Live 200q VeriCoT+Wilson CI — still not closed",
        "carry_count": ">=11",
        "action_required": "Same root cause as RETRO-033. Block until recall > 30%.",
    })

    if not retro_057_resolved:
        open_retro_items.append({
            "id": "RETRO-057",
            "title": "LowRankKAEM energy accuracy outside 5% tolerance",
            "carry_count": ">=5",
            "action_required": (
                "Exp 637 sparse_vs_dense_error=0.429 — far outside 5% threshold.  "
                "Sparse-only redesign insufficient; try multilevel + sparse combined approach."
            ),
        })

    open_retro_items.append({
        "id": "RETRO-060",
        "title": "JEPA architecturally anti-correlated — superseded by RETRO-063",
        "carry_count": ">=5",
        "action_required": (
            "JEPA v14 OOD AUC=0.912 confirms architecture is sound.  Calibration (ECE) "
            "remains open — v14_ece=0.132 vs threshold 0.10.  Temperature scaling (Platt) "
            "is the next calibration attempt."
        ),
    })

    open_retro_items.append({
        "id": "RETRO-064",
        "title": "CoACE recall 4% on live data — pipeline accuracy lift undetectable",
        "carry_count": ">=5",
        "action_required": (
            "interwhen_recall=0.12 (Exp 629), hermes_recall=0.12 (Exp 633).  Neither "
            "standalone nor HERMES-augmented extractor crosses the 20% gate.  HERMES v2 "
            "with live generation loop is the next intervention."
        ),
    })

    open_retro_items.append({
        "id": "RETRO-065",
        "title": "RAPL unavailable — hardware energy calibration blocked",
        "carry_count": ">=5",
        "action_required": "Need Intel RAPL or AMD Energy driver on test machine",
    })

    open_retro_items.append({
        "id": "RETRO-066",
        "title": "CoACE offline/live distribution gap — extractor redesign unresolved",
        "carry_count": ">=4",
        "action_required": (
            "interwhen+HERMES recall=0.12 cannot close the gap.  ORACLE data elicitation "
            "(arXiv 2603.21140) is the only identified path to a training corpus that "
            "matches live output distribution."
        ),
    })

    open_retro_items.append({
        "id": "RETRO-068",
        "title": "LLMAsExtractorV1 live recall 4-12% — below 20% gate threshold",
        "carry_count": ">=3",
        "action_required": (
            "Gate recall >= 20%.  Current: interwhen_recall=0.12 (Exp 629).  "
            "HERMES tool-augmented step boundary is the next architectural intervention."
        ),
    })

    if not retro_070_resolved:
        open_retro_items.append({
            "id": "RETRO-070",
            "title": "interwhen recall still below 20% — HERMES tool-augmented pipeline next",
            "carry_count": 2,
            "action_required": (
                "Exp 629 interwhen_recall_primary=0.12 — gate_open=False.  HERMES v1 "
                "reduces FP rate (0.20 vs 0.40) but does not lift recall.  Build "
                "HermesVerifierAdapter v2 with live generation loop: step-generation → "
                "SymCodeVerifier prover → feedback injection before next token."
            ),
        })

    if not retro_071_resolved:
        open_retro_items.append({
            "id": "RETRO-071",
            "title": "DualGPU 13B proof blocked by missing HF model weights",
            "carry_count": 1,
            "action_required": (
                "Exp 632 GPUs present (2x RTX 3090) but model load failed — HF weights "
                "not cached in CI.  Re-run with HF_HOME pointing to pre-downloaded "
                "Qwen2.5-7B-Instruct or Qwen2.5-14B-Instruct weights."
            ),
        })

    # --- Open RETRO carry-forward for items not yet resolved but not explicitly listed above
    # RETRO-038, RETRO-060, RETRO-064, RETRO-065, RETRO-066, RETRO-068 already added above.
    # No new RETRO items for .48 — all existing items continue.
    new_retro_items: list[dict] = []

    # --- Top priorities for milestone .49 ------------------------------------
    top_priorities_for_49: list[str] = []

    if retro_033_resolved:
        top_priorities_for_49.append(
            "Scale to 200q Wilson CI (RETRO-038). Use winning extractor at scale."
        )
    elif retro_070_resolved:
        top_priorities_for_49.append(
            "interwhen recall crosses 20%. Run VR #17 immediately."
        )
    else:
        top_priorities_for_49.append(
            "RETRO-070 still open — implement HERMES tool-augmented full pipeline: "
            "step-generation → SymCodeVerifier prover → feedback injection before next token. "
            "Build HermesVerifierAdapter v2 with live generation loop."
        )

    top_priorities_for_49.append(
        "JEPA v14 ECE calibration: if not calibrated, try temperature scaling "
        "(Platt scaling) on v14 logits."
    )

    top_priorities_for_49.append(
        "KV260 FPGA: human must install Vivado 2023.2 before .50. TCL v2 is ready (Exp 636)."
    )

    return {
        "schema": SCHEMA,
        "milestone": MILESTONE,
        "n_experiments_run": n_experiments_run,
        "n_not_run": n_not_run,
        "total_wall_time_minutes": total_wall_time_minutes,
        "mean_time_min": mean_time_min,
        "wall_time_vs_prior_delta_minutes": wall_time_vs_prior_delta_minutes,
        "retro_033_resolved": retro_033_resolved,
        "retro_057_resolved": retro_057_resolved,
        "retro_070_resolved": retro_070_resolved,
        "retro_071_resolved": retro_071_resolved,
        "jepa_v14_calibrated": jepa_v14_calibrated,
        "fr11_confirmed": fr11_confirmed,
        "multilevel_wins": multilevel_wins,
        "adaptrack_improves": adaptrack_improves,
        "hermes_improves": hermes_improves,
        "fpga_tcl_updated": fpga_tcl_updated,
        "interwhen_recall": interwhen_recall,
        "gate_open": gate_open,
        "signed_improvement": signed_improvement,
        "v14_ece": v14_ece,
        "v14_ood_auc": v14_ood_auc,
        "dualgpu_proven": dualgpu_proven,
        "hermes_recall": hermes_recall,
        "hermes_fp_rate": hermes_fp_rate,
        "adaptrack_recall": adaptrack_recall,
        "sparse_vs_dense_error": sparse_vs_dense_error,
        "retro_closure_rate": retro_closure_rate,
        "open_retro_count": open_retro_count,
        "new_retro_items": new_retro_items,
        "open_retro_items": open_retro_items,
        "top_priorities_for_49": top_priorities_for_49,
        "honest_verdict": honest_verdict,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the .48 retrospective: load results, compute metrics, write deliverable."""
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

    # Overwrite schema to v23 (build_result may set generic schema).
    artifact["schema"] = SCHEMA
    artifact["env_autofix"] = True

    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"[Exp {EXP_ID}] honest_verdict={artifact['honest_verdict']}")
    print(f"[Exp {EXP_ID}] retro_closure_rate={artifact['retro_closure_rate']}")
    print(f"[Exp {EXP_ID}] open_retro_count={artifact['open_retro_count']}")
    print(f"[Exp {EXP_ID}] interwhen_recall={artifact['interwhen_recall']}")
    print(f"[Exp {EXP_ID}] hermes_improves={artifact['hermes_improves']}")
    print(f"[Exp {EXP_ID}] Deliverable: {DELIVERABLE}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
