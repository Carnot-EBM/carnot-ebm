#!/usr/bin/env python3
"""Experiment 665: Milestone 2026.04.50 Operational Retrospective.

**Researcher summary:**
    Milestone 2026.04.50 ran Exps 652-664 under themes:
    "Prompt-Injection Safety, StructuredEquationForcer (RETRO-070), HERMES v2 Structured
    Live, Ensemble Recall Gate v3, Live VR Attempt #18, JEPA v14 Cascade Deploy,
    SpecGuard Step Verifier, FR-11 Tier 2 Cross-Session Relay, LSEBMCL Constraint Memory,
    KV260 N=64 FPGA Benchmark, Ising v3 RTL, HALP Pre-Generative Probe, DualGPU Retrain".

    Key questions answered for .50:

    prompt_injection_auroc (Exp 652): classifier_auroc=0.9262 >= 0.90 threshold.  CRITERION MET.
        KAN classifier distilled from gpt-oss-safeguard-20b, 3432 params, 19.7ms latency.
        Ready for production cascade as Tier 0f safeguard.

    equation_forcer (Exp 653): detection_rate_on_forced=1.0 (100% on structured format).
        CRITERION MET.  Free-form detection_rate=0.0 — forcer only helps when model cooperates.
        StructuredEquationForcer integrated at Tier 2.6 as generation-layer intervention.

    hermes_v2_structured_recall (Exp 654): recall=0.20 vs threshold 0.30.  CRITERION NOT MET.
        Improvement from baseline 0.12 to 0.20 (+0.08) is real but insufficient.
        FP rate=0.40 — too many false positives at this recall level.

    ensemble_gate_v3 (Exp 655): ensemble_recall=0.224 vs gate threshold 0.30.  GATE CLOSED.
        Combined SymCode=0.12, Structured=0.20, Causal=0.36, Ensemble=0.224.
        Ensemble aggregation regression vs .49 (was 0.36, now 0.224) due to new weighted formula.

    retro_033 (Exp 656): VR #18 blocked because gate closed.  RETRO-033 STILL OPEN.
        18 consecutive failed attempts.  Root cause: structured recall gate not met.
        CI stub mode prevents real violation generation.  Requires live corpus > 0.30 recall.

    jepa_v14_cascade (Exp 657): BLOCKED on Exp 646 dependency not satisfied in CI context.
        cascade_ece and auc_delta not measured.  JEPA v14 Platt-calibrated weights exist
        from .49 but cascade integration not wired.

    specguard_viable (Exp 658): specguard_auc=0.216 vs threshold 0.70.  CRITERION NOT MET.
        TP=15, FP=10, TN=10, FN=65.  High false-negative rate.  Tier 0f SpecGuard not viable.

    fr11_real_violations (Exp 659): fr11_real_violations_confirmed=True.  CRITERION MET.
        3 cross-session templates added, 0.0 FP rate.  Violations sourced from Exp 656
        synthetic patterns (even though VR was blocked).

    lsebmcl_no_forgetting (Exp 660): forgetting_rate=0.0 across 3 sessions.  CRITERION MET.
        EBM replay mechanism working.  Constraint memory stable.

    kv260_n64_hardware (Exp 661): PARTIAL — DFX Manager timeout (server 192.168.51.98).
        hardware_latency_us not measured.  Bitstream deployed and verified locally.
        FPGA board present (KV260 revB, XRT 2.18.0) but AXI helper timeout.  RETRO-072 OPEN.

    ising_v3_rtl (Exp 662): rtl_written=True, 295 lines Verilog, h_ema register.  CRITERION MET.
        Testbench written.  Vivado not available for synthesis; RTL ready for future synthesis pass.

    halp_viable (Exp 663): halp_auc=0.442 vs threshold 0.75.  CRITERION NOT MET.
        281 train / 69 test samples.  Pre-generative hallucination probe needs more training data.

    dualgpu_parallel (Exp 664): peak_gpu1_util=0.0 vs threshold 50%.  CRITERION NOT MET.
        EORM on cuda:0 and JEPA on cuda:1 both ran 50 steps but GPU1 showed 0% utilization.
        ThreadPoolExecutor scheduling may not trigger real concurrent CUDA kernels.
        RETRO-071 UNRESOLVED.

    Headline: 5 of 13 criteria met (38.5%).  RETRO-033 still blocked at attempt #18.
    Prompt-injection classifier is milestone's headline win (AUROC 0.9262).
    DualGPU parallelism, SpecGuard viability, HALP, and KV260 live benchmark all failed.
    No RETROs closed this milestone.

Spec: REQ-INFRA-058, REQ-INFRA-076, SCENARIO-INFRA-069, SCENARIO-INFRA-075
"""

from __future__ import annotations

# apply_env_autofix MUST be called first, before any other carnot import.
# It injects CARNOT_FORCE_LIVE and related env vars so downstream pipeline code
# reads them correctly at import time, not after module-level constants are evaluated.
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

EXP_ID = 665
TITLE = "Milestone 2026.04.50 Retrospective"
DELIVERABLE = "results/experiment_665_retro_2026_04_50.json"
MILESTONE = "2026.04.50"
SCHEMA = "carnot.retro.v1"

# Cumulative project wall time in minutes as of the end of milestone .49.
# Source: Exp 651 retro context (509 experiments × 8.6 min/exp average).
PRIOR_CUMULATIVE_WALL_TIME_MINUTES = 4380.0

# Upstream experiment result files for milestone .50.
# Exp 665 (this retro) is the final entry and is computed here, not loaded.
_MILESTONE_RESULTS = [
    ("652", "results/experiment_652_prompt_injection_kan.json"),
    ("653", "results/experiment_653_equation_forcer.json"),
    ("654", "results/experiment_654_hermes_v2_structured.json"),
    ("655", "results/experiment_655_ensemble_gate_v3.json"),
    ("656", "results/experiment_656_live_vr_attempt_18.json"),
    ("657", "results/experiment_657_jepa_cascade_deploy.json"),
    ("658", "results/experiment_658_specguard_verifier.json"),
    ("659", "results/experiment_659_tier2_fr11_relay.json"),
    ("660", "results/experiment_660_lsebmcl_memory.json"),
    ("661", "results/experiment_661_kv260_n64_benchmark.json"),
    ("662", "results/experiment_662_ising_v3_rtl.json"),
    ("663", "results/experiment_663_halp_probe.json"),
    ("664", "results/experiment_664_dualgpu_retrain.json"),
]

# RETROs open at the START of milestone .50 (carry-forward from .49 retro, Exp 651).
# Source: Exp 651 open_retro_count=9, plus RETRO-072 filed at .50 planning.
_RETROS_OPEN_AT_MILESTONE_START = [
    "RETRO-031",
    "RETRO-033",
    "RETRO-038",
    "RETRO-057",
    "RETRO-064",
    "RETRO-065",
    "RETRO-066",
    "RETRO-068",
    "RETRO-071",
    "RETRO-072",
    "RETRO-CRITICAL",
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
    """Load all 13 upstream results and compute .50 retrospective metrics.

    Why this is a pure function: the test suite can monkeypatch _REPO_ROOT and
    call compute_retro() with controlled fake result files, verifying every
    boolean branch without touching the real filesystem or running actual experiments.
    """
    # --- Load all upstream results ----------------------------------------
    results = {}
    for exp_id_str, rel_path in _MILESTONE_RESULTS:
        results[exp_id_str] = _load_result(rel_path)

    exp652 = results["652"]
    exp653 = results["653"]
    exp654 = results["654"]
    exp655 = results["655"]
    exp656 = results["656"]
    exp657 = results["657"]
    exp658 = results["658"]
    exp659 = results["659"]
    exp660 = results["660"]
    exp661 = results["661"]
    exp662 = results["662"]
    exp663 = results["663"]
    exp664 = results["664"]

    # --- Extract raw metric values ----------------------------------------
    # Exp 652: KAN classifier AUROC (key is classifier_auroc, not auroc).
    classifier_auroc: float = float(exp652.get("classifier_auroc", 0.0))

    # Exp 653: StructuredEquationForcer detection rate on forced format.
    detection_rate_on_forced: float = float(exp653.get("detection_rate_on_forced", 0.0))

    # Exp 654: HERMES v2 structured recall on 25-question live sample.
    hermes_v2_structured_recall: float = float(
        exp654.get("hermes_v2_structured_recall", 0.0)
    )

    # Exp 655: Ensemble gate status — recall across all extractors combined.
    ensemble_recall: float = float(exp655.get("ensemble_recall", 0.0))
    gate_open: bool = bool(exp655.get("gate_open", ensemble_recall >= 0.30))

    # Exp 656: VR attempt #18 — signed improvement and RETRO-033 closure flag.
    signed_improvement: float = float(exp656.get("signed_improvement", 0.0) or 0.0)
    retro_033_resolved: bool = bool(exp656.get("retro_033_resolved", False))

    # Exp 657: JEPA v14 cascade deployment metrics.
    # cascade_ece=None if blocked; treat None as not meeting target.
    cascade_ece_raw = exp657.get("cascade_ece")
    cascade_ece: float = float(cascade_ece_raw) if cascade_ece_raw is not None else 1.0
    auc_delta_raw = exp657.get("auc_delta")
    auc_delta: float = float(auc_delta_raw) if auc_delta_raw is not None else 1.0

    # Exp 658: SpecGuard AUC and viability flag.
    specguard_auc: float = float(exp658.get("specguard_auc", 0.0))

    # Exp 659: FR-11 cross-session relay confirmation.
    fr11_real_violations_confirmed: bool = bool(
        exp659.get("fr11_real_violations_confirmed", False)
    )

    # Exp 660: LSEBMCL forgetting rate.
    forgetting_rate: float = float(exp660.get("forgetting_rate", 1.0))
    lsebmcl_no_forgetting: bool = bool(
        exp660.get("lsebmcl_no_forgetting", forgetting_rate < 0.05)
    )

    # Exp 661: KV260 FPGA hardware latency — absent if benchmark timed out.
    hardware_latency_us_raw = exp661.get("hardware_latency_us")
    hardware_latency_us: float = (
        float(hardware_latency_us_raw) if hardware_latency_us_raw is not None else float("inf")
    )

    # Exp 662: Ising v3 RTL written flag.
    rtl_written: bool = bool(exp662.get("rtl_written", False))

    # Exp 663: HALP pre-generative probe AUC.
    halp_auc: float = float(exp663.get("halp_auc", 0.0))

    # Exp 664: DualGPU peak GPU1 utilization.
    peak_gpu1_util: float = float(exp664.get("peak_gpu1_util", 0.0))
    retro_071_resolved_664: bool = bool(exp664.get("retro_071_resolved", False))

    # --- Evaluate 13 success criteria (per research-roadmap-v50.md) ------
    criteria: dict[str, bool] = {
        # Criterion 1: Prompt-injection KAN classifier AUROC >= 0.90
        "prompt_injection_auroc_met": classifier_auroc >= 0.90,
        # Criterion 2: StructuredEquationForcer detection rate on forced format == 1.0
        "equation_forcer_parses_100pct": detection_rate_on_forced == 1.0,
        # Criterion 3: HERMES v2 structured recall >= 0.30
        "hermes_v2_structured_recall": hermes_v2_structured_recall >= 0.30,
        # Criterion 4: Ensemble gate v3 open (recall >= 0.30)
        "ensemble_gate_v3_open": gate_open,
        # Criterion 5: RETRO-033 resolved (signed_improvement > 0 in live VR attempt #18)
        "retro_033_resolved": retro_033_resolved,
        # Criterion 6: JEPA v14 deployed in cascade (cascade_ece < 0.10 AND auc_delta <= 0.02)
        "jepa_v14_deployed": cascade_ece < 0.10 and auc_delta <= 0.02,
        # Criterion 7: SpecGuard step verifier viable as Tier 0f (AUC >= 0.70)
        "specguard_viable": specguard_auc >= 0.70,
        # Criterion 8: FR-11 real violations confirmed in cross-session relay
        "fr11_real_violations": fr11_real_violations_confirmed,
        # Criterion 9: LSEBMCL constraint memory: forgetting rate < 0.05
        "lsebmcl_no_forgetting": lsebmcl_no_forgetting,
        # Criterion 10: KV260 N=64 hardware latency < 100 µs
        "kv260_n64_hardware": hardware_latency_us < 100.0,
        # Criterion 11: Ising sampler v3 RTL written (Vivado synthesis not required)
        "ising_v3_rtl_written": rtl_written,
        # Criterion 12: HALP pre-generative probe viable as Tier 0g (AUC >= 0.75)
        "halp_viable": halp_auc >= 0.75,
        # Criterion 13: DualGPU parallel training proven (peak GPU1 util > 50%)
        "dualgpu_parallel_proven": peak_gpu1_util > 50.0,
    }

    n_criteria_met: int = sum(criteria.values())
    n_criteria_total: int = 13
    milestone_success_rate: float = round(n_criteria_met / n_criteria_total, 4)

    # --- Wall time aggregation -------------------------------------------
    # Sum duration_s across all 13 upstream experiments (not this retro).
    upstream_duration_s: float = sum(r.get("duration_s", 0.0) for r in results.values())
    # Retro itself (Exp 665) is negligible; we do not add a placeholder.
    wall_time_50_delta_min: float = round(upstream_duration_s / 60.0, 3)

    # Cumulative project wall time INCLUDING .50 experiments.
    # Why: the comparison context is cumulative (4380 min for 509 exps through .49).
    wall_time_49: float = PRIOR_CUMULATIVE_WALL_TIME_MINUTES
    wall_time_50: float = round(wall_time_49 + wall_time_50_delta_min, 3)
    wall_time_delta: float = round(wall_time_50_delta_min, 3)
    wall_time_pct_change: float = round(wall_time_delta / wall_time_49 * 100.0, 4)

    n_experiments_run: int = len(_MILESTONE_RESULTS) + 1  # +1 for this retro
    n_not_run: int = sum(1 for r in results.values() if not r)
    n_experiments_run -= n_not_run

    # --- RETRO status updates (per task spec) ----------------------------
    # RETRO-033: VR #18 blocked because gate closed (not even attempted).
    retro_033_status = (
        "resolved" if retro_033_resolved else "attempt_18_failed_open"
    )
    # RETRO-057: No .50 action taken; filed for .51 multilevel redesign.
    retro_057_status = "filed_for_51_multilevel_needed"
    # RETRO-070: requires BOTH equation_forcer_parses_100pct AND hermes_v2 recall >= 0.30.
    retro_070_status = (
        "resolved"
        if (criteria["equation_forcer_parses_100pct"] and criteria["hermes_v2_structured_recall"])
        else "equation_forcer_integrated_recall_still_below_threshold"
    )
    # RETRO-071: DualGPU GPU1 util 0% — ThreadPoolExecutor not triggering real CUDA concurrency.
    retro_071_status = "resolved" if criteria["dualgpu_parallel_proven"] else "unresolved"
    # RETRO-072: KV260 DFX Manager timeout — hardware_latency_us not measured.
    retro_072_status = "resolved" if criteria["kv260_n64_hardware"] else "unresolved"
    # RETRO-CRITICAL: Exclusion manifest wired in Exps 549/575/589/601/614.
    # Exp 651 manifest_wired=True.  Still requires human confirmation that conductor
    # calls _task_is_excluded() on every scheduling iteration.
    retro_critical_status = "wired_confirmed_prior_milestones_human_verify_pending"

    retro_statuses: dict[str, str] = {
        "RETRO-033": retro_033_status,
        "RETRO-057": retro_057_status,
        "RETRO-070": retro_070_status,
        "RETRO-071": retro_071_status,
        "RETRO-072": retro_072_status,
        "RETRO-CRITICAL": retro_critical_status,
    }

    # --- Open RETROs for milestone .51 -----------------------------------
    open_retros_for_51: list[str] = []

    open_retros_for_51.append("RETRO-031")  # carry: unverified closure

    if not retro_033_resolved:
        open_retros_for_51.append("RETRO-033")  # 18 failed VR attempts

    open_retros_for_51.append("RETRO-038")  # carry: 200q Wilson CI blocked by RETRO-033

    open_retros_for_51.append("RETRO-057")  # carry: KAEM multilevel filed for .51

    open_retros_for_51.append("RETRO-064")  # carry: extraction recall below pipeline threshold

    open_retros_for_51.append("RETRO-065")  # carry: RAPL unavailable for energy calibration

    open_retros_for_51.append("RETRO-066")  # carry: extractor offline/live distribution gap

    open_retros_for_51.append("RETRO-068")  # carry: LLMAsExtractorV1 recall 4-12%

    if not criteria["hermes_v2_structured_recall"]:
        open_retros_for_51.append("RETRO-070")  # structured recall 0.20 below 0.30 threshold

    if not criteria["dualgpu_parallel_proven"]:
        open_retros_for_51.append("RETRO-071")  # GPU1 0% util, ThreadPoolExecutor insufficient

    if not criteria["kv260_n64_hardware"]:
        open_retros_for_51.append("RETRO-072")  # KV260 DFX Manager network timeout

    open_retros_for_51.append("RETRO-CRITICAL")  # human must verify exclusion manifest live

    # --- Honest verdict ---------------------------------------------------
    if n_criteria_met == n_criteria_total:
        honest_verdict = "all_13_criteria_met_milestone_complete"
    elif n_criteria_met >= 10:
        honest_verdict = f"strong_milestone_{n_criteria_met}_of_13_criteria_met"
    elif retro_033_resolved:
        honest_verdict = f"retro_033_finally_closed_{n_criteria_met}_of_13_criteria_met"
    else:
        honest_verdict = (
            f"partial_milestone_{n_criteria_met}_of_13_criteria_met_"
            f"retro_033_still_open_after_18_attempts"
        )

    return {
        "schema": SCHEMA,
        "milestone": MILESTONE,
        "n_criteria_met": n_criteria_met,
        "n_criteria_total": n_criteria_total,
        "milestone_success_rate": milestone_success_rate,
        "criteria": criteria,
        "wall_time_50": wall_time_50,
        "wall_time_49": wall_time_49,
        "wall_time_delta": wall_time_delta,
        "wall_time_pct_change": wall_time_pct_change,
        "n_experiments_run": n_experiments_run,
        "n_not_run": n_not_run,
        "retro_statuses": retro_statuses,
        "open_retros_for_51": open_retros_for_51,
        "honest_verdict": honest_verdict,
        # Raw metrics for traceability
        "classifier_auroc": classifier_auroc,
        "detection_rate_on_forced": detection_rate_on_forced,
        "hermes_v2_structured_recall": hermes_v2_structured_recall,
        "ensemble_recall": ensemble_recall,
        "signed_improvement": signed_improvement,
        "specguard_auc": specguard_auc,
        "halp_auc": halp_auc,
        "peak_gpu1_util": peak_gpu1_util,
        "forgetting_rate": forgetting_rate,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the .50 retrospective: load results, compute metrics, write deliverable."""
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

    # Override schema and milestone — build_result may set generic schema.
    artifact["schema"] = SCHEMA
    artifact["milestone"] = MILESTONE
    artifact["env_autofix"] = True

    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"[Exp {EXP_ID}] honest_verdict={artifact['honest_verdict']}")
    print(f"[Exp {EXP_ID}] n_criteria_met={artifact['n_criteria_met']}/{artifact['n_criteria_total']}")
    print(f"[Exp {EXP_ID}] milestone_success_rate={artifact['milestone_success_rate']}")
    print(f"[Exp {EXP_ID}] wall_time_50={artifact['wall_time_50']} min (delta={artifact['wall_time_delta']} min)")
    print(f"[Exp {EXP_ID}] open_retros_for_51={artifact['open_retros_for_51']}")
    print(f"[Exp {EXP_ID}] Deliverable: {DELIVERABLE}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
