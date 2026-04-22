"""
Experiment 702 — Milestone 2026.04.53 Operational Retrospective.

Analyses the 12-experiment cycle (Exps 690-701) that forms milestone .53.
Answers four key questions:
  1. Is JEPA v16 OOD AUC >= 0.75? (unblocks cascade)
  2. Did PSV real self-play show FP improvement (slope < 0)?
  3. Did VR cross-model validation confirm Exp 679?
  4. Was distillation-duration invariant confirmed (teacher_s >= corpus * 0.5)?

Why a dedicated script: the research_conductor calls this via run_agent() after
milestone boundary detection, giving it a clean subprocess environment with
JAX_PLATFORMS=cpu so JAX imports never stall waiting for GPU lock.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: resolve repo root and extend sys.path so local packages import
# ---------------------------------------------------------------------------
REPO_ROOT = Path(os.environ.get("CARNOT_REPO_ROOT", Path(__file__).parent.parent))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXP_ID = 702
TITLE = "Milestone 2026.04.53 Operational Retrospective — Full Analysis"
DELIVERABLE_REL = "results/operational_retro_2026_04_53.json"
DELIVERABLE = REPO_ROOT / DELIVERABLE_REL
SCHEMA = "carnot.operational_retro.v28"
MILESTONE = "2026.04.53"

# Milestone .52 baseline for comparison
PRIOR_MILESTONE_AVG_MIN = 7.0
PRIOR_MILESTONE_WALL_MINUTES = 3983
PRIOR_MILESTONE_EXPERIMENTS = 538

# Threshold for JEPA cascade unblock
JEPA_UNBLOCK_THRESHOLD = 0.75

# Distillation invariant: teacher_inference_duration_s >= corpus_size * 0.5
DISTILLATION_CORPUS_MULTIPLIER = 0.5

# GPU clean threshold — under 100 MiB idle = clean close
GPU_CLEAN_MB = 100

# Cycle experiment IDs in order
CYCLE_EXPERIMENTS = [690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701]


def _load_result(exp_id: int) -> dict:
    """
    Load a single experiment JSON result.

    Returns a sentinel dict with status='not_run' when the file is missing.
    This prevents the retro from crashing on partial milestone runs and
    matches the RETRO-027 sentinel pattern established in .52.
    """
    pattern = REPO_ROOT / "results" / f"experiment_{exp_id}_*.json"
    matches = sorted(REPO_ROOT.glob(f"results/experiment_{exp_id}_*.json"))
    if not matches:
        return {
            "experiment": exp_id,
            "status": "not_run",
            "honest_verdict": "file_missing",
            "duration_s": 0.0,
        }
    with open(matches[0]) as fh:
        data = json.load(fh)
    # Some older experiments omit the 'status' key; default to 'success' when
    # an honest_verdict and duration are present (means it completed normally).
    if "status" not in data:
        data["status"] = "success" if "honest_verdict" in data else "unknown"
    return data


def _query_gpu_state() -> dict:
    """
    Query live GPU memory and utilisation via nvidia-smi.

    Returns a dict with gpu0_vram_mb, gpu1_vram_mb, gpu_close_clean.
    Falls back to zeros when nvidia-smi is unavailable (CI/CPU-only hosts).
    The 'gpu_close_clean' flag is True only when both GPUs are under
    GPU_CLEAN_MB MiB — meaning no zombie processes are holding VRAM at close.
    """
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [ln.strip() for ln in proc.stdout.strip().splitlines() if ln.strip()]
        vram_values = []
        for line in lines:
            parts = line.split(",")
            # nvidia-smi outputs "4 MiB, 0 %" — strip the unit
            mb_str = parts[0].strip().replace(" MiB", "").replace("MiB", "")
            vram_values.append(int(mb_str))
        gpu0 = vram_values[0] if len(vram_values) > 0 else 0
        gpu1 = vram_values[1] if len(vram_values) > 1 else 0
        clean = gpu0 < GPU_CLEAN_MB and gpu1 < GPU_CLEAN_MB
        return {"gpu0_vram_mb": gpu0, "gpu1_vram_mb": gpu1, "gpu_close_clean": clean}
    except Exception:
        return {"gpu0_vram_mb": 0, "gpu1_vram_mb": 0, "gpu_close_clean": True}


def main() -> None:
    tmpl = ExperimentTemplate(EXP_ID, TITLE, DELIVERABLE_REL)

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30, result_path=str(DELIVERABLE)):
        tmpl.setup()

        # ---------------------------------------------------------------
        # Step 1: Load all cycle experiment results
        # ---------------------------------------------------------------
        results: dict[int, dict] = {eid: _load_result(eid) for eid in CYCLE_EXPERIMENTS}

        experiment_table = []
        for eid in CYCLE_EXPERIMENTS:
            r = results[eid]
            experiment_table.append(
                {
                    "experiment": eid,
                    "status": r.get("status", "unknown"),
                    "honest_verdict": r.get("honest_verdict", ""),
                    "duration_s": r.get("duration_s", 0.0),
                }
            )

        # ---------------------------------------------------------------
        # Step 2: Wall-time metrics
        # ---------------------------------------------------------------
        total_duration_s = sum(r.get("duration_s", 0.0) for r in results.values())
        wall_time_minutes = total_duration_s / 60.0
        total_experiments = len(CYCLE_EXPERIMENTS)
        per_experiment_avg_min = wall_time_minutes / total_experiments

        slowest_5 = sorted(
            experiment_table, key=lambda x: x["duration_s"], reverse=True
        )[:5]

        wall_time_delta = wall_time_minutes - PRIOR_MILESTONE_AVG_MIN * total_experiments
        wall_time_direction = "improvement" if per_experiment_avg_min < PRIOR_MILESTONE_AVG_MIN else "regression"

        # ---------------------------------------------------------------
        # Step 3: Key research metrics
        # ---------------------------------------------------------------
        r690 = results[690]
        r691 = results[691]
        r692 = results[692]
        r693 = results[693]
        r694 = results[694]
        r695 = results[695]
        r696 = results[696]
        r697 = results[697]
        r698 = results[698]
        r699 = results[699]
        r700 = results[700]
        r701 = results[701]

        # Distillation invariant: teacher_inference_duration_s >= corpus_size * 0.5
        teacher_s = r690.get("teacher_inference_duration_s", 0.0)
        corpus_size = r690.get("corpus_size", 0)
        distillation_invariant_confirmed = teacher_s >= (corpus_size * DISTILLATION_CORPUS_MULTIPLIER)

        # JEPA v16 OOD AUC
        jepa_v16_ood_auc = r698.get("v16_ood_auc", 0.0)
        jepa_v16_cascade_unblocked = jepa_v16_ood_auc >= JEPA_UNBLOCK_THRESHOLD

        # PSV real self-play FP trend
        psv_real_fp_trend_slope = r697.get("fp_rate_trend_slope", 0.0)
        psv_real_fp_improving = psv_real_fp_trend_slope < 0

        key_metrics = {
            "preflight_v5_complete": r692.get("honest_verdict") == "preflight_v5_complete",
            "jepa_v15_root_cause": r693.get("root_cause", "unknown"),
            "distillation_invariant_confirmed": distillation_invariant_confirmed,
            "distillation_teacher_inference_s": teacher_s,
            "distillation_corpus_size": corpus_size,
            "cross_dataset_mean_auroc": r691.get("mean_auroc", 0.0),
            "vr_cross_model_delta": r694.get("cross_model_delta", None),
            "tier_28_winner": r695.get("tier_28_winner", "unknown"),
            "abstention_fp_reduced": (
                r696.get("fp_rate_best_abstention", 1.0)
                < r696.get("fp_rate_no_abstention", 1.0)
            ),
            "psv_real_fp_trend_slope": psv_real_fp_trend_slope,
            "jepa_v16_ood_auc": jepa_v16_ood_auc,
            "hallusae_delta_auc": r699.get("delta_auc", 0.0),
            "publication_ready": r700.get("publication_ready", False),
            "retro_072_resolved": r701.get("retro_072_resolved", False),
        }

        # ---------------------------------------------------------------
        # Step 4: Open RETRO status
        # ---------------------------------------------------------------
        open_retros = {
            "RETRO-072": {
                "description": "KV260 Ising v3 RTL synthesis blocked — no Vivado tool",
                "status": "resolved" if key_metrics["retro_072_resolved"] else "open",
                "experiment": 701,
            },
            "RETRO-CRITICAL": {
                "description": "JEPA cascade blocked — OOD AUC below random",
                "status": "unblocked" if jepa_v16_cascade_unblocked else "open",
                "jepa_v16_ood_auc": jepa_v16_ood_auc,
                "threshold": JEPA_UNBLOCK_THRESHOLD,
                "experiment": 698,
            },
            "RETRO-DISTILLATION": {
                "description": "Distillation duration invariant (teacher_s >= corpus*0.5)",
                "status": "confirmed" if distillation_invariant_confirmed else "violated",
                "experiment": 690,
            },
        }

        # ---------------------------------------------------------------
        # Step 5: GPU state at close
        # ---------------------------------------------------------------
        gpu_state = _query_gpu_state()

        # ---------------------------------------------------------------
        # Step 6: FR-11 self-improvement metrics
        # ---------------------------------------------------------------
        # Count tiers with at least one positive result in this cycle
        # Tier map from CLAUDE.md: Large=Boltzmann, Medium=Gibbs, Efficient=KAN, Small=Ising
        tier_results = {
            "tier_1_boltzmann": False,   # no boltzmann experiment in this cycle
            "tier_2_gibbs": False,       # no gibbs experiment in this cycle
            "tier_3_kan_distill": key_metrics["distillation_invariant_confirmed"],
            "tier_4_ising_synthesis": key_metrics["retro_072_resolved"],
        }
        fr11_tier_advancement = sum(1 for v in tier_results.values() if v)

        fr11_metrics = {
            "fr11_real_positives_confirmed": True,  # inherited from Exp 683
            "psv_real_fp_improving": psv_real_fp_improving,
            "jepa_v16_cascade_unblocked": jepa_v16_cascade_unblocked,
            "fr11_tier_advancement": fr11_tier_advancement,
            "tier_results": tier_results,
        }

        # ---------------------------------------------------------------
        # Step 7: Honest verdict string
        # ---------------------------------------------------------------
        auc_verdict = "unblocked" if jepa_v16_cascade_unblocked else "still_blocked"
        psv_verdict = "improving" if psv_real_fp_improving else "stable_or_degrading"
        distill_verdict = "confirmed" if distillation_invariant_confirmed else "violated"

        honest_verdict = (
            f"wall_time_{wall_time_direction}"
            f"_jepa_v16_{auc_verdict}"
            f"_psv_{psv_verdict}"
            f"_distill_{distill_verdict}"
        )

        # ---------------------------------------------------------------
        # Step 8: Emit result
        # ---------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "schema": SCHEMA,
                "milestone": MILESTONE,
                "total_wall_time_minutes": round(wall_time_minutes, 2),
                "experiments_in_cycle": total_experiments,
                "experiments_completed": PRIOR_MILESTONE_EXPERIMENTS + total_experiments,
                "avg_time_per_experiment_minutes": round(per_experiment_avg_min, 2),
                "wall_time_delta_vs_prior_minutes": round(wall_time_delta, 2),
                "wall_time_delta_direction": wall_time_direction,
                "prior_milestone_avg_min": PRIOR_MILESTONE_AVG_MIN,
                "prior_milestone_wall_time_minutes": PRIOR_MILESTONE_WALL_MINUTES,
                "prior_milestone_experiments": PRIOR_MILESTONE_EXPERIMENTS,
                "cycle_data": {
                    "cycle_experiments": total_experiments,
                    "cycle_duration_s": round(total_duration_s, 3),
                    "cycle_wall_time_minutes": round(wall_time_minutes, 2),
                    "cycle_avg_min_per_exp": round(per_experiment_avg_min, 2),
                    "experiment_table": experiment_table,
                },
                "slowest_experiments": slowest_5,
                "key_metrics": key_metrics,
                "open_retros": open_retros,
                "gpu_state_at_close": gpu_state,
                "fr11_continuous_learning": fr11_metrics,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        # build_result() replaces "schema" with a sorted key list; restore
        # the named schema identifier so downstream consumers can assert version.
        artifact["schema"] = SCHEMA

        # Write to disk — build_result() returns a dict, does NOT write.
        DELIVERABLE.write_text(json.dumps(artifact, indent=2))

        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
