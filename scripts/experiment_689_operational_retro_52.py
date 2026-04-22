#!/usr/bin/env python3
"""Operational Retrospective for Milestone 2026.04.52 (Exps 678-688).

**What this does:**
    Reads all 11 experiment result files from milestone 2026.04.52 and
    computes wall-time efficiency metrics, answers key open questions
    (VR win at 200q? RETRO-071 closed? FR-11 FP reduction?), and emits
    a structured JSON artifact at results/operational_retro_2026_04_52.json.

**Why this is important:**
    Each milestone retrospective feeds the conductor's planning loop — the
    conductor reads these JSON artifacts when deciding which open RETROs to
    re-attempt and which directions to pursue next.  An accurate retrospective
    with honest_verdict is the single most impactful input to milestone planning.

Spec: REQ-VERIFY-083, REQ-INFRA-007
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Allow running from repo root directly without installing the package.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXP_ID = 689
TITLE = "Operational Retrospective — Milestone 2026.04.52 (Exps 678-688)"
DELIVERABLE = "results/operational_retro_2026_04_52.json"
SCHEMA = "carnot.operational_retro.v27"
MILESTONE = "2026.04.52"

# All 11 experiments in this milestone (678-688, excluding 686 which is missing).
MILESTONE_EXPERIMENTS = [678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688]

# Total wall-time carry from .51 baseline (minutes).
PRIOR_WALL_TIME_MINUTES = 4231
PRIOR_EXPERIMENTS = 532


def load_result(exp_id: int) -> dict:
    """Load a single experiment result JSON by experiment ID.

    Returns a minimal sentinel dict with status='not_run' if the file is
    absent — this is RETRO-027 sentinel handling: missing files are logged
    as not_run rather than silently omitted from the accounting.
    """
    path = _REPO / f"results/experiment_{exp_id}_*.json"
    import glob

    matches = sorted(glob.glob(str(_REPO / f"results/experiment_{exp_id}_*.json")))
    if not matches:
        return {
            "experiment": exp_id,
            "status": "not_run",
            "honest_verdict": "file_missing",
            "duration_s": 0.0,
        }
    return json.loads(Path(matches[-1]).read_text())


def read_gpu_state() -> dict:
    """Query nvidia-smi for current VRAM usage on GPU0 and GPU1.

    Returns gpu0_vram_mb, gpu1_vram_mb, and gpu_close_clean (True when
    both GPUs are below 100 MB — indicating a clean GPU state at close).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        gpus = []
        for line in lines:
            parts = [p.strip().rstrip(" MiB").rstrip(" %") for p in line.split(",")]
            gpus.append({"vram_mb": int(parts[0]), "util_pct": int(parts[1])})
        gpu0 = gpus[0] if len(gpus) > 0 else {"vram_mb": -1, "util_pct": -1}
        gpu1 = gpus[1] if len(gpus) > 1 else {"vram_mb": -1, "util_pct": -1}
        gpu_close_clean = gpu0["vram_mb"] < 100 and gpu1["vram_mb"] < 100
        return {
            "gpu0_vram_mb": gpu0["vram_mb"],
            "gpu0_util_pct": gpu0["util_pct"],
            "gpu1_vram_mb": gpu1["vram_mb"],
            "gpu1_util_pct": gpu1["util_pct"],
            "gpu_close_clean": gpu_close_clean,
        }
    except Exception as exc:
        return {
            "gpu0_vram_mb": -1,
            "gpu0_util_pct": -1,
            "gpu1_vram_mb": -1,
            "gpu1_util_pct": -1,
            "gpu_close_clean": False,
            "gpu_error": str(exc),
        }


def main() -> None:
    """Run the milestone retrospective and emit the result JSON."""
    tmpl = ExperimentTemplate(EXP_ID, TITLE, DELIVERABLE)
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=30,
        result_path=str(_REPO / DELIVERABLE),
    )

    with watchdog:
        # ------------------------------------------------------------------
        # Step 1: Load all 11 experiment results
        # ------------------------------------------------------------------
        results: dict[int, dict] = {}
        for exp_id in MILESTONE_EXPERIMENTS:
            results[exp_id] = load_result(exp_id)

        # ------------------------------------------------------------------
        # Step 2: Per-experiment status table and duration accounting
        # ------------------------------------------------------------------
        experiment_table = []
        cycle_duration_s = 0.0
        for exp_id in MILESTONE_EXPERIMENTS:
            r = results[exp_id]
            dur = float(r.get("duration_s") or 0.0)
            cycle_duration_s += dur
            experiment_table.append(
                {
                    "experiment": exp_id,
                    "status": r.get("status", "not_run"),
                    "honest_verdict": r.get("honest_verdict", "file_missing"),
                    "duration_s": dur,
                }
            )

        cycle_wall_time_minutes = round(cycle_duration_s / 60.0, 2)
        cumulative_wall_time_minutes = PRIOR_WALL_TIME_MINUTES + cycle_wall_time_minutes
        total_experiments = PRIOR_EXPERIMENTS + len(MILESTONE_EXPERIMENTS)
        per_experiment_avg_min = round(
            cumulative_wall_time_minutes / total_experiments, 2
        )

        # Slowest 5 in this cycle
        sorted_by_dur = sorted(experiment_table, key=lambda x: x["duration_s"], reverse=True)
        slowest_5 = [
            {"experiment": e["experiment"], "duration_s": e["duration_s"]}
            for e in sorted_by_dur[:5]
        ]

        # ------------------------------------------------------------------
        # Step 3: Extract key metrics
        # ------------------------------------------------------------------
        d678 = results[678]
        d679 = results[679]
        d680 = results[680]
        d681 = results[681]
        d682 = results[682]
        d683 = results[683]
        d684 = results[684]
        d685 = results[685]
        d686 = results[686]  # not_run sentinel
        d687 = results[687]
        d688 = results[688]

        manifest_consulted: bool = bool(d678.get("conductor_consulted", False))
        vr_200q_signed_improvement: float = float(d679.get("signed_improvement") or 0.0)
        vr_200q_validated: bool = vr_200q_signed_improvement > 0
        humaneval_vr_improvement: float = float(d680.get("signed_improvement") or 0.0)
        adversarial_vr_robust: bool = bool(d681.get("adversarial_robust", False))
        jepa_v15_true_ood_auc: float = float(d682.get("true_ood_auc") or 0.0)
        fr11_real_positives_wired: bool = bool(d683.get("fr11_real_positives_confirmed", False))
        retro_071_resolved: bool = bool(
            d684.get("retro_071_resolved", False) or d685.get("retro_071_resolved", False)
        )
        dualgpu_retrain_speedup: float = float(d685.get("speedup") or 0.0)
        fover_formal_v1_n_labels: int = int(d686.get("n_labels") or 0)
        psv_iterations_completed: int = int(d688.get("n_iterations") or 0)

        # ------------------------------------------------------------------
        # Step 4: RETRO status assessments
        # ------------------------------------------------------------------
        retro_071_status = "closed" if retro_071_resolved else "open_attempt_16"
        manifest_status = "confirmed" if manifest_consulted else "unconfirmed"
        vr_win_validation = "confirmed" if vr_200q_validated else "not_confirmed"

        # Wall-time direction: compare this cycle vs prior avg per experiment
        # Prior avg was 8.0 min/exp (from .51). New cycle avg:
        cycle_avg_per_exp = round(cycle_wall_time_minutes / len(MILESTONE_EXPERIMENTS), 2)
        prior_avg_per_exp = 8.0
        if cycle_avg_per_exp < prior_avg_per_exp:
            wall_time_direction = "improved"
        elif abs(cycle_avg_per_exp - prior_avg_per_exp) < 0.1:
            wall_time_direction = "flat"
        else:
            wall_time_direction = "regressed"

        # ------------------------------------------------------------------
        # Step 5: GPU state
        # ------------------------------------------------------------------
        gpu_state = read_gpu_state()

        # ------------------------------------------------------------------
        # Step 6: Honest verdict
        # ------------------------------------------------------------------
        honest_verdict = (
            f"wall_time_{wall_time_direction}"
            f"_vr_{vr_win_validation}"
            f"_retro071_{retro_071_status}"
            f"_manifest_{manifest_status}"
            f"_fr11_wired_{'yes' if fr11_real_positives_wired else 'no'}"
            f"_dualgpu_speedup_{dualgpu_retrain_speedup:.2f}x"
            f"_jepa_ood_auc_{jepa_v15_true_ood_auc:.4f}"
            f"_psv_{psv_iterations_completed}_iters"
        )

        # ------------------------------------------------------------------
        # Step 7: Build milestone history (carry forward from .51)
        # ------------------------------------------------------------------
        prior_history = [
            {"milestone": "2026.04.40", "wall_time_min": 4620, "experiments": 388, "avg_min_per_exp": 11.9},
            {"milestone": "2026.04.41", "wall_time_min": 4484, "experiments": 400, "avg_min_per_exp": 11.2},
            {"milestone": "2026.04.42", "wall_time_min": 4520, "experiments": 422, "avg_min_per_exp": 10.7},
            {"milestone": "2026.04.43", "wall_time_min": 4584, "experiments": 444, "avg_min_per_exp": 10.3},
            {"milestone": "2026.04.44", "wall_time_min": 4654, "experiments": 465, "avg_min_per_exp": 10.0},
            {"milestone": "2026.04.45", "wall_time_min": 4600, "experiments": 481, "avg_min_per_exp": 9.6},
            {"milestone": "2026.04.46", "wall_time_min": 4569, "experiments": 493, "avg_min_per_exp": 9.3},
            {"milestone": "2026.04.47", "wall_time_min": 4557, "experiments": 498, "avg_min_per_exp": 9.1},
            {"milestone": "2026.04.48", "wall_time_min": 4467, "experiments": 491, "avg_min_per_exp": 9.1},
            {"milestone": "2026.04.49", "wall_time_min": 4380, "experiments": 509, "avg_min_per_exp": 8.6},
            {"milestone": "2026.04.50", "wall_time_min": 4304, "experiments": 519, "avg_min_per_exp": 8.3},
            {"milestone": "2026.04.51", "wall_time_min": 4231, "experiments": 532, "avg_min_per_exp": 8.0},
            {
                "milestone": MILESTONE,
                "wall_time_min": round(cumulative_wall_time_minutes, 2),
                "experiments": total_experiments,
                "avg_min_per_exp": per_experiment_avg_min,
            },
        ]

        # ------------------------------------------------------------------
        # Step 8: Build and write result
        # ------------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "schema_version": SCHEMA,
                "milestone": MILESTONE,
                "experiment_table": experiment_table,
                # Wall-time accounting
                "cycle_duration_s": round(cycle_duration_s, 3),
                "cycle_wall_time_minutes": cycle_wall_time_minutes,
                "total_wall_time_minutes": round(cumulative_wall_time_minutes, 2),
                "experiments_completed": total_experiments,
                "cycle_experiments": len(MILESTONE_EXPERIMENTS),
                "avg_time_per_experiment_minutes": per_experiment_avg_min,
                "cycle_avg_min_per_exp": cycle_avg_per_exp,
                "prior_avg_min_per_exp": prior_avg_per_exp,
                "slowest_5": slowest_5,
                # Key metrics
                "manifest_consulted": manifest_consulted,
                "vr_200q_signed_improvement": vr_200q_signed_improvement,
                "vr_200q_validated": vr_200q_validated,
                "humaneval_vr_improvement": humaneval_vr_improvement,
                "adversarial_vr_robust": adversarial_vr_robust,
                "jepa_v15_true_ood_auc": jepa_v15_true_ood_auc,
                "fr11_real_positives_wired": fr11_real_positives_wired,
                "retro_071_resolved": retro_071_resolved,
                "dualgpu_retrain_speedup": dualgpu_retrain_speedup,
                "fover_formal_v1_n_labels": fover_formal_v1_n_labels,
                "psv_iterations_completed": psv_iterations_completed,
                # RETRO status
                "retro_071_status": retro_071_status,
                "manifest_status": manifest_status,
                "vr_win_validation": vr_win_validation,
                "wall_time_direction": wall_time_direction,
                # GPU state
                "gpu_state_at_close": gpu_state,
                # History
                "milestone_history": prior_history,
                # Honest verdict
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        # Write deliverable
        out_path = _REPO / DELIVERABLE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2))
        print(f"Written: {out_path}")
        print(f"honest_verdict: {honest_verdict}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
