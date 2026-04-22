#!/usr/bin/env python3
"""Experiment 677: Milestone 2026.04.51 Operational Retrospective.

Computes wall-time trends, per-experiment efficiency, and research outcome
summary for Exps 666-676.  Emits results/operational_retro_2026_04_51.json.

Spec: REQ-INFRA-007, REQ-INFRA-023, REQ-INFRA-062
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as a script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

DELIVERABLE = "results/operational_retro_2026_04_51.json"

# Canonical list of Exp IDs in this milestone (666-676 inclusive).
MILESTONE_EXP_IDS = [666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676]

# File names keyed by experiment ID.
_RESULT_FILES = {
    666: "results/experiment_666_manifest_wireIn_v3.json",
    667: "results/experiment_667_gate_v4_redesign.json",
    668: "results/experiment_668_vr_attempt_18_v2.json",
    669: "results/experiment_669_prompt_injection_rescue.json",
    670: "results/experiment_670_jepa_cascade_deploy.json",
    671: "results/experiment_671_jepa_v15.json",
    672: "results/experiment_672_kv260_dfx_fix.json",
    673: "results/experiment_673_dualgpu_v3.json",
    674: "results/experiment_674_ias_adaptive_gate.json",
    675: "results/experiment_675_losnet_detector.json",
    676: "results/experiment_676_metajuls_adaptive.json",
}

# Cumulative totals from the .50 retrospective (Exp 665).
PRIOR_TOTAL_WALL_TIME_MIN = 4304.0
PRIOR_EXPERIMENTS_COMPLETED = 519


def load_experiment_results(repo_root: Path) -> dict[int, dict]:
    """Load all milestone experiment result files from disk.

    Returns a mapping of experiment ID to parsed JSON artifact.  Missing files
    raise FileNotFoundError so the retro never silently omits an experiment.
    """
    results: dict[int, dict] = {}
    for exp_id, rel_path in _RESULT_FILES.items():
        full_path = repo_root / rel_path
        if not full_path.exists():
            raise FileNotFoundError(
                f"Experiment {exp_id} result not found: {full_path}"
            )
        results[exp_id] = json.loads(full_path.read_text())
    return results


def compute_milestone_metrics(
    results: dict[int, dict],
    prior_wall_time_min: float,
    prior_experiments: int,
) -> dict:
    """Derive all milestone-level metrics from the loaded experiment results.

    Returns a flat dict ready to embed in the retro artifact.

    Parameters
    ----------
    results:
        Mapping of exp_id -> parsed artifact dict.
    prior_wall_time_min:
        Cumulative wall-time total from the prior retrospective (.50).
    prior_experiments:
        Cumulative experiment count from the prior retrospective.
    """
    # --- Wall-time and experiment counts ---
    this_cycle_duration_s = sum(
        float(r.get("duration_s", 0.0)) for r in results.values()
    )
    this_cycle_min = this_cycle_duration_s / 60.0
    total_wall_time_min = prior_wall_time_min + this_cycle_min
    n_experiments_this_cycle = len(results)
    total_experiments = prior_experiments + n_experiments_this_cycle
    per_exp_avg_min = total_wall_time_min / total_experiments

    wall_time_delta_min = total_wall_time_min - prior_wall_time_min
    wall_time_delta_direction = "improvement" if wall_time_delta_min < 0 else "regression"
    wall_time_delta_pct = round(wall_time_delta_min / prior_wall_time_min * 100, 2)

    # --- Slowest 5 experiments this cycle ---
    sorted_by_duration = sorted(
        results.values(),
        key=lambda r: float(r.get("duration_s", 0.0)),
        reverse=True,
    )
    slowest_5 = [
        {
            "experiment": r.get("experiment"),
            "title": r.get("title", ""),
            "duration_s": float(r.get("duration_s", 0.0)),
            "duration_minutes": round(float(r.get("duration_s", 0.0)) / 60.0, 3),
            "status": r.get("status", "unknown"),
            "honest_verdict": r.get("honest_verdict", ""),
        }
        for r in sorted_by_duration[:5]
    ]

    # --- Milestone-specific outcome metrics ---
    r666 = results[666]
    r668 = results[668]
    r670 = results[670]
    r671 = results[671]
    r672 = results[672]
    r673 = results[673]

    manifest_consulted = "confirmed" if r666.get("manifest_loaded") else "not_confirmed"
    vr_signed_improvement = float(r668.get("signed_improvement", 0.0))
    jepa_v14_deployed = bool(r670.get("jepa_v14_deployed", False))
    jepa_v15_ood_auc = float(r671.get("ood_auc", 0.0))
    # KV260 was blocked (bitfile not configured) — no latency measurement.
    kv260_hardware_latency_us = None
    kv260_status = r672.get("honest_verdict", "unknown")
    gpu1_utilization = float(r673.get("max_gpu1_util_pct", 0.0))

    # RETRO-033: closed if VR attempt returned a positive signed improvement.
    retro_033_status = "closed" if vr_signed_improvement > 0 else "open_attempt_19"

    # RETRO-071: DualGPU proof required pynvml for utilization measurement;
    # pynvml was unavailable (pynvml_available=False in Exp 673 artifact).
    retro_071_status = (
        "resolved" if r673.get("retro_071_resolved", False) else "open_partial"
    )

    # --- Honest verdict summary ---
    vr_tag = "vr_positive" if vr_signed_improvement > 0 else "vr_negative"
    manifest_tag = "manifest_confirmed" if manifest_consulted == "confirmed" else "manifest_unconfirmed"
    dualgpu_tag = "dualgpu_proven" if retro_071_status == "resolved" else "dualgpu_partial"
    retro033_tag = "retro033_closed" if retro_033_status == "closed" else "retro033_open"
    kv260_tag = "kv260_blocked" if kv260_hardware_latency_us is None else "kv260_active"
    per_exp_tag = f"per_exp_avg_{str(round(per_exp_avg_min, 1)).replace('.', 'pt')}min"

    honest_verdict = (
        f"wall_time_{wall_time_delta_direction}_{abs(round(this_cycle_min, 0)):.0f}min"
        f"_{vr_tag}_{retro033_tag}_{manifest_tag}"
        f"_{dualgpu_tag}_retro071_{retro_071_status}"
        f"_jepa_v15_ood_auc_{str(jepa_v15_ood_auc).replace('.', 'pt')}"
        f"_{kv260_tag}_{per_exp_tag}"
    )

    return {
        "n_experiments_this_cycle": n_experiments_this_cycle,
        "total_experiments": total_experiments,
        "this_cycle_duration_s": round(this_cycle_duration_s, 3),
        "this_cycle_wall_time_minutes": round(this_cycle_min, 3),
        "total_wall_time_minutes": round(total_wall_time_min, 3),
        "prior_milestone_wall_time_minutes": prior_wall_time_min,
        "prior_milestone_experiments": prior_experiments,
        "wall_time_delta_minutes": round(wall_time_delta_min, 3),
        "wall_time_delta_direction": wall_time_delta_direction,
        "wall_time_delta_pct": wall_time_delta_pct,
        "per_experiment_avg_min": round(per_exp_avg_min, 3),
        "slowest_5": slowest_5,
        "manifest_consulted": manifest_consulted,
        "vr_signed_improvement": vr_signed_improvement,
        "jepa_v14_deployed": jepa_v14_deployed,
        "jepa_v15_ood_auc": jepa_v15_ood_auc,
        "kv260_hardware_latency_us": kv260_hardware_latency_us,
        "kv260_status": kv260_status,
        "gpu1_utilization": gpu1_utilization,
        "retro_033_status": retro_033_status,
        "retro_071_status": retro_071_status,
        "honest_verdict": honest_verdict,
    }


def main() -> None:
    """Run the Exp 677 milestone retrospective and write the deliverable JSON."""
    tmpl = ExperimentTemplate(
        677,
        "Milestone 2026.04.51 Operational Retrospective",
        DELIVERABLE,
    )

    with ExperimentTimeoutWatchdog(
        677,
        timeout_minutes=30,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    ):
        tmpl.setup()
        tmpl.check_exclusion_manifest()

        # Load all 11 experiment result files for this milestone.
        results = load_experiment_results(_REPO_ROOT)

        # Derive all metrics.
        metrics = compute_milestone_metrics(
            results,
            PRIOR_TOTAL_WALL_TIME_MIN,
            PRIOR_EXPERIMENTS_COMPLETED,
        )

        # Build the standardised artifact via ExperimentTemplate.
        artifact = tmpl.build_result(
            {
                "schema": "carnot.operational_retro.v26",
                "milestone": "2026.04.51",
                **metrics,
            },
            status="success",
        )

        # Write atomically (RETRO-030 pattern).
        writer = AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE))
        writer.write(artifact)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
