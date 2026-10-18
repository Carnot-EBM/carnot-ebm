"""
Experiment 879: Milestone 2026.04.67 Operational Retrospective

Reads result JSONs for Exps 868-878, evaluates the 11 success criteria defined
for milestone 2026.04.67, and writes a structured artifact. This is the
authoritative milestone close record; the conductor uses it to decide whether
to activate the next roadmap.

Why this script exists separately from the conductor: the retro must run even
when the conductor itself is the thing being debugged. Keeping the retro as a
standalone script means you can always close the milestone manually.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone, UTC
from pathlib import Path
import contextlib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
EXCLUSION_MANIFEST = REPO_ROOT / "ops" / "exclusion_manifest.yaml"

MILESTONE = "2026.04.67"
RETRO_SCHEMA = "carnot.operational_retro.v42"

# Experiments in this milestone (868-878; 879 is the retro itself)
MILESTONE_EXP_IDS = list(range(868, 879))

# Baseline from prior milestone close (Exp 868 preflight records these)
PRIOR_WALL_TIME_MINUTES = 4107.0
PRIOR_EXPERIMENT_COUNT = 794


def load_result(exp_id: int) -> dict:
    """
    Load an experiment result JSON by experiment ID.

    Searches results/ for a file whose name starts with experiment_{exp_id}_.
    Returns empty dict if not found — the retro should degrade gracefully when
    a single experiment artifact is missing rather than crashing.
    """
    matches = sorted(RESULTS_DIR.glob(f"experiment_{exp_id}_*.json"))
    if not matches:
        return {}
    return json.loads(matches[0].read_text())


def get_retired_ids() -> set:
    """
    Parse the YAML exclusion manifest to get retired experiment IDs.

    We parse line-by-line instead of importing a YAML library to keep this
    script dependency-free. The format is stable (experiment_id: <int>).
    """
    retired = set()
    if not EXCLUSION_MANIFEST.exists():
        return retired
    for line in EXCLUSION_MANIFEST.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("- experiment_id:"):
            with contextlib.suppress(ValueError):
                retired.add(int(stripped.split(":")[1].strip()))
    return retired


def gpu_state() -> dict:
    """
    Query GPU state via nvidia-smi.

    Returns a dict with gpu_available, gpu_count, and per-GPU details.
    Failure (e.g. no NVIDIA driver) is recorded gracefully — a missing GPU
    is never a retro blocker.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {"gpu_available": False, "error": result.stderr.strip()}
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            gpus.append(
                {
                    "name": parts[0],
                    "memory_used_mib": parts[1],
                    "memory_total_mib": parts[2],
                    "utilization_pct": parts[3],
                }
            )
        return {"gpu_available": True, "gpu_count": len(gpus), "gpus": gpus}
    except Exception as exc:  # noqa: BLE001
        return {"gpu_available": False, "error": str(exc)}


def evaluate_criteria(results: dict[int, dict]) -> dict:
    """
    Evaluate the 11 milestone success criteria.

    Each criterion is True/False. The docstring explains what each one means
    so future engineers don't have to hunt through the milestone spec.

    Returns a dict mapping criterion name -> bool.
    """
    r = results

    # 1. Manifest enforcer deployed (Exp 868 side-channel deployment)
    manifest_enforcer_deployed = bool(r.get(868, {}).get("manifest_enforcer_deployed", False))

    # 2. GGUF model download verified (Exp 869)
    download_verified = bool(r.get(869, {}).get("download_verified", False))

    # 3. Live code repair positive — signed_improvement > 0 means the repair
    #    stage produced a measurable improvement on the LLM's output (Exp 870)
    signed_improvement = r.get(870, {}).get("signed_improvement")
    live_code_repair_positive = signed_improvement is not None and float(signed_improvement) > 0

    # 4. Live cascade benchmark — inference_mode must be "live_gpu" to count;
    #    simulation_fallback does NOT satisfy this criterion (Exp 871)
    inference_mode = r.get(871, {}).get("inference_mode", "")
    live_cascade_benchmark = inference_mode == "live_gpu"

    # 5. JEPA OOD AUC improved above 0.65 (Exp 872)
    ood_auc = r.get(872, {}).get("ood_auc", 0.0)
    jepa_ood_improved = float(ood_auc) > 0.65

    # 6. JEPA cascade deployed in pipeline (Exp 873)
    cascade_deployed = bool(r.get(873, {}).get("cascade_deployed", False))

    # 7. StreamingCoT wired into pipeline (Exp 874)
    #    The honest_verdict == "streaming_cot_wired" is the authoritative signal
    streaming_cot_wired = r.get(874, {}).get("honest_verdict") == "streaming_cot_wired"

    # 8. FR-11 Tier-2 relay loop closed (Exp 875)
    fr11_tier2_loop_closed = "loop_closed" in str(r.get(875, {}).get("honest_verdict", ""))

    # 9. Inertia sweeps reduced by >= 5x (Exp 876)
    sweeps_reduction = r.get(876, {}).get("sweeps_reduction", 0.0)
    inertia_5x = float(sweeps_reduction) >= 5.0

    # 10. VJEPA predictor tier-3 seed viable (Exp 877)
    vjepa_viable = r.get(877, {}).get("honest_verdict") == "tier3_seed_viable"

    # 11. HalluSAE AUC v2 >= 0.65 (Exp 878)
    auc_v2 = r.get(878, {}).get("auc_v2", 0.0)
    hallusae_closed = float(auc_v2) >= 0.65

    return {
        "manifest_enforcer_deployed": manifest_enforcer_deployed,
        "download_verified": download_verified,
        "live_code_repair_positive": live_code_repair_positive,
        "live_cascade_benchmark": live_cascade_benchmark,
        "jepa_ood_improved": jepa_ood_improved,
        "cascade_deployed": cascade_deployed,
        "streaming_cot_wired": streaming_cot_wired,
        "fr11_tier2_loop_closed": fr11_tier2_loop_closed,
        "inertia_5x": inertia_5x,
        "vjepa_viable": vjepa_viable,
        "hallusae_closed": hallusae_closed,
    }


def evaluate_retro_closures(results: dict[int, dict]) -> tuple[list, list]:
    """
    Determine which open RETROs are closed by this milestone's results.

    Returns (retros_closed, open_retros).

    A RETRO is closed only when the numeric threshold is actually met — not just
    when the experiment ran. This is a hard boundary to prevent false closure
    claims that the conductor would then propagate into future milestones.
    """
    open_retros_from_preflight = [
        "RETRO-MANIFEST-FULL-SCOPE",
        "RETRO-JEPA-OOD",
        "RETRO-SVAMP-ZERO-AUC",
        "RETRO-XILINX-TOOLS-UNAVAILABLE",
        "RETRO-SOTA-MODEL-DOWNLOAD",
        "RETRO-HALLUSAE-AUC-BELOW-THRESHOLD",
        "RETRO-INERTIA-SWEEPS-TARGET-MISSED",
    ]

    retros_closed = []

    download_verified = bool(results.get(869, {}).get("download_verified", False))
    if download_verified:
        retros_closed.append("RETRO-SOTA-MODEL-DOWNLOAD")

    ood_auc = float(results.get(872, {}).get("ood_auc", 0.0))
    if ood_auc > 0.65:
        retros_closed.append("RETRO-JEPA-OOD")

    svamp_auc = float(results.get(872, {}).get("svamp_auc", 0.0))
    if svamp_auc > 0.50:
        retros_closed.append("RETRO-SVAMP-ZERO-AUC")

    auc_v2 = float(results.get(878, {}).get("auc_v2", 0.0))
    if auc_v2 >= 0.65:
        retros_closed.append("RETRO-HALLUSAE-AUC-BELOW-THRESHOLD")

    sweeps_reduction = float(results.get(876, {}).get("sweeps_reduction", 0.0))
    if sweeps_reduction >= 5.0:
        retros_closed.append("RETRO-INERTIA-SWEEPS-TARGET-MISSED")

    remaining_open = [r for r in open_retros_from_preflight if r not in retros_closed]
    return retros_closed, remaining_open


def build_honest_verdict(
    criteria: dict,
    n_criteria_met: int,
    total_criteria: int,
    retros_closed: list,
    wall_time_minutes: float,
    prior_wall_time: float,
) -> str:
    """
    Build the honest_verdict string.

    This string is machine-readable (the conductor parses tokens) and
    human-readable (engineers scan it in the retro logs). Key tokens:
    - CODE_REPAIR_POSITIVE: live code repair produced a measurable improvement
    - REGRESSION_N_CONSECUTIVE: wall time exceeded prior milestone baseline
    - The fraction n/total shows milestone goal completion at a glance
    """
    tokens = [f"{n_criteria_met}/{total_criteria}_criteria_met"]

    if criteria.get("live_code_repair_positive"):
        tokens.append("CODE_REPAIR_POSITIVE")

    # Wall time regression check
    if wall_time_minutes > prior_wall_time:
        tokens.append("REGRESSION_CONSECUTIVE")

    tokens.append(f"{len(retros_closed)}_retros_closed")

    if criteria.get("streaming_cot_wired"):
        tokens.append("STREAMING_COT_WIRED")
    if criteria.get("fr11_tier2_loop_closed"):
        tokens.append("FR11_TIER2_LOOP_CLOSED")
    if criteria.get("vjepa_viable"):
        tokens.append("VJEPA_TIER3_SEED_VIABLE")
    if criteria.get("manifest_enforcer_deployed"):
        tokens.append("MANIFEST_ENFORCER_DEPLOYED")

    if not criteria.get("live_cascade_benchmark"):
        tokens.append("NO_LIVE_GPU")
    if not criteria.get("live_code_repair_positive"):
        tokens.append("NO_CODE_REPAIR_POSITIVE")

    return "_".join(tokens)


def main() -> None:
    started_at = datetime.now(UTC).isoformat()

    # 1. Load all experiment results
    results = {exp_id: load_result(exp_id) for exp_id in MILESTONE_EXP_IDS}

    # 2. Compute wall-time metrics
    durations = {
        exp_id: float(results[exp_id].get("duration_s", 0.0)) for exp_id in MILESTONE_EXP_IDS
    }
    total_duration_s = sum(durations.values())
    total_wall_time_minutes = total_duration_s / 60.0
    exp_count = len(MILESTONE_EXP_IDS)
    per_experiment_avg = total_wall_time_minutes / exp_count

    # Slowest 5
    sorted_by_dur = sorted(durations.items(), key=lambda kv: kv[1], reverse=True)
    slowest_5 = [
        {"exp_id": exp_id, "duration_min": round(dur / 60.0, 4)}
        for exp_id, dur in sorted_by_dur[:5]
    ]

    # 3. Status counts
    status_counts: dict[str, int] = {}
    for exp_id in MILESTONE_EXP_IDS:
        s = results[exp_id].get("status", "unknown") or "unknown"
        status_counts[s] = status_counts.get(s, 0) + 1
    experiments_completed = status_counts.get("success", 0)
    experiments_blocked = status_counts.get("blocked", 0)

    # 4. Evaluate criteria
    criteria = evaluate_criteria(results)
    n_criteria_met = sum(1 for v in criteria.values() if v)
    total_criteria = len(criteria)

    # 5. RETRO closures
    retros_closed, open_retros = evaluate_retro_closures(results)

    # 6. Governance check — none of the slowest-5 should be retired
    retired_ids = get_retired_ids()
    slowest_5_ids = {item["exp_id"] for item in slowest_5}
    slowest5_governance_violation = bool(slowest_5_ids & retired_ids)

    # 7. Wall-time regression vs .66 baseline
    delta_minutes = total_wall_time_minutes - PRIOR_WALL_TIME_MINUTES
    regression = total_wall_time_minutes > PRIOR_WALL_TIME_MINUTES

    # 8. GPU state
    gpu = gpu_state()

    # 9. honest_verdict
    honest_verdict = build_honest_verdict(
        criteria=criteria,
        n_criteria_met=n_criteria_met,
        total_criteria=total_criteria,
        retros_closed=retros_closed,
        wall_time_minutes=total_wall_time_minutes,
        prior_wall_time=PRIOR_WALL_TIME_MINUTES,
    )

    finished_at = datetime.now(UTC).isoformat()
    duration_retro_s = (
        datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)
    ).total_seconds()

    # Total experiment count at close includes 879 (the retro itself)
    experiment_count_at_close = PRIOR_EXPERIMENT_COUNT + exp_count + 1  # 806

    artifact = {
        "schema": RETRO_SCHEMA,
        "milestone": MILESTONE,
        "retro_type": "milestone_close",
        "retro_date": "20260425",
        "experiment": 879,
        "title": "Milestone 2026.04.67 Operational Retrospective",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(duration_retro_s, 3),
        # Milestone scope
        "experiment_count": experiment_count_at_close,
        "experiments_this_milestone": exp_count,
        "experiments_completed": experiments_completed,
        "experiments_blocked": experiments_blocked,
        "status_counts": status_counts,
        # Wall time
        "wall_time_minutes": round(total_wall_time_minutes, 4),
        "per_experiment_avg_minutes": round(per_experiment_avg, 4),
        "prior_milestone_wall_time_minutes": PRIOR_WALL_TIME_MINUTES,
        "delta_vs_prior_minutes": round(delta_minutes, 4),
        "wall_time_regression": regression,
        # Slowest experiments
        "slowest_5": slowest_5,
        # Criteria
        "n_criteria_met": n_criteria_met,
        "n_criteria_total": total_criteria,
        "criteria": criteria,
        # RETRO closure
        "retros_closed_this_milestone": retros_closed,
        "retros_closed_count": len(retros_closed),
        "open_retros": open_retros,
        "open_retros_count": len(open_retros),
        # Governance
        "governance": {
            "manifest_enforcer_deployed": bool(
                results.get(868, {}).get("manifest_enforcer_deployed", False)
            ),
            "slowest5_governance_violation": slowest5_governance_violation,
            "slowest5_exp_ids": sorted(slowest_5_ids),
            "retired_exp_ids_in_slowest5": sorted(slowest_5_ids & retired_ids),
        },
        # GPU
        "gpu_state_at_retro": gpu,
        # Verdict
        "honest_verdict": honest_verdict,
        "invariant_violations": [],
        # Per-experiment summary
        "per_experiment_summary": [
            {
                "exp_id": exp_id,
                "status": results[exp_id].get("status"),
                "honest_verdict": results[exp_id].get("honest_verdict"),
                "duration_s": durations[exp_id],
            }
            for exp_id in MILESTONE_EXP_IDS
        ],
    }

    # Write primary deliverable
    out_primary = RESULTS_DIR / "experiment_879_milestone_retro.json"
    out_primary.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote: {out_primary}")

    # Write conductor-compatible copy
    out_retro = RESULTS_DIR / "operational_retro_2026_04_67.json"
    out_retro.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote: {out_retro}")

    # Print summary
    print(f"\nMilestone {MILESTONE} retrospective complete.")
    print(
        f"  Experiments: {exp_count} ({experiments_completed} success, "
        f"{experiments_blocked} blocked)"
    )
    print(
        f"  Wall time:   {total_wall_time_minutes:.2f} min (avg {per_experiment_avg:.2f} min/exp)"
    )
    print(f"  Criteria:    {n_criteria_met}/{total_criteria}")
    print(f"  RETROs closed: {len(retros_closed)}")
    print(f"  Verdict:     {honest_verdict}")

    # assert_deliverable_written equivalent
    assert out_primary.exists(), f"Deliverable not written: {out_primary}"
    assert out_retro.exists(), f"Conductor copy not written: {out_retro}"
    print("\nassert_deliverable_written: PASS")


if __name__ == "__main__":
    main()
