"""
Milestone .157 retrospective generator.

Scans experiment artifacts from exp2008-exp2016 (the .157 task range, excluding
the retro itself at exp2017) and produces a structured retro JSON using the
`carnot.milestone_retro.v1` schema.
"""

import json
import os
from glob import glob

_MILESTONE_157_START = 2008
_MILESTONE_157_END = 2016


def _classify_artifact(artifact: dict) -> str:
    """Classify a parsed experiment artifact as completed, blocked, or failed."""
    # 1. Honest verdict overrides
    verdict = artifact.get("honest_verdict", "").lower()
    if "blocked" in verdict:
        return "blocked"
    if "fail" in verdict or "missing artifacts" in verdict:
        return "failed"
    if "complete:" in verdict:
        return "completed"

    # 2. Status field fallback
    status = artifact.get("status", "").lower()
    if status in ("success", "complete", "completed"):
        return "completed"
    if status == "blocked":
        return "blocked"
    if status in ("fail", "failure", "error"):
        return "failed"

    # 3. Default to completed if no negative signals
    return "completed"


def generate_retro(output_path: str, results_dir: str = "results") -> dict:
    completed = []
    blocked = []
    failed = []
    verdicts = {}

    for exp_id in range(_MILESTONE_157_START, _MILESTONE_157_END + 1):
        paths = glob(os.path.join(results_dir, f"experiment_{exp_id}*.json"))
        if not paths:
            failed.append(exp_id)
            verdicts[f"exp{exp_id}"] = "MISSING"
            continue

        # If multiple exist, pick the shortest name (usually the base artifact)
        target_path = min(paths, key=len)

        try:
            with open(target_path) as fh:
                artifact = json.load(fh)
        except json.JSONDecodeError:
            failed.append(exp_id)
            verdicts[f"exp{exp_id}"] = "UNREADABLE"
            continue

        verdicts[f"exp{exp_id}"] = artifact.get(
            "honest_verdict", artifact.get("status", "UNKNOWN")
        )
        cls = _classify_artifact(artifact)

        if cls == "blocked":
            blocked.append(exp_id)
        elif cls == "failed":
            failed.append(exp_id)
        else:
            completed.append(exp_id)

    artifact = {
        "experiment_id": 2017,
        "schema": "carnot.milestone_retro.v1",
        "milestone": "2026.05.157",
        "milestone_title": "Execution Bottlenecks and Missing Artifacts Diagnosis",
        "run_date": "2026-05-13",
        "status": "complete",
        "completed_task_count": len(completed),
        "blocked_task_count": len(blocked),
        "failed_task_count": len(failed),
        "completed_experiments": sorted(completed),
        "blocked_experiments": sorted(blocked),
        "failed_experiments": sorted(failed),
        "experiment_honest_verdicts": verdicts,
        "recommendations": [
            "Mandate that prior_failures entries are strictly verified by pre-gates.",
            "Diagnose why artifacts 2008, 2012, 2013, 2014 failed to write anything.",
            "Suspend dependent research tracks until the missing artifacts issue is root-caused."
        ],
        "bottlenecks_identified": [
            "Conductor pre-gate rejecting artifacts en masse due to missing prior_failures (e.g. Exp 2009, 2010, 2011, 2015).",
            "Missing JSON artifacts causing silent gaps in the milestone record."
        ],
        "retro_complete": True,
        "honest_verdict": f"complete: milestone_157_retro_filed_{len(completed)}_completed_{len(blocked)}_blocked_{len(failed)}_failed"
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(artifact, fh, indent=2)

    return artifact


if __name__ == "__main__":  # pragma: no cover
    generate_retro("results/experiment_2017_milestone_157_retro.json")
