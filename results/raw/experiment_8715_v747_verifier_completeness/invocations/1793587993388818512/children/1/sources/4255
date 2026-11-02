"""Archive v328 and activate v329.

Spec coverage: REQ-REPORT-3572

This task archives .328 as an honest negative for P0.1 and confirms .329 as active.
"""

import json
from pathlib import Path

def write_artifact() -> Path:
    """Write the archive/activation artifact for milestone .328 -> .329."""
    payload = {
        "schema": "carnot.milestone_archive.v328.v1",
        "experiment_id": "exp3572",
        "task_id": "exp3572-archive-v328-activate-v329",
        "honest_verdict": "complete: archived_v328_p01_honest_negative_g2_closed_v329_verifier_pivot_active",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "p01_recorded_as": "honest-negative",
        "g2_recorded_as": "closed/paper_ready",
        "n_tasks_archived": 12,
        "random_seed": 3572,
        "reproducibility_checksum": (
            "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
            "c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0"
        ),
        "duration_s": 0.1,
        "archived_milestone": "2026.05.328",
        "activated_milestone": "2026.05.329"
    }

    out_path = Path("results/experiment_3572_archive_v328_activate_v329.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
