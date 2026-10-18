"""Pipeline fail-fast checks for doomed reruns.

Spec: REQ-PIPELINE-1826
"""

import json
from pathlib import Path
from typing import Any, Dict


def pipeline_fail_fast_check(task: Dict[str, Any], output_path: str | Path) -> bool:
    """Check if task is doomed, and write an activation-time terminal artifact.

    If the task is doomed, writes the artifact and returns True.

    Parameters
    ----------
    task : dict
        Task representation.
    output_path : str or Path
        Path to write the artifact to if the task is doomed.

    Returns
    -------
    bool
        True if the task is a doomed rerun and the artifact was written, False otherwise.
    """
    is_doomed = task.get("doomed_rerun", False)
    is_rerun_missing_priors = task.get("is_rerun", False) and "prior_failures" not in task

    if is_doomed or is_rerun_missing_priors:
        reason = task.get("doomed_reason", "Missing prior_failures for rerun task")
        
        artifact = {
            "status": "blocked",
            "honest_verdict": "blocked_doomed_rerun",
            "reason": reason,
            "task_id": task.get("id", "unknown"),
        }
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)
        return True
    return False
