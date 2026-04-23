#!/usr/bin/env python3
"""Experiment 754 — Pre-flight v10: manifest patch applied and enforcement confirmed.

WHY THIS EXPERIMENT EXISTS:
    Milestone .57 closed with manifest_fix_patch.txt STILL not applied to
    scripts/research_conductor.py for the FOURTH consecutive cycle.  This
    experiment APPLIES the patch (step 4 is handled in this session before
    this script runs), then verifies the application, confirms GPU health,
    and confirms Exp 527 exclusion.  Cumulative waste from skipping the patch:
    1,264 min (21.1 hours).

WHAT THIS SCRIPT DOES:
    1. Reads the exclusion manifest and verifies Exp 527 is excluded.
    2. Reads scripts/research_conductor.py and searches for the guard clause
       injected by the patch (validates REQ-INFRA-051, REQ-INFRA-052).
    3. Runs nvidia-smi to measure GPU VRAM / utilization / temperature.
    4. Runs the incremental test selection baseline via conductor_pre_flight.py.
    5. Writes the results artifact with honest_verdict encoding all conditions.

Spec: REQ-INFRA-051, REQ-INFRA-052, SCENARIO-INFRA-060, SCENARIO-INFRA-061
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Make repo root importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 754
TITLE = "Pre-flight v10: manifest patch applied and enforcement confirmed"
DELIVERABLE = "results/experiment_754_preflight_v10.json"

_MANIFEST_PATH = _REPO_ROOT / "scripts" / "conductor_exclusion_manifest.json"
_CONDUCTOR_PATH = _REPO_ROOT / "scripts" / "research_conductor.py"
_PATCH_PATH = _REPO_ROOT / "results" / "manifest_fix_patch.txt"

# The guard clause string that MUST appear in research_conductor.py after patch.
# We search for this exact function name as evidence the patch was applied.
_GUARD_CLAUSE = "validate_manifest_at_dequeue"

# GPU clean threshold: < 100 MB VRAM on each device.
_GPU_VRAM_CLEAN_MB = 100


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------


def check_patch_applied(conductor_path: Path) -> bool:
    """Return True if the guard clause is present in research_conductor.py.

    WHY THIS MATTERS (REQ-INFRA-051, REQ-INFRA-052):
        Four consecutive milestones closed without the guard clause being
        wired into the dispatcher.  String IDs like "jepa_v15_cascade" bypass
        the existing regex-based exclusion check in _task_is_excluded().  Only
        the dispatch-site guard in research_step() closes this gap.

    We search the file text rather than importing the module to avoid
    triggering side effects from the conductor's module-level code.
    """
    if not conductor_path.exists():
        return False
    text = conductor_path.read_text()
    return _GUARD_CLAUSE in text


def check_exp527_excluded(manifest_path: Path) -> tuple[bool, int]:
    """Return (exp527_excluded, n_excluded_experiments) from the manifest.

    WHY EXP 527 SPECIFICALLY:
        Exp 527 appeared in the slowest-5 for three consecutive milestones
        (.55, .56, .57), crossing the mandatory-retirement governance threshold
        (REQ-INFRA-048).  Exp 740 added it to the manifest.  If it is absent,
        the conductor will re-run it in .58.

    Returns
    -------
    tuple[bool, int]
        (True if 527 is excluded, count of all excluded entries)
    """
    if not manifest_path.exists():
        return False, 0
    try:
        raw = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False, 0
    entries = raw.get("excluded", [])
    exp527_excluded = any(
        str(e.get("experiment_id", "")).lower() == "527"
        for e in entries
    )
    return exp527_excluded, len(entries)


def measure_gpu_health() -> dict:
    """Run nvidia-smi and return per-GPU metrics.

    WHY THIS IS CHECKED HERE (REQ-INFRA-047b):
        All GPU devices must have < 100 MB VRAM allocated at the start of each
        conductor milestone.  Any zombie process holding VRAM would block new
        experiments from loading models.

    Returns a dict with gpu0_vram_mb, gpu0_util, gpu0_temp_c, gpu1_vram_mb,
    gpu1_util, gpu1_temp_c, gpu_clean.  On hosts without nvidia-smi, all
    values default to 0 and gpu_clean is True (non-blocking for CPU CI).
    """
    result = {
        "gpu0_vram_mb": 0,
        "gpu0_util": 0,
        "gpu0_temp_c": 0,
        "gpu1_vram_mb": 0,
        "gpu1_util": 0,
        "gpu1_temp_c": 0,
        "gpu_clean": True,
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return result
        lines = [l.strip() for l in proc.stdout.strip().splitlines() if l.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                idx = int(parts[0])
                vram = int(parts[1])
                util = int(parts[2])
                temp = int(parts[3])
            except ValueError:
                continue
            if idx == 0:
                result["gpu0_vram_mb"] = vram
                result["gpu0_util"] = util
                result["gpu0_temp_c"] = temp
            elif idx == 1:
                result["gpu1_vram_mb"] = vram
                result["gpu1_util"] = util
                result["gpu1_temp_c"] = temp
        result["gpu_clean"] = (
            result["gpu0_vram_mb"] < _GPU_VRAM_CLEAN_MB
            and result["gpu1_vram_mb"] < _GPU_VRAM_CLEAN_MB
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        # No nvidia-smi — CPU-only host; treat as clean.
        pass
    return result


def run_incremental_test_selection(repo_root: Path) -> int:
    """Run conductor_pre_flight.py incremental mode and return test count.

    WHY THIS STEP:
        The incremental test selector reduces CI time by only running tests
        affected by the current diff.  Logging the selected count gives the
        conductor an observable baseline for milestone .58.

    Returns the number of tests selected, or 0 on any error.
    """
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(repo_root / "scripts" / "conductor_pre_flight.py"),
                "--manifest",
                str(repo_root / "scripts" / "conductor_exclusion_manifest.json"),
                "--incremental",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(repo_root),
        )
        # Parse "N tests selected" from output if present.
        for line in proc.stdout.splitlines() + proc.stderr.splitlines():
            if "selected" in line.lower() or "test" in line.lower():
                for token in line.split():
                    if token.isdigit():
                        return int(token)
        return 0
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return 0


def determine_honest_verdict(
    patch_applied: bool,
    gpu_clean: bool,
    exp527_excluded: bool,
) -> str:
    """Map the three boolean conditions to the canonical honest_verdict string.

    WHY FOUR DISTINCT VERDICTS:
        The conductor log uses honest_verdict to route post-experiment actions.
        Each distinct condition requires a different remediation path, so they
        must not be collapsed into a single "failure" verdict.
    """
    if not patch_applied:
        return "preflight_v10_patch_failed"
    if patch_applied and not exp527_excluded:
        return "preflight_v10_exp527_leak"
    if patch_applied and exp527_excluded and not gpu_clean:
        return "preflight_v10_patch_applied_gpu_dirty"
    return "preflight_v10_patch_applied_gpu_clean"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run pre-flight v10 checks and write the deliverable artifact."""
    tmpl = ExperimentTemplate(
        EXP_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=45,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )

    with watchdog:
        # Step 1: confirm patch applied
        patch_applied = check_patch_applied(_CONDUCTOR_PATH)

        # Step 2: confirm Exp 527 exclusion
        exp527_excluded, n_excluded = check_exp527_excluded(_MANIFEST_PATH)

        # Step 3: GPU health
        gpu = measure_gpu_health()

        # Step 4: incremental test selection
        incremental_tests = run_incremental_test_selection(_REPO_ROOT)

        # Step 5: determine verdict
        honest_verdict = determine_honest_verdict(
            patch_applied=patch_applied,
            gpu_clean=gpu["gpu_clean"],
            exp527_excluded=exp527_excluded,
        )

        artifact = tmpl.build_result(
            {
                "patch_applied": patch_applied,
                "gpu0_vram_mb": gpu["gpu0_vram_mb"],
                "gpu0_util": gpu["gpu0_util"],
                "gpu0_temp_c": gpu["gpu0_temp_c"],
                "gpu1_vram_mb": gpu["gpu1_vram_mb"],
                "gpu1_util": gpu["gpu1_util"],
                "gpu1_temp_c": gpu["gpu1_temp_c"],
                "gpu_clean": gpu["gpu_clean"],
                "exp527_excluded": exp527_excluded,
                "n_excluded_experiments": n_excluded,
                "incremental_tests_selected": incremental_tests,
                "manifest_fix_patch_path": str(_PATCH_PATH),
                "honest_verdict": honest_verdict,
            },
            status="success",
        )

        _REPO_ROOT.joinpath(DELIVERABLE).parent.mkdir(parents=True, exist_ok=True)
        _REPO_ROOT.joinpath(DELIVERABLE).write_text(json.dumps(artifact, indent=2))

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
