"""experiment_703_preflight_v6.py — Formal retirement of Exps 380-382, 346, 425, 410, 383
and JEPA v15+v16 cascade blocks.

WHY THIS EXPERIMENT EXISTS (governance failure, not a research question):
    The Slowest-5 composition was UNCHANGED for the FIFTH consecutive milestone (2026.04.53),
    the longest frozen streak in project history.  Six experiments exceeded the 3-milestone
    retirement threshold established by the Exp 308/309 precedent (RETRO-056) but were not
    retired, accumulating over 2,200 minutes of avoidable wall-clock overhead:

    - Exp 425: 17 consecutive milestones, 1,292 min cumulative overhead (MUST retire)
    - Exp 410: 14 consecutive milestones, 716 min cumulative overhead (MUST retire)
    - Exp 383: 8 consecutive milestones — DualGPU fix validated (Exp 685, 2.0175x speedup)
               but NOT deployed; retire and replace with Exp 685 pattern
    - Exp 380-382: 5 consecutive milestones, formal threshold crossed (>= 3 consecutive)
    - Exp 346: 5 consecutive milestones, formal threshold crossed

    Additionally, JEPA v16 OOD AUC=0.4759 is below random chance (0.5), same problem as
    v15 (OOD AUC=0.4751).  Both cascade entries must be blocked in the manifest until v17
    achieves OOD AUC >= 0.75.

    This experiment verifies all retirements are in place and the conductor pre-flight
    script reads the manifest correctly.

Spec: REQ-INFRA-039, REQ-INFRA-040, SCENARIO-INFRA-048, SCENARIO-INFRA-049
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Allow running from the repo root without install.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate
from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog

# The seven experiments that must be formally retired in this cycle.
# Maps experiment ID -> (consecutive_appearances, cumulative_overhead_min or None).
RETIRING_EXPERIMENTS: dict[int, tuple[int, int | None]] = {
    425: (17, 1292),
    410: (14, 716),
    383: (8, 248),
    380: (5, None),
    381: (5, None),
    382: (5, None),
    346: (5, None),
}

MANIFEST_PATH = _REPO_ROOT / "scripts" / "conductor_exclusion_manifest.json"
DELIVERABLE = "results/experiment_703_preflight_v6.json"


def _load_manifest() -> dict:
    """Load conductor_exclusion_manifest.json; return dict with 'excluded' list.

    WHY: We need the raw dict so we can check string IDs (jepa_v15_cascade,
    jepa_v16_cascade) alongside integer experiment IDs without type confusion.
    """
    if not MANIFEST_PATH.exists():
        return {"excluded": []}
    return json.loads(MANIFEST_PATH.read_text())


def _check_retirement_files(repo_root: Path) -> tuple[list[int], list[int]]:
    """Check which retirement files exist and have the expected schema.

    Returns (present, missing) lists of experiment IDs.

    WHY: The retirement files must exist AND have valid schema before we can
    claim retirements_added_this_cycle.  A missing file means governance
    action was deferred again — exactly the failure mode we are fixing.
    """
    present = []
    missing = []
    for exp_id in RETIRING_EXPERIMENTS:
        path = repo_root / "results" / f"experiment_{exp_id}_retired.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                # Validate required schema field — must be carnot.retirement.v1.
                if data.get("schema") == "carnot.retirement.v1" and data.get("status") == "retired":
                    present.append(exp_id)
                else:
                    missing.append(exp_id)
            except (json.JSONDecodeError, KeyError):
                missing.append(exp_id)
        else:
            missing.append(exp_id)
    return present, missing


def _check_manifest_entries(manifest: dict) -> tuple[list, list, bool, bool]:
    """Verify which experiment IDs are covered in the manifest.

    Returns (present_ids, missing_ids, jepa_v15_blocked, jepa_v16_blocked).

    WHY: The manifest is the conductor's gate.  If an experiment is retired
    in a JSON file but not in the manifest, the conductor can still schedule
    it.  Both layers must agree.
    """
    excluded = manifest.get("excluded", [])
    # Collect all experiment_ids as strings for uniform comparison.
    manifest_ids = {str(e.get("experiment_id", "")) for e in excluded}

    present = [exp_id for exp_id in RETIRING_EXPERIMENTS if str(exp_id) in manifest_ids]
    missing = [exp_id for exp_id in RETIRING_EXPERIMENTS if str(exp_id) not in manifest_ids]

    jepa_v15_blocked = "jepa_v15_cascade" in manifest_ids
    jepa_v16_blocked = "jepa_v16_cascade" in manifest_ids

    return present, missing, jepa_v15_blocked, jepa_v16_blocked


def _run_pre_flight(manifest_path: Path) -> tuple[bool, str]:
    """Run conductor_pre_flight.py as a subprocess and capture stdout.

    Returns (manifest_consulted, stdout_text).

    WHY: We run as a subprocess rather than importing the function directly
    so we verify the CLI entry-point works end-to-end, not just the library
    code.  The conductor invokes it as a shell command, so that is what we test.
    """
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "conductor_pre_flight.py"),
         "--manifest", str(manifest_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    stdout = result.stdout
    # The pre-flight script prints "Excluded experiments" when the manifest is read.
    manifest_consulted = "Excluded experiments" in stdout
    return manifest_consulted, stdout


def main() -> None:
    """Run Exp 703: verify retirements and JEPA cascade blocks, emit artifact."""
    tmpl = ExperimentTemplate(
        703,
        "Pre-flight v6: Retire Exps 380-382/346/425/410/383, block JEPA v15+v16 cascades",
        DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(703, timeout_minutes=20,
                                   result_path=str(_REPO_ROOT / DELIVERABLE)):

        manifest = _load_manifest()

        # Check retirement files on disk.
        retired_present, retired_missing = _check_retirement_files(tmpl._repo_root)

        # Check manifest entries.
        manifest_present, manifest_missing, jepa_v15_blocked, jepa_v16_blocked = (
            _check_manifest_entries(manifest)
        )

        # Run conductor pre-flight script.
        manifest_consulted, pre_flight_stdout = _run_pre_flight(MANIFEST_PATH)

        total_manifest_entries = len(manifest.get("excluded", []))
        retirements_added_this_cycle = len(retired_present)

        # Determine honest_verdict.
        if (
            len(retired_missing) == 0
            and len(manifest_missing) == 0
            and jepa_v15_blocked
            and jepa_v16_blocked
            and manifest_consulted
        ):
            honest_verdict = "preflight_v6_complete"
        elif len(retired_missing) > 0:
            honest_verdict = "preflight_v6_partial_retirement_missing"
        else:
            honest_verdict = "preflight_v6_partial_manifest_issue"

        artifact = tmpl.build_result(
            {
                "retirements_added": retired_present,
                "retirements_missing": retired_missing,
                "manifest_entries_present": manifest_present,
                "manifest_entries_missing": manifest_missing,
                "jepa_v15_cascade_blocked": jepa_v15_blocked,
                "jepa_v16_cascade_blocked": jepa_v16_blocked,
                "manifest_consulted": manifest_consulted,
                "total_manifest_entries": total_manifest_entries,
                "retirements_added_this_cycle": retirements_added_this_cycle,
                "honest_verdict": honest_verdict,
                "pre_flight_stdout_lines": len(pre_flight_stdout.splitlines()),
            },
            status="success" if honest_verdict == "preflight_v6_complete" else "partial",
        )

    import json as _json
    (tmpl._repo_root / DELIVERABLE).write_text(_json.dumps(artifact, indent=2) + "\n")
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
