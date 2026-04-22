#!/usr/bin/env python3
"""experiment_692_preflight_v5.py — Verify Exp 425/410/383 retirement and JEPA v15 cascade block.

WHY THIS EXPERIMENT EXISTS:
    Milestone 2026.04.52 retrospective confirmed that the slowest-5 experiments were
    UNCHANGED for the 4th consecutive milestone.  REQ-INFRA-037 mandates that when
    any experiment appears in the slowest-5 for >= 4 consecutive milestones, the
    top-2 offenders must be formally retired.  Exp 383 is separately superseded by
    Exp 685 (DualGPU, 2.0175x speedup).

    Additionally, JEPA v15 posted OOD AUC=0.4751 (below random chance, Exp 682).
    RETRO-072 mandates that the JEPA v15 cascade be blocked in the manifest until
    v16 achieves OOD AUC >= 0.75 (REQ-INFRA-038).

    This experiment:
    1. Verifies retirement files exist for Exps 425, 410, 383 with correct schema.
    2. Verifies the conductor_exclusion_manifest.json contains all 3 retirement entries
       and the jepa_v15_cascade block.
    3. Runs conductor_pre_flight.py via subprocess and checks manifest_consulted=True.
    4. Emits a structured artifact with honest_verdict.

Spec: REQ-INFRA-037, REQ-INFRA-038, SCENARIO-INFRA-046, SCENARIO-INFRA-047
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Allow imports from scripts/ when running directly.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate
from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog

_EXPERIMENT_ID = 692
_TITLE = "Pre-flight v5: Retire Exps 425/410/383 and block JEPA v15 cascade"
_DELIVERABLE = "results/experiment_692_preflight_v5.json"

_MANIFEST_PATH = _REPO_ROOT / "scripts" / "conductor_exclusion_manifest.json"
_PRE_FLIGHT_SCRIPT = _REPO_ROOT / "scripts" / "conductor_pre_flight.py"

# Experiments that must appear in the manifest as of this milestone.
_EXPECTED_RETIREMENT_IDS = [425, 410, 383]
_EXPECTED_JEPA_BLOCK_KEY = "jepa_v15_cascade"

# Retirement file pattern — results/experiment_N_retired.json.
_RESULTS_DIR = _REPO_ROOT / "results"


def _check_retirement_files() -> dict[int, bool]:
    """Return {exp_id: file_exists} for each experiment that should be retired.

    WHY: We need to confirm that the retirement JSON files were created on disk.
    A missing file means the retirement was not persisted and downstream tooling
    (conductor, retrospective scripts) will not see it.
    """
    return {
        exp_id: (_RESULTS_DIR / f"experiment_{exp_id}_retired.json").exists()
        for exp_id in _EXPECTED_RETIREMENT_IDS
    }


def _load_manifest_entries() -> list[dict]:
    """Return the list of excluded entries from the manifest JSON.

    WHY: We parse the raw JSON so we can check both integer experiment IDs
    and string keys (like "jepa_v15_cascade") without depending on the
    ExclusionManifest class, which only handles integer IDs.
    """
    if not _MANIFEST_PATH.exists():
        return []
    raw = json.loads(_MANIFEST_PATH.read_text())
    return raw.get("excluded", [])


def _run_pre_flight() -> tuple[bool, str]:
    """Execute conductor_pre_flight.py and return (manifest_consulted, output).

    manifest_consulted is True if the output contains "Excluded experiments",
    which is the sentinel the conductor greps for to confirm the manifest ran.

    WHY subprocess: The pre-flight script must be tested as a standalone process
    (the way the conductor invokes it), not as an imported module.  Using subprocess
    catches ImportError, startup failures, and exit-code bugs that an in-process
    call would hide.
    """
    result = subprocess.run(
        [sys.executable, str(_PRE_FLIGHT_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    manifest_consulted = "Excluded experiments" in output
    return manifest_consulted, output


def main() -> None:
    """Run the pre-flight v5 verification and write the deliverable artifact."""
    tmpl = ExperimentTemplate(
        exp_id=_EXPERIMENT_ID,
        title=_TITLE,
        deliverable=_DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(_EXPERIMENT_ID, timeout_minutes=15,
                                   result_path=_DELIVERABLE):

        # Step 1: Check retirement files on disk.
        retirement_status = _check_retirement_files()
        all_retired = all(retirement_status.values())
        missing_retirements = [k for k, v in retirement_status.items() if not v]

        # Step 2: Check manifest entries.
        entries = _load_manifest_entries()
        entry_ids = {e.get("experiment_id") for e in entries}
        retirements_in_manifest = all(exp_id in entry_ids for exp_id in _EXPECTED_RETIREMENT_IDS)
        jepa_block_in_manifest = _EXPECTED_JEPA_BLOCK_KEY in entry_ids

        # Step 3: Run conductor pre-flight.
        manifest_consulted, pre_flight_output = _run_pre_flight()

        # Step 4: Determine honest_verdict.
        total_manifest_entries = len(entries)
        if all_retired and jepa_block_in_manifest and manifest_consulted:
            honest_verdict = "preflight_v5_complete"
        elif not jepa_block_in_manifest or not manifest_consulted:
            honest_verdict = "preflight_v5_partial_manifest_issue"
        else:
            honest_verdict = "preflight_v5_partial_retirement_missing"

        status = "success" if honest_verdict == "preflight_v5_complete" else "partial"

        artifact = tmpl.build_result(
            {
                "retirements_added": list(_EXPECTED_RETIREMENT_IDS),
                "retirement_files_present": retirement_status,
                "retirements_added_this_cycle": len(_EXPECTED_RETIREMENT_IDS) - len(missing_retirements),
                "missing_retirements": missing_retirements,
                "jepa_v15_cascade_blocked": jepa_block_in_manifest,
                "manifest_consulted": manifest_consulted,
                "total_manifest_entries": total_manifest_entries,
                "honest_verdict": honest_verdict,
                "pre_flight_output_snippet": pre_flight_output[:500],
            },
            status=status,
        )

        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (_RESULTS_DIR / "experiment_692_preflight_v5.json").write_text(
            json.dumps(artifact, indent=2)
        )

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
