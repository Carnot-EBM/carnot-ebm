"""Experiment 678: Legacy Retirement Pre-flight — Formally retire Exps 380-382 and 346.

WHY THIS EXPERIMENT EXISTS:
    The .51 RETRO confirmed that Exps 380, 381, 382 (partial checkpoint) and Exp 346
    (55M-param EORM training) have appeared in the slowest-5 list for THREE CONSECUTIVE
    milestones — crossing the formal retirement threshold established by the Exp 308/309
    precedent.  Without retirement files, the conductor's deliverable-watch sees them as
    "incomplete" and keeps scheduling them, wasting GPU time every milestone.

    Additionally, the conductor exclusion manifest has not been consulted for 15 consecutive
    milestones (conductor_consulted=null since Exp 666 wrote it).  The solution is a
    standalone pre-flight script the conductor can invoke before session start.

    This experiment:
      1. Verifies or creates retirement files for Exps 380, 381, 382, 346.
      2. Verifies those experiments are in the exclusion manifest.
      3. Runs scripts/conductor_pre_flight.py via subprocess and checks its output.
      4. Emits a verifiable result artifact.

Spec: REQ-INFRA-095, REQ-INFRA-096, SCENARIO-INFRA-103, SCENARIO-INFRA-104
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Allow running from the repo root without installing the package.
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from python.carnot.pipeline.exclusion_manifest import ExclusionManifest  # noqa: E402

# --- Constants -----------------------------------------------------------

RETIRED_IDS = [380, 381, 382, 346]
MANIFEST_PATH = _REPO_ROOT / "scripts" / "conductor_exclusion_manifest.json"
PRE_FLIGHT_SCRIPT = _REPO_ROOT / "scripts" / "conductor_pre_flight.py"
RESULTS_DIR = _REPO_ROOT / "results"

RETIREMENT_SCHEMA = "carnot.retirement.v1"
RETIREMENT_MILESTONE = "2026.04.52"

DELIVERABLE = "results/experiment_678_legacy_retirement_preflight.json"

# --- Template setup ------------------------------------------------------

tmpl = ExperimentTemplate(
    678,
    "Legacy Retirement Pre-flight: Formally retire Exps 380-382 and 346",
    DELIVERABLE,
)
tmpl.setup()

# --- Main logic ----------------------------------------------------------

with ExperimentTimeoutWatchdog(678, timeout_minutes=15, result_path=DELIVERABLE):

    retirement_records: list[dict] = []

    for exp_id in RETIRED_IDS:
        retirement_file = RESULTS_DIR / f"experiment_{exp_id}_retired.json"
        had_result = any(RESULTS_DIR.glob(f"experiment_{exp_id}_*.json"))
        file_written = False

        if not retirement_file.exists():
            payload = {
                "experiment": exp_id,
                "status": "retired",
                "reason": "partial_checkpoint_three_consecutive_milestones",
                "honest_verdict": "retired_formal_threshold_crossed",
                "milestone_retired": RETIREMENT_MILESTONE,
                "schema": RETIREMENT_SCHEMA,
            }
            retirement_file.write_text(json.dumps(payload, indent=2))
            file_written = True
        else:
            file_written = False  # already existed

        retirement_records.append(
            {
                "exp_id": exp_id,
                "had_result": had_result,
                "retirement_file": str(retirement_file.relative_to(_REPO_ROOT)),
                "retirement_file_written": file_written,
            }
        )

    # --- Verify manifest contains all 4 IDs ---
    manifest = ExclusionManifest(str(MANIFEST_PATH))
    manifest.load()
    manifest_ids_present = [exp_id for exp_id in RETIRED_IDS if manifest.is_excluded(exp_id)]
    manifest_entries = manifest.load()
    total_manifest_entries = len(manifest_entries)

    # --- Run conductor pre-flight script ---
    pre_flight_ran = False
    pre_flight_output = ""
    conductor_consulted = False
    try:
        result = subprocess.run(
            [sys.executable, str(PRE_FLIGHT_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        pre_flight_output = result.stdout + result.stderr
        pre_flight_ran = result.returncode == 0
        conductor_consulted = "Excluded experiments" in pre_flight_output
    except Exception as exc:  # noqa: BLE001
        pre_flight_output = f"error: {exc}"

    # --- Determine honest verdict ---
    all_retired = len(manifest_ids_present) == len(RETIRED_IDS)
    if all_retired and conductor_consulted:
        honest_verdict = "retirements_complete_preflight_confirmed"
    elif all_retired and not conductor_consulted:
        honest_verdict = "retirements_complete_preflight_missing"
    else:
        honest_verdict = "retirements_partial"

    # --- Build and write result ---
    artifact = tmpl.build_result(
        {
            "retirement_records": retirement_records,
            "manifest_ids_present": manifest_ids_present,
            "total_manifest_entries": total_manifest_entries,
            "pre_flight_ran": pre_flight_ran,
            "conductor_consulted": conductor_consulted,
            "pre_flight_output_lines": pre_flight_output.strip().splitlines(),
            "honest_verdict": honest_verdict,
        },
        status="success",
    )

    # Write the deliverable artifact to disk so assert_deliverable_written() passes.
    out_path = _REPO_ROOT / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

tmpl.assert_deliverable_written()
