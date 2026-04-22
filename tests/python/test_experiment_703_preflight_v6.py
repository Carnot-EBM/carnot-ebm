"""Tests for experiment_703_preflight_v6 retirement governance and JEPA v15/v16 cascade blocks.

WHY THIS TEST FILE EXISTS:
    The Slowest-5 composition was UNCHANGED for 5 consecutive milestones (2026.04.53),
    the longest frozen streak in project history.  Six experiments exceeded the 3-milestone
    retirement threshold but were deferred.  This test suite validates that:
    1. All 7 retirement files exist with the correct schema and honest_verdict.
    2. The conductor exclusion manifest contains entries for all 7 experiments.
    3. JEPA v15 and v16 cascades are both blocked in the manifest.
    4. The conductor pre-flight script runs without error and reports "Excluded experiments".

Spec: REQ-INFRA-039, REQ-INFRA-040, SCENARIO-INFRA-048, SCENARIO-INFRA-049
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
_RESULTS_DIR = _REPO_ROOT / "results"
_MANIFEST_PATH = _REPO_ROOT / "scripts" / "conductor_exclusion_manifest.json"
_PRE_FLIGHT_SCRIPT = _REPO_ROOT / "scripts" / "conductor_pre_flight.py"

# All 7 experiments that must be formally retired in milestone 2026.04.54.
_RETIREMENT_IDS = [346, 380, 381, 382, 383, 410, 425]


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-048: All seven experiments retired with correct schema.
# Traces to REQ-INFRA-039.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", _RETIREMENT_IDS)
def test_retirement_file_has_valid_schema(exp_id: int) -> None:
    """Each retirement file must be valid JSON with schema='carnot.retirement.v1' and status='retired'.

    WHY: Downstream tooling reads these files to confirm an experiment is permanently
    retired.  A missing or malformed schema field silently corrupts the retirement record
    and allows the conductor to re-schedule the experiment.  Traces to REQ-INFRA-039,
    SCENARIO-INFRA-048.
    """
    retirement_file = _RESULTS_DIR / f"experiment_{exp_id}_retired.json"
    assert retirement_file.exists(), (
        f"Retirement file missing for Exp {exp_id}: {retirement_file}"
    )

    data = json.loads(retirement_file.read_text())
    assert data.get("schema") == "carnot.retirement.v1", (
        f"Exp {exp_id} retirement file must have schema='carnot.retirement.v1', got: {data.get('schema')}"
    )
    assert data.get("status") == "retired", (
        f"Exp {exp_id} retirement file must have status='retired', got: {data.get('status')}"
    )
    assert data.get("experiment") == exp_id, (
        f"Exp {exp_id} retirement file 'experiment' field mismatch, got: {data.get('experiment')}"
    )
    assert data.get("honest_verdict") == "retired_governance_action", (
        f"Exp {exp_id} retirement file must have honest_verdict='retired_governance_action', "
        f"got: {data.get('honest_verdict')}.  "
        "This distinguishes milestone .54 governance retirements from earlier .52/.53 records."
    )
    assert "consecutive_appearances" in data, (
        f"Exp {exp_id} retirement file missing 'consecutive_appearances' field"
    )
    assert "cumulative_overhead_min" in data, (
        f"Exp {exp_id} retirement file missing 'cumulative_overhead_min' field"
    )
    assert "superseded_by" in data, (
        f"Exp {exp_id} retirement file missing 'superseded_by' field"
    )


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-049: JEPA v15 and v16 cascade blocks present in manifest.
# Traces to REQ-INFRA-040.
# ---------------------------------------------------------------------------

def test_manifest_contains_all_seven_experiment_ids() -> None:
    """conductor_exclusion_manifest.json must contain entries for all 7 retired experiments.

    WHY: A retirement file on disk without a matching manifest entry means the conductor
    can still schedule the experiment.  Both layers must agree.  Traces to REQ-INFRA-039,
    SCENARIO-INFRA-048.
    """
    assert _MANIFEST_PATH.exists(), f"Manifest not found: {_MANIFEST_PATH}"
    manifest = json.loads(_MANIFEST_PATH.read_text())
    excluded = manifest.get("excluded", [])
    manifest_ids = {str(e.get("experiment_id", "")) for e in excluded}

    for exp_id in _RETIREMENT_IDS:
        assert str(exp_id) in manifest_ids, (
            f"Exp {exp_id} missing from conductor_exclusion_manifest.json — "
            "conductor will re-schedule a formally retired experiment."
        )


def test_manifest_blocks_jepa_v15_cascade() -> None:
    """conductor_exclusion_manifest.json must block jepa_v15_cascade.

    WHY: JEPA v15 OOD AUC=0.4751 is below random chance (0.5).  Enabling the v15
    cascade actively inverts the correctness signal.  Traces to REQ-INFRA-040,
    SCENARIO-INFRA-049.
    """
    assert _MANIFEST_PATH.exists(), f"Manifest not found: {_MANIFEST_PATH}"
    manifest = json.loads(_MANIFEST_PATH.read_text())
    excluded = manifest.get("excluded", [])
    manifest_ids = {str(e.get("experiment_id", "")) for e in excluded}

    assert "jepa_v15_cascade" in manifest_ids, (
        "jepa_v15_cascade missing from conductor_exclusion_manifest.json.  "
        "JEPA v15 OOD AUC=0.4751 (below random) must remain blocked until v17 OOD AUC >= 0.75."
    )


def test_manifest_blocks_jepa_v16_cascade() -> None:
    """conductor_exclusion_manifest.json must block jepa_v16_cascade.

    WHY: JEPA v16 OOD AUC=0.4759 is also below random chance (0.5).  v16 does not fix
    the pure_loss_anti_correlation problem despite switching to InfoNCE.  Traces to
    REQ-INFRA-040, SCENARIO-INFRA-049.
    """
    assert _MANIFEST_PATH.exists(), f"Manifest not found: {_MANIFEST_PATH}"
    manifest = json.loads(_MANIFEST_PATH.read_text())
    excluded = manifest.get("excluded", [])
    manifest_ids = {str(e.get("experiment_id", "")) for e in excluded}

    assert "jepa_v16_cascade" in manifest_ids, (
        "jepa_v16_cascade missing from conductor_exclusion_manifest.json.  "
        "JEPA v16 OOD AUC=0.4759 (below random) must remain blocked until v17 OOD AUC >= 0.75."
    )


def test_conductor_pre_flight_runs_without_error() -> None:
    """conductor_pre_flight.py must exit 0 and print 'Excluded experiments'.

    WHY: The pre-flight script is the conductor's gate for checking the manifest before
    any experiments run.  If it fails or does not read the manifest, retirements have no
    effect on the running system.  Traces to REQ-INFRA-039, REQ-INFRA-040.
    """
    result = subprocess.run(
        [sys.executable, str(_PRE_FLIGHT_SCRIPT), "--manifest", str(_MANIFEST_PATH)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"conductor_pre_flight.py exited with code {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "Excluded experiments" in result.stdout, (
        "conductor_pre_flight.py output does not contain 'Excluded experiments' — "
        "manifest was not read or is empty.  stdout:\n" + result.stdout
    )
