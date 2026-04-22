"""Tests for experiment_692_preflight_v5 retirement and JEPA v15 manifest block.

Spec: REQ-INFRA-037, REQ-INFRA-038, SCENARIO-INFRA-046, SCENARIO-INFRA-047
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
_RETIREMENT_IDS = [425, 410, 383]


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-046: Retirement files created with required schema fields.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exp_id", _RETIREMENT_IDS)
def test_retirement_file_valid_json_with_schema(exp_id: int) -> None:
    """Each retirement file must be valid JSON with the required schema fields.

    WHY: Downstream tooling (conductor, retrospective scripts) reads these files.
    A missing field or invalid JSON would silently corrupt the retirement record.
    Traces to REQ-INFRA-037, SCENARIO-INFRA-046.
    """
    retirement_file = _RESULTS_DIR / f"experiment_{exp_id}_retired.json"
    assert retirement_file.exists(), f"Retirement file missing: {retirement_file}"

    data = json.loads(retirement_file.read_text())
    assert data.get("schema") == "carnot.retirement.v1", (
        f"Exp {exp_id} retirement file missing schema='carnot.retirement.v1'"
    )
    assert data.get("status") == "retired", (
        f"Exp {exp_id} retirement file missing status='retired'"
    )
    assert data.get("experiment") == exp_id, (
        f"Exp {exp_id} retirement file experiment field mismatch"
    )
    assert data.get("honest_verdict") == "retired_formal_threshold_crossed", (
        f"Exp {exp_id} retirement file missing correct honest_verdict"
    )


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-047: Manifest contains retirement entries and JEPA block.
# ---------------------------------------------------------------------------

def test_manifest_includes_retired_experiments() -> None:
    """conductor_exclusion_manifest.json must include entries for Exps 425, 410, 383.

    WHY: If an experiment is retired on disk but absent from the manifest, the
    conductor will continue to schedule it, defeating the purpose of retirement.
    Traces to REQ-INFRA-037, SCENARIO-INFRA-047.
    """
    assert _MANIFEST_PATH.exists(), f"Manifest not found: {_MANIFEST_PATH}"
    data = json.loads(_MANIFEST_PATH.read_text())
    entry_ids = {e.get("experiment_id") for e in data.get("excluded", [])}

    for exp_id in _RETIREMENT_IDS:
        assert exp_id in entry_ids, (
            f"Exp {exp_id} missing from conductor_exclusion_manifest.json"
        )


def test_manifest_includes_jepa_v15_cascade_block() -> None:
    """conductor_exclusion_manifest.json must include 'jepa_v15_cascade' block entry.

    WHY: JEPA v15 posted OOD AUC=0.4751 (below random chance, Exp 682, RETRO-072).
    The cascade of experiments depending on v15 must be blocked until v16 achieves
    OOD AUC >= 0.75.  A missing manifest entry means the cascade runs unchecked.
    Traces to REQ-INFRA-038, SCENARIO-INFRA-047.
    """
    assert _MANIFEST_PATH.exists(), f"Manifest not found: {_MANIFEST_PATH}"
    data = json.loads(_MANIFEST_PATH.read_text())
    entry_ids = {e.get("experiment_id") for e in data.get("excluded", [])}

    assert "jepa_v15_cascade" in entry_ids, (
        "jepa_v15_cascade block missing from conductor_exclusion_manifest.json"
    )


# ---------------------------------------------------------------------------
# SCENARIO-INFRA-047: conductor_pre_flight.py runs without ImportError.
# ---------------------------------------------------------------------------

def test_conductor_pre_flight_runs_without_error() -> None:
    """conductor_pre_flight.py must execute cleanly and print 'Excluded experiments'.

    WHY: The pre-flight script is the non-invasive way the conductor confirms the
    manifest was consulted.  If it crashes (ImportError, JSON error, etc.) the
    conductor loses visibility into exclusions.  manifest_consulted sentinel must
    appear in stdout.
    Traces to REQ-INFRA-037, REQ-INFRA-038, SCENARIO-INFRA-047.
    """
    result = subprocess.run(
        [sys.executable, str(_PRE_FLIGHT_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"conductor_pre_flight.py exited {result.returncode}:\n{result.stderr}"
    )
    assert "Excluded experiments" in result.stdout, (
        "conductor_pre_flight.py stdout missing 'Excluded experiments' sentinel"
    )
