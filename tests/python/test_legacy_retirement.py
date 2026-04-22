"""Tests for Exp 678 legacy retirement — retirement files, pre-flight script, manifest.

Spec: REQ-INFRA-095, REQ-INFRA-096, SCENARIO-INFRA-103, SCENARIO-INFRA-104

WHY THESE TESTS EXIST:
    Exp 678 formally retires Exps 380, 381, 382, and 346 by creating retirement
    placeholder files and adding them to the conductor exclusion manifest.  These tests
    verify the structural correctness of those artifacts so future milestones can trust
    the conductor will skip the retired experiments.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RETIRED_IDS = [380, 381, 382, 346]
MANIFEST_PATH = _REPO / "scripts" / "conductor_exclusion_manifest.json"
PRE_FLIGHT_SCRIPT = _REPO / "scripts" / "conductor_pre_flight.py"
DELIVERABLE = _REPO / "results" / "experiment_678_legacy_retirement_preflight.json"


# ---------------------------------------------------------------------------
# REQ-INFRA-095: ExclusionManifest must contain retired experiments
# ---------------------------------------------------------------------------


class TestRetirementFiles:
    """Retirement placeholder files exist and have the required schema.

    Spec: REQ-INFRA-095, SCENARIO-INFRA-103
    """

    @pytest.mark.parametrize("exp_id", RETIRED_IDS)
    def test_retirement_file_exists(self, exp_id: int) -> None:
        """Each retired experiment has a *_retired.json file in results/."""
        retirement_file = _REPO / "results" / f"experiment_{exp_id}_retired.json"
        assert retirement_file.exists(), f"Missing retirement file for Exp {exp_id}"

    @pytest.mark.parametrize("exp_id", RETIRED_IDS)
    def test_retirement_file_is_valid_json(self, exp_id: int) -> None:
        """Retirement file must be parseable JSON."""
        retirement_file = _REPO / "results" / f"experiment_{exp_id}_retired.json"
        data = json.loads(retirement_file.read_text())
        assert isinstance(data, dict)

    @pytest.mark.parametrize("exp_id", RETIRED_IDS)
    def test_retirement_file_has_schema_field(self, exp_id: int) -> None:
        """Retirement file must have a 'schema' field equal to 'carnot.retirement.v1'."""
        retirement_file = _REPO / "results" / f"experiment_{exp_id}_retired.json"
        data = json.loads(retirement_file.read_text())
        assert data.get("schema") == "carnot.retirement.v1"

    @pytest.mark.parametrize("exp_id", RETIRED_IDS)
    def test_retirement_file_status_is_retired(self, exp_id: int) -> None:
        """Retirement file must have status='retired'."""
        retirement_file = _REPO / "results" / f"experiment_{exp_id}_retired.json"
        data = json.loads(retirement_file.read_text())
        assert data.get("status") == "retired"

    @pytest.mark.parametrize("exp_id", RETIRED_IDS)
    def test_retirement_file_experiment_matches(self, exp_id: int) -> None:
        """The 'experiment' field in the retirement file must match the file's exp_id."""
        retirement_file = _REPO / "results" / f"experiment_{exp_id}_retired.json"
        data = json.loads(retirement_file.read_text())
        assert data.get("experiment") == exp_id


class TestExclusionManifest:
    """Exclusion manifest contains all 4 newly retired experiments.

    Spec: REQ-INFRA-095, SCENARIO-INFRA-103
    """

    def test_manifest_exists(self) -> None:
        """The conductor exclusion manifest must exist."""
        assert MANIFEST_PATH.exists()

    def test_manifest_is_valid_json(self) -> None:
        """Manifest must be parseable JSON with an 'excluded' key."""
        data = json.loads(MANIFEST_PATH.read_text())
        assert "excluded" in data

    @pytest.mark.parametrize("exp_id", RETIRED_IDS)
    def test_manifest_contains_retired_id(self, exp_id: int) -> None:
        """Each newly retired experiment ID must appear in the manifest."""
        data = json.loads(MANIFEST_PATH.read_text())
        ids = [e["experiment_id"] for e in data["excluded"]]
        assert exp_id in ids, f"Exp {exp_id} not found in exclusion manifest"

    def test_manifest_has_ten_or_more_entries(self) -> None:
        """After adding 4 new retirements the manifest must have >= 10 entries."""
        data = json.loads(MANIFEST_PATH.read_text())
        assert len(data["excluded"]) >= 10


# ---------------------------------------------------------------------------
# REQ-INFRA-096: conductor_pre_flight.py must run and print excluded IDs
# ---------------------------------------------------------------------------


class TestConductorPreFlight:
    """conductor_pre_flight.py is importable, runnable, and non-blocking.

    Spec: REQ-INFRA-096, SCENARIO-INFRA-104
    """

    def test_pre_flight_script_exists(self) -> None:
        """scripts/conductor_pre_flight.py must be present."""
        assert PRE_FLIGHT_SCRIPT.exists()

    def test_pre_flight_runs_without_error(self) -> None:
        """Running the script must exit 0."""
        result = subprocess.run(
            [sys.executable, str(PRE_FLIGHT_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"pre-flight exited non-zero:\n{result.stderr}"

    def test_pre_flight_prints_excluded_experiments(self) -> None:
        """Output must contain 'Excluded experiments' so the conductor can grep it."""
        result = subprocess.run(
            [sys.executable, str(PRE_FLIGHT_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "Excluded experiments" in result.stdout

    def test_pre_flight_lists_retired_ids(self) -> None:
        """Output must mention each of the 4 newly retired experiment IDs."""
        result = subprocess.run(
            [sys.executable, str(PRE_FLIGHT_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for exp_id in RETIRED_IDS:
            assert str(exp_id) in result.stdout, f"Exp {exp_id} not in pre-flight output"

    def test_pre_flight_exits_zero_with_missing_manifest(self, tmp_path: Path) -> None:
        """Script must exit 0 even when manifest is missing (non-blocking)."""
        missing = tmp_path / "nonexistent.json"
        result = subprocess.run(
            [sys.executable, str(PRE_FLIGHT_SCRIPT), "--manifest", str(missing)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Deliverable integrity check
# ---------------------------------------------------------------------------


class TestDeliverable:
    """Exp 678 deliverable JSON has the expected structure.

    Spec: REQ-INFRA-095, REQ-INFRA-096
    """

    def test_deliverable_exists(self) -> None:
        assert DELIVERABLE.exists()

    def test_deliverable_honest_verdict(self) -> None:
        data = json.loads(DELIVERABLE.read_text())
        assert data["honest_verdict"] == "retirements_complete_preflight_confirmed"

    def test_deliverable_conductor_consulted(self) -> None:
        data = json.loads(DELIVERABLE.read_text())
        assert data["conductor_consulted"] is True

    def test_deliverable_all_manifest_ids_present(self) -> None:
        data = json.loads(DELIVERABLE.read_text())
        for exp_id in RETIRED_IDS:
            assert exp_id in data["manifest_ids_present"]
