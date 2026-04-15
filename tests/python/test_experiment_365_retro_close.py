"""Tests for scripts/experiment_365_retro_close.py — RETRO-012/013/014 close.

Coverage targets (100% required)
---------------------------------
- run_experiment: all three RETROs closed in the happy path
- run_experiment: RETRO-012 stays open when env script export fails (edge case)
- Artifact schema: 'carnot.retro_close.v2', all required fields present
- env_script_created: True when scripts/ dir exists in repo_root
- missing_jsons_audit: list type, reflects real directory state
- all_closed: True in happy path
- retro_items_closed / retro_items_open: correct split

Spec: REQ-INFRA-015, REQ-INFRA-016,
      SCENARIO-INFRA-016, SCENARIO-INFRA-017, SCENARIO-INFRA-018
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_365_retro_close import (
    EXP_ID,
    MODULE_PRIMARY_EXPS,
    RETRO_ITEMS,
    TITLE,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_repo(tmp_path: Path, *, add_missing_jsons: bool = False) -> Path:
    """Create a minimal fake repo structure for testing run_experiment."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "checkpoints").mkdir(parents=True)
    if add_missing_jsons:
        for eid in MODULE_PRIMARY_EXPS:
            (tmp_path / "results" / f"experiment_{eid}_results.json").write_text("{}")
    return tmp_path


# ---------------------------------------------------------------------------
# Artifact schema
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    """Validate the artifact shape produced by run_experiment."""

    def test_required_result_fields_present(self, tmp_path: Path) -> None:
        """All REQUIRED_RESULT_FIELDS must be in the artifact."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        for field in ("experiment", "schema", "run_date", "started_at",
                      "finished_at", "duration_s", "status", "title"):
            assert field in artifact, f"Missing field: {field}"

    def test_schema_is_retro_close_v2(self, tmp_path: Path) -> None:
        """retro_schema field must be 'carnot.retro_close.v2'.

        ExperimentTemplate.build_result() sets 'schema' to sorted key list;
        the retro-specific version identifier is stored as 'retro_schema'.
        """
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        assert artifact["retro_schema"] == "carnot.retro_close.v2"

    def test_retro_specific_fields_present(self, tmp_path: Path) -> None:
        """Retro-specific keys must be present."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        for key in ("retro_items_closed", "retro_items_open",
                    "env_script_created", "missing_jsons_audit", "all_closed"):
            assert key in artifact, f"Missing retro field: {key}"

    def test_experiment_id(self, tmp_path: Path) -> None:
        """experiment field must be 365."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        assert artifact["experiment"] == EXP_ID

    def test_title_present(self, tmp_path: Path) -> None:
        """title field is non-empty."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        assert artifact["title"] == TITLE


# ---------------------------------------------------------------------------
# RETRO-012 closure
# ---------------------------------------------------------------------------


class TestRetro012:
    def test_env_script_created(self, tmp_path: Path) -> None:
        """env_script_created is True when scripts/ dir exists."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        assert artifact["env_script_created"] is True

    def test_env_script_file_exists_on_disk(self, tmp_path: Path) -> None:
        """scripts/conductor_gpu_env.sh is written to disk."""
        repo = _make_repo(tmp_path)
        run_experiment(repo)
        assert (repo / "scripts" / "conductor_gpu_env.sh").exists()

    def test_retro012_in_closed_list(self, tmp_path: Path) -> None:
        """RETRO-012 appears in retro_items_closed."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        closed_ids = [i["retro_id"] for i in artifact["retro_items_closed"]]
        assert "RETRO-012" in closed_ids

    def test_env_exports_verified_true(self, tmp_path: Path) -> None:
        """env_exports_verified is True when script has correct export."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        assert artifact["env_exports_verified"] is True

    def test_retro012_stays_open_when_verify_fails(self, tmp_path: Path) -> None:
        """If verify_env_script_exports returns False, RETRO-012 is not closed."""
        repo = _make_repo(tmp_path)
        with patch(
            "scripts.experiment_365_retro_close.verify_env_script_exports",
            return_value=False,
        ):
            artifact = run_experiment(repo)
        open_ids = [i["retro_id"] for i in artifact["retro_items_open"]]
        assert "RETRO-012" in open_ids

    def test_env_script_path_in_artifact(self, tmp_path: Path) -> None:
        """env_script_path key is present and non-empty."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        assert "env_script_path" in artifact
        assert len(artifact["env_script_path"]) > 0


# ---------------------------------------------------------------------------
# RETRO-013 closure
# ---------------------------------------------------------------------------


class TestRetro013:
    def test_retro013_in_closed_list(self, tmp_path: Path) -> None:
        """RETRO-013 is always closed (documented gap, addressed by Exp 366)."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        closed_ids = [i["retro_id"] for i in artifact["retro_items_closed"]]
        assert "RETRO-013" in closed_ids

    def test_retro013_rationale_mentions_exp366(self, tmp_path: Path) -> None:
        """RETRO-013 rationale references Exp 366."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        item = next(
            i for i in artifact["retro_items_closed"]
            if i["retro_id"] == "RETRO-013"
        )
        assert "366" in item["rationale"]


# ---------------------------------------------------------------------------
# RETRO-014 closure
# ---------------------------------------------------------------------------


class TestRetro014:
    def test_retro014_in_closed_list(self, tmp_path: Path) -> None:
        """RETRO-014 is always closed (enforcer pattern documented)."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        closed_ids = [i["retro_id"] for i in artifact["retro_items_closed"]]
        assert "RETRO-014" in closed_ids

    def test_missing_jsons_audit_is_list(self, tmp_path: Path) -> None:
        """missing_jsons_audit is a list."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        assert isinstance(artifact["missing_jsons_audit"], list)

    def test_missing_jsons_all_missing_when_empty_results(self, tmp_path: Path) -> None:
        """All three IDs reported missing when results/ is empty."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        for eid in MODULE_PRIMARY_EXPS:
            assert eid in artifact["missing_jsons_audit"]

    def test_missing_jsons_empty_when_all_present(self, tmp_path: Path) -> None:
        """Empty list when all module-primary JSONs exist."""
        repo = _make_repo(tmp_path, add_missing_jsons=True)
        artifact = run_experiment(repo)
        assert artifact["missing_jsons_audit"] == []


# ---------------------------------------------------------------------------
# all_closed
# ---------------------------------------------------------------------------


class TestAllClosed:
    def test_all_closed_true_in_happy_path(self, tmp_path: Path) -> None:
        """all_closed is True when all three RETROs are resolved."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        assert artifact["all_closed"] is True

    def test_all_closed_false_when_retro012_open(self, tmp_path: Path) -> None:
        """all_closed is False when RETRO-012 could not be closed."""
        repo = _make_repo(tmp_path)
        with patch(
            "scripts.experiment_365_retro_close.verify_env_script_exports",
            return_value=False,
        ):
            artifact = run_experiment(repo)
        assert artifact["all_closed"] is False


# ---------------------------------------------------------------------------
# retro_items_closed / retro_items_open split
# ---------------------------------------------------------------------------


class TestRetroSplit:
    def test_three_closed_zero_open_in_happy_path(self, tmp_path: Path) -> None:
        """Happy path: 3 closed, 0 open."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        assert len(artifact["retro_items_closed"]) == 3
        assert len(artifact["retro_items_open"]) == 0

    def test_two_closed_one_open_when_retro012_fails(self, tmp_path: Path) -> None:
        """When RETRO-012 cannot close: 2 closed, 1 open."""
        repo = _make_repo(tmp_path)
        with patch(
            "scripts.experiment_365_retro_close.verify_env_script_exports",
            return_value=False,
        ):
            artifact = run_experiment(repo)
        assert len(artifact["retro_items_closed"]) == 2
        assert len(artifact["retro_items_open"]) == 1

    def test_closed_items_have_rationale(self, tmp_path: Path) -> None:
        """Each closed item has a non-empty rationale."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        for item in artifact["retro_items_closed"]:
            assert item.get("rationale"), f"{item['retro_id']} missing rationale"

    def test_closed_items_have_closed_by_exp(self, tmp_path: Path) -> None:
        """Each closed item records closed_by_exp == 365."""
        repo = _make_repo(tmp_path)
        artifact = run_experiment(repo)
        for item in artifact["retro_items_closed"]:
            assert item["closed_by_exp"] == EXP_ID


# ---------------------------------------------------------------------------
# RETRO_ITEMS and MODULE_PRIMARY_EXPS constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_retro_items_has_three_entries(self) -> None:
        """RETRO_ITEMS has exactly three entries."""
        assert len(RETRO_ITEMS) == 3

    def test_retro_item_ids(self) -> None:
        """RETRO_ITEMS IDs are RETRO-012, 013, 014."""
        ids = [r[0] for r in RETRO_ITEMS]
        assert ids == ["RETRO-012", "RETRO-013", "RETRO-014"]

    def test_module_primary_exps(self) -> None:
        """MODULE_PRIMARY_EXPS lists 357, 358, 362."""
        assert set(MODULE_PRIMARY_EXPS) == {357, 358, 362}
