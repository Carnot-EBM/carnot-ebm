"""Tests for experiment 1102 — HuggingFace Spaces gallery update.

REQ-WOPR-GALLERY-001: NQueensGame must appear in ALL_GAMES.
SCENARIO-DEPLOY-001: gallery update script produces valid artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULT_PATH = REPO_ROOT / "results" / "experiment_1102_hf_spaces_gallery_update.json"
INIT_PATH = REPO_ROOT / "spaces" / "wopr-games" / "games" / "__init__.py"
NQUEENS_PATH = REPO_ROOT / "spaces" / "wopr-games" / "games" / "nqueens.py"


def _load_result() -> dict:
    assert RESULT_PATH.exists(), f"Result artifact missing: {RESULT_PATH}"
    return json.loads(RESULT_PATH.read_text())


def test_result_artifact_exists():
    # REQ-WOPR-GALLERY-001
    assert RESULT_PATH.exists()


def test_result_honest_verdict_valid():
    # SCENARIO-DEPLOY-001: verdict must be one of the defined enum values
    result = _load_result()
    valid_verdicts = {
        "gallery_updated_n_queens_live",
        "deploy_attempted_verify_pending",
        "upstream_cartridge_missing",
        "hf_token_not_found",
        "failed",
    }
    assert result["honest_verdict"] in valid_verdicts


def test_result_schema_fields_present():
    # Every required field must exist in the artifact
    required = [
        "cartridge_found",
        "hf_token_found",
        "hf_token_source",
        "n_cartridges_deployed",
        "app_py_updated",
        "deploy_attempted",
        "gallery_updated",
        "live_http_status",
        "honest_verdict",
    ]
    result = _load_result()
    for field in required:
        assert field in result, f"Missing field: {field}"


def test_nqueens_cartridge_file_exists():
    # REQ-WOPR-GALLERY-001: the cartridge source must be present
    assert NQUEENS_PATH.exists(), "nqueens.py missing from games/"


def test_nqueens_registered_in_all_games():
    # REQ-WOPR-GALLERY-001: NQueensGame must appear in games/__init__.py ALL_GAMES
    assert INIT_PATH.exists()
    content = INIT_PATH.read_text()
    assert "NQueensGame" in content
    assert "ALL_GAMES" in content
    # Verify the instantiation line is present
    assert "NQueensGame()" in content


def test_result_n_cartridges_deployed_gte_5():
    # After adding N-Queens we should have at least 5 cartridges
    result = _load_result()
    assert result["n_cartridges_deployed"] >= 5


def test_result_app_py_updated_true():
    result = _load_result()
    assert result["app_py_updated"] is True


def test_result_cartridge_found_true():
    result = _load_result()
    assert result["cartridge_found"] is True
