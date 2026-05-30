"""Tests for Capstone v314 aggregation."""
import pytest
from carnot.reporting.capstone_v314_3415 import run_capstone

def test_run_capstone():
    """Verify run_capstone produces a valid artifact dictionary."""
    result = run_capstone()
    assert result["schema"] == "carnot.milestone_capstone.v314.v1"
    assert result["experiment_id"] == "exp3415"
    assert result["milestone"] == "2026.05.314"
    assert "upstreams" in result
    assert "exp3404" in result["upstreams"]
    assert "reproducibility_checksum" in result
    assert result["reproducibility_checksum"] != ""
