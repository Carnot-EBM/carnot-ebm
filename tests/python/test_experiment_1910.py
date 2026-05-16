import json
import os
import pytest

def test_experiment_1910_blocked_tag():
    """Verify that experiment 1910 correctly reports blocked due to existing tag."""
    result_path = os.path.join(os.path.dirname(__file__), "../../results/experiment_1910_pypi_publish.json")
    with open(result_path, "r") as f:
        data = json.load(f)

    assert data["schema"] == "carnot.pypi_publish.v4_ci_tagged_release"
    assert data["experiment"] == 1910
    assert data["honest_verdict"] == "blocked_version_already_tagged_needs_bump"
    assert "version_already_tagged_needs_bump" in data["preconditions_checked"]
    assert data["acceptance_gate_passed"] is False
