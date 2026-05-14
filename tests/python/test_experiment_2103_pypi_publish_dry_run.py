"""
Tests for Experiment 2103: PyPI Publish Dry Run.

REQ-PUBLISH-025
SCENARIO-PUBLISH-027
"""
import json
from pathlib import Path

from carnot.reporting.experiment_2103_pypi_publish_dry_run import (
    get_publish_dry_run_results,
    write_results,
)


def test_experiment_2103_results():
    """Test the structure and values of the PyPI publish dry run results."""
    res = get_publish_dry_run_results()
    assert res["schema"] == "carnot.phase1_pypi_publish_dry_run.v1"
    assert res["pyproject_path"] == "pyproject.toml"
    assert res["sdist_filename"].startswith("carnot_ebm")
    assert res["wheel_filename"].startswith("carnot_ebm")
    assert isinstance(res["sdist_size_bytes"], int)
    assert isinstance(res["wheel_size_bytes"], int)
    assert res["twine_check_passed"] is True
    assert res["package_name_prefix_correct"] is True
    assert res["n_samples"] == 1
    assert res["acceptance_gate_passed"] is True
    assert "shipped:" in res["honest_verdict"]


def test_experiment_2103_write(tmp_path: Path):
    """Test writing the results JSON."""
    output_file = tmp_path / "experiment_2103_pypi_publish_dry_run.json"
    write_results(str(output_file))
    assert output_file.exists()

    with open(output_file, "r") as f:
        data = json.load(f)
    assert data["schema"] == "carnot.phase1_pypi_publish_dry_run.v1"
