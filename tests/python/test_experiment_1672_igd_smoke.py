"""Test for experiment_1672_igd_smoke.py. (REQ-IGD-001)"""

import json
import os
from pathlib import Path

from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV
from experiment_1672_igd_smoke import main


def _artifact_path(tmp_path: Path) -> Path:
    artifact_root = os.environ.get(ARTIFACT_ROOT_ENV)
    if artifact_root is not None:
        return Path(artifact_root) / "experiment_1672_igd.json"
    return tmp_path / "results" / "experiment_1672_igd.json"


def test_experiment_1672_igd_smoke(monkeypatch, tmp_path):
    """Test the script produces the correct JSON output."""
    monkeypatch.chdir(tmp_path)
    output_file = _artifact_path(tmp_path)

    main()

    assert output_file.exists()

    with open(output_file) as f:
        data = json.load(f)

    assert data["experiment_id"] == 1672
    assert "metrics" in data
    assert "satisfied_clauses" in data["metrics"]
    assert "total_clauses" in data["metrics"]
