"""
Tests for Experiment 1704 Vivado Synthesis for q=3 Potts machine on KV260.
References: REQ-POTTS-008.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from carnot.experiment_artifacts import ARTIFACT_ROOT_ENV

import scripts.experiment_1704_kv260 as exp


def _artifact_path(tmp_path: Path) -> Path:
    artifact_root = os.environ.get(ARTIFACT_ROOT_ENV)
    if artifact_root is not None:
        return Path(artifact_root) / "experiment_1704_kv260.json"
    return tmp_path / "results" / "experiment_1704_kv260.json"


@pytest.fixture
def mock_subprocess_run():
    with patch("subprocess.run") as mock_run:
        yield mock_run


def test_rtl_file_not_found(tmp_path, monkeypatch, caplog):
    """
    Test when RTL file is not found.
    """
    monkeypatch.chdir(tmp_path)

    # Don't create the hardware directory
    exp.run_experiment()

    assert "RTL file not found" in caplog.text


def test_vivado_not_installed_artifact(tmp_path, monkeypatch, mock_subprocess_run):
    """
    SCENARIO-POTTS-008: Vivado not available -> synthesis_success=False.
    """
    # Mock subprocess.run to return non-zero (not found)
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_subprocess_run.return_value = mock_result

    # Change working directory so results/ goes into tmp_path
    monkeypatch.chdir(tmp_path)

    # Create the hardware directory structure in tmp_path
    hardware_dir = tmp_path / "hardware" / "kv260"
    hardware_dir.mkdir(parents=True, exist_ok=True)
    (hardware_dir / "potts_sampler_v1.v").write_text("// dummy")

    exp.run_experiment()

    artifact_path = _artifact_path(tmp_path)
    assert artifact_path.exists()

    with open(artifact_path) as f:
        data = json.load(f)

    assert "synthesis_success" in data
    assert data["synthesis_success"] is False
    assert data["vivado_available"] is False
    assert data["honest_verdict"] == "vivado_not_installed"
    assert data["performance"] is None
    assert data["resource_utilization"] is None


def test_vivado_is_installed_synthesis_success(tmp_path, monkeypatch, mock_subprocess_run):
    """
    SCENARIO-POTTS-008: Vivado is installed and synthesis succeeds.
    """

    def side_effect(args, **kwargs):
        mock_res = MagicMock()
        if args[0] == "which" or args[0] == "vivado":
            mock_res.returncode = 0
            return mock_res
        mock_res.returncode = 1
        return mock_res

    mock_subprocess_run.side_effect = side_effect

    monkeypatch.chdir(tmp_path)

    hardware_dir = tmp_path / "hardware" / "kv260"
    hardware_dir.mkdir(parents=True, exist_ok=True)
    (hardware_dir / "potts_sampler_v1.v").write_text("// dummy")

    exp.run_experiment()

    artifact_path = _artifact_path(tmp_path)
    assert artifact_path.exists()

    with open(artifact_path) as f:
        data = json.load(f)

    assert data["vivado_available"] is True
    assert data["synthesis_success"] is True
    assert data["honest_verdict"] == "vivado_synthesis_successful"
    assert data["performance"] == "unknown"
    assert data["resource_utilization"] == "unknown"


def test_vivado_is_installed_synthesis_failed(tmp_path, monkeypatch, mock_subprocess_run):
    """
    SCENARIO-POTTS-008: Vivado is installed and synthesis fails.
    """

    def side_effect(args, **kwargs):
        mock_res = MagicMock()
        if args[0] == "which":
            mock_res.returncode = 0
            return mock_res
        elif args[0] == "vivado":
            mock_res.returncode = 1
            return mock_res
        mock_res.returncode = 1
        return mock_res

    mock_subprocess_run.side_effect = side_effect

    monkeypatch.chdir(tmp_path)

    hardware_dir = tmp_path / "hardware" / "kv260"
    hardware_dir.mkdir(parents=True, exist_ok=True)
    (hardware_dir / "potts_sampler_v1.v").write_text("// dummy")

    exp.run_experiment()

    artifact_path = _artifact_path(tmp_path)
    assert artifact_path.exists()

    with open(artifact_path) as f:
        data = json.load(f)

    assert data["vivado_available"] is True
    assert data["synthesis_success"] is False
    assert data["honest_verdict"] == "vivado_synthesis_failed"


def test_vivado_is_installed_synthesis_exception(tmp_path, monkeypatch, mock_subprocess_run):
    """
    SCENARIO-POTTS-008: Vivado is installed and an exception occurs.
    """

    def side_effect(args, **kwargs):
        if args[0] == "which":
            mock_res = MagicMock()
            mock_res.returncode = 0
            return mock_res
        elif args[0] == "vivado":
            raise RuntimeError("Unexpected error")
        mock_res = MagicMock()
        mock_res.returncode = 1
        return mock_res

    mock_subprocess_run.side_effect = side_effect

    monkeypatch.chdir(tmp_path)

    hardware_dir = tmp_path / "hardware" / "kv260"
    hardware_dir.mkdir(parents=True, exist_ok=True)
    (hardware_dir / "potts_sampler_v1.v").write_text("// dummy")

    exp.run_experiment()

    artifact_path = _artifact_path(tmp_path)
    assert artifact_path.exists()

    with open(artifact_path) as f:
        data = json.load(f)

    assert data["vivado_available"] is True
    assert data["synthesis_success"] is False
    assert data["honest_verdict"] == "error_Unexpected error"


def test_vivado_check_filenotfound(tmp_path, monkeypatch, mock_subprocess_run):
    """
    SCENARIO-POTTS-008: FileNotFoundError when checking for vivado.
    """

    def side_effect(args, **kwargs):
        raise FileNotFoundError("which not found")

    mock_subprocess_run.side_effect = side_effect

    monkeypatch.chdir(tmp_path)

    hardware_dir = tmp_path / "hardware" / "kv260"
    hardware_dir.mkdir(parents=True, exist_ok=True)
    (hardware_dir / "potts_sampler_v1.v").write_text("// dummy")

    exp.run_experiment()

    artifact_path = _artifact_path(tmp_path)
    assert artifact_path.exists()

    with open(artifact_path) as f:
        data = json.load(f)

    assert data["vivado_available"] is False
    assert data["synthesis_success"] is False
    assert data["honest_verdict"] == "vivado_not_installed"
