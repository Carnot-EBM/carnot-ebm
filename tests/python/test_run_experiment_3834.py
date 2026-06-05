import json
import sys
import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# We need to import the script, but it's in scripts/ directory.
# Let's adjust sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.run_experiment_3834 import (
    check_preconditions,
    append_changelog,
    run_publication_gate,
    verify_fover_unchanged,
    write_artifact,
    main
)

def test_check_preconditions_false_not_exists(tmp_path):
    assert not check_preconditions(tmp_path / "missing.md")

def test_check_preconditions_false_not_writable(tmp_path):
    f = tmp_path / "readonly.md"
    f.touch()
    f.chmod(0o444)
    if os.access(f, os.W_OK):
        pytest.skip("Test requires OS that respects readonly")
    assert not check_preconditions(f)

def test_check_preconditions_true(tmp_path):
    f = tmp_path / "writable.md"
    f.touch()
    assert check_preconditions(f)

def test_append_changelog(tmp_path):
    f = tmp_path / "changelog.md"
    f.touch()
    append_changelog(f, "2026-06-04")
    content = f.read_text()
    assert "- 2026-06-04: Archive milestone .352" in content
    assert "results/experiment_3834" in content

@patch("subprocess.check_output")
def test_run_publication_gate(mock_check_output):
    mock_check_output.return_value = '{"paper_ready": true}'
    res = run_publication_gate("python")
    assert res == {"paper_ready": True}
    mock_check_output.assert_called_once()

def test_verify_fover_unchanged_no_source(tmp_path):
    gate_data = {}
    fover_unchanged, seed, checksum = verify_fover_unchanged(gate_data, tmp_path)
    assert not fover_unchanged
    assert seed == "unknown"
    assert checksum == "unknown"

def test_verify_fover_unchanged_success(tmp_path):
    gate_data = {"gates": {"G4": {"source": "fover.json"}}}
    fover_file = tmp_path / "fover.json"
    fover_file.write_text(json.dumps({
        "headline_auroc": 0.9131,
        "random_seeds_used": [1, 2, 3],
        "reproducibility_checksum": "abc"
    }))
    fover_unchanged, seed, checksum = verify_fover_unchanged(gate_data, tmp_path)
    assert fover_unchanged
    assert seed == "1, 2, 3"
    assert checksum == "abc"

def test_verify_fover_unchanged_fallback_keys(tmp_path):
    gate_data = {"gates": {"G4": {"source": "fover.json"}}}
    fover_file = tmp_path / "fover.json"
    fover_file.write_text(json.dumps({
        "production_auroc": 0.9131,
        "random_seed": "42",
        "reproducibility_checksum": "def"
    }))
    fover_unchanged, seed, checksum = verify_fover_unchanged(gate_data, tmp_path)
    assert fover_unchanged
    assert seed == "42"
    assert checksum == "def"

def test_verify_fover_unchanged_exception(tmp_path):
    gate_data = {"gates": {"G4": {"source": "fover.json"}}}
    fover_unchanged, seed, checksum = verify_fover_unchanged(gate_data, tmp_path)
    assert not fover_unchanged

def test_write_artifact(tmp_path):
    f = tmp_path / "art.json"
    write_artifact(f, True, True, "42", "chk")
    data = json.loads(f.read_text())
    assert data["paper_ready_at_boundary"] is True
    assert data["frozen_fover_auroc_unchanged"] is True
    assert data["random_seed"] == "42"
    assert data["reproducibility_checksum"] == "chk"

@patch("scripts.run_experiment_3834.check_preconditions")
def test_main_blocked_changelog(mock_check_preconditions, capsys):
    mock_check_preconditions.return_value = False
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    assert "blocked_changelog_not_writable" in capsys.readouterr().out

@patch("scripts.run_experiment_3834.check_preconditions")
@patch("scripts.run_experiment_3834.append_changelog")
@patch("scripts.run_experiment_3834.run_publication_gate")
def test_main_blocked_gate(mock_run_gate, mock_append, mock_check, capsys):
    mock_check.return_value = True
    mock_run_gate.side_effect = subprocess.CalledProcessError(1, "cmd")
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 1
    assert "blocked_gate_failed" in capsys.readouterr().out

@patch("scripts.run_experiment_3834.check_preconditions")
@patch("scripts.run_experiment_3834.append_changelog")
@patch("scripts.run_experiment_3834.run_publication_gate")
@patch("scripts.run_experiment_3834.verify_fover_unchanged")
@patch("scripts.run_experiment_3834.write_artifact")
def test_main_success(mock_write, mock_verify, mock_run_gate, mock_append, mock_check):
    mock_check.return_value = True
    mock_run_gate.return_value = {"paper_ready": True}
    mock_verify.return_value = (True, "42", "abc")
    main()
    mock_append.assert_called_once()
    mock_write.assert_called_once()
