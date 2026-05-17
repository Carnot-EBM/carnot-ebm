"""Tests for experiment 2167."""
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
import experiment_2167_archive as exp  # noqa: E402

def test_main_writes_deliverable(monkeypatch):
    monkeypatch.chdir(_REPO_ROOT)
    exp.main()
    deliverable = _REPO_ROOT / "results" / "experiment_2167_archive.json"
    assert deliverable.exists()
    data = json.loads(deliverable.read_text())
    assert data["experiment"] == 2167
    assert data["status"] == "success"
    assert data["honest_verdict"] == "archive_complete"
