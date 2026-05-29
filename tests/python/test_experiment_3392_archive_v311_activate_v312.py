"""Tests for Exp 3392 Archive V311 Activate V312.

Spec coverage: REQ-REPORT-3392
"""

import json
import subprocess
from pathlib import Path
from carnot.reporting.archive_v311_activate_v312_3392 import write_artifact

def test_experiment_3392_module() -> None:
    """REQ-REPORT-3392: The python module writes the correct JSON artifact."""
    out_path = write_artifact()
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["status"] == "success"
    assert payload["archived_milestone"] == "2026.05.311"
    assert payload["activated_milestone"] == "2026.05.312"

def test_experiment_3392_script() -> None:
    """REQ-REPORT-3392: The script delegates to the module successfully."""
    script_path = Path("scripts/experiment_3392_archive_v311_activate_v312.py")
    res = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
    assert res.returncode == 0
    assert "success" in res.stdout
