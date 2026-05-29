"""Tests for Exp 3377 Archive V310 Activate V311.

Spec coverage: REQ-REPORT-3377
"""

import json
import subprocess
from pathlib import Path
from carnot.reporting.archive_v310_activate_v311_3377 import write_artifact

def test_experiment_3377_module() -> None:
    """REQ-REPORT-3377: The python module writes the correct JSON artifact."""
    out_path = write_artifact()
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["status"] == "success"
    assert payload["archived_milestone"] == "2026.05.310"
    assert payload["activated_milestone"] == "2026.05.311"

def test_experiment_3377_script() -> None:
    """REQ-REPORT-3377: The script delegates to the module successfully."""
    script_path = Path("scripts/experiment_3377_archive_v310_activate_v311.py")
    res = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
    assert res.returncode == 0
    assert "success" in res.stdout
