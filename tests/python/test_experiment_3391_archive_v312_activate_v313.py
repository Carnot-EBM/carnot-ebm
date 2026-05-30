"""Tests for Exp 3391 Archive V312 Activate V313.

Spec coverage: REQ-REPORT-3391
"""

import json
import subprocess
from pathlib import Path
from carnot.reporting.archive_v312_activate_v313_3391 import write_artifact


def test_experiment_3391_module() -> None:
    """REQ-REPORT-3391: The python module writes the correct JSON artifact."""
    out_path = write_artifact()
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["status"] == "success"
    assert payload["archived_milestone"] == "2026.05.312"
    assert payload["activated_milestone"] == "2026.05.313"


def test_experiment_3391_script() -> None:
    """REQ-REPORT-3391: The script delegates to the module successfully."""
    script_path = Path("scripts/experiment_3391_archive_v312_activate_v313.py")
    res = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
    assert res.returncode == 0
    assert "success" in res.stdout
