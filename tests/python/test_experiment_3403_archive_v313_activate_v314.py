"""Tests for Exp 3403 Archive V313 Activate V314.

Spec coverage: REQ-REPORT-3403
"""

import json
import subprocess
from pathlib import Path
from carnot.reporting.archive_v313_activate_v314_3403 import write_artifact


def test_experiment_3403_module() -> None:
    """REQ-REPORT-3403: The python module writes the correct JSON artifact."""
    out_path = write_artifact()
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["status"] == "success"
    assert payload["archived_milestone"] == "2026.05.313"
    assert payload["activated_milestone"] == "2026.05.314"


def test_experiment_3403_script() -> None:
    """REQ-REPORT-3403: The script delegates to the module successfully."""
    script_path = Path("scripts/experiment_3403_archive_v313_activate_v314.py")
    res = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
    assert res.returncode == 0
    assert "success" in res.stdout
