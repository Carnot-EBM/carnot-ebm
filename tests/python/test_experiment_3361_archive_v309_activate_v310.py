import json
import subprocess
from pathlib import Path
from carnot.reporting.archive_v309_activate_v310_3361 import write_artifact

def test_experiment_3361_module():
    out_path = write_artifact()
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert payload["status"] == "success"

def test_experiment_3361_script():
    script_path = Path("scripts/experiment_3361_archive_v309_activate_v310.py")
    res = subprocess.run(["python", str(script_path)], capture_output=True, text=True)
    assert res.returncode == 0
    assert "success" in res.stdout
