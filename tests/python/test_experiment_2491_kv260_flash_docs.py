"""Tests for KV260 flash documentation experiment (exp2491)."""

import json
from pathlib import Path

from scripts.experiment_2491_kv260_flash_docs import generate_artifact, run


def test_generate_artifact():
    artifact = generate_artifact()
    assert artifact["experiment_id"] == "2491"
    assert artifact["dirtyjtag_kv260_compatible"] is False
    assert artifact["openocd_flash_feasible"] is False
    assert artifact["kv260_flash_requirements_written"] is True
    assert artifact["honest_verdict"].startswith("complete_")


def test_run(tmp_path: Path):
    out_file = run(tmp_path)
    assert out_file.exists()
    with out_file.open() as fh:
        loaded = json.load(fh)
    assert loaded["experiment_id"] == "2491"
    assert loaded["dirtyjtag_kv260_compatible"] is False
