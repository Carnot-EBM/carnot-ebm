"""Tests for Exp 3400 matrix v38.

Spec refs: REQ-REPORT-3400, SCENARIO-REPORT-3400.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from carnot.reporting import cross_corpus_matrix_v38_3400 as mod

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/research-reporting/spec.md"


def test_req_report_3400_spec_anchor_declares_matrix_schema() -> None:
    """REQ-REPORT-3400: OpenSpec declares matrix v38 before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-3400" in spec
    assert "SCENARIO-REPORT-3400" in spec


def test_scenario_report_3400_builds_v38(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3400: builds matrix v38 from .312 and .313."""
    
    res = tmp_path / "results"
    res.mkdir()
    f1 = res / "experiment_3390_capstone_v312.json"
    f1.write_text(json.dumps({"milestone": "2026.05.312"}), encoding="utf-8")
    f2 = res / "experiment_3402_capstone_v313.json"
    f2.write_text(json.dumps({"milestone": "2026.05.313"}), encoding="utf-8")
    
    artifact = mod.build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    
    assert artifact["experiment_id"] == "exp3400"
    assert artifact["task_id"] == "exp3400-cross-corpus-matrix-v38"
    assert "experiment_3390_capstone_v312.json" in artifact["gathered_artifacts"]
    assert "experiment_3402_capstone_v313.json" in artifact["gathered_artifacts"]
    assert artifact["honest_verdict"].startswith("complete:")


def test_writer_and_validation(tmp_path: Path) -> None:
    out = mod.write_artifact(tmp_path)
    assert out.is_file()

    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact({})
    
    with pytest.raises(ValueError, match="honest_verdict must begin with complete:"):
        artifact = mod.build_artifact(tmp_path)
        artifact["honest_verdict"] = "blocked:"
        mod.validate_artifact(artifact)

    # coverage for malformed file branch
    res = tmp_path / "results"
    res.mkdir(exist_ok=True)
    f_bad = res / "experiment_bad.json"
    f_bad.write_text("{bad", encoding="utf-8")
    mod.build_artifact(tmp_path)

