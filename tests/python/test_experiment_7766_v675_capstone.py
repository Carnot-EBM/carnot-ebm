"""REQ-REPORT-7766: V675 accounting must survive independent source checks."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

import carnot.experiment_7766_v675_capstone as capstone
from carnot.experiment_7753_v675_contract_methods import compare_contract
from carnot.experiment_7766_v675_capstone import (
    ROOT as SOURCE_ROOT,
    account,
    authority,
    build_artifact,
    cold_replay,
)

ROOT = SOURCE_ROOT
V675_FIXTURES = SOURCE_ROOT / "tests/python/fixtures"


@pytest.fixture(autouse=True)
def historical_v675_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use frozen V675 design bytes after the live roadmap advances to V676."""
    for relative, fixture in (
        ("research-roadmap.yaml", "roadmap_2026_09_675.yaml"),
        ("openspec/change-proposals/research-roadmap-vNEXT.md", "roadmap_design_2026_09_675.md"),
    ):
        content = (V675_FIXTURES / fixture).read_bytes()
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (tmp_path / "results").symlink_to(SOURCE_ROOT / "results", target_is_directory=True)
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)


def test_contract_matches_literal_sources() -> None:
    """SCENARIO-REPORT-7766-CONTRACT: the three authorities must agree."""
    source = authority(ROOT)
    assert source["comparison"]["passed"] is True
    assert len(source["tasks"]) == 14
    assert source["candidates"][0]["exists"] is False
    assert source["candidates"][1]["milestone"] == "2026.09.675"
    changed = deepcopy(source["roadmap"])
    changed["tasks"][4]["title"] = "Changed title"
    assert not compare_contract((ROOT / source["design_path"]).read_text(), changed)["passed"]


def test_missing_producers_are_separate_from_receipts() -> None:
    """SCENARIO-REPORT-7766-BLOCKED: a queue receipt cannot become science."""
    rows, sources, failures = account(ROOT, authority(ROOT)["tasks"])
    assert len(rows) == 14
    assert [row["experiment_id"] for row in rows] == list(range(7753, 7767))
    fit = rows[4]
    assert fit["availability"] == "pre_gate_receipt"
    assert fit["producer_path"].endswith("_v675_view_energy_fit.json")
    assert fit["producer_hash"] is None
    assert fit["pre_gate_receipt_hash"] is not None
    assert rows[5]["availability"] in {"absent", "pre_gate_receipt"}
    assert rows[9]["raw_metrics"]["independent_static_eligible"] is False
    assert rows[9]["raw_metrics"]["independent_online_eligible"] is False
    assert rows[11]["producer_hash"] is None
    assert rows[-1]["raw_paths"] == []
    assert len(sources) == 13
    assert any(item["upstream_id"] == "Exp7757" for item in failures)
    assert any(item["upstream_id"] == "Exp7764" for item in failures)


def test_terminal_block_and_cold_replay() -> None:
    """SCENARIO-REPORT-7766-REPLAY: raw rows and sources bind the verdict."""
    publication = {
        "G1": True,
        "G2": True,
        "G3": True,
        "G4": True,
        "paper_ready": True,
        "unmet_gates": [],
        "command_exit": 0,
        "result_hash": "test-only",
    }
    artifact = build_artifact(ROOT, publication, [{"passed": True}], [], 1.0)
    assert artifact["honest_verdict"] == "complete_blocked_required_v675_evidence"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["capstone_complete_score"] == 0
    assert artifact["acceptance_gate_results"]["readiness"] == 0
    assert artifact["prd_gap_findings"]["verification"]["qualified"] is False
    assert artifact["prd_gap_findings"]["retained_learning"]["qualified"] is False
    assert artifact["prd_gap_findings"]["live_path_efficiency"]["qualified"] is False
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["verifier_is_oracle"] is False
    assert cold_replay(artifact, ROOT) == []
    changed = deepcopy(artifact)
    changed["rows"][0]["availability"] = "absent"
    assert "rows" in cold_replay(changed, ROOT)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"][0]["sha256"] = "sha256:wrong"
    assert "source_artifact_hashes" in cold_replay(changed, ROOT)
    assert json.loads(json.dumps(artifact))["sample_size_budget"]["intended"] == 14


def test_private_fixture_does_not_write_repository(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7766-REPLAY: a changed source fails in private space."""
    fixture = tmp_path / "source.json"
    fixture.write_text('{"verdict_class":"null"}')
    assert fixture.is_file()


def test_changed_contract_cannot_open_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7766-CONTRACT: wrong order or design blocks a claim."""
    with pytest.raises(ValueError, match="fourteen-task order"):
        account(ROOT, [])
    source = authority(ROOT)
    source["comparison"] = {"passed": False, "errors": ["row_mismatch"]}
    monkeypatch.setattr(capstone, "authority", lambda _root: source)
    result = build_artifact(ROOT, {}, [{"passed": True}])
    assert any(item["field"] == "table_json_yaml_match" for item in result["gate_check_summary"])
    assert result["acceptance_gate_results"]["validity"] is False


def test_failed_validation_is_an_explicit_operand() -> None:
    """SCENARIO-REPORT-7766-BLOCKED: a failed command disqualifies readiness."""
    receipt = {
        "name": "full_python_suite",
        "passed": False,
        "exit_code": 2,
        "log_path": "private/full.log",
        "log_sha256": "sha256:test",
    }
    result = build_artifact(ROOT, {}, [receipt])
    assert result["verdict_class"] == "disqualified"
    assert result["acceptance_gate_results"]["readiness"] == 0
    assert any(
        item["field"] == "validation.full_python_suite.exit_code"
        for item in result["gate_check_summary"]
    )
