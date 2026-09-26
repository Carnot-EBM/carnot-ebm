"""REQ-REPORT-7684 and REQ-CAPSTONE-7684 custody and cold-reader controls."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import yaml

from carnot.reporting import v669_capstone
from carnot import experiment_7684_v669_capstone as capstone


ROOT = Path(__file__).resolve().parents[2]


def _private(tmp_path: Path) -> tuple[Path, list[dict]]:
    """Keep a small private copy so deletion never changes real evidence."""

    tasks = yaml.safe_load((ROOT / "research-roadmap.yaml").read_text())["tasks"]
    (tmp_path / "results").mkdir()
    for task in tasks[:-1]:
        source = ROOT / task["deliverable"]
        if source.exists():
            target = tmp_path / task["deliverable"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return tmp_path, tasks


def test_req_report_7684_exact_roster_and_blocked_science(tmp_path: Path) -> None:
    root, tasks = _private(tmp_path)
    result = v669_capstone.account(root, tasks)
    assert [x["task_id"] for x in result["prior_dispositions"]] == [x["id"] for x in tasks]
    assert len(result["prior_dispositions"]) == 14
    assert result["verdict_class"] == "blocked"
    assert result["honest_verdict"].startswith("complete_blocked_")
    assert {"exp7675-fresh-decision-evaluation", "exp7678-continuous-relation-learning"} <= {
        x["upstream"] for x in result["gate_check_summary"]["failed_checks"]
    }
    assert result["prior_dispositions"][-1]["availability"] == "current_capstone"
    assert all(x["verdict_class"] != "partial" for x in result["prior_dispositions"])


def test_scenario_capstone_7684_cold_deleted_producer(tmp_path: Path) -> None:
    root, tasks = _private(tmp_path)
    (root / tasks[1]["deliverable"]).unlink()
    result = v669_capstone.account(root, tasks)
    row = result["prior_dispositions"][1]
    assert row["availability"] == "absent"
    assert row["scientific_disposition"] is None
    assert result["verdict_class"] == "blocked"
    assert any(
        x["upstream"] == tasks[1]["id"] and x["field"] == "exists"
        for x in result["gate_check_summary"]["failed_checks"]
    )


def test_scenario_report_7684_custody_missing_gate_field(tmp_path: Path) -> None:
    root, tasks = _private(tmp_path)
    source = root / tasks[1]["deliverable"]
    value = json.loads(source.read_text())
    source.unlink()
    del value["relation_protocol_ready_score"]
    source.write_text(json.dumps(value))
    result = v669_capstone.account(root, tasks)
    assert result["verdict_class"] == "blocked"
    assert any(
        x["upstream"] == tasks[1]["id"]
        and x["field"] == "relation_protocol_ready_score"
        and x["observed"] is None
        for x in result["gate_check_summary"]["failed_checks"]
    )


def test_scenario_report_7684_terminal_cold_reduction(tmp_path: Path) -> None:
    root, tasks = _private(tmp_path)
    value = v669_capstone.account(root, tasks)
    assert v669_capstone.cold_reduce(value, root, tasks) == []
    changed = deepcopy(value)
    changed["prior_dispositions"][0]["verdict_class"] = "positive"
    assert v669_capstone.cold_reduce(changed, root, tasks)
    changed = deepcopy(value)
    changed["source_artifact_hashes"]["producers"][0]["sha256"] = "0" * 64
    assert v669_capstone.cold_reduce(changed, root, tasks)


def test_req_report_7684_contract_authority() -> None:
    authority = v669_capstone.authority(ROOT)
    assert authority["comparison"]["passed"] is True
    assert len(authority["tasks"]) == 14
    assert authority["path"] == "research-roadmap.yaml"


def test_req_report_7684_raw_audit_reduction() -> None:
    audit = json.loads(
        (ROOT / "results/experiment_7679_v669_independent_evidence_audit.json").read_text()
    )
    reduced = v669_capstone.reduce_audit_rows(audit["rows"])
    assert reduced == {
        "cohort_families": 480,
        "cohort_unknown_families": 480,
        "fixture_families": 80,
        "fixture_correct_families": 72,
        "quote_families": 24,
        "quote_supported_relations": 0,
    }


def test_scenario_report_7684_candidate_cold_reader(tmp_path: Path) -> None:
    artifact = capstone.build_artifact(ROOT)
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_specs_declaration"] == "no_current_model"
    assert artifact["cold_raw_reduction"]["cohort_unknown_families"] == 480
    assert artifact["capstone_accounting_ready_score"] == 0
    assert {row["gate"] for row in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "coverage",
        "freshness",
        "probability",
        "decision_utility",
        "retention",
        "efficiency",
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact))
    assert capstone.read_candidate(path, ROOT) == []
    artifact["publication_gates"]["headline_auroc"] = 0.99
    path.write_text(json.dumps(artifact))
    assert "publication_gates_mismatch" in capstone.read_candidate(path, ROOT)
    assert "cold_reader_failed:FileNotFoundError" in capstone.read_candidate(
        tmp_path / "missing.json", ROOT
    )


def test_scenario_report_7684_failure_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, tasks = _private(tmp_path)
    with pytest.raises(ValueError, match="fourteen-task"):
        v669_capstone.account(root, tasks[:-1])
    malformed = root / "results/experiment_7674_bad.json"
    malformed.write_text("{")
    assert v669_capstone.account(root, tasks)["verdict_class"] == "blocked"
    assert v669_capstone.cold_reduce(None, root, tasks) == ["artifact_object_required"]
    assert v669_capstone.cold_reduce({}, root, tasks[:-1]) == ["source_reduction_failed:ValueError"]
    monkeypatch.setattr(
        v669_capstone.v669_contract,
        "compare_authorities",
        lambda *_: {"passed": False, "errors": ["private_mutation"]},
    )
    with pytest.raises(ValueError, match="authority mismatch"):
        v669_capstone.authority(ROOT)


def test_scenario_report_7684_failed_required_validation() -> None:
    candidate = capstone.build_artifact(
        ROOT, receipts=[{"name": "focused_pytest", "passed": False}]
    )
    assert candidate["verdict_class"] == "disqualified"
    assert candidate["capstone_accounting_ready_score"] == 0
