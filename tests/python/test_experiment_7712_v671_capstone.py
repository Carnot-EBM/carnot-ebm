"""REQ-REPORT-7712 and REQ-CAPSTONE-7712 custody and claim boundaries."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

from carnot import experiment_7712_v671_capstone as capstone
from carnot.reporting import v671_capstone

ROOT = Path(__file__).resolve().parents[2]


def test_req_report_7712_exact_authority() -> None:
    selected = v671_capstone.authority(ROOT)
    assert selected["comparison"]["passed"]
    assert len(selected["tasks"]) == 14
    assert selected["tasks"][-1]["id"] == "exp7712-capstone"
    assert selected["tasks"][-1]["MODEL_SPECS"] == []


def test_req_capstone_7712_blocked_science_accounted() -> None:
    selected = v671_capstone.authority(ROOT)
    result = v671_capstone.account(ROOT, selected["tasks"])
    rows = result["prior_dispositions"]
    assert [row["task_id"] for row in rows] == [task["id"] for task in selected["tasks"]]
    assert rows[-1]["availability"] == "planned_output"
    assert result["verdict_class"] == "blocked"
    assert result["honest_verdict"].startswith("complete_blocked_")
    assert any(
        item["upstream"] == "exp7706-continuous-acquisition"
        and item["field"] == "exists"
        and item["observed"] is False
        for item in result["gate_check_summary"]["failed_checks"]
    )
    assert not any(
        item["path"] == selected["tasks"][-1]["deliverable"]
        for item in result["source_artifact_hashes"]["producers"]
    )


def test_scenario_report_7712_custody_private_missing_producer(tmp_path: Path) -> None:
    selected = v671_capstone.authority(ROOT)
    tasks = selected["tasks"]
    for task in tasks[:-1]:
        source = ROOT / task["deliverable"]
        if source.is_file() and task["id"] != "exp7704-heldout-decisions":
            target = tmp_path / task["deliverable"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    result = v671_capstone.account(tmp_path, tasks)
    assert len(result["prior_dispositions"]) == 14
    assert result["verdict_class"] == "blocked"
    assert result["prior_dispositions"][5]["scientific_disposition"] is None
    assert any(
        item["upstream"] == "exp7704-heldout-decisions"
        and item["path"] == tasks[5]["deliverable"]
        and item["field"] == "exists"
        and item["observed"] is False
        for item in result["gate_check_summary"]["failed_checks"]
    )


def test_scenario_report_7712_terminal_cold_custody() -> None:
    tasks = v671_capstone.authority(ROOT)["tasks"]
    value = v671_capstone.account(ROOT, tasks)
    assert v671_capstone.cold_reduce(value, ROOT, tasks) == []
    altered = deepcopy(value)
    altered["prior_dispositions"][0]["verdict_class"] = "positive"
    assert v671_capstone.cold_reduce(altered, ROOT, tasks)
    altered = deepcopy(value)
    altered["source_artifact_hashes"]["producers"][0]["sha256"] = "sha256:wrong"
    assert v671_capstone.cold_reduce(altered, ROOT, tasks)


def test_req_report_7712_artifact_boundaries(tmp_path: Path) -> None:
    artifact = capstone.build_artifact(ROOT)
    assert artifact["verdict_class"] == "blocked"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["capstone_accounting_ready_score"] == 1
    assert len(artifact["rows"]) == 14
    assert len(artifact["three_prd_gaps"]) == 3
    assert artifact["publication_gates"]["headline_auroc"] == 0.9131
    assert {gate["gate"] for gate in artifact["acceptance_gate_results"]} == {
        "validity",
        "readiness",
        "coverage",
        "freshness",
        "probability",
        "utility",
        "retention",
        "efficiency",
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact))
    assert capstone.read_candidate(path, ROOT) == []
    artifact["rows"][0]["raw_metrics"]["producer_present"] = False
    path.write_text(json.dumps(artifact))
    assert capstone.read_candidate(path, ROOT)


def test_scenario_report_7712_terminal_invalid_validation(tmp_path: Path) -> None:
    artifact = capstone.build_artifact(ROOT, receipts=[{"name": "focused_pytest", "passed": False}])
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"].startswith("complete_disqualified_")
    assert artifact["capstone_accounting_ready_score"] == 0
    assert artifact["rows"][-1]["raw_metrics"]["scientific_disposition"] == "disqualified"
    assert capstone.read_candidate(tmp_path / "missing.json", ROOT) == [
        "cold_reader_failed:FileNotFoundError"
    ]


def test_scenario_report_7712_terminal_bad_authority(monkeypatch) -> None:
    monkeypatch.setattr(
        v671_capstone.v671_contract,
        "compare_authorities",
        lambda _design, _roadmap: {"passed": False, "errors": ["row_mismatch"]},
    )
    try:
        v671_capstone.authority(ROOT)
    except ValueError as error:
        assert "row_mismatch" in str(error)
    else:
        raise AssertionError("changed authority accepted")


def test_scenario_report_7712_custody_missing_and_gate_receipt(tmp_path: Path) -> None:
    tasks = v671_capstone.authority(ROOT)["tasks"]
    assert v671_capstone.cold_reduce(None, tmp_path, tasks) == ["artifact_object_required"]
    assert v671_capstone.cold_reduce({}, tmp_path, []) == ["source_reduction_failed:ValueError"]
    try:
        v671_capstone.account(tmp_path, tasks[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("short roster accepted")
    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_7706_bad.json").write_text("not json")
    receipt = results / "experiment_7706_gate.json"
    receipt.write_text(json.dumps({"schema": "blocked_gate_check_v1"}))
    log = tmp_path / "ops/conductor-log.md"
    log.parent.mkdir()
    log.write_text(f"{tasks[5]['title'][:44]} | GATE_BLOCK | pre-emptive skip\n")
    result = v671_capstone.account(tmp_path, tasks)
    assert result["prior_dispositions"][7]["availability"] == "pre_gate_receipt"
    assert result["prior_dispositions"][5]["availability"] == "gate_skipped"
    assert result["source_artifact_hashes"]["pre_gate_receipts"][0]["path"].endswith(
        "experiment_7706_gate.json"
    )


def test_req_capstone_7712_valid_null_possible(tmp_path: Path) -> None:
    tasks = v671_capstone.authority(ROOT)["tasks"]
    for task in tasks[:-1]:
        source = ROOT / task["deliverable"]
        if source.is_file():
            target = tmp_path / task["deliverable"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    overrides = {
        7705: {"constraint_bank_ready_score": 1},
        7707: {"verdict_class": "null", "honest_verdict": "complete_null_no_benefit"},
        7710: {"native_record_ready_score": 1, "verdict_class": "null"},
    }
    for number, updates in overrides.items():
        path = tmp_path / next(
            t["deliverable"] for t in tasks if t["id"].startswith(f"exp{number}-")
        )
        value = json.loads(path.read_text())
        value.update(updates)
        path.write_text(json.dumps(value))
    source = ROOT / tasks[5]["deliverable"]
    value = json.loads(source.read_text())
    value.update(
        {
            "verdict_class": "null",
            "honest_verdict": "complete_null_no_benefit",
            "flagged_adversarial": False,
            "continuous_acquisition_complete_score": 1,
        }
    )
    target = tmp_path / tasks[7]["deliverable"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value))
    result = v671_capstone.account(tmp_path, tasks)
    assert result["verdict_class"] == "null"
    assert result["honest_verdict"].startswith("complete_null_")
    assert not result["gate_check_summary"]["failed_checks"]
