"""REQ-REPORT-7725: terminal V672 accounting uses immutable upstream evidence."""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import v672_contract
from carnot.reporting.v672_capstone import account, cold_reduce
from carnot.experiment_7725_v672_capstone import build_artifact, read_candidate


ROOT = Path(__file__).resolve().parents[2]


def test_current_custody_blocks_required_science() -> None:
    """SCENARIO-REPORT-7725-CUSTODY: no synthetic producer fills a gap."""
    _, roadmap, _ = v672_contract.resolve_authority(ROOT)
    result = account(ROOT, roadmap["tasks"])
    assert len(result["prior_dispositions"]) == 13
    assert [row["task_id"] for row in result["prior_dispositions"]] == [
        task["id"] for task in roadmap["tasks"]
    ]
    assert result["prior_dispositions"][-1]["availability"] == "planned_output"
    assert result["source_artifact_hashes"]["planned_output_is_input"] is False
    assert result["verdict_class"] == "blocked"
    assert result["honest_verdict"] == "complete_blocked_required_v672_evidence"
    failures = result["gate_check_summary"]["failed_checks"]
    assert {f["upstream"] for f in failures} >= {
        "exp7718-natural-decisions",
        "exp7720-continuous-acquisition",
        "exp7721-independent-evidence-audit",
    }
    assert all(
        {"check", "upstream", "path", "field", "operator", "expected", "observed"} <= f.keys()
        for f in failures
    )


def test_cold_replay_rejects_changed_disposition(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7725-CUSTODY: terminal rows cannot be relabeled."""
    _, roadmap, _ = v672_contract.resolve_authority(ROOT)
    value = account(ROOT, roadmap["tasks"])
    assert cold_reduce(value, ROOT, roadmap["tasks"]) == []
    changed = json.loads(json.dumps(value))
    changed["prior_dispositions"][0]["verdict_class"] = "positive"
    assert "prior_dispositions_mismatch" in cold_reduce(changed, ROOT, roadmap["tasks"])


def test_artifact_separates_unmeasured_claims() -> None:
    """REQ-REPORT-7725: blocked science leaves benefit gates unmeasured."""
    result = build_artifact(ROOT)
    assert result["capstone_accounting_ready_score"] == 1
    assert result["inference_substrate_class"] == "aggregation"
    assert result["MODEL_SPECS"] == result["model_specs"] == []
    assert result["model_invoked"] is False
    assert result["acceptance_gate_results"]["probability"]["measured_operands"] is None
    assert result["acceptance_gate_results"]["retention"]["measured_operands"] is None
    assert len(result["three_prd_gaps"]) == 3
    assert result["publication_gates"]["headline_auroc"] == 0.9131


def test_candidate_cold_reader_detects_changed_claim(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7725-TERMINAL: a fresh reduction detects a changed gate."""
    path = tmp_path / "candidate.json"
    value = build_artifact(ROOT)
    path.write_text(json.dumps(value))
    assert read_candidate(path, ROOT) == []
    value["acceptance_gate_results"]["readiness"]["passed"] = True
    path.write_text(json.dumps(value))
    assert "acceptance_gate_results_mismatch" in read_candidate(path, ROOT)


def test_reducer_rejects_malformed_inputs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7725-CUSTODY: invalid task order and JSON are errors."""
    _, roadmap, _ = v672_contract.resolve_authority(ROOT)
    tasks = json.loads(json.dumps(roadmap["tasks"]))
    assert cold_reduce([], ROOT, tasks) == ["artifact_object_required"]
    tasks[0]["id"] = "wrong"
    try:
        account(ROOT, tasks)
    except ValueError as exc:
        assert "thirteen-task" in str(exc)
    else:
        raise AssertionError("invalid task order accepted")
    tasks = json.loads(json.dumps(roadmap["tasks"]))
    path = tmp_path / "invalid.json"
    path.write_text("[]")
    tasks[0]["deliverable"] = str(path)
    try:
        account(ROOT, tasks)
    except ValueError as exc:
        assert "producer object" in str(exc)
    else:
        raise AssertionError("non-object producer accepted")
