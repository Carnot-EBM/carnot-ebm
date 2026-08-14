"""REQ-OPS-RECURRING-GATE-6425 tests for recurring gate-block diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_6425_recurring_gate_block_root_cause as exp


REPO = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_req_ops_recurring_gate_6425_collects_one_row_per_structured_blocker(tmp_path):
    """REQ-OPS-RECURRING-GATE-6425 binds each blocker to a replayable gate row."""
    _write_json(tmp_path / "results" / "experiment_101_upstream.json", {"ready_score": 0.0})
    _write_json(
        tmp_path / "results" / "experiment_102_downstream.json",
        {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "1 of 1 gate(s) failed",
            "gates_evaluated": [
                {
                    "upstream": "exp101-upstream",
                    "artifact_field": "ready_score",
                    "op": "==",
                    "expected": 1.0,
                    "actual": 0.0,
                    "passed": False,
                    "reason": "actual=0.0 == expected=1.0",
                }
            ],
        },
    )
    (tmp_path / "research-complete.yaml").write_text(
        """
milestones:
  - id: 2026.08.536
    tasks:
      - id: exp102-downstream
        title: Downstream
        deliverable: results/experiment_102_downstream.json
""",
        encoding="utf-8",
    )

    rows = exp.collect_blocker_population(
        tmp_path,
        start_milestone="2026.08.536",
        end_milestone="2026.08.549",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["milestone"] == "2026.08.536"
    assert row["task_id"] == "exp102-downstream"
    assert row["upstream_id"] == "exp101-upstream"
    assert row["gate_field"] == "ready_score"
    assert row["operator"] == "=="
    assert row["expected_type"] == "float"
    assert row["observed_type"] == "float"
    assert row["classification"] == "correct_expected_refusal"
    assert row["terminal_artifact"].endswith("experiment_102_downstream.json")
    assert row["upstream_artifact"].endswith("experiment_101_upstream.json")
    assert row["replayed_gate_passed"] is False


def test_req_ops_recurring_gate_6425_classifies_required_root_cause_shapes(tmp_path):
    """REQ-OPS-RECURRING-GATE-6425 separates contract faults from valid refusals."""
    upstream_path = tmp_path / "results" / "experiment_201_upstream.json"
    _write_json(upstream_path, {"ready_score": 0.0, "status": "complete", "honest_verdict": "complete"})

    assert (
        exp.classify_gate_binding(
            upstream="exp201-upstream",
            artifact_path=upstream_path,
            artifact_payload={"ready_score": 0.0, "status": "complete", "honest_verdict": "complete"},
            field="ready_score",
            op="==",
            expected=1.0,
            observed=0.0,
            passed=False,
            reason="actual=0.0 == expected=1.0",
            retired_upstreams=set(),
        )["classification"]
        == "correct_expected_refusal"
    )
    assert (
        exp.classify_gate_binding(
            upstream="exp202-missing",
            artifact_path=None,
            artifact_payload=None,
            field="ready_score",
            op="==",
            expected=1.0,
            observed=None,
            passed=False,
            reason="upstream artifact not found",
            retired_upstreams={"exp202-missing"},
        )["classification"]
        == "retired_dependency"
    )
    assert (
        exp.classify_gate_binding(
            upstream="exp203-stale",
            artifact_path=upstream_path,
            artifact_payload={"ready_score": None, "status": "preconditions_recorded"},
            field="ready_score",
            op="==",
            expected=1.0,
            observed=None,
            passed=False,
            reason="actual=None == expected=1.0",
            retired_upstreams=set(),
        )["classification"]
        == "stale_artifact"
    )
    assert (
        exp.classify_gate_binding(
            upstream="exp204-drift",
            artifact_path=upstream_path,
            artifact_payload={"delta": {"pooled": 0.5}, "status": "complete", "honest_verdict": "complete"},
            field="delta",
            op=">",
            expected=0.0,
            observed={"pooled": 0.5},
            passed=False,
            reason="numeric comparison rejected",
            retired_upstreams=set(),
        )["classification"]
        == "wrong_field_type"
    )


def test_scenario_ops_recurring_gate_6425_mutation_matrix_fails_closed():
    """SCENARIO-OPS-RECURRING-GATE-6425-MUTATIONS-FAIL-CLOSED."""
    matrix = exp.build_mutation_attack_matrix()

    expected = {
        "missing_field",
        "string_numeric_gate",
        "nan_numeric_gate",
        "stale_hash",
        "retired_upstream_id",
        "contradictory_status_fields",
    }
    assert set(matrix) == expected
    for attack in matrix.values():
        assert attack["killed"] is True
        assert attack["gate_bypassed"] is False
        assert attack["diagnostic"]


def test_req_ops_recurring_gate_6425_builds_valid_real_report() -> None:
    """REQ-OPS-RECURRING-GATE-6425 reports the frozen 31-row population."""
    before = exp.protected_hashes(REPO)
    report = exp.build_report(REPO, date="20260814", before_hashes=before, duration_s=0.0)

    assert exp.validate_report(report) == []
    assert report["frozen_blocker_population_receipt"]["population_count"] == 31
    assert report["root_cause_class_counts"] == {
        "correct_expected_refusal": 16,
        "retired_dependency": 10,
        "stale_artifact": 4,
        "wrong_field_type": 1,
    }
    assert report["highest_count_root_cause"] == "correct_expected_refusal"
    assert report["diagnostic_loss_count"] == 31
    assert report["recurring_gate_diagnostic_ready_score"] == 1.0
    assert report["protected_files_unchanged"]["ok"] is True
    assert set(report["field_provenance"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert report["reproducibility_checksum"] == exp.payload_checksum(report)


def test_req_ops_recurring_gate_6425_run_and_write_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-OPS-RECURRING-GATE-6425 exposes a CLI and atomic writer."""
    report = exp.run(date="20260814", root=REPO, write=False)
    assert report["status"] == "complete_recurring_gate_block_root_cause_reported"

    path = exp.write_report({"status": "demo"}, root=tmp_path)
    assert path.name == exp.RESULT_RELATIVE_PATH.name
    assert json.loads(path.read_text()) == {"status": "demo"}

    monkeypatch.setattr(
        exp,
        "run",
        lambda *, date: {
            "status": "complete_recurring_gate_block_root_cause_reported",
        },
    )
    assert exp.main(["--date", "20260814"]) == 0
    assert "complete_recurring_gate_block_root_cause_reported" in capsys.readouterr().out


def test_req_ops_recurring_gate_6425_helper_edges(tmp_path: Path) -> None:
    """REQ-OPS-RECURRING-GATE-6425 keeps malformed inputs fail-closed."""
    assert exp._read_json(tmp_path / "missing.json") is None
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert exp._read_json(bad_json) is None
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert exp._read_json(list_json) is None

    assert exp._read_yaml(tmp_path / "missing.yaml") == {}
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("[]", encoding="utf-8")
    assert exp._read_yaml(list_yaml) == {}
    assert exp._verdict({"honest_verdict": {"value": "blocked_gate_check_failed"}}) == "blocked_gate_check_failed"

    for value, name in (
        (None, "null"),
        (True, "bool"),
        (1, "int"),
        (1.0, "float"),
        ("x", "str"),
        ([], "list"),
        ({}, "dict"),
        (object(), "object"),
    ):
        assert exp._type_name(value) == name
    assert exp._artifact_hash(None) is None

    complete = tmp_path / "research-complete.yaml"
    complete.write_text(
        """
milestones:
  - bad
  - id: 2026.08.536
    tasks:
      - bad
      - id: exp0-no-title
      - id: exp0-no-deliverable
        title: No deliverable
      - id: exp1-demo
        title: Demo retired task
        deliverable: results/experiment_1_demo.json
""",
        encoding="utf-8",
    )
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "conductor-log.md").write_text(
        "| Demo retired task | GATE_BLOCK | Pre-emptive skip: upstream retired (exp0) |\n",
        encoding="utf-8",
    )
    tasks = exp._tasks_by_id(tmp_path)
    assert "exp1-demo" in tasks
    assert exp._retired_dependency_ids_from_log(tmp_path, tasks) == {"exp1-demo"}
    assert exp._mem_available_kb(tmp_path / "missing-meminfo") is None
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 1 kB\n", encoding="utf-8")
    assert exp._mem_available_kb(meminfo) is None

    artifact_path = tmp_path / "results" / "experiment_1_demo.json"
    _write_json(
        artifact_path,
        {
            "status": "blocked",
            "honest_verdict": "blocked_gate_check_failed",
            "gates_evaluated": [],
        },
    )
    rows = exp.collect_blocker_population(tmp_path)
    assert rows[0]["classification"] == "diagnostic_loss"

    assert (
        exp.classify_gate_binding(
            upstream="exp-missing",
            artifact_path=None,
            artifact_payload=None,
            field="ready",
            op="==",
            expected=1,
            observed=None,
            passed=False,
            reason="missing",
            retired_upstreams=set(),
        )["classification"]
        == "missing_upstream"
    )
    assert (
        exp.classify_gate_binding(
            upstream="exp-unloadable",
            artifact_path=tmp_path / "x.json",
            artifact_payload=None,
            field="ready",
            op="==",
            expected=1,
            observed=None,
            passed=False,
            reason="unloadable",
            retired_upstreams=set(),
        )["classification"]
        == "missing_upstream"
    )
    assert (
        exp.classify_gate_binding(
            upstream="exp-null",
            artifact_path=tmp_path / "x.json",
            artifact_payload={"ready": None, "status": "complete", "honest_verdict": "complete"},
            field="ready",
            op="==",
            expected=1,
            observed=None,
            passed=False,
            reason="null",
            retired_upstreams=set(),
        )["classification"]
        == "stale_artifact"
    )
    assert (
        exp.classify_gate_binding(
            upstream="exp-passed",
            artifact_path=tmp_path / "x.json",
            artifact_payload={"ready": 1, "status": "complete", "honest_verdict": "complete"},
            field="ready",
            op="==",
            expected=1,
            observed=1,
            passed=True,
            reason="passed",
            retired_upstreams=set(),
        )["classification"]
        == "other_with_evidence"
    )

    assert exp._root_cause_decision({}).startswith("complete_null:")
    assert exp._ready_score([], {}) == 0.0
    assert exp._ready_score([{"classification": "bad"}], {}) == 0.0
    assert exp._ready_score(
        [{"classification": "correct_expected_refusal"}],
        {"attack": {"killed": False, "gate_bypassed": False}},
    ) == 0.0

    bad = {
        "per_unit_rows": [],
        "verifier_is_oracle": True,
        "no_scientific_gate_bypassed": False,
        "no_historical_task_rerun": False,
        "honest_verdict": "bad",
        "reproducibility_checksum": "bad",
    }
    errors = exp.validate_report(bad)
    assert any("missing required fields" in e for e in errors)
    assert any("31 frozen" in e for e in errors)
    assert any("verifier_is_oracle" in e for e in errors)


def test_req_ops_recurring_gate_6425_run_error_and_write_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-OPS-RECURRING-GATE-6425 keeps run failures explicit."""
    monkeypatch.setattr(exp, "build_report", lambda *args, **kwargs: {"status": "demo"})
    monkeypatch.setattr(exp, "validate_report", lambda report: ["boom"])
    with pytest.raises(ValueError, match="boom"):
        exp.run(date="20260814", root=REPO, write=False)

    wrote: list[dict] = []
    monkeypatch.setattr(exp, "validate_report", lambda report: [])
    monkeypatch.setattr(exp, "write_report", lambda report, root: wrote.append(dict(report)) or root)
    assert exp.run(date="20260814", root=REPO, write=True) == {"status": "demo"}
    assert wrote == [{"status": "demo"}]
