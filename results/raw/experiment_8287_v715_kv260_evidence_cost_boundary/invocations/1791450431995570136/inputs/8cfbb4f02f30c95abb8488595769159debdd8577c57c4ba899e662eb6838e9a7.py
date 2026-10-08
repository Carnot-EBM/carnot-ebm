"""REQ-REPORT-8287 / REQ-VERIFY-8287: private evidence and real CLI coverage."""

import json
from pathlib import Path

import pytest

from carnot.reporting import kv260_evidence_cost_boundary_8287 as h
from carnot.reporting import kv260_evidence_cost_execution_8287 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.primary_publication import publish_primary
import test_kv260_evidence_cost_boundary_8273 as qualified

PRIOR_DATA = qualified.data


def data():
    """Keep inherited clocks private and bind each source to a fixture invocation."""
    value = PRIOR_DATA()
    value["request_bindings"] = {"q": {"task_id": "exp8279-fixture", "invocation": "private"}}
    return value


def test_scopes_and_numerical_boundary():
    """SCENARIO-VERIFY-8287-BOUNDARY: missing clocks differ from measured zero."""
    value = data()
    value["requests"] = []
    value["cold_costs"] = []
    result = h.reduce(value, h.precision(value))
    assert result["ideal_whole_request_bound"]["status"] == "unavailable"
    assert result["ideal_whole_request_bound"]["compatible_fraction"] is None
    assert result["historical_ideal_whole_request_bound"]["maximum_gain"] == 1
    assert result["kv260_boundary_ready_score"] == 1
    assert result["verdict_class"] == "blocked"
    assert all(r["source_cost_scope"] == "historical_v712" for r in result["phase_cost_rows"])
    value = data()
    value["current"] = dict.fromkeys(["boundary", *h.CURRENT], True)
    value["requests"][0].update(workload="fit_concurrent", arm="fit")
    value["requests"].append(dict(value["requests"][0], request_id="q2"))
    value["cold_costs"][0]["workload"] = "fit_concurrent"
    result = h.reduce(value, h.precision(value))
    current = [r for r in result["phase_cost_rows"] if r["source_cost_scope"] == "current_v715"]
    assert current[0]["request_work_sum_s"] == 2
    assert current[0]["observed_parallel_makespan_s"] == pytest.approx(3.1)
    assert current[0]["cold_start_count"] == 1
    assert result["ideal_whole_request_bound"]["maximum_gain"] == 1
    assert result["nfr01_met"] is False
    assert result["generalized_learning_benefit_score"] == 0
    assert all(r["final_action_changed"] == 0 for r in result["fixed_point_rows"])
    assert any(r["overflow"] for r in result["fixed_point_rows"])
    assert all(
        r["evidence_scope"] == "synthetic_numerical_mechanics" for r in result["fixed_point_rows"]
    )
    assert (
        result["rows"][len(result["fixed_point_rows"])]["invocation_binding"]["invocation"]
        == "private"
    )
    value["resources_available"] = False
    assert h.reduce(value, [])["kv260_boundary_ready_score"] == 0


def test_custody_and_missing_operands(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8287-CUSTODY: inherited custody tests retain every assertion."""
    monkeypatch.setattr(qualified, "h", h)
    monkeypatch.setattr(qualified, "data", data)
    qualified.test_custody_and_missing_operands(tmp_path, monkeypatch)


def test_real_cli_and_tamper(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8287-REPLAY: execute real children and rehashed negative controls."""
    monkeypatch.setattr(qualified, "h", h)
    monkeypatch.setattr(qualified, "e", e)
    monkeypatch.setattr(qualified, "data", data)
    qualified.test_real_cli_and_tamper(tmp_path)


def test_failure_paths(tmp_path, monkeypatch):
    """REQ-REPORT-8287: owned failures disqualify; external missing resources block."""
    monkeypatch.setattr(qualified, "h", h)
    monkeypatch.setattr(qualified, "e", e)
    monkeypatch.setattr(qualified, "data", data)
    qualified.test_owned_and_terminal_failure_paths(tmp_path, monkeypatch)


def test_frozen_commands(tmp_path, monkeypatch):
    """REQ-REPORT-8287: freeze real coverage, E2E, consumer and terminal commands."""
    monkeypatch.setattr(qualified, "e", e)
    qualified.test_frozen_validation_commands(tmp_path)
    assert h.MODEL_SPECS == []
    assert h.CONFIG["seed"] == 7158287
    assert all("8287" in p for p in e.OWNED)


def test_exact_terminal_operands(tmp_path):
    """SCENARIO-REPORT-8287-CUSTODY: report exact missing sidecar and invalid hash fields."""
    path = tmp_path / "results/experiment_8279_v715_fit_view_capture.json"
    value = dict(
        experiment_id=8279,
        task_id="exp8279-fixture",
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        raw_shard_hashes=[],
    )
    publish_primary(path, value, lambda p: dict(passed=True))
    side = (
        path.parent / "raw" / path.stem / "validators" / (reference(path)["sha256"][7:] + ".json")
    )
    original = json.loads(side.read_bytes())
    for changes, field in [
        (dict(primary_sha256="wrong"), "primary_sha256"),
        (dict(primary_path="wrong"), "primary_path"),
        (dict(report=dict(passed=False)), "report.passed"),
    ]:
        atomic_json(side, dict(original, **changes))
        inputs = dict(checks=[], references=[])
        with pytest.raises(ValueError, match=field):
            h.authenticated(path, tmp_path / "raw", inputs)
        check = inputs["checks"][-1]
        assert check["artifact_field"] == field and check["path"] == str(side)
        assert check["hash"] == reference(side)["sha256"] and check["passed"] is False
    side.unlink()
    inputs = dict(checks=[], references=[])
    with pytest.raises(ValueError, match="exists"):
        h.authenticated(path, tmp_path / "raw", inputs)
    assert inputs["checks"][-1]["observed"] is False
    assert inputs["checks"][-1]["hash"] is None
