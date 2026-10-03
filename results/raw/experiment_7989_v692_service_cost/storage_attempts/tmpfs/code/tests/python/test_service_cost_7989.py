"""REQ-REPORT-7989 and REQ-VERIFY-7989: source-bound engineering measurements."""

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting import service_cost_7989 as s
from carnot.reporting.current_work_receipt import atomic_json


def test_independent_gates(tmp_path):
    """SCENARIO-REPORT-7989-GATES: a failed stream cannot veto fitted heads."""
    plan = s.authenticate(s.ROOT)
    assert plan["branch_readiness"] == dict(multivariate=1, durable_learning=0, scalar_current=0)
    assert len(plan["requests"]) == 32
    assert len(plan["heads"]["heads"]) == 6
    assert {r["role"] for r in plan["requests"]} == {"policy_design"}
    assert sum(r["q"] is None for r in plan["requests"]) == 3
    assert all(r["status"] == "excluded" for r in plan["requests"] if r["q"] is None)
    assert all("y" not in r for r in plan["requests"])
    assert s.authenticate(tmp_path)["requests"] == []


def test_measurement_and_drift(tmp_path):
    """SCENARIO-VERIFY-7989-ACCOUNTING: source counts survive timing repeats."""
    inputs, heads = s.fixture()
    value = s.measure(inputs, heads, tmp_path)
    assert value["sample_size_budget"]["independent"] == 4
    assert len(value["rows"]) == 4 * 2 * 2 * 10
    assert value["compatible_fraction"] == 0
    assert value["complete_service_cost"][0]["unknown_count"] == 1
    assert value["paired_cpu_comparisons"][0]["independent"] == 4
    assert value["durable_costs"]["online_gradient"] is None
    s.replay(value)
    for field in ("wall_ns", "request_sha256", "response", "repetition"):
        bad = copy.deepcopy(value)
        bad["rows"][0][field] = None
        with pytest.raises((ValueError, TypeError), match="drift|pair|span"):
            s.replay(bad)
    for field in ("service_summary", "paired_cpu_comparisons", "complete_service_cost"):
        bad = copy.deepcopy(value)
        bad[field] = []
        with pytest.raises(ValueError, match="reduction_drift"):
            s.replay(bad)


def test_request_bounds(tmp_path):
    """SCENARIO-REPORT-7989-SERVICE: missing probabilities cannot become zero."""
    inputs, heads = s.fixture()
    atomic_json(tmp_path / "heads.json", heads)
    for item in inputs:
        atomic_json(tmp_path / "request.json", item)
        a = s.request(
            tmp_path / "request.json", tmp_path / "heads.json", tmp_path / "reply", "gibbs", "fsync"
        )
        b = s.request(
            tmp_path / "request.json",
            tmp_path / "heads.json",
            tmp_path / "reply",
            "gibbs",
            "no_write",
        )
        assert a["response"] == b["response"]
        assert a["wall_ns"] == sum(a["exclusive_phase_spans"].values())
        assert json.loads((tmp_path / "reply").read_text()) == a["response"]
    assert s.reduce([])["service_summary"] == []


def test_sidecar_block_is_local(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7989-GATES: changed head bytes block only their owner."""
    real = s.checked

    def changed(item):
        if Path(item["path"]).name == "heads.json":
            raise ValueError("hash_changed")
        return real(item)

    monkeypatch.setattr(s, "checked", changed)
    value = s.authenticate(s.ROOT)
    assert value["branch_readiness"]["multivariate"] == 0
    assert any(
        r["field"] == "checkpoint_replay"
        for r in value["branch_gate_check_summary"]["multivariate"]
    )


@pytest.mark.parametrize("fault", ["parent_gate", "gpu_identity", "public_join"])
def test_authenticated_join_failures(monkeypatch, fault):
    """SCENARIO-REPORT-7989-GATES: broken public or GPU custody stays blocked."""
    authenticate = s.fitted.authenticate

    def corrupted(root):
        failures, plan = authenticate(root)
        if fault == "parent_gate":
            return [dict(field="parent_readiness", observed=0)], plan
        if fault == "gpu_identity":
            plan["upstream"][7969]["model_identity_receipt"]["authenticated"] = False
        else:
            for row in plan["upstream"][7969]["rows"]:
                if row["role"] == "policy_design":
                    row["parsed"]["probability"] = 0.314159
        return failures, plan

    monkeypatch.setattr(s.fitted, "authenticate", corrupted)
    plan = s.authenticate(s.ROOT)
    assert plan["branch_readiness"]["multivariate"] == 0
    assert plan["requests"] == []
    failed = plan["branch_gate_check_summary"]["multivariate"][-1]
    assert failed["field"] == "checkpoint_replay"
    assert failed["passed"] is False
