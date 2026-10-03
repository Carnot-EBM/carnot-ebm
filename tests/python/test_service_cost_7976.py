"""REQ-REPORT-7976: branch gates, equivalent service and cold cost accounting."""

import copy
import json
from pathlib import Path

import pytest

from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting import service_cost_7976 as s


def test_independent_gates(tmp_path):
    """SCENARIO-REPORT-7976-1: one missing branch cannot block the other."""
    assert not s.branch_gate(tmp_path / "absent.json", 7970, "ready", None)[0]
    p = tmp_path / "branch.json"
    atomic_json(
        p,
        dict(
            experiment_id=7972,
            qwen_calibration_ready_score=1,
            verdict_class="null",
            flagged_adversarial=False,
        ),
    )
    assert s.branch_gate(p, 7972, "qwen_calibration_ready_score", s.sha256_file(p))[0]
    assert not s.branch_gate(p, 7972, "qwen_calibration_ready_score", "drift")[0]
    assert not s.branch_gate(p, 7970, "ready", None, retired=True)[0]


@pytest.mark.parametrize(
    "f,t,ideal,modeled",
    [
        (0, 0, 1, 1),
        (1, 0, None, 100),
        (0.5, 0.1, 2, 1 / 0.605),
        (None, 0, None, None),
        (0, None, None, None),
    ],
)
def test_bounds(f, t, ideal, modeled):
    """SCENARIO-REPORT-7976-2: estimates retain missing operands."""
    assert s.bounds(f, t) == {"ideal_amdahl_bound": ideal, "modeled_100x_bound": modeled}
    with pytest.raises(ValueError, match="fraction"):
        s.bounds(2, 0)


def test_service_and_cold_reduction(tmp_path):
    """SCENARIO-REPORT-7976-2: response agreement does not imply science benefit."""
    inputs, heads = s.fixture()
    value = s.measure(inputs, heads, tmp_path)
    assert len(value["service_rows"]) == 80
    assert value["sample_size_budget"]["independent"] == 4
    assert s.replay(value) == value["rows"]
    broken = copy.deepcopy(value)
    broken["service_rows"][0]["wall_ns"] *= 2
    with pytest.raises(ValueError, match="span"):
        s.replay(broken)
    broken = copy.deepcopy(value)
    broken["rows"][0]["p50_s"] *= 2
    with pytest.raises(ValueError, match="reduction"):
        s.replay(broken)
    broken = copy.deepcopy(value)
    broken["service_rows"][0]["response"]["action"] = "tampered"
    with pytest.raises(ValueError, match="decision"):
        s.replay(broken)
    inputs[0]["original_model_s"] = None
    other = s.measure(inputs, heads, tmp_path / "unknown")
    assert other["complete_service_cost"]["known_count"] == 3
    assert s.reduce_rows([]) == []
    broken = copy.deepcopy(value)
    broken["complete_service_cost"]["p50_s"] = 123
    with pytest.raises(ValueError, match="cost"):
        s.replay(broken)
    broken = copy.deepcopy(value)
    broken["service_rows"].pop()
    with pytest.raises(ValueError, match="paired"):
        s.replay(broken)
    checkpoint = tmp_path / "checkpoint.json"
    atomic_json(checkpoint, dict(requests=s.fixture()[0], heads=heads))
    value["input_checkpoint"] = s.prior.reference(checkpoint)
    assert s.replay(value) == value["rows"]
    bad = copy.deepcopy(value)
    for row in bad["service_rows"][:2]:
        row["original_model_s"] = 123
    with pytest.raises(ValueError, match="input"):
        s.replay(bad)


def test_authenticate_current_and_external_absence(tmp_path):
    """SCENARIO-REPORT-7976-1: missing inputs are terminal external blocks."""
    plan = s.authenticate(tmp_path)
    assert plan["branch_readiness"] == {
        "source_energy": 0,
        "qwen_calibration": 0,
        "durable_learning": 0,
    }
    assert plan["requests"] == []
    current = s.authenticate(s.ROOT)
    assert current["branch_readiness"]["qwen_calibration"] == 1
    assert len(current["requests"]) == 64
    assert len(current["original_model_cost_rows"]) >= 64


def test_external_checkpoint_drift(monkeypatch):
    """SCENARIO-REPORT-7976-1: external byte drift blocks that branch explicitly."""

    def drift(value):
        raise ValueError("checkpoint_hash_drift")

    monkeypatch.setattr(s.prior, "replay", drift)
    plan = s.authenticate(s.ROOT)
    assert plan["branch_readiness"]["qwen_calibration"] == 0
    assert (
        plan["branch_gate_check_summary"]["qwen_calibration"][-1]["observed"]
        == "checkpoint_hash_drift"
    )
