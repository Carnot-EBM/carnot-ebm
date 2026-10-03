"""REQ-VERIFY-7997: frozen probabilities, costs and independent uncertainty."""

import copy

import numpy as np
import pytest

from carnot.verify import typed_development_7997 as m
from carnot.reporting import typed_evaluation_7997 as v


def fixture():
    """Separate oracle targets test wiring without posing as natural evidence."""
    heads = {
        a: [
            dict(
                arm=a,
                seed=17,
                parameters=[0.0] * n,
                decay_scale=1.0,
                temperature=1.0,
                scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9),
            )
        ]
        for a, n in dict(spline=109, logistic=10, mlp=89).items()
    }
    heads["scalar"] = [dict(arm="platt", seed=17, parameters=[2.0, -1.0], temperature=1.0)]
    data, targets = {}, {}
    for role, n in [("calibration", 64), ("stream", 256)]:
        data[role] = [
            dict(
                family_id=f"{role}-{i}",
                source_cluster_id=f"{role}-{i}",
                q=0.0 if i % 2 == 0 else 1.0,
                features=[0.5] * 8,
                status="completed",
            )
            for i in range(n)
        ]
        targets[role] = [dict(family_id=r["family_id"], y=i % 2) for i, r in enumerate(data[role])]
    return dict(heads=heads, public=data, targets=targets)


def test_predictor_boundary_and_temperature():
    f = fixture()
    rows = m.predictions(f["heads"], f["public"]["calibration"])
    assert len(rows) == 64 * 5 * 3
    assert all("y" not in r for r in rows)
    policies, scores = v.calibrate(rows, f["targets"]["calibration"])
    assert policies["raw_q"]["temperature"] == 1
    assert len(scores) == 15
    selected = m.select(rows, policies)
    assert len(selected) == 64 * 5
    assert m.action(0.05) == m.action(0.75) == m.action(None) == "escalate"
    assert m.action(0.049) == "accept" and m.action(0.751) == "reject"
    assert m.transform(0.0, 0.5) == 0 and m.transform(1.0, 2) == 1
    bad = copy.deepcopy(f["public"]["stream"])
    bad[0]["y"] = 1
    with pytest.raises(ValueError, match="predictor_fields"):
        m.predictions(f["heads"], bad)
    for key, value in [("q", float("nan")), ("features", [0.0] * 7)]:
        bad = copy.deepcopy(f["public"]["stream"])
        bad[0][key] = value
        with pytest.raises(ValueError):
            m.predictions(f["heads"], bad)
    failed = copy.deepcopy(f["public"]["stream"][:1])
    failed[0].update(q=None, status="censored")
    assert all(r["probability"] is None for r in m.predictions(f["heads"], failed))
    duplicate = f["public"]["stream"][:1] * 2
    with pytest.raises(ValueError, match="duplicate"):
        m.predictions(f["heads"], duplicate)


def test_reduction_controls_failures_and_bounds():
    controls = v.controls()
    assert controls["passed"]
    assert controls["known_benefit"]["benefit"]
    assert not controls["no_headroom"]["benefit"]
    assert controls["known_benefit"]["verdict_class"] == "circular_positive"
    f = fixture()
    candidate = m.predictions(f["heads"], f["public"]["stream"])
    policies = {a: dict(temperature=1.0) for a in m.ARMS}
    rows = m.select(candidate, policies)
    result = v.evaluate(rows, f["targets"]["stream"])
    assert result["evaluation_support"]["passed"]
    assert not result["benefit"]
    assert result["sample_size_budget"]["independent"] == 256
    assert result["summary"]["spline"]["automation"] == 1
    assert result["equivalent_classifier_identity"]["passed"]
    targets = copy.deepcopy(f["targets"]["stream"])
    targets[0]["y"] = None
    for r in rows:
        if r["family_id"] == "stream-1":
            r.update(probability=None, failure_status=True, censor_status=True)
    result = v.evaluate(rows, targets)
    assert result["sample_size_budget"]["failed"] == 1
    assert result["sample_size_budget"]["censored"] == 1
    assert result["all_intended_bounds"]["raw_q"]["unknown_labels"] == 1
    assert (
        result["all_intended_bounds"]["raw_q"]["upper"]
        >= result["all_intended_bounds"]["raw_q"]["lower"]
    )
    assert all(r["actual_cost"] == 0.25 for r in result["rows"] if r["family_id"] == "stream-1")
    small = v.evaluate(rows[:5], targets[:1])
    assert not small["evaluation_support"]["passed"]
    assert small["summary"]["raw_q"]["auroc"] is None
    assert v.uncertainty(np.array([]))["interval"] == [None, None]
    assert v.uncertainty(np.zeros(24))["raw_p"] == 1
    with pytest.raises(ValueError, match="target_contract"):
        v.evaluate(rows, [dict(family_id="stream-0")])
    with pytest.raises(ValueError, match="target_contract"):
        v.evaluate(rows, [dict(family_id="stream-0", y=2)])
    with pytest.raises(ValueError, match="target_roster"):
        v.evaluate(rows, [])
    with pytest.raises(ValueError, match="calibration_support"):
        v.calibrate(candidate[:1], [dict(family_id="stream-0", y=None)])
