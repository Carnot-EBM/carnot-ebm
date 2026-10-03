"""REQ-VERIFY-7972 and REQ-REPORT-7972: scalar calibration custody and science."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import patch

from carnot.verify import qwen_energy_calibration_7972 as c


def fixture():
    """Separate sources let the fixture test the reducer without human claims."""
    return {
        role: [
            dict(
                family_id=f"{role}-{i}",
                source_cluster_id=f"{role}-{i}",
                q=0.2 if i % 2 == 0 else 0.8,
                y=i % 2,
                status="completed",
            )
            for i in range(n)
        ]
        for role, n in dict(fit=128, tune=32, policy_design=32, evaluation=64).items()
    }


def test_decision_ties_and_invalid():
    assert [c.decision(q) for q in (None, 0.05, 0.75, 0.01, 0.99)] == [
        "escalate",
        "escalate",
        "escalate",
        "accept",
        "reject",
    ]


def test_exact_basis_and_energy_gradient():
    q = np.array([0.0, 0.1, 0.2, 0.45, 0.9, 1.0])
    basis = c.basis(q)
    assert basis.shape == (6, 8)
    np.testing.assert_allclose(basis.sum(axis=1), 1)
    assert np.max(np.count_nonzero(basis, axis=1)) <= 4
    for arm, count in [("platt", 2), ("gibbs", 33), ("spline", 8)]:
        theta = np.random.default_rng(42).normal(0, 0.2, count)
        z, jac = c.logits_jacobian(arm, theta, q)
        for i in range(count):
            delta = np.zeros(count)
            delta[i] = 1e-6
            expected = (
                c.logits_jacobian(arm, theta + delta, q)[0]
                - c.logits_jacobian(arm, theta - delta, q)[0]
            ) / 2e-6
            np.testing.assert_allclose(jac[:, i], expected, atol=1e-7)
        assert np.isfinite(z).all()
    with pytest.raises(ValueError, match="arm"):
        c.logits_jacobian("unknown", np.zeros(1), q)


def test_fitting_work_and_cold_parameters():
    data = fixture()
    heads = c.fit(data["fit"], data["tune"])
    assert set(heads) == {"raw_qwen", "platt", "isotonic", "gibbs", "spline"}
    assert [len(heads[a]) for a in ("platt", "gibbs", "spline")] == [3, 3, 3]
    for arm in ("platt", "gibbs", "spline"):
        for h in heads[arm]:
            assert h["changed_coefficients"] > 0
            assert h["optimizer_steps"] == 200
            assert h["coefficient_touches"] == 200 * h["parameter_count"]
            assert h["temperature"] in c.TEMPERATURES if arm != "platt" else h["temperature"] == 1
            cold = json.loads(json.dumps(h))
            np.testing.assert_array_equal(
                c.predict(h, np.array([0.1, 0.9])), c.predict(cold, np.array([0.1, 0.9]))
            )


def test_support_and_feature_rejection():
    data = fixture()
    assert c.support(data["fit"], 128, 16)["passed"]
    assert not c.support(data["fit"][:20], 128, 16)["passed"]
    bad = copy.deepcopy(data)
    bad["fit"][0]["text"] = "leak"
    with pytest.raises(ValueError, match="scalar_fields"):
        c.validate_data(bad)
    bad = copy.deepcopy(data)
    bad["tune"][0]["source_cluster_id"] = "fit-0"
    with pytest.raises(ValueError, match="cross_role"):
        c.validate_data(bad)
    bad = copy.deepcopy(data)
    bad["fit"][0]["q"] = float("nan")
    with pytest.raises(ValueError, match="probability"):
        c.validate_data(bad)
    bad = copy.deepcopy(data)
    bad["fit"][0]["y"] = 3
    with pytest.raises(ValueError, match="label"):
        c.validate_data(bad)


def test_reducer_denominators_and_registered_primary():
    data = fixture()
    heads = c.fit(data["fit"], data["tune"])
    policies = c.design(heads, data["policy_design"])
    data["evaluation"][0]["q"] = None
    result = c.evaluate(heads, policies, data["evaluation"])
    assert result["qwen_calibration_ready_score"] == 1
    assert len(result["adjusted_p_values"]) == 6
    assert set(result["confidence_intervals"]) == {
        a + "_" + m for a in ("raw_qwen", "platt", "isotonic") for m in ("cost", "brier")
    }
    for row in result["rows"]:
        assert row["intended_denominator"] == 64
        assert row["probability_denominator"] == 63
    assert all(
        r["decision"] == "escalate"
        for r in result["decision_rows"]
        if r["family_id"] == "evaluation-0"
    )
    assert result["qwen_calibration_benefit_score"] == 0
    assert all(r["target_automation"] == 0.5 for r in result["risk_coverage_rows"])
    assert c.positive_control()["detected"]
    tiny = c.evaluate(heads, policies, data["evaluation"][:8])
    assert tiny["qwen_calibration_ready_score"] == 0


def test_holm_and_positive_benefit_gate():
    assert c.holm(dict(a=0.01, b=0.03, c=0.9)) == dict(a=0.03, b=0.06, c=0.9)
    summary = {a: dict(automation=0.5, false_accepts=0) for a in c.ARMS}
    comparisons = {
        a + "_" + m: dict(gain=0.1, interval=[0.02, 0.2], adjusted_p=0.01)
        for a in c.CONTROLS
        for m in ("cost", "brier")
    }
    assert c.benefit(summary, comparisons)
    summary["gibbs"]["false_accepts"] = 1
    assert not c.benefit(summary, comparisons)


def test_positive_control_uses_the_registered_reducer():
    with patch.object(c, "evaluate", wraps=c.evaluate) as reducer:
        result = c.positive_control()
    assert reducer.called
    assert result["comparisons"]["raw_qwen_brier"]["gain"] > 0.1
    assert result["verdict_class"] == "circular_positive"
