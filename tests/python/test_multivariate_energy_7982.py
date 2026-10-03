"""REQ-VERIFY-7982: matched fitting, gradients, leakage and circular controls."""

import copy

import numpy as np
import pytest

from carnot.verify import multivariate_energy_7982 as m


def fixture(shuffled=False):
    rng = np.random.default_rng(69282)
    data = {}
    for role, count in [("fit", 128), ("tune", 32), ("policy_design", 32)]:
        labels = np.arange(count) % 2
        if shuffled:
            rng.shuffle(labels)
        data[role] = [
            dict(
                family_id=f"{role}-{i}",
                source_cluster_id=f"{role}-{i}",
                q=0.5,
                features=[float(i % 2)] * 8,
                y=int(labels[i]),
                status="completed",
            )
            for i in range(count)
        ]
    return data


@pytest.mark.parametrize(
    "arm,count", [("logistic", 10), ("quadratic", 55), ("gibbs", 97), ("mlp", 177), ("spline", 63)]
)
def test_exact_gradients_and_normalization(arm, count):
    """SCENARIO-VERIFY-7982-FITTING: finite differences check every coefficient."""
    x = np.random.default_rng(2).normal(size=(5, 9))
    knots = m.knots(x)
    theta = np.random.default_rng(3).normal(0, 0.2, count)
    z, jac = m.logits_jacobian(arm, theta, x, knots)
    for i in range(count):
        delta = np.zeros(count)
        delta[i] = 1e-6
        numerical = (
            m.logits_jacobian(arm, theta + delta, x, knots)[0]
            - m.logits_jacobian(arm, theta - delta, x, knots)[0]
        ) / 2e-6
        np.testing.assert_allclose(jac[:, i], numerical, atol=1e-8)
    head = dict(arm=arm, parameters=theta.tolist(), temperature=0.5, knots=knots)
    p = m.predict(head, x)
    assert np.all((p >= 0) & (p <= 1))
    if arm == "gibbs":
        e = m.energies(theta, x)
        weights = np.exp(-e / 0.5 - np.max(-e / 0.5, axis=1, keepdims=True))
        np.testing.assert_allclose(p, weights[:, 1] / weights.sum(axis=1), atol=1e-12)
        np.testing.assert_allclose(p, m.sigmoid_conversion(head, x), atol=1e-12)
    assert np.isfinite(z).all()


def test_circular_separable_and_shuffle():
    data = fixture()
    measured = m.fit(data)
    assert measured["fit_support"]["passed"] and measured["tune_support"]["passed"]
    assert measured["feature_normalization"]["means"][0] == 0
    assert measured["feature_normalization"]["scales"][0] == 1
    assert all(r["passed"] for r in measured["gradient_checks"])
    assert measured["parity_max_error"] <= 1e-12
    assert measured["summary"]["gibbs"]["fit_brier"] < 0.02
    assert measured["optimizer_work"]["total_steps"] == 3000
    assert measured["optimizer_work"]["pretrained_model_calls"] == 0
    altered = fixture()
    altered["tune"][0]["features"] = [100.0] * 8
    changed = m.fit(altered)
    assert changed["feature_normalization"] == measured["feature_normalization"]
    for arm in m.ARMS:
        assert [h["parameters"] for h in changed["heads"][arm]] == [
            h["parameters"] for h in measured["heads"][arm]
        ]
    shuffled = m.fit(fixture(True))
    assert shuffled["summary"]["gibbs"]["fit_brier"] > 0.15
    assert m.predict(measured["heads"]["gibbs"][0], np.empty((0, 9))).size == 0
    with pytest.raises(ValueError, match="arm"):
        m.logits_jacobian("unknown", np.zeros(1), np.zeros((1, 9)), [])


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("fields", "predictor_fields"),
        ("overlap", "cross_role"),
        ("label", "label"),
        ("q", "probability"),
        ("features", "features"),
        ("roles", "role_roster"),
        ("support", "support_floor"),
    ],
)
def test_invalid_inputs_fail_closed(mutation, reason):
    data = fixture()
    if mutation == "fields":
        data["fit"][0]["annotation"] = "hidden"
    elif mutation == "overlap":
        data["tune"][0]["source_cluster_id"] = "fit-0"
    elif mutation == "label":
        data["fit"][0]["y"] = -1
    elif mutation == "q":
        data["fit"][0]["q"] = float("nan")
    elif mutation == "features":
        data["fit"][0]["features"] = [1.0] * 7
    elif mutation == "roles":
        del data["fit"]
    else:
        data["fit"] = data["fit"][:127]
    with pytest.raises(ValueError, match=reason):
        m.fit(data)


def test_unknowns_and_predictor_only_policy():
    data = fixture()
    data["policy_design"][0]["y"] = None
    data["policy_design"][0]["q"] = None
    data["policy_design"][1]["features"] = None
    measured = m.fit(data)
    rows = m.predictions(measured, data)
    for arm in ("gibbs", "exact_sigmoid_conversion"):
        selected = {(r["family_id"], r["seed"]): r["p"] for r in rows if r["arm"] == arm}
        assert len(selected) == sum(len(rr) for rr in data.values()) * 4
    assert rows[0]["role"] == "fit"
    assert all(
        r["p"] is None for r in rows if r["family_id"] in {"policy_design-0", "policy_design-1"}
    )
    changed = copy.deepcopy(data)
    for r in changed["policy_design"]:
        r["y"] = 1
    assert m.predictions(measured, changed) == rows
