"""REQ-VERIFY-7996 and REQ-SELF-7996: local updates must equal dense math."""

import copy
import json

import numpy as np
import pytest

from carnot.verify import sparse_energy_7996 as m


def fixture():
    return {
        role: [
            dict(
                family_id=f"{role}-{i}",
                source_cluster_id=f"{role}-{i}",
                q=0.5,
                features=[float(i % 2)] * 8,
                y=i % 2,
                status="completed",
            )
            for i in range(n)
        ]
        for role, n in [("fit", 128), ("tune", 32)]
    }


def test_basis_scaler_and_invalid_inputs():
    x = np.array([[0.0] * 9, [1.0] * 9])
    scaler = m.fit_scaler(x)
    clipped, counters = m.scale(np.array([[-1.0] * 9, [2.0] * 9]), scaler)
    assert counters == {"below": [1] * 9, "above": [1] * 9}
    assert np.array_equal(clipped, x)
    for knot in m.KNOTS:
        for value in [knot, np.nextafter(knot, 0), np.nextafter(knot, 1)]:
            matrix = m.basis(np.full((1, 9), np.clip(value, 0, 1)))
            assert matrix.shape == (1, 109)
            assert np.allclose(matrix[0, :-1].reshape(9, 12).sum(axis=1), 1)
            assert np.count_nonzero(matrix[0, :-1]) <= 36
    m.validate_data(fixture())
    for change in [dict(extra=[]), dict(fit=[])]:
        bad = fixture()
        bad.update(change)
        with pytest.raises(ValueError):
            m.validate_data(bad)
    bad = fixture()
    bad["tune"][0]["family_id"] = "fit-0"
    with pytest.raises(ValueError):
        m.validate_data(bad)
    bad = fixture()
    bad["fit"][0]["q"] = float("nan")
    with pytest.raises(ValueError):
        m.validate_data(bad)


def test_sparse_dense_finite_differences_sign_and_serialization():
    measured = m.train(fixture())
    head = measured["heads"]["spline"][0]
    x = np.array([[0.5] + [0.4] * 8])
    p = m.predict(head, x)[0]
    assert m.update(head, x[0], 1, 0)[0] == head
    up, work = m.update(head, x[0], 1, 0.01)
    down, _ = m.update(head, x[0], 0, 0.01)
    assert m.predict(up, x)[0] > p > m.predict(down, x)[0]
    assert work["coefficient_touches"] <= 37
    assert work["global_decay_writes"] == 1
    assert m.audit(head, x[0])["passed"]
    assert np.array_equal(m.predict(head, x), m.predict(json.loads(json.dumps(head)), x))
    assert np.max(np.abs(m.predict(head, x) - m.sigmoid_predict(head, x))) < 1e-10
    for y in [0, 1]:
        for lr in [0.0, 0.01]:
            dense = m.dense_gradient(head, x[0], y)
            sparse, report = m.update(head, x[0], y, lr)
            assert np.allclose(m.parameters(sparse), m.parameters(head) - lr * dense, atol=1e-12)
            assert report["coefficient_touches"] <= 37
    assert np.isfinite(m.predict(head, np.array([[0.0] * 9, [1.0] * 9]))).all()
    with pytest.raises(ValueError):
        m.update(head, x[0], 2, 0.01)
    with pytest.raises(ValueError):
        m.update(head, x[0], 1, -1)


def test_training_controls_all_seeds_and_loss_accounting():
    result = m.train(fixture())
    assert result["optimizer_work"]["total_steps"] == 3000
    assert result["positive_control_results"]["passed"]
    for arm, heads in result["heads"].items():
        assert [h["seed"] for h in heads] == list(m.SEEDS)
        for h in heads:
            assert h["parameter_count"] == m.COUNTS[arm]
            assert h["temperature"] in [0.5, 1.0, 2.0]
            assert h["loss_increased"] == (h["final_loss"] > h["initial_loss"])
            x = np.array([[0.5] + [0.3] * 8])
            assert m.finite_difference(h, x[0]) < 1e-7
    empty = np.empty((0, 9))
    assert m.predict(result["heads"]["spline"][0], empty).size == 0
    bad = copy.deepcopy(result["heads"]["spline"][0])
    bad["arm"] = "unknown"
    with pytest.raises(ValueError):
        m.predict(bad, np.array([[0.5] * 9]))
    missing = fixture()
    missing["fit"][0]["q"] = None
    assert len(m.usable(missing["fit"])) == 127


def test_sparse_reads_only_local_weights_and_counts_snapshot_cost():
    head = m.train(fixture())["heads"]["spline"][0]
    x = np.array([0.5] + [0.4] * 8)
    from unittest.mock import patch

    with patch.object(m, "parameters", side_effect=AssertionError("dense coefficient read")):
        ids, gradient = m.sparse_gradient(head, x, 1)
        changed, work = m.update(head, x, 1, 0.01)
    assert len(ids) <= 37 and len(gradient) == len(ids)
    assert work["snapshot_coefficient_copies"] == 109
    assert np.isfinite(m.predict(changed, x[None, :])).all()
    control = m.train(fixture())["positive_control_results"]
    assert set(control["fit_ids"]).isdisjoint(control["holdout_ids"])
    assert len(control["rows"]) == 32
