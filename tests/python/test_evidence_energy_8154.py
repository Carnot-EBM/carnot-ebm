"""REQ-VERIFY-8154: matched information, bounded fitting and numerical parity."""

from copy import deepcopy
import time

import numpy as np
import pytest

from carnot.verify import evidence_energy_8154 as e


def data():
    rng = np.random.default_rng(8154)
    rows = []
    for role, count in (("fit", 32), ("tune", 16)):
        for i in range(count):
            x = rng.normal(size=12)
            rows.append(
                dict(
                    unit_id=f"{role}{i}",
                    source_cluster_id=f"{role}{i}",
                    role=role,
                    x=x.tolist(),
                    y=i % 2,
                    status="completed",
                    exclusion_reason=None,
                )
            )
    return rows


def test_geometry_and_information_controls():
    """SCENARIO-VERIFY-8154: zero scales, knots and public centers stay fit-only."""
    x = np.arange(384, dtype=float).reshape(32, 12)
    x[:, 11] = 0
    ids = [str(i) for i in range(32)]
    g = e.geometry(x, ids)
    assert g["scale"][11] == 1
    assert len(g["centers"]) == 16
    assert g == e.geometry(x, ids)
    for arm in e.FITTED_ARMS:
        a = e.design(arm, x, g)
        assert np.isfinite(a).all() and np.all(a[:, 0] == 1)
        assert np.array_equal(a[:1], e.design(arm, x[:1], g))
    assert e.design("linear12", x, g).shape == (32, 13)
    assert e.design("radial16", x, g).shape == (32, 17)
    assert e.design("additive_cubic", x, g).shape == (32, 85)
    assert e.geometry(np.zeros((32, 12)), ids)["width"] == 1
    with pytest.raises(ValueError):
        e.design("unknown", x, g)
    assert e.choose_ridge({0.001: 1.0, 1.0: 1.0}) == 1.0


def test_solver_failures_and_intercept():
    """REQ-VERIFY-8154: invalid targets and timeouts cannot become ready heads."""
    phi = np.column_stack((np.ones(32), np.arange(32) / 32))
    y = np.arange(32) % 2
    fitted = e.solve(phi, y, 0.01, time.monotonic() + 10)
    assert fitted["converged"] and fitted["iterations"] <= 256
    assert fitted["final_loss"] <= fitted["initial_loss"]
    theta = np.array([2.0, 0.0])
    assert e.objective(theta, phi, y, 1.0)[0] == e.objective(theta, phi, y, 0.0)[0]
    for bad in (np.zeros(32), np.full(32, 2)):
        with pytest.raises(ValueError):
            e.solve(phi, bad, 0.01, time.monotonic() + 10)
    with pytest.raises(TimeoutError):
        e.solve(phi, y, 0.01, 0)


def test_fit_freeze_parity_and_permutation(tmp_path):
    """REQ-VERIFY-8154: all architectures freeze before any reserved capture."""
    rows = data()
    fitted = e.train(rows, tmp_path)
    assert len(fitted["heads"]) == 6 and not fitted["failures"]
    assert len(fitted["fit_fold_rows"]) == 6 * 5 * 4
    assert all(r["fit_source_ids"] != r["held_source_ids"] for r in fitted["fit_fold_rows"])
    measured = e.evaluate(rows, fitted["heads"])
    assert all(r["passed"] for r in measured["energy_logistic_parity_rows"])
    assert {r["arm"] for r in measured["rows"]} == set(e.ARMS)
    assert len(measured["tuning_rows"]) == 16 * 8
    assert all(r["E0"] == 0 and r["E1"] == -r["z"] for r in measured["energy_logistic_parity_rows"])
    swapped = deepcopy(rows)
    for r in swapped:
        r["y"] = 1 - r["y"]
    permuted = e.train(swapped, tmp_path / "permuted")
    assert fitted["heads"][0]["weights"] != permuted["heads"][0]["weights"]
    leaked = deepcopy(rows)
    leaked[-1]["source_cluster_id"] = leaked[0]["source_cluster_id"]
    with pytest.raises(ValueError, match="source_leak"):
        e.train(leaked, tmp_path / "leak")
    zeros = deepcopy(rows)
    for r in zeros:
        r["x"] = [0.0] * 12
    assert not e.train(zeros, tmp_path / "zeros")["failures"]


def test_recorded_fit_failures_and_heartbeat(tmp_path, monkeypatch, capsys):
    """SCENARIO-VERIFY-8154: unfinished solves retain failed fold/full/calibration receipts."""
    from types import SimpleNamespace

    original = e.solve
    for stage, length in (("fold", 24), ("full_fit", 32), ("calibration", 16)):

        def unfinished(phi, y, *args, **kwargs):
            result = original(phi, y, *args, **kwargs)
            if len(y) == length:
                result["converged"] = False
            return result

        monkeypatch.setattr(e, "solve", unfinished)
        result = e.train(data(), tmp_path / stage)
        assert result["failures"]
        assert all(r.get("passed") is False for r in result["failures"] if "fold" in r)
    monkeypatch.setattr(e, "solve", original)
    degenerate = data()
    for row in degenerate:
        row["y"] = 0
    assert e.train(degenerate, tmp_path / "degenerate")["failures"]
    ticks = iter([0.0, 0.0, 61.0, 62.0])
    monkeypatch.setattr(e, "time", SimpleNamespace(monotonic=lambda: next(ticks)))

    def optimizer(fn, theta, **kwargs):
        kwargs["callback"](theta)
        value, gradient = fn(theta)
        return SimpleNamespace(
            x=theta, fun=value, jac=gradient, success=True, nit=1, message="fixture"
        )

    monkeypatch.setattr(e, "minimize", optimizer)
    e.solve(np.ones((4, 2)), np.array([0.0, 1.0, 0.0, 1.0]), 0.01, 500)
    assert "fit_iterations" in capsys.readouterr().out


def test_independent_headline_reduction_and_ties(tmp_path):
    """REQ-VERIFY-8154: independently recomputed decision cost binds per-source rows."""
    from carnot.verify import evidence_energy_fit_8154 as runner

    rows = data()
    measured = e.evaluate(rows, e.train(rows, tmp_path)["heads"])
    independent = runner.independent_reduce(measured["rows"])
    for actual, expected in zip(independent, measured["development_metrics"], strict=True):
        for key in ("typed_cost", "brier", "log_loss", "coverage", "false_accept"):
            assert (
                actual[key] == pytest.approx(expected[key])
                if actual[key] is not None
                else expected[key] is None
            )
    assert e.action(0.1) == e.action(0.5) == "escalate"
    changed = deepcopy(measured["rows"])
    changed[0]["numerator"] = 700
    with pytest.raises(ValueError, match="primitive_cost_drift"):
        runner.independent_reduce(changed)
