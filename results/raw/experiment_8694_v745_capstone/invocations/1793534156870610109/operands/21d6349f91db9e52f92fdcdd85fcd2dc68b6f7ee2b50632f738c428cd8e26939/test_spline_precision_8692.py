"""REQ-VERIFY-8692: integer checks use independent float64 and hand-computed operands."""

from copy import deepcopy
import numpy as np
import pytest

from carnot.verify import spline_precision_8692 as e
from carnot.verify import objective_checkpoint_8461 as original


def head():
    """A finite constructed head isolates arithmetic from historical admission."""
    return dict(
        arm="spline",
        parameters=np.linspace(-0.04, 0.04, 109).tolist(),
        decay_scale=1.0,
        temperature=2.0,
        parameter_count=109,
        scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9),
    )


def test_integer_rounding_saturation_and_checked_accumulators():
    """SCENARIO-VERIFY-8692-ARITHMETIC: signed ties are even and never wrap."""
    f = e.Fixed(16, 12)
    assert [f.ratio(n, 2) for n in [1, 3, 5, -1, -3, -5]] == [0, 2, 2, 0, -2, -2]
    assert f.q(0.5) == 2048
    assert f.sat(40000) == 32767
    assert f.sat(-40000) == -32768
    assert f.counts["saturation"] == 2
    assert f.mul(2048, 2048) == 1024
    assert f.div(2048, 1024) == 8192
    with pytest.raises(ValueError):
        f.div(1, 0)
    with pytest.raises(OverflowError):
        f.checked(2**63)
    with pytest.raises(OverflowError):
        f.checked(-(2**63) - 1)
    with pytest.raises(ValueError):
        e.Fixed(8, 8)
    with pytest.raises(ValueError):
        f.q(float("nan"))


@pytest.mark.parametrize("width,frac", [(16, 12), (24, 18), (32, 24)])
def test_local_basis_and_one_update(width, frac):
    """SCENARIO-VERIFY-8692-ARITHMETIC: cubic boundaries preserve support and Brier sign."""
    h = head()
    f = e.Fixed(width, frac)
    for t in [0.0, 1e-9, 1 / 9, 0.5, 8 / 9, 1.0]:
        ids, weights = e.local_basis(f, [t] * 9)
        dense = original.sparse.basis(np.array([[t] * 9]))[0]
        reconstructed = np.zeros(109)
        reconstructed[ids] = np.array(weights) / f.unit
        assert len(ids) <= 37
        np.testing.assert_allclose(reconstructed, dense, atol=8 / f.unit, rtol=0)
    state = e.encode(h, f)
    untouched = deepcopy(state)
    x = np.array([0.4] * 9)
    p, _ = e.probability(state, x.tolist(), f)
    assert abs(p - original.sparse.predict(h, x[None, :])[0]) < 0.001
    state, receipt = e.step(state, x.tolist(), 1, f)
    reference, _ = original.update(h, x, 1)
    assert receipt["local_writes"] <= 37
    assert receipt["global_decay_writes"] == 1
    assert state["temperature"] == untouched["temperature"]
    np.testing.assert_allclose(
        e.effective(state, f), original.sparse.parameters(reference), atol=0.001
    )
    assert untouched == e.encode(h, e.Fixed(width, frac))
    with pytest.raises(ValueError):
        e.step(state, x.tolist(), 2, f)


def test_decay_underflow_and_contract():
    """SCENARIO-VERIFY-8692-ARITHMETIC: underflow is counted and renormalization explicit."""
    f = e.Fixed(16, 12)
    s = e.encode(head(), f)
    s["decay_scale"] = 1
    e.step(s, [0.5] * 9, 0, f)
    assert f.counts["decay_underflow"] == 1
    assert f.counts["renormalization"] == 1
    for change in [dict(temperature=0), dict(decay_scale=0), dict(parameters=[0.0] * 108)]:
        with pytest.raises(ValueError):
            e.encode(dict(head(), **change), f)
    with pytest.raises(ValueError):
        e.probability(s, [float("inf")] * 9, f)


def test_frozen_geometry_sequences_and_decisions():
    """SCENARIO-VERIFY-8692-CUSTODY: vector selection cannot read labels."""
    h = head()
    frozen = e.vectors([h])
    assert frozen == e.vectors([h])
    assert len(frozen["vectors"]) == 256
    assert [r["kind"] for r in frozen["vectors"]].count("interior") == 128
    assert len(e.sequences()) == 4
    assert all(len(v) == 256 for v in e.sequences().values())
    assert e.decision(0.2) == "accept"
    assert e.decision(0.8) == "reject"
    assert e.decision(0.5) == "escalate"
    unreachable = dict(h, temperature=1e8)
    assert all(r["unreachable"] for r in e.vectors([unreachable])["threshold_searches"])
