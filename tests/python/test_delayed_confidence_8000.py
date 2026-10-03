"""REQ-VERIFY-8000: tests precede the delayed confidence implementation."""

import copy
import json

import pytest

from carnot.verify import delayed_confidence_8000 as m


def fixture():
    """SCENARIO-VERIFY-8000-CAUSAL: a deterministic fixture has no external labels."""
    return dict(
        calibration=[
            dict(family_id=f"c{i}", source_cluster_id=f"c{i}", p=0.1, y=0, status="completed")
            for i in range(64)
        ],
        stream=[
            dict(family_id=f"s{i}", source_cluster_id=f"s{i}", p=0.1, status="completed")
            for i in range(256)
        ],
        targets={f"s{i}": int(i % 10 == 0) for i in range(256)},
    )


def test_quantile_sets_temperature():
    """REQ-VERIFY-8000: finite sample correction keeps boundary sets explicit."""
    assert m.cutoff([], 0.1) is None
    assert m.cutoff([0.1] * 8, 0.1) is None
    assert m.cutoff(list(range(9)), 0.1) == 8
    assert m.cutoff([0.1, 0.9], 0.5) == 0.9
    assert m.prediction_set(0.5, 0.1) == []
    assert m.prediction_set(0.5, None) == [0, 1]
    assert m.prediction_set(0.1, 0.2) == [0]
    assert m.prediction_set(0.9, 0.2) == [1]
    assert [m.action(s) for s in ([0], [1], [], [0, 1])] == [
        "accept",
        "reject",
        "escalate",
        "escalate",
    ]
    assert m.transform(0, 0.5) == 0 and m.transform(1, 2) == 1
    assert m.transform(0.5, 1) == 0.5
    t, scores = m.calibrate([dict(p=0.5, y=0)])
    assert t == 0.5 and scores == [0.5]


def test_issued_base_and_restart():
    """SCENARIO-VERIFY-8000-CAUSAL: current and issued bases must differ."""
    f = fixture()
    state = m.initial([0.1] * 64, 20, "interleaved")
    state["alpha"] = 0.3
    for slot, row in enumerate(f["stream"][:21], 1):
        m.step(state, row, slot, f["targets"].__getitem__)
    feedback = state["feedback"][-1]
    assert feedback["base_alpha"] == 0.3
    assert feedback["issue_slot"] == 1 and feedback["release_slot"] == 21
    assert feedback["alpha_after"] == pytest.approx(0.291)
    assert len(state["pending"]) == 20
    full = m.run(f, "scalar", 20)
    restored = m.run(f, "scalar", 20, restart=True)
    assert full == restored
    frozen = m.run(f, "frozen", 20)
    assert all(r["issue_alpha"] == 0.1 for r in frozen["issued"])


def test_future_labels_and_censor():
    """SCENARIO-VERIFY-8000-CAUSAL: unseen outcomes cannot change earlier sets."""
    f = fixture()
    a = m.run(f, "interleaved", 20)
    changed = copy.deepcopy(f)
    changed["targets"]["s100"] ^= 1
    b = m.run(changed, "interleaved", 20)
    assert a["issued"][:121] == b["issued"][:121]
    f["stream"][5]["p"] = None
    f["stream"][5]["status"] = "excluded"
    f["targets"]["s8"] = None
    censored = m.run(f, "scalar", 20)
    assert not censored["issued"][5]["eligibility"]
    assert any(r["censor_status"] for r in censored["feedback"])
    assert len(censored["pool"]) == 64
    assert json.loads(json.dumps(censored)) == censored


def test_controls_and_support():
    """REQ-REPORT-8000: fixtures prove responsiveness without scientific claims."""
    f = fixture()
    result = m.measure(f)
    assert result["point_brier_parity"]["passed"]
    assert all(r["passed"] for r in result["restart_rows"])
    assert result["positive_control_results"]["passed"]
    assert len(result["coverage_windows"]) == 27
    small = fixture()
    for r in small["stream"]:
        r["source_cluster_id"] = "one"
    assert not m.measure(small)["acceptance_gate_results"]["support"]
    assert m.controls()["no_shift"]["passed"]
    with pytest.raises(ValueError, match="probability"):
        m.prediction_set(2, 0.1)
