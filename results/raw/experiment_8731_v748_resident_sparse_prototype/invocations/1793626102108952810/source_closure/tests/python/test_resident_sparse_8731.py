"""REQ-REPORT-8731, REQ-VERIFY-8731: resident arithmetic crosses the real binding."""

from copy import deepcopy
from pathlib import Path
import json

import pytest

from carnot.verify import resident_sparse_8731 as n
from test_spline_lifecycle_8706 import fixture


def test_resident_exact_updates_and_checkpoints(tmp_path):
    """SCENARIO-REPORT-8731-PARITY: state and counters match after each commit."""
    extension = n.load(Path(n.DEFAULT_EXTENSION))
    for boundary in [False, True]:
        job = dict(fixture(), boundary=boundary, boundary_offset=-1, updates=8)
        full = n.drive(job, tmp_path / str(boundary), 0, 8, extension)
        assert full["passed"] and full["observed_updates"] == 8
        assert full["checkpoint"]["counts"]["renormalization"] == int(boundary)
        assert full["binding_call_rows"] and full["transferred_byte_rows"]
        assert all(
            r["coefficient_uploads"] == 0
            for r in full["binding_call_rows"]
            if r["operation"] in ["prepare", "apply"]
        )
    n.drive(job, tmp_path / "restart", 0, 4, extension)
    split = n.drive(job, tmp_path / "restart", 4, 8, extension)
    assert split["checkpoint"] == full["checkpoint"]
    assert all(r["passed"] for r in n.controls(extension))


def test_resident_rejected_feedback_and_errors():
    """REQ-VERIFY-8731: duplicate, late and rejected work cannot advance state."""
    extension = n.load(Path(n.DEFAULT_EXTENSION))
    f = n.ResidentFixed(extension)
    state = n.numeric.encode(fixture()["head"], f)
    f.bind(state)
    for label in [True, 2]:
        with pytest.raises(ValueError, match="binary_label"):
            n.engine.step(state, [0.4] * 9, label, f)
    for raw in [[float("nan")] * 9, [0.4] * 8]:
        with pytest.raises(ValueError, match="input_contract"):
            n.probability(state, raw, f)
    with pytest.raises(ValueError, match="finite_operand"):
        f.q(float("nan"))
    assert f.q(1e9) == 2**31 - 1
    before = f.resident.audit(0)
    for method in ["audit", "telemetry", "serialize"]:
        with pytest.raises(ValueError, match="diagnostic_token"):
            getattr(f.resident, method)(1)
    with pytest.raises(ValueError, match="pending_probability"):
        f.resident.apply(2**23, 1, 0)
    assert before == f.resident.audit(0)
    f.resident.prepare([0] * 16, [0] * 9)
    for probability, label in [(-1, 1), (0, 2)]:
        with pytest.raises(ValueError, match="supported_domain"):
            f.resident.apply(probability, label, 1)
    for field, value in [("decay_scale", 1), ("temperature", 1), ("parameters", [2**31] * 109)]:
        bad = dict(deepcopy(state), **{field: value})
        with pytest.raises(ValueError, match="supported_domain"):
            n.ResidentFixed(extension).bind(bad)
    assert all(row["passed"] for row in n.feedback_controls(extension))
    with pytest.raises(ValueError, match="resident_operation"):
        f.call("unknown")


def test_missing_resident_entrypoint(monkeypatch, tmp_path):
    """REQ-VERIFY-8731: loading an unrelated binary is insufficient evidence."""
    monkeypatch.setattr(n.original, "load", lambda path: object())
    with pytest.raises(ValueError, match="resident_entrypoint"):
        n.load(tmp_path / "unrelated.so")


def test_historical_native_witness_is_required(tmp_path):
    """SCENARIO-VERIFY-8731-REPLAY: altered native witnesses cannot pass parity."""
    path = tmp_path / "history.jsonl"
    path.write_text(
        json.dumps(dict(state_hash="bad", ack="bad", probability=0, passed=True)) + "\n"
    )
    result = n.drive(
        dict(fixture(), updates=1),
        tmp_path / "drift",
        0,
        1,
        n.load(Path(n.DEFAULT_EXTENSION)),
        path,
    )
    assert not result["passed"]
