"""REQ-REPORT-8719, REQ-VERIFY-8719: cross the real machine arithmetic boundary."""

from copy import deepcopy
from pathlib import Path
import json

import pytest

from carnot.verify import compiled_sparse_8719 as n
from carnot.reporting import compiled_sparse_study_8719 as e
from carnot.reporting import compiled_sparse_runner_8719 as runner
from test_spline_lifecycle_8706 import fixture


def test_native_parity_and_serialization(tmp_path):
    """SCENARIO-REPORT-8719-PARITY: compare states and counters at every update."""
    extension = n.load(Path(n.DEFAULT_EXTENSION))
    for boundary in [False, True]:
        job = dict(fixture(), boundary=boundary, boundary_offset=-1, updates=8)
        result = n.drive(job, tmp_path / str(boundary), 0, 8, extension)
        assert result["passed"] and result["observed_updates"] == 8
        assert result["checkpoint"]["counts"]["overflow"] == 0
        assert result["checkpoint"]["counts"]["renormalization"] == int(boundary)
    assert all(r["passed"] for r in n.controls(extension))


def test_restart_identity(tmp_path):
    """REQ-VERIFY-8719: independently resumed bytes preserve acknowledgements."""
    extension = n.load(Path(n.DEFAULT_EXTENSION))
    job = dict(fixture(), updates=8)
    full = n.drive(job, tmp_path / "full", 0, 8, extension)
    n.drive(job, tmp_path / "split", 0, 4, extension)
    split = n.drive(job, tmp_path / "split", 4, 8, extension)
    assert full["checkpoint"] == split["checkpoint"]
    with pytest.raises(ValueError, match="restart_identity"):
        n.drive(dict(job, sequence="other"), tmp_path / "split", 4, 8, extension)
    (tmp_path / "split" / "checkpoint.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="restart_serialization"):
        n.drive(job, tmp_path / "split", 8, 9, extension)


def test_host_ingress():
    """SCENARIO-REPORT-8719-PARITY: reject invalid host inputs before native work."""
    extension = n.load(Path(n.DEFAULT_EXTENSION))
    f = n.NativeFixed(extension)
    with pytest.raises(ValueError, match="finite_operand"):
        f.q(float("nan"))
    assert f.q(1e9) == (1 << 31) - 1
    state = n.numeric.encode(fixture()["head"], f)
    for raw in [[float("nan")] * 9, [0.5] * 8]:
        with pytest.raises(ValueError, match="input_contract"):
            n.probability(state, raw, f)
    for label in [True, 2]:
        with pytest.raises(ValueError, match="binary_label"):
            n.step(state, [0.4] * 9, label, f)
    for key, value in [("decay_scale", 1), ("temperature", 1), ("parameters", [2**31] * 109)]:
        bad = dict(deepcopy(state), **{key: value})
        with pytest.raises(ValueError, match="supported_domain"):
            n.step(bad, [0.4] * 9, 0, f)


def test_gate_states(tmp_path):
    """REQ-VERIFY-8719: absence and observed zero are different operands."""
    p = tmp_path / "upstream.json"
    r = e.reference(p)
    assert e.gate(r, "ready", 1)["operand_state"] == "missing_file"
    p.write_text("{}")
    r = e.reference(p)
    assert e.gate(r, "ready", 1)["operand_state"] == "missing_field"
    p.write_text('{"ready": 0}')
    assert e.gate(r, "ready", 1)["operand_state"] == "changed_bytes"
    r = e.reference(p)
    assert e.gate(r, "ready", 1)["operand_state"] == "observed_zero"
    assert e.gate(r, "ready", 0)["passed"]


def test_authority():
    """REQ-REPORT-8719: this independent branch authenticates its own authority."""
    value = e.inputs()
    assert not value["failures"]
    assert len(value["jobs"]) == 25
    assert value["contract"]["ready"]


def test_original_gate_rejections(monkeypatch):
    """REQ-REPORT-8719: flags, malformed primitives and changed jobs cannot qualify."""
    admitted = e.inputs()
    real_read = e.read
    for mode in ["gate", "primitive", "job"]:
        with monkeypatch.context() as m:
            if mode == "gate":
                real_gate = e.gate
                m.setattr(e, "gate", lambda *args: dict(real_gate(*args), passed=False))
            else:

                def altered(ref):
                    value = real_read(ref)
                    if (
                        mode == "primitive"
                        and ref["path"].endswith("/measurement.json")
                        and "8706" in ref["path"]
                    ):
                        value["private_control"] = True
                    if mode == "job" and ref["path"].endswith("/jobs/00.json"):
                        value["updates"] = 1
                    return value

                m.setattr(e, "read", altered)
            assert e.inputs()["failures"]
    assert len(admitted["jobs"]) == 25


def test_negative_or_altered_witnesses(tmp_path, monkeypatch):
    """REQ-VERIFY-8719: missing historical parity and bad serialization are failures."""
    extension = n.load(Path(n.DEFAULT_EXTENSION))
    job = dict(fixture(), updates=2)
    path = tmp_path / "history.jsonl"
    path.write_text(
        json.dumps(dict(state_hash="wrong", ack="wrong", probability=0, passed=True)) + "\n"
    )
    result = n.drive(job, tmp_path / "drift", 0, 1, extension, path)
    assert not result["passed"]
    with monkeypatch.context() as m:
        m.setattr(n, "serialize", lambda *args: b"bad")
        with pytest.raises(ValueError, match="serialized_order"):
            n.drive(job, tmp_path / "order", 0, 1, extension)
    real_call = n.NativeFixed.call

    def permissive(self, operation, **operands):
        if operation == "basis" and operands.get("scaled", [0])[0] == -1:
            return None
        return real_call(self, operation, **operands)

    with monkeypatch.context() as m:
        m.setattr(n.NativeFixed, "q", lambda self, value: 0)
        assert not all(row["passed"] for row in n.controls(extension))
    with monkeypatch.context() as m:
        m.setattr(n.NativeFixed, "call", permissive)
        assert not all(row["passed"] for row in n.controls(extension))


def test_observed_absence_binding(tmp_path):
    """REQ-VERIFY-8719: observed absence stays bound separately from required bytes."""
    path = tmp_path / "absent"
    ref = e.reference(path)
    e.verify(ref)
    path.write_bytes(b"appeared")
    with pytest.raises(ValueError, match="changed_absence"):
        e.verify(ref)
