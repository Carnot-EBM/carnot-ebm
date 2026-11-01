"""REQ-VERIFY-8608, SCENARIO-REPORT-8608-PROTOTYPE: preserve serving work."""

import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/carnot/inference"))
energy = importlib.import_module("resident_energy")
protocol = importlib.import_module("resident_protocol")


@pytest.fixture
def head():
    return dict(
        arm="spline",
        parameters=[0.01] * 109,
        decay_scale=1.0,
        temperature=1.0,
        scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9),
    )


@pytest.fixture
def row():
    return dict(
        unit_id="private-1",
        q=0.4,
        public=dict(
            family_id="fixture",
            source_bytes=b"There are 12 apples. No pears.".hex(),
            answer_bytes=b"There are 12 apples.".hex(),
        ),
    )


def test_complete_numerical_work(head, row):
    """REQ-VERIFY-8608: exact text feature and serialized response parity."""
    from carnot.reporting.resident_worker_8572 import evaluate

    expected, _ = evaluate(head, row, None)
    actual, spans = energy.evaluate(head, row, None)
    assert actual == expected
    assert [s["operation"] for s in spans] == [
        "public_text_features",
        "scaling",
        "offset",
        "binding_residual",
        "normalization_and_python_fallback",
        "typed_decision",
        "serialization",
    ]
    raw = np.asarray([[0.4] + [0.5] * 8])
    assert energy.basis(raw).shape == (1, 109)


def test_protocol_rejects_stale_and_duplicate(head, row):
    """REQ-VERIFY-8608: failures never inherit a successful answer."""
    value = protocol.envelope("a", "hash", row)
    seen = set()
    assert protocol.decide(value, dict(head=head), "hash", None, seen)["status"] == "ok"
    bad = protocol.decide(value, dict(head=head), "hash", None, seen)
    assert bad["error"] == "duplicate_request_id" and "response" not in bad
    value = protocol.envelope("b", "hash", row)
    value["deadline_ns"] = time.monotonic_ns() - 1
    assert (
        protocol.decide(value, dict(head=head), "hash", None, seen)["error"] == "request_deadline"
    )


def test_features_and_policies(head, row, monkeypatch):
    """REQ-VERIFY-8608: retain abstentions, byte checks and all policy branches."""
    assert [energy.action(p) for p in [0.2, 0.5, 0.8]] == ["accept", "escalate", "reject"]
    for public in [{}, dict(row["public"], source_bytes="xx")]:
        with pytest.raises(ValueError):
            energy.extract(public)
    empty = dict(row["public"], source_bytes="")
    assert energy.extract(empty)["values"] is None
    with pytest.raises(ValueError, match="request_features"):
        energy.batch(head, [dict(row, public=empty)])
    for q in [None, True, float("nan"), -0.1, 1.1]:
        with pytest.raises(ValueError, match="probability_input"):
            energy.evaluate(head, dict(row, q=q), None)
    monkeypatch.setattr(energy.importlib.util, "spec_from_file_location", lambda *a: None)
    with pytest.raises(ValueError, match="extension_spec"):
        energy.load_module(Path("missing"), "missing")


def test_numerical_and_boundary(head, row):
    """REQ-VERIFY-8608: the real extension keeps finite and boundary controls."""
    original = json.loads(
        (ROOT / "results/experiment_8593_v738_resident_provenance_qualification.json").read_bytes()
    )
    ext = original["target_model"]["native_extension_build"]
    native = energy.load_module(Path(ext["path"]), "_nine_input_8533")
    control = dict(
        unit_id="finite", kind="numerical", scaled_features=[0.5] * 9, decay_multiplier=0.125
    )
    a, _ = energy.evaluate(head, control, None)
    b, _ = energy.evaluate(head, control, native)
    assert abs(a["probability"] - b["probability"]) < 1e-12
    zero = dict(head, parameters=[0.0] * 109)
    for q in [0.2, 0.8]:
        a, _ = energy.evaluate(zero, dict(row, q=q), native)
        assert a["boundary_fallback"]
        control["scaled_features"][0] = q
        a, _ = energy.evaluate(zero, control, native)
        assert a["boundary_fallback"]


@pytest.mark.parametrize(
    "patch,error",
    [
        ({"request_id": ""}, "request_id"),
        ({"request_id": "x" * 65}, "request_id"),
        ({"config_hash": "wrong"}, "config_hash"),
        ({"deadline_ns": None}, "request_deadline"),
        ({"deadline_ns": 10**30}, "request_deadline"),
        ({"operation": "bad"}, "operation"),
    ],
)
def test_bad_envelopes(head, row, patch, error):
    """REQ-VERIFY-8608: reject each malformed envelope without response leakage."""
    value = dict(protocol.envelope("id", "hash", row), **patch)
    actual = protocol.decide(value, dict(head=head), "hash", None, set())
    assert actual["error"] == error and "response" not in actual
    assert protocol.decide(None, {}, "hash", None, set())["request_id"] is None


def test_second_deadline_read(head, row, monkeypatch):
    """REQ-VERIFY-8608: both deadline reads and post-evaluation expiry execute."""
    value = protocol.envelope("id", "hash", row)
    deadline = value["deadline_ns"]
    reads = iter([deadline - 1, deadline - 1, deadline])
    monkeypatch.setattr(protocol.time, "monotonic_ns", lambda: next(reads))
    monkeypatch.setattr(energy, "evaluate", lambda *a: ({}, []))
    assert (
        protocol.decide(value, dict(head=head), "hash", None, set())["error"]
        == "evaluation_deadline"
    )


def test_frames(monkeypatch):
    """REQ-VERIFY-8608: exact framing includes EOF, truncation and timeout failures."""
    import os
    import struct

    r, w = os.pipe()
    try:
        protocol.write_frame(w, {"ok": True})
        assert json.loads(protocol.read_frame(r, 1)) == {"ok": True}
        with pytest.raises(TimeoutError, match="frame_deadline"):
            protocol.read_frame(r, 0.001)
        os.write(w, struct.pack("!I", 0))
        with pytest.raises(ValueError, match="oversized_or_empty_frame"):
            protocol.read_frame(r, 1)
        os.write(w, struct.pack("!I", 4) + b"a")
        os.close(w)
        w = -1
        with pytest.raises(ValueError, match="truncated_frame"):
            protocol.read_frame(r, 1)
        assert protocol.read_frame(r, 1) is None
    finally:
        os.close(r)
        if w >= 0:
            os.close(w)
    r, w = os.pipe()
    try:
        with pytest.raises(ValueError, match="oversized_response"):
            protocol.write_frame(w, {"x": "a" * 65536})
        os.set_blocking(w, False)
        while True:
            try:
                os.write(w, b"x" * 4096)
            except BlockingIOError:
                break
        with pytest.raises(TimeoutError, match="write_deadline"):
            protocol.write_frame(w, {"x": "a" * 60000}, 0.001)
    finally:
        os.close(r)
        os.close(w)


def test_configuration_rejections(head, tmp_path, monkeypatch):
    """REQ-VERIFY-8608: pin real loaded paths and reject hash or loader drift."""
    from types import SimpleNamespace

    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(head=head)))
    digest = protocol.sha256_file(path)
    with pytest.raises(ValueError, match="immutable_config"):
        protocol.configuration(path, "bad", "python")
    with pytest.raises(ValueError, match="immutable_config"):
        protocol.configuration(path, digest, "bad")
    ext = tmp_path / "fake.so"
    ext.write_bytes(b"invalid")
    config = dict(head=head, extension=dict(path=str(ext), sha256="bad"))
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="extension_hash"):
        protocol.configuration(path, protocol.sha256_file(path), "native")
    config["extension"]["sha256"] = protocol.sha256_file(ext)
    path.write_text(json.dumps(config))
    monkeypatch.setattr(energy, "load_module", lambda *a: SimpleNamespace())
    with pytest.raises(ValueError, match="extension_entrypoint"):
        protocol.configuration(path, protocol.sha256_file(path), "native")
    wrong = tmp_path / "other.so"
    wrong.write_bytes(b"changed")
    monkeypatch.setattr(
        energy,
        "load_module",
        lambda *a: SimpleNamespace(__file__=str(wrong), nine_input_residual_8533=lambda *a: []),
    )
    with pytest.raises(ValueError, match="loaded_extension"):
        protocol.configuration(path, protocol.sha256_file(path), "native")


def test_serve_owned_limits(head, row, tmp_path, monkeypatch):
    """REQ-VERIFY-8608: unchanged idle, mutation, malformed and request limits execute."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(head=head)))
    digest = protocol.sha256_file(path)
    sent = []
    monkeypatch.setattr(protocol, "write_frame", lambda fd, value, *a: sent.append(value))
    monkeypatch.setattr(protocol, "read_frame", lambda *a: None)
    assert protocol.serve(path, digest, "python") == 0

    def changed(*a):
        path.write_text("{}")
        return b"{}"

    monkeypatch.setattr(protocol, "read_frame", changed)
    assert protocol.serve(path, digest, "python") == 1
    path.write_text(json.dumps(dict(head=head)))
    monkeypatch.setattr(protocol, "MAX_REQUESTS", 101)
    monkeypatch.setattr(protocol, "read_frame", lambda *a: b"invalid-json")
    assert protocol.serve(path, digest, "python") == 1
    assert all("response" not in v for v in sent if v["status"] == "error")

    def idle(*a):
        raise TimeoutError("frame_deadline")

    monkeypatch.setattr(protocol, "read_frame", idle)
    assert protocol.serve(path, digest, "python", 0.001) == 1


@pytest.mark.parametrize(
    "payload", [b"", b"\x00", b"\x00\x00\x00\x00", b"\x00\x01\x00\x01", b"\x00\x00\x00\x04x"]
)
def test_actual_child_rejections(head, tmp_path, payload):
    """SCENARIO-VERIFY-8608-PRIVATE: actual children reject bad frames and accept EOF."""
    import subprocess

    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(head=head)))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts/experiments/resident_worker_8608.py"),
            str(path),
            protocol.sha256_file(path),
            "python",
            "--idle-timeout",
            "0.1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = proc.communicate(payload, timeout=30)
    assert proc.returncode == (0 if not payload else 1), err
    assert b"ready" in out
