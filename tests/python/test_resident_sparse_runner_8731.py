"""REQ-VERIFY-8731: bind current authority and seal genuine private execution."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import resident_sparse_study_8731 as api


def test_original_inputs_and_isolation():
    """REQ-REPORT-8731: V748 and original numerical methods are separate operands."""
    value = api.inputs()
    assert not value["failures"]
    assert value["contract"]["ready"] and len(value["contract"]["rows"]) == 14
    assert len(value["jobs"]) == 25
    from carnot.reporting import compiled_sparse_study_8719 as historical

    assert historical.NAME == "experiment_8719_v747_compiled_sparse_update"


@pytest.fixture(scope="module")
def private(tmp_path_factory):
    """Actual private workers exercise hard exit, import and unchanged publication."""
    output = tmp_path_factory.mktemp("resident8731") / (api.e.NAME + ".json")
    assert api.runtime.run(output, "20261102", True, Path(api.n.DEFAULT_EXTENSION)) == 0
    return output


def test_private_cli_and_replay(private, tmp_path):
    """SCENARIO-VERIFY-8731-REPLAY: real processes retain exact restart state."""
    value = json.loads(private.read_bytes())
    assert value["completed_count"] == 2 and value["resident_update_ready_score"] == 0
    assert value["restart_rows"][0]["real_exit"] == 73
    assert value["binding_call_rows"] and value["transferred_byte_rows"]
    api.runtime.replay(private)
    command = [sys.executable, str(api.e.ROOT / api.e.CLI)]
    for args, expected in [
        (["--cold-replay", str(private)], 0),
        (["--date", "bad"], 1),
        (
            ["--private-control", "--output", str(api.e.ROOT / "results" / (api.e.NAME + ".json"))],
            1,
        ),
    ]:
        result = subprocess.run(command + args, capture_output=True, timeout=120)
        assert result.returncode == expected, result.stderr.decode()
    changed = deepcopy(value)
    changed["resident_update_ready_score"] = 1
    target = tmp_path / "changed.json"
    api.e.atomic_json(target, changed)
    with pytest.raises(ValueError, match="report_drift"):
        api.runtime.replay(target)
    work = api.e.read(value["primitive_reference"])
    shard = Path(work["raw_refs"][0]["path"])
    saved = shard.read_bytes()
    shard.unlink()
    try:
        with pytest.raises(OSError):
            api.runtime.replay(private)
    finally:
        shard.write_bytes(saved)
    primitive = Path(value["primitive_reference"]["path"])
    saved = primitive.read_bytes()
    try:
        work["duration_s"] += 1
        api.e.atomic_json(primitive, work)
        changed = deepcopy(value)
        changed["primitive_reference"] = api.e.reference(primitive)
        api.e.atomic_json(target, changed)
        with pytest.raises(ValueError, match="independent_measurement_seal"):
            api.runtime.replay(target)
    finally:
        primitive.write_bytes(saved)


def test_build_rejects_failed_checks(private):
    """REQ-VERIFY-8731: failed owned checks cannot release performance work."""
    value = json.loads(private.read_bytes())
    work = api.e.read(value["primitive_reference"])
    assert api.build(work, [dict(passed=False)], private)["verdict_class"] == "disqualified"


def test_native_build_and_qualification_failures(tmp_path, monkeypatch):
    """REQ-VERIFY-8731: native checks run and altered upstream primitives block."""
    assert all(r["passed"] for r in api.native_build(tmp_path / "native"))
    with monkeypatch.context() as m:
        m.setattr(api, "base_native_build", lambda path: [dict(passed=False)])
        assert not api.native_build(tmp_path / "failed")[0]["passed"]
    admitted = api.base_inputs()
    monkeypatch.setattr(api, "base_inputs", lambda: deepcopy(admitted))
    real_read, real_gate = api.e.read, api.e.previous.gate
    for mode in ["flag", "primitive", "count", "job", "missing", "changed"]:
        with monkeypatch.context() as m:

            def altered(ref):
                if ref["path"].endswith(api.PRODUCER) and mode == "missing":
                    raise FileNotFoundError(ref["path"])
                if ref["path"].endswith(api.PRODUCER) and mode == "changed":
                    raise ValueError("changed_bytes:" + ref["path"])
                value = real_read(ref)
                if "8719" in ref["path"] and ref["path"].endswith("/measurement.json"):
                    if mode == "primitive":
                        value["private_control"] = True
                    if mode == "count":
                        value["summaries"] = []
                    if mode == "job":
                        value["summaries"][0]["passed"] = False
                return value

            m.setattr(api.e, "read", altered)
            if mode == "flag":
                m.setattr(
                    api.e.previous, "gate", lambda *args: dict(real_gate(*args), passed=False)
                )
            result = api.inputs()
            assert result["failures"]
            assert result["failures"][-1]["path"]
