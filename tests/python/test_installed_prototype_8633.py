"""REQ-REPORT-8633 / SCENARIO-REPORT-8633-INSTALL: retain actual package evidence."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import installed_lean_8633 as e


def test_authenticated_inputs_and_rejection(tmp_path, monkeypatch):
    """REQ-REPORT-8633: original producers supply head authority, not fixture weights."""
    bundle = e.inputs(e.ROOT)
    assert len(bundle["requests"]) == 64
    assert bundle["head"]["parameter_count"] == 109
    assert bundle["target_model"]["original_producer"] == 8431
    with pytest.raises((OSError, ValueError)):
        e.inputs(tmp_path)
    ref = (
        e.reference(tmp_path / "missing")
        if False
        else {"path": str(tmp_path / "missing"), "sha256": "x"}
    )
    with pytest.raises(OSError):
        e.load(ref)
    path = tmp_path / "changed.json"
    path.write_text("{}")
    with pytest.raises(ValueError):
        e.load(dict(path=str(path), sha256="bad"))
    original = e.sha256_file
    checkpoint = bundle["target_model"]["checkpoint"]["path"]
    monkeypatch.setattr(e, "sha256_file", lambda p: "bad" if str(p) == checkpoint else original(p))
    with pytest.raises(ValueError, match="original_operand_hash"):
        e.inputs(e.ROOT)


def test_original_reduction_and_tamper(tmp_path):
    """REQ-REPORT-8633: reductions reject internally consistent rewritten evidence."""
    bundle = e.inputs(e.ROOT)
    rows = [
        dict(
            unit_id=r["unit_id"],
            arm=arm,
            response=bundle["references"][i],
            baseline_response=bundle["references"][i],
            status="ok",
            parity=True,
        )
        for arm in ["python", "native"]
        for i, r in enumerate(bundle["requests"])
    ]
    assert e.reduce_rows(rows, bundle)["completed_count"] == 128
    for mutation in ["count", "response", "identity", "arm", "baseline", "status"]:
        bad = deepcopy(rows)
        if mutation == "count":
            bad.pop()
        elif mutation == "response":
            bad[0]["response"]["decision"] = "bad"
        elif mutation == "identity":
            bad[0]["unit_id"] = "bad"
        elif mutation == "arm":
            bad[0]["arm"] = "bad"
        elif mutation == "baseline":
            bad[0]["baseline_response"] = {}
        else:
            bad[0]["status"] = "failed"
        with pytest.raises(ValueError):
            e.reduce_rows(bad, bundle)


def test_real_package_sessions(tmp_path):
    """SCENARIO-REPORT-8633-INSTALL: actual wheel children keep the original64 inputs."""
    bundle = e.inputs(e.ROOT)
    evidence = e.measure(tmp_path, bundle)
    assert len(evidence["prototype_parity_rows"]) == 128
    assert all(r["parity"] for r in evidence["prototype_parity_rows"])
    assert all(r["passed"] for r in evidence["export_contract_rows"])
    assert all(r["clean"] for r in evidence["import_trace_rows"])
    assert Path(evidence["wheel_reference"]["path"]).is_file()
