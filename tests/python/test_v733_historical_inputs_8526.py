"""REQ-REPORT-8526, REQ-VERIFY-8526: original family authority and rejection paths."""

import json
from pathlib import Path

import pytest

from carnot.reporting import v733_contract_methods as e
from carnot.reporting import v733_historical_inputs as h
from carnot.reporting.current_work_receipt import atomic_json


@pytest.mark.parametrize("number", [8486, 8487, 8516, 8520])
@pytest.mark.parametrize("mode", ["valid", "pin", "authority", "readiness", "missing"])
def test_family_consumer(tmp_path, monkeypatch, number, mode):
    """SCENARIO-REPORT-8526-METHODS: each original consumer keeps its own gates."""
    p = e.protocol()
    cap = h.v730 if number < 8512 else h.v732
    family = "v730_authority_references" if number < 8512 else "v732_authority_references"
    authority = [cap.freeze(Path(ref["path"]), tmp_path / "authority") for ref in p[family]]
    primary = cap.freeze(
        next((e.ROOT / "results").glob(f"experiment_{number}_*.json")), tmp_path / "primary"
    )
    args = dict(number=number, primary=primary, authority=authority)
    monkeypatch.setattr(cap, "outcome", lambda *a: dict(eligible=mode != "readiness"))
    if mode == "pin":
        primary["sha256"] = "changed"
    if mode == "authority":
        authority[0]["sha256"] = "changed"
    if mode == "missing":
        primary["snapshot_path"] = str(tmp_path / "missing.json")
    row = h.qualify(args, tmp_path / "scratch")
    assert row["qualified"] == (mode == "valid")
    assert row["field"] == h.FIELDS[number]
    if mode != "valid":
        assert row["gate_check_summary"][0]["observed"] == (1 if mode == "readiness" else None)
        assert row["gate_check_summary"][0]["upstream_id"] == f"exp{number}"


def test_worker_and_killed_history(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8526-CONTROLS: worker output and failed child remain evidence."""
    monkeypatch.setattr(h.old, "qualify", lambda *a: dict(qualified=True))
    request, output = tmp_path / "request.json", tmp_path / "output.json"
    atomic_json(request, dict(number=8430, primary={}))
    assert h.worker(request, output) == 0
    assert json.loads(output.read_bytes())["qualified"]
    monkeypatch.setattr(
        h.old, "historical", lambda *a: dict(families=[], receipts=[], references=[])
    )
    monkeypatch.setattr(h, "child", lambda *a, **kw: dict(passed=False, actual_exit=-9))
    missing = tmp_path / "missing"
    (missing / "results").mkdir(parents=True)
    result = h.historical(missing, tmp_path / "raw", e.CLI, e.protocol())
    assert len(result["families"]) == 4
    assert not any(row["qualified"] for row in result["families"])
    assert all(row["gate_check_summary"][0]["observed"] == -9 for row in result["families"])
