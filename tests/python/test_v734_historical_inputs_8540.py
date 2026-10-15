"""REQ-REPORT-8540, REQ-VERIFY-8540: original family authority and rejection paths."""

import json
from pathlib import Path

import pytest

from carnot.reporting import v734_contract_methods as e
from carnot.reporting import v734_historical_inputs as h
from carnot.reporting.current_work_receipt import atomic_json


@pytest.mark.parametrize("number", [8529, 8536])
@pytest.mark.parametrize("mode", ["valid", "rejected", "missing"])
def test_family_consumer(tmp_path, monkeypatch, number, mode):
    """SCENARIO-REPORT-8540-METHODS: independent original consumers keep their own gates."""
    monkeypatch.setattr(h.cap, "outcome", lambda *a: dict(eligible=mode != "rejected", checks=[]))
    monkeypatch.setattr(h.cap, "load", lambda *a: {h.FIELDS[number]: 1})
    if mode == "missing":

        def absent(*args):
            raise ValueError("missing_original_input")

        monkeypatch.setattr(h.cap, "outcome", absent)
    row = h.qualify(
        dict(number=number, primary=dict(path="missing", sha256="original"), authority=[], task={}),
        tmp_path,
    )
    assert row["qualified"] == (mode == "valid")
    assert row["field"] == h.FIELDS[number]


def test_worker_and_killed_history(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8540-CONTROLS: worker output and failed child remain evidence."""
    monkeypatch.setattr(h.old, "qualify", lambda *a: dict(qualified=True))
    request, output = tmp_path / "request.json", tmp_path / "output.json"
    atomic_json(request, dict(number=8430, primary={}))
    assert h.worker(request, output) == 0
    assert json.loads(output.read_bytes())["qualified"]
    monkeypatch.setattr(
        h.old, "historical", lambda *a: dict(families=[], receipts=[], references=[])
    )
    monkeypatch.setattr(h, "child", lambda *a, **kw: dict(passed=False, actual_exit=-9))
    (tmp_path / "results").mkdir()
    result = h.historical(tmp_path, tmp_path / "raw", e.CLI, e.protocol())
    assert len(result["families"]) == 2
    assert not any(row["qualified"] for row in result["families"])
    assert all(row["gate_check_summary"][0]["observed"] == -9 for row in result["families"])


def test_successful_history(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8540-METHODS: completed original children retain terminal rows."""
    monkeypatch.setattr(
        h.old, "historical", lambda *a: dict(families=[], receipts=[], references=[])
    )

    def completed(name, argv, logs, **kwargs):
        atomic_json(Path(argv[-1]), dict(qualified=True))
        return dict(passed=True)

    monkeypatch.setattr(h, "child", completed)
    (tmp_path / "results").mkdir()
    result = h.historical(tmp_path, tmp_path / "raw", e.CLI, e.protocol())
    assert all(row["qualified"] for row in result["families"])


@pytest.mark.parametrize("changed", [True, False])
def test_causal_runtime_operand(tmp_path, monkeypatch, changed):
    """SCENARIO-REPORT-8540-METHODS: a context-only change cannot admit the canary."""
    monkeypatch.setattr(
        e,
        "_RUNTIME",
        lambda *a: dict(
            ready=True, failures=[], changes=[dict(changed=changed, field="libraries")]
        ),
    )
    assert e.runtime(tmp_path)["ready"] == changed


def test_original_ready_fields_exist():
    """SCENARIO-REPORT-8540-METHODS: adapters bind actual producer field names."""
    for number in [8529, 8536]:
        primary = next((e.ROOT / "results").glob(f"experiment_{number}_*.json"))
        assert h.FIELDS[number] in json.loads(primary.read_bytes())
