"""REQ-REPORT-8552, REQ-VERIFY-8552: isolate original inputs and terminal evidence."""

import json
from pathlib import Path

import pytest

from carnot.reporting import v735_contract_methods as e
from carnot.reporting import v735_historical_inputs as h
from carnot.reporting.current_work_receipt import atomic_json


def test_worker(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8552-CONTROLS: fresh workers use shipped original consumers."""
    monkeypatch.setattr(h.previous, "qualify", lambda *a: dict(qualified=True))
    request, output = tmp_path / "request.json", tmp_path / "output.json"
    atomic_json(request, dict(number=8430, primary={}))
    assert h.worker(request, output) == 0
    assert json.loads(output.read_bytes())["qualified"]


@pytest.mark.parametrize("mode", ["success", "killed", "missing"])
def test_history(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-8552-METHODS: missing originals cannot borrow current authority."""
    if mode != "missing":
        (tmp_path / "results").mkdir()
    monkeypatch.setattr(h.source, "custody", lambda *a: [])

    def run(name, argv, logs, **kwargs):
        if mode == "success":
            atomic_json(
                Path(argv[-1]), dict(experiment_id=int(name.split("_")[-1]), qualified=True)
            )
        return dict(passed=mode == "success", actual_exit=0 if mode == "success" else -9)

    monkeypatch.setattr(h, "child", run)
    result = h.historical(tmp_path, tmp_path / "raw", e.CLI, e.protocol())
    assert len(result["families"]) == 5
    assert all(row["qualified"] == (mode == "success") for row in result["families"])
    if mode != "success":
        assert all(row["gate_check_summary"] for row in result["families"])


@pytest.mark.parametrize("changed", [True, False])
def test_runtime(tmp_path, monkeypatch, changed):
    """SCENARIO-REPORT-8552-METHODS: optional context-only changes do not authorize CUDA."""
    monkeypatch.setattr(
        e.previous,
        "_RUNTIME",
        lambda *a: dict(
            ready=True, failures=[], changes=[dict(changed=changed, field="libraries")]
        ),
    )
    assert e.runtime(tmp_path)["ready"] == changed


def test_release_and_literature():
    """SCENARIO-REPORT-8552-METHODS: HALLMARK labels never join historical learning."""
    p = e.protocol()
    release = p["hallmark_release_admissibility"]
    assert release["registered_before_row_inspection"]
    assert not release["join_existing_learning_stream"]
    assert release["current_row_inspection_count"] == 0
    assert p["literature_mapping_v735"]["HALLMARK_v2"]["experiments"] == [8553]
    assert p["literature_mapping_v735"]["proper_losses_delayed_capacity"]["experiments"] == [
        8555,
        8556,
        8557,
    ]
    assert p["literature_mapping_v735"]["sparse_KAN_operations"]["experiments"] == [
        8558,
        8559,
        8563,
    ]
