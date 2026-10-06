"""REQ-VERIFY-8193: profile callback statements also need direct measurement."""

import json
import os
import sys

import coverage

from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import learning_qualification_8193 as e


def test_exit_event_handler_without_replacing_child(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8193-CHILD: Python disables tracing inside profile calls.

    The real child test proves exit73 and durable coverage. Calling the installed
    handler as a normal function measures its statements without changing exit.
    """
    rows, labels = e.legacy.fixture("learnable")
    inputs = tmp_path / "input.json"
    atomic_json(inputs, dict(rows=rows, labels=labels, seed=101))
    original = e.legacy.schedule.progress
    observed = []

    def progress(phase, completed=0, pending=0):
        original(phase, completed, pending)
        handler = sys.getprofile()
        assert handler is not None
        handler(sys._getframe(), "c_call", os._exit)
        observed.append(completed)

    monkeypatch.setattr(e.legacy.schedule, "progress", progress)
    previous = sys.getprofile()
    active = coverage.Coverage.current()
    local = None
    if active is None:
        local = coverage.Coverage(data_file=str(tmp_path / ".coverage"), config_file=False)
        local.start()
    try:
        e.seed_child(inputs, tmp_path / "output", None, 0)
    finally:
        if local is not None:
            local.stop()
            local.save()
    assert sys.getprofile() is previous
    assert observed[-1] == 256
    assert json.loads((tmp_path / "output/final.json").read_text())["cursor"] == 257
