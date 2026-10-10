"""REQ-VERIFY-8373: worker memory bounds belong to a fresh authentication process."""

import json

import pytest

from carnot.reporting import v721_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def worker_request(tmp_path):
    """Use the authentic blocked terminal; no science evidence is synthesized."""
    task = e.contract.authority(e.ROOT, tmp_path / "authority")["tasks"][12]
    primary = e.freeze(e.ROOT / task["deliverable"], tmp_path / "primary")
    summary = e.outcome(task, primary, tmp_path / "capture")
    request, output = tmp_path / "request.json", tmp_path / "summary.json"
    atomic_json(
        request,
        dict(
            task=task, task_sha256=canonical_hash(task), primary=primary, closure=summary["closure"]
        ),
    )
    return request, output


def test_embedded_worker_ignores_unrelated_host_memory_peak(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8373-REPLAY: valid input survives an earlier 2139 MiB host peak."""
    request, output = worker_request(tmp_path)
    monkeypatch.setattr(e, "memory", lambda: dict(current_rss_mb=629, peak_rss_mb=2139))
    assert e.worker(request, output) == 0
    result = json.loads(output.read_bytes())
    assert result["honest_verdict"] == "complete_blocked_gatemate_missing_evidence"
    assert result["branch_replay"]["status"] == "terminal_disposition_preserved"
    assert result["memory"]["passed"]
    assert result["memory"]["after"]["peak_rss_mb"] <= 1500


@pytest.mark.parametrize(
    "before,after,exit_code",
    [(1000, 1500, 0), (1500, 1501, 1), (500, 1001, 1)],
)
def test_child_enforces_absolute_and_growth_memory_bounds(
    tmp_path, monkeypatch, before, after, exit_code
):
    """REQ-VERIFY-8373: either exceeded child bound fails; exact limits still pass."""
    request, output = worker_request(tmp_path)
    samples = iter(
        [
            dict(current_rss_mb=before, peak_rss_mb=before),
            dict(current_rss_mb=after, peak_rss_mb=after),
        ]
    )
    monkeypatch.setattr(e, "memory", lambda: next(samples))
    assert e.worker_process(request, output) == exit_code
    memory = json.loads(output.read_bytes())["memory"]
    assert memory["peak_bound_mb"] == 1500
    assert memory["growth_bound_mb"] == 500
    assert memory["growth_mb"] == after - before
    assert memory["passed"] is (exit_code == 0)
