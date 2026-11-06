"""REQ-REPORT-8860: sealed reconstruction, joint gates and private replay."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from carnot.reporting import linear_benefit_audit_8860 as r
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    """Preserved real events prevent fixtures from inventing scientific evidence."""
    upstream = r.load(dict(path=str(r.SOURCE), sha256=r.PIN))
    return r.load(upstream["raw_shard_hashes"][0])


@pytest.fixture(scope="module")
def result(source: dict[str, Any]) -> dict[str, Any]:
    """Reduce the full paired roster once without evaluating any gradient."""
    with patch.object(r.a.c, "update", side_effect=AssertionError("no training")):
        return r.reduce(source)


def test_full_reconstruction(result: dict[str, Any]) -> None:
    """SCENARIO-REPORT-8860-CONTRACT: every claim reduces from scheduled events."""
    assert len(result["rows"]) == result["completed_count"] == 27648
    assert len(result["causal_integrity_rows"]) == len(result["per_stream_results"]) == 144
    assert result["independent_count"] == 24
    assert len(result["paired_interval_rows"]) == 6
    assert len(result["retention_interval_rows"]) == 4
    assert len(result["movement_intervals"]) == 288
    assert result["delay_subgroups"]
    assert all(
        x["draws"] == 10000 and x["tail_probability"] == 0.01 / 12
        for x in result["paired_interval_rows"]
    )
    assert all(x["feedback_ordinal"] < x["use_ordinal"] for x in result["template_reuse_rows"])
    assert result["learning_audit_ready_score"] == 1
    assert result["causal_benefit_score"] <= result["retention_pass_score"]


@pytest.mark.parametrize(
    "mutation",
    [
        "roster",
        "seal",
        "truth",
        "release",
        "weights",
        "restart",
        "retention",
        "witness",
        "reuse",
        "movement",
        "final",
    ],
)
def test_reject_events(source: dict[str, Any], mutation: str) -> None:
    """REQ-REPORT-8860: reconstructed identities reject forged event summaries."""
    stream = deepcopy(source["stream_results"][0])
    if mutation == "roster":
        stream["rows"].pop()
    elif mutation == "seal":
        stream["rows"][0]["sealed"]["state_hash"] = "changed"
    elif mutation == "truth":
        stream["rows"][0]["truth"] = "support"
    elif mutation == "release":
        stream["feedback_rows"][0]["feedback_ordinal"] -= 1
    elif mutation == "weights":
        stream["feedback_rows"][0]["parameters_before"] = {}
    elif mutation == "restart":
        stream["restart_rows"][0]["after"] = "changed"
    elif mutation == "retention":
        stream["retention_rows"][0]["state_hash"] = "changed"
    elif mutation == "witness":
        next(x for x in stream["feedback_rows"] if x["template_added"])["addition_witness"] = {}
    elif mutation == "reuse":
        stream["template_reuse_rows"][0]["provenance"] = {}
    elif mutation == "movement":
        stream["movement_rows"][0]["template_admissions"] = 99
    else:
        stream["final_state_refs"][0]["sha256"] = "changed"
    with pytest.raises((ValueError, KeyError)):
        r.reconstruct_stream(stream, source["initialization_rows"])


def test_intervals_joint_gates() -> None:
    """REQ-REPORT-8860: retention failure vetoes future success; margins stay fixed."""
    intervals = r.intervals(np.full((24, 6), -0.02), [str(i) for i in range(6)], 0.01 / 12)
    retention = [dict(upper=0.01) for _ in range(4)]
    assert r.gates(intervals, retention)["causal_benefit_score"] == 1
    intervals[1]["upper"] = -0.004
    assert r.gates(intervals, retention)["addition_benefit_score"] == 0
    intervals[1]["upper"] = -0.005
    intervals[2]["upper"] = 0
    assert r.gates(intervals, retention)["causal_benefit_score"] == 0
    intervals[2]["upper"] = -0.001
    retention[3]["upper"] = 0.010001
    gates = r.gates(intervals, retention)
    assert gates["causal_benefit_score"] == gates["retention_pass_score"] == 0
    assert gates["addition_benefit_score"] == 1


def test_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8860: external bytes and writable private resources precede outcomes."""
    work, pre, refs = r.preflight(tmp_path / "private", r.SOURCE)
    assert len(work["stream_results"]) == 24 and refs and pre["time_budget_s"] == 4200
    with pytest.raises(FileNotFoundError):
        r.preflight(tmp_path / "missing", tmp_path / "absent")
    with patch.object(r.sys, "prefix", "/tmp/wrong"), pytest.raises(RuntimeError):
        r.preflight(tmp_path / "env", r.SOURCE)
    monkeypatch.setattr(r, "load", lambda _: dict(stream_rows_ready_score=0))
    with pytest.raises(PermissionError):
        r.preflight(tmp_path / "zero", r.SOURCE)


def test_cold_replay(
    result: dict[str, Any],
    source: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-8860-CONTRACT: outside-checkout replay rejects rehashed claims."""
    raw = tmp_path / "evidence.json"
    atomic_json(raw, source)
    value = dict(
        deepcopy(result),
        verdict_class="null",
        acceptance_gates=r.a.GATES,
        raw_shard_hashes=[r.reference(raw)],
        source_artifact_hashes=[],
        code_config_hashes=[],
        validation_receipts=[],
    )
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(r.ROOT / r.CLI), "--replay", str(candidate)]
    child = subprocess.run(
        command, cwd="/tmp", env=env, capture_output=True, text=True, timeout=180
    )
    assert child.returncode == 0 and '"replay_passed": true' in child.stdout, (
        child.stdout + child.stderr
    )
    monkeypatch.setattr(r, "reduce", lambda _: deepcopy(result))
    value["paired_interval_rows"][0]["upper"] = -99
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(candidate)
    value["raw_shard_hashes"][0]["sha256"] = "changed"
    atomic_json(candidate, value)
    with pytest.raises(ValueError):
        r.replay(candidate)


@pytest.mark.parametrize(
    "failure",
    [
        "",
        "null",
        "missing",
        "gate",
        "rejected",
        "field",
        "integrity",
        "owned",
        "terminal",
        "deadline",
        "partial",
    ],
)
def test_runner_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """REQ-REPORT-8860: external absence blocks, owned failures disqualify readiness."""
    from carnot.reporting import linear_benefit_runner_8860 as runner

    minimal = r.empty("fixture")
    minimal.update(learning_audit_ready_score=1, causal_benefit_score=int(failure != "null"))

    def preflight(raw: Path, path: Path) -> Any:
        if failure in {"missing", "gate", "rejected", "field", "integrity", "partial"}:
            raise {
                "missing": FileNotFoundError("absent"),
                "gate": PermissionError("observed_zero"),
                "rejected": ValueError("rejected_producer"),
                "field": KeyError("authority"),
                "integrity": RuntimeError("internal_resources"),
                "partial": TimeoutError("deadline"),
            }[failure]
        return {}, {}, []

    monkeypatch.setattr(r, "preflight", preflight)
    monkeypatch.setattr(r, "reduce", lambda _: deepcopy(minimal))
    monkeypatch.setattr(
        runner, "validate", lambda _: dict(passed=failure != "owned", receipts=[], coverage={})
    )
    monkeypatch.setattr(
        runner, "terminal", lambda *a: dict(passed=failure != "terminal", receipts=[])
    )
    output = tmp_path / (r.NAME + ".json")
    if failure == "deadline":
        monkeypatch.setattr(runner, "BUDGET", -1)
    assert runner.main(["--private-output", str(output), "--raw-dir", str(tmp_path / "raw")]) == 0
    value = json.loads(output.read_bytes())
    expected = (
        "blocked"
        if failure in {"missing", "gate", "rejected", "field"}
        else "partial"
        if failure == "partial"
        else "null"
        if failure == "null"
        else "disqualified"
        if failure
        else "circular_positive"
    )
    assert value["verdict_class"] == expected and value["honest_verdict"].startswith("complete_")
    if failure not in {"", "null"}:
        assert value["learning_audit_ready_score"] == 0
    assert runner.main(["--date", "wrong"]) == 1
    assert runner.main(["--replay", str(tmp_path / "absent")]) == 1
    monkeypatch.setattr(r, "replay", lambda _: dict(passed=True))
    assert runner.main(["--replay", str(output)]) == 0


def test_runner_delegates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8860: retain the existing bounded validation supervisor unchanged."""
    from carnot.reporting import linear_benefit_runner_8860 as runner

    monkeypatch.setattr(runner.validation, "validate", lambda _: dict(passed=True))
    monkeypatch.setattr(runner.validation, "terminal", lambda *a: dict(passed=True))
    assert runner.validate(tmp_path)["passed"]
    assert runner.terminal(tmp_path / "candidate", tmp_path)["passed"]


def test_cli_errors(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8860-CONTRACT: the direct CLI imports without PYTHONPATH."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for args, marker in [
        (["--date", "wrong"], "date_rejected"),
        (["--replay", str(tmp_path / "absent")], "replay_rejected"),
    ]:
        child = subprocess.run(
            [sys.executable, "-u", str(r.ROOT / r.CLI), *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert child.returncode == 1 and marker in child.stdout


def test_replay_blocked_and_receipts(tmp_path: Path) -> None:
    """REQ-REPORT-8860: blocked census and executed logs remain replayable operands."""
    raw, log = tmp_path / "empty.json", tmp_path / "check.log"
    atomic_json(raw, {})
    log.write_text("actual owned test log\n")
    value = dict(
        r.empty("missing_file"),
        verdict_class="blocked",
        acceptance_gates=r.a.GATES,
        raw_shard_hashes=[r.reference(raw)],
        source_artifact_hashes=[],
        code_config_hashes=[],
        validation_receipts=[dict(log_path=str(log), log_sha256=r.reference(log)["sha256"])],
    )
    candidate = tmp_path / "blocked.json"
    atomic_json(candidate, value)
    assert r.replay(candidate)["passed"]
    log.write_text("changed")
    with pytest.raises(ValueError, match="validation_log"):
        r.replay(candidate)
    value["validation_receipts"] = []
    value["acceptance_gates"] = {}
    atomic_json(candidate, value)
    with pytest.raises(ValueError, match="acceptance_drift"):
        r.replay(candidate)


def test_owned_coverage_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8860: instrumented children have a finite budget without extending global health."""
    from carnot.reporting import linear_benefit_runner_8860 as runner

    def commands(plan: list[Any], logs: Path) -> list[Any]:
        assert plan[0][3] == 600 and plan[1][3] == 180
        return []

    def validate(raw: Path) -> dict[str, Any]:
        runner.validation.commands(
            [
                ("owned_pytest", [], "owned", 300),
                ("global_python_health", [], "repository_health", 180),
            ],
            raw,
        )
        return dict(passed=True)

    monkeypatch.setattr(runner.validation, "commands", commands)
    monkeypatch.setattr(runner.validation, "validate", validate)
    assert runner.validate(tmp_path)["passed"]
