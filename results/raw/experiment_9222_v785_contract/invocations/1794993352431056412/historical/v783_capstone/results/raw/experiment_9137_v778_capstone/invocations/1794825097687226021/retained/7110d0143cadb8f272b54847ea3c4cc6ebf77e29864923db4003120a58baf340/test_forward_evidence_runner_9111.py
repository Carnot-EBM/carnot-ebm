"""REQ-REPORT-9111-V777: immutable evidence, actual CLI and cold replay controls."""

from copy import deepcopy
import json
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from carnot.reporting import forward_evidence_runner_9111 as runner
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def measured_work(tmp_path_factory: Any) -> dict[str, Any]:
    """Retain one expensive historical audit and copy it before mutation controls."""
    raw = tmp_path_factory.mktemp("forward_evidence")
    return runner.measure(runner.authenticate(runner.ROOT, raw), raw)


def test_authentication_and_historical_flags(measured_work: dict[str, Any]) -> None:
    """SCENARIO-REPORT-9111-V777: authenticate the measured bytes before dependent work."""
    work = deepcopy(measured_work)
    evidence = work["evidence"]
    assert all(c["passed"] for c in evidence["checks"])
    assert len(work["original_flag_rows"]) == 3
    assert all(r["passed"] for r in work["semantic_control_rows"])
    assert all(r["passed"] for r in work["preserved_detection_rows"])
    assert work["historical_judge_diagnostic"]["judge_ready_score"] == 0
    assert work["historical_judge_diagnostic"]["original_quarantine_preserved"] is True


def test_missing_input_is_terminal_block(tmp_path: Path) -> None:
    """REQ-REPORT-9111-V777: missing external inputs cannot become partial work."""
    evidence = runner.authenticate(tmp_path / "absent", tmp_path / "raw")
    assert evidence["checks"][0]["observed"] == "missing_file"
    assert not evidence["checks"][0]["passed"]
    work = runner.measure(evidence, tmp_path / "raw")
    value = runner.build(work, [], {}, "20261115", 0, 1)
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert value["forward_evidence_ready_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["current_llm_invocations"] == 0


def test_measured_fields_and_rehashed_tamper(tmp_path: Path, measured_work: dict[str, Any]) -> None:
    """REQ-REPORT-9111-V777: changes to derived totals fail replay even if rehashed."""
    work = deepcopy(measured_work)
    value = runner.build(work, [], {}, "20261115", 0, 1)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert runner.replay(path)["passed"]
    altered = deepcopy(value)
    altered["forward_evidence_ready_score"] = 1
    atomic_json(path, altered)
    with pytest.raises(ValueError, match="reduction_drift"):
        runner.replay(path)


def test_real_cli_success_refusal_tamper_and_cold_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-9111-V777: real commands run outside checkout without PYTHONPATH."""
    receipts = runner.cli_controls(tmp_path)
    assert len(receipts) == 5 and all(r["passed"] for r in receipts)
    assert all(r["argv"][:3] == ["env", "-u", "PYTHONPATH"] for r in receipts)
    good = json.loads((tmp_path / "success" / (runner.NAME + ".json")).read_bytes())
    blocked = json.loads((tmp_path / "blocked" / (runner.NAME + ".json")).read_bytes())
    assert good["current_llm_invocations"] == blocked["current_llm_invocations"] == 0
    assert good["forward_evidence_ready_score"] == 0  # A fixture alone cannot qualify validation.
    assert blocked["verdict_class"] == "blocked"


def test_date_and_replay_errors(tmp_path: Path, capsys: Any) -> None:
    """SCENARIO-REPORT-9111-V777: CLI refusals are explicit terminal failures."""
    assert runner.main(["--date", "bad"]) == 1
    path = tmp_path / "bad.json"
    atomic_json(path, {})
    assert runner.main(["--replay", str(path)]) == 1
    assert "refused" in capsys.readouterr().out


@pytest.mark.parametrize("observed", [0, "missing_field", True])
def test_gate_zero_missing_and_boolean_are_distinct(
    tmp_path: Path, monkeypatch: Any, observed: Any
) -> None:
    """REQ-REPORT-9111-V777: observed zero and a missing field are different refusals."""
    contract = {} if observed == "missing_field" else dict(contract_ready_score=observed)
    path = tmp_path / runner.CONTRACT
    atomic_json(path, contract)
    monkeypatch.setitem(runner.PINS, runner.CONTRACT, runner.ref(path)["sha256"])
    evidence = runner.authenticate(tmp_path, tmp_path / "raw")
    assert evidence["checks"][-1]["observed"] == observed
    assert evidence["checks"][-1]["passed"] is False


def test_missing_historical_snapshot_and_changed_bytes(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-9111-V777: no mutable code may replace a missing authenticated snapshot."""
    path = tmp_path / runner.CONTRACT
    atomic_json(path, dict(contract_ready_score=1, authority_snapshots=[]))
    monkeypatch.setitem(runner.PINS, runner.CONTRACT, runner.ref(path)["sha256"])
    evidence = runner.authenticate(tmp_path, tmp_path / "raw")
    assert [c["observed"] for c in evidence["checks"][2:]] == ["missing_field"] * 4
    with pytest.raises(ValueError, match="reference_drift"):
        runner.bound(dict(path=str(path), sha256="sha256:wrong"))
    monkeypatch.setattr(runner, "sha256_file", lambda p: "sha256:changed_during_copy")
    with pytest.raises(ValueError, match="input_changed_during_retention"):
        runner.authenticate(tmp_path, tmp_path / "raced")


def test_reproduction_and_roster_refusal(tmp_path: Path, monkeypatch: Any) -> None:
    """SCENARIO-REPORT-9111-V777: neither a failed reproduction nor a changed roster passes."""
    evidence = runner.authenticate(runner.ROOT, tmp_path)
    with monkeypatch.context() as m:
        m.setattr(runner, "historical_report", lambda e: dict(flags=[]))
        with pytest.raises(ValueError, match="original_flag_reproduction"):
            runner.measure(evidence, tmp_path)
    evidence["roster_rows"].pop()
    with pytest.raises(ValueError, match="historical_roster_drift"):
        runner.measure(evidence, tmp_path)


def test_readiness_reducer_requires_every_control(
    measured_work: dict[str, Any], monkeypatch: Any
) -> None:
    """REQ-REPORT-9111-V777: even a qualified validation cannot rescue a failed control."""
    work = deepcopy(measured_work)
    monkeypatch.setattr(runner, "validation_qualified", lambda *args: True)
    value = runner.build(work, [], {}, "20261115", 0, 1)
    assert value["forward_evidence_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    work["semantic_control_rows"][0]["passed"] = False
    assert runner.build(work, [], {}, "20261115", 0, 1)["forward_evidence_ready_score"] == 0


def test_actual_coverage_data_and_receipt_refusals(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-9111-V777: real SQLite data with uncovered statements cannot qualify."""
    from coverage import Coverage

    scope = runner.plan(tmp_path)
    evidence = runner.authenticate(runner.ROOT, tmp_path)
    tracer = Coverage(config_file=False, data_file=str(tmp_path / ".coverage"), data_suffix=True)
    tracer.start()
    runner.semantic_control_rows()
    tracer.stop()
    tracer.save()
    report = runner.statement_coverage(tmp_path, evidence)
    assert report["excluded_statements"] == 0 and report["statement_percent"] < 100
    receipt = runner.child(
        "tiny",
        [str(runner.ROOT / ".venv/bin/python"), "-c", "print('real receipt')"],
        tmp_path / "tiny",
    )
    scope["commands"] = [dict(name="tiny", argv=receipt["argv"], expected=0)]
    scope["cli_commands"] = []
    monkeypatch.setattr(runner, "validation_commands", lambda p: scope["commands"])
    monkeypatch.setattr(runner, "cli_plan", lambda p: [])
    work = dict(scope=scope, evidence=evidence)
    assert runner.validation_qualified(work, [receipt], report) is False
    for key, bad in [("passed", False), ("normal_exit", False), ("timed_out", True)]:
        changed = dict(receipt, **{key: bad})
        assert runner.validation_qualified(work, [changed], report) is False
    assert runner.validation_qualified(work, [], report) is False
    altered = deepcopy(report)
    altered["data_reference"]["sha256"] = "sha256:wrong"
    assert runner.validation_qualified(work, [receipt], altered) is False


def test_validation_plan_cannot_drop_or_replace_a_command(tmp_path: Path) -> None:
    """REQ-REPORT-9111-V777: a rehashed scope cannot replace owned validation with an empty list."""
    scope = runner.plan(tmp_path)
    assert scope["commands"] == runner.validation_commands(tmp_path)
    scope["commands"] = []
    assert runner.validation_qualified(dict(scope=scope), [], {}) is False


def test_replay_authenticates_actual_validation_logs(tmp_path: Path) -> None:
    """REQ-REPORT-9111-V777: changed command logs invalidate an otherwise unchanged replay."""
    evidence = runner.authenticate(tmp_path / "absent", tmp_path / "raw")
    work = runner.measure(evidence, tmp_path / "raw")
    receipt = runner.child(
        "real_log",
        [str(runner.ROOT / ".venv/bin/python"), "-c", "print('durable log')"],
        tmp_path / "logs",
    )
    value = runner.build(work, [receipt], {}, "20261115", 0, 1)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert runner.replay(path)["passed"]
    Path(receipt["stdout_path"]).write_text("changed log")
    with pytest.raises(ValueError, match="reference_drift"):
        runner.replay(path)


def test_normal_cli_blocked_path(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-REPORT-9111-V777: normal mode publishes a terminal external block after validation."""
    health_started = Event()

    def execute(*args: Any) -> list[Any]:
        assert health_started.wait(5), "repository health must run alongside scoped validation"
        return []

    monkeypatch.setattr(runner, "execute", execute)
    monkeypatch.setattr(runner, "statement_coverage", lambda *args: {})
    monkeypatch.setattr(runner, "cli_controls", lambda *args: [])
    real_child = runner.child

    def child(name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if name == "repository_health":
            health_started.set()
            return dict(passed=False)
        return real_child(name, *args, **kwargs)

    monkeypatch.setattr(runner, "child", child)
    path = tmp_path / (runner.NAME + ".json")
    assert runner.main(["--input-root", str(tmp_path / "missing"), "--output", str(path)]) == 0
    value = json.loads(path.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["forward_evidence_ready_score"] == 0


@pytest.mark.parametrize(
    "change",
    [
        "checksum",
        "derivation",
        "flags",
        "roster",
        "diagnostic_input",
        "diagnostic_report",
        "structural_control",
    ],
)
def test_rehashed_primitive_tampering(
    tmp_path: Path, measured_work: dict[str, Any], change: str
) -> None:
    """SCENARIO-REPORT-9111-V777: rehashing invented derived evidence does not authorize it."""
    work = deepcopy(measured_work)
    if change == "derivation":
        work["structural_count_derivations"]["half_denominator"] += 1
    elif change == "flags":
        work["original_flag_rows"][0]["detail"] = "fabricated finding"
    elif change == "roster":
        work["evidence"]["roster_rows"].pop()
    elif change == "diagnostic_input":
        original = json.loads(runner.bound(work["diagnostic_input_reference"]).read_bytes())
        original["effective_independent_groups"] += 1
        path = tmp_path / "invented.json"
        atomic_json(path, original)
        work["diagnostic_input_reference"] = runner.ref(path)
    elif change == "diagnostic_report":
        work["historical_judge_diagnostic"]["repaired_reader_report"]["flag_count"] = 1
    elif change == "structural_control":
        work["preserved_detection_rows"][0]["flags"][0]["detail"] = "fabricated flag"
    value = runner.build(work, [], {}, "20261115", 0, 1)
    if change == "checksum":
        value["reproducibility_checksum"] = "sha256:wrong"
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="drift|replay_hash"):
        runner.replay(path)
