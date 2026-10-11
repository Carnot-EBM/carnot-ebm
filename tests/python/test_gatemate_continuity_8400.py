"""REQ-REPORT-8400 / REQ-VERIFY-8400: documentation cannot manufacture device evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import pytest

from carnot.reporting import gatemate_continuity_8400 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child


def test_hard_deadline_restores_timer() -> None:
    """SCENARIO-VERIFY-8400-CONTROLS: a stalled receipt cannot outlive its bounded call."""
    import signal

    with pytest.raises(TimeoutError, match="invocation_deadline"):
        with e.budget(0.01):
            time.sleep(0.1)
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0
    with e.budget(1):
        with e.budget(0.5):
            assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 0.5
        assert 0 < signal.getitimer(signal.ITIMER_REAL)[0] <= 1


def candidate(work: dict, raw: Path, output: Path, passed: bool = True) -> dict:
    """SCENARIO-REPORT-8400-REPLAY: each private candidate binds its actual work bytes."""
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [dict(passed=passed, scope="owned")], raw, output)
    atomic_json(output, value)
    return value


def test_continuity_and_rehashed_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8400-CONTINUITY: six absent units remain externally blocked."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    value = candidate(work, raw, output)
    assert value["experiment_id"] == 8400 and value["run_date"] == "20261011"
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert value["obligation_recorded_score"] == value["current_contract_ready_score"] == 1
    assert value["intended_count"] == 7 and value["completed_count"] == 1
    assert value["censored_count"] == 6 and value["independent_count"] == 0
    assert value["exact_missing_hashes"] == e.old.old.MISSING
    assert value["next_evidence_conditions"] == e.old.CONDITIONS
    assert value["device_command_count"] == value["execution_ready_score"] == 0
    assert value["original_idcode"] == "0xffffffff"
    assert value["required_idcode"] == "0x20000001"
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert value["evidence_delta_scan_limit_s"] == 60
    assert value["repository_health"]["scope"] == "global"
    assert e.replay(output)
    for field, replacement in [
        ("current_contract_ready_score", 0),
        ("execution_ready_score", 1),
        ("exact_missing_hashes", []),
        ("rows", []),
        ("MODEL_SPECS", [{}]),
    ]:
        changed = deepcopy(value)
        changed[field] = replacement
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(output, changed)
        assert not e.replay(output)
    assert not e.replay(tmp_path / "missing")
    assert candidate(work, raw, output, False)["verdict_class"] == "disqualified"
    missing = e.measure(tmp_path / "absent-root", raw)
    absent = candidate(missing, raw, output)
    assert absent["verdict_class"] == "blocked" and absent["obligation_recorded_score"] == 0
    assert e.replay(output)


def receipt(path: Path) -> None:
    """SCENARIO-REPORT-8400-CONTINUITY: current receipts queue only changed physical setup."""
    atomic_json(
        path,
        dict(
            schema=e.old.old.RECEIPT_SCHEMA,
            upstream_primary_sha256=e.old.old.HISTORY_PIN,
            exp8372_primary_sha256=e.old.HISTORY_PIN,
            received_date="20261011",
            received_wall_ns=e.old.CUTOFF_NS + 1,
            authorized_location="operator_physical_receipt",
            source_rows=[],
            physical_change=dict(
                date="20261011",
                operator="private operator",
                changed_fields=["cable", "power"],
                original_transcript_sha256=e.old.old.TRANSCRIPT_PIN,
            ),
        ),
    )


def test_receipt_scope_and_deadline(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8400-CONTINUITY: preserve real dates while reusing the source consumer."""
    raw, path, output = tmp_path / "raw", tmp_path / "receipt.json", tmp_path / (e.NAME + ".json")
    receipt(path)
    work = e.measure(e.ROOT, raw, [path])
    value = candidate(work, raw, output)
    assert value["supplied_evidence_delta"][0]["could_reopen"] == ["physical_change"]
    assert value["physical_change_receipt"]["date"] == "20261011"
    assert value["completed_count"] == 2 and e.replay(output)
    document = json.loads(path.read_bytes())
    document["physical_change"]["changed_fields"] = ["software"]
    atomic_json(path, document)
    assert not e.scan([path], raw)[0]["authenticated"]
    document["received_date"] = "20261012"
    atomic_json(path, document)
    assert not e.scan([path], raw)[0]["authenticated"]
    receipt(path)
    document = json.loads(path.read_bytes())
    document["received_wall_ns"] = e.old.CUTOFF_NS
    atomic_json(path, document)
    assert not e.scan([path], raw)[0]["authenticated"]
    assert not e.scan([tmp_path / "missing"], raw)[0]["authenticated"]
    with patch.object(e.time, "monotonic", side_effect=[0, 61]):
        with pytest.raises(TimeoutError, match="delta_scan_deadline"):
            e.scan([path], raw)


def test_source_positive_and_upstream_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8400-REPLAY: matching source bytes do not reopen device execution."""
    raw, path, output = tmp_path / "raw", tmp_path / "receipt.json", tmp_path / (e.NAME + ".json")
    source = tmp_path / "source"
    source.write_bytes(b"private exact historical source")
    missing = [dict(path="private-source.py", sha256=e.base.sha256_file(source))]
    receipt(path)
    document = json.loads(path.read_bytes())
    document.update(
        physical_change=None,
        source_rows=[
            dict(original_path="private-source.py", path=str(source), sha256=missing[0]["sha256"])
        ],
    )
    atomic_json(path, document)
    with patch.object(e.old.old, "MISSING", missing):
        work = e.measure(tmp_path / "absent-root", raw, [path])
        value = candidate(work, raw, output)
        assert value["supplied_evidence_delta"][0]["could_reopen"] == [
            "source_custody:private-source.py"
        ]
        assert value["physical_change_receipt"] is None and e.replay(output)
        source.write_bytes(b"current replacement source")
        assert not e.scan([path], raw)[0]["authenticated"]
    work = e.measure(e.ROOT, raw)
    row = next(r for r in work["inputs"] if r["name"] == e.UPSTREAM)
    altered = tmp_path / "altered.json"
    value = json.loads(e.base.checked(row["reference"]).read_bytes())
    value["original_idcode"] = "invented"
    atomic_json(altered, value)
    row["reference"].update(e.base.reference(altered))
    row["observed_sha256"] = row["reference"]["sha256"]
    with patch.dict(e.PINS, {e.UPSTREAM: row["observed_sha256"]}):
        with pytest.raises(ValueError, match="upstream_obligation_contract"):
            candidate(work, raw, output)


def test_real_cli_and_owned_failures(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8400-CONTROLS: actual children and readers qualify exact private bytes."""
    from carnot.reporting import gatemate_continuity_runner_8400 as runner

    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    ran = child(
        "private_cli",
        [*cli, "--private-e2e", "--output", str(output)],
        tmp_path / "logs",
        deadline=120,
    )
    assert ran["passed"], Path(ran["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert e.replay(output)
    selected = reader_receipt(e.TASK, tmp_path, field="obligation_recorded_score", expected=1)
    assert selected["passed"] and selected["gate_sha256"] == e.base.sha256_file(output)
    for name, args, expected in [
        ("cold", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("date", ["--date", "20261010"], 2),
        ("unsafe", ["--private-e2e"], 2),
    ]:
        assert child(name, [*cli, *args], tmp_path / "logs", expected=expected, deadline=60)[
            "passed"
        ]
    plan = runner.manifest(tmp_path)
    assert any(p["name"] == "private_E2E018_consumers" for p in plan)
    assert not any(p["scope"] == "global" for p in plan)
    private = tmp_path / "scratch"
    private.mkdir()
    atomic_json(private / "coverage.json", dict(private_control=True))
    failed = child(
        "error",
        [sys.executable, "-u", "-c", "raise SystemExit(7)"],
        tmp_path / "error",
        deadline=10,
    )
    with patch.object(runner, "execute", return_value=[failed]):
        bad = tmp_path / "bad" / (e.NAME + ".json")
        assert runner.run(e.ROOT, bad, private, control=True) == 0
        assert json.loads(bad.read_bytes())["verdict_class"] == "disqualified"
    with patch.object(
        runner.qualified, "preflight", return_value=[e.gate(tmp_path, "tool", True, None)]
    ):
        bad = tmp_path / "absent" / (e.NAME + ".json")
        assert runner.run(tmp_path / "missing", bad, private, control=True) == 0
        assert json.loads(bad.read_bytes())["verdict_class"] == "disqualified"
