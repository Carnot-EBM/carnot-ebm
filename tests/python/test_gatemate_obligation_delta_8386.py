"""REQ-REPORT-8386 / REQ-VERIFY-8386: custody and physical obligations stay separate."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import gatemate_obligation_delta_8386 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


def candidate(work: dict, raw: Path, output: Path, passed: bool = True) -> dict:
    """SCENARIO-REPORT-8386-CUSTODY: private candidates bind actual primitive bytes."""
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [dict(passed=passed, scope="owned")], raw, output)
    atomic_json(output, value)
    return value


def test_natural_delta_and_rehashed_claims(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8386-DELTA: six missing units cannot become positive observations."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    value = candidate(work, raw, output)
    assert value["verdict_class"] == "blocked" and value["obligation_recorded"]
    assert value["history_authenticated"] is False
    assert value["exact_missing_hashes"] == e.old.MISSING
    assert value["intended_count"] == 7 and value["completed_count"] == 1
    assert value["censored_count"] == 6 and value["independent_count"] == 0
    assert value["device_command_count"] == 0 and value["MODEL_SPECS"] == []
    assert len(value["next_evidence_conditions"]) == 4
    assert all(r["missing_reason"] for r in value["rows"] if r["censored"])
    assert all(
        c["observed"] is None
        for c in value["gate_check_summary"]
        if c["check"].startswith("future.")
    )
    assert e.replay(output)
    for field, replacement in [
        ("device_command_count", 1),
        ("history_authenticated", True),
        ("exact_missing_hashes", []),
        ("obligation_recorded", False),
        ("MODEL_SPECS", [{}]),
        ("rows", []),
    ]:
        changed = deepcopy(value)
        changed[field] = replacement
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(output, changed)
        assert not e.replay(output), field
    assert not e.replay(tmp_path / "missing")
    assert candidate(work, raw, output, False)["verdict_class"] == "disqualified"


def test_missing_and_substituted_inputs(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8386-CUSTODY: missing and wrong historical bytes are both explicit."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(tmp_path / "absent-root", raw)
    value = candidate(work, raw, output)
    assert not value["obligation_recorded"] and e.replay(output)
    natural = e.measure(e.ROOT, raw)
    valid = candidate(natural, raw, output)
    for name in [e.HISTORY, e.old.TRANSCRIPT, e.ACTIVE, e.DESIGN]:
        changed = deepcopy(natural)
        row = next(r for r in changed["inputs"] if r["name"] == name)
        false = tmp_path / "false-input"
        false.write_text("{}")
        row["reference"].update(e.base.reference(false))
        with pytest.raises(ValueError, match="input_observation_hash"):
            candidate(changed, raw, output)
        bad = deepcopy(valid)
        bad["work_reference"] = e.base.reference(raw / "measurement.json")
        bad["raw_shard_hashes"] = [bad["work_reference"]]
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(output, bad)
        assert not e.replay(output)
    changed = deepcopy(natural)
    changed["code_config_hashes"][0]["sha256"] = "wrong"
    candidate(changed, raw, output)
    assert not e.replay(output)


def test_private_source_reader_and_reopening_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8386-DELTA: exact private bytes reopen only source custody."""
    source = tmp_path / "source"
    source.write_bytes(b"private positive consumer bytes")
    missing = [dict(path="private-producer.py", sha256=e.base.sha256_file(source))]
    path, raw, output = tmp_path / "receipt.json", tmp_path / "raw", tmp_path / (e.NAME + ".json")
    receipt(path)
    document = json.loads(path.read_bytes())
    document["physical_change"] = None
    document["source_rows"] = [
        dict(original_path=missing[0]["path"], path=str(source), sha256=missing[0]["sha256"])
    ]
    atomic_json(path, document)
    with patch.object(e.old, "MISSING", missing):
        work = e.measure(tmp_path / "absent-root", raw, [path])
        value = candidate(work, raw, output)
        assert value["supplied_evidence_delta"][0]["could_reopen"] == [
            "source_custody:private-producer.py"
        ]
        assert value["physical_change_receipt"] is None and not value["history_authenticated"]
        assert e.replay(output)
        changed = deepcopy(work)
        changed["supplied"][0]["could_reopen"] = ["idcode"]
        with pytest.raises(ValueError, match="receipt_reopening_scope"):
            candidate(changed, raw, output)
        changed = deepcopy(work)
        changed["supplied"][0]["imported"][0]["sha256"] = "wrong"
        with pytest.raises(ValueError, match="receipt_reduction"):
            candidate(changed, raw, output)
        document["received_wall_ns"] = e.CUTOFF_NS
        altered = tmp_path / "stale.json"
        atomic_json(altered, document)
        changed = deepcopy(work)
        changed["supplied"][0]["reference"].update(e.base.reference(altered))
        with pytest.raises(ValueError, match="receipt_frontier"):
            candidate(changed, raw, output)


def test_private_authority_and_historical_inconsistency(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8386-CUSTODY: matching authority never repairs inconsistent history."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    natural = e.measure(e.ROOT, raw)
    changed = deepcopy(natural)
    import yaml

    tasks = yaml.safe_load((e.ROOT / e.ACTIVE).read_bytes())["tasks"]
    design = tmp_path / "design.md"
    design.write_text(
        "## Exact task contract\n<!-- V722_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(dict(milestone="2026.10.722", tasks=tasks))
        + "\n```\n"
    )
    row = next(r for r in changed["inputs"] if r["name"] == e.DESIGN)
    row["reference"].update(e.base.reference(design))
    row["observed_sha256"] = row["reference"]["sha256"]
    value = candidate(changed, raw, output)
    assert next(
        c
        for c in value["gate_check_summary"]
        if c["check"] == "independent_design.full_task_sha256"
    )["passed"]
    assert value["verdict_class"] == "blocked" and e.replay(output)
    with patch.object(e.old, "MISSING", []):
        with pytest.raises(ValueError, match="sealed_history_contract"):
            e.derive(natural)
    changed = deepcopy(natural)
    alternate = tmp_path / "changed-code"
    alternate.write_bytes(b"changed code")
    changed["code_config_hashes"][0]["original_path"] = str(alternate)
    candidate(changed, raw, output)
    assert not e.replay(output)


def receipt(path: Path, **changes: object) -> None:
    """SCENARIO-REPORT-8386-DELTA: private physical evidence names its exact frontier."""
    atomic_json(
        path,
        dict(
            schema=e.old.RECEIPT_SCHEMA,
            upstream_primary_sha256=e.old.HISTORY_PIN,
            exp8372_primary_sha256=e.HISTORY_PIN,
            received_date="20261010",
            received_wall_ns=e.CUTOFF_NS + 1,
            authorized_location="operator_physical_receipt",
            source_rows=[],
            physical_change=dict(
                date="20261010",
                operator="private operator",
                changed_fields=["cable", "power"],
                original_transcript_sha256=e.old.TRANSCRIPT_PIN,
            ),
            **changes,
        ),
    )


def test_receipt_delta_and_cutoff(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8386-DELTA: a new physical change cannot repair historical custody."""
    path, raw, output = tmp_path / "receipt.json", tmp_path / "raw", tmp_path / (e.NAME + ".json")
    receipt(path)
    work = e.measure(e.ROOT, raw, [path])
    value = candidate(work, raw, output)
    assert value["supplied_evidence_delta"][0]["authenticated"]
    assert value["supplied_evidence_delta"][0]["could_reopen"] == ["physical_change"]
    assert value["physical_change_receipt"] and not value["history_authenticated"]
    assert value["completed_count"] == 2 and value["device_command_count"] == 0
    assert e.replay(output)
    for replacement in [e.CUTOFF_NS, e.CUTOFF_NS - 1, "unknown"]:
        document = json.loads(path.read_bytes())
        document["received_wall_ns"] = replacement
        atomic_json(path, document)
        work = e.measure(e.ROOT, raw, [path])
        value = candidate(work, raw, output)
        assert not value["supplied_evidence_delta"][0]["authenticated"] and e.replay(output)
    receipt(path)
    document = json.loads(path.read_bytes())
    document["physical_change"]["changed_fields"] = ["software"]
    atomic_json(path, document)
    value = candidate(e.measure(e.ROOT, raw, [path]), raw, output)
    assert not value["supplied_evidence_delta"][0]["authenticated"]
    value = candidate(e.measure(e.ROOT, raw, [tmp_path / "missing"]), raw, output)
    assert not value["supplied_evidence_delta"][0]["authenticated"] and e.replay(output)
    with patch.object(e.time, "monotonic", side_effect=[0, 301]):
        with pytest.raises(TimeoutError, match="delta_scan"):
            e.scan([path], raw)


def test_cli_children_and_failure_paths(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8386-CLI: real child outcomes distinguish external and owned failure."""
    from carnot.reporting import gatemate_obligation_runner_8386 as runner

    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    ran = child(
        "private_cli",
        [*cli, "--private-e2e", "--output", str(output)],
        tmp_path / "logs",
        deadline=120,
    )
    assert ran["passed"], Path(ran["stderr_path"]).read_text()
    assert e.replay(output)
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] and value["verdict_class"] == "blocked"
    for name, args, expected in [
        ("valid", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "missing")], 1),
        ("date", ["--date", "20261009"], 2),
        ("unsafe_private", ["--private-e2e"], 2),
    ]:
        assert child(name, [*cli, *args], tmp_path / "logs", expected=expected, deadline=60)[
            "passed"
        ]
    assert any(p["name"] == "private_E2E018_consumers" for p in runner.manifest(tmp_path))
    private = tmp_path / "scratch"
    private.mkdir()
    atomic_json(private / "coverage.json", dict(private_control=True))
    failed = child(
        "deliberate_failure",
        [sys.executable, "-u", "-c", "raise SystemExit(7)"],
        tmp_path / "failed",
        deadline=10,
    )
    with patch.object(runner, "execute", return_value=[failed]):
        assert (
            runner.run(
                e.ROOT, tmp_path / "failed-result" / (e.NAME + ".json"), private, control=True
            )
            == 0
        )
    assert (
        json.loads((tmp_path / "failed-result" / (e.NAME + ".json")).read_bytes())["verdict_class"]
        == "disqualified"
    )
    with patch.object(
        runner.qualified, "preflight", return_value=[e.gate(tmp_path, "missing_tool", True, None)]
    ):
        assert (
            runner.run(
                tmp_path / "missing",
                tmp_path / "absent-result" / (e.NAME + ".json"),
                private,
                control=True,
            )
            == 0
        )
    assert (
        json.loads((tmp_path / "absent-result" / (e.NAME + ".json")).read_bytes())["verdict_class"]
        == "disqualified"
    )


def test_replay_primitive_and_log_failures(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8386-CLI: repaired checksums do not authorize altered primitive meaning."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    assert e.replay(output) is False
    candidate(work, raw, output)
    log = tmp_path / "log"
    log.write_text("actual validation")
    checked = dict(passed=True, stdout_path=str(log), stdout_sha256=e.base.sha256_file(log))
    value = e.build(work, [checked], raw, output)
    atomic_json(output, value)
    assert e.replay(output)
    log.write_text("tamper")
    assert not e.replay(output)
    changed = deepcopy(work)
    row = next(r for r in changed["inputs"] if r["name"] == e.ACTIVE)
    altered = tmp_path / "task.yaml"
    import yaml

    plan = yaml.safe_load(e.base.checked(row["reference"]).read_bytes())
    task = next(t for t in plan["tasks"] if t["id"] == e.TASK)
    task["priority"] = "invented"
    altered.write_text(yaml.safe_dump(plan))
    row["reference"].update(e.base.reference(altered))
    row["observed_sha256"] = row["reference"]["sha256"]
    candidate(changed, raw, output)
    assert not e.replay(output)
