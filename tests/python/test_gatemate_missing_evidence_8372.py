"""REQ-REPORT-8372 / REQ-VERIFY-8372: keep source custody and hardware work separate."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import gatemate_missing_evidence_8372 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_natural_obligation_and_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8372-OBLIGATION: absent evidence cannot erase physical obligations."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [dict(passed=True)], raw, output)
    assert value["verdict_class"] == "blocked"
    assert value["obligation_recorded_score"] == 1
    assert value["history_authentication"] is False
    assert value["missing_source_hashes"] == e.MISSING
    assert value["execution_ready_score"] == value["current_jtag_retry_count"] == 0
    assert value["physical_change_frontier"]["required_next_evidence"] == e.PHYSICAL
    atomic_json(output, value)
    assert e.replay(output)
    for field, changed in [
        ("execution_ready_score", 1),
        ("history_authentication", True),
        ("missing_source_hashes", []),
        ("MODEL_SPECS", [{}]),
        ("original_transcript_sha256", "wrong"),
    ]:
        bad = deepcopy(value)
        bad[field] = changed
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(output, bad)
        assert not e.replay(output), field
    assert not e.replay(tmp_path / "absent")
    assert e.build(work, [dict(passed=False)], raw, output)["verdict_class"] == "disqualified"


def test_private_receipt_consumers(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8372-RECEIPTS: synthetic bytes exercise parsing only in private scratch."""
    source = tmp_path / "source"
    source.write_bytes(b"private consumer positive bytes")
    missing = [dict(path="producer.py", sha256=e.base.sha256_file(source))]
    receipt = tmp_path / "receipt.json"
    value = dict(
        schema=e.RECEIPT_SCHEMA,
        upstream_primary_sha256=e.HISTORY_PIN,
        received_date="20261010",
        authorized_location="operator_supplied_snapshot",
        source_rows=[
            dict(original_path="producer.py", path=str(source), sha256=missing[0]["sha256"])
        ],
        physical_change=dict(
            date="20261010",
            operator="private operator",
            changed_fields=["cable", "power"],
            original_transcript_sha256=e.TRANSCRIPT_PIN,
        ),
    )
    atomic_json(receipt, value)
    rows = e.consume_receipt(receipt, tmp_path / "raw", missing)
    assert [r["disposition"] for r in rows] == ["supplied_exact_bytes", "queued_next_hardware_task"]
    assert e.consume_receipt(None, tmp_path / "raw", missing) == []
    assert (
        e.consume_receipt(tmp_path / "absent", tmp_path / "raw", missing)[0]["disposition"]
        == "rejected"
    )
    for field, replacement in [
        ("received_date", "20261009"),
        ("schema", "wrong"),
        ("authorized_location", "unknown"),
        ("upstream_primary_sha256", "wrong"),
    ]:
        atomic_json(receipt, dict(value, **{field: replacement}))
        assert e.consume_receipt(receipt, tmp_path / "raw", missing)[0]["disposition"] == "rejected"
    atomic_json(receipt, dict(value, physical_change=None))
    assert len(e.consume_receipt(receipt, tmp_path / "raw", missing)) == 1
    for change in [
        dict(value["physical_change"], changed_fields=["software"]),
        dict(value["physical_change"], date="20261009"),
        dict(value["physical_change"], operator=""),
        dict(value["physical_change"], original_transcript_sha256="wrong"),
    ]:
        atomic_json(receipt, dict(value, physical_change=change))
        assert e.consume_receipt(receipt, tmp_path / "raw", missing)[0]["disposition"] == "rejected"
    atomic_json(receipt, value)
    source.write_bytes(b"corrupt")
    assert e.consume_receipt(receipt, tmp_path / "raw", missing)[0]["disposition"] == "rejected"
    atomic_json(
        receipt,
        dict(value, source_rows=[dict(original_path="unknown", path=str(source), sha256="wrong")]),
    )
    assert e.consume_receipt(receipt, tmp_path / "raw", missing)[0]["disposition"] == "rejected"


def test_authenticated_failure_rejections(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8372-REPLAY: old history is read once without executing its replay."""
    work = dict(refs=[])
    with patch.object(e.base, "sha256_file", return_value="wrong"):
        with pytest.raises(ValueError, match="historical_primary_hash"):
            e.read_failure(e.ROOT, tmp_path, work)
    with patch.object(e, "sealed", return_value={}):
        with pytest.raises((KeyError, ValueError)):
            e.read_failure(e.ROOT, tmp_path, work)
    value = json.loads((e.ROOT / e.HISTORY).read_bytes())
    value["history_authentication"]["passed"] = True
    with patch.object(e, "sealed", return_value=value):
        with pytest.raises(ValueError, match="historical_missing_hashes"):
            e.read_failure(e.ROOT, tmp_path, work)
    with patch.object(
        e, "read_bound_sidecar", return_value=dict(primary_path="wrong", report=dict(passed=True))
    ):
        with pytest.raises(ValueError, match="terminal_custody"):
            e.sealed(e.ROOT / e.HISTORY, e.HISTORY_PIN, tmp_path, work)


def test_real_cli_and_owned_failures(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8372-CLI: real positive and negative children preserve blocked outcomes."""
    from carnot.reporting import gatemate_missing_runner_8372 as r
    from carnot.reporting.v709_execution import child

    output = tmp_path / (e.NAME + ".json")
    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    ran = subprocess.run(
        [*cli, "--private-e2e", "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"]
    assert value["obligation_recorded_score"] == 1 and e.replay(output)
    assert (
        subprocess.run(
            [*cli, "--cold-replay", str(output)], capture_output=True, timeout=60
        ).returncode
        == 0
    )
    assert (
        subprocess.run(
            [*cli, "--cold-replay", str(tmp_path / "missing")], capture_output=True, timeout=60
        ).returncode
        == 1
    )
    assert subprocess.run([*cli, "--private-e2e"], capture_output=True, timeout=30).returncode == 2
    assert (
        subprocess.run([*cli, "--date", "20261009"], capture_output=True, timeout=30).returncode
        == 2
    )
    assert any(row["name"] == "private_E2E018_consumers" for row in r.manifest(tmp_path))
    failed = child(
        "actual_failure",
        [sys.executable, "-u", "-c", "raise SystemExit(7)"],
        tmp_path / "failed",
        deadline=10,
    )
    assert failed["exit_code"] == 7 and not failed["passed"]
    private = tmp_path / "preflight"
    private.mkdir()
    atomic_json(private / "coverage.json", dict(private_control=True))
    with patch.object(
        r.qualified, "preflight", return_value=[e.gate(tmp_path, "executable", True, None)]
    ):
        assert (
            r.run(
                tmp_path / "absent",
                tmp_path / "missing" / (e.NAME + ".json"),
                private,
                control=True,
            )
            == 0
        )
    missing = json.loads((tmp_path / "missing" / (e.NAME + ".json")).read_bytes())
    assert missing["verdict_class"] == "disqualified" and missing["obligation_recorded_score"] == 0


def test_rehashed_primitive_controls(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8372-REPLAY: fresh hashes cannot bless altered historical meaning."""
    raw = tmp_path / "raw"
    natural = e.measure(e.ROOT, raw)
    output = tmp_path / (e.NAME + ".json")

    def candidate(work: dict) -> bool:
        atomic_json(raw / "measurement.json", work)
        atomic_json(output, e.build(work, [dict(passed=True)], raw, output))
        return e.replay(output)

    assert candidate(natural)
    changed = deepcopy(natural)
    changed["contract"]["task_sha256"] = "wrong"
    assert not candidate(changed)
    changed = deepcopy(natural)
    changed["history"]["honest_verdict"] = "invented"
    assert not candidate(changed)
    changed = deepcopy(natural)
    changed["final"]["required_checks_passed"] = True
    assert not candidate(changed)
    changed = deepcopy(natural)
    changed["code_config_hashes"][0]["original_path"] = str(tmp_path / "wrong-code")
    (tmp_path / "wrong-code").write_bytes(b"changed source")
    assert not candidate(changed)
    for suffix, body in [
        (e.HISTORY, dict(history_authentication=dict(passed=False))),
        (e.TRANSCRIPT, dict(changed=True)),
    ]:
        changed = deepcopy(natural)
        ref = next(r for r in changed["refs"] if r["original_path"] == str(e.ROOT / suffix))
        private = tmp_path / (Path(suffix).name + ".changed")
        atomic_json(private, body)
        ref.update(e.base.reference(private))
        assert not candidate(changed)
    changed = deepcopy(natural)
    terminal_ref = next(
        r for r in changed["refs"] if r["original_path"].endswith("terminal_validation.json")
    )
    terminal = json.loads(e.base.checked(terminal_ref).read_bytes())
    terminal["publication"]["primary_sha256"] = "wrong"
    private = tmp_path / "bad-terminal.json"
    atomic_json(private, terminal)
    terminal_ref.update(e.base.reference(private))
    assert not candidate(changed)
    changed = deepcopy(natural)
    request = json.loads(e.base.checked(changed["reopening_request_reference"]).read_bytes())
    request["required_next_evidence"] = []
    atomic_json(tmp_path / "bad-request.json", request)
    changed["reopening_request_reference"] = e.base.reference(tmp_path / "bad-request.json")
    assert not candidate(changed)
    wrong = tmp_path / "wrong-authority"
    with patch.object(e.authority, "authority", side_effect=ValueError("changed full task")):
        work = e.measure(e.ROOT, wrong)
    assert not work["contract"]


def test_supplied_primitive_controls(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8372-RECEIPTS: private supplied evidence is never a natural history recovery."""
    source = tmp_path / "private-source"
    source.write_bytes(b"private bytes only")
    missing = [dict(path="producer.py", sha256=e.base.sha256_file(source))]
    receipt = tmp_path / "receipt.json"
    atomic_json(
        receipt,
        dict(
            schema=e.RECEIPT_SCHEMA,
            upstream_primary_sha256=e.HISTORY_PIN,
            received_date="20261010",
            authorized_location="operator_supplied_snapshot",
            source_rows=[
                dict(original_path="producer.py", path=str(source), sha256=missing[0]["sha256"])
            ],
            physical_change=dict(
                date="20261010",
                operator="private",
                changed_fields=["port"],
                original_transcript_sha256=e.TRANSCRIPT_PIN,
            ),
        ),
    )
    raw = tmp_path / "raw"
    output = tmp_path / (e.NAME + ".json")
    with patch.object(e, "MISSING", missing):
        work = e.measure(tmp_path / "absent-root", raw, receipt)

        def replay(changed: dict) -> bool:
            atomic_json(raw / "measurement.json", changed)
            atomic_json(output, e.build(changed, [dict(passed=True)], raw, output))
            return e.replay(output)

        assert replay(work)
        changed = deepcopy(work)
        changed["supplied_evidence_rows"][0]["sha256"] = "wrong"
        assert not replay(changed)
        changed = deepcopy(work)
        changed["supplied_evidence_rows"][1]["physical_change"]["operator"] = "tampered"
        assert not replay(changed)
        changed = deepcopy(work)
        ref = changed["supplied_evidence_rows"][0]["receipt_reference"]
        document = json.loads(e.base.checked(ref).read_bytes())
        document["received_date"] = "20261009"
        atomic_json(tmp_path / "stale.json", document)
        ref.update(e.base.reference(tmp_path / "stale.json"))
        assert not replay(changed)
    atomic_json(receipt, dict(schema="corrupt"))
    work = e.measure(tmp_path / "absent-root", raw, receipt)
    assert any(c["artifact_field"] == "supplied_receipt.authentication" for c in work["checks"])
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, e.build(work, [dict(passed=True)], raw, output))
    assert e.replay(output)
