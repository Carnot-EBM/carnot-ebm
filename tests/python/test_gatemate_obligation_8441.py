"""REQ-REPORT-8441 / REQ-VERIFY-8441: continuity earns no device observation."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import yaml

from carnot.reporting import gatemate_obligation_8441 as e
from carnot.reporting import gatemate_obligation_runner_8441 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child


def candidate(work: dict, raw: Path, output: Path, passed: bool = True) -> dict:
    """SCENARIO-REPORT-8441-REPLAY: reduce actual private bytes for each control."""
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [dict(passed=passed, scope="owned")], raw, output)
    atomic_json(output, value)
    return value


def test_exact_continuity_and_rehashed_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8441-CONTINUITY: absent evidence stays null and blocked."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    value = candidate(work, raw, output)
    upstream = json.loads((e.ROOT / e.UPSTREAM).read_bytes())
    assert value["experiment_id"] == 8441 and value["run_date"] == "20261012"
    assert value["task_id"] == e.TASK and value["milestone"] == "2026.10.726"
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert value["honest_verdict"] == "complete_blocked_gatemate_obligation"
    assert value["gatemate_obligation_recorded_score"] == value["current_contract_ready_score"] == 1
    assert (value["intended_count"], value["completed_count"], value["censored_count"]) == (7, 1, 6)
    assert value["missing_source_hashes"] == upstream["exact_missing_hashes"]
    assert value["reopening_condition"] == upstream["next_evidence_conditions"]
    assert value["last_authenticated_idcode"] == "0xffffffff"
    assert value["physical_change_reference"] is None and not value["gatemate_bitstream_flashed"]
    assert value["device_commands_executed"] == [] and value["execution_ready_score"] == 0
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert value["canonical_tasks_sha256"] == e.TASKS_PIN
    assert len(value["obligation_rows"]) == 7
    assert all(
        r["change_status"] == "unchanged" and r["source_hashes"] for r in value["obligation_rows"]
    )
    assert value["evidence_delta_scan_limit_s"] == 60 and e.replay(output)
    for field, replacement in [
        ("rows", []),
        ("missing_source_hashes", []),
        ("device_commands_executed", ["flash"]),
        ("current_contract_ready_score", 0),
    ]:
        changed = deepcopy(value)
        changed[field] = replacement
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(output, changed)
        assert not e.replay(output)
    assert candidate(work, raw, output, False)["verdict_class"] == "disqualified"
    missing = candidate(e.measure(tmp_path / "absent", raw), raw, output)
    assert (
        missing["verdict_class"] == "blocked" and missing["gatemate_obligation_recorded_score"] == 0
    )
    assert e.replay(output) and not e.replay(tmp_path / "missing.json")
    altered = deepcopy(work)
    row = next(r for r in altered["inputs"] if r["name"] == e.ACTIVE)
    plan = yaml.safe_load(e.base.checked(row["reference"]).read_bytes())
    plan["tasks"][0]["prompt"] += " unauthorized drift"
    path = tmp_path / "altered.yaml"
    path.write_text(yaml.safe_dump(plan))
    row["reference"].update(e.base.reference(path))
    row["observed_sha256"] = row["reference"]["sha256"]
    assert candidate(altered, raw, output)["current_contract_ready_score"] == 0


def test_explicit_receipt_frontier_and_forgery(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8441-CONTINUITY: changed setup queues only future work."""
    from test_gatemate_continuity_8400 import receipt

    raw, path, output = tmp_path / "raw", tmp_path / "receipt.json", tmp_path / (e.NAME + ".json")
    receipt(path)
    document = json.loads(path.read_bytes())
    document["received_date"] = document["physical_change"]["date"] = "20261012"
    atomic_json(path, document)
    work = e.measure(e.ROOT, raw, [path])
    value = candidate(work, raw, output)
    assert value["completed_count"] == 2 and value["execution_ready_score"] == 0
    assert value["physical_change_reference"]["date"] == "20261012" and e.replay(output)
    assert (
        next(r for r in value["obligation_rows"] if r["source_id"] == "physical_change")[
            "change_status"
        ]
        == "new"
    )
    for field, bad in [
        ("changed_fields", ["software"]),
        ("original_transcript_sha256", "sha256:forged"),
        ("date", "20261013"),
    ]:
        forged = deepcopy(document)
        forged["physical_change"][field] = bad
        atomic_json(path, forged)
        assert not e.measure(e.ROOT, raw, [path])["supplied"][0]["authenticated"]
    document["received_date"] = "20261013"
    atomic_json(path, document)
    assert not e.measure(e.ROOT, raw, [path])["supplied"][0]["authenticated"]


def test_real_private_cli_and_child_failure(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8441-CONTROLS: CLI children bind logs and real consumers."""
    cli = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    ran = child(
        "private_cli",
        [*cli, "--date", "20261012", "--private-e2e", "--output", str(output)],
        tmp_path / "logs",
        deadline=120,
    )
    assert ran["passed"], Path(ran["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] and e.replay(output)
    selected = reader_receipt(
        e.TASK, tmp_path, field="gatemate_obligation_recorded_score", expected=1
    )
    assert selected["passed"] and selected["gate_sha256"] == e.base.sha256_file(output)
    for name, args, expected in [
        ("cold", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("date", ["--date", "20261011"], 2),
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
