"""REQ-REPORT-8414 / REQ-VERIFY-8414: continuity cannot imply hardware readiness."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import yaml

from carnot.reporting import gatemate_obligation_8414 as e
from carnot.reporting import gatemate_obligation_runner_8414 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child


def candidate(work: dict, raw: Path, output: Path, passed: bool = True) -> dict:
    """SCENARIO-REPORT-8414-REPLAY: candidates bind real private primitive bytes."""
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [dict(passed=passed, scope="owned")], raw, output)
    atomic_json(output, value)
    return value


def test_obligation_and_full_design_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8414-CONTINUITY: absent obligations are null, never zero evidence."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    value = candidate(work, raw, output)
    upstream = json.loads((e.ROOT / e.UPSTREAM).read_bytes())
    assert value["experiment_id"] == 8414 and value["run_date"] == "20261011"
    assert value["task_id"] == e.TASK and value["milestone"] == "2026.10.724"
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert value["obligation_recorded_score"] == value["current_contract_ready_score"] == 1
    assert value["canonical_tasks_sha256"] == e.TASKS_PIN
    assert (value["intended_count"], value["completed_count"], value["censored_count"]) == (7, 1, 6)
    assert value["exact_missing_hashes"] == upstream["exact_missing_hashes"]
    assert value["next_evidence_conditions"] == upstream["next_evidence_conditions"]
    assert value["physical_change_receipt"] is None and value["supplied_evidence_delta"] == []
    assert value["original_idcode"] == "0xffffffff" and value["required_idcode"] == "0x20000001"
    assert value["execution_ready_score"] == value["device_command_count"] == 0
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["inference_substrate_class"] == "no_model_load" and value["MODEL_SPECS"] == []
    assert not any(value["model_invocation_counts"].values())
    assert value["evidence_delta_scan_limit_s"] == 60 and e.replay(output)
    for field, replacement in [
        ("execution_ready_score", 1),
        ("rows", []),
        ("exact_missing_hashes", []),
    ]:
        changed = deepcopy(value)
        changed[field] = replacement
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(output, changed)
        assert not e.replay(output)
    assert candidate(work, raw, output, False)["verdict_class"] == "disqualified"
    missing = candidate(e.measure(tmp_path / "absent", raw), raw, output)
    assert missing["verdict_class"] == "blocked" and missing["obligation_recorded_score"] == 0
    assert e.replay(output) and not e.replay(tmp_path / "absent.json")
    altered = deepcopy(work)
    row = next(r for r in altered["inputs"] if r["name"] == e.ACTIVE)
    plan = yaml.safe_load(e.base.checked(row["reference"]).read_bytes())
    plan["tasks"][0]["title"] = "unrelated task drift"
    path = tmp_path / "altered.yaml"
    path.write_text(yaml.safe_dump(plan))
    row["reference"].update(e.base.reference(path))
    row["observed_sha256"] = row["reference"]["sha256"]
    value = candidate(altered, raw, output)
    assert value["current_contract_ready_score"] == value["obligation_recorded_score"] == 0
    assert e.replay(output)


def test_explicit_receipts_only(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8414-CONTINUITY: receipt parsing queues future physical work only."""
    from test_gatemate_continuity_8400 import receipt

    path, raw, output = tmp_path / "receipt.json", tmp_path / "raw", tmp_path / (e.NAME + ".json")
    receipt(path)
    work = e.measure(e.ROOT, raw, [path])
    value = candidate(work, raw, output)
    assert value["completed_count"] == 2 and value["execution_ready_score"] == 0
    assert value["physical_change_receipt"]["date"] == "20261011" and e.replay(output)


def test_real_cli_controls_and_owned_failure(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8414-CONTROLS: real private children qualify exact reader bytes."""
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
    assert value["required_checks_passed"] and e.replay(output)
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
