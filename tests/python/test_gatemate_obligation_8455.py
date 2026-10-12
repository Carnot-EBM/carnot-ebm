"""REQ-REPORT-8455 / REQ-VERIFY-8455: recording earns no physical progress."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import yaml

from carnot.reporting import gatemate_obligation_8455 as e
from carnot.reporting import gatemate_obligation_runner_8455 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child


def candidate(work: dict, raw: Path, output: Path, passed: bool = True) -> dict:
    """SCENARIO-REPORT-8455-REPLAY: each control binds its actual primitive bytes."""
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [dict(passed=passed, scope="owned")], raw, output)
    atomic_json(output, value)
    return value


def test_exact_continuity_and_rehashed_tamper(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8455-CONTINUITY: carry hashes without inventing observations."""
    raw, output = tmp_path / "raw", tmp_path / (e.NAME + ".json")
    work = e.measure(e.ROOT, raw)
    value = candidate(work, raw, output)
    upstream = json.loads((e.ROOT / e.UPSTREAM).read_bytes())
    assert value["experiment_id"] == 8455 and value["run_date"] == "20261012"
    assert value["task_id"] == e.TASK and value["milestone"] == "2026.10.727"
    assert value["honest_verdict"] == "complete_blocked_gatemate_obligation"
    assert value["verdict_class"] == "blocked" and value["required_checks_passed"]
    assert value["original_producer_replay_passed"]
    assert len(value["historical_authority_references"]) == 3
    assert value["missing_source_hashes"] == upstream["missing_source_hashes"]
    assert value["reopening_condition"][0]["id"] == "source_authority"
    assert value["reopening_condition"][1:] == upstream["reopening_condition"]
    assert value["last_authenticated_idcode"] == "0xffffffff"
    assert value["physical_change_reference"] is None
    assert not value["gatemate_bitstream_flashed"] and value["device_commands_executed"] == []
    assert value["execution_ready_score"] == value["independent_count"] == 0
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["intended_count"] == len(value["obligation_rows"]) == 7
    assert all(r["change_status"] == "unchanged" for r in value["obligation_rows"])
    assert value["evidence_delta_scan_limit_s"] == 60 and e.replay(output)
    assert value["canonical_tasks_sha256"] == e.TASKS_PIN
    for field, replacement in [
        ("rows", []),
        ("missing_source_hashes", []),
        ("device_commands_executed", ["flash"]),
        ("original_producer_replay_passed", False),
        ("current_contract_ready_score", 1 - value["current_contract_ready_score"]),
    ]:
        changed = deepcopy(value)
        changed.pop("reproducibility_checksum")
        changed[field] = replacement
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(output, changed)
        assert not e.replay(output)
    assert candidate(work, raw, output, False)["verdict_class"] == "disqualified"
    missing = candidate(e.measure(tmp_path / "absent", raw), raw, output)
    assert missing["verdict_class"] == "blocked" and not missing["original_producer_replay_passed"]
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


def test_supplied_receipt_frontier(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8455-CONTINUITY: authentic changed setup queues future work."""
    from test_gatemate_continuity_8400 import receipt

    raw, path, output = tmp_path / "raw", tmp_path / "receipt.json", tmp_path / (e.NAME + ".json")
    receipt(path)
    document = json.loads(path.read_bytes())
    document["received_date"] = document["physical_change"]["date"] = "20261012"
    atomic_json(path, document)
    value = candidate(e.measure(e.ROOT, raw, [path]), raw, output)
    assert value["physical_change_reference"]["date"] == "20261012" and e.replay(output)
    row = next(r for r in value["obligation_rows"] if r["source_id"] == "physical_change")
    assert row["change_status"] == "new" and row["supplied_delta_references"]
    document["physical_change"]["original_transcript_sha256"] = "sha256:forged"
    atomic_json(path, document)
    assert not e.measure(e.ROOT, raw, [path])["supplied"][0]["authenticated"]


def test_authority_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8455-REPLAY: all complete task objects and visible rows matter."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    tasks = yaml.safe_load((e.ROOT / e.ACTIVE).read_bytes())["tasks"]
    design = tmp_path / "complete-private-design.md"
    table = "\n".join(
        f"| {i} | {t['id']} | {t['title']} | {t['phase']} | {t['deliverable']} |"
        for i, t in enumerate(tasks, 1)
    )
    design.write_text(
        "## Exact task contract\n"
        + table
        + "\nCanonical full-task SHA-256: `"
        + e.AUTHORITY_DIGEST
        + "`\n<!-- V727_TASK_CONTRACT_START -->\n```json\n"
        + json.dumps(dict(milestone="2026.10.727", tasks=tasks))
        + "\n```\n"
    )
    assert len(e.current_design(design.read_text(), milestone="2026.10.727")[0]) == 14
    row = next(r for r in work["inputs"] if r["name"] == e.DESIGN)
    row["reference"].update(e.base.reference(design))
    row["observed_sha256"] = row["reference"]["sha256"]
    assert all(c["passed"] for c in e.design_checks(work))
    value = candidate(work, tmp_path / "raw", tmp_path / (e.NAME + ".json"))
    assert value["gatemate_obligation_recorded_score"] == 1
    assert (value["completed_count"], value["censored_count"]) == (1, 6)
    assert value["verdict_class"] == "blocked" and e.replay(tmp_path / (e.NAME + ".json"))


def test_real_private_cli_and_child_failure(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8455-CONTROLS: shipped children bind logs and live consumers."""
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
    work = json.loads(e.base.checked(value["work_reference"]).read_bytes())
    manifest = json.loads(e.base.checked(work["execution_manifest_reference"]).read_bytes())
    assert manifest["measurement_cap_s"] == 3600 and manifest["validation_reserve_s"] == 900
    selected = reader_receipt(e.TASK, tmp_path, field="verdict_class", expected="blocked")
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
    private = tmp_path / "scratch"
    private.mkdir()
    plan = runner.manifest(private)
    assert any(p["name"] == "private_E2E018_consumers" for p in plan)
    assert any(p["name"] == "private_E2E021" for p in plan)
    assert not any(p["scope"] == "global" for p in plan)
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
    with patch.object(runner.old, "run", return_value=0):
        assert runner.run(e.ROOT, tmp_path / "delegated.json", private) == 0
    script = tmp_path / "coverage_control.py"
    script.write_text(
        "from carnot.reporting.gatemate_obligation_runner_8455 import main\nraise SystemExit(main(['--cold-replay', '/absent-8455-control']))\n"
    )
    covered = child(
        "covered_missing",
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=/dev/null",
            "--parallel-mode",
            "--data-file=" + str(private / ".coverage"),
            str(script),
        ],
        tmp_path / "coverage-child",
        expected=1,
        deadline=60,
    )
    assert covered["passed"]
    with patch.object(runner.old, "run", return_value=0):
        assert runner.run(e.ROOT, tmp_path / "retained.json", private) == 0
    inventory = json.loads((tmp_path / "raw/retained/coverage_shards/inventory.json").read_bytes())
    assert inventory and all(e.base.checked(r).is_file() for r in inventory)


def test_history_rejection(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8455-REPLAY: immutable original authority is a required operand."""
    with patch.object(e.old, "replay", return_value=False):
        assert not e.measure(e.ROOT, tmp_path / "raw")["original_history"]["passed"]
    with patch.object(e.old, "replay", side_effect=ValueError("private producer error")):
        assert not e.measure(e.ROOT, tmp_path / "error")["original_history"]["passed"]
    work = e.measure(e.ROOT, tmp_path / "rehashed")
    work["original_history"]["passed"] = False
    import pytest

    with pytest.raises(ValueError, match="original_history_reduction"):
        candidate(work, tmp_path / "rehashed", tmp_path / (e.NAME + ".json"))
