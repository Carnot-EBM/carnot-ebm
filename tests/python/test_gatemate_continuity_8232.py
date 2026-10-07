"""REQ-REPORT-8232 / REQ-VERIFY-8232: a receipt audit earns no hardware credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import gatemate_continuity_8232 as h
from carnot.reporting import gatemate_execution_8232 as cli
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.primary_publication import publish_primary


def receipt():
    """Explicit private operator evidence tests parsing, never the current board."""
    return dict(
        operator_authored=True,
        receipt_date="20261007",
        board=h.legacy.EXPECTED_BOARD,
        board_present=True,
        power_state="powered with verified supply",
        usb_jtag_cable_state="replacement cable",
        host_path="private fixture port",
        intended_recovery_action="detect",
        changed_physical_fields=["cable"],
    )


def panel(changed=False):
    """One private board row keeps the physical gate separate from audit completion."""
    rows, change = h.legacy.audit_receipts(
        Path("/tmp"),
        cutoff_date="20260823",
        run_date="20261007",
        candidates=[receipt()] if changed else [],
        dry_run=True,
    )
    return dict(
        board=dict(board="GateMate", blocked_idcode="0xffffffff", source_path="private"),
        history_ready=True,
        checks=[],
        references=[],
        cited=[],
        fixture=True,
        receipt_rows=rows,
        physical_change=change,
    )


def invoke(*args):
    """Actual script children must find their imports from outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(h.ROOT / h.CLI), *map(str, args)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("changed", [False, True])
def test_receipt_reduction_never_executes_board(changed):
    """SCENARIO-VERIFY-8232-RECEIPTS: changed fixtures only qualify future preflight."""
    value = h.reduce(panel(changed))
    assert value["verdict_class"] == "blocked"
    assert value["gatemate_obligation_ready_score"] == 1
    assert value["physical_change_evidence"]["exists"] is changed
    assert value["rows"][0]["blocked_idcode"] == "0xffffffff"
    assert not value["rows"][0]["terminal_criterion_met"]
    assert value["current_device_execution_count"] == 0
    assert value["completed_count"] == 0 and value["excluded_count"] == 1
    assert value["independent_count"] == value["generalized_learning_benefit_score"] == 0
    if not changed:
        assert value["honest_verdict"] == "complete_blocked_gatemate_physical_change"
    data = panel(changed)
    data["history_ready"] = False
    assert h.reduce(data)["honest_verdict"] == "complete_blocked_gatemate_history"
    assert h.reduce(data)["rows"][0]["numerator"] is None


@pytest.mark.parametrize("changed", [False, True])
def test_private_cli_and_cold_replay(tmp_path, changed):
    """SCENARIO-REPORT-8232-CLI: fresh replay checks primitives and exact summaries."""
    source, output = tmp_path / "fixture.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, panel(changed))
    run = invoke("--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    original = json.loads(output.read_bytes())
    assert original["MODEL_SPECS"] == [] and original["current_model_calls"] == 0
    assert original["fixture_mode"] and original["verdict_class"] == "blocked"
    assert invoke("--cold-replay", output).returncode == 0
    for field, replacement in [
        ("config", {}),
        ("completed_count", 1),
        ("reproducibility_checksum", "bad"),
    ]:
        value = deepcopy(original)
        value[field] = replacement
        if field != "reproducibility_checksum":
            value["reproducibility_checksum"] = cli.checksum(value)
        atomic_json(output, value)
        assert invoke("--cold-replay", output).returncode == 1
    assert invoke("--date", "bad").returncode == 2
    assert invoke("--input", source, "--output", h.ROOT / "results" / output.name).returncode == 1
    assert invoke("--input", tmp_path / "missing", "--output", output).returncode == 1


def test_receipt_mutations_and_missing_history_cli(tmp_path):
    """SCENARIO-VERIFY-8232-RECEIPTS: plans, stale dates and wrong boards earn no change."""
    for mutation in [
        dict(receipt_date="20260823"),
        dict(receipt_date="20261008"),
        dict(operator_authored=False),
        dict(board="wrong"),
        dict(changed_physical_fields=[]),
    ]:
        _, change = h.legacy.audit_receipts(
            tmp_path,
            cutoff_date="20260823",
            run_date="20261007",
            candidates=[dict(receipt(), **mutation)],
            dry_run=True,
        )
        assert not change["exists"]
    data = panel()
    data["history_ready"] = False
    source, output = tmp_path / "missing.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, data)
    run = invoke("--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(output.read_bytes())["honest_verdict"] == "complete_blocked_gatemate_history"


def private_history(root, monkeypatch):
    """Publish a private upstream through the real byte-bound publication helper."""
    source = root / "history.json"
    atomic_json(source, dict(run_date="20260823"))
    board = dict(
        board="GateMate",
        blocked_idcode="0xffffffff",
        source_path=str(source),
        source_hash=reference(source)["sha256"],
        source_transcript=str(source),
        source_transcript_sha256=reference(source)["sha256"],
    )
    output = root / h.UPSTREAM
    value = dict(
        experiment_id=8216,
        task_id="exp8216-hardware-workload-obligations",
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        board_rows=[board],
    )
    publication = publish_primary(output, value, lambda p: dict(passed=True))
    monkeypatch.setattr(h, "PIN", reference(output)["sha256"])
    monkeypatch.setattr(h, "HISTORY_PIN", reference(source)["sha256"])
    for relative in h.DOCS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".yaml" else "GateMate reference\n")
    return output, source, publication


def test_authenticated_inputs_and_missing_bytes(tmp_path, monkeypatch):
    """REQ-VERIFY-8232: missing history retains its row and never substitutes evidence."""
    assert not h.load(tmp_path, tmp_path / "missing")["history_ready"]
    upstream, source, publication = private_history(tmp_path, monkeypatch)
    data = h.load(tmp_path, tmp_path / "valid")
    assert data["history_ready"] and not data["physical_change"]["exists"]
    source.write_text("{}")
    assert not h.load(tmp_path, tmp_path / "changed")["history_ready"]
    source.unlink()
    assert not h.load(tmp_path, tmp_path / "absent")["history_ready"]
    upstream.write_text("[]")
    assert not h.load(tmp_path, tmp_path / "schema")["history_ready"]
    assert Path(publication["sidecar_path"]).exists()


def test_changed_operator_document_and_schema_failures(tmp_path, monkeypatch):
    """REQ-VERIFY-8232: dated structured receipts remain distinct from host success."""
    upstream, _, _ = private_history(tmp_path, monkeypatch)
    followup = tmp_path / "ops/operator-followup.md"
    followup.write_text("# GateMate changed fixture\n```json\n" + json.dumps(receipt()) + "\n```\n")
    data = h.load(tmp_path, tmp_path / "changed-receipt")
    assert data["physical_change"]["exists"]
    assert h.reduce(data)["honest_verdict"] == "complete_blocked_gatemate_device_preflight"
    (tmp_path / h.REOPEN).unlink()
    data = h.load(tmp_path, tmp_path / "missing-note")
    assert h.reduce(data)["honest_verdict"] == "complete_blocked_named_input"
    assert h.reduce(data)["gatemate_obligation_ready_score"] == 0
    value = json.loads(upstream.read_bytes())
    value["board_rows"] = []
    publish_primary(upstream, value, lambda p: dict(passed=True))
    monkeypatch.setattr(h, "PIN", reference(upstream)["sha256"])
    assert any(
        c["observed"] == "gatemate_row_count"
        for c in h.load(tmp_path, tmp_path / "no-board")["checks"]
    )


def test_owned_checks_and_atomic_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8232-CHECKS: failed owned or terminal checks preserve prior bytes."""
    plan = cli.commands(tmp_path / "plan")
    assert {"e2e015", "e2e019", "affected_consumers"} <= {s.name for s in plan}
    typed = next(s for s in plan if s.name == "changed_module_mypy")
    assert "--strict" in typed.argv and h.CLI in typed.argv
    assert len(cli.validators(tmp_path / "candidate.json")) == 3
    monkeypatch.setattr(cli, "commands", lambda p: [])
    monkeypatch.setattr(cli, "validators", lambda p: [])
    monkeypatch.setattr(h, "load", lambda root, raw: panel(True))
    monkeypatch.setattr(cli, "execute", lambda plan, raw: [])
    output = tmp_path / (h.NAME + ".json")
    args = ["--root", str(tmp_path), "--output", str(output)]
    assert cli.main(args) == 0
    assert cli.replay(output)["passed"]
    with patch.object(
        cli, "execute", side_effect=[[], [dict(passed=False, normal_exit=True)], [], [], []]
    ):
        assert cli.main(args) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert not value["required_checks_passed"] and value["gatemate_obligation_ready_score"] == 0
    assert cli.replay(output)["passed"]
    original = output.read_bytes()
    monkeypatch.setattr(cli, "execute", lambda plan, raw: [dict(passed=False, normal_exit=True)])
    assert cli.main(args) == 1
    assert output.read_bytes() == original
    assert list((tmp_path / "raw" / h.NAME).rglob("failed_terminal_candidate.json"))
    monkeypatch.setattr(cli, "execute", lambda plan, raw: [])
    with patch.object(cli, "publish_primary", side_effect=ValueError("publication_rejected")):
        assert cli.main(args) == 1
    assert output.read_bytes() == original


def test_rehashed_receipt_primitive_attacks(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8232-CLI: rehashing altered parser rows cannot excuse drift."""
    monkeypatch.setattr(cli, "commands", lambda p: [])
    monkeypatch.setattr(cli, "validators", lambda p: [])
    monkeypatch.setattr(cli, "execute", lambda plan, raw: [])
    source, output = tmp_path / "fixture.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, panel(True))
    assert cli.main(["--input", str(source), "--output", str(output)]) == 0
    original = json.loads(output.read_bytes())
    original_data = json.loads(Path(original["replay_input_reference"]["path"]).read_bytes())
    for attack, error in [("row", "receipt_drift"), ("exists", "physical_change_drift")]:
        data = deepcopy(original_data)
        if attack == "row":
            data["receipt_rows"][0]["valid"] = False
        else:
            data["physical_change"]["exists"] = False
        value = deepcopy(original)
        primitive = Path(value["replay_input_reference"]["path"])
        atomic_json(primitive, data)
        value["replay_input_reference"] = reference(primitive)
        value["raw_shard_hashes"] = [
            reference(primitive) if r["path"] == str(primitive) else r
            for r in value["raw_shard_hashes"]
        ]
        value["reproducibility_checksum"] = cli.checksum(value)
        atomic_json(output, value)
        with pytest.raises(ValueError, match=error):
            cli.replay(output)
