"""REQ-REPORT-8246 / REQ-VERIFY-8246: private evidence cannot become board success."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import gatemate_change_ledger_8246 as h
from carnot.reporting import gatemate_ledger_execution_8246 as cli
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.primary_publication import publish_primary


def receipt(**changes):
    """A timestamp distinguishes a same-day physical change from old evidence."""
    return dict(
        operator_authored=True,
        receipt_date="20261007",
        receipt_timestamp="2026-10-07T12:00:00+00:00",
        board=h.legacy.EXPECTED_BOARD,
        board_present=True,
        power_state="private verified supply",
        usb_jtag_cable_state="private replacement cable",
        host_path="private port",
        intended_recovery_action="detect_then_existing_n16_smoke",
        changed_physical_fields=["cable"],
        **changes,
    )


def panel(changed=False):
    """Scripted operator evidence certifies parser behavior only."""
    data = h.empty_data()
    data.update(
        fixture=True,
        history_ready=True,
        board=dict(board="GateMate", blocked_idcode="0xffffffff"),
        frontier=dict(
            cutoff_date="20261007",
            cutoff_wall_ns=1791367200000000000,
            observed_wall_ns=1791403200000000000,
            seen_receipt_hashes=[],
            documents={},
        ),
    )
    if changed:
        data["candidate_rows"], _ = h.legacy.audit_receipts(
            Path("/tmp"),
            cutoff_date="20260823",
            run_date="20261007",
            candidates=[receipt()],
            dry_run=True,
        )
    data["receipt_rows"], data["physical_change"] = h.select_change(data)
    return data


def invoke(*args):
    """The real CLI must resolve imports without relying on the caller's cwd."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(h.ROOT / h.CLI), *map(str, args)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.parametrize("changed", [False, True])
def test_reduction_and_real_cli(tmp_path, changed):
    """SCENARIO-REPORT-8246-CLI: retain exactly one obligation with zero execution."""
    source, output = tmp_path / "fixture.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, panel(changed))
    run = invoke("--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["physical_change_evidence"]["exists"] is changed
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"] == "complete_blocked_" + (
        "gatemate_device_preflight" if changed else "gatemate_physical_change"
    )
    assert len(value["rows"]) == value["intended_count"] == value["excluded_count"] == 1
    assert value["current_device_execution_count"] == value["current_model_calls"] == 0
    assert value["MODEL_SPECS"] == value["trained_head_specs"] == []
    assert value["execution_ready_score"] == value["scientific_benefit_score"] == 0
    assert bool(value["future_probe_contract"]["eligible"]) is changed
    assert not value["rows"][0]["terminal_criterion_met"]
    assert value["rows"][0]["missing_status"] == (None if changed else "physical_receipt_absent")
    assert invoke("--cold-replay", output).returncode == 0
    original = deepcopy(value)
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


@pytest.mark.parametrize(
    "mutation",
    [
        dict(receipt_timestamp=None),
        dict(receipt_timestamp="bad"),
        dict(receipt_timestamp="2026-10-07T12:00:00"),
        dict(receipt_timestamp="2026-10-07T08:00:00Z"),
        dict(receipt_timestamp="2026-10-07T23:00:00Z"),
        dict(receipt_timestamp="2026-10-08T12:00:00Z"),
        dict(receipt_date="20261006"),
        dict(receipt_date="20261008"),
        dict(operator_authored=False),
        dict(board="wrong"),
        dict(changed_physical_fields=[]),
    ],
)
def test_receipt_frontier_rejections(mutation):
    """SCENARIO-VERIFY-8246-FRONTIER: dates and operator identity are actual gates."""
    data = panel()
    data["candidate_rows"], _ = h.legacy.audit_receipts(
        Path("/tmp"),
        cutoff_date="20260823",
        run_date="20261007",
        candidates=[dict(receipt(), **mutation)],
        dry_run=True,
    )
    rows, change = h.select_change(data)
    assert not change["exists"] and not rows[0]["valid"]
    data = panel(True)
    data["frontier"]["seen_receipt_hashes"] = [canonical_hash(receipt())]
    rows, change = h.select_change(data)
    assert not rows and not change["exists"]


def private_previous(root, monkeypatch):
    """A private primary exercises the existing terminal binding without board contact."""
    primitive = root / "previous-primitives.json"
    atomic_json(primitive, dict(receipt_rows=[], board=panel()["board"]))
    output = root / h.UPSTREAM
    value = dict(
        experiment_id=8232,
        task_id="exp8232-gatemate-continuity",
        schema="carnot.gatemate_continuity.v711.v1",
        run_date="20261007",
        honest_verdict="complete_blocked_gatemate_physical_change",
        verdict_class="blocked",
        required_checks_passed=True,
        flagged_adversarial=False,
        fixture_mode=False,
        invocation=dict(
            started_wall_ns=1791367200000000000, started_monotonic_ns=10, ended_monotonic_ns=20
        ),
        source_artifact_hashes=[],
        replay_input_reference=reference(primitive),
        gatemate_obligation=panel()["board"],
    )
    publish_primary(output, value, lambda p: dict(passed=True))
    monkeypatch.setattr(h, "PIN", reference(output)["sha256"])
    monkeypatch.setattr(h.previous_cli, "replay", lambda p: dict(passed=True))
    for relative in dict.fromkeys([*h.DOCS, *map(str, h.legacy.RECEIPT_PATHS)]):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("GateMate reference\n")
    return output, primitive


def test_load_and_authenticated_frontier(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8246-FRONTIER: prior bytes, documents and binding are required."""
    assert (
        h.reduce(h.load(tmp_path, tmp_path / "missing"))["honest_verdict"]
        == "complete_blocked_gatemate_history"
    )
    upstream, primitive = private_previous(tmp_path, monkeypatch)
    data = h.load(tmp_path, tmp_path / "good")
    assert data["history_ready"]
    h.verify_primitives(data)
    followup = tmp_path / "ops/operator-followup.md"
    followup.write_text("# GateMate\n```json\n" + json.dumps(receipt()) + "\n```\n")
    changed = h.load(tmp_path, tmp_path / "changed")
    assert changed["physical_change"]["exists"]
    h.verify_primitives(changed)
    (tmp_path / h.REOPEN).unlink()
    assert (
        h.reduce(h.load(tmp_path, tmp_path / "no-note"))["honest_verdict"]
        == "complete_blocked_named_input"
    )
    value = json.loads(upstream.read_bytes())
    value["gatemate_obligation"]["board"] = "wrong"
    publish_primary(upstream, value, lambda p: dict(passed=True))
    monkeypatch.setattr(h, "PIN", reference(upstream)["sha256"])
    assert not h.load(tmp_path, tmp_path / "bad-row")["history_ready"]
    primitive.unlink()
    assert not h.load(tmp_path, tmp_path / "no-primitive")["history_ready"]


def test_unchanged_docs_are_not_reparsed(tmp_path, monkeypatch):
    """REQ-VERIFY-8246: stored document hashes are the no-repeat frontier."""
    upstream, _ = private_previous(tmp_path, monkeypatch)
    value = json.loads(upstream.read_bytes())
    value["source_artifact_hashes"] = [
        dict(reference(tmp_path / p), original_path=str(tmp_path / p))
        for p in map(str, h.legacy.RECEIPT_PATHS)
        if (tmp_path / p).exists()
    ]
    publish_primary(upstream, value, lambda p: dict(passed=True))
    monkeypatch.setattr(h, "PIN", reference(upstream)["sha256"])
    with patch.object(h.legacy, "audit_receipts", side_effect=AssertionError("reparsed")):
        data = h.load(tmp_path, tmp_path / "unchanged")
    assert all(
        r["disposition"] == "unchanged" for r in data["document_rows"] if r["receipt_source"]
    )


def test_owned_failures_and_primitive_tamper(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8246-CLI: failed owned checks zero readiness and retain bytes."""
    plan = cli.commands(tmp_path / "plan")
    assert {"e2e015", "e2e019", "affected_consumers"} <= {s.name for s in plan}
    assert h.CLI in next(s for s in plan if s.name == "changed_module_mypy").argv
    monkeypatch.setattr(cli, "commands", lambda p: [])
    monkeypatch.setattr(cli, "validators", lambda p: [])
    monkeypatch.setattr(h, "load", lambda root, raw: panel(True))
    monkeypatch.setattr(cli, "execute", lambda plan, raw: [])
    output = tmp_path / (h.NAME + ".json")
    args = ["--root", str(tmp_path), "--output", str(output)]
    assert cli.main(args) == 0
    original = json.loads(output.read_bytes())
    primitive = Path(original["replay_input_reference"]["path"])
    original_data = json.loads(primitive.read_bytes())
    for key in ["board", "receipt_rows", "physical_change"]:
        data = deepcopy(original_data)
        data[key] = {} if key != "receipt_rows" else []
        atomic_json(primitive, data)
        value = deepcopy(original)
        value["replay_input_reference"] = reference(primitive)
        value["raw_shard_hashes"][0] = reference(primitive)
        value["reproducibility_checksum"] = cli.checksum(value)
        atomic_json(output, value)
        with pytest.raises(ValueError):
            cli.replay(output)
        assert invoke("--cold-replay", output).returncode == 1
    with patch.object(
        cli, "execute", side_effect=[[], [dict(passed=False, normal_exit=True)], [], [], []]
    ):
        assert cli.main(args) == 0
    value = json.loads(output.read_bytes())
    assert (
        value["verdict_class"] == "disqualified" and value["gatemate_obligation_ready_score"] == 0
    )
    assert cli.replay(output)["passed"]
    original_bytes = output.read_bytes()
    monkeypatch.setattr(cli, "execute", lambda plan, raw: [dict(passed=False, normal_exit=True)])
    assert cli.main(args) == 1
    assert output.read_bytes() == original_bytes
    assert list((tmp_path / "raw" / h.NAME).rglob("failed_terminal_candidate.json"))
    monkeypatch.setattr(cli, "execute", lambda plan, raw: [])
    with patch.object(cli, "publish_primary", side_effect=ValueError("publication_rejected")):
        assert cli.main(args) == 1


def test_frontier_and_document_tampering(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8246-FRONTIER: rehashing metadata cannot hide changed documents."""
    upstream, _ = private_previous(tmp_path, monkeypatch)
    (tmp_path / "ops/operator-followup.md").write_text(
        "# GateMate\n```json\n" + json.dumps(receipt()) + "\n```\n"
    )
    data = h.load(tmp_path, tmp_path / "valid")
    for attack in [
        "frontier",
        "candidate",
        "disposition",
        "current_hash",
        "roster",
        "previous_pin",
    ]:
        altered = deepcopy(data)
        if attack == "frontier":
            altered["frontier"]["cutoff_wall_ns"] += 1
        elif attack == "candidate":
            altered["candidate_rows"] = []
        elif attack == "roster":
            altered["document_rows"] = []
        elif attack == "previous_pin":
            altered["previous_reference"]["sha256"] = "sha256:bad"
        else:
            row = next(
                r
                for r in altered["document_rows"]
                if r["relative_path"] == "ops/operator-followup.md"
            )
            row["disposition" if attack == "disposition" else "current_sha256"] = (
                "unchanged" if attack == "disposition" else "sha256:bad"
            )
        with pytest.raises(ValueError):
            h.verify_primitives(altered)
    altered = deepcopy(data)
    monkeypatch.setattr(h, "PIN", "sha256:bad")
    with pytest.raises(ValueError, match="previous_pin_drift"):
        h.verify_primitives(altered)
    assert not h.load(tmp_path, tmp_path / "wrong-pin")["history_ready"]
    assert upstream.exists()


def test_missing_history_private_cli(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8246-CLI: missing upstream is terminal blocked with one row."""
    data = h.empty_data()
    source, output = tmp_path / "missing-fixture.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, data)
    run = invoke("--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(output.read_bytes())["honest_verdict"] == "complete_blocked_gatemate_history"
    assert invoke("--cold-replay", output).returncode == 0


def test_candidate_parser_tamper():
    """REQ-VERIFY-8246: even a rehashed parser row must derive from its raw receipt."""
    data = panel(True)
    data["candidate_rows"][0]["raw_receipt"]["operator_authored"] = False
    with pytest.raises(ValueError, match="candidate_parser_drift"):
        h.verify_primitives(data)
