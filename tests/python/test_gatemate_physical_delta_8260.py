"""REQ-REPORT-8260 / REQ-VERIFY-8260: private receipts prove no board success."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import gatemate_physical_delta_8260 as h
from carnot.reporting import gatemate_delta_execution_8260 as cli
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.primary_publication import publish_primary
from test_gatemate_change_ledger_8246 import panel as previous_panel, receipt


def panel(changed=False):
    """The new fixture uses the current frontier's explanation, not its predecessor's."""
    data = previous_panel(changed)
    data["receipt_rows"], data["physical_change"] = h.select_change(data)
    return data


def invoke(*args):
    """External cwd proves the runner does not depend on an ambient import path."""
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
def test_real_cli_and_rehashed_tamper(tmp_path, changed):
    """SCENARIO-REPORT-8260-CLI: terminal audit readiness grants no execution."""
    source, output = tmp_path / "fixture.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, panel(changed))
    run = invoke("--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["experiment_id"] == 8260 and value["milestone"] == "2026.10.713"
    assert value["physical_change_evidence"]["exists"] is changed
    assert value["honest_verdict"] == "complete_blocked_" + (
        "gatemate_device_preflight" if changed else "gatemate_physical_change"
    )
    assert value["intended_count"] == value["excluded_count"] == len(value["rows"]) == 1
    assert value["current_device_execution_count"] == value["current_model_calls"] == 0
    assert value["MODEL_SPECS"] == value["trained_head_specs"] == []
    assert value["execution_ready_score"] == value["generalized_learning_benefit_score"] == 0
    assert value["reopen_contract_path"] == h.REOPEN
    assert value["future_probe_contract"]["eligible"] is changed
    assert invoke("--cold-replay", output).returncode == 0
    original = deepcopy(value)
    for field in ["experiment_id", "schema", "task_id"]:
        altered = dict(original, **{field: "wrong"})
        altered["reproducibility_checksum"] = cli.checksum(altered)
        atomic_json(output, altered)
        with pytest.raises(ValueError):
            cli.replay(output)
    altered = dict(original, reproducibility_checksum="bad")
    atomic_json(output, altered)
    with pytest.raises(ValueError, match="checksum_drift"):
        cli.replay(output)
    altered = deepcopy(original)
    altered.update(
        honest_verdict="complete_disqualified_owned_checks",
        verdict_class="disqualified",
        required_checks_passed=False,
        gatemate_obligation_ready_score=0,
    )
    altered["future_probe_contract"]["eligible"] = False
    altered["reproducibility_checksum"] = cli.checksum(altered)
    atomic_json(output, altered)
    assert cli.replay(output)["passed"]
    if changed:
        primitive = Path(original["replay_input_reference"]["path"])
        data = json.loads(primitive.read_bytes())
        data["candidate_rows"][0]["raw_receipt"]["operator_authored"] = False
        atomic_json(primitive, data)
        altered = deepcopy(original)
        altered["replay_input_reference"] = reference(primitive)
        altered["raw_shard_hashes"][0] = reference(primitive)
        altered["reproducibility_checksum"] = cli.checksum(altered)
        atomic_json(output, altered)
        assert invoke("--cold-replay", output).returncode == 1
        atomic_json(primitive, panel(True) | {"references": data["references"]})
    value["completed_count"] = 1
    value["reproducibility_checksum"] = cli.checksum(value)
    atomic_json(output, value)
    assert invoke("--cold-replay", output).returncode == 1
    assert invoke("--date", "bad").returncode == 2
    assert invoke("--input", source, "--output", h.ROOT / "results" / output.name).returncode == 1


def private_previous(root, monkeypatch):
    """A bound private upstream exercises custody without reading a device."""
    primitive = root / "prior.json"
    data = panel()
    atomic_json(primitive, data)
    output = root / h.UPSTREAM
    value = dict(
        experiment_id=8246,
        task_id="exp8246-gatemate-change-ledger",
        schema="carnot.gatemate_change_ledger.v712.v1",
        run_date="20261007",
        honest_verdict="complete_blocked_gatemate_physical_change",
        verdict_class="blocked",
        required_checks_passed=True,
        flagged_adversarial=False,
        fixture_mode=False,
        invocation=dict(
            started_wall_ns=1791367200000000000, started_monotonic_ns=10, ended_monotonic_ns=20
        ),
        source_artifact_hashes=[dict(reference(primitive), original_path=str(primitive))],
        replay_input_reference=reference(primitive),
        gatemate_obligation=data["board"],
        physical_change_frontier=data["frontier"],
    )
    publish_primary(output, value, lambda p: dict(passed=True))
    monkeypatch.setattr(h, "PIN", reference(output)["sha256"])
    monkeypatch.setattr(h, "_upstream_replay", lambda p: dict(passed=True))
    for relative in dict.fromkeys([*h.DOCS, *map(str, h.legacy.RECEIPT_PATHS)]):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("GateMate reference\n")
    return output


def test_authenticated_load_and_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8260-FRONTIER: only post-frontier receipts can qualify."""
    assert not h.load(tmp_path, tmp_path / "missing")["history_ready"]
    upstream = private_previous(tmp_path, monkeypatch)
    data = h.load(tmp_path, tmp_path / "good")
    assert data["history_ready"]
    h.verify_primitives(data)
    followup = tmp_path / "ops/operator-followup.md"
    followup.write_text("```json\n" + json.dumps(receipt()) + "\n```\n")
    changed = h.load(tmp_path, tmp_path / "changed")
    assert changed["physical_change"]["exists"]
    h.verify_primitives(changed)
    for field in ["frontier", "board", "candidate_rows", "document_rows"]:
        tamper = deepcopy(changed)
        tamper[field] = {} if field in {"frontier", "board"} else []
        with pytest.raises((ValueError, KeyError)):
            h.verify_primitives(tamper)
    value = json.loads(upstream.read_bytes())
    value["gatemate_obligation"]["board"] = "wrong"
    publish_primary(upstream, value, lambda p: dict(passed=True))
    monkeypatch.setattr(h, "PIN", reference(upstream)["sha256"])
    assert not h.load(tmp_path, tmp_path / "wrong")["history_ready"]
    monkeypatch.setattr(h, "PIN", "sha256:bad")
    assert not h.load(tmp_path, tmp_path / "wrong-pin")["history_ready"]
    (tmp_path / h.REOPEN).unlink()
    assert h.reduce(h.load(tmp_path, tmp_path / "missing-note"))["verdict_class"] == "blocked"


def test_missing_history_cli(tmp_path):
    """SCENARIO-REPORT-8260-CLI: missing units remain explicit and replayable."""
    source, output = tmp_path / "missing.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, h.empty_data())
    run = invoke("--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(output.read_bytes())["honest_verdict"] == "complete_blocked_gatemate_history"
    assert invoke("--cold-replay", output).returncode == 0


@pytest.mark.parametrize(
    "stamp", [None, "bad", "2026-10-07T10:00:00Z", "2026-10-07T09:59:59Z", "2026-10-08T00:00:00Z"]
)
def test_frontier_negative_cases(stamp):
    """REQ-VERIFY-8260: age and repeated evidence cannot authorize a fresh probe."""
    data = panel(True)
    data["candidate_rows"][0]["raw_receipt"]["receipt_timestamp"] = stamp
    assert not h.select_change(data)[1]["exists"]
    data = panel(True)
    data["frontier"]["seen_receipt_hashes"] = [canonical_hash(receipt())]
    assert h.select_change(data) == (
        [],
        dict(exists=False, reason="no authenticated physical change since Exp8246"),
    )


def test_plan_normalization_and_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8260: missing tools block, while failed owned checks disqualify."""
    plan = cli.commands(tmp_path / "plan")
    assert {"e2e015", "e2e019", "affected_consumers"} <= {p.name for p in plan}
    value = dict(
        experiment_id=8246,
        task_id="exp8246-gatemate-change-ledger",
        milestone="2026.10.712",
        run_date="20261007",
        honest_verdict="complete_disqualified_owned_checks",
        verdict_class="disqualified",
        required_checks_passed=False,
        code_config_hashes=[],
        gate_check_summary=[],
        precondition_receipts=[
            dict(
                passed=False,
                normal_exit=True,
                actual_exit=1,
                stdout_path=str(tmp_path / "missing-tool"),
            )
        ],
    )
    blocked = cli.normalize(value)
    assert blocked["honest_verdict"] == "complete_blocked_required_tools"
    assert not blocked["gate_check_summary"][-1]["passed"]
    monkeypatch.setattr(cli.qualified, "commands", lambda p: [])
    monkeypatch.setattr(cli.qualified, "validators", lambda p: [])
    monkeypatch.setattr(h, "load", lambda root, raw: panel(True))
    monkeypatch.setattr(cli.qualified, "execute", lambda plan, raw: [])
    output = tmp_path / (h.NAME + ".json")
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    original = output.read_bytes()
    monkeypatch.setattr(
        cli.qualified,
        "execute",
        lambda plan, raw: [
            dict(
                passed=False, normal_exit=True, actual_exit=1, stdout_path=str(tmp_path / "failed")
            )
        ],
    )
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 1
    assert output.read_bytes() == original
