"""REQ-REPORT-8244 / REQ-VERIFY-8244: private evidence bounds CPU arithmetic."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import kv260_decision_boundary_8244 as h
from carnot.reporting import kv260_decision_execution_8244 as cli
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.primary_publication import publish_primary


def panel():
    """Private threshold crossings reveal raw rounding while permissions stay fixed."""
    head = dict(
        arm="logistic_margin",
        basis="logistic",
        weights=[math.log(0.1 / 0.9), 0.03] + [0.0] * 14 + [4.0],
        temperature=1.0,
        geometry=dict(mean=[0.0] * 16, scale=[1.0] * 16),
    )
    samples = [
        dict(
            unit_id=str(i),
            source_cluster_id="public" + str(i // 2),
            x=[x] + [0.0] * 15 if x is not None else None,
            p0=0.1,
            baseline_action=a,
            y=y,
            role="calibration",
        )
        for i, (x, a, y) in enumerate(
            [(0.0, "accept", 1), (0.3, "reject", 0), (-0.3, "accept", 0), (None, "escalate", None)]
        )
    ]
    request = dict(
        request_id="q",
        source_cluster_id="s",
        arm="serial",
        workload="s1_serial",
        status="completed",
        clocks=dict(issue=100, durability=1000000100),
        acquisition_s=0.8,
        scoring_s=0.01,
        service_s=0.1,
        queue_s=0.05,
        issue_fsync_s=0.01,
        terminal_fsync_s=0.01,
    )
    return dict(
        checks=[],
        references=[],
        cited=[],
        branches=dict(historical=True, heads=True, service=True),
        board=dict(custody_valid=True, k_max=5),
        heads=[head],
        samples=samples,
        requests=[request],
        cold_costs=[dict(workload="s1_serial", startup_s=2.0, shutdown_s=0.1)],
        fixture=True,
    )


def test_precision_permission_and_source_counts():
    """SCENARIO-VERIFY-8244-PRECISION: errors and raw changes survive fallback."""
    data = panel()
    rows = h.precision(data)
    assert len(rows) == 8 and sum(r["status"] == "excluded" for r in rows) == 2
    complete = [r for r in rows if r["status"] == "completed"]
    assert {r["bits"] for r in rows} == {8, 16}
    assert any(r["raw_action_changed"] for r in complete)
    assert all(r["final_action_changed"] == 0 for r in complete)
    assert all(r["probability_error"] <= r["probability_error_bound"] for r in complete)
    assert all(r["final_action"] != "accept" or r["baseline_action"] == "accept" for r in complete)
    value = h.reduce(data, rows)
    assert value["verdict_class"] == "circular_positive"
    assert value["independent_count"] == 0 and value["kv260_boundary_ready_score"] == 1
    assert len(value["precision_source_summaries"]) == 4
    data["branches"]["service"] = False
    data["requests"] = []
    value = h.reduce(data, rows)
    assert value["verdict_class"] == "blocked" and value["kv260_boundary_ready_score"] == 1
    assert value["kv260_obligation"]["historical"] == data["board"]
    data["fixture"] = False
    data["branches"]["service"] = True
    data["requests"] = panel()["requests"]
    assert h.reduce(data, rows)["verdict_class"] == "null"
    rows[0]["final_action_changed"] = 1
    assert h.reduce(data, rows)["verdict_class"] == "disqualified"
    with pytest.raises(ValueError, match="head_schema"):
        h.validate_heads([dict(data["heads"][0], temperature=0)])
    h.validate_heads(data["heads"])


def test_whole_request_costs():
    """SCENARIO-VERIFY-8244-COSTS: retain all work and distinguish absent clocks."""
    data = panel()
    costs, bound = h.costs(data)
    assert costs[0]["maximum_gain"] == pytest.approx(1 / 0.99)
    assert bound["cold_inclusive_work_sum_gain"] == pytest.approx(3.1 / 3.09)
    assert bound["existing_fabric_gain"] == 1.0
    for change in [
        dict(scoring_s=None),
        dict(scoring_s=-1),
        dict(scoring_s=True),
        dict(scoring_s=1.0),
        dict(status="error"),
        dict(clocks={}),
        dict(acquisition_s=2.0),
    ]:
        data["requests"] = [dict(panel()["requests"][0], **change)]
        assert h.costs(data)[0][0]["maximum_gain"] is None
    data["requests"] = []
    assert h.costs(data)[1]["maximum_request_gain"] is None
    data = panel()
    data["cold_costs"] = []
    assert h.costs(data)[1]["cold_inclusive_work_sum_gain"] is None


def primary(root, branch, **extra):
    """Unchanged publication creates hash-bound private upstream receipts."""
    eid, suffix, score = h.PRODUCERS[branch]
    path = root / "results" / f"experiment_{eid}_{suffix}.json"
    value = dict(
        experiment_id=eid,
        task_id=f"exp{eid}-" + suffix.split("_", 1)[1].replace("_", "-"),
        honest_verdict="complete_null_fixture",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
        **{score: 1},
        **extra,
    )
    publish_primary(path, value, lambda p: dict(passed=True))
    return path


def test_authentication_independent_branches(tmp_path, monkeypatch):
    """REQ-REPORT-8244: missing heads cannot erase authenticated board history."""
    data = h.load(tmp_path, tmp_path / "raw")
    assert not any(data["branches"].values())
    transcript = tmp_path / "historical.json"
    atomic_json(transcript, dict(scope="historical quadratic fabric"))
    historical = dict(
        panel()["board"],
        source_path="historical.json",
        source_hash=reference(transcript)["sha256"],
        source_transcript=str(transcript),
        source_transcript_sha256=reference(transcript)["sha256"],
    )
    board = primary(tmp_path, "historical", kv260_obligation=dict(historical=historical))
    heads = tmp_path / "heads.json"
    work = tmp_path / "work.json"
    atomic_json(heads, dict(schema="carnot.v712.margin-heads.v1", heads=panel()["heads"]))
    atomic_json(work, dict(public_rows=panel()["samples"], fitted=dict(heads=panel()["heads"])))
    head_primary = primary(
        tmp_path,
        "heads",
        trained_heads_path=str(heads),
        trained_heads_sha256=reference(heads)["sha256"],
        work_reference=reference(work),
    )
    requests = tmp_path / "request_rows.json"
    atomic_json(requests, dict(rows=panel()["requests"]))
    service = primary(
        tmp_path,
        "service",
        request_rows_path=str(requests),
        raw_shard_hashes=[reference(requests)],
        rows=panel()["requests"],
        cold_costs=panel()["cold_costs"],
    )
    monkeypatch.setattr(
        h,
        "PINS",
        {
            8230: reference(board)["sha256"],
            8237: reference(head_primary)["sha256"],
            8242: reference(service)["sha256"],
        },
    )
    data = h.load(tmp_path, tmp_path / "raw2")
    assert all(data["branches"].values()) and len(data["heads"]) == 1
    heads.write_text("{}")
    data = h.load(tmp_path, tmp_path / "raw3")
    assert not data["branches"]["heads"] and data["branches"]["historical"]
    mismatch = next(c for c in data["checks"] if c["artifact_field"] == "trained_heads_sha256")
    assert not mismatch["passed"] and mismatch["path"] == str(heads)
    assert mismatch["expected"] != mismatch["observed"]
    service.write_text("[]")
    assert not h.load(tmp_path, tmp_path / "raw4")["branches"]["service"]
    primary(
        tmp_path,
        "heads",
        trained_heads_path=str(heads),
        trained_heads_sha256=reference(heads)["sha256"],
        work_reference=reference(work),
    )
    h.PINS[8237] = reference(head_primary)["sha256"]
    atomic_json(heads, dict(schema="wrong", heads=panel()["heads"]))
    primary(
        tmp_path,
        "heads",
        trained_heads_path=str(heads),
        trained_heads_sha256=reference(heads)["sha256"],
        work_reference=reference(work),
    )
    h.PINS[8237] = reference(head_primary)["sha256"]
    assert not h.load(tmp_path, tmp_path / "raw5")["branches"]["heads"]
    service.unlink()
    primary(
        tmp_path,
        "service",
        request_rows_path=str(requests),
        raw_shard_hashes=[reference(requests)],
        rows=[],
        cold_costs=panel()["cold_costs"],
    )
    h.PINS[8242] = reference(service)["sha256"]
    assert not h.load(tmp_path, tmp_path / "raw6")["branches"]["service"]
    primary(tmp_path, "historical", kv260_obligation=dict(historical=dict(historical, k_max=6)))
    h.PINS[8230] = reference(board)["sha256"]
    assert not h.load(tmp_path, tmp_path / "raw7")["branches"]["historical"]
    primary(tmp_path, "historical", kv260_obligation=dict(historical=historical), fixture_mode=True)
    h.PINS[8230] = reference(board)["sha256"]
    assert not h.load(tmp_path, tmp_path / "raw8")["branches"]["historical"]


def invoke(*args):
    """Real children run outside the checkout so replay does not rely on ambient imports."""
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


def test_real_cli_and_tampering(tmp_path):
    """SCENARIO-REPORT-8244-CLI: fresh processes reject even rehashed evidence edits."""
    source = tmp_path / "fixture.json"
    output = tmp_path / (h.NAME + ".json")
    atomic_json(source, panel())
    run = invoke("--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    original = json.loads(output.read_bytes())
    assert original["MODEL_SPECS"] == [] and original["current_device_execution_count"] == 0
    assert invoke("--cold-replay", output).returncode == 0
    for key, replacement in [
        ("config", {}),
        ("reproducibility_checksum", "wrong"),
        ("precision_rows", []),
        ("kv260_boundary_ready_score", 0),
    ]:
        value = deepcopy(original)
        value[key] = replacement
        if key != "reproducibility_checksum":
            value["reproducibility_checksum"] = cli.checksum(value)
        atomic_json(output, value)
        assert invoke("--cold-replay", output).returncode == 1
    primitive_path = Path(original["primitive_reference"]["path"])
    atomic_json(primitive_path, dict(precision_rows=[]))
    value = deepcopy(original)
    value["primitive_reference"] = reference(primitive_path)
    value["raw_shard_hashes"] = [
        reference(primitive_path) if r["path"] == str(primitive_path) else r
        for r in value["raw_shard_hashes"]
    ]
    value["reproducibility_checksum"] = cli.checksum(value)
    atomic_json(output, value)
    assert "primitive_drift" in invoke("--cold-replay", output).stdout
    assert invoke("--date", "bad").returncode == 2
    assert invoke("--input", source, "--output", h.ROOT / "results" / output.name).returncode == 1
    assert invoke("--input", tmp_path / "missing", "--output", output).returncode == 1
    blocked = invoke("--root", tmp_path / "missing-root", "--output", output, "--fixture-e2e")
    assert blocked.returncode == 0, blocked.stdout + blocked.stderr
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"


def test_owned_failure_and_frozen_plan(tmp_path, monkeypatch):
    """REQ-REPORT-8244: failed owned checks zero readiness without hiding evidence."""
    plan = cli.commands(tmp_path / "plan")
    assert {s.name for s in plan} >= {"e2e015", "e2e019", "coverage_combine"}
    assert all("--strict" in s.argv for s in plan if s.name == "changed_module_mypy")
    assert len(cli.validators(tmp_path / "candidate.json")) == 3
    monkeypatch.setattr(cli, "commands", lambda p: [])
    monkeypatch.setattr(cli, "validators", lambda p: [])
    monkeypatch.setattr(h, "load", lambda root, raw: panel())
    monkeypatch.setattr(cli, "execute", lambda plan, raw: [])
    output = tmp_path / (h.NAME + ".json")
    assert cli.main(["--output", str(output)]) == 0
    monkeypatch.setattr(cli, "execute", lambda plan, raw: [dict(passed=False, normal_exit=True)])
    assert cli.main(["--output", str(output)]) == 1
    from unittest.mock import patch

    with patch.object(
        cli, "execute", side_effect=[[], [dict(passed=False, normal_exit=True)], [], [], []]
    ):
        assert cli.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["kv260_boundary_ready_score"] == 0
    assert cli.replay(output)["passed"]
    assert canonical_hash(value)
