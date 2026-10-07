"""REQ-REPORT-8230 / REQ-VERIFY-8230: arithmetic cannot establish hardware benefit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import kv260_workload_boundary_8230 as h
from carnot.reporting import kv260_boundary_execution_8230 as cli
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.primary_publication import publish_primary


def panel():
    """Private boundary cases keep original permission and missing units visible."""
    group = dict(name="global", interval=None, reject_only=False)
    model = dict(kind="patch", base=dict(kind="input"), patches=[dict(group=group, delta=0.05)])
    rows = [
        dict(unit_id=str(i), source_cluster_id=str(i), p=p, baseline_p=p, baseline_action=a)
        for i, (p, a) in enumerate(
            [(0.04, "accept"), (0.05, "reject"), (0.5, "reject"), (None, "escalate")]
        )
    ]
    return dict(
        checks=[],
        references=[],
        cited=[],
        branches={},
        board={},
        requests=[],
        cases=[dict(scope="fixture", arm="frozen", model=model, rows=rows)],
        fixture=True,
    )


def test_fixed_lookup_order_and_unsupported():
    """SCENARIO-VERIFY-8230-PRECISION: integer clipping preserves each add."""
    data = panel()
    model = data["cases"][0]["model"]
    row = dict(data["cases"][0]["rows"][0], p=0.99)
    model["patches"] = [dict(group=model["patches"][0]["group"], delta=d) for d in (0.05, -0.05)]
    q, unsupported = h.qpredict(model, row)
    assert q == (65535 - round(0.05 * 65536)) / 65536
    assert not unsupported and q < 0.96
    g = dict(name="bin", interval=[0.1, 0.25], reject_only=True)
    model["patches"] = [dict(group=g, delta=0.05)]
    assert (
        h.qpredict(model, dict(row, baseline_p=0.1, baseline_action="reject"))[0] == 65535 / 65536
    )
    assert h.qpredict(model, dict(row, baseline_p=0.25))[0] == round(0.99 * 65536) / 65536
    g["interval"] = [0.75, 1]
    assert h.qpredict(model, dict(row, baseline_p=1, baseline_action="reject"))[0] == 65535 / 65536
    mixture = dict(kind="mixture", base=dict(kind="input"), candidate=model, step=0.5)
    assert h.qpredict(mixture, row)[1]
    with pytest.raises(ValueError, match="model_kind"):
        h.qpredict(dict(kind="wrong", base=dict(kind="input")), row)
    with pytest.raises(ValueError, match="probability"):
        h.qpredict(dict(kind="input"), dict(row, p=2))
    assert h.qpredict(model, dict(row, p=None)) == (None, False)


def test_precision_and_readiness():
    """REQ-VERIFY-8230: fallback and fixture/natural denominators stay separate."""
    data = panel()
    rows = h.precision(data)
    assert len(rows) == 4 and rows[-1]["status"] == "excluded"
    assert rows[1]["fallback"] and rows[1]["final_action"] != "accept"
    value = h.reduce(data, rows)
    assert value["kv260_boundary_ready_score"] == 0  # no authenticated historical fabric
    data["board"] = dict(custody_valid=True, k_max=5)
    data["branches"] = dict(kernel=dict(ready=True), service=dict(ready=False))
    value = h.reduce(data, rows)
    assert value["kv260_boundary_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["whole_request_bounds"][0]["ideal_upper_bound"] is None
    assert all(not m["existing_fabric_supported"] for m in value["operation_mapping"])
    assert value["independent_generalization_score"] == 0
    data["cases"][0]["scope"] = "natural_public_fixture_model"
    assert h.precision(data)[0]["scope"] == "natural_public_fixture_model"
    rows[0]["numerator"] = 1
    assert h.reduce(data, rows)["verdict_class"] == "disqualified"


def test_request_bounds_missing_and_practical():
    """SCENARIO-VERIFY-8230-BOUNDS: transfers and durability remain in total."""
    assert h.bounds([])[0]["status"] == "unavailable"
    r = dict(
        unit_id="q",
        source_cluster_id="s",
        total_ns=1000,
        lookup_add_ns=10,
        acquisition_ns=900,
        transfer_ns=10,
        readout_ns=10,
        durable_host_ns=20,
        unsupported_cpu_ns=50,
        device_lookup_add_ns=2,
        board_transfer_ns=3,
        board_readout_ns=4,
    )
    b = h.bounds([r])[0]
    assert b["ideal_upper_bound"] == pytest.approx(1000 / 990)
    assert b["practical_estimate"] == pytest.approx(1000 / 999)
    assert b["existing_fabric_upper_bound"] == 1
    del r["board_transfer_ns"]
    assert h.bounds([r])[0]["practical_estimate"] is None
    r["lookup_add_ns"] = None
    assert h.bounds([r])[0]["ideal_upper_bound"] is None
    r["lookup_add_ns"] = 11
    assert h.bounds([r])[0]["status"] == "unavailable"


def primary(root, branch, **extra):
    """Private producers use the unchanged publisher's location and byte checks."""
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
    publication = publish_primary(path, value, lambda p: dict(passed=True))
    # Sidecar location is named by a separate terminal receipt, as in real producers.
    terminal = root / "terminal" / (branch + ".json")
    atomic_json(terminal, dict(publication=publication))
    value["terminal_validation_sidecar_path"] = str(terminal)
    publish_primary(path, value, lambda p: dict(passed=True))
    publication["primary_sha256"] = reference(path)["sha256"]
    publication["sidecar_path"] = str(
        path.parent
        / "raw"
        / path.stem
        / "validators"
        / (publication["primary_sha256"].split(":")[1] + ".json")
    )
    atomic_json(terminal, dict(publication=publication))
    return path


def test_authentication_separate_branches(tmp_path, monkeypatch):
    """REQ-REPORT-8230: disqualified learning never gates historical KV260."""
    data = h.load(tmp_path, tmp_path / "raw")
    assert len(data["branches"]) == 4 and not data["cases"]
    assert all(c["hash"] is None for c in data["checks"] if c["artifact_field"] == "exists")
    transcript = tmp_path / "transcript.json"
    atomic_json(transcript, dict(scope="historical quadratic fabric"))
    board = dict(
        board="KV260",
        custody_valid=True,
        k_max=5,
        source_transcript=str(transcript),
        source_transcript_sha256=reference(transcript)["sha256"],
    )
    historic = primary(tmp_path, "historical", board_rows=[board])
    work = tmp_path / "work.json"
    atomic_json(
        work,
        dict(
            static=dict(model=panel()["cases"][0]["model"]),
            protocol=dict(public_fit_membership=panel()["cases"][0]["rows"]),
        ),
    )
    restart = tmp_path / "restart-input.json"
    atomic_json(restart, dict(rows=panel()["cases"][0]["rows"]))
    kernel = primary(
        tmp_path,
        "kernel",
        measurement_reference=reference(work),
        raw_shard_hashes=[reference(restart)],
    )
    learning = primary(tmp_path, "learning", fixture_mode=True)
    service = primary(tmp_path, "service", whole_request_spans=[])
    monkeypatch.setattr(
        h, "PINS", {8216: reference(historic)["sha256"], 8221: reference(kernel)["sha256"]}
    )
    data = h.load(tmp_path, tmp_path / "raw2")
    assert data["board"] == board and len(data["cases"]) == 2
    assert not data["branches"]["learning"]["ready"]
    assert data["branches"]["service"]["ready"]
    value = json.loads(service.read_bytes())
    value["experiment_id"] = 0
    atomic_json(service, value)
    data = h.load(tmp_path, tmp_path / "raw3")
    assert not data["branches"]["service"]["ready"]
    assert data["branches"]["historical"]["ready"]
    work.write_text("{}")
    assert not h.load(tmp_path, tmp_path / "raw4")["branches"]["kernel"]["ready"]
    kernel.write_text("[]")
    assert not h.load(tmp_path, tmp_path / "raw5")["branches"]["kernel"]["ready"]
    assert learning.exists()


def invoke(*args):
    """Run script-path children from outside the checkout with no ambient imports."""
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


def test_real_cli_replay_failures(tmp_path):
    """SCENARIO-REPORT-8230-CLI: checked bytes replay and rehashed edits fail."""
    source = tmp_path / "fixture.json"
    atomic_json(source, panel())
    output = tmp_path / (h.NAME + ".json")
    run = invoke("--date", "20261007", "--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["MODEL_SPECS"] == [] and value["current_model_calls"] == 0
    assert invoke("--cold-replay", output).returncode == 0
    original = deepcopy(value)
    value["precision_rows"][0]["numerator"] = 1
    value["reproducibility_checksum"] = cli.checksum(value)
    atomic_json(output, value)
    assert invoke("--cold-replay", output).returncode == 1
    value = deepcopy(original)
    value["config"]["seed"] = 0
    atomic_json(output, value)
    assert invoke("--cold-replay", output).returncode == 1
    value = deepcopy(original)
    value["reproducibility_checksum"] = "bad"
    atomic_json(output, value)
    assert invoke("--cold-replay", output).returncode == 1
    value = deepcopy(original)
    value["primitive_reference"]["sha256"] = "bad"
    atomic_json(output, value)
    assert invoke("--cold-replay", output).returncode == 1
    assert invoke("--date", "bad").returncode == 2
    assert (
        invoke("--input", source, "--output", h.ROOT / "results" / (h.NAME + ".json")).returncode
        == 1
    )
    assert invoke("--input", tmp_path / "absent", "--output", output).returncode == 1
    primitive_path = Path(original["primitive_reference"]["path"])
    primitive = json.loads(primitive_path.read_bytes())
    primitive["precision_rows"][0]["numerator"] = 1
    atomic_json(primitive_path, primitive)
    value = deepcopy(original)
    value["primitive_reference"] = reference(primitive_path)
    value["raw_shard_hashes"] = [
        reference(primitive_path) if r["path"] == str(primitive_path) else r
        for r in value["raw_shard_hashes"]
    ]
    value["reproducibility_checksum"] = cli.checksum(value)
    atomic_json(output, value)
    with pytest.raises(ValueError, match="primitive_drift"):
        cli.replay(output)


def test_runner_owned_failures_and_plans(tmp_path, monkeypatch):
    """REQ-REPORT-8230: owned failures zero readiness and reject unsafe publication."""
    plans = cli.commands(tmp_path / "plan")
    assert any(s.name == "e2e015" for s in plans)
    assert any("--strict" in s.argv for s in plans if s.name == "changed_module_mypy")
    assert len(cli.validators(tmp_path / "candidate.json")) == 3
    monkeypatch.setattr(cli, "commands", lambda p: [])
    monkeypatch.setattr(cli, "validators", lambda p: [])
    monkeypatch.setattr(h, "load", lambda r, raw: panel())
    monkeypatch.setattr(cli, "execute", lambda plan, raw: [])
    output = tmp_path / (h.NAME + ".json")
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    monkeypatch.setattr(cli, "execute", lambda plan, raw: [dict(passed=False, normal_exit=True)])
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 1
    with patch.object(
        cli, "execute", side_effect=[[], [dict(passed=False, normal_exit=True)], [], [], []]
    ):
        assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["kv260_boundary_ready_score"] == 0
    assert cli.replay(output)["passed"]
    assert canonical_hash(value)
