"""REQ-REPORT-8231 / REQ-VERIFY-8231: host parity grants no device benefit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import polarfire_state_boundary_8231 as h
from carnot.reporting import polarfire_boundary_execution_8231 as cli
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.evidence_features_custody_7980 import reference
from test_kv260_workload_boundary_8230 import primary


def panel():
    """Private ordered clipping and exact mixtures exercise the portable tree."""
    group = dict(name="global", interval=None, reject_only=False)
    patch_model = dict(
        kind="patch",
        base=dict(kind="input"),
        patches=[dict(group=group, delta=0.05), dict(group=group, delta=-0.05)],
    )
    model = dict(kind="mixture", base=dict(kind="input"), candidate=patch_model, step=0.5)
    rows = [
        dict(unit_id=str(i), source_cluster_id=str(i), p=p, baseline_p=p, baseline_action=a)
        for i, (p, a) in enumerate([(0.99, "accept"), (0.09, "reject"), (None, "escalate")])
    ]
    return dict(
        checks=[],
        references=[],
        cited=[],
        branches={},
        fixture=True,
        board=dict(custody_valid=True, actual_substrate="linux_cpu"),
        cases=[
            dict(
                scope="private_fixture",
                arm="mixture",
                state=dict(schema_version=1, model=model, pending=[1], rng_state=[1, 2]),
                rows=rows,
            )
        ],
    )


def test_envelope_order_versions_and_inventory(tmp_path):
    """SCENARIO-VERIFY-8231-STATE: preserve order, missing values and exact mixtures."""
    state = panel()["cases"][0]["state"]
    payload = h.encode(state)
    assert h.decode(payload) == state
    inventory = h.inventory(state)
    assert [p["delta"] for p in inventory["ordered_corrections"]] == [0.05, -0.05]
    assert inventory["exact_mixtures"][0]["step"] == 0.5
    assert inventory["global_parameters"]["clip_bounds"] == [1e-6, 1 - 1e-6]
    assert "rng_state" in inventory["restart_fields"]
    for field, value, error in [
        ("version", 2, "version"),
        ("payload_sha256", "bad", "payload_hash"),
    ]:
        changed = json.loads(payload)
        changed[field] = value
        with pytest.raises(ValueError, match=error):
            h.decode(json.dumps(changed).encode())
    invalid = dict(state, schema_version=2)
    with pytest.raises(ValueError, match="state_version"):
        h.encode(invalid)
    with pytest.raises(ValueError, match="model_kind"):
        h.inventory(dict(model=dict(kind="unsupported")))
    measured = h.measure(panel(), tmp_path)
    assert len(measured) == 1 and measured[0]["parity"]
    assert measured[0]["serialized_bytes"] == len(payload)
    assert measured[0]["predictions"][-1]["p"] is None
    assert measured[0]["predictions"][0]["p"] < 0.99
    assert measured[0]["host_storage"]["bytes"] == len(payload)


def test_reduction_contract_and_external_blocks(tmp_path):
    """SCENARIO-VERIFY-8231-CONTRACT: symbolic costs never become board timings."""
    data = panel()
    measured = h.measure(data, tmp_path)
    value = h.reduce(data, measured)
    assert value["verdict_class"] == "circular_positive"
    assert value["polarfire_boundary_ready_score"] == 1
    contract = h.contract()
    assert contract["measured_board_bandwidth_bytes_s"] is None
    assert "directory_fsync" in contract["durable_commit_steps"]
    data["branches"]["learning"] = dict(
        ready=False,
        verdict="complete_disqualified_owned_validation",
        failed_operands=[dict(artifact_field="ready", observed=0)],
    )
    value = h.reduce(data, measured)
    assert value["verdict_class"] == "blocked" and value["polarfire_boundary_ready_score"] == 1
    assert value["blocked_obligations"][0]["verdict"] == "complete_disqualified_owned_validation"
    measured[0]["parity"] = False
    assert h.reduce(data, measured)["verdict_class"] == "disqualified"
    assert h.reduce(data, measured)["polarfire_boundary_ready_score"] == 0
    data["cases"] = []
    data["board"] = {}
    assert h.reduce(data, [])["verdict_class"] == "blocked"


def invoke(*args):
    """Real child statements run outside the checkout without ambient imports."""
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


def test_cli_replay_and_failures(tmp_path):
    """SCENARIO-REPORT-8231-CLI: rehashing changes cannot excuse altered primitives."""
    source, output = tmp_path / "fixture.json", tmp_path / (h.NAME + ".json")
    atomic_json(source, panel())
    run = invoke("--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    original = json.loads(output.read_bytes())
    assert original["current_model_calls"] == 0 and original["MODEL_SPECS"] == []
    assert invoke("--cold-replay", output).returncode == 0
    with patch.object(h, "contract", return_value=dict(version=2)):
        with pytest.raises(ValueError, match="contract_drift"):
            cli.replay(output)
    for key, replacement in [
        ("config", {}),
        ("reproducibility_checksum", "bad"),
        ("completed_count", 999),
    ]:
        value = deepcopy(original)
        value[key] = replacement
        if key != "reproducibility_checksum":
            value["reproducibility_checksum"] = cli.checksum(value)
        atomic_json(output, value)
        assert invoke("--cold-replay", output).returncode == 1
    value = deepcopy(original)
    primitive_path = Path(value["primitive_reference"]["path"])
    primitive = json.loads(primitive_path.read_bytes())
    primitive["serialized_state_rows"][0]["serialized_bytes"] += 1
    atomic_json(primitive_path, primitive)
    value["primitive_reference"] = reference(primitive_path)
    value["raw_shard_hashes"] = [
        reference(primitive_path) if r["path"] == str(primitive_path) else r
        for r in value["raw_shard_hashes"]
    ]
    value["reproducibility_checksum"] = cli.checksum(value)
    atomic_json(output, value)
    assert invoke("--cold-replay", output).returncode == 1
    assert invoke("--date", "bad").returncode == 2
    assert invoke("--input", source, "--output", h.ROOT / "results" / output.name).returncode == 1
    assert invoke("--input", tmp_path / "missing", "--output", output).returncode == 1


def test_authenticate_branch_inventory(tmp_path, monkeypatch):
    """REQ-REPORT-8231: kernel and historical CPU survive a disqualified sibling."""
    assert not h.load(tmp_path, tmp_path / "empty")["cases"]
    transcript = tmp_path / "transcript.json"
    atomic_json(transcript, dict(scope="historical linux CPU"))
    board = dict(
        board="PolarFire",
        custody_valid=True,
        actual_substrate="linux_cpu",
        source_path=str(transcript),
        source_hash=reference(transcript)["sha256"],
        source_transcript=str(transcript),
        source_transcript_sha256=reference(transcript)["sha256"],
    )
    historic = primary(tmp_path, "historical", board_rows=[board])
    work = tmp_path / "work.json"
    state = panel()["cases"][0]["state"]
    atomic_json(work, dict(static=dict(model=state["model"])))
    restart = tmp_path / "restart-input.json"
    atomic_json(restart, dict(rows=panel()["cases"][0]["rows"]))
    final = tmp_path / "uninterrupted/final.json"
    atomic_json(final, state)
    kernel = primary(
        tmp_path,
        "kernel",
        measurement_reference=reference(work),
        raw_shard_hashes=[reference(restart), reference(final)],
    )
    learning = primary(tmp_path, "learning", fixture_mode=True)
    monkeypatch.setattr(
        h, "PINS", {8216: reference(historic)["sha256"], 8221: reference(kernel)["sha256"]}
    )
    data = h.load(tmp_path, tmp_path / "raw1")
    assert data["board"] == board and len(data["cases"]) == 2
    assert not data["branches"]["learning"]["ready"]
    states = tmp_path / "final_states.json"
    atomic_json(states, [dict(learned=state)])
    learning = primary(
        tmp_path,
        "learning",
        final_states_path=str(states),
        raw_shard_hashes=[reference(states), reference(restart)],
    )
    data = h.load(tmp_path, tmp_path / "raw2")
    assert data["branches"]["learning"]["ready"]
    assert data["cases"][-1]["scope"] == "natural_learning"
    work.write_text("{}")
    assert not h.load(tmp_path, tmp_path / "raw3")["branches"]["kernel"]["ready"]
    kernel.write_text("[]")
    assert not h.load(tmp_path, tmp_path / "raw4")["branches"]["kernel"]["ready"]
    transcript.write_text("{}")
    assert not h.load(tmp_path, tmp_path / "raw5")["branches"]["historical"]["ready"]
    assert learning.exists()


def test_inventory_pending_candidate_and_missing(tmp_path):
    """SCENARIO-VERIFY-8231-STATE: pending candidates and clock defects stay visible."""
    data = panel()
    state = data["cases"][0]["state"]
    state["candidate"] = dict(model=dict(kind="global", scale=1.25, intercept=-0.1))
    assert any(
        r["path"] == "candidate.model"
        for r in h.inventory(state)["global_parameters"]["fit_parameters"]
    )
    rows = h.measure(data, tmp_path)
    h.verify_primitives(data, rows)
    with pytest.raises(ValueError, match="primitive_count"):
        h.verify_primitives(data, [])
    rows[0]["encode_ns"] += 1
    with pytest.raises(ValueError, match="clock_drift"):
        h.verify_primitives(data, rows)
    state["model"] = state["candidate"]["model"]
    assert h.measure(data, tmp_path)[0]["parity"]


def test_runner_owned_failures_and_plans(tmp_path, monkeypatch):
    """REQ-REPORT-8231: owned failed exits zero readiness before publication."""
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
    with patch.object(
        cli, "execute", side_effect=[[], [dict(passed=False, normal_exit=True)], [], [], []]
    ):
        assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["polarfire_boundary_ready_score"] == 0
    assert cli.replay(output)["passed"]
    monkeypatch.setattr(cli, "execute", lambda plan, raw: [dict(passed=False, normal_exit=True)])
    assert cli.main(["--root", str(tmp_path), "--output", str(output)]) == 1
