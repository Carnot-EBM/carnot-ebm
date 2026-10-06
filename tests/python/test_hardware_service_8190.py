"""REQ-REPORT-8190, REQ-VERIFY-8190: private evidence cannot buy fabric credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import runpy
import subprocess

import pytest

from carnot.reporting import hardware_service_8190 as h
from carnot.reporting import hardware_service_inputs_8190 as inputs
from carnot.reporting import hardware_service_execution_8190 as cli
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.primary_publication import publish_primary
from test_hardware_workload_8162 import panel as old_panel


def panel():
    """Known clocks expose dropped storage and double-counted acquisition."""
    previous = old_panel()
    components = dict(
        generation_ns=50,
        key_hash_ns=5,
        cache_read_ns=5,
        cache_write_ns=5,
        native_crossing_including_arithmetic_ns=10,
        decision_write_ack_ns=15,
        queue_ns=0,
        load_ns=0,
        parse_ns=1,
        features_ns=1,
        decision_serialization_ns=1,
        other_host_and_ack_ns=7,
    )
    requests = []
    for condition in h.CONDITIONS:
        r = deepcopy(previous["branches"]["exp8159"]["pairs"][0]["arms"][0]["requests"][0])
        r.update(
            unit_id=condition,
            arm=condition,
            condition=condition,
            status="completed",
            numerator=100,
            denominator=1,
            exclusion_reason=None,
        )
        requests.append(dict(request=r, components=dict(components), qualified=True))
    return dict(
        fixture=True,
        checks=[],
        references=[],
        cited=[],
        boards=previous["boards"],
        state=previous["state"],
        trained_head_specs=[],
        requests=requests,
        service_score=1,
        service_qualified=True,
        historical_context={},
        startup_ns=96,
        source=dict(path="private", hash="private"),
    )


def test_separate_branches_and_storage():
    """SCENARIO-VERIFY-8190-COST: mixed clocks never become measured arithmetic."""
    data = panel()
    value = h.reduce(data)
    assert value["verdict_class"] == "circular_positive"
    assert (
        value["hardware_boundary_ready_score"] == value["measured_workload_bound_ready_score"] == 1
    )
    assert {r["condition"] for r in value["workload_rows"]} == set(h.CONDITIONS)
    for row in value["workload_rows"]:
        assert row["numerator"] == 100 and row["denominator"] == 90
        assert row["outer_ceiling"] == pytest.approx(100 / 90)
        assert row["measured_arithmetic_fraction"] is None
        assert row["retained_components"]["generation_ns"] == 50
        assert row["retained_components"]["decision_write_ack_ns"] == 15
    assert {r["bits"] for r in value["quantization_rows"]} == {8, 12, 16, 64}
    assert all(r["measured_incremental_fallback_ns"] is None for r in value["fallback_cost_rows"])
    assert all(b["required_arithmetic_fraction_for_100x"] == 0.99 for b in value["amdahl_bounds"])
    assert value["measured_repeat_frequency"] is None
    data["fixture"] = False
    assert h.reduce(data)["independent_count"] == 1
    data["requests"][0]["components"]["arithmetic_ns"] = 2
    assert h.reduce(data)["workload_rows"][0]["exact_arithmetic_only_ceiling"] == pytest.approx(
        100 / 98
    )


@pytest.mark.parametrize(
    "failure", ["missing", "negative", "sum", "board", "score", "empty", "unqualified"]
)
def test_preserve_qualified_scopes(failure):
    """SCENARIO-REPORT-8190-INDEPENDENT: one failed operand preserves other work."""
    data = panel()
    if failure == "missing":
        data["requests"][0]["components"].pop("cache_write_ns")
    elif failure == "negative":
        data["requests"][0]["components"]["other_host_and_ack_ns"] = -1
    elif failure == "sum":
        data["requests"][0]["components"]["complete_latency_ns"] = 101
    elif failure == "board":
        data["boards"][0]["k_max"] = 6
    elif failure == "score":
        data.update(service_score=None, service_qualified=False)
    elif failure == "empty":
        data["requests"] = []
    else:
        data["requests"][0]["qualified"] = False
    value = h.reduce(data)
    assert value["verdict_class"] == "blocked"
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert any(not c["passed"] for c in value["gate_check_summary"])
    if failure in {"missing", "negative", "sum", "unqualified", "board"}:
        assert value["measured_workload_bound_ready_score"] == 1
        assert value["board_rows"][1]["custody_valid"]


def invoke(args, tmp_path, expected=0):
    """Use the installed script path so ambient imports cannot hide transport bugs."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    prefix = [str(cli.ROOT / ".venv/bin/python")]
    if env.get("CARNOT_8190_COVERAGE_CONFIG"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + env["CARNOT_8190_COVERAGE_CONFIG"]]
    result = subprocess.run(
        prefix + [str(cli.ROOT / cli.SCRIPT), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    return result


def test_private_cli_success_missing_tamper_cold(tmp_path):
    """REQ-REPORT-8190: E2E-016 private publication and replay preserve exact bytes."""
    source, output = tmp_path / "input.json", tmp_path / (cli.NAME + ".json")
    atomic_json(source, panel())
    invoke(["--input", str(source), "--output", str(output)], tmp_path)
    invoke(["--cold-replay", str(output)], tmp_path)
    v = json.loads(output.read_bytes())
    assert v["required_checks_passed"] and v["MODEL_SPECS"] == []
    assert v["model_invocation_counts"]["model_loads_attempted"] == 0
    v["workload_rows"][0]["numerator"] += 1
    atomic_json(output, v)
    invoke(["--cold-replay", str(output)], tmp_path, 1)
    data = panel()
    data["requests"][0]["components"].pop("cache_write_ns")
    atomic_json(source, data)
    invoke(["--input", str(source), "--output", str(output)], tmp_path)
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    invoke(["--input", str(tmp_path / "missing"), "--output", str(output)], tmp_path, 1)
    invoke(["--input", str(source)], tmp_path, 1)


def published(path, value):
    """Make a genuine private byte-bound publication instead of a pretend signature."""
    value.update(
        task_id=f"exp{value['experiment_id']}-private",
        honest_verdict="complete_null_private",
        verdict_class="null",
        required_checks_passed=True,
        flagged_adversarial=False,
    )
    terminal = path.parent / "raw" / path.stem / "terminal.json"
    value["terminal_validation_sidecar_path"] = str(terminal)
    receipt = publish_primary(path, value, lambda p: dict(passed=True))
    atomic_json(terminal, dict(publication=receipt))


@pytest.mark.parametrize(
    "change", ["none", "board", "missing", "score", "retired", "terminal", "clock", "history"]
)
def test_loader_original_receipts(tmp_path, monkeypatch, change):
    """REQ-REPORT-8190: original pins and actual readiness operands gate separately."""
    monkeypatch.setattr(inputs, "RESOURCES", ["CODEX.md"])
    (tmp_path / "CODEX.md").write_text("private resource")
    boards = panel()["boards"]
    pins = {}
    for board in boards:
        name = board["board"]
        transcript = tmp_path / (name + ".transcript.json")
        atomic_json(transcript, dict(board=name, blocker=board["blocker"]))
        receipt = tmp_path / (name + ".json")
        atomic_json(
            receipt,
            dict(
                run_date="20260913",
                raw_dispatch_transcript_path=str(transcript),
                board_rows=[dict(board=name, latest_receipt_hash=reference(transcript)["sha256"])],
            ),
        )
        board["source_path"] = str(receipt)
        pins[name] = reference(receipt)["sha256"]
    monkeypatch.setattr(inputs, "PINS", pins)
    resultdir = tmp_path / "results"
    resultdir.mkdir()
    published(
        resultdir / (inputs.NAMES[8176] + ".json"),
        dict(experiment_id=8176, hardware_boundary_ready_score=1, board_rows=boards),
    )
    components = panel()["requests"][0]["components"]
    rows = []
    for item in panel()["requests"]:
        r = item["request"]
        clock = 0
        for a, b, cost in inputs.CLOCKS:
            r[a] = clock
            clock += components[cost]
            r[b] = clock
        r.update(arrival_ns=0, response_ns=100, evidence=[])
        store = tmp_path / (r["unit_id"] + ".store.json")
        atomic_json(store, dict(action=r["action"]))
        r["store"] = reference(store)
        rows.append(r)
    work = tmp_path / "primitive_rows.json"
    atomic_json(work, dict(requests=rows, startup_ns=96))
    state = panel()["state"]
    snapshot = tmp_path / "input_data.json"
    atomic_json(
        snapshot,
        dict(
            head=dict(centers=state["centers"], intercept=0, weights=[0]),
            geometry=state["geometry"],
        ),
    )
    service = resultdir / (inputs.NAMES[8188] + ".json")
    published(
        service,
        dict(
            experiment_id=8188,
            cached_service_ready_score=1,
            raw_shard_hashes=[reference(work), reference(snapshot)],
        ),
    )
    published(
        resultdir / (inputs.NAMES[8174] + ".json"),
        dict(
            experiment_id=8174, complete_service_ready_score=1, raw_shard_hashes=[reference(work)]
        ),
    )
    manifest = tmp_path / "ops/exclusion_manifest.yaml"
    manifest.parent.mkdir()
    manifest.write_text("retired: []\n")
    if change == "board":
        Path(boards[0]["source_path"]).write_text("{}")
    elif change == "missing":
        work.unlink()
    elif change == "score":
        v = json.loads(service.read_bytes())
        v["cached_service_ready_score"] = None
        published(service, v)
    elif change == "retired":
        manifest.write_text("retired:\n- experiment_id: 8188\n")
    elif change == "terminal":
        (resultdir / "raw" / service.stem / "terminal.json").unlink()
    elif change == "clock":
        rows[0].pop("generation_end_ns")
        atomic_json(work, dict(requests=rows, startup_ns=96))
        v = json.loads(service.read_bytes())
        v["raw_shard_hashes"] = [reference(work), reference(snapshot)]
        published(service, v)
    elif change == "history":
        (resultdir / (inputs.NAMES[8176] + ".json")).unlink()
    data = inputs.load(tmp_path, tmp_path / "custody")
    value = h.reduce(data)
    if change == "none":
        assert value["verdict_class"] == "null"
        assert len(data["requests"]) == 4
    else:
        assert value["verdict_class"] == "blocked"
    if change == "board":
        assert not data["boards"][0]["custody_valid"] and data["boards"][1]["custody_valid"]
        assert value["measured_workload_bound_ready_score"] == 1
    if change == "score":
        assert any(
            c["artifact_field"] == "cached_service_ready_score" and c["observed"] is None
            for c in value["gate_check_summary"]
        )


def test_orchestration_failure_and_replay(tmp_path, monkeypatch):
    """REQ-REPORT-8190: failed owned checks zero readiness and retain failed receipts."""
    plan = cli.commands(tmp_path)
    assert any(s.name == "coverage_combine" for s in plan)
    assert {"E2E_016_fixture", "E2E_016_cold"} <= {s.name for s in plan}
    assert "--date" in next(s.argv for s in plan if s.name == "E2E_016_cold")
    assert all(
        "::" not in a
        for s in plan
        if s.name in {"ruff_check", "ruff_format", "scoped_spec_coverage"}
        for a in s.argv
    )
    monkeypatch.setattr(inputs, "load", lambda root, raw: panel())
    fail = dict(
        name="private",
        passed=False,
        normal_exit=True,
        actual_exit=1,
        expected_exit=0,
        duration_s=0.01,
    )
    log = tmp_path / "failed_owned.log"
    log.write_text("private owned failure\n")
    fail.update(log_path=str(log), log_sha256=reference(log)["sha256"])
    monkeypatch.setattr(
        cli,
        "execute",
        lambda specs, raw, env=None: [fail] if specs and specs[0].scope != "terminal" else [],
    )
    output = tmp_path / (cli.NAME + ".json")
    assert cli.main(["--output", str(output)]) == 1
    v = json.loads(output.read_bytes())
    assert v["verdict_class"] == "disqualified" and v["hardware_boundary_ready_score"] == 0
    assert cli.replay(output)["passed"]
    altered = deepcopy(v)
    altered["required_checks_passed"] = True
    atomic_json(output, altered)
    with pytest.raises(ValueError, match="validation_receipt_drift"):
        cli.replay(output)
    atomic_json(output, v)
    original_reduce = h.reduce
    forged = original_reduce(panel())
    forged["amdahl_bounds"][0]["outer_ceiling"] = 99
    altered = deepcopy(v)
    altered["amdahl_bounds"] = forged["amdahl_bounds"]
    atomic_json(output, altered)
    with monkeypatch.context() as context:
        context.setattr(h, "reduce", lambda data: forged)
        with pytest.raises(ValueError, match="independent_ceiling_drift"):
            cli.replay(output)
    v["config"]["seed"] += 1
    atomic_json(output, v)
    with pytest.raises(ValueError, match="configuration_drift"):
        cli.replay(output)
    monkeypatch.setattr(cli, "execute", lambda specs, raw, env=None: [fail])
    assert cli.main(["--output", str(output)]) == 1
    health = tmp_path / "health.json"
    atomic_json(health, [fail])
    assert cli.main(["--output", str(output), "--repository-health-receipt", str(health)]) == 1
    monkeypatch.setattr(cli, "main", lambda: 0)
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(cli.ROOT / cli.SCRIPT), run_name="__main__")
    assert caught.value.code == 0
