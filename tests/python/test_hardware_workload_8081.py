"""REQ-REPORT-8081: private custody, complete costs and terminal mutation controls."""

import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import pytest

from carnot.reporting import hardware_workload_8081 as h
from carnot import experiment_8081_v699_hardware_workload_boundary as e


def fixture():
    """Explicit numerical controls grant no scientific or hardware credit."""
    boards = json.loads((h.ROOT / h.old.CUSTODY).read_bytes())["board_rows"]
    partitions = {}
    for eid, modes in h.MODES.items():
        rows = []
        for mode in modes:
            for arm in h.ARMS:
                for repetition in range(30):
                    for condition, kind in h.CELLS:
                        rows.append(
                            dict(
                                unit=f"{mode}/{arm}/{repetition}/{condition}",
                                source="fixture",
                                arm=arm,
                                seed=101,
                                condition=condition,
                                transaction_class=kind,
                                mode=mode,
                                repetition=repetition,
                                status="completed",
                                transaction_ns=1000,
                                components=dict(
                                    gradient_arithmetic_ns=100,
                                    cache_extraction_ns=200,
                                    cache_hashing_ns=50,
                                    cache_storage_ns=50,
                                    storage_fsync_ns=300,
                                    guard_scans_ns=100,
                                    feature_gather_ns=100,
                                    ffi_ns=100,
                                ),
                            )
                        )
        partitions[str(eid)] = dict(
            qualified=True, path="/tmp/fixture.json", rows=rows, population_rows=[]
        )
    return dict(
        boards=boards,
        partitions=partitions,
        checks=[],
        references=[],
        fixture=True,
        projection_receipts=[],
        trained_head_specs=[],
    )


def test_independent_custody_and_halves():
    """SCENARIO-REPORT-8081-CUSTODY: a missing half never erases valid custody."""
    data = fixture()
    value = h.reduce(data)
    assert value["hardware_custody_ready_score"] == 1
    assert value["combined_workload_bound_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["current_device_execution_count"] == value["independent_count"] == 0
    assert value["purchase_recommendation"] == "none"
    for eid in h.MODES:
        missing = copy.deepcopy(data)
        missing["partitions"][str(eid)]["qualified"] = False
        reduced = h.reduce(missing)
        assert reduced["verdict_class"] == "blocked"
        assert reduced["hardware_custody_ready_score"] == 1
        assert reduced["combined_workload_bound_ready_score"] == 0
        assert sum(r["qualified"] for r in reduced["mode_qualification_rows"]) == 3
    data["boards"][1]["custody_valid"] = False
    value = h.reduce(data)
    assert [r["custody_valid"] for r in value["board_rows"]] == [True, False, True]
    assert value["combined_workload_bound_ready_score"] == 1
    data["boards"][0]["k_max"] = 6
    assert not h.reduce(data)["board_rows"][0]["custody_valid"]


def test_arithmetic_and_exact_missing_operands():
    """SCENARIO-REPORT-8081-COST: component arithmetic uses actual eligible work."""
    data = fixture()
    data["fixture"] = False
    value = h.reduce(data)
    assert value["verdict_class"] == "null"
    for row in value["acceleration_bounds"]:
        assert row["eligible_fraction"] == 0.1
        assert row["arithmetic_only100x_ceiling"] == pytest.approx(1 / 0.901)
        assert row["infinite_arithmetic_ceiling"] == pytest.approx(1 / 0.9)
        assert row["device_transfer_ns"] is None
        assert row["projection_ns"] is None
    assert value["ideal_f1_hypothetical100x"] == 100
    for key, invalid in (
        ("gradient_arithmetic_ns", None),
        ("cache_hashing_ns", -1),
        ("storage_fsync_ns", float("nan")),
        ("ffi_ns", 101),
    ):
        changed = copy.deepcopy(data)
        changed["partitions"]["8078"]["rows"][0]["components"][key] = invalid
        result = h.reduce(changed)
        assert result["verdict_class"] == "blocked"
        assert result["hardware_custody_ready_score"] == 1
        assert any("cost" in r["field"] for r in result["gate_check_summary"])
    data["partitions"]["8078"]["rows"].pop()
    assert h.reduce(data)["combined_workload_bound_ready_score"] == 0


def test_loading_real_and_missing_evidence(tmp_path):
    """SCENARIO-REPORT-8081-PROJECTION: real historical receipts remain host evidence."""
    data = h.load(h.ROOT, tmp_path / "raw")
    value = h.reduce(data)
    assert value["hardware_custody_ready_score"] == 1
    assert len(value["board_rows"]) == 3
    assert data["projection_receipts"]
    assert value["projection_precision_obligations"]["device_accelerator_qualified"] is False
    absent = h.load(tmp_path, tmp_path / "missing")
    assert h.reduce(absent)["verdict_class"] == "blocked"
    path = tmp_path / h.INPUTS[8078][0]
    e.atomic_json(
        path,
        dict(
            experiment_id=8078,
            required_checks_passed=True,
            flagged_adversarial=False,
            verdict_class="null",
            cache_core_ready_score=1,
            terminal_validation_sidecar_path="/missing",
        ),
    )
    plan = dict(checks=[], references=[])
    assert not h.authenticate(tmp_path, 8078, tmp_path / "bad", plan)[1]
    assert plan["checks"][-1]["field"] == "terminal_contract"


def test_private_real_cli_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8081-TERMINAL: private CLI routes run outside the checkout."""
    source, output = tmp_path / "inputs.json", tmp_path / "worker.json"
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    records = []
    for route, wanted in (
        ("success", "circular_positive"),
        ("missing_cost", "blocked"),
        ("forged_board", "blocked"),
    ):
        data = fixture()
        if route == "missing_cost":
            data["partitions"]["8079"]["qualified"] = False
        if route == "forged_board":
            data["boards"][1]["processor_class"] = "fpga_fabric"
        e.atomic_json(source, data)
        argv = [
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            str(e.ROOT / e.SCRIPT),
            "--worker-input",
            str(source),
            "--output",
            str(output),
        ]
        began = time.monotonic()
        child = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, timeout=60)
        assert child.returncode == 0, child.stdout + child.stderr
        assert json.loads(output.read_bytes())["verdict_class"] == wanted
        records.append(
            dict(
                command_argv=argv,
                exit_code=child.returncode,
                duration_s=time.monotonic() - began,
                route=route,
                log_sha256=e.canonical_hash((child.stdout + child.stderr).decode()),
            )
        )
    receipt = os.environ.get("CARNOT_8081_PRIVATE_CLI_RECEIPTS")
    if receipt:
        e.atomic_json(Path(receipt), dict(rows=records))
    assert e.main(["--date", "bad"]) == 1
    monkeypatch.setattr(
        sys, "argv", [e.SCRIPT, "--worker-input", str(source), "--output", str(output)]
    )
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(e.ROOT / e.SCRIPT), run_name="__main__")
    assert exited.value.code == 0


def test_parent_validation_and_cold_replay(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8081-TERMINAL: owned failures disqualify, replay detects drift."""
    data = fixture()
    monkeypatch.setattr(h, "load", lambda *_: copy.deepcopy(data))
    fail = dict(checks=False, child=False, reader=False, coverage=True)

    def run(root, specs, *, log_dir, **kwargs):
        if specs[0].name == "reduction_normal_exit":
            e.atomic_json(
                Path(specs[0].argv[-1]), h.reduce(json.loads(Path(specs[0].argv[-3]).read_bytes()))
            )
        if specs[0].name == "environment":
            scratch = Path(kwargs["extra_env"]["CARNOT_EXPERIMENT_ARTIFACT_ROOT"])
            if fail["coverage"]:
                e.atomic_json(
                    scratch / "coverage.json",
                    dict(files={p: dict(summary=dict(missing_lines=0)) for p in e.OWNED}),
                )
            e.atomic_json(scratch / "private_cli.json", dict(rows=[]))
        return [
            dict(
                name=s.name,
                scope=s.scope,
                exit_code=int(fail["checks"]),
                passed=not (
                    fail["checks"]
                    and s.scope == "owned"
                    and s.name != "reduction_normal_exit"
                    or fail["child"]
                    and s.name == "reduction_normal_exit"
                ),
                command_argv=list(s.argv),
                duration_s=0.01,
                log_sha256="fixture",
            )
            for s in specs
        ]

    monkeypatch.setattr(e, "run_commands", run)
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **k: dict(passed=not fail["reader"]))
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = ["--output", str(output)]
    assert e.main(argv) == 0
    assert e.main(["--cold-replay", str(output)]) == 0
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    child = subprocess.run(
        [
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            str(e.ROOT / e.SCRIPT),
            "--cold-replay",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    original = json.loads(output.read_bytes())
    original["compatible_component_fractions"] = []
    e.atomic_json(output, original)
    assert e.main(["--cold-replay", str(output)]) == 1
    fail["checks"] = True
    monkeypatch.setattr(e, "terminal", lambda _: dict(passed=True))
    assert e.main(argv) == 0
    result = json.loads(output.read_bytes())
    assert result["verdict_class"] == "disqualified"
    assert result["hardware_custody_ready_score"] == 0
    assert result["combined_workload_bound_ready_score"] == 0
    assert e.replay(output)["passed"]
    fail.update(checks=False, child=True)
    assert e.main(argv) == 1
    fail.update(child=False, reader=True, coverage=False)
    assert e.main(argv) == 1
    assert e.main(["--worker-input", str(tmp_path / "missing"), "--output", str(output)]) == 1


def test_terminal_and_primitive_hash_mutations(tmp_path):
    """SCENARIO-REPORT-8081-COST: clean status cannot conceal stale timing bytes."""
    from carnot.reporting.primary_publication import publish_primary

    primitive = tmp_path / "primitive.json"
    e.atomic_json(primitive, dict(cost=100))
    path = tmp_path / h.INPUTS[8078][0]
    terminal = tmp_path / "terminal.json"
    value = dict(
        experiment_id=8078,
        task_id="exp8078-private",
        required_checks_passed=True,
        flagged_adversarial=False,
        verdict_class="null",
        honest_verdict="complete_null",
        cache_core_ready_score=1,
        terminal_validation_sidecar_path=str(terminal),
        raw_shard_hashes=[e.reference(primitive)],
    )
    binding = publish_primary(path, value, lambda _: dict(passed=True))
    e.atomic_json(terminal, dict(publication=binding))
    plan = dict(checks=[], references=[])
    assert h.authenticate(tmp_path, 8078, tmp_path / "raw", plan)[1]
    e.atomic_json(primitive, dict(cost=101))
    plan = dict(checks=[], references=[])
    assert not h.authenticate(tmp_path, 8078, tmp_path / "changed", plan)[1]
    assert any(r["field"] == "sha256" and not r["passed"] for r in plan["checks"])
    value["verdict_class"] = "circular_positive"
    e.atomic_json(path, value)
    plan = dict(checks=[], references=[])
    assert not h.authenticate(tmp_path, 8078, tmp_path / "stale", plan)[1]
    assert any(r["field"] == "terminal_contract" for r in plan["checks"])
