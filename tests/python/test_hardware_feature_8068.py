"""REQ-REPORT-8068: historical custody and conditional bounds stay separate."""

import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import pytest

from carnot.reporting import hardware_feature_8068 as h
from carnot import experiment_8068_v698_hardware_feature_boundary as e


def fixture():
    """Private controls use exposed board metadata and invented costs as fixtures."""
    prior = json.loads((e.ROOT / h.CUSTODY).read_text())
    costs = [
        dict(
            unit=condition,
            source="private",
            arm="native_cached",
            seed=0,
            cache_condition=condition,
            repetition=0,
            status="completed",
            transaction_ns=1000,
            components=dict(
                gradient_arithmetic_ns=100,
                guard_scans_ns=200,
                storage_fsync_ns=300,
                cache_extraction_ns=400,
                cache_hashing_ns=10,
                cache_loading_ns=10,
            ),
        )
        for condition in ("cold", "warm", "all_miss")
    ]
    return dict(
        boards=prior["board_rows"],
        costs=costs,
        checks=[],
        references=[],
        feature_qualified=True,
        fixture=True,
        numerical_obligations={"overflow": "float64 fallback before commit"},
    )


def test_bounds_and_missing_costs():
    """SCENARIO-REPORT-8068-BOUNDS: unknown transfer cannot become measured zero."""
    data = fixture()
    value = h.reduce(data)
    assert value["verdict_class"] == "circular_positive"
    assert value["hardware_custody_ready_score"] == 1
    assert value["current_device_execution_count"] == 0
    assert value["independent_count"] == 0
    for row in value["acceleration_bounds"]:
        assert row["hypothetical100x_zero_overhead_ceiling"] == pytest.approx(1000 / 901)
        assert row["break_even_transfer_queue_budget_ns"] == 99
        assert row["device_transfer_ns"] is None
        assert row["current_fpga_fabric_ceiling"] == 1
    for key in (
        "transaction_ns",
        "gradient_arithmetic_ns",
        "guard_scans_ns",
        "storage_fsync_ns",
        "cache_extraction_ns",
        "cache_hashing_ns",
        "cache_loading_ns",
    ):
        altered = copy.deepcopy(data)
        target = (
            altered["costs"][0] if key == "transaction_ns" else altered["costs"][0]["components"]
        )
        del target[key]
        result = h.reduce(altered)
        assert result["workload_bound_status"] == "blocked"
        assert result["hardware_custody_ready_score"] == 1
        assert any(key in r["field"] for r in result["gate_check_summary"])
    for bad in (0, -1, float("nan"), True, "1000"):
        altered = copy.deepcopy(data)
        altered["costs"][0]["transaction_ns"] = bad
        assert h.reduce(altered)["verdict_class"] == "blocked"
    altered = copy.deepcopy(data)
    altered["costs"][0]["components"]["gradient_arithmetic_ns"] = 2000
    assert h.reduce(altered)["workload_bound_status"] == "blocked"
    data["costs"] = data["costs"][:1]
    assert h.reduce(data)["workload_bound_status"] == "blocked"
    data["feature_qualified"] = False
    assert h.reduce(data)["acceleration_bounds"] == []


def test_board_integrity_and_current_inputs(tmp_path):
    """SCENARIO-REPORT-8068-CUSTODY: one failed board does not erase the others."""
    data = fixture()
    for index, field, forged in (
        (0, "k_max", 6),
        (1, "processor_class", "fpga_fabric"),
        (2, "blocker", None),
        (0, "current_hardware_execution", True),
    ):
        bad = copy.deepcopy(data)
        bad["boards"][index][field] = forged
        assert h.reduce(bad)["hardware_custody_ready_score"] == 0
    data["boards"] = []
    assert h.reduce(data)["verdict_class"] == "blocked"
    current = h.load(e.ROOT, tmp_path / "evidence")
    result = h.reduce(current)
    assert result["hardware_custody_ready_score"] == 1
    assert result["workload_bound_status"] == "blocked"
    assert len(result["board_rows"]) == 3
    assert result["acceleration_bounds"] == []
    assert any(
        r["field"] == "verdict_class" and r["upstream"] == "exp8066"
        for r in result["gate_check_summary"]
    )
    absent = h.load(tmp_path, tmp_path / "missing")
    assert h.reduce(absent)["hardware_custody_ready_score"] == 0


def test_real_cli_and_mutation_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8068-TERMINAL: E2E-016 private cold reader rejects mutations."""
    source, output = tmp_path / "input.json", tmp_path / "reduced.json"
    e.atomic_json(source, fixture())
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    records = []
    for changes, expected in (
        (None, "circular_positive"),
        ("missing_cost", "blocked"),
        ("forged_board", "blocked"),
    ):
        data = fixture()
        if changes == "missing_cost":
            del data["costs"][0]["components"]["guard_scans_ns"]
        if changes == "forged_board":
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
        assert json.loads(output.read_bytes())["verdict_class"] == expected
        records.append(
            dict(
                command_argv=argv,
                exit_code=child.returncode,
                duration_s=time.monotonic() - began,
                log_sha256=e.canonical_hash(child.stdout.decode() + child.stderr.decode()),
                route=changes or "success",
            )
        )
    receipt = os.environ.get("CARNOT_8068_PRIVATE_CLI_RECEIPTS")
    if receipt:
        e.atomic_json(Path(receipt), dict(rows=records))
    assert e.main(["--date", "bad"]) == 1
    monkeypatch.setattr(
        sys, "argv", [e.SCRIPT, "--worker-input", str(source), "--output", str(output)]
    )
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(e.ROOT / e.SCRIPT), run_name="__main__")
    assert exited.value.code == 0


def test_terminal_authentication_and_qualified_cost_loading(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8068-CUSTODY: missing terminal bindings fail closed."""
    path = tmp_path / h.FEATURE
    e.atomic_json(
        path,
        dict(
            experiment_id=8066,
            feature_service_ready_score=1,
            required_checks_passed=True,
            flagged_adversarial=False,
            verdict_class="null",
            terminal_validation_sidecar_path=str(tmp_path / "absent"),
        ),
    )
    data = dict(checks=[], references=[])
    _, valid = h.authenticate(tmp_path, h.FEATURE, 8066, tmp_path / "raw", data)
    assert not valid and data["checks"][-1]["field"] == "terminal_contract"
    original = h.authenticate

    def qualified(root, relative, eid, raw, plan):
        if eid == 8055:
            return original(root, relative, eid, raw, plan)
        rows = fixture()["costs"]
        for row in rows:
            row["cached"] = True
        shard = tmp_path / "costs.json"
        e.atomic_json(shard, dict(rows=rows))
        return dict(rows=rows, raw_shard_hashes=[e.reference(shard)]), True

    monkeypatch.setattr(h, "authenticate", qualified)
    assert h.load(e.ROOT, tmp_path / "qualified")["feature_qualified"]


def test_parent_publication_and_cold_mutations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8068-TERMINAL: parent validation and cold readers gate publication."""
    data = fixture()
    data["fixture"] = False
    monkeypatch.setattr(h, "load", lambda *_: copy.deepcopy(data))
    fail = {"checks": False, "child": False, "reader": False, "coverage": True}

    def run(root, specs, *, log_dir, **kwargs):
        if specs[0].name == "reduction_normal_exit":
            source = Path(specs[0].argv[-3])
            e.atomic_json(Path(specs[0].argv[-1]), h.reduce(json.loads(source.read_bytes())))
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
                log_sha256="private-control",
            )
            for s in specs
        ]

    monkeypatch.setattr(e, "run_commands", run)
    monkeypatch.setattr(
        e, "reader_receipt", lambda *_args, **_kwargs: dict(passed=not fail["reader"])
    )
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = ["--output", str(output)]
    assert e.main(argv) == 0
    assert e.main(["--cold-replay", str(output)]) == 0
    for key in ("completed_count", "board_rows", "numerical_obligations"):
        original = json.loads(output.read_bytes())
        changed = copy.deepcopy(original)
        changed[key] = "forged"
        e.atomic_json(output, changed)
        assert e.main(["--cold-replay", str(output)]) == 1
        e.atomic_json(output, original)
    fail["checks"] = True
    # Terminal validators are separately exercised; this control accepts the
    # diagnostic candidate so the owned failure's class can be inspected.
    terminal = e.terminal
    monkeypatch.setattr(e, "terminal", lambda _: dict(passed=True))
    assert e.main(argv) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    assert e.replay(output)["passed"]
    monkeypatch.setattr(e, "terminal", terminal)
    fail.update(checks=False, child=True)
    assert e.main(argv) == 1
    fail.update(child=False, reader=True, coverage=False)
    assert e.main(argv) == 1
    assert e.main(["--worker-input", str(tmp_path / "absent"), "--output", str(output)]) == 1
    data["fixture"] = False
    assert h.reduce(data)["verdict_class"] == "null"
