"""REQ-REPORT-8132 / REQ-VERIFY-8132: private qualified service receipts."""

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import pytest

from carnot import experiment_8132_v703_service_cost as e


@pytest.fixture(scope="module")
def workload(tmp_path_factory):
    """Keep authenticated source copies outside published results."""
    return e.inputs(e.ROOT, tmp_path_factory.mktemp("8132-inputs"))


@pytest.fixture(scope="module")
def native(workload):
    """E2E-003/004 use the qualified extension already present on this host."""
    return e.old.host.load_binding(workload)[0]


@pytest.fixture
def small():
    """Small private panels test the same lifecycle and reduction paths."""
    return dict(e.CONFIG, batches=[1, 4], repetitions=2, warmups=1)


def test_frozen_schedule_and_prerequisites(workload, small, tmp_path):
    """REQ-VERIFY-8132: repetitions improve timing precision, not independence."""
    plan = e.schedule(e.CONFIG)
    assert len(plan) == 900
    assert e.CONFIG["warmups"] == 5 and e.CONFIG["repetitions"] == 30
    assert e.CONFIG["batches"] == [1, 4, 16, 64, 256]
    assert e.CONFIG["measurement_ceiling_s"] == 3000
    assert plan[0]["order"] == list(e.ARMS)
    assert plan[1]["order"] == list(reversed(e.ARMS))
    assert e.schedule(small) == e.schedule(small)
    assert workload["library"] and workload["acquisition"]
    assert not workload["learning"]
    assert e.inputs(tmp_path / "missing", tmp_path / "blocked")["library"] == {}


@pytest.mark.parametrize("condition", e.CONDITIONS)
def test_real_transactions_and_checkpoints(tmp_path, workload, native, condition):
    """SCENARIO-VERIFY-8132: all arms preserve actions, order and durable state."""
    slot = dict(e.schedule(dict(e.CONFIG, batches=[4], repetitions=1))[0], condition=condition)
    arms = [e.transaction(workload, native, slot, arm, tmp_path / arm) for arm in e.ARMS]
    assert e.parity(arms)
    for row in arms:
        assert row["full_latency_ns"] >= row["arithmetic_ns"] > 0
        assert row["input_bytes"] > 0
        assert row["full_latency_ns"] == sum(row["components"].values()) + row["residual_ns"]
        state = e.old.host.read_state(Path(row["state_path"]))
        restored = native.RustRadial8105.restore(
            native.RustRadial8105(json.dumps(state)).checkpoint()
        )
        assert json.loads(restored.state_json()) == state
        assert np.allclose(
            restored.predict(row["values"]), row["probabilities"], atol=1e-10, rtol=0
        )
        assert any(
            event["status"] == ("hit" if condition in ("warm", "restart") else "miss")
            for event in row["cache_events"]
        )
    arms[0]["actions"][0] = "corrupted"
    assert not e.parity(arms)


def test_paired_reduction_and_missing_masks(tmp_path, workload, native, small):
    """REQ-VERIFY-8132: intervals resample paired batches and retain all units."""
    evidence = e.measure(workload, native, tmp_path, small)
    reduced = e.reduce_rows(evidence, small)
    assert reduced["passed"] and reduced["completed_count"] == 24
    assert len(reduced["paired_intervals"]) == 36
    assert reduced["bootstrap_repetitions"] == 10000
    assert all(row["amdahl_zero_arithmetic_ceiling"] >= 1 for row in reduced["paired_intervals"])
    changed = deepcopy(evidence)
    changed["pairs"][0]["arms"][0]["full_latency_ns"] = -1
    assert not e.reduce_rows(changed, small)["passed"]
    changed = deepcopy(evidence)
    changed["pairs"].pop()
    assert not e.reduce_rows(changed, small)["passed"]
    censored = e.measure(
        workload, native, tmp_path / "censored", dict(small, measurement_ceiling_s=0)
    )
    assert len(censored["pairs"]) == 24
    assert all(row["status"] == "censored" for row in censored["pairs"])
    assert e.reduce_rows(censored, small)["censored_count"] == 24


def test_build_modeled_bounds_and_cold_replay(tmp_path, workload, native, small):
    """REQ-REPORT-8132: missing acquisition affects only explicitly modeled bounds."""
    raw = tmp_path / "raw"
    evidence = e.measure(workload, native, raw, small)
    e.atomic_json(raw / "input_data.json", workload)
    bounds = e.modeled_bounds(workload, evidence)
    assert all(not row["directly_measured_complete_service"] for row in bounds)
    assert any(row["matched"] for row in bounds)
    e.atomic_json(raw / "modeled_acquisition_bounds.json", bounds)
    value = e.build(workload, evidence, raw, [dict(passed=True)], 2)
    assert value["host_service_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["complete_service_ready_score"] == value["natural_update_cost_ready_score"] == 0
    value["raw_shard_hashes"] = [e.reference(p) for p in raw.glob("*.json")]
    primary = tmp_path / "candidate.json"
    e.atomic_json(primary, value)
    assert e.replay(primary)
    changed = deepcopy(value)
    changed["paired_intervals"][0]["one_sided_95_lower"] *= 2
    e.atomic_json(primary, changed)
    assert not e.replay(primary)
    e.atomic_json(primary, value)
    state = Path(evidence["pairs"][0]["arms"][0]["state_path"])
    state.write_text("{}")
    assert not e.replay(primary)
    assert (
        e.build(workload, evidence, raw, [dict(passed=False)], 2)["host_service_ready_score"] == 0
    )
    missing = dict(workload, acquisition=[])
    assert all(row["modeled_total_s"] is None for row in e.modeled_bounds(missing, evidence))
    assert not e.replay(tmp_path / "missing.json")


def test_bounded_children_and_frozen_validation(tmp_path):
    """SCENARIO-REPORT-8132: normal exit differs from expected rejection and timeout."""
    py = str(e.ROOT / ".venv/bin/python")
    commands = [e.CommandSpec("ok", (py, "-c", "print('done', flush=True)"), "owned", 10)]
    receipts = e.execute(commands, tmp_path)
    assert receipts[0]["passed"] and receipts[0]["normal_exit"]
    reject = e.CommandSpec("mutation", (py, "-c", "raise SystemExit(1)"), "private_cli", 10)
    assert e.execute([reject], tmp_path, expected=1)[0]["passed"]
    timeout = e.CommandSpec("timeout", (py, "-c", "import time; time.sleep(10)"), "owned", 0.01)
    assert not e.execute([timeout], tmp_path)[0]["normal_exit"]
    plan = e.validation_plan(tmp_path)
    assert any("--strict" in c.argv for c in plan)
    assert any("--fail-under=100" in c.argv for c in plan)
    assert all("tests/python/test_primary_publication.py" not in c.argv for c in plan)


def test_outside_checkout_cli_routes(tmp_path):
    """SCENARIO-REPORT-8132: success, external block, mutation and cold replay."""
    import os
    import subprocess

    py = str(e.ROOT / ".venv/bin/python")
    cli = str(e.ROOT / e.CLI)
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    output = tmp_path / "success" / (e.NAME + ".json")
    commands = [
        ("success", [py, "-u", cli, "--fixture-output", str(output), "--fixture-small"], 0),
        ("cold-replay", [py, "-u", cli, "--cold-replay", str(output)], 0),
        ("preserve", [py, "-u", cli, "--fixture-output", str(output), "--fixture-small"], 1),
    ]
    rows = []
    for name, argv, expected in commands:
        started = time.monotonic()
        log = tmp_path / (name + ".log")
        with log.open("w") as stream:
            result = subprocess.run(
                argv, cwd=tmp_path, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=120
            )
        assert result.returncode == expected, log.read_text()[-4000:]
        rows.append(
            dict(
                name=name,
                command_argv=argv,
                expected_exit=expected,
                actual_exit=result.returncode,
                normal_exit=True,
                passed=result.returncode == expected,
                scope="private_cli",
                duration_s=time.monotonic() - started,
                log_path=str(log),
                log_sha256=e.sha256_file(log),
            )
        )
    value = json.loads(output.read_text())
    assert value["host_service_ready_score"] == 1 and value["required_checks_passed"]
    assert value["MODEL_SPECS"] == [] and value["call_ledger"] == []
    value["rows"][0]["numerator"] += 1
    e.atomic_json(output, value)
    started = time.monotonic()
    mutation_argv = [py, "-u", cli, "--cold-replay", str(output)]
    with (tmp_path / "mutation.log").open("w") as stream:
        result = subprocess.run(
            mutation_argv,
            cwd=tmp_path,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    assert result.returncode == 1
    rows.append(
        dict(
            name="mutation",
            command_argv=mutation_argv,
            expected_exit=1,
            actual_exit=result.returncode,
            normal_exit=True,
            passed=True,
            scope="private_cli",
            duration_s=time.monotonic() - started,
            log_path=str(tmp_path / "mutation.log"),
            log_sha256=e.sha256_file(tmp_path / "mutation.log"),
        )
    )
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    started = time.monotonic()
    blocked_argv = [
        py,
        "-u",
        cli,
        "--fixture-output",
        str(blocked),
        "--fixture-small",
        "--root",
        str(tmp_path / "absent"),
    ]
    with (tmp_path / "blocked.log").open("w") as stream:
        result = subprocess.run(
            blocked_argv,
            cwd=tmp_path,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    assert result.returncode == 0, (tmp_path / "blocked.log").read_text()[-4000:]
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    rows.append(
        dict(
            name="external_block",
            command_argv=blocked_argv,
            expected_exit=0,
            actual_exit=result.returncode,
            normal_exit=True,
            passed=True,
            scope="private_cli",
            duration_s=time.monotonic() - started,
            log_path=str(tmp_path / "blocked.log"),
            log_sha256=e.sha256_file(tmp_path / "blocked.log"),
        )
    )
    destination = os.environ.get("CARNOT_8132_E2E_RECEIPTS")
    if destination:
        e.atomic_json(Path(destination), dict(receipts=rows))


def test_main_owned_paths(tmp_path, monkeypatch):
    """REQ-REPORT-8132: owned failures cannot publish host readiness."""
    assert e.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        e.main(["--fixture-small"])
    output = tmp_path / "direct" / (e.NAME + ".json")
    assert e.main(["--fixture-output", str(output), "--fixture-small"]) == 0
    assert e.main(["--cold-replay", str(output)]) == 0
    assert e.main(["--fixture-output", str(output), "--fixture-small"]) == 1
    raw = output.parent / "raw" / output.stem
    assert e.main(["--worker-output", str(raw / "primitive_rows.json")]) == 0
    broken = tmp_path / "failure" / (e.NAME + ".json")
    monkeypatch.setattr(e, "terminal", lambda path: dict(passed=False))
    assert e.main(["--fixture-output", str(broken), "--fixture-small"]) == 0
    value = json.loads(broken.read_text())
    assert value["verdict_class"] == "disqualified" and value["host_service_ready_score"] == 0
    assert (broken.parent / "raw" / broken.stem / "failed_terminal_candidate.json").exists()


def test_panel_and_optional_update_custody(tmp_path, workload):
    """REQ-VERIFY-8132: fixed public panels and optional updates have separate custody."""
    assert len({r["source_cluster_id"] for r in workload["public"]}) == 3
    root = tmp_path / "upstream"
    path = root / e.LEARNING
    e.atomic_json(path, {})
    assert not e.inputs(root, tmp_path / "invalid-update")["learning"]
    raw = path.parent / "raw" / path.stem
    side = raw / "validators" / "fixture.json"
    terminal = raw / "terminal.json"
    value = dict(
        required_checks_passed=True,
        flagged_adversarial=False,
        learning_trajectory_ready_score=1,
        update_rows=[dict(unit_id="fixture")],
        terminal_validation_sidecar_path=str(terminal),
    )
    e.atomic_json(path, value)
    e.atomic_json(side, dict(primary_sha256=e.sha256_file(path), report=dict(passed=True)))
    e.atomic_json(
        terminal, dict(publication=dict(sidecar_path=str(side), primary_sha256=e.sha256_file(path)))
    )
    assert e.inputs(root, tmp_path / "qualified-update")["learning"] == value["update_rows"]


def test_replay_rejects_each_custody_boundary(tmp_path, workload, native, monkeypatch):
    """SCENARIO-REPORT-8132: exact hashes and independent numerical replay reject drift."""
    config = dict(e.CONFIG, batches=[1], repetitions=1, warmups=0)
    raw = tmp_path / "raw"
    evidence = e.measure(workload, native, raw, config)
    e.atomic_json(raw / "input_data.json", workload)
    e.atomic_json(raw / "modeled_acquisition_bounds.json", e.modeled_bounds(workload, evidence))
    value = e.build(workload, evidence, raw, [dict(passed=True)], 2)
    path = tmp_path / "candidate.json"
    for operand in ("source", "code", "config", "modeled", "state"):
        changed = deepcopy(value)
        if operand == "source":
            changed["source_artifact_hashes"][0]["sha256"] = "wrong"
        elif operand == "code":
            changed["code_config_hashes"][e.OWNED[0]] = "wrong"
        elif operand == "config":
            changed["code_config_hashes"]["config"] = "wrong"
        elif operand == "modeled":
            e.atomic_json(raw / "modeled_acquisition_bounds.json", [])
        else:
            state = Path(evidence["pairs"][0]["arms"][0]["state_path"])
            altered = deepcopy(evidence["pairs"][0]["arms"][0]["durable_state"])
            altered["commit_hash"] = "mutated"
            e.atomic_json(state, dict(state=altered, sha256=e.canonical_hash(altered)))
            evidence["pairs"][0]["arms"][0]["state_sha256"] = e.sha256_file(state)
            e.atomic_json(raw / "primitive_rows.json", evidence)
            changed = e.build(workload, evidence, raw, [dict(passed=True)], 2)
        e.atomic_json(path, changed)
        assert not e.replay(path), operand
        e.atomic_json(raw / "modeled_acquisition_bounds.json", e.modeled_bounds(workload, evidence))
    e.atomic_json(path, e.build(workload, evidence, raw, [dict(passed=True)], 2))
    loader = e.old.host.load_binding
    monkeypatch.setattr(e.old.host, "load_binding", lambda data: (native, dict(sha256="wrong")))
    assert not e.replay(path)
    monkeypatch.setattr(e.old.host, "load_binding", loader)


def test_worker_blocks_and_native_identity(tmp_path, workload, native, monkeypatch):
    """SCENARIO-VERIFY-8132: missing host bytes block; changed loaded bytes fail owned work."""
    e.atomic_json(
        tmp_path / "validation_commands.json",
        dict(config=dict(e.CONFIG, batches=[1], repetitions=1)),
    )
    e.atomic_json(tmp_path / "input_data.json", dict(workload, library={}))
    assert e.worker(tmp_path) == 0
    assert all(
        row["status"] == "blocked"
        for row in json.loads((tmp_path / "primitive_rows.json").read_text())["pairs"]
    )
    e.atomic_json(tmp_path / "input_data.json", workload)
    monkeypatch.setattr(
        e.old.host,
        "load_binding",
        lambda data: (native, dict(path=native.__file__, sha256="wrong")),
    )
    with pytest.raises(ValueError, match="actual_loaded_binding_identity"):
        e.worker(tmp_path)


def test_main_failures_and_scoped_owned_branch(tmp_path, monkeypatch):
    """REQ-REPORT-8132: premeasurement failure stops clocks; health is a separate scope."""
    monkeypatch.setattr(e, "qualify_receipt", lambda data, raw: [dict(passed=False)])
    assert (
        e.main(
            [
                "--fixture-output",
                str(tmp_path / "qualification" / (e.NAME + ".json")),
                "--fixture-small",
            ]
        )
        == 1
    )
    monkeypatch.setattr(e, "qualify_receipt", lambda data, raw: [dict(passed=True)])
    actual_execute = e.execute
    monkeypatch.setattr(e, "execute", lambda commands, raw: [dict(passed=False)])
    assert (
        e.main(
            [
                "--fixture-output",
                str(tmp_path / "missing-primitives" / (e.NAME + ".json")),
                "--fixture-small",
            ]
        )
        == 1
    )
    monkeypatch.setattr(e, "CONFIG", dict(e.CONFIG, batches=[1], repetitions=1, warmups=0))
    monkeypatch.setattr(e, "validation_plan", lambda private: [])

    def private_execute(commands, raw):
        if commands and commands[0].name == "measurement_normal_exit":
            e.worker(raw)
        if not commands:
            import os

            destination = Path(os.environ["CARNOT_8132_E2E_RECEIPTS"])
            log = destination.parent / "fixture-route.log"
            log.write_text("private receipt-copy fixture")
            e.atomic_json(
                destination,
                dict(
                    receipts=[
                        dict(
                            passed=True,
                            scope="private_fixture",
                            log_path=str(log),
                            log_sha256=e.sha256_file(log),
                        )
                    ]
                ),
            )
        return [dict(passed=True, scope="private_test_fixture")]

    monkeypatch.setattr(e, "execute", private_execute)
    output = tmp_path / "owned" / (e.NAME + ".json")
    assert e.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["host_service_ready_score"] == 1
    monkeypatch.setattr(e, "execute", actual_execute)


def test_script_wrapper_resolves_checkout(monkeypatch, tmp_path):
    """SCENARIO-REPORT-8132: the script entry point resolves imports and propagates exit."""
    import runpy
    import sys

    monkeypatch.setattr(
        sys, "argv", [str(e.ROOT / e.CLI), "--cold-replay", str(tmp_path / "missing")]
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(e.ROOT / e.CLI), run_name="__main__")
    assert error.value.code == 1
