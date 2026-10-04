"""REQ-VERIFY-8119 / REQ-REPORT-8119: private batch and custody evidence."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_8119_v702_batched_service_cost as e


@pytest.fixture
def native():
    """SCENARIO-VERIFY-8119: use the qualified real PyO3 bytes."""
    value = json.loads((e.ROOT / e.NATIVE).read_text())
    return e.host.load_binding(
        dict(library=dict(path=value["native_library_path"], sha256=value["native_library_sha256"]))
    )[0]


@pytest.fixture
def small():
    """REQ-VERIFY-8119: fixtures keep validation bounded without changing production."""
    return dict(e.CONFIG, batches=[1, 8], centers=[16], strata=3, repetitions=2, warmups=1)


@pytest.fixture(scope="module")
def workload(tmp_path_factory):
    """REQ-REPORT-8119: authenticate large external history once per private panel."""
    return e.inputs(e.ROOT, tmp_path_factory.mktemp("batched-inputs"))


def test_schedule_and_real_inputs(tmp_path, small):
    """REQ-VERIFY-8119: production matrix and external gates remain separate."""
    assert len(e.schedule(e.CONFIG)) == 3600
    assert e.schedule(small) == e.schedule(small)
    data = e.inputs(e.ROOT, tmp_path)
    assert len(data["public"]) == 48
    assert data["library"] and data["acquisition"]
    assert not data["learning"]
    assert any(c["check"] == "learning_qualified" and not c["passed"] for c in data["checks"])
    assert len(set(r["stratum"] for r in data["public"])) == 3
    assert len(data["panel_source_ids"]) == 3
    blocked = e.inputs(tmp_path / "absent", tmp_path / "blocked")
    assert not blocked["library"] and not blocked["acquisition"]
    assert all("artifact_field" in c and "passed" in c for c in blocked["checks"])


@pytest.mark.parametrize("mode", e.MODES)
def test_batch_transactions(tmp_path, native, mode, workload):
    """SCENARIO-VERIFY-8119: vector crossing, durable parity and real cache modes."""
    data = workload
    slot = dict(batch=8, centers=16, stratum=0, mode=mode, repetition=0, unit_id="private")
    pair = [e.transaction(data, native, slot, arm, tmp_path / arm) for arm in e.ARMS]
    assert e.parity(pair)
    for row in pair:
        assert len(row["probabilities"]) == 8
        assert row["transaction_ns"] >= sum(row["components"].values())
        assert row["components"]["durable_write_ns"] > 0
        assert (
            all(x["status"] == "hit" for x in row["cache_events"])
            if mode in ("exact-reuse", "restart")
            else any(x["status"] == "miss" for x in row["cache_events"])
        )
        assert e.host.read_state(Path(row["state_path"])) == row["durable_state"]
    altered = deepcopy(pair)
    altered[1]["actions"][0] = "mutated"
    assert not e.parity(altered)


def test_measure_reduce_and_replay(tmp_path, native, small, workload):
    """REQ-REPORT-8119: primitive reductions authenticate measured denominators."""
    data = workload
    evidence = e.measure(data, native, tmp_path / "raw", small)
    reduced = e.reduce_rows(evidence, small)
    assert reduced["passed"] and reduced["completed_count"] == 60
    assert len(evidence["batch_rows"]) == 12
    assert len(reduced["summaries"]) == 36
    changed = deepcopy(evidence)
    changed["paired_service_rows"][0]["arms"][0]["transaction_ns"] = -1
    assert not e.reduce_rows(changed, small)["passed"]
    assert not e.reduce_rows(dict(batch_rows=[], paired_service_rows=[]), small)["passed"]
    state = e.prior.fixture(9, 16, 12)
    state["coefficients"] = [0.0] * 17
    values = [[0.0] * 9] * 8
    model = native.RustRadial8105(json.dumps(state))
    assert np.allclose(e.kernel(e.prepare(state), values, model, "rust"), [0.5] * 8)
    with pytest.raises(ValueError):
        native.RustRadial8105(json.dumps(state)).predict([[float("nan")] * 9])


def bind(root, label, value):
    """Private transport receipts exercise custody without changing historical primaries."""
    path = root / label
    raw = path.parent / "raw" / path.stem
    value = dict(value, terminal_validation_sidecar_path=str(raw / "terminal.json"))
    e.atomic_json(path, value)
    side = raw / "validators" / "fixture.json"
    e.atomic_json(side, dict(primary_sha256=e.sha256_file(path), report=dict(passed=True)))
    e.atomic_json(
        raw / "terminal.json",
        dict(publication=dict(sidecar_path=str(side), primary_sha256=e.sha256_file(path))),
    )


def test_external_custody_and_invalid_acquisition(tmp_path):
    """REQ-REPORT-8119: exact key/hash gates reject qualified-looking mutations."""
    acquired = json.loads((e.ROOT / e.ACQUISITION).read_text())
    original = json.loads(Path(acquired["acquisition_manifest"]["path"]).read_text())
    for name in ("hash", "rows", "key", "ledger", "identity", "body", "missing"):
        root = tmp_path / name
        changed = deepcopy(acquired)
        manifest = deepcopy(original)
        if name == "rows":
            manifest["current_acquisition_rows"] = []
        elif name == "key":
            changed["current_acquisition_rows"][0]["judgment_key"] = "changed"
            manifest["current_acquisition_rows"] = changed["current_acquisition_rows"]
        elif name == "ledger":
            changed["call_ledger"][1]["response_sha256"] = "changed"
        elif name == "identity":
            manifest["runtime_identity"] = {}
        elif name == "body":
            changed["current_acquisition_rows"][0]["source_bytes"] += "20"
            manifest["current_acquisition_rows"] = changed["current_acquisition_rows"]
        path = root / "manifest.json"
        e.atomic_json(path, manifest)
        changed["acquisition_manifest"] = dict(
            path=str(path), sha256="bad" if name == "hash" else e.sha256_file(path)
        )
        if name == "missing":
            path.unlink()
        bind(root, e.ACQUISITION, changed)
        data = e.inputs(root, root / "observed")
        assert not data["acquisition"]
        assert any(c["check"] == "acquisition_join" and not c["passed"] for c in data["checks"])
    root = tmp_path / "qualified-learning"
    bind(
        root,
        e.LEARNING,
        dict(
            required_checks_passed=True,
            flagged_adversarial=False,
            verdict_class="null",
            learning_trajectory_ready_score=1,
            update_rows=[dict(unit_id="private")],
        ),
    )
    assert e.inputs(root, root / "observed")["learning"]
    native = json.loads((e.ROOT / e.NATIVE).read_text())
    native["native_library_sha256"] = "mutated"
    bind(root, e.NATIVE, native)
    assert not e.inputs(root, root / "bad-native")["library"]


def test_build_worker_and_unavailable_keys(tmp_path, native, small):
    """REQ-REPORT-8119: unavailable operands cannot become zero-cost acquisitions."""
    work = e.worker(e.ROOT, tmp_path / "work", small)
    reduced = e.reduce_rows(work["evidence"], small)
    value = e.build(work, [dict(passed=True)], tmp_path, reduced)
    assert value["host_service_ready_score"] == 1
    assert value["complete_service_ready_score"] == 0
    assert value["verdict_class"] == "blocked"
    assert value["whole_service_speedup"] is None and not value["nfr01_met"]
    assert value["call_ledger"] == [] and value["MODEL_SPECS"] == []
    assert all(
        r["matched"] and r["no_reuse_total_s"] >= r["composed_total_s"]
        for r in value["acquisition_join_rows"]
    )
    assert e.build(work, [dict(passed=False)], tmp_path, reduced)["verdict_class"] == "disqualified"
    data = deepcopy(work["data"])
    data["acquisition"] = []
    slot = e.schedule(small)[0]
    pair = [
        e.transaction(data, native, slot, arm, tmp_path / "missing-keys" / arm) for arm in e.ARMS
    ]
    assert e.parity(pair) and set(pair[0]["actions"]) == {"abstain"}
    missing = e.accounting(data, dict(paired_service_rows=[dict(slot, arms=pair)]))
    assert all(r["acquisition_s"] is None and r["composed_total_s"] is None for r in missing)
    empty = e.worker(tmp_path / "absent", tmp_path / "blocked", small)
    assert not empty["evidence"]["paired_service_rows"]


def test_private_cli_and_cold_mutations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8119: actual script-path routes run outside checkout."""
    import os
    import runpy
    import shutil

    py, cli = str(e.ROOT / ".venv/bin/python"), str(e.ROOT / e.CLI)
    output = tmp_path / "success" / "experiment_8119_private.json"
    blocked = tmp_path / "blocked" / "experiment_8119_blocked.json"
    commands = [
        e.CommandSpec(
            "private_success",
            (py, "-u", cli, "--fixture-output", str(output), "--fixture-small"),
            "private_cli",
            60,
        ),
        e.CommandSpec(
            "private_cold_replay", (py, "-u", cli, "--cold-replay", str(output)), "private_cli", 60
        ),
        e.CommandSpec(
            "private_blocked",
            (
                py,
                "-u",
                cli,
                "--root",
                str(tmp_path / "absent"),
                "--fixture-output",
                str(blocked),
                "--fixture-small",
            ),
            "private_cli",
            60,
        ),
    ]
    receipts = e.run_commands(
        tmp_path,
        commands,
        log_dir=tmp_path / "logs",
        heartbeat_s=30,
        extra_env={"PYTHONPATH": "", "JAX_PLATFORMS": "cpu"},
    )
    assert all(r["passed"] for r in receipts)
    value = json.loads(output.read_text())
    assert value["host_service_ready_score"] == 1
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    assert e.replay(output)
    for name in (
        "ratio",
        "actions",
        "raw_hash",
        "kernel",
        "reduction",
        "code",
        "durable",
        "log",
        "accounting",
    ):
        changed = deepcopy(value)
        if name == "ratio":
            changed["paired_service_rows"][0]["arms"][0]["transaction_ns"] = -1
        elif name == "actions":
            changed["paired_service_rows"][0]["arms"][0]["actions"][0] = "changed"
        elif name == "raw_hash":
            changed["raw_shard_hashes"][0]["sha256"] = "changed"
        elif name == "kernel":
            changed["batch_rows"][0]["arms"][0]["probabilities"][0] = -1.0
        elif name == "reduction":
            changed["reduction"]["completed_count"] = -1
        elif name == "code":
            changed["code_config_hashes"][e.OWNED[0]] = "changed"
        elif name == "durable":
            changed["paired_service_rows"][0]["arms"][0]["state_path"] = str(
                tmp_path / "absent.json"
            )
        elif name == "accounting":
            changed["acquisition_join_rows"][0]["composed_total_s"] = -1.0
        else:
            changed["validation_receipts"][0]["log_sha256"] = "changed"
        mutated = tmp_path / ("experiment_8119_" + name + ".json")
        e.atomic_json(mutated, changed)
        assert not e.replay(mutated)
    mutated = tmp_path / "experiment_8119_ratio.json"
    receipt = e.run_commands(
        tmp_path,
        [
            e.CommandSpec(
                "private_mutation",
                (py, "-u", cli, "--cold-replay", str(mutated)),
                "private_cli",
                60,
            )
        ],
        log_dir=tmp_path / "negative-log",
        heartbeat_s=30,
    )[0]
    assert receipt["exit_code"] == 1
    receipt.update(passed=True, expected_exit=1)
    receipts.append(receipt)
    assert e.replay(blocked)
    assert not e.replay(tmp_path / "missing.json")
    with monkeypatch.context() as patch:
        loaded, binding = e.host.load_binding(dict(library=value["loaded_binding_receipt"]))
        patch.setattr(
            e.host, "load_binding", lambda data: (loaded, dict(binding, sha256="changed"))
        )
        assert not e.replay(output)
    changed = deepcopy(value)
    changed["kernel_operands"][changed["batch_rows"][0]["cell"]]["values"][0][0] += 100.0
    primitive = tmp_path / "changed-kernel" / "primitive_rows.json"
    evidence = {
        key: changed[key]
        for key in ("batch_rows", "paired_service_rows", "warmup_rows", "kernel_operands")
    }
    e.atomic_json(primitive, evidence)
    changed["raw_shard_hashes"] = [
        dict(path=str(primitive), sha256=e.sha256_file(primitive))
        if Path(r["path"]).name == "primitive_rows.json"
        else r
        for r in changed["raw_shard_hashes"]
    ]
    e.atomic_json(tmp_path / "kernel-operands.json", changed)
    assert not e.replay(tmp_path / "kernel-operands.json")
    path = Path(value["paired_service_rows"][0]["arms"][0]["state_path"])
    saved = json.loads(path.read_text())
    state = dict(saved["state"], version=saved["state"]["version"] + 100)
    e.atomic_json(path, dict(state=state, sha256=e.canonical_hash(state)))
    assert not e.replay(output)
    e.atomic_json(path, saved)
    destination = os.getenv("CARNOT_8119_E2E_RECEIPTS")
    if destination:
        for row in receipts:
            copied = Path(destination).parent / "private_cli_logs" / (row["name"] + ".log")
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp_path / row["log_path"], copied)
            row["log_path"] = str(copied)
        e.atomic_json(Path(destination), dict(receipts=receipts))
    monkeypatch.setattr(e, "main", lambda: 0)
    runpy.run_path(cli, run_name="loaded_cli")
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(cli, run_name="__main__")
    assert stopped.value.code == 0


def test_main_and_terminal_routes(tmp_path, monkeypatch, small):
    """SCENARIO-REPORT-8119: owned failures retain disqualified zero readiness."""
    blocked = e.worker(tmp_path / "absent", tmp_path / "blocked-work", small)
    mode = {"write": True, "failure": False}

    def execute(commands, raw, private):
        receipts = [
            dict(name=c.name, passed=True, scope=c.scope, exit_code=0, duration_s=0.01)
            for c in commands
        ]
        if commands[0].name == "measurement_normal_exit" and mode["write"]:
            e.atomic_json(raw / "work.json", blocked)
            e.atomic_json(raw / "private_cli_receipts.json", dict(receipts=[]))
        return receipts

    publish = e.publish_primary

    def publication(output, value, validator):
        if mode["failure"] and value["verdict_class"] != "disqualified":
            return publish(output, value, lambda p: dict(passed=False))
        return publish(output, value, validator)

    monkeypatch.setattr(e, "execute", execute)
    monkeypatch.setattr(e, "publish_primary", publication)
    output = tmp_path / "normal" / "experiment_8119_private.json"
    assert e.main(["--output", str(output)]) == 0
    assert e.main(["--output", str(output)]) == 1
    with pytest.raises(ValueError, match="retry_requires_owned_failure"):
        e.main(["--output", str(output), "--retry-owned-validation"])
    assert json.loads(output.read_text())["repository_health"]
    mode["failure"] = True
    failed = tmp_path / "failed" / "experiment_8119_private.json"
    assert e.main(["--output", str(failed)]) == 0
    assert json.loads(failed.read_text())["verdict_class"] == "disqualified"
    mode["failure"] = False
    assert e.main(["--output", str(failed), "--retry-owned-validation"]) == 0
    assert list(
        (failed.parent / "raw" / failed.stem / "invocations").glob("*/superseded_primary.json")
    )
    fixture = tmp_path / "fixture" / "experiment_8119_private.json"
    assert e.main(["--fixture-output", str(fixture), "--fixture-small"]) == 0
    mode["write"] = False
    assert e.main(["--output", str(tmp_path / "nowork" / "experiment_8119_private.json")]) == 1
    monkeypatch.setattr(e, "worker", lambda *a: blocked)
    assert e.main(["--worker-output", str(tmp_path / "worker.json")]) == 0
    monkeypatch.setattr(e, "replay", lambda path: True)
    assert e.main(["--cold-replay", str(output)]) == 0
    monkeypatch.setattr(e, "replay", lambda path: False)
    assert e.main(["--cold-replay", str(output)]) == 1
    with pytest.raises(SystemExit):
        e.main(["--fixture-small"])
    with pytest.raises(SystemExit):
        e.main(["--date", "20261005"])


def test_binding_identity_rejected(tmp_path, monkeypatch, native, small, workload):
    """SCENARIO-VERIFY-8119: the actual loaded module must match the qualified inode."""
    monkeypatch.setattr(e, "inputs", lambda *a: workload)
    monkeypatch.setattr(
        e.host, "load_binding", lambda data: (native, dict(path=native.__file__, sha256="changed"))
    )
    with pytest.raises(ValueError, match="actual_loaded_binding_identity"):
        e.worker(tmp_path, tmp_path / "worker", small)


def test_execute_receipts(tmp_path):
    """REQ-REPORT-8119: argv, normal exit, duration and log hash are actual child evidence."""
    command = e.CommandSpec(
        "private_probe",
        (str(e.ROOT / ".venv/bin/python"), "-u", "-c", 'print("completed", flush=True)'),
        "owned",
        60,
    )
    rows = e.execute([command], tmp_path, tmp_path)
    assert rows[0]["passed"] and rows[0]["exit_code"] == 0
    assert e.sha256_file(Path(rows[0]["log_path"])) == rows[0]["log_sha256"]


def test_heartbeat_counts_and_failure_report(tmp_path):
    """REQ-REPORT-8119: pending counts are real and failure publication grants no readiness."""

    class Stop:
        def __init__(self):
            self.calls = 0

        def wait(self, seconds):
            self.calls += 1
            return self.calls > 1

    stopped = Stop()
    e.wait_progress(stopped, "private", 2, 3)
    assert stopped.calls == 2
    candidate = tmp_path / "candidate.json"
    e.atomic_json(candidate, dict(verdict_class="positive", host_service_ready_score=1))
    assert not e.disqualified_report(candidate, "failed")["passed"]
    e.atomic_json(
        candidate,
        dict(
            verdict_class="disqualified",
            host_service_ready_score=0,
            complete_service_ready_score=0,
            required_checks_passed=False,
        ),
    )
    assert e.disqualified_report(candidate, "failed")["passed"]


def test_validation_argv_uses_real_files(tmp_path):
    """REQ-REPORT-8119: pytest selectors must not become Ruff filesystem operands."""
    for command in e.validation_plan(tmp_path):
        if command.name in ("ruff_check", "ruff_format", "scoped_spec_coverage"):
            assert all("::" not in arg for arg in command.argv)
