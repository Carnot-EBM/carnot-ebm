"""REQ-VERIFY-8106 / REQ-REPORT-8106: private service and custody checks."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_8106_v701_radial_service_cost as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import native_radial_8105 as k


@pytest.fixture
def native():
    """SCENARIO-VERIFY-8106: exercise qualified loaded bytes, not a mock binding."""
    value = json.loads(
        (e.ROOT / "results/experiment_8105_v701_native_radial_kernel.json").read_text()
    )
    return k.build.load_native_extension(Path(value["native_library_path"]))


@pytest.fixture
def workload():
    """REQ-VERIFY-8106: small complete bytes still use the full declared matrix."""
    public = [
        dict(
            family_id=f"private-{i}",
            source_bytes=f"Water has {i} atoms.".encode().hex(),
            answer_bytes=f"Water has {i} atoms.".encode().hex(),
        )
        for i in range(30)
    ]
    identity = dict(model="fixture", template="fixture")
    return dict(
        public=public,
        judgments={e.judgment_key(p, identity): 0.25 for p in public},
        identity=identity,
        acquisition=[],
        library={},
        fixture=True,
    )


def test_schedule_and_keys(workload):
    """REQ-VERIFY-8106: freeze counts, seeded alternation and exact identity keys."""
    schedule = e.schedule()
    assert len(schedule) == 1080
    assert schedule == e.schedule()
    assert {r["mode"] for r in schedule} == set(e.MODES)
    assert {r["centers"] for r in schedule} == {16, 28}
    assert all(schedule[i]["order"] != schedule[i + 1]["order"] for i in range(1079))
    p = workload["public"][0]
    key = e.judgment_key(p, workload["identity"])
    assert (
        e.judgment_key(dict(p, source_bytes=p["source_bytes"] + "20"), workload["identity"]) != key
    )
    assert e.judgment_key(p, dict(model="other", template="fixture")) != key


@pytest.mark.parametrize("mode", e.MODES)
def test_transactions(tmp_path, native, workload, mode):
    """SCENARIO-VERIFY-8106: matched durable service covers all real cache modes."""
    pair = [
        e.transaction(
            workload,
            native,
            dict(family=0, centers=16, repetition=0, mode=mode, unit_id="private-unit"),
            arm,
            tmp_path / arm,
        )
        for arm in ("python", "rust")
    ]
    assert e.parity(pair)
    for row in pair:
        assert row["transaction_ns"] >= sum(row["components"].values())
        assert row["durable_state"]["version"] == 4
        assert row["components"]["durable_commit_ns"] > 0
        assert row["cache_event"]["status"] == ("hit" if mode in ("warm", "restart") else "miss")
        if mode == "content-change":
            assert row["judgment_status"] == "blocked_content_key"
            assert row["actions"] == ["abstain"]
        assert e.read_state(Path(row["state_path"])) == row["durable_state"]
    changed = deepcopy(pair)
    changed[1]["actions"] = ["mutated"]
    assert not e.parity(changed)


def test_measure_reduction_and_mutation(tmp_path, native, workload):
    """REQ-REPORT-8106: primitive full-matrix reductions reject altered evidence."""
    evidence = e.measure(workload, native, tmp_path)
    reduced = e.reduce_rows(evidence)
    assert reduced["passed"]
    assert reduced["completed_count"] == 1080
    assert len(reduced["summaries"]) == 12
    assert len(evidence["warmup_rows"]) == 2160
    altered = deepcopy(evidence)
    altered["paired_service_rows"][0]["arms"][0]["transaction_ns"] = -1
    assert not e.reduce_rows(altered)["passed"]
    assert not e.reduce_rows(dict(paired_service_rows=[]))["passed"]


def test_state_hash_and_fallback(tmp_path, native, workload):
    """SCENARIO-VERIFY-8106: corrupt durable bytes and close ties cannot pass."""
    state = e.prior.fixture(9, 16, 1)
    state["coefficients"] = [0.0] * 17
    p, actions, flags = e.score(state, [0.0] * 9, native, "rust")
    assert p == pytest.approx([0.5]) and all(flags)
    assert actions == ["escalate"]
    path = tmp_path / "state.json"
    atomic_json(path, dict(state=state, sha256="mutated"))
    with pytest.raises(ValueError, match="state_hash"):
        e.read_state(path)


def test_real_inputs_and_acquisition(tmp_path):
    """REQ-REPORT-8106: historical receipts qualify only exact current-capture joins."""
    data, checks, refs = e.inputs(e.ROOT, tmp_path)
    assert len(data["public"]) == 30
    assert not [c for c in checks if c.get("scope") == "service" and c["expected"] != c["observed"]]
    assert refs and data["acquisition"]
    assert len(data["judgments"]) == 30
    assert any(c["check"] == "capture_input" and c["expected"] != c["observed"] for c in checks)
    joins = e.acquisition_totals(
        data, [dict(mode="cold", centers=16, python_median_ns=1000, rust_median_ns=1000)]
    )
    assert {r["reuse_count"] for r in joins} == {1, 10, 100}
    assert all(
        r["composed_estimate"] and r["model_load_s"] > 0 and r["fresh_judgment_s"] > 0
        for r in joins
    )
    assert joins[1]["total_s"] > joins[0]["total_s"]
    assert e.acquisition_totals(dict(acquisition=[]), []) == []
    missing, observations, _ = e.inputs(tmp_path / "missing", tmp_path / "blocked")
    assert not missing["public"]
    assert any(c["observed"] is False for c in observations)


def test_artifact_scopes_and_owned_failure(tmp_path, workload):
    """REQ-REPORT-8106: host readiness survives upstream blocks, owned failure does not."""
    work = dict(
        data=workload,
        evidence=dict(paired_service_rows=[]),
        checks=[],
        refs=[],
        loaded_binding_receipt={},
        duration_s=1.0,
        phase_spans=[],
        code_config_hashes={},
    )
    valid = dict(passed=True, completed_count=1080, failed_count=0, summaries=[])
    receipt = dict(passed=True, name="owned", scope="owned")
    value = e.build(work, [receipt], tmp_path, valid)
    assert value["service_cost_ready_score"] == 1
    assert value["acquisition_cost_ready_score"] == 0
    assert value["verdict_class"] == "blocked"
    assert value["verifier_is_oracle"] == 0
    assert value["claim_scope"] == 0 and value["exposure_scope"] == 0
    assert value["nfr01_scope"]["closed"] is False
    failed = e.build(work, [dict(receipt, passed=False)], tmp_path, valid)
    assert failed["verdict_class"] == "disqualified"
    assert failed["service_cost_ready_score"] == 0
    work["checks"] = [
        dict(
            check="input_exists",
            path="absent",
            upstream="native",
            hash=None,
            field="exists",
            op="==",
            expected=True,
            observed=False,
            scope="service",
        )
    ]
    blocked = e.build(work, [receipt], tmp_path, valid)
    assert blocked["service_cost_ready_score"] == 0
    assert blocked["honest_verdict"] == "complete_blocked_native"


def test_worker_and_replay(tmp_path, monkeypatch, native, workload):
    """SCENARIO-REPORT-8106: fresh reads reject evidence or persisted-state mutation."""
    monkeypatch.setattr(e, "inputs", lambda root, raw: (workload, [], []))
    monkeypatch.setattr(e, "load_binding", lambda data: (native, dict(actual_loaded=True)))
    work = e.worker(tmp_path, tmp_path / "raw")
    value = e.build(
        work,
        [dict(passed=True, scope="owned", duration_s=0.01)],
        tmp_path,
        e.reduce_rows(work["evidence"]),
    )
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    value["paired_service_rows"][0]["arms"][0]["actions"] = ["mutation"]
    atomic_json(path, value)
    assert not e.replay(path)
    atomic_json(path, dict(value, paired_service_rows=[]))
    assert not e.replay(path)
    monkeypatch.setattr(e, "inputs", lambda root, raw: (dict(workload, public=[]), [], []))
    blocked = e.worker(tmp_path, tmp_path / "blocked")
    assert not blocked["evidence"]["paired_service_rows"]


def test_main_routes_and_validation(tmp_path, monkeypatch, workload):
    """SCENARIO-REPORT-8106: orchestration preserves evidence and rejects failed workers."""
    fixture = tmp_path / "input.json"
    atomic_json(fixture, workload)
    assert (
        e.main(
            [
                "--worker-output",
                str(tmp_path / "w" / "work.json"),
                "--root",
                str(tmp_path / "absent"),
            ]
        )
        == 0
    )
    assert e.main(["--cold-replay", str(tmp_path / "absent.json")]) == 1
    private = tmp_path / "private"
    private.mkdir()
    commands = e.validation_plan(private)
    assert any("--strict" in c.argv for c in commands)
    assert any("--fail-under=100" in c.argv for c in commands)
    assert e.terminal(tmp_path / "w" / "work.json")["passed"] is False
    output = tmp_path / "experiment_8106_private.json"
    empty = json.loads((tmp_path / "w" / "work.json").read_text())
    monkeypatch.setattr(
        e,
        "execute",
        lambda commands, raw, private: [dict(passed=True, scope="owned", duration_s=0.01)],
    )
    monkeypatch.setattr(e, "worker", lambda root, raw: empty)
    monkeypatch.setattr(e, "publish_primary", lambda output, value, validator: dict(passed=True))
    assert e.main(["--output", str(output), "--root", str(tmp_path / "absent")]) == 0
    assert e.main(["--output", str(output)]) == 1
    mutation = tmp_path / "experiment_8106_mutated.json"
    assert e.main(["--output", str(mutation), "--root", str(tmp_path / "absent"), "--mutate"]) == 1
    monkeypatch.setattr(
        e,
        "execute",
        lambda commands, raw, private: [dict(passed=False, scope="owned", duration_s=0.01)],
    )
    failure = tmp_path / "experiment_8106_failed.json"
    assert e.main(["--output", str(failure), "--root", str(tmp_path / "absent")]) == 0


def test_binding_hash_rejected(tmp_path):
    """REQ-REPORT-8106: replaced native bytes cannot inherit upstream readiness."""
    path = tmp_path / "library.so"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="loaded_binding_hash"):
        e.load_binding(dict(library=dict(path=str(path), sha256="bad")))


def test_private_real_cli(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8106: real success, block, mutation and cold replay exit outside checkout."""
    import os
    import runpy
    import shutil

    py = str(e.ROOT / ".venv/bin/python")
    cli = str(e.ROOT / e.CLI)
    # A private transport fixture keeps CPU qualification independent of natural-data acquisition.
    fixture_root = tmp_path / "fixture-root"
    labels = [
        *e.prior.INPUTS[:11],
        "scripts/experiments/experiment_8078_v699_feature_cache_core.py",
        "scripts/experiments/experiment_8079_v699_feature_cache_lifecycle.py",
        "python/carnot/experiment_8078_v699_feature_cache_core.py",
        "python/carnot/experiment_8079_v699_feature_cache_lifecycle.py",
    ]
    for label in labels:
        target = fixture_root / label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(e.ROOT / label)
    primary = fixture_root / "results/experiment_8105_v701_native_radial_kernel.json"
    raw = primary.parent / "raw" / primary.stem
    native_value = json.loads((e.ROOT / "results" / primary.name).read_text())
    native_value["terminal_validation_sidecar_path"] = str(raw / "terminal.json")
    atomic_json(primary, native_value)
    sidecar = raw / "validators" / "fixture.json"
    atomic_json(sidecar, dict(primary_sha256=e.sha256_file(primary), report=dict(passed=True)))
    atomic_json(raw / "terminal.json", dict(publication=dict(sidecar_path=str(sidecar))))
    fixture_data, _, _ = e.inputs(fixture_root, tmp_path / "input-observations")
    assert fixture_data["fixture"]
    loaded, receipt = e.load_binding(fixture_data)
    assert loaded.RustRadial8105 and receipt["actual_loaded"]
    output = tmp_path / "success" / "experiment_8106_private_success.json"
    blocked = tmp_path / "blocked" / "experiment_8106_private_blocked.json"
    mutated = tmp_path / "mutation" / "experiment_8106_private_mutated.json"
    commands = [
        e.CommandSpec(
            "private_success_changed_content",
            (py, "-u", cli, "--root", str(fixture_root), "--fixture-output", str(output)),
            "private_cli",
            300,
        ),
        e.CommandSpec(
            "private_cold_replay", (py, "-u", cli, "--cold-replay", str(output)), "private_cli", 120
        ),
        e.CommandSpec(
            "private_blocked",
            (py, "-u", cli, "--root", str(tmp_path / "absent"), "--fixture-output", str(blocked)),
            "private_cli",
            120,
        ),
        e.CommandSpec(
            "private_mutation",
            (py, "-u", cli, "--cold-replay", str(mutated)),
            "private_cli",
            120,
        ),
    ]
    receipts = e.execute(commands[:2], tmp_path / "logs", tmp_path)
    changed = json.loads(output.read_text())
    changed["paired_service_rows"][0]["arms"][0]["actions"] = ["mutated"]
    atomic_json(mutated, changed)
    receipts += e.execute(commands[2:], tmp_path / "logs", tmp_path)
    assert [r["exit_code"] for r in receipts] == [0, 0, 0, 1]
    candidate = json.loads(output.read_text())
    assert len(candidate["paired_service_rows"]) == 1080
    assert any(
        r["judgment_status"] == "blocked_content_key"
        for p in candidate["paired_service_rows"]
        for r in p["arms"]
    )
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    assert mutated.exists()
    destination = os.environ.get("CARNOT_8106_E2E_RECEIPTS")
    if destination:
        for row in receipts:
            source = tmp_path / row["log_path"]
            copied = Path(destination).parent / "private_cli_logs" / (row["name"] + ".log")
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied)
            row.update(
                log_path=str(copied),
                passed=row["exit_code"] == (1 if row["name"] == "private_mutation" else 0),
                expected_exit=1 if row["name"] == "private_mutation" else 0,
            )
        atomic_json(Path(destination), dict(receipts=receipts))
    monkeypatch.setattr(e, "main", lambda: 0)
    runpy.run_path(cli, run_name="loaded_cli")
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(cli, run_name="__main__")
    assert stopped.value.code == 0


def test_private_capture_mutations(tmp_path):
    """REQ-REPORT-8106: private authenticated transport still rejects content and byte drift."""

    def bind(number, suffix, value):
        primary = tmp_path / f"results/experiment_{number}_v701_{suffix}.json"
        raw = primary.parent / "raw" / primary.stem
        value = deepcopy(value)
        value["terminal_validation_sidecar_path"] = str(raw / "terminal.json")
        atomic_json(primary, value)
        side = raw / "validators" / "fixture.json"
        atomic_json(side, dict(primary_sha256=e.sha256_file(primary), report=dict(passed=True)))
        atomic_json(raw / "terminal.json", dict(publication=dict(sidecar_path=str(side))))

    methods = json.loads(
        (e.ROOT / "results/experiment_8098_v701_development_methods.json").read_text()
    )
    capture = json.loads(
        (e.ROOT / "results/experiment_8102_v701_learning_stream_capture.json").read_text()
    )
    native = json.loads(
        (e.ROOT / "results/experiment_8105_v701_native_radial_kernel.json").read_text()
    )
    bind(8098, "development_methods", methods)
    bind(8105, "native_radial_kernel", dict(native, required_checks_passed=False))
    for name in ("unqualified", "bad_hash", "bad_content", "no_rows"):
        changed = deepcopy(capture)
        if name == "unqualified":
            changed["required_checks_passed"] = False
        elif name == "bad_hash":
            changed["capture_manifest"]["sha256"] = "changed"
        elif name == "bad_content":
            changed["rows"][0]["public_hash"] = "changed"
        else:
            changed["rows"] = []
        bind(8102, "learning_stream_capture", changed)
        data, checks, _ = e.inputs(tmp_path, tmp_path / name)
        assert not data["acquisition"]
        assert any(c["check"] == "capture_input" and c["expected"] != c["observed"] for c in checks)


def test_replay_durable_and_shard_mutations(tmp_path, monkeypatch, native, workload):
    """SCENARIO-VERIFY-8106: parity alone cannot authenticate a changed durable state."""
    evidence = e.measure(workload, native, tmp_path / "raw")
    data, _, _ = e.inputs(e.ROOT, tmp_path / "inputs")
    work = dict(
        data=workload,
        evidence=evidence,
        checks=[],
        refs=[],
        duration_s=1.0,
        phase_spans=[],
        code_config_hashes={},
        loaded_binding_receipt=data["library"],
    )
    value = e.build(work, [dict(passed=True, scope="owned")], tmp_path, e.reduce_rows(evidence))
    path = tmp_path / "candidate.json"
    row = value["paired_service_rows"][0]["arms"][0]
    saved = Path(row["state_path"]).read_bytes()
    changed = deepcopy(row["durable_state"])
    changed["version"] += 1
    atomic_json(Path(row["state_path"]), dict(state=changed, sha256=canonical_hash(changed)))
    atomic_json(path, value)
    assert not e.replay(path)
    Path(row["state_path"]).write_bytes(saved)
    changed_value = deepcopy(value)
    for r in changed_value["paired_service_rows"][0]["arms"]:
        r["probabilities"] = [0.123456]
    atomic_json(path, changed_value)
    assert not e.replay(path)
    ref = tmp_path / "evidence.json"
    atomic_json(ref, dict(primitive=True))
    value["raw_shard_hashes"] = [dict(path=str(ref), sha256="bad")]
    atomic_json(path, value)
    assert not e.replay(path)
    primitive = tmp_path / "primitive_rows.json"
    atomic_json(primitive, dict(paired_service_rows=[]))
    value["raw_shard_hashes"] = [dict(path=str(primitive), sha256=e.sha256_file(primitive))]
    atomic_json(path, value)
    assert not e.replay(path)


def test_bound_fixture_parent_receipts(tmp_path, monkeypatch):
    """REQ-REPORT-8106: bound work and current private receipts remain distinct from global health."""
    work = e.worker(tmp_path / "absent", tmp_path / "child")

    def completed_child(commands, raw, private):
        atomic_json(raw / "work.json", work)
        atomic_json(
            raw / "private_cli_receipts.json",
            dict(receipts=[dict(passed=True, scope="owned", duration_s=0.01)]),
        )
        atomic_json(
            raw / "repository_health_once.json",
            dict(receipts=[dict(passed=False, scope="repository_health")]),
        )
        return [dict(passed=True, scope="owned", duration_s=0.01)]

    monkeypatch.setattr(e, "execute", completed_child)
    observed = []
    monkeypatch.setattr(
        e,
        "publish_primary",
        lambda output, value, validator: observed.append(value) or dict(passed=True),
    )
    output = tmp_path / "experiment_8106_fixture.json"
    assert e.main(["--fixture-output", str(output)]) == 0
    assert observed[0]["fixture_protocol_only"]
    assert observed[0]["service_cost_ready_score"] == 0
    assert observed[0]["repository_health"][0]["passed"] is False
    assert observed[0]["validation_receipts"][-1]["passed"] is True


def test_cold_native_restore_mismatch(tmp_path, monkeypatch, native):
    """SCENARIO-VERIFY-8106: a loaded binding must retain complete serialized state."""
    from types import SimpleNamespace

    state = e.prior.fixture(9, 16, 1)
    checkpoint = tmp_path / "state.json"
    atomic_json(checkpoint, dict(state=state, sha256=canonical_hash(state)))
    path = tmp_path / "candidate.json"
    atomic_json(
        path,
        dict(
            paired_service_rows=[
                dict(arms=[dict(state_path=str(checkpoint), durable_state=state)])
            ],
            loaded_binding_receipt={},
        ),
    )

    class CorruptRestore:
        def __call__(self, encoded):
            return native.RustRadial8105(encoded)

        def restore(self, encoded):
            return SimpleNamespace(state_json=lambda: "{}")

    monkeypatch.setattr(e, "reduce_rows", lambda evidence: dict(passed=True))
    monkeypatch.setattr(
        e, "load_binding", lambda data: (SimpleNamespace(RustRadial8105=CorruptRestore()), {})
    )
    assert not e.replay(path)
