"""REQ-REPORT-8146 / REQ-VERIFY-8146: fresh requests retain every original slot."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import runpy
from types import SimpleNamespace

import pytest

from carnot import experiment_8146_v704_live_service_cost as e


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """Only public manifests and sealed coefficients enter private test storage."""
    return e.inputs(e.ROOT, tmp_path_factory.mktemp("8146-input"))


@pytest.fixture(scope="module")
def native(data):
    """E2E-003 crosses the actual previously qualified native extension."""
    return e.host.prior.old.host.load_binding(data)[0]


def test_public_roster_and_independent_identity(data, tmp_path):
    """REQ-VERIFY-8146 preserves an ineligible slot and never opens labels."""
    assert data["ready"] and len(data["slots"]) == 24
    assert [r["slot"] for r in data["slots"]] == list(range(1, 25))
    assert not data["slots"][6]["public_eligible"]
    assert len(data["geometry"]["mean"]) == 9
    assert all("8118" not in r["path"] for r in data["refs"])
    assert e.CONFIG["output_tokens"] == 7168 and e.CONFIG["call_limit"] == 56
    missing = e.inputs(tmp_path, tmp_path / "raw")
    assert not missing["ready"] and missing["checks"][0]["observed"] is False


def test_transactions_parity_commit_and_failure(data, native, tmp_path):
    """SCENARIO-VERIFY-8146 charges generation and commits before returning."""
    ledger = e.Ledger(tmp_path / "ledger.json")
    rows = [
        e.transaction(
            data, data["slots"][0], arm, e.FixtureRuntime(), native, tmp_path / arm, ledger, arm
        )
        for arm in e.ARMS
    ]
    assert e.matched(rows) and ledger.counts()["generation_calls_completed"] == 2
    assert all(r["full_latency_ns"] >= sum(r["components"].values()) > 0 for r in rows)
    assert rows[0]["request_bytes"] == rows[1]["request_bytes"]
    assert rows[0]["response_bytes"] == rows[1]["response_bytes"]
    state = e.host.state_for(data["head"], data["geometry"])
    model = native.RustRadial8105(json.dumps(state))
    assert json.loads(native.RustRadial8105.restore(model.checkpoint()).state_json()) == state
    bad = deepcopy(rows)
    bad[0]["generated_text"] += "changed"
    assert not e.matched(bad)
    bad[0]["status"] = "failed"
    assert not e.matched(bad)
    failing = SimpleNamespace(
        count=lambda _: 1,
        generate=lambda _: (_ for _ in ()).throw(TimeoutError("private_deadline")),
    )
    row = e.transaction(
        data,
        data["slots"][0],
        "python_scalar",
        failing,
        native,
        tmp_path / "failed",
        ledger,
        "failed",
    )
    assert row["status"] == "failed" and "private_deadline" in row["exclusion_reason"]
    oversized = SimpleNamespace(count=lambda _: 6001)
    row = e.transaction(
        data,
        data["slots"][0],
        "python_scalar",
        oversized,
        native,
        tmp_path / "context",
        ledger,
        "context",
    )
    assert row["status"] == "excluded" and row["exclusion_reason"] == "context_over_budget"


def test_complete_panel_bootstrap_and_censoring(data, native, tmp_path):
    """REQ-VERIFY-8146 keeps failed pairs and treats repeats as dependent."""
    ledger = e.Ledger(tmp_path / "ledger.json")
    work = e.capture(data, e.FixtureRuntime(), native, tmp_path / "panel", ledger)
    reduced = e.reduce_rows(work)
    assert len(work["pairs"]) == 24 and len(work["warmups"]) == 8
    assert reduced["completed_count"] == 46 and reduced["excluded_count"] == 2
    assert reduced["matched_pair_count"] == 23 and reduced["complete_service_ready_score"] == 0
    assert reduced["paired_speed_interval"]["resamples"] == 10000
    full = deepcopy(work)
    full["pairs"][6]["arms"] = deepcopy(full["pairs"][0]["arms"])
    assert e.reduce_rows(full)["complete_service_ready_score"] == 1
    full["pairs"][0]["arms"][0]["generated_text"] += "drift"
    assert e.reduce_rows(full)["failed_pair_count"] == 1
    censored = e.capture(
        data,
        e.FixtureRuntime(),
        native,
        tmp_path / "censored",
        e.Ledger(tmp_path / "censored-ledger.json"),
        deadline_s=-1,
    )
    assert e.reduce_rows(censored)["censored_count"] == 46
    assert e.reduce_rows(censored)["paired_speed_interval"] is None


def test_build_and_replay_mutations(data, native, tmp_path):
    """SCENARIO-REPORT-8146 reopens primitive bytes and rejects aggregate drift."""
    raw = tmp_path / "raw"
    ledger = e.Ledger(raw / "ledger.json")
    work = e.capture(data, e.FixtureRuntime(), native, raw, ledger)
    result = dict(work=work, ledger=ledger.rows, checks=[], measured_duration_s=11)
    value = e.build(data, result, raw, [dict(passed=True, normal_exit=True)], 11, True)
    path = tmp_path / (e.NAME + ".json")
    e.atomic_json(path, value)
    assert e.replay(path) and value["MODEL_SPECS"] == [] and not value["call_ledger"]
    value["completed_count"] += 1
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(path, value)
    assert not e.replay(path)
    value = e.build(data, result, raw, [dict(passed=False, normal_exit=True)], 11, False)
    assert value["verdict_class"] == "disqualified" and value["complete_service_ready_score"] == 0
    absent = e.inputs(tmp_path / "missing", raw / "missing")
    value = e.build(absent, {}, raw / "blocked", [], 0.01, False)
    assert (
        value["verdict_class"] == "blocked"
        and value["inference_substrate_class"] == "no_model_load"
    )
    assert value["gate_check_summary"][0]["artifact_field"] == "resource_exists"


def test_private_cli_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8146 uses the script outside checkout without PYTHONPATH."""
    cli = [str(e.ROOT / ".venv/bin/python"), str(e.ROOT / e.CLI), "--date", "20261005"]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    output = tmp_path / (e.NAME + ".json")
    for extra in [["--private-small"], ["--root", str(tmp_path / "missing"), "--private-small"]]:
        run = subprocess.run(
            cli + extra + ["--output", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        replay = subprocess.run(
            cli + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert replay.returncode == 0
    value = json.loads(output.read_text())
    value["model_invocation_counts"]["generation_calls_completed"] = 1
    e.atomic_json(output, value)
    assert (
        subprocess.run(
            cli + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        ).returncode
        == 1
    )
    assert e.validation_plan(tmp_path)


def test_owned_main_and_terminal_failure(data, native, monkeypatch, tmp_path):
    """REQ-REPORT-8146 owned failures cannot silently grant complete readiness."""
    output = tmp_path / (e.NAME + ".json")
    seen = []

    def commands(plan, private):
        seen.extend(c.name for c in plan)
        return [dict(passed=True, normal_exit=True, name=c.name) for c in plan]

    monkeypatch.setattr(e.host.prior, "execute", commands)
    monkeypatch.setattr(e, "live", lambda *args: {})
    assert e.main(["--output", str(output)]) == 0
    assert "repository_health_once" in seen
    assert e.main(["--cold-replay", str(output)]) == 0
    assert e.main(["--private-small", "--output", str(output)]) == 0
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    with pytest.raises(SystemExit):
        e.main(["--private-small"])
    monkeypatch.setattr(
        e.host.prior, "execute", lambda plan, private: [dict(passed=False, normal_exit=True)]
    )
    assert e.main(["--output", str(output)]) == 1
    script = runpy.run_path(str(e.ROOT / e.CLI))
    assert script["ROOT"] == e.ROOT
    monkeypatch.setattr(e, "main", lambda: 0)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(e.ROOT / e.CLI), run_name="__main__")
    assert exit_info.value.code == 0


def test_runtime_identity_load_and_failure(data, native, monkeypatch, tmp_path):
    """SCENARIO-VERIFY-8146 model identity is checked before and after a load."""
    legacy = e.qualified.legacy
    spec = legacy.cached_current_model()
    model = Path(spec["model_path"])
    metadata = legacy.read_gguf_metadata(model)
    position = metadata["field_provenance"]["metadata_keys"]["tokenizer.chat_template"]
    with model.open("rb") as source:
        source.seek(position["value_offset"])
        size = int.from_bytes(source.read(8), "little")
        template = source.read(size).decode()

    class Runtime(e.FixtureRuntime):
        def __init__(self, *args):
            pass

        def load(self):
            return dict(authenticated=True, props=dict(chat_template=template), duration_s=3)

    def run(plan, raw, scratch):
        runtime = legacy.QwenRuntime(model, scratch, 0)
        try:
            identity = runtime.load()
        except ValueError:
            return dict(checks=[], measured_duration_s=3)
        rows = legacy.capture.capture(
            legacy.capture.freeze({}), runtime, raw, "fixture", started=e.time.monotonic()
        )
        assert len(rows) == 48
        return dict(
            checks=[dict(field="fixture", upstream_id="fixture", passed=True)],
            model_identity_receipt=identity,
            gpu_lease_receipt=dict(fixture=True),
            measured_duration_s=11,
            resident_gpu_receipt=dict(output_tail="10000"),
            unloaded_gpu_receipt=dict(output_tail="4"),
        )

    monkeypatch.setattr(legacy, "QwenRuntime", Runtime)
    monkeypatch.setattr(legacy, "live_capture", run)
    result = e.live(data, tmp_path / "live", tmp_path)
    assert len(result["ledger"]) == 55 and result["ledger"][0]["status"] == "completed"
    value = e.build(
        data, result, tmp_path / "live", [dict(passed=True, normal_exit=True)], 11, False
    )
    path = tmp_path / (e.NAME + ".json")
    e.atomic_json(path, value)
    assert e.replay(path) and value["gpu_memory_delta_mb"] == 9996
    assert value["inference_substrate_class"] == "model_bounded_generation"
    execution_path = Path(value["raw_shard_hashes"][2]["path"])
    execution = json.loads(execution_path.read_text())
    execution["ledger"][1]["request_sha256"] = "changed"
    e.atomic_json(execution_path, execution)
    value["call_ledger"] = execution["ledger"]
    value["raw_shard_hashes"][2] = e.reference(execution_path)
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(path, value)
    assert not e.replay(path)
    monkeypatch.setattr(Runtime, "load", lambda _: dict(props=dict(chat_template="changed")))
    failed = e.live(data, tmp_path / "failed-load", tmp_path)
    assert failed["ledger"][0]["status"] == "failed"
    assert (
        e.build(data, failed, tmp_path / "failed-load", [], 3, False)["inference_substrate_class"]
        == "model_load_no_generation"
    )
    changed = deepcopy(data)
    changed["protocol"]["chat_template_sha256"] = "missing"
    assert not e.live(changed, tmp_path / "bad-template", tmp_path)["ledger"]
    monkeypatch.setattr(legacy, "cached_current_model", lambda: {})
    assert not e.live(data, tmp_path / "missing-model", tmp_path)["ledger"]


def test_input_and_replay_faults(data, native, monkeypatch, tmp_path):
    """SCENARIO-REPORT-8146 rejects changed sources and rehashed false reductions."""
    root = tmp_path / "root"
    (root / "results").mkdir(parents=True)
    for name in e.PINS:
        (root / name).write_text("{")
    assert not e.inputs(root, tmp_path / "bad-json")["ready"]
    for name in e.PINS:
        (root / name).write_bytes((e.ROOT / name).read_bytes())
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *args: dict(report=dict(passed=True)))
    old_loads = e.json.loads

    def missing_ref(text):
        value = old_loads(text)
        if isinstance(value, dict) and "final_head_manifest" in value:
            value["final_head_manifest"]["path"] = str(tmp_path / "absent-head")
        return value

    with monkeypatch.context() as patcher:
        patcher.setattr(e.json, "loads", missing_ref)
        assert not e.inputs(root, tmp_path / "bad-reference")["ready"]
    raw = tmp_path / "replay"
    work = e.capture(data, e.FixtureRuntime(), native, raw, e.Ledger(raw / "fixture.json"))
    value = e.build(data, dict(work=work), raw, [], 2, True)
    path = tmp_path / (e.NAME + ".json")
    for field, replacement in [
        ("completed_count", 0),
        ("rows", []),
        ("call_ledger", [{}]),
        ("code_config_hashes", {e.MODULE: "changed"}),
        ("source_artifact_hashes", [dict(path=str(tmp_path / "absent"), sha256="x")]),
        ("source_artifact_hashes", [dict(path=str(e.ROOT / e.MODULE), sha256="changed")]),
    ]:
        bad = deepcopy(value)
        bad[field] = replacement
        bad["reproducibility_checksum"] = e.checksum(bad)
        e.atomic_json(path, bad)
        assert not e.replay(path)
    changed = deepcopy(work)
    changed["pairs"][0]["arms"][0]["values"][0] += 1
    e.atomic_json(raw / "primitive_rows.json", changed)
    bad = deepcopy(value)
    bad["rows"] = [r for p in changed["pairs"] for r in p["arms"]]
    bad["request_pair_rows"] = changed["pairs"]
    bad["raw_shard_hashes"][0] = e.reference(raw / "primitive_rows.json")
    bad["reproducibility_checksum"] = e.checksum(bad)
    e.atomic_json(path, bad)
    assert not e.replay(path)


def test_remaining_private_failure_boundaries(data, native, monkeypatch, tmp_path):
    """SCENARIO-REPORT-8146 dirty misses and forged reduction receipts fail closed."""
    ledger = e.Ledger(tmp_path / "ledger.json")
    original = e.atomic_json

    def dirty(path, value):
        if path.name == "initial.json":
            value = dict(value, cache={"existing": True})
        original(path, value)

    with monkeypatch.context() as patcher:
        patcher.setattr(e, "atomic_json", dirty)
        row = e.transaction(
            data,
            data["slots"][0],
            e.ARMS[0],
            e.FixtureRuntime(),
            native,
            tmp_path / "dirty",
            ledger,
            "dirty",
        )
        assert row["status"] == "failed" and "owned_cache_not_empty" in row["exclusion_reason"]
    invalid = SimpleNamespace(count=lambda _: 1, generate=lambda _: {})
    row = e.transaction(
        data, data["slots"][0], e.ARMS[0], invalid, native, tmp_path / "parse", ledger, "parse"
    )
    assert row["status"] == "failed" and "unusable_generation" in row["exclusion_reason"]
    raw = tmp_path / "raw"
    missing = e.inputs(tmp_path / "absent", raw)
    value = e.build(missing, {}, raw, [], 0.01, False)
    path = tmp_path / (e.NAME + ".json")
    bad = dict(value, completed_count=1)
    e.atomic_json(path, bad)
    assert not e.replay(path)
    reduction = Path(value["raw_shard_hashes"][3]["path"])
    e.atomic_json(reduction, dict(changed=True))
    value["raw_shard_hashes"][3] = e.reference(reduction)
    value["reproducibility_checksum"] = e.checksum(value)
    e.atomic_json(path, value)
    assert not e.replay(path)
