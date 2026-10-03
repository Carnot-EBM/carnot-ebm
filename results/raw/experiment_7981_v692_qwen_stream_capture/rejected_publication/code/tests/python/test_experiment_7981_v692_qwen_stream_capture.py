"""REQ-REPORT-7981: authenticated inputs and exact private CLI publication."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot import experiment_7981_v692_qwen_stream_capture as producer
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from test_qwen_stream_capture_7981 import views


def test_authentication_and_separate_fresh_gate(tmp_path):
    """SCENARIO-REPORT-7981-CUSTODY: fresh zero leaves stream available."""
    failures, upstream = producer.authenticate(producer.ROOT)
    assert not failures
    assert set(upstream["public_role_manifests"]) == set(producer.capture.ROLES)
    assert upstream["branch_gate_checks"][0]["observed"] == 0
    assert upstream["branch_gate_checks"][0]["expected"] == 1
    assert len(producer.load_public(upstream["public_role_manifests"])) == 4
    failures, _ = producer.authenticate(tmp_path)
    assert failures and all("path" in r and "op" in r for r in failures)


def test_fixture_replay_tampering_and_validation(tmp_path):
    """SCENARIO-REPORT-7981-CUSTODY: fixtures never supply live readiness."""
    fixture = tmp_path / "input.json"
    atomic_json(fixture, views(True))
    output = tmp_path / "fixture" / (producer.NAME + ".json")
    assert producer.main(["--fixture-input", str(fixture), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["experiment_id"] == 7981 and value["milestone"] == "2026.10.692"
    assert value["verdict_class"] == "circular_positive"
    assert value["stream_capture_ready_score"] == value["fresh_capture_ready_score"] == 0
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert read_bound_sidecar(output, Path(value["terminal_validation_sidecar_path"]))[
        "primary_sha256"
    ] == sha256_file(output)
    assert producer.main(["--cold-replay", str(output)]) == 0
    value["generated_tokens"] += 1
    bad = tmp_path / "bad.json"
    atomic_json(bad, value)
    assert producer.main(["--cold-replay", str(bad)]) == 1
    value = producer.base([])
    producer.apply_validation(value, [dict(passed=False, required=True)])
    assert value["verdict_class"] == "disqualified"
    assert value["stream_capture_ready_score"] == value["fresh_capture_ready_score"] == 0


def test_real_private_cli_routes(tmp_path):
    """SCENARIO-REPORT-7981-CUSTODY: cold replay needs no PYTHONPATH."""
    cli = str(producer.ROOT / producer.OWNED[2])
    py = str(producer.ROOT / ".venv/bin/python")
    output = tmp_path / "blocked" / (producer.NAME + ".json")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for args, expected in [
        (["--root", str(tmp_path / "missing"), "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--date", "20260930"], 2),
    ]:
        child = subprocess.run(
            [py, "-u", cli, *args], cwd=tmp_path, env=env, capture_output=True, timeout=60
        )
        assert child.returncode == expected, child.stdout.decode() + child.stderr.decode()
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["MODEL_SPECS"] == []
    assert value["stream_capture_ready_score"] == 0


def test_build_branch_verdict_and_current_counts(tmp_path):
    """SCENARIO-VERIFY-7981-BRANCHES: resumed work cannot inflate calls."""
    from test_qwen_calibration_capture_7969 import CountingRuntime

    raw = tmp_path / "raw"
    slots = producer.capture.freeze(views())
    rows = producer.capture.capture(slots, CountingRuntime(), raw / "slots", "identity")
    atomic_json(raw / "request_manifest.json", dict(rows=slots, capture_identity="identity"))
    result = dict(
        rows=rows,
        capture_identity="identity",
        model_loads_attempted=1,
        model_loads_completed=1,
        model_identity_receipt=dict(offload_layers=[65, 65]),
        resident_gpu_receipt={},
        measured_duration_s=11,
        current_family_ids=[rows[0]["family_id"]],
    )
    failures, upstream = producer.authenticate(producer.ROOT)
    assert not failures
    with producer.qualified_protocol():
        value = producer.build(raw, result, upstream)
    assert value["verdict_class"] == "null" and value["stream_capture_ready_score"] == 1
    assert value["fresh_capture_ready_score"] == 0
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 1
    assert value["resumed_invocation_counts"]["generation_calls_attempted"] == 223
    broken = deepcopy(result)
    broken["measured_duration_s"] = 1
    with producer.qualified_protocol():
        assert producer.build(raw, broken, upstream)["verdict_class"] == "disqualified"


def private_inputs(tmp_path, monkeypatch, *, stream=True, fresh=True):
    """Scripted custody tests change private bytes, never production evidence."""
    original = json.loads((producer.ROOT / producer.UPSTREAM).read_text())
    original.update(feature_views_ready_score=int(stream), fresh_panel_ready_score=int(fresh))
    manifests = {}
    data = views(True)
    for role in producer.capture.ROLES:
        path = tmp_path / "public" / (role + ".json")
        atomic_json(path, data[role])
        manifests[role] = producer.reference(path)
    fresh_path = tmp_path / "public/fresh.json"
    atomic_json(fresh_path, dict(request_rows=data[producer.capture.FRESH]["request_rows"]))
    original.update(
        public_role_manifests=manifests, fresh_public_manifest=producer.reference(fresh_path)
    )
    atomic_json(tmp_path / producer.UPSTREAM, original)
    atomic_json(
        tmp_path / producer.HISTORY, json.loads((producer.ROOT / producer.HISTORY).read_text())
    )
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops/exclusion_manifest.yaml").write_text("retired: []\n")
    monkeypatch.setattr(producer, "UPSTREAM_PIN", sha256_file(tmp_path / producer.UPSTREAM))
    return original


def test_authentication_fresh_only_and_changed_protocol(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7981-CUSTODY: branch absence and hash failure differ."""
    private_inputs(tmp_path, monkeypatch, stream=False)
    failures, upstream = producer.authenticate(tmp_path)
    assert not failures
    assert set(upstream["public_role_manifests"]) == {producer.capture.FRESH}
    history = json.loads((tmp_path / producer.HISTORY).read_text())
    history["capture_budget"]["seed"] = 1
    atomic_json(tmp_path / producer.HISTORY, history)
    monkeypatch.setattr(producer, "HISTORY_PIN", sha256_file(tmp_path / producer.HISTORY))
    failures, _ = producer.authenticate(tmp_path)
    assert any(r["field"] == "decoder_protocol" for r in failures)


def test_bad_fresh_bytes_do_not_suppress_stream(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-7981-BRANCHES: stream survives sibling failure."""
    original = private_inputs(tmp_path, monkeypatch)
    Path(original["fresh_public_manifest"]["path"]).write_text("{}")
    failures, upstream = producer.authenticate(tmp_path)
    assert not failures
    assert set(upstream["public_role_manifests"]) == set(producer.capture.ROLES)
    assert any(not r["passed"] for r in upstream["branch_gate_checks"])


def test_live_adapter_and_manifest(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7981-CUSTODY: reuse keeps current counters honest."""
    raw = tmp_path / "raw"
    atomic_json(raw / "slots/slot-000.json", dict(family_id="old", started=True))
    result = dict(rows=[dict(family_id="old", started=True), dict(family_id="new", started=True)])
    monkeypatch.setattr(producer, "_LIVE", lambda *args: result)
    assert producer.live_capture({}, raw, tmp_path)["current_family_ids"] == ["new"]
    with producer.qualified_protocol():
        manifest = producer.freeze_commands(raw, tmp_path, views())
    assert any(r["name"] == "e2e019_private" and r["required"] for r in manifest["commands"])
    assert manifest["affected_files"] == producer.OWNED


def test_suppressed_fresh_readiness_and_reader_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7981-CUSTODY: invalid claims cannot self-certify."""
    source = tmp_path / "input.json"
    atomic_json(source, views(True))
    output = tmp_path / "fixture" / (producer.NAME + ".json")
    assert producer.main(["--fixture-input", str(source), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    value["fresh_capture_ready_score"] = 1
    with pytest.raises(ValueError, match="unsafe_readiness"):
        producer.replay(value)
    value.update(
        verdict_class="null",
        inference_mode="live_gpu",
        validation_pending=True,
        flagged_adversarial=False,
    )
    value["qwen_capture_ready_score"] = 1
    value["stream_capture_ready_score"] = 1
    producer.replay(value)
    monkeypatch.setattr(producer.legacy.prior.targets, "check_receipts", lambda _: None)
    value["validation_pending"] = False
    producer.replay(value)
    monkeypatch.setattr(producer, "reader_receipt", lambda *a, **k: dict(passed=False))
    monkeypatch.setattr(producer.legacy, "terminal_check", lambda _: dict(passed=True))
    with pytest.raises(ValueError, match="primary_resolution"):
        producer.publish(tmp_path / "reject" / output.name, producer.base([]))


def test_no_branch_and_fresh_duration_floor(tmp_path):
    """SCENARIO-VERIFY-7981-BRANCHES: zero usable sources cannot qualify."""
    from test_qwen_calibration_capture_7969 import CountingRuntime

    raw = tmp_path / "raw"
    slots = producer.capture.freeze({producer.capture.FRESH: views(True)[producer.capture.FRESH]})
    rows = producer.capture.capture(slots, CountingRuntime(), raw / "slots", "identity")
    result = dict(
        rows=rows,
        model_loads_attempted=1,
        model_loads_completed=1,
        model_identity_receipt=dict(offload_layers=[65, 65]),
        resident_gpu_receipt={},
        measured_duration_s=1,
    )
    with producer.qualified_protocol():
        value = producer.build(raw, result, {})
    assert value["verdict_class"] == "disqualified" and value["fresh_capture_ready_score"] == 0
    result["rows"] = result["rows"][:1]
    result["measured_duration_s"] = 11
    with producer.qualified_protocol():
        value = producer.build(raw, result, {})
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]


def test_missing_protocol_fields_remain_exact_gates(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7981-CUSTODY: malformed producer data fail closed."""
    private_inputs(tmp_path, monkeypatch)
    history = json.loads((tmp_path / producer.HISTORY).read_text())
    history.pop("capture_budget")
    atomic_json(tmp_path / producer.HISTORY, history)
    monkeypatch.setattr(producer, "HISTORY_PIN", sha256_file(tmp_path / producer.HISTORY))
    failures, _ = producer.authenticate(tmp_path)
    assert any(r["field"] == "authenticated_public_protocol" for r in failures)


def test_owned_generated_token_receipts(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7981-CUSTODY: output use binds to owned transport."""
    from carnot.reporting.current_work_receipt import canonical_hash
    from test_qwen_calibration_capture_7969 import CountingRuntime

    rows = producer.capture.capture(
        producer.capture.freeze(views())[:1], CountingRuntime(), tmp_path / "input", "identity"
    )
    receipt = dict(
        request_sha256=canonical_hash(rows[0]["request"]),
        response_sha256=canonical_hash(rows[0]["raw_response"]),
        output_tokens=rows[0]["raw_response"]["usage"]["completion_tokens"],
    )
    result = dict(rows=rows, model_loads_completed=1, runtime_receipts=[receipt], checks=[])
    monkeypatch.setattr(producer, "_LIVE", lambda *args: result)
    observed = producer.live_capture({}, tmp_path / "capture", tmp_path)
    assert all(r["passed"] for r in observed["checks"])
    receipt["output_tokens"] += 1
    observed = producer.live_capture({}, tmp_path / "capture", tmp_path)
    assert any(
        not r["passed"] and r["field"] == "generated_token_receipt_binding"
        for r in observed["checks"]
    )
