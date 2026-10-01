"""REQ-REPORT-7958: private publication, ownership and cold replay."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7958_v690_qwen_response_risk as producer
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from test_qwen_response_risk_7958 import Runtime, labels, public


def fixture(path):
    value = dict(public=public(), labels=labels())
    atomic_json(path, value)
    return value


def test_private_fixture_replay_and_primary(tmp_path):
    """SCENARIO-REPORT-7958-TERMINAL: exact primary bytes reach both readers."""
    source = tmp_path / "input.json"
    fixture(source)
    output = tmp_path / "success" / (producer.NAME + ".json")
    assert producer.main(["--fixture-input", str(source), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["qwen_response_measurement_ready_score"] == 0
    assert producer.replay(value)["sample_size_budget"]["completed"] == 128
    shard = Path(value["raw_response_shards"][0]["path"])
    saved = json.loads(shard.read_text())
    changed_raw = deepcopy(saved)
    changed_raw["rows"][0]["request"]["seed"] = 1
    atomic_json(shard, changed_raw)
    changed = {**value, "raw_response_shards": [producer.reference(shard)]}
    with pytest.raises(ValueError, match="request_drift"):
        producer.replay(changed)
    atomic_json(shard, saved)
    manifest = shard.parent / "incorrect-frozen.json"
    atomic_json(manifest, {"rows": [], "config": {}})
    changed = {**value, "frozen_manifest": producer.reference(manifest)}
    with pytest.raises(ValueError, match="request_manifest_drift"):
        producer.replay(changed)
    assert read_bound_sidecar(output, Path(value["terminal_validation_sidecar_path"]))
    receipt = reader_receipt(
        producer.TASK, output.parent, field="qwen_response_measurement_ready_score", expected=0
    )
    assert receipt["passed"] and receipt["gate_sha256"] == sha256_file(output)
    assert (
        producer.main(
            ["--cold-replay", str(output), "--output", str(tmp_path / "replay" / "result.json")]
        )
        == 0
    )
    changed = deepcopy(value)
    changed["probability_metrics"]["complete_pairs"] -= 1
    atomic_json(tmp_path / "bad.json", changed)
    assert (
        producer.main(
            [
                "--cold-replay",
                str(tmp_path / "bad.json"),
                "--output",
                str(tmp_path / "negative" / "result.json"),
            ]
        )
        == 1
    )


def test_external_block_date_and_required_failure(tmp_path):
    """SCENARIO-REPORT-7958-TERMINAL: blocked work claims zero inference."""
    output = tmp_path / "blocked" / (producer.NAME + ".json")
    assert producer.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    assert (
        value["model_specs"] == []
        and value["model_invocation_counts"]["generation_calls_attempted"] == 0
    )
    assert producer.replay(value)["rows"] == []
    with pytest.raises(SystemExit):
        producer.main(["--date", "20260929"])
    producer.apply_validation(value, [dict(required=True, passed=False, command_argv=[])])
    assert value["verdict_class"] == "disqualified"


def test_frozen_manifest_and_authentication(tmp_path):
    """REQ-REPORT-7958: no historical publishing against current authorities."""
    manifest = producer.freeze_commands(tmp_path)
    assert manifest["coverage_includes"] == producer.INCLUDE
    for c in manifest["commands"]:
        if c["name"].startswith("e2e016"):
            assert c["argv"][c["argv"].index("--date") + 1] == "20260929"
    failed, upstream = producer.authenticate(producer.ROOT)
    assert not failed and len(upstream["public"]) == 64
    failed, _ = producer.authenticate(tmp_path)
    assert failed[0]["observed"] is None
    assert len(producer.human_targets(upstream)) == 64
    assert (
        producer.execute_commands(
            {"commands": [], "coverage_file": str(tmp_path / "coverage")}, tmp_path
        )
        == []
    )


def test_live_orchestration_uses_sealed_raw_before_labels(tmp_path, monkeypatch):
    """REQ-REPORT-7958: current compute is separate from fixtures and history."""
    path = tmp_path / "input.json"
    data = fixture(path)
    _, upstream = producer.authenticate(producer.ROOT)
    upstream = {**upstream, "public": data["public"]}
    monkeypatch.setattr(producer, "authenticate", lambda root: ([], upstream))
    monkeypatch.setattr(
        producer,
        "live_capture",
        lambda frozen, raw: (
            dict(
                authenticated=True,
                model_loads_attempted=1,
                model_loads_completed=1,
                measured_duration_s=11.0,
                offload_layers=[65, 65],
            ),
            producer.risk.capture(producer.risk.freeze(frozen, len), Runtime(), raw),
            [],
            [],
        ),
    )
    monkeypatch.setattr(producer, "human_targets", lambda u: data["labels"])
    monkeypatch.setattr(
        producer, "authenticate", lambda root: ([], {**upstream, "public": data["public"]})
    )
    monkeypatch.setattr(producer, "execute_commands", lambda m, r: [])
    monkeypatch.setattr(producer, "publish", lambda output, value: atomic_json(output, value))
    output = tmp_path / (producer.NAME + ".json")
    assert producer.run(producer.ROOT, output, validate=False) == 0
    value = json.loads(output.read_text())
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 128
    assert value["verdict_class"] == "null"


def test_owned_cuda_paths_and_cleanup(tmp_path, monkeypatch):
    """REQ-REPORT-7958: cache, capacity, lease and load failures stay explicit."""
    from carnot import gpu_lease_phase_journal as leases
    from carnot.inference import sota_models

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF fixture")
    spec = dict(hf_id=producer.risk.MODEL, model_path=str(model))
    monkeypatch.setattr(sota_models, "cached_current_model", lambda: None)
    assert not producer.live_capture(public(1), tmp_path)[1]
    monkeypatch.setattr(sota_models, "cached_current_model", lambda: spec)
    monkeypatch.setattr(producer, "read_gguf_metadata", lambda p: dict(quantization="Q4_K_M"))
    assert not producer.live_capture(public(1), tmp_path)[1]
    monkeypatch.setattr(producer, "MODEL_PIN", sha256_file(model))
    monkeypatch.setattr(
        producer, "read_gguf_metadata", lambda p: (_ for _ in ()).throw(ValueError("header"))
    )
    assert not producer.live_capture(public(1), tmp_path)[1]
    monkeypatch.setattr(producer, "read_gguf_metadata", lambda p: {"quantization": "Q4_K_M"})
    monkeypatch.setattr(
        producer,
        "run_commands",
        lambda *a, **k: [dict(passed=True, output_tail="", log_path=str(model))],
    )
    assert not producer.live_capture(public(1), tmp_path)[1]
    monkeypatch.setattr(
        producer,
        "run_commands",
        lambda *a, **k: [dict(passed=True, output_tail="0, uuid, 4, 24000", log_path=str(model))],
    )

    class Lease:
        document = {"phase": "preflight"}

        def owner_receipt(self):
            return {"owned": True}

        def transition(self, phase, **kwargs):
            self.document["phase"] = phase

        def release(self):
            return {"released": True}

    class Native(Runtime):
        def __init__(self, path, scratch, gpu):
            super().__init__()
            self.log = scratch / "native.log"
            self.log.write_text("offloaded 65/65 layers to GPU")
            self.receipts = []
            self.command = [str(model)]
            self.worker = type("Worker", (), {"receipt": {"pid": 42}})()

        def load(self):
            return {"authenticated": True, "offload_layers": [65, 65]}

        def count(self, text):
            return 30

        def close(self):
            return {"leak_free": True}

    monkeypatch.setattr(leases.GpuLease, "acquire", lambda **k: Lease())
    monkeypatch.setattr(producer, "QwenRuntime", Native)
    monkeypatch.setattr(producer.completion.prior, "gpu_memory", lambda *a: (100, {}))
    identity, rows, checks, frozen = producer.live_capture(public(1), tmp_path)
    assert len(rows) == 2 and frozen and all(r["passed"] for r in checks)
    assert identity["device"]["pid"] == 42 and identity["gpu_lease"]["release"]["released"]
    monkeypatch.setattr(Native, "load", lambda self: (_ for _ in ()).throw(RuntimeError("load")))
    identity, rows, checks, frozen = producer.live_capture(public(1), tmp_path)
    assert not rows and any(not r["passed"] for r in checks)
    monkeypatch.setattr(
        leases.GpuLease, "acquire", lambda **k: (_ for _ in ()).throw(RuntimeError("lease"))
    )
    identity, rows, checks, frozen = producer.live_capture(public(1), tmp_path)
    assert identity["cleanup"]["leak_free"] and any(not r["passed"] for r in checks)


def test_cold_integrity_readiness_and_owned_outcomes(tmp_path, monkeypatch):
    """REQ-REPORT-7958: receipts and original rows bind readiness and benefit."""
    source = tmp_path / "input.json"
    data = fixture(source)
    raw = tmp_path / "raw"
    rows = producer.risk.capture(producer.risk.freeze(data["public"], len), Runtime(), raw)
    for row in rows:
        if row["arm"] == "full_source":
            from test_qwen_response_risk_7958 import reply

            row["raw_response"] = reply(int(row["family_id"]) % 2)
            row["parsed"] = producer.risk.transport.parse_response(
                row["raw_response"], row["visible_ids"]
            )
    _, upstream = producer.authenticate(producer.ROOT)
    value = producer.build(
        rows, data["labels"], raw, dict(model_loads_attempted=1, model_loads_completed=1), upstream
    )
    assert value["verdict_class"] == "positive"
    monkeypatch.setattr(producer, "human_targets", lambda u: data["labels"])
    monkeypatch.setattr(
        producer, "authenticate", lambda root: ([], {**upstream, "public": data["public"]})
    )
    assert producer.replay(value)["qwen_response_benefit_score"] == 1
    monkeypatch.setattr(producer.targets, "check_receipts", lambda v: None)
    value["qwen_response_measurement_ready_score"] = 1
    value["flagged_adversarial"] = True
    with pytest.raises(ValueError, match="unsafe_readiness"):
        producer.replay(value)
    monkeypatch.setattr(producer, "authenticate", lambda root: ([{}], {}))
    with pytest.raises(ValueError, match="cold_custody"):
        producer.replay(value)
    empty = producer.base([])
    empty["qwen_response_measurement_ready_score"] = 1
    with pytest.raises(ValueError, match="unsafe_readiness"):
        producer.replay(empty)


def test_run_block_and_validation_paths(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7958-TERMINAL: owned failures cannot keep readiness."""
    _, upstream = producer.authenticate(producer.ROOT)
    monkeypatch.setattr(producer, "publish", lambda output, value: atomic_json(output, value))
    monkeypatch.setattr(
        producer, "authenticate", lambda root: (_ for _ in ()).throw(ValueError("custody"))
    )
    assert producer.run(tmp_path, tmp_path / "a.json") == 0
    monkeypatch.setattr(producer, "authenticate", lambda root: ([], upstream))
    monkeypatch.setattr(producer, "live_capture", lambda p, r: ({}, [], [{"passed": False}], []))
    assert producer.run(tmp_path, tmp_path / "b.json") == 0
    data = dict(public=public(), labels=labels())
    monkeypatch.setattr(producer, "human_targets", lambda u: data["labels"])
    monkeypatch.setattr(
        producer,
        "live_capture",
        lambda p, r: (
            {"model_loads_attempted": 1, "model_loads_completed": 1},
            producer.risk.capture(producer.risk.freeze(data["public"], len), Runtime(), r),
            [{"passed": False}],
            [],
        ),
    )
    monkeypatch.setattr(producer, "execute_commands", lambda manifest, raw: [])
    atomic_json(tmp_path / "raw" / producer.NAME / "coverage.json", {"files": {"measured": 1}})
    assert producer.run(tmp_path, tmp_path / "c.json") == 0
    assert json.loads((tmp_path / "c.json").read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(
        producer,
        "live_capture",
        lambda p, r: (
            {"model_loads_attempted": 1, "model_loads_completed": 0},
            [],
            [{"passed": False}],
            [],
        ),
    )
    assert producer.run(tmp_path, tmp_path / "d.json") == 0
    assert (
        json.loads((tmp_path / "d.json").read_text())["model_invocation_counts"][
            "model_loads_attempted"
        ]
        == 1
    )


def test_primary_resolution_failure_is_not_hidden(tmp_path, monkeypatch):
    """REQ-REPORT-7958: both actual consumers must agree before completion."""
    monkeypatch.setattr(
        producer, "publish_primary", lambda *a: {"primary_sha256": "x", "sidecar_path": "x"}
    )
    monkeypatch.setattr(producer, "reader_receipt", lambda *a, **k: {"passed": False})
    with pytest.raises(ValueError, match="primary_resolution"):
        producer.publish(tmp_path / (producer.NAME + ".json"), producer.base([]))
