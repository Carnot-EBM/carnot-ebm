"""REQ-VERIFY-8124 / REQ-REPORT-8124: private protocol and production gates."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.verify import evidence_protocol_8124 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from test_methods_stream_custody_8111 import world, primary  # noqa: F401


def runtime_world(root):
    """Add tiny authenticated runtime operands, never a model implementation."""
    model = root / "cache/revision-fixture/model.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"private gguf identity operand")
    binary = root / "runtime"
    binary.write_bytes(b"private executable operand")
    binary.chmod(0o700)
    identity = dict(
        model_revision=model.parent.name,
        model_path=str(model),
        native_binary=dict(path=str(binary), sha256=sha256_file(binary)),
        model_identity_receipt=dict(props=dict(chat_template="private template")),
    )
    fields = dict(
        runtime_identity=identity,
        gguf_sha256=sha256_file(model),
        runtime_sha256=sha256_file(binary),
        chat_template_sha256=canonical_hash("private template"),
        MODEL_SPECS=["unsloth/Qwen3.8-27B-GGUF"],
    )
    history = root / e.STREAM
    value = json.loads(history.read_text())
    primary(history, dict(value, **fields))
    primary(root / e.ACQUISITION, dict(fields, experiment_id=8118))
    if not (root / e.METHODS).exists():
        work = e.qualified.measure(root, root / "private-methods", fixture=True)
        primary(root / e.METHODS, dict(work, experiment_id=8111))
    return model


def test_freeze_and_probability_quote_validation():
    """REQ-VERIFY-8124: offsets count UTF-8 bytes, never entailment labels."""
    config = e.protocol()
    assert len(config["features"]) == 12
    assert config["maximum_output_tokens_per_call"] == 128
    assert config["features"][1:9] == list(e.qualified.stream.lexical.FEATURES)
    source = "évidence fact.".encode()
    parsed = e.parse(
        json.dumps(dict(p_hallucination=0.2, quote="fact", byte_start=10, byte_end=14)),
        source,
        "source_span",
    )
    assert parsed["valid_quote"] == 1 and parsed["entailment_label"] is None
    assert parsed["quote_source_byte_ratio"] == 4 / len(source)
    for quote, start, end in [
        (None, None, None),
        (["fact"], 10, 14),
        ("fact", 9, 13),
        ("x " * 33, 0, 66),
    ]:
        result = e.parse(
            json.dumps(dict(p_hallucination=0.2, quote=quote, byte_start=start, byte_end=end)),
            source,
            "source_span",
        )
        assert result["valid_quote"] == 0 and result["quote_status"] != "valid"
        assert result["p_hallucination"] == 0.2
    for payload in [
        "bad",
        "[]",
        '{"p_hallucination":true}',
        '{"p_hallucination":2}',
        '{"p_hallucination":NaN}',
    ]:
        assert e.parse(payload, source, "holistic")["status"] == "missing_probability"
    assert e.parse('{"p_hallucination":0}', source, "holistic")["status"] == "parsed"


@pytest.mark.parametrize(
    "mutation", ["", "absent", "revision", "cache", "gguf", "runtime", "template"]
)
def test_production_precondition_identity(world, tmp_path, mutation):
    """SCENARIO-VERIFY-8124: expected nested revision exists before any comparison."""
    model = runtime_world(world)
    acquisition = world / e.ACQUISITION
    original = json.loads(acquisition.read_text())
    value = deepcopy(original)
    if mutation == "absent":
        value["runtime_identity"].pop("model_revision")
    elif mutation == "revision":
        value["runtime_identity"]["model_revision"] = "mutated"
    elif mutation in {"gguf", "runtime", "template"}:
        value[
            {
                "gguf": "gguf_sha256",
                "runtime": "runtime_sha256",
                "template": "chat_template_sha256",
            }[mutation]
        ] = "mutated"
    primary(acquisition, value)
    observed = model if mutation != "cache" else tmp_path / "mutated/model.gguf"
    plan = e.preconditions(world, tmp_path / "raw", fixture=True, model_path=observed)
    assert plan["expected_runtime_identity"]["model_revision"] == "revision-fixture"
    assert all(r["passed"] for r in plan["checks"]) == (mutation == "")
    if mutation in {"absent", "revision"}:
        failed = next(r for r in plan["checks"] if not r["passed"])
        assert failed["artifact_field"] == "runtime_identity.model_revision"
        assert failed["observed"] == (None if mutation == "absent" else "mutated")
    primary(acquisition, original)


def test_private_cli_and_replay(world, tmp_path):
    """SCENARIO-REPORT-8124: outside checkout routes exercise actual publication."""
    runtime_world(world)
    output = tmp_path / (e.NAME + ".json")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)

    def run(*args):
        command = [sys.executable]
        if env.get("COVERAGE_RCFILE"):
            command += ["-m", "coverage", "run", "--rcfile=" + env["COVERAGE_RCFILE"]]
        print("before private CLI subprocess", flush=True)
        done = subprocess.run(
            [*command, str(e.ROOT / e.CLI), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print("after private CLI subprocess exit=" + str(done.returncode), flush=True)
        return done

    done = run("--root", str(world), "--fixture-output", str(output))
    assert done.returncode == 0, done.stdout + done.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert (
        value["methods_ready_score"]
        == value["stream_input_ready_score"]
        == value["capture_protocol_ready_score"]
        == 1
    )
    assert value["MODEL_SPECS"] == value["call_ledger"] == []
    assert run("--cold-replay", str(output)).returncode == 0
    value["completed_count"] += 1
    atomic_json(output, value)
    assert run("--cold-replay", str(output)).returncode == 1
    done = run("--root", str(world), "--fixture-output", str(output), "--mutation", "labels")
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    (world / e.ACQUISITION).unlink()
    done = run("--root", str(world), "--fixture-output", str(output))
    assert done.returncode == 0, done.stdout + done.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked"
    assert value["methods_ready_score"] == value["stream_input_ready_score"] == 1
    assert value["capture_protocol_ready_score"] == 0


def test_freeze_precedes_evaluator_and_mutated_replay(world, tmp_path, monkeypatch):
    """REQ-VERIFY-8124: actual capture manifest is sealed before labels open."""
    runtime_world(world)
    original = e.qualified.measure

    def measure(root, raw, **kwargs):
        assert (raw / "evidence_capture_manifest.json").is_file()
        return original(root, raw, **kwargs)

    monkeypatch.setattr(e.qualified, "measure", measure)
    work = e.measure(world, tmp_path / "raw", fixture=True)
    assert len(work["protocol_conformance_rows"]) == 640
    assert all("y" not in r for r in work["protocol_conformance_rows"])
    receipts = [dict(passed=True)]
    raw = tmp_path / "raw"
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    value = e.build(work, raw, receipts, fixture=True)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.replay(output)
    shard = raw / "evidence_capture_manifest.json"
    old = shard.read_bytes()
    shard.write_bytes(b"changed")
    assert not e.replay(output)
    shard.write_bytes(old)
    work["code_config_hashes"][e.MODULE] = "changed"
    atomic_json(raw / "measurement.json", work)
    assert not e.replay(output)
    work["code_config_hashes"][e.MODULE] = sha256_file(e.ROOT / e.MODULE)
    log = tmp_path / "log"
    log.write_text("owned log")
    receipts = [dict(passed=True, log_path=str(log), log_sha256=sha256_file(log))]
    atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, e.build(work, raw, receipts, fixture=True))
    assert e.replay(output)
    log.write_text("changed log")
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent")
