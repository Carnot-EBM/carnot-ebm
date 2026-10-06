"""REQ-REPORT-8179 / SCENARIO-REPORT-8179-CUSTODY: private frozen evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from test_sentence_methods_8166 import world as world

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import sentence_transport_methods_8179 as e


@pytest.fixture
def inputs(world):
    """Retain upstream fixture assertions and add separate failed-capture receipts."""
    for name in e.PINS:
        atomic_json(
            world / name,
            dict(
                required_checks_passed=True,
                flagged_adversarial=False,
                raw_shard_hashes=[],
                MODEL_SPECS=["imported-only"],
                model_invocation_counts={"generate": 1},
            ),
        )
    return world


def test_measure_readiness_and_replay(inputs, tmp_path):
    """REQ-REPORT-8179: readiness concerns frozen methods, never semantic benefit."""
    raw = tmp_path / "raw"
    work = e.measure(inputs, raw, fixture=True)
    value = e.build(work, raw, [dict(passed=True)], fixture=True)
    assert value["sentence_protocol_ready_score"] == 1
    assert (
        value["completed_count"] == 320
        and value["eligibility_by_role"]["evaluation"]["eligible"] == 128
    )
    assert value["canary_source_ids"] == work["config"]["canary"]["source_ids"]
    assert value["MODEL_SPECS"] == [] and value["call_ledger"] == []
    assert value["verifier_is_oracle"] and value["independent_generalization_score"] == 0
    assert len(value["feature_schema"]) == 16 and value["H1"]["minimum_sources"] == 96
    path = tmp_path / "primary.json"
    atomic_json(path, value)
    assert e.replay(path)
    for key in ["completed_count", "H1", "source_partition_rows"]:
        bad = deepcopy(value)
        bad[key] = "changed"
        atomic_json(path, bad)
        assert not e.replay(path)
    atomic_json(path, value)
    work["rows"][0]["source_bytes"] = b"Tampered.".hex()
    atomic_json(raw / "measurement.json", work)
    forged = e.build(work, raw, [dict(passed=True)], fixture=True)
    atomic_json(path, forged)
    assert not e.replay(path)
    assert not e.replay(tmp_path / "missing")
    assert e.build(work, raw, [dict(passed=False)])["sentence_protocol_ready_score"] == 0


def test_external_missing_and_source_tamper(inputs, tmp_path):
    """SCENARIO-REPORT-8179-CUSTODY: outside evidence blocks terminally."""
    (inputs / next(iter(e.PINS))).unlink()
    work = e.measure(inputs, tmp_path / "absent", fixture=True)
    value = e.build(work, tmp_path / "absent", [dict(passed=True)], fixture=True)
    assert value["verdict_class"] == "blocked"
    assert any(not c["passed"] for c in value["gate_check_summary"])
    changed = e.measure(inputs, tmp_path / "tamper", fixture=True, mutation="source")
    assert any(
        c["check"] == "private_source_tamper" and not c["passed"] for c in changed["plan"]["checks"]
    )


def test_manifest_and_outside_cli(inputs, tmp_path, monkeypatch):
    """E2E-019: direct execution uses this checkout with no PYTHONPATH."""
    private = tmp_path / "private"
    private.mkdir()
    specs = e.manifest(private, private / "candidate.json")
    assert e.TEST in specs["commands"][0]["argv"]
    for i in (5, 6, 7, 8):
        assert all("::" not in x for x in specs["commands"][i]["argv"])
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    output = tmp_path / (e.NAME + ".json")
    prefix = [sys.executable]
    if env.get("COVERAGE_RCFILE"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + env["COVERAGE_RCFILE"]]
    cli = prefix + [str(e.ROOT / e.CLI)]

    def run(argv):
        return subprocess.run(
            cli + argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )

    result = run(["--root", str(inputs), "--fixture-output", str(output)])
    assert result.returncode == 0, result.stderr
    assert run(["--cold-replay", str(output)]).returncode == 0
    saved = json.loads(output.read_text())
    saved["completed_count"] = -1
    atomic_json(output, saved)
    assert run(["--cold-replay", str(output)]).returncode == 1
    monkeypatch.setattr(e.execution, "main", lambda argv: 0)
    assert e.main([]) == 0


def test_vocabulary_only_and_missing_tokenizer(inputs, tmp_path, monkeypatch):
    """REQ-VERIFY-8179: vocabulary preflight never calls neural generation."""
    import llama_cpp

    path = tmp_path / "vocabulary.gguf"
    path.write_bytes(b"private vocabulary fixture")
    monkeypatch.setattr(e, "TOKENIZER_PATH", path)
    monkeypatch.setattr(e, "TOKENIZER_PIN", sha256_file(path))

    class Vocabulary:
        def tokenize(self, data, **kwargs):
            assert kwargs == dict(add_bos=False, special=False)
            return list(data)

    monkeypatch.setattr(llama_cpp, "Llama", lambda **kwargs: Vocabulary())
    plan = dict(checks=[])
    count, receipt = e.tokenizer(plan)
    assert count("é") == 2 and receipt["neural_weights_loaded"] is False
    path.unlink()
    count, receipt = e.tokenizer(plan)
    assert count("é") == 2 and receipt["status"] == "blocked"
    assert not plan["checks"][-1]["passed"]


def test_authenticated_inputs_and_live_preflight(inputs, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8179-CUSTODY: verify actual sidecars in a private world."""
    pins = {}
    for name in [*e.methods.PINS, *e.PINS]:
        path = inputs / name
        value = json.loads(path.read_text())
        raw = path.parent / "raw" / path.stem
        terminal = raw / "terminal.json"
        sidecar = raw / "validators/check.json"
        value["terminal_validation_sidecar_path"] = str(terminal)
        atomic_json(path, value)
        atomic_json(terminal, dict(publication=dict(sidecar_path=str(sidecar))))
        atomic_json(sidecar, dict(primary_sha256=sha256_file(path), report=dict(passed=True)))
        pins[name] = sha256_file(path)
    monkeypatch.setattr(e.methods, "PINS", {n: pins[n] for n in e.methods.PINS})
    monkeypatch.setattr(e, "PINS", {n: pins[n] for n in e.PINS})
    monkeypatch.setattr(
        e,
        "tokenizer",
        lambda plan, fixture=False: (lambda text: len(text.encode()), dict(status="private")),
    )
    work = e.measure(inputs, tmp_path / "authenticated")
    assert work["baseline"]["selected_control"] == "radial16"
    assert work["plan"]["checks"][-1]["check"] == "canary_preflight_source_count"
    first = inputs / next(iter(e.PINS))
    first.write_text("{}")
    plan = e.inputs(inputs, tmp_path / "changed")
    assert any(not c["passed"] and c["check"] == "upstream_sha256" for c in plan["checks"])


def test_replay_rejects_log_config_baseline_and_primitive_tamper(inputs, tmp_path):
    """REQ-REPORT-8179: rehashed rows cannot replace original bytes or controls."""
    raw = tmp_path / "raw"
    work = e.measure(inputs, raw, fixture=True)
    log = tmp_path / "owned.log"
    log.write_text("normal exit")
    receipts = [dict(passed=True, log_path=str(log), log_sha256=sha256_file(log))]
    value = e.build(work, raw, receipts, fixture=True)
    path = tmp_path / "primary.json"
    atomic_json(path, value)
    assert e.replay(path)
    log.write_text("altered")
    assert not e.replay(path)
    log.write_text("normal exit")
    original = deepcopy(work)
    for kind in ("config", "baseline_rows", "baseline", "tokenizer"):
        work = deepcopy(original)
        if kind == "config":
            work["config"]["H1"]["minimum_sources"] = 1
        elif kind == "baseline_rows":
            work["plan"]["baselines"][0]["numerator"] = 7
        elif kind == "baseline":
            work["baseline"]["selected_control"] = "scalar_span"
        else:
            work["tokenizer_receipt"]["status"] = "forged"
        atomic_json(raw / "measurement.json", work)
        atomic_json(path, e.build(work, raw, receipts, fixture=True))
        assert not e.replay(path)
    Path(value["raw_shard_hashes"][0]["path"]).write_text("changed primitive")
    atomic_json(path, value)
    assert not e.replay(path)


def test_output_overflow_keeps_original_slot(monkeypatch):
    """REQ-VERIFY-8179: any bound overflow escalates without removing the source."""
    from test_sentence_transport_8179 import source

    monkeypatch.setattr(
        e.transport, "output_budget", lambda *args: dict(maximum_encoded_output_tokens=257)
    )
    rows = e.freeze_sources(
        [dict(source(), unit_id="one", source_cluster_id="one", role="fit")], len
    )
    assert rows[0]["status"] == "excluded"
    assert rows[0]["exclusion_reason"] == "output_token_limit" and not rows[0]["requests"]


def test_missing_source_world_cold_replay(tmp_path):
    """E2E-015/019: absent external input is a completed blocked receipt."""
    raw = tmp_path / "raw"
    work = e.measure(tmp_path / "missing-world", raw, fixture=True)
    value = e.build(work, raw, [dict(passed=True)], fixture=True)
    path = tmp_path / "blocked.json"
    atomic_json(path, value)
    assert value["verdict_class"] == "blocked" and value["failed_count"] == 320
    assert e.replay(path)


def test_upstream_malformed_structure_blocks(inputs, tmp_path):
    """SCENARIO-REPORT-8179-CUSTODY: malformed receipts do not leave owned work partial."""
    (inputs / next(iter(e.PINS))).write_text("{}")
    plan = e.inputs(inputs, tmp_path / "raw", fixture=True)
    assert any(c["check"] == "upstream_structure" and not c["passed"] for c in plan["checks"])


def test_vocabulary_failure_blocks(tmp_path, monkeypatch):
    """REQ-REPORT-8179: an unavailable tokenizer has an explicit observed operand."""
    import llama_cpp

    path = tmp_path / "bad.gguf"
    path.write_bytes(b"not a gguf")
    monkeypatch.setattr(e, "TOKENIZER_PATH", path)
    monkeypatch.setattr(e, "TOKENIZER_PIN", sha256_file(path))

    def fail(**kwargs):
        raise ValueError("invalid vocabulary")

    monkeypatch.setattr(llama_cpp, "Llama", fail)
    plan = dict(checks=[])
    count, receipt = e.tokenizer(plan)
    assert count("x") == 1 and receipt["status"] == "blocked"
    assert plan["checks"][-1]["observed"] == "invalid vocabulary"


def test_vocabulary_owner_survives_until_count_finishes(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8179-TRANSPORT: keep the native vocabulary owner alive."""
    import gc
    import weakref
    import llama_cpp

    path = tmp_path / "vocab.gguf"
    path.write_bytes(b"private lifetime test")
    monkeypatch.setattr(e, "TOKENIZER_PATH", path)
    monkeypatch.setattr(e, "TOKENIZER_PIN", sha256_file(path))
    owners = []

    class Owner:
        def __init__(self, **kwargs):
            assert kwargs["vocab_only"] is True and kwargs["n_gpu_layers"] == 0
            owners.append(weakref.ref(self))

        def tokenize(self, data, **kwargs):
            return list(data)

    monkeypatch.setattr(llama_cpp, "Llama", Owner)
    count, receipt = e.tokenizer(dict(checks=[]))
    gc.collect()
    assert owners[0]() is not None and count("abc") == 3
    assert receipt["neural_weights_loaded"] is False
