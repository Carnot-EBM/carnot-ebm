"""REQ-REPORT-8154: source custody, private CLI success/block/tamper/replay."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.verify import evidence_energy_fit_8154 as e
from carnot.reporting.current_work_receipt import atomic_json


def cli(tmp_path, *args):
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private subprocess", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print("after private subprocess", result.returncode, flush=True)
    return result


def test_private_success_block_and_tamper(tmp_path):
    """SCENARIO-REPORT-8154: actual external script requires no PYTHONPATH."""
    output = tmp_path / "experiment_8154_fixture.json"
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["energy_fit_ready_score"] == 1
    assert value["MODEL_SPECS"] == [] and value["verdict_class"] == "circular_positive"
    assert value["model_invocation_counts"]["model_loads_attempted"] == 0
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    saved = deepcopy(value)
    value["development_metrics"][0]["typed_cost"] = 77
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, saved)
    weights = Path(value["frozen_head_manifest"]["path"])
    original = weights.read_bytes()
    weights.write_text("{}")
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    weights.write_bytes(original)
    block = tmp_path / "blocked" / "experiment_8154_block.json"
    assert cli(tmp_path, "--fixture-output", block, "--mutation", "block").returncode == 0
    blocked = json.loads(block.read_text())
    assert blocked["honest_verdict"] == "complete_blocked_fit_trainable_score"
    assert blocked["gate_check_summary"][0]["observed"] == 0
    assert blocked["energy_fit_ready_score"] == 0 and e.replay(block)
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/private.json").returncode == 2
    assert cli(tmp_path, "--date", "19990101").returncode == 2
    assert cli(tmp_path, "--mutation", "block").returncode == 2


def test_external_block_and_owned_failure(tmp_path):
    """REQ-REPORT-8154: absent operands block; failed owned checks disqualify."""
    work = e.measure(tmp_path, tmp_path / "raw")
    assert (
        next(r for r in work["plan"]["checks"] if r["check"] == "upstream_exists")["passed"]
        is False
    )
    good = [dict(passed=True)]
    value = e.build(work, tmp_path / "raw", good)
    assert value["verdict_class"] == "blocked" and value["energy_fit_ready_score"] == 0
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"
    specs = e.manifest(tmp_path, tmp_path / "candidate.json")
    assert any(r["name"] == "coverage_report" for r in specs["commands"])
    assert specs["repository_health"]["argv"][-2:] == ["tests/python", "-q"]
    assert e.main(["--cold-replay", str(tmp_path / "missing.json")]) == 1


def test_matched_primitive_features_and_gate_binding(tmp_path, monkeypatch):
    """REQ-VERIFY-8154: targets join by source identity, never quote correctness."""
    from carnot.verify import fit_evidence_capture_8153 as capture
    from carnot.reporting.current_work_receipt import sha256_file
    from test_fit_evidence_capture_8153 import public, Runtime
    from carnot.verify.qwen_development_capture_7995 import Ledger

    slots = capture.freeze(public())
    calls = capture.capture(
        slots, Runtime(), tmp_path / "calls", "fixture", ledger=Ledger(tmp_path / "ledger.json")
    )
    ref = tmp_path / "primitive_calls.json"
    atomic_json(ref, dict(rows=calls))
    target = tmp_path / "fit_tune_targets.json"
    labels = {r["unit_id"]: i % 2 for i, r in enumerate(capture.reduce(calls)["rows"])}
    atomic_json(target, labels)
    refs = [e.reference(ref), e.reference(target)]
    for role in ("fit", "tune"):
        p = tmp_path / "public" / (role + ".json")
        atomic_json(
            p,
            dict(
                request_rows=[
                    dict(
                        family_id=r["unit_id"],
                        source_bytes=r["source_bytes"],
                        answer_bytes=r["answer_bytes"],
                    )
                    for r in slots
                    if r["role"] == role and r["arm"] == "holistic" and r["condition"] == "original"
                ]
            ),
        )
        refs.append(e.reference(p))
    v = dict(
        fit_capture_ready_score=1,
        fit_trainable_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        rows=capture.reduce(calls)["rows"],
        raw_shard_hashes=refs[:2],
        source_artifact_hashes=refs[2:],
        MODEL_SPECS=["historical"],
        model_invocation_counts={},
        terminal_validation_sidecar_path="unused",
    )
    p = tmp_path / e.UPSTREAM
    atomic_json(p, v)
    monkeypatch.setattr(e, "PIN", sha256_file(p))
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *a: dict(report=dict(passed=True)))
    monkeypatch.setattr(e, "publication_sidecar", lambda *a: tmp_path / "unused")
    plan = e.inputs(tmp_path, tmp_path / "raw")
    assert all(r["passed"] for r in plan["checks"])
    assert len(plan["rows"]) == 192 and len(plan["rows"][0]["x"]) == 12
    assert plan["rows"][0]["y"] == 0 and plan["rows"][0]["x"][10] == 0
    assert plan["historical_model_provenance"]["MODEL_SPECS"] == ["historical"]
    monkeypatch.setattr(
        e.lexical, "extract", lambda row: dict(values=None, abstention="private_abstention")
    )
    excluded = e.inputs(tmp_path, tmp_path / "excluded")
    assert all(r["status"] == "excluded" for r in excluded["rows"])
    ref.write_text("{}")
    assert not all(r["passed"] for r in e.inputs(tmp_path, tmp_path / "tamper")["checks"])


def test_live_orchestration_and_owned_exception(tmp_path, monkeypatch):
    """REQ-REPORT-8154: live orchestration records normal owned exits without rewriting history."""
    original = e.measure
    monkeypatch.setattr(e, "measure", lambda root, raw, **kwargs: original(root, raw, fixture=True))

    def checked(root, spec, *a, **kw):
        if spec["name"] == "coverage_json":
            Path(spec["argv"][-1]).write_text("{}")
        return dict(passed=True, name=spec["name"])

    monkeypatch.setattr(e.execution, "run_check", checked)
    captured = []
    monkeypatch.setattr(e.execution, "publish", lambda value, *a: captured.append(value))
    output = tmp_path / "experiment_8154_test.json"
    output.write_text('{"old":true}')
    assert e.main(["--output", str(output), "--root", str(tmp_path)]) == 0
    assert captured[0]["repository_health"]["passed"]
    assert captured[0]["energy_fit_ready_score"] == 1

    def failing(*a):
        raise ValueError("owned_fit_failure")

    monkeypatch.setattr(e.energy, "train", failing)
    failed = original(tmp_path, tmp_path / "failure", fixture=True)
    assert failed["owned_failure"] == "owned_fit_failure"
    assert (
        e.build(failed, tmp_path / "failure", [dict(passed=True)])["verdict_class"]
        == "disqualified"
    )


def test_replay_log_and_measurement_tamper(tmp_path):
    """REQ-REPORT-8154: successful byte custody rejects independent log and work mutations."""
    raw = tmp_path / "raw"
    work = e.measure(tmp_path, raw, fixture=True)
    value = e.build(work, raw, [dict(passed=True)], fixture=True)
    path = tmp_path / "experiment_8154_test.json"
    atomic_json(path, value)
    assert e.replay(path)
    log = tmp_path / "log.txt"
    log.write_text("changed")
    bad = deepcopy(value)
    bad["validation_receipts"][0].update(log_path=str(log), log_sha256="wrong")
    atomic_json(path, bad)
    assert not e.replay(path)
    work["fitted"]["failures"].append(dict(error="changed"))
    atomic_json(raw / "measurement.json", work)
    value["measurement_reference"] = e.reference(raw / "measurement.json")
    atomic_json(path, value)
    assert not e.replay(path)


def test_publication_and_malformed_external_input(tmp_path, monkeypatch):
    """REQ-REPORT-8154: malformed upstream bytes and sidecars cannot authorize fitting."""
    terminal = tmp_path / "terminal.json"
    atomic_json(terminal, dict(publication=dict(sidecar_path=str(tmp_path / "bound.json"))))
    assert (
        e.publication_sidecar(dict(terminal_validation_sidecar_path=str(terminal)))
        == tmp_path / "bound.json"
    )
    upstream = tmp_path / e.UPSTREAM
    upstream.parent.mkdir(parents=True)
    upstream.write_text("{")
    monkeypatch.setattr(e, "PIN", e.sha256_file(upstream))
    plan = e.inputs(tmp_path, tmp_path / "invalid")
    assert plan["checks"][-1]["check"] == "input_structure" and not plan["checks"][-1]["passed"]
