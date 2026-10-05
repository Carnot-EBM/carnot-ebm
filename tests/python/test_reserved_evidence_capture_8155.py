"""REQ-REPORT-8155 / REQ-VERIFY-8155: private unlabeled capture boundaries."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.verify import reserved_evidence_capture_8155 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify.qwen_development_capture_7995 import Ledger
from test_fit_evidence_capture_8153 import Runtime


def public():
    rows = []
    for i in range(128):
        for order, arm in enumerate(("holistic", "source_span")):
            source = f"é fact evaluation {i}"
            prompt = e.capture.protocol.protocol()["prompt_prefixes"][arm] + json.dumps(
                dict(source=source, answer="fact"), ensure_ascii=False
            )
            rows.append(
                dict(
                    unit_id=f"evaluation-{i}",
                    source_cluster_id=f"cluster-{i}",
                    role="evaluation",
                    arm=arm,
                    order=order,
                    source_bytes=source.encode().hex(),
                    answer_bytes=b"fact".hex(),
                    prompt=prompt,
                    prompt_sha256=canonical_hash(prompt),
                    entailment_label=None,
                )
            )
    return rows


def cli(tmp_path, *args):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private subprocess", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print("after private subprocess", result.returncode, flush=True)
    return result


def test_freeze_and_forced_failures():
    """SCENARIO-VERIFY-8155: no target, role, prompt or source substitutions."""
    rows = public()
    assert len(e.freeze(rows)) == 256
    swapped = deepcopy(rows)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    swapped[0]["order"], swapped[1]["order"] = 0, 1
    assert [r["arm"] for r in e.freeze(swapped)[:2]] == ["source_span", "holistic"]
    assert all(r["human_target"] is None for r in e.freeze(rows))
    for key, value in [
        ("prompt", "changed"),
        ("prompt_sha256", "wrong"),
        ("source_cluster_id", "cluster-1"),
        ("unit_id", "evaluation-1"),
        ("order", 7),
        ("entailment_label", 1),
    ]:
        bad = deepcopy(rows)
        bad[0][key] = value
        with pytest.raises(ValueError):
            e.freeze(bad)
    with pytest.raises(ValueError):
        e.freeze(rows[:-1])


def test_capture_and_prediction_boundaries(tmp_path):
    """REQ-VERIFY-8155: bad quotes preserve probabilities; absent calls stay missing."""
    slots = e.freeze(public())
    calls = e.capture.capture(
        slots, Runtime(), tmp_path / "slots", "fixture", ledger=Ledger(tmp_path / "ledger.json")
    )
    reduced = e.pairs(calls)
    assert len(reduced) == 128 and all(r["status"] == "completed" for r in reduced)
    heads = e.fixture_heads()
    predictions = e.predict(calls, heads)
    assert len(predictions) == 1024 and all(r["human_target"] is None for r in predictions)
    assert all(r["action"] in ("accept", "reject", "escalate") for r in predictions)
    assert len(e.pairs([])) == 128
    calls[0]["probability"] = 0.8
    with pytest.raises(ValueError):
        e.predict(calls, heads)


def test_private_cli_success_block_tamper(tmp_path):
    """SCENARIO-REPORT-8155: real external CLI seals and independently replays."""
    output = tmp_path / "experiment_8155_fixture.json"
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["evaluation_capture_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    saved = deepcopy(value)
    value["rows"][0]["numerator"] = 5
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, saved)
    head = Path(value["frozen_head_manifest"]["path"])
    head.write_text("{}")
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    for mutation in ("block", "labels"):
        block = tmp_path / mutation / "experiment_8155_fixture.json"
        assert cli(tmp_path, "--fixture-output", block, "--mutation", mutation).returncode == 0
        blocked = json.loads(block.read_text())
        assert (
            blocked["verdict_class"] == "blocked" and blocked["evaluation_capture_ready_score"] == 0
        )
        assert e.replay(block)
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/private.json").returncode == 2
    assert cli(tmp_path, "--mutation", "labels").returncode == 2
    assert cli(tmp_path, "--date", "19990101").returncode == 2
    assert e.main(["--cold-replay", str(tmp_path / "missing.json")]) == 1


def world(tmp_path, monkeypatch):
    """Private upstream byte manifests exercise custody without production mutations."""
    root = tmp_path / "root"
    data = tmp_path / "upstream"
    rows = public()
    manifests = {}
    for role in ("fit", "tune", "evaluation"):
        chosen = rows[::2] if role == "evaluation" else rows[:2:2]
        roster = [
            dict(
                unit_id=r["unit_id"] if role == "evaluation" else role,
                source_cluster_id=r["source_cluster_id"] if role == "evaluation" else role,
            )
            for r in chosen
        ]
        requests = [
            dict(
                family_id=r["unit_id"],
                source_bytes=r["source_bytes"],
                answer_bytes=r["answer_bytes"],
            )
            for r in chosen
        ]
        path = data / (role + ".json")
        atomic_json(path, dict(roster=roster, request_rows=requests))
        manifests[role] = e.reference(path)
    path = data / "capture.json"
    atomic_json(path, dict(rows=rows))
    method = data / "method.json"
    atomic_json(method, {})
    runtime = dict(model_path="model", gguf_sha256="gguf", model_revision="revision")
    source = dict(
        source_protocol_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        expected_runtime_identity=runtime,
        source_role_manifests=manifests,
        source_role_masks=dict(evaluation=list(range(128))),
        method_freeze=e.reference(method),
        pinned_method_paths=[],
        capture_manifest=e.reference(path),
    )
    heads = e.fixture_heads()
    for head in heads:
        head["tune_source_ids"] = ["tune"]
    path = data / "heads.json"
    atomic_json(
        path, dict(heads=heads, decision_rule=e.fit.energy.CONFIG["costs"], tie_rule="escalate")
    )
    trained = dict(
        energy_fit_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        trained_head_specs=[],
        frozen_head_manifest=e.reference(path),
    )
    historical = dict(
        fit_capture_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
        model_receipt=dict(runtime, resolved_library=dict(libraries=[])),
    )
    for name, value in [
        (e.UPSTREAM, trained),
        (e.capture.UPSTREAM, source),
        (e.fit.UPSTREAM, historical),
    ]:
        atomic_json(root / name, value)
    monkeypatch.setattr(e, "PIN", e.reference(root / e.UPSTREAM)["sha256"])
    monkeypatch.setattr(e.capture, "PIN", e.reference(root / e.capture.UPSTREAM)["sha256"])
    monkeypatch.setattr(e.fit, "PIN", e.reference(root / e.fit.UPSTREAM)["sha256"])
    monkeypatch.setattr(e.fit, "publication_sidecar", lambda v: data / "unused")
    monkeypatch.setattr(e.fit, "read_bound_sidecar", lambda *a: dict(report=dict(passed=True)))
    return root, source, trained, historical


def test_input_custody_and_role_overlap(tmp_path, monkeypatch):
    """REQ-VERIFY-8155: authenticate heads, original roles and matched source bytes."""
    root, source, trained, historical = world(tmp_path, monkeypatch)
    plan = e.inputs(root, tmp_path / "raw")
    assert all(r["passed"] for r in plan["checks"]) and len(plan["slots"]) == 256
    fitpath = Path(source["source_role_manifests"]["fit"]["path"])
    fitvalue = json.loads(fitpath.read_text())
    fitvalue["roster"][0]["source_cluster_id"] = "cluster-0"
    atomic_json(fitpath, fitvalue)
    source["source_role_manifests"]["fit"] = e.reference(fitpath)
    atomic_json(root / e.capture.UPSTREAM, source)
    monkeypatch.setattr(e.capture, "PIN", e.reference(root / e.capture.UPSTREAM)["sha256"])
    failed = e.inputs(root, tmp_path / "overlap")
    assert next(r for r in failed["checks"] if not r["passed"])["check"] == "role_overlap"
    monkeypatch.setattr(e, "PIN", "wrong")
    assert not all(r["passed"] for r in e.inputs(root, tmp_path / "head-tamper")["checks"])
    missing = e.inputs(tmp_path / "absent", tmp_path / "missing")
    assert missing["checks"][0]["observed"] is False
    monkeypatch.setattr(e, "PIN", e.reference(root / e.UPSTREAM)["sha256"])
    monkeypatch.setattr(
        e.fit, "publication_sidecar", lambda v: (_ for _ in ()).throw(KeyError("sidecar"))
    )
    assert e.inputs(root, tmp_path / "sidecar")["checks"][-1]["check"] == "authenticated_inputs"


def test_missing_features_and_owned_validation(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8155: missing public features never become usable predictions."""
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    monkeypatch.setattr(
        e.fit.lexical, "extract", lambda row: dict(values=None, abstention="empty_source")
    )
    rows = e.predict(work["result"]["rows"], work["plan"]["heads"])
    assert all(r["status"] == "excluded" and r["p"] is None for r in rows)
    value = e.build(work, tmp_path / "raw", [dict(passed=False)], fixture=True)
    assert value["verdict_class"] == "disqualified" and value["evaluation_capture_ready_score"] == 0


def test_live_adapter_and_load_only_accounting(tmp_path, monkeypatch):
    """REQ-VERIFY-8155: owned ledger and fit-library drift control comparability."""
    base = e.measure(tmp_path, tmp_path / "base", fixture=True)
    plan = deepcopy(base["plan"])
    library = tmp_path / "library"
    library.write_text("frozen CUDA build")
    plan["fit_runtime_libraries"] = [e.reference(library)]
    monkeypatch.setattr(e, "inputs", lambda *a: deepcopy(plan))
    monkeypatch.setattr(e.capture, "runtime_preflight", lambda *a: None)
    result = deepcopy(base["result"])
    ledger = Ledger(tmp_path / "live-ledger.json")
    ledger.start("model_load", "owned-load", {})
    ledger.finish("owned-load", "completed", {})
    for row in result["rows"]:
        ledger.start("generation", row["call_id"], row["request"])
        ledger.finish(row["call_id"], "completed", row["raw_response"])
    result.update(
        ledger=ledger.rows,
        model_identity_receipt=dict(authenticated=True),
        resolved_library=dict(libraries=[]),
        gpu_lease_receipt=dict(owned=True),
    )
    monkeypatch.setattr(e.capture, "live", lambda *a: deepcopy(result))
    work = e.measure(tmp_path, tmp_path / "live")
    assert work["result"]["rows"] == result["rows"]
    work["duration_s"] = 11.0
    value = e.build(work, tmp_path / "live", [dict(passed=True)])
    assert value["evaluation_capture_ready_score"] == 1
    assert (
        value["inference_mode"] == "live_gpu"
        and value["model_invocation_counts"]["generation_calls_completed"] == 256
    )
    owned_failure = deepcopy(work)
    owned_failure["result"]["checks"] = [
        dict(check="owned_runtime_authenticated_capture", passed=False)
    ]
    assert (
        e.build(owned_failure, tmp_path / "live", [dict(passed=True)])["verdict_class"]
        == "disqualified"
    )
    work["result"]["ledger"][-1]["response_sha256"] = "wrong"
    with pytest.raises(ValueError, match="ledger_binding"):
        e.build(work, tmp_path / "live", [dict(passed=True)])
    load_only = deepcopy(result)
    load_only.update(rows=[], ledger=ledger.rows[:1])
    monkeypatch.setattr(
        e.capture,
        "live",
        lambda *a: dict(load_only, checks=[dict(check="load_only", passed=False)]),
    )
    work = e.measure(tmp_path, tmp_path / "load-only")
    value = e.build(work, tmp_path / "load-only", [dict(passed=True)])
    assert value["inference_substrate_class"] == "model_load_no_generation"
    assert value["completed_count"] == 0 and value["verdict_class"] == "blocked"
    library.write_text("changed")
    work = e.measure(tmp_path, tmp_path / "drift")
    assert (
        next(r for r in work["plan"]["checks"] if not r["passed"])["check"]
        == "fit_runtime_library_sha256"
    )
    assert all(not r["started"] for r in work["result"]["rows"])
    missing = e.measure(tmp_path / "missing", tmp_path / "missing-raw")
    assert len(missing["result"]["rows"]) == 256


def test_main_owned_checks_and_preserved_history(tmp_path, monkeypatch):
    """REQ-REPORT-8155: frozen argv, normal receipts and preserved prior primaries."""
    original = e.measure
    monkeypatch.setattr(e, "measure", lambda root, raw, **kwargs: original(root, raw, fixture=True))

    def checked(root, spec, *a, **kw):
        if spec["name"] == "coverage_json":
            Path(spec["argv"][-1]).write_text("{}")
        return dict(passed=True, name=spec["name"])

    monkeypatch.setattr(e.execution, "run_check", checked)
    values = []
    monkeypatch.setattr(e.execution, "publish", lambda value, *a: values.append(value))
    output = tmp_path / "experiment_8155_test.json"
    output.write_text('{"old":true}')
    assert e.main(["--output", str(output)]) == 0
    assert values[0]["required_checks_passed"] and values[0]["repository_health"]["passed"]
    raw = Path(values[0]["terminal_validation_sidecar_path"]).parent
    assert json.loads((raw / "preserved_historical_primary.json").read_text()) == dict(old=True)
    assert any(
        Path(r["path"]).name == "changed_code_coverage.json" for r in values[0]["raw_shard_hashes"]
    )


def test_replay_code_log_and_rehashed_tamper(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8155: rehashed tampering still fails independent reconstruction."""
    raw = tmp_path / "raw"
    work = e.measure(tmp_path, raw, fixture=True)
    log = tmp_path / "validation.log"
    log.write_text("normal exit")
    receipts = [dict(passed=True, log_path=str(log), log_sha256=e.reference(log)["sha256"])]
    output = tmp_path / "experiment_8155_fixture.json"
    atomic_json(output, e.build(work, raw, receipts, fixture=True))
    assert e.replay(output)
    real_hash = e.sha256_file
    monkeypatch.setattr(
        e, "sha256_file", lambda p: "wrong" if p == e.ROOT / e.MODULE else real_hash(p)
    )
    assert not e.replay(output)
    monkeypatch.setattr(e, "sha256_file", real_hash)
    log.write_text("changed")
    assert not e.replay(output)
    log.write_text("normal exit")
    predictions = Path(work["raw_shard_hashes"][2]["path"])
    changed = json.loads(predictions.read_text())
    changed["rows"][0]["p"] = 0.9
    atomic_json(predictions, changed)
    work["raw_shard_hashes"][2] = e.reference(predictions)
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, e.build(work, raw, receipts, fixture=True))
    assert not e.replay(output)


def test_validation_pending_counts(tmp_path, monkeypatch):
    """REQ-REPORT-8155: an unfinished child reports a real pending count."""

    class Event:
        def __init__(self):
            self.calls = 0

        def wait(self, timeout):
            self.calls += 1
            return self.calls > 1

        def set(self):
            pass

    class Thread:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout):
            pass

    monkeypatch.setattr(e.threading, "Event", Event)
    monkeypatch.setattr(e.threading, "Thread", Thread)
    monkeypatch.setattr(e.execution, "run_check", lambda *a, **kw: dict(passed=True))
    assert e.checked(dict(name="private"), tmp_path, tmp_path)["passed"]


def test_replay_rejects_rehashed_slot_substitution(tmp_path):
    """SCENARIO-REPORT-8155: a new checksum cannot replace an original source slot."""
    raw = tmp_path / "raw"
    work = e.measure(tmp_path, raw, fixture=True)
    for row in work["result"]["rows"][:2]:
        row["unit_id"] = "substituted"
    calls = raw / "primitive_calls.json"
    atomic_json(calls, dict(rows=work["result"]["rows"]))
    work["raw_shard_hashes"][1] = e.reference(calls)
    predictions = raw / "sealed_predictions.json"
    atomic_json(predictions, dict(rows=e.predict(work["result"]["rows"], work["plan"]["heads"])))
    work["raw_shard_hashes"][2] = e.reference(predictions)
    manifest = raw / "sealed_prediction_manifest.json"
    seal = json.loads(manifest.read_text())
    seal.update(calls=work["raw_shard_hashes"][1], predictions=work["raw_shard_hashes"][2])
    atomic_json(manifest, seal)
    work["raw_shard_hashes"][3] = e.reference(manifest)
    atomic_json(raw / "measurement.json", work)
    output = tmp_path / "experiment_8155_fixture.json"
    atomic_json(output, e.build(work, raw, [dict(passed=True)], fixture=True))
    assert not e.replay(output)
