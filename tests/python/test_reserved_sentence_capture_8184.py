"""REQ-REPORT-8184 / REQ-VERIFY-8184: private label-blind capture custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import reserved_sentence_capture_8184 as e


def sources():
    """Distinct private source identities prevent fixtures from opening targets."""
    return [
        dict(
            unit_id=f"evaluation{i}",
            source_cluster_id=f"cluster{i}",
            role="evaluation",
            slot=i + 1,
            source_bytes=b"Evidence. More evidence.".hex(),
            answer_bytes=b"First claim. Second claim.".hex(),
            **e.transport.requests(
                dict(
                    source_bytes=b"Evidence. More evidence.".hex(),
                    answer_bytes=b"First claim. Second claim.".hex(),
                ),
                lambda _: 30,
            ),
        )
        for i in range(128)
    ]


def heads():
    """Zero weights make expected probabilities independently calculable."""
    x = np.zeros((2, 16))
    g = e.energy.geometry(x, ["fit-a", "fit-b"])
    return [
        dict(
            arm=a,
            geometry=g if a in e.energy.NEW_ARMS else g["ablation_geometry"],
            weights=[0.0]
            * e.energy.design(
                a, x[:1], g if a in e.energy.NEW_ARMS else g["ablation_geometry"]
            ).shape[1],
            calibration=[0.0, 1.0],
            policy=dict(thresholds=[0.2, 0.6]),
        )
        for a in [*e.energy.NEW_ARMS, *e.energy.CONTROL_ARMS]
    ]


def test_roster_and_label_isolation():
    """SCENARIO-VERIFY-8184-ISOLATION: labels and changed slots are rejected."""
    slots = e.freeze(sources())
    assert len(slots) == 128 and all(s["human_target"] is None for s in slots)
    for key, value in [("slot", 2), ("human_target", 1), ("y", 1)]:
        bad = deepcopy(slots)
        bad[0][key] = value
        with pytest.raises(ValueError):
            e.freeze(bad)
    with pytest.raises(ValueError):
        e.freeze([])


def test_reduce_and_frozen_predictions(tmp_path):
    """REQ-VERIFY-8184: complete sources produce eight unchanged arm rows."""
    slots = e.freeze(sources())
    calls = e.canary.capture(slots, e.fit.FixtureRuntime(), tmp_path, dict(fixture=True))
    baseline = [dict(s, x=[0.0] * 12, status="completed") for s in slots]
    reduced = e.reduce(slots, calls, baseline)
    predicted = e.predict(reduced["feature_rows"], heads())
    assert reduced["completed_count"] == 128
    assert len(predicted) == 128 * 8
    assert all(r["p"] == 0.5 for r in predicted if r["arm"] != "always_escalate")
    assert all(r["action"] == "escalate" for r in predicted)
    assert not any("y" in r for r in predicted)
    incomplete = e.reduce(slots, calls[1:], baseline)
    assert incomplete["failed_count"] == 1
    assert all(r["action"] == "escalate" for r in e.predict(incomplete["feature_rows"], heads()))
    changed = deepcopy(calls)
    changed[0]["request"]["payload"] += "x"
    with pytest.raises(ValueError, match="identity"):
        e.reduce(slots, changed, baseline)
    changed = deepcopy(baseline)
    changed[0]["source_bytes"] = b"Other evidence.".hex()
    with pytest.raises(ValueError, match="identity"):
        e.reduce(slots, calls, changed)
    assert e.reduce(slots, [], [])["completed_count"] == 0


def test_measure_build_replay(tmp_path):
    """SCENARIO-REPORT-8184-CUSTODY: aggregate tamper and bad checks fail."""
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    assert value["evaluation_capture_ready_score"] == 0
    assert value["evaluation_support_score"] == 0
    assert value["label_access_ledger"] == []
    value["completed_count"] -= 1
    atomic_json(path, value)
    assert not e.replay(path)
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"
    blocked = e.measure(tmp_path, tmp_path / "missing")
    assert e.build(blocked, tmp_path / "missing", [dict(passed=True)])["verdict_class"] == "blocked"
    assert not e.replay(tmp_path / "absent")


def test_external_cli(tmp_path):
    """E2E-015/019: script paths and replay need no ambient PYTHONPATH."""
    script = str(e.ROOT / e.CLI)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    output = tmp_path / (e.NAME + ".json")
    for tail, expected in [
        (["--fixture-output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--fixture-output", str(output), "--mutation", "source"], 0),
        (
            [
                "--root",
                str(tmp_path / "missing-input"),
                "--worker-output",
                str(tmp_path / "missing-worker" / "measurement.json"),
            ],
            0,
        ),
        (["--fixture-output", str(e.ROOT / "results" / (e.NAME + ".json"))], 2),
    ]:
        argv = [str(e.ROOT / ".venv/bin/python"), "-u", script, *tail]
        if env.get("COVERAGE_RCFILE"):
            argv[1:2] = ["-m", "coverage", "run"]
        result = subprocess.run(
            argv,
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == expected, result.stdout + result.stderr
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    missing = json.loads((tmp_path / "missing-worker" / "measurement.json").read_text())
    assert any(not c["passed"] for c in missing["checks"])
    assert not missing["calls"]


def input_world(tmp_path, monkeypatch):
    """Private upstream bytes test authentication without opening real labels."""
    from carnot.reporting.current_work_receipt import sha256_file
    from carnot.verify.qwen_development_capture_7995 import Ledger
    from test_fit_evidence_capture_8153 import Runtime

    root = tmp_path / "checkout"
    legacy = []
    slots = sources()
    for s in slots:
        for order, arm in enumerate(("holistic", "source_span")):
            prefix = e.historical.capture.protocol.protocol()["prompt_prefixes"][arm]
            prompt = prefix + json.dumps(
                dict(
                    source=bytes.fromhex(s["source_bytes"]).decode(),
                    answer=bytes.fromhex(s["answer_bytes"]).decode(),
                ),
                ensure_ascii=False,
            )
            legacy.append(
                dict(
                    s,
                    order=order,
                    arm=arm,
                    prompt=prompt,
                    prompt_sha256=e.canonical_hash(prompt),
                    entailment_label=None,
                )
            )
    calls = e.historical.capture.capture(
        e.historical.freeze(legacy),
        Runtime(),
        tmp_path / "legacy",
        "private",
        ledger=Ledger(tmp_path / "ledger.json"),
    )
    assert len(e.baseline(calls)) == 128

    def save(name, value):
        path = tmp_path / name
        atomic_json(path, value)
        return e.reference(path)

    head_ref = save(
        "heads.json",
        dict(
            heads=heads(),
            comparator_id="radial16",
            prediction_code_hashes=[e.reference(e.ROOT / e.MODULE)],
        ),
    )
    public = save(
        "public.json",
        dict(
            roster=[dict(unit_id=s["unit_id"]) for s in slots],
            request_rows=[
                dict(
                    family_id=s["unit_id"],
                    source_bytes=s["source_bytes"],
                    answer_bytes=s["answer_bytes"],
                )
                for s in slots
            ],
        ),
    )
    requests = save("requests.json", dict(rows=slots))
    protocol = save("protocol.json", {})
    call_ref = save("calls.json", dict(rows=calls))
    values = {
        e.UPSTREAM: dict(
            energy_fit_ready_score=1,
            frozen_head_manifest=head_ref,
            comparator_id="radial16",
            trained_head_specs=[],
            code_config_hashes=[e.reference(e.ROOT / e.trained.NUMERIC)],
        ),
        e.fit.canary.UPSTREAM: dict(
            protocol_path=protocol["path"],
            protocol_sha256=protocol["sha256"],
            source_manifest=dict(evaluation=public),
            raw_shard_hashes=[requests],
            tokenizer_receipt={},
        ),
        e.fit.UPSTREAM: dict(qualified_transport_configuration=dict(identity=dict(fixture=True))),
        e.CONTROL: dict(
            raw_shard_hashes=[head_ref, call_ref],
            code_config_hashes={
                "python/carnot/verify/evidence_features_7980.py": e.sha256_file(
                    e.ROOT / "python/carnot/verify/evidence_features_7980.py"
                )
            },
        ),
    }
    for name, value in values.items():
        value.update(
            experiment_id=int(Path(name).name.split("_")[1]),
            required_checks_passed=True,
            flagged_adversarial=False,
        )
        atomic_json(root / name, value)
        monkeypatch.setitem(e.PINS, name, sha256_file(root / name))
    monkeypatch.setattr(
        e.historical.fit, "publication_sidecar", lambda _: tmp_path / "sidecar.json"
    )
    monkeypatch.setattr(e, "read_bound_sidecar", lambda *_: dict(report=dict(passed=True)))
    return root, calls


def test_input_authentication_and_baseline(tmp_path, monkeypatch):
    """REQ-VERIFY-8184: non-JSON prediction code is bound by immutable bytes."""
    root, calls = input_world(tmp_path, monkeypatch)
    plan = e.inputs(root, tmp_path / "raw")
    assert all(c["passed"] for c in plan["checks"]), plan["checks"]
    assert len(plan["slots"]) == 128 and len(plan["heads"]) == 6
    monkeypatch.setattr(e.historical.fit.lexical, "extract", lambda _: dict(values=None))
    assert all(r["x"] is None for r in e.baseline(calls))
    (root / e.UPSTREAM).write_text("{}")
    assert not all(c["passed"] for c in e.inputs(root, tmp_path / "tamper")["checks"])

    def malformed(*_):
        raise KeyError("malformed_private_manifest")

    monkeypatch.setattr(e.trained, "bind", malformed)
    assert e.inputs(root, tmp_path / "structure")["checks"][-1]["check"] == "input_structure"


def test_live_adapter_without_model_or_labels(tmp_path, monkeypatch):
    """REQ-VERIFY-8184: scripted live dispatch certifies plumbing only."""
    root, _ = input_world(tmp_path, monkeypatch)
    plan = e.inputs(root, tmp_path / "plan")
    monkeypatch.setattr(e, "inputs", lambda *_: deepcopy(plan))

    def preflight(p, _):
        p.update(identity=dict(fixture=True))
        e.canary.progress("after_CPU_grammar_preflight", 128, -92)

    monkeypatch.setattr(e.canary, "preflight", preflight)

    def live(p, raw):
        return dict(
            rows=e.canary.capture(p["slots"], e.fit.FixtureRuntime(), raw / "slots", p["identity"]),
            checks=[],
            model_loads_attempted=1,
            model_loads_completed=1,
            runtime_receipts=[],
        )

    monkeypatch.setattr(e.canary, "live", live)
    work = e.measure(root, tmp_path / "live")
    assert len(work["calls"]) == 128
    work["duration_s"] = 11
    atomic_json(tmp_path / "live" / "measurement.json", work)
    value = e.build(work, tmp_path / "live", [dict(passed=True)])
    assert value["evaluation_capture_ready_score"] == value["evaluation_support_score"] == 1
    plan["qualified_identity"] = dict(changed=True)
    assert not e.measure(root, tmp_path / "identity-drift")["live_result"]["rows"]


def test_replay_primitive_and_log_tamper(tmp_path):
    """SCENARIO-REPORT-8184-CUSTODY: rehashed primitives still need parity."""
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    log = tmp_path / "log.txt"
    log.write_text("completed")
    receipt = dict(passed=True, log_path=str(log), log_sha256=e.sha256_file(log))
    value = e.build(work, tmp_path / "raw", [receipt])
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    log.write_text("tampered")
    assert not e.replay(candidate)
    log.write_text("completed")
    for index in range(4):
        ref = work["raw_shard_hashes"][index]
        path = Path(ref["path"])
        original = path.read_bytes()
        path.write_text("{}")
        assert not e.replay(candidate)
        path.write_bytes(original)


def test_supervisor_reports_actual_pending(tmp_path, monkeypatch, capsys):
    """REQ-REPORT-8184: child waits report actual completed/pending counts."""
    from carnot.reporting import experiment_7303_validation_scope as scope

    def child(*args, **kwargs):
        for event in ("before_subprocess", "subprocess_outstanding", "after_subprocess"):
            scope._progress("private", event, 0)
        return dict(passed=True)

    monkeypatch.setattr(e.canary, "supervise", child)
    assert e.supervise(tmp_path, {}, tmp_path, tmp_path)["passed"]
    out = capsys.readouterr().out
    assert "completed=0 pending=1" in out and "completed=1 pending=0" in out


@pytest.mark.parametrize("mutation", ["calls", "requests", "features"])
def test_rehashed_replay_drift(tmp_path, mutation):
    """SCENARIO-REPORT-8184-CUSTODY: new hashes cannot bless changed evidence."""
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    if mutation == "calls":
        work["calls"] = []
    elif mutation == "requests":
        work["slots"] = deepcopy(work["slots"])
        work["slots"][0]["requests"][0]["payload"] += "tamper"
    else:
        path = Path(work["raw_shard_hashes"][1]["path"])
        atomic_json(path, {})
        work["raw_shard_hashes"][1] = e.reference(path)
        value["raw_shard_hashes"] = deepcopy(work["raw_shard_hashes"])
    path = tmp_path / "raw" / "measurement.json"
    atomic_json(path, work)
    value["measurement_reference"] = e.reference(path)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert not e.replay(candidate)
