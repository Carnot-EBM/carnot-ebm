"""REQ-REPORT-8973 / SCENARIO-REPORT-8973-V767: fixed development admission."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting import shared_feature_methods_8973 as d
from carnot.reporting import shared_feature_runner_8973 as runner
from carnot.reporting.current_work_receipt import atomic_json

# The immutable corpus and JAX caches outlive individual tests in module fixtures.
pytestmark = pytest.mark.memory_watchdog_skip


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    """REQ-REPORT-8973: original pinned evidence supplies frozen states and roles."""
    value = d.authenticate(d.ROOT, tmp_path_factory.mktemp("8973-inputs"))
    assert not value["failures"]
    return value


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8973: one actual CPU run backs all numerical assertions."""
    return runner.freeze(d.ROOT, tmp_path_factory.mktemp("8973-work"))


def test_protocol():
    """REQ-REPORT-8973: future methods remain fixed after a development null."""
    p = d.protocol()
    assert p["bootstrap_draws"] == 10000 and p["family_alpha"] == 0.05
    assert len(p["detector_bounds"]) == 5
    assert p["seeds"] == [71, 72, 73]
    assert p["binary_energies"] == {"correct": "0", "error": "-z", "p_error": "sigmoid(z)"}
    assert p["development_admission"] == dict(
        auroc_min=0.65,
        prevalence_brier_gain_min=0.005,
        shared_logistic_brier_slack=0.01,
        temperature=1.0,
    )
    assert p["fresh_count"] == 1024 and not p["response_selection"]


def test_fixed_gate():
    """SCENARIO-REPORT-8973-V767: threshold equality passes; no margin is tuned."""
    good = d.admission(
        dict(auroc=0.65, energy_brier=0.20, prevalence_brier=0.21, logistic_brier=0.20)
    )
    assert all(r["passed"] for r in good)
    bad = d.admission(
        dict(auroc=0.649, energy_brier=0.23, prevalence_brier=0.21, logistic_brier=0.20)
    )
    assert not any(r["passed"] for r in bad)


def test_actual_shared_features_and_null(work):
    """REQ-REPORT-8973: failure closes collection and preserves every arm and state."""
    r = work["result"]
    assert len(r["development_guard_rows"]) == 64 * 5
    assert r["development_metrics"]["auroc"] == pytest.approx(0.3502604166666667)
    assert not all(v["passed"] for v in r["development_admission_rows"])
    assert all(v["passed"] for v in r["invariant_control_rows"])
    assert len(r["encoder_hashes"]) == 3
    assert r["numerical_training_calls"] == 3
    assert len(r["graph_manifest"]) == 1024
    assert len({r["graph_hash"] for r in r["graph_manifest"]}) == 1024
    roles = r["role_manifest"]
    for shard in ("A", "B"):
        assert {
            k: sum(v["role"] == k and v["shard"] == shard for v in roles)
            for k in ("calibration", "test", "online")
        } == dict(calibration=64, test=64, online=384)
    streams = {v["stream_id"] for v in roles if v["role"] == "online"}
    assert len(streams) == 24
    assert all(sum(v["stream_id"] == stream for v in roles) == 32 for stream in streams)
    value = d.build(work, dict(receipts=[], coverage={}), True)
    assert value["verdict_class"] == "null" and value["shared_methods_ready_score"] == 0
    assert value["independent_count"] == 1088
    assert len(value["rows"]) == 1088 * 5
    assert not any(value["model_invocation_counts"].values())
    assert value["MODEL_SPECS"] == [] and value["independent_generalization_score"] == 0
    assert set(value) <= set(value["field_principles"])
    assert d.build(work, dict(receipts=[], coverage={}), False)["verdict_class"] == "disqualified"


def test_authentication_refuses_missing_changed_and_gate(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8973-V767: byte and typed gate failures remain distinct."""
    monkeypatch.setattr(d, "INPUT", "upstream.json")
    monkeypatch.setattr(d, "PIN", "sha256:missing")
    p = tmp_path / "upstream.json"
    assert d.authenticate(tmp_path, tmp_path / "raw")["failures"][0]["observed"] == "missing_file"
    atomic_json(p, {})
    assert d.authenticate(tmp_path, tmp_path / "raw")["failures"][0]["observed"] == "changed_bytes"
    for value in (
        {},
        {"diagnosis_ready_score": 0},
        {"diagnosis_ready_score": True},
        {"diagnosis_ready_score": 1, "required_checks_passed": False},
        [],
    ):
        atomic_json(p, value)
        monkeypatch.setattr(d, "PIN", d.reference(p)["sha256"])
        assert d.authenticate(tmp_path, tmp_path / "raw")["failures"]


def test_reduction_replay_and_tamper(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8973-V767: immutable numerical primitives reject a fresh rehash."""
    raw = Path(work["raw"])
    assert d.evaluate(work["evidence"], raw, replay=True) == work["result"]
    path = raw / "reduction.json"
    original = path.read_bytes()
    atomic_json(path, {})
    with pytest.raises(ValueError, match="primitive_drift"):
        d.evaluate(work["evidence"], raw, replay=True)
    path.write_bytes(original)
    guard = work["evidence"]["work"]["evidence"]["views"]["guard"]
    monkeypatch.setitem(guard, "unit_ids", ["altered", *guard["unit_ids"][1:]])
    with pytest.raises(ValueError, match="development_roles"):
        d.evaluate(work["evidence"], tmp_path)


def test_private_cli_and_rehashed_refusal(tmp_path):
    """SCENARIO-REPORT-8973-V767: actual child publication and replay reject rehashed tampering."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["JAX_PLATFORMS"] = "cpu"
    output = tmp_path / (d.NAME + ".json")

    def run(*args):
        result = subprocess.run(
            [sys.executable, "-u", str(d.ROOT / d.CLI), *args],
            cwd=tmp_path,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=300,
        )
        print(result.stdout, flush=True)
        return result

    assert run("--date", "20261110", "--fixture-e2e", str(output)).returncode == 0
    value = json.loads(output.read_text())
    assert value["completed_count"] == 1088 and value["fixture_only"]
    assert value["shared_methods_ready_score"] == 0
    assert run("--cold-replay", str(output)).returncode == 0
    changed = deepcopy(value)
    changed["development_metrics"]["auroc"] += 0.1
    changed["reproducibility_checksum"] = d.checksum(changed)
    atomic_json(output, changed)
    assert run("--cold-replay", str(output)).returncode == 1
    work_path = Path(value["work_reference"]["path"])
    work = json.loads(work_path.read_text())
    work["result"]["development_guard_rows"][0]["p_error"] += 0.1
    atomic_json(work_path, work)
    changed = deepcopy(value)
    changed["work_reference"] = d.reference(work_path)
    changed["raw_shard_hashes"] = [d.reference(Path(r["path"])) for r in value["raw_shard_hashes"]]
    changed["reproducibility_checksum"] = d.checksum(changed)
    atomic_json(output, changed)
    assert run("--cold-replay", str(output)).returncode == 1
    assert run("--date", "wrong").returncode == 2
    assert run("--date", "20261110", "--fixture-e2e", str(d.ROOT / "fixture.json")).returncode == 2


def test_runner_errors_and_blocked_work(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8973-V767: changed code, missing inputs and pinned-work drift fail closed."""
    assert not runner.replay(tmp_path / "missing.json")["passed"]
    raw = Path(work["raw"])
    local = dict(work, evidence=dict(work["evidence"], work=dict(work["evidence"]["work"])))
    validation = dict(receipts=[], coverage={})
    atomic_json(raw / "work.json", local)
    atomic_json(raw / "validation.json", validation)
    local["raw_references"] = [d.reference(raw / "work.json"), d.reference(raw / "validation.json")]
    path = tmp_path / (d.NAME + ".json")
    value = runner.artifact(local, validation, True)
    changed = deepcopy(value)
    changed["code_config_hashes"][0]["sha256"] = "changed"
    atomic_json(path, changed)
    assert runner.replay(path)["reason"] == "retained_bytes_drift"
    changed = deepcopy(value)
    source = tmp_path / "source.py"
    source.write_text("changed")
    changed["code_config_hashes"][0]["source_path"] = str(source)
    atomic_json(path, changed)
    assert runner.replay(path)["reason"] == "executed_bytes_drift"
    local["evidence"]["work"]["result"] = {}
    atomic_json(raw / "work.json", local)
    local["raw_references"] = [d.reference(raw / "work.json"), d.reference(raw / "validation.json")]
    atomic_json(path, runner.artifact(local, validation, True))
    assert runner.replay(path)["reason"] == "pinned_operand_drift"
    monkeypatch.setattr(
        d,
        "authenticate",
        lambda root, raw: dict(
            failures=[dict(observed="missing_file")], references=[], producer={}, work={}
        ),
    )
    blocked = runner.freeze(d.ROOT, tmp_path / "blocked")
    assert blocked["result"] == {}
    value = d.build(blocked, validation, True)
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 1088
    assert len(value["rows"]) == 1088 * 5
    missing = dict(path=str(tmp_path / "operand"), sha256="expected")
    with pytest.raises(ValueError, match="retained_operand_drift"):
        d.load(
            dict(references=[dict(missing, source_path="original")]),
            dict(path="original", sha256="expected"),
        )


def test_production_validation_flow(work, tmp_path, monkeypatch):
    """REQ-REPORT-8973: production runs validators before atomic publication and retains global debt."""
    from carnot.reporting import matched_corpus_runner_8949 as shared
    from carnot.reporting import v756_branch_qualification as qualification
    from carnot.reporting import matched_corpus_a_8949 as admission
    import sys

    local = dict(work)

    def frozen(root, raw):
        local["raw"] = str(raw)
        atomic_json(raw / "work.json", local)
        return local

    def commands(root, plan, *, log_dir, heartbeat_s):
        log_dir.mkdir(parents=True, exist_ok=True)
        return [dict(name=c.name, passed=True, exit_code=0, timed_out=False) for c in plan]

    def plan(private, *, producer):
        (private / ".coverage").write_text("private mocked data")
        return [runner.CommandSpec("owned_unit_child", (sys.executable, "-c", "pass"), "owned", 1)]

    monkeypatch.setattr(runner, "freeze", frozen)
    monkeypatch.setattr(runner, "run_commands", commands)
    monkeypatch.setattr(shared, "validation_plan", plan)
    monkeypatch.setattr(qualification, "measured", lambda *a: {})
    monkeypatch.setattr(admission, "qualified", lambda *a: True)
    monkeypatch.setattr(d, "ROOT", tmp_path)
    assert runner.main(["--date", "20261110"]) == 0
    value = json.loads((tmp_path / "results" / (d.NAME + ".json")).read_text())
    assert value["verdict_class"] == "null" and not value["fresh_collection_permitted"]
    assert value["global_test_debt"]["repeated"] is False
    good = dict(local, result=dict(local["result"]))
    good["result"]["development_admission_rows"] = [
        dict(r, passed=True) for r in local["result"]["development_admission_rows"]
    ]
    good["result"]["invariant_control_rows"] = [
        dict(r) for r in local["result"]["invariant_control_rows"]
    ]
    assert (
        d.build(good, dict(receipts=[dict(name="private-only")], coverage={}), False)[
            "shared_methods_ready_score"
        ]
        == 1
    )
    good["result"]["invariant_control_rows"][0]["passed"] = False
    assert (
        d.build(good, dict(receipts=[dict(name="private-only")], coverage={}), False)[
            "verdict_class"
        ]
        == "disqualified"
    )


def test_retained_operand_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8973-V767: a valid producer cannot bless missing retained bytes."""
    path = tmp_path / "producer.json"
    atomic_json(
        path,
        dict(
            diagnosis_ready_score=1,
            required_checks_passed=True,
            source_artifact_hashes=[
                dict(path=str(tmp_path / "absent.json"), sha256="sha256:expected")
            ],
            code_config_hashes=[],
            raw_shard_hashes=[],
        ),
    )
    monkeypatch.setattr(d, "INPUT", path.name)
    monkeypatch.setattr(d, "PIN", d.reference(path)["sha256"])
    failure = d.authenticate(tmp_path, tmp_path / "raw")["failures"][0]
    assert failure["check"] == "retained_bytes" and failure["observed"] == "missing_file"
    assert failure["path"] == str(tmp_path / "absent.json")


def test_resource_blocked_publication(work, tmp_path, monkeypatch):
    """REQ-REPORT-8973: a missing CPU dependency is terminal blocked with zero current fits."""
    local = dict(work, preconditions=dict(work["preconditions"]), evidence=dict(work["evidence"]))
    local["preconditions"]["resource_failures"] = [
        dict(
            upstream="environment",
            check="dependency",
            path=str(tmp_path),
            sha256=None,
            artifact_field="jax",
            operator="==",
            expected="installed",
            observed="missing_dependency",
        )
    ]
    local["evidence"]["failures"] = local["preconditions"]["resource_failures"]
    local["result"] = {}

    def frozen(root, raw):
        local["raw"] = str(raw)
        atomic_json(raw / "work.json", local)
        return local

    def commands(root, plan, *, log_dir, heartbeat_s):
        return [dict(name=c.name, passed=True) for c in plan]

    monkeypatch.setattr(runner, "freeze", frozen)
    monkeypatch.setattr(runner, "run_commands", commands)
    monkeypatch.setattr(d, "ROOT", tmp_path)
    assert runner.main(["--date", "20261110"]) == 0
    value = json.loads((tmp_path / "results" / (d.NAME + ".json")).read_text())
    assert value["verdict_class"] == "blocked" and value["numerical_training_calls"] == 0


def test_numerical_reduction_refusal(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8973-V767: a fresh numerical disagreement cannot publish."""
    raw = Path(work["raw"])
    local = dict(work)
    validation = dict(receipts=[], coverage={})
    atomic_json(raw / "work.json", local)
    atomic_json(raw / "validation.json", validation)
    local["raw_references"] = [d.reference(raw / "work.json"), d.reference(raw / "validation.json")]
    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, runner.artifact(local, validation, True))
    monkeypatch.setattr(d, "evaluate", lambda *a, **k: {})
    assert runner.replay(path)["reason"] == "reduction_drift"
