"""REQ-VERIFY-8347 / REQ-REPORT-8347: historical custody cannot relax live policy."""

import json
from pathlib import Path

import pytest

from carnot.reporting.current_work_receipt import sha256_file
from carnot.reporting import local_consumer_qualification_8347 as e


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """SCENARIO-VERIFY-8347-REPLAY: reuse real byte-bound constructed primitives."""
    raw = tmp_path_factory.mktemp("8347-primitives")
    return e.measure(e.ROOT, raw), raw


def test_complete_private_closure_and_absent_operand(tmp_path):
    """SCENARIO-VERIFY-8347-CLOSURE: preserve old policy bytes and reject missingness."""
    closure = e.freeze_historical(tmp_path / "closure")
    refs = json.loads(closure.read_text())["operands"]
    assert len(refs) == 133
    binder = e.HistoricalCustody(tmp_path / "custody", closure)
    policy = e.ROOT / "ops/exclusion_manifest.yaml"
    ref = binder.bind(policy, e.HISTORICAL_POLICY_HASH)
    assert sha256_file(Path(ref["snapshot_path"])) == e.HISTORICAL_POLICY_HASH
    assert sha256_file(policy) != e.HISTORICAL_POLICY_HASH
    with pytest.raises(ValueError):
        binder.bind(tmp_path / "unknown")
    Path(ref["snapshot_path"]).unlink()
    with pytest.raises(ValueError):
        binder.bind(policy)
    with e.historical_operands(closure):
        blocked = e.consumer.measure(e.ROOT, tmp_path / "missing-historical", fixture=True)
    result = e.consumer.build(
        blocked, tmp_path / "missing-historical", [{"passed": True}], fixture=True
    )
    assert result["verdict_class"] == "blocked"
    assert result["calibrated_memory_ready_score"] == result["stream_input_ready_score"] == 0


def test_current_policy_is_separate(tmp_path):
    """SCENARIO-VERIFY-8347-CLOSURE: a historical pass cannot bypass retirement."""
    root = tmp_path
    policy = root / "ops/exclusion_manifest.yaml"
    policy.parent.mkdir()
    policy.write_text("retired_experiments: []\n")
    assert e.current_policy(root)["passed"]
    policy.write_text("retired_experiments:\n  - experiment_id: 8347\n")
    assert not e.current_policy(root)["passed"]


def test_rehashed_closure_cannot_change_historical_authority(tmp_path):
    """SCENARIO-VERIFY-8347-CLOSURE: private manifest rehashing cannot invent old inputs."""
    from carnot.reporting.current_work_receipt import atomic_json

    closure = e.freeze_historical(tmp_path / "closure")
    value = json.loads(closure.read_bytes())
    value["operands"][0]["sha256"] = "sha256:invented"
    atomic_json(closure, value)
    with pytest.raises(ValueError, match="historical_operand_closure"):
        e.HistoricalCustody(tmp_path / "custody", closure)


def test_reduction_and_rehashed_tamper(measured, tmp_path):
    """SCENARIO-VERIFY-8347-REPLAY: causal evidence rejects rehashed claims."""
    from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

    work, raw = measured
    value = e.build(work, raw, [{"name": "unit_control", "passed": True}])
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    assert value["local_kernel_ready_score"] == 0
    assert value["verdict_class"] == "disqualified"
    for field in ["local_kernel_ready_score", "dense_sparse_error_max", "completed_count"]:
        bad = dict(value)
        bad[field] = 99
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path)
    atomic_json(path, {})
    assert not e.replay(path)
    assert not e.replay(tmp_path / "absent")


def test_absent_external_input_is_blocked(tmp_path):
    """SCENARIO-VERIFY-8347-CLOSURE: external missingness stays blocked and zero."""
    work = e.measure(tmp_path, tmp_path / "raw")
    value = e.build(work, tmp_path / "raw", [{"passed": True}])
    assert value["verdict_class"] == "blocked"
    assert value["local_kernel_ready_score"] == 0
    assert value["gate_check_summary"][-1]["passed"] is False


def test_manifest_and_real_cli(measured, tmp_path):
    """SCENARIO-REPORT-8347-CLI: real bounded children cover direct entry and rejection."""
    from carnot.reporting import local_consumer_execution_8347 as runner
    from carnot.reporting.current_work_receipt import atomic_json

    plan = runner.manifest(tmp_path, tmp_path / "candidate.json")
    assert "repository_health" not in plan
    assert "--files" in plan["commands"][-1]["argv"]
    assert any(r["name"] == "first_three_consumers" for r in plan["commands"])
    work, raw = measured
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, e.build(work, raw, [{"name": "unit_control", "passed": True}]))
    for name, args, expected in [
        ("valid", ["--cold-replay", str(path)], 0),
        ("absent", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("date", ["--date", "20261008"], 2),
    ]:
        receipt = runner.check(
            dict(name=name, argv=runner.cli() + args, deadline_s=60, expected_exit=expected),
            tmp_path / "logs",
        )
        assert receipt["passed"]
    assert runner.main(["--cold-replay", str(path)]) == 0


def test_private_cli_publication_and_guard(tmp_path):
    """SCENARIO-REPORT-8347-CLI: missing real operands produce a private blocked primary."""
    from carnot.reporting import local_consumer_execution_8347 as runner

    path = tmp_path / (e.NAME + ".json")
    assert runner.main(["--root", str(tmp_path), "--fixture-output", str(path)]) == 0
    value = json.loads(path.read_bytes())
    assert value["verdict_class"] == "blocked" and value["local_kernel_ready_score"] == 0
    with pytest.raises(SystemExit):
        runner.main(["--fixture-output", str(e.ROOT / "results" / path.name)])


def test_historical_authentication_rejects_changed_bytes(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8347-CLOSURE: deliberate bad bytes or failed validation block."""
    real_hash = e.sha256_file
    primary = e.ROOT / "results" / (e.consumer.NAME + ".json")
    monkeypatch.setattr(
        e, "sha256_file", lambda path: "sha256:wrong" if path == primary else real_hash(path)
    )
    with pytest.raises(ValueError, match="historical_primary_sha256"):
        e.freeze_historical(tmp_path / "bad-primary")
    monkeypatch.undo()
    real_read = e.read_bound_sidecar

    def failed_report(primary, sidecar):
        report = real_read(primary, sidecar)
        return dict(report, report={"passed": False})

    monkeypatch.setattr(e, "read_bound_sidecar", failed_report)
    with pytest.raises(ValueError, match="historical_terminal"):
        e.freeze_historical(tmp_path / "bad-terminal")
    monkeypatch.undo()
    snapshot = Path(json.loads(primary.read_bytes())["source_artifact_hashes"][0]["snapshot_path"])
    monkeypatch.setattr(
        e, "sha256_file", lambda path: "sha256:wrong" if path == snapshot else real_hash(path)
    )
    with pytest.raises(ValueError, match="historical_operand_sha256"):
        e.freeze_historical(tmp_path / "bad-operand")


def test_resource_and_authority_rejection(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8347-CLOSURE: unavailable resources and failed authorities block."""
    from types import SimpleNamespace

    monkeypatch.setattr(e.os, "access", lambda path, mode: False)
    assert not e.measure(e.ROOT, tmp_path / "missing-tool")["rows"]
    monkeypatch.undo()
    monkeypatch.setattr(e.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    assert not e.measure(e.ROOT, tmp_path / "missing-storage")["rows"]
    monkeypatch.undo()
    policy = tmp_path / "ops/exclusion_manifest.yaml"
    policy.parent.mkdir()
    policy.write_text("retired_experiments:\n  - experiment_id: 8347\n")
    assert not e.measure(tmp_path, tmp_path / "retired")["rows"]
    policy.write_text("retired_experiments: []\n")
    monkeypatch.setattr(
        e.authority,
        "authority",
        lambda *args: dict(activated=False, tasks=[dict(id=e.TASK, deliverable="wrong")]),
    )
    assert not e.measure(tmp_path, tmp_path / "wrong-authority")["rows"]
    monkeypatch.undo()
    monkeypatch.setattr(e.authority, "authenticate", lambda *args: {})
    assert not e.measure(e.ROOT, tmp_path / "bad-terminal")["rows"]
    monkeypatch.undo()
    real_manifest = e.kernel.manifest
    monkeypatch.setattr(e.kernel, "manifest", lambda: dict(real_manifest(), seed=-1))
    assert not e.measure(e.ROOT, tmp_path / "protocol-drift")["rows"]


def test_real_coverage_report_is_retained(measured, tmp_path):
    """SCENARIO-REPORT-8347-CLI: readiness reads actual owned statement counts."""
    import coverage
    from carnot.reporting import local_consumer_execution_8347 as runner

    cov = coverage.Coverage(
        config_file=False, data_file=str(tmp_path / "data"), include=[str(e.ROOT / e.MODULE)]
    )
    with cov.collect():
        e.progress("measured_coverage_control")
    cov.json_report(outfile=str(tmp_path / "coverage.json"))
    original, raw = measured
    work = dict(original)
    runner.retain_coverage(tmp_path, tmp_path / "retained", work)
    value = e.build(work, raw, [{"name": "unit_control", "passed": True}])
    assert not value["acceptance_gates"]["owned_coverage"]
    assert value["owned_coverage_reference"]["sha256"]


def test_primitive_rejection_paths(measured, tmp_path):
    """SCENARIO-VERIFY-8347-REPLAY: rehashed primitive changes fail independent replay."""
    from copy import deepcopy
    from carnot.reporting.current_work_receipt import atomic_json

    original, _ = measured
    path = tmp_path / (e.NAME + ".json")

    def candidate(work, name):
        raw = tmp_path / name
        raw.mkdir()
        protocol = json.loads(Path(original["protocol_reference"]["path"]).read_bytes())
        if name == "protocol":
            protocol["seed"] = -1
        atomic_json(raw / "protocol.json", protocol)
        work["protocol_reference"] = e.base.reference(raw / "protocol.json")
        atomic_json(raw / "measurement.json", work)
        value = e.build(work, raw, [{"name": "unit_control", "passed": True}])
        atomic_json(path, value)
        assert not e.replay(path)

    candidate(dict(original), "protocol")
    candidate(dict(original, numeric={"passed": False}), "numeric")
    stored = deepcopy(original["states"][0])
    for arm in ["full", "indexed"]:
        stored["arms"][arm]["cache"]["0"]["p"] += 0.125
    candidate(dict(original, states=[stored, *original["states"][1:]]), "durable-cache")
    saved = dict(original["recovered_states"][0], version=-1)
    candidate(
        dict(original, recovered_states=[saved, *original["recovered_states"][1:]]), "recovery"
    )
    value = e.build(
        original,
        Path(original["protocol_reference"]["path"]).parent,
        [{"name": "unit_control", "passed": True}],
    )
    atomic_json(path, dict(value, reproducibility_checksum="sha256:wrong"))
    assert not e.replay(path)
    atomic_json(path, value)
    closure = json.loads(Path(original["historical_fixture_manifest"]["path"]).read_bytes())
    operand = Path(closure["operands"][0]["snapshot_path"])
    before = operand.read_bytes()
    operand.write_bytes(b"changed")
    assert not e.replay(path)
    operand.write_bytes(before)
