"""REQ-REPORT-8317 / REQ-VERIFY-8317: real accounting and private execution."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import v717_capstone as runner
from carnot.reporting import v717_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


def fixture(root):
    root.mkdir()
    for name in [e.DESIGN, e.STAGED, e.ACTIVE, e.PROTOCOL]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        source = e.ROOT / name
        if source.is_file():
            path.write_bytes(source.read_bytes())
    return root


def cli(parent, *args):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)],
        cwd=parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_absence_and_failed_checks(tmp_path):
    """SCENARIO-REPORT-8317-DISPOSITIONS: unavailable observations are not zeros."""
    work = e.measure(fixture(tmp_path / "root"), tmp_path / "raw")
    value = e.reduce(work, [dict(passed=True, normal_exit=True, scope="owned")])
    assert value["intended_count"] == value["completed_count"] == 14
    assert [r["experiment_id"] for r in value["rows"]] == list(range(8304, 8318))
    assert value["missing_output_count"] == 13
    assert all(r["honest_verdict"] is None for r in value["rows"][:-1])
    assert value["capstone_execution_ready_score"] == 1
    assert value["honest_verdict"] == "complete_blocked_upstream_evidence"
    assert value["H1"]["statistics"] is value["H2"]["statistics"] is None
    assert value["H2"]["intended_count"] == 88
    assert value["H2"]["retention_intended_count"] == 32
    assert not value["polarfire_graduation"]["graduated"]
    assert e.reduce(work, [])["verdict_class"] == "disqualified"
    changed = deepcopy(work)
    changed["tasks"][0]["title"] = "changed"
    with pytest.raises(ValueError, match="contract"):
        e.reduce(changed, [])


def test_actual_inputs(tmp_path, monkeypatch):
    """REQ-REPORT-8317: current disqualifications cannot erase board obligations."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    value = e.reduce(work, [dict(passed=True, scope="owned")])
    assert value["actual_executed_task_count"] == 8
    assert value["pre_gate_count"] == 2 and value["missing_output_count"] == 4
    assert value["rows"][2]["disposition"] == "authenticated_upstream_failure"
    assert value["rows"][3]["disposition"] == "authenticated_upstream_failure"
    assert value["local_mechanics_ready_score"] == 0
    assert value["polarfire_graduation"]["graduated"]
    assert value["live_call_accounting"]["canary"]["producer_executed"] is False
    assert value["live_call_accounting"]["canary"]["counts"] is None
    assert value["science_ready_score"] == 0
    assert any(r["decision"] == "retire_exact_repeated_scope" for r in value["retirements"])
    changed = deepcopy(work)
    gate = changed["inputs"][9]["reference"]
    body = e.read(gate)
    body["gates_evaluated"][0]["artifact_sha256"] = "sha256:changed"
    path = tmp_path / "gate.json"
    atomic_json(path, body)
    gate.update(snapshot_path=str(path), sha256=sha256_file(path))
    assert (
        e.reduce(changed, [dict(passed=True, scope="owned")])["rows"][9]["disposition"]
        == "unbound_pre_gate"
    )
    changed = deepcopy(work)
    next(r for r in changed["references"] if "expected_sha256" in r)["expected_sha256"] = (
        "sha256:changed"
    )
    assert any(
        g["artifact_field"] == "primitive_or_validation_sha256"
        for g in e.reduce(changed, [dict(passed=True, scope="owned")])["gate_check_summary"]
    )
    monkeypatch.setattr(
        e.prior.qualified, "outcome", lambda *a: (_ for _ in ()).throw(RuntimeError("reader"))
    )
    assert e.reduce(work, [dict(passed=True, scope="owned")])["verdict_class"] == "disqualified"


def test_private_cli(tmp_path):
    """SCENARIO-VERIFY-8317-CLI: standalone publication has real child receipts."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--root", root, "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    assert e.replay(output)
    value = json.loads(output.read_bytes())
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    for field, observed in [
        ("completed_count", 13),
        ("paper_ready", False),
        ("MODEL_SPECS", [{}]),
        ("experiment_id", 9),
    ]:
        changed = dict(value, **{field: observed})
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        path = tmp_path / (field + ".json")
        atomic_json(path, changed)
        assert cli(tmp_path, "--cold-replay", path).returncode == 1
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    assert cli(tmp_path, "--date", "20260101").returncode == 2
    assert not e.replay(tmp_path / "absent")
    assert (root / "docs/research-notes/v717-outcomes.md").is_file()
    altered = dict(value, reproducibility_checksum="invalid")
    atomic_json(tmp_path / "negative.json", altered)
    assert not e.replay(tmp_path / "negative.json")
    altered = deepcopy(value)
    altered["validation_receipts"][0]["stdout_sha256"] = "sha256:changed"
    altered["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in altered.items() if k != "reproducibility_checksum"}
    )
    atomic_json(tmp_path / "receipt_tamper.json", altered)
    assert not e.replay(tmp_path / "receipt_tamper.json")


def test_manifest(tmp_path):
    """REQ-VERIFY-8317: freeze scoped coverage and both applicable private E2Es."""
    plan = runner.manifest(tmp_path)
    assert any(p["name"] == "private_E2E021" for p in plan)
    assert all("tests/python" not in p["argv"] for p in plan)
    assert "patch=subprocess" in (tmp_path / "coverage.ini").read_text()


def test_scratch_write_check(tmp_path, monkeypatch):
    """REQ-VERIFY-8317: reject a scratch write that cannot be read back."""
    original = Path.read_bytes
    monkeypatch.setattr(
        Path, "read_bytes", lambda p: b"corrupt" if p.name == "write_probe" else original(p)
    )
    with pytest.raises(ValueError, match="private_scratch"):
        runner.manifest(tmp_path)


def test_append_only_retirement(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8317-LEGACY-RETIREMENT: preserve history and deduplicate repeats."""
    root = tmp_path / "root"
    manifest = root / "ops/exclusion_manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "retired_extras:\n- id: preserved\n  reason: historical\n- name: legacy\n  reason: preserved without id\n"
    )
    original_manifest = manifest.read_bytes()
    path = tmp_path / "prior.json"
    atomic_json(
        path,
        dict(
            task_id="exp8302-gatemate-physical-delta",
            honest_verdict="complete_blocked_gatemate_physical_change",
        ),
    )
    ref = dict(path=str(path), exists=True, snapshot_path=str(path), sha256=sha256_file(path))
    value = dict(
        retirements=[
            dict(
                task_id="exp8316-gatemate-obligation",
                same_verdict_entries=[
                    dict(
                        experiment_id="exp8302-gatemate-physical-delta",
                        verdict="complete_blocked_gatemate_physical_change",
                        retire_if_same_verdict=True,
                    )
                ],
                prior_evidence=[dict(authenticated=True, references=[ref])],
                scope="unchanged physical probe only",
                reopening_condition="dated physical change",
            )
        ]
    )
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    runner.reconcile_manifest(root, output, value)
    once = manifest.read_bytes()
    assert once.startswith(b"retired_extras:\n- id: preserved")
    assert once.startswith(original_manifest)
    assert b"v717_exact_repeat_" in once
    entries = runner.yaml.safe_load(once)["retired_extras"]
    assert len(entries) == 3
    assert entries[1] == dict(name="legacy", reason="preserved without id")
    runner.reconcile_manifest(root, output, value)
    assert manifest.read_bytes() == once
    runner.reconcile_manifest(tmp_path / "absent", output, value)
    value["retirements"][0]["task_id"] = "exp8316-private-repeat-control"
    monkeypatch.setattr(runner.yaml, "safe_dump", lambda *a, **kw: "")
    with pytest.raises(ValueError, match="retirement_append_schema"):
        runner.reconcile_manifest(root, output, value)
    assert manifest.read_bytes() == once


def test_missing_authority_corruption_and_controls(tmp_path, monkeypatch):
    """REQ-REPORT-8317: missing authorities and corrupt bytes fail independently."""
    root = fixture(tmp_path / "root")
    (root / e.DESIGN).unlink()
    (root / e.ACTIVE).unlink()
    work = e.measure(root, tmp_path / "raw")
    good = [dict(scope="owned", passed=True)]
    assert e.reduce(work, good)["capstone_execution_ready_score"] == 0
    work["owned_validation_complete"] = False
    assert e.reduce(work, good)["honest_verdict"] == "complete_blocked_required_resources"
    work["owned_validation_complete"] = True
    foreign = deepcopy(work)
    foreign["inputs"][0]["reference"] = dict(foreign["inputs"][0]["reference"], path="foreign")
    with pytest.raises(ValueError, match="input_reference"):
        e.reduce(foreign, good)
    work["audits"] = [dict(name="failed_reader", passed=False, actual_exit=1, expected_exit=0)]
    assert any(
        g["artifact_field"] == "normal_exit" for g in e.reduce(work, good)["gate_check_summary"]
    )
    design = root / e.DESIGN
    design.write_text(
        (e.ROOT / e.DESIGN).read_text().replace("exp8304-contract-methods", "exp9999-wrong")
    )
    with pytest.raises(ValueError, match="fourteen"):
        e.measure(root, tmp_path / "wrong")
    design.write_bytes((e.ROOT / e.DESIGN).read_bytes())
    path = root / "results/experiment_8305_v717_cached_sentence_custody.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text("corrupt")
    corrupt = e.measure(root, tmp_path / "corrupt")
    assert e.reduce(corrupt, good)["rows"][1]["disposition"] == "corrupted_artifact"
    atomic_json(
        path,
        dict(
            schema="blocked_gate_check_v1",
            experiment=8305,
            blocked_at_layer="conductor_pre_gate",
            gates_evaluated=[],
        ),
    )
    gated = e.measure(root, tmp_path / "pre_gate_control")
    assert e.reduce(gated, good)["rows"][1]["honest_verdict"] is None


def test_failure_and_recovery(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8317-CLI: failed real child checks clear readiness and recover."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (e.NAME + ".json")
    original = runner.qualified.publish_primary
    attempts = []

    def reject_once(path, value, validator):
        attempts.append(value["verdict_class"])
        if len(attempts) == 1:
            raise ValueError("candidate_rejected")
        return original(path, value, validator)

    def fail_manifest(private):
        return [
            dict(
                name="real_owned_failure",
                argv=[sys.executable, "-c", "raise SystemExit(1)"],
                expected=0,
                deadline=10,
                scope="owned",
            )
        ]

    monkeypatch.setattr(runner, "manifest", fail_manifest)
    monkeypatch.setattr(runner.qualified, "publish_primary", reject_once)
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["capstone_execution_ready_score"] == 0
    assert e.replay(output)
    assert len(attempts) == 2
    altered = deepcopy(value)
    altered["source_artifact_hashes"] = []
    altered["raw_shard_hashes"] = []
    altered["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in altered.items() if k != "reproducibility_checksum"}
    )
    atomic_json(tmp_path / "foreign.json", altered)
    assert not e.replay(tmp_path / "foreign.json")
    altered = deepcopy(value)
    altered["validation_receipts"][0]["stdout_sha256"] = "sha256:changed"
    altered["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in altered.items() if k != "reproducibility_checksum"}
    )
    atomic_json(tmp_path / "stream.json", altered)
    assert not e.replay(tmp_path / "stream.json")
