"""REQ-VERIFY-8235 / REQ-REPORT-8235: qualify the two diagnosed failure routes."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import delayed_utility_execution_8225 as legacy
from carnot.verify import learning_validation_8235 as e
from test_delayed_utility_learning_8225 import cli as legacy_cli


def save_control(tmp_path, value):
    """REQ-VERIFY-8235: keep receipts outside pytest's successful-fixture cleanup."""
    directory = Path(os.environ.get("CARNOT_8235_CONTROL_DIR", str(tmp_path)))
    atomic_json(directory / (value["case"] + "-8235-control.json"), value)


def test_authenticated_schema_exception(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8235-FAILURES: bind real bytes before malformed rows are read."""
    malformed = tmp_path / "authenticated-malformed.json"
    atomic_json(malformed, [])
    original = legacy.authenticate

    def authenticate(root, raw, work):
        inputs = original(root, raw, work)
        legacy.k.frozen.bind(work, malformed, sha256_file(malformed), raw)
        inputs["manifests"]["stream_feature_manifest"] = legacy.reference(malformed)
        return inputs

    monkeypatch.setattr(legacy, "authenticate", authenticate)
    work = legacy.measure(legacy.ROOT, tmp_path / "schema")
    assert not work["input_ready"] and not work["owned_failure"]
    assert all(c["passed"] for c in work["checks"][:-1])
    assert work["checks"][-1]["check"] == "input_schema"
    assert work["checks"][-1]["passed"] is False
    assert any(r["upstream_path"] == str(malformed) for r in work["refs"])
    save_control(
        tmp_path,
        dict(
            case="authenticated_schema_failure",
            checks=work["checks"],
            refs=work["refs"],
            passed=True,
        ),
    )


@pytest.mark.parametrize("prefix", ["stdout", "stderr"])
def test_actual_receipt_mismatch_after_byte_checks(tmp_path, prefix):
    """SCENARIO-VERIFY-8235-FAILURES: a receipt outside raw shards reaches rejection."""
    raw = tmp_path / "fixture"
    work = legacy.measure(legacy.ROOT, raw, fixture=True)
    receipt = legacy.run_check(
        legacy.ROOT,
        dict(name="receipt_control", argv=["/bin/true"], expected_exit=0, deadline_s=5),
        tmp_path,
        tmp_path / "receipt_logs",
    )
    value = legacy.build(work, raw, [receipt], fixture=True)
    output = tmp_path / (legacy.NAME + ".json")
    atomic_json(output, value)
    assert legacy.replay(output)
    assert receipt[prefix + "_path"] not in {r["path"] for r in value["raw_shard_hashes"]}
    Path(receipt[prefix + "_path"]).write_bytes(b"actual log mismatch")
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    reached = []

    def trace(frame, event, arg):
        if event == "return" and frame.f_code is legacy.replay.__code__:
            reached.append(frame.f_lineno)
        return trace

    previous = sys.getprofile()
    try:
        sys.setprofile(trace)
        assert not legacy.replay(output)
    finally:
        sys.setprofile(previous)
    assert e.failure_line_map()["replay_log_hash_mismatch"] in reached
    save_control(
        tmp_path,
        dict(
            case=prefix + "_receipt_mismatch",
            reached_line=e.failure_line_map()["replay_log_hash_mismatch"],
            passed=True,
            child_receipts=[
                dict(
                    r,
                    stdout=Path(r["stdout_path"]).read_text(),
                    stderr=Path(r["stderr_path"]).read_text(),
                )
                for r in work["child_exit_rows"]
            ],
            recovery=work["restart_state_hashes"],
        ),
    )


def test_validation_build_authentication_and_tamper(tmp_path):
    """REQ-REPORT-8235: readiness requires current checks, not the historical verdict."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    assert all(c["passed"] for c in work["checks"])
    assert work["original_failure_receipt"]["actual_exit"] == 2
    receipts = [dict(name="coverage_report", passed=True)]
    value = e.build(work, tmp_path / "raw", receipts, fixture=True)
    assert value["learning_execution_ready_score"] == 1
    assert value["verdict_class"] == "null"
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert value["generalized_learning_benefit_score"] == 0
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, value)
    assert e.replay(output)
    changed = deepcopy(value)
    changed["learning_execution_ready_score"] = 9
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(output, changed)
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent.json")
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"
    blocked = e.measure(tmp_path / "absent", tmp_path / "blocked")
    assert e.build(blocked, tmp_path / "blocked", receipts)["verdict_class"] == "blocked"
    assert blocked["checks"][-1]["observed"] is None


def test_coverage_and_replay_failure_gates(tmp_path):
    """REQ-REPORT-8235: absent coverage, changed primitive bytes and receipt logs fail closed."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    receipts = [dict(passed=True)]
    assert e.build(work, raw, receipts)["learning_execution_ready_score"] == 0
    report = raw / "logs/changed_code_coverage.json"
    atomic_json(
        report,
        dict(
            files={
                name: dict(summary=dict(num_statements=1, covered_lines=1), missing_lines=[])
                for name in e.OWNED
            }
        ),
    )
    assert e.build(work, raw, receipts)["learning_execution_ready_score"] == 0
    controls = []
    for case in [
        "authenticated_schema_failure",
        "stdout_receipt_mismatch",
        "stderr_receipt_mismatch",
    ]:
        path = tmp_path / (case + ".json")
        atomic_json(
            path,
            dict(
                case=case,
                passed=True,
                child_receipts=[dict(actual_exit=code) for code in [0, 73, 0, 73, 0]],
                recovery=[dict(passed=True)],
            ),
        )
        controls.append(e.reference(path))
    assert (
        e.build(work, raw, [dict(passed=True, control_evidence=controls)])[
            "learning_execution_ready_score"
        ]
        == 1
    )
    candidate = tmp_path / (e.NAME + ".json")
    value = e.build(work, raw, receipts, fixture=True)
    atomic_json(candidate, dict(value, reproducibility_checksum="bad"))
    assert not e.replay(candidate)
    ref = work["refs"][0]
    snapshot = Path(ref["path"])
    saved = snapshot.read_bytes()
    snapshot.write_bytes(saved + b" ")
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    snapshot.write_bytes(saved)
    for prefix in ["stdout", "stderr"]:
        log = tmp_path / (prefix + ".log")
        log.write_bytes(b"original")
        receipt = dict(
            passed=True, **{prefix + "_path": str(log), prefix + "_sha256": sha256_file(log)}
        )
        value = e.build(work, raw, [receipt], fixture=True)
        atomic_json(candidate, value)
        assert e.replay(candidate)
        log.write_bytes(b"changed")
        assert not e.replay(candidate)
    root = tmp_path / "malformed_root"
    path = root / e.UPSTREAM
    path.parent.mkdir(parents=True)
    path.write_text("{broken")
    malformed = e.measure(root, tmp_path / "malformed")
    assert malformed["checks"][-1]["check"] == "input_schema"


def test_control_archive_real_command(tmp_path):
    """REQ-VERIFY-8235: actual child exits and complete control bytes become durable evidence."""
    private, durable = tmp_path / "private", tmp_path / "durable"
    atomic_json(
        private / "control_reports/archive-8235-control.json",
        dict(case="test_archive", passed=True),
    )
    receipt = e.run_check(
        e.ROOT,
        dict(name="owned_unit_and_private_CLI", argv=["/bin/true"], expected_exit=0, deadline_s=5),
        private,
        durable,
    )
    assert receipt["passed"] and receipt["control_evidence"]
    assert Path(receipt["control_evidence"][0]["path"]).is_file()
    work = e.measure(e.ROOT, tmp_path / "work")
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work, tmp_path / "work", [receipt], fixture=True))
    assert e.replay(output)
    Path(receipt["control_evidence"][0]["path"]).write_bytes(b"changed control")
    assert not e.replay(output)


def test_manifest_and_real_private_cli(tmp_path):
    """SCENARIO-REPORT-8235-CLI: direct wrapper publishes, replays and rejects bad inputs."""
    specs = e.manifest(tmp_path, tmp_path / (e.NAME + ".json"))
    assert "patch = _exit" in (tmp_path / "coverage.ini").read_text()
    assert all(
        str(e.ROOT / name) in (tmp_path / "coverage.ini").read_text().split("[report]")[1]
        for name in e.OWNED
    )
    commands = {r["name"]: r for r in specs["commands"]}
    assert "--fail-under=100" in commands["coverage_report"]["argv"]
    assert (
        "tests/python/test_restricted_decision_audit_8210.py"
        in commands["consumer_and_E2E015_019"]["argv"]
    )
    from unittest.mock import patch

    with patch.object(legacy, "CLI", e.CLI):
        output = tmp_path / (e.NAME + ".json")
        assert legacy_cli(tmp_path, "--fixture-output", output).returncode == 0
        assert legacy_cli(tmp_path, "--cold-replay", output).returncode == 0
        assert legacy_cli(tmp_path, "--date", "19000101").returncode == 2
        assert legacy_cli(tmp_path, "--fixture-output", e.ROOT / "results/no.json").returncode == 2
    value = json.loads(output.read_bytes())
    assert value["learning_execution_ready_score"] == 1
