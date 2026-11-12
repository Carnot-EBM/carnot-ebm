"""REQ-REPORT-9030-CUSTODY: authentic historical rows cannot erase exposure."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import retained_shard_a_9030 as d
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def audited(tmp_path_factory):
    """Use retained observations for science; test mutations stay in private scratch."""
    raw = tmp_path_factory.mktemp("retained-a")
    return d.audit(d.ROOT, raw)


def test_actual_retained_bytes_and_exposure(audited):
    """SCENARIO-REPORT-9030-CUSTODY: all actual rows survive the failed exposure gate."""
    assert len(audited["rows"]) == 512
    assert len({r["unit_id"] for r in audited["rows"]}) == 512
    assert audited["role_counts"] == dict(calibration=64, test=64, online=384)
    assert audited["raw_receipt_audit"]["passed"]
    assert audited["nonexposure_audit"]["published_test_outcomes"] == 64
    assert not audited["nonexposure_audit"]["passed"]
    assert {r["check"] for r in audited["checks"] if not r["passed"]} == {"nonexposure_audit"}
    assert audited["original_disqualified_primary"]["verdict_class"] == "disqualified"
    assert len(audited["original_disqualified_primary"]["flags"]) >= 2


def test_missing_and_changed_authority(tmp_path):
    """REQ-REPORT-9030-CUSTODY: neither absent nor edited gate bytes authorize an audit."""
    evidence = d.audit(tmp_path, tmp_path / "raw")
    assert evidence["rows"] == []
    assert evidence["checks"][0]["observed"] == "missing_file"
    name = next(iter(d.PINS))
    path = tmp_path / name
    atomic_json(path, {"provenance_ready_score": 1})
    assert d.audit(tmp_path, tmp_path / "changed")["checks"][0]["passed"] is False


def test_build_keeps_current_and_historical_work_separate(audited):
    """REQ-REPORT-9030-CUSTODY: failed current checks disqualify; exposure alone blocks."""
    value = d.build(audited, [], {}, {}, 1.0, accepted=True)
    assert value["honest_verdict"] == "complete_blocked_nonexposure_audit"
    assert value["shard_ready_score"] == value["independent_generalization_score"] == 0
    assert value["MODEL_SPECS"] == value["model_specs"] == []
    assert not any(value["model_invocation_counts"].values())
    assert value["completed_count"] == value["independent_count"] == 512
    assert value["verifier_is_oracle"]
    assert value["repository_health_passed"] is False
    assert d.build(audited, [], {}, {}, 1.0, accepted=False)["verdict_class"] == "disqualified"
    clean = deepcopy(audited)
    clean["checks"] = [r for r in clean["checks"] if r["passed"]]
    assert d.build(clean, [], {}, {}, 1.0, accepted=True)["verdict_class"] == "circular_positive"


def child(tmp_path, *args):
    """Run the installed entrypoint with no inherited repository import path."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    cmd = [sys.executable, "-u", str(d.ROOT / d.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        cmd[1:2] = ["-m", "coverage", "run"]
    print("before subprocess", args, flush=True)
    with subprocess.Popen(
        cmd, cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        for _ in range(4):
            try:
                stdout, stderr = process.communicate(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print("waiting subprocess completed=0 pending=1", flush=True)
        else:
            process.kill()
            stdout, stderr = process.communicate()
    result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
    print(
        "after subprocess",
        result.returncode,
        result.stdout[-500:],
        result.stderr[-500:],
        flush=True,
    )
    return result


def test_real_private_cli_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-9030-CUSTODY: primitive replay rejects repaired aggregate hashes."""
    output = tmp_path / (d.NAME + ".json")
    assert child(tmp_path, "--audit-only", "--output", output).returncode == 0
    assert child(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    original = deepcopy(value)
    value["completed_count"] -= 1
    value["reproducibility_checksum"] = d.checksum(value)
    atomic_json(output, value)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    value = deepcopy(original)
    work_path = Path(value["raw_shard_hashes"][0]["path"])
    work = json.loads(work_path.read_bytes())
    work["evidence"]["rows"][0]["value"] = 1 - work["evidence"]["rows"][0]["value"]
    atomic_json(work_path, work)
    value["raw_shard_hashes"] = [d.reference(work_path)]
    for field in ("rows", "response_rows", "first_response_rows"):
        value[field] = work["evidence"]["rows"]
    value["summary_recomputation"]["error_count"] = sum(row["numerator"] for row in value["rows"])
    value["reproducibility_checksum"] = d.checksum(value)
    atomic_json(output, value)
    assert child(tmp_path, "--cold-replay", output).returncode == 1
    missing = tmp_path / "missing" / (d.NAME + ".json")
    assert (
        child(
            tmp_path, "--audit-only", "--input-root", tmp_path / "absent", "--output", missing
        ).returncode
        == 0
    )
    assert child(tmp_path, "--cold-replay", missing).returncode == 0
    altered_root = tmp_path / "altered-authority"
    authority = altered_root / next(iter(d.PINS))
    atomic_json(authority, {"provenance_ready_score": 1})
    altered = tmp_path / "altered" / (d.NAME + ".json")
    assert (
        child(
            tmp_path, "--audit-only", "--input-root", altered_root, "--output", altered
        ).returncode
        == 0
    )
    changed = json.loads(altered.read_bytes())["gate_check_summary"][0]
    assert changed["passed"] is False and changed["observed"] != "missing_file"
    assert child(tmp_path, "--date", "bad").returncode == 2


def test_gate_and_receipt_refusal(tmp_path, monkeypatch):
    """REQ-REPORT-9030-CUSTODY: a false current gate and missing raw receipts block."""
    name = next(iter(d.PINS))
    path = tmp_path / name
    atomic_json(
        path,
        {"provenance_ready_score": 0, "required_checks_passed": True, "flagged_adversarial": False},
    )
    monkeypatch.setattr(d, "PINS", {name: (d.reference(path)["sha256"], "provenance_ready_score")})
    assert not d.audit(tmp_path, tmp_path / "bad-gate")["checks"][-3]["passed"]
    value = json.loads(path.read_bytes())
    value["provenance_ready_score"] = 1
    atomic_json(path, value)
    monkeypatch.setattr(d, "PINS", {name: (d.reference(path)["sha256"], "provenance_ready_score")})
    from carnot.reporting import invocation_provenance_9000 as original

    monkeypatch.setattr(
        original, "authenticate", lambda *_: dict(checks=[], references=[], failures=[{}])
    )
    assert d.audit(tmp_path, tmp_path / "bad-receipt")["rows"] == []


def test_runner_production_and_scanner_paths(audited, tmp_path, monkeypatch):
    """SCENARIO-REPORT-9030-CUSTODY: the writer requires owned exits and both validators."""
    from carnot.reporting import retained_shard_runner_9030 as r
    from carnot.reporting.experiment_7303_validation_scope import CommandSpec

    monkeypatch.setattr(r, "plan", lambda _: [CommandSpec("unit", ("true",), "current_scoped", 10)])
    monkeypatch.setattr(r.d, "audit", lambda *_: deepcopy(audited))
    monkeypatch.setattr(r.checks, "run_check", lambda *_: dict(passed=True, normal_exit=True))
    monkeypatch.setattr(r, "measured", lambda *_: dict(complete=True, files=[]))
    monkeypatch.setattr(r, "verify_measured", lambda *_: True)
    monkeypatch.setattr(r.checks, "receipts_pass", lambda *_: True)
    published = []

    def publish(path, value, validator):
        candidate = tmp_path / "validator.json"
        atomic_json(candidate, value)
        assert validator(candidate)["passed"]
        atomic_json(path, value)
        published.append(value)

    monkeypatch.setattr(r, "publish_primary", publish)
    output = tmp_path / (d.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    assert published[-1]["verdict_class"] == "blocked"
    monkeypatch.setattr(r.checks, "run_check", lambda *_: dict(passed=False, normal_exit=True))
    assert r.main(["--output", str(output)]) == 1
    assert len(published) == 1


def test_replay_error_paths(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9030-CUSTODY: changed authorities and primitives cannot be rehashed away."""
    from carnot.reporting import retained_shard_runner_9030 as r

    assert not r.replay(tmp_path / "missing")["passed"]
    path = tmp_path / "bad.json"
    atomic_json(path, {"reproducibility_checksum": "bad"})
    assert r.replay(path)["reason"] == "checksum_drift"
    value = {
        "source_artifact_hashes": [dict(path=str(tmp_path / "missing"), sha256="bad")],
        "raw_shard_hashes": [],
        "code_config_hashes": [],
    }
    value["reproducibility_checksum"] = d.checksum(value)
    atomic_json(path, value)
    assert r.replay(path)["reason"] == "authority_bytes"
    work = tmp_path / "work.json"
    atomic_json(
        work, dict(input_root=str(tmp_path), accepted=False, evidence=dict(rows=["original"]))
    )
    value.update(source_artifact_hashes=[], raw_shard_hashes=[d.reference(work)])
    value["reproducibility_checksum"] = d.checksum(value)
    atomic_json(path, value)
    monkeypatch.setattr(r.d, "audit", lambda *_: dict(rows=["changed"]))
    value.update(
        validation_receipts=[],
        owned_statement_coverage={},
        validation_scope_manifest={},
        duration_s=1.0,
    )
    value["reproducibility_checksum"] = d.checksum(value)
    atomic_json(path, value)
    monkeypatch.setattr(r.d, "build", lambda *_, **kw: json.loads(path.read_bytes()))
    assert r.replay(path)["reason"] == "primitive_drift:rows"
    evidence = dict(
        rows=[],
        role_counts={},
        nonexposure_audit={},
        raw_receipt_audit={},
        original_disqualified_primary={},
        downstream_evaluation_attempts=[],
        checks=[],
    )
    atomic_json(work, dict(input_root=str(tmp_path), accepted=False, evidence=evidence))
    value["raw_shard_hashes"] = [d.reference(work)]
    value["reproducibility_checksum"] = d.checksum(value)
    atomic_json(path, value)
    changed = deepcopy(evidence)
    changed["checks"] = [dict(check="exposure", artifact_field="test", observed=64, passed=False)]
    monkeypatch.setattr(r.d, "audit", lambda *_: changed)
    assert r.replay(path)["reason"] == "gate_drift"
    atomic_json(work, dict(accepted=True))
    value["raw_shard_hashes"] = [d.reference(work)]
    value["validation_scope_manifest"] = dict(frozen_ns=0)
    value["reproducibility_checksum"] = d.checksum(value)
    atomic_json(path, value)
    assert r.replay(path)["reason"] == "validation_drift"
