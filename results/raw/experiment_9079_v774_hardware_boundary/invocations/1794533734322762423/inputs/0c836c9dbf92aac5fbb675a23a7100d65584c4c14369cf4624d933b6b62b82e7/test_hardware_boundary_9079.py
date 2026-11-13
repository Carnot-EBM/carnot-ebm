"""REQ-REPORT-9079-V774: hardware accounting preserves absent science and device limits."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hardware_boundary_9079 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def test_bounds():
    """SCENARIO-REPORT-9079-V774: only complete disjoint clocks supply a ceiling."""
    assert m.bounds(None)["acceleration_bound"] is None
    clocks = dict(total_s=100, host_s=1, startup_s=0, transfer_s=0, accelerated_s=99)
    result = m.bounds(clocks)
    assert result["nonaccelerable_fraction"] == 0.01
    assert result["acceleration_bound"] == 100
    assert result["strictly_more_than_100x_possible"] is False
    assert m.bounds(dict(clocks, host_s=0, accelerated_s=100))["acceleration_bound"] is None
    for change in [
        dict(total_s=0),
        dict(host_s=-1),
        dict(total_s=float("nan")),
        dict(accelerated_s=98),
        dict(transfer_s=None),
    ]:
        assert m.bounds(dict(clocks, **change))["acceleration_bound"] is None


def test_mapping_and_current_primitives():
    """REQ-VERIFY-9079-V774: unsupported deterministic work never becomes sampling."""
    reduced = m.reduce({"cost": {}, "feature": {}})
    assert len(reduced["per_board_rows"]) == 5
    assert len(reduced["operation_support_rows"]) == 35
    assert all(r["value"] is None for r in reduced["rows"])
    assert all(r["measured_here"] is False for r in reduced["operation_support_rows"])
    assert "k_max<=5" in reduced["per_board_rows"][2]["boundary"]
    assert "CPU" in reduced["per_board_rows"][3]["boundary"]
    assert "0xffffffff" in reduced["per_board_rows"][4]["boundary"]
    primitive = dict(
        unit_id="u",
        arm="a",
        status="completed",
        operation="basis_scoring",
        metric="duration_s",
        value=3,
        numerator=3,
        denominator=1,
    )
    out = m.reduce({"cost": {"operation_rows": [primitive]}, "feature": {"rows": [primitive]}})
    assert out["upstream_operation_rows"] == [primitive]
    assert out["upstream_feature_rows"] == [primitive]
    assert out["acceleration_bound"] is None
    assert all(r["value"] is None for r in out["rows"])


def test_collect_distinguishes_missing_and_zero(tmp_path):
    """SCENARIO-REPORT-9079-V774: authenticate before reduction, never require upstream success."""
    work = m.collect(m.ROOT, tmp_path / "raw")
    checks = work["checks"]
    assert any(r["observed"] == "missing_file" for r in checks)
    assert any(type(r["observed"]) is int and r["observed"] == 0 for r in checks)
    assert all(
        {
            "check",
            "path",
            "sha256",
            "upstream",
            "artifact_field",
            "op",
            "expected",
            "observed",
            "passed",
        }
        <= r.keys()
        for r in checks
    )
    assert all(Path(r["path"]).is_file() for r in work["references"])
    assert (tmp_path / "raw").stat().st_mode & 0o777 == 0o700
    for mutation in ["missing", "changed"]:
        root = tmp_path / mutation
        root.mkdir()
        (root / m.PROTOCOL).parent.mkdir(parents=True)
        protocol = dict(
            inputs=[dict(path="absent", sha256="sha256:no", experiment_id=9078)],
            vendor_claim_sources=[],
            dependencies={},
            upstream_cost_path="cost.json",
        )
        atomic_json(root / m.PROTOCOL, protocol)
        if mutation == "changed":
            (root / "absent").write_text("changed")
        found = m.collect(root, tmp_path / (mutation + "-raw"))
        assert found["failures"]


def test_missing_field_and_complete_current_rows(tmp_path):
    """REQ-REPORT-9079-V774: present incomplete science differs from absent bytes."""
    root = tmp_path / "repo"
    (root / m.PROTOCOL).parent.mkdir(parents=True)
    cost = root / "cost.json"
    atomic_json(cost, dict(experiment_id=9078, operation_rows=[dict(operation="basis_scoring")]))
    atomic_json(root / "feature.json", {})
    protocol = dict(
        inputs=[
            dict(
                path="feature.json",
                sha256=m.ref(root / "feature.json")["sha256"],
                experiment_id=9073,
            )
        ],
        vendor_claim_sources=[],
        dependencies={},
        upstream_cost_path="cost.json",
    )
    atomic_json(root / m.PROTOCOL, protocol)
    work = m.collect(root, tmp_path / "raw")
    assert work["cost"]["operation_rows"]
    assert any(r["observed"] == "missing_field" for r in work["checks"])


def test_build_and_replay(tmp_path):
    """SCENARIO-VERIFY-9079-V774: rehashed aggregates and changed operands fail replay."""
    work = m.collect(m.ROOT, tmp_path / "raw")
    validation = dict(passed=True, receipts=[], coverage=dict(complete=True))
    value = m.build(work, validation, "20261113", fixture=False)
    assert value["verdict_class"] == "blocked"
    assert value["hardware_map_ready_score"] == 0
    assert value["MODEL_SPECS"] == value["model_specs"] == value["trained_head_specs"] == []
    assert value["required_checks_passed"] is True
    assert (
        m.build(work, dict(validation, passed=False), "20261113", fixture=False)["verdict_class"]
        == "disqualified"
    )
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert m.replay(path)["passed"]
    corrupted = deepcopy(value)
    corrupted["reproducibility_checksum"] = "wrong"
    atomic_json(path, corrupted)
    assert not m.replay(path)["passed"]
    tamper = deepcopy(value)
    tamper["per_board_rows"][0]["boundary"] = "invented"
    tamper["reproducibility_checksum"] = m.checksum(tamper)
    atomic_json(path, tamper)
    assert not m.replay(path)["passed"]
    atomic_json(path, value)
    Path(value["raw_shard_hashes"][0]["path"]).write_text("changed")
    assert not m.replay(path)["passed"]
    assert not m.replay(tmp_path / "missing.json")["passed"]
    clean = dict(work, failures=[], checks=[])
    assert m.build(clean, validation, "20261113", fixture=False)["verdict_class"] == "null"


def test_replay_producer_and_validation_seals(tmp_path):
    """SCENARIO-VERIFY-9079-V774: original producers and validation logs cannot drift."""
    work = m.collect(m.ROOT, tmp_path / "raw")
    log = tmp_path / "log"
    log.write_text("passed")
    validation = dict(
        passed=True,
        receipts=[dict(log_path=str(log), log_sha256=m.ref(log)["sha256"])],
        coverage={},
    )
    value = m.build(work, validation, "20261113", fixture=False)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert m.replay(path)["passed"]
    log.write_text("changed")
    assert not m.replay(path)["passed"]
    value["validation_receipts"] = []
    saved = json.loads(Path(work["work_reference"]["path"]).read_bytes())
    saved["feature"] = {}
    atomic_json(Path(work["work_reference"]["path"]), saved)
    value["work_reference"] = m.ref(Path(work["work_reference"]["path"]))
    value["reproducibility_checksum"] = m.checksum(value)
    atomic_json(path, value)
    assert m.replay(path)["reason"] == "original_producer_drift"


def test_private_cli(tmp_path):
    """SCENARIO-REPORT-9079-V774: actual foreign CLI and cold replay need no PYTHONPATH."""
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    run = subprocess.run(
        [*command, "--date", "20261113", "--fixture", "--output", str(output)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(output.read_bytes())["fixture_only"] is True
    replay = subprocess.run(
        [*command, "--replay", str(output)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    for args in [["--date", "bad"], ["--fixture"], ["--replay", str(tmp_path / "missing")]]:
        failed = subprocess.run(
            [*command, *args], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=90
        )
        assert failed.returncode != 0


def test_validation_adapter(tmp_path, monkeypatch):
    """REQ-REPORT-9079-V774: affected consumer failure disqualifies owned work."""
    monkeypatch.setattr(m.checker, "validate", lambda raw: dict(passed=True, receipts=[]))
    monkeypatch.setattr(m, "run_commands", lambda *args, **kwargs: [dict(passed=False)])
    assert not m.validate(tmp_path)["passed"]
    monkeypatch.setattr(m, "run_commands", lambda *args, **kwargs: [dict(passed=True)])
    assert m.validate(tmp_path)["passed"]


def test_main_qualification(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9079-V774: the real producer validates before publishing."""
    validation = dict(passed=True, receipts=[], coverage=dict(complete=True))
    monkeypatch.setattr(m, "validate", lambda raw: validation)
    monkeypatch.setattr(m.checker, "terminal", lambda p, raw: m.replay(p))
    assert m.main(["--output", str(tmp_path / (m.NAME + ".json"))]) == 0
    with pytest.raises(ValueError, match="fixture_output"):
        m.main(["--fixture"])
    assert canonical_hash(dict(a=1))
