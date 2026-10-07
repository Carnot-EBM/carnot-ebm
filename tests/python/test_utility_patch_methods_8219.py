"""REQ-VERIFY-8219 and REQ-REPORT-8219: freeze choices before future labels."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import utility_patch_methods_8219 as e


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """A fresh interpreter proves the script works outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private CLI", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print("after private CLI", result.returncode, flush=True)
    return result


def test_public_dictionary_and_cost_residuals() -> None:
    """SCENARIO-VERIFY-8219-WITNESSES: endpoints and duplicates are target blind."""
    public = [
        dict(unit_id=str(i), source_cluster_id=str(i), baseline_p=p, baseline_action="reject")
        for i, p in enumerate([0, 0.1, 0.25, 0.5, 0.75, 1, None])
    ]
    groups = e.freeze_dictionary(public)
    assert len(groups) == 6
    assert [g["distinct_fit_members"] for g in groups] == [7, 1, 1, 1, 1, 2]
    assert e.freeze_dictionary([])[0]["distinct_fit_members"] == 0
    rows = [dict(p=0.2, y=1, **r) for r in public]
    rows[-1]["p"] = None
    assert e.member(public[-1], groups[0])
    result = e.residuals(rows, groups)
    assert result[0]["probability_residual"] == pytest.approx(0.8 * 6 / 7)
    assert result[0]["accept_cost_residual"] == pytest.approx(5 * 0.8 * 6 / 7)
    assert result[0]["reject_cost_residual"] == pytest.approx(-0.8 * 6 / 7)
    assert result[0]["escalate_cost_residual"] == 0
    assert not result[0]["eligible"]
    assert e.residuals([], groups)[0]["probability_residual"] is None
    rows[0]["p"] = None
    assert e.residuals(rows, groups)[0]["available_members"] == 5
    for field in ["y", "source_label", "evaluation_selected"]:
        with pytest.raises(ValueError, match="public_schema"):
            e.freeze_dictionary([dict(public[0], **{field: 1})])
    for probability in [-1, 2, float("nan")]:
        with pytest.raises(ValueError, match="public_schema"):
            e.freeze_dictionary([dict(public[0], baseline_p=probability)])


def test_protocol_and_measurement(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8219-SCOPE: authenticated fit/tune-only reductions."""
    work = e.measure(e.ROOT, tmp_path / "good", fixture=True)
    assert all(c["passed"] for c in work["checks"])
    p = work["protocol"]
    assert p["H1"]["alpha"] == p["H2"]["alpha"] == 0.025
    assert p["H1"]["draws"] == p["H2"]["draws"] == 10000
    assert p["H1"]["minimum_valid_draws"] == p["H2"]["minimum_valid_draws"] == 9500
    assert len(p["role_manifest"]["reserved"]) == 128
    assert len(p["witness_dictionary"]) <= 11
    assert p["patch_rule"]["minimum_distinct_members"] == 8
    assert p["patch_rule"]["maximum_patches"] == 4
    assert len(p["static_arms"]) == 15
    assert p["H2"]["retention"]["minimum_complete"] == 48
    assert work["evidence"]["reserved_mask"][0]["y"] is None
    assert all(r["role"] in {"fit", "tune"} for r in work["evidence"]["rows"])
    assert work["diagnostics"]["energy_logistic_maximum_error"] <= 1e-10
    assert e.reduce_primitives(work["evidence"], p) == work["diagnostics"]
    receipts = [dict(name="owned", passed=True)]
    value = e.build(work, tmp_path / "good", receipts)
    assert value["verdict_class"] == "null" and value["utility_protocol_ready_score"] == 1
    assert not value["H1"]["measured_here"] and not value["H2"]["measured_here"]
    assert value["independent_generalization_score"] == 0
    assert len(value["rows"]) == value["intended_count"] == 128
    assert e.build(work, tmp_path / "good", [dict(passed=False)])["verdict_class"] == "disqualified"
    assert (
        e.build(work, tmp_path / "good", receipts, fixture=True)["verdict_class"]
        == "circular_positive"
    )
    changed = deepcopy(work["evidence"])
    changed["rows"][0]["role"] = "evaluation"
    with pytest.raises(ValueError, match="future_target"):
        e.reduce_primitives(changed, p)
    changed = deepcopy(work["evidence"])
    changed["reserved_mask"][0]["y"] = 1
    with pytest.raises(ValueError, match="future_target"):
        e.reduce_primitives(changed, p)
    missing = e.measure(tmp_path / "absent", tmp_path / "missing")
    blocked = e.build(missing, tmp_path / "missing", receipts)
    assert blocked["verdict_class"] == "blocked" and blocked["utility_protocol_ready_score"] == 0
    assert blocked["gate_check_summary"][-1]["observed"] is None
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    drift = e.measure(e.ROOT, tmp_path / "drift", fixture=True, mutation="source")
    assert e.build(drift, tmp_path / "drift", receipts)["verdict_class"] == "blocked"


def test_real_cli_replay_and_failures(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8219-CLI: actual children retain blocking and tamper exits."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    original = json.loads(output.read_bytes())
    assert original["utility_protocol_ready_score"] == 1
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    changed = deepcopy(original)
    changed["utility_protocol_ready_score"] = 9
    atomic_json(output, changed)
    assert not e.replay(output)
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, original)
    path = Path(original["measurement_reference"]["path"])
    saved = path.read_bytes()
    path.write_bytes(saved + b" ")
    assert not e.replay(output)
    path.write_bytes(saved)
    assert e.replay(output)
    path = Path(original["raw_shard_hashes"][0]["path"])
    saved = path.read_bytes()
    path.write_bytes(saved + b" ")
    assert not e.replay(output)
    path.write_bytes(saved)
    primitive = path.parent / "fixture_primitives.json"
    saved = primitive.read_bytes()
    atomic_json(primitive, {})
    changed = deepcopy(original)
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(primitive):
            ref["sha256"] = sha256_file(primitive)
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, changed)
    assert not e.replay(output)
    primitive.write_bytes(saved)
    atomic_json(output, original)
    assert e.replay(output)
    assert not e.replay(tmp_path / "absent.json")
    blocked = tmp_path / "blocked" / output.name
    assert (
        cli(tmp_path, "--root", tmp_path / "missing", "--fixture-output", blocked).returncode == 0
    )
    assert json.loads(blocked.read_bytes())["verdict_class"] == "blocked"
    assert cli(tmp_path, "--date", "19000101").returncode == 2
    assert (
        cli(tmp_path, "--fixture-output", e.ROOT / "results" / "never-write.json").returncode == 2
    )
    worker = tmp_path / "worker" / "measurement.json"
    assert cli(tmp_path, "--worker-output", worker).returncode == 0
    assert json.loads(worker.read_bytes())["diagnostics"]


def test_frozen_validation_and_schema_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8219-CLI: real manifest includes CLI coverage and private E2E."""
    plan = e.manifest(tmp_path, tmp_path / (e.NAME + ".json"))
    commands = {r["name"]: r for r in plan["commands"]}
    assert "--files" in commands["spec_coverage"]["argv"]
    assert "--strict" in commands["strict_mypy"]["argv"]
    assert "--fail-under=100" in commands["coverage_report"]["argv"]
    assert (
        "tests/python/test_source_boundary_7852.py" in commands["consumer_and_E2E015_019"]["argv"]
    )
    assert commands["owned_unit_and_private_CLI"]["deadline_s"] <= 240
    receipt = e.run_check(
        e.ROOT,
        dict(name="actual_exit", argv=["/bin/false"], deadline_s=5, expected_exit=1),
        tmp_path,
        tmp_path / "logs",
        heartbeat_s=0.05,
    )
    assert receipt["passed"] and receipt["actual_exit"] == 1
    report = tmp_path / "private_coverage_fixture.json"
    atomic_json(report, dict(fixture_only=True))
    copied = e.run_check(
        e.ROOT,
        dict(
            name="coverage_json",
            argv=["/bin/true", "-o", str(report)],
            deadline_s=5,
            expected_exit=0,
        ),
        tmp_path,
        tmp_path / "logs",
    )
    assert Path(copied["coverage_reference"]["path"]).read_bytes() == report.read_bytes()
    monkeypatch.setattr(
        e, "reduce_primitives", lambda *a: (_ for _ in ()).throw(ValueError("owned failure"))
    )
    work = e.measure(e.ROOT, tmp_path / "failed", fixture=True)
    assert (
        e.build(work, tmp_path / "failed", [dict(passed=True)])["verdict_class"] == "disqualified"
    )
    monkeypatch.undo()
    monkeypatch.setattr(e, "reduce_primitives", lambda *a: dict(energy_logistic_maximum_error=1))
    work = e.measure(e.ROOT, tmp_path / "parity_failure", fixture=True)
    assert work["owned_failure"] == "energy_logistic_identity"
    monkeypatch.undo()
    protocol = deepcopy(e.PROTOCOL_VALUE)
    protocol["source_artifact_hashes"] = []
    monkeypatch.setattr(e, "PROTOCOL_VALUE", protocol)
    work = e.measure(e.ROOT, tmp_path / "schema", fixture=True)
    assert any(not c["passed"] for c in work["checks"])


def test_replay_rehashed_primitive_and_receipt_drift(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8219-CLI: hashing edited operands cannot create science."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw, fixture=True)
    receipt = e.run_check(
        e.ROOT,
        dict(name="actual_pass", argv=["/bin/true"], deadline_s=5, expected_exit=0),
        tmp_path,
        raw / "logs",
    )
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, raw, [receipt], fixture=True)
    atomic_json(output, value)
    assert e.replay(output)
    stdout = Path(receipt["stdout_path"])
    saved = stdout.read_bytes()
    stdout.write_bytes(b"changed receipt")
    assert not e.replay(output)
    stdout.write_bytes(saved)
    changed = deepcopy(work)
    changed["diagnostics"]["fitted_patch_count"] = 1
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [receipt], fixture=True))
    assert not e.replay(output)
    changed = deepcopy(work)
    row = next(r for r in changed["evidence"]["rows"] if r["x"] is not None)
    row["y"] = 1 - row["y"]
    changed["diagnostics"] = e.reduce_primitives(changed["evidence"], changed["protocol"])
    atomic_json(raw / "fixture_primitives.json", changed["evidence"])
    changed["raw_shard_hashes"] = [e.reference(raw / "fixture_primitives.json")]
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [receipt], fixture=True))
    assert not e.replay(output)
