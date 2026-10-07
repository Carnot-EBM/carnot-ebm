"""REQ-VERIFY-8222 / REQ-REPORT-8222: fit, freeze and replay permitted corrections."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import utility_fit_execution_8222 as e


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """Use a fresh interpreter without ambient import paths to test the shipped runner."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private fitting CLI", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print("after private fitting CLI", result.returncode, flush=True)
    return result


def test_matched_fitting_and_selection(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8222-FIT: all roles, sources and random controls remain explicit."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    assert all(c["passed"] for c in work["checks"]), work["checks"]
    data, result = work["evidence"], work["diagnostics"]
    assert len(work["models"]) == len(work["fit_costs"]) == 72
    assert len(result["rows"]) == 192 * 72
    assert set(result["selected_depths"]) == set(work["models"])
    assert result["frozen_comparator"]["arm"] in e.PROTOCOL_VALUE["comparator"]["eligible"]
    assert set(result["mandatory_group_controls"]) == {"additive_group", "logistic_group"}
    assert set(r["condition"] for r in result["rows"]) == {
        "head_fit",
        "temperature_fit",
        "calibration",
    }
    assert len({r["source_cluster_id"] for r in result["rows"]}) == 192
    for key, model in work["models"].items():
        arm, seed = key.split(":")
        assert int(seed) in range(101, 121) if arm.endswith("random") else seed == "none"
        assert len(model["patches"]) <= 4
        if arm.endswith("local"):
            assert all(p["group"]["name"] != "global" for p in model["patches"])
        if arm.endswith("original"):
            assert result["selected_depths"][key] == 0
    assert e.numeric.reduce(data, work["models"]) == result
    resumed, costs = e.numeric.fit(data, tmp_path / "raw" / "arms")
    assert resumed == work["models"] and costs == work["fit_costs"]
    assert any(r["p"] is None and r["status"] == "excluded" for r in result["rows"])
    assert all(r["y"] is None for r in data["reserved_mask"])
    assert all(c["duration_s"] >= 0 for c in costs)
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["utility_fit_ready_score"] == 1 and value["verdict_class"] == "null"
    assert value["MODEL_SPECS"] == [] and value["independent_generalization_score"] == 0
    assert value["generalized_learning_benefit_score"] == 0
    assert value["completed_count"] + value["excluded_count"] == value["intended_count"]
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"
    changed = deepcopy(data)
    changed["rows"][0]["role"] = "reserved"
    with pytest.raises(ValueError, match="future_target"):
        e.numeric.fit(changed, tmp_path / "invalid")
    changed = deepcopy(data)
    changed["reserved_mask"][0]["y"] = 1
    with pytest.raises(ValueError, match="future_target"):
        e.numeric.reduce(changed, work["models"])
    with pytest.raises(TimeoutError, match="fit_deadline"):
        e.numeric.fit(data, tmp_path / "deadline", deadline_s=-1)
    checkpoint = tmp_path / "raw" / "arms" / "energy_original:none.json"
    saved = json.loads(checkpoint.read_bytes())
    saved["input_sha256"] = "wrong"
    atomic_json(checkpoint, saved)
    with pytest.raises(ValueError, match="checkpoint_input"):
        e.numeric.fit(data, tmp_path / "raw" / "arms")


def test_role_isolation_and_zero_depth(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8222-FIT: tune labels cannot fit a delta; zero depth can win."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    data = deepcopy(work["evidence"])
    fit_ids = {r["unit_id"] for r in data["roles"]["head_fit"]}
    for row in data["rows"]:
        if row["unit_id"] not in fit_ids:
            row["y"] = 1 - row["y"]
    models, _ = e.numeric.fit(data, tmp_path / "changed")
    assert models == work["models"]
    assert data["heads"] == work["evidence"]["heads"]
    for row in e.numeric.base_rows(data, "energy"):
        assert row["baseline_action"] in {"accept", "reject", "escalate"}
    assert e.numeric.role_hashes(data) == e.numeric.role_hashes(work["evidence"])


def test_real_cli_and_blocked_operands(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8222-CLI: natural private fit, cold replay and failure exits."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["utility_fit_ready_score"] == 1
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value["selected_depths"]["energy_original:none"] = 4
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert not e.replay(tmp_path / "absent.json")
    blocked = tmp_path / "blocked" / output.name
    assert cli(tmp_path, "--root", tmp_path / "absent", "--fixture-output", blocked).returncode == 0
    value = json.loads(blocked.read_bytes())
    assert value["verdict_class"] == "blocked" and value["utility_fit_ready_score"] == 0
    assert value["gate_check_summary"][-1]["observed"] is None
    assert cli(tmp_path, "--date", "19000101").returncode == 2
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/forbidden.json").returncode == 2
    worker = tmp_path / "worker/measurement.json"
    assert cli(tmp_path, "--worker-output", worker).returncode == 0
    assert json.loads(worker.read_bytes())["models"]


def test_authentication_and_owned_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8222-CLI: missing external evidence differs from failed fitting."""
    work = e.measure(e.ROOT, tmp_path / "mutation", mutation="source")
    assert not all(c["passed"] for c in work["checks"])
    assert e.build(work, tmp_path / "mutation", [dict(passed=True)])["verdict_class"] == "blocked"
    monkeypatch.setattr(
        e.numeric, "fit", lambda *a: (_ for _ in ()).throw(TimeoutError("fit_deadline"))
    )
    work = e.measure(e.ROOT, tmp_path / "timeout")
    assert work["owned_failure"] == "fit_deadline"
    assert (
        e.build(work, tmp_path / "timeout", [dict(passed=True)])["verdict_class"] == "disqualified"
    )
    monkeypatch.undo()
    monkeypatch.setattr(e, "PROTOCOL_VALUE", {})
    work = e.measure(e.ROOT, tmp_path / "schema")
    assert work["checks"][-1]["artifact_field"] == "authenticated_input_schema"
    assert not work["checks"][-1]["passed"]


def test_frozen_manifest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8222-CLI: owned checks include child coverage and private E2E."""
    specs = e.manifest(tmp_path, tmp_path / (e.NAME + ".json"))
    commands = {s["name"]: s for s in specs["commands"]}
    assert "--fail-under=100" in commands["coverage_report"]["argv"]
    assert "--strict" in commands["strict_mypy"]["argv"]
    assert "--files" in commands["spec_coverage"]["argv"]
    assert (
        "tests/python/test_source_boundary_7852.py" in commands["consumer_and_E2E015_019"]["argv"]
    )
    assert (
        "tests/python/test_experiment_7942_v689_sentence_labels.py"
        in commands["consumer_and_E2E015_019"]["argv"]
    )
    assert specs["repository_health"]["deadline_s"] <= 180
    assert all(s["expected_exit"] == 0 for s in specs["terminal_commands"])


def test_replay_rejects_rehashed_primitives(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8222-CLI: new hashes cannot authorize invented model deltas."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    receipt = e.run_check(
        e.ROOT,
        dict(name="true", argv=["/bin/true"], deadline_s=5, expected_exit=0),
        tmp_path,
        raw / "logs",
    )
    output = tmp_path / (e.NAME + ".json")
    original = e.build(work, raw, [receipt])
    atomic_json(output, original)
    assert e.replay(output)
    changed = deepcopy(original)
    changed["utility_fit_ready_score"] = 9
    atomic_json(output, changed)
    assert not e.replay(output)
    atomic_json(output, original)
    heads = raw / "fitted_heads.json"
    saved_heads = heads.read_bytes()
    atomic_json(heads, {})
    changed_work = deepcopy(work)
    for ref in changed_work["raw_shard_hashes"]:
        if ref["path"] == str(heads):
            ref["sha256"] = sha256_file(heads)
    atomic_json(raw / "measurement.json", changed_work)
    atomic_json(output, e.build(changed_work, raw, [receipt]))
    assert not e.replay(output)
    heads.write_bytes(saved_heads)
    changed_work = deepcopy(work)
    changed_work["protocol"] = dict(work["protocol"], task_id="invented")
    atomic_json(raw / "measurement.json", changed_work)
    atomic_json(output, e.build(changed_work, raw, [receipt]))
    assert not e.replay(output)
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, original)
    for reference in [
        original["measurement_reference"],
        original["raw_shard_hashes"][0],
        original["code_config_hashes"][0],
    ]:
        path = Path(reference["path"])
        saved = path.read_bytes()
        try:
            path.write_bytes(saved + b" ")
            assert not e.replay(output)
        finally:
            path.write_bytes(saved)
    log = Path(receipt["stdout_path"])
    log.write_bytes(b"invented stdout")
    assert not e.replay(output)
    log.write_bytes(b"")
    work["models"]["energy_global:none"]["patches"] = []
    atomic_json(raw / "fitted_heads.json", e.fitted_heads(work))
    for ref in work["raw_shard_hashes"]:
        if ref["path"] == str(raw / "fitted_heads.json"):
            ref["sha256"] = sha256_file(raw / "fitted_heads.json")
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, e.build(work, raw, [receipt]))
    assert not e.replay(output)
    data = deepcopy(work["evidence"])
    data["rows"][0]["y"] = 1 - data["rows"][0]["y"]
    work["evidence"] = data
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, e.build(work, raw, [receipt]))
    assert not e.replay(output)
    assert sha256_file(log) == receipt["stdout_sha256"]
