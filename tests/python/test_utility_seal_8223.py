"""REQ-VERIFY-8223 / REQ-REPORT-8223: target-free predictions and terminal custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import utility_seal_execution_8223 as e


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """An external working directory proves that the installed runner resolves imports."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private seal CLI", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print("after private seal CLI", result.returncode, flush=True)
    return result


def test_natural_seal_and_negative_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-8223-SEAL: original features, slots and permission survive every arm."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    assert all(c["passed"] for c in work["checks"]), work["checks"]
    data, reduced = work["evidence"], work["diagnostics"]
    assert len(reduced["rows"]) == 128
    assert len(reduced["prediction_rows"]) == 128 * 74
    assert reduced["intended_count"] == 128
    assert (
        sum(
            reduced[k]
            for k in ["completed_count", "failed_count", "excluded_count", "censored_count"]
        )
        == 128
    )
    assert len({r["source_cluster_id"] for r in reduced["rows"]}) == 128
    assert reduced["energy_parity_max_error"] < 1e-10
    assert e.numeric.reduce(data) == reduced
    for row in reduced["prediction_rows"]:
        assert row["chosen_action"] in ["accept", "reject", "escalate"]
        assert row["action"] != "accept" or row["baseline_action"] == "accept"
        assert row["state_sha256"]
        assert row["p"] is not None or row["action"] == "escalate"
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["utility_predictions_ready_score"] == 1
    assert len(value["original_reserved_mask"]) == 128
    assert value["verdict_class"] == "null" and value["labels_opened"] is False
    assert value["MODEL_SPECS"] == []
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert (
        e.build(work, tmp_path / "raw", [dict(passed=False)])["utility_predictions_ready_score"]
        == 0
    )
    for key, expected in [
        ("target", "evaluator_label"),
        ("heads", "modified_head"),
        ("slots", "original_source_join"),
    ]:
        changed = deepcopy(data)
        if key == "target":
            changed["public"]["features"][0]["nested"] = {"oracle_y": 1}
        elif key == "heads":
            changed["parameters"]["models"]["energy_original:none"]["patches"].append({"delta": 1})
        else:
            changed["public"]["roster"].pop()
        with pytest.raises(ValueError, match=expected):
            e.numeric.reduce(changed)
    monkeypatch.setattr(e.numeric.kernel, "energies", lambda p: (0.0, 0.0))
    with pytest.raises(ValueError, match="energy_parity"):
        e.numeric.reduce(data)


def test_real_cli_and_blocked_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8223-CLI: normal child, date failure and absent operands have real exits."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    assert value["utility_predictions_ready_score"] == 1
    value["rows"][0]["numerator"] = 99
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    blocked = tmp_path / "blocked" / output.name
    assert cli(tmp_path, "--root", tmp_path / "absent", "--fixture-output", blocked).returncode == 0
    value = json.loads(blocked.read_bytes())
    assert value["verdict_class"] == "blocked" and value["utility_predictions_ready_score"] == 0
    assert value["gate_check_summary"][-1]["observed"] is None
    assert cli(tmp_path, "--cold-replay", blocked).returncode == 0
    assert cli(tmp_path, "--date", "19000101").returncode == 2
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/forbidden.json").returncode == 2
    worker = tmp_path / "worker/measurement.json"
    assert cli(tmp_path, "--worker-output", worker).returncode == 0
    assert json.loads(worker.read_bytes())["diagnostics"]


def test_failures_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8223-CLI: schema blocks differ from failed owned numerical work."""
    work = e.measure(e.ROOT, tmp_path / "mutated", mutation="source")
    assert e.build(work, tmp_path / "mutated", [dict(passed=True)])["verdict_class"] == "blocked"
    monkeypatch.setattr(e, "PROTOCOL_VALUE", {})
    work = e.measure(e.ROOT, tmp_path / "schema")
    assert work["checks"][-1]["artifact_field"] == "authenticated_input_schema"
    monkeypatch.undo()
    monkeypatch.setattr(
        e.numeric, "reduce", lambda *a: (_ for _ in ()).throw(ValueError("owned_error"))
    )
    work = e.measure(e.ROOT, tmp_path / "owned")
    assert work["owned_failure"] == "owned_error"
    assert e.build(work, tmp_path / "owned", [dict(passed=True)])["verdict_class"] == "disqualified"
    specs = e.manifest(tmp_path, tmp_path / (e.NAME + ".json"))
    checks = {s["name"]: s for s in specs["commands"]}
    assert "--fail-under=100" in checks["coverage_report"]["argv"]
    assert "--files" in checks["spec_coverage"]["argv"]
    assert "--strict" in checks["strict_mypy"]["argv"]
    assert "tests/python/test_source_boundary_7852.py" in checks["consumer_and_E2E015_019"]["argv"]


def test_replay_custody_and_rehashed_primitives(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8223-SEAL: replacing hashes cannot authorize changed original evidence."""
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
    assert not e.replay(tmp_path / "absent")
    changed = deepcopy(original)
    changed["reproducibility_checksum"] = "wrong"
    atomic_json(output, changed)
    assert not e.replay(output)
    changed = deepcopy(original)
    changed["raw_shard_hashes"][0]["sha256"] = "wrong"
    changed["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, changed)
    assert not e.replay(output)
    log = Path(receipt["stdout_path"])
    log.write_bytes(b"tampered")
    atomic_json(output, original)
    assert not e.replay(output)
    log.write_bytes(b"")
    saved = (raw / "primitive_evidence.json").read_bytes()
    changed = deepcopy(work)
    changed["evidence"]["parameters"]["models"]["energy_original:none"]["patches"] = []
    changed["evidence"]["parameters"]["baseline"]["weights"][0] += 1
    changed["evidence"]["parameters_sha256"] = canonical_hash(changed["evidence"]["parameters"])
    atomic_json(raw / "primitive_evidence.json", changed["evidence"])
    atomic_json(raw / "measurement.json", changed)
    changed["raw_shard_hashes"][0]["sha256"] = sha256_file(raw / "primitive_evidence.json")
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [receipt]))
    assert not e.replay(output)
    (raw / "primitive_evidence.json").write_bytes(saved)
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, original)
    assert e.replay(output)
    changed = deepcopy(work)
    changed["evidence"]["public"]["features"][0]["x"][0] += 1
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [receipt]))
    assert not e.replay(output)


def test_state_and_seal_drift(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8223-SEAL: self-consistent new hashes still require original custody."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    for field, message in [
        ("baseline", "modified_head"),
        ("role_hashes", "role_hash"),
        ("models", "arm_manifest"),
    ]:
        data = deepcopy(work["evidence"])
        if field == "baseline":
            data["parameters"][field]["weights"][0] += 1
        elif field == "role_hashes":
            data["parameters"][field]["reserved"] = "wrong"
        else:
            data["parameters"][field].pop("energy_original:none")
        data["parameters_sha256"] = canonical_hash(data["parameters"])
        with pytest.raises(ValueError, match=message):
            e.numeric.reduce(data)
    output = tmp_path / (e.NAME + ".json")
    saved = (raw / "sealed_predictions.json").read_bytes()
    seal = json.loads(saved)
    seal["labels_opened"] = True
    (raw / "sealed_predictions.json").chmod(0o600)
    atomic_json(raw / "sealed_predictions.json", seal)
    work["raw_shard_hashes"][1]["sha256"] = sha256_file(raw / "sealed_predictions.json")
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, e.build(work, raw, [dict(passed=True)]))
    assert not e.replay(output)
    (raw / "sealed_predictions.json").write_bytes(saved)
    work["raw_shard_hashes"][1]["sha256"] = sha256_file(raw / "sealed_predictions.json")
    work["refs"][1]["sha256"] = "wrong"
    atomic_json(raw / "measurement.json", work)
    atomic_json(output, e.build(work, raw, [dict(passed=True)]))
    assert not e.replay(output)


def test_rehashed_upstream_and_reduction(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8223-SEAL: copied custody and cached arithmetic are independent checks."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    output = tmp_path / (e.NAME + ".json")
    changed = deepcopy(work)
    changed["diagnostics"]["energy_parity_max_error"] = 0.99
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [dict(passed=True)]))
    assert not e.replay(output)
    changed = deepcopy(work)
    ref = next(r for r in changed["refs"] if r["upstream_path"] == str(e.ROOT / e.UPSTREAM))
    forged = tmp_path / "forged_upstream.json"
    value = json.loads(Path(ref["path"]).read_bytes())
    value["utility_fit_ready_score"] = 9
    atomic_json(forged, value)
    ref.update(path=str(forged), sha256=sha256_file(forged))
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [dict(passed=True)]))
    assert not e.replay(output)
