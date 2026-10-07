"""REQ-VERIFY-8238 / REQ-REPORT-8238: public decisions and immutable custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting import margin_prediction_seal_8238 as e


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """A fresh interpreter must resolve imports without relying on the checkout cwd."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private prediction CLI", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print("after private prediction CLI", result.returncode, flush=True)
    return result


def test_natural_and_negative_inputs(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8238-SEAL: original slots and permissions survive all heads."""
    work = e.measure(e.ROOT, tmp_path / "raw")
    assert all(c["passed"] for c in work["checks"]), work["checks"]
    assert not work["owned_failure"]
    data, result = work["evidence"], work["diagnostics"]
    assert len(result["rows"]) == 128
    assert len(result["prediction_rows"]) == 128 * 9
    assert result["intended_count"] == 128
    assert (
        sum(
            result[k]
            for k in ["completed_count", "failed_count", "censored_count", "excluded_count"]
        )
        == 128
    )
    assert e.numeric.reduce(data) == result
    for row in result["prediction_rows"]:
        assert row["chosen_action"] != "accept" or row["baseline_action"] == "accept"
        assert row["p_bad"] is not None or row["chosen_action"] == "escalate"
        assert row["source_sha256"] and row["head_sha256"]
        assert row["feasible_expected_losses"]
    value = e.build(work, tmp_path / "raw", [dict(passed=True)])
    assert value["margin_predictions_ready_score"] == 1
    assert value["verdict_class"] == "null" and value["labels_opened"] is False
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["public_worker_input_manifest"]["evaluator_label_paths"] == []
    assert e.build(work, tmp_path / "raw", [dict(passed=False)])["verdict_class"] == "disqualified"
    for mutation, message in [
        ("missing", "original_source_join"),
        ("duplicate", "original_source_join"),
        ("reordered", "slot_order"),
        ("head", "head_hash"),
        ("label", "evaluator_label"),
        ("role", "role_hash"),
        ("comparator", "comparator_hash"),
    ]:
        changed = deepcopy(data)
        if mutation == "missing":
            changed["public"]["roster"].pop()
        elif mutation == "duplicate":
            changed["public"]["roster"][1] = changed["public"]["roster"][0]
        elif mutation == "reordered":
            changed["public"]["roster"].reverse()
        elif mutation == "head":
            changed["heads"][0]["weights"][0] += 1
        elif mutation == "label":
            changed["public"]["features"][0]["nested"] = {"oracle_y": 1}
        elif mutation == "role":
            changed["roles"]["reserved"].reverse()
        else:
            changed["comparator"]["arm"] = "energy_margin"
        with pytest.raises(ValueError, match=message):
            e.numeric.reduce(changed)


def test_cli_and_custody(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8238-CLI: actual children publish, replay and reject tampering."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value = json.loads(output.read_bytes())
    value["rows"][0]["numerator"] = 99
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    blocked = tmp_path / "blocked" / output.name
    assert cli(tmp_path, "--root", tmp_path / "absent", "--fixture-output", blocked).returncode == 0
    value = json.loads(blocked.read_bytes())
    assert value["verdict_class"] == "blocked" and value["margin_predictions_ready_score"] == 0
    assert value["gate_check_summary"][-1]["observed"] is None
    assert len(value["rows"]) == 128
    assert cli(tmp_path, "--cold-replay", blocked).returncode == 0
    assert cli(tmp_path, "--date", "19000101").returncode == 2
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/forbidden.json").returncode == 2
    worker = tmp_path / "worker/measurement.json"
    assert cli(tmp_path, "--worker-output", worker).returncode == 0
    assert json.loads(worker.read_bytes())["diagnostics"]
    payload = worker.parent / "public_worker_input.json"
    assert (
        cli(
            tmp_path, "--public-input", payload, "--prediction-output", tmp_path / "direct.json"
        ).returncode
        == 0
    )
    changed = json.loads(payload.read_bytes())
    changed["public"]["roster"][0]["target"] = 1
    atomic_json(tmp_path / "injected.json", changed)
    assert (
        cli(
            tmp_path,
            "--public-input",
            tmp_path / "injected.json",
            "--prediction-output",
            tmp_path / "rejected.json",
        ).returncode
        == 1
    )


def test_replay_and_failure_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8238-SEAL: new hashes cannot legitimize altered frozen primitives."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    output = tmp_path / (e.NAME + ".json")
    receipt = e.run_check(
        e.ROOT,
        dict(name="true", argv=["/bin/true"], deadline_s=5, expected_exit=0),
        tmp_path,
        raw / "logs",
    )
    original = e.build(work, raw, [receipt])
    atomic_json(output, original)
    assert e.replay(output)
    assert not e.replay(tmp_path / "absent")
    for field in ["reproducibility_checksum", "raw_shard_hashes"]:
        changed = deepcopy(original)
        if field == "reproducibility_checksum":
            changed[field] = "wrong"
        else:
            changed[field][0]["sha256"] = "wrong"
            changed["reproducibility_checksum"] = canonical_hash(
                {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
            )
        atomic_json(output, changed)
        assert not e.replay(output)
    atomic_json(output, original)
    log = Path(receipt["stdout_path"])
    log.write_bytes(b"tampered")
    assert not e.replay(output)
    log.write_bytes(b"")
    for field in ["evidence", "diagnostics", "refs", "seal"]:
        changed = deepcopy(work)
        if field == "evidence":
            changed[field]["heads"][0]["weights"][0] += 1
        elif field == "diagnostics":
            changed[field]["prediction_rows"][0]["p_bad"] = 0.99
        elif field == "refs":
            changed[field][0]["sha256"] = "wrong"
        else:
            seal = json.loads((raw / "sealed_predictions.json").read_bytes())
            seal["labels_opened"] = True
            atomic_json(raw / "sealed_predictions.json", seal)
            changed["raw_shard_hashes"] = [
                e.reference(Path(r["path"])) for r in changed["raw_shard_hashes"]
            ]
        atomic_json(raw / "measurement.json", changed)
        atomic_json(output, e.build(changed, raw, [receipt]))
        assert not e.replay(output)
    work = e.measure(e.ROOT, tmp_path / "mutation", mutation="source")
    assert e.build(work, tmp_path / "mutation", [dict(passed=True)])["verdict_class"] == "blocked"
    monkeypatch.setattr(e, "inputs", lambda *a: (_ for _ in ()).throw(KeyError("schema")))
    work = e.measure(e.ROOT, tmp_path / "schema")
    assert work["checks"][-1]["artifact_field"] == "authenticated_input_schema"
    monkeypatch.undo()
    monkeypatch.setattr(e, "run_check", lambda *a, **k: dict(actual_exit=1, passed=False))
    work = e.measure(e.ROOT, tmp_path / "owned")
    assert work["owned_failure"]
    assert e.build(work, tmp_path / "owned", [dict(passed=True)])["verdict_class"] == "disqualified"
    specs = e.manifest(tmp_path, output)
    checks = {s["name"]: s for s in specs["commands"]}
    assert "--fail-under=100" in checks["coverage_report"]["argv"]
    assert "--files" in checks["spec_coverage"]["argv"]
    assert "--strict" in checks["strict_mypy"]["argv"]
    assert any(
        "test_restricted_decision_audit_8210.py" in arg
        for s in specs["commands"]
        for arg in s["argv"]
    )


def test_missing_probability_and_numerical_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-8238-SEAL: missing inputs escalate and safety checks can fail."""
    data = e.measure(e.ROOT, tmp_path / "raw")["evidence"]
    changed = deepcopy(data)
    changed["native"][0]["p0"] = None
    result = e.numeric.reduce(changed)
    assert len(result["rows"]) == 128 and len(result["prediction_rows"]) == 1152
    assert result["rows"][0]["status"] == "excluded"
    assert all(
        r["chosen_action"] == "escalate" and r["missing_reason"] == "missing_native_probability"
        for r in result["prediction_rows"]
        if r["slot"] == 1
    )
    for field, message in [("heads", "arm_manifest"), ("native", "native_source_join")]:
        changed = deepcopy(data)
        changed[field].pop()
        if field == "heads":
            changed["head_hashes"] = [canonical_hash(h) for h in changed["heads"]]
        with pytest.raises(ValueError, match=message):
            e.numeric.reduce(changed)
    monkeypatch.setattr(e.numeric.kernel, "energies", lambda p: (0.0, 0.0))
    with pytest.raises(ValueError, match="energy_parity"):
        e.numeric.reduce(data)
    monkeypatch.undo()
    monkeypatch.setattr(e.numeric.fitted, "score", lambda *a: dict(p_bad=0.25, action="accept"))
    with pytest.raises(ValueError, match="acceptance_subset_violation"):
        e.numeric.reduce(data)
    monkeypatch.setattr(e.numeric.fitted, "score", lambda *a: dict(p_bad=0.5, action="escalate"))
    result = e.numeric.reduce(data)
    assert all(
        r["chosen_action"] == "escalate"
        for r in result["prediction_rows"]
        if r["arm"] in e.numeric.ARMS
    )


def test_rehashed_public_payload_and_access_manifest(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8238-CLI: replay binds the public projection and access manifest."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    output = tmp_path / (e.NAME + ".json")
    changed = deepcopy(work)
    changed["evidence"]["heads"][0]["weights"][0] += 1
    for name in ["primitive_evidence", "public_worker_input"]:
        atomic_json(raw / (name + ".json"), changed["evidence"])
    changed["raw_shard_hashes"] = [
        e.reference(Path(r["path"])) for r in changed["raw_shard_hashes"]
    ]
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [dict(passed=True)]))
    assert not e.replay(output)
    for name in ["primitive_evidence", "public_worker_input"]:
        atomic_json(raw / (name + ".json"), work["evidence"])
    access = deepcopy(work["public_worker_input_manifest"])
    access["evaluator_label_paths"] = ["injected"]
    atomic_json(raw / "public_worker_input_manifest.json", access)
    changed = deepcopy(work)
    changed["raw_shard_hashes"] = [
        e.reference(Path(r["path"])) for r in changed["raw_shard_hashes"]
    ]
    atomic_json(raw / "measurement.json", changed)
    atomic_json(output, e.build(changed, raw, [dict(passed=True)]))
    assert not e.replay(output)
