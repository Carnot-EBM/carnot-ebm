"""REQ-VERIFY-8195, REQ-REPORT-8195: private calibration and sealed custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import selective_energy_fit_8195 as e


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """Exercise the real entry point without relying on checkout imports."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private subprocess", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print("after private subprocess", result.returncode, flush=True)
    return result


def test_private_cli_and_cold_custody(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8195: success, block, aggregate and byte tampering."""
    output = tmp_path / "experiment_8195_fixture.json"
    child = cli(tmp_path, "--fixture-output", output)
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_text())
    assert value["selective_fit_ready_score"] == 1
    assert value["MODEL_SPECS"] == [] and value["call_ledger"] == []
    assert value["independent_generalization_score"] == 0
    assert value["verifier_is_oracle"] is True
    assert len(value["fit_rows"]) == 96 and len(value["temperature_rows"]) == 32
    assert len(value["calibration_rows"]) == 64 and len(value["quantile_rows"]) == 4
    assert value["equivalent_logistic_parity"]["passed"]
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    atomic_json(output, dict(value, eligible_count=1))
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    frozen = Path(value["frozen_heads_path"])
    old = frozen.read_bytes()
    frozen.write_text("{}")
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    frozen.write_bytes(old)
    missing = tmp_path / "absent.json"
    assert cli(tmp_path, "--cold-replay", missing).returncode == 1
    blocked = tmp_path / "block" / "experiment_8195_block.json"
    assert cli(tmp_path, "--fixture-output", blocked, "--mutation", "block").returncode == 0
    b = json.loads(blocked.read_text())
    assert b["honest_verdict"] == "complete_blocked_selective_protocol_ready_score"
    assert b["selective_fit_ready_score"] == 0 and e.replay(blocked)
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/forbidden.json").returncode == 2
    assert cli(tmp_path, "--mutation", "block").returncode == 2
    assert cli(tmp_path, "--date", "19990101").returncode == 2


def test_reduction_roles_masks_and_label_rejection(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8195: scalar arithmetic, isolation and missing evidence."""
    work = e.measure(e.ROOT, tmp_path / "fit", fixture_mode=True)
    data = work["evidence"]
    result = e.reduce(data)
    assert result["eligible_count"] == 192
    assert result["equivalent_logistic_parity"]["passed"]
    assert all(h["converged"] and len(h["geometry"]["centers"]) <= 16 for h in data["heads"])
    assert len(work["trained_head_specs"]) == 6
    for head in data["heads"]:
        assert set(head["head_fit_source_ids"]).isdisjoint(head["temperature_source_ids"])
        assert set(head["temperature_source_ids"]).isdisjoint(head["calibration_source_ids"])
    bad = deepcopy(data)
    bad["rows"][0]["evaluator_label"] = 1
    with pytest.raises(ValueError, match="evaluator_label"):
        e.reduce(bad)
    bad = deepcopy(data)
    bad["rows"][0]["source_cluster_id"] = bad["rows"][1]["source_cluster_id"]
    with pytest.raises(ValueError, match="source_identity"):
        e.reduce(bad)
    bad = deepcopy(data)
    bad["roles"]["head_fit"].reverse()
    with pytest.raises(ValueError, match="source_identity"):
        e.reduce(bad)
    masked = deepcopy(data)
    masked["rows"][0].update(x=None, status="excluded", exclusion_reason="missing_evidence")
    reduced = e.reduce(masked)
    assert reduced["eligible_count"] == 191
    unknown = deepcopy(data)
    unknown["rows"][0]["y"] = None
    assert e.reduce(unknown)["eligible_count"] == 191
    assert all(
        r["action"] == "escalate"
        for r in reduced["rows"]
        if r["unit_id"] == data["rows"][0]["unit_id"] and r["arm"] != "frozen_v707_radial"
    )
    disqualified = e.build(work, tmp_path / "fit", [dict(passed=False)], fixture=True)
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["selective_fit_ready_score"] == 0
    output = tmp_path / "experiment_8195_unit.json"
    atomic_json(output, e.build(work, tmp_path / "fit", [dict(passed=True)], fixture=True))
    assert e.replay(output)
    value = json.loads(output.read_text())
    shard = Path(value["measurement_reference"]["path"])
    saved = shard.read_bytes()
    changed = json.loads(saved)
    changed["evidence"]["heads"][0]["weights"][0] += 1
    atomic_json(shard, changed)
    value["measurement_reference"]["sha256"] = sha256_file(shard)
    atomic_json(output, value)
    assert not e.replay(output)
    shard.write_bytes(saved)


def test_natural_inputs_and_owned_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8195: authenticated roles, real operands and normal validation."""
    specs = e.manifest(tmp_path, tmp_path / "candidate.json")
    assert specs["repository_health"]["argv"][-2:] == ["tests/python", "-q"]
    assert all("::" not in arg for spec in specs["commands"][5:] for arg in spec["argv"])
    work = e.measure(e.ROOT, tmp_path / "natural")
    assert work["evidence"], work["checks"][-1]
    assert all(c["passed"] for c in work["checks"])
    output = tmp_path / "experiment_8195_natural.json"
    value = e.build(work, tmp_path / "natural", [dict(passed=True)])
    atomic_json(output, value)
    assert value["honest_verdict"] == "complete_null_selective_fit_sealed"
    assert value["selective_fit_ready_score"] == 1 and e.replay(output)
    frozen = json.loads(Path(value["frozen_heads_path"]).read_text())
    assert frozen["arms"] == e.n.ARMS and frozen["costs"] == e.CONFIG["costs"]
    missing = e.measure(tmp_path, tmp_path / "missing")
    blocked = e.build(missing, tmp_path / "missing", [dict(passed=True)])
    assert (
        blocked["verdict_class"] == "blocked"
        and blocked["gate_check_summary"][-1]["passed"] is False
    )
    with monkeypatch.context() as patcher:
        patcher.setattr(e.execution, "run_check", lambda *a, **kw: dict(actual_exit=0))
        patcher.setattr(
            e.fit, "inputs", lambda *a: dict(checks=[dict(check="external", passed=False)], refs=[])
        )
        external = e.measure(e.ROOT, tmp_path / "external")
        assert external["checks"][-1]["check"] == "external"
    with monkeypatch.context() as patcher:
        rows, reserved, control = e.methods.fixture()
        rows[0].pop("source_id")
        patcher.setattr(e.methods, "fixture", lambda: (rows, reserved, control))
        malformed = e.measure(e.ROOT, tmp_path / "malformed", fixture_mode=True)
        assert malformed["checks"][-1]["check"] == "input_custody"
    monkeypatch.setattr(
        e.n, "train", lambda *a: (_ for _ in ()).throw(ValueError("fit_nonconvergence"))
    )
    failed = e.measure(e.ROOT, tmp_path / "failed", fixture_mode=True)
    assert failed["owned_failure"] == "fit_nonconvergence"
    assert (
        e.build(failed, tmp_path / "failed", [dict(passed=True)])["verdict_class"] == "disqualified"
    )


def test_parity_and_replay_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-8195: forged arithmetic and altered validation logs fail replay."""
    work = e.measure(e.ROOT, tmp_path / "fit", fixture_mode=True)
    output = tmp_path / "experiment_8195_replay.json"
    log = tmp_path / "validation.log"
    log.write_text("normal exit\n")
    receipts = [dict(passed=True, log_path=str(log), log_sha256=sha256_file(log))]
    atomic_json(output, e.build(work, tmp_path / "fit", receipts, fixture=True))
    assert e.replay(output)
    log.write_text("changed\n")
    assert not e.replay(output)
    log.write_text("normal exit\n")
    reduced_path = tmp_path / "fit/independent_reduction.json"
    saved = reduced_path.read_bytes()
    reduced_path.write_text("{}")
    value = json.loads(output.read_text())
    for ref in value["raw_shard_hashes"]:
        if ref["path"] == str(reduced_path):
            ref["sha256"] = sha256_file(reduced_path)
    atomic_json(output, value)
    assert not e.replay(output)
    reduced_path.write_bytes(saved)
    predict = e.n.predict

    def drift(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        result = predict(*args, **kwargs)
        if kwargs.get("scalar"):
            result[0]["p"] += 0.01
        return result

    monkeypatch.setattr(e.n, "predict", drift)
    with pytest.raises(ValueError, match="probability_decision_parity"):
        e.reduce(work["evidence"])
