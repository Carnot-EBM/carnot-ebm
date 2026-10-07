"""REQ-VERIFY-8208 / REQ-REPORT-8208: fit roles and checked private publication."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pytest

from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import restricted_energy_8208 as n
from carnot.verify import restricted_energy_fit_8208 as e


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """Exercise real children outside checkout so import and replay paths matter."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private8208 CLI", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print("after private8208 CLI", result.returncode, flush=True)
    return result


def test_fit_roles_and_parameters(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8208-FIT: real solves change weights, never consume tune labels."""
    rows, roles, baseline = e.methods.fixture()
    trained = n.train(rows, roles)
    assert len(trained["heads"]) == 3
    assert len(trained["fit_fold_rows"]) == 60
    for head in trained["heads"]:
        assert len(head["weights"]) == 17
        assert head["parameter_hashes"]["before"] != head["parameter_hashes"]["after"]
        assert (
            head["solve_receipt"]["loss_trajectory"][-1]
            <= head["solve_receipt"]["loss_trajectory"][0]
        )
        assert head["ridge"] in n.base.RIDGES
        assert 0.25 <= head["temperature"] <= 4
        assert set(head["geometry"]["fit_source_ids"]) == {
            r["source_cluster_id"] for r in roles["head_fit"]
        }
    for fold in trained["fit_fold_rows"]:
        assert not set(fold["held_source_ids"]) & set(fold["geometry"]["fit_source_ids"])
    changed = deepcopy(rows)
    tune = {r["unit_id"] for r in roles["calibration"]}
    for row in changed:
        if row["unit_id"] in tune:
            row["y"] = 1 - row["y"]
    other = n.train(changed, roles)
    assert [h["weights"] for h in trained["heads"]] == [h["weights"] for h in other["heads"]]
    assert [h["temperature"] for h in trained["heads"]] == [
        h["temperature"] for h in other["heads"]
    ]
    data = dict(rows=rows, roles=roles, heads=trained["heads"], baseline=baseline)
    reduced = n.reduce(data)
    assert reduced["equivalent_logistic_parity"]["passed"]
    assert reduced["common_shift_invariance"]["passed"]
    assert reduced["selected_simple_control"]["selected"] in {"additive", "logistic"}
    assert all(
        r["action"] != "accept" or r["baseline_action"] == "accept"
        for r in reduced["rows"]
        if r["arm"] in n.rule.ARMS
    )
    rows[0]["x"] = None
    reduced = n.reduce(dict(data, rows=rows))
    assert any(r["exclusion_reason"] for r in reduced["rows"])
    rows[0]["x"] = [0.0] * 16
    rows[0]["historical_x"] = None
    assert n.reduce(dict(data, rows=rows))["rows"][0]["action"] == "escalate"


def test_reject_invalid_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8208-FIT: reject leakage, bad operands and solver failures."""
    rows, roles, _ = e.methods.fixture()
    for key, value in [("role", "reserved"), ("oracle_y", 1), ("x", [float("nan")] * 16)]:
        bad = deepcopy(rows)
        bad[0][key] = value
        with pytest.raises(ValueError):
            n.train(bad, roles)
    bad = deepcopy(rows)
    bad[0]["source_cluster_id"] = bad[1]["source_cluster_id"]
    with pytest.raises(ValueError):
        n.train(bad, roles)
    with pytest.raises(ValueError):
        n.train([], roles)
    with pytest.raises(TimeoutError):
        n.solve(np.ones((4, 17)), np.array([0, 1, 0, 1]), 0.01, 0)
    monkeypatch.setattr(n.base, "solve", lambda *a, **kw: {"converged": False})
    with pytest.raises(ValueError, match="nonconvergence"):
        n.train(rows, roles)


def test_private_cli_replay_and_block(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8208-CLI: real fit, missing branch and tampered custody."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--date", "20261006", "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["action_fit_ready_score"] == 1
    assert value["reserved_outcomes_opened"] is False
    assert value["verdict_class"] == "circular_positive"
    assert value["independent_generalization_score"] == 0
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    changed = deepcopy(value)
    changed["rows"][0]["numerator"] += 1
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    blocked = tmp_path / "block" / output.name
    assert cli(tmp_path, "--fixture-output", blocked, "--mutation", "source").returncode == 0
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    assert cli(tmp_path, "--date", "20261005").returncode == 2
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results" / output.name).returncode == 2


def test_additional_fit_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-8208-FIT: missing training data and failed temperature are fatal."""
    from types import SimpleNamespace

    rows, roles, baseline = e.methods.fixture()
    bad = deepcopy(rows)
    fit_ids = {r["unit_id"] for r in roles["head_fit"]}
    for row in bad:
        if row["unit_id"] in fit_ids:
            row["x"] = None
    with pytest.raises(ValueError, match="fit_operands"):
        n.train(bad, roles)
    monkeypatch.setattr(n, "minimize_scalar", lambda *a, **kw: SimpleNamespace(success=False))
    with pytest.raises(ValueError, match="temperature_nonconvergence"):
        n.train(rows, roles)
    monkeypatch.undo()
    trained = n.train(rows, roles)
    monkeypatch.setattr(n, "expit", lambda z: 0.5)
    with pytest.raises(ValueError, match="probability_decision_parity"):
        n.reduce(dict(rows=rows, roles=roles, baseline=baseline, **trained))


def test_owned_failure_external_schema_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-8208-CLI: execution failures differ from missing external operands."""

    def failed(*args: Any, **kw: Any) -> Any:
        raise ValueError("owned_solver_failure")

    monkeypatch.setattr(n, "train", failed)
    work = e.measure(e.ROOT, tmp_path / "owned", fixture=True)
    value = e.build(work, tmp_path / "owned", [dict(passed=True)])
    assert value["verdict_class"] == "disqualified" and value["action_fit_ready_score"] == 0
    monkeypatch.undo()
    monkeypatch.setattr(e.fit, "bind", lambda *a, **kw: [])
    work = e.measure(e.ROOT, tmp_path / "schema")
    assert e.build(work, tmp_path / "schema", [dict(passed=True)])["verdict_class"] == "blocked"
    monkeypatch.undo()
    work = e.measure(tmp_path / "absent", tmp_path / "absent_work")
    assert work["checks"][-1]["observed"] is None
    assert (
        e.build(work, tmp_path / "absent_work", [dict(passed=False)])["verdict_class"]
        == "disqualified"
    )
    specs = e.manifest(tmp_path, tmp_path / "candidate.json")
    assert e.CLI in " ".join(specs["terminal_commands"][0]["argv"])
    assert any("--files" in s["argv"] for s in specs["commands"])
    assert any("--fail-under=100" in s["argv"] for s in specs["commands"])
    assert specs["repository_health"]["argv"][1:3] == ["tests/python", "-q"]


def test_real_worker_and_replay_rejections(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8208-CLI: real manifest inputs, complete streams and forged custody."""
    from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

    raw = tmp_path / "worker"
    raw.mkdir()
    result = cli(tmp_path, "--worker-output", raw / "measurement.json")
    assert result.returncode == 0, result.stdout + result.stderr
    work = json.loads((raw / "measurement.json").read_bytes())
    assert not work["owned_failure"], work["checks"]
    assert all(c["passed"] for c in work["checks"]), work["checks"]
    value = e.build(work, raw, [dict(passed=True)])
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert value["verdict_class"] == "null" and value["action_fit_ready_score"] == 1
    assert value["completed_count"] == value["independent_count"]
    frozen_value = json.loads((raw / "frozen_heads.json").read_bytes())
    assert frozen_value["costs"] == n.base.CONFIG["costs"]
    assert frozen_value["source_artifact_hashes"]
    assert frozen_value["train_tune_action_choices"]
    assert not (raw / "frozen_heads.json").stat().st_mode & 0o222
    assert "original_local_set" in {r["arm"] for r in value["rows"]}
    assert e.replay(candidate)

    def write_changed(changed: dict[str, Any]) -> None:
        changed["reproducibility_checksum"] = canonical_hash(
            {k: v for k, v in changed.items() if k != "reproducibility_checksum"}
        )
        atomic_json(candidate, changed)

    changed = deepcopy(value)
    changed["action_fit_ready_score"] = 0
    write_changed(changed)
    assert not e.replay(candidate)
    primitive = raw / "primitive_evidence.json"
    saved = primitive.read_bytes()
    primitive.write_bytes(saved + b" ")
    atomic_json(candidate, value)
    assert not e.replay(candidate)
    primitive.write_bytes(saved)
    changed = deepcopy(value)
    log = tmp_path / "child.stdout"
    log.write_bytes(b"actual completed stdout")
    changed["validation_receipts"].append(
        dict(passed=True, stdout_path=str(log), stdout_sha256=sha256_file(log))
    )
    write_changed(changed)
    log.write_bytes(b"stream changed")
    assert not e.replay(candidate)
    changed = deepcopy(value)
    atomic_json(primitive, {})
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(primitive):
            ref["sha256"] = sha256_file(primitive)
    write_changed(changed)
    assert not e.replay(candidate)
    primitive.write_bytes(saved)
    frozen = raw / "frozen_heads.json"
    frozen.chmod(0o644)
    saved_heads = frozen.read_bytes()
    altered = json.loads(saved_heads)
    altered["selected_simple_control"]["selected"] = "forged"
    atomic_json(frozen, altered)
    changed = deepcopy(value)
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(frozen):
            ref["sha256"] = sha256_file(frozen)
    write_changed(changed)
    assert not e.replay(candidate)
    frozen.write_bytes(saved_heads)
    candidate.write_text("{}")
    assert not e.replay(candidate)


def test_failed_primitive_manifest_stops_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8208: authenticated readiness cannot excuse a failed primitive shard."""
    monkeypatch.setattr(
        e.fit,
        "inputs",
        lambda *a, **kw: dict(
            checks=[
                dict(
                    passed=False,
                    check="input_sha256",
                    path="missing",
                    observed=None,
                    expected="sha256:required",
                )
            ],
            refs=[],
        ),
    )
    work = e.measure(e.ROOT, tmp_path / "failed_shard")
    assert not work["evidence"]
    assert work["checks"][-1]["observed"] is None
    assert (
        e.build(work, tmp_path / "failed_shard", [dict(passed=True)])["verdict_class"] == "blocked"
    )
