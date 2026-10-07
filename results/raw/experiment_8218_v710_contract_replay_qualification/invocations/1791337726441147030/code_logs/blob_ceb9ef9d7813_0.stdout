"""REQ-VERIFY-8207 / REQ-REPORT-8207: qualify methods without reserved fitting."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import restricted_action_methods_8207 as e
from carnot.verify import restricted_action_rule_8207 as n


def cli(tmp_path: Path, *args: Any) -> subprocess.CompletedProcess[str]:
    """Private children exercise the real entry point without caller import paths."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private CLI", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print("after private CLI", result.returncode, flush=True)
    return result


def test_actions_and_identity() -> None:
    """SCENARIO-VERIFY-8207-ACTIONS: the mask holds for arbitrary probabilities."""
    for baseline in ["accept", "reject", "escalate"]:
        for p in [None, 0, 0.1, 0.5, 1, *np.linspace(0, 1, 1001)]:
            chosen = n.action(p, baseline)
            costs = {"reject": 1 - p, "escalate": 0.5} if p is not None else {}
            if p is not None and baseline == "accept":
                costs["accept"] = 5 * p
            assert chosen != "accept" or baseline == "accept"
            if costs:
                assert costs[chosen] == min(costs.values())
    assert n.action(0.1, "accept") == "escalate"
    assert n.action(0.5, "reject") == "escalate"
    for p in [-1, 2, math.nan, math.inf]:
        with pytest.raises(ValueError, match="probability"):
            n.action(p, "accept")
    with pytest.raises(ValueError, match="baseline_action"):
        n.action(0, "oracle")
    for z in [-1000, -4, 0, 4, 1000]:
        assert n.probability(0, -z, 1) == pytest.approx(e.old.n.logit_probability([0, z], 1))
    with pytest.raises(ValueError):
        n.probability(0, 1, 0)


def test_matched_training_and_tune_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-8207-CONTROLS: all heads use the same information and budget."""
    rows, roles, baseline = e.fixture()
    trained = n.train(rows, roles)
    assert len(trained) == 3
    assert {h["arm"] for h in trained} == {"energy", "additive", "logistic"}
    assert all(len(h["weights"]) == 17 and 0.25 <= h["temperature"] <= 4 for h in trained)
    assert all(h["fit_ids"] == trained[0]["fit_ids"] for h in trained)
    assert all(h["temperature_ids"] == trained[0]["temperature_ids"] for h in trained)
    query = e.query(rows[0])
    for head in trained:
        prediction = n.predict(head, query, baseline)
        assert prediction["action"] != "accept" or prediction["baseline_action"] == "accept"
        missing = dict(query, x=None)
        assert n.predict(head, missing, baseline)["action"] == "escalate"
        assert n.predict(head, dict(query, historical_x=None), baseline)["action"] == "escalate"
        with pytest.raises(ValueError, match="evaluator_label"):
            n.predict(head, dict(query, y=1), baseline)
    selected = n.select_simple(trained, rows, roles, baseline)
    changed = deepcopy(rows)
    tune_ids = {r["unit_id"] for r in roles["calibration"]}
    for row in changed:
        if row["unit_id"] in tune_ids:
            row["y"] = 1 - row["y"]
    other = n.train(changed, roles)
    assert [h["weights"] for h in trained] == [h["weights"] for h in other]
    assert [h["temperature"] for h in trained] == [h["temperature"] for h in other]
    assert selected["selected"] in {"additive", "logistic"}
    with pytest.raises(ValueError, match="arm"):
        n.design("unknown", np.zeros((1, 16)), trained[0]["geometry"])
    with pytest.raises(ValueError):
        n.train([], roles)
    monkeypatch.setattr(n.base, "solve", lambda *a: dict(converged=False))
    with pytest.raises(ValueError, match="fit_nonconvergence"):
        n.train(rows, roles)
    monkeypatch.undo()
    monkeypatch.setattr(n, "minimize_scalar", lambda *a, **k: SimpleNamespace(success=False))
    with pytest.raises(ValueError, match="temperature_nonconvergence"):
        n.train(rows, roles)


def test_protocol_fixtures_and_blocks(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-8207-HYPOTHESES: protocol and fixtures cannot claim H1."""
    work = e.measure(e.ROOT, tmp_path / "good", fixture=True)
    assert not any(not c["passed"] for c in work["checks"])
    assert len(work["protocol"]["role_manifest"]["reserved"]) == 128
    assert work["protocol"]["H1"]["alpha"] == work["protocol"]["H2"]["alpha"] == 0.025
    assert work["diagnostics"]["empty"]["action"] == "escalate"
    assert work["diagnostics"]["constant"]["maximum_probability_spread"] < 1e-12
    assert (
        work["diagnostics"]["shuffled"]["label_order_sha256"]
        != work["diagnostics"]["natural"]["label_order_sha256"]
    )
    receipts = [dict(name="owned", passed=True)]
    value = e.build(work, tmp_path / "good", receipts, fixture=True)
    assert value["action_protocol_ready_score"] == 1 and not value["H1"]["measured_here"]
    assert e.build(work, tmp_path / "good", [dict(passed=False)])["verdict_class"] == "disqualified"
    blocked = e.measure(tmp_path / "absent", tmp_path / "blocked")
    assert e.build(blocked, tmp_path / "blocked", receipts)["verdict_class"] == "blocked"
    assert blocked["checks"][-1]["observed"] is None
    assert (
        e.measure(e.ROOT, tmp_path / "mutation", fixture=True, mutation="source")["checks"][-1][
            "passed"
        ]
        is False
    )


def test_real_cli_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8207-CLI: real private publication, blocks and tamper checks."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--fixture-output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "circular_positive"
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    original = deepcopy(value)
    value["action_protocol_ready_score"] = 9
    atomic_json(output, value)
    assert not e.replay(output)
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, original)
    ref = original["raw_shard_hashes"][0]
    evidence_path = Path(ref["path"])
    saved = evidence_path.read_bytes()
    evidence_path.write_bytes(saved + b" ")
    assert not e.replay(output)
    evidence_path.write_bytes(saved)
    assert e.replay(output)
    primitive = next(
        r for r in original["raw_shard_hashes"] if Path(r["path"]).name == "fixture_primitives.json"
    )
    primitive_path = Path(primitive["path"])
    primitive_bytes = primitive_path.read_bytes()
    atomic_json(primitive_path, {})
    forged = deepcopy(original)
    from carnot.reporting.current_work_receipt import sha256_file

    for ref in forged["raw_shard_hashes"]:
        if ref["path"] == str(primitive_path):
            ref["sha256"] = sha256_file(primitive_path)
    forged["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in forged.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, forged)
    assert not e.replay(output)
    primitive_path.write_bytes(primitive_bytes)
    forged = deepcopy(original)
    log = tmp_path / "stream.stdout"
    log.write_bytes(b"original stdout")
    forged["validation_receipts"].append(
        dict(passed=True, stdout_path=str(log), stdout_sha256=sha256_file(log))
    )
    forged["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in forged.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, forged)
    log.write_bytes(b"changed stdout")
    assert not e.replay(output)
    atomic_json(output, original)
    assert cli(tmp_path, "--date", "wrong").returncode == 2
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results" / output.name).returncode == 2
    blocked = tmp_path / "block" / output.name
    result = cli(
        tmp_path, "--fixture-output", blocked, "--root", tmp_path / "absent", "--mutation", "source"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(blocked.read_bytes())["verdict_class"] == "blocked"
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing.json").returncode == 1


def test_authentic_registration_and_validation_plan(tmp_path: Path) -> None:
    """REQ-REPORT-8207: real historical custody never turns into current model calls."""
    work = e.measure(e.ROOT, tmp_path / "natural")
    assert all(c["passed"] for c in work["checks"]), work["checks"]
    value = e.build(work, tmp_path / "natural", [dict(passed=True)])
    assert value["verdict_class"] == "null" and value["action_protocol_ready_score"] == 1
    assert len(value["rows"]) == value["intended_count"] == 128
    assert value["MODEL_SPECS"] == [] and value["independent_generalization_score"] == 0
    for row in value["rows"]:
        assert row["evaluation_status"] == "pending_exp8209"
    plan = e.manifest(tmp_path, tmp_path / "candidate.json")
    assert plan["repository_health"]["argv"][-2:] == ["tests/python", "-q"]
    assert any("--files" in s["argv"] for s in plan["commands"])
    assert any("--fail-under=100" in s["argv"] for s in plan["commands"])
    spec = dict(
        name="actual_child",
        argv=[str(e.ROOT / ".venv/bin/python"), "-c", "print('actual child')"],
        deadline_s=10,
        expected_exit=0,
    )
    receipt = e.run_check(e.ROOT, spec, tmp_path, tmp_path / "logs")
    assert receipt["passed"] and Path(receipt["stdout_path"]).read_text() == "actual child\n"


def test_owned_failure_and_schema_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8207: owned failure differs from an absent external schema."""

    def failed_fit() -> Any:
        raise ValueError("owned_solver_failure")

    monkeypatch.setattr(e, "diagnostics", failed_fit)
    work = e.measure(e.ROOT, tmp_path / "owned", fixture=True)
    assert work["owned_failure"] == "owned_solver_failure"
    assert e.build(work, tmp_path / "owned", [dict(passed=True)])["verdict_class"] == "disqualified"
    monkeypatch.undo()

    def malformed(*args: Any) -> Any:
        raise TypeError("external_schema_invalid")

    monkeypatch.setattr(e.old.fit, "bind", malformed)
    work = e.measure(e.ROOT, tmp_path / "schema", fixture=True)
    assert work["checks"][-1]["observed"] == "external_schema_invalid"
    assert e.build(work, tmp_path / "schema", [dict(passed=True)])["verdict_class"] == "blocked"


def test_real_measurement_worker_custody(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8207-CLI: a running worker cannot seal its parent's open logs."""
    raw = tmp_path / "worker"
    worker = raw / "measurement.json"
    receipt = e.run_check(
        e.ROOT,
        dict(
            name="measurement",
            argv=[
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                str(e.ROOT / e.CLI),
                "--worker-output",
                str(worker),
            ],
            expected_exit=0,
            deadline_s=60,
        ),
        tmp_path,
        raw / "logs",
    )
    assert receipt["passed"]
    work = json.loads(worker.read_bytes())
    atomic_json(raw / "validation_receipts.json", dict(rows=[receipt]))
    value = e.build(work, raw, [receipt])
    candidate = tmp_path / "worker_candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate)
    assert all(
        not Path(ref["path"]).is_relative_to(raw / "logs") for ref in value["raw_shard_hashes"]
    )
