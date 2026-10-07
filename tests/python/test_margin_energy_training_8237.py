"""REQ-VERIFY-8237 / REQ-REPORT-8237: fit readiness cannot imply benefit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.verify import margin_energy_training_8237 as n
from carnot.reporting import margin_energy_training_8237 as q
from carnot.reporting import margin_energy_runner_8237 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from test_decision_margin_methods_8234 import fixture


def inputs():
    """Use disjoint private roles so a fixture never impersonates natural evidence."""
    rows = [
        dict(
            unit_id=str(i),
            source_cluster_id=str(i),
            role="head_fit" if i < 24 else "temperature_fit" if i < 32 else "calibration",
            x=[float(i % 2)] * 16,
            y=i % 2,
            p0=0.5,
            baseline_action="escalate",
        )
        for i in range(40)
    ]
    roles = {
        k: [dict(unit_id=str(i), source_cluster_id=str(i)) for i in ids]
        for k, ids in [
            ("head_fit", range(24)),
            ("temperature_fit", range(24, 32)),
            ("calibration", range(32, 40)),
            ("reserved", range(40, 168)),
        ]
    }
    return rows, roles


def cli(parent, *args):
    """Real script-path children must resolve imports outside this checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(q.ROOT / q.CLI), *map(str, args)],
        cwd=parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_numerics_fitting_and_scoring(tmp_path):
    """SCENARIO-VERIFY-8237-FIT: uniform weights and zero benefit are valid."""
    assert n.numerical_checks()["uniform_error"] <= 1e-10
    rows, roles = inputs()
    fitted = n.fit(rows, roles, tmp_path / "checkpoints")
    assert len(fitted["heads"]) == 6 and len(fitted["fit_fold_rows"]) == 120
    assert n.signature(n.fit(rows, roles, tmp_path / "checkpoints")) == n.signature(fitted)
    for receipt in fitted["optimizer_receipts"]:
        assert receipt["converged"] and len(receipt["final_gradient"]) == 17
        assert receipt["final_loss"] <= receipt["initial_loss"] + 1e-10
        assert receipt["parameter_hashes"]["after"] == canonical_hash(receipt["weights"])
    for record in fitted["fit_fold_rows"]:
        assert set(record["fit_source_ids"]).isdisjoint(record["held_source_ids"])
        assert len(record["source_weight_rows"]) == 24
    for head in fitted["heads"]:
        assert len(head["weights"]) == 17
        pred = n.score(head, dict(x=rows[0]["x"], p0=0.5, baseline_action="reject"))
        assert pred["action"] != "accept" and pred["energy_probability_error"] <= 1e-10
        assert (
            n.score(head, dict(x=None, p0=None, baseline_action="accept"))["action"] == "escalate"
        )
        with pytest.raises(ValueError, match="evaluator_label"):
            n.score(head, dict(x=rows[0]["x"], p0=0.5, baseline_action="accept", y=0))
    bad = deepcopy(rows)
    bad[0]["role"] = "reserved"
    with pytest.raises(ValueError, match="future_target"):
        n.fit(bad, roles, tmp_path / "bad")
    path = next((tmp_path / "checkpoints").glob("*.json"))
    saved = json.loads(path.read_bytes())
    saved["input_sha256"] = "changed"
    atomic_json(path, saved)
    with pytest.raises(ValueError, match="checkpoint_input"):
        n.fit(rows, roles, tmp_path / "checkpoints")


def test_reduction_requires_checks_and_keeps_slots(tmp_path):
    """REQ-REPORT-8237: mechanics grant readiness with no science claim."""
    work = q.measure(q.ROOT, tmp_path / "raw")
    assert not work["failures"], work["failures"]
    value = q.reduce(work, [dict(passed=True)])
    assert value["margin_fit_ready_score"] == 1
    assert value["verdict_class"] == "null"
    assert len(value["rows"]) == 192 * 6
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["MODEL_SPECS"] == []
    assert value["comparator_name"] in q.methods.n.COMPARATORS
    assert len(value["equally_weighted_simple_comparisons"]) == 2
    assert q.reduce(work, [])["verdict_class"] == "disqualified"
    assert q.reduce(work, [dict(passed=False)])["margin_fit_ready_score"] == 0
    blocked = q.measure(tmp_path / "missing", tmp_path / "blocked")
    assert q.reduce(blocked, [dict(passed=True)])["verdict_class"] == "blocked"


def test_private_cli_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8237-CLI: fresh children reject rehashed changes."""
    root = fixture(tmp_path / "root")
    target = root / q.METHODS
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((q.ROOT / q.METHODS).read_bytes())
    output = tmp_path / (q.NAME + ".json")
    result = cli(tmp_path, "--root", root, "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["margin_fit_ready_score"] == 1
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    for field in ["margin_fit_ready_score", "comparator_name", "reproducibility_checksum"]:
        changed = deepcopy(value)
        changed[field] = "tampered"
        atomic_json(output, changed)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    work_path = Path(value["work_reference"]["path"])
    saved = work_path.read_bytes()
    work = json.loads(saved)
    work["fitted"]["heads"][0]["weights"][0] += 0.01
    atomic_json(work_path, work)
    changed = deepcopy(value)
    changed["work_reference"]["sha256"] = sha256_file(work_path)
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(work_path):
            ref["sha256"] = sha256_file(work_path)
    changed["reproducibility_checksum"] = canonical_hash(
        [changed["work_reference"], changed["code_config_hashes"], q.methods.PIN]
    )
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    work_path.write_bytes(saved)
    atomic_json(output, value)
    Path(value["validation_receipts"][0]["stdout_path"]).write_text("tampered log")
    with pytest.raises(ValueError, match="receipt"):
        runner.replay(output)
    assert cli(tmp_path, "--private-fixture").returncode == 1
    assert cli(tmp_path, "--date", "20261006").returncode == 2
    blocked = tmp_path / "blocked" / (q.NAME + ".json")
    assert (
        cli(
            tmp_path, "--root", tmp_path / "absent", "--output", blocked, "--private-fixture"
        ).returncode
        == 0
    )
    assert json.loads(blocked.read_bytes())["verdict_class"] == "blocked"
    assert cli(tmp_path, "--cold-replay", blocked).returncode == 0


def test_commands_and_owned_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8237: coverage includes real children; health is separate."""
    plan = runner.commands(tmp_path / "plan")
    assert "--fail-under=100" in json.dumps(plan) and "--strict" in json.dumps(plan)
    assert any(p["scope"] == "repository_health" for p in plan)
    monkeypatch.setattr(
        runner,
        "commands",
        lambda p: [
            dict(
                name="owned_failure",
                argv=[sys.executable, "-c", "raise SystemExit(2)"],
                expected=0,
                deadline=5,
                scope="owned",
            ),
            dict(
                name="health_failure",
                argv=[sys.executable, "-c", "raise SystemExit(3)"],
                expected=0,
                deadline=5,
                scope="repository_health",
            ),
        ],
    )
    output = tmp_path / (q.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and value["margin_fit_ready_score"] == 0
    assert not value["repository_health"]["receipts"][0]["passed"]
    monkeypatch.setattr(
        runner, "publish_primary", lambda *a: (_ for _ in ()).throw(ValueError("refused"))
    )
    assert runner.main(["--output", str(output)]) == 1


def test_failed_numerics_and_optimizer(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8237-FIT: owned numerical failures cannot grant readiness."""
    monkeypatch.setattr(n, "numerical_checks", lambda: dict(passed=False, uniform_error=1.0))
    work = q.measure(q.ROOT, tmp_path / "numerics")
    assert work["owned_errors"] == ["numerical_checks"]
    assert q.reduce(work, [dict(passed=True)])["verdict_class"] == "disqualified"
    monkeypatch.setattr(n, "numerical_checks", lambda: dict(passed=True))
    monkeypatch.setattr(n, "fit", lambda *a: (_ for _ in ()).throw(TimeoutError("actual_budget")))
    work = q.measure(q.ROOT, tmp_path / "timeout")
    assert work["owned_errors"] == ["actual_budget"]
    assert q.reduce(work, [dict(passed=True)])["margin_fit_ready_score"] == 0


def test_changed_methods_are_blocked(tmp_path, monkeypatch):
    """REQ-REPORT-8237: missing upstream readiness is an exact external operand."""
    root = fixture(tmp_path / "root")
    target = root / q.METHODS
    target.parent.mkdir(parents=True, exist_ok=True)
    methods = json.loads((q.ROOT / q.METHODS).read_bytes())
    methods["margin_protocol_ready_score"] = 0
    atomic_json(target, methods)
    monkeypatch.setattr(q, "METHODS_PIN", sha256_file(target))
    work = q.measure(root, tmp_path / "raw")
    assert work["failures"][-1]["artifact_field"] == "margin_protocol_ready_score"
    value = q.reduce(work, [dict(passed=True)])
    assert value["verdict_class"] == "blocked" and value["margin_fit_ready_score"] == 0
    assert value["energy_probability_error"] is None
    assert len(value["rows"]) == value["intended_count"] == 1152
    receipt = runner.x.child(
        "blocked_upstream", [sys.executable, "-c", "print('actual')"], tmp_path / "logs", deadline=5
    )
    value = q.reduce(work, [receipt])
    work_path = tmp_path / "work.json"
    atomic_json(work_path, work)
    value.update(
        work_reference=dict(path=str(work_path), sha256=sha256_file(work_path)), raw_shard_hashes=[]
    )
    value["reproducibility_checksum"] = canonical_hash(
        [value["work_reference"], value["code_config_hashes"], q.methods.PIN]
    )
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert runner.replay(candidate)["passed"]


def test_required_e2e_manifest(tmp_path):
    """REQ-REPORT-8237: all three applicable private E2E checks are frozen."""
    names = {r["name"] for r in runner.commands(tmp_path / "plan")}
    assert {"private_E2E015", "private_E2E019", "private_E2E021"} <= names


def test_public_api_rejects_invalid_values(tmp_path):
    """SCENARIO-VERIFY-8237-FIT: malformed public inputs cannot become predictions."""
    rows, roles = inputs()
    head = n.fit(rows, roles, tmp_path / "fit")["heads"][0]
    for query in [
        dict(x=[1.0] * 15, p0=0.2, baseline_action="accept"),
        dict(x=[float("nan")] * 16, p0=0.2, baseline_action="accept"),
        dict(x=[1.0] * 16, p0=2.0, baseline_action="accept"),
    ]:
        with pytest.raises(ValueError, match="public_input"):
            n.score(head, query)
    assert head["temperature_receipt"]["objective"] == "unweighted_log_loss"


def test_authenticated_replay_failure_paths(tmp_path):
    """SCENARIO-VERIFY-8237-REPLAY: rehashed primitives must still reproduce fitting."""
    work = q.measure(q.ROOT, tmp_path / "raw")
    receipt = runner.x.child(
        "actual", [sys.executable, "-c", 'print("actual child")'], tmp_path / "logs", deadline=5
    )
    value = q.reduce(work, [receipt])
    work_path = tmp_path / "work.json"
    candidate = tmp_path / "candidate.json"
    value.update(work_reference=dict(path=str(work_path), sha256="pending"), raw_shard_hashes=[])

    def save(changed):
        atomic_json(work_path, changed)
        v = deepcopy(value)
        v["work_reference"]["sha256"] = sha256_file(work_path)
        v["reproducibility_checksum"] = canonical_hash(
            [v["work_reference"], v["code_config_hashes"], q.methods.PIN]
        )
        atomic_json(candidate, v)
        return v

    for name, message in [
        ("protocol", "protocol_drift"),
        ("public_rows", "source_primitive_drift"),
        ("failures", "source_gate_drift"),
        ("global_probabilities", "global_head_drift"),
    ]:
        changed = deepcopy(work)
        if name == "protocol":
            changed[name]["frozen_date"] = "tampered"
        elif name == "public_rows":
            changed[name][0]["weight"] = 99
        elif name == "failures":
            changed[name].append(dict(invented=True))
        else:
            changed[name][next(iter(changed[name]))] = 0.12345
        save(changed)
        with pytest.raises(ValueError, match=message):
            runner.replay(candidate)
    v = save(work)
    v["trained_heads_sha256"] = "changed"
    atomic_json(candidate, v)
    with pytest.raises(ValueError, match="head_bundle_drift"):
        runner.replay(candidate)
    bundle_path = Path(work["trained_heads_path"])
    bundle = json.loads(bundle_path.read_bytes())
    bundle["heads"][0]["weights"][0] += 0.1
    atomic_json(bundle_path, bundle)
    v = save(work)
    v["trained_heads_sha256"] = sha256_file(bundle_path)
    atomic_json(candidate, v)
    with pytest.raises(ValueError, match="head_bundle_drift"):
        runner.replay(candidate)
