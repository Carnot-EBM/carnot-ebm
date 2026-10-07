"""REQ-REPORT-8234 / REQ-VERIFY-8234: controls cannot promote development benefit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting import decision_margin_methods_8234 as q
from carnot.reporting import decision_margin_runner_8234 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import decision_margin_8234 as n


def fixture(root):
    """Copy only declared operands so private tests cannot rewrite research outputs."""
    root.mkdir(parents=True, exist_ok=True)
    for ref in q.PROTOCOL_VALUE["source_artifact_hashes"]:
        source = Path(ref["path"])
        target = root / source.relative_to(q.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    for name in q.INPUTS:
        source, target = q.ROOT / name, root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return root


def cli(parent, *args):
    """Run real import and exit statements from outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-u", str(q.ROOT / q.CLI), *map(str, args)],
        cwd=parent,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_weight_and_derivative():
    """SCENARIO-VERIFY-8234-NUMERICS: native input margins never consume labels."""
    assert n.margin_weight(None, "accept") is None
    assert n.margin_weight(0.5, "reject") == 5
    assert n.margin_weight(1 / 12, "accept") == pytest.approx(1 + 4 * np.exp(-1 / 12 / 0.05))
    for p in [0, 0.1, 0.5, 1]:
        for permission in ["accept", "reject", "escalate"]:
            assert 1 <= n.margin_weight(p, permission) <= 5
    derivative = (
        n.margin_weight(0.2 + 1e-6, "reject") - n.margin_weight(0.2 - 1e-6, "reject")
    ) / 2e-6
    assert derivative == pytest.approx(80 * np.exp(-0.3 / 0.05), abs=1e-8)
    for p in [-1, 2, float("nan")]:
        with pytest.raises(ValueError, match="probability"):
            n.margin_weight(p, "accept")
    with pytest.raises(ValueError, match="permission"):
        n.margin_weight(0.2, "unknown")
    phi = np.array([[1.0, -2.0], [1.0, 0.3], [1.0, 1.2]])
    y, theta, w = np.array([0.0, 1.0, 0.0]), np.array([0.2, -0.3]), np.array([1.0, 3.0, 5.0])
    loss, gradient = n.objective(theta, phi, y, 0.01, w)
    numerical = [
        (
            n.objective(theta + np.eye(2)[i] * 1e-6, phi, y, 0.01, w)[0]
            - n.objective(theta - np.eye(2)[i] * 1e-6, phi, y, 0.01, w)[0]
        )
        / 2e-6
        for i in range(2)
    ]
    assert np.isfinite(loss) and gradient == pytest.approx(numerical, abs=1e-8)
    a, b = n.objective(theta, phi, y, 0.01, np.ones(3)), n.BASE_OBJECTIVE(theta, phi, y, 0.01)
    assert a[0] == b[0] and np.array_equal(a[1], b[1])
    for bad in [np.zeros(3), np.array([1.0, float("nan"), 1.0])]:
        with pytest.raises(ValueError, match="weights"):
            n.objective(theta, phi, y, 0.01, bad)


def test_protocol_and_primitives(tmp_path):
    """REQ-VERIFY-8234: authenticated roles and missing slots stay fixed."""
    root = fixture(tmp_path / "root")
    work = q.measure(root, tmp_path / "raw")
    assert not work["failures"], work["failures"]
    assert work["contract"]["activated"] and len(work["contract"]["contract_rows"]) == 14
    assert len(work["public_rows"]) == 320
    assert len(work["protocol"]["role_manifest"]["reserved"]) == 128
    assert all(r["y"] is None for r in work["public_rows"] if r["role"] == "reserved")
    assert work["protocol"]["H1"]["seed"] == 7128239
    assert work["protocol"]["H2"]["alpha"] == 0.025
    assert work["protocol"]["H2"]["retention"]["minimum_complete"] == 48
    assert q.reduce(work, [dict(name="owned", passed=True)])["margin_protocol_ready_score"] == 1
    assert q.reduce(work, [])["verdict_class"] == "disqualified"
    assert q.reduce(work, [dict(name="owned", passed=False)])["current_contract_ready_score"] == 0
    roles = deepcopy(work["protocol"]["role_manifest"])
    roles["calibration"].append(roles["head_fit"][0])
    with pytest.raises(ValueError, match="roles"):
        n.validate_roles(roles)
    public = deepcopy(work["public_rows"])
    public[0]["p0"] = None
    assert n.weights(public)[0]["weight"] is None
    assert n.weights([dict(public[1], y=0)]) == n.weights([dict(public[1], y=1)])
    assert not n.null_gate([0.0] * 128, work["protocol"]["H1"])


def test_fits_and_selection():
    """SCENARIO-VERIFY-8234-NUMERICS: all six actual fits share input geometry."""
    rows = [
        dict(
            unit_id=str(i),
            source_cluster_id=str(i),
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
    fitted = n.train(rows, roles)
    assert len(fitted["heads"]) == 6
    for arm in ["energy", "additive", "logistic"]:
        pair = [h for h in fitted["heads"] if h["basis"] == arm]
        assert pair[0]["geometry"] == pair[1]["geometry"]
        assert len(pair[0]["weights"]) == len(pair[1]["weights"]) == 17
        assert pair[0]["weights"] == pair[1]["weights"]
    scores = [
        dict(
            arm=a,
            unit_id=str(i),
            source_cluster_id=str(i),
            role="calibration",
            y=i % 2,
            p=0.5,
            action="escalate",
        )
        for a in n.COMPARATORS
        for i in range(32, 40)
    ]
    selected = n.select_comparator(scores, roles)
    assert selected["arm"] == n.COMPARATORS[0]
    assert selected["all_slot_cost"]["denominator"] == 8
    with pytest.raises(ValueError, match="calibration"):
        n.select_comparator([dict(scores[0], role="reserved")], roles)
    with pytest.raises(ValueError, match="fit_operands"):
        n.train([], roles)


def test_private_cli_and_tamper(tmp_path):
    """SCENARIO-REPORT-8234-CLI: actual publication and negative replay use fresh children."""
    root = fixture(tmp_path / "root")
    output = tmp_path / (q.NAME + ".json")
    result = cli(tmp_path, "--root", root, "--output", output, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["current_contract_ready_score"] == value["margin_protocol_ready_score"] == 1
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    for key in ["protocol_sha256", "reproducibility_checksum"]:
        changed = deepcopy(value)
        changed[key] = "sha256:" + "0" * 64
        atomic_json(output, changed)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    for key in [
        "rows",
        "current_contract_ready_score",
        "margin_protocol_ready_score",
        "frozen_roles",
    ]:
        changed = deepcopy(value)
        changed[key] = [] if isinstance(changed[key], list) else 9
        atomic_json(output, changed)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    atomic_json(output, value)
    primitive = Path(value["work_reference"]["path"])
    work = json.loads(primitive.read_bytes())
    saved = primitive.read_bytes()
    changed_work = deepcopy(work)
    changed_work["contract"]["activated"] = False
    atomic_json(primitive, changed_work)
    changed = deepcopy(value)
    changed["work_reference"]["sha256"] = sha256_file(primitive)
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(primitive):
            ref["sha256"] = sha256_file(primitive)
    changed["reproducibility_checksum"] = canonical_hash(
        [changed["work_reference"], changed["code_config_hashes"], q.PIN]
    )
    atomic_json(output, changed)
    with pytest.raises(ValueError, match="authority_drift"):
        runner.replay(output)
    changed_work = deepcopy(work)
    changed_work["failures"].append(q.failure(tmp_path / "invented", "exists", True, None))
    atomic_json(primitive, changed_work)
    changed = deepcopy(value)
    changed["work_reference"]["sha256"] = sha256_file(primitive)
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(primitive):
            ref["sha256"] = sha256_file(primitive)
    changed["reproducibility_checksum"] = canonical_hash(
        [changed["work_reference"], changed["code_config_hashes"], q.PIN]
    )
    atomic_json(output, changed)
    with pytest.raises(ValueError, match="source_gate_drift"):
        runner.replay(output)
    primitive.write_bytes(saved)
    work["public_rows"][1]["weight"] = 99
    atomic_json(primitive, work)
    changed = deepcopy(value)
    changed["work_reference"]["sha256"] = sha256_file(primitive)
    for ref in changed["raw_shard_hashes"]:
        if ref["path"] == str(primitive):
            ref["sha256"] = sha256_file(primitive)
    changed["reproducibility_checksum"] = canonical_hash(
        [changed["work_reference"], changed["code_config_hashes"], q.PIN]
    )
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--private-fixture").returncode == 1
    assert cli(tmp_path, "--date", "20261006").returncode == 2
    absent = tmp_path / "absent"
    absent.mkdir()
    blocked = tmp_path / "blocked" / (q.NAME + ".json")
    result = cli(tmp_path, "--root", absent, "--output", blocked, "--private-fixture")
    assert result.returncode == 0, result.stdout + result.stderr
    v = json.loads(blocked.read_bytes())
    assert v["verdict_class"] == "blocked" and v["gate_check_summary"]
    assert cli(tmp_path, "--cold-replay", blocked).returncode == 0


def test_sources_and_authorities(tmp_path):
    """REQ-REPORT-8234: external schema and missing authority failures retain operands."""
    root = fixture(tmp_path / "root")
    (root / q.ACTIVE).unlink()
    work = q.measure(root, tmp_path / "missing-active")
    assert not work["contract"]["activated"]
    assert q.reduce(work, [dict(passed=True)])["current_contract_ready_score"] == 0
    assert q.reduce(work, [dict(passed=True)])["margin_protocol_ready_score"] == 1
    root = fixture(root)
    (root / q.PROTOCOL).write_text("{}")
    work = q.measure(root, tmp_path / "changed-protocol")
    assert q.reduce(work, [dict(passed=True)])["verdict_class"] == "blocked"
    root = fixture(root)
    ref = q.PROTOCOL_VALUE["native_fit_rows"]
    (root / Path(ref["path"]).relative_to(q.ROOT)).unlink()
    work = q.measure(root, tmp_path / "missing-native")
    assert work["failures"][-1]["observed"] is None
    root = fixture(root)
    (root / q.DESIGN).write_text("missing contract")
    assert not q.measure(root, tmp_path / "bad-design")["contract"]["activated"]
    root = fixture(root)
    (root / q.HISTORY).unlink()
    assert (
        q.measure(root, tmp_path / "missing-context")["failures"][-1]["artifact_field"] == "exists"
    )


def test_owned_commands_and_main(tmp_path, monkeypatch):
    """REQ-VERIFY-8234-EXECUTION: real child checks distinguish failures from repository health."""
    plan = runner.commands(tmp_path / "plan")
    assert "--fail-under=100" in json.dumps(plan) and "--strict" in json.dumps(plan)
    root = fixture(tmp_path / "root")

    def bounded(private):
        return [
            dict(
                name="owned",
                argv=[sys.executable, "-c", "raise SystemExit(0)"],
                expected=0,
                deadline=5,
                scope="owned",
            ),
            dict(
                name="health",
                argv=[sys.executable, "-c", "raise SystemExit(3)"],
                expected=0,
                deadline=5,
                scope="repository_health",
            ),
        ]

    monkeypatch.setattr(runner, "commands", bounded)
    output = tmp_path / (q.NAME + ".json")
    assert runner.main(["--root", str(root), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["margin_protocol_ready_score"] == 1
    assert not value["repository_health"]["receipts"][0]["passed"]
    assert runner.replay(output)["passed"]
    Path(value["validation_receipts"][0]["stdout_path"]).write_text("tampered")
    with pytest.raises(ValueError, match="receipt"):
        runner.replay(output)
    monkeypatch.setattr(
        runner, "publish_primary", lambda *a: (_ for _ in ()).throw(ValueError("refused"))
    )
    assert runner.main(["--root", str(root), "--output", str(output)]) == 1


def test_numerical_failure_paths(monkeypatch):
    """SCENARIO-VERIFY-8234-NUMERICS: a failed solver cannot count as ready fitting."""
    from types import SimpleNamespace
    from carnot.verify import evidence_energy_8154 as base

    rows = [
        dict(
            unit_id=str(i),
            source_cluster_id=str(i),
            x=[float(i % 2)] * 16,
            y=i % 2,
            p0=0.2 if i % 2 else 0.8,
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
    bad = deepcopy(rows)
    for r in bad:
        r["p0"] = None
    with pytest.raises(ValueError, match="fit_operands"):
        n.train(bad, roles)
    bad = deepcopy(rows)
    bad[0]["x"] = [1.0] * 15
    for r in bad:
        r["x"] = [1.0] * 15
    with pytest.raises(ValueError, match="fit_operands"):
        n.train(bad, roles)
    calls = []

    def solved(phi, y, ridge, deadline):
        calls.append(1)
        return dict(weights=[0.0] * 17, converged=len(calls) != 21)

    monkeypatch.setattr(base, "solve", solved)
    with pytest.raises(ValueError, match="fit_nonconvergence"):
        n.train(rows, roles)
    monkeypatch.setattr(base, "solve", lambda *a: dict(weights=[0.0] * 17, converged=False))
    with pytest.raises(ValueError, match="fit_nonconvergence"):
        n.train(rows, roles)
    monkeypatch.setattr(base, "solve", lambda *a: dict(weights=[0.0] * 17, converged=True))
    monkeypatch.setattr(n, "minimize_scalar", lambda *a, **kw: SimpleNamespace(success=False))
    with pytest.raises(ValueError, match="temperature_nonconvergence"):
        n.train(rows, roles)
    with pytest.raises(ValueError, match="calibration"):
        n.select_comparator([], roles)
    scores = [
        dict(
            arm=a,
            unit_id=str(i),
            source_cluster_id=str(i),
            role="calibration",
            y=i % 2,
            p=None,
            action="escalate",
        )
        for a in n.COMPARATORS
        for i in range(32, 40)
    ]
    assert n.select_comparator(scores, roles)["arm"] == n.COMPARATORS[0]


def test_source_schema_and_replay_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8234-CLI: malformed source primitives and rehashed claims fail closed."""
    work = q.measure(q.ROOT, tmp_path / "inputs")
    refs = {r["path"]: r for r in work["refs"] if r["exists"]}
    read = lambda ref: json.loads(Path(refs[ref["path"]]["snapshot_path"]).read_bytes())
    original = q.primitives(read, work["protocol"])

    def changed_read(ref):
        v = read(ref)
        if ref == work["protocol"]["fit_measurement"]:
            v["evidence"]["roles"] = {}
        return v

    with pytest.raises(ValueError, match="source_schema"):
        q.primitives(changed_read, work["protocol"])

    def missing_read(ref):
        v = read(ref)
        if ref == work["protocol"]["reserved_primary"]:
            v["feature_rows"].pop()
        return v

    with pytest.raises(ValueError, match="source_schema"):
        q.primitives(missing_read, work["protocol"])

    def invalid_read(ref):
        v = read(ref)
        if ref == work["protocol"]["reserved_primary"]:
            v["feature_rows"][0]["source_cluster_id"] = "foreign"
        return v

    with pytest.raises(ValueError, match="source_schema"):
        q.primitives(invalid_read, work["protocol"])

    def feature_read(ref):
        v = read(ref)
        if ref == work["protocol"]["reserved_primary"]:
            v["feature_rows"][0]["x"] = [1.0] * 15
        return v

    with pytest.raises(ValueError, match="source_schema"):
        q.primitives(feature_read, work["protocol"])
    assert original
    monkeypatch.setattr(q, "primitives", lambda *a: (_ for _ in ()).throw(ValueError("schema")))
    assert (
        q.measure(q.ROOT, tmp_path / "schema")["failures"][-1]["artifact_field"] == "source_schema"
    )


def test_paired_native_identity_and_global_schema(tmp_path, monkeypatch):
    """REQ-VERIFY-8234: native probabilities and the frozen comparator retain identities."""
    work = q.measure(q.ROOT, tmp_path / "inputs")
    refs = {r["path"]: r for r in work["refs"] if r["exists"]}

    def changed_read(ref):
        v = json.loads(Path(refs[ref["path"]]["snapshot_path"]).read_bytes())
        if ref == work["protocol"]["native_reserved_measurement"]:
            v["plan"]["baseline"][0]["source_cluster_id"] = "foreign"
        return v

    with pytest.raises(ValueError, match="source_schema"):
        q.primitives(changed_read, work["protocol"])
    root = fixture(tmp_path / "root")
    protocol = deepcopy(q.PROTOCOL_VALUE)
    ref = protocol["v711_energy_global_measurement"]
    target = root / Path(ref["path"]).relative_to(q.ROOT)
    v = json.loads(target.read_bytes())
    v["models"][protocol["v711_energy_global_model_key"]]["kind"] = "foreign"
    atomic_json(target, v)
    for item in protocol["source_artifact_hashes"]:
        if item["path"] == ref["path"]:
            item["sha256"] = sha256_file(target)
    monkeypatch.setattr(q, "PROTOCOL_VALUE", protocol)
    assert (
        q.measure(root, tmp_path / "invalid-model")["failures"][-1]["artifact_field"]
        == "source_schema"
    )
