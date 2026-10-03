"""REQ-REPORT-8073: fixed representation, isolated labels and checked publication."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from carnot import experiment_8073_v699_interaction_energy_fit as m


def artificial():
    """Private source groups provide both classes without scientific credit."""
    rng = np.random.default_rng(8073)
    return {
        role: [
            dict(
                unit=f"{role}/{i}",
                source=f"{role}-{i}",
                family_id=f"{role}-{i}",
                role=role,
                slot=i,
                q=float(rng.uniform(0.1, 0.9)),
                features=rng.uniform(size=8).tolist(),
                y=i % 2,
                status="completed",
                exclusion_reason=None,
            )
            for i in range(n)
        ]
        for role, n in (("fit", 64), ("tune", 32))
    }


def test_fixed_basis_and_fold_geometry():
    """SCENARIO-REPORT-8073-BASIS: only three preselected centered terms enter."""
    x = m.sparse.inputs(artificial()["fit"])
    geo = m.geometry(x)
    base = m.design("additive", x, geo)
    full = m.design("interaction", x, geo)
    assert base.shape == (64, 110) and full.shape == (64, 113)
    assert np.array_equal(base, full[:, :110])
    assert np.max(np.abs(full[:, -3:].mean(axis=0))) < 1e-12
    assert m.design("intercept", x, geo).shape[1] == 1
    assert m.design("scalar_affine", x, geo).shape[1] == 2
    assert m.design("linear", x, geo).shape[1] == 10
    assert m.choose_ridge({0.0001: 1.0, 0.001: 1.0, 0.01: 1.0, 0.1: 1.0, 1.0: 1.0}) == 1.0
    with pytest.raises(ValueError, match="feature_columns"):
        m.design("interaction", x, dict(geo, feature_names=list(reversed(geo["feature_names"]))))
    with pytest.raises(ValueError, match="arm"):
        m.design("unknown", x, geo)


def test_numerical_fit_and_mutations(tmp_path):
    """SCENARIO-REPORT-8073-SEAL: fit-only CV and tune-only calibration freeze once."""
    data = artificial()
    fit = m.train(data, tmp_path)
    assert fit["ready"] and len(fit["heads"]) == 5
    assert len(fit["convergence_rows"]) == 105
    assert fit["equivalent_logistic_parity"]["maximum_error"] < 1e-10
    assert fit["feature_permutation_check"]["passed"]
    assert all(r["fit_roles"] == ["fit"] for r in fit["convergence_rows"])
    assert set(fit["fold_assignment"].values()) == {0, 1, 2, 3}
    assert all(r["passed"] for r in fit["gradient_checks"])
    rows = m.measure(data, fit)
    assert len(rows) == 480
    assert all(r["denominator"] == 1 for r in rows)
    for head, ref in zip(fit["heads"], fit["head_checkpoints"], strict=True):
        x = m.sparse.inputs(data["fit"])
        assert np.array_equal(
            m.predict(head, x), m.predict(json.loads(Path(ref["path"]).read_text()), x)
        )
    data["evaluation"] = deepcopy(data["fit"])
    with pytest.raises(ValueError, match="role_roster"):
        m.train(data, tmp_path)
    with pytest.raises(ValueError, match="fit_budget"):
        m.solve(
            "linear",
            x,
            np.array([r["y"] for r in data["fit"]]),
            m.geometry(x),
            0.1,
            time.monotonic() - 1,
        )


def test_authenticated_sources_and_support(tmp_path):
    """REQ-REPORT-8073: every source/response joins; reserved outcomes stay closed."""
    source = m.load_sources(m.ROOT, tmp_path / "good")
    assert not source["failures"]
    assert len(source["data"]["fit"]) == 64 and len(source["data"]["tune"]) == 32
    assert all(r["role"] in {"fit", "tune"} for r in source["label_access_events"])
    assert source["support"]["fit"]["independent"] >= 48
    assert source["support"]["tune"]["independent"] >= 24
    assert m.load_sources(tmp_path / "absent", tmp_path / "missing")["failures"]
    mutated = m.load_sources(m.ROOT, tmp_path / "mutated", mutate=True)
    assert mutated["failures"] and mutated["failures"][0]["field"] == "source_cluster_overlap"


@pytest.mark.parametrize("route", ["success", "blocked", "mutation"])
def test_private_cli_and_cold_replay(tmp_path, route):
    """SCENARIO-REPORT-8073-TERMINAL: private real CLI routes exit outside checkout."""
    output = tmp_path / route / (m.NAME + ".json")
    cmd = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    config = os.environ.get("CARNOT_8073_COVERAGE_CONFIG")
    if config:
        cmd = [sys.executable, "-m", "coverage", "run", "--rcfile=" + config, *cmd[2:]]
    args = ["--fixture-output", str(output)]
    if route == "blocked":
        args += ["--root", str(tmp_path / "absent")]
    if route == "mutation":
        args += ["--mutate"]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    print("8073 subprocess before " + route, flush=True)
    p = subprocess.run(
        cmd + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    print("8073 subprocess after " + route + " exit=" + str(p.returncode), flush=True)
    assert p.returncode == 0, p.stdout + p.stderr
    v = json.loads(output.read_text())
    assert v["verdict_class"] == ("null" if route == "success" else "blocked")
    assert m.replay(output)
    p = subprocess.run(
        cmd + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    v["completed_count"] += 1
    m.atomic_json(output, v)
    assert m.main(["--cold-replay", str(output)]) == 1
    assert m.main(["--cold-replay", str(tmp_path / "absent.json")]) == 1


def test_owned_receipts_and_terminal_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8073-TERMINAL: private receipt controls exercise fail-closed gates."""
    output = tmp_path / "good" / (m.NAME + ".json")
    cov = {p: {"summary": {"num_statements": 1, "missing_lines": 0}} for p in m.OWNED}

    def checks(root, spec, private, durable):
        if spec["name"] == "fitting_child_normal_exit":
            raw = Path(spec["argv"][-1])
            source = m.load_sources(m.ROOT, raw)
            fitted = m.train(source["data"], raw)
            m.atomic_json(raw / "fitting.json", dict(source=source, fitted=fitted))
        m.atomic_json(private / "coverage.json", {"files": cov})
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("private exact-receipt control; no scientific credit\n")
        return dict(spec, passed=True, log_path=str(log), log_sha256=m.sha256_file(log))

    monkeypatch.setattr(m, "run_check", checks)
    monkeypatch.setattr(m, "terminal", lambda p: {"passed": m.replay(p)})
    assert m.main(["--output", str(output)]) == 0
    original = json.loads(output.read_text())
    assert original["interaction_fit_ready_score"] == 1
    assert set(original) - {"field_principles"} <= set(original["field_principles"])
    assert m.main(["--output", str(output)]) == 1
    raw = Path(original["terminal_validation_sidecar_path"]).parent
    work = json.loads((raw / "work.json").read_text())
    validation = json.loads((raw / "validation.json").read_text())
    assert m.build(work, raw, validation["receipts"], {}, False)["verdict_class"] == "disqualified"
    failed = deepcopy(validation["receipts"])
    failed[0]["passed"] = False
    assert m.build(work, raw, failed, cov, False)["interaction_fit_ready_score"] == 0
    for key in ("code_config_hashes", "validation_receipts"):
        changed = deepcopy(original)
        if key == "code_config_hashes":
            changed[key][m.MODULE] = "wrong"
        else:
            changed[key][0]["log_sha256"] = "wrong"
        m.atomic_json(output, changed)
        assert not m.replay(output)
    m.atomic_json(output, original)
    config = json.loads((raw / "configuration.json").read_text())
    changed = deepcopy(config)
    changed["config"]["inputs"] = 10
    m.atomic_json(raw / "configuration.json", changed)
    mutated = deepcopy(original)
    for ref in mutated["raw_shard_hashes"]:
        if ref["path"] == str(raw / "configuration.json"):
            ref["sha256"] = m.sha256_file(raw / "configuration.json")
    m.atomic_json(output, mutated)
    assert not m.replay(output)
    m.atomic_json(raw / "configuration.json", config)
    m.atomic_json(output, original)
    for mutation in ("geometry", "head"):
        changed = deepcopy(work)
        changed["fitted"]["heads"][0]["geometry"]["logit_center"] += (
            1 if mutation == "geometry" else 0
        )
        if mutation == "head":
            changed["fitted"]["heads"][0]["parameters"][0] += 1
        m.atomic_json(raw / "work.json", changed)
        mutated = deepcopy(original)
        mutated["raw_shard_hashes"][0] = m.reference(raw / "work.json")
        m.atomic_json(output, mutated)
        assert not m.replay(output)
    m.atomic_json(raw / "work.json", work)
    m.atomic_json(output, original)
    assert m.replay(output)
    predictor = m.predict
    calls = []

    def unstable(head, x, **kwargs):
        calls.append(1)
        return predictor(head, x, **kwargs) + (0.01 if len(calls) % 2 else 0)

    monkeypatch.setattr(m, "predict", unstable)
    assert not m.replay(output)
    monkeypatch.setattr(m, "predict", predictor)

    def failed_child(root, spec, private, durable):
        receipt = checks(root, dict(spec, name="fake_child"), private, durable)
        receipt.update(name=spec["name"], passed=False)
        return receipt

    monkeypatch.setattr(m, "run_check", failed_child)
    bad = tmp_path / "failed" / (m.NAME + ".json")
    assert m.main(["--output", str(bad)]) == 0
    assert json.loads(bad.read_text())["verdict_class"] == "disqualified"
    monkeypatch.setattr(m, "terminal", lambda p: {"passed": False})
    assert m.main(["--fixture-output", str(tmp_path / "reject" / (m.NAME + ".json"))]) == 1


def test_fitting_exception_and_role_leakage(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8073-SEAL: numerical failure cannot become readiness."""
    data = artificial()
    data["tune"][0]["source"] = data["fit"][0]["source"]
    with pytest.raises(ValueError, match="held_out_role_exclusion"):
        m.train(data, tmp_path)
    monkeypatch.setattr(m, "train", lambda *a: (_ for _ in ()).throw(ValueError("fit_budget")))
    assert m.main(["--fit-child", str(tmp_path / "failed")]) == 0
    value = json.loads((tmp_path / "failed/fitting.json").read_text())
    assert value["owned_fitting_error"] == "fit_budget" and value["fitted"] == {}


def test_terminal_binding_operand(tmp_path, monkeypatch):
    """REQ-REPORT-8073: a stale terminal binding reports its exact failed contract."""
    monkeypatch.setattr(
        m, "read_bound_sidecar", lambda *a: (_ for _ in ()).throw(ValueError("stale_primary_hash"))
    )
    failed = m.load_sources(m.ROOT, tmp_path)
    assert failed["failures"][0]["field"] == "authenticated_terminal"
    assert failed["failures"][0]["observed"] == "stale_primary_hash"
