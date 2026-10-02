"""REQ-REPORT-8008: frozen roles, numerical conditioning and durable publication."""

import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from carnot import experiment_8008_v694_conditioned_energy_fit as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.multivariate_validation_7982 import fixture_data
from carnot.verify import sparse_energy_7996 as sparse


def fixture():
    """Artificial groups exercise the same contracts without natural evidence credit."""
    data = {k: v for k, v in fixture_data().items() if k in ("fit", "tune")}
    fitx = sparse.inputs(data["fit"])
    head = dict(
        arm="spline",
        seed=17,
        parameters=[0.0] * 109,
        decay_scale=1.0,
        temperature=1.0,
        scaler=sparse.fit_scaler(fitx),
    )
    development = {}
    for role, n in (("calibration", 64), ("stream", 8), ("retention", 8)):
        development[role] = [
            dict(
                data["fit"][i % len(data["fit"])],
                family_id=f"{role}-{i}",
                source_cluster_id=f"{role}-{i}",
            )
            for i in range(n)
        ]
    for r in development["calibration"][-2:]:
        r["q"] = None
    return dict(
        data=data,
        heads={"spline": [head]},
        development=development,
        references=[],
        role_manifests={},
    )


def test_conditioning_and_roundtrip(tmp_path):
    """SCENARIO-REPORT-8008-NUMERICS: objective gradients, local writes and stable endpoints."""
    data = fixture()["data"]
    x = sparse.inputs(data["fit"])
    y = np.array([r["y"] for r in data["fit"]], dtype=float)
    geo = e.geometry(x)
    matrix = e.design("conditioned_energy", x, geo)
    theta = np.zeros(matrix.shape[1])
    theta[0] = 0.3
    f, g, h = e.objective(theta, matrix, y, 0.001)
    assert np.isfinite(f) and np.linalg.eigvalsh(h).min() > 0
    assert e.audit(theta, x[0], 1, geo, 0.001)["passed"]
    head = e.optimize(matrix, y, 0.001, 17)
    assert head["converged"] and head["final_loss"] < head["initial_loss"]
    assert len(head["epochs"]) == head["optimizer_steps"] + 1
    fixed = e.optimize(matrix, y, 0.001, 17, fixed_steps=head["optimizer_steps"])
    assert fixed["optimizer_steps"] == head["optimizer_steps"]
    assert head["final_loss"] <= fixed["final_loss"]
    endpoints = x[:2].copy()
    endpoints[:, 0] = [0, 1]
    assert np.isfinite(e.design("conditioned_energy", endpoints, geo)).all()
    assert e.design("linear", x, geo).shape[1] == 10
    assert e.design("intercept_only", x, geo).shape[1] == 1
    path = tmp_path / "head.json"
    head.update(arm="conditioned_energy", geometry=geo)
    atomic_json(path, head)
    assert np.array_equal(
        e.probabilities(head, x), e.probabilities(json.loads(path.read_text()), x)
    )
    with pytest.raises(ValueError, match="fit_budget"):
        e.optimize(matrix, y, 0.001, 17, deadline=0)


def test_isolation_and_reduction(tmp_path):
    """SCENARIO-REPORT-8008-ISOLATION: reserved targets cannot change base fitting."""
    bundle = fixture()
    fitted = e.train(bundle["data"], bundle["heads"], tmp_path)
    assert all(
        h["linear_initialization"]["converged"]
        for h in fitted["heads"]
        if h["arm"] == "conditioned_energy"
    )
    assert all(
        h["converged"]
        for h in fitted["heads"]
        if h["arm"] not in ("fixed_step", "isotonic", "frozen_v693")
    )
    altered = copy.deepcopy(bundle["data"])
    altered["stream"] = []
    with pytest.raises(ValueError, match="role_roster"):
        e.train(altered, bundle["heads"], tmp_path)
    calibrated = e.calibrate(fitted, bundle["development"]["calibration"], tmp_path)
    assert calibrated["independent"] == 62
    measured = e.reduce_rows(fitted, calibrated, bundle)
    assert measured["decision_benefit"] == "unassessed"
    assert measured["rows"] and measured["sample_size_budget"]["independent"] < len(
        measured["rows"]
    )
    altered = copy.deepcopy(bundle)
    altered["development"]["stream"][0]["y"] ^= 1
    changed = e.reduce_rows(fitted, calibrated, altered)
    assert changed["rows"] != measured["rows"]
    assert fitted["heads"][0] == json.loads(Path(fitted["checkpoints"][0]["path"]).read_text())
    with pytest.raises(ValueError, match="calibration_support"):
        e.calibrate(fitted, bundle["development"]["stream"], tmp_path)


def test_real_cli(tmp_path):
    """SCENARIO-REPORT-8008-PUBLICATION: real private CLI publication and cold replay."""
    bundle = tmp_path / "fixture.json"
    atomic_json(bundle, fixture())
    output = tmp_path / "out" / (e.NAME + ".json")
    cli = str(e.ROOT / e.OWNED[-1])
    prefix = [str(e.ROOT / ".venv/bin/python"), "-u", cli]
    if os.environ.get("CARNOT_8008_COVERAGE_FILE"):
        prefix = [
            str(e.ROOT / ".venv/bin/coverage"),
            "run",
            "--parallel-mode",
            "--data-file=" + os.environ["CARNOT_8008_COVERAGE_FILE"],
            "--include=" + ",".join(str(e.ROOT / p) for p in e.OWNED),
            cli,
        ]
    run = subprocess.run(
        prefix + ["--fixture-input", str(bundle), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["conditioned_fit_ready_score"] == 0
    assert all(Path(r["path"]).is_relative_to(output.parent) for r in value["raw_shard_hashes"])
    assert e.terminal(output)["passed"]
    run = subprocess.run(
        prefix + ["--cold-replay", str(output)], capture_output=True, text=True, timeout=120
    )
    assert run.returncode == 0, run.stdout + run.stderr
    changed = copy.deepcopy(value)
    changed["gradient_checks"][0]["dense_sparse_update_error"] = 1.0
    atomic_json(output, changed)
    assert e.main(["--cold-replay", str(output)]) == 1
    atomic_json(output, value)
    value["rows"][0]["numerator"] += 1
    atomic_json(output, value)
    assert e.main(["--cold-replay", str(output)]) == 1
    assert e.main(["--fixture-input", str(tmp_path / "absent"), "--output", str(output)]) == 1
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(blocked)]) == 0
    assert json.loads(blocked.read_text())["conditioned_fit_ready_score"] == 0
    with pytest.raises(SystemExit):
        e.main(["--date", "20261001"])


def test_orchestration(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8008-PUBLICATION: real numerical work with isolated validation receipts."""
    source = tmp_path / "upstream.json"
    atomic_json(source, dict(methods_ready_score=1, task_id=e.UPSTREAM_TASK))
    bundle = fixture()
    monkeypatch.setattr(e, "load_bundle", lambda root, raw: (bundle, []))
    monkeypatch.setattr(e, "terminal", e.replay)

    def commands(root, specs, *, log_dir, **kwargs):
        scratch = Path(next(c.argv[-1] for c in specs if c.name == "coverage_json"))
        atomic_json(
            scratch,
            dict(
                files={
                    p: dict(summary=dict(num_statements=1, covered_lines=1, missing_lines=0))
                    for p in e.OWNED
                }
            ),
        )
        return [dict(name=c.name, passed=True, scope=c.scope, exit_code=0) for c in specs]

    monkeypatch.setattr(e, "run_commands", commands)
    output = tmp_path / "natural" / (e.NAME + ".json")
    assert e.main(["--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["conditioned_fit_ready_score"] == 1
    assert value["verdict_class"] == "null"
    value["conditioned_fit_ready_score"] = 1
    value["validation_receipts"] = []
    atomic_json(output, value)
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(output)
    monkeypatch.setattr(e, "run_commands", lambda *a, **kw: [dict(passed=False, scope="owned")])
    assert e.main(["--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    bundle["development"]["calibration"][0]["y"] = None
    assert e.main(["--output", str(output)]) == 0
    blocked_value = json.loads(output.read_text())
    assert blocked_value["verdict_class"] == "blocked"
    assert blocked_value["calibration_support_summary"]["known_target_groups"] == 61
    assert blocked_value["conditioned_fit_ready_score"] == 0
    assert blocked_value["gate_check_summary"][0]["observed"] is None
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    assert e.main(["--output", str(output)]) == 1


def test_prior_custody_and_development(tmp_path):
    """SCENARIO-REPORT-8008-ISOLATION: checked roles survive durable relocation."""
    bundle = fixture()
    path = tmp_path / "source" / "bundle.json"
    atomic_json(path, bundle)
    primary = tmp_path / "results" / "experiment_8007_v694_conditioning_diagnosis.json"
    value = e.prior.base([])
    value.update(methods_ready_score=1, checkpoints=dict(bundle=e.reference(path)))
    atomic_json(primary, value)
    loaded, failures = e.load_bundle(tmp_path, tmp_path / "raw")
    assert not failures and loaded["data"] == bundle["data"]
    assert e.development(loaded, "calibration") == bundle["development"]["calibration"]
    extra = tmp_path / "extra.json"
    atomic_json(extra, dict(value=1))
    loaded["nested"] = dict(checked=e.reference(extra), plain=["constant"])
    atomic_json(path, loaded)
    value["checkpoints"]["bundle"] = e.reference(path)
    atomic_json(primary, value)
    loaded, failures = e.load_bundle(tmp_path, tmp_path / "raw2")
    assert not failures and Path(loaded["nested"]["checked"]["path"]).is_relative_to(
        tmp_path / "raw2"
    )
    value.pop("methods_ready_score")
    atomic_json(primary, value)
    with pytest.raises(ValueError, match="upstream_contract"):
        e.load_bundle(tmp_path, tmp_path / "raw")
    public, labels = tmp_path / "public.json", tmp_path / "labels.json"
    row = bundle["development"]["stream"][0]
    atomic_json(public, dict(features=[dict(family_id=row["family_id"], values=row["features"])]))
    atomic_json(labels, dict(rows=[row]))
    real = dict(
        role_manifests=dict(
            public=dict(stream=e.reference(public)), evaluator=dict(stream=e.reference(labels))
        ),
        slots=[dict(row, role="stream")],
    )
    assert e.development(real, "stream")[0] == row


def test_backtracking_and_checkpoint_mismatch(tmp_path):
    """SCENARIO-REPORT-8008-NUMERICS: backtracking and altered embedded state fail closed."""
    matrix = np.array([[1.0], [1.0]])
    head = e.optimize(matrix, np.array([0.0, 1.0]), 0.0001, 17, initial=np.array([10.0]))
    assert head["converged"] and head["objective_evaluations"] > 2 * head["optimizer_steps"] + 1
    geo = e.geometry(sparse.inputs(fixture()["data"]["fit"]))
    h = dict(arm="intercept_only", geometry=geo, parameters=[0.0])
    path = tmp_path / "head.json"
    atomic_json(path, h)
    trial = tmp_path / "trial.json"
    atomic_json(trial, h)
    fitted = dict(
        heads=[dict(h, parameters=[1.0])],
        checkpoints=[e.reference(path)],
        convergence_rows=[dict(checkpoint=e.reference(trial))],
    )
    fitpath, bundle = tmp_path / "fitted.json", tmp_path / "bundle.json"
    atomic_json(fitpath, fitted)
    atomic_json(bundle, {})
    primary = tmp_path / "primary.json"
    atomic_json(
        primary,
        dict(
            raw_shard_hashes=[],
            code_config_hashes=[],
            checkpoints=dict(bundle=e.reference(bundle), fitted=e.reference(fitpath)),
        ),
    )
    with pytest.raises(ValueError, match="checkpoint_state"):
        e.replay(primary)
