"""REQ-REPORT-8020: qualified CPU fitting and sealed prediction custody."""

import copy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from carnot import experiment_8020_v695_qualified_energy_fit as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.multivariate_validation_7982 import fixture_data


def fixture(root):
    """Private artificial targets exercise the registered role support floors."""
    data = fixture_data()
    public, evaluator = {}, {}
    for role, n in (
        ("fit", 256),
        ("tune", 64),
        ("calibration", 64),
        ("stream", 8),
        ("retention", 8),
    ):
        rows = [
            dict(
                data["fit"][i % len(data["fit"])],
                family_id=f"{role}-{i}",
                source_cluster_id=f"{role}-{i}",
                role=role,
                slot=i,
                public_eligible=True,
                exclusion_reason=None,
            )
            for i in range(n)
        ]
        pub = [dict(r) for r in rows]
        for r in pub:
            r.pop("y")
        if role == "calibration":
            rows[-1]["y"] = None
        labels = [
            dict(
                r,
                eligible_y=r["y"],
                target_eligible=r["y"] is not None,
                eligibility_reason="unknown_target" if r["y"] is None else None,
            )
            for r in rows
        ]
        pp, ep = root / f"public-{role}.json", root / f"labels-{role}.json"
        atomic_json(pp, dict(role=role, rows=pub))
        atomic_json(ep, dict(role=role, rows=labels))
        public[role], evaluator[role] = e.reference(pp), e.reference(ep)
    value = dict(
        experiment_id=8019,
        task_id="exp8019-eligible-targets",
        fit_targets_ready_score=1,
        calibration_targets_ready_score=1,
        public_manifests=public,
        evaluator_manifests=evaluator,
        support_by_role={},
        exclusion_manifest=e.reference(root / "public-fit.json"),
    )
    path = root / "results" / "experiment_8019_v695_eligible_targets.json"
    atomic_json(path, value)
    return path


def test_gates_and_isolation(tmp_path):
    """SCENARIO-REPORT-8020-ISOLATION: missing contracts and changed bytes close gates."""
    p = fixture(tmp_path)
    source, failures = e.load_sources(tmp_path, tmp_path / "raw")
    assert not failures
    assert set(source["evaluator_manifests"]) == {"fit", "tune", "calibration"}
    for role in ("stream", "retention"):
        assert all("y" not in r for r in e.read_role(source, role, labels=False))
    value = json.loads(p.read_text())
    value.pop("fit_targets_ready_score")
    atomic_json(p, value)
    assert e.load_sources(tmp_path, tmp_path / "raw")[1][0]["observed"] == "MISSING_CONTRACT_FIELD"
    value["fit_targets_ready_score"] = 0
    atomic_json(p, value)
    assert e.load_sources(tmp_path, tmp_path / "raw")[1][0]["observed"] == 0
    value["fit_targets_ready_score"] = 1
    atomic_json(p, value)
    Path(value["public_manifests"]["fit"]["path"]).write_text("{}")
    assert e.load_sources(tmp_path, tmp_path / "raw")[1]
    assert e.load_sources(tmp_path / "absent", tmp_path / "raw")[1]


def test_numerics(tmp_path):
    """SCENARIO-REPORT-8020-NUMERICS: convergent fit, exact identity and reload."""
    fixture(tmp_path)
    source, _ = e.load_sources(tmp_path, tmp_path / "raw")
    fitted = e.train(source, tmp_path / "raw")
    assert fitted["ready"]
    assert len(fitted["heads"]) == 35
    assert all(
        h["parameter_count"] == 110 for h in fitted["heads"] if h["arm"] == "conditioned_energy"
    )
    calibration = e.calibrate(source, fitted, tmp_path / "raw")
    assert calibration["independent"] == 63
    assert calibration["ready"]
    reduced = e.measure(source, fitted, calibration)
    assert reduced["sample_size_budget"]["independent"] < len(reduced["primitive_rows"])
    assert all(
        r["y"] is None for r in reduced["primitive_rows"] if r["role"] in {"stream", "retention"}
    )
    for h, ref in zip(fitted["heads"], fitted["checkpoints"], strict=True):
        x = e.sparse.inputs(e.sparse.usable(e.read_role(source, "fit")))
        assert np.array_equal(
            e.predict(h, x), e.predict(json.loads(Path(ref["path"]).read_text()), x)
        )
    blocked = copy.deepcopy(source)
    ep = Path(blocked["evaluator_manifests"]["calibration"]["path"])
    v = json.loads(ep.read_text())
    for r in v["rows"]:
        r["eligible_y"] = None
    atomic_json(ep, v)
    blocked["evaluator_manifests"]["calibration"] = e.reference(ep)
    with pytest.raises(ValueError, match="calibration_support"):
        e.calibrate(blocked, fitted, tmp_path / "raw")


def prefix():
    """Run script-path branches under the same private coverage configuration."""
    cli = str(e.ROOT / e.OWNED[-1])
    config = os.environ.get("CARNOT_8020_COVERAGE_CONFIG")
    return (
        [str(e.ROOT / ".venv/bin/python"), "-m", "coverage", "run", "--rcfile=" + config, cli]
        if config
        else [str(e.ROOT / ".venv/bin/python"), "-u", cli]
    )


def test_direct_cli(tmp_path):
    """SCENARIO-REPORT-8020-PUBLICATION: private fit precedes natural measurements."""
    fixture(tmp_path)
    output = tmp_path / "out" / (e.NAME + ".json")
    run = subprocess.run(
        prefix() + ["--root", str(tmp_path), "--output", str(output), "--fixture"],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    v = json.loads(output.read_text())
    assert v["verdict_class"] == "circular_positive" and v["energy_fit_ready_score"] == 0
    assert e.terminal(output)["passed"]
    run = subprocess.run(
        prefix() + ["--cold-replay", str(output)], capture_output=True, text=True, timeout=90
    )
    assert run.returncode == 0, run.stdout + run.stderr
    v["rows"][0]["numerator"] += 1
    atomic_json(output, v)
    assert e.main(["--cold-replay", str(output)]) == 1
    with pytest.raises(SystemExit):
        e.main(["--date", "20261001"])


def test_owned_readiness_and_terminal_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8020-PUBLICATION: coverage and owned failures close readiness."""
    fixture(tmp_path)
    monkeypatch.setattr(e, "terminal", e.replay)

    def commands(root, specs, *, log_dir, **kwargs):
        for c in specs:
            if c.name == "coverage_json":
                atomic_json(
                    Path(c.argv[-1]),
                    dict(
                        files={
                            p: dict(summary=dict(num_statements=1, missing_lines=0))
                            for p in e.OWNED
                        }
                    ),
                )
        return [dict(name=c.name, passed=True, scope=c.scope, exit_code=0) for c in specs]

    monkeypatch.setattr(e, "run_commands", commands)
    output = tmp_path / "natural" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["energy_fit_ready_score"] == 1 and value["verdict_class"] == "null"
    unsafe = dict(value, validation_receipts=[])
    atomic_json(output, unsafe)
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(output)
    atomic_json(output, value)
    fitted = json.loads(Path(value["checkpoint_references"][1]["path"]).read_text())
    altered = copy.deepcopy(fitted)
    altered["heads"][0]["parameters"][0] += 1
    changed = tmp_path / "changed-fit.json"
    atomic_json(changed, altered)
    v = copy.deepcopy(value)
    v["checkpoint_references"][1] = e.reference(changed)
    atomic_json(output, v)
    with pytest.raises(ValueError, match="checkpoint_state"):
        e.replay(output)
    atomic_json(output, value)
    altered_measurement = json.loads(Path(value["measurement_checkpoint"]["path"]).read_text())
    altered_measurement["primitive_rows"][0]["p"] += 0.01
    mp = tmp_path / "changed-measurement.json"
    atomic_json(mp, altered_measurement)
    v = copy.deepcopy(value)
    v["measurement_checkpoint"] = e.reference(mp)
    atomic_json(output, v)
    with pytest.raises(ValueError, match="primitive_reduction"):
        e.replay(output)
    with pytest.raises(ValueError, match="unfrozen_head"):
        e.calibrate(
            json.loads(Path(value["checkpoint_references"][0]["path"]).read_text()),
            altered,
            tmp_path,
        )
    monkeypatch.setattr(e, "run_commands", lambda *a, **kw: [dict(passed=False, scope="owned")])
    assert e.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(blocked)]) == 0
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    assert e.main(["--root", str(tmp_path / "absent"), "--output", str(blocked)]) == 1


def test_public_role_poisoning_and_missing_slots(tmp_path):
    """SCENARIO-REPORT-8020-ISOLATION: bad roles and exclusions remain explicit."""
    p = fixture(tmp_path)
    source, _ = e.load_sources(tmp_path, tmp_path / "raw")
    pp = Path(source["public_manifests"]["fit"]["path"])
    public = json.loads(pp.read_text())
    public["rows"][0]["public_eligible"] = False
    atomic_json(pp, public)
    source["public_manifests"]["fit"] = e.reference(pp)
    assert e.read_role(source, "fit")[0]["q"] is None
    fitted = e.train(source, tmp_path / "raw")
    calibration = e.calibrate(source, fitted, tmp_path / "raw")
    measured = e.measure(source, fitted, calibration)
    assert measured["sample_size_budget"]["excluded"] == 1
    assert any(r["exclusion_reason"] == "missing_public_inputs" for r in measured["primitive_rows"])
    v = json.loads(p.read_text())
    original = Path(v["public_manifests"]["fit"]["path"])
    poisoned = json.loads(original.read_text())
    poisoned["rows"][0]["role"] = "stream"
    atomic_json(original, poisoned)
    v["public_manifests"]["fit"] = e.reference(original)
    v["exclusion_manifest"] = e.reference(tmp_path / "public-tune.json")
    atomic_json(p, v)
    assert e.load_sources(tmp_path, tmp_path / "raw2")[1]
