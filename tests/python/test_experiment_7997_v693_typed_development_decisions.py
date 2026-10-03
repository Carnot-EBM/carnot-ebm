"""REQ-REPORT-7997: sealed label custody and private real CLI routes."""

import copy
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from carnot import experiment_7997_v693_typed_development_decisions as e
from carnot.reporting.current_work_receipt import atomic_json
from test_typed_development_7997 import fixture


def test_private_real_cli_success_blocked_cold(tmp_path):
    src = tmp_path / "fixture.json"
    atomic_json(src, fixture())
    out = tmp_path / "success" / (e.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for args in [
        ["--fixture-input", str(src), "--validation-worker", "--output", str(out)],
        ["--cold-replay", str(out)],
        [
            "--root",
            str(tmp_path / "absent"),
            "--validation-worker",
            "--output",
            str(tmp_path / "blocked" / (e.NAME + ".json")),
        ],
    ]:
        print(f"[7997-test] subprocess_begin={args[0]}", flush=True)
        run = subprocess.run(
            [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.OWNED[-1]), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print(f"[7997-test] subprocess_end={args[0]} exit={run.returncode}", flush=True)
        assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(out.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["retention_labels_opened"] is False
    assert value["decision_measurement_ready_score"] == 0
    assert e.main(["--cold-replay", str(out)]) == 0
    assert e.replay(value)["passed"]
    bad = copy.deepcopy(value)
    bad["rows"][0]["actual_cost"] += 1
    atomic_json(out, bad)
    assert e.main(["--cold-replay", str(out)]) == 1
    assert e.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    bad = copy.deepcopy(value)
    bad["policies"]["spline"]["temperature"] = 999
    with pytest.raises(ValueError):
        e.replay(bad)
    bad = e.base([dict(passed=False)])
    assert e.replay(bad)["passed"]
    bad["decision_measurement_ready_score"] = 1
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(bad)
    with pytest.raises(SystemExit):
        e.main(["--date", "20260930"])
    atomic_json(src, {})
    assert e.main(["--fixture-input", str(src), "--validation-worker", "--output", str(out)]) == 1


def test_custody_and_parent_validation(tmp_path):
    from carnot.reporting import typed_validation_7997 as v

    failed, plan = e.authenticate(e.ROOT)
    assert not failed
    heads, public, labels, roles = e.load_public(plan)
    assert set(public) == {"calibration", "stream"}
    assert all(
        r["status"] == "completed"
        for sources in public.values()
        for r in sources
        if r["q"] is not None
    )
    assert set(labels) == {"calibration", "stream"}
    assert set(roles) >= {"fit", "tune", "calibration", "stream", "retention"}
    failed, missing = e.authenticate(tmp_path)
    assert failed and all(r["observed"] is None for r in failed)
    broken = copy.deepcopy(plan)
    broken["upstream"][7996]["checkpoints"]["heads"]["sha256"] = "bad"
    with pytest.raises(ValueError):
        e.load_public(broken)
    public["stream"][0]["source_cluster_id"] = public["calibration"][0]["source_cluster_id"]
    with pytest.raises(ValueError, match="role_overlap"):
        e.disjoint(public)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    manifest = v.freeze(tmp_path / "raw", scratch)
    assert any(c["name"] == "full_pytest" and not c["required"] for c in manifest["commands"])
    with patch.object(
        v,
        "run_commands",
        side_effect=lambda root, commands, **kw: [
            dict(name=c.name, passed=True, exit_code=0) for c in commands
        ],
    ):
        assert all(r["passed"] for r in v.execute(manifest, tmp_path / "raw"))
    counts = {p: dict(num_statements=1, covered_lines=1, missing_lines=0) for p in e.OWNED}
    atomic_json(
        scratch / "coverage.json", dict(files={p: dict(summary=c) for p, c in counts.items()})
    )
    assert v.coverage_counts(scratch) == counts
    value = e.base([])
    e.apply_validation(value, [dict(name="health", required=False, passed=False)], counts)
    assert value["verdict_class"] == "null"
    e.apply_validation(value, [dict(name="owned", required=True, passed=False)], counts)
    assert value["verdict_class"] == "disqualified"
    src = tmp_path / "fixture.json"
    atomic_json(src, fixture())
    receipts = [dict(name="owned", required=True, passed=True, exit_code=0)]
    with (
        patch.object(v, "execute", return_value=receipts),
        patch.object(v, "coverage_counts", return_value=counts),
    ):
        assert (
            e.main(
                [
                    "--fixture-input",
                    str(src),
                    "--output",
                    str(tmp_path / "parent" / (e.NAME + ".json")),
                ]
            )
            == 0
        )


def test_terminal_failures_replay_mutations_and_ready(tmp_path):
    """REQ-REPORT-7997: failed owned checks cannot keep readiness or clear flags."""
    from carnot.reporting import typed_validation_7997 as v
    import numpy as np

    with patch.object(e.json, "loads", return_value={}):
        with pytest.raises(ValueError, match="upstream_contract"):
            e.authenticate(e.ROOT)
    f = fixture()
    for sources in f["public"].values():
        for i, r in enumerate(sources):
            r["q"] = 0.1 if i % 2 == 0 else 0.9
    z = float(np.log(9))
    f["heads"]["logistic"][0]["parameters"][0] = -2 * z / 0.8
    f["heads"]["logistic"][0]["parameters"][9] = 1.25 * z
    theta = f["heads"]["mlp"][0]["parameters"]
    theta[0], theta[72], theta[80] = -10.0, 5.0, z / float(np.tanh(4))
    labels = {}
    for role, targets in f["targets"].items():
        path = tmp_path / "targets" / (role + ".json")
        atomic_json(path, dict(rows=targets))
        labels[role] = e.reference(path)
    result = e.measure(f["heads"], f["public"], labels, f["public"], tmp_path / "evidence")
    assert result["decision_benefit_score"] == 1
    counts = {p: dict(num_statements=1, missing_lines=0) for p in e.OWNED}
    receipts = [dict(name="owned", required=True, passed=True)]
    e.apply_validation(result, receipts, counts)
    assert result["decision_measurement_ready_score"] == 1
    for field, changed in [
        ("role_hashes", {}),
        ("policies", {}),
        ("checkpoint_hashes", dict(before="a", after="b")),
    ]:
        bad = copy.deepcopy(result)
        bad[field] = changed
        with pytest.raises(ValueError):
            e.replay(bad)
    for key in ("calibration_predictions", "stream_predictions"):
        bad = copy.deepcopy(result)
        old = json.loads(Path(bad["checkpoints"][key]["path"]).read_text())
        old["rows"][0]["probability"] = 0.333
        path = tmp_path / (key + "-mutated.json")
        atomic_json(path, old)
        bad["checkpoints"][key] = e.reference(path)
        with pytest.raises(ValueError, match="prediction_drift"):
            e.replay(bad)
    e.apply_validation(result, [dict(name="coverage", required=True, passed=True)], {})
    assert result["verdict_class"] == "disqualified"
    value = e.base([])
    out = tmp_path / "terminal" / (e.NAME + ".json")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with (
        patch.object(
            e,
            "terminal_check",
            side_effect=[dict(passed=False, flagged_adversarial=True), dict(passed=True)],
        ),
        patch.object(e, "reader_receipt", return_value=dict(passed=False)),
    ):
        with pytest.raises(ValueError, match="primary_resolution"):
            e.publish(out, value, scratch)
    saved = json.loads(out.read_text())
    assert saved["flagged_adversarial"] is True and saved["verdict_class"] == "disqualified"
    with (
        patch.object(e, "authenticate", return_value=([], dict(checks=[], refs=[], upstream={}))),
        patch.object(e, "load_public", return_value=(f["heads"], f["public"], labels, f["public"])),
        patch.object(e, "measure", return_value=e.base([])),
        patch.object(v, "execute", return_value=receipts),
        patch.object(v, "coverage_counts", return_value=counts),
    ):
        assert e.main(["--output", str(tmp_path / "natural-route" / (e.NAME + ".json"))]) == 0


def test_prior_target_exposure_is_an_owned_failure(tmp_path):
    """REQ-REPORT-7997: irreversible custody breaches remain disqualified."""
    from carnot.reporting import typed_validation_7997 as v

    path = tmp_path / "exposure.json"
    atomic_json(path, dict(predictions_sealed_before_stream_label_access=False))
    assert e.main(["--check-exposure", str(path)]) == 1
    manifest = v.freeze(tmp_path / "raw", tmp_path / "scratch", path)
    assert any(
        c["name"] == "stream_target_exposure_order" and c["required"] for c in manifest["commands"]
    )
    value = e.base([])
    value["prior_exposure_receipt"] = dict(predictions_sealed_before_stream_label_access=False)
    e.apply_validation(value, [], {})
    assert value["verdict_class"] == "disqualified"
    atomic_json(path, dict(predictions_sealed_before_stream_label_access=True))
    assert e.main(["--check-exposure", str(path)]) == 0
    with (
        patch.object(
            v, "execute", return_value=[dict(name="custody", required=True, passed=False)]
        ),
        patch.object(v, "coverage_counts", return_value={}),
    ):
        assert (
            e.main(
                [
                    "--root",
                    str(tmp_path / "absent"),
                    "--prior-exposure-receipt",
                    str(path),
                    "--output",
                    str(tmp_path / "parent" / (e.NAME + ".json")),
                ]
            )
            == 0
        )
        assert (
            e.main(
                [
                    "--root",
                    str(tmp_path / "absent"),
                    "--output",
                    str(tmp_path / "parent-blocked" / (e.NAME + ".json")),
                ]
            )
            == 0
        )
