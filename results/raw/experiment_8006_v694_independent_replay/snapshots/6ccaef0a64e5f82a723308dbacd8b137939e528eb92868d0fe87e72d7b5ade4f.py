"""REQ-REPORT-8000, SCENARIO-REPORT-8000-CLI: real private terminal paths."""

import copy
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot import experiment_8000_v693_delayed_confidence as e
from carnot.reporting.current_work_receipt import atomic_json
from test_delayed_confidence_8000 import fixture


def test_real_cli(tmp_path):
    """SCENARIO-REPORT-8000-CLI: success, blocked and replay use the real script."""
    src, out = tmp_path / "fixture.json", tmp_path / "success" / (e.NAME + ".json")
    atomic_json(src, fixture())
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    prefix = [str(e.ROOT / ".venv/bin/python"), "-u"]
    if env.get("CARNOT_AUDIT_COVERAGE_FILE"):
        prefix = [
            str(e.ROOT / ".venv/bin/coverage"),
            "run",
            "--parallel-mode",
            "--data-file=" + env["CARNOT_AUDIT_COVERAGE_FILE"],
            "--include=" + ",".join(str(e.ROOT / p) for p in e.OWNED),
        ]
    for args in (
        ["--fixture-input", str(src), "--validation-worker", "--output", str(out)],
        ["--cold-replay", str(out)],
        [
            "--root",
            str(tmp_path / "missing"),
            "--validation-worker",
            "--output",
            str(tmp_path / "blocked" / (e.NAME + ".json")),
        ],
    ):
        print("[8000-test] subprocess_begin=" + args[0], flush=True)
        run = subprocess.run(
            [*prefix, str(e.ROOT / e.OWNED[-1]), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print(f"[8000-test] subprocess_end={args[0]} exit={run.returncode}", flush=True)
        assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(out.read_text())
    assert value["verifier_is_oracle"] and value["verdict_class"] == "circular_positive"
    assert e.replay(value)["passed"]
    value["issued_state_rows"][0]["issue_alpha"] = 0.49
    atomic_json(out, value)
    assert e.main(["--cold-replay", str(out)]) == 1
    assert e.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        e.main(["--date", "20260930"])


def test_apply_owned_gates(tmp_path):
    """REQ-REPORT-8000: scientific null readiness is independent of benefit."""
    value = e.base([])
    e.finish_measurement(value, fixture(), tmp_path / "checked", True)
    value.update(verifier_is_oracle=False, verdict_class="null")
    good = {p: dict(num_statements=2, missing_lines=0) for p in e.OWNED}
    e.apply_validation(value, [dict(required=True, passed=True)], good)
    assert value["confidence_measurement_ready_score"] == 1
    assert value["confidence_benefit_score"] == 0
    e.apply_validation(value, [dict(required=True, passed=False)], good)
    assert value["verdict_class"] == "disqualified"
    blocked = e.base([dict(field="sha256", observed=None)])
    e.apply_validation(blocked, [], good)
    assert e.replay(blocked)["blocked"]
    blocked["confidence_measurement_ready_score"] = 1
    with pytest.raises(ValueError, match="readiness"):
        e.replay(blocked)
    with patch.object(e, "terminal_check", return_value=dict(passed=False)):
        with pytest.raises(ValueError, match="rejected"):
            e.publish(tmp_path / (e.NAME + ".json"), e.base([]), tmp_path)
    with (
        patch.object(e, "reader_receipt", return_value=dict(passed=False)),
        patch.object(e, "terminal_check", return_value=dict(passed=True)),
    ):
        with pytest.raises(ValueError, match="reader"):
            e.publish(tmp_path / "other" / (e.NAME + ".json"), e.base([]), tmp_path)


def test_authentication_and_natural_load(tmp_path):
    """REQ-REPORT-8000: immutable producers and shard bytes qualify independently."""
    failures, plan = e.authenticate(tmp_path)
    assert len(failures) == 3 and plan["refs"] == []
    failures, plan = e.authenticate(e.ROOT)
    assert not failures
    bundle = e.load(plan)
    assert len(bundle["stream"]) == 256 and len(bundle["calibration"]) > 0
    assert bundle["head"]["parameter_count"] == 33
    natural = e.base([])
    e.finish_measurement(natural, bundle, tmp_path / "natural", False)
    assert natural["point_prediction_seal"]["sha256"]
    assert not natural["verifier_is_oracle"]
    bad = copy.deepcopy(plan)
    bad["upstream"][7994]["public_role_manifests"]["stream"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        e.load(bad)
    with (
        patch.object(e, "INPUTS", {7995: ("fake", "capture_ready_score", "pin")}),
        patch.object(e, "sha256_file", return_value="pin"),
    ):
        atomic_json(tmp_path / "results/fake.json", dict(experiment_id=7995))
        with pytest.raises(ValueError, match="contract"):
            e.authenticate(tmp_path)
    value = e.base([])
    e.finish_measurement(value, fixture(), tmp_path, True)
    assert value["positive_control_results"]["passed"]
    assert not value["confidence_measurement_ready_score"]


def test_owned_validation_orchestration(tmp_path):
    """SCENARIO-REPORT-8000-CLI: orchestration preserves receipts without recursive suites."""
    from carnot.reporting import delayed_confidence_validation_8000 as v

    raw = tmp_path / "evidence"
    manifest = v.freeze(raw, tmp_path)
    with patch.object(v, "run_commands", return_value=[dict(passed=True, exit_code=0)]):
        receipts = v.execute(manifest, raw)
    assert len(receipts) == len(manifest["commands"])
    files = {p: dict(summary=dict(num_statements=1, missing_lines=0)) for p in e.OWNED}
    files[str(e.ROOT / e.OWNED[0])] = files.pop(e.OWNED[0])
    files["unowned.py"] = dict(summary={})
    atomic_json(tmp_path / "coverage.json", dict(files=files))
    counts = v.coverage_counts(tmp_path)
    src = tmp_path / "fixture.json"
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
                    str(tmp_path / "result" / (e.NAME + ".json")),
                ]
            )
            == 0
        )
    empty = copy.deepcopy(fixture())
    for row in empty["stream"]:
        row["p"] = None
        row["status"] = "failed"
    from carnot.verify import delayed_confidence_8000 as m

    measured = m.measure(empty)
    assert not measured["acceptance_gate_results"]["support"]
    assert measured["delay_sensitivity"][0]["lower95"] is None


def test_role_overlap():
    """REQ-REPORT-8000: overlapping calibration and stream groups are contract failures."""
    _, plan = e.authenticate(e.ROOT)
    cohort = plan["upstream"][7994]
    cohort["public_role_manifests"]["calibration"] = cohort["public_role_manifests"]["stream"]
    cohort["evaluator_role_manifests"]["calibration"] = cohort["evaluator_role_manifests"]["stream"]
    with pytest.raises(ValueError, match="role_overlap"):
        e.load(plan)


def test_saved_state_tamper(tmp_path):
    """SCENARIO-VERIFY-8000-CAUSAL: a changed pending state cannot pass cold restart."""
    value = e.base([])
    e.finish_measurement(value, fixture(), tmp_path, True)
    checkpoint = Path(value["restart_state_checkpoint"]["path"])
    saved = json.loads(checkpoint.read_text())
    saved["states"]["scalar-20"]["alpha"] = 0.49
    altered = tmp_path / "altered-state.json"
    atomic_json(altered, saved)
    value["restart_state_checkpoint"] = e.custody.reference(altered)
    with pytest.raises(ValueError, match="cold_restart_drift"):
        e.replay(value)
