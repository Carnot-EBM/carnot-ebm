"""REQ-REPORT-7996: exact custody, private CLI and terminal validation."""

import copy
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from carnot import experiment_7996_v693_sparse_energy_training as e
from carnot.reporting.current_work_receipt import atomic_json
from test_sparse_energy_7996 import fixture


def test_real_private_cli_success_blocked_cold(tmp_path):
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
        run = subprocess.run(
            [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.OWNED[-1]), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(out.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["sparse_fit_ready_score"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert e.replay(value)["passed"]
    assert e.main(["--cold-replay", str(out)]) == 0
    bad = copy.deepcopy(value)
    bad["rows"][0]["probability"] = 0.999
    atomic_json(out, bad)
    assert e.main(["--cold-replay", str(out)]) == 1
    assert e.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    atomic_json(src, {})
    assert e.main(["--fixture-input", str(src), "--validation-worker", "--output", str(out)]) == 1
    with pytest.raises(SystemExit):
        e.main(["--date", "20260930"])


def test_historical_only_rows_and_custody(tmp_path):
    failed, plan = e.authenticate(e.ROOT)
    assert not failed
    data, events = e.load_data(plan)
    assert set(data) == {"fit", "tune"}
    assert [len(data[r]) for r in data] == [256, 64]
    assert {r["role"] for r in events} == {"fit", "tune"}
    result = e.measure(data, tmp_path, plan)
    assert e.replay(result)["passed"]
    assert result["equivalent_classifier_parity"]["max_absolute_error"] < 1e-10
    assert result["sample_size_budget"]["independent"] == 300
    assert result["frozen_scalar_comparator"]["fitted_current_steps"] == 0
    bad = copy.deepcopy(result)
    bad["checkpoints"]["heads"]["sha256"] = "bad"
    with pytest.raises(ValueError):
        e.replay(bad)
    bad = copy.deepcopy(result)
    bad["state_bytes"] = {}
    with pytest.raises(ValueError):
        e.replay(bad)
    bad = e.base([])
    bad.update(verdict_class="blocked", sparse_fit_ready_score=1)
    with pytest.raises(ValueError):
        e.replay(bad)
    assert e.replay(e.base([{"passed": False}]))["passed"]
    badplan = copy.deepcopy(plan)
    badplan["upstream"][7980]["public_role_manifests"]["fit"]["sha256"] = "bad"
    with pytest.raises(ValueError):
        e.load_data(badplan)


def test_manifest_failure_disposition_and_parent(tmp_path):
    from carnot.reporting import sparse_validation_7996 as v

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    manifest = v.freeze(tmp_path / "raw", scratch)
    assert any(r["name"] == "full_pytest" and not r["required"] for r in manifest["commands"])

    def fake(root, commands, **kwargs):
        return [dict(name=r.name, passed=True, exit_code=0) for r in commands]

    with patch.object(v, "run_commands", side_effect=fake):
        assert all(r["passed"] for r in v.execute(manifest, tmp_path / "raw"))
    atomic_json(
        scratch / "coverage.json",
        dict(
            files={
                p: dict(summary=dict(num_statements=1, covered_lines=1, missing_lines=0))
                for p in e.OWNED
            }
        ),
    )
    assert len(v.coverage_counts(scratch)) == 4
    value = e.base([])
    e.apply_validation(value, [dict(required=False, passed=False, name="health")], {})
    assert value["verdict_class"] == "null"
    e.apply_validation(value, [dict(required=True, passed=False, name="owned")], {})
    assert value["verdict_class"] == "disqualified"
    counts = {p: dict(num_statements=1, missing_lines=0) for p in e.OWNED}
    with (
        patch.object(v, "execute", return_value=[dict(required=True, passed=True, name="owned")]),
        patch.object(v, "coverage_counts", return_value=counts),
    ):
        assert (
            e.main(
                [
                    "--root",
                    str(tmp_path / "absent"),
                    "--output",
                    str(tmp_path / "parent" / (e.NAME + ".json")),
                ]
            )
            == 0
        )
    with (
        patch.object(v, "execute", return_value=[dict(required=True, passed=True, name="owned")]),
        patch.object(v, "coverage_counts", return_value=counts),
    ):
        src = tmp_path / "src.json"
        atomic_json(src, fixture())
        assert (
            e.main(
                [
                    "--fixture-input",
                    str(src),
                    "--output",
                    str(tmp_path / "parent-fit" / (e.NAME + ".json")),
                ]
            )
            == 0
        )
    e.apply_validation(value, [dict(required=True, passed=True, name="owned")], counts)
    assert value["sparse_fit_ready_score"] == 0


def test_rejected_terminal_preserves_flag_and_reader_mismatch(tmp_path):
    value = e.base([dict(passed=False)])
    with patch.object(
        e,
        "terminal_check",
        side_effect=[
            dict(passed=False, flagged_adversarial=True, receipts=[]),
            dict(passed=True, receipts=[]),
        ],
    ):
        e.publish(tmp_path / (e.NAME + ".json"), value, tmp_path)
    assert value["flagged_adversarial"] is True
    assert value["verdict_class"] == "disqualified"
    with patch.object(e, "reader_receipt", return_value=dict(passed=False)):
        with pytest.raises(ValueError, match="primary_resolution"):
            e.publish(tmp_path / (e.NAME + ".json"), e.base([dict(passed=False)]), tmp_path)


def test_response_and_coverage_contract_errors(tmp_path):
    failed, plan = e.authenticate(e.ROOT)
    assert not failed
    plan = copy.deepcopy(plan)
    plan["upstream"][7969]["rows"][0]["parsed"] = {}
    with pytest.raises(ValueError, match="response_custody"):
        e.load_data(plan)
    value = e.measure(fixture(), tmp_path, dict(upstream={}))
    e.apply_validation(value, [dict(name="owned", required=True, passed=True)], {})
    assert value["verdict_class"] == "disqualified"
    value = e.measure(fixture(), tmp_path / "bad", dict(upstream={}))
    value["numerical_passed"] = False
    e.apply_validation(value, [], {})
    assert value["sparse_fit_ready_score"] == 0
