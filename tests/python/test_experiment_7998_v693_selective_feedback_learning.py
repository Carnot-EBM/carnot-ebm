"""REQ-REPORT-7998, SCENARIO-REPORT-7998-CLI: real private custody paths."""

import copy
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot import experiment_7998_v693_selective_feedback_learning as e
from carnot.reporting.current_work_receipt import atomic_json
from test_selective_feedback_7998 import fixture


def test_real_cli_success_blocked_cold(tmp_path):
    data = fixture()
    data["public"]["calibration"][0].update(q=None, features=None, status="failed")
    data["public"]["stream"][0].update(q=None, features=None, status="censored")
    src, out = tmp_path / "fixture.json", tmp_path / "success" / (e.NAME + ".json")
    atomic_json(src, data)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for args in [
        ["--fixture-input", str(src), "--validation-worker", "--output", str(out)],
        ["--cold-replay", str(out)],
        [
            "--root",
            str(tmp_path / "missing"),
            "--validation-worker",
            "--output",
            str(tmp_path / "blocked" / (e.NAME + ".json")),
        ],
    ]:
        print(f"[7998-test] subprocess_begin={args[0]}", flush=True)
        run = subprocess.run(
            [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.OWNED[-1]), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print(f"[7998-test] subprocess_end={args[0]} exit={run.returncode}", flush=True)
        assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(out.read_text())
    assert value["verdict_class"] == "null" and value["verifier_is_oracle"]
    assert not value["retention_labels_opened"]
    assert e.replay(value)["passed"]
    bad = copy.deepcopy(value)
    bad["issued_predictions"][1]["probability"] += 0.1
    with pytest.raises(ValueError):
        e.replay(bad)
    atomic_json(out, bad)
    assert e.main(["--cold-replay", str(out)]) == 1
    assert e.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    blocked = e.base([dict(passed=False)])
    assert e.replay(blocked)["passed"]
    blocked["learning_measurement_ready_score"] = 1
    with pytest.raises(ValueError):
        e.replay(blocked)
    with pytest.raises(SystemExit):
        e.main(["--date", "20260930"])


def test_validation_and_parent_paths(tmp_path):
    from carnot.reporting import selective_validation_7998 as v

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    manifest = v.freeze(tmp_path / "raw", scratch)
    assert any(c["name"] == "full_pytest" for c in manifest["commands"])
    counts = {p: dict(num_statements=1, missing_lines=0) for p in e.OWNED}
    atomic_json(
        scratch / "coverage.json",
        dict(files={str(e.ROOT / p): dict(summary=c) for p, c in counts.items()}),
    )
    assert v.coverage_counts(scratch) == counts
    with patch.object(
        v,
        "run_commands",
        side_effect=lambda root, commands, **kw: [
            dict(name=c.name, passed=True, exit_code=0) for c in commands
        ],
    ):
        receipts = v.execute(manifest, tmp_path / "raw")
    value = e.base([])
    value["mechanism_checks"] = dict(passed=True)
    value["checkpoints"] = dict(x=1)
    e.apply_validation(value, receipts, counts)
    assert value["learning_measurement_ready_score"] == 1
    e.apply_validation(value, [dict(required=True, passed=False)], counts)
    assert (
        value["verdict_class"] == "disqualified" and value["learning_measurement_ready_score"] == 0
    )
    src = tmp_path / "fixture.json"
    atomic_json(src, fixture())
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
    with patch.object(
        e, "terminal_check", return_value=dict(passed=False, flagged_adversarial=True)
    ):
        with pytest.raises(ValueError):
            e.publish(tmp_path / "reject" / (e.NAME + ".json"), e.base([]), scratch)
    e.apply_validation(e.base([]), [], {})


def test_replay_rejects_shard_mutations_and_reader_failure(tmp_path):
    """REQ-REPORT-7998: recomputation rejects locally rehashed malicious shards."""
    from carnot.reporting.current_work_receipt import sha256_file

    src, out = tmp_path / "fixture.json", tmp_path / "source" / (e.NAME + ".json")
    atomic_json(src, fixture())
    assert e.main(["--fixture-input", str(src), "--validation-worker", "--output", str(out)]) == 0
    original = json.loads(out.read_text())
    assert e.main(["--cold-replay", str(out)]) == 0
    bad = copy.deepcopy(original)
    bad["acquisition_rows"][0]["pi"] = 0.0
    with pytest.raises(ValueError, match="acquisition_drift"):
        e.replay(bad)
    key = "targeted_ipw-101"
    for field, error in [
        ("head_checksum", "head_drift"),
        ("probability", "prediction_drift"),
        ("final", "update_drift"),
    ]:
        bad = copy.deepcopy(original)
        shard = json.loads(Path(bad["checkpoints"][key]["path"]).read_text())
        if field == "final":
            shard["trajectory"]["final_state"]["head"]["decay_scale"] = 0.4
        else:
            shard["trajectory"]["issued_predictions"][0][field] = 999
        path = tmp_path / (field + ".json")
        atomic_json(path, shard)
        bad["checkpoints"][key] = dict(path=str(path), sha256=sha256_file(path))
        with pytest.raises(ValueError, match=error):
            e.replay(bad)
    with patch.object(e, "reader_receipt", return_value=dict(passed=False)):
        with pytest.raises(ValueError, match="primary_resolution"):
            e.publish(
                tmp_path / "reader-reject" / (e.NAME + ".json"), copy.deepcopy(original), tmp_path
            )
    with patch.object(e, "measure", return_value=copy.deepcopy(original)):
        assert (
            e.main(
                [
                    "--validation-worker",
                    "--output",
                    str(tmp_path / "natural-control" / (e.NAME + ".json")),
                ]
            )
            == 0
        )
    assert (
        e.main(
            [
                "--root",
                str(tmp_path / "absent"),
                "--validation-worker",
                "--output",
                str(tmp_path / "blocked2" / (e.NAME + ".json")),
            ]
        )
        == 0
    )


def test_named_capture_gates_and_contract_errors():
    """REQ-REPORT-7998: named role gates are explicit operands, not imputed zeros."""
    failures, plan = e.upstream.authenticate(e.ROOT)
    assert not failures
    broken = copy.deepcopy(plan)
    broken["upstream"][7995].pop("stream_capture_ready_score")
    with patch.object(e.upstream, "authenticate", return_value=([], broken)):
        with pytest.raises(ValueError, match="upstream_contract"):
            e.authenticate(e.ROOT)
    broken = copy.deepcopy(plan)
    broken["upstream"][7995]["stream_capture_ready_score"] = 0
    with patch.object(e.upstream, "authenticate", return_value=([], broken)):
        failed, _ = e.authenticate(e.ROOT)
    assert any(r["field"] == "stream_capture_ready_score" and r["observed"] == 0 for r in failed)


def test_past_shuffle_never_samples_future_labels():
    """REQ-SELF-7998: shuffled diagnostics use only already due past labels."""
    reveals = [
        dict(family_id=f"id-{i}", origin_slot=i, due_slot=i + 20, y=i % 2) for i in range(40)
    ]
    callback, rows = e.past_shuffle(reveals)
    assert callback("id-0") == 0
    for r in reveals[1:]:
        callback(r["family_id"])
    assert all(r["sampled_origin_slot"] <= r["origin_slot"] for r in rows)


def test_prior_full_suite_receipt_is_historical(tmp_path):
    """REQ-REPORT-7998: retain failed real-command diagnostics without another suite."""
    from carnot.reporting import selective_validation_7998 as v

    log, config, path = (
        tmp_path / "suite.log",
        tmp_path / "producer.json",
        tmp_path / "receipt.json",
    )
    log.write_text("prior full-suite failure\n")
    atomic_json(config, dict(config=e.m.CONFIG))
    receipt = dict(
        name="full_pytest",
        command_argv=[str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        exit_code=1,
        passed=False,
        log_path=str(log),
        log_sha256=e.sha256_file(log),
        required=False,
    )
    atomic_json(
        path,
        dict(
            receipt=receipt,
            producer_execution_date="20261001",
            producer_configuration=e.reference(config),
        ),
    )
    assert e.main(["--check-health-receipt", str(path)]) == 0
    manifest = v.freeze(tmp_path / "raw", tmp_path, path)
    assert not any(c["name"] == "full_pytest" for c in manifest["commands"])
    with patch.object(
        v,
        "run_commands",
        side_effect=lambda root, commands, **kw: [
            dict(name=c.name, passed=True, exit_code=0) for c in commands
        ],
    ):
        receipts = v.execute(manifest, tmp_path / "raw")
    value = e.base([])
    e.apply_validation(value, receipts, {})
    assert value["repository_health"]["historical"][0]["exit_code"] == 1
    assert not value["repository_health"]["current"]
    log.write_text("changed")
    assert e.main(["--check-health-receipt", str(path)]) == 1
