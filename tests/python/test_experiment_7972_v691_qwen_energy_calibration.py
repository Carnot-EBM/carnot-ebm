"""REQ-REPORT-7972: private publication, authenticated blocking and cold reduction."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot import experiment_7972_v691_qwen_energy_calibration as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from test_qwen_energy_calibration_7972 import fixture


def test_authentication_missing_and_pinned_current(tmp_path):
    failures, _ = e.authenticate(tmp_path)
    assert failures and all(
        set(("upstream_id", "path", "hash", "field", "op", "expected", "observed")) <= set(r)
        for r in failures
    )
    failures, plan = e.authenticate(e.ROOT)
    assert not failures
    assert plan["protocol_match_rows"] and all(r["passed"] for r in plan["protocol_match_rows"])
    assert all(v["run_date"] == "20261001" for v in plan["upstream"].values())


def test_private_fitting_seals_and_cold_reduction(tmp_path):
    data = fixture()
    value = e.measure(data, tmp_path)
    assert value["qwen_calibration_ready_score"] == 1
    assert [r["role"] for r in value["role_label_access_events"]] == [
        "fit",
        "tune",
        "policy_design",
        "evaluation",
    ]
    assert e.replay(value)["rows"] == value["rows"]
    bad = copy.deepcopy(value)
    bad["rows"][0]["brier"] += 0.01
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(bad)
    bad = copy.deepcopy(value)
    bad["calibrator_checkpoints"]["heads"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        e.replay(bad)
    bad = copy.deepcopy(value)
    bad["positive_control_rows"][0]["detected"] = False
    with pytest.raises(ValueError, match="positive_control_drift"):
        e.replay(bad)


def test_insufficient_support_and_empty_evaluation(tmp_path):
    data = fixture()
    data["fit"] = data["fit"][:20]
    value = e.measure(data, tmp_path / "small")
    assert value["honest_verdict"] == "complete_null_insufficient_calibration_support"
    assert value["qwen_calibration_ready_score"] == 0
    with patch.object(e.c, "positive_control", return_value=dict(detected=False)):
        failed = e.measure(fixture(), tmp_path / "control-failed")
    assert failed["honest_verdict"] == "complete_disqualified_positive_control"
    assert e.replay(value)["qwen_calibration_ready_score"] == 0
    data = fixture()
    data["evaluation"] = []
    value = e.measure(data, tmp_path / "empty")
    assert value["qwen_calibration_ready_score"] == 0


def test_cli_fitting_replay_blocking_and_rejection(tmp_path):
    input_path = tmp_path / "data.json"
    atomic_json(input_path, fixture())
    output = tmp_path / "success" / (e.NAME + ".json")
    assert (
        e.main(["--date", "20261001", "--fixture-input", str(input_path), "--output", str(output)])
        == 0
    )
    assert e.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["MODEL_SPECS"] == []
    assert not any(value["model_invocation_counts"].values())
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert terminal["primary_sha256"] == sha256_file(output)
    missing = tmp_path / "blocked" / (e.NAME + ".json")
    assert (
        e.main(
            ["--root", str(tmp_path / "absent"), "--validation-worker", "--output", str(missing)]
        )
        == 0
    )
    assert json.loads(missing.read_text())["verdict_class"] == "blocked"
    assert e.main(["--cold-replay", str(missing)]) == 0
    atomic_json(input_path, dict(evaluation=[]))
    assert e.main(["--fixture-input", str(input_path), "--output", str(output)]) == 1
    assert e.main(["--cold-replay", str(tmp_path / "absent.json")]) == 1
    with pytest.raises(SystemExit) as err:
        e.main(["--date", "20260930"])
    assert err.value.code == 2


def test_required_failure_disqualifies(tmp_path):
    value = e.base([])
    e.apply_validation(value, [dict(name="owned", passed=False, exit_code=1, required=True)])
    assert value["verdict_class"] == "disqualified"
    assert value["qwen_calibration_ready_score"] == 0
    e.apply_validation(value, [dict(name="health", passed=False, exit_code=2, required=False)])
    assert value["repository_health"]["current"][0]["name"] == "health"


def test_terminal_failure_and_recheck(tmp_path):
    value = e.base([])
    out = tmp_path / (e.NAME + ".json")
    with patch.object(
        e,
        "terminal_check",
        side_effect=[
            dict(passed=False, flagged_adversarial=True, receipts=[]),
            dict(passed=True, receipts=[]),
        ],
    ):
        e.publish(out, value)
    assert json.loads(out.read_text())["verdict_class"] == "disqualified"
    with patch.object(e, "terminal_check", return_value=dict(passed=False, receipts=[])):
        with pytest.raises(ValueError, match="candidate_rejected"):
            e.publish(tmp_path / "rejected" / out.name, e.base([]))


def test_live_roles_and_label_or_parse_drift(tmp_path):
    failures, plan = e.authenticate(e.ROOT)
    assert not failures
    value = e.measure(e.role_loader(plan), tmp_path)
    assert value["qwen_calibration_ready_score"] == 1
    assert value["fit_support"]["independent"] >= 128
    assert len(value["role_label_access_events"][-1]["seals"]) == 2
    assert e.replay(value)["rows"] == value["rows"]
    predictions = copy.deepcopy(plan["upstream"]["exp7969"]["rows"][:1])
    predictions[0]["parsed"]["probability"] = -1
    with pytest.raises(ValueError, match="parse_drift"):
        e.scalar_rows(predictions, plan["upstream"]["exp7968"]["rows"])
    for r in plan["upstream"]["exp7955"]["response_union_rows"]:
        if r["role"] == "evaluation":
            r["y"] = 1 - r["y"] if r["y"] is not None else 1
            break
    with pytest.raises(ValueError, match="historical_label_drift"):
        e.role_loader(plan)(
            "evaluation", {k: value["calibrator_checkpoints"][k] for k in ("heads", "policies")}
        )


def test_authenticated_request_drift_fails_closed():
    original = e.capture.capture.freeze

    def changed(views):
        frozen = original(views)
        frozen[0]["request"]["temperature"] = 1
        return frozen

    with (
        patch.object(e.capture, "replay", return_value={}),
        patch.object(e.capture.capture, "freeze", side_effect=changed),
    ):
        failed, _ = e.authenticate(e.ROOT)
    assert any(
        r["field"] == "authenticated_requests" and "request_parse_drift" in str(r["observed"])
        for r in failed
    )


def test_cold_guard_mutations(tmp_path):
    unsafe = e.base([])
    unsafe["qwen_calibration_ready_score"] = 1
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(unsafe)
    data = fixture()
    small = copy.deepcopy(data)
    small["fit"] = small["fit"][:20]
    value = e.measure(small, tmp_path / "small")
    p = Path(value["calibrator_checkpoints"]["primitives"]["path"])
    atomic_json(p, {k: data[k] for k in ("fit", "tune")})
    value["calibrator_checkpoints"]["primitives"] = e.reference(p)
    with pytest.raises(ValueError, match="support_drift"):
        e.replay(value)
    value = e.measure(data, tmp_path / "valid")
    hp = Path(value["calibrator_checkpoints"]["heads"]["path"])
    original = json.loads(hp.read_text())
    for field, expected in [
        ("config", "checkpoint_identity_drift"),
        ("parameters", "no_parameter_changes"),
    ]:
        h = copy.deepcopy(original)
        if field == "config":
            h["config"]["steps"] = 1
        else:
            h["heads"]["gibbs"][0]["parameters"] = h["heads"]["gibbs"][0]["initial_parameters"]
            value.update(
                e.c.evaluate(
                    h["heads"],
                    json.loads(Path(value["policies_seal"]["path"]).read_text()),
                    data["evaluation"],
                )
            )
        atomic_json(hp, h)
        value["calibrator_checkpoints"]["heads"] = e.reference(hp)
        if field == "parameters":
            pp = Path(value["policies_seal"]["path"])
            policies = e.c.design(h["heads"], data["policy_design"])
            atomic_json(pp, policies)
            value["calibrator_checkpoints"]["policies"] = e.reference(pp)
            value.update(e.c.evaluate(h["heads"], policies, data["evaluation"]))
        with pytest.raises(ValueError, match=expected):
            e.replay(value)
    atomic_json(hp, original)
    value = e.measure(data, tmp_path / "policy")
    pp = Path(value["policies_seal"]["path"])
    atomic_json(pp, {})
    value["calibrator_checkpoints"]["policies"] = e.reference(pp)
    with pytest.raises(ValueError, match="policy_drift"):
        e.replay(value)


def test_frozen_commands_real_expected_exit_and_parent_failure(tmp_path):
    raw, scratch = tmp_path / "raw", tmp_path / "scratch"
    scratch.mkdir()
    manifest = e.freeze_commands(raw, scratch)
    assert manifest["coverage_includes"] == e.INCLUDE
    assert all(c["deadline_s"] <= 300 for c in manifest["commands"])
    spec = dict(
        name="expected_failure",
        argv=[str(e.ROOT / ".venv/bin/python"), "-c", "print('rejected');raise SystemExit(1)"],
        deadline_s=10,
        required=True,
        expected_exit=1,
        reason="rejected",
    )
    receipts = e.execute_commands(dict(commands=[spec]), raw, scratch)
    assert receipts[0]["passed"] and receipts[0]["exit_code"] == 1
    cached = dict(
        receipts[0],
        name="repository_health_full_suite",
        passed=False,
        required=False,
        command_argv=[str(e.ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
        receipt_scope="synthetic_cache_transport_fixture",
    )
    atomic_json(raw / "repository_health_receipt.json", cached)
    cached_manifest = e.freeze_commands(raw, scratch)
    assert "reuse_receipt" in cached_manifest["commands"][-1]
    replayed = e.execute_commands(dict(commands=[cached_manifest["commands"][-1]]), raw, scratch)
    assert replayed[0]["reused"] and not replayed[0]["passed"]
    cached["name"] = "wrong_name"
    atomic_json(raw / "repository_health_receipt.json", cached)
    with pytest.raises(ValueError, match="repository_health_receipt_identity"):
        e.execute_commands(
            dict(commands=[e.freeze_commands(raw, scratch)["commands"][-1]]), raw, scratch
        )
    output = tmp_path / "parent" / (e.NAME + ".json")

    def incomplete_coverage(manifest, raw, workspace):
        observed = e.run_commands(
            e.ROOT,
            [
                e.CommandSpec(
                    "private_coverage_fixture",
                    (
                        str(e.ROOT / ".venv/bin/coverage"),
                        "run",
                        "--data-file=" + str(workspace / ".coverage"),
                        "--include=" + e.INCLUDE,
                        str(e.ROOT / e.OWNED[2]),
                        "--date",
                        "20260930",
                    ),
                    "private_expected_rejection",
                    30,
                )
            ],
            log_dir=raw / "private-coverage",
            heartbeat_s=10,
        )
        assert observed[0]["exit_code"] == 2
        atomic_json(
            workspace / "coverage.json",
            dict(files={e.OWNED[0]: dict(summary=dict(num_statements=1, missing_lines=1))}),
        )
        return receipts

    with patch.object(e, "execute_commands", side_effect=incomplete_coverage):
        assert e.main(["--root", str(tmp_path / "absent"), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"
    assert e.main(["--output", str(tmp_path / "wrong.json")]) == 1
    assert e.main(["--validation-worker"]) == 1
    with patch.object(e, "terminal_check", return_value=dict(passed=False)):
        assert e.main(["--terminal-recheck", str(output)]) == 1
    with (
        patch.object(e, "reader_receipt", return_value=dict(passed=False)),
        patch.object(e, "terminal_check", return_value=dict(passed=True)),
    ):
        with pytest.raises(ValueError, match="primary_resolution"):
            e.publish(tmp_path / "readers" / (e.NAME + ".json"), e.base([]))
