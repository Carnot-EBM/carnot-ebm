"""REQ-REPORT-7982: private CLI, exact custody, publication and cold replay."""

import copy
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from carnot import experiment_7982_v692_multivariate_energy as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from test_multivariate_energy_7982 import fixture


def test_private_cli_and_replay(tmp_path):
    src = tmp_path / "input.json"
    atomic_json(src, fixture())
    out = tmp_path / "success" / (e.NAME + ".json")
    cli = e.ROOT / "scripts/experiments" / (e.NAME + ".py")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for args in [
        ["--fixture-input", str(src), "--validation-worker", "--output", str(out)],
        ["--cold-replay", str(out)],
    ]:
        p = subprocess.run(
            [str(e.ROOT / ".venv/bin/python"), "-u", str(cli), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert p.returncode == 0, p.stdout + p.stderr
    value = json.loads(out.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == []
    assert not any(value["model_invocation_counts"].values())
    assert {r["role"] for r in value["label_access_events"]} == {"fit", "tune"}
    assert json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())[
        "primary_sha256"
    ] == sha256_file(out)
    assert e.main(["--cold-replay", str(out)]) == 0
    tampered = copy.deepcopy(value)
    tampered["rows"][0]["p"] = 0.99
    atomic_json(out, tampered)
    assert e.main(["--cold-replay", str(out)]) == 1
    assert e.main(["--cold-replay", str(tmp_path / "missing.json")]) == 1
    with pytest.raises(SystemExit):
        e.main(["--date", "20260930"])
    atomic_json(src, dict(fit=[]))
    assert e.main(["--fixture-input", str(src), "--validation-worker", "--output", str(out)]) == 1


def test_live_authentication_join_and_seals(tmp_path):
    failed, plan = e.authenticate(e.ROOT)
    assert not failed
    data, accesses = e.load_data(plan)
    assert len(data["fit"]) == 256
    assert sum(r["q"] is not None for r in data["fit"]) == 238
    assert len(accesses) == 2
    assert all(r["y"] is None for r in data["policy_design"])
    value = e.measure(data, tmp_path, fixture=False)
    assert value["energy_fit_ready_score"] == 1
    assert value["fit_support"]["independent"] >= 128
    assert e.replay(value)["passed"]
    for field in ["heads_seal", "prediction_seal"]:
        broken = copy.deepcopy(value)
        broken[field]["sha256"] = "sha256:bad"
        with pytest.raises(ValueError, match="hash"):
            e.replay(broken)
    broken = copy.deepcopy(value)
    broken["energy_fit_ready_score"] = 1
    broken["verdict_class"] = "disqualified"
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(broken)
    broken = copy.deepcopy(value)
    broken["optimizer_work"]["total_steps"] = 1
    with pytest.raises(ValueError, match="optimizer_work"):
        e.replay(broken)


def test_missing_external_and_failed_owned(tmp_path):
    out = tmp_path / "blocked" / (e.NAME + ".json")
    assert (
        e.main(["--root", str(tmp_path / "missing"), "--validation-worker", "--output", str(out)])
        == 0
    )
    value = json.loads(out.read_text())
    assert value["verdict_class"] == "blocked" and value["energy_fit_ready_score"] == 0
    assert value["gate_check_summary"][0]["observed"] is None
    assert e.replay(value)["passed"]
    e.apply_validation(value, [dict(name="health", required=False, passed=False, exit_code=2)])
    assert value["repository_health"]["current"][0]["name"] == "health"
    e.apply_validation(value, [dict(name="owned", required=True, passed=False, exit_code=1)])
    assert value["verdict_class"] == "disqualified"
    with patch.object(
        e,
        "terminal_check",
        side_effect=[
            dict(passed=False, receipts=[], flagged_adversarial=True),
            dict(passed=True, receipts=[]),
        ],
    ):
        e.publish(tmp_path / "terminal" / (e.NAME + ".json"), e.base([]))
    assert (
        json.loads((tmp_path / "terminal" / (e.NAME + ".json")).read_text())["verdict_class"]
        == "disqualified"
    )


def test_frozen_validation_routes(tmp_path):
    from carnot.reporting import multivariate_validation_7982 as v

    raw, scratch = tmp_path / "raw", tmp_path / "scratch"
    scratch.mkdir()
    manifest = v.freeze(raw, scratch)
    assert any(r["name"] == "full_pytest" and not r["required"] for r in manifest["commands"])
    assert any("test_source_boundary_7852.py" in str(r) for r in manifest["commands"])
    assert any(
        "test_experiment_7942_v689_sentence_labels.py" in str(r) for r in manifest["commands"]
    )

    def fake(root, commands, **kwargs):
        return [dict(name=r.name, passed=True, exit_code=0) for r in commands]

    with patch.object(v, "run_commands", side_effect=fake):
        receipts = v.execute(manifest, raw)
    assert all(r["passed"] for r in receipts)
    atomic_json(
        scratch / "coverage.json",
        dict(
            files={
                str(e.ROOT / e.OWNED[0]): dict(
                    summary=dict(num_statements=1, covered_lines=1, missing_lines=0)
                )
            }
        ),
    )
    assert v.coverage_counts(scratch)[e.OWNED[0]]["num_statements"] == 1
    atomic_json(scratch / "coverage.json", dict(files={}))
    assert not v.coverage_counts(scratch)
    with patch.object(e, "run_validation", return_value=(manifest, receipts, {})):
        out = tmp_path / "parent" / (e.NAME + ".json")
        assert e.main(["--root", str(tmp_path / "missing"), "--output", str(out)]) == 0


def test_authentication_and_join_mutations():
    failed, plan = e.authenticate(e.ROOT)
    assert not failed
    broken = copy.deepcopy(plan)
    broken["upstream"][7969]["rows"][0]["parsed"]["probability"] = -0.5
    with pytest.raises(ValueError, match="parse_drift"):
        e.load_data(broken)
    with patch.object(e, "PINS", {**e.PINS, 7980: "sha256:wrong"}):
        failures, _ = e.authenticate(e.ROOT)
    assert any(r["field"] == "sha256" for r in failures)


def test_live_scalar_control_cli(tmp_path):
    """SCENARIO-REPORT-7982-CUSTODY: exact frozen scalar controls join successfully."""
    out = tmp_path / "live" / (e.NAME + ".json")
    assert e.main(["--validation-worker", "--output", str(out)]) == 0
    value = json.loads(out.read_text())
    controls = json.loads(Path(value["frozen_scalar_controls"]["path"]).read_text())
    assert len(controls["rows"]) == 352 * 5
    assert (
        controls["heads"]["sha256"]
        == e.authenticate(e.ROOT)[1]["upstream"][7972]["heads_seal"]["sha256"]
    )
    assert value["energy_fit_ready_score"] == 1


def test_public_roster_mutations(tmp_path):
    _, plan = e.authenticate(e.ROOT)
    for mutation, reason in [
        ("role", "role_roster"),
        ("hidden", "public_fields"),
        ("overlap", "cross_role"),
    ]:
        broken = copy.deepcopy(plan)
        view = json.loads(
            e.checked(broken["upstream"][7980]["public_role_manifests"]["fit"]).read_text()
        )
        if mutation == "role":
            view["role"] = "evaluation"
        elif mutation == "hidden":
            view["request_rows"][0]["y"] = 1
        else:
            tune = json.loads(
                e.checked(
                    broken["upstream"][7980]["public_role_manifests"]["calibration_replay"]
                ).read_text()
            )
            view["request_rows"][0]["source_bytes"] = tune["request_rows"][0]["source_bytes"]
        path = tmp_path / (mutation + ".json")
        atomic_json(path, view)
        broken["upstream"][7980]["public_role_manifests"]["fit"] = e.reference(path)
        with pytest.raises(ValueError, match=reason):
            e.load_data(broken)


def test_support_and_numerical_failure_dispositions(tmp_path):
    from carnot.verify import multivariate_energy_7982 as m

    data = fixture()
    data["fit"] = data["fit"][:127]
    value = e.measure(data, tmp_path / "shortfall", fixture=False)
    assert value["verdict_class"] == "blocked"
    assert value["gate_check_summary"][0]["field"] == "fit.independent"
    assert value["gate_check_summary"][0]["op"] == ">="
    assert value["gate_check_summary"][0]["observed"] == 127
    measured = m.fit(fixture())
    measured["gradient_checks"][0]["passed"] = False
    with patch.object(m, "fit", return_value=measured):
        value = e.measure(fixture(), tmp_path / "gradient", fixture=True)
    assert value["verdict_class"] == "disqualified" and value["energy_fit_ready_score"] == 0
    with (
        patch.object(e, "terminal_check", return_value=dict(passed=True)),
        patch.object(e, "reader_receipt", return_value=dict(passed=False)),
    ):
        with pytest.raises(ValueError, match="primary_resolution"):
            e.publish(tmp_path / "resolution" / (e.NAME + ".json"), e.base([]))


def test_validation_wrapper(tmp_path):
    from carnot.reporting import multivariate_validation_7982 as v

    with (
        patch.object(v, "freeze", return_value={"commands": []}),
        patch.object(v, "execute", return_value=[]),
        patch.object(v, "coverage_counts", return_value={}),
    ):
        assert e.run_validation(tmp_path / "raw", tmp_path / "scratch") == (
            {"commands": []},
            [],
            {},
        )
