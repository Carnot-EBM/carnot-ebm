"""REQ-REPORT-7984: byte custody, private CLI and failed-check dispositions."""

import copy
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from carnot import experiment_7984_v692_evidence_ablation as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from test_evidence_ablation_7984 import fixture


def test_private_cli_replay_and_mutations(tmp_path):
    public, data = fixture()
    src = tmp_path / "input.json"
    atomic_json(src, dict(public=public, data=data))
    output = tmp_path / "results" / (e.NAME + ".json")
    cli = e.ROOT / e.OWNED[-1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for args in [
        ["--fixture-input", str(src), "--validation-worker", "--output", str(output)],
        ["--cold-replay", str(output)],
    ]:
        child = subprocess.run(
            [str(e.ROOT / ".venv/bin/python"), "-u", str(cli), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert e.replay(value)["passed"]
    assert json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())[
        "primary_sha256"
    ] == sha256_file(output)
    for field in ["rows", "paired_comparisons", "duplicate_parity"]:
        broken = copy.deepcopy(value)
        broken[field] = []
        with pytest.raises(ValueError, match="reduction_drift"):
            e.replay(broken)
    broken = copy.deepcopy(value)
    broken["ablation_ready_score"] = 1
    broken["verdict_class"] = "disqualified"
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(broken)
    broken = copy.deepcopy(value)
    broken["checkpoints"]["heads"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="hash"):
        e.replay(broken)
    atomic_json(output, broken)
    assert e.main(["--cold-replay", str(output)]) == 1
    with pytest.raises(SystemExit):
        e.main(["--date", "20260930"])


def test_external_missing_and_validation_failures(tmp_path):
    output = tmp_path / "blocked" / (e.NAME + ".json")
    assert (
        e.main(
            ["--root", str(tmp_path / "missing"), "--validation-worker", "--output", str(output)]
        )
        == 0
    )
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "blocked" and value["ablation_ready_score"] == 0
    assert any(r["observed"] is None for r in value["gate_check_summary"])
    assert e.replay(value)["blocked"]
    e.apply_validation(value, [dict(name="required", required=True, passed=False, exit_code=1)])
    assert value["verdict_class"] == "disqualified"
    e.apply_validation(value, [dict(name="health", required=False, passed=False, exit_code=1)])
    assert len(value["validation_receipts"]) == 2
    with patch.object(
        e,
        "terminal_check",
        side_effect=[dict(passed=False, flagged_adversarial=False), dict(passed=True)],
    ):
        e.publish(tmp_path / "corrected" / (e.NAME + ".json"), value)
    with (
        patch.object(e, "reader_receipt", return_value=dict(passed=False)),
        patch.object(e, "terminal_check", return_value=dict(passed=True)),
    ):
        with pytest.raises(ValueError, match="primary_resolution"):
            e.publish(tmp_path / "resolver" / (e.NAME + ".json"), e.base([]))


def test_authenticated_live_original_roles(tmp_path):
    failures, plan = e.authenticate(e.ROOT)
    assert not failures
    public = e.load_public(plan)
    assert set(public) == {"fit", "tune", "policy_design", "evaluation"}
    assert len(public["evaluation"]) == 64
    value = e.measure(public, plan, tmp_path, fixture_data=None)
    assert value["ablation_ready_score"] == 1
    assert {r["role"] for r in value["label_access_events"]} == set(public)
    assert value["intervention_manifest"]["frozen_before_labels"]
    assert value["protocol_contrast"]["truth_claim"] is False
    assert e.replay(value)["passed"]
    with patch.object(e, "PINS", {**e.PINS, 7982: "bad"}):
        failed, _ = e.authenticate(e.ROOT)
    assert any(r["field"] == "sha256" for r in failed)
    broken = copy.deepcopy(plan)
    broken["upstream"][7980]["public_role_manifests"]["fit"] = dict(
        path=str(tmp_path / "bad.json"), sha256="bad"
    )
    with pytest.raises(ValueError, match="hash"):
        e.load_public(broken)


def test_fixture_shortfall_and_numerical_validation(tmp_path):
    public, data = fixture()
    data["fit"] = data["fit"][:4]
    public["fit"] = public["fit"][:4]
    value = e.measure(public, {}, tmp_path / "small", fixture_data=data)
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    src = tmp_path / "bad.json"
    atomic_json(src, dict(public=public, data={}))
    assert (
        e.main(
            [
                "--fixture-input",
                str(src),
                "--validation-worker",
                "--output",
                str(tmp_path / (e.NAME + ".json")),
            ]
        )
        == 1
    )


def test_validation_manifest_and_parent_route(tmp_path):
    from carnot.reporting import evidence_ablation_validation_7984 as v

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    manifest = v.freeze(tmp_path / "raw", scratch)
    names = {r["name"] for r in manifest["commands"]}
    assert {"full_pytest", "coverage_report", "cold_replay", "private_live_cli"} <= names
    assert not next(r for r in manifest["commands"] if r["name"] == "full_pytest")["required"]
    with patch.object(v, "run_commands", return_value=[dict(passed=True, exit_code=0)]):
        receipts = v.execute(manifest, tmp_path / "raw")
    assert len(receipts) == len(manifest["commands"])
    atomic_json(
        scratch / "coverage.json",
        dict(
            files={
                e.OWNED[0]: dict(summary=dict(num_statements=4, covered_lines=4)),
                "other.py": dict(summary={}),
            }
        ),
    )
    assert v.coverage_counts(scratch)[e.OWNED[0]]["num_statements"] == 4
    with (
        patch.object(v, "freeze", return_value=manifest),
        patch.object(v, "execute", return_value=receipts),
        patch.object(v, "coverage_counts", return_value={}),
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


def test_role_custody_and_parse_rejection(tmp_path):
    """SCENARIO-REPORT-7984-CUSTODY: drift cannot enter fit labels or predictor views."""
    _, plan = e.authenticate(e.ROOT)
    for mutation, reason in [
        ("role", "role_roster"),
        ("label", "public_fields"),
        ("cross", "cross_role"),
    ]:
        broken = copy.deepcopy(plan)
        view = json.loads(
            e.checked(plan["upstream"][7980]["public_role_manifests"]["fit"]).read_text()
        )
        if mutation == "role":
            view["role"] = "evaluation"
        elif mutation == "label":
            view["request_rows"][0]["y"] = 1
        else:
            tune = json.loads(
                e.checked(plan["upstream"][7980]["public_role_manifests"]["tune"]).read_text()
            )
            view["request_rows"][0]["source_bytes"] = tune["request_rows"][0]["source_bytes"]
        path = tmp_path / (mutation + ".json")
        atomic_json(path, view)
        broken["upstream"][7980]["public_role_manifests"]["fit"] = e.reference(path)
        with pytest.raises(ValueError, match=reason):
            e.load_public(broken)
    broken = copy.deepcopy(plan)
    broken["upstream"][7969]["rows"][0]["parsed"]["probability"] = -0.5
    with pytest.raises(ValueError, match="parse_drift"):
        e.measure(e.load_public(plan), broken, tmp_path / "parse", fixture_data=None)


def test_intervention_drift_and_scientific_dispositions(tmp_path):
    """REQ-REPORT-7984-DISPOSITION: numerical failures never qualify a scientific gain."""
    public, data = fixture()
    views = e.a.interventions(public)
    fitted = e.a.fit(data, views)
    reduced = e.a.evaluate(
        fitted,
        e.a.design(fitted, data["policy_design"], views["policy_design"]),
        data["evaluation"],
        views["evaluation"],
    )
    failed = copy.deepcopy(fitted)
    failed["arms"]["full"]["gradient_checks"][0]["passed"] = False
    with patch.object(e.a, "fit", return_value=failed):
        value = e.measure(public, {}, tmp_path / "numerical", fixture_data=data)
    assert value["verdict_class"] == "disqualified" and value["ablation_ready_score"] == 0
    assert e.replay(value)["passed"]
    path = Path(value["checkpoints"]["interventions"]["path"])
    altered = json.loads(path.read_text())
    altered["fit"][0]["donor_family_id"] = "tampered"
    atomic_json(path, altered)
    value["checkpoints"]["interventions"] = e.reference(path)
    with pytest.raises(ValueError, match="intervention_drift"):
        e.replay(value)
    positive = dict(reduced, added_information_score=1)
    with (
        patch.object(e.a, "fit", return_value=fitted),
        patch.object(e.a, "evaluate", return_value=positive),
    ):
        value = e.measure(public, {}, tmp_path / "benefit", fixture_data=data)
    assert value["acceptance_gate_results"]["added_information"]
    deficient = dict(
        reduced, evaluation_support=dict(passed=False, independent=0), added_information_score=0
    )
    with (
        patch.object(e.a, "fit", return_value=fitted),
        patch.object(e.a, "evaluate", return_value=deficient),
    ):
        value = e.measure(public, {}, tmp_path / "descriptive", fixture_data=data)
    assert value["ablation_ready_score"] == 1 and not value["added_information_score"]
