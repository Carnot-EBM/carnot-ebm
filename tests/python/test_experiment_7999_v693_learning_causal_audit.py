"""REQ-REPORT-7999, SCENARIO-REPORT-7999-CLI: private publication and cold replay."""

import copy
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot import experiment_7999_v693_learning_causal_audit as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from test_learning_causal_audit_7999 import bundle  # noqa: F401


def fixture_input(bundle):
    """REQ-REPORT-7999: synthetic retention sources are separate from training."""
    value = copy.deepcopy(bundle)
    value["retention_public"] = [
        dict(value["sources"][i], family_id=f"retention-{i}", source_cluster_id=f"retention-{i}")
        for i in range(64)
    ]
    value["retention_targets"] = {
        r["family_id"]: i % 2 for i, r in enumerate(value["retention_public"])
    }
    return value


def test_real_cli_success_blocked_cold(bundle, tmp_path):
    """SCENARIO-REPORT-7999-CLI: real children run from an external directory with the guard enabled."""
    src, out = tmp_path / "fixture.json", tmp_path / "success" / (e.NAME + ".json")
    atomic_json(src, fixture_input(bundle))
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
        print(f"[7999-test] subprocess_begin={args[0]}", flush=True)
        run = subprocess.run(
            [*prefix, str(e.ROOT / e.OWNED[-1]), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        print(f"[7999-test] subprocess_end={args[0]} exit={run.returncode}", flush=True)
        assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(out.read_text())
    assert value["verdict_class"] == "null" and value["verifier_is_oracle"]
    assert value["retention_labels_opened"] and not value["generalized_learning_benefit_score"]
    assert e.replay(value)["passed"]
    bad = copy.deepcopy(value)
    bad["paired_comparisons"]["uniform_ipw"]["gain"] = 100
    with pytest.raises(ValueError, match="reduction"):
        e.replay(bad)
    atomic_json(out, bad)
    assert e.main(["--cold-replay", str(out)]) == 1
    assert e.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        e.main(["--date", "20260930"])


def test_measure_custody_recovery_and_validation(bundle, tmp_path):
    """REQ-REPORT-7999: final head and prediction seals precede retention targets."""
    value = e.measure(fixture_input(bundle), tmp_path / "evidence", True)
    assert value["positive_control_results"]["passed"]
    assert all(r["passed"] for r in value["crash_recovery_rows"])
    events = value["label_access_events"]
    assert (
        events[-1]["preceded_by"]["sha256"]
        == value["checkpoints"]["retention_predictions"]["sha256"]
    )
    counts = {p: dict(num_statements=1, missing_lines=0) for p in e.OWNED}
    e.apply_validation(value, [dict(required=True, passed=True)], counts)
    assert not value["learning_audit_ready_score"]  # Protocol fixtures cannot close a natural gate.
    value["verifier_is_oracle"] = False
    e.apply_validation(value, [dict(required=True, passed=True)], counts)
    assert value["learning_audit_ready_score"] == 1
    e.apply_validation(value, [dict(required=True, passed=False)], counts)
    assert value["verdict_class"] == "disqualified" and not value["learning_audit_ready_score"]
    blocked = e.base([dict(field="test", passed=False)])
    assert e.replay(blocked)["blocked"]
    blocked["learning_audit_ready_score"] = 1
    with pytest.raises(ValueError, match="readiness"):
        e.replay(blocked)
    with patch.object(e, "terminal_check", return_value=dict(passed=False)):
        with pytest.raises(ValueError, match="rejected"):
            e.publish(tmp_path / "reject" / (e.NAME + ".json"), e.base([]), tmp_path)
    with (
        patch.object(e, "reader_receipt", return_value=dict(passed=False)),
        patch.object(e, "terminal_check", return_value=dict(passed=True)),
    ):
        with pytest.raises(ValueError, match="primary"):
            e.publish(tmp_path / "reader" / (e.NAME + ".json"), e.base([]), tmp_path)


def test_authenticate_missing_and_contract(tmp_path):
    """REQ-REPORT-7999: exact gates distinguish missing artifacts from contract errors."""
    failures, plan = e.authenticate(tmp_path)
    assert len(failures) == 3 and all(r["field"] == "sha256" for r in failures)
    assert plan["checks"] == failures
    with (
        patch.object(e, "INPUTS", {7998: ("fake", "learning_measurement_ready_score", "pin")}),
        patch.object(e, "sha256_file", return_value="pin"),
    ):
        atomic_json(tmp_path / "results/fake.json", dict(experiment_id=7998))
        with pytest.raises(ValueError, match="contract"):
            e.authenticate(tmp_path)
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


def test_validation_manifest_and_main(bundle, tmp_path):
    """REQ-REPORT-7999: command receipts and nonempty coverage control readiness."""
    from carnot.reporting import learning_audit_validation_7999 as v

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    manifest = v.freeze(tmp_path / "raw", scratch)
    assert any(r["name"] == "full_pytest" and not r["required"] for r in manifest["commands"])
    counts = {p: dict(num_statements=1, missing_lines=0) for p in e.OWNED}
    atomic_json(
        scratch / "coverage.json",
        dict(files={str(e.ROOT / p): dict(summary=c) for p, c in counts.items()}),
    )
    assert v.coverage_counts(scratch) == counts
    with patch.object(
        v,
        "run_commands",
        side_effect=lambda root, specs, **kw: [
            dict(name=c.name, passed=True, exit_code=0) for c in specs
        ],
    ):
        assert all(r["passed"] for r in v.execute(manifest, tmp_path / "raw"))
    fixture = fixture_input(bundle)
    src = tmp_path / "fixture.json"
    atomic_json(src, fixture)
    with (
        patch.object(v, "execute", return_value=[dict(required=True, passed=True)]),
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
    value = e.base([])
    value["acceptance_gate_results"] = dict(validity=True, benefit=True)
    e.apply_validation(value, [dict(required=True, passed=True)], counts)
    assert value["verdict_class"] == "positive" and value["finite_replay_benefit_score"] == 1
    value = e.base([])
    value["positive_control_results"] = dict(passed=False)
    e.apply_validation(value, [dict(required=True, passed=True)], {})
    assert value["verdict_class"] == "disqualified"


def test_natural_loader_and_contract_fields(tmp_path):
    """REQ-REPORT-7999: natural loading authenticates historical bytes and keeps retention labels closed."""
    failures, plan = e.authenticate(e.ROOT)
    assert not failures
    loaded = e.load_bundle(plan)
    assert len(loaded["sources"]) == 256 and "retention_targets" not in loaded
    assert len(loaded["retention_public"]) == 64
    assert loaded["retention_target_ref"]["sha256"]
    for eid, (_, field, _) in e.INPUTS.items():
        assert plan["upstream"][eid][field] == 1
    source = copy.deepcopy(plan)
    capture = source["upstream"][7995]
    for row in capture["rows"]:
        if row["role"] == "retention":
            row.update(status="excluded", parsed=None)
    assert all(r["q"] is None for r in e.load_bundle(source)["retention_public"])


def test_nonfixture_measure_and_natural_main(bundle, tmp_path):
    """REQ-SELF-7999: a failed positive control disqualifies the natural measurement."""
    fixture = fixture_input(bundle)
    label_path = tmp_path / "targets.json"
    atomic_json(
        label_path,
        dict(rows=[dict(family_id=k, y=v) for k, v in fixture.pop("retention_targets").items()]),
    )
    fixture["retention_target_ref"] = e.reference(label_path)
    overlap = copy.deepcopy(fixture)
    overlap["retention_public"][0]["source_cluster_id"] = fixture["sources"][0]["source_cluster_id"]
    with pytest.raises(ValueError, match="role_overlap"):
        e.measure(overlap, tmp_path / "overlap")
    reducer, calls = e.m.reduce, []

    def missed_mutation(*args):
        calls.append(1)
        return reducer(*args) if len(calls) <= 5 else {}

    with (
        patch.object(e.m, "controls", return_value=dict(passed=False)),
        patch.object(e, "recover", return_value=[]),
        patch.object(e.m, "reduce", side_effect=missed_mutation),
    ):
        value = e.measure(fixture, tmp_path / "nonfixture")
    assert value["verdict_class"] == "disqualified" and not value["learning_audit_ready_score"]
    assert not any(r["passed"] for r in value["mutation_rows"])
    prediction_path = Path(value["checkpoints"]["retention_predictions"]["path"])
    previous = prediction_path.read_bytes()
    changed = json.loads(previous)
    changed["rows"][0]["probability"] = None
    atomic_json(prediction_path, changed)
    value["checkpoints"]["retention_predictions"] = e.reference(prediction_path)
    with pytest.raises(ValueError, match="retention_prediction"):
        e.replay(value)
    prediction_path.write_bytes(previous)
    value["checkpoints"]["retention_predictions"] = e.reference(prediction_path)
    with (
        patch.object(e, "authenticate", return_value=([], dict(checks=[], refs=[]))),
        patch.object(e, "load_bundle", return_value=fixture),
        patch.object(e, "measure", return_value=value),
        patch.object(e, "publish"),
    ):
        assert e.main(["--validation-worker", "--output", str(tmp_path / (e.NAME + ".json"))]) == 0
