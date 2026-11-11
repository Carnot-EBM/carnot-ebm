"""REQ-REPORT-8972 / SCENARIO-REPORT-8972-V767: diagnose only exposed evidence."""

from copy import deepcopy
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.autoresearch import encoder_head_8972 as n
from carnot.reporting import encoder_head_diagnosis_8972 as d
from carnot.reporting import encoder_head_runner_8972 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    """REQ-REPORT-8972: actual original bytes supply every development operand."""
    value = d.authenticate(d.ROOT, tmp_path_factory.mktemp("8972-inputs"))
    assert not value["failures"]
    return value


def test_feature_identity_and_frozen_heads(evidence):
    """REQ-REPORT-8972: identical readouts retain the original coordinate basis."""
    state = evidence["states"]["relational_logistic"][0]
    view = evidence["views"]["fit"]
    rows, labels = view["rows"][:8], view["labels"][:8]
    x = n.features(state, rows)
    assert x.shape == (8, 27)
    expected = n.g.predict(state, rows)
    actual = n.b._sigmoid(x @ np.asarray(state["params"]["head"][0]) + state["params"]["head"][1])
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    before = canonical_hash(state)
    for energy in (False, True):
        trained, gradients = n.fit_head(state, rows, labels, energy)
        assert canonical_hash(state) == before
        assert trained["params"]["encoder"] == state["params"]["encoder"]
        assert trained["steps"] == 200 and trained["rate"] == 0.01
        assert [r["step"] for r in gradients] == [0, 1, 200]
        assert all(r["actual_encoder_update_norm"] == 0 for r in gradients)
        if energy:
            assert gradients[0]["encoder_gradient_norm"] == 0
            assert gradients[1]["encoder_gradient_norm"] > 0
        assert trained["loss_after"] < trained["loss_before"]
        assert np.isfinite(n.g.predict(trained, rows)).all()
    with pytest.raises(ValueError, match="detector_schema"):
        n.features(state, [dict(rows[0], label=1)])
    with pytest.raises(ValueError, match="training_class_support"):
        n.fit_head(state, rows, [0] * len(rows), False)
    with pytest.raises(ValueError, match="binary_labels"):
        n.fit_head(state, rows, [True] * len(rows), False)


def test_authentication_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8972-V767: exact failures distinguish missing, zero and drift."""
    path = tmp_path / "upstream.json"
    monkeypatch.setattr(d, "INPUTS", [(9999, "upstream.json", "pin", "ready")])
    for value, reason in ((None, "missing_file"), ({"ready": 1}, "changed_bytes")):
        if value is not None:
            atomic_json(path, value)
        result = d.authenticate(tmp_path, tmp_path / "raw")
        assert result["failures"][0]["observed"] == reason
    for value in ({}, {"ready": 0}, {"ready": True}, {"ready": 1, "required_checks_passed": False}):
        atomic_json(path, value)
        monkeypatch.setattr(
            d, "INPUTS", [(9999, "upstream.json", d.reference(path)["sha256"], "ready")]
        )
        failed = d.authenticate(tmp_path, tmp_path / "raw")["failures"][0]
        assert failed["operator"] == "=="
        assert failed["artifact_field"] in ("ready", "required_checks_passed")
    path.write_text("[]")
    monkeypatch.setattr(
        d, "INPUTS", [(9999, "upstream.json", d.reference(path)["sha256"], "ready")]
    )
    assert d.authenticate(tmp_path, tmp_path / "raw")["failures"]


@pytest.fixture(scope="module")
def measured_work(tmp_path_factory):
    """REQ-REPORT-8972: one real twelve-head reduction backs all semantic assertions."""
    return runner.freeze(d.ROOT, tmp_path_factory.mktemp("8972-measurement"))


def test_complete_reduction_and_original_bounds(measured_work):
    """REQ-REPORT-8972: a reproducible null is ready without inventing confirmation."""
    result = measured_work["result"]
    assert result["completed_count"] == result["independent_count"] == 1344
    assert len(result["rows"]) == 5376
    assert result["role_counts"] == dict(fit=256, guard=64, calibration=128, test=128, online=768)
    assert result["numerical_training_calls"] == 12
    assert len(result["head_state_references"]) == 12
    assert len(result["feature_hashes"]) == 30
    assert len(result["gradient_rows"]) == 54
    assert len(result["original_bound_rows"]) == 5
    assert not any(r["passed"] for r in result["original_bound_rows"])
    auc = {r["arm"]: r["value"] for r in result["original_metric_rows"] if r["metric"] == "auroc"}
    assert auc["relational_logistic"] == pytest.approx(0.8079877112135178)
    assert auc["relational_energy"] == pytest.approx(0.4344086021505376)
    for row in result["rows"]:
        assert row["value"] == (row["p_error"] - row["error"]) ** 2
        assert row["p_error"] == pytest.approx(np.mean(row["seed_probabilities"]))
        assert row["status"] == "completed"
    original_energy = [
        r
        for r in result["gradient_rows"]
        if r["condition"] == "original_joint_training" and r["head"] == "energy"
    ]
    assert all(r["encoder_gradient_norm"] == 0 for r in original_energy if r["step"] == 0)
    assert all(r["encoder_gradient_norm"] > 0 for r in original_energy if r["step"] in (1, 200))
    candidate = json.loads(Path(result["candidate_reference"]["path"]).read_text())
    assert [s["seed"] for s in candidate["encoder_states"]] == [71, 72, 73]
    assert "logistic" in candidate["selection"]
    validation = dict(receipts=[], coverage={})
    value = d.build(measured_work, validation, True)
    assert value["verdict_class"] == "null" and value["diagnosis_ready_score"] == 0
    assert value["independent_generalization_score"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert not value["MODEL_SPECS"]
    for ref in value["cited_upstream_artifacts"]:
        assert set(ref["fields_imported"]) == set(
            measured_work["evidence"]["producers"][str(ref["experiment_id"])]
        )
    assert d.checksum(value) == value["reproducibility_checksum"]
    assert set(value) <= set(value["field_principles"])
    failed = d.build(measured_work, validation, False)
    assert failed["verdict_class"] == "disqualified"


def test_protocol_and_primitive_refusals(evidence, measured_work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8972-V767: role, seed, metric and frozen-byte changes fail closed."""
    raw = Path(measured_work["raw"])
    d.seal(raw / "candidate.json", json.loads((raw / "candidate.json").read_text()), True)
    with pytest.raises(ValueError, match="primitive_drift"):
        d.seal(raw / "candidate.json", {}, True)
    with pytest.raises(ValueError, match="operand_bytes"):
        d.retain(dict(path=str(tmp_path / "missing"), sha256="pin"), tmp_path)
    operand = tmp_path / "operand.json"
    atomic_json(operand, {"a": 1})
    ref = d.reference(operand)
    saved = d.retain(ref, tmp_path / "saved")
    Path(saved["path"]).write_text("changed")
    with pytest.raises(ValueError, match="snapshot_bytes"):
        d.retain(ref, tmp_path / "saved")
    changed = deepcopy(evidence)
    changed["views"]["guard"]["unit_ids"][0] = changed["views"]["fit"]["unit_ids"][0]
    with pytest.raises(ValueError, match="development_roles"):
        d.evaluate(changed, tmp_path)
    changed = deepcopy(evidence)
    changed["states"]["relational_energy"][0]["seed"] = 999
    with pytest.raises(ValueError, match="seed_roster"):
        d.evaluate(changed, tmp_path)
    changed = deepcopy(evidence)
    changed["producers"]["8964"]["descriptive_rows"][0]["value"] += 1
    with pytest.raises(ValueError, match="original_metric_reproduction"):
        d.evaluate(changed, tmp_path)
    original = n.fit_head

    def altered(*args, **kwargs):
        state, grad = original(*args, **kwargs)
        state["params"]["encoder"]["bias"][0] += 1
        return state, grad

    monkeypatch.setattr(n, "fit_head", altered)
    with pytest.raises(ValueError, match="frozen_encoder_changed"):
        d.evaluate(evidence, tmp_path)


def test_resources_and_blocked_rows(tmp_path, monkeypatch, measured_work):
    """SCENARIO-REPORT-8972-V767: CPU resource absence names exact checks and all slots."""
    assert d.resources(tmp_path) == []
    monkeypatch.setattr(d.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(d.os, "sysconf", lambda name: 1)
    monkeypatch.setattr(d.shutil, "disk_usage", lambda path: type("Disk", (), dict(free=0))())
    failures = d.resources(tmp_path)
    assert len(failures) == 9
    assert all(
        {
            "upstream",
            "check",
            "artifact_field",
            "operator",
            "expected",
            "observed",
            "sha256",
            "path",
        }
        <= set(r)
        for r in failures
    )
    monkeypatch.setattr(
        d.tempfile, "TemporaryFile", lambda **kwargs: (_ for _ in ()).throw(OSError("denied"))
    )
    assert d.resources(tmp_path)[-1]["artifact_field"] == "writable_disk"
    work = deepcopy(measured_work)
    work["evidence"]["failures"] = failures
    work["result"] = d.empty_result()
    value = d.build(work, dict(receipts=[], coverage={}), True)
    assert value["verdict_class"] == "blocked"
    assert value["censored_count"] == value["intended_count"] == 1344
    assert len(value["rows"]) == 5376
    assert all(r["value"] is None for r in value["rows"])


def test_private_cli_and_rehashed_refusal(tmp_path):
    """SCENARIO-REPORT-8972-V767: actual outside-checkout CLI publication and replay use real heads."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["JAX_PLATFORMS"] = "cpu"
    output = tmp_path / (d.NAME + ".json")
    cli = str(d.ROOT / d.CLI)

    def run(*args):
        result = subprocess.run(
            [sys.executable, "-u", cli, *args],
            cwd=tmp_path,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
        print(result.stdout, flush=True)
        return result

    result = run("--date", "20261110", "--fixture-e2e", str(output))
    assert result.returncode == 0, result.stdout
    value = json.loads(output.read_text())
    assert value["completed_count"] == 1344 and value["fixture_only"]
    assert value["diagnosis_ready_score"] == 0
    assert run("--cold-replay", str(output)).returncode == 0
    changed = deepcopy(value)
    changed["diagnostic_metric_rows"][0]["value"] += 0.1
    changed["reproducibility_checksum"] = d.checksum(changed)
    atomic_json(output, changed)
    assert run("--cold-replay", str(output)).returncode == 1
    atomic_json(output, value)
    work_path = Path(value["work_reference"]["path"])
    work = json.loads(work_path.read_text())
    work["result"]["rows"][0]["p_error"] += 0.1
    atomic_json(work_path, work)
    changed["work_reference"] = d.reference(work_path)
    changed["raw_shard_hashes"] = [d.reference(Path(r["path"])) for r in value["raw_shard_hashes"]]
    changed["reproducibility_checksum"] = d.checksum(changed)
    atomic_json(output, changed)
    assert run("--cold-replay", str(output)).returncode == 1
    assert run("--date", "wrong").returncode == 2
    assert (
        run(
            "--date", "20261110", "--fixture-e2e", str(d.ROOT / "results" / (d.NAME + ".json"))
        ).returncode
        == 2
    )


def test_runner_error_paths(measured_work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8972-V767: missing, changed and blocked evidence cannot gain readiness."""
    assert not runner.replay(tmp_path / "missing.json")["passed"]
    raw = Path(measured_work["raw"])
    work = deepcopy(measured_work)
    atomic_json(raw / "work.json", work)
    validation = dict(receipts=[], coverage={})
    atomic_json(raw / "validation.json", validation)
    work["raw_references"] = [d.reference(raw / "work.json"), d.reference(raw / "validation.json")]
    value = runner.artifact(work, validation, True)
    path = tmp_path / (d.NAME + ".json")
    atomic_json(path, value)
    value["code_config_hashes"][0]["sha256"] = "changed"
    atomic_json(path, value)
    assert runner.replay(path)["reason"] == "retained_bytes_drift"
    value = runner.artifact(work, validation, True)
    source = tmp_path / "source.py"
    source.write_text("changed")
    value["code_config_hashes"][0]["source_path"] = str(source)
    atomic_json(path, value)
    assert runner.replay(path)["reason"] == "executed_bytes_drift"
    monkeypatch.setattr(runner, "replay", lambda p: dict(passed=False))
    assert runner.main(["--cold-replay", str(path)]) == 1
    monkeypatch.setattr(runner, "replay", lambda p: dict(passed=True))
    assert runner.main(["--cold-replay", str(path)]) == 0
    for args in (
        ["--date", "wrong"],
        ["--date", "20261110", "--fixture-e2e", str(d.ROOT / "fixture.json")],
    ):
        with pytest.raises(SystemExit):
            runner.main(args)
    monkeypatch.undo()
    monkeypatch.setattr(
        d,
        "authenticate",
        lambda root, raw: dict(
            failures=[{"observed": "missing_file"}],
            producers={},
            references=[],
            states={},
            views={},
            method_source_rows=[],
        ),
    )
    blocked = runner.freeze(d.ROOT, tmp_path / "blocked")
    assert blocked["result"]["censored_count"] == 1344
    assert blocked["result"]["numerical_training_calls"] == 0


def test_production_validation_flow(measured_work, tmp_path, monkeypatch):
    """REQ-REPORT-8972: production reaches unchanged scanners and validates before publication."""
    from carnot.reporting import matched_corpus_runner_8949 as shared
    from carnot.reporting import v756_branch_qualification as qualification
    import sys

    work = deepcopy(measured_work)
    work["result"] = d.empty_result()
    work["evidence"]["failures"] = [{"observed": "external"}]

    def frozen(root, raw):
        work["raw"] = str(raw)
        atomic_json(raw / "work.json", work)
        return work

    def commands(root, plan, *, log_dir, heartbeat_s):
        log_dir.mkdir(parents=True, exist_ok=True)
        return [dict(name=c.name, passed=True, exit_code=0, timed_out=False) for c in plan]

    def plan(private, *, producer):
        (private / ".coverage").write_text("unit test only")
        return [runner.CommandSpec("unit", (sys.executable, "-c", "pass"), "owned", 1)]

    monkeypatch.setattr(runner, "freeze", frozen)
    monkeypatch.setattr(runner, "run_commands", commands)
    monkeypatch.setattr(shared, "validation_plan", plan)
    monkeypatch.setattr(qualification, "measured", lambda *a: {})
    monkeypatch.setattr(d, "ROOT", tmp_path)
    assert runner.main(["--date", "20261110"]) == 0
    value = json.loads((tmp_path / "results" / (d.NAME + ".json")).read_text())
    assert value["diagnosis_ready_score"] == 0
    assert value["global_test_debt"]["repeated"] is False


def test_retained_bytes_and_label_roster_refusal(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8972-V767: valid producer gates cannot bless altered operands."""
    path = tmp_path / "producer.json"
    atomic_json(
        path,
        dict(
            ready=1,
            required_checks_passed=True,
            source_artifact_hashes=[dict(path=str(tmp_path / "absent"), sha256="pin")],
            code_config_hashes=[],
            raw_shard_hashes=[],
        ),
    )
    monkeypatch.setattr(
        d, "INPUTS", [(9999, "producer.json", d.reference(path)["sha256"], "ready")]
    )
    failure = d.authenticate(tmp_path, tmp_path / "raw")["failures"][0]
    assert "retained_bytes:" in failure["observed"]
    monkeypatch.undo()
    original = d.json.loads

    def altered(payload):
        value = original(payload)
        if (
            isinstance(value, dict)
            and set(value) == {"rows"}
            and value["rows"]
            and isinstance(value["rows"][0], dict)
            and "label" in value["rows"][0]
        ):
            value["rows"][0]["unit_id"] = "changed"
        return value

    monkeypatch.setattr(d.json, "loads", altered)
    failures = d.authenticate(d.ROOT, tmp_path / "labels")["failures"]
    assert failures[-1]["observed"] == "label_roster"
    assert failures[-1]["artifact_field"] == "frozen_views"


def test_qualified_null_and_owned_failure(measured_work, monkeypatch):
    """REQ-REPORT-8972: complete controls qualify readiness while exposed benefit stays null."""
    from carnot.reporting import matched_corpus_a_8949 as shared

    monkeypatch.setattr(shared, "qualified", lambda validation, owned: True)
    validation = dict(receipts=[dict(name="test-only-qualified-control")], coverage={})
    value = d.build(measured_work, validation, False)
    assert value["verdict_class"] == "null" and value["diagnosis_ready_score"] == 1
    assert not value["acceptance_gates"]["scientific_benefit"]["passed"]
    work = deepcopy(measured_work)
    work["result"]["controls"] = {}
    assert d.build(work, validation, False)["diagnosis_ready_score"] == 0
    validation["terminal_rejected"] = True
    work["evidence"]["failures"] = [dict(observed="external")]
    value = d.build(work, validation, False)
    assert value["verdict_class"] == "disqualified" and value["flagged_adversarial"]


def test_resource_block_skips_numerical_and_validation_work(measured_work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8972-V767: missing dependencies produce a complete blocked CPU record."""
    work = deepcopy(measured_work)
    work["preconditions"]["resource_failures"] = [dict(observed="missing_dependency")]
    work["evidence"]["failures"] = work["preconditions"]["resource_failures"]
    work["result"] = d.empty_result()

    def frozen(root, raw):
        work["raw"] = str(raw)
        atomic_json(raw / "work.json", work)
        return work

    monkeypatch.setattr(runner, "freeze", frozen)
    monkeypatch.setattr(runner, "replay", lambda path: dict(passed=True))
    monkeypatch.setattr(d, "ROOT", tmp_path)
    assert runner.main(["--date", "20261110"]) == 0
    value = json.loads((tmp_path / "results" / (d.NAME + ".json")).read_text())
    assert value["verdict_class"] == "blocked" and value["completed_count"] == 0
    assert not value["required_checks_passed"]
