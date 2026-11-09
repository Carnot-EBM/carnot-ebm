"""REQ-REPORT-8958; SCENARIO-REPORT-8958-CONTRACT: exposed graph diagnosis."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import confidence_diagnosis_8958 as d
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def evidence():
    """REQ-REPORT-8958: natural retained responses must authenticate before analysis."""
    value = d.authenticate(d.ROOT)
    assert not value["failures"]
    assert len(value["corpus"]) == len(value["units"]) == 320
    return value


@pytest.fixture(scope="module")
def diagnosis(evidence):
    """REQ-REPORT-8958: measure the complete development diagnosis once."""
    return d.diagnose(evidence)


def test_original_null_and_all_errors(diagnosis):
    """REQ-REPORT-8958: custody and readiness do not depend on a favorable explanation."""
    assert diagnosis["completed_count"] == diagnosis["independent_count"] == 320
    assert diagnosis["class_balance"]["all"]["errors"] == 233
    assert diagnosis["out_of_graph_count"] == 6
    assert len(diagnosis["confidence_rows"]) == 320
    assert len(diagnosis["original_bound_rows"]) == 3
    assert all(not r["passed"] for r in diagnosis["original_bound_rows"])
    assert diagnosis["original_paired_intervals"]["independent_clusters"] == 80
    assert diagnosis["numerical_fit_count"] == 12
    assert diagnosis["numerical_training_calls"] == 2400
    assert len(diagnosis["diagnostic_fit_rows"]) == 3 * 5 * 64
    assert all(
        r["exposure_scope"] == "exposed_development" for r in diagnosis["diagnostic_fit_rows"]
    )
    assert all(
        not r["available_at_inference"]
        for r in diagnosis["feature_access_rows"]
        if r["feature"] in {"source_graph", "query_node", "proposed_node"}
    )
    assert {r["metric"] for r in diagnosis["original_metric_rows"]} == {
        "brier",
        "auroc",
        "accepted_error_50",
    }


def test_split_and_missing_candidates(evidence):
    """SCENARIO-REPORT-8958-CONTRACT: outcomes cannot select development roles or remove nodes."""
    view, labels = d.views(evidence["units"], evidence["corpus"])
    assert sum(r["development_role"] == "fit" for r in view) == 256
    assert sum(r["development_role"] == "guard" for r in view) == 64
    assert all(
        not ({"value", "error", "expected", "valid_sets", "true_distances"} & set(r)) for r in view
    )
    changed = deepcopy(evidence["corpus"])
    for row in changed:
        row["value"] = 1 - row["value"]
    other, _ = d.views(evidence["units"][::-1], changed[::-1])
    assert view == other
    assert sum(r["error"] for r in labels) == 233
    for candidate in (None, "", "not a node", "ZZ"):
        one = deepcopy(evidence["corpus"])
        one[0]["answer"] = candidate
        test_view, _ = d.views(evidence["units"], one)
        row = next(r for r in test_view if r["unit_id"] == one[0]["unit_id"])
        assert row["proposed_node"] is None and row["missing_node"] == 1
    with pytest.raises(ValueError, match="development_roster"):
        d.views(evidence["units"][:-1], evidence["corpus"][:-1])


def test_missing_authentication(tmp_path):
    """REQ-REPORT-8958: external absence is terminal blocked with exact failed operands."""
    evidence = d.authenticate(tmp_path)
    assert evidence["failures"][0]["check"] == "missing_file"
    assert all(
        {"upstream", "path", "hash", "check", "artifact_field", "expected", "observed"} <= set(r)
        for r in evidence["failures"]
    )
    empty = d.empty_result()
    assert empty["censored_count"] == 320 and len(empty["rows"]) == 320


def test_original_semantic_tamper(evidence, monkeypatch):
    """SCENARIO-REPORT-8958-CONTRACT: rehashed original metrics cannot redefine the null."""
    changed = deepcopy(evidence)
    changed["original"]["rows"][0]["error"] ^= 1
    with pytest.raises(ValueError, match="original"):
        d.original_reduction(changed)


def test_private_cli_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-8958-CONTRACT: cold publication works outside the checkout without PYTHONPATH."""
    output = tmp_path / (d.NAME + ".json")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    cli = str(d.ROOT / d.CLI)

    def run(*args):
        return subprocess.run(
            [sys.executable, "-u", cli, *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

    result = run("--date", "20261109", "--fixture-e2e", str(output))
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["fixture_only"] and value["diagnosis_ready_score"] == 0
    assert run("--cold-replay", str(output)).returncode == 0
    value["diagnosis_ready_score"] = 1
    value["reproducibility_checksum"] = d.calibration.checksum(value)
    atomic_json(output, value)
    assert run("--cold-replay", str(output)).returncode == 1
    assert run("--date", "wrong").returncode == 2
    assert (
        run("--date", "20261109", "--fixture-e2e", str(d.ROOT / "forbidden.json")).returncode == 2
    )


@pytest.mark.parametrize("fault", ["pin", "operand", "structure", "rows", "helper"])
def test_authentication_refuses_drift(evidence, tmp_path, monkeypatch, fault):
    """SCENARIO-REPORT-8958-CONTRACT: byte and semantic changes fail before fitting."""
    reference = d.calibration.reference
    if fault in {"pin", "structure"}:
        for name, _, _ in d.INPUTS:
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}")
        if fault == "structure":
            pins = {str(tmp_path / name): pin for name, pin, _ in d.INPUTS}
            monkeypatch.setattr(
                d.calibration, "reference", lambda p: dict(path=str(p), sha256=pins[str(p)])
            )
        result = d.authenticate(tmp_path)
    else:
        if fault == "operand":
            original_ref = evidence["original"]["raw_shard_hashes"][0]["path"]
            monkeypatch.setattr(
                d.calibration,
                "reference",
                lambda p: (
                    dict(path=str(p), sha256="changed") if str(p) == original_ref else reference(p)
                ),
            )
        elif fault == "helper":
            monkeypatch.setattr(
                d.calibration,
                "reference",
                lambda p: (
                    dict(path=str(p), sha256="changed") if p == Path(d.b.__file__) else reference(p)
                ),
            )
        else:
            monkeypatch.setattr(d.admission, "join", lambda _: ([], [], []))
            monkeypatch.setattr(d.admission, "reduce", lambda *_: dict(rows=[], corpus_manifest=[]))
        result = d.authenticate(d.ROOT)
    assert result["failures"] and result["corpus"] == []


@pytest.mark.parametrize("fault", ["coverage", "interval", "selection", "roster"])
def test_original_reduction_negative_controls(evidence, monkeypatch, fault):
    """REQ-REPORT-8958: changed selection or metric bounds cannot pass reduction."""
    changed = deepcopy(evidence)
    if fault == "coverage":
        changed["original"]["coverage_rows"][0]["value"] = -1
    elif fault == "interval":
        changed["original"]["paired_intervals"]["draws"] = 0
    elif fault == "selection":
        changed["original"]["rows"][0]["selected_50"] = not changed["original"]["rows"][0][
            "selected_50"
        ]
    else:
        changed["original"]["rows"].pop()
    monkeypatch.setattr(d.b, "summarize", lambda _: evidence["original"]["paired_intervals"])
    with pytest.raises(ValueError, match="original"):
        d.original_reduction(changed)


def test_fit_custody_and_access(diagnosis):
    """REQ-REPORT-8958: permutations preserve balance and guard labels never train heads."""
    manifest = diagnosis["development_manifest"]["units"]
    fit_ids = {r["unit_id"] for r in manifest if r["development_role"] == "fit"}
    for state in diagnosis["diagnostic_states"]:
        assert len(state["sample_ids"]) == 256
        assert set(state["sample_ids"]) <= fit_ids
        if state["arm"].startswith("permuted_"):
            regular = next(
                r
                for r in diagnosis["diagnostic_states"]
                if r["seed"] == state["seed"] and r["arm"] == state["arm"].removeprefix("permuted_")
            )
            assert state["training_errors"] == regular["training_errors"]
            assert state["training_labels_hash"] != regular["training_labels_hash"]
    with pytest.raises(ValueError, match="feature_schema"):
        d.b.features([dict(mean_token_nll=1, first_answer_top2_margin=1, query_node="AA")])
    rows = [
        dict(unit_id=str(i), arm="constant", condition="single", error=1, p_error=0.5)
        for i in range(2)
    ]
    assert next(r for r in d.metric_rows(rows) if r["metric"] == "auroc")["value"] is None


def test_build_terminal_classes(tmp_path, monkeypatch, diagnosis):
    """REQ-REPORT-8958: failed controls disqualify while valid null diagnosis is ready."""
    from carnot.reporting import confidence_diagnosis_runner_8958 as runner

    missing = d.authenticate(tmp_path)
    monkeypatch.setattr(d, "authenticate", lambda _: missing)
    work = runner.freeze(d.ROOT, tmp_path / "raw")
    validation = dict(receipts=[], coverage={})
    assert d.build(work, validation, False)["verdict_class"] == "disqualified"
    monkeypatch.setattr(d.admission.shard_a, "qualified", lambda *_: True)
    value = d.build(work, validation, False)
    assert value["verdict_class"] == "blocked" and value["diagnosis_ready_score"] == 0
    work["evidence"]["failures"] = []
    work["result"] = diagnosis
    value = d.build(work, validation, False)
    assert value["verdict_class"] == "null" and value["diagnosis_ready_score"] == 1
    assert value["independent_generalization_score"] == 0
    assert value["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert value["MODEL_SPECS"] == [] and value["model_specs"] == []
    assert all(n == 0 for n in value["model_invocation_counts"].values())


@pytest.mark.parametrize("coverage", [False, True])
def test_nonfixture_validation_pipeline(tmp_path, monkeypatch, coverage):
    """SCENARIO-REPORT-8958-CONTRACT: publication consumes actual owned-check receipts."""
    from carnot.reporting import confidence_diagnosis_runner_8958 as runner
    from carnot.reporting.current_work_receipt import canonical_hash

    missing = d.authenticate(tmp_path)
    monkeypatch.setattr(d, "authenticate", lambda _: missing)
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "freeze", lambda *_: deepcopy(work))
    monkeypatch.setattr(runner, "replay", lambda _: dict(passed=True))
    monkeypatch.setattr(runner.calibration, "terminal_commands", lambda _: [])
    monkeypatch.setattr(d.admission.shard_a.pilot, "terminal_passed", lambda _: True)
    private = tmp_path / "private"
    private.mkdir()
    if coverage:
        (private / ".coverage").touch()
    monkeypatch.setattr(runner.tempfile, "mkdtemp", lambda **_: str(private))
    monkeypatch.setattr(runner.shared, "validation_plan", lambda *a, **k: [])
    monkeypatch.setattr(runner, "run_commands", lambda *a, **k: [])
    monkeypatch.setattr(runner, "measured", lambda *a: dict(percent=100))
    dev, truth = tmp_path / "dev.json", tmp_path / "truth.json"
    atomic_json(dev, dict(rows=[]))
    atomic_json(truth, dict(rows=[]))
    work = dict(
        evidence=missing,
        result=d.empty_result(),
        input_references=[],
        sources=[],
        developer_view=d.calibration.reference(dev),
        protected_label_view=d.calibration.reference(truth),
        raw_references=[d.calibration.reference(dev)],
        duration_s=0,
        phase_spans=[dict(phase="fixture", duration_s=0)],
        global_test_debt={},
    )
    assert runner.main(["--date", "20261109"]) == 0
    value = json.loads((tmp_path / "results" / (d.NAME + ".json")).read_text())
    assert value["verdict_class"] == "disqualified"
    assert value["global_test_debt"]["repeated"] is False
    assert value["reproducibility_checksum"] == d.calibration.checksum(value)
    assert canonical_hash(value)


@pytest.mark.parametrize(
    "fault", ["missing", "bytes", "executed", "measurement", "view", "permissions"]
)
def test_replay_refusals(tmp_path, monkeypatch, fault):
    """SCENARIO-REPORT-8958-CONTRACT: replay rejects changed evidence, code and view custody."""
    from carnot.reporting import confidence_diagnosis_runner_8958 as runner

    path = tmp_path / "primary.json"
    if fault == "missing":
        assert not runner.replay(path)["passed"]
        return
    raw = tmp_path / "raw"
    raw.mkdir()
    measurement = raw / "measurement.json"
    atomic_json(measurement, {})
    dev, label = raw / "developer.json", raw / "labels.json"
    atomic_json(dev, dict(rows=["changed"] if fault == "view" else []))
    atomic_json(label, dict(rows=[]))
    label.chmod(0o644 if fault == "permissions" else 0o600)
    result = d.empty_result()
    evidence = dict(failures=[])
    work = dict(
        evidence=dict(changed=True) if fault == "measurement" else evidence,
        result=result,
        raw_references=[],
    )
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", {})
    ref = d.calibration.reference(measurement)
    if fault == "bytes":
        ref["sha256"] = "changed"
    if fault == "executed":
        other = raw / "other.py"
        other.write_text("changed")
        ref.update(executed_source=True, source_path=str(other))
    value = dict(
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[ref],
        developer_view=d.calibration.reference(dev),
        protected_label_view=d.calibration.reference(label),
    )
    atomic_json(path, value)
    monkeypatch.setattr(d, "authenticate", lambda _: evidence)
    monkeypatch.setattr(d, "diagnose", lambda _: result)
    assert not runner.replay(path)["passed"]
