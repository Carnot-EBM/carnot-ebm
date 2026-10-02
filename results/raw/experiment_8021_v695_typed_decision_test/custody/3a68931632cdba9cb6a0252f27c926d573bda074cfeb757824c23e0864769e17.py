"""REQ-REPORT-8021: decision gates use sealed probabilities and known labels."""

import copy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot import experiment_8021_v695_typed_decision_test as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(headroom=True, n=256):
    """Separate artificial targets exercise gates without natural evidence."""
    rows, targets = [], []
    for i in range(n):
        y = i % 2
        targets.append(dict(family_id=str(i), eligible_y=y))
        for arm in ("conditioned_energy", "linear", "sigmoid_identity"):
            p = float(1 - y) if headroom and arm == "linear" else float(y)
            rows.append(
                dict(
                    family_id=str(i),
                    source_cluster_id=str(i),
                    role="stream",
                    arm=arm,
                    seed=17,
                    condition="posthoc",
                    p=p,
                    y=None,
                    exclusion_reason=None,
                    failure_reason=None,
                    censor_reason=None,
                )
            )
    return rows, targets


def test_cost_and_unknown():
    """SCENARIO-REPORT-8021-REDUCTION: ties escalate and unknowns are unscored."""
    assert [e.action(p) for p in (None, 0, 0.1, 0.5, 1)] == [
        "escalate",
        "accept",
        "escalate",
        "escalate",
        "reject",
    ]
    assert e.loss("accept", 1) == 5 and e.loss("reject", 0) == 1
    assert e.loss("escalate", 0) == 0.5 and e.loss("accept", None) is None
    rows, targets = fixture()
    targets[0]["eligible_y"] = None
    result = e.reduce(rows, targets, "linear/posthoc")
    assert result["sample_size_budget"]["eligible"] == 255
    assert result["rows"][0]["actual_cost"] is None
    for bad in (-1, float("nan"), 2):
        with pytest.raises(ValueError, match="probability"):
            e.action(bad)
    targets[0]["eligible_y"] = 2
    with pytest.raises(ValueError, match="target_contract"):
        e.reduce(rows, targets, "linear/posthoc")


def test_controls_and_support():
    """REQ-REPORT-8021: a valid null is ready science, without benefit credit."""
    for room in (True, False):
        rows, targets = fixture(room)
        result = e.reduce(rows, targets, "linear/posthoc")
        assert result["decision_benefit_score"] == int(room)
        assert result["acceptance_gate_results"]["support"]
        assert result["adjusted_intervals"]["holm_adjusted_p"] <= 1
    rows, targets = fixture(n=8)
    result = e.reduce(rows, targets, "linear/posthoc")
    assert not result["acceptance_gate_results"]["support"]
    empty_targets = [dict(r, eligible_y=None) for r in targets]
    assert e.reduce(rows, empty_targets, "linear/posthoc")["sample_size_budget"]["eligible"] == 0
    rows[0]["p"] = None
    result = e.reduce(rows, targets, "linear/posthoc")
    assert result["rows"][0]["denominator"] == 0
    targets.pop()
    with pytest.raises(ValueError, match="target_roster"):
        e.reduce(rows, targets, "linear/posthoc")


def test_tune_selection_and_seal():
    """REQ-REPORT-8021: tune selection excludes the duplicate identity arm."""
    rows, targets = fixture()
    tune = [dict(r, role="tune", y=int(r["family_id"]) % 2) for r in rows]
    assert e.select_comparator(tune) == "linear/posthoc"
    assert e.verify_seal(rows, canonical_hash(rows))
    with pytest.raises(ValueError, match="prediction_seal"):
        e.verify_seal(rows, "sha256:wrong")
    with pytest.raises(ValueError, match="comparator_support"):
        e.select_comparator([])


def source_fixture(root):
    """Private upstream snapshots let custody checks run without live labels."""
    rows, targets = fixture(False)
    public = [
        dict(
            family_id=str(i),
            source_cluster_id=str(i),
            public_eligible=True,
            q=float(i % 2),
            features=[0.0] * 8,
            status="completed",
        )
        for i in range(256)
    ]

    def save(name, value):
        p = root / (name + ".json")
        atomic_json(p, value)
        return e.reference(p)

    measurement = save(
        "measurement",
        dict(
            primitive_rows=rows + [dict(r, role="tune", y=int(r["family_id"]) % 2) for r in rows],
            public_prediction_seals=dict(stream=canonical_hash(rows)),
        ),
    )
    upstream = dict(
        experiment_id=8020,
        task_id=e.prior.TASK,
        verdict_class="null",
        energy_fit_ready_score=1,
        measurement_checkpoint=measurement,
        public_prediction_seals=dict(stream=canonical_hash(rows)),
        head_checkpoints=[],
        calibration_checkpoint=save("calibration", {}),
        trained_head_specs=[],
    )
    fit = root / "results" / (e.prior.NAME + ".json")
    atomic_json(fit, upstream)
    stream = save(
        "labels",
        dict(
            rows=targets,
            annotation_rows=[dict(family_id=str(i)) for i in range(1, 256, 2)],
            role="stream",
            access_policy="evaluator_only",
        ),
    )
    eligible = dict(
        experiment_id=8019,
        task_id=e.prior.eligible.TASK,
        stream_targets_ready_score=1,
        verdict_class="null",
        public_manifests=dict(stream=save("public", dict(rows=public))),
        evaluator_manifests=dict(stream=stream),
        exclusion_manifest=save("exclusions", {}),
        schema_manifests=dict(evaluator=save("schema", {})),
        exposure_rows=[],
        exposure_scope="historical development",
    )
    ep = root / "results" / (e.prior.eligible.NAME + ".json")
    atomic_json(ep, eligible)
    atomic_json(
        root / "results" / "experiment_7955_v690_response_targets.json",
        dict(
            target_definition=dict(
                primary="any authenticated human source-unsupported span anywhere in the complete response"
            )
        ),
    )
    atomic_json(
        root / "results" / "experiment_8006_v694_independent_replay.json", dict(experiment_id=8006)
    )
    eligible["historical_failure_logs"] = []
    atomic_json(ep, eligible)
    return fit, ep


def test_source_custody(tmp_path, monkeypatch):
    """REQ-REPORT-8021: byte tampering and orientation errors are terminal blocks."""
    fit, ep = source_fixture(tmp_path)
    monkeypatch.setattr(e.prior, "replay", lambda path: dict(passed=True))
    data, failures = e.load_inputs(tmp_path, tmp_path / "raw")
    assert not failures and len(data["predictions"]) == 768
    seal = json.loads(Path(data["prediction_seal"]["path"]).read_text())
    access = json.loads(Path(data["label_access_receipt"]["path"]).read_text())
    assert access["stream_access_monotonic_ns"] > seal["invocation_seal_checked_monotonic_ns"]
    assert not access["retention_labels_opened"]
    v = json.loads(fit.read_text())
    v["public_prediction_seals"]["stream"] = "bad"
    atomic_json(fit, v)
    assert e.load_inputs(tmp_path, tmp_path / "tamper")[1]
    source_fixture(tmp_path)
    eligible = json.loads(ep.read_text())
    p = Path(eligible["evaluator_manifests"]["stream"]["path"])
    labels = json.loads(p.read_text())
    labels["rows"][0]["eligible_y"] = 1
    atomic_json(p, labels)
    eligible["evaluator_manifests"]["stream"] = e.reference(p)
    atomic_json(ep, eligible)
    assert (
        e.load_inputs(tmp_path, tmp_path / "orientation")[1][0]["observed"]
        == "source_target_orientation"
    )


@pytest.mark.parametrize("mutation", ["seal_receipt", "roster", "source_id", "schema"])
def test_source_contract_mutations(tmp_path, monkeypatch, mutation):
    """REQ-REPORT-8021: each original byte-bound contract rejects drift."""
    fit, ep = source_fixture(tmp_path)
    monkeypatch.setattr(e.prior, "replay", lambda path: dict(passed=True))
    f, v = json.loads(fit.read_text()), json.loads(ep.read_text())
    if mutation == "seal_receipt":
        ref = f["measurement_checkpoint"]
    elif mutation == "schema":
        ref = e.reference(tmp_path / "results" / "experiment_7955_v690_response_targets.json")
    else:
        ref = v["public_manifests"]["stream"]
    p = Path(ref["path"])
    value = json.loads(p.read_text())
    if mutation == "seal_receipt":
        value["public_prediction_seals"]["stream"] = "bad"
    elif mutation == "roster":
        value["rows"].pop()
    elif mutation == "schema":
        value["target_definition"]["primary"] = "wrong_orientation"
    else:
        value["rows"][0]["source_cluster_id"] = "wrong_id"
    atomic_json(p, value)
    if mutation == "seal_receipt":
        f["measurement_checkpoint"] = e.reference(p)
        atomic_json(fit, f)
    elif mutation != "schema":
        v["public_manifests"]["stream"] = e.reference(p)
        atomic_json(ep, v)
    assert e.load_inputs(tmp_path, tmp_path / "raw")[1]


def test_missing_support_and_coverage(tmp_path):
    """REQ-REPORT-8021: missing contracts and empty coverage cannot qualify."""
    assert e.coverage_counts(tmp_path) == {}
    atomic_json(
        tmp_path / "coverage.json",
        dict(files={"new": dict(summary=dict(num_statements=1, missing_lines=0))}),
    )
    assert e.coverage_counts(tmp_path)["new"]["missing_lines"] == 0
    fit, _ = source_fixture(tmp_path)
    v = json.loads(fit.read_text())
    v.pop("energy_fit_ready_score")
    atomic_json(fit, v)
    assert e.load_inputs(tmp_path, tmp_path / "raw")[1][0]["observed"] == "MISSING_CONTRACT_FIELD"
    rows, targets = fixture(False)
    rows += [dict(r, seed=29) for r in rows]
    assert e.reduce(rows, targets, "linear/posthoc")["sample_size_budget"]["independent"] == 256


def prefix():
    """Direct script coverage uses the private configuration from the runner."""
    py = str(e.ROOT / ".venv/bin/python")
    config = os.environ.get("CARNOT_8021_COVERAGE_CONFIG")
    return ([py, "-m", "coverage", "run", "--rcfile=" + config] if config else [py, "-u"]) + [
        str(e.ROOT / e.OWNED[-1])
    ]


def test_cli_and_replay(tmp_path):
    """SCENARIO-REPORT-8021-REDUCTION: real CLI publishes and cold reduces."""
    rows, targets = fixture()
    src = tmp_path / "fixture.json"
    atomic_json(src, dict(predictions=rows, targets=targets, comparator="linear/posthoc"))
    out = tmp_path / (e.NAME + ".json")
    for extra in (["--fixture-input", str(src)], ["--root", str(tmp_path / "absent")]):
        run = subprocess.run(
            prefix() + ["--output", str(out), "--validation-worker", *extra],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        assert e.replay(out)["passed"]
        v = json.loads(out.read_text())
        assert v["decision_measurement_ready_score"] == 0
        cold = subprocess.run(
            prefix() + ["--cold-replay", str(out)], capture_output=True, text=True, timeout=60
        )
        assert cold.returncode == 0, cold.stdout + cold.stderr
        if extra[0] == "--fixture-input":
            assert v["verdict_class"] == "circular_positive"
            tampered = copy.deepcopy(v)
            tampered["arm_metrics"] = {}
            atomic_json(tmp_path / "tampered.json", tampered)
            with pytest.raises(ValueError, match="reduction"):
                e.replay(tmp_path / "tampered.json")
    bad = subprocess.run(
        prefix() + ["--date", "20261003"], capture_output=True, text=True, timeout=30
    )
    assert bad.returncode == 2
    bad = subprocess.run(
        prefix() + ["--cold-replay", str(tmp_path / "absent")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert bad.returncode == 1


def test_owned_main_branches(tmp_path, monkeypatch):
    """REQ-REPORT-8021: owned checks gate readiness independently of benefit."""
    monkeypatch.setattr(e, "terminal", e.replay)
    monkeypatch.setattr(e, "validation_plan", lambda scratch: [])
    for room, passed in ((False, True), (True, True), (True, False)):
        rows, targets = fixture(room)
        monkeypatch.setattr(
            e,
            "load_inputs",
            lambda root, raw: (
                dict(predictions=rows, targets=targets, comparator="linear/posthoc", references=[]),
                [],
            ),
        )

        def commands(*args, **kwargs):
            return [dict(scope="owned", passed=passed, command_argv=[], exit_code=int(not passed))]

        monkeypatch.setattr(e, "run_commands", commands)
        monkeypatch.setattr(
            e, "coverage_counts", lambda scratch: {"owned": dict(num_statements=1, missing_lines=0)}
        )
        out = tmp_path / (e.NAME + ".json")
        assert e.main(["--output", str(out)]) == 0
        v = json.loads(out.read_text())
        assert v["decision_measurement_ready_score"] == int(passed)
        assert v["verdict_class"] == (
            ("positive" if room else "null") if passed else "disqualified"
        )
    assert e.coverage_counts(tmp_path) == {"owned": dict(num_statements=1, missing_lines=0)}


def test_terminal_and_unsafe_readers(tmp_path, monkeypatch):
    """REQ-REPORT-8021: candidate readers and final identity fail closed."""
    out = tmp_path / (e.NAME + ".json")
    monkeypatch.setattr(e, "run_commands", lambda *a, **k: [dict(passed=True)])
    atomic_json(tmp_path / "terminal.json", {})
    assert e.terminal(tmp_path / "terminal.json")["passed"]
    rows, targets = fixture(False)
    src = tmp_path / "fixture.json"
    atomic_json(src, dict(predictions=rows, targets=targets, comparator="linear/posthoc"))
    assert e.main(["--fixture-input", str(src), "--output", str(out), "--validation-worker"]) == 0
    v = json.loads(out.read_text())
    v["decision_measurement_ready_score"] = 1
    atomic_json(tmp_path / "unsafe.json", v)
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(tmp_path / "unsafe.json")
    v["decision_measurement_ready_score"] = 0
    v["acceptance_gate_results"]["support"] = False
    atomic_json(tmp_path / "gate.json", v)
    with pytest.raises(ValueError, match="reduction"):
        e.replay(tmp_path / "gate.json")
    v["acceptance_gate_results"]["support"] = True
    v["verdict_class"] = "disqualified"
    v["decision_benefit_score"] = 1
    atomic_json(tmp_path / "benefit.json", v)
    with pytest.raises(ValueError, match="reduction|unsafe_benefit"):
        e.replay(tmp_path / "benefit.json")
    v["positive_control_results"]["known_headroom"]["decision_benefit_score"] = 0
    atomic_json(tmp_path / "control.json", v)
    with pytest.raises(ValueError, match="control_reduction"):
        e.replay(tmp_path / "control.json")
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **k: dict(passed=False))
    assert (
        e.main(["--root", str(tmp_path / "absent"), "--output", str(out), "--validation-worker"])
        == 1
    )
