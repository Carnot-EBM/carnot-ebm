"""REQ-REPORT-8529, REQ-VERIFY-8529: test registered source gains and sealed operands."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import v733_incremental_source_audit as e
from carnot.reporting import v733_incremental_source_runner as runner
from carnot.verify import v733_incremental_source_metrics as n
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Real sealed predictions keep synthetic controls out of the scientific artifact."""
    raw = tmp_path_factory.mktemp("incremental-source")
    return e.measure(e.ROOT, raw), raw


def panel():
    """Two questions with siblings and repeated seeds expose incorrect independence."""
    return [
        dict(
            unit_id=f"a{i}",
            question_hash=f"q{i // 2}",
            arm=arm,
            seed=seed,
            probability=p,
            y=0,
            y_sensitivity=0,
            dataset="d",
            generator="g",
        )
        for i in range(4)
        for seed in n.SEEDS
        for arm, p in zip(n.ARMS, [0.2, 0.5, 0.2, 0.1, 0.3], strict=True)
    ]


def test_registered_question_losses():
    """SCENARIO-REPORT-8529-PAIR: average siblings and seeds before shared draws."""
    rows = panel()
    result = n.contrast(rows, "y")
    assert result["independent_questions"] == 2
    assert result["draws"] == 10000 and result["seed"] == 7318502
    assert result["confidence_level"] == 0.983333333333
    assert result["family_size"] == 3
    assert result["contrasts"][n.RAW] == pytest.approx(0.03)
    assert result["contrasts"][n.SCALAR] == pytest.approx(0.03)
    assert result["contrasts"][n.PERMUTED] == pytest.approx(0.08)
    assert result["familywise_intervals"][n.RAW] == pytest.approx([0.03, 0.03])
    assert set(result["paired_question_rows"][0]["arm_losses"]) == set(n.ARMS)
    assert result["paired_question_rows"][0]["sibling_seed_count"] == 10
    rows[0]["probability"] = None
    assert n.contrast(rows, "y")["independent_questions"] == 1
    assert n.contrast([], "y")["contrasts"][n.RAW] is None


def test_strong_controls_and_degeneracy():
    """REQ-REPORT-8529: weak legacy gains and identical controls cannot establish benefit."""
    rows = panel()
    predictions = [{k: v for k, v in r.items() if k not in {"y", "y_sensitivity"}} for r in rows]
    targets = [dict(unit_id=f"a{i}", y=0, y_sensitivity=0) for i in range(4)]
    value = n.evaluate(predictions, targets)
    assert value["exploratory_benefit_score"] == 1
    assert value["class_support"]["A1"]["0"]["questions"] == 2
    for r in predictions:
        if r["arm"] == n.FULL:
            r["probability"] = 0.2
    value = n.evaluate(predictions, targets)
    assert value["exploratory_benefit_score"] == 0
    assert value["negative_control_degeneracy"][n.RAW]["identical_predictions"]
    for r in predictions:
        r["probability"] = None
    assert n.evaluate(predictions, [])["bootstrap"]["independent_questions"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        n.evaluate([*predictions, predictions[0]], [])
    predictions[0]["probability"] = float("nan")
    with pytest.raises(ValueError, match="invalid_probability"):
        n.evaluate(predictions, [])


def test_real_seals_and_readiness(measured):
    """REQ-REPORT-8529: a qualified null has readiness without benefit or generalization."""
    work, raw = measured
    assert not work["failures"] and not work["owned_failure"]
    assert work["clock"]["predictions_sealed_ns"] < work["clock"]["targets_opened_ns"]
    assert work["result"]["bootstrap"]["independent_questions"] == 100
    assert len(work["result"]["rows"]) == 7500
    value = e.build(work, [dict(passed=True)], raw, raw / (e.NAME + ".json"))
    assert value["source_audit_ready_score"] == 1
    assert value["primary_contrasts"] == e.methods.protocol()["primary_contrasts"]
    assert len(value["primary_contrast_rows"]) == 3
    assert value["independent_count"] == 100
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert set(value["familywise_intervals"]) == set(n.CONTROLS)
    assert value["next_evidence_condition"]
    assert value["clipping"]["probability_clipping_applied"] is False


def test_missing_owned_failures_and_cache(measured, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8529-VERDICT: retain missing slots and separate failure ownership."""
    work, raw = measured
    failed = e.build(work, [dict(passed=False, name="failed")], raw, raw / (e.NAME + ".json"))
    assert failed["verdict_class"] == "disqualified" and failed["source_audit_ready_score"] == 0
    missing = e.measure(tmp_path / "absent", tmp_path / "missing")
    value = e.build(
        missing, [dict(passed=True)], tmp_path / "missing", tmp_path / (e.NAME + ".json")
    )
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 7500
    assert any(g["observed"] is None for g in value["gate_check_summary"])
    monkeypatch.setattr(n, "evaluate", lambda *_: (_ for _ in ()).throw(ValueError("owned")))
    assert e.measure(e.ROOT, tmp_path / "owned")["owned_failure"]


def test_forged_upstream_and_registration(monkeypatch):
    """REQ-VERIFY-8529: authentic bytes cannot replace a failed live qualification."""
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="identity_heads_unqualified"):
        e.operands(e.ROOT)


def test_private_cli_and_controls(tmp_path):
    """SCENARIO-VERIFY-8529-CONTROLS: real children exercise publication and rejections."""
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for i, (args, expected) in enumerate(
        [
            (["--date", "20261015", "--private-control", "--output", str(output)], 0),
            (["--cold-replay", str(output)], 0),
            (["--cold-replay", str(tmp_path / "absent")], 1),
            (["--deliberate-error"], 7),
            (["--date", "20261014"], 2),
            (["--private-control"], 2),
            (
                [
                    "--private-control",
                    "--root",
                    str(tmp_path / "absent"),
                    "--output",
                    str(tmp_path / "blocked" / (e.NAME + ".json")),
                ],
                0,
            ),
        ]
    ):
        receipt = child(f"cli-{i}", argv + args, tmp_path / "logs", expected=expected, deadline=180)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["source_audit_ready_score"] == 1
    value["contrasts"][n.SCALAR] = 999
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)


def test_manifest_exact_union(tmp_path):
    """REQ-VERIFY-8529: original collected assertions execute once in bounded shards."""
    plan = runner.manifest(tmp_path)
    for collection in [p for p in plan if p["name"].startswith("collect_")]:
        nodes = collection["collected_node_ids"]
        if nodes[0].startswith(e.TEST):
            continue
        actual = [
            node
            for p in plan
            if not p["name"].startswith("collect_")
            for node in p.get("collected_node_ids", [])
            if node.split("::")[0] == nodes[0].split("::")[0]
        ]
        assert actual == nodes and len(actual) == len(set(actual))


def test_independent_method_and_seal_rejections(monkeypatch):
    """REQ-VERIFY-8529: altered code, registration and seal clocks reject authentic envelopes."""
    original_hash = e.sha256_file
    monkeypatch.setattr(
        e,
        "sha256_file",
        lambda p: (
            "sha256:forged" if str(p).endswith("v733_identity_brier.py") else original_hash(p)
        ),
    )
    with pytest.raises(ValueError, match="original_predictor_method_drift"):
        e.operands(e.ROOT)
    monkeypatch.setattr(e, "sha256_file", original_hash)
    original_read = e.read

    def changed(ref):
        value = deepcopy(original_read(ref))
        if ref["path"].endswith(e.UPSTREAM):
            value["seal_clock"]["evaluator_opened"] = True
        return value

    monkeypatch.setattr(e, "read", changed)
    with pytest.raises(ValueError, match="prediction_seal_drift"):
        e.operands(e.ROOT)
    monkeypatch.setattr(e, "read", original_read)
    protocol = deepcopy(e.methods.protocol())
    protocol["primary_contrasts"]["bootstrap_seed"] += 1
    monkeypatch.setattr(e.methods, "protocol", lambda: protocol)
    with pytest.raises(ValueError, match="primary_contrasts_registration_drift"):
        e.operands(e.ROOT)


def test_original_authority_rejection(tmp_path, monkeypatch):
    """REQ-VERIFY-8529: private original-test authority cannot be locally replaced."""
    monkeypatch.setattr(runner, "sha256_file", lambda _: "sha256:forged")
    with pytest.raises(ValueError, match="original_audit_authority"):
        runner.original_workspace(tmp_path)


@pytest.mark.parametrize(
    "fault", ["prediction", "target", "summary", "pin", "seal", "registration", "diagnostics"]
)
def test_rehashed_tampering(fault, measured, tmp_path):
    """SCENARIO-VERIFY-8529-CONTROLS: local repaired hashes cannot replace independent seals."""
    source, _ = measured
    work = deepcopy(source)
    raw = tmp_path / "raw"
    raw.mkdir()
    for ref in work["primitive_files"]:
        (raw / Path(ref["path"]).name).write_bytes(Path(ref["path"]).read_bytes())
    if fault == "prediction":
        value = json.loads((raw / "sealed_predictions.json").read_bytes())
        value["rows"][0]["probability"] += 0.01
        atomic_json(raw / "sealed_predictions.json", value)
    if fault == "target":
        value = json.loads((raw / "evaluator_targets.json").read_bytes())
        value["rows"][0]["y"] = 1 - value["rows"][0]["y"]
        atomic_json(raw / "evaluator_targets.json", value)
    if fault == "summary":
        work["result"]["contrasts"][n.RAW] += 0.01
    if fault == "pin":
        work["upstream_reference"]["sha256"] = "sha256:forged"
    if fault == "seal":
        work["clock"]["predictions_sealed_ns"] = work["clock"]["targets_opened_ns"] + 1
    if fault == "registration":
        work["primary_contrasts"]["bootstrap_seed"] += 1
    if fault == "diagnostics":
        work["upstream_diagnostics"]["seal_clock"]["evaluator_opened"] = True
    work["primitive_files"] = [
        e.reference(raw / Path(r["path"]).name) for r in work["primitive_files"]
    ]
    work["prediction_reference"] = e.reference(raw / "sealed_predictions.json")
    receipt = child("proof", [sys.executable, "-c", "print('independent log')"], raw / "logs")
    atomic_json(raw / "measurement.json", work)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work, [receipt], raw, output))
    assert not e.replay(output)
