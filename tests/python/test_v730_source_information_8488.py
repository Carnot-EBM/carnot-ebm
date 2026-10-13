"""REQ-REPORT-8488, REQ-VERIFY-8488: fixed information contrasts need sealed operands."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from carnot.reporting import v730_source_information_audit as e
from carnot.reporting import v730_source_information_runner as runner
from carnot.verify import v730_information_metrics as n
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    """Real cached features exercise joins without a synthetic scientific result."""
    raw = tmp_path_factory.mktemp("source-audit")
    return e.measure(e.ROOT, raw), raw


def test_question_contrasts_and_fixed_losses():
    """SCENARIO-REPORT-8488-PAIR: siblings and repeated seeds average before bootstrap."""
    rows = [
        dict(
            unit_id=f"a{i}",
            question_hash=f"q{i // 2}",
            arm=arm,
            seed=seed,
            probability=p,
            y=0,
            y_sensitivity=0,
        )
        for i in range(4)
        for seed in n.SEEDS
        for arm, p in [(n.FULL, 0.1), (n.SCALAR, 0.2), (n.PERMUTED, 0.3)]
    ]
    result = n.contrast(rows, "y")
    assert result["independent_questions"] == 2
    assert result["contrasts"][n.SCALAR] == pytest.approx(0.03)
    assert result["contrasts"][n.PERMUTED] == pytest.approx(0.08)
    assert result["draws"] == 10000 and result["seed"] == 7278446
    assert result["confidence_level"] == 0.975
    assert result["familywise_intervals"][n.SCALAR] == pytest.approx([0.03, 0.03])
    assert [n.shipped.loss(p, y) for p, y in [(0.1, 1), (0.9, 0), (0.1, 0), (0.5, 1)]] == [
        1,
        1,
        0,
        0.2,
    ]
    rows[0]["probability"] = None
    assert n.contrast(rows, "y")["independent_questions"] == 1
    assert n.contrast([], "y")["contrasts"][n.SCALAR] is None


def test_real_seal_denominators_and_qualification(measured):
    """REQ-REPORT-8488: independent producer seals qualify all intended predictions."""
    work, raw = measured
    assert not work["failures"] and not work["owned_failure"]
    assert work["clock"]["predictions_sealed_ns"] < work["clock"]["targets_opened_ns"]
    result = work["result"]
    assert len(result["rows"]) == 7500
    assert result["bootstrap"]["independent_questions"] == 100
    assert set(result["A1_metrics"]) == set(n.ARMS)
    assert set(result["per_seed_metrics"]) == {str(seed) for seed in n.SEEDS}
    assert all(s["intended_prediction_count"] == 1500 for s in result["A1_metrics"].values())
    assert any(s["missing_target_count"] > 0 for s in result["A2_sensitivity"].values())
    value = e.build(work, [dict(passed=True)], raw, raw / (e.NAME + ".json"))
    assert value["source_audit_ready_score"] == 1
    assert value["independent_count"] == 100 and value["intended_count"] == 7500
    assert value["historical_v726_gain"]["gain"] == 0.01969866346324697
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert all(d["decision_flip_rate"] == 0 for d in value["repeat_fit_audit"].values())


def test_predictor_invariance(measured):
    """SCENARIO-REPORT-8488-INVARIANCE: metadata and order do not enter numerical prediction."""
    _, panel, frozen, features = e.operands(e.ROOT)
    original = e.predict_rows(frozen, features["rows"], panel["stratum_roster"])
    renamed = deepcopy(features["rows"])
    for row in renamed:
        row["answer_id"] = "renamed:" + row["answer_id"]
        row["unit_id"] = "renamed:" + row["unit_id"]
    strata = [dict(r, unit_id="renamed:" + r["unit_id"]) for r in panel["stratum_roster"]]
    changed = e.predict_rows(frozen, list(reversed(renamed)), strata)
    key = lambda r: (r["unit_id"], r["arm"], r["seed"])
    a = {key(r): r["probability"] for r in original}
    b = {
        (r["unit_id"].removeprefix("renamed:"), r["arm"], r["seed"]): r["probability"]
        for r in changed
    }
    assert a == b
    assert np.asarray(list(a.values())).tobytes() == np.asarray([b[k] for k in a]).tobytes()


def test_missing_and_owned_failures(measured, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8488-VERDICT: external absence differs from failed owned work."""
    work, raw = measured
    value = e.build(work, [dict(passed=False, name="failed")], raw, raw / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified" and value["source_audit_ready_score"] == 0
    missing = e.measure(tmp_path / "absent", tmp_path / "missing")
    value = e.build(
        missing, [dict(passed=True)], tmp_path / "missing", tmp_path / (e.NAME + ".json")
    )
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 7500
    assert any(g["observed"] is None for g in value["gate_check_summary"])
    monkeypatch.setattr(n, "evaluate", lambda *_: (_ for _ in ()).throw(ValueError("owned")))
    failed = e.measure(e.ROOT, tmp_path / "failed")
    assert failed["owned_failure"]


def test_private_cli_and_controls(tmp_path):
    """SCENARIO-VERIFY-8488-CLI: genuine children publish and reject invalid invocations."""
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for i, (args, expected) in enumerate(
        [
            (["--date", "20261013", "--private-control", "--output", str(output)], 0),
            (["--cold-replay", str(output)], 0),
            (["--cold-replay", str(tmp_path / "absent")], 1),
            (["--deliberate-error"], 7),
            (["--date", "20261012"], 2),
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


def test_metrics_missing_and_rejection_paths():
    """SCENARIO-REPORT-8488-PAIR: invalid probabilities fail; missing labels keep denominators."""
    predictions = [
        dict(
            unit_id="a",
            question_hash="q",
            dataset="d",
            generator="g",
            arm=a,
            seed=s,
            probability=None,
            missing_reason="upstream_missing",
        )
        for a in n.ARMS
        for s in n.SEEDS
    ]
    empty = n.evaluate(predictions, [])
    assert empty["bootstrap"]["independent_questions"] == 0
    assert all(s["deterministic_duplicate_rate"] == 0.8 for s in empty["repeat_fit_audit"].values())
    assert n.matched_risk([], 0)["risk"] is None
    assert all(s["decision_flip_rate"] is None for s in empty["repeat_fit_audit"].values())
    with pytest.raises(ValueError, match="duplicate"):
        n.evaluate([*predictions, predictions[0]], [])
    with pytest.raises(ValueError, match="duplicate"):
        n.evaluate(predictions, [dict(unit_id="a", y=0), dict(unit_id="a", y=0)])
    predictions[0]["probability"] = float("nan")
    with pytest.raises(ValueError, match="invalid_probability"):
        n.evaluate(predictions, [])
    risk = n.matched_risk([dict(probability=0.1, y=0), dict(probability=0.9, y=0)], 1)
    assert risk["risk"] == 0.5


def test_forged_qualification_is_rejected(monkeypatch):
    """REQ-VERIFY-8488: a passing original file cannot replace a failed live consumer."""
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="head_controls_not_qualified"):
        e.operands(e.ROOT)


def test_inherited_predictor_method_pin(monkeypatch):
    """REQ-VERIFY-8488: current predictor bytes must match the sealed head producer."""
    original = e.sha256_file
    monkeypatch.setattr(
        e,
        "sha256_file",
        lambda p: "sha256:forged" if str(p).endswith("v730_source_control_fit.py") else original(p),
    )
    with pytest.raises(ValueError, match="head_predictor_method_drift"):
        e.operands(e.ROOT)


def test_manifest_exact_union(tmp_path):
    """REQ-VERIFY-8488: every collected historical assertion executes once."""
    plan = runner.manifest(tmp_path)
    for collection in [p for p in plan if p["name"].startswith("collect_")]:
        nodes = collection["collected_node_ids"]
        if nodes[0].startswith(e.TEST):
            continue
        actual = [
            n
            for p in plan
            if not p["name"].startswith("collect_")
            for n in p.get("collected_node_ids", [])
            if n.split("::")[0] == nodes[0].split("::")[0]
        ]
        assert actual == nodes and len(actual) == len(set(actual))


@pytest.mark.parametrize("fault", ["prediction", "target", "summary", "pin", "seal", "historical"])
def test_self_consistently_rehashed_tampering(fault, measured, tmp_path):
    """SCENARIO-VERIFY-8488-CLI: independent operands defeat repaired envelope hashes."""
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
        work["result"]["contrasts"][n.SCALAR] += 0.01
    if fault == "pin":
        work["upstream_reference"]["sha256"] = "sha256:forged"
    if fault == "seal":
        work["clock"]["predictions_sealed_ns"] = work["clock"]["targets_opened_ns"] + 1
    if fault == "historical":
        work["historical_v726_gain"]["gain"] += 0.01
    work["primitive_files"] = [
        e.reference(raw / Path(ref["path"]).name) for ref in work["primitive_files"]
    ]
    work["prediction_reference"] = e.reference(raw / "sealed_predictions.json")
    receipt = child("proof", [sys.executable, "-c", "print('independent log')"], raw / "logs")
    atomic_json(raw / "measurement.json", work)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.build(work, [receipt], raw, output))
    assert not e.replay(output)
