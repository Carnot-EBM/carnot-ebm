"""REQ-REPORT-8074: private evidence tests cannot earn scientific credit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from carnot import experiment_8074_v699_interaction_decision_audit as m


def artificial(mode="benefit", n=96):
    """Known probabilities test statistical sensitivity without fitting to labels."""
    predictions, targets = [], []
    for i in range(n):
        y = i % 2
        targets.append(dict(family_id=str(i), y=y, exclusion_reason=None))
        for arm in m.ARMS:
            p = (0.9 if y else 0.01) if arm == "interaction" else 0.3
            if mode == "tie":
                p = 0.3
            if mode == "unsafe" and arm == "interaction":
                p = 0.01
            predictions.append(
                dict(
                    unit=str(i),
                    source=str(i),
                    family_id=str(i),
                    arm=arm,
                    seed=m.SEED,
                    condition="evaluation",
                    probability=p,
                    decision=m.action(p),
                    status="completed",
                    exclusion_reason=None,
                )
            )
    return predictions, targets


def test_margin_bootstrap_and_safety():
    """SCENARIO-REPORT-8074-H1: real benefit, null and unsafe outcomes differ."""
    rows, targets = artificial()
    result = m.reduce(rows, targets)
    h = result["H1"]
    assert h["support_passed"] and h["safety_passed"]
    assert h["observed_gain"] == 0.5 and h["beneficial_changed_sources"] == 96
    assert h["raw_p_value"] == 1 / 10001 and h["family_p_value"] == h["raw_p_value"]
    assert len(result["descriptive_arm_metrics"]) == 5
    assert len(result["bootstrap_draws"]) == 10000
    gains = np.array([r["gain"] for r in result["paired_source_rows"]])
    rng = np.random.default_rng(m.SEED)
    expected = gains[rng.integers(0, len(gains), (10000, len(gains)))].mean(axis=1)
    assert np.array_equal(expected, result["bootstrap_draws"])
    assert m.reduce(*artificial("tie"))["H1"]["raw_p_value"] == 1
    unsafe = m.reduce(*artificial("unsafe"))["H1"]
    assert not unsafe["safety_passed"] and unsafe["family_p_value"] == 1
    unsupported = m.reduce(*artificial(n=10))["H1"]
    assert not unsupported["support_passed"] and unsupported["family_p_value"] == 1
    targets[0]["y"] = None
    result = m.reduce(rows, targets)
    assert result["excluded_count"] == 5 and result["independent_count"] == 95
    assert m.reduce([], [])["H1"]["raw_p_value"] is None


def test_cost_boundaries_and_target_contract():
    """SCENARIO-REPORT-8074-H1: missing targets never become supported answers."""
    assert [m.action(p) for p in (None, 0.0, 0.1, 0.5, 1.0)] == [
        "escalate",
        "accept",
        "escalate",
        "escalate",
        "reject",
    ]
    with pytest.raises(ValueError, match="probability"):
        m.action(float("nan"))
    rows, targets = artificial()
    targets.append(targets[0])
    with pytest.raises(ValueError, match="target_contract"):
        m.reduce(rows, targets)
    with pytest.raises(ValueError, match="target_contract"):
        m.reduce(rows, [dict(r, y=2) for r in targets[:-1]])


def test_independent_predictions_and_label_order(tmp_path):
    """SCENARIO-REPORT-8074-PREDICTIONS: the evaluator cannot alter sealed heads."""
    raw = tmp_path / "raw"
    plan = m.seal(m.ROOT, raw)
    assert not plan["failures"] and len(plan["prediction_rows"]) == 960
    assert all("y" not in r for r in plan["prediction_rows"])
    assert plan["labels_opened"] == 0
    result = m.evaluate(raw)
    assert result["label_access_receipt"]["opened_after_seal"]
    assert len(result["independent_prediction_rows"]) == 960
    assert result["equivalent_logistic_parity"]["maximum_error"] < 1e-10
    assert result["independent_count"] >= 72
    public = [r for r in plan["public"] if r["public_eligible"] and r["q"] is not None]
    x = np.array([[r["q"], *r["features"]] for r in public])
    shifted, donors = m.permute(public)
    assert np.array_equal(x[:, 0], shifted[:, 0])
    assert all(r["source"] != d for r, d in zip(public, donors, strict=True))
    from carnot import experiment_8073_v699_interaction_energy_fit as producer

    for head in plan["heads"]:
        assert np.max(np.abs(m.predict(head, x) - producer.predict(head, x))) < 1e-12
    head = deepcopy(plan["heads"][-1])
    head["geometry"]["feature_names"].reverse()
    with pytest.raises(ValueError, match="feature_columns"):
        m.predict(head, x)
    altered = deepcopy(plan)
    altered["prediction_rows"][0]["probability"] += 0.1
    m.atomic_json(raw / "predictions.json", altered)
    with pytest.raises(ValueError, match="prediction_seal"):
        m.evaluate(raw)
    m.atomic_json(raw / "predictions.json", plan)
    altered = deepcopy(plan)
    altered["heads"][0]["parameters"][0] += 1
    altered["prediction_seal"]["payload_hash"] = m.canonical_hash(m.payload(altered))
    m.atomic_json(raw / "predictions.json", altered)
    with pytest.raises(ValueError, match="independent_prediction"):
        m.evaluate(raw)
    altered = deepcopy(plan)
    altered["public"][0]["features"][0] += 0.1
    altered["prediction_seal"]["payload_hash"] = m.canonical_hash(m.payload(altered))
    m.atomic_json(raw / "predictions.json", altered)
    with pytest.raises(ValueError, match="independent_prediction"):
        m.evaluate(raw)
    m.atomic_json(raw / "predictions.json", plan)


def test_owned_validation_and_replay_mutations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8074-TERMINAL: receipt fixtures only test readiness logic."""
    monkeypatch.setenv(
        "CARNOT_8074_COVERAGE_CONFIG", os.environ.get("CARNOT_8074_COVERAGE_CONFIG", "")
    )
    output = tmp_path / "control" / (m.NAME + ".json")
    coverage = {p: {"summary": {"num_statements": 1, "missing_lines": 0}} for p in m.OWNED}

    def checks(root, spec, private, durable):
        if spec["name"] in {"prediction_child_normal_exit", "evaluator_child_normal_exit"}:
            flag = (
                "--predict-child" if spec["name"].startswith("prediction") else "--evaluate-child"
            )
            assert m.main([flag, spec["argv"][spec["argv"].index(flag) + 1]]) == 0
        m.atomic_json(private / "coverage.json", dict(files=coverage))
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("private exact-receipt control; no scientific credit\n")
        return dict(spec, passed=True, log_path=str(log), log_sha256=m.sha256_file(log))

    monkeypatch.setattr(m, "run_check", checks)
    monkeypatch.setattr(m, "terminal", lambda p: dict(passed=m.replay(p)))
    assert m.main(["--output", str(output)]) == 0
    original = json.loads(output.read_text())
    assert original["decision_audit_ready_score"] == 1
    assert m.main(["--output", str(output)]) == 1
    assert set(original) - {"field_principles"} <= set(original["field_principles"])
    raw = Path(original["terminal_validation_sidecar_path"]).parent
    work = json.loads((raw / "work.json").read_text())
    validation = json.loads((raw / "validation.json").read_text())
    bad = m.build(work, raw, validation["receipts"], {}, False)
    assert bad["verdict_class"] == "disqualified" and bad["decision_audit_ready_score"] == 0
    changed = deepcopy(work)
    changed["measurement"]["H1"]["support_passed"] = False
    assert (
        m.build(changed, raw, validation["receipts"], coverage, False)["verdict_class"] == "blocked"
    )
    for mode in ("code", "log", "config", "reduction"):
        value = deepcopy(original)
        if mode == "code":
            value["code_config_hashes"][m.MODULE] = "wrong"
        elif mode == "log":
            value["validation_receipts"][0]["log_sha256"] = "wrong"
        elif mode == "config":
            config = json.loads((raw / "configuration.json").read_text())
            altered = deepcopy(config)
            altered["config"]["margin"] = 1
            m.atomic_json(raw / "configuration.json", altered)
        else:
            altered = deepcopy(work)
            altered["measurement"]["H1"]["observed_gain"] += 1
            m.atomic_json(raw / "work.json", altered)
        for ref in value["raw_shard_hashes"]:
            ref["sha256"] = m.sha256_file(Path(ref["path"]))
        m.atomic_json(output, value)
        assert not m.replay(output)
        m.atomic_json(raw / "work.json", work)
        if mode == "config":
            m.atomic_json(raw / "configuration.json", config)
    m.atomic_json(output, original)
    assert m.replay(output)


def test_evaluator_rejects_invalid_targets_and_parity(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8074-PREDICTIONS: malformed labels cannot earn readiness."""
    plan = m.seal(m.ROOT, tmp_path)
    public = [r for r in plan["public"] if r["public_eligible"] and r["q"] is not None]
    labels = json.loads(Path(plan["label_reference"]["path"]).read_text())
    altered = deepcopy(labels)
    target = next(r for r in altered["rows"] if r["family_id"] == public[0]["family_id"])
    target["y"] = 1 - target["y"]
    with pytest.raises(m.InputBlock, match="unsupported_orientation"):
        m.complete_targets(public, altered)
    extract = m.features.extract
    monkeypatch.setattr(m.features, "extract", lambda row: dict(extract(row), values=[-1.0] * 8))
    with pytest.raises(ValueError, match="independent_prediction"):
        m.evaluate(tmp_path)
    monkeypatch.setattr(m.features, "extract", extract)
    prediction = m.predict
    monkeypatch.setattr(
        m, "predict", lambda h, x, **kw: prediction(h, x) + (0.1 if kw.get("logistic") else 0)
    )
    with pytest.raises(ValueError, match="equivalent_logistic_parity"):
        m.evaluate(tmp_path)
    monkeypatch.setattr(m, "predict", prediction)
    monkeypatch.setattr(
        m, "evaluate", lambda raw: (_ for _ in ()).throw(m.InputBlock(raw, "labels", True, False))
    )
    assert m.main(["--evaluate-child", str(tmp_path)]) == 0
    assert (
        json.loads((tmp_path / "evaluation.json").read_text())["failures"][0]["field"] == "labels"
    )


def test_external_preconditions(tmp_path):
    """SCENARIO-REPORT-8074-PREDICTIONS: absent heads and future labels block distinctly."""
    assert (
        m.public_inputs(m.ROOT, tmp_path / "head", absent_head=True)["failures"][0]["field"]
        == "resource_exists"
    )
    assert (
        m.public_inputs(m.ROOT, tmp_path / "future", future_label=True)["failures"][0]["field"]
        == "future_label_fields"
    )
    blocked = m.seal(tmp_path / "absent", tmp_path / "blocked")
    assert blocked["failures"] and not blocked["prediction_rows"]
    assert m.evaluate(tmp_path / "blocked")["failures"]


def test_public_report_projection(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8074-PREDICTIONS: public metadata never decodes outcome fields."""
    path = tmp_path / "capture.json"
    m.atomic_json(path, dict(capture_identity="public-only", human_labels=[1, 0]))
    monkeypatch.setattr(
        m.json, "loads", lambda text: (_ for _ in ()).throw(AssertionError("full_report_decode"))
    )
    assert m.bind(path, tmp_path / "raw", [], fields=("capture_identity",)) == dict(
        capture_identity="public-only"
    )
    with pytest.raises(m.InputBlock, match="public_metadata_field.absent"):
        m.bind(path, tmp_path / "raw", [], fields=("absent",))


@pytest.mark.parametrize("route", ["success", "absent_head", "future_label", "blocked"])
def test_real_private_cli(tmp_path, route):
    """SCENARIO-REPORT-8074-TERMINAL: real children exit outside the checkout."""
    output = tmp_path / route / (m.NAME + ".json")
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    config = os.environ.get("CARNOT_8074_COVERAGE_CONFIG")
    if config:
        command = [sys.executable, "-m", "coverage", "run", "--rcfile=" + config, *command[2:]]
    args = ["--fixture-output", str(output)]
    if route in {"absent_head", "future_label"}:
        args += ["--" + route.replace("_", "-")]
    if route == "blocked":
        args += ["--root", str(tmp_path / "missing")]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    print("8074 subprocess before " + route, flush=True)
    p = subprocess.run(
        command + args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
    )
    print("8074 subprocess after " + route, flush=True)
    assert p.returncode == 0, p.stdout + p.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == ("null" if route == "success" else "blocked")
    assert value["decision_audit_ready_score"] == 0
    assert m.replay(output)
    p = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    value["completed_count"] += 1
    m.atomic_json(output, value)
    assert m.main(["--cold-replay", str(output)]) == 1
    assert m.main(["--cold-replay", str(tmp_path / "missing.json")]) == 1
