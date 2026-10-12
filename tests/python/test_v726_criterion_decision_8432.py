"""REQ-REPORT-8432 / REQ-VERIFY-8432: independent operands and real child checks."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import v726_criterion_decision_audit as e
from carnot.verify import v726_decision_metrics as n
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def fixture():
    """Two sibling groups make independence different from the row count."""
    rows, targets = [], []
    for q in range(2):
        for sibling in range(3):
            uid = f"q{q}:{sibling}"
            targets.append(dict(unit_id=uid, y=q, y_sensitivity=q if sibling else None))
            for arm in n.ARMS:
                for seed in n.SEEDS:
                    p = 0.1 if q == 0 else 0.9
                    if arm == "logistic9":
                        p = 0.3 if q == 0 else 0.7
                    rows.append(
                        dict(
                            unit_id=uid,
                            question_hash=f"q{q}",
                            dataset="d",
                            generator=str(sibling),
                            arm=arm,
                            seed=seed,
                            probability=p,
                            missing_reason=None,
                        )
                    )
    return rows, targets


def test_registered_metrics_and_bootstrap():
    """SCENARIO-REPORT-8432-METRICS: exact losses and question-level contrast."""
    rows, targets = fixture()
    result = n.evaluate(rows, targets)
    assert result["brier_gain_vs_logistic"] == pytest.approx(0.08)
    assert result["brier_gain_ci95"] == pytest.approx([0.08, 0.08])
    assert result["decision_loss_delta"] == pytest.approx(-0.2)
    assert result["exploratory_benefit_score"] == 0  # inert permuted control
    assert result["negative_control_degeneracy"]["identical_predictions"]
    assert result["bootstrap"]["independent_questions"] == 2
    assert result["bootstrap"]["draws"] == 10000
    assert result["A2_sensitivity"]["spline9"]["observed_answer_count"] == 4
    assert result["per_stratum_metrics"]
    assert len(result["rows"]) == 150
    assert n.action(0.2) == "accept" and n.action(0.8) == "reject"
    assert n.action(0.5) == "escalate"
    assert n.loss(0.2, 1) == 1 and n.loss(0.8, 0) == 1
    assert n.loss(0.5, 0) == 0.2


def test_missing_and_empty_denominators():
    """SCENARIO-REPORT-8432-METRICS: unknowns stay in intended bounds."""
    rows, targets = fixture()
    for row in rows:
        if row["unit_id"] == "q0:0":
            row.update(probability=None, missing_reason="missing_feature")
    result = n.evaluate(rows, targets)
    bounds = result["missing_worst_case_bounds"]["A1"]["spline9"]
    assert bounds["missing_prediction_count"] == 5
    assert bounds["brier"][1] - bounds["brier"][0] == pytest.approx(1 / 6)
    assert result["bootstrap"]["independent_questions"] == 1
    empty = n.evaluate([], [])
    assert empty["brier_gain_ci95"] is None
    assert empty["A1_metrics"]["raw_p2"]["undefined_reasons"]
    with pytest.raises(ValueError, match="duplicate"):
        n.evaluate(rows + [rows[0]], targets)
    rows[0]["probability"] = 2
    with pytest.raises(ValueError, match="probability"):
        n.evaluate(rows, targets)


def test_sealing_measure_and_rehashed_tampering(tmp_path):
    """SCENARIO-REPORT-8432-CUSTODY: prediction bytes precede evaluator opens."""
    raw = tmp_path / "raw"
    work = e.measure(e.ROOT, raw)
    assert not work["failures"]
    assert work["clock"]["predictions_sealed_ns"] < work["clock"]["targets_opened_ns"]
    assert len(work["result"]["rows"]) == 7500
    assert all("y" not in p for p in e.read(work["prediction_reference"])["rows"])
    receipt = child("proof", [sys.executable, "-c", "print('real child')"], tmp_path / "logs")
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work, [receipt], raw, output)
    atomic_json(output, value)
    assert e.replay(output)
    for field in ["brier_gain_vs_logistic", "decision_audit_ready_score"]:
        changed = deepcopy(value)
        changed[field] = 999
        changed["reproducibility_checksum"] = e.checksum(changed)
        atomic_json(output, changed)
        assert not e.replay(output)
    assert not e.replay(tmp_path / "absent")
    assert (
        e.build(work, [dict(receipt, passed=False)], raw, output)["verdict_class"] == "disqualified"
    )
    flagged = e.build(
        dict(
            work,
            adversarial_findings=[
                dict(kind="INFORMATIONAL_NULL", severity="info", detail="unsupported zero delta")
            ],
        ),
        [receipt],
        raw,
        output,
    )
    assert flagged["decision_audit_ready_score"] == 0 and flagged["scientific_measurement_complete"]
    blocked = e.measure(tmp_path / "missing", tmp_path / "blocked")
    result = e.build(blocked, [receipt], tmp_path / "blocked", output)
    assert result["verdict_class"] == "blocked" and len(result["rows"]) == 7500


def test_preconditions_and_missing_operands(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8432-CUSTODY: external absence differs from owned failure."""
    upstream, panel, frozen, features = e.operands(e.ROOT)
    with monkeypatch.context() as patch:
        read = e.read
        patch.setattr(
            e,
            "read",
            lambda ref: (
                dict(read(ref), head_ready_score=0)
                if ref["path"].endswith(e.UPSTREAM)
                else read(ref)
            ),
        )
        with pytest.raises(ValueError, match="upstream_not_qualified"):
            e.operands(e.ROOT)
    with monkeypatch.context() as patch:
        sidecar = e.read_bound_sidecar
        patch.setattr(
            e, "read_bound_sidecar", lambda *args: dict(sidecar(*args), report=dict(passed=False))
        )
        with pytest.raises(ValueError, match="upstream_publication_failed"):
            e.operands(e.ROOT)
    with monkeypatch.context() as patch:
        resources = e.methods.legacy.resources(tmp_path)
        patch.setattr(
            e.methods.legacy,
            "resources",
            lambda path: dict(resources, private_disk_backed_scratch=False),
        )
        assert e.measure(e.ROOT, tmp_path / "resource")["failures"]
    with monkeypatch.context() as patch:
        predictor = e.predict_rows
        patch.setattr(
            e,
            "predict_rows",
            lambda head, rows, strata: [] if head else predictor(head, rows, strata),
        )
        assert e.measure(e.ROOT, tmp_path / "predict")["owned_failure"]
    with monkeypatch.context() as patch:
        patch.setattr(
            e,
            "operands",
            lambda root: (
                upstream,
                dict(
                    panel,
                    evaluator_custody=dict(path=str(tmp_path / "absent-labels"), sha256="absent"),
                ),
                frozen,
                features,
            ),
        )
        work = e.measure(e.ROOT, tmp_path / "labels")
        assert not work["owned_failure"] and work["failures"]


def test_rehashed_primitive_and_receipt_controls(tmp_path):
    """SCENARIO-VERIFY-8432-REPLAY: independently pinned operands defeat rehashing."""
    raw = tmp_path / "raw"
    original = e.measure(e.ROOT, raw)
    sealed_original = Path(original["prediction_reference"]["path"]).read_bytes()
    receipt = child(
        "proof", [sys.executable, "-c", "print('independent receipt')"], tmp_path / "logs"
    )
    output = tmp_path / (e.NAME + ".json")

    def publish(work, receipts=None):
        atomic_json(raw / "measurement.json", work)
        atomic_json(output, e.build(work, receipts or [receipt], raw, output))

    for kind in [
        "hash",
        "snapshot",
        "authority",
        "pin",
        "prediction",
        "reduction",
        "clock",
        "receipt",
        "forged_exit",
        "forged_expected",
    ]:
        work = deepcopy(original)
        predictions = e.read(original["prediction_reference"])
        if kind == "hash":
            work["code_refs"][0]["sha256"] = "wrong"
        if kind == "snapshot":
            work["authority"]["authority_snapshots"]["active"]["sha256"] = "wrong"
        if kind == "authority":
            work["authority"]["activated"] = False
        if kind == "pin":
            work["upstream_reference"]["sha256"] = "wrong"
        if kind == "prediction":
            predictions["rows"][0]["probability"] = 0.123456
            atomic_json(Path(work["prediction_reference"]["path"]), predictions)
            work["prediction_reference"] = e.reference(Path(work["prediction_reference"]["path"]))
            work["primitive_files"][0] = work["prediction_reference"]
        if kind == "reduction":
            work["result"]["brier_gain_vs_logistic"] = 999
        if kind == "clock":
            work["clock"]["targets_opened_ns"] = 0
        receipts = [dict(receipt, stdout_sha256="wrong")] if kind == "receipt" else [receipt]
        if kind == "forged_exit":
            receipts = [dict(receipt, exit_code=1, actual_exit=1, passed=True)]
        if kind == "forged_expected":
            receipts = [dict(receipt, expected_exit=1)]
        publish(work, receipts)
        assert not e.replay(output), kind
        Path(original["prediction_reference"]["path"]).write_bytes(sealed_original)
    publish(original)
    assert e.replay(output)
    failed = child(
        "genuine_failure", [sys.executable, "-c", "raise SystemExit(1)"], tmp_path / "logs"
    )
    publish(original, [dict(failed, passed=True)])
    assert not e.replay(output)


def test_real_owned_child_shards_and_outside_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8432-REPLAY: actual child shards survive private publication."""
    from carnot.reporting import v726_criterion_decision_runner as r

    private = tmp_path / "private"
    plan = r.manifest(private)
    cov = str(e.ROOT / ".venv/bin/coverage")
    config = "--rcfile=" + str(private / "coverage.ini")
    commands = [
        dict(
            name="actual_child_coverage",
            argv=[cov, "run", config, str(e.ROOT / e.CLI), "--deliberate-error"],
            expected=7,
            deadline=30,
            scope="owned",
        )
    ]
    commands += [p for p in plan if p["name"] in {"coverage_combine", "coverage_json"}]
    monkeypatch.setattr(r, "manifest", lambda path: commands)
    output = tmp_path / "results" / (e.NAME + ".json")
    assert r.run(e.ROOT, output, private) == 0
    value = json.loads(output.read_bytes())
    assert len(value["raw_shard_hashes"]) > 3
    assert child(
        "outside_replay",
        [
            "/usr/bin/env",
            "-u",
            "PYTHONPATH",
            sys.executable,
            "-c",
            "import os,runpy,sys; os.chdir(sys.argv[1]); sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')",
            str(tmp_path),
            str(e.ROOT / e.CLI),
            "--cold-replay",
            str(output),
        ],
        tmp_path / "outside",
        deadline=120,
    )["passed"]


def test_private_cli_and_owned_failures(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8432-REPLAY: genuine child publication and failure receipts."""
    from carnot.reporting import v726_criterion_decision_runner as r

    plan = r.manifest(tmp_path / "private")
    assert any(p["name"] == "private_E2E021" for p in plan)
    output = tmp_path / "results" / (e.NAME + ".json")
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = child(
        "cli", [*argv, "--output", str(output), "--private-control"], tmp_path / "cli", deadline=120
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["completed_count"] == 7500
    for name, args, expected in [
        ("replay", ["--cold-replay", str(output)], 0),
        ("absent", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("error", ["--deliberate-error"], 7),
        ("date", ["--date", "20000101"], 2),
        ("guard", ["--private-control"], 2),
    ]:
        assert child(name, [*argv, *args], tmp_path / "cli", expected=expected, deadline=120)[
            "passed"
        ]
    with monkeypatch.context() as patch:
        patch.setattr(
            e.n, "evaluate", lambda *args: (_ for _ in ()).throw(ValueError("owned_numeric_error"))
        )
        work = e.measure(e.ROOT, tmp_path / "failed")
        assert work["owned_failure"]
        assert (
            e.build(work, [receipt], tmp_path / "failed", output)["verdict_class"] == "disqualified"
        )
