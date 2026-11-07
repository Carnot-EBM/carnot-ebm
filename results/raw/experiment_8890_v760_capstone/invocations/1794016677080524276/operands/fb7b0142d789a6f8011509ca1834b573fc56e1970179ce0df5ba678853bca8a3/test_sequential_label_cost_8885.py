"""REQ-REPORT-8885: label savings require paired cost, coverage and safe service."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import sequential_certificate_methods_8880 as frozen
from carnot.reporting import sequential_label_cost_8885 as m
from carnot.reporting import sequential_label_cost_runner_8885 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(size=8, replicates=40):
    """Identical safe policies expose exact costs without relying on learned heads."""
    units = [dict(unit_id=f"{g}/{i:04d}", group=g) for g in frozen.base.GROUPS for i in range(size)]
    scores = [
        dict(
            unit_id=u["unit_id"],
            cluster_id=u["unit_id"],
            arm=a,
            seed=s,
            candidate="support",
            probabilities=[0.96, 0.02, 0.02],
        )
        for a, s in frozen.base.POLICIES
        for u in units
    ]
    cohort, truth = dict(inputs=units), dict.fromkeys([u["unit_id"] for u in units], "support")
    trace = frozen.draws(size, replicates=replicates)
    chosen = frozen.select(cohort, scores, truth, trace)
    cert = frozen.certify(cohort, scores, truth, trace, chosen)
    return cohort, scores, truth, trace, cert


def test_costs_intervals_and_correlated_seeds():
    """REQ-REPORT-8885: every repeat costs a label; five seeds give only forty pairs."""
    args = fixture()
    result = m.evaluate(*args)
    assert len(result["rows"]) == 2240
    assert len(result["census_risk_rows"]) == 2800
    assert len(result["calibration_bound_rows"]) == 5600
    assert len(result["family_violation_rows"]) == 40
    assert not any(x["violation"] for x in result["family_violation_rows"])
    assert result["actual_label_ledger"]["joint_calibration_labels"] == 192000
    assert result["actual_label_ledger"]["fresh_external_annotations"] == 0
    assert all(x["fixed"] == 4800 and x["sequential"] == 1920 for x in result["standalone_costs"])
    assert all(x["sequential_total_labels"] == 1920 for x in result["shared_union_costs"])
    assert len(result["paired_intervals"]) == 23
    assert all(x["independent_count"] == 40 for x in result["paired_intervals"])
    assert len(result["per_replicate_results"]) == 40
    # Observed equality does not erase uncertainty in the coverage difference.
    coverage = [x for x in result["paired_intervals"] if "coverage_difference" in x["contrast"]]
    assert all(x["estimate"] == 0 and x["lower_99"] < -0.01 for x in coverage)
    assert result["label_savings_score"] == 0
    assert m.protocol()["methods"] == frozen.protocol()


def test_refusals_and_unvisited_family_bounds():
    """SCENARIO-REPORT-8885-CONTRACT: refused and unvisited policies stay in the audit."""
    cohort, scores, truth, trace, cert = fixture(replicates=1)
    bad = dict.fromkeys(truth, "refute")
    chosen = frozen.select(cohort, scores, truth, trace)
    refused = frozen.certify(cohort, scores, bad, trace, chosen)
    result = m.evaluate(cohort, scores, bad, trace, refused)
    assert all(x["served_risk"] is None and x["metric"] == 0 for x in result["census_risk_rows"])
    assert all(x["candidate_draws"] == 960 for x in result["stopping_rows"])
    assert all(not x["certificate_issued"] for x in result["rows"])
    assert not any(x["bound_violation"] for x in result["calibration_bound_rows"])
    # The fixed selected policy is wrong on the entire census, despite the safe calibration fixture.
    mismatched = m.evaluate(cohort, scores, bad, trace, cert, calibration_truth=truth)
    assert mismatched["observed_family_violation_count"] == 1
    assert mismatched["worst_served_group_risk"] == 1
    assert mismatched["label_savings_score"] == 0


def test_gates_distinguish_original_absence(tmp_path):
    """REQ-REPORT-8885: missing, missing-field, zero and foreign producer differ."""
    assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    name, (_, field) = next(iter(r.UPSTREAM.items()))
    for value, expected in [
        ({}, "missing_field"),
        ({field: 0}, "observed_zero"),
        ({field: 1}, "rejected_producer"),
    ]:
        atomic_json(tmp_path / (name + ".json"), value)
        assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == expected
    assert r.preconditions(tmp_path / "originals")["external_ready"]


def test_real_private_cli_and_rehashed_tampering(tmp_path):
    """SCENARIO-REPORT-8885-CONTRACT: actual cold replay works without PYTHONPATH."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    output = tmp_path / (m.NAME + ".json")

    def child(options):
        return subprocess.run(
            command + options, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )

    run = child(["--private-output", str(output), "--sources", str(tmp_path / "missing")])
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    assert child(["--cold-replay", str(output)]).returncode == 0
    value = json.loads(output.read_bytes())
    value["sequential_rows_ready_score"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert child(["--cold-replay", str(output)]).returncode == 1
    assert child(["--date", "wrong"]).returncode != 0


def test_positive_gate_requires_cost_coverage_and_risk():
    """REQ-REPORT-8885: benefit requires every gate, independently of readiness."""
    result = m.evaluate(*fixture())
    for row in result["census_risk_rows"]:
        row["metric"] = int(row["protocol"] == "sequential")
    assert m.summarize(result)["label_savings_score"] == 1
    result["rows"][0]["served_risk"] = 0.06
    assert m.summarize(result)["label_savings_score"] == 0


def test_measurement_replay_and_semantic_tampering(tmp_path, monkeypatch):
    """REQ-REPORT-8885: frozen certificates precede census, and rehashed stages are rejected."""
    import shutil

    cohort, scores, truth, trace, cert = fixture(replicates=1)
    for unit in cohort["inputs"]:
        unit["split"] = "population"
    source = dict(rows=cert["rows"], historical_training_calls=5, raw_shard_hashes=[])
    for stage, value in [
        ("population", cohort),
        ("draws", trace),
        ("scores", dict(rows=scores)),
        ("selection_labels", dict(labels=truth)),
        ("certification_labels", dict(labels=truth)),
    ]:
        path = tmp_path / "inputs" / (stage + ".json")
        atomic_json(path, value)
        source["raw_shard_hashes"].append(r.reference(path))
    raw = tmp_path / "measurement"
    atomic_json(raw / (r.SOURCE + ".json"), source)
    monkeypatch.setattr(frozen.base.old.old.p, "labels", lambda population: truth)
    result = r.measurement(raw)
    assert result == r.reduction(raw)
    assert result["actual_label_ledger"]["joint_calibration_labels"] == 4800
    for stage in m.protocol()["access_order"]:
        path, order = raw / (stage + ".json"), raw / "access_order.json"
        saved, access = path.read_bytes(), order.read_bytes()
        atomic_json(path, {})
        events = json.loads(access)
        for event in events["events"]:
            if event["stage"] == stage:
                event.update(r.reference(path))
        atomic_json(order, events)
        with pytest.raises(ValueError, match="measurement_drift"):
            r.reduction(raw)
        path.write_bytes(saved)
        order.write_bytes(access)
    order = json.loads((raw / "access_order.json").read_bytes())
    order["events"].reverse()
    atomic_json(raw / "access_order.json", order)
    with pytest.raises(ValueError, match="access_order"):
        r.reduction(raw)
    monkeypatch.setattr(r, "DEADLINE", -1)
    with pytest.raises(ValueError, match="measurement_deadline"):
        r.measurement(raw)
    monkeypatch.setattr(r, "DEADLINE", 1800)
    with monkeypatch.context() as patch:
        patch.setattr(r.shutil, "copyfile", lambda src, dst: Path(dst).write_text("changed"))
        with pytest.raises(ValueError, match="original_operand"):
            r.measurement(raw)
    monkeypatch.setattr(frozen, "certify", lambda *a: dict(rows=[]))
    with pytest.raises(ValueError, match="frozen_certificate"):
        r.measurement(raw)
    assert shutil.disk_usage(raw).free > 0


def test_main_validation_and_owned_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8885-CONTRACT: failed owned checks disqualify every scheduled cell."""
    from types import SimpleNamespace

    valid = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(complete=True, percent=100, files=[]),
        raw_references=[],
    )
    adapter = SimpleNamespace(validate=lambda raw, private: deepcopy(valid))
    monkeypatch.setattr(r, "namespace", lambda *a: adapter)
    assert r.validate(tmp_path, tmp_path) == valid and adapter.m is m
    monkeypatch.setattr(r, "validate", lambda *a: deepcopy(valid))
    monkeypatch.setattr(
        r,
        "preconditions",
        lambda *a: dict(
            external_ready=True, resources=dict(passed=True), references=[], diagnostics=[]
        ),
    )

    def measured(raw):
        atomic_json(raw / "evaluation.json", {})
        value = r.scheduled("control")
        value["rows"] = [
            dict(x, status="completed", missing_reason=None, metric=0) for x in value["rows"]
        ]
        return value

    monkeypatch.setattr(r, "measurement", measured)
    captures = []
    monkeypatch.setattr(
        r, "publish_primary", lambda out, value, validator: captures.append(deepcopy(value))
    )

    def checked(name, argv, logs, **kw):
        logs.mkdir(parents=True, exist_ok=True)
        receipt = dict(name=name, passed=True)
        for suffix in ["stdout", "stderr"]:
            path = logs / (name + "." + suffix)
            path.write_text("checked")
            receipt[suffix + "_path"] = str(path)
        return receipt

    monkeypatch.setattr(r, "stream", checked)
    health = tmp_path / "health.json"
    stdout = tmp_path / "health.stdout"
    stderr = tmp_path / "health.stderr"
    stdout.write_text("unrelated health failures")
    stderr.write_text("")
    atomic_json(health, dict(stdout_path=str(stdout), stderr_path=str(stderr)))
    monkeypatch.setattr(r, "HEALTH", health)
    command = ["--output", str(tmp_path / (m.NAME + ".json"))]
    assert r.main(command) == 0 and captures[-1]["sequential_rows_ready_score"] == 1
    monkeypatch.setattr(r, "stream", lambda *a, **kw: dict(checked(*a, **kw), passed=False))
    assert r.main(command) == 0 and captures[-1]["verdict_class"] == "disqualified"

    def failed(raw):
        raise ValueError("owned_control")

    monkeypatch.setattr(r, "measurement", failed)
    assert r.main(command) == 0 and captures[-1]["failed_count"] == 2240
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))])
    monkeypatch.setattr(r, "replay", lambda p: dict(passed=True))
    assert r.main(["--cold-replay", str(tmp_path / "fixture")]) == 0


def test_replay_rejects_identity_counts_receipts_and_rehashed_results(tmp_path, monkeypatch):
    """REQ-REPORT-8885: new checksums cannot authorize changed science or readiness."""
    receipt = r.build_current_work_receipt(
        run_id="fixture",
        owner_pid=1,
        events=[],
        inference_substrate="fixture",
        inference_substrate_class="blocked_no_run",
        inference_substrate_details={},
        execution_venue="host",
        started_monotonic_ns=1,
        ended_monotonic_ns=2,
    )
    value = dict(
        r.scheduled("external"),
        experiment_id=8885,
        task_id=m.TASK,
        milestone="2026.11.760",
        run_date="20261106",
        protocol=m.protocol(),
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        current_work_receipt=receipt,
        completed_count=0,
        failed_count=0,
        censored_count=2240,
        excluded_count=0,
        independent_count=0,
        sequential_rows_ready_score=0,
        required_checks_passed=False,
        verdict_class="blocked",
    )
    path = tmp_path / (m.NAME + ".json")

    def write(v):
        v = deepcopy(v)
        v["reproducibility_checksum"] = canonical_hash(v)
        atomic_json(path, v)

    write(value)
    assert r.replay(path)["passed"]
    for field, bad, match in [
        ("experiment_id", 1, "producer"),
        ("protocol", {}, "protocol"),
        ("completed_count", 1, "count"),
        ("independent_count", 40, "count"),
        ("sequential_rows_ready_score", 1, "readiness"),
        ("label_savings_score", 1, "benefit"),
    ]:
        write(dict(value, **{field: bad}))
        with pytest.raises(ValueError, match=match):
            r.replay(path)
    write(value)
    broken = json.loads(path.read_bytes())
    broken["run_date"] = "wrong"
    atomic_json(path, broken)
    with pytest.raises(ValueError, match="checksum"):
        r.replay(path)
    operand = tmp_path / "operand"
    operand.write_text("original")
    bound = dict(value, source_artifact_hashes=[r.reference(operand)])
    write(bound)
    assert r.replay(path)["passed"]
    operand.write_text("changed")
    with pytest.raises(ValueError, match="operand_hash"):
        r.replay(path)
    with monkeypatch.context() as patch:
        patch.setattr(r, "validate_current_work_receipt", lambda v: ["bad"])
        write(value)
        with pytest.raises(ValueError, match="current_receipt"):
            r.replay(path)
    full = dict(
        value,
        rows=[dict(x, status="completed") for x in value["rows"]],
        completed_count=2240,
        censored_count=0,
        independent_count=40,
        measurement_manifest=dict(path=str(tmp_path / "evaluation.json")),
    )
    (tmp_path / (r.SOURCE + ".json")).write_bytes(
        (m.ROOT / "results" / (r.SOURCE + ".json")).read_bytes()
    )
    monkeypatch.setattr(r, "reduction", lambda raw: dict(rows=full["rows"]))
    write(full)
    assert r.replay(path)["passed"]
    qualified = dict(
        full,
        required_checks_passed=True,
        verdict_class="null",
        sequential_rows_ready_score=1,
        owned_statement_coverage=dict(
            complete=True,
            percent=100,
            files=[dict(path=p, missing=[], excluded=[]) for p in m.OWNED],
        ),
        validation_receipts=[dict(passed=True)],
    )
    write(qualified)
    assert r.replay(path)["passed"]
    write(dict(qualified, owned_statement_coverage=dict(complete=False, percent=0, files=[])))
    with pytest.raises(ValueError, match="qualification"):
        r.replay(path)
    write(full)
    monkeypatch.setattr(r, "reduction", lambda raw: dict(rows=[]))
    with pytest.raises(ValueError, match="reduction"):
        r.replay(path)
    (tmp_path / (r.SOURCE + ".json")).write_text("changed")
    with pytest.raises(ValueError, match="original_producer"):
        r.replay(path)
