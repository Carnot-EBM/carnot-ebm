"""REQ-REPORT-8896: measurements preserve inclusion and the two cost estimands."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from carnot.reporting import shared_terminal_evaluation_8896 as m
from carnot.reporting import shared_terminal_evaluation_runner_8896 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(replicates=40):
    """Safe repeated inputs expose certificate costs without claiming generalization."""
    units = [dict(unit_id=f"{g}/{i}", group=g) for g in m.frozen.base.GROUPS for i in range(2)]
    scores = [
        dict(
            unit_id=u["unit_id"],
            cluster_id=u["unit_id"],
            arm=a,
            seed=s,
            candidate="support",
            probabilities=[0.96, 0.02, 0.02],
        )
        for a, s in m.frozen.base.POLICIES
        for u in units
    ]
    cohort = dict(inputs=units)
    truth = dict.fromkeys([u["unit_id"] for u in units], "support")
    trace = m.frozen.draws(2, replicates=replicates)
    selection = m.frozen.select(cohort, scores, truth, trace)
    cert = m.frozen.certify(cohort, scores, truth, trace, selection)
    return cohort, scores, truth, trace, cert


def test_frozen_protocol_and_separate_costs():
    """REQ-REPORT-8896: all labels count and learned seeds do not inflate independent n."""
    p = m.protocol()
    assert p["methods"] == m.frozen.protocol()
    assert p["family_size"] == 23 and p["family_error"] == 0.01
    cohort, scores, truth, trace, cert = fixture()
    result = m.evaluate(cohort, scores, truth, trace, cert)
    assert result["standalone_label_benefit_score"] == 1
    assert result["shared_union_benefit_score"] == 1
    assert len(result["inclusion_rows"]) == 1120
    assert len(result["bound_rows"]) == 5600
    assert all(x["passed"] for x in result["inclusion_rows"])
    assert result["theoretical_family_error"]["total"] == 0.01
    assert all(x["independent_count"] == 40 for x in result["paired_intervals"])
    assert result["paid_label_ledger"]["joint_calibration_labels"] == 192000
    for row in cert["rows"]:
        if row["arm"] == "constant_uniform" and row["protocol"] == "sequential":
            row.update(
                candidate_draws=960,
                total_labels=1200,
                certificate_issued=False,
                looks=row["scheduled_looks"],
            )
    # A refused terminal baseline is a legitimate zero-union-savings control.
    baseline = m.frozen.base.POLICIES[0]
    for row in cert["rows"]:
        if (row["arm"], row["seed"]) == baseline:
            for event in row["scheduled_looks"]:
                event.update(certificate_issued=False, risk_upper=1)
            row.update(
                certificate_issued=False,
                candidate_draws=960,
                total_labels=1200,
                looks=row["scheduled_looks"],
            )
    cert["standalone_costs"], cert["shared_union_costs"] = m.frozen.ledgers(cert["rows"])
    mixed = m.evaluate(cohort, scores, truth, trace, cert)
    assert mixed["standalone_label_benefit_score"] == 1
    assert mixed["shared_union_benefit_score"] == 0
    unsafe = m.evaluate(cohort, scores, dict.fromkeys(truth, "refute"), trace, cert)
    assert unsafe["standalone_label_benefit_score"] == unsafe["shared_union_benefit_score"] == 0


def test_inclusion_rejects_corrupt_premises_and_unsafe_controls():
    """SCENARIO-REPORT-8896-CONTRACT: one failed inclusion cell disqualifies the method."""
    operands = fixture(1)
    for field, value in [("cutoff", None), ("final_prefix_hash", "changed")]:
        bad = deepcopy(operands[-1])
        bad["rows"][0][field] = value
        with pytest.raises(ValueError, match="inclusion"):
            m.evaluate(*operands[:-1], bad)
    bad = deepcopy(operands[-1])
    bad["rows"][0]["certificate_issued"] = False
    with pytest.raises(ValueError, match="inclusion"):
        m.evaluate(*operands[:-1], bad)
    bad = deepcopy(operands[-1])
    bad["rows"][0]["scheduled_looks"][-1]["alpha"] = 0.01
    with pytest.raises(ValueError, match="inclusion"):
        m.evaluate(*operands[:-1], bad)
    cohort, scores, truth, trace, _ = operands
    unsafe = dict.fromkeys(truth, "refute")
    chosen = m.frozen.select(cohort, scores, unsafe, trace)
    refused = m.frozen.certify(cohort, scores, unsafe, trace, chosen)
    result = m.evaluate(cohort, scores, unsafe, trace, refused)
    assert result["standalone_label_benefit_score"] == 0
    assert all(not x["certificate_issued"] for x in result["stopping_rows"])
    assert all(x["passed"] for x in m.frozen.controls())


def test_external_gates_and_cold_private_cli(tmp_path):
    """SCENARIO-REPORT-8896-CONTRACT: absence blocks; zero and malformed inputs stay explicit."""
    assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    for value, check in [
        ({}, "missing_field"),
        ({"shared_terminal_ready_score": 0}, "observed_zero"),
        ({"shared_terminal_ready_score": 1}, "rejected_producer"),
    ]:
        atomic_json(tmp_path / (r.SOURCE + ".json"), value)
        assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == check
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    output = tmp_path / (m.NAME + ".json")

    def child(options):
        return subprocess.run(
            command + options, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
        )

    run = child(["--private-output", str(output), "--sources", str(tmp_path / "absent")])
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["certificate_rows_ready_score"] == 0
    assert child(["--cold-replay", str(output)]).returncode == 0
    value.pop("reproducibility_checksum")
    value["certificate_rows_ready_score"] = 1
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert child(["--replay", str(output)]).returncode == 1
    assert child(["--date", "wrong"]).returncode != 0
    assert child(["--private-output", str(m.ROOT / "results" / output.name)]).returncode != 0


def test_original_inputs_authenticate(tmp_path):
    """REQ-REPORT-8896: the actual original assertions remain immutable and readable."""
    pre = r.preconditions(tmp_path / "originals", m.ROOT / "results")
    assert pre["external_ready"], pre["diagnostics"]
    assert pre["resources"]["passed"] and pre["authority"]["ready"]


def test_measurement_reconstructs_and_rejects_drift(tmp_path, monkeypatch):
    """REQ-REPORT-8896: selection and certification seal before post-decision census access."""
    cohort, scores, truth, trace, cert = fixture(1)
    for u in cohort["inputs"]:
        u["split"] = "population"
    chosen = m.frozen.select(cohort, scores, truth, trace)
    operands = dict(
        population=cohort,
        scores=dict(rows=scores),
        draws=trace,
        fit=dict(numerical_training_calls=5),
        selection=chosen,
        selection_labels=dict(labels=truth),
        certification_labels=dict(labels=truth),
    )
    refs = []
    for name, value in operands.items():
        path = tmp_path / "source" / f"{name}.json"
        atomic_json(path, value)
        refs.append(r.reference(path))
    source = dict(raw_shard_hashes=refs, stopping_rows=cert["rows"], historical_training_calls=5)
    raw = tmp_path / "raw"
    raw.mkdir()
    atomic_json(raw / (r.SOURCE + ".json"), source)
    monkeypatch.setattr(m.frozen.base.old.old.p, "labels", lambda *a: truth)
    result = r.measurement(raw)
    assert r.reduction(raw) == result
    assert result["numerical_training_calls"] == 0
    assert result["historical_training_calls"] == 5
    path = raw / "evaluation.json"
    original = path.read_bytes()
    atomic_json(path, {})
    with pytest.raises(ValueError, match="measurement_drift"):
        r.measurement(raw, replaying=True)
    path.write_bytes(original)
    order = json.loads((raw / "access_order.json").read_bytes())
    order["events"].reverse()
    atomic_json(raw / "access_order.json", order)
    with pytest.raises(ValueError, match="access_order"):
        r.reduction(raw)
    source["stopping_rows"] = []
    atomic_json(raw / (r.SOURCE + ".json"), source)
    with pytest.raises(ValueError, match="certificate_drift"):
        r.measurement(raw)
    (raw / "retained_inputs" / "population.json").write_text("{}")
    with pytest.raises(ValueError, match="operand_drift"):
        r.measurement(raw, replaying=True)
    monkeypatch.setattr(r, "DEADLINE", -1)
    with pytest.raises(ValueError, match="deadline"):
        r.measurement(raw)


def test_main_and_owned_failure_routes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8896-CONTRACT: failed owned checks never grant readiness."""
    valid = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(complete=True, percent=100, files=[]),
        raw_references=[],
    )
    adapter = SimpleNamespace(validate=lambda *a: deepcopy(valid))
    monkeypatch.setattr(r, "namespace", lambda *a: adapter)
    assert r.validate(tmp_path, tmp_path) == valid and adapter.m is m
    monkeypatch.setattr(r, "validate", lambda *a: deepcopy(valid))
    monkeypatch.setattr(r, "HEALTH", tmp_path / "health.json")
    monkeypatch.setattr(
        r,
        "preconditions",
        lambda *a: dict(
            external_ready=True,
            resources=dict(passed=True),
            references=[],
            diagnostics=[],
            authority={},
        ),
    )

    def measured(raw):
        atomic_json(raw / "evaluation.json", {})
        result = r.scheduled("fixture")
        result["rows"] = [
            dict(x, status="completed", missing_reason=None, metric=0) for x in result["rows"]
        ]
        return result

    monkeypatch.setattr(r, "measurement", measured)
    captured = []
    monkeypatch.setattr(
        r, "publish_primary", lambda out, v, validator: captured.append(deepcopy(v))
    )

    def terminal(candidate, raw):
        reports = []
        for name in ["adversarial", "rows", "replay"]:
            report = dict(name=name, passed=True)
            for suffix in ["stdout", "stderr"]:
                p = raw / (name + "." + suffix)
                p.write_text("fixture check")
                report[suffix + "_path"] = str(p)
            reports.append(report)
        return reports

    adapter.terminal = terminal
    args = ["--output", str(tmp_path / (m.NAME + ".json"))]
    assert r.main(args) == 0 and captured[-1]["certificate_rows_ready_score"] == 1
    log = tmp_path / "health.log"
    log.write_text("unrelated repository health")
    atomic_json(r.HEALTH, dict(log=r.reference(log), passed=False))
    monkeypatch.setattr(
        r, "measurement", lambda raw: dict(measured(raw), standalone_label_benefit_score=1)
    )
    assert r.main(args) == 0 and captured[-1]["verdict_class"] == "circular_positive"
    adapter.terminal = lambda *a: [dict(x, passed=False) for x in terminal(*a)]
    assert r.main(args) == 0 and captured[-1]["verdict_class"] == "disqualified"

    def failed(raw):
        raise ValueError("owned_failure")

    monkeypatch.setattr(r, "measurement", failed)
    assert r.main(args) == 0 and captured[-1]["failed_count"] == 2240
    monkeypatch.setattr(r, "validate", lambda *a: r.unqualified())
    assert r.main(args) == 0 and not captured[-1]["required_checks_passed"]


def test_replay_rejects_rehashed_claims(tmp_path, monkeypatch):
    """REQ-REPORT-8896: checksum repair cannot invent science, authority or qualification."""
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
        experiment_id=8896,
        task_id=m.TASK,
        milestone="2026.11.761",
        run_date="20261107",
        protocol=m.protocol(),
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        current_work_receipt=receipt,
        duration_s=receipt["duration_s"],
        model_invocation_counts=receipt["invocation_counts"],
        verifier_is_oracle=True,
        independent_generalization_score=0,
        completed_count=0,
        failed_count=0,
        censored_count=2240,
        excluded_count=0,
        independent_count=0,
        intended_count=2240,
        certificate_rows_ready_score=0,
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
        ("intended_count", 1, "count"),
        ("verifier_is_oracle", False, "compute"),
        ("certificate_rows_ready_score", 1, "readiness"),
        ("standalone_label_benefit_score", 1, "benefit"),
    ]:
        write(dict(value, **{field: bad}))
        with pytest.raises(ValueError, match=match):
            r.replay(path)
    write(value)
    changed = json.loads(path.read_bytes())
    changed["run_date"] = "changed"
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="checksum"):
        r.replay(path)
    operand = tmp_path / "operand"
    operand.write_text("original")
    write(dict(value, source_artifact_hashes=[r.reference(operand)]))
    assert r.replay(path)["passed"]
    operand.write_text("changed")
    with pytest.raises(ValueError, match="operand_hash"):
        r.replay(path)
    with monkeypatch.context() as patch:
        patch.setattr(r, "validate_current_work_receipt", lambda *a: ["bad"])
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
    for name in r.UPSTREAM:
        (tmp_path / (name + ".json")).write_bytes(
            (m.ROOT / "results" / (name + ".json")).read_bytes()
        )
    monkeypatch.setattr(r, "reduction", lambda *a: dict(rows=full["rows"]))
    write(full)
    assert r.replay(path)["passed"]
    qualified = dict(
        full,
        required_checks_passed=True,
        verdict_class="null",
        certificate_rows_ready_score=1,
        validation_receipts=[dict(passed=True)],
        owned_statement_coverage=dict(
            complete=True,
            percent=100,
            files=[dict(path=p, missing=[], excluded=[]) for p in m.OWNED],
        ),
    )
    write(qualified)
    assert r.replay(path)["passed"]
    write(dict(qualified, owned_statement_coverage=dict(complete=False, percent=0, files=[])))
    with pytest.raises(ValueError, match="qualification"):
        r.replay(path)
    monkeypatch.setattr(r, "reduction", lambda *a: dict(rows=[]))
    write(full)
    with pytest.raises(ValueError, match="reduction"):
        r.replay(path)
    (tmp_path / (r.SOURCE + ".json")).write_text("changed original")
    write(full)
    with pytest.raises(ValueError, match="original_producer"):
        r.replay(path)


def test_owned_control_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8896-CONTRACT: a failed owned control is disqualified, never blocked."""
    raw = tmp_path / "raw"
    raw.mkdir()
    refs = []
    for stage in [
        "population",
        "draws",
        "fit",
        "scores",
        "selection_labels",
        "certification_labels",
    ]:
        p = tmp_path / "source" / (stage + ".json")
        atomic_json(p, dict(rows=[]))
        refs.append(r.reference(p))
    atomic_json(raw / (r.SOURCE + ".json"), dict(raw_shard_hashes=refs))
    monkeypatch.setattr(m.frozen, "controls", lambda: [dict(passed=False)])
    with pytest.raises(ValueError, match="owned_control_failure"):
        r.measurement(raw)
