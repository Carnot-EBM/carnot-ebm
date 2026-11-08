"""REQ-REPORT-8910: refusal must save fetched work without changing decisions."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from carnot.reporting import finite_budget_evaluation_8910 as m
from carnot.reporting import finite_budget_evaluation_runner_8910 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(replicates=1):
    """Repeated safe worlds expose fetched prefixes without adding independent worlds."""
    units = [
        dict(unit_id=f"{g}/{i}", group=g, split="population")
        for g in m.base.GROUPS
        for i in range(2)
    ]
    scores = [
        dict(
            unit_id=u["unit_id"],
            cluster_id=u["unit_id"],
            arm=a,
            seed=s,
            candidate="support",
            probabilities=[0.96, 0.02, 0.02],
        )
        for a, s in m.base.POLICIES
        for u in units
    ]
    cohort = dict(inputs=units)
    truth = dict.fromkeys([u["unit_id"] for u in units], "support")
    trace = m.frozen.frozen.draws(2, replicates=replicates)
    selection = m.frozen.frozen.select(cohort, scores, truth, trace)
    return cohort, scores, truth, trace, selection


def run_fixture(replicates=1, wrong=False):
    cohort, scores, truth, trace, selection = fixture(replicates)
    calls = []

    def fetch(arm, rep, group, stage, units):
        calls.append((arm, rep, group, stage, len(units)))
        return {u["unit_id"]: "refute" if wrong else "support" for u in units}

    cert = m.certify(cohort, scores, trace, selection, fetch)
    result = m.evaluate(cohort, scores, truth, trace, cert)
    return result, cert, calls, (cohort, scores, truth, trace, selection)


def test_service_prefixes_and_primary_interval():
    """REQ-REPORT-8910: every paid draw and policy counts, with one paired primary contrast."""
    result, cert, calls, _ = run_fixture(40, wrong=True)
    assert len(result["rows"]) == 3360
    assert result["shared_union_benefit_score"] == 1
    assert all(
        x["policy_input_evaluations"]
        == x["selection_policy_input_evaluations"] + x["candidate_draws"]
        for x in result["rows"]
    )
    assert all(x["selection_fetch_requests"] == 0 for x in cert["evaluator_label_ledger"])
    assert result["pathwise_inclusion_violations"] == result["decision_equality_violations"] == 0
    assert all(x["passed"] for x in result["decision_comparison_rows"])
    service = [x for x in calls if x[0] == "early_acceptance_refusal"]
    assert sum(x[-1] for x in service) == sum(
        x["certification_labels"] for x in cert["paid_label_ledger"]
    )
    assert sum(x[-1] for x in service) < 40 * 3840
    assert all(
        x["group_prefixes"][g] < 960
        for x in cert["shared_union_costs"]
        if x["protocol"] == "early_acceptance_refusal"
        for g in m.base.GROUPS
    )
    assert all(
        x["selection_labels"] == 240 and x["replacement_repeats_paid"]
        for x in cert["paid_label_ledger"]
    )
    assert all(
        x["distinct_worlds"] == 2 and x["policy_input_evaluations"] > 0
        for x in cert["paid_label_ledger"]
    )
    primary = next(
        x for x in result["paired_intervals"] if x["contrast"] == "shared_union/label_savings"
    )
    assert primary["lower_99"] > 0 and primary["independent_count"] == 40
    assert m.protocol()["primary_contrast"] == primary["contrast"]
    assert m.protocol()["methods"] == m.frozen.protocol()


def test_valid_zero_unsafe_census_and_semantic_tampering():
    """SCENARIO-REPORT-8910-CONTRACT: valid zero is null; changed decisions cannot pass."""
    result, cert, _, operands = run_fixture()
    assert result["shared_union_benefit_score"] == 0
    assert (
        next(
            x for x in result["paired_intervals"] if x["contrast"] == "shared_union/label_savings"
        )["estimate"]
        == 0
    )
    cohort, scores, truth, trace, selection = operands
    unsafe = m.evaluate(cohort, scores, dict.fromkeys(truth, "refute"), trace, cert)
    assert unsafe["shared_union_benefit_score"] == unsafe["standalone_label_benefit_score"] == 0
    for field, value in [("certificate_issued", False), ("cutoff", None), ("candidate_draws", 960)]:
        bad = deepcopy(cert)
        stop = next(x for x in bad["rows"] if x["protocol"] == "early_acceptance_refusal")
        stop[field] = value
        with pytest.raises(ValueError, match="decision|ledger"):
            m.evaluate(cohort, scores, truth, trace, bad)
    bad = dict(selection, selection_hash="changed")
    with pytest.raises(ValueError, match="selection"):
        m.certify(cohort, scores, trace, bad, lambda *a: {})
    with pytest.raises(ValueError, match="fetch"):
        m.certify(cohort, scores, trace, selection, lambda *a: {})
    bad_trace = deepcopy(trace)
    bad_trace["replicates"][0]["groups"][next(iter(m.base.GROUPS))]["certification"] = [99] * 960
    with pytest.raises(ValueError, match="draw"):
        m.certify(cohort, scores, bad_trace, selection, lambda *a: {})


def test_external_gates_and_original_bytes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8910-CONTRACT: external absence and zero are distinct terminal gates."""
    assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    for value, check in [
        ({}, "missing_field"),
        ({"finite_budget_ready_score": 0}, "observed_zero"),
        ({"finite_budget_ready_score": 1}, "rejected_producer"),
        ([], "rejected_producer"),
    ]:
        atomic_json(tmp_path / (r.SOURCE + ".json"), value)
        assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == check
    pre = r.preconditions(tmp_path / "originals", m.ROOT / "results")
    assert pre["external_ready"], pre["diagnostics"]
    assert pre["proof_receipt_hash"] and pre["resources"]["passed"]
    monkeypatch.setattr(r, "PIN", "changed")
    assert not r.preconditions(tmp_path / "drift", m.ROOT / "results")["external_ready"]


def test_private_cli_and_rehashed_mutations(tmp_path):
    """SCENARIO-REPORT-8910-CONTRACT: real cold CLI preserves blocked schedules and rejects forgery."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    output = tmp_path / (m.NAME + ".json")

    def child(args):
        return subprocess.run(
            [sys.executable, "-u", str(m.ROOT / m.CLI)] + args,
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=180,
        )

    run = child(["--private-output", str(output), "--sources", str(tmp_path / "absent")])
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 3360
    assert value["gate_check_summary"] and child(["--replay", str(output)]).returncode == 0
    for field, bad in [
        ("certificate_rows_ready_score", 1),
        ("experiment_id", 1),
        ("protocol", {}),
        ("intended_count", 1),
        ("verifier_is_oracle", False),
        ("shared_union_benefit_score", 1),
        ("gate_check_summary", []),
    ]:
        changed = dict(value, **{field: bad})
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(output, changed)
        assert child(["--replay", str(output)]).returncode == 1
    atomic_json(output, dict(value, run_date="changed"))
    assert child(["--replay", str(output)]).returncode == 1
    assert child(["--date", "wrong"]).returncode != 0
    assert child(["--private-output", str(m.ROOT / "results" / output.name)]).returncode != 0


def test_measurement_replay_and_owned_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8910: frozen stops precede census and independently reconstruct from operands."""
    _, _, _, operands = run_fixture()
    cohort, scores, truth, trace, _ = operands
    refs = []
    for stage, value in [
        ("population", cohort),
        ("scores", dict(rows=scores)),
        ("fit", dict(numerical_training_calls=5)),
        ("draws", trace),
    ]:
        p = tmp_path / "inputs" / (stage + ".json")
        atomic_json(p, value)
        refs.append(r.reference(p))
    source = dict(source_artifact_hashes=refs, calibration_manifest=refs[-1])
    raw = tmp_path / "raw"
    raw.mkdir()
    atomic_json(raw / (r.SOURCE + ".json"), source)
    monkeypatch.setattr(
        m.frozen.frozen.base.old.old.p,
        "labels",
        lambda units: {u["unit_id"]: truth[u["unit_id"]] for u in units},
    )
    result = r.measurement(raw)
    assert r.reduction(raw) == result
    order_path = raw / "access_order.json"
    order = json.loads(order_path.read_bytes())
    assert [x["stage"] for x in order["events"]] == m.protocol()["access_order"]
    bad = deepcopy(order)
    bad["events"].reverse()
    atomic_json(order_path, bad)
    with pytest.raises(ValueError, match="access_order"):
        r.reduction(raw)
    atomic_json(order_path, order)
    original = (raw / "evaluation.json").read_bytes()
    atomic_json(raw / "evaluation.json", {})
    with pytest.raises(ValueError, match="measurement_drift"):
        r.measurement(raw, replaying=True)
    (raw / "evaluation.json").write_bytes(original)
    monkeypatch.setattr(m.frozen.frozen, "controls", lambda: [dict(passed=False)])
    with pytest.raises(ValueError, match="owned_control"):
        r.measurement(raw)
    monkeypatch.setattr(r, "DEADLINE", -1)
    with pytest.raises(ValueError, match="deadline"):
        r.measurement(raw)


def test_main_routes_and_replay_qualification(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8910-CONTRACT: owned failures disqualify and forged readiness fails replay."""
    result, _, _, _ = run_fixture(40)
    valid = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        raw_references=[],
        coverage=dict(
            complete=True,
            percent=100,
            files=[dict(path=p, missing=[], excluded=[]) for p in m.OWNED],
        ),
    )
    adapter = SimpleNamespace(validate=lambda *a: deepcopy(valid))
    monkeypatch.setattr(r, "namespace", lambda *a: adapter)
    assert r.validate(tmp_path, tmp_path) == valid and adapter.m is m
    monkeypatch.setattr(r, "validate", lambda *a: deepcopy(valid))
    monkeypatch.setattr(r, "HEALTH", tmp_path / "health.json")
    pre = dict(
        external_ready=True,
        resources=dict(passed=True),
        references=[],
        diagnostics=[],
        proof_receipt_hash="fixture",
    )
    monkeypatch.setattr(r, "preconditions", lambda *a: deepcopy(pre))

    def measured(raw):
        atomic_json(raw / "evaluation.json", result)
        return deepcopy(result)

    monkeypatch.setattr(r, "measurement", measured)
    captured = []
    monkeypatch.setattr(
        r, "publish_primary", lambda out, value, check: captured.append(deepcopy(value))
    )

    def terminal(candidate, raw):
        p = raw / "terminal.log"
        p.write_text("fixture receipt")
        return [dict(passed=True, stdout_path=str(p), stderr_path=str(p))]

    adapter.terminal = terminal
    args = ["--output", str(tmp_path / (m.NAME + ".json"))]
    assert r.main(args) == 0
    value = captured[-1]
    assert value["full_wall_costs"]["through_terminal_validation_s"] == value["duration_s"]
    assert value["certificate_rows_ready_score"] == 1 and value["verdict_class"] == "null"
    monkeypatch.setattr(r, "reduction", lambda *a: deepcopy(result))
    path = tmp_path / (m.NAME + ".json")

    # Main was replaced only at publication; exercise the real semantic reader.
    def write(value):
        value = deepcopy(value)
        value.pop("reproducibility_checksum", None)
        value["reproducibility_checksum"] = canonical_hash(value)
        atomic_json(path, value)

    value["source_artifact_hashes"] = []
    value["raw_shard_hashes"] = []
    value["measurement_manifest"] = dict(path=str(tmp_path / "evaluation.json"))
    monkeypatch.setattr(r, "PIN", r.sha256_file(m.ROOT / "results" / (r.SOURCE + ".json")))
    (tmp_path / (r.SOURCE + ".json")).write_bytes(
        (m.ROOT / "results" / (r.SOURCE + ".json")).read_bytes()
    )
    write(value)
    assert r.replay(path)["passed"]
    for field, bad, match in [
        ("owned_statement_coverage", dict(complete=False), "qualification"),
        ("completed_count", 0, "count"),
        ("verdict_class", "positive", "oracle"),
    ]:
        write(dict(value, **{field: bad}))
        with pytest.raises(ValueError, match=match):
            r.replay(path)
    monkeypatch.setattr(r, "reduction", lambda *a: dict(rows=[]))
    write(value)
    with pytest.raises(ValueError, match="reduction"):
        r.replay(path)
    monkeypatch.setattr(r, "validate_current_work_receipt", lambda *a: ["bad"])
    with pytest.raises(ValueError, match="receipt"):
        r.replay(path)
    log = tmp_path / "health.log"
    log.write_text("unrelated health")
    atomic_json(r.HEALTH, dict(log=r.reference(log), passed=False))
    monkeypatch.setattr(
        r, "measurement", lambda raw: dict(measured(raw), shared_union_benefit_score=1)
    )
    assert r.main(args) == 0 and captured[-1]["verdict_class"] == "circular_positive"
    adapter.terminal = lambda *a: [dict(x, passed=False) for x in terminal(*a)]
    assert r.main(args) == 0 and captured[-1]["verdict_class"] == "disqualified"

    def fail(raw):
        raise ValueError("owned_failure")

    monkeypatch.setattr(r, "measurement", fail)
    assert r.main(args) == 0 and captured[-1]["failed_count"] == 3360
    monkeypatch.setattr(r, "validate", lambda *a: r.unqualified())
    assert r.main(args) == 0 and not captured[-1]["required_checks_passed"]


def test_mixed_policy_union_and_roster(tmp_path):
    """REQ-REPORT-8910: an unresolved policy keeps the group open after other policies finish."""
    cohort, scores, truth, trace, selection = fixture()
    selection["choices"][0]["cutoffs"] = dict.fromkeys(m.base.GROUPS, None)
    selection["selection_hash"] = canonical_hash(selection["choices"])
    cert = m.certify(
        cohort,
        scores,
        trace,
        selection,
        lambda arm, rep, group, stage, units: {u["unit_id"]: truth[u["unit_id"]] for u in units},
    )
    result = m.evaluate(cohort, scores, truth, trace, cert)
    assert result["shared_union_benefit_score"] == 0
    assert all(x["certification_labels"] == 960 for x in result["paid_label_ledger"])
    assert all(
        x["certification_policy_input_evaluations"] == 960 + 6 * 240
        for x in result["paid_label_ledger"]
    )
    selection["choices"].pop()
    selection["selection_hash"] = canonical_hash(selection["choices"])
    with pytest.raises(ValueError, match="roster"):
        m.certify(cohort, scores, trace, selection, lambda *a: {})


def test_policy_cost_ledger_tampering():
    """SCENARIO-REPORT-8910-CONTRACT: correct decisions cannot authorize a forged cost ledger."""
    _, cert, _, operands = run_fixture()
    cohort, scores, truth, trace, _ = operands
    cert["paid_label_ledger"][0]["certification_policy_input_evaluations"] += 1
    with pytest.raises(ValueError, match="ledger_policy_cost_drift"):
        m.evaluate(cohort, scores, truth, trace, cert)


def test_requested_label_cache_keeps_unrequested_truth_closed(monkeypatch):
    """REQ-REPORT-8910: cached oracle computation never opens an unrequested world."""
    calls = []

    def labels(units):
        calls.append([u["unit_id"] for u in units])
        return {u["unit_id"]: "refute" for u in units}

    monkeypatch.setattr(m.base.old.old.p, "labels", labels)
    cache = {"seen": "support"}
    units = [dict(unit_id="seen"), dict(unit_id="new"), dict(unit_id="new")]
    assert r.requested_labels(cache, units) == {"seen": "support", "new": "refute"}
    assert calls == [["new"]]
    assert r.requested_labels(cache, [dict(unit_id="new")]) == {"new": "refute"}
    assert calls == [["new"]] and "unrequested" not in cache
