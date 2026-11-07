"""REQ-REPORT-8893: a shared final decision makes coverage inclusion deterministic."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from scipy.stats import binomtest

from carnot.reporting import shared_terminal_methods_8893 as m
from carnot.reporting import shared_terminal_runner_8893 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(size=2, replicates=1):
    """Identical safe policies expose costs without making a learned-performance claim."""
    units = [dict(unit_id=f"{g}/{i}", group=g) for g in m.base.GROUPS for i in range(size)]
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
    cohort, truth = dict(inputs=units), dict.fromkeys([u["unit_id"] for u in units], "support")
    trace = m.draws(size, replicates=replicates)
    chosen = m.select(cohort, scores, truth, trace)
    return cohort, scores, truth, trace, chosen


def test_registered_allocations_and_common_final():
    """REQ-REPORT-8893: distinct early and shared terminal events spend exactly .01."""
    p = m.protocol()
    assert p["population_per_group"] == 1024 and p["replicates"] == 40
    assert p["look_schedule"] == [120, 240, 480, 960]
    assert m.EARLY == 0.005 / (7 * 4 * 3) and m.FINAL == 0.005 / (7 * 4)
    assert p["alpha_allocation"]["distinct_total"] <= 0.01
    assert p["alpha_allocation"]["final_event_count"] == 28
    assert m.common_final(m.FINAL, m.FINAL, [0] * 960, [0] * 960)
    assert not m.common_final(m.EARLY, m.FINAL, [0] * 960, [0] * 960)
    assert not m.common_final(m.FINAL, m.FINAL, [0] * 960, [1] * 960)
    assert not m.common_final(m.FINAL, m.FINAL, [0] * 959, [0] * 959)
    assert all(c["passed"] for c in m.controls())


def test_count_transitions_and_exact_bound_boundaries():
    """SCENARIO-REPORT-8893-CONTRACT: empty support refuses and every boundary is exact."""
    for accepted, errors in [(0, 0), (120, 120), (120, 0)]:
        path = m.decision_path([(accepted, errors)] * 4)
        assert path[0]["certificate_issued"] is False
        for event in path:
            n, e, alpha = event["accepted"], event["errors"], event["alpha"]
            independent = (
                binomtest(e, n, alternative="less").proportion_ci(1 - alpha).high if n else 1
            )
            assert event["risk_upper"] == pytest.approx(independent, abs=1e-10)
    path = m.decision_path([(look, 0) for look in m.LOOKS])
    assert [x["certificate_issued"] for x in path] == [False, True, True, True]
    # Every possible final accepted count crosses the threshold at the same integer.
    for n in range(961):
        assert (m.base.old.upper(n, 0, m.FINAL) <= 0.05) == (n >= m.old.minimum(m.FINAL))
    for stats in [
        [],
        [(0, 0)] * 3,
        [(121, 0)] * 4,
        [(1, 2)] * 4,
        [(1.0, 0)] * 4,
        [(1, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (1, 0), (1, 1), (1, 1)],
        [(-1, 0)] * 4,
    ]:
        with pytest.raises(ValueError, match="statistic"):
            m.decision_path(stats)


def test_certification_coverage_costs_and_unsafe_controls():
    """REQ-REPORT-8893: fixed service is included; refusals still consume all candidates."""
    cohort, scores, truth, trace, chosen = fixture(replicates=40)
    cert = m.certify(cohort, scores, truth, trace, chosen)
    result = m.evaluate(cohort, scores, truth, trace, cert)
    assert len(result["rows"]) == 2240
    assert result["pathwise_inclusion_violations"] == 0
    assert len(result["fixed_alpha_sensitivity_rows"]) == 1120
    assert all(x["fixed"] == 4800 and x["sequential"] == 1920 for x in result["standalone_costs"])
    assert all(x["sequential_total_labels"] == 1920 for x in result["shared_union_costs"])
    assert result["actual_label_ledger"]["joint_calibration_labels"] == 192000
    assert len(result["standalone_learned_policy_costs"]) == 200
    assert result["label_savings_score"] == 1
    assert all(x["independent_count"] == 40 for x in result["paired_intervals"])
    assert all(x["served_risk"] == 0 for x in result["rows"])
    assert all(x["metric"] == 1 for x in result["census_risk_rows"])
    bad = dict.fromkeys(truth, "refute")
    unsafe = m.certify(cohort, scores, bad, trace, chosen)
    refused = m.evaluate(cohort, scores, bad, trace, unsafe)
    assert all(not x["certificate_issued"] for x in refused["rows"])
    assert all(x["sequential_total_labels"] == 4800 for x in refused["shared_union_costs"])
    assert refused["label_savings_score"] == 0
    # One unfinished policy per group prevents union savings despite six early stops.
    mixed = deepcopy(cert)
    for row in mixed["rows"]:
        if row["protocol"] == "sequential" and (row["arm"], row["seed"]) == m.base.POLICIES[0]:
            row.update(candidate_draws=960, total_labels=1200, certificate_issued=False)
    assert all(x["sequential_total_labels"] == 4800 for x in m.ledgers(mixed["rows"])[1])
    with pytest.raises(ValueError, match="selection_commitment"):
        m.certify(cohort, scores, truth, trace, dict(chosen, selection_hash="changed"))
    with pytest.raises(ValueError, match="candidate_prefix"):
        short = deepcopy(trace)
        short["replicates"][0]["groups"]["chain"]["certification"].pop()
        m.certify(cohort, scores, truth, short, chosen)


def test_external_gates_and_private_cold_cli(tmp_path):
    """SCENARIO-REPORT-8893-CONTRACT: external absence blocks and rehashed changes fail."""
    assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    name, (_, field) = next(iter(r.UPSTREAM.items()))
    for value, check in [
        ({}, "missing_field"),
        ({field: 0}, "observed_zero"),
        ({field: 1}, "rejected_producer"),
    ]:
        atomic_json(tmp_path / (name + ".json"), value)
        assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == check
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    output = tmp_path / (m.NAME + ".json")

    def child(options):
        return subprocess.run(
            command + options, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120
        )

    run = child(["--private-output", str(output), "--sources", str(tmp_path / "absent")])
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["shared_terminal_ready_score"] == 0
    assert child(["--cold-replay", str(output)]).returncode == 0
    value["shared_terminal_ready_score"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert child(["--cold-replay", str(output)]).returncode == 1
    assert child(["--date", "wrong"]).returncode != 0
    assert child(["--private-output", str(m.ROOT / "results" / output.name)]).returncode != 0


def test_retained_original_bytes_are_authenticated(tmp_path, monkeypatch):
    """REQ-REPORT-8893: changed paths may resolve only to preserved byte-identical operands."""
    from types import SimpleNamespace

    original, retained = tmp_path / "live", tmp_path / "retained"
    original.write_text("changed")
    retained.write_text("original")
    monkeypatch.setattr(r, "PRESERVED", {original: retained})

    def checked(raw, sources):
        raw.mkdir(parents=True)
        assert checker.sha256_file(original) == r.sha256_file(retained)
        assert checker.reference(original)["path"] == str(retained)
        return dict(
            references=[dict(path=str(original), sha256=r.sha256_file(retained))],
            diagnostics=[],
            resources=dict(passed=True),
            external_ready=True,
        )

    checker = SimpleNamespace(preconditions=checked)
    monkeypatch.setattr(r, "namespace", lambda *a: checker)
    monkeypatch.setattr(r.authority, "authority", lambda root: dict(ready=True))
    pre = r.preconditions(tmp_path / "raw", tmp_path)
    assert pre["external_ready"]
    assert pre["references"][0]["original_path"] == str(original)
    assert Path(pre["references"][0]["path"]).read_text() == "original"
    monkeypatch.setattr(r.authority, "authority", lambda root: dict(ready=False))
    rejected = r.preconditions(tmp_path / "raw2", tmp_path)
    assert not rejected["external_ready"]
    assert rejected["diagnostics"][0]["artifact_field"] == "execution_authority"


def test_actual_original_provenance_is_readable(tmp_path):
    """REQ-REPORT-8893: all imported original assertions and byte hashes must authenticate."""
    pre = r.preconditions(tmp_path / "originals", m.ROOT / "results")
    assert pre["external_ready"], pre["diagnostics"]
    assert pre["resources"]["passed"] and pre["authority"]["ready"]


def test_new_population_and_measurement_replay(tmp_path, monkeypatch):
    """REQ-REPORT-8893: fresh worlds precede labels and replay reconstructs both certificates."""
    from types import SimpleNamespace

    cohort, scores, truth, trace, chosen = fixture()
    for u in cohort["inputs"]:
        u["split"] = "population"
    cert = dict(m.certify(cohort, scores, truth, trace, chosen), historical_training_calls=5)
    builder = SimpleNamespace(
        freeze=lambda exposed, count: dict(worlds=[dict(world_id="v760/8880/x")])
    )
    with monkeypatch.context() as patch:
        patch.setattr(m, "namespace", lambda *a: builder)
        assert m.freeze(set(), count=1)["worlds"][0]["world_id"] == "v761/8893/x"
        assert builder.SEED == m.SEED

    def measured(raw, pre, replaying=False):
        if not replaying:
            for stage, value in [
                ("population", cohort),
                ("scores", dict(rows=scores)),
                ("draws", trace),
            ]:
                atomic_json(raw / (stage + ".json"), value)
            events = []
            for stage in m.protocol()["access_order"][:-2]:
                path = raw / (stage + ".json")
                if not path.exists():
                    atomic_json(path, {})
                events.append(dict(stage=stage, monotonic_ns=len(events), **r.reference(path)))
            atomic_json(raw / "access_order.json", dict(events=events))
        return cert

    adapter = SimpleNamespace(measurement=measured)
    monkeypatch.setattr(r, "namespace", lambda *a: adapter)
    monkeypatch.setattr(m.base.old.old.p, "labels", lambda *a: truth)
    result = r.measurement(tmp_path, {})
    assert r.reduction(tmp_path, {}) == result
    assert result["historical_training_calls"] == 5
    for stage in ["census_labels", "evaluation"]:
        p = tmp_path / (stage + ".json")
        saved = p.read_bytes()
        atomic_json(p, {})
        with pytest.raises(ValueError, match="measurement_drift"):
            r.measurement(tmp_path, {}, replaying=True)
        p.write_bytes(saved)
    order = json.loads((tmp_path / "access_order.json").read_bytes())
    order["events"].reverse()
    atomic_json(tmp_path / "access_order.json", order)
    with pytest.raises(ValueError, match="access_order"):
        r.reduction(tmp_path, {})


def test_runner_terminal_controls_and_owned_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8893-CONTRACT: owned failures cannot become ready or positive."""
    from types import SimpleNamespace

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
    health = tmp_path / "health.json"
    monkeypatch.setattr(r, "HEALTH", health)
    monkeypatch.setattr(
        r,
        "preconditions",
        lambda *a: dict(
            external_ready=True, resources=dict(passed=True), references=[], diagnostics=[]
        ),
    )

    def measured(raw, pre):
        atomic_json(raw / "evaluation.json", {})
        atomic_json(raw / "draws.json", {})
        result = r.scheduled("fixture")
        result["rows"] = [
            dict(x, status="completed", missing_reason=None, metric=0) for x in result["rows"]
        ]
        return result

    captures = []
    monkeypatch.setattr(r, "measurement", measured)
    monkeypatch.setattr(
        r, "publish_primary", lambda out, value, validator: captures.append(deepcopy(value))
    )

    def terminal(candidate, raw):
        reports = []
        for name in ["adversarial", "strict_rows", "cold_replay"]:
            report = dict(name=name, passed=True)
            for suffix in ["stdout", "stderr"]:
                path = raw / (name + "." + suffix)
                path.write_text("checked fixture")
                report[suffix + "_path"] = str(path)
            reports.append(report)
        return reports

    adapter.terminal = terminal
    args = ["--output", str(tmp_path / (m.NAME + ".json"))]
    assert r.main(args) == 0 and captures[-1]["shared_terminal_ready_score"] == 1
    assert (
        captures[-1]["target_model"]
        == "seven frozen numerical policies: two baselines and five retained heads; no pretrained model"
    )
    from scripts.adversarial_verify import check_methodology_present

    flags = []
    check_methodology_present(captures[-1], flags)
    assert not flags
    log = tmp_path / "health.log"
    log.write_text("unrelated health timeout")
    atomic_json(health, dict(log=r.reference(log), scope="repository_health", passed=False))
    monkeypatch.setattr(
        r, "measurement", lambda raw, pre: dict(measured(raw, pre), label_savings_score=1)
    )
    assert r.main(args) == 0 and captures[-1]["verdict_class"] == "circular_positive"
    adapter.terminal = lambda *a: [dict(x, passed=False) for x in terminal(*a)]
    assert r.main(args) == 0 and captures[-1]["verdict_class"] == "disqualified"
    log.write_text("changed")
    with pytest.raises(ValueError, match="repository_health"):
        r.main(args)
    health.unlink()

    def failed(raw, pre):
        raise ValueError("owned_failure")

    monkeypatch.setattr(r, "measurement", failed)
    assert r.main(args) == 0 and captures[-1]["failed_count"] == 2240
    monkeypatch.setattr(
        r,
        "preconditions",
        lambda *a: dict(
            external_ready=False,
            resources=dict(passed=False),
            references=[],
            diagnostics=[dict(check="missing_file")],
        ),
    )
    assert r.main(args) == 0 and captures[-1]["verdict_class"] == "disqualified"


def test_replay_identity_counts_hashes_and_readiness(tmp_path, monkeypatch):
    """REQ-REPORT-8893: rehashed bytes cannot invent coverage, complete rows or scientific benefit."""
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
        experiment_id=8893,
        task_id=m.TASK,
        milestone="2026.11.761",
        run_date="20261107",
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
        intended_count=2240,
        shared_terminal_ready_score=0,
        required_checks_passed=False,
        verdict_class="blocked",
        preconditions_checked={},
        alpha_allocation=m.protocol()["alpha_allocation"],
        coverage_inclusion_proof=m.INCLUSION,
        fixed_final_test=m.protocol()["fixed_final_test"],
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
        ("fixed_final_test", {}, "protocol"),
        ("shared_terminal_ready_score", 1, "readiness"),
        ("label_savings_score", 1, "benefit"),
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
    write(dict(full, label_savings_score=1, verdict_class="disqualified"))
    assert r.replay(path)["passed"]
    qualified = dict(
        full,
        required_checks_passed=True,
        verdict_class="null",
        shared_terminal_ready_score=1,
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
    monkeypatch.setattr(r, "reduction", lambda *a: dict(rows=[]))
    write(full)
    with pytest.raises(ValueError, match="reduction"):
        r.replay(path)
    (tmp_path / (r.SOURCE + ".json")).write_text("changed original")
    write(full)
    with pytest.raises(ValueError, match="original_producer"):
        r.replay(path)
