"""REQ-REPORT-8880: paid draws and finite-look safety precede any benefit claim."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from scipy.stats import binomtest

from carnot.reporting import sequential_certificate_methods_8880 as m
from carnot.reporting import sequential_certificate_runner_8880 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(size=8):
    """Identical fixed policies make every expected stopping cost independently visible."""
    units = [dict(unit_id=f"{g}/{i:04d}", group=g) for g in m.base.GROUPS for i in range(size)]
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
    return (
        dict(inputs=units),
        scores,
        dict.fromkeys([u["unit_id"] for u in units], "support"),
        m.draws(size, replicates=2),
    )


def test_exact_bounds_and_union():
    """REQ-REPORT-8880: all possible looks, including unvisited looks, receive error budget."""
    p = m.protocol()
    assert p["look_schedule"] == [120, 240, 480, 960]
    assert sum(
        p["alpha_allocation"][k] for k in ["sequential_total", "fixed_total"]
    ) == pytest.approx(0.01)
    for alpha in m.ALPHAS.values():
        floor = m.minimum(alpha)
        assert m.base.old.upper(floor - 1, 0, alpha) > 0.05
        assert m.base.old.upper(floor, 0, alpha) <= 0.05
        for n, e in [(0, 0), (floor, 0), (960, 3), (960, 960)]:
            expected = binomtest(e, n, alternative="less").proportion_ci(1 - alpha).high if n else 1
            assert m.base.old.upper(n, e, alpha) == pytest.approx(expected)
    assert all(c["passed"] for c in m.controls())


def test_selection_isolation_stopping_refusals_and_accounting():
    """SCENARIO-REPORT-8880-CONTRACT: all policies share draws, never independent n."""
    cohort, scores, truth, trace = fixture()
    chosen = m.select(cohort, scores, truth, trace)
    changed = deepcopy(trace)
    changed["replicates"][0]["groups"]["chain"]["certification"] = [0] * 960
    assert chosen == m.select(cohort, scores, truth, changed)
    result = m.certify(cohort, scores, truth, trace, chosen)
    assert len(result["rows"]) == 2 * 2 * 7 * 4
    assert {x["candidate_draws"] for x in result["rows"] if x["protocol"] == "sequential"} == {240}
    assert {x["candidate_draws"] for x in result["rows"] if x["protocol"] == "fixed"} == {960}
    assert all(x["certificate_issued"] for x in result["rows"])
    assert all(x["total_labels"] == 240 + x["candidate_draws"] for x in result["rows"])
    assert result["actual_research_labels"] == 2 * 4 * (240 + 960)
    assert all(
        x["sequential_total_labels"] == 4 * (240 + 240) for x in result["shared_union_costs"]
    )
    refused = m.certify(cohort, scores, dict.fromkeys(truth, "refute"), trace, chosen)
    assert not any(x["certificate_issued"] for x in refused["rows"])
    assert all(x["candidate_draws"] == 960 for x in refused["rows"])
    assert all(len(x["looks"]) == 4 for x in refused["rows"] if x["protocol"] == "sequential")
    broken = deepcopy(chosen)
    broken["choices"][0]["cutoffs"]["chain"] = 0.5
    with pytest.raises(ValueError, match="selection_commitment"):
        m.certify(cohort, scores, truth, trace, broken)
    for s in scores:
        s["probabilities"] = [0, 0, 1]
    empty = m.select(cohort, scores, truth, trace)
    assert all(c is None for x in empty["choices"] for c in x["cutoffs"].values())
    assert not any(
        x["certificate_issued"] for x in m.certify(cohort, scores, truth, trace, empty)["rows"]
    )


def test_new_population_and_draws():
    """REQ-REPORT-8880: relevant structures, not unused markers, define separation."""
    first = m.freeze(set(), count=2)
    exposed = {s for w in first["worlds"] for s in w["relevant_structures"]}
    second = m.freeze(exposed, count=2)
    assert exposed.isdisjoint(s for w in second["worlds"] for s in w["relevant_structures"])
    assert all(u["unit_id"].startswith("v760/8880/") for u in second["inputs"])
    assert all(m.base.group(u["input"]["query"]) == u["group"] for u in second["inputs"])
    trace = m.draws(1024)
    assert len(trace["replicates"]) == 40 and trace == m.draws(1024)
    assert all(
        len(v["selection"]) == 240 and len(v["certification"]) == 960
        for x in trace["replicates"]
        for v in x["groups"].values()
    )
    assert any(
        len(set(v["certification"])) < 960
        for x in trace["replicates"]
        for v in x["groups"].values()
    )


def test_external_gate_diagnostics(tmp_path):
    """REQ-REPORT-8880: missing, zero and foreign assertions remain distinct blocked gates."""
    assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    name, (_, field) = next(iter(r.UPSTREAM.items()))
    path = tmp_path / (name + ".json")
    for value, check in [
        ({}, "missing_field"),
        ({field: 0}, "observed_zero"),
        ({field: 1}, "rejected_producer"),
    ]:
        atomic_json(path, value)
        assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == check
    pre = r.preconditions(tmp_path / "originals")
    assert pre["external_ready"] and pre["resources"]["passed"]


def test_real_cli_blocked_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8880-CONTRACT: the actual CLI works outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    path = tmp_path / (m.NAME + ".json")
    result = subprocess.run(
        command + ["--private-output", str(path), "--sources", str(tmp_path / "missing")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(path.read_bytes())["verdict_class"] == "blocked"
    result = subprocess.run(
        command + ["--cold-replay", str(path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0 and "replay_passed" in result.stdout
    value = json.loads(path.read_bytes())
    value["sequential_methods_ready_score"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(path, value)
    result = subprocess.run(
        command + ["--cold-replay", str(path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 1 and "replay_rejected" in result.stdout


def test_measurement_replay_and_rehashed_tampering(tmp_path, monkeypatch):
    """REQ-REPORT-8880: replay derives stages from original operands rather than assertions."""
    import shutil

    freeze, draws = m.freeze, m.draws
    monkeypatch.setattr(m, "freeze", lambda exposed, **kw: freeze(exposed, **(kw or dict(count=3))))
    monkeypatch.setattr(
        m, "draws", lambda size, **kw: draws(size if kw else 3, replicates=kw.get("replicates", 2))
    )
    raw = tmp_path / "measurement"
    raw.mkdir()
    shutil.copyfile(m.ROOT / "results" / (r.SOURCE + ".json"), raw / (r.SOURCE + ".json"))
    pre = dict(references=[])
    result = r.measurement(raw, pre)
    assert result == r.reduction(raw, pre)
    assert result["actual_research_labels"] == 9600
    for stage, change in [
        ("protocol", lambda v: v.update(seed=0)),
        ("pilot", lambda v: v.update(controls=[])),
        ("population", lambda v: v.update(inputs=[])),
        ("draws", lambda v: v.update(seed=0)),
        ("fit", lambda v: v.update(parameters=[])),
        ("scores", lambda v: v.update(rows=[])),
        ("selection_labels", lambda v: v.update(paid_labels=0)),
        ("selection", lambda v: v.update(selection_hash="bad")),
        ("certification_labels", lambda v: v.update(paid_labels=0)),
        ("certificates", lambda v: v.update(rows=[])),
    ]:
        path = raw / (stage + ".json")
        saved, access = path.read_bytes(), (raw / "access_order.json").read_bytes()
        value = json.loads(saved)
        change(value)
        atomic_json(path, value)
        events = json.loads(access)
        for e in events["events"]:
            if e["stage"] == stage:
                e.update(r.reference(path))
        atomic_json(raw / "access_order.json", events)
        with pytest.raises(ValueError, match="measurement_drift"):
            r.reduction(raw, pre)
        path.write_bytes(saved)
        (raw / "access_order.json").write_bytes(access)
    access = json.loads((raw / "access_order.json").read_bytes())
    access["events"].reverse()
    atomic_json(raw / "access_order.json", access)
    with pytest.raises(ValueError, match="access_order"):
        r.reduction(raw, pre)
    monkeypatch.setattr(r, "DEADLINE", -1)
    with pytest.raises(ValueError, match="measurement_deadline"):
        r.measurement(raw, pre)
    monkeypatch.setattr(r, "DEADLINE", 1800)
    with monkeypatch.context() as patch:
        patch.setattr(m, "controls", lambda: [dict(passed=False)])
        with pytest.raises(ValueError, match="pilot_control"):
            r.measurement(raw, pre)
    atomic_json(raw / "pilot_runtime.json", dict(projected_total_s=99999))
    with pytest.raises(ValueError, match="pilot_runtime"):
        r.measurement(raw, pre, replaying=True)


def test_main_owned_failures_and_validation_adapter(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8880-CONTRACT: only checked complete methods release readiness."""
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

    def measured(raw, pre):
        for name in ["population", "draws", "selection", "fit"]:
            atomic_json(raw / (name + ".json"), {})
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
    command = ["--output", str(tmp_path / (m.NAME + ".json"))]
    assert r.main(command) == 0 and captures[-1]["sequential_methods_ready_score"] == 1
    monkeypatch.setattr(r, "stream", lambda *a, **kw: dict(checked(*a, **kw), passed=False))
    assert r.main(command) == 0 and captures[-1]["verdict_class"] == "disqualified"

    def failed(raw, pre):
        raise ValueError("owned_control")

    monkeypatch.setattr(r, "measurement", failed)
    assert r.main(command) == 0 and captures[-1]["failed_count"] == 2240
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))])
    monkeypatch.setattr(r, "replay", lambda p: dict(passed=True))
    assert r.main(["--cold-replay", str(tmp_path / "fixture")]) == 0


def test_replay_identity_counts_and_receipt_mutations(tmp_path, monkeypatch):
    """REQ-REPORT-8880: recalculated checksums cannot authorize changed gates or counts."""
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
        experiment_id=8880,
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
        sequential_methods_ready_score=0,
        required_checks_passed=False,
        verdict_class="blocked",
        **{
            k: m.protocol()[k]
            for k in [
                "look_schedule",
                "alpha_allocation",
                "bound_proof",
                "label_cost_definition",
                "exact_sample_requirements",
            ]
        },
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
        ("alpha_allocation", {}, "frozen"),
        ("completed_count", 1, "count"),
        ("independent_count", 40, "count"),
        ("sequential_methods_ready_score", 1, "readiness"),
    ]:
        write(dict(value, **{field: bad}))
        with pytest.raises(ValueError, match=match):
            r.replay(path)
    write(value)
    broken = json.loads(path.read_bytes())
    broken["run_date"] = "bad"
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
        population_manifest=dict(path=str(tmp_path / "population.json")),
        preconditions_checked={},
    )
    for name in r.UPSTREAM:
        (tmp_path / (name + ".json")).write_bytes(
            (m.ROOT / "results" / (name + ".json")).read_bytes()
        )
    monkeypatch.setattr(r, "reduction", lambda raw, pre: dict(rows=full["rows"]))
    write(full)
    assert r.replay(path)["passed"]
    qualified = dict(
        full,
        required_checks_passed=True,
        verdict_class="null",
        sequential_methods_ready_score=1,
        owned_statement_coverage=dict(
            complete=True,
            percent=100,
            files=[dict(path=p, missing=[], excluded=[]) for p in m.OWNED],
        ),
        validation_receipts=[dict(passed=True)],
    )
    write(qualified)
    assert r.replay(path)["passed"]
    write(
        dict(
            full,
            required_checks_passed=True,
            verdict_class="null",
            sequential_methods_ready_score=1,
            owned_statement_coverage=dict(complete=False, percent=0, files=[]),
            validation_receipts=[dict(passed=False)],
        )
    )
    with pytest.raises(ValueError, match="qualification_drift"):
        r.replay(path)
    write(full)
    monkeypatch.setattr(r, "reduction", lambda raw, pre: dict(rows=[]))
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(path)
    (tmp_path / (r.SOURCE + ".json")).write_text("changed")
    with pytest.raises(ValueError, match="original_producer"):
        r.replay(path)
