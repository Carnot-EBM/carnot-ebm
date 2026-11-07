"""REQ-REPORT-8868: fixed grouping and independent labels precede every certificate."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import binomtest

from carnot.reporting import conditional_risk_methods_8868 as m
from carnot.reporting import conditional_risk_runner_8868 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(size=16):
    """Identical policies expose weighting mistakes without a favorable fitted result."""
    units = [dict(unit_id=f"{g}/{i:04d}", group=g) for g in m.GROUPS for i in range(size)]
    scores = [
        dict(
            unit_id=u["unit_id"],
            cluster_id=u["unit_id"],
            arm=a,
            seed=s,
            candidate="support",
            probabilities=[0.96, 0.02, 0.02],
        )
        for a, s in m.POLICIES
        for u in units
    ]
    truth = {u["unit_id"]: "support" for u in units}
    return dict(inputs=units), scores, truth, m.draws(size, replicates=2)


def test_groups_ignore_facts_names_and_predicate_order():
    """SCENARIO-REPORT-8868-CONTRACT: only canonical query syntax chooses membership."""
    for name, edges in m.GROUPS.items():
        q = [[f"?v{a}", f"r{i}", f"?v{b}"] for i, (a, b) in enumerate(edges)]
        assert m.group(q) == name
        renamed = [[a.replace("v", "renamed"), p + "x", b.replace("v", "renamed")] for a, p, b in q]
        assert m.group(renamed[::-1]) == name
    with pytest.raises(ValueError, match="five_predicate"):
        m.group([])
    with pytest.raises(ValueError, match="unregistered"):
        m.group([[f"?v{i}", "r" + str(i), "?v0"] for i in range(5)])
    payload = dict(query=q, facts=[["a", "r0", "b", True]])
    assert m.relevant(payload) == m.relevant(
        dict(payload, facts=payload["facts"] + [["x", "unused", "y", True]])
    )


def test_freeze_and_exact_boundaries():
    """REQ-REPORT-8868: real queried facts establish separation; zero errors need169 accepts."""
    first = m.freeze(set(), count=3, fit_count=1)
    structures = {s for w in first["worlds"] for s in w["relevant_structures"]}
    second = m.freeze(structures, count=3, fit_count=1)
    assert structures.isdisjoint(s for w in second["worlds"] for s in w["relevant_structures"])
    assert len([u for u in second["inputs"] if u["split"] == "population"]) == 12
    for u in second["inputs"]:
        assert m.group(u["input"]["query"]) == u["group"]
    with pytest.raises(ValueError, match="freeze_deadline"):
        m.freeze(set(), count=1, fit_count=1, deadline_s=-1)
    assert m.protocol()["alpha_allocation"]["per_terminal_bound"] == 0.01 / 56
    assert m.FLOOR == 169
    assert m.upper(168, 0) > 0.05 and m.upper(169, 0) <= 0.05
    for n, e in [(0, 0), (169, 0), (480, 1), (480, 480)]:
        expected = binomtest(e, n, alternative="less").proportion_ci(1 - m.ALPHA).high if n else 1
        assert m.upper(n, e) == pytest.approx(expected)
    with pytest.raises(ValueError, match="binomial"):
        m.upper(1, 2)


def test_draws_selection_isolation_refusal_and_null():
    """REQ-REPORT-8868: both protocols pay the same labels and never reselect."""
    cohort, scores, truth, trace = fixture()
    assert trace == m.draws(16, replicates=2)
    assert len(m.draws(1024)["replicates"]) == 40
    for rep in trace["replicates"]:
        assert (
            sum(len(v["selection"]) + len(v["certification"]) for v in rep["groups"].values())
            == 2880
        )
        assert all(len(set(v["certification"])) < 480 for v in rep["groups"].values())
    selected = m.select(cohort, scores, truth, trace)
    changed = deepcopy(trace)
    changed["replicates"][0]["groups"]["chain"]["certification"] = [0] * 480
    assert selected == m.select(cohort, scores, truth, changed)
    cert = m.certify(cohort, scores, truth, trace, selected)
    assert all(all(x["served"].values()) for x in cert["choices"])
    rows = m.census(cohort, scores, truth, cert)
    assert len(rows) == 2 * 2 * 7 * 5
    assert m.summarize(rows)["paired_coverage_difference"] == 0
    broken = deepcopy(selected)
    broken["choices"][0]["cutoffs"]["chain"] = 0.5
    with pytest.raises(ValueError, match="selection_commitment"):
        m.certify(cohort, scores, truth, trace, broken)
    refused = m.certify(cohort, scores, dict.fromkeys(truth, "refute"), trace, selected)
    assert not any(any(c["served"].values()) for c in refused["choices"])
    for score in scores:
        score["probabilities"] = [0, 0, 1]
    empty = m.select(cohort, scores, truth, trace)
    assert all(all(v is None for v in x["cutoffs"].values()) for x in empty["choices"])
    assert not any(
        any(c["served"].values()) for c in m.certify(cohort, scores, truth, trace, empty)["choices"]
    )
    with pytest.raises(ValueError, match="roster"):
        m.select(cohort, scores[:-1], truth, trace)


def test_weighting_counterexample_and_controls():
    """REQ-REPORT-8868: selective risk uses served mass; prevalence shift can break pooling."""
    counts = {g: dict(accepted=100, errors=0, denominator=100) for g in m.GROUPS}
    counts["chain"]["errors"] = 10
    balanced = m.mixture(counts, [0.25] * 4)
    stress = m.mixture(counts, [0.7, 0.1, 0.1, 0.1])
    assert balanced["risk"] == pytest.approx(0.025)
    assert stress["risk"] == pytest.approx(0.07)
    counts["chain"]["accepted"] = 10
    counts["chain"]["errors"] = 1
    assert m.mixture(counts, [0.25] * 4)["risk"] == pytest.approx(1 / 310)
    assert (
        m.mixture({g: dict(accepted=0, errors=0, denominator=10) for g in m.GROUPS}, [0.25] * 4)[
            "risk"
        ]
        is None
    )
    controls = m.controls()
    assert all(x["passed"] for x in controls)
    assert m.summarize([])["paired_coverage_difference"] is None


def test_preconditions_reject_missing_changed_and_zero(tmp_path, monkeypatch):
    """REQ-REPORT-8868: original producer and operand bytes are mandatory external inputs."""
    assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "missing_file"
    name = next(iter(r.UPSTREAM))
    path = tmp_path / (name + ".json")
    atomic_json(path, {})
    assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "missing_field"
    atomic_json(path, dict(split_methods_ready_score=0))
    assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == "observed_zero"
    atomic_json(path, dict(split_methods_ready_score=1))
    assert (
        r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"]
        == "rejected_producer"
    )
    pre = r.preconditions(tmp_path / "auth")
    assert pre["external_ready"] and pre["resources"]["passed"]
    assert r.exposed(pre)
    assert r.validate.__doc__
    operand = tmp_path / "operand.txt"
    operand.write_text("original")
    producer = dict(
        experiment_id=8857,
        verdict_class="null",
        split_methods_ready_score=1,
        source_artifact_hashes=[r.reference(operand)],
        code_config_hashes=[],
        raw_shard_hashes=[],
    )
    atomic_json(path, producer)
    monkeypatch.setattr(r, "UPSTREAM", {name: (r.sha256_file(path), "split_methods_ready_score")})
    assert r.preconditions(tmp_path / "bound", tmp_path)["external_ready"]
    operand.write_text("changed")
    assert (
        r.preconditions(tmp_path / "bad", tmp_path)["diagnostics"][0]["error"]
        == "original_operand_drift"
    )
    monkeypatch.setattr(r, "DESIGN", str(tmp_path / "absent-design"))
    assert (
        r.preconditions(tmp_path / "design", tmp_path)["diagnostics"][-1]["upstream"]
        == "preserved_v758_design"
    )


def test_measure_reduction_and_semantic_mutations(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8868-CONTRACT: replay recomputes each stage from immutable operands."""
    freeze, draws = m.freeze, m.draws
    monkeypatch.setattr(
        m, "freeze", lambda exposed, **kw: freeze(exposed, **(kw or dict(count=3, fit_count=1)))
    )
    monkeypatch.setattr(
        m, "draws", lambda size, **kw: draws(size if kw else 3, replicates=kw.get("replicates", 2))
    )
    pre = dict(references=[])
    result = r.measure(tmp_path, pre)
    assert result == r.reduction(tmp_path, pre)
    for name, mutate, match in [
        ("protocol", lambda v: v.update(seed=0), "protocol_drift"),
        ("pilot", lambda v: v.update(projected_total_s=99999), "pilot_drift"),
        ("population", lambda v: v.update(inputs=[]), "population_or_draw"),
        ("scores", lambda v: v["rows"][0].update(candidate="refute"), "prediction_drift"),
        (
            "selection_labels",
            lambda v: v["labels"].update({next(iter(v["labels"])): "bad"}),
            "label_drift",
        ),
        ("selection", lambda v: v.update(selection_hash="bad"), "stage_reduction"),
    ]:
        path = tmp_path / (name + ".json")
        original = path.read_bytes()
        value = json.loads(original)
        mutate(value)
        atomic_json(path, value)
        access = json.loads((tmp_path / "access_order.json").read_bytes())
        saved = deepcopy(access)
        for event in access["events"]:
            if event["stage"] == name:
                event.update(r.reference(path))
        atomic_json(tmp_path / "access_order.json", access)
        with pytest.raises(ValueError, match=match):
            r.reduction(tmp_path, pre)
        path.write_bytes(original)
        atomic_json(tmp_path / "access_order.json", saved)
    access = json.loads((tmp_path / "access_order.json").read_bytes())
    access["events"].reverse()
    atomic_json(tmp_path / "access_order.json", access)
    with pytest.raises(ValueError, match="access_order"):
        r.reduction(tmp_path, pre)
    monkeypatch.setattr(r, "MEASUREMENT_DEADLINE_S", -1)
    with pytest.raises(ValueError, match="measurement_deadline"):
        r.measure(tmp_path / "late", pre)
    monkeypatch.setattr(r, "MEASUREMENT_DEADLINE_S", 1800)
    monkeypatch.setattr(m, "controls", lambda: [dict(passed=False)])
    with pytest.raises(ValueError, match="pilot_control"):
        r.measure(tmp_path / "failed", pre)
    corrupted = m.corrupt(dict(inputs=[dict(input=dict(facts=[["a", "p", "b", True]]))]))
    assert corrupted["inputs"][0]["input"]["facts"][-1] == ["a", "p", "b", False]


def test_replay_identity_contract_and_readiness(tmp_path, monkeypatch):
    """REQ-REPORT-8868: a new checksum cannot grant an unsupported scientific claim."""
    path = tmp_path / "artifact.json"
    receipt = r.build_current_work_receipt(
        run_id="x",
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
        experiment_id=8868,
        task_id=m.TASK,
        source_artifact_hashes=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        current_work_receipt=receipt,
        group_definition_hash=canonical_hash(m.GROUPS),
        run_date="20261106",
        milestone="2026.11.759",
        completed_count=0,
        failed_count=0,
        censored_count=3360,
        conditional_methods_ready_score=0,
        required_checks_passed=False,
        verdict_class="blocked",
        **{
            k: m.protocol()[k]
            for k in [
                "alpha_allocation",
                "exact_sample_requirements",
                "risk_bound_construction",
                "mixture_weights",
                "expected_row_counts",
            ]
        },
    )

    def write(v):
        v = deepcopy(v)
        v["reproducibility_checksum"] = canonical_hash(v)
        atomic_json(path, v)

    write(value)
    assert r.replay(path)["passed"]
    for field, bad, match in [
        ("experiment_id", 1, "producer"),
        ("alpha_allocation", {}, "frozen"),
        ("group_definition_hash", "bad", "group_or"),
        ("completed_count", 1, "count"),
        ("conditional_methods_ready_score", 1, "readiness"),
    ]:
        v = dict(value, **{field: bad})
        write(v)
        with pytest.raises(ValueError, match=match):
            r.replay(path)
    write(value)
    v = json.loads(path.read_bytes())
    v["run_date"] = "bad"
    atomic_json(path, v)
    with pytest.raises(ValueError, match="checksum"):
        r.replay(path)
    operand = tmp_path / "operand.txt"
    operand.write_text("original")
    bound = dict(value, source_artifact_hashes=[r.reference(operand)])
    write(bound)
    assert r.replay(path)["passed"]
    operand.write_text("changed")
    with pytest.raises(ValueError, match="operand_hash"):
        r.replay(path)
    complete = dict(
        value,
        rows=[dict(row, status="completed") for row in value["rows"]],
        completed_count=3360,
        censored_count=0,
        population_manifest=dict(path=str(tmp_path / "population.json")),
        preconditions_checked={},
    )
    monkeypatch.setattr(r, "reduction", lambda raw, pre: dict(rows=complete["rows"]))
    write(complete)
    assert r.replay(path)["passed"]
    monkeypatch.setattr(r, "reduction", lambda raw, pre: dict(rows=[]))
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(path)
    v = dict(value, current_work_receipt={})
    write(v)
    monkeypatch.setattr(r, "validate_current_work_receipt", lambda v: ["bad"])
    with pytest.raises(ValueError, match="current_receipt"):
        r.replay(path)


def test_cli_cold_blocked_foreign_directory(tmp_path):
    """SCENARIO-REPORT-8868-CONTRACT: real CLI cold replay needs no checkout cwd or PYTHONPATH."""
    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    missing = tmp_path / "absent.json"
    result = subprocess.run(
        command + ["--cold-replay", str(missing)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 1 and "replay_rejected" in result.stdout
    output = tmp_path / "results" / (m.NAME + ".json")
    result = subprocess.run(
        command + ["--private-output", str(output), "--sources", str(tmp_path / "absent")],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    result = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_main_terminal_controls_and_validation_adapter(tmp_path, monkeypatch):
    """REQ-REPORT-8868: only checked complete methods release readiness; owned failures disqualify."""
    from types import SimpleNamespace

    valid = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(complete=True, percent=100, files=[]),
        raw_references=[],
    )
    adapter = SimpleNamespace(validate=lambda raw, private: valid)
    monkeypatch.setattr(r, "namespace", lambda *args: adapter)
    assert r.validate(tmp_path, tmp_path) == valid
    assert adapter.m is m
    monkeypatch.setattr(
        r,
        "preconditions",
        lambda *a: dict(
            external_ready=True, resources=dict(passed=True), diagnostics=[], references=[]
        ),
    )

    def measured(raw, pre):
        for name in ["population", "fit", "draws"]:
            atomic_json(raw / (name + ".json"), {})
        result = r.scheduled("control")
        result["rows"] = [dict(x, status="completed", metric=0) for x in result["rows"]]
        return result

    monkeypatch.setattr(r, "measure", measured)
    captures = []
    monkeypatch.setattr(
        r, "publish_primary", lambda out, value, replay: captures.append(deepcopy(value))
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
    atomic_json(health, checked("health", [], tmp_path / "health"))
    monkeypatch.setattr(r, "HEALTH", health)
    command = ["--output", str(tmp_path / (m.NAME + ".json"))]
    assert r.main(command) == 0
    assert captures[-1]["conditional_methods_ready_score"] == 1
    assert captures[-1]["repository_health"]

    def rejected(name, argv, logs, **kw):
        return dict(checked(name, argv, logs, **kw), passed=False)

    monkeypatch.setattr(r, "stream", rejected)
    assert r.main(command) == 0
    assert captures[-1]["verdict_class"] == "disqualified"

    def failed(raw, pre):
        atomic_json(raw / "fit.json", dict(numerical_training_calls=3))
        raise ValueError("owned_control")

    monkeypatch.setattr(r, "measure", failed)
    assert r.main(command) == 0
    assert captures[-1]["owned_errors"] == ["owned_control"]
    assert captures[-1]["failed_count"] == 3360
    assert captures[-1]["numerical_training_calls"] == 3
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))])
    monkeypatch.setattr(r, "replay", lambda p: dict(passed=True))
    assert r.main(["--cold-replay", str(tmp_path / "fixture")]) == 0


def test_acceptance_mass_and_certificate_tampering():
    """REQ-REPORT-8868: pooled bounds weight each group's actual served mass."""
    cohort, scores, truth, trace = fixture(8)
    for score in scores:
        if score["unit_id"].startswith("chain/") and int(score["unit_id"].split("/")[1]) % 2:
            score["probabilities"] = [0, 0, 1]
    selected = m.select(cohort, scores, truth, trace)
    certificates = m.certify(cohort, scores, truth, trace, selected)
    choice = next(c for c in certificates["choices"] if c["protocol"] == "pooled")
    mass = [
        choice["bounds"][g]["population_accepted"] / choice["bounds"][g]["population_denominator"]
        for g in m.GROUPS
    ]
    assert mass == [0.5, 1, 1, 1]
    assert choice["pooled_risk_upper"] == pytest.approx(
        sum(w * choice["bounds"][g]["risk_upper"] for w, g in zip(mass, m.GROUPS, strict=True))
        / sum(mass)
    )
    certificates["choices"][0]["served"]["chain"] = False
    with pytest.raises(ValueError, match="certificate_commitment"):
        m.census(cohort, scores, truth, certificates)
    assert len(m.freeze(set(), count=60, fit_count=1)["worlds"]) == 244
