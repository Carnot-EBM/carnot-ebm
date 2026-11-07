"""REQ-REPORT-8870: distinguish frozen measurement, utility and robustness."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import conditional_risk_evaluation_8870 as m
from carnot.reporting import conditional_risk_evaluation_runner_8870 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(size=4, replicates=40):
    """Shared identical policies make any claimed comparative advantage an error."""
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
    cohort, truth = dict(inputs=units), {u["unit_id"]: "support" for u in units}
    trace = m.base.draws(size, replicates=replicates)
    selection = m.base.select(cohort, scores, truth, trace)
    cert = m.base.certify(cohort, scores, truth, trace, selection)
    return cohort, scores, truth, trace, selection, cert


def test_weighted_counts_refusals_and_corruption():
    """SCENARIO-REPORT-8870-CONTRACT: risks weight accepted mass, not group risks."""
    cohort, scores, truth, _, _, cert = fixture(replicates=2)
    result = m.evaluate(cohort, scores, truth, cert)
    assert len(result["rows"]) == 140 and len(result["group_count_rows"]) == 112
    assert all(x["metric"] == 1 for x in result["rows"])
    refused = deepcopy(cert)
    for c in refused["choices"]:
        c["served"]["chain"] = False
    refused["certificate_hash"] = canonical_hash(refused["choices"])
    result = m.evaluate(cohort, scores, truth, refused)
    assert all(
        g["accepted"] == 0 and g["risk"] is None
        for g in result["group_count_rows"]
        if g["group"] == "chain"
    )
    assert all(g["selected_population_accepted"] == 4 for g in result["group_count_rows"])
    assert result["rows"][1]["metric"] == pytest.approx(0.3)
    counts = {g: dict(accepted=100, errors=0, denominator=100) for g in m.base.GROUPS}
    counts["chain"].update(accepted=10, errors=1)
    assert m.base.mixture(counts, [0.25] * 4)["risk"] == pytest.approx(1 / 310)
    stress = m.evaluate(cohort, scores, truth, cert, corrupted=True)
    assert len(stress["rows"]) == 28 and not any(x["validity_guarantee"] for x in stress["rows"])


def test_simultaneous_pairing_parity_and_terminal_null():
    """REQ-REPORT-8870: repeats never increase n; identical arms establish parity."""
    cohort, scores, truth, _, _, cert = fixture()
    result = m.evaluate(cohort, scores, truth, cert)
    summary = m.summarize(result["rows"], result["group_count_rows"])
    assert len(summary["paired_replicate_rows"]) == 40
    assert len(summary["simultaneous_intervals"]) == 36
    assert summary["conditional_utility_score"] == 1
    assert summary["robustness_advantage_score"] == 0 and summary["comparison"] == "parity"
    assert all(x["independent_count"] == 40 for x in summary["simultaneous_intervals"])
    assert summary["calibration_family_violation_count"] == 0
    for c in cert["choices"]:
        c["served"] = {g: False for g in m.base.GROUPS}
    cert["certificate_hash"] = canonical_hash(cert["choices"])
    empty = m.evaluate(cohort, scores, truth, cert)
    null = m.summarize(empty["rows"], empty["group_count_rows"])
    assert (
        null["conditional_utility_score"] == 0
        and null["comparison"] == "terminal_null_low_coverage"
    )
    assert all(x["metric"] is None for x in null["census_risk_rows"])


def test_preconditions_authenticate_original_bytes(tmp_path):
    """REQ-REPORT-8870: absence blocks and original producer readiness is mandatory."""
    pre = r.preconditions(tmp_path / "real")
    assert pre["external_ready"] and pre["resources"]["passed"]
    bad = r.preconditions(tmp_path / "bad", tmp_path / "absent")
    assert bad["diagnostics"][0]["check"] == "missing_file"
    assert r.validate.__doc__


def test_measure_order_and_reconstruction(tmp_path):
    """REQ-REPORT-8870: persisted selection and certificates precede later label access."""
    pre = r.preconditions(tmp_path / "raw")
    result = r.measure(tmp_path / "raw", pre)
    assert len(result["rows"]) == 3360
    assert result == r.reduction(tmp_path / "raw")
    access = json.loads((tmp_path / "raw/access_order.json").read_bytes())
    stages = [x["stage"] for x in access["events"]]
    assert stages.index("selection") < stages.index("certification_labels")
    assert stages.index("certificates") < stages.index("census_labels")
    path = tmp_path / "raw/measurement.json"
    changed = json.loads(path.read_bytes())
    changed["conditional_utility_score"] = 9
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="measurement_drift"):
        r.reduction(tmp_path / "raw")


def test_cli_blocked_and_rehashed_replay(tmp_path):
    """SCENARIO-REPORT-8870-CONTRACT: real private CLI rejects checksum and semantic drift."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    output = tmp_path / (m.NAME + ".json")

    def child(options):
        return subprocess.run(
            command + options, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90
        )

    blocked = child(["--private-output", str(output), "--sources", str(tmp_path / "missing")])
    assert blocked.returncode == 0, blocked.stdout + blocked.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["conditional_rows_ready_score"] == 0
    assert child(["--cold-replay", str(output)]).returncode == 0
    value["experiment_id"] = 0
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    bad = child(["--cold-replay", str(output)])
    assert bad.returncode == 1 and "producer_identity" in bad.stdout
    assert child(["--date", "20261107"]).returncode == 2
    assert child(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))]).returncode == 2


def test_measurement_mutations_without_touching_originals(tmp_path, monkeypatch):
    """REQ-REPORT-8870: owned replay rejects semantic changes even with refreshed hashes."""
    raw = tmp_path / "raw"
    raw.mkdir()
    cohort = m.base.freeze(set(), count=3, fit_count=1)
    units = [u for u in cohort["inputs"] if u["split"] == "population"]
    truth = m.base.old.old.p.labels(units)
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
    trace = m.base.draws(3)
    selection = m.base.select(cohort, scores, truth, trace)
    cert = m.base.certify(cohort, scores, truth, trace, selection)
    stages = dict(
        population=cohort,
        fit=dict(numerical_training_calls=5),
        draws=trace,
        scores=dict(rows=scores),
        selection_labels=dict(labels=truth),
        selection=selection,
        certification_labels=dict(labels=truth),
        certificates=cert,
        census_labels=dict(labels=truth),
        corruption=dict(
            scores=scores,
            labels=m.base.old.old.p.labels(
                [u for u in m.base.corrupt(cohort)["inputs"] if u["split"] == "population"]
            ),
        ),
    )
    origin = tmp_path / "source"
    origin.mkdir()
    for name, value in stages.items():
        atomic_json(origin / (name + ".json"), value)
    source = dict(
        population_manifest=r.reference(origin / "population.json"),
        fit_manifest=r.reference(origin / "fit.json"),
        raw_shard_hashes=[r.reference(p) for p in origin.glob("*.json")],
    )
    atomic_json(raw / (r.SOURCE + ".json"), source)
    monkeypatch.setattr(m.base, "draws", lambda size: deepcopy(trace))
    monkeypatch.setattr(m.base.old.old, "audit_fit", lambda *a: None)
    monkeypatch.setattr(m.base.old.old, "predict", lambda *a: deepcopy(scores))
    assert r.measure(raw, dict(external_ready=True))["numerical_training_calls"] == 0
    real_load = r.load
    mutations = [
        ("draws", lambda v: v.update(seed=0), "sealed_draw"),
        ("population", lambda v: v["inputs"][0].update(group="bad"), "group_assignment"),
        ("scores", lambda v: v["rows"][0].update(candidate="bad"), "prediction"),
        ("selection", lambda v: v.update(selection_hash="bad"), "selection_drift"),
        ("certificates", lambda v: v.update(certificate_hash="bad"), "certificate_drift"),
        (
            "census_labels",
            lambda v: v["labels"].update({next(iter(v["labels"])): "bad"}),
            "label_drift",
        ),
        ("corruption", lambda v: v.update(scores=[]), "corruption_drift"),
    ]
    for stage, mutate, match in mutations:

        def changed(ref):
            value = real_load(ref)
            if Path(ref["path"]) == origin / (stage + ".json"):
                mutate(value)
            return value

        with monkeypatch.context() as patch:
            patch.setattr(r, "load", changed)
            with pytest.raises(ValueError, match=match):
                r.measurement(raw)
    with pytest.raises(ValueError, match="external_not_ready"):
        r.measure(raw, dict(external_ready=False))
    monkeypatch.setattr(r, "DEADLINE", -1)
    with pytest.raises(ValueError, match="measurement_deadline"):
        r.measurement(raw)
    access = json.loads((raw / "access_order.json").read_bytes())
    access["events"].reverse()
    atomic_json(raw / "access_order.json", access)
    with pytest.raises(ValueError, match="access_order"):
        r.reduction(raw)


def test_main_validation_failures_and_publication(tmp_path, monkeypatch):
    """REQ-REPORT-8870: real owned failures disqualify while refusals remain valid data."""
    from types import SimpleNamespace

    valid = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(complete=True, percent=100, files=[]),
        raw_references=[],
    )
    adapter = SimpleNamespace(validate=lambda raw, private: deepcopy(valid))
    real_namespace = r.namespace
    monkeypatch.setattr(r, "namespace", lambda *a: adapter)
    assert r.validate(tmp_path, tmp_path) == valid and adapter.m is m
    monkeypatch.setattr(r, "namespace", real_namespace)
    monkeypatch.setattr(r, "validate", lambda *a: deepcopy(valid))
    monkeypatch.setattr(
        r,
        "preconditions",
        lambda raw, *a: (
            raw.mkdir(parents=True),
            dict(external_ready=True, resources=dict(passed=True), references=[], diagnostics=[]),
        )[1],
    )

    def measured(raw, pre):
        result = r.scheduled("fixture")
        result["rows"] = [
            dict(x, status="completed", metric=0, missing_reason=None) for x in result["rows"]
        ]
        atomic_json(raw / "measurement.json", result)
        return result

    monkeypatch.setattr(r, "measure", measured)

    def checked(name, command, logs, **kw):
        logs.mkdir(parents=True, exist_ok=True)
        result = dict(name=name, passed=True)
        for suffix in ["stdout", "stderr"]:
            path = logs / (name + "." + suffix)
            path.write_text("checked")
            result[suffix + "_path"] = str(path)
        return result

    monkeypatch.setattr(r, "stream", checked)
    captures = []
    monkeypatch.setattr(
        r, "publish_primary", lambda out, value, validator: captures.append(deepcopy(value))
    )
    health = tmp_path / "health.json"
    atomic_json(health, checked("health", [], tmp_path / "health"))
    monkeypatch.setattr(r, "HEALTH", health)
    command = ["--output", str(tmp_path / (m.NAME + ".json"))]
    assert r.main(command) == 0 and captures[-1]["conditional_rows_ready_score"] == 1
    value = deepcopy(captures[-1])
    path = tmp_path / "replay.json"

    def write(value):
        v = deepcopy(value)
        v.pop("reproducibility_checksum", None)
        v["reproducibility_checksum"] = canonical_hash(v)
        atomic_json(path, v)

    raw = Path(value["measurement_manifest"]["path"]).parent
    (raw / (r.SOURCE + ".json")).write_text("fixture")
    monkeypatch.setattr(r, "PIN", r.sha256_file(raw / (r.SOURCE + ".json")))
    monkeypatch.setattr(r, "reduction", lambda raw: dict(rows=value["rows"]))
    write(value)
    assert r.replay(path)["passed"]
    for field, bad, match in [
        ("protocol_hash", "bad", "protocol"),
        ("completed_count", 1, "count"),
        ("conditional_rows_ready_score", 2, "readiness"),
        ("current_work_receipt", {}, "current_receipt"),
    ]:
        write(dict(value, **{field: bad}))
        with pytest.raises(ValueError, match=match):
            r.replay(path)
    write(value)
    v = json.loads(path.read_bytes())
    v["run_date"] = "bad"
    atomic_json(path, v)
    with pytest.raises(ValueError, match="checksum"):
        r.replay(path)
    operand = tmp_path / "operand"
    operand.write_text("before")
    write(dict(value, source_artifact_hashes=[r.reference(operand)]))
    operand.write_text("after")
    with pytest.raises(ValueError, match="operand_hash"):
        r.replay(path)
    write(value)
    monkeypatch.setattr(r, "PIN", "bad")
    with pytest.raises(ValueError, match="original_producer"):
        r.replay(path)
    monkeypatch.setattr(r, "PIN", r.sha256_file(raw / (r.SOURCE + ".json")))
    monkeypatch.setattr(r, "reduction", lambda raw: dict(rows=[]))
    with pytest.raises(ValueError, match="reduction_drift"):
        r.replay(path)
    empty = dict(
        value,
        rows=r.scheduled("missing")["rows"],
        completed_count=0,
        censored_count=3360,
        independent_count=0,
        conditional_utility_score=1,
        conditional_rows_ready_score=0,
    )
    write(empty)
    with pytest.raises(ValueError, match="unmeasured_benefit"):
        r.replay(path)
    monkeypatch.setattr(r, "stream", lambda *a, **kw: dict(checked(*a, **kw), passed=False))
    assert r.main(command) == 0 and captures[-1]["verdict_class"] == "disqualified"

    def failed(raw, pre):
        raise ValueError("owned_control")

    monkeypatch.setattr(r, "measure", failed)
    assert r.main(command) == 0 and captures[-1]["failed_count"] == 3360
    monkeypatch.setattr(r, "replay", lambda p: dict(passed=True))
    assert r.main(["--cold-replay", str(path)]) == 0


def test_pooled_acceptance_can_hide_weak_group_and_invalid_scores():
    """REQ-REPORT-8870: the complete census exposes risk hidden by balanced prevalence."""
    cohort, scores, truth, _, _, cert = fixture(size=20, replicates=2)
    truth["chain/0000"] = truth["chain/0001"] = "refute"
    result = m.evaluate(cohort, scores, truth, cert)
    balanced = [x for x in result["rows"] if x["scenario"] == "balanced"]
    shifted = [x for x in result["rows"] if x["scenario"] == "heavy_chain"]
    assert all(x["served_risk"] == pytest.approx(0.025) for x in balanced)
    assert all(x["served_risk"] == pytest.approx(0.07) for x in shifted)
    summary = m.summarize(result["rows"], result["group_count_rows"])
    assert summary["worst_served_group_risk"] == pytest.approx(0.10)
    assert summary["calibration_family_violation_count"] == 2
    scores[0]["probabilities"] = [float("nan"), 0, 1]
    with pytest.raises(ValueError, match="invalid_or_failed_score"):
        m.evaluate(cohort, scores, truth, cert)
