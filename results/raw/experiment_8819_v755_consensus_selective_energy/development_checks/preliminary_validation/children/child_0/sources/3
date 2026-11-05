"""REQ-REPORT-8819 / REQ-VERIFY-8819: selection cannot hide wrong decisions."""

from copy import deepcopy
import json
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pytest

from carnot.reporting import consensus_selective_energy_8819 as m
from carnot.reporting import consensus_selective_runner_8819 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v746_retained_evidence import reference


def examples():
    """Known action targets expose wrong rejection as well as wrong acceptance."""
    return [
        dict(
            unit_id=f"w/{i}",
            cluster_id="w",
            split="calibration",
            candidate="support",
            truth=t,
            probabilities=p,
            action=a,
            arm="learned_energy",
            seed=1,
            agreement="unanimous_wrong",
            condition="intact",
            candidate_index=i,
        )
        for i, (t, p, a) in enumerate(
            [
                ("support", [0.9, 0.05, 0.05], "accept"),
                ("refute", [0.1, 0.8, 0.1], "reject"),
                ("insufficient", [0.1, 0.1, 0.8], "escalate"),
            ]
        )
    ]


def test_metrics_and_calibration():
    """SCENARIO-REPORT-8819-SELECTIVE: no cutoff means undefined risk, never zero."""
    rows = examples()
    stats = m.metrics(rows)
    assert stats["risk"] == 0 and stats["correct_coverage"] == 2 / 3
    selected = m.select(rows, "learned_energy", 1, draws=40)
    assert selected["threshold"] == 0.5
    rows[0]["probabilities"] = [0.4, 0.3, 0.3]
    rows[1]["probabilities"] = [0.4, 0.3, 0.3]
    assert m.select(rows, "learned_energy", 1, draws=40)["threshold"] is None
    for x in rows:
        x["action"] = "escalate"
    assert m.metrics(rows)["risk"] is None
    assert m.metrics([])["risk"] is None
    rows[0]["split"] = "evaluation"
    with pytest.raises(ValueError, match="calibration_only"):
        m.select(rows, "learned_energy", 1)


def test_intervals_and_missing_comparator():
    """SCENARIO-REPORT-8819-SELECTIVE: five intervals resample whole worlds."""
    rows = examples()
    for x in rows:
        x["split"] = "evaluation"
    rows += [dict(x, arm="fixed_evidence", seed=0) for x in rows]
    selected = [dict(arm="fixed_evidence", seed=0, threshold=None)]
    result = m.summarize(rows, selected, draws=40)
    assert len(result["simultaneous_intervals"]) == 5
    assert not result["benefit_passed"]
    assert result["simultaneous_intervals"]["brier_gain"]["status"] == "not_estimable"
    assert result["independent_count"] == 1
    selected[0]["threshold"] = 0.5
    assert (
        m.summarize(rows, selected, draws=40)["simultaneous_intervals"]["brier_gain"]["estimate"]
        == 0
    )
    duplicated = rows + [dict(x, seed=2) for x in rows if x["arm"] == "learned_energy"]
    assert (
        m.summarize(duplicated, selected, draws=40)["simultaneous_intervals"]
        == m.summarize(rows, selected, draws=40)["simultaneous_intervals"]
    )


def test_source_admission(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8819-CUSTODY: missing, zero and rejected differ."""
    path = tmp_path / "producer.json"
    assert r.admit(path)["diagnostics"][0]["check"] == "missing_file"
    for value, expected in [
        ({}, "missing_field"),
        ({"consensus_protocol_ready_score": 0}, "observed_zero"),
        ([], "rejected_producer"),
        ({"consensus_protocol_ready_score": 1}, "rejected_producer"),
    ]:
        path.write_text(json.dumps(value))
        assert r.admit(path)["diagnostics"][0]["check"] == expected
    value = dict(
        consensus_protocol_ready_score=1,
        required_checks_passed=True,
        raw_shard_hashes=[],
        code_config_hashes=[],
        split_manifest={"worlds": []},
    )
    atomic_json(path, value)
    monkeypatch.setattr(r, "SOURCE_PIN", reference(path)["sha256"])
    assert r.admit(path)["passed"]
    value["raw_shard_hashes"] = [dict(path=str(tmp_path / "absent"), sha256="x")]
    atomic_json(path, value)
    monkeypatch.setattr(r, "SOURCE_PIN", reference(path)["sha256"])
    assert not r.admit(path)["passed"]


def test_fixed_representation_training_and_seal(tmp_path):
    """SCENARIO-VERIFY-8819-CUSTODY: labels release after complete input-only rows."""
    worlds = [
        dict(
            world_id=f"private/{i}", split=s, edges=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]
        )
        for i, s in enumerate(["fit"] * 3 + ["calibration"] * 3 + ["evaluation"] * 3)
    ]
    cohort = m.construct(worlds)
    fitted = m.fit(cohort, seeds=[1], steps=2)
    assert fitted["train_worlds"] == [f"private/{i}" for i in range(3)]
    assert fitted["numerical_training_calls"] == 9
    inputs = [x for x in cohort["inputs"] if x["split"] == "evaluation"]
    predicted, costs = m.predict(inputs, fitted["parameters"], fitted["selected_threshold_rows"])
    assert len(costs) == len(inputs)
    assert all(x["paid_atom_checks"] == 5 for x in costs)
    assert all(len(x["atom_outcomes"]) == 5 and x["feature_record_visits"] > 0 for x in costs)
    assert all("truth" not in x for x in predicted)
    seal = canonical_hash(dict(rows=predicted, checks=costs))
    rows, oracle = m.release(cohort, predicted, costs, seal)
    assert len(rows) == len(predicted) + len(inputs) * 3
    assert oracle["exact_solver_calls"] == len(inputs)
    assert all(x["metric"] >= 0 for x in rows)
    with pytest.raises(ValueError, match="prediction_seal"):
        m.release(cohort, predicted[:-1], costs, seal)
    contaminated = deepcopy(inputs)
    contaminated[0]["input"]["truth"] = "support"
    with pytest.raises(ValueError):
        m.predict(contaminated, fitted["parameters"], [])
    assert (
        m.predict(inputs, fitted["parameters"], fitted["selected_threshold_rows"])[0] == predicted
    )


def test_private_cli_and_replay(tmp_path):
    """SCENARIO-REPORT-8819-REPLAY: real child and cold interpreter reject drift."""
    output = tmp_path / (r.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    cli = str(r.ROOT / r.CLI)
    commands = [
        (["--date", "19000101"], 2),
        (["--private-output"], 2),
        (["--date", "20261105", "--private-output", "--output", str(output)], 0),
    ]
    for argv, expected in commands:
        completed = subprocess.run(
            [sys.executable, "-u", cli, *argv],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == expected, completed.stdout + completed.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified"
    assert value["consensus_rows_ready_score"] == 0
    assert r.replay(output)
    completed = subprocess.run(
        [sys.executable, "-u", cli, "--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    value["selective_benefit_score"] = 1
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert not r.replay(output)
    assert r.main(["--cold-replay", str(output)]) == 1
    assert not r.replay(tmp_path / "absent")


def test_build_failure_classes(tmp_path, monkeypatch):
    """REQ-VERIFY-8819: owned failure disqualifies and external absence blocks."""
    monkeypatch.setattr(r, "SOURCE", tmp_path / "missing")
    work = r.prepare(tmp_path / "raw")
    validation = r.unqualified()
    value = r.build(work, validation)
    assert value["verdict_class"] == "blocked"
    assert value["censored_count"] == 2160
    work["preconditions_checked"]["passed"] = True
    assert r.build(work, validation)["verdict_class"] == "disqualified"
    validation.update(union={"passed": True}, coverage={"complete": True}, receipts=[])
    work["fit"]["shortcut_controls"] = [{"passed": True}]
    assert r.build(work, validation)["verdict_class"] == "null"
    monkeypatch.setattr(
        m,
        "summarize",
        lambda *a, **k: {
            "benefit_passed": True,
            "acceptance_gates": {},
            "simultaneous_intervals": {},
        },
    )
    assert r.build(work, validation)["verdict_class"] == "positive"


def test_validator_and_terminal_wiring(tmp_path, monkeypatch):
    """REQ-VERIFY-8819: reuse scoped validation and run actual terminal commands."""
    from types import SimpleNamespace

    fake = SimpleNamespace(validate=lambda raw, private: {"actual": True})
    monkeypatch.setattr(r, "namespace", lambda *args: fake)
    assert r.validate(tmp_path, tmp_path) == {"actual": True}
    assert fake.e.OWNED == r.OWNED
    monkeypatch.setattr(r, "stream", lambda *a, **k: {"passed": True})
    monkeypatch.setattr(r, "replay", lambda path: True)
    assert r.terminal(tmp_path / "candidate", tmp_path)["passed"]
    monkeypatch.setattr(r, "replay", lambda path: False)
    assert not r.terminal(tmp_path / "candidate", tmp_path)["passed"]


def test_progress_and_empty_calibration():
    """SCENARIO-REPORT-8819-SELECTIVE: empty menus stay explicit and long rosters report progress."""
    assert m.select([], "learned_energy", 1, draws=2)["threshold"] is None
    row = examples()[0]
    assert m.distribution([row], 2)["insufficient_to_decision"][0] is None
    unit = dict(
        unit_id="u",
        world_id="w",
        split="calibration",
        condition="intact",
        agreement="disagreement",
        candidates=list(m.old.LABELS),
        input=dict(
            facts=[["a", f"r{i}", "b", True] for i in range(5)],
            query=[["a", f"r{i}", "b"] for i in range(5)],
        ),
    )
    predicted, checks = m.predict([unit] * 180, [], [])
    assert len(checks) == 180 and len(predicted) == 1080


def test_existing_operand_drift(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8819-CUSTODY: present byte drift differs from missing sidecars."""
    path = tmp_path / "producer.json"
    operand = tmp_path / "operand"
    operand.write_text("actual")
    value = dict(
        consensus_protocol_ready_score=1,
        required_checks_passed=True,
        raw_shard_hashes=[dict(path=str(operand), sha256="sha256:wrong")],
        code_config_hashes=[],
    )
    atomic_json(path, value)
    monkeypatch.setattr(r, "SOURCE_PIN", reference(path)["sha256"])
    assert r.admit(path)["diagnostics"][0]["error"] == "bound_operand_drift"


def test_production_main_wiring(tmp_path, monkeypatch):
    """REQ-VERIFY-8819: production must call scoped validation before publication."""
    monkeypatch.setattr(r, "SOURCE", tmp_path / "absent")
    work = r.prepare(tmp_path / "original")
    validation = dict(
        union={"passed": True}, coverage={"complete": True}, receipts=[], raw_references=[]
    )
    monkeypatch.setattr(r, "prepare", lambda *a: work)
    monkeypatch.setattr(r, "validate", lambda *a: validation)
    seen = []
    health = tmp_path / "health.json"
    atomic_json(health, dict(status="global_debt", raw_references=[]))
    monkeypatch.setattr(r, "publish_primary", lambda output, value, validator: seen.append(value))
    assert (
        r.main(["--output", str(tmp_path / (r.NAME + ".json")), "--repository-health", str(health)])
        == 0
    )
    assert seen[0]["verdict_class"] == "blocked"


def test_replay_rejects_repaired_primitives(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8819-CUSTODY: repaired outer hashes cannot excuse changed science."""
    work = r.prepare(tmp_path / "raw", private=True)
    raw = tmp_path / "raw"
    validation = r.unqualified()
    atomic_json(raw / "validation.json", validation)
    output = tmp_path / (r.NAME + ".json")

    def save():
        atomic_json(raw / "work.json", work)
        atomic_json(output, r.candidate_value(work, validation, raw))

    save()
    assert r.replay(output)
    original = deepcopy(work)
    for mutation in ["operand", "cohort", "selector", "seal", "rows"]:
        work = deepcopy(original)
        if mutation == "operand":
            work["raw_shard_hashes"][0]["sha256"] = "sha256:wrong"
        elif mutation == "cohort":
            cohort = json.loads(Path(work["cohort_reference"]["path"]).read_text())
            cohort["labels"][next(iter(cohort["labels"]))] = "altered"
            atomic_json(raw / "changed_cohort.json", cohort)
            work["cohort_reference"] = reference(raw / "changed_cohort.json")
        elif mutation == "selector":
            work["fit"]["selected_threshold_rows"][0]["threshold"] = 0.123
        elif mutation == "seal":
            work["prediction_seal"] = "altered"
        else:
            work["per_candidate_rows"][0]["metric"] = 999
        save()
        assert not r.replay(output), mutation
