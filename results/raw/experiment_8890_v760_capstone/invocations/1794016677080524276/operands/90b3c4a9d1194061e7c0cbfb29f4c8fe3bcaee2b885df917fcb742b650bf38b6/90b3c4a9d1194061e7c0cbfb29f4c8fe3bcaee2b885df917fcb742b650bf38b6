"""REQ-REPORT-8831 / REQ-VERIFY-8831: paid work must retain exact decisions."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import ordered_evidence_cost_8831 as m
from carnot.reporting import ordered_evidence_runner_8831 as r
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting import selective_methods_8829 as upstream


def cohort():
    """Small disjoint worlds exercise real loops without replacing solver calls."""
    return upstream.freeze(set(), dict(calibration=2, evaluation=2))


def test_paid_decisions_and_controls():
    """SCENARIO-REPORT-8831-PAID-ORDER: complete witnesses and missing hops differ."""
    c = cohort()
    rows = m.measure(c["inputs"])
    scored = m.score(rows, c["inputs"])
    assert len(rows) == 72
    for row in scored:
        assert row["primitive_operations"] == sum(row["operation_counts"].values())
        assert row["primitive_operations"] <= m.CAP
        assert row["wall_time_s"] >= 0 and row["cpu_time_s"] >= 0
        assert row["decision"] in ["support", "refute", "insufficient"]
        assert row["decision"] == row["truth"]
    for name in m.SCHEDULES:
        row = m.attempt(name, c["inputs"][0]["input"], 0)
        assert row["budget_exhausted"] and row["decision"] == "insufficient"
        assert row["stop_reason"] == "operation_budget_exhausted"
    with pytest.raises(ValueError, match="schedule"):
        m.attempt("bad", c["inputs"][0]["input"])
    controls = m.controls()
    assert all(x["passed"] for x in controls)
    assert any(x["condition"] == "valid_alternative" for x in controls)
    with pytest.raises(ValueError, match="attempt_roster"):
        m.score(rows[:-1], c["inputs"])
    with pytest.raises(ValueError, match="input_fields"):
        m.attempt("always_joint", dict(c["inputs"][0]["input"], label="support"))


def test_selection_and_paired_intervals():
    """SCENARIO-REPORT-8831-CONTRACT: calibration owns selection, worlds own intervals."""
    c = cohort()
    rows = m.score(m.measure(c["inputs"]), c["inputs"])
    calibration = [x for x in rows if x["split"] == "calibration"]
    selected = m.select(calibration)
    assert selected["schedule"] in m.SCHEDULES[:2]
    with pytest.raises(ValueError, match="calibration_only"):
        m.select(rows)
    result = m.reduce([x for x in rows if x["split"] == "evaluation"], selected)
    assert len(result["whole_world_intervals"]) == 4
    assert len(result["per_world_results"]) == 2
    assert result["schedule_benefit_score"] == 0
    assert not result["acceptance_gates"]["paid_operation_reduction"]["passed"]
    assert m.stable(dict(wall_time_s=1, rows=[dict(decision="support")])) == dict(
        rows=[dict(decision="support")]
    )


def test_authority_and_preconditions(tmp_path, monkeypatch):
    """REQ-VERIFY-8831: exact authority bytes are required even when flags look ready."""
    pre = r.preconditions(tmp_path / "raw")
    assert pre["external_ready"] and pre["resources"]["passed"]
    assert len(pre["references"]) > 3
    with monkeypatch.context() as patch:
        patch.setattr(m, "UPSTREAM", tmp_path / "missing")
        pre = r.preconditions(tmp_path / "absent")
        assert pre["diagnostics"][0]["check"] == "missing_file"
    value = json.loads(m.UPSTREAM.read_bytes())
    for mutation, check in [
        ({}, "missing_field"),
        ({"evidence_protocol_ready_score": 0}, "observed_zero"),
        (dict(value, task_id="wrong"), "rejected_producer"),
    ]:
        path = tmp_path / "bad.json"
        atomic_json(path, mutation)
        with monkeypatch.context() as patch:
            patch.setattr(m, "UPSTREAM", path)
            assert r.preconditions(tmp_path / check)["diagnostics"][0]["check"] == check


def test_private_cli_and_cold_replay(tmp_path):
    """SCENARIO-VERIFY-8831-REPLAY: real children run outside the checkout."""
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    child = subprocess.run(
        command
        + [
            "--date",
            "20261105",
            "--private-output",
            "--private-world-limit",
            "2",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["intended_count"] == 4320 and len(value["rows"]) == 36
    assert value["evidence_rows_ready_score"] == 0
    replay = subprocess.run(
        command + ["--cold-replay", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value["whole_world_intervals"]["paid_operation_reduction"]["estimate"] = 999
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    atomic_json(output, value)
    assert not r.replay(output)
    assert not r.replay(tmp_path / "absent")


def test_cli_boundaries(tmp_path):
    """REQ-VERIFY-8831: private output and invalid dates fail before measurement."""
    for argv in [["--date", "20250101"], ["--private-output"]]:
        with pytest.raises(SystemExit):
            r.main(argv)
    assert r.main(["--cold-replay", str(tmp_path / "absent")]) == 1


def test_short_shared_join_requires_complete_witness():
    """REQ-VERIFY-8831: local matching cannot accept incompatible shared variables."""
    payload = dict(
        facts=[["a", "r", "b", True], ["c", "s", "d", True]],
        query=[["?x", "r", "b"], ["?x", "s", "d"]],
    )
    assert all(m.attempt(arm, payload)["decision"] == "insufficient" for arm in m.SCHEDULES)


def test_empty_work_and_owned_outcomes(tmp_path, monkeypatch):
    """REQ-VERIFY-8831: external absence blocks and owned failures disqualify."""
    with monkeypatch.context() as patch:
        patch.setattr(m, "UPSTREAM", tmp_path / "absent")
        work = r.prepare(tmp_path / "raw")
    work.update(code_config_hashes=[], raw_shard_hashes=[])
    validation = r.unqualified()
    value = r.artifact(work, validation, "20261105", {}, {})
    assert value["verdict_class"] == "blocked"
    work["preconditions_checked"]["external_ready"] = True
    assert r.artifact(work, validation, "20261105", {}, {})["verdict_class"] == "disqualified"
    work["rows"] = [dict(status="completed")] * 4320
    work["per_world_results"] = [{}] * 240
    validation = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(complete=True),
        raw_references=[],
    )
    assert r.artifact(work, validation, "20261105", {}, {})["verdict_class"] == "null"
    work["schedule_benefit_score"] = 1
    assert r.artifact(work, validation, "20261105", {}, {})["verdict_class"] == "circular_positive"
    work["controls"] = [dict(passed=False)]
    assert r.artifact(work, validation, "20261105", {}, {})["verdict_class"] == "disqualified"


def test_validation_configuration(tmp_path, monkeypatch):
    """REQ-VERIFY-8831: owned validation keeps both required private consumer suites."""
    from types import SimpleNamespace

    configured = SimpleNamespace(validate=lambda raw, private: dict(raw=str(raw)))
    monkeypatch.setattr(r, "namespace", lambda *args: configured)
    assert r.validate(tmp_path / "raw", tmp_path)["raw"] == str(tmp_path / "raw")
    assert len(configured.original.CONSUMERS) == 2


def test_heartbeat_and_uncompiled_conflicts(monkeypatch):
    """REQ-VERIFY-8831: progress also runs inside loops and conflicting signs stay unknown."""
    c = cohort()
    assert not m.conflicts(c["inputs"][0]["input"])
    conflict = next(u["input"] for u in c["inputs"] if u["condition"] == "conflicting_fact")
    assert m.conflicts(conflict)
    ticks = iter(range(0, 100000, 31))
    with monkeypatch.context() as patch:
        patch.setattr(m.time, "monotonic", lambda: next(ticks))
        measured = m.measure(c["inputs"][:1])
        assert len(m.score(measured, c["inputs"][:1])) == 3


def test_internal_schedule_and_split_failures(tmp_path, monkeypatch):
    """REQ-VERIFY-8831: internal frozen menu or split drift cannot become external blocking."""
    original = r.preconditions
    for kind in ["protocol", "cohort"]:

        def changed(raw):
            pre = original(raw)
            path = raw / (kind + ".json")
            value = json.loads(path.read_bytes())
            if kind == "protocol":
                value["schedules"] = []
            else:
                value["worlds"].append(value["worlds"][0])
            atomic_json(path, value)
            return pre

        with monkeypatch.context() as patch:
            patch.setattr(r, "preconditions", changed)
            with pytest.raises(ValueError, match="drift"):
                r.prepare(tmp_path / kind, 1)
    with monkeypatch.context() as patch:
        patch.setattr(r, "sha256_file", lambda path: "changed")
        assert r.preconditions(tmp_path / "drift")["diagnostics"][0]["error"] == "operand_drift"
    for args in [
        ["--private-world-limit", "1"],
        [
            "--private-output",
            "--private-world-limit",
            "0",
            "--output",
            str(tmp_path / "private.json"),
        ],
    ]:
        with pytest.raises(SystemExit):
            r.main(args)


def test_main_internal_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8831: a failed owned measurement publishes a disqualified result."""
    original = r.prepare

    def fail(raw, limit=None, measurement=True):
        if measurement:
            raise ValueError("internal_fixture_failure")
        return original(raw, limit, False)

    monkeypatch.setattr(r, "prepare", fail)
    monkeypatch.setattr(r, "validate", lambda raw, private: r.unqualified())
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert value["gate_check_summary"][-1]["check"] == "internal_failure"


def test_replay_rejects_changed_operands_and_reductions(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8831-REPLAY: mutation routes execute real independent replay checks."""
    raw = tmp_path / "raw"
    original = r.prepare(raw, 1)
    original.update(code_config_hashes=[], raw_shard_hashes=[])
    output = tmp_path / (m.NAME + ".json")

    def write(work, validation=None):
        if validation is None:
            validation = r.unqualified()
        atomic_json(raw / "work.json", work)
        atomic_json(raw / "validation.json", validation)
        value = r.artifact(
            work,
            validation,
            "20261105",
            r.reference(raw / "work.json"),
            r.reference(raw / "validation.json"),
        )
        atomic_json(output, value)
        return value

    value = write(deepcopy(original))
    value["code_config_hashes"].append(dict(path=str(raw / "writable.json"), sha256="wrong"))
    atomic_json(output, value)
    assert not r.replay(output)
    for kind in ["receipt", "intervals", "selection", "rows", "controls", "reference"]:
        work = deepcopy(original)
        if kind == "receipt":
            work["current_work_receipt"]["event_count"] = 99
        elif kind == "intervals":
            work["whole_world_intervals"]["decisive_coverage"]["estimate"] = 99
        elif kind == "selection":
            work["selected_schedule"]["calibration_rows_sha256"] = "wrong"
        elif kind == "rows":
            work["rows"][0]["missing_reason"] = "invented"
        elif kind == "controls":
            work["controls"][0]["passed"] = False
        else:
            work["reference_costs"][0]["primitive_operations"] += 1
        write(work)
        assert not r.replay(output), kind
    write(
        deepcopy(original),
        dict(
            receipts=[dict(passed=True)],
            union=dict(passed=True),
            coverage=dict(complete=True),
            raw_references=[],
        ),
    )
    assert not r.replay(output)
    write(deepcopy(original))
    with monkeypatch.context() as patch:
        patch.setattr(m, "measure", lambda inputs: [])
        assert not r.replay(output)
    units = json.loads((raw / "measurement_inputs.json").read_bytes())
    units["rows"].pop()
    atomic_json(raw / "measurement_inputs.json", units)
    assert not r.replay(output)
