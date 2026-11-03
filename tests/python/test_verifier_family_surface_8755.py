"""REQ-VERIFY-8755 / REQ-REPORT-8755: frozen transfer cannot borrow surface tuning."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

from carnot.reporting import verifier_family_surface_8755 as m
from carnot.reporting import verifier_family_analysis_8755 as analysis
from carnot.reporting import verifier_family_runner_8755 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


def test_signed_variants():
    """SCENARIO-VERIFY-8755-SURFACE: shared bindings and absent evidence differ."""
    facts = [["a", "r", "b", True], ["c", "s", "d", True], ["a", "r", "d", False]]
    query = [["a", "r", "?x"], ["?x", "s", "d"]]
    assert m.verifier("correct", facts, query) == "unknown"
    assert m.verifier("correct", facts, [["a", "r", "d"]]) == "false"
    assert m.verifier("omitted_shared_variable_join", facts, query) == "true"
    assert m.verifier("missing_required_premise_as_false", facts, query) == "false"
    assert m.verifier("missing_required_premise_as_false", facts, None) == "unknown"
    assert m.verifier("reversed_arguments", facts, [["b", "r", "a"]]) == "true"
    assert m.verifier("reject_all", facts, query) == "false"
    assert m.verifier("accept_all", facts, query) == "true"
    with pytest.raises(ValueError, match="variant"):
        m.verifier("forged", facts, query)


def test_full_measurement_and_reduction(tmp_path):
    """SCENARIO-VERIFY-8755-TRANSFER: every call and paired world is retained."""
    inputs = m.inputs(m.UPSTREAM)
    assert inputs["passed"]
    measured = m.measure(inputs["workloads"], tmp_path)
    rows = measured["rows"]
    assert len(rows) == 1440
    assert len({p["probe_id"] for r in rows for p in r["probes"]}) == 7200
    assert len(measured["call_cost_rows"]) == 14400
    assert all(len(r["probes"]) == 5 for r in rows)
    assert m.audit(inputs["workloads"], measured)
    summary = analysis.summarize(rows)
    assert summary["paired_world_intervals"]["new"]["draws"] == 10000
    assert summary["paired_world_intervals"]["new"]["independent_clusters"] == 30
    assert all(r["denominator"] == 30 for r in summary["family_detection_rows"])
    assert len(summary["surface_pair_rows"]) == 720
    assert summary["scientific_promotion_gates"]["truth_invariance"]
    assert summary["scientific_promotion_gates"]["correct_control_invariance"]
    assert summary["scientific_promotion_gates"]["reject_all_fails"]
    forged = deepcopy(measured)
    forged["rows"][0]["probes"][0]["actual"] = "false"
    assert not m.audit(inputs["workloads"], forged)
    for field, change in [
        ("count", lambda x: x["rows"].pop()),
        ("identity", lambda x: x["rows"][0].update(world_id="forged")),
        ("probes", lambda x: x["rows"][0]["probes"].pop()),
        ("timing", lambda x: x["rows"][0]["probes"][0].update(elapsed_s=-1)),
        ("aggregate", lambda x: x["rows"][0].update(raw_numerator=-1)),
        ("cost", lambda x: x["call_cost_rows"][0].update(calls=4)),
    ]:
        forged = deepcopy(measured)
        change(forged)
        assert not m.audit(inputs["workloads"], forged), field
    assert analysis.summarize([])["paired_world_intervals"]["new"]["draws"] == 0


def test_upstream_gates(tmp_path):
    """REQ-REPORT-8755: missing file, changed hash, zero and rejection stay distinct."""
    assert not m.inputs(tmp_path / "absent")["passed"]
    upstream = json.loads(m.UPSTREAM.read_bytes())
    path = tmp_path / "upstream.json"
    for field, value in [
        ("verifier_transfer_methods_ready_score", 0),
        ("verdict_class", "disqualified"),
    ]:
        changed = dict(upstream, **{field: value})
        atomic_json(path, changed)
        assert not m.inputs(path, m.reference(path)["sha256"])["passed"]
    upstream.pop("verifier_transfer_methods_ready_score")
    atomic_json(path, upstream)
    assert not m.inputs(path, m.reference(path)["sha256"])["passed"]
    assert not m.inputs(path)["passed"]
    upstream = json.loads(m.UPSTREAM.read_bytes())
    canonical = m.load(upstream["canonical_workload"]["reference"])
    canonical["rows"].pop()
    altered = tmp_path / "canonical.json"
    atomic_json(altered, canonical)
    upstream["canonical_workload"]["reference"] = m.reference(altered)
    atomic_json(path, upstream)
    assert not m.inputs(path, m.reference(path)["sha256"])["passed"]
    upstream = json.loads(m.UPSTREAM.read_bytes())
    upstream["code_config_hashes"][0]["sha256"] = "sha256:" + "0" * 64
    atomic_json(path, upstream)
    assert not m.inputs(path, m.reference(path)["sha256"])["passed"]


def test_private_cli_replay_and_attacks(tmp_path):
    """SCENARIO-REPORT-8755-CLI: real children reject rehashed rows and aggregates."""
    output = tmp_path / (m.NAME + ".json")
    argv = [sys.executable, str(m.ROOT / m.CLI)]
    if os.environ.get("COVERAGE_RCFILE"):
        argv = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + os.environ["COVERAGE_RCFILE"],
            str(m.ROOT / m.CLI),
        ]
    logs = tmp_path / "logs"
    assert child(
        "private", [*argv, "--private-control", "--output", str(output)], logs, deadline=180
    )["passed"]
    assert runner.replay(output)
    assert child("replay", [*argv, "--cold-replay", str(output)], logs, deadline=120)["passed"]
    original = output.read_bytes()
    value = json.loads(original)
    value["verifier_transfer_rows_ready_score"] = 1
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert not runner.replay(output)
    assert child("tamper", [*argv, "--cold-replay", str(output)], logs, expected=1, deadline=120)[
        "passed"
    ]
    output.write_bytes(original)
    seal = Path(value["primitive_reference"]["path"]).parent / "independent_seal.json"
    saved = seal.read_bytes()
    atomic_json(seal, {})
    assert not runner.replay(output)
    seal.write_bytes(saved)
    assert not runner.replay(tmp_path / "missing")
    assert child("unsafe", [*argv, "--private-control"], logs, expected=2, deadline=60)["passed"]
    assert child("date", [*argv, "--date", "20260101"], logs, expected=2, deadline=60)["passed"]


def test_blocked_cli(tmp_path):
    """REQ-REPORT-8755: absent external methods produce terminal blocked bytes."""
    output = tmp_path / (m.NAME + ".json")
    assert (
        runner.main(
            ["--private-control", "--upstream", str(tmp_path / "absent"), "--output", str(output)]
        )
        == 0
    )
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"
    assert runner.replay(output)


def test_production_dispatch_and_rehashed_rows(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8755-CLI: owned failures suppress readiness; row edits fail replay."""
    from datetime import datetime, UTC
    from types import SimpleNamespace
    from carnot.reporting import verifier_family_report_8755 as report

    def validation(*args):
        return dict(
            receipts=[dict(passed=True)],
            union=dict(passed=True),
            coverage=dict(complete=True),
            ended_at_utc=datetime.now(UTC).isoformat(),
        )

    monkeypatch.setattr(runner, "Coverage", lambda **kw: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(runner, "validate", validation)
    output = tmp_path / (m.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verifier_transfer_rows_ready_score"] == 1
    assert value["verdict_class"] in {"null", "circular_positive"}
    assert runner.replay(output)
    work_path = Path(value["primitive_reference"]["path"])
    work = m.load(value["primitive_reference"])
    checked = m.load(value["validation_reference"])
    checked["receipts"][0]["passed"] = False
    assert report.build(work, checked, output)["verdict_class"] == "disqualified"
    analysis_path = Path(work["analysis_reference"]["path"])
    original_analysis = analysis_path.read_bytes()
    atomic_json(analysis_path, {})
    forged = dict(work, analysis_reference=m.reference(analysis_path))
    with pytest.raises(ValueError, match="sealed_analysis_drift"):
        report.build(forged, checked, output)
    analysis_path.write_bytes(original_analysis)
    measured_path = Path(work["measurement_reference"]["path"])
    measured = m.load(work["measurement_reference"])
    measured["rows"][0]["probes"][0]["actual"] = "false"
    atomic_json(measured_path, measured)
    work["measurement_reference"] = m.reference(measured_path)
    atomic_json(work_path, work)
    value["primitive_reference"] = m.reference(work_path)
    atomic_json(output, value)
    seal = work_path.parent / "independent_seal.json"
    atomic_json(seal, {k: value[k] for k in ["primitive_reference", "validation_reference"]})
    assert not runner.replay(output)
    source = tmp_path / "private-source"
    source.write_text("original")
    work["sources"].append(m.reference(source))
    atomic_json(work_path, work)
    value["primitive_reference"] = m.reference(work_path)
    atomic_json(output, value)
    atomic_json(seal, {k: value[k] for k in ["primitive_reference", "validation_reference"]})
    source.write_text("changed")
    assert not runner.replay(output)


def test_retired_precondition(tmp_path, monkeypatch):
    """REQ-REPORT-8755: a retired task is an external blocker, never a substitute study."""
    monkeypatch.setattr(
        runner.yaml, "safe_load", lambda _: dict(retired=[dict(experiment_id=8755)])
    )
    private = tmp_path / "private"
    private.mkdir()
    work = runner.prepare(tmp_path / "raw", private, m.UPSTREAM, True)
    assert not work["input_ready"]
    assert work["gate_check_summary"][-1]["check"] == "task_not_retired"
