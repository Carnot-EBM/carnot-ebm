"""REQ-REPORT-8803 / REQ-VERIFY-8803: independent seals and original repair custody."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import repaired_evidence_8803 as m
from carnot.reporting import repaired_evidence_runner_8803 as runner
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


def test_disjoint_sealed_worlds_and_balanced_views():
    """SCENARIO-REPORT-8803-COHORTS: exposed graph classes cannot become new worlds."""
    plan = m.freeze()
    manifest = plan["split_manifest"]
    assert manifest["counts"] == dict(
        intervention=120, consensus_dev=60, consensus_test=120, canary=12
    )
    assert manifest["disjoint"] and manifest["collision_rejections"] == 120
    assert len({w["structure"] for w in manifest["worlds"]}) == 312
    assert {w["edge_count"] for w in manifest["worlds"]} == {11, 12, 13}
    assert not set(manifest["historical_structures"]) & {w["structure"] for w in manifest["worlds"]}
    assert len(plan["intervention"]) == 5760 and len(plan["consensus"]) == 6480
    assert all(r["passed"] for r in plan["length_balance_rows"])
    assert {r["condition"] for r in plan["intervention"]} >= set(m.CONDITIONS)
    assert {r["group"] for r in plan["consensus"]} == set(m.GROUPS)
    for row in plan["intervention"] + plan["consensus"]:
        assert set(row["input"]) == {"facts", "query"}
        assert m.c.tokens(row["input"]) == row["token_bin"]
        assert row["expected"] in m.LABELS
    for group in m.GROUPS:
        rows = [r for r in plan["consensus"] if r["group"] == group]
        assert {r["sufficient"] for r in rows} == {True, False}
        assert all(len(r["candidates"]) == 3 for r in rows)


def test_budget_and_independent_truth_controls():
    """SCENARIO-REPORT-8803-BUDGET: absence, conflict and valid alternatives retain semantics."""
    fixtures = m.controls()
    rows = m.measure([r["input"] for r in fixtures])
    assert len(rows) == len(fixtures) * 4
    for i, fixture in enumerate(fixtures):
        assert m.oracle.enumerate_truth(fixture["input"])["decision"] == fixture["expected"]
        assert rows[i * 4 + 2]["decision"] == rows[i * 4 + 3]["decision"] == fixture["expected"]
        for row in rows[i * 4 : i * 4 + 3]:
            assert 0 < row["primitive_operations"] <= m.BUDGET
            assert row["primitive_limit"] == m.BUDGET
    exhausted = m.bounded("joint_symbolic", fixtures[0]["input"], limit=1)
    assert exhausted["budget_exhausted"] and exhausted["decision"] == "insufficient"
    with pytest.raises(ValueError, match="arm"):
        m.bounded("invalid", fixtures[0]["input"])
    with pytest.raises(ValueError, match="input_fields"):
        m.measure([dict(fixtures[0]["input"], group="leaked")])
    meter = m.PrimitiveCounter(1)
    meter(m.bounded.__code__, 0)
    assert meter.count == 0
    meter(m.old.strong.__code__, 0)
    assert meter.count == 1
    sys.monitoring.use_tool_id(4, "direct_meter_control")
    try:
        with pytest.raises(m.BudgetExceeded):
            meter(m.old.strong.__code__, 0)
    finally:
        sys.monitoring.free_tool_id(4)


def test_json_repair_controls_and_preconditions(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8803-ORIGINAL: types and exact historical authority remain mandatory."""
    receipts = m.repair_controls(tmp_path)
    assert all(r["passed"] for r in receipts)
    assert {r["control"] for r in receipts} >= {"null", "scalar", "pairs", "empty", "byte_drift"}
    pre = runner.preflight(tmp_path)
    assert pre["passed"] and pre["resource_projection"]["reserved_validation_seconds"] == 900
    assert pre["historical_authority"]["milestone"] == "2026.11.753"
    assert pre["current_authority"]["ready"]
    monkeypatch.setattr(runner.shutil, "which", lambda _: None)
    assert not runner.preflight(tmp_path)["passed"]


def test_private_cli_replay_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8803-TERMINAL: private children preserve negative verdict and cold replay."""
    output = tmp_path / (m.NAME + ".json")
    cmd = [sys.executable, str(m.ROOT / m.CLI)]
    for name, args, expected in [
        ("private", ["--private-control", "--output", str(output)], 0),
        ("cold", ["--cold-replay", str(output)], 0),
        ("missing", ["--cold-replay", str(tmp_path / "absent")], 1),
        ("unsafe", ["--private-control"], 2),
        ("date", ["--date", "20260101"], 2),
    ]:
        receipt = child(name, cmd + args, tmp_path / "logs", expected=expected, deadline=120)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert (
        value["evidence_interventions_ready_score"] == value["consensus_protocol_ready_score"] == 0
    )
    assert runner.replay(output)
    value["split_manifest"]["counts"]["intervention"] = 121
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert not runner.replay(output)


def test_build_failure_gates_and_production_dispatch(tmp_path, monkeypatch):
    """REQ-VERIFY-8803: missing external resources block; owned failures disqualify."""
    output = tmp_path / (m.NAME + ".json")
    monkeypatch.setattr(runner, "validate", lambda raw, private: runner.unqualified())
    assert runner.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified" and runner.replay(output)
    work = m.load(value["primitive_reference"])
    for target, name, replacement in [
        (runner, "reference", lambda path: dict(m.reference(path), sha256="changed")),
        (m, "HISTORY_PIN", "changed"),
        (m, "freeze", lambda: {}),
        (m, "measure", lambda _: []),
        (runner, "consensus_ok", lambda _: False),
    ]:
        with monkeypatch.context() as patch:
            patch.setattr(target, name, replacement)
            assert not runner.replay(output)
    altered = deepcopy(value)
    altered["primitive_reference"]["sha256"] = "changed"
    atomic_json(tmp_path / "altered.json", altered)
    assert not runner.replay(tmp_path / "altered.json")
    valid = runner.unqualified()
    valid.update(
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(complete=True),
        original_coverage=dict(complete=True, statements=377),
    )
    qualified = runner.build(work, valid)
    assert qualified["verdict_class"] == "circular_positive"
    assert (
        qualified["evidence_interventions_ready_score"]
        == qualified["consensus_protocol_ready_score"]
        == 1
    )
    changed = deepcopy(work)
    changed["consensus_controls_passed"] = False
    assert runner.build(changed, valid)["consensus_protocol_ready_score"] == 0
    monkeypatch.setattr(
        runner,
        "preflight",
        lambda raw: dict(
            passed=False,
            diagnostics=[
                dict(
                    check="missing_file",
                    upstream=m.TASK,
                    artifact_field="resources",
                    expected=True,
                    observed=None,
                    path=str(tmp_path / "absent"),
                    sha256=None,
                )
            ],
        ),
    )
    assert runner.main(["--output", str(output)]) == 0
    blocked = json.loads(output.read_bytes())
    assert blocked["verdict_class"] == "blocked" and blocked["completed_count"] == 0
    assert blocked["gate_check_summary"] and runner.replay(output)
    assert not runner.replay(tmp_path / "absent")


def test_input_boundary_and_failed_measurement(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8803-TERMINAL: invalid transports and missing scratch never qualify."""
    path, out = tmp_path / "input.json", tmp_path / "decisions.json"
    with pytest.raises(SystemExit):
        runner.main(["--decide-input", str(path)])
    atomic_json(path, dict(rows=[m.controls()[0]["input"]]))
    assert runner.main(["--decide-input", str(path), "--decisions", str(out)]) == 0
    atomic_json(path, dict(rows=[dict(m.controls()[0]["input"], group="hidden")]))
    with pytest.raises(ValueError, match="input_fields"):
        runner.main(["--decide-input", str(path), "--decisions", str(out)])

    def unavailable(private):
        raise OSError("private scratch unavailable")

    monkeypatch.setattr(runner, "preflight", unavailable)
    output = tmp_path / (m.NAME + ".json")
    assert runner.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"


def test_shared_validation_uses_real_coverage(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8803-ORIGINAL: actual SQLite inventories cannot be replaced by percentages."""
    from types import SimpleNamespace
    from coverage import Coverage

    database = tmp_path / "probe.sqlite"
    receipt = child(
        "coverage_probe",
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--data-file=" + str(database),
            "-m",
            "carnot.reporting.repaired_evidence_8803",
        ],
        tmp_path / "logs",
    )
    assert receipt["passed"]
    cov = Coverage(data_file=str(database), config_file=False)
    cov.combine(keep=True)
    cov.save()
    monkeypatch.setattr(
        runner,
        "namespace",
        lambda *args: SimpleNamespace(
            validate=lambda raw, private: dict(
                coverage=dict(combined_reference=m.reference(database))
            )
        ),
    )
    measured = runner.validate(tmp_path / "raw", tmp_path)
    assert not measured["coverage"]["complete"]
    assert not measured["original_coverage"]["complete"]
    assert measured["original_coverage"]["statements"] == 377


def test_consensus_control_negative_cases():
    """SCENARIO-REPORT-8803-COHORTS: agreement and sufficiency are separately falsifiable."""
    assert not runner.consensus_ok(dict(consensus=[]))
    row = dict(
        candidates=["support"] * 3, expected="support", group="unanimous_wrong", sufficient=True
    )
    assert not runner.consensus_ok(dict(consensus=[row] * 6480))
