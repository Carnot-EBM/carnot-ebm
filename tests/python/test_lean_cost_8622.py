"""REQ-REPORT-8622 / REQ-VERIFY-8622: full clocks precede any speed claim."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot.reporting import lean_cost_8622 as e
from carnot.reporting.current_work_receipt import atomic_json


def test_original_admission_and_missing(tmp_path):
    """SCENARIO-REPORT-8622-COST: original registered operands earn admission."""
    bundle = e.admit(e.ROOT)
    assert len(bundle["requests"]) == 64
    assert bundle["target_model"]["parameter_count"] == 109
    assert bundle["protocol"]["paired_repeats"] == 10
    with pytest.raises((OSError, ValueError)):
        e.admit(tmp_path)
    with patch.dict(e.PINS, {next(iter(e.PINS)): "sha256:wrong"}):
        with pytest.raises(ValueError):
            e.admit(e.ROOT)


def test_fixed_roster_and_gates():
    """REQ-REPORT-8622: repetitions and lifecycle gains cannot replace NFR01."""
    slots = e.roster()
    assert len(slots) == 240
    assert len({json.dumps(s, sort_keys=True) for s in slots}) == 240
    assert [s["arm"] for s in slots[:8]] == e.ARMS + [
        "legacy_native",
        "legacy_python",
        "lean_native",
        "lean_python",
    ]
    assert e.base.benefit([1.1, 1.2], 1.0, True) == (1, 0)
    assert e.base.benefit([10, 11], 1.05, True) == (1, 1)
    assert e.base.benefit([11, 12], 1.06, True) == (0, 0)


def test_real_four_arm_sessions(tmp_path):
    """REQ-VERIFY-8622: actual children retain responses, RSS and disjoint spans."""
    from carnot.reporting.lean_cost_session_8622 import session

    bundle = e.admit(e.ROOT)
    for slot in e.roster()[:4]:
        result = session(bundle, slot, tmp_path / slot["arm"])
        assert result["exit_code"] == 0
        assert len(result["requests"]) == 64
        assert result["peak_rss_kib"] > 0
        assert result["requests"][0]["latency_s"] >= result["startup_s"]
        assert sum(s["end_ns"] - s["start_ns"] for s in result["operations"]) == (
            result["end_ns"] - result["start_ns"]
        )
        summary = e.summarize_session(result, bundle)
        assert summary["decision_mismatches"] == 0
        assert summary["completed_count"] == 64
        assert summary["censored_count"] == summary["failed_count"] == 0
        changed = deepcopy(result)
        changed["operations"][0]["end_ns"] += 1
        with pytest.raises(ValueError):
            e.summarize_session(changed, bundle)
        for kind in ["configuration", "frame", "response"]:
            changed = deepcopy(result)
            if kind == "configuration":
                config = tmp_path / "wrong_config.json"
                atomic_json(config, {})
                changed["config_reference"] = e.reference(config)
            elif kind == "frame":
                changed["observations"][0]["request"]["row"]["rows"] = []
            else:
                changed["requests"][0]["response"]["decision"] = "wrong"
            with pytest.raises(ValueError):
                e.summarize_session(changed, bundle)


def test_missing_measurement_and_replay(tmp_path):
    """SCENARIO-VERIFY-8622-REPLAY: rehashing a changed aggregate grants no authority."""
    work = e.measure(tmp_path, tmp_path / "raw", "20261017")
    assert work["gate_check_summary"][0]["observed"] == "missing"
    assert len(work["rows"]) == 240
    output = tmp_path / (e.NAME + ".json")
    value = e.artifact(work, [], output)
    assert value["verdict_class"] == "blocked"
    assert value["lean_cost_ready_score"] == 0
    atomic_json(output, value)
    e.replay(output)
    changed = deepcopy(value)
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(output, changed)
    with pytest.raises(ValueError, match="reduction"):
        e.replay(output)
    changed["reproducibility_checksum"] = "wrong"
    atomic_json(output, changed)
    with pytest.raises(ValueError, match="checksum"):
        e.replay(output)
    with pytest.raises(OSError):
        e.replay(tmp_path / "missing.json")
    failed = e.artifact(work, [{"name": "owned", "passed": False}], output)
    assert failed["verdict_class"] == "disqualified"


def test_measure_success_failure_and_budget(tmp_path):
    """SCENARIO-REPORT-8622-COST: forecast and failures preserve the intended roster."""
    from carnot.reporting import lean_cost_session_8622 as peer

    with (
        patch.object(e, "admit", return_value={"head": {}}),
        patch.object(e, "roster", return_value=e.roster()[:3]),
        patch.object(
            peer,
            "session",
            side_effect=[{"duration_s": 0.001}, ValueError("owned_failure"), {"duration_s": 0.002}],
        ),
    ):
        work = e.measure(e.ROOT, tmp_path / "measured", "20261017")
    assert [r["status"] for r in work["rows"]] == ["completed", "failed", "completed"]
    assert work["first_cell_forecast"]["observed_s"] == 0.001
    with (
        patch.object(e, "admit", return_value={"head": {}}),
        patch.object(e, "roster", return_value=e.roster()[:1]),
        patch.object(e.time, "monotonic_ns", side_effect=[1, 4000_000_000_000, 4000_000_000_001]),
    ):
        work = e.measure(e.ROOT, tmp_path / "budget", "20261017")
    assert work["rows"][0]["missing_reason"] == "measurement_deadline"
    with patch.object(e.subprocess, "check_output", return_value="tmpfs"):
        with pytest.raises(ValueError, match="resource"):
            e.preconditions(tmp_path)
    with patch.object(e, "read_bound_sidecar", return_value={"report": {"passed": False}}):
        with pytest.raises(ValueError, match="unqualified"):
            e.admit(e.ROOT)


def test_complete_reduction_and_qualification(tmp_path):
    """REQ-REPORT-8622: synthetic unit controls exercise all three separate contrasts."""
    bundle = e.admit(e.ROOT)
    work = e.measure(tmp_path, tmp_path / "raw", "20261017")
    primitive = {}
    rows = []
    for index, slot in enumerate(e.roster()):
        name = str(index)
        primitive[name] = dict(
            slot=slot,
            requests=[dict(latency_s=0.01)],
            operations=[dict(operation="startup", start_ns=0, end_ns=1000000)],
        )
        rows.append(
            dict(
                slot,
                unit_id=name,
                status="completed",
                primitive_reference=dict(path=name, sha256="fixture"),
                missing_reason=None,
            )
        )
    work.update(bundle=bundle, rows=rows, gate_check_summary=[])

    def summary(value, unused):
        return dict(
            value["slot"],
            total_session_wall_s=1.0 if value["slot"]["arm"].startswith("legacy") else 0.5,
            decision_mismatches=0,
        )

    with (
        patch.object(e, "load", side_effect=lambda ref: primitive[ref["path"]]),
        patch.object(e, "summarize_session", side_effect=summary),
    ):
        reduced = e.reduce(work)
    assert len(reduced["paired_session_rows"]) == 240
    assert all(r["paired_repeats"] == 10 for r in reduced["throughput_intervals"])
    with patch.object(e, "reduce", return_value=deepcopy(reduced)):
        value = e.artifact(
            work,
            [dict(name=name, passed=True) for name in e.REQUIRED],
            tmp_path / (e.NAME + ".json"),
        )
    assert value["lean_cost_ready_score"] == value["lean_service_benefit_score"] == 1
    assert value["nfr01_score"] == 0
    assert value["verdict_class"] == "circular_positive"
    with (
        patch.object(e, "load", return_value=dict(slot={})),
        pytest.raises(ValueError, match="slot"),
    ):
        e.reduce(work)
    work["rows"] = []
    with patch.object(e, "reduce", return_value=deepcopy(dict(reduced, paired_session_rows=[]))):
        value = e.artifact(work, [], tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "partial"


def test_registered_readiness_and_immutable_reduction(tmp_path):
    """SCENARIO-VERIFY-8622-REPLAY: repaired hashes cannot invent permission or request rows."""
    original = e.load

    def changed(ref):
        value = original(ref)
        if ref["path"].endswith("experiment_8615_v740_contract_methods.json"):
            value["methods_ready_score"] = 0
        return value

    with (
        patch.object(e, "load", side_effect=changed),
        pytest.raises(ValueError, match="registered"),
    ):
        e.admit(e.ROOT)
    work = e.measure(tmp_path, tmp_path / "raw", "20261017")
    output = tmp_path / (e.NAME + ".json")
    value = e.artifact(work, [], output)
    atomic_json(Path(value["per_request_rows_reference"]["path"]), dict(rows=["tampered"]))
    with pytest.raises(ValueError, match="per_request"):
        e.artifact(work, [], output)
    work["bundle"] = {"forged": True}
    work["input_root"] = str(e.ROOT)
    atomic_json(tmp_path / "raw/work.json", work)
    value["work_reference"] = e.reference(tmp_path / "raw/work.json")
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    with pytest.raises(ValueError, match="original_authority"):
        e.replay(output)


def test_raw_coverage_receipt_authentication(tmp_path):
    """REQ-VERIFY-8622: cold readers authenticate real Coverage.py data and report bytes."""
    import coverage

    source = tmp_path / "coverage_fixture.py"
    source.write_text("x = 1\n")
    data = tmp_path / ".coverage"
    report = tmp_path / "coverage.json"
    cov = coverage.Coverage(data_file=str(data), config_file=False)
    cov.start()
    exec(compile(source.read_text(), str(source), "exec"), {})
    cov.stop()
    cov.save()
    cov.json_report(outfile=str(report))
    work = e.measure(tmp_path, tmp_path / "raw", "20261017")
    output = tmp_path / (e.NAME + ".json")
    value = e.artifact(
        work,
        [
            dict(
                name="coverage_fixture",
                passed=True,
                data_reference=e.reference(data),
                report=e.reference(report),
            )
        ],
        output,
    )
    atomic_json(output, value)
    e.replay(output)
    data.write_bytes(b"changed data")
    with pytest.raises(ValueError, match="hash"):
        e.replay(output)
