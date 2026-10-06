"""REQ-REPORT-8203, REQ-VERIFY-8203: private bounds cannot qualify a board."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import hardware_decision_8203 as h


def panel():
    """A small radial head has ties, in-domain rows and a separate outside row."""
    head = dict(
        arm="scalar",
        dimensions=1,
        weights=[0.0, 1.0],
        temperature=1.0,
        geometry=dict(mean=[0.0], scale=[1.0], centers=[[0.0]], width=1.0),
        quantiles=[dict(infinity=False, threshold=0.5), dict(infinity=False, threshold=0.5)],
    )
    rows = [
        dict(unit_id=str(i), source_cluster_id=str(i), x=[x], role="fit")
        for i, x in enumerate([-1.0, 0.0, 1.0, 2.0])
    ]
    return dict(
        heads=[head],
        samples=rows,
        training=rows[:3],
        boards=[],
        history={},
        checks=[],
        references=[],
        cited=[],
        trained_head_specs=[],
        branches={"selective": True, "learning": False, "service": False},
    )


def test_precision_and_domain():
    """SCENARIO-VERIFY-8203-1: every final decision matches its scalar reference."""
    value = h.reduce(panel())
    assert {r["precision"] for r in value["precision_rows"]} == {"float64", "float32", "fixed16"}
    assert all(
        r["final_membership_mismatch"] == 0 and r["final_typed_mismatch"] == 0
        for r in value["precision_rows"]
    )
    assert all(
        r["fallback"]
        for r in value["precision_rows"]
        if r["unit_id"] == "3" and r["precision"] != "float64"
    )
    assert value["current_device_execution_count"] == 0
    assert value["verdict_class"] == "blocked"
    data = panel()
    data["heads"][0]["weights"] = [0.0, 0.0]
    assert all(
        r["fallback"] for r in h.reduce(data)["precision_rows"] if r["precision"] != "float64"
    )
    data["heads"][0]["quantiles"] = [dict(infinity=True, threshold=None)] * 2
    assert all(r["reference_set"] == [0, 1] for r in h.reduce(data)["precision_rows"])


def test_cost_bounds_and_branch_independence():
    """SCENARIO-VERIFY-8203-2: mixed time is never a pure arithmetic estimate."""
    row = dict(
        unit_id="cost",
        source_cluster_id="source",
        condition="cold",
        arm="cold",
        status="completed",
        numerator=100,
        denominator=90,
        measured_arithmetic_fraction=None,
        scoring_envelope_fraction=0.1,
        retained_components=dict(generation_ns=50, decision_write_ack_ns=20),
    )
    costs, bounds = h.costs([row])
    assert bounds[0]["optimistic_ceiling"] == pytest.approx(100 / 90)
    assert bounds[0]["arithmetic_only_ceiling"] is None
    assert costs[0]["acquisition_ns"] == 50
    row["measured_arithmetic_fraction"] = 0.02
    assert h.costs([row])[1][0]["arithmetic_only_ceiling"] == pytest.approx(100 / 98)
    row["status"] = "excluded"
    assert h.costs([row]) == ([], [])
    data = panel()
    data["branches"] = dict(selective=True, learning=True, service=True)
    assert h.reduce(data)["verdict_class"] == "null"
    data["checks"] = [dict(check="tamper", passed=False)]
    assert h.reduce(data)["honest_verdict"] == "complete_blocked_tamper"


def test_private_cli_and_cold_tamper(tmp_path):
    """SCENARIO-REPORT-8203-2: direct transport and cold replay need no PYTHONPATH."""
    import os
    import subprocess
    from carnot.reporting import hardware_decision_execution_8203 as cli
    from carnot.reporting.current_work_receipt import atomic_json

    fixture = tmp_path / "input.json"
    fixture_data = panel()
    fixture_data["branches"] = dict(selective=True, learning=True, service=True)
    atomic_json(fixture, fixture_data)
    output = tmp_path / (cli.NAME + ".json")
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    prefix = [str(cli.ROOT / ".venv/bin/python")]
    if env.get("CARNOT_8203_COVERAGE_CONFIG"):
        prefix += ["-m", "coverage", "run", "--rcfile=" + env["CARNOT_8203_COVERAGE_CONFIG"]]
    script = str(cli.ROOT / cli.SCRIPT)
    for args in (
        ["--input", str(fixture), "--output", str(output)],
        ["--cold-replay", str(output)],
    ):
        process = subprocess.run(
            [*prefix, script, *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(output.read_bytes())
    assert result["MODEL_SPECS"] == [] and result["hardware_boundary_ready_score"] == 0
    assert cli.replay(output)["passed"]
    result["completed_count"] += 1
    atomic_json(output, result)
    with pytest.raises(ValueError, match="reduction_drift"):
        cli.replay(output)
    assert cli.main(["--cold-replay", str(output)]) == 1
    assert (
        cli.main(["--input", str(fixture), "--output", str(cli.ROOT / "results" / output.name)])
        == 1
    )
    assert cli.main(["--input", str(tmp_path / "missing"), "--output", str(output)]) == 1


def test_missing_external_inputs(tmp_path):
    """SCENARIO-REPORT-8203-1: all missing operands stay terminal and explicit."""
    from carnot.reporting import hardware_decision_inputs_8203 as inputs

    data = inputs.load(tmp_path, tmp_path / "raw")
    assert not any(data["branches"].values())
    value = h.reduce(data)
    assert value["verdict_class"] == "blocked"
    assert len(value["board_rows"]) == 5
    assert any(not c["passed"] and c["observed"] is False for c in value["gate_check_summary"])


def test_owned_execution_and_disqualification(tmp_path, monkeypatch):
    """REQ-REPORT-8203: owned checks and terminal failures cannot earn readiness."""
    from carnot.reporting import hardware_decision_execution_8203 as cli
    from carnot.reporting.current_work_receipt import atomic_json

    output = tmp_path / (cli.NAME + ".json")
    monkeypatch.setattr(cli.inputs, "load", lambda root, raw: panel())
    monkeypatch.setattr(cli, "execute", lambda *args, **kwargs: [])
    assert cli.commands(tmp_path)
    assert cli.main(["--output", str(output)]) == 0
    assert cli.main(["--output", str(output)]) == 0
    data = json.loads(output.read_bytes())
    for key, match in [
        ("config", "configuration_drift"),
        ("precision_rows", "precision_drift"),
        ("validation_receipts", "validation_receipt_drift"),
    ]:
        changed = deepcopy(data)
        if key == "config":
            changed[key] = {}
        elif key == "precision_rows":
            changed[key][0]["final_typed_mismatch"] = 1
        else:
            changed[key] = [dict(passed=False, normal_exit=True)]
        atomic_json(output, changed)
        with pytest.raises(ValueError, match=match):
            cli.replay(output)
    atomic_json(output, data)
    primitive = Path(data["replay_input_reference"]["path"])
    primitive.write_text("{}")
    with pytest.raises(ValueError):
        cli.replay(output)
    log = tmp_path / "failure.log"
    log.write_text("normal failure")
    from carnot.reporting.evidence_features_custody_7980 import reference

    ref = reference(log)
    bad = dict(passed=False, normal_exit=True, log_path=str(log), log_sha256=ref["sha256"])
    monkeypatch.setattr(cli, "execute", lambda *args, **kwargs: [bad])
    assert cli.main(["--output", str(output)]) == 1
    assert list((tmp_path / "raw").rglob("failed_terminal_candidate.json"))


def test_precision_mismatch_disqualifies(monkeypatch):
    """REQ-VERIFY-8203: a guarded parity failure disqualifies instead of retrying."""
    original = h.precision

    def wrong(*args):
        rows, domain = original(*args)
        rows[0]["final_typed_mismatch"] = 1
        return rows, domain

    monkeypatch.setattr(h, "precision", wrong)
    assert h.reduce(panel())["verdict_class"] == "disqualified"


def test_qualified_sources_and_bad_head(tmp_path, monkeypatch):
    """REQ-REPORT-8203: terminal-qualified branches are reduced without hiding boards."""
    from carnot.reporting import hardware_decision_inputs_8203 as inputs
    from carnot.reporting.current_work_receipt import atomic_json
    from carnot.reporting.evidence_features_custody_7980 import reference

    head = panel()["heads"][0]
    head["head_fit_source_ids"] = ["0", "1", "2"]
    heads = tmp_path / "heads.json"
    evidence = tmp_path / "evidence.json"
    atomic_json(heads, dict(heads=[head]))
    atomic_json(evidence, dict(materialized_rows=panel()["samples"]))
    boards = [
        dict(board=n, custody_valid=True, scope=n, last_authenticated_evidence_date="20261006")
        for n in ("KV260", "PolarFire", "GateMate")
    ]
    values = {eid: dict(task_id=f"exp{eid}-fixture") for eid, _, _ in inputs.BRANCHES.values()}
    values[8190].update(board_rows=boards, workload_rows=[])
    values[8195].update(
        frozen_heads_path=str(heads),
        frozen_heads_sha256=reference(heads)["sha256"],
        measurement_reference=reference(evidence),
        trained_head_specs=[],
    )
    values[8201].update(workload_rows=[])
    monkeypatch.setattr(inputs.reader, "authenticate", lambda path, eid, *args: (values[eid], True))
    monkeypatch.setattr(inputs, "reader_receipt", lambda *args, **kwargs: dict(passed=True))
    monkeypatch.setattr(
        inputs.old, "load", lambda *args: dict(boards=boards, references=[], checks=[])
    )
    data = inputs.load(tmp_path, tmp_path / "raw")
    assert data["branches"]["selective"] and not data["branches"]["learning"]
    assert len(data["heads"]) == 1 and len(data["training"]) == 3
    monkeypatch.setattr(inputs, "reader_receipt", lambda *args, **kwargs: dict(passed=False))
    assert not any(inputs.load(tmp_path, tmp_path / "bad-readers")["branches"].values())


def test_health_custody_and_independent_cost_replay(tmp_path, monkeypatch):
    """REQ-REPORT-8203: health logs are sealed and independent sums detect drift."""
    from carnot.reporting import hardware_decision_execution_8203 as cli
    from carnot.reporting.current_work_receipt import atomic_json
    from carnot.reporting.evidence_features_custody_7980 import reference

    data = panel()
    data["history"] = dict(
        workload_rows=[
            dict(
                unit_id="cost",
                source_cluster_id="s",
                status="completed",
                numerator=100,
                denominator=90,
                condition="cold",
                arm="cold",
                retained_components=dict(generation_ns=50),
                measured_arithmetic_fraction=None,
            )
        ]
    )
    monkeypatch.setattr(cli.inputs, "load", lambda *args: data)
    monkeypatch.setattr(cli, "execute", lambda *args, **kwargs: [])
    log = tmp_path / "health.log"
    log.write_text("historical health observation")
    ref = reference(log)
    health = tmp_path / "health.json"
    atomic_json(
        health, [dict(log_path=str(log), log_sha256=ref["sha256"], passed=False, normal_exit=True)]
    )
    output = tmp_path / (cli.NAME + ".json")
    assert cli.main(["--output", str(output), "--repository-health-receipt", str(health)]) == 0
    value = json.loads(output.read_bytes())
    assert value["required_checks_passed"] and not value["repository_health"][0]["passed"]
    assert cli.replay(output)["passed"]
    value["workload_rows"][0]["denominator"] += 1
    atomic_json(output, value)
    with pytest.raises(ValueError, match="independent_ceiling_drift"):
        cli.replay(output)


def test_missing_source_slots(tmp_path):
    """REQ-REPORT-8203: missing evidence stays in the original source denominator."""
    data = panel()
    data["excluded_samples"] = [
        dict(
            unit_id="missing",
            source_cluster_id="missing",
            x=None,
            exclusion_reason="missing_features",
        )
    ]
    value = h.reduce(data)
    assert value["excluded_count"] == 3
    assert value["intended_count"] == value["completed_count"] + value["excluded_count"]


def test_probability_headline_tamper(tmp_path, monkeypatch):
    """REQ-REPORT-8203: independently recomputed probability headlines reject tampering."""
    from carnot.reporting import hardware_decision_execution_8203 as cli
    from carnot.reporting.current_work_receipt import atomic_json

    output = tmp_path / (cli.NAME + ".json")
    monkeypatch.setattr(cli.inputs, "load", lambda *args: panel())
    monkeypatch.setattr(cli, "execute", lambda *args, **kwargs: [])
    assert cli.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    value["probability_error_summary"][1]["maximum_error"] += 1
    atomic_json(output, value)
    with pytest.raises(ValueError, match="headline_drift"):
        cli.replay(output)
