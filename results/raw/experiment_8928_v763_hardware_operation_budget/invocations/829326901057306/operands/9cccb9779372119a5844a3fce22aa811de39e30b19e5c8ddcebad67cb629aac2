"""REQ-REPORT-8928: counts and old clocks cannot manufacture acceleration."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hardware_operation_budget_8928 as m


def producer(branch, gate=1):
    """SCENARIO-REPORT-8928-CONTRACT: valid nulls qualify evidence, never benefit."""
    _, _, field, identity = m.PRODUCERS[branch]
    return dict(
        experiment_id=identity,
        task_id=m.TASKS[branch],
        milestone="2026.11.763",
        run_date="20261108",
        honest_verdict="complete_null",
        verdict_class="null",
        required_checks_passed=True,
        **{field: gate},
        rows=[
            dict(
                unit_id="fixture",
                cluster_id="fixture",
                arm="fixture",
                seed=1,
                denominator=1,
                metric=0,
                status="completed",
                missing_reason=None,
            )
        ],
    )


def test_admission_and_amdahl():
    """SCENARIO-REPORT-8928-CONTRACT: missing, zero and rejected inputs differ."""
    for branch in m.PRODUCERS:
        if branch.endswith("skip"):
            continue
        assert m.admission(producer(branch), branch) is None
        assert m.admission(producer(branch, 0), branch) is None
        assert m.admission(producer(branch, None), branch) == "missing_field"
        for change in [
            dict(required_checks_passed=False),
            dict(experiment_id=-1),
            dict(task_id="foreign"),
            dict(milestone="old"),
            dict(run_date="20261107"),
            dict(rows=[]),
            dict(verdict_class="positive", verifier_is_oracle=True),
            {m.PRODUCERS[branch][2]: True},
        ]:
            assert m.admission(dict(producer(branch), **change), branch) == "rejected_producer"
    assert m.amdahl(None) is None
    assert m.amdahl(0.5) == pytest.approx(1 / 0.505)
    assert m.amdahl(10 / 11) == pytest.approx(10)
    with pytest.raises(ValueError):
        m.amdahl(float("nan"))


def test_reduction_scopes():
    """REQ-REPORT-8928: current counts stay separate from historical elapsed time."""
    causal = producer("causal")
    causal.update(
        gradient_call_ledger=[
            dict(
                cluster_id="world", arm="interpolated_true", calls=2, update_count=2, guard_scores=7
            )
        ],
        optimizer_sequences=[dict(arm="interpolated_true")],
        prediction_rows=[dict(arm="interpolated_true", primitive_checks=3)],
        label_call_rows=[dict(arm="interpolated_true")],
        scoring_rows=[dict(arm="interpolated_true")],
    )
    durable = producer("durable")
    durable["service_cost_rows"] = [dict(durable["rows"][0], complete_ns=100)]
    durable["inclusive_timer_rows"] = [dict(timer_ns=90)]
    value = m.reduce(dict(causal=causal, durable=durable))
    counts = {r["operation"]: r["metric"] for r in value["operation_rows"]}
    assert counts["gradient"] == counts["fast_slow_interpolation"] == 2
    assert counts["guard_scoring"] == 7 and counts["persistence"] == 1
    assert counts["policy_mask"] is None and counts["label_oracle"] == 1
    assert value["whole_service_rows"][0]["cost_scope"] == "historical_guarded_only"
    assert value["inclusive_timer_rows"][0]["additive_partition"] is False
    assert all(r["fraction"] is None for r in value["unknown_fraction_rows"])
    assert all(r["ceiling_100x"] is None for r in value["amdahl_rows"])
    assert not value["physical_acceleration_demonstrated"]
    assert not value["nfr_01"]["proved"]
    request = producer("request")
    request["component_rows"] = [dict(component="gradient", elapsed_ns=2)]
    assert m.reduce(dict(request=request))["qualified_component_rows"]
    request["cost_attribution_ready_score"] = 0
    assert not m.reduce(dict(request=request))["qualified_component_rows"]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8928: authenticate actual evidence without rerunning science."""
    return m.collect(tmp_path_factory.mktemp("hardware8928"))


def test_actual_accounting(work):
    """SCENARIO-REPORT-8928-CONTRACT: blocked science can have a complete map."""
    validation = dict(passed=True, receipts=[], coverage=dict(complete=True))
    value = m.build(work, validation)
    assert value["hardware_map_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["required_checks_passed"] is True
    assert len(value["whole_service_rows"]) == 48
    assert value["numerical_training_calls"] == value["independent_count"] == 0
    assert value["completed_count"] == value["intended_count"]
    assert {r["missing_reason"] for r in value["gate_check_summary"]} >= {
        "missing_file",
        "observed_zero",
        "rejected_producer",
    }
    assert all(r["fraction"] is None for r in value["unknown_fraction_rows"])
    assert value["physical_evidence_state"]["GPU_0"]["actual_workload_receipt"]
    assert m.build(work, dict(validation, passed=False))["verdict_class"] == "disqualified"


def test_actual_cli_and_tampering(tmp_path):
    """SCENARIO-REPORT-8928-CONTRACT: cold children reject rehashed altered maps."""
    output = tmp_path / (m.NAME + ".json")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for args, expected in [
        (["--private-output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--replay", str(tmp_path / "absent")], 1),
        (["--date", "20261107"], 2),
    ]:
        child = subprocess.run(
            [sys.executable, "-u", str(m.ROOT / m.CLI), *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert child.returncode == expected, child.stdout + child.stderr
    original = json.loads(output.read_bytes())
    changed = deepcopy(original)
    changed["amdahl_rows"][0]["ceiling_100x"] = 100
    changed["reproducibility_checksum"] = m.canonical_hash(changed["rows"])
    m.atomic_json(output, changed)
    assert m.main(["--replay", str(output)]) == 1
    m.atomic_json(output, original)
    ref = original["source_artifact_hashes"][0]
    Path(ref["path"]).write_bytes(b"changed")
    assert m.main(["--replay", str(output)]) == 1


def test_skip_authentication(work):
    """SCENARIO-REPORT-8928-CONTRACT: skips must name the frozen rejecting producer."""
    for branch in ("request_skip", "label_skip"):
        value = json.loads(Path(work["diagnostic_evidence"][branch]["path"]).read_bytes())
        assert m.admission(value, branch) is None
        for change in [
            dict(failed_evidence_sha256="foreign"),
            dict(failed_observed=True),
            dict(experiment=-1),
            dict(failed_expected=0),
            dict(failed_field="foreign"),
        ]:
            assert m.admission(dict(value, **change), branch) == "rejected_producer"


def test_missing_operand_and_immutable_bytes(tmp_path, monkeypatch):
    """REQ-REPORT-8928: absent and changed operands cannot qualify external evidence."""
    branch = "durable"
    path = tmp_path / "producer.json"
    value = producer(branch)
    value["source_artifact_hashes"] = [
        dict(path=str(tmp_path / "missing"), sha256="sha256:missing")
    ]
    m.atomic_json(path, value)
    producers = {branch: (str(path), m.sha256_file(path), m.PRODUCERS[branch][2], 8926)}
    monkeypatch.setattr(m, "PRODUCERS", producers)
    monkeypatch.setattr(m.engine, "PRODUCERS", producers)
    monkeypatch.setattr(m.engine, "SOURCES", [])
    work = m.collect(tmp_path / "raw")
    assert not work["evidence"]
    assert any(r["check"] == "diagnostic_operand" for r in work["failures"])
    operand = tmp_path / "operand"
    operand.write_bytes(b"actual")
    value["source_artifact_hashes"] = [dict(path=str(operand), sha256="sha256:wrong")]
    m.atomic_json(path, value)
    producers[branch] = (str(path), m.sha256_file(path), m.PRODUCERS[branch][2], 8926)
    assert m.collect(tmp_path / "wrong")["failures"]
    value["source_artifact_hashes"] = []
    m.atomic_json(path, value)
    producers[branch] = (str(path), m.sha256_file(path), m.PRODUCERS[branch][2], 8926)
    raw = tmp_path / "immutable"
    target = raw / "operands" / m.sha256_file(path).split(":")[1]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="immutable_operand_drift"):
        m.collect(raw)


def test_production_route(work, tmp_path, monkeypatch):
    """REQ-REPORT-8928: a failed owned validation keeps production disqualified."""
    monkeypatch.setattr(m, "collect", lambda _: deepcopy(work))
    monkeypatch.setattr(
        m.checker,
        "validate",
        lambda _: dict(passed=False, receipts=[], coverage=dict(complete=False)),
    )
    monkeypatch.setattr(m.checker, "terminal", lambda p, r: dict(passed=True))
    monkeypatch.setattr(m, "DEFAULT_OUTPUT", tmp_path / (m.NAME + ".json"))
    assert m.main([]) == 0
    assert json.loads(m.DEFAULT_OUTPUT.read_bytes())["verdict_class"] == "disqualified"
