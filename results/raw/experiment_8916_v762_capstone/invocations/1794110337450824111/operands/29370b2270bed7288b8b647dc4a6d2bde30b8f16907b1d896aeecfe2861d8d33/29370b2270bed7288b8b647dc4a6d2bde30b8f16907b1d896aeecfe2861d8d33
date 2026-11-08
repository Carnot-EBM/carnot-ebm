"""REQ-REPORT-8915: unknown compatible costs cannot become hardware gains."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hardware_operation_budget_8915 as m


def producer(branch, gate=1):
    """SCENARIO-REPORT-8915-CONTRACT: construct a valid null without claiming benefit."""
    _, _, field, identity = m.PRODUCERS[branch]
    return dict(
        experiment_id=identity,
        task_id=m.TASKS[branch],
        honest_verdict="complete_null",
        verdict_class="null",
        required_checks_passed=True,
        rows=[
            dict(
                unit_id="fixture",
                cluster_id="fixture",
                arm="fixture",
                seed=0,
                status="completed",
                denominator=1,
            )
        ],
        **{field: gate},
    )


def test_admission():
    """SCENARIO-REPORT-8915-CONTRACT: distinguish zero, missing and rejected gates."""
    for branch in m.PRODUCERS:
        assert m.admission(producer(branch), branch) is None
        assert m.admission(producer(branch, 0), branch) is None
        assert m.admission(producer(branch, None), branch) == "missing_field"
        for change in [
            {m.PRODUCERS[branch][2]: True},
            {m.PRODUCERS[branch][2]: 2},
            dict(experiment_id=-1),
            dict(task_id="wrong"),
            dict(required_checks_passed=False),
            dict(verdict_class="disqualified"),
            dict(verdict_class="positive", verifier_is_oracle=True),
            dict(rows=[]),
            dict(rows=[dict(status="unknown", denominator=1)]),
            dict(rows=[dict(status="completed", denominator=0)]),
        ]:
            assert m.admission(dict(producer(branch), **change), branch) == "rejected_producer"


def test_amdahl_and_operations():
    """REQ-REPORT-8915: only a measured compatible fraction admits the 100x kernel ceiling."""
    assert m.amdahl(None) is None
    assert m.amdahl(0) == 1
    assert m.amdahl(1) == 100
    assert m.amdahl(0.5) == pytest.approx(1 / 0.505)
    for bad in [-1, 2, float("nan")]:
        with pytest.raises(ValueError):
            m.amdahl(bad)
    value = m.reduce({})
    assert len(value["operation_rows"]) == 9
    assert len(value["compatibility_rows"]) == 45
    assert not value["measured_fraction_rows"]
    assert all(r["fraction"] is None for r in value["unknown_fraction_rows"])
    assert all(r["ceiling_100x"] is None for r in value["amdahl_rows"])
    assert not value["physical_acceleration_demonstrated"]
    assert value["nfr_01"]["whole_service_target_speedup"] == 10


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8915: read authentic originals without executing their science."""
    return m.collect(tmp_path_factory.mktemp("hardware8915"))


def test_actual_accounting(work):
    """SCENARIO-REPORT-8915-CONTRACT: valid cost nulls survive failed attribution and service."""
    validation = dict(passed=True, receipts=[], coverage=dict(complete=True))
    value = m.build(work, validation)
    assert value["verdict_class"] == "blocked"
    assert value["hardware_map_ready_score"] == 1
    assert value["required_checks_passed"] is True
    assert value["numerical_training_calls"] == 0
    assert value["independent_count"] == 0
    assert value["completed_count"] + value["blocked_count"] == value["intended_count"]
    assert {r["missing_reason"] for r in value["gate_check_summary"]} >= {
        "observed_zero",
        "rejected_producer",
    }
    assert len(value["whole_service_rows"]) == 3504
    assert all(
        not r["source_admitted"]
        for r in value["whole_service_rows"]
        if r["source_branch"] == "durable"
    )
    assert value["inclusive_timer_rows"]
    assert all(r["additive_partition"] is False for r in value["inclusive_timer_rows"])
    assert all(r["fraction"] is None for r in value["unknown_fraction_rows"])
    assert m.build(work, dict(validation, passed=False))["verdict_class"] == "disqualified"
    assert m.build(json.loads(json.dumps(work)), validation) == value
    qualified = producer("request")
    qualified["component_rows"] = [dict(components_ns=dict(lookup=12))]
    assert (
        m.reduce(dict(request=qualified))["qualified_component_rows"] == qualified["component_rows"]
    )


def test_cli_and_replay(tmp_path):
    """SCENARIO-REPORT-8915-CONTRACT: actual cold child rejects modified aggregates and bytes."""
    output = tmp_path / (m.NAME + ".json")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for args, expected in [
        (["--private-output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--replay", str(tmp_path / "missing")], 1),
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
    Path(ref["path"]).chmod(0o600)
    Path(ref["path"]).write_bytes(b"changed")
    assert m.main(["--replay", str(output)]) == 1


def test_production_route(work, tmp_path, monkeypatch):
    """REQ-REPORT-8915: failed owned controls cannot qualify the production route."""
    monkeypatch.setattr(m, "collect", lambda _: work)
    monkeypatch.setattr(
        m.checker,
        "validate",
        lambda _: dict(passed=False, receipts=[], coverage=dict(complete=False)),
    )
    monkeypatch.setattr(m.checker, "terminal", lambda p, r: dict(passed=True))
    monkeypatch.setattr(m, "DEFAULT_OUTPUT", tmp_path / (m.NAME + ".json"))
    assert m.main([]) == 0


def test_retained_bytes_and_external_failures(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8915-CONTRACT: immutable originals cannot be replaced by changed bytes."""
    source = tmp_path / "source"
    source.write_text("original")
    digest = m.sha256_file(source)
    monkeypatch.setattr(m, "PRODUCERS", {})

    def controlled(raw):
        first = m.engine.retain(source, raw, digest)
        assert m.engine.retain(source, raw, digest) == first
        with pytest.raises(ValueError, match="operand_hash_drift"):
            m.engine.retain(source, raw, "sha256:wrong")
        target = Path(first["path"])
        target.chmod(0o600)
        target.write_text("changed")
        with pytest.raises(ValueError, match="immutable_operand_drift"):
            m.engine.retain(source, raw)
        return dict(checks={}, evidence={}, refs=[])

    monkeypatch.setattr(m.engine, "collect", controlled)
    assert m.collect(tmp_path / "raw")["checks"]["physical_probes_performed"] == 0


def test_absent_and_malformed_producers(tmp_path, monkeypatch):
    """REQ-REPORT-8915: unavailable external inputs terminate as an auditable blocked map."""
    path = tmp_path / "producer.json"
    m.atomic_json(path, dict(experiment_id=8912, rows=[]))
    producers = dict(
        absent=(str(tmp_path / "missing"), "sha256:absent", "cost_attribution_ready_score", 8912),
        malformed=(str(path), m.sha256_file(path), "cost_attribution_ready_score", 8912),
    )
    monkeypatch.setattr(m, "PRODUCERS", producers)
    monkeypatch.setattr(m.engine, "PRODUCERS", producers)
    monkeypatch.setattr(m.engine, "HISTORY", {})
    monkeypatch.setattr(m.engine, "SOURCES", [])
    work = m.collect(tmp_path / "raw")
    assert {r["missing_reason"] for r in work["failures"]} == {"missing_file", "missing_field"}
    value = m.build(work, dict(passed=True, receipts=[], coverage=dict(complete=True)))
    assert value["verdict_class"] == "blocked"
    assert value["hardware_map_ready_score"] == 1


def test_diagnostic_operand_failure(tmp_path, monkeypatch):
    """REQ-REPORT-8915: rejected producers retain primary bytes despite missing operands."""
    path = tmp_path / "producer.json"
    value = producer("durable")
    value.update(
        required_checks_passed=False,
        source_artifact_hashes=[dict(path=str(tmp_path / "missing"), sha256="sha256:absent")],
    )
    m.atomic_json(path, value)
    producers = dict(durable=(str(path), m.sha256_file(path), "adaptation_cost_ready_score", 8914))
    monkeypatch.setattr(m, "PRODUCERS", producers)
    monkeypatch.setattr(m.engine, "PRODUCERS", producers)
    monkeypatch.setattr(m.engine, "HISTORY", {})
    monkeypatch.setattr(m.engine, "SOURCES", [])
    work = m.collect(tmp_path / "raw")
    assert any(r["check"] == "diagnostic_operand" for r in work["failures"])
