"""REQ-REPORT-8902: separate branch accounting from device acceleration."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import hardware_operation_map_8902 as m


def test_admission():
    """SCENARIO-REPORT-8902-CONTRACT: valid nulls and zero attribution retain evidence."""
    for branch, (_, _, field, identity) in m.PRODUCERS.items():
        value = dict(
            experiment_id=identity,
            required_checks_passed=True,
            verdict_class="null",
            honest_verdict="complete_null",
            rows=[
                dict(status="complete" if branch == "adaptation" else "completed", denominator=1)
            ],
            **{field: 1},
        )
        assert m.admission(value, branch) is None
        for change, expected in [
            (None, "missing_field"),
            (0, None if branch == "attribution" else "observed_zero"),
            (2, "rejected_producer"),
        ]:
            assert m.admission(dict(value, **{field: change}), branch) == expected
        assert m.admission(dict(value, required_checks_passed=False), branch) == "rejected_producer"


def test_amdahl():
    """REQ-REPORT-8902: no fraction means no bound; a 100x target needs over 99% removable work."""
    assert m.amdahl(None)["infinite_component_speed_bound"] is None
    assert m.amdahl(0)["infinite_component_speed_bound"] == 1
    assert m.amdahl(0.5)["required_component_speed_for_100x"] is None
    assert m.amdahl(0.999)["required_component_speed_for_100x"] > 100
    assert m.amdahl(1)["required_component_speed_for_100x"] == 100
    for fraction in (-0.1, 1.1):
        with pytest.raises(ValueError):
            m.amdahl(fraction)


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-REPORT-8902: authenticate the actual independent producers once."""
    return m.collect(tmp_path_factory.mktemp("hardware8902") / "raw")


def test_reduce_and_build(work, monkeypatch):
    """SCENARIO-REPORT-8902-CONTRACT: preserve installed, label and adaptation operands."""
    assert "adaptation" in work["evidence"], work["failures"]
    assert "labels" in work["evidence"], work["failures"]
    invalid = dict(passed=False, receipts=[], coverage=dict(complete=False))
    assert m.build(work, invalid)["verdict_class"] == "disqualified"
    valid = dict(passed=True, receipts=[], coverage=dict(complete=True))
    value = m.build(work, valid)
    assert json.loads(json.dumps(value)) == value
    assert m.build(json.loads(json.dumps(work, sort_keys=True)), valid) == value
    assert value["hardware_map_ready_score"] == 1
    assert value["verdict_class"] == "blocked"
    assert value["sampling_call_count"] == 0
    assert any(r["missing_reason"] == "observed_zero" for r in value["gate_check_summary"])
    assert value["numerical_training_calls"] == 0
    assert value["upstream_numerical_training_calls"] == 6912
    assert value["label_call_savings"]["wall_time_saved_s"] is None
    assert value["label_call_savings"]["joules_saved"] is None
    assert all(r["qualified_fraction"] is None for r in value["compatible_fraction_rows"])
    assert all(r["infinite_component_speed_bound"] is None for r in value["amdahl_bounds"])
    assert len(value["board_operation_rows"]) == 27
    assert value["label_call_savings"]["saved_calls"] == 0
    assert sum(r["saved_calls"] for r in value["label_call_savings"]["standalone_rows"]) > 0
    assert all(r["status"] == "deferred" for r in value["acquisition_priority"])
    assert "local_binding_intersections" in value["operation_map"]
    assert value["censored_count"] == 3456
    assert value["execution_venue"] == "host"
    assert value["physical_evidence_state"]["GateMate"]["retry_allowed"] is False
    assert (
        value["completed_count"]
        + value["censored_count"]
        + value["failed_count"]
        + value["blocked_count"]
        == value["intended_count"]
    )
    assert value["independent_count"] == 0
    empty = deepcopy(work)
    empty["evidence"] = {}
    empty["diagnostic_evidence"] = {}
    empty["refs"] = []
    empty["failures"] = [dict(upstream="adaptation", missing_reason="missing_file")]
    blocked = m.build(empty, valid)
    assert blocked["hardware_map_ready_score"] == 1
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"][0]["status"] == "blocked"
    assert blocked["label_call_savings"]["saved_calls"] is None
    assert m.reduce({})["sampling_call_count"] is None


def test_cli_and_rehashed_replay(tmp_path):
    """SCENARIO-REPORT-8902-CONTRACT: cold CLI outside checkout rejects repaired checksums."""
    output = tmp_path / (m.NAME + ".json")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for args, expected in [
        (["--private-output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--replay", str(tmp_path / "missing")], 1),
        (["--date", "20261106"], 2),
    ]:
        process = subprocess.run(
            [sys.executable, "-u", str(m.ROOT / m.CLI), *args],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert process.returncode == expected, process.stdout + process.stderr
    original = json.loads(output.read_bytes())
    value = deepcopy(original)
    value["sampling_call_count"] = 100
    value["reproducibility_checksum"] = m.canonical_hash(value["rows"])
    m.atomic_json(output, value)
    assert m.main(["--replay", str(output)]) == 1
    m.atomic_json(output, original)
    operand = Path(original["source_artifact_hashes"][0]["path"])
    operand.chmod(0o600)
    operand.write_bytes(operand.read_bytes() + b"changed")
    assert m.main(["--replay", str(output)]) == 1


def test_production_route(work, tmp_path, monkeypatch):
    """REQ-REPORT-8902: production invokes owned validation and terminal scanners."""
    monkeypatch.setattr(
        m.checker,
        "validate",
        lambda _: dict(passed=False, receipts=[], coverage=dict(complete=False)),
    )
    monkeypatch.setattr(m.checker, "terminal", lambda p, r: dict(passed=True))
    monkeypatch.setattr(m, "collect", lambda _: work)
    monkeypatch.setattr(m, "DEFAULT_OUTPUT", tmp_path / (m.NAME + ".json"))
    assert m.main([]) == 0


def test_qualified_attribution_and_partial_evidence(work):
    """REQ-REPORT-8902: qualified clocks cannot make unsupported fabrics compatible."""
    producers = {
        b: json.loads(Path(ref["path"]).read_bytes()) for b, ref in work["evidence"].items()
    }
    assert m.reduce({})["sampling_call_count"] is None
    producers["attribution"] = dict(rows=[], cost_attribution_ready_score=1)
    value = m.reduce(producers)
    assert all(r["qualified_fraction"] == 0 for r in value["compatible_fraction_rows"])
    assert all(r["infinite_component_speed_bound"] == 1 for r in value["amdahl_bounds"])
    assert all(r["host_durability_remainder"] == 1 for r in value["amdahl_bounds"])
    assert (
        m.admission(
            dict(producers["labels"], certificate_rows_ready_score=1, verdict_class="positive"),
            "labels",
        )
        == "rejected_producer"
    )


def test_retention_controls(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8902-CONTRACT: reject changed bytes and reuse immutable originals."""
    source = tmp_path / "source"
    source.write_text("original")
    digest = m.sha256_file(source)
    manifest = tmp_path / "authority.json"
    m.atomic_json(
        manifest, dict(source_artifact_hashes={"source": dict(path=str(source), sha256=digest)})
    )
    absent = tmp_path / "missing"
    monkeypatch.setattr(
        m, "HISTORY", {str(manifest): m.sha256_file(manifest), str(absent): "missing"}
    )
    monkeypatch.setattr(m, "PRODUCERS", {})

    def controlled(raw):
        first = m.engine.retain(source, raw, digest)
        assert m.engine.retain(source, raw, digest) == first
        assert m.engine.retain(absent, raw, digest)["sha256"] == digest
        with pytest.raises(ValueError, match="operand_hash_drift"):
            m.engine.retain(source, raw, "sha256:incorrect")
        target = Path(first["path"])
        target.chmod(0o600)
        target.write_text("changed")
        with pytest.raises(ValueError, match="immutable_operand_drift"):
            m.engine.retain(source, raw)
        return dict(checks={}, evidence={}, refs=[])

    monkeypatch.setattr(m.engine, "collect", controlled)
    assert m.collect(tmp_path / "raw")["checks"]["physical_probes_performed"] == 0


def test_missing_and_rejected_upstreams(tmp_path, monkeypatch):
    """REQ-REPORT-8902: admit each branch independently and name missing producer fields."""
    path = tmp_path / "producer.json"
    m.atomic_json(path, dict(experiment_id=8896, rows=[]))
    missing = (str(tmp_path / "missing"), "sha256:absent", "certificate_rows_ready_score", 8896)
    malformed = (str(path), m.sha256_file(path), "certificate_rows_ready_score", 8896)
    monkeypatch.setattr(m, "HISTORY", {})
    monkeypatch.setattr(m, "PRODUCERS", dict(missing=missing, malformed=malformed))
    monkeypatch.setattr(m.engine, "PRODUCERS", m.PRODUCERS)
    monkeypatch.setattr(m.engine, "HISTORY", {})
    monkeypatch.setattr(m.engine, "SOURCES", [])
    work = m.collect(tmp_path / "raw")
    assert {r["missing_reason"] for r in work["failures"]} == {"missing_file", "missing_field"}
