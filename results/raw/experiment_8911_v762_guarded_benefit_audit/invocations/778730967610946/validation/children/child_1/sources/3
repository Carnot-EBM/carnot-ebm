"""REQ-REPORT-8911 / SCENARIO-REPORT-8911-CONTRACT: audit sealed paired streams."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import guarded_benefit_audit_8911 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    """Keep original producer bytes so tests cannot regenerate favorable outcomes."""
    return json.loads((m.ROOT / "results" / (m.SOURCE + ".json")).read_bytes())


@pytest.fixture(scope="module")
def stream(source: dict[str, Any]) -> dict[str, Any]:
    """Use one sealed shard for adverse controls without training a replacement."""
    return json.loads(Path(source["stream_refs"][0]["path"]).read_bytes())


@pytest.fixture(scope="module")
def reduced(source: dict[str, Any]) -> dict[str, Any]:
    """REQ-REPORT-8911: reduction must not request truth or fit any head."""
    with patch.object(m.causal.old, "world", side_effect=AssertionError("oracle forbidden")):
        with patch.object(m.causal.signed, "update", side_effect=AssertionError("fit forbidden")):
            return m.reduce(source)


def test_original_preflight_and_external_gates(tmp_path: Path) -> None:
    """REQ-REPORT-8911: original qualified bytes authorize work; absence blocks it."""
    with patch.object(m.causal.old, "world", side_effect=AssertionError("oracle forbidden")):
        pre = m.preflight(tmp_path / "original", m.ROOT / "results")
    assert pre["external_ready"] and pre["resources"]["passed"] and pre["references"]
    for value, expected in [
        (None, "missing_file"),
        ({}, "missing_field"),
        ({"causal_rows_ready_score": 0}, "observed_zero"),
        ({"causal_rows_ready_score": 1}, "rejected_producer"),
    ]:
        if value is not None:
            atomic_json(tmp_path / (m.SOURCE + ".json"), value)
        blocked = m.preflight(tmp_path / expected, tmp_path)
        assert not blocked["external_ready"]
        assert blocked["diagnostics"][0]["check"] == expected
    (tmp_path / (m.SOURCE + ".json")).write_text("{")
    assert not m.preflight(tmp_path / "malformed", tmp_path)["external_ready"]
    assert m.empty("absent")["censored_count"] == 18432


def test_full_reduction_preserves_every_gate(reduced: dict[str, Any]) -> None:
    """REQ-REPORT-8911: one conjunction owns benefit, while zeros remain evidence."""
    assert reduced["completed_count"] == len(reduced["rows"]) == 18432
    assert reduced["independent_count"] == 24
    assert len(reduced["paired_intervals"]) == len(reduced["acceptance_gate_rows"]) == 9
    assert len(reduced["paired_stream_rows"]) == 216
    assert all(r["independent_n"] == 24 for r in reduced["paired_intervals"])
    assert reduced["multiplicity_ledger"]["upper_tail_alpha"] == 0.01 / 9
    assert all(r["passed"] for r in reduced["control_rows"])
    assert reduced["paired_intervals"] == m.summarize(reduced["rows"])["paired_intervals"]
    assert all(r["attempts"] == r["promoted"] + r["rejected"] for r in reduced["promotion_summary"])


@pytest.mark.parametrize(
    "mutation",
    ["swapped", "dropped_failure", "duplicate", "leak", "denominator", "frozen", "prediction"],
)
def test_rehashed_stream_tampering(stream: dict[str, Any], mutation: str) -> None:
    """SCENARIO-REPORT-8911-CONTRACT: repaired hashes cannot excuse semantic changes."""
    bad = deepcopy(stream)
    if mutation == "swapped":
        bad["rows"][0]["arm"] = "ordinary_true"
    elif mutation == "dropped_failure":
        bad["rows"][0]["status"] = "failed"
        bad["rows"].pop(0)
    elif mutation == "duplicate":
        bad["rows"][0]["cluster_id"] = "v762-evaluation-201"
    elif mutation == "leak":
        bad["states"]["guarded_true"]["promotions"][0]["training_groups"].append(
            stream["plan"]["retention"][0]["unit_id"]
        )
    elif mutation == "denominator":
        bad["rows"][0]["denominator"] = 24
    elif mutation == "frozen":
        bad["states"]["frozen_verified"]["committed_version"] = 1
    else:
        bad["rows"][0]["prediction"] = [1.0, 0.0, 0.0]
    bad.pop("stream_checksum")
    bad["stream_checksum"] = canonical_hash(bad)
    with pytest.raises((ValueError, KeyError, StopIteration)):
        m.reconstruct(bad, stream["plan"])


def test_all_null_and_simultaneous_boundaries(source: dict[str, Any]) -> None:
    """REQ-REPORT-8911: no effect never becomes scientific success."""
    rows = [dict(r, metric=0.0, error=0) for r in source["rows"]]
    summary = m.summarize(rows)
    assert summary["guarded_benefit_score"] == 0
    assert all(r["mean"] == r["upper"] == 0 for r in summary["paired_intervals"])
    intervals = deepcopy(summary["paired_intervals"])
    for row in intervals:
        row["upper"] = -0.01 if row["split"] == "online" else 0.01
    assert m.gates(intervals)["guarded_benefit_score"] == 1
    for index in range(9):
        changed = deepcopy(intervals)
        changed[index]["upper"] = -0.009 if index == 0 else 0 if index < 3 else 0.011
        assert m.gates(changed)["guarded_benefit_score"] == 0
    for bad in (
        rows[:-1],
        [dict(r, denominator=24) for r in rows],
        [dict(r, cluster_id="same") for r in rows],
    ):
        with pytest.raises(ValueError):
            m.summarize(bad)


def test_imported_row_and_cluster_drift(source: dict[str, Any]) -> None:
    """REQ-REPORT-8911: the producer census must equal the original shard rows."""
    with pytest.raises(ValueError):
        m.reduce(dict(source, rows=source["rows"][:-1]))
    with pytest.raises(ValueError):
        m.reduce(dict(source, stream_refs=[source["stream_refs"][0]] * 24))


@pytest.mark.parametrize(
    "failure", ["", "external", "resources", "reduction", "validation", "terminal"]
)
def test_runner_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reduced: dict[str, Any], failure: str
) -> None:
    """REQ-REPORT-8911: scientific null, external blocks and owned failures stay distinct."""
    from carnot.reporting import guarded_benefit_runner_8911 as r

    def preflight(raw: Path, sources: Path) -> dict[str, Any]:
        atomic_json(raw / (m.SOURCE + ".json"), {})
        return dict(
            external_ready=failure != "external",
            resources=dict(passed=failure != "resources"),
            references=[],
            diagnostics=[dict(check="fixture")] if failure == "external" else [],
        )

    def reduction(value: dict[str, Any]) -> dict[str, Any]:
        if failure == "reduction":
            raise ValueError("owned_reduction_failure")
        return deepcopy(reduced)

    def validation(*args: Any) -> dict[str, Any]:
        if failure == "validation":
            raise ValueError("owned_validation_failure")
        return dict(
            receipts=[dict(passed=True)],
            union=dict(passed=True),
            coverage=dict(
                complete=True,
                percent=100,
                files=[dict(path=p, statements=1, missing=[], excluded=[]) for p in m.OWNED],
            ),
            raw_references=[],
        )

    monkeypatch.setattr(m, "preflight", preflight)
    monkeypatch.setattr(m, "reduce", reduction)
    monkeypatch.setattr(r, "validate", validation)
    monkeypatch.setattr(r, "replay", lambda *args: dict(passed=True))
    log = tmp_path / "terminal.log"
    log.write_text("terminal fixture")
    monkeypatch.setattr(
        r, "terminal", lambda *args: [dict(passed=failure != "terminal", stdout_path=str(log))]
    )
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    expected = (
        "blocked"
        if failure == "external"
        else "disqualified"
        if failure
        else "circular_positive"
        if reduced["guarded_benefit_score"]
        else "null"
    )
    assert value["verdict_class"] == expected
    assert value["audit_ready_score"] == int(not failure)
    assert value["numerical_training_calls"] == 0
    assert not failure or value["guarded_benefit_score"] == 0


def test_private_cli_replay_and_tampering(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8911-CONTRACT: a new checksum cannot authorize false claims."""
    from carnot.reporting import guarded_benefit_runner_8911 as r

    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--private-output", str(output), "--sources", str(tmp_path)]) == 0
    assert r.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    for field, replacement in [
        ("rows", []),
        ("audit_ready_score", 1),
        ("guarded_benefit_score", 1),
        ("verdict_class", "positive"),
        ("experiment_id", 1),
        ("numerical_training_calls", 1),
        ("acceptance_gates", {}),
        ("MODEL_SPECS", [{}]),
    ]:
        bad = dict(value, **{field: replacement})
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(tmp_path / "bad.json", bad)
        assert r.main(["--replay", str(tmp_path / "bad.json")]) == 1
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "bad.json")])


def test_adapters_and_actual_cold_children(tmp_path: Path) -> None:
    """REQ-REPORT-8911: execute the dated CLI outside checkout with no PYTHONPATH."""
    import os
    import subprocess
    import sys
    from types import SimpleNamespace
    from carnot.reporting import guarded_benefit_runner_8911 as r

    adapter = SimpleNamespace(validate=lambda *a: {}, terminal=lambda *a: [])
    with patch.object(r, "namespace", return_value=adapter):
        assert r.validate(tmp_path, tmp_path) == {}
        assert r.terminal(tmp_path, tmp_path) == []
    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    cmd = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    for args in (
        ["--private-output", str(output), "--sources", str(tmp_path)],
        ["--replay", str(output)],
    ):
        completed = subprocess.run(
            cmd + args, cwd="/tmp", env=env, capture_output=True, text=True, timeout=120
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_qualified_original_publication_and_replay(tmp_path: Path) -> None:
    """REQ-REPORT-8911: a qualified null replays original rows without refitting."""
    from carnot.reporting import guarded_benefit_runner_8911 as r

    validation = dict(
        receipts=[dict(passed=True)],
        union=dict(passed=True),
        coverage=dict(
            complete=True,
            percent=100,
            files=[dict(path=p, statements=1, missing=[], excluded=[]) for p in m.OWNED],
        ),
        raw_references=[],
    )
    output = tmp_path / (m.NAME + ".json")
    with patch.object(r, "validate", return_value=validation):
        with patch.object(r, "terminal", return_value=[dict(passed=True)]):
            assert r.main(["--output", str(output)]) == 0
    assert r.replay(output)["passed"]
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "null" and value["audit_ready_score"] == 1
    assert value["guarded_benefit_score"] == 0
    for field, replacement in [
        ("completed_count", 1),
        ("paired_intervals", []),
        ("guarded_benefit_score", 1),
        ("required_checks_passed", False),
    ]:
        changed = dict(value, **{field: replacement})
        changed.pop("reproducibility_checksum")
        changed["reproducibility_checksum"] = canonical_hash(changed)
        atomic_json(tmp_path / "bad.json", changed)
        with pytest.raises(ValueError):
            r.replay(tmp_path / "bad.json")
