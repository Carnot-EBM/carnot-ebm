"""REQ-REPORT-8925 / SCENARIO-REPORT-8925-CONTRACT: cold paired benefit audit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import interpolated_benefit_audit_8925 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    """Read original future evidence; fixtures never become production measurements."""
    return json.loads((m.ROOT / "results" / (m.SOURCE + ".json")).read_bytes())


@pytest.fixture(scope="module")
def stream(source: dict[str, Any]) -> dict[str, Any]:
    """Keep one sealed stream for negative controls without fitting a replacement."""
    return json.loads(Path(source["stream_refs"][0]["path"]).read_bytes())


@pytest.fixture(scope="module")
def reduced(source: dict[str, Any]) -> dict[str, Any]:
    """REQ-REPORT-8925: no world construction, numerical update or new prediction."""
    with (
        patch.object(m.causal.base.old, "world", side_effect=AssertionError("oracle forbidden")),
        patch.object(
            m.causal.methods.signed, "update", side_effect=AssertionError("fit forbidden")
        ),
        patch.object(m.causal, "run_stream", side_effect=AssertionError("rerun forbidden")),
    ):
        return m.reduce(source)


def test_preflight(tmp_path: Path) -> None:
    """REQ-REPORT-8925: authenticate exact gates, producer bytes and retained operands."""
    with patch.object(m.causal.base.old, "world", side_effect=AssertionError("oracle forbidden")):
        pre = m.preflight(tmp_path / "auth", m.ROOT / "results")
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
        assert {
            "check",
            "upstream",
            "artifact_field",
            "expected",
            "observed",
            "path",
            "hash",
        } <= blocked["diagnostics"][0].keys()
    (tmp_path / (m.SOURCE + ".json")).write_text("{")
    assert not m.preflight(tmp_path / "malformed", tmp_path)["external_ready"]
    assert m.empty("missing")["censored_count"] == 18432


def test_complete_null_and_registered_family(reduced: dict[str, Any]) -> None:
    """REQ-REPORT-8925: all 24 pairs and nine original bounds survive the reduction."""
    assert reduced["completed_count"] == len(reduced["rows"]) == 18432
    assert reduced["independent_count"] == 24
    assert len(reduced["paired_intervals"]) == 9
    assert len(reduced["paired_stream_rows"]) == 216
    assert len(reduced["retention_bounds"]) == 6
    assert reduced["multiplicity_ledger"]["upper_tail_alpha"] == 0.01 / 9
    assert reduced["multiplicity_ledger"]["seed"] == 763200
    assert all(r["independent_n"] == 24 for r in reduced["paired_intervals"])
    assert all(r["passed"] for r in reduced["information_controls"])
    assert reduced["interpolated_benefit_score"] == 0
    assert (
        reduced["continuation_decision"]["decision"]
        == "retire_fixed_interpolated_optimizer_guard_schedule"
    )
    assert reduced["continuation_decision"]["continuous_learning_in_general_retired"] is False
    assert sum(r["training_calls"] for r in reduced["operation_counts"]) == 11144
    assert sum(r["prediction_calls"] for r in reduced["operation_counts"]) == 18432
    assert len(reduced["promotion_rows"]) == sum(
        r["attempts"] for r in reduced["promotion_summary"]
    )
    assert reduced["paired_intervals"] == m.summarize(reduced["rows"])["paired_intervals"]


@pytest.mark.parametrize(
    "mutation",
    ["truth", "release", "retention", "prediction", "frozen", "denominator", "drop", "scoring"],
)
def test_repaired_hash_rejects_semantic_drift(stream: dict[str, Any], mutation: str) -> None:
    """SCENARIO-REPORT-8925-CONTRACT: future-label and retention mutations must fail."""
    bad = deepcopy(stream)
    if mutation == "truth":
        bad["rows"][100]["truth"] = "invalid"
    elif mutation == "release":
        bad["feedback_rows"][0]["release_tick"] = 0
    elif mutation == "retention":
        bad["states"]["interpolated_true"]["promotions"][0]["guard_groups"].append(
            stream["plan"]["retention"][0]["unit_id"]
        )
    elif mutation == "prediction":
        bad["rows"][0]["prediction"] = [1.0, 0.0, 0.0]
    elif mutation == "frozen":
        bad["states"]["frozen_verified"]["committed_version"] = 1
    elif mutation == "denominator":
        bad["rows"][0]["denominator"] = 24
    elif mutation == "drop":
        bad["rows"].pop()
    else:
        bad["scoring_rows"][100]["truth"] = "invalid"
    bad.pop("stream_checksum")
    bad["stream_checksum"] = canonical_hash(bad)
    with pytest.raises((ValueError, KeyError, StopIteration)):
        m.reconstruct(bad, stream["plan"])


def test_bounds_and_missing_pairs(source: dict[str, Any]) -> None:
    """REQ-REPORT-8925: equality is allowed only for frozen and retention margins."""
    rows = [dict(r, metric=0.0, error=0) for r in source["rows"]]
    summary = m.summarize(rows)
    assert summary["interpolated_benefit_score"] == 0
    intervals = deepcopy(summary["paired_intervals"])
    for row in intervals:
        row["upper"] = -0.01 if row["split"] == "online" else 0.01
    assert m.gates(intervals)["interpolated_benefit_score"] == 1
    for index in range(9):
        bad = deepcopy(intervals)
        bad[index]["upper"] = -0.009 if index == 0 else 0 if index < 3 else 0.011
        assert m.gates(bad)["interpolated_benefit_score"] == 0
    for bad_rows in (rows[:-1], [dict(r, denominator=24) for r in rows]):
        with pytest.raises(ValueError):
            m.summarize(bad_rows)
    with pytest.raises(ValueError):
        m.reduce(dict(source, stream_refs=source["stream_refs"][:-1]))


@pytest.mark.parametrize(
    "failure", ["", "external", "resources", "reduction", "validation", "terminal"]
)
def test_runner_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reduced: dict[str, Any], failure: str
) -> None:
    """REQ-REPORT-8925: null, blocked and disqualified are distinct terminal routes."""
    from carnot.reporting import interpolated_benefit_runner_8925 as r

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
    monkeypatch.setattr(r, "terminal", lambda *args: [dict(passed=failure != "terminal")])
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == (
        "blocked" if failure == "external" else "disqualified" if failure else "null"
    )
    assert value["audit_ready_score"] == int(not failure)
    assert value["numerical_training_calls"] == 0
    assert value["experiment_id"] == 8925 and value["milestone"] == "2026.11.763"
    assert value["interpolated_benefit_score"] == 0


def test_private_cli_and_rehashed_tampering(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8925-CONTRACT: foreign-directory children validate current bytes."""
    from carnot.reporting import interpolated_benefit_runner_8925 as r

    output = tmp_path / (m.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    for args in (
        ["--private-output", str(output), "--sources", str(tmp_path)],
        ["--cold-replay", str(output)],
    ):
        child = subprocess.run(
            command + args, cwd="/tmp", env=env, capture_output=True, text=True, timeout=120
        )
        assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_bytes())
    assert value["gate_check_summary"] and value["verdict_class"] == "blocked"
    for field, replacement in [
        ("rows", []),
        ("audit_ready_score", 1),
        ("interpolated_benefit_score", 1),
        ("verdict_class", "positive"),
        ("experiment_id", 1),
        ("numerical_training_calls", 1),
        ("acceptance_gates", {}),
        ("MODEL_SPECS", [{}]),
        ("producer_invocation", {}),
    ]:
        bad = dict(value, **{field: replacement})
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(tmp_path / "bad.json", bad)
        assert r.main(["--replay", str(tmp_path / "bad.json")]) == 1
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "bad.json")])


def test_qualified_original_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8925: the complete original null cold-reduces without training."""
    from carnot.reporting import interpolated_benefit_runner_8925 as r

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
    monkeypatch.setattr(r, "validate", lambda *args: deepcopy(validation))
    monkeypatch.setattr(r, "terminal", lambda *args: [dict(passed=True)])
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    assert r.replay(output)["passed"]
    value = json.loads(output.read_bytes())
    assert value["audit_ready_score"] == 1 and value["verdict_class"] == "null"
    for field, replacement in [
        ("completed_count", 1),
        ("paired_intervals", []),
        ("interpolated_benefit_score", 1),
        ("operation_counts", []),
        ("information_controls", []),
        ("required_checks_passed", False),
    ]:
        bad = dict(value, **{field: replacement})
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(tmp_path / "bad.json", bad)
        with pytest.raises(ValueError):
            r.replay(tmp_path / "bad.json")


def test_validation_adapters(tmp_path: Path) -> None:
    """REQ-REPORT-8925: reuse unchanged validation and terminal implementations."""
    from carnot.reporting import interpolated_benefit_runner_8925 as r

    with patch.object(r, "_original_validate", return_value={}):
        assert r.validate(tmp_path, tmp_path) == {}
    with patch.object(r, "_original_terminal", return_value=[]):
        assert r.terminal(tmp_path, tmp_path) == []
