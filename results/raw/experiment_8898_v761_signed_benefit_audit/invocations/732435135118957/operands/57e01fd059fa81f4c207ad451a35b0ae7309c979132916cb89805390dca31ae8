"""REQ-REPORT-8886 / SCENARIO-REPORT-8886-CONTRACT: independent oracle audit."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import label_benefit_audit_8886 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    """Keep original producer bytes as evidence, never regenerate a training stream."""
    return json.loads((m.ROOT / "results" / (m.SOURCE + ".json")).read_bytes())


@pytest.fixture(scope="module")
def original(source: dict[str, Any]) -> dict[str, Any]:
    """Use an original causal shard so reconstruction tests cannot choose easy rows."""
    return json.loads(Path(source["stream_refs"][0]["path"]).read_bytes())


@pytest.fixture(scope="module")
def reduced(source: dict[str, Any]) -> dict[str, Any]:
    """REQ-REPORT-8886: all24 paired streams reduce with training disabled."""
    with patch.object(m.base.c, "update", side_effect=AssertionError("training forbidden")):
        return m.reduce(source)


def test_frozen_null_and_cluster_census(source: dict[str, Any], reduced: dict[str, Any]) -> None:
    """REQ-REPORT-8886: query utility cannot pass correct-label gates or inflate n."""
    assert len(reduced["rows"]) == 18432
    assert reduced["independent_count"] == 24
    assert len(reduced["paired_stream_rows"]) == 24 * 9
    assert len(reduced["simultaneous_intervals"]) == 9
    expected = m.base.summarize(source["rows"])["intervals"]
    assert reduced["simultaneous_intervals"] == expected
    assert reduced["label_benefit_score"] == 0
    assert all(r["passed"] for r in reduced["leakage_checks"])
    assert (
        reduced["continuation_decision"]["decision"] == "retire_unchanged_matched_label_mechanism"
    )
    assert len(reduced["template_use_changes"]) == 24
    assert reduced["topology_support"]["online_topologies"] == 8
    assert m.empty("absent")["censored_count"] == 18432
    assert m.empty("absent")["independent_count"] == 0


@pytest.mark.parametrize("which", ["specificity", "future", "retention", "none"])
def test_joint_frozen_boundaries(which: str, reduced: dict[str, Any]) -> None:
    """REQ-REPORT-8886: strict label specificity and inclusive margins apply jointly."""
    intervals = deepcopy(reduced["simultaneous_intervals"])
    for r in intervals:
        r["upper"] = -0.02 if r["split"] == "online" else 0.01
    assert m.gates(intervals)["label_benefit_score"] == 1
    selected = {"specificity": 1, "future": 0, "retention": 3, "none": 2}[which]
    intervals[selected]["upper"] = (
        0 if which == "specificity" else -0.009 if which == "future" else 0.011
    )
    assert m.gates(intervals)["label_benefit_score"] == int(which == "none")


@pytest.mark.parametrize(
    "mutation",
    ["probability", "early", "multiset", "admission", "retention", "checkpoint", "identity"],
)
def test_reconstruction_rejects_semantic_changes(original: dict[str, Any], mutation: str) -> None:
    """SCENARIO-REPORT-8886-CONTRACT: changed operands cannot repair their own hashes."""
    value = deepcopy(original)
    if mutation == "probability":
        value["rows"][0]["sealed"]["prediction"] = [1.0, 0.0, 0.0]
    elif mutation == "early":
        value["batches"][0]["release_ordinal"] = 1
    elif mutation == "multiset":
        value["batches"][0]["permuted_labels"] = ["support"] * 6
    elif mutation == "admission":
        value["transaction_rows"][4]["state"]["templates"] = ["foreign"]
    elif mutation == "retention":
        value["transaction_rows"][0]["state"]["released"] = ["retention_truth"]
    elif mutation == "checkpoint":
        value["rows"][0]["sealed"]["state_hash"] = "foreign"
    else:
        value["stream"]["seed"] = 999
    with pytest.raises((ValueError, KeyError)):
        m.reconstruct(value, m.base.protocol()["cohort_manifest"][0])


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "missing_file"),
        ({}, "missing_field"),
        ({"causal_rows_ready_score": 0}, "observed_zero"),
        ({"causal_rows_ready_score": 1}, "rejected_producer"),
    ],
)
def test_external_gates(tmp_path: Path, value: Any, expected: str) -> None:
    """REQ-REPORT-8886: external absence and rejected assertions are terminal blocking."""
    if value is not None:
        atomic_json(tmp_path / (m.SOURCE + ".json"), value)
    pre = m.preflight(tmp_path / "raw", tmp_path)
    assert not pre["external_ready"]
    assert pre["diagnostics"][0]["check"] == expected


def test_original_preflight(tmp_path: Path) -> None:
    """REQ-REPORT-8886: retained inputs and execution authority authenticate before reduction."""
    pre = m.preflight(tmp_path, m.ROOT / "results")
    assert pre["external_ready"] and pre["resources"]["passed"]
    assert pre["references"]


@pytest.mark.parametrize(
    "failure", ["", "external", "resources", "reduction", "validation", "terminal"]
)
def test_runner_terminal_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reduced: dict[str, Any],
    source: dict[str, Any],
    failure: str,
) -> None:
    """REQ-REPORT-8886: owned failures disqualify while external absence blocks."""
    from carnot.reporting import label_benefit_runner_8886 as r

    raw_pre = dict(
        external_ready=failure != "external",
        resources=dict(passed=failure != "resources"),
        references=[],
        diagnostics=[],
    )

    def preflight(raw: Path, sources: Path) -> dict[str, Any]:
        atomic_json(raw / (m.SOURCE + ".json"), source)
        return raw_pre

    monkeypatch.setattr(m, "preflight", preflight)

    def reduction(value: dict[str, Any]) -> dict[str, Any]:
        if failure == "reduction":
            raise ValueError("owned_reduction_failure")
        return deepcopy(reduced)

    monkeypatch.setattr(m, "reduce", reduction)

    def validation(*args: Any) -> dict[str, Any]:
        if failure == "validation":
            raise ValueError("owned_validation_failure")
        return dict(
            receipts=[dict(passed=True)],
            union=dict(passed=True),
            coverage=dict(complete=True, percent=100, files=[]),
            raw_references=[],
        )

    monkeypatch.setattr(r, "validate", validation)
    log = tmp_path / "terminal.log"
    log.write_text("fixture terminal receipt")
    monkeypatch.setattr(
        r, "terminal", lambda *args: [dict(passed=failure != "terminal", stdout_path=str(log))]
    )
    monkeypatch.setattr(r, "replay", lambda *args: dict(passed=True))
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    artifact = json.loads(output.read_bytes())
    assert artifact["verdict_class"] == (
        "blocked" if failure == "external" else "null" if not failure else "disqualified"
    )
    assert artifact["audit_ready_score"] == int(not failure)
    assert artifact["numerical_training_calls"] == 0
    assert artifact["independent_generalization_score"] == 0


def test_cli_private_and_replay(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8886-CONTRACT: run actual CLI outside checkout without PYTHONPATH."""
    import os
    import subprocess
    import sys

    output = tmp_path / (m.NAME + ".json")
    config = dict(os.environ)
    config.pop("PYTHONPATH", None)
    cmd = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    run = subprocess.run(
        cmd + ["--private-output", str(output), "--sources", str(tmp_path)],
        cwd="/tmp",
        env=config,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["gate_check_summary"]
    replay = subprocess.run(
        cmd + ["--cold-replay", str(output)],
        cwd="/tmp",
        env=config,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    value["label_benefit_score"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    bad = subprocess.run(
        cmd + ["--replay", str(output)],
        cwd="/tmp",
        env=config,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert bad.returncode == 1 and "rejected" in bad.stdout


def test_runner_options_and_wrappers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8886: reject wrong dates and unsafe private paths before writing."""
    from carnot.reporting import label_benefit_runner_8886 as r
    from types import SimpleNamespace

    for argv in (["--date", "20261107"], ["--private-output", str(m.ROOT / "bad.json")]):
        with pytest.raises(SystemExit):
            r.main(argv)
    with patch.object(
        r,
        "namespace",
        return_value=SimpleNamespace(validate=lambda *args: {}, terminal=lambda *args: []),
    ):
        assert r.validate(tmp_path, tmp_path) == {}
        assert r.terminal(tmp_path, tmp_path) == []
    monkeypatch.setattr(r, "replay", lambda *args: {"passed": True})
    assert r.main(["--replay", "anything"]) == 0


def test_rejected_original_qualification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8886: a rehashed readiness claim cannot replace original qualification."""
    from types import SimpleNamespace

    for name in (m.SOURCE, m.METHOD):
        atomic_json(
            tmp_path / (name + ".json"),
            dict(
                required_checks_passed=False,
                owned_statement_coverage=dict(complete=False, percent=0),
                control_rows=[],
                protocol=m.base.protocol(),
            ),
        )
    monkeypatch.setattr(
        m,
        "namespace",
        lambda *args: SimpleNamespace(
            preconditions=lambda *args: dict(external_ready=True, resources={}, diagnostics=[])
        ),
    )
    assert not m.preflight(tmp_path, tmp_path)["external_ready"]


def test_complete_replay_and_rehashed_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reduced: dict[str, Any]
) -> None:
    """SCENARIO-REPORT-8886-CONTRACT: complete private evidence rejects repaired summaries."""
    from carnot.reporting import label_benefit_runner_8886 as r
    import shutil

    def preflight(raw: Path, sources: Path) -> dict[str, Any]:
        shutil.copyfile(m.ROOT / "results" / (m.SOURCE + ".json"), raw / (m.SOURCE + ".json"))
        return dict(external_ready=True, resources=dict(passed=True), references=[], diagnostics=[])

    monkeypatch.setattr(m, "preflight", preflight)
    monkeypatch.setattr(m, "reduce", lambda *args: deepcopy(reduced))
    files = [dict(path=p, statements=1, missing=[], excluded=[]) for p in m.OWNED]
    monkeypatch.setattr(
        r,
        "validate",
        lambda *args: dict(
            receipts=[dict(passed=True)],
            union=dict(passed=True),
            coverage=dict(complete=True, percent=100, files=files),
            raw_references=[],
        ),
    )
    monkeypatch.setattr(r, "terminal", lambda *args: [dict(passed=True)])
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    assert r.replay(output)["passed"]
    original = json.loads(output.read_bytes())
    for key, changed in [
        ("acceptance_gates", {}),
        ("label_benefit_score", 1),
        ("independent_count", 120),
        ("audit_ready_score", 0),
        ("independent_generalization_score", 1),
        ("required_checks_passed", False),
    ]:
        bad = deepcopy(original)
        bad[key] = changed
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(output, bad)
        with pytest.raises(ValueError):
            r.replay(output)
