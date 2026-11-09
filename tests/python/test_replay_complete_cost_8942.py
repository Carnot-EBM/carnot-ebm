"""REQ-REPORT-8942 / SCENARIO-REPORT-8942-CONTRACT: complete service evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from unittest.mock import patch

import pytest

from carnot.reporting import replay_complete_cost_8942 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def pre(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Authenticate actual inputs; fixture work never counts as research cost."""
    return m.preconditions(tmp_path_factory.mktemp("auth8942"), m.ROOT / "results")


@pytest.fixture(scope="module")
def cell(tmp_path_factory: pytest.TempPathFactory, pre: dict[str, Any]) -> dict[str, Any]:
    """Run a real private restarted cell and a separate uninterrupted control."""
    assert pre["external_ready"]
    return m.execute(tmp_path_factory.mktemp("cell8942"), m.roster()[0], pre["heads"])


def test_roster() -> None:
    """REQ-REPORT-8942: fresh worlds and balanced arm orders must be frozen."""
    roster = m.roster()
    assert roster == m.roster() and len(roster) == 96
    assert len({r["seed"] for r in roster}) == 24
    prior = m.methods.protocol()
    used = {
        u["world_seed"]
        for k in ("pilot_manifest", "frozen_evaluation_manifest")
        for p in prior[k]
        for u in p["online"] + p["retention"]
    }
    assert not used & {u["world_seed"] for r in roster for u in r["plan"]["online"]}
    for arm in m.ARMS:
        assert [sum(r["arm"] == arm and r["order"] == i for r in roster) for i in range(4)] == [
            6
        ] * 4


def test_authentication(tmp_path: Path, pre: dict[str, Any]) -> None:
    """REQ-REPORT-8942: missing inputs block; zero gates cannot become readiness."""
    assert pre["external_ready"] and pre["heads"]
    assert pre["quality_reference"]["compatible_benefit_score"] == 0
    for value, expected in [
        (None, "missing_file"),
        ({}, "missing_field"),
        ({"replay_rows_ready_score": 0}, "observed_zero"),
        ({"replay_rows_ready_score": True}, "rejected_producer"),
        ({"replay_rows_ready_score": 1}, "rejected_producer"),
    ]:
        if value is not None:
            atomic_json(tmp_path / (m.SOURCE + ".json"), value)
        result = m.preconditions(tmp_path / expected, tmp_path)
        assert not result["external_ready"]
        assert result["diagnostics"][0]["check"] == expected
        assert {
            "check",
            "upstream",
            "artifact_field",
            "expected",
            "observed",
            "path",
            "hash",
        } <= result["diagnostics"][0].keys()
    (tmp_path / (m.SOURCE + ".json")).write_text("{")
    assert not m.preconditions(tmp_path / "malformed", tmp_path)["external_ready"]
    with patch.dict(m.PINS, {m.SOURCE: ("wrong", "replay_rows_ready_score")}, clear=True):
        assert not m.preconditions(tmp_path / "badpin", m.ROOT / "results")["external_ready"]


def test_cell(cell: dict[str, Any]) -> None:
    """REQ-REPORT-8942: restart, clocks and full causal work must all qualify."""
    m.verify(cell)
    assert cell["restart"]["passed"] and cell["restart"]["prediction_cursor"] == 24
    assert cell["operations"]["requests"] == cell["operations"]["updates"] == 48
    assert cell["operations"]["gradients"] == 2 * cell["operations"]["train_releases"]
    assert cell["operations"]["sampling_call_count"] == 0
    assert cell["operations"]["file_fsync"] == cell["operations"]["directory_fsync"]
    assert cell["operations"]["file_fsync"] == 56
    for field in ("parent_bounds", "warm_bounds", "cold_bounds"):
        assert cell[field][1] > cell[field][0]
    for mutation in ("restart", "clock", "gradient", "denominator", "identity", "child_bounds"):
        bad = deepcopy(cell)
        if mutation == "restart":
            bad["segments"][1]["initial_state"]["cursor"] = -1
        if mutation == "clock":
            bad["warm_bounds"][1] = 0
        if mutation == "gradient":
            bad["operations"]["gradients"] = 0
        if mutation == "denominator":
            bad["operations"]["requests"] = 47
        if mutation == "identity":
            bad["spec"]["arm"] = "uniform_replay"
        if mutation == "child_bounds":
            bad["commands"][0]["started_monotonic_ns"] = 0
        with pytest.raises(ValueError):
            m.verify(bad)


def test_reduction(tmp_path: Path, cell: dict[str, Any]) -> None:
    """REQ-REPORT-8942: paired process n cannot grow with within-process repeats."""
    empty = m.reduce(tmp_path, "missing")
    assert empty["censored_count"] == 288 and empty["independent_count"] == 0
    for spec in m.roster():
        value = deepcopy(cell)
        value["spec"] = spec
        atomic_json(tmp_path / spec["cluster_id"] / spec["arm"] / "unit.json", value)
    with patch.object(m, "verify"):
        result = m.reduce(tmp_path, "unfinished")
    assert result["completed_count"] == 288 and result["independent_count"] == 24
    assert len(result["service_cost_rows"]) == 12
    assert len(result["paired_cost_intervals"]) == 9
    assert all(r["independent_count"] == 24 for r in result["paired_cost_intervals"])
    assert result["numerical_training_calls"] == sum(
        r["training_calls"] for r in result["process_seed_rows"]
    )
    assert result["sampling_call_count"] == 0
    assert result["removable_fraction"] is result["service_100x_projection"] is None
    bad = deepcopy(cell)
    bad["restart"]["passed"] = False
    atomic_json(tmp_path / m.roster()[0]["cluster_id"] / m.ARMS[0] / "unit.json", bad)
    with pytest.raises(ValueError):
        m.reduce(tmp_path, "bad")


def test_worker_resume_and_failure(tmp_path: Path, pre: dict[str, Any]) -> None:
    """REQ-REPORT-8942: execute all four authentic trainable arms without inner clocks."""
    for spec in m.roster()[:4]:
        request = tmp_path / spec["arm"] / "request.json"
        atomic_json(request, dict(spec=spec, head=next(iter(pre["heads"].values()))))
        m.worker(request, 24)
        m.worker(request, 48)
    with patch.object(m, "stream", return_value=dict(passed=False)):
        with pytest.raises(ValueError):
            m.execute(tmp_path / "failed", m.roster()[0], pre["heads"])


def test_private_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8942-CONTRACT: actual CLI cold replay rejects rehashed drift."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    output = tmp_path / (m.NAME + ".json")
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI)]
    result = subprocess.run(
        argv + ["--private-output", str(output), "--sources", str(tmp_path / "absent")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        subprocess.run(
            argv + ["--replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=120,
        ).returncode
        == 0
    )
    value = json.loads(output.read_bytes())
    value["completed_count"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert (
        subprocess.run(
            argv + ["--replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=120,
        ).returncode
        == 1
    )
    assert (
        subprocess.run(
            argv + ["--date", "20261108"], cwd=tmp_path, env=env, capture_output=True
        ).returncode
        == 2
    )


def test_runner_paths(tmp_path: Path, pre: dict[str, Any], cell: dict[str, Any]) -> None:
    """REQ-REPORT-8942: readiness needs complete costs and never implies benefit."""
    from carnot.reporting import replay_complete_cost_runner_8942 as r

    value = m.reduce(tmp_path, "missing")
    validation = dict(
        coverage=dict(
            complete=True,
            percent=100,
            files=[dict(path=p, missing=[], excluded=[]) for p in m.OWNED],
        ),
        union=dict(passed=True),
        receipts=[dict(passed=True)],
        raw_references=[],
    )
    value.update(
        preconditions_checked=pre,
        owned_errors=[],
        owned_statement_coverage=validation["coverage"],
        test_union=validation["union"],
        validation_receipts=validation["receipts"],
    )
    assert r.semantics(value) == (False, "disqualified")
    value["completed_count"], value["independent_count"] = 288, 24
    value["durability_rows"] = [dict(passed=True)] * 96
    assert r.semantics(value) == (True, "null")
    value["preconditions_checked"] = dict(pre, external_ready=False)
    assert r.semantics(value) == (False, "blocked")
    value["preconditions_checked"] = pre
    with (
        patch.object(r, "validation_runtime") as runtime,
        patch.object(r.terminal_runtime, "e2e018", return_value=dict(passed=True)),
    ):
        runtime.validate.return_value = validation
        runtime.e2e018.return_value = dict(passed=True)
        assert r.validate(tmp_path / "val", tmp_path / "scratch")["coverage"]["percent"] == 100
    with patch.object(r, "terminal_runtime") as runtime:
        runtime.terminal.return_value = [dict(passed=True)]
        assert r.terminal(tmp_path / "candidate", tmp_path) == [dict(passed=True)]
    with (
        patch.object(m, "roster", return_value=[cell["spec"]]),
        patch.object(m, "execute") as execute,
    ):
        m.measure(tmp_path / "measurement", pre)
        execute.assert_called_once()
    output = tmp_path / "main" / (m.NAME + ".json")
    with (
        patch.object(m, "preconditions", return_value=pre),
        patch.object(m, "measure"),
        patch.object(r, "validate", return_value=validation),
        patch.object(r, "terminal", return_value=[dict(passed=True)]),
        patch.object(r, "replay", return_value=dict(passed=True)),
    ):
        assert r.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "disqualified"
    output = tmp_path / "error" / (m.NAME + ".json")
    with (
        patch.object(m, "preconditions", return_value=pre),
        patch.object(m, "measure", side_effect=ValueError("owned_control")),
        patch.object(r, "validate", side_effect=ValueError("validation_control")),
        patch.object(r, "terminal", return_value=[dict(passed=True)]),
        patch.object(r, "replay", return_value=dict(passed=True)),
    ):
        assert r.main(["--output", str(output)]) == 0
    assert json.loads(output.read_bytes())["owned_errors"] == [
        "owned_control",
        "validation_control",
    ]
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))])
    with patch.object(m, "worker") as worker:
        assert r.main(["--segment", str(tmp_path / "request"), "--stop", "24"]) == 0
        worker.assert_called_once()


def test_failed_census_and_absent_quality(
    tmp_path: Path, pre: dict[str, Any], cell: dict[str, Any]
) -> None:
    """REQ-REPORT-8942: retain failed owned cells and preserve costs without quality."""
    spec = m.roster()[0]
    with (
        patch.object(m, "roster", return_value=[spec]),
        patch.object(m, "execute", side_effect=ValueError("owned_control")),
    ):
        with pytest.raises(ValueError):
            m.measure(tmp_path, pre)
    result = m.reduce(tmp_path, "not_attempted")
    assert result["failed_count"] == 3 and result["censored_count"] == 285
    original = m.authenticate

    def without_quality(path: Path, pin: str, field: str | None) -> dict[str, Any]:
        if path.stem == m.QUALITY:
            return original(tmp_path / "missing_quality.json", pin, field)
        return original(path, pin, field)

    with patch.object(m, "authenticate", side_effect=without_quality):
        actual = m.preconditions(tmp_path / "no-quality", m.ROOT / "results")
    assert actual["external_ready"] and not actual["quality_reference"]["authenticated"]
    assert actual["quality_reference"]["missing_quality_gate"][0]["check"] == "missing_file"


def test_deployment_gates() -> None:
    """REQ-REPORT-8942: only separate benefit and complete saving can permit deployment."""
    from carnot.reporting import replay_complete_cost_runner_8942 as r

    value = dict(
        cost_rows_ready_score=1,
        quality_reference=dict(authenticated=True, compatible_benefit_score=1),
        paired_cost_intervals=[
            dict(
                arm="compatible_replay",
                metric="complete_process_s",
                independent_count=24,
                upper=-0.01,
            )
        ],
    )
    assert r.recommendation(value)
    value["quality_reference"]["compatible_benefit_score"] = 0
    assert not r.recommendation(value)
    value["quality_reference"]["compatible_benefit_score"] = True
    assert not r.recommendation(value)
    value["quality_reference"]["compatible_benefit_score"] = 1
    value["paired_cost_intervals"][0]["upper"] = 0.01
    assert not r.recommendation(value)
