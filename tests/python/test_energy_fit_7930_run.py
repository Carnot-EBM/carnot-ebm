"""Execution and consumer regression cases for REQ-VERIFY-7930-V688."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.verify import energy_fit_7930 as core
from carnot.verify import energy_fit_7930_run as run


def test_frozen_scope(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7930-TERMINAL: exact includes and dates precede results."""
    path, commands = run.freeze(tmp_path / "private", tmp_path / "raw")
    frozen = json.loads(path.read_text())
    assert frozen["coverage_includes"] == run.INCLUDES
    assert frozen["configuration"]["arms"] == list(core.ARMS)
    assert len(core.ARMS) * len(core.SEEDS) == 27
    for command in commands:
        if command.name.startswith("e2e_016"):
            assert command.argv[command.argv.index("--date") + 1] == "20260929"
    assert next(c for c in commands if c.name == "full_pytest").scope == "repository_health"


def test_real_expected_failure_receipt(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7930-TERMINAL: child exit and reason must both match."""
    py = str(core.ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "expected",
            (py, "-c", "print('source evidence blocked');raise SystemExit(2)"),
            "expected_failure",
            30,
        ),
        CommandSpec("success", (py, "-c", "print('done')"), "required", 30),
    ]
    rows = run.execute(commands, tmp_path)
    assert all(row["passed"] for row in rows)
    assert rows[0]["actual_exit"] == 2
    assert sha256_file(Path(rows[0]["log_path"])) == rows[0]["log_sha256"]


def test_publish_real_consumers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-VERIFY-7930-TERMINAL: newer attestations select the checked primary."""
    upstream = tmp_path / "upstream.json"
    atomic_json(upstream, {})
    output = tmp_path / "experiment_7930_fixture.json"
    q = core.library(upstream, output)
    value = run.base(
        q,
        upstream,
        [],
        [
            {
                "upstream_id": "exp7916",
                "artifact_field": "artifact",
                "op": "exists",
                "expected": True,
                "observed": None,
            }
        ],
    )
    monkeypatch.setattr(
        run, "terminal_checks", lambda p, raw: {"passed": True, "flagged_adversarial": False}
    )
    run.publish(value, output)
    assert len(list(tmp_path.glob("experiment_7930_*.json"))) == 1
    selected = json.loads(
        (tmp_path / "raw" / output.stem / "primary_resolution_receipt.json").read_text()
    )
    assert selected["gate_sha256"] == sha256_file(output)
    monkeypatch.setattr(
        run, "terminal_checks", lambda p, raw: {"passed": True, "flagged_adversarial": True}
    )
    run.publish(value, output)
    assert json.loads(output.read_text())["verdict_class"] == "disqualified"


@pytest.mark.parametrize("failure", [None, "blocked", "timeout", "owned", "heads", "late_timeout"])
def test_pipeline_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    """SCENARIO-VERIFY-7930-TERMINAL: external, unfinished and owned failures differ."""
    upstream = tmp_path / "upstream.json"
    atomic_json(upstream, {})
    output = tmp_path / "experiment_7930_fixture.json"
    raw = tmp_path / "raw" / output.stem
    real = core.library(upstream, output)
    q = SimpleNamespace(
        base=real.base, controls=lambda *a: [], score=lambda *a: (tmp_path / "rows.jsonl", [])
    )
    monkeypatch.setattr(core, "library", lambda *a: q)
    monkeypatch.setattr(
        core, "authenticate", lambda *a: ([], {}, {"historical_required_failures": []})
    )
    monkeypatch.setattr(core, "public_records", lambda *a: ([{"role": "fit"}], [], []))
    monkeypatch.setattr(core, "attach_labels", lambda records, *a: records)
    monkeypatch.setattr(run, "freeze", lambda *a: (tmp_path / "manifest.json", []))
    atomic_json(tmp_path / "manifest.json", {"dependencies": {}, "configuration": {}})
    monkeypatch.setattr(run, "execute", lambda *a: [])
    monkeypatch.setattr(run, "publish", lambda value, path: atomic_json(path, value))
    checkpoints = [{"arm": "local_set", "seed": 67801, "parameter_count": 266}] * 27
    monkeypatch.setattr(core, "fit_heads", lambda *a: (checkpoints, tmp_path / "heads.json"))
    monkeypatch.setattr(run, "enrich", lambda value, *a: value)
    q.candidate = lambda *a: {
        **q.base(upstream, [], []),
        "honest_verdict": "complete_null_energy_fit",
        "verdict_class": "null",
        "rows": [],
        "trained_head_specs": checkpoints,
    }

    def score(*args: object) -> tuple[Path, list[object]]:
        q.progress("score", "before arm=local_set seed=67801 role=fit")
        q.progress("score", "after arm=local_set seed=67801 role=fit")
        q.progress("controls", "boundary")
        return tmp_path / "rows.jsonl", []

    q.score = score
    if failure == "heads":
        checkpoints.pop()
    if failure == "late_timeout":
        clock = [core.START + 1]
        monkeypatch.setattr(run, "time", SimpleNamespace(monotonic=lambda: clock[0]))

        def controls(*args: object) -> list[object]:
            clock[0] += 3001
            return []

        q.controls = controls
    if failure == "blocked":

        def blocked(*args: object) -> object:
            raise core.custody.InputBlocked([{"upstream_id": "exp7916", "observed": None}])

        monkeypatch.setattr(core, "authenticate", blocked)
    elif failure in ("timeout", "owned"):

        def fail(*args: object) -> object:
            raise (
                TimeoutError("budget")
                if failure == "timeout"
                else ValueError("invalid numeric head")
            )

        monkeypatch.setattr(core, "fit_heads", fail)
    result = run.produce(upstream, tmp_path / "runtime.json", output)
    value = json.loads(output.read_text())
    assert result == (1 if failure in ("timeout", "late_timeout") else 0)
    assert value["experiment_id"] == 7930
    assert (
        value["verdict_class"]
        == {
            None: "null",
            "blocked": "blocked",
            "timeout": "partial",
            "owned": "disqualified",
            "heads": "disqualified",
            "late_timeout": "partial",
        }[failure]
    )


@pytest.mark.parametrize("ready", [True, False])
def test_energy_rows_and_coverage_reduction(tmp_path: Path, ready: bool) -> None:
    """SCENARIO-VERIFY-7930-TERMINAL: primitive energies and nonempty coverage gate readiness."""
    upstream = tmp_path / "upstream.json"
    atomic_json(upstream, {})
    q = core.library(upstream, tmp_path / "experiment_7930_fixture.json")
    value = run.base(q, upstream, [], [])
    value.update(
        {
            "trained_head_specs": [{}] * 27,
            "sample_size_budget": {"eligible": 1, "by_role": {"fit": {"eligible": 1}}},
        }
    )
    prediction = tmp_path / "prediction.jsonl"
    prediction.write_text(
        json.dumps({"arm": "local_set", "seed": 67801, "role": "fit", "raw_risk": 0.3}) + "\n"
    )
    if ready:
        atomic_json(
            tmp_path / "coverage.json",
            {
                "files": {
                    name: {"summary": {"num_statements": 1, "missing_lines": 0}}
                    for name in core.OWNED
                }
            },
        )
    result = run.enrich(
        value, prediction, [], {"arm=local_set seed=67801 role=fit": 0.1}, tmp_path, []
    )
    assert result["energy_fit_ready_score"] == int(ready)
    assert result["rows"][0]["energy_unsupported"] > 0
    assert result["sample_size_budget"]["independent"] == 0


def test_consumer_mismatch_rejects_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7930-TERMINAL: a reader cannot silently consume another artifact."""
    upstream = tmp_path / "upstream.json"
    atomic_json(upstream, {})
    output = tmp_path / "experiment_7930_fixture.json"
    value = run.base(core.library(upstream, output), upstream, [], [])
    monkeypatch.setattr(
        run, "terminal_checks", lambda *a: {"passed": True, "flagged_adversarial": False}
    )
    monkeypatch.setattr(core.publication, "reader_receipt", lambda *a, **kw: {"passed": False})
    with pytest.raises(ValueError, match="consumer mismatch"):
        run.publish(value, output)
