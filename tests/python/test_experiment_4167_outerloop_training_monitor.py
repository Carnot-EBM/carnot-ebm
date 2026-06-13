"""Tests for Exp 4167 outer-loop TRM training monitor.

Spec refs: REQ-LEARN-4167, SCENARIO-LEARN-4167-READONLY-MONITOR,
SCENARIO-LEARN-4167-FAITHFUL-STABLE.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from carnot import experiment_4167_outerloop_training_monitor as exp4167


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _write_checkpoint(path: Path, *, epoch: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "state_dict": {"weight": torch.tensor([1.0])}}, path)
    return path


def _write_metrics(
    root: Path,
    *,
    version: int,
    rows: list[tuple[int, int, float | str | None]],
    column: str = "val/exact_accuracy",
) -> Path:
    metrics = (
        root
        / "results"
        / "trm_runs"
        / "contiguous_run_hydra"
        / "csv"
        / f"version_{version}"
        / "metrics.csv"
    )
    metrics.parent.mkdir(parents=True, exist_ok=True)
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "step", column])
        for epoch, step, val in rows:
            writer.writerow([epoch, step, "" if val is None else val])
    return metrics


def _write_pid(root: Path, text: str = "1234\n") -> Path:
    pid_path = root / "results" / "trm_runs" / "contiguous_run.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(text, encoding="utf-8")
    return pid_path


def _process(*, alive: bool, etime: str = "01:02:03") -> exp4167.ProcessStatus:
    return exp4167.ProcessStatus(pid=1234, alive=alive, etime=etime, detail="fixture")


def _make_config(root: Path) -> exp4167.MonitorConfig:
    return exp4167.MonitorConfig(repo_root=root)


def test_req_learn_4167_spec_declares_monitor_contract() -> None:
    """REQ-LEARN-4167: OpenSpec declares the read-only monitor artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4167" in spec
    assert "SCENARIO-LEARN-4167-READONLY-MONITOR" in spec
    assert "SCENARIO-LEARN-4167-FAITHFUL-STABLE" in spec
    assert "results/experiment_4167_outerloop_training_monitor.json" in spec
    assert "SHALL NOT launch native training" in spec
    for field in exp4167.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4167.FIELD_PRINCIPLES


def test_req_learn_4167_parses_checkpoint_pid_and_metrics(tmp_path: Path) -> None:
    """REQ-LEARN-4167: checkpoint, PID, and CSV reads are scalar-only."""

    checkpoint = _write_checkpoint(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=9012,
    )
    _write_metrics(tmp_path, version=1, rows=[(100, 200, 0.42), (101, 201, None)])
    latest_path = _write_metrics(
        tmp_path,
        version=2,
        rows=[(102, 202, 0.6), (103, 203, "0.75")],
        column="val_exact_accuracy",
    )
    _write_pid(tmp_path, "pid=1234\n")
    config = _make_config(tmp_path)

    ckpt = exp4167.read_checkpoint_status(checkpoint)
    assert ckpt.exists is True
    assert ckpt.load_ok is True
    assert ckpt.epoch == 9012
    assert ckpt.mtime_iso is not None
    assert ckpt.to_dict()["path"] == str(checkpoint)

    snapshot = exp4167.read_metrics(config.contiguous_run_dir)
    assert snapshot.current_val == pytest.approx(0.75)
    assert snapshot.latest_row is not None
    assert snapshot.latest_row.metrics_path == latest_path
    assert [row.val_exact_accuracy for row in snapshot.rows] == [0.42, 0.6, 0.75]
    assert snapshot.to_trajectory()[-1]["delta_vs_previous"] == pytest.approx(0.15)
    assert snapshot.metrics_paths == [
        str(tmp_path / "results" / "trm_runs" / "contiguous_run_hydra" / "csv" / "version_1" / "metrics.csv"),
        str(latest_path),
    ]

    assert exp4167.read_pid(config.pid_path) == 1234
    assert exp4167._float_or_none(True) is None
    assert exp4167._float_or_none("bad") is None
    assert exp4167._float_or_none(float("inf")) is None
    assert exp4167._int_or_none(None) is None
    assert exp4167._version_number(Path("csv/version_bad/metrics.csv")) == -1
    assert exp4167.MetricsSnapshot([]).current_val is None
    assert exp4167.MetricsSnapshot([]).latest_row is None
    assert exp4167.MetricRow(Path("x"), -1, 2, None, None, 0.5).signature == ("x", 2, None, None, 0.5)


def test_scenario_learn_4167_live_run_reports_not_stable(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4167-READONLY-MONITOR: live writer blocks graft stability."""

    checkpoint = _write_checkpoint(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=9100,
    )
    _write_metrics(tmp_path, version=10, rows=[(9000, 22000, 0.501), (9100, 22100, 0.7)])
    _write_pid(tmp_path)
    output = tmp_path / "results" / exp4167.RESULT_FILENAME

    artifact = exp4167.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        process_checker=lambda _pid: _process(alive=True),
    )

    assert artifact["honest_verdict"] == "complete: outerloop_training_alive_val_0.7000_below_0.85"
    assert artifact["outerloop_train_alive"] is True
    assert artifact["current_val_exact_accuracy"] == 0.7
    assert artifact["val_crossed_085"] is False
    assert artifact["baseline_faithful"] is False
    assert artifact["checkpoint_epoch"] == 9100
    assert artifact["checkpoint_path"] == str(checkpoint)
    assert artifact["checkpoint_mtime"] is not None
    assert artifact["outerloop_pid"] == 1234
    assert artifact["outerloop_pid_etime"] == "01:02:03"
    assert artifact["read_only_actions"] == {
        "torch_load_cpu_only": True,
        "ps_etime_probe": True,
        "training_launched": False,
        "train_process_stop_attempted": False,
        "stable_checkpoint_written": False,
    }
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_learn_4167_live_crossed_threshold_is_not_stable(tmp_path: Path) -> None:
    """REQ-LEARN-4167: val>=0.85 is not faithful while the process is alive."""

    _write_checkpoint(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=9200,
    )
    _write_metrics(tmp_path, version=10, rows=[(9200, 22200, 0.851)])
    _write_pid(tmp_path)

    artifact = exp4167.build_artifact(
        _make_config(tmp_path),
        process_checker=lambda _pid: _process(alive=True, etime="02:00"),
    )

    assert artifact["honest_verdict"] == "complete: outerloop_val_crossed_0.85_but_checkpoint_live_val_0.8510"
    assert artifact["outerloop_train_alive"] is True
    assert artifact["val_crossed_085"] is True
    assert artifact["baseline_faithful"] is False


def test_scenario_learn_4167_stopped_above_threshold_is_faithful(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4167-FAITHFUL-STABLE: stopped val>=0.85 unlocks graft."""

    _write_checkpoint(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=9300,
    )
    _write_metrics(tmp_path, version=11, rows=[(9300, 22300, 0.84), (9400, 22400, 0.872)])
    _write_pid(tmp_path)

    artifact = exp4167.build_artifact(
        _make_config(tmp_path),
        process_checker=lambda _pid: _process(alive=False, etime=""),
    )

    assert artifact["honest_verdict"] == "complete: outerloop_stable_faithful_val_0.8720"
    assert artifact["outerloop_train_alive"] is False
    assert artifact["current_val_exact_accuracy"] == 0.872
    assert artifact["val_crossed_085"] is True
    assert artifact["baseline_faithful"] is True


def test_req_learn_4167_schema_edges_and_read_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-LEARN-4167: missing inputs remain complete status, not fabricated science."""

    config = _make_config(tmp_path)
    artifact = exp4167.build_artifact(config, process_checker=lambda _pid: _process(alive=False))

    assert artifact["honest_verdict"] == "complete: outerloop_status_missing_val"
    assert artifact["outerloop_train_alive"] is False
    assert artifact["current_val_exact_accuracy"] is None
    assert artifact["checkpoint_mtime"] is None
    assert artifact["baseline_faithful"] is False
    assert exp4167._mtime_iso(None) is None
    assert exp4167._version_number(Path("metrics.csv")) == -1

    corrupt = tmp_path / "bad.ckpt"
    corrupt.write_text("not a checkpoint", encoding="utf-8")
    assert exp4167.read_checkpoint_status(corrupt).load_ok is False
    list_ckpt = tmp_path / "list.ckpt"
    torch.save([1, 2, 3], list_ckpt)
    assert "unexpected checkpoint payload" in exp4167.read_checkpoint_status(list_ckpt).detail
    assert exp4167.read_pid(tmp_path / "missing.pid") is None
    pid_file = tmp_path / "no_digits.pid"
    pid_file.write_text("pid: none\n", encoding="utf-8")
    assert exp4167.read_pid(pid_file) is None

    _write_metrics(tmp_path, version=1, rows=[(1, 1, 0.7)])
    original_open = Path.open

    def disappearing_open(path: Path, *args: object, **kwargs: object) -> object:
        if path.name == "metrics.csv":
            raise FileNotFoundError
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", disappearing_open)
    assert exp4167.read_metrics(config.contiguous_run_dir).rows == []
    monkeypatch.setattr(Path, "open", original_open)

    _write_checkpoint(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=9400,
    )
    stopped_unfaithful = exp4167.build_artifact(
        config,
        process_checker=lambda _pid: _process(alive=False),
    )
    assert (
        stopped_unfaithful["honest_verdict"]
        == "complete: outerloop_training_stopped_unfaithful_val_0.7000"
    )

    bad = dict(artifact)
    del bad["checkpoint_mtime"]
    bad["honest_verdict"] = "not_terminal"
    bad["outerloop_train_alive"] = "yes"
    bad["baseline_faithful"] = "no"
    bad["current_val_exact_accuracy"] = 2
    bad["field_principles"] = {}
    errors = exp4167.artifact_schema_errors(bad)
    assert "missing required field checkpoint_mtime" in errors
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "outerloop_train_alive must be a bare bool" in errors
    assert "baseline_faithful must be a bare bool" in errors
    assert "current_val_exact_accuracy must be numeric between 0 and 1 or null" in errors
    assert "field_principles must include the required operator principles" in errors
    bad["checkpoint_mtime"] = 1
    assert "checkpoint_mtime must be an ISO string or null" in exp4167.artifact_schema_errors(bad)
    with pytest.raises(ValueError):
        exp4167.validate_artifact(bad)


def test_req_learn_4167_process_probe_uses_ps_etime_only() -> None:
    """REQ-LEARN-4167: process liveness uses the requested ps etime probe."""

    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "  12:34\n"
        stderr = ""

    def runner(command: list[str], **_kwargs: object) -> Result:
        calls.append(command)
        return Result()

    status = exp4167.check_pid_alive(999, runner=runner)

    assert status.alive is True
    assert status.etime == "12:34"
    assert calls == [["ps", "-o", "etime=", "-p", "999"]]
    source = (REPO / "python" / "carnot" / "experiment_4167_outerloop_training_monitor.py").read_text(
        encoding="utf-8"
    )
    assert "src/nn/train.py" not in source
    assert "pkill" not in source
