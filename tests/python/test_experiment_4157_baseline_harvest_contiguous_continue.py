"""Tests for Exp 4157 contiguous Sudoku baseline harvest.

Spec refs: REQ-LEARN-4157, SCENARIO-LEARN-4157-LIVE,
SCENARIO-LEARN-4157-FAITHFUL, SCENARIO-LEARN-4157-CONTINUE.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4157_baseline_harvest_contiguous_continue as exp4157


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _make_ready_repo(root: Path) -> Path:
    trainer = root / "nano-trm" / "src" / "nn" / "train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# trainer fixture\n", encoding="utf-8")
    return trainer


def _write_checkpoint(path: Path, *, epoch: int, manual_lr_step: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "nano_trm_manual_lr_step": manual_lr_step,
            "global_step": 0,
            "state_dict": {"weight": torch.tensor([1.0])},
        },
        path,
    )
    return path


def _write_metrics(root: Path, *, version: int, rows: list[tuple[int, int, float | str | None]]) -> Path:
    metrics = root / "results" / "trm_runs" / "contiguous_run_hydra" / "csv" / f"version_{version}" / "metrics.csv"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "step", "train/lr", "val/exact_accuracy"])
        for epoch, step, val in rows:
            writer.writerow([epoch, step, "", "" if val is None else val])
    return metrics


def _write_pid(root: Path, pid: str = "1234\n") -> Path:
    pid_path = root / "results" / "trm_runs" / "contiguous_run.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(pid, encoding="utf-8")
    return pid_path


def _live_liveness(_config: exp4157.Exp4157Config, _snapshot: exp4157.MetricsSnapshot) -> exp4157.RunLiveness:
    return exp4157.RunLiveness(
        pid=1234,
        process_alive=True,
        csv_advancing=True,
        run_alive=True,
        detail="fixture live and advancing",
    )


def _dead_liveness(_config: exp4157.Exp4157Config, _snapshot: exp4157.MetricsSnapshot) -> exp4157.RunLiveness:
    return exp4157.RunLiveness(
        pid=1234,
        process_alive=False,
        csv_advancing=False,
        run_alive=False,
        detail="fixture stale pid",
    )


def test_req_learn_4157_spec_declares_contiguous_harvest_contract() -> None:
    """REQ-LEARN-4157: OpenSpec declares the 4157 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4157" in spec
    assert "SCENARIO-LEARN-4157-LIVE" in spec
    assert "SCENARIO-LEARN-4157-FAITHFUL" in spec
    assert "SCENARIO-LEARN-4157-CONTINUE" in spec
    assert "results/experiment_4157_baseline_harvest_contiguous_continue.json" in spec
    for field in exp4157.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert field in exp4157.FIELD_PRINCIPLES


def test_req_learn_4157_parses_csv_checkpoint_and_command(tmp_path: Path) -> None:
    """REQ-LEARN-4157: CSV and checkpoint reads are lightweight and deterministic."""

    _make_ready_repo(tmp_path)
    stable = _write_checkpoint(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=6399,
        manual_lr_step=19006,
    )
    _write_metrics(tmp_path, version=0, rows=[(6399, 19006, 0.278), (6499, 19106, "")])
    best_path = _write_metrics(tmp_path, version=2, rows=[(6599, 19206, 0.42), (6699, 19306, 0.501)])
    _write_pid(tmp_path, "pid=1234\n")
    config = exp4157.Exp4157Config(repo_root=tmp_path)

    assert config.trainer_path == tmp_path / "nano-trm" / "src" / "nn" / "train.py"
    snapshot = exp4157.read_contiguous_metrics(config.contiguous_run_dir)
    assert snapshot.current_val == pytest.approx(0.501)
    assert snapshot.max_val == pytest.approx(0.501)
    assert snapshot.latest_row.metrics_path == best_path
    assert snapshot.latest_row.epoch == 6699
    assert [row.val_exact_accuracy for row in snapshot.rows] == [0.278, 0.42, 0.501]
    assert snapshot.to_trajectory()[-1]["delta_vs_previous"] == pytest.approx(0.081)
    assert exp4157.estimate_passes_to_085(snapshot)["estimated_additional_val_intervals"] == 5

    scalars = exp4157.read_checkpoint_scalars(stable)
    assert scalars.load_ok is True
    assert scalars.epoch == 6399
    assert scalars.manual_lr_step == 19006
    assert exp4157.MetricsSnapshot([]).current_val is None
    assert exp4157.MetricsSnapshot([]).max_val is None
    assert exp4157.MetricsSnapshot([]).latest_signature is None
    assert exp4157.estimate_passes_to_085(exp4157.MetricsSnapshot([]))["basis"] == "missing_current_val"
    assert exp4157.estimate_passes_to_085(
        exp4157.MetricsSnapshot(
            [
                exp4157.MetricsRow(Path("m.csv"), 0, 2, 1, 1, 0.5),
                exp4157.MetricsRow(Path("m.csv"), 0, 3, 2, 2, 0.49),
            ]
        )
    )["basis"] == "no_positive_val_delta"
    assert exp4157.LaunchResult(process_pid=1, return_code=None, stdout_tail=["x"]).to_dict()["process_pid"] == 1
    assert exp4157._float_or_none(True) is None
    assert exp4157._float_or_none("not-a-number") is None
    assert exp4157._version_number(Path("csv/version_bad/metrics.csv")) == -1
    assert exp4157._version_number(Path("csv/not_version/metrics.csv")) == -1

    command = exp4157.build_train_command(config)
    assert command[:4] == ["uv", "run", "python", "src/nn/train.py"]
    assert f"hydra.run.dir={tmp_path / 'results' / 'trm_runs' / 'contiguous_run_hydra'}" in command
    assert f"ckpt_path={stable}" in command
    assert "+trainer.max_time=00:11:30:00" in command
    assert not any("trainer.max_epochs" in part for part in command)
    env = exp4157.build_train_env(config)
    assert env["DISABLE_COMPILE"] == "1"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["WANDB_MODE"] == "disabled"


def test_scenario_learn_4157_live_run_records_only(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4157-LIVE: live owner prevents competing launch."""

    _make_ready_repo(tmp_path)
    stable = _write_checkpoint(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=6399,
        manual_lr_step=19006,
    )
    _write_metrics(tmp_path, version=4, rows=[(8499, 21104, 0.4906), (8599, 21204, 0.501)])
    _write_pid(tmp_path)
    trainer_calls = 0

    def forbidden_runner(
        _config: exp4157.Exp4157Config,
        _seed_checkpoint: exp4157.CheckpointScalars,
        _seed_metrics: exp4157.MetricsSnapshot,
    ) -> exp4157.LaunchResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("live run must be record-only")

    artifact = exp4157.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4157.RESULT_FILENAME,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        liveness_checker=_live_liveness,
        trainer_runner=forbidden_runner,
    )

    assert trainer_calls == 0
    assert artifact["honest_verdict"] == "complete: baseline_advancing_contiguous_run_live_val_0.5010"
    assert artifact["current_val"] == 0.501
    assert artifact["max_val"] == 0.501
    assert artifact["baseline_faithful"] is False
    assert artifact["run_alive"] is True
    assert artifact["manual_lr_step"] == 19006
    assert artifact["stable_checkpoint_path"] == str(stable)
    assert artifact["native_trainer_launched"] is False
    assert artifact["acceptance_gate_passed"] is True
    assert json.loads((tmp_path / "results" / exp4157.RESULT_FILENAME).read_text(encoding="utf-8")) == artifact


def test_scenario_learn_4157_dead_faithful_confirms_without_training(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4157-FAITHFUL: dead run at val>=0.85 is already usable."""

    _make_ready_repo(tmp_path)
    stable = _write_checkpoint(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=9000,
        manual_lr_step=21400,
    )
    _write_metrics(tmp_path, version=7, rows=[(8899, 21504, 0.84), (8999, 21604, 0.851)])
    _write_pid(tmp_path)

    artifact = exp4157.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4157.RESULT_FILENAME,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        liveness_checker=_dead_liveness,
        trainer_runner=lambda *_args: (_ for _ in ()).throw(AssertionError("faithful run must not train")),
    )

    assert artifact["honest_verdict"] == "complete: baseline_faithful_val_0.8510"
    assert artifact["current_val"] == 0.851
    assert artifact["baseline_faithful"] is True
    assert artifact["run_alive"] is False
    assert artifact["manual_lr_step"] == 21400
    assert artifact["stable_checkpoint_path"] == str(stable)
    assert artifact["native_trainer_launched"] is False
    assert artifact["estimated_passes_to_085"]["estimated_additional_val_intervals"] == 0


def test_scenario_learn_4157_dead_below_gate_launches_once_and_checks_step(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4157-CONTINUE: launched run needs step advance and new val."""

    _make_ready_repo(tmp_path)
    _write_checkpoint(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=6399,
        manual_lr_step=19006,
    )
    _write_metrics(tmp_path, version=4, rows=[(8499, 21104, 0.4906), (8599, 21204, 0.501)])
    _write_pid(tmp_path)
    calls = 0

    def fake_runner(
        config: exp4157.Exp4157Config,
        seed_checkpoint: exp4157.CheckpointScalars,
        seed_metrics: exp4157.MetricsSnapshot,
    ) -> exp4157.LaunchResult:
        nonlocal calls
        calls += 1
        assert seed_checkpoint.manual_lr_step == 19006
        assert seed_metrics.current_val == pytest.approx(0.501)
        _write_metrics(tmp_path, version=5, rows=[(8699, 21304, 0.521)])
        _write_checkpoint(config.stable_checkpoint_path, epoch=8699, manual_lr_step=21304)
        return exp4157.LaunchResult(process_pid=4321, return_code=None, stdout_tail=["progress fixture"])

    artifact = exp4157.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4157.RESULT_FILENAME,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        liveness_checker=_dead_liveness,
        trainer_runner=fake_runner,
    )

    assert calls == 1
    assert artifact["honest_verdict"] == "complete: baseline_contiguous_run_relaunched_val_0.5210"
    assert artifact["current_val"] == 0.521
    assert artifact["max_val"] == 0.521
    assert artifact["run_alive"] is True
    assert artifact["baseline_faithful"] is False
    assert artifact["manual_lr_step"] == 21304
    assert artifact["native_trainer_launched"] is True
    assert artifact["task_launched_run"]["manual_lr_step_advanced"] is True
    assert artifact["task_launched_run"]["new_val_row_written"] is True
    assert artifact["task_launched_run"]["process_pid"] == 4321
    assert artifact["acceptance_gate_passed"] is True


def test_scenario_learn_4157_launch_noop_reports_step_blocker(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4157-CONTINUE: stalled manual step is blocked, not complete."""

    _make_ready_repo(tmp_path)
    _write_checkpoint(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=6399,
        manual_lr_step=19006,
    )
    _write_metrics(tmp_path, version=4, rows=[(8599, 21204, 0.501)])
    _write_pid(tmp_path)

    def noop_runner(
        _config: exp4157.Exp4157Config,
        _seed_checkpoint: exp4157.CheckpointScalars,
        _seed_metrics: exp4157.MetricsSnapshot,
    ) -> exp4157.LaunchResult:
        return exp4157.LaunchResult(process_pid=None, return_code=0, stdout_tail=["noop fixture"])

    artifact = exp4157.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4157.RESULT_FILENAME,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        liveness_checker=_dead_liveness,
        trainer_runner=noop_runner,
    )

    assert artifact["honest_verdict"] == "blocked_noop_step_unchanged"
    assert artifact["current_val"] == 0.501
    assert artifact["manual_lr_step"] == 19006
    assert artifact["task_launched_run"]["manual_lr_step_advanced"] is False
    assert artifact["task_launched_run"]["new_val_row_written"] is False
    assert "manual_lr_step did not advance" in artifact["blocked_cause"]
    assert artifact["acceptance_gate_passed"] is True


def test_req_learn_4157_liveness_preconditions_and_schema_edges(tmp_path: Path) -> None:
    """REQ-LEARN-4157: liveness, blockers, and schema checks fail closed."""

    _make_ready_repo(tmp_path)
    _write_checkpoint(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=6399,
        manual_lr_step=19006,
    )
    _write_metrics(tmp_path, version=0, rows=[(6399, 19006, 0.278)])
    _write_pid(tmp_path, "1234abc\n")
    config = exp4157.Exp4157Config(repo_root=tmp_path)
    before = exp4157.read_contiguous_metrics(config.contiguous_run_dir)
    later_path = _write_metrics(tmp_path, version=1, rows=[(6499, 19106, 0.3)])

    liveness = exp4157.detect_liveness(
        config,
        before,
        process_checker=lambda _pid: exp4157.ProcessStatus(alive=True, detail="ps fixture"),
        metrics_reader=lambda _path: exp4157.read_contiguous_metrics(config.contiguous_run_dir),
        sleeper=lambda _seconds: None,
        liveness_probe_s=0,
    )
    assert liveness.pid == 1234
    assert liveness.process_alive is True
    assert liveness.csv_advancing is True
    assert liveness.run_alive is True
    assert str(later_path) in liveness.detail

    missing_ckpt = exp4157.read_checkpoint_scalars(tmp_path / "missing.ckpt")
    assert missing_ckpt.load_ok is False
    assert missing_ckpt.manual_lr_step is None
    corrupt_ckpt = tmp_path / "corrupt.ckpt"
    corrupt_ckpt.write_text("not a checkpoint", encoding="utf-8")
    assert exp4157.read_checkpoint_scalars(corrupt_ckpt).load_ok is False
    list_ckpt = tmp_path / "list.ckpt"
    torch.save([1, 2, 3], list_ckpt)
    assert "unexpected checkpoint payload" in exp4157.read_checkpoint_scalars(list_ckpt).detail
    missing_pid_config = exp4157.Exp4157Config(repo_root=tmp_path, pid_path=tmp_path / "missing.pid")
    missing_pid = exp4157.detect_liveness(
        missing_pid_config,
        before,
        process_checker=lambda _pid: exp4157.ProcessStatus(alive=True, detail="unused"),
    )
    assert missing_pid.run_alive is False
    assert missing_pid.pid is None
    dead_pid = exp4157.detect_liveness(
        config,
        before,
        process_checker=lambda _pid: exp4157.ProcessStatus(alive=False, detail="dead fixture"),
    )
    assert dead_pid.run_alive is False
    assert dead_pid.pid == 1234
    slept: list[float] = []
    probed = exp4157.detect_liveness(
        config,
        before,
        process_checker=lambda _pid: exp4157.ProcessStatus(alive=True, detail="ps fixture"),
        metrics_reader=lambda _path: exp4157.read_contiguous_metrics(config.contiguous_run_dir),
        sleeper=slept.append,
        liveness_probe_s=0.25,
    )
    assert probed.run_alive is True
    assert slept == [0.25]

    blocked = exp4157.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4157.RESULT_FILENAME,
        uv_resolver=lambda _name: None,
        cuda_checker=lambda: (True, "cuda fixture"),
        liveness_checker=_dead_liveness,
    )
    assert blocked["honest_verdict"] == "blocked_uv"
    assert blocked["acceptance_gate_passed"] is False
    assert any(check["resource"] == "uv" and not check["available"] for check in blocked["preconditions_checked"])
    trainer_block = exp4157.check_preconditions(
        exp4157.Exp4157Config(repo_root=tmp_path / "no_trainer"),
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
    )
    assert trainer_block[1] == "blocked_trainer"
    cuda_block = exp4157.check_preconditions(
        config,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (False, "no cuda"),
    )
    assert cuda_block[1] == "blocked_cuda"
    stable_block_config = exp4157.Exp4157Config(
        repo_root=tmp_path,
        stable_dir=tmp_path / "results" / "trm_runs" / "missing_stable",
    )
    stable_block = exp4157.check_preconditions(
        stable_block_config,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
    )
    assert stable_block[1] == "blocked_stable_checkpoint"

    errors = exp4157.artifact_schema_errors({})
    assert "missing required field honest_verdict" in errors
    bad = dict(blocked)
    bad.update(
        {
            "honest_verdict": "pending",
            "current_val": 2.0,
            "max_val": "0.5",
            "baseline_faithful": "no",
            "run_alive": "no",
            "manual_lr_step": "19006",
            "val_trajectory": "bad",
            "stable_checkpoint_path": "",
            "field_principles": {"honest_verdict": "wrong"},
        }
    )
    bad_errors = exp4157.artifact_schema_errors(bad)
    assert "honest_verdict must be terminal-prefixed" in bad_errors
    assert "current_val must be numeric between 0 and 1 or null" in bad_errors
    assert "max_val must be numeric between 0 and 1 or null" in bad_errors
    assert "baseline_faithful must be a bare bool" in bad_errors
    assert "run_alive must be a bare bool" in bad_errors
    assert "manual_lr_step must be an integer or null" in bad_errors
    assert "val_trajectory must be a list" in bad_errors
    assert "stable_checkpoint_path must be a non-empty string" in bad_errors
    assert "field_principles must include the required operator principles" in bad_errors
    bad_launch = dict(blocked)
    bad_launch["native_trainer_launched"] = True
    bad_launch["task_launched_run"] = None
    assert "task_launched_run must describe launched-run progress" in exp4157.artifact_schema_errors(bad_launch)
    with pytest.raises(ValueError, match="honest_verdict must be terminal-prefixed"):
        exp4157.validate_artifact(bad)

    check = exp4107.PreconditionCheck("fixture", True, "ok")
    assert exp4157._checks_to_dicts([check]) == [{"resource": "fixture", "available": True, "detail": "ok"}]
    assert exp4157._acceptance_gate({"honest_verdict": "complete: x", "current_val": None, "run_alive": True}) is False
    assert exp4157._acceptance_gate(
        {"honest_verdict": "complete: x", "current_val": 0.5, "run_alive": True, "baseline_faithful": "no"}
    ) is False
    assert exp4157._acceptance_gate(
        {"honest_verdict": "blocked_other", "current_val": 0.5, "run_alive": False, "baseline_faithful": False}
    ) is False

    bad_load_root = tmp_path / "bad_load"
    _make_ready_repo(bad_load_root)
    bad_stable = bad_load_root / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    bad_stable.parent.mkdir(parents=True, exist_ok=True)
    bad_stable.write_text("not a checkpoint", encoding="utf-8")
    _write_metrics(bad_load_root, version=0, rows=[(1, 1, 0.2)])
    bad_load = exp4157.run_experiment(
        repo_root=bad_load_root,
        output_path=bad_load_root / "results" / exp4157.RESULT_FILENAME,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        liveness_checker=_dead_liveness,
    )
    assert bad_load["honest_verdict"] == "blocked_stable_checkpoint_load"

    missing_metrics_root = tmp_path / "missing_metrics"
    _make_ready_repo(missing_metrics_root)
    _write_checkpoint(
        missing_metrics_root / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=1,
        manual_lr_step=1,
    )
    missing_metrics = exp4157.run_experiment(
        repo_root=missing_metrics_root,
        output_path=missing_metrics_root / "results" / exp4157.RESULT_FILENAME,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        liveness_checker=_dead_liveness,
    )
    assert missing_metrics["honest_verdict"] == "blocked_contiguous_metrics_missing"

    seed = exp4157.CheckpointScalars(True, "seed", 1, 10)
    post_step_only = exp4157.CheckpointScalars(True, "post", 2, 11)
    one_row = exp4157.MetricsSnapshot([exp4157.MetricsRow(Path("m.csv"), 0, 2, 1, 1, 0.5)])
    step_only_artifact = exp4157.build_launched_artifact(
        config=config,
        seed_checkpoint=seed,
        post_checkpoint=post_step_only,
        seed_metrics=one_row,
        post_metrics=one_row,
        launch_result=exp4157.LaunchResult(process_pid=None, return_code=0, stdout_tail=[]),
        preconditions_checked=[],
        duration_s=1.0,
    )
    assert step_only_artifact["honest_verdict"] == "blocked_noop_step_unchanged"
    assert "no new val row" in step_only_artifact["blocked_cause"]

    faithful_post = exp4157.MetricsSnapshot(
        [
            exp4157.MetricsRow(Path("m.csv"), 0, 2, 1, 1, 0.5),
            exp4157.MetricsRow(Path("m.csv"), 0, 3, 2, 2, 0.86),
        ]
    )
    faithful_launch = exp4157.build_launched_artifact(
        config=config,
        seed_checkpoint=seed,
        post_checkpoint=post_step_only,
        seed_metrics=one_row,
        post_metrics=faithful_post,
        launch_result=exp4157.LaunchResult(process_pid=44, return_code=None, stdout_tail=[]),
        preconditions_checked=[],
        duration_s=1.0,
    )
    assert faithful_launch["honest_verdict"] == "complete: baseline_faithful_val_0.8600"
