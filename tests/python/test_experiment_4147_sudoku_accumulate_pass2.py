"""Tests for Exp 4147 Sudoku Extreme pass2 continuation.

Spec refs: REQ-LEARN-4147, SCENARIO-LEARN-4147,
SCENARIO-LEARN-4147-BLOCKED-PASS1.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116
from carnot import experiment_4146_sudoku_accumulate_pass1_epochfix as exp4146
from carnot import experiment_4147_sudoku_accumulate_pass2 as exp4147


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _make_ready_repo(root: Path, *, max_epochs: int = 50000) -> None:
    trainer = root / "nano-trm" / "src" / "nn" / "train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# trainer fixture\n", encoding="utf-8")
    configs = root / "nano-trm" / "src" / "nn" / "configs"
    (configs / "experiment").mkdir(parents=True, exist_ok=True)
    (configs / "data").mkdir(parents=True, exist_ok=True)
    (configs / "experiment" / "trm_sudoku_extreme_1k_aug_1k.yaml").write_text(
        "\n".join(
            [
                "timekeeping:",
                f"  max_epochs: {max_epochs}",
                "  batch_size: 128",
                "trainer:",
                "  check_val_every_n_epoch: 100",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (configs / "data" / "sudoku_extreme_1k_aug1k.yaml").write_text(
        "data_dir: ./data/sudoku_extreme_1k_aug_1k\n", encoding="utf-8"
    )


def _write_checkpoint(path: Path, *, epoch: int, timer_train_s: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": epoch * 7,
            "callbacks": {"Timer": {"time_elapsed": {"train": timer_train_s}}},
            "state_dict": {"weight": torch.tensor([1.0])},
        },
        path,
    )
    return path


def _write_metrics(run_dir: Path, *, val: float, epoch: int) -> Path:
    metrics = run_dir / "csv" / "version_0" / "metrics.csv"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(
        "\n".join(
            [
                "epoch,step,train/lr,val/exact_accuracy",
                f"{epoch},{epoch * 7},9.9e-05,",
                f"{epoch},{epoch * 7},,{val}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return metrics


def _blocked_pass1(stable: Path) -> dict[str, object]:
    return {
        "honest_verdict": "blocked_noop_cap_not_confirmed_timer_elapsed",
        "seed_epoch": 6399,
        "post_epoch": 6399,
        "duration_s": 0.121,
        "val_exact_accuracy": None,
        "stable_checkpoint_path": str(stable),
        "diagnosis": {
            "checkpoint_epoch": 6399,
            "config_max_epochs": 50000,
            "max_epochs_cap_confirmed": False,
            "timer_train_elapsed_s": 3641.993,
            "max_time_s": 3600.0,
        },
    }


def _complete_pass1(stable: Path, *, epoch: int = 6402, val: float = 0.28) -> dict[str, object]:
    return {
        "honest_verdict": f"complete: epochfix_trained_seed_epoch=6399_post_epoch={epoch}_val={val:.4f}",
        "seed_epoch": 6399,
        "post_epoch": epoch,
        "duration_s": 130.0,
        "val_exact_accuracy": val,
        "stable_checkpoint_path": str(stable),
        "diagnosis": {
            "checkpoint_epoch": 6399,
            "config_max_epochs": 50000,
            "max_epochs_cap_confirmed": True,
        },
    }


def test_req_learn_4147_spec_declares_pass2_contract() -> None:
    """REQ-LEARN-4147: OpenSpec declares the blocked-pass1 pass2 contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4147" in spec
    assert "SCENARIO-LEARN-4147" in spec
    assert "SCENARIO-LEARN-4147-BLOCKED-PASS1" in spec
    assert "results/experiment_4147_sudoku_accumulate_pass2.json" in spec
    for field in exp4147.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_learn_4147_blocked_pass1_stops_before_training(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4147-BLOCKED-PASS1: upstream no-op forbids pass2 retrain."""

    _make_ready_repo(tmp_path)
    stable = _write_checkpoint(tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt", epoch=6399)
    pass1_path = tmp_path / "results" / exp4146.RESULT_FILENAME
    pass1_path.parent.mkdir(parents=True, exist_ok=True)
    pass1_path.write_text(json.dumps(_blocked_pass1(stable)), encoding="utf-8")
    trainer_calls = 0

    def forbidden_runner(_config: exp4147.Exp4147Config, _current_epoch: int) -> exp4116.ResumeRunResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("blocked pass1 must stop before native training")

    output = tmp_path / "results" / exp4147.RESULT_FILENAME
    artifact = exp4147.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        trainer_runner=forbidden_runner,
    )

    assert trainer_calls == 0
    assert artifact["honest_verdict"] == "blocked_pass1_noop_unresolved"
    assert artifact["val_exact_accuracy"] is None
    assert artifact["delta_vs_pass1"] is None
    assert artifact["post_epoch"] == 6399
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["native_trainer_launched"] is False
    assert artifact["command"] == []
    assert "checkpoint_epoch=6399" in artifact["blocked_cause"]
    assert "config_max_epochs=50000" in artifact["blocked_cause"]
    assert "timer_train_elapsed_s=3641.993" in artifact["blocked_cause"]
    for field, principle in exp4147.FIELD_PRINCIPLES.items():
        assert artifact["field_principles"][field] == principle
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_learn_4147_real_pass_reports_delta_and_command(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4147: pass2 must advance epoch and improve or plateau honestly."""

    _make_ready_repo(tmp_path)
    stable = _write_checkpoint(tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt", epoch=6402)
    pass1_path = tmp_path / "results" / exp4146.RESULT_FILENAME
    pass1_path.parent.mkdir(parents=True, exist_ok=True)
    pass1_path.write_text(json.dumps(_complete_pass1(stable, epoch=6402, val=0.28)), encoding="utf-8")
    calls = 0

    def fake_runner(config: exp4147.Exp4147Config, current_epoch: int) -> exp4116.ResumeRunResult:
        nonlocal calls
        calls += 1
        assert current_epoch == 6402
        post_epoch = current_epoch + 3
        _write_metrics(config.pass_run_dir(), val=0.31, epoch=post_epoch)
        _write_checkpoint(config.stable_checkpoint_path, epoch=post_epoch)
        metric = exp4107.ExactAccuracy(
            "val/exact_accuracy",
            0.31,
            config.pass_run_dir() / "csv" / "version_0" / "metrics.csv",
        )
        return exp4116.ResumeRunResult(
            return_code=0,
            stable_checkpoint_path=config.stable_checkpoint_path,
            checkpoint_reload_ok=True,
            checkpoint_reload_detail="loadable fixture",
            val_exact_accuracy=metric,
            cumulative_epochs=post_epoch + 1,
            duration_s=130.25,
            command=exp4147.build_train_command(config, current_epoch=current_epoch),
            stdout_tail=["trained fixture"],
            run_dir=config.pass_run_dir(),
        )

    artifact = exp4147.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4147.RESULT_FILENAME,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        trainer_runner=fake_runner,
    )

    assert calls == 1
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["val_exact_accuracy"] == 0.31
    assert artifact["delta_vs_pass1"] == 0.03
    assert artifact["post_epoch"] == 6405
    assert artifact["duration_s"] == 130.25
    assert artifact["honest_plateau"] is False
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["native_trainer_launched"] is True
    assert f"ckpt_path={stable}" in artifact["command"]
    assert "+trainer.max_epochs=9402" in artifact["command"]
    assert "+trainer.max_time=00:01:00:00" in artifact["command"]
    assert any("callbacks.model_checkpoint.dirpath=" in part for part in artifact["command"])
    assert any("+callbacks.exp4147_progress._target_=" in part for part in artifact["command"])


def test_req_learn_4147_schema_edges_and_plateau(tmp_path: Path) -> None:
    """REQ-LEARN-4147: schema rejects fake complete and accepts honest plateau."""

    _make_ready_repo(tmp_path)
    config = exp4147.Exp4147Config(repo_root=tmp_path)
    seed = exp4146.CheckpointState(True, "ok", 50, 350, None, 0.0)
    post = exp4146.CheckpointState(True, "ok", 55, 385, None, 0.0)
    run_result = exp4116.ResumeRunResult(
        return_code=0,
        stable_checkpoint_path=config.stable_checkpoint_path,
        checkpoint_reload_ok=True,
        checkpoint_reload_detail="ok",
        val_exact_accuracy=None,
        cumulative_epochs=56,
        duration_s=140.0,
        command=exp4147.build_train_command(config, current_epoch=50),
        stdout_tail=[],
        run_dir=config.pass_run_dir(),
    )
    plateau = exp4147.build_result_artifact(
        run_config=config,
        pass1_artifact=_complete_pass1(config.stable_checkpoint_path, epoch=50, val=0.33),
        seed_state=seed,
        post_state=post,
        run_result=run_result,
        val_exact_accuracy=0.33,
        val_metrics_path=config.pass_run_dir() / "metrics.csv",
    )

    assert plateau["honest_plateau"] is True
    assert plateau["delta_vs_pass1"] == 0.0
    assert plateau["acceptance_gate_passed"] is True
    assert exp4147.artifact_schema_errors(plateau) == []

    blocked = exp4147.build_blocked_pass1_artifact(
        run_config=config,
        pass1_artifact={},
        preconditions_checked=[exp4107.PreconditionCheck("uv", True, "ok")],
        duration_s=0.5,
    )
    assert blocked["honest_verdict"] == "blocked_pass1_noop_unresolved"
    assert blocked["post_epoch"] is None
    assert exp4147.pass1_has_real_training({}) is False
    assert exp4147.pass1_has_real_training(_complete_pass1(config.stable_checkpoint_path)) is True

    invalid = dict(blocked)
    invalid.update(
        {
            "honest_verdict": "pending",
            "val_exact_accuracy": 2.0,
            "delta_vs_pass1": True,
            "post_epoch": "50",
            "duration_s": [1.0],
            "acceptance_gate_passed": "yes",
            "field_principles": {"honest_verdict": "wrong"},
        }
    )
    errors = exp4147.artifact_schema_errors(invalid)
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "val_exact_accuracy must be numeric between 0 and 1 or null" in errors
    assert "delta_vs_pass1 must be numeric or null" in errors
    assert "post_epoch must be an int or null" in errors
    assert "duration_s must be a scalar bounded number below 86400" in errors
    assert "acceptance_gate_passed must be a bare bool" in errors
    assert "field_principles must include the required operator principles" in errors

    fake_complete = dict(blocked)
    fake_complete["honest_verdict"] = "complete: fake"
    assert "complete/plateau verdict requires duration>120, epoch advance, real val, and delta/plateau proof" in (
        exp4147.artifact_schema_errors(fake_complete)
    )
    assert "missing required field honest_verdict" in exp4147.artifact_schema_errors({})
    exp4147.write_result_artifact(tmp_path / "plateau.json", plateau)
    assert json.loads((tmp_path / "plateau.json").read_text(encoding="utf-8")) == plateau
    try:
        exp4147.validate_artifact(invalid)
    except ValueError as exc:
        assert "honest_verdict must be terminal-prefixed" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("validate_artifact should reject malformed artifacts")
