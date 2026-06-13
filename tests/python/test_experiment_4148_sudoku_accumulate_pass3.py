"""Tests for Exp 4148 Sudoku Extreme pass3 continuation.

Spec refs: REQ-LEARN-4148, SCENARIO-LEARN-4148,
SCENARIO-LEARN-4148-BLOCKED-PASS2, SCENARIO-LEARN-4148-EARLY-CONVERGED.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116
from carnot import experiment_4146_sudoku_accumulate_pass1_epochfix as exp4146
from carnot import experiment_4147_sudoku_accumulate_pass2 as exp4147
from carnot import experiment_4148_sudoku_accumulate_pass3 as exp4148


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _make_ready_repo(root: Path) -> None:
    trainer = root / "nano-trm" / "src" / "nn" / "train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# trainer fixture\n", encoding="utf-8")


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


def _write_checkpoint_without_epoch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": 0,
            "callbacks": {"Timer": {"time_elapsed": {"train": 0.0}}},
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


def _blocked_pass2(stable: Path) -> dict[str, object]:
    return {
        "honest_verdict": "blocked_pass1_noop_unresolved",
        "post_epoch": 6399,
        "duration_s": 0.125,
        "val_exact_accuracy": None,
        "delta_vs_pass1": None,
        "stable_checkpoint_path": str(stable),
        "pass1_honest_verdict": "blocked_noop_cap_not_confirmed_timer_elapsed",
        "blocked_cause": (
            "pass1 verdict=blocked_noop_cap_not_confirmed_timer_elapsed; "
            "checkpoint_epoch=6399; config_max_epochs=50000; "
            "timer_train_elapsed_s=3641.993"
        ),
    }


def _complete_pass2(stable: Path, *, epoch: int = 6405, val: float = 0.31) -> dict[str, object]:
    return {
        "honest_verdict": f"complete: pass2_trained_post_epoch={epoch}_val={val:.4f}_delta=0.0300",
        "post_epoch": epoch,
        "pass1_post_epoch": 6402,
        "duration_s": 130.0,
        "val_exact_accuracy": val,
        "delta_vs_pass1": 0.03,
        "stable_checkpoint_path": str(stable),
    }


def test_req_learn_4148_spec_declares_pass3_contract() -> None:
    """REQ-LEARN-4148: OpenSpec declares the pass3 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4148" in spec
    assert "SCENARIO-LEARN-4148" in spec
    assert "SCENARIO-LEARN-4148-BLOCKED-PASS2" in spec
    assert "SCENARIO-LEARN-4148-EARLY-CONVERGED" in spec
    assert "results/experiment_4148_sudoku_accumulate_pass3.json" in spec
    for field in exp4148.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_learn_4148_blocked_pass2_stops_before_training(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4148-BLOCKED-PASS2: upstream no-op forbids pass3 retrain."""

    _make_ready_repo(tmp_path)
    stable = _write_checkpoint(tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt", epoch=6399)
    pass2_path = tmp_path / "results" / exp4147.RESULT_FILENAME
    pass2_path.parent.mkdir(parents=True, exist_ok=True)
    pass2_path.write_text(json.dumps(_blocked_pass2(stable)), encoding="utf-8")
    trainer_calls = 0

    def forbidden_runner(_config: exp4148.Exp4148Config, _current_epoch: int) -> exp4116.ResumeRunResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("blocked pass2 must stop before native training")

    output = tmp_path / "results" / exp4148.RESULT_FILENAME
    artifact = exp4148.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        trainer_runner=forbidden_runner,
    )

    assert trainer_calls == 0
    assert artifact["honest_verdict"] == "blocked_pass2_noop_unresolved"
    assert artifact["val_exact_accuracy"] is None
    assert artifact["delta_vs_pass2"] is None
    assert artifact["post_epoch"] == 6399
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["native_trainer_launched"] is False
    assert artifact["command"] == []
    assert "pass2 verdict=blocked_pass1_noop_unresolved" in artifact["blocked_cause"]
    assert "pass1_honest_verdict=blocked_noop_cap_not_confirmed_timer_elapsed" in artifact["blocked_cause"]
    assert "timer_train_elapsed_s=3641.993" in artifact["blocked_cause"]
    for field, principle in exp4148.FIELD_PRINCIPLES.items():
        assert artifact["field_principles"][field] == principle
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_learn_4148_early_converged_skips_training(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4148-EARLY-CONVERGED: pass2 target hit stops pass3 cleanly."""

    _make_ready_repo(tmp_path)
    stable = _write_checkpoint(tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt", epoch=6410)
    pass2_path = tmp_path / "results" / exp4147.RESULT_FILENAME
    pass2_path.parent.mkdir(parents=True, exist_ok=True)
    pass2_path.write_text(json.dumps(_complete_pass2(stable, epoch=6410, val=0.872)), encoding="utf-8")
    trainer_calls = 0

    def forbidden_runner(_config: exp4148.Exp4148Config, _current_epoch: int) -> exp4116.ResumeRunResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("early convergence must skip native training")

    artifact = exp4148.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4148.RESULT_FILENAME,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        trainer_runner=forbidden_runner,
    )

    assert trainer_calls == 0
    assert artifact["honest_verdict"].startswith("complete: early_converged")
    assert artifact["val_exact_accuracy"] == 0.872
    assert artifact["delta_vs_pass2"] == 0.0
    assert artifact["post_epoch"] == 6410
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["native_trainer_launched"] is False
    assert artifact["command"] == []


def test_scenario_learn_4148_real_pass_reports_delta_and_command(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4148: pass3 must advance epoch and improve or plateau honestly."""

    _make_ready_repo(tmp_path)
    stable = _write_checkpoint_without_epoch(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    )
    pass2_path = tmp_path / "results" / exp4147.RESULT_FILENAME
    pass2_path.parent.mkdir(parents=True, exist_ok=True)
    pass2_path.write_text(json.dumps(_complete_pass2(stable, epoch=6405, val=0.31)), encoding="utf-8")
    calls = 0

    def fake_runner(config: exp4148.Exp4148Config, current_epoch: int) -> exp4116.ResumeRunResult:
        nonlocal calls
        calls += 1
        assert current_epoch == 6405
        post_epoch = current_epoch + 4
        _write_metrics(config.pass_run_dir(), val=0.34, epoch=post_epoch)
        _write_checkpoint(config.stable_checkpoint_path, epoch=post_epoch)
        metric = exp4107.ExactAccuracy(
            "val/exact_accuracy",
            0.34,
            config.pass_run_dir() / "csv" / "version_0" / "metrics.csv",
        )
        return exp4116.ResumeRunResult(
            return_code=0,
            stable_checkpoint_path=config.stable_checkpoint_path,
            checkpoint_reload_ok=True,
            checkpoint_reload_detail="loadable fixture",
            val_exact_accuracy=metric,
            cumulative_epochs=post_epoch + 1,
            duration_s=131.5,
            command=exp4148.build_train_command(config, current_epoch=current_epoch),
            stdout_tail=["trained fixture"],
            run_dir=config.pass_run_dir(),
        )

    artifact = exp4148.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4148.RESULT_FILENAME,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        trainer_runner=fake_runner,
    )

    assert calls == 1
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["val_exact_accuracy"] == 0.34
    assert artifact["delta_vs_pass2"] == 0.03
    assert artifact["post_epoch"] == 6409
    assert artifact["duration_s"] == 131.5
    assert artifact["honest_plateau"] is False
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["native_trainer_launched"] is True
    assert f"ckpt_path={stable}" in artifact["command"]
    assert "+trainer.max_epochs=9405" in artifact["command"]
    assert "+trainer.max_time=00:01:00:00" in artifact["command"]
    assert any("callbacks.model_checkpoint.dirpath=" in part for part in artifact["command"])
    assert any("+callbacks.exp4148_progress._target_=" in part for part in artifact["command"])


def test_req_learn_4148_schema_edges_plateau_and_loaders(tmp_path: Path) -> None:
    """REQ-LEARN-4148: schema rejects fake complete and accepts honest plateau."""

    _make_ready_repo(tmp_path)
    config = exp4148.Exp4148Config(repo_root=tmp_path)
    assert config.trainer_path == tmp_path / "nano-trm" / "src" / "nn" / "train.py"
    assert config.to_4127_config().stable_checkpoint_path == config.stable_checkpoint_path
    assert exp4148.build_train_env(config)["WANDB_DISABLED"] == "true"
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
        command=exp4148.build_train_command(config, current_epoch=50),
        stdout_tail=[],
        run_dir=config.pass_run_dir(),
    )
    plateau = exp4148.build_result_artifact(
        run_config=config,
        pass2_artifact=_complete_pass2(config.stable_checkpoint_path, epoch=50, val=0.33),
        seed_state=seed,
        post_state=post,
        run_result=run_result,
        val_exact_accuracy=0.33,
        val_metrics_path=config.pass_run_dir() / "metrics.csv",
    )

    assert plateau["honest_plateau"] is True
    assert plateau["delta_vs_pass2"] == 0.0
    assert plateau["acceptance_gate_passed"] is True
    assert exp4148.artifact_schema_errors(plateau) == []

    blocked = exp4148.build_blocked_pass2_artifact(
        run_config=config,
        pass2_artifact={},
        preconditions_checked=[exp4107.PreconditionCheck("uv", True, "ok")],
        duration_s=0.5,
    )
    assert blocked["honest_verdict"] == "blocked_pass2_noop_unresolved"
    assert blocked["post_epoch"] is None
    assert exp4148._acceptance_gate(blocked) is True
    assert exp4148.pass2_has_real_training({}) is False
    assert exp4148.pass2_has_real_training(_complete_pass2(config.stable_checkpoint_path)) is True
    assert exp4148.pass2_is_early_converged(_complete_pass2(config.stable_checkpoint_path, val=0.87)) is True

    invalid = dict(blocked)
    invalid.update(
        {
            "honest_verdict": "pending",
            "val_exact_accuracy": 2.0,
            "delta_vs_pass2": True,
            "post_epoch": "50",
            "duration_s": [1.0],
            "acceptance_gate_passed": "yes",
            "field_principles": {"honest_verdict": "wrong"},
        }
    )
    errors = exp4148.artifact_schema_errors(invalid)
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "val_exact_accuracy must be numeric between 0 and 1 or null" in errors
    assert "delta_vs_pass2 must be numeric or null" in errors
    assert "post_epoch must be an int or null" in errors
    assert "duration_s must be a scalar bounded number below 86400" in errors
    assert "acceptance_gate_passed must be a bare bool" in errors
    assert "field_principles must include the required operator principles" in errors

    fake_complete = dict(blocked)
    fake_complete["honest_verdict"] = "complete: fake"
    assert "complete/plateau verdict requires duration>120, epoch advance, real val, or early convergence" in (
        exp4148.artifact_schema_errors(fake_complete)
    )
    assert "missing required field honest_verdict" in exp4148.artifact_schema_errors({})
    assert exp4148._verdict_for_result(
        duration_s=1.0,
        pass2_epoch=1,
        post_epoch=2,
        val_exact_accuracy=0.1,
        delta_vs_pass2=0.1,
        honest_plateau=False,
    ) == "blocked_noop_duration_too_short"
    assert exp4148._verdict_for_result(
        duration_s=130.0,
        pass2_epoch=2,
        post_epoch=2,
        val_exact_accuracy=0.1,
        delta_vs_pass2=0.1,
        honest_plateau=False,
    ) == "blocked_noop_epoch_not_advanced"
    assert exp4148._verdict_for_result(
        duration_s=130.0,
        pass2_epoch=1,
        post_epoch=2,
        val_exact_accuracy=None,
        delta_vs_pass2=0.1,
        honest_plateau=False,
    ) == "blocked_noop_missing_val_exact_accuracy"
    assert exp4148._verdict_for_result(
        duration_s=130.0,
        pass2_epoch=1,
        post_epoch=2,
        val_exact_accuracy=0.1,
        delta_vs_pass2=None,
        honest_plateau=False,
    ) == "blocked_noop_missing_delta_vs_pass2"
    assert exp4148._verdict_for_result(
        duration_s=130.0,
        pass2_epoch=1,
        post_epoch=2,
        val_exact_accuracy=0.1,
        delta_vs_pass2=0.0,
        honest_plateau=False,
    ) == "blocked_noop_nonpositive_delta_without_plateau"
    exp4148.write_result_artifact(tmp_path / "plateau.json", plateau)
    assert json.loads((tmp_path / "plateau.json").read_text(encoding="utf-8")) == plateau
    try:
        exp4148.validate_artifact(invalid)
    except ValueError as exc:
        assert "honest_verdict must be terminal-prefixed" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("validate_artifact should reject malformed artifacts")

    missing = exp4148.load_pass2_artifact(tmp_path / "missing.json")
    assert "missing pass2 artifact" in missing["load_error"]
    assert "missing pass2 artifact" in exp4148.summarize_pass2_blocker(missing)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert "JSONDecodeError" in exp4148.load_pass2_artifact(bad_json)["load_error"]
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert "unexpected pass2 payload" in exp4148.load_pass2_artifact(list_payload)["load_error"]


def test_req_learn_4148_runtime_blocker_preserves_pass2_context(tmp_path: Path) -> None:
    """REQ-LEARN-4148: runtime blockers write schema-valid artifacts without training."""

    _make_ready_repo(tmp_path)
    stable = _write_checkpoint(tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt", epoch=6405)
    pass2_path = tmp_path / "results" / exp4147.RESULT_FILENAME
    pass2_path.parent.mkdir(parents=True, exist_ok=True)
    pass2_path.write_text(json.dumps(_complete_pass2(stable, epoch=6405, val=0.31)), encoding="utf-8")
    trainer_calls = 0

    def forbidden_runner(_config: exp4148.Exp4148Config, _current_epoch: int) -> exp4116.ResumeRunResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("runtime blocker must stop before native training")

    artifact = exp4148.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4148.RESULT_FILENAME,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (False, "cuda unavailable fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        trainer_runner=forbidden_runner,
    )

    assert trainer_calls == 0
    assert artifact["honest_verdict"] == "blocked_cuda_unavailable"
    assert artifact["acceptance_gate_passed"] is False
    assert artifact["native_trainer_launched"] is False
    assert artifact["post_epoch"] == 6405
    assert artifact["pass2_val_exact_accuracy"] == 0.31
    assert artifact["command"] == exp4148.build_train_command(exp4148.Exp4148Config(repo_root=tmp_path), current_epoch=6405)
