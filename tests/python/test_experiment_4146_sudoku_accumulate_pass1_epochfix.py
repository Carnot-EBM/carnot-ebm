"""Tests for Exp 4146 Sudoku Extreme epoch-ceiling no-op guard.

Spec refs: REQ-LEARN-4146, SCENARIO-LEARN-4146,
SCENARIO-LEARN-4146-BLOCKED-NOOP.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116
from carnot import experiment_4146_sudoku_accumulate_pass1_epochfix as exp4146


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
                "  batch_size: 768",
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


def _make_dataset(dataset_dir: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "dataset.json").write_text('{"num_puzzles": 1}\n', encoding="utf-8")
        for name in ("all__inputs.npy", "all__labels.npy", "all__puzzle_identifiers.npy"):
            (split_dir / name).write_bytes(f"{split}:{name}".encode("ascii"))
    (dataset_dir / "metadata.json").write_text('{"seed": 4146}\n', encoding="utf-8")


def _write_checkpoint(path: Path, *, epoch: int, timer_train_s: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": epoch * 7,
            "nano_trm_manual_lr_step": epoch * 7,
            "callbacks": {"Timer": {"time_elapsed": {"train": timer_train_s}}},
            "loops": {
                "fit_loop": {
                    "epoch_progress": {
                        "total": {"completed": epoch, "processed": epoch},
                        "current": {"completed": epoch, "processed": epoch},
                    }
                }
            },
            "state_dict": {"weight": torch.tensor([1.0])},
        },
        path,
    )
    return path


def _write_metrics(run_dir: Path, *, val: float | None, epoch: int) -> Path:
    metrics = run_dir / "csv" / "version_0" / "metrics.csv"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    rows = ["epoch,step,train/lr,val/exact_accuracy", f"{epoch},{epoch * 7},9.9e-05,"]
    if val is not None:
        rows.append(f"{epoch},{epoch * 7},,{val}")
    rows.append("")
    metrics.write_text("\n".join(rows), encoding="utf-8")
    return metrics


def test_req_learn_4146_spec_declares_epochfix_contract() -> None:
    """REQ-LEARN-4146: OpenSpec declares the Exp 4146 anti-no-op contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4146" in spec
    assert "SCENARIO-LEARN-4146" in spec
    assert "SCENARIO-LEARN-4146-BLOCKED-NOOP" in spec
    assert "results/experiment_4146_sudoku_accumulate_pass1_epochfix.json" in spec
    for field in exp4146.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_learn_4146_diagnosis_blocks_false_epoch_cap_fix(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4146-BLOCKED-NOOP: cap mismatch stops before training."""

    _make_ready_repo(tmp_path, max_epochs=50000)
    _write_checkpoint(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=6399,
        timer_train_s=3641.0,
    )
    trainer_calls = 0

    def forbidden_runner(_config: exp4146.Exp4146Config, _seed_epoch: int) -> exp4116.ResumeRunResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("blocked_noop diagnosis must not train")

    output = tmp_path / "results" / exp4146.RESULT_FILENAME
    artifact = exp4146.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        trainer_runner=forbidden_runner,
    )

    assert trainer_calls == 0
    assert artifact["honest_verdict"].startswith("blocked_noop_")
    assert "cap_not_confirmed" in artifact["honest_verdict"]
    assert artifact["max_epochs_cap_confirmed"] is False
    assert artifact["seed_epoch"] == 6399
    assert artifact["post_epoch"] == 6399
    assert artifact["val_exact_accuracy"] is None
    assert artifact["duration_s"] < 120
    assert artifact["diagnosis"]["checkpoint_epoch"] == 6399
    assert artifact["diagnosis"]["config_max_epochs"] == 50000
    assert artifact["diagnosis"]["timer_train_elapsed_s"] == 3641.0
    assert artifact["acceptance_gate_passed"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_learn_4146_epochfixed_pass_proves_real_training(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4146: complete verdict requires epoch advance and real val."""

    _make_ready_repo(tmp_path, max_epochs=50000)
    _make_dataset(tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k")
    stable = _write_checkpoint(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        epoch=50000,
    )
    env_config = exp4146.Exp4146Config(repo_root=tmp_path)
    env = exp4146.build_train_env(env_config)
    assert env["DISABLE_COMPILE"] == "1"
    assert env_config.trainer_path == tmp_path / "nano-trm" / "src" / "nn" / "train.py"
    assert env_config.data_config_path == tmp_path / "nano-trm" / "src" / "nn" / "configs" / "data" / "sudoku_extreme_1k_aug1k.yaml"
    assert env_config.to_4127_config().stable_checkpoint_path == stable
    calls = 0

    def fake_runner(config: exp4146.Exp4146Config, seed_epoch: int) -> exp4116.ResumeRunResult:
        nonlocal calls
        calls += 1
        post_epoch = seed_epoch + 4
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
            duration_s=121.5,
            command=exp4146.build_train_command(config, seed_epoch=seed_epoch),
            stdout_tail=["trained fixture"],
            run_dir=config.pass_run_dir(),
        )

    artifact = exp4146.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4146.RESULT_FILENAME,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        dataset_builder=lambda _config: False,
        trainer_runner=fake_runner,
    )

    assert calls == 1
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["max_epochs_cap_confirmed"] is True
    assert artifact["seed_epoch"] == 50000
    assert artifact["post_epoch"] == 50004
    assert artifact["val_exact_accuracy"] == 0.31
    assert artifact["duration_s"] == 121.5
    assert artifact["stable_checkpoint_path"] == str(stable)
    assert "trainer.max_epochs=53000" in artifact["command"]
    assert artifact["acceptance_gate_passed"] is True


def test_req_learn_4146_anti_noop_edges_and_schema_errors(tmp_path: Path) -> None:
    """REQ-LEARN-4146: schema rejects fake-complete and wrapped fields."""

    _make_ready_repo(tmp_path, max_epochs=10)
    config = exp4146.Exp4146Config(repo_root=tmp_path)
    seed = exp4146.CheckpointState(
        load_ok=True,
        detail="ok",
        epoch=10,
        global_step=70,
        manual_lr_step=70,
        timer_train_elapsed_s=0.0,
    )
    post_same = exp4146.CheckpointState(
        load_ok=True,
        detail="ok",
        epoch=10,
        global_step=70,
        manual_lr_step=70,
        timer_train_elapsed_s=0.0,
    )
    post_advanced = exp4146.CheckpointState(
        load_ok=True,
        detail="ok",
        epoch=11,
        global_step=77,
        manual_lr_step=77,
        timer_train_elapsed_s=10.0,
    )

    no_epoch = exp4146.build_result_artifact(
        run_config=config,
        diagnosis=exp4146.diagnose_epoch_cap(config, checkpoint_state=seed),
        seed_state=seed,
        post_state=post_same,
        run_result=exp4116.ResumeRunResult(
            0,
            config.stable_checkpoint_path,
            True,
            "ok",
            None,
            None,
            130.0,
            exp4146.build_train_command(config, seed_epoch=10),
            [],
            config.pass_run_dir(),
        ),
        val_exact_accuracy=0.3,
        val_metrics_path=config.pass_run_dir() / "metrics.csv",
    )
    short = exp4146.build_result_artifact(
        run_config=config,
        diagnosis=exp4146.diagnose_epoch_cap(config, checkpoint_state=seed),
        seed_state=seed,
        post_state=post_advanced,
        run_result=exp4116.ResumeRunResult(
            0,
            config.stable_checkpoint_path,
            True,
            "ok",
            None,
            None,
            7.0,
            exp4146.build_train_command(config, seed_epoch=10),
            [],
            config.pass_run_dir(),
        ),
        val_exact_accuracy=0.3,
        val_metrics_path=config.pass_run_dir() / "metrics.csv",
    )
    no_val = exp4146.build_result_artifact(
        run_config=config,
        diagnosis=exp4146.diagnose_epoch_cap(config, checkpoint_state=seed),
        seed_state=seed,
        post_state=post_advanced,
        run_result=exp4116.ResumeRunResult(
            0,
            config.stable_checkpoint_path,
            True,
            "ok",
            None,
            None,
            130.0,
            exp4146.build_train_command(config, seed_epoch=10),
            [],
            config.pass_run_dir(),
        ),
        val_exact_accuracy=None,
        val_metrics_path=None,
    )

    assert no_epoch["honest_verdict"].startswith("blocked_noop_epoch_not_advanced")
    assert short["honest_verdict"].startswith("blocked_noop_duration_too_short")
    assert no_val["honest_verdict"].startswith("blocked_noop_missing_val_exact_accuracy")
    assert no_epoch["acceptance_gate_passed"] is True

    blocked = exp4146.build_blocked_artifact(
        "blocked_cuda_unavailable",
        run_config=config,
        diagnosis=exp4146.diagnose_epoch_cap(config, checkpoint_state=seed),
        preconditions_checked=[exp4107.PreconditionCheck("cuda_available", False, "no cuda")],
        duration_s=1.0,
    )
    assert exp4146.artifact_schema_errors(blocked) == []

    invalid = dict(blocked)
    invalid.update(
        {
            "honest_verdict": "pending",
            "max_epochs_cap_confirmed": "false",
            "seed_epoch": True,
            "post_epoch": "10",
            "val_exact_accuracy": 2,
            "stable_checkpoint_path": "elsewhere.ckpt",
            "duration_s": [1.0],
            "random_seed": True,
            "acceptance_gate_passed": "yes",
        }
    )
    errors = exp4146.artifact_schema_errors(invalid)
    assert "honest_verdict must be terminal-prefixed or blocked_noop" in errors
    assert "max_epochs_cap_confirmed must be a bare bool" in errors
    assert "seed_epoch must be an int or null" in errors
    assert "post_epoch must be an int or null" in errors
    assert "val_exact_accuracy must be numeric between 0 and 1 or null" in errors
    assert "stable_checkpoint_path must be the shared sudoku_extreme_baseline/last.ckpt path" in errors
    assert "duration_s must be a scalar bounded number below 86400" in errors
    assert "random_seed must be a bare int" in errors
    assert "acceptance_gate_passed must be a bare bool" in errors
    complete_without_proof = dict(blocked)
    complete_without_proof["honest_verdict"] = "complete: fake"
    assert "complete verdict requires duration>120, epoch advance, and real val_exact_accuracy" in (
        exp4146.artifact_schema_errors(complete_without_proof)
    )
    assert "missing required field honest_verdict" in exp4146.artifact_schema_errors({})
    assert "honest_verdict must be a string" in exp4146.artifact_schema_errors(
        {**blocked, "honest_verdict": 7}
    )

    exp4146.write_result_artifact(tmp_path / "artifact.json", blocked)
    assert json.loads((tmp_path / "artifact.json").read_text(encoding="utf-8")) == blocked
    try:
        exp4146.validate_artifact(invalid)
    except ValueError as exc:
        assert "honest_verdict must be terminal-prefixed or blocked_noop" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("validate_artifact should reject schema errors")


def test_req_learn_4146_preconditions_and_loader_edges(tmp_path: Path) -> None:
    """REQ-LEARN-4146: failed resources block and loaders fail closed."""

    _make_ready_repo(tmp_path, max_epochs=50000)
    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    _write_checkpoint(stable, epoch=50000)
    invalid_config = tmp_path / "bad.yaml"
    invalid_config.write_text("[bad", encoding="utf-8")
    list_config = tmp_path / "list.yaml"
    list_config.write_text("- 1\n", encoding="utf-8")
    trainer_config = tmp_path / "trainer.yaml"
    trainer_config.write_text("trainer:\n  max_epochs: 77\n", encoding="utf-8")
    loop_only = tmp_path / "loop_only.ckpt"
    torch.save(
        {
            "loops": {
                "fit_loop": {
                    "epoch_progress": {
                        "total": {"completed": 88},
                    }
                }
            },
            "callbacks": {},
        },
        loop_only,
    )
    trainer_calls = 0

    def forbidden_runner(_config: exp4146.Exp4146Config, _seed_epoch: int) -> exp4116.ResumeRunResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("precondition failures must not train")

    assert exp4146.read_config_max_epochs(tmp_path / "missing.yaml") is None
    assert exp4146.read_config_max_epochs(invalid_config) is None
    assert exp4146.read_config_max_epochs(list_config) is None
    assert exp4146.read_config_max_epochs(trainer_config) == 77
    assert exp4146.read_checkpoint_state(tmp_path / "missing.ckpt").load_ok is False
    assert exp4146.read_checkpoint_state(loop_only).epoch == 88
    assert exp4146._float_or_none("0.25") == 0.25
    assert exp4146._float_or_none("not-a-number") is None
    assert exp4146._float_or_none(True) is None
    assert exp4146._nested_get({"a": 1}, ("a", "b")) is None
    assert exp4146._timer_train_elapsed(None) is None
    assert exp4146._timer_train_elapsed({}) is None
    assert exp4146._max_time_seconds("bad") is None
    assert exp4146._max_time_seconds("aa:bb:cc:dd") is None
    incomplete = exp4146.EpochCapDiagnosis(None, None, None, None, False, None, 3600.0)
    no_timer = exp4146.EpochCapDiagnosis(100, 0, 50000, 3100, False, 0.0, 3600.0)
    assert exp4146._noop_reason_for_diagnosis(incomplete) == "blocked_noop_cap_diagnosis_incomplete"
    assert exp4146._noop_reason_for_diagnosis(no_timer) == "blocked_noop_cap_not_confirmed"

    blocked_uv = exp4146.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "blocked_uv.json",
        uv_resolver=lambda _name: None,
        cuda_checker=lambda: (True, "cuda fixture"),
        trainer_runner=forbidden_runner,
    )
    assert blocked_uv["honest_verdict"] == "blocked_uv_missing"

    (tmp_path / "nano-trm" / "src" / "nn" / "train.py").unlink()
    missing_trainer = exp4146.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "missing_trainer.json",
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        trainer_runner=forbidden_runner,
    )
    assert missing_trainer["honest_verdict"] == "blocked_nanotrm_train_missing"
    _make_ready_repo(tmp_path, max_epochs=50000)

    blocked_cuda = exp4146.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "blocked_cuda.json",
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (False, "no cuda"),
        trainer_runner=forbidden_runner,
    )
    assert blocked_cuda["honest_verdict"] == "blocked_cuda_unavailable"

    stable.unlink()
    missing_stable = exp4146.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "missing_stable.json",
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        trainer_runner=forbidden_runner,
    )
    assert missing_stable["honest_verdict"] == "blocked_stable_checkpoint_missing"
    _write_checkpoint(stable, epoch=50000)
    dataset_missing = exp4146.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "dataset_missing.json",
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        dataset_builder=lambda _config: None,
        trainer_runner=forbidden_runner,
    )
    assert dataset_missing["honest_verdict"] == "blocked_dataset_missing"
    assert trainer_calls == 0
