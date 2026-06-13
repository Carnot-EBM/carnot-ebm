"""Tests for Exp 4135 fixed-LR Sudoku Extreme accumulation pass 1.

Spec refs: REQ-LEARN-4135, SCENARIO-LEARN-4135,
SCENARIO-LEARN-4135-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116
from carnot import experiment_4126_lr_resume_correctness_fix as exp4126
from carnot import experiment_4135_sudoku_accumulate_pass1_fixed_lr as exp4135


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _make_ready_repo(root: Path) -> None:
    trainer = root / "nano-trm" / "src" / "nn" / "train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# trainer fixture\n", encoding="utf-8")
    configs = root / "nano-trm" / "src" / "nn" / "configs"
    (configs / "experiment").mkdir(parents=True, exist_ok=True)
    (configs / "data").mkdir(parents=True, exist_ok=True)
    (configs / "experiment" / "trm_sudoku_extreme_1k_aug_1k.yaml").write_text(
        "model_tuning:\n  learning_rate: 1e-4\n", encoding="utf-8"
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
    (dataset_dir / "metadata.json").write_text('{"seed": 4135}\n', encoding="utf-8")


def _write_lr_artifact(path: Path, *, continuous: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: lr fixture",
                "lr_continuous_across_resume": continuous,
                "stable_checkpoint_path": str(
                    path.parents[1] / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_metrics(run_dir: Path, *, val: float | None, lrs: list[float]) -> Path:
    metrics = run_dir / "csv" / "version_0" / "metrics.csv"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    rows = ["epoch,step,train/lr,val/exact_accuracy"]
    for index, lr in enumerate(lrs):
        rows.append(f"{4300 + index},{4300 + index},{lr},")
    if val is not None:
        rows.append(f"4400,4400,,{val}")
    rows.append("")
    metrics.write_text("\n".join(rows), encoding="utf-8")
    return metrics


def _resume_result(
    config: exp4135.Exp4135Config,
    *,
    val: float | None,
    duration_s: float = 91.0,
    return_code: int = 0,
) -> exp4116.ResumeRunResult:
    run_dir = config.pass_run_dir()
    metric = None
    if val is not None:
        metric = exp4107.ExactAccuracy(
            "val/exact_accuracy",
            val,
            run_dir / "csv" / "version_0" / "metrics.csv",
        )
    return exp4116.ResumeRunResult(
        return_code=return_code,
        stable_checkpoint_path=config.stable_checkpoint_path,
        checkpoint_reload_ok=True,
        checkpoint_reload_detail="loadable fixture",
        val_exact_accuracy=metric,
        cumulative_epochs=4401 if val is not None else None,
        duration_s=duration_s,
        command=exp4135.build_train_command(config),
        stdout_tail=["pass fixture"],
        run_dir=run_dir,
    )


def test_req_learn_4135_spec_declares_pass1_contract() -> None:
    """REQ-LEARN-4135: OpenSpec declares the pass1 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4135" in spec
    assert "SCENARIO-LEARN-4135" in spec
    assert "SCENARIO-LEARN-4135-BLOCKED" in spec
    assert "results/experiment_4135_sudoku_accumulate_pass1_fixed_lr.json" in spec
    for field in exp4135.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4135_command_env_model_specs_and_checksum(tmp_path: Path) -> None:
    """REQ-LEARN-4135: native command and checksum use stable resume inputs."""

    _make_ready_repo(tmp_path)
    _make_dataset(tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k")
    config = exp4135.Exp4135Config(repo_root=tmp_path)
    config.stable_dir.mkdir(parents=True)
    config.stable_checkpoint_path.write_bytes(b"checkpoint-a")

    command = exp4135.build_train_command(config)
    env = exp4135.build_train_env(config)
    specs = exp4135.model_specs(config)
    checksum_a = exp4135.compute_reproducibility_checksum(config)
    config.stable_checkpoint_path.write_bytes(b"checkpoint-b")
    checksum_b = exp4135.compute_reproducibility_checksum(config)
    as_4116 = config.to_4116_config()

    assert command[:4] == ["uv", "run", "python", "src/nn/train.py"]
    assert f"hydra.run.dir={config.pass_run_dir()}" in command
    assert f"ckpt_path={config.stable_checkpoint_path}" in command
    assert "+trainer.max_time=00:01:00:00" in command
    assert "timekeeping.batch_size=128" in command
    assert f"callbacks.model_checkpoint.dirpath={config.stable_dir}" in command
    assert env["DISABLE_COMPILE"] == "1"
    assert env["WANDB_MODE"] == "disabled"
    assert specs["model"] == "nano-trm"
    assert specs["experiment_config"] == "trm_sudoku_extreme_1k_aug_1k"
    assert checksum_a.startswith("sha256:")
    assert checksum_a != checksum_b
    assert as_4116.hydra_run_dir == config.pass_run_dir()


def test_scenario_learn_4135_metric_summary_detects_rewarm_guard(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4135: CSV metrics report val and non-rewarmed LR."""

    _make_ready_repo(tmp_path)
    config = exp4135.Exp4135Config(repo_root=tmp_path)
    metrics = _write_metrics(
        config.pass_run_dir(),
        val=0.31,
        lrs=[9.99e-5, 9.98e-5],
    )

    summary = exp4135.summarize_pass_metrics(config.pass_run_dir())
    rewarm_summary = exp4135.PassMetricSummary(
        val_exact_accuracy=0.29,
        first_train_lr=exp4126.FRESH_WARMUP_FIRST_LR,
        train_lr_point_count=1,
        val_metrics_path=metrics,
        first_train_lr_metrics_path=metrics,
    )
    empty = exp4135.summarize_pass_metrics(tmp_path / "missing")

    assert summary.val_exact_accuracy == 0.31
    assert summary.first_train_lr == 9.99e-5
    assert summary.lr_continued_not_rewarmed is True
    assert summary.to_dict()["val_metrics_path"] == str(metrics)
    assert rewarm_summary.lr_continued_not_rewarmed is False
    assert empty.val_exact_accuracy is None
    assert empty.lr_continued_not_rewarmed is False
    assert exp4135._float_or_none("0.25") == 0.25
    assert exp4135._float_or_none("not-a-number") is None
    assert exp4135._float_or_none(True) is None
    assert exp4135._rounded(None) is None


def test_req_learn_4135_checkpoint_timer_reset_preserves_lr_step(tmp_path: Path) -> None:
    """REQ-LEARN-4135: exhausted Lightning timer is reset without LR rewarm."""

    checkpoint = tmp_path / "last.ckpt"
    torch.save(
        {
            "callbacks": {
                "Timer": {
                    "time_elapsed": {
                        "train": 3661.0,
                        "sanity_check": 2.0,
                        "validate": -3.0,
                        "test": 0.5,
                        "predict": 0.25,
                    }
                }
            },
            "nano_trm_manual_lr_step": 19006,
            "state_dict": {"weight": torch.tensor([1.0])},
        },
        checkpoint,
    )

    result = exp4135.reset_checkpoint_timer_state(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    unchanged = exp4135.reset_checkpoint_timer_state(tmp_path / "missing.ckpt")

    assert result.changed is True
    assert result.manual_lr_step == 19006
    assert payload["callbacks"]["Timer"]["time_elapsed"] == {
        "train": 0.0,
        "sanity_check": 0.0,
        "validate": 0.0,
        "test": 0.0,
        "predict": 0.0,
    }
    assert payload["nano_trm_manual_lr_step"] == 19006
    assert unchanged.changed is False
    assert "missing checkpoint" in unchanged.detail


def test_scenario_learn_4135_artifact_reports_scalar_fields_and_verdicts(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-4135: result artifacts expose accuracy, delta, and LR bool."""

    _make_ready_repo(tmp_path)
    _make_dataset(tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k")
    config = exp4135.Exp4135Config(repo_root=tmp_path)
    config.stable_dir.mkdir(parents=True)
    config.stable_checkpoint_path.write_bytes(b"checkpoint")
    metrics = _write_metrics(config.pass_run_dir(), val=0.35, lrs=[9.99e-5])
    summary = exp4135.summarize_pass_metrics(config.pass_run_dir())

    artifact = exp4135.build_result_artifact(
        run_config=config,
        run_result=_resume_result(config, val=0.35, duration_s=91.25),
        pass_metrics=summary,
        preconditions_checked=[],
        lr_fix_artifact={"lr_continuous_across_resume": True},
        dataset_generated=False,
    )

    assert artifact["honest_verdict"] == (
        "complete: val=0.3500 improved_delta=0.0720 still_below_0.87 -> pass2 continues"
    )
    assert artifact["val_exact_accuracy"] == 0.35
    assert artifact["delta_vs_previous"] == 0.072
    assert artifact["lr_continued_not_rewarmed"] is True
    assert artifact["matches_published_087"] is False
    assert artifact["duration_s"] == 91.25
    assert artifact["random_seed"] == 4108
    assert artifact["exact_accuracy_metrics_path"] == str(metrics)
    assert artifact["acceptance_gate_passed"] is True
    exp4135.validate_artifact(artifact)

    matched = exp4135.build_result_artifact(
        run_config=config,
        run_result=_resume_result(config, val=0.86),
        pass_metrics=exp4135.PassMetricSummary(0.86, 9.99e-5, 1, metrics, metrics),
        preconditions_checked=[],
        lr_fix_artifact={"lr_continuous_across_resume": True},
        dataset_generated=False,
    )
    assert matched["matches_published_087"] is True
    assert matched["honest_verdict"] == "complete: val=0.8600 reproduced_within_0.02_of_0.87"

    stalled = exp4135.build_result_artifact(
        run_config=config,
        run_result=_resume_result(config, val=0.278),
        pass_metrics=exp4135.PassMetricSummary(0.278, 9.99e-5, 1, metrics, metrics),
        preconditions_checked=[],
        lr_fix_artifact={"lr_continuous_across_resume": True},
        dataset_generated=False,
    )
    assert "stall_flagged" in stalled["honest_verdict"]
    assert stalled["acceptance_gate_passed"] is True

    rewarm = exp4135.build_result_artifact(
        run_config=config,
        run_result=_resume_result(config, val=0.35),
        pass_metrics=exp4135.PassMetricSummary(
            0.35, exp4126.FRESH_WARMUP_FIRST_LR, 1, metrics, metrics
        ),
        preconditions_checked=[],
        lr_fix_artifact={"lr_continuous_across_resume": True},
        dataset_generated=False,
    )
    assert rewarm["lr_continued_not_rewarmed"] is False
    assert rewarm["acceptance_gate_passed"] is False


def test_req_learn_4135_schema_guards_required_scalar_fields(tmp_path: Path) -> None:
    """REQ-LEARN-4135: schema rejects wrappers and ambiguous fields."""

    _make_ready_repo(tmp_path)
    config = exp4135.Exp4135Config(repo_root=tmp_path)
    artifact = exp4135.build_blocked_artifact(
        "blocked_cuda_unavailable",
        run_config=config,
        preconditions_checked=[exp4107.PreconditionCheck("cuda_available", False, "no cuda")],
        duration_s=1.0,
    )

    assert exp4135.artifact_schema_errors(artifact) == []

    invalid = dict(artifact)
    invalid.update(
        {
            "honest_verdict": "pending",
            "val_exact_accuracy": 7,
            "delta_vs_previous": "0.1",
            "lr_continued_not_rewarmed": "true",
            "matches_published_087": "false",
            "stable_checkpoint_path": "somewhere/else.ckpt",
            "random_seed": True,
            "reproducibility_checksum": "nope",
            "model_specs": {"model": "other"},
            "duration_s": [1.0],
        }
    )
    errors = exp4135.artifact_schema_errors(invalid)
    assert "honest_verdict must be terminal-prefixed or blocked" in errors
    assert "val_exact_accuracy must be numeric between 0 and 1 or null" in errors
    assert "delta_vs_previous must be numeric or null" in errors
    assert "lr_continued_not_rewarmed must be a bare bool" in errors
    assert "matches_published_087 must be a bare bool" in errors
    assert "stable_checkpoint_path must be the shared sudoku_extreme_baseline/last.ckpt path" in errors
    assert "random_seed must be a bare int" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors
    assert "model_specs must name nano-trm and trm_sudoku_extreme_1k_aug_1k" in errors
    assert "duration_s must be a scalar bounded number below 4800" in errors
    assert "honest_verdict must be a string" in exp4135.artifact_schema_errors(
        {**artifact, "honest_verdict": 7}
    )
    assert "acceptance_gate_passed=true requires val, LR continuity, and improvement or stall verdict" in (
        exp4135.artifact_schema_errors({**artifact, "acceptance_gate_passed": True})
    )
    assert "acceptance_gate_passed must be a bare bool" in (
        exp4135.artifact_schema_errors({**artifact, "acceptance_gate_passed": "yes"})
    )
    assert "matches_published_087=true requires val within 0.02 of 0.87" in (
        exp4135.artifact_schema_errors(
            {**artifact, "matches_published_087": True, "val_exact_accuracy": 0.1}
        )
    )
    assert "missing required field honest_verdict" in exp4135.artifact_schema_errors({})

    output = tmp_path / "artifact.json"
    exp4135.write_result_artifact(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    try:
        exp4135.validate_artifact(invalid)
    except ValueError as exc:
        assert "honest_verdict must be terminal-prefixed or blocked" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("validate_artifact should reject schema errors")


def test_req_learn_4135_run_experiment_blocks_before_training(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4135-BLOCKED: failed preconditions do not train."""

    _make_ready_repo(tmp_path)
    lr_artifact = _write_lr_artifact(
        tmp_path / "results" / "experiment_4126_lr_resume_correctness_fix.json",
        continuous=True,
    )
    trainer_calls = 0

    def forbidden_runner(_config: exp4135.Exp4135Config) -> exp4116.ResumeRunResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("blocked branches must not train")

    blocked_uv = exp4135.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "blocked_uv.json",
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: None,
        cuda_checker=lambda: (True, "cuda fixture"),
        trainer_runner=forbidden_runner,
    )
    assert blocked_uv["honest_verdict"] == "blocked_uv_missing"

    (tmp_path / "nano-trm" / "src" / "nn" / "train.py").unlink()
    missing_trainer = exp4135.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "missing_trainer.json",
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        trainer_runner=forbidden_runner,
    )
    assert missing_trainer["honest_verdict"] == "blocked_nanotrm_train_missing"
    _make_ready_repo(tmp_path)

    blocked_cuda = exp4135.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "blocked_cuda.json",
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (False, "no cuda"),
        trainer_runner=forbidden_runner,
    )
    assert blocked_cuda["honest_verdict"] == "blocked_cuda_unavailable"

    missing_stable = exp4135.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "missing_stable.json",
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (False, "missing"),
        trainer_runner=forbidden_runner,
    )
    assert missing_stable["honest_verdict"] == "blocked_stable_checkpoint_missing"

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    dataset_missing = exp4135.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "dataset_missing.json",
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        dataset_builder=lambda _config: None,
        trainer_runner=forbidden_runner,
    )
    assert dataset_missing["honest_verdict"] == "blocked_dataset_missing"
    assert dataset_missing["dataset_generated"] is True

    _write_lr_artifact(lr_artifact, continuous=False)
    blocked_lr = exp4135.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "blocked_lr.json",
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        trainer_runner=forbidden_runner,
    )
    assert blocked_lr["honest_verdict"] == "blocked_lr_fix_not_landed"
    assert trainer_calls == 0


def test_req_learn_4135_run_experiment_writes_measured_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4135: runner writes one-pass artifact from native metrics."""

    _make_ready_repo(tmp_path)
    _make_dataset(tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k")
    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    lr_artifact = _write_lr_artifact(
        tmp_path / "results" / "experiment_4126_lr_resume_correctness_fix.json",
        continuous=True,
    )
    calls = 0

    def fake_runner(config: exp4135.Exp4135Config) -> exp4116.ResumeRunResult:
        nonlocal calls
        calls += 1
        _write_metrics(config.pass_run_dir(), val=0.35, lrs=[9.99e-5, 9.98e-5])
        return exp4135.verify_completed_resume_pass(
            config,
            duration_s=91.25,
            return_code=0,
            command=exp4135.build_train_command(config),
            stdout_tail=["done"],
            checkpoint_loader=lambda _path: (True, "loadable fixture"),
        )

    output = tmp_path / "results" / exp4135.RESULT_FILENAME
    artifact = exp4135.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        dataset_builder=lambda _config: False,
        trainer_runner=fake_runner,
    )

    assert calls == 1
    assert artifact["val_exact_accuracy"] == 0.35
    assert artifact["delta_vs_previous"] == 0.072
    assert artifact["lr_continued_not_rewarmed"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    def exploding_runner(_config: exp4135.Exp4135Config) -> exp4116.ResumeRunResult:
        raise RuntimeError("trainer exploded")

    failed = exp4135.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "trainer_error.json",
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        dataset_builder=lambda _config: False,
        trainer_runner=exploding_runner,
    )
    assert failed["honest_verdict"] == "complete: missing_real_val_exact_accuracy"
    assert "trainer exploded" in failed["stdout_tail"][0]
