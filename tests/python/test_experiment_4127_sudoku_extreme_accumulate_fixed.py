"""Tests for Exp 4127 fixed-LR Sudoku Extreme accumulation.

Spec refs: REQ-LEARN-4127, SCENARIO-LEARN-4127,
SCENARIO-LEARN-4127-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116
from carnot import experiment_4127_sudoku_extreme_accumulate_fixed as exp4127


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _make_ready_repo(root: Path) -> None:
    trainer = root / "nano-trm" / "src" / "nn" / "train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# trainer fixture\n", encoding="utf-8")


def _make_dataset(dataset_dir: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "dataset.json").write_text('{"num_puzzles": 1}\n', encoding="utf-8")
        for name in ("all__inputs.npy", "all__labels.npy", "all__puzzle_identifiers.npy"):
            (split_dir / name).write_bytes(f"{split}:{name}".encode("ascii"))
    (dataset_dir / "metadata.json").write_text('{"seed": 4127}\n', encoding="utf-8")


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


def _write_prior_artifact(root: Path, *, val: float) -> Path:
    path = root / "results" / "experiment_4118_sudoku_extreme_resume_pass3.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: prior fixture",
                "val_exact_accuracy": val,
                "stable_checkpoint_path": str(
                    root / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _resume_result(
    config: exp4127.Exp4127Config,
    *,
    pass_index: int,
    val: float | None,
    duration_s: float,
    return_code: int = 0,
) -> exp4116.ResumeRunResult:
    run_dir = config.pass_run_dir(pass_index)
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
        cumulative_epochs=4_300 + pass_index * 100 if val is not None else None,
        duration_s=duration_s,
        command=exp4127.build_train_command(config, pass_index),
        stdout_tail=[f"pass {pass_index} fixture"],
        run_dir=run_dir,
    )


def test_req_learn_4127_spec_declares_accumulation_contract() -> None:
    """REQ-LEARN-4127: OpenSpec declares the fixed-LR accumulation artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4127" in spec
    assert "SCENARIO-LEARN-4127" in spec
    assert "SCENARIO-LEARN-4127-BLOCKED" in spec
    assert "blocked_lr_fix_not_landed" in spec
    assert "results/experiment_4127_sudoku_extreme_accumulate_fixed.json" in spec
    for field in exp4127.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_learn_4127_blocked_lr_fix_does_not_train(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4127-BLOCKED: stale LR fix stops native training."""

    _make_ready_repo(tmp_path)
    lr_artifact = _write_lr_artifact(
        tmp_path / "results" / "experiment_4126_lr_resume_correctness_fix.json",
        continuous=False,
    )
    output = tmp_path / "results" / exp4127.RESULT_FILENAME
    trainer_calls = 0

    def forbidden_runner(
        _config: exp4127.Exp4127Config,
        _pass_index: int,
    ) -> exp4116.ResumeRunResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("LR-blocked branch must not train")

    artifact = exp4127.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        trainer_runner=forbidden_runner,
    )

    assert artifact["honest_verdict"] == "blocked_lr_fix_not_landed"
    assert artifact["val_trajectory"] == []
    assert artifact["matches_published_087"] is False
    assert artifact["per_pass_delta_vs_v381"]["beats_v381"] is False
    assert "contiguous" in artifact["contiguous_run_recommendation"]
    assert trainer_calls == 0
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_req_learn_4127_command_and_env_use_stable_checkpoint_and_compile_off(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-4127: command resumes/saves stable path with one-hour bound."""

    _make_ready_repo(tmp_path)
    config = exp4127.Exp4127Config(repo_root=tmp_path)
    config.stable_dir.mkdir(parents=True)
    config.stable_checkpoint_path.write_bytes(b"checkpoint")

    command = exp4127.build_train_command(config, 2)
    env = exp4127.build_train_env(config)

    assert command[:4] == ["uv", "run", "python", "src/nn/train.py"]
    assert config.trainer_path == tmp_path / "nano-trm" / "src" / "nn" / "train.py"
    assert f"hydra.run.dir={config.pass_run_dir(2)}" in command
    assert f"ckpt_path={config.stable_checkpoint_path}" in command
    assert "+trainer.max_time=00:01:00:00" in command
    assert "timekeeping.batch_size=128" in command
    assert f"callbacks.model_checkpoint.dirpath={config.stable_dir}" in command
    assert f"+callbacks.exp4127_progress.checkpoint_dir={config.stable_dir}" in command
    assert "DISABLE_COMPILE" in env
    assert env["WANDB_MODE"] == "disabled"


def test_req_learn_4127_loaders_and_pass_verifier_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-4127: loaders and pass verifier handle missing evidence."""

    assert exp4127._float_or_none(True) is None
    assert exp4127._float_or_none("0.1") is None
    assert exp4127.find_lr_fix_artifact(tmp_path) == (
        tmp_path / "results" / "experiment_4126_lr_resume_correctness_fix.json"
    )
    first = _write_lr_artifact(tmp_path / "results" / "experiment_4126_a.json", continuous=True)
    second = _write_lr_artifact(tmp_path / "results" / "experiment_4126_b.json", continuous=True)
    assert first.exists()
    assert exp4127.find_lr_fix_artifact(tmp_path) == second
    assert exp4127.load_lr_fix_artifact(tmp_path / "missing.json") == {}
    bad_json = tmp_path / "results" / "bad_lr.json"
    bad_json.write_text("{not json", encoding="utf-8")
    assert exp4127.load_lr_fix_artifact(bad_json) == {}
    assert exp4127.lr_fix_landed({"lr_continuous_across_resume": "true"}) is False

    assert exp4127.load_starting_val(tmp_path / "no_prior") is None
    (tmp_path / "results" / "experiment_4118_sudoku_extreme_resume_pass3.json").write_text(
        "{not json",
        encoding="utf-8",
    )
    prior_4117 = tmp_path / "results" / "experiment_4117_sudoku_extreme_resume_pass2.json"
    prior_4117.write_text(json.dumps({"val_exact_accuracy": 0.12}) + "\n", encoding="utf-8")
    recovered = exp4127.load_starting_val(tmp_path)
    assert recovered is not None
    assert recovered.val_exact_accuracy == 0.12
    assert recovered.source == str(prior_4117)

    config = exp4127.Exp4127Config(repo_root=tmp_path)
    missing_check = exp4127.check_stable_checkpoint(
        config,
        checkpoint_loader=lambda _path: (True, "unused"),
    )
    assert missing_check.available is False

    missing_metrics = exp4127.verify_completed_resume_pass(
        config,
        1,
        duration_s=1.0,
        checkpoint_loader=lambda _path: (True, "unused"),
    )
    assert missing_metrics.checkpoint_reload_ok is False
    assert missing_metrics.val_exact_accuracy is None
    assert missing_metrics.command == exp4127.build_train_command(config, 1)


def test_scenario_learn_4127_artifact_reports_trajectory_and_delta_comparison(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-4127: trajectory yields match bool and .381 comparison."""

    config = exp4127.Exp4127Config(repo_root=tmp_path)
    starting = exp4127.StartingVal(
        val_exact_accuracy=0.105,
        source="results/experiment_4118_sudoku_extreme_resume_pass3.json",
    )
    pass_results = [
        _resume_result(config, pass_index=1, val=0.125, duration_s=91.0),
        _resume_result(config, pass_index=2, val=0.145, duration_s=92.0),
    ]

    artifact = exp4127.build_result_artifact(
        run_config=config,
        lr_fix_artifact={"lr_continuous_across_resume": True},
        starting_val=starting,
        run_results=pass_results,
        preconditions_checked=[],
        dataset_generated=False,
    )

    assert artifact["honest_verdict"] == (
        "complete: val=0.1450 faster_than_.381_but_not_yet_0.87 -> .383 continues"
    )
    assert artifact["val_trajectory"][0]["source"] == starting.source
    assert artifact["val_trajectory"][1]["delta_vs_previous"] == 0.02
    assert artifact["val_trajectory"][2]["delta_vs_previous"] == 0.02
    assert artifact["matches_published_087"] is False
    assert artifact["per_pass_delta_vs_v381"]["mean_delta"] == 0.02
    assert artifact["per_pass_delta_vs_v381"]["beats_v381"] is True
    assert artifact["duration_s"] == [91.0, 92.0]
    assert artifact["total_duration_s"] == 183.0
    exp4127.validate_artifact(artifact)

    matched = exp4127.build_result_artifact(
        run_config=config,
        lr_fix_artifact={"lr_continuous_across_resume": True},
        starting_val=starting,
        run_results=[_resume_result(config, pass_index=1, val=0.86, duration_s=93.0)],
        preconditions_checked=[],
        dataset_generated=False,
    )
    assert matched["matches_published_087"] is True
    assert matched["honest_verdict"] == "complete: val=0.8600 reproduced_within_0.02_of_0.87"

    slower = exp4127.build_result_artifact(
        run_config=config,
        lr_fix_artifact={"lr_continuous_across_resume": True},
        starting_val=starting,
        run_results=[_resume_result(config, pass_index=1, val=0.11, duration_s=94.0)],
        preconditions_checked=[],
        dataset_generated=False,
    )
    assert slower["honest_verdict"] == (
        "complete: val=0.1100 not_faster_than_.381_and_not_yet_0.87 -> .383 continues"
    )


def test_req_learn_4127_schema_guards_reject_wrapper_and_unbounded_fields(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-4127: artifact schema rejects stale or ambiguous fields."""

    config = exp4127.Exp4127Config(repo_root=tmp_path)
    artifact = exp4127.build_blocked_artifact(
        "blocked_cuda_unavailable",
        stable_checkpoint_path=config.stable_checkpoint_path,
        preconditions_checked=[exp4107.PreconditionCheck("cuda_available", False, "no cuda")],
        duration_s=[],
    )

    assert exp4127.artifact_schema_errors(artifact) == []

    invalid = dict(artifact)
    invalid.update(
        {
            "honest_verdict": "pending",
            "val_trajectory": "0.1",
            "matches_published_087": "false",
            "per_pass_delta_vs_v381": {},
            "stable_checkpoint_path": "somewhere/else.ckpt",
            "duration_s": [4_800.0],
        }
    )
    errors = exp4127.artifact_schema_errors(invalid)
    assert "honest_verdict must be terminal-prefixed or blocked" in errors
    assert "val_trajectory must be a list" in errors
    assert "matches_published_087 must be a bare bool" in errors
    assert "per_pass_delta_vs_v381.beats_v381 must be a bare bool" in errors
    assert "stable_checkpoint_path must be the shared sudoku_extreme_baseline/last.ckpt path" in errors
    assert "each duration_s entry must be a bounded number below 4800" in errors

    missing_errors = exp4127.artifact_schema_errors({})
    assert "missing required field honest_verdict" in missing_errors
    assert "honest_verdict must be a string" in exp4127.artifact_schema_errors(
        {**artifact, "honest_verdict": 7}
    )
    assert "each duration_s entry must be a bounded number below 4800" in exp4127.artifact_schema_errors(
        {**artifact, "duration_s": "slow"}
    )
    assert exp4127.artifact_schema_errors({**artifact, "duration_s": 1.0}) == []
    assert "matches_published_087=true requires a final val within 0.02 of 0.87" in (
        exp4127.artifact_schema_errors({**artifact, "matches_published_087": True})
    )

    output = tmp_path / "artifact.json"
    exp4127.write_result_artifact(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    try:
        exp4127.validate_artifact(invalid)
    except ValueError as exc:
        assert "honest_verdict must be terminal-prefixed or blocked" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("validate_artifact should reject schema errors")


def test_req_learn_4127_run_experiment_executes_two_passes_and_writes_json(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-4127: run_experiment writes the measured two-pass artifact."""

    _make_ready_repo(tmp_path)
    _make_dataset(tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k")
    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    lr_artifact = _write_lr_artifact(
        tmp_path / "results" / "experiment_4126_lr_resume_correctness_fix.json",
        continuous=True,
    )
    _write_prior_artifact(tmp_path, val=0.105)
    calls: list[int] = []

    def fake_runner(
        config: exp4127.Exp4127Config,
        pass_index: int,
    ) -> exp4116.ResumeRunResult:
        calls.append(pass_index)
        val = 0.105 + 0.02 * pass_index
        run_dir = config.pass_run_dir(pass_index)
        metrics = run_dir / "csv" / "version_0" / "metrics.csv"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.write_text(
            "\n".join(
                [
                    "epoch,step,train/exact_accuracy,val/exact_accuracy",
                    f"{4300 + pass_index},0,0.2,",
                    f"{4300 + pass_index},0,,{val}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return exp4127.verify_completed_resume_pass(
            config,
            pass_index,
            duration_s=90.0 + pass_index,
            return_code=0,
            command=exp4127.build_train_command(config, pass_index),
            stdout_tail=[f"pass {pass_index} done"],
            checkpoint_loader=lambda _path: (True, "loadable fixture"),
        )

    output = tmp_path / "results" / exp4127.RESULT_FILENAME
    artifact = exp4127.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        dataset_builder=lambda _config: False,
        trainer_runner=fake_runner,
    )

    assert calls == [1, 2]
    assert artifact["val_trajectory"][-1]["val_exact_accuracy"] == 0.145
    assert artifact["per_pass_delta_vs_v381"]["beats_v381"] is True
    assert artifact["duration_s"] == [91.0, 92.0]
    assert artifact["acceptance_gate_passed"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    calls.clear()
    _write_prior_artifact(tmp_path, val=0.84)

    def matching_runner(
        config: exp4127.Exp4127Config,
        pass_index: int,
    ) -> exp4116.ResumeRunResult:
        calls.append(pass_index)
        return _resume_result(config, pass_index=pass_index, val=0.86, duration_s=91.0)

    matched = exp4127.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "matched.json",
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        dataset_builder=lambda _config: False,
        trainer_runner=matching_runner,
    )
    assert calls == [1]
    assert matched["matches_published_087"] is True


def test_req_learn_4127_run_experiment_blockers_and_trainer_exception(tmp_path: Path) -> None:
    """REQ-LEARN-4127: run_experiment records blockers and trainer exceptions."""

    _make_ready_repo(tmp_path)
    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    lr_artifact = _write_lr_artifact(
        tmp_path / "results" / "experiment_4126_lr_resume_correctness_fix.json",
        continuous=True,
    )

    blocked_precondition = exp4127.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "blocked_precondition.json",
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: None,
        cuda_checker=lambda: (True, "cuda fixture"),
    )
    assert blocked_precondition["honest_verdict"] == "blocked_nanotrm_or_uv_missing"

    missing_stable = exp4127.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "missing_stable.json",
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (False, "missing"),
    )
    assert missing_stable["honest_verdict"] == "blocked_stable_checkpoint_missing"

    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    dataset_missing = exp4127.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "dataset_missing.json",
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        dataset_builder=lambda _config: None,
    )
    assert dataset_missing["honest_verdict"] == "blocked_dataset_missing"
    assert dataset_missing["dataset_generated"] is True

    _make_dataset(tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k")
    _write_prior_artifact(tmp_path, val=0.105)

    def exploding_runner(
        _config: exp4127.Exp4127Config,
        _pass_index: int,
    ) -> exp4116.ResumeRunResult:
        raise RuntimeError("trainer exploded")

    trainer_error = exp4127.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "trainer_error.json",
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        dataset_builder=lambda _config: False,
        trainer_runner=exploding_runner,
    )
    assert trainer_error["honest_verdict"] == "complete: missing_real_val_trajectory"
    assert trainer_error["pass_results"][0]["return_code"] == 1
    assert "trainer exploded" in trainer_error["pass_results"][0]["stdout_tail"][0]
