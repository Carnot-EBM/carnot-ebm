"""Tests for Exp 4117 resumable nano-trm Sudoku Extreme pass 2.

Spec refs: REQ-LEARN-4117, SCENARIO-LEARN-4117,
SCENARIO-LEARN-4117-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116
from carnot import experiment_4117_sudoku_extreme_resume_pass2 as exp4117


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _make_ready_repo(tmp_path: Path) -> None:
    trainer = tmp_path / "nano-trm" / "src" / "nn" / "train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# trainer fixture\n", encoding="utf-8")
    builder = tmp_path / "nano-trm" / "scripts" / "data" / "build_sudoku_extreme_dataset.py"
    builder.parent.mkdir(parents=True, exist_ok=True)
    builder.write_text("# builder fixture\n", encoding="utf-8")


def _make_dataset(dataset_dir: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "dataset.json").write_text('{"num_puzzles": 1}\n', encoding="utf-8")
        for name in ("all__inputs.npy", "all__labels.npy", "all__puzzle_identifiers.npy"):
            (split_dir / name).write_bytes(f"{split}:{name}".encode("ascii"))
    (dataset_dir / "metadata.json").write_text('{"seed": 4117}\n', encoding="utf-8")


def _write_metrics(path: Path, *, val: float = 0.125, epoch: int = 4199) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "epoch,step,train/exact_accuracy,val/exact_accuracy,val/q_halt_accuracy",
                f"{epoch},{epoch},0.20,,",
                f"{epoch},{epoch},,{val},0.9",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _pass1_artifact(path: Path, *, stable: Path, run_dir: Path, val: float | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: val reported by pass1",
                "stable_checkpoint_path": str(stable),
                "val_exact_accuracy": val,
                "run_dir": str(run_dir),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_req_learn_4117_spec_declares_accumulation_floor_contract() -> None:
    """REQ-LEARN-4117: OpenSpec declares pass2 delta and stall fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4117" in spec
    assert "SCENARIO-LEARN-4117" in spec
    assert "SCENARIO-LEARN-4117-BLOCKED" in spec
    assert "blocked_stable_checkpoint_missing" in spec
    for field in exp4117.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4117_pass1_context_recovers_val_from_run_dir(tmp_path: Path) -> None:
    """REQ-LEARN-4117: pass1 val comes from artifact or its linked CSV metrics."""

    run_dir = tmp_path / "results" / "trm_runs" / "experiment_4116_hydra"
    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    _write_metrics(run_dir / "csv" / "version_1" / "metrics.csv", val=0.08541666716337204)
    artifact_path = _pass1_artifact(
        tmp_path / "results" / "experiment_4116_sudoku_extreme_resume_pass1.json",
        stable=stable,
        run_dir=run_dir,
        val=None,
    )

    context = exp4117.load_pass1_context(artifact_path)

    assert exp4117.find_pass1_artifact(tmp_path) == artifact_path
    assert exp4117.find_pass1_artifact(tmp_path / "no_artifacts") == (
        tmp_path / "no_artifacts" / "results" / exp4116.RESULT_FILENAME
    )
    assert exp4117._numeric_or_none("not numeric") is None
    assert context.artifact_path == artifact_path
    assert context.stable_checkpoint_path == stable
    assert context.val_exact_accuracy == 0.08541666716337204
    assert context.val_source == str(run_dir / "csv" / "version_1" / "metrics.csv")

    direct_artifact = _pass1_artifact(artifact_path, stable=stable, run_dir=run_dir, val=0.2)
    direct_context = exp4117.load_pass1_context(direct_artifact)
    assert direct_context.val_exact_accuracy == 0.2
    assert direct_context.val_source == str(direct_artifact)

    missing_metrics = _pass1_artifact(
        tmp_path / "results" / "experiment_4116_missing_metrics.json",
        stable=stable,
        run_dir=tmp_path / "empty_pass1",
        val=None,
    )
    missing_context = exp4117.load_pass1_context(missing_metrics)
    assert missing_context.val_exact_accuracy is None
    assert missing_context.val_source is None


def test_req_learn_4117_command_uses_stable_checkpoint_same_save_path_and_bound(tmp_path: Path) -> None:
    """REQ-LEARN-4117: pass2 resumes from and saves back to the stable checkpoint."""

    _make_ready_repo(tmp_path)
    config = exp4117.Exp4117Config(repo_root=tmp_path)
    config.stable_dir.mkdir(parents=True)
    config.stable_checkpoint_path.write_bytes(b"checkpoint")

    command = exp4117.build_train_command(config)

    assert config.trainer_path == tmp_path / "nano-trm" / "src" / "nn" / "train.py"
    assert config.dataset_builder_path == (
        tmp_path / "nano-trm" / "scripts" / "data" / "build_sudoku_extreme_dataset.py"
    )
    assert command[:4] == ["uv", "run", "python", "src/nn/train.py"]
    assert "experiment=trm_sudoku_extreme_1k_aug_1k" in command
    assert f"ckpt_path={config.stable_checkpoint_path}" in command
    assert "+trainer.max_time=00:01:00:00" in command
    assert f"callbacks.model_checkpoint.dirpath={config.stable_dir}" in command
    assert "callbacks.model_checkpoint.save_last=true" in command
    assert f"hydra.run.dir={config.hydra_run_dir}" in command
    assert "experiment_4117_sudoku_extreme_resume_pass2_hydra" in str(config.hydra_run_dir)
    assert any(
        "carnot.experiment_4117_sudoku_extreme_resume_pass2.NanoTrmResumePass2ProgressPrinter" in item
        for item in command
    )


def test_scenario_learn_4117_artifact_reports_delta_improvement_and_stall(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4117: artifact computes delta and the accumulate floor."""

    _make_ready_repo(tmp_path)
    config = exp4117.Exp4117Config(repo_root=tmp_path)
    config.stable_dir.mkdir(parents=True)
    config.stable_checkpoint_path.write_bytes(b"checkpoint")
    pass1 = exp4117.Pass1Context(
        artifact_path=tmp_path / "results" / "experiment_4116.json",
        stable_checkpoint_path=config.stable_checkpoint_path,
        val_exact_accuracy=0.08541666716337204,
        val_source="artifact",
        run_dir=config.hydra_run_dir.with_name("pass1"),
    )
    exact = exp4107.ExactAccuracy("val/exact_accuracy", 0.125, config.hydra_run_dir / "metrics.csv")
    result = exp4116.ResumeRunResult(
        return_code=0,
        stable_checkpoint_path=config.stable_checkpoint_path,
        checkpoint_reload_ok=True,
        checkpoint_reload_detail="loadable fixture",
        val_exact_accuracy=exact,
        cumulative_epochs=4200,
        duration_s=3599.0,
        command=exp4117.build_train_command(config),
        stdout_tail=["val/exact_accuracy=0.125"],
        run_dir=config.hydra_run_dir,
    )

    artifact = exp4117.build_result_artifact(
        run_config=config,
        run_result=result,
        pass1_context=pass1,
        preconditions_checked=[],
        dataset_generated=False,
    )

    assert artifact["honest_verdict"] == "complete: val=0.1250 delta=0.0396 improved"
    assert artifact["val_exact_accuracy"] == 0.125
    assert artifact["val_delta_vs_pass1"] == pytest.approx(0.03958333283662796)
    assert artifact["accumulation_stalled"] is False
    assert artifact["cumulative_epochs"] == 4200
    assert artifact["acceptance_gate_passed"] is True
    assert exp4117.artifact_schema_errors(artifact) == []

    stalled_result = exp4116.ResumeRunResult(**{**result.__dict__, "val_exact_accuracy": exp4107.ExactAccuracy("val/exact_accuracy", 0.08, config.hydra_run_dir / "metrics.csv")})
    stalled = exp4117.build_result_artifact(
        run_config=config,
        run_result=stalled_result,
        pass1_context=pass1,
        preconditions_checked=[],
        dataset_generated=False,
    )
    assert stalled["val_delta_vs_pass1"] == pytest.approx(-0.005416667163372033)
    assert stalled["accumulation_stalled"] is True
    assert stalled["acceptance_gate_passed"] is True
    assert stalled["honest_verdict"] == "complete: val=0.0800 delta=-0.0054 accumulation_stalled_config_audit_recommended"

    no_pass1_delta = exp4117.build_result_artifact(
        run_config=config,
        run_result=result,
        pass1_context=exp4117.Pass1Context(
            artifact_path=pass1.artifact_path,
            stable_checkpoint_path=config.stable_checkpoint_path,
            val_exact_accuracy=None,
            val_source=None,
            run_dir=None,
        ),
        preconditions_checked=[],
        dataset_generated=False,
    )
    assert no_pass1_delta["honest_verdict"] == (
        "complete: val=0.1250 pass1_delta_unavailable_config_audit_recommended"
    )
    assert no_pass1_delta["acceptance_gate_passed"] is False

    for return_code, expected in (
        (0, "complete: missing_real_val_exact_accuracy"),
        (3, "complete: nanotrm_resume_pass2_failed_return_code_3"),
    ):
        missing_exact_result = exp4116.ResumeRunResult(
            **{**result.__dict__, "return_code": return_code, "val_exact_accuracy": None}
        )
        missing_exact = exp4117.build_result_artifact(
            run_config=config,
            run_result=missing_exact_result,
            pass1_context=pass1,
            preconditions_checked=[],
            dataset_generated=False,
        )
        assert missing_exact["honest_verdict"] == expected


def test_req_learn_4117_blocked_and_schema_guards(tmp_path: Path) -> None:
    """REQ-LEARN-4117: blocked artifacts are honest and schema guards the gate."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    artifact = exp4117.build_blocked_artifact(
        "blocked_stable_checkpoint_missing",
        preconditions_checked=[exp4107.PreconditionCheck("stable_checkpoint", False, "missing")],
        stable_checkpoint_path=stable,
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "blocked_stable_checkpoint_missing"
    assert artifact["val_exact_accuracy"] is None
    assert artifact["val_delta_vs_pass1"] is None
    assert artifact["accumulation_stalled"] is False
    assert artifact["acceptance_gate_passed"] is False
    assert exp4117.artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad.update(
        {
            "honest_verdict": "not terminal",
            "val_exact_accuracy": True,
            "val_delta_vs_pass1": "flat",
            "accumulation_stalled": "true",
            "duration_s": False,
            "acceptance_gate_passed": True,
        }
    )
    errors = exp4117.artifact_schema_errors(bad)
    assert "honest_verdict must be terminal-prefixed or blocked" in errors
    assert "val_exact_accuracy must be numeric or null" in errors
    assert "val_delta_vs_pass1 must be numeric or null" in errors
    assert "accumulation_stalled must be a bare bool" in errors
    assert "duration_s must be numeric" in errors
    assert "accepted artifact requires val_exact_accuracy" in errors
    assert "accepted artifact requires positive delta or accumulation_stalled true" in errors

    assert "missing required field honest_verdict" in exp4117.artifact_schema_errors({})
    non_string_verdict = dict(artifact)
    non_string_verdict["honest_verdict"] = 1
    assert "honest_verdict must be a string" in exp4117.artifact_schema_errors(non_string_verdict)
    out_of_range = dict(artifact)
    out_of_range["val_exact_accuracy"] = 2.0
    assert "val_exact_accuracy must be between 0 and 1" in exp4117.artifact_schema_errors(out_of_range)
    bad_epoch = dict(artifact)
    bad_epoch["cumulative_epochs"] = -1
    assert "cumulative_epochs must be a non-negative int or null" in exp4117.artifact_schema_errors(bad_epoch)
    bad_stable = dict(artifact)
    bad_stable["stable_checkpoint_path"] = "not-stable.ckpt"
    assert "stable_checkpoint_path must be the shared sudoku_extreme_baseline/last.ckpt path" in (
        exp4117.artifact_schema_errors(bad_stable)
    )
    bad_gate_type = dict(artifact)
    bad_gate_type["acceptance_gate_passed"] = "yes"
    assert "acceptance_gate_passed must be a bare bool" in exp4117.artifact_schema_errors(bad_gate_type)

    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    too_long = dict(artifact)
    too_long.update(
        {
            "honest_verdict": "complete: val=0.2000 delta=0.1000 improved",
            "val_exact_accuracy": 0.2,
            "val_delta_vs_pass1": 0.1,
            "duration_s": 4800.0,
            "acceptance_gate_passed": True,
        }
    )
    assert "accepted artifact requires duration_s < 4800" in exp4117.artifact_schema_errors(too_long)

    try:
        exp4117.validate_artifact(bad)
    except ValueError as exc:
        assert "honest_verdict must be terminal-prefixed or blocked" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("validate_artifact should reject schema errors")


def test_req_learn_4117_run_experiment_blocks_or_writes_measured_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-4117: run_experiment stops on broken stable ckpt or writes pass2."""

    _make_ready_repo(tmp_path)
    dataset_dir = tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k"
    _make_dataset(dataset_dir)
    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    pass1_run = tmp_path / "results" / "trm_runs" / "experiment_4116_hydra"
    pass1_artifact = _pass1_artifact(
        tmp_path / "results" / "experiment_4116_sudoku_extreme_resume_pass1.json",
        stable=stable,
        run_dir=pass1_run,
        val=0.08541666716337204,
    )

    trainer_calls = 0

    def forbidden_runner(_config: exp4117.Exp4117Config) -> exp4116.ResumeRunResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("must not run without a stable checkpoint")

    blocked = exp4117.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "blocked.json",
        pass1_artifact_path=pass1_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (False, "missing"),
        dataset_builder=lambda _config: False,
        trainer_runner=forbidden_runner,
    )

    assert blocked["honest_verdict"] == "blocked_stable_checkpoint_missing"
    assert trainer_calls == 0

    precondition_blocked = exp4117.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "precondition_blocked.json",
        pass1_artifact_path=pass1_artifact,
        uv_resolver=lambda _name: None,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable"),
        dataset_builder=lambda _config: False,
        trainer_runner=forbidden_runner,
    )
    assert precondition_blocked["honest_verdict"] == "blocked_nanotrm_or_uv_missing"

    missing_pass1 = exp4117.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "missing_pass1.json",
        pass1_artifact_path=tmp_path / "results" / "missing_4116.json",
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (False, "missing"),
        dataset_builder=lambda _config: False,
        trainer_runner=forbidden_runner,
    )
    assert missing_pass1["honest_verdict"] == "blocked_stable_checkpoint_missing"

    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    output_path = tmp_path / "results" / exp4117.RESULT_FILENAME

    def fake_loader(_path: Path) -> tuple[bool, str]:
        return True, "loadable fixture"

    def fake_runner(config: exp4117.Exp4117Config) -> exp4116.ResumeRunResult:
        _write_metrics(config.hydra_run_dir / "csv" / "version_0" / "metrics.csv", val=0.09, epoch=4299)
        return exp4116.verify_completed_resume_run(
            config,
            duration_s=12.0,
            return_code=0,
            command=exp4117.build_train_command(config),
            stdout_tail=["done"],
            checkpoint_loader=fake_loader,
        )

    artifact = exp4117.run_experiment(
        repo_root=tmp_path,
        output_path=output_path,
        pass1_artifact_path=pass1_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=fake_loader,
        dataset_builder=lambda _config: False,
        trainer_runner=fake_runner,
    )

    assert artifact["honest_verdict"] == "complete: val=0.0900 delta=0.0046 improved"
    assert artifact["val_exact_accuracy"] == 0.09
    assert artifact["val_delta_vs_pass1"] == pytest.approx(0.004583332836627956)
    assert artifact["accumulation_stalled"] is False
    assert artifact["stable_checkpoint_path"] == str(stable)
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact

    missing_repo = tmp_path / "dataset_missing_repo"
    _make_ready_repo(missing_repo)
    missing_stable = missing_repo / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    missing_stable.parent.mkdir(parents=True, exist_ok=True)
    missing_stable.write_bytes(b"checkpoint")
    missing_pass1_artifact = _pass1_artifact(
        missing_repo / "results" / "experiment_4116_sudoku_extreme_resume_pass1.json",
        stable=missing_stable,
        run_dir=missing_repo / "results" / "trm_runs" / "experiment_4116_hydra",
        val=0.1,
    )
    dataset_missing = exp4117.run_experiment(
        repo_root=missing_repo,
        output_path=missing_repo / "results" / "dataset_missing.json",
        pass1_artifact_path=missing_pass1_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        dataset_builder=lambda _config: None,
        trainer_runner=forbidden_runner,
    )
    assert dataset_missing["honest_verdict"] == "blocked_dataset_missing"
    assert dataset_missing["dataset_generated"] is True
