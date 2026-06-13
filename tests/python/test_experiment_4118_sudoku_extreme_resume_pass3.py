"""Tests for Exp 4118 resumable nano-trm Sudoku Extreme pass 3.

Spec refs: REQ-LEARN-4118, SCENARIO-LEARN-4118,
SCENARIO-LEARN-4118-AUDIT, SCENARIO-LEARN-4118-CONFIRM.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116
from carnot import experiment_4118_sudoku_extreme_resume_pass3 as exp4118


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _make_ready_repo(tmp_path: Path) -> None:
    trainer = tmp_path / "nano-trm" / "src" / "nn" / "train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# trainer fixture\n", encoding="utf-8")
    experiment = tmp_path / "nano-trm" / "src" / "nn" / "configs" / "experiment"
    experiment.mkdir(parents=True, exist_ok=True)
    (experiment / "trm_sudoku_extreme_1k_aug_1k.yaml").write_text(
        "\n".join(
            [
                "model_tuning:",
                "  learning_rate: 1e-4",
                "  learning_rate_emb: 1e-4",
                "  warmup_steps: 2000",
                "timekeeping:",
                "  batch_size: 768",
                "trainer:",
                "  check_val_every_n_epoch: 100",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "nano-trm" / "README.md").write_text(
        "Sudoku Extreme training time ~1h on an H100 SXM5. You should get to ~87% exact accuracy.\n",
        encoding="utf-8",
    )


def _make_dataset(dataset_dir: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "dataset.json").write_text('{"num_puzzles": 1}\n', encoding="utf-8")
        for name in ("all__inputs.npy", "all__labels.npy", "all__puzzle_identifiers.npy"):
            (split_dir / name).write_bytes(f"{split}:{name}".encode("ascii"))
    (dataset_dir / "metadata.json").write_text('{"seed": 4118}\n', encoding="utf-8")


def _write_metrics(path: Path, *, val: float = 0.125, epoch: int = 4299) -> None:
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


def _pass2_artifact(
    path: Path,
    *,
    stable: Path,
    run_dir: Path,
    val: float | None,
    stalled: bool = False,
    cumulative_epochs: int | None = 4200,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: pass2 fixture",
                "stable_checkpoint_path": str(stable),
                "val_exact_accuracy": val,
                "accumulation_stalled": stalled,
                "cumulative_epochs": cumulative_epochs,
                "run_dir": str(run_dir),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_req_learn_4118_spec_declares_pass3_contract() -> None:
    """REQ-LEARN-4118: OpenSpec declares pass3 branches and fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4118" in spec
    assert "SCENARIO-LEARN-4118" in spec
    assert "SCENARIO-LEARN-4118-AUDIT" in spec
    assert "SCENARIO-LEARN-4118-CONFIRM" in spec
    assert "early-converged-confirm" in spec
    assert "config-audit" in spec
    for field in exp4118.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4118_pass2_context_recovers_val_and_branch(tmp_path: Path) -> None:
    """REQ-LEARN-4118: pass2 context drives train/confirm/audit decisions."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    run_dir = tmp_path / "results" / "trm_runs" / "experiment_4117_hydra"
    _write_metrics(run_dir / "csv" / "version_0" / "metrics.csv", val=0.09661458432674408)
    artifact_path = _pass2_artifact(
        tmp_path / "results" / "experiment_4117_sudoku_extreme_resume_pass2.json",
        stable=stable,
        run_dir=run_dir,
        val=None,
    )

    context = exp4118.load_pass2_context(artifact_path)

    assert exp4118.find_pass2_artifact(tmp_path) == artifact_path
    assert exp4118.find_pass2_artifact(tmp_path / "missing") == (
        tmp_path / "missing" / "results" / "experiment_4117_sudoku_extreme_resume_pass2.json"
    )
    assert context.artifact_path == artifact_path
    assert context.stable_checkpoint_path == stable
    assert context.val_exact_accuracy == 0.09661458432674408
    assert context.val_source == str(run_dir / "csv" / "version_0" / "metrics.csv")
    assert context.total_cumulative_epochs == 4200
    assert exp4118.decide_branch(context) == "train"

    confirm = exp4118.Pass2Context(
        artifact_path=artifact_path,
        stable_checkpoint_path=stable,
        val_exact_accuracy=0.86,
        val_source="artifact",
        accumulation_stalled=False,
        total_cumulative_epochs=4200,
        run_dir=run_dir,
    )
    assert exp4118.decide_branch(confirm) == "early-converged-confirm"
    assert exp4118.matches_published_087(0.86) is True

    stalled = exp4118.Pass2Context(
        artifact_path=artifact_path,
        stable_checkpoint_path=stable,
        val_exact_accuracy=0.08,
        val_source="artifact",
        accumulation_stalled=True,
        total_cumulative_epochs=None,
        run_dir=None,
    )
    assert exp4118.decide_branch(stalled) == "config-audit"
    assert exp4118._numeric_or_none(True) is None
    assert exp4118._numeric_or_none("0.1") is None
    assert exp4118._int_or_none(2.0) == 2
    assert exp4118._int_or_none("2") is None
    assert exp4118.matches_published_087(None) is False

    no_metric_artifact = _pass2_artifact(
        tmp_path / "results" / "experiment_4117_no_metric.json",
        stable=stable,
        run_dir=tmp_path / "missing_metrics",
        val=None,
        cumulative_epochs=None,
    )
    no_metric = exp4118.load_pass2_context(no_metric_artifact)
    assert no_metric.val_exact_accuracy is None
    assert no_metric.val_source is None
    assert no_metric.total_cumulative_epochs is None


def test_req_learn_4118_command_uses_stable_checkpoint_same_save_path_and_bound(tmp_path: Path) -> None:
    """REQ-LEARN-4118: pass3 training resumes and saves the stable checkpoint."""

    _make_ready_repo(tmp_path)
    config = exp4118.Exp4118Config(repo_root=tmp_path)
    config.stable_dir.mkdir(parents=True)
    config.stable_checkpoint_path.write_bytes(b"checkpoint")

    command = exp4118.build_train_command(config)

    assert config.trainer_path == tmp_path / "nano-trm" / "src" / "nn" / "train.py"
    assert command[:4] == ["uv", "run", "python", "src/nn/train.py"]
    assert "experiment=trm_sudoku_extreme_1k_aug_1k" in command
    assert f"ckpt_path={config.stable_checkpoint_path}" in command
    assert "+trainer.max_time=00:01:00:00" in command
    assert f"callbacks.model_checkpoint.dirpath={config.stable_dir}" in command
    assert "callbacks.model_checkpoint.save_last=true" in command
    assert f"hydra.run.dir={config.hydra_run_dir}" in command
    assert "experiment_4118_sudoku_extreme_resume_pass3_hydra" in str(config.hydra_run_dir)
    assert any(
        "carnot.experiment_4118_sudoku_extreme_resume_pass3.NanoTrmResumePass3ProgressPrinter" in item
        for item in command
    )


def test_scenario_learn_4118_training_artifact_reports_match_bool(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4118: train branch emits final val and 0.87 match bool."""

    _make_ready_repo(tmp_path)
    config = exp4118.Exp4118Config(repo_root=tmp_path)
    config.stable_dir.mkdir(parents=True)
    config.stable_checkpoint_path.write_bytes(b"checkpoint")
    pass2 = exp4118.Pass2Context(
        artifact_path=tmp_path / "results" / "experiment_4117.json",
        stable_checkpoint_path=config.stable_checkpoint_path,
        val_exact_accuracy=0.09661458432674408,
        val_source="artifact",
        accumulation_stalled=False,
        total_cumulative_epochs=4200,
        run_dir=config.hydra_run_dir.with_name("pass2"),
    )
    result = exp4116.ResumeRunResult(
        return_code=0,
        stable_checkpoint_path=config.stable_checkpoint_path,
        checkpoint_reload_ok=True,
        checkpoint_reload_detail="loadable fixture",
        val_exact_accuracy=exp4107.ExactAccuracy("val/exact_accuracy", 0.12, config.hydra_run_dir / "metrics.csv"),
        cumulative_epochs=4300,
        duration_s=3599.0,
        command=exp4118.build_train_command(config),
        stdout_tail=["val/exact_accuracy=0.12"],
        run_dir=config.hydra_run_dir,
    )

    artifact = exp4118.build_result_artifact(
        run_config=config,
        run_result=result,
        pass2_context=pass2,
        preconditions_checked=[],
        dataset_generated=False,
    )

    assert artifact["honest_verdict"] == "complete: val=0.1200 still_below_0.87 -> .382 continues"
    assert artifact["val_exact_accuracy"] == 0.12
    assert artifact["matches_published_087"] is False
    assert artifact["total_cumulative_epochs"] == 4300
    assert artifact["stable_checkpoint_path"] == str(config.stable_checkpoint_path)
    assert artifact["branch_taken"] == "train"
    assert artifact["acceptance_gate_passed"] is True
    assert exp4118.artifact_schema_errors(artifact) == []

    matched_result = exp4116.ResumeRunResult(
        **{**result.__dict__, "val_exact_accuracy": exp4107.ExactAccuracy("val/exact_accuracy", 0.86, config.hydra_run_dir / "metrics.csv")}
    )
    matched = exp4118.build_result_artifact(
        run_config=config,
        run_result=matched_result,
        pass2_context=pass2,
        preconditions_checked=[],
        dataset_generated=False,
    )
    assert matched["honest_verdict"] == "complete: val=0.8600 reproduced_within_0.02_of_0.87"
    assert matched["matches_published_087"] is True

    missing_exact = exp4118.build_result_artifact(
        run_config=config,
        run_result=exp4116.ResumeRunResult(
            **{
                **result.__dict__,
                "return_code": 3,
                "val_exact_accuracy": None,
                "cumulative_epochs": None,
            }
        ),
        pass2_context=pass2,
        preconditions_checked=[],
        dataset_generated=False,
    )
    assert missing_exact["honest_verdict"] == "complete: nanotrm_resume_pass3_failed_return_code_3"
    assert missing_exact["total_cumulative_epochs"] == 4200
    assert missing_exact["acceptance_gate_passed"] is False


def test_scenario_learn_4118_confirm_and_audit_artifacts_do_not_train(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4118-AUDIT/CONFIRM: non-train branches are terminal."""

    _make_ready_repo(tmp_path)
    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    config = exp4118.Exp4118Config(repo_root=tmp_path, stable_dir=stable.parent)
    confirm_context = exp4118.Pass2Context(
        artifact_path=tmp_path / "results" / "experiment_4117.json",
        stable_checkpoint_path=stable,
        val_exact_accuracy=0.86,
        val_source="artifact",
        accumulation_stalled=False,
        total_cumulative_epochs=4200,
        run_dir=None,
    )

    confirm = exp4118.build_early_converged_artifact(
        run_config=config,
        pass2_context=confirm_context,
        preconditions_checked=[],
        duration_s=1.0,
    )

    assert confirm["honest_verdict"] == "complete: val=0.8600 reproduced_within_0.02_of_0.87"
    assert confirm["branch_taken"] == "early-converged-confirm"
    assert confirm["matches_published_087"] is True
    assert confirm["total_cumulative_epochs"] == 4200
    assert exp4118.artifact_schema_errors(confirm) == []

    audit_context = exp4118.Pass2Context(
        artifact_path=tmp_path / "results" / "experiment_4117.json",
        stable_checkpoint_path=stable,
        val_exact_accuracy=0.08,
        val_source="artifact",
        accumulation_stalled=True,
        total_cumulative_epochs=4200,
        run_dir=tmp_path / "results" / "trm_runs" / "experiment_4117_hydra",
    )
    audit = exp4118.run_config_audit(config, audit_context)
    artifact = exp4118.build_config_audit_artifact(
        run_config=config,
        pass2_context=audit_context,
        audit=audit,
        preconditions_checked=[],
        duration_s=2.0,
    )

    assert artifact["branch_taken"] == "config-audit"
    assert artifact["val_exact_accuracy"] == 0.08
    assert artifact["matches_published_087"] is False
    assert artifact["acceptance_gate_passed"] is True
    assert "RTX 3090" in artifact["config_audit"]["likely_root_cause"]
    assert "batch_size" in artifact["config_audit"]["evidence"]
    assert exp4118.artifact_schema_errors(artifact) == []

    no_recipe_repo = tmp_path / "no_recipe"
    _make_ready_repo(no_recipe_repo)
    (no_recipe_repo / "nano-trm" / "README.md").write_text("No benchmark recipe here.\n", encoding="utf-8")
    no_recipe_config = exp4118.Exp4118Config(repo_root=no_recipe_repo)
    no_recipe_audit = exp4118.run_config_audit(no_recipe_config, audit_context)
    assert no_recipe_audit.evidence["readme_recipe"] is None


def test_req_learn_4118_blocked_and_schema_guards(tmp_path: Path) -> None:
    """REQ-LEARN-4118: blocked artifacts are honest and schema-checked."""

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    artifact = exp4118.build_blocked_artifact(
        "blocked_stable_checkpoint_missing",
        branch_taken="train",
        preconditions_checked=[exp4107.PreconditionCheck("stable_checkpoint", False, "missing")],
        stable_checkpoint_path=stable,
        duration_s=0.5,
    )

    assert artifact["honest_verdict"] == "blocked_stable_checkpoint_missing"
    assert artifact["val_exact_accuracy"] is None
    assert artifact["matches_published_087"] is False
    assert artifact["acceptance_gate_passed"] is False
    assert exp4118.artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad.update(
        {
            "honest_verdict": "not terminal",
            "val_exact_accuracy": "0.1",
            "matches_published_087": "false",
            "total_cumulative_epochs": -1,
            "stable_checkpoint_path": "not-stable.ckpt",
            "branch_taken": "other",
            "duration_s": False,
            "acceptance_gate_passed": True,
        }
    )
    errors = exp4118.artifact_schema_errors(bad)
    assert "honest_verdict must be terminal-prefixed or blocked" in errors
    assert "val_exact_accuracy must be numeric or null" in errors
    assert "matches_published_087 must be a bare bool" in errors
    assert "total_cumulative_epochs must be a non-negative int or null" in errors
    assert "stable_checkpoint_path must be the shared sudoku_extreme_baseline/last.ckpt path" in errors
    assert "branch_taken must be one of train, early-converged-confirm, config-audit" in errors
    assert "duration_s must be numeric" in errors
    assert "accepted artifact requires val_exact_accuracy unless config-audit has a likely root cause" in errors

    assert "missing required field honest_verdict" in exp4118.artifact_schema_errors({})
    non_string_verdict = dict(artifact)
    non_string_verdict["honest_verdict"] = 1
    assert "honest_verdict must be a string" in exp4118.artifact_schema_errors(non_string_verdict)
    out_of_range = dict(artifact)
    out_of_range["val_exact_accuracy"] = 2.0
    assert "val_exact_accuracy must be between 0 and 1" in exp4118.artifact_schema_errors(out_of_range)
    bad_gate_type = dict(artifact)
    bad_gate_type["acceptance_gate_passed"] = "yes"
    assert "acceptance_gate_passed must be a bare bool" in exp4118.artifact_schema_errors(bad_gate_type)

    too_long = dict(artifact)
    too_long.update(
        {
            "honest_verdict": "complete: val=0.1000 still_below_0.87 -> .382 continues",
            "val_exact_accuracy": 0.1,
            "duration_s": 4800.0,
            "acceptance_gate_passed": True,
        }
    )
    assert "accepted artifact requires duration_s < 4800" in exp4118.artifact_schema_errors(too_long)

    try:
        exp4118.validate_artifact(bad)
    except ValueError as exc:
        assert "honest_verdict must be terminal-prefixed or blocked" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("validate_artifact should reject schema errors")


def test_req_learn_4118_run_experiment_branches_and_writes_json(tmp_path: Path) -> None:
    """REQ-LEARN-4118: run_experiment chooses train, confirm, audit, or blocked."""

    _make_ready_repo(tmp_path)
    _make_dataset(tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k")
    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    run_dir = tmp_path / "results" / "trm_runs" / "experiment_4117_hydra"
    pass2_artifact = _pass2_artifact(
        tmp_path / "results" / "experiment_4117_sudoku_extreme_resume_pass2.json",
        stable=stable,
        run_dir=run_dir,
        val=0.09661458432674408,
    )

    def fake_loader(_path: Path) -> tuple[bool, str]:
        return True, "loadable fixture"

    def fake_runner(config: exp4118.Exp4118Config) -> exp4116.ResumeRunResult:
        _write_metrics(config.hydra_run_dir / "csv" / "version_0" / "metrics.csv", val=0.13, epoch=4399)
        return exp4116.verify_completed_resume_run(
            config,
            duration_s=12.0,
            return_code=0,
            command=exp4118.build_train_command(config),
            stdout_tail=["done"],
            checkpoint_loader=fake_loader,
        )

    output_path = tmp_path / "results" / exp4118.RESULT_FILENAME
    artifact = exp4118.run_experiment(
        repo_root=tmp_path,
        output_path=output_path,
        pass2_artifact_path=pass2_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=fake_loader,
        dataset_builder=lambda _config: False,
        trainer_runner=fake_runner,
    )

    assert artifact["branch_taken"] == "train"
    assert artifact["val_exact_accuracy"] == 0.13
    assert artifact["total_cumulative_epochs"] == 4400
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact

    trainer_calls = 0

    def forbidden_runner(_config: exp4118.Exp4118Config) -> exp4116.ResumeRunResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("non-train branches must not launch trainer")

    confirm_artifact_path = _pass2_artifact(
        tmp_path / "results" / "experiment_4117_confirm.json",
        stable=stable,
        run_dir=run_dir,
        val=0.86,
    )
    confirm = exp4118.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "confirm.json",
        pass2_artifact_path=confirm_artifact_path,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=fake_loader,
        dataset_builder=lambda _config: False,
        trainer_runner=forbidden_runner,
    )
    assert confirm["branch_taken"] == "early-converged-confirm"
    assert trainer_calls == 0

    audit_artifact_path = _pass2_artifact(
        tmp_path / "results" / "experiment_4117_stalled.json",
        stable=stable,
        run_dir=run_dir,
        val=0.07,
        stalled=True,
    )
    audit = exp4118.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "audit.json",
        pass2_artifact_path=audit_artifact_path,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=fake_loader,
        dataset_builder=lambda _config: False,
        trainer_runner=forbidden_runner,
    )
    assert audit["branch_taken"] == "config-audit"
    assert trainer_calls == 0

    blocked = exp4118.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "blocked.json",
        pass2_artifact_path=pass2_artifact,
        uv_resolver=lambda _name: None,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=fake_loader,
        dataset_builder=lambda _config: False,
        trainer_runner=forbidden_runner,
    )
    assert blocked["honest_verdict"] == "blocked_nanotrm_or_uv_missing"

    missing_pass2 = exp4118.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "missing_pass2.json",
        pass2_artifact_path=tmp_path / "results" / "does_not_exist.json",
        uv_resolver=lambda _name: None,
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=fake_loader,
        dataset_builder=lambda _config: False,
        trainer_runner=forbidden_runner,
    )
    assert missing_pass2["honest_verdict"] == "blocked_nanotrm_or_uv_missing"

    stable.unlink()
    missing_checkpoint = exp4118.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "missing_checkpoint.json",
        pass2_artifact_path=pass2_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (False, "missing"),
        dataset_builder=lambda _config: False,
        trainer_runner=forbidden_runner,
    )
    assert missing_checkpoint["honest_verdict"] == "blocked_stable_checkpoint_missing"

    dataset_missing_repo = tmp_path / "dataset_missing_repo"
    _make_ready_repo(dataset_missing_repo)
    dataset_missing_stable = (
        dataset_missing_repo / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    )
    dataset_missing_stable.parent.mkdir(parents=True, exist_ok=True)
    dataset_missing_stable.write_bytes(b"checkpoint")
    dataset_missing_pass2 = _pass2_artifact(
        dataset_missing_repo / "results" / "experiment_4117_sudoku_extreme_resume_pass2.json",
        stable=dataset_missing_stable,
        run_dir=dataset_missing_repo / "results" / "trm_runs" / "experiment_4117_hydra",
        val=0.1,
    )
    dataset_missing = exp4118.run_experiment(
        repo_root=dataset_missing_repo,
        output_path=dataset_missing_repo / "results" / "dataset_missing.json",
        pass2_artifact_path=dataset_missing_pass2,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=fake_loader,
        dataset_builder=lambda _config: None,
        trainer_runner=forbidden_runner,
    )
    assert dataset_missing["honest_verdict"] == "blocked_dataset_missing"
    assert dataset_missing["dataset_generated"] is True

    _make_dataset(dataset_missing_repo / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k")

    def raising_runner(_config: exp4118.Exp4118Config) -> exp4116.ResumeRunResult:
        raise RuntimeError("trainer exploded")

    trainer_error = exp4118.run_experiment(
        repo_root=dataset_missing_repo,
        output_path=dataset_missing_repo / "results" / "trainer_error.json",
        pass2_artifact_path=dataset_missing_pass2,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=fake_loader,
        dataset_builder=lambda _config: False,
        trainer_runner=raising_runner,
    )
    assert trainer_error["honest_verdict"] == "complete: nanotrm_resume_pass3_failed_return_code_1"
