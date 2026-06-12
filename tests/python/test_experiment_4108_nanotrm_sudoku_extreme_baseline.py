"""Tests for Exp 4108 nano-trm Sudoku Extreme baseline.

Spec refs: REQ-LEARN-4108, SCENARIO-LEARN-4108,
SCENARIO-LEARN-4108-SHORT.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4108_nanotrm_sudoku_extreme_baseline as exp4108


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _make_ready_repo(tmp_path: Path) -> Path:
    trainer = tmp_path / "nano-trm" / "src" / "nn" / "train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# trainer fixture\n", encoding="utf-8")
    builder = tmp_path / "nano-trm" / "scripts" / "data" / "build_sudoku_extreme_dataset.py"
    builder.parent.mkdir(parents=True, exist_ok=True)
    builder.write_text("# builder fixture\n", encoding="utf-8")
    config = tmp_path / "nano-trm" / "src" / "nn" / "configs" / "experiment"
    config.mkdir(parents=True, exist_ok=True)
    (config / "trm_sudoku_extreme_1k_aug_1k.yaml").write_text("trainer:\n  max_epochs: 50000\n", encoding="utf-8")
    save_parent = tmp_path / "results" / "trm_runs"
    save_parent.mkdir(parents=True, exist_ok=True)
    return save_parent


def _make_dataset(dataset_dir: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "dataset.json").write_text('{"num_puzzles": 1}\n', encoding="utf-8")
        for name in (
            "all__inputs.npy",
            "all__labels.npy",
            "all__puzzle_identifiers.npy",
            "all__puzzle_indices.npy",
            "all__group_indices.npy",
        ):
            (split_dir / name).write_bytes(f"{split}:{name}".encode("ascii"))
    (dataset_dir / "metadata.json").write_text('{"seed": 42, "num_train": 1001}\n', encoding="utf-8")


def _exp4107_artifact(path: Path, *, checkpoint_ok: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: fixture",
                "nanotrm_trainer_checkpoint_ok": checkpoint_ok,
                "checkpoint_path": "/tmp/fixture.ckpt" if checkpoint_ok else None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_req_learn_4108_spec_declares_required_contract() -> None:
    """REQ-LEARN-4108: OpenSpec declares the baseline artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4108" in spec
    assert "SCENARIO-LEARN-4108" in spec
    assert "SCENARIO-LEARN-4108-SHORT" in spec
    for field in exp4108.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    assert "matches_published_087" in spec
    assert "reproducibility_checksum" in spec
    assert "shorter reproduction mode" in spec


def test_req_learn_4108_dataset_presence_and_commands(tmp_path: Path) -> None:
    """REQ-LEARN-4108: dataset builder and trainer commands use native nano-trm paths."""

    save_parent = _make_ready_repo(tmp_path)
    config = exp4108.NanoTrmExtremeRunConfig(
        repo_root=tmp_path,
        save_parent=save_parent,
        save_dir=save_parent / "run",
    )

    assert config.trainer_path == tmp_path / "nano-trm" / "src" / "nn" / "train.py"
    assert config.dataset_builder_path == (
        tmp_path / "nano-trm" / "scripts" / "data" / "build_sudoku_extreme_dataset.py"
    )
    assert exp4108.dataset_is_complete(config.dataset_dir) is False
    Path(config.dataset_dir).mkdir(parents=True)
    (Path(config.dataset_dir) / "metadata.json").write_text("{}\n", encoding="utf-8")
    assert exp4108.dataset_is_complete(config.dataset_dir) is False
    (Path(config.dataset_dir) / "train").mkdir()
    (Path(config.dataset_dir) / "train" / "dataset.json").write_text("{}\n", encoding="utf-8")
    assert exp4108.dataset_is_complete(config.dataset_dir) is False
    _make_dataset(Path(config.dataset_dir))
    assert exp4108.dataset_is_complete(config.dataset_dir) is True

    dataset_command = exp4108.build_dataset_command(config)
    assert dataset_command == [
        "uv",
        "run",
        "python",
        "scripts/data/build_sudoku_extreme_dataset.py",
        "--output-dir",
        "./data/sudoku_extreme_1k_aug_1k",
        "--subsample-size",
        "1000",
        "--num-aug",
        "1000",
        "--eval-ratio",
        "0.01",
    ]

    train_command = exp4108.build_train_command(config)
    assert "experiment=trm_sudoku_extreme_1k_aug_1k" in train_command
    assert f"save_dir={config.save_dir}" in train_command
    assert "append_wandb_name_to_save_dir=false" in train_command
    assert f"seed={exp4108.RANDOM_SEED}" in train_command
    assert f"+callbacks.exp4108_progress.checkpoint_dir={config.hydra_run_dir / 'checkpoints'}" in train_command

    short_command = exp4108.build_train_command(
        exp4108.NanoTrmExtremeRunConfig(
            repo_root=tmp_path,
            save_parent=save_parent,
            save_dir=save_parent / "short",
            shorter_attempt=True,
        )
    )
    assert "timekeeping.max_epochs=100" in short_command
    assert "trainer.check_val_every_n_epoch=10" in short_command


def test_req_learn_4108_progress_print_pipe_closure_is_nonfatal() -> None:
    """REQ-LEARN-4108: progress printing is best-effort and cannot abort training."""

    calls: list[tuple[str, bool]] = []

    def ok_printer(message: str, *, flush: bool) -> None:
        calls.append((message, flush))

    def closed_pipe(_message: str, *, flush: bool) -> None:
        assert flush is True
        raise BrokenPipeError("closed")

    exp4108._safe_progress_print("hello", printer=ok_printer)
    exp4108._safe_progress_print("ignored", printer=closed_pipe)

    assert calls == [("hello", True)]


def test_req_learn_4108_exp4107_status_and_save_dir_helpers(tmp_path: Path) -> None:
    """REQ-LEARN-4108: prior mechanism status and fresh save dirs are deterministic."""

    missing = exp4108.load_exp4107_status(tmp_path / "missing.json")
    assert missing.checkpoint_ok is False
    assert missing.honest_verdict == "missing_exp4107_artifact"

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{not json\n", encoding="utf-8")
    invalid = exp4108.load_exp4107_status(invalid_path)
    assert invalid.checkpoint_ok is False
    assert invalid.honest_verdict.startswith("invalid_exp4107_artifact")

    ok_path = _exp4107_artifact(tmp_path / "ok.json", checkpoint_ok=True)
    ok = exp4108.load_exp4107_status(ok_path)
    assert ok.checkpoint_ok is True
    assert ok.to_dict()["artifact_path"] == str(ok_path)

    save_parent = tmp_path / "results" / "trm_runs"
    save_parent.mkdir(parents=True)
    first = exp4108._fresh_save_dir(save_parent, shorter_attempt=False)
    assert first.name == "experiment_4108_nanotrm_sudoku_extreme_baseline"
    first.mkdir()
    assert exp4108._fresh_save_dir(save_parent, shorter_attempt=False).name.endswith("_001")
    assert exp4108._fresh_save_dir(save_parent, shorter_attempt=True).name.endswith("_short")


def test_scenario_learn_4108_checksum_changes_with_dataset_or_config(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4108: checksum catches dataset and native config drift."""

    save_parent = _make_ready_repo(tmp_path)
    config = exp4108.NanoTrmExtremeRunConfig(repo_root=tmp_path, save_parent=save_parent)
    _make_dataset(Path(config.dataset_dir))

    first = exp4108.compute_reproducibility_checksum(config)
    second = exp4108.compute_reproducibility_checksum(config)
    assert first == second
    assert first.startswith("sha256:")

    (Path(config.dataset_dir) / "metadata.json").write_text('{"seed": 43}\n', encoding="utf-8")
    assert exp4108.compute_reproducibility_checksum(config) != first

    changed_dataset = exp4108.compute_reproducibility_checksum(config)
    Path(config.experiment_config_path).write_text("trainer:\n  max_epochs: 1\n", encoding="utf-8")
    assert exp4108.compute_reproducibility_checksum(config) != changed_dataset


def test_scenario_learn_4108_success_artifact_reports_real_accuracy(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4108: real val exact accuracy is the load-bearing number."""

    save_parent = _make_ready_repo(tmp_path)
    config = exp4108.NanoTrmExtremeRunConfig(repo_root=tmp_path, save_parent=save_parent)
    _make_dataset(Path(config.dataset_dir))
    checkpoint = save_parent / "run" / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    exact = exp4107.ExactAccuracy("val/exact_accuracy", 0.8725, save_parent / "run" / "metrics.csv")
    run = exp4107.NanoTrmRunResult(
        return_code=0,
        checkpoint_path=checkpoint,
        checkpoint_reload_ok=True,
        exact_accuracy=exact,
        duration_s=3725.4,
        command=exp4108.build_train_command(config),
        stdout_tail=["val/exact_accuracy=0.8725"],
        save_dir=save_parent / "run",
    )

    artifact = exp4108.build_result_artifact(
        run_config=config,
        run_result=run,
        mechanism_status=exp4108.Exp4107Status(
            artifact_path=tmp_path / "results" / "experiment_4107_nanotrm_mechanism_smoke.json",
            checkpoint_ok=True,
            honest_verdict="complete: fixture",
        ),
        dataset_generated=False,
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproduced_exact_accuracy"] == 0.8725
    assert artifact["matches_published_087"] is True
    assert artifact["checkpoint_path"] == str(checkpoint)
    assert artifact["duration_s"] == 3725.4
    assert artifact["random_seed"] == exp4108.RANDOM_SEED
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert artifact["acceptance_gate_passed"] is True
    assert exp4108.artifact_schema_errors(artifact) == []


def test_req_learn_4108_failure_artifacts_are_still_terminal(tmp_path: Path) -> None:
    """REQ-LEARN-4108: failed runs produce terminal honest artifacts."""

    save_parent = _make_ready_repo(tmp_path)
    config = exp4108.NanoTrmExtremeRunConfig(repo_root=tmp_path, save_parent=save_parent)
    _make_dataset(Path(config.dataset_dir))
    status = exp4108.Exp4107Status(
        artifact_path=tmp_path / "results" / "experiment_4107_nanotrm_mechanism_smoke.json",
        checkpoint_ok=True,
        honest_verdict="complete: fixture",
    )

    failed = exp4108.build_result_artifact(
        run_config=config,
        run_result=exp4107.NanoTrmRunResult(
            return_code=2,
            checkpoint_path=None,
            checkpoint_reload_ok=False,
            exact_accuracy=None,
            duration_s=12.0,
            command=exp4108.build_train_command(config),
            stdout_tail=["failed"],
            save_dir=save_parent / "failed",
        ),
        mechanism_status=status,
        dataset_generated=False,
    )
    assert failed["honest_verdict"] == "complete: nanotrm_sudoku_extreme_training_failed_return_code_2"
    assert failed["acceptance_gate_passed"] is False

    no_checkpoint = exp4108.build_result_artifact(
        run_config=config,
        run_result=exp4107.NanoTrmRunResult(
            return_code=0,
            checkpoint_path=None,
            checkpoint_reload_ok=False,
            exact_accuracy=exp4107.ExactAccuracy("val/exact_accuracy", 0.5, save_parent / "metrics.csv"),
            duration_s=12.0,
            command=[],
            stdout_tail=[],
            save_dir=save_parent / "no_checkpoint",
        ),
        mechanism_status=status,
        dataset_generated=False,
    )
    assert no_checkpoint["honest_verdict"] == (
        "complete: nanotrm_sudoku_extreme_checkpoint_missing_or_reload_failed"
    )

    interrupted_checkpoint = save_parent / "interrupted" / "checkpoints" / "last.ckpt"
    interrupted_checkpoint.parent.mkdir(parents=True)
    interrupted_checkpoint.write_bytes(b"checkpoint")
    interrupted = exp4108.build_result_artifact(
        run_config=config,
        run_result=exp4107.NanoTrmRunResult(
            return_code=124,
            checkpoint_path=interrupted_checkpoint,
            checkpoint_reload_ok=True,
            exact_accuracy=exp4107.ExactAccuracy(
                "val/exact_accuracy", 0.25, save_parent / "interrupted" / "metrics.csv"
            ),
            duration_s=3600.0,
            command=[],
            stdout_tail=["timeout"],
            save_dir=save_parent / "interrupted",
        ),
        mechanism_status=status,
        dataset_generated=False,
    )
    assert interrupted["honest_verdict"] == "complete: interrupted_return_code_124_reproduced_0.2500"
    assert interrupted["acceptance_gate_passed"] is True

    checkpoint = save_parent / "no_exact" / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    no_exact = exp4108.build_result_artifact(
        run_config=config,
        run_result=exp4107.NanoTrmRunResult(
            return_code=0,
            checkpoint_path=checkpoint,
            checkpoint_reload_ok=True,
            exact_accuracy=None,
            duration_s=12.0,
            command=[],
            stdout_tail=[],
            save_dir=save_parent / "no_exact",
        ),
        mechanism_status=status,
        dataset_generated=False,
    )
    assert no_exact["honest_verdict"] == "complete: nanotrm_sudoku_extreme_missing_real_val_exact_accuracy"


def test_scenario_learn_4108_below_target_is_complete_but_not_match(tmp_path: Path) -> None:
    """REQ-LEARN-4108: below-target exact accuracy is complete honest evidence."""

    save_parent = _make_ready_repo(tmp_path)
    config = exp4108.NanoTrmExtremeRunConfig(repo_root=tmp_path, save_parent=save_parent)
    _make_dataset(Path(config.dataset_dir))
    checkpoint = save_parent / "run" / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    run = exp4107.NanoTrmRunResult(
        return_code=0,
        checkpoint_path=checkpoint,
        checkpoint_reload_ok=True,
        exact_accuracy=exp4107.ExactAccuracy("val/exact_accuracy", 0.81, save_parent / "run" / "metrics.csv"),
        duration_s=611.0,
        command=exp4108.build_train_command(config),
        stdout_tail=[],
        save_dir=save_parent / "run",
    )

    artifact = exp4108.build_result_artifact(
        run_config=config,
        run_result=run,
        mechanism_status=exp4108.Exp4107Status(
            artifact_path=tmp_path / "results" / "experiment_4107_nanotrm_mechanism_smoke.json",
            checkpoint_ok=True,
            honest_verdict="complete: fixture",
        ),
        dataset_generated=True,
    )

    assert artifact["honest_verdict"] == "complete: reproduced_0.8100_below_published_0.87"
    assert artifact["matches_published_087"] is False
    assert artifact["acceptance_gate_passed"] is True
    assert exp4108.artifact_schema_errors(artifact) == []


def test_scenario_learn_4108_short_attempt_when_mechanism_unproven(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4108-SHORT: unproven 4107 status forces shorter mode."""

    save_parent = _make_ready_repo(tmp_path)
    exp4107_path = _exp4107_artifact(
        tmp_path / "results" / "experiment_4107_nanotrm_mechanism_smoke.json",
        checkpoint_ok=False,
    )
    _make_dataset(tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k")
    seen: dict[str, object] = {}

    def fake_runner(config: exp4108.NanoTrmExtremeRunConfig) -> exp4107.NanoTrmRunResult:
        seen["shorter_attempt"] = config.shorter_attempt
        seen["command"] = exp4108.build_train_command(config)
        checkpoint = Path(config.save_dir) / "checkpoints" / "last.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        return exp4107.NanoTrmRunResult(
            return_code=0,
            checkpoint_path=checkpoint,
            checkpoint_reload_ok=True,
            exact_accuracy=exp4107.ExactAccuracy("val/exact_accuracy", 0.25, Path(config.save_dir) / "metrics.csv"),
            duration_s=91.0,
            command=exp4108.build_train_command(config),
            stdout_tail=["short"],
            save_dir=Path(config.save_dir),
        )

    output_path = tmp_path / "results" / exp4108.RESULT_FILENAME
    artifact = exp4108.run_experiment(
        repo_root=tmp_path,
        output_path=output_path,
        save_parent=save_parent,
        exp4107_artifact_path=exp4107_path,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        dataset_builder=lambda _config: None,
        trainer_runner=fake_runner,
    )

    assert seen["shorter_attempt"] is True
    assert "timekeeping.max_epochs=100" in seen["command"]
    assert artifact["shorter_attempt"] is True
    assert artifact["mechanism_checkpoint_ok"] is False
    assert artifact["matches_published_087"] is False
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_req_learn_4108_run_experiment_blocks_and_handles_runner_failure(tmp_path: Path) -> None:
    """REQ-LEARN-4108: run_experiment writes honest blocked and failed artifacts."""

    save_parent = _make_ready_repo(tmp_path)
    exp4107_path = _exp4107_artifact(
        tmp_path / "results" / "experiment_4107_nanotrm_mechanism_smoke.json",
        checkpoint_ok=True,
    )
    blocked_output = tmp_path / "results" / "blocked.json"
    blocked = exp4108.run_experiment(
        repo_root=tmp_path,
        output_path=blocked_output,
        save_parent=save_parent,
        exp4107_artifact_path=exp4107_path,
        uv_resolver=lambda _name: None,
        cuda_checker=lambda: (True, "cuda fixture"),
    )
    assert blocked["honest_verdict"] == "blocked_nanotrm_or_uv_missing"
    assert blocked["reproduced_exact_accuracy"] is None
    assert json.loads(blocked_output.read_text(encoding="utf-8")) == blocked

    generated: list[Path] = []

    def fake_builder(config: exp4108.NanoTrmExtremeRunConfig) -> None:
        generated.append(Path(config.dataset_dir))
        _make_dataset(Path(config.dataset_dir))

    def failing_runner(_config: exp4108.NanoTrmExtremeRunConfig) -> exp4107.NanoTrmRunResult:
        raise RuntimeError("trainer fixture failed")

    failed_output = tmp_path / "results" / "failed.json"
    failed = exp4108.run_experiment(
        repo_root=tmp_path,
        output_path=failed_output,
        save_parent=save_parent,
        exp4107_artifact_path=exp4107_path,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        dataset_builder=fake_builder,
        trainer_runner=failing_runner,
    )
    assert generated == [tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k"]
    assert failed["dataset_generated"] is True
    assert failed["honest_verdict"] == "complete: nanotrm_sudoku_extreme_training_failed_return_code_1"
    assert "RuntimeError: trainer fixture failed" in failed["stdout_tail"]


def test_req_learn_4108_schema_rejects_fabrication_signals(tmp_path: Path) -> None:
    """REQ-LEARN-4108: schema requires real accuracy, bool match flag, and checkpoint."""

    artifact = {
        "honest_verdict": "complete: fabricated",
        "reproduced_exact_accuracy": None,
        "matches_published_087": "false",
        "checkpoint_path": str(tmp_path / "missing.ckpt"),
        "duration_s": 1.0,
        "random_seed": exp4108.RANDOM_SEED,
        "reproducibility_checksum": "not-a-hash",
        "acceptance_gate_passed": True,
    }

    errors = exp4108.artifact_schema_errors(artifact)

    assert "matches_published_087 must be a bare bool" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors
    assert "accepted artifact requires reproduced_exact_accuracy" in errors
    assert "accepted artifact requires an existing .ckpt checkpoint_path" in errors

    more_bad = dict(artifact)
    for field in exp4108.REQUIRED_ARTIFACT_FIELDS:
        more_bad.pop(field, None)
    more_bad.update(
        {
            "honest_verdict": 7,
            "reproduced_exact_accuracy": True,
            "duration_s": False,
            "random_seed": False,
            "acceptance_gate_passed": "yes",
            "exact_accuracy_metric": "val/q_halt_accuracy",
        }
    )
    more_errors = exp4108.artifact_schema_errors(more_bad)
    assert "missing required field matches_published_087" in more_errors
    assert "honest_verdict must be a string" in more_errors
    assert "reproduced_exact_accuracy must be numeric or null" in more_errors
    assert "duration_s must be numeric" in more_errors
    assert "random_seed must be a bare int" in more_errors
    assert "acceptance_gate_passed must be a bare bool" in more_errors
    assert "exact_accuracy_metric must not be q_halt_accuracy" in more_errors

    out_of_range = dict(artifact)
    out_of_range["reproduced_exact_accuracy"] = 2.0
    assert "reproduced_exact_accuracy must be between 0 and 1" in exp4108.artifact_schema_errors(
        out_of_range
    )

    invalid_verdict = dict(artifact)
    invalid_verdict["honest_verdict"] = "fabricated"
    assert "honest_verdict must be terminal-prefixed or blocked" in exp4108.artifact_schema_errors(
        invalid_verdict
    )

    try:
        exp4108.validate_artifact(invalid_verdict)
    except ValueError as exc:
        assert "honest_verdict must be terminal-prefixed or blocked" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("validate_artifact should reject invalid verdict")
