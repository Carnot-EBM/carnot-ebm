"""Tests for Exp 4116 resumable nano-trm Sudoku Extreme pass 1.

Spec refs: REQ-LEARN-4116, SCENARIO-LEARN-4116,
SCENARIO-LEARN-4116-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _make_ready_repo(tmp_path: Path) -> None:
    trainer = tmp_path / "nano-trm" / "src" / "nn" / "train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# trainer fixture\n", encoding="utf-8")
    builder = tmp_path / "nano-trm" / "scripts" / "data" / "build_sudoku_extreme_dataset.py"
    builder.parent.mkdir(parents=True, exist_ok=True)
    builder.write_text("# builder fixture\n", encoding="utf-8")
    config = tmp_path / "nano-trm" / "src" / "nn" / "configs" / "experiment"
    config.mkdir(parents=True, exist_ok=True)
    (config / "trm_sudoku_extreme_1k_aug_1k.yaml").write_text(
        "trainer:\n  check_val_every_n_epoch: 100\n",
        encoding="utf-8",
    )
    data_config = tmp_path / "nano-trm" / "src" / "nn" / "configs" / "data"
    data_config.mkdir(parents=True, exist_ok=True)
    (data_config / "sudoku_extreme_1k_aug1k.yaml").write_text("batch_size: 768\n", encoding="utf-8")


def _make_dataset(dataset_dir: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "dataset.json").write_text('{"num_puzzles": 1}\n', encoding="utf-8")
        for name in ("all__inputs.npy", "all__labels.npy", "all__puzzle_identifiers.npy"):
            (split_dir / name).write_bytes(f"{split}:{name}".encode("ascii"))
    (dataset_dir / "metadata.json").write_text('{"seed": 4108}\n', encoding="utf-8")


def _write_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "epoch,step,val/exact_accuracy,val/q_halt_accuracy,train/exact_accuracy",
                "4099,4099,,1.0,0.25",
                "4199,4199,0.02317708358168602,0.9768,",
                "4249,4249,,0.99,0.5",
                "4299,4299,0.03125,0.9687,",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _exp4108_artifact(path: Path, checkpoint: Path | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: interrupted_return_code_1_reproduced_0.0232",
                "checkpoint_path": None if checkpoint is None else str(checkpoint),
                "checkpoint_reload_ok": checkpoint is not None,
                "reproduced_exact_accuracy": 0.02317708358168602,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_req_learn_4116_spec_declares_resume_contract() -> None:
    """REQ-LEARN-4116: OpenSpec declares the stable resume artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4116" in spec
    assert "SCENARIO-LEARN-4116" in spec
    assert "SCENARIO-LEARN-4116-BLOCKED" in spec
    assert "results/trm_runs/sudoku_extreme_baseline/last.ckpt" in spec
    assert 'trainer.max_time="00:01:00:00"' in spec
    for field in exp4116.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4116_command_uses_stable_checkpoint_and_bound(tmp_path: Path) -> None:
    """REQ-LEARN-4116: train command resumes from and saves into the stable path."""

    _make_ready_repo(tmp_path)
    config = exp4116.Exp4116Config(repo_root=tmp_path)
    config.stable_dir.mkdir(parents=True)
    config.stable_checkpoint_path.write_bytes(b"checkpoint")

    command = exp4116.build_train_command(config)

    assert command[:4] == ["uv", "run", "python", "src/nn/train.py"]
    assert "experiment=trm_sudoku_extreme_1k_aug_1k" in command
    assert "logger=csv" in command
    assert f"ckpt_path={config.stable_checkpoint_path}" in command
    assert "+trainer.max_time=00:01:00:00" in command
    assert "save_dir=null" in command
    assert f"callbacks.model_checkpoint.dirpath={config.stable_dir}" in command
    assert "callbacks.model_checkpoint.save_last=true" in command
    assert "callbacks.model_checkpoint.monitor=val/exact_accuracy" in command
    assert "callbacks.model_checkpoint.mode=max" in command
    assert f"+callbacks.exp4116_progress.checkpoint_dir={config.stable_dir}" in command
    assert f"hydra.run.dir={config.hydra_run_dir}" in command
    assert f"seed={exp4116.RANDOM_SEED}" in command

    missing_resume = exp4116.Exp4116Config(repo_root=tmp_path, stable_dir=tmp_path / "missing")
    assert "ckpt_path=null" in exp4116.build_train_command(missing_resume)


def test_req_learn_4116_native_trainer_restores_full_trusted_checkpoints() -> None:
    """REQ-LEARN-4116: native ckpt_path resume must not use restricted weights-only load."""

    train_source = (REPO / "nano-trm" / "src" / "nn" / "train.py").read_text(encoding="utf-8")

    assert "trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get(\"ckpt_path\"), weights_only=False)" in train_source


def test_scenario_learn_4116_seeds_stable_checkpoint_from_exp4108(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4116: loadable Exp 4108 checkpoint seeds the stable lineage."""

    _make_ready_repo(tmp_path)
    prior = tmp_path / "results" / "trm_runs" / "exp4108" / "checkpoints" / "last.ckpt"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b"exp4108")
    artifact_path = _exp4108_artifact(tmp_path / "results" / "experiment_4108.json", prior)
    config = exp4116.Exp4116Config(repo_root=tmp_path, exp4108_artifact_path=artifact_path)

    seen: list[Path] = []

    def fake_loader(path: Path) -> tuple[bool, str]:
        seen.append(path)
        return True, "loadable fixture"

    seed = exp4116.ensure_stable_checkpoint_seed(config, checkpoint_loader=fake_loader)

    assert seed.seed_status == "seeded_from_exp4108"
    assert seed.source_checkpoint_path == prior
    assert config.stable_checkpoint_path.read_bytes() == b"exp4108"
    assert seen == [prior, config.stable_checkpoint_path]

    config.stable_checkpoint_path.write_bytes(b"already-stable")
    seed_again = exp4116.ensure_stable_checkpoint_seed(config, checkpoint_loader=fake_loader)
    assert seed_again.seed_status == "existing_stable_checkpoint"
    assert config.stable_checkpoint_path.read_bytes() == b"already-stable"


def test_scenario_learn_4116_metrics_use_val_exact_not_q_halt(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4116: val exact accuracy and cumulative epochs come from CSV."""

    metrics_path = tmp_path / "run" / "csv" / "version_0" / "metrics.csv"
    _write_metrics(metrics_path)

    exact = exp4116.extract_latest_val_exact_accuracy(tmp_path / "run")
    epochs = exp4116.extract_cumulative_epochs(tmp_path / "run")

    assert exact.metric_name == "val/exact_accuracy"
    assert exact.value == 0.03125
    assert exact.metrics_path == metrics_path
    assert epochs == 4300

    q_halt_only = tmp_path / "qhalt" / "metrics.csv"
    q_halt_only.parent.mkdir()
    q_halt_only.write_text("epoch,step,val/q_halt_accuracy\n1,1,1.0\n", encoding="utf-8")
    try:
        exp4116.extract_latest_val_exact_accuracy(q_halt_only.parent)
    except ValueError as exc:
        assert "val/exact_accuracy" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("q_halt-only metrics must not satisfy Exp 4116")


def test_scenario_learn_4116_success_artifact_reports_real_resume_metrics(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4116: below-target validation accuracy is complete evidence."""

    _make_ready_repo(tmp_path)
    config = exp4116.Exp4116Config(repo_root=tmp_path)
    config.stable_dir.mkdir(parents=True)
    config.stable_checkpoint_path.write_bytes(b"checkpoint")
    metrics_path = config.hydra_run_dir / "csv" / "version_0" / "metrics.csv"
    _write_metrics(metrics_path)
    exact = exp4116.extract_latest_val_exact_accuracy(config.hydra_run_dir)
    result = exp4116.ResumeRunResult(
        return_code=0,
        stable_checkpoint_path=config.stable_checkpoint_path,
        checkpoint_reload_ok=True,
        checkpoint_reload_detail="loadable fixture",
        val_exact_accuracy=exact,
        cumulative_epochs=exp4116.extract_cumulative_epochs(config.hydra_run_dir),
        duration_s=3599.5,
        command=exp4116.build_train_command(config),
        stdout_tail=["val/exact_accuracy=0.03125"],
        run_dir=config.hydra_run_dir,
    )

    artifact = exp4116.build_result_artifact(
        run_config=config,
        run_result=result,
        seed_result=exp4116.StableSeedResult(
            seed_status="seeded_from_exp4108",
            source_checkpoint_path=tmp_path / "exp4108.ckpt",
            stable_checkpoint_path=config.stable_checkpoint_path,
            checkpoint_reload_ok=True,
            checkpoint_reload_detail="loadable fixture",
        ),
        preconditions_checked=[],
        dataset_generated=False,
    )

    assert artifact["honest_verdict"] == "complete: val=0.0312 still_below_0.87"
    assert artifact["val_exact_accuracy"] == 0.03125
    assert artifact["cumulative_epochs"] == 4300
    assert artifact["stable_checkpoint_path"] == str(config.stable_checkpoint_path)
    assert artifact["checkpoint_reload_ok"] is True
    assert artifact["duration_s"] == 3599.5
    assert artifact["random_seed"] == exp4116.RANDOM_SEED
    assert artifact["acceptance_gate_passed"] is True
    assert exp4116.artifact_schema_errors(artifact) == []


def test_req_learn_4116_blocked_and_schema_guards(tmp_path: Path) -> None:
    """REQ-LEARN-4116: blocked artifacts stay honest and schema rejects fabrications."""

    artifact = exp4116.build_blocked_artifact(
        "blocked_cuda_unavailable",
        preconditions_checked=[
            exp4107.PreconditionCheck("uv", True, "/usr/bin/uv"),
            exp4107.PreconditionCheck("cuda_available", False, "no cuda"),
        ],
        stable_checkpoint_path=tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt",
        duration_s=0.25,
    )

    assert artifact["honest_verdict"] == "blocked_cuda_unavailable"
    assert artifact["val_exact_accuracy"] is None
    assert artifact["checkpoint_reload_ok"] is False
    assert artifact["acceptance_gate_passed"] is False
    assert exp4116.artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad.update(
        {
            "honest_verdict": "not terminal",
            "val_exact_accuracy": 2.0,
            "cumulative_epochs": -1,
            "checkpoint_reload_ok": "true",
            "duration_s": False,
            "random_seed": False,
            "acceptance_gate_passed": True,
            "exact_accuracy_metric": "val/q_halt_accuracy",
        }
    )
    errors = exp4116.artifact_schema_errors(bad)
    assert "honest_verdict must be terminal-prefixed or blocked" in errors
    assert "val_exact_accuracy must be between 0 and 1" in errors
    assert "cumulative_epochs must be a non-negative int or null" in errors
    assert "checkpoint_reload_ok must be a bare bool" in errors
    assert "duration_s must be numeric" in errors
    assert "random_seed must be a bare int" in errors
    assert "exact_accuracy_metric must not be q_halt_accuracy" in errors
    assert "accepted artifact requires checkpoint_reload_ok true" in errors

    missing_exact = dict(artifact)
    missing_exact["acceptance_gate_passed"] = True
    assert "accepted artifact requires val_exact_accuracy" in exp4116.artifact_schema_errors(missing_exact)


def test_req_learn_4116_run_experiment_blocks_and_writes_success(tmp_path: Path) -> None:
    """REQ-LEARN-4116: run_experiment writes blocked or measured artifacts."""

    _make_ready_repo(tmp_path)
    dataset_dir = tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k"
    _make_dataset(dataset_dir)
    save_parent = tmp_path / "results" / "trm_runs"
    save_parent.mkdir(parents=True)

    blocked_output = tmp_path / "results" / "blocked.json"
    blocked = exp4116.run_experiment(
        repo_root=tmp_path,
        output_path=blocked_output,
        uv_resolver=lambda _name: None,
        cuda_checker=lambda: (True, "cuda fixture"),
    )
    assert blocked["honest_verdict"] == "blocked_nanotrm_or_uv_missing"
    assert json.loads(blocked_output.read_text(encoding="utf-8")) == blocked

    prior = tmp_path / "results" / "trm_runs" / "exp4108" / "checkpoints" / "last.ckpt"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b"exp4108")
    artifact_path = _exp4108_artifact(tmp_path / "results" / "experiment_4108.json", prior)
    output_path = tmp_path / "results" / exp4116.RESULT_FILENAME

    def fake_loader(_path: Path) -> tuple[bool, str]:
        return True, "loadable fixture"

    def fake_runner(config: exp4116.Exp4116Config) -> exp4116.ResumeRunResult:
        config.hydra_run_dir.mkdir(parents=True, exist_ok=True)
        _write_metrics(config.hydra_run_dir / "csv" / "version_0" / "metrics.csv")
        return exp4116.verify_completed_resume_run(
            config,
            duration_s=10.0,
            return_code=0,
            command=exp4116.build_train_command(config),
            stdout_tail=["done"],
            checkpoint_loader=fake_loader,
        )

    artifact = exp4116.run_experiment(
        repo_root=tmp_path,
        output_path=output_path,
        exp4108_artifact_path=artifact_path,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=fake_loader,
        dataset_builder=lambda _config: False,
        trainer_runner=fake_runner,
    )

    assert artifact["honest_verdict"] == "complete: val=0.0312 still_below_0.87"
    assert artifact["stable_checkpoint_path"].endswith("results/trm_runs/sudoku_extreme_baseline/last.ckpt")
    assert artifact["checkpoint_reload_ok"] is True
    assert artifact["acceptance_gate_passed"] is True
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact


def test_req_learn_4116_defensive_branches_without_live_training(tmp_path: Path) -> None:
    """REQ-LEARN-4116: defensive branches stay deterministic and non-live."""

    printed: list[tuple[str, bool]] = []

    def ok_printer(message: str, *, flush: bool) -> None:
        printed.append((message, flush))

    def closed_pipe(_message: str, *, flush: bool) -> None:
        assert flush is True
        raise BrokenPipeError("closed")

    exp4116._safe_progress_print("ok", printer=ok_printer)
    exp4116._safe_progress_print("ignored", printer=closed_pipe)
    assert printed == [("ok", True)]
    assert exp4116._parse_metric_value("not-a-number") is None
    assert exp4116._parse_metric_value("inf") is None

    repo = tmp_path / "repo"
    _make_ready_repo(repo)
    checks, blocker = exp4116.check_preconditions(
        repo_root=repo,
        stable_dir=repo / "results" / "trm_runs" / "sudoku_extreme_baseline",
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (False, "no cuda"),
    )
    assert blocker == "blocked_cuda_unavailable"
    assert checks[2].resource == "cuda_available"

    _, stable_blocker = exp4116.check_preconditions(
        repo_root=repo,
        stable_dir=tmp_path / "outside" / "sudoku_extreme_baseline",
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda"),
    )
    assert stable_blocker == "blocked_save_dir_unwritable"

    config = exp4116.Exp4116Config(repo_root=repo)
    assert config.trainer_path == repo / "nano-trm" / "src" / "nn" / "train.py"
    assert config.dataset_builder_path == (
        repo / "nano-trm" / "scripts" / "data" / "build_sudoku_extreme_dataset.py"
    )
    missing_seed = exp4116.ensure_stable_checkpoint_seed(config, checkpoint_loader=lambda _path: (True, "ok"))
    assert missing_seed.seed_status == "no_exp4108_artifact"

    bad_artifact = repo / "results" / "bad.json"
    bad_artifact.parent.mkdir(parents=True, exist_ok=True)
    bad_artifact.write_text("{not json\n", encoding="utf-8")
    bad_config = exp4116.Exp4116Config(repo_root=repo, exp4108_artifact_path=bad_artifact)
    assert exp4116.ensure_stable_checkpoint_seed(bad_config).seed_status == "invalid_exp4108_artifact"

    missing_ckpt_artifact = _exp4108_artifact(repo / "results" / "missing_ckpt.json", repo / "missing.ckpt")
    missing_ckpt_config = exp4116.Exp4116Config(repo_root=repo, exp4108_artifact_path=missing_ckpt_artifact)
    assert (
        exp4116.ensure_stable_checkpoint_seed(missing_ckpt_config).seed_status
        == "no_loadable_exp4108_checkpoint"
    )

    unloadable_prior = repo / "results" / "trm_runs" / "prior" / "last.ckpt"
    unloadable_prior.parent.mkdir(parents=True)
    unloadable_prior.write_bytes(b"prior")
    unloadable_artifact = _exp4108_artifact(repo / "results" / "unloadable.json", unloadable_prior)
    unloadable_config = exp4116.Exp4116Config(repo_root=repo, exp4108_artifact_path=unloadable_artifact)

    def source_fails(path: Path) -> tuple[bool, str]:
        return (False, "source failed") if path == unloadable_prior else (True, "stable ok")

    assert exp4116.ensure_stable_checkpoint_seed(unloadable_config, checkpoint_loader=source_fails).seed_status == (
        "no_loadable_exp4108_checkpoint"
    )

    def stable_fails(path: Path) -> tuple[bool, str]:
        return (False, "stable failed") if path == unloadable_config.stable_checkpoint_path else (True, "source ok")

    unloadable_config.stable_checkpoint_path.unlink(missing_ok=True)
    assert exp4116.ensure_stable_checkpoint_seed(unloadable_config, checkpoint_loader=stable_fails).seed_status == (
        "exp4108_copy_unloadable"
    )

    single_metrics = repo / "metrics.csv"
    _write_metrics(single_metrics)
    assert exp4116.extract_latest_val_exact_accuracy(single_metrics).value == 0.03125

    config.stable_checkpoint_path.unlink(missing_ok=True)
    no_checkpoint = exp4116.verify_completed_resume_run(
        config,
        duration_s=1.0,
        checkpoint_loader=lambda _path: (False, "missing"),
    )
    assert no_checkpoint.checkpoint_reload_ok is False
    assert no_checkpoint.val_exact_accuracy is None

    config.stable_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config.stable_checkpoint_path.write_bytes(b"checkpoint")
    exact = exp4107.ExactAccuracy("val/exact_accuracy", 0.9, repo / "metrics.csv")
    base_result = exp4116.ResumeRunResult(
        return_code=0,
        stable_checkpoint_path=config.stable_checkpoint_path,
        checkpoint_reload_ok=True,
        checkpoint_reload_detail="ok",
        val_exact_accuracy=exact,
        cumulative_epochs=7,
        duration_s=20.0,
        command=[],
        stdout_tail=[],
        run_dir=config.hydra_run_dir,
    )
    seed_result = exp4116.StableSeedResult(
        "existing_stable_checkpoint",
        config.stable_checkpoint_path,
        config.stable_checkpoint_path,
        True,
        "ok",
    )
    reached = exp4116.build_result_artifact(
        run_config=config,
        run_result=base_result,
        seed_result=seed_result,
        preconditions_checked=[],
        dataset_generated=False,
    )
    assert reached["honest_verdict"] == "complete: val=0.9000 reached_0.87"

    return_code_result = exp4116.ResumeRunResult(
        **{**base_result.__dict__, "return_code": 7, "val_exact_accuracy": exact}
    )
    assert exp4116.build_result_artifact(
        run_config=config,
        run_result=return_code_result,
        seed_result=seed_result,
        preconditions_checked=[],
        dataset_generated=False,
    )["honest_verdict"] == "complete: return_code_7_val=0.9000"

    for result, expected in (
        (
            exp4116.ResumeRunResult(
                **{**base_result.__dict__, "return_code": 2, "val_exact_accuracy": None}
            ),
            "complete: nanotrm_resume_failed_return_code_2",
        ),
        (
            exp4116.ResumeRunResult(
                **{**base_result.__dict__, "checkpoint_reload_ok": False, "val_exact_accuracy": None}
            ),
            "complete: stable_checkpoint_missing_or_reload_failed",
        ),
        (
            exp4116.ResumeRunResult(**{**base_result.__dict__, "val_exact_accuracy": None}),
            "complete: missing_real_val_exact_accuracy",
        ),
    ):
        assert exp4116.build_result_artifact(
            run_config=config,
            run_result=result,
            seed_result=seed_result,
            preconditions_checked=[],
            dataset_generated=False,
        )["honest_verdict"] == expected

    assert "missing required field honest_verdict" in exp4116.artifact_schema_errors({})
    schema_probe = {
        "honest_verdict": 1,
        "val_exact_accuracy": True,
        "cumulative_epochs": None,
        "stable_checkpoint_path": "not-stable.ckpt",
        "checkpoint_reload_ok": False,
        "duration_s": 1.0,
        "random_seed": exp4116.RANDOM_SEED,
        "acceptance_gate_passed": "yes",
    }
    schema_errors = exp4116.artifact_schema_errors(schema_probe)
    assert "honest_verdict must be a string" in schema_errors
    assert "val_exact_accuracy must be numeric or null" in schema_errors
    assert "stable_checkpoint_path must be the shared sudoku_extreme_baseline/last.ckpt path" in schema_errors
    assert "acceptance_gate_passed must be a bare bool" in schema_errors

    too_long = dict(reached)
    too_long["duration_s"] = 4800.0
    assert "accepted artifact requires duration_s < 4800" in exp4116.artifact_schema_errors(too_long)
    bad_existing = dict(reached)
    bad_existing["stable_checkpoint_path"] = str(config.stable_checkpoint_path.with_name("missing.ckpt"))
    assert "accepted artifact requires existing stable checkpoint" in exp4116.artifact_schema_errors(bad_existing)
    try:
        exp4116.validate_artifact(schema_probe)
    except ValueError as exc:
        assert "honest_verdict must be a string" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("validate_artifact should reject schema errors")

    dataset_missing_output = repo / "results" / "dataset_missing.json"
    dataset_missing = exp4116.run_experiment(
        repo_root=repo,
        output_path=dataset_missing_output,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda"),
        dataset_builder=lambda _config: None,
    )
    assert dataset_missing["honest_verdict"] == "blocked_dataset_missing"
    assert dataset_missing["dataset_generated"] is True

    _make_dataset(repo / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k")

    def raising_runner(_config: exp4116.Exp4116Config) -> exp4116.ResumeRunResult:
        raise RuntimeError("runner failed")

    failed = exp4116.run_experiment(
        repo_root=repo,
        output_path=repo / "results" / "failed.json",
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda"),
        trainer_runner=raising_runner,
    )
    assert failed["honest_verdict"] == "complete: nanotrm_resume_failed_return_code_1"
    assert "RuntimeError: runner failed" in failed["stdout_tail"]
