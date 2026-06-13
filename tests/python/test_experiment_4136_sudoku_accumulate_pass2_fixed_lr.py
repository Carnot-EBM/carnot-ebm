"""Tests for Exp 4136 fixed-LR Sudoku Extreme accumulation pass 2.

Spec refs: REQ-LEARN-4136, SCENARIO-LEARN-4136,
SCENARIO-LEARN-4136-AUDIT.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116
from carnot import experiment_4126_lr_resume_correctness_fix as exp4126
from carnot import experiment_4136_sudoku_accumulate_pass2_fixed_lr as exp4136


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
        "\n".join(
            [
                "timekeeping:",
                "  max_epochs: 50000",
                "  batch_size: 768",
                "model_tuning:",
                "  learning_rate: 1e-4",
                "  warmup_steps: 2000",
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
    (dataset_dir / "metadata.json").write_text('{"seed": 4136}\n', encoding="utf-8")


def _write_lr_artifact(path: Path, *, continuous: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: lr fixture",
                "lr_continuous_across_resume": continuous,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_pass1_artifact(
    path: Path,
    *,
    val: float | None,
    delta: float | None,
    verdict: str = "complete: pass1 fixture",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": verdict,
                "val_exact_accuracy": val,
                "delta_vs_previous": delta,
                "stable_checkpoint_path": str(
                    path.parents[1] / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
                ),
                "duration_s": 7.0,
                "train_lr_point_count": 0,
                "exact_accuracy_metrics_path": None,
                "stdout_tail": [
                    "Time limit reached. Elapsed time is 1:00:41. Signaling Trainer to stop.",
                    "val_exact_accuracy=val/exact_accuracy metric missing",
                ],
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
        rows.append(f"{5000 + index},{5000 + index},{lr},")
    if val is not None:
        rows.append(f"5100,5100,,{val}")
    rows.append("")
    metrics.write_text("\n".join(rows), encoding="utf-8")
    return metrics


def _resume_result(
    config: exp4136.Exp4136Config,
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
        cumulative_epochs=5101 if val is not None else None,
        duration_s=duration_s,
        command=exp4136.build_train_command(config),
        stdout_tail=["pass2 fixture"],
        run_dir=run_dir,
    )


def test_req_learn_4136_spec_declares_pass2_contract() -> None:
    """REQ-LEARN-4136: OpenSpec declares the pass2 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4136" in spec
    assert "SCENARIO-LEARN-4136" in spec
    assert "SCENARIO-LEARN-4136-AUDIT" in spec
    assert "results/experiment_4136_sudoku_accumulate_pass2_fixed_lr.json" in spec
    for field in exp4136.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_learn_4136_audit_for_missing_pass1_delta_does_not_train(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4136-AUDIT: bad pass1 evidence triggers audit, not training."""

    _make_ready_repo(tmp_path)
    pass1 = _write_pass1_artifact(
        tmp_path / "results" / "experiment_4135_sudoku_accumulate_pass1_fixed_lr.json",
        val=None,
        delta=None,
        verdict="complete: missing_real_val_exact_accuracy",
    )
    calls = 0

    def forbidden_runner(_config: exp4136.Exp4136Config) -> exp4116.ResumeRunResult:
        nonlocal calls
        calls += 1
        raise AssertionError("audit branch must not train")

    output = tmp_path / "results" / exp4136.RESULT_FILENAME
    artifact = exp4136.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        pass1_artifact_path=pass1,
        trainer_runner=forbidden_runner,
    )

    assert calls == 0
    assert artifact["plateau_audit_done"] is True
    assert artifact["val_exact_accuracy"] is None
    assert artifact["delta_vs_previous"] is None
    assert artifact["duration_s"] < 600
    assert "pass1 did not report positive accumulation" in artifact["suspected_cause"]
    assert "rerun Exp 4135" in artifact["corrected_config_recommendation"]
    assert artifact["config_audit"]["batch_size_vs_published_recipe"]["actual"] == 128
    assert artifact["config_audit"]["peak_lr_vs_published_recipe"]["actual"] == 0.0001
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_learn_4136_positive_pass1_runs_one_pass_and_reports_delta(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4136: positive pass1 resumes once and reports pass2 delta."""

    _make_ready_repo(tmp_path)
    _make_dataset(tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k")
    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    pass1 = _write_pass1_artifact(
        tmp_path / "results" / "experiment_4135_sudoku_accumulate_pass1_fixed_lr.json",
        val=0.35,
        delta=0.072,
    )
    lr_artifact = _write_lr_artifact(
        tmp_path / "results" / "experiment_4126_lr_resume_correctness_fix.json",
        continuous=True,
    )
    calls = 0

    def fake_runner(config: exp4136.Exp4136Config) -> exp4116.ResumeRunResult:
        nonlocal calls
        calls += 1
        _write_metrics(config.pass_run_dir(), val=0.42, lrs=[9.9e-5, 9.8e-5])
        return exp4136.verify_completed_resume_pass(
            config,
            duration_s=91.25,
            return_code=0,
            command=exp4136.build_train_command(config),
            stdout_tail=["done"],
            checkpoint_loader=lambda _path: (True, "loadable fixture"),
        )

    artifact = exp4136.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4136.RESULT_FILENAME,
        pass1_artifact_path=pass1,
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        dataset_builder=lambda _config: False,
        trainer_runner=fake_runner,
    )

    assert calls == 1
    assert artifact["plateau_audit_done"] is False
    assert artifact["val_exact_accuracy"] == 0.42
    assert artifact["delta_vs_previous"] == 0.07
    assert artifact["matches_published_087"] is False
    assert artifact["acceptance_gate_passed"] is True
    assert "pass_2_hydra" in " ".join(artifact["command"])
    assert artifact["stable_checkpoint_path"] == str(stable)
    assert artifact["pass1_artifact"]["val_exact_accuracy"] == 0.35


def test_req_learn_4136_blocks_after_positive_pass1_without_training(tmp_path: Path) -> None:
    """REQ-LEARN-4136: positive pass1 still honors runtime and LR gates."""

    _make_ready_repo(tmp_path)
    pass1 = _write_pass1_artifact(
        tmp_path / "results" / "experiment_4135_sudoku_accumulate_pass1_fixed_lr.json",
        val=0.35,
        delta=0.072,
    )
    lr_artifact = _write_lr_artifact(
        tmp_path / "results" / "experiment_4126_lr_resume_correctness_fix.json",
        continuous=True,
    )
    trainer_calls = 0

    def forbidden_runner(_config: exp4136.Exp4136Config) -> exp4116.ResumeRunResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("blocked branches must not train")

    blocked_uv = exp4136.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "blocked_uv.json",
        pass1_artifact_path=pass1,
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: None,
        cuda_checker=lambda: (True, "cuda fixture"),
        trainer_runner=forbidden_runner,
    )
    assert blocked_uv["honest_verdict"] == "blocked_uv_missing"

    _write_lr_artifact(lr_artifact, continuous=False)
    blocked_lr = exp4136.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "blocked_lr.json",
        pass1_artifact_path=pass1,
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        trainer_runner=forbidden_runner,
    )
    assert blocked_lr["honest_verdict"] == "blocked_lr_fix_not_landed"

    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    _write_lr_artifact(lr_artifact, continuous=True)
    blocked_dataset = exp4136.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / "blocked_dataset.json",
        pass1_artifact_path=pass1,
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        dataset_builder=lambda _config: None,
        trainer_runner=forbidden_runner,
    )
    assert blocked_dataset["honest_verdict"] == "blocked_dataset_missing"
    assert blocked_dataset["dataset_generated"] is True
    assert trainer_calls == 0


def test_req_learn_4136_schema_guards_required_scalar_fields(tmp_path: Path) -> None:
    """REQ-LEARN-4136: schema rejects missing or ambiguous required fields."""

    _make_ready_repo(tmp_path)
    config = exp4136.Exp4136Config(repo_root=tmp_path)
    pass1 = {"val_exact_accuracy": None, "delta_vs_previous": None, "honest_verdict": "complete: missing"}
    artifact = exp4136.build_audit_artifact(
        run_config=config,
        pass1_artifact=pass1,
        config_audit=exp4136.audit_pass1_config(config, pass1),
        duration_s=1.0,
    )

    assert exp4136.artifact_schema_errors(artifact) == []

    invalid = dict(artifact)
    invalid.update(
        {
            "honest_verdict": "pending",
            "val_exact_accuracy": 7,
            "delta_vs_previous": "0.1",
            "plateau_audit_done": "true",
            "matches_published_087": "false",
            "stable_checkpoint_path": "somewhere/else.ckpt",
            "random_seed": True,
            "reproducibility_checksum": "nope",
            "duration_s": [1.0],
        }
    )
    errors = exp4136.artifact_schema_errors(invalid)
    assert "honest_verdict must be terminal-prefixed or blocked" in errors
    assert "val_exact_accuracy must be numeric between 0 and 1 or null" in errors
    assert "delta_vs_previous must be numeric or null" in errors
    assert "plateau_audit_done must be a bare bool" in errors
    assert "matches_published_087 must be a bare bool" in errors
    assert "stable_checkpoint_path must be the shared sudoku_extreme_baseline/last.ckpt path" in errors
    assert "random_seed must be a bare int" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors
    assert "duration_s must be a scalar bounded number below 4800" in errors
    assert "missing required field honest_verdict" in exp4136.artifact_schema_errors({})
    assert "honest_verdict must be a string" in exp4136.artifact_schema_errors(
        {**artifact, "honest_verdict": 7}
    )
    assert "plateau audit duration_s must be below 600" in exp4136.artifact_schema_errors(
        {**artifact, "duration_s": 700.0}
    )
    assert "acceptance_gate_passed must be a bare bool" in exp4136.artifact_schema_errors(
        {**artifact, "acceptance_gate_passed": "yes"}
    )
    assert "matches_published_087=true requires val within 0.02 of 0.87" in (
        exp4136.artifact_schema_errors(
            {**artifact, "matches_published_087": True, "val_exact_accuracy": 0.1}
        )
    )
    assert "audit acceptance requires suspected_cause and corrected_config_recommendation" in (
        exp4136.artifact_schema_errors(
            {**artifact, "acceptance_gate_passed": True, "suspected_cause": ""}
        )
    )
    assert "training acceptance requires val_exact_accuracy and delta_vs_previous" in (
        exp4136.artifact_schema_errors(
            {
                **artifact,
                "plateau_audit_done": False,
                "acceptance_gate_passed": True,
                "val_exact_accuracy": None,
                "delta_vs_previous": None,
            }
        )
    )

    output = tmp_path / "artifact.json"
    exp4136.write_result_artifact(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    try:
        exp4136.validate_artifact(invalid)
    except ValueError as exc:
        assert "honest_verdict must be terminal-prefixed or blocked" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("validate_artifact should reject schema errors")


def test_req_learn_4136_rewarm_lr_keeps_acceptance_false(tmp_path: Path) -> None:
    """REQ-LEARN-4136: measured pass2 artifacts fail closed on LR rewarm."""

    _make_ready_repo(tmp_path)
    _make_dataset(tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k")
    config = exp4136.Exp4136Config(repo_root=tmp_path)
    config.stable_dir.mkdir(parents=True)
    config.stable_checkpoint_path.write_bytes(b"checkpoint")
    metrics = _write_metrics(config.pass_run_dir(), val=0.42, lrs=[exp4126.FRESH_WARMUP_FIRST_LR])
    summary = exp4136.summarize_pass_metrics(config.pass_run_dir())

    artifact = exp4136.build_result_artifact(
        run_config=config,
        run_result=_resume_result(config, val=0.42),
        pass_metrics=summary,
        pass1_artifact={"val_exact_accuracy": 0.35, "delta_vs_previous": 0.072},
        preconditions_checked=[],
        lr_fix_artifact={"lr_continuous_across_resume": True},
        dataset_generated=False,
    )

    assert artifact["lr_continued_not_rewarmed"] is False
    assert artifact["acceptance_gate_passed"] is False
    assert artifact["exact_accuracy_metrics_path"] == str(metrics)

    missing_val = exp4136.build_result_artifact(
        run_config=config,
        run_result=_resume_result(config, val=None),
        pass_metrics=exp4136.summarize_pass_metrics(tmp_path / "missing"),
        pass1_artifact={"val_exact_accuracy": 0.35, "delta_vs_previous": 0.072},
        preconditions_checked=[],
        lr_fix_artifact={"lr_continuous_across_resume": True},
        dataset_generated=False,
    )
    assert missing_val["honest_verdict"] == "complete: missing_real_val_exact_accuracy"

    matched = exp4136.build_result_artifact(
        run_config=config,
        run_result=_resume_result(config, val=0.86),
        pass_metrics=exp4136.PassMetricSummary(0.86, 9.99e-5, 1, metrics, metrics),
        pass1_artifact={"val_exact_accuracy": 0.80, "delta_vs_previous": 0.01},
        preconditions_checked=[],
        lr_fix_artifact={"lr_continuous_across_resume": True},
        dataset_generated=False,
    )
    assert matched["matches_published_087"] is True

    plateau = exp4136.build_result_artifact(
        run_config=config,
        run_result=_resume_result(config, val=0.35),
        pass_metrics=exp4136.PassMetricSummary(0.35, 9.99e-5, 1, metrics, metrics),
        pass1_artifact={"val_exact_accuracy": 0.35, "delta_vs_previous": 0.072},
        preconditions_checked=[],
        lr_fix_artifact={"lr_continuous_across_resume": True},
        dataset_generated=False,
    )
    assert "plateau_delta=0.0000" in plateau["honest_verdict"]


def test_req_learn_4136_helper_edges_are_deterministic(tmp_path: Path) -> None:
    """REQ-LEARN-4136: helper edges remain deterministic for audit/schema use."""

    config = exp4136.Exp4136Config(repo_root=tmp_path)
    invalid_json = tmp_path / "bad.json"
    invalid_json.write_text("{bad json", encoding="utf-8")

    assert config.to_4116_config().hydra_run_dir == config.pass_run_dir()
    assert exp4136.build_train_env(config)["DISABLE_COMPILE"] == "1"
    assert exp4136.load_pass1_artifact(tmp_path / "missing.json") == {}
    assert exp4136.load_pass1_artifact(invalid_json) == {}
    assert exp4136._float_or_none("not-a-number") is None
    assert exp4136._nested({"outer": 1}, "outer", "inner") is None
    assert exp4136._read_yaml_mapping(tmp_path / "missing.yaml") == {}
    assert "delta_vs_previous=0" in exp4136._suspected_cause(
        {"val_exact_accuracy": 0.35, "delta_vs_previous": 0.0}
    )
    assert exp4136._suspected_cause({"val_exact_accuracy": 0.35, "delta_vs_previous": 0.1}) == (
        "pass1 reported positive accumulation"
    )
