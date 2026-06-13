"""Tests for Exp 4138 fixed-LR Sudoku Extreme accumulation pass 4.

Spec refs: REQ-LEARN-4138, SCENARIO-LEARN-4138,
SCENARIO-LEARN-4138-AUDIT.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107
from carnot import experiment_4116_sudoku_extreme_resume_pass1 as exp4116
from carnot import experiment_4126_lr_resume_correctness_fix as exp4126
from carnot import experiment_4138_sudoku_accumulate_pass4_convergence_check as exp4138


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
    (dataset_dir / "metadata.json").write_text('{"seed": 4138}\n', encoding="utf-8")


def _write_lr_artifact(path: Path, *, continuous: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"lr_continuous_across_resume": continuous}) + "\n", encoding="utf-8")
    return path


def _write_4127_artifact(path: Path, *, anchor: float = 0.278172343969) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment": "experiment_4127_sudoku_extreme_accumulate_fixed",
                "val_trajectory": [
                    {"pass_index": 0, "val_exact_accuracy": 0.105989582837},
                    {"pass_index": 1, "val_exact_accuracy": anchor, "delta_vs_previous": 0.172182761132},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_pass3_artifact(
    path: Path,
    *,
    val: float | None,
    delta: float | None,
    plateau: bool = False,
    baseline_status: str | None = None,
    recommendation: str = "for the .384 baseline: reset Lightning Timer elapsed state before Trainer.fit",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "experiment_4137_sudoku_accumulate_pass3_fixed_lr",
        "honest_verdict": "complete: pass3 fixture",
        "val_exact_accuracy": val,
        "delta_vs_previous": delta,
        "plateau_audit_done": plateau,
        "stable_checkpoint_path": str(path.parent / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"),
        "random_seed": 4108,
        "duration_s": 7.0,
        "corrected_config_recommendation": recommendation,
        "suspected_cause": "pass3 fixture cause",
        "pass2_artifact": {
            "experiment": "experiment_4136_sudoku_accumulate_pass2_fixed_lr",
            "val_exact_accuracy": 0.68 if val is not None else None,
            "delta_vs_previous": 0.16 if val is not None else None,
            "pass1_artifact": {
                "experiment": "experiment_4135_sudoku_accumulate_pass1_fixed_lr",
                "val_exact_accuracy": 0.52 if val is not None else None,
                "delta_vs_previous": 0.24 if val is not None else None,
            },
        },
    }
    if baseline_status is not None:
        payload["baseline_status"] = baseline_status
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _write_metrics(run_dir: Path, *, val: float, lrs: list[float]) -> Path:
    metrics = run_dir / "csv" / "version_0" / "metrics.csv"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    rows = ["epoch,step,train/lr,val/exact_accuracy"]
    for index, lr in enumerate(lrs):
        rows.append(f"{7000 + index},{7000 + index},{lr},")
    rows.append(f"7100,7100,,{val}")
    rows.append("")
    metrics.write_text("\n".join(rows), encoding="utf-8")
    return metrics


def test_req_learn_4138_spec_declares_pass4_contract() -> None:
    """REQ-LEARN-4138: OpenSpec declares the pass4 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4138" in spec
    assert "SCENARIO-LEARN-4138" in spec
    assert "SCENARIO-LEARN-4138-AUDIT" in spec
    assert "results/experiment_4138_sudoku_accumulate_pass4_convergence_check.json" in spec
    for field in exp4138.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_learn_4138_audited_pass3_stops_without_training(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4138-AUDIT: pass3 config-blocked branch must not train."""

    _make_ready_repo(tmp_path)
    pass3 = _write_pass3_artifact(
        tmp_path / "results" / "experiment_4137_sudoku_accumulate_pass3_fixed_lr.json",
        val=None,
        delta=None,
        plateau=True,
        baseline_status="config-blocked",
    )
    _write_4127_artifact(tmp_path / "results" / "experiment_4127_sudoku_extreme_accumulate_fixed.json")
    calls = 0

    def forbidden_runner(_config: exp4138.Exp4138Config) -> exp4116.ResumeRunResult:
        nonlocal calls
        calls += 1
        raise AssertionError("audit branch must not train")

    output = tmp_path / "results" / exp4138.RESULT_FILENAME
    artifact = exp4138.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        pass3_artifact_path=pass3,
        trainer_runner=forbidden_runner,
    )

    assert calls == 0
    assert artifact["baseline_status"] == "config-blocked"
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["val_exact_accuracy"] is None
    assert artifact["matches_published_087"] is False
    assert artifact["near_faithful_080"] is False
    assert artifact["estimated_passes_to_converge"] is None
    assert artifact["duration_s"] < 600
    assert "pass3 did not report positive accumulation" in artifact["honest_verdict"]
    assert ".384 baseline" in artifact["corrected_config_recommendation"]
    assert [entry["val_exact_accuracy"] for entry in artifact["val_trajectory_383"]] == [
        0.278172343969,
        None,
        None,
        None,
        None,
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_learn_4138_positive_pass3_runs_one_pass_and_sets_gates(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4138: pass4 reports faithful and near-faithful gates."""

    _make_ready_repo(tmp_path)
    _make_dataset(tmp_path / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k")
    stable = tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    stable.parent.mkdir(parents=True, exist_ok=True)
    stable.write_bytes(b"checkpoint")
    pass3 = _write_pass3_artifact(
        tmp_path / "results" / "experiment_4137_sudoku_accumulate_pass3_fixed_lr.json",
        val=0.78,
        delta=0.10,
    )
    lr_artifact = _write_lr_artifact(
        tmp_path / "results" / "experiment_4126_lr_resume_correctness_fix.json",
        continuous=True,
    )
    _write_4127_artifact(tmp_path / "results" / "experiment_4127_sudoku_extreme_accumulate_fixed.json")
    calls = 0

    def fake_runner(config: exp4138.Exp4138Config) -> exp4116.ResumeRunResult:
        nonlocal calls
        calls += 1
        _write_metrics(config.pass_run_dir(), val=0.81, lrs=[9.8e-5])
        metric = exp4107.ExactAccuracy(
            "val/exact_accuracy",
            0.81,
            config.pass_run_dir() / "csv" / "version_0" / "metrics.csv",
        )
        return exp4116.ResumeRunResult(
            return_code=0,
            stable_checkpoint_path=config.stable_checkpoint_path,
            checkpoint_reload_ok=True,
            checkpoint_reload_detail="loadable fixture",
            val_exact_accuracy=metric,
            cumulative_epochs=7101,
            duration_s=91.25,
            command=exp4138.build_train_command(config),
            stdout_tail=["pass4 fixture"],
            run_dir=config.pass_run_dir(),
        )

    artifact = exp4138.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4138.RESULT_FILENAME,
        pass3_artifact_path=pass3,
        lr_artifact_path=lr_artifact,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        dataset_builder=lambda _config: False,
        timer_reset=lambda _path: {"timer_elapsed_reset": False},
        trainer_runner=fake_runner,
    )

    assert calls == 1
    assert artifact["val_exact_accuracy"] == 0.81
    assert artifact["matches_published_087"] is False
    assert artifact["near_faithful_080"] is True
    assert artifact["estimated_passes_to_converge"] is None
    assert "near-faithful" in artifact["honest_verdict"]
    assert "pass_4_hydra" in " ".join(artifact["command"])
    assert artifact["stable_checkpoint_path"] == str(stable)
    assert [entry["val_exact_accuracy"] for entry in artifact["val_trajectory_383"]] == [
        0.278172343969,
        0.52,
        0.68,
        0.78,
        0.81,
    ]
    assert artifact["acceptance_gate_passed"] is True


def test_req_learn_4138_schema_and_helper_edges(tmp_path: Path) -> None:
    """REQ-LEARN-4138: schema and helper edges are deterministic."""

    _make_ready_repo(tmp_path)
    config = exp4138.Exp4138Config(repo_root=tmp_path)
    invalid_json = tmp_path / "bad.json"
    invalid_json.write_text("{bad json", encoding="utf-8")
    pass3 = {"val_exact_accuracy": None, "delta_vs_previous": None, "plateau_audit_done": True}
    artifact = exp4138.build_config_blocked_artifact(
        run_config=config,
        pass3_artifact=pass3,
        exp4127_artifact={},
        duration_s=1.0,
    )

    assert exp4138.artifact_schema_errors(artifact) == []
    assert exp4138.build_train_env(config)["DISABLE_COMPILE"] == "1"
    assert config.to_4116_config().hydra_run_dir == config.pass_run_dir()
    assert exp4138.load_pass3_artifact(tmp_path / "missing.json") == {}
    assert exp4138.load_pass3_artifact(invalid_json) == {}
    assert exp4138.pass3_allows_training({"val_exact_accuracy": 0.78, "delta_vs_previous": 0.1}) is True
    assert exp4138.pass3_allows_training({"val_exact_accuracy": 0.78, "delta_vs_previous": 0}) is False
    assert exp4138.pass3_allows_training(
        {"val_exact_accuracy": 0.78, "delta_vs_previous": 0.1, "baseline_status": "config-blocked"}
    ) is False
    assert exp4138.matches_published_087(0.86) is True
    assert exp4138.near_faithful_080(0.80) is True
    assert exp4138.estimate_passes_to_converge([0.278, 0.52, 0.68, 0.78, 0.79]) == 1
    assert exp4138.estimate_passes_to_converge([0.278, None, None]) is None

    invalid = dict(artifact)
    invalid.update(
        {
            "honest_verdict": "pending",
            "val_exact_accuracy": 7,
            "val_trajectory_383": "0.278",
            "matches_published_087": "false",
            "near_faithful_080": "false",
            "estimated_passes_to_converge": True,
            "stable_checkpoint_path": "somewhere/else.ckpt",
            "random_seed": True,
            "reproducibility_checksum": "nope",
            "duration_s": [1.0],
            "acceptance_gate_passed": "yes",
        }
    )
    errors = exp4138.artifact_schema_errors(invalid)
    assert "honest_verdict must be terminal-prefixed or blocked" in errors
    assert "val_exact_accuracy must be numeric between 0 and 1 or null" in errors
    assert "val_trajectory_383 must be a list" in errors
    assert "matches_published_087 must be a bare bool" in errors
    assert "near_faithful_080 must be a bare bool" in errors
    assert "estimated_passes_to_converge must be a non-negative int or null" in errors
    assert "stable_checkpoint_path must be the shared sudoku_extreme_baseline/last.ckpt path" in errors
    assert "random_seed must be a bare int" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors
    assert "duration_s must be a scalar bounded number below 4800" in errors
    assert "acceptance_gate_passed must be a bare bool" in errors
    assert "missing required field honest_verdict" in exp4138.artifact_schema_errors({})
    assert "honest_verdict must be a string" in exp4138.artifact_schema_errors(
        {**artifact, "honest_verdict": 7}
    )
    assert "config-blocked duration_s must be below 600" in exp4138.artifact_schema_errors(
        {**artifact, "duration_s": 700.0}
    )
    assert "matches_published_087=true requires val within 0.02 of 0.87" in (
        exp4138.artifact_schema_errors(
            {**artifact, "matches_published_087": True, "val_exact_accuracy": 0.1}
        )
    )
    assert "near_faithful_080=true requires val >= 0.80" in (
        exp4138.artifact_schema_errors({**artifact, "near_faithful_080": True, "val_exact_accuracy": 0.1})
    )
    assert "config-blocked acceptance requires corrected_config_recommendation" in (
        exp4138.artifact_schema_errors(
            {**artifact, "acceptance_gate_passed": True, "corrected_config_recommendation": ""}
        )
    )

    exp4138.write_result_artifact(tmp_path / "artifact.json", artifact)
    assert json.loads((tmp_path / "artifact.json").read_text(encoding="utf-8")) == artifact
    try:
        exp4138.validate_artifact(invalid)
    except ValueError as exc:
        assert "honest_verdict must be terminal-prefixed or blocked" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("validate_artifact should reject schema errors")


def test_req_learn_4138_verdict_edges(tmp_path: Path) -> None:
    """REQ-LEARN-4138: measured pass4 verdicts distinguish gate outcomes."""

    _make_ready_repo(tmp_path)
    config = exp4138.Exp4138Config(repo_root=tmp_path)
    run_result = exp4116.ResumeRunResult(
        return_code=0,
        stable_checkpoint_path=config.stable_checkpoint_path,
        checkpoint_reload_ok=True,
        checkpoint_reload_detail="loadable fixture",
        val_exact_accuracy=None,
        cumulative_epochs=None,
        duration_s=50.0,
        command=exp4138.build_train_command(config),
        stdout_tail=[],
        run_dir=config.pass_run_dir(),
    )

    low = exp4138.build_result_artifact(
        run_config=config,
        run_result=run_result,
        pass_metrics=exp4138.PassMetricSummary(0.79, 9.9e-5, 1, None, None),
        pass3_artifact={"val_exact_accuracy": 0.78, "delta_vs_previous": 0.1},
        exp4127_artifact={"val_trajectory": [{"val_exact_accuracy": 0.278}]},
        preconditions_checked=[],
        lr_fix_artifact={"lr_continuous_across_resume": True},
        dataset_generated=False,
        checkpoint_timer_reset={"timer_elapsed_reset": False},
    )
    faithful = exp4138.build_result_artifact(
        run_config=config,
        run_result=run_result,
        pass_metrics=exp4138.PassMetricSummary(0.86, 9.9e-5, 1, None, None),
        pass3_artifact={"val_exact_accuracy": 0.84, "delta_vs_previous": 0.1},
        exp4127_artifact={"val_trajectory": [{"val_exact_accuracy": 0.278}]},
        preconditions_checked=[],
        lr_fix_artifact={"lr_continuous_across_resume": True},
        dataset_generated=False,
    )

    assert "val=0.7900, .384 finishes convergence" in low["honest_verdict"]
    assert low["estimated_passes_to_converge"] == 1
    assert "faithful, graft runs full" in faithful["honest_verdict"]
    assert faithful["matches_published_087"] is True
