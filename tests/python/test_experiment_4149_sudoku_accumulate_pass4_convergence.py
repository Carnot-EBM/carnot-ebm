"""Tests for Exp 4149 Sudoku Extreme final `.384` accumulation pass.

Spec refs: REQ-LEARN-4149, SCENARIO-LEARN-4149,
SCENARIO-LEARN-4149-BLOCKED-PASS3, SCENARIO-LEARN-4149-EARLY-CONVERGED.
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
from carnot import experiment_4149_sudoku_accumulate_pass4_convergence as exp4149


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
BASELINE_VAL = 0.278172343969


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


def _write_baseline(root: Path, stable: Path, *, val: float = BASELINE_VAL) -> Path:
    path = root / "results" / "experiment_4145_archive_v383_activate_v384.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "honest_verdict": "success: activated_v384_fixture",
                "v383_close_state": {
                    "baseline_val_exact_accuracy": val,
                    "stable_checkpoint_path": str(stable),
                    "matches_published_087": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_pass(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_prior_lineage(root: Path, stable: Path) -> None:
    _write_baseline(root, stable)
    _write_pass(
        root / "results" / exp4146.RESULT_FILENAME,
        {
            "honest_verdict": "blocked_noop_cap_not_confirmed_timer_elapsed",
            "val_exact_accuracy": None,
            "post_epoch": 6399,
            "duration_s": 0.121,
            "stable_checkpoint_path": str(stable),
        },
    )
    _write_pass(
        root / "results" / exp4147.RESULT_FILENAME,
        {
            "honest_verdict": "blocked_pass1_noop_unresolved",
            "val_exact_accuracy": None,
            "post_epoch": 6399,
            "duration_s": 0.125,
            "stable_checkpoint_path": str(stable),
        },
    )


def _blocked_pass3(stable: Path) -> dict[str, object]:
    return {
        "honest_verdict": "blocked_pass2_noop_unresolved",
        "val_exact_accuracy": None,
        "post_epoch": 6399,
        "pass2_post_epoch": 6399,
        "duration_s": 0.115,
        "stable_checkpoint_path": str(stable),
        "blocked_cause": "pass2 verdict=blocked_pass1_noop_unresolved; upstream no-op fixture",
    }


def _complete_pass3(stable: Path, *, epoch: int = 6409, val: float = 0.34) -> dict[str, object]:
    return {
        "honest_verdict": f"complete: pass3_trained_post_epoch={epoch}_val={val:.4f}_delta=0.0300",
        "val_exact_accuracy": val,
        "delta_vs_pass2": 0.03,
        "post_epoch": epoch,
        "pass2_post_epoch": epoch - 4,
        "duration_s": 131.5,
        "stable_checkpoint_path": str(stable),
    }


def test_req_learn_4149_spec_declares_pass4_contract() -> None:
    """REQ-LEARN-4149: OpenSpec declares the final pass4 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4149" in spec
    assert "SCENARIO-LEARN-4149" in spec
    assert "SCENARIO-LEARN-4149-BLOCKED-PASS3" in spec
    assert "SCENARIO-LEARN-4149-EARLY-CONVERGED" in spec
    assert "results/experiment_4149_sudoku_accumulate_pass4_convergence.json" in spec
    for field in exp4149.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_learn_4149_blocked_pass3_stops_and_reports_trajectory(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4149-BLOCKED-PASS3: upstream no-op forbids pass4 retrain."""

    _make_ready_repo(tmp_path)
    stable = _write_checkpoint(tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt", epoch=6399)
    _write_prior_lineage(tmp_path, stable)
    _write_pass(tmp_path / "results" / exp4148.RESULT_FILENAME, _blocked_pass3(stable))
    trainer_calls = 0

    def forbidden_runner(_config: exp4149.Exp4149Config, _current_epoch: int) -> exp4116.ResumeRunResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("blocked pass3 must stop before native training")

    output = tmp_path / "results" / exp4149.RESULT_FILENAME
    artifact = exp4149.run_experiment(
        repo_root=tmp_path,
        output_path=output,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        trainer_runner=forbidden_runner,
    )

    assert trainer_calls == 0
    assert artifact["honest_verdict"] == "blocked_pass3_noop_unresolved"
    assert artifact["val_exact_accuracy"] == BASELINE_VAL
    assert artifact["matches_published_087"] is False
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["native_trainer_launched"] is False
    assert artifact["command"] == []
    assert "pass3 verdict=blocked_pass2_noop_unresolved" in artifact["blocked_cause"]
    assert "upstream no-op fixture" in artifact["blocked_cause"]
    assert artifact["stable_checkpoint_path"] == str(stable)
    assert [row["pass_label"] for row in artifact["val_trajectory_v384"]] == [
        "v384_start",
        "pass1",
        "pass2",
        "pass3",
        "pass4",
    ]
    assert artifact["val_trajectory_v384"][0]["val_exact_accuracy"] == BASELINE_VAL
    assert artifact["val_trajectory_v384"][-1]["effective_val_exact_accuracy"] == BASELINE_VAL
    for field, principle in exp4149.FIELD_PRINCIPLES.items():
        assert artifact["field_principles"][field] == principle
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_scenario_learn_4149_early_converged_skips_training(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4149-EARLY-CONVERGED: pass3 target hit stops pass4 cleanly."""

    _make_ready_repo(tmp_path)
    stable = _write_checkpoint(tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt", epoch=6410)
    _write_prior_lineage(tmp_path, stable)
    _write_pass(tmp_path / "results" / exp4148.RESULT_FILENAME, _complete_pass3(stable, epoch=6410, val=0.872))
    trainer_calls = 0

    def forbidden_runner(_config: exp4149.Exp4149Config, _current_epoch: int) -> exp4116.ResumeRunResult:
        nonlocal trainer_calls
        trainer_calls += 1
        raise AssertionError("early convergence must skip native training")

    artifact = exp4149.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4149.RESULT_FILENAME,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        trainer_runner=forbidden_runner,
    )

    assert trainer_calls == 0
    assert artifact["honest_verdict"].startswith("complete: early_converged")
    assert artifact["val_exact_accuracy"] == 0.872
    assert artifact["matches_published_087"] is True
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["native_trainer_launched"] is False
    assert artifact["command"] == []
    assert artifact["val_trajectory_v384"][-1]["pass_label"] == "pass4"
    assert artifact["val_trajectory_v384"][-1]["val_exact_accuracy"] == 0.872


def test_scenario_learn_4149_real_pass_reports_final_metric_and_command(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4149: pass4 must advance epoch and report final solve metric."""

    _make_ready_repo(tmp_path)
    stable = _write_checkpoint_without_epoch(
        tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    )
    _write_prior_lineage(tmp_path, stable)
    _write_pass(tmp_path / "results" / exp4148.RESULT_FILENAME, _complete_pass3(stable, epoch=6409, val=0.84))
    calls = 0

    def fake_runner(config: exp4149.Exp4149Config, current_epoch: int) -> exp4116.ResumeRunResult:
        nonlocal calls
        calls += 1
        assert current_epoch == 6409
        post_epoch = current_epoch + 5
        _write_metrics(config.pass_run_dir(), val=0.866, epoch=post_epoch)
        _write_checkpoint(config.stable_checkpoint_path, epoch=post_epoch)
        metric = exp4107.ExactAccuracy(
            "val/exact_accuracy",
            0.866,
            config.pass_run_dir() / "csv" / "version_0" / "metrics.csv",
        )
        return exp4116.ResumeRunResult(
            return_code=0,
            stable_checkpoint_path=config.stable_checkpoint_path,
            checkpoint_reload_ok=True,
            checkpoint_reload_detail="loadable fixture",
            val_exact_accuracy=metric,
            cumulative_epochs=post_epoch + 1,
            duration_s=132.25,
            command=exp4149.build_train_command(config, current_epoch=current_epoch),
            stdout_tail=["trained fixture"],
            run_dir=config.pass_run_dir(),
        )

    artifact = exp4149.run_experiment(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp4149.RESULT_FILENAME,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=lambda: (True, "cuda fixture"),
        checkpoint_loader=lambda _path: (True, "loadable fixture"),
        trainer_runner=fake_runner,
    )

    assert calls == 1
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["val_exact_accuracy"] == 0.866
    assert artifact["matches_published_087"] is True
    assert artifact["post_epoch"] == 6414
    assert artifact["duration_s"] == 132.25
    assert artifact["acceptance_gate_passed"] is True
    assert artifact["native_trainer_launched"] is True
    assert f"ckpt_path={stable}" in artifact["command"]
    assert "+trainer.max_epochs=9409" in artifact["command"]
    assert "+trainer.max_time=00:01:00:00" in artifact["command"]
    assert any("+callbacks.exp4149_progress._target_=" in part for part in artifact["command"])
    assert artifact["val_trajectory_v384"][-1]["pass_label"] == "pass4"
    assert artifact["val_trajectory_v384"][-1]["val_exact_accuracy"] == 0.866


def test_req_learn_4149_schema_helpers_and_runtime_blockers(tmp_path: Path) -> None:
    """REQ-LEARN-4149: schema rejects fake complete and preserves blockers."""

    _make_ready_repo(tmp_path)
    stable = _write_checkpoint(tmp_path / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt", epoch=6409)
    _write_prior_lineage(tmp_path, stable)
    _write_pass(tmp_path / "results" / exp4148.RESULT_FILENAME, _complete_pass3(stable, epoch=6409, val=0.34))
    config = exp4149.Exp4149Config(repo_root=tmp_path)

    assert config.trainer_path == tmp_path / "nano-trm" / "src" / "nn" / "train.py"
    assert config.to_4127_config().stable_checkpoint_path == config.stable_checkpoint_path
    assert exp4149.build_train_env(config)["WANDB_DISABLED"] == "true"
    assert exp4149.matches_published_087(0.85) is True
    assert exp4149.matches_published_087(0.849) is False
    assert exp4149.pass3_has_real_training(_complete_pass3(stable)) is True
    assert exp4149.pass3_has_real_training(_blocked_pass3(stable)) is False
    assert exp4149.pass3_is_early_converged(_complete_pass3(stable, val=0.87)) is True

    blocked = exp4149.build_runtime_blocked_artifact(
        "blocked_cuda_unavailable",
        run_config=config,
        pass3_artifact=_complete_pass3(stable, epoch=6409, val=0.34),
        preconditions_checked=[exp4107.PreconditionCheck("cuda_available", False, "fixture")],
        duration_s=0.5,
    )
    assert blocked["honest_verdict"] == "blocked_cuda_unavailable"
    assert blocked["val_exact_accuracy"] == 0.34
    assert blocked["matches_published_087"] is False
    assert blocked["acceptance_gate_passed"] is False
    assert blocked["command"] == exp4149.build_train_command(config, current_epoch=6409)
    assert exp4149.artifact_schema_errors(blocked) == []

    invalid = dict(blocked)
    invalid.update(
        {
            "honest_verdict": "pending",
            "val_exact_accuracy": 2.0,
            "matches_published_087": "yes",
            "val_trajectory_v384": [],
            "stable_checkpoint_path": "",
            "duration_s": [1.0],
            "field_principles": {"honest_verdict": "wrong"},
        }
    )
    errors = exp4149.artifact_schema_errors(invalid)
    assert "honest_verdict must be terminal-prefixed" in errors
    assert "val_exact_accuracy must be numeric between 0 and 1" in errors
    assert "matches_published_087 must be a bare bool" in errors
    assert "val_trajectory_v384 must be a non-empty list with the v384_start row" in errors
    assert "stable_checkpoint_path must be a non-empty string" in errors
    assert "duration_s must be a scalar bounded number below 86400" in errors
    assert "field_principles must include the required operator principles" in errors

    fake_complete = dict(blocked)
    fake_complete["honest_verdict"] = "complete: fake"
    assert "complete trained verdict requires duration>120, epoch advance, real val, and a checkpoint" in (
        exp4149.artifact_schema_errors(fake_complete)
    )
    assert "missing required field honest_verdict" in exp4149.artifact_schema_errors({})

    missing = exp4149.load_json_artifact(tmp_path / "missing.json", label="pass3")
    assert "missing pass3 artifact" in missing["load_error"]
    assert "missing pass3 artifact" in exp4149.summarize_pass3_blocker(missing)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")
    assert "JSONDecodeError" in exp4149.load_json_artifact(bad_json, label="pass3")["load_error"]
    list_payload = tmp_path / "list.json"
    list_payload.write_text("[]", encoding="utf-8")
    assert "unexpected pass3 payload" in exp4149.load_json_artifact(list_payload, label="pass3")["load_error"]

    exp4149.write_result_artifact(tmp_path / "blocked.json", blocked)
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == blocked
    try:
        exp4149.validate_artifact(invalid)
    except ValueError as exc:
        assert "honest_verdict must be terminal-prefixed" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("validate_artifact should reject malformed artifacts")
