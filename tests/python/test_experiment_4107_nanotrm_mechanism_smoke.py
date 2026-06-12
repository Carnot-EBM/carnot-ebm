"""Tests for Exp 4107 native nano-trm mechanism smoke.

Spec refs: REQ-LEARN-4107, SCENARIO-LEARN-4107,
SCENARIO-LEARN-4107-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"


def _cuda_ok() -> tuple[bool, str]:
    return True, "cuda fixture"


def _cuda_missing() -> tuple[bool, str]:
    return False, "torch.cuda.is_available()=False"


def _make_ready_repo(tmp_path: Path) -> Path:
    trainer = tmp_path / "nano-trm" / "src" / "nn" / "train.py"
    trainer.parent.mkdir(parents=True, exist_ok=True)
    trainer.write_text("# trainer fixture\n", encoding="utf-8")
    save_parent = tmp_path / "results" / "trm_runs"
    save_parent.mkdir(parents=True, exist_ok=True)
    return save_parent


def test_req_learn_4107_spec_declares_required_contract() -> None:
    """REQ-LEARN-4107: OpenSpec declares the exact metric and checkpoint gate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4107" in spec
    assert "SCENARIO-LEARN-4107" in spec
    assert "SCENARIO-LEARN-4107-BLOCKED" in spec
    for field in exp4107.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    assert "blocked_nanotrm_or_uv_missing" in spec
    assert "duration_s > 60" in spec
    assert "val/q_halt_accuracy alone SHALL NOT satisfy" in spec


def test_req_learn_4107_preconditions_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-4107: missing runtime resources map to blocked verdicts."""

    save_parent = _make_ready_repo(tmp_path)

    checks, blocker = exp4107.check_preconditions(
        repo_root=tmp_path,
        save_parent=save_parent,
        uv_resolver=lambda _name: None,
        cuda_checker=_cuda_ok,
    )
    assert blocker == "blocked_nanotrm_or_uv_missing"
    assert checks[0].resource == "uv"
    assert checks[0].available is False

    (tmp_path / "nano-trm" / "src" / "nn" / "train.py").unlink()
    checks, blocker = exp4107.check_preconditions(
        repo_root=tmp_path,
        save_parent=save_parent,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=_cuda_ok,
    )
    assert blocker == "blocked_nanotrm_or_uv_missing"
    assert checks[1].resource == "nanotrm_trainer"
    assert checks[1].available is False

    _make_ready_repo(tmp_path)
    checks, blocker = exp4107.check_preconditions(
        repo_root=tmp_path,
        save_parent=save_parent,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=_cuda_missing,
    )
    assert blocker == "blocked_cuda_unavailable"
    assert any(check.resource == "cuda_available" and not check.available for check in checks)

    checks, blocker = exp4107.check_preconditions(
        repo_root=tmp_path,
        save_parent=tmp_path / "missing" / "trm_runs",
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=_cuda_ok,
    )
    assert blocker == "blocked_save_dir_unwritable"
    assert checks[-1].resource == "persistent_save_parent"


def test_req_learn_4107_metric_parser_requires_real_exact_accuracy(tmp_path: Path) -> None:
    """REQ-LEARN-4107: q_halt-only metrics are not accepted as solve accuracy."""

    metrics = tmp_path / "csv" / "version_0" / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        "epoch,step,val/q_halt_accuracy,val/exact_accuracy,train/exact_accuracy\n"
        "0,0,1.0,,0.125\n"
        "1,10,1.0,0.375,0.5\n",
        encoding="utf-8",
    )

    exact = exp4107.extract_latest_exact_accuracy(tmp_path)

    assert exact == exp4107.ExactAccuracy(
        metric_name="val/exact_accuracy",
        value=0.375,
        metrics_path=metrics,
    )

    metrics.write_text("epoch,step,val/q_halt_accuracy\n0,0,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact_accuracy"):
        exp4107.extract_latest_exact_accuracy(tmp_path)


def test_scenario_learn_4107_success_artifact_clears_acceptance_gate(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4107: checkpoint reload plus exact metric clears the gate."""

    checkpoint = tmp_path / "results" / "trm_runs" / "run" / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    exact = exp4107.ExactAccuracy("val/exact_accuracy", 0.75, tmp_path / "metrics.csv")
    run = exp4107.NanoTrmRunResult(
        return_code=0,
        checkpoint_path=checkpoint,
        checkpoint_reload_ok=True,
        exact_accuracy=exact,
        duration_s=125.25,
        command=["uv", "run", "python", "src/nn/train.py"],
        stdout_tail=["val/exact_accuracy=0.75"],
        save_dir=checkpoint.parents[1],
    )

    artifact = exp4107.build_success_artifact(
        run_result=run,
        preconditions_checked=[
            exp4107.PreconditionCheck("uv", True, "/usr/bin/uv"),
            exp4107.PreconditionCheck("cuda_available", True, "cuda fixture"),
        ],
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["nanotrm_trainer_checkpoint_ok"] is True
    assert artifact["exact_accuracy"] == 0.75
    assert artifact["exact_accuracy_metric"] == "val/exact_accuracy"
    assert artifact["checkpoint_path"] == str(checkpoint)
    assert artifact["duration_s"] == 125.25
    assert artifact["acceptance_gate_passed"] is True
    assert exp4107.artifact_schema_errors(artifact) == []


def test_scenario_learn_4107_verifies_persistent_hydra_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-LEARN-4107: native Hydra output can carry checkpoint and metrics."""

    save_parent = _make_ready_repo(tmp_path)
    config = exp4107.NanoTrmRunConfig(
        repo_root=tmp_path,
        save_parent=save_parent,
        save_dir=save_parent / "run",
    )
    checkpoint = Path(config.hydra_run_dir) / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    metrics = Path(config.hydra_run_dir) / "csv" / "version_0" / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("epoch,val/exact_accuracy,val/q_halt_accuracy\n99,1.0,1.0\n", encoding="utf-8")
    monkeypatch.setattr(exp4107, "_load_torch_checkpoint", lambda _path: (True, "torch.load ok"))

    result = exp4107.verify_completed_native_run(
        config,
        duration_s=741.76,
        return_code=0,
        stdout_tail=["Trainer.fit stopped: max_epochs reached"],
    )

    assert result.save_dir == Path(config.hydra_run_dir)
    assert result.checkpoint_path == checkpoint
    assert result.checkpoint_reload_ok is True
    assert result.exact_accuracy == exp4107.ExactAccuracy("val/exact_accuracy", 1.0, metrics)


def test_req_learn_4107_schema_rejects_fabrication_signals(tmp_path: Path) -> None:
    """REQ-LEARN-4107: schema catches q_halt-only and sub-minute success claims."""

    artifact = exp4107.build_blocked_artifact(
        "blocked_cuda_unavailable",
        preconditions_checked=[exp4107.PreconditionCheck("cuda_available", False, "no cuda")],
    )
    assert artifact["nanotrm_trainer_checkpoint_ok"] is False
    assert artifact["exact_accuracy"] is None
    assert artifact["checkpoint_path"] is None
    assert exp4107.artifact_schema_errors(artifact) == []

    bad = dict(artifact)
    bad["honest_verdict"] = "complete: fabricated"
    bad["nanotrm_trainer_checkpoint_ok"] = True
    bad["exact_accuracy"] = None
    bad["checkpoint_path"] = str(tmp_path / "missing.ckpt")
    bad["duration_s"] = 12.0
    bad["acceptance_gate_passed"] = True

    errors = exp4107.artifact_schema_errors(bad)

    assert "successful gate requires exact_accuracy" in errors
    assert "successful gate requires duration_s > 60" in errors
    assert "successful gate requires an existing .ckpt checkpoint_path" in errors


def test_scenario_learn_4107_run_experiment_writes_artifact_with_injected_runner(
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-4107: run_experiment writes the stable deliverable JSON."""

    save_parent = _make_ready_repo(tmp_path)
    def fake_runner(config: exp4107.NanoTrmRunConfig) -> exp4107.NanoTrmRunResult:
        assert config.save_dir.parent == save_parent
        checkpoint = config.save_dir / "checkpoints" / "last.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        metrics = config.save_dir / "csv" / "version_0" / "metrics.csv"
        metrics.parent.mkdir(parents=True)
        metrics.write_text("epoch,val/exact_accuracy\n0,0.5\n", encoding="utf-8")
        return exp4107.NanoTrmRunResult(
            return_code=0,
            checkpoint_path=checkpoint,
            checkpoint_reload_ok=True,
            exact_accuracy=exp4107.extract_latest_exact_accuracy(config.save_dir),
            duration_s=75.0,
            command=exp4107.build_train_command(config),
            stdout_tail=["done"],
            save_dir=config.save_dir,
        )

    output_path = tmp_path / "results" / exp4107.RESULT_FILENAME
    artifact = exp4107.run_experiment(
        repo_root=tmp_path,
        output_path=output_path,
        save_parent=save_parent,
        uv_resolver=lambda _name: "/usr/bin/uv",
        cuda_checker=_cuda_ok,
        trainer_runner=fake_runner,
    )

    assert artifact["acceptance_gate_passed"] is True
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
