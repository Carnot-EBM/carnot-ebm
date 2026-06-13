"""Tests for Exp 4126 nano-trm LR resume correctness.

Spec refs: REQ-LEARN-4126, SCENARIO-LEARN-4126.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

from carnot import experiment_4126_lr_resume_correctness_fix as exp4126


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "self-learning" / "spec.md"
NANO_TRM_ROOT = REPO / "nano-trm"
if str(NANO_TRM_ROOT) not in sys.path:
    sys.path.insert(0, str(NANO_TRM_ROOT))

from src.nn.models.trm import TRMModule  # noqa: E402
from src.nn.modules import utils as trm_utils  # noqa: E402


def _tiny_trm() -> TRMModule:
    return TRMModule(
        hidden_size=16,
        num_layers=1,
        num_heads=1,
        max_grid_size=2,
        H_cycles=1,
        L_cycles=1,
        N_supervision=1,
        N_supervision_val=1,
        ffn_expansion=1,
        learning_rate=1e-4,
        learning_rate_emb=1e-4,
        warmup_steps=2_000,
        lr_min_ratio=0.01,
        puzzle_emb_dim=0,
        puzzle_emb_len=0,
        pos_emb_type="null",
        use_mlp_t=True,
        vocab_size=4,
        num_puzzles=0,
        batch_size=1,
        pad_value=0,
        seq_len=4,
        forward_dtype=torch.float32,
    )


def _legacy_checkpoint(completed_batches: int = 4_300) -> dict:
    return {
        "global_step": 0,
        "epoch": completed_batches - 1,
        "loops": {
            "fit_loop": {
                "epoch_loop.batch_progress": {
                    "total": {
                        "completed": completed_batches,
                        "processed": completed_batches,
                    }
                },
                "epoch_loop.state_dict": {"_batches_that_stepped": completed_batches - 1},
            }
        },
    }


def test_req_learn_4126_spec_declares_lr_resume_contract() -> None:
    """REQ-LEARN-4126: OpenSpec declares the LR resume artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-4126" in spec
    assert "SCENARIO-LEARN-4126" in spec
    assert "results/experiment_4126_lr_resume_correctness_fix.json" in spec
    for field in exp4126.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_req_learn_4126_manual_lr_step_is_checkpointed_and_restored() -> None:
    """REQ-LEARN-4126: manual LR step is persisted and beats legacy fallbacks."""

    checkpoint: dict = {}
    trm_utils.save_manual_lr_step(checkpoint, 4_350)

    assert checkpoint[trm_utils.MANUAL_LR_STEP_CHECKPOINT_KEY] == 4_350
    assert trm_utils.manual_lr_step_from_checkpoint(checkpoint) == 4_350

    legacy = _legacy_checkpoint(completed_batches=4_300)
    assert trm_utils.manual_lr_step_from_checkpoint(legacy) == 4_300
    assert trm_utils.manual_lr_step_from_checkpoint({"global_step": 3_200, "epoch": 10}) == 3_200
    assert trm_utils.manual_lr_step_from_checkpoint({"global_step": 0, "epoch": 4_299}) == 4_300


def test_scenario_learn_4126_trm_hooks_continue_legacy_checkpoint_step() -> None:
    """SCENARIO-LEARN-4126: TRM hook restores LR step from legacy loop progress."""

    model = _tiny_trm()
    model.on_load_checkpoint(_legacy_checkpoint(completed_batches=4_300))
    assert model.manual_step == 4_300

    checkpoint: dict = {}
    model.manual_step = 4_377
    model.on_save_checkpoint(checkpoint)
    resumed = _tiny_trm()
    resumed.on_load_checkpoint(checkpoint)

    assert resumed.manual_step == 4_377


def test_req_learn_4126_post_warmup_lr_uses_cosine_decay() -> None:
    """REQ-LEARN-4126: post-warmup LR is computed, not pinned to base LR."""

    model = _tiny_trm()
    model.total_steps = 4_000

    lr = model._lr_for_manual_step(base_lr=1e-4, current_step=3_000)
    expected = trm_utils.compute_lr(
        base_lr=1e-4,
        lr_warmup_steps=2_000,
        lr_min_ratio=0.01,
        current_step=3_000,
        total_steps=4_000,
    )

    assert math.isclose(lr, expected, rel_tol=0.0, abs_tol=1e-12)
    assert lr < 1e-4
    assert not math.isclose(lr, 1e-4, rel_tol=0.0, abs_tol=1e-12)


def test_scenario_learn_4126_metrics_and_artifact_prove_no_rewarm(tmp_path: Path) -> None:
    """SCENARIO-LEARN-4126: first validation-pass LR differs from fresh warmup."""

    metrics = tmp_path / "run" / "csv" / "version_0" / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(
        "\n".join(
            [
                "epoch,step,train/lr,val/exact_accuracy",
                "4349,4349,0.0000993,",
                "4399,4399,,0.11875",
                "",
            ]
        ),
        encoding="utf-8",
    )

    lr_points = exp4126.extract_train_lr_points(tmp_path / "run")
    val = exp4126.extract_latest_val_exact_accuracy(tmp_path / "run")
    artifact = exp4126.build_result_artifact(
        root_cause=exp4126.ROOT_CAUSE_LOCAL_MANUAL_STEP,
        lr_points=lr_points,
        val_exact_accuracy=val,
        stable_checkpoint_path=tmp_path / "stable" / "last.ckpt",
        duration_s=91.5,
        prior_last_lr=4.949999947712058e-06,
        command=["uv", "run", "python", "src/nn/train.py"],
    )

    assert artifact["lr_continuous_across_resume"] is True
    assert artifact["validation_first_lr"] == 0.0000993
    assert artifact["val_exact_accuracy"] == 0.11875
    assert artifact["fresh_warmup_lr"] == exp4126.FRESH_WARMUP_FIRST_LR
    assert artifact["prior_pass_last_lr"] == 4.949999947712058e-06
    exp4126.validate_artifact(artifact)


def test_req_learn_4126_artifact_helpers_fail_closed(tmp_path: Path) -> None:
    """REQ-LEARN-4126: parser and schema helpers reject stale or wrapper fields."""

    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "\n".join(
            [
                "epoch,step,train/lr,val/exact_accuracy",
                "bad,1.25,not-a-float,",
                "1.25,1.25,0.0000990,",
                "4350,4350,0.0000991,not-a-float",
                "4351,4351,,0.12",
                "",
            ]
        ),
        encoding="utf-8",
    )

    lr_points = exp4126.extract_train_lr_points(metrics)
    assert len(lr_points) == 2
    assert lr_points[0].epoch is None
    assert lr_points[0].step is None
    assert lr_points[1].epoch == 4350
    assert lr_points[1].step == 4350
    assert exp4126.extract_latest_val_exact_accuracy(metrics) == 0.12

    rewarm_artifact = exp4126.build_result_artifact(
        root_cause=exp4126.ROOT_CAUSE_LOCAL_MANUAL_STEP,
        lr_points=[],
        val_exact_accuracy=None,
        stable_checkpoint_path=tmp_path / "stable" / "last.ckpt",
        duration_s=12.0,
        prior_last_lr=None,
        command=[],
        stdout_tail=["no lr rows"],
    )
    assert rewarm_artifact["lr_continuous_across_resume"] is False
    assert rewarm_artifact["honest_verdict"].startswith("complete:")
    exp4126.validate_artifact(rewarm_artifact)

    output = tmp_path / "artifact.json"
    exp4126.write_result_artifact(output, rewarm_artifact)
    assert '"lr_continuous_across_resume": false' in output.read_text(encoding="utf-8")

    invalid_cases = [
        ({}, "missing required fields"),
        ({**rewarm_artifact, "honest_verdict": "pending"}, "terminal-prefixed"),
        ({**rewarm_artifact, "lr_continuous_across_resume": "true"}, "bare bool"),
        ({**rewarm_artifact, "duration_s": True}, "duration_s"),
        ({**rewarm_artifact, "duration_s": 4_800}, "duration_s"),
        (
            {
                **rewarm_artifact,
                "lr_continuous_across_resume": True,
                "validation_first_lr": exp4126.FRESH_WARMUP_FIRST_LR,
            },
            "non-rewarm",
        ),
    ]
    for artifact, message in invalid_cases:
        try:
            exp4126.validate_artifact(artifact)
        except ValueError as exc:
            assert message in str(exc)
        else:  # pragma: no cover - assertion guard
            raise AssertionError(f"expected validation failure containing {message!r}")
