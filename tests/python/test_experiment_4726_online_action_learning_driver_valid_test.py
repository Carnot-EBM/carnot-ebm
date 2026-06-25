"""Tests for Exp 4726 valid online action-learning driver gate.

Spec refs: REQ-ARC-FCP-4726, SCENARIO-ARC-FCP-4726.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from carnot.agentic.arc_frame_change_predictor import FrameChangeScorer, SmallFrameChangeCNN
from carnot.agentic.arc_online_action_effect_scorer import OnlineActionEffectScorer


REPO_ROOT = Path(__file__).resolve().parents[2]


def _frame(value: int, *, shape: tuple[int, int] = (5, 5)) -> Any:
    grid = np.full(shape, value, dtype=np.int16)
    grid[0, 0] = value
    return SimpleNamespace(frame=grid, levels_completed=0)


def test_req_arc_fcp_4726_spec_declares_non_degeneracy_gate() -> None:
    """REQ-ARC-FCP-4726: OpenSpec declares the valid-driver gate and required fields."""

    spec = (
        REPO_ROOT / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
    ).read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4726" in spec
    assert "Valid Online Driver Non-Degeneracy Gate" in spec
    for field in (
        "arms_non_degenerate",
        "per_arm_action_distribution_distinct",
        "online_train_steps_executed",
        "online_warm_vs_frozen_delta",
        "cpu_train_step_ms",
        "proposer_served_model",
        "preconditions_checked",
    ):
        assert field in spec


def test_req_arc_fcp_4726_online_scorer_reports_positive_gradient_norm() -> None:
    """REQ-ARC-FCP-4726: diagnostics expose positive-gradient train steps."""

    torch.manual_seed(4726)
    scorer = OnlineActionEffectScorer(
        memory=None,
        cnn_scorer=FrameChangeScorer(SmallFrameChangeCNN(num_colors=16, hidden_channels=4)),
        fit_every=1,
        max_batch=1,
    )

    scorer.observe_transition(_frame(0), action_id=1, data=None, after_frame=_frame(3))
    diag = scorer.diagnostics()

    assert diag["fits"] >= 1
    assert diag["online_train_steps_executed"] >= 1
    assert diag["train_steps_with_positive_grad_norm"] >= 1
    assert diag["last_gradient_norm"] > 0.0
    assert diag["max_gradient_norm"] >= diag["last_gradient_norm"]


def test_req_arc_fcp_4726_non_degeneracy_gate_passes_on_live_scorer_fixture() -> None:
    """REQ-ARC-FCP-4726: synthetic live-hook fixture proves arms differ before lift."""

    from carnot import experiment_4726_online_action_learning_driver_valid_test as mod

    gate = mod.run_non_degeneracy_gate(seed=4726)

    assert gate["arms_non_degenerate"] is True
    assert gate["per_arm_action_distribution_distinct"] is True
    assert gate["online_train_steps_executed"] > 0
    assert gate["gradient_norms_positive"] is True
    assert gate["coordinate_head_differs_from_frozen"] is True
    assert gate["arm_action_histograms"]["frozen"] != gate["arm_action_histograms"]["online-scratch"]
    assert gate["arm_action_histograms"]["online-warm"] != gate["arm_action_histograms"]["frozen"]


def test_req_arc_fcp_4726_artifact_schema_flat_delta_honest_null() -> None:
    """REQ-ARC-FCP-4726: flat non-degenerate arms emit the TAUTOLOGY-safe null markers."""

    from carnot import experiment_4726_online_action_learning_driver_valid_test as mod

    gate = {
        "arms_non_degenerate": True,
        "per_arm_action_distribution_distinct": True,
        "online_train_steps_executed": 2,
        "gradient_norms_positive": True,
        "coordinate_head_differs_from_frozen": True,
        "arm_action_histograms": {
            "frozen": {"1": 2},
            "online-scratch": {"2": 2},
            "online-warm": {"6": 2},
        },
    }
    artifact = mod.build_artifact(
        arm_metrics={"frozen": 0.04, "online-scratch": 0.04, "online-warm": 0.04},
        preconditions_checked={
            "ok": True,
            "cuda_available": True,
            "qwen_gguf_cached": True,
            "offline_arcade_ok": True,
            "arc_go_explore_importable": True,
            "qwen_props_verified": True,
        },
        non_degeneracy_gate=gate,
        cpu_train_step_ms=3.25,
        proposer_served_model="Qwen3.5-9B-MTP",
        parity_test_green=True,
        live_path_reachable=True,
        bare_control_passed=True,
        false_negative_risk_checked=True,
        goal_free_probe={"goal_free_l2_reached": False, "offline_reproduced": False, "reproduced_levels": 0},
        source_artifacts={"frozen": "f", "online-scratch": "s", "online-warm": "w"},
        source_artifact_checksums={"frozen": "sha256:f", "online-scratch": "sha256:s", "online-warm": "sha256:w"},
    )

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"].startswith(
        "complete: online_action_learning_no_first_win_lift_residual_"
    )
    assert artifact["arms_non_degenerate"] is True
    assert artifact["positive_control_passed"] is True
    assert artifact["null_delta_methodology_note"]
    assert artifact["online_warm_vs_frozen_delta"] == 0.0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_fcp_4726_success_branches_and_schema_errors() -> None:
    """REQ-ARC-FCP-4726: success can come from first-win lift or reproduced L2."""

    from carnot import experiment_4726_online_action_learning_driver_valid_test as mod

    gate = {
        "arms_non_degenerate": True,
        "per_arm_action_distribution_distinct": True,
        "online_train_steps_executed": 3,
        "gradient_norms_positive": True,
        "coordinate_head_differs_from_frozen": True,
    }
    common = dict(
        preconditions_checked={"ok": True},
        non_degeneracy_gate=gate,
        cpu_train_step_ms=2.0,
        proposer_served_model="Qwen3.5-9B-MTP",
        parity_test_green=True,
        live_path_reachable=True,
        bare_control_passed=True,
        false_negative_risk_checked=True,
        source_artifacts={"frozen": "f", "online-scratch": "s", "online-warm": "w"},
    )
    first_win = mod.build_artifact(
        arm_metrics={"frozen": 0.04, "online-scratch": 0.05, "online-warm": 0.10},
        goal_free_probe={"goal_free_l2_reached": False, "offline_reproduced": False, "reproduced_levels": 0},
        **common,
    )
    assert first_win["honest_verdict"].startswith("success: online_warm_beats_frozen_")
    assert first_win["solve_provenance"] == "development_proxy"

    l2 = mod.build_artifact(
        arm_metrics={"frozen": 0.04, "online-scratch": 0.04, "online-warm": 0.04},
        goal_free_probe={"goal_free_l2_reached": True, "offline_reproduced": True, "reproduced_levels": 2},
        **common,
    )
    assert l2["honest_verdict"].startswith("success:")
    assert l2["solve_provenance"] == "live_agent_self_discovery"

    missing = dict(first_win)
    missing.pop("cpu_train_step_ms")
    missing["reproducibility_checksum"] = mod.payload_checksum(missing)
    assert "missing:cpu_train_step_ms" in mod.artifact_schema_errors(missing)

    bad = dict(first_win, honest_verdict="no_prefix", verifier_is_oracle=True)
    bad["proposer_served_model"] = "gemma-4-12B-it"
    bad["reproducibility_checksum"] = "sha256:bad"
    errors = mod.artifact_schema_errors(bad)
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "proposer_served_model_not_qwen" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_req_arc_fcp_4726_non_degeneracy_gate_diagnostic_lists_failed_witnesses(
    monkeypatch: Any,
) -> None:
    """REQ-ARC-FCP-4726: failed gate names which no-op signature occurred."""

    from carnot import experiment_4726_online_action_learning_driver_valid_test as mod
    from carnot.agentic.arc_online_action_effect_scorer import OnlineActionEffectScorer

    monkeypatch.setattr(mod, "_train_online_fixture", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "_pairwise_histogram_distances",
        lambda _histograms: {
            "frozen__online-scratch": 0.0,
            "frozen__online-warm": 0.0,
            "online-scratch__online-warm": 0.0,
        },
    )
    monkeypatch.setattr(mod, "_top_clicks", lambda *_args, **_kwargs: [(1, 1)])
    monkeypatch.setattr(OnlineActionEffectScorer, "propose_coords", lambda *_args, **_kwargs: [(1, 1)])

    gate = mod.run_non_degeneracy_gate(seed=4726)

    assert gate["arms_non_degenerate"] is False
    assert "action_distributions_identical" in gate["diagnostic"]
    assert "online_train_steps_missing_positive_grad" in gate["diagnostic"]
    assert "coordinate_head_matches_frozen_prior" in gate["diagnostic"]


def test_req_arc_fcp_4726_degenerate_and_blocked_artifacts_are_schema_valid() -> None:
    """REQ-ARC-FCP-4726: degenerate arms and blocked preconditions stop honestly."""

    from carnot import experiment_4726_online_action_learning_driver_valid_test as mod

    degenerate = mod.build_artifact(
        arm_metrics={"frozen": 0.04, "online-scratch": 0.04, "online-warm": 0.04},
        preconditions_checked={"ok": True},
        non_degeneracy_gate={
            "arms_non_degenerate": False,
            "per_arm_action_distribution_distinct": False,
            "online_train_steps_executed": 0,
            "gradient_norms_positive": False,
            "coordinate_head_differs_from_frozen": False,
            "diagnostic": "byte_identical_actions",
        },
        cpu_train_step_ms=1.0,
        proposer_served_model="Qwen3.5-9B-MTP",
        parity_test_green=False,
        live_path_reachable=False,
        bare_control_passed=False,
        false_negative_risk_checked=False,
        goal_free_probe={"goal_free_l2_reached": False, "offline_reproduced": False, "reproduced_levels": 0},
        source_artifacts={"frozen": "f", "online-scratch": "s", "online-warm": "w"},
    )
    assert degenerate["honest_verdict"] == (
        "complete: online_driver_arms_degenerate_confirmed_harness_bug"
    )
    assert degenerate["chosen_submitted_config"] == "unchanged"
    assert mod.artifact_schema_errors(degenerate) == []

    blocked = mod._blocked_artifact(
        {"blocked_resource": "blocked_cuda_unavailable", "proposer_served_model": ""},
        duration_s=0.1,
    )
    assert blocked["honest_verdict"] == "blocked_cuda_unavailable"
    assert blocked["preconditions_checked"]["blocked_resource"] == "blocked_cuda_unavailable"
    assert mod.artifact_schema_errors(blocked) == []


def test_req_arc_fcp_4726_load_arm_metrics_accepts_4710_summary_alias(tmp_path: Path, monkeypatch: Any) -> None:
    """REQ-ARC-FCP-4726: arm metrics are content-addressed and alias prior summary names."""

    from carnot import experiment_4726_online_action_learning_driver_valid_test as mod

    rels = {
        "frozen": Path("frozen.json"),
        "online-scratch": Path("scratch.json"),
        "online-warm": Path("warm.json"),
    }
    for index, rel in enumerate(rels.values()):
        (tmp_path / rel).write_text(
            json.dumps({"first_win_rate": 0.04 + index * 0.01}), encoding="utf-8"
        )
    monkeypatch.setattr(mod, "ARM_ARTIFACTS", rels)

    metrics, sources, checksums = mod.load_arm_metrics(tmp_path)

    assert metrics == {"frozen": 0.04, "online-scratch": 0.05, "online-warm": 0.06}
    assert sources["online-warm"] == "warm.json"
    assert checksums["frozen"].startswith("sha256:")
