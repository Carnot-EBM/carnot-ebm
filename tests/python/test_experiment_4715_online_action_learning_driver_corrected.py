"""Tests for Exp 4715 corrected goal-free online action-learning driver.

Spec refs: REQ-ARC-FCP-4715, SCENARIO-ARC-FCP-4715.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from carnot.agentic.arc_frame_change_predictor import FrameChangeScorer, SmallFrameChangeCNN
from carnot.agentic.arc_online_action_effect_scorer import OnlineActionEffectScorer


REPO_ROOT = Path(__file__).resolve().parents[2]


def _frame(value: int, *, level: int = 0) -> Any:
    return SimpleNamespace(frame=np.full((4, 4), value, dtype=np.int16), levels_completed=level)


def test_req_arc_fcp_4715_spec_declares_corrected_driver_contract() -> None:
    """REQ-ARC-FCP-4715: OpenSpec declares the corrected driver fields and gates."""

    spec = (
        REPO_ROOT / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
    ).read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4715" in spec
    assert "goal-free online action-learning driver" in spec
    for field in (
        "online_warm_first_win",
        "online_scratch_first_win",
        "frozen_first_win",
        "online_warm_vs_frozen_delta",
        "cpu_train_step_ms",
        "goal_free_l2_reached",
        "proposer_served_model",
        "preconditions_checked",
    ):
        assert field in spec


def test_req_arc_fcp_4715_online_scorer_reset_restores_prior_and_optimizer() -> None:
    """REQ-ARC-FCP-4715: level reset restores the initial CNN prior and clears Adam state."""

    torch.manual_seed(4715)
    model = SmallFrameChangeCNN(num_colors=16, hidden_channels=4)
    scorer = OnlineActionEffectScorer(
        memory=None,
        cnn_scorer=FrameChangeScorer(model),
        train_enabled=True,
        fit_every=1,
        max_batch=1,
    )
    initial = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    scorer.observe_transition(_frame(0), action_id=1, data=None, after_frame=_frame(3))
    assert scorer._optimizer is not None
    assert scorer._optimizer.state, "Adam state should exist after a train step"

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.25)

    scorer.reset(level=1, reset_to_prior=True)

    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, initial[name]), f"{name} was not restored to the prior"
    assert scorer._buffer == []
    assert scorer._seen == set()
    assert scorer._obs_since_fit == 0
    assert scorer._optimizer is not None
    assert not scorer._optimizer.state, "Adam moments must be cleared at level reset"
    assert scorer.diagnostics()["resets_to_prior"] == 1


def test_req_arc_fcp_4715_stepwise_explorer_resets_online_scorer_on_level_up(
    monkeypatch: Any,
) -> None:
    """REQ-ARC-FCP-4715: StepwiseExplorer resets the scorer only after observed level progress."""

    import carnot.agentic.arc_competition_agent as comp

    monkeypatch.setattr(comp.StepwiseExplorer, "_candidates", lambda *_args, **_kwargs: [])

    class _FakeScorer:
        def __init__(self) -> None:
            self.observed = 0
            self.resets: list[dict[str, Any]] = []

        def observe_transition(self, *_args: Any, **_kwargs: Any) -> None:
            self.observed += 1

        def reset(self, **kwargs: Any) -> None:
            self.resets.append(dict(kwargs))

    scorer = _FakeScorer()
    explorer = comp.StepwiseExplorer(online_discriminative=False, frame_change_scorer=scorer)
    start = _frame(0, level=0)
    explorer._ingest(start)
    assert scorer.resets == []

    explorer.awaiting = {
        "origin": explorer.root,
        "action": 1,
        "data": None,
        "grid": np.asarray(start.frame),
        "level_before": 0,
        "previous_frame": start,
    }
    explorer._ingest(_frame(1, level=1))

    assert scorer.observed == 1
    assert scorer.resets == [{"level": 1, "reset_to_prior": True}]


def test_req_arc_fcp_4715_gated_engine_defaults_to_cell_recall() -> None:
    """REQ-ARC-FCP-4715: cheap floor defaults gated_engine_from_transitions to cell_recall."""

    from carnot.agentic.arc_live_ttt import gated_engine_from_transitions

    signature = inspect.signature(gated_engine_from_transitions)
    assert signature.parameters["trust_metric"].default == "cell_recall"


def test_req_arc_fcp_4715_artifact_schema_and_null_verdict() -> None:
    """REQ-ARC-FCP-4715: corrected 4715 artifact emits required fields for an honest null."""

    from carnot import experiment_4715_online_action_learning_driver_corrected as mod

    artifact = mod.build_artifact(
        arm_metrics={
            "frozen": 0.04,
            "online-scratch": 0.04,
            "online-warm": 0.04,
        },
        preconditions_checked={
            "cuda_available": True,
            "qwen_gguf_cached": True,
            "offline_arcade_ok": True,
            "arc_go_explore_importable": True,
            "qwen_props_verified": True,
            "ok": True,
        },
        cpu_train_step_ms=2.75,
        proposer_served_model="Qwen3.5-9B-MTP",
        parity_test_green=True,
        live_path_reachable=True,
        bare_control_passed=True,
        false_negative_risk_checked=True,
        goal_free_probe={
            "goal_free_l2_reached": False,
            "offline_reproduced": False,
            "reproduced_levels": 0,
        },
        source_artifacts={
            "frozen": "results/experiment_4710_online_action_learning_arms_frozen.json",
            "online-scratch": "results/experiment_4710_online_action_learning_arms_online_scratch.json",
            "online-warm": "results/experiment_4710_online_action_learning_arms_online_warm_propose.json",
        },
    )

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"].startswith(
        "complete: online_action_learning_no_first_win_lift_residual_"
    )
    assert artifact["online_warm_vs_frozen_delta"] == 0.0
    assert artifact["null_methodology_note"]
    assert artifact["verifier_is_oracle"] is False
    assert artifact["proposer_served_model"] == "Qwen3.5-9B-MTP"
    assert artifact["chosen_submitted_config"]["trust_metric"] == "cell_recall"
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_req_arc_fcp_4715_success_branch_and_schema_errors() -> None:
    """REQ-ARC-FCP-4715: success requires warm lift plus reproduced goal-free L2."""

    from carnot import experiment_4715_online_action_learning_driver_corrected as mod

    artifact = mod.build_artifact(
        arm_metrics={"frozen": 0.04, "online-scratch": 0.05, "online-warm": 0.11},
        preconditions_checked={"ok": True},
        cpu_train_step_ms=5.0,
        proposer_served_model="Qwen3.5-9B-MTP",
        parity_test_green=True,
        live_path_reachable=True,
        bare_control_passed=True,
        false_negative_risk_checked=True,
        goal_free_probe={
            "goal_free_l2_reached": True,
            "offline_reproduced": True,
            "reproduced_levels": 2,
        },
        source_artifacts={"frozen": "a", "online-scratch": "b", "online-warm": "c"},
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["chosen_submitted_config"]["submitted_recommendation"] == (
        "enable_online_warm_goal_free_driver"
    )
    assert artifact["null_methodology_note"] == ""
    assert mod.artifact_schema_errors(artifact) == []

    missing = dict(artifact)
    missing.pop("cpu_train_step_ms")
    missing["reproducibility_checksum"] = mod.payload_checksum(missing)
    assert "missing:cpu_train_step_ms" in mod.artifact_schema_errors(missing)

    bad = dict(artifact, honest_verdict="no_prefix", verifier_is_oracle=True)
    bad["proposer_served_model"] = "gemma-4-12B-it"
    bad["reproducibility_checksum"] = "sha256:bad"
    errors = mod.artifact_schema_errors(bad)
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "proposer_served_model_not_qwen" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_req_arc_fcp_4715_load_arm_metrics_and_blocked_artifact(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """REQ-ARC-FCP-4715: arm metrics are content-addressed and blocked artifacts stay schema-valid."""

    from carnot import experiment_4715_online_action_learning_driver_corrected as mod

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

    blocked = mod._blocked_artifact(
        {"blocked_resource": "blocked_cuda_unavailable", "proposer_served_model": ""},
        duration_s=0.1,
    )
    assert blocked["honest_verdict"] == "blocked_cuda_unavailable"
    assert blocked["preconditions_checked"]["blocked_resource"] == "blocked_cuda_unavailable"


def test_req_arc_fcp_4715_cpu_train_step_and_run_check(monkeypatch: Any) -> None:
    """REQ-ARC-FCP-4715: CPU timing and subprocess check helpers report concrete outcomes."""

    from carnot import experiment_4715_online_action_learning_driver_corrected as mod

    assert mod.measure_cpu_train_step_ms() > 0.0

    ok = mod._run_check(["python", "-c", "print('ok')"], REPO_ROOT)
    assert ok["passed"] is True
    assert ok["returncode"] == 0
    assert "ok" in ok["output_tail"]

    class _Boom:
        pass

    def _raise(*_args: Any, **_kwargs: Any) -> _Boom:
        raise RuntimeError("boom")

    monkeypatch.setattr(mod.subprocess, "run", _raise)
    failed = mod._run_check(["python", "-c", "pass"], REPO_ROOT)
    assert failed["passed"] is False
    assert failed["returncode"] == -1
    assert "boom" in failed["output_tail"]
