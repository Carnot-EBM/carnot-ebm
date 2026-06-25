"""Tests for the Exp 4727 active-probe controller.

Spec refs: REQ-ARC-WMTE-4727,
SCENARIO-ARC-WMTE-4727-ACTIVE-PROBE-SPLITS-POSTERIOR,
SCENARIO-ARC-WMTE-4727-ARTIFACT-CONTRACT.
"""

from __future__ import annotations

import numpy as np

from carnot import experiment_4727_active_probe_disambiguation as exp4727
from carnot.agentic.arc_active_probe import (
    ActiveProbeController,
    ProbeAction,
    make_hypothesis_posterior,
)
from carnot.agentic.arc_executable_world_model import predict_hypothesis_transition
from carnot.agentic.arc_world_model_trust_energy import WorldModelCandidate


def _set_cell(row: int, col: int, value: int):
    def _engine(grid, _action, _data):
        out = np.asarray(grid).copy()
        out[row, col] = value
        return out

    return _engine


def _noop(grid, _action, _data):
    return np.asarray(grid).copy()


def test_req_arc_wmte_4727_probe_scorer_splits_posterior_and_updates() -> None:
    """SCENARIO-ARC-WMTE-4727-ACTIVE-PROBE-SPLITS-POSTERIOR."""

    grid = np.zeros((2, 2), dtype=int)
    candidates = [
        WorldModelCandidate("left", _set_cell(0, 0, 1)),
        WorldModelCandidate("right", _set_cell(0, 1, 1)),
    ]
    controller = ActiveProbeController(
        make_hypothesis_posterior(candidates),
        probe_budget=2,
        concentration_threshold=0.9,
    )

    split = ProbeAction(6, {"x": 8, "y": 8})
    alias = ProbeAction(1, None)
    scores = controller.rank_probe_actions(
        grid,
        [alias, split],
        predictor=lambda hyp, start, action, data: (
            _noop(start, action, data)
            if action == 1
            else predict_hypothesis_transition(hyp, start, action, data)
        ),
    )

    assert scores[0].action == split
    assert scores[0].expected_information_gain > 0.6
    assert scores[0].verifier_is_oracle is False
    before = controller.posterior.entropy()
    observed = candidates[0].engine(grid, split.action, split.data)

    update = controller.observe_transition(grid, split, observed)

    assert controller.posterior.probability("left") > 0.9
    assert update.posterior_entropy_before == before
    assert update.posterior_entropy_after < update.posterior_entropy_before
    assert update.posterior_entropy_reduction > 0.6
    assert controller.diagnostics()["probe_actions_taken"] == 1


def test_req_arc_wmte_4727_energy_verifier_does_not_call_goal_oracle() -> None:
    """REQ-ARC-WMTE-4727: probe scoring is oracle-distinct from level completion."""

    def _forbidden_goal(_grid):
        raise AssertionError("probe scorer must not call the win-check")

    grid = np.zeros((2, 2), dtype=int)
    posterior = make_hypothesis_posterior(
        [
            WorldModelCandidate("left", _set_cell(1, 0, 2), _forbidden_goal),
            WorldModelCandidate("right", _set_cell(1, 1, 2), _forbidden_goal),
        ]
    )
    controller = ActiveProbeController(posterior)

    chosen = controller.choose_probe(grid, [ProbeAction(6, {"x": 16, "y": 16})])

    assert chosen is not None
    assert chosen.verifier_is_oracle is False
    assert chosen.expected_information_gain > 0.0


def test_req_arc_wmte_4727_artifact_contract_and_checksum() -> None:
    """SCENARIO-ARC-WMTE-4727-ARTIFACT-CONTRACT."""

    artifact = exp4727.build_artifact(
        preconditions_checked={
            "world_model_importable": True,
            "per_hypothesis_prediction_supported": True,
            "qwen_gguf_cached": True,
            "offline_arcade_ok": True,
            "qwen_props_verified": True,
            "proposer_port": 8920,
        },
        active_probe={
            "hypothesis_posterior_built": True,
            "probe_actions_taken": 1,
            "posterior_entropy_reduction": 0.693147,
            "generic_agent_reached_level": 1,
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "residual_cause": "probe_outcomes_aliased",
            "trace": [{"action": 6, "expected_information_gain": 0.693147}],
        },
        no_probe_ablation={"reached_level": 1, "budget": 2},
        live_path_lint={"passed": True},
        parity_test={"passed": True},
        proposer_served_model="Qwen3.5-9B-MTP",
        bare_control_passed=True,
        missing_verifier_gap_logged=True,
        duration_s=60.0,
    )

    errors = exp4727.artifact_schema_errors(artifact)

    assert errors == []
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["hypothesis_posterior_built"] is True
    assert artifact["probe_actions_taken"] == 1
    assert artifact["posterior_entropy_reduction"] == 0.693147
    assert artifact["no_probe_ablation_reached_level"] == 1
    assert artifact["null_methodology_note"]
    assert artifact["reproducibility_checksum"] == exp4727.payload_checksum(artifact)
