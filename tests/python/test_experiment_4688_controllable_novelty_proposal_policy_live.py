"""Tests for Exp 4688 controllable-novelty proposal policy.

Spec refs: REQ-ARC-WMTE-4688, SCENARIO-ARC-WMTE-4688.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import pytest

if "coverage" in sys.modules or os.environ.get("CARNOT_SKIP_LIVE_IMPORT_UNDER_COVERAGE") == "1":
    comp = None
else:
    from carnot.agentic import arc_competition_agent as comp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _frame(values: list[list[int]], *, level: int = 0) -> SimpleNamespace:
    return SimpleNamespace(frame=np.asarray(values, dtype=np.int16), levels_completed=level)


class _EffectScorer:
    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = dict(scores)

    def candidate_score(self, _frame: Any, candidate: Any) -> float:
        if isinstance(candidate, dict):
            action = int(candidate.get("action", candidate.get("action_id", 0)))
        else:
            action = int(getattr(candidate, "action_id", 0))
        return float(self.scores.get(action, 0.0))


def test_req_arc_wmte_4688_spec_declares_controllable_novelty_contract() -> None:
    """REQ-ARC-WMTE-4688: OpenSpec declares the live proposal contract."""

    from carnot import experiment_4688_controllable_novelty_proposal_policy_live as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4688" in spec
    assert "SCENARIO-ARC-WMTE-4688" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4688_controllability_gate_rejects_cosmetic_raw_change() -> None:
    """REQ-ARC-WMTE-4688: gate uses action-effect evidence, not raw frame novelty."""

    from carnot.agentic.arc_controllable_novelty import (
        ControllableNoveltyConfig,
        ControllableNoveltyProposalPolicy,
    )

    before = _frame([[0, 0], [0, 0]])
    cosmetic = _frame([[9, 0], [0, 0]])
    action = {"action": 2, "data": None}

    gated = ControllableNoveltyProposalPolicy(
        ControllableNoveltyConfig(enabled=True, controllability_gate=True),
        action_effect_scorer=_EffectScorer({2: 0.0}),
    )
    event = gated.record_transition(before, cosmetic, action)

    raw = ControllableNoveltyProposalPolicy(
        ControllableNoveltyConfig(
            enabled=True,
            controllability_gate=False,
            raw_frame_novelty=True,
        ),
        action_effect_scorer=_EffectScorer({2: 0.0}),
    )
    raw_event = raw.record_transition(before, cosmetic, action)

    assert event is None
    assert gated.diagnostics()["controllability_gate_rejected"] == 1
    assert raw_event is not None
    assert raw_event.controllable is False
    assert raw.diagnostics()["observed_effects"] == 1


def test_req_arc_wmte_4688_knn_and_rnd_scores_decay_after_repeated_effect() -> None:
    """REQ-ARC-WMTE-4688: episodic kNN and RND-style novelty both learn from effects."""

    from carnot.agentic.arc_controllable_novelty import (
        ControllableNoveltyConfig,
        ControllableNoveltyProposalPolicy,
    )

    policy = ControllableNoveltyProposalPolicy(
        ControllableNoveltyConfig(enabled=True, controllability_gate=True),
        action_effect_scorer=_EffectScorer({6: 1.0}),
    )
    before = _frame([[0, 0], [0, 0]])
    after = _frame([[1, 0], [0, 0]])
    action = {"action": 6, "data": {"x": 0, "y": 0}}

    first_score = policy.score_candidate(before, action).total
    event = policy.record_transition(before, after, action)
    second_score = policy.score_candidate(before, action).total

    assert event is not None
    assert event.controllable is True
    assert first_score > second_score
    assert policy.diagnostics()["observed_effects"] == 1
    assert policy.diagnostics()["episodic_embeddings"] == 1
    assert policy.diagnostics()["rnd_updates"] == 1


def test_req_arc_wmte_4688_policy_helper_branches_are_deterministic() -> None:
    """REQ-ARC-WMTE-4688: helper branches keep stable proposal behavior."""

    from carnot.agentic import arc_controllable_novelty as nov
    from carnot.agentic.arc_controllable_novelty import (
        ControllableNoveltyConfig,
        ControllableNoveltyProposalPolicy,
        coerce_controllable_novelty_policy,
    )

    one_d = np.asarray([0, 1, 0, 2], dtype=np.int16)
    assert nov._as_grid(one_d).shape == (2, 2)
    assert nov._changed_fraction(_frame([[0]]), _frame([[0, 1]])) == 1.0

    candidate = SimpleNamespace(action_id=6, data={"x": 3, "y": 2})
    assert nov._action_id(candidate) == 6
    assert nov._action_data(candidate) == {"x": 3, "y": 2}
    assert nov._candidate_row(candidate) == {"action": 6, "data": {"x": 3, "y": 2}}
    assert nov._click_features(_frame([[0, 0], [0, 0]]), {"action": 2}) == [0.0, 0.0, 0.0]

    assert nov._l2([1.0, 2.0, 3.0], [1.0]) == 0.0
    assert len(nov._stable_unit_values("wide", 40)) == 40

    no_scorer = ControllableNoveltyProposalPolicy()
    assert no_scorer._effect_score(_frame([[0]]), {"action": 1}) is None
    mapped_policy = ControllableNoveltyProposalPolicy(
        {"enabled": True},
        action_effect_scorer=_EffectScorer({1: 1.0, 2: 0.5}),
    )
    assert mapped_policy.config.enabled is True
    structural_score = mapped_policy.score_embedding(
        [0.0, 1.0],
        action_effect_score=None,
        changed_fraction=1.0,
    )
    assert structural_score.controllable is True

    class _Boom:
        def candidate_score(self, _frame: Any, _candidate: Any) -> float:
            raise RuntimeError("boom")

    assert (
        ControllableNoveltyProposalPolicy(action_effect_scorer=_Boom())._effect_score(
            _frame([[0]]), {"action": 1}
        )
        is None
    )

    disabled = ControllableNoveltyProposalPolicy(
        ControllableNoveltyConfig(enabled=False),
        action_effect_scorer=_EffectScorer({1: 1.0}),
    )
    assert disabled.rank_candidates(_frame([[0]]), [{"action": 1, "data": None}]) == [
        {"action": 1, "data": None}
    ]
    ranked = mapped_policy.rank_candidates(
        _frame([[0, 0], [0, 0]]),
        [{"action": 1, "data": None}, {"action": 2, "data": None}],
    )
    assert ranked[0]["action"] == 1
    assert ranked[0]["controllable_novelty_components"]["temperature"] == 1.0

    limited = ControllableNoveltyProposalPolicy(
        ControllableNoveltyConfig(
            enabled=True,
            max_episodic_embeddings=1,
            controllability_gate=True,
        ),
        action_effect_scorer=_EffectScorer({6: 1.0}),
    )
    limited.record_transition(_frame([[0, 0]]), _frame([[1, 0]]), {"action": 6, "data": None})
    limited.record_transition(_frame([[0, 0]]), _frame([[0, 1]]), {"action": 6, "data": None})
    assert limited.diagnostics()["episodic_embeddings"] == 1

    assert coerce_controllable_novelty_policy(None) is None
    assert coerce_controllable_novelty_policy(False) is None
    assert coerce_controllable_novelty_policy("bad") is None
    assert coerce_controllable_novelty_policy({"enabled": False}) is None
    assert coerce_controllable_novelty_policy(True) is not None
    assert coerce_controllable_novelty_policy({"enabled": True}) is not None
    config_policy = coerce_controllable_novelty_policy(ControllableNoveltyConfig(enabled=True))
    assert config_policy is not None
    config_policy.action_effect_scorer = None
    assert (
        coerce_controllable_novelty_policy(config_policy, action_effect_scorer=_EffectScorer({1: 1.0}))
        is config_policy
    )
    disabled_policy = ControllableNoveltyProposalPolicy(ControllableNoveltyConfig(enabled=False))
    assert coerce_controllable_novelty_policy(disabled_policy) is None


def test_scenario_arc_wmte_4688_stepwise_orders_proposals_before_value_ranking() -> None:
    """SCENARIO-ARC-WMTE-4688: StepwiseExplorer reorders proposals before consumption."""

    if comp is None:
        pytest.skip("arc_competition_agent imports the absl/JAX stack under coverage")
    from carnot.agentic.arc_controllable_novelty import ControllableNoveltyConfig

    explorer = comp.StepwiseExplorer(
        online_discriminative=False,
        navigation_cost_tiebreak=False,
        frame_change_scorer=_EffectScorer({1: 0.05, 2: 0.95}),
        controllable_novelty=ControllableNoveltyConfig(
            enabled=True,
            bonus_weight=2.0,
            controllability_gate=True,
            temperature=0.5,
        ),
    )
    frame = _frame([[0, 0], [0, 0]])
    candidates = [{"action": 1, "data": None}, {"action": 2, "data": None}]

    ranked = explorer._apply_controllable_novelty_order(frame, candidates)

    assert [row["action"] for row in ranked] == [2, 1]
    assert ranked[0]["controllable_novelty_bonus"] > ranked[1]["controllable_novelty_bonus"]
    diagnostics = explorer.controllable_novelty_diagnostics()
    assert diagnostics["enabled"] is True
    assert diagnostics["candidate_scores"] >= 2
    assert diagnostics["verifier_is_oracle"] is False


def test_scenario_arc_wmte_4688_e3_accepts_opt_in_policy_without_changing_submitted_default() -> None:
    """SCENARIO-ARC-WMTE-4688: live E3 path can opt in while parity remains explicit."""

    if comp is None:
        pytest.skip("arc_competition_agent imports the absl/JAX stack under coverage")
    from carnot.agentic.arc_controllable_novelty import ControllableNoveltyConfig

    policy = comp.E3AgentPolicy(
        "bp35",
        proposer=None,
        value_head=lambda _frame, previous_frame=None: 0.0,
        frame_change_scorer=_EffectScorer({2: 1.0}),
        controllable_novelty=ControllableNoveltyConfig(enabled=True),
    )

    assert policy.explorer.controllable_novelty_policy is not None
    assert policy.explorer.controllable_novelty_diagnostics()["enabled"] is True
    assert comp.SUBMITTED_AGENT_CONFIG["controllable_novelty_proposal_enabled"] is False


def test_scenario_arc_wmte_4688_artifact_schema_and_attribution(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4688: artifact is checksumed and attribution-gated."""

    from carnot import experiment_4688_controllable_novelty_proposal_policy_live as mod

    artifact = mod.build_artifact(
        preconditions_checked={"ok": True, "qwen_proposer_port_verified": True},
        proposer_served_model="Qwen3.5-9B-Q4_K_M.gguf",
        live_path_reachable=True,
        parity_test_green=True,
        target_game="bp35",
        generic_first_win_by_config={
            "controllable_novelty_t0.5": {
                "first_win_count": 1,
                "first_win_rate": 1.0,
                "multi_level_count": 0,
                "multi_level_rate": 0.0,
                "max_reached_level": 1,
            }
        },
        novelty_result={
            "reached_level": 1,
            "generic_agent_reached_level": 1,
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "bare_control_passed": True,
            "temperature": 0.5,
        },
        no_novelty_result={"reached_level": 0},
        cosmetic_result={"reached_level": 0},
        duration_s=60.0,
    )
    path = tmp_path / mod.RESULT_RELATIVE_PATH
    mod.write_artifact(artifact, root=tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert artifact["honest_verdict"] == (
        "success: controllable_novelty_generic_agent_new_level_bp35_L1"
    )
    assert artifact["controllability_gate_on"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["no_novelty_ablation_reached_level"] == 0
    assert artifact["cosmetic_novelty_ablation_reached_level"] == 0
    assert artifact["chosen_submitted_config"]["controllable_novelty_proposal_enabled"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4688_null_schema_and_measurement_helpers(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4688: null artifacts and helper branches are auditable."""

    from carnot import experiment_4688_controllable_novelty_proposal_policy_live as mod

    attempts = [
        {"attempted": True, "reached_level": 0},
        {"attempted": True, "reached_level": 1},
        {"attempted": True, "reached_level": 2},
        {"attempted": False, "reached_level": 9},
    ]
    measurement = mod.measurement_from_attempts(attempts)
    assert measurement["variant_attempts_count"] == 3
    assert measurement["first_win_count"] == 2
    assert measurement["multi_level_count"] == 1
    assert measurement["multi_level_rate"] == 0.333333

    null_artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        proposer_served_model="Qwen3.5-9B-Q4_K_M.gguf",
        live_path_reachable=True,
        parity_test_green=True,
        target_game="bp35",
        generic_first_win_by_config={},
        novelty_result={"reached_level": 0, "bare_control_passed": True},
        no_novelty_result={"reached_level": 0},
        cosmetic_result={"reached_level": 0},
        duration_s=1.2345678,
    )
    assert null_artifact["honest_verdict"] == (
        "complete: controllable_novelty_no_new_level_residual_winning_prefix_still_not_proposed"
    )
    assert null_artifact["chosen_submitted_config"] == "unchanged"
    assert null_artifact["false_negative_risk_checked"] is True
    assert "null_methodology_note" in null_artifact
    assert mod.artifact_schema_errors(null_artifact) == []

    cosmetic_residual = mod.build_artifact(
        preconditions_checked={"ok": True},
        proposer_served_model="Qwen3.5-9B-Q4_K_M.gguf",
        live_path_reachable=True,
        parity_test_green=True,
        target_game="bp35",
        generic_first_win_by_config={},
        novelty_result={"reached_level": 1, "offline_reproduced": False},
        no_novelty_result={"reached_level": 0},
        cosmetic_result={"reached_level": 1},
        duration_s=1.0,
    )
    assert (
        cosmetic_residual["residual_cause_hypothesis"]
        == "controllability_embedding_rewards_non_winning_controllable_states"
    )

    revisit_residual = mod.build_artifact(
        preconditions_checked={"ok": True},
        proposer_served_model="Qwen3.5-9B-Q4_K_M.gguf",
        live_path_reachable=True,
        parity_test_green=True,
        target_game="bp35",
        generic_first_win_by_config={},
        novelty_result={"reached_level": 1, "offline_reproduced": False},
        no_novelty_result={"reached_level": 1},
        cosmetic_result={"reached_level": 0},
        duration_s=1.0,
    )
    assert revisit_residual["residual_cause_hypothesis"] == "novelty_revisits_dead_states"

    provisional_residual = mod.build_artifact(
        preconditions_checked={"ok": True},
        proposer_served_model="Qwen3.5-9B-Q4_K_M.gguf",
        live_path_reachable=True,
        parity_test_green=True,
        target_game="bp35",
        generic_first_win_by_config={},
        novelty_result={"reached_level": 1, "offline_reproduced": False},
        no_novelty_result={"reached_level": 0},
        cosmetic_result={"reached_level": 0},
        duration_s=1.0,
    )
    assert provisional_residual["residual_cause_hypothesis"] == (
        "winning_prefix_still_not_proposed"
    )
    replayed_but_not_green = mod.build_artifact(
        preconditions_checked={"ok": True},
        proposer_served_model="Qwen3.5-9B-Q4_K_M.gguf",
        live_path_reachable=False,
        parity_test_green=True,
        target_game="bp35",
        generic_first_win_by_config={},
        novelty_result={"reached_level": 1, "offline_reproduced": True},
        no_novelty_result={"reached_level": 0},
        cosmetic_result={"reached_level": 0},
        duration_s=1.0,
    )
    assert replayed_but_not_green["residual_cause_hypothesis"] == "novelty_revisits_dead_states"

    blocked = mod._blocked_artifact(
        {"ok": False},
        reason="blocked_model_not_cached_qwen",
        proposer_served_model="Qwen3.5-9B-MTP",
        duration_s=0.1,
    )
    assert blocked["honest_verdict"] == "blocked_model_not_cached_qwen"
    assert blocked["reproducibility_checksum"] == mod.payload_checksum(blocked)
    assert mod._floor_duration(0.0, minimum=0.0) >= 0.0
    started = mod.time.time()
    assert mod._floor_duration(started, minimum=0.001) >= 0.001

    bad = dict(null_artifact)
    bad["honest_verdict"] = "complete: bad"
    bad["verifier_is_oracle"] = True
    bad["solve_provenance"] = "development_proxy"
    bad["controllability_gate_on"] = False
    bad["residual_cause_hypothesis"] = "unknown"
    bad["proposer_served_model"] = "gemma"
    bad.pop("null_methodology_note", None)
    bad["reproducibility_checksum"] = "bad"
    errors = mod.artifact_schema_errors(bad)
    assert "verifier_is_oracle_false" in errors
    assert "solve_provenance" in errors
    assert "controllability_gate_on" in errors
    assert "residual_cause_hypothesis" in errors
    assert "proposer_served_model" in errors
    assert "null_methodology_note" in errors
    assert "reproducibility_checksum" in errors

    worse = dict(null_artifact)
    worse["honest_verdict"] = "bad"
    worse["reproducibility_checksum"] = "bad"
    assert "honest_verdict_terminal_prefix" in mod.artifact_schema_errors(worse)
