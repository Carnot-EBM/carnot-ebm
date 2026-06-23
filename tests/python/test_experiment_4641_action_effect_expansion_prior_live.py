"""Tests for Exp 4641 action-effect expansion prior.

Spec refs: REQ-ARC-FCP-4641, SCENARIO-ARC-FCP-4641.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot import experiment_4641_action_effect_expansion_prior_live as mod
from carnot.agentic import arc_competition_agent as comp
from carnot.agentic import arc_frame_change_predictor as fcp
from carnot.agentic.arc_agi3_live_adapter import ArcAction
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _frame(state: str, actions: tuple[int, ...] = (1,)) -> SimpleNamespace:
    values = {"root": 0, "bad": 1, "near": 2, "trap": 3, "win": 9}
    return SimpleNamespace(
        frame=np.array([[values[state]]], dtype=np.int16),
        levels_completed=1 if state == "win" else 0,
        available_actions=list(actions),
        state=state,
    )


class StateActionScore:
    def __init__(self, scores: dict[tuple[str, int], float]) -> None:
        self.scores = scores

    def candidate_score(self, frame: Any, candidate: Any) -> float:
        if isinstance(candidate, dict):
            action_id = int(candidate.get("action_id", candidate.get("action", 0)) or 0)
        else:
            action_id = int(getattr(candidate, "action_id", 0) or 0)
        return float(self.scores.get((str(getattr(frame, "state", "")), action_id), 0.0))


def test_req_arc_fcp_4641_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-FCP-4641: OpenSpec declares the expansion-prior live schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4641" in spec
    assert "SCENARIO-ARC-FCP-4641" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_fcp_4641_expansion_prior_scores_frontier_actions() -> None:
    """REQ-ARC-FCP-4641: branch priority comes from remaining action effects."""

    scorer = StateActionScore({("near", 1): 0.95, ("bad", 1): 0.05})
    prior = fcp.ActionEffectExpansionPrior(scorer)

    near_priority = prior.frontier_priority(_frame("near"), [ArcAction(1, None, "go")])
    bad_priority = prior.frontier_priority(_frame("bad"), [{"action": 1, "data": None}])

    assert prior.verifier_is_oracle is False
    assert near_priority < bad_priority
    assert prior.diagnostics()["scored_frontiers"] == 2


class _ExpansionToyEnv:
    def __init__(self) -> None:
        self.state = "root"

    def reset(self) -> Any:
        self.state = "root"
        return _frame("root", (1, 2))

    @staticmethod
    def _action_id(action: Any) -> int:
        if hasattr(action, "value"):
            return int(action.value)
        text = str(action)
        if "ACTION" in text:
            return int(text.rsplit("ACTION", 1)[-1])
        return int(action)

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> Any:
        aid = self._action_id(action)
        transitions = {
            ("root", 1): "bad",
            ("root", 2): "near",
            ("bad", 1): "trap",
            ("bad", 2): "trap",
            ("near", 1): "win",
            ("near", 2): "win",
            ("trap", 1): "trap",
            ("trap", 2): "trap",
        }
        self.state = transitions[(self.state, aid)]
        if self.state == "root":
            return _frame("root", (1, 2))
        if self.state in {"bad", "near", "trap"}:
            return _frame(self.state, (1, 2))
        return _frame("win", ())


def test_scenario_arc_fcp_4641_graph_explore_uses_action_effect_expansion_prior() -> None:
    """SCENARIO-ARC-FCP-4641: expansion priority changes which branch generates next."""

    scorer = StateActionScore({("near", 1): 0.95, ("near", 2): 0.95, ("bad", 1): 0.05})
    ranker_stats: dict[str, Any] = {}
    ranker_only, ranker_level = graph_explore_solve_v2(
        _ExpansionToyEnv(),
        0,
        max_expansions=3,
        max_depth=4,
        frame_change_scorer=scorer,
        action_effect_expansion_prior=False,
        stats=ranker_stats,
    )
    expansion_stats: dict[str, Any] = {}
    expansion, expansion_level = graph_explore_solve_v2(
        _ExpansionToyEnv(),
        0,
        max_expansions=3,
        max_depth=4,
        frame_change_scorer=scorer,
        action_effect_expansion_prior=True,
        stats=expansion_stats,
    )

    assert ranker_only is None
    assert ranker_level == 0
    assert expansion_level == 1
    assert expansion == [{"action": 2, "data": None}, {"action": 1, "data": None}]
    assert expansion_stats["action_effect_expansion_prior_enabled"] is True


def test_scenario_arc_fcp_4641_stepwise_frontier_and_e3_default_are_wired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-FCP-4641: the scored E3 path receives the expansion prior."""

    scorer = StateActionScore({("near", 1): 0.95, ("bad", 1): 0.05})
    explorer = comp.StepwiseExplorer(
        online_discriminative=False,
        navigation_cost_tiebreak=False,
        frame_change_scorer=scorer,
        action_effect_expansion_prior=True,
    )
    explorer.graph = {
        "bad": {
            "path": [{"action": 1, "data": None}],
            "untested": [{"action": 1, "data": None}],
            "value": 0.0,
            "frame": _frame("bad"),
        },
        "near": {
            "path": [{"action": 2, "data": None}],
            "untested": [{"action": 1, "data": None}],
            "value": 0.0,
            "frame": _frame("near"),
        },
    }

    assert explorer._frontier() == "near"
    assert explorer.action_effect_expansion_prior is not None

    monkeypatch.setattr(comp, "_load_submitted_frame_change_scorer", lambda: scorer)
    policy = comp.E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)

    assert comp.SUBMITTED_AGENT_CONFIG["action_effect_expansion_prior_enabled"] is True
    assert comp.SUBMITTED_AGENT_CONFIG["action_effect_expansion_prior_mode"] == (
        "persistent_aem_plus_optional_cnn_frontier_prior"
    )
    assert policy.explorer.action_effect_expansion_prior is not None


def test_scenario_arc_fcp_4641_artifact_success_and_honest_null(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4641: success needs deeper solve; nulls are explicit."""

    baseline = [
        mod.attempt("g1~color01", solved=True, depth=1, first_win=True, actions=3),
        mod.attempt("g2~color01", solved=False, depth=1, first_win=True, actions=None),
    ]
    expansion = [
        mod.attempt("g1~color01", solved=True, depth=2, first_win=True, actions=3),
        mod.attempt("g2~color01", solved=True, depth=2, first_win=True, actions=4),
    ]
    artifact = mod.build_artifact(
        root=tmp_path,
        preconditions_checked=mod.ok_preconditions_for_tests(),
        ranker_measurement=mod.measurement_from_attempts(baseline),
        expansion_measurement=mod.measurement_from_attempts(expansion),
        live_path_check={"passed": True},
        parity_test={"passed": True},
        duration_s=1.0,
        n_bootstrap=0,
    )

    assert artifact["honest_verdict"] == (
        "success: action_effect_expansion_prior_live_deeper_solve_1"
    )
    assert artifact["solve_rate_delta"] == pytest.approx(0.5)
    assert artifact["depth_of_live_solve_delta"] == pytest.approx(1.0)
    assert artifact["first_win_rate_delta"] == pytest.approx(0.0)
    assert artifact["chosen_submitted_config"]["action_effect_expansion_prior_enabled"] is True
    assert mod.validate_artifact(artifact) == []

    null_artifact = mod.build_artifact(
        root=tmp_path,
        preconditions_checked=mod.ok_preconditions_for_tests(),
        ranker_measurement=mod.measurement_from_attempts(baseline),
        expansion_measurement=mod.measurement_from_attempts(baseline),
        live_path_check={"passed": True},
        parity_test={"passed": True},
        duration_s=1.0,
        n_bootstrap=0,
    )

    assert null_artifact["honest_verdict"] == (
        "complete: action_effect_expansion_prior_no_deeper_solve_honest_null_gap_sharpened"
    )
    assert "honest no-value null" in null_artifact["null_delta_methodology_note"]
    assert null_artifact["chosen_submitted_config"] == "unchanged"
    assert mod.validate_artifact(null_artifact) == []
