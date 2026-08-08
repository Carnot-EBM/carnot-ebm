"""Tests for Exp 4821 S3 structural-energy generation lift.

Spec refs: REQ-ARC-WMTE-4821,
SCENARIO-ARC-WMTE-4821-GENERATION-LIFT,
SCENARIO-ARC-WMTE-4821-LIVE-PLAN-WIRING.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from carnot import experiment_4821_structural_energy_s3_generation_lift as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _attempt(
    game: str,
    *,
    first_win: bool = False,
    reproduced: bool = False,
    reached_level: int = 0,
    reachable_headroom: bool = True,
) -> dict[str, Any]:
    return {
        "game": game,
        "attempted": True,
        "first_win": bool(first_win),
        "solved": bool(first_win),
        "reached_level": int(reached_level),
        "reachable_headroom": bool(reachable_headroom),
        "solution_labels": [{"action": 1, "data": None}] if first_win else [],
        "reproduction_gate": {
            "game": game,
            "reproduced": bool(reproduced),
            "reached_level": int(reached_level),
        },
    }


def _manual_game_result(game: str, *, e_banked: bool, already_bare: bool = False) -> dict[str, Any]:
    return {
        "game": game,
        "winner_rank": {"rank": 1, "candidate_count": 8, "source": "unit_fixture"},
        "winner-rank": {"rank": 1, "candidate_count": 8, "source": "unit_fixture"},
        "banked_by_E": bool(e_banked),
        "banked-by-E": bool(e_banked),
        "banked_by_bare": False,
        "banked-by-bare": False,
        "was_already_in_bare_pool": bool(already_bare),
        "was-already-in-bare-pool": bool(already_bare),
        "positive_control_reachable": True,
        "winner_newly_entered_pool": bool(e_banked and not already_bare),
        "bare_reached_level": 0,
        "e_guided_reached_level": 1 if e_banked else 0,
        "bare_offline_reproduced": False,
        "e_guided_offline_reproduced": bool(e_banked),
        "lambda0_attempts": 1,
        "e_guided_attempts": 1,
    }


def test_req_arc_wmte_4821_spec_declares_s3_contract() -> None:
    """REQ-ARC-WMTE-4821: OpenSpec declares the S3 artifact and principles."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-ARC-WMTE-4821")
    end = spec.index("### REQ-ARC-WMTE-4781", start)
    section = _normalise(spec[start:end])

    for ref in mod.SPEC_REFS:
        assert ref in section
    assert mod.RESULT_RELATIVE_PATH in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in section
        assert _normalise(principle["principle"]) in section


@pytest.mark.memory_watchdog_skip
def test_scenario_arc_wmte_4821_live_e3_passes_goal_energy_to_plan(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-4821-LIVE-PLAN-WIRING: lambda controls planner guidance."""

    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    fake_ttt = ModuleType("carnot.agentic.arc_live_ttt")
    fake_ttt.gated_engine_from_transitions = lambda *_args, **_kwargs: (None, None, {})
    monkeypatch.setitem(sys.modules, "carnot.agentic.arc_live_ttt", fake_ttt)

    class _FakeProposer:
        def induce(self, *_args: Any, **_kwargs: Any) -> tuple[bool, str]:
            return True, "ok"

        def world_model_candidates(self, _game: str) -> list[Any]:
            return []

    class _FakeVerifier:
        def score(self, _engine: Any) -> SimpleNamespace:
            # hud_mask_status/cells/swallow: added to the real VerifyResult dataclass by commit
            # d9de98aedd (2026-07-27), which also added the hud_mask= kwarg on the WorldModelVerifier
            # call below and the change_gate_decision() call these fields feed -- read directly at
            # the call site before change_gate_decision runs, so omitting them here raises an
            # AttributeError that gets silently swallowed by _induce_and_plan's outer except, and
            # plan_in_model is never reached.
            return SimpleNamespace(
                accuracy=1.0,
                cell_recall=1.0,
                hud_mask_status="not_requested",
                hud_mask_cells=0,
                hud_mask_swallow={},
            )

    def engine(grid: np.ndarray, _action: int, _data: Any) -> np.ndarray:
        return np.asarray(grid)

    def is_done(grid: np.ndarray) -> bool:
        return not np.asarray(grid).any()

    monkeypatch.setattr(e3, "load_engine", lambda _game: (engine, is_done))
    # **_kwargs: the plain path now calls WorldModelVerifier(active_transitions, hud_mask=_hud_mask)
    # (same commit as above); the old single-positional-arg fake raised a TypeError that was
    # silently swallowed the same way.
    monkeypatch.setattr(e3, "WorldModelVerifier", lambda _transitions, **_kwargs: _FakeVerifier())
    # change_gate_decision() (added by the same commit) reads ~15 VerifyResult fields the bare fake
    # above doesn't have. The change-gate feature itself is default-off in production
    # (SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED = False), so this canned dict reflects its true
    # default-off runtime shape rather than hand-faking the full VerifyResult surface.
    monkeypatch.setattr(
        e3,
        "change_gate_decision",
        lambda _vr, **_kwargs: {
            "gate_enabled": False,
            "passed": True,
            "reason": "gate_disabled",
            "change_fidelity": 1.0,
            "correct_changed_cells": 1,
            "spurious_changed_cells": 0,
            "change_accuracy": 1.0,
            "legacy_accuracy_would_pass_at_live_threshold": True,
            "noop_ok_is_vacuous": False,
        },
    )

    captured: list[dict[str, Any]] = []

    def plan_in_model(
        _engine: Any, _is_done: Any, _root_grid: np.ndarray, **kwargs: Any
    ) -> list[dict[str, Any]]:
        captured.append(dict(kwargs))
        return [{"action": 1, "data": None}]

    monkeypatch.setattr(e3, "plan_in_model", plan_in_model)

    guided = E3AgentPolicy(
        "zz99",
        proposer=_FakeProposer(),
        value_head=lambda _frame: 0.0,
        goal_bias=lambda _grid: 0.0,
        goal_candidate_guidance=False,
        goal_guidance_lambda=2.5,
    )
    guided.transitions = [object()]
    guided.root_grid = np.ones((2, 2), dtype=np.int16)
    guided._induce_and_plan()

    assert callable(captured[-1]["goal_energy"])
    assert captured[-1]["goal_energy"](np.ones((2, 2), dtype=np.int16)) == pytest.approx(2.5)
    assert captured[-1]["goal_energy"](np.zeros((2, 2), dtype=np.int16)) == pytest.approx(0.0)

    bare = E3AgentPolicy(
        "zz99",
        proposer=_FakeProposer(),
        value_head=lambda _frame: 0.0,
        goal_bias=lambda _grid: 0.0,
        goal_candidate_guidance=False,
        goal_guidance_lambda=0.0,
    )
    bare.transitions = [object()]
    bare.root_grid = np.ones((2, 2), dtype=np.int16)
    bare._induce_and_plan()

    # Not exact-{} equality: commit c48b6a853d (2026-07-15, REQ-ARC-FCP-5699-15) made the plain
    # path unconditionally pass a diagnostics= kwarg into plan_in_model, so captured[-1] now always
    # carries a diagnostics dict regardless of goal_guidance_lambda. What this assertion actually
    # cares about -- lambda=0 means no goal energy is supplied -- is still exactly captured below.
    assert "goal_energy" not in captured[-1]


def test_scenario_arc_wmte_4821_builds_success_artifact_when_ci_excludes_zero() -> None:
    """SCENARIO-ARC-WMTE-4821-GENERATION-LIFT: new E-only winners authorize S4."""

    rows = [_manual_game_result(f"g{i}", e_banked=True) for i in range(5)]
    artifact = mod.build_artifact(
        rows,
        preconditions_checked={"offline_arcade": True, "e3_agent_policy_import": True},
        live_path_reachable=True,
        bootstrap_resamples=100,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.SUCCESS_VERDICT
    assert artifact["n_headroom_games"] == 5
    assert artifact["winners_newly_entering_pool_delta"] == 1.0
    assert artifact["winners_newly_entering_pool_delta_ci95"] == [1.0, 1.0]
    assert len(artifact["new_levels_not_in_bare_pool"]) == 5


def test_req_arc_wmte_4821_bounded_and_inconclusive_generation_paths() -> None:
    """REQ-ARC-WMTE-4821: nulls and insufficient headroom are distinct verdicts."""

    games = [f"g{i}" for i in range(6)]
    rows = mod.generation_lift_rows(
        bare_attempts=[_attempt(game) for game in games],
        guided_attempts=[_attempt(game) for game in games],
        positive_control_attempts=[_attempt(game, reachable_headroom=True) for game in games],
    )
    bounded = mod.build_artifact(
        rows,
        preconditions_checked={"offline_arcade": True, "e3_agent_policy_import": True},
        live_path_reachable=True,
        bootstrap_resamples=50,
    )
    mod.validate_artifact(bounded)
    assert bounded["honest_verdict"] == mod.BOUNDED_VERDICT
    assert bounded["positive_control_passed"] is True
    assert bounded["new_levels_not_in_bare_pool"] == []
    assert bounded["winners_newly_entering_pool_delta_ci95"] == [0.0, 0.0]

    inconclusive = mod.build_artifact(
        rows[:4],
        preconditions_checked={"offline_arcade": True, "e3_agent_policy_import": True},
        live_path_reachable=True,
        bootstrap_resamples=50,
    )
    mod.validate_artifact(inconclusive)
    assert inconclusive["honest_verdict"] == mod.INCONCLUSIVE_VERDICT
    assert inconclusive["n_headroom_games"] == 4
    assert inconclusive["winners_newly_entering_pool_delta_ci95"] is None


def test_req_arc_wmte_4821_validation_rejects_fabricated_generation_win() -> None:
    """REQ-ARC-WMTE-4821: success needs E-only offline-reproduced levels."""

    rows = [_manual_game_result(f"g{i}", e_banked=True, already_bare=True) for i in range(5)]
    artifact = mod.build_artifact(
        rows,
        preconditions_checked={"offline_arcade": True, "e3_agent_policy_import": True},
        live_path_reachable=True,
        bootstrap_resamples=50,
    )
    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == mod.BOUNDED_VERDICT

    fabricated = copy.deepcopy(artifact)
    fabricated["honest_verdict"] = mod.SUCCESS_VERDICT
    fabricated["reproducibility_checksum"] = mod.payload_checksum(fabricated)
    with pytest.raises(ValueError, match="success requires"):
        mod.validate_artifact(fabricated)


def test_req_arc_wmte_4821_committed_artifact_satisfies_schema() -> None:
    """REQ-ARC-WMTE-4821: committed S3 deliverable is schema-valid."""

    artifact = json.loads((REPO / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["result_path"] == mod.RESULT_RELATIVE_PATH
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == mod.SOLVE_PROVENANCE
    assert artifact["n_headroom_games"] >= mod.MIN_HEADROOM_GAMES
