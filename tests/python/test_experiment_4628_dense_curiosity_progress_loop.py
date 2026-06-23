"""Tests for Exp 4628 dense curiosity/learning-progress loop.

Spec refs: REQ-CAPSTONE-4628, SCENARIO-CAPSTONE-4628,
SCENARIO-CAPSTONE-4628-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _preconditions() -> dict[str, Any]:
    return {
        "ok": True,
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade": True,
        "e3_policy_import": True,
        "live_ttt_import": True,
        "arc_graph_explore_import": True,
        "dense_curiosity_import": True,
        "spec_has_req_4628": True,
        "arc_orphan_solver_lint_passed": True,
        "leaderboard_submission": False,
        "live_llm_inference": False,
        "qwen35_9b_mtp_igpu_precondition": "not_used",
    }


def _attempt(
    mode: str,
    signature: str,
    *,
    solved: bool,
    actions: int | None,
    state_coverage: int,
    reached_level: int = 1,
    reproduced: bool | None = None,
) -> dict[str, Any]:
    reproduced = bool(solved) if reproduced is None else bool(reproduced)
    return {
        "game": signature.split("~", 1)[0],
        "variant_signature": signature,
        "variant": 1,
        "kind": "color",
        "reflect": None,
        "attempted": True,
        "solved": bool(solved),
        "first_win": bool(solved),
        "reached_level": int(reached_level if solved else 0),
        "actions": actions if actions is not None else 200,
        "actions_to_first_levelup": actions if solved else None,
        "state_coverage": int(state_coverage),
        "distinct_win_relevant_states": int(state_coverage),
        "reachable_headroom": True,
        "reproduction_gate": {
            "game": signature.split("~", 1)[0],
            "claimed_level": int(reached_level if solved else 0),
            "reached_level": int(reached_level if solved and reproduced else 0),
            "reproduced": bool(reproduced),
        },
        "blocked_reason": "",
        "policy_mode": mode,
    }


def _runner_factory(rows_by_mode: Mapping[str, Mapping[str, dict[str, Any]]]):
    def _runner(mode: str):
        def run(game: str, spec: Mapping[str, Any], _budget: int) -> dict[str, Any]:
            signature = str(spec["variant_signature"])
            row = dict(rows_by_mode[mode][signature])
            row.setdefault("game", game)
            row.setdefault("variant_signature", signature)
            row.setdefault("attempted", True)
            return row

        return run

    return _runner


def test_req_capstone_4628_spec_declares_dense_loop_contract() -> None:
    """REQ-CAPSTONE-4628: OpenSpec declares the dense-progress loop contract."""

    from carnot import experiment_4628_dense_curiosity_progress_loop as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4628" in spec
    assert "SCENARIO-CAPSTONE-4628" in spec
    assert "SCENARIO-CAPSTONE-4628-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_capstone_4628_curiosity_rewards_reducible_prediction_error() -> None:
    """REQ-CAPSTONE-4628: bonus is prediction-error improvement, not raw surprise."""

    from carnot.agentic.arc_dense_curiosity_progress import DenseCuriosityProgress

    loop = DenseCuriosityProgress(game="unit", bonus_weight=1.0, backup_discount=0.5)
    grid = np.zeros((2, 2), dtype=np.int64)
    changed = grid.copy()
    changed[0, 0] = 1

    event = loop.observe_transition("root", "n1", grid, 6, {"x": 0, "y": 0}, changed)
    repeat = loop.observe_transition("n1", "n1_again", grid, 6, {"x": 0, "y": 0}, changed)

    assert event.before_error == 0.25
    assert event.after_error == 0.0
    assert event.raw_progress == 0.25
    assert event.aleatoric_estimate == 0.0
    assert event.bonus == 0.25
    assert repeat.before_error == 0.0
    assert repeat.bonus == 0.0
    assert loop.score_state("n1") == 0.25
    assert loop.score_state("root") == 0.375
    assert loop.diagnostics()["prediction_error_events"] == 2
    assert loop.diagnostics()["verifier_is_oracle"] is False


def test_req_capstone_4628_curiosity_suppresses_aleatoric_conflicts() -> None:
    """REQ-CAPSTONE-4628: repeated stochastic outcomes are aleatoric, not progress."""

    from carnot.agentic.arc_dense_curiosity_progress import DenseCuriosityProgress

    loop = DenseCuriosityProgress(game="unit", bonus_weight=1.0)
    grid = np.zeros((2, 2), dtype=np.int64)
    first = np.ones((2, 2), dtype=np.int64)
    second = np.full((2, 2), 2, dtype=np.int64)

    event1 = loop.observe_transition("root", "a", grid, 1, None, first)
    event2 = loop.observe_transition("root", "b", grid, 1, None, second)

    assert event1.bonus == 1.0
    assert event2.raw_progress == 1.0
    assert event2.aleatoric_estimate == 1.0
    assert event2.bonus == 0.0
    assert loop.diagnostics()["aleatoric_conflicts"] == 1


def test_req_capstone_4628_stepwise_explorer_uses_backup_in_frontier() -> None:
    """REQ-CAPSTONE-4628: StepwiseExplorer action selection reads backed-up value."""

    if "coverage" in sys.modules:
        pytest.skip("arc_competition_agent imports the absl/JAX stack under coverage")
    from carnot.agentic.arc_competition_agent import StepwiseExplorer

    explorer = StepwiseExplorer(
        dense_curiosity=True,
        dense_curiosity_weight=1.0,
        navigation_cost_tiebreak=False,
        candidate_router=None,
    )
    explorer.cur = "cold"
    explorer.graph = {
        "cold": {
            "path": [{"action": 1, "data": None}],
            "untested": [{"action": 1, "data": None}],
            "value": 0.0,
            "frame": None,
            "discriminative_features": None,
        },
        "hot": {
            "path": [{"action": 2, "data": None}],
            "untested": [{"action": 2, "data": None}],
            "value": 0.0,
            "frame": None,
            "discriminative_features": None,
        },
    }
    assert explorer.dense_curiosity is not None
    explorer.dense_curiosity.record_state_bonus("hot", 0.75)

    assert explorer._frontier() == "hot"
    diagnostics = explorer.curiosity_diagnostics()
    assert diagnostics["enabled"] is True
    assert diagnostics["state_values"] == 1


def test_scenario_capstone_4628_runner_writes_success_artifact(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4628: runner writes loop-vs-bare measurements."""

    from carnot import experiment_4628_dense_curiosity_progress_loop as mod

    rows_by_mode = {
        "loop": {
            "aa00~color01": _attempt("loop", "aa00~color01", solved=True, actions=7, state_coverage=5),
            "bb00~color01": _attempt("loop", "bb00~color01", solved=True, actions=9, state_coverage=6),
        },
        "bare": {
            "aa00~color01": _attempt("bare", "aa00~color01", solved=False, actions=None, state_coverage=2),
            "bb00~color01": _attempt("bare", "bb00~color01", solved=False, actions=None, state_coverage=3),
        },
    }

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=_preconditions(),
        public_games=("aa00", "bb00"),
        variant_ids=(1,),
        budget=50,
        variant_runner_factory=_runner_factory(rows_by_mode),
        live_path_check=lambda _root: {"passed": True, "command": "arc_orphan_solver_lint"},
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert artifact["honest_verdict"] == "success: dense_curiosity_loop_live_solverate_up_2"
    assert artifact["inference_substrate"].startswith("verifier_ensemble_against_cached_candidates")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["live_path_reachable"] is True
    assert artifact["live_solve_rate_loop"] == 1.0
    assert artifact["live_solve_rate_bare"] == 0.0
    assert artifact["solve_rate_delta"] == 1.0
    assert artifact["state_coverage_delta"] == 6
    assert artifact["first_win_rate_delta"] == 1.0
    assert artifact["live_lift_ci"]["metric"] == "solve_rate_delta"
    assert artifact["bare_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["offline_reproduced"] is True
    assert artifact["chosen_submitted_config"]["dense_curiosity_progress_loop_enabled"] is True
    assert artifact["submitted_to_leaderboard"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_capstone_4628_null_and_blocked_artifacts_are_auditable(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4628: honest nulls and missing resources fail closed."""

    from carnot import experiment_4628_dense_curiosity_progress_loop as mod

    assert mod._positive_int(True) is None
    assert mod._actions_to_first_levelup({"attempted": True, "solved": True}) is None
    assert mod._chosen_metric(0.0, 1, 0.0) == "state_coverage_delta"
    assert mod._chosen_metric(0.0, 0, 1.0) == "first_win_rate_delta"

    loop = mod.measurement_from_attempts(
        [_attempt("loop", "aa00~color01", solved=False, actions=None, state_coverage=3)]
    )
    bare = mod.measurement_from_attempts(
        [_attempt("bare", "aa00~color01", solved=False, actions=None, state_coverage=3)]
    )
    coverage_ci = mod.paired_metric_delta_ci(
        loop["variant_attempts"],
        bare["variant_attempts"],
        metric="state_coverage_delta",
        n_bootstrap=0,
    )
    assert coverage_ci["point"] == 0.0

    null_artifact = mod.build_artifact(
        preconditions_checked=_preconditions(),
        loop_measurement=loop,
        bare_measurement=bare,
        live_path_check={"passed": True},
        duration_s=1.0,
    )

    assert null_artifact["honest_verdict"] == (
        "complete: dense_curiosity_loop_no_live_lift_honest_null_gap_sharpened"
    )
    assert null_artifact["solve_rate_delta"] == 0.0
    assert null_artifact["state_coverage_delta"] == 0
    assert null_artifact["first_win_rate_delta"] == 0.0
    assert "null_delta_methodology_note" in null_artifact
    assert null_artifact["chosen_submitted_config"] == "unchanged"
    assert mod.artifact_schema_errors(null_artifact) == []

    unreproduced = mod.measurement_from_attempts(
        [
            _attempt(
                "loop",
                "yy00~color01",
                solved=True,
                actions=4,
                state_coverage=4,
                reproduced=False,
            )
        ]
    )
    assert mod._offline_reproduced(unreproduced, mod.measurement_from_attempts([])) is False

    broken = dict(null_artifact)
    broken["honest_verdict"] = "not_terminal"
    broken["verifier_is_oracle"] = True
    broken["live_path_reachable"] = False
    broken["reproducibility_checksum"] = "sha256:bad"
    broken.pop("null_delta_methodology_note")
    errors = mod.artifact_schema_errors(broken)
    assert "honest_verdict_terminal_prefix" in errors
    assert "verifier_is_oracle_false" in errors
    assert "live_path_reachable" in errors
    assert "reproducibility_checksum" in errors
    assert "null_delta_methodology_note" in errors

    blocked = mod.run(
        root=tmp_path,
        preconditions_checked={"ok": False, "blocked_resource": "offline_arcade"},
        live_path_check=lambda _root: {"passed": False},
        now=lambda: 1.0,
        sleep_fn=lambda _seconds: None,
    )
    assert blocked["honest_verdict"] == "blocked_offline_arcade"
    assert blocked["chosen_submitted_config"] == "unchanged"
