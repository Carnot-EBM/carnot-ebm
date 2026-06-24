"""Tests for Exp 4701 amortized first-contact prior plus Go-Explore live wiring.

Spec refs: REQ-ARC-WMTE-4701,
SCENARIO-ARC-WMTE-4701-LIVE-WIRING,
SCENARIO-ARC-WMTE-4701-COVERAGE-ABLATION.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"

if "coverage" in sys.modules or os.environ.get("CARNOT_SKIP_LIVE_IMPORT_UNDER_COVERAGE") == "1":
    comp = None
else:
    from carnot.agentic import arc_competition_agent as comp


def _frame(values: list[list[int]], *, level: int = 0) -> SimpleNamespace:
    return SimpleNamespace(frame=np.asarray(values, dtype=np.int16), levels_completed=level)


def test_req_arc_wmte_4701_spec_declares_amortized_archive_contract() -> None:
    """REQ-ARC-WMTE-4701: OpenSpec declares the live prior/archive artifact."""

    from carnot import experiment_4701_amortized_exploration_prior_go_explore_live as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4701" in spec
    assert "SCENARIO-ARC-WMTE-4701-LIVE-WIRING" in spec
    assert "SCENARIO-ARC-WMTE-4701-COVERAGE-ABLATION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4701_frequency_prior_ranks_reusable_first_contact_actions() -> None:
    """REQ-ARC-WMTE-4701: degraded prior distills reusable action families, not game IDs."""

    from carnot.agentic.arc_amortized_exploration import AmortizedFirstContactPrior

    traces = [
        {
            "game_id": "train_a",
            "outcome": "success",
            "steps": [{"action": 2, "data": None}, {"action": 6, "data": {"x": 1, "y": 1}}],
        },
        {
            "game_id": "train_b",
            "outcome": "near_miss",
            "steps": [{"action": 3, "data": None}],
        },
        {
            "game_id": "heldout_like_name_must_not_matter",
            "outcome": "success",
            "steps": [{"action": 2, "data": None}],
        },
    ]
    prior = AmortizedFirstContactPrior.from_traces(traces, max_depth=2)
    candidates = [
        {"action": 1, "data": None},
        {"action": 3, "data": None},
        {"action": 2, "data": None},
    ]

    ranked = prior.rank_candidates(_frame([[0, 0], [0, 0]]), candidates, path=[])

    assert [row["action"] for row in ranked[:3]] == [2, 3, 1]
    assert all("game_id" not in key for key in prior.diagnostics()["learned_family_keys"])
    assert prior.diagnostics()["distillation_mode"] == "frequency_prior"
    assert prior.diagnostics()["trace_count"] == 3
    assert ranked[0]["amortized_prior_score"] > ranked[1]["amortized_prior_score"] > 0.0


def test_scenario_arc_wmte_4701_go_explore_archive_selects_replayable_prefixes() -> None:
    """SCENARIO-ARC-WMTE-4701-LIVE-WIRING: archive returns under-visited replay prefixes."""

    from carnot.agentic.arc_go_explore import GoExploreReplayArchive

    archive = GoExploreReplayArchive(enabled=True, bins=2)
    root = _frame([[0, 0], [0, 0]], level=0)
    first = _frame([[1, 0], [0, 0]], level=0)
    deeper = _frame([[1, 2], [0, 0]], level=0)
    prefix_a = [{"action": 2, "data": None}]
    prefix_b = [{"action": 2, "data": None}, {"action": 6, "data": {"x": 1, "y": 0}}]

    archive.observe(root, [])
    archive.observe(first, prefix_a)
    archive.observe(deeper, prefix_b)

    selected = archive.select_prefix(current_path=[])

    assert selected == prefix_b
    assert archive.diagnostics()["stored_cells"] == 3
    assert archive.diagnostics()["selected_prefixes"] == 1
    assert archive.select_prefix(current_path=prefix_b) == prefix_a


def test_scenario_arc_wmte_4701_stepwise_orders_prior_and_exposes_archive() -> None:
    """SCENARIO-ARC-WMTE-4701-LIVE-WIRING: StepwiseExplorer consumes prior and archive hooks."""

    if comp is None:
        pytest.skip("arc_competition_agent imports the absl/JAX stack under coverage")

    from carnot.agentic.arc_amortized_exploration import AmortizedFirstContactPrior
    from carnot.agentic.arc_go_explore import GoExploreReplayArchive

    prior = AmortizedFirstContactPrior.from_traces(
        [{"outcome": "success", "steps": [{"action": 4, "data": None}]}]
    )
    archive = GoExploreReplayArchive(enabled=True, bins=2)
    explorer = comp.StepwiseExplorer(
        online_discriminative=False,
        navigation_cost_tiebreak=False,
        amortized_first_contact_prior=prior,
        go_explore_archive=archive,
    )

    ranked = explorer._apply_amortized_prior_order(
        _frame([[0, 0], [0, 0]]),
        [{"action": 1, "data": None}, {"action": 4, "data": None}],
        path=[],
    )
    archive.observe(_frame([[0, 0], [0, 0]]), [])
    archive.observe(_frame([[0, 9], [0, 0]]), [{"action": 4, "data": None}])

    replay = explorer._go_explore_replay_sequence(current_path=[])

    assert ranked[0]["action"] == 4
    assert ranked[0]["amortized_prior_score"] > 0.0
    assert replay == [{"action": 4, "data": None}]
    assert explorer.amortized_prior_diagnostics()["enabled"] is True
    assert explorer.go_explore_archive_diagnostics()["enabled"] is True


def test_req_arc_wmte_4701_arc_go_explore_is_solver_like_and_live_reachable() -> None:
    """REQ-ARC-WMTE-4701: orphan lint treats Go-Explore as live-path-reachable."""

    import scripts.arc_orphan_solver_lint as lint

    go_path = REPO / "python" / "carnot" / "agentic" / "arc_go_explore.py"
    closure = lint._closure(lint.ENTRYPOINTS) | {path.stem for path in lint.ENTRYPOINTS}

    assert lint._is_solver_like(go_path) == "defines solver function go_explore_solve()"
    assert "arc_go_explore" in closure


def test_scenario_arc_wmte_4701_artifact_records_coverage_null() -> None:
    """SCENARIO-ARC-WMTE-4701-COVERAGE-ABLATION: artifact records honest no-value nulls."""

    from carnot import experiment_4701_amortized_exploration_prior_go_explore_live as mod

    artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        proposer_served_model="Qwen3.5-9B-MTP",
        live_path_reachable=True,
        go_explore_now_live_reachable=True,
        parity_test_green=True,
        target_games=["r11l"],
        candidate_generation_coverage_with_prior=0.0,
        candidate_generation_coverage_no_prior_baseline=0.0,
        live_first_win_rate_with_prior=0.0,
        live_baseline_no_prior={"first_win_rate": 0.0, "attempts": 2},
        live_lift_ci={"low": 0.0, "high": 0.0, "confidence": 0.95},
        no_prior_ablation_failed=False,
        bare_control_passed=True,
        offline_reproduced=False,
        duration_s=60.0,
        target_arm_results={"with_prior": [], "no_prior": []},
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["coverage_delta"] == 0.0
    assert artifact["first_win_rate_delta"] == 0.0
    assert artifact["null_methodology_note"]
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert artifact["residual_bridge_gap"] == "archive_expands_dead_cells_no_goal_gradient"
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4701_artifact_records_attributable_success() -> None:
    """SCENARIO-ARC-WMTE-4701-COVERAGE-ABLATION: success needs prior lift and failed no-prior."""

    from carnot import experiment_4701_amortized_exploration_prior_go_explore_live as mod

    artifact = mod.build_artifact(
        preconditions_checked={"ok": True},
        proposer_served_model="Qwen3.5-9B-MTP",
        live_path_reachable=True,
        go_explore_now_live_reachable=True,
        parity_test_green=True,
        target_games=["r11l"],
        candidate_generation_coverage_with_prior=1.0,
        candidate_generation_coverage_no_prior_baseline=0.0,
        live_first_win_rate_with_prior=0.5,
        live_baseline_no_prior={"first_win_rate": 0.0, "attempts": 2},
        live_lift_ci={"low": 0.1, "high": 0.9, "confidence": 0.95},
        no_prior_ablation_failed=True,
        bare_control_passed=True,
        offline_reproduced=True,
        duration_s=60.0,
        target_arm_results={
            "with_prior": [{"reached_level": 1}],
            "no_prior": [{"reached_level": 0}],
        },
    )

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["coverage_delta"] == 1.0
    assert artifact["first_win_rate_delta"] == 0.5
    assert artifact["chosen_submitted_config"]["amortized_first_contact_prior_enabled"] is True
    assert artifact["chosen_submitted_config"]["go_explore_archive_enabled"] is True
    assert artifact["residual_bridge_gap"] == "none"
    assert mod.artifact_schema_errors(artifact) == []
