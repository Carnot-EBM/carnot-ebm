"""Tests for Exp 4831 amortized in-context exploration prior.

Spec refs: REQ-ARC-WMTE-4831,
SCENARIO-ARC-WMTE-4831-IN-CONTEXT-PRIOR,
SCENARIO-ARC-WMTE-4831-HELDOUT-GATE.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _frame(values: list[list[int]], *, level: int = 0) -> SimpleNamespace:
    return SimpleNamespace(frame=np.asarray(values, dtype=np.int16), levels_completed=level)


def test_req_arc_wmte_4831_spec_declares_required_artifact_contract() -> None:
    """REQ-ARC-WMTE-4831: OpenSpec declares the 4831 artifact fields and principles."""

    from carnot import experiment_4831_amortized_incontext_exploration_prior_live as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4831" in spec
    assert "SCENARIO-ARC-WMTE-4831-IN-CONTEXT-PRIOR" in spec
    assert "SCENARIO-ARC-WMTE-4831-HELDOUT-GATE" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4831_prior_uses_prefix_context_not_game_ids() -> None:
    """SCENARIO-ARC-WMTE-4831-IN-CONTEXT-PRIOR: context changes the proposed action."""

    from carnot.agentic.arc_amortized_exploration import (
        AmortizedInContextExplorationPrior,
    )

    traces = [
        {
            "game_id": "train_a",
            "outcome": "success",
            "steps": [
                {"action": 2, "data": None},
                {"action": 6, "data": {"x": 1, "y": 1}},
            ],
        },
        {
            "game_id": "train_b",
            "outcome": "success",
            "steps": [
                {"action": 3, "data": None},
                {"action": 4, "data": None},
            ],
        },
    ]
    prior = AmortizedInContextExplorationPrior.from_traces(traces, max_context=2)
    candidates = [
        {"action": 4, "data": None},
        {"action": 6, "data": {"x": 7, "y": 8}},
    ]

    after_action_2 = prior.rank_candidates(
        _frame([[0, 0], [0, 0]]),
        candidates,
        path=[{"action": 2, "data": None}],
    )
    after_action_3 = prior.rank_candidates(
        _frame([[0, 0], [0, 0]]),
        candidates,
        path=[{"action": 3, "data": None}],
    )
    diagnostics = prior.diagnostics()

    assert after_action_2[0]["action"] == 6
    assert after_action_3[0]["action"] == 4
    assert diagnostics["distillation_mode"] == "in_context_exploration_prior"
    assert diagnostics["context_hits"] == 2
    assert diagnostics["proposal_changes"] == 1
    assert diagnostics["game_id_features_used"] is False
    assert all("train_" not in key for key in diagnostics["learned_context_keys"])


def test_req_arc_wmte_4831_coerce_mapping_selects_incontext_prior() -> None:
    """REQ-ARC-WMTE-4831: live prior config can select the in-context distillation mode."""

    from carnot.agentic.arc_amortized_exploration import (
        AmortizedInContextExplorationPrior,
        coerce_amortized_first_contact_prior,
    )

    prior = coerce_amortized_first_contact_prior(
        {
            "mode": "in_context_exploration_prior",
            "max_context": 1,
            "traces": [
                {
                    "outcome": "success",
                    "steps": [{"action": 1, "data": None}, {"action": 2, "data": None}],
                }
            ],
        }
    )

    assert isinstance(prior, AmortizedInContextExplorationPrior)
    assert prior.diagnostics()["trace_count"] == 1


def test_scenario_arc_wmte_4831_archive_alive_guard_blocks_dead_archive() -> None:
    """SCENARIO-ARC-WMTE-4831-HELDOUT-GATE: zero archive cells emit the dead-archive verdict."""

    from carnot import experiment_4831_amortized_incontext_exploration_prior_live as mod

    artifact = mod.build_artifact(
        preconditions_checked={"offline_arcade": True, "go_explore_import": True},
        go_explore_archive_alive={
            "observations": 0,
            "stored_cells": 0,
            "prefixes_injected": 0,
        },
        prior_changed_proposals=True,
        first_win_rate_with_prior=0.08,
        first_win_rate_no_prior_ablation=0.04,
        first_win_delta_ci95={"low": 0.01, "high": 0.09, "confidence": 0.95},
        imitation_control_heldout_games={"lift_holds": True},
        live_path_reachable=True,
        duration_s=60.0,
    )

    assert artifact["honest_verdict"] == "blocked_dead_go_explore_archive"
    assert artifact["go_explore_archive_alive"]["alive"] is False
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4831_artifact_distinguishes_success_and_genuine_null() -> None:
    """SCENARIO-ARC-WMTE-4831-HELDOUT-GATE: success needs CI lift and imitation control."""

    from carnot import experiment_4831_amortized_incontext_exploration_prior_live as mod

    alive = {"observations": 2, "stored_cells": 2, "prefixes_injected": 1}
    null_artifact = mod.build_artifact(
        preconditions_checked={"offline_arcade": True, "go_explore_import": True},
        go_explore_archive_alive=alive,
        prior_changed_proposals=True,
        first_win_rate_with_prior=0.0,
        first_win_rate_no_prior_ablation=0.0,
        first_win_delta_ci95={"low": 0.0, "high": 0.0, "confidence": 0.95},
        imitation_control_heldout_games={
            "heldout_not_in_distillation_set": True,
            "lift_holds": False,
        },
        live_path_reachable=True,
        duration_s=60.0,
    )
    success_artifact = mod.build_artifact(
        preconditions_checked={"offline_arcade": True, "go_explore_import": True},
        go_explore_archive_alive=alive,
        prior_changed_proposals=True,
        first_win_rate_with_prior=0.08,
        first_win_rate_no_prior_ablation=0.04,
        first_win_delta_ci95={"low": 0.01, "high": 0.09, "confidence": 0.95},
        imitation_control_heldout_games={
            "heldout_not_in_distillation_set": True,
            "lift_holds": True,
        },
        live_path_reachable=True,
        duration_s=60.0,
    )

    assert (
        null_artifact["honest_verdict"]
        == "complete_amortized_prior_no_first_win_lift_l1_wall_survives"
    )
    assert null_artifact["go_explore_archive_alive"]["alive"] is True
    assert null_artifact["prior_changed_proposals"] is True
    assert success_artifact["honest_verdict"] == (
        "success_amortized_prior_raises_first_win_above_baseline"
    )
    assert mod.artifact_schema_errors(null_artifact) == []
    assert mod.artifact_schema_errors(success_artifact) == []


def test_req_arc_wmte_4831_helpers_cover_ci_probe_and_schema_edges(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4831: helper gates are deterministic and schema failures are explicit."""

    from carnot import experiment_4831_amortized_incontext_exploration_prior_live as mod
    from carnot.agentic.arc_amortized_exploration import AmortizedInContextExplorationPrior

    assert mod._rate([True, False, True]) == 0.666667
    assert mod._rate([]) == 0.0
    assert mod._ci_positive({"low": "bad"}) is False
    assert mod._bootstrap_delta_ci([], [], n_boot=3)["n_boot"] == 3
    ci = mod._bootstrap_delta_ci([True, True], [False, False], seed=1, n_boot=5)
    assert ci["low"] == 1.0 and ci["high"] == 1.0
    assert mod._action_for_family("click") == {"action": 6, "data": {"x": 0, "y": 0}}
    assert mod._action_for_family("unknown") == {"action": 1, "data": None}

    prior = AmortizedInContextExplorationPrior.from_traces(
        [{"outcome": "success", "steps": [{"action": 5, "data": None}]}]
    )
    changed = mod.proposal_change_probe(prior)
    assert changed["changed"] is True

    class EmptyPrior:
        context_family_scores: dict = {}

        def rank_candidates(self, _frame, candidates, *, path=None):
            return [dict(row) for row in candidates]

    unchanged = mod.proposal_change_probe(EmptyPrior())
    assert unchanged["changed"] is False

    artifact = mod.build_artifact(
        preconditions_checked={"offline_arcade": True, "go_explore_import": True},
        go_explore_archive_alive={"observations": 1, "stored_cells": 1, "prefixes_injected": 1},
        prior_changed_proposals=True,
        first_win_rate_with_prior=0.08,
        first_win_rate_no_prior_ablation=0.04,
        first_win_delta_ci95={"low": 0.01, "high": 0.09, "confidence": 0.95},
        imitation_control_heldout_games={
            "heldout_not_in_distillation_set": True,
            "lift_holds": True,
        },
        live_path_reachable=True,
        duration_s=60.0,
        prior_diagnostics={"enabled": True},
        prior_change_diagnostics=changed,
        measurement={"with_prior": [], "no_prior": []},
    )
    written = mod.write_artifact(artifact, root=tmp_path)
    assert written.read_text(encoding="utf-8").startswith("{\n")

    bad = dict(artifact)
    bad["honest_verdict"] = "nonsense"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "honest_verdict_terminal_prefix" in mod.artifact_schema_errors(bad)

    bad = dict(artifact)
    bad["go_explore_archive_alive"] = "alive"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "go_explore_archive_alive" in mod.artifact_schema_errors(bad)

    bad = dict(artifact)
    bad["honest_verdict"] = mod.DEAD_ARCHIVE_VERDICT
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "dead_archive_verdict_requires_dead_archive" in mod.artifact_schema_errors(bad)

    bad = dict(artifact)
    bad["solve_provenance"] = "development_proxy"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "solve_provenance" in mod.artifact_schema_errors(bad)

    bad = dict(artifact)
    bad["inference_substrate"] = "aggregation"
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "inference_substrate" in mod.artifact_schema_errors(bad)

    bad = dict(artifact)
    bad["prior_changed_proposals"] = False
    bad["first_win_rate_with_prior"] = 0.01
    bad["first_win_delta_ci95"] = {"low": 0.0, "high": 0.01}
    bad["imitation_control_heldout_games"] = {"lift_holds": False}
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    errors = mod.artifact_schema_errors(bad)
    assert "success_requires_prior_changed_proposals" in errors
    assert "success_requires_above_baseline" in errors
    assert "success_requires_positive_ci" in errors
    assert "success_requires_imitation_lift" in errors

    bad = dict(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum" in mod.artifact_schema_errors(bad)
