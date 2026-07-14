"""Tests for Exp5609 ARC filter intermediate-invariance A/B.

Spec refs: REQ-ARC-FCP-5609,
SCENARIO-ARC-FCP-5609-REACHABILITY-GATES-BLOCK-OUTCOME-TUNING,
SCENARIO-ARC-FCP-5609-MATCHED-BUDGET-ARM-ISOLATION,
SCENARIO-ARC-FCP-5609-DOWNSTREAM-PROMOTION-GATE.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_5609_arc_filter_intermediate_invariance_ab as mod
from carnot.agentic import arc_solve_artifact_discipline as discipline
from scripts import adversarial_verify


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_arc_fcp_5609_spec_declares_controlled_ab_contract() -> None:
    """REQ-ARC-FCP-5609: OpenSpec pins the reachability-controlled A/B contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5609") :]

    for marker in (
        "REQ-ARC-FCP-5609",
        "SCENARIO-ARC-FCP-5609-REACHABILITY-GATES-BLOCK-OUTCOME-TUNING",
        "SCENARIO-ARC-FCP-5609-MATCHED-BUDGET-ARM-ISOLATION",
        "SCENARIO-ARC-FCP-5609-DOWNSTREAM-PROMOTION-GATE",
        "offline_arcade_live_agent_runtime_filters_no_new_llm",
        "Candidate-count reduction alone SHALL NOT promote",
    ):
        assert marker in section


def test_req_arc_fcp_5609_substrate_is_linted_as_arc_no_llm_runtime() -> None:
    """REQ-ARC-FCP-5609: the filter-runtime substrate is a valid no-LLM ARC substrate."""

    artifact = {
        "experiment": "experiment_5609_arc_filter_intermediate_invariance_ab",
        "schema": "carnot.exp5609.arc_filter_intermediate_invariance_ab.v1",
        "honest_verdict": "complete: fixture",
        "duration_s": 0.02,
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
    }

    assert discipline.validate_arc_solve_artifact(artifact) == []
    floor = adversarial_verify.duration_floor_for_artifact(artifact)
    assert floor is not None
    assert floor["substrate"] == mod.INFERENCE_SUBSTRATE
    assert floor["reason"] == "arc_live_agent_filter_runtime_no_llm"


def test_scenario_5609_reachability_failure_blocks_outcome_runs(monkeypatch) -> None:
    """SCENARIO-ARC-FCP-5609-REACHABILITY-GATES-BLOCK-OUTCOME-TUNING: failed controls stop
    outcome A/B execution and produce interpretable blocked decisions."""

    monkeypatch.setattr(mod, "preconditions", lambda root=mod.REPO_ROOT: {"ok": True})
    monkeypatch.setattr(
        mod,
        "registry_precheck",
        lambda roster, root=mod.REPO_ROOT: {
            "ok": True,
            "duplicate_solve_targets_excluded": True,
            "selected_games": list(roster),
        },
    )
    monkeypatch.setattr(
        mod,
        "run_reachability_controls",
        lambda roster: {
            "inert_click": {"reachable": False, "reason": "fixture_no_signature"},
            "object_history": {"reachable": True, "same_base_ordering_changes": 1},
            "ok": False,
        },
    )

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("outcome arms must not run after failed controls")

    monkeypatch.setattr(mod, "run_all_arms", _fail_if_called)

    artifact = mod.build_artifact(roster=("dc22", "bp35", "s5i5"))

    assert artifact["honest_verdict"] == "complete: arc_filter_ab_mechanism_unreachable"
    assert artifact["candidate_counts_by_arm"] == {}
    assert (
        artifact["filter_promotion_decisions"]["inert_click"]["decision"] == "blocked_unreachable"
    )
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def test_scenario_5609_arm_configs_isolate_only_filter_toggles(monkeypatch) -> None:
    """SCENARIO-ARC-FCP-5609-MATCHED-BUDGET-ARM-ISOLATION: the four arms differ only by
    inert_click_pruner and object_history_salience."""

    monkeypatch.setattr(mod, "preconditions", lambda root=mod.REPO_ROOT: {"ok": True})
    monkeypatch.setattr(
        mod,
        "registry_precheck",
        lambda roster, root=mod.REPO_ROOT: {
            "ok": True,
            "duplicate_solve_targets_excluded": True,
            "selected_games": list(roster),
        },
    )
    monkeypatch.setattr(
        mod,
        "run_reachability_controls",
        lambda roster: {
            "inert_click": {"reachable": True, "pruned_signatures": 1},
            "object_history": {"reachable": True, "same_base_ordering_changes": 1},
            "ok": True,
        },
    )
    monkeypatch.setattr(mod, "run_all_arms", lambda *args, **kwargs: _fake_arm_results())

    artifact = mod.build_artifact(roster=("dc22", "bp35", "s5i5"), action_budget=12)
    configs = artifact["arm_configs"]
    receipt = artifact["matched_budget_receipt"]

    assert sorted(configs) == ["baseline", "combined", "history_only", "inert_only"]
    invariant_keys = (
        "roster",
        "random_seed",
        "action_budget",
        "target_levels",
        "proposer_available",
        "stopping_rule",
    )
    for key in invariant_keys:
        assert len({json.dumps(configs[arm][key], sort_keys=True) for arm in configs}) == 1
        assert receipt[key] == configs["baseline"][key]
    assert configs["baseline"]["inert_click_pruner"] is False
    assert configs["baseline"]["object_history_salience"] is False
    assert configs["inert_only"]["inert_click_pruner"] is True
    assert configs["history_only"]["object_history_salience"] is True
    assert configs["combined"]["inert_click_pruner"] is True
    assert configs["combined"]["object_history_salience"] is True


def test_scenario_5609_candidate_count_reduction_alone_retires_mechanism() -> None:
    """SCENARIO-ARC-FCP-5609-DOWNSTREAM-PROMOTION-GATE: candidate reduction alone is not
    enough when downstream live-path intermediates do not improve."""

    decisions = mod.decide_filter_promotions(
        controls={
            "inert_click": {"reachable": True},
            "object_history": {"reachable": True},
            "ok": True,
        },
        paired_effects={
            "inert_only_vs_baseline": {
                "candidate_count_delta_mean": -5.0,
                "environment_actions_delta_mean": 0.0,
                "distinct_states_delta_mean": 0.0,
                "nodes_expanded_delta_mean": 0.0,
                "levels_gained_delta_mean": 0.0,
                "safety_regression": False,
            },
            "history_only_vs_baseline": {
                "candidate_count_delta_mean": 0.0,
                "environment_actions_delta_mean": 0.0,
                "distinct_states_delta_mean": 0.0,
                "nodes_expanded_delta_mean": 0.0,
                "levels_gained_delta_mean": 0.0,
                "safety_regression": False,
            },
        },
    )

    assert decisions["inert_click"]["candidate_reduction_only"] is True
    assert decisions["inert_click"]["decision"] == "retire_reachable_downstream_noop"
    assert decisions["object_history"]["decision"] == "retire_reachable_downstream_noop"


def test_req_arc_fcp_5609_repository_artifact_has_required_schema() -> None:
    """REQ-ARC-FCP-5609: the checked-in artifact is the final decision-grade A/B result."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert artifact["experiment"] == mod.EXPERIMENT_ID
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["offline_reproduced"]["exact_known_level_safety"] is True
    assert artifact["registry_precheck"]["duplicate_solve_targets_excluded"] is True
    assert len(artifact["roster"]) >= 3
    assert sorted(artifact["arm_configs"]) == ["baseline", "combined", "history_only", "inert_only"]
    assert artifact["honest_verdict"].startswith("complete:")
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact


def _fake_arm_results() -> dict[str, object]:
    per_arm = {
        "baseline": [_row("dc22"), _row("bp35"), _row("s5i5")],
        "inert_only": [_row("dc22"), _row("bp35"), _row("s5i5")],
        "history_only": [_row("dc22"), _row("bp35"), _row("s5i5")],
        "combined": [_row("dc22"), _row("bp35"), _row("s5i5")],
    }
    return mod.summarize_arm_results(per_arm)


def _row(game: str) -> dict[str, object]:
    return {
        "game": game,
        "proposed_candidates": 10,
        "pruned_or_reranked_candidates": 0,
        "environment_actions": 12,
        "distinct_states": 4,
        "nodes_expanded": 4,
        "levels_gained": 0,
        "actions_to_level": None,
        "wall_time_s": 0.01,
        "offline_reproduced": True,
        "reproduced_level": 0,
        "safety_regression": False,
    }
