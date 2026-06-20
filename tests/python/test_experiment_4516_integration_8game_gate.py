"""Tests for Exp 4516 submitted integration gate.

Spec refs: REQ-ARC-FCP-4516, SCENARIO-ARC-FCP-4516.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from carnot import experiment_4516_integration_8game_gate as exp4516
from carnot.agentic.arc_competition_agent import StepwiseExplorer


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _preconditions() -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import": True,
        "baseline_file_present": True,
        "a1_a4_artifacts_present": True,
    }


def _baseline() -> dict[str, object]:
    return {
        "policy": "e3",
        "games": list(exp4516.GATE_GAMES),
        "solved_count": 4,
        "median_actions_on_solved": 7760.0,
        "per_game": [
            {"game": "lp85", "solved": True, "actions": 7792},
            {"game": "m0r0", "solved": True, "actions": 7789},
            {"game": "sp80", "solved": True, "actions": 7724},
            {"game": "vc33", "solved": True, "actions": 7731},
        ],
    }


def _summary(
    *,
    median: float,
    solved_games: list[str] | None = None,
    nav: dict[str, object] | None = None,
) -> dict[str, object]:
    solved = solved_games or list(exp4516.CORE_GAMES)
    rows = [
        {
            "game": game,
            "solved": game in solved,
            "actions": int(median),
            "actions_to_first_levelup": int(median) if game in solved else None,
            "reproduced": True if game in solved else None,
            "navigation_diagnostics": nav
            or {
                "navigation_attempts": 4,
                "forward_walk_hits": 1,
                "exact_shortest_path_hits": 0,
                "partial_forward_walk_hits": 1,
                "reset_replay_fallbacks": 3,
                "forward_walk_hit_rate": 0.25,
            },
        }
        for game in exp4516.GATE_GAMES
    ]
    return {
        "policy": "e3",
        "games": list(exp4516.GATE_GAMES),
        "per_game": rows,
        "solved_count": len(solved),
        "solved_games": sorted(solved),
        "actions_by_game": {game: int(median) for game in solved},
        "median_actions_on_solved": float(median),
        "median_actions_on_core": float(median),
        "heldout_solve_rate": len(solved) / len(exp4516.GATE_GAMES),
        "timed_out_count": 0,
        "navigation_diagnostics": {
            "navigation_attempts": 4,
            "forward_walk_hits": 1,
            "exact_shortest_path_hits": 0,
            "partial_forward_walk_hits": 1,
            "reset_replay_fallbacks": 3,
            "forward_walk_hit_rate": 0.25,
        },
    }


def test_req_arc_fcp_4516_spec_declares_integration_contract() -> None:
    """REQ-ARC-FCP-4516: OpenSpec anchors the integration artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4516" in spec
    assert "SCENARIO-ARC-FCP-4516" in spec
    assert exp4516.RESULT_RELATIVE_PATH in spec
    assert "flagged_adversarial: true" in spec
    assert "deepest known reachable ancestor" in spec
    for field, principle in exp4516.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_fcp_4516_selects_only_nonflagged_core_winners() -> None:
    """REQ-ARC-FCP-4516: A1-A4 integration obeys the CORE set-containment gate."""

    artifacts = {
        "A1_prune_predictor": {
            "flagged_adversarial": False,
            "core_solves_preserved": True,
            "median_actions_on_core": 7600.0,
        },
        "A2_imitation_prior": {
            "flagged_adversarial": False,
            "core_solves_preserved": False,
            "median_actions_on_core": 7000.0,
        },
        "A3_adaptive_budget": {
            "flagged_adversarial": True,
            "core_solves_preserved": True,
            "median_actions_on_core": 100.0,
        },
        "A4_lazy_best_first": {
            "flagged_adversarial": False,
            "chosen_submitted_value_weight": 0.0,
            "core_solves_preserved": True,
            "median_actions_on_core": 7760.0,
        },
    }

    decision = exp4516.select_integrated_levers(
        artifacts,
        control_median_actions_on_core=7760.0,
    )

    assert decision["accepted_a1_a4_levers"] == ["A1_prune_predictor"]
    assert decision["rejected_a1_a4_levers"]["A2_imitation_prior"]["reason"] == "core_gate_failed"
    assert decision["rejected_a1_a4_levers"]["A3_adaptive_budget"]["reason"] == "flagged_adversarial"
    assert decision["selected_value_weight"] == 0.0


def test_req_arc_fcp_4516_resource_and_loader_helpers(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4516: preconditions and upstream artifact loaders are explicit."""

    (tmp_path / "AGENTS.md").write_text("# ok\n", encoding="utf-8")
    (tmp_path / "OPENCODE.md").write_text("# ok\n", encoding="utf-8")
    (tmp_path / "ops").mkdir()
    (tmp_path / "ops" / "arc-submission-baseline.json").write_text(
        json.dumps(_baseline()),
        encoding="utf-8",
    )
    for relative in exp4516.UPSTREAM_ARTIFACTS.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"honest_verdict": "complete: fixture"}), encoding="utf-8")

    old_kit = exp4516._kit
    exp4516._kit = lambda: SimpleNamespace(offline_arcade=lambda: object())
    try:
        checks = exp4516.check_preconditions(tmp_path)
    finally:
        exp4516._kit = old_kit

    assert checks["offline_arcade_import"] is True
    assert checks["a1_a4_artifacts_present"] is True
    assert exp4516.load_gate_baseline(tmp_path)["median_actions_on_solved"] == 7760.0
    assert exp4516.load_gate_baseline(tmp_path / "missing")["median_actions_on_solved"] == 7760.0
    loaded = exp4516.load_a1_a4_artifacts(tmp_path)
    assert sorted(loaded) == sorted(exp4516.UPSTREAM_ARTIFACTS)


def test_req_arc_fcp_4516_upstream_summary_derives_core_gate_from_rows() -> None:
    """REQ-ARC-FCP-4516: artifacts without explicit core fields are derived conservatively."""

    artifact = {
        "local_gate_metrics": {
            "with_prior": {
                "per_game": [
                    {"game": "lp85", "solved": True, "actions": 70},
                    {"game": "m0r0", "solved": True, "actions": 80},
                    {"game": "sp80", "solved": True, "actions": 90},
                    {"game": "vc33", "solved": True, "actions_to_first_levelup": 100},
                ]
            }
        },
        "decision": {"selected_value_weight": 2.0},
    }

    decision = exp4516.select_integrated_levers(
        {"A4_lazy_best_first": artifact},
        control_median_actions_on_core=7760.0,
    )

    assert decision["accepted_a1_a4_levers"] == ["A4_lazy_best_first"]
    summary = decision["upstream_summaries"]["A4_lazy_best_first"]
    assert summary["core_solves_preserved"] is True
    assert summary["median_actions_on_core"] == 85.0
    assert decision["selected_value_weight"] == 2.0


def test_scenario_arc_fcp_4516_partial_forward_walk_replays_suffix_only() -> None:
    """SCENARIO-ARC-FCP-4516: partial forward navigation beats RESET replay."""

    explorer = StepwiseExplorer(online_discriminative=False)
    explorer.root = "R"
    explorer.cur = "A"
    explorer.start_level = 0
    explorer.best_level = 0
    explorer.graph = {
        "A": {"path": [{"action": 8, "data": None}], "untested": [], "value": 0.0},
        "B": {"path": [{"action": 1, "data": None}], "untested": [], "value": 0.0},
        "D": {
            "path": [
                {"action": 1, "data": None},
                {"action": 2, "data": None},
                {"action": 3, "data": None},
            ],
            "untested": [{"action": 4, "data": {"x": 1, "y": 1}}],
            "value": 0.0,
        },
    }
    explorer.adj = {"A": [({"action": 9, "data": None}, "B")]}

    assert explorer.next_move([], None) == (9, None)

    assert [item["kind"] for item in explorer.pending] == [2, 3, 4]
    assert explorer.pending[-1]["probe"] is True
    assert explorer.pending[-1]["origin"] == "D"
    diagnostics = explorer.navigation_diagnostics()
    assert diagnostics["navigation_attempts"] == 1
    assert diagnostics["partial_forward_walk_hits"] == 1
    assert diagnostics["reset_replay_fallbacks"] == 0
    assert diagnostics["forward_walk_hit_rate"] == 1.0
    assert explorer.graph["D"]["path"] == [
        {"action": 1, "data": None},
        {"action": 2, "data": None},
        {"action": 3, "data": None},
    ]


def test_scenario_arc_fcp_4516_artifact_schema_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4516: run writes the principle-annotated gate artifact."""

    def load_artifacts(_root: Path) -> dict[str, dict[str, object]]:
        return {
            "A1_prune_predictor": {
                "flagged_adversarial": False,
                "core_solves_preserved": False,
                "median_actions_on_core": 7766.0,
            },
            "A3_adaptive_budget": {
                "flagged_adversarial": True,
                "core_solves_preserved": True,
                "median_actions_on_core": 2984.0,
            },
            "A4_lazy_best_first": {
                "flagged_adversarial": False,
                "chosen_submitted_value_weight": 0.0,
                "core_solves_preserved": True,
                "median_actions_on_core": 7760.0,
            },
        }

    artifact = exp4516.run(
        root=tmp_path,
        write=True,
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        load_upstream_artifacts=load_artifacts,
        measure_submitted_gate=lambda **_kwargs: _summary(median=7500.0),
        random_seed=4516,
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"] == "success: integrated_median_actions_7500_below_7760"
    assert artifact["inference_substrate"] == exp4516.INFERENCE_SUBSTRATE
    assert artifact["median_actions_baseline"] == 7760.0
    assert artifact["median_actions_integrated"] == 7500.0
    assert artifact["levers_integrated"] == ["forward_navigation_partial_ancestor"]
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert exp4516.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / exp4516.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["median_actions_integrated"] == 7500.0


def test_req_arc_fcp_4516_gate_summary_and_nav_findings_cover_edge_cases() -> None:
    """REQ-ARC-FCP-4516: gate summaries and nav-loop findings are deterministic."""

    assert exp4516._json_action_label(6, {"x": 1, "y": 2}) == (
        '{"action": 6, "data": {"x": 1, "y": 2}}'
    )
    rows = [
        {
            "game": "lp85",
            "solved": True,
            "actions": 10,
            "actions_to_first_levelup": 5,
            "timed_out": False,
            "navigation_diagnostics": {
                "navigation_attempts": 2,
                "exact_shortest_path_hits": 2,
                "partial_forward_walk_hits": 0,
                "forward_walk_hits": 2,
                "reset_replay_fallbacks": 0,
                "forward_edges_recorded": 3,
                "forward_navigation_steps": 2,
                "reset_replay_steps": 0,
            },
        },
        {
            "game": "cd82",
            "solved": False,
            "actions": 11,
            "timed_out": True,
            "navigation_diagnostics": "not-a-dict",
        },
    ]

    summary = exp4516.summarize_gate_rows(rows, games=("lp85", "cd82"))

    assert summary["solved_count"] == 1
    assert summary["median_actions_on_solved"] == 10.0
    assert summary["median_actions_to_first_levelup"] == 5.0
    assert summary["timed_out_count"] == 1
    assert summary["navigation_diagnostics"]["forward_walk_hit_rate"] == 1.0
    assert exp4516._nav_loop_finding({"navigation_diagnostics": "bad"}) == (
        "navigation_diagnostics_missing"
    )
    assert exp4516._nav_loop_finding({"navigation_diagnostics": {"navigation_attempts": 0}}) == (
        "no_frontier_navigation_attempts_observed_on_gate"
    )
    assert "forward_walk_hit_rate_0" in exp4516._nav_loop_finding(
        {"navigation_diagnostics": {"navigation_attempts": 1, "forward_walk_hit_rate": 0.0}}
    )
    assert "exact_shortest_path_engaged_2_times" in exp4516._nav_loop_finding(summary)


def test_req_arc_fcp_4516_null_artifact_is_schema_valid() -> None:
    """REQ-ARC-FCP-4516: an honest null is valid with the fixed 7760 baseline."""

    decision = exp4516.select_integrated_levers(
        {
            "A2_imitation_prior": {
                "flagged_adversarial": False,
                "core_solves_preserved": False,
                "median_actions_on_core": 7733.0,
            }
        },
        control_median_actions_on_core=7760.0,
    )
    artifact = exp4516.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        integrated_measurement=_summary(median=7760.0),
        random_seed=4516,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: no_lever_beats_7760_honest_null"
    assert artifact["levers_integrated"] == []
    assert artifact["false_negative_risk_checked"] is True
    assert exp4516.artifact_schema_errors(artifact) == []


def test_req_arc_fcp_4516_schema_rejects_bad_artifacts() -> None:
    """REQ-ARC-FCP-4516: schema rejects unprincipled integration artifacts."""

    decision = exp4516.select_integrated_levers({}, control_median_actions_on_core=7760.0)
    artifact = exp4516.build_artifact(
        preconditions_checked=_preconditions(),
        baseline=_baseline(),
        upstream_decision=decision,
        integrated_measurement=_summary(median=7500.0),
        random_seed=4516,
        duration_s=1.0,
    )

    bad = {
        **artifact,
        "honest_verdict": "done",
        "inference_substrate": "live_llm",
        "field_principles": {},
        "median_actions_baseline": 1.0,
        "leaderboard_submission": True,
        "false_negative_risk_checked": False,
        "preconditions_checked": "bad",
        "reproducibility_checksum": "bad",
        "integrated_measurement": "bad",
        "upstream_decision": "bad",
    }
    del bad["random_seed"]

    errors = exp4516.artifact_schema_errors(bad)

    assert "missing required field random_seed" in errors
    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must match the required substrate" in errors
    assert "field_principles must match required field principles" in errors
    assert "median_actions_baseline must be the fixed 7760 control" in errors
    assert "leaderboard_submission must be false" in errors
    assert "false_negative_risk_checked must be true for complete/success artifacts" in errors
    assert "preconditions_checked must be a mapping" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors
    assert "integrated_measurement must be a mapping" in errors
    assert "upstream_decision must be a mapping" in errors

    duplicate = {
        **artifact,
        "integrated_measurement": {
            **artifact["integrated_measurement"],
            "navigation_diagnostics": "bad",
            "median_actions_on_solved": 1.0,
        },
        "upstream_decision": {
            "accepted_a1_a4_levers": ["A1"],
            "rejected_a1_a4_levers": {"A1": {"flagged_adversarial": False}},
        },
    }
    errors = exp4516.artifact_schema_errors(duplicate)
    assert "integrated_measurement must include navigation diagnostics" in errors
    assert "median_actions_integrated must mirror integrated measurement" in errors
    assert "accepted levers must not also appear in rejected levers" in errors


def test_req_arc_fcp_4516_run_blocked_and_schema_error_paths(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4516: run reports blocked resources and refuses bad artifacts."""

    blocked = exp4516.run(
        root=tmp_path,
        write=False,
        preconditions_checked={"offline_arcade_import": False},
        baseline=_baseline(),
        random_seed=4516,
        now=lambda: 10.0,
    )

    assert blocked["honest_verdict"] == "blocked_offline_arcade_import"
    assert blocked["false_negative_risk_checked"] is False

    try:
        exp4516.run(
            root=tmp_path,
            write=False,
            preconditions_checked=_preconditions(),
            baseline=_baseline(),
            load_upstream_artifacts=lambda _root: {},
            measure_submitted_gate=lambda **_kwargs: {
                "median_actions_on_solved": 1.0,
                "solved_count": 1,
                "solved_games": ["lp85"],
            },
            random_seed=4516,
            now=lambda: 10.0,
        )
    except ValueError as exc:
        assert "integrated_measurement must include navigation diagnostics" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("expected schema rejection")
