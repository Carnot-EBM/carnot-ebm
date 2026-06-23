"""Tests for Exp 4616 offline->live bridge disambiguation.

Spec refs: REQ-ARC-WMTE-4616, SCENARIO-ARC-WMTE-4616-BRIDGE-CAUSE,
SCENARIO-ARC-WMTE-4616-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_4616_offline_live_bridge_disambiguation as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_root(root: Path) -> None:
    _write_json(
        root / "results" / "experiment_value_q_head_v4.json",
        {
            "honest_verdict": "success: spatial_value_weight_tuned_strong_routing_speedup",
            "best_weight": {"won": True, "speedup": 7.63, "exp": 233},
            "blind_expansions": 1777,
            "verifier_is_oracle": False,
        },
    )
    _write_json(
        root / "results" / "arc_offline_to_live_bridge_v2.json",
        {
            "honest_verdict": (
                "complete_bridge_plumbed_v2_unlocks_cn04_offline_best_first_but_"
                "does_not_transfer_to_live_depth_first_explorer"
            ),
            "baseline_live_explorer_bfs": "8/32",
            "v2_value_head_41feat_spatial": {
                "offline_graph_explore_solve_v2_Astar_w5": "UNLOCKED cn04",
                "live_StepwiseExplorer_Astar_w5": "6/32 (REGRESS)",
            },
        },
    )
    _write_json(
        root / "results" / "experiment_4545_cross_game_discrimination_v3.json",
        {
            "loo_auroc_mean": 0.725,
            "loo_auroc_ci": [0.649, 0.806],
            "per_game_loo_auroc": {
                "aa00": 0.379,
                "bb00": 0.725,
                "cc00": 1.0,
            },
        },
    )


def test_req_arc_wmte_4616_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-4616: OpenSpec declares the diagnostic contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4616" in spec
    assert "SCENARIO-ARC-WMTE-4616-BRIDGE-CAUSE" in spec
    assert "SCENARIO-ARC-WMTE-4616-BLOCKED-PRECONDITION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4616_arm_metrics_are_separable() -> None:
    """REQ-ARC-WMTE-4616: each arm computes its binding criterion explicitly."""

    compute = mod.compute_cost_evidence(
        [
            {
                "condition": "equal_node_budget",
                "game": "ls20",
                "bare_first_win": False,
                "value_head_first_win": True,
                "bare_nodes": 1777,
                "value_head_nodes": 233,
            },
            {
                "condition": "equal_wall_clock",
                "game": "live_25_game_sim",
                "bare_solves": 8,
                "value_head_solves": 6,
                "bare_first_win": True,
                "value_head_first_win": False,
            },
        ]
    )
    shift = mod.distribution_shift_evidence(
        [
            {"split": "winning_path", "score": 0.95, "label": 1},
            {"split": "winning_path", "score": 0.85, "label": 1},
            {"split": "winning_path", "score": 0.25, "label": 0},
            {"split": "winning_path", "score": 0.15, "label": 0},
            {"split": "off_path_frontier", "score": 0.70, "label": 1},
            {"split": "off_path_frontier", "score": 0.55, "label": 1},
            {"split": "off_path_frontier", "score": 0.45, "label": 0},
            {"split": "off_path_frontier", "score": 0.25, "label": 0},
        ]
    )
    calibration = mod.calibration_evidence(
        [
            {"raw_score": 1.0, "true_steps_to_go": 1.0},
            {"raw_score": 2.0, "true_steps_to_go": 2.0},
            {"raw_score": 3.0, "true_steps_to_go": 3.0},
            {"raw_score": 4.0, "true_steps_to_go": 4.0},
        ],
        [
            {"node": "a", "depth": 1.0, "raw_score": 1.0},
            {"node": "b", "depth": 3.0, "raw_score": 3.0},
        ],
    )

    assert compute["binds"] is True
    assert compute["equal_node_budget"]["value_head_wins"] is True
    assert compute["equal_wall_clock"]["value_head_loses"] is True
    assert shift["binds"] is False
    assert shift["winning_path_auroc"] == 1.0
    assert shift["off_path_frontier_auroc"] == 1.0
    assert calibration["binds"] is False
    assert calibration["routing_changed_after_recalibration"] is False


def test_scenario_arc_wmte_4616_runner_writes_compute_cost_diagnosis(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4616-BRIDGE-CAUSE: runner isolates one binding cause."""

    _fixture_root(tmp_path)

    artifact = mod.run(
        root=tmp_path,
        precondition_checker=lambda: {
            "offline_arcade": {"available": True},
            "value_learner_imports": {"available": True},
        },
        diagnostic_corpus=mod.default_diagnostic_corpus(),
        started_s=1.0,
        now_s=2.25,
    )
    loaded = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert loaded == artifact
    assert artifact["honest_verdict"] == "success: bridge_cause_isolated_compute_fix_identified"
    assert artifact["binding_bridge_cause"] == "compute_cost"
    assert artifact["indicated_fix"] == mod.INDICATED_FIXES["compute_cost"]
    assert artifact["verifier_is_oracle"] is False
    assert artifact["offline_win_confirmed"] is True
    assert artifact["positive_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["compute_cost_evidence"]["binds"] is True
    assert artifact["distribution_shift_evidence"]["binds"] is False
    assert artifact["calibration_evidence"]["binds"] is False
    assert artifact["per_game_variance"]["min_loo_auroc"] == 0.379
    assert artifact["per_game_variance"]["max_loo_auroc"] == 1.0
    assert artifact["residual_bridge_gaps"]
    assert artifact["reproducibility_checksum"] == "sha256:" + mod.payload_checksum(artifact)
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_arc_wmte_4616_multi_cause_and_blocked_paths_are_auditable(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4616-BLOCKED-PRECONDITION: fail closed without evidence."""

    corpus = mod.default_diagnostic_corpus()
    corpus["distribution_shift_rows"] = [
        {"split": "winning_path", "score": 0.9, "label": 1},
        {"split": "winning_path", "score": 0.8, "label": 1},
        {"split": "winning_path", "score": 0.2, "label": 0},
        {"split": "winning_path", "score": 0.1, "label": 0},
        {"split": "off_path_frontier", "score": 0.5, "label": 1},
        {"split": "off_path_frontier", "score": 0.5, "label": 1},
        {"split": "off_path_frontier", "score": 0.5, "label": 0},
        {"split": "off_path_frontier", "score": 0.5, "label": 0},
    ]

    multi = mod.build_artifact(
        root=tmp_path,
        precondition_checker=lambda: {
            "offline_arcade": {"available": True},
            "value_learner_imports": {"available": True},
        },
        diagnostic_corpus=corpus,
        started_s=1.0,
        now_s=1.5,
    )

    assert multi["honest_verdict"] == (
        "complete: bridge_cause_inseparable_multi_cause_honest_residual_logged"
    )
    assert multi["binding_bridge_cause"] == "inseparable_multi_cause"
    assert "distribution_shift" in multi["residual_bridge_gaps"]
    assert mod.artifact_schema_errors(multi) == []

    blocked = mod.run(
        root=tmp_path,
        precondition_checker=lambda: {
            "offline_arcade": {"available": False, "error": "missing"},
            "value_learner_imports": {"available": True},
        },
        diagnostic_corpus=mod.default_diagnostic_corpus(),
        started_s=1.0,
        now_s=1.1,
    )

    assert blocked["honest_verdict"] == "blocked_offline_arcade"
    assert blocked["binding_bridge_cause"] == "not_evaluated"
    assert blocked["offline_win_confirmed"] is False
    assert blocked["positive_control_passed"] is False
    assert blocked["compute_cost_evidence"] == {}
    assert blocked["distribution_shift_evidence"] == {}
    assert blocked["calibration_evidence"] == {}
    assert blocked["reproducibility_checksum"] == "sha256:" + mod.payload_checksum(blocked)
    assert mod.artifact_schema_errors(blocked) == []


def test_req_arc_wmte_4616_helper_edges_and_schema_errors_are_explicit(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-4616: edge helpers and invalid artifacts report auditable states."""

    assert mod._duration(None, None) == 0.0
    assert (
        mod._first_precondition_blocker(
            {"offline_arcade": True, "value_learner_imports": False}
        )
        == "value_learner_imports"
    )
    assert mod.tie_aware_auroc([0.1, 0.2], [1.0, 1.0]) == 0.5
    assert mod._mean_abs_error([], []) == 0.0
    assert (
        mod.calibration_evidence(
            [{"raw_score": 1.0, "true_steps_to_go": 1.0}],
            [{"node": "single", "depth": 0.0, "raw_score": 1.0}],
        )["rank_cost_monotonicity"]
        == 0.0
    )
    assert mod._diagnose({}, {}, {}, positive_control_passed=False) == "not_evaluated"
    assert mod._verdict("not_evaluated") == "blocked_positive_control"
    assert mod._verdict("distribution_shift") == "success: bridge_cause_isolated_shift_fix_identified"
    assert mod._verdict("calibration") == "success: bridge_cause_isolated_calibration_fix_identified"
    assert mod._residual_bridge_gaps("not_evaluated", {}, {}, {}) == [
        "positive_control_missing_or_bare_control_not_run"
    ]

    routing_root = tmp_path / "routing_positive_control"
    _write_json(
        routing_root / "results" / "arc3_value_routing_v2.json",
        {"v2_unlocked_over_bfs": ["cn04"]},
    )
    artifact = mod.build_artifact(
        root=routing_root,
        precondition_checker=lambda: {
            "offline_arcade": True,
            "value_learner_imports": True,
        },
        diagnostic_corpus=mod.default_diagnostic_corpus(),
        started_s=1.0,
        now_s=1.0,
    )
    assert artifact["offline_win_confirmed"] is True

    missing = dict(artifact)
    missing.pop("binding_bridge_cause")
    assert "missing:binding_bridge_cause" in mod.artifact_schema_errors(missing)

    bad = dict(artifact)
    bad["honest_verdict"] = "not_terminal"
    bad["inference_substrate"] = "wrong"
    bad["verifier_is_oracle"] = True
    bad["field_principles"] = {}
    bad["reproducibility_checksum"] = "not-sha"
    errors = mod.artifact_schema_errors(bad)
    assert "honest_verdict_terminal_prefix" in errors
    assert "inference_substrate" in errors
    assert "verifier_is_oracle" in errors
    assert "field_principles" in errors
    assert "checksum_prefix" in errors

    bad_checksum = dict(artifact)
    bad_checksum["reproducibility_checksum"] = "sha256:" + ("0" * 64)
    assert "checksum_mismatch" in mod.artifact_schema_errors(bad_checksum)

    fabricated_blocked = dict(artifact)
    fabricated_blocked["honest_verdict"] = "blocked_positive_control"
    fabricated_blocked["compute_cost_evidence"] = {"fabricated": True}
    fabricated_blocked["positive_control_passed"] = False
    fabricated_blocked["reproducibility_checksum"] = (
        "sha256:" + mod.payload_checksum(fabricated_blocked)
    )
    assert "blocked_fabricated:compute_cost_evidence" in mod.artifact_schema_errors(
        fabricated_blocked
    )

    broken_positive_control = dict(artifact)
    broken_positive_control["positive_control_passed"] = False
    broken_positive_control["reproducibility_checksum"] = (
        "sha256:" + mod.payload_checksum(broken_positive_control)
    )
    assert "positive_control_required" in mod.artifact_schema_errors(
        broken_positive_control
    )
