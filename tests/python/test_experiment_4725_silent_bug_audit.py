"""Tests for Exp 4725 silent representation no-op audit.

Spec refs: REQ-ARC-WMTE-4725, SCENARIO-ARC-WMTE-4725-AUDIT-CLASSIFICATION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def test_req_arc_wmte_4725_spec_declares_audit_contract() -> None:
    """REQ-ARC-WMTE-4725: OpenSpec declares the audit fields and scenario."""

    from carnot import experiment_4725_silent_bug_audit as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4725" in spec
    assert "SCENARIO-ARC-WMTE-4725-AUDIT-CLASSIFICATION" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert mod.REPORT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4725_classifies_dead_archive_and_a4_tautology() -> None:
    """SCENARIO-ARC-WMTE-4725-AUDIT-CLASSIFICATION: dead evidence must reopen."""

    from carnot import experiment_4725_silent_bug_audit as mod

    dead_archive = {
        "honest_verdict": "complete: amortized_prior_go_explore_no_coverage_gain_residual_logged",
        "candidate_generation_coverage_with_prior": 0.0,
        "candidate_generation_coverage_no_prior_baseline": 0.0,
        "coverage_delta": 0.0,
        "target_arm_results": {
            "coverage": {
                "with_prior": {"coverage": 0.0, "total_steps": 0},
                "no_prior": {"coverage": 0.0, "total_steps": 0},
            },
            "with_prior": [
                {
                    "go_explore_archive_diagnostics": {
                        "enabled": True,
                        "observations": 0,
                        "stored_cells": 0,
                        "actions_injected": 0,
                        "prefixes_injected": 0,
                    },
                    "amortized_prior_diagnostics": {"enabled": True, "rank_calls": 8},
                }
            ],
        },
    }
    a4_tautology = {
        "honest_verdict": "complete: online_action_learning_no_first_win_lift_residual_x",
        "flagged_adversarial": True,
        "frozen_first_win": 0.04,
        "online_scratch_first_win": 0.04,
        "online_warm_first_win": 0.04,
        "online_warm_vs_frozen_delta": 0.0,
    }

    archive_verdict = mod.classify_null(
        "experiment_4701_amortized_exploration_prior_go_explore_live",
        dead_archive,
    )
    a4_verdict = mod.classify_null(
        "experiment_4715_online_action_learning_driver_corrected",
        a4_tautology,
    )

    assert archive_verdict["verdict"] == "silent_bug_must_reopen"
    assert "dead_go_explore_archive" in archive_verdict["silent_bug_signatures"]
    assert "empty_candidate_generation_pool" in archive_verdict["silent_bug_signatures"]
    assert a4_verdict["verdict"] == "silent_bug_must_reopen"
    assert "byte_identical_online_driver_arms" in a4_verdict["silent_bug_signatures"]


def test_scenario_arc_wmte_4725_keeps_exercised_nulls_trustworthy() -> None:
    """SCENARIO-ARC-WMTE-4725-AUDIT-CLASSIFICATION: exercised mechanisms are not reopened."""

    from carnot import experiment_4725_silent_bug_audit as mod

    object_centric = {
        "honest_verdict": "complete: object_centric_perception_no_new_level_residual_x",
        "proposal_coverage_by_representation": {
            "object_centric": {
                "coverage": 1.0,
                "step_hits": [{"candidate_count": 186, "rank": 59}],
            },
            "order1": {"coverage": 0.75, "step_hits": [{"candidate_count": 34, "rank": 7}]},
        },
        "target_arm_results": {
            "object_centric": {
                "object_centric_proposal_diagnostics": {
                    "enabled": True,
                    "augmented_candidates": 6913,
                    "candidate_scores": 8477,
                    "offpath_transition_observations": 155,
                }
            }
        },
        "bare_control_passed": True,
        "false_negative_risk_checked": True,
    }
    corrected_online = {
        "honest_verdict": "complete: online_action_learning_no_first_win_lift_null",
        "positive_control_passed": True,
        "arms": [
            {
                "arm": "frozen",
                "first_win_rate": 0.04,
                "scorer_diagnostics": {"observed": 0, "fits": 0, "errors": 0},
            },
            {
                "arm": "online-warm",
                "first_win_rate": 0.04,
                "scorer_diagnostics": {"observed": 4737, "fits": 935, "errors": 0},
            },
            {
                "arm": "online-warm-propose",
                "first_win_rate": 0.04,
                "scorer_diagnostics": {"observed": 4737, "fits": 935, "errors": 0},
            },
        ],
    }

    perception_verdict = mod.classify_null(
        "experiment_4700_object_centric_perception_proposal_live",
        object_centric,
    )
    online_verdict = mod.classify_null("experiment_4710_arms_summary", corrected_online)

    assert perception_verdict["verdict"] == "trustworthy_null"
    assert "object_centric_pool_nonempty" in perception_verdict["exercise_evidence"]
    assert online_verdict["verdict"] == "trustworthy_null"
    assert "online_cnn_observed_4737_fits_935_errors_0" in online_verdict["exercise_evidence"]


def test_req_arc_wmte_4725_run_audits_checked_in_artifacts() -> None:
    """REQ-ARC-WMTE-4725: checked-in artifacts produce the required audit schema."""

    from carnot import experiment_4725_silent_bug_audit as mod

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"].startswith(
        f"complete: silent_bug_audit_{artifact['nulls_audited']}_nulls_"
    )
    assert artifact["go_explore_fix_confirmed"] is True
    assert artifact["a4_tautology_verdict"].startswith("online_driver_arms_degenerate")
    assert artifact["audit_report_path"] == mod.REPORT_RELATIVE_PATH
    assert artifact["verifier_is_oracle"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    silent_ids = {row["null_id"] for row in artifact["silent_bug_nulls"]}
    assert "experiment_4701_amortized_exploration_prior_go_explore_live" in silent_ids
    assert "experiment_4715_online_action_learning_driver_corrected" in silent_ids
    assert artifact["preconditions_checked"]["arc_go_explore_importable"] is True
    retimed = dict(artifact)
    retimed["duration_s"] = artifact["duration_s"] + 123.0
    assert mod.payload_checksum(retimed) == artifact["reproducibility_checksum"]


def test_req_arc_wmte_4725_report_write_is_stable(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4725: report and JSON artifacts write deterministically."""

    from carnot import experiment_4725_silent_bug_audit as mod

    artifact = {
        "experiment": mod.EXPERIMENT,
        "schema": mod.SCHEMA,
        "honest_verdict": "complete: silent_bug_audit_1_nulls_1_must_reopen",
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "spec_refs": [
            "REQ-ARC-WMTE-4725",
            "SCENARIO-ARC-WMTE-4725-AUDIT-CLASSIFICATION",
        ],
        "nulls_audited": 1,
        "silent_bug_nulls": [
            {
                "null_id": "experiment_4715_online_action_learning_driver_corrected",
                "verdict": "silent_bug_must_reopen",
                "evidence": ["all arms first_win=0.04"],
                "silent_bug_signatures": ["byte_identical_online_driver_arms"],
                "artifact_path": "results/experiment_4715_online_action_learning_driver_corrected.json",
            }
        ],
        "a4_tautology_verdict": "online_driver_arms_degenerate (no-op, must reopen)",
        "trustworthy_nulls": [],
        "reopen_recommendations": [
            {
                "priority": "P0",
                "lever": "online_action_learning_driver",
                "recommendation": "reopen_as_435_A1",
                "source_nulls": ["experiment_4715_online_action_learning_driver_corrected"],
                "reason": "all driver arms are byte-identical",
            }
        ],
        "go_explore_fix_confirmed": True,
        "audit_report_path": mod.REPORT_RELATIVE_PATH,
        "verifier_is_oracle": False,
        "random_seed": mod.RANDOM_SEED,
        "preconditions_checked": {
            "ok": True,
            "arc_go_explore_importable": True,
            "missing_artifacts": [],
        },
        "field_principles": mod.FIELD_PRINCIPLES,
        "audited_artifacts": [
            "results/experiment_4715_online_action_learning_driver_corrected.json"
        ],
        "audited_artifact_checksums": {
            "results/experiment_4715_online_action_learning_driver_corrected.json": "sha256:test"
        },
        "per_null_verdicts": [
            {
                "null_id": "experiment_4715_online_action_learning_driver_corrected",
                "verdict": "silent_bug_must_reopen",
                "evidence": ["all arms first_win=0.04"],
                "silent_bug_signatures": ["byte_identical_online_driver_arms"],
                "artifact_path": "results/experiment_4715_online_action_learning_driver_corrected.json",
            }
        ],
        "duration_s": 0.001,
    }
    artifact["reproducibility_checksum"] = mod.payload_checksum(artifact)

    report = mod.write_report(artifact, root=tmp_path)
    result_path = mod.write_artifact(artifact, root=tmp_path)
    loaded = json.loads(result_path.read_text(encoding="utf-8"))

    assert report == tmp_path / mod.REPORT_RELATIVE_PATH
    assert report.exists()
    assert "online_driver_arms_degenerate" in report.read_text(encoding="utf-8")
    assert loaded == artifact
    assert mod.artifact_schema_errors(loaded) == []


def test_req_arc_wmte_4725_defensive_paths_are_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4725: malformed inputs fail closed instead of fabricating audit data."""

    from carnot import experiment_4725_silent_bug_audit as mod
    from carnot.agentic import arc_go_explore

    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(non_object)

    assert mod._finite_float(True) is None
    assert mod._finite_float("not-a-number") is None

    classifier_cases: list[tuple[str, dict[str, Any], str]] = [
        (
            "experiment_4628_dense_curiosity_progress_loop",
            {},
            "curiosity_loop_no_exercise_evidence",
        ),
        (
            "experiment_4640_goal_energy_generation_live",
            {},
            "goal_energy_measurements_missing",
        ),
        (
            "experiment_4640_goal_energy_generation_live",
            {
                "baseline_measurement": {"variant_attempts": [{"game": "a", "score": 1}]},
                "goal_energy_measurement": {
                    "variant_attempts": [
                        {
                            "game": "a",
                            "score": 2,
                            "goal_energy_neutral_on_cached_frame": False,
                        }
                    ]
                },
                "uniform_measurement": {"variant_attempts": [{"game": "b", "score": 1}]},
            },
            "goal_energy rows=1",
        ),
        (
            "experiment_4653_energy_fitness_qd_generation_live",
            {},
            "qd_measurements_missing",
        ),
        (
            "experiment_4653_energy_fitness_qd_generation_live",
            {
                "search_measurement": {"attempts": [{"game": "a", "score": 1}]},
                "random_mutation_measurement": {"attempts": [{"game": "b", "score": 1}]},
                "qd_measurement": {"attempts": [{"game": "a", "score": 2}]},
            },
            "search_attempts=1",
        ),
        (
            "experiment_4676_hierarchical_subgoal_search_live",
            {},
            "subgoal_evidence_missing",
        ),
        (
            "experiment_4676_hierarchical_subgoal_search_live",
            {
                "subgoal_decomposition": ["reach-key"],
                "per_subgoal_reachable": [True],
                "target_arm_results": {"hierarchical_subgoal": {"reached_level": 1}},
            },
            "subgoals=1",
        ),
        (
            "experiment_4700_object_centric_perception_proposal_live",
            {},
            "empty_object_centric_candidate_pool",
        ),
        (
            "experiment_4710_arms_summary",
            {},
            "online_arm_summary_missing",
        ),
        (
            "experiment_4710_arms_summary",
            {
                "arms": [
                    {"arm": "online-warm", "first_win_rate": 0.04},
                    {
                        "arm": "online-warm-propose",
                        "first_win_rate": 0.04,
                        "scorer_diagnostics": {"observed": 0, "fits": 0, "errors": 1},
                    },
                ]
            },
            "dropped_dict_candidate",
        ),
        (
            "experiment_4712_perception_grounded_l2_goal_lp85",
            {},
            "structural_goal_detector_degenerate",
        ),
        (
            "experiment_4713_surface_present_winner_verifier_ranker",
            {},
            "surfacing_ranker_no_exercise_evidence",
        ),
        (
            "experiment_4715_online_action_learning_driver_corrected",
            {
                "frozen_first_win": 0.04,
                "online_scratch_first_win": 0.08,
                "online_warm_first_win": 0.12,
                "online_warm_vs_frozen_delta": 0.08,
            },
            "driver arm rates",
        ),
        ("unknown_null", {}, "unknown_null_scope"),
    ]
    for null_id, payload, expected in classifier_cases:
        verdict = mod.classify_null(null_id, payload)
        combined = json.dumps(verdict, sort_keys=True)
        assert expected in combined

    monkeypatch.setitem(mod.CLASSIFIERS, "synthetic_noop", lambda artifact, evidence, signatures: None)
    no_evidence = mod.classify_null("synthetic_noop", {})
    assert no_evidence["verdict"] == "trustworthy_null"
    assert no_evidence["evidence"] == ["no silent representation no-op signature found"]

    original_frame_grid = arc_go_explore._frame_grid

    def raising_frame_grid(frame: Any) -> Any:
        raise RuntimeError("synthetic")

    monkeypatch.setattr(arc_go_explore, "_frame_grid", raising_frame_grid)
    assert mod.go_explore_fix_confirmed() is False
    monkeypatch.setattr(arc_go_explore, "_frame_grid", original_frame_grid)

    import builtins

    original_import = builtins.__import__

    def raising_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "carnot.agentic":
            raise ImportError("synthetic arc_go_explore import failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raising_import)
    checks = mod.check_preconditions(REPO)
    assert checks["arc_go_explore_importable"] is False
    assert "synthetic arc_go_explore import failure" in checks["arc_go_explore_error"]
    monkeypatch.setattr(builtins, "__import__", original_import)

    recommendations = mod._recommendations(
        [{"null_id": "experiment_4710_arms_summary", "verdict": "silent_bug_must_reopen"}]
    )
    assert recommendations[0]["recommendation"] == "rerun_after_dict_candidate_normalization"

    blocked = mod.run(root=tmp_path, write=True)
    assert blocked["honest_verdict"] == "blocked_null_artifacts_missing"
    assert (tmp_path / mod.REPORT_RELATIVE_PATH).exists()
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()
    assert (
        mod._blocked_artifact(
            {
                "arc_go_explore_importable": True,
                "go_explore_frame_grid_fix_confirmed": False,
            },
            0.001,
        )["honest_verdict"]
        == "blocked_arc_go_explore_frame_grid_fix_unconfirmed"
    )
    assert (
        mod._blocked_artifact(
            {
                "arc_go_explore_importable": False,
                "go_explore_frame_grid_fix_confirmed": False,
            },
            0.001,
        )["honest_verdict"]
        == "blocked_arc_go_explore_import"
    )


def test_req_arc_wmte_4725_validation_and_write_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4725: schema validation catches drift before writes."""

    from carnot import experiment_4725_silent_bug_audit as mod

    valid = mod.build_artifact(
        preconditions_checked={"go_explore_frame_grid_fix_confirmed": True},
        per_null_verdicts=[
            {
                "null_id": "experiment_4628_dense_curiosity_progress_loop",
                "verdict": "trustworthy_null",
                "evidence": ["synthetic exercised"],
                "exercise_evidence": ["synthetic exercised"],
                "silent_bug_signatures": [],
            }
        ],
        audited_artifact_checksums={},
        duration_s=0.001,
    )
    assert "No reopen recommendations." in mod.render_report(valid)

    invalid = dict(valid)
    invalid.update(
        {
            "honest_verdict": "not terminal",
            "inference_substrate": "wrong",
            "verifier_is_oracle": True,
            "audit_report_path": "wrong.md",
            "field_principles": {},
            "silent_bug_nulls": {},
            "trustworthy_nulls": {},
            "reopen_recommendations": {},
            "nulls_audited": 99,
            "reproducibility_checksum": "sha256:wrong",
        }
    )
    errors = mod.artifact_schema_errors(invalid)
    assert "honest_verdict_missing_terminal_prefix" in errors
    assert "inference_substrate_mismatch" in errors
    assert "verifier_is_oracle_must_be_false" in errors
    assert "audit_report_path_mismatch" in errors
    assert "field_principles_mismatch" in errors
    assert "reproducibility_checksum_mismatch" in errors
    assert "silent_bug_nulls_must_be_list" in errors
    assert "trustworthy_nulls_must_be_list" in errors
    assert "reopen_recommendations_must_be_list" in errors

    invalid_complete = dict(valid)
    invalid_complete["nulls_audited"] = 99
    invalid_complete["reproducibility_checksum"] = mod.payload_checksum(invalid_complete)
    assert "nulls_audited_does_not_match_verdict_lists" in mod.artifact_schema_errors(
        invalid_complete
    )

    with pytest.raises(ValueError, match="honest_verdict_missing_terminal_prefix"):
        mod.write_artifact(invalid, root=tmp_path)

    target = {
        "null_id": "experiment_4715_online_action_learning_driver_corrected",
        "artifact_path": "results/synthetic_4715.json",
        "module_path": "python/carnot/synthetic_4715.py",
    }
    artifact_path = tmp_path / target["artifact_path"]
    module_path = tmp_path / target["module_path"]
    artifact_path.parent.mkdir(parents=True)
    module_path.parent.mkdir(parents=True)
    artifact_path.write_text(
        json.dumps(
            {
                "frozen_first_win": 0.04,
                "online_scratch_first_win": 0.04,
                "online_warm_first_win": 0.04,
                "online_warm_vs_frozen_delta": 0.0,
                "flagged_adversarial": True,
            }
        ),
        encoding="utf-8",
    )
    module_path.write_text("# synthetic module\n", encoding="utf-8")
    monkeypatch.setattr(mod, "AUDIT_TARGETS", (target,))
    monkeypatch.setattr(
        mod,
        "check_preconditions",
        lambda root: {"ok": True, "go_explore_frame_grid_fix_confirmed": True},
    )
    written = mod.run(root=tmp_path, write=True)
    assert written["honest_verdict"] == "complete: silent_bug_audit_1_nulls_1_must_reopen"
    assert (tmp_path / mod.REPORT_RELATIVE_PATH).exists()
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    monkeypatch.setattr(mod, "artifact_schema_errors", lambda artifact: ["synthetic_schema_error"])
    with pytest.raises(ValueError, match="synthetic_schema_error"):
        mod.run(root=tmp_path, write=False)
