"""Tests for Exp 4755 generation-lever silent no-op audit.

Spec refs: REQ-ARC-WMTE-4755, SCENARIO-ARC-WMTE-4755-GENERATION-LEVER-AUDIT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def test_req_arc_wmte_4755_spec_declares_artifact_contract() -> None:
    """REQ-ARC-WMTE-4755: the spec names the audit and required principles."""

    from carnot import experiment_4755_silent_bug_audit as mod

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4755" in spec
    assert "SCENARIO-ARC-WMTE-4755-GENERATION-LEVER-AUDIT" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_4755_classifies_silent_no_op_signatures() -> None:
    """SCENARIO-ARC-WMTE-4755-GENERATION-LEVER-AUDIT: dead-code nulls reopen."""

    from carnot import experiment_4755_silent_bug_audit as mod

    cloned_goal_energy = {
        "baseline_measurement": {"variant_attempts": [{"game": "a", "score": 1}]},
        "goal_energy_measurement": {
            "variant_attempts": [
                {
                    "game": "a",
                    "score": 1,
                    "goal_energy_neutral_on_cached_frame": True,
                }
            ]
        },
        "uniform_measurement": {
            "variant_attempts": [
                {"game": "a", "score": 1, "uniform_energy_ablation": True}
            ]
        },
    }
    dead_archive = {
        "candidate_generation_coverage_with_prior": 0.0,
        "target_arm_results": {
            "coverage": {"with_prior": {"total_steps": 0}},
            "with_prior": [
                {
                    "go_explore_archive_diagnostics": {
                        "observations": 0,
                        "stored_cells": 0,
                        "actions_injected": 0,
                        "prefixes_injected": 0,
                    }
                }
            ],
        },
    }
    dropped_cnn = {
        "arms": [
            {
                "arm": "online-warm",
                "first_win_rate": 0.04,
                "scorer_diagnostics": {"observed": 0, "fits": 0, "errors": 2},
            }
        ]
    }

    goal = mod.classify_lever("experiment_4640_goal_energy_generation_live", cloned_goal_energy)
    archive = mod.classify_lever(
        "experiment_4701_amortized_exploration_prior_go_explore_live",
        dead_archive,
    )
    cnn = mod.classify_lever("experiment_4710_arms_summary", dropped_cnn)

    assert goal["classification"] == "must_reopen"
    assert "byte_identical_arms" in goal["no_op_signature"]
    assert "scorer_or_energy_never_fires" in goal["no_op_signature"]
    assert archive["classification"] == "must_reopen"
    assert "dead_archive" in archive["no_op_signature"]
    assert "empty_candidate_pool" in archive["no_op_signature"]
    assert cnn["classification"] == "must_reopen"
    assert "scorer_or_cnn_never_fires" in cnn["no_op_signature"]


def test_scenario_arc_wmte_4755_keeps_genuine_nulls_closed() -> None:
    """SCENARIO-ARC-WMTE-4755-GENERATION-LEVER-AUDIT: exercised levers stay genuine."""

    from carnot import experiment_4755_silent_bug_audit as mod

    exercised_online = {
        "arms": [
            {
                "arm": "online-warm",
                "first_win_rate": 0.04,
                "scorer_diagnostics": {"observed": 4737, "fits": 935, "errors": 0},
            }
        ]
    }
    exercised_novelty = {
        "target_arm_results": {
            "controllable_novelty": {
                "controllable_novelty_diagnostics": {
                    "candidate_scores": 4641,
                    "observed_effects": 194,
                    "rnd_updates": 194,
                }
            }
        }
    }
    exercised_program_filter = {
        "target_arm_results": {
            "candidate_generation_probe": {
                "heldout_programs_kept": 0,
                "heldout_programs_rejected": 2,
                "program_trust_weights": [{"kept": False, "trust": 0.0}],
            }
        }
    }

    online = mod.classify_lever("experiment_4710_arms_summary", exercised_online)
    novelty = mod.classify_lever(
        "experiment_4688_controllable_novelty_proposal_policy_live",
        exercised_novelty,
    )
    program_filter = mod.classify_lever(
        "experiment_4689_program_synthesis_action_effect_proposal_filter",
        exercised_program_filter,
    )

    assert online["classification"] == "genuine_null"
    assert online["no_op_signature"] == []
    assert novelty["classification"] == "genuine_null"
    assert "controllable_novelty_exercised" in novelty["evidence"][0]
    assert program_filter["classification"] == "genuine_null"
    assert "program_filter_exercised" in program_filter["evidence"][0]


def test_req_arc_wmte_4755_run_audits_checked_in_artifacts() -> None:
    """REQ-ARC-WMTE-4755: checked-in artifacts produce a stable complete audit."""

    from carnot import experiment_4755_silent_bug_audit as mod

    artifact = mod.run(root=REPO, write=False)

    assert mod.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"].startswith("complete_generation_lever_silent_bug_audit_")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["upstream_artifacts_present"] is True
    assert artifact["preconditions_checked"]["arcade_import_exits_0"] is True
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    assert set(artifact["levers_audited"]) == {target["lever"] for target in mod.AUDIT_TARGETS}
    assert len(artifact["silent_no_op_findings"]) == len(mod.AUDIT_TARGETS)
    must_reopen = set(artifact["must_reopen"])
    assert "experiment_4640_goal_energy_generation_live" in must_reopen
    assert "experiment_4653_energy_fitness_qd_generation_live" in must_reopen
    assert "experiment_4701_amortized_exploration_prior_go_explore_live" in must_reopen
    assert "experiment_4710_arms_summary" not in must_reopen


def test_req_arc_wmte_4755_write_and_blocked_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-4755: blocked preconditions and schema guards fail closed."""

    from carnot import experiment_4755_silent_bug_audit as mod

    original_check_preconditions = mod.check_preconditions
    artifact = mod.build_artifact(
        preconditions_checked={"ok": True, "upstream_artifacts_present": True},
        findings=[
            {
                "lever": "experiment_4640_goal_energy_generation_live",
                "no_op_signature": ["byte_identical_arms"],
                "classification": "must_reopen",
                "evidence": ["synthetic cloned arms"],
            }
        ],
        audited_artifact_checksums={},
        duration_s=0.001,
    )
    path = mod.write_artifact(artifact, root=tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == artifact

    invalid = dict(artifact)
    invalid["honest_verdict"] = "not terminal"
    invalid["reproducibility_checksum"] = mod.payload_checksum(invalid)
    with pytest.raises(ValueError, match="honest_verdict_missing_terminal_prefix"):
        mod.write_artifact(invalid, root=tmp_path)

    missing = mod.run(root=tmp_path, write=True)
    assert missing["honest_verdict"] == "blocked_upstream_artifacts"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()

    monkeypatch.setattr(
        mod,
        "check_preconditions",
        lambda root: {
            "ok": False,
            "upstream_artifacts_present": True,
            "missing_upstream_artifacts": [],
            "arcade_import_exits_0": False,
            "arcade_import_error": "synthetic",
        },
    )
    blocked_arcade = mod.run(root=REPO, write=False)
    assert blocked_arcade["honest_verdict"] == "blocked_arcade_import"

    synthetic_target = {
        "lever": "experiment_4688_controllable_novelty_proposal_policy_live",
        "artifact_path": "results/synthetic_4688.json",
    }
    prior_path = tmp_path / mod.PRIOR_AUDIT_RELATIVE_PATH
    artifact_path = tmp_path / synthetic_target["artifact_path"]
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path.write_text('{"reopen_recommendations": []}', encoding="utf-8")
    artifact_path.write_text(
        json.dumps(
            {
                "target_arm_results": {
                    "controllable_novelty": {
                        "controllable_novelty_diagnostics": {
                            "candidate_scores": 3,
                            "observed_effects": 1,
                            "rnd_updates": 1,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "AUDIT_TARGETS", (synthetic_target,))
    monkeypatch.setattr(mod, "check_preconditions", original_check_preconditions)
    written = mod.run(root=tmp_path, write=True)
    assert written["honest_verdict"] == "complete_generation_lever_silent_bug_audit_1_levers_0_must_reopen"
    assert (tmp_path / mod.RESULT_RELATIVE_PATH).exists()


def test_req_arc_wmte_4755_defensive_helpers_are_covered() -> None:
    """REQ-ARC-WMTE-4755: helper branches handle malformed inputs explicitly."""

    from carnot import experiment_4755_silent_bug_audit as mod

    bad_json = Path("/tmp/exp4755_non_object.json")
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        mod._read_json(bad_json)

    assert mod._finite_float(True) is None
    assert mod._finite_float("nan") is None
    assert mod._int_value("not-int") == 0
    assert mod._attempts(None) == []

    qd_missing = mod.classify_lever("experiment_4653_energy_fitness_qd_generation_live", {})
    assert qd_missing["classification"] == "must_reopen"
    assert "missing_expected_measurements" in qd_missing["no_op_signature"]

    classifier_cases: list[tuple[str, dict[str, Any], str, str]] = [
        (
            "experiment_4640_goal_energy_generation_live",
            {},
            "must_reopen",
            "missing_expected_measurements",
        ),
        (
            "experiment_4640_goal_energy_generation_live",
            {
                "baseline_measurement": {"variant_attempts": [{"game": "a", "score": 1}]},
                "goal_energy_measurement": {"variant_attempts": [{"game": "a", "score": 2}]},
                "uniform_measurement": {"variant_attempts": [{"game": "b", "score": 3}]},
            },
            "genuine_null",
            "cloned_goal=False",
        ),
        (
            "experiment_4641_action_effect_expansion_prior_live",
            {},
            "must_reopen",
            "missing_expected_measurements",
        ),
        (
            "experiment_4641_action_effect_expansion_prior_live",
            {
                "ranker_measurement": {"attempts": [{"game": "a", "depth": 1}]},
                "expansion_measurement": {"attempts": [{"game": "a", "depth": 2}]},
            },
            "genuine_null",
            "rows differ",
        ),
        (
            "experiment_4653_energy_fitness_qd_generation_live",
            {
                "search_measurement": {"attempts": [{"game": "a", "score": 1}]},
                "random_mutation_measurement": {"attempts": [{"game": "b", "score": 1}]},
                "qd_measurement": {"attempts": [{"game": "a", "score": 2}]},
            },
            "genuine_null",
            "qd_cloned=False",
        ),
        (
            "experiment_4676_hierarchical_subgoal_search_live",
            {},
            "must_reopen",
            "missing_expected_measurements",
        ),
        (
            "experiment_4676_hierarchical_subgoal_search_live",
            {"subgoal_decomposition": ["reach"], "per_subgoal_reachable": [True]},
            "genuine_null",
            "subgoals=1",
        ),
        (
            "experiment_4688_controllable_novelty_proposal_policy_live",
            {},
            "must_reopen",
            "scorer_or_cnn_never_fires",
        ),
        (
            "experiment_4689_program_synthesis_action_effect_proposal_filter",
            {},
            "must_reopen",
            "scorer_or_energy_never_fires",
        ),
        (
            "experiment_4700_object_centric_perception_proposal_live",
            {},
            "must_reopen",
            "empty_candidate_pool",
        ),
        (
            "experiment_4710_arms_summary",
            {},
            "must_reopen",
            "missing_expected_measurements",
        ),
        (
            "experiment_4710_arms_summary",
            {"arms": [{"arm": "online-warm", "scorer_diagnostics": "malformed"}]},
            "must_reopen",
            "scorer_or_cnn_never_fires",
        ),
        (
            "experiment_4713_surface_present_winner_verifier_ranker",
            {},
            "must_reopen",
            "scorer_or_energy_never_fires",
        ),
    ]
    for lever, payload, classification, expected_text in classifier_cases:
        result = mod.classify_lever(lever, payload)
        assert result["classification"] == classification
        assert expected_text in json.dumps(result, sort_keys=True)

    unknown = mod.classify_lever("unknown", {})
    assert unknown["classification"] == "must_reopen"
    assert "unknown_generation_lever" in unknown["no_op_signature"]

    mod.CLASSIFIERS["synthetic_empty"] = lambda artifact, evidence, signatures: None
    empty = mod.classify_lever("synthetic_empty", {})
    assert empty["classification"] == "genuine_null"
    assert empty["evidence"] == ["no silent representation no-op signature found"]
    del mod.CLASSIFIERS["synthetic_empty"]

    assert mod._prior_reopen_list({}) == []

    blocked = mod._blocked_artifact(
        {"arcade_import_exits_0": True, "missing_upstream_artifacts": ["x"]},
        0.001,
    )
    assert blocked["honest_verdict"] == "blocked_upstream_artifacts"
    errors = mod.artifact_schema_errors({**blocked, "must_reopen": {}})
    assert "must_reopen_must_be_list" in errors

    invalid = dict(blocked)
    invalid.update(
        {
            "inference_substrate": "wrong",
            "field_principles": {},
            "verifier_is_oracle": True,
            "levers_audited": {},
            "silent_no_op_findings": {},
            "must_reopen": [],
            "reproducibility_checksum": "sha256:wrong",
        }
    )
    schema_errors = mod.artifact_schema_errors(invalid)
    assert "inference_substrate_mismatch" in schema_errors
    assert "field_principles_mismatch" in schema_errors
    assert "verifier_is_oracle_must_be_false" in schema_errors
    assert "levers_audited_must_be_list" in schema_errors
    assert "silent_no_op_findings_must_be_list" in schema_errors
    assert "reproducibility_checksum_mismatch" in schema_errors

    mismatch = mod.build_artifact(
        preconditions_checked={},
        findings=[{"lever": "a", "classification": "genuine_null", "no_op_signature": []}],
        audited_artifact_checksums={},
        duration_s=0.001,
    )
    mismatch["levers_audited"] = []
    mismatch["reproducibility_checksum"] = mod.payload_checksum(mismatch)
    assert "levers_audited_does_not_match_findings" in mod.artifact_schema_errors(mismatch)

    with pytest.raises(ValueError, match="synthetic_schema_error"):
        original_schema = mod.artifact_schema_errors
        try:
            mod.artifact_schema_errors = lambda artifact: ["synthetic_schema_error"]  # type: ignore[method-assign]
            mod.run(root=REPO, write=False)
        finally:
            mod.artifact_schema_errors = original_schema  # type: ignore[method-assign]
