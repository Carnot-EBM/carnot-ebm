"""Tests for Exp 4423 .408 milestone capstone.

Spec refs: REQ-CAPSTONE-4423, SCENARIO-CAPSTONE-4423.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import capstone_v408_4423 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_support_files(root: Path, total: int = 34, games: int = 17) -> None:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "publication_gate.py").write_text("# fixture\n", encoding="utf-8")
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "arc_solve_registry.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "reproducible_total_levels": total,
                "reproducible_total_games": games,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _clean_live_flags(_: Path) -> list[dict[str, str]]:
    return []


def _summarize_zero(_: Path, __: Path) -> int:
    return 0


def _publication_gate(paper_ready: bool = True) -> JsonDict:
    return {
        "paper_ready": paper_ready,
        "gates": {
            "G1": {"pass": True, "detail": "fixture", "source": "exp2850"},
            "G2": {"pass": paper_ready, "detail": "fixture"},
            "G3": {"pass": True, "detail": "fixture", "hits": []},
            "G4": {"pass": True, "detail": "fixture", "source": "exp2850"},
        },
        "unmet_gates": [] if paper_ready else ["G2"],
    }


def _fixture_payloads(
    *,
    new_levels: int = 0,
    grounded_rule: bool = True,
    hidden_signal: bool = False,
    sovereign_holds: bool = True,
    vocabulary_transfers: bool = False,
    calibrated: bool = False,
) -> dict[str, JsonDict]:
    return {
        "4412_prior_capstone": {
            "experiment": 4412,
            "honest_verdict": "complete: v407_fixture",
            "localizer_state": "position_bound_retired",
            "localizer_compounds": False,
            "detection_calibrated_multi_domain": False,
            "reproducible_total_levels": 34,
            "publication_gate": _publication_gate(True),
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4414_config_rule": {
            "experiment": "experiment_4414_config_rule_induction_solve",
            "honest_verdict": "success_config_rule_fixture" if new_levels else "complete_config_rule_partial",
            "new_levels_reproduced": new_levels,
            "reproducible_total_levels": 34 + new_levels,
            "config_win_rules_grounded": [
                {
                    "game": "ka59",
                    "tier": 2 if grounded_rule else 1,
                    "predicate": "editable_count_4_equals_reference_count_4_32",
                    "fires_on_win": grounded_rule,
                    "false_positive_rate": 0.0,
                    "literal_hardcode": False,
                }
            ],
            "per_target_scorecard": [
                {
                    "game": "ka59",
                    "grounding_tier": 2 if grounded_rule else 1,
                    "offline_reproduced": new_levels > 0,
                    "new_reproduced_level": 2 if new_levels else 1,
                    "prior_best_level": 1,
                    "search_blocker": "" if new_levels else "no_registered_next_level_config_adapter",
                    "win_rule_predicate": "editable_count_4_equals_reference_count_4_32",
                }
            ],
            "preconditions_checked": {"trm_training_stood_down": True},
            "verifier_is_oracle": True,
            "random_seed": 4414,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4415_agent2world": {
            "experiment": "experiment_4415_agent2world_adaptive_e3_repair",
            "honest_verdict": "complete_e3_adaptive_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34 + new_levels,
            "per_target_scorecard": [
                {
                    "game": "tn36",
                    "adaptive_tests_passed": 1,
                    "adaptive_tests_total": 2,
                    "held_out_mechanic_test_pass": True,
                    "offline_reproduced": False,
                    "new_reproduced_level": 7,
                    "prior_best_level": 7,
                    "residual_failing_behavior": "fixture_residual",
                }
            ],
            "preconditions_checked": {"trm_training_stood_down": True},
            "verifier_is_oracle": True,
            "random_seed": 4415,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4416_hidden_state": {
            "experiment": "experiment_4416_hidden_state_localizer_falsification_audit",
            "honest_verdict": (
                "success: hidden_state_off_text_signal"
                if hidden_signal
                else "complete: clean_powered_null_position_only_not_beaten"
            ),
            "hidden_state_localizer_has_nonposition_signal": hidden_signal,
            "position_only_baseline_f1": 1.0,
            "localization_f1_comparison": {
                "hidden_state_probe_f1": 1.04 if hidden_signal else 1.0,
                "position_only_baseline_f1": 1.0,
                "delta_vs_position_only": 0.04 if hidden_signal else 0.0,
                "delta_ci95": [0.01, 0.07] if hidden_signal else [0.0, 0.0],
                "n_traces": 1000,
            },
            "preconditions_checked": [
                {"resource": "trm_training_stand_down", "available": True}
            ],
            "verifier_is_oracle": False,
            "random_seed": 4416,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4417_sovereign_gap4": {
            "experiment": "experiment_4417_gap4_local_generator_sovereign_arm",
            "honest_verdict": (
                "complete: sovereign_gap4_local_gate_holds"
                if sovereign_holds
                else "complete: sovereign_gap4_local_gate_null"
            ),
            "sovereign_gap4_gate_holds": sovereign_holds,
            "pass2_vs_vote": {
                "vote_pass2": 0.45,
                "gated_pass2": 0.45 if sovereign_holds else 0.41,
                "delta": 0.0 if sovereign_holds else -0.04,
                "delta_ci95": [0.0, 0.0] if sovereign_holds else [-0.08, -0.01],
                "pass2_vote_wins_lost": 0 if sovereign_holds else 2,
                "graded_gate_fires": 0 if sovereign_holds else 3,
            },
            "local_generator_coverage": 0.2333,
            "preconditions_checked": [
                {"resource": "trm_training_stood_down", "available": True}
            ],
            "verifier_is_oracle": True,
            "random_seed": 4417,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4418_vocabulary": {
            "experiment": "experiment_4418_config_rule_vocabulary_transfer",
            "honest_verdict": (
                "success: config_rule_vocabulary_transfers"
                if vocabulary_transfers
                else "blocked_local_model_unavailable"
            ),
            "config_rule_vocabulary_transfers": vocabulary_transfers,
            "preconditions_checked": {"trm_training_stood_down": True},
            "verifier_is_oracle": False,
            "random_seed": 4418,
            "reproducibility_checksum": "sha256:" + "f" * 64,
        },
        "4419_detection": {
            "experiment": "experiment_4419_steerconf_code_detection_calibration_repair",
            "honest_verdict": (
                "success: steered_confidence_rescues_code_detector"
                if calibrated
                else "complete: clean_null_steered_confidence_does_not_rescue_code_detector"
            ),
            "detection_calibrated_multi_domain": calibrated,
            "detection_by_domain": [
                {"domain": "code_humaneval", "detection_auroc": 0.7 if calibrated else 0.60191}
            ],
            "domains_at_chance": [] if calibrated else ["code_humaneval"],
            "preconditions_checked": [
                {"resource": "trm_training_stand_down", "available": True}
            ],
            "verifier_is_oracle": False,
            "random_seed": 4419,
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4423_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4423: OpenSpec declares the .408 scorecard contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4423" in spec
    assert "SCENARIO-CAPSTONE-4423" in spec
    assert "experiment_4423_capstone_v408.json" in spec
    assert "publication_gate.py --json" in spec
    assert "CIRCULAR_MOAT_OVERCLAIM" in spec
    assert "aggregate-available" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_capstone_4423_current_artifacts_report_headline_decision() -> None:
    """SCENARIO-CAPSTONE-4423: current .408 artifacts report the honest scorecard."""

    artifact = mod.build_artifact(
        Path.cwd(),
        started_s=1.0,
        now_s=1.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: v408_config_rule_grounded_no_new_levels_localizer_closed_"
        "sovereign_gap4_holds_vocab_false_detection_false_arc_levels_34_publication_ready"
    )
    assert artifact["arc_config_rule_state"] == "grounded_config_rules_no_new_reproducible_levels"
    assert artifact["arc_config_rule"]["grounded_win_rules_count"] == 1
    assert artifact["arc_config_rule"]["new_levels_reproduced_from_artifacts"] == 0
    assert artifact["arc_config_rule"]["execution_grounded"] is True
    assert artifact["localizer_program_state"] == "closed_position_bound_text_and_hidden"
    assert artifact["localizer_program"]["hidden_state_localizer_has_nonposition_signal"] is False
    assert artifact["sovereign_verifier_state"] == "sovereign_gap4_local_gate_holds_execution_grounded"
    assert artifact["sovereign_verifier"]["sovereign_gap4_gate_holds"] is True
    assert artifact["sovereign_verifier"]["pass2_vs_vote"]["pass2_vote_wins_lost"] == 0
    assert artifact["config_rule_vocabulary_transfers"] is False
    assert artifact["detection_calibrated_multi_domain"] is False
    assert artifact["reproducible_total_levels"] == 34
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["publication_gate"]["unmet_gates"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verifier_is_oracle_honored"] is True
    assert artifact["preconditions_checked"]["trm_training_stood_down"] is True
    assert artifact["verifier_thesis_state"] == (
        "config_rule_grounded_no_new_levels_localizer_closed_sovereign_gap4_holds_"
        "vocab_no_transfer_detection_domain_bound_arc_levels_34"
    )
    assert {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]} == {
        4412,
        4414,
        4415,
        4416,
        4417,
        4418,
        4419,
    }
    assert all("sha256" in row and row["fields_imported"] for row in artifact["cited_upstream_artifacts"])
    assert artifact["capstone_live_adversarial_recheck"] == {"status": "not_run_until_write"}


def test_scenario_capstone_4423_positive_fixture_records_all_axes(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4423: positive fixture reports new ARC and transfer axes."""

    _write_support_files(tmp_path, total=37, games=18)
    _write_default_artifacts(
        tmp_path,
        _fixture_payloads(
            new_levels=3,
            hidden_signal=True,
            sovereign_holds=True,
            vocabulary_transfers=True,
            calibrated=True,
        ),
    )

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=3.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["arc_config_rule_state"] == "new_reproducible_levels_added"
    assert artifact["arc_config_rule"]["new_levels_reproduced_from_artifacts"] == 3
    assert artifact["arc_reproducible_progress"]["new_levels_since_prior"] == 3
    assert artifact["arc_reproducible_progress"]["new_games_since_prior"] == 1
    assert artifact["localizer_program_state"] == "off_text_signal_logged_gap"
    assert artifact["config_rule_vocabulary_transfers"] is True
    assert artifact["detection_calibrated_multi_domain"] is True
    assert artifact["verifier_thesis_state"] == (
        "config_rule_new_levels_localizer_off_text_signal_sovereign_gap4_holds_"
        "vocab_transfers_detection_calibrated_arc_levels_37"
    )
    assert artifact["honest_verdict"].endswith("arc_levels_37_publication_ready")


def test_req_capstone_4423_missing_and_flagged_inputs_do_not_erase_other_axes(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4423: missing and flagged upstreams are per-axis gaps only."""

    _write_support_files(tmp_path, total=35, games=17)
    payloads = _fixture_payloads(new_levels=1, sovereign_holds=True, calibrated=True)
    payloads.pop("4418_vocabulary")
    payloads["4419_detection"]["flagged_adversarial"] = True
    _write_default_artifacts(tmp_path, payloads)

    def critical_detection(path: Path) -> list[dict[str, str]]:
        if path.name == "experiment_4419_steerconf_code_detection_calibration_repair.json":
            return [{"kind": "FIXTURE_CRITICAL", "severity": "critical", "detail": "fixture"}]
        return []

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=5.0,
        live_flag_runner=critical_detection,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["arc_config_rule_state"] == "new_reproducible_levels_added"
    assert artifact["sovereign_verifier_state"] == "sovereign_gap4_local_gate_holds_execution_grounded"
    assert artifact["config_rule_vocabulary_transfers"] is False
    assert artifact["detection_calibrated_multi_domain"] is False
    assert artifact["per_axis_gaps"] == [
        {"axis": "vocabulary", "artifact_key": "4418_vocabulary", "experiment_id": 4418}
    ]
    assert artifact["flagged_artifacts_excluded"] == [
        {
            "artifact_key": "4419_detection",
            "experiment_id": 4419,
            "path": "results/experiment_4419_steerconf_code_detection_calibration_repair.json",
            "sha256": artifact["upstream_provenance"][-1]["sha256"],
            "stamped_flagged_adversarial": True,
            "live_critical": True,
            "parse_error": "",
            "live_critical_flags": [
                {"kind": "FIXTURE_CRITICAL", "severity": "critical", "detail": "fixture"}
            ],
            "reason": "flagged_adversarial",
        }
    ]
    assert 4419 not in {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}
    assert artifact["availability_report"]["axes"]["detection"]["flagged_artifacts"] == [
        {
            "axis": "detection",
            "artifact_key": "4419_detection",
            "experiment_id": 4419,
            "reason": "flagged_adversarial",
        }
    ]


def test_req_capstone_4423_branch_helpers_keep_alternate_states_honest() -> None:
    """REQ-CAPSTONE-4423: alternate reads stay bounded and non-headline."""

    assert mod._summary_rows([object()]) == []  # noqa: SLF001
    assert mod._grounded_win_rules({"config_win_rules_grounded": [object()]}) == []  # noqa: SLF001
    assert mod.config_rule_read(None, True) == {"status": "excluded_flagged_adversarial"}
    assert mod.config_rule_read(None, False) == {"status": "missing_or_excluded"}
    assert mod.agent2world_read(None, True) == {"status": "excluded_flagged_adversarial"}
    assert mod.agent2world_read(None, False) == {"status": "missing_or_excluded"}
    assert mod.localizer_program_read(None, True) == {"status": "excluded_flagged_adversarial"}
    assert mod.localizer_program_read(None, False) == {"status": "missing_or_excluded"}
    assert (
        mod.localizer_program_read({"localization_f1_comparison": []}, False)[
            "localization_f1_comparison"
        ]
        == {}
    )
    assert mod.decide_localizer_program_state({"status": "missing_or_excluded"}) == (
        "localizer_program_missing_or_excluded"
    )
    assert mod.sovereign_verifier_read(None, True) == {"status": "excluded_flagged_adversarial"}
    assert mod.sovereign_verifier_read(None, False) == {"status": "missing_or_excluded"}
    assert mod.decide_sovereign_verifier_state({"status": "missing_or_excluded"}) == (
        "sovereign_gap4_missing_or_excluded"
    )
    assert mod.decide_sovereign_verifier_state({"status": "null"}) == "sovereign_gap4_local_gate_null"
    assert mod.vocabulary_read(None, True) == {
        "status": "excluded_flagged_adversarial",
        "config_rule_vocabulary_transfers": False,
    }
    assert mod.detection_read(None, False) == {
        "status": "missing_or_excluded",
        "detection_calibrated_multi_domain": False,
    }
    assert (
        mod._cited_upstream_artifacts(  # noqa: SLF001
            [{"skipped": False, "fields_imported": [], "experiment_id": 1}]
        )
        == []
    )
    assert mod._trm_stood_down({"note": "no TRM training invoked"}) is True  # noqa: SLF001
    assert mod._trm_stood_down({"outer": {"trm_training_stood_down": True}}) is True  # noqa: SLF001
    assert mod.decide_arc_config_rule_state(
        {"new_levels_reproduced_from_artifacts": 0, "grounded_win_rules_count": 0},
        {"new_levels_since_prior": 2},
    ) == "new_reproducible_levels_added"
    assert mod.decide_arc_config_rule_state(
        {"new_levels_reproduced_from_artifacts": 0, "grounded_win_rules_count": 0},
        {"new_levels_since_prior": 0},
    ) == "config_toggle_class_blocked"
    assert mod.verifier_thesis_state(
        "config_toggle_class_blocked",
        "localizer_program_missing_or_excluded",
        "sovereign_gap4_local_gate_null",
        False,
        False,
        0,
    ) == "config_toggle_blocked_localizer_closed_sovereign_gap4_null_vocab_no_transfer_detection_domain_bound_arc_levels_0"
    assert mod._honest_verdict(  # noqa: SLF001
        "config_toggle_class_blocked",
        "localizer_program_missing_or_excluded",
        "sovereign_gap4_local_gate_null",
        False,
        False,
        0,
        True,
        False,
    ).endswith("publication_not_ready")
    assert mod._honest_verdict(  # noqa: SLF001
        "config_toggle_class_blocked",
        "localizer_program_missing_or_excluded",
        "sovereign_gap4_local_gate_null",
        False,
        False,
        0,
        False,
        False,
    ).endswith("publication_gate_gap")


def test_scenario_capstone_4423_write_artifact_records_clean_live_recheck(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4423: written capstone carries the live adversarial re-check."""

    _write_support_files(tmp_path, total=34, games=17)
    _write_default_artifacts(tmp_path, _fixture_payloads())

    output = mod.write_artifact(
        tmp_path,
        started_s=6.0,
        now_s=7.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
        capstone_live_flag_runner=_clean_live_flags,
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert artifact["capstone_live_adversarial_recheck"] == {
        "status": "clean",
        "flags": [],
        "circular_moat_overclaim": False,
    }
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert artifact["verifier_is_oracle"] is False
