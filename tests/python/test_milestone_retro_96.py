"""Tests for the Exp 1241 milestone .96 retrospective.

Spec: REQ-REPORT-018, SCENARIO-REPORT-015.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.milestone_retro_96 import (
    CRITERION_NAMES,
    build_artifact,
    evaluate_criteria,
    run,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_scenario_report_015_counts_only_autofill_and_self_when_sources_are_missing_or_incomplete(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-015: missing and incomplete .96 sources stay unmet."""

    results_dir = tmp_path / "results"
    _write_json(
        results_dir / "experiment_1229_milestone_retro_95.json",
        {"retro_complete": False},
    )
    _write_json(
        results_dir / "experiment_1230_auto_populate_prior_failures_v2.json",
        {"autofill_script_exists": True},
    )
    _write_json(
        results_dir / "experiment_1231_llms_gaming_verifiers_defense.json",
        {"gaming_defense_measured": False},
    )
    _write_json(
        results_dir / "experiment_1232_verifier_joint_orthogonality_audit.json",
        {"pairwise_correlation_matrix_measured": False, "k_eff": None},
    )
    _write_json(
        results_dir / "experiment_1235_grpo_v6_fspo_vps_extended.json",
        {"grpo_v6_training_completed": False},
    )
    _write_json(
        results_dir / "experiment_1237_boltzmann_gpt_contrastive_training.json",
        {"boltzmann_gpt_contrastive_auroc": None},
    )
    _write_json(
        results_dir / "experiment_1238_phase5d_intermediate_scale.json",
        {"gates_measured": 0},
    )
    _write_json(
        results_dir / "experiment_1239_nrgpt_frozen_prefix_evaluation.json",
        {"frozen_prefix_regime_classified": False},
    )
    _write_json(
        results_dir / "experiment_1240_wopr_kakuro_cartridge.json",
        {"status": "blocked"},
    )

    artifact = run(
        results_dir=results_dir,
        out_path=results_dir / "experiment_1241_milestone_retro_96.json",
    )

    assert artifact["criteria_total"] == 13
    assert artifact["criteria_met"] == 2
    assert artifact["honest_verdict"] == "milestone_2_of_13_criteria_met"
    assert artifact["retro_complete"] is True
    assert artifact["criteria_results"] == {
        "retro_95_complete": False,
        "autofill_script_v2_shipped": True,
        "gaming_defense_measured": False,
        "verifier_orthogonality_matrix_measured_6x6": False,
        "k_eff_documented_and_honest": False,
        "verifier_redesign_k_eff_above_3": False,
        "arxiv_v6_submitted": False,
        "grpo_v6_improvement_measured": False,
        "boltzmann_gpt_contrastive_auroc_above_0p80": False,
        "phase5d_all_8_gates_measured": False,
        "nrgpt_frozen_prefix_resolved": False,
        "kakuro_cartridge_shipped": False,
        "retro_96_complete": True,
    }
    assert len([s for s in artifact["findings_summary"].split(".") if s.strip()]) == 4
    assert len(artifact["key_carry_forwards"]) == 5
    assert (results_dir / "experiment_1241_milestone_retro_96.json").exists()


def test_req_report_018_counts_every_named_source_field_and_direct_redesign_threshold() -> None:
    """REQ-REPORT-018: criteria are derived from source fields, not status text."""

    sources: dict[int, dict[str, object]] = {
        1229: {"retro_complete": True},
        1230: {"autofill_script_exists": True},
        1231: {"gaming_defense_measured": True},
        1232: {"pairwise_correlation_matrix_measured": True, "k_eff": 5},
        1233: {"k_eff_after_redesign": 4},
        1234: {"pdf_compiled": True, "arxiv_submitted": False},
        1235: {"grpo_v6_improvement_measured": True},
        1237: {"boltzmann_gpt_above_0p80": True},
        1238: {"phase5d_all_8_gates_measured": True},
        1239: {"frozen_prefix_regime_classified": True},
        1240: {"kakuro_cartridge_shipped": True},
    }

    artifact = build_artifact(sources)

    assert tuple(artifact["criteria_results"]) == CRITERION_NAMES
    assert artifact["criteria_met"] == 13
    assert artifact["honest_verdict"] == "milestone_13_of_13_criteria_met"


def test_req_report_018_accepts_merged_redesign_equivalent() -> None:
    """REQ-REPORT-018: merged k_eff evidence can satisfy the redesign criterion."""

    criteria = evaluate_criteria(
        {
            1232: {
                "pairwise_correlation_matrix_measured": True,
                "k_eff": 2,
                "verifier_redesign_k_eff_above_3": True,
            }
        },
        retro_complete=False,
    )

    assert criteria["verifier_redesign_k_eff_above_3"] is True
    assert criteria["retro_96_complete"] is False
