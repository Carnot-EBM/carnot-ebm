"""Tests for the Exp 1255 combined .95/.96/.97 retrospective.

Spec: REQ-REPORT-021, SCENARIO-REPORT-018.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting.combined_retro_95_96_97 import (
    CRITERIA_95_NAMES,
    CRITERIA_96_NAMES,
    CRITERIA_97_NAMES,
    build_artifact,
    run,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scenario_sources() -> dict[int, dict[str, object]]:
    return {
        1216: {"staged_files_only_disabled": True, "precommit_fail_forward_enabled": True},
        1217: {"honest_verdict": "blocked_gate_check_failed"},
        1218: {"all_5_citations_added": True, "novelty_boundary_applied": True},
        1219: {"diagnosis_complete": True},
        1220: {"beats_v4_floor": True},
        1221: {"grpo_v6_fspo_delta_measured": False},
        1222: {"phase5a_prototype_ready": True},
        1223: {"phase5b_stability_confirmed": True},
        1224: {"adversarial_probe_complete": True},
        1225: {"gaming_defense_measured": False},
        1226: {"boltzmann_gpt_auroc_measured": True},
        1227: {"futoshiki_cartridge_shipped": True},
        1229: {"retro_complete": False},
        1230: {"autofill_script_exists": True},
        1231: {"gaming_defense_measured": False},
        1232: {"pairwise_correlation_matrix_measured": False, "k_eff": None},
        1235: {"grpo_v6_improvement_measured": False},
        1237: {"boltzmann_gpt_contrastive_auroc": None},
        1238: {"gates_measured": 0},
        1239: {"frozen_prefix_regime_classified": False},
        1240: {"honest_verdict": "blocked_gate_check_failed"},
        1242: {"criteria_96_met": 0, "criteria_95_met": 0, "retro_complete": False},
        1248: {"post_cd_auroc": 0.9607438016528925},
        1251: {
            "nonmonotonicity_characterized": True,
            "nonmonotonicity_classification": "b_causal_context_shift",
        },
        1254: {"retro_complete": False},
    }


def test_scenario_report_018_counts_stale_retros_and_source_fields() -> None:
    """SCENARIO-REPORT-018: Exp1255 reports .97 4/13, .96 2/13, .95 10/13."""

    artifact = build_artifact(_scenario_sources())

    assert tuple(artifact["criteria_97_results"]) == CRITERIA_97_NAMES
    assert tuple(artifact["criteria_96_results"]) == CRITERIA_96_NAMES
    assert tuple(artifact["criteria_95_results"]) == CRITERIA_95_NAMES
    assert artifact["criteria_97_met"] == 4
    assert artifact["criteria_96_met"] == 2
    assert artifact["criteria_95_met"] == 10
    assert artifact["criteria_97_results"]["retro_96_complete"] is False
    assert artifact["criteria_96_results"]["retro_96_complete"] is True
    assert artifact["criteria_95_results"]["retro_95_complete"] is True
    assert "AUROC=0.96" in artifact["findings_summary"]
    assert "NRGPT Type-B" in artifact["findings_summary"]
    assert len(artifact["key_carry_forwards"]) == 5
    assert artifact["retro_complete"] is True
    assert artifact["honest_verdict"] == "milestone_97_4_of_13_criteria_met"


def test_req_report_021_run_writes_required_schema(tmp_path: Path) -> None:
    """REQ-REPORT-021: run loads result JSON files and writes the final artifact."""

    results_dir = tmp_path / "results"
    out_path = results_dir / "experiment_1255_combined_retro_95_96_97.json"
    for exp_id, payload in _scenario_sources().items():
        filename = {
            1216: "experiment_1216_precommit_staged_files_only_fix.json",
            1217: "experiment_1217_auto_populate_prior_failures.json",
            1218: "experiment_1218_paper_v6_related_work_overhaul.json",
            1219: "experiment_1219_grpo_v5_regression_diagnosis.json",
            1220: "experiment_1220_grpo_vps_full_training.json",
            1221: "experiment_1221_grpo_v6_fspo_vps_combined.json",
            1222: "experiment_1222_phase5a_insitu_prototype.json",
            1223: "experiment_1223_phase5b_insitu_training_loop.json",
            1224: "experiment_1224_phase5c_adversarial_probe.json",
            1225: "experiment_1225_llms_gaming_verifiers_defense.json",
            1226: "experiment_1226_boltzmann_gpt_phase3_seed.json",
            1227: "experiment_1227_wopr_futoshiki_cartridge.json",
            1229: "experiment_1229_milestone_retro_95.json",
            1230: "experiment_1230_auto_populate_prior_failures_v2.json",
            1231: "experiment_1231_llms_gaming_verifiers_defense.json",
            1232: "experiment_1232_verifier_joint_orthogonality_audit.json",
            1235: "experiment_1235_grpo_v6_fspo_vps_extended.json",
            1237: "experiment_1237_boltzmann_gpt_contrastive_training.json",
            1238: "experiment_1238_phase5d_intermediate_scale.json",
            1239: "experiment_1239_nrgpt_frozen_prefix_evaluation.json",
            1240: "experiment_1240_wopr_kakuro_cartridge.json",
            1242: "experiment_1242_combined_retro_95_96.json",
            1248: "experiment_1248_boltzmann_gpt_cd_training_v2.json",
            1251: "experiment_1251_nrgpt_frozen_prefix_evaluation_v2.json",
            1254: "experiment_1254_milestone_retro_97.json",
        }[exp_id]
        _write_json(results_dir / filename, payload)

    artifact = run(results_dir=results_dir, out_path=out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))

    assert artifact == written
    assert written["schema"] == "milestone_retro_combined_v2"
    assert written["status"] == "complete"
    assert written["criteria_97_total"] == 13
    assert written["criteria_96_total"] == 13
    assert written["criteria_95_total"] == 13
