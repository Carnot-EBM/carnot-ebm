"""Tests for Exp 4412 .407 milestone capstone.

Spec refs: REQ-CAPSTONE-4412, SCENARIO-CAPSTONE-4412.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import capstone_v407_4412 as mod


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
            "G1": {"pass": True, "detail": "fixture"},
            "G2": {"pass": paper_ready, "detail": "fixture"},
            "G3": {"pass": True, "detail": "fixture"},
            "G4": {"pass": True, "detail": "fixture"},
        },
        "unmet_gates": [] if paper_ready else ["G2"],
    }


def _fixture_payloads(
    *,
    genuine: bool = False,
    generalizes: bool = False,
    compounds: bool = False,
    calibrated: bool = False,
    e3_new_levels: int = 0,
) -> dict[str, JsonDict]:
    return {
        "4403_localizer": {
            "honest_verdict": (
                "success: localizer_genuinely_beats_position_only"
                if genuine
                else "complete: clean_powered_null_position_only_not_beaten"
            ),
            "localizer_genuinely_beats_position_only": genuine,
            "beats_position_only_baseline": genuine,
            "position_only_baseline_f1": 0.42 if genuine else 1.0,
            "localization_f1_by_domain": {
                "fover": {
                    "domain": "fover",
                    "position_only_baseline_f1": 0.42 if genuine else 1.0,
                    "real_intervention_localizer_f1": 0.73 if genuine else 1.0,
                    "delta_vs_position_only": 0.31 if genuine else 0.0,
                }
            },
            "template_family_holdout_drop": 0.01,
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4404_typed_generalization": {
            "honest_verdict": (
                "success: localizer_generalizes_typed"
                if generalizes
                else "blocked_gate_check_failed"
            ),
            "status": "success" if generalizes else "blocked",
            "localizer_generalizes_typed": generalizes,
            "typed_taxonomy_agreement_above_chance": generalizes,
            "non_fover_domain_delta_ci95": [0.08, 0.19] if generalizes else [0.0, 0.0],
            "gate_check_summary": "fixture",
            "gates_evaluated": [{"passed": generalizes}],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4405_e3_deeper": {
            "honest_verdict": "complete_e3_deeper_partial",
            "new_levels_reproduced": e3_new_levels,
            "reproducible_total_levels": 34 + e3_new_levels,
            "per_target_scorecard": [
                {
                    "game": "tn36",
                    "new_reproduced_level": 8,
                    "offline_reproduced": e3_new_levels > 0,
                    "prior_best_level": 7,
                    "residual_win_mechanic_gap_class": "fixture_gap",
                }
            ],
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4406_e3_blocked": {
            "honest_verdict": "complete_e3_ar25_ka59_ft09_partial",
            "new_levels_reproduced": 0,
            "reproducible_total_levels": 34,
            "per_game_scorecard": [
                {
                    "game": "ar25",
                    "new_reproduced_level": 1,
                    "offline_reproduced": False,
                    "prior_best_level": 1,
                    "residual_gap_class": "fixture_gap",
                }
            ],
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4407_compounds": {
            "honest_verdict": (
                "success: localizer_compounds_active_selection"
                if compounds
                else "complete: clean_null_position_bound_or_saturated"
            ),
            "localizer_compounds": compounds,
            "active_vs_random_learning_curve": [
                {"train_corpus_size": 512, "active_f1": 0.41, "random_f1": 0.4},
                {"train_corpus_size": 4096, "active_f1": 0.56 if compounds else 0.41, "random_f1": 0.4},
            ],
            "compounding_delta_ci95": [0.03, 0.12] if compounds else [0.0, 0.0],
            "gate_summary": {
                "localizer_compounds": compounds,
                "position_only_control_beaten": compounds,
                "positive_control_headroom": compounds,
            },
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4408_calibration": {
            "honest_verdict": (
                "success: calibrated_multi_domain_contract_deconfounded"
                if calibrated
                else "complete: calibrated_multi_domain_contract_false_deconfounded"
            ),
            "detection_calibrated_multi_domain": calibrated,
            "detection_by_domain": [
                {
                    "domain": "fover",
                    "detection_auroc": 0.918304,
                    "auroc_ci95": [0.90944, 0.926582],
                    "ece_lodo_calibrated": 0.02 if calibrated else 0.134538,
                    "n": 8829,
                },
                {
                    "domain": "gap4_arc",
                    "detection_auroc": 0.963006,
                    "auroc_ci95": [0.921708, 0.9912],
                    "ece_lodo_calibrated": 0.01 if calibrated else 0.0055,
                    "n": 28443,
                },
            ],
            "domains_at_chance": [] if calibrated else ["code_humaneval"],
            "verifier_is_oracle": False,
            "reproducibility_checksum": "sha256:" + "f" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4412_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4412: OpenSpec declares the .407 scorecard contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4412" in spec
    assert "SCENARIO-CAPSTONE-4412" in spec
    assert "experiment_4412_capstone_v407.json" in spec
    assert "publication_gate.py --json" in spec
    assert "CIRCULAR_MOAT_OVERCLAIM" in spec
    assert "aggregation_from_upstream_artifacts" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_capstone_4412_current_artifacts_report_headline_decision() -> None:
    """SCENARIO-CAPSTONE-4412: current .407 artifacts report the honest scorecard."""

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
        "complete: v407_localizer_position_bound_retired_compounds_false_"
        "calibrated_false_arc_levels_34_publication_ready"
    )
    assert artifact["localizer_state"] == "position_bound_retired"
    assert artifact["localizer"]["real_intervention"]["localizer_genuinely_beats_position_only"] is False
    assert artifact["localizer"]["real_intervention"]["position_only_baseline_f1"] == pytest.approx(1.0)
    assert artifact["localizer"]["typed_generalization"]["status"] == "not_generalized"
    assert "actual=False" in artifact["localizer"]["typed_generalization"]["gate_check_summary"]
    assert artifact["localizer_compounds"] is False
    assert artifact["self_learning"]["compounding_delta_ci95"] == pytest.approx([0.0, 0.0])
    assert artifact["detection_calibrated_multi_domain"] is False
    assert artifact["calibration"]["domains_at_chance"] == ["code_humaneval"]
    assert artifact["reproducible_total_levels"] == 34
    assert artifact["arc_e3_outcomes"]["execution_grounded"] is True
    assert artifact["arc_e3_outcomes"]["new_levels_reproduced_from_artifacts"] == 0
    assert artifact["verifier_thesis_state"] == (
        "localizer_position_bound_retired_localizer_compounding_open_"
        "detection_not_calibrated_multi_domain_arc_progress_34"
    )
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["publication_gate"]["unmet_gates"] == []
    assert all(gate["pass"] is True for gate in artifact["publication_gate"]["gates"].values())
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verifier_is_oracle_honored"] is True
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]} == {
        4403,
        4404,
        4405,
        4406,
        4407,
        4408,
    }
    assert all("sha256" in row and row["fields_imported"] for row in artifact["cited_upstream_artifacts"])


def test_scenario_capstone_4412_positive_fixture_graduates_and_records_gate_gap(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4412: genuine plus typed generalization graduates localizer."""

    _write_support_files(tmp_path, total=38, games=17)
    _write_default_artifacts(
        tmp_path,
        _fixture_payloads(genuine=True, generalizes=True, compounds=True, calibrated=True, e3_new_levels=2),
    )

    def broken_publication_gate(_: Path) -> JsonDict:
        raise RuntimeError("fixture gate down")

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=3.25,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=broken_publication_gate,
    )

    mod.validate_artifact(artifact)
    assert artifact["localizer_state"] == "localizes_genuine_cross_domain"
    assert artifact["localizer_compounds"] is True
    assert artifact["detection_calibrated_multi_domain"] is True
    assert artifact["reproducible_total_levels"] == 38
    assert artifact["arc_reproducible_progress"]["new_levels_since_prior"] == 4
    assert artifact["publication_gate"]["paper_ready"] is False
    assert artifact["publication_gate"]["unmet_gates"] == ["publication_gate_unrunnable"]
    assert artifact["preconditions_checked"]["publication_gate"]["runnable"] is False
    assert artifact["per_axis_gaps"] == [
        {
            "axis": "publication_gate",
            "artifact_key": "publication_gate",
            "reason": "unrunnable",
        }
    ]
    assert artifact["honest_verdict"].endswith("arc_levels_38_publication_gate_gap")


def test_req_capstone_4412_flagged_inputs_are_skipped_and_not_cited(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4412: flagged upstreams are excluded before metrics are imported."""

    _write_support_files(tmp_path, total=35, games=17)
    payloads = _fixture_payloads(genuine=True, generalizes=True, compounds=True, calibrated=True)
    payloads["4407_compounds"]["flagged_adversarial"] = True
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=5.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["localizer_state"] == "localizes_genuine_cross_domain"
    assert artifact["localizer_compounds"] is False
    assert artifact["self_learning"]["status"] == "excluded_flagged_adversarial"
    assert artifact["flagged_artifacts_excluded"] == [
        {
            "artifact_key": "4407_compounds",
            "experiment_id": 4407,
            "path": "results/experiment_4407_active_learning_self_learning_compounds.json",
            "sha256": artifact["upstream_provenance"][4]["sha256"],
            "stamped_flagged_adversarial": True,
            "live_critical": False,
            "parse_error": "",
            "live_critical_flags": [],
            "reason": "flagged_adversarial",
        }
    ]
    assert 4407 not in {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}
    assert artifact["availability_report"]["axes"]["self_learning"]["flagged_artifacts"] == [
        {
            "axis": "self_learning",
            "artifact_key": "4407_compounds",
            "experiment_id": 4407,
            "reason": "flagged_adversarial",
        }
    ]


def test_scenario_capstone_4412_write_artifact_records_clean_live_recheck(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4412: written capstone carries the live adversarial re-check."""

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
