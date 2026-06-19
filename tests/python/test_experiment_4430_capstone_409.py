"""Tests for Exp 4430 .409 milestone capstone.

Spec refs: REQ-CAPSTONE-4430, SCENARIO-CAPSTONE-4430.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v409_4430 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_support_files(root: Path) -> None:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "publication_gate.py").write_text("# fixture\n", encoding="utf-8")


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


def _gate(
    experiment: str,
    *,
    flagged: bool,
    new_levels: int,
    offline_reproduced: bool,
    reproduced_levels: int,
) -> JsonDict:
    return {
        "experiment": experiment,
        "artifact": f"results/experiment_{experiment.removeprefix('exp')}_fixture.json",
        "artifact_present": True,
        "artifact_flagged_adversarial": flagged,
        "honest_verdict": f"complete: {experiment}_fixture",
        "new_levels_counted": new_levels,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "reproduction_gated": True,
        "gate_evidence_keys": ["reproduced", "reached_level"],
    }


def _fixture_payloads(
    *,
    config_flagged: bool = True,
    config_new_level: bool = True,
    glyph_grounded: bool = True,
    glyph_solved: bool = True,
    first_contact_reproduced: bool = False,
    first_contact_new_game: bool = False,
    deepening_new_level: bool = False,
    vocab_flagged: bool = True,
    vocab_transfers: bool = False,
    audit_total: int = 36,
    audit_games: int = 18,
) -> dict[str, JsonDict]:
    return {
        "4421_config_rule": {
            "experiment": "experiment_4421_config_rule_solve_unseen",
            "honest_verdict": "success_s5i5_L1_offline_reproduced",
            "offline_reproduced": config_new_level,
            "reproduced_levels": 1 if config_new_level else 0,
            "new_levels_reproduced": 1 if config_new_level else 0,
            "target_game": "s5i5",
            "verifier_is_oracle": True,
            "flagged_adversarial": config_flagged,
            "random_seed": 4421,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4422_glyph": {
            "experiment": "experiment_4422_glyph_rewrite_perception",
            "honest_verdict": (
                "success_glyph_rewrite_perception_tr87_grounded_reproduced"
                if glyph_grounded and glyph_solved
                else "complete: glyph_fixture_null"
            ),
            "grounded": glyph_grounded,
            "fires_on_win": glyph_grounded,
            "false_positive_rate": 0.0 if glyph_grounded else 1.0,
            "offline_reproduced": glyph_solved,
            "reproduced_levels": 6 if glyph_solved else 0,
            "target_game": "tr87",
            "verifier_is_oracle": True,
            "inference_substrate": "offline_arc_agi3_glyph_rewrite_pixel_decode_cpu_no_llm",
            "random_seed": 4422,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4423_first_contact": {
            "experiment": "experiment_4423_generic_first_contact_breadth",
            "honest_verdict": (
                "success: generic_first_contact_added_game"
                if first_contact_reproduced
                else "partial: generic_first_contact_g50t_routed_missing_verifier_gap_logged"
            ),
            "target_game": "g50t",
            "offline_reproduced": first_contact_reproduced,
            "reproduced_levels": 1 if first_contact_reproduced else 0,
            "new_games_reproduced": 1 if first_contact_new_game else 0,
            "missing_verifier_gaps": []
            if first_contact_reproduced
            else [{"gap_id": "GAP-4423-G50T-UNSELECTABLE-FIRST-CONTACT"}],
            "verifier_is_oracle": False,
            "random_seed": 4423,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4424_deepening": {
            "experiment": "experiment_4424_deeper_solved_game",
            "honest_verdict": (
                "success: sc25_L2_reproduced"
                if deepening_new_level
                else "complete: sc25_L2_hud_cleanup_fixed_reproduction_gap"
            ),
            "game": "sc25",
            "offline_reproduced": deepening_new_level,
            "new_levels_reproduced": 1 if deepening_new_level else 0,
            "reproduced_levels": 2 if deepening_new_level else 1,
            "per_mechanic_test_pass_rate": 1.0 if deepening_new_level else 0.5,
            "residual_failing_mechanic": ""
            if deepening_new_level
            else "sc25_l2_route_search_still_missing_after_hud_cleanup",
            "verifier_is_oracle": True,
            "random_seed": 4424,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4425_vocabulary": {
            "experiment": "experiment_4425_config_rule_vocabulary_transfer",
            "honest_verdict": (
                "success: config_rule_vocabulary_transfers"
                if vocab_transfers
                else "complete: null_config_rule_vocabulary_transfer_seeded_arm_missing"
            ),
            "config_rule_vocabulary_transfers": vocab_transfers,
            "verifier_is_oracle": False,
            "flagged_adversarial": vocab_flagged,
            "random_seed": 4425,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4426_registry_audit": {
            "experiment": "experiment_4426_arc_registry_repro_audit",
            "honest_verdict": f"complete: registry_repro_audit_total_35_asserted_{audit_total}_audited",
            "reproducible_total_levels": audit_total,
            "registry_claimed_reproducible_total_levels": 35,
            "registry_claimed_reproducible_total_games": audit_games,
            "counted_entries_audited": audit_games,
            "all_counted_entries_reproduced": True,
            "milestone_409_reproduction_gates": [
                _gate(
                    "exp4421",
                    flagged=config_flagged,
                    new_levels=1 if config_new_level else 0,
                    offline_reproduced=config_new_level,
                    reproduced_levels=1 if config_new_level else 0,
                ),
                _gate(
                    "exp4422",
                    flagged=False,
                    new_levels=0,
                    offline_reproduced=glyph_solved,
                    reproduced_levels=6 if glyph_solved else 0,
                ),
                _gate(
                    "exp4423",
                    flagged=False,
                    new_levels=0,
                    offline_reproduced=first_contact_reproduced,
                    reproduced_levels=1 if first_contact_reproduced else 0,
                ),
                _gate(
                    "exp4424",
                    flagged=False,
                    new_levels=1 if deepening_new_level else 0,
                    offline_reproduced=deepening_new_level,
                    reproduced_levels=2 if deepening_new_level else 1,
                ),
            ],
            "inference_substrate": "offline_arc_registry_repro_audit_cpu_no_llm",
            "random_seed": 4426,
            "reproducibility_checksum": "sha256:" + "f" * 64,
        },
        "4429_sota_ingestion": {
            "honest_verdict": "complete: sota_ingestion_409_mapped",
            "flagged_for_v410": (
                "Executable ARC-AGI-3 world-model agent with verifier-grounded planning "
                "(arXiv:2605.05138)"
            ),
            "outcome_conditioning": {
                "generic_first_contact_partial": not first_contact_reproduced,
                "vocabulary_artifact_adversarial": vocab_flagged,
            },
            "preconditions_checked": {
                "trm_training_stood_down": True,
                "cpu_only": True,
                "deep_research_invoked": False,
            },
            "inference_substrate": "cpu_reliable_channel_sota_ingestion_no_training",
            "random_seed": 4429,
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4430_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4430: OpenSpec declares the .409 scorecard contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4430" in spec
    assert "SCENARIO-CAPSTONE-4430" in spec
    assert "results/experiment_4430_capstone_409.json" in spec
    assert "publication_gate.py --json" in spec
    assert "aggregate-available-report-gaps" in spec
    assert "CIRCULAR_MOAT_OVERCLAIM" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec


def test_scenario_capstone_4430_current_artifacts_answer_headline_questions() -> None:
    """SCENARIO-CAPSTONE-4430: current .409 artifacts produce the honest scorecard."""

    artifact = mod.build_artifact(
        Path.cwd(),
        started_s=1.0,
        now_s=1.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["reproducible_total_levels"] == 36
    assert artifact["new_levels"] == 2
    assert artifact["new_games"] == 1
    assert artifact["config_rule_unseen_state"] == (
        "direct_artifact_flagged_registry_audit_counted_execution_grounded"
    )
    assert artifact["glyph_rewrite_state"] == "grounded_and_offline_solved"
    assert artifact["generic_pipeline_state"] == "first_contact_verifier_gap_open_no_new_game"
    assert artifact["generic_first_contact_state"] == "verifier_gap_logged_no_new_game"
    assert artifact["multi_level_deepening_state"] == "mechanic_repair_no_new_level"
    assert artifact["config_rule_vocabulary_transfer_state"] == "excluded_flagged_adversarial"
    assert artifact["headline_question_answers"]["exp4421"]["execution_grounded"] is True
    assert artifact["headline_question_answers"]["exp4421"]["direct_artifact_imported"] is False
    assert artifact["headline_question_answers"]["exp4422"]["grounded"] is True
    assert artifact["headline_question_answers"]["exp4423"]["new_game_added"] is False
    assert artifact["headline_question_answers"]["exp4424"]["new_level_added"] is False
    assert artifact["headline_question_answers"]["exp4425"]["transferred"] is False
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["publication_gate"]["unmet_gates"] == []
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verifier_is_oracle_honored"] is True
    assert artifact["preconditions_checked"]["trm_training_stood_down"] is True
    assert artifact["preconditions_checked"]["robust_aggregate_available_helper"] == (
        "capstone_aggregate_available.aggregate_available_report_gaps"
    )
    assert "gated_on" not in artifact
    assert {row["experiment_id"] for row in artifact["flagged_artifacts_excluded"]} == {
        4421,
        4425,
    }
    assert 4421 not in {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}
    assert 4426 in {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]}
    assert artifact["capstone_live_adversarial_recheck"] == {"status": "not_run_until_write"}


def test_scenario_capstone_4430_positive_clean_fixture_records_all_branches(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4430: clean positive upstreams report positive axes."""

    _write_support_files(tmp_path)
    _write_default_artifacts(
        tmp_path,
        _fixture_payloads(
            config_flagged=False,
            first_contact_reproduced=True,
            first_contact_new_game=True,
            deepening_new_level=True,
            vocab_flagged=False,
            vocab_transfers=True,
            audit_total=38,
            audit_games=19,
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
    assert artifact["reproducible_total_levels"] == 38
    assert artifact["new_levels"] == 4
    assert artifact["new_games"] == 2
    assert artifact["config_rule_unseen_state"] == "clean_execution_grounded_reproduced_level"
    assert artifact["generic_pipeline_state"] == "first_contact_added_new_game"
    assert artifact["generic_first_contact_state"] == "new_game_added"
    assert artifact["multi_level_deepening_state"] == "new_level_added"
    assert artifact["config_rule_vocabulary_transfer_state"] == "transfers"
    assert artifact["config_rule_vocabulary_transfers"] is True
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["honest_verdict"].endswith("vocab_transfers_publication_ready")


def test_req_capstone_4430_missing_and_flagged_inputs_are_reported_per_axis(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4430: missing and flagged upstreams do not erase other axes."""

    _write_support_files(tmp_path)
    payloads = _fixture_payloads(audit_total=35, audit_games=18, vocab_flagged=False)
    payloads.pop("4422_glyph")
    payloads["4425_vocabulary"]["flagged_adversarial"] = True
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=5.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(False),
    )

    mod.validate_artifact(artifact)
    assert artifact["glyph_rewrite_state"] == "missing_or_excluded"
    assert artifact["generic_first_contact_state"] == "verifier_gap_logged_no_new_game"
    assert artifact["config_rule_vocabulary_transfer_state"] == "excluded_flagged_adversarial"
    assert {"axis": "glyph_rewrite", "artifact_key": "4422_glyph", "experiment_id": 4422} in (
        artifact["per_axis_gaps"]
    )
    assert artifact["availability_report"]["axes"]["vocabulary_transfer"]["flagged_artifacts"] == [
        {
            "axis": "vocabulary_transfer",
            "artifact_key": "4425_vocabulary",
            "experiment_id": 4425,
            "reason": "flagged_adversarial",
        }
    ]
    assert artifact["publication_gate"]["paper_ready"] is False
    assert artifact["honest_verdict"].endswith("vocab_skipped_publication_not_ready")


def test_req_capstone_4430_branch_helpers_keep_alternate_states_honest() -> None:
    """REQ-CAPSTONE-4430: alternate helper states stay explicit and bounded."""

    assert mod._gate_by_experiment("bad", "exp4421") == {}  # noqa: SLF001
    assert mod._gate_by_experiment([{"experiment": "other"}], "exp4421") == {}  # noqa: SLF001
    assert mod.registry_audit_read(None, True) == {"status": "excluded_flagged_adversarial"}
    assert mod.registry_audit_read(None, False) == {"status": "missing_or_excluded"}
    assert mod.config_rule_unseen_read(None, False, {}) == {
        "state": "missing_or_excluded",
        "direct_artifact_imported": False,
        "registry_audit_counted": False,
        "execution_grounded": False,
        "registry_gate": {},
        "verifier_is_oracle": None,
    }
    assert (
        mod.glyph_rewrite_read(
            {
                "grounded": True,
                "fires_on_win": True,
                "false_positive_rate": 0.0,
                "offline_reproduced": False,
            },
            False,
        )["state"]
        == "grounded_not_solved"
    )
    assert (
        mod.glyph_rewrite_read(
            {"grounded": False, "fires_on_win": False, "offline_reproduced": False},
            False,
        )["state"]
        == "not_grounded"
    )
    assert mod.generic_first_contact_read(None, False) == {"state": "missing_or_excluded"}
    assert (
        mod.generic_first_contact_read(
            {"offline_reproduced": False, "missing_verifier_gaps": []}, False
        )["state"]
        == "no_new_game"
    )
    assert mod.generic_pipeline_state("missing_or_excluded") == "first_contact_missing_or_excluded"
    assert mod.generic_pipeline_state("no_new_game") == "first_contact_no_new_game"
    assert mod.multi_level_deepening_read(None, False) == {"state": "missing_or_excluded"}
    assert (
        mod.multi_level_deepening_read(
            {"offline_reproduced": False, "new_levels_reproduced": 0},
            False,
        )["state"]
        == "no_new_level"
    )
    assert mod.sota_ingestion_read(None, False) == {"status": "missing_or_excluded"}
    assert (
        mod._cited_upstream_artifacts(  # noqa: SLF001
            [
                {"skipped": True, "fields_imported": ["x"]},
                {"skipped": False, "fields_imported": []},
            ]
        )
        == []
    )
    assert (
        mod._trm_stood_down(  # noqa: SLF001
            {"resource": "trm_training_stand_down", "available": True}
        )
        is True
    )
    assert mod._trm_stood_down({"note": "no TRM training invoked"}) is True  # noqa: SLF001
    assert mod._trm_stood_down({"outer": {"trm_training_stood_down": True}}) is True  # noqa: SLF001
    assert mod._trm_stood_down([{"no_trm_training": True}]) is True  # noqa: SLF001


def test_scenario_capstone_4430_write_artifact_records_clean_live_recheck(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4430: written capstone carries live adversarial re-check."""

    _write_support_files(tmp_path)
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


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"honest_verdict": "draft"}, "honest_verdict"),
        ({"reproducible_total_levels": "36"}, "reproducible_total_levels"),
        ({"new_levels": True}, "new_levels"),
        ({"new_games": "1"}, "new_games"),
        ({"generic_pipeline_state": ""}, "generic_pipeline_state"),
        ({"publication_gate": []}, "publication_gate"),
        ({"verifier_is_oracle": True}, "verifier_is_oracle"),
        ({"inference_substrate": "training"}, "inference_substrate"),
        ({"reproducibility_checksum": "bad"}, "reproducibility_checksum"),
        ({"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
    ],
)
def test_validate_artifact_rejects_schema_violations(
    tmp_path: Path,
    patch: JsonDict,
    message: str,
) -> None:
    """SCENARIO-CAPSTONE-4430: schema violations fail closed."""

    _write_support_files(tmp_path)
    _write_default_artifacts(tmp_path, _fixture_payloads())
    artifact = mod.build_artifact(
        tmp_path,
        started_s=8.0,
        now_s=9.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    artifact.update(patch)

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)
