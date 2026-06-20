"""Tests for Exp 4478 .413 milestone capstone.

Spec refs: REQ-CAPSTONE-4478, SCENARIO-CAPSTONE-4478.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v413_4478 as mod


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


def _fixture_payloads(*, sc25_operator_flagged: bool = False) -> dict[str, JsonDict]:
    return {
        "4467_dc22": {
            "experiment": "experiment_4467_solve_dc22_cegis_nocov",
            "honest_verdict": "success: dc22_cegis_L1_offline_reproduced",
            "target_game": "dc22",
            "dc22_grounded": True,
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "reproducible_total_levels": 40,
            "reproducible_total_games": 21,
            "missing_verifier_gaps": [],
            "flagged_adversarial": False,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4468_sc25_deep": {
            "experiment": "experiment_4468_bank_sc25_provisional_levels",
            "honest_verdict": "success: sc25_L5_offline_reproduced_banked_4_new_levels",
            "target_game": "sc25",
            "offline_reproduced": True,
            "reproduced_levels": 4,
            "new_sc25_levels_reproduced": 4,
            "prior_sc25_levels_reproduced": 1,
            "sc25_levels_reproduced_total": 5,
            "missing_verifier_gaps": [],
            "flagged_adversarial": False,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4469_sc25_operator": {
            "experiment": "experiment_4469_generic_cast_grid_fsm_operator",
            "honest_verdict": "success: sc25_generic_cast_grid_fsm_L1_offline_reproduced",
            "target_game": "sc25",
            "offline_reproduced": True,
            "sc25_resolved_generically": True,
            "sc25_generic_level_reproduced": 1,
            "missing_verifier_gaps": [],
            "no_regression": True,
            "flagged_adversarial": sc25_operator_flagged,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4470_sb26": {
            "experiment": "experiment_4470_color_match_slot_operator_solve_sb26",
            "honest_verdict": "success: sb26_color_match_slot_sequence_L1_offline_reproduced",
            "target_game": "sb26",
            "color_match_operator_built": True,
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "missing_verifier_gaps": [],
            "flagged_adversarial": False,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4471_first_contact": {
            "experiment": "experiment_4471_first_contact_rotated_new_game",
            "honest_verdict": "complete: generic_first_contact_re86_routed_no_new_level",
            "target_game": "re86",
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "missing_verifier_gaps": [
                {
                    "game": "re86",
                    "gap_id": "GAP-4471-RE86-MISSING-PATTERN-MATCH-SPRITE-RESIZE-VERIFIER",
                    "operator": "graph_astar_action_cost",
                    "residual_delta": "missing_pattern_match_sprite_resize_verifier",
                    "routed_to": "tu93",
                    "status": "open",
                }
            ],
            "submitted_to_leaderboard": False,
            "flagged_adversarial": False,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4472_variant_loo_v4": {
            "experiment": "experiment_4472_variant_generic_transfer_benchmark_v4",
            "honest_verdict": "complete: blocked_baseline_smoke",
            "generic_loo_solve_count_v3_baseline": 6,
            "generic_loo_solve_count_v4": 0,
            "generic_transfer_rate_over_variants": 0.0,
            "variants_attempted": 0,
            "variants_solved": 0,
            "closed_residuals_by_413_operator": [],
            "missing_verifier_gaps": [],
            "flagged_adversarial": False,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "f" * 64,
        },
        "4473_submission": {
            "experiment": "experiment_4473_submission_package_prep_refresh",
            "honest_verdict": (
                "success: submission_package_ready_45_levels_beats_13_grew_vs_412_quarantined_0"
            ),
            "submission_package_ready": True,
            "total_reproduced_levels_in_package": 45,
            "prior_submitted_baseline_levels": 13,
            "prior_package_412_levels": 39,
            "beats_prior_baseline": True,
            "grew_vs_412": True,
            "submitted_to_leaderboard": False,
            "quarantined_games": [],
            "flagged_adversarial": False,
            "verifier_is_oracle": True,
            "reproducibility_checksum": "sha256:" + "0" * 64,
        },
        "4474_hygiene": {
            "experiment": "experiment_4474_registry_gaps_hygiene",
            "honest_verdict": "complete: registry_gaps_hygiene_4474_guard_passed",
            "reproducible_total_games": 22,
            "reproducible_total_levels": 45,
            "regression_guard_passed": True,
            "open_gap_ids": ["GAP-4471-RE86-MISSING-PATTERN-MATCH-SPRITE-RESIZE-VERIFIER"],
            "registry_reconciliation": {
                "filled_gap_ids": [
                    "GAP-4423-DC22-UNSELECTABLE-FIRST-CONTACT",
                    "GAP-4432-LOO-SC25-MISSING-CAST-GRID-SPELL-SHRINK-TANK-EXIT-VERIFIER",
                    "GAP-4458-SB26-COLOR-MATCH-SLOT-SEQUENCE",
                ],
                "open_gap_ids": [
                    "GAP-4471-RE86-MISSING-PATTERN-MATCH-SPRITE-RESIZE-VERIFIER"
                ],
            },
            "availability_report": {"available_artifact_keys": []},
            "flagged_adversarial": False,
            "submitted_to_leaderboard": False,
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4478_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4478: OpenSpec declares the .413 capstone contract first."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4478" in spec
    assert "SCENARIO-CAPSTONE-4478" in spec
    assert "results/experiment_4478_capstone_v413.json" in spec
    assert "publication_gate.py --json" in spec
    assert "aggregate-available-report-gaps" in spec
    assert "CIRCULAR_MOAT_OVERCLAIM" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_capstone_4478_current_artifacts_answer_headline_questions() -> None:
    """SCENARIO-CAPSTONE-4478: current .413 artifacts produce the honest scorecard."""

    artifact = mod.build_artifact(
        Path.cwd(),
        started_s=1.0,
        now_s=1.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducible_total_levels_grew"] is True
    assert artifact["generic_solver_gap_state"] == "partial"
    assert artifact["generic_loo_solve_count_v4"] == 0
    assert artifact["generic_transfer_rate_over_variants"] == 0.0
    assert artifact["generic_loo_v4"]["v4_rose_above_baseline"] is False
    assert artifact["reproducible_total_levels"] == 45
    assert artifact["reproducible_total_games"] == 22
    assert artifact["submission_package_ready"] is True
    assert artifact["submission_readiness_decision"] == "ready_beats_412_flat_metric"
    assert artifact["headline_question_answers"]["exp4467"]["dc22_l1_cleanly_banked"] is True
    assert artifact["headline_question_answers"]["exp4468"]["new_sc25_level_banked"] is True
    assert artifact["headline_question_answers"]["exp4469"]["gap_closed"] is True
    assert artifact["headline_question_answers"]["exp4470"]["sb26_banked"] is True
    assert artifact["headline_question_answers"]["exp4471"]["banked_new_rotated_game"] is False
    assert artifact["headline_question_answers"]["exp4472"]["generic_loo_solve_count_v4"] == 0
    assert artifact["headline_question_answers"]["exp4472"]["v4_rose_above_baseline"] is False
    assert artifact["headline_question_answers"]["exp4473"]["submission_package_ready"] is True
    assert artifact["headline_question_answers"]["exp4473"]["total_reproduced_levels_in_package"] == 45
    assert artifact["publication_gate"]["paper_ready"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verifier_is_oracle_honored"] is True
    assert artifact["next_backlog"]["open_gap_ids"] == [
        "GAP-4471-RE86-MISSING-PATTERN-MATCH-SPRITE-RESIZE-VERIFIER"
    ]
    assert artifact["next_backlog"]["missing_primitives"] == [
        "pattern_match_sprite_resize_verifier"
    ]
    assert artifact["next_backlog"]["operational_residuals"] == [
        "variant_transfer_baseline_smoke_failed"
    ]
    assert "gated_on" not in artifact
    assert artifact["submitted_to_leaderboard"] is False
    assert {row["experiment_id"] for row in artifact["cited_upstream_artifacts"]} == {
        4467,
        4468,
        4469,
        4470,
        4471,
        4472,
        4473,
        4474,
    }


def test_req_capstone_4478_missing_and_flagged_inputs_are_per_axis_gaps(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4478: missing/flagged upstreams do not erase unrelated axes."""

    _write_support_files(tmp_path)
    payloads = _fixture_payloads(sc25_operator_flagged=True)
    payloads.pop("4473_submission")
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=3.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(False),
    )

    mod.validate_artifact(artifact)
    assert artifact["reproducible_total_levels"] == 45
    assert artifact["dc22_bank"]["dc22_l1_cleanly_banked"] is True
    assert artifact["sc25_deeper_bank"]["new_sc25_level_banked"] is True
    assert artifact["sc25_generic_operator"]["state"] == "excluded_flagged_adversarial"
    assert artifact["submission_package"]["state"] == "missing_or_excluded"
    assert {"axis": "submission_package", "artifact_key": "4473_submission", "experiment_id": 4473} in (
        artifact["per_axis_gaps"]
    )
    assert artifact["availability_report"]["axes"]["sc25_generic_operator"][
        "flagged_artifacts"
    ] == [
        {
            "axis": "sc25_generic_operator",
            "artifact_key": "4469_sc25_operator",
            "experiment_id": 4469,
            "reason": "flagged_adversarial",
        }
    ]
    flagged_cite = next(
        row for row in artifact["cited_upstream_artifacts"] if row["experiment_id"] == 4469
    )
    assert flagged_cite["fields_imported"] == []
    assert artifact["publication_gate"]["paper_ready"] is False
    assert artifact["honest_verdict"].endswith("publication_not_ready")


def test_req_capstone_4478_branch_helpers_keep_alternate_states_honest() -> None:
    """REQ-CAPSTONE-4478: alternate helper branches stay explicit and bounded."""

    assert mod.dc22_read(None, False)["state"] == "missing_or_excluded"
    assert mod.dc22_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.sc25_deep_read(None, False)["state"] == "missing_or_excluded"
    assert mod.sc25_deep_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.sc25_operator_read(None, False)["state"] == "missing_or_excluded"
    assert mod.sc25_operator_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.sb26_read(None, False)["state"] == "missing_or_excluded"
    assert mod.sb26_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.rotated_first_contact_read(None, False)["state"] == "missing_or_excluded"
    assert mod.rotated_first_contact_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.variant_transfer_read(None, False)["state"] == "missing_or_excluded"
    assert mod.variant_transfer_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.submission_read(None, False)["state"] == "missing_or_excluded"
    assert mod.submission_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.hygiene_read(None, False)["state"] == "missing_or_excluded"
    assert mod.hygiene_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod._publication_gate_or_gap(Path("/tmp/definitely_missing_v413"), lambda _: {})[2] == [  # noqa: SLF001
        {"axis": "publication_gate", "artifact_key": "publication_gate", "reason": "unrunnable"}
    ]
    assert (
        mod.sc25_operator_read(
            {
                "target_game": "sc25",
                "sc25_resolved_generically": True,
                "offline_reproduced": True,
                "sc25_generic_level_reproduced": 1,
            },
            False,
        )["state"]
        == "sc25_generic_gap_closed"
    )
    assert (
        mod.sb26_read(
            {
                "target_game": "sb26",
                "color_match_operator_built": True,
                "offline_reproduced": True,
                "reproduced_levels": 1,
            },
            False,
        )["state"]
        == "sb26_banked"
    )
    assert (
        mod.sb26_read(
            {
                "target_game": "sb26",
                "selected_operator": {"operator": "color_match_slot_sequence_verifier"},
                "offline_reproduced": True,
                "reproduced_levels": 1,
            },
            False,
        )["color_match_operator_built"]
        is True
    )
    assert (
        mod.sb26_read(
            {
                "target_game": "sb26",
                "operator_result": {"operator": "color_match_slot_sequence_verifier"},
                "offline_reproduced": True,
                "reproduced_levels": 1,
            },
            False,
        )["state"]
        == "sb26_banked"
    )
    assert mod.hygiene_read(
        {
            "reproducible_total_levels": 1,
            "reproducible_total_games": 1,
            "registry_reconciliation": {
                "filled_gap_ids": ["GAP-F"],
                "open_gap_ids": ["GAP-X"],
            },
        },
        False,
    )["open_gap_ids"] == ["GAP-X"]
    assert mod.hygiene_read(
        {
            "reproducible_total_levels": 1,
            "reproducible_total_games": 1,
            "registry_reconciliation": {"filled_gap_ids": ["GAP-F"]},
        },
        False,
    )["filled_gap_ids"] == ["GAP-F"]
    assert mod._missing_primitive_name("missing_example") == "example"  # noqa: SLF001
    assert mod._residual_rows({"missing_verifier_gaps": [object(), {"residual_delta": "none"}]}) == []  # noqa: SLF001
    assert mod._residual_rows(  # noqa: SLF001
        {
            "missing_verifier_gaps": ["missing_alpha", "missing_alpha"],
            "per_game": [
                {
                    "game": "g1",
                    "gap_id": "GAP-1",
                    "residual_delta": "missing_beta",
                    "status": "open",
                }
            ],
        },
        default_game="dg",
    ) == [
        {"game": "dg", "residual_delta": "missing_alpha", "status": "open"},
        {
            "game": "g1",
            "gap_id": "GAP-1",
            "residual_delta": "missing_beta",
            "status": "open",
        },
    ]
    assert mod.decide_generic_solver_gap_state(
        dc22={"dc22_l1_cleanly_banked": True},
        sc25_deep={"new_sc25_level_banked": True},
        sc25_operator={"gap_closed": True},
        sb26={"sb26_banked": True},
        first_contact={"banked_new_rotated_game": True},
        variant_transfer={"v4_rose_above_baseline": True},
        next_backlog={"residual_deltas": []},
    ) == "closing"
    assert mod.decide_generic_solver_gap_state(
        dc22={},
        sc25_deep={},
        sc25_operator={},
        sb26={},
        first_contact={},
        variant_transfer={},
        next_backlog={"residual_deltas": []},
    ) == "total-gap"
    backlog = mod.build_next_backlog(
        dc22={"residual_deltas": [object(), {"game": "x", "residual_delta": ""}]},
        sc25_deep={"residual_deltas": [{"game": "x", "residual_delta": "missing_y"}]},
        sc25_operator={"residual_deltas": [{"game": "x", "residual_delta": "missing_y"}]},
        sb26={"residual_deltas": []},
        first_contact={"residual_deltas": []},
        variant_transfer={"state": "v4_not_above_baseline"},
        hygiene={"filled_gap_ids": ["GAP-F"], "open_gap_ids": ["GAP-O"]},
    )
    assert backlog["residual_deltas"] == [{"game": "x", "residual_delta": "missing_y"}]
    assert backlog["missing_primitives"] == ["y"]
    assert backlog["operational_residuals"] == []


def test_req_capstone_4478_unparseable_input_is_excluded(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4478: unparsable upstreams are excluded before import."""

    bad_path = tmp_path / mod.DEFAULT_UPSTREAMS["4467_dc22"].path
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not-json", encoding="utf-8")

    raw, provenance, exclusions = mod._read_inputs(  # noqa: SLF001
        tmp_path,
        _clean_live_flags,
        _summarize_zero,
    )

    assert raw["4467_dc22"] is None
    assert provenance[0]["parse_error"].startswith("JSONDecodeError")
    assert provenance[0]["fields_imported"] == []
    assert exclusions[0]["reason"] == "unparsable_or_non_object"


def test_req_capstone_4478_write_path_and_validation_guards(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4478: write path validates and records capstone re-checks."""

    _write_support_files(tmp_path)
    _write_default_artifacts(tmp_path, _fixture_payloads())

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=4.0,
        now_s=5.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
        capstone_live_flag_runner=lambda _: [
            {"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}
        ],
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["capstone_live_adversarial_recheck"] == {
        "circular_moat_overclaim": True,
        "flags": [{"kind": "CIRCULAR_MOAT_OVERCLAIM", "severity": "critical"}],
        "status": "critical_flags",
    }

    valid = mod.build_artifact(
        tmp_path,
        started_s=6.0,
        now_s=7.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
        publication_gate_runner=lambda _: _publication_gate(True),
    )
    mutations = [
        ("__delete__honest_verdict", None, "missing required field"),
        ("honest_verdict", "blocked", "honest_verdict"),
        ("reproducible_total_levels_grew", "yes", "reproducible_total_levels_grew"),
        ("generic_solver_gap_state", "unknown", "generic_solver_gap_state"),
        ("generic_loo_solve_count_v4", True, "generic_loo_solve_count_v4"),
        ("generic_transfer_rate_over_variants", "0", "generic_transfer_rate_over_variants"),
        ("submission_package_ready", "yes", "submission_package_ready"),
        ("next_backlog", [], "next_backlog"),
        ("publication_gate", [], "publication_gate"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("cited_upstream_artifacts", {}, "cited_upstream_artifacts"),
        ("random_seed", 1, "random_seed"),
        ("inference_substrate", "other", "inference_substrate"),
        ("reproducibility_checksum", "sha256:not-a-real-sha", "reproducibility_checksum"),
        ("field_principles", {}, "field_principles"),
        ("upstream_provenance", {}, "upstream_provenance"),
        ("upstream_provenance", [1], "upstream provenance row"),
        ("upstream_provenance", [{"sha256": "bad"}], "invalid sha256"),
        (
            "upstream_provenance",
            [{"sha256": "1" * 64, "skipped": True, "fields_imported": ["x"]}],
            "skipped upstreams",
        ),
        ("__gated_on__", True, "gated_on"),
        ("__checksum_mismatch__", True, "reproducibility_checksum"),
    ]
    for field, value, message in mutations:
        invalid = json.loads(json.dumps(valid))
        if field == "__delete__honest_verdict":
            invalid.pop("honest_verdict")
        elif field == "__gated_on__":
            invalid["gated_on"] = value
        elif field == "__checksum_mismatch__":
            invalid["reproducibility_checksum"] = "sha256:" + "1" * 64
        else:
            invalid[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(invalid)
