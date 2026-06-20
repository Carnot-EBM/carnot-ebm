"""Tests for Exp 4486 .414 milestone capstone.

Spec refs: REQ-CAPSTONE-4486, SCENARIO-CAPSTONE-4486.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v414_4486 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_live_flags(_: Path) -> list[dict[str, str]]:
    return []


def _summarize_zero(_: Path, __: Path) -> int:
    return 0


def _fixture_payloads(*, a2_flagged: bool = False) -> dict[str, JsonDict]:
    return {
        "4475_a1_stack": {
            "experiment": "experiment_4475_wire_stronger_generic_stack",
            "honest_verdict": "complete: submitted_default_stronger_generic_stack_wired",
            "before_generic_solve_rate": 1 / 7,
            "after_generic_solve_rate": 1 / 7,
            "generic_solve_rate_delta": 0.0,
            "before_solved": 1,
            "after_solved": 1,
            "attempted_games": 7,
            "benchmark": {"measurement": "heldout_loo_generic_set_exact_submitted_default"},
            "offline_reproduced": True,
            "reproduced_levels": 45,
            "preconditions_checked": {"offline_arcade_available": True},
            "reproducibility_checksum": "sha256:" + "a" * 64,
        },
        "4476_a2_features": {
            "experiment": "experiment_4476_verifier_features_v3_loo_gate",
            "honest_verdict": "success: cross_game_features_v3_loo_auroc_0.674_passes_gate",
            "v2_baseline_loo_auroc": 0.503096152732577,
            "v3_loo_auroc": 0.6744657162333668,
            "v3_in_sample_auroc": 0.8710834214701216,
            "target_loo_auroc": 0.6,
            "loo_gate_passed": True,
            "feature_class_deltas": {"v3_full": 0.1713695635007898},
            "feature_class_loo_auroc": {"v3_full": 0.6744657162333668},
            "offline_reproduced": True,
            "reproduced_levels": 51,
            "preconditions_checked": {"offline_arcade": True, "seed": 0},
            "flagged_adversarial": a2_flagged,
            "reproducibility_checksum": "sha256:" + "b" * 64,
        },
        "4477_a3_routing": {
            "experiment": "experiment_4477_per_game_online_discriminative",
            "honest_verdict": "complete: per_game_online_discriminative_no_solve_rate_gain",
            "baseline_solve_rate": 1 / 3,
            "online_solve_rate": 1 / 3,
            "solve_rate_delta": 0.0,
            "baseline_actions_to_first_levelup": 58,
            "online_actions_to_first_levelup": 58,
            "actions_to_first_levelup_delta": 0,
            "per_game_results": [{"game": "bp35"}, {"game": "lp85"}, {"game": "tu93"}],
            "online_verifier": {"trained_games": 1, "frontier_pruned": 0},
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "preconditions_checked": {"offline_fixtures_present": True},
            "reproducibility_checksum": "sha256:" + "c" * 64,
        },
        "4479_a4_re86": {
            "experiment": "experiment_4479_solve_re86",
            "honest_verdict": "success: re86_L1_sprite_overlay_resize_offline_reproduced",
            "target_game": "re86",
            "registered_verifier_operator": "sprite_overlay_resize_verifier",
            "sprite_overlay_verifier_built": True,
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "reproducible_total_levels": 46,
            "verifier_is_oracle": True,
            "preconditions_checked": {"offline_arcade_reachable": True},
            "reproducibility_checksum": "sha256:" + "d" * 64,
        },
        "4480_a4_bp35": {
            "experiment": "experiment_4480_solve_bp35_goal_directed",
            "honest_verdict": "success: bp35_L1_goal_directed_offline_reproduced",
            "target_game": "bp35",
            "goal_directed_solver_built": True,
            "goal_region_identified": True,
            "shape_aware_state_key": True,
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "reproducible_total_levels": 47,
            "verifier_is_oracle": True,
            "preconditions_checked": {"offline_arcade_reachable": True},
            "reproducibility_checksum": "sha256:" + "e" * 64,
        },
        "4481_closeout": {
            "experiment": "experiment_4481_variant_transfer_benchmark",
            "honest_verdict": "success: reflection_variant_transfer_1_of_25_rate_0.0400_games_25",
            "solved_games": ["g1", "g2", "g3"],
            "variants_attempted": 3,
            "variants_solved": 1,
            "transfer_solve_rate": 1 / 3,
            "offline_reproduced": True,
            "reproduced_levels": 1,
            "reproducible_total_levels": 47,
            "verifier_is_oracle": True,
            "preconditions_checked": {"registry_parseable": True},
            "reproducibility_checksum": "sha256:" + "f" * 64,
        },
        "4482_lint": {
            "experiment_id": 4482,
            "honest_verdict": "shipped: lint guard",
            "roadmap_lint_shipped": True,
            "coverage_new_code_100": True,
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "preconditions_checked": {"focused_tests_passed": True},
            "reproducibility_checksum": "sha256:" + "0" * 64,
        },
        "4483_registry": {
            "experiment": "experiment_4483_gate_decouple_registry_reconcile",
            "honest_verdict": "complete: gate_decoupled_registry_reconciled_4483",
            "registry_reconciliation": {
                "authoritative_header": {
                    "reproducible_total_games": 24,
                    "reproducible_total_levels": 47,
                },
                "computed_from_game_rows": {
                    "reproducible_total_games": 25,
                    "reproducible_total_levels": 48,
                },
                "reproduced_counts_match_header": False,
            },
            "offline_reproduced": True,
            "reproduced_levels": 47,
            "preconditions_checked": {"ok": True},
            "reproducibility_checksum": "sha256:" + "1" * 64,
        },
        "4484_hardware": {
            "experiment": 4484,
            "honest_verdict": "complete: hardware_continuity_audit_4484",
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "preconditions_checked": [{"resource": "kv260_ssh", "available": True}],
            "reproducibility_checksum": "sha256:" + "2" * 64,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def test_req_capstone_4486_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4486: OpenSpec declares the .414 capstone contract first."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4486" in spec
    assert "SCENARIO-CAPSTONE-4486" in spec
    assert "results/experiment_4486_capstone_v414.json" in spec
    assert "scripts/summarize_artifact.py" in spec
    assert "flagged_adversarial:true" in spec
    assert "verifier_is_oracle" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_capstone_4486_current_artifacts_answer_scorecard_questions() -> None:
    """SCENARIO-CAPSTONE-4486: current .414 artifacts produce the honest scorecard."""

    artifact = mod.build_artifact(
        Path.cwd(),
        started_s=1.0,
        now_s=1.5,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["offline_reproduced"] is True
    assert artifact["reproduced_levels"] == 47
    assert artifact["a1_generic_solve_rate"]["before_generic_solve_rate"] == pytest.approx(
        0.1428571429
    )
    assert artifact["a1_generic_solve_rate"]["after_generic_solve_rate"] == pytest.approx(
        0.1428571429
    )
    assert artifact["a1_generic_solve_rate"]["generic_solve_rate_delta"] == 0.0
    assert artifact["a1_generic_solve_rate"]["signal"] == "flat"
    assert artifact["a2_cross_game_loo_auroc_v3"]["v2_baseline_loo_auroc"] == pytest.approx(
        0.503096152732577
    )
    assert artifact["a2_cross_game_loo_auroc_v3"]["v3_loo_auroc"] == pytest.approx(
        0.6744657162333668
    )
    assert artifact["a2_cross_game_loo_auroc_v3"]["richer_features_beat_baseline"] is True
    assert artifact["a3_per_game_discriminative_delta"]["solve_rate_delta"] == 0.0
    assert artifact["a3_per_game_discriminative_delta"]["routing_helped"] is False
    assert artifact["a4_goal_state_deepen"]["new_reproduced_levels"] == 2
    assert artifact["a4_goal_state_deepen"]["re86"]["offline_reproduced"] is True
    assert artifact["a4_goal_state_deepen"]["bp35"]["offline_reproduced"] is True
    assert artifact["twenty_five_game_closeout"]["solved_games_count"] == 25
    assert artifact["twenty_five_game_closeout"]["registry_authoritative_total_levels"] == 47
    assert artifact["twenty_five_game_closeout"]["registry_stale_mismatch"] is True
    assert artifact["verifier_is_oracle"] is False
    assert {claim["claim"] for claim in artifact["verifier_claims"]} == {
        "a1_generic_solve_rate",
        "a2_cross_game_loo_auroc_v3",
        "a3_per_game_discriminative_delta",
        "a4_re86_sprite_overlay_resize",
        "a4_bp35_goal_state_deepen",
        "twenty_five_game_variant_transfer_closeout",
    }
    assert all(isinstance(claim["verifier_is_oracle"], bool) for claim in artifact["verifier_claims"])
    assert artifact["flagged_artifacts_skipped"] == []
    assert "gated_on" not in artifact
    assert artifact["submitted_to_leaderboard"] is False


def test_req_capstone_4486_missing_and_flagged_inputs_are_recorded(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4486: missing/flagged upstreams do not fabricate clean metrics."""

    payloads = _fixture_payloads(a2_flagged=True)
    payloads.pop("4480_a4_bp35")
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=2.0,
        now_s=3.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
    )

    mod.validate_artifact(artifact)
    assert artifact["a2_cross_game_loo_auroc_v3"]["state"] == "excluded_flagged_adversarial"
    assert artifact["a2_cross_game_loo_auroc_v3"]["richer_features_beat_baseline"] is False
    assert artifact["a4_goal_state_deepen"]["new_reproduced_levels"] == 1
    assert {"axis": "a4_goal_state_deepen", "artifact_key": "4480_a4_bp35", "experiment_id": 4480} in (
        artifact["per_axis_gaps"]
    )
    flagged = next(row for row in artifact["cited_upstream_artifacts"] if row["experiment_id"] == 4476)
    assert flagged["fields_imported"] == []
    assert artifact["flagged_artifacts_skipped"][0]["experiment_id"] == 4476


def test_req_capstone_4486_branch_helpers_and_validation_guards(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4486: helper branches and schema validation fail closed."""

    assert mod.a1_read(None, False)["state"] == "missing_or_excluded"
    assert mod.a1_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.a2_read(None, False)["state"] == "missing_or_excluded"
    assert mod.a2_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.a3_read(None, False)["state"] == "missing_or_excluded"
    assert mod.a3_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.solve_read(None, False, "x", "operator")["state"] == "missing_or_excluded"
    assert mod.solve_read(None, True, "x", "operator")["state"] == "excluded_flagged_adversarial"
    assert mod.variant_closeout_read(None, False)["state"] == "missing_or_excluded"
    assert mod.variant_closeout_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.registry_closeout_read(None, False)["state"] == "missing_or_excluded"
    assert mod.registry_closeout_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod.operational_context_read(None, False)["state"] == "missing_or_excluded"
    assert mod.operational_context_read(None, True)["state"] == "excluded_flagged_adversarial"
    assert mod._signal_from_delta(0.1) == "improved"  # noqa: SLF001
    assert mod._signal_from_delta(0.0) == "flat"  # noqa: SLF001
    assert mod._signal_from_delta(-0.1) == "regressed"  # noqa: SLF001
    assert mod.a1_read(
        {
            "benchmark": {
                "before_generic_solve_rate": 0.25,
                "after_generic_solve_rate": 0.5,
                "before_solved": 1,
                "after_solved": 2,
                "attempted_games": 4,
            }
        },
        False,
    )["signal"] == "improved"

    _write_default_artifacts(tmp_path, _fixture_payloads())
    valid = mod.build_artifact(
        tmp_path,
        started_s=4.0,
        now_s=5.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
    )
    invalid_cases = [
        ("__delete__honest_verdict", None, "missing required field"),
        ("honest_verdict", "blocked", "honest_verdict"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("offline_reproduced", "yes", "offline_reproduced"),
        ("reproduced_levels", True, "reproduced_levels"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("a1_generic_solve_rate", [], "a1_generic_solve_rate"),
        ("a2_cross_game_loo_auroc_v3", [], "a2_cross_game_loo_auroc_v3"),
        ("a3_per_game_discriminative_delta", [], "a3_per_game_discriminative_delta"),
        ("a4_goal_state_deepen", [], "a4_goal_state_deepen"),
        ("twenty_five_game_closeout", [], "twenty_five_game_closeout"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("verifier_claims", {}, "verifier_claims"),
        ("flagged_artifacts_skipped", {}, "flagged_artifacts_skipped"),
        ("cited_upstream_artifacts", {}, "cited_upstream_artifacts"),
        ("random_seed", 1, "random_seed"),
        ("reproducibility_checksum", "bad", "reproducibility_checksum"),
        ("field_principles", {}, "field_principles"),
        ("upstream_provenance", {}, "upstream_provenance"),
        ("upstream_provenance", [1], "upstream provenance row"),
        ("upstream_provenance", [{"sha256": "bad"}], "invalid sha256"),
        (
            "upstream_provenance",
            [{"sha256": "1" * 64, "skipped": True, "fields_imported": ["x"]}],
            "skipped upstreams",
        ),
        ("verifier_claims", [{"claim": "x"}], "verifier_claims"),
        ("__gated_on__", True, "gated_on"),
        ("__checksum_mismatch__", True, "reproducibility_checksum"),
    ]
    for field, value, message in invalid_cases:
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


def test_req_capstone_4486_write_path_records_capstone_recheck(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4486: writer validates and records the live capstone re-check."""

    _write_default_artifacts(tmp_path, _fixture_payloads())

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=6.0,
        now_s=7.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summarize_zero,
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


def test_req_capstone_4486_unparseable_input_is_excluded(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4486: unparsable upstreams are excluded before metric import."""

    bad_path = tmp_path / mod.DEFAULT_UPSTREAMS["4475_a1_stack"].path
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("{not-json", encoding="utf-8")

    raw, provenance, exclusions = mod._read_inputs(  # noqa: SLF001
        tmp_path,
        _clean_live_flags,
        _summarize_zero,
    )

    assert raw["4475_a1_stack"] is None
    assert provenance[0]["parse_error"].startswith("JSONDecodeError")
    assert provenance[0]["fields_imported"] == []
    assert exclusions[0]["reason"] == "unparsable_or_non_object"
