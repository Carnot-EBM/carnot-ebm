"""Tests for Exp 4554 .420 capstone aggregation.

Spec refs: REQ-CAPSTONE-4554, SCENARIO-CAPSTONE-4554,
SCENARIO-CAPSTONE-4554-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4554_capstone_v420 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_codes(codes: dict[str, int]):
    def _runner(path: Path, _root: Path) -> int:
        return int(codes.get(path.name, 0))

    return _runner


def _clean_live_flags(_: Path) -> list[dict[str, Any]]:
    return []


def _live_flags_for_current_fixture(path: Path) -> list[dict[str, Any]]:
    if path.name == "experiment_4544_llm_proposer_reinduction.json":
        return [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": (
                    "core_efficiency_baseline=2.0074 and core_efficiency_best=2.0074 "
                    "agree to >5 sig figs."
                ),
            },
            {
                "kind": "IMPLAUSIBLE_PERFECT",
                "severity": "info",
                "detail": "efficiency_delta=0.0",
            },
        ]
    if path.name == "experiment_4548_integration_8game_gate.json":
        return [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": (
                    "core_efficiency_baseline=2.0074 and "
                    "core_efficiency_integrated=2.0074 agree to >5 sig figs."
                ),
            }
        ]
    return []


def _a1_payload(*, clean_success: bool = False, flagged: bool = True) -> JsonDict:
    best = 2.2134 if clean_success else mod.CORE_EFFICIENCY_BASELINE
    llm_levels = {"lp85": 2 if clean_success else 1, "m0r0": 1, "sp80": 1, "vc33": 1}
    return {
        "honest_verdict": (
            "success: llm_proposer_lp85_reached_L2_core_efficiency_2.2134_above_2.0074"
            if clean_success
            else "complete: llm_proposer_positive_control_failed_false_negative_risk_open"
        ),
        "flagged_adversarial": flagged,
        "core_efficiency_baseline": mod.CORE_EFFICIENCY_BASELINE,
        "core_efficiency_best": best,
        "efficiency_delta": round(best - mod.CORE_EFFICIENCY_BASELINE, 10),
        "null_delta_methodology_note": (
            "baseline==best because no lever reached a deeper offline-reproduced CORE level "
            "with CORE solves preserved; not a measurement bug."
            if not clean_success
            else ""
        ),
        "llm_proposer_value": {
            "count": 1 if clean_success else 0,
            "opportunities": 1,
            "rate": 1.0 if clean_success else 0.0,
            "events": ["lp85:L2"] if clean_success else [],
        },
        "deepest_level_reached_per_core_game": {
            "offline_dsl_baseline": {"lp85": 1, "m0r0": 1, "sp80": 1, "vc33": 1},
            "llm_proposer": llm_levels,
        },
        "core_solves_preserved": True,
        "positive_control_passed": clean_success,
        "false_negative_risk_checked": clean_success,
        "verifier_is_oracle": False,
        "offline_reproduced": clean_success,
        "barrier_refinement": (
            "positive_control_failed: live Qwen proposer did not produce the known reachable fixture plan."
            if not clean_success
            else ""
        ),
        "chosen_submitted_config": "llm_proposer" if clean_success else "unchanged",
    }


def _a2_payload() -> JsonDict:
    return {
        "honest_verdict": "success: cross_game_discrimination_loo_auroc_0.674_above_chance",
        "loo_auroc_mean": 0.6744657162333668,
        "loo_auroc_ci": [0.6058303817975523, 0.7451888709482918],
        "loo_ci_excludes_chance": True,
        "in_sample_auroc": 0.8710834214701216,
        "verifier_is_oracle": False,
        "positive_control_passed": True,
        "false_negative_risk_checked": True,
    }


def _a3_payload(*, reproduced: bool = True) -> JsonDict:
    return {
        "honest_verdict": "success: su15_L2_offline_reproduced",
        "offline_reproduced": reproduced,
        "reproduced_levels": 1 if reproduced else 0,
        "target_game": "su15",
        "target_level": 2,
        "reproduction_gate": {
            "game": "su15",
            "claimed_level": 2,
            "reached_level": 2 if reproduced else 1,
            "reproduced": reproduced,
        },
        "registry_update": {
            "prior_total_declared": 51,
            "new_total_declared": 52 if reproduced else 51,
            "reconciled_total_delta": 1 if reproduced else 0,
            "banked_levels": 1 if reproduced else 0,
            "updated": reproduced,
        },
    }


def _a4_payload(*, improved: bool = False) -> JsonDict:
    blind = 5.0 if improved else 1.0
    cnn = 3.0 if improved else 1.0
    return {
        "honest_verdict": (
            "success: frame_change_cnn_median_actions_reduced_3.0"
            if improved
            else "complete: frame_change_cnn_no_action_reduction_honest_null"
        ),
        "median_actions_to_first_levelup_blind": blind,
        "median_actions_to_first_levelup_cnn": cnn,
        "solve_rate_blind": 1.0,
        "solve_rate_cnn": 1.0,
        "solve_rate_preserved": True,
        "cnn_held_out_delta_auroc": 0.7092359400538687,
        "positive_control_passed": True,
        "false_negative_risk_checked": True,
        "ranking_metrics": {
            "median_actions_to_first_levelup_blind": blind,
            "median_actions_to_first_levelup_cnn": cnn,
            "solve_rate_blind": 1.0,
            "solve_rate_cnn": 1.0,
            "solve_rate_preserved": True,
            "action_reduction": improved,
        },
    }


def _a5_payload(*, clean_success: bool = False, flagged: bool = True, gate_pass: bool = True) -> JsonDict:
    integrated = 2.2134 if clean_success else mod.CORE_EFFICIENCY_BASELINE
    return {
        "honest_verdict": (
            "success: integrated_core_efficiency_2.2134_above_2.0074"
            if clean_success
            else "complete: no_lever_raises_core_efficiency_honest_null"
        ),
        "flagged_adversarial": flagged,
        "core_efficiency_baseline": mod.CORE_EFFICIENCY_BASELINE,
        "core_efficiency_integrated": integrated,
        "core_solves_preserved": True,
        "levers_integrated": ["A1_llm_proposer"] if clean_success else [],
        "heldout_solve_rate": 0.0,
        "ready_for_operator_submit": clean_success and gate_pass,
        "operator_submission_performed": False,
        "gate_result": {
            "pass": gate_pass,
            "verdict": "PASS" if gate_pass else "FAIL",
            "current": {"core_efficiency": integrated},
        },
    }


def _a6_payload(*, new_levels_banked: int = 0) -> JsonDict:
    return {
        "honest_verdict": "complete: llm_proposer_primitive_persisted_transfer_null_characterized",
        "primitive_persisted": {
            "operator": "llm_proposer_reinduction_operator",
            "registry_general_gotcha_id": "primitive_per_level_reinduction_operator",
        },
        "transfer_games": ["tu93", "tr87", "sc25"],
        "transfer_deepest_level_per_game": {"tu93": 5, "tr87": 6, "sc25": 5},
        "reachable_plan_produced": {"tu93": False, "tr87": False, "sc25": False},
        "representation_transfer": {"tu93": True, "tr87": True, "sc25": True},
        "offline_reproduced": new_levels_banked > 0,
        "registry_updated": True,
        "new_levels_banked": new_levels_banked,
    }


def _b1_payload() -> JsonDict:
    return {
        "honest_verdict": "shipped: honest_sprint_metric_variant_transfer_wired",
        "reproducible_total_levels": 52,
        "generic_transfer_rate_over_variants": 0.04,
        "variant_attempts_count": 25,
        "variant_solved_count": 1,
        "metric_wired_into_capstone": {
            "reported_side_by_side": [
                "reproducible_total_levels",
                "generic_transfer_rate_over_variants",
            ],
            "known_game_bank_inflates_transfer": False,
        },
    }


def _payloads(
    *,
    clean_success: bool = False,
    default_flags: bool = True,
    a3_reproduced: bool = True,
    a4_improved: bool = False,
    gate_pass: bool = True,
) -> dict[str, JsonDict]:
    return {
        "A1_llm_proposer": _a1_payload(clean_success=clean_success, flagged=default_flags),
        "A2_cross_game_discrimination": _a2_payload(),
        "A3_levelup_attempt": _a3_payload(reproduced=a3_reproduced),
        "A4_frame_change_predictor": _a4_payload(improved=a4_improved),
        "A5_integration": _a5_payload(
            clean_success=clean_success,
            flagged=default_flags,
            gate_pass=gate_pass,
        ),
        "A6_transfer": _a6_payload(),
        "B1_honest_sprint_metric": _b1_payload(),
    }


def _write_payloads(root: Path, payloads: dict[str, JsonDict], *, registry_total: int = 52) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)
    registry = root / mod.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        f"schema_version: 1\nreproducible_total_levels: {registry_total}\n",
        encoding="utf-8",
    )


def test_req_capstone_4554_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4554: OpenSpec declares the .420 capstone before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4554" in spec
    assert "SCENARIO-CAPSTONE-4554" in spec
    assert "SCENARIO-CAPSTONE-4554-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "scripts/summarize_artifact.py" in spec
    assert "flagged_adversarial:true" in spec
    assert "2.0074" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_capstone_4554_current_read_is_honest_null_with_capability_growth(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4554: flagged A1/A5 are not laundered, A3/B1 are reported."""

    _write_payloads(tmp_path, _payloads())

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=11.0,
        live_flag_runner=_live_flags_for_current_fixture,
        summarize_runner=_summary_codes(
            {
                "experiment_4544_llm_proposer_reinduction.json": 2,
                "experiment_4548_integration_8game_gate.json": 2,
            }
        ),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: llm_proposer_null_efficiency_unmoved_barrier_refined"
    assert artifact["efficiency_moved"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["leaderboard_submission"] is False
    assert artifact["llm_proposer_value_summary"]["status"] == "diagnosis_only_null_delta_carve_out"
    assert artifact["llm_proposer_value_summary"]["headline_numbers_aggregated"] is False
    assert artifact["llm_proposer_value_summary"]["value"] == {
        "count": None,
        "opportunities": None,
        "rate": None,
        "events": [],
    }
    assert artifact["llm_proposer_value_summary"]["diagnosis"]["barrier_refinement"].startswith(
        "positive_control_failed"
    )
    assert artifact["deepest_level_gains_per_core_game"] == {
        "status": "no_clean_score_lever_evidence",
        "headline_numbers_aggregated": False,
        "clean_before_after": {},
        "gains": {},
        "any_core_game_deeper_clean": False,
    }
    assert artifact["cross_game_discrimination_above_chance"]["above_chance"] is True
    assert artifact["cross_game_discrimination_above_chance"]["loo_auroc_mean"] == pytest.approx(
        0.6744657162333668
    )
    assert artifact["cross_game_discrimination_above_chance"]["verifier_is_oracle"] is False
    assert artifact["action_efficiency_improved"]["improved"] is False
    assert artifact["action_efficiency_improved"]["solve_rate_preserved"] is True
    assert artifact["reproducible_total_levels_delta"] == {
        "prior_total": 51,
        "current_total": 52,
        "delta": 1,
        "banked_levels": 1,
        "a6_new_levels_banked": 0,
        "source": "A3_levelup_attempt+A6_transfer+ops/arc_solve_registry.yaml",
        "capability_grew": True,
    }
    assert artifact["generic_transfer_rate_over_variants"] == pytest.approx(0.04)
    assert artifact["scorecard"]["a5_integration"]["status"] == (
        "excluded_flagged_adversarial_or_live_critical"
    )
    assert artifact["scorecard"]["a6_transfer"]["representation_generalized"] is True
    assert artifact["scorecard"]["a6_transfer"]["new_levels_banked"] == 0
    assert artifact["operator_resubmission_verdict"] == {
        "resubmission_warranted": False,
        "reason": "no_clean_integrated_core_efficiency_improvement",
        "operator_only": True,
    }

    handled = artifact["flagged_artifacts_handled"]
    excluded = {row["artifact_key"] for row in handled["excluded"]}
    carved = {row["artifact_key"] for row in handled["null_delta_carve_out_diagnosis_read"]}
    assert excluded == {"A1_llm_proposer", "A5_integration"}
    assert carved == {"A1_llm_proposer"}
    cited = {row["artifact_key"]: row for row in artifact["cited_upstream_artifacts"]}
    assert cited["A1_llm_proposer"]["fields_imported"] == []
    assert cited["A5_integration"]["fields_imported"] == []
    assert cited["A2_cross_game_discrimination"]["fields_imported"]
    assert cited["A3_levelup_attempt"]["fields_imported"]
    assert cited["B1_honest_sprint_metric"]["fields_imported"]
    assert artifact["preconditions_checked"]["registry"]["reproducible_total_levels"] == 52


def test_req_capstone_4554_clean_success_requires_llm_depth_and_integration(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4554: clean LLM depth plus submitted integration allows readiness."""

    _write_payloads(
        tmp_path,
        _payloads(clean_success=True, default_flags=False, a4_improved=True),
    )

    artifact = mod.build_artifact(
        tmp_path,
        started_s=20.0,
        now_s=21.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "success: llm_proposer_core_efficiency_2.2134_above_2.0074"
    assert artifact["efficiency_moved"] is True
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["llm_proposer_value_summary"]["value"]["count"] == 1
    assert artifact["deepest_level_gains_per_core_game"]["gains"]["lp85"] == 1
    assert artifact["action_efficiency_improved"]["improved"] is True
    assert artifact["scorecard"]["a5_integration"]["submitted_config_improved"] is True
    assert artifact["operator_resubmission_verdict"]["resubmission_warranted"] is True
    assert artifact["flagged_artifacts_handled"]["excluded"] == []


def test_req_capstone_4554_failed_acceptance_gate_overrides_success(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4554: failed acceptance gates prevent celebratory verdicts."""

    _write_payloads(
        tmp_path,
        _payloads(
            clean_success=True,
            default_flags=False,
            a3_reproduced=False,
            a4_improved=True,
            gate_pass=False,
        ),
        registry_total=51,
    )

    artifact = mod.build_artifact(
        tmp_path,
        started_s=30.0,
        now_s=31.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: llm_proposer_null_efficiency_unmoved_barrier_refined"
    assert artifact["efficiency_moved"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["scorecard"]["a3_levelup"]["status"] == "failed_acceptance_gate"
    assert artifact["scorecard"]["a5_integration"]["status"] == "failed_acceptance_gate"
    assert artifact["reproducible_total_levels_delta"]["capability_grew"] is False
    assert artifact["operator_resubmission_verdict"]["reason"] == "failed_acceptance_gate"


def test_req_capstone_4554_write_missing_and_validation_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4554: writer, missing inputs, helpers, and validation fail closed."""

    raw, provenance, handled = mod._read_inputs(  # noqa: SLF001
        tmp_path,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )
    assert set(raw) == set(mod.DEFAULT_UPSTREAMS)
    assert all(value is None for value in raw.values())
    assert provenance == []
    assert handled == {"excluded": [], "null_delta_carve_out_diagnosis_read": []}
    assert mod.load_registry_totals(tmp_path / "missing") == {
        "registry_path": str(mod.REGISTRY_RELATIVE_PATH),
        "registry_present": False,
        "reproducible_total_levels": 0,
    }
    assert mod._gate_value_failed(False) is True  # noqa: SLF001
    assert mod._gate_value_failed([{"ok": False}]) is True  # noqa: SLF001
    assert mod._gate_value_failed({"pass": True, "telemetry_flag": False}) is False  # noqa: SLF001
    assert mod._gate_value_failed({"pass": False}) is True  # noqa: SLF001
    assert mod._gate_value_failed({"nested": False}) is True  # noqa: SLF001
    assert mod._gate_value_failed("not_a_gate") is False  # noqa: SLF001
    assert mod._acceptance_gate_failed(None) is False  # noqa: SLF001
    assert mod._payload_status({"acceptance_gate_failed": True}) == "failed_acceptance_gate"  # noqa: SLF001
    assert mod._int_mapping(None) == {}  # noqa: SLF001
    assert mod._level_pair({"deepest_level_reached_per_core_game": {"baseline": {"g": 1}, "best": {"g": 2}}}) == (  # noqa: SLF001
        {"g": 1},
        {"g": 2},
    )
    assert mod._level_pair({"deepest_level_reached_per_core_game": {"1": {"g": 1}, "3": {"g": 2}}}) == (  # noqa: SLF001
        {"g": 1},
        {"g": 2},
    )
    assert mod._level_pair({}) == ({}, {})  # noqa: SLF001
    assert mod._diagnosis_from_row({"diagnosis_context": []}) == {}  # noqa: SLF001
    assert mod._cross_game_discrimination(None, {"skipped": True})["status"] == (  # noqa: SLF001
        "excluded_flagged_adversarial_or_live_critical"
    )
    assert mod._action_efficiency(None, {})["status"] == "missing_or_excluded"  # noqa: SLF001
    assert mod._a3_levelup(None, {}, {"reproducible_total_levels": 5}) == {  # noqa: SLF001
        "status": "missing_or_excluded",
        "level_up_banked": False,
        "target_game": "",
        "target_level": None,
        "banked_levels": 0,
        "prior_total": None,
        "current_total": 5,
        "delta": 0,
    }
    assert mod._a6_transfer(None, {})["status"] == "missing_or_excluded"  # noqa: SLF001
    assert mod._a6_transfer(_a6_payload(new_levels_banked=1), {})["status"] == (  # noqa: SLF001
        "representation_generalized_and_level_banked"
    )
    assert mod._a6_transfer(  # noqa: SLF001
        {**_a6_payload(), "representation_transfer": {"tu93": False}},
        {},
    )["status"] == "transfer_null"
    assert mod._honest_verdict(True, {"core_efficiency_best": 2.1}, {"core_efficiency_integrated": None}) == (  # noqa: SLF001
        "success: llm_proposer_core_efficiency_2.1000_above_2.0074"
    )
    assert mod._b1_metric(None, {})["generic_transfer_rate_over_variants"] == 0.0  # noqa: SLF001

    bad_json = tmp_path / mod.DEFAULT_UPSTREAMS["A2_cross_game_discrimination"].path
    bad_json.parent.mkdir(parents=True, exist_ok=True)
    bad_json.write_text("{", encoding="utf-8")
    raw_bad, provenance_bad, _handled_bad = mod._read_inputs(  # noqa: SLF001
        tmp_path,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )
    bad_row = provenance_bad[0]
    assert raw_bad["A2_cross_game_discrimination"] is None
    assert "JSONDecodeError" in bad_row["parse_error"]
    assert bad_row["fields_imported"] == []
    bad_json.unlink()

    _write_payloads(tmp_path, _payloads())
    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=40.0,
        now_s=41.0,
        live_flag_runner=_live_flags_for_current_fixture,
        summarize_runner=_summary_codes(
            {
                "experiment_4544_llm_proposer_reinduction.json": 2,
                "experiment_4548_integration_8game_gate.json": 2,
            }
        ),
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["result_path"] == "results/out.json"
    assert mod.run(
        root=tmp_path,
        write=False,
        started_s=41.0,
        now_s=42.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )["duration_s"] == 1.0
    mod.run(
        root=tmp_path,
        write=True,
        started_s=42.0,
        now_s=43.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()

    invalid_cases = [
        ("__delete__honest_verdict", None, "missing required field"),
        ("honest_verdict", "not-terminal", "honest_verdict"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("efficiency_moved", "no", "efficiency_moved"),
        ("llm_proposer_value_summary", [], "llm_proposer_value_summary"),
        ("deepest_level_gains_per_core_game", [], "deepest_level_gains_per_core_game"),
        ("cross_game_discrimination_above_chance", [], "cross_game_discrimination_above_chance"),
        ("action_efficiency_improved", [], "action_efficiency_improved"),
        ("reproducible_total_levels_delta", [], "reproducible_total_levels_delta"),
        ("generic_transfer_rate_over_variants", 2, "generic_transfer_rate_over_variants"),
        ("flagged_artifacts_handled", [], "flagged_artifacts_handled"),
        ("cited_upstream_artifacts", {}, "cited_upstream_artifacts"),
        ("ready_for_operator_submit", "yes", "ready_for_operator_submit"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("scorecard", [], "scorecard"),
        ("operator_resubmission_verdict", [], "operator_resubmission_verdict"),
        ("duration_s", True, "duration_s"),
        ("random_seed", 1, "random_seed"),
        ("reproducibility_checksum", "bad", "reproducibility_checksum"),
        ("leaderboard_submission", True, "leaderboard_submission"),
        ("upstream_provenance", {}, "upstream_provenance"),
        ("upstream_provenance", [1], "upstream provenance row"),
        ("__bad_sha__", True, "invalid sha256"),
        ("__ready_without_efficiency__", True, "ready_for_operator_submit requires"),
        ("__gated_on__", True, "gated_on"),
        ("__checksum_mismatch__", True, "reproducibility_checksum"),
        ("__skipped_imports__", True, "skipped upstreams"),
    ]
    for field, value, message in invalid_cases:
        invalid = json.loads(json.dumps(written))
        if field == "__delete__honest_verdict":
            invalid.pop("honest_verdict")
        elif field == "__bad_sha__":
            invalid["upstream_provenance"][0]["sha256"] = "bad"
        elif field == "__ready_without_efficiency__":
            invalid["ready_for_operator_submit"] = True
            invalid["efficiency_moved"] = False
            invalid["reproducibility_checksum"] = mod.checksum_from_artifact(invalid)
        elif field == "__gated_on__":
            invalid["gated_on"] = value
        elif field == "__checksum_mismatch__":
            invalid["reproducibility_checksum"] = "sha256:" + "1" * 64
        elif field == "__skipped_imports__":
            invalid["upstream_provenance"][0]["skipped"] = True
            invalid["upstream_provenance"][0]["fields_imported"] = ["core_efficiency_best"]
        else:
            invalid[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(invalid)
