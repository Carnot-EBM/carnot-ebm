"""Tests for Exp 4578 .422 capstone aggregation.

Spec refs: REQ-CAPSTONE-4578, SCENARIO-CAPSTONE-4578,
SCENARIO-CAPSTONE-4578-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4578_capstone_v422 as mod


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


def _current_live_flags(path: Path) -> list[dict[str, Any]]:
    if path.name == "experiment_4569_verifier_guided_expansion.json":
        return [
            {
                "kind": "FALSE_NEGATIVE_RISK",
                "severity": "warn",
                "detail": "false_negative_risk_open: false_negative_risk_checked=False",
            }
        ]
    if path.name == "experiment_4572_integration_gate.json":
        return [
            {
                "kind": "FALSE_NEGATIVE_RISK",
                "severity": "warn",
                "detail": "false_negative_risk_open: positive_control_passed=None",
            }
        ]
    if path.name in {
        "experiment_4568_clickability_action_effect_predictor.json",
        "experiment_4573_primitive_persist_transfer.json",
    }:
        return [
            {
                "kind": "METHODOLOGY_MISSING",
                "severity": "warn",
                "detail": "Compute-bound artifact missing: model_specs/target_model.",
            }
        ]
    return []


def _null_delta_flags(path: Path) -> list[dict[str, Any]]:
    if path.name == "experiment_4568_clickability_action_effect_predictor.json":
        return [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "median_actions_to_first_levelup_baseline=1.0 and "
                "median_actions_to_first_levelup_best=1.0 match",
            }
        ]
    return []


def _a1_payload(*, improved: bool = False, flagged: bool | None = None) -> JsonDict:
    baseline = 5.0 if improved else 1.0
    with_predictor = 3.0 if improved else 1.0
    delta = baseline - with_predictor
    return {
        "honest_verdict": (
            "success: clickability_predictor_actions_to_levelup_3.0000_below_blind"
            if improved
            else "complete: clickability_predictor_no_efficiency_gain_honest_null_gap_sharpened"
        ),
        "flagged_adversarial": flagged,
        "inference_substrate": (
            "verifier_ensemble_against_cached_candidates -- the CNN trains/scores "
            "against cached transitions; fast CPU forward pass declared."
        ),
        "verifier_is_oracle": False,
        "median_actions_to_first_levelup_with_predictor": with_predictor,
        "median_actions_to_first_levelup_baseline": baseline,
        "median_actions_to_first_levelup_best": with_predictor,
        "actions_delta": delta,
        "efficiency_delta": delta,
        "actions_delta_ci": [0.5, 2.5] if improved else [-0.02, 0.01],
        "efficiency_score_min_human_agent_sq": 1.0,
        "generic_transfer_rate_with_predictor": 0.08 if improved else 0.04,
        "solve_rate_preserved": True,
        "positive_control_passed": True,
        "false_negative_risk_checked": True,
        "null_delta_methodology_note": (
            "actions_delta==0.0 from matched held-out candidate groups; honest no-gain null."
            if not improved
            else ""
        ),
        "barrier_diagnosis": "control_equals_best_null_delta" if flagged else "",
        "chosen_submitted_config": "clickability_predictor" if improved else "unchanged",
        "offline_reproduced": {"all_new_solves_reproduced": True, "newly_solved_variants": []},
    }


def _a2_payload(*, improved: bool = False, control: bool = False) -> JsonDict:
    rate = 0.12 if improved else 0.0
    delta = rate - mod.GENERIC_TRANSFER_BASELINE
    return {
        "honest_verdict": (
            "success: verifier_guided_expansion_generic_transfer_0.1200_above_0.04"
            if improved
            else "complete: verifier_guided_expansion_no_value_honest_null_generation_gap_sharpened"
        ),
        "flagged_adversarial": None,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "generic_transfer_rate_with_expansion": rate,
        "generic_transfer_rate_baseline": mod.GENERIC_TRANSFER_BASELINE,
        "transfer_delta": delta,
        "transfer_ci": [0.02, 0.14] if improved else [-0.12, 0.0],
        "expanded_states_to_goal_with_vs_without": {
            "strictly_lower_than_without": improved,
            "with_expansion_median": 6.0 if improved else None,
            "without_expansion_median": 11.0,
        },
        "winner_generated": {
            "attempted_count": 25,
            "generated_count": 3 if improved else 0,
            "not_generated_count": 22 if improved else 25,
            "with_expansion": improved,
            "without_expansion": not improved,
            "random_priority": control,
        },
        "expansions_used": {"max_expansions": 200, "with_expansion_max": 57},
        "random_priority_control_passed": control,
        "positive_control_passed": control if improved else None,
        "false_negative_risk_checked": control,
        "null_delta_methodology_note": "",
        "solve_rate_preserved": improved,
        "chosen_submitted_config": "verifier_guided_expansion" if improved else "unchanged",
        "missing_verifier_gaps": []
        if improved
        else ["verifier_guided_expansion_no_value_added; winner_not_generated_for=25"],
        "offline_reproduced": True,
    }


def _a3_payload(*, banked: int = 1) -> JsonDict:
    return {
        "honest_verdict": "success: cn04_L2_offline_reproduced",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "offline_reproduced": True,
        "reproduced_levels": banked,
        "target_game": "cn04",
        "target_level": 2,
        "registry_update": {
            "prior_total_declared": 52,
            "new_total_declared": 52 + banked,
            "banked_levels": banked,
            "reconciled_total_delta": banked,
            "updated": bool(banked),
        },
        "reproduction_gate": {"game": "cn04", "claimed_level": 2, "reached_level": 2, "reproduced": True},
    }


def _a4_payload(*, banked: int = 0) -> JsonDict:
    return {
        "honest_verdict": (
            "success: hidden_field_state_ka59_L2_offline_reproduced"
            if banked
            else "complete: hidden_field_state_ka59_gap_sharpened_no_bank_honest_null"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "offline_reproduced": bool(banked),
        "reproduced_levels": banked,
        "target_game": "ka59",
        "target_level": 2 if banked else None,
        "registry_update": {"path": "ops/arc_solve_registry.yaml", "updated": True},
    }


def _a5_payload(*, improved: bool = False, ready: bool = False) -> JsonDict:
    return {
        "honest_verdict": (
            "success: integrated_actions_to_levelup_below_blind_or_generic_transfer_above_0.04"
            if improved
            else "complete: no_lever_raises_a_metric_honest_null"
        ),
        "flagged_adversarial": None,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "median_actions_to_first_levelup_integrated": 12.0 if improved else 2824.5,
        "generic_transfer_rate_integrated": 0.12 if improved else 0.04,
        "levers_integrated": ["A1_clickability_predictor"] if improved else [],
        "additivity_checked": {
            "integrated_actions_delta": 3.0 if improved else 0.0,
            "integrated_generic_transfer_delta": 0.08 if improved else 0.0,
        },
        "core_solves_preserved": True,
        "heldout_solve_rate": 0.12 if improved else 0.04,
        "ready_for_operator_submit": ready,
        "false_negative_risk_checked": True,
        "positive_control_passed": True if improved else None,
        "operator_submission_performed": False,
    }


def _a6_payload() -> JsonDict:
    return {
        "honest_verdict": "success: primitive_persisted_transfer_m0r0_value_added",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "primitive_persisted": {
            "operator": "persistent_action_effect_memory_operator",
            "registry_general_gotcha_id": "primitive_persistent_action_effect_memory_operator",
            "source": "A1_action_effect_predictor",
        },
        "transfer_games": ["dc22", "m0r0", "ka59"],
        "transfer_value_per_game": {
            "dc22": {"actions_reduced": 1.0, "value_added": True, "winner_generated": False},
            "m0r0": {"actions_reduced": 1.0, "value_added": True, "winner_generated": False},
            "ka59": {"actions_reduced": 1.0, "value_added": True, "winner_generated": False},
        },
        "offline_reproduced": False,
        "registry_updated": True,
        "new_levels_banked": 0,
    }


def _b1_payload(*, total: int = 53, rate: float = 0.04) -> JsonDict:
    return {
        "honest_verdict": "shipped: action_efficiency_coheadline_with_ci_wired",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "reproducible_total_levels": total,
        "generic_transfer_rate_over_variants": rate,
        "generic_transfer_ci": [0.0, 0.1] if rate == 0.04 else [0.05, 0.18],
        "median_actions_to_first_levelup": 20.0,
        "human_baseline_actions": 27.0,
        "action_efficiency_score": 1.0,
        "action_efficiency_ci": [1.0, 1.0],
        "variant_attempts_count": 50,
        "variant_solved_count": int(rate * 50),
        "agent_actions_to_first_levelup": [20, 20],
        "human_baseline_sample_count": 300,
    }


def _payloads(*, clean_success: bool = False, a1_flagged: bool | None = None) -> dict[str, JsonDict]:
    return {
        "A1_clickability_predictor": _a1_payload(improved=clean_success, flagged=a1_flagged),
        "A2_verifier_guided_expansion": _a2_payload(improved=clean_success, control=clean_success),
        "A3_levelup_attempt": _a3_payload(banked=1),
        "A4_hidden_state_probe_ka59": _a4_payload(),
        "A5_integration": _a5_payload(improved=clean_success, ready=clean_success),
        "A6_primitive_persist_transfer": _a6_payload(),
        "B1_action_efficiency_coheadline": _b1_payload(rate=0.12 if clean_success else 0.04),
    }


def _write_payloads(root: Path, payloads: dict[str, JsonDict], *, registry_total: int = 53) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)
    registry = root / mod.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        f"schema_version: 1\nreproducible_total_levels: {registry_total}\n",
        encoding="utf-8",
    )


def test_req_capstone_4578_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4578: OpenSpec declares the .422 capstone contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4578" in spec
    assert "SCENARIO-CAPSTONE-4578" in spec
    assert "SCENARIO-CAPSTONE-4578-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "scripts/summarize_artifact.py" in spec
    assert "flagged_adversarial:true" in spec
    assert "false_negative_risk_open" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, wrapped in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert wrapped["principle"] in spec


def test_scenario_capstone_4578_current_read_reports_nulls_and_three_coheadlines(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4578: current .422 numbers are aggregated honestly."""

    _write_payloads(tmp_path, _payloads(), registry_total=53)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=11.0,
        live_flag_runner=_current_live_flags,
        summarize_runner=_summary_codes(
            {
                "experiment_4568_clickability_action_effect_predictor.json": 1,
                "experiment_4569_verifier_guided_expansion.json": 1,
                "experiment_4572_integration_gate.json": 1,
                "experiment_4573_primitive_persist_transfer.json": 1,
            }
        ),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: action_efficiency_null_gaps_sharpened"
    assert artifact["action_efficiency_moved"]["moved"] is False
    assert artifact["action_efficiency_moved"]["median_actions_to_first_levelup_baseline"] == 1.0
    assert artifact["action_efficiency_moved"]["median_actions_to_first_levelup_with_predictor"] == 1.0
    assert artifact["action_efficiency_moved"]["actions_delta_ci"] == [-0.02, 0.01]
    assert artifact["generic_transfer_moved"]["moved"] is False
    assert artifact["generic_transfer_moved"]["status"] == "false_negative_risk_open"
    assert artifact["winner_generated_root_cause_addressed"]["addressed"] is False
    assert artifact["winner_generated_root_cause_addressed"]["evidence_status"] == (
        "false_negative_risk_open"
    )
    assert artifact["reproducible_total_levels_delta"] == {
        "prior_total": 52,
        "current_total": 53,
        "delta": 1,
        "a3_new_levels_banked": 1,
        "a4_new_levels_banked": 0,
        "capability_grew": True,
        "source": "A3_levelup_attempt+A4_hidden_state_probe_ka59+ops/arc_solve_registry.yaml",
    }
    assert artifact["reproducible_total_levels"] == 53
    assert artifact["generic_transfer_rate_over_variants"] == pytest.approx(0.04)
    assert artifact["generic_transfer_ci"] == [0.0, 0.1]
    assert artifact["action_efficiency_score"] == pytest.approx(1.0)
    assert artifact["action_efficiency_ci"] == [1.0, 1.0]
    assert artifact["verifier_is_oracle_distinct_levers"]["oracle_distinct"] is True
    assert artifact["scorecard"]["a5_integration"]["status"] == "false_negative_risk_open"
    assert artifact["scorecard"]["a6_transfer"]["any_transfer_value_added"] is True
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["operator_resubmission_verdict"]["reason"] == (
        "bank_count_53_beats_last_submitted_33"
    )
    handled = artifact["flagged_artifacts_handled"]
    assert handled["learned_cnn_substrate_guard_honored"][0]["artifact_key"] == (
        "A1_clickability_predictor"
    )
    fnr = {row["artifact_key"] for row in handled["positive_control_failed_or_false_negative_risk_open"]}
    assert {"A2_verifier_guided_expansion", "A5_integration"}.issubset(fnr)
    cited = {row["artifact_key"]: row for row in artifact["cited_upstream_artifacts"]}
    assert cited["A2_verifier_guided_expansion"]["fields_imported"] == []
    assert cited["B1_action_efficiency_coheadline"]["fields_imported"]
    assert artifact["preconditions_checked"]["registry"]["reproducible_total_levels"] == 53
    assert artifact["leaderboard_submission"] is False


def test_req_capstone_4578_clean_action_and_expansion_win_path(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4578: A1/A2 wins can produce a success verdict and readiness."""

    _write_payloads(tmp_path, _payloads(clean_success=True), registry_total=53)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=20.0,
        now_s=21.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "success: clickability_predictor_actions_below_blind_or_expansion_generic_transfer_above_0.04"
    )
    assert artifact["action_efficiency_moved"]["moved"] is True
    assert artifact["generic_transfer_moved"]["moved"] is True
    assert artifact["winner_generated_root_cause_addressed"]["addressed"] is True
    assert artifact["scorecard"]["a5_integration"]["integrated_metric_improved"] is True
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["operator_resubmission_verdict"]["resubmission_warranted"] is True


def test_req_capstone_4578_guards_and_validation_fail_closed(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4578: flagged/FNR/carve-out/malformed inputs fail closed."""

    _write_payloads(tmp_path, _payloads(a1_flagged=True), registry_total=53)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=30.0,
        now_s=31.0,
        live_flag_runner=_null_delta_flags,
        summarize_runner=_summary_codes({"experiment_4568_clickability_action_effect_predictor.json": 2}),
    )

    mod.validate_artifact(artifact)
    assert artifact["action_efficiency_moved"]["status"] == "diagnosis_only_null_delta_carve_out"
    assert artifact["action_efficiency_moved"]["headline_numbers_aggregated"] is False
    handled = artifact["flagged_artifacts_handled"]
    assert handled["null_delta_carve_out_diagnosis_read"][0]["artifact_key"] == (
        "A1_clickability_predictor"
    )
    assert handled["excluded"][0]["artifact_key"] == "A1_clickability_predictor"

    empty_root = tmp_path / "empty"
    raw_empty, provenance_empty, handled_empty = mod._read_inputs(  # noqa: SLF001
        empty_root,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )
    assert set(raw_empty) == set(mod.DEFAULT_UPSTREAMS)
    assert provenance_empty == []
    assert handled_empty == {
        "excluded": [],
        "null_delta_carve_out_diagnosis_read": [],
        "positive_control_failed_or_false_negative_risk_open": [],
        "learned_cnn_substrate_guard_honored": [],
    }
    assert mod.load_registry_totals(empty_root)["registry_present"] is False
    assert mod._gate_value_failed(False) is True  # noqa: SLF001
    assert mod._gate_value_failed({"pass": False}) is True  # noqa: SLF001
    assert mod._gate_value_failed([{"nested": False}]) is True  # noqa: SLF001
    assert mod._gate_value_failed("not_a_gate") is False  # noqa: SLF001
    assert mod._acceptance_gate_failed(None) is False  # noqa: SLF001
    assert mod._acceptance_gate_failed({"acceptance_gate_core": False}) is True  # noqa: SLF001
    assert mod._payload_false_negative_risk_open(None) is False  # noqa: SLF001
    assert mod._payload_false_negative_risk_open({"positive_control_passed": False}) is True  # noqa: SLF001
    assert mod._payload_false_negative_risk_open({"honest_verdict": "complete: null", "false_negative_risk_checked": False}) is True  # noqa: E501, SLF001
    assert mod._payload_status({"acceptance_gate_failed": True}) == "failed_acceptance_gate"  # noqa: SLF001
    assert mod._payload_status({"skipped": True}) == "excluded_flagged_adversarial_or_live_critical"  # noqa: SLF001
    assert mod._payload_status({}) == "missing_or_excluded"  # noqa: SLF001
    assert mod._skip_reason(  # noqa: SLF001
        stamped=False,
        critical=False,
        parse_error="",
        acceptance_gate_failed=True,
        diagnosis_context_read=False,
        false_negative_risk_open=False,
    ) == "failed_acceptance_gate"
    assert mod._banked_levels(None) == 0  # noqa: SLF001
    assert mod._a2_transfer(None, {})["generic_transfer_rate_with_expansion"] is None  # noqa: SLF001
    assert mod._b1_metric(None, {})["action_efficiency_score"] == 0.0  # noqa: SLF001
    assert mod._a5_integration(None, {})["ready_for_operator_submit"] is False  # noqa: SLF001
    assert mod._a6_transfer(None, {})["new_levels_banked"] == 0  # noqa: SLF001
    assert mod._operator_resubmission_verdict(  # noqa: SLF001
        ready=True,
        level_delta={"current_total": 53, "capability_grew": True},
        integration={},
        score_gate_failed=True,
    )["reason"] == "failed_acceptance_gate"
    assert mod._operator_resubmission_verdict(  # noqa: SLF001
        ready=True,
        level_delta={"current_total": 10, "capability_grew": False},
        integration={"integrated_metric_improved": True},
        score_gate_failed=False,
    )["reason"] == "clean_integrated_metric_improvement"
    assert mod._operator_resubmission_verdict(  # noqa: SLF001
        ready=False,
        level_delta={"current_total": 10, "capability_grew": False},
        integration={},
        score_gate_failed=False,
    )["reason"] == "no_clean_resubmission_metric_improvement"
    assert mod._is_sha256_prefixed("bad") is False  # noqa: SLF001

    bad_json = tmp_path / mod.DEFAULT_UPSTREAMS["A1_clickability_predictor"].path
    bad_json.write_text("{", encoding="utf-8")
    raw_bad, provenance_bad, _handled_bad = mod._read_inputs(  # noqa: SLF001
        tmp_path,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )
    assert raw_bad["A1_clickability_predictor"] is None
    assert "JSONDecodeError" in provenance_bad[0]["parse_error"]
    bad_json.unlink()
    _write_json(bad_json, _a1_payload())

    written = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=40.0,
        now_s=41.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )
    assert json.loads(written.read_text(encoding="utf-8"))["result_path"] == "results/out.json"
    assert mod.run(
        tmp_path,
        write=False,
        started_s=41.0,
        now_s=42.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )["duration_s"] == 1.0
    mod.run(
        tmp_path,
        write=True,
        started_s=42.0,
        now_s=43.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()

    bad = json.loads(written.read_text(encoding="utf-8"))
    invalid_cases = [
        ("honest_verdict", "not-terminal", "honest_verdict"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("action_efficiency_moved", [], "action_efficiency_moved"),
        ("generic_transfer_moved", [], "generic_transfer_moved"),
        ("winner_generated_root_cause_addressed", [], "winner_generated_root_cause_addressed"),
        ("reproducible_total_levels_delta", [], "reproducible_total_levels_delta"),
        ("action_efficiency_score", 2.0, "action_efficiency_score"),
        ("generic_transfer_rate_over_variants", 2.0, "generic_transfer_rate_over_variants"),
        ("verifier_is_oracle_distinct_levers", [], "verifier_is_oracle_distinct_levers"),
        ("flagged_artifacts_handled", [], "flagged_artifacts_handled"),
        ("cited_upstream_artifacts", {}, "cited_upstream_artifacts"),
        ("ready_for_operator_submit", "yes", "ready_for_operator_submit"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("leaderboard_submission", True, "leaderboard_submission"),
        ("random_seed", 1, "random_seed"),
        ("upstream_provenance", {}, "upstream_provenance"),
        ("reproducibility_checksum", "sha256:" + "1" * 64, "reproducibility_checksum"),
        ("__delete__honest_verdict", None, "missing required field"),
        ("__row_not_mapping__", True, "upstream provenance row"),
        ("__skipped_imports__", True, "skipped upstreams"),
        ("__bad_sha__", True, "invalid sha256"),
        ("__ready_without_reason__", True, "ready_for_operator_submit requires"),
    ]
    for field, value, message in invalid_cases:
        invalid = json.loads(json.dumps(bad))
        if field == "__delete__honest_verdict":
            invalid.pop("honest_verdict")
        elif field == "__row_not_mapping__":
            invalid["upstream_provenance"] = [1]
        elif field == "__skipped_imports__":
            invalid["upstream_provenance"][0]["skipped"] = True
            invalid["upstream_provenance"][0]["fields_imported"] = ["actions_delta"]
        elif field == "__bad_sha__":
            invalid["upstream_provenance"][0]["sha256"] = "bad"
        elif field == "__ready_without_reason__":
            invalid["ready_for_operator_submit"] = True
            invalid["operator_resubmission_verdict"]["resubmission_warranted"] = False
            invalid["reproducibility_checksum"] = mod.checksum_from_artifact(invalid)
        else:
            invalid[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(invalid)
