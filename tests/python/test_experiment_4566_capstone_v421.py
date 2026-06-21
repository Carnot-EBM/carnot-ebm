"""Tests for Exp 4566 .421 capstone aggregation.

Spec refs: REQ-CAPSTONE-4566, SCENARIO-CAPSTONE-4566,
SCENARIO-CAPSTONE-4566-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4566_capstone_v421 as mod


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
    if path.name == "experiment_4556_verifier_router_generic_transfer.json":
        return [
            {
                "kind": "FALSE_NEGATIVE_RISK",
                "severity": "warn",
                "detail": "false_negative_risk_open: false_negative_risk_checked=False",
            }
        ]
    if path.name == "experiment_4557_executable_world_model_proposer.json":
        return [
            {
                "kind": "FALSE_NEGATIVE_RISK",
                "severity": "warn",
                "detail": "false_negative_risk_open: positive_control_passed=False",
            }
        ]
    if path.name == "experiment_4560_integration_8game_gate.json":
        return [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": "live model marker with too-short duration",
            },
            {
                "kind": "FALSE_NEGATIVE_RISK",
                "severity": "warn",
                "detail": "false_negative_risk_open: no positive control",
            },
        ]
    return []


def _null_delta_flags(path: Path) -> list[dict[str, Any]]:
    if path.name == "experiment_4557_executable_world_model_proposer.json":
        return [
            {
                "kind": "TAUTOLOGY",
                "severity": "critical",
                "detail": "core_efficiency_baseline=2.0074 and core_efficiency_best=2.0074 match",
            }
        ]
    return _current_live_flags(path)


def _a1_payload(*, clean_success: bool = False, flagged: bool = True) -> JsonDict:
    rate = 0.12 if clean_success else mod.GENERIC_TRANSFER_BASELINE
    delta = round(rate - mod.GENERIC_TRANSFER_BASELINE, 10)
    return {
        "honest_verdict": (
            "success: verifier_router_generic_transfer_0.1200_above_0.04"
            if clean_success
            else "complete: verifier_router_no_value_added_honest_null_gap_sharpened"
        ),
        "flagged_adversarial": flagged,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "generic_transfer_rate_with_verifier": rate,
        "generic_transfer_rate_baseline": mod.GENERIC_TRANSFER_BASELINE,
        "generic_transfer_delta": delta,
        "generic_transfer_ci": [0.02, 0.14] if clean_success else [0.0, 0.0],
        "solve_rate_preserved": True,
        "random_router_control_passed": clean_success,
        "positive_control_passed": clean_success,
        "false_negative_risk_checked": clean_success,
        "offline_reproduced": True,
        "chosen_submitted_config": "verifier_router" if clean_success else "unchanged",
        "missing_verifier_gaps": [] if clean_success else ["verifier_router_no_value_added"],
    }


def _a2_payload(*, positive: bool = False, null_delta_carveout: bool = False) -> JsonDict:
    best = mod.CORE_EFFICIENCY_BASELINE if null_delta_carveout else (2.2134 if positive else None)
    payload: JsonDict = {
        "honest_verdict": (
            "success: executable_proposer_lp85_reached_L2_core_efficiency_2.2134_above_2.0074"
            if positive
            else "complete: executable_proposer_positive_control_failed_no_deeper_barrier_refined"
        ),
        "flagged_adversarial": True if null_delta_carveout else None,
        "inference_substrate": "live_llm_inference",
        "positive_control_passed": positive or null_delta_carveout,
        "false_negative_risk_checked": positive or null_delta_carveout,
        "core_efficiency_baseline": mod.CORE_EFFICIENCY_BASELINE,
        "core_efficiency_best": best,
        "efficiency_delta": None if best is None else round(best - mod.CORE_EFFICIENCY_BASELINE, 10),
        "llm_proposer_value": {
            "count": 1 if positive else 0,
            "opportunities": 1,
            "rate": 1.0 if positive else 0.0,
            "events": ["lp85:L2"] if positive else [],
        },
        "deepest_level_reached_per_core_game": {
            "offline_dsl_baseline": {"lp85": 1, "m0r0": 1},
            "executable_proposer": {"lp85": 2 if positive else 1, "m0r0": 1},
        },
        "core_solves_preserved": True if positive else None,
        "barrier_refinement": "positive_control_failed: proposer_failed",
        "verifier_is_oracle": False,
        "offline_reproduced": positive,
        "chosen_submitted_config": "executable_proposer" if positive else "unchanged",
    }
    if null_delta_carveout:
        payload["null_delta_methodology_note"] = (
            "baseline equals best because no deeper CORE level was reached; diagnosis only."
        )
        payload["barrier_diagnosis"] = "control_equals_best_null_delta"
    return payload


def _a3_payload(*, banked: int = 0) -> JsonDict:
    return {
        "honest_verdict": "success: m0r0_L2_offline_reproduced" if banked else "complete: m0r0_delta_identified_no_bank",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "offline_reproduced": True,
        "reproduced_levels": banked,
        "target_game": "m0r0",
        "target_level": 2,
        "reproduction_gate": {"game": "m0r0", "claimed_level": 2, "reached_level": 2, "reproduced": True},
        "registry_update": {
            "prior_total_declared": 52,
            "new_total_declared": 52 + banked,
            "banked_levels": banked,
            "reconciled_total_delta": banked,
            "updated": bool(banked),
        },
    }


def _a4_payload() -> JsonDict:
    return {
        "honest_verdict": "complete: hidden_field_state_gap_sharpened_no_bank_honest_null",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "registry_update": {"path": "ops/arc_solve_registry.yaml", "updated": True},
    }


def _a5_payload(*, clean_success: bool = False, flagged: bool = True, gate_pass: bool = True) -> JsonDict:
    rate = 0.12 if clean_success else mod.GENERIC_TRANSFER_BASELINE
    core = 2.2134 if clean_success else mod.CORE_EFFICIENCY_BASELINE
    return {
        "honest_verdict": (
            "success: integrated_generic_transfer_0.1200_above_0.04_or_core_efficiency_above_2.0074"
            if clean_success
            else "complete: no_lever_raises_a_metric_honest_null"
        ),
        "flagged_adversarial": flagged,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "core_efficiency_integrated": core,
        "generic_transfer_rate_integrated": rate,
        "core_solves_preserved": True,
        "levers_integrated": ["A1_verifier_router"] if clean_success else [],
        "heldout_solve_rate": rate,
        "ready_for_operator_submit": clean_success and gate_pass,
        "false_negative_risk_checked": True,
        "operator_submission_performed": False,
        "gate_result": {"pass": gate_pass, "current": {"core_efficiency": core}},
    }


def _a6_payload() -> JsonDict:
    return {
        "honest_verdict": "complete: primitive_persisted_transfer_null_characterized",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "primitive_persisted": {"operator": "verifier_router_candidate_ranking_operator"},
        "transfer_games": ["tu93", "tr87", "sc25"],
        "transfer_value_per_game": {
            "tu93": {"ordering_gain": 0, "value_added": False},
            "tr87": {"ordering_gain": 0, "value_added": False},
            "sc25": {"ordering_gain": 0, "value_added": False},
        },
        "offline_reproduced": False,
        "registry_updated": True,
        "new_levels_banked": 0,
    }


def _b1_payload(*, rate: float = 0.04, total: int = 52) -> JsonDict:
    return {
        "honest_verdict": "shipped: generic_transfer_coheadline_with_ci_wired",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "reproducible_total_levels": total,
        "generic_transfer_rate_over_variants": rate,
        "generic_transfer_ci": [0.0, 0.1] if rate == 0.04 else [0.05, 0.18],
        "variant_attempts_count": 50,
        "variant_solved_count": int(rate * 50),
        "metric_wired_into_capstone": {"known_game_bank_inflates_transfer": False},
    }


def _prior_payload() -> JsonDict:
    return {
        "honest_verdict": "complete: archive_420_activate_421_true_close_state_recorded",
        "inference_substrate": mod.INFERENCE_SUBSTRATE,
        "close_state_420": {
            "a2_cross_game_discrimination": {
                "status": "clean_cross_game_discrimination_above_chance",
                "above_chance": True,
                "loo_auroc_mean": 0.6744657162333668,
                "loo_auroc_display": 0.674,
                "loo_auroc_ci": [0.6058303817975523, 0.7451888709482918],
                "ci_excludes_chance": True,
                "verifier_is_oracle": False,
                "positive_control_passed": True,
            },
            "reproducible_total_levels": 52,
        },
    }


def _payloads(
    *,
    clean_success: bool = False,
    null_delta_carveout: bool = False,
    gate_pass: bool = True,
) -> dict[str, JsonDict]:
    return {
        "A1_verifier_router": _a1_payload(clean_success=clean_success, flagged=not clean_success),
        "A2_executable_proposer": _a2_payload(
            positive=clean_success,
            null_delta_carveout=null_delta_carveout,
        ),
        "A3_levelup_attempt": _a3_payload(banked=1 if clean_success else 0),
        "A4_hidden_state_probe": _a4_payload(),
        "A5_integration": _a5_payload(
            clean_success=clean_success,
            flagged=not clean_success,
            gate_pass=gate_pass,
        ),
        "A6_transfer": _a6_payload(),
        "B1_generic_transfer_coheadline": _b1_payload(
            rate=0.12 if clean_success else 0.04,
            total=53 if clean_success else 52,
        ),
        "P0_prior_420_transition": _prior_payload(),
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


def test_req_capstone_4566_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4566: OpenSpec declares the .421 capstone before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4566" in spec
    assert "SCENARIO-CAPSTONE-4566" in spec
    assert "SCENARIO-CAPSTONE-4566-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "scripts/summarize_artifact.py" in spec
    assert "flagged_adversarial:true" in spec
    assert "false_negative_risk_open" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, wrapped in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert wrapped["principle"] in spec


def test_scenario_capstone_4566_current_read_reports_honest_null_and_coheadline(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4566: flagged/FNR inputs do not become clean wins."""

    _write_payloads(tmp_path, _payloads(), registry_total=52)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=11.0,
        live_flag_runner=_current_live_flags,
        summarize_runner=_summary_codes(
            {
                "experiment_4556_verifier_router_generic_transfer.json": 1,
                "experiment_4557_executable_world_model_proposer.json": 1,
                "experiment_4560_integration_8game_gate.json": 2,
            }
        ),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: verifier_router_null_reinduction_retired_or_refined"
    assert artifact["generic_transfer_moved"]["moved"] is False
    assert artifact["generic_transfer_moved"]["coheadline_rate"] == pytest.approx(0.04)
    assert artifact["generic_transfer_moved"]["generic_transfer_ci"] == [0.0, 0.1]
    assert artifact["verifier_router_value_added"]["headline_numbers_aggregated"] is False
    assert artifact["verifier_router_value_added"]["value_added"] is False
    assert artifact["executable_proposer_positive_control"]["positive_control_passed"] is False
    assert artifact["executable_proposer_positive_control"]["false_negative_risk_open"] is True
    assert artifact["efficiency_moved"] is False
    assert artifact["reinduction_retired"] is True
    assert artifact["reproducible_total_levels_delta"] == {
        "prior_total": 52,
        "current_total": 52,
        "delta": 0,
        "a3_new_levels_banked": 0,
        "a4_new_levels_banked": 0,
        "a6_new_levels_banked": 0,
        "capability_grew": False,
        "source": "A3_levelup_attempt+A4_hidden_state_probe+A6_transfer+ops/arc_solve_registry.yaml",
    }
    assert artifact["generic_transfer_rate_over_variants"] == pytest.approx(0.04)
    assert artifact["cross_game_discrimination_above_chance"]["above_chance"] is True
    assert artifact["cross_game_discrimination_above_chance"]["loo_auroc_display"] == pytest.approx(0.674)
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["operator_resubmission_verdict"]["resubmission_warranted"] is False
    assert artifact["scorecard"]["a5_integration"]["status"] == "false_negative_risk_open"

    handled = artifact["flagged_artifacts_handled"]
    excluded = {row["artifact_key"] for row in handled["excluded"]}
    fnr = {row["artifact_key"] for row in handled["positive_control_failed_or_false_negative_risk_open"]}
    assert {"A1_verifier_router", "A5_integration"}.issubset(excluded)
    assert {"A1_verifier_router", "A2_executable_proposer", "A5_integration"}.issubset(fnr)
    cited = {row["artifact_key"]: row for row in artifact["cited_upstream_artifacts"]}
    assert cited["A1_verifier_router"]["fields_imported"] == []
    assert cited["A5_integration"]["fields_imported"] == []
    assert cited["B1_generic_transfer_coheadline"]["fields_imported"]
    assert artifact["preconditions_checked"]["registry"]["reproducible_total_levels"] == 52
    assert artifact["leaderboard_submission"] is False


def test_req_capstone_4566_clean_success_can_recommend_operator_slot(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4566: clean router transfer plus integration can become a win."""

    _write_payloads(tmp_path, _payloads(clean_success=True), registry_total=53)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=20.0,
        now_s=21.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "success: verifier_router_generic_transfer_0.1200_above_0.04"
    assert artifact["generic_transfer_moved"]["moved"] is True
    assert artifact["verifier_router_value_added"]["value_added"] is True
    assert artifact["executable_proposer_positive_control"]["positive_control_passed"] is True
    assert artifact["efficiency_moved"] is True
    assert artifact["reinduction_retired"] is False
    assert artifact["reproducible_total_levels_delta"]["delta"] == 1
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["operator_resubmission_verdict"] == {
        "resubmission_warranted": True,
        "reason": "clean_integrated_metric_improvement",
        "operator_only": True,
        "hidden_eval_gate": "beat_13_levels",
    }


def test_req_capstone_4566_acceptance_gate_failure_and_carveout_are_bounded(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4566: failed gates override, null-delta carve-out is diagnosis-only."""

    _write_payloads(
        tmp_path,
        _payloads(clean_success=True, null_delta_carveout=True, gate_pass=False),
        registry_total=52,
    )

    artifact = mod.build_artifact(
        tmp_path,
        started_s=30.0,
        now_s=31.0,
        live_flag_runner=_null_delta_flags,
        summarize_runner=_summary_codes({"experiment_4557_executable_world_model_proposer.json": 2}),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: verifier_router_null_reinduction_retired_or_refined"
    assert artifact["generic_transfer_moved"]["moved"] is False
    assert artifact["efficiency_moved"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["scorecard"]["a5_integration"]["status"] == "failed_acceptance_gate"
    assert artifact["operator_resubmission_verdict"]["reason"] == "failed_acceptance_gate"
    handled = artifact["flagged_artifacts_handled"]
    carved = {
        row["artifact_key"]
        for row in handled["null_delta_carve_out_diagnosis_read"]
    }
    assert "A2_executable_proposer" in carved
    a2 = artifact["executable_proposer_positive_control"]
    assert a2["status"] == "diagnosis_only_null_delta_carve_out"
    assert a2["headline_numbers_aggregated"] is False

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
    }
    assert mod.load_registry_totals(empty_root)["registry_present"] is False
    assert mod._gate_value_failed(False) is True  # noqa: SLF001
    assert mod._gate_value_failed({"nested": False}) is True  # noqa: SLF001
    assert mod._gate_value_failed([{"pass": False}]) is True  # noqa: SLF001
    assert mod._gate_value_failed("not_a_gate") is False  # noqa: SLF001
    assert mod._acceptance_gate_failed(None) is False  # noqa: SLF001
    assert mod._payload_false_negative_risk_open(None) is False  # noqa: SLF001
    assert mod._payload_false_negative_risk_open({"positive_control_passed": False}) is True  # noqa: SLF001
    assert mod._payload_status({}) == "missing_or_excluded"  # noqa: SLF001
    assert mod._b1_metric(None, {})["variant_attempts_count"] == 0  # noqa: SLF001
    assert mod._executable_proposer_positive_control(  # noqa: SLF001
        None,
        {"diagnosis_context": {"barrier_refinement": "missing proposer"}},
    )["barrier_refinement"] == "missing proposer"
    assert mod._banked_levels(None) == 0  # noqa: SLF001
    assert mod._a6_transfer(None, {})["new_levels_banked"] == 0  # noqa: SLF001
    assert mod._prior_total(None, {"prior_total": 51}, 52) == 51  # noqa: SLF001
    assert mod._prior_total(None, {}, 52) == 52  # noqa: SLF001
    assert mod._cross_game_discrimination(None, {})["above_chance"] is False  # noqa: SLF001
    assert mod._is_sha256_prefixed("bad") is False  # noqa: SLF001

    bad_json = tmp_path / mod.DEFAULT_UPSTREAMS["A1_verifier_router"].path
    bad_json.write_text("{", encoding="utf-8")
    raw_bad, provenance_bad, _handled_bad = mod._read_inputs(  # noqa: SLF001
        tmp_path,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )
    assert raw_bad["A1_verifier_router"] is None
    assert "JSONDecodeError" in provenance_bad[0]["parse_error"]
    bad_json.unlink()
    _write_json(bad_json, _a1_payload(clean_success=True, flagged=False))

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
        ("generic_transfer_moved", [], "generic_transfer_moved"),
        ("verifier_router_value_added", [], "verifier_router_value_added"),
        ("executable_proposer_positive_control", [], "executable_proposer_positive_control"),
        ("efficiency_moved", "no", "efficiency_moved"),
        ("reinduction_retired", "yes", "reinduction_retired"),
        ("reproducible_total_levels_delta", [], "reproducible_total_levels_delta"),
        ("generic_transfer_rate_over_variants", 2, "generic_transfer_rate_over_variants"),
        ("cross_game_discrimination_above_chance", [], "cross_game_discrimination_above_chance"),
        ("flagged_artifacts_handled", [], "flagged_artifacts_handled"),
        ("cited_upstream_artifacts", {}, "cited_upstream_artifacts"),
        ("ready_for_operator_submit", "yes", "ready_for_operator_submit"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("random_seed", 1, "random_seed"),
        ("leaderboard_submission", True, "leaderboard_submission"),
        ("upstream_provenance", {}, "upstream_provenance"),
        ("reproducibility_checksum", "sha256:" + "1" * 64, "reproducibility_checksum"),
        ("__delete__honest_verdict", None, "missing required field"),
        ("__row_not_mapping__", True, "upstream provenance row"),
        ("__skipped_imports__", True, "skipped upstreams"),
        ("__bad_sha__", True, "invalid sha256"),
        ("__ready_without_generic__", True, "ready_for_operator_submit requires"),
    ]
    for field, value, message in invalid_cases:
        invalid = json.loads(json.dumps(bad))
        if field == "__delete__honest_verdict":
            invalid.pop("honest_verdict")
        elif field == "__row_not_mapping__":
            invalid["upstream_provenance"] = [1]
        elif field == "__skipped_imports__":
            invalid["upstream_provenance"][0]["skipped"] = True
            invalid["upstream_provenance"][0]["fields_imported"] = ["generic_transfer_delta"]
        elif field == "__bad_sha__":
            invalid["upstream_provenance"][0]["sha256"] = "bad"
        elif field == "__ready_without_generic__":
            invalid["ready_for_operator_submit"] = True
            invalid["generic_transfer_moved"]["moved"] = False
            invalid["reproducibility_checksum"] = mod.checksum_from_artifact(invalid)
        else:
            invalid[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(invalid)
