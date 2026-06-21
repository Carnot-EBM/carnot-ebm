"""Tests for Exp 4542 .419 capstone aggregation.

Spec refs: REQ-CAPSTONE-4542, SCENARIO-CAPSTONE-4542,
SCENARIO-CAPSTONE-4542-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4542_capstone_v419 as mod


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


def _live_flags_for_default_fixture(path: Path) -> list[dict[str, Any]]:
    if path.name == "experiment_4533_per_level_goal_reinduction.json":
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
    if path.name == "experiment_4534_energy_trust_next_level_routing.json":
        return [
            {
                "kind": "DURATION_TOO_SHORT",
                "severity": "critical",
                "detail": "duration_s=0.885 but verifier scoring needs a real floor.",
            }
        ]
    if path.name == "experiment_4536_integration_8game_gate.json":
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
    best_levels = {"lp85": 2 if clean_success else 1, "m0r0": 1, "sp80": 1, "vc33": 1}
    return {
        "honest_verdict": (
            "success: reinduction_core_efficiency_2.2134_above_2.0074"
            if clean_success
            else "complete: reinduction_no_deeper_level_barrier_refined_honest_null"
        ),
        "flagged_adversarial": flagged,
        "core_efficiency_baseline": mod.CORE_EFFICIENCY_BASELINE,
        "core_efficiency_best": best,
        "efficiency_delta": round(best - mod.CORE_EFFICIENCY_BASELINE, 10),
        "null_delta_methodology_note": "Annotated honest null." if not clean_success else "",
        "core_solves_preserved": True,
        "deepest_level_reached_per_core_game": {
            "baseline": {"lp85": 1, "m0r0": 1, "sp80": 1, "vc33": 1},
            "best": best_levels,
        },
        "barrier_refinement": (
            "post_level_reinduction_triggered_but_no_reachable_l2_plan"
            if not clean_success
            else ""
        ),
        "chosen_submitted_config": "candidate" if clean_success else "unchanged",
    }


def _a2_payload(*, flagged: bool = True) -> JsonDict:
    return {
        "honest_verdict": "complete: energy_routing_no_deeper_level_signal_characterized_honest_null",
        "flagged_adversarial": flagged,
        "core_efficiency_baseline": mod.CORE_EFFICIENCY_BASELINE,
        "core_efficiency_energy_routed": 2.0084,
        "core_solves_preserved": True,
        "deepest_level_reached_per_core_game": {
            "no_energy_control": {"lp85": 1, "m0r0": 1, "sp80": 1, "vc33": 1},
            "energy_routed": {"lp85": 2, "m0r0": 1, "sp80": 1, "vc33": 1},
        },
        "energy_separation_auroc": 1.0,
        "verifier_is_oracle": False,
    }


def _a3_payload(*, reproduced: bool = True) -> JsonDict:
    return {
        "honest_verdict": "success: sp80_L2_offline_reproduced",
        "offline_reproduced": reproduced,
        "reproduced_levels": 1 if reproduced else 0,
        "target_game": "sp80",
        "target_level": 2,
        "reproduction_gate": {
            "game": "sp80",
            "claimed_level": 2,
            "reached_level": 2 if reproduced else 1,
            "reproduced": reproduced,
        },
        "registry_update": {
            "prior_total_declared": 50,
            "new_total_declared": 51 if reproduced else 50,
            "reconciled_total_delta": 1 if reproduced else 0,
            "banked_levels": 1 if reproduced else 0,
            "updated": reproduced,
        },
    }


def _a4_payload(*, clean_success: bool = False, flagged: bool = True, gate_pass: bool = True) -> JsonDict:
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
        "ready_for_operator_submit": clean_success and gate_pass,
        "operator_submission_performed": False,
        "gate_result": {
            "pass": gate_pass,
            "current": {
                "core_efficiency": integrated,
                "deepest_level_by_game": {
                    "lp85": 2 if clean_success else 1,
                    "m0r0": 1,
                    "sp80": 1,
                    "vc33": 1,
                },
            },
        },
        "per_game_deepest_level_reached": {
            "lp85": 2 if clean_success else 1,
            "m0r0": 1,
            "sp80": 1,
            "vc33": 1,
        },
    }


def _a5_payload(*, representation_transfer: bool = True, new_levels_banked: int = 0) -> JsonDict:
    return {
        "honest_verdict": "complete: reinduction_primitive_persisted_transfer_null_characterized",
        "offline_reproduced": new_levels_banked > 0,
        "new_levels_banked": new_levels_banked,
        "primitive_persisted": {
            "operator": "per_level_reinduction_operator",
            "registry_general_gotcha_id": "primitive_per_level_reinduction_operator",
        },
        "representation_transfer": {
            "tu93": representation_transfer,
            "tr87": representation_transfer,
            "sc25": representation_transfer,
        },
        "transfer_deepest_level_per_game": {"tu93": 5, "tr87": 6, "sc25": 5},
        "transfer_games": ["tu93", "tr87", "sc25"],
        "registry_updated": True,
    }


def _payloads(
    *,
    clean_success: bool = False,
    default_flags: bool = True,
    a3_reproduced: bool = True,
    gate_pass: bool = True,
) -> dict[str, JsonDict]:
    return {
        "A1_goal_reinduction": _a1_payload(clean_success=clean_success, flagged=default_flags),
        "A2_energy_routing": _a2_payload(flagged=default_flags),
        "A3_levelup_attempt": _a3_payload(reproduced=a3_reproduced),
        "A4_integration": _a4_payload(
            clean_success=clean_success,
            flagged=default_flags,
            gate_pass=gate_pass,
        ),
        "A5_primitive_transfer": _a5_payload(),
    }


def _write_payloads(root: Path, payloads: dict[str, JsonDict], *, registry_total: int = 51) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)
    registry = root / mod.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        f"schema_version: 1\nreproducible_total_levels: {registry_total}\n",
        encoding="utf-8",
    )


def test_req_capstone_4542_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4542: OpenSpec declares the .419 capstone before code."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4542" in spec
    assert "SCENARIO-CAPSTONE-4542" in spec
    assert "SCENARIO-CAPSTONE-4542-FIELD-PRINCIPLES" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "scripts/summarize_artifact.py" in spec
    assert "flagged_adversarial:true" in spec
    assert "2.0074" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_capstone_4542_default_read_is_null_efficiency_with_a3_growth(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4542: flagged A1/A2/A4 are not laundered, A3 grows levels."""

    _write_payloads(tmp_path, _payloads())

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=11.0,
        live_flag_runner=_live_flags_for_default_fixture,
        summarize_runner=_summary_codes(
            {
                "experiment_4533_per_level_goal_reinduction.json": 2,
                "experiment_4534_energy_trust_next_level_routing.json": 2,
                "experiment_4536_integration_8game_gate.json": 2,
            }
        ),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: reinduction_null_efficiency_unmoved_barrier_refined"
    assert artifact["efficiency_moved"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["leaderboard_submission"] is False
    assert artifact["scorecard"]["a1_reinduction"]["status"] == "diagnosis_only_null_delta_carve_out"
    assert artifact["scorecard"]["a1_reinduction"]["headline_numbers_aggregated"] is False
    assert artifact["scorecard"]["a1_reinduction"]["diagnosis"]["barrier_refinement"].startswith(
        "post_level_reinduction"
    )
    assert artifact["deepest_level_gains_per_core_game"] == {
        "status": "no_clean_score_lever_evidence",
        "headline_numbers_aggregated": False,
        "clean_before_after": {},
        "gains": {},
        "any_core_game_deeper_clean": False,
    }
    assert artifact["energy_routing_generalization"] == {
        "status": "excluded_flagged_adversarial_or_live_critical",
        "generalized": False,
        "core_efficiency_baseline": mod.CORE_EFFICIENCY_BASELINE,
        "core_efficiency_energy_routed": None,
        "core_efficiency_delta": None,
        "energy_separation_auroc": None,
        "verifier_is_oracle": None,
        "reason": "headline numbers quarantined",
    }
    assert artifact["reproducible_total_levels_delta"] == {
        "prior_total": 50,
        "current_total": 51,
        "delta": 1,
        "banked_levels": 1,
        "source": "A3_levelup_attempt+ops/arc_solve_registry.yaml",
        "capability_grew": True,
    }
    assert artifact["primitive_transfer_generalization"]["representation_generalized"] is True
    assert artifact["primitive_transfer_generalization"]["new_levels_banked"] == 0
    assert artifact["primitive_transfer_generalization"]["status"] == (
        "representation_generalized_no_reproducible_level_bank"
    )
    assert artifact["operator_resubmission_verdict"] == {
        "resubmission_warranted": False,
        "reason": "no_clean_integrated_core_efficiency_improvement",
        "operator_only": True,
    }

    handled = artifact["flagged_artifacts_handled"]
    excluded = {row["artifact_key"] for row in handled["excluded"]}
    carved = {row["artifact_key"] for row in handled["null_delta_carve_out_diagnosis_read"]}
    assert excluded == {"A1_goal_reinduction", "A2_energy_routing", "A4_integration"}
    assert carved == {"A1_goal_reinduction"}
    cited = {row["artifact_key"]: row for row in artifact["cited_upstream_artifacts"]}
    assert cited["A1_goal_reinduction"]["fields_imported"] == []
    assert cited["A2_energy_routing"]["fields_imported"] == []
    assert cited["A4_integration"]["fields_imported"] == []
    assert cited["A3_levelup_attempt"]["fields_imported"]
    assert cited["A5_primitive_transfer"]["fields_imported"]
    assert artifact["preconditions_checked"]["registry"]["reproducible_total_levels"] == 51


def test_req_capstone_4542_clean_success_requires_core_depth_and_integration(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4542: clean deeper CORE evidence plus integration allows operator readiness."""

    _write_payloads(tmp_path, _payloads(clean_success=True, default_flags=False))

    artifact = mod.build_artifact(
        tmp_path,
        started_s=20.0,
        now_s=21.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "success: reinduction_core_efficiency_2.2134_above_2.0074"
    assert artifact["efficiency_moved"] is True
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["deepest_level_gains_per_core_game"]["gains"]["lp85"] == 1
    assert artifact["deepest_level_gains_per_core_game"]["any_core_game_deeper_clean"] is True
    assert artifact["energy_routing_generalization"]["generalized"] is True
    assert artifact["energy_routing_generalization"]["energy_separation_auroc"] == pytest.approx(1.0)
    assert artifact["scorecard"]["a4_integration"]["submitted_config_improved"] is True
    assert artifact["operator_resubmission_verdict"]["resubmission_warranted"] is True
    assert artifact["flagged_artifacts_handled"]["excluded"] == []


def test_req_capstone_4542_failed_acceptance_gate_overrides_success_verdict(
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4542: failed acceptance gates prevent growth and submitted readiness."""

    _write_payloads(
        tmp_path,
        _payloads(clean_success=True, default_flags=False, a3_reproduced=False, gate_pass=False),
        registry_total=50,
    )

    artifact = mod.build_artifact(
        tmp_path,
        started_s=30.0,
        now_s=31.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )

    mod.validate_artifact(artifact)
    assert artifact["efficiency_moved"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["scorecard"]["a3_levelup"]["status"] == "failed_acceptance_gate"
    assert artifact["scorecard"]["a4_integration"]["status"] == "failed_acceptance_gate"
    assert artifact["reproducible_total_levels_delta"]["capability_grew"] is False
    assert artifact["operator_resubmission_verdict"]["reason"] == "failed_acceptance_gate"


def test_req_capstone_4542_write_run_missing_and_validation_guards(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4542: writer, missing inputs, and schema validation fail closed."""

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
    assert mod._payload_status({}) == "missing_or_excluded"  # noqa: SLF001
    assert mod._int_mapping(None) == {}  # noqa: SLF001
    assert mod._deepest_pair(  # noqa: SLF001
        {
            "deepest_level_reached_per_core_game": {
                "1": {"lp85": 1},
                "3": {"lp85": 2},
            }
        }
    ) == ({"lp85": 1}, {"lp85": 2})
    assert mod._deepest_pair({}) == ({}, {})  # noqa: SLF001
    assert mod._diagnosis_from_row({"diagnosis_context": []}) == {}  # noqa: SLF001
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
    assert mod._a5_primitive_transfer(None, {})["status"] == "missing_or_excluded"  # noqa: SLF001
    banked_transfer = mod._a5_primitive_transfer(  # noqa: SLF001
        _a5_payload(representation_transfer=True, new_levels_banked=1),
        {},
    )
    assert banked_transfer["status"] == "representation_generalized_and_level_banked"
    assert banked_transfer["offline_reproduced_new_level"] is True
    assert mod._a5_primitive_transfer(  # noqa: SLF001
        _a5_payload(representation_transfer=False),
        {},
    )["status"] == "transfer_null"
    assert mod._honest_verdict(  # noqa: SLF001
        True,
        {"core_efficiency_best": 2.1},
        {"core_efficiency_integrated": None},
    ) == "success: reinduction_core_efficiency_2.1000_above_2.0074"

    _write_payloads(tmp_path, _payloads())
    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=40.0,
        now_s=41.0,
        live_flag_runner=_live_flags_for_default_fixture,
        summarize_runner=_summary_codes(
            {
                "experiment_4533_per_level_goal_reinduction.json": 2,
                "experiment_4534_energy_trust_next_level_routing.json": 2,
                "experiment_4536_integration_8game_gate.json": 2,
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
        ("deepest_level_gains_per_core_game", [], "deepest_level_gains_per_core_game"),
        ("reproducible_total_levels_delta", [], "reproducible_total_levels_delta"),
        ("flagged_artifacts_handled", [], "flagged_artifacts_handled"),
        ("cited_upstream_artifacts", {}, "cited_upstream_artifacts"),
        ("ready_for_operator_submit", "yes", "ready_for_operator_submit"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("scorecard", [], "scorecard"),
        ("energy_routing_generalization", [], "energy_routing_generalization"),
        ("primitive_transfer_generalization", [], "primitive_transfer_generalization"),
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
        elif field == "__gated_on__":
            invalid["gated_on"] = value
        elif field == "__checksum_mismatch__":
            invalid["reproducibility_checksum"] = "sha256:" + "1" * 64
        elif field == "__skipped_imports__":
            invalid["upstream_provenance"][0]["skipped"] = True
            invalid["upstream_provenance"][0]["fields_imported"] = ["core_efficiency_best"]
        elif field == "__bad_sha__":
            invalid["upstream_provenance"][0]["sha256"] = "bad"
        elif field == "__ready_without_efficiency__":
            invalid["ready_for_operator_submit"] = True
            invalid["efficiency_moved"] = False
            invalid["reproducibility_checksum"] = mod.checksum_from_artifact(invalid)
        else:
            invalid[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(invalid)
