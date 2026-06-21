"""Tests for Exp 4531 .418 capstone aggregation.

Spec refs: REQ-CAPSTONE-4531, SCENARIO-CAPSTONE-4531,
SCENARIO-CAPSTONE-4531-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4531_capstone_v418 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_live_flags(_: Path) -> list[dict[str, Any]]:
    return []


def _live_critical_for(*names: str):
    flagged = set(names)

    def _runner(path: Path) -> list[dict[str, Any]]:
        if path.name in flagged:
            return [{"kind": "TAUTOLOGY", "severity": "critical", "detail": path.name}]
        return []

    return _runner


def _summary_codes(codes: dict[str, int]):
    def _runner(path: Path, _root: Path) -> int:
        return int(codes.get(path.name, 0))

    return _runner


def _payloads(*, clean_integration: bool = False, a3_gate_reproduced: bool = True) -> dict[str, JsonDict]:
    integration = {
        "honest_verdict": "complete: no_lever_raises_core_efficiency_honest_null",
        "flagged_adversarial": not clean_integration,
        "core_efficiency_baseline": mod.CORE_EFFICIENCY_BASELINE,
        "core_efficiency_integrated": 2.0101 if clean_integration else mod.CORE_EFFICIENCY_BASELINE,
        "core_solves_preserved": True,
        "ready_for_operator_submit": clean_integration,
        "operator_submission_performed": False,
        "heldout_solve_rate": 0.0,
        "gate_result": {
            "current": {
                "median_actions_on_core": 7200.0 if clean_integration else 2824.5,
                "core_efficiency": 2.0101 if clean_integration else mod.CORE_EFFICIENCY_BASELINE,
                "solved_count": 4,
            },
            "pass": True,
        },
        "nav_diagnostics": {
            "integrated_config": "improved" if clean_integration else "unchanged",
            "reset_replay_steps_integrated": 4500,
            "reset_replay_steps_candidate_after": 4533,
        },
    }
    return {
        "A1_forward_walk_navigation": {
            "honest_verdict": "complete: forward_walk_no_reduction_honest_null",
            "flagged_adversarial": True,
            "median_actions_on_core_control": 7761.5,
            "median_actions_on_core_best": 7761.5,
            "core_solves_preserved": True,
            "nav_diagnostics_before_after": {
                "before": {"reset_replay_steps": 4576, "forward_walk_hits": 26},
                "after": {"reset_replay_steps": 4533, "forward_walk_hits": 34},
            },
            "chosen_submitted_config": "unchanged",
            "leaderboard_submission": False,
        },
        "A2_reach_deeper_levels": {
            "honest_verdict": "complete: l1_l2_barrier_diagnosed_depth_cap_honest_null",
            "flagged_adversarial": True,
            "core_efficiency_baseline": mod.CORE_EFFICIENCY_BASELINE,
            "core_efficiency_best": mod.CORE_EFFICIENCY_BASELINE,
            "barrier_diagnosis": {
                "root_cause": "depth_cap",
                "new_win_condition_likely": True,
                "induction_not_engaged": True,
                "actionable_next_step": "force post-L1 DSL/goal-predicate induction",
            },
            "leaderboard_submission": False,
        },
        "A2_stop_after_levelup": {
            "honest_verdict": "success: stop_after_levelup_core_actions_2825_below_control",
            "median_actions_on_core_control": 7761.5,
            "median_actions_on_core_best": 2825.5,
            "core_solves_preserved": True,
            "levels_per_game_preserved": {"passed": True, "lost_level_depth_games": []},
            "leaderboard_submission": False,
        },
        "A3_levelup_attempt": {
            "honest_verdict": "success: cd82_L2_offline_reproduced",
            "offline_reproduced": a3_gate_reproduced,
            "reproduced_levels": 1,
            "target_game": "cd82",
            "target_level": 2,
            "reproduction_gate": {
                "game": "cd82",
                "claimed_level": 2,
                "reached_level": 2 if a3_gate_reproduced else 1,
                "reproduced": a3_gate_reproduced,
            },
            "registry_update": {
                "prior_total_declared": 48,
                "new_total_declared": 50 if a3_gate_reproduced else 48,
                "reconciled_total_delta": 2 if a3_gate_reproduced else 0,
                "banked_levels": 1 if a3_gate_reproduced else 0,
                "updated": a3_gate_reproduced,
            },
        },
        "A4_integration": integration,
    }


def _write_payloads(root: Path, payloads: dict[str, JsonDict], *, registry_total: int = 50) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)
    registry = root / mod.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        f"schema_version: 1\nreproducible_total_levels: {registry_total}\n"
        "prior_submitted_baseline_levels: 13\n",
        encoding="utf-8",
    )


def test_req_capstone_4531_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4531: OpenSpec declares the .418 capstone contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4531" in spec
    assert "SCENARIO-CAPSTONE-4531" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "scripts/summarize_artifact.py" in spec
    assert "core_efficiency" in spec
    assert "flagged_adversarial:true" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_capstone_4531_skips_flagged_artifacts_and_reports_clean_a3_growth(
    tmp_path: Path,
) -> None:
    """SCENARIO-CAPSTONE-4531: flagged A1/A2/A4 import no metrics while clean A3 grows levels."""

    _write_payloads(tmp_path, _payloads())

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=11.0,
        live_flag_runner=_live_critical_for("experiment_4526_integration_8game_gate.json"),
        summarize_runner=_summary_codes({"experiment_4526_integration_8game_gate.json": 2}),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: nav_fix_null_efficiency_unmoved"
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["efficiency_moved"] is False
    assert artifact["ready_for_operator_submit"] is False
    assert artifact["leaderboard_submission"] is False
    assert artifact["scorecard"]["core_efficiency"] == {
        "baseline": mod.CORE_EFFICIENCY_BASELINE,
        "integrated": None,
        "delta": None,
        "moved": False,
        "reason": "integration_excluded_flagged_or_live_critical",
    }
    assert artifact["scorecard"]["nav_fix_delta"]["status"] == "excluded_flagged_adversarial"
    assert artifact["scorecard"]["stop_after_levelup_delta"]["status"] == "retired_action_trimming_context"
    assert artifact["scorecard"]["stop_after_levelup_delta"]["median_actions_delta"] == pytest.approx(-4936.0)
    assert artifact["scorecard"]["integration_headline"]["submitted_config_improved"] is False
    assert artifact["a2_l1_l2_barrier_diagnosis"] == {
        "status": "excluded_flagged_adversarial",
        "cleanly_reportable": False,
        "what_blocks_deeper_levels": None,
        "what_to_build_next": "not_cleanly_reportable_from_flagged_artifact",
    }
    assert artifact["reproducible_total_levels_delta"] == {
        "prior_total": 48,
        "current_total": 50,
        "delta": 2,
        "banked_levels": 1,
        "source": "A3_levelup_attempt+ops/arc_solve_registry.yaml",
        "capability_grew": True,
    }

    skipped = {row["artifact_key"]: row for row in artifact["flagged_artifacts_skipped"]}
    assert set(skipped) == {"A1_forward_walk_navigation", "A2_reach_deeper_levels", "A4_integration"}
    assert skipped["A4_integration"]["live_critical"] is True
    cited = {row["artifact_key"]: row for row in artifact["cited_upstream_artifacts"]}
    assert cited["A1_forward_walk_navigation"]["fields_imported"] == []
    assert cited["A2_reach_deeper_levels"]["fields_imported"] == []
    assert cited["A4_integration"]["fields_imported"] == []
    assert cited["A3_levelup_attempt"]["fields_imported"]
    assert artifact["preconditions_checked"]["registry"]["reproducible_total_levels"] == 50


def test_req_capstone_4531_clean_integration_is_required_for_operator_submit(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4531: operator readiness needs a clean integrated core-efficiency rise."""

    _write_payloads(tmp_path, _payloads(clean_integration=True))

    artifact = mod.build_artifact(
        tmp_path,
        started_s=20.0,
        now_s=21.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "success: nav_fix_core_actions_7200_below_7760"
    assert artifact["efficiency_moved"] is True
    assert artifact["ready_for_operator_submit"] is True
    assert artifact["scorecard"]["core_efficiency"]["integrated"] == pytest.approx(2.0101)
    assert artifact["scorecard"]["core_efficiency"]["delta"] == pytest.approx(0.0027)
    assert artifact["scorecard"]["integration_headline"]["submitted_config_improved"] is True
    assert "A4_integration" not in {row["artifact_key"] for row in artifact["flagged_artifacts_skipped"]}


def test_req_capstone_4531_failed_a3_acceptance_gate_blocks_growth(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4531: a failed upstream gate overrides a success-flavored verdict."""

    _write_payloads(tmp_path, _payloads(a3_gate_reproduced=False), registry_total=48)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=30.0,
        now_s=31.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )

    mod.validate_artifact(artifact)
    assert artifact["scorecard"]["a3_levelup"]["status"] == "failed_acceptance_gate"
    assert artifact["scorecard"]["a3_levelup"]["level_up_banked"] is False
    assert artifact["reproducible_total_levels_delta"]["capability_grew"] is False
    assert artifact["reproducible_total_levels_delta"]["delta"] == 0
    assert artifact["efficiency_moved"] is False


def test_req_capstone_4531_write_path_and_validation_guards(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4531: writer emits stable JSON and validation fails closed."""

    _write_payloads(tmp_path, _payloads())
    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=40.0,
        now_s=41.0,
        live_flag_runner=_live_critical_for("experiment_4526_integration_8game_gate.json"),
        summarize_runner=_summary_codes({"experiment_4526_integration_8game_gate.json": 2}),
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
        ("reproducible_total_levels_delta", [], "reproducible_total_levels_delta"),
        ("flagged_artifacts_skipped", {}, "flagged_artifacts_skipped"),
        ("cited_upstream_artifacts", {}, "cited_upstream_artifacts"),
        ("ready_for_operator_submit", "yes", "ready_for_operator_submit"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("scorecard", [], "scorecard"),
        ("a2_l1_l2_barrier_diagnosis", [], "a2_l1_l2_barrier_diagnosis"),
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
            invalid["upstream_provenance"][0]["fields_imported"] = ["median_actions_on_core_best"]
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


def test_req_capstone_4531_helper_edges_cover_missing_and_clean_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4531: helper edges keep optional upstreams honest and bounded."""

    raw, provenance, flagged = mod._read_inputs(  # noqa: SLF001
        tmp_path,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )
    assert set(raw) == set(mod.DEFAULT_UPSTREAMS)
    assert all(value is None for value in raw.values())
    assert provenance == []
    assert flagged == []
    assert mod.load_registry_totals(tmp_path / "missing") == {
        "registry_path": str(mod.REGISTRY_RELATIVE_PATH),
        "registry_present": False,
        "reproducible_total_levels": 0,
    }
    assert mod._gate_value_failed([{"ok": False}]) is True  # noqa: SLF001
    assert mod._acceptance_gate_failed(None) is False  # noqa: SLF001

    nav = mod._nav_fix_delta(  # noqa: SLF001
        {
            "honest_verdict": "complete: nav_context",
            "median_actions_on_core_control": 7761.5,
            "median_actions_on_core_best": 7600.0,
            "core_solves_preserved": True,
            "chosen_submitted_config": "candidate",
            "nav_diagnostics_before_after": {
                "before": {"reset_replay_steps": 10, "forward_walk_hits": 1},
                "after": {"reset_replay_steps": 7, "forward_walk_hits": 3},
            },
        },
        {},
    )
    assert nav["status"] == "nav_fix_context"
    assert nav["reset_replay_steps_delta"] == -3.0
    assert mod._stop_after_levelup_delta(None, {})["status"] == "missing_or_excluded"  # noqa: SLF001

    a2 = mod._a2_l1_l2_barrier_diagnosis(  # noqa: SLF001
        {
            "barrier_diagnosis": {
                "root_cause": "depth_cap",
                "actionable_next_step": "build level-conditioned goal induction",
                "new_win_condition_likely": True,
                "induction_not_engaged": True,
            }
        },
        {},
    )
    assert a2["cleanly_reportable"] is True
    assert a2["what_blocks_deeper_levels"] == "depth_cap"
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
    fallback_a3 = mod._a3_levelup(  # noqa: SLF001
        {
            "offline_reproduced": True,
            "target_game": "zz99",
            "target_level": 2,
            "reproduction_gate": {"reproduced": True},
            "registry_update": {
                "prior_total_declared": 6,
                "new_total_declared": 7,
                "banked_levels": 1,
            },
        },
        {},
        {"reproducible_total_levels": 0},
    )
    assert fallback_a3["current_total"] == 7
    assert fallback_a3["delta"] == 1
