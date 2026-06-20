"""Tests for Exp 4521 .417 action-efficiency capstone.

Spec refs: REQ-CAPSTONE-4521, SCENARIO-CAPSTONE-4521.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4521_capstone_v417 as mod


JsonDict = dict[str, Any]
REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "capstone" / "spec.md"


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_live_flags(_: Path) -> list[dict[str, Any]]:
    return []


def _a6_live_critical(path: Path) -> list[dict[str, Any]]:
    if path.name == "experiment_4516_integration_8game_gate.json":
        return [{"kind": "TAUTOLOGY", "severity": "critical", "detail": "heldout equals solve"}]
    return []


def _summary_codes(codes: dict[str, int]):
    def _runner(path: Path, _root: Path) -> int:
        return int(codes.get(path.name, 0))

    return _runner


def _payloads(*, a4_wins: bool = True, a3_flagged: bool = True, a6_flagged: bool = True) -> dict[str, JsonDict]:
    selected_weight = 0.5 if a4_wins else 0.0
    selected_actions = 7500.0 if a4_wins else 7760.0
    return {
        "A1_prune": {
            "honest_verdict": "complete: frame_change_prune_solve_rate_guard_failed",
            "median_actions_baseline": 7760.0,
            "median_actions_with_prune": 5000.0,
            "solve_rate_baseline": 4,
            "solve_rate_with_prune": 3,
            "solve_rate_denominator": 8,
            "verifier_is_oracle": False,
        },
        "A2_imitation": {
            "honest_verdict": "complete: imitation_prior_solve_rate_guard_failed",
            "median_actions_baseline": 7760.0,
            "median_actions_with_prior": 5200.0,
            "solve_rate_baseline": 4,
            "solve_rate_with_prior": 3,
            "solve_rate_denominator": 8,
            "verifier_is_oracle": False,
        },
        "A3_adaptive_budget": {
            "honest_verdict": "success: adaptive_budget_median_actions_2984_below_7760",
            "flagged_adversarial": a3_flagged,
            "median_actions_baseline": 7760.0,
            "median_actions_with_adaptive": 2984.0,
            "solve_rate_baseline": 4,
            "solve_rate_with_adaptive": 4,
            "solve_rate_denominator": 8,
            "verifier_is_oracle": False,
        },
        "A4_lazy_best_first": {
            "honest_verdict": (
                "success: lazy_value_weight_0.5_beats_0"
                if a4_wins
                else "complete: lazy_value_weight_null_keep_0"
            ),
            "per_weight_results": {
                "0.0": {
                    "value_weight": 0.0,
                    "heldout_solve_rate": 0.5,
                    "median_actions_on_core": 7760.0,
                    "median_per_game_wall_s": 120.0,
                    "core_solves_preserved": True,
                },
                "0.5": {
                    "value_weight": 0.5,
                    "heldout_solve_rate": 0.5,
                    "median_actions_on_core": selected_actions,
                    "median_per_game_wall_s": 120.0,
                    "core_solves_preserved": True,
                },
            },
            "control_value_weight_0": {
                "value_weight": 0.0,
                "heldout_solve_rate": 0.5,
                "median_actions_on_core": 7760.0,
                "core_solves_preserved": True,
            },
            "chosen_submitted_value_weight": selected_weight,
            "decision": {"selected_value_weight": selected_weight},
            "verifier_is_oracle": False,
        },
        "A5_level_up": {
            "honest_verdict": "success: m0r0_L2_offline_reproduced",
            "target_game": "m0r0",
            "offline_reproduced": True,
            "reproduced_levels": 2,
            "reproduction_gate": {"game": "m0r0", "claimed_level": 2, "reached_level": 2, "reproduced": True},
            "verifier_is_oracle": True,
        },
        "A6_integration": {
            "honest_verdict": "complete: no_lever_beats_7760_honest_null",
            "flagged_adversarial": a6_flagged,
            "median_actions_baseline": 7760.0,
            "median_actions_integrated": 7000.0,
            "solve_rate_integrated": 0.5,
            "heldout_solve_rate": 0.5,
            "verifier_is_oracle": False,
        },
        "scoreboard_context": {
            "honest_verdict": "complete: submitted_agent_scoreboard_refresh_generic_1_of_7_variant_7_of_25_value_weight_0",
            "headline_metrics": {
                "submitted_default_heldout_generic_solve_rate": 1 / 7,
                "submitted_default_heldout_generic_solved": 1,
                "submitted_default_heldout_generic_attempted": 7,
                "variant_transfer_rate": 0.28,
                "variant_transfer_solved": 7,
                "variant_transfer_attempted": 25,
            },
            "leaderboard_submission": False,
            "verifier_is_oracle": False,
        },
    }


def _write_payloads(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)
    registry = root / mod.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        "schema_version: 1\nreproducible_total_levels: 48\nprior_submitted_baseline_levels: 13\n",
        encoding="utf-8",
    )


def test_req_capstone_4521_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4521: OpenSpec declares the .417 capstone contract first."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4521" in spec
    assert "SCENARIO-CAPSTONE-4521" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    assert "scripts/summarize_artifact.py" in spec
    assert "flagged_adversarial:true" in spec
    assert "reproducible_total_levels > 13" in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_capstone_4521_clean_equal_solve_lever_wins_and_flagged_lower_medians_skip(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4521: lower flagged medians never beat a clean equal-solve lever."""

    _write_payloads(tmp_path, _payloads(a4_wins=True, a3_flagged=True, a6_flagged=True))

    artifact = mod.build_artifact(
        tmp_path,
        started_s=10.0,
        now_s=11.0,
        live_flag_runner=_a6_live_critical,
        summarize_runner=_summary_codes({"experiment_4516_integration_8game_gate.json": 2}),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "success: v417_A4_lazy_best_first_median_actions_7500_vs_7760_heldout_0.143"
    assert artifact["median_actions_baseline"] == 7760.0
    assert artifact["median_actions_best_lever"]["lever"] == "A4_lazy_best_first"
    assert artifact["median_actions_best_lever"]["median_actions"] == 7500.0
    assert artifact["action_efficiency_decision"]["beats_7760_at_equal_solve_rate"] is True
    assert artifact["action_efficiency_decision"]["winning_lever"] == "A4_lazy_best_first"
    assert artifact["heldout_solve_rate"] == pytest.approx(1 / 7)
    assert artifact["variant_transfer_rate"] == pytest.approx(0.28)
    assert artifact["reproducible_total_levels"] == 48
    assert artifact["submission_package_ready"] is True
    assert artifact["leaderboard_submission"] is False
    assert artifact["integrated_scorecard"]["status"] == "excluded_flagged_adversarial"
    assert artifact["preconditions_checked"]["a6_integration_artifact_present"] is True
    assert artifact["preconditions_checked"]["a6_summarize_exit_code"] == 2
    assert artifact["preconditions_checked"]["a6_clean_for_aggregation"] is False
    assert artifact["level_up_context"]["status"] == "level_up_context"
    assert artifact["verifier_is_oracle"] is False
    assert any(claim["source"] == "A5_level_up" and claim["verifier_is_oracle"] is True for claim in artifact["verifier_claims"])

    excluded = {row["artifact_key"]: row for row in artifact["flagged_artifacts_excluded"]}
    assert set(excluded) == {"A3_adaptive_budget", "A6_integration"}
    assert excluded["A6_integration"]["live_critical"] is True
    cited = {row["artifact_key"]: row for row in artifact["cited_upstream_artifacts"]}
    assert cited["A3_adaptive_budget"]["fields_imported"] == []
    assert cited["A6_integration"]["fields_imported"] == []
    skipped_row = next(row for row in artifact["per_lever_scorecard"] if row["lever"] == "A3_adaptive_budget")
    assert skipped_row["median_actions"] is None


def test_req_capstone_4521_no_clean_equal_solve_win_keeps_baseline_best(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4521: solve-rate drops cannot claim the action-efficiency win."""

    _write_payloads(tmp_path, _payloads(a4_wins=False, a3_flagged=True, a6_flagged=True))

    artifact = mod.build_artifact(
        tmp_path,
        started_s=20.0,
        now_s=21.0,
        live_flag_runner=_a6_live_critical,
        summarize_runner=_summary_codes({"experiment_4516_integration_8game_gate.json": 2}),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: v417_none_clean_equal_solve_rate_median_actions_7760_vs_7760_heldout_0.143"
    assert artifact["median_actions_best_lever"] == {
        "lever": "none_clean_equal_solve_rate",
        "median_actions": 7760.0,
        "median_actions_delta_vs_baseline": 0.0,
        "action_reduction_vs_baseline": 0.0,
        "heldout_solve_rate": pytest.approx(1 / 7),
        "reason": "no_clean_lever_beat_7760_at_equal_or_better_solve_rate",
    }
    assert artifact["action_efficiency_decision"]["beats_7760_at_equal_solve_rate"] is False
    a1 = next(row for row in artifact["per_lever_scorecard"] if row["lever"] == "A1_prune")
    assert a1["median_actions"] == 5000.0
    assert a1["equal_or_better_solve_rate"] is False
    assert a1["action_efficiency_win"] is False


def test_req_capstone_4521_missing_a6_blocks_without_fabricating_metrics(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4521: missing required A6 produces a blocked artifact."""

    payloads = _payloads()
    payloads.pop("A6_integration")
    _write_payloads(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path,
        started_s=30.0,
        now_s=31.0,
        live_flag_runner=_clean_live_flags,
        summarize_runner=_summary_codes({}),
    )

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_a6_integration_artifact_missing"
    assert artifact["preconditions_checked"]["a6_integration_artifact_present"] is False
    assert artifact["action_efficiency_decision"]["beats_7760_at_equal_solve_rate"] is False
    assert artifact["median_actions_best_lever"]["lever"] == "blocked"
    assert artifact["flagged_artifacts_excluded"] == []
    assert artifact["cited_upstream_artifacts"]


def test_req_capstone_4521_write_path_and_validation_guards(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4521: writer emits JSON and schema validation fails closed."""

    _write_payloads(tmp_path, _payloads(a4_wins=True, a3_flagged=True, a6_flagged=True))
    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/out.json"),
        started_s=40.0,
        now_s=41.0,
        live_flag_runner=_a6_live_critical,
        summarize_runner=_summary_codes({"experiment_4516_integration_8game_gate.json": 2}),
    )

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["honest_verdict"].startswith("success:")
    assert written["result_path"] == "results/out.json"
    assert mod.run(
        root=tmp_path,
        write=False,
        started_s=41.0,
        now_s=42.0,
        live_flag_runner=_a6_live_critical,
        summarize_runner=_summary_codes({"experiment_4516_integration_8game_gate.json": 2}),
    )["duration_s"] == 1.0
    mod.run(
        root=tmp_path,
        write=True,
        started_s=42.0,
        now_s=43.0,
        live_flag_runner=_a6_live_critical,
        summarize_runner=_summary_codes({"experiment_4516_integration_8game_gate.json": 2}),
    )
    assert (tmp_path / mod.OUTPUT_REL_PATH).exists()

    invalid_cases = [
        ("__delete__honest_verdict", None, "missing required field"),
        ("honest_verdict", "not-terminal", "honest_verdict"),
        ("inference_substrate", "live_llm_inference", "inference_substrate"),
        ("median_actions_best_lever", [], "median_actions_best_lever"),
        ("median_actions_baseline", 1.0, "median_actions_baseline"),
        ("heldout_solve_rate", True, "heldout_solve_rate"),
        ("submission_package_ready", "yes", "submission_package_ready"),
        ("flagged_artifacts_excluded", {}, "flagged_artifacts_excluded"),
        ("preconditions_checked", [], "preconditions_checked"),
        ("per_lever_scorecard", {}, "per_lever_scorecard"),
        ("integrated_scorecard", [], "integrated_scorecard"),
        ("action_efficiency_decision", [], "action_efficiency_decision"),
        ("variant_transfer_rate", True, "variant_transfer_rate"),
        ("reproducible_total_levels", True, "reproducible_total_levels"),
        ("verifier_is_oracle", True, "verifier_is_oracle"),
        ("verifier_claims", {}, "verifier_claims"),
        ("cited_upstream_artifacts", {}, "cited_upstream_artifacts"),
        ("random_seed", 1, "random_seed"),
        ("reproducibility_checksum", "bad", "reproducibility_checksum"),
        ("field_principles", {}, "field_principles"),
        ("leaderboard_submission", True, "leaderboard_submission"),
        ("upstream_provenance", {}, "upstream_provenance"),
        ("upstream_provenance", [1], "upstream provenance row"),
        ("upstream_provenance", [{"sha256": "bad", "skipped": False, "fields_imported": []}], "invalid sha256"),
        ("__gated_on__", True, "gated_on"),
        ("__checksum_mismatch__", True, "reproducibility_checksum"),
        ("__skipped_imports__", True, "skipped upstreams"),
        ("__claim_missing_bool__", True, "verifier_claims"),
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
            invalid["upstream_provenance"][0]["fields_imported"] = ["median_actions_with_prune"]
        elif field == "__claim_missing_bool__":
            invalid["verifier_claims"] = [{"source": "A1_prune"}]
        else:
            invalid[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(invalid)


def test_req_capstone_4521_helper_branches_cover_defensive_paths(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4521: helper branches keep missing/skipped inputs fail-closed."""

    assert mod._rate(None, 8) is None  # noqa: SLF001
    assert mod.load_registry_totals(tmp_path / "missing") == {
        "registry_path": str(mod.REGISTRY_RELATIVE_PATH),
        "registry_present": False,
        "reproducible_total_levels": 0,
        "prior_submitted_baseline_levels": 13,
    }
    assert mod._direct_lever_row(  # noqa: SLF001
        lever="A1_prune",
        payload=None,
        skipped=False,
        median_field="median_actions_with_prune",
        solve_field="solve_rate_with_prune",
    )["status"] == "missing_or_excluded"
    assert mod._a4_scorecard(None, True)["status"] == "excluded_flagged_adversarial"  # noqa: SLF001
    assert mod._a4_scorecard(None, False)["status"] == "missing_or_excluded"  # noqa: SLF001
    assert mod._a4_scorecard(  # noqa: SLF001
        {
            "decision": {"selected_value_weight": 0.5},
            "per_weight_results": {
                "0.0": {"median_actions_on_core": 7760.0, "heldout_solve_rate": 0.5, "core_solves_preserved": True},
                "0.5": {"median_actions_on_core": 7700.0, "heldout_solve_rate": 0.5, "core_solves_preserved": True},
            },
        },
        False,
    )["selected_value_weight"] == 0.5
    assert mod._integrated_scorecard(None, False)["status"] == "missing_or_excluded"  # noqa: SLF001
    assert mod._integrated_scorecard(  # noqa: SLF001
        {"median_actions_integrated": 7000.0, "solve_rate_integrated": 0.5},
        False,
    )["status"] == "integrated_action_efficiency_win"
    assert mod._level_up_context({}, True)["status"] == "excluded_flagged_adversarial"  # noqa: SLF001
    assert mod._level_up_context(None, False)["status"] == "missing_or_excluded"  # noqa: SLF001
    assert mod._scoreboard_context(None, False)["heldout_solve_rate"] == 0.0  # noqa: SLF001
    assert mod._verifier_claims(  # noqa: SLF001
        {"A1_prune": None, "A2_imitation": {"verifier_is_oracle": True}},
        {},
    ) == [
        {
            "source": "A2_imitation",
            "experiment_id": 4512,
            "verifier_is_oracle": True,
            "skipped": False,
        }
    ]
