"""Tests for Exp 4144 .383 capstone aggregation.

Spec refs: REQ-CAPSTONE-4144, SCENARIO-CAPSTONE-4144.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v383_4144 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_payloads() -> dict[int, JsonDict]:
    return {
        4135: {
            "honest_verdict": "complete: pass1_val_0.55",
            "flagged_adversarial": False,
            "pass_index": 1,
            "val_exact_accuracy": 0.55,
            "delta_vs_previous": 0.271827656031,
            "matches_published_087": False,
        },
        4136: {
            "honest_verdict": "complete: pass2_val_0.72",
            "flagged_adversarial": False,
            "pass_index": 2,
            "val_exact_accuracy": 0.72,
            "delta_vs_previous": 0.17,
            "matches_published_087": False,
        },
        4137: {
            "honest_verdict": "complete: pass3_val_0.84",
            "flagged_adversarial": False,
            "pass_index": 3,
            "val_exact_accuracy": 0.84,
            "delta_vs_previous": 0.12,
            "matches_published_087": False,
        },
        4138: {
            "honest_verdict": "complete: pass4_val_0.872",
            "flagged_adversarial": False,
            "baseline_status": "faithful",
            "pass_index": 4,
            "val_exact_accuracy": 0.872,
            "matches_published_087": True,
            "near_faithful_080": True,
            "val_trajectory_383": [
                {
                    "experiment": "experiment_4127_sudoku_extreme_accumulate_fixed",
                    "label": ".382_anchor",
                    "pass_index": 0,
                    "status": "measured",
                    "val_exact_accuracy": 0.278172343969,
                    "delta_vs_previous": None,
                },
                {
                    "experiment": "experiment_4135_sudoku_accumulate_pass1_fixed_lr",
                    "label": ".383_pass1",
                    "pass_index": 1,
                    "status": "measured",
                    "val_exact_accuracy": 0.55,
                    "delta_vs_previous": 0.271827656031,
                },
                {
                    "experiment": "experiment_4136_sudoku_accumulate_pass2_fixed_lr",
                    "label": ".383_pass2",
                    "pass_index": 2,
                    "status": "measured",
                    "val_exact_accuracy": 0.72,
                    "delta_vs_previous": 0.17,
                },
                {
                    "experiment": "experiment_4137_sudoku_accumulate_pass3_fixed_lr",
                    "label": ".383_pass3",
                    "pass_index": 3,
                    "status": "measured",
                    "val_exact_accuracy": 0.84,
                    "delta_vs_previous": 0.12,
                },
                {
                    "experiment": "experiment_4138_sudoku_accumulate_pass4_convergence_check",
                    "label": ".383_pass4",
                    "pass_index": 4,
                    "status": "measured",
                    "val_exact_accuracy": 0.872,
                    "delta_vs_previous": 0.032,
                },
            ],
        },
        4139: {
            "honest_verdict": "success: transferable_verifier_value_added",
            "flagged_adversarial": False,
            "baseline_matches_published_087": True,
            "baseline_near_faithful_080": True,
            "headroom_present": True,
            "verifier_value_added": True,
            "verifier_value_added_basis": ["ensemble_rerank_lift_vs_vote", "rft_vs_ablation_delta"],
            "ensemble_rerank_lift_vs_vote": {
                "metric": "pass@1_exact_accuracy",
                "delta": 0.07,
                "ci95": [0.02, 0.12],
                "meaningful": True,
                "uses_exact_validity_check": False,
            },
            "rft_vs_ablation_delta": {
                "metric": "heldout_exact_accuracy",
                "delta": 0.09,
                "ci95": [0.01, 0.17],
                "status": "measured",
                "n_matched": 64,
            },
            "executable_verifier_is_oracle": True,
            "executable_oracle_upper_bound": {
                "metric": "pass@1_exact_accuracy",
                "delta": 0.4,
                "ci95": [0.3, 0.5],
                "interpretation": "oracle_upper_bound_not_verifier_value",
            },
        },
        4140: {
            "honest_verdict": "complete: incremental_progress_one_level",
            "total_levels_solved": 14,
            "total_games_solved": 13,
            "prior_total_levels_solved": 13,
            "new_levels_solved_this_task": 1,
            "real_env_confirmed": True,
            "verifier_validated": True,
            "target_game": "r11l-495a7899",
            "target_level": 5,
        },
        4141: {
            "honest_verdict": "complete: sota_ingestion_recursive_reasoner_verifier_mapped",
            "methods_mapped": [{"name": "recursive verifier"}],
            "flagged_for_v384": "diffusiongemma_if_transferable_value_added",
        },
        4142: {
            "honest_verdict": "complete: registry_gaps_reconciled_to_value_added",
            "flagged_adversarial": False,
            "regression_guard_passed": True,
            "diffusiongemma_gate_state": {
                "state": "unlocked",
                "verifier_value_added": True,
                "uses_executable_oracle_upper_bound": False,
            },
            "sudoku_baseline": {"status": "baseline_reproduced"},
            "sudoku_decisive_graft": {"status": "verifier_value_added"},
        },
        4143: {
            "honest_verdict": "complete: hardware_continuity_4143",
            "kv260_terminal_confirmed": True,
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
            "gatemate_step_taken": "blocked_gatemate_unreachable",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4144_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4144: OpenSpec declares the .383 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4144" in spec
    assert "SCENARIO-CAPSTONE-4144" in spec
    assert "baseline_config_blocked" in spec
    assert "verifier_value_added_verdict" in spec
    assert "diffusiongemma_unlocks" in spec
    assert "executable_oracle_upper_bound" in spec
    assert "sha256" in spec


def test_scenario_capstone_4144_current_artifacts_emit_config_blocked() -> None:
    """SCENARIO-CAPSTONE-4144: current flagged upstreams are excluded."""

    artifact = mod.build_artifact(Path.cwd(), started_s=10.0, now_s=12.0)

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "baseline_config_blocked"
    assert artifact["honest_verdict"].startswith(
        "blocked: capstone_v383_baseline_config_blocked_baseline_converged0_"
        "near_faithful0_headroom0_verifier_deferred_diffusiongemma0_levels13_"
        "flagged_skipped6"
    )
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["headline_answers"] == {
        "exp4138_fixed_lr_baseline_converged": False,
        "exp4138_matches_published_087": None,
        "exp4138_near_faithful_080": None,
        "exp4139_headroom_present": None,
        "exp4139_transferable_verifier_value_added": False,
        "total_arc_levels_solved": 13,
        "diffusiongemma_unlocks": False,
    }
    assert artifact["baseline_val_trajectory"]["status"] == "baseline_config_blocked"
    assert artifact["baseline_val_trajectory"]["matches_published_087"] is None
    assert artifact["baseline_val_trajectory"]["near_faithful_080"] is None
    assert artifact["baseline_val_trajectory"]["values"] == pytest.approx([0.278172343969])
    assert artifact["baseline_val_trajectory"]["attempted_passes"] == [
        {"experiment_id": 4135, "pass_index": 1, "included": False, "status": "skipped_flagged_adversarial"},
        {"experiment_id": 4136, "pass_index": 2, "included": False, "status": "skipped_flagged_adversarial"},
        {"experiment_id": 4137, "pass_index": 3, "included": False, "status": "skipped_flagged_adversarial"},
        {"experiment_id": 4138, "pass_index": 4, "included": False, "status": "skipped_flagged_adversarial"},
    ]
    verifier = artifact["verifier_value_added_verdict"]
    assert verifier["status"] == "deferred"
    assert verifier["verifier_value_added"] is False
    assert verifier["uses_executable_oracle_upper_bound_for_gate"] is False
    assert artifact["diffusiongemma_unlocks"] is False
    assert artifact["diffusiongemma_gate_state"]["state"] == "kept_gated"
    assert artifact["total_arc_levels_solved"] == 13
    assert artifact["arc_levels"]["status"] == "measured_no_new_level"

    skipped = artifact["flagged_artifacts_skipped"]
    assert [row["experiment_id"] for row in skipped] == [4135, 4136, 4137, 4138, 4139, 4142]
    for row in skipped:
        expected_sha = hashlib.sha256(
            mod.DEFAULT_UPSTREAM_PATHS[row["experiment_id"]].read_bytes()
        ).hexdigest()
        assert row["sha256"] == expected_sha

    provenance = {row["experiment_id"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.UPSTREAM_IDS)
    for experiment_id, row in provenance.items():
        expected_sha = hashlib.sha256(
            mod.DEFAULT_UPSTREAM_PATHS[experiment_id].read_bytes()
        ).hexdigest()
        assert row["sha256"] == expected_sha
    for experiment_id in [4135, 4136, 4137, 4138, 4139, 4142]:
        assert provenance[experiment_id]["fields_imported"] == []
        assert provenance[experiment_id]["skipped"] is True
    assert "total_levels_solved" in provenance[4140]["fields_imported"]
    assert "methods_mapped" in provenance[4141]["fields_imported"]
    assert "kv260_terminal_confirmed" in provenance[4143]["fields_imported"]


def test_req_capstone_4144_clean_fixture_unlocks_diffusiongemma(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4144: clean faithful baseline plus transferable lift unlocks."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "baseline_converged_verifier_value_added"
    assert artifact["honest_verdict"].startswith(
        "success: capstone_v383_baseline_converged_verifier_value_added_"
        "baseline_converged1_near_faithful1_headroom1_verifier_true_"
        "diffusiongemma1_levels14_flagged_skipped0"
    )
    assert artifact["baseline_val_trajectory"]["converged_toward_087"] is True
    assert artifact["baseline_val_trajectory"]["matches_published_087"] is True
    assert artifact["baseline_val_trajectory"]["values"] == pytest.approx(
        [0.278172343969, 0.55, 0.72, 0.84, 0.872]
    )
    verifier = artifact["verifier_value_added_verdict"]
    assert verifier["status"] == "true"
    assert verifier["verifier_value_added"] is True
    assert verifier["transferable_ensemble_value_added"] is True
    assert verifier["rft_label_deconfound_value_added"] is True
    assert verifier["oracle_context"]["used_for_gate"] is False
    assert artifact["diffusiongemma_unlocks"] is True


def test_req_capstone_4144_null_near_and_accumulating_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4144: null, near-faithful, and accumulating branches stay distinct."""

    payloads = _clean_payloads()
    payloads[4139]["verifier_value_added"] = False
    payloads[4139]["ensemble_rerank_lift_vs_vote"]["delta"] = 0.0
    payloads[4139]["ensemble_rerank_lift_vs_vote"]["ci95"] = [-0.02, 0.03]
    payloads[4139]["ensemble_rerank_lift_vs_vote"]["meaningful"] = False
    payloads[4139]["rft_vs_ablation_delta"]["delta"] = 0.0
    payloads[4139]["rft_vs_ablation_delta"]["ci95"] = [-0.01, 0.02]
    payloads[4139]["verifier_value_added_basis"] = []
    _write_default_artifacts(tmp_path, payloads)
    null = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.1)
    assert null["headline_outcome"] == "baseline_converged_verifier_null_honest"
    assert null["verifier_value_added_verdict"]["status"] == "false-with-headroom"
    assert null["diffusiongemma_unlocks"] is False

    payloads = _clean_payloads()
    payloads[4138]["matches_published_087"] = False
    payloads[4138]["near_faithful_080"] = True
    payloads[4138]["val_exact_accuracy"] = 0.82
    payloads[4138]["val_trajectory_383"][-1]["val_exact_accuracy"] = 0.82
    payloads[4138]["val_trajectory_383"][-1]["delta_vs_previous"] = -0.02
    _write_default_artifacts(tmp_path, payloads)
    near = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.1)
    assert near["headline_outcome"] == "baseline_near_faithful_rft_measured"
    assert near["baseline_val_trajectory"]["near_faithful_080"] is True
    assert near["verifier_value_added_verdict"]["status"] == "true"

    payloads = _clean_payloads()
    payloads[4138]["matches_published_087"] = False
    payloads[4138]["near_faithful_080"] = False
    payloads[4138]["val_exact_accuracy"] = 0.62
    payloads[4138]["val_trajectory_383"][-1]["val_exact_accuracy"] = 0.62
    payloads[4139]["headroom_present"] = False
    _write_default_artifacts(tmp_path, payloads)
    accumulating = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.1)
    assert accumulating["headline_outcome"] == "baseline_accumulating_graft_deferred_v384_continues"
    assert accumulating["verifier_value_added_verdict"]["status"] == "deferred"


def test_scenario_capstone_4144_write_and_validate(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4144: write_artifact emits the deliverable JSON."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4144_capstone_v383.json"),
        started_s=6.0,
        now_s=6.5,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == 4144
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)


def test_req_capstone_4144_validation_and_helper_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4144: validation protects headline and provenance integrity."""

    _write_default_artifacts(tmp_path, _clean_payloads())
    artifact = mod.build_artifact(tmp_path, started_s=7.0, now_s=7.1)

    artifact["headline_outcome"] = "not_enumerated"
    with pytest.raises(ValueError, match="headline_outcome"):
        mod.validate_artifact(artifact)

    artifact["headline_outcome"] = "baseline_converged_verifier_value_added"
    artifact["upstream_provenance"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(artifact)

    artifact["upstream_provenance"][0]["sha256"] = "a" * 64
    artifact["baseline_val_trajectory"]["values"] = "bad"
    with pytest.raises(ValueError, match="baseline_val_trajectory"):
        mod.validate_artifact(artifact)

    artifact["baseline_val_trajectory"]["values"] = [0.1]
    artifact["diffusiongemma_unlocks"] = "bad"
    with pytest.raises(ValueError, match="diffusiongemma_unlocks"):
        mod.validate_artifact(artifact)

    assert mod.bool_metric({"x": 1}, "x") is None
    assert mod.bool_metric({"x": False}, "x") is False
    assert mod.int_metric({"x": True}, "x") == 0
    assert mod.float_metric({"x": "0.1"}, "x") is None
    assert mod.list_float_metric({"x": ["bad", 0.1]}, "x") == [0.1]
    assert mod.list_float_metric({"x": "bad"}, "x") == []
    assert mod.clean_trajectory_points("bad") == []
    assert mod.clean_trajectory_points([42, {"val_exact_accuracy": None}]) == []
    assert mod.baseline_val_trajectory({}, root=tmp_path, skipped_ids=set())["status"] == "missing"
    assert (
        mod.verifier_value_added_answer(
            {"flagged_adversarial": True},
            baseline={"matches_published_087": True, "near_faithful_080": True},
            was_skipped=True,
        )["status"]
        == "deferred"
    )
    assert (
        mod.verifier_value_added_answer(
            {"headroom_present": False},
            baseline={"matches_published_087": True, "near_faithful_080": True},
            was_skipped=False,
        )["reason"]
        == "no_headroom_false_negative_risk"
    )
    assert mod.arc_levels_answer({"flagged_adversarial": True}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.arc_levels_answer(None, was_skipped=False)["status"] == "missing"

    missing_root = tmp_path / "missing_upstream"
    payloads = _clean_payloads()
    payloads.pop(4143)
    _write_default_artifacts(missing_root, payloads)
    missing = mod.build_artifact(missing_root, started_s=8.0, now_s=8.1)
    assert missing["missing_upstream_artifacts"] == [{"experiment_id": 4143}]
    assert {row["experiment_id"] for row in missing["upstream_provenance"]} == set(
        mod.UPSTREAM_IDS
    ) - {4143}
