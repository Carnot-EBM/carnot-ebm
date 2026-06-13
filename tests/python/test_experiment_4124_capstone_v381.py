"""Tests for Exp 4124 .381 capstone aggregation.

Spec refs: REQ-CAPSTONE-4124, SCENARIO-CAPSTONE-4124.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v381_4124 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_payloads() -> dict[int, JsonDict]:
    return {
        4116: {
            "honest_verdict": "complete: val=0.0500 still_below_0.87",
            "val_exact_accuracy": 0.05,
            "duration_s": 90.0,
        },
        4117: {
            "honest_verdict": "complete: val=0.1500 delta=0.1000 improved",
            "pass1": {"val_exact_accuracy": 0.05},
            "pass1_val_exact_accuracy": 0.05,
            "val_exact_accuracy": 0.15,
            "val_delta_vs_pass1": 0.10,
            "accumulation_stalled": False,
            "cumulative_epochs": 4200,
            "duration_s": 91.0,
        },
        4118: {
            "honest_verdict": "complete: val=0.8700 matched_0.87",
            "pass2": {"val_exact_accuracy": 0.15},
            "val_exact_accuracy": 0.87,
            "matches_published_087": True,
            "total_cumulative_epochs": 4300,
            "duration_s": 92.0,
            "branch_taken": "train",
        },
        4119: {
            "honest_verdict": "success: verifier_value_added_A_gt_B_ci95_excludes_zero",
            "graft_deferred": False,
            "verifier_value_added": True,
            "verifier_value_added_meaningful": True,
            "rft_vs_ablation_delta": {
                "metric": "heldout_exact_accuracy",
                "delta": 0.12,
                "ci95": [0.02, 0.20],
            },
            "rerank_lift_vs_vote": {
                "metric": "pass@1_exact_accuracy",
                "delta": 0.06,
                "ci95": [0.01, 0.11],
            },
        },
        4120: {
            "honest_verdict": "success: thirteenth_game_solved_fixture",
            "prior_total_games_solved": 12,
            "total_games_solved": 13,
            "game_solved": True,
            "real_env_confirmed": True,
            "levels_completed": 1,
            "failure_reason": "",
        },
        4121: {
            "honest_verdict": "complete: sota_ingestion_trm_baseline_graft_mapped",
            "flagged_for_v382": "verifier_guided_adaptive_candidate_expansion_over_resumed_trm",
            "methods_mapped": [
                {"name": "TRM resumable Sudoku baseline gate", "arxiv_id": "2510.04871"}
            ],
        },
        4122: {
            "honest_verdict": "complete: registry_gaps_reconciled_to_v381_truth",
            "gaps_updated": [
                "GAP-SUDOKU-BASELINE-REPRODUCTION-4118",
                "GAP-SUDOKU-EXECUTABLE-VERIFIER-4119",
            ],
            "regression_guard_passed": True,
        },
        4123: {
            "honest_verdict": "complete: hardware_continuity",
            "kv260_terminal_confirmed": True,
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
            "gatemate_step_taken": "blocked_gatemate_unreachable",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4124_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4124: OpenSpec declares the .381 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4124" in spec
    assert "SCENARIO-CAPSTONE-4124" in spec
    assert "baseline_still_accumulating_v382_continues" in spec
    assert "baseline_val_trajectory" in spec
    assert "flagged_adversarial:true" in spec
    assert "sha256" in spec


def test_scenario_capstone_4124_current_artifacts_emit_still_accumulating() -> None:
    """SCENARIO-CAPSTONE-4124: current flagged upstreams are excluded."""

    artifact = mod.build_artifact(Path.cwd(), started_s=10.0, now_s=12.0)

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "baseline_still_accumulating_v382_continues"
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v381_baseline_still_accumulating_v382_continues_"
        "val_climbed1_bounded1_baseline0870_graft_deferred_by_baseline_not_reproduced_"
        "games12_flagged_skipped4"
    )
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["headline_answers"] == {
        "resume_val_climbed": True,
        "bounded_runs_under_cap": True,
        "resume_mechanism_status": "climbed_and_bounded",
        "exp4118_matches_published_087": False,
        "exp4119_graft_or_defer": "deferred_by_baseline_not_reproduced",
        "exp4119_verifier_value_added": False,
        "total_arc_games_solved": 12,
    }
    assert artifact["baseline_val_trajectory"]["values"] == pytest.approx(
        [0.08541666716337204, 0.09661458432674408, 0.10598958283662796]
    )
    assert artifact["baseline_val_trajectory"]["climbed"] is True
    assert artifact["baseline_val_trajectory"]["bounded_runs_under_cap"] is True
    assert artifact["baseline_reproduction"]["matches_published_087"] is False
    assert artifact["baseline_reproduction"]["val_exact_accuracy"] == pytest.approx(
        0.10598958283662796
    )
    assert artifact["sudoku_verifier_graft"]["status"] == "deferred_by_baseline_not_reproduced"
    assert (
        artifact["sudoku_verifier_graft"]["exp4119_artifact_status"]
        == "skipped_flagged_adversarial"
    )
    assert artifact["sudoku_verifier_graft"]["verifier_value_added"] is False
    assert artifact["total_arc_games_solved"] == 12
    assert artifact["arc_games"]["status"] == "skipped_flagged_adversarial"

    skipped = artifact["flagged_artifacts_skipped"]
    assert [row["experiment_id"] for row in skipped] == [4116, 4119, 4120, 4122]
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
    assert provenance[4116]["fields_imported"] == []
    assert provenance[4119]["fields_imported"] == []
    assert provenance[4122]["fields_imported"] == []
    assert "val_exact_accuracy" in provenance[4117]["fields_imported"]
    assert "matches_published_087" in provenance[4118]["fields_imported"]
    assert provenance[4120]["fields_imported"] == []


def test_req_capstone_4124_clean_fixture_validates_graft(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4124: clean baseline and positive graft validate the result."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "baseline_reproduced_graft_validated"
    assert artifact["honest_verdict"].startswith(
        "success: capstone_v381_baseline_reproduced_graft_validated_"
        "val_climbed1_bounded1_baseline0871_graft_verifier_value_added_games13_"
        "flagged_skipped0"
    )
    assert artifact["baseline_reproduction"]["matches_published_087"] is True
    assert artifact["sudoku_verifier_graft"]["status"] == "verifier_value_added"
    assert artifact["sudoku_verifier_graft"]["rft_vs_ablation_delta"] == 0.12
    assert artifact["sudoku_verifier_graft"]["rft_vs_ablation_ci95"] == [0.02, 0.20]
    assert artifact["total_arc_games_solved"] == 13
    assert artifact["flagged_artifacts_skipped"] == []
    provenance = {row["experiment_id"]: row for row in artifact["upstream_provenance"]}
    assert "verifier_value_added" in provenance[4119]["fields_imported"]


def test_req_capstone_4124_other_headline_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4124: null, deferred, and stalled branches stay distinct."""

    payloads = _clean_payloads()
    payloads[4119]["verifier_value_added"] = False
    payloads[4119]["rft_vs_ablation_delta"]["ci95"] = [-0.02, 0.03]
    _write_default_artifacts(tmp_path, payloads)
    null = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.1)
    assert null["headline_outcome"] == "baseline_reproduced_graft_null"
    assert null["sudoku_verifier_graft"]["status"] == "null_or_inconclusive"

    payloads = _clean_payloads()
    payloads[4119]["graft_deferred"] = True
    payloads[4119]["verifier_value_added"] = False
    _write_default_artifacts(tmp_path, payloads)
    deferred = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.1)
    assert deferred["headline_outcome"] == "baseline_reproduced_graft_deferred"
    assert deferred["sudoku_verifier_graft"]["status"] == "graft_deferred"

    payloads = _clean_payloads()
    payloads[4118]["val_exact_accuracy"] = 0.14
    payloads[4118]["matches_published_087"] = False
    _write_default_artifacts(tmp_path, payloads)
    stalled = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.1)
    assert stalled["headline_outcome"] == "resume_mechanism_stalled"
    assert stalled["baseline_val_trajectory"]["climbed"] is False

    payloads = _clean_payloads()
    payloads[4118]["duration_s"] = mod.BOUNDED_RUN_CAP_S + 1.0
    _write_default_artifacts(tmp_path, payloads)
    over_cap = mod.build_artifact(tmp_path, started_s=5.0, now_s=5.1)
    assert over_cap["headline_outcome"] == "resume_mechanism_stalled"
    assert over_cap["baseline_val_trajectory"]["bounded_runs_under_cap"] is False


def test_scenario_capstone_4124_write_and_validate(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4124: write_artifact emits the deliverable JSON."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4124_capstone_v381.json"),
        started_s=6.0,
        now_s=6.5,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == 4124
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)


def test_req_capstone_4124_validation_rejects_drift(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4124: validation protects the headline and provenance."""

    _write_default_artifacts(tmp_path, _clean_payloads())
    artifact = mod.build_artifact(tmp_path, started_s=7.0, now_s=7.1)

    artifact["headline_outcome"] = "not_enumerated"
    with pytest.raises(ValueError, match="headline_outcome"):
        mod.validate_artifact(artifact)

    artifact["headline_outcome"] = "baseline_reproduced_graft_validated"
    artifact["upstream_provenance"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(artifact)

    artifact["upstream_provenance"][0]["sha256"] = "a" * 64
    artifact["baseline_val_trajectory"]["values"] = "bad"
    with pytest.raises(ValueError, match="baseline_val_trajectory"):
        mod.validate_artifact(artifact)


def test_req_capstone_4124_helper_edge_states(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4124: helper states keep skipped and missing inputs explicit."""

    assert mod.bool_metric({"x": 1}, "x") is False
    assert mod.int_metric({"x": True}, "x") == 0
    assert mod.float_metric({"x": "0.1"}, "x") == 0.0
    assert mod.list_float_metric({"x": ["bad", 0.1]}, "x") == [0.1]
    assert mod.list_float_metric({"x": "bad"}, "x") == []
    assert mod._trajectory_point("passx", 4124, "missing", None) is None

    nested_fallback = mod.baseline_val_trajectory(
        {
            4117: {
                "pass1": {"val_exact_accuracy": 0.1},
                "val_exact_accuracy": 0.2,
                "duration_s": 1.0,
            },
            4118: {"val_exact_accuracy": 0.3, "duration_s": 1.0},
        }
    )
    assert nested_fallback["points"][0]["source_field"] == "pass1.val_exact_accuracy"
    assert mod.baseline_val_trajectory({})["status"] == "missing_clean_resume_metrics"

    assert (
        mod.baseline_answer(None, trajectory={"values": []}, was_skipped=False)["status"]
        == "missing"
    )
    assert (
        mod.baseline_answer(
            {"flagged_adversarial": True}, trajectory={"values": []}, was_skipped=True
        )["status"]
        == "skipped_flagged_adversarial"
    )
    assert (
        mod.graft_answer(None, baseline_matches=False, was_skipped=False)["status"]
        == "deferred_by_baseline_not_reproduced"
    )
    assert mod.graft_answer(None, baseline_matches=True, was_skipped=False)["status"] == "missing"
    assert (
        mod.graft_answer({"flagged_adversarial": True}, baseline_matches=True, was_skipped=True)[
            "status"
        ]
        == "skipped_flagged_adversarial"
    )
    assert (
        mod.arc_games_answer({"flagged_adversarial": True}, was_skipped=True)["status"]
        == "skipped_flagged_adversarial"
    )
    assert mod.arc_games_answer(None, was_skipped=False)["status"] == "missing"
    assert (
        mod.arc_games_answer({"game_solved": False}, was_skipped=False)["status"]
        == "measured_no_new_solve"
    )

    payloads = _clean_payloads()
    payloads.pop(4123)
    _write_default_artifacts(tmp_path, payloads)
    artifact = mod.build_artifact(tmp_path, started_s=8.0, now_s=8.1)
    assert artifact["missing_upstream_artifacts"] == [{"experiment_id": 4123}]
    assert {row["experiment_id"] for row in artifact["upstream_provenance"]} == set(
        mod.UPSTREAM_IDS
    ) - {4123}
