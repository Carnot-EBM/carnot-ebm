"""Tests for Exp 4133 .382 capstone aggregation.

Spec refs: REQ-CAPSTONE-4133, SCENARIO-CAPSTONE-4133.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v382_4133 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_payloads() -> dict[int, JsonDict]:
    return {
        4126: {
            "honest_verdict": "complete: lr_resume_continuous_first_lr=1e-04",
            "lr_continuous_across_resume": True,
            "validation_first_lr": 0.0001,
            "fresh_warmup_lr": 0.00000245,
            "prior_pass_last_lr": 0.00000495,
            "stable_checkpoint_path": "/tmp/stable.ckpt",
            "manual_lr_step_restored": 4300,
            "duration_s": 350.0,
        },
        4127: {
            "honest_verdict": "complete: val=0.8700 reproduced",
            "matches_published_087": True,
            "val_trajectory": [
                {
                    "kind": "starting_baseline",
                    "pass_index": 0,
                    "val_exact_accuracy": 0.105989582837,
                    "delta_vs_previous": None,
                },
                {
                    "kind": "fixed_lr_resume_pass",
                    "pass_index": 1,
                    "val_exact_accuracy": 0.87,
                    "delta_vs_previous": 0.764010417163,
                    "checkpoint_reload_ok": True,
                    "duration_s": 1200.0,
                },
            ],
            "per_pass_delta_vs_v381": {
                "beats_v381": True,
                "comparison": "faster_than_v381",
                "deltas": [0.764010417163],
                "mean_delta": 0.764010417163,
                "reference_delta": 0.01,
            },
            "stable_checkpoint_path": "/tmp/stable.ckpt",
            "duration_s": [1200.0],
            "acceptance_gate_passed": True,
            "contiguous_run_recommendation": None,
        },
        4128: {
            "honest_verdict": "success: verifier_value_added_A_gt_B_ci95_excludes_zero",
            "baseline_matches_published_087": True,
            "baseline_val_exact_accuracy": 0.87,
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
            "duration_s": 10.0,
        },
        4129: {
            "honest_verdict": "success: fourteenth_game_solved_fixture",
            "prior_total_games_solved": 12,
            "total_games_solved": 13,
            "game_solved": True,
            "real_env_confirmed": True,
            "levels_completed": 1,
            "target_game": "bp35-0a0ad940",
            "duration_s": 1.0,
        },
        4130: {
            "honest_verdict": "complete: sota_ingestion_resumable_training_mapped",
            "methods_mapped": [
                {
                    "name": "PyTorch Lightning full-state checkpoint resume",
                    "url": "https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html",
                }
            ],
        },
        4131: {
            "honest_verdict": "complete: registry_gaps_reconciled_to_v382_truth",
            "regression_guard_passed": True,
            "lr_resume_fix": {"status": "fixed_lr_resume_continuous"},
            "sudoku_baseline": {"status": "baseline_reproduced"},
            "sudoku_graft": {"status": "verifier_value_added"},
            "duration_s": 0.5,
        },
        4132: {
            "honest_verdict": "complete: hardware_continuity_4132",
            "kv260_terminal_confirmed": True,
            "per_board_reachability": {"kv260": True, "gatemate": False, "polarfire": True},
            "gatemate_step_taken": "blocked_gatemate_unreachable",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
            "duration_s": 2.0,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4133_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4133: OpenSpec declares the .382 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4133" in spec
    assert "SCENARIO-CAPSTONE-4133" in spec
    assert "lr_fixed_accumulating_v383_continues" in spec
    assert "baseline_val_trajectory" in spec
    assert "flagged_adversarial:true" in spec
    assert "sha256" in spec


def test_scenario_capstone_4133_current_artifacts_emit_lr_fixed_accumulating() -> None:
    """SCENARIO-CAPSTONE-4133: current flagged upstreams are excluded."""

    artifact = mod.build_artifact(Path.cwd(), started_s=10.0, now_s=12.0)

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "lr_fixed_accumulating_v383_continues"
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v382_lr_fixed_accumulating_v383_continues_"
        "lr_fixed1_accelerated1_baseline0870_graft_deferred_by_baseline_not_reproduced_"
        "games13_flagged_skipped2"
    )
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["headline_answers"] == {
        "exp4126_lr_resume_fix_landed": True,
        "exp4127_corrected_schedule_accelerated": True,
        "exp4127_matches_published_087": False,
        "exp4128_graft_or_defer": "deferred_by_baseline_not_reproduced",
        "exp4128_verifier_value_added": False,
        "total_arc_games_solved": 13,
    }
    assert artifact["baseline_val_trajectory"]["values"] == pytest.approx(
        [0.02317708358168602, 0.105989582837, 0.278172343969]
    )
    assert artifact["baseline_val_trajectory"]["accelerated_vs_v381"] is True
    assert artifact["baseline_val_trajectory"]["final_val_exact_accuracy"] == pytest.approx(
        0.278172343969
    )
    assert artifact["lr_resume_fix"]["lr_continuous_across_resume"] is True
    assert artifact["baseline_reproduction"]["matches_published_087"] is False
    assert artifact["baseline_reproduction"]["status"] == "still_accumulating"
    assert artifact["sudoku_verifier_graft"]["status"] == "deferred_by_baseline_not_reproduced"
    assert (
        artifact["sudoku_verifier_graft"]["exp4128_artifact_status"]
        == "skipped_flagged_adversarial"
    )
    assert artifact["sudoku_verifier_graft"]["verifier_value_added"] is False
    assert artifact["total_arc_games_solved"] == 13
    assert artifact["arc_games"]["status"] == "new_game_solved"

    skipped = artifact["flagged_artifacts_skipped"]
    assert [row["experiment_id"] for row in skipped] == [4128, 4131]
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
    assert provenance[4128]["fields_imported"] == []
    assert provenance[4131]["fields_imported"] == []
    assert "lr_continuous_across_resume" in provenance[4126]["fields_imported"]
    assert "val_trajectory" in provenance[4127]["fields_imported"]
    assert "total_games_solved" in provenance[4129]["fields_imported"]


def test_req_capstone_4133_clean_fixture_validates_graft(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4133: clean baseline and positive graft validate the result."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "lr_fixed_baseline_reproduced_graft_validated"
    assert artifact["honest_verdict"].startswith(
        "success: capstone_v382_lr_fixed_baseline_reproduced_graft_validated_"
        "lr_fixed1_accelerated1_baseline0871_graft_verifier_value_added_games13_"
        "flagged_skipped0"
    )
    assert artifact["baseline_reproduction"]["matches_published_087"] is True
    assert artifact["sudoku_verifier_graft"]["status"] == "verifier_value_added"
    assert artifact["sudoku_verifier_graft"]["rft_vs_ablation_delta"] == 0.12
    assert artifact["sudoku_verifier_graft"]["rft_vs_ablation_ci95"] == [0.02, 0.20]
    assert artifact["total_arc_games_solved"] == 13
    assert artifact["flagged_artifacts_skipped"] == []
    provenance = {row["experiment_id"]: row for row in artifact["upstream_provenance"]}
    assert "verifier_value_added" in provenance[4128]["fields_imported"]


def test_req_capstone_4133_other_headline_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4133: null, LR-failed, and blocked branches stay distinct."""

    payloads = _clean_payloads()
    payloads[4128]["verifier_value_added"] = False
    payloads[4128]["rft_vs_ablation_delta"]["ci95"] = [-0.02, 0.03]
    _write_default_artifacts(tmp_path, payloads)
    null = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.1)
    assert null["headline_outcome"] == "lr_fixed_baseline_reproduced_graft_null"
    assert null["sudoku_verifier_graft"]["status"] == "null_or_inconclusive"

    payloads = _clean_payloads()
    payloads[4126]["lr_continuous_across_resume"] = False
    payloads[4127]["contiguous_run_recommendation"] = "contiguous_run_recommended"
    _write_default_artifacts(tmp_path, payloads)
    lr_failed = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.1)
    assert lr_failed["headline_outcome"] == "lr_fix_failed_contiguous_run_recommended"
    assert lr_failed["lr_resume_fix"]["status"] == "lr_fix_failed"

    payloads = _clean_payloads()
    payloads[4127]["matches_published_087"] = False
    payloads[4127]["val_trajectory"][1]["val_exact_accuracy"] = 0.1061
    payloads[4127]["val_trajectory"][1]["delta_vs_previous"] = 0.0001
    payloads[4127]["per_pass_delta_vs_v381"]["beats_v381"] = False
    payloads[4127]["per_pass_delta_vs_v381"]["mean_delta"] = 0.0001
    _write_default_artifacts(tmp_path, payloads)
    blocked = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.1)
    assert blocked["headline_outcome"] == "baseline_still_blocked"
    assert blocked["baseline_val_trajectory"]["accelerated_vs_v381"] is False


def test_scenario_capstone_4133_write_and_validate(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4133: write_artifact emits the deliverable JSON."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4133_capstone_v382.json"),
        started_s=6.0,
        now_s=6.5,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == 4133
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)


def test_req_capstone_4133_validation_and_helper_edges(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4133: validation protects headline and provenance integrity."""

    _write_default_artifacts(tmp_path, _clean_payloads())
    artifact = mod.build_artifact(tmp_path, started_s=7.0, now_s=7.1)

    artifact["headline_outcome"] = "not_enumerated"
    with pytest.raises(ValueError, match="headline_outcome"):
        mod.validate_artifact(artifact)

    artifact["headline_outcome"] = "lr_fixed_baseline_reproduced_graft_validated"
    artifact["upstream_provenance"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(artifact)

    artifact["upstream_provenance"][0]["sha256"] = "a" * 64
    artifact["baseline_val_trajectory"]["values"] = "bad"
    with pytest.raises(ValueError, match="baseline_val_trajectory"):
        mod.validate_artifact(artifact)

    assert mod.bool_metric({"x": 1}, "x") is False
    assert mod.int_metric({"x": True}, "x") == 0
    assert mod.float_metric({"x": "0.1"}, "x") == 0.0
    assert mod.list_float_metric({"x": ["bad", 0.1]}, "x") == [0.1]
    assert mod.list_float_metric({"x": "bad"}, "x") == []
    assert mod.clean_val_points([]) == []
    assert mod.clean_val_points("bad") == []
    assert mod.clean_val_points([42, {"val_exact_accuracy": None}]) == []

    bad_reference_root = tmp_path / "bad_reference"
    _write_json(
        bad_reference_root / mod.INITIAL_REFERENCE_PATH,
        {"honest_verdict": "complete: missing_metric", "reproduced_exact_accuracy": None},
    )
    assert mod.initial_reference_point(bad_reference_root) is None
    assert (
        mod.baseline_val_trajectory({}, root=tmp_path / "empty")["status"]
        == "missing_clean_val_trajectory"
    )
    assert (
        mod.lr_fix_answer({"flagged_adversarial": True}, was_skipped=True)["status"]
        == "skipped_flagged_adversarial"
    )
    assert mod.lr_fix_answer(None, was_skipped=False)["status"] == "missing"
    assert (
        mod.baseline_answer({}, trajectory={"accelerated_vs_v381": False}, was_skipped=True)[
            "status"
        ]
        == "skipped_flagged_adversarial"
    )
    assert (
        mod.baseline_answer(None, trajectory={"accelerated_vs_v381": False}, was_skipped=False)[
            "status"
        ]
        == "missing"
    )
    assert (
        mod.graft_answer({"flagged_adversarial": True}, baseline_matches=True, was_skipped=True)[
            "status"
        ]
        == "skipped_flagged_adversarial"
    )
    assert mod.graft_answer(None, baseline_matches=True, was_skipped=False)["status"] == "missing"
    assert (
        mod.graft_answer({"graft_deferred": True}, baseline_matches=True, was_skipped=False)[
            "status"
        ]
        == "graft_deferred"
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

    missing_root = tmp_path / "missing_upstream"
    payloads = _clean_payloads()
    payloads.pop(4132)
    _write_default_artifacts(missing_root, payloads)
    missing = mod.build_artifact(missing_root, started_s=8.0, now_s=8.1)
    assert missing["missing_upstream_artifacts"] == [{"experiment_id": 4132}]
    assert {row["experiment_id"] for row in missing["upstream_provenance"]} == set(
        mod.UPSTREAM_IDS
    ) - {4132}
