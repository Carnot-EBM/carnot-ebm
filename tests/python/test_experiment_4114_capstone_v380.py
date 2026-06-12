"""Tests for Exp 4114 .380 capstone aggregation.

Spec refs: REQ-CAPSTONE-4114, SCENARIO-CAPSTONE-4114.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v380_4114 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _clean_payloads() -> dict[int, JsonDict]:
    return {
        4107: {
            "honest_verdict": "complete: nanotrm_trainer_checkpoint_ok_exact_accuracy_1.0000",
            "nanotrm_trainer_checkpoint_ok": True,
            "exact_accuracy": 1.0,
            "exact_accuracy_metric": "val/exact_accuracy",
        },
        4108: {
            "honest_verdict": "complete: reproduced_0.8700",
            "mechanism_checkpoint_ok": True,
            "checkpoint_reload_ok": True,
            "reproduced_exact_accuracy": 0.87,
            "published_exact_accuracy_target": 0.87,
            "published_match_tolerance": 0.02,
            "matches_published_087": True,
            "return_code": 0,
        },
        4109: {
            "honest_verdict": "success: verifier_value_added",
            "verifier_value_added": True,
            "rft_vs_ablation_delta": {
                "delta": 0.08,
                "ci95": [0.02, 0.14],
                "metric": "heldout_exact_accuracy",
                "status": "verifier_beats_vote_ablation",
            },
            "rerank_lift_vs_vote": {
                "delta": 0.04,
                "ci95": [0.01, 0.07],
                "metric": "pass@1_exact_accuracy",
            },
        },
        4110: {
            "honest_verdict": "success: twelfth_game_solved_fixture",
            "prior_total_games_solved": 11,
            "total_games_solved": 12,
            "game_solved": True,
            "real_env_confirmed": True,
            "target_game": "fixture-game",
            "first_solve_at_action": 9,
        },
        4111: {
            "honest_verdict": "complete: sota_ingestion_trm_verifier_training_mapped",
            "flagged_for_v381": "verifier_guided_adaptive_sudoku_search_before_training",
            "methods_mapped": [
                {"name": "Verifier-guided adaptive Sudoku search", "arxiv_id": "2602.01070"}
            ],
        },
        4112: {
            "honest_verdict": "complete: registry_gaps_reconciled",
            "gaps_updated": [
                "GAP-TRM-GRID-DISCRIMINATION",
                "GAP-SUDOKU-EXECUTABLE-VERIFIER-4109",
            ],
            "regression_guard_passed": True,
        },
        4113: {
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


def test_req_capstone_4114_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4114: OpenSpec declares the .380 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4114" in spec
    assert "SCENARIO-CAPSTONE-4114" in spec
    assert "honest_null_verifier_no_added_value" in spec
    assert "flagged_adversarial:true" in spec
    assert "upstream artifact by sha256" in spec


def test_scenario_capstone_4114_current_artifacts_emit_honest_null() -> None:
    """SCENARIO-CAPSTONE-4114: current flagged graft metrics are excluded."""

    artifact = mod.build_artifact(Path.cwd(), started_s=10.0, now_s=12.0)

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "honest_null_verifier_no_added_value"
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v380_honest_null_verifier_no_added_value_"
        "mechanism_derisked1_baseline0870_graft_skipped_flagged_adversarial_"
        "games12_flagged_skipped1"
    )
    assert "gated_on" not in artifact
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["nanotrm_trainer_mechanism"]["derisked"] is True
    assert artifact["nanotrm_trainer_mechanism"]["checkpoint_ok"] is True
    assert artifact["nanotrm_trainer_mechanism"]["exact_accuracy"] == 1.0
    assert artifact["published_baseline_reproduction"]["published_087_baseline_reproduced"] is False
    assert artifact["published_baseline_reproduction"]["reproduced_exact_accuracy"] == pytest.approx(
        0.02317708358168602
    )
    assert artifact["sudoku_verifier_graft"]["status"] == "skipped_flagged_adversarial"
    assert artifact["sudoku_verifier_graft"]["beat_vote_ablation"] is False
    assert artifact["sudoku_verifier_graft"]["a_vs_b_ci95"] is None
    assert artifact["total_arc_games_solved"] == 12

    skipped = artifact["flagged_artifacts_skipped"]
    assert [row["experiment_id"] for row in skipped] == [4109]
    assert skipped[0]["sha256"] == hashlib.sha256(
        Path("results/experiment_4109_carnot_verifier_graft_sudoku.json").read_bytes()
    ).hexdigest()

    provenance = {row["experiment_id"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.UPSTREAM_IDS)
    for experiment_id, row in provenance.items():
        expected_sha = hashlib.sha256(mod.DEFAULT_UPSTREAM_PATHS[experiment_id].read_bytes()).hexdigest()
        assert row["sha256"] == expected_sha
    assert provenance[4109]["skipped"] is True
    assert provenance[4109]["fields_imported"] == []
    assert "nanotrm_trainer_checkpoint_ok" in provenance[4107]["fields_imported"]
    assert "matches_published_087" in provenance[4108]["fields_imported"]
    assert "total_games_solved" in provenance[4110]["fields_imported"]


def test_req_capstone_4114_clean_fixture_validates_reward(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4114: clean baseline and positive graft validate the reward."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "verifier_as_reward_validated_on_executable_domain"
    assert artifact["honest_verdict"].startswith(
        "success: capstone_v380_verifier_as_reward_validated_on_executable_domain_"
        "mechanism_derisked1_baseline0871_graft_verifier_value_added_games12_"
        "flagged_skipped0"
    )
    assert artifact["sudoku_verifier_graft"]["beat_vote_ablation"] is True
    assert artifact["sudoku_verifier_graft"]["a_vs_b_delta"] == 0.08
    assert artifact["sudoku_verifier_graft"]["a_vs_b_ci95"] == [0.02, 0.14]
    assert artifact["flagged_artifacts_skipped"] == []
    provenance = {row["experiment_id"]: row for row in artifact["upstream_provenance"]}
    assert "verifier_value_added" in provenance[4109]["fields_imported"]


def test_req_capstone_4114_other_headline_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4114: mechanism-blocked and inconclusive branches stay distinct."""

    payloads = _clean_payloads()
    payloads[4109]["flagged_adversarial"] = True
    _write_default_artifacts(tmp_path, payloads)
    inconclusive = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.2)
    assert inconclusive["headline_outcome"] == "baseline_reproduced_graft_inconclusive"
    assert inconclusive["sudoku_verifier_graft"]["status"] == "skipped_flagged_adversarial"

    payloads = _clean_payloads()
    payloads[4107]["nanotrm_trainer_checkpoint_ok"] = False
    payloads[4107]["exact_accuracy"] = 0.0
    _write_default_artifacts(tmp_path, payloads)
    blocked = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.4)
    assert blocked["headline_outcome"] == "mechanism_still_blocked"
    assert blocked["honest_verdict"].startswith("blocked: capstone_v380_mechanism_still_blocked_")


def test_scenario_capstone_4114_write_and_validate(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4114: write_artifact emits the deliverable JSON."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4114_capstone_v380.json"),
        started_s=4.0,
        now_s=4.5,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == 4114
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)


def test_req_capstone_4114_validation_rejects_drift(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4114: validation protects the headline and provenance."""

    _write_default_artifacts(tmp_path, _clean_payloads())
    artifact = mod.build_artifact(tmp_path, started_s=5.0, now_s=5.1)

    artifact["headline_outcome"] = "not_enumerated"
    with pytest.raises(ValueError, match="headline_outcome"):
        mod.validate_artifact(artifact)

    artifact["headline_outcome"] = "verifier_as_reward_validated_on_executable_domain"
    artifact["upstream_provenance"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(artifact)


def test_req_capstone_4114_helper_edge_states(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4114: helper states keep skipped and missing inputs explicit."""

    assert mod.list_float_metric({"ci": "bad"}, "ci") == []
    assert mod.mechanism_answer({"flagged_adversarial": True}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.mechanism_answer(None, was_skipped=False)["status"] == "missing"
    assert mod.baseline_answer({"flagged_adversarial": True}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.baseline_answer(None, was_skipped=False)["status"] == "missing"
    assert mod.graft_answer(None, was_skipped=False)["status"] == "missing"
    assert mod.arc_games_answer({"flagged_adversarial": True}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.arc_games_answer(None, was_skipped=False)["status"] == "missing"
    assert mod.arc_games_answer({"game_solved": False}, was_skipped=False)["status"] == (
        "measured_no_new_solve"
    )

    payloads = _clean_payloads()
    payloads.pop(4113)
    _write_default_artifacts(tmp_path, payloads)
    artifact = mod.build_artifact(tmp_path, started_s=6.0, now_s=6.1)
    assert artifact["missing_upstream_artifacts"] == [{"experiment_id": 4113}]
    assert {row["experiment_id"] for row in artifact["upstream_provenance"]} == set(
        mod.UPSTREAM_IDS
    ) - {4113}
