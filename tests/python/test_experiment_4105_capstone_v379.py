"""Tests for Exp 4105 .379 TRM-pivot capstone aggregation.

Spec refs: REQ-CAPSTONE-4105, SCENARIO-CAPSTONE-4105.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v379_4105 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_statuses(
    experiment_ids: tuple[int, ...] = mod.UPSTREAM_IDS,
    *,
    returncodes: dict[int, int] | None = None,
) -> dict[int, JsonDict]:
    overrides = returncodes or {}
    return {
        experiment_id: {
            "returncode": overrides.get(experiment_id, 0),
            "stdout": f"summarized {experiment_id}",
            "stderr": "",
        }
        for experiment_id in experiment_ids
    }


def _clean_payloads() -> dict[int, JsonDict]:
    return {
        4099: {
            "honest_verdict": "success: verifier_beats_trm_vote",
            "best_reranker": "STACK_ALL",
            "captured_pp_directional": 0.08,
            "verifier_beats_trm_vote": True,
            "pool_n_tasks": 120,
            "n_tasks_scored": 120,
            "underpowered": False,
            "per_reranker": {
                "TRM_VOTE": {"pass@2": 0.25},
                "STACK_ALL": {"pass@2": 0.33, "captured_pp_ci95": [0.02, 0.14]},
            },
        },
        4100: {
            "honest_verdict": "success: verifier_rft_beats_vote_sft",
            "branch_taken": "rft",
            "rft_vs_ablation_delta": {
                "delta": 0.06,
                "ci95": [0.01, 0.11],
                "metric": "heldout_pass@2",
                "status": "rft_beats_vote_sft",
            },
            "trm_native_trainer_checkpoint_ok": True,
        },
        4101: {
            "honest_verdict": "success: eleventh_game_solved_fixture",
            "game_solved": True,
            "real_env_confirmed": True,
            "prior_total_games_solved": 10,
            "total_games_solved": 11,
            "target_game": "fixture-game",
            "first_solve_at_action": 13,
        },
        4102: {
            "honest_verdict": "complete: sota_ingestion_trm_self_training_mapped",
            "flagged_for_v380": "vstar_rejected_trace_selector_for_trm_rft",
            "methods_mapped": [
                {"name": "V-STaR keep-rejected verifier training", "arxiv_id": "2402.06457"},
                {"name": "Verifiable process rewards for recursive steps", "arxiv_id": "2605.10325"},
            ],
        },
        4103: {
            "honest_verdict": "complete: registry_gaps_reconciled",
            "gaps_updated": ["GAP-TRM-GRID-DISCRIMINATION"],
            "regression_guard_passed": True,
            "exp4099_gap": {
                "gap_id": "GAP-TRM-GRID-DISCRIMINATION",
                "missing_discriminator": "signal_separating_correct_trm_grid_from_wrong_grid",
            },
        },
        4104: {
            "honest_verdict": "complete: hardware_continuity",
            "kv260_terminal_confirmed": True,
            "per_board_reachability": {"kv260": True, "gatemate": True, "polarfire": True},
            "gatemate_step_taken": "gatemate_fixture",
            "polarfire_step_taken": "polarfire_fixture",
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def test_req_capstone_4105_spec_anchor_exists() -> None:
    """REQ-CAPSTONE-4105: OpenSpec declares the .379 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4105" in spec
    assert "SCENARIO-CAPSTONE-4105" in spec
    assert "headline_outcome" in spec
    assert "upstream_provenance" in spec
    assert "honest_negative_no_grid_discrimination" in spec


def test_scenario_capstone_4105_current_artifacts_report_honest_negative() -> None:
    """SCENARIO-CAPSTONE-4105: current artifacts skip flagged exp4100 metrics."""

    artifact = mod.build_artifact(
        Path.cwd(),
        summary_statuses=_summary_statuses(returncodes={4100: 2}),
        started_s=10.0,
        now_s=12.0,
    )

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "honest_negative_no_grid_discrimination"
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v379_honest_negative_no_grid_discrimination_"
        "best_K_OF_N_AGREEMENT_captured_0.0000_games11_flagged_skipped1"
    )
    assert artifact["headline_answer"]["verifier_beats_trm_vote"] is False
    assert artifact["headline_answer"]["captured_pp_vs_prior"] == {
        "prior_captured_pp": 0.0,
        "captured_pp": 0.0,
        "delta_vs_prior_pp": 0.0,
    }
    assert artifact["trm_grid_discrimination"]["bottleneck"] == (
        "signal_separating_correct_trm_grid_from_confident_wrong_trm_grid_on_pool"
    )
    assert artifact["verifier_rft_followthrough"]["status"] == "skipped_flagged"
    assert artifact["verifier_rft_followthrough"]["rft_vs_ablation_delta"] is None
    assert artifact["verifier_rft_followthrough"]["native_trainer_checkpoint"] == "skipped_flagged"
    assert artifact["total_arc_games_solved"] == 11
    assert artifact["arc_games"]["target_game"] == "s5i5-18d95033"
    assert "GAP-TRM-GRID-DISCRIMINATION" in artifact["candidate_v380_directions"][0]

    skipped = artifact["flagged_artifacts_skipped"]
    assert [row["experiment_id"] for row in skipped] == [4100]
    assert skipped[0]["sha256"] == hashlib.sha256(
        Path("results/experiment_4100_trm_verifier_rft_conditional.json").read_bytes()
    ).hexdigest()

    provenance = {row["experiment_id"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.UPSTREAM_IDS)
    assert provenance[4100]["fields_imported"] == []
    assert provenance[4100]["skipped"] is True
    assert provenance[4099]["sha256"] == hashlib.sha256(
        Path("results/experiment_4099_trm_pool_verifier_discrimination_probe.json").read_bytes()
    ).hexdigest()
    assert "verifier_beats_trm_vote" in provenance[4099]["fields_imported"]


def test_req_capstone_4105_clean_fixture_validates_verifier_rft(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4105: clean verifier win plus RFT win validates the TRM pivot."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    artifact = mod.build_artifact(
        tmp_path, summary_statuses=_summary_statuses(), started_s=1.0, now_s=1.5
    )

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "verifier_rft_on_trm_validated"
    assert artifact["honest_verdict"].startswith(
        "success: capstone_v379_verifier_rft_on_trm_validated_"
        "best_STACK_ALL_captured_0.0800_games11_flagged_skipped0"
    )
    assert artifact["verifier_rft_followthrough"]["status"] == "rft_beats_vote_sft"
    assert artifact["verifier_rft_followthrough"]["rft_vs_ablation_delta"]["delta"] == 0.06
    assert artifact["verifier_rft_followthrough"]["rft_vs_ablation_delta"]["ci95"] == [0.01, 0.11]
    assert artifact["verifier_rft_followthrough"]["native_trainer_checkpoint"] == "checkpoint_ok"
    assert artifact["headline_answer"]["native_trm_trainer_checkpoint_produced"] is True
    assert artifact["flagged_artifacts_skipped"] == []
    assert {row["experiment_id"] for row in artifact["upstream_provenance"]} == set(
        mod.UPSTREAM_IDS
    )


def test_req_capstone_4105_trainer_derisked_branch_is_distinct(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4105: clean trainer success without RFT win leaves science open."""

    payloads = _clean_payloads()
    payloads[4100]["rft_vs_ablation_delta"] = {
        "delta": 0.0,
        "ci95": [-0.03, 0.03],
        "metric": "heldout_pass@2",
        "status": "measured_null",
    }
    _write_default_artifacts(tmp_path, payloads)

    artifact = mod.build_artifact(
        tmp_path, summary_statuses=_summary_statuses(), started_s=2.0, now_s=2.25
    )

    mod.validate_artifact(artifact)

    assert artifact["headline_outcome"] == "trainer_derisked_science_open"
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v379_trainer_derisked_science_open_"
        "best_STACK_ALL_captured_0.0800_games11_flagged_skipped0"
    )
    assert artifact["verifier_rft_followthrough"]["status"] == "measured_null"
    assert artifact["headline_answer"]["verifier_rft_beat_vote_sft_ablation"] is False
    assert artifact["headline_answer"]["native_trm_trainer_checkpoint_produced"] is True


def test_scenario_capstone_4105_write_artifact_validates_output(tmp_path: Path) -> None:
    """SCENARIO-CAPSTONE-4105: write_artifact emits the required deliverable JSON."""

    _write_default_artifacts(tmp_path, _clean_payloads())

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4105_capstone_v379.json"),
        summary_statuses=_summary_statuses(),
        started_s=3.0,
        now_s=3.5,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(payload)
    assert payload["experiment_id"] == 4105
    assert payload["reproducibility_checksum"] == mod.payload_checksum(payload)


def test_req_capstone_4105_validation_and_branch_states(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4105: helpers keep missing, flagged, and invalid states explicit."""

    fallback = tmp_path / "results/experiment_4099_fixture.json"
    _write_json(fallback, {"honest_verdict": "complete: fallback"})
    paths = mod.selected_upstream_paths(tmp_path)
    statuses = mod.summarize_existing_artifacts(tmp_path, paths, {4099: {"returncode": 0}})
    assert paths[4099] == fallback
    assert paths[4100] is None
    assert statuses == {4099: {"returncode": 0}}

    assert mod.float_metric({"x": True}, "x") == 0.0
    assert mod.int_metric({"x": False}, "x") == 0
    assert mod.list_float_metric({"ci": "bad"}, "ci") == []
    assert mod.nested_str({"outer": {"inner": "value"}}, ("outer", "inner")) == "value"
    assert mod.nested_str({"outer": {"inner": 3}}, ("outer", "inner")) == ""
    assert mod.nested_bool({"outer": {"inner": True}}, ("outer", "inner")) is True
    assert mod.nested_int({"outer": {"inner": 4}}, ("outer", "inner")) == 4
    assert mod.nested_int({"outer": {"inner": True}}, ("outer", "inner")) == 0
    assert mod.is_sha256("0" * 64) is True
    assert mod.is_sha256("bad") is False

    missing_disc = mod.trm_grid_discrimination(None, None, was_skipped=False)
    assert missing_disc["status"] == "missing"
    skipped_disc = mod.trm_grid_discrimination({"flagged_adversarial": True}, None, was_skipped=True)
    assert skipped_disc["status"] == "skipped_flagged"
    blocked_disc = mod.trm_grid_discrimination(
        {"honest_verdict": "blocked_cache_missing"}, None, was_skipped=False
    )
    assert blocked_disc["status"] == "blocked"
    missing_rft = mod.verifier_rft_followthrough(
        None,
        verifier_beat_vote=True,
        was_skipped=False,
    )
    assert missing_rft["status"] == "missing"
    blocked_rft = mod.verifier_rft_followthrough(
        {"honest_verdict": "blocked_resources_missing"},
        verifier_beat_vote=True,
        was_skipped=False,
    )
    assert blocked_rft["status"] == "blocked"
    not_applicable_rft = mod.verifier_rft_followthrough(
        {"honest_verdict": "success: would_be_clean"},
        verifier_beat_vote=False,
        was_skipped=False,
    )
    assert not_applicable_rft["status"] == "not_applicable_no_grid_discrimination"
    no_checkpoint = mod.verifier_rft_followthrough(
        {
            "honest_verdict": "success: rft_win_no_checkpoint",
            "rft_vs_ablation_delta": {"delta": 0.04, "ci95": [0.01, 0.07]},
        },
        verifier_beat_vote=True,
        was_skipped=False,
    )
    assert no_checkpoint["status"] == "rft_beats_without_checkpoint"
    assert mod.arc_games(None, {"total_games_solved": 10}, was_skipped=True)["status"] == (
        "skipped_flagged"
    )

    artifact = mod.build_artifact(
        tmp_path,
        summary_statuses={4099: {"returncode": 0}},
        started_s=1.0,
        now_s=1.1,
    )
    artifact["headline_outcome"] = "not_enumerated"
    with pytest.raises(ValueError, match="headline_outcome"):
        mod.validate_artifact(artifact)
    artifact["headline_outcome"] = "honest_negative_no_grid_discrimination"
    artifact["upstream_provenance"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="sha256"):
        mod.validate_artifact(artifact)
