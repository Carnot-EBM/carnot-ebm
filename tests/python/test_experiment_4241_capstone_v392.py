"""Tests for Exp 4241 .392 oracle-distinct capstone aggregation.

Spec refs: REQ-CAPSTONE-4241, SCENARIO-CAPSTONE-4241.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v392_4241 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_payloads() -> dict[str, JsonDict]:
    return {
        "4208_detector": {
            "honest_verdict": "complete: detector fixture",
            "detection_auroc_by_domain": {"arc": 0.9, "code": 0.98},
            "detection_auroc_ci95_by_domain": {"arc": [0.8, 0.99]},
            "selector_headroom_by_domain": {"arc": 0.13, "code": 0.04},
            "verifier_is_oracle_by_domain": {"arc": False, "code": False},
            "n_by_domain": {"arc": 100, "code": 160},
            "divergence_domains": ["sudoku"],
        },
        "4231_build": {
            "honest_verdict": "complete: aggregator build fixture",
            "oracle_distinct_auroc": 0.91,
            "held_out_task_n": 30,
            "baseline_auroc_391": mod.BASELINE_AUROC_391,
            "verifier_is_oracle": False,
        },
        "4231_model": {
            "held_out_task_n": 30,
            "model_type": "standardized_logistic_regression_isotonic_calibrated",
            "accepted_rejected_n": {"accepted": 2, "rejected": 2, "total": 4},
            "oof_rows": [
                {"task_id": "a", "candidate_id": "a0", "score": 0.9, "correct": True},
                {"task_id": "b", "candidate_id": "b0", "score": 0.8, "correct": True},
                {"task_id": "c", "candidate_id": "c0", "score": 0.2, "correct": False},
                {"task_id": "d", "candidate_id": "d0", "score": 0.1, "correct": False},
            ],
            "verifier_is_oracle": False,
        },
        "4232_arc_gate": {
            "honest_verdict": "complete: arc aggregator win fixture",
            "status": "complete",
            "oracle_distinct_beats_vote": True,
            "aggregator_minus_vote_delta": 0.05,
            "aggregator_minus_vote_ci95": [0.01, 0.09],
            "margin_override_minus_vote": 0.04,
            "matched_control_delta": 0.02,
            "matched_control_policy": "deterministic_first_of_k_no_verifier",
            "headroom_exists": True,
            "held_out_task_n": 30,
            "ci95_excludes_zero": True,
            "oracle_at_k": 0.8,
            "pass_rates": {"aggregator_at_1": 0.55, "vote_at_1": 0.5},
            "task_rows": [{"vote_correct": False, "oracle_hit": True}],
            "verifier_is_oracle": False,
        },
        "4233_code": {
            "honest_verdict": "complete: code oracle-distinct win fixture",
            "status": "complete",
            "code_oracle_distinct_beats_vote": True,
            "code_predictor_minus_vote_delta": 0.03,
            "code_predictor_minus_vote_ci95": [0.01, 0.05],
            "matched_control_delta": 0.01,
            "headroom_exists": True,
            "held_out_task_n": 160,
            "off_fold_auroc": 0.97,
            "ci95_excludes_zero": True,
            "oracle_at_k": 0.96,
            "pass_rates": {"predictor_at_1": 0.95, "vote_at_1": 0.92},
            "candidate_pool": {"task_n": 160, "candidate_n": 2000, "positive_n": 1200},
            "disambiguation_read": "ARC_null_is_data_sparsity",
            "verifier_is_oracle": False,
        },
        "4234_smoke": {
            "honest_verdict": "complete: lora smoke fixture",
            "harness_smoke_passed": True,
            "steps_run": 20,
            "trainable_param_count": 10,
            "lora_attach_path": "linear",
            "loss_initial": 1.0,
            "loss_final": 0.5,
            "verifier_is_oracle": True,
        },
        "4235_reward": {
            "honest_verdict": "complete: reward null fixture",
            "verifier_label_carries_signal": False,
            "positive_control_confirmed": True,
            "a_vs_b_delta": 0.0,
            "a_vs_b_ci95": [-0.02, 0.02],
            "live_lora_retired": False,
        },
        "4236_arc_progress": {
            "honest_verdict": "success: arc progress fixture",
            "total_levels_solved": 18,
            "total_games_solved": 13,
            "new_levels_solved_this_task": 1,
            "levels_completed": 4,
            "prior_total_levels_solved": 17,
            "real_env_confirmed": True,
            "acceptance_gate_passed": True,
        },
        "4237_live_solver": {
            "honest_verdict": "complete: live solver fixture",
            "solver_completes_level": False,
            "live_env_metrics": {
                "levels_completed": 0,
                "observed_frame_levels_completed": 1,
                "score": 0.0,
            },
            "solver_beats_floor": {
                "accuracy": {"beats": False},
                "efficiency": {"beats": True},
                "overall": True,
            },
            "live_env_reachable": True,
        },
        "4238_sota": {
            "honest_verdict": "complete: sota fixture",
            "flagged_for_v393": "bigger_arc_pool_full_set_encoder_agglm_aggregator_v393",
            "methods_mapped": [{"name": "Set-Encoder full cross-candidate attention"}],
        },
        "4239_registry": {
            "honest_verdict": "complete: registry fixture",
            "regression_guard_passed": True,
            "oracle_distinct_outcome": {"status": "open_a2_ties_vote_with_headroom_at_power"},
            "code_disambiguation_outcome": {"disambiguation_read": "ARC_null_is_data_sparsity"},
            "verifier_reward_outcome": {
                "status": "open_live_lora_blocked_pre_gate",
                "live_lora_retired": False,
            },
        },
        "4240_hardware": {
            "honest_verdict": "complete: hardware fixture",
            "per_board_reachability": {"gatemate": False, "kv260": True, "polarfire": True},
            "gatemate_step_taken": "blocked_gatemate_unreachable",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
            "kv260_terminal_confirmed": True,
            "fabric_acceleration_claimed": False,
            "speedup_claim_made": False,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[str, JsonDict]) -> None:
    for key, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAMS[key].path, payload)


def _expect_validation_error(artifact: JsonDict, mutator: Any, match: str) -> None:
    mutated = json.loads(json.dumps(artifact))
    mutator(mutated)
    with pytest.raises(ValueError, match=match):
        mod.validate_artifact(mutated)


def test_req_capstone_4241_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4241: OpenSpec declares the .392 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4241" in spec
    assert "SCENARIO-CAPSTONE-4241" in spec
    for outcome in mod.HEADLINE_OUTCOMES:
        assert outcome in spec
    for status in mod.ORACLE_DISTINCT_STATUSES | mod.VERIFIER_AS_REWARD_STATUSES:
        assert status in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "verifier_is_oracle:false" in spec
    assert "flagged_adversarial:true" in spec


def test_scenario_capstone_4241_current_artifacts_decide_frontier_and_skip_flags() -> None:
    """SCENARIO-CAPSTONE-4241: current .392 artifacts skip flagged upstreams first."""

    artifact = mod.build_artifact(Path.cwd(), started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "oracle_distinct_arc_null_is_data_sparsity_code_wins"
    assert artifact["oracle_distinct_status"] == "ARC-NULL-IS-DATA-SPARSITY"
    assert artifact["verifier_as_reward_status"] == "HARNESS-DEFERRED"
    assert artifact["diffusiongemma_gate_resolvable"] is True
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v392_oracle_distinct_arc_null_is_data_sparsity_code_wins_"
    )
    assert artifact["arc_aggregator_gate"]["arc_status"] == "TIES-AT-POWER-NULL"
    assert artifact["arc_aggregator_gate"]["oracle_distinct_beats_vote"] is False
    assert artifact["arc_aggregator_gate"]["held_out_task_n"] == 52
    assert artifact["arc_aggregator_gate"]["verifier_is_oracle"] is False
    assert artifact["arc_aggregator_gate"]["matched_control_present"] is True
    assert artifact["arc_aggregator_gate"]["headroom_present"] is True
    assert artifact["arc_aggregator_gate"]["ci95_excludes_zero"] is False
    assert artifact["arc_aggregator_gate"]["margin_override_minus_vote"] == 0.0
    assert artifact["arc_aggregator_gate"]["matched_control_delta"] == pytest.approx(0.0384615385)
    assert artifact["arc_aggregator_model"]["build_artifact_status"] == (
        "skipped_flagged_adversarial"
    )
    assert artifact["arc_aggregator_model"]["off_fold_auroc"] == pytest.approx(0.8397117856)
    assert artifact["arc_aggregator_model"]["held_out_task_n"] == 52
    assert artifact["arc_aggregator_model"]["improved_over_391"] is True
    assert artifact["arc_aggregator_model"]["wrong_majority_n"] == 0
    assert artifact["arc_aggregator_gate"]["wrong_majority_n"] == 9
    assert artifact["code_disambiguation"]["code_oracle_distinct_beats_vote"] is True
    assert artifact["code_disambiguation"]["disambiguation_read"] == "ARC_null_is_data_sparsity"
    assert artifact["code_disambiguation"]["held_out_task_n"] == 160
    assert artifact["detector_selection_divergence"]["detection_auroc_by_domain"]["arc"] == 0.9016
    assert artifact["detector_selection_divergence"]["selector_headroom_by_domain"]["arc"] == 0.129
    assert artifact["verifier_as_reward"]["a_vs_b_delta"] is None
    assert artifact["verifier_as_reward"]["b1_real_training_smoke"]["status"] == (
        "skipped_flagged_adversarial"
    )
    assert artifact["verifier_as_reward"]["b1_real_training_smoke"]["harness_smoke_passed"] is False
    assert artifact["verifier_as_reward"]["live_lora_retired"] is False
    assert artifact["total_arc_levels_solved"] == 18
    assert artifact["arc_progress"]["levels_completed"] == 4
    assert artifact["live_solver_accuracy"]["solver_completes_level"] is False
    assert artifact["live_solver_accuracy"]["observed_frame_levels_completed"] == 1
    assert artifact["strongest_sota_flagged_for_v393"] == (
        "bigger_arc_pool_full_set_encoder_agglm_aggregator_v393"
    )
    assert artifact["sota_v393"]["strongest_method_name"] == (
        "Set-Encoder full cross-candidate attention"
    )
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    skipped = {row["artifact_key"] for row in artifact["flagged_artifacts_skipped"]}
    assert skipped == {"4231_build", "4234_smoke"}

    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.DEFAULT_UPSTREAMS)
    for key, upstream in mod.DEFAULT_UPSTREAMS.items():
        expected_sha = hashlib.sha256((Path.cwd() / upstream.path).read_bytes()).hexdigest()
        assert provenance[key]["sha256"] == expected_sha
    for key in ("4231_build", "4234_smoke"):
        assert provenance[key]["skipped"] is True
        assert provenance[key]["fields_imported"] == []
    assert "oof_rows" in provenance["4231_model"]["fields_imported"]
    assert "aggregator_minus_vote_delta" in provenance["4232_arc_gate"]["fields_imported"]
    assert "code_predictor_minus_vote_delta" in provenance["4233_code"]["fields_imported"]


def test_req_capstone_4241_headline_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4241: clean inputs distinguish ARC, code, and reward outcomes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    arc_win = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    assert arc_win["headline_outcome"] == "oracle_distinct_aggregator_beats_vote_first_moat"
    assert arc_win["oracle_distinct_status"] == "MOAT-WON"
    assert arc_win["diffusiongemma_gate_resolvable"] is True
    assert arc_win["verifier_as_reward_status"] == "NULL"

    payloads = _minimal_payloads()
    payloads["4232_arc_gate"]["oracle_distinct_beats_vote"] = False
    payloads["4232_arc_gate"]["aggregator_minus_vote_delta"] = 0.0
    payloads["4232_arc_gate"]["aggregator_minus_vote_ci95"] = [-0.03, 0.03]
    payloads["4232_arc_gate"]["ci95_excludes_zero"] = False
    _write_default_artifacts(tmp_path, payloads)
    code_win = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.25)
    assert code_win["headline_outcome"] == "oracle_distinct_arc_null_is_data_sparsity_code_wins"
    assert code_win["oracle_distinct_status"] == "ARC-NULL-IS-DATA-SPARSITY"

    payloads = _minimal_payloads()
    payloads["4232_arc_gate"]["oracle_distinct_beats_vote"] = False
    payloads["4232_arc_gate"]["aggregator_minus_vote_delta"] = 0.0
    payloads["4232_arc_gate"]["aggregator_minus_vote_ci95"] = [-0.03, 0.03]
    payloads["4232_arc_gate"]["ci95_excludes_zero"] = False
    payloads["4233_code"]["code_oracle_distinct_beats_vote"] = False
    payloads["4233_code"]["code_predictor_minus_vote_delta"] = 0.0
    payloads["4233_code"]["code_predictor_minus_vote_ci95"] = [-0.02, 0.02]
    payloads["4233_code"]["ci95_excludes_zero"] = False
    payloads["4233_code"]["disambiguation_read"] = "selection_thesis_bounded"
    _write_default_artifacts(tmp_path, payloads)
    bounded = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.25)
    assert bounded["headline_outcome"] == "oracle_distinct_selection_thesis_bounded_both_tie"
    assert bounded["oracle_distinct_status"] == "THESIS-BOUNDED"
    assert bounded["diffusiongemma_gate_resolvable"] is False

    payloads = _minimal_payloads()
    payloads["4232_arc_gate"]["oracle_distinct_beats_vote"] = False
    payloads["4232_arc_gate"]["aggregator_minus_vote_delta"] = 0.0
    payloads["4232_arc_gate"]["aggregator_minus_vote_ci95"] = [-0.03, 0.03]
    payloads["4232_arc_gate"]["ci95_excludes_zero"] = False
    payloads["4233_code"] = {"honest_verdict": "blocked: no pool", "status": "blocked"}
    _write_default_artifacts(tmp_path, payloads)
    arc_only_null = mod.build_artifact(tmp_path, started_s=5.0, now_s=5.25)
    assert arc_only_null["headline_outcome"] == (
        "oracle_distinct_aggregator_ties_vote_at_power_stronger_null"
    )
    assert arc_only_null["oracle_distinct_status"] == "TIES-AT-POWER-NULL"

    payloads = _minimal_payloads()
    payloads["4232_arc_gate"] = {"honest_verdict": "blocked: no headroom", "status": "blocked"}
    payloads["4233_code"] = {"honest_verdict": "blocked: no headroom", "status": "blocked"}
    payloads["4235_reward"]["a_vs_b_delta"] = 0.05
    payloads["4235_reward"]["a_vs_b_ci95"] = [0.01, 0.09]
    payloads["4235_reward"]["verifier_label_carries_signal"] = True
    _write_default_artifacts(tmp_path, payloads)
    reward_real = mod.build_artifact(tmp_path, started_s=6.0, now_s=6.25)
    assert reward_real["headline_outcome"] == "verifier_reward_real_label_carries_signal"
    assert reward_real["verifier_as_reward_status"] == "REAL"

    payloads = _minimal_payloads()
    payloads["4232_arc_gate"] = {"honest_verdict": "blocked: no headroom", "status": "blocked"}
    payloads["4233_code"] = {"honest_verdict": "blocked: no headroom", "status": "blocked"}
    payloads["4235_reward"]["live_lora_retired"] = True
    _write_default_artifacts(tmp_path, payloads)
    retired = mod.build_artifact(tmp_path, started_s=7.0, now_s=7.25)
    assert retired["headline_outcome"] == "verifier_reward_live_lora_retired"
    assert retired["verifier_as_reward_status"] == "RETIRED-LIVE-LORA"


def test_req_capstone_4241_validation_write_and_missing_edges(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4241: validation protects schema, checksums, and writes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    artifact = mod.build_artifact(tmp_path, started_s=8.0, now_s=8.25)
    mod.validate_artifact(artifact)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        mod.read_json_object(malformed)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4241_capstone_v392.json"),
        started_s=9.0,
        now_s=9.25,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    missing_root = tmp_path / "missing"
    payloads_without_sota = _minimal_payloads()
    payloads_without_sota.pop("4238_sota")
    _write_default_artifacts(missing_root, payloads_without_sota)
    missing = mod.build_artifact(missing_root, started_s=10.0, now_s=10.25)
    assert missing["missing_upstream_artifacts"] == [
        {"artifact_key": "4238_sota", "experiment_id": 4238}
    ]
    assert missing["sota_v393"]["status"] == "missing"

    assert mod.ci95({"x": ["bad", 1.0]}, "x") is None
    assert mod.ci95({"x": [1.0, "bad"]}, "x") is None
    assert mod.ci95({"x": [1.0]}, "x") is None
    assert mod.rank_auc([]) is None
    assert mod.detector_selection_divergence({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.detector_selection_divergence(None, was_skipped=False)["status"] == "missing"
    assert mod.arc_aggregator_model({}, {}, was_model_skipped=True, was_build_skipped=True)[
        "status"
    ] == "skipped_flagged_adversarial"
    assert mod.arc_aggregator_model(None, None, was_model_skipped=False, was_build_skipped=False)[
        "status"
    ] == "missing"
    assert mod.arc_aggregator_gate({}, {}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.arc_aggregator_gate(None, None, was_skipped=False)["status"] == "missing"
    assert mod.code_disambiguation({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.code_disambiguation(None, was_skipped=False)["status"] == "missing"
    assert mod.verifier_as_reward({}, {}, {}, was_reward_skipped=True, was_smoke_skipped=False)[
        "status"
    ] == "skipped_flagged_adversarial"
    assert mod.verifier_as_reward(None, None, None, was_reward_skipped=False, was_smoke_skipped=True)[
        "verifier_as_reward_status"
    ] == "HARNESS-DEFERRED"
    assert mod.verifier_as_reward(None, None, {}, was_reward_skipped=False, was_smoke_skipped=False)[
        "b1_real_training_smoke"
    ]["status"] == "missing"
    passing_smoke = {
        "harness_smoke_passed": True,
        "steps_run": 20,
        "trainable_param_count": 1,
        "loss_initial": 1.0,
        "loss_final": 0.5,
    }
    assert mod.verifier_as_reward(
        None,
        passing_smoke,
        {},
        was_reward_skipped=False,
        was_smoke_skipped=False,
    )["verifier_as_reward_status"] == "INVALID-or-UNDERPOWERED"
    assert mod.verifier_as_reward(
        {
            "positive_control_confirmed": False,
            "verifier_label_carries_signal": True,
            "a_vs_b_delta": 0.03,
            "a_vs_b_ci95": [0.01, 0.05],
        },
        passing_smoke,
        {},
        was_reward_skipped=False,
        was_smoke_skipped=False,
    )["verifier_as_reward_status"] == "INVALID-or-UNDERPOWERED"
    assert mod.headline_outcome("NO-HEADROOM", "NULL") == "verifier_reward_null_distillation"
    assert mod.headline_outcome("NO-HEADROOM", "HARNESS-DEFERRED") == (
        "oracle_distinct_aggregator_ties_vote_at_power_stronger_null"
    )
    assert mod.arc_progress({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.arc_progress(None, was_skipped=False)["status"] == "missing"
    assert mod.live_solver_accuracy({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.live_solver_accuracy(None, was_skipped=False)["status"] == "missing"
    assert mod.sota_v393({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.registry_hygiene({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.registry_hygiene(None, was_skipped=False)["status"] == "missing"
    assert mod.hardware_continuity({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.hardware_continuity(None, was_skipped=False)["status"] == "missing"

    _expect_validation_error(artifact, lambda a: a.pop("honest_verdict"), "missing required")
    _expect_validation_error(artifact, lambda a: a.update({"honest_verdict": "bad"}), "terminal")
    _expect_validation_error(artifact, lambda a: a.update({"headline_outcome": "bad"}), "headline")
    _expect_validation_error(
        artifact, lambda a: a.update({"oracle_distinct_status": "bad"}), "oracle"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"verifier_as_reward_status": "bad"}), "reward"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"diffusiongemma_gate_resolvable": "bad"}), "DiffusionGemma"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"total_arc_levels_solved": 16}), "ARC levels"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"field_principles": []}), "field_principles"
    )
    _expect_validation_error(
        artifact,
        lambda a: a["field_principles"].update({"honest_verdict": "wrong"}),
        "principle",
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"upstream_provenance": {}}), "upstream_provenance"
    )
    _expect_validation_error(
        artifact, lambda a: a["upstream_provenance"].append(42), "entries must be objects"
    )
    _expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"experiment_id": "4208"}),
        "integer experiment_id",
    )
    _expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].pop("artifact_key"),
        "artifact_key",
    )
    _expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"fields_imported": "bad"}),
        "fields_imported",
    )
    _expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][0].update({"skipped": True}),
        "skipped upstreams",
    )
    _expect_validation_error(
        artifact, lambda a: a["upstream_provenance"][0].update({"sha256": "bad"}), "sha256"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"flagged_artifacts_skipped": {}}), "flagged"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"inference_substrate": "bad"}), "inference_substrate"
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"reproducibility_checksum": "bad"}), "checksum"
    )
    _expect_validation_error(artifact, lambda a: a.update({"duration_s": 99.0}), "checksum")

    output_path = tmp_path / "results" / "experiment_4241_capstone_v392.json"
    monkeypatch.setattr(mod, "write_artifact", lambda root: output_path)
    assert mod.main() == 0
    assert str(output_path) in capsys.readouterr().out
