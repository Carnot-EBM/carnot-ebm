"""Tests for Exp 4254 .393 oracle-distinct capstone aggregation.

Spec refs: REQ-CAPSTONE-4254, SCENARIO-CAPSTONE-4254.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v393_4254 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_payloads() -> dict[str, JsonDict]:
    return {
        "4243_pool": {
            "honest_verdict": "complete: pool fixture",
            "arc_pool_grown": True,
            "positive_candidate_n": 48,
            "wrong_majority_n": 30,
            "held_out_task_n": 52,
            "pool_artifact_path": "results/pool.json.gz",
            "verifier_is_oracle": False,
            "model_specs": {"baseline_392": {"positive_candidate_n": 20, "wrong_majority_n": 9}},
        },
        "4244_build": {
            "honest_verdict": "complete: set encoder fixture",
            "aggregator_trained": True,
            "oracle_distinct_auroc": 0.96,
            "oracle_distinct_auroc_ci95": [0.91, 0.99],
            "logistic_auroc": 0.95,
            "set_encoder_vs_logistic_auroc_delta": 0.01,
            "positive_candidate_n": 48,
            "wrong_majority_n": 30,
            "held_out_task_n": 52,
            "verifier_is_oracle": False,
            "learned_verifier_path": "results/model.json",
        },
        "4244_model": {
            "model_type": "standardized_deepsets_context_temperature_calibrated",
            "set_encoder_oof": {"auroc": 0.96, "ci95": [0.91, 0.99], "rows": []},
            "logistic_ablation": {"auroc": 0.95, "ci95": [0.9, 0.98]},
            "positive_candidate_n": 48,
            "wrong_majority_n": 30,
            "held_out_task_n": 52,
            "verifier_is_oracle": False,
        },
        "4245_arc_gate": {
            "honest_verdict": "complete: arc win fixture",
            "status": "complete",
            "oracle_distinct_beats_vote": True,
            "set_encoder_minus_vote_delta": 0.1,
            "set_encoder_minus_vote_ci95": [0.02, 0.18],
            "margin_override_minus_vote": 0.08,
            "matched_control_delta": 0.07,
            "matched_control_policy": "deterministic_first_of_k_no_verifier",
            "headroom_exists": True,
            "held_out_task_n": 52,
            "ci95_excludes_zero": True,
            "oracle_at_k": 0.8,
            "oracle_minus_vote": 0.3,
            "pass_rates": {"set_encoder_at_1": 0.6, "vote_at_1": 0.5},
            "task_rows": [{"vote_correct": False, "oracle_hit": True}],
            "verifier_is_oracle": False,
        },
        "4246_code": {
            "honest_verdict": "complete: code replication fixture",
            "status": "complete",
            "code_replication_beats_vote": False,
            "code_predictor_minus_vote_delta": 0.0,
            "code_predictor_minus_vote_ci95": [-0.02, 0.02],
            "matched_control_delta": 0.0,
            "matched_control_policy": "deterministic_first_of_k_no_verifier",
            "headroom_exists": True,
            "held_out_task_n": 80,
            "ci95_excludes_zero": False,
            "replication_read": "corpus_specific",
            "off_fold_auroc": 0.7,
            "candidate_pool": {"task_n": 80, "candidate_n": 800, "positive_n": 90},
            "verifier_is_oracle": False,
        },
        "4247_reward_retire": {
            "honest_verdict": "blocked: flagged live lora fixture",
            "flagged_adversarial": True,
            "harness_smoke_passed": False,
            "live_lora_retired": True,
            "steps_run": 0,
            "trainable_param_count": 0,
            "verifier_is_oracle": True,
        },
        "4248_reward_offline": {
            "honest_verdict": "blocked_gate_check_failed",
            "status": "blocked",
            "blocked_at_layer": "conductor_pre_gate",
            "gate_check_summary": "harness_smoke_passed false",
        },
        "4249_arc_progress": {
            "honest_verdict": "success: progress fixture",
            "total_levels_solved": 19,
            "total_games_solved": 13,
            "new_levels_solved_this_task": 1,
            "levels_completed": 5,
            "prior_total_levels_solved": 18,
            "real_env_confirmed": True,
            "acceptance_gate_passed": True,
        },
        "4250_live_solver": {
            "honest_verdict": "complete: live solver fixture",
            "solver_completes_level": False,
            "live_env_metrics": {"levels_completed": 0, "observed_frame_levels_completed": 1},
            "solver_beats_floor": {"accuracy": {"beats": False}, "efficiency": {"beats": True}},
            "live_env_reachable": True,
        },
        "4251_sota": {
            "honest_verdict": "complete: sota fixture",
            "flagged_for_v394": "agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394",
            "methods_mapped": [{"name": "AggLM review-reconcile-synthesize aggregation"}],
        },
        "4252_registry": {
            "honest_verdict": "complete: registry fixture",
            "regression_guard_passed": True,
            "live_lora_retired_recorded": True,
            "oracle_distinct_outcome": {"status": "filled_arc_a3_set_encoder_beats_vote_non_oracle"},
            "code_replication_outcome": {"status": "corpus_specific"},
            "verifier_reward_outcome": {
                "status": "blocked_offline_reward_gate_failed_live_lora_retired",
                "live_lora_retired": True,
            },
        },
        "4253_hardware": {
            "honest_verdict": "complete: hardware fixture",
            "per_board_reachability": {"gatemate": False, "kv260": True, "polarfire": True},
            "per_board_status": {},
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


def test_req_capstone_4254_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4254: OpenSpec declares the .393 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4254" in spec
    assert "SCENARIO-CAPSTONE-4254" in spec
    for outcome in mod.HEADLINE_OUTCOMES:
        assert outcome in spec
    for status in mod.ORACLE_DISTINCT_STATUSES | mod.VERIFIER_AS_REWARD_STATUSES:
        assert status in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "verifier_is_oracle:false" in spec
    assert "flagged_adversarial:true" in spec


def test_scenario_capstone_4254_current_artifacts_decide_first_arc_win() -> None:
    """SCENARIO-CAPSTONE-4254: current .393 artifacts decide the ARC frontier."""

    artifact = mod.build_artifact(Path.cwd(), started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "arc_oracle_distinct_set_encoder_beats_vote_first_arc_win"
    assert artifact["oracle_distinct_status"] == "ARC-MOAT-WON"
    assert artifact["verifier_as_reward_status"] == "LIVE-LORA-RETIRED-OFFLINE-PENDING"
    assert artifact["diffusiongemma_gate_resolvable"] is True
    assert artifact["honest_verdict"].startswith("complete: capstone_v393_arc_oracle_distinct")

    assert artifact["pool_growth"]["positive_candidate_n"] == 48
    assert artifact["pool_growth"]["wrong_majority_n"] == 30
    assert artifact["pool_growth"]["grew_over_392_positive_candidate_baseline"] is True
    assert artifact["pool_growth"]["grew_over_392_wrong_majority_baseline"] is True

    assert artifact["set_encoder_build"]["off_fold_auroc"] == pytest.approx(0.9633173387)
    assert artifact["set_encoder_build"]["set_encoder_vs_logistic_auroc_delta"] == pytest.approx(
        -0.0161846276
    )
    assert artifact["set_encoder_build"]["beat_logistic_ablation"] is False

    assert artifact["arc_set_encoder_gate"]["oracle_distinct_beats_vote"] is True
    assert artifact["arc_set_encoder_gate"]["verifier_is_oracle"] is False
    assert artifact["arc_set_encoder_gate"]["matched_control_present"] is True
    assert artifact["arc_set_encoder_gate"]["headroom_present"] is True
    assert artifact["arc_set_encoder_gate"]["powered_held_out_n"] is True
    assert artifact["arc_set_encoder_gate"]["ci95_excludes_zero"] is True
    assert artifact["arc_set_encoder_gate"]["set_encoder_minus_vote_delta"] == pytest.approx(
        0.4423076923
    )

    assert artifact["code_replication"]["code_status"] == "BLOCKED"
    assert artifact["code_replication"]["replication_read"] == "blocked_code_second_corpus_missing"
    assert artifact["code_replication"]["code_replication_beats_vote"] is False

    assert artifact["verifier_as_reward"]["offline_a_vs_b_ran"] is False
    assert artifact["verifier_as_reward"]["live_lora_retired_recorded"] is True
    assert artifact["verifier_as_reward"]["retirement_artifact_skipped"] is True

    assert artifact["total_arc_levels_solved"] == 19
    assert artifact["arc_progress"]["levels_completed"] == 5
    assert artifact["live_solver_accuracy"]["solver_completes_level"] is False
    assert artifact["live_solver_accuracy"]["observed_frame_levels_completed"] == 1
    assert artifact["strongest_sota_flagged_for_v394"] == (
        "agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394"
    )
    assert artifact["sota_v394"]["strongest_method_name"] == (
        "Set-LLM permutation-invariant set architecture"
    )
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    skipped = {row["artifact_key"] for row in artifact["flagged_artifacts_skipped"]}
    assert skipped == {"4247_reward_retire"}

    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.DEFAULT_UPSTREAMS)
    for key, upstream in mod.DEFAULT_UPSTREAMS.items():
        expected_sha = hashlib.sha256((Path.cwd() / upstream.path).read_bytes()).hexdigest()
        assert provenance[key]["sha256"] == expected_sha
    assert provenance["4247_reward_retire"]["skipped"] is True
    assert provenance["4247_reward_retire"]["fields_imported"] == []
    assert "positive_candidate_n" in provenance["4243_pool"]["fields_imported"]
    assert "oracle_distinct_auroc" in provenance["4244_build"]["fields_imported"]
    assert "set_encoder_minus_vote_delta" in provenance["4245_arc_gate"]["fields_imported"]
    assert "replication_read" in provenance["4246_code"]["fields_imported"]


def test_req_capstone_4254_headline_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4254: clean inputs distinguish ARC, code, and reward outcomes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    arc_win = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    assert arc_win["headline_outcome"] == "arc_oracle_distinct_set_encoder_beats_vote_first_arc_win"
    assert arc_win["oracle_distinct_status"] == "ARC-MOAT-WON"
    assert arc_win["diffusiongemma_gate_resolvable"] is True

    payloads = _minimal_payloads()
    payloads["4245_arc_gate"]["oracle_distinct_beats_vote"] = False
    payloads["4245_arc_gate"]["set_encoder_minus_vote_delta"] = 0.0
    payloads["4245_arc_gate"]["set_encoder_minus_vote_ci95"] = [-0.03, 0.03]
    payloads["4245_arc_gate"]["ci95_excludes_zero"] = False
    payloads["4246_code"]["code_replication_beats_vote"] = True
    payloads["4246_code"]["code_predictor_minus_vote_delta"] = 0.04
    payloads["4246_code"]["code_predictor_minus_vote_ci95"] = [0.01, 0.07]
    payloads["4246_code"]["ci95_excludes_zero"] = True
    payloads["4246_code"]["replication_read"] = "replicates"
    _write_default_artifacts(tmp_path, payloads)
    code_robust = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.25)
    assert code_robust["headline_outcome"] == "oracle_distinct_code_robust_arc_still_data_bound"
    assert code_robust["oracle_distinct_status"] == "CODE-ROBUST-ARC-BOUND"
    assert code_robust["diffusiongemma_gate_resolvable"] is False

    payloads = _minimal_payloads()
    payloads["4245_arc_gate"]["oracle_distinct_beats_vote"] = False
    payloads["4245_arc_gate"]["set_encoder_minus_vote_delta"] = 0.0
    payloads["4245_arc_gate"]["set_encoder_minus_vote_ci95"] = [-0.03, 0.03]
    payloads["4245_arc_gate"]["ci95_excludes_zero"] = False
    payloads["4246_code"]["headroom_exists"] = False
    payloads["4248_reward_offline"] = {
        "honest_verdict": "complete: reward real fixture",
        "status": "complete",
        "verifier_label_carries_signal": True,
        "positive_control_confirmed": True,
        "a_vs_b_delta": 0.05,
        "a_vs_b_ci95": [0.01, 0.09],
    }
    payloads["4252_registry"]["verifier_reward_outcome"] = {"live_lora_retired": False}
    payloads["4252_registry"]["live_lora_retired_recorded"] = False
    _write_default_artifacts(tmp_path, payloads)
    reward_real = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.25)
    assert reward_real["headline_outcome"] == "arc_oracle_distinct_ties_vote_at_power_on_grown_pool_real_bound"
    assert reward_real["oracle_distinct_status"] == "TIES-AT-POWER-ON-GROWN-POOL"
    assert reward_real["verifier_as_reward_status"] == "OFFLINE-REAL"

    payloads = _minimal_payloads()
    payloads["4245_arc_gate"]["headroom_exists"] = False
    payloads["4245_arc_gate"]["oracle_distinct_beats_vote"] = False
    payloads["4245_arc_gate"]["set_encoder_minus_vote_ci95"] = [0.0, 0.0]
    payloads["4246_code"]["headroom_exists"] = False
    payloads["4248_reward_offline"] = {
        "honest_verdict": "complete: reward null fixture",
        "status": "complete",
        "positive_control_confirmed": True,
        "a_vs_b_delta": 0.0,
        "a_vs_b_ci95": [-0.01, 0.01],
    }
    payloads["4252_registry"]["verifier_reward_outcome"] = {"live_lora_retired": False}
    payloads["4252_registry"]["live_lora_retired_recorded"] = False
    _write_default_artifacts(tmp_path, payloads)
    reward_null = mod.build_artifact(tmp_path, started_s=5.0, now_s=5.25)
    assert reward_null["headline_outcome"] == "verifier_reward_offline_null_distillation"
    assert reward_null["oracle_distinct_status"] == "NO-HEADROOM"
    assert reward_null["verifier_as_reward_status"] == "OFFLINE-NULL"


def test_req_capstone_4254_validation_write_and_missing_edges(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4254: validation protects schema, checksums, and writes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    artifact = mod.build_artifact(tmp_path, started_s=6.0, now_s=6.25)
    mod.validate_artifact(artifact)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        mod.read_json_object(malformed)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4254_capstone_v393.json"),
        started_s=7.0,
        now_s=7.25,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    missing_root = tmp_path / "missing"
    payloads_without_sota = _minimal_payloads()
    payloads_without_sota.pop("4251_sota")
    _write_default_artifacts(missing_root, payloads_without_sota)
    missing = mod.build_artifact(missing_root, started_s=8.0, now_s=8.25)
    assert missing["missing_upstream_artifacts"] == [
        {"artifact_key": "4251_sota", "experiment_id": 4251}
    ]
    assert missing["sota_v394"]["status"] == "missing"

    assert mod.ci95({"x": ["bad", 1.0]}, "x") is None
    assert mod.ci95({"x": [1.0, "bad"]}, "x") is None
    assert mod.ci95({"x": [1.0]}, "x") is None
    assert mod.ci_excludes_zero([0.01, 0.02]) is True
    assert mod.ci_includes_zero([-0.01, 0.02]) is True
    assert mod.wrong_majority_count({"wrong_majority_n": 2}) == 2
    assert mod.wrong_majority_count({}) == 0
    assert mod.pool_growth({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.pool_growth(None, was_skipped=False)["status"] == "missing"
    assert mod.set_encoder_build({}, {}, was_build_skipped=True, was_model_skipped=False)[
        "status"
    ] == "skipped_flagged_adversarial"
    assert mod.set_encoder_build(None, None, was_build_skipped=False, was_model_skipped=False)[
        "status"
    ] == "missing"
    fallback_build = mod.set_encoder_build(
        {"honest_verdict": "complete: fallback", "verifier_is_oracle": False},
        {
            "set_encoder_oof": {"auroc": 0.8, "ci95": [0.7, 0.9]},
            "logistic_ablation": {"auroc": 0.7, "ci95": [0.6, 0.8]},
        },
        was_build_skipped=False,
        was_model_skipped=False,
    )
    assert fallback_build["off_fold_auroc"] == pytest.approx(0.8)
    assert fallback_build["logistic_auroc"] == pytest.approx(0.7)
    assert fallback_build["set_encoder_vs_logistic_auroc_delta"] == pytest.approx(0.1)
    assert mod.arc_set_encoder_gate({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.arc_set_encoder_gate(None, was_skipped=False)["status"] == "missing"
    inferred_headroom = mod.arc_set_encoder_gate(
        {
            "oracle_distinct_beats_vote": False,
            "set_encoder_minus_vote_delta": 0.0,
            "set_encoder_minus_vote_ci95": [-0.1, 0.1],
            "matched_control_delta": 0.0,
            "held_out_task_n": 52,
            "oracle_minus_vote": 0.2,
            "verifier_is_oracle": False,
        },
        was_skipped=False,
    )
    assert inferred_headroom["headroom_present"] is True
    assert mod.code_replication({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.code_replication(None, was_skipped=False)["status"] == "missing"
    legacy_code_win = mod.code_replication(
        {
            "code_oracle_distinct_beats_vote": True,
            "code_predictor_minus_vote_delta": 0.05,
            "code_predictor_minus_vote_ci95": [0.01, 0.09],
            "matched_control_delta": 0.01,
            "headroom_exists": True,
            "held_out_task_n": 80,
            "verifier_is_oracle": False,
        },
        was_skipped=False,
    )
    assert legacy_code_win["code_status"] == "CODE-ROBUST"
    assert mod.verifier_as_reward(
        None,
        None,
        {},
        reward_skipped=False,
        retirement_skipped=True,
    )["verifier_as_reward_status"] == "INVALID-or-UNDERPOWERED"
    assert mod.arc_progress({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.arc_progress(None, was_skipped=False)["status"] == "missing"
    assert mod.live_solver_accuracy({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.live_solver_accuracy(None, was_skipped=False)["status"] == "missing"
    assert mod.sota_v394({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.registry_hygiene({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.registry_hygiene(None, was_skipped=False)["status"] == "missing"
    assert mod.hardware_continuity({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.hardware_continuity(None, was_skipped=False)["status"] == "missing"
    assert mod.headline_outcome("NO-HEADROOM", "OFFLINE-REAL") == (
        "verifier_reward_offline_real_label_carries_signal"
    )
    assert mod.headline_outcome("NO-HEADROOM", "LIVE-LORA-RETIRED-OFFLINE-PENDING") == (
        "verifier_reward_live_lora_retired_offline_pending"
    )
    assert mod.headline_outcome("NO-HEADROOM", "INVALID-or-UNDERPOWERED") == (
        "arc_oracle_distinct_ties_vote_at_power_on_grown_pool_real_bound"
    )
    assert mod.imported_fields_by_key({"4247_reward_retire"})["4247_reward_retire"] == [
        "harness_smoke_passed",
        "live_lora_retired",
        "steps_run",
        "trainable_param_count",
        "lora_attach_path",
        "loss_initial",
        "loss_final",
    ]

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
        artifact,
        lambda a: a.update(
            {"oracle_distinct_status": "NO-HEADROOM", "diffusiongemma_gate_resolvable": True}
        ),
        "DiffusionGemma gate is resolvable",
    )
    _expect_validation_error(
        artifact, lambda a: a.update({"total_arc_levels_solved": 17}), "ARC levels"
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
        lambda a: a["upstream_provenance"][0].update({"experiment_id": "4243"}),
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
    skipped_index = next(
        i for i, row in enumerate(artifact["upstream_provenance"]) if row["skipped"] is True
    )
    _expect_validation_error(
        artifact,
        lambda a: a["upstream_provenance"][skipped_index].update({"fields_imported": ["bad"]}),
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

    output_path = tmp_path / "results" / "experiment_4254_capstone_v393.json"
    monkeypatch.setattr(mod, "write_artifact", lambda root: output_path)
    assert mod.main() == 0
    assert str(output_path) in capsys.readouterr().out
