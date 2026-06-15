"""Tests for Exp 4229 .391 oracle-distinct capstone aggregation.

Spec refs: REQ-CAPSTONE-4229, SCENARIO-CAPSTONE-4229.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v391_4229 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_payloads() -> dict[str, JsonDict]:
    return {
        "4208_detector": {
            "honest_verdict": "complete: detector fixture",
            "detection_auroc_by_domain": {"arc": 0.9, "math": 1.0},
            "detection_auroc_ci95_by_domain": {"arc": [0.8, 0.99]},
            "selector_headroom_by_domain": {"arc": 0.13, "math": 0.0},
            "verifier_is_oracle_by_domain": {"arc": False, "math": True},
            "n_by_domain": {"arc": 100, "math": 20},
        },
        "4220_build_labeled": {
            "honest_verdict": "complete: summary fixture",
            "flagged_adversarial": True,
            "oracle_distinct_auroc": 0.75,
            "wrong_majority_n": 3,
        },
        "4220_model": {
            "accepted_rejected_n": {"accepted": 2, "rejected": 2, "total": 4},
            "model_type": "standardized_logistic_regression",
            "oof_rows": [
                {"score": 0.9, "correct": True},
                {"score": 0.8, "correct": True},
                {"score": 0.2, "correct": False},
                {"score": 0.1, "correct": False},
            ],
            "verifier_is_oracle": False,
        },
        "4221_gate": {
            "honest_verdict": "complete: oracle distinct win fixture",
            "status": "complete",
            "oracle_distinct_beats_vote": True,
            "verifier_minus_vote_delta": 0.04,
            "verifier_minus_vote_ci95": [0.01, 0.07],
            "verifier_is_oracle": False,
            "matched_control_delta": 0.01,
            "matched_control_policy": "deterministic_first_of_k_no_verifier",
            "headroom_exists": True,
            "ci95_excludes_zero": True,
            "arbiter_override_minus_vote": 0.02,
            "pass_rates": {"vote_at_1": 0.5, "verifier_at_1": 0.54},
            "task_rows": [
                {"vote_correct": False, "oracle_hit": True},
                {"vote_correct": True, "oracle_hit": True},
            ],
        },
        "4222_harness": {
            "honest_verdict": "complete: harness fixture",
            "harness_smoke_passed": True,
            "trainable_param_count": 10,
            "lora_attach_path": "linear",
        },
        "4223_reward": {
            "honest_verdict": "complete: reward fixture",
            "verifier_label_carries_signal": False,
            "positive_control_confirmed": True,
            "a_vs_b_delta": 0.0,
            "a_vs_b_ci95": [-0.02, 0.02],
            "accumulated_n": {"eval": 20},
            "evaluation": {"status": "complete"},
        },
        "4224_arc_progress": {
            "honest_verdict": "success: arc progress fixture",
            "total_levels_solved": 17,
            "total_games_solved": 13,
            "new_levels_solved_this_task": 1,
            "levels_completed": 3,
            "real_env_confirmed": True,
            "acceptance_gate_passed": True,
        },
        "4225_live_solver": {
            "honest_verdict": "complete: live solver fixture",
            "solver_completes_level": False,
            "live_env_metrics": {"levels_completed": 0, "score": 0.0},
            "solver_beats_floor": {
                "accuracy": {"beats": False},
                "efficiency": {"beats": True},
                "overall": True,
            },
            "live_env_reachable": True,
        },
        "4226_sota": {
            "honest_verdict": "complete: sota fixture",
            "flagged_for_v392": "agglm_style_arc_review_reconcile_aggregator_v392",
            "methods_mapped": [{"name": "AggLM review-and-reconcile solution aggregation"}],
        },
        "4227_registry": {
            "honest_verdict": "complete: registry fixture",
            "regression_guard_passed": True,
            "oracle_distinct_outcome": {"status": "open_a2_ties_vote_with_headroom"},
            "verifier_reward_outcome": {"status": "open_accumulating_reward_no_eval_yet"},
        },
        "4228_hardware": {
            "honest_verdict": "complete: hardware fixture",
            "per_board_reachability": {"gatemate": False, "kv260": True, "polarfire": True},
            "gatemate_step_taken": "blocked_gatemate_unreachable",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
            "kv260_terminal_confirmed": True,
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


def test_req_capstone_4229_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4229: OpenSpec declares the .391 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4229" in spec
    assert "SCENARIO-CAPSTONE-4229" in spec
    for outcome in mod.HEADLINE_OUTCOMES:
        assert outcome in spec
    for status in mod.ORACLE_DISTINCT_STATUSES | mod.VERIFIER_AS_REWARD_STATUSES:
        assert status in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "verifier_is_oracle:false" in spec
    assert "flagged_adversarial:true" in spec


def test_scenario_capstone_4229_current_artifacts_decide_frontier_and_skip_flags() -> None:
    """SCENARIO-CAPSTONE-4229: current .391 artifacts skip flagged upstreams first."""

    artifact = mod.build_artifact(Path.cwd(), started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "oracle_distinct_verifier_ties_vote_with_headroom_null"
    assert artifact["oracle_distinct_status"] == "TIES-VOTE-NULL"
    assert artifact["verifier_as_reward_status"] == "HARNESS-DEFERRED"
    assert artifact["diffusiongemma_gate_resolvable"] is False
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v391_oracle_distinct_verifier_ties_vote_with_headroom_null_"
    )
    assert artifact["learned_arc_verifier"]["off_fold_auroc"] == pytest.approx(0.7790203623536957)
    assert artifact["learned_arc_verifier"]["wrong_majority_n"] == 5
    assert artifact["learned_arc_verifier"]["summary_artifact_status"] == (
        "skipped_flagged_adversarial"
    )
    assert artifact["oracle_distinct_frontier"]["gate_ran"] is True
    assert artifact["oracle_distinct_frontier"]["oracle_distinct_beats_vote"] is False
    assert artifact["oracle_distinct_frontier"]["verifier_is_oracle"] is False
    assert artifact["oracle_distinct_frontier"]["matched_control_present"] is True
    assert artifact["oracle_distinct_frontier"]["headroom_present"] is True
    assert artifact["oracle_distinct_frontier"]["ci95_excludes_zero"] is False
    assert artifact["detector_selection_divergence"]["detection_auroc_by_domain"]["arc"] == 0.9016
    assert artifact["detector_selection_divergence"]["selector_headroom_by_domain"]["arc"] == 0.129
    assert artifact["verifier_as_reward"]["a_vs_b_delta"] is None
    assert artifact["verifier_as_reward"]["verifier_label_carries_signal"] is None
    assert artifact["verifier_as_reward"]["b1_harness_smoke"]["status"] == (
        "skipped_flagged_adversarial"
    )
    assert artifact["total_arc_levels_solved"] == 17
    assert artifact["arc_progress"]["levels_completed"] == 3
    assert artifact["live_solver_accuracy"]["solver_completes_level"] is False
    assert artifact["live_solver_accuracy"]["observed_frame_levels_completed"] == 1
    assert artifact["strongest_sota_flagged_for_v392"] == (
        "agglm_style_arc_review_reconcile_aggregator_v392"
    )
    assert artifact["sota_v392"]["strongest_method_name"] == (
        "AggLM review-and-reconcile solution aggregation"
    )
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    skipped = {row["artifact_key"] for row in artifact["flagged_artifacts_skipped"]}
    assert skipped == {"4220_build_labeled", "4222_harness", "4223_reward"}

    provenance = {row["artifact_key"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.DEFAULT_UPSTREAMS)
    for key, upstream in mod.DEFAULT_UPSTREAMS.items():
        expected_sha = hashlib.sha256((Path.cwd() / upstream.path).read_bytes()).hexdigest()
        assert provenance[key]["sha256"] == expected_sha
    for key in ("4220_build_labeled", "4222_harness", "4223_reward"):
        assert provenance[key]["skipped"] is True
        assert provenance[key]["fields_imported"] == []
    assert "oof_rows" in provenance["4220_model"]["fields_imported"]
    assert "task_rows" in provenance["4221_gate"]["fields_imported"]
    assert "detection_auroc_by_domain" in provenance["4208_detector"]["fields_imported"]


def test_req_capstone_4229_headline_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4229: clean inputs distinguish oracle and reward outcomes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    win = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    assert win["headline_outcome"] == "oracle_distinct_verifier_beats_vote_first_moat"
    assert win["oracle_distinct_status"] == "MOAT-WON"
    assert win["diffusiongemma_gate_resolvable"] is True
    assert win["verifier_as_reward_status"] == "NULL"

    payloads = _minimal_payloads()
    payloads["4221_gate"]["oracle_distinct_beats_vote"] = False
    payloads["4221_gate"]["verifier_minus_vote_delta"] = 0.0
    payloads["4221_gate"]["verifier_minus_vote_ci95"] = [-0.03, 0.03]
    payloads["4221_gate"]["ci95_excludes_zero"] = False
    _write_default_artifacts(tmp_path, payloads)
    tie = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.25)
    assert tie["headline_outcome"] == "oracle_distinct_verifier_ties_vote_with_headroom_null"
    assert tie["oracle_distinct_status"] == "TIES-VOTE-NULL"

    payloads = _minimal_payloads()
    payloads["4221_gate"] = {"honest_verdict": "blocked: no headroom", "status": "blocked"}
    payloads["4223_reward"]["a_vs_b_delta"] = 0.05
    payloads["4223_reward"]["a_vs_b_ci95"] = [0.01, 0.09]
    payloads["4223_reward"]["verifier_label_carries_signal"] = True
    _write_default_artifacts(tmp_path, payloads)
    reward_real = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.25)
    assert reward_real["headline_outcome"] == "verifier_reward_real_label_carries_signal"
    assert reward_real["oracle_distinct_status"] == "NO-HEADROOM-OR-NO-SIGNAL"
    assert reward_real["verifier_as_reward_status"] == "REAL"

    payloads = _minimal_payloads()
    payloads["4221_gate"] = {"honest_verdict": "blocked: no headroom", "status": "blocked"}
    _write_default_artifacts(tmp_path, payloads)
    reward_null = mod.build_artifact(tmp_path, started_s=5.0, now_s=5.25)
    assert reward_null["headline_outcome"] == "verifier_reward_null_distillation"
    assert reward_null["verifier_as_reward_status"] == "NULL"

    payloads = _minimal_payloads()
    payloads["4221_gate"] = {"honest_verdict": "blocked: no headroom", "status": "blocked"}
    payloads["4222_harness"]["harness_smoke_passed"] = False
    _write_default_artifacts(tmp_path, payloads)
    deferred = mod.build_artifact(tmp_path, started_s=6.0, now_s=6.25)
    assert deferred["headline_outcome"] == "oracle_distinct_no_headroom_or_no_learnable_signal"
    assert deferred["verifier_as_reward_status"] == "HARNESS-DEFERRED"


def test_req_capstone_4229_validation_write_and_missing_edges(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4229: validation protects schema, checksums, and writes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    artifact = mod.build_artifact(tmp_path, started_s=7.0, now_s=7.25)
    mod.validate_artifact(artifact)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        mod.read_json_object(malformed)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4229_capstone_v391.json"),
        started_s=8.0,
        now_s=8.25,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    missing_root = tmp_path / "missing"
    payloads_without_sota = _minimal_payloads()
    payloads_without_sota.pop("4226_sota")
    _write_default_artifacts(missing_root, payloads_without_sota)
    missing = mod.build_artifact(missing_root, started_s=9.0, now_s=9.25)
    assert missing["missing_upstream_artifacts"] == [
        {"artifact_key": "4226_sota", "experiment_id": 4226}
    ]
    assert missing["sota_v392"]["status"] == "missing"

    assert mod.ci95({"x": ["bad", 1.0]}, "x") is None
    assert mod.ci95({"x": [1.0, "bad"]}, "x") is None
    assert mod.ci95({"x": [1.0]}, "x") is None
    assert mod.rank_auc([]) is None
    assert mod.detector_selection_divergence({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.detector_selection_divergence(None, was_skipped=False)["status"] == "missing"
    assert (
        mod.learned_arc_verifier({}, {}, was_model_skipped=True, was_summary_skipped=True)["status"]
        == "skipped_flagged_adversarial"
    )
    assert (
        mod.learned_arc_verifier(None, None, was_model_skipped=False, was_summary_skipped=False)[
            "status"
        ]
        == "missing"
    )
    assert mod.oracle_distinct_frontier({}, None, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.oracle_distinct_frontier(None, None, was_skipped=False)["status"] == "missing"
    assert (
        mod.verifier_as_reward(
            None,
            {"harness_smoke_passed": True, "trainable_param_count": 1},
            was_reward_skipped=False,
            was_harness_skipped=False,
        )["verifier_as_reward_status"]
        == "INVALID-or-UNDERPOWERED"
    )
    assert (
        mod.verifier_as_reward(
            {"honest_verdict": "progress: accumulating", "evaluation": {"status": "pending"}},
            {"harness_smoke_passed": True, "trainable_param_count": 1},
            was_reward_skipped=False,
            was_harness_skipped=False,
        )["verifier_as_reward_status"]
        == "ACCUMULATING"
    )
    assert (
        mod.verifier_as_reward(
            {"positive_control_confirmed": False, "a_vs_b_ci95": [0.1, 0.2]},
            {"harness_smoke_passed": True, "trainable_param_count": 1},
            was_reward_skipped=False,
            was_harness_skipped=False,
        )["verifier_as_reward_status"]
        == "INVALID-or-UNDERPOWERED"
    )
    assert (
        mod.verifier_as_reward(None, None, was_reward_skipped=False, was_harness_skipped=False)[
            "b1_harness_smoke"
        ]["status"]
        == "missing"
    )
    assert (
        mod.verifier_as_reward(None, None, was_reward_skipped=False, was_harness_skipped=True)[
            "verifier_as_reward_status"
        ]
        == "HARNESS-DEFERRED"
    )
    assert mod.arc_progress({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.arc_progress(None, was_skipped=False)["status"] == "missing"
    assert mod.live_solver_accuracy({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.live_solver_accuracy(None, was_skipped=False)["status"] == "missing"
    assert mod.sota_v392({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
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
        artifact, lambda a: a.update({"total_arc_levels_solved": 15}), "ARC levels"
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

    output_path = tmp_path / "results" / "experiment_4229_capstone_v391.json"
    monkeypatch.setattr(mod, "write_artifact", lambda root: output_path)
    assert mod.main() == 0
    assert str(output_path) in capsys.readouterr().out
