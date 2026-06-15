"""Tests for Exp 4218 .390 oracle-distinct capstone aggregation.

Spec refs: REQ-CAPSTONE-4218, SCENARIO-CAPSTONE-4218.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v390_4218 as mod


JsonDict = dict[str, Any]


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _minimal_payloads() -> dict[int, JsonDict]:
    return {
        4208: {
            "honest_verdict": "complete: detector fixture",
            "detection_auroc_by_domain": {"arc": 0.9, "math": 1.0},
            "detection_auroc_ci95_by_domain": {"arc": [0.8, 0.99]},
            "selector_headroom_by_domain": {"arc": 0.13, "math": 0.0},
            "verifier_is_oracle_by_domain": {"arc": False, "math": True},
            "n_by_domain": {"arc": 8000, "math": 650},
        },
        4209: {
            "honest_verdict": "complete: selector fixture",
            "selector_trained": True,
            "oracle_distinct_auroc": 0.72,
            "oracle_distinct_auroc_ci95": [0.62, 0.82],
            "verifier_is_oracle": False,
            "learned_verifier_path": "results/fixture-selector.pkl",
            "accepted_rejected_n": {"accepted": 12, "rejected": 18, "total": 30},
        },
        4210: {
            "honest_verdict": "complete: oracle distinct win fixture",
            "status": "complete",
            "oracle_distinct_beats_vote": True,
            "oracle_distinct_delta": 0.04,
            "oracle_distinct_ci95": [0.01, 0.07],
            "verifier_is_oracle": False,
            "matched_control": True,
            "headroom_present": True,
        },
        4211: {
            "honest_verdict": "complete: reward fixture",
            "verifier_label_carries_signal": False,
            "positive_control_confirmed": True,
            "a_vs_b_delta": 0.0,
            "a_vs_b_ci95": [-0.02, 0.02],
            "accumulated_n": {"eval": 20},
            "evaluation": {"status": "complete"},
            "verifier_is_oracle": True,
            "youden_j": 0.4,
        },
        4212: {
            "honest_verdict": "complete: certified corpus fixture",
            "certified_corpus_size": 16,
            "certification_precision": {"rate": 0.9375},
            "distill_lift_delta": 0.0,
            "distill_lift_ci95": [0.0, 0.0],
            "distill_lift_latent_vs_absent": "absent",
        },
        4213: {
            "honest_verdict": "success: arc progress fixture",
            "total_levels_solved": 16,
            "total_games_solved": 13,
            "new_levels_solved_this_task": 1,
            "levels_completed": 2,
            "real_env_confirmed": True,
            "acceptance_gate_passed": True,
        },
        4214: {
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
        4215: {
            "honest_verdict": "complete: sota fixture",
            "flagged_for_v391": "arbiter_conservative_override_arc_wrong_majority_v391",
            "methods_mapped": [{"name": "ARBITER conservative override"}],
        },
        4216: {
            "honest_verdict": "complete: registry fixture",
            "regression_guard_passed": True,
        },
        4217: {
            "honest_verdict": "complete: hardware fixture",
            "per_board_reachability": {"gatemate": False, "kv260": True, "polarfire": True},
            "gatemate_step_taken": "blocked_gatemate_unreachable",
            "polarfire_step_taken": "polarfire_hash_verified_cpu_dispatch_succeeded",
            "kv260_terminal_confirmed": True,
        },
    }


def _write_default_artifacts(root: Path, payloads: dict[int, JsonDict]) -> None:
    for experiment_id, payload in payloads.items():
        _write_json(root / mod.DEFAULT_UPSTREAM_PATHS[experiment_id], payload)


def _expect_validation_error(artifact: JsonDict, mutator: Any, match: str) -> None:
    mutated = json.loads(json.dumps(artifact))
    mutator(mutated)
    with pytest.raises(ValueError, match=match):
        mod.validate_artifact(mutated)


def test_req_capstone_4218_spec_anchor_declares_required_contract() -> None:
    """REQ-CAPSTONE-4218: OpenSpec declares the .390 capstone contract."""

    spec = Path("openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")

    assert "REQ-CAPSTONE-4218" in spec
    assert "SCENARIO-CAPSTONE-4218" in spec
    for outcome in mod.HEADLINE_OUTCOMES:
        assert outcome in spec
    for status in mod.ORACLE_DISTINCT_STATUSES | mod.VERIFIER_AS_REWARD_STATUSES:
        assert status in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
        assert mod.FIELD_PRINCIPLES[field] in spec
    assert "verifier_is_oracle:false" in spec
    assert "flagged_adversarial:true" in spec


def test_scenario_capstone_4218_current_artifacts_decide_frontier_and_reward() -> None:
    """SCENARIO-CAPSTONE-4218: current .390 artifacts skip flagged upstreams first."""

    artifact = mod.build_artifact(Path.cwd(), started_s=1.0, now_s=1.5)

    mod.validate_artifact(artifact)
    assert artifact["headline_outcome"] == "oracle_distinct_no_headroom_or_no_learnable_signal"
    assert artifact["oracle_distinct_status"] == "NO-HEADROOM-OR-NO-SIGNAL"
    assert artifact["verifier_as_reward_status"] == "ACCUMULATING"
    assert artifact["diffusiongemma_gate_resolvable"] is False
    assert artifact["honest_verdict"].startswith(
        "complete: capstone_v390_oracle_distinct_no_headroom_or_no_learnable_signal_"
    )
    assert artifact["learned_arc_verifier"]["off_fold_auroc"] == 0.0
    assert artifact["learned_arc_verifier"]["selector_trained"] is False
    assert artifact["oracle_distinct_frontier"]["oracle_distinct_beats_vote"] is False
    assert artifact["oracle_distinct_frontier"]["comparison_ran"] is False
    assert artifact["oracle_distinct_frontier"]["verifier_is_oracle"] is False
    assert artifact["detector_selection_divergence"]["detection_auroc_by_domain"]["arc"] == 0.9016
    assert artifact["detector_selection_divergence"]["selector_headroom_by_domain"]["arc"] == 0.129
    assert artifact["verifier_as_reward"]["a_vs_b_delta"] is None
    assert artifact["verifier_as_reward"]["a_vs_b_ci95"] is None
    assert artifact["verifier_as_reward"]["positive_control_confirmed"] is False
    assert artifact["verifier_as_reward"]["verifier_is_oracle"] is True
    assert artifact["certified_corpus_latent_or_absent"]["status"] == (
        "skipped_flagged_adversarial"
    )
    assert artifact["certified_corpus_latent_or_absent"]["latent_or_absent"] == (
        "SKIPPED-FLAGGED"
    )
    assert artifact["total_arc_levels_solved"] == 16
    assert artifact["arc_progress"]["new_levels_solved_this_task"] == 1
    assert artifact["arc_progress"]["real_env_confirmed"] is True
    assert artifact["live_solver_accuracy"]["solver_completes_level"] is False
    assert artifact["live_solver_accuracy"]["levels_completed"] == 0
    assert artifact["strongest_sota_flagged_for_v391"] == (
        "arbiter_conservative_override_arc_wrong_majority_v391"
    )
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    skipped = {row["experiment_id"] for row in artifact["flagged_artifacts_skipped"]}
    assert skipped == {4212, 4216}

    provenance = {row["experiment_id"]: row for row in artifact["upstream_provenance"]}
    assert set(provenance) == set(mod.DEFAULT_UPSTREAM_PATHS)
    for experiment_id, rel_path in mod.DEFAULT_UPSTREAM_PATHS.items():
        expected_sha = hashlib.sha256((Path.cwd() / rel_path).read_bytes()).hexdigest()
        assert provenance[experiment_id]["sha256"] == expected_sha
    for experiment_id in (4212, 4216):
        assert provenance[experiment_id]["skipped"] is True
        assert provenance[experiment_id]["fields_imported"] == []
    assert "detection_auroc_by_domain" in provenance[4208]["fields_imported"]
    assert "oracle_distinct_auroc" in provenance[4209]["fields_imported"]
    assert "gate_check_summary" in provenance[4210]["fields_imported"]
    assert "a_vs_b_delta" in provenance[4211]["fields_imported"]
    assert "total_levels_solved" in provenance[4213]["fields_imported"]
    assert "solver_completes_level" in provenance[4214]["fields_imported"]
    assert "flagged_for_v391" in provenance[4215]["fields_imported"]
    assert "per_board_reachability" in provenance[4217]["fields_imported"]


def test_req_capstone_4218_headline_branches(tmp_path: Path) -> None:
    """REQ-CAPSTONE-4218: clean inputs distinguish oracle and reward outcomes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    win = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    mod.validate_artifact(win)
    assert win["headline_outcome"] == "oracle_distinct_verifier_beats_vote_first_moat"
    assert win["oracle_distinct_status"] == "MOAT-WON"
    assert win["diffusiongemma_gate_resolvable"] is True
    assert win["verifier_as_reward_status"] == "NULL"

    payloads = _minimal_payloads()
    payloads[4210]["oracle_distinct_beats_vote"] = False
    payloads[4210]["oracle_distinct_delta"] = 0.0
    payloads[4210]["oracle_distinct_ci95"] = [-0.03, 0.03]
    _write_default_artifacts(tmp_path, payloads)
    tie = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.25)
    assert tie["headline_outcome"] == "oracle_distinct_verifier_ties_vote_with_headroom_null"
    assert tie["oracle_distinct_status"] == "TIES-VOTE-NULL"
    assert tie["diffusiongemma_gate_resolvable"] is False

    payloads = _minimal_payloads()
    payloads[4209]["selector_trained"] = False
    payloads[4210] = {"honest_verdict": "blocked_gate_check_failed", "status": "blocked"}
    payloads[4211]["a_vs_b_delta"] = 0.05
    payloads[4211]["a_vs_b_ci95"] = [0.01, 0.09]
    payloads[4211]["verifier_label_carries_signal"] = True
    _write_default_artifacts(tmp_path, payloads)
    reward_real = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.25)
    assert reward_real["headline_outcome"] == "verifier_reward_real_label_carries_signal"
    assert reward_real["oracle_distinct_status"] == "NO-HEADROOM-OR-NO-SIGNAL"
    assert reward_real["verifier_as_reward_status"] == "REAL"

    payloads = _minimal_payloads()
    payloads[4209]["selector_trained"] = False
    payloads[4210] = {"honest_verdict": "blocked_gate_check_failed", "status": "blocked"}
    _write_default_artifacts(tmp_path, payloads)
    reward_null = mod.build_artifact(tmp_path, started_s=5.0, now_s=5.25)
    assert reward_null["headline_outcome"] == "verifier_reward_null_distillation"
    assert reward_null["verifier_as_reward_status"] == "NULL"

    payloads = _minimal_payloads()
    payloads[4209]["selector_trained"] = False
    payloads[4210] = {"honest_verdict": "blocked_gate_check_failed", "status": "blocked"}
    payloads[4211]["positive_control_confirmed"] = False
    _write_default_artifacts(tmp_path, payloads)
    no_signal = mod.build_artifact(tmp_path, started_s=6.0, now_s=6.25)
    assert no_signal["headline_outcome"] == "oracle_distinct_no_headroom_or_no_learnable_signal"
    assert no_signal["verifier_as_reward_status"] == "INVALID-or-UNDERPOWERED"


def test_req_capstone_4218_validation_write_and_missing_edges(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """REQ-CAPSTONE-4218: validation protects schema, checksums, and writes."""

    payloads = _minimal_payloads()
    _write_default_artifacts(tmp_path, payloads)
    artifact = mod.build_artifact(tmp_path, started_s=7.0, now_s=7.25)

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        mod.read_json_object(malformed)

    output = mod.write_artifact(
        tmp_path,
        output_path=Path("results/experiment_4218_capstone_v390.json"),
        started_s=8.0,
        now_s=8.25,
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(written)
    assert written["reproducibility_checksum"] == mod.payload_checksum(written)

    missing_root = tmp_path / "missing"
    payloads_without_4215 = _minimal_payloads()
    payloads_without_4215.pop(4215)
    _write_default_artifacts(missing_root, payloads_without_4215)
    missing = mod.build_artifact(missing_root, started_s=9.0, now_s=9.25)
    assert missing["missing_upstream_artifacts"] == [{"experiment_id": 4215}]
    assert missing["sota_v391"]["status"] == "missing"

    assert mod.ci95({"x": ["bad", 1.0]}, "x") is None
    assert mod.ci95({"x": [1.0, "bad"]}, "x") is None
    assert mod.detector_selection_divergence({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.detector_selection_divergence(None, was_skipped=False)["status"] == "missing"
    assert mod.learned_arc_verifier({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.learned_arc_verifier(None, was_skipped=False)["status"] == "missing"
    assert mod.oracle_distinct_frontier(None, None, None, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.oracle_distinct_frontier(None, None, None, was_skipped=False)["status"] == "missing"
    assert mod.certified_corpus_latent_or_absent(None, was_skipped=False)["status"] == "missing"
    assert mod.certified_corpus_latent_or_absent({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert (
        mod.certified_corpus_latent_or_absent({"distill_lift_ci95": [0.01, 0.02]}, was_skipped=False)[
            "latent_or_absent"
        ]
        == "LATENT"
    )
    assert mod.arc_progress({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.arc_progress(None, was_skipped=False)["status"] == "missing"
    assert mod.live_solver_accuracy({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.live_solver_accuracy(None, was_skipped=False)["status"] == "missing"
    assert mod.sota_v391({}, was_skipped=True)["status"] == "skipped_flagged_adversarial"
    assert mod.registry_hygiene(None, was_skipped=False)["status"] == "missing"
    assert mod.hardware_continuity({}, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.hardware_continuity(None, was_skipped=False)["status"] == "missing"
    assert mod.verifier_as_reward(None, was_skipped=True)["status"] == (
        "skipped_flagged_adversarial"
    )
    assert mod.verifier_as_reward(None, was_skipped=False)["status"] == "missing"

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
        artifact, lambda a: a.update({"total_arc_levels_solved": 14}), "ARC levels"
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

    output_path = tmp_path / "results" / "experiment_4218_capstone_v390.json"
    monkeypatch.setattr(mod, "write_artifact", lambda root: output_path)
    assert mod.main() == 0
    assert str(output_path) in capsys.readouterr().out
