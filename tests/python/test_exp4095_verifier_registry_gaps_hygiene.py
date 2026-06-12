"""Tests for Exp 4095 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4095, SCENARIO-VERIFY-4095.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import verifier_registry_gaps_hygiene_4095 as exp4095


REPO_ROOT = Path(__file__).parents[2]


def _minimal_registry() -> dict[str, Any]:
    return {
        "verifiers": [
            {
                "verifier_id": "gap4_program_induction_stack",
                "domain": "arc_agi2_grid",
                "version": 1,
                "kind": "process_verifier",
                "code_path": "python/carnot/agentic/gap4_program_induction_stack.py",
                "eval": {"metric": "pass_at_1"},
                "status": "candidate",
                "training_time_roles": [
                    {
                        "role_id": "verifier_as_reward_rft_4079",
                        "experiment": "results/experiment_4079_verifier_reward_rft_eval_collect.json",
                        "role": "training_time_reward_signal",
                        "status": "blocked",
                        "outcome": "verifier_as_reward_rft_blocked",
                        "honest_verdict": "blocked_gate_check_failed",
                    }
                ],
            }
        ]
    }


def _write_minimal_repo(tmp_path: Path) -> None:
    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")
    for name in (
        "arc3_gap3_stage2_eval_pool.json.gz",
        "arc3_gap4_induced_programs.json",
        "experiment_4087_certification_precision_rescue.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def _tiny_pool_and_programs() -> tuple[dict[str, Any], dict[str, Any]]:
    gold = [[1]]
    wrong = [[0]]
    pool = {
        "entries": [
            {
                "task": "tiny",
                "candidates": [
                    {"votes": 10, "q_mean": 0.0, "correct": False, "grid": wrong},
                    {"votes": 1, "q_mean": 0.0, "correct": True, "grid": gold},
                ],
            }
        ]
    }
    programs = {
        "programs": [
            {
                "task": "tiny",
                "entry_i": 0,
                "demo_perfect": True,
                "pred_grid": gold,
                "pred_hash": exp4095.grid_hash(gold),
            }
        ]
    }
    return pool, programs


def test_req_4095_spec_declared() -> None:
    # REQ-VERIFY-4095: OpenSpec declares the runner and required artifact fields.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4095",
        "SCENARIO-VERIFY-4095",
        "exp4095_verifier_registry_gaps_hygiene.py",
        "experiment_4095_verifier_registry_gaps_hygiene.json",
        "arc3_gap4_induced_programs.json",
        "experiment_4087_certification_precision_rescue.json",
        "gap4_arc1_reproduced",
        "precision_rescue_recorded",
        "rft_outcome_recorded",
        "verifier_ensemble_against_cached_candidates",
    ):
        assert marker in spec


def test_replay_gap4_arc1_from_cached_programs_bitexact() -> None:
    # SCENARIO-VERIFY-4095: cached programs plus pool reproduce vote 0.4516 -> gated 0.5806.
    replay = exp4095.replay_gap4_arc1(REPO_ROOT)
    assert replay["gap4_arc1_reproduced"] is True
    assert replay["arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }
    assert replay["no_codex_calls"] is True
    assert replay["no_gguf_inference"] is True
    assert replay["cached_programs_path"] == exp4095.ARC1_PROGRAMS_PATH


def test_replay_gap4_arc1_detects_drift_from_fixture() -> None:
    # REQ-VERIFY-4095: drift keeps observed and expected values rather than silently passing.
    pool, programs = _tiny_pool_and_programs()
    replay = exp4095.replay_gap4_arc1_fixture(pool, programs)
    assert replay["gap4_arc1_reproduced"] is False
    assert replay["arc1_rule_exec"] == {
        "n": 1,
        "vote_pass2": pytest.approx(1.0),
        "gated_pass2": pytest.approx(1.0),
        "headroom_recovered": 0,
        "vote_wins_lost": 0,
    }
    assert replay["expected"]["arc1_rule_exec"]["n"] == 31


def test_classifies_actual_precision_rescue_and_pending_rft() -> None:
    # REQ-VERIFY-4095: .378 precision rescue is recorded; absent Exp 4090 stays pending.
    precision = exp4095.classify_precision_rescue(REPO_ROOT)
    assert precision["precision_rescue_recorded"] == "precision_rescue_succeeded"
    assert precision["best_certified_precision"] == pytest.approx(0.8824)
    assert precision["best_op_point_recall"] == pytest.approx(0.7143)
    assert precision["precision_floor_reached"] is True
    assert precision["any_stack_reached_0_85"] is True
    assert precision["best_operating_point"]["filter_stack"] == "k_of_n_agreement"

    rft = exp4095.classify_rft_outcome(REPO_ROOT)
    assert rft["rft_outcome_recorded"] == "rft_a_vs_b_pending_absent"
    assert rft["present"] is False
    assert rft["status"] == "pending"


def test_rft_classifier_handles_present_exp4090(tmp_path: Path) -> None:
    # REQ-VERIFY-4095: a present Exp 4090 A-vs-B artifact is summarized, not ignored.
    results = tmp_path / "results"
    results.mkdir()
    path = results / "experiment_4090_rft_a_vs_b_eval.json"
    path.write_text(
        json.dumps(
            {
                "experiment": 4090,
                "status": "complete",
                "honest_verdict": "complete: arm_a_beats_arm_b_fixture",
                "arm_a_score": 0.42,
                "arm_b_score": 0.35,
                "arm_a_vs_b_delta": 0.07,
            }
        ),
        encoding="utf-8",
    )
    rft = exp4095.classify_rft_outcome(tmp_path)
    assert rft["present"] is True
    assert rft["status"] == "complete"
    assert rft["rft_outcome_recorded"] == "rft_a_vs_b_complete"
    assert rft["arm_a_vs_b_delta"] == pytest.approx(0.07)


def test_ensure_ledgers_record_precision_and_rft_state() -> None:
    # SCENARIO-VERIFY-4095: registry and gaps carry replay, precision point, and pending RFT.
    replay = exp4095.replay_gap4_arc1(REPO_ROOT)
    precision = exp4095.classify_precision_rescue(REPO_ROOT)
    rft = exp4095.classify_rft_outcome(REPO_ROOT)

    registry, gaps, summary = exp4095.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n",
        replay,
        precision,
        rft,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": True,
        "precision_rescue_recorded": True,
        "rft_outcome_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4095"] == exp4095.EXP4095_ARTIFACT_PATH
    assert gap4["eval"]["exp4095_gap4_arc1_reproduced"] is True
    point = gap4["certification_precision_operating_point"]
    assert point["experiment"] == exp4095.EXP4087_PATH
    assert point["best_certified_precision"] == pytest.approx(0.8824)
    assert point["precision_floor_reached"] is True
    assert gap4["training_time_roles"][0]["role_id"] == "verifier_as_reward_rft_4079"
    assert exp4095.PRECISION_BLOCK_ID in gaps
    assert exp4095.RFT_BLOCK_ID in gaps
    assert "any_stack_reached_0_85=true" in gaps
    assert "rft_a_vs_b_pending_absent" in gaps


def test_build_artifact_validates_required_fields() -> None:
    # REQ-VERIFY-4095: terminal artifact exposes the required bare bools and substrate.
    artifact = exp4095.build_artifact(
        offline_replay=exp4095.replay_gap4_arc1(REPO_ROOT),
        precision_rescue=exp4095.classify_precision_rescue(REPO_ROOT),
        rft_outcome=exp4095.classify_rft_outcome(REPO_ROOT),
        registry_updated=True,
        gaps_updated=True,
        precision_rescue_recorded=True,
        rft_outcome_recorded=True,
        duration_s=0.0123,
    )

    exp4095.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["gap4_arc1_reproduced"] is True
    assert artifact["precision_rescue_recorded"] is True
    assert artifact["rft_outcome_recorded"] is True
    assert artifact["inference_substrate"] == exp4095.INFERENCE_SUBSTRATE


def test_run_hygiene_writes_terminal_artifact_and_ledgers(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4095: run writes the deliverable JSON and stable ledger entries.
    _write_minimal_repo(tmp_path)
    artifact = exp4095.run_hygiene(tmp_path)
    exp4095.validate_artifact(artifact)

    out_path = tmp_path / exp4095.EXP4095_ARTIFACT_PATH
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["gap4_arc1_reproduced"] is True
    assert written["precision_rescue_recorded"] is True
    assert written["rft_outcome_recorded"] is True
    assert written["inference_substrate"] == exp4095.INFERENCE_SUBSTRATE

    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text())
    point = registry["verifiers"][0]["certification_precision_operating_point"]
    assert point["filter_stack"] == "k_of_n_agreement"
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert "Exp 4095 precision-rescue registry update" in gaps
    assert "Exp 4095 RFT A-vs-B outcome update" in gaps
