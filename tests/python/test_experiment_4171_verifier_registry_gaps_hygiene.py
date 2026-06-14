"""Tests for Exp 4171 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4171, SCENARIO-VERIFY-4171.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4171_verifier_registry_gaps_hygiene as exp4171_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4171 as exp4171


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
                "training_time_roles": [
                    {
                        "role_id": "sudoku_executable_verifier_training_time_4163",
                        "experiment": "results/experiment_4163_verifier_registry_gaps_hygiene.json",
                        "status": "prior_fixture",
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
        "experiment_4167_outerloop_training_monitor.json",
        "experiment_4168_decisive_verifier_graft_defensive.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def test_req_4171_spec_declared() -> None:
    # REQ-VERIFY-4171: OpenSpec declares runner, inputs, outputs, and principles.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4171",
        "SCENARIO-VERIFY-4171",
        "python/carnot/experiment_4171_verifier_registry_gaps_hygiene.py",
        "experiment_4171_verifier_registry_gaps_hygiene.json",
        "experiment_4167_outerloop_training_monitor.json",
        "experiment_4168_decisive_verifier_graft_defensive.json",
        "0.4516",
        "0.5806",
        "current_val_exact_accuracy=0.5042",
        "baseline_status.current_val_exact_accuracy=0.5148",
        "graft_deferred=true",
        "verifier_value_added=false",
        "checkpoint_copy_performed=false",
        "diffusiongemma_gate_state=kept_gated",
    ):
        assert marker in spec
    assert exp4171.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4171.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4171.FIELD_PRINCIPLES["gaps_updated"] in spec
    assert exp4171_wrapper.main is exp4171.main


def test_scenario_4171_preconditions_and_replay_are_bitexact() -> None:
    # SCENARIO-VERIFY-4171: resources parse before cached ARC-1 replay runs.
    preflight = exp4171.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None
    assert {check["resource"] for check in preflight["checks"]} == {
        "gap4_arc1_candidate_fixtures",
        "verifier_registry",
        "verifier_gaps",
        "exp4167_outerloop_monitor",
        "exp4168_decisive_graft",
    }

    replay = exp4171.replay_gap4_arc1(REPO_ROOT)
    assert replay["regression_guard_passed"] is True
    assert replay["arc1_rule_exec"] == {
        "n": 31,
        "vote_pass2": pytest.approx(0.4516),
        "gated_pass2": pytest.approx(0.5806),
        "headroom_recovered": 4,
        "vote_wins_lost": 0,
    }
    assert replay["no_codex_calls"] is True
    assert replay["no_gguf_inference"] is True


def test_req_4171_classifies_outerloop_baseline_graft_and_gate() -> None:
    # REQ-VERIFY-4171: .386 outcomes are recorded with read-only caveats.
    baseline = exp4171.classify_sudoku_baseline(REPO_ROOT)
    assert baseline["gap_id"] == exp4171.SUDOKU_BASELINE_GAP_ID
    assert baseline["status"] == "open_outerloop_training_alive_val_0.5042"
    assert baseline["baseline_faithful"] is False
    assert baseline["outerloop_train_alive"] is True
    assert baseline["current_val_exact_accuracy"] == pytest.approx(0.504166662693)
    assert baseline["current_val_exact_accuracy_rounded"] == pytest.approx(0.5042)
    assert baseline["max_val_exact_accuracy_rounded"] == pytest.approx(0.5042)
    assert baseline["checkpoint_mtime"] == "2026-06-13T04:41:29.093138Z"
    assert baseline["read_only_actions"]["training_launched"] is False
    assert baseline["read_only_actions"]["train_process_stop_attempted"] is False
    assert baseline["read_only_actions"]["stable_checkpoint_written"] is False
    assert baseline["val_trajectory_386_rounded"][0] == pytest.approx(0.2005)
    assert baseline["val_trajectory_386_rounded"][-1] == pytest.approx(0.5042)
    assert len(baseline["val_trajectory_386_rounded"]) == 68

    graft = exp4171.classify_sudoku_decisive_graft(REPO_ROOT)
    assert graft["gap_id"] == exp4171.SUDOKU_GRAFT_GAP_ID
    assert graft["status"] == "open_graft_deferred_outerloop_training_val_0.5148"
    assert graft["graft_deferred"] is True
    assert graft["verifier_value_added"] is False
    assert graft["acceptance_gate_passed"] is True
    assert graft["checkpoint_copy_performed"] is False
    assert graft["baseline_status"]["current_val_exact_accuracy"] == pytest.approx(0.514843761921)
    assert graft["baseline_status"]["current_val_exact_accuracy_rounded"] == pytest.approx(0.5148)
    assert graft["rerank_lift_vs_vote"]["status"] == "deferred_outerloop_training"
    assert graft["rft_vs_ablation_delta"]["status"] == "deferred_outerloop_training"
    assert graft["read_only_actions"]["training_launched"] is False
    assert graft["read_only_actions"]["stable_checkpoint_written"] is False

    gate = exp4171.classify_diffusiongemma_gate(graft)
    assert gate == {
        "state": "kept_gated",
        "reason": "no_positive_rerank_signal_and_no_training_time_value_added",
        "rerank_ci_excludes_zero_positive": False,
        "verifier_value_added": False,
        "graft_deferred": True,
        "uses_executable_oracle_upper_bound": False,
        "basis": "exp4168_rerank_lift_vs_vote_or_verifier_value_added",
    }


def test_scenario_4171_ensure_ledgers_record_baseline_graft_and_roles() -> None:
    # SCENARIO-VERIFY-4171: registry and gaps carry the .386 truth.
    replay = exp4171.replay_gap4_arc1(REPO_ROOT)
    baseline = exp4171.classify_sudoku_baseline(REPO_ROOT)
    graft = exp4171.classify_sudoku_decisive_graft(REPO_ROOT)
    gate = exp4171.classify_diffusiongemma_gate(graft)

    registry, gaps, summary = exp4171.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n",
        replay,
        baseline,
        graft,
        gate,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": [
            exp4171.SUDOKU_BASELINE_GAP_ID,
            exp4171.SUDOKU_GRAFT_GAP_ID,
        ],
        "sudoku_baseline_recorded": True,
        "sudoku_graft_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4171"] == exp4171.EXP4171_ARTIFACT_PATH
    assert gap4["eval"]["exp4171_regression_guard_passed"] is True
    assert gap4["role_sudoku_executable"]["status"] == "graft_deferred_outerloop_training"
    assert gap4["role_sudoku_executable"]["promoted_toward_candidate"] is False

    training_role = next(
        role
        for role in gap4["training_time_roles"]
        if role["role_id"] == exp4171.SUDOKU_TRAINING_ROLE_ID
    )
    assert training_role["status"] == "graft_deferred_outerloop_training"
    assert training_role["graft_deferred"] is True
    assert training_role["verifier_value_added"] is False
    assert training_role["checkpoint_copy_performed"] is False
    assert training_role["baseline_current_val_rounded"] == pytest.approx(0.5042)
    assert training_role["graft_baseline_current_val_rounded"] == pytest.approx(0.5148)
    assert exp4171._registry_contains_outcomes(registry) is True
    assert exp4171._registry_contains_outcomes({}) is False

    assert exp4171.SUDOKU_BASELINE_GAP_ID in gaps
    assert "current_val=0.5042" in gaps
    assert "outerloop_train_alive=true" in gaps
    assert "baseline_faithful=false" in gaps
    assert "val_trajectory_386_rounded=" in gaps
    assert exp4171.SUDOKU_GRAFT_GAP_ID in gaps
    assert "baseline_current_val=0.5148" in gaps
    assert "graft_deferred=true" in gaps
    assert "verifier_value_added=false" in gaps
    assert "checkpoint_copy_performed=false" in gaps
    assert "diffusiongemma_gate_state=kept_gated" in gaps


def test_req_4171_build_artifact_validates_required_fields() -> None:
    # REQ-VERIFY-4171: terminal artifact exposes required schema fields and principles.
    replay = exp4171.replay_gap4_arc1(REPO_ROOT)
    baseline = exp4171.classify_sudoku_baseline(REPO_ROOT)
    graft = exp4171.classify_sudoku_decisive_graft(REPO_ROOT)
    gate = exp4171.classify_diffusiongemma_gate(graft)
    artifact = exp4171.build_artifact(
        offline_replay=replay,
        sudoku_baseline=baseline,
        sudoku_decisive_graft=graft,
        diffusiongemma_gate_state=gate,
        registry_updated=True,
        gaps_updated=[
            exp4171.SUDOKU_BASELINE_GAP_ID,
            exp4171.SUDOKU_GRAFT_GAP_ID,
        ],
        duration_s=0.012,
    )

    exp4171.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4171.SUDOKU_BASELINE_GAP_ID,
        exp4171.SUDOKU_GRAFT_GAP_ID,
    ]
    assert artifact["field_principles"] == exp4171.FIELD_PRINCIPLES
    assert artifact["sudoku_baseline"]["current_val_exact_accuracy_rounded"] == pytest.approx(0.5042)
    assert artifact["sudoku_decisive_graft"]["graft_deferred"] is True
    assert artifact["diffusiongemma_gate_state"]["state"] == "kept_gated"
    assert artifact["cited_upstream_artifacts"] == [
        exp4171.ARC1_POOL_PATH,
        exp4171.ARC1_PROGRAMS_PATH,
        exp4171.EXP4167_PATH,
        exp4171.EXP4168_PATH,
    ]

    for field in exp4171.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4171.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4171.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4171.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4171.validate_artifact({**artifact, "registry_updated": "yes"})
    with pytest.raises(ValueError, match="list"):
        exp4171.validate_artifact({**artifact, "gaps_updated": "GAP"})
    with pytest.raises(ValueError, match="state"):
        exp4171.validate_artifact({**artifact, "diffusiongemma_gate_state": {}})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4171.validate_artifact({**artifact, "inference_substrate": "live"})
    with pytest.raises(ValueError, match="field_principles"):
        exp4171.validate_artifact({**artifact, "field_principles": {}})


def test_req_4171_helper_edges_are_explicit(tmp_path: Path) -> None:
    # REQ-VERIFY-4171: schema helpers expose edge states without hidden inference.
    assert exp4171._numeric_or_none("bad") is None
    assert exp4171._round4(None) is None
    assert exp4171._check_json_resource(tmp_path, "missing", "missing.json") == {
        "resource": "missing",
        "available": False,
        "detail": "missing: missing.json",
    }
    assert exp4171._ci_excludes_zero_positive({}) is False
    assert (
        exp4171.classify_diffusiongemma_gate(
            {
                "graft_deferred": False,
                "verifier_value_added": True,
                "rerank_lift_vs_vote": {"delta": 0.1, "ci95": [0.01, 0.2]},
            }
        )["state"]
        == "unlocked_by_training_time_value_added"
    )
    assert (
        exp4171.classify_diffusiongemma_gate(
            {
                "graft_deferred": False,
                "verifier_value_added": False,
                "rerank_lift_vs_vote": {"delta": 0.1, "ci95": [0.01, 0.2]},
            }
        )["state"]
        == "unlocked_by_rerank_discrimination"
    )
    assert exp4171._training_role_status({"verifier_value_added": True}) == "value_added_diffusiongemma_unlocked"
    assert exp4171._training_role_status({"graft_deferred": False}) == "honest_null_no_transferable_value_added"


def test_scenario_4171_run_hygiene_writes_artifact_and_ledgers(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4171: run writes the deliverable JSON and stable ledger entries.
    _write_minimal_repo(tmp_path)
    artifact = exp4171.run_hygiene(tmp_path)
    exp4171.validate_artifact(artifact)

    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4171.SUDOKU_BASELINE_GAP_ID,
        exp4171.SUDOKU_GRAFT_GAP_ID,
    ]
    assert artifact["registry_updated"] is True
    written = json.loads((tmp_path / exp4171.EXP4171_ARTIFACT_PATH).read_text(encoding="utf-8"))
    assert written == artifact
    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text(encoding="utf-8"))
    assert exp4171._registry_contains_outcomes(registry) is True
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert exp4171.SUDOKU_BASELINE_GAP_ID in gaps
    assert exp4171.SUDOKU_GRAFT_GAP_ID in gaps
