"""Tests for Exp 4163 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4163, SCENARIO-VERIFY-4163.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot import experiment_4163_verifier_registry_gaps_hygiene as exp4163_wrapper
from carnot.reporting import verifier_registry_gaps_hygiene_4163 as exp4163


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
                        "role_id": "sudoku_executable_verifier_training_time_4150",
                        "experiment": "results/experiment_4150_decisive_verifier_graft_sudoku.json",
                        "status": "graft_deferred_baseline_below_0.85",
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
        "experiment_4157_baseline_harvest_contiguous_continue.json",
        "experiment_4158_verifier_rerank_recovery_moat.json",
        "experiment_4159_decisive_verifier_reward_graft.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def test_req_4163_spec_declared() -> None:
    # REQ-VERIFY-4163: OpenSpec declares runner, inputs, outputs, and principles.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4163",
        "SCENARIO-VERIFY-4163",
        "python/carnot/experiment_4163_verifier_registry_gaps_hygiene.py",
        "experiment_4163_verifier_registry_gaps_hygiene.json",
        "experiment_4157_baseline_harvest_contiguous_continue.json",
        "experiment_4158_verifier_rerank_recovery_moat.json",
        "experiment_4159_decisive_verifier_reward_graft.json",
        "0.4516",
        "0.5806",
        "current_val=0.5010",
        "headroom_present=false",
        "rerank_lift_vs_vote.delta=0.0",
        "verifier_recovers_outvoted=0",
        "graft_deferred=true",
        "verifier_value_added=false",
        "DiffusionGemma",
        "diffusiongemma_gate_state=kept_gated",
        "role_sudoku_executable",
    ):
        assert marker in spec
    assert exp4163.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4163.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4163.FIELD_PRINCIPLES["gaps_updated"] in spec
    assert exp4163_wrapper.main is exp4163.main


def test_scenario_4163_preconditions_and_replay_are_bitexact() -> None:
    # SCENARIO-VERIFY-4163: resources parse before cached ARC-1 replay runs.
    preflight = exp4163.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None
    assert {check["resource"] for check in preflight["checks"]} == {
        "gap4_arc1_candidate_fixtures",
        "verifier_registry",
        "verifier_gaps",
        "exp4157_baseline",
        "exp4158_rerank_moat",
        "exp4159_decisive_graft",
    }

    replay = exp4163.replay_gap4_arc1(REPO_ROOT)
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


def test_req_4163_classifies_baseline_rerank_graft_and_gate() -> None:
    # REQ-VERIFY-4163: .385 outcomes are recorded with caveats, not promoted.
    baseline = exp4163.classify_sudoku_baseline(REPO_ROOT)
    assert baseline["gap_id"] == exp4163.SUDOKU_BASELINE_GAP_ID
    assert baseline["status"] == "open_baseline_blocked_noop_step_unchanged_val_0.5010_flagged"
    assert baseline["baseline_faithful"] is False
    assert baseline["current_val"] == pytest.approx(0.501041650772)
    assert baseline["max_val"] == pytest.approx(0.501041650772)
    assert baseline["current_val_rounded"] == pytest.approx(0.501)
    assert baseline["native_trainer_launched"] is True
    assert baseline["flagged_adversarial"] is True
    assert baseline["val_trajectory_385_rounded"][:3] == [0.2005, 0.1992, 0.1995]
    assert baseline["val_trajectory_385_rounded"][-1] == pytest.approx(0.501)
    assert len(baseline["val_trajectory_385_rounded"]) == 27

    rerank = exp4163.classify_sudoku_rerank_moat(REPO_ROOT)
    assert rerank["gap_id"] == exp4163.SUDOKU_RERANK_GAP_ID
    assert rerank["status"] == "open_rerank_uninformative_no_headroom_flagged"
    assert rerank["headroom_present"] is False
    assert rerank["ci_excludes_zero_positive"] is False
    assert rerank["verifier_recovers_outvoted"] == 0
    assert rerank["vote_at_1"] == pytest.approx(0.140625)
    assert rerank["oracle_at_k"] == pytest.approx(0.140625)
    assert rerank["rerank_lift_vs_vote"]["delta"] == pytest.approx(0.0)
    assert rerank["rerank_lift_vs_vote"]["ci95"] == [0.0, 0.0]
    assert rerank["rerank_lift_vs_vote"]["per_puzzle_count"] == 64
    assert "per_puzzle" not in rerank["rerank_lift_vs_vote"]

    graft = exp4163.classify_sudoku_decisive_graft(REPO_ROOT)
    assert graft["gap_id"] == exp4163.SUDOKU_GRAFT_GAP_ID
    assert graft["status"] == "open_graft_deferred_baseline_below_0.85_flagged"
    assert graft["graft_deferred"] is True
    assert graft["verifier_value_added"] is False
    assert graft["current_val"] == pytest.approx(0.501041650772)
    assert graft["rft_vs_ablation_delta"]["status"] == "deferred_baseline_below_0.85"

    gate = exp4163.classify_diffusiongemma_gate(rerank, graft)
    assert gate == {
        "state": "kept_gated",
        "reason": "no_positive_rerank_signal_and_no_training_time_value_added",
        "moved_by_rerank_signal": False,
        "rerank_ci_excludes_zero_positive": False,
        "verifier_value_added": False,
        "graft_deferred": True,
        "uses_executable_oracle_upper_bound": False,
        "basis": "exp4158_rerank_lift_vs_vote_or_exp4159_verifier_value_added",
    }


def test_scenario_4163_ensure_ledgers_record_baseline_rerank_graft_and_roles() -> None:
    # SCENARIO-VERIFY-4163: registry and gaps carry the .385 truth.
    replay = exp4163.replay_gap4_arc1(REPO_ROOT)
    baseline = exp4163.classify_sudoku_baseline(REPO_ROOT)
    rerank = exp4163.classify_sudoku_rerank_moat(REPO_ROOT)
    graft = exp4163.classify_sudoku_decisive_graft(REPO_ROOT)
    gate = exp4163.classify_diffusiongemma_gate(rerank, graft)

    registry, gaps, summary = exp4163.ensure_ledgers_record_outcomes(
        _minimal_registry(),
        "# Verifier Gaps\n",
        replay,
        baseline,
        rerank,
        graft,
        gate,
    )

    assert summary == {
        "registry_updated": True,
        "gaps_updated": [
            exp4163.SUDOKU_BASELINE_GAP_ID,
            exp4163.SUDOKU_RERANK_GAP_ID,
            exp4163.SUDOKU_GRAFT_GAP_ID,
        ],
        "sudoku_baseline_recorded": True,
        "sudoku_rerank_recorded": True,
        "sudoku_graft_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4163"] == exp4163.EXP4163_ARTIFACT_PATH
    assert gap4["eval"]["exp4163_regression_guard_passed"] is True
    assert gap4["role_sudoku_executable"]["status"] == "uninformative_no_headroom_flagged"
    assert gap4["role_sudoku_executable"]["promoted_toward_candidate"] is False

    training_role = next(
        role
        for role in gap4["training_time_roles"]
        if role["role_id"] == exp4163.SUDOKU_TRAINING_ROLE_ID
    )
    assert training_role["status"] == "graft_deferred_baseline_below_0.85_flagged"
    assert training_role["verifier_value_added"] is False

    rerank_role = next(
        role
        for role in gap4["rerank_time_roles"]
        if role["role_id"] == exp4163.SUDOKU_RERANK_ROLE_ID
    )
    assert rerank_role["status"] == "uninformative_no_headroom_flagged"
    assert rerank_role["headroom_present"] is False
    assert rerank_role["verifier_recovers_outvoted"] == 0
    assert rerank_role["diffusiongemma_gate_state"]["state"] == "kept_gated"
    assert exp4163._registry_contains_outcomes(registry) is True
    assert exp4163._registry_contains_outcomes({}) is False

    assert exp4163.SUDOKU_BASELINE_GAP_ID in gaps
    assert "current_val=0.501" in gaps
    assert "baseline_faithful=false" in gaps
    assert "flagged_adversarial=true" in gaps
    assert exp4163.SUDOKU_RERANK_GAP_ID in gaps
    assert "headroom_present=false" in gaps
    assert "verifier_recovers_outvoted=0" in gaps
    assert "rerank_lift_vs_vote_delta=0.0" in gaps
    assert exp4163.SUDOKU_GRAFT_GAP_ID in gaps
    assert "graft_deferred=true" in gaps
    assert "verifier_value_added=false" in gaps
    assert "diffusiongemma_gate_state=kept_gated" in gaps


def test_req_4163_build_artifact_validates_required_fields() -> None:
    # REQ-VERIFY-4163: terminal artifact exposes required schema fields and principles.
    replay = exp4163.replay_gap4_arc1(REPO_ROOT)
    baseline = exp4163.classify_sudoku_baseline(REPO_ROOT)
    rerank = exp4163.classify_sudoku_rerank_moat(REPO_ROOT)
    graft = exp4163.classify_sudoku_decisive_graft(REPO_ROOT)
    gate = exp4163.classify_diffusiongemma_gate(rerank, graft)
    artifact = exp4163.build_artifact(
        offline_replay=replay,
        sudoku_baseline=baseline,
        sudoku_rerank_moat=rerank,
        sudoku_decisive_graft=graft,
        diffusiongemma_gate_state=gate,
        registry_updated=True,
        gaps_updated=[
            exp4163.SUDOKU_BASELINE_GAP_ID,
            exp4163.SUDOKU_RERANK_GAP_ID,
            exp4163.SUDOKU_GRAFT_GAP_ID,
        ],
        duration_s=0.012,
    )

    exp4163.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4163.SUDOKU_BASELINE_GAP_ID,
        exp4163.SUDOKU_RERANK_GAP_ID,
        exp4163.SUDOKU_GRAFT_GAP_ID,
    ]
    assert artifact["field_principles"] == exp4163.FIELD_PRINCIPLES
    assert artifact["sudoku_baseline"]["current_val_rounded"] == pytest.approx(0.501)
    assert artifact["sudoku_rerank_moat"]["headroom_present"] is False
    assert artifact["sudoku_decisive_graft"]["graft_deferred"] is True
    assert artifact["diffusiongemma_gate_state"]["state"] == "kept_gated"

    for field in exp4163.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4163.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4163.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4163.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4163.validate_artifact({**artifact, "registry_updated": "yes"})
    with pytest.raises(ValueError, match="list"):
        exp4163.validate_artifact({**artifact, "gaps_updated": "GAP"})
    with pytest.raises(ValueError, match="state"):
        exp4163.validate_artifact({**artifact, "diffusiongemma_gate_state": {}})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4163.validate_artifact({**artifact, "inference_substrate": "live"})
    with pytest.raises(ValueError, match="field_principles"):
        exp4163.validate_artifact({**artifact, "field_principles": {}})


def test_scenario_4163_run_hygiene_writes_artifact_and_ledgers(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4163: run writes the deliverable JSON and stable ledger entries.
    _write_minimal_repo(tmp_path)
    artifact = exp4163.run_hygiene(tmp_path)
    exp4163.validate_artifact(artifact)

    out_path = tmp_path / exp4163.EXP4163_ARTIFACT_PATH
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["regression_guard_passed"] is True
    assert written["sudoku_baseline"]["current_val_rounded"] == pytest.approx(0.501)
    assert written["sudoku_rerank_moat"]["headroom_present"] is False
    assert written["sudoku_decisive_graft"]["graft_deferred"] is True
    assert written["diffusiongemma_gate_state"]["state"] == "kept_gated"

    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text())
    gap4 = registry["verifiers"][0]
    assert gap4["role_sudoku_executable"]["status"] == "uninformative_no_headroom_flagged"
    assert exp4163.SUDOKU_TRAINING_ROLE_ID in [
        role["role_id"] for role in gap4["training_time_roles"]
    ]
    assert exp4163.SUDOKU_RERANK_ROLE_ID in [
        role["role_id"] for role in gap4["rerank_time_roles"]
    ]
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert "Exp 4163 .385 Sudoku baseline trajectory status" in gaps
    assert "Exp 4163 .385 Sudoku executable-verifier rerank moat status" in gaps
    assert "Exp 4163 .385 Sudoku decisive executable-verifier graft status" in gaps


def test_req_4163_blocked_precondition_stops_without_fabricating(tmp_path: Path) -> None:
    # REQ-VERIFY-4163: failed preconditions write blocked_<resource> and no ledger win.
    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")

    preflight = exp4163.check_preconditions(tmp_path)
    assert preflight["ok"] is False
    assert preflight["blocked_resource"] == "gap4_arc1_candidate_fixtures"

    artifact = exp4163.run_hygiene(tmp_path)
    exp4163.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_gap4_arc1_candidate_fixtures"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_updated"] is False
    assert artifact["gaps_updated"] == []
    assert artifact["diffusiongemma_gate_state"]["state"] == "blocked"
    assert "Exp 4163" not in (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")


def test_req_4163_defensive_and_candidate_status_branches(tmp_path: Path) -> None:
    # REQ-VERIFY-4163: helper branches keep promotion gated by a real positive CI.
    assert exp4163._numeric_or_none("0.1") is None
    assert exp4163._numeric_or_none(True) is None
    assert exp4163._round4(None) is None
    assert exp4163._trajectory_rows({"val_trajectory": "bad"}) == []
    assert exp4163._trajectory_rows({"val_trajectory": ["bad"]}) == []
    assert exp4163._compact_metric("bad") == {}
    assert exp4163._compact_metric({"per_puzzle": [1, 2], "delta": 0.25}) == {
        "per_puzzle_count": 2,
        "delta": 0.25,
    }
    assert exp4163._ci_excludes_zero_positive({"delta": 0.1}) is False
    assert exp4163._ci_excludes_zero_positive({"ci95": [0.01, 0.2], "delta": 0.1}) is True
    assert exp4163._ci_excludes_zero_positive({"ci95": [-0.1, 0.2], "delta": 0.1}) is False
    assert exp4163._rerank_role_status(
        {"headroom_present": True, "ci_excludes_zero_positive": False, "flagged_adversarial": False}
    ) == "honest_null_ci_includes_zero"
    assert exp4163._training_role_status(
        {"graft_deferred": False, "verifier_value_added": False, "flagged_adversarial": False}
    ) == "honest_null_no_transferable_value_added"

    positive_rerank = {
        "ci_excludes_zero_positive": True,
        "headroom_present": True,
        "flagged_adversarial": False,
        "rerank_lift_vs_vote": {"delta": 0.05},
        "status": "filled_rerank_recovery_moat",
    }
    deferred_graft = {"verifier_value_added": False, "graft_deferred": True}
    assert exp4163.classify_diffusiongemma_gate(positive_rerank, deferred_graft)["state"] == (
        "unlocked_by_rerank_discrimination"
    )
    value_graft = {"verifier_value_added": True, "graft_deferred": False}
    assert exp4163.classify_diffusiongemma_gate({}, value_graft)["state"] == (
        "unlocked_by_training_time_value_added"
    )

    registry = {"verifiers": [{"verifier_id": exp4163.GAP4_VERIFIER_ID, "eval": {}}]}
    exp4163._ensure_sudoku_roles(
        registry,
        {"status": "reproduced", "current_val": 0.9, "val_trajectory_385_rounded": [0.9]},
        positive_rerank,
        value_graft,
        {"state": "unlocked_by_training_time_value_added"},
    )
    gap4 = registry["verifiers"][0]
    assert gap4["role_sudoku_executable"]["status"] == "candidate_rerank_signal"
    assert gap4["training_time_roles"][0]["status"] == "value_added_diffusiongemma_unlocked"
    assert gap4["rerank_time_roles"][0]["status"] == "candidate_rerank_signal"

    training_candidate_registry = {
        "verifiers": [{"verifier_id": exp4163.GAP4_VERIFIER_ID, "eval": {}}]
    }
    exp4163._ensure_sudoku_roles(
        training_candidate_registry,
        {},
        {"headroom_present": True, "ci_excludes_zero_positive": False},
        {"verifier_value_added": True},
        {"state": "unlocked_by_training_time_value_added"},
    )
    assert training_candidate_registry["verifiers"][0]["role_sudoku_executable"]["status"] == (
        "candidate_training_time_value_added"
    )

    exp4163._ensure_sudoku_roles({}, {}, {}, {}, {})
    exp4163._ensure_gap4_eval(
        {},
        {"regression_guard_passed": True, "arc1_rule_exec": {"vote_pass2": 0.4516}},
    )

    (tmp_path / "results").mkdir()
    malformed_path = tmp_path / "results" / "not_object.json"
    malformed_path.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp4163._check_json_resource(
        tmp_path, "not_object", "results/not_object.json"
    ) == {
        "resource": "not_object",
        "available": False,
        "detail": "not_json_object",
    }
    (tmp_path / "results" / Path(exp4163.EXP4157_PATH).name).write_text(
        json.dumps(
            {
                "honest_verdict": "complete: reproduced-probe",
                "baseline_faithful": True,
                "current_val": 0.86,
                "max_val": 0.88,
                "val_trajectory": [{"val_exact_accuracy": 0.86}],
            }
        ),
        encoding="utf-8",
    )
    assert exp4163.classify_sudoku_baseline(tmp_path)["status"] == "reproduced_val_0.8600"

    (tmp_path / "results" / Path(exp4163.EXP4157_PATH).name).write_text(
        json.dumps(
            {
                "honest_verdict": "complete: fallback-probe",
                "baseline_faithful": False,
                "val_trajectory": [
                    {"val_exact_accuracy": 0.4},
                    {"val_exact_accuracy": 0.5},
                ],
            }
        ),
        encoding="utf-8",
    )
    fallback_baseline = exp4163.classify_sudoku_baseline(tmp_path)
    assert fallback_baseline["status"] == "open_baseline_not_faithful_val_0.5000"
    assert fallback_baseline["max_val"] == pytest.approx(0.5)

    (tmp_path / "results" / Path(exp4163.EXP4158_PATH).name).write_text(
        json.dumps(
            {
                "honest_verdict": "complete: positive-probe",
                "headroom_present": True,
                "flagged_adversarial": False,
                "rerank_lift_vs_vote": {"delta": 0.1, "ci95": [0.01, 0.2]},
            }
        ),
        encoding="utf-8",
    )
    assert exp4163.classify_sudoku_rerank_moat(tmp_path)["status"] == (
        "filled_rerank_recovery_moat"
    )

    (tmp_path / "results" / Path(exp4163.EXP4158_PATH).name).write_text(
        json.dumps(
            {
                "honest_verdict": "complete: null-probe",
                "headroom_present": True,
                "flagged_adversarial": False,
                "rerank_lift_vs_vote": {"delta": 0.0, "ci95": [-0.1, 0.1]},
            }
        ),
        encoding="utf-8",
    )
    assert exp4163.classify_sudoku_rerank_moat(tmp_path)["status"] == (
        "open_honest_null_ci_includes_zero"
    )

    (tmp_path / "results" / Path(exp4163.EXP4159_PATH).name).write_text(
        json.dumps(
            {
                "honest_verdict": "success: value-probe",
                "graft_deferred": False,
                "verifier_value_added": True,
                "flagged_adversarial": False,
            }
        ),
        encoding="utf-8",
    )
    assert exp4163.classify_sudoku_decisive_graft(tmp_path)["status"] == (
        "filled_training_time_verifier_value_added"
    )

    (tmp_path / "results" / Path(exp4163.EXP4159_PATH).name).write_text(
        json.dumps(
            {
                "honest_verdict": "complete: null-probe",
                "graft_deferred": False,
                "verifier_value_added": False,
                "flagged_adversarial": False,
            }
        ),
        encoding="utf-8",
    )
    assert exp4163.classify_sudoku_decisive_graft(tmp_path)["status"] == (
        "open_honest_null_no_transferable_value_added"
    )
