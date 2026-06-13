"""Tests for Exp 4153 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4153, SCENARIO-VERIFY-4153.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import verifier_registry_gaps_hygiene_4153 as exp4153


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
                        "role_id": "sudoku_executable_verifier_training_time_4139",
                        "experiment": "results/experiment_4139_decisive_verifier_graft_sudoku.json",
                        "role": "candidate_trm_training_time_reward_signal_executable_domain",
                        "status": "graft_deferred_no_headroom",
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
        "experiment_4149_sudoku_accumulate_pass4_convergence.json",
        "experiment_4150_decisive_verifier_graft_sudoku.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def test_req_4153_spec_declared() -> None:
    # REQ-VERIFY-4153: OpenSpec declares runner, inputs, fields, and principles.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4153",
        "SCENARIO-VERIFY-4153",
        "python/carnot/experiment_4153_verifier_registry_gaps_hygiene.py",
        "experiment_4153_verifier_registry_gaps_hygiene.json",
        "0.4516",
        "0.5806",
        "experiment_4149_sudoku_accumulate_pass4_convergence.json",
        "experiment_4150_decisive_verifier_graft_sudoku.json",
        "blocked_pass3_noop_unresolved",
        "graft_deferred=true",
        "verifier_value_added=false",
        "DiffusionGemma",
        "diffusiongemma_gate_state=kept_gated",
    ):
        assert marker in spec
    assert exp4153.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4153.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4153.FIELD_PRINCIPLES["gaps_updated"] in spec
    assert exp4153.FIELD_PRINCIPLES["diffusiongemma_gate_state"] in spec


def test_scenario_4153_preconditions_and_replay_are_bitexact() -> None:
    # SCENARIO-VERIFY-4153: resources parse before cached ARC-1 replay runs.
    preflight = exp4153.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None
    assert {check["resource"] for check in preflight["checks"]} == {
        "gap4_arc1_candidate_fixtures",
        "verifier_registry",
        "verifier_gaps",
    }

    replay = exp4153.replay_gap4_arc1(REPO_ROOT)
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


def test_req_4153_classifies_baseline_graft_and_diffusiongemma_gate() -> None:
    # REQ-VERIFY-4153: .384 records deferral instead of laundering value-added.
    baseline = exp4153.classify_sudoku_baseline(REPO_ROOT)
    assert baseline["gap_id"] == exp4153.SUDOKU_BASELINE_GAP_ID
    assert baseline["status"] == "open_baseline_blocked_pass3_noop_val_0.2782"
    assert baseline["baseline_status"] == "blocked_pass3_noop_unresolved"
    assert baseline["matches_published_087"] is False
    assert baseline["faithful_for_graft_085"] is False
    assert baseline["published_target_val_exact_accuracy"] == pytest.approx(0.87)
    assert baseline["final_val_exact_accuracy"] == pytest.approx(0.278172343969)
    assert baseline["raw_val_trajectory_v384_rounded"] == [0.2782, None, None, None, 0.2782]
    assert baseline["effective_val_trajectory_v384_rounded"] == [
        0.2782,
        0.2782,
        0.2782,
        0.2782,
        0.2782,
    ]
    assert baseline["native_trainer_launched"] is False

    graft = exp4153.classify_sudoku_decisive_graft(REPO_ROOT)
    assert graft["gap_id"] == exp4153.SUDOKU_GRAFT_GAP_ID
    assert graft["status"] == "open_graft_deferred_baseline_below_0.85"
    assert graft["graft_deferred"] is True
    assert graft["verifier_value_added"] is False
    assert graft["baseline_val_exact_accuracy"] == pytest.approx(0.278172343969)
    assert graft["baseline_faithful_085"] is False
    assert graft["candidate_source"] == "none_baseline_below_0.85"
    assert graft["n_candidate_pools"] == 0
    assert graft["rerank_lift_vs_vote"]["status"] == "deferred_baseline_below_0.85"
    assert graft["rft_vs_ablation_delta"]["status"] == "deferred_baseline_below_0.85"

    gate = exp4153.classify_diffusiongemma_gate(graft)
    assert gate == {
        "state": "kept_gated",
        "reason": "no_transferable_verifier_value_added",
        "verifier_value_added": False,
        "graft_deferred": True,
        "uses_executable_oracle_upper_bound": False,
        "basis": "rerank_lift_vs_vote_or_rft_vs_ablation_delta",
    }


def test_scenario_4153_ensure_ledgers_record_baseline_graft_role_and_gate() -> None:
    # SCENARIO-VERIFY-4153: registry and gaps carry the .384 baseline and graft truth.
    replay = exp4153.replay_gap4_arc1(REPO_ROOT)
    baseline = exp4153.classify_sudoku_baseline(REPO_ROOT)
    graft = exp4153.classify_sudoku_decisive_graft(REPO_ROOT)
    gate = exp4153.classify_diffusiongemma_gate(graft)

    registry, gaps, summary = exp4153.ensure_ledgers_record_outcomes(
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
            exp4153.SUDOKU_BASELINE_GAP_ID,
            exp4153.SUDOKU_GRAFT_GAP_ID,
        ],
        "sudoku_baseline_recorded": True,
        "sudoku_graft_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4153"] == exp4153.EXP4153_ARTIFACT_PATH
    assert gap4["eval"]["exp4153_regression_guard_passed"] is True
    role = next(
        role
        for role in gap4["training_time_roles"]
        if role["role_id"] == exp4153.SUDOKU_TRAINING_ROLE_ID
    )
    assert role["status"] == "graft_deferred_baseline_below_0.85"
    assert role["executable_sudoku_verifier_as_reward_status"] == (
        "deferred_baseline_below_0.85_no_transferable_value_added"
    )
    assert role["baseline_effective_val_trajectory_v384_rounded"] == [
        0.2782,
        0.2782,
        0.2782,
        0.2782,
        0.2782,
    ]
    assert role["graft_deferred"] is True
    assert role["verifier_value_added"] is False
    assert role["diffusiongemma_gate_state"]["state"] == "kept_gated"
    assert exp4153._registry_contains_outcomes(registry) is True
    assert exp4153._registry_contains_outcomes({}) is False

    assert exp4153.SUDOKU_BASELINE_GAP_ID in gaps
    assert "baseline_status=blocked_pass3_noop_unresolved" in gaps
    assert "effective_val_trajectory_v384=[0.2782, 0.2782, 0.2782, 0.2782, 0.2782]" in gaps
    assert "matches_published_087=false" in gaps
    assert exp4153.SUDOKU_GRAFT_GAP_ID in gaps
    assert "graft_deferred=true" in gaps
    assert "verifier_value_added=false" in gaps
    assert "rerank_lift_vs_vote_status=deferred_baseline_below_0.85" in gaps
    assert "rft_vs_ablation_delta_status=deferred_baseline_below_0.85" in gaps
    assert "diffusiongemma_gate_state=kept_gated" in gaps


def test_req_4153_build_artifact_validates_required_fields() -> None:
    # REQ-VERIFY-4153: terminal artifact exposes required schema fields and principles.
    replay = exp4153.replay_gap4_arc1(REPO_ROOT)
    baseline = exp4153.classify_sudoku_baseline(REPO_ROOT)
    graft = exp4153.classify_sudoku_decisive_graft(REPO_ROOT)
    gate = exp4153.classify_diffusiongemma_gate(graft)
    artifact = exp4153.build_artifact(
        offline_replay=replay,
        sudoku_baseline=baseline,
        sudoku_decisive_graft=graft,
        diffusiongemma_gate_state=gate,
        registry_updated=True,
        gaps_updated=[
            exp4153.SUDOKU_BASELINE_GAP_ID,
            exp4153.SUDOKU_GRAFT_GAP_ID,
        ],
        duration_s=0.012,
    )

    exp4153.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4153.SUDOKU_BASELINE_GAP_ID,
        exp4153.SUDOKU_GRAFT_GAP_ID,
    ]
    assert artifact["field_principles"] == exp4153.FIELD_PRINCIPLES
    assert artifact["sudoku_baseline"]["status"] == "open_baseline_blocked_pass3_noop_val_0.2782"
    assert artifact["sudoku_decisive_graft"]["graft_deferred"] is True
    assert artifact["sudoku_decisive_graft"]["verifier_value_added"] is False
    assert artifact["diffusiongemma_gate_state"]["state"] == "kept_gated"

    for field in exp4153.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4153.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4153.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4153.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4153.validate_artifact({**artifact, "registry_updated": "yes"})
    with pytest.raises(ValueError, match="list"):
        exp4153.validate_artifact({**artifact, "gaps_updated": "GAP"})
    with pytest.raises(ValueError, match="state"):
        exp4153.validate_artifact({**artifact, "diffusiongemma_gate_state": {}})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4153.validate_artifact({**artifact, "inference_substrate": "live"})
    with pytest.raises(ValueError, match="field_principles"):
        exp4153.validate_artifact({**artifact, "field_principles": {}})


def test_scenario_4153_run_hygiene_writes_artifact_and_ledgers(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4153: run writes the deliverable JSON and stable ledger entries.
    _write_minimal_repo(tmp_path)
    artifact = exp4153.run_hygiene(tmp_path)
    exp4153.validate_artifact(artifact)

    out_path = tmp_path / exp4153.EXP4153_ARTIFACT_PATH
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["regression_guard_passed"] is True
    assert written["sudoku_baseline"]["baseline_status"] == "blocked_pass3_noop_unresolved"
    assert written["sudoku_decisive_graft"]["graft_deferred"] is True
    assert written["sudoku_decisive_graft"]["verifier_value_added"] is False
    assert written["diffusiongemma_gate_state"]["state"] == "kept_gated"

    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text())
    role_ids = [role["role_id"] for role in registry["verifiers"][0]["training_time_roles"]]
    assert exp4153.SUDOKU_TRAINING_ROLE_ID in role_ids
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert "Exp 4153 .384 Sudoku baseline trajectory status" in gaps
    assert "Exp 4153 .384 Sudoku decisive executable-verifier graft status" in gaps


def test_req_4153_blocked_precondition_stops_without_fabricating(tmp_path: Path) -> None:
    # REQ-VERIFY-4153: failed preconditions write blocked_<resource> and no ledger win.
    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")

    preflight = exp4153.check_preconditions(tmp_path)
    assert preflight["ok"] is False
    assert preflight["blocked_resource"] == "gap4_arc1_candidate_fixtures"

    artifact = exp4153.run_hygiene(tmp_path)
    exp4153.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_gap4_arc1_candidate_fixtures"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_updated"] is False
    assert artifact["gaps_updated"] == []
    assert artifact["diffusiongemma_gate_state"]["state"] == "blocked"
    assert "Exp 4153" not in (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")


def test_req_4153_defensive_branches_cover_malformed_and_alternate_inputs(tmp_path: Path) -> None:
    # REQ-VERIFY-4153: alternate inputs do not fabricate a DiffusionGemma unlock.
    assert exp4153._numeric_or_none("0.1") is None
    assert exp4153._numeric_or_none(True) is None
    assert exp4153._round4(None) is None
    assert exp4153._rel_or_str(tmp_path, str(tmp_path / "results" / "x.json")) == "results/x.json"
    assert exp4153._rel_or_str(tmp_path, "/outside/x.json") == "/outside/x.json"
    assert exp4153._trajectory_rows({"val_trajectory_v384": "bad"}) == []
    assert exp4153._trajectory_rows(
        {
            "val_trajectory_v384": [
                "bad",
                {
                    "pass_label": "x",
                    "val_exact_accuracy": 0.1,
                    "effective_val_exact_accuracy": 0.2,
                },
            ]
        }
    )[0]["effective_val_exact_accuracy_rounded"] == 0.2

    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    baseline_path = results / Path(exp4153.EXP4149_PATH).name
    baseline_path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: reproduced-probe",
                "matches_published_087": True,
                "val_exact_accuracy": 0.88,
                "val_trajectory_v384": [{"val_exact_accuracy": 0.88}],
            }
        ),
        encoding="utf-8",
    )
    assert exp4153.classify_sudoku_baseline(tmp_path)["status"] == "reproduced_val_0.8800"
    baseline_path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: open-probe",
                "matches_published_087": False,
                "val_exact_accuracy": 0.5,
                "val_trajectory_v384": [],
            }
        ),
        encoding="utf-8",
    )
    assert (
        exp4153.classify_sudoku_baseline(tmp_path)["status"]
        == "open_baseline_not_reproduced_val_0.5000"
    )
    baseline_path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: missing-final-probe",
                "matches_published_087": False,
                "val_trajectory_v384": [{"effective_val_exact_accuracy": 0.4}],
            }
        ),
        encoding="utf-8",
    )
    assert (
        exp4153.classify_sudoku_baseline(tmp_path)["final_val_exact_accuracy"]
        == pytest.approx(0.4)
    )

    graft_path = results / Path(exp4153.EXP4150_PATH).name
    graft_path.write_text(
        json.dumps(
            {
                "honest_verdict": "success: value-added-probe",
                "verifier_value_added": True,
                "graft_deferred": False,
            }
        ),
        encoding="utf-8",
    )
    value_graft = exp4153.classify_sudoku_decisive_graft(tmp_path)
    assert value_graft["status"] == "filled_transferable_verifier_value_added"
    assert exp4153.classify_diffusiongemma_gate(value_graft)["state"] == "unlocked"

    graft_path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: null-probe",
                "graft_deferred": False,
                "verifier_value_added": False,
            }
        ),
        encoding="utf-8",
    )
    assert (
        exp4153.classify_sudoku_decisive_graft(tmp_path)["status"]
        == "open_honest_null_no_transferable_value_added"
    )

    repaired_registry: dict[str, Any] = {}
    exp4153._ensure_gap4_eval(
        repaired_registry,
        {"regression_guard_passed": True, "arc1_rule_exec": {"vote_pass2": 0.4516}},
    )
    assert repaired_registry["verifiers"][0]["verifier_id"] == exp4153.GAP4_VERIFIER_ID
    exp4153._ensure_sudoku_training_role({}, {}, {}, {})

    value_registry = _minimal_registry()
    exp4153._ensure_sudoku_training_role(
        value_registry,
        {"effective_val_trajectory_v384_rounded": [0.9]},
        {"verifier_value_added": True},
        {"state": "unlocked"},
    )
    assert value_registry["verifiers"][0]["training_time_roles"][-1]["status"] == (
        "value_added_diffusiongemma_unlocked"
    )

    null_registry = _minimal_registry()
    exp4153._ensure_sudoku_training_role(
        null_registry,
        {"effective_val_trajectory_v384_rounded": [0.5]},
        {"verifier_value_added": False, "graft_deferred": False},
        {"state": "kept_gated"},
    )
    assert null_registry["verifiers"][0]["training_time_roles"][-1]["status"] == (
        "honest_null_no_transferable_value_added"
    )
