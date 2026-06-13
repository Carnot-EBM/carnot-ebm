"""Tests for Exp 4142 verifier registry/gaps hygiene.

Spec refs: REQ-VERIFY-4142, SCENARIO-VERIFY-4142.
"""

from __future__ import annotations

import json
import gzip
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from carnot.reporting import verifier_registry_gaps_hygiene_4142 as exp4142


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
                        "role_id": "sudoku_executable_verifier_training_time_4128",
                        "experiment": "results/experiment_4128_carnot_verifier_graft_sudoku.json",
                        "role": "candidate_trm_training_time_reward_signal_executable_domain",
                        "status": "graft_deferred_baseline_not_reproduced",
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
        "experiment_4138_sudoku_accumulate_pass4_convergence_check.json",
        "experiment_4139_decisive_verifier_graft_sudoku.json",
    ):
        shutil.copy2(REPO_ROOT / "results" / name, tmp_path / "results" / name)


def test_req_4142_spec_declared() -> None:
    # REQ-VERIFY-4142: OpenSpec declares the runner, inputs, fields, and principles.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4142",
        "SCENARIO-VERIFY-4142",
        "python/carnot/experiment_4142_verifier_registry_gaps_hygiene.py",
        "experiment_4142_verifier_registry_gaps_hygiene.json",
        "0.4516",
        "0.5806",
        "baseline_status=config-blocked",
        "headroom_present=false",
        "executable_oracle_upper_bound",
        "ensemble_rerank_lift_vs_vote",
        "rft_vs_ablation_delta",
        "DiffusionGemma",
        "diffusiongemma_gate_state=kept_gated",
    ):
        assert marker in spec
    assert exp4142.FIELD_PRINCIPLES["honest_verdict"] in spec
    assert exp4142.FIELD_PRINCIPLES["regression_guard_passed"] in spec
    assert exp4142.FIELD_PRINCIPLES["gaps_updated"] in spec
    assert exp4142.FIELD_PRINCIPLES["diffusiongemma_gate_state"] in spec


def test_scenario_4142_preconditions_and_replay_are_bitexact() -> None:
    # SCENARIO-VERIFY-4142: resources parse before cached ARC-1 replay runs.
    preflight = exp4142.check_preconditions(REPO_ROOT)
    assert preflight["ok"] is True
    assert preflight["blocked_resource"] is None
    assert {check["resource"] for check in preflight["checks"]} == {
        "gap4_arc1_candidate_fixtures",
        "verifier_registry",
        "verifier_gaps",
    }

    replay = exp4142.replay_gap4_arc1(REPO_ROOT)
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


def test_req_4142_classifies_baseline_graft_and_diffusiongemma_gate() -> None:
    # REQ-VERIFY-4142: .383 distinguishes executable oracle from transferable value.
    baseline = exp4142.classify_sudoku_baseline(REPO_ROOT)
    assert baseline["gap_id"] == exp4142.SUDOKU_BASELINE_GAP_ID
    assert baseline["status"] == "open_baseline_config_blocked_val_0.2782"
    assert baseline["baseline_status"] == "config-blocked"
    assert baseline["matches_published_087"] is False
    assert baseline["near_faithful_080"] is False
    assert baseline["published_target_val_exact_accuracy"] == pytest.approx(0.87)
    assert baseline["final_val_exact_accuracy"] == pytest.approx(0.278172343969)
    assert baseline["val_trajectory_383_rounded"] == [0.2782, None, None, None, None]
    assert baseline["measured_val_trajectory_rounded"] == [0.2782]
    assert baseline["config_blocked"] is True

    graft = exp4142.classify_sudoku_decisive_graft(REPO_ROOT)
    assert graft["gap_id"] == exp4142.SUDOKU_GRAFT_GAP_ID
    assert graft["status"] == "open_graft_deferred_no_transferable_value_added"
    assert graft["headroom_present"] is False
    assert graft["oracle_vs_vote_gap"] == pytest.approx(0.0)
    assert graft["executable_verifier_is_oracle"] is True
    assert graft["executable_oracle_upper_bound"]["interpretation"] == "oracle_upper_bound_not_verifier_value"
    assert graft["ensemble_rerank_lift_vs_vote"]["status"] == "uninterpretable_no_headroom"
    assert graft["rft_vs_ablation_delta"]["status"] == "deferred_baseline_below_0.80"
    assert graft["graft_deferred"] is True
    assert graft["verifier_value_added"] is False

    gate = exp4142.classify_diffusiongemma_gate(graft)
    assert gate == {
        "state": "kept_gated",
        "reason": "no_transferable_verifier_value_added",
        "verifier_value_added": False,
        "headroom_present": False,
        "uses_executable_oracle_upper_bound": False,
        "basis": "ensemble_rerank_lift_vs_vote_or_rft_vs_ablation_delta_not_oracle",
    }


def test_scenario_4142_ensure_ledgers_record_baseline_graft_role_and_gate() -> None:
    # SCENARIO-VERIFY-4142: registry and gaps carry the .383 baseline and graft truth.
    replay = exp4142.replay_gap4_arc1(REPO_ROOT)
    baseline = exp4142.classify_sudoku_baseline(REPO_ROOT)
    graft = exp4142.classify_sudoku_decisive_graft(REPO_ROOT)
    gate = exp4142.classify_diffusiongemma_gate(graft)

    registry, gaps, summary = exp4142.ensure_ledgers_record_outcomes(
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
            exp4142.SUDOKU_BASELINE_GAP_ID,
            exp4142.SUDOKU_GRAFT_GAP_ID,
        ],
        "sudoku_baseline_recorded": True,
        "sudoku_graft_recorded": True,
    }
    gap4 = registry["verifiers"][0]
    assert gap4["eval"]["eval_exp_4142"] == exp4142.EXP4142_ARTIFACT_PATH
    assert gap4["eval"]["exp4142_regression_guard_passed"] is True
    role = next(
        role
        for role in gap4["training_time_roles"]
        if role["role_id"] == exp4142.SUDOKU_TRAINING_ROLE_ID
    )
    assert role["status"] == "graft_deferred_no_headroom"
    assert role["executable_sudoku_verifier_as_reward_status"] == (
        "deferred_oracle_upper_bound_only_no_transferable_value_added"
    )
    assert role["baseline_val_trajectory_383_rounded"] == [0.2782, None, None, None, None]
    assert role["headroom_present"] is False
    assert role["verifier_value_added"] is False
    assert role["diffusiongemma_gate_state"]["state"] == "kept_gated"
    assert exp4142._registry_contains_outcomes(registry) is True
    assert exp4142._registry_contains_outcomes({}) is False

    assert exp4142.SUDOKU_BASELINE_GAP_ID in gaps
    assert "baseline_status=config-blocked" in gaps
    assert "val_trajectory_383=[0.2782, None, None, None, None]" in gaps
    assert "matches_published_087=false" in gaps
    assert exp4142.SUDOKU_GRAFT_GAP_ID in gaps
    assert "headroom_present=false" in gaps
    assert "executable_oracle_upper_bound_delta=0.0" in gaps
    assert "ensemble_rerank_lift_vs_vote_delta=0.0" in gaps
    assert "rft_vs_ablation_delta_status=deferred_baseline_below_0.80" in gaps
    assert "verifier_value_added=false" in gaps
    assert "diffusiongemma_gate_state=kept_gated" in gaps


def test_req_4142_build_artifact_validates_required_fields() -> None:
    # REQ-VERIFY-4142: terminal artifact exposes required schema fields and principles.
    replay = exp4142.replay_gap4_arc1(REPO_ROOT)
    baseline = exp4142.classify_sudoku_baseline(REPO_ROOT)
    graft = exp4142.classify_sudoku_decisive_graft(REPO_ROOT)
    gate = exp4142.classify_diffusiongemma_gate(graft)
    artifact = exp4142.build_artifact(
        offline_replay=replay,
        sudoku_baseline=baseline,
        sudoku_decisive_graft=graft,
        diffusiongemma_gate_state=gate,
        registry_updated=True,
        gaps_updated=[
            exp4142.SUDOKU_BASELINE_GAP_ID,
            exp4142.SUDOKU_GRAFT_GAP_ID,
        ],
        duration_s=0.012,
    )

    exp4142.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["regression_guard_passed"] is True
    assert artifact["gaps_updated"] == [
        exp4142.SUDOKU_BASELINE_GAP_ID,
        exp4142.SUDOKU_GRAFT_GAP_ID,
    ]
    assert artifact["field_principles"] == exp4142.FIELD_PRINCIPLES
    assert artifact["sudoku_baseline"]["val_trajectory_383_rounded"] == [
        0.2782,
        None,
        None,
        None,
        None,
    ]
    assert artifact["sudoku_decisive_graft"]["headroom_present"] is False
    assert artifact["sudoku_decisive_graft"]["verifier_value_added"] is False
    assert artifact["diffusiongemma_gate_state"]["state"] == "kept_gated"

    for field in exp4142.REQUIRED_ARTIFACT_FIELDS:
        malformed = dict(artifact)
        malformed.pop(field)
        with pytest.raises(ValueError, match=field):
            exp4142.validate_artifact(malformed)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4142.validate_artifact({**artifact, "honest_verdict": "not terminal"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4142.validate_artifact({**artifact, "regression_guard_passed": "yes"})
    with pytest.raises(ValueError, match="bare bool"):
        exp4142.validate_artifact({**artifact, "registry_updated": "yes"})
    with pytest.raises(ValueError, match="list"):
        exp4142.validate_artifact({**artifact, "gaps_updated": "GAP"})
    with pytest.raises(ValueError, match="state"):
        exp4142.validate_artifact({**artifact, "diffusiongemma_gate_state": {}})
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4142.validate_artifact({**artifact, "inference_substrate": "live"})
    with pytest.raises(ValueError, match="field_principles"):
        exp4142.validate_artifact({**artifact, "field_principles": {}})


def test_scenario_4142_run_hygiene_writes_artifact_and_ledgers(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4142: run writes the deliverable JSON and stable ledger entries.
    _write_minimal_repo(tmp_path)
    artifact = exp4142.run_hygiene(tmp_path)
    exp4142.validate_artifact(artifact)

    out_path = tmp_path / exp4142.EXP4142_ARTIFACT_PATH
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["regression_guard_passed"] is True
    assert written["sudoku_baseline"]["status"] == "open_baseline_config_blocked_val_0.2782"
    assert written["sudoku_decisive_graft"]["headroom_present"] is False
    assert written["sudoku_decisive_graft"]["verifier_value_added"] is False
    assert written["diffusiongemma_gate_state"]["state"] == "kept_gated"

    registry = yaml.safe_load((tmp_path / "ops" / "verifier_registry.yaml").read_text())
    role_ids = [role["role_id"] for role in registry["verifiers"][0]["training_time_roles"]]
    assert exp4142.SUDOKU_TRAINING_ROLE_ID in role_ids
    gaps = (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")
    assert "Exp 4142 .383 Sudoku baseline trajectory status" in gaps
    assert "Exp 4142 .383 Sudoku decisive executable-verifier graft status" in gaps


def test_req_4142_blocked_precondition_stops_without_fabricating(tmp_path: Path) -> None:
    # REQ-VERIFY-4142: failed preconditions write blocked_<resource> and no ledger win.
    (tmp_path / "ops").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "ops" / "verifier_registry.yaml").write_text(
        yaml.safe_dump(_minimal_registry(), sort_keys=False),
        encoding="utf-8",
    )
    (tmp_path / "ops" / "verifier_gaps.md").write_text("# Verifier Gaps\n", encoding="utf-8")

    preflight = exp4142.check_preconditions(tmp_path)
    assert preflight["ok"] is False
    assert preflight["blocked_resource"] == "gap4_arc1_candidate_fixtures"

    artifact = exp4142.run_hygiene(tmp_path)
    exp4142.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_gap4_arc1_candidate_fixtures"
    assert artifact["regression_guard_passed"] is False
    assert artifact["registry_updated"] is False
    assert artifact["gaps_updated"] == []
    assert artifact["diffusiongemma_gate_state"]["state"] == "blocked"
    assert "Exp 4142" not in (tmp_path / "ops" / "verifier_gaps.md").read_text(encoding="utf-8")


def test_req_4142_defensive_branches_cover_malformed_and_alternate_inputs(tmp_path: Path) -> None:
    # REQ-VERIFY-4142: malformed optional inputs do not fabricate a DiffusionGemma unlock.
    assert exp4142._numeric_or_none("0.1") is None
    malformed_gzip = tmp_path / "not_object.json.gz"
    with gzip.open(malformed_gzip, "wt", encoding="utf-8") as handle:
        json.dump([], handle)
    with pytest.raises(ValueError, match="expected JSON object"):
        exp4142._load_gzip_json(malformed_gzip)

    not_mapping = tmp_path / "registry_list.yaml"
    not_mapping.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a mapping"):
        exp4142._load_registry_for_check(not_mapping)
    missing_verifiers = tmp_path / "registry_missing_verifiers.yaml"
    missing_verifiers.write_text("x: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing verifiers"):
        exp4142._load_registry_for_check(missing_verifiers)
    empty_gaps = tmp_path / "empty_gaps.md"
    empty_gaps.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        exp4142._load_gaps_for_check(empty_gaps)

    assert exp4142._trajectory_rows({"val_trajectory_383": "bad"}) == []
    assert exp4142._trajectory_rows(
        {"val_trajectory_383": ["bad", {"label": "x", "val_exact_accuracy": 0.1}]}
    )[0]["val_exact_accuracy_rounded"] == 0.1

    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    baseline_path = results / Path(exp4142.EXP4138_PATH).name
    baseline_path.write_text(
        json.dumps(
            {
                "honest_verdict": "success: reproduced-probe",
                "matches_published_087": True,
                "near_faithful_080": True,
                "val_exact_accuracy": 0.88,
                "val_trajectory_383": [{"val_exact_accuracy": 0.88}],
            }
        ),
        encoding="utf-8",
    )
    assert exp4142.classify_sudoku_baseline(tmp_path)["status"] == "reproduced_val_0.8800"
    baseline_path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: open-probe",
                "matches_published_087": False,
                "baseline_status": "running",
                "baseline": {"val_exact_accuracy": 0.5},
                "val_trajectory_383": [],
            }
        ),
        encoding="utf-8",
    )
    assert (
        exp4142.classify_sudoku_baseline(tmp_path)["status"]
        == "open_baseline_not_reproduced_val_0.5000"
    )

    graft_path = results / Path(exp4142.EXP4139_PATH).name
    graft_path.write_text(
        json.dumps(
            {
                "honest_verdict": "success: value-added-probe",
                "verifier_value_added": True,
                "headroom_present": True,
            }
        ),
        encoding="utf-8",
    )
    value_graft = exp4142.classify_sudoku_decisive_graft(tmp_path)
    assert value_graft["status"] == "filled_transferable_verifier_value_added"
    assert exp4142.classify_diffusiongemma_gate(value_graft)["state"] == "unlocked"

    graft_path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: no-headroom-probe",
                "graft_deferred": False,
                "verifier_value_added": False,
                "headroom_present": False,
            }
        ),
        encoding="utf-8",
    )
    assert (
        exp4142.classify_sudoku_decisive_graft(tmp_path)["status"]
        == "open_uninformative_no_headroom_no_transferable_value_added"
    )
    graft_path.write_text(
        json.dumps(
            {
                "honest_verdict": "complete: null-probe",
                "graft_deferred": False,
                "verifier_value_added": False,
                "headroom_present": True,
            }
        ),
        encoding="utf-8",
    )
    assert (
        exp4142.classify_sudoku_decisive_graft(tmp_path)["status"]
        == "open_honest_null_no_transferable_value_added"
    )

    repaired_registry: dict[str, Any] = {}
    exp4142._ensure_gap4_eval(
        repaired_registry,
        {"regression_guard_passed": True, "arc1_rule_exec": {"vote_pass2": 0.4516}},
    )
    assert repaired_registry["verifiers"][0]["verifier_id"] == exp4142.GAP4_VERIFIER_ID
    exp4142._ensure_sudoku_training_role({}, {}, {}, {})

    value_registry = _minimal_registry()
    exp4142._ensure_sudoku_training_role(
        value_registry,
        {"val_trajectory_383_rounded": [0.9]},
        {"verifier_value_added": True},
        {"state": "unlocked"},
    )
    assert value_registry["verifiers"][0]["training_time_roles"][-1]["status"] == (
        "value_added_diffusiongemma_unlocked"
    )

    null_registry = _minimal_registry()
    exp4142._ensure_sudoku_training_role(
        null_registry,
        {"val_trajectory_383_rounded": [0.5]},
        {"verifier_value_added": False, "graft_deferred": False},
        {"state": "kept_gated"},
    )
    assert null_registry["verifiers"][0]["training_time_roles"][-1]["status"] == (
        "honest_null_no_transferable_value_added"
    )
